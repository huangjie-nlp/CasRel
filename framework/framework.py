import torch
import torch.nn.functional as F
import json
from models.models import Casrel
from dataloader.dataloader import MyDataset, collate_fn
from torch.utils.data import DataLoader
from tqdm import tqdm
from Logger.logger import Logger
import numpy as np

class Framework():
    def __init__(self, config):
        self.config = config
        with open(self.config.schema, "r", encoding='utf-8') as f:
            self.id2label = json.load(f)[1]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log = Logger(self.config.log)

    def train(self):

        train_dataset = MyDataset(self.config, self.config.train_data)
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=self.config.batch_size, collate_fn=collate_fn, pin_memory=True)

        dev_dataset = MyDataset(self.config, self.config.dev_data)
        dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=1, collate_fn=collate_fn, pin_memory=True)

        def loss_fn(input, target, mask):
            """
            :param input: [batch_size, sequence_length] or [batch_size, sequence_length, rel_num]
            :param target: [batch_size, sequence_length] or [batch_size, sequence_length, rel_num]
            :return:
            """
            if mask.shape != input.shape:
                mask = mask.unsqueeze(dim=-1)
            loss_ = F.binary_cross_entropy(input, target, reduction='none')
            loss = torch.sum(loss_ * mask) / torch.sum(mask)
            return loss

        model = Casrel(self.config).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate)

        global_loss = 0
        global_step = 0
        best_f1 = 0
        precision = 0
        recall = 0
        best_epoch = 0

        for epoch in range(1, self.config.epoch + 1):
            for data in tqdm(train_dataloader):
                subject_heads, subject_tails, object_heads, object_tails = model(data)
                model.zero_grad()
                subject_loss = loss_fn(subject_heads, data["subject_head"].to(self.device), data["mask"].to(self.device)) + \
                               loss_fn(subject_tails, data["subject_tail"].to(self.device), data["mask"].to(self.device))
                object_loss = loss_fn(object_heads, data["object_heads"].to(self.device), data["mask"].to(self.device)) + \
                              loss_fn(object_tails, data["object_tails"].to(self.device), data["mask"].to(self.device))
                loss = (subject_loss + object_loss) / 2
                loss.backward()
                optimizer.step()
                global_loss += loss.item()
                if (global_step + 1) % self.config.step == 0:
                    self.log.logger.info("epoch: {} global_step: {:5.4f} global_loss: {:5.4f}".format(epoch, global_step, global_loss))
                    global_loss = 0
                global_step += 1

            if epoch % self.config.val_epoch == 0:
                self.log.logger.info("last save epoch: {}, precision: {:5.4f}, recall: {:5.4f}, f1:{:5.4f}".format(best_epoch, precision, recall, best_f1))
                p, r, f, predict = self.evaluate(model, dev_dataloader)
                if f > best_f1:
                    with open(self.config.dev_result, "w", encoding="utf-8") as fd:
                        json.dump(predict, fd, indent=4, ensure_ascii=False)
                    self.log.logger.info("save model......")
                    torch.save(model.state_dict(), self.config.save_model)
                    best_f1 = f
                    recall = r
                    precision = p
                    best_epoch = epoch
                    self.log.logger.info("epoch: {}, precision: {:5.4f} recall: {:5.4f} f1_score: {:5.4f}".format(epoch, precision, recall, best_f1))
        self.log.logger.info("epoch: {}, precision: {:5.4f} recall: {:5.4f} f1_score: {:5.4f}".format(best_epoch, precision, recall, best_f1))


    def evaluate(self, model, dataloader):

        def to_tuple(data):
            ret = []
            for i in data:
                ret.append(tuple(i))
            return tuple(ret)

        gold_num, correct_num, predict_num = 0, 0, 0
        predict = []

        model.eval()

        with torch.no_grad():
            for data in tqdm(dataloader):
                token = data["token"][0]
                triple_list = data["triple_list"][0]
                text = data["text"][0]
                input_ids, mask = data["input_ids"].to(self.device), data["mask"].to(self.device)
                # [1, sequence_length, bert_dim]
                text_encode = model.get_text_encode(input_ids, mask)
                subject_heads, subject_tails = model.get_subject(text_encode)

                subjects = []
                subj_heads, subj_tails = np.where(subject_heads.cpu()[0] > self.config.h_bar)[0], \
                                         np.where(subject_tails.cpu()[0] > self.config.t_bar)[0]
                for s_ids in subj_heads:
                    t_ids = np.array(subj_tails)[subj_tails >= s_ids]
                    if len(t_ids) > 0:
                        tail_idx = t_ids[0]
                        subjects.append(("".join(token[s_ids: tail_idx+1]), s_ids, tail_idx))
                predict_triple = []
                if len(subjects) > 0:
                    sequence_length = text_encode.size(1)
                    # [len(subjects), sequence_length, bert_dim]
                    text_encode = text_encode.repeat(len(subjects), 1, 1)
                    # [len(subjects), 1, sequence_length]
                    subject_heads_mapping = torch.LongTensor(len(subjects), 1, sequence_length).zero_()
                    subject_tails_mapping = torch.LongTensor(len(subjects), 1, sequence_length).zero_()
                    for k, s in enumerate(subjects):
                        subject_heads_mapping[k][0][s[1]] = 1
                        subject_tails_mapping[k][0][s[2]] = 1
                    subject_heads_mapping.to(text_encode)
                    subject_tails_mapping.to(text_encode)
                    object_heads, object_tails = model.get_special_relation_object(text_encode, subject_heads, subject_tails)

                    for sub_ids, sub in enumerate(subjects):
                        subject = sub[0]
                        object_heads_idx, object_tails_idx = np.where(object_heads.cpu()[sub_ids] > self.config.h_bar), \
                                                             np.where(object_tails.cpu()[sub_ids] > self.config.t_bar)
                        for obj_head, obj_head_rel in zip(*object_heads_idx):
                            for obj_tail, obj_tail_rel in zip(*object_tails_idx):
                                if obj_tail >= obj_head and obj_head_rel == obj_tail_rel:
                                    obj = ''.join(token[obj_head: obj_tail + 1])
                                    relation = self.id2label[str(obj_tail_rel)]
                                    predict_triple.append((subject, relation, obj))
                                    break
                triple_tuple = to_tuple(triple_list)
                predict_triple = to_tuple(predict_triple)
                new = set(predict_triple) - set(triple_tuple)
                lack = set(triple_tuple) - set(predict_triple)
                predict.append({"sentence": text, "gold": triple_tuple, "predict": predict_triple, "new": list(new), "lack": list(lack)})
                gold_num += len(triple_list)
                predict_num += len(predict_triple)
                correct_num += len(set(triple_tuple) & set(predict_triple))
        print("predict_num: {}, gold_num: {}, correct_num: {}".format(predict_num, gold_num, correct_num))
        precision = correct_num / (predict_num + 1e-10)
        recall = correct_num / (gold_num + 1e-10)
        f1_score = 2 * precision * recall / (recall + precision + 1e-10)
        model.train()
        return precision, recall, f1_score, predict

    def test(self, test_fn):
        model = Casrel(self.config)
        dataset = MyDataset(self.config, test_fn)
        dataloader = DataLoader(dataset, shuffle=True, batch_size=1, collate_fn=collate_fn, pin_memory=True)
        print("load model......")
        model.load_state_dict(torch.load(self.config.save_model))
        model.to(self.device)
        precision, recall, f1_score, predict= self.evaluate(model, dataloader)
        with open(self.config.test_result, "w", encoding="utf-8") as f:
            json.dump(predict, f, indent=4, ensure_ascii=False)
        print("test result!!!")
        print("precision:{:5.4f}, recall:{:5.4f}, f1_score:{:5.4f}".format(precision, recall, f1_score))


if __name__ == '__main__':
    s = [(1, 2), (2, 3)]
    b = []
    print(set(s) - set(b))
    print(set(b) - set(s))
    print(set([]) - set([]))


