import json
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import numpy as np
import random
random.seed(2022)


def find_idx(token, trip):
    trip_length = len(trip)
    for k, v in enumerate(token):
        if token[k: k+trip_length] == trip:
            return k
    return -1


class MyDataset(Dataset):
    def __init__(self, config, fn):
        self.config = config
        self.json_data = json.load(open(fn, "r", encoding="utf-8"))
        with open(self.config.schema, "r", encoding="utf-8") as fs:
            self.label2id = json.load(fs)[0]
        self.tokenizer = BertTokenizer.from_pretrained(self.config.bert_path)

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, idx):
        json_data = self.json_data[idx]
        text = json_data["text"]
        triple_list = json_data["triple_list"]

        token = ['[CLS]'] + list(text)[:510] + ['[SEP]']
        token_len = len(token)

        input_ids = self.tokenizer.convert_tokens_to_ids(token)
        mask = [1] * token_len

        input_ids = np.array(input_ids)
        mask = np.array(mask)

        if len(triple_list) > 0:
            sro = {}
            for trip in triple_list:
                triple = (list(trip[0]), trip[1], list(trip[2]))
                subject_head_ids = find_idx(token, triple[0])
                object_head_ids = find_idx(token, triple[2])
                if subject_head_ids != -1 and object_head_ids != -1:
                    subject_ids = (subject_head_ids, subject_head_ids + len(trip[0]) - 1)
                    object_ids = (object_head_ids, object_head_ids + len(trip[2]) - 1, self.label2id[triple[1]])
                    if subject_ids not in sro:
                        sro[subject_ids] = []
                    sro[subject_ids].append(object_ids)
                # else:
                #     print("sentence:", text)
                #     print("triple:", triple)

            if len(sro) > 0:
                subject_heads = np.zeros(token_len)
                subject_tails = np.zeros(token_len)

                subject_head = np.zeros(token_len)
                subject_tail = np.zeros(token_len)

                object_heads = np.zeros((token_len, self.config.rel_num))
                object_tails = np.zeros((token_len, self.config.rel_num))

                # 随机采样
                subject = random.choice(list(sro.keys()))
                subject_head[subject[0]] = 1
                subject_tail[subject[1]] = 1

                for s in sro:
                    subject_heads[s[0]] = 1
                    subject_tails[s[1]] = 1

                for o in sro[subject]:
                    object_heads[o[0]][o[2]] = 1
                    object_tails[o[1]][o[2]] = 1

            else:
                subject_heads = np.zeros(token_len)
                subject_tails = np.zeros(token_len)

                subject_head = np.zeros(token_len)
                subject_tail = np.zeros(token_len)

                object_heads = np.zeros((token_len, self.config.rel_num))
                object_tails = np.zeros((token_len, self.config.rel_num))

            return text, triple_list, input_ids, mask, subject_heads, subject_tails, \
                   subject_head, subject_tail, object_heads, object_tails, token_len, token

        else:
            subject_heads = np.zeros(token_len)
            subject_tails = np.zeros(token_len)

            subject_head = np.zeros(token_len)
            subject_tail = np.zeros(token_len)

            object_heads = np.zeros((token_len, self.config.rel_num))
            object_tails = np.zeros((token_len, self.config.rel_num))

        return text, triple_list, input_ids, mask, subject_heads, subject_tails, \
               subject_head, subject_tail, object_heads, object_tails, token_len, token

def collate_fn(batch):
    text, triple_list, input_ids, mask, subject_heads, subject_tails, \
    subject_head, subject_tail, object_heads, object_tails, token_len, token = zip(*batch)
    cur_batch = len(batch)
    max_token_length = max(token_len)

    batch_input_ids = torch.LongTensor(cur_batch, max_token_length).zero_()
    batch_mask = torch.LongTensor(cur_batch, max_token_length).zero_()
    batch_subject_head = torch.Tensor(cur_batch, max_token_length).zero_()
    batch_subject_tail = torch.Tensor(cur_batch, max_token_length).zero_()
    batch_subject_heads = torch.Tensor(cur_batch, max_token_length).zero_()
    batch_subject_tails = torch.Tensor(cur_batch, max_token_length).zero_()
    batch_object_heads = torch.Tensor(cur_batch, max_token_length, 53).zero_()
    batch_object_tails = torch.Tensor(cur_batch, max_token_length, 53).zero_()

    for i in range(cur_batch):
        batch_input_ids[i, :token_len[i]].copy_(torch.from_numpy(input_ids[i]))
        batch_mask[i, :token_len[i]].copy_(torch.from_numpy(mask[i]))
        batch_subject_head[i, :token_len[i]].copy_(torch.from_numpy(subject_head[i]))
        batch_subject_tail[i, :token_len[i]].copy_(torch.from_numpy(subject_tail[i]))
        batch_subject_heads[i, :token_len[i]].copy_(torch.from_numpy(subject_heads[i]))
        batch_subject_tails[i, :token_len[i]].copy_(torch.from_numpy(subject_tails[i]))
        batch_object_heads[i, :token_len[i], :].copy_(torch.from_numpy(object_heads[i]))
        batch_object_tails[i, :token_len[i], :].copy_(torch.from_numpy(object_tails[i]))

    return {"text": text,
            "token": token,
            "triple_list": triple_list,
            "input_ids": batch_input_ids,
            "mask": batch_mask,
            "subject_head": batch_subject_head,
            "subject_tail": batch_subject_tail,
            "subject_heads": batch_subject_heads,
            "subject_tails": batch_subject_tails,
            "object_heads": batch_object_heads,
            "object_tails": batch_object_tails}

if __name__ == '__main__':

    # token = ['[CLS]'] + list("2011年8月13日，杰西卡阿尔芭生下第二个女儿海雯·加纳·沃伦（Haven Garner Warren），重7磅，长19英寸，健康活泼 with Haven") + ['[SEP]']
    # print(token)
    # triple = (['H', 'a', 'v', 'e', 'n', ' ', 'G', 'a', 'r', 'n', 'e', 'r', ' ', 'W', 'a', 'r', 'r', 'e', 'n'], '人物/母亲/人物', ['杰', '西', '卡', '·', '阿', '尔', '芭'])
    # ids = find_idx(token, triple[0])
    # print(ids)
    # raise
    from config.config import Config
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    config = Config()
    dataset = MyDataset(config, config.train_data)
    dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)
    for data in tqdm(dataloader):
        print(end='')