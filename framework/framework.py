# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :CasRel
# @File     :framework
# @Date     :2021/7/24 17:32
# @Author   :huangjie
# @Email    :728155808@qq.com
# @Software :PyCharm
-------------------------------------------------
"""
import time
import numpy as np
import json
import torch
import torch.nn.functional as F
from dataloader.dataloader import MyDataset,collate_fn
from torch.utils.data import DataLoader


class Framework():
    def __init__(self,con):
        self.con = con
        self.id2rel = json.load(open(self.con.schemas,"r",encoding="utf-8"))[1]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def train(self,model,train_fn,dev_fn):

        def loss_fn(pred,target,mask):
            """
            :param pred: [batch_size,seq_len,1]
            :param target: [batch_size,seq_len] or [batch_size,seq_len,num_rel]
            :param mask: [batch_size,seq_len]
            :return:
            """
            pred = pred.squeeze(dim=-1)
            if mask.shape != pred.shape:
                # [batch_size,seq_len,1]
                mask = mask.unsqueeze(dim=-1)
            los = F.binary_cross_entropy(pred,target,reduction="none")
            loss = torch.sum(los * mask) / torch.sum(mask)
            return loss


        train_dataset = MyDataset(self.con,train_fn)
        trainloader = DataLoader(train_dataset,shuffle=True,batch_size=self.con.batch_size,
                                 collate_fn=collate_fn,pin_memory=True)
        dev_dataset = MyDataset(self.con,dev_fn,is_test=True)
        devloader = DataLoader(dev_dataset,batch_size=1,collate_fn=collate_fn,pin_memory=True)

        model = model.to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(),lr=self.con.lr)

        best_F1 = -1
        best_epoch = -1
        best_precision = -1
        best_recall = -1
        global_step = 0
        global_step_loss = 0
        for epoch in range(self.con.epoch):
            epoch_loss = 0
            init_time = time.time()
            for data in trainloader:
                pred_subject_heads, pred_subject_tails, pred_object_heads, pred_object_tails = model(data)
                sub_heads_loss = loss_fn(pred_subject_heads,data["sub_heads"].to(self.device),data["mask"].to(self.device))
                sub_tails_loss = loss_fn(pred_subject_tails,data["sub_tails"].to(self.device),data["mask"].to(self.device))
                obj_heads_loss = loss_fn(pred_object_heads,data["obj_heads"].to(self.device),data["mask"].to(self.device))
                obj_tails_loss = loss_fn(pred_object_tails,data["obj_tails"].to(self.device),data["mask"].to(self.device))

                loss = (sub_heads_loss+sub_tails_loss) + (obj_heads_loss+obj_tails_loss)
                model.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                global_step_loss += loss.item()
                if (global_step+1) % 100 == 0:
                    print("epoch:{},global_step:{},global_step_loss:{:5.4f}".format(epoch,global_step,global_step_loss))
                    global_step_loss = 0
                global_step += 1
            if (epoch+1) % 5 == 0:
                precision,recall,f1_score = self.evaluate(model,devloader,self.con.eval_fn_name)
                if f1_score > best_F1:
                    best_F1 = f1_score
                    best_epoch = epoch
                    best_recall = recall
                    best_precision = precision
                    print("precision:{:5.4f},recall:{:5.4f},f1_score:{:5.4f},best_f1_score:{:5.4f},best_epoch:{:3d}".
                          format(precision,recall,f1_score,best_F1,best_epoch))
                    print("save model...")
                    torch.save(model.state_dict(),self.con.save_model_name)
            print("epoch:{:3d}, epoch_loss:{:5.4f}, cost:{:5.2f} min".format(epoch,epoch_loss,(time.time()-init_time)/60))
        print("best_epoch:{:3d},best_precision:{:5.4f},best_recall:{:5.4f},best_f1_score:{:5.4f}".
              format(best_epoch,best_precision,best_recall,best_F1))
    def evaluate(self,model,dataloader,save_fn):
        init_time = time.time()
        model.eval()
        correct_num = 0
        predict_num = 0
        gold_num = 0
        predict = []
        def to_tuple(data):
            ret = []
            for i in data:
                ret.append(tuple(i))
            return tuple(ret)
        with torch.no_grad():
            for data in dataloader:
                token_ids = data["token_ids"].to(self.device)
                mask = data["mask"].to(self.device)
                token = data["token"][0]
                triple = data["triple"][0]
                encoded_text = model.encoded_text(token_ids,mask)
                pred_sub_heads, pred_sub_tails = model.get_subs(encoded_text)
                pred_sub_heads_idx,pred_sub_tails_idx = np.where(pred_sub_heads.cpu()[0] > self.con.h_bar)[0],\
                                                        np.where(pred_sub_tails.cpu()[0] > self.con.t_bar)[0]
                subjects = []
                for sub_head in pred_sub_heads_idx:
                    sub_tails = pred_sub_tails_idx[pred_sub_tails_idx > sub_head]
                    if len(sub_tails):
                        sub_tail = sub_tails[0]
                        subjects.append(("".join(token[sub_head:sub_tail+1]),sub_head,sub_tail))
                pred_triple = []
                if len(subjects) > 0:
                    # [subject_num, seq_len, bert_dim]
                    encoded_text_repeat = encoded_text.repeat(len(subjects),1,1)
                    # [subject_num, 1, seq_len]
                    sub_heads_mapping = torch.Tensor(len(subjects),1,encoded_text.size(1)).zero_()
                    sub_tails_mapping = torch.Tensor(len(subjects),1,encoded_text.size(1)).zero_()
                    for text_ids,sub_idx in enumerate(subjects):
                        sub_heads_mapping[text_ids][0][sub_idx[1]] = 1
                        sub_tails_mapping[text_ids][0][sub_idx[2]] = 1
                    pred_object_heads,pred_object_tails = model.get_object_relation(sub_heads_mapping.to(self.device),
                                                                                    sub_tails_mapping.to(self.device),
                                                                                    encoded_text_repeat.to(self.device))

                    for sub_idx ,sub in enumerate(subjects):
                        subject = sub[0]
                        obj_heads,obj_tails = np.where(pred_object_heads.cpu()[sub_idx] > self.con.h_bar),\
                                              np.where(pred_object_tails.cpu()[sub_idx] > self.con.t_bar)
                        for object_head,head_rel in zip(*obj_heads):
                            for object_tail,tail_rel in zip(*obj_tails):
                                if object_tail >= object_head and head_rel == tail_rel:
                                    obj = "".join(token[object_head:object_tail+1])
                                    relation = self.id2rel[str(head_rel)]
                                    pred_triple.append((subject,relation,obj))
                                    break
                triple = set(to_tuple(triple))
                pred_triple = set(to_tuple(pred_triple))
                predict.append({"text":"".join(token[1:-1]),
                                "gold":list(triple),
                                "predict":list(pred_triple),
                                "lack":list(triple - pred_triple),
                                "new":list(pred_triple - triple)})
                gold_num += len(triple)
                predict_num += len(pred_triple)
                correct_num += len(triple & pred_triple)
        json.dump(predict,open(save_fn,"w",encoding="utf-8"),indent=4,ensure_ascii=False)
        precision = correct_num / (predict_num + 1e-10)
        recall = correct_num / (gold_num + 1e-10)
        f1_score = 2 * precision * recall / (precision + recall + 1e-10)
        print("evauate model cost {}s".format(time.time()-init_time))
        print("predict_num:{},correct:{},gold_num{}".format(predict_num,correct_num,gold_num))
        model.train()
        return precision,recall,f1_score

    def test(self,model,test_fn):
        dataset = MyDataset(self.con,test_fn,is_test=True)
        dataloader = DataLoader(dataset,shuffle=True,batch_size=1,collate_fn=collate_fn,pin_memory=True)
        print("load model......")
        model.load_state_dict(torch.load(self.con.save_model_name))
        model.to(self.device)
        precision, recall, f1_score = self.evaluate(model,dataloader,self.con.test_fn_result)
        print("test result!!!")
        print("precision:{:5.4f}, recall:{:5.4f}, f1_score:{:5.4f}".format(precision, recall, f1_score))
