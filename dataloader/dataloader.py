# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :CasRel
# @File     :dataloader
# @Date     :2021/7/24 14:57
# @Author   :huangjie
# @Email    :728155808@qq.com
# @Software :PyCharm
-------------------------------------------------
"""
from torch.utils.data import Dataset,DataLoader
import random
import torch
from transformers import BertTokenizer
import numpy as np
import json
torch.manual_seed(2021)
random.seed(2021)

def find_indx(source,target):
    entity_len = len(target)
    for k,v in enumerate(source):
        if source[k:k+entity_len] == target:
            return k
    return -1

class MyDataset(Dataset):
    def __init__(self,con,fn,is_test=False):
        self.con = con
        self.data = json.load(open(fn,"r",encoding="utf-8"))
        self.is_test = is_test
        self.rel2id = json.load(open(self.con.schemas,"r",encoding="utf-8"))[0]
        self.tokenizer = BertTokenizer.from_pretrained(self.con.bert_model_path)
        self.bert_len = 512

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ins_json = self.data[idx]
        text = ins_json["text"]
        text = "".join(list(text.replace(" ",""))[:self.con.max_len])
        token = ["[CLS]"] + list(text) + ["[SEP]"]
        if len(token) > self.bert_len:
            token = token[:self.bert_len]
        token_len = len(token)

        if not self.is_test:
            sro_map = {}
            for sro in ins_json["triple_list"]:
                triple = (self.tokenizer.tokenize(sro[0]),sro[1],self.tokenizer.tokenize(sro[2]))
                sub_head_idx = find_indx(token,triple[0])
                obj_head_idx = find_indx(token,triple[2])
                if sub_head_idx != -1 and obj_head_idx != -1:
                    sub = (sub_head_idx,sub_head_idx+len(triple[0])-1)
                    if sub not in sro_map:
                        sro_map[sub] = []
                    sro_map[sub].append((obj_head_idx,obj_head_idx+len(triple[2])-1,self.rel2id[triple[1]]))
            if sro_map:
                token_ids = self.tokenizer.convert_tokens_to_ids(token)
                mask = [1] * len(token_ids)
                if len(token_ids) > self.bert_len:
                    token_ids = token_ids[:self.bert_len]
                    mask = mask[:self.bert_len]
                token_ids = np.array(token_ids)
                mask = np.array(mask)

                subject_head,subject_tail= np.zeros(len(token_ids)),np.zeros(len(token_ids))
                # 随机采样
                sub = random.choice(list(sro_map.keys()))
                subject_head[sub[0]] = 1
                subject_tail[sub[1]] = 1

                object_heads,object_tails = np.zeros((len(token_ids),len(self.rel2id))),\
                                            np.zeros((len(token_ids),len(self.rel2id)))

                for obj in sro_map.get(sub):
                    object_heads[obj[0]][obj[2]] = 1
                    object_tails[obj[1]][obj[2]] = 1

                subject_heads, subject_tails = np.zeros(len(token_ids)), np.zeros(len(token_ids))
                for s in sro_map:
                    subject_heads[s[0]] = 1
                    subject_tails[s[1]] = 1

                return token_len,token,token_ids,mask,subject_head,subject_tail,subject_heads,subject_tails,\
                       object_heads,object_tails,ins_json["triple_list"]
            else:
                token_ids = self.tokenizer.convert_tokens_to_ids(token)
                mask = [1] * len(token_ids)
                if len(token_ids) > self.bert_len:
                    token_ids = token_ids[:self.bert_len]
                    mask = mask[:self.bert_len]
                token_ids = np.array(token_ids)
                mask = np.array(mask)
                subject_head, subject_tail = np.zeros(len(token_ids)), np.zeros(len(token_ids))
                object_heads,object_tails = np.zeros((len(token_ids),len(self.rel2id))),\
                                            np.zeros((len(token_ids),len(self.rel2id)))
                subject_heads, subject_tails = np.zeros(len(token_ids)), np.zeros(len(token_ids))
                return token_len,token,token_ids,mask,subject_head,subject_tail,subject_heads,subject_tails,\
                       object_heads,object_tails,ins_json["triple_list"]
        else:
            token_ids = self.tokenizer.convert_tokens_to_ids(token)
            mask = [1] * len(token_ids)
            if len(token_ids) > self.bert_len:
                token_ids = token_ids[:self.bert_len]
                mask = mask[:self.bert_len]
            token_ids = np.array(token_ids)
            mask = np.array(mask)
            subject_head, subject_tail = np.zeros(len(token_ids)), np.zeros(len(token_ids))
            object_heads, object_tails = np.zeros((len(token_ids), len(self.rel2id))), \
                                         np.zeros((len(token_ids), len(self.rel2id)))
            subject_heads, subject_tails = np.zeros(len(token_ids)), np.zeros(len(token_ids))
            return token_len, token, token_ids, mask, subject_head, subject_tail, subject_heads, subject_tails, \
                   object_heads, object_tails, ins_json["triple_list"]

def collate_fn(batch):
    token_len, token, token_ids, mask, subject_head, subject_tail, subject_heads, subject_tails, \
    object_heads, object_tails, triple_list = zip(*batch)

    max_token_lenth = max(token_len)
    cur_batch = len(batch)

    batch_token_ids = torch.LongTensor(cur_batch,max_token_lenth).zero_()
    batch_mask = torch.LongTensor(cur_batch,max_token_lenth).zero_()
    batch_sub_head = torch.Tensor(cur_batch,max_token_lenth).zero_()
    batch_sub_tail = torch.Tensor(cur_batch,max_token_lenth).zero_()
    batch_sub_heads = torch.Tensor(cur_batch,max_token_lenth).zero_()
    batch_sub_tails = torch.Tensor(cur_batch,max_token_lenth).zero_()
    batch_obj_heads = torch.Tensor(cur_batch,max_token_lenth,53).zero_()
    batch_obj_tails = torch.Tensor(cur_batch,max_token_lenth,53).zero_()

    for i in range(cur_batch):
        batch_token_ids[i,:token_len[i]].copy_(torch.from_numpy(token_ids[i]))
        batch_mask[i,:token_len[i]].copy_(torch.from_numpy(mask[i]))
        batch_sub_head[i,:token_len[i]].copy_(torch.from_numpy(subject_head[i]))
        batch_sub_tail[i,:token_len[i]].copy_(torch.from_numpy(subject_tail[i]))
        batch_sub_heads[i,:token_len[i]].copy_(torch.from_numpy(subject_heads[i]))
        batch_sub_tails[i,:token_len[i]].copy_(torch.from_numpy(subject_tails[i]))
        batch_obj_heads[i,:token_len[i],:].copy_(torch.from_numpy(object_heads[i]))
        batch_obj_tails[i,:token_len[i],:].copy_(torch.from_numpy(object_tails[i]))

    return {
            "token":token,
            "token_ids":batch_token_ids,
            "mask":batch_mask,
            "sub_head":batch_sub_head,
            "sub_tail":batch_sub_tail,
            "sub_heads":batch_sub_heads,
            "sub_tails":batch_sub_tails,
            "obj_heads":batch_obj_heads,
            "obj_tails":batch_obj_tails,
            "triple":triple_list
            }

if __name__ == '__main__':
    from config.config import Config
    con = Config()
    fn = "../dataset/chip/train_data.json"
    dataset = MyDataset(con,fn)
    dataloader = DataLoader(dataset,batch_size=2,collate_fn=collate_fn,pin_memory=True)
    for i in dataloader:
        print(i["mask"])
