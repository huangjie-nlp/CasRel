# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :CasRel
# @File     :temp
# @Date     :2021/7/24 15:56
# @Author   :huangjie
# @Email    :728155808@qq.com
# @Software :PyCharm
-------------------------------------------------
"""
from model.casrel import Casrel
from config.config import Config
import  torch
import json
import numpy as np
from transformers import BertTokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

con = Config()
model = Casrel(con).to(device)
tokenizer = BertTokenizer.from_pretrained(con.bert_model_path)
model.load_state_dict(torch.load(con.save_model_name))
def test(model,sentence,con):
    model.eval()
    id2rel = json.load(open(con.schemas,"r",encoding="utf-8"))[1]
    token = ["[CLS]"] + tokenizer.tokenize(sentence) + ["[SEP]"]
    token_ids = tokenizer.convert_tokens_to_ids(token)
    mask = [1] * len(token_ids)
    if len(token_ids) > 512:
        token_ids = token_ids[:512]
        mask = mask[:512]
    token_ids = torch.LongTensor([token_ids]).to(device)
    mask = torch.LongTensor([mask]).to(device)

    encoded_text = model.encoded_text(token_ids, mask)
    pred_sub_heads, pred_sub_tails = model.get_subs(encoded_text)
    pred_sub_heads_idx, pred_sub_tails_idx = np.where(pred_sub_heads.cpu()[0] > con.h_bar)[0], \
                                             np.where(pred_sub_tails.cpu()[0] > con.t_bar)[0]
    subjects = []
    for sub_head in pred_sub_heads_idx:
        sub_tails = pred_sub_tails_idx[pred_sub_tails_idx > sub_head]
        if len(sub_tails):
            sub_tail = sub_tails[0]
            subjects.append(("".join(token[sub_head:sub_tail + 1]), sub_head, sub_tail))
    pred_triple = []
    if len(subjects) > 0:
        # [subject_num, seq_len, bert_dim]
        encoded_text_repeat = encoded_text.repeat(len(subjects), 1, 1)
        # [subject_num, 1, seq_len]
        sub_heads_mapping = torch.Tensor(len(subjects), 1, encoded_text.size(1)).zero_()
        sub_tails_mapping = torch.Tensor(len(subjects), 1, encoded_text.size(1)).zero_()
        for text_ids, sub_idx in enumerate(subjects):
            sub_heads_mapping[text_ids][0][sub_idx[1]] = 1
            sub_tails_mapping[text_ids][0][sub_idx[2]] = 1
        pred_object_heads, pred_object_tails = model.get_object_relation(sub_heads_mapping.to(device),
                                                                         sub_tails_mapping.to(device),
                                                                         encoded_text_repeat.to(device))

        for sub_idx, sub in enumerate(subjects):
            subject = sub[0]
            obj_heads, obj_tails = np.where(pred_object_heads.cpu()[sub_idx] > con.h_bar), \
                                   np.where(pred_object_tails.cpu()[sub_idx] > con.t_bar)
            for object_head, head_rel in zip(*obj_heads):
                for object_tail, tail_rel in zip(*obj_tails):
                    if object_tail >= object_head and head_rel == tail_rel:
                        obj = "".join(token[object_head:object_tail + 1])
                        relation = id2rel[str(head_rel)]
                        pred_triple.append((subject, relation, obj))
                        break
    print("triple_result:\n",pred_triple)
if __name__ == '__main__':
    while True:
        sentence = input("请输入句子:")
        test(model,sentence,con)
