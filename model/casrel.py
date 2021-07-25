# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :CasRel
# @File     :casrel
# @Date     :2021/7/24 14:19
# @Author   :huangjie
# @Email    :728155808@qq.com
# @Software :PyCharm
-------------------------------------------------
"""

from transformers import BertModel
import torch
import torch.nn as nn
torch.manual_seed(2021)

class Casrel(nn.Module):
    def __init__(self,con):
        super(Casrel, self).__init__()
        self.con = con
        self.bert_dim = self.con.bert_dim
        self.bert_encode = BertModel.from_pretrained(self.con.bert_model_path)
        self.subject_heads_linear = nn.Linear(self.bert_dim,1)
        self.subject_tails_linear = nn.Linear(self.bert_dim,1)
        self.object_heads_relation_linear = nn.Linear(self.bert_dim,self.con.num_rel)
        self.object_tails_relation_linear = nn.Linear(self.bert_dim,self.con.num_rel)
        self.layernor = nn.LayerNorm(self.bert_dim)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def encoded_text(self,input_ids,mask):
        encoded_text = self.bert_encode(input_ids,mask)[0]
        return encoded_text

    def get_subs(self,encoded_text):
        """
        :param encoded_text: [batch_size,seq_len,bert_dim]
        :return:
        """
        # [batch_size,seq_len,bert_dim]
        encoded_text = self.layernor(encoded_text)
        # [batch_size,seq_len,1]
        sub_heads_predict = self.subject_heads_linear(encoded_text)
        sub_heads_predict = torch.sigmoid(sub_heads_predict)
        # [batch_size,seq_len,1]
        sub_tails_predict = self.subject_tails_linear(encoded_text)
        sub_tails_predict = torch.sigmoid(sub_tails_predict)
        return sub_heads_predict,sub_tails_predict

    def get_object_relation(self,subject_heads_mapping,subject_tails_mapping,encoded_text):
        """
        :param subject_heads_mapping: [batch_size,1,seq_len]
        :param suject_tails_mapping: [batch_size,1,seq_len]
        :param encoded_text: [batch_size,seq_len,bert_dim]
        :return:
        """
        # [batch_size,1,bert_dim]
        sub_heads = torch.matmul(subject_heads_mapping,encoded_text)
        # [batch_size,1,bert_dim]
        sub_tails = torch.matmul(subject_tails_mapping,encoded_text)
        # [batch_size,1,bert_dim]
        sub = (sub_heads + sub_tails) / 2
        # [batch_size,seq_len,bert_dim]
        encoded_text = encoded_text + sub
        # [batch_size,seq_len,num_rel]
        pred_object_heads = self.object_heads_relation_linear(encoded_text)
        pred_object_heads = torch.sigmoid(pred_object_heads)
        # [batch_size,seq_len,num_rel]
        pred_object_tails = self.object_tails_relation_linear(encoded_text)
        pred_object_tails = torch.sigmoid(pred_object_tails)
        return pred_object_heads,pred_object_tails

    def forward(self,data):
        # [batch_size,seq_len]
        token_ids = data["token_ids"].to(self.device)
        # [batch_size,seq_len]
        mask = data["mask"].to(self.device)
        # [batch_size,seq_len,bert_dim]
        encoded_text = self.encoded_text(token_ids,mask)
        # [batch_size,seq_len,1]
        pred_subject_heads,pred_subject_tails = self.get_subs(encoded_text)
        # [batch_size,1,seq_len]
        subject_heads = data["sub_heads"].unsqueeze(dim=1).to(self.device)
        # [batch_size,1,seq_len]
        subject_tails = data["sub_tails"].unsqueeze(dim=1).to(self.device)
        pred_object_heads,pred_object_tails = self.get_object_relation(subject_heads,subject_tails,encoded_text)
        return pred_subject_heads,pred_subject_tails,pred_object_heads,pred_object_tails
