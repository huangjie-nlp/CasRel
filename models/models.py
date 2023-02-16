import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

class Casrel(nn.Module):
    def __init__(self, config):
        super(Casrel, self).__init__()
        self.config = config
        self.bert_dim = 768
        self.bert = BertModel.from_pretrained(self.config.bert_path)
        self.subject_heads_fc = nn.Linear(self.bert_dim, 1)
        self.subject_tails_fc = nn.Linear(self.bert_dim, 1)
        self.object_heads_fc = nn.Linear(self.bert_dim, self.config.rel_num)
        self.object_tails_fc = nn.Linear(self.bert_dim, self.config.rel_num)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ln = nn.LayerNorm(self.bert_dim)

    def get_text_encode(self, input_ids, mask):
        encode = self.bert(input_ids, attention_mask=mask)[0]
        encode = self.ln(encode)
        return encode

    def get_subject(self, text_encode):
        batch_size = text_encode.size(0)
        # [batch_size, sequence_length, 1]
        self.subject_heads = self.subject_heads_fc(text_encode)
        # [batch_size, sequence_length]
        self.subject_heads = self.subject_heads.view([batch_size, -1])
        # [batch_size, sequence_length, 1]
        self.subject_tails = self.subject_tails_fc(text_encode)
        # [batch_size, sequence_length]
        self.subject_tails = self.subject_tails.view([batch_size, -1])
        return F.sigmoid(self.subject_heads), F.sigmoid(self.subject_tails)

    def get_special_relation_object(self, text_encode, subject_head_map, subject_tail_map):
        """
        :param text_encode: [batch_size, sequence_length, bert_dim]
        :param subject_head_map: [batch_size, 1, sequence_length]
        :param subeject_tail_map: [batch_size, 1, sequence_length]
        :return: [batch_size, sequence, num_rel]
        """
        # [batch_size, 1, bert_dim]
        subject_heads = torch.matmul(subject_head_map, text_encode)
        subject_tails = torch.matmul(subject_tail_map, text_encode)

        # [batch_size, 1, bert_dim]
        subjects = (subject_heads +subject_tails) / 2

        # [batch_size, sequence_length, bert_dim]
        feature = text_encode + subjects

        object_heads = self.object_heads_fc(feature)
        object_tails = self.object_tails_fc(feature)
        return F.sigmoid(object_heads), F.sigmoid(object_tails)

    def forward(self, data):
        input_ids = data["input_ids"].to(self.device)
        mask = data["mask"].to(self.device)

        # [batch_size, sequence_length]
        subject_heads_map = data["subject_heads"]
        subject_tails_map = data["subject_tails"]

        # [batch_size, 1, sequence_length]
        subject_heads_map = subject_heads_map.unsqueeze(dim=1).to(self.device)
        subject_tails_map = subject_tails_map.unsqueeze(dim=1).to(self.device)

        # [batch_size, sequence_length, bert_dim]
        text_encode = self.get_text_encode(input_ids, mask)

        # [batch_size, sequence_length]
        subject_heads, subject_tails = self.get_subject(text_encode)
        object_heads, object_tails = self.get_special_relation_object(text_encode, subject_heads_map, subject_tails_map)
        return subject_heads, subject_tails, object_heads, object_tails
