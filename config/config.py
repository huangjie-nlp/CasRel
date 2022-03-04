# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :CasRel
# @File     :config
# @Date     :2021/7/24 14:22
# @Author   :huangjie
# @Email    :728155808@qq.com
# @Software :PyCharm
-------------------------------------------------
"""

class Config():
    def __init__(self):
        self.dataset = "chip"
        self.bert_dim = 768
        self.bert_model_path = "pretrain/cased_L-12_H-768_A-12"
        self.num_rel = 53
        self.schemas = "dataset/"+self.dataset+"/schemas.json"
        self.max_len = 256
        self.batch_size = 16
        self.lr = 1e-5
        self.epoch = 300
        self.eval_fn_name = "eval_result/" + self.dataset+"_casrel_eval.json"
        self.h_bar = 0.5
        self.t_bar = 0.5
        self.save_model_name = "checkpoint/" + self.dataset + "_casrel_model"
        self.test_fn_result = "test_result/" + self.dataset+"_casrel_test.json"
        self.log = "log/"+self.dataset+"_log.log"