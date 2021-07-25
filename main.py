# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :CasRel
# @File     :train
# @Date     :2021/7/24 22:28
# @Author   :huangjie
# @Email    :728155808@qq.com
# @Software :PyCharm
-------------------------------------------------
"""
from model.casrel import Casrel
from config.config import Config
from framework.framework import Framework
con = Config()

train_fn = "dataset/"+con.dataset+"/train_data.json"
dev_fn = "dataset/"+con.dataset+"/val_data.json"

model = Casrel(con)
framework = Framework(con)
framework.train(model,train_fn,dev_fn)
print("start test......")
framework.test(model,dev_fn)