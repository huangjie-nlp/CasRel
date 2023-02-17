# CasRel

快速实现中文实体关系抽取

## 实验环境安装 
pip install -r requirements

## 说明
1、下载中文bert模型放到bert-base-chinese文件下  
2、dataset文件夹下有chip文件，把你的数据按照chip下文件的格式处理好，在dataset文件夹下建立你的文件夹，并把config.py中的dataset名改你的文件夹名

## 运行

快速实现CasRel模型中文实体关系抽取,2020ACL论文：A Novel Cascade Binary Tagging Framework for Relational Triple Extraction


实验环境安装 pip install -r requirements


dataset文件夹下有chip文件，把你的数据按照chip下文件的格式处理好，在dataset文件夹下建立你的文件夹，并把config.py中的dataset名改你的文件夹名


run main.py 可以训练模型并且输出测试结果的PRF


test.py 控制台抽取句子三元组  

## 结果  
模型最后训练完的结果: F1_score: 51.28%  

## 更新日志  
2023.02.16更新代码

