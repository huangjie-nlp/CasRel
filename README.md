# casrel-torch
快速实现中文实体关系抽取


实验环境安装 pip install -r requirements


下载中文bert模型放到pretrain文件下


dataset文件夹下有chip文件，把你的数据按照chip下文件的格式处理好，在dataset文件夹下建立你的文件夹，并把config.py中的dataset名改你的文件夹名


run main.py 可以训练模型并且输出测试结果的PRF


test.py 控制台抽取句子三元组
