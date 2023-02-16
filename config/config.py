
class Config():
    def __init__(self):
        self.dataset = "chip"
        self.rel_num = 53
        self.bert_path = "./bert-base-chinese"
        self.train_data = "./dataset/" + self.dataset + "/train_data.json"
        self.dev_data = "./dataset/" + self.dataset + "/dev_data.json"
        self.test_data = "./dataset/" + self.dataset + "/dev_data.json"
        self.schema = "./dataset/" + self.dataset + "/schemas.json"
        self.log = "log/{}_log.log".format(self.dataset)
        self.learning_rate = 1e-5
        self.batch_size = 16
        self.epoch = 300
        self.step = 1000
        self.val_epoch = 5
        self.h_bar = 0.5
        self.t_bar = 0.5
        self.save_model = "checkpoint/{}_model.pt".format(self.dataset)
        self.dev_result = "dev_result/{}_dev.json".format(self.dataset)
        self.test_result = "test_result/{}_test.json".format(self.dataset)