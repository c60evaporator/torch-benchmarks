import os

from yolox.exp import Exp as MyExp

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.67
        self.width = 0.75
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # Define yourself dataset path
        self.data_dir = "datasets/COCO"
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"

        self.input_size = (640, 640)  # (height, width)
        self.num_classes = 80  # Number of classes of the dataset

        self.max_epoch = 2  # Number of epochs
        self.data_num_workers = 4  # Number of subprocesses for loading the dataset (> GPUx2 is recommended)