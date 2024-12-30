# Multi GPU PyTorch Test

## PyTorch fundamentals

### Installation

See [Start PyTorch locally](https://pytorch.org/get-started/locally/)

### 

## Object Detection

How to training of [COCO2017 dataset](https://cocodataset.org)

### Data preparation

#### VOC

#### COCO2017

Go to [COCO2017 dataset page](https://cocodataset.org/#download) and download `2017 Train images [118K/18GB]`, `2017 Val images [5K/1GB]` and `2017 Train/Val annotations [241MB]` files.

Then unzip and locate the downloaded files as follows

```console
```

### Model usage

#### YOLOX

[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) is high-performance anchor-free YOLO with user-friendly lisence (Apache-2.0)

##### Installation

```bash
git clone git@github.com:Megvii-BaseDetection/YOLOX.git
cd YOLOX
pip3 install -v -e .
```

##### Training

First, create the Exp file with referring to the [examples](https://github.com/Megvii-BaseDetection/YOLOX/tree/main/exps/example). You can also use the [default Exp files](https://github.com/Megvii-BaseDetection/YOLOX/tree/main/exps/default).

For example, if you want to train YOLOX-m with COCO dataset, the following Exp file is recommended.

```python
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
```

Then, run the training script with the following command.

```bash
cd /path/to/yolox/root
python -m yolox.tools.train -f /path/to/exp/file -d 8 -b 64 --fp16 -o [--cache]
```

- -f: path to the Exp file
- -d: number of gpu devices
- -b: total batch size, the recommended number for -b is num-gpu * 8
- --fp16: mixed precision training
- --cache: caching imgs into RAM to accelarate training, which need large system RAM.

For example, if you want to use 1GPU, fp16, and default yolox-m Exp file, the following command is recommended

```bash
python -m yolox.tools.train -n exps/default/yolox_m.py -d 1 -b 8 --fp16 -o --cache
```

If you want to use 2GPUs, fp16, and the created Exp file above, the following command is recommended

```bash
python -m yolox.tools.train -n path/to/exp/file/above -d 2 -b 16 --fp16 -o --cache
```

#### DETR

[DETR](https://github.com/facebookresearch/detr) is an object detection pipeline with a Transformer

##### Installation

Install DETR

```bash
git clone https://github.com/facebookresearch/detr.git
```

Then, install cython, scipy, and pycocotools.

```bash
pip install Cython scipy pycocotools
```

##### Training

```bash
cd /path/to/detr/root
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --coco_path /path/to/coco
```

- --nproc_per_node: number of gpu devices
- --batch_size: batch size (default=2)
- --epochs: number of epochs (default=300)
- --num_workers: number of subprocesses for loading the dataset (default=2)
- --coco_path: path to the training data

The arguments above are based on [torch.distributed.launch](https://pytorch.org/docs/stable/distributed.html#launch-utility) and [main.py](https://github.com/facebookresearch/detr/blob/main/main.py) arguments.

For example, if you want to use 2GPUs, the following command is recommended.

```bash
python -m torch.distributed.launch --nproc_per_node=2 --use_env scripts/object_detection/detr/main.py --coco_path datasets/COCO --num_workers 4
```

#### Performance

##### YOLOX-s

|GPU Model|Number of GPU|fp|batch size|Score|
|---|---|---|---|---|
|||||

##### YOLOX-m

### DETR

### RT-DETR