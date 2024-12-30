# Create "models" folder
sudo rm -rf models/benchmark_env
sudo rm -rf models/detr
sudo rm -rf models/YOLOX
cd models
# Create Python virtual environment
python3 -m venv benchmark_env
source benchmark_env/bin/activate
# Install PyTorch 2.4.1
echo "Specified CUDA version is $1"
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/$1
# Install other python packages
cd ..
pip3 install -r requirements.txt
cd models
# Install YOLOX
sudo apt install cmake
git clone https://github.com/Megvii-BaseDetection/YOLOX.git
cd YOLOX
# Modify and install YOLOX dependencies
sed -i -e 's/onnx-simplifier==/onnx-simplifier>=/g' ./requirements.txt
pip3 install -v -e . --no-build-isolation
cd ..
# Install DETR
git clone https://github.com/facebookresearch/detr.git
cd ..
# Create "pretrained_weights" folder
sudo rm -rf pretrained_weights
mkdir -p pretrained_weights
cd pretrained_weights
# Download YOLOX pretrained weights
mkdir -p YOLOX
cd YOLOX
wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth
wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.pth
wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_l.pth
wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_x.pth
cd ..
# Download DETR pretrained weights
mkdir -p detr
cd detr
wget https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth
wget https://dl.fbaipublicfiles.com/detr/detr-r101-2c7b67e5.pth
wget https://dl.fbaipublicfiles.com/detr/detr-r101-dc5-a2e86def.pth
cd ..
cd ..
# Create "datasets" folder
mkdir -p datasets
cd datasets
# Download COCO dataset
sudo rm -rf COCO
mkdir -p COCO
cd COCO
echo "Downloading COCO training images"
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip
rm train2017.zip
echo "Downloading COCO validation images"
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip
rm val2017.zip
echo "Downloading COCO annotations"
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip
rm annotations_trainval2017.zip
cd ..
cd ..
# Create "results" folder
mkdir -p results
# Create "tmp" folder
mkdir -p tmp
