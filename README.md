# torch-benchmarks
Run benchmarks of popular PyTorch models locally

## Installation

Install [Nvidia GPU Driver](https://www.nvidia.com/en-us/drivers/), [CUDA Toolkit](https://developer.nvidia.com/cuda-12-4-0-download-archive), and [cuDNN](https://developer.nvidia.com/cudnn-downloads)

### Installation for Object Detection

Run the installation script.

> Note: The following command is for CUDA12.4 (cu124). If you are using a different version of CUDA, change the command in accordance with your version.

```bash
cd object_detection
sudo chmod +x installation.sh
./installation.sh cu124
```

Then, give the execution permittion to the benchmark scripts

```bash
sudo chmod +x train_benchmark.sh
```

## Usage

### Object detection

#### Training speed benchmark

```bash
cd object_detection
./train_benchmark.sh experiments.csv
```

### Training accuracy benchmark
