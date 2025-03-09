# Fine-Grained Visual Classification via Adaptive Attention Quantization Transformer


## Framework
![p2_v4_00](https://github.com/user-attachments/assets/2a0b90bc-c7ee-4cfd-a2bc-f7aadd5135e9)

## Prerequisites

The following packages are required to run the scripts:

- Python >= 3.6
- PyTorch = 1.8.1
- Torchvision = 0.9.1
- Apex

## Download Google pre-trained ViT models

- [Get models in this link](https://console.cloud.google.com/storage/vit_models/): ViT-B_16, ViT-B_32...

```shell
wget https://storage.googleapis.com/vit_models/imagenet21k/{MODEL_NAME}.npz
```

## Dataset

You can download the datasets from the links below:

- [CUB-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)
- [Stanford Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/)
- [NABirds](http://dl.allaboutbirds.org/nabirds)
- [iNaturalist 2017](https://github.com/visipedia/inat_comp/tree/master/2017)

## Install required packages

Install Prerequisites with the following command:

```shell
pip3 install -r requirements.txt
```

## Train

 CUB-200-2011

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 --master_port 12345 train.py --dataset CUB --img_size 400 --train_batch_size 4 --eval_batch_size 4 --learning_rate 0.02 --num_steps 40000 --fp16 --low_memory --eval_every 200 --name sample_run --aplly_BE
```

 Stanford dogs

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 --master_port 12345 train.py --dataset dogs --img_size 400 --train_batch_size 4 --eval_batch_size 4 --learning_rate 0.003 --num_steps 10000 --fp16 --low_memory --eval_every 200 --name sample_run --aplly_BE
```

NAbirds

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 --master_port 12345 train.py --dataset nabirds --img_size 400 --train_batch_size 4 --eval_batch_size 4 --learning_rate 0.02 --num_steps 60000 --fp16 --low_memory --eval_every 200 --name sample_run --aplly_BE
```

INat2017

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 --master_port 12345 train.py --dataset INat2017 --img_size 304 --train_batch_size 16 --eval_batch_size 16 --learning_rate 0.01 --num_steps 271500 --fp16 --low_memory --eval_every 9050 --name sample_run 
```

## Acknowledge

Our project references the codes in the following repos. Thanks for thier works and sharing.
- [ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch)
- [TransFG](https://github.com/TACJu/TransFG)
