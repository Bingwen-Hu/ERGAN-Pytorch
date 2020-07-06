# Unsupervised Eyeglasses Removal in the Wild [[arXiv]](https://arxiv.org/abs/1909.06989)
  By Bingwen Hu, Zhedong Zheng, Ping Liu, Wankou Yang and Mingwu Ren. 

## Prerequisites
- Python 3.6, Ubuntu 14.04
- GPU Memory >= 11G
- conda install pytorch>=0.4.1 torchvision cuda91 -y -c pytorch
- conda install -y -c anaconda pip
- conda install -y -c anaconda pyyaml
- pip install tensorboard tensorboardX

## Getting started
Clone ERGAN source code
```
git clone https://github.com/Bingwen-Hu/ERGAN-Pytorch
```

The folder is structured as follows:
```
├── ERGAN-Pytorch/
│   ├── configs/                /* Files for configs  		
│   ├── models/                 /* Files for pretrained model    	
│   ├── outputs/		/* Intermediate image outputs 		
│   ├── datasets/celebA/
                 ├── trainA/	/* Training set: face images without glasses		
                 ├── trainB/	/* Training set: face images with glasses		
                 ├── testA/	/* Testing set: face images without glasses		
                 └── testB/	/* Testing set: face images with glasses		
	
```

## Dataset Preparation
Download the celebA Dataset [Here]( https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8 ).
We split the CelebA dataset into one subset with glasses and another without glasses, based on the annotated attributes.
```bash
python celeba_glass.py
```
Note to modify the dataset path to your own path.

## Train
Setup the yaml file. Check out configs/celeba.yaml for folder-based dataset organization. Change the data_root field to the path of your downloaded dataset.
```bash
python train.py --config configs/celeba.yaml
```
Intermediate image outputs and model binary files are stored in outputs/celeba.

## Test
First, download our pretrained models [Google Drive](https://drive.google.com/open?id=1tMq9U1Tmn76HBufw7Y3lcETuvHZ5R1PY) for the eyeglasses removal task and put them in models folder.
```bash
python test_batch.py --config configs/celeba.yaml --A input_path_A --B input_path_B --output_folder results/celeba --checkpoint models/celeba.pt
```
`--A` The PATH of the test set (without glasses).

`--B` The PATH of the test set (with glasses).
The results are stored in results/celeba folder.

### NOTE 
The pre-trained model we provided was trained on the CelebA dataset. If you want to use the pre-trained model to process other data (such as LFW, Meglass), you need to adjust the size of the eye area according to the shape of the input image and then fine-tune the model to get better results.

## Citation
If you find ERGAN useful in your research, please consider citing:
```
@article{hu2019unsupervised,
  title={Unsupervised Eyeglasses Removal in the Wild},
  author={Hu, Bingwen and Yang, Wankou and Ren, Mingwu},
  journal={arXiv preprint arXiv:1909.06989},
  year={2019}
}

```

## Related Repos
1. [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
2. [UNIT](https://github.com/mingyuliutw/UNIT)
3. [MUNIT](https://github.com/NVlabs/MUNIT)

## Acknowledgments
Our code is inspired by MUNIT.
