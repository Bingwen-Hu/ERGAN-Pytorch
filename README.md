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
|   |    ├──  celeba.yaml
|   |    ├──  meglass.yaml
│   ├── models/                 /* Files for pretrained model    	
│   ├── outputs/		/* Intermediate image outputs 		
│   ├── datasets/CelebA/
                 ├── trainA/	/* Training set: face images without glasses		
                 ├── trainB/	/* Training set: face images with glasses		
                 ├── testA/	/* Testing set: face images without glasses		
                 └── testB/	/* Testing set: face images with glasses		
│   ├── datasets/MeGlass/
│                ├── trainA/	/* Training set: face images without glasses		
│                ├── trainB/	/* Training set: face images with glasses		
│                ├── testA/	/* Testing set: face images without glasses		
│                └── testB/	/* Testing set: face images with glasses
```

## Dataset Preparation
Download the celebA Dataset [Here]( https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8 ). Download the MeGlass Dataset [Here](https://drive.google.com/file/d/1V0c8p6MOlSFY5R-Hu9LxYZYLXd8B8j9q/view).

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
First, download our pretrained models ([Google Drive](https://drive.google.com/file/d/1ap7qB6rkKjx5K2lrnzJ8eIHlpzW4fnh5/view?usp=sharing)) for the eyeglasses removal task and put them in models folder. 
If you want to test your own data with our pre-trained model, you need to align the data first ( refer to CelebA) or retrain with your own data.

for CelebA:
```bash
python test_batch.py --config configs/celeba.yaml --A input_path_A --B input_path_B --output_folder results/celeba --checkpoint models/celeba.pt
```

for MeGlass:
```bash
python test_batch.py --config configs/meglass.yaml --A input_path_A --B input_path_B --output_folder results/meglass --checkpoint models/meglasss.pt
```

`--A` The PATH of the test set (without glasses).

`--B` The PATH of the test set (with glasses).


The results are stored in results/celeba folder and results/meglass folder, respectively.

## Citation
If you find ERGAN is useful in your research, please consider citing:
```
@article{hu2020unsupervised,
  title={Unsupervised eyeglasses removal in the wild},
  author={Hu, Bingwen and Zheng, Zhedong and Liu, Ping and Yang, Wankou and Ren, Mingwu},
  journal={IEEE Transactions on Cybernetics},
  year={2020},
  publisher={IEEE}
}

```

## Related Repos
1. [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
2. [UNIT](https://github.com/mingyuliutw/UNIT)
3. [MUNIT](https://github.com/NVlabs/MUNIT)
4. [LPIPS](https://github.com/richzhang/PerceptualSimilarity)
5. [FID](https://github.com/bioinf-jku/TTUR)
6. [MeGlass](https://github.com/cleardusk/MeGlass)
## Acknowledgments
Our code is inspired by MUNIT.
