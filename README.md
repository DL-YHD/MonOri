# MonOri
Released code for MonOri: Orientation-Guided PnP For Monocular 3D Object Detection


**Work in progress.**


## Installation
This repo is tested with Ubuntu 22.04, gcc=11.4, g++=11.4, python==3.7, pytorch==1.7.1 and cuda==11

```bash
conda create -n monori python=3.7

conda activate monori
```

Install PyTorch and other dependencies:

```bash
# conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

pip install -r requirements.txt

pip install inplace_abn
```

Build DCNv2 (We provide the latest DCNv2 version that can build on the CUDA 11 and lets the project to run at RTX30 GPU) and the project, you can download at yourself on [DCNv2_lates](https://github.com/lucasjinreal/DCNv2_latest)

```bash
cd model/backbone/DCNv2

. make.sh
```

## Data Preparation
- Please download [KITTI dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) and organize the data as follows:

```
ImageSets		
  |training/
    |calib/
      |000000.txt
        ......
      |007517.txt
    |image_2/
      |000000.png
        ......
      |007517.png
    |label_2/
      |000000.txt
        ......
      |007517.txt
    |test.txt

  |testing/
    |calib/
      |000000.txt
        ......
      |007480.txt
    |image_2/
      |000000.png
        ......
      |007517.png
    |trainval.txt
    |train.txt
    |val.txt
```
- If you want to use semi-supervised training methods, you can follow the [LPCG](https://github.com/SPengLiang/LPCG) to prepare the dataset and then specify the parameters in config/paths_catalog.py.

## Config Parameters Setting
- First of all. You must specify either 'trainval', 'test'  or 'train', 'val' in 'DATASETS:TRAIN_SPLIT: and TEST_SPLIT: ' parameters at '. /run/monori.yaml'.

- Then modify the paths in config/paths_catalog.py according to your data path. 
(If the kitti dataset is placed in the program's 'ImageSets' directory, you don't need to do any changes.) 

## Training
You can set 'CUDA_VISIBLE_DEVICES' and '--gpus' to use one or more GPU for training.

```bash
CUDA_VISIBLE_DEVICES=0 python tools/plain_train_net.py --num_gpus 1 --batch_size 8 --config runs/monori.yaml --output output/exp
```

You can also set the '--ckpt' parameter to specify a pre-trained model.

```bash
CUDA_VISIBLE_DEVICES=0 python tools/plain_train_net.py --num_gpus 1 --batch_size 8 --config runs/monori.yaml --ckpt YOUR_CKPT --output output/exp
```

#  Evaluation
During the evaluation phase, you can evaluate the checkpoint by specifying the '--eval' parameter and executing the following command.

```bash
CUDA_VISIBLE_DEVICES=0 python tools/plain_train_net.py --num_gpus 1 --config runs/monori.yaml --ckpt YOUR_CKPT  --eval
```

If you want to evaluate on the test set and upload it to the official website, you can specify the '--test' parameter.

```bash
CUDA_VISIBLE_DEVICES=0 python tools/plain_train_net.py --num_gpus 1 --config runs/monori.yaml --ckpt YOUR_CKPT  --eval --test
```

#  Visualization
If you want to see the visualization results on predicted heatmap and 3D bounding boxes, you can specify the '--vis' parameter on the evaluate command.

```bash
CUDA_VISIBLE_DEVICES=0 python tools/plain_train_net.py --num_gpus 1 --config runs/monoflex.yaml --ckpt YOUR_CKPT --eval --vis
```
or visualize on the test set.

```bash
CUDA_VISIBLE_DEVICES=0 python tools/plain_train_net.py --num_gpus 1 --config runs/monori.yaml --ckpt YOUR_CKPT  --eval --test --vis
```
You can find our trining logs at [./output](./output/), We also provid our pretrained weight model at [here](https://drive.google.com/file/d/1Qi0DlZImQHY6SJKTRFuwuyuB_epmbYdR/view?usp=sharing)

## Results
The performance on KITTI validation set (3D) is as follows:
<table align="center">
    <tr>
        <td rowspan="2",div align="center">Models</td>
        <td colspan="3",div align="center">Car</td>    
        <td colspan="3",div align="center">Pedestrian</td>  
        <td colspan="3",div align="center">Cyclist</td>  
    </tr>
    <tr>
        <td div align="center">Easy</td> 
        <td div align="center">Mod</td> 
        <td div align="center">Hard</td> 
        <td div align="center">Easy</td> 
        <td div align="center">Mod</td> 
        <td div align="center">Hard</td> 
        <td div align="center">Easy</td> 
        <td div align="center">Mod</td> 
        <td div align="center">Hard</td>  
    </tr>
    <tr>
        <td div align="center">bbox</td>
        <td div align="center">98.3223</td> 
        <td div align="center">92.2792</td> 
        <td div align="center">89.6202</td> 
        <td div align="center">71.7401</td> 
        <td div align="center">62.6488</td> 
        <td div align="center">53.7245</td> 
        <td div align="center">81.6990</td> 
        <td div align="center">57.6980</td> 
        <td div align="center">55.2180</td>  
    </tr>    
    <tr>
        <td div align="center">bev</td>
        <td div align="center">41.5991</td> 
        <td div align="center">32.8592</td> 
        <td div align="center">29.9210</td> 
        <td div align="center">18.8957</td> 
        <td div align="center">14.7460</td> 
        <td div align="center">11.5613</td> 
        <td div align="center">10.4313</td> 
        <td div align="center">6.1389</td> 
        <td div align="center">5.4691</td>  
    </tr>
    <tr>
        <td div align="center">3d</td>
        <td div align="center">29.9924</td> 
        <td div align="center">23.4357</td> 
        <td div align="center">20.7904</td> 
        <td div align="center">16.3173</td> 
        <td div align="center">12.5832</td> 
        <td div align="center">9.7363</td> 
        <td div align="center">10.3087</td> 
        <td div align="center">5.9455</td> 
        <td div align="center">5.3871</td>  
    </tr>
    <tr>
        <td div align="center">aos</td>
        <td div align="center">98.29</td> 
        <td div align="center">92.22</td> 
        <td div align="center">89.31</td> 
        <td div align="center">64.57</td> 
        <td div align="center">54.80</td> 
        <td div align="center">46.74</td> 
        <td div align="center">77.95</td> 
        <td div align="center">54.06</td> 
        <td div align="center">51.71</td>  
    </tr>
</table>

## Citation

If you find our work useful in your research, please consider citing:

```bibtex
@article{yao2023occlusion,
  title={Occlusion-aware plane-constraints for monocular 3D object detection},
  author={Yao, Hongdou and Chen, Jun and Wang, Zheng and Wang, Xiao and Han, Pengfei and Chai, Xiaoyu and Qiu, Yansheng},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  volume={25},
  number={5},
  pages={4593--4605},
  year={2023},
  publisher={IEEE}
},

@article{yao2025monori,
  title={MonOri: Orientation-Guided PnP for Monocular 3-D Object Detection},
  author={Yao, Hongdou and Han, Pengfei and Chen, Jun and Wang, Zheng and Qiu, Yansheng and Wang, Xiao and Chai, Xiaoyu and Cao, Chenglong and Jin, Wei and others},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2025},
  publisher={IEEE}
}
```

## Acknowlegment
The code is extended from [MonoFlex](https://github.com/zhangyp15/MonoFlex), [DSCNet](https://github.com/YaoleiQi/DSCNet) and [CondConv](https://github.com/xmu-xiaoma666/External-Attention-pytorch) thanks to their contribution.
