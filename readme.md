# Vision Transformer 图片分类

#### 介绍
智能语音助理

#### 软件架构
软件架构说明


#### 本地环境安装

1. 创建环境  `conda create -n vit-classification python=3.10` 
2. 切换环境 `conda activate vit-classification`
3. 安装依赖 `pip install -r requirements.txt`
4. 环境变量 `export HF_ENDPOINT=https://hf-mirror.com 
           export HF_HOME=/hugging_face_home`
5. 在/hugging_face_home/hub目录下放模型
6. 运行 `python train.py` 注意要先设置环置变量


#### 使用说明

1.  数据存放到datasets/train和datasets/test目录。
2.  datasets/train和datasets/test目录的子每个目录名代表一个分类标签，在它里面存放训练图片。






