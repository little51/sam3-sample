# Meta SAM 3 样例

## 一、环境安装

### 1、Python虚拟环境

```shell
# 创建Python3.12虚拟环境
conda create -n sam3 python=3.12 -y
# 激活虚拟环境
conda activate sam3
```

### 2、克隆代码

```shell
# 克隆sam3源码，用于sam3相关库安装
git clone https://github.com/facebookresearch/sam3
# 切换到sam3目录
cd sam3
# 检出特定的版本
git checkout 84cc43b
```

### 3、安装依赖库

```shell
# 安装sam3依赖库
pip install -e . -i https://pypi.mirrors.ustc.edu.cn/simple
# 安装其他依赖库
pip install requests einops decord pycocotools psutil matplotlib \
opencv-python pandas scikit-image scikit-learn \
-i https://pypi.mirrors.ustc.edu.cn/simple
# 校验PyTorch是否正确安装
python -c "import torch; print(torch.cuda.is_available())"
# 如果为False，则重装PyTorch(一般出现在Windows上)
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
# Windows
pip install -U "triton-windows<3.6" -i https://pypi.mirrors.ustc.edu.cn/simple
# 切换回上一级目录
cd ..
```

## 二、模型下载

```shell
# 下载脚本
wget https://aliendao.cn/model_download2.py
# 下载权重
python model_download2.py --repo_id facebook/sam3 
```

## 三、运行例程

### 1、最简图片分割

```shell
python 01_image_segment.py
```

### 2、优化提示词

```shell
python 02_image_segment.py
```

### 3、视频分割（简单）

```shell
python 03_video_segment.py
```

### 4、视频分割（完整）

```shell
python 04_video_segment.py
```

### 5、MobileSAM

```shell
python 05_mobile_sam.py
```

### 6、FastSAM

```shell
python 06_fast_sam.py
```

### 7、SAM Agent

```shell
# 运行Qwen/Qwen3-VL-2B-Thinking模型
HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=1 \
vllm serve Qwen/Qwen3-VL-2B-Thinking \
--max-model-len 32000 --disable-log-stats --enforce-eager \
--host 0.0.0.0 --port 8000 --served-model-name Qwen/Qwen3-VL-2B-Thinking \
--dtype=half --gpu-memory-utilization 0.9
# 安装其他依赖库
pip install ipython openai -i https://pypi.mirrors.ustc.edu.cn/simple
# 运行程序
python 08_sam3_agent.py

```

## 四、SAM_3D_BODY

### 1、安装基础环境

```shell
# 创建虚拟环境
conda create -n sam_3d_body python=3.11 -y
# 激活虚拟环境
conda activate sam_3d_body
# 安装基础库
pip install pytorch-lightning pyrender opencv-python yacs scikit-image einops timm dill pandas rich hydra-core hydra-submitit-launcher hydra-colorlog pyrootutils webdataset chump networkx==3.2.1 roma joblib seaborn wandb appdirs appnope ffmpeg cython jsonlines pytest xtcocotools loguru optree fvcore black pycocotools tensorboard huggingface_hub -i https://pypi.mirrors.ustc.edu.cn/simple
# 安装PyTorch
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
```

### 2、安装Detectron2

```shell
pip install git+https://github.com/facebookresearch/detectron2.git@a1ce2f9 --no-build-isolation --no-deps
```

### 3、安装MoGe

```shell
pip install git+https://github.com/microsoft/MoGe.git
```

### 4、克隆Sam_3d_body源码

```shell
git clone https://github.com/facebookresearch/sam-3d-body
```

### 5、运行例程

```shell
CUDA_VISIBLE_DEVICES=0 HF_ENDPOINT=https://hf-mirror.com python 09_sam3_body.py
```



## 作者其他作品

《大模型项目实战：多领域智能应用开发》配套资源：[https://github.com/little51/llm-dev](https://github.com/little51/llm-dev)
《大模型项目实战：Agent开发与应用》配套资源：[https://github.com/little51/agent-dev](https://github.com/little51/agent-dev)
DINOv3训练示例：[https://github.com/little51/dinov3-train](https://github.com/little51/dinov3-train)
DINOv3其他示例：[https://github.com/little51/dinov3-samples](https://github.com/little51/dinov3-samples)

