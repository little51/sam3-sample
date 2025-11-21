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
-i https://pypi.mirrors.ustc.edu.cn/simple
# 校验PyTorch是否正确安装
python -c "import torch; print(torch.cuda.is_available())"
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

## 作者其他作品

《大模型项目实战：多领域智能应用开发》配套资源：[https://github.com/little51/llm-dev](https://github.com/little51/llm-dev)
《大模型项目实战：Agent开发与应用》配套资源：[https://github.com/little51/agent-dev](https://github.com/little51/agent-dev)
DINOv3训练示例：[https://github.com/little51/dinov3-train](https://github.com/little51/dinov3-train)
DINOv3其他示例：[https://github.com/little51/dinov3-samples](https://github.com/little51/dinov3-samples)

