# SAM 3.1视频分割和跟踪

## 一、程序参照

[sam3.1_video_predictor_example.ipynb](https://github.com/facebookresearch/sam3/blob/main/examples/sam3.1_video_predictor_example.ipynb)

## 二、建立虚拟环境

```shell
conda create -n sam3 python=3.12 -y
conda activate sam3
```

## 三、安装依赖库

```shell
# 安装PyTorch
pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
# 安装SAM3库
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
# 降级setuptools
pip install "setuptools<82"
# 安装flash-attn-3
pip install flash-attn-3 --no-deps --index-url https://download.pytorch.org/whl/cu128
# 安装其他依赖库
pip install \
einops ninja pycocotools psutil scikit-image \
opencv-python matplotlib scikit-learn pandas \
-i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 四、准备数据

```shell
只保留sam3/assets/videos/0001文件夹下的前30帧图像
```

## 五、安装ffmpeg

```
sudo apt install ffmpeg
```

## 六、修改模型装载程序

```shell
修改sam3/model_builder.py
将facebook/sam3.1 修改为 jetjodh/sam3.1
```

## 七、运行程序

```shell
export HF_ENDPOINT=https://hf-mirror.com
python 11_sam3_multiplex.py
```

