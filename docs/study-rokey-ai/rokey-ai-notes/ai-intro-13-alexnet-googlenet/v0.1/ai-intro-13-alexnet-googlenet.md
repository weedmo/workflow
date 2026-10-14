# 13장 영상 분류 사전 학습 모델 활용하기 1

> **Migrated from Quartz wiki** (`wiki/study/rokey-ai/basic/ai-intro-13-alexnet_googlenet.md`, updated 2026-09-26)  
> tags: rokey, ai, notebook, basic, study  
> source: raw/2026-09-26-rokey-ai/basic/강의_3기_AI개론_13차시__AlexNet_GoogleNet_.ipynb  
> source: https://github.com/weedmo/Rokey-AI  

* "부록3 매트플롯립 입문"에서 한글 폰트를 올바르게 출력하기 위한 설치 방법을 설명했다. 설치 방법은 다음과 같다.


```python
!sudo apt-get install -y fonts-nanum* | tail -n 1
!sudo fc-cache -fv
!rm -rf ~/.cache/matplotlib
```

    debconf: unable to initialize frontend: Dialog
    debconf: (No usable dialog-like program is installed, so the dialog based frontend cannot be used. at /usr/share/perl5/Debconf/FrontEnd/Dialog.pm line 76, <> line 4.)
    debconf: falling back to frontend: Readline
    debconf: unable to initialize frontend: Readline
    debconf: (This frontend requires a controlling tty.)
    debconf: falling back to frontend: Teletype
    dpkg-preconfigure: unable to re-open stdin: 
    Processing triggers for fontconfig (2.12.6-0ubuntu2) ...
    /usr/share/fonts: caching, new cache contents: 0 fonts, 1 dirs
    /usr/share/fonts/truetype: caching, new cache contents: 0 fonts, 3 dirs
    /usr/share/fonts/truetype/humor-sans: caching, new cache contents: 1 fonts, 0 dirs
    /usr/share/fonts/truetype/liberation: caching, new cache contents: 16 fonts, 0 dirs
    /usr/share/fonts/truetype/nanum: caching, new cache contents: 31 fonts, 0 dirs
    /usr/local/share/fonts: caching, new cache contents: 0 fonts, 0 dirs
    /root/.local/share/fonts: skipping, no such directory
    /root/.fonts: skipping, no such directory
    /var/cache/fontconfig: cleaning cache directory
    /root/.cache/fontconfig: not cleaning non-existent cache directory
    /root/.fontconfig: not cleaning non-existent cache directory
    fc-cache: succeeded



```python
# 필요 라이브러리 설치

!pip install torchviz | tail -n 1
!pip install torchinfo | tail -n 1
```

    Successfully installed torchviz-0.0.2
    Successfully installed torchinfo-1.6.5


* 모든 설치가 끝나면 한글 폰트를 바르게 출력하기 위해 **[런타임]** -> **[런타임 다시시작]**을 클릭한 다음, 아래 셀부터 코드를 실행해 주십시오.


```python
# 라이브러리 임포트

%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display

# 폰트 관련 용도
import matplotlib.font_manager as fm

# 폰트 관련 용도
import matplotlib.font_manager as fm

# Colab, Linux
# 나눔 고딕 폰트의 경로 명시
path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
font_name = fm.FontProperties(fname=path, size=10).get_name()

# Window 
# font_name = "NanumBarunGothic"

# Mac
# font_name = "AppleGothic"
```


```python
# 파이토치 관련 라이브러리

import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
from torchviz import make_dot
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
```

    c:\Users\user\anaconda3\envs\torchgpu_py3.9\lib\site-packages\google\protobuf\runtime_version.py:112: UserWarning: Protobuf gencode version 5.27.5 is older than the runtime version 5.28.2 at onnx/onnx-ml.proto. Please avoid checked-in Protobuf gencode that can be obsolete.
      warnings.warn(
    c:\Users\user\anaconda3\envs\torchgpu_py3.9\lib\site-packages\google\protobuf\runtime_version.py:112: UserWarning: Protobuf gencode version 5.27.5 is older than the runtime version 5.28.2 at onnx/onnx-operators-ml.proto. Please avoid checked-in Protobuf gencode that can be obsolete.
      warnings.warn(
    c:\Users\user\anaconda3\envs\torchgpu_py3.9\lib\site-packages\google\protobuf\runtime_version.py:112: UserWarning: Protobuf gencode version 5.27.5 is older than the runtime version 5.28.2 at onnx/onnx-data.proto. Please avoid checked-in Protobuf gencode that can be obsolete.
      warnings.warn(



```python
# warning 표시 끄기
import warnings
warnings.simplefilter('ignore')

# 기본 폰트 설정
plt.rcParams['font.family'] = font_name  # window font

# 기본 폰트 사이즈 변경
plt.rcParams['font.size'] = 14

# 기본 그래프 사이즈 변경
plt.rcParams['figure.figsize'] = (6,6)

# 기본 그리드 표시
# 필요에 따라 설정할 때는, plt.grid()
plt.rcParams['axes.grid'] = True
plt.rcParams["grid.linestyle"] = ":"

# 마이너스 기호 정상 출력
plt.rcParams['axes.unicode_minus'] = False

# 넘파이 부동소수점 자릿수 표시
np.set_printoptions(suppress=True, precision=4)
```


```python
# GPU 디바이스 할당

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
```

    cuda:0


### 공통 함수 불러오기


```python
# 공통 함수 다운로드
!git clone https://github.com/wikibook/pythonlibs.git

# 공통 함수 불러오기
from pythonlibs.torch_lib1 import *
# from torch_lib1 import *


# 공통 함수 확인
print(README)
```

    Common Library for PyTorch
    Author: M. Akaishi


## 적응형 풀링 함수(nn.AdaptiveAvgPool2d 함수)


```python
# nn.AdaptiveAvgPool2d 정의
p = nn.AdaptiveAvgPool2d((1,1))
print(p)

# 선형 함수의 정의
l1 = nn.Linear(32, 10)
print(l1)
```

    AdaptiveAvgPool2d(output_size=(1, 1))
    Linear(in_features=32, out_features=10, bias=True)



```python
m = nn.AdaptiveAvgPool2d((5, 7))
input = torch.randn(1, 64, 8, 9)
print(m(input).shape)

input2 = torch.randn(1, 64, 32, 30)
print(m(input2).shape)
```

    torch.Size([1, 64, 5, 7])
    torch.Size([1, 64, 5, 7])



```python
# 사전 학습 모델 시뮬레이션
inputs = torch.randn(100, 32, 16, 16)
m1 = p(inputs)
m2 = m1.view(m1.shape[0],-1)
m3 = l1(m2)

# shape 확인
print(m1.shape)
print(m2.shape)
print(m3.shape)
```

    torch.Size([100, 32, 1, 1])
    torch.Size([100, 32])
    torch.Size([100, 10])


## 데이터 준비


```python
# 분류 클래스명 정의

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 분류 클래스 수는 10
n_output = len(classes)
```


```python
# Transforms 정의

# 학습 데이터용 : 정규화에 반전과 RandomErasing 추가
transform_train = transforms.Compose([
  transforms.Resize(112),
  transforms.RandomHorizontalFlip(p=0.5), 
  transforms.ToTensor(),
  transforms.Normalize(0.5, 0.5), 
  transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)
])

# 검증 데이터용 : 정규화만 실시
transform = transforms.Compose([
  transforms.Resize(112),
  transforms.ToTensor(),
  transforms.Normalize(0.5, 0.5)
])
```


```python
# 데이터 취득용 함수 dataset

data_root = './data'

train_set = datasets.CIFAR10(
    root = data_root, train = True,
    download = True, transform = transform_train)

# 검증 데이터셋
test_set = datasets.CIFAR10(
    root = data_root, train = False, 
    download = True, transform = transform)
```

    Files already downloaded and verified
    Files already downloaded and verified



```python
# 배치 사이즈 지정
batch_size = 50

# 데이터로더

# 훈련용 데이터로더
# 훈련용이므로 셔플을 True로 설정함
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

# 검증용 데이터로더
# 검증용은 셔플이 필요하지 않음
test_loader = DataLoader(test_set,  batch_size=batch_size, shuffle=False) 
```

## AlexNet 불러 오기

### 모델 불러오기


```python
#  라이브러리 임포트
from torchvision import models

dir(models)
```




    ['AlexNet',
     'AlexNet_Weights',
     'ConvNeXt',
     'ConvNeXt_Base_Weights',
     'ConvNeXt_Large_Weights',
     'ConvNeXt_Small_Weights',
     'ConvNeXt_Tiny_Weights',
     'DenseNet',
     'DenseNet121_Weights',
     'DenseNet161_Weights',
     'DenseNet169_Weights',
     'DenseNet201_Weights',
     'EfficientNet',
     'EfficientNet_B0_Weights',
     'EfficientNet_B1_Weights',
     'EfficientNet_B2_Weights',
     'EfficientNet_B3_Weights',
     'EfficientNet_B4_Weights',
     'EfficientNet_B5_Weights',
     'EfficientNet_B6_Weights',
     'EfficientNet_B7_Weights',
     'EfficientNet_V2_L_Weights',
     'EfficientNet_V2_M_Weights',
     'EfficientNet_V2_S_Weights',
     'GoogLeNet',
     'GoogLeNetOutputs',
     'GoogLeNet_Weights',
     'Inception3',
     'InceptionOutputs',
     'Inception_V3_Weights',
     'MNASNet',
     'MNASNet0_5_Weights',
     'MNASNet0_75_Weights',
     'MNASNet1_0_Weights',
     'MNASNet1_3_Weights',
     'MaxVit',
     'MaxVit_T_Weights',
     'MobileNetV2',
     'MobileNetV3',
     'MobileNet_V2_Weights',
     'MobileNet_V3_Large_Weights',
     'MobileNet_V3_Small_Weights',
     'RegNet',
     'RegNet_X_16GF_Weights',
     'RegNet_X_1_6GF_Weights',
     'RegNet_X_32GF_Weights',
     'RegNet_X_3_2GF_Weights',
     'RegNet_X_400MF_Weights',
     'RegNet_X_800MF_Weights',
     'RegNet_X_8GF_Weights',
     'RegNet_Y_128GF_Weights',
     'RegNet_Y_16GF_Weights',
     'RegNet_Y_1_6GF_Weights',
     'RegNet_Y_32GF_Weights',
     'RegNet_Y_3_2GF_Weights',
     'RegNet_Y_400MF_Weights',
     'RegNet_Y_800MF_Weights',
     'RegNet_Y_8GF_Weights',
     'ResNeXt101_32X8D_Weights',
     'ResNeXt101_64X4D_Weights',
     'ResNeXt50_32X4D_Weights',
     'ResNet',
     'ResNet101_Weights',
     'ResNet152_Weights',
     'ResNet18_Weights',
     'ResNet34_Weights',
     'ResNet50_Weights',
     'ShuffleNetV2',
     'ShuffleNet_V2_X0_5_Weights',
     'ShuffleNet_V2_X1_0_Weights',
     'ShuffleNet_V2_X1_5_Weights',
     'ShuffleNet_V2_X2_0_Weights',
     'SqueezeNet',
     'SqueezeNet1_0_Weights',
     'SqueezeNet1_1_Weights',
     'SwinTransformer',
     'Swin_B_Weights',
     'Swin_S_Weights',
     'Swin_T_Weights',
     'Swin_V2_B_Weights',
     'Swin_V2_S_Weights',
     'Swin_V2_T_Weights',
     'VGG',
     'VGG11_BN_Weights',
     'VGG11_Weights',
     'VGG13_BN_Weights',
     'VGG13_Weights',
     'VGG16_BN_Weights',
     'VGG16_Weights',
     'VGG19_BN_Weights',
     'VGG19_Weights',
     'ViT_B_16_Weights',
     'ViT_B_32_Weights',
     'ViT_H_14_Weights',
     'ViT_L_16_Weights',
     'ViT_L_32_Weights',
     'VisionTransformer',
     'Weights',
     'WeightsEnum',
     'Wide_ResNet101_2_Weights',
     'Wide_ResNet50_2_Weights',
     '_GoogLeNetOutputs',
     '_InceptionOutputs',
     '__builtins__',
     '__cached__',
     '__doc__',
     '__file__',
     '__loader__',
     '__name__',
     '__package__',
     '__path__',
     '__spec__',
     '_api',
     '_meta',
     '_utils',
     'alexnet',
     'convnext',
     'convnext_base',
     'convnext_large',
     'convnext_small',
     'convnext_tiny',
     'densenet',
     'densenet121',
     'densenet161',
     'densenet169',
     'densenet201',
     'detection',
     'efficientnet',
     'efficientnet_b0',
     'efficientnet_b1',
     'efficientnet_b2',
     'efficientnet_b3',
     'efficientnet_b4',
     'efficientnet_b5',
     'efficientnet_b6',
     'efficientnet_b7',
     'efficientnet_v2_l',
     'efficientnet_v2_m',
     'efficientnet_v2_s',
     'get_model',
     'get_model_builder',
     'get_model_weights',
     'get_weight',
     'googlenet',
     'inception',
     'inception_v3',
     'list_models',
     'maxvit',
     'maxvit_t',
     'mnasnet',
     'mnasnet0_5',
     'mnasnet0_75',
     'mnasnet1_0',
     'mnasnet1_3',
     'mobilenet',
     'mobilenet_v2',
     'mobilenet_v3_large',
     'mobilenet_v3_small',
     'mobilenetv2',
     'mobilenetv3',
     'optical_flow',
     'quantization',
     'regnet',
     'regnet_x_16gf',
     'regnet_x_1_6gf',
     'regnet_x_32gf',
     'regnet_x_3_2gf',
     'regnet_x_400mf',
     'regnet_x_800mf',
     'regnet_x_8gf',
     'regnet_y_128gf',
     'regnet_y_16gf',
     'regnet_y_1_6gf',
     'regnet_y_32gf',
     'regnet_y_3_2gf',
     'regnet_y_400mf',
     'regnet_y_800mf',
     'regnet_y_8gf',
     'resnet',
     'resnet101',
     'resnet152',
     'resnet18',
     'resnet34',
     'resnet50',
     'resnext101_32x8d',
     'resnext101_64x4d',
     'resnext50_32x4d',
     'segmentation',
     'shufflenet_v2_x0_5',
     'shufflenet_v2_x1_0',
     'shufflenet_v2_x1_5',
     'shufflenet_v2_x2_0',
     'shufflenetv2',
     'squeezenet',
     'squeezenet1_0',
     'squeezenet1_1',
     'swin_b',
     'swin_s',
     'swin_t',
     'swin_transformer',
     'swin_v2_b',
     'swin_v2_s',
     'swin_v2_t',
     'vgg',
     'vgg11',
     'vgg11_bn',
     'vgg13',
     'vgg13_bn',
     'vgg16',
     'vgg16_bn',
     'vgg19',
     'vgg19_bn',
     'video',
     'vision_transformer',
     'vit_b_16',
     'vit_b_32',
     'vit_h_14',
     'vit_l_16',
     'vit_l_32',
     'wide_resnet101_2',
     'wide_resnet50_2']




```python
# net = models.alexnet(pretrained = True)

weights = models.AlexNet_Weights.IMAGENET1K_V1
net = models.alexnet(weights = weights)

# weights.transforms()
```


```python
print(net)
```

    AlexNet(
      (features): Sequential(
        (0): Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
        (1): ReLU(inplace=True)
        (2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        (3): Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        (4): ReLU(inplace=True)
        (5): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        (6): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (7): ReLU(inplace=True)
        (8): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (9): ReLU(inplace=True)
        (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (11): ReLU(inplace=True)
        (12): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
      (avgpool): AdaptiveAvgPool2d(output_size=(6, 6))
      (classifier): Sequential(
        (0): Dropout(p=0.5, inplace=False)
        (1): Linear(in_features=9216, out_features=4096, bias=True)
        (2): ReLU(inplace=True)
        (3): Dropout(p=0.5, inplace=False)
        (4): Linear(in_features=4096, out_features=4096, bias=True)
        (5): ReLU(inplace=True)
        (6): Linear(in_features=4096, out_features=1000, bias=True)
      )
    )



```python
# 모델 개요 표시 2
# net = net.to(device)
summary(net,(100, 3, 112, 112))
```




    ==========================================================================================
    Layer (type:depth-idx)                   Output Shape              Param #
    ==========================================================================================
    AlexNet                                  [100, 1000]               --
    ├─Sequential: 1-1                        [100, 256, 2, 2]          --
    │    └─Conv2d: 2-1                       [100, 64, 27, 27]         23,296
    │    └─ReLU: 2-2                         [100, 64, 27, 27]         --
    │    └─MaxPool2d: 2-3                    [100, 64, 13, 13]         --
    │    └─Conv2d: 2-4                       [100, 192, 13, 13]        307,392
    │    └─ReLU: 2-5                         [100, 192, 13, 13]        --
    │    └─MaxPool2d: 2-6                    [100, 192, 6, 6]          --
    │    └─Conv2d: 2-7                       [100, 384, 6, 6]          663,936
    │    └─ReLU: 2-8                         [100, 384, 6, 6]          --
    │    └─Conv2d: 2-9                       [100, 256, 6, 6]          884,992
    │    └─ReLU: 2-10                        [100, 256, 6, 6]          --
    │    └─Conv2d: 2-11                      [100, 256, 6, 6]          590,080
    │    └─ReLU: 2-12                        [100, 256, 6, 6]          --
    │    └─MaxPool2d: 2-13                   [100, 256, 2, 2]          --
    ├─AdaptiveAvgPool2d: 1-2                 [100, 256, 6, 6]          --
    ├─Sequential: 1-3                        [100, 1000]               --
    │    └─Dropout: 2-14                     [100, 9216]               --
    │    └─Linear: 2-15                      [100, 4096]               37,752,832
    │    └─ReLU: 2-16                        [100, 4096]               --
    │    └─Dropout: 2-17                     [100, 4096]               --
    │    └─Linear: 2-18                      [100, 4096]               16,781,312
    │    └─ReLU: 2-19                        [100, 4096]               --
    │    └─Linear: 2-20                      [100, 1000]               4,097,000
    ==========================================================================================
    Total params: 61,100,840
    Trainable params: 61,100,840
    Non-trainable params: 0
    Total mult-adds (G): 20.46
    ==========================================================================================
    Input size (MB): 15.05
    Forward/backward pass size (MB): 96.44
    Params size (MB): 244.40
    Estimated Total Size (MB): 355.90
    ==========================================================================================




```python
## Access to the layers

print(net.classifier)
print(net.classifier[6])
print(net.classifier[6].in_features)
print(net.classifier[6].out_features)
print(net.classifier[6].bias)

```

    Sequential(
      (0): Dropout(p=0.5, inplace=False)
      (1): Linear(in_features=9216, out_features=4096, bias=True)
      (2): ReLU(inplace=True)
      (3): Dropout(p=0.5, inplace=False)
      (4): Linear(in_features=4096, out_features=4096, bias=True)
      (5): ReLU(inplace=True)
      (6): Linear(in_features=4096, out_features=1000, bias=True)
    )
    Linear(in_features=4096, out_features=1000, bias=True)
    4096
    1000
    Parameter containing:
    tensor([ 5.3252e-02,  5.6475e-02,  1.2015e-02,  1.0475e-02,  1.4073e-02,
             2.4921e-02,  4.5943e-02, -1.2418e-02, -5.2491e-02, -1.5580e-02,
            -2.1215e-02, -3.3407e-02,  9.5835e-03,  1.8659e-02,  7.1095e-03,
            -2.5249e-02, -2.9553e-03,  6.2285e-03, -3.0338e-02,  1.7713e-02,
             4.8128e-02,  5.5310e-02,  4.2137e-02, -3.4339e-02,  1.1161e-02,
            -3.7005e-02, -3.7998e-02, -2.2497e-02, -1.3564e-02,  1.1125e-01,
             3.1010e-02,  6.8569e-03, -1.5973e-02, -9.2437e-03,  4.3681e-02,
            -2.6168e-02, -3.0454e-03,  2.6335e-02,  1.0302e-02,  2.9396e-02,
            -1.6149e-02,  3.0833e-02,  4.0436e-02,  6.6803e-02,  2.4527e-02,
             4.6312e-02,  6.1914e-03,  8.0594e-02,  5.9732e-02,  6.1413e-02,
             1.6579e-03,  6.7179e-02, -4.2294e-03, -1.4659e-02, -6.7676e-02,
            -8.3818e-03, -5.8036e-02,  8.1914e-03,  3.9684e-02,  2.8477e-02,
            -1.2424e-01,  3.9262e-02,  9.1787e-03,  6.8728e-02,  4.0663e-02,
            -1.0124e-02,  1.2239e-02, -2.7275e-03, -2.1134e-02,  9.3186e-02,
             4.6140e-03,  2.6338e-02,  4.4615e-02, -1.2071e-02, -3.0606e-02,
             6.9681e-02,  4.3573e-02, -7.0400e-03,  3.4302e-02,  1.8671e-02,
            -2.3980e-03, -7.9588e-03, -2.6701e-02, -2.3112e-02, -2.1024e-02,
             7.5927e-03, -4.0854e-02,  9.6504e-02,  1.6273e-02,  6.8265e-02,
            -1.2029e-02,  1.8616e-02, -2.3254e-02,  6.6254e-04,  6.5770e-02,
             2.0797e-02,  4.6046e-02, -1.2563e-02,  1.5837e-02, -6.2019e-02,
             1.6890e-02,  2.9346e-02,  1.2199e-02,  1.1579e-01,  1.8052e-02,
             8.3501e-02,  4.6795e-02, -5.9661e-03,  4.3978e-02, -8.9776e-02,
            -8.6210e-02,  4.8310e-02, -2.1315e-02,  7.1201e-03, -3.1428e-02,
            -2.2256e-02,  9.2478e-02,  5.7419e-02, -2.9094e-04, -1.5966e-02,
             9.0139e-02, -1.8068e-02,  5.2080e-02, -1.0922e-02, -5.9916e-02,
             6.9528e-02, -4.0415e-03, -2.4078e-02,  1.2984e-02,  8.9963e-03,
            -3.2033e-02,  1.7807e-03,  3.9556e-02, -7.1310e-03, -8.6408e-02,
            -5.2836e-02, -3.0279e-02, -2.9701e-03,  2.8985e-02, -6.0586e-03,
            -1.5632e-02, -1.5263e-02, -1.4647e-02, -6.3525e-02, -3.9613e-02,
            -7.7837e-03, -7.8425e-03,  7.1100e-04, -7.7680e-04,  4.9023e-02,
            -2.2289e-03,  6.4397e-03, -9.0882e-02,  4.1336e-02, -6.3460e-03,
             1.4306e-02,  3.8303e-03, -3.1278e-02, -7.2626e-02, -3.5031e-02,
            -2.1359e-02,  6.8324e-02,  4.5042e-02,  2.8514e-02,  3.2959e-02,
            -4.6693e-02, -1.0623e-02, -6.0456e-02,  2.3472e-02,  1.9999e-02,
             2.5433e-02,  7.6092e-02, -6.2514e-03, -2.5429e-02,  6.6237e-02,
            -5.7913e-02, -1.8797e-02,  4.2716e-02,  5.1821e-02, -6.9029e-02,
            -1.5346e-02,  2.3568e-02,  5.2703e-02,  5.5551e-02, -8.4720e-03,
            -1.2149e-02, -2.7647e-03, -1.4819e-03, -2.3124e-02,  8.9908e-03,
            -3.7236e-03,  5.5263e-02,  3.0676e-02, -3.2228e-02, -6.9401e-03,
             5.0185e-02,  2.8821e-02,  3.6680e-03, -2.8823e-02,  5.0426e-02,
            -1.0344e-01, -2.1276e-03,  4.3640e-02,  3.0670e-02,  1.1708e-02,
            -8.2692e-03, -8.7994e-04, -7.9968e-03, -5.8294e-02,  7.0524e-02,
            -2.0286e-02,  2.0245e-03, -5.0412e-03, -1.7596e-02,  2.1311e-03,
            -1.2316e-02,  1.5342e-02,  3.2776e-02,  5.1759e-03, -4.1486e-02,
            -7.3661e-03, -1.9380e-02,  3.3047e-02,  8.3802e-02, -1.9553e-02,
             7.2874e-02, -3.5580e-03, -8.1869e-02,  2.6474e-02,  4.9446e-02,
             2.9175e-02, -7.6044e-02, -2.3432e-02,  2.2784e-02,  1.0188e-02,
             1.0420e-02,  5.3774e-03,  5.4046e-02,  1.4067e-02,  4.0287e-02,
            -4.9321e-02, -1.4429e-02, -4.6192e-02, -1.4586e-02, -2.6139e-02,
            -4.2562e-04, -5.0145e-02,  3.3987e-02, -5.3159e-02, -5.5430e-02,
             3.6954e-03, -1.1041e-03,  2.8349e-02, -4.4211e-02,  5.9482e-02,
            -1.5721e-02, -2.6858e-02,  2.9261e-02,  9.5011e-03,  2.4154e-03,
             1.3513e-02,  2.8245e-02, -6.4663e-02,  5.3230e-02, -4.3924e-02,
             3.4698e-04,  1.7564e-02, -9.1725e-02, -2.3233e-02,  2.2276e-02,
             4.0636e-02,  5.2172e-02,  3.5888e-02,  6.8130e-03,  1.0692e-02,
             2.1173e-03, -1.7580e-03, -2.7247e-03, -4.8340e-02,  1.3375e-02,
            -2.6605e-02,  8.5712e-02, -7.3576e-02,  2.3194e-02,  5.6535e-02,
             3.6531e-02,  7.1076e-02, -2.0128e-02, -5.6684e-02,  4.1176e-02,
            -1.7057e-02,  1.3072e-03,  1.3561e-02, -8.1499e-02, -1.9378e-02,
             3.5581e-02,  3.2518e-02,  6.2056e-02, -3.8972e-02,  4.0444e-02,
            -3.7356e-02, -2.5330e-02, -4.5012e-02, -5.6890e-02, -6.5692e-02,
             5.4484e-02,  4.5054e-02, -7.4308e-02,  1.1787e-03,  5.3284e-04,
            -6.7275e-02, -5.6026e-02, -7.0426e-02,  6.4871e-02,  4.1639e-02,
             8.3475e-02,  3.5982e-02,  2.0956e-02,  2.5103e-02, -1.6456e-02,
            -1.1024e-02, -2.6935e-02, -9.1414e-03, -5.0469e-02,  5.2238e-02,
            -2.6817e-02,  1.4414e-02, -1.0621e-01, -4.6598e-02, -2.5114e-02,
             8.4723e-03, -2.5711e-02,  9.3712e-02,  7.2801e-02,  2.5253e-02,
             2.1812e-02,  3.3336e-02,  1.6326e-02, -5.2736e-02,  5.5630e-02,
            -1.3830e-02, -4.0054e-02,  8.7340e-03,  3.1333e-02,  3.4241e-02,
            -5.1627e-02,  2.7252e-02,  2.6041e-02, -5.2250e-02,  2.6162e-02,
             5.1007e-02,  2.8195e-02, -2.4470e-02,  1.2946e-02,  5.2765e-02,
            -1.8762e-02, -3.1476e-02,  5.5163e-04,  1.0595e-02,  6.4026e-02,
            -1.0145e-02,  6.1711e-02,  2.9520e-02,  3.8700e-02,  4.4010e-02,
             2.8614e-02,  5.7040e-02,  4.6143e-02, -3.3167e-02,  2.5789e-02,
            -9.9446e-03, -3.0128e-03,  9.8479e-03,  3.3981e-02, -1.5649e-02,
            -9.9509e-03,  4.9874e-02, -6.3548e-04,  2.8995e-02,  7.9144e-03,
            -6.2499e-02, -5.1307e-02,  2.6480e-02,  1.4117e-02, -2.4593e-02,
             9.7300e-03, -1.0901e-02,  2.7393e-02,  1.5526e-02,  4.2757e-02,
            -4.2605e-02,  1.3845e-02, -1.6240e-02,  4.3815e-02, -2.3015e-03,
            -2.2745e-03,  2.7163e-02,  3.5608e-02, -3.9027e-02,  7.4795e-02,
             2.9545e-03,  2.4383e-02,  3.8495e-03,  7.3041e-03, -3.5850e-02,
             9.0172e-02, -1.9558e-03, -9.6829e-02, -6.6019e-02, -1.2339e-01,
             8.5293e-02, -2.8016e-02, -4.2111e-02,  3.4543e-03, -5.9704e-03,
            -4.0699e-02,  9.3167e-02,  3.8482e-03, -4.1334e-03,  9.7206e-03,
             1.7187e-02, -1.8781e-02, -2.0588e-02,  6.4882e-02,  6.1634e-02,
            -4.5338e-05, -4.7090e-02, -1.3213e-01,  2.8466e-02, -2.8057e-02,
             5.8503e-02,  6.6895e-02, -3.4372e-02, -1.4239e-02, -3.0599e-02,
             1.9456e-02, -3.3238e-02, -2.4988e-02, -9.0367e-05, -4.6692e-02,
            -4.8098e-02,  1.9271e-02,  2.4073e-02,  2.2539e-02, -5.8785e-03,
             1.5558e-02,  4.0886e-03, -7.8306e-02,  8.6316e-02, -1.4157e-02,
             8.7703e-02,  1.1080e-02,  2.4186e-02,  8.9802e-04, -1.2056e-02,
            -1.7418e-02, -3.5627e-03, -3.2366e-02, -1.3965e-03, -2.6253e-02,
            -2.4457e-02,  1.6563e-02, -1.8416e-02, -1.0767e-01,  9.6398e-03,
             4.2801e-02,  6.0262e-02,  3.9423e-02, -7.1208e-02,  3.1756e-02,
            -5.8451e-02, -4.1126e-02, -3.6470e-02,  3.2047e-02,  1.0938e-02,
             1.5454e-01,  3.8895e-02,  4.0750e-02,  2.8544e-02, -8.7241e-02,
             4.4254e-02, -5.8567e-03, -2.4539e-02, -3.7177e-02, -6.1798e-02,
             2.9119e-03, -1.5438e-02, -6.9551e-02, -1.3111e-01,  2.5559e-02,
             1.5085e-02,  7.0103e-02,  3.3266e-02, -2.6814e-02, -1.1635e-01,
            -1.3400e-02,  1.0656e-01, -1.6285e-01,  3.3475e-02, -3.2177e-02,
             4.8456e-02, -1.1730e-02, -8.8067e-02, -3.5880e-02,  1.3474e-02,
            -2.0326e-02, -1.2884e-01, -5.6742e-02, -6.5963e-02,  1.2026e-02,
            -2.5221e-02, -2.3785e-02, -9.6762e-03, -3.7816e-02,  1.9221e-02,
             4.8619e-03, -2.4410e-03, -2.6034e-02, -1.9117e-02, -8.2225e-04,
             1.7868e-02, -2.7427e-02,  4.1341e-02,  2.4172e-02,  6.8962e-02,
             6.3656e-02,  4.3324e-02, -1.6802e-02, -1.7103e-02,  3.2263e-02,
            -4.4776e-02, -8.3217e-02, -1.8283e-02,  5.8367e-02,  3.1406e-02,
             5.6282e-02, -1.1132e-01,  7.2988e-02, -1.0903e-01,  2.9206e-02,
            -2.7821e-02, -1.2398e-01, -2.5645e-02, -5.7258e-02,  8.1258e-03,
             2.6332e-02, -2.0495e-02, -4.6250e-02,  2.8908e-03,  9.5556e-02,
             4.4201e-02, -2.7812e-03,  2.2221e-03, -4.5316e-02, -4.3130e-02,
            -5.8415e-02,  3.2564e-02,  5.7614e-02, -7.8569e-02, -6.7936e-02,
            -6.6392e-03,  4.6499e-02, -5.6938e-02,  6.3510e-02,  6.6341e-02,
             1.3054e-02, -1.0774e-02, -5.5007e-02,  4.9877e-02,  2.0793e-02,
             1.5054e-02, -1.7921e-02, -6.6430e-02,  5.9132e-02,  2.1106e-02,
             1.8961e-02, -1.0129e-02,  1.8008e-02, -3.5435e-02,  1.4764e-02,
            -7.5889e-03, -8.3661e-02, -5.2211e-02,  6.8491e-02, -2.9039e-02,
            -1.9383e-02,  1.5508e-02, -1.8306e-02, -3.6809e-03,  5.0420e-02,
            -5.5348e-02,  5.0071e-03,  3.2704e-03, -5.4693e-03,  8.3264e-02,
            -2.6980e-02, -3.8524e-02,  7.7673e-02,  3.8679e-02, -4.3476e-02,
            -6.3778e-02,  7.1726e-03,  3.6365e-02,  3.5581e-02,  1.8565e-02,
            -1.5428e-02,  3.9404e-02,  1.0108e-02,  9.3341e-03, -4.7146e-02,
             3.6313e-02,  1.3648e-03,  4.9428e-02,  1.1902e-02, -6.4542e-03,
            -4.9254e-02, -1.1963e-01,  9.9042e-02, -2.6934e-02, -7.8272e-02,
             5.6361e-03,  1.2645e-02, -3.9618e-03,  2.2964e-02, -3.2064e-02,
            -4.6960e-02, -3.3381e-02,  2.9637e-02, -3.6173e-02,  3.0285e-03,
             1.7763e-02, -2.4117e-02, -5.6028e-02,  3.5110e-02,  7.2502e-02,
            -5.6726e-02, -6.1277e-02, -6.2330e-02, -6.4275e-02, -2.1981e-02,
            -1.6378e-02, -7.2263e-02,  3.9917e-02, -9.7765e-02, -4.7685e-02,
            -1.9302e-03, -2.4836e-02,  2.4878e-03,  5.6834e-02,  1.0820e-02,
            -3.2613e-02,  2.7537e-02,  5.3602e-03,  1.1122e-02, -3.4763e-02,
             3.4889e-02,  1.1450e-02, -3.3565e-02,  1.8182e-02, -2.5285e-02,
             7.6259e-02, -1.7923e-02,  8.6244e-03, -5.6801e-02, -2.0029e-02,
            -7.3464e-03,  7.8287e-03, -2.6822e-02,  4.8273e-03,  9.6238e-02,
             1.8904e-02,  3.5164e-02,  7.9271e-02, -1.4469e-02,  8.4794e-02,
             2.3145e-02, -4.8635e-02, -1.8980e-02,  1.3211e-02, -1.1367e-02,
             5.6345e-02, -3.2129e-02,  9.4927e-03, -4.3672e-02, -5.8385e-02,
             2.4477e-02, -4.1123e-02, -1.7410e-02, -2.2212e-02, -4.1425e-02,
             4.2289e-02,  8.7909e-02,  2.7870e-02, -7.8393e-02,  6.7135e-04,
             3.5587e-03, -2.1712e-02,  2.3001e-02, -8.1032e-02,  7.3319e-03,
            -3.3296e-04,  1.6864e-02,  1.7028e-02,  4.5052e-02,  2.4304e-03,
            -7.4777e-02,  6.0089e-02,  6.9517e-02, -5.7710e-02,  4.6902e-03,
            -3.8819e-03, -5.8210e-02, -1.1185e-03,  9.0919e-02, -8.7339e-03,
             6.4967e-02,  1.8759e-02, -5.0487e-02, -4.9778e-02,  4.9940e-02,
             2.3460e-02, -1.4869e-02,  7.7824e-03, -2.4576e-02, -4.5487e-02,
            -2.6208e-02,  1.3017e-01,  1.6476e-02, -3.0707e-02, -2.2029e-02,
            -2.6967e-02,  5.2982e-03,  1.7465e-02, -9.5463e-02, -8.5460e-02,
            -2.0266e-03, -2.9333e-03, -6.4850e-02,  7.8749e-02,  1.2722e-01,
            -3.0474e-02,  7.9202e-03, -1.4629e-02,  4.9757e-02, -6.1835e-02,
             4.2074e-02,  7.3102e-03,  4.7387e-02, -1.0757e-02,  6.5734e-02,
            -4.5547e-03,  2.7735e-03,  2.9592e-02,  5.8648e-03, -1.2238e-01,
             5.7966e-02,  6.0513e-02, -1.6859e-02, -4.4747e-02, -3.4610e-02,
             4.5194e-02,  6.8550e-04, -3.0971e-02,  6.2202e-02, -3.6581e-02,
            -2.7143e-02,  1.4357e-02, -2.3183e-03, -1.3557e-03, -2.3419e-02,
             7.2945e-02, -1.6167e-02, -6.4322e-02, -6.6394e-02,  8.7976e-03,
             5.5808e-03,  3.4428e-02,  4.3121e-02, -9.2526e-02, -6.9069e-02,
             7.7242e-03,  5.1836e-03, -5.7449e-02, -2.2806e-02, -4.3065e-02,
             1.4340e-01,  3.4542e-02, -3.3381e-02,  6.8073e-02, -4.3122e-02,
            -4.7611e-02,  1.2633e-02,  5.0245e-03,  5.3107e-02,  6.6606e-02,
             7.7941e-04,  2.1567e-02,  2.8193e-02, -6.4781e-03,  4.6802e-02,
            -8.1779e-02,  4.5102e-02,  4.3365e-02,  5.0884e-02,  5.0874e-03,
            -4.0991e-02, -5.6369e-02,  4.3932e-02, -1.1832e-02, -3.3235e-02,
            -7.4081e-02, -3.1906e-02,  1.6910e-02, -3.5907e-02,  1.9498e-03,
             3.7387e-03,  7.7815e-02, -3.8847e-02, -5.2373e-02,  3.9722e-02,
            -1.2653e-02, -4.8730e-02,  2.3529e-02,  1.2015e-02, -2.3584e-02,
            -4.2143e-03, -7.1829e-02, -8.9830e-02, -1.8455e-02, -5.9362e-02,
             1.7717e-02,  5.4384e-02,  3.6537e-03,  5.3803e-03,  7.9031e-02,
             2.2240e-02, -1.0549e-02, -4.5449e-03, -2.8834e-02, -1.0402e-02,
             2.6980e-03, -4.7863e-02, -7.3498e-04,  8.8465e-02, -2.0006e-02,
            -1.0358e-02, -1.3307e-02,  2.0518e-02,  5.8219e-03, -4.0321e-02,
            -8.3064e-03, -4.7484e-02, -7.1105e-02,  2.8095e-02,  7.1692e-03,
             5.3178e-02, -7.2905e-03, -2.0805e-02, -6.9260e-02,  4.5134e-02,
            -1.2814e-02,  1.2746e-02, -4.9280e-03,  2.4691e-02, -3.4422e-02,
             6.3144e-02, -2.1781e-02, -5.0597e-02, -7.5548e-02, -3.2391e-02,
             1.2470e-02, -6.6609e-02, -3.3134e-02, -2.6591e-02, -4.1465e-02,
            -1.8827e-02, -3.0473e-04,  1.5325e-02, -4.7332e-02, -5.5676e-02,
             2.1460e-02,  1.9186e-02,  5.3556e-03, -3.1528e-02,  9.7987e-03,
            -5.7906e-02, -3.7041e-02,  2.0125e-02, -5.3023e-03,  3.0509e-03,
             3.0903e-02, -1.9810e-02, -2.5124e-02,  2.5123e-02,  2.1905e-02,
             1.6592e-03,  8.0100e-03,  2.1628e-02, -4.9679e-02, -6.8297e-02,
             2.9881e-03,  1.1875e-02, -6.6792e-02,  1.3855e-02,  6.1322e-02,
             7.8280e-02,  4.3107e-02, -4.0548e-02,  1.3512e-02,  3.3229e-02,
            -5.1434e-02, -7.5863e-02, -3.1879e-02, -1.8831e-02, -5.0711e-03,
             4.9725e-02,  8.4448e-03,  3.9326e-02,  7.1417e-02,  4.9369e-02,
            -2.7340e-02,  7.9479e-02,  1.8443e-04,  3.4903e-02,  2.6848e-02,
             2.9325e-02,  2.4565e-02,  1.3714e-02,  1.0439e-02,  8.2166e-02,
             2.2898e-02, -4.9901e-02, -1.2849e-01,  4.4965e-02,  5.4320e-02,
             3.0903e-02,  2.7644e-02, -5.0354e-02, -2.5691e-02, -6.2493e-03,
             2.7136e-02,  1.1583e-02,  1.8871e-02, -3.5744e-02, -6.0619e-02,
            -1.2422e-02, -1.4326e-02, -9.8677e-02, -3.8423e-02, -3.8647e-02,
            -9.1581e-02, -4.2368e-02, -4.9885e-02, -1.6033e-02, -4.5562e-02,
             2.4515e-02, -2.1699e-02,  3.7827e-03, -3.4757e-02, -4.1276e-02,
             3.3561e-02,  5.7945e-02,  6.3927e-02,  7.1584e-03,  2.8452e-02,
             1.1123e-01, -2.2850e-02,  1.3239e-02, -8.6398e-02,  4.5526e-02,
            -2.9062e-03,  6.4437e-02,  2.3639e-02, -6.8218e-02,  3.5062e-02,
            -1.6846e-02,  2.8718e-02,  2.8398e-02, -9.9861e-04, -4.5618e-03,
             3.5558e-02,  4.4268e-02,  7.9080e-02,  1.6179e-02, -5.6045e-03,
            -3.0647e-02,  2.7647e-02, -1.0381e-01, -3.2340e-02, -8.2798e-03,
            -1.2683e-02, -6.8346e-02, -8.5445e-03, -1.1209e-02,  3.1321e-02,
            -1.0558e-02, -2.0959e-02,  3.0059e-02, -5.2112e-02,  2.3731e-02],
           device='cuda:0', requires_grad=True)


## 최종 레이어 함수 교체하기


```python
# 난수 고정
torch_seed()

# 최종 레이어 함수 교체
in_features = net.classifier[6].in_features
net.classifier[6] = nn.Linear(in_features, n_output)

# features 마지막의 MaxPool2d 제거
# net.features = net.features[:-1]

# AdaptiveAvgPool2d 제거
# net.avgpool = nn.Identity()
```


```python
# 모델 개요 표시 1
print(net)
```

    AlexNet(
      (features): Sequential(
        (0): Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
        (1): ReLU(inplace=True)
        (2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        (3): Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        (4): ReLU(inplace=True)
        (5): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        (6): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (7): ReLU(inplace=True)
        (8): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (9): ReLU(inplace=True)
        (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (11): ReLU(inplace=True)
        (12): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
      (avgpool): AdaptiveAvgPool2d(output_size=(6, 6))
      (classifier): Sequential(
        (0): Dropout(p=0.5, inplace=False)
        (1): Linear(in_features=9216, out_features=4096, bias=True)
        (2): ReLU(inplace=True)
        (3): Dropout(p=0.5, inplace=False)
        (4): Linear(in_features=4096, out_features=4096, bias=True)
        (5): ReLU(inplace=True)
        (6): Linear(in_features=4096, out_features=10, bias=True)
      )
    )



```python
# 손실 계산 그래프 시각화
net = net.to(device)
criterion = nn.CrossEntropyLoss()
loss = eval_loss(test_loader, device, net, criterion)
g = make_dot(loss, params=dict(net.named_parameters()))
display(g)
```


    
*(이미지 없음 / image not preserved: output_30_0.svg)*
    


## 학습과 결과 평가

### 초기 설정


```python
# 난수 고정
torch_seed()

# 사전 학습 모델 불러오기
# pretraind = True로 학습을 마친 파라미터도 함께 불러오기
weights = models.AlexNet_Weights.IMAGENET1K_V1
net = models.alexnet(weights = weights)

# 최종 레이어 함수 입력 차원수 확인
in_features = net.classifier[6].in_features
net.classifier[6] = nn.Linear(in_features, n_output)

# 최종 레이어 함수 교체
net.fc = nn.Linear(in_features, n_output)

# GPU 사용
net = net.to(device)

# 학습률
lr = 0.001

# 손실 함수 정의
criterion = nn.CrossEntropyLoss()

# 최적화 함수 정의
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

# history 파일 초기화
history = np.zeros((0, 5))
```

### 학습


```python
# 학습
num_epochs = 5
history = fit(net, optimizer, criterion, num_epochs, 
        train_loader, test_loader, device, history)
```


      0%|          | 0/1000 [00:00<?, ?it/s]


    Epoch [1/5], loss: 0.88723 acc: 0.68882 val_loss: 0.52400, val_acc: 0.81490



      0%|          | 0/1000 [00:00<?, ?it/s]


    Epoch [2/5], loss: 0.64617 acc: 0.77202 val_loss: 0.46861, val_acc: 0.83910



      0%|          | 0/1000 [00:00<?, ?it/s]


    Epoch [3/5], loss: 0.56864 acc: 0.80250 val_loss: 0.41946, val_acc: 0.85550



      0%|          | 0/1000 [00:00<?, ?it/s]


    Epoch [4/5], loss: 0.51698 acc: 0.81954 val_loss: 0.38899, val_acc: 0.86610



      0%|          | 0/1000 [00:00<?, ?it/s]


    Epoch [5/5], loss: 0.47502 acc: 0.83350 val_loss: 0.39600, val_acc: 0.86080


### 학습 결과 평가


```python
# 결과 요약
evaluate_history(history)
```

    초기상태 : 손실 : 0.52400  정확도 : 0.81490
    최종상태 : 손실 : 0.39600 정확도 : 0.86080



    
*(이미지 없음 / image not preserved: output_37_1.png)*
    



    
*(이미지 없음 / image not preserved: output_37_2.png)*
    



```python
# 이미지와 정답, 예측 결과를 함께 표시
show_images_labels(test_loader, classes, net, device)
```

    len(images) =  50



    
*(이미지 없음 / image not preserved: output_38_1.png)*
    


## GoogLeNet 불러 오기

### 모델 불러오기


```python
#  라이브러리 임포트
from torchvision import models

# dir(models)
weights = models.GoogLeNet_Weights.IMAGENET1K_V1
net = models.googlenet(weights=weights)
```


```python
# 모델 개요 표시 1
print(net)
```

    GoogLeNet(
      (conv1): BasicConv2d(
        (conv): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
      (maxpool1): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
      (conv2): BasicConv2d(
        (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
      (conv3): BasicConv2d(
        (conv): Conv2d(64, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
      (maxpool2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
      (inception3a): Inception(
        (branch1): BasicConv2d(
          (conv): Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch2): Sequential(
          (0): BasicConv2d(
            (conv): Conv2d(192, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): BasicConv2d(
            (conv): Conv2d(96, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (branch3): Sequential(
          (0): BasicConv2d(
            (conv): Conv2d(192, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(16, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): BasicConv2d(
            (conv): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (branch4): Sequential(
          (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)
          (1): BasicConv2d(
            (conv): Conv2d(192, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
      )
      (inception3b): Inception(
        (branch1): BasicConv2d(
          (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch2): Sequential(
          (0): BasicConv2d(
            (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): BasicConv2d(
            (conv): Conv2d(128, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (branch3): Sequential(
          (0): BasicConv2d(
            (conv): Conv2d(256, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): BasicConv2d(
            (conv): Conv2d(32, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (branch4): Sequential(
          (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)
          (1): BasicConv2d(
            (conv): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
      )
      (maxpool3): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
      (inception4a): Inception(
        (branch1): BasicConv2d(
          (conv): Conv2d(480, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch2): Sequential(
          (0): BasicConv2d(
            (conv): Conv2d(480, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): BasicConv2d(
            (conv): Conv2d(96, 208, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(208, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (branch3): Sequential(
          (0): BasicConv2d(
            (conv): Conv2d(480, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(16, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): BasicConv2d(
            (conv): Conv2d(16, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(48, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (branch4): Sequential(
          (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)
          (1): BasicConv2d(
            (conv): Conv2d(480, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
      )
      (inception4b): Inception(
        (branch1): BasicConv2d(
          (conv): Conv2d(512, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch2): Sequential(
          (0): BasicConv2d(
            (conv): Conv2d(512, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(112, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): BasicConv2d(
            (conv): Conv2d(112, 224, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(224, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (branch3): Sequential(
          (0): BasicConv2d(
            (conv): Conv2d(512, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(24, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): BasicConv2d(
            (conv): Conv2d(24, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (branch4): Sequential(
          (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)
          (1): BasicConv2d(
            (conv): Conv2d(512, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
      )
      (inception4c): Inception(
        (branch1): BasicConv2d(
          (conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch2): Sequential(
          (0): BasicConv2d(
            (conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): BasicConv2d(
            (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(256, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (branch3): Sequential(
          (0): BasicConv2d(
            (conv): Conv2d(512, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(24, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): BasicConv2d(
            (conv): Conv2d(24, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (branch4): Sequential(
          (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)
          (1): BasicConv2d(
            (conv): Conv2d(512, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
      )
      (inception4d): Inception(
        (branch1): BasicConv2d(
          (conv): Conv2d(512, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(112, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch2): Sequential(
          (0): BasicConv2d(
            (conv): Conv2d(512, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(144, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): BasicConv2d(
            (conv): Conv2d(144, 288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(288, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (branch3): Sequential(
          (0): BasicConv2d(
            (conv): Conv2d(512, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): BasicConv2d(
            (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (branch4): Sequential(
          (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)
          (1): BasicConv2d(
            (conv): Conv2d(512, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
      )
      (inception4e): Inception(
        (branch1): BasicConv2d(
          (conv): Conv2d(528, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(256, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch2): Sequential(
          (0): BasicConv2d(
            (conv): Conv2d(528, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): BasicConv2d(
            (conv): Conv2d(160, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(320, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (branch3): Sequential(
          (0): BasicConv2d(
            (conv): Conv2d(528, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): BasicConv2d(
            (conv): Conv2d(32, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (branch4): Sequential(
          (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)
          (1): BasicConv2d(
            (conv): Conv2d(528, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
      )
      (maxpool4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)
      (inception5a): Inception(
        (branch1): BasicConv2d(
          (conv): Conv2d(832, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(256, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch2): Sequential(
          (0): BasicConv2d(
            (conv): Conv2d(832, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): BasicConv2d(
            (conv): Conv2d(160, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(320, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (branch3): Sequential(
          (0): BasicConv2d(
            (conv): Conv2d(832, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): BasicConv2d(
            (conv): Conv2d(32, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (branch4): Sequential(
          (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)
          (1): BasicConv2d(
            (conv): Conv2d(832, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
      )
      (inception5b): Inception(
        (branch1): BasicConv2d(
          (conv): Conv2d(832, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch2): Sequential(
          (0): BasicConv2d(
            (conv): Conv2d(832, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): BasicConv2d(
            (conv): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (branch3): Sequential(
          (0): BasicConv2d(
            (conv): Conv2d(832, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(48, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): BasicConv2d(
            (conv): Conv2d(48, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (branch4): Sequential(
          (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)
          (1): BasicConv2d(
            (conv): Conv2d(832, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
      )
      (aux1): None
      (aux2): None
      (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
      (dropout): Dropout(p=0.2, inplace=False)
      (fc): Linear(in_features=1024, out_features=1000, bias=True)
    )



```python
# 모델 개요 표시 2
# net = net.to(device)
summary(net, (100, 3, 224, 224))
```




    ==========================================================================================
    Layer (type:depth-idx)                   Output Shape              Param #
    ==========================================================================================
    GoogLeNet                                [100, 1000]               --
    ├─BasicConv2d: 1-1                       [100, 64, 112, 112]       --
    │    └─Conv2d: 2-1                       [100, 64, 112, 112]       9,408
    │    └─BatchNorm2d: 2-2                  [100, 64, 112, 112]       128
    ├─MaxPool2d: 1-2                         [100, 64, 56, 56]         --
    ├─BasicConv2d: 1-3                       [100, 64, 56, 56]         --
    │    └─Conv2d: 2-3                       [100, 64, 56, 56]         4,096
    │    └─BatchNorm2d: 2-4                  [100, 64, 56, 56]         128
    ├─BasicConv2d: 1-4                       [100, 192, 56, 56]        --
    │    └─Conv2d: 2-5                       [100, 192, 56, 56]        110,592
    │    └─BatchNorm2d: 2-6                  [100, 192, 56, 56]        384
    ├─MaxPool2d: 1-5                         [100, 192, 28, 28]        --
    ├─Inception: 1-6                         [100, 256, 28, 28]        --
    │    └─BasicConv2d: 2-7                  [100, 64, 28, 28]         --
    │    │    └─Conv2d: 3-1                  [100, 64, 28, 28]         12,288
    │    │    └─BatchNorm2d: 3-2             [100, 64, 28, 28]         128
    │    └─Sequential: 2-8                   [100, 128, 28, 28]        --
    │    │    └─BasicConv2d: 3-3             [100, 96, 28, 28]         18,624
    │    │    └─BasicConv2d: 3-4             [100, 128, 28, 28]        110,848
    │    └─Sequential: 2-9                   [100, 32, 28, 28]         --
    │    │    └─BasicConv2d: 3-5             [100, 16, 28, 28]         3,104
    │    │    └─BasicConv2d: 3-6             [100, 32, 28, 28]         4,672
    │    └─Sequential: 2-10                  [100, 32, 28, 28]         --
    │    │    └─MaxPool2d: 3-7               [100, 192, 28, 28]        --
    │    │    └─BasicConv2d: 3-8             [100, 32, 28, 28]         6,208
    ├─Inception: 1-7                         [100, 480, 28, 28]        --
    │    └─BasicConv2d: 2-11                 [100, 128, 28, 28]        --
    │    │    └─Conv2d: 3-9                  [100, 128, 28, 28]        32,768
    │    │    └─BatchNorm2d: 3-10            [100, 128, 28, 28]        256
    │    └─Sequential: 2-12                  [100, 192, 28, 28]        --
    │    │    └─BasicConv2d: 3-11            [100, 128, 28, 28]        33,024
    │    │    └─BasicConv2d: 3-12            [100, 192, 28, 28]        221,568
    │    └─Sequential: 2-13                  [100, 96, 28, 28]         --
    │    │    └─BasicConv2d: 3-13            [100, 32, 28, 28]         8,256
    │    │    └─BasicConv2d: 3-14            [100, 96, 28, 28]         27,840
    │    └─Sequential: 2-14                  [100, 64, 28, 28]         --
    │    │    └─MaxPool2d: 3-15              [100, 256, 28, 28]        --
    │    │    └─BasicConv2d: 3-16            [100, 64, 28, 28]         16,512
    ├─MaxPool2d: 1-8                         [100, 480, 14, 14]        --
    ├─Inception: 1-9                         [100, 512, 14, 14]        --
    │    └─BasicConv2d: 2-15                 [100, 192, 14, 14]        --
    │    │    └─Conv2d: 3-17                 [100, 192, 14, 14]        92,160
    │    │    └─BatchNorm2d: 3-18            [100, 192, 14, 14]        384
    │    └─Sequential: 2-16                  [100, 208, 14, 14]        --
    │    │    └─BasicConv2d: 3-19            [100, 96, 14, 14]         46,272
    │    │    └─BasicConv2d: 3-20            [100, 208, 14, 14]        180,128
    │    └─Sequential: 2-17                  [100, 48, 14, 14]         --
    │    │    └─BasicConv2d: 3-21            [100, 16, 14, 14]         7,712
    │    │    └─BasicConv2d: 3-22            [100, 48, 14, 14]         7,008
    │    └─Sequential: 2-18                  [100, 64, 14, 14]         --
    │    │    └─MaxPool2d: 3-23              [100, 480, 14, 14]        --
    │    │    └─BasicConv2d: 3-24            [100, 64, 14, 14]         30,848
    ├─Inception: 1-10                        [100, 512, 14, 14]        --
    │    └─BasicConv2d: 2-19                 [100, 160, 14, 14]        --
    │    │    └─Conv2d: 3-25                 [100, 160, 14, 14]        81,920
    │    │    └─BatchNorm2d: 3-26            [100, 160, 14, 14]        320
    │    └─Sequential: 2-20                  [100, 224, 14, 14]        --
    │    │    └─BasicConv2d: 3-27            [100, 112, 14, 14]        57,568
    │    │    └─BasicConv2d: 3-28            [100, 224, 14, 14]        226,240
    │    └─Sequential: 2-21                  [100, 64, 14, 14]         --
    │    │    └─BasicConv2d: 3-29            [100, 24, 14, 14]         12,336
    │    │    └─BasicConv2d: 3-30            [100, 64, 14, 14]         13,952
    │    └─Sequential: 2-22                  [100, 64, 14, 14]         --
    │    │    └─MaxPool2d: 3-31              [100, 512, 14, 14]        --
    │    │    └─BasicConv2d: 3-32            [100, 64, 14, 14]         32,896
    ├─Inception: 1-11                        [100, 512, 14, 14]        --
    │    └─BasicConv2d: 2-23                 [100, 128, 14, 14]        --
    │    │    └─Conv2d: 3-33                 [100, 128, 14, 14]        65,536
    │    │    └─BatchNorm2d: 3-34            [100, 128, 14, 14]        256
    │    └─Sequential: 2-24                  [100, 256, 14, 14]        --
    │    │    └─BasicConv2d: 3-35            [100, 128, 14, 14]        65,792
    │    │    └─BasicConv2d: 3-36            [100, 256, 14, 14]        295,424
    │    └─Sequential: 2-25                  [100, 64, 14, 14]         --
    │    │    └─BasicConv2d: 3-37            [100, 24, 14, 14]         12,336
    │    │    └─BasicConv2d: 3-38            [100, 64, 14, 14]         13,952
    │    └─Sequential: 2-26                  [100, 64, 14, 14]         --
    │    │    └─MaxPool2d: 3-39              [100, 512, 14, 14]        --
    │    │    └─BasicConv2d: 3-40            [100, 64, 14, 14]         32,896
    ├─Inception: 1-12                        [100, 528, 14, 14]        --
    │    └─BasicConv2d: 2-27                 [100, 112, 14, 14]        --
    │    │    └─Conv2d: 3-41                 [100, 112, 14, 14]        57,344
    │    │    └─BatchNorm2d: 3-42            [100, 112, 14, 14]        224
    │    └─Sequential: 2-28                  [100, 288, 14, 14]        --
    │    │    └─BasicConv2d: 3-43            [100, 144, 14, 14]        74,016
    │    │    └─BasicConv2d: 3-44            [100, 288, 14, 14]        373,824
    │    └─Sequential: 2-29                  [100, 64, 14, 14]         --
    │    │    └─BasicConv2d: 3-45            [100, 32, 14, 14]         16,448
    │    │    └─BasicConv2d: 3-46            [100, 64, 14, 14]         18,560
    │    └─Sequential: 2-30                  [100, 64, 14, 14]         --
    │    │    └─MaxPool2d: 3-47              [100, 512, 14, 14]        --
    │    │    └─BasicConv2d: 3-48            [100, 64, 14, 14]         32,896
    ├─Inception: 1-13                        [100, 832, 14, 14]        --
    │    └─BasicConv2d: 2-31                 [100, 256, 14, 14]        --
    │    │    └─Conv2d: 3-49                 [100, 256, 14, 14]        135,168
    │    │    └─BatchNorm2d: 3-50            [100, 256, 14, 14]        512
    │    └─Sequential: 2-32                  [100, 320, 14, 14]        --
    │    │    └─BasicConv2d: 3-51            [100, 160, 14, 14]        84,800
    │    │    └─BasicConv2d: 3-52            [100, 320, 14, 14]        461,440
    │    └─Sequential: 2-33                  [100, 128, 14, 14]        --
    │    │    └─BasicConv2d: 3-53            [100, 32, 14, 14]         16,960
    │    │    └─BasicConv2d: 3-54            [100, 128, 14, 14]        37,120
    │    └─Sequential: 2-34                  [100, 128, 14, 14]        --
    │    │    └─MaxPool2d: 3-55              [100, 528, 14, 14]        --
    │    │    └─BasicConv2d: 3-56            [100, 128, 14, 14]        67,840
    ├─MaxPool2d: 1-14                        [100, 832, 7, 7]          --
    ├─Inception: 1-15                        [100, 832, 7, 7]          --
    │    └─BasicConv2d: 2-35                 [100, 256, 7, 7]          --
    │    │    └─Conv2d: 3-57                 [100, 256, 7, 7]          212,992
    │    │    └─BatchNorm2d: 3-58            [100, 256, 7, 7]          512
    │    └─Sequential: 2-36                  [100, 320, 7, 7]          --
    │    │    └─BasicConv2d: 3-59            [100, 160, 7, 7]          133,440
    │    │    └─BasicConv2d: 3-60            [100, 320, 7, 7]          461,440
    │    └─Sequential: 2-37                  [100, 128, 7, 7]          --
    │    │    └─BasicConv2d: 3-61            [100, 32, 7, 7]           26,688
    │    │    └─BasicConv2d: 3-62            [100, 128, 7, 7]          37,120
    │    └─Sequential: 2-38                  [100, 128, 7, 7]          --
    │    │    └─MaxPool2d: 3-63              [100, 832, 7, 7]          --
    │    │    └─BasicConv2d: 3-64            [100, 128, 7, 7]          106,752
    ├─Inception: 1-16                        [100, 1024, 7, 7]         --
    │    └─BasicConv2d: 2-39                 [100, 384, 7, 7]          --
    │    │    └─Conv2d: 3-65                 [100, 384, 7, 7]          319,488
    │    │    └─BatchNorm2d: 3-66            [100, 384, 7, 7]          768
    │    └─Sequential: 2-40                  [100, 384, 7, 7]          --
    │    │    └─BasicConv2d: 3-67            [100, 192, 7, 7]          160,128
    │    │    └─BasicConv2d: 3-68            [100, 384, 7, 7]          664,320
    │    └─Sequential: 2-41                  [100, 128, 7, 7]          --
    │    │    └─BasicConv2d: 3-69            [100, 48, 7, 7]           40,032
    │    │    └─BasicConv2d: 3-70            [100, 128, 7, 7]          55,552
    │    └─Sequential: 2-42                  [100, 128, 7, 7]          --
    │    │    └─MaxPool2d: 3-71              [100, 832, 7, 7]          --
    │    │    └─BasicConv2d: 3-72            [100, 128, 7, 7]          106,752
    ├─AdaptiveAvgPool2d: 1-17                [100, 1024, 1, 1]         --
    ├─Dropout: 1-18                          [100, 1024]               --
    ├─Linear: 1-19                           [100, 1000]               1,025,000
    ==========================================================================================
    Total params: 6,624,904
    Trainable params: 6,624,904
    Non-trainable params: 0
    Total mult-adds (G): 149.84
    ==========================================================================================
    Input size (MB): 60.21
    Forward/backward pass size (MB): 5162.66
    Params size (MB): 26.50
    Estimated Total Size (MB): 5249.37
    ==========================================================================================



## 파인 튜닝 없이 사용하기
### 영상 읽기


```python
### Step 2: Read image
## rgb format, <class 'torch.Tensor'>
from torchvision.io import read_image

net.eval()

filename = "./beagle.jpg"
img = read_image(filename) # torch.Size([3, 366, 640]) 
img = img.to(device)

##
print("img type = ", type(img))
print("img shape = ", img.shape) # torch.Size([3, 366, 640]))
```

    img type =  <class 'torch.Tensor'>
    img shape =  torch.Size([3, 366, 640])


### 영상 변환


```python
# preprocess
# Scaling pixel values down to the [0, 1] range from their original [0, 255] range before applying normalization.
# ImageClassification(
#     crop_size=[224]
#     resize_size=[256]
#     mean=[0.485, 0.456, 0.406]
#     std=[0.229, 0.224, 0.225]
#     interpolation=InterpolationMode.BILINEAR
# )

preprocess = weights.transforms()
print(preprocess)
```

    ImageClassification(
        crop_size=[224]
        resize_size=[256]
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        interpolation=InterpolationMode.BILINEAR
    )


### 변환 영상 확인하기


```python
batch = preprocess(img).unsqueeze(0).to(device)
print(batch.shape)

##
processed_img = batch.data[0]
plt_img = processed_img.permute(1, 2, 0)

# plt_img.shape
plt.imshow(plt_img.cpu().numpy())
plt.grid(visible = None)
plt.axis("off")
plt.show()
```

    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-2.0665298..2.3611333].


    torch.Size([1, 3, 224, 224])



    
*(이미지 없음 / image not preserved: output_49_2.png)*
    



```python
# Step 5: Use the model and print the predicted category
net = net.to(device)
prediction = net(batch).softmax(1) # (1, 1000)
class_id = prediction.argmax().item() 
print("class id = ", class_id)
```

    class id =  162


### 결과 확인 하기


```python
display("category = \n", weights.meta["categories"])
print("category number = ", len(weights.meta["categories"]))
```


    'category = \n'



    ['tench',
     'goldfish',
     'great white shark',
     'tiger shark',
     'hammerhead',
     'electric ray',
     'stingray',
     'cock',
     'hen',
     'ostrich',
     'brambling',
     'goldfinch',
     'house finch',
     'junco',
     'indigo bunting',
     'robin',
     'bulbul',
     'jay',
     'magpie',
     'chickadee',
     'water ouzel',
     'kite',
     'bald eagle',
     'vulture',
     'great grey owl',
     'European fire salamander',
     'common newt',
     'eft',
     'spotted salamander',
     'axolotl',
     'bullfrog',
     'tree frog',
     'tailed frog',
     'loggerhead',
     'leatherback turtle',
     'mud turtle',
     'terrapin',
     'box turtle',
     'banded gecko',
     'common iguana',
     'American chameleon',
     'whiptail',
     'agama',
     'frilled lizard',
     'alligator lizard',
     'Gila monster',
     'green lizard',
     'African chameleon',
     'Komodo dragon',
     'African crocodile',
     'American alligator',
     'triceratops',
     'thunder snake',
     'ringneck snake',
     'hognose snake',
     'green snake',
     'king snake',
     'garter snake',
     'water snake',
     'vine snake',
     'night snake',
     'boa constrictor',
     'rock python',
     'Indian cobra',
     'green mamba',
     'sea snake',
     'horned viper',
     'diamondback',
     'sidewinder',
     'trilobite',
     'harvestman',
     'scorpion',
     'black and gold garden spider',
     'barn spider',
     'garden spider',
     'black widow',
     'tarantula',
     'wolf spider',
     'tick',
     'centipede',
     'black grouse',
     'ptarmigan',
     'ruffed grouse',
     'prairie chicken',
     'peacock',
     'quail',
     'partridge',
     'African grey',
     'macaw',
     'sulphur-crested cockatoo',
     'lorikeet',
     'coucal',
     'bee eater',
     'hornbill',
     'hummingbird',
     'jacamar',
     'toucan',
     'drake',
     'red-breasted merganser',
     'goose',
     'black swan',
     'tusker',
     'echidna',
     'platypus',
     'wallaby',
     'koala',
     'wombat',
     'jellyfish',
     'sea anemone',
     'brain coral',
     'flatworm',
     'nematode',
     'conch',
     'snail',
     'slug',
     'sea slug',
     'chiton',
     'chambered nautilus',
     'Dungeness crab',
     'rock crab',
     'fiddler crab',
     'king crab',
     'American lobster',
     'spiny lobster',
     'crayfish',
     'hermit crab',
     'isopod',
     'white stork',
     'black stork',
     'spoonbill',
     'flamingo',
     'little blue heron',
     'American egret',
     'bittern',
     'crane bird',
     'limpkin',
     'European gallinule',
     'American coot',
     'bustard',
     'ruddy turnstone',
     'red-backed sandpiper',
     'redshank',
     'dowitcher',
     'oystercatcher',
     'pelican',
     'king penguin',
     'albatross',
     'grey whale',
     'killer whale',
     'dugong',
     'sea lion',
     'Chihuahua',
     'Japanese spaniel',
     'Maltese dog',
     'Pekinese',
     'Shih-Tzu',
     'Blenheim spaniel',
     'papillon',
     'toy terrier',
     'Rhodesian ridgeback',
     'Afghan hound',
     'basset',
     'beagle',
     'bloodhound',
     'bluetick',
     'black-and-tan coonhound',
     'Walker hound',
     'English foxhound',
     'redbone',
     'borzoi',
     'Irish wolfhound',
     'Italian greyhound',
     'whippet',
     'Ibizan hound',
     'Norwegian elkhound',
     'otterhound',
     'Saluki',
     'Scottish deerhound',
     'Weimaraner',
     'Staffordshire bullterrier',
     'American Staffordshire terrier',
     'Bedlington terrier',
     'Border terrier',
     'Kerry blue terrier',
     'Irish terrier',
     'Norfolk terrier',
     'Norwich terrier',
     'Yorkshire terrier',
     'wire-haired fox terrier',
     'Lakeland terrier',
     'Sealyham terrier',
     'Airedale',
     'cairn',
     'Australian terrier',
     'Dandie Dinmont',
     'Boston bull',
     'miniature schnauzer',
     'giant schnauzer',
     'standard schnauzer',
     'Scotch terrier',
     'Tibetan terrier',
     'silky terrier',
     'soft-coated wheaten terrier',
     'West Highland white terrier',
     'Lhasa',
     'flat-coated retriever',
     'curly-coated retriever',
     'golden retriever',
     'Labrador retriever',
     'Chesapeake Bay retriever',
     'German short-haired pointer',
     'vizsla',
     'English setter',
     'Irish setter',
     'Gordon setter',
     'Brittany spaniel',
     'clumber',
     'English springer',
     'Welsh springer spaniel',
     'cocker spaniel',
     'Sussex spaniel',
     'Irish water spaniel',
     'kuvasz',
     'schipperke',
     'groenendael',
     'malinois',
     'briard',
     'kelpie',
     'komondor',
     'Old English sheepdog',
     'Shetland sheepdog',
     'collie',
     'Border collie',
     'Bouvier des Flandres',
     'Rottweiler',
     'German shepherd',
     'Doberman',
     'miniature pinscher',
     'Greater Swiss Mountain dog',
     'Bernese mountain dog',
     'Appenzeller',
     'EntleBucher',
     'boxer',
     'bull mastiff',
     'Tibetan mastiff',
     'French bulldog',
     'Great Dane',
     'Saint Bernard',
     'Eskimo dog',
     'malamute',
     'Siberian husky',
     'dalmatian',
     'affenpinscher',
     'basenji',
     'pug',
     'Leonberg',
     'Newfoundland',
     'Great Pyrenees',
     'Samoyed',
     'Pomeranian',
     'chow',
     'keeshond',
     'Brabancon griffon',
     'Pembroke',
     'Cardigan',
     'toy poodle',
     'miniature poodle',
     'standard poodle',
     'Mexican hairless',
     'timber wolf',
     'white wolf',
     'red wolf',
     'coyote',
     'dingo',
     'dhole',
     'African hunting dog',
     'hyena',
     'red fox',
     'kit fox',
     'Arctic fox',
     'grey fox',
     'tabby',
     'tiger cat',
     'Persian cat',
     'Siamese cat',
     'Egyptian cat',
     'cougar',
     'lynx',
     'leopard',
     'snow leopard',
     'jaguar',
     'lion',
     'tiger',
     'cheetah',
     'brown bear',
     'American black bear',
     'ice bear',
     'sloth bear',
     'mongoose',
     'meerkat',
     'tiger beetle',
     'ladybug',
     'ground beetle',
     'long-horned beetle',
     'leaf beetle',
     'dung beetle',
     'rhinoceros beetle',
     'weevil',
     'fly',
     'bee',
     'ant',
     'grasshopper',
     'cricket',
     'walking stick',
     'cockroach',
     'mantis',
     'cicada',
     'leafhopper',
     'lacewing',
     'dragonfly',
     'damselfly',
     'admiral',
     'ringlet',
     'monarch',
     'cabbage butterfly',
     'sulphur butterfly',
     'lycaenid',
     'starfish',
     'sea urchin',
     'sea cucumber',
     'wood rabbit',
     'hare',
     'Angora',
     'hamster',
     'porcupine',
     'fox squirrel',
     'marmot',
     'beaver',
     'guinea pig',
     'sorrel',
     'zebra',
     'hog',
     'wild boar',
     'warthog',
     'hippopotamus',
     'ox',
     'water buffalo',
     'bison',
     'ram',
     'bighorn',
     'ibex',
     'hartebeest',
     'impala',
     'gazelle',
     'Arabian camel',
     'llama',
     'weasel',
     'mink',
     'polecat',
     'black-footed ferret',
     'otter',
     'skunk',
     'badger',
     'armadillo',
     'three-toed sloth',
     'orangutan',
     'gorilla',
     'chimpanzee',
     'gibbon',
     'siamang',
     'guenon',
     'patas',
     'baboon',
     'macaque',
     'langur',
     'colobus',
     'proboscis monkey',
     'marmoset',
     'capuchin',
     'howler monkey',
     'titi',
     'spider monkey',
     'squirrel monkey',
     'Madagascar cat',
     'indri',
     'Indian elephant',
     'African elephant',
     'lesser panda',
     'giant panda',
     'barracouta',
     'eel',
     'coho',
     'rock beauty',
     'anemone fish',
     'sturgeon',
     'gar',
     'lionfish',
     'puffer',
     'abacus',
     'abaya',
     'academic gown',
     'accordion',
     'acoustic guitar',
     'aircraft carrier',
     'airliner',
     'airship',
     'altar',
     'ambulance',
     'amphibian',
     'analog clock',
     'apiary',
     'apron',
     'ashcan',
     'assault rifle',
     'backpack',
     'bakery',
     'balance beam',
     'balloon',
     'ballpoint',
     'Band Aid',
     'banjo',
     'bannister',
     'barbell',
     'barber chair',
     'barbershop',
     'barn',
     'barometer',
     'barrel',
     'barrow',
     'baseball',
     'basketball',
     'bassinet',
     'bassoon',
     'bathing cap',
     'bath towel',
     'bathtub',
     'beach wagon',
     'beacon',
     'beaker',
     'bearskin',
     'beer bottle',
     'beer glass',
     'bell cote',
     'bib',
     'bicycle-built-for-two',
     'bikini',
     'binder',
     'binoculars',
     'birdhouse',
     'boathouse',
     'bobsled',
     'bolo tie',
     'bonnet',
     'bookcase',
     'bookshop',
     'bottlecap',
     'bow',
     'bow tie',
     'brass',
     'brassiere',
     'breakwater',
     'breastplate',
     'broom',
     'bucket',
     'buckle',
     'bulletproof vest',
     'bullet train',
     'butcher shop',
     'cab',
     'caldron',
     'candle',
     'cannon',
     'canoe',
     'can opener',
     'cardigan',
     'car mirror',
     'carousel',
     "carpenter's kit",
     'carton',
     'car wheel',
     'cash machine',
     'cassette',
     'cassette player',
     'castle',
     'catamaran',
     'CD player',
     'cello',
     'cellular telephone',
     'chain',
     'chainlink fence',
     'chain mail',
     'chain saw',
     'chest',
     'chiffonier',
     'chime',
     'china cabinet',
     'Christmas stocking',
     'church',
     'cinema',
     'cleaver',
     'cliff dwelling',
     'cloak',
     'clog',
     'cocktail shaker',
     'coffee mug',
     'coffeepot',
     'coil',
     'combination lock',
     'computer keyboard',
     'confectionery',
     'container ship',
     'convertible',
     'corkscrew',
     'cornet',
     'cowboy boot',
     'cowboy hat',
     'cradle',
     'crane',
     'crash helmet',
     'crate',
     'crib',
     'Crock Pot',
     'croquet ball',
     'crutch',
     'cuirass',
     'dam',
     'desk',
     'desktop computer',
     'dial telephone',
     'diaper',
     'digital clock',
     'digital watch',
     'dining table',
     'dishrag',
     'dishwasher',
     'disk brake',
     'dock',
     'dogsled',
     'dome',
     'doormat',
     'drilling platform',
     'drum',
     'drumstick',
     'dumbbell',
     'Dutch oven',
     'electric fan',
     'electric guitar',
     'electric locomotive',
     'entertainment center',
     'envelope',
     'espresso maker',
     'face powder',
     'feather boa',
     'file',
     'fireboat',
     'fire engine',
     'fire screen',
     'flagpole',
     'flute',
     'folding chair',
     'football helmet',
     'forklift',
     'fountain',
     'fountain pen',
     'four-poster',
     'freight car',
     'French horn',
     'frying pan',
     'fur coat',
     'garbage truck',
     'gasmask',
     'gas pump',
     'goblet',
     'go-kart',
     'golf ball',
     'golfcart',
     'gondola',
     'gong',
     'gown',
     'grand piano',
     'greenhouse',
     'grille',
     'grocery store',
     'guillotine',
     'hair slide',
     'hair spray',
     'half track',
     'hammer',
     'hamper',
     'hand blower',
     'hand-held computer',
     'handkerchief',
     'hard disc',
     'harmonica',
     'harp',
     'harvester',
     'hatchet',
     'holster',
     'home theater',
     'honeycomb',
     'hook',
     'hoopskirt',
     'horizontal bar',
     'horse cart',
     'hourglass',
     'iPod',
     'iron',
     "jack-o'-lantern",
     'jean',
     'jeep',
     'jersey',
     'jigsaw puzzle',
     'jinrikisha',
     'joystick',
     'kimono',
     'knee pad',
     'knot',
     'lab coat',
     'ladle',
     'lampshade',
     'laptop',
     'lawn mower',
     'lens cap',
     'letter opener',
     'library',
     'lifeboat',
     'lighter',
     'limousine',
     'liner',
     'lipstick',
     'Loafer',
     'lotion',
     'loudspeaker',
     'loupe',
     'lumbermill',
     'magnetic compass',
     'mailbag',
     'mailbox',
     'maillot',
     'maillot tank suit',
     'manhole cover',
     'maraca',
     'marimba',
     'mask',
     'matchstick',
     'maypole',
     'maze',
     'measuring cup',
     'medicine chest',
     'megalith',
     'microphone',
     'microwave',
     'military uniform',
     'milk can',
     'minibus',
     'miniskirt',
     'minivan',
     'missile',
     'mitten',
     'mixing bowl',
     'mobile home',
     'Model T',
     'modem',
     'monastery',
     'monitor',
     'moped',
     'mortar',
     'mortarboard',
     'mosque',
     'mosquito net',
     'motor scooter',
     'mountain bike',
     'mountain tent',
     'mouse',
     'mousetrap',
     'moving van',
     'muzzle',
     'nail',
     'neck brace',
     'necklace',
     'nipple',
     'notebook',
     'obelisk',
     'oboe',
     'ocarina',
     'odometer',
     'oil filter',
     'organ',
     'oscilloscope',
     'overskirt',
     'oxcart',
     'oxygen mask',
     'packet',
     'paddle',
     'paddlewheel',
     'padlock',
     'paintbrush',
     'pajama',
     'palace',
     'panpipe',
     'paper towel',
     'parachute',
     'parallel bars',
     'park bench',
     'parking meter',
     'passenger car',
     'patio',
     'pay-phone',
     'pedestal',
     'pencil box',
     'pencil sharpener',
     'perfume',
     'Petri dish',
     'photocopier',
     'pick',
     'pickelhaube',
     'picket fence',
     'pickup',
     'pier',
     'piggy bank',
     'pill bottle',
     'pillow',
     'ping-pong ball',
     'pinwheel',
     'pirate',
     'pitcher',
     'plane',
     'planetarium',
     'plastic bag',
     'plate rack',
     'plow',
     'plunger',
     'Polaroid camera',
     'pole',
     'police van',
     'poncho',
     'pool table',
     'pop bottle',
     'pot',
     "potter's wheel",
     'power drill',
     'prayer rug',
     'printer',
     'prison',
     'projectile',
     'projector',
     'puck',
     'punching bag',
     'purse',
     'quill',
     'quilt',
     'racer',
     'racket',
     'radiator',
     'radio',
     'radio telescope',
     'rain barrel',
     'recreational vehicle',
     'reel',
     'reflex camera',
     'refrigerator',
     'remote control',
     'restaurant',
     'revolver',
     'rifle',
     'rocking chair',
     'rotisserie',
     'rubber eraser',
     'rugby ball',
     'rule',
     'running shoe',
     'safe',
     'safety pin',
     'saltshaker',
     'sandal',
     'sarong',
     'sax',
     'scabbard',
     'scale',
     'school bus',
     'schooner',
     'scoreboard',
     'screen',
     'screw',
     'screwdriver',
     'seat belt',
     'sewing machine',
     'shield',
     'shoe shop',
     'shoji',
     'shopping basket',
     'shopping cart',
     'shovel',
     'shower cap',
     'shower curtain',
     'ski',
     'ski mask',
     'sleeping bag',
     'slide rule',
     'sliding door',
     'slot',
     'snorkel',
     'snowmobile',
     'snowplow',
     'soap dispenser',
     'soccer ball',
     'sock',
     'solar dish',
     'sombrero',
     'soup bowl',
     'space bar',
     'space heater',
     'space shuttle',
     'spatula',
     'speedboat',
     'spider web',
     'spindle',
     'sports car',
     'spotlight',
     'stage',
     'steam locomotive',
     'steel arch bridge',
     'steel drum',
     'stethoscope',
     'stole',
     'stone wall',
     'stopwatch',
     'stove',
     'strainer',
     'streetcar',
     'stretcher',
     'studio couch',
     'stupa',
     'submarine',
     'suit',
     'sundial',
     'sunglass',
     'sunglasses',
     'sunscreen',
     'suspension bridge',
     'swab',
     'sweatshirt',
     'swimming trunks',
     'swing',
     'switch',
     'syringe',
     'table lamp',
     'tank',
     'tape player',
     'teapot',
     'teddy',
     'television',
     'tennis ball',
     'thatch',
     'theater curtain',
     'thimble',
     'thresher',
     'throne',
     'tile roof',
     'toaster',
     'tobacco shop',
     'toilet seat',
     'torch',
     'totem pole',
     'tow truck',
     'toyshop',
     'tractor',
     'trailer truck',
     'tray',
     'trench coat',
     'tricycle',
     'trimaran',
     'tripod',
     'triumphal arch',
     'trolleybus',
     'trombone',
     'tub',
     'turnstile',
     'typewriter keyboard',
     'umbrella',
     'unicycle',
     'upright',
     'vacuum',
     'vase',
     'vault',
     'velvet',
     'vending machine',
     'vestment',
     'viaduct',
     'violin',
     'volleyball',
     'waffle iron',
     'wall clock',
     'wallet',
     'wardrobe',
     'warplane',
     'washbasin',
     'washer',
     'water bottle',
     'water jug',
     'water tower',
     'whiskey jug',
     'whistle',
     'wig',
     'window screen',
     'window shade',
     'Windsor tie',
     'wine bottle',
     'wing',
     'wok',
     'wooden spoon',
     'wool',
     'worm fence',
     'wreck',
     'yawl',
     'yurt',
     'web site',
     'comic book',
     'crossword puzzle',
     'street sign',
     'traffic light',
     'book jacket',
     'menu',
     'plate',
     'guacamole',
     'consomme',
     'hot pot',
     'trifle',
     'ice cream',
     'ice lolly',
     'French loaf',
     'bagel',
     'pretzel',
     'cheeseburger',
     'hotdog',
     'mashed potato',
     'head cabbage',
     'broccoli',
     'cauliflower',
     'zucchini',
     'spaghetti squash',
     'acorn squash',
     'butternut squash',
     'cucumber',
     'artichoke',
     'bell pepper',
     'cardoon',
     'mushroom',
     'Granny Smith',
     'strawberry',
     'orange',
     'lemon',
     'fig',
     'pineapple',
     'banana',
     'jackfruit',
     'custard apple',
     'pomegranate',
     'hay',
     'carbonara',
     'chocolate sauce',
     'dough',
     'meat loaf',
     'pizza',
     'potpie',
     'burrito',
     'red wine',
     'espresso',
     'cup',
     'eggnog',
     'alp',
     'bubble',
     'cliff',
     'coral reef',
     'geyser',
     'lakeside',
     'promontory',
     'sandbar',
     'seashore',
     'valley',
     'volcano',
     'ballplayer',
     'groom',
     'scuba diver',
     'rapeseed',
     'daisy',
     "yellow lady's slipper",
     'corn',
     'acorn',
     'hip',
     'buckeye',
     'coral fungus',
     'agaric',
     'gyromitra',
     'stinkhorn',
     'earthstar',
     'hen-of-the-woods',
     'bolete',
     'ear',
     'toilet tissue']


    category number =  1000



```python
category_name = weights.meta["categories"][class_id]
category_name
```




    'beagle'



## 최종 레이어 함수 교체하기 (전이 학습)



```python
# 모델 개요 표시 1
print(net)
```

    GoogLeNet(
      (conv1): BasicConv2d(
        (conv): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
      (maxpool1): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
      (conv2): BasicConv2d(
        (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
      (conv3): BasicConv2d(
        (conv): Conv2d(64, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
      (maxpool2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
      (inception3a): Inception(
        (branch1): BasicConv2d(
          (conv): Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch2): Sequential(
          (0): BasicConv2d(
            (conv): Conv2d(192, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): BasicConv2d(
            (conv): Conv2d(96, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (branch3): Sequential(
          (0): BasicConv2d(
            (conv): Conv2d(192, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(16, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): BasicConv2d(
            (conv): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (branch4): Sequential(
          (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)
          (1): BasicConv2d(
            (conv): Conv2d(192, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
      )
      (inception3b): Inception(
        (branch1): BasicConv2d(
          (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch2): Sequential(
          (0): BasicConv2d(
            (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): BasicConv2d(
            (conv): Conv2d(128, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (branch3): Sequential(
          (0): BasicConv2d(
            (conv): Conv2d(256, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): BasicConv2d(
            (conv): Conv2d(32, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (branch4): Sequential(
          (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)
          (1): BasicConv2d(
            (conv): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
      )
      (maxpool3): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
      (inception4a): Inception(
        (branch1): BasicConv2d(
          (conv): Conv2d(480, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch2): Sequential(
          (0): BasicConv2d(
            (conv): Conv2d(480, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): BasicConv2d(
            (conv): Conv2d(96, 208, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(208, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (branch3): Sequential(
          (0): BasicConv2d(
            (conv): Conv2d(480, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(16, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): BasicConv2d(
            (conv): Conv2d(16, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(48, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (branch4): Sequential(
          (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)
          (1): BasicConv2d(
            (conv): Conv2d(480, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
      )
      (inception4b): Inception(
        (branch1): BasicConv2d(
          (conv): Conv2d(512, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch2): Sequential(
          (0): BasicConv2d(
            (conv): Conv2d(512, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(112, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): BasicConv2d(
            (conv): Conv2d(112, 224, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(224, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (branch3): Sequential(
          (0): BasicConv2d(
            (conv): Conv2d(512, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(24, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): BasicConv2d(
            (conv): Conv2d(24, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (branch4): Sequential(
          (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)
          (1): BasicConv2d(
            (conv): Conv2d(512, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
      )
      (inception4c): Inception(
        (branch1): BasicConv2d(
          (conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch2): Sequential(
          (0): BasicConv2d(
            (conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): BasicConv2d(
            (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(256, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (branch3): Sequential(
          (0): BasicConv2d(
            (conv): Conv2d(512, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(24, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): BasicConv2d(
            (conv): Conv2d(24, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (branch4): Sequential(
          (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)
          (1): BasicConv2d(
            (conv): Conv2d(512, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
      )
      (inception4d): Inception(
        (branch1): BasicConv2d(
          (conv): Conv2d(512, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(112, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch2): Sequential(
          (0): BasicConv2d(
            (conv): Conv2d(512, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(144, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): BasicConv2d(
            (conv): Conv2d(144, 288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(288, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (branch3): Sequential(
          (0): BasicConv2d(
            (conv): Conv2d(512, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): BasicConv2d(
            (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (branch4): Sequential(
          (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)
          (1): BasicConv2d(
            (conv): Conv2d(512, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
      )
      (inception4e): Inception(
        (branch1): BasicConv2d(
          (conv): Conv2d(528, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(256, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch2): Sequential(
          (0): BasicConv2d(
            (conv): Conv2d(528, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): BasicConv2d(
            (conv): Conv2d(160, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(320, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (branch3): Sequential(
          (0): BasicConv2d(
            (conv): Conv2d(528, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): BasicConv2d(
            (conv): Conv2d(32, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (branch4): Sequential(
          (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)
          (1): BasicConv2d(
            (conv): Conv2d(528, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
      )
      (maxpool4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)
      (inception5a): Inception(
        (branch1): BasicConv2d(
          (conv): Conv2d(832, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(256, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch2): Sequential(
          (0): BasicConv2d(
            (conv): Conv2d(832, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): BasicConv2d(
            (conv): Conv2d(160, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(320, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (branch3): Sequential(
          (0): BasicConv2d(
            (conv): Conv2d(832, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): BasicConv2d(
            (conv): Conv2d(32, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (branch4): Sequential(
          (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)
          (1): BasicConv2d(
            (conv): Conv2d(832, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
      )
      (inception5b): Inception(
        (branch1): BasicConv2d(
          (conv): Conv2d(832, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch2): Sequential(
          (0): BasicConv2d(
            (conv): Conv2d(832, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): BasicConv2d(
            (conv): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (branch3): Sequential(
          (0): BasicConv2d(
            (conv): Conv2d(832, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(48, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): BasicConv2d(
            (conv): Conv2d(48, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (branch4): Sequential(
          (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)
          (1): BasicConv2d(
            (conv): Conv2d(832, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
      )
      (aux1): None
      (aux2): None
      (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
      (dropout): Dropout(p=0.2, inplace=False)
      (fc): Linear(in_features=1024, out_features=1000, bias=True)
    )



```python
# 모델 개요 표시 
summary(net,(100, 3, 112, 112))
```




    ==========================================================================================
    Layer (type:depth-idx)                   Output Shape              Param #
    ==========================================================================================
    GoogLeNet                                [100, 1000]               --
    ├─BasicConv2d: 1-1                       [100, 64, 56, 56]         --
    │    └─Conv2d: 2-1                       [100, 64, 56, 56]         9,408
    │    └─BatchNorm2d: 2-2                  [100, 64, 56, 56]         128
    ├─MaxPool2d: 1-2                         [100, 64, 28, 28]         --
    ├─BasicConv2d: 1-3                       [100, 64, 28, 28]         --
    │    └─Conv2d: 2-3                       [100, 64, 28, 28]         4,096
    │    └─BatchNorm2d: 2-4                  [100, 64, 28, 28]         128
    ├─BasicConv2d: 1-4                       [100, 192, 28, 28]        --
    │    └─Conv2d: 2-5                       [100, 192, 28, 28]        110,592
    │    └─BatchNorm2d: 2-6                  [100, 192, 28, 28]        384
    ├─MaxPool2d: 1-5                         [100, 192, 14, 14]        --
    ├─Inception: 1-6                         [100, 256, 14, 14]        --
    │    └─BasicConv2d: 2-7                  [100, 64, 14, 14]         --
    │    │    └─Conv2d: 3-1                  [100, 64, 14, 14]         12,288
    │    │    └─BatchNorm2d: 3-2             [100, 64, 14, 14]         128
    │    └─Sequential: 2-8                   [100, 128, 14, 14]        --
    │    │    └─BasicConv2d: 3-3             [100, 96, 14, 14]         18,624
    │    │    └─BasicConv2d: 3-4             [100, 128, 14, 14]        110,848
    │    └─Sequential: 2-9                   [100, 32, 14, 14]         --
    │    │    └─BasicConv2d: 3-5             [100, 16, 14, 14]         3,104
    │    │    └─BasicConv2d: 3-6             [100, 32, 14, 14]         4,672
    │    └─Sequential: 2-10                  [100, 32, 14, 14]         --
    │    │    └─MaxPool2d: 3-7               [100, 192, 14, 14]        --
    │    │    └─BasicConv2d: 3-8             [100, 32, 14, 14]         6,208
    ├─Inception: 1-7                         [100, 480, 14, 14]        --
    │    └─BasicConv2d: 2-11                 [100, 128, 14, 14]        --
    │    │    └─Conv2d: 3-9                  [100, 128, 14, 14]        32,768
    │    │    └─BatchNorm2d: 3-10            [100, 128, 14, 14]        256
    │    └─Sequential: 2-12                  [100, 192, 14, 14]        --
    │    │    └─BasicConv2d: 3-11            [100, 128, 14, 14]        33,024
    │    │    └─BasicConv2d: 3-12            [100, 192, 14, 14]        221,568
    │    └─Sequential: 2-13                  [100, 96, 14, 14]         --
    │    │    └─BasicConv2d: 3-13            [100, 32, 14, 14]         8,256
    │    │    └─BasicConv2d: 3-14            [100, 96, 14, 14]         27,840
    │    └─Sequential: 2-14                  [100, 64, 14, 14]         --
    │    │    └─MaxPool2d: 3-15              [100, 256, 14, 14]        --
    │    │    └─BasicConv2d: 3-16            [100, 64, 14, 14]         16,512
    ├─MaxPool2d: 1-8                         [100, 480, 7, 7]          --
    ├─Inception: 1-9                         [100, 512, 7, 7]          --
    │    └─BasicConv2d: 2-15                 [100, 192, 7, 7]          --
    │    │    └─Conv2d: 3-17                 [100, 192, 7, 7]          92,160
    │    │    └─BatchNorm2d: 3-18            [100, 192, 7, 7]          384
    │    └─Sequential: 2-16                  [100, 208, 7, 7]          --
    │    │    └─BasicConv2d: 3-19            [100, 96, 7, 7]           46,272
    │    │    └─BasicConv2d: 3-20            [100, 208, 7, 7]          180,128
    │    └─Sequential: 2-17                  [100, 48, 7, 7]           --
    │    │    └─BasicConv2d: 3-21            [100, 16, 7, 7]           7,712
    │    │    └─BasicConv2d: 3-22            [100, 48, 7, 7]           7,008
    │    └─Sequential: 2-18                  [100, 64, 7, 7]           --
    │    │    └─MaxPool2d: 3-23              [100, 480, 7, 7]          --
    │    │    └─BasicConv2d: 3-24            [100, 64, 7, 7]           30,848
    ├─Inception: 1-10                        [100, 512, 7, 7]          --
    │    └─BasicConv2d: 2-19                 [100, 160, 7, 7]          --
    │    │    └─Conv2d: 3-25                 [100, 160, 7, 7]          81,920
    │    │    └─BatchNorm2d: 3-26            [100, 160, 7, 7]          320
    │    └─Sequential: 2-20                  [100, 224, 7, 7]          --
    │    │    └─BasicConv2d: 3-27            [100, 112, 7, 7]          57,568
    │    │    └─BasicConv2d: 3-28            [100, 224, 7, 7]          226,240
    │    └─Sequential: 2-21                  [100, 64, 7, 7]           --
    │    │    └─BasicConv2d: 3-29            [100, 24, 7, 7]           12,336
    │    │    └─BasicConv2d: 3-30            [100, 64, 7, 7]           13,952
    │    └─Sequential: 2-22                  [100, 64, 7, 7]           --
    │    │    └─MaxPool2d: 3-31              [100, 512, 7, 7]          --
    │    │    └─BasicConv2d: 3-32            [100, 64, 7, 7]           32,896
    ├─Inception: 1-11                        [100, 512, 7, 7]          --
    │    └─BasicConv2d: 2-23                 [100, 128, 7, 7]          --
    │    │    └─Conv2d: 3-33                 [100, 128, 7, 7]          65,536
    │    │    └─BatchNorm2d: 3-34            [100, 128, 7, 7]          256
    │    └─Sequential: 2-24                  [100, 256, 7, 7]          --
    │    │    └─BasicConv2d: 3-35            [100, 128, 7, 7]          65,792
    │    │    └─BasicConv2d: 3-36            [100, 256, 7, 7]          295,424
    │    └─Sequential: 2-25                  [100, 64, 7, 7]           --
    │    │    └─BasicConv2d: 3-37            [100, 24, 7, 7]           12,336
    │    │    └─BasicConv2d: 3-38            [100, 64, 7, 7]           13,952
    │    └─Sequential: 2-26                  [100, 64, 7, 7]           --
    │    │    └─MaxPool2d: 3-39              [100, 512, 7, 7]          --
    │    │    └─BasicConv2d: 3-40            [100, 64, 7, 7]           32,896
    ├─Inception: 1-12                        [100, 528, 7, 7]          --
    │    └─BasicConv2d: 2-27                 [100, 112, 7, 7]          --
    │    │    └─Conv2d: 3-41                 [100, 112, 7, 7]          57,344
    │    │    └─BatchNorm2d: 3-42            [100, 112, 7, 7]          224
    │    └─Sequential: 2-28                  [100, 288, 7, 7]          --
    │    │    └─BasicConv2d: 3-43            [100, 144, 7, 7]          74,016
    │    │    └─BasicConv2d: 3-44            [100, 288, 7, 7]          373,824
    │    └─Sequential: 2-29                  [100, 64, 7, 7]           --
    │    │    └─BasicConv2d: 3-45            [100, 32, 7, 7]           16,448
    │    │    └─BasicConv2d: 3-46            [100, 64, 7, 7]           18,560
    │    └─Sequential: 2-30                  [100, 64, 7, 7]           --
    │    │    └─MaxPool2d: 3-47              [100, 512, 7, 7]          --
    │    │    └─BasicConv2d: 3-48            [100, 64, 7, 7]           32,896
    ├─Inception: 1-13                        [100, 832, 7, 7]          --
    │    └─BasicConv2d: 2-31                 [100, 256, 7, 7]          --
    │    │    └─Conv2d: 3-49                 [100, 256, 7, 7]          135,168
    │    │    └─BatchNorm2d: 3-50            [100, 256, 7, 7]          512
    │    └─Sequential: 2-32                  [100, 320, 7, 7]          --
    │    │    └─BasicConv2d: 3-51            [100, 160, 7, 7]          84,800
    │    │    └─BasicConv2d: 3-52            [100, 320, 7, 7]          461,440
    │    └─Sequential: 2-33                  [100, 128, 7, 7]          --
    │    │    └─BasicConv2d: 3-53            [100, 32, 7, 7]           16,960
    │    │    └─BasicConv2d: 3-54            [100, 128, 7, 7]          37,120
    │    └─Sequential: 2-34                  [100, 128, 7, 7]          --
    │    │    └─MaxPool2d: 3-55              [100, 528, 7, 7]          --
    │    │    └─BasicConv2d: 3-56            [100, 128, 7, 7]          67,840
    ├─MaxPool2d: 1-14                        [100, 832, 4, 4]          --
    ├─Inception: 1-15                        [100, 832, 4, 4]          --
    │    └─BasicConv2d: 2-35                 [100, 256, 4, 4]          --
    │    │    └─Conv2d: 3-57                 [100, 256, 4, 4]          212,992
    │    │    └─BatchNorm2d: 3-58            [100, 256, 4, 4]          512
    │    └─Sequential: 2-36                  [100, 320, 4, 4]          --
    │    │    └─BasicConv2d: 3-59            [100, 160, 4, 4]          133,440
    │    │    └─BasicConv2d: 3-60            [100, 320, 4, 4]          461,440
    │    └─Sequential: 2-37                  [100, 128, 4, 4]          --
    │    │    └─BasicConv2d: 3-61            [100, 32, 4, 4]           26,688
    │    │    └─BasicConv2d: 3-62            [100, 128, 4, 4]          37,120
    │    └─Sequential: 2-38                  [100, 128, 4, 4]          --
    │    │    └─MaxPool2d: 3-63              [100, 832, 4, 4]          --
    │    │    └─BasicConv2d: 3-64            [100, 128, 4, 4]          106,752
    ├─Inception: 1-16                        [100, 1024, 4, 4]         --
    │    └─BasicConv2d: 2-39                 [100, 384, 4, 4]          --
    │    │    └─Conv2d: 3-65                 [100, 384, 4, 4]          319,488
    │    │    └─BatchNorm2d: 3-66            [100, 384, 4, 4]          768
    │    └─Sequential: 2-40                  [100, 384, 4, 4]          --
    │    │    └─BasicConv2d: 3-67            [100, 192, 4, 4]          160,128
    │    │    └─BasicConv2d: 3-68            [100, 384, 4, 4]          664,320
    │    └─Sequential: 2-41                  [100, 128, 4, 4]          --
    │    │    └─BasicConv2d: 3-69            [100, 48, 4, 4]           40,032
    │    │    └─BasicConv2d: 3-70            [100, 128, 4, 4]          55,552
    │    └─Sequential: 2-42                  [100, 128, 4, 4]          --
    │    │    └─MaxPool2d: 3-71              [100, 832, 4, 4]          --
    │    │    └─BasicConv2d: 3-72            [100, 128, 4, 4]          106,752
    ├─AdaptiveAvgPool2d: 1-17                [100, 1024, 1, 1]         --
    ├─Dropout: 1-18                          [100, 1024]               --
    ├─Linear: 1-19                           [100, 1000]               1,025,000
    ==========================================================================================
    Total params: 6,624,904
    Trainable params: 6,624,904
    Non-trainable params: 0
    Total mult-adds (G): 38.41
    ==========================================================================================
    Input size (MB): 15.05
    Forward/backward pass size (MB): 1304.99
    Params size (MB): 26.50
    Estimated Total Size (MB): 1346.54
    ==========================================================================================




```python
print(net.fc)
print(net.fc.in_features)
print(net.fc.out_features)
```

    Linear(in_features=1024, out_features=1000, bias=True)
    1024
    1000



```python
torch_seed()
in_features = net.fc.in_features
net.fc = nn.Linear(in_features, n_output)
```


```python
# 모델 개요 표시 1
print(net)
```

    GoogLeNet(
      (conv1): BasicConv2d(
        (conv): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
      (maxpool1): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
      (conv2): BasicConv2d(
        (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
      (conv3): BasicConv2d(
        (conv): Conv2d(64, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
      (maxpool2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
      (inception3a): Inception(
        (branch1): BasicConv2d(
          (conv): Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch2): Sequential(
          (0): BasicConv2d(
            (conv): Conv2d(192, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): BasicConv2d(
            (conv): Conv2d(96, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (branch3): Sequential(
          (0): BasicConv2d(
            (conv): Conv2d(192, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(16, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): BasicConv2d(
            (conv): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (branch4): Sequential(
          (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)
          (1): BasicConv2d(
            (conv): Conv2d(192, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
      )
      (inception3b): Inception(
        (branch1): BasicConv2d(
          (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch2): Sequential(
          (0): BasicConv2d(
            (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): BasicConv2d(
            (conv): Conv2d(128, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (branch3): Sequential(
          (0): BasicConv2d(
            (conv): Conv2d(256, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): BasicConv2d(
            (conv): Conv2d(32, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (branch4): Sequential(
          (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)
          (1): BasicConv2d(
            (conv): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
      )
      (maxpool3): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
      (inception4a): Inception(
        (branch1): BasicConv2d(
          (conv): Conv2d(480, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch2): Sequential(
          (0): BasicConv2d(
            (conv): Conv2d(480, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): BasicConv2d(
            (conv): Conv2d(96, 208, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(208, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (branch3): Sequential(
          (0): BasicConv2d(
            (conv): Conv2d(480, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(16, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): BasicConv2d(
            (conv): Conv2d(16, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(48, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (branch4): Sequential(
          (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)
          (1): BasicConv2d(
            (conv): Conv2d(480, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
      )
      (inception4b): Inception(
        (branch1): BasicConv2d(
          (conv): Conv2d(512, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch2): Sequential(
          (0): BasicConv2d(
            (conv): Conv2d(512, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(112, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): BasicConv2d(
            (conv): Conv2d(112, 224, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(224, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (branch3): Sequential(
          (0): BasicConv2d(
            (conv): Conv2d(512, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(24, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): BasicConv2d(
            (conv): Conv2d(24, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (branch4): Sequential(
          (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)
          (1): BasicConv2d(
            (conv): Conv2d(512, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
      )
      (inception4c): Inception(
        (branch1): BasicConv2d(
          (conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch2): Sequential(
          (0): BasicConv2d(
            (conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): BasicConv2d(
            (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(256, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (branch3): Sequential(
          (0): BasicConv2d(
            (conv): Conv2d(512, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(24, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): BasicConv2d(
            (conv): Conv2d(24, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (branch4): Sequential(
          (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)
          (1): BasicConv2d(
            (conv): Conv2d(512, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
      )
      (inception4d): Inception(
        (branch1): BasicConv2d(
          (conv): Conv2d(512, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(112, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch2): Sequential(
          (0): BasicConv2d(
            (conv): Conv2d(512, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(144, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): BasicConv2d(
            (conv): Conv2d(144, 288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(288, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (branch3): Sequential(
          (0): BasicConv2d(
            (conv): Conv2d(512, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): BasicConv2d(
            (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (branch4): Sequential(
          (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)
          (1): BasicConv2d(
            (conv): Conv2d(512, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
      )
      (inception4e): Inception(
        (branch1): BasicConv2d(
          (conv): Conv2d(528, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(256, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch2): Sequential(
          (0): BasicConv2d(
            (conv): Conv2d(528, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): BasicConv2d(
            (conv): Conv2d(160, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(320, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (branch3): Sequential(
          (0): BasicConv2d(
            (conv): Conv2d(528, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): BasicConv2d(
            (conv): Conv2d(32, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (branch4): Sequential(
          (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)
          (1): BasicConv2d(
            (conv): Conv2d(528, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
      )
      (maxpool4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)
      (inception5a): Inception(
        (branch1): BasicConv2d(
          (conv): Conv2d(832, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(256, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch2): Sequential(
          (0): BasicConv2d(
            (conv): Conv2d(832, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): BasicConv2d(
            (conv): Conv2d(160, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(320, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (branch3): Sequential(
          (0): BasicConv2d(
            (conv): Conv2d(832, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): BasicConv2d(
            (conv): Conv2d(32, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (branch4): Sequential(
          (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)
          (1): BasicConv2d(
            (conv): Conv2d(832, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
      )
      (inception5b): Inception(
        (branch1): BasicConv2d(
          (conv): Conv2d(832, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch2): Sequential(
          (0): BasicConv2d(
            (conv): Conv2d(832, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): BasicConv2d(
            (conv): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (branch3): Sequential(
          (0): BasicConv2d(
            (conv): Conv2d(832, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(48, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): BasicConv2d(
            (conv): Conv2d(48, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (branch4): Sequential(
          (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)
          (1): BasicConv2d(
            (conv): Conv2d(832, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
      )
      (aux1): None
      (aux2): None
      (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
      (dropout): Dropout(p=0.2, inplace=False)
      (fc): Linear(in_features=1024, out_features=10, bias=True)
    )



```python
# 손실 계산 그래프 시각화
net = net.to(device)
criterion = nn.CrossEntropyLoss()
loss = eval_loss(test_loader, device, net, criterion)
g = make_dot(loss, params=dict(net.named_parameters()))
display(g)
```


    
*(이미지 없음 / image not preserved: output_60_0.svg)*
    


## 학습과 결과 평가

### 초기 설정


```python
# 난수 고정
torch_seed()

# 사전 학습 모델 불러오기
# pretraind = True로 학습을 마친 파라미터도 함께 불러오기
weights = models.GoogLeNet_Weights.IMAGENET1K_V1
net = models.googlenet(weights=weights)

# 최종 레이어 함수 입력 차원수 확인
in_features = net.fc.in_features
net.fc = nn.Linear(in_features, n_output)

# 최종 레이어 함수 교체
net.fc = nn.Linear(in_features, n_output)

# GPU 사용
net = net.to(device)

# 학습률
lr = 0.001

# 손실 함수 정의
criterion = nn.CrossEntropyLoss()

# 최적화 함수 정의
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

# history 파일 초기화
history = np.zeros((0, 5))
```

### 학습


```python
# 학습
num_epochs = 5
history = fit(net, optimizer, criterion, num_epochs, 
        train_loader, test_loader, device, history)
```


      0%|          | 0/1000 [00:00<?, ?it/s]


    Epoch [1/5], loss: 0.82228 acc: 0.72988 val_loss: 0.31814, val_acc: 0.89200



      0%|          | 0/1000 [00:00<?, ?it/s]


    Epoch [2/5], loss: 0.41222 acc: 0.86000 val_loss: 0.23791, val_acc: 0.91640



      0%|          | 0/1000 [00:00<?, ?it/s]


    Epoch [3/5], loss: 0.33283 acc: 0.88782 val_loss: 0.20396, val_acc: 0.92910



      0%|          | 0/1000 [00:00<?, ?it/s]


    Epoch [4/5], loss: 0.28708 acc: 0.90082 val_loss: 0.19425, val_acc: 0.93320



      0%|          | 0/1000 [00:00<?, ?it/s]


    Epoch [5/5], loss: 0.24786 acc: 0.91622 val_loss: 0.17743, val_acc: 0.94010


### 학습 결과 평가


```python
# 결과 요약
evaluate_history(history)
```

    초기상태 : 손실 : 0.31814  정확도 : 0.89200
    최종상태 : 손실 : 0.17743 정확도 : 0.94010



    
*(이미지 없음 / image not preserved: output_67_1.png)*
    



    
*(이미지 없음 / image not preserved: output_67_2.png)*
    



```python
# 이미지와 정답, 예측 결과를 함께 표시
show_images_labels(test_loader, classes, net, device)
```

    len(images) =  50



    
*(이미지 없음 / image not preserved: output_68_1.png)*
    

