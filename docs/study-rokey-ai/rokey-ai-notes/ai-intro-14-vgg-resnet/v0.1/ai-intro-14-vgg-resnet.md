# 14장 영상 분류 사전 학습 모델 활용하기 2 (ResNet, VGG-19)

> **Migrated from Quartz wiki** (`wiki/study/rokey-ai/basic/ai-intro-14-vgg_resnet.md`, updated 2026-09-26)  
> tags: rokey, ai, notebook, basic, study  
> source: raw/2026-09-26-rokey-ai/basic/강의_3기_AI개론_14차시__VGG_ResNet_.ipynb  
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
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
```


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


# 공통 함수 확인
print(README)
```

    Common Library for PyTorch
    Author: M. Akaishi


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
    root = data_root, 
    train = True,
    download = True, 
    transform = transform_train)

# 검증 데이터셋
test_set = datasets.CIFAR10(
    root = data_root, 
    train = False, 
    download = True, 
    transform = transform)
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

## ResNet18 불러오기

### 모델 불러오기


```python
#  라이브러리 임포트
from torchvision import models


# 사전 학습 모델 불러오기
weights = models.ResNet18_Weights.IMAGENET1K_V1
net = models.resnet18(weights = weights)
# pretraind = True로 학습을 마친 파라미터를 동시에 불러오기
# net = models.resnet18(pretrained = True)
```

### 모델 구조 확인


```python
# 모델 개요 표시 1

print(net)
# net.layer1[0].bn1
```

    ResNet(
      (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
      (layer1): Sequential(
        (0): BasicBlock(
          (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (1): BasicBlock(
          (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (layer2): Sequential(
        (0): BasicBlock(
          (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (downsample): Sequential(
            (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): BasicBlock(
          (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (layer3): Sequential(
        (0): BasicBlock(
          (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (downsample): Sequential(
            (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): BasicBlock(
          (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (layer4): Sequential(
        (0): BasicBlock(
          (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (downsample): Sequential(
            (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): BasicBlock(
          (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
      (fc): Linear(in_features=512, out_features=1000, bias=True)
    )



```python
# 모델 개요 표시 2
net = net.to(device)
summary(net,(100,3,112,112))
```




    ==========================================================================================
    Layer (type:depth-idx)                   Output Shape              Param #
    ==========================================================================================
    ResNet                                   [100, 1000]               --
    ├─Conv2d: 1-1                            [100, 64, 56, 56]         9,408
    ├─BatchNorm2d: 1-2                       [100, 64, 56, 56]         128
    ├─ReLU: 1-3                              [100, 64, 56, 56]         --
    ├─MaxPool2d: 1-4                         [100, 64, 28, 28]         --
    ├─Sequential: 1-5                        [100, 64, 28, 28]         --
    │    └─BasicBlock: 2-1                   [100, 64, 28, 28]         --
    │    │    └─Conv2d: 3-1                  [100, 64, 28, 28]         36,864
    │    │    └─BatchNorm2d: 3-2             [100, 64, 28, 28]         128
    │    │    └─ReLU: 3-3                    [100, 64, 28, 28]         --
    │    │    └─Conv2d: 3-4                  [100, 64, 28, 28]         36,864
    │    │    └─BatchNorm2d: 3-5             [100, 64, 28, 28]         128
    │    │    └─ReLU: 3-6                    [100, 64, 28, 28]         --
    │    └─BasicBlock: 2-2                   [100, 64, 28, 28]         --
    │    │    └─Conv2d: 3-7                  [100, 64, 28, 28]         36,864
    │    │    └─BatchNorm2d: 3-8             [100, 64, 28, 28]         128
    │    │    └─ReLU: 3-9                    [100, 64, 28, 28]         --
    │    │    └─Conv2d: 3-10                 [100, 64, 28, 28]         36,864
    │    │    └─BatchNorm2d: 3-11            [100, 64, 28, 28]         128
    │    │    └─ReLU: 3-12                   [100, 64, 28, 28]         --
    ├─Sequential: 1-6                        [100, 128, 14, 14]        --
    │    └─BasicBlock: 2-3                   [100, 128, 14, 14]        --
    │    │    └─Conv2d: 3-13                 [100, 128, 14, 14]        73,728
    │    │    └─BatchNorm2d: 3-14            [100, 128, 14, 14]        256
    │    │    └─ReLU: 3-15                   [100, 128, 14, 14]        --
    │    │    └─Conv2d: 3-16                 [100, 128, 14, 14]        147,456
    │    │    └─BatchNorm2d: 3-17            [100, 128, 14, 14]        256
    │    │    └─Sequential: 3-18             [100, 128, 14, 14]        8,448
    │    │    └─ReLU: 3-19                   [100, 128, 14, 14]        --
    │    └─BasicBlock: 2-4                   [100, 128, 14, 14]        --
    │    │    └─Conv2d: 3-20                 [100, 128, 14, 14]        147,456
    │    │    └─BatchNorm2d: 3-21            [100, 128, 14, 14]        256
    │    │    └─ReLU: 3-22                   [100, 128, 14, 14]        --
    │    │    └─Conv2d: 3-23                 [100, 128, 14, 14]        147,456
    │    │    └─BatchNorm2d: 3-24            [100, 128, 14, 14]        256
    │    │    └─ReLU: 3-25                   [100, 128, 14, 14]        --
    ├─Sequential: 1-7                        [100, 256, 7, 7]          --
    │    └─BasicBlock: 2-5                   [100, 256, 7, 7]          --
    │    │    └─Conv2d: 3-26                 [100, 256, 7, 7]          294,912
    │    │    └─BatchNorm2d: 3-27            [100, 256, 7, 7]          512
    │    │    └─ReLU: 3-28                   [100, 256, 7, 7]          --
    │    │    └─Conv2d: 3-29                 [100, 256, 7, 7]          589,824
    │    │    └─BatchNorm2d: 3-30            [100, 256, 7, 7]          512
    │    │    └─Sequential: 3-31             [100, 256, 7, 7]          33,280
    │    │    └─ReLU: 3-32                   [100, 256, 7, 7]          --
    │    └─BasicBlock: 2-6                   [100, 256, 7, 7]          --
    │    │    └─Conv2d: 3-33                 [100, 256, 7, 7]          589,824
    │    │    └─BatchNorm2d: 3-34            [100, 256, 7, 7]          512
    │    │    └─ReLU: 3-35                   [100, 256, 7, 7]          --
    │    │    └─Conv2d: 3-36                 [100, 256, 7, 7]          589,824
    │    │    └─BatchNorm2d: 3-37            [100, 256, 7, 7]          512
    │    │    └─ReLU: 3-38                   [100, 256, 7, 7]          --
    ├─Sequential: 1-8                        [100, 512, 4, 4]          --
    │    └─BasicBlock: 2-7                   [100, 512, 4, 4]          --
    │    │    └─Conv2d: 3-39                 [100, 512, 4, 4]          1,179,648
    │    │    └─BatchNorm2d: 3-40            [100, 512, 4, 4]          1,024
    │    │    └─ReLU: 3-41                   [100, 512, 4, 4]          --
    │    │    └─Conv2d: 3-42                 [100, 512, 4, 4]          2,359,296
    │    │    └─BatchNorm2d: 3-43            [100, 512, 4, 4]          1,024
    │    │    └─Sequential: 3-44             [100, 512, 4, 4]          132,096
    │    │    └─ReLU: 3-45                   [100, 512, 4, 4]          --
    │    └─BasicBlock: 2-8                   [100, 512, 4, 4]          --
    │    │    └─Conv2d: 3-46                 [100, 512, 4, 4]          2,359,296
    │    │    └─BatchNorm2d: 3-47            [100, 512, 4, 4]          1,024
    │    │    └─ReLU: 3-48                   [100, 512, 4, 4]          --
    │    │    └─Conv2d: 3-49                 [100, 512, 4, 4]          2,359,296
    │    │    └─BatchNorm2d: 3-50            [100, 512, 4, 4]          1,024
    │    │    └─ReLU: 3-51                   [100, 512, 4, 4]          --
    ├─AdaptiveAvgPool2d: 1-9                 [100, 512, 1, 1]          --
    ├─Linear: 1-10                           [100, 1000]               513,000
    ==========================================================================================
    Total params: 11,689,512
    Trainable params: 11,689,512
    Non-trainable params: 0
    Total mult-adds (G): 48.54
    ==========================================================================================
    Input size (MB): 15.05
    Forward/backward pass size (MB): 1009.64
    Params size (MB): 46.76
    Estimated Total Size (MB): 1071.46
    ==========================================================================================




```python
print(net.fc)
print(net.fc.in_features)
print(net.fc.out_features)
```

    Linear(in_features=512, out_features=1000, bias=True)
    512
    1000


## 최종 레이어 함수 교체하기


```python
# 난수 고정
torch_seed()

# 최종 레이어 함수의 입력 차원수 확인
fc_in_features = net.fc.in_features

# 최종 레이어 함수 교체
net.fc = nn.Linear(fc_in_features, n_output)
```


```python
# 모델 개요 표시 1
print(net)
```

    ResNet(
      (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
      (layer1): Sequential(
        (0): BasicBlock(
          (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (1): BasicBlock(
          (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (layer2): Sequential(
        (0): BasicBlock(
          (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (downsample): Sequential(
            (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): BasicBlock(
          (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (layer3): Sequential(
        (0): BasicBlock(
          (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (downsample): Sequential(
            (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): BasicBlock(
          (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (layer4): Sequential(
        (0): BasicBlock(
          (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (downsample): Sequential(
            (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): BasicBlock(
          (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
      (fc): Linear(in_features=512, out_features=10, bias=True)
    )



```python
# 모델 개요 표시 2

net = net.to(device)
summary(net,(100, 3, 224, 224))
```




    ==========================================================================================
    Layer (type:depth-idx)                   Output Shape              Param #
    ==========================================================================================
    ResNet                                   [100, 10]                 --
    ├─Conv2d: 1-1                            [100, 64, 112, 112]       9,408
    ├─BatchNorm2d: 1-2                       [100, 64, 112, 112]       128
    ├─ReLU: 1-3                              [100, 64, 112, 112]       --
    ├─MaxPool2d: 1-4                         [100, 64, 56, 56]         --
    ├─Sequential: 1-5                        [100, 64, 56, 56]         --
    │    └─BasicBlock: 2-1                   [100, 64, 56, 56]         --
    │    │    └─Conv2d: 3-1                  [100, 64, 56, 56]         36,864
    │    │    └─BatchNorm2d: 3-2             [100, 64, 56, 56]         128
    │    │    └─ReLU: 3-3                    [100, 64, 56, 56]         --
    │    │    └─Conv2d: 3-4                  [100, 64, 56, 56]         36,864
    │    │    └─BatchNorm2d: 3-5             [100, 64, 56, 56]         128
    │    │    └─ReLU: 3-6                    [100, 64, 56, 56]         --
    │    └─BasicBlock: 2-2                   [100, 64, 56, 56]         --
    │    │    └─Conv2d: 3-7                  [100, 64, 56, 56]         36,864
    │    │    └─BatchNorm2d: 3-8             [100, 64, 56, 56]         128
    │    │    └─ReLU: 3-9                    [100, 64, 56, 56]         --
    │    │    └─Conv2d: 3-10                 [100, 64, 56, 56]         36,864
    │    │    └─BatchNorm2d: 3-11            [100, 64, 56, 56]         128
    │    │    └─ReLU: 3-12                   [100, 64, 56, 56]         --
    ├─Sequential: 1-6                        [100, 128, 28, 28]        --
    │    └─BasicBlock: 2-3                   [100, 128, 28, 28]        --
    │    │    └─Conv2d: 3-13                 [100, 128, 28, 28]        73,728
    │    │    └─BatchNorm2d: 3-14            [100, 128, 28, 28]        256
    │    │    └─ReLU: 3-15                   [100, 128, 28, 28]        --
    │    │    └─Conv2d: 3-16                 [100, 128, 28, 28]        147,456
    │    │    └─BatchNorm2d: 3-17            [100, 128, 28, 28]        256
    │    │    └─Sequential: 3-18             [100, 128, 28, 28]        8,448
    │    │    └─ReLU: 3-19                   [100, 128, 28, 28]        --
    │    └─BasicBlock: 2-4                   [100, 128, 28, 28]        --
    │    │    └─Conv2d: 3-20                 [100, 128, 28, 28]        147,456
    │    │    └─BatchNorm2d: 3-21            [100, 128, 28, 28]        256
    │    │    └─ReLU: 3-22                   [100, 128, 28, 28]        --
    │    │    └─Conv2d: 3-23                 [100, 128, 28, 28]        147,456
    │    │    └─BatchNorm2d: 3-24            [100, 128, 28, 28]        256
    │    │    └─ReLU: 3-25                   [100, 128, 28, 28]        --
    ├─Sequential: 1-7                        [100, 256, 14, 14]        --
    │    └─BasicBlock: 2-5                   [100, 256, 14, 14]        --
    │    │    └─Conv2d: 3-26                 [100, 256, 14, 14]        294,912
    │    │    └─BatchNorm2d: 3-27            [100, 256, 14, 14]        512
    │    │    └─ReLU: 3-28                   [100, 256, 14, 14]        --
    │    │    └─Conv2d: 3-29                 [100, 256, 14, 14]        589,824
    │    │    └─BatchNorm2d: 3-30            [100, 256, 14, 14]        512
    │    │    └─Sequential: 3-31             [100, 256, 14, 14]        33,280
    │    │    └─ReLU: 3-32                   [100, 256, 14, 14]        --
    │    └─BasicBlock: 2-6                   [100, 256, 14, 14]        --
    │    │    └─Conv2d: 3-33                 [100, 256, 14, 14]        589,824
    │    │    └─BatchNorm2d: 3-34            [100, 256, 14, 14]        512
    │    │    └─ReLU: 3-35                   [100, 256, 14, 14]        --
    │    │    └─Conv2d: 3-36                 [100, 256, 14, 14]        589,824
    │    │    └─BatchNorm2d: 3-37            [100, 256, 14, 14]        512
    │    │    └─ReLU: 3-38                   [100, 256, 14, 14]        --
    ├─Sequential: 1-8                        [100, 512, 7, 7]          --
    │    └─BasicBlock: 2-7                   [100, 512, 7, 7]          --
    │    │    └─Conv2d: 3-39                 [100, 512, 7, 7]          1,179,648
    │    │    └─BatchNorm2d: 3-40            [100, 512, 7, 7]          1,024
    │    │    └─ReLU: 3-41                   [100, 512, 7, 7]          --
    │    │    └─Conv2d: 3-42                 [100, 512, 7, 7]          2,359,296
    │    │    └─BatchNorm2d: 3-43            [100, 512, 7, 7]          1,024
    │    │    └─Sequential: 3-44             [100, 512, 7, 7]          132,096
    │    │    └─ReLU: 3-45                   [100, 512, 7, 7]          --
    │    └─BasicBlock: 2-8                   [100, 512, 7, 7]          --
    │    │    └─Conv2d: 3-46                 [100, 512, 7, 7]          2,359,296
    │    │    └─BatchNorm2d: 3-47            [100, 512, 7, 7]          1,024
    │    │    └─ReLU: 3-48                   [100, 512, 7, 7]          --
    │    │    └─Conv2d: 3-49                 [100, 512, 7, 7]          2,359,296
    │    │    └─BatchNorm2d: 3-50            [100, 512, 7, 7]          1,024
    │    │    └─ReLU: 3-51                   [100, 512, 7, 7]          --
    ├─AdaptiveAvgPool2d: 1-9                 [100, 512, 1, 1]          --
    ├─Linear: 1-10                           [100, 10]                 5,130
    ==========================================================================================
    Total params: 11,181,642
    Trainable params: 11,181,642
    Non-trainable params: 0
    Total mult-adds (G): 181.36
    ==========================================================================================
    Input size (MB): 60.21
    Forward/backward pass size (MB): 3973.95
    Params size (MB): 44.73
    Estimated Total Size (MB): 4078.89
    ==========================================================================================




```python
# 손실 계산 그래프 시각화

criterion = nn.CrossEntropyLoss()
loss = eval_loss(test_loader, device, net, criterion)
g = make_dot(loss, params=dict(net.named_parameters()))
display(g)
```


    
*(이미지 없음 / image not preserved: output_27_0.svg)*
    



```python
# 모델 개요 표시 1
print(net)
```

    ResNet(
      (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
      (layer1): Sequential(
        (0): BasicBlock(
          (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (1): BasicBlock(
          (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (layer2): Sequential(
        (0): BasicBlock(
          (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (downsample): Sequential(
            (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): BasicBlock(
          (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (layer3): Sequential(
        (0): BasicBlock(
          (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (downsample): Sequential(
            (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): BasicBlock(
          (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (layer4): Sequential(
        (0): BasicBlock(
          (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (downsample): Sequential(
            (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): BasicBlock(
          (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
      (fc): Linear(in_features=512, out_features=10, bias=True)
    )



```python
# 모델 개요 표시 2
net = net.to(device)
summary(net,(100,3,112,112))
```




    ==========================================================================================
    Layer (type:depth-idx)                   Output Shape              Param #
    ==========================================================================================
    ResNet                                   [100, 10]                 --
    ├─Conv2d: 1-1                            [100, 64, 56, 56]         9,408
    ├─BatchNorm2d: 1-2                       [100, 64, 56, 56]         128
    ├─ReLU: 1-3                              [100, 64, 56, 56]         --
    ├─MaxPool2d: 1-4                         [100, 64, 28, 28]         --
    ├─Sequential: 1-5                        [100, 64, 28, 28]         --
    │    └─BasicBlock: 2-1                   [100, 64, 28, 28]         --
    │    │    └─Conv2d: 3-1                  [100, 64, 28, 28]         36,864
    │    │    └─BatchNorm2d: 3-2             [100, 64, 28, 28]         128
    │    │    └─ReLU: 3-3                    [100, 64, 28, 28]         --
    │    │    └─Conv2d: 3-4                  [100, 64, 28, 28]         36,864
    │    │    └─BatchNorm2d: 3-5             [100, 64, 28, 28]         128
    │    │    └─ReLU: 3-6                    [100, 64, 28, 28]         --
    │    └─BasicBlock: 2-2                   [100, 64, 28, 28]         --
    │    │    └─Conv2d: 3-7                  [100, 64, 28, 28]         36,864
    │    │    └─BatchNorm2d: 3-8             [100, 64, 28, 28]         128
    │    │    └─ReLU: 3-9                    [100, 64, 28, 28]         --
    │    │    └─Conv2d: 3-10                 [100, 64, 28, 28]         36,864
    │    │    └─BatchNorm2d: 3-11            [100, 64, 28, 28]         128
    │    │    └─ReLU: 3-12                   [100, 64, 28, 28]         --
    ├─Sequential: 1-6                        [100, 128, 14, 14]        --
    │    └─BasicBlock: 2-3                   [100, 128, 14, 14]        --
    │    │    └─Conv2d: 3-13                 [100, 128, 14, 14]        73,728
    │    │    └─BatchNorm2d: 3-14            [100, 128, 14, 14]        256
    │    │    └─ReLU: 3-15                   [100, 128, 14, 14]        --
    │    │    └─Conv2d: 3-16                 [100, 128, 14, 14]        147,456
    │    │    └─BatchNorm2d: 3-17            [100, 128, 14, 14]        256
    │    │    └─Sequential: 3-18             [100, 128, 14, 14]        8,448
    │    │    └─ReLU: 3-19                   [100, 128, 14, 14]        --
    │    └─BasicBlock: 2-4                   [100, 128, 14, 14]        --
    │    │    └─Conv2d: 3-20                 [100, 128, 14, 14]        147,456
    │    │    └─BatchNorm2d: 3-21            [100, 128, 14, 14]        256
    │    │    └─ReLU: 3-22                   [100, 128, 14, 14]        --
    │    │    └─Conv2d: 3-23                 [100, 128, 14, 14]        147,456
    │    │    └─BatchNorm2d: 3-24            [100, 128, 14, 14]        256
    │    │    └─ReLU: 3-25                   [100, 128, 14, 14]        --
    ├─Sequential: 1-7                        [100, 256, 7, 7]          --
    │    └─BasicBlock: 2-5                   [100, 256, 7, 7]          --
    │    │    └─Conv2d: 3-26                 [100, 256, 7, 7]          294,912
    │    │    └─BatchNorm2d: 3-27            [100, 256, 7, 7]          512
    │    │    └─ReLU: 3-28                   [100, 256, 7, 7]          --
    │    │    └─Conv2d: 3-29                 [100, 256, 7, 7]          589,824
    │    │    └─BatchNorm2d: 3-30            [100, 256, 7, 7]          512
    │    │    └─Sequential: 3-31             [100, 256, 7, 7]          33,280
    │    │    └─ReLU: 3-32                   [100, 256, 7, 7]          --
    │    └─BasicBlock: 2-6                   [100, 256, 7, 7]          --
    │    │    └─Conv2d: 3-33                 [100, 256, 7, 7]          589,824
    │    │    └─BatchNorm2d: 3-34            [100, 256, 7, 7]          512
    │    │    └─ReLU: 3-35                   [100, 256, 7, 7]          --
    │    │    └─Conv2d: 3-36                 [100, 256, 7, 7]          589,824
    │    │    └─BatchNorm2d: 3-37            [100, 256, 7, 7]          512
    │    │    └─ReLU: 3-38                   [100, 256, 7, 7]          --
    ├─Sequential: 1-8                        [100, 512, 4, 4]          --
    │    └─BasicBlock: 2-7                   [100, 512, 4, 4]          --
    │    │    └─Conv2d: 3-39                 [100, 512, 4, 4]          1,179,648
    │    │    └─BatchNorm2d: 3-40            [100, 512, 4, 4]          1,024
    │    │    └─ReLU: 3-41                   [100, 512, 4, 4]          --
    │    │    └─Conv2d: 3-42                 [100, 512, 4, 4]          2,359,296
    │    │    └─BatchNorm2d: 3-43            [100, 512, 4, 4]          1,024
    │    │    └─Sequential: 3-44             [100, 512, 4, 4]          132,096
    │    │    └─ReLU: 3-45                   [100, 512, 4, 4]          --
    │    └─BasicBlock: 2-8                   [100, 512, 4, 4]          --
    │    │    └─Conv2d: 3-46                 [100, 512, 4, 4]          2,359,296
    │    │    └─BatchNorm2d: 3-47            [100, 512, 4, 4]          1,024
    │    │    └─ReLU: 3-48                   [100, 512, 4, 4]          --
    │    │    └─Conv2d: 3-49                 [100, 512, 4, 4]          2,359,296
    │    │    └─BatchNorm2d: 3-50            [100, 512, 4, 4]          1,024
    │    │    └─ReLU: 3-51                   [100, 512, 4, 4]          --
    ├─AdaptiveAvgPool2d: 1-9                 [100, 512, 1, 1]          --
    ├─Linear: 1-10                           [100, 10]                 5,130
    ==========================================================================================
    Total params: 11,181,642
    Trainable params: 11,181,642
    Non-trainable params: 0
    Total mult-adds (G): 48.49
    ==========================================================================================
    Input size (MB): 15.05
    Forward/backward pass size (MB): 1008.85
    Params size (MB): 44.73
    Estimated Total Size (MB): 1068.63
    ==========================================================================================



## 학습과 결과 평가

### 초기 설정


```python
# 난수 고정
torch_seed()

# 사전 학습 모델 불러오기
weights = models.ResNet18_Weights.IMAGENET1K_V1
net = models.resnet18(weights = weights)

# 최종 레이어 함수 입력 차원수 확인
fc_in_features = net.fc.in_features

# 최종 레이어 함수 교체
net.fc = nn.Linear(fc_in_features, n_output)

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


    Epoch [1/5], loss: 0.60050 acc: 0.79544 val_loss: 0.30019, val_acc: 0.89720



      0%|          | 0/1000 [00:00<?, ?it/s]


    Epoch [2/5], loss: 0.32100 acc: 0.88816 val_loss: 0.23029, val_acc: 0.92010



      0%|          | 0/1000 [00:00<?, ?it/s]


    Epoch [3/5], loss: 0.25802 acc: 0.91102 val_loss: 0.19530, val_acc: 0.93400



      0%|          | 0/1000 [00:00<?, ?it/s]


    Epoch [4/5], loss: 0.21796 acc: 0.92298 val_loss: 0.17268, val_acc: 0.94230



      0%|          | 0/1000 [00:00<?, ?it/s]


    Epoch [5/5], loss: 0.19113 acc: 0.93398 val_loss: 0.17815, val_acc: 0.94070


### 학습 결과 평가


```python
# 결과 요약
evaluate_history(history)
```

    초기상태 : 손실 : 0.30019  정확도 : 0.89720
    최종상태 : 손실 : 0.17815 정확도 : 0.94070



    
*(이미지 없음 / image not preserved: output_36_1.png)*
    



    
*(이미지 없음 / image not preserved: output_36_2.png)*
    



```python
# 이미지와 정답, 예측 결과를 함께 표시
show_images_labels(test_loader, classes, net, device)
```

    len(images) =  50



    
*(이미지 없음 / image not preserved: output_37_1.png)*
    


## VGG-19-BN 활용하기

### 모델 불러오기


```python
# 사전 학습 모델 불러오기
from torchvision import models

weights = models.VGG19_BN_Weights.DEFAULT
net = models.vgg19_bn(weights = weights)
```

### 모델 구조 확인


```python
# 모델 개요 표시 1
print(net)
```

    VGG(
      (features): Sequential(
        (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU(inplace=True)
        (6): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (7): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (8): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (9): ReLU(inplace=True)
        (10): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (11): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (12): ReLU(inplace=True)
        (13): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (14): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (15): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (16): ReLU(inplace=True)
        (17): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (18): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (19): ReLU(inplace=True)
        (20): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (21): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (22): ReLU(inplace=True)
        (23): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (24): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (25): ReLU(inplace=True)
        (26): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (27): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (28): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (29): ReLU(inplace=True)
        (30): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (31): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (32): ReLU(inplace=True)
        (33): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (34): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (35): ReLU(inplace=True)
        (36): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (37): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (38): ReLU(inplace=True)
        (39): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (40): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (41): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (42): ReLU(inplace=True)
        (43): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (44): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (45): ReLU(inplace=True)
        (46): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (47): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (48): ReLU(inplace=True)
        (49): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (50): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (51): ReLU(inplace=True)
        (52): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
      (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
      (classifier): Sequential(
        (0): Linear(in_features=25088, out_features=4096, bias=True)
        (1): ReLU(inplace=True)
        (2): Dropout(p=0.5, inplace=False)
        (3): Linear(in_features=4096, out_features=4096, bias=True)
        (4): ReLU(inplace=True)
        (5): Dropout(p=0.5, inplace=False)
        (6): Linear(in_features=4096, out_features=1000, bias=True)
      )
    )


최종 레이어 함수는``classifier[6]``임을 알 수 있다.


```python
# 최종 레이어 함수 확인
print(net.classifier[6])

```

    Linear(in_features=4096, out_features=1000, bias=True)


### 최종 레이어 함수 교체


```python
# 난수 고정
torch_seed()

# 최종 레이어 함수 교체
in_features = net.classifier[6].in_features
net.classifier[6] = nn.Linear(in_features, n_output)

# features 마지막의 MaxPool2d 제거
net.features = net.features[:-1]

# AdaptiveAvgPool2d 제거
net.avgpool = nn.Identity()
```


```python
# 모델 개요 표시 2
net = net.to(device)
summary(net,(100,3,112,112))
```




    ==========================================================================================
    Layer (type:depth-idx)                   Output Shape              Param #
    ==========================================================================================
    VGG                                      [100, 10]                 --
    ├─Sequential: 1-1                        [100, 512, 7, 7]          --
    │    └─Conv2d: 2-1                       [100, 64, 112, 112]       1,792
    │    └─BatchNorm2d: 2-2                  [100, 64, 112, 112]       128
    │    └─ReLU: 2-3                         [100, 64, 112, 112]       --
    │    └─Conv2d: 2-4                       [100, 64, 112, 112]       36,928
    │    └─BatchNorm2d: 2-5                  [100, 64, 112, 112]       128
    │    └─ReLU: 2-6                         [100, 64, 112, 112]       --
    │    └─MaxPool2d: 2-7                    [100, 64, 56, 56]         --
    │    └─Conv2d: 2-8                       [100, 128, 56, 56]        73,856
    │    └─BatchNorm2d: 2-9                  [100, 128, 56, 56]        256
    │    └─ReLU: 2-10                        [100, 128, 56, 56]        --
    │    └─Conv2d: 2-11                      [100, 128, 56, 56]        147,584
    │    └─BatchNorm2d: 2-12                 [100, 128, 56, 56]        256
    │    └─ReLU: 2-13                        [100, 128, 56, 56]        --
    │    └─MaxPool2d: 2-14                   [100, 128, 28, 28]        --
    │    └─Conv2d: 2-15                      [100, 256, 28, 28]        295,168
    │    └─BatchNorm2d: 2-16                 [100, 256, 28, 28]        512
    │    └─ReLU: 2-17                        [100, 256, 28, 28]        --
    │    └─Conv2d: 2-18                      [100, 256, 28, 28]        590,080
    │    └─BatchNorm2d: 2-19                 [100, 256, 28, 28]        512
    │    └─ReLU: 2-20                        [100, 256, 28, 28]        --
    │    └─Conv2d: 2-21                      [100, 256, 28, 28]        590,080
    │    └─BatchNorm2d: 2-22                 [100, 256, 28, 28]        512
    │    └─ReLU: 2-23                        [100, 256, 28, 28]        --
    │    └─Conv2d: 2-24                      [100, 256, 28, 28]        590,080
    │    └─BatchNorm2d: 2-25                 [100, 256, 28, 28]        512
    │    └─ReLU: 2-26                        [100, 256, 28, 28]        --
    │    └─MaxPool2d: 2-27                   [100, 256, 14, 14]        --
    │    └─Conv2d: 2-28                      [100, 512, 14, 14]        1,180,160
    │    └─BatchNorm2d: 2-29                 [100, 512, 14, 14]        1,024
    │    └─ReLU: 2-30                        [100, 512, 14, 14]        --
    │    └─Conv2d: 2-31                      [100, 512, 14, 14]        2,359,808
    │    └─BatchNorm2d: 2-32                 [100, 512, 14, 14]        1,024
    │    └─ReLU: 2-33                        [100, 512, 14, 14]        --
    │    └─Conv2d: 2-34                      [100, 512, 14, 14]        2,359,808
    │    └─BatchNorm2d: 2-35                 [100, 512, 14, 14]        1,024
    │    └─ReLU: 2-36                        [100, 512, 14, 14]        --
    │    └─Conv2d: 2-37                      [100, 512, 14, 14]        2,359,808
    │    └─BatchNorm2d: 2-38                 [100, 512, 14, 14]        1,024
    │    └─ReLU: 2-39                        [100, 512, 14, 14]        --
    │    └─MaxPool2d: 2-40                   [100, 512, 7, 7]          --
    │    └─Conv2d: 2-41                      [100, 512, 7, 7]          2,359,808
    │    └─BatchNorm2d: 2-42                 [100, 512, 7, 7]          1,024
    │    └─ReLU: 2-43                        [100, 512, 7, 7]          --
    │    └─Conv2d: 2-44                      [100, 512, 7, 7]          2,359,808
    │    └─BatchNorm2d: 2-45                 [100, 512, 7, 7]          1,024
    │    └─ReLU: 2-46                        [100, 512, 7, 7]          --
    │    └─Conv2d: 2-47                      [100, 512, 7, 7]          2,359,808
    │    └─BatchNorm2d: 2-48                 [100, 512, 7, 7]          1,024
    │    └─ReLU: 2-49                        [100, 512, 7, 7]          --
    │    └─Conv2d: 2-50                      [100, 512, 7, 7]          2,359,808
    │    └─BatchNorm2d: 2-51                 [100, 512, 7, 7]          1,024
    │    └─ReLU: 2-52                        [100, 512, 7, 7]          --
    ├─Identity: 1-2                          [100, 512, 7, 7]          --
    ├─Sequential: 1-3                        [100, 10]                 --
    │    └─Linear: 2-53                      [100, 4096]               102,764,544
    │    └─ReLU: 2-54                        [100, 4096]               --
    │    └─Dropout: 2-55                     [100, 4096]               --
    │    └─Linear: 2-56                      [100, 4096]               16,781,312
    │    └─ReLU: 2-57                        [100, 4096]               --
    │    └─Dropout: 2-58                     [100, 4096]               --
    │    └─Linear: 2-59                      [100, 10]                 40,970
    ==========================================================================================
    Total params: 139,622,218
    Trainable params: 139,622,218
    Non-trainable params: 0
    Total mult-adds (G): 500.04
    ==========================================================================================
    Input size (MB): 15.05
    Forward/backward pass size (MB): 5947.40
    Params size (MB): 558.49
    Estimated Total Size (MB): 6520.94
    ==========================================================================================




```python
# 손실 계산 그래프 시각화

criterion = nn.CrossEntropyLoss()
loss = eval_loss(test_loader, device, net, criterion)
g = make_dot(loss, params=dict(net.named_parameters()))
display(g)
```


    
*(이미지 없음 / image not preserved: output_48_0.svg)*
    


### 초기 설정


```python
# 난수 고정
torch_seed()

# 사전 학습 모델 불러오기
net = models.vgg19_bn(pretrained = True)

# 최종 레이어 함수 교체
in_features = net.classifier[6].in_features
net.classifier[6] = nn.Linear(in_features, n_output)

# features 마지막의 MaxPool2d 제거
net.features = net.features[:-1]

# AdaptiveAvgPool2d 제거
net.avgpool = nn.Identity()

# 모델을 GPU로 전송
net = net.to(device)

# 학습률
lr = 0.001

# 손실 함수 정의
criterion = nn.CrossEntropyLoss()

# 최적화 함수 정의
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

# history 초기화
history = np.zeros((0, 5))

```

### 학습


```python
num_epochs = 5
history = fit(net, optimizer, criterion, num_epochs, 
          train_loader, test_loader, device, history)
```


      0%|          | 0/1000 [00:00<?, ?it/s]


    Epoch [1/5], loss: 0.49669 acc: 0.83160 val_loss: 0.19144, val_acc: 0.93610



      0%|          | 0/1000 [00:00<?, ?it/s]


    Epoch [2/5], loss: 0.24048 acc: 0.91826 val_loss: 0.15483, val_acc: 0.94740



      0%|          | 0/1000 [00:00<?, ?it/s]


    Epoch [3/5], loss: 0.18289 acc: 0.93778 val_loss: 0.13438, val_acc: 0.95410



      0%|          | 0/1000 [00:00<?, ?it/s]


    Epoch [4/5], loss: 0.15228 acc: 0.94810 val_loss: 0.12692, val_acc: 0.95720



      0%|          | 0/1000 [00:00<?, ?it/s]


    Epoch [5/5], loss: 0.13073 acc: 0.95556 val_loss: 0.12888, val_acc: 0.95750


### 결과 확인


```python
# 결과 요약
evaluate_history(history)
```

    초기상태 : 손실 : 0.19144  정확도 : 0.93610
    최종상태 : 손실 : 0.12888 정확도 : 0.95750



    
*(이미지 없음 / image not preserved: output_54_1.png)*
    



    
*(이미지 없음 / image not preserved: output_54_2.png)*
    



```python
# 이미지와 정답, 예측 결과를 함께 표시
show_images_labels(test_loader, classes, net, device)
```

    len(images) =  50



    
*(이미지 없음 / image not preserved: output_55_1.png)*
    


## CIFAR-10에 전이 학습을 적용한 경우 (parameters freezing)


```python
# 사전 학습 모델 불러오기
weights = models.ResNet18_Weights.IMAGENET1K_V1
net = models.resnet18(weights = weights)
next(iter(net.parameters())).requires_grad
```




    True




```python
print(net)
```

    ResNet(
      (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
      (layer1): Sequential(
        (0): BasicBlock(
          (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (1): BasicBlock(
          (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (layer2): Sequential(
        (0): BasicBlock(
          (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (downsample): Sequential(
            (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): BasicBlock(
          (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (layer3): Sequential(
        (0): BasicBlock(
          (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (downsample): Sequential(
            (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): BasicBlock(
          (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (layer4): Sequential(
        (0): BasicBlock(
          (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (downsample): Sequential(
            (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): BasicBlock(
          (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
      (fc): Linear(in_features=512, out_features=10, bias=True)
    )



```python
# 전이 학습

# 사전 학습 모델 불러오기
weights = models.ResNet18_Weights.IMAGENET1K_V1
net = models.resnet18(weights = weights)

# 모든 파라미터의 경사 계산을 OFF로 설정
for param in net.parameters():
    param.requires_grad = False

# 난수 고정
torch_seed()

# 최종 레이어 함수 교체
net.fc = nn.Linear(net.fc.in_features, n_output)

# GPU 사용
net = net.to(device)

# 학습률
lr = 0.001

# 손실 함수 정의
criterion = nn.CrossEntropyLoss()

# 최적화 함수 정의
# 파라미터 변경은 최종 레이어 함수로 한정
optimizer = optim.SGD(net.fc.parameters(), lr=lr, momentum=0.9)

# history 파일 초기화
history = np.zeros((0, 5))
```


```python
# 학습
num_epochs = 5
history = fit(net, optimizer, criterion, num_epochs, 
        train_loader, test_loader, device, history)
```


      0%|          | 0/1000 [00:00<?, ?it/s]



```python
# 결과 요약
evaluate_history(history)
```

## 범용적인 사전 학습 모델을 작성하는 법

### 모델 불러오기


```python
# 사전 학습 모델 불러오기
from torchvision import models

weights = models.VGG19_BN_Weights.DEFAULT
net = models.vgg19_bn(weights = weights)
```

### 모델 개요 표시 1


```python
print(net)
```

    VGG(
      (features): Sequential(
        (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU(inplace=True)
        (6): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (7): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (8): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (9): ReLU(inplace=True)
        (10): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (11): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (12): ReLU(inplace=True)
        (13): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (14): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (15): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (16): ReLU(inplace=True)
        (17): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (18): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (19): ReLU(inplace=True)
        (20): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (21): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (22): ReLU(inplace=True)
        (23): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (24): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (25): ReLU(inplace=True)
        (26): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (27): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (28): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (29): ReLU(inplace=True)
        (30): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (31): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (32): ReLU(inplace=True)
        (33): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (34): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (35): ReLU(inplace=True)
        (36): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (37): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (38): ReLU(inplace=True)
        (39): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (40): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (41): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (42): ReLU(inplace=True)
        (43): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (44): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (45): ReLU(inplace=True)
        (46): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (47): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (48): ReLU(inplace=True)
        (49): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (50): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (51): ReLU(inplace=True)
        (52): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
      (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
      (classifier): Sequential(
        (0): Linear(in_features=25088, out_features=4096, bias=True)
        (1): ReLU(inplace=True)
        (2): Dropout(p=0.5, inplace=False)
        (3): Linear(in_features=4096, out_features=4096, bias=True)
        (4): ReLU(inplace=True)
        (5): Dropout(p=0.5, inplace=False)
        (6): Linear(in_features=4096, out_features=1000, bias=True)
      )
    )


### 중간 텐서 확인


```python
# 원본 데이터 사이즈의 경우(배치사이즈 100)
net = net.to(device)
summary(net, (100, 3, 224, 224))
```




    ==========================================================================================
    Layer (type:depth-idx)                   Output Shape              Param #
    ==========================================================================================
    VGG                                      [100, 1000]               --
    ├─Sequential: 1-1                        [100, 512, 7, 7]          --
    │    └─Conv2d: 2-1                       [100, 64, 224, 224]       1,792
    │    └─BatchNorm2d: 2-2                  [100, 64, 224, 224]       128
    │    └─ReLU: 2-3                         [100, 64, 224, 224]       --
    │    └─Conv2d: 2-4                       [100, 64, 224, 224]       36,928
    │    └─BatchNorm2d: 2-5                  [100, 64, 224, 224]       128
    │    └─ReLU: 2-6                         [100, 64, 224, 224]       --
    │    └─MaxPool2d: 2-7                    [100, 64, 112, 112]       --
    │    └─Conv2d: 2-8                       [100, 128, 112, 112]      73,856
    │    └─BatchNorm2d: 2-9                  [100, 128, 112, 112]      256
    │    └─ReLU: 2-10                        [100, 128, 112, 112]      --
    │    └─Conv2d: 2-11                      [100, 128, 112, 112]      147,584
    │    └─BatchNorm2d: 2-12                 [100, 128, 112, 112]      256
    │    └─ReLU: 2-13                        [100, 128, 112, 112]      --
    │    └─MaxPool2d: 2-14                   [100, 128, 56, 56]        --
    │    └─Conv2d: 2-15                      [100, 256, 56, 56]        295,168
    │    └─BatchNorm2d: 2-16                 [100, 256, 56, 56]        512
    │    └─ReLU: 2-17                        [100, 256, 56, 56]        --
    │    └─Conv2d: 2-18                      [100, 256, 56, 56]        590,080
    │    └─BatchNorm2d: 2-19                 [100, 256, 56, 56]        512
    │    └─ReLU: 2-20                        [100, 256, 56, 56]        --
    │    └─Conv2d: 2-21                      [100, 256, 56, 56]        590,080
    │    └─BatchNorm2d: 2-22                 [100, 256, 56, 56]        512
    │    └─ReLU: 2-23                        [100, 256, 56, 56]        --
    │    └─Conv2d: 2-24                      [100, 256, 56, 56]        590,080
    │    └─BatchNorm2d: 2-25                 [100, 256, 56, 56]        512
    │    └─ReLU: 2-26                        [100, 256, 56, 56]        --
    │    └─MaxPool2d: 2-27                   [100, 256, 28, 28]        --
    │    └─Conv2d: 2-28                      [100, 512, 28, 28]        1,180,160
    │    └─BatchNorm2d: 2-29                 [100, 512, 28, 28]        1,024
    │    └─ReLU: 2-30                        [100, 512, 28, 28]        --
    │    └─Conv2d: 2-31                      [100, 512, 28, 28]        2,359,808
    │    └─BatchNorm2d: 2-32                 [100, 512, 28, 28]        1,024
    │    └─ReLU: 2-33                        [100, 512, 28, 28]        --
    │    └─Conv2d: 2-34                      [100, 512, 28, 28]        2,359,808
    │    └─BatchNorm2d: 2-35                 [100, 512, 28, 28]        1,024
    │    └─ReLU: 2-36                        [100, 512, 28, 28]        --
    │    └─Conv2d: 2-37                      [100, 512, 28, 28]        2,359,808
    │    └─BatchNorm2d: 2-38                 [100, 512, 28, 28]        1,024
    │    └─ReLU: 2-39                        [100, 512, 28, 28]        --
    │    └─MaxPool2d: 2-40                   [100, 512, 14, 14]        --
    │    └─Conv2d: 2-41                      [100, 512, 14, 14]        2,359,808
    │    └─BatchNorm2d: 2-42                 [100, 512, 14, 14]        1,024
    │    └─ReLU: 2-43                        [100, 512, 14, 14]        --
    │    └─Conv2d: 2-44                      [100, 512, 14, 14]        2,359,808
    │    └─BatchNorm2d: 2-45                 [100, 512, 14, 14]        1,024
    │    └─ReLU: 2-46                        [100, 512, 14, 14]        --
    │    └─Conv2d: 2-47                      [100, 512, 14, 14]        2,359,808
    │    └─BatchNorm2d: 2-48                 [100, 512, 14, 14]        1,024
    │    └─ReLU: 2-49                        [100, 512, 14, 14]        --
    │    └─Conv2d: 2-50                      [100, 512, 14, 14]        2,359,808
    │    └─BatchNorm2d: 2-51                 [100, 512, 14, 14]        1,024
    │    └─ReLU: 2-52                        [100, 512, 14, 14]        --
    │    └─MaxPool2d: 2-53                   [100, 512, 7, 7]          --
    ├─AdaptiveAvgPool2d: 1-2                 [100, 512, 7, 7]          --
    ├─Sequential: 1-3                        [100, 1000]               --
    │    └─Linear: 2-54                      [100, 4096]               102,764,544
    │    └─ReLU: 2-55                        [100, 4096]               --
    │    └─Dropout: 2-56                     [100, 4096]               --
    │    └─Linear: 2-57                      [100, 4096]               16,781,312
    │    └─ReLU: 2-58                        [100, 4096]               --
    │    └─Dropout: 2-59                     [100, 4096]               --
    │    └─Linear: 2-60                      [100, 1000]               4,097,000
    ==========================================================================================
    Total params: 143,678,248
    Trainable params: 143,678,248
    Non-trainable params: 0
    Total mult-adds (T): 1.96
    ==========================================================================================
    Input size (MB): 60.21
    Forward/backward pass size (MB): 23770.71
    Params size (MB): 574.71
    Estimated Total Size (MB): 24405.63
    ==========================================================================================




```python
# 실습용 데이터 사이즈의 경우(배치사이즈 100)
summary(net, (100, 3, 112, 112))
```




    ==========================================================================================
    Layer (type:depth-idx)                   Output Shape              Param #
    ==========================================================================================
    VGG                                      [100, 1000]               --
    ├─Sequential: 1-1                        [100, 512, 3, 3]          --
    │    └─Conv2d: 2-1                       [100, 64, 112, 112]       1,792
    │    └─BatchNorm2d: 2-2                  [100, 64, 112, 112]       128
    │    └─ReLU: 2-3                         [100, 64, 112, 112]       --
    │    └─Conv2d: 2-4                       [100, 64, 112, 112]       36,928
    │    └─BatchNorm2d: 2-5                  [100, 64, 112, 112]       128
    │    └─ReLU: 2-6                         [100, 64, 112, 112]       --
    │    └─MaxPool2d: 2-7                    [100, 64, 56, 56]         --
    │    └─Conv2d: 2-8                       [100, 128, 56, 56]        73,856
    │    └─BatchNorm2d: 2-9                  [100, 128, 56, 56]        256
    │    └─ReLU: 2-10                        [100, 128, 56, 56]        --
    │    └─Conv2d: 2-11                      [100, 128, 56, 56]        147,584
    │    └─BatchNorm2d: 2-12                 [100, 128, 56, 56]        256
    │    └─ReLU: 2-13                        [100, 128, 56, 56]        --
    │    └─MaxPool2d: 2-14                   [100, 128, 28, 28]        --
    │    └─Conv2d: 2-15                      [100, 256, 28, 28]        295,168
    │    └─BatchNorm2d: 2-16                 [100, 256, 28, 28]        512
    │    └─ReLU: 2-17                        [100, 256, 28, 28]        --
    │    └─Conv2d: 2-18                      [100, 256, 28, 28]        590,080
    │    └─BatchNorm2d: 2-19                 [100, 256, 28, 28]        512
    │    └─ReLU: 2-20                        [100, 256, 28, 28]        --
    │    └─Conv2d: 2-21                      [100, 256, 28, 28]        590,080
    │    └─BatchNorm2d: 2-22                 [100, 256, 28, 28]        512
    │    └─ReLU: 2-23                        [100, 256, 28, 28]        --
    │    └─Conv2d: 2-24                      [100, 256, 28, 28]        590,080
    │    └─BatchNorm2d: 2-25                 [100, 256, 28, 28]        512
    │    └─ReLU: 2-26                        [100, 256, 28, 28]        --
    │    └─MaxPool2d: 2-27                   [100, 256, 14, 14]        --
    │    └─Conv2d: 2-28                      [100, 512, 14, 14]        1,180,160
    │    └─BatchNorm2d: 2-29                 [100, 512, 14, 14]        1,024
    │    └─ReLU: 2-30                        [100, 512, 14, 14]        --
    │    └─Conv2d: 2-31                      [100, 512, 14, 14]        2,359,808
    │    └─BatchNorm2d: 2-32                 [100, 512, 14, 14]        1,024
    │    └─ReLU: 2-33                        [100, 512, 14, 14]        --
    │    └─Conv2d: 2-34                      [100, 512, 14, 14]        2,359,808
    │    └─BatchNorm2d: 2-35                 [100, 512, 14, 14]        1,024
    │    └─ReLU: 2-36                        [100, 512, 14, 14]        --
    │    └─Conv2d: 2-37                      [100, 512, 14, 14]        2,359,808
    │    └─BatchNorm2d: 2-38                 [100, 512, 14, 14]        1,024
    │    └─ReLU: 2-39                        [100, 512, 14, 14]        --
    │    └─MaxPool2d: 2-40                   [100, 512, 7, 7]          --
    │    └─Conv2d: 2-41                      [100, 512, 7, 7]          2,359,808
    │    └─BatchNorm2d: 2-42                 [100, 512, 7, 7]          1,024
    │    └─ReLU: 2-43                        [100, 512, 7, 7]          --
    │    └─Conv2d: 2-44                      [100, 512, 7, 7]          2,359,808
    │    └─BatchNorm2d: 2-45                 [100, 512, 7, 7]          1,024
    │    └─ReLU: 2-46                        [100, 512, 7, 7]          --
    │    └─Conv2d: 2-47                      [100, 512, 7, 7]          2,359,808
    │    └─BatchNorm2d: 2-48                 [100, 512, 7, 7]          1,024
    │    └─ReLU: 2-49                        [100, 512, 7, 7]          --
    │    └─Conv2d: 2-50                      [100, 512, 7, 7]          2,359,808
    │    └─BatchNorm2d: 2-51                 [100, 512, 7, 7]          1,024
    │    └─ReLU: 2-52                        [100, 512, 7, 7]          --
    │    └─MaxPool2d: 2-53                   [100, 512, 3, 3]          --
    ├─AdaptiveAvgPool2d: 1-2                 [100, 512, 7, 7]          --
    ├─Sequential: 1-3                        [100, 1000]               --
    │    └─Linear: 2-54                      [100, 4096]               102,764,544
    │    └─ReLU: 2-55                        [100, 4096]               --
    │    └─Dropout: 2-56                     [100, 4096]               --
    │    └─Linear: 2-57                      [100, 4096]               16,781,312
    │    └─ReLU: 2-58                        [100, 4096]               --
    │    └─Dropout: 2-59                     [100, 4096]               --
    │    └─Linear: 2-60                      [100, 1000]               4,097,000
    ==========================================================================================
    Total params: 143,678,248
    Trainable params: 143,678,248
    Non-trainable params: 0
    Total mult-adds (G): 500.45
    ==========================================================================================
    Input size (MB): 15.05
    Forward/backward pass size (MB): 5948.19
    Params size (MB): 574.71
    Estimated Total Size (MB): 6537.96
    ==========================================================================================



### 레이어 함수 교체하기


```python
# 난수 고정
torch_seed()

# 최종 레이어 함수 교체
in_features = net.classifier[6].in_features
net.classifier[6] = nn.Linear(in_features, n_output)
```


```python
# features의 마지막 요소(MaxPool2d)를 제거
net.features = net.features[:-1]
print(net.features)
```

    Sequential(
      (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU(inplace=True)
      (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (5): ReLU(inplace=True)
      (6): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (7): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (8): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (9): ReLU(inplace=True)
      (10): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (11): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (12): ReLU(inplace=True)
      (13): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (14): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (15): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (16): ReLU(inplace=True)
      (17): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (18): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (19): ReLU(inplace=True)
      (20): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (21): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (22): ReLU(inplace=True)
      (23): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (24): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (25): ReLU(inplace=True)
      (26): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (27): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (28): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (29): ReLU(inplace=True)
      (30): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (31): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (32): ReLU(inplace=True)
      (33): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (34): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (35): ReLU(inplace=True)
      (36): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (37): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (38): ReLU(inplace=True)
      (39): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (40): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (41): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (42): ReLU(inplace=True)
      (43): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (44): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (45): ReLU(inplace=True)
      (46): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (47): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (48): ReLU(inplace=True)
      (49): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (50): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (51): ReLU(inplace=True)
    )



```python
# avgpool에 위치한AdaptiveAvgPool2d을 아무것도 하지 않는 함수(nn.Identity)로 치환
net.avgpool = nn.Identity()
```

### 결과 확인


```python
print(net)
```

    VGG(
      (features): Sequential(
        (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU(inplace=True)
        (6): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (7): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (8): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (9): ReLU(inplace=True)
        (10): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (11): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (12): ReLU(inplace=True)
        (13): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (14): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (15): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (16): ReLU(inplace=True)
        (17): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (18): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (19): ReLU(inplace=True)
        (20): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (21): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (22): ReLU(inplace=True)
        (23): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (24): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (25): ReLU(inplace=True)
        (26): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (27): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (28): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (29): ReLU(inplace=True)
        (30): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (31): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (32): ReLU(inplace=True)
        (33): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (34): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (35): ReLU(inplace=True)
        (36): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (37): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (38): ReLU(inplace=True)
        (39): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (40): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (41): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (42): ReLU(inplace=True)
        (43): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (44): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (45): ReLU(inplace=True)
        (46): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (47): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (48): ReLU(inplace=True)
        (49): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (50): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (51): ReLU(inplace=True)
      )
      (avgpool): Identity()
      (classifier): Sequential(
        (0): Linear(in_features=25088, out_features=4096, bias=True)
        (1): ReLU(inplace=True)
        (2): Dropout(p=0.5, inplace=False)
        (3): Linear(in_features=4096, out_features=4096, bias=True)
        (4): ReLU(inplace=True)
        (5): Dropout(p=0.5, inplace=False)
        (6): Linear(in_features=4096, out_features=10, bias=True)
      )
    )



```python
# 실습용 데이터 사이즈로 중간 텐서 확인(배치사이즈 100)
net = net.to(device)
summary(net,(100, 3, 112, 112))
```




    ==========================================================================================
    Layer (type:depth-idx)                   Output Shape              Param #
    ==========================================================================================
    VGG                                      [100, 10]                 --
    ├─Sequential: 1-1                        [100, 512, 7, 7]          --
    │    └─Conv2d: 2-1                       [100, 64, 112, 112]       1,792
    │    └─BatchNorm2d: 2-2                  [100, 64, 112, 112]       128
    │    └─ReLU: 2-3                         [100, 64, 112, 112]       --
    │    └─Conv2d: 2-4                       [100, 64, 112, 112]       36,928
    │    └─BatchNorm2d: 2-5                  [100, 64, 112, 112]       128
    │    └─ReLU: 2-6                         [100, 64, 112, 112]       --
    │    └─MaxPool2d: 2-7                    [100, 64, 56, 56]         --
    │    └─Conv2d: 2-8                       [100, 128, 56, 56]        73,856
    │    └─BatchNorm2d: 2-9                  [100, 128, 56, 56]        256
    │    └─ReLU: 2-10                        [100, 128, 56, 56]        --
    │    └─Conv2d: 2-11                      [100, 128, 56, 56]        147,584
    │    └─BatchNorm2d: 2-12                 [100, 128, 56, 56]        256
    │    └─ReLU: 2-13                        [100, 128, 56, 56]        --
    │    └─MaxPool2d: 2-14                   [100, 128, 28, 28]        --
    │    └─Conv2d: 2-15                      [100, 256, 28, 28]        295,168
    │    └─BatchNorm2d: 2-16                 [100, 256, 28, 28]        512
    │    └─ReLU: 2-17                        [100, 256, 28, 28]        --
    │    └─Conv2d: 2-18                      [100, 256, 28, 28]        590,080
    │    └─BatchNorm2d: 2-19                 [100, 256, 28, 28]        512
    │    └─ReLU: 2-20                        [100, 256, 28, 28]        --
    │    └─Conv2d: 2-21                      [100, 256, 28, 28]        590,080
    │    └─BatchNorm2d: 2-22                 [100, 256, 28, 28]        512
    │    └─ReLU: 2-23                        [100, 256, 28, 28]        --
    │    └─Conv2d: 2-24                      [100, 256, 28, 28]        590,080
    │    └─BatchNorm2d: 2-25                 [100, 256, 28, 28]        512
    │    └─ReLU: 2-26                        [100, 256, 28, 28]        --
    │    └─MaxPool2d: 2-27                   [100, 256, 14, 14]        --
    │    └─Conv2d: 2-28                      [100, 512, 14, 14]        1,180,160
    │    └─BatchNorm2d: 2-29                 [100, 512, 14, 14]        1,024
    │    └─ReLU: 2-30                        [100, 512, 14, 14]        --
    │    └─Conv2d: 2-31                      [100, 512, 14, 14]        2,359,808
    │    └─BatchNorm2d: 2-32                 [100, 512, 14, 14]        1,024
    │    └─ReLU: 2-33                        [100, 512, 14, 14]        --
    │    └─Conv2d: 2-34                      [100, 512, 14, 14]        2,359,808
    │    └─BatchNorm2d: 2-35                 [100, 512, 14, 14]        1,024
    │    └─ReLU: 2-36                        [100, 512, 14, 14]        --
    │    └─Conv2d: 2-37                      [100, 512, 14, 14]        2,359,808
    │    └─BatchNorm2d: 2-38                 [100, 512, 14, 14]        1,024
    │    └─ReLU: 2-39                        [100, 512, 14, 14]        --
    │    └─MaxPool2d: 2-40                   [100, 512, 7, 7]          --
    │    └─Conv2d: 2-41                      [100, 512, 7, 7]          2,359,808
    │    └─BatchNorm2d: 2-42                 [100, 512, 7, 7]          1,024
    │    └─ReLU: 2-43                        [100, 512, 7, 7]          --
    │    └─Conv2d: 2-44                      [100, 512, 7, 7]          2,359,808
    │    └─BatchNorm2d: 2-45                 [100, 512, 7, 7]          1,024
    │    └─ReLU: 2-46                        [100, 512, 7, 7]          --
    │    └─Conv2d: 2-47                      [100, 512, 7, 7]          2,359,808
    │    └─BatchNorm2d: 2-48                 [100, 512, 7, 7]          1,024
    │    └─ReLU: 2-49                        [100, 512, 7, 7]          --
    │    └─Conv2d: 2-50                      [100, 512, 7, 7]          2,359,808
    │    └─BatchNorm2d: 2-51                 [100, 512, 7, 7]          1,024
    │    └─ReLU: 2-52                        [100, 512, 7, 7]          --
    ├─Identity: 1-2                          [100, 512, 7, 7]          --
    ├─Sequential: 1-3                        [100, 10]                 --
    │    └─Linear: 2-53                      [100, 4096]               102,764,544
    │    └─ReLU: 2-54                        [100, 4096]               --
    │    └─Dropout: 2-55                     [100, 4096]               --
    │    └─Linear: 2-56                      [100, 4096]               16,781,312
    │    └─ReLU: 2-57                        [100, 4096]               --
    │    └─Dropout: 2-58                     [100, 4096]               --
    │    └─Linear: 2-59                      [100, 10]                 40,970
    ==========================================================================================
    Total params: 139,622,218
    Trainable params: 139,622,218
    Non-trainable params: 0
    Total mult-adds (G): 500.04
    ==========================================================================================
    Input size (MB): 15.05
    Forward/backward pass size (MB): 5947.40
    Params size (MB): 558.49
    Estimated Total Size (MB): 6520.94
    ==========================================================================================


