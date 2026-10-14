# 15장 사용자 정의 데이터를 활용한 이미지 분류

> **Migrated from Quartz wiki** (`wiki/study/rokey-ai/basic/ai-intro-15-transfer_learning.md`, updated 2026-09-26)  
> tags: rokey, ai, notebook, basic, study  
> source: raw/2026-09-26-rokey-ai/basic/강의_3기_AI개론_15차시__Transfer_learning_.ipynb  
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
w = !apt install tree
print(w[-2])
```

    'apt' is not recognized as an internal or external command,


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
from torch import tensor
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
plt.rcParams['font.family'] = font_name

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

# # 공통 함수 불러오기
from pythonlibs.torch_lib1 import *


# # 공통 함수 확인
print(README)
```

    Common Library for PyTorch
    Author: M. Akaishi


## 데이터 준비

### 데이터 다운로드, 압축 해제, 트리 구조 출력


```python
# 데이터 다운로드
w = !wget -nc https://download.pytorch.org/tutorial/hymenoptera_data.zip

# 결과 확인
print(w[-2])
```

    2022-04-10 14:24:10 (46.5 MB/s) - ‘hymenoptera_data.zip’ saved [47286322/47286322]



```python
# 압축 해제
w = !unzip -o hymenoptera_data.zip

# 결과 확인
print(w[-1])
```

      inflating: hymenoptera_data/val/bees/abeja.jpg  



```python
# 트리 구조 출력
!tree hymenoptera_data
```

    Folder PATH listing for volume 드라이브D
    Volume serial number is 00000092 32BC:C9D8
    D:\ONEDRIVE\DOCUMENTS\LECTURE_2019\ACADEMY\30_SPARKX\01_AI_BASIC\15_TRANSFER_LEARNING\HYMENOPTERA_DATA
    ├───train
    │   ├───ants
    │   └───bees
    └───val
        ├───ants
        └───bees


### Transforms 정의


```python
# Transforms 정의
# 훈련 데이터 : 정규화에 반전과 RandomErasing 추가
# 입력 이미지를 주어진 크기(resize: 224×224)로 조정, scale은 원래 이미지를 임의의 크기(0.5~1.0(50~100%))만큼 면적을 무작위로 자르겠다는 의미

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(p = 0.5),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)
])

# 검증 데이터 : 정규화
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
])


```

### 데이터셋 정의


```python
# 베이스 디렉터리
data_dir = 'hymenoptera_data'

# 훈련 데이터 디렉터리와 검증 데이터 디렉터리 지정
import os
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'val')

# join 함수 결과 확인
print(train_dir, test_dir)

# 분류하려는 클래스의 리스트 작성
classes = ['ants', 'bees']
```

    hymenoptera_data\train hymenoptera_data\val



```python
# 데이터셋 정의

# 훈련용
train_data = datasets.ImageFolder(train_dir, 
                                  transform=train_transform)
# 훈련 데이터 이미지 출력용
train_data2 = datasets.ImageFolder(train_dir, 
                                   transform=test_transform)
# 검증용
test_data = datasets.ImageFolder(test_dir, 
                                 transform=test_transform)
```


```python
# 데이터 건수 확인

print(f'훈련 데이터 : {len(train_data)} 건')
print(f'검증 데이터 : {len(test_data)} 건')
```

    훈련 데이터 : 244 건
    검증 데이터 : 153 건



```python
# 검증 데이터　
# 처음 10개와 마지막 10개 이미지 출력

plt.figure(figsize=(15, 4))
for i in range(10):
    ax = plt.subplot(2, 10, i + 1)
    image, label = test_data[i]
    img = (np.transpose(image.numpy(), (1, 2, 0)) + 1)/2
    plt.imshow(img)
    ax.set_title(classes[label])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, 10, i + 11)
    image, label = test_data[-i-1]
    img = (np.transpose(image.numpy(), (1, 2, 0)) + 1)/2
    plt.imshow(img)
    ax.set_title(classes[label])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()
```


    
*(이미지 없음 / image not preserved: output_22_0.png)*
    


### 데이터로더 정의


```python
# 데이터로더 정의

batch_size = 10

# 훈련용
train_loader = DataLoader(train_data, 
      batch_size=batch_size, shuffle=True)

# 검증용
test_loader = DataLoader(test_data, 
      batch_size=batch_size, shuffle=False)

# 이미지 출력용
train_loader2 = DataLoader(train_data2, 
      batch_size=50, shuffle=True)
test_loader2 = DataLoader(test_data, 
      batch_size=50, shuffle=True)
```

### 이미지 출력


```python
# 검증 데이터(50건)
torch_seed()
show_images_labels(test_loader2, classes, None, None)
```

    len(images) =  50



    
*(이미지 없음 / image not preserved: output_26_1.png)*
    


## 파인 튜닝의 경우


```python
# 파인 튜닝의 경우

# 사전 학습 모델 불러오기
# VGG-19-BN 모델을 학습이 끝난 파라미터와 함께 불러오기
# from torchvision import models
# net = models.vgg19_bn(pretrained = True)
weights = models.VGG19_BN_Weights.DEFAULT
net = models.vgg19_bn(weights = weights)
```


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



```python
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

# 난수 고정
torch_seed()

# 최종 노드의 출력을 2로 변경
in_features = net.classifier[6].in_features
net.classifier[6] = nn.Linear(in_features, 2)

# AdaptiveAvgPool2d 함수 제거
net.avgpool = nn.Identity()

# GPU 사용
net = net.to(device)

# 학습률
lr = 0.001

# 손실 함수 정의
criterion = nn.CrossEntropyLoss()

# 최적화 함수 정의
optimizer = optim.SGD(net.parameters(),lr=lr,momentum=0.9)

# history 파일도 동시에 초기화
history = np.zeros((0, 5))
```


```python
# 학습
num_epochs = 5
history = fit(net, optimizer, criterion, num_epochs, 
          train_loader, test_loader, device, history)
```


      0%|          | 0/25 [00:00<?, ?it/s]


    Epoch [1/5], loss: 0.42943 acc: 0.78000 val_loss: 0.10684, val_acc: 0.96250



      0%|          | 0/25 [00:00<?, ?it/s]


    Epoch [2/5], loss: 0.18828 acc: 0.92000 val_loss: 0.11837, val_acc: 0.96875



      0%|          | 0/25 [00:00<?, ?it/s]


    Epoch [3/5], loss: 0.21427 acc: 0.91200 val_loss: 0.10815, val_acc: 0.96250



      0%|          | 0/25 [00:00<?, ?it/s]


    Epoch [4/5], loss: 0.14176 acc: 0.93600 val_loss: 0.17399, val_acc: 0.95000



      0%|          | 0/25 [00:00<?, ?it/s]


    Epoch [5/5], loss: 0.16827 acc: 0.91400 val_loss: 0.10927, val_acc: 0.96875



```python
# 결과 확인
evaluate_history(history)
```

    초기상태 : 손실 : 0.10684  정확도 : 0.96250
    최종상태 : 손실 : 0.10927 정확도 : 0.96875



    
*(이미지 없음 / image not preserved: output_33_1.png)*
    



    
*(이미지 없음 / image not preserved: output_33_2.png)*
    



```python
# 난수 고정
torch_seed()

# 검증 데이터 결과 출력
show_images_labels(test_loader2, classes, net, device)
```

    len(images) =  50



    
*(이미지 없음 / image not preserved: output_34_1.png)*
    



## 전이 학습의 경우


```python
# VGG-19-BN 모델을 학습이 끝난 파라미터와 함께 불러오기
# from torchvision import models
# net = models.vgg19_bn(pretrained = True)
weights = models.VGG19_BN_Weights.DEFAULT
net = models.vgg19_bn(weights = weights)

# 모든 파라미터의 경사 계산을 OFF로 설정
for param in net.parameters():
    param.requires_grad = False

# 난수 고정
torch_seed()

# 최종 노드의 출력을 2로 변경
# 이 노드에 대해서만 경사 계산을 수행하게 됨
in_features = net.classifier[6].in_features
net.classifier[6] = nn.Linear(in_features, 2)

# AdaptiveAvgPool2d 함수 제거
net.avgpool = nn.Identity()

# GPU 사용
net = net.to(device)

# 학습률
lr = 0.001

# 손실 함수로 교차 엔트로피 사용
criterion = nn.CrossEntropyLoss()

# 최적화 함수 정의
# 파라미터 수정 대상을 최종 노드로 제한
optimizer = optim.SGD(net.classifier[6].parameters(),lr=lr,momentum=0.9)

# history 파일도 동시에 초기화
history = np.zeros((0, 5))
```


```python
# 학습
num_epochs = 5
history = fit(net, optimizer, criterion, num_epochs, 
          train_loader, test_loader, device, history)
```


      0%|          | 0/25 [00:00<?, ?it/s]


    Epoch [1/5], loss: 0.45079 acc: 0.77600 val_loss: 0.13135, val_acc: 0.96250



      0%|          | 0/25 [00:00<?, ?it/s]


    Epoch [2/5], loss: 0.22003 acc: 0.92000 val_loss: 0.12127, val_acc: 0.96250



      0%|          | 0/25 [00:00<?, ?it/s]


    Epoch [3/5], loss: 0.23990 acc: 0.90400 val_loss: 0.11304, val_acc: 0.95625



      0%|          | 0/25 [00:00<?, ?it/s]


    Epoch [4/5], loss: 0.19803 acc: 0.90400 val_loss: 0.12182, val_acc: 0.95625



      0%|          | 0/25 [00:00<?, ?it/s]


    Epoch [5/5], loss: 0.20835 acc: 0.88600 val_loss: 0.11390, val_acc: 0.96250



```python
# 결과 확인
evaluate_history(history)
```

    초기상태 : 손실 : 0.13135  정확도 : 0.96250
    최종상태 : 손실 : 0.11390 정확도 : 0.96250



    
*(이미지 없음 / image not preserved: output_38_1.png)*
    



    
*(이미지 없음 / image not preserved: output_38_2.png)*
    



```python
# 난수 고정
torch_seed()

# 검증 데이터 결과 출력
show_images_labels(test_loader2, classes, net, device)
```

    len(images) =  50



    
*(이미지 없음 / image not preserved: output_39_1.png)*
    


## 사용자 정의 데이터를 사용하는 경우
시베리안 허스키와 늑대 이미지를 사용함

### 데이터 다운로드, 압축 해제


```python
# 데이터 다운로드
w = !wget https://github.com/makaishi2/pythonlibs/raw/main/images/dog_wolf.zip
print(w[-2])

# 압축 해제
!unzip dog_wolf.zip | tail -n 1

# 트리 구조 확인
!tree dog_wolf
```

    2022-04-10 14:43:13 (142 MB/s) - ‘dog_wolf.zip’ saved [21811374/21811374]
      inflating: dog_wolf/train/wolf/wolf-09.png  
    dog_wolf
    ├── test
    │   ├── dog
    │   │   ├── dog-21.png
    │   │   ├── dog-22.png
    │   │   ├── dog-23.png
    │   │   ├── dog-24.png
    │   │   └── dog-25.png
    │   └── wolf
    │       ├── wolf-21.png
    │       ├── wolf-22.png
    │       ├── wolf-23.png
    │       ├── wolf-24.png
    │       └── wolf-25.png
    └── train
        ├── dog
        │   ├── dog-01.png
        │   ├── dog-02.png
        │   ├── dog-03.png
        │   ├── dog-04.png
        │   ├── dog-05.png
        │   ├── dog-06.png
        │   ├── dog-07.png
        │   ├── dog-08.png
        │   ├── dog-09.png
        │   ├── dog-10.png
        │   ├── dog-11.png
        │   ├── dog-12.png
        │   ├── dog-13.png
        │   ├── dog-14.png
        │   ├── dog-15.png
        │   ├── dog-16.png
        │   ├── dog-17.png
        │   ├── dog-18.png
        │   ├── dog-19.png
        │   └── dog-20.png
        └── wolf
            ├── wolf-01.png
            ├── wolf-02.png
            ├── wolf-03.png
            ├── wolf-04.png
            ├── wolf-05.png
            ├── wolf-06.png
            ├── wolf-07.png
            ├── wolf-08.png
            ├── wolf-09.png
            ├── wolf-10.png
            ├── wolf-11.png
            ├── wolf-12.png
            ├── wolf-13.png
            ├── wolf-14.png
            ├── wolf-15.png
            ├── wolf-16.png
            ├── wolf-17.png
            ├── wolf-18.png
            ├── wolf-19.png
            └── wolf-20.png
    
    6 directories, 50 files


### Transforms 정의


```python
# Transforms 정의

# 검증 데이터 : 정규화
test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
])

# 훈련 데이터 : 정규화에 반전과 RandomErasing 추가
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5), 
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)   
])
```

### 데이터셋 정의


```python
# 데이터셋 정의

data_dir = 'dog_wolf'

import os
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')

classes = ['dog', 'wolf']

train_data = datasets.ImageFolder(train_dir, 
            transform=train_transform)
train_data2 = datasets.ImageFolder(train_dir, 
            transform=test_transform)
test_data = datasets.ImageFolder(test_dir, 
            transform=test_transform)
```


```python
# 데이터 건수 확인

print(f'학습 데이터 : {len(train_data)} 건')
print(f'검증 데이터 : {len(test_data)} 건')
```

    학습 데이터 : 40 건
    검증 데이터 : 10 건


### 데이터로더 정의


```python
# 데이터로더 정의

batch_size = 5
# 훈련 데이터
train_loader = DataLoader(train_data, 
            batch_size=batch_size, shuffle=True)
# 훈련 데이터, 이미지 출력용
train_loader2 = DataLoader(train_data2, 
            batch_size=40, shuffle=False)
# 검증 데이터
test_loader = DataLoader(test_data, 
            batch_size=batch_size, shuffle=False)
# 검증데이터, 이미지 출력용
test_loader2 = DataLoader(test_data, 
            batch_size=10, shuffle=True)
```

### 이미지 출력


```python
# 훈련 데이터(40건)
show_images_labels(train_loader2, classes, None, None)
```


    
*(이미지 없음 / image not preserved: output_51_0.png)*
    



```python
# 검증 데이터(10건)
torch_seed()
show_images_labels(test_loader2, classes, None, None)
```


    
*(이미지 없음 / image not preserved: output_52_0.png)*
    


### 모델 정의


```python
# 사전 학습 모델 불러오기
net = models.vgg19_bn(pretrained = True)

for param in net.parameters():
    param.requires_grad = False

# 난수 고정
torch_seed()

# 마지막 노드 출력을 2로 변경
in_features = net.classifier[6].in_features
net.classifier[6] = nn.Linear(in_features, 2)

# AdaptiveAvgPool2d 함수 제거
net.avgpool = nn.Identity()

# GPU 사용
net = net.to(device)

# 학습률
lr = 0.001

# 손실 함수 정의
criterion = nn.CrossEntropyLoss()

# 최적화 함수 정의
# 파라미터 수정 대상을 최종 노드로 제한
optimizer = optim.SGD(net.classifier[6].parameters(),lr=lr,momentum=0.9)

# history 파일도 동시에 초기화
history = np.zeros((0, 5))
```


```python
# 학습
num_epochs = 10
history = fit(net, optimizer, criterion, num_epochs, 
          train_loader, test_loader, device, history)
```


      0%|          | 0/8 [00:00<?, ?it/s]


    Epoch [1/10], loss: 0.12345 acc: 0.65000 val_loss: 0.07783, val_acc: 1.00000



      0%|          | 0/8 [00:00<?, ?it/s]


    Epoch [2/10], loss: 0.07584 acc: 0.85000 val_loss: 0.04895, val_acc: 0.90000



      0%|          | 0/8 [00:00<?, ?it/s]


    Epoch [3/10], loss: 0.03976 acc: 0.92500 val_loss: 0.05762, val_acc: 0.80000



      0%|          | 0/8 [00:00<?, ?it/s]


    Epoch [4/10], loss: 0.04213 acc: 0.92500 val_loss: 0.03992, val_acc: 1.00000



      0%|          | 0/8 [00:00<?, ?it/s]


    Epoch [5/10], loss: 0.01836 acc: 0.97500 val_loss: 0.02970, val_acc: 1.00000



      0%|          | 0/8 [00:00<?, ?it/s]


    Epoch [6/10], loss: 0.02144 acc: 0.97500 val_loss: 0.04182, val_acc: 0.90000



      0%|          | 0/8 [00:00<?, ?it/s]


    Epoch [7/10], loss: 0.03019 acc: 0.95000 val_loss: 0.03631, val_acc: 0.90000



      0%|          | 0/8 [00:00<?, ?it/s]


    Epoch [8/10], loss: 0.04319 acc: 0.92500 val_loss: 0.03186, val_acc: 1.00000



      0%|          | 0/8 [00:00<?, ?it/s]


    Epoch [9/10], loss: 0.01086 acc: 1.00000 val_loss: 0.02766, val_acc: 1.00000



      0%|          | 0/8 [00:00<?, ?it/s]


    Epoch [10/10], loss: 0.04419 acc: 0.92500 val_loss: 0.03189, val_acc: 1.00000



```python
# 결과 확인
evaluate_history(history)
```

    초기상태 : 손실 : 0.07783  정확도 : 1.00000
    최종상태 : 손실 : 0.03189 정확도 : 1.00000



    
*(이미지 없음 / image not preserved: output_56_1.png)*
    



    
*(이미지 없음 / image not preserved: output_56_2.png)*
    



```python
# 예측 결과 출력
torch_seed()
show_images_labels(test_loader2, classes, net, device)
```


    
*(이미지 없음 / image not preserved: output_57_0.png)*
    

