# 12장 튜닝 기법

> **Migrated from Quartz wiki** (`wiki/study/rokey-ai/basic/ai-intro-12-sgd_adam.md`, updated 2026-09-26)  
> tags: rokey, ai, notebook, basic, study  
> source: raw/2026-09-26-rokey-ai/basic/강의_3기_AI개론_12차시__SGD_Adam_.ipynb  
> source: https://github.com/weedmo/Rokey-AI  

* "부록3 매트플롯립 입문"에서 한글 폰트를 올바르게 출력하기 위한 설치 방법을 설명했다. 설치 방법은 다음과 같다.


```python
!sudo apt-get install -y fonts-nanum* | tail -n 1
!sudo fc-cache -fv
!rm -rf ~/.cache/matplotlib
```

    debconf: unable to initialize frontend: Dialog
    debconf: (No usable dialog-like program is installed, so the dialog based frontend cannot be used. at /usr/share/perl5/Debconf/FrontEnd/Dialog.pm line 78, <> line 4.)
    debconf: falling back to frontend: Readline
    debconf: unable to initialize frontend: Readline
    debconf: (This frontend requires a controlling tty.)
    debconf: falling back to frontend: Teletype
    dpkg-preconfigure: unable to re-open stdin: 
    Processing triggers for fontconfig (2.13.1-4.2ubuntu5) ...
    /usr/share/fonts: caching, new cache contents: 0 fonts, 1 dirs
    /usr/share/fonts/truetype: caching, new cache contents: 0 fonts, 3 dirs
    /usr/share/fonts/truetype/humor-sans: caching, new cache contents: 1 fonts, 0 dirs
    /usr/share/fonts/truetype/liberation: caching, new cache contents: 16 fonts, 0 dirs
    /usr/share/fonts/truetype/nanum: caching, new cache contents: 39 fonts, 0 dirs
    /usr/local/share/fonts: caching, new cache contents: 0 fonts, 0 dirs
    /root/.local/share/fonts: skipping, no such directory
    /root/.fonts: skipping, no such directory
    /usr/share/fonts/truetype: skipping, looped directory detected
    /usr/share/fonts/truetype/humor-sans: skipping, looped directory detected
    /usr/share/fonts/truetype/liberation: skipping, looped directory detected
    /usr/share/fonts/truetype/nanum: skipping, looped directory detected
    /var/cache/fontconfig: cleaning cache directory
    /root/.cache/fontconfig: not cleaning non-existent cache directory
    /root/.fontconfig: not cleaning non-existent cache directory
    fc-cache: succeeded



```python
# 필요 라이브러리 설치

!pip install torchviz | tail -n 1
!pip install torchinfo | tail -n 1
```

    Successfully installed nvidia-cublas-cu12-12.4.5.8 nvidia-cuda-cupti-cu12-12.4.127 nvidia-cuda-nvrtc-cu12-12.4.127 nvidia-cuda-runtime-cu12-12.4.127 nvidia-cudnn-cu12-9.1.0.70 nvidia-cufft-cu12-11.2.1.3 nvidia-curand-cu12-10.3.5.147 nvidia-cusolver-cu12-11.6.1.9 nvidia-cusparse-cu12-12.3.1.170 nvidia-nvjitlink-cu12-12.4.127 torchviz-0.0.3
    Successfully installed torchinfo-1.8.0


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
from torch import nn, optim
from torchinfo import summary
from torchviz import make_dot
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

```


```python
# plt.rcParams.items()
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



```python
# 분류 클래스 명칭 리스트
# CIFAR10
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 분류 클래스 수,　10
n_output = len(list(set(classes)))
# n_output = len(set(classes))

# 결과 확인
print(n_output)
```

    10


## 과학습의 대응 방법

### 드랍 아웃 함수 동작 확인


```python
# 드랍 아웃 실험용 더미 데이터 작성

torch.manual_seed(123)
inputs = torch.randn(1, 10)
print(inputs)
```

    tensor([[-0.1115,  0.1204, -0.3696, -0.2404, -1.1969,  0.2093, -0.9724, -0.7550,
              0.3239, -0.1085]])



```python
# 드랍 아웃 함수 정의
dropout = nn.Dropout(0.5)

# 훈련 페이즈에서의 거동
dropout.train()
print("dropout.training = ", dropout.training) #
outputs = dropout(inputs)
print(outputs)

# 예측 페이즈에서의 거동
print("="*50)
dropout.eval()
print("dropout.training = ", dropout.training)
outputs = dropout(inputs)
print(outputs)
```

    dropout.training =  True
    tensor([[-0.0000,  0.2407, -0.0000, -0.4808, -0.0000,  0.0000, -1.9447, -0.0000,
              0.6478, -0.2170]])
    ==================================================
    dropout.training =  False
    tensor([[-0.1115,  0.1204, -0.3696, -0.2404, -1.1969,  0.2093, -0.9724, -0.7550,
              0.3239, -0.1085]])


## 공통 함수의 라이브러리화


```python
# # 공통 함수 다운로드
!git clone https://github.com/wikibook/pythonlibs.git

# # 공통 함수 불러오기
from pythonlibs.torch_lib1 import *

# # 공통 함수 확인
print(README)
```

    Cloning into 'pythonlibs'...
    remote: Enumerating objects: 25, done.[K
    remote: Counting objects: 100% (25/25), done.[K
    remote: Compressing objects: 100% (16/16), done.[K
    remote: Total 25 (delta 6), reused 25 (delta 6), pack-reused 0 (from 0)[K
    Receiving objects: 100% (25/25), 21.10 MiB | 19.24 MiB/s, done.
    Resolving deltas: 100% (6/6), done.
    Common Library for PyTorch
    Author: M. Akaishi


## 데이터 준비


```python
# Transforms의 정의

transform = transforms.Compose([
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
    transform = transform)

# 검증 데이터셋
test_set = datasets.CIFAR10(
    root = data_root,
    train = False,
    download = True,
    transform = transform)
```

    Downloading https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to ./data/cifar-10-python.tar.gz


    100%|██████████| 170M/170M [00:05<00:00, 31.1MB/s]


    Extracting ./data/cifar-10-python.tar.gz to ./data
    Files already downloaded and verified



```python
# 미니 배치 사이즈 지정
batch_size = 100

# 훈련용 데이터로더
# 훈련용이므로 셔플을 True로 설정
train_loader = DataLoader(train_set,
    batch_size = batch_size, shuffle = True)

# 검증용 데이터로더
# 검증용이므로 셔플하지 않음
test_loader = DataLoader(test_set,
    batch_size = batch_size, shuffle = False) # len(test_set)

# next(iter(train_loader))[0].shape # torch.Size([100, 3, 32, 32])
```


```python
# 처음 50개 이미지 출력
show_images_labels(test_loader, classes, None, None)
```


    
*(이미지 없음 / image not preserved: output_21_0.png)*
    


## 층을 깊게 쌓은 모델 구현하기


```python
class CNN_v2(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=(1,1))
        self.conv2 = nn.Conv2d(32, 32, 3, padding=(1,1))
        self.conv3 = nn.Conv2d(32, 64, 3, padding=(1,1))
        self.conv4 = nn.Conv2d(64, 64, 3, padding=(1,1))
        self.conv5 = nn.Conv2d(64, 128, 3, padding=(1,1))
        self.conv6 = nn.Conv2d(128, 128, 3, padding=(1,1))
        self.relu = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()
        self.maxpool = nn.MaxPool2d((2,2))
        self.l1 = nn.Linear(4*4*128, 128)
        self.l2 = nn.Linear(128, num_classes)

        self.features = nn.Sequential(
            self.conv1,
            self.relu,
            self.conv2,
            self.relu,
            self.maxpool,
            self.conv3,
            self.relu,
            self.conv4,
            self.relu,
            self.maxpool,
            self.conv5,
            self.relu,
            self.conv6,
            self.relu,
            self.maxpool,
            )

        self.classifier = nn.Sequential(
            self.l1,
            self.relu,
            self.l2
        )

    def forward(self, x):
        x1 = self.features(x)
        x2 = self.flatten(x1)
        x3 = self.classifier(x2)
        return x3
```


```python
# 손실 계산 그래프 시각화
net = CNN_v2(n_output).to(device)
criterion = nn.CrossEntropyLoss()
loss = eval_loss(test_loader, device, net, criterion)
g = make_dot(loss, params=dict(net.named_parameters()))
display(g)
```


    
*(이미지 없음 / image not preserved: output_24_0.svg)*
    



```python
# 난수 고정
torch_seed()

# 모델 인스턴스 생성
lr = 0.01
net = CNN_v2(n_output).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr)
history = np.zeros((0, 5))
```


```python
# 학습

num_epochs = 50
history = fit(net, optimizer, criterion, num_epochs, train_loader, test_loader, device, history)
```


      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [1/50], loss: 0.02303 acc: 0.10000 val_loss: 0.02303, val_acc: 0.10000



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [2/50], loss: 0.02303 acc: 0.10000 val_loss: 0.02303, val_acc: 0.10000



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [3/50], loss: 0.02302 acc: 0.10000 val_loss: 0.02302, val_acc: 0.10000



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [4/50], loss: 0.02302 acc: 0.10842 val_loss: 0.02302, val_acc: 0.10170



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [5/50], loss: 0.02302 acc: 0.13026 val_loss: 0.02302, val_acc: 0.17460



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [6/50], loss: 0.02301 acc: 0.16746 val_loss: 0.02300, val_acc: 0.19090



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [7/50], loss: 0.02298 acc: 0.18536 val_loss: 0.02294, val_acc: 0.21840



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [8/50], loss: 0.02245 acc: 0.21884 val_loss: 0.02097, val_acc: 0.23860



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [9/50], loss: 0.02016 acc: 0.26152 val_loss: 0.01946, val_acc: 0.29340



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [10/50], loss: 0.01907 acc: 0.31276 val_loss: 0.01891, val_acc: 0.31430



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [11/50], loss: 0.01797 acc: 0.35134 val_loss: 0.01699, val_acc: 0.38670



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [12/50], loss: 0.01697 acc: 0.38422 val_loss: 0.01625, val_acc: 0.41410



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [13/50], loss: 0.01629 acc: 0.40634 val_loss: 0.01560, val_acc: 0.42870



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [14/50], loss: 0.01574 acc: 0.42162 val_loss: 0.01614, val_acc: 0.41770



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [15/50], loss: 0.01530 acc: 0.43782 val_loss: 0.01511, val_acc: 0.44100



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [16/50], loss: 0.01482 acc: 0.45892 val_loss: 0.01441, val_acc: 0.47230



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [17/50], loss: 0.01442 acc: 0.47328 val_loss: 0.01401, val_acc: 0.48600



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [18/50], loss: 0.01399 acc: 0.49358 val_loss: 0.01419, val_acc: 0.48750



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [19/50], loss: 0.01356 acc: 0.51322 val_loss: 0.01344, val_acc: 0.51200



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [20/50], loss: 0.01312 acc: 0.53244 val_loss: 0.01295, val_acc: 0.53320



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [21/50], loss: 0.01260 acc: 0.54886 val_loss: 0.01234, val_acc: 0.55380



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [22/50], loss: 0.01219 acc: 0.56632 val_loss: 0.01208, val_acc: 0.56660



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [23/50], loss: 0.01176 acc: 0.58132 val_loss: 0.01205, val_acc: 0.57050



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [24/50], loss: 0.01143 acc: 0.59390 val_loss: 0.01157, val_acc: 0.58640



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [25/50], loss: 0.01099 acc: 0.61028 val_loss: 0.01120, val_acc: 0.59660



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [26/50], loss: 0.01063 acc: 0.62464 val_loss: 0.01097, val_acc: 0.61150



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [27/50], loss: 0.01023 acc: 0.64048 val_loss: 0.01107, val_acc: 0.60870



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [28/50], loss: 0.00987 acc: 0.65190 val_loss: 0.01052, val_acc: 0.63410



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [29/50], loss: 0.00956 acc: 0.66656 val_loss: 0.01003, val_acc: 0.64890



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [30/50], loss: 0.00919 acc: 0.68016 val_loss: 0.00976, val_acc: 0.65600



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [31/50], loss: 0.00887 acc: 0.68872 val_loss: 0.00976, val_acc: 0.65950



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [32/50], loss: 0.00858 acc: 0.69868 val_loss: 0.00948, val_acc: 0.66890



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [33/50], loss: 0.00825 acc: 0.71334 val_loss: 0.00987, val_acc: 0.65870



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [34/50], loss: 0.00796 acc: 0.72344 val_loss: 0.00919, val_acc: 0.68030



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [35/50], loss: 0.00762 acc: 0.73620 val_loss: 0.00939, val_acc: 0.67430



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [36/50], loss: 0.00734 acc: 0.74322 val_loss: 0.00929, val_acc: 0.67910



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [37/50], loss: 0.00707 acc: 0.75332 val_loss: 0.00939, val_acc: 0.68690



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [38/50], loss: 0.00676 acc: 0.76416 val_loss: 0.00923, val_acc: 0.68910



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [39/50], loss: 0.00643 acc: 0.77690 val_loss: 0.00930, val_acc: 0.69350



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [40/50], loss: 0.00617 acc: 0.78578 val_loss: 0.00927, val_acc: 0.69530



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [41/50], loss: 0.00589 acc: 0.79472 val_loss: 0.00950, val_acc: 0.68780



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [42/50], loss: 0.00560 acc: 0.80366 val_loss: 0.00934, val_acc: 0.70240



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [43/50], loss: 0.00530 acc: 0.81412 val_loss: 0.00919, val_acc: 0.70460



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [44/50], loss: 0.00498 acc: 0.82586 val_loss: 0.00946, val_acc: 0.70290



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [45/50], loss: 0.00473 acc: 0.83450 val_loss: 0.01020, val_acc: 0.68780



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [46/50], loss: 0.00439 acc: 0.84554 val_loss: 0.01047, val_acc: 0.69070



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [47/50], loss: 0.00409 acc: 0.85482 val_loss: 0.01021, val_acc: 0.69160



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [48/50], loss: 0.00379 acc: 0.86824 val_loss: 0.01048, val_acc: 0.69790



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [49/50], loss: 0.00348 acc: 0.87920 val_loss: 0.01061, val_acc: 0.70270



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [50/50], loss: 0.00318 acc: 0.88910 val_loss: 0.01101, val_acc: 0.69860



```python
evaluate_history(history)
```

    초기상태 : 손실 : 0.02303  정확도 : 0.10000
    최종상태 : 손실 : 0.01101 정확도 : 0.69860



    
*(이미지 없음 / image not preserved: output_27_1.png)*
    



    
*(이미지 없음 / image not preserved: output_27_2.png)*
    


## 최적화 함수 선택

### 모멘텀 설정


```python
# 난수 고정
torch_seed()

# 모델 인스턴스 생성
lr = 0.01
net = CNN_v2(n_output).to(device)
criterion = nn.CrossEntropyLoss()

# 최적화 함수에 모멘텀 값 설정
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
history2 = np.zeros((0, 5))
```


```python
# 학습

num_epochs = 20
history2 = fit(net, optimizer, criterion, num_epochs, train_loader, test_loader, device, history2)
```


      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [1/20], loss: 0.02286 acc: 0.12124 val_loss: 0.02051, val_acc: 0.24080



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [2/20], loss: 0.01800 acc: 0.33642 val_loss: 0.01524, val_acc: 0.42710



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [3/20], loss: 0.01413 acc: 0.48200 val_loss: 0.01265, val_acc: 0.54390



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [4/20], loss: 0.01160 acc: 0.58314 val_loss: 0.01092, val_acc: 0.61290



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [5/20], loss: 0.00968 acc: 0.65812 val_loss: 0.00887, val_acc: 0.68770



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [6/20], loss: 0.00815 acc: 0.71094 val_loss: 0.00826, val_acc: 0.71040



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [7/20], loss: 0.00686 acc: 0.76058 val_loss: 0.00765, val_acc: 0.73660



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [8/20], loss: 0.00583 acc: 0.79706 val_loss: 0.00726, val_acc: 0.75350



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [9/20], loss: 0.00491 acc: 0.82626 val_loss: 0.00709, val_acc: 0.76350



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [10/20], loss: 0.00421 acc: 0.85188 val_loss: 0.00785, val_acc: 0.75440



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [11/20], loss: 0.00340 acc: 0.87918 val_loss: 0.00756, val_acc: 0.76870



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [12/20], loss: 0.00283 acc: 0.89888 val_loss: 0.00788, val_acc: 0.76930



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [13/20], loss: 0.00235 acc: 0.91628 val_loss: 0.00895, val_acc: 0.74930



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [14/20], loss: 0.00190 acc: 0.93238 val_loss: 0.00996, val_acc: 0.75730



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [15/20], loss: 0.00159 acc: 0.94376 val_loss: 0.00988, val_acc: 0.76600



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [16/20], loss: 0.00134 acc: 0.95216 val_loss: 0.01132, val_acc: 0.75400



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [17/20], loss: 0.00127 acc: 0.95392 val_loss: 0.01084, val_acc: 0.76470



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [18/20], loss: 0.00104 acc: 0.96336 val_loss: 0.01188, val_acc: 0.76610



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [19/20], loss: 0.00096 acc: 0.96612 val_loss: 0.01390, val_acc: 0.75650



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [20/20], loss: 0.00088 acc: 0.96932 val_loss: 0.01271, val_acc: 0.76330



```python
evaluate_history(history2)
```

    초기상태 : 손실 : 0.02051  정확도 : 0.24080
    최종상태 : 손실 : 0.01271 정확도 : 0.76330



    
*(이미지 없음 / image not preserved: output_32_1.png)*
    



    
*(이미지 없음 / image not preserved: output_32_2.png)*
    


### Adam 함수 사용


```python
# 난수 고정
torch_seed()

# 모델 인스턴스 생성
net = CNN_v2(n_output).to(device)
criterion = nn.CrossEntropyLoss()

# 최적화 함수를 Adam으로 교체
optimizer = optim.Adam(net.parameters())
history3 = np.zeros((0, 5))
```


```python
# 학습

num_epochs = 20
history3 = fit(net, optimizer, criterion, num_epochs, train_loader, test_loader, device, history3)
```


      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [1/20], loss: 0.01599 acc: 0.41138 val_loss: 0.01251, val_acc: 0.54530



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [2/20], loss: 0.01100 acc: 0.60582 val_loss: 0.01017, val_acc: 0.63780



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [3/20], loss: 0.00886 acc: 0.68580 val_loss: 0.00896, val_acc: 0.68720



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [4/20], loss: 0.00738 acc: 0.73916 val_loss: 0.00795, val_acc: 0.72130



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [5/20], loss: 0.00621 acc: 0.78130 val_loss: 0.00715, val_acc: 0.75080



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [6/20], loss: 0.00534 acc: 0.81154 val_loss: 0.00712, val_acc: 0.75570



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [7/20], loss: 0.00451 acc: 0.84036 val_loss: 0.00714, val_acc: 0.76190



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [8/20], loss: 0.00375 acc: 0.86806 val_loss: 0.00735, val_acc: 0.76740



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [9/20], loss: 0.00308 acc: 0.89048 val_loss: 0.00792, val_acc: 0.76050



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [10/20], loss: 0.00250 acc: 0.91074 val_loss: 0.00868, val_acc: 0.76510



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [11/20], loss: 0.00201 acc: 0.92756 val_loss: 0.00994, val_acc: 0.76490



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [12/20], loss: 0.00169 acc: 0.94052 val_loss: 0.01072, val_acc: 0.75970



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [13/20], loss: 0.00146 acc: 0.94828 val_loss: 0.01099, val_acc: 0.75710



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [14/20], loss: 0.00129 acc: 0.95406 val_loss: 0.01236, val_acc: 0.75800



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [15/20], loss: 0.00123 acc: 0.95614 val_loss: 0.01213, val_acc: 0.75500



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [16/20], loss: 0.00105 acc: 0.96330 val_loss: 0.01296, val_acc: 0.75240



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [17/20], loss: 0.00098 acc: 0.96622 val_loss: 0.01316, val_acc: 0.74750



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [18/20], loss: 0.00098 acc: 0.96628 val_loss: 0.01438, val_acc: 0.74470



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [19/20], loss: 0.00093 acc: 0.96794 val_loss: 0.01480, val_acc: 0.75860



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [20/20], loss: 0.00077 acc: 0.97378 val_loss: 0.01433, val_acc: 0.75310



```python
evaluate_history(history3)
```

    초기상태 : 손실 : 0.01251  정확도 : 0.54530
    최종상태 : 손실 : 0.01433 정확도 : 0.75310



    
*(이미지 없음 / image not preserved: output_36_1.png)*
    



    
*(이미지 없음 / image not preserved: output_36_2.png)*
    


### 결과 비교


```python
# 결과 비교(검증 데이터의 정확도)
plt.figure(figsize=(9,8))
plt.plot(history[:,0], history[:,4], label='SGD', c='k',ls='dashed' )
plt.plot(history2[:,0], history2[:,4], label='SGD momentum=0.9', c='k')
plt.plot(history3[:,0], history3[:,4], label='Adam', c='b')
plt.title('최적화 함수 비교 결과(검증 데이터의 정확도)')
plt.xlabel('반복 횟수')
plt.ylabel('정확도')
plt.legend()
plt.show()
```


    
*(이미지 없음 / image not preserved: output_38_0.png)*
    


## 드랍 아웃 (Dropout)


```python
# 모델 정의

class CNN_v3(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=(1,1))
        self.conv2 = nn.Conv2d(32, 32, 3, padding=(1,1))
        self.conv3 = nn.Conv2d(32, 64, 3, padding=(1,1))
        self.conv4 = nn.Conv2d(64, 64, 3, padding=(1,1))
        self.conv5 = nn.Conv2d(64, 128, 3, padding=(1,1))
        self.conv6 = nn.Conv2d(128, 128, 3, padding=(1,1))
        self.relu = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()
        self.maxpool = nn.MaxPool2d((2,2))
        self.l1 = nn.Linear(4*4*128, 128)
        self.l2 = nn.Linear(128, 10)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.3)
        self.dropout3 = nn.Dropout(0.4)

        self.features = nn.Sequential(
            self.conv1,
            self.relu,
            self.conv2,
            self.relu,
            self.maxpool,
            self.dropout1,
            self.conv3,
            self.relu,
            self.conv4,
            self.relu,
            self.maxpool,
            self.dropout2,
            self.conv5,
            self.relu,
            self.conv6,
            self.relu,
            self.maxpool,
            self.dropout3,
            )

        self.classifier = nn.Sequential(
            self.l1,
            self.relu,
            self.dropout3,
            self.l2
        )

    def forward(self, x):
        x1 = self.features(x)
        x2 = self.flatten(x1)
        x3 = self.classifier(x2)
        return x3
```


```python
# 손실 계산 그래프 시각화
net = CNN_v3(n_output).to(device)
criterion = nn.CrossEntropyLoss()
loss = eval_loss(test_loader, device, net, criterion)
g = make_dot(loss, params=dict(net.named_parameters()))
display(g)
```


    
*(이미지 없음 / image not preserved: output_41_0.svg)*
    



```python
# 난수 고정
torch_seed()

# 모델 인스턴스 생성
net = CNN_v3(n_output).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())
history = np.zeros((0, 5))
```


```python
# 학습

num_epochs = 50
history = fit(net, optimizer, criterion, num_epochs, train_loader, test_loader, device, history)
```


      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [1/50], loss: 0.01706 acc: 0.36640 val_loss: 0.01291, val_acc: 0.52560



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [2/50], loss: 0.01274 acc: 0.53650 val_loss: 0.01046, val_acc: 0.62190



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [3/50], loss: 0.01108 acc: 0.60248 val_loss: 0.00986, val_acc: 0.64780



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [4/50], loss: 0.00991 acc: 0.64668 val_loss: 0.00858, val_acc: 0.69520



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [5/50], loss: 0.00908 acc: 0.68170 val_loss: 0.00779, val_acc: 0.72800



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [6/50], loss: 0.00841 acc: 0.70578 val_loss: 0.00748, val_acc: 0.73610



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [7/50], loss: 0.00794 acc: 0.72386 val_loss: 0.00716, val_acc: 0.75060



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [8/50], loss: 0.00753 acc: 0.73912 val_loss: 0.00698, val_acc: 0.75490



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [9/50], loss: 0.00715 acc: 0.75206 val_loss: 0.00637, val_acc: 0.78290



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [10/50], loss: 0.00684 acc: 0.76330 val_loss: 0.00684, val_acc: 0.76890



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [11/50], loss: 0.00662 acc: 0.77024 val_loss: 0.00657, val_acc: 0.77470



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [12/50], loss: 0.00643 acc: 0.77688 val_loss: 0.00629, val_acc: 0.78820



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [13/50], loss: 0.00624 acc: 0.78442 val_loss: 0.00594, val_acc: 0.79780



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [14/50], loss: 0.00597 acc: 0.79274 val_loss: 0.00579, val_acc: 0.80790



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [15/50], loss: 0.00583 acc: 0.79728 val_loss: 0.00601, val_acc: 0.80180



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [16/50], loss: 0.00572 acc: 0.80336 val_loss: 0.00574, val_acc: 0.80800



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [17/50], loss: 0.00551 acc: 0.80806 val_loss: 0.00592, val_acc: 0.79930



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [18/50], loss: 0.00545 acc: 0.81150 val_loss: 0.00605, val_acc: 0.80170



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [19/50], loss: 0.00531 acc: 0.81578 val_loss: 0.00554, val_acc: 0.81620



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [20/50], loss: 0.00526 acc: 0.81584 val_loss: 0.00565, val_acc: 0.81670



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [21/50], loss: 0.00511 acc: 0.82310 val_loss: 0.00569, val_acc: 0.81300



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [22/50], loss: 0.00507 acc: 0.82492 val_loss: 0.00572, val_acc: 0.81480



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [23/50], loss: 0.00498 acc: 0.82546 val_loss: 0.00540, val_acc: 0.82600



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [24/50], loss: 0.00498 acc: 0.82744 val_loss: 0.00548, val_acc: 0.81800



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [25/50], loss: 0.00482 acc: 0.83118 val_loss: 0.00570, val_acc: 0.81530



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [26/50], loss: 0.00476 acc: 0.83398 val_loss: 0.00549, val_acc: 0.81970



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [27/50], loss: 0.00472 acc: 0.83668 val_loss: 0.00538, val_acc: 0.82570



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [28/50], loss: 0.00474 acc: 0.83604 val_loss: 0.00551, val_acc: 0.81810



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [29/50], loss: 0.00458 acc: 0.84072 val_loss: 0.00560, val_acc: 0.82160



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [30/50], loss: 0.00459 acc: 0.83894 val_loss: 0.00524, val_acc: 0.82900



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [31/50], loss: 0.00451 acc: 0.84376 val_loss: 0.00524, val_acc: 0.83260



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [32/50], loss: 0.00448 acc: 0.84620 val_loss: 0.00540, val_acc: 0.82550



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [33/50], loss: 0.00440 acc: 0.84592 val_loss: 0.00538, val_acc: 0.82730



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [34/50], loss: 0.00433 acc: 0.84826 val_loss: 0.00524, val_acc: 0.83040



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [35/50], loss: 0.00440 acc: 0.84622 val_loss: 0.00533, val_acc: 0.83070



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [36/50], loss: 0.00433 acc: 0.84856 val_loss: 0.00530, val_acc: 0.83090



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [37/50], loss: 0.00428 acc: 0.85002 val_loss: 0.00551, val_acc: 0.82670



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [38/50], loss: 0.00421 acc: 0.85340 val_loss: 0.00536, val_acc: 0.83010



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [39/50], loss: 0.00419 acc: 0.85436 val_loss: 0.00554, val_acc: 0.82750



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [40/50], loss: 0.00420 acc: 0.85218 val_loss: 0.00551, val_acc: 0.83090



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [41/50], loss: 0.00414 acc: 0.85538 val_loss: 0.00554, val_acc: 0.82690



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [42/50], loss: 0.00410 acc: 0.85678 val_loss: 0.00533, val_acc: 0.83280



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [43/50], loss: 0.00409 acc: 0.85778 val_loss: 0.00545, val_acc: 0.82690



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [44/50], loss: 0.00403 acc: 0.85872 val_loss: 0.00552, val_acc: 0.82860



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [45/50], loss: 0.00398 acc: 0.86124 val_loss: 0.00554, val_acc: 0.82280



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [46/50], loss: 0.00400 acc: 0.86014 val_loss: 0.00533, val_acc: 0.82940



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [47/50], loss: 0.00399 acc: 0.86098 val_loss: 0.00536, val_acc: 0.83470



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [48/50], loss: 0.00395 acc: 0.86294 val_loss: 0.00555, val_acc: 0.83180



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [49/50], loss: 0.00393 acc: 0.86204 val_loss: 0.00535, val_acc: 0.83450



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [50/50], loss: 0.00390 acc: 0.86488 val_loss: 0.00531, val_acc: 0.83480



```python
evaluate_history(history)
```

    초기상태 : 손실 : 0.01291  정확도 : 0.52560
    최종상태 : 손실 : 0.00531 정확도 : 0.83480



    
*(이미지 없음 / image not preserved: output_44_1.png)*
    



    
*(이미지 없음 / image not preserved: output_44_2.png)*
    


## 배치 정규화


```python
class CNN_v4(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=(1,1))
        self.conv2 = nn.Conv2d(32, 32, 3, padding=(1,1))
        self.conv3 = nn.Conv2d(32, 64, 3, padding=(1,1))
        self.conv4 = nn.Conv2d(64, 64, 3, padding=(1,1))
        self.conv5 = nn.Conv2d(64, 128, 3, padding=(1,1))
        self.conv6 = nn.Conv2d(128, 128, 3, padding=(1,1))
        self.relu = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()
        self.maxpool = nn.MaxPool2d((2,2))
        self.l1 = nn.Linear(4*4*128, 128)
        self.l2 = nn.Linear(128, 10)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.3)
        self.dropout3 = nn.Dropout(0.4)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(128)
        self.bn6 = nn.BatchNorm2d(128)

        self.features = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.conv2,
            self.bn2,
            self.relu,
            self.maxpool,
            self.dropout1,
            self.conv3,
            self.bn3,
            self.relu,
            self.conv4,
            self.bn4,
            self.relu,
            self.maxpool,
            self.dropout2,
            self.conv5,
            self.bn5,
            self.relu,
            self.conv6,
            self.bn6,
            self.relu,
            self.maxpool,
            self.dropout3,
            )

        self.classifier = nn.Sequential(
            self.l1,
            self.relu,
            self.dropout3,
            self.l2
        )

    def forward(self, x):
        x1 = self.features(x)
        x2 = self.flatten(x1)
        x3 = self.classifier(x2)
        return x3
```


```python
# 난수 고정
torch_seed(1234)

# 모델 인스턴스 생성
net = CNN_v4(n_output).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())
history = np.zeros((0, 5))
```


```python
# 학습

num_epochs = 50
history = fit(net, optimizer, criterion, num_epochs, train_loader, test_loader, device, history)
```


      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [1/50], loss: 0.01528 acc: 0.43470 val_loss: 0.01214, val_acc: 0.55300



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [2/50], loss: 0.01144 acc: 0.59318 val_loss: 0.00939, val_acc: 0.66280



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [3/50], loss: 0.00985 acc: 0.65654 val_loss: 0.00916, val_acc: 0.66870



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [4/50], loss: 0.00897 acc: 0.68818 val_loss: 0.00826, val_acc: 0.72070



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [5/50], loss: 0.00826 acc: 0.71466 val_loss: 0.00745, val_acc: 0.73780



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [6/50], loss: 0.00782 acc: 0.73358 val_loss: 0.00764, val_acc: 0.73690



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [7/50], loss: 0.00740 acc: 0.74790 val_loss: 0.00617, val_acc: 0.78970



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [8/50], loss: 0.00705 acc: 0.76186 val_loss: 0.00612, val_acc: 0.78980



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [9/50], loss: 0.00668 acc: 0.77510 val_loss: 0.00656, val_acc: 0.78210



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [10/50], loss: 0.00640 acc: 0.78562 val_loss: 0.00543, val_acc: 0.81470



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [11/50], loss: 0.00619 acc: 0.79262 val_loss: 0.00553, val_acc: 0.81700



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [12/50], loss: 0.00595 acc: 0.79942 val_loss: 0.00579, val_acc: 0.80710



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [13/50], loss: 0.00572 acc: 0.80792 val_loss: 0.00553, val_acc: 0.81370



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [14/50], loss: 0.00553 acc: 0.81402 val_loss: 0.00522, val_acc: 0.83010



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [15/50], loss: 0.00530 acc: 0.82250 val_loss: 0.00503, val_acc: 0.83100



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [16/50], loss: 0.00516 acc: 0.82656 val_loss: 0.00530, val_acc: 0.82570



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [17/50], loss: 0.00501 acc: 0.83194 val_loss: 0.00509, val_acc: 0.83200



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [18/50], loss: 0.00481 acc: 0.83874 val_loss: 0.00494, val_acc: 0.83980



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [19/50], loss: 0.00464 acc: 0.84418 val_loss: 0.00457, val_acc: 0.85220



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [20/50], loss: 0.00452 acc: 0.84624 val_loss: 0.00479, val_acc: 0.84800



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [21/50], loss: 0.00442 acc: 0.85170 val_loss: 0.00483, val_acc: 0.84570



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [22/50], loss: 0.00431 acc: 0.85694 val_loss: 0.00453, val_acc: 0.85200



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [23/50], loss: 0.00423 acc: 0.85944 val_loss: 0.00465, val_acc: 0.85180



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [24/50], loss: 0.00414 acc: 0.86174 val_loss: 0.00449, val_acc: 0.85640



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [25/50], loss: 0.00399 acc: 0.86660 val_loss: 0.00435, val_acc: 0.85700



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [26/50], loss: 0.00391 acc: 0.87126 val_loss: 0.00469, val_acc: 0.85420



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [27/50], loss: 0.00377 acc: 0.87372 val_loss: 0.00458, val_acc: 0.85880



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [28/50], loss: 0.00372 acc: 0.87420 val_loss: 0.00457, val_acc: 0.85300



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [29/50], loss: 0.00364 acc: 0.87772 val_loss: 0.00457, val_acc: 0.85820



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [30/50], loss: 0.00357 acc: 0.88054 val_loss: 0.00486, val_acc: 0.84910



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [31/50], loss: 0.00350 acc: 0.88028 val_loss: 0.00449, val_acc: 0.86350



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [32/50], loss: 0.00345 acc: 0.88308 val_loss: 0.00445, val_acc: 0.86280



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [33/50], loss: 0.00331 acc: 0.88866 val_loss: 0.00437, val_acc: 0.86590



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [34/50], loss: 0.00324 acc: 0.89076 val_loss: 0.00471, val_acc: 0.86070



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [35/50], loss: 0.00317 acc: 0.89258 val_loss: 0.00441, val_acc: 0.86410



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [36/50], loss: 0.00311 acc: 0.89514 val_loss: 0.00456, val_acc: 0.86290



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [37/50], loss: 0.00312 acc: 0.89386 val_loss: 0.00447, val_acc: 0.86610



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [38/50], loss: 0.00306 acc: 0.89698 val_loss: 0.00443, val_acc: 0.86450



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [39/50], loss: 0.00300 acc: 0.89802 val_loss: 0.00447, val_acc: 0.86440



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [40/50], loss: 0.00289 acc: 0.90246 val_loss: 0.00463, val_acc: 0.86520



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [41/50], loss: 0.00289 acc: 0.90152 val_loss: 0.00460, val_acc: 0.86280



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [42/50], loss: 0.00283 acc: 0.90374 val_loss: 0.00470, val_acc: 0.86310



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [43/50], loss: 0.00284 acc: 0.90410 val_loss: 0.00448, val_acc: 0.86800



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [44/50], loss: 0.00276 acc: 0.90616 val_loss: 0.00448, val_acc: 0.86810



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [45/50], loss: 0.00275 acc: 0.90514 val_loss: 0.00438, val_acc: 0.86970



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [46/50], loss: 0.00269 acc: 0.90808 val_loss: 0.00465, val_acc: 0.86320



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [47/50], loss: 0.00267 acc: 0.91102 val_loss: 0.00457, val_acc: 0.86990



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [48/50], loss: 0.00259 acc: 0.91300 val_loss: 0.00453, val_acc: 0.86730



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [49/50], loss: 0.00254 acc: 0.91284 val_loss: 0.00451, val_acc: 0.86760



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [50/50], loss: 0.00254 acc: 0.91330 val_loss: 0.00471, val_acc: 0.86690



```python
evaluate_history(history)
```

    초기상태 : 손실 : 0.01214  정확도 : 0.55300
    최종상태 : 손실 : 0.00471 정확도 : 0.86690



    
*(이미지 없음 / image not preserved: output_49_1.png)*
    



    
*(이미지 없음 / image not preserved: output_49_2.png)*
    


## 데이터 증강 기법


```python
# 훈련 데이터용: 정규화에 반전과 RandomErasing 추가
transform_train = transforms.Compose([
  transforms.RandomHorizontalFlip(p=0.5), # 랜덤으로 좌우 반전
  transforms.RandomRotation(30),  # 랜덤으로 -30~30도 회전
  transforms.ColorJitter(brightness=0.5, contrast=0.5),  # 밝기, 대비 조절
  transforms.ToTensor(),
  transforms.Normalize(0.5, 0.5),
  transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)
])

# class RandomErasing(
#     p: float = 0.5, # Probability that the Random Erasing operation will be performed.
#     scale: Any = (0.02, 0.33), # Range of proportion of erased area against the input image area.
#     ratio: Any = (0.3, 3.3), # Aspect ratio range of the erased rectangle.
#     value: int = 0,
#     inplace: bool = False
# )
```


```python
# transfrom_train을 사용한 데이터셋 정의
train_set2 = datasets.CIFAR10(
    root = data_root,
    train = True,
    download = True,
    transform = transform_train) # n = 50000

# traisform_train을 사용한 데이터로더 정의
batch_size = 100
train_loader2 = DataLoader(train_set2, batch_size=batch_size, shuffle=True)
```

    Files already downloaded and verified



```python
# 새로운 훈련 데이터의 처음 50개 표시

# 난수 고정
# torch_seed(12345)

show_images_labels(train_loader2, classes, None, None)
```


    
*(이미지 없음 / image not preserved: output_53_0.png)*
    



```python
# 난수 고정
torch_seed()

# 모델 인스턴스 생성
net = CNN_v4(n_output).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())
history = np.zeros((0, 5))
```


```python
# 학습
# 동일한 모델에서 train_loader2로 데이터를 변경

num_epochs = 100
history = fit(net, optimizer, criterion, num_epochs,
        train_loader2, test_loader, device, history)
```


      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [1/100], loss: 0.01909 acc: 0.28624 val_loss: 0.01492, val_acc: 0.43330



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [2/100], loss: 0.01633 acc: 0.39462 val_loss: 0.01272, val_acc: 0.53950



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [3/100], loss: 0.01525 acc: 0.44124 val_loss: 0.01203, val_acc: 0.55940



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [4/100], loss: 0.01460 acc: 0.46666 val_loss: 0.01097, val_acc: 0.59450



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [5/100], loss: 0.01405 acc: 0.49264 val_loss: 0.01097, val_acc: 0.60270



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [6/100], loss: 0.01367 acc: 0.50700 val_loss: 0.00968, val_acc: 0.65150



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [7/100], loss: 0.01323 acc: 0.52248 val_loss: 0.00940, val_acc: 0.65820



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [8/100], loss: 0.01291 acc: 0.53722 val_loss: 0.00897, val_acc: 0.68570



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [9/100], loss: 0.01260 acc: 0.54792 val_loss: 0.00879, val_acc: 0.68760



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [10/100], loss: 0.01245 acc: 0.55782 val_loss: 0.00822, val_acc: 0.70140



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [11/100], loss: 0.01219 acc: 0.56516 val_loss: 0.00829, val_acc: 0.69980



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [12/100], loss: 0.01208 acc: 0.57138 val_loss: 0.00781, val_acc: 0.71990



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [13/100], loss: 0.01188 acc: 0.57888 val_loss: 0.00825, val_acc: 0.69420



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [14/100], loss: 0.01169 acc: 0.58570 val_loss: 0.00767, val_acc: 0.73030



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [15/100], loss: 0.01155 acc: 0.59370 val_loss: 0.00746, val_acc: 0.74000



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [16/100], loss: 0.01138 acc: 0.59840 val_loss: 0.00736, val_acc: 0.74180



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [17/100], loss: 0.01123 acc: 0.60944 val_loss: 0.00750, val_acc: 0.74150



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [18/100], loss: 0.01105 acc: 0.61722 val_loss: 0.00728, val_acc: 0.74580



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [19/100], loss: 0.01095 acc: 0.61762 val_loss: 0.00722, val_acc: 0.74990



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [20/100], loss: 0.01074 acc: 0.62700 val_loss: 0.00677, val_acc: 0.76910



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [21/100], loss: 0.01057 acc: 0.63530 val_loss: 0.00682, val_acc: 0.76630



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [22/100], loss: 0.01052 acc: 0.64064 val_loss: 0.00659, val_acc: 0.77010



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [23/100], loss: 0.01031 acc: 0.64692 val_loss: 0.00641, val_acc: 0.77840



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [24/100], loss: 0.01021 acc: 0.65134 val_loss: 0.00638, val_acc: 0.77990



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [25/100], loss: 0.01017 acc: 0.65226 val_loss: 0.00628, val_acc: 0.78550



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [26/100], loss: 0.00999 acc: 0.65648 val_loss: 0.00659, val_acc: 0.77610



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [27/100], loss: 0.00992 acc: 0.66218 val_loss: 0.00642, val_acc: 0.77660



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [28/100], loss: 0.00968 acc: 0.66978 val_loss: 0.00606, val_acc: 0.79070



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [29/100], loss: 0.00950 acc: 0.67684 val_loss: 0.00587, val_acc: 0.80350



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [30/100], loss: 0.00944 acc: 0.67974 val_loss: 0.00579, val_acc: 0.80410



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [31/100], loss: 0.00923 acc: 0.68678 val_loss: 0.00582, val_acc: 0.80350



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [32/100], loss: 0.00916 acc: 0.68824 val_loss: 0.00568, val_acc: 0.80420



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [33/100], loss: 0.00911 acc: 0.68886 val_loss: 0.00576, val_acc: 0.80600



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [34/100], loss: 0.00906 acc: 0.69312 val_loss: 0.00554, val_acc: 0.81220



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [35/100], loss: 0.00892 acc: 0.69842 val_loss: 0.00540, val_acc: 0.81810



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [36/100], loss: 0.00884 acc: 0.69916 val_loss: 0.00560, val_acc: 0.80830



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [37/100], loss: 0.00881 acc: 0.69796 val_loss: 0.00541, val_acc: 0.81530



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [38/100], loss: 0.00864 acc: 0.70714 val_loss: 0.00529, val_acc: 0.81650



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [39/100], loss: 0.00864 acc: 0.70732 val_loss: 0.00530, val_acc: 0.82240



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [40/100], loss: 0.00861 acc: 0.70976 val_loss: 0.00535, val_acc: 0.81900



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [41/100], loss: 0.00848 acc: 0.71040 val_loss: 0.00508, val_acc: 0.82800



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [42/100], loss: 0.00845 acc: 0.71416 val_loss: 0.00511, val_acc: 0.82990



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [43/100], loss: 0.00837 acc: 0.71588 val_loss: 0.00513, val_acc: 0.82560



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [44/100], loss: 0.00838 acc: 0.71636 val_loss: 0.00513, val_acc: 0.82560



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [45/100], loss: 0.00836 acc: 0.71560 val_loss: 0.00520, val_acc: 0.82630



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [46/100], loss: 0.00824 acc: 0.71866 val_loss: 0.00508, val_acc: 0.82460



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [47/100], loss: 0.00821 acc: 0.72232 val_loss: 0.00508, val_acc: 0.82770



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [48/100], loss: 0.00816 acc: 0.72570 val_loss: 0.00518, val_acc: 0.82670



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [49/100], loss: 0.00811 acc: 0.72672 val_loss: 0.00507, val_acc: 0.82880



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [50/100], loss: 0.00811 acc: 0.72316 val_loss: 0.00486, val_acc: 0.83420



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [51/100], loss: 0.00805 acc: 0.72550 val_loss: 0.00502, val_acc: 0.82920



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [52/100], loss: 0.00804 acc: 0.72694 val_loss: 0.00503, val_acc: 0.83090



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [53/100], loss: 0.00793 acc: 0.73008 val_loss: 0.00492, val_acc: 0.83440



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [54/100], loss: 0.00795 acc: 0.72896 val_loss: 0.00489, val_acc: 0.83540



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [55/100], loss: 0.00793 acc: 0.73024 val_loss: 0.00497, val_acc: 0.83040



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [56/100], loss: 0.00785 acc: 0.73370 val_loss: 0.00478, val_acc: 0.84020



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [57/100], loss: 0.00788 acc: 0.73582 val_loss: 0.00484, val_acc: 0.83520



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [58/100], loss: 0.00787 acc: 0.73234 val_loss: 0.00472, val_acc: 0.84030



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [59/100], loss: 0.00776 acc: 0.73724 val_loss: 0.00464, val_acc: 0.84440



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [60/100], loss: 0.00772 acc: 0.73860 val_loss: 0.00494, val_acc: 0.83260



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [61/100], loss: 0.00773 acc: 0.73612 val_loss: 0.00467, val_acc: 0.84380



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [62/100], loss: 0.00768 acc: 0.74134 val_loss: 0.00492, val_acc: 0.83360



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [63/100], loss: 0.00770 acc: 0.73904 val_loss: 0.00466, val_acc: 0.84390



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [64/100], loss: 0.00769 acc: 0.73768 val_loss: 0.00470, val_acc: 0.83980



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [65/100], loss: 0.00767 acc: 0.74166 val_loss: 0.00476, val_acc: 0.83290



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [66/100], loss: 0.00762 acc: 0.73976 val_loss: 0.00463, val_acc: 0.84040



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [67/100], loss: 0.00759 acc: 0.74436 val_loss: 0.00463, val_acc: 0.84290



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [68/100], loss: 0.00756 acc: 0.74104 val_loss: 0.00470, val_acc: 0.83770



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [69/100], loss: 0.00758 acc: 0.74248 val_loss: 0.00487, val_acc: 0.83830



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [70/100], loss: 0.00747 acc: 0.74422 val_loss: 0.00454, val_acc: 0.84520



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [71/100], loss: 0.00747 acc: 0.74526 val_loss: 0.00453, val_acc: 0.84930



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [72/100], loss: 0.00747 acc: 0.74640 val_loss: 0.00451, val_acc: 0.84690



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [73/100], loss: 0.00747 acc: 0.74662 val_loss: 0.00463, val_acc: 0.84710



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [74/100], loss: 0.00740 acc: 0.75070 val_loss: 0.00453, val_acc: 0.84650



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [75/100], loss: 0.00739 acc: 0.74802 val_loss: 0.00453, val_acc: 0.84610



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [76/100], loss: 0.00744 acc: 0.74570 val_loss: 0.00457, val_acc: 0.84780



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [77/100], loss: 0.00741 acc: 0.74688 val_loss: 0.00439, val_acc: 0.85030



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [78/100], loss: 0.00732 acc: 0.75102 val_loss: 0.00436, val_acc: 0.84900



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [79/100], loss: 0.00735 acc: 0.75086 val_loss: 0.00435, val_acc: 0.85230



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [80/100], loss: 0.00732 acc: 0.75116 val_loss: 0.00462, val_acc: 0.84310



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [81/100], loss: 0.00728 acc: 0.75200 val_loss: 0.00426, val_acc: 0.85560



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [82/100], loss: 0.00725 acc: 0.75360 val_loss: 0.00438, val_acc: 0.85010



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [83/100], loss: 0.00718 acc: 0.75484 val_loss: 0.00439, val_acc: 0.85010



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [84/100], loss: 0.00723 acc: 0.75180 val_loss: 0.00446, val_acc: 0.85080



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [85/100], loss: 0.00724 acc: 0.75460 val_loss: 0.00438, val_acc: 0.85210



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [86/100], loss: 0.00717 acc: 0.75600 val_loss: 0.00448, val_acc: 0.85250



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [87/100], loss: 0.00724 acc: 0.75446 val_loss: 0.00441, val_acc: 0.85280



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [88/100], loss: 0.00722 acc: 0.75380 val_loss: 0.00435, val_acc: 0.85320



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [89/100], loss: 0.00717 acc: 0.75504 val_loss: 0.00440, val_acc: 0.85190



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [90/100], loss: 0.00715 acc: 0.75584 val_loss: 0.00422, val_acc: 0.85560



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [91/100], loss: 0.00714 acc: 0.75746 val_loss: 0.00451, val_acc: 0.84120



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [92/100], loss: 0.00711 acc: 0.75714 val_loss: 0.00457, val_acc: 0.84820



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [93/100], loss: 0.00713 acc: 0.75748 val_loss: 0.00438, val_acc: 0.84870



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [94/100], loss: 0.00707 acc: 0.76024 val_loss: 0.00440, val_acc: 0.85050



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [95/100], loss: 0.00712 acc: 0.75752 val_loss: 0.00433, val_acc: 0.85420



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [96/100], loss: 0.00706 acc: 0.76022 val_loss: 0.00435, val_acc: 0.85400



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [97/100], loss: 0.00709 acc: 0.75894 val_loss: 0.00436, val_acc: 0.85380



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [98/100], loss: 0.00706 acc: 0.75904 val_loss: 0.00438, val_acc: 0.85440



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [99/100], loss: 0.00710 acc: 0.75758 val_loss: 0.00448, val_acc: 0.84800



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [100/100], loss: 0.00698 acc: 0.76160 val_loss: 0.00426, val_acc: 0.85370



```python
evaluate_history(history)
```

    초기상태 : 손실 : 0.01492  정확도 : 0.43330
    최종상태 : 손실 : 0.00426 정확도 : 0.85370



    
*(이미지 없음 / image not preserved: output_56_1.png)*
    



    
*(이미지 없음 / image not preserved: output_56_2.png)*
    



```python
show_images_labels(test_loader, classes, net, device)
```


    
*(이미지 없음 / image not preserved: output_57_0.png)*
    



```python
# 잘못 예측한 38번째 데이터 추출
for images, labels in test_loader:
    break
image = images[37]
label = labels[37]

# 이미지 확인
plt.figure(figsize=(3,3))
w = image.numpy().copy()
w2 = np.transpose(w, (1, 2, 0))
w3 = (w2 + 1)/2
plt.title(classes[label])
plt.imshow(w3)
plt.show()
```


    
*(이미지 없음 / image not preserved: output_58_0.png)*
    



```python
# 예측 값 출력
image = image.view(1, 3, 32, 32)
image = image.to(device)
output = net(image)

# 라벨 별 확률 값 출력
probs = torch.softmax(output, dim=1)
probs_np = probs.data.to('cpu').numpy()[0]
print(probs_np)
values = np.frompyfunc(lambda x: f'{x:.04f}', 1, 1)(probs_np)
values
names = np.array(classes)
tbl = np.array([names, values]).T
tbl
```

    [0.0002 0.4003 0.0001 0.0003 0.     0.0006 0.     0.     0.0015 0.5971]





    array([['plane', '0.0002'],
           ['car', '0.4003'],
           ['bird', '0.0001'],
           ['cat', '0.0003'],
           ['deer', '0.0000'],
           ['dog', '0.0006'],
           ['frog', '0.0000'],
           ['horse', '0.0000'],
           ['ship', '0.0015'],
           ['truck', '0.5971']], dtype=object)



## 칼럼 배치 정규화를 사용할 때 주의할 점

### 잘못된 모델 클래스 정의의 예시


```python
class CNN_v5(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=(1,1))
        self.conv2 = nn.Conv2d(32, 32, 3, padding=(1,1))
        self.conv3 = nn.Conv2d(32, 64, 3, padding=(1,1))
        self.conv4 = nn.Conv2d(64, 64, 3, padding=(1,1))
        self.conv5 = nn.Conv2d(64, 128, 3, padding=(1,1))
        self.conv6 = nn.Conv2d(128, 128, 3, padding=(1,1))
        self.relu = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()
        self.maxpool = nn.MaxPool2d((2,2))
        self.l1 = nn.Linear(4*4*128, 128)
        self.l2 = nn.Linear(128, 10)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.3)
        self.dropout3 = nn.Dropout(0.4)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        self.features = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.conv2,
            self.bn1,
            self.relu,
            self.maxpool,
            self.dropout1,
            self.conv3,
            self.bn2,
            self.relu,
            self.conv4,
            self.bn2,
            self.relu,
            self.maxpool,
            self.dropout2,
            self.conv5,
            self.bn3,
            self.relu,
            self.conv6,
            self.bn3,
            self.relu,
            self.maxpool,
            self.dropout3,
            )

        self.classifier = nn.Sequential(
            self.l1,
            self.relu,
            self.dropout3,
            self.l2
        )

    def forward(self, x):
        x1 = self.features(x)
        x2 = self.flatten(x1)
        x3 = self.classifier(x2)
        return x3
```


```python
# 난수 고정
torch_seed()

# 모델 인스턴스 생성
net = CNN_v5(n_output).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())
history = np.zeros((0, 5))
```


```python
# 학습

num_epochs = 50
history = fit(net, optimizer, criterion, num_epochs, train_loader, test_loader, device, history)
```


```python
# 손실 계산 그래프 시각화
net = CNN_v5(n_output).to(device)
criterion = nn.CrossEntropyLoss()
loss = eval_loss(test_loader, device, net, criterion)
g = make_dot(loss, params=dict(net.named_parameters()))
display(g)
```


    
*(이미지 없음 / image not preserved: output_65_0.svg)*
    


## 칼럼 배치 정규화에서 처리하는 내용


```python
# 입력용 더미 데이터 작성

torch.manual_seed(123)
inputs = torch.randn(1, 1, 10)
print(inputs)
```

    tensor([[[-0.1115,  0.1204, -0.3696, -0.2404, -1.1969,  0.2093, -0.9724,
              -0.7550,  0.3239, -0.1085]]])



```python
# 입력 미니 배치 데이터의 통계량 산출

i_mean = inputs.mean()
i_var = inputs.var(unbiased=True)
i_std = inputs.std(unbiased=False)
print(i_mean, i_std, i_var)
```

    tensor(-0.3101) tensor(0.4867) tensor(0.2632)



```python
# BN 함수의 정의

bn = nn.BatchNorm1d(1)
print(bn.running_mean) # 러닝 평균 (배치 평균의 지수 이동 평균)
print(bn.running_var) # 러닝 분산 (배치 분산의 지수 이동 평균)
print(bn.weight.data) # 배치 정규화 스케일링
print(bn.bias.data) # 배치 정규화 이동
```

    tensor([0.])
    tensor([1.])
    tensor([1.])
    tensor([0.])



```python
# BN 함수의 유사 호출

bn.train()
print('===훈련 페이즈 1===')
outputs1 = bn(inputs)
print(outputs1.data)
print(bn.running_mean)
print(bn.running_var)

bn.eval()
print('===예측 페이즈 1===')
outputs2 = bn(inputs)
print(outputs2.data)
print(bn.running_mean)
print(bn.running_var)

bn.train()
print('===훈련 페이즈 2===')
outputs3 = bn(inputs)
print(outputs3.data)
print(bn.running_mean)
print(bn.running_var)

bn.eval()
print('===예측 페이즈 2===')
outputs4 = bn(inputs)
print(outputs4.data)
print(bn.running_mean)
print(bn.running_var)

```

    ===훈련 페이즈 1===
    tensor([[[ 0.4081,  0.8844, -0.1224,  0.1431, -1.8222,  1.0671, -1.3608,
              -0.9143,  1.3027,  0.4142]]])
    tensor([-0.0310])
    tensor([0.9263])
    ===예측 페이즈 1===
    tensor([[[-0.0836,  0.1573, -0.3518, -0.2176, -1.2114,  0.2496, -0.9781,
              -0.7523,  0.3688, -0.0805]]])
    tensor([-0.0310])
    tensor([0.9263])
    ===훈련 페이즈 2===
    tensor([[[ 0.4081,  0.8844, -0.1224,  0.1431, -1.8222,  1.0671, -1.3608,
              -0.9143,  1.3027,  0.4142]]])
    tensor([-0.0589])
    tensor([0.8600])
    ===예측 페이즈 2===
    tensor([[[-0.0567,  0.1933, -0.3351, -0.1957, -1.2271,  0.2892, -0.9850,
              -0.7507,  0.4128, -0.0535]]])
    tensor([-0.0589])
    tensor([0.8600])



```python
# 훈련 페이즈의 출력

xt = (inputs - i_mean)/i_std * bn.weight + bn.bias
print(xt.data)

print(outputs1.data)
```

    tensor([[[ 0.4081,  0.8845, -0.1224,  0.1431, -1.8223,  1.0671, -1.3608,
              -0.9143,  1.3027,  0.4142]]])
    tensor([[[ 0.4081,  0.8844, -0.1224,  0.1431, -1.8222,  1.0671, -1.3608,
              -0.9143,  1.3027,  0.4142]]])



```python
# 예측 페이즈의 출력

xp = (inputs-bn.running_mean)/torch.sqrt(bn.running_var)
print(xp.data)

print(outputs4.data)
```

    tensor([[[-0.0567,  0.1933, -0.3351, -0.1957, -1.2271,  0.2892, -0.9850,
              -0.7507,  0.4128, -0.0535]]])
    tensor([[[-0.0567,  0.1933, -0.3351, -0.1957, -1.2271,  0.2892, -0.9850,
              -0.7507,  0.4128, -0.0535]]])



```python
# running_mean과 runnung_var의 계산식

# 초깃값
mean0 = 0
var0 = 1
momentum = bn.momentum
print("momentum = ", momentum)
# 이동 평균 계산 1회차
mean1 = (1-momentum) * mean0 +  momentum * i_mean
var1 = (1-momentum) * var0 +  momentum * i_var
print(mean1, var1)

# 이동 평균 계산 2회차
mean2 = (1-momentum) * mean1 +  momentum * i_mean
var2 = (1-momentum) * var1 +  momentum * i_var
print(mean2, var2)
```
