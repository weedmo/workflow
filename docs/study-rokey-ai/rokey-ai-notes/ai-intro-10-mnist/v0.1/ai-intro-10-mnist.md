# 10장 MNIST를 활용한 숫자 인식

> **Migrated from Quartz wiki** (`wiki/study/rokey-ai/basic/ai-intro-10-mnist.md`, updated 2026-09-26)  
> tags: rokey, ai, notebook, basic, study  
> source: raw/2026-09-26-rokey-ai/basic/강의_3기_AI개론_10차시__MNIST_.ipynb  
> source: https://github.com/weedmo/Rokey-AI  

* "부록3 매트플롯립 입문"에서 한글 폰트를 올바르게 출력하기 위한 설치 방법을 설명했다. 설치 방법은 다음과 같다.


```python
# 한글 폰트 설치

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
import torch.nn.functional as F
from torchviz import make_dot
from torchinfo import summary
from tqdm.notebook import tqdm

import torchvision.transforms as transforms
import torchvision.datasets as datasets
```


```python
# 기본 폰트 설정
# 윈도우에서는 "malgun.ttf" 혹은 "NanumBarunGothic.ttf" 등을 사용할 수 있다. 맥에서는 "AppleGothic.ttf"
plt.rcParams['font.family'] = font_name

# 기본 폰트 사이즈 변경
plt.rcParams['font.size'] = 14

# 기본 그래프 사이즈 변경
plt.rcParams['figure.figsize'] = (6,6)

# 기본 그리드 표시
# 필요에 따라 설정할 때는, plt.grid()
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.linestyle'] = ':'


# 마이너스 기호 정상 출력
plt.rcParams['axes.unicode_minus'] = False

# 넘파이 부동소수점 자릿수 표시
np.set_printoptions(suppress=True, precision=4)
```

## MNIST 숫자 인식

### 활성화 함수와 ReLU 함수


```python
# ReLU 함수의 그래프

relu = nn.ReLU()
x_np = np.arange(-2, 2.1, 0.25)
x = torch.tensor(x_np).float()
y = relu(x)

plt.plot(x.data, y.data)
plt.title('ReLU 함수')
plt.show()
```


    
*(이미지 없음 / image not preserved: output_10_0.png)*
    


### GPU 디바이스 확인


```python
# 디바이스 할당
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
```

    cpu



```python
# 텐서 변수 x, y
x_np = np.arange(-2.0, 2.1, 0.25)
y_np = np.arange(-1.0, 3.1, 0.25)
x = torch.tensor(x_np).float()
y = torch.tensor(y_np).float()

# x와 y 사이의 연산
z = x * y
print(z)
```

    tensor([ 2.0000,  1.3125,  0.7500,  0.3125, -0.0000, -0.1875, -0.2500, -0.1875,
             0.0000,  0.3125,  0.7500,  1.3125,  2.0000,  2.8125,  3.7500,  4.8125,
             6.0000])



```python
# 변수 x를 GPU로 보냄
x = x.to(device)

# 변수 x와 y의 디바이스 속성 확인
print('x: ', x.device)
print('y: ', y.device)
```

    x:  cpu
    y:  cpu



```python
# 이 상태에서 x와 y의 연산을 수행하면...

z = x * y
```


```python
# y도 GPU로 보냄
y = y.to(device)

# 연산이 가능해짐
z = x * y
print(z)
print("z.device = ", z.device)
```

    tensor([ 2.0000,  1.3125,  0.7500,  0.3125, -0.0000, -0.1875, -0.2500, -0.1875,
             0.0000,  0.3125,  0.7500,  1.3125,  2.0000,  2.8125,  3.7500,  4.8125,
             6.0000])
    z.device =  cpu


### MNIST Dataset을 활용해 불러오기


```python
# 라이브러리 임포트
import torchvision.datasets as datasets

# 다운로드받을 디렉터리명
data_root = './data'

train_set0 = datasets.MNIST(
    # 원본 데이터를 다운로드받을 디렉터리 지정
    root = data_root,
    # 훈련 데이터인지 또는 검증 데이터인지
    train = True,
    # 원본 데이터가 없는 경우, 다운로드를 실행하는지 여부
    download = True)
```

    Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
    Failed to download (trying next):
    HTTP Error 404: Not Found
    
    Downloading https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz
    Downloading https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz to ./data/MNIST/raw/train-images-idx3-ubyte.gz


    100%|██████████| 9.91M/9.91M [00:00<00:00, 91.8MB/s]

    Extracting ./data/MNIST/raw/train-images-idx3-ubyte.gz to ./data/MNIST/raw


    


    
    Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
    Failed to download (trying next):
    HTTP Error 404: Not Found
    
    Downloading https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz
    Downloading https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz to ./data/MNIST/raw/train-labels-idx1-ubyte.gz


    100%|██████████| 28.9k/28.9k [00:00<00:00, 35.9MB/s]


    Extracting ./data/MNIST/raw/train-labels-idx1-ubyte.gz to ./data/MNIST/raw
    
    Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
    Failed to download (trying next):
    HTTP Error 404: Not Found
    
    Downloading https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz
    Downloading https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz to ./data/MNIST/raw/t10k-images-idx3-ubyte.gz


    100%|██████████| 1.65M/1.65M [00:00<00:00, 63.8MB/s]


    Extracting ./data/MNIST/raw/t10k-images-idx3-ubyte.gz to ./data/MNIST/raw
    
    Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
    Failed to download (trying next):
    HTTP Error 404: Not Found
    
    Downloading https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz
    Downloading https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz to ./data/MNIST/raw/t10k-labels-idx1-ubyte.gz


    100%|██████████| 4.54k/4.54k [00:00<00:00, 2.99MB/s]


    Extracting ./data/MNIST/raw/t10k-labels-idx1-ubyte.gz to ./data/MNIST/raw
    



```python
# 다운로드한 파일 확인
# 리눅스 명령어
!ls -lR ./data/MNIST

# Window 명령어
# !dir /s data\MNIST
```

    ./data/MNIST:
    total 4
    drwxr-xr-x 2 root root 4096 Feb 10 03:14 raw
    
    ./data/MNIST/raw:
    total 65008
    -rw-r--r-- 1 root root  7840016 Feb 10 03:14 t10k-images-idx3-ubyte
    -rw-r--r-- 1 root root  1648877 Feb 10 03:14 t10k-images-idx3-ubyte.gz
    -rw-r--r-- 1 root root    10008 Feb 10 03:14 t10k-labels-idx1-ubyte
    -rw-r--r-- 1 root root     4542 Feb 10 03:14 t10k-labels-idx1-ubyte.gz
    -rw-r--r-- 1 root root 47040016 Feb 10 03:14 train-images-idx3-ubyte
    -rw-r--r-- 1 root root  9912422 Feb 10 03:14 train-images-idx3-ubyte.gz
    -rw-r--r-- 1 root root    60008 Feb 10 03:14 train-labels-idx1-ubyte
    -rw-r--r-- 1 root root    28881 Feb 10 03:14 train-labels-idx1-ubyte.gz



```python
# 데이터 건수 확인
print("train_set0 타입:", type(train_set0))
print("train_set0 : \n", train_set0)
print('데이터 건수: ', len(train_set0))


# 첫번째 요소 가져오기
image, label = train_set0[0]

# 데이터 타입 확인
print("="*50)
print('입력 데이터 타입 : ', type(image)) # <class 'PIL.Image.Image'>
print('정답 데이터 타입 : ', type(label)) # <class 'int'>

print("max = ", np.array(image).max())
print("min = ", np.array(image).min())

```

    train_set0 타입: <class 'torchvision.datasets.mnist.MNIST'>
    train_set0 : 
     Dataset MNIST
        Number of datapoints: 60000
        Root location: ./data
        Split: Train
    데이터 건수:  60000
    ==================================================
    입력 데이터 타입 :  <class 'PIL.Image.Image'>
    정답 데이터 타입 :  <class 'int'>
    max =  255
    min =  0



```python
# 입력 데이터를 이미지로 출력

plt.figure(figsize=(1,1))
plt.title(f'{label}')
plt.imshow(image, cmap='gray_r')
plt.axis('off')
plt.show()
```


    
*(이미지 없음 / image not preserved: output_21_0.png)*
    



```python
## plt.subplot

plt.figure(figsize=(4, 2))
plt.subplot(1,2,1), plt.imshow(image, cmap = 'gray_r')
plt.subplot(1,2,2), plt.imshow(image, cmap = 'gray_r')
plt.show()

```


    
*(이미지 없음 / image not preserved: output_22_0.png)*
    



```python
# 정답 데이터와 함께 처음 20개 데이터를 이미지로 출력

plt.figure(figsize=(10, 3))
for i in range(20):
    ax = plt.subplot(2, 10, i + 1)

    # image와 label 취득
    image, label = train_set0[i]

    # 이미지 출력
    plt.imshow(image, cmap='gray_r')
    ax.set_title(f'{label}')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
```


    
*(이미지 없음 / image not preserved: output_23_0.png)*
    


### Transforms를 활용한 데이터 전처리


```python
# 라이브러리 임포트
# import torchvision.transforms as transforms

transform1 = transforms.Compose([
    # 데이터를 텐서로 변환
    transforms.ToTensor(),
])

train_set1 = datasets.MNIST(
    root=data_root,
    train=True,
    download=True,
    transform = transform1)
```


```python
# 변환 결과 확인

image, label = train_set1[0]
print('입력 데이터 타입 : ', type(image)) # <class 'torch.Tensor'>
print('입력 데이터 shape : ', image.shape)
print('최솟값 : ', image.data.min())
print('최댓값 : ', image.data.max())
```

    입력 데이터 타입 :  <class 'torch.Tensor'>
    입력 데이터 shape :  torch.Size([1, 28, 28])
    최솟값 :  tensor(0.)
    최댓값 :  tensor(1.)


### Normalize 사용 하기


```python
## 순서 중요
transform2 = transforms.Compose([
    # 데이터를 텐서로 변환
    transforms.ToTensor(),

    # 데이터 정규화
    transforms.Normalize(mean = 0.5,  std = 0.5), # z-transform
])

train_set2 = datasets.MNIST(
    root = data_root,
    train = True,
    download = True,
    transform = transform2)
```


```python
# 변환 결과 확인

image, label = train_set2[0]
print('shape : ', image.shape)
print('최솟값 : ', image.data.min())
print('최댓값 : ', image.data.max())
```

    shape :  torch.Size([1, 28, 28])
    최솟값 :  tensor(-1.)
    최댓값 :  tensor(1.)


### 람다 표현식을 활용한 함수 정의


```python
def f(x):
    return 1/np.exp(-10*x)


lambda x: 1/np.exp(-10*x)
```




    <function __main__.<lambda>(x)>




```python
# 일반적인 함수의 정의

def f(x):
    return (2 * x**2 + 2)

x = np.arange(-2, 2.1, 0.25)
y = f(x)
print(y)


# 람다 표현식으로 함수 정의
print("="*50)
g = lambda x: 2 * x**2 + 2

y = g(x)
print(y)
```

    [10.     8.125  6.5    5.125  4.     3.125  2.5    2.125  2.     2.125
      2.5    3.125  4.     5.125  6.5    8.125 10.   ]
    ==================================================
    [10.     8.125  6.5    5.125  4.     3.125  2.5    2.125  2.     2.125
      2.5    3.125  4.     5.125  6.5    8.125 10.   ]


### Lambda 클래스를 사용해 1차원으로 텐서 변환하기


```python
transform = transforms.Compose([
    # 데이터를 텐서로 변환
    transforms.ToTensor(),

    # 데이터 정규화
    transforms.Normalize(0.5, 0.5),

    # 현재 텐서를 1계 텐서로 변환
    transforms.Lambda(lambda x: x.view(-1))

])

train_set = datasets.MNIST(
    root = data_root,
    train = True,
    download=True,
    transform = transform)
```


```python
transform3 = transforms.Compose([
    # 데이터를 텐서로 변환
    transforms.ToTensor(),

    # 데이터 정규화
    transforms.Normalize(0.5, 0.5),

    # 현재 텐서를 1계 텐서로 변환
    transforms.Lambda(lambda x: x.view(-1)),
])

train_set3 = datasets.MNIST(
    root = data_root,
    train = True,
    download=True,
    transform = transform3)
```


```python
# 변환 결과 확인

image, label = train_set3[0]
print('shape : ', image.shape)
print('최솟값 : ', image.data.min())
print('최댓값 : ', image.data.max())
```

    shape :  torch.Size([784])
    최솟값 :  tensor(-1.)
    최댓값 :  tensor(1.)


### 최종 구현 형태


```python
# 데이터 변환용 함수 Transforms
# (1) Image를 텐서화
# (2) [0, 1] 범위의 값을 [-1, 1] 범위로 조정
# (3) 데이터의 shape을 [1, 28, 28] 에서 [784] 로 변환

transform = transforms.Compose([
    # (1) 데이터를 텐서로 변환
    transforms.ToTensor(),

    # (2) 데이터 정규화
    transforms.Normalize(0.5, 0.5),

    # (3) 1계 텐서로 변환
    transforms.Lambda(lambda x: x.view(-1)),
])
```


```python
# 데이터 입수를 위한 Dataset 함수

# 훈련용 데이터셋 정의
train_set = datasets.MNIST(
    root = data_root,
    train = True,
    download = True,
    transform = transform)

# 검증용 데이터셋 정의
test_set = datasets.MNIST(
    root = data_root,
    train = False,
    download = True,
    transform = transform)
```

### 데이터로더를 활용한 미니 배치 데이터 생성


```python
# 라이브러리 임포트
from torch.utils.data import DataLoader

# 미니 배치 사이즈 지정
batch_size = 500

# 훈련용 데이터로더
# 훈련용이므로, 셔플을 적용함
train_loader = DataLoader(
    dataset = train_set,
    batch_size = batch_size,
    shuffle = True)

# 검증용 데이터로더
# 검증시에는 셔플을 필요로하지 않음
test_loader = DataLoader(
    dataset = test_set,
    batch_size = batch_size,
    shuffle = False)
```


```python
# 몇 개의 그룹으로 데이터를 가져올 수 있는가
# images, labels = next(iter(train_loader))

print(len(train_loader))

# 데이터로더로부터 가장 처음 한 세트를 가져옴
for images, labels in train_loader:
    break

print(images.shape)
print(labels.shape)

# print("max value = ", images.max())
```

    120
    torch.Size([500, 784])
    torch.Size([500])



```python
# 이미지 출력
plt.figure(figsize=(10, 3))
# fig, axs = plt.subplots(2, 10, figsize = (10, 3))

for i in range(20):
    ax = plt.subplot(2, 10, i + 1)

    # row = i//10
    # col = i % 10

    # 넘파이로 배열로 변환
    image = images[i].numpy()
    label = labels[i]

    # 이미지의 범위를 [0, 1] 로 되돌림
    image2 = (image + 1)/ 2

    # 이미지 출력
    plt.imshow(image2.reshape(28, 28), cmap='gray_r')
    # ax.set_title(str(label.item()))
    ax.set_title(f'{label}')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # axs[row, col].imshow(image2.reshape(28, 28), cmap='gray_r')
    # axs[row, col].set_title(f'{label}')
    # axs[row, col].get_xaxis().set_visible(False)
    # axs[row, col].get_yaxis().set_visible(False)
plt.show()
```


    
*(이미지 없음 / image not preserved: output_43_0.png)*
    


### 모델 정의


```python
# 입력 차원수
n_input = image.shape[0]

# 출력 차원수
# 분류 클래스 수는 10
n_output = len(set(list(labels.data.numpy())))

# 은닉층의 노드 수
n_hidden = 128

# 결과 확인
print(f'n_input: {n_input}  n_hidden: {n_hidden} n_output: {n_output}')
```

    n_input: 784  n_hidden: 128 n_output: 10



```python
# 모델 정의
# 784입력 10출력 1은닉층의 신경망 모델

class Net(nn.Module):
    def __init__(self, n_input, n_output, n_hidden):
        super().__init__()

        # 은닉층 정의(은닉층 노드 수 : n_hidden)
        self.l1 = nn.Linear(n_input, n_hidden)

        # 출력층 정의
        self.l2 = nn.Linear(n_hidden, n_output)

        # ReLU 함수 정의
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.l1(x)
        x2 = self.relu(x1)
        x3 = self.l2(x2)
        return x3
```


```python
# 난수 고정
torch.manual_seed(123)
torch.cuda.manual_seed(123)
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms = True

# 모델 인스턴스 생성
net = Net(n_input, n_output, n_hidden)

# 모델을 GPU로 전송
net = net.to(device)
# next(net.parameters()).is_cuda
```


```python
# 학습률
lr = 0.01

# 최적화 알고리즘: 경사 하강법
optimizer = optim.SGD(net.parameters(), lr=lr)

# 손실 함수： 교차 엔트로피 함수
criterion = nn.CrossEntropyLoss()
```


```python
# 모델 내부 파라미터 확인
# l1.weight, l1.bias, l2.weight, l2.bias를 확인할 수 있음

for parameter in net.named_parameters():
    print(parameter)

# list(net.named_parameters())[0][1].data.cpu().numpy()
```

    ('l1.weight', Parameter containing:
    tensor([[-0.0146,  0.0012, -0.0177,  ...,  0.0277,  0.0200,  0.0315],
            [ 0.0184, -0.0322,  0.0175,  ...,  0.0089, -0.0028, -0.0033],
            [ 0.0092,  0.0261,  0.0075,  ...,  0.0061,  0.0267, -0.0258],
            ...,
            [ 0.0235, -0.0026, -0.0129,  ...,  0.0322, -0.0059, -0.0169],
            [-0.0328, -0.0258,  0.0124,  ..., -0.0049,  0.0006,  0.0334],
            [ 0.0187, -0.0076, -0.0202,  ...,  0.0325, -0.0159, -0.0240]],
           requires_grad=True))
    ('l1.bias', Parameter containing:
    tensor([ 0.0325, -0.0298,  0.0013,  0.0199,  0.0268, -0.0248, -0.0172, -0.0355,
             0.0122, -0.0048,  0.0214,  0.0202, -0.0243,  0.0015, -0.0276,  0.0296,
             0.0341, -0.0228,  0.0230,  0.0347, -0.0091, -0.0346,  0.0206, -0.0060,
             0.0329,  0.0047,  0.0180,  0.0101,  0.0177, -0.0309,  0.0228, -0.0224,
             0.0321,  0.0179,  0.0321,  0.0184,  0.0219, -0.0089,  0.0310, -0.0039,
            -0.0074, -0.0317,  0.0192, -0.0021,  0.0190,  0.0038,  0.0334, -0.0027,
            -0.0127,  0.0229, -0.0265,  0.0023, -0.0162, -0.0134, -0.0027,  0.0212,
            -0.0205, -0.0144,  0.0121,  0.0001,  0.0086,  0.0033,  0.0123,  0.0213,
            -0.0177,  0.0247, -0.0109, -0.0222,  0.0228, -0.0110, -0.0074, -0.0089,
            -0.0205,  0.0323, -0.0207, -0.0205, -0.0028, -0.0341, -0.0304,  0.0144,
             0.0072,  0.0326, -0.0342, -0.0329, -0.0032, -0.0200, -0.0029, -0.0098,
             0.0220, -0.0160,  0.0099,  0.0033, -0.0289,  0.0110,  0.0199,  0.0131,
            -0.0279,  0.0122,  0.0237,  0.0126, -0.0055, -0.0088, -0.0057, -0.0048,
             0.0007, -0.0017, -0.0324,  0.0048, -0.0134,  0.0334,  0.0298, -0.0060,
             0.0263,  0.0113, -0.0113,  0.0150,  0.0091, -0.0311, -0.0079,  0.0002,
            -0.0282, -0.0016,  0.0304, -0.0237, -0.0157, -0.0255,  0.0006,  0.0100],
           requires_grad=True))
    ('l2.weight', Parameter containing:
    tensor([[ 0.0107,  0.0714,  0.0153,  ...,  0.0704,  0.0505, -0.0382],
            [-0.0066,  0.0348,  0.0143,  ..., -0.0039, -0.0141,  0.0130],
            [-0.0251, -0.0654,  0.0567,  ..., -0.0435,  0.0154,  0.0256],
            ...,
            [-0.0131,  0.0147, -0.0452,  ...,  0.0344, -0.0539,  0.0466],
            [ 0.0771, -0.0510,  0.0769,  ..., -0.0257, -0.0351,  0.0670],
            [ 0.0456,  0.0628, -0.0649,  ..., -0.0804,  0.0707,  0.0119]],
           requires_grad=True))
    ('l2.bias', Parameter containing:
    tensor([-0.0787, -0.0282, -0.0108,  0.0021, -0.0330, -0.0162, -0.0825,  0.0590,
             0.0566, -0.0631], requires_grad=True))



```python
# 모델 개요 표시 1

print(net)
```

    Net(
      (l1): Linear(in_features=784, out_features=128, bias=True)
      (l2): Linear(in_features=128, out_features=10, bias=True)
      (relu): ReLU(inplace=True)
    )



```python
# 모델 개요 표시 2

summary(net, (784,))
```




    ==========================================================================================
    Layer (type:depth-idx)                   Output Shape              Param #
    ==========================================================================================
    Net                                      [10]                      --
    ├─Linear: 1-1                            [128]                     100,480
    ├─ReLU: 1-2                              [128]                     --
    ├─Linear: 1-3                            [10]                      1,290
    ==========================================================================================
    Total params: 101,770
    Trainable params: 101,770
    Non-trainable params: 0
    Total mult-adds (Units.MEGABYTES): 12.87
    ==========================================================================================
    Input size (MB): 0.00
    Forward/backward pass size (MB): 0.00
    Params size (MB): 0.41
    Estimated Total Size (MB): 0.41
    ==========================================================================================



### 경사 하강법


```python
# 훈련 데이터셋의 가장 처음 항목을 취득
# 데이터로더에서 가장 처음 항목을 취득
for images, labels in train_loader:
    break
```


```python
# 데이터로더에서 취득한 데이터를 GPU로 보냄
inputs = images.to(device)
labels = labels.to(device)
```


```python
# 예측 계산
outputs = net(inputs)

# 결과 확인
print(outputs)

```

    tensor([[-0.3622, -0.1927, -0.0179,  ...,  0.1073,  0.1025, -0.0615],
            [-0.4072, -0.1814,  0.0716,  ...,  0.1866,  0.1975,  0.1161],
            [-0.3221, -0.0547, -0.2868,  ...,  0.1967, -0.0103,  0.1591],
            ...,
            [-0.2091, -0.1058,  0.2365,  ...,  0.1360,  0.0665,  0.0987],
            [-0.2756, -0.2012,  0.1703,  ...,  0.1223,  0.2388,  0.0233],
            [-0.3045, -0.2458,  0.1416,  ...,  0.1012,  0.0820, -0.1457]],
           grad_fn=<AddmmBackward0>)



```python
#  손실 계산
loss = criterion(outputs, labels)

# 손실값 가져오기
print(loss)

# 손실 계산 그래프 시각화
g = make_dot(loss, params=dict(net.named_parameters()))
display(g)
```

    tensor(2.3329, grad_fn=<NllLossBackward0>)



    
*(이미지 없음 / image not preserved: output_56_1.svg)*
    


### 경사 계산


```python
next(net.parameters()).is_cuda
```




    False




```python
# 경사 계산 실행
loss.backward()
```


```python
# 경사 계산 결과
w = net.to('cpu')
print(w.l1.weight.grad.numpy())
print(w.l1.bias.grad.numpy())
print(w.l2.weight.grad.numpy())
print(w.l2.bias.grad.numpy())
```

    [[-0.0007 -0.0007 -0.0007 ... -0.0007 -0.0007 -0.0007]
     [ 0.0077  0.0077  0.0077 ...  0.0077  0.0077  0.0077]
     [-0.0018 -0.0018 -0.0018 ... -0.0018 -0.0018 -0.0018]
     ...
     [-0.0008 -0.0008 -0.0008 ... -0.0008 -0.0008 -0.0008]
     [ 0.0011  0.0011  0.0011 ...  0.0011  0.0011  0.0011]
     [-0.0001 -0.0001 -0.0001 ... -0.0001 -0.0001 -0.0001]]
    [ 0.0007 -0.0077  0.0018  0.0008 -0.      0.      0.0014 -0.0008  0.0025
     -0.0016  0.0009 -0.002   0.0006  0.0025 -0.0026  0.0008  0.0061 -0.0011
     -0.0018  0.008   0.0063  0.0026 -0.0036  0.0056 -0.0006 -0.0038  0.0034
      0.     -0.0026 -0.0032 -0.0006  0.0034  0.0018  0.      0.0001  0.0002
      0.0047 -0.0012  0.0022  0.0018  0.0037 -0.0061  0.0011  0.0097 -0.0017
     -0.0012 -0.0004 -0.001  -0.0031 -0.0003 -0.0008  0.0004  0.0001 -0.0016
     -0.002  -0.0001 -0.0006 -0.0024 -0.0004  0.0029  0.0013 -0.0085  0.0013
      0.0015  0.     -0.0006  0.004  -0.0016 -0.0052  0.0003 -0.0031  0.0001
      0.0009 -0.0017 -0.0069 -0.0028  0.0017 -0.003   0.0012  0.0024  0.0011
     -0.002   0.0053 -0.0001  0.007   0.0024  0.003   0.0038 -0.0001 -0.0017
     -0.0006 -0.0021  0.0026  0.      0.0045  0.0037  0.0058 -0.0032 -0.
     -0.0003 -0.0006  0.      0.0029  0.0017  0.0022 -0.0034 -0.0001  0.0006
     -0.0015 -0.0035  0.0017 -0.0021 -0.0022  0.0013 -0.0002  0.0035 -0.0027
      0.0006 -0.002   0.002  -0.0036  0.0004  0.0006  0.0006 -0.0011  0.0008
     -0.0011  0.0001]
    [[-0.0198 -0.0018 -0.02   ... -0.0068 -0.0056 -0.0021]
     [ 0.0061 -0.0106  0.0044 ... -0.0123  0.0017  0.0009]
     [-0.0059  0.0061  0.0035 ...  0.002   0.0012 -0.0008]
     ...
     [ 0.0067 -0.0137  0.0041 ...  0.0053 -0.0006  0.0019]
     [-0.0066 -0.0007  0.0034 ...  0.0073 -0.0021 -0.0036]
     [ 0.0088  0.0024 -0.0002 ... -0.0002  0.0019  0.0012]]
    [-0.053  -0.033   0.0125 -0.005   0.0229  0.0163  0.0168  0.0102  0.0214
     -0.0091]


### 파라미터 수정


```python
# 경사 하강법 적용
optimizer.step()
```


```python
# 파라미터 값 출력
print(net.l1.weight)
print(net.l1.bias)
```

    Parameter containing:
    tensor([[-0.0146,  0.0012, -0.0177,  ...,  0.0278,  0.0200,  0.0316],
            [ 0.0183, -0.0322,  0.0174,  ...,  0.0088, -0.0029, -0.0034],
            [ 0.0092,  0.0261,  0.0075,  ...,  0.0061,  0.0267, -0.0258],
            ...,
            [ 0.0235, -0.0026, -0.0129,  ...,  0.0323, -0.0059, -0.0169],
            [-0.0329, -0.0258,  0.0124,  ..., -0.0049,  0.0006,  0.0334],
            [ 0.0187, -0.0076, -0.0202,  ...,  0.0325, -0.0159, -0.0240]],
           requires_grad=True)
    Parameter containing:
    tensor([ 3.2475e-02, -2.9682e-02,  1.2742e-03,  1.9874e-02,  2.6836e-02,
            -2.4759e-02, -1.7201e-02, -3.5517e-02,  1.2199e-02, -4.7449e-03,
             2.1379e-02,  2.0187e-02, -2.4297e-02,  1.4928e-03, -2.7613e-02,
             2.9618e-02,  3.4051e-02, -2.2777e-02,  2.2983e-02,  3.4580e-02,
            -9.1870e-03, -3.4619e-02,  2.0599e-02, -6.0632e-03,  3.2937e-02,
             4.7784e-03,  1.7949e-02,  1.0102e-02,  1.7700e-02, -3.0853e-02,
             2.2817e-02, -2.2391e-02,  3.2049e-02,  1.7890e-02,  3.2113e-02,
             1.8418e-02,  2.1852e-02, -8.8597e-03,  3.0939e-02, -3.9572e-03,
            -7.4435e-03, -3.1608e-02,  1.9150e-02, -2.2176e-03,  1.9040e-02,
             3.7815e-03,  3.3376e-02, -2.7366e-03, -1.2678e-02,  2.2926e-02,
            -2.6499e-02,  2.2708e-03, -1.6189e-02, -1.3415e-02, -2.7006e-03,
             2.1242e-02, -2.0511e-02, -1.4376e-02,  1.2089e-02,  9.8037e-05,
             8.5776e-03,  3.3507e-03,  1.2323e-02,  2.1314e-02, -1.7690e-02,
             2.4736e-02, -1.0986e-02, -2.2139e-02,  2.2898e-02, -1.1038e-02,
            -7.4188e-03, -8.9315e-03, -2.0528e-02,  3.2279e-02, -2.0665e-02,
            -2.0434e-02, -2.7932e-03, -3.4027e-02, -3.0392e-02,  1.4364e-02,
             7.1700e-03,  3.2612e-02, -3.4299e-02, -3.2920e-02, -3.2781e-03,
            -2.0019e-02, -2.9709e-03, -9.8261e-03,  2.1964e-02, -1.5987e-02,
             9.8720e-03,  3.2919e-03, -2.8945e-02,  1.0965e-02,  1.9866e-02,
             1.3074e-02, -2.7974e-02,  1.2213e-02,  2.3668e-02,  1.2602e-02,
            -5.4937e-03, -8.7514e-03, -5.7194e-03, -4.8619e-03,  6.6892e-04,
            -1.7088e-03, -3.2382e-02,  4.8306e-03, -1.3428e-02,  3.3444e-02,
             2.9813e-02, -5.9374e-03,  2.6309e-02,  1.1309e-02, -1.1252e-02,
             1.4970e-02,  9.1236e-03, -3.1057e-02, -7.8487e-03,  1.3641e-04,
            -2.8135e-02, -1.6511e-03,  3.0365e-02, -2.3754e-02, -1.5655e-02,
            -2.5556e-02,  6.5686e-04,  9.9645e-03], requires_grad=True)


### 반복 계산


```python
# 난수 고정
torch.manual_seed(123)
torch.cuda.manual_seed(123)


# 학습률
lr = 0.01

# 모델 초기화
net = Net(n_input, n_output, n_hidden).to(device)

# 손실 함수： 교차 엔트로피 함수
criterion = nn.CrossEntropyLoss()

# 최적화 함수: 경사 하강법
optimizer = optim.SGD(net.parameters(), lr=lr)

# 반복 횟수
# num_epochs = 100
num_epochs = 10


# 평가 결과 기록
history = np.zeros((0,5))
```


```python
# tqdm 라이브러리 임포트
from tqdm.notebook import tqdm

# 반복 계산 메인 루프
for epoch in range(num_epochs):
    train_acc, train_loss = 0, 0
    val_acc, val_loss = 0, 0
    n_train, n_test = 0, 0

    # 훈련 페이즈
    for inputs, labels in tqdm(train_loader):
        n_train += len(labels)

        # GPU로 전송
        inputs = inputs.to(device)
        labels = labels.to(device)

        # 경사 초기화
        optimizer.zero_grad()

        # 예측 계산
        outputs = net(inputs)

        # 손실 계산
        loss = criterion(outputs, labels)

        # 경사 계산
        loss.backward()

        # 파라미터 수정
        optimizer.step()

        # 예측 라벨 산출
        predicted = torch.max(outputs, 1)[1]

        # 손실과 정확도 계산
        train_loss += loss.item()
        train_acc += (predicted == labels).sum().item()

    # 예측 페이즈
    for inputs_test, labels_test in test_loader:
        n_test += len(labels_test)

        inputs_test = inputs_test.to(device)
        labels_test = labels_test.to(device)


        # 예측 계산
        outputs_test = net(inputs_test)

        # 손실 계산
        loss_test = criterion(outputs_test, labels_test)

        # 예측 라벨 산출
        predicted_test = torch.max(outputs_test, 1)[1]

        # 손실과 정확도 계산
        val_loss +=  loss_test.item()
        val_acc +=  (predicted_test == labels_test).sum().item()

    # 평가 결과 산출, 기록
    train_acc = train_acc / n_train
    val_acc = val_acc / n_test
    train_loss = train_loss * batch_size / n_train
    val_loss = val_loss * batch_size / n_test
    print (f'Epoch [{epoch+1}/{num_epochs}], loss: {train_loss:.5f} acc: {train_acc:.5f} val_loss: {val_loss:.5f}, val_acc: {val_acc:.5f}')
    item = np.array([epoch+1 , train_loss, train_acc, val_loss, val_acc])
    history = np.vstack((history, item))
```


      0%|          | 0/120 [00:00<?, ?it/s]


    Epoch [1/10], loss: 1.82932 acc: 0.56960 val_loss: 1.32629, val_acc: 0.74660



      0%|          | 0/120 [00:00<?, ?it/s]


    Epoch [2/10], loss: 1.03889 acc: 0.79537 val_loss: 0.79661, val_acc: 0.83180



      0%|          | 0/120 [00:00<?, ?it/s]


    Epoch [3/10], loss: 0.70809 acc: 0.84110 val_loss: 0.60256, val_acc: 0.85850



      0%|          | 0/120 [00:00<?, ?it/s]


    Epoch [4/10], loss: 0.57300 acc: 0.86057 val_loss: 0.51192, val_acc: 0.87140



      0%|          | 0/120 [00:00<?, ?it/s]


    Epoch [5/10], loss: 0.50223 acc: 0.87102 val_loss: 0.45827, val_acc: 0.87930



      0%|          | 0/120 [00:00<?, ?it/s]


    Epoch [6/10], loss: 0.45883 acc: 0.87877 val_loss: 0.42422, val_acc: 0.88650



      0%|          | 0/120 [00:00<?, ?it/s]


    Epoch [7/10], loss: 0.42938 acc: 0.88327 val_loss: 0.40076, val_acc: 0.88970



      0%|          | 0/120 [00:00<?, ?it/s]


    Epoch [8/10], loss: 0.40813 acc: 0.88743 val_loss: 0.38285, val_acc: 0.89370



      0%|          | 0/120 [00:00<?, ?it/s]


    Epoch [9/10], loss: 0.39176 acc: 0.89065 val_loss: 0.36857, val_acc: 0.89680



      0%|          | 0/120 [00:00<?, ?it/s]


    Epoch [10/10], loss: 0.37875 acc: 0.89313 val_loss: 0.35741, val_acc: 0.89930


### 결과 확인


```python
# 손실과 정확도 확인

print(f'초기상태 : 손실 : {history[0,3]:.5f}  정확도 : {history[0,4]:.5f}' )
print(f'최종상태 : 손실 : {history[-1,3]:.5f}  정확도 : {history[-1,4]:.5f}' )
```

    초기상태 : 손실 : 1.32629  정확도 : 0.74660
    최종상태 : 손실 : 0.35741  정확도 : 0.89930



```python
# 학습 곡선 출력(손실)

plt.plot(history[:,0], history[:,1], 'b', label='훈련')
plt.plot(history[:,0], history[:,3], 'k', label='검증')
plt.xlabel('반복 횟수')
plt.ylabel('손실')
plt.title('학습 곡선(손실)')
plt.legend()
plt.show()
```


    
*(이미지 없음 / image not preserved: output_69_0.png)*
    



```python
# 학습 곡선 출력(정확도)

plt.plot(history[:,0], history[:,2], 'b', label='훈련')
plt.plot(history[:,0], history[:,4], 'k', label='검증')
plt.xlabel('반복 횟수')
plt.ylabel('정확도')
plt.title('학습 곡선(정확도)')
plt.legend()
plt.show()
```


    
*(이미지 없음 / image not preserved: output_70_0.png)*
    


### 이미지 출력 확인


```python
# 데이터로더에서 처음 한 세트 가져오기
for images, labels in test_loader:
    break

# 예측 결과 가져오기
inputs = images.to(device)
labels = labels.to(device)
outputs = net(inputs)
predicted = torch.max(outputs, 1)[1]
```


```python
# 처음 50건의 이미지에 대해 "정답:예측"으로 출력

plt.figure(figsize=(10, 8))
for i in range(50):
  ax = plt.subplot(5, 10, i + 1)

  # 넘파이 배열로 변환
  image = images[i]
  label = labels[i]
  pred = predicted[i]
  if (pred == label):
    c = 'k'
  else:
    c = 'b'

  # 이미지의 범위를 [0, 1] 로 되돌림
  image2 = (image + 1)/ 2

  # 이미지 출력
  plt.imshow(image2.reshape(28, 28),cmap='gray_r')
  ax.set_title(f'{label}:{pred}', c=c)
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
plt.show()
```


    
*(이미지 없음 / image not preserved: output_73_0.png)*
    


### 은닉층 추가하기


```python
# 모델 정의
# 784입력 10출력을 갖는 2개의 은닉층을 포함한 신경망

class Net2(nn.Module):
    def __init__(self, n_input, n_output, n_hidden):
        super().__init__()

        # 첫번째 은닉층 정의(은닉층 노드 수: n_hidden)
        self.l1 = nn.Linear(n_input, n_hidden)

        # 두번째 은닉층 정의(은닉층 노드 수: n_hidden)
        self.l2 = nn.Linear(n_hidden, n_hidden)

        # 출력층 정의
        self.l3 = nn.Linear(n_hidden, n_output)

        # ReLU 함수 정의
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.l1(x)
        x2 = self.relu(x1)
        x3 = self.l2(x2)
        x4 = self.relu(x3)
        x5 = self.l3(x4)
        return x5
```


```python
# 난수 고정
torch.manual_seed(123)
torch.cuda.manual_seed(123)

# 모델 초기화
net = Net2(n_input, n_output, n_hidden).to(device)

# 손실 함수： 교차 엔트로피 함수
criterion = nn.CrossEntropyLoss()

# 최적화 함수: 경사 하강법
optimizer = torch.optim.SGD(net.parameters(), lr=lr)
```


```python
print(net)
```

    Net2(
      (l1): Linear(in_features=784, out_features=128, bias=True)
      (l2): Linear(in_features=128, out_features=128, bias=True)
      (l3): Linear(in_features=128, out_features=10, bias=True)
      (relu): ReLU(inplace=True)
    )



```python
# 모델 개요 표시 2

summary(net, (784,))
```




    ==========================================================================================
    Layer (type:depth-idx)                   Output Shape              Param #
    ==========================================================================================
    Net2                                     [10]                      --
    ├─Linear: 1-1                            [128]                     100,480
    ├─ReLU: 1-2                              [128]                     --
    ├─Linear: 1-3                            [128]                     16,512
    ├─ReLU: 1-4                              [128]                     --
    ├─Linear: 1-5                            [10]                      1,290
    ==========================================================================================
    Total params: 118,282
    Trainable params: 118,282
    Non-trainable params: 0
    Total mult-adds (Units.MEGABYTES): 14.99
    ==========================================================================================
    Input size (MB): 0.00
    Forward/backward pass size (MB): 0.00
    Params size (MB): 0.47
    Estimated Total Size (MB): 0.48
    ==========================================================================================




```python
# 데이터로더에서 처음 한 세트 가져오기
for images, labels in test_loader:
    break

# 예측 결과 가져오기
inputs = images.to(device)
labels = labels.to(device)
```


```python
# 예측 계산
outputs = net(inputs)

# 손실 계산
loss = criterion(outputs, labels)

# 손실 계산 그래프 시각화
make_dot(loss, params=dict(net.named_parameters()))
```




    
*(이미지 없음 / image not preserved: output_80_0.svg)*
    



### 경사 계산


```python
# 경사 계산
loss.backward()

# 경사 계산 결과 일부
w = net.to('cpu').l1.weight.grad.numpy()
print("w = ", w)

# 각 요소의 절댓값 평균
print(np.abs(w).mean())
```

    w =  [[-0.0007 -0.0007 -0.0007 ... -0.0007 -0.0007 -0.0007]
     [-0.0001 -0.0001 -0.0001 ... -0.0001 -0.0001 -0.0001]
     [-0.0005 -0.0005 -0.0005 ... -0.0005 -0.0005 -0.0005]
     ...
     [ 0.0015  0.0015  0.0015 ...  0.0015  0.0015  0.0015]
     [ 0.0002  0.0002  0.0002 ...  0.0002  0.0002  0.0002]
     [ 0.0003  0.0003  0.0003 ...  0.0003  0.0003  0.0003]]
    0.0008487979


### 반복 계산


```python
# 난수 고정
torch.manual_seed(123)
torch.cuda.manual_seed(123)


# 모델 초기화
net = Net2(n_input, n_output, n_hidden).to(device)

# 손실 함수： 교차 엔트로피 함수
criterion = nn.CrossEntropyLoss()

# 최적화 함수: 경사 하강법
optimizer = optim.SGD(net.parameters(), lr=lr)

# 반복 횟수
num_epochs = 10

# 평가 결과 기록
history2 = np.zeros((0,5))
```


```python
# 반복 계산 메인 루프

for epoch in range(num_epochs):
    train_acc = 0
    train_loss = 0
    val_acc = 0
    val_loss = 0
    n_train = 0
    n_test = 0

    # 훈련 페이즈
    for inputs, labels in tqdm(train_loader):
        n_train += len(labels)

        # GPU로 전송
        inputs = inputs.to(device)
        labels = labels.to(device)

        # 경사 초기화
        optimizer.zero_grad()

        # 예측 계산
        outputs = net(inputs)

        # 손실 계산
        loss = criterion(outputs, labels)

        # 경사 계산
        loss.backward()

        # 파라미터 수정
        optimizer.step()

        # 예측 라벨 산출
        predicted = torch.max(outputs, 1)[1]

        # 손실과 정확도 계산
        train_loss += loss.item()
        train_acc += (predicted == labels).sum().item()

    # 예측 페이즈
    for inputs_test, labels_test in test_loader:
        n_test += len(labels_test)

        inputs_test = inputs_test.to(device)
        labels_test = labels_test.to(device)

        # 예측 계산
        outputs_test = net(inputs_test)

        # 손실 계산
        loss_test = criterion(outputs_test, labels_test)

        # 예측 라벨 산출
        predicted_test = torch.max(outputs_test, 1)[1]

        # 손실과 정확도 계산
        val_loss +=  loss_test.item()
        val_acc +=  (predicted_test == labels_test).sum().item()

    # 평가 결과 산출, 기록
    train_acc = train_acc / n_train
    val_acc = val_acc / n_test
    train_loss = train_loss * batch_size / n_train
    val_loss = val_loss * batch_size / n_test
    print (f'Epoch [{epoch+1}/{num_epochs}], loss: {train_loss:.5f} acc: {train_acc:.5f} val_loss: {val_loss:.5f}, val_acc: {val_acc:.5f}')
    item = np.array([epoch+1 , train_loss, train_acc, val_loss, val_acc])
    history2 = np.vstack((history2, item))
```


      0%|          | 0/120 [00:00<?, ?it/s]


    Epoch [1/10], loss: 2.20163 acc: 0.25380 val_loss: 2.04576, val_acc: 0.49800



      0%|          | 0/120 [00:00<?, ?it/s]


    Epoch [2/10], loss: 1.75820 acc: 0.60442 val_loss: 1.39272, val_acc: 0.68680



      0%|          | 0/120 [00:00<?, ?it/s]


    Epoch [3/10], loss: 1.11285 acc: 0.75652 val_loss: 0.86511, val_acc: 0.80820



      0%|          | 0/120 [00:00<?, ?it/s]


    Epoch [4/10], loss: 0.75171 acc: 0.82292 val_loss: 0.63478, val_acc: 0.84390



      0%|          | 0/120 [00:00<?, ?it/s]


    Epoch [5/10], loss: 0.59030 acc: 0.84978 val_loss: 0.52463, val_acc: 0.86360



      0%|          | 0/120 [00:00<?, ?it/s]


    Epoch [6/10], loss: 0.50672 acc: 0.86653 val_loss: 0.46401, val_acc: 0.87430



      0%|          | 0/120 [00:00<?, ?it/s]


    Epoch [7/10], loss: 0.45680 acc: 0.87543 val_loss: 0.42149, val_acc: 0.88570



      0%|          | 0/120 [00:00<?, ?it/s]


    Epoch [8/10], loss: 0.42336 acc: 0.88188 val_loss: 0.39552, val_acc: 0.89020



      0%|          | 0/120 [00:00<?, ?it/s]


    Epoch [9/10], loss: 0.39965 acc: 0.88770 val_loss: 0.37600, val_acc: 0.89360



      0%|          | 0/120 [00:00<?, ?it/s]


    Epoch [10/10], loss: 0.38195 acc: 0.89187 val_loss: 0.35886, val_acc: 0.89880



```python
# 손실과 정확도 확인

print(f'초기상태 : 손실 : {history2[0,3]:.5f}  정확도 : {history2[0,4]:.5f}' )
print(f'최종상태 : 손실 : {history2[-1,3]:.5f}  정확도 : {history2[-1,4]:.5f}' )
```

    초기상태 : 손실 : 2.04576  정확도 : 0.49800
    최종상태 : 손실 : 0.35886  정확도 : 0.89880



```python
# 학습 곡선 출력(손실)
plt.plot(history2[:,0], history2[:,1], 'b', label='훈련')
plt.plot(history2[:,0], history2[:,3], 'k', label='검증')
plt.xlabel('반복 횟수')
plt.ylabel('손실')
plt.title('학습 곡선(손실)')
plt.legend()
plt.show()
```


    
*(이미지 없음 / image not preserved: output_87_0.png)*
    



```python
# 학습 곡선 출력(정확도)

plt.plot(history2[:,0], history2[:,2], 'b', label='훈련')
plt.plot(history2[:,0], history2[:,4], 'k', label='검증')
plt.xlabel('반복 횟수')
plt.ylabel('정확도')
plt.title('학습 곡선(정확도)')
plt.legend()
plt.show()
```


    
*(이미지 없음 / image not preserved: output_88_0.png)*
    


### 경사 소실과 ReLU 함수


```python
# 모델 정의 -　시그모이드 함수 버전
# 784입력 10출력을 갖는 2개의 은닉층을 포함한 신경망

class Net3(nn.Module):
    def __init__(self, n_input, n_output, n_hidden):
        super().__init__()

        # 첫번째 은닉층 정의(은닉층 노드 수: n_hidden)
        self.l1 = nn.Linear(n_input, n_hidden)

        # 두번째 은닉층 정의(은닉층 노드 수: n_hidden)
        self.l2 = nn.Linear(n_hidden, n_hidden)

        # 출력층 정의
        self.l3 = nn.Linear(n_hidden, n_output)

        # 시그모이드 함수 정의
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.l1(x)
        x2 = self.sigmoid(x1)
        x3 = self.l2(x2)
        x4 = self.sigmoid(x3)
        x5 = self.l3(x4)
        return x5
```


```python
# 난수 고정
torch.manual_seed(123)
torch.cuda.manual_seed(123)

# 모델 초기화
net = Net3(n_input, n_output, n_hidden).to(device)

# 손실 함수： 교차 엔트로피 함수
criterion = nn.CrossEntropyLoss()

# 최적화 함수: 경사 하강법
optimizer = torch.optim.SGD(net.parameters(), lr=lr)
```


```python
# 데이터로더에서 처음 한 세트 가져오기
for images, labels in test_loader:
    break

# 예측 결과 가져오기
inputs = images.to(device)
labels = labels.to(device)
```


```python
# 예측 계산
outputs = net(inputs)

# 손실 계산
loss = criterion(outputs, labels)

# 손실 계산 그래프 시각화
make_dot(loss, params=dict(net.named_parameters()))
```




    
*(이미지 없음 / image not preserved: output_93_0.svg)*
    




```python
# 경사 계산
loss.backward()

# 경사 계산 결과의 일부
w = net.to('cpu').l1.weight.grad.numpy()
print(w)

# 각 요소의 절댓값 평균
print(np.abs(w).mean())
```

    [[ 0.0001  0.0001  0.0001 ...  0.0001  0.0001  0.0001]
     [ 0.0001  0.0001  0.0001 ...  0.0001  0.0001  0.0001]
     [-0.0001 -0.0001 -0.0001 ... -0.0001 -0.0001 -0.0001]
     ...
     [-0.0001 -0.0001 -0.0001 ... -0.0001 -0.0001 -0.0001]
     [ 0.0002  0.0002  0.0002 ...  0.0002  0.0002  0.0002]
     [-0.0001 -0.0001 -0.0001 ... -0.0001 -0.0001 -0.0001]]
    0.00017514593


### 배치 사이즈와 정확도의 관계


```python
# 학습용 함수
def fit(net, optimizer, criterion, num_epochs, train_loader, test_loader, device, history):
    base_epochs = len(history)
    batch_size_train = len(train_loader)
    batch_size_test = len(test_loader)


    for epoch in range(base_epochs, num_epochs+base_epochs):
        train_loss = 0
        train_acc = 0
        val_loss = 0
        val_acc = 0

        # 훈련 페이즈
        # count = 0

        for inputs, labels in tqdm(train_loader):

            # count += len(labels)
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 경사 초기화
            optimizer.zero_grad()

            # 예측 계산
            outputs = net(inputs)

            # 손실 계산
            loss = criterion(outputs, labels)
            train_loss += loss.item()

            # 경사 계산
            loss.backward()

            # 파라미터 수정
            optimizer.step()

            # 예측 라벨 산출
            predicted = torch.max(outputs, 1)[1]

            # 정답 건수 산출
            train_acc += (predicted == labels).sum().item() /len(labels)

            # 훈련 데이터에 대해 손실과 정확도 계산
        avg_train_loss = train_loss / batch_size_train
        avg_train_acc = train_acc / batch_size_train

        # 예측 페이즈
        # count = 0

        for inputs, labels in test_loader:
            # count += len(labels)

            inputs = inputs.to(device)
            labels = labels.to(device)

            # 예측 계산
            outputs = net(inputs)

            # 손실 계산
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # 예측 라벨 산출
            predicted = torch.max(outputs, 1)[1]

            # 정답 건수 산출
            val_acc += (predicted == labels).sum().item() /len(labels)

            # 검증 데이터에 대해 손실과 정확도 계산
        avg_val_loss = val_loss / batch_size_test
        avg_val_acc = val_acc / batch_size_test

        print (f'Epoch [{(epoch+1)}/{num_epochs+base_epochs}], loss: {avg_train_loss:.5f} acc: {avg_train_acc:.5f} val_loss: {avg_val_loss:.5f}, val_acc: {avg_val_acc:.5f}')
        item = np.array([epoch+1, avg_train_loss, avg_train_acc, avg_val_loss, avg_val_acc])
        history = np.vstack((history, item))
    return history
```


```python
# 파이토치 난수 고정

def torch_seed(seed=123):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)
```

### Batch size  500


```python
# 미니 배치 사이즈 지정
batch_size_train = 500

# 훈련용 데이터로더
# 훈련용이므로 셔플을 적용함
train_loader = DataLoader(
    train_set,
    batch_size = batch_size_train,
    shuffle = True)

# 난수 고정
torch_seed()

# 학습률
lr = 0.01

# 모델 초기화
net = Net(n_input, n_output, n_hidden).to(device)

# 최적화 함수: 경사 하강법
optimizer = optim.SGD(net.parameters(), lr=lr)

# 손실 함수： 교차 엔트로피 함수
criterion = nn.CrossEntropyLoss()

# 반복 횟수
num_epochs = 10

# 평가 결과 기록
history = np.zeros((0,5))
```


```python
history = fit(net, optimizer, criterion, num_epochs, train_loader, test_loader, device, history)
```


      0%|          | 0/120 [00:00<?, ?it/s]


    Epoch [1/10], loss: 1.82932 acc: 0.56960 val_loss: 1.32629, val_acc: 0.74660



      0%|          | 0/120 [00:00<?, ?it/s]


    Epoch [2/10], loss: 1.03889 acc: 0.79537 val_loss: 0.79661, val_acc: 0.83180



      0%|          | 0/120 [00:00<?, ?it/s]


    Epoch [3/10], loss: 0.70809 acc: 0.84110 val_loss: 0.60256, val_acc: 0.85850



      0%|          | 0/120 [00:00<?, ?it/s]


    Epoch [4/10], loss: 0.57300 acc: 0.86057 val_loss: 0.51192, val_acc: 0.87140



      0%|          | 0/120 [00:00<?, ?it/s]


    Epoch [5/10], loss: 0.50223 acc: 0.87102 val_loss: 0.45827, val_acc: 0.87930



      0%|          | 0/120 [00:00<?, ?it/s]


    Epoch [6/10], loss: 0.45883 acc: 0.87877 val_loss: 0.42422, val_acc: 0.88650



      0%|          | 0/120 [00:00<?, ?it/s]


    Epoch [7/10], loss: 0.42938 acc: 0.88327 val_loss: 0.40076, val_acc: 0.88970



      0%|          | 0/120 [00:00<?, ?it/s]


    Epoch [8/10], loss: 0.40813 acc: 0.88743 val_loss: 0.38285, val_acc: 0.89370



      0%|          | 0/120 [00:00<?, ?it/s]


    Epoch [9/10], loss: 0.39176 acc: 0.89065 val_loss: 0.36857, val_acc: 0.89680



      0%|          | 0/120 [00:00<?, ?it/s]


    Epoch [10/10], loss: 0.37875 acc: 0.89313 val_loss: 0.35741, val_acc: 0.89930


### batch_size=200


```python
# 미니 배치 사이즈 지정
batch_size_train = 200

# 훈련용 데이터로더
# 훈련용이므로 셔플을 적용함
train_loader = DataLoader(
    train_set, batch_size = batch_size_train,
    shuffle = True)

# 난수 고정
torch_seed()

# 학습률
lr = 0.01

# 모델 초기화
net = Net(n_input, n_output, n_hidden).to(device)

# 최적화 함수: 경사 하강법
optimizer = optim.SGD(net.parameters(), lr=lr)

# 손실 함수： 교차 엔트로피 함수
criterion = nn.CrossEntropyLoss()

# 반복 횟수
num_epochs = 10

# 평가 결과 기록
history3 = np.zeros((0,5))
```


```python
history3 = fit(net, optimizer, criterion, num_epochs, train_loader, test_loader, device, history3)
```


      0%|          | 0/300 [00:00<?, ?it/s]


    Epoch [1/10], loss: 1.30017 acc: 0.71105 val_loss: 0.68051, val_acc: 0.84730



      0%|          | 0/300 [00:00<?, ?it/s]


    Epoch [2/10], loss: 0.56331 acc: 0.86208 val_loss: 0.45811, val_acc: 0.87970



      0%|          | 0/300 [00:00<?, ?it/s]


    Epoch [3/10], loss: 0.43948 acc: 0.88183 val_loss: 0.39073, val_acc: 0.89210



      0%|          | 0/300 [00:00<?, ?it/s]


    Epoch [4/10], loss: 0.39051 acc: 0.89073 val_loss: 0.36061, val_acc: 0.89740



      0%|          | 0/300 [00:00<?, ?it/s]


    Epoch [5/10], loss: 0.36304 acc: 0.89678 val_loss: 0.33796, val_acc: 0.90370



      0%|          | 0/300 [00:00<?, ?it/s]


    Epoch [6/10], loss: 0.34487 acc: 0.90107 val_loss: 0.32460, val_acc: 0.90620



      0%|          | 0/300 [00:00<?, ?it/s]


    Epoch [7/10], loss: 0.33107 acc: 0.90442 val_loss: 0.31342, val_acc: 0.91130



      0%|          | 0/300 [00:00<?, ?it/s]


    Epoch [8/10], loss: 0.32004 acc: 0.90785 val_loss: 0.30469, val_acc: 0.91430



      0%|          | 0/300 [00:00<?, ?it/s]


    Epoch [9/10], loss: 0.31084 acc: 0.91033 val_loss: 0.29686, val_acc: 0.91330



      0%|          | 0/300 [00:00<?, ?it/s]


    Epoch [10/10], loss: 0.30229 acc: 0.91278 val_loss: 0.28928, val_acc: 0.91760


### batch_size=100


```python
# 미니 배치 사이즈 지정
batch_size_train = 100

# 훈련용 데이터로더
# 훈련용이므로 셔플을 적용함
train_loader = DataLoader(
    train_set, batch_size = batch_size_train,
    shuffle = True)

# 난수 고정
torch_seed()

# 학습률
lr = 0.01

# 모델 초기화
net = Net(n_input, n_output, n_hidden).to(device)

# 최적화 함수: 경사 하강법
optimizer = optim.SGD(net.parameters(), lr=lr)

# 손실 함수： 교차 엔트로피 함수
criterion = nn.CrossEntropyLoss()

# 반복 횟수
num_epochs = 10

# 평가 결과 기록
history4 = np.zeros((0,5))
```


```python
history4 = fit(net, optimizer, criterion, num_epochs, train_loader, test_loader, device, history4)
```


      0%|          | 0/600 [00:00<?, ?it/s]


    Epoch [1/10], loss: 0.93449 acc: 0.78320 val_loss: 0.46005, val_acc: 0.87920



      0%|          | 0/600 [00:00<?, ?it/s]


    Epoch [2/10], loss: 0.41716 acc: 0.88513 val_loss: 0.35982, val_acc: 0.89870



      0%|          | 0/600 [00:00<?, ?it/s]


    Epoch [3/10], loss: 0.35608 acc: 0.89830 val_loss: 0.32409, val_acc: 0.90800



      0%|          | 0/600 [00:00<?, ?it/s]


    Epoch [4/10], loss: 0.32769 acc: 0.90545 val_loss: 0.30662, val_acc: 0.91020



      0%|          | 0/600 [00:00<?, ?it/s]


    Epoch [5/10], loss: 0.30828 acc: 0.91092 val_loss: 0.29081, val_acc: 0.91770



      0%|          | 0/600 [00:00<?, ?it/s]


    Epoch [6/10], loss: 0.29329 acc: 0.91597 val_loss: 0.28169, val_acc: 0.91810



      0%|          | 0/600 [00:00<?, ?it/s]


    Epoch [7/10], loss: 0.28022 acc: 0.91935 val_loss: 0.26869, val_acc: 0.92430



      0%|          | 0/600 [00:00<?, ?it/s]


    Epoch [8/10], loss: 0.26842 acc: 0.92298 val_loss: 0.25890, val_acc: 0.92690



      0%|          | 0/600 [00:00<?, ?it/s]


    Epoch [9/10], loss: 0.25724 acc: 0.92583 val_loss: 0.25020, val_acc: 0.92730



      0%|          | 0/600 [00:00<?, ?it/s]


    Epoch [10/10], loss: 0.24618 acc: 0.92952 val_loss: 0.23854, val_acc: 0.93200


### batch_size=50


```python
# 미니 배치 사이즈 지정
batch_size_train = 50

# 훈련용 데이터로더
# 훈련용이므로 셔플을 적용함
train_loader = DataLoader(
    train_set, batch_size = batch_size_train,
    shuffle = True)

# 난수 고정
torch_seed()

# 학습률
lr = 0.01

# 모델 초기화
net = Net(n_input, n_output, n_hidden).to(device)

# 최적화 함수: 경사 하강법
optimizer = optim.SGD(net.parameters(), lr=lr)

# 손실 함수： 교차 엔트로피 함수
criterion = nn.CrossEntropyLoss()

# 반복 횟수
num_epochs = 10

# 평가 결과 기록
history5 = np.zeros((0,5))
```


```python
history5 = fit(net, optimizer, criterion, num_epochs, train_loader, test_loader, device, history5)
```


      0%|          | 0/1200 [00:00<?, ?it/s]


    Epoch [1/10], loss: 0.68133 acc: 0.82922 val_loss: 0.36122, val_acc: 0.89640



      0%|          | 0/1200 [00:00<?, ?it/s]


    Epoch [2/10], loss: 0.34658 acc: 0.89972 val_loss: 0.31089, val_acc: 0.91170



      0%|          | 0/1200 [00:00<?, ?it/s]


    Epoch [3/10], loss: 0.30507 acc: 0.91165 val_loss: 0.28181, val_acc: 0.91760



      0%|          | 0/1200 [00:00<?, ?it/s]


    Epoch [4/10], loss: 0.27770 acc: 0.91998 val_loss: 0.26108, val_acc: 0.92370



      0%|          | 0/1200 [00:00<?, ?it/s]


    Epoch [5/10], loss: 0.25497 acc: 0.92593 val_loss: 0.24184, val_acc: 0.93190



      0%|          | 0/1200 [00:00<?, ?it/s]


    Epoch [6/10], loss: 0.23419 acc: 0.93302 val_loss: 0.22800, val_acc: 0.93360



      0%|          | 0/1200 [00:00<?, ?it/s]


    Epoch [7/10], loss: 0.21595 acc: 0.93800 val_loss: 0.20686, val_acc: 0.94190



      0%|          | 0/1200 [00:00<?, ?it/s]


    Epoch [8/10], loss: 0.19983 acc: 0.94313 val_loss: 0.19247, val_acc: 0.94300



      0%|          | 0/1200 [00:00<?, ?it/s]


    Epoch [9/10], loss: 0.18607 acc: 0.94723 val_loss: 0.18570, val_acc: 0.94650



      0%|          | 0/1200 [00:00<?, ?it/s]


    Epoch [10/10], loss: 0.17366 acc: 0.95130 val_loss: 0.17183, val_acc: 0.94890



```python
# 학습 곡선 출력(정확도)

plt.plot(history[:,0], history[:,4], label='batch_size=500', c='k', linestyle='-.')
plt.plot(history3[:,0], history3[:,4], label='batch_size=200', c='b', linestyle='-.')
plt.plot(history4[:,0], history4[:,4], label='batch_size=100', c='k')
plt.plot(history5[:,0], history5[:,4], label='batch_size=50', c='b')
plt.xlabel('반복 횟수')
plt.ylabel('정확도')
plt.title('학습 곡선(정확도)')
plt.legend()
plt.show()
```


    
*(이미지 없음 / image not preserved: output_110_0.png)*
    

