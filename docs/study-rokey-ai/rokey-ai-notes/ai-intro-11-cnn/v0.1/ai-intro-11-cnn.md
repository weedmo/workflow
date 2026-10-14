# 11장 CNN을 활용한 이미지 인식

> **Migrated from Quartz wiki** (`wiki/study/rokey-ai/basic/ai-intro-11-cnn.md`, updated 2026-09-26)  
> tags: rokey, ai, notebook, basic, study  
> source: raw/2026-09-26-rokey-ai/basic/강의_3기_AI개론_11차시__CNN_.ipynb  
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
import torch
from torch import nn, optim
from torchinfo import summary
from torchviz import make_dot
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
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

### GPU 확인하기


```python
# 디바이스 할당

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
```

    cpu


## CNN의 처리 개요


```python
data_root = './data'

# 샘플 손글씨 숫자 데이터 가져오기
transform = transforms.Compose([
    transforms.ToTensor(),
])

train_set = datasets.MNIST(
    root = data_root,
    train = True,
    download = True,
    transform = transform)

image, label = train_set[0]  # torch.Size([1, 28, 28])
image = image.view(1,1,28,28)
```

    Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
    Failed to download (trying next):
    HTTP Error 404: Not Found
    
    Downloading https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz
    Downloading https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz to ./data/MNIST/raw/train-images-idx3-ubyte.gz


    100%|██████████| 9.91M/9.91M [00:00<00:00, 52.6MB/s]


    Extracting ./data/MNIST/raw/train-images-idx3-ubyte.gz to ./data/MNIST/raw
    
    Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
    Failed to download (trying next):
    HTTP Error 404: Not Found
    
    Downloading https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz
    Downloading https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz to ./data/MNIST/raw/train-labels-idx1-ubyte.gz


    100%|██████████| 28.9k/28.9k [00:00<00:00, 2.04MB/s]


    Extracting ./data/MNIST/raw/train-labels-idx1-ubyte.gz to ./data/MNIST/raw
    
    Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
    Failed to download (trying next):
    HTTP Error 404: Not Found
    
    Downloading https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz
    Downloading https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz to ./data/MNIST/raw/t10k-images-idx3-ubyte.gz


    100%|██████████| 1.65M/1.65M [00:00<00:00, 14.1MB/s]


    Extracting ./data/MNIST/raw/t10k-images-idx3-ubyte.gz to ./data/MNIST/raw
    
    Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
    Failed to download (trying next):
    HTTP Error 404: Not Found
    
    Downloading https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz
    Downloading https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz to ./data/MNIST/raw/t10k-labels-idx1-ubyte.gz


    100%|██████████| 4.54k/4.54k [00:00<00:00, 7.70MB/s]


    Extracting ./data/MNIST/raw/t10k-labels-idx1-ubyte.gz to ./data/MNIST/raw
    



```python
# 대각선상에만 가중치를 갖는 특수한 합성곱 함수를 만듦
conv1 = nn.Conv2d(1, 1, 3)
print("conv1.weight.shape = ", conv1.weight.shape) # [outputs, channel, kernel size (3x3)]
print("="*50)
print("conv1.weight = \n", conv1.weight)
print("conv1.bias = ", conv1.bias)

# bias를 0으로
nn.init.constant_(conv1.bias, 0.0)
# conv1.bias.data = torch.tensor([0]).float()
```

    conv1.weight.shape =  torch.Size([1, 1, 3, 3])
    ==================================================
    conv1.weight = 
     Parameter containing:
    tensor([[[[-0.0957,  0.1489, -0.0058],
              [ 0.0169, -0.1085, -0.1669],
              [-0.1860,  0.1392, -0.2057]]]], requires_grad=True)
    conv1.bias =  Parameter containing:
    tensor([-0.1928], requires_grad=True)





    Parameter containing:
    tensor([0.], requires_grad=True)




```python

# weight를 특수한 값으로
w1_np = np.array([[0,0,1],[0,1,0],[1,0,0]])
print("w1_np = \n", w1_np)
w1 = torch.tensor(w1_np).float() # torch.Size([3, 3])
w1 = w1.view(1,1,3,3)
conv1.weight.data = w1
# conv1.weight
```

    w1_np = 
     [[0 0 1]
     [0 1 0]
     [1 0 0]]



```python
# 손글씨 숫자에 3번 합성곱 처리를 함
import cv2

image, label = train_set[0] # torch.Size([1, 28, 28])
image = image.view(1,1,28,28)
w1 = conv1(image)
w2 = conv1(w1)
w3 = conv1(w2)
images = [image, w1, w2, w3]
```


```python
# 결과 화면 출력

plt.figure(figsize=(5, 1))
for i in range(4):
    size = (28 - i*2)
    ax = plt.subplot(1, 4, i+1)
    img = images[i].data.numpy()
    plt.imshow(img.reshape(size, size),cmap='gray_r')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
```


    
*(이미지 없음 / image not preserved: output_15_0.png)*
    


### nn.Conv2d 와 nn.MaxPool2d


```python
# CNN 모델 전반 부분, 레이어 함수 정의
# torch.nn.Conv2d(in_channels, out_channels, kernel_size,
#                 stride=1, padding=0, dilation=1, groups=1,
#                 bias=True, padding_mode='zeros', device=None, dtype=None)

conv1 = nn.Conv2d(3, 32, 3)
relu = nn.ReLU(inplace=True)
conv2 = nn.Conv2d(32, 32, 3)
maxpool = nn.MaxPool2d((2, 2))

print("conv1.weight.shape = \n", conv1.weight.shape)
```

    conv1.weight.shape = 
     torch.Size([32, 3, 3, 3])



```python
# conv1 확인
print("conv1")
print(conv1)

# conv1 내부 변수의 shape 확인
print(conv1.weight.shape) # torch.Size([32, 3, 3, 3]), (N, C, H, W)
print(conv1.bias.shape)

# conv2 내부 변수의 shape 확인
print("="*50)
print("conv2")
print(conv2.weight.shape)
print(conv2.bias.shape)
```

    conv1
    Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1))
    torch.Size([32, 3, 3, 3])
    torch.Size([32])
    ==================================================
    conv2
    torch.Size([32, 32, 3, 3])
    torch.Size([32])



```python
# conv1의 weight[0]는 0번째 출력 채널의 가중치
w = conv1.weight[0]

# weight[0]의 shape과 값 확인
print(w.shape)
print(w.data.numpy())
```

    torch.Size([3, 3, 3])
    [[[ 0.016   0.1674  0.0329]
      [-0.1008 -0.0676 -0.101 ]
      [ 0.0564  0.1219 -0.1526]]
    
     [[-0.1257 -0.1425 -0.1288]
      [ 0.0243  0.0016 -0.122 ]
      [ 0.0139 -0.0518  0.1894]]
    
     [[-0.0984 -0.0113 -0.1561]
      [ 0.      0.006   0.0492]
      [-0.0433  0.1602  0.1758]]]



```python
# 더미로 입력과 같은 사이즈를 갖는 텐서를 생성
inputs = torch.randn(100, 3, 32, 32)
print(inputs.shape)

## image show
plt.imshow(inputs[0].permute(1, 2, 0))
plt.show()
```

    WARNING:matplotlib.image:Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-3.3060312..3.282588].


    torch.Size([100, 3, 32, 32])



    
*(이미지 없음 / image not preserved: output_20_2.png)*
    



```python
inputs.shape
```




    torch.Size([100, 3, 32, 32])




```python
# CNN 전반부 처리 시뮬레이션

x1 = conv1(inputs) # input size = torch.Size([100, 3, 32, 32])
x2 = relu(x1)
x3 = conv2(x2)
x4 = relu(x3)
x5 = maxpool(x4)
```


```python
# 각 변수의 shape 확인

print(inputs.shape)
print(x1.shape)
print(x2.shape)
print(x3.shape)
print(x4.shape)
print(x5.shape)
```

    torch.Size([100, 3, 32, 32])
    torch.Size([100, 32, 30, 30])
    torch.Size([100, 32, 30, 30])
    torch.Size([100, 32, 28, 28])
    torch.Size([100, 32, 28, 28])
    torch.Size([100, 32, 14, 14])


### nn.Sequential


```python
# conv1 = nn.Conv2d(3, 32, 3)
# relu = nn.ReLU(inplace=True)
# conv2 = nn.Conv2d(32, 32, 3)
# maxpool = nn.MaxPool2d((2, 2))

# 함수 정의
features = nn.Sequential(
    conv1,
    relu,
    conv2,
    relu,
    maxpool
)

# 동작 테스트
outputs = features(inputs)
```


```python
# 동작 테스트
outputs = features(inputs)

# 결과 확인
print(outputs.shape)
```

    torch.Size([100, 32, 14, 14])


### nn.Flatten


```python
# 함수 정의
flatten = nn.Flatten()

# 동작 테스트
outputs2 = flatten(outputs)

# 결과 확인
print(outputs.shape)
print(outputs2.shape)
```

    torch.Size([100, 32, 14, 14])
    torch.Size([100, 6272])


### eval_loss(손실 계산)


```python
# 손실 계산용
def eval_loss(loader, device, net, criterion):

    # 데이터로더에서 처음 한 개 세트를 가져옴
    for images, labels in loader:
        break

    # 디바이스 할당
    inputs = images.to(device)
    labels = labels.to(device)

    # 예측 계산
    outputs = net(inputs)

    # 손실 계산
    loss = criterion(outputs, labels)

    return loss
```

### fit(학습)


```python
# 학습용 함수
def fit(net, optimizer, criterion, num_epochs, train_loader, test_loader, device, history):

    # tqdm 라이브러리 임포트
    from tqdm.notebook import tqdm

    base_epochs = len(history) # => 0
    batch_size_train = len(train_loader)
    batch_size_test = len(test_loader)

    for epoch in range(base_epochs, num_epochs+base_epochs):
        train_loss = 0
        train_acc = 0
        val_loss = 0
        val_acc = 0

        # 훈련 페이즈
        net.train() # dropout, batch normalization 활성화
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

            # 손실과 정확도 계산
        avg_train_loss = train_loss / batch_size_train
        avg_train_acc = train_acc / batch_size_train

        # 예측 페이즈
        net.eval()
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

            # 손실과 정확도 계산
        avg_val_loss = val_loss / batch_size_test
        avg_val_acc = val_acc / batch_size_test

        print (f'Epoch [{(epoch+1)}/{num_epochs+base_epochs}], loss: {avg_train_loss:.5f} acc: {avg_train_acc:.5f} val_loss: {avg_val_loss:.5f}, val_acc: {avg_val_acc:.5f}')
        item = np.array([epoch+1, avg_train_loss, avg_train_acc, avg_val_loss, avg_val_acc])
        history = np.vstack((history, item))
    return history
```

### eval_history(학습 로그)


```python
# 학습 로그 해석

def evaluate_history(history):
    # 손실과 정확도 확인
    print(f'초기상태 : 손실 : {history[0,3]:.5f}  정확도 : {history[0,4]:.5f}')
    print(f'최종상태 : 손실 : {history[-1,3]:.5f}  정확도 : {history[-1,4]:.5f}' )

    num_epochs = len(history)
    unit = num_epochs / 10

    # 학습 곡선 출력(손실)
    plt.figure(figsize=(9,8))
    plt.plot(history[:,0], history[:,1], 'b', label='훈련')
    plt.plot(history[:,0], history[:,3], 'k', label='검증')
    plt.xticks(np.arange(0,num_epochs+1, unit))
    plt.xlabel('반복 횟수')
    plt.ylabel('손실')
    plt.title('학습 곡선(손실)')
    plt.legend()
    plt.show()

    # 학습 곡선 출력(정확도)
    plt.figure(figsize=(9,8))
    plt.plot(history[:,0], history[:,2], 'b', label='훈련')
    plt.plot(history[:,0], history[:,4], 'k', label='검증')
    plt.xticks(np.arange(0,num_epochs+1,unit))
    plt.xlabel('반복 횟수')
    plt.ylabel('정확도')
    plt.title('학습 곡선(정확도)')
    plt.legend()
    plt.show()
```

### show_images_labels(예측 결과 표시)


```python
# 이미지와 라벨 표시
def show_images_labels(loader, classes, net, device):

    # 데이터로더에서 처음 1세트를 가져오기
    for images, labels in loader:
        break
    # 표시 수는 50개
    n_size = min(len(images), 50)
    print("n_size = ", n_size)

    if net is not None:
      # 디바이스 할당
      inputs = images.to(device)
      labels = labels.to(device)

      # 예측 계산
      outputs = net(inputs)
      predicted = torch.max(outputs,1)[1]
      #images = images.to('cpu')

    # 처음 n_size개 표시
    plt.figure(figsize=(20, 15))
    for i in range(n_size):
        ax = plt.subplot(5, 10, i + 1)
        label_name = classes[labels[i]]
        # net이 None이 아닌 경우는 예측 결과도 타이틀에 표시함
        if net is not None:
          predicted_name = classes[predicted[i]]
          # 정답인지 아닌지 색으로 구분함
          if label_name == predicted_name:
            c = 'k'
          else:
            c = 'b'
          ax.set_title(label_name + ':' + predicted_name, c=c, fontsize=20)
        # net이 None인 경우는 정답 라벨만 표시
        else:
          ax.set_title(label_name, fontsize=20)
        # 텐서를 넘파이로 변환
        image_np = images[i].numpy().copy()
        # 축의 순서 변경 (channel, row, column) -> (row, column, channel)
        img = np.transpose(image_np, (1, 2, 0))
        # 값의 범위를[-1, 1] -> [0, 1]로 되돌림
        img = (img + 1)/2
        # 결과 표시
        plt.imshow(img)
        ax.set_axis_off()
    plt.show()

```

### torch_seed(난수 초기화)


```python
# 파이토치 난수 고정

def torch_seed(seed=123):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True #
    torch.use_deterministic_algorithms = True
```

### 데이터 준비


```python
# Transforms의 정의

# transformer1 1계 텐서화

transform1 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5),
    transforms.Lambda(lambda x: x.view(-1)),
])

# transformer2 정규화만 실시

# 검증 데이터용 : 정규화만 실시
transform2 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5),
])
```


```python
# 데이터 취득용 함수 datasets

data_root = './data'

# 훈련 데이터셋 (1계 텐서 버전)
train_set1 = datasets.CIFAR10(
    root = data_root,
    train = True,
    download = True,
    transform = transform1)

# 검증 데이터셋 (1계 텐서 버전)
test_set1 = datasets.CIFAR10(
    root = data_root,
    train = False,
    download = True,
    transform = transform1)

# 훈련 데이터셋 (3계 텐서 버전)
train_set2 = datasets.CIFAR10(
    root =  data_root,
    train = True,
    download = True,
    transform = transform2)

# 검증 데이터셋 (3계 텐서 버전)
test_set2 = datasets.CIFAR10(
    root = data_root,
    train = False,
    download = True,
    transform = transform2)
```

    Downloading https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to ./data/cifar-10-python.tar.gz


    100%|██████████| 170M/170M [00:07<00:00, 24.3MB/s]


    Extracting ./data/cifar-10-python.tar.gz to ./data
    Files already downloaded and verified
    Files already downloaded and verified
    Files already downloaded and verified


### 데이터셋 확인


```python
len(train_set1)
```




    50000




```python
image1, label1 = train_set1[0] # 3 x 32 x 32 = [3072]
image2, label2 = train_set2[0]

print(image1.shape)
print(image2.shape)
```

    torch.Size([3072])
    torch.Size([3, 32, 32])



```python
# 데이터로더 정의

# 미니 배치 사이즈 지정
batch_size = 100

# 훈련용 데이터로더
# 훈련용이므로 셔플을 True로 설정
train_loader1 = DataLoader(train_set1, batch_size=batch_size, shuffle=True)

# 검증용 데이터로더
# 검증용이므로 셔플하지 않음
test_loader1 = DataLoader(test_set1,  batch_size=batch_size, shuffle=False)

# 훈련용 데이터로더
# 훈련용이므로 셔플을 True로 설정
train_loader2 = DataLoader(train_set2, batch_size=batch_size, shuffle=True)

# 검증용 데이터로더
# 검증용이므로 셔플하지 않음
test_loader2 = DataLoader(test_set2,  batch_size=batch_size, shuffle=False)

```


```python
len(train_loader1)
```




    500




```python
# train_loader1에서 한 세트 가져오기
for images1, labels1 in train_loader1:
    break

# train_loader2에서 한 세트 가져오기
for images2, labels2 in train_loader2:
    break

#
print(images1.shape)
print(images2.shape)
```

    torch.Size([100, 3072])
    torch.Size([100, 3, 32, 32])



```python
# 정답 라벨 정의
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 검증 데이터의 처음 50개를 출력
show_images_labels(test_loader2, classes, None, None)
```

    n_size =  50



    
*(이미지 없음 / image not preserved: output_48_1.png)*
    


### 학습용 파라미터 설정


```python
# 입력 차원수는 3*32*32=3072
n_input = image1.view(-1).shape[0]

# 출력 차원수
# 분류 클래스의 수이므로　10
n_output = len(set(list(labels1.data.numpy())))
# np.unique(labels1.data.numpy()).size
# 은닉층의 노드수
n_hidden = 128

# 결과 확인
print(f'n_input: {n_input}  n_hidden: {n_hidden} n_output: {n_output}')
```

    n_input: 3072  n_hidden: 128 n_output: 10



```python
# 모델 정의
# 3072입력 10출력 1은닉층을 포함한 신경망 모델

class Net(nn.Module):
    def __init__(self, n_input, n_output, n_hidden):
        super().__init__()

        # 은닉층 정의(은닉층의 노드수 : n_hidden)
        self.l1 = nn.Linear(n_input, n_hidden)

        # 출력층의 정의
        self.l2 = nn.Linear(n_hidden, n_output)

        # ReLU 함수 정의
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.l1(x)
        x2 = self.relu(x1)
        x3 = self.l2(x2)
        return x3
```

### 모델 인스턴스 생성과 GPU 할당


```python
# 모델 인스턴스 생성
net = Net(n_input, n_output, n_hidden).to(device)

# 손실 함수： 교차 엔트로피 함수
criterion = nn.CrossEntropyLoss()

# 학습률
lr = 0.01

# 최적화 함수: 경사 하강법
optimizer = torch.optim.SGD(net.parameters(), lr=lr)
```


```python
# 모델 개요 표시 1

print(net)
```

    Net(
      (l1): Linear(in_features=3072, out_features=128, bias=True)
      (l2): Linear(in_features=128, out_features=10, bias=True)
      (relu): ReLU(inplace=True)
    )



```python
# 모델 개요 표시 2

summary(net, (100, 3072), depth=1)
```




    ==========================================================================================
    Layer (type:depth-idx)                   Output Shape              Param #
    ==========================================================================================
    Net                                      [100, 10]                 --
    ├─Linear: 1-1                            [100, 128]                393,344
    ├─ReLU: 1-2                              [100, 128]                --
    ├─Linear: 1-3                            [100, 10]                 1,290
    ==========================================================================================
    Total params: 394,634
    Trainable params: 394,634
    Non-trainable params: 0
    Total mult-adds (Units.MEGABYTES): 39.46
    ==========================================================================================
    Input size (MB): 1.23
    Forward/backward pass size (MB): 0.11
    Params size (MB): 1.58
    Estimated Total Size (MB): 2.92
    ==========================================================================================




```python
# 손실 계산
loss = eval_loss(test_loader1, device, net, criterion)

# 손실 계산 그래프 시각화
g = make_dot(loss, params=dict(net.named_parameters()))
display(g)
```


    
*(이미지 없음 / image not preserved: output_56_0.svg)*
    


### 학습


```python
# 난수 초기화
torch_seed()

# 모델 인스턴스 생성
net = Net(n_input, n_output, n_hidden).to(device)

# 손실 함수： 교차 엔트로피 함수
criterion = nn.CrossEntropyLoss()

# 학습률
lr = 0.01

# 최적화 함수: 경사 하강법
optimizer = optim.SGD(net.parameters(), lr=lr)

# 반복 횟수
num_epochs = 10

# 평가 결과 기록
history = np.zeros((0,5))

# 학습
history = fit(net, optimizer, criterion, num_epochs, train_loader1, test_loader1, device, history)
```


      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [1/10], loss: 1.94965 acc: 0.32218 val_loss: 1.79424, val_acc: 0.37710



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [2/10], loss: 1.73836 acc: 0.39598 val_loss: 1.68423, val_acc: 0.41850



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [3/10], loss: 1.65492 acc: 0.42398 val_loss: 1.62226, val_acc: 0.43860



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [4/10], loss: 1.60225 acc: 0.44256 val_loss: 1.58253, val_acc: 0.45150



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [5/10], loss: 1.56317 acc: 0.45540 val_loss: 1.55320, val_acc: 0.46170



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [6/10], loss: 1.53229 acc: 0.46760 val_loss: 1.52983, val_acc: 0.46830



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [7/10], loss: 1.50488 acc: 0.47688 val_loss: 1.51209, val_acc: 0.47400



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [8/10], loss: 1.48005 acc: 0.48632 val_loss: 1.49287, val_acc: 0.47750



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [9/10], loss: 1.45687 acc: 0.49624 val_loss: 1.47964, val_acc: 0.48740



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [10/10], loss: 1.43482 acc: 0.50422 val_loss: 1.46307, val_acc: 0.48860


### 평가


```python
# 평가
evaluate_history(history)
```

    초기상태 : 손실 : 1.79424  정확도 : 0.37710
    최종상태 : 손실 : 1.46307  정확도 : 0.48860



    
*(이미지 없음 / image not preserved: output_60_1.png)*
    



    
*(이미지 없음 / image not preserved: output_60_2.png)*
    


### 모델 정의(CNN)


```python
class CNN(nn.Module):
  def __init__(self, n_output, n_hidden):
    super().__init__()
    self.conv1 = nn.Conv2d(3, 32, 3)
    self.conv2 = nn.Conv2d(32, 32, 3)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d((2,2))
    self.flatten = nn.Flatten()
    self.l1 = nn.Linear(6272, n_hidden)
    self.l2 = nn.Linear(n_hidden, n_output)

    self.features = nn.Sequential(
        self.conv1,
        self.relu,
        self.conv2,
        self.relu,
        self.maxpool)

    self.classifier = nn.Sequential(
       self.l1,
       self.relu,
       self.l2)

  def forward(self, x):
    x1 = self.features(x)
    x2 = self.flatten(x1)
    x3 = self.classifier(x2)
    return x3
```

### 모델 인스턴스 생성


```python
# 모델 인스턴스 생성
net = CNN(n_output, n_hidden).to(device)

# 손실 함수： 교차 엔트로피 함수
criterion = nn.CrossEntropyLoss()

# 학습률
lr = 0.01

# 최적화 함수: 경사 하강법
optimizer = torch.optim.SGD(net.parameters(), lr=lr)
```


```python
# 모델 개요 표시 1

print(net)
```

    CNN(
      (conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1))
      (conv2): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1))
      (relu): ReLU(inplace=True)
      (maxpool): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
      (flatten): Flatten(start_dim=1, end_dim=-1)
      (l1): Linear(in_features=6272, out_features=128, bias=True)
      (l2): Linear(in_features=128, out_features=10, bias=True)
      (features): Sequential(
        (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1))
        (1): ReLU(inplace=True)
        (2): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1))
        (3): ReLU(inplace=True)
        (4): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
      )
      (classifier): Sequential(
        (0): Linear(in_features=6272, out_features=128, bias=True)
        (1): ReLU(inplace=True)
        (2): Linear(in_features=128, out_features=10, bias=True)
      )
    )



```python
# 모델 개요 표시2

summary(net, (100,3,32,32), depth = 2)
```




    ==========================================================================================
    Layer (type:depth-idx)                   Output Shape              Param #
    ==========================================================================================
    CNN                                      [100, 10]                 --
    ├─Sequential: 1-1                        [100, 32, 14, 14]         9,248
    │    └─Conv2d: 2-1                       [100, 32, 30, 30]         896
    ├─Sequential: 1-4                        --                        (recursive)
    │    └─ReLU: 2-2                         [100, 32, 30, 30]         --
    ├─Sequential: 1-5                        --                        (recursive)
    │    └─Conv2d: 2-3                       [100, 32, 28, 28]         9,248
    ├─Sequential: 1-4                        --                        (recursive)
    │    └─ReLU: 2-4                         [100, 32, 28, 28]         --
    ├─Sequential: 1-5                        --                        (recursive)
    │    └─MaxPool2d: 2-5                    [100, 32, 14, 14]         --
    ├─Flatten: 1-6                           [100, 6272]               --
    ├─Sequential: 1-7                        [100, 10]                 --
    │    └─Linear: 2-6                       [100, 128]                802,944
    │    └─ReLU: 2-7                         [100, 128]                --
    │    └─Linear: 2-8                       [100, 10]                 1,290
    ==========================================================================================
    Total params: 823,626
    Trainable params: 823,626
    Non-trainable params: 0
    Total mult-adds (Units.MEGABYTES): 886.11
    ==========================================================================================
    Input size (MB): 1.23
    Forward/backward pass size (MB): 43.22
    Params size (MB): 3.26
    Estimated Total Size (MB): 47.71
    ==========================================================================================




```python
# 손실 계산
loss = eval_loss(test_loader2, device, net, criterion)

# 손실 계산 그래프 시각화
g = make_dot(loss, params=dict(net.named_parameters()))
display(g)
```


    
*(이미지 없음 / image not preserved: output_67_0.svg)*
    


### 결과(CNN)


```python
# 난수 초기화
torch_seed()

# 모델 인스턴스 생성
net = CNN(n_output, n_hidden).to(device)

# 손실 함수： 교차 엔트로피 함수
criterion = nn.CrossEntropyLoss()

# 학습률
lr = 0.01

# 최적화 함수: 경사 하강법
optimizer = optim.SGD(net.parameters(), lr=lr)

# 반복 횟수
num_epochs = 10

# 평가 결과 기록
history2 = np.zeros((0,5))

# 학습
history2 = fit(net, optimizer, criterion, num_epochs, train_loader2, test_loader2, device, history2)
```


      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [1/10], loss: 2.08246 acc: 0.26084 val_loss: 1.86593, val_acc: 0.34690



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [2/10], loss: 1.78080 acc: 0.37296 val_loss: 1.67678, val_acc: 0.40950



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [3/10], loss: 1.61318 acc: 0.43058 val_loss: 1.53056, val_acc: 0.45960



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [4/10], loss: 1.48527 acc: 0.47320 val_loss: 1.44834, val_acc: 0.49010



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [5/10], loss: 1.40808 acc: 0.49936 val_loss: 1.37022, val_acc: 0.51260



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [6/10], loss: 1.34984 acc: 0.52108 val_loss: 1.33102, val_acc: 0.52650



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [7/10], loss: 1.30325 acc: 0.53764 val_loss: 1.29277, val_acc: 0.53840



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [8/10], loss: 1.25244 acc: 0.55482 val_loss: 1.25406, val_acc: 0.55170



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [9/10], loss: 1.20528 acc: 0.57400 val_loss: 1.23566, val_acc: 0.56080



      0%|          | 0/500 [00:00<?, ?it/s]


    Epoch [10/10], loss: 1.15801 acc: 0.59202 val_loss: 1.18459, val_acc: 0.58010



```python
# 평가

evaluate_history(history2)
```

    초기상태 : 손실 : 1.86593  정확도 : 0.34690
    최종상태 : 손실 : 1.18459  정확도 : 0.58010



    
*(이미지 없음 / image not preserved: output_70_1.png)*
    



    
*(이미지 없음 / image not preserved: output_70_2.png)*
    



```python
# 처음 50개 데이터 표시

show_images_labels(test_loader2, classes, net, device)
```

    n_size =  50



    
*(이미지 없음 / image not preserved: output_71_1.png)*
    

