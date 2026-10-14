# 8장 이진 분류 (Binary classification)

> **Migrated from Quartz wiki** (`wiki/study/rokey-ai/basic/ai-intro-08-binary.md`, updated 2026-09-26)  
> tags: rokey, ai, notebook, basic, study  
> source: raw/2026-09-26-rokey-ai/basic/강의_3기_AI개론_8차시__Binary_.ipynb  
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

# Iris dataset
import pandas  as pd
# from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
```


```python
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
# warning 표시 끄기
import warnings
warnings.simplefilter('ignore')
```

## Iris data 분석

### 데이터 준비


```python
# 학습용 데이터 준비
# 라이브러리 임포트
# import sklearn


# 데이터 불러오기
iris = load_iris()
print(iris.keys())

# 입력 데이터와 정답 데이터
x_org, y_org = iris.data, iris.target

# 결과 확인
print('원본 데이터', x_org.shape, y_org.shape)
```

    dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])
    원본 데이터 (150, 4) (150,)



```python
# 데이터 추출
# 클래스는 0 또는 1
# 항목은 sepal_length와 sepal_width

x_data = iris.data[:100,:2] # 2-dim
y_data = iris.target[:100] # 1-dim

print("x data = \n", x_data[:10])
print("y data = \n", y_data[:10])
print("feature names = ", iris.feature_names[:2])
# 결과 확인
print('대상 데이터', x_data.shape, y_data.shape)
```

    x data = 
     [[5.1 3.5]
     [4.9 3. ]
     [4.7 3.2]
     [4.6 3.1]
     [5.  3.6]
     [5.4 3.9]
     [4.6 3.4]
     [5.  3.4]
     [4.4 2.9]
     [4.9 3.1]]
    y data = 
     [0 0 0 0 0 0 0 0 0 0]
    feature names =  ['sepal length (cm)', 'sepal width (cm)']
    대상 데이터 (100, 2) (100,)


### 훈련 데이터와 검증 데이터 분할


```python
# 원본 데이터의 사이즈
print("Original data shape = ")
print(x_data.shape, y_data.shape)

# 훈련 데이터와 검증 데이터로 분할(동시에 셔플)

x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, train_size=70, test_size=30, random_state=123)

print("x_train, x_test, y_train, y_test :")
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
```

    Original data shape = 
    (100, 2) (100,)
    x_train, x_test, y_train, y_test :
    (70, 2) (30, 2) (70,) (30,)


### 산포도 출력


```python
# 산포도 출력

x_t0 = x_train[y_train == 0]
x_t1 = x_train[y_train == 1]
plt.scatter(x_t0[:,0], x_t0[:,1], marker='x', c='b', label='0 (setosa)')
plt.scatter(x_t1[:,0], x_t1[:,1], marker='o', c='k', label='1 (versicolor)')
plt.xlabel('sepal_length')
plt.ylabel('sepal_width')
plt.legend()
plt.show()
```


    
*(이미지 없음 / image not preserved: output_16_0.png)*
    


### 모델 정의


```python
# 입력 차원수(지금의 경우는 2)
n_input= x_train.shape[1]

# 출력 차원수
n_output = 1

# 결과 확인
print(f'n_input: {n_input}  n_output:{n_output}')
```

    n_input: 2  n_output:1



```python
# 모델 정의
# 2입력 1출력 로지스틱 회귀 모델

class Net(nn.Module):
    def __init__(self, n_input, n_output):
        super().__init__()
        self.l1 = nn.Linear(n_input, n_output)
        self.sigmoid = nn.Sigmoid()

        # 초깃값을 전부 1로 함
        # "딥러닝을 위한 수학"과 조건을 맞추기 위한 목적
        # self.l1.weight.data.fill_(1.0)
        # self.l1.bias.data.fill_(1.0)

        nn.init.constant_(self.l1.weight, 1.0)
        nn.init.constant_(self.l1.bias, 1.0)


    # 예측 함수 정의
    def forward(self, x):
        # 선형 함수에 입력값을 넣고 계산한 결과
        x1 = self.l1(x)
        # 계산 결과에 시그모이드 함수를 적용
        x2 = self.sigmoid(x1)
        return x2
```


```python
# 인스턴스 생성

net = Net(n_input, n_output)
```

### 모델 확인


```python
# 모델 안의 파라미터 확인
# l1.weight와 l1.bias가 존재함을 알 수 있음

for parameter in net.named_parameters():
    print(parameter[1])
```

    Parameter containing:
    tensor([[1., 1.]], requires_grad=True)
    Parameter containing:
    tensor([1.], requires_grad=True)



```python
# 모델의 개요 표시 1

print(net)

# 모델의 개요 표시 2
print("="*50)
summary(net, (2,), device = "cpu") # device default = "cuda"
```

    Net(
      (l1): Linear(in_features=2, out_features=1, bias=True)
      (sigmoid): Sigmoid()
    )
    ==================================================





    ==========================================================================================
    Layer (type:depth-idx)                   Output Shape              Param #
    ==========================================================================================
    Net                                      [1]                       --
    ├─Linear: 1-1                            [1]                       3
    ├─Sigmoid: 1-2                           [1]                       --
    ==========================================================================================
    Total params: 3
    Trainable params: 3
    Non-trainable params: 0
    Total mult-adds (Units.MEGABYTES): 0.00
    ==========================================================================================
    Input size (MB): 0.00
    Forward/backward pass size (MB): 0.00
    Params size (MB): 0.00
    Estimated Total Size (MB): 0.00
    ==========================================================================================



### 최적화 알고리즘과 손실 함수의 정의


```python
for parameter in net.parameters():
    print(parameter)
```

    Parameter containing:
    tensor([[1., 1.]], requires_grad=True)
    Parameter containing:
    tensor([1.], requires_grad=True)



```python
# 손실 함수： 교차 엔트로피 함수
loss = nn.BCELoss()

# 학습률
lr = 0.01

# 최적화 함수: 경사 하강법
optimizer = optim.SGD(net.parameters(), lr=lr)
```

### 경사 하강법


```python
# 입력 데이터 x_train과 정답 데이터 y_train의 텐서화

inputs = torch.tensor(x_train).float()
labels = torch.tensor(y_train).float()

# 정답 데이터는 N행 1열 행렬로 변환
labels1 = labels.view((-1,1))
print("labels1 shape = ", labels1.shape)

# 검증 데이터의 텐서화
inputs_test = torch.tensor(x_test).float()
labels_test = torch.tensor(y_test).float()

# 검증용 정답 데이터도 N행 1열 행렬로 변환
labels1_test = labels_test.view((-1,1))
```

    labels1 shape =  torch.Size([70, 1])



```python
# 예측 계산
outputs = net(inputs) # outputs.shape = torch.Size([70, 1])

# 손실 계산
cost = loss(outputs, labels1)

# 손실을 계산 그래프로 출력
g = make_dot(cost, params=dict(net.named_parameters()))
display(g)
```


    
*(이미지 없음 / image not preserved: output_29_0.svg)*
    


### 반복 계산 (Iterative learning)


```python
# 학습률
lr = 0.01

# 초기화
net = Net(n_input, n_output)

# 손실 함수
criterion = nn.BCELoss()

# 최적화 함수: 경사 하강법
optimizer = optim.SGD(net.parameters(), lr=lr)

# 반복 횟수
num_epochs = 10000

# 기록용 리스트 초기화
history = np.zeros((0,5))
```


```python
# 반복 계산 메인 루프

for epoch in range(num_epochs):
    # 훈련 페이즈

    # 경삿값 초기화
    optimizer.zero_grad()

    # 예측 계산
    outputs = net(inputs)

    # 손실 계산
    loss = criterion(outputs, labels1)

    # 경사 계산
    loss.backward()

    # 파라미터 수정
    optimizer.step()

    # 손실 저장(스칼라 값 취득)
    train_loss = loss.item()

    # 예측 라벨(1 또는 0) 계산
    predicted = torch.where(outputs < 0.5, 0, 1)

    # 정확도 계산
    train_acc = (predicted == labels1).sum() / len(y_train)

    # 예측 페이즈

    # 예측 계산
    outputs_test = net(inputs_test)

    # 손실 계산
    loss_test = criterion(outputs_test, labels1_test)

    # 손실 저장(스칼라 값 취득)
    val_loss =  loss_test.item()

    # 예측 라벨(1 또는 0) 계산
    predicted_test = torch.where(outputs_test < 0.5, 0, 1)

    # 정확도 계산
    val_acc = (predicted_test == labels1_test).sum() / len(y_test)

    if ( epoch % 10 == 0):
        print (f'Epoch [{epoch}/{num_epochs}], loss: {train_loss:.5f} acc: {train_acc:.5f} val_loss: {val_loss:.5f}, val_acc: {val_acc:.5f}')
        item = np.array([epoch, train_loss, train_acc, val_loss, val_acc])
        history = np.vstack((history, item))
```

    Epoch [0/10000], loss: 4.77289 acc: 0.50000 val_loss: 4.49384, val_acc: 0.50000
    Epoch [10/10000], loss: 3.80546 acc: 0.50000 val_loss: 3.56537, val_acc: 0.50000
    Epoch [20/10000], loss: 2.84329 acc: 0.50000 val_loss: 2.64328, val_acc: 0.50000
    Epoch [30/10000], loss: 1.91613 acc: 0.50000 val_loss: 1.76244, val_acc: 0.50000
    Epoch [40/10000], loss: 1.17137 acc: 0.50000 val_loss: 1.08537, val_acc: 0.50000
    Epoch [50/10000], loss: 0.84140 acc: 0.50000 val_loss: 0.81872, val_acc: 0.50000
    Epoch [60/10000], loss: 0.77087 acc: 0.50000 val_loss: 0.77093, val_acc: 0.50000
    Epoch [70/10000], loss: 0.75450 acc: 0.34286 val_loss: 0.76105, val_acc: 0.33333
    Epoch [80/10000], loss: 0.74542 acc: 0.25714 val_loss: 0.75447, val_acc: 0.20000
    Epoch [90/10000], loss: 0.73734 acc: 0.24286 val_loss: 0.74778, val_acc: 0.16667
    Epoch [100/10000], loss: 0.72949 acc: 0.24286 val_loss: 0.74098, val_acc: 0.13333
    Epoch [110/10000], loss: 0.72180 acc: 0.27143 val_loss: 0.73419, val_acc: 0.16667
    Epoch [120/10000], loss: 0.71423 acc: 0.31429 val_loss: 0.72749, val_acc: 0.20000
    Epoch [130/10000], loss: 0.70680 acc: 0.41429 val_loss: 0.72087, val_acc: 0.20000
    Epoch [140/10000], loss: 0.69949 acc: 0.47143 val_loss: 0.71437, val_acc: 0.26667
    Epoch [150/10000], loss: 0.69230 acc: 0.52857 val_loss: 0.70797, val_acc: 0.30000
    Epoch [160/10000], loss: 0.68524 acc: 0.60000 val_loss: 0.70167, val_acc: 0.36667
    Epoch [170/10000], loss: 0.67829 acc: 0.62857 val_loss: 0.69548, val_acc: 0.43333
    Epoch [180/10000], loss: 0.67147 acc: 0.68571 val_loss: 0.68938, val_acc: 0.50000
    Epoch [190/10000], loss: 0.66476 acc: 0.75714 val_loss: 0.68339, val_acc: 0.56667
    Epoch [200/10000], loss: 0.65816 acc: 0.81429 val_loss: 0.67749, val_acc: 0.70000
    Epoch [210/10000], loss: 0.65168 acc: 0.84286 val_loss: 0.67169, val_acc: 0.70000
    Epoch [220/10000], loss: 0.64531 acc: 0.85714 val_loss: 0.66599, val_acc: 0.73333
    Epoch [230/10000], loss: 0.63904 acc: 0.85714 val_loss: 0.66037, val_acc: 0.76667
    Epoch [240/10000], loss: 0.63288 acc: 0.88571 val_loss: 0.65485, val_acc: 0.80000
    Epoch [250/10000], loss: 0.62682 acc: 0.88571 val_loss: 0.64942, val_acc: 0.83333
    Epoch [260/10000], loss: 0.62087 acc: 0.90000 val_loss: 0.64408, val_acc: 0.83333
    Epoch [270/10000], loss: 0.61501 acc: 0.91429 val_loss: 0.63882, val_acc: 0.83333
    Epoch [280/10000], loss: 0.60925 acc: 0.92857 val_loss: 0.63364, val_acc: 0.86667
    Epoch [290/10000], loss: 0.60359 acc: 0.94286 val_loss: 0.62855, val_acc: 0.90000
    Epoch [300/10000], loss: 0.59803 acc: 0.94286 val_loss: 0.62354, val_acc: 0.90000
    Epoch [310/10000], loss: 0.59255 acc: 0.94286 val_loss: 0.61861, val_acc: 0.90000
    Epoch [320/10000], loss: 0.58717 acc: 0.94286 val_loss: 0.61376, val_acc: 0.93333
    Epoch [330/10000], loss: 0.58187 acc: 0.94286 val_loss: 0.60899, val_acc: 0.93333
    Epoch [340/10000], loss: 0.57667 acc: 0.97143 val_loss: 0.60429, val_acc: 0.93333
    Epoch [350/10000], loss: 0.57154 acc: 0.97143 val_loss: 0.59967, val_acc: 0.93333
    Epoch [360/10000], loss: 0.56650 acc: 0.97143 val_loss: 0.59512, val_acc: 0.93333
    Epoch [370/10000], loss: 0.56155 acc: 0.98571 val_loss: 0.59064, val_acc: 0.93333
    Epoch [380/10000], loss: 0.55667 acc: 0.98571 val_loss: 0.58623, val_acc: 0.93333
    Epoch [390/10000], loss: 0.55188 acc: 0.98571 val_loss: 0.58189, val_acc: 0.93333
    Epoch [400/10000], loss: 0.54716 acc: 0.98571 val_loss: 0.57762, val_acc: 0.93333
    Epoch [410/10000], loss: 0.54251 acc: 0.98571 val_loss: 0.57341, val_acc: 0.93333
    Epoch [420/10000], loss: 0.53795 acc: 0.98571 val_loss: 0.56927, val_acc: 0.93333
    Epoch [430/10000], loss: 0.53345 acc: 1.00000 val_loss: 0.56519, val_acc: 0.93333
    Epoch [440/10000], loss: 0.52902 acc: 1.00000 val_loss: 0.56117, val_acc: 0.93333
    Epoch [450/10000], loss: 0.52467 acc: 1.00000 val_loss: 0.55722, val_acc: 0.93333
    Epoch [460/10000], loss: 0.52038 acc: 1.00000 val_loss: 0.55333, val_acc: 0.93333
    Epoch [470/10000], loss: 0.51617 acc: 1.00000 val_loss: 0.54949, val_acc: 0.93333
    Epoch [480/10000], loss: 0.51201 acc: 1.00000 val_loss: 0.54571, val_acc: 0.93333
    Epoch [490/10000], loss: 0.50793 acc: 1.00000 val_loss: 0.54199, val_acc: 0.93333
    Epoch [500/10000], loss: 0.50390 acc: 1.00000 val_loss: 0.53833, val_acc: 0.93333
    Epoch [510/10000], loss: 0.49994 acc: 1.00000 val_loss: 0.53472, val_acc: 0.93333
    Epoch [520/10000], loss: 0.49604 acc: 1.00000 val_loss: 0.53116, val_acc: 0.93333
    Epoch [530/10000], loss: 0.49219 acc: 1.00000 val_loss: 0.52766, val_acc: 0.93333
    Epoch [540/10000], loss: 0.48841 acc: 1.00000 val_loss: 0.52421, val_acc: 0.93333
    Epoch [550/10000], loss: 0.48468 acc: 1.00000 val_loss: 0.52080, val_acc: 0.93333
    Epoch [560/10000], loss: 0.48101 acc: 1.00000 val_loss: 0.51745, val_acc: 0.93333
    Epoch [570/10000], loss: 0.47740 acc: 1.00000 val_loss: 0.51415, val_acc: 0.93333
    Epoch [580/10000], loss: 0.47384 acc: 1.00000 val_loss: 0.51089, val_acc: 0.93333
    Epoch [590/10000], loss: 0.47033 acc: 1.00000 val_loss: 0.50769, val_acc: 0.93333
    Epoch [600/10000], loss: 0.46687 acc: 1.00000 val_loss: 0.50452, val_acc: 0.93333
    Epoch [610/10000], loss: 0.46347 acc: 1.00000 val_loss: 0.50141, val_acc: 0.93333
    Epoch [620/10000], loss: 0.46011 acc: 1.00000 val_loss: 0.49833, val_acc: 0.93333
    Epoch [630/10000], loss: 0.45680 acc: 1.00000 val_loss: 0.49530, val_acc: 0.93333
    Epoch [640/10000], loss: 0.45355 acc: 1.00000 val_loss: 0.49232, val_acc: 0.93333
    Epoch [650/10000], loss: 0.45033 acc: 1.00000 val_loss: 0.48937, val_acc: 0.93333
    Epoch [660/10000], loss: 0.44717 acc: 1.00000 val_loss: 0.48647, val_acc: 0.93333
    Epoch [670/10000], loss: 0.44405 acc: 1.00000 val_loss: 0.48360, val_acc: 0.93333
    Epoch [680/10000], loss: 0.44097 acc: 1.00000 val_loss: 0.48078, val_acc: 0.93333
    Epoch [690/10000], loss: 0.43794 acc: 1.00000 val_loss: 0.47800, val_acc: 0.93333
    Epoch [700/10000], loss: 0.43495 acc: 1.00000 val_loss: 0.47525, val_acc: 0.93333
    Epoch [710/10000], loss: 0.43200 acc: 1.00000 val_loss: 0.47254, val_acc: 0.93333
    Epoch [720/10000], loss: 0.42909 acc: 1.00000 val_loss: 0.46987, val_acc: 0.93333
    Epoch [730/10000], loss: 0.42623 acc: 1.00000 val_loss: 0.46723, val_acc: 0.93333
    Epoch [740/10000], loss: 0.42340 acc: 1.00000 val_loss: 0.46463, val_acc: 0.93333
    Epoch [750/10000], loss: 0.42061 acc: 1.00000 val_loss: 0.46206, val_acc: 0.93333
    Epoch [760/10000], loss: 0.41786 acc: 1.00000 val_loss: 0.45953, val_acc: 0.93333
    Epoch [770/10000], loss: 0.41515 acc: 1.00000 val_loss: 0.45703, val_acc: 0.93333
    Epoch [780/10000], loss: 0.41247 acc: 1.00000 val_loss: 0.45457, val_acc: 0.93333
    Epoch [790/10000], loss: 0.40983 acc: 1.00000 val_loss: 0.45213, val_acc: 0.93333
    Epoch [800/10000], loss: 0.40722 acc: 1.00000 val_loss: 0.44973, val_acc: 0.93333
    Epoch [810/10000], loss: 0.40465 acc: 1.00000 val_loss: 0.44736, val_acc: 0.93333
    Epoch [820/10000], loss: 0.40211 acc: 1.00000 val_loss: 0.44502, val_acc: 0.93333
    Epoch [830/10000], loss: 0.39961 acc: 1.00000 val_loss: 0.44271, val_acc: 0.93333
    Epoch [840/10000], loss: 0.39714 acc: 1.00000 val_loss: 0.44043, val_acc: 0.93333
    Epoch [850/10000], loss: 0.39470 acc: 1.00000 val_loss: 0.43818, val_acc: 0.93333
    Epoch [860/10000], loss: 0.39229 acc: 1.00000 val_loss: 0.43596, val_acc: 0.93333
    Epoch [870/10000], loss: 0.38992 acc: 1.00000 val_loss: 0.43377, val_acc: 0.93333
    Epoch [880/10000], loss: 0.38757 acc: 1.00000 val_loss: 0.43160, val_acc: 0.93333
    Epoch [890/10000], loss: 0.38525 acc: 1.00000 val_loss: 0.42946, val_acc: 0.96667
    Epoch [900/10000], loss: 0.38297 acc: 1.00000 val_loss: 0.42735, val_acc: 0.96667
    Epoch [910/10000], loss: 0.38071 acc: 1.00000 val_loss: 0.42526, val_acc: 0.96667
    Epoch [920/10000], loss: 0.37848 acc: 1.00000 val_loss: 0.42320, val_acc: 0.96667
    Epoch [930/10000], loss: 0.37628 acc: 1.00000 val_loss: 0.42116, val_acc: 0.96667
    Epoch [940/10000], loss: 0.37410 acc: 1.00000 val_loss: 0.41915, val_acc: 0.96667
    Epoch [950/10000], loss: 0.37196 acc: 1.00000 val_loss: 0.41717, val_acc: 0.96667
    Epoch [960/10000], loss: 0.36983 acc: 1.00000 val_loss: 0.41520, val_acc: 0.96667
    Epoch [970/10000], loss: 0.36774 acc: 1.00000 val_loss: 0.41327, val_acc: 0.96667
    Epoch [980/10000], loss: 0.36567 acc: 1.00000 val_loss: 0.41135, val_acc: 0.96667
    Epoch [990/10000], loss: 0.36362 acc: 1.00000 val_loss: 0.40946, val_acc: 0.96667
    Epoch [1000/10000], loss: 0.36160 acc: 1.00000 val_loss: 0.40759, val_acc: 0.96667
    Epoch [1010/10000], loss: 0.35961 acc: 1.00000 val_loss: 0.40574, val_acc: 0.96667
    Epoch [1020/10000], loss: 0.35763 acc: 1.00000 val_loss: 0.40391, val_acc: 0.96667
    Epoch [1030/10000], loss: 0.35568 acc: 1.00000 val_loss: 0.40211, val_acc: 0.96667
    Epoch [1040/10000], loss: 0.35376 acc: 1.00000 val_loss: 0.40032, val_acc: 0.96667
    Epoch [1050/10000], loss: 0.35186 acc: 1.00000 val_loss: 0.39856, val_acc: 0.96667
    Epoch [1060/10000], loss: 0.34997 acc: 1.00000 val_loss: 0.39682, val_acc: 0.96667
    Epoch [1070/10000], loss: 0.34811 acc: 1.00000 val_loss: 0.39509, val_acc: 0.96667
    Epoch [1080/10000], loss: 0.34628 acc: 1.00000 val_loss: 0.39339, val_acc: 0.96667
    Epoch [1090/10000], loss: 0.34446 acc: 1.00000 val_loss: 0.39171, val_acc: 0.96667
    Epoch [1100/10000], loss: 0.34266 acc: 1.00000 val_loss: 0.39004, val_acc: 0.96667
    Epoch [1110/10000], loss: 0.34089 acc: 1.00000 val_loss: 0.38839, val_acc: 0.96667
    Epoch [1120/10000], loss: 0.33913 acc: 1.00000 val_loss: 0.38677, val_acc: 0.96667
    Epoch [1130/10000], loss: 0.33739 acc: 1.00000 val_loss: 0.38516, val_acc: 0.96667
    Epoch [1140/10000], loss: 0.33568 acc: 1.00000 val_loss: 0.38357, val_acc: 0.96667
    Epoch [1150/10000], loss: 0.33398 acc: 1.00000 val_loss: 0.38199, val_acc: 0.96667
    Epoch [1160/10000], loss: 0.33230 acc: 1.00000 val_loss: 0.38043, val_acc: 0.96667
    Epoch [1170/10000], loss: 0.33064 acc: 1.00000 val_loss: 0.37889, val_acc: 0.96667
    Epoch [1180/10000], loss: 0.32900 acc: 1.00000 val_loss: 0.37737, val_acc: 0.96667
    Epoch [1190/10000], loss: 0.32737 acc: 1.00000 val_loss: 0.37586, val_acc: 0.96667
    Epoch [1200/10000], loss: 0.32577 acc: 1.00000 val_loss: 0.37437, val_acc: 0.96667
    Epoch [1210/10000], loss: 0.32418 acc: 1.00000 val_loss: 0.37290, val_acc: 0.96667
    Epoch [1220/10000], loss: 0.32260 acc: 1.00000 val_loss: 0.37144, val_acc: 0.96667
    Epoch [1230/10000], loss: 0.32105 acc: 1.00000 val_loss: 0.37000, val_acc: 0.96667
    Epoch [1240/10000], loss: 0.31951 acc: 1.00000 val_loss: 0.36857, val_acc: 0.96667
    Epoch [1250/10000], loss: 0.31799 acc: 1.00000 val_loss: 0.36716, val_acc: 0.96667
    Epoch [1260/10000], loss: 0.31648 acc: 1.00000 val_loss: 0.36576, val_acc: 0.96667
    Epoch [1270/10000], loss: 0.31499 acc: 1.00000 val_loss: 0.36437, val_acc: 0.96667
    Epoch [1280/10000], loss: 0.31351 acc: 1.00000 val_loss: 0.36301, val_acc: 0.96667
    Epoch [1290/10000], loss: 0.31205 acc: 1.00000 val_loss: 0.36165, val_acc: 0.96667
    Epoch [1300/10000], loss: 0.31061 acc: 1.00000 val_loss: 0.36031, val_acc: 0.96667
    Epoch [1310/10000], loss: 0.30918 acc: 1.00000 val_loss: 0.35898, val_acc: 0.96667
    Epoch [1320/10000], loss: 0.30776 acc: 1.00000 val_loss: 0.35767, val_acc: 0.96667
    Epoch [1330/10000], loss: 0.30636 acc: 1.00000 val_loss: 0.35637, val_acc: 0.96667
    Epoch [1340/10000], loss: 0.30498 acc: 1.00000 val_loss: 0.35508, val_acc: 0.96667
    Epoch [1350/10000], loss: 0.30360 acc: 1.00000 val_loss: 0.35381, val_acc: 0.96667
    Epoch [1360/10000], loss: 0.30224 acc: 1.00000 val_loss: 0.35255, val_acc: 0.96667
    Epoch [1370/10000], loss: 0.30090 acc: 1.00000 val_loss: 0.35130, val_acc: 0.96667
    Epoch [1380/10000], loss: 0.29957 acc: 1.00000 val_loss: 0.35006, val_acc: 0.96667
    Epoch [1390/10000], loss: 0.29825 acc: 1.00000 val_loss: 0.34884, val_acc: 0.96667
    Epoch [1400/10000], loss: 0.29694 acc: 1.00000 val_loss: 0.34763, val_acc: 0.96667
    Epoch [1410/10000], loss: 0.29565 acc: 1.00000 val_loss: 0.34643, val_acc: 0.96667
    Epoch [1420/10000], loss: 0.29437 acc: 1.00000 val_loss: 0.34524, val_acc: 0.96667
    Epoch [1430/10000], loss: 0.29310 acc: 1.00000 val_loss: 0.34406, val_acc: 0.96667
    Epoch [1440/10000], loss: 0.29184 acc: 1.00000 val_loss: 0.34290, val_acc: 0.96667
    Epoch [1450/10000], loss: 0.29060 acc: 1.00000 val_loss: 0.34174, val_acc: 0.96667
    Epoch [1460/10000], loss: 0.28937 acc: 1.00000 val_loss: 0.34060, val_acc: 0.96667
    Epoch [1470/10000], loss: 0.28815 acc: 1.00000 val_loss: 0.33947, val_acc: 0.96667
    Epoch [1480/10000], loss: 0.28694 acc: 1.00000 val_loss: 0.33834, val_acc: 0.96667
    Epoch [1490/10000], loss: 0.28574 acc: 1.00000 val_loss: 0.33723, val_acc: 0.96667
    Epoch [1500/10000], loss: 0.28456 acc: 1.00000 val_loss: 0.33613, val_acc: 0.96667
    Epoch [1510/10000], loss: 0.28338 acc: 1.00000 val_loss: 0.33504, val_acc: 0.96667
    Epoch [1520/10000], loss: 0.28222 acc: 1.00000 val_loss: 0.33396, val_acc: 0.96667
    Epoch [1530/10000], loss: 0.28106 acc: 1.00000 val_loss: 0.33289, val_acc: 0.96667
    Epoch [1540/10000], loss: 0.27992 acc: 1.00000 val_loss: 0.33183, val_acc: 0.96667
    Epoch [1550/10000], loss: 0.27879 acc: 1.00000 val_loss: 0.33078, val_acc: 0.96667
    Epoch [1560/10000], loss: 0.27767 acc: 1.00000 val_loss: 0.32974, val_acc: 0.96667
    Epoch [1570/10000], loss: 0.27656 acc: 1.00000 val_loss: 0.32871, val_acc: 0.96667
    Epoch [1580/10000], loss: 0.27545 acc: 1.00000 val_loss: 0.32769, val_acc: 0.96667
    Epoch [1590/10000], loss: 0.27436 acc: 1.00000 val_loss: 0.32668, val_acc: 0.96667
    Epoch [1600/10000], loss: 0.27328 acc: 1.00000 val_loss: 0.32568, val_acc: 0.96667
    Epoch [1610/10000], loss: 0.27221 acc: 1.00000 val_loss: 0.32468, val_acc: 0.96667
    Epoch [1620/10000], loss: 0.27115 acc: 1.00000 val_loss: 0.32370, val_acc: 0.96667
    Epoch [1630/10000], loss: 0.27009 acc: 1.00000 val_loss: 0.32272, val_acc: 0.96667
    Epoch [1640/10000], loss: 0.26905 acc: 1.00000 val_loss: 0.32175, val_acc: 0.96667
    Epoch [1650/10000], loss: 0.26802 acc: 1.00000 val_loss: 0.32080, val_acc: 0.96667
    Epoch [1660/10000], loss: 0.26699 acc: 1.00000 val_loss: 0.31985, val_acc: 0.96667
    Epoch [1670/10000], loss: 0.26597 acc: 1.00000 val_loss: 0.31890, val_acc: 0.96667
    Epoch [1680/10000], loss: 0.26497 acc: 1.00000 val_loss: 0.31797, val_acc: 0.96667
    Epoch [1690/10000], loss: 0.26397 acc: 1.00000 val_loss: 0.31704, val_acc: 0.96667
    Epoch [1700/10000], loss: 0.26298 acc: 1.00000 val_loss: 0.31613, val_acc: 0.96667
    Epoch [1710/10000], loss: 0.26199 acc: 1.00000 val_loss: 0.31522, val_acc: 0.96667
    Epoch [1720/10000], loss: 0.26102 acc: 1.00000 val_loss: 0.31431, val_acc: 0.96667
    Epoch [1730/10000], loss: 0.26005 acc: 1.00000 val_loss: 0.31342, val_acc: 0.96667
    Epoch [1740/10000], loss: 0.25910 acc: 1.00000 val_loss: 0.31253, val_acc: 0.96667
    Epoch [1750/10000], loss: 0.25815 acc: 1.00000 val_loss: 0.31165, val_acc: 0.96667
    Epoch [1760/10000], loss: 0.25721 acc: 1.00000 val_loss: 0.31078, val_acc: 0.96667
    Epoch [1770/10000], loss: 0.25627 acc: 1.00000 val_loss: 0.30992, val_acc: 0.96667
    Epoch [1780/10000], loss: 0.25535 acc: 1.00000 val_loss: 0.30906, val_acc: 0.96667
    Epoch [1790/10000], loss: 0.25443 acc: 1.00000 val_loss: 0.30821, val_acc: 0.96667
    Epoch [1800/10000], loss: 0.25352 acc: 1.00000 val_loss: 0.30737, val_acc: 0.96667
    Epoch [1810/10000], loss: 0.25262 acc: 1.00000 val_loss: 0.30653, val_acc: 0.96667
    Epoch [1820/10000], loss: 0.25172 acc: 1.00000 val_loss: 0.30571, val_acc: 0.96667
    Epoch [1830/10000], loss: 0.25083 acc: 1.00000 val_loss: 0.30488, val_acc: 0.96667
    Epoch [1840/10000], loss: 0.24995 acc: 1.00000 val_loss: 0.30407, val_acc: 0.96667
    Epoch [1850/10000], loss: 0.24908 acc: 1.00000 val_loss: 0.30326, val_acc: 0.96667
    Epoch [1860/10000], loss: 0.24821 acc: 1.00000 val_loss: 0.30246, val_acc: 0.96667
    Epoch [1870/10000], loss: 0.24735 acc: 1.00000 val_loss: 0.30166, val_acc: 0.96667
    Epoch [1880/10000], loss: 0.24650 acc: 1.00000 val_loss: 0.30088, val_acc: 0.96667
    Epoch [1890/10000], loss: 0.24565 acc: 1.00000 val_loss: 0.30009, val_acc: 0.96667
    Epoch [1900/10000], loss: 0.24481 acc: 1.00000 val_loss: 0.29932, val_acc: 0.96667
    Epoch [1910/10000], loss: 0.24398 acc: 1.00000 val_loss: 0.29855, val_acc: 0.96667
    Epoch [1920/10000], loss: 0.24315 acc: 1.00000 val_loss: 0.29778, val_acc: 0.96667
    Epoch [1930/10000], loss: 0.24233 acc: 1.00000 val_loss: 0.29702, val_acc: 0.96667
    Epoch [1940/10000], loss: 0.24152 acc: 1.00000 val_loss: 0.29627, val_acc: 0.96667
    Epoch [1950/10000], loss: 0.24071 acc: 1.00000 val_loss: 0.29553, val_acc: 0.96667
    Epoch [1960/10000], loss: 0.23991 acc: 1.00000 val_loss: 0.29479, val_acc: 0.96667
    Epoch [1970/10000], loss: 0.23911 acc: 1.00000 val_loss: 0.29405, val_acc: 0.96667
    Epoch [1980/10000], loss: 0.23833 acc: 1.00000 val_loss: 0.29332, val_acc: 0.96667
    Epoch [1990/10000], loss: 0.23754 acc: 1.00000 val_loss: 0.29260, val_acc: 0.96667
    Epoch [2000/10000], loss: 0.23677 acc: 1.00000 val_loss: 0.29188, val_acc: 0.96667
    Epoch [2010/10000], loss: 0.23599 acc: 1.00000 val_loss: 0.29117, val_acc: 0.96667
    Epoch [2020/10000], loss: 0.23523 acc: 1.00000 val_loss: 0.29047, val_acc: 0.96667
    Epoch [2030/10000], loss: 0.23447 acc: 1.00000 val_loss: 0.28977, val_acc: 0.96667
    Epoch [2040/10000], loss: 0.23372 acc: 1.00000 val_loss: 0.28907, val_acc: 0.96667
    Epoch [2050/10000], loss: 0.23297 acc: 1.00000 val_loss: 0.28838, val_acc: 0.96667
    Epoch [2060/10000], loss: 0.23223 acc: 1.00000 val_loss: 0.28770, val_acc: 0.96667
    Epoch [2070/10000], loss: 0.23149 acc: 1.00000 val_loss: 0.28702, val_acc: 0.96667
    Epoch [2080/10000], loss: 0.23076 acc: 1.00000 val_loss: 0.28634, val_acc: 0.96667
    Epoch [2090/10000], loss: 0.23003 acc: 1.00000 val_loss: 0.28567, val_acc: 0.96667
    Epoch [2100/10000], loss: 0.22931 acc: 1.00000 val_loss: 0.28501, val_acc: 0.96667
    Epoch [2110/10000], loss: 0.22859 acc: 1.00000 val_loss: 0.28435, val_acc: 0.96667
    Epoch [2120/10000], loss: 0.22788 acc: 1.00000 val_loss: 0.28369, val_acc: 0.96667
    Epoch [2130/10000], loss: 0.22718 acc: 1.00000 val_loss: 0.28304, val_acc: 0.96667
    Epoch [2140/10000], loss: 0.22648 acc: 1.00000 val_loss: 0.28240, val_acc: 0.96667
    Epoch [2150/10000], loss: 0.22578 acc: 1.00000 val_loss: 0.28176, val_acc: 0.96667
    Epoch [2160/10000], loss: 0.22509 acc: 1.00000 val_loss: 0.28112, val_acc: 0.96667
    Epoch [2170/10000], loss: 0.22441 acc: 1.00000 val_loss: 0.28049, val_acc: 0.96667
    Epoch [2180/10000], loss: 0.22373 acc: 1.00000 val_loss: 0.27986, val_acc: 0.96667
    Epoch [2190/10000], loss: 0.22305 acc: 1.00000 val_loss: 0.27924, val_acc: 0.96667
    Epoch [2200/10000], loss: 0.22238 acc: 1.00000 val_loss: 0.27862, val_acc: 0.96667
    Epoch [2210/10000], loss: 0.22171 acc: 1.00000 val_loss: 0.27801, val_acc: 0.96667
    Epoch [2220/10000], loss: 0.22105 acc: 1.00000 val_loss: 0.27740, val_acc: 0.96667
    Epoch [2230/10000], loss: 0.22039 acc: 1.00000 val_loss: 0.27680, val_acc: 0.96667
    Epoch [2240/10000], loss: 0.21974 acc: 1.00000 val_loss: 0.27620, val_acc: 0.96667
    Epoch [2250/10000], loss: 0.21909 acc: 1.00000 val_loss: 0.27560, val_acc: 0.96667
    Epoch [2260/10000], loss: 0.21845 acc: 1.00000 val_loss: 0.27501, val_acc: 0.96667
    Epoch [2270/10000], loss: 0.21781 acc: 1.00000 val_loss: 0.27442, val_acc: 0.96667
    Epoch [2280/10000], loss: 0.21718 acc: 1.00000 val_loss: 0.27384, val_acc: 0.96667
    Epoch [2290/10000], loss: 0.21655 acc: 1.00000 val_loss: 0.27326, val_acc: 0.96667
    Epoch [2300/10000], loss: 0.21592 acc: 1.00000 val_loss: 0.27269, val_acc: 0.96667
    Epoch [2310/10000], loss: 0.21530 acc: 1.00000 val_loss: 0.27211, val_acc: 0.96667
    Epoch [2320/10000], loss: 0.21468 acc: 1.00000 val_loss: 0.27155, val_acc: 0.96667
    Epoch [2330/10000], loss: 0.21407 acc: 1.00000 val_loss: 0.27098, val_acc: 0.96667
    Epoch [2340/10000], loss: 0.21346 acc: 1.00000 val_loss: 0.27042, val_acc: 0.96667
    Epoch [2350/10000], loss: 0.21285 acc: 1.00000 val_loss: 0.26987, val_acc: 0.96667
    Epoch [2360/10000], loss: 0.21225 acc: 1.00000 val_loss: 0.26932, val_acc: 0.96667
    Epoch [2370/10000], loss: 0.21165 acc: 1.00000 val_loss: 0.26877, val_acc: 0.96667
    Epoch [2380/10000], loss: 0.21106 acc: 1.00000 val_loss: 0.26822, val_acc: 0.96667
    Epoch [2390/10000], loss: 0.21047 acc: 1.00000 val_loss: 0.26768, val_acc: 0.96667
    Epoch [2400/10000], loss: 0.20988 acc: 1.00000 val_loss: 0.26715, val_acc: 0.96667
    Epoch [2410/10000], loss: 0.20930 acc: 1.00000 val_loss: 0.26661, val_acc: 0.96667
    Epoch [2420/10000], loss: 0.20872 acc: 1.00000 val_loss: 0.26608, val_acc: 0.96667
    Epoch [2430/10000], loss: 0.20815 acc: 1.00000 val_loss: 0.26556, val_acc: 0.96667
    Epoch [2440/10000], loss: 0.20758 acc: 1.00000 val_loss: 0.26503, val_acc: 0.96667
    Epoch [2450/10000], loss: 0.20701 acc: 1.00000 val_loss: 0.26451, val_acc: 0.96667
    Epoch [2460/10000], loss: 0.20645 acc: 1.00000 val_loss: 0.26400, val_acc: 0.96667
    Epoch [2470/10000], loss: 0.20589 acc: 1.00000 val_loss: 0.26349, val_acc: 0.96667
    Epoch [2480/10000], loss: 0.20533 acc: 1.00000 val_loss: 0.26298, val_acc: 0.96667
    Epoch [2490/10000], loss: 0.20478 acc: 1.00000 val_loss: 0.26247, val_acc: 0.96667
    Epoch [2500/10000], loss: 0.20423 acc: 1.00000 val_loss: 0.26197, val_acc: 0.96667
    Epoch [2510/10000], loss: 0.20369 acc: 1.00000 val_loss: 0.26147, val_acc: 0.96667
    Epoch [2520/10000], loss: 0.20314 acc: 1.00000 val_loss: 0.26097, val_acc: 0.96667
    Epoch [2530/10000], loss: 0.20261 acc: 1.00000 val_loss: 0.26048, val_acc: 0.96667
    Epoch [2540/10000], loss: 0.20207 acc: 1.00000 val_loss: 0.25999, val_acc: 0.96667
    Epoch [2550/10000], loss: 0.20154 acc: 1.00000 val_loss: 0.25950, val_acc: 0.96667
    Epoch [2560/10000], loss: 0.20101 acc: 1.00000 val_loss: 0.25902, val_acc: 0.96667
    Epoch [2570/10000], loss: 0.20048 acc: 1.00000 val_loss: 0.25854, val_acc: 0.96667
    Epoch [2580/10000], loss: 0.19996 acc: 1.00000 val_loss: 0.25806, val_acc: 0.96667
    Epoch [2590/10000], loss: 0.19944 acc: 1.00000 val_loss: 0.25759, val_acc: 0.96667
    Epoch [2600/10000], loss: 0.19893 acc: 1.00000 val_loss: 0.25712, val_acc: 0.96667
    Epoch [2610/10000], loss: 0.19841 acc: 1.00000 val_loss: 0.25665, val_acc: 0.96667
    Epoch [2620/10000], loss: 0.19791 acc: 1.00000 val_loss: 0.25618, val_acc: 0.96667
    Epoch [2630/10000], loss: 0.19740 acc: 1.00000 val_loss: 0.25572, val_acc: 0.96667
    Epoch [2640/10000], loss: 0.19690 acc: 1.00000 val_loss: 0.25526, val_acc: 0.96667
    Epoch [2650/10000], loss: 0.19640 acc: 1.00000 val_loss: 0.25481, val_acc: 0.96667
    Epoch [2660/10000], loss: 0.19590 acc: 1.00000 val_loss: 0.25435, val_acc: 0.96667
    Epoch [2670/10000], loss: 0.19540 acc: 1.00000 val_loss: 0.25390, val_acc: 0.96667
    Epoch [2680/10000], loss: 0.19491 acc: 1.00000 val_loss: 0.25345, val_acc: 0.96667
    Epoch [2690/10000], loss: 0.19442 acc: 1.00000 val_loss: 0.25301, val_acc: 0.96667
    Epoch [2700/10000], loss: 0.19394 acc: 1.00000 val_loss: 0.25257, val_acc: 0.96667
    Epoch [2710/10000], loss: 0.19346 acc: 1.00000 val_loss: 0.25213, val_acc: 0.96667
    Epoch [2720/10000], loss: 0.19298 acc: 1.00000 val_loss: 0.25169, val_acc: 0.96667
    Epoch [2730/10000], loss: 0.19250 acc: 1.00000 val_loss: 0.25125, val_acc: 0.96667
    Epoch [2740/10000], loss: 0.19202 acc: 1.00000 val_loss: 0.25082, val_acc: 0.96667
    Epoch [2750/10000], loss: 0.19155 acc: 1.00000 val_loss: 0.25039, val_acc: 0.96667
    Epoch [2760/10000], loss: 0.19108 acc: 1.00000 val_loss: 0.24997, val_acc: 0.96667
    Epoch [2770/10000], loss: 0.19062 acc: 1.00000 val_loss: 0.24954, val_acc: 0.96667
    Epoch [2780/10000], loss: 0.19016 acc: 1.00000 val_loss: 0.24912, val_acc: 0.96667
    Epoch [2790/10000], loss: 0.18970 acc: 1.00000 val_loss: 0.24870, val_acc: 0.96667
    Epoch [2800/10000], loss: 0.18924 acc: 1.00000 val_loss: 0.24828, val_acc: 0.96667
    Epoch [2810/10000], loss: 0.18878 acc: 1.00000 val_loss: 0.24787, val_acc: 0.96667
    Epoch [2820/10000], loss: 0.18833 acc: 1.00000 val_loss: 0.24746, val_acc: 0.96667
    Epoch [2830/10000], loss: 0.18788 acc: 1.00000 val_loss: 0.24705, val_acc: 0.96667
    Epoch [2840/10000], loss: 0.18743 acc: 1.00000 val_loss: 0.24664, val_acc: 0.96667
    Epoch [2850/10000], loss: 0.18699 acc: 1.00000 val_loss: 0.24624, val_acc: 0.96667
    Epoch [2860/10000], loss: 0.18654 acc: 1.00000 val_loss: 0.24584, val_acc: 0.96667
    Epoch [2870/10000], loss: 0.18610 acc: 1.00000 val_loss: 0.24544, val_acc: 0.96667
    Epoch [2880/10000], loss: 0.18567 acc: 1.00000 val_loss: 0.24504, val_acc: 0.96667
    Epoch [2890/10000], loss: 0.18523 acc: 1.00000 val_loss: 0.24464, val_acc: 0.96667
    Epoch [2900/10000], loss: 0.18480 acc: 1.00000 val_loss: 0.24425, val_acc: 0.96667
    Epoch [2910/10000], loss: 0.18437 acc: 1.00000 val_loss: 0.24386, val_acc: 0.96667
    Epoch [2920/10000], loss: 0.18394 acc: 1.00000 val_loss: 0.24347, val_acc: 0.96667
    Epoch [2930/10000], loss: 0.18352 acc: 1.00000 val_loss: 0.24309, val_acc: 0.96667
    Epoch [2940/10000], loss: 0.18309 acc: 1.00000 val_loss: 0.24270, val_acc: 0.96667
    Epoch [2950/10000], loss: 0.18267 acc: 1.00000 val_loss: 0.24232, val_acc: 0.96667
    Epoch [2960/10000], loss: 0.18225 acc: 1.00000 val_loss: 0.24194, val_acc: 0.96667
    Epoch [2970/10000], loss: 0.18184 acc: 1.00000 val_loss: 0.24156, val_acc: 0.96667
    Epoch [2980/10000], loss: 0.18142 acc: 1.00000 val_loss: 0.24119, val_acc: 0.96667
    Epoch [2990/10000], loss: 0.18101 acc: 1.00000 val_loss: 0.24081, val_acc: 0.96667
    Epoch [3000/10000], loss: 0.18060 acc: 1.00000 val_loss: 0.24044, val_acc: 0.96667
    Epoch [3010/10000], loss: 0.18019 acc: 1.00000 val_loss: 0.24008, val_acc: 0.96667
    Epoch [3020/10000], loss: 0.17979 acc: 1.00000 val_loss: 0.23971, val_acc: 0.96667
    Epoch [3030/10000], loss: 0.17939 acc: 1.00000 val_loss: 0.23934, val_acc: 0.96667
    Epoch [3040/10000], loss: 0.17899 acc: 1.00000 val_loss: 0.23898, val_acc: 0.96667
    Epoch [3050/10000], loss: 0.17859 acc: 1.00000 val_loss: 0.23862, val_acc: 0.96667
    Epoch [3060/10000], loss: 0.17819 acc: 1.00000 val_loss: 0.23826, val_acc: 0.96667
    Epoch [3070/10000], loss: 0.17780 acc: 1.00000 val_loss: 0.23790, val_acc: 0.96667
    Epoch [3080/10000], loss: 0.17740 acc: 1.00000 val_loss: 0.23755, val_acc: 0.96667
    Epoch [3090/10000], loss: 0.17701 acc: 1.00000 val_loss: 0.23720, val_acc: 0.96667
    Epoch [3100/10000], loss: 0.17662 acc: 1.00000 val_loss: 0.23685, val_acc: 0.96667
    Epoch [3110/10000], loss: 0.17624 acc: 1.00000 val_loss: 0.23650, val_acc: 0.96667
    Epoch [3120/10000], loss: 0.17585 acc: 1.00000 val_loss: 0.23615, val_acc: 0.96667
    Epoch [3130/10000], loss: 0.17547 acc: 1.00000 val_loss: 0.23580, val_acc: 0.96667
    Epoch [3140/10000], loss: 0.17509 acc: 1.00000 val_loss: 0.23546, val_acc: 0.96667
    Epoch [3150/10000], loss: 0.17471 acc: 1.00000 val_loss: 0.23512, val_acc: 0.96667
    Epoch [3160/10000], loss: 0.17434 acc: 1.00000 val_loss: 0.23478, val_acc: 0.96667
    Epoch [3170/10000], loss: 0.17396 acc: 1.00000 val_loss: 0.23444, val_acc: 0.96667
    Epoch [3180/10000], loss: 0.17359 acc: 1.00000 val_loss: 0.23411, val_acc: 0.96667
    Epoch [3190/10000], loss: 0.17322 acc: 1.00000 val_loss: 0.23377, val_acc: 0.96667
    Epoch [3200/10000], loss: 0.17285 acc: 1.00000 val_loss: 0.23344, val_acc: 0.96667
    Epoch [3210/10000], loss: 0.17249 acc: 1.00000 val_loss: 0.23311, val_acc: 0.96667
    Epoch [3220/10000], loss: 0.17212 acc: 1.00000 val_loss: 0.23278, val_acc: 0.96667
    Epoch [3230/10000], loss: 0.17176 acc: 1.00000 val_loss: 0.23245, val_acc: 0.96667
    Epoch [3240/10000], loss: 0.17140 acc: 1.00000 val_loss: 0.23213, val_acc: 0.96667
    Epoch [3250/10000], loss: 0.17104 acc: 1.00000 val_loss: 0.23180, val_acc: 0.96667
    Epoch [3260/10000], loss: 0.17068 acc: 1.00000 val_loss: 0.23148, val_acc: 0.96667
    Epoch [3270/10000], loss: 0.17032 acc: 1.00000 val_loss: 0.23116, val_acc: 0.96667
    Epoch [3280/10000], loss: 0.16997 acc: 1.00000 val_loss: 0.23084, val_acc: 0.96667
    Epoch [3290/10000], loss: 0.16962 acc: 1.00000 val_loss: 0.23053, val_acc: 0.96667
    Epoch [3300/10000], loss: 0.16927 acc: 1.00000 val_loss: 0.23021, val_acc: 0.96667
    Epoch [3310/10000], loss: 0.16892 acc: 1.00000 val_loss: 0.22990, val_acc: 0.96667
    Epoch [3320/10000], loss: 0.16857 acc: 1.00000 val_loss: 0.22959, val_acc: 0.96667
    Epoch [3330/10000], loss: 0.16823 acc: 1.00000 val_loss: 0.22928, val_acc: 0.96667
    Epoch [3340/10000], loss: 0.16788 acc: 1.00000 val_loss: 0.22897, val_acc: 0.96667
    Epoch [3350/10000], loss: 0.16754 acc: 1.00000 val_loss: 0.22866, val_acc: 0.96667
    Epoch [3360/10000], loss: 0.16720 acc: 1.00000 val_loss: 0.22835, val_acc: 0.96667
    Epoch [3370/10000], loss: 0.16686 acc: 1.00000 val_loss: 0.22805, val_acc: 0.96667
    Epoch [3380/10000], loss: 0.16653 acc: 1.00000 val_loss: 0.22775, val_acc: 0.96667
    Epoch [3390/10000], loss: 0.16619 acc: 1.00000 val_loss: 0.22745, val_acc: 0.96667
    Epoch [3400/10000], loss: 0.16586 acc: 1.00000 val_loss: 0.22715, val_acc: 0.96667
    Epoch [3410/10000], loss: 0.16553 acc: 1.00000 val_loss: 0.22685, val_acc: 0.96667
    Epoch [3420/10000], loss: 0.16520 acc: 1.00000 val_loss: 0.22655, val_acc: 0.96667
    Epoch [3430/10000], loss: 0.16487 acc: 1.00000 val_loss: 0.22626, val_acc: 0.96667
    Epoch [3440/10000], loss: 0.16454 acc: 1.00000 val_loss: 0.22596, val_acc: 0.96667
    Epoch [3450/10000], loss: 0.16421 acc: 1.00000 val_loss: 0.22567, val_acc: 0.96667
    Epoch [3460/10000], loss: 0.16389 acc: 1.00000 val_loss: 0.22538, val_acc: 0.96667
    Epoch [3470/10000], loss: 0.16357 acc: 1.00000 val_loss: 0.22509, val_acc: 0.96667
    Epoch [3480/10000], loss: 0.16325 acc: 1.00000 val_loss: 0.22480, val_acc: 0.96667
    Epoch [3490/10000], loss: 0.16293 acc: 1.00000 val_loss: 0.22452, val_acc: 0.96667
    Epoch [3500/10000], loss: 0.16261 acc: 1.00000 val_loss: 0.22423, val_acc: 0.96667
    Epoch [3510/10000], loss: 0.16229 acc: 1.00000 val_loss: 0.22395, val_acc: 0.96667
    Epoch [3520/10000], loss: 0.16198 acc: 1.00000 val_loss: 0.22367, val_acc: 0.96667
    Epoch [3530/10000], loss: 0.16166 acc: 1.00000 val_loss: 0.22339, val_acc: 0.96667
    Epoch [3540/10000], loss: 0.16135 acc: 1.00000 val_loss: 0.22311, val_acc: 0.96667
    Epoch [3550/10000], loss: 0.16104 acc: 1.00000 val_loss: 0.22283, val_acc: 0.96667
    Epoch [3560/10000], loss: 0.16073 acc: 1.00000 val_loss: 0.22255, val_acc: 0.96667
    Epoch [3570/10000], loss: 0.16043 acc: 1.00000 val_loss: 0.22228, val_acc: 0.96667
    Epoch [3580/10000], loss: 0.16012 acc: 1.00000 val_loss: 0.22200, val_acc: 0.96667
    Epoch [3590/10000], loss: 0.15981 acc: 1.00000 val_loss: 0.22173, val_acc: 0.96667
    Epoch [3600/10000], loss: 0.15951 acc: 1.00000 val_loss: 0.22146, val_acc: 0.96667
    Epoch [3610/10000], loss: 0.15921 acc: 1.00000 val_loss: 0.22119, val_acc: 0.96667
    Epoch [3620/10000], loss: 0.15891 acc: 1.00000 val_loss: 0.22092, val_acc: 0.96667
    Epoch [3630/10000], loss: 0.15861 acc: 1.00000 val_loss: 0.22065, val_acc: 0.96667
    Epoch [3640/10000], loss: 0.15831 acc: 1.00000 val_loss: 0.22039, val_acc: 0.96667
    Epoch [3650/10000], loss: 0.15801 acc: 1.00000 val_loss: 0.22012, val_acc: 0.96667
    Epoch [3660/10000], loss: 0.15772 acc: 1.00000 val_loss: 0.21986, val_acc: 0.96667
    Epoch [3670/10000], loss: 0.15743 acc: 1.00000 val_loss: 0.21960, val_acc: 0.96667
    Epoch [3680/10000], loss: 0.15713 acc: 1.00000 val_loss: 0.21934, val_acc: 0.96667
    Epoch [3690/10000], loss: 0.15684 acc: 1.00000 val_loss: 0.21908, val_acc: 0.96667
    Epoch [3700/10000], loss: 0.15655 acc: 1.00000 val_loss: 0.21882, val_acc: 0.96667
    Epoch [3710/10000], loss: 0.15626 acc: 1.00000 val_loss: 0.21856, val_acc: 0.96667
    Epoch [3720/10000], loss: 0.15598 acc: 1.00000 val_loss: 0.21830, val_acc: 0.96667
    Epoch [3730/10000], loss: 0.15569 acc: 1.00000 val_loss: 0.21805, val_acc: 0.96667
    Epoch [3740/10000], loss: 0.15540 acc: 1.00000 val_loss: 0.21780, val_acc: 0.96667
    Epoch [3750/10000], loss: 0.15512 acc: 1.00000 val_loss: 0.21754, val_acc: 0.96667
    Epoch [3760/10000], loss: 0.15484 acc: 1.00000 val_loss: 0.21729, val_acc: 0.96667
    Epoch [3770/10000], loss: 0.15456 acc: 1.00000 val_loss: 0.21704, val_acc: 0.96667
    Epoch [3780/10000], loss: 0.15428 acc: 1.00000 val_loss: 0.21679, val_acc: 0.96667
    Epoch [3790/10000], loss: 0.15400 acc: 1.00000 val_loss: 0.21655, val_acc: 0.96667
    Epoch [3800/10000], loss: 0.15372 acc: 1.00000 val_loss: 0.21630, val_acc: 0.96667
    Epoch [3810/10000], loss: 0.15345 acc: 1.00000 val_loss: 0.21605, val_acc: 0.96667
    Epoch [3820/10000], loss: 0.15317 acc: 1.00000 val_loss: 0.21581, val_acc: 0.96667
    Epoch [3830/10000], loss: 0.15290 acc: 1.00000 val_loss: 0.21557, val_acc: 0.96667
    Epoch [3840/10000], loss: 0.15262 acc: 1.00000 val_loss: 0.21532, val_acc: 0.96667
    Epoch [3850/10000], loss: 0.15235 acc: 1.00000 val_loss: 0.21508, val_acc: 0.96667
    Epoch [3860/10000], loss: 0.15208 acc: 1.00000 val_loss: 0.21484, val_acc: 0.96667
    Epoch [3870/10000], loss: 0.15181 acc: 1.00000 val_loss: 0.21460, val_acc: 0.96667
    Epoch [3880/10000], loss: 0.15155 acc: 1.00000 val_loss: 0.21436, val_acc: 0.96667
    Epoch [3890/10000], loss: 0.15128 acc: 1.00000 val_loss: 0.21413, val_acc: 0.96667
    Epoch [3900/10000], loss: 0.15101 acc: 1.00000 val_loss: 0.21389, val_acc: 0.96667
    Epoch [3910/10000], loss: 0.15075 acc: 1.00000 val_loss: 0.21366, val_acc: 0.96667
    Epoch [3920/10000], loss: 0.15049 acc: 1.00000 val_loss: 0.21342, val_acc: 0.96667
    Epoch [3930/10000], loss: 0.15022 acc: 1.00000 val_loss: 0.21319, val_acc: 0.96667
    Epoch [3940/10000], loss: 0.14996 acc: 1.00000 val_loss: 0.21296, val_acc: 0.96667
    Epoch [3950/10000], loss: 0.14970 acc: 1.00000 val_loss: 0.21273, val_acc: 0.96667
    Epoch [3960/10000], loss: 0.14944 acc: 1.00000 val_loss: 0.21250, val_acc: 0.96667
    Epoch [3970/10000], loss: 0.14919 acc: 1.00000 val_loss: 0.21227, val_acc: 0.96667
    Epoch [3980/10000], loss: 0.14893 acc: 1.00000 val_loss: 0.21204, val_acc: 0.96667
    Epoch [3990/10000], loss: 0.14867 acc: 1.00000 val_loss: 0.21182, val_acc: 0.96667
    Epoch [4000/10000], loss: 0.14842 acc: 1.00000 val_loss: 0.21159, val_acc: 0.96667
    Epoch [4010/10000], loss: 0.14816 acc: 1.00000 val_loss: 0.21137, val_acc: 0.96667
    Epoch [4020/10000], loss: 0.14791 acc: 1.00000 val_loss: 0.21114, val_acc: 0.96667
    Epoch [4030/10000], loss: 0.14766 acc: 1.00000 val_loss: 0.21092, val_acc: 0.96667
    Epoch [4040/10000], loss: 0.14741 acc: 1.00000 val_loss: 0.21070, val_acc: 0.96667
    Epoch [4050/10000], loss: 0.14716 acc: 1.00000 val_loss: 0.21048, val_acc: 0.96667
    Epoch [4060/10000], loss: 0.14691 acc: 1.00000 val_loss: 0.21026, val_acc: 0.96667
    Epoch [4070/10000], loss: 0.14667 acc: 1.00000 val_loss: 0.21004, val_acc: 0.96667
    Epoch [4080/10000], loss: 0.14642 acc: 1.00000 val_loss: 0.20982, val_acc: 0.96667
    Epoch [4090/10000], loss: 0.14617 acc: 1.00000 val_loss: 0.20961, val_acc: 0.96667
    Epoch [4100/10000], loss: 0.14593 acc: 1.00000 val_loss: 0.20939, val_acc: 0.96667
    Epoch [4110/10000], loss: 0.14569 acc: 1.00000 val_loss: 0.20918, val_acc: 0.96667
    Epoch [4120/10000], loss: 0.14544 acc: 1.00000 val_loss: 0.20896, val_acc: 0.96667
    Epoch [4130/10000], loss: 0.14520 acc: 1.00000 val_loss: 0.20875, val_acc: 0.96667
    Epoch [4140/10000], loss: 0.14496 acc: 1.00000 val_loss: 0.20854, val_acc: 0.96667
    Epoch [4150/10000], loss: 0.14472 acc: 1.00000 val_loss: 0.20833, val_acc: 0.96667
    Epoch [4160/10000], loss: 0.14448 acc: 1.00000 val_loss: 0.20812, val_acc: 0.96667
    Epoch [4170/10000], loss: 0.14425 acc: 1.00000 val_loss: 0.20791, val_acc: 0.96667
    Epoch [4180/10000], loss: 0.14401 acc: 1.00000 val_loss: 0.20770, val_acc: 0.96667
    Epoch [4190/10000], loss: 0.14377 acc: 1.00000 val_loss: 0.20749, val_acc: 0.96667
    Epoch [4200/10000], loss: 0.14354 acc: 1.00000 val_loss: 0.20728, val_acc: 0.96667
    Epoch [4210/10000], loss: 0.14331 acc: 1.00000 val_loss: 0.20708, val_acc: 0.96667
    Epoch [4220/10000], loss: 0.14307 acc: 1.00000 val_loss: 0.20687, val_acc: 0.96667
    Epoch [4230/10000], loss: 0.14284 acc: 1.00000 val_loss: 0.20667, val_acc: 0.96667
    Epoch [4240/10000], loss: 0.14261 acc: 1.00000 val_loss: 0.20647, val_acc: 0.96667
    Epoch [4250/10000], loss: 0.14238 acc: 1.00000 val_loss: 0.20626, val_acc: 0.96667
    Epoch [4260/10000], loss: 0.14215 acc: 1.00000 val_loss: 0.20606, val_acc: 0.96667
    Epoch [4270/10000], loss: 0.14192 acc: 1.00000 val_loss: 0.20586, val_acc: 0.96667
    Epoch [4280/10000], loss: 0.14170 acc: 1.00000 val_loss: 0.20566, val_acc: 0.96667
    Epoch [4290/10000], loss: 0.14147 acc: 1.00000 val_loss: 0.20546, val_acc: 0.96667
    Epoch [4300/10000], loss: 0.14124 acc: 1.00000 val_loss: 0.20526, val_acc: 0.96667
    Epoch [4310/10000], loss: 0.14102 acc: 1.00000 val_loss: 0.20507, val_acc: 0.96667
    Epoch [4320/10000], loss: 0.14080 acc: 1.00000 val_loss: 0.20487, val_acc: 0.96667
    Epoch [4330/10000], loss: 0.14057 acc: 1.00000 val_loss: 0.20467, val_acc: 0.96667
    Epoch [4340/10000], loss: 0.14035 acc: 1.00000 val_loss: 0.20448, val_acc: 0.96667
    Epoch [4350/10000], loss: 0.14013 acc: 1.00000 val_loss: 0.20429, val_acc: 0.96667
    Epoch [4360/10000], loss: 0.13991 acc: 1.00000 val_loss: 0.20409, val_acc: 0.96667
    Epoch [4370/10000], loss: 0.13969 acc: 1.00000 val_loss: 0.20390, val_acc: 0.96667
    Epoch [4380/10000], loss: 0.13947 acc: 1.00000 val_loss: 0.20371, val_acc: 0.96667
    Epoch [4390/10000], loss: 0.13925 acc: 1.00000 val_loss: 0.20352, val_acc: 0.96667
    Epoch [4400/10000], loss: 0.13904 acc: 1.00000 val_loss: 0.20333, val_acc: 0.96667
    Epoch [4410/10000], loss: 0.13882 acc: 1.00000 val_loss: 0.20314, val_acc: 0.96667
    Epoch [4420/10000], loss: 0.13860 acc: 1.00000 val_loss: 0.20295, val_acc: 0.96667
    Epoch [4430/10000], loss: 0.13839 acc: 1.00000 val_loss: 0.20276, val_acc: 0.96667
    Epoch [4440/10000], loss: 0.13818 acc: 1.00000 val_loss: 0.20257, val_acc: 0.96667
    Epoch [4450/10000], loss: 0.13796 acc: 1.00000 val_loss: 0.20239, val_acc: 0.96667
    Epoch [4460/10000], loss: 0.13775 acc: 1.00000 val_loss: 0.20220, val_acc: 0.96667
    Epoch [4470/10000], loss: 0.13754 acc: 1.00000 val_loss: 0.20202, val_acc: 0.96667
    Epoch [4480/10000], loss: 0.13733 acc: 1.00000 val_loss: 0.20183, val_acc: 0.96667
    Epoch [4490/10000], loss: 0.13712 acc: 1.00000 val_loss: 0.20165, val_acc: 0.96667
    Epoch [4500/10000], loss: 0.13691 acc: 1.00000 val_loss: 0.20147, val_acc: 0.96667
    Epoch [4510/10000], loss: 0.13670 acc: 1.00000 val_loss: 0.20128, val_acc: 0.96667
    Epoch [4520/10000], loss: 0.13650 acc: 1.00000 val_loss: 0.20110, val_acc: 0.96667
    Epoch [4530/10000], loss: 0.13629 acc: 1.00000 val_loss: 0.20092, val_acc: 0.96667
    Epoch [4540/10000], loss: 0.13608 acc: 1.00000 val_loss: 0.20074, val_acc: 0.96667
    Epoch [4550/10000], loss: 0.13588 acc: 1.00000 val_loss: 0.20056, val_acc: 0.96667
    Epoch [4560/10000], loss: 0.13567 acc: 1.00000 val_loss: 0.20038, val_acc: 0.96667
    Epoch [4570/10000], loss: 0.13547 acc: 1.00000 val_loss: 0.20021, val_acc: 0.96667
    Epoch [4580/10000], loss: 0.13527 acc: 1.00000 val_loss: 0.20003, val_acc: 0.96667
    Epoch [4590/10000], loss: 0.13507 acc: 1.00000 val_loss: 0.19985, val_acc: 0.96667
    Epoch [4600/10000], loss: 0.13486 acc: 1.00000 val_loss: 0.19968, val_acc: 0.96667
    Epoch [4610/10000], loss: 0.13466 acc: 1.00000 val_loss: 0.19950, val_acc: 0.96667
    Epoch [4620/10000], loss: 0.13446 acc: 1.00000 val_loss: 0.19933, val_acc: 0.96667
    Epoch [4630/10000], loss: 0.13426 acc: 1.00000 val_loss: 0.19916, val_acc: 0.96667
    Epoch [4640/10000], loss: 0.13407 acc: 1.00000 val_loss: 0.19898, val_acc: 0.96667
    Epoch [4650/10000], loss: 0.13387 acc: 1.00000 val_loss: 0.19881, val_acc: 0.96667
    Epoch [4660/10000], loss: 0.13367 acc: 1.00000 val_loss: 0.19864, val_acc: 0.96667
    Epoch [4670/10000], loss: 0.13348 acc: 1.00000 val_loss: 0.19847, val_acc: 0.96667
    Epoch [4680/10000], loss: 0.13328 acc: 1.00000 val_loss: 0.19830, val_acc: 0.96667
    Epoch [4690/10000], loss: 0.13308 acc: 1.00000 val_loss: 0.19813, val_acc: 0.96667
    Epoch [4700/10000], loss: 0.13289 acc: 1.00000 val_loss: 0.19796, val_acc: 0.96667
    Epoch [4710/10000], loss: 0.13270 acc: 1.00000 val_loss: 0.19779, val_acc: 0.96667
    Epoch [4720/10000], loss: 0.13250 acc: 1.00000 val_loss: 0.19762, val_acc: 0.96667
    Epoch [4730/10000], loss: 0.13231 acc: 1.00000 val_loss: 0.19746, val_acc: 0.96667
    Epoch [4740/10000], loss: 0.13212 acc: 1.00000 val_loss: 0.19729, val_acc: 0.96667
    Epoch [4750/10000], loss: 0.13193 acc: 1.00000 val_loss: 0.19712, val_acc: 0.96667
    Epoch [4760/10000], loss: 0.13174 acc: 1.00000 val_loss: 0.19696, val_acc: 0.96667
    Epoch [4770/10000], loss: 0.13155 acc: 1.00000 val_loss: 0.19679, val_acc: 0.96667
    Epoch [4780/10000], loss: 0.13136 acc: 1.00000 val_loss: 0.19663, val_acc: 0.96667
    Epoch [4790/10000], loss: 0.13117 acc: 1.00000 val_loss: 0.19647, val_acc: 0.96667
    Epoch [4800/10000], loss: 0.13099 acc: 1.00000 val_loss: 0.19630, val_acc: 0.96667
    Epoch [4810/10000], loss: 0.13080 acc: 1.00000 val_loss: 0.19614, val_acc: 0.96667
    Epoch [4820/10000], loss: 0.13061 acc: 1.00000 val_loss: 0.19598, val_acc: 0.96667
    Epoch [4830/10000], loss: 0.13043 acc: 1.00000 val_loss: 0.19582, val_acc: 0.96667
    Epoch [4840/10000], loss: 0.13024 acc: 1.00000 val_loss: 0.19566, val_acc: 0.96667
    Epoch [4850/10000], loss: 0.13006 acc: 1.00000 val_loss: 0.19550, val_acc: 0.96667
    Epoch [4860/10000], loss: 0.12988 acc: 1.00000 val_loss: 0.19534, val_acc: 0.96667
    Epoch [4870/10000], loss: 0.12969 acc: 1.00000 val_loss: 0.19518, val_acc: 0.96667
    Epoch [4880/10000], loss: 0.12951 acc: 1.00000 val_loss: 0.19502, val_acc: 0.96667
    Epoch [4890/10000], loss: 0.12933 acc: 1.00000 val_loss: 0.19487, val_acc: 0.96667
    Epoch [4900/10000], loss: 0.12915 acc: 1.00000 val_loss: 0.19471, val_acc: 0.96667
    Epoch [4910/10000], loss: 0.12897 acc: 1.00000 val_loss: 0.19455, val_acc: 0.96667
    Epoch [4920/10000], loss: 0.12879 acc: 1.00000 val_loss: 0.19440, val_acc: 0.96667
    Epoch [4930/10000], loss: 0.12861 acc: 1.00000 val_loss: 0.19424, val_acc: 0.96667
    Epoch [4940/10000], loss: 0.12843 acc: 1.00000 val_loss: 0.19409, val_acc: 0.96667
    Epoch [4950/10000], loss: 0.12825 acc: 1.00000 val_loss: 0.19394, val_acc: 0.96667
    Epoch [4960/10000], loss: 0.12808 acc: 1.00000 val_loss: 0.19378, val_acc: 0.96667
    Epoch [4970/10000], loss: 0.12790 acc: 1.00000 val_loss: 0.19363, val_acc: 0.96667
    Epoch [4980/10000], loss: 0.12772 acc: 1.00000 val_loss: 0.19348, val_acc: 0.96667
    Epoch [4990/10000], loss: 0.12755 acc: 1.00000 val_loss: 0.19333, val_acc: 0.96667
    Epoch [5000/10000], loss: 0.12737 acc: 1.00000 val_loss: 0.19318, val_acc: 0.96667
    Epoch [5010/10000], loss: 0.12720 acc: 1.00000 val_loss: 0.19302, val_acc: 0.96667
    Epoch [5020/10000], loss: 0.12703 acc: 1.00000 val_loss: 0.19287, val_acc: 0.96667
    Epoch [5030/10000], loss: 0.12685 acc: 1.00000 val_loss: 0.19273, val_acc: 0.96667
    Epoch [5040/10000], loss: 0.12668 acc: 1.00000 val_loss: 0.19258, val_acc: 0.96667
    Epoch [5050/10000], loss: 0.12651 acc: 1.00000 val_loss: 0.19243, val_acc: 0.96667
    Epoch [5060/10000], loss: 0.12634 acc: 1.00000 val_loss: 0.19228, val_acc: 0.96667
    Epoch [5070/10000], loss: 0.12617 acc: 1.00000 val_loss: 0.19213, val_acc: 0.96667
    Epoch [5080/10000], loss: 0.12600 acc: 1.00000 val_loss: 0.19199, val_acc: 0.96667
    Epoch [5090/10000], loss: 0.12583 acc: 1.00000 val_loss: 0.19184, val_acc: 0.96667
    Epoch [5100/10000], loss: 0.12566 acc: 1.00000 val_loss: 0.19169, val_acc: 0.96667
    Epoch [5110/10000], loss: 0.12549 acc: 1.00000 val_loss: 0.19155, val_acc: 0.96667
    Epoch [5120/10000], loss: 0.12532 acc: 1.00000 val_loss: 0.19140, val_acc: 0.96667
    Epoch [5130/10000], loss: 0.12515 acc: 1.00000 val_loss: 0.19126, val_acc: 0.96667
    Epoch [5140/10000], loss: 0.12499 acc: 1.00000 val_loss: 0.19112, val_acc: 0.96667
    Epoch [5150/10000], loss: 0.12482 acc: 1.00000 val_loss: 0.19097, val_acc: 0.96667
    Epoch [5160/10000], loss: 0.12465 acc: 1.00000 val_loss: 0.19083, val_acc: 0.96667
    Epoch [5170/10000], loss: 0.12449 acc: 1.00000 val_loss: 0.19069, val_acc: 0.96667
    Epoch [5180/10000], loss: 0.12432 acc: 1.00000 val_loss: 0.19055, val_acc: 0.96667
    Epoch [5190/10000], loss: 0.12416 acc: 1.00000 val_loss: 0.19041, val_acc: 0.96667
    Epoch [5200/10000], loss: 0.12400 acc: 1.00000 val_loss: 0.19027, val_acc: 0.96667
    Epoch [5210/10000], loss: 0.12383 acc: 1.00000 val_loss: 0.19013, val_acc: 0.96667
    Epoch [5220/10000], loss: 0.12367 acc: 1.00000 val_loss: 0.18999, val_acc: 0.96667
    Epoch [5230/10000], loss: 0.12351 acc: 1.00000 val_loss: 0.18985, val_acc: 0.96667
    Epoch [5240/10000], loss: 0.12335 acc: 1.00000 val_loss: 0.18971, val_acc: 0.96667
    Epoch [5250/10000], loss: 0.12319 acc: 1.00000 val_loss: 0.18957, val_acc: 0.96667
    Epoch [5260/10000], loss: 0.12303 acc: 1.00000 val_loss: 0.18943, val_acc: 0.96667
    Epoch [5270/10000], loss: 0.12287 acc: 1.00000 val_loss: 0.18930, val_acc: 0.96667
    Epoch [5280/10000], loss: 0.12271 acc: 1.00000 val_loss: 0.18916, val_acc: 0.96667
    Epoch [5290/10000], loss: 0.12255 acc: 1.00000 val_loss: 0.18902, val_acc: 0.96667
    Epoch [5300/10000], loss: 0.12239 acc: 1.00000 val_loss: 0.18889, val_acc: 0.96667
    Epoch [5310/10000], loss: 0.12223 acc: 1.00000 val_loss: 0.18875, val_acc: 0.96667
    Epoch [5320/10000], loss: 0.12207 acc: 1.00000 val_loss: 0.18862, val_acc: 0.96667
    Epoch [5330/10000], loss: 0.12192 acc: 1.00000 val_loss: 0.18848, val_acc: 0.96667
    Epoch [5340/10000], loss: 0.12176 acc: 1.00000 val_loss: 0.18835, val_acc: 0.96667
    Epoch [5350/10000], loss: 0.12160 acc: 1.00000 val_loss: 0.18821, val_acc: 0.96667
    Epoch [5360/10000], loss: 0.12145 acc: 1.00000 val_loss: 0.18808, val_acc: 0.96667
    Epoch [5370/10000], loss: 0.12129 acc: 1.00000 val_loss: 0.18795, val_acc: 0.96667
    Epoch [5380/10000], loss: 0.12114 acc: 1.00000 val_loss: 0.18782, val_acc: 0.96667
    Epoch [5390/10000], loss: 0.12099 acc: 1.00000 val_loss: 0.18768, val_acc: 0.96667
    Epoch [5400/10000], loss: 0.12083 acc: 1.00000 val_loss: 0.18755, val_acc: 0.96667
    Epoch [5410/10000], loss: 0.12068 acc: 1.00000 val_loss: 0.18742, val_acc: 0.96667
    Epoch [5420/10000], loss: 0.12053 acc: 1.00000 val_loss: 0.18729, val_acc: 0.96667
    Epoch [5430/10000], loss: 0.12037 acc: 1.00000 val_loss: 0.18716, val_acc: 0.96667
    Epoch [5440/10000], loss: 0.12022 acc: 1.00000 val_loss: 0.18703, val_acc: 0.96667
    Epoch [5450/10000], loss: 0.12007 acc: 1.00000 val_loss: 0.18690, val_acc: 0.96667
    Epoch [5460/10000], loss: 0.11992 acc: 1.00000 val_loss: 0.18678, val_acc: 0.96667
    Epoch [5470/10000], loss: 0.11977 acc: 1.00000 val_loss: 0.18665, val_acc: 0.96667
    Epoch [5480/10000], loss: 0.11962 acc: 1.00000 val_loss: 0.18652, val_acc: 0.96667
    Epoch [5490/10000], loss: 0.11947 acc: 1.00000 val_loss: 0.18639, val_acc: 0.96667
    Epoch [5500/10000], loss: 0.11932 acc: 1.00000 val_loss: 0.18627, val_acc: 0.96667
    Epoch [5510/10000], loss: 0.11918 acc: 1.00000 val_loss: 0.18614, val_acc: 0.96667
    Epoch [5520/10000], loss: 0.11903 acc: 1.00000 val_loss: 0.18601, val_acc: 0.96667
    Epoch [5530/10000], loss: 0.11888 acc: 1.00000 val_loss: 0.18589, val_acc: 0.96667
    Epoch [5540/10000], loss: 0.11873 acc: 1.00000 val_loss: 0.18576, val_acc: 0.96667
    Epoch [5550/10000], loss: 0.11859 acc: 1.00000 val_loss: 0.18564, val_acc: 0.96667
    Epoch [5560/10000], loss: 0.11844 acc: 1.00000 val_loss: 0.18551, val_acc: 0.96667
    Epoch [5570/10000], loss: 0.11830 acc: 1.00000 val_loss: 0.18539, val_acc: 0.96667
    Epoch [5580/10000], loss: 0.11815 acc: 1.00000 val_loss: 0.18527, val_acc: 0.96667
    Epoch [5590/10000], loss: 0.11801 acc: 1.00000 val_loss: 0.18514, val_acc: 0.96667
    Epoch [5600/10000], loss: 0.11786 acc: 1.00000 val_loss: 0.18502, val_acc: 0.96667
    Epoch [5610/10000], loss: 0.11772 acc: 1.00000 val_loss: 0.18490, val_acc: 0.96667
    Epoch [5620/10000], loss: 0.11757 acc: 1.00000 val_loss: 0.18478, val_acc: 0.96667
    Epoch [5630/10000], loss: 0.11743 acc: 1.00000 val_loss: 0.18465, val_acc: 0.96667
    Epoch [5640/10000], loss: 0.11729 acc: 1.00000 val_loss: 0.18453, val_acc: 0.96667
    Epoch [5650/10000], loss: 0.11715 acc: 1.00000 val_loss: 0.18441, val_acc: 0.96667
    Epoch [5660/10000], loss: 0.11701 acc: 1.00000 val_loss: 0.18429, val_acc: 0.96667
    Epoch [5670/10000], loss: 0.11686 acc: 1.00000 val_loss: 0.18417, val_acc: 0.96667
    Epoch [5680/10000], loss: 0.11672 acc: 1.00000 val_loss: 0.18405, val_acc: 0.96667
    Epoch [5690/10000], loss: 0.11658 acc: 1.00000 val_loss: 0.18393, val_acc: 0.96667
    Epoch [5700/10000], loss: 0.11644 acc: 1.00000 val_loss: 0.18381, val_acc: 0.96667
    Epoch [5710/10000], loss: 0.11630 acc: 1.00000 val_loss: 0.18370, val_acc: 0.96667
    Epoch [5720/10000], loss: 0.11616 acc: 1.00000 val_loss: 0.18358, val_acc: 0.96667
    Epoch [5730/10000], loss: 0.11603 acc: 1.00000 val_loss: 0.18346, val_acc: 0.96667
    Epoch [5740/10000], loss: 0.11589 acc: 1.00000 val_loss: 0.18334, val_acc: 0.96667
    Epoch [5750/10000], loss: 0.11575 acc: 1.00000 val_loss: 0.18323, val_acc: 0.96667
    Epoch [5760/10000], loss: 0.11561 acc: 1.00000 val_loss: 0.18311, val_acc: 0.96667
    Epoch [5770/10000], loss: 0.11547 acc: 1.00000 val_loss: 0.18299, val_acc: 0.96667
    Epoch [5780/10000], loss: 0.11534 acc: 1.00000 val_loss: 0.18288, val_acc: 0.96667
    Epoch [5790/10000], loss: 0.11520 acc: 1.00000 val_loss: 0.18276, val_acc: 0.96667
    Epoch [5800/10000], loss: 0.11507 acc: 1.00000 val_loss: 0.18265, val_acc: 0.96667
    Epoch [5810/10000], loss: 0.11493 acc: 1.00000 val_loss: 0.18253, val_acc: 0.96667
    Epoch [5820/10000], loss: 0.11480 acc: 1.00000 val_loss: 0.18242, val_acc: 0.96667
    Epoch [5830/10000], loss: 0.11466 acc: 1.00000 val_loss: 0.18231, val_acc: 0.96667
    Epoch [5840/10000], loss: 0.11453 acc: 1.00000 val_loss: 0.18219, val_acc: 0.96667
    Epoch [5850/10000], loss: 0.11439 acc: 1.00000 val_loss: 0.18208, val_acc: 0.96667
    Epoch [5860/10000], loss: 0.11426 acc: 1.00000 val_loss: 0.18197, val_acc: 0.96667
    Epoch [5870/10000], loss: 0.11413 acc: 1.00000 val_loss: 0.18185, val_acc: 0.96667
    Epoch [5880/10000], loss: 0.11399 acc: 1.00000 val_loss: 0.18174, val_acc: 0.96667
    Epoch [5890/10000], loss: 0.11386 acc: 1.00000 val_loss: 0.18163, val_acc: 0.96667
    Epoch [5900/10000], loss: 0.11373 acc: 1.00000 val_loss: 0.18152, val_acc: 0.96667
    Epoch [5910/10000], loss: 0.11360 acc: 1.00000 val_loss: 0.18141, val_acc: 0.96667
    Epoch [5920/10000], loss: 0.11347 acc: 1.00000 val_loss: 0.18130, val_acc: 0.96667
    Epoch [5930/10000], loss: 0.11333 acc: 1.00000 val_loss: 0.18119, val_acc: 0.96667
    Epoch [5940/10000], loss: 0.11320 acc: 1.00000 val_loss: 0.18108, val_acc: 0.96667
    Epoch [5950/10000], loss: 0.11307 acc: 1.00000 val_loss: 0.18097, val_acc: 0.96667
    Epoch [5960/10000], loss: 0.11294 acc: 1.00000 val_loss: 0.18086, val_acc: 0.96667
    Epoch [5970/10000], loss: 0.11282 acc: 1.00000 val_loss: 0.18075, val_acc: 0.96667
    Epoch [5980/10000], loss: 0.11269 acc: 1.00000 val_loss: 0.18064, val_acc: 0.96667
    Epoch [5990/10000], loss: 0.11256 acc: 1.00000 val_loss: 0.18053, val_acc: 0.96667
    Epoch [6000/10000], loss: 0.11243 acc: 1.00000 val_loss: 0.18042, val_acc: 0.96667
    Epoch [6010/10000], loss: 0.11230 acc: 1.00000 val_loss: 0.18031, val_acc: 0.96667
    Epoch [6020/10000], loss: 0.11217 acc: 1.00000 val_loss: 0.18021, val_acc: 0.96667
    Epoch [6030/10000], loss: 0.11205 acc: 1.00000 val_loss: 0.18010, val_acc: 0.96667
    Epoch [6040/10000], loss: 0.11192 acc: 1.00000 val_loss: 0.17999, val_acc: 0.96667
    Epoch [6050/10000], loss: 0.11179 acc: 1.00000 val_loss: 0.17989, val_acc: 0.96667
    Epoch [6060/10000], loss: 0.11167 acc: 1.00000 val_loss: 0.17978, val_acc: 0.96667
    Epoch [6070/10000], loss: 0.11154 acc: 1.00000 val_loss: 0.17968, val_acc: 0.96667
    Epoch [6080/10000], loss: 0.11142 acc: 1.00000 val_loss: 0.17957, val_acc: 0.96667
    Epoch [6090/10000], loss: 0.11129 acc: 1.00000 val_loss: 0.17947, val_acc: 0.96667
    Epoch [6100/10000], loss: 0.11117 acc: 1.00000 val_loss: 0.17936, val_acc: 0.96667
    Epoch [6110/10000], loss: 0.11104 acc: 1.00000 val_loss: 0.17926, val_acc: 0.96667
    Epoch [6120/10000], loss: 0.11092 acc: 1.00000 val_loss: 0.17915, val_acc: 0.96667
    Epoch [6130/10000], loss: 0.11079 acc: 1.00000 val_loss: 0.17905, val_acc: 0.96667
    Epoch [6140/10000], loss: 0.11067 acc: 1.00000 val_loss: 0.17894, val_acc: 0.96667
    Epoch [6150/10000], loss: 0.11055 acc: 1.00000 val_loss: 0.17884, val_acc: 0.96667
    Epoch [6160/10000], loss: 0.11043 acc: 1.00000 val_loss: 0.17874, val_acc: 0.96667
    Epoch [6170/10000], loss: 0.11030 acc: 1.00000 val_loss: 0.17864, val_acc: 0.96667
    Epoch [6180/10000], loss: 0.11018 acc: 1.00000 val_loss: 0.17853, val_acc: 0.96667
    Epoch [6190/10000], loss: 0.11006 acc: 1.00000 val_loss: 0.17843, val_acc: 0.96667
    Epoch [6200/10000], loss: 0.10994 acc: 1.00000 val_loss: 0.17833, val_acc: 0.96667
    Epoch [6210/10000], loss: 0.10982 acc: 1.00000 val_loss: 0.17823, val_acc: 0.96667
    Epoch [6220/10000], loss: 0.10970 acc: 1.00000 val_loss: 0.17813, val_acc: 0.96667
    Epoch [6230/10000], loss: 0.10958 acc: 1.00000 val_loss: 0.17803, val_acc: 0.96667
    Epoch [6240/10000], loss: 0.10946 acc: 1.00000 val_loss: 0.17793, val_acc: 0.96667
    Epoch [6250/10000], loss: 0.10934 acc: 1.00000 val_loss: 0.17783, val_acc: 0.96667
    Epoch [6260/10000], loss: 0.10922 acc: 1.00000 val_loss: 0.17773, val_acc: 0.96667
    Epoch [6270/10000], loss: 0.10910 acc: 1.00000 val_loss: 0.17763, val_acc: 0.96667
    Epoch [6280/10000], loss: 0.10898 acc: 1.00000 val_loss: 0.17753, val_acc: 0.96667
    Epoch [6290/10000], loss: 0.10886 acc: 1.00000 val_loss: 0.17743, val_acc: 0.96667
    Epoch [6300/10000], loss: 0.10874 acc: 1.00000 val_loss: 0.17733, val_acc: 0.96667
    Epoch [6310/10000], loss: 0.10863 acc: 1.00000 val_loss: 0.17723, val_acc: 0.96667
    Epoch [6320/10000], loss: 0.10851 acc: 1.00000 val_loss: 0.17713, val_acc: 0.96667
    Epoch [6330/10000], loss: 0.10839 acc: 1.00000 val_loss: 0.17704, val_acc: 0.96667
    Epoch [6340/10000], loss: 0.10828 acc: 1.00000 val_loss: 0.17694, val_acc: 0.96667
    Epoch [6350/10000], loss: 0.10816 acc: 1.00000 val_loss: 0.17684, val_acc: 0.96667
    Epoch [6360/10000], loss: 0.10804 acc: 1.00000 val_loss: 0.17675, val_acc: 0.96667
    Epoch [6370/10000], loss: 0.10793 acc: 1.00000 val_loss: 0.17665, val_acc: 0.96667
    Epoch [6380/10000], loss: 0.10781 acc: 1.00000 val_loss: 0.17655, val_acc: 0.96667
    Epoch [6390/10000], loss: 0.10770 acc: 1.00000 val_loss: 0.17646, val_acc: 0.96667
    Epoch [6400/10000], loss: 0.10758 acc: 1.00000 val_loss: 0.17636, val_acc: 0.96667
    Epoch [6410/10000], loss: 0.10747 acc: 1.00000 val_loss: 0.17627, val_acc: 0.96667
    Epoch [6420/10000], loss: 0.10735 acc: 1.00000 val_loss: 0.17617, val_acc: 0.96667
    Epoch [6430/10000], loss: 0.10724 acc: 1.00000 val_loss: 0.17608, val_acc: 0.96667
    Epoch [6440/10000], loss: 0.10713 acc: 1.00000 val_loss: 0.17598, val_acc: 0.96667
    Epoch [6450/10000], loss: 0.10701 acc: 1.00000 val_loss: 0.17589, val_acc: 0.96667
    Epoch [6460/10000], loss: 0.10690 acc: 1.00000 val_loss: 0.17579, val_acc: 0.96667
    Epoch [6470/10000], loss: 0.10679 acc: 1.00000 val_loss: 0.17570, val_acc: 0.96667
    Epoch [6480/10000], loss: 0.10667 acc: 1.00000 val_loss: 0.17561, val_acc: 0.96667
    Epoch [6490/10000], loss: 0.10656 acc: 1.00000 val_loss: 0.17551, val_acc: 0.96667
    Epoch [6500/10000], loss: 0.10645 acc: 1.00000 val_loss: 0.17542, val_acc: 0.96667
    Epoch [6510/10000], loss: 0.10634 acc: 1.00000 val_loss: 0.17533, val_acc: 0.96667
    Epoch [6520/10000], loss: 0.10623 acc: 1.00000 val_loss: 0.17523, val_acc: 0.96667
    Epoch [6530/10000], loss: 0.10612 acc: 1.00000 val_loss: 0.17514, val_acc: 0.96667
    Epoch [6540/10000], loss: 0.10600 acc: 1.00000 val_loss: 0.17505, val_acc: 0.96667
    Epoch [6550/10000], loss: 0.10589 acc: 1.00000 val_loss: 0.17496, val_acc: 0.96667
    Epoch [6560/10000], loss: 0.10578 acc: 1.00000 val_loss: 0.17487, val_acc: 0.96667
    Epoch [6570/10000], loss: 0.10567 acc: 1.00000 val_loss: 0.17478, val_acc: 0.96667
    Epoch [6580/10000], loss: 0.10556 acc: 1.00000 val_loss: 0.17468, val_acc: 0.96667
    Epoch [6590/10000], loss: 0.10546 acc: 1.00000 val_loss: 0.17459, val_acc: 0.96667
    Epoch [6600/10000], loss: 0.10535 acc: 1.00000 val_loss: 0.17450, val_acc: 0.96667
    Epoch [6610/10000], loss: 0.10524 acc: 1.00000 val_loss: 0.17441, val_acc: 0.96667
    Epoch [6620/10000], loss: 0.10513 acc: 1.00000 val_loss: 0.17432, val_acc: 0.96667
    Epoch [6630/10000], loss: 0.10502 acc: 1.00000 val_loss: 0.17423, val_acc: 0.96667
    Epoch [6640/10000], loss: 0.10491 acc: 1.00000 val_loss: 0.17414, val_acc: 0.96667
    Epoch [6650/10000], loss: 0.10481 acc: 1.00000 val_loss: 0.17406, val_acc: 0.96667
    Epoch [6660/10000], loss: 0.10470 acc: 1.00000 val_loss: 0.17397, val_acc: 0.96667
    Epoch [6670/10000], loss: 0.10459 acc: 1.00000 val_loss: 0.17388, val_acc: 0.96667
    Epoch [6680/10000], loss: 0.10448 acc: 1.00000 val_loss: 0.17379, val_acc: 0.96667
    Epoch [6690/10000], loss: 0.10438 acc: 1.00000 val_loss: 0.17370, val_acc: 0.96667
    Epoch [6700/10000], loss: 0.10427 acc: 1.00000 val_loss: 0.17361, val_acc: 0.96667
    Epoch [6710/10000], loss: 0.10417 acc: 1.00000 val_loss: 0.17353, val_acc: 0.96667
    Epoch [6720/10000], loss: 0.10406 acc: 1.00000 val_loss: 0.17344, val_acc: 0.96667
    Epoch [6730/10000], loss: 0.10395 acc: 1.00000 val_loss: 0.17335, val_acc: 0.96667
    Epoch [6740/10000], loss: 0.10385 acc: 1.00000 val_loss: 0.17326, val_acc: 0.96667
    Epoch [6750/10000], loss: 0.10374 acc: 1.00000 val_loss: 0.17318, val_acc: 0.96667
    Epoch [6760/10000], loss: 0.10364 acc: 1.00000 val_loss: 0.17309, val_acc: 0.96667
    Epoch [6770/10000], loss: 0.10354 acc: 1.00000 val_loss: 0.17301, val_acc: 0.96667
    Epoch [6780/10000], loss: 0.10343 acc: 1.00000 val_loss: 0.17292, val_acc: 0.96667
    Epoch [6790/10000], loss: 0.10333 acc: 1.00000 val_loss: 0.17283, val_acc: 0.96667
    Epoch [6800/10000], loss: 0.10322 acc: 1.00000 val_loss: 0.17275, val_acc: 0.96667
    Epoch [6810/10000], loss: 0.10312 acc: 1.00000 val_loss: 0.17266, val_acc: 0.96667
    Epoch [6820/10000], loss: 0.10302 acc: 1.00000 val_loss: 0.17258, val_acc: 0.96667
    Epoch [6830/10000], loss: 0.10292 acc: 1.00000 val_loss: 0.17249, val_acc: 0.96667
    Epoch [6840/10000], loss: 0.10281 acc: 1.00000 val_loss: 0.17241, val_acc: 0.96667
    Epoch [6850/10000], loss: 0.10271 acc: 1.00000 val_loss: 0.17233, val_acc: 0.96667
    Epoch [6860/10000], loss: 0.10261 acc: 1.00000 val_loss: 0.17224, val_acc: 0.96667
    Epoch [6870/10000], loss: 0.10251 acc: 1.00000 val_loss: 0.17216, val_acc: 0.96667
    Epoch [6880/10000], loss: 0.10241 acc: 1.00000 val_loss: 0.17207, val_acc: 0.96667
    Epoch [6890/10000], loss: 0.10230 acc: 1.00000 val_loss: 0.17199, val_acc: 0.96667
    Epoch [6900/10000], loss: 0.10220 acc: 1.00000 val_loss: 0.17191, val_acc: 0.96667
    Epoch [6910/10000], loss: 0.10210 acc: 1.00000 val_loss: 0.17182, val_acc: 0.96667
    Epoch [6920/10000], loss: 0.10200 acc: 1.00000 val_loss: 0.17174, val_acc: 0.96667
    Epoch [6930/10000], loss: 0.10190 acc: 1.00000 val_loss: 0.17166, val_acc: 0.96667
    Epoch [6940/10000], loss: 0.10180 acc: 1.00000 val_loss: 0.17158, val_acc: 0.96667
    Epoch [6950/10000], loss: 0.10170 acc: 1.00000 val_loss: 0.17150, val_acc: 0.96667
    Epoch [6960/10000], loss: 0.10160 acc: 1.00000 val_loss: 0.17141, val_acc: 0.96667
    Epoch [6970/10000], loss: 0.10150 acc: 1.00000 val_loss: 0.17133, val_acc: 0.96667
    Epoch [6980/10000], loss: 0.10140 acc: 1.00000 val_loss: 0.17125, val_acc: 0.96667
    Epoch [6990/10000], loss: 0.10130 acc: 1.00000 val_loss: 0.17117, val_acc: 0.96667
    Epoch [7000/10000], loss: 0.10121 acc: 1.00000 val_loss: 0.17109, val_acc: 0.96667
    Epoch [7010/10000], loss: 0.10111 acc: 1.00000 val_loss: 0.17101, val_acc: 0.96667
    Epoch [7020/10000], loss: 0.10101 acc: 1.00000 val_loss: 0.17093, val_acc: 0.96667
    Epoch [7030/10000], loss: 0.10091 acc: 1.00000 val_loss: 0.17085, val_acc: 0.96667
    Epoch [7040/10000], loss: 0.10081 acc: 1.00000 val_loss: 0.17077, val_acc: 0.96667
    Epoch [7050/10000], loss: 0.10072 acc: 1.00000 val_loss: 0.17069, val_acc: 0.96667
    Epoch [7060/10000], loss: 0.10062 acc: 1.00000 val_loss: 0.17061, val_acc: 0.96667
    Epoch [7070/10000], loss: 0.10052 acc: 1.00000 val_loss: 0.17053, val_acc: 0.96667
    Epoch [7080/10000], loss: 0.10043 acc: 1.00000 val_loss: 0.17045, val_acc: 0.96667
    Epoch [7090/10000], loss: 0.10033 acc: 1.00000 val_loss: 0.17037, val_acc: 0.96667
    Epoch [7100/10000], loss: 0.10023 acc: 1.00000 val_loss: 0.17029, val_acc: 0.96667
    Epoch [7110/10000], loss: 0.10014 acc: 1.00000 val_loss: 0.17021, val_acc: 0.96667
    Epoch [7120/10000], loss: 0.10004 acc: 1.00000 val_loss: 0.17014, val_acc: 0.96667
    Epoch [7130/10000], loss: 0.09995 acc: 1.00000 val_loss: 0.17006, val_acc: 0.96667
    Epoch [7140/10000], loss: 0.09985 acc: 1.00000 val_loss: 0.16998, val_acc: 0.96667
    Epoch [7150/10000], loss: 0.09976 acc: 1.00000 val_loss: 0.16990, val_acc: 0.96667
    Epoch [7160/10000], loss: 0.09966 acc: 1.00000 val_loss: 0.16982, val_acc: 0.96667
    Epoch [7170/10000], loss: 0.09957 acc: 1.00000 val_loss: 0.16975, val_acc: 0.96667
    Epoch [7180/10000], loss: 0.09947 acc: 1.00000 val_loss: 0.16967, val_acc: 0.96667
    Epoch [7190/10000], loss: 0.09938 acc: 1.00000 val_loss: 0.16959, val_acc: 0.96667
    Epoch [7200/10000], loss: 0.09928 acc: 1.00000 val_loss: 0.16952, val_acc: 0.96667
    Epoch [7210/10000], loss: 0.09919 acc: 1.00000 val_loss: 0.16944, val_acc: 0.96667
    Epoch [7220/10000], loss: 0.09910 acc: 1.00000 val_loss: 0.16936, val_acc: 0.96667
    Epoch [7230/10000], loss: 0.09900 acc: 1.00000 val_loss: 0.16929, val_acc: 0.96667
    Epoch [7240/10000], loss: 0.09891 acc: 1.00000 val_loss: 0.16921, val_acc: 0.96667
    Epoch [7250/10000], loss: 0.09882 acc: 1.00000 val_loss: 0.16914, val_acc: 0.96667
    Epoch [7260/10000], loss: 0.09873 acc: 1.00000 val_loss: 0.16906, val_acc: 0.96667
    Epoch [7270/10000], loss: 0.09863 acc: 1.00000 val_loss: 0.16899, val_acc: 0.96667
    Epoch [7280/10000], loss: 0.09854 acc: 1.00000 val_loss: 0.16891, val_acc: 0.96667
    Epoch [7290/10000], loss: 0.09845 acc: 1.00000 val_loss: 0.16884, val_acc: 0.96667
    Epoch [7300/10000], loss: 0.09836 acc: 1.00000 val_loss: 0.16876, val_acc: 0.96667
    Epoch [7310/10000], loss: 0.09827 acc: 1.00000 val_loss: 0.16869, val_acc: 0.96667
    Epoch [7320/10000], loss: 0.09817 acc: 1.00000 val_loss: 0.16861, val_acc: 0.96667
    Epoch [7330/10000], loss: 0.09808 acc: 1.00000 val_loss: 0.16854, val_acc: 0.96667
    Epoch [7340/10000], loss: 0.09799 acc: 1.00000 val_loss: 0.16846, val_acc: 0.96667
    Epoch [7350/10000], loss: 0.09790 acc: 1.00000 val_loss: 0.16839, val_acc: 0.96667
    Epoch [7360/10000], loss: 0.09781 acc: 1.00000 val_loss: 0.16832, val_acc: 0.96667
    Epoch [7370/10000], loss: 0.09772 acc: 1.00000 val_loss: 0.16824, val_acc: 0.96667
    Epoch [7380/10000], loss: 0.09763 acc: 1.00000 val_loss: 0.16817, val_acc: 0.96667
    Epoch [7390/10000], loss: 0.09754 acc: 1.00000 val_loss: 0.16810, val_acc: 0.96667
    Epoch [7400/10000], loss: 0.09745 acc: 1.00000 val_loss: 0.16802, val_acc: 0.96667
    Epoch [7410/10000], loss: 0.09736 acc: 1.00000 val_loss: 0.16795, val_acc: 0.96667
    Epoch [7420/10000], loss: 0.09727 acc: 1.00000 val_loss: 0.16788, val_acc: 0.96667
    Epoch [7430/10000], loss: 0.09718 acc: 1.00000 val_loss: 0.16781, val_acc: 0.96667
    Epoch [7440/10000], loss: 0.09710 acc: 1.00000 val_loss: 0.16774, val_acc: 0.96667
    Epoch [7450/10000], loss: 0.09701 acc: 1.00000 val_loss: 0.16766, val_acc: 0.96667
    Epoch [7460/10000], loss: 0.09692 acc: 1.00000 val_loss: 0.16759, val_acc: 0.96667
    Epoch [7470/10000], loss: 0.09683 acc: 1.00000 val_loss: 0.16752, val_acc: 0.96667
    Epoch [7480/10000], loss: 0.09674 acc: 1.00000 val_loss: 0.16745, val_acc: 0.96667
    Epoch [7490/10000], loss: 0.09665 acc: 1.00000 val_loss: 0.16738, val_acc: 0.96667
    Epoch [7500/10000], loss: 0.09657 acc: 1.00000 val_loss: 0.16731, val_acc: 0.96667
    Epoch [7510/10000], loss: 0.09648 acc: 1.00000 val_loss: 0.16724, val_acc: 0.96667
    Epoch [7520/10000], loss: 0.09639 acc: 1.00000 val_loss: 0.16717, val_acc: 0.96667
    Epoch [7530/10000], loss: 0.09631 acc: 1.00000 val_loss: 0.16710, val_acc: 0.96667
    Epoch [7540/10000], loss: 0.09622 acc: 1.00000 val_loss: 0.16703, val_acc: 0.96667
    Epoch [7550/10000], loss: 0.09613 acc: 1.00000 val_loss: 0.16696, val_acc: 0.96667
    Epoch [7560/10000], loss: 0.09605 acc: 1.00000 val_loss: 0.16689, val_acc: 0.96667
    Epoch [7570/10000], loss: 0.09596 acc: 1.00000 val_loss: 0.16682, val_acc: 0.96667
    Epoch [7580/10000], loss: 0.09587 acc: 1.00000 val_loss: 0.16675, val_acc: 0.96667
    Epoch [7590/10000], loss: 0.09579 acc: 1.00000 val_loss: 0.16668, val_acc: 0.96667
    Epoch [7600/10000], loss: 0.09570 acc: 1.00000 val_loss: 0.16661, val_acc: 0.96667
    Epoch [7610/10000], loss: 0.09562 acc: 1.00000 val_loss: 0.16654, val_acc: 0.96667
    Epoch [7620/10000], loss: 0.09553 acc: 1.00000 val_loss: 0.16647, val_acc: 0.96667
    Epoch [7630/10000], loss: 0.09545 acc: 1.00000 val_loss: 0.16640, val_acc: 0.96667
    Epoch [7640/10000], loss: 0.09536 acc: 1.00000 val_loss: 0.16633, val_acc: 0.96667
    Epoch [7650/10000], loss: 0.09528 acc: 1.00000 val_loss: 0.16626, val_acc: 0.96667
    Epoch [7660/10000], loss: 0.09519 acc: 1.00000 val_loss: 0.16620, val_acc: 0.96667
    Epoch [7670/10000], loss: 0.09511 acc: 1.00000 val_loss: 0.16613, val_acc: 0.96667
    Epoch [7680/10000], loss: 0.09502 acc: 1.00000 val_loss: 0.16606, val_acc: 0.96667
    Epoch [7690/10000], loss: 0.09494 acc: 1.00000 val_loss: 0.16599, val_acc: 0.96667
    Epoch [7700/10000], loss: 0.09486 acc: 1.00000 val_loss: 0.16593, val_acc: 0.96667
    Epoch [7710/10000], loss: 0.09477 acc: 1.00000 val_loss: 0.16586, val_acc: 0.96667
    Epoch [7720/10000], loss: 0.09469 acc: 1.00000 val_loss: 0.16579, val_acc: 0.96667
    Epoch [7730/10000], loss: 0.09461 acc: 1.00000 val_loss: 0.16572, val_acc: 0.96667
    Epoch [7740/10000], loss: 0.09452 acc: 1.00000 val_loss: 0.16566, val_acc: 0.96667
    Epoch [7750/10000], loss: 0.09444 acc: 1.00000 val_loss: 0.16559, val_acc: 0.96667
    Epoch [7760/10000], loss: 0.09436 acc: 1.00000 val_loss: 0.16552, val_acc: 0.96667
    Epoch [7770/10000], loss: 0.09428 acc: 1.00000 val_loss: 0.16546, val_acc: 0.96667
    Epoch [7780/10000], loss: 0.09419 acc: 1.00000 val_loss: 0.16539, val_acc: 0.96667
    Epoch [7790/10000], loss: 0.09411 acc: 1.00000 val_loss: 0.16533, val_acc: 0.96667
    Epoch [7800/10000], loss: 0.09403 acc: 1.00000 val_loss: 0.16526, val_acc: 0.96667
    Epoch [7810/10000], loss: 0.09395 acc: 1.00000 val_loss: 0.16519, val_acc: 0.96667
    Epoch [7820/10000], loss: 0.09387 acc: 1.00000 val_loss: 0.16513, val_acc: 0.96667
    Epoch [7830/10000], loss: 0.09378 acc: 1.00000 val_loss: 0.16506, val_acc: 0.96667
    Epoch [7840/10000], loss: 0.09370 acc: 1.00000 val_loss: 0.16500, val_acc: 0.96667
    Epoch [7850/10000], loss: 0.09362 acc: 1.00000 val_loss: 0.16493, val_acc: 0.96667
    Epoch [7860/10000], loss: 0.09354 acc: 1.00000 val_loss: 0.16487, val_acc: 0.96667
    Epoch [7870/10000], loss: 0.09346 acc: 1.00000 val_loss: 0.16480, val_acc: 0.96667
    Epoch [7880/10000], loss: 0.09338 acc: 1.00000 val_loss: 0.16474, val_acc: 0.96667
    Epoch [7890/10000], loss: 0.09330 acc: 1.00000 val_loss: 0.16468, val_acc: 0.96667
    Epoch [7900/10000], loss: 0.09322 acc: 1.00000 val_loss: 0.16461, val_acc: 0.96667
    Epoch [7910/10000], loss: 0.09314 acc: 1.00000 val_loss: 0.16455, val_acc: 0.96667
    Epoch [7920/10000], loss: 0.09306 acc: 1.00000 val_loss: 0.16448, val_acc: 0.96667
    Epoch [7930/10000], loss: 0.09298 acc: 1.00000 val_loss: 0.16442, val_acc: 0.96667
    Epoch [7940/10000], loss: 0.09290 acc: 1.00000 val_loss: 0.16436, val_acc: 0.96667
    Epoch [7950/10000], loss: 0.09282 acc: 1.00000 val_loss: 0.16429, val_acc: 0.96667
    Epoch [7960/10000], loss: 0.09274 acc: 1.00000 val_loss: 0.16423, val_acc: 0.96667
    Epoch [7970/10000], loss: 0.09266 acc: 1.00000 val_loss: 0.16417, val_acc: 0.96667
    Epoch [7980/10000], loss: 0.09259 acc: 1.00000 val_loss: 0.16410, val_acc: 0.96667
    Epoch [7990/10000], loss: 0.09251 acc: 1.00000 val_loss: 0.16404, val_acc: 0.96667
    Epoch [8000/10000], loss: 0.09243 acc: 1.00000 val_loss: 0.16398, val_acc: 0.96667
    Epoch [8010/10000], loss: 0.09235 acc: 1.00000 val_loss: 0.16392, val_acc: 0.96667
    Epoch [8020/10000], loss: 0.09227 acc: 1.00000 val_loss: 0.16385, val_acc: 0.96667
    Epoch [8030/10000], loss: 0.09219 acc: 1.00000 val_loss: 0.16379, val_acc: 0.96667
    Epoch [8040/10000], loss: 0.09212 acc: 1.00000 val_loss: 0.16373, val_acc: 0.96667
    Epoch [8050/10000], loss: 0.09204 acc: 1.00000 val_loss: 0.16367, val_acc: 0.96667
    Epoch [8060/10000], loss: 0.09196 acc: 1.00000 val_loss: 0.16361, val_acc: 0.96667
    Epoch [8070/10000], loss: 0.09188 acc: 1.00000 val_loss: 0.16354, val_acc: 0.96667
    Epoch [8080/10000], loss: 0.09181 acc: 1.00000 val_loss: 0.16348, val_acc: 0.96667
    Epoch [8090/10000], loss: 0.09173 acc: 1.00000 val_loss: 0.16342, val_acc: 0.96667
    Epoch [8100/10000], loss: 0.09165 acc: 1.00000 val_loss: 0.16336, val_acc: 0.96667
    Epoch [8110/10000], loss: 0.09158 acc: 1.00000 val_loss: 0.16330, val_acc: 0.96667
    Epoch [8120/10000], loss: 0.09150 acc: 1.00000 val_loss: 0.16324, val_acc: 0.96667
    Epoch [8130/10000], loss: 0.09142 acc: 1.00000 val_loss: 0.16318, val_acc: 0.96667
    Epoch [8140/10000], loss: 0.09135 acc: 1.00000 val_loss: 0.16312, val_acc: 0.96667
    Epoch [8150/10000], loss: 0.09127 acc: 1.00000 val_loss: 0.16306, val_acc: 0.96667
    Epoch [8160/10000], loss: 0.09120 acc: 1.00000 val_loss: 0.16300, val_acc: 0.96667
    Epoch [8170/10000], loss: 0.09112 acc: 1.00000 val_loss: 0.16294, val_acc: 0.96667
    Epoch [8180/10000], loss: 0.09105 acc: 1.00000 val_loss: 0.16288, val_acc: 0.96667
    Epoch [8190/10000], loss: 0.09097 acc: 1.00000 val_loss: 0.16282, val_acc: 0.96667
    Epoch [8200/10000], loss: 0.09089 acc: 1.00000 val_loss: 0.16276, val_acc: 0.96667
    Epoch [8210/10000], loss: 0.09082 acc: 1.00000 val_loss: 0.16270, val_acc: 0.96667
    Epoch [8220/10000], loss: 0.09074 acc: 1.00000 val_loss: 0.16264, val_acc: 0.96667
    Epoch [8230/10000], loss: 0.09067 acc: 1.00000 val_loss: 0.16258, val_acc: 0.96667
    Epoch [8240/10000], loss: 0.09060 acc: 1.00000 val_loss: 0.16252, val_acc: 0.96667
    Epoch [8250/10000], loss: 0.09052 acc: 1.00000 val_loss: 0.16246, val_acc: 0.96667
    Epoch [8260/10000], loss: 0.09045 acc: 1.00000 val_loss: 0.16240, val_acc: 0.96667
    Epoch [8270/10000], loss: 0.09037 acc: 1.00000 val_loss: 0.16234, val_acc: 0.96667
    Epoch [8280/10000], loss: 0.09030 acc: 1.00000 val_loss: 0.16228, val_acc: 0.96667
    Epoch [8290/10000], loss: 0.09023 acc: 1.00000 val_loss: 0.16222, val_acc: 0.96667
    Epoch [8300/10000], loss: 0.09015 acc: 1.00000 val_loss: 0.16217, val_acc: 0.96667
    Epoch [8310/10000], loss: 0.09008 acc: 1.00000 val_loss: 0.16211, val_acc: 0.96667
    Epoch [8320/10000], loss: 0.09000 acc: 1.00000 val_loss: 0.16205, val_acc: 0.96667
    Epoch [8330/10000], loss: 0.08993 acc: 1.00000 val_loss: 0.16199, val_acc: 0.96667
    Epoch [8340/10000], loss: 0.08986 acc: 1.00000 val_loss: 0.16193, val_acc: 0.96667
    Epoch [8350/10000], loss: 0.08979 acc: 1.00000 val_loss: 0.16188, val_acc: 0.96667
    Epoch [8360/10000], loss: 0.08971 acc: 1.00000 val_loss: 0.16182, val_acc: 0.96667
    Epoch [8370/10000], loss: 0.08964 acc: 1.00000 val_loss: 0.16176, val_acc: 0.96667
    Epoch [8380/10000], loss: 0.08957 acc: 1.00000 val_loss: 0.16170, val_acc: 0.96667
    Epoch [8390/10000], loss: 0.08950 acc: 1.00000 val_loss: 0.16165, val_acc: 0.96667
    Epoch [8400/10000], loss: 0.08942 acc: 1.00000 val_loss: 0.16159, val_acc: 0.96667
    Epoch [8410/10000], loss: 0.08935 acc: 1.00000 val_loss: 0.16153, val_acc: 0.96667
    Epoch [8420/10000], loss: 0.08928 acc: 1.00000 val_loss: 0.16148, val_acc: 0.96667
    Epoch [8430/10000], loss: 0.08921 acc: 1.00000 val_loss: 0.16142, val_acc: 0.96667
    Epoch [8440/10000], loss: 0.08914 acc: 1.00000 val_loss: 0.16136, val_acc: 0.96667
    Epoch [8450/10000], loss: 0.08907 acc: 1.00000 val_loss: 0.16131, val_acc: 0.96667
    Epoch [8460/10000], loss: 0.08899 acc: 1.00000 val_loss: 0.16125, val_acc: 0.96667
    Epoch [8470/10000], loss: 0.08892 acc: 1.00000 val_loss: 0.16119, val_acc: 0.96667
    Epoch [8480/10000], loss: 0.08885 acc: 1.00000 val_loss: 0.16114, val_acc: 0.96667
    Epoch [8490/10000], loss: 0.08878 acc: 1.00000 val_loss: 0.16108, val_acc: 0.96667
    Epoch [8500/10000], loss: 0.08871 acc: 1.00000 val_loss: 0.16103, val_acc: 0.96667
    Epoch [8510/10000], loss: 0.08864 acc: 1.00000 val_loss: 0.16097, val_acc: 0.96667
    Epoch [8520/10000], loss: 0.08857 acc: 1.00000 val_loss: 0.16092, val_acc: 0.96667
    Epoch [8530/10000], loss: 0.08850 acc: 1.00000 val_loss: 0.16086, val_acc: 0.96667
    Epoch [8540/10000], loss: 0.08843 acc: 1.00000 val_loss: 0.16081, val_acc: 0.96667
    Epoch [8550/10000], loss: 0.08836 acc: 1.00000 val_loss: 0.16075, val_acc: 0.96667
    Epoch [8560/10000], loss: 0.08829 acc: 1.00000 val_loss: 0.16070, val_acc: 0.96667
    Epoch [8570/10000], loss: 0.08822 acc: 1.00000 val_loss: 0.16064, val_acc: 0.96667
    Epoch [8580/10000], loss: 0.08815 acc: 1.00000 val_loss: 0.16059, val_acc: 0.96667
    Epoch [8590/10000], loss: 0.08808 acc: 1.00000 val_loss: 0.16053, val_acc: 0.96667
    Epoch [8600/10000], loss: 0.08801 acc: 1.00000 val_loss: 0.16048, val_acc: 0.96667
    Epoch [8610/10000], loss: 0.08794 acc: 1.00000 val_loss: 0.16042, val_acc: 0.96667
    Epoch [8620/10000], loss: 0.08787 acc: 1.00000 val_loss: 0.16037, val_acc: 0.96667
    Epoch [8630/10000], loss: 0.08780 acc: 1.00000 val_loss: 0.16031, val_acc: 0.96667
    Epoch [8640/10000], loss: 0.08774 acc: 1.00000 val_loss: 0.16026, val_acc: 0.96667
    Epoch [8650/10000], loss: 0.08767 acc: 1.00000 val_loss: 0.16021, val_acc: 0.96667
    Epoch [8660/10000], loss: 0.08760 acc: 1.00000 val_loss: 0.16015, val_acc: 0.96667
    Epoch [8670/10000], loss: 0.08753 acc: 1.00000 val_loss: 0.16010, val_acc: 0.96667
    Epoch [8680/10000], loss: 0.08746 acc: 1.00000 val_loss: 0.16005, val_acc: 0.96667
    Epoch [8690/10000], loss: 0.08739 acc: 1.00000 val_loss: 0.15999, val_acc: 0.96667
    Epoch [8700/10000], loss: 0.08733 acc: 1.00000 val_loss: 0.15994, val_acc: 0.96667
    Epoch [8710/10000], loss: 0.08726 acc: 1.00000 val_loss: 0.15989, val_acc: 0.96667
    Epoch [8720/10000], loss: 0.08719 acc: 1.00000 val_loss: 0.15983, val_acc: 0.96667
    Epoch [8730/10000], loss: 0.08712 acc: 1.00000 val_loss: 0.15978, val_acc: 0.96667
    Epoch [8740/10000], loss: 0.08706 acc: 1.00000 val_loss: 0.15973, val_acc: 0.96667
    Epoch [8750/10000], loss: 0.08699 acc: 1.00000 val_loss: 0.15967, val_acc: 0.96667
    Epoch [8760/10000], loss: 0.08692 acc: 1.00000 val_loss: 0.15962, val_acc: 0.96667
    Epoch [8770/10000], loss: 0.08685 acc: 1.00000 val_loss: 0.15957, val_acc: 0.96667
    Epoch [8780/10000], loss: 0.08679 acc: 1.00000 val_loss: 0.15952, val_acc: 0.96667
    Epoch [8790/10000], loss: 0.08672 acc: 1.00000 val_loss: 0.15946, val_acc: 0.96667
    Epoch [8800/10000], loss: 0.08665 acc: 1.00000 val_loss: 0.15941, val_acc: 0.96667
    Epoch [8810/10000], loss: 0.08659 acc: 1.00000 val_loss: 0.15936, val_acc: 0.96667
    Epoch [8820/10000], loss: 0.08652 acc: 1.00000 val_loss: 0.15931, val_acc: 0.96667
    Epoch [8830/10000], loss: 0.08646 acc: 1.00000 val_loss: 0.15926, val_acc: 0.96667
    Epoch [8840/10000], loss: 0.08639 acc: 1.00000 val_loss: 0.15921, val_acc: 0.96667
    Epoch [8850/10000], loss: 0.08632 acc: 1.00000 val_loss: 0.15915, val_acc: 0.96667
    Epoch [8860/10000], loss: 0.08626 acc: 1.00000 val_loss: 0.15910, val_acc: 0.96667
    Epoch [8870/10000], loss: 0.08619 acc: 1.00000 val_loss: 0.15905, val_acc: 0.96667
    Epoch [8880/10000], loss: 0.08613 acc: 1.00000 val_loss: 0.15900, val_acc: 0.96667
    Epoch [8890/10000], loss: 0.08606 acc: 1.00000 val_loss: 0.15895, val_acc: 0.96667
    Epoch [8900/10000], loss: 0.08600 acc: 1.00000 val_loss: 0.15890, val_acc: 0.96667
    Epoch [8910/10000], loss: 0.08593 acc: 1.00000 val_loss: 0.15885, val_acc: 0.96667
    Epoch [8920/10000], loss: 0.08587 acc: 1.00000 val_loss: 0.15880, val_acc: 0.96667
    Epoch [8930/10000], loss: 0.08580 acc: 1.00000 val_loss: 0.15875, val_acc: 0.96667
    Epoch [8940/10000], loss: 0.08574 acc: 1.00000 val_loss: 0.15870, val_acc: 0.96667
    Epoch [8950/10000], loss: 0.08567 acc: 1.00000 val_loss: 0.15865, val_acc: 0.96667
    Epoch [8960/10000], loss: 0.08561 acc: 1.00000 val_loss: 0.15859, val_acc: 0.96667
    Epoch [8970/10000], loss: 0.08554 acc: 1.00000 val_loss: 0.15854, val_acc: 0.96667
    Epoch [8980/10000], loss: 0.08548 acc: 1.00000 val_loss: 0.15849, val_acc: 0.96667
    Epoch [8990/10000], loss: 0.08541 acc: 1.00000 val_loss: 0.15844, val_acc: 0.96667
    Epoch [9000/10000], loss: 0.08535 acc: 1.00000 val_loss: 0.15839, val_acc: 0.96667
    Epoch [9010/10000], loss: 0.08529 acc: 1.00000 val_loss: 0.15834, val_acc: 0.96667
    Epoch [9020/10000], loss: 0.08522 acc: 1.00000 val_loss: 0.15830, val_acc: 0.96667
    Epoch [9030/10000], loss: 0.08516 acc: 1.00000 val_loss: 0.15825, val_acc: 0.96667
    Epoch [9040/10000], loss: 0.08509 acc: 1.00000 val_loss: 0.15820, val_acc: 0.96667
    Epoch [9050/10000], loss: 0.08503 acc: 1.00000 val_loss: 0.15815, val_acc: 0.96667
    Epoch [9060/10000], loss: 0.08497 acc: 1.00000 val_loss: 0.15810, val_acc: 0.96667
    Epoch [9070/10000], loss: 0.08490 acc: 1.00000 val_loss: 0.15805, val_acc: 0.96667
    Epoch [9080/10000], loss: 0.08484 acc: 1.00000 val_loss: 0.15800, val_acc: 0.96667
    Epoch [9090/10000], loss: 0.08478 acc: 1.00000 val_loss: 0.15795, val_acc: 0.96667
    Epoch [9100/10000], loss: 0.08472 acc: 1.00000 val_loss: 0.15790, val_acc: 0.96667
    Epoch [9110/10000], loss: 0.08465 acc: 1.00000 val_loss: 0.15785, val_acc: 0.96667
    Epoch [9120/10000], loss: 0.08459 acc: 1.00000 val_loss: 0.15781, val_acc: 0.96667
    Epoch [9130/10000], loss: 0.08453 acc: 1.00000 val_loss: 0.15776, val_acc: 0.96667
    Epoch [9140/10000], loss: 0.08446 acc: 1.00000 val_loss: 0.15771, val_acc: 0.96667
    Epoch [9150/10000], loss: 0.08440 acc: 1.00000 val_loss: 0.15766, val_acc: 0.96667
    Epoch [9160/10000], loss: 0.08434 acc: 1.00000 val_loss: 0.15761, val_acc: 0.96667
    Epoch [9170/10000], loss: 0.08428 acc: 1.00000 val_loss: 0.15756, val_acc: 0.96667
    Epoch [9180/10000], loss: 0.08422 acc: 1.00000 val_loss: 0.15752, val_acc: 0.96667
    Epoch [9190/10000], loss: 0.08415 acc: 1.00000 val_loss: 0.15747, val_acc: 0.96667
    Epoch [9200/10000], loss: 0.08409 acc: 1.00000 val_loss: 0.15742, val_acc: 0.96667
    Epoch [9210/10000], loss: 0.08403 acc: 1.00000 val_loss: 0.15737, val_acc: 0.96667
    Epoch [9220/10000], loss: 0.08397 acc: 1.00000 val_loss: 0.15733, val_acc: 0.96667
    Epoch [9230/10000], loss: 0.08391 acc: 1.00000 val_loss: 0.15728, val_acc: 0.96667
    Epoch [9240/10000], loss: 0.08385 acc: 1.00000 val_loss: 0.15723, val_acc: 0.96667
    Epoch [9250/10000], loss: 0.08379 acc: 1.00000 val_loss: 0.15718, val_acc: 0.96667
    Epoch [9260/10000], loss: 0.08372 acc: 1.00000 val_loss: 0.15714, val_acc: 0.96667
    Epoch [9270/10000], loss: 0.08366 acc: 1.00000 val_loss: 0.15709, val_acc: 0.96667
    Epoch [9280/10000], loss: 0.08360 acc: 1.00000 val_loss: 0.15704, val_acc: 0.96667
    Epoch [9290/10000], loss: 0.08354 acc: 1.00000 val_loss: 0.15700, val_acc: 0.96667
    Epoch [9300/10000], loss: 0.08348 acc: 1.00000 val_loss: 0.15695, val_acc: 0.96667
    Epoch [9310/10000], loss: 0.08342 acc: 1.00000 val_loss: 0.15690, val_acc: 0.96667
    Epoch [9320/10000], loss: 0.08336 acc: 1.00000 val_loss: 0.15686, val_acc: 0.96667
    Epoch [9330/10000], loss: 0.08330 acc: 1.00000 val_loss: 0.15681, val_acc: 0.96667
    Epoch [9340/10000], loss: 0.08324 acc: 1.00000 val_loss: 0.15676, val_acc: 0.96667
    Epoch [9350/10000], loss: 0.08318 acc: 1.00000 val_loss: 0.15672, val_acc: 0.96667
    Epoch [9360/10000], loss: 0.08312 acc: 1.00000 val_loss: 0.15667, val_acc: 0.96667
    Epoch [9370/10000], loss: 0.08306 acc: 1.00000 val_loss: 0.15662, val_acc: 0.96667
    Epoch [9380/10000], loss: 0.08300 acc: 1.00000 val_loss: 0.15658, val_acc: 0.96667
    Epoch [9390/10000], loss: 0.08294 acc: 1.00000 val_loss: 0.15653, val_acc: 0.96667
    Epoch [9400/10000], loss: 0.08288 acc: 1.00000 val_loss: 0.15649, val_acc: 0.96667
    Epoch [9410/10000], loss: 0.08282 acc: 1.00000 val_loss: 0.15644, val_acc: 0.96667
    Epoch [9420/10000], loss: 0.08276 acc: 1.00000 val_loss: 0.15640, val_acc: 0.96667
    Epoch [9430/10000], loss: 0.08270 acc: 1.00000 val_loss: 0.15635, val_acc: 0.96667
    Epoch [9440/10000], loss: 0.08265 acc: 1.00000 val_loss: 0.15630, val_acc: 0.96667
    Epoch [9450/10000], loss: 0.08259 acc: 1.00000 val_loss: 0.15626, val_acc: 0.96667
    Epoch [9460/10000], loss: 0.08253 acc: 1.00000 val_loss: 0.15621, val_acc: 0.96667
    Epoch [9470/10000], loss: 0.08247 acc: 1.00000 val_loss: 0.15617, val_acc: 0.96667
    Epoch [9480/10000], loss: 0.08241 acc: 1.00000 val_loss: 0.15612, val_acc: 0.96667
    Epoch [9490/10000], loss: 0.08235 acc: 1.00000 val_loss: 0.15608, val_acc: 0.96667
    Epoch [9500/10000], loss: 0.08229 acc: 1.00000 val_loss: 0.15603, val_acc: 0.96667
    Epoch [9510/10000], loss: 0.08224 acc: 1.00000 val_loss: 0.15599, val_acc: 0.96667
    Epoch [9520/10000], loss: 0.08218 acc: 1.00000 val_loss: 0.15594, val_acc: 0.96667
    Epoch [9530/10000], loss: 0.08212 acc: 1.00000 val_loss: 0.15590, val_acc: 0.96667
    Epoch [9540/10000], loss: 0.08206 acc: 1.00000 val_loss: 0.15586, val_acc: 0.96667
    Epoch [9550/10000], loss: 0.08200 acc: 1.00000 val_loss: 0.15581, val_acc: 0.96667
    Epoch [9560/10000], loss: 0.08195 acc: 1.00000 val_loss: 0.15577, val_acc: 0.96667
    Epoch [9570/10000], loss: 0.08189 acc: 1.00000 val_loss: 0.15572, val_acc: 0.96667
    Epoch [9580/10000], loss: 0.08183 acc: 1.00000 val_loss: 0.15568, val_acc: 0.96667
    Epoch [9590/10000], loss: 0.08177 acc: 1.00000 val_loss: 0.15563, val_acc: 0.96667
    Epoch [9600/10000], loss: 0.08172 acc: 1.00000 val_loss: 0.15559, val_acc: 0.96667
    Epoch [9610/10000], loss: 0.08166 acc: 1.00000 val_loss: 0.15555, val_acc: 0.96667
    Epoch [9620/10000], loss: 0.08160 acc: 1.00000 val_loss: 0.15550, val_acc: 0.96667
    Epoch [9630/10000], loss: 0.08154 acc: 1.00000 val_loss: 0.15546, val_acc: 0.96667
    Epoch [9640/10000], loss: 0.08149 acc: 1.00000 val_loss: 0.15542, val_acc: 0.96667
    Epoch [9650/10000], loss: 0.08143 acc: 1.00000 val_loss: 0.15537, val_acc: 0.96667
    Epoch [9660/10000], loss: 0.08137 acc: 1.00000 val_loss: 0.15533, val_acc: 0.96667
    Epoch [9670/10000], loss: 0.08132 acc: 1.00000 val_loss: 0.15529, val_acc: 0.96667
    Epoch [9680/10000], loss: 0.08126 acc: 1.00000 val_loss: 0.15524, val_acc: 0.96667
    Epoch [9690/10000], loss: 0.08120 acc: 1.00000 val_loss: 0.15520, val_acc: 0.96667
    Epoch [9700/10000], loss: 0.08115 acc: 1.00000 val_loss: 0.15516, val_acc: 0.96667
    Epoch [9710/10000], loss: 0.08109 acc: 1.00000 val_loss: 0.15511, val_acc: 0.96667
    Epoch [9720/10000], loss: 0.08103 acc: 1.00000 val_loss: 0.15507, val_acc: 0.96667
    Epoch [9730/10000], loss: 0.08098 acc: 1.00000 val_loss: 0.15503, val_acc: 0.96667
    Epoch [9740/10000], loss: 0.08092 acc: 1.00000 val_loss: 0.15499, val_acc: 0.96667
    Epoch [9750/10000], loss: 0.08087 acc: 1.00000 val_loss: 0.15494, val_acc: 0.96667
    Epoch [9760/10000], loss: 0.08081 acc: 1.00000 val_loss: 0.15490, val_acc: 0.96667
    Epoch [9770/10000], loss: 0.08076 acc: 1.00000 val_loss: 0.15486, val_acc: 0.96667
    Epoch [9780/10000], loss: 0.08070 acc: 1.00000 val_loss: 0.15482, val_acc: 0.96667
    Epoch [9790/10000], loss: 0.08064 acc: 1.00000 val_loss: 0.15477, val_acc: 0.96667
    Epoch [9800/10000], loss: 0.08059 acc: 1.00000 val_loss: 0.15473, val_acc: 0.96667
    Epoch [9810/10000], loss: 0.08053 acc: 1.00000 val_loss: 0.15469, val_acc: 0.96667
    Epoch [9820/10000], loss: 0.08048 acc: 1.00000 val_loss: 0.15465, val_acc: 0.96667
    Epoch [9830/10000], loss: 0.08042 acc: 1.00000 val_loss: 0.15461, val_acc: 0.96667
    Epoch [9840/10000], loss: 0.08037 acc: 1.00000 val_loss: 0.15456, val_acc: 0.96667
    Epoch [9850/10000], loss: 0.08031 acc: 1.00000 val_loss: 0.15452, val_acc: 0.96667
    Epoch [9860/10000], loss: 0.08026 acc: 1.00000 val_loss: 0.15448, val_acc: 0.96667
    Epoch [9870/10000], loss: 0.08020 acc: 1.00000 val_loss: 0.15444, val_acc: 0.96667
    Epoch [9880/10000], loss: 0.08015 acc: 1.00000 val_loss: 0.15440, val_acc: 0.96667
    Epoch [9890/10000], loss: 0.08009 acc: 1.00000 val_loss: 0.15436, val_acc: 0.96667
    Epoch [9900/10000], loss: 0.08004 acc: 1.00000 val_loss: 0.15432, val_acc: 0.96667
    Epoch [9910/10000], loss: 0.07999 acc: 1.00000 val_loss: 0.15427, val_acc: 0.96667
    Epoch [9920/10000], loss: 0.07993 acc: 1.00000 val_loss: 0.15423, val_acc: 0.96667
    Epoch [9930/10000], loss: 0.07988 acc: 1.00000 val_loss: 0.15419, val_acc: 0.96667
    Epoch [9940/10000], loss: 0.07982 acc: 1.00000 val_loss: 0.15415, val_acc: 0.96667
    Epoch [9950/10000], loss: 0.07977 acc: 1.00000 val_loss: 0.15411, val_acc: 0.96667
    Epoch [9960/10000], loss: 0.07972 acc: 1.00000 val_loss: 0.15407, val_acc: 0.96667
    Epoch [9970/10000], loss: 0.07966 acc: 1.00000 val_loss: 0.15403, val_acc: 0.96667
    Epoch [9980/10000], loss: 0.07961 acc: 1.00000 val_loss: 0.15399, val_acc: 0.96667
    Epoch [9990/10000], loss: 0.07955 acc: 1.00000 val_loss: 0.15395, val_acc: 0.96667


### 결과 확인


```python
# 손실과 정확도 확인

print(f'초기 상태 : 손실 : {history[0,3]:.5f}  정확도 : {history[0,4]:.5f}' )
print(f'최종 상태 : 손실 : {history[-1,3]:.5f}  정확도 : {history[-1,4]:.5f}' )
```

    초기 상태 : 손실 : 4.49384  정확도 : 0.50000
    최종 상태 : 손실 : 0.15395  정확도 : 0.96667



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


    
*(이미지 없음 / image not preserved: output_35_0.png)*
    



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

### 결정  경계 그래프 출력



```python
# 검증 데이터 준비

x_t0 = x_test[y_test==0]
x_t1 = x_test[y_test==1]
```


```python
# 파라미터 취득
bias = net.l1.bias.data.numpy()
weight = net.l1.weight.data.numpy()
print(f'BIAS = {bias}, WEIGHT = {weight}')

# 결정 경계를 그리기 위해 x1로부터 x2를 계산
def decision(x):
    return(-(bias + weight[0,0] * x)/ weight[0,1])

# 산포도의 x1의 최솟값과 최댓값
xl = np.array([x_test[:,0].min(), x_test[:,0].max()])
yl = decision(xl)

# 결과 확인
print(f'xl = {xl}  yl = {yl}')
```


```python
# 산포도 출력
plt.scatter(x_t0[:,0], x_t0[:,1], marker='x',
        c='b', s=50, label='class 0')
plt.scatter(x_t1[:,0], x_t1[:,1], marker='o',
        c='k', s=50, label='class 1')

# 결정 경계 직선
plt.plot(xl, yl, c='r')
plt.xlabel('sepal_length')
plt.ylabel('sepal_width')
plt.legend()
plt.show()
```

## 칼럼 BCELoss 함수와 BCEWithLogitsLoss 함수의 차이


```python
# 모델 정의
# 2입력 1출력 로지스틱 회귀 모델

class Net(nn.Module):
    def __init__(self, n_input, n_output):
        super().__init__()
        self.l1 = nn.Linear(n_input, n_output)

        # 초깃값을 모두 1로 함
        # "딥러닝을 위한 수학"과 조건을 맞추기 위한 목적
        self.l1.weight.data.fill_(1.0)
        self.l1.bias.data.fill_(1.0)

    # 예측 함수 정의
    def forward(self, x):
        # 입력 값과 행렬 곱을 계산
        x1 = self.l1(x)
        return x1
```


```python
# 기록용 리스트 초기화
# 학습률
lr = 0.01

# 초기화
net = Net(n_input, n_output)

# 손실 함수 ： logits가 붙은 교차 엔트로피 함수
criterion = nn.BCEWithLogitsLoss()

# 최적화 함수 : 경사 하강법
optimizer = optim.SGD(net.parameters(), lr=lr)

# 반복 횟수
num_epochs = 10000

# 기록용 리스트 초기화
history = np.zeros((0,5))
```


```python
# 반복 계산 메인 루프

for epoch in range(num_epochs):
    # 훈련 페이즈


    # 예측 계산
    outputs = net(inputs)

    # 손실 계산
    loss = criterion(outputs, labels1)

       # 경삿값 초기화
    optimizer.zero_grad()

    # 경사 계산
    loss.backward()

    # 파라미터 수정
    optimizer.step()

    # 손실값 스칼라화
    train_loss = loss.item()

    # 예측 라벨(1 또는 0) 계산
    predicted = torch.where(outputs < 0.0, 0, 1)

    # 정확도 계산
    train_acc = (predicted == labels1).sum() / len(y_train)

    # 예측 페이즈

    # 예측 계산
    outputs_test = net(inputs_test)

    # 손실 계산
    loss_test = criterion(outputs_test, labels1_test)

    # 손실값 스칼라화
    val_loss =  loss_test.item()

    # 예측 라벨(1 또는 0) 계산
    predicted_test = torch.where(outputs_test < 0.0, 0, 1)

    # 정확도 계산
    val_acc = (predicted_test == labels1_test).sum() / len(y_test)

    if ( epoch % 10 == 0):
        print (f'Epoch [{epoch}/{num_epochs}], loss: {train_loss:.5f} acc: {train_acc:.5f} val_loss: {val_loss:.5f}, val_acc: {val_acc:.5f}')
        item = np.array([epoch, train_loss, train_acc, val_loss, val_acc])
        history = np.vstack((history, item))
```


```python
# 손실과 정확도 확인

print(f'초기 상태 : 손실 : {history[0,3]:.5f}  정확도 : {history[0,4]:.5f}' )
print(f'최종 상태 : 손실 : {history[-1,3]:.5f}  정확도 : {history[-1,4]:.5f}' )
```


```python
# 학습 곡선 표시(손실)

plt.plot(history[:,0], history[:,1], 'b', label='훈련')
plt.plot(history[:,0], history[:,3], 'k', label='검증')
plt.xlabel('반복 횟수')
plt.ylabel('손실')
plt.title('학습 곡선(손실)')
plt.legend()
plt.show()
```


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


```python
# 파라미터 취득

bias = net.l1.bias.data.numpy()
weight = net.l1.weight.data.numpy()
print(f'BIAS = {bias}, WEIGHT = {weight}')

# 결정 경계를 그리기 위해 x1로부터 x2를 계산
def decision(x):
    return(-(bias + weight[0,0] * x)/ weight[0,1])

# 산포도의 x1의 최솟값과 최댓값
xl = np.array([x_test[:,0].min(), x_test[:,0].max()])
yl = decision(xl)

# 결과 확인
print(f'xl = {xl}  yl = {yl}')
```


```python
# 산포도 출력
plt.scatter(x_t0[:,0], x_t0[:,1], marker='x',
        c='b', s=50, label='class 0')
plt.scatter(x_t1[:,0], x_t1[:,1], marker='o',
        c='k', s=50, label='class 1')

# 결정 경계 직선
plt.plot(xl, yl, c='r')
plt.xlabel('sepal_length')
plt.ylabel('sepal_width')
plt.legend()
plt.show()
```
