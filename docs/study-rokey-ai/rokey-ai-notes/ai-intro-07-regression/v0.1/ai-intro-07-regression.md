# 7장 선형회귀 (Regression analysis)

> **Migrated from Quartz wiki** (`wiki/study/rokey-ai/basic/ai-intro-07-regression.md`, updated 2026-09-26)  
> tags: rokey, ai, notebook, basic, study  
> source: raw/2026-09-26-rokey-ai/basic/강의_3기_AI개론_7차시__Regression_.ipynb  
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

# Boston dataset
import pandas  as pd
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

## 단순 선형 회귀 분석

### 입력 :1 출력 :1인 선형 함수


```python
# 난수 시드값 고정
torch.manual_seed(123)

# 입력 :1 출력 :1 선형 함수의 정의
l1 = nn.Linear(1, 1)

# 선형 함수 확인
print(l1)
# print(list(l1.parameters()))
print(list(l1.named_parameters()))
```

    Linear(in_features=1, out_features=1, bias=True)
    [('weight', Parameter containing:
    tensor([[-0.4078]], requires_grad=True)), ('bias', Parameter containing:
    tensor([0.0331], requires_grad=True))]



```python
### Check named parameters
name, tensor = list(l1.named_parameters())[0]
print(name, tensor[0], tensor[0].shape)
```

    weight tensor([-0.4078], grad_fn=<SelectBackward0>) torch.Size([1])



```python
# 파라미터명, 파라미터 값, shape 표시

for param in l1.named_parameters():
    print('name: ', param[0])
    print('tensor: ', param[1])
    print('shape: ', param[1].shape)
    print("="*50)
```

    name:  weight
    tensor:  Parameter containing:
    tensor([[-0.4078]], requires_grad=True)
    shape:  torch.Size([1, 1])
    ==================================================
    name:  bias
    tensor:  Parameter containing:
    tensor([0.0331], requires_grad=True)
    shape:  torch.Size([1])
    ==================================================



```python
# 초깃값 설정
nn.init.constant_(l1.weight, 2.0)
nn.init.constant_(l1.bias, 1.0)

# 결과 확인
print(l1.weight)
print(l1.bias)
```

    Parameter containing:
    tensor([[2.]], requires_grad=True)
    Parameter containing:
    tensor([1.], requires_grad=True)



```python
# 테스트용 데이터 생성

# x_np를 넘파이 배열로 정의
x_np = np.arange(-2, 2.1, 1) # float64

# 텐서 변수화
x = torch.tensor(x_np) # float32
print("x = \n", x)

# (N,1) 사이즈로 변경
x = x.view(-1,1)
print("x.view(-1,1) = \n", x)

# 결과 확인
print(x.shape)
print(x)
```

    x = 
     tensor([-2., -1.,  0.,  1.,  2.], dtype=torch.float64)
    x.view(-1,1) = 
     tensor([[-2.],
            [-1.],
            [ 0.],
            [ 1.],
            [ 2.]], dtype=torch.float64)
    torch.Size([5, 1])
    tensor([[-2.],
            [-1.],
            [ 0.],
            [ 1.],
            [ 2.]], dtype=torch.float64)


### 입력 :2 개 출력 :1인 선형 함수


```python
# 입력 :2, 출력:1 선형 함수 정의
l2 = nn.Linear(2, 1)
print("Initial weights and bias", "="*50)
print(l2.weight)
print(l2.bias)
print()

# 초깃값 설정
print("constant weights and bias", "="*50)
nn.init.constant_(l2.weight, 1.0)
nn.init.constant_(l2.bias, 2.0)

# 결과 확인
print(l2.weight)
print(l2.bias)
```

    Initial weights and bias ==================================================
    Parameter containing:
    tensor([[-0.3512,  0.2667]], requires_grad=True)
    Parameter containing:
    tensor([-0.6025], requires_grad=True)
    
    constant weights and bias ==================================================
    Parameter containing:
    tensor([[1., 1.]], requires_grad=True)
    Parameter containing:
    tensor([2.], requires_grad=True)



```python
# 2차원 넘파이 배열
x2_np = np.array([[0, 0], [0, 1], [1, 0], [1,1]])

# 텐서 변수화
x2 =  torch.tensor(x2_np).float()

# 결과 확인
print(x2.shape)
print(x2)

# 함수 값 계산
y2 = l2(x2)

# shape 확인
print(y2.shape)

# 값 확인
print(y2.data)
```

    torch.Size([4, 2])
    tensor([[0., 0.],
            [0., 1.],
            [1., 0.],
            [1., 1.]])
    torch.Size([4, 1])
    tensor([[2.],
            [3.],
            [3.],
            [4.]])


### 입력 :2, 출력 :3 선형 함수 정의


```python
# 입력 :2, 출력 :3 선형 함수 정의

l3 = nn.Linear(2, 3)

# 초깃값 설정
nn.init.constant_(l3.weight[0,:], 1.0)
nn.init.constant_(l3.weight[1,:], 2.0)
nn.init.constant_(l3.weight[2,:], 3.0)
nn.init.constant_(l3.bias, 2.0)

# 결과 확인
print(l3.weight)
print(l3.bias)
```

    Parameter containing:
    tensor([[1., 1.],
            [2., 2.],
            [3., 3.]], requires_grad=True)
    Parameter containing:
    tensor([2., 2., 2.], requires_grad=True)



```python
# 함수 값 계산
y3 = l3(x2)

# shape 확인
print(y3.shape)

# 값 확인
print(y3.data)
```

    torch.Size([4, 3])
    tensor([[2., 2., 2.],
            [3., 4., 5.],
            [3., 4., 5.],
            [4., 6., 8.]])


### 클래스를 이용한 모델 정의


```python
# 모델 정의

class Net(nn.Module):
    def __init__(self, n_input, n_output):
        #  부모 클래스 nn.Module 초기화
        super().__init__()

        # 출력층 정의
        self.l1 = nn.Linear(n_input, n_output)

    # 예측 함수 정의
    def forward(self, x):
        x1 = self.l1(x) # 선형 회귀
        return x1

# 더미 입력
inputs = torch.rand(100,1)
labels1 = torch.rand(100,1)

# 인스턴스 생성(１ 입력, 1 출력 선형 모델)
n_input = 1
n_output = 1
net = Net(n_input, n_output)
```


```python
print(net)
print(net.l1)
print(net.l1.weight)
print(net.l1.bias)

```

    Net(
      (l1): Linear(in_features=1, out_features=1, bias=True)
    )
    Linear(in_features=1, out_features=1, bias=True)
    Parameter containing:
    tensor([[0.2678]], requires_grad=True)
    Parameter containing:
    tensor([-0.2211], requires_grad=True)



```python
# 예측
# torch.matmul(inputs, net.l1.weight) + net.l1.bias
outputs = net(inputs)
print("outputs = \n", outputs)
```

    outputs = 
     tensor([[-0.1364],
            [-0.1135],
            [-0.1894],
            [ 0.0004],
            [-0.1188],
            [-0.0443],
            [ 0.0074],
            [-0.0623],
            [-0.0506],
            [ 0.0420],
            [-0.1476],
            [-0.0448],
            [-0.1468],
            [ 0.0084],
            [ 0.0197],
            [-0.2107],
            [ 0.0270],
            [-0.0233],
            [-0.0289],
            [-0.0321],
            [ 0.0241],
            [-0.1049],
            [-0.2005],
            [-0.1257],
            [-0.1815],
            [-0.0784],
            [-0.1122],
            [-0.1590],
            [-0.0994],
            [ 0.0396],
            [-0.0978],
            [-0.0830],
            [-0.1081],
            [-0.0662],
            [ 0.0321],
            [-0.0054],
            [-0.0397],
            [-0.0581],
            [-0.0557],
            [-0.0355],
            [-0.1045],
            [-0.2117],
            [-0.1700],
            [ 0.0270],
            [-0.0792],
            [-0.1957],
            [-0.0661],
            [ 0.0234],
            [-0.2138],
            [-0.1774],
            [-0.1406],
            [-0.0819],
            [-0.1185],
            [-0.1019],
            [-0.2178],
            [-0.0245],
            [ 0.0303],
            [-0.0054],
            [-0.1820],
            [-0.1952],
            [-0.0316],
            [-0.0842],
            [-0.0324],
            [-0.2181],
            [-0.0952],
            [ 0.0072],
            [-0.0251],
            [-0.0823],
            [-0.0609],
            [-0.0999],
            [-0.1608],
            [-0.1378],
            [-0.1688],
            [ 0.0240],
            [-0.0136],
            [-0.0404],
            [-0.1899],
            [ 0.0161],
            [-0.0453],
            [ 0.0054],
            [-0.1399],
            [-0.0589],
            [ 0.0435],
            [ 0.0028],
            [ 0.0201],
            [-0.1153],
            [ 0.0148],
            [-0.1921],
            [-0.0757],
            [-0.1626],
            [-0.1185],
            [-0.1215],
            [-0.0772],
            [ 0.0346],
            [-0.0210],
            [-0.0878],
            [ 0.0078],
            [-0.1558],
            [-0.0182],
            [-0.0997]], grad_fn=<AddmmBackward0>)


### MSELoss 클래스를 이용한 손실 함수


```python
criterion = nn.MSELoss()
```


```python
loss = criterion(outputs, labels1)
print("loss = ", loss)
loss.backward()
```

    loss =  tensor(0.4184, grad_fn=<MseLossBackward0>)



```python
print(net.l1.weight.grad)
print(net.l1.bias.grad)
```

    tensor([[-0.5785]])
    tensor([-1.1541])


## 회귀 분석 예제: Boston dataset


```python
# 학습용 데이터셋 준비

# '보스턴 데이터셋'은 현재 사이킷런 라이브러리에서 가져올 수 있지만,
# 사이킷런에서 앞으로 이 데이터를 사용할 수 없기 때문에 웹 url에서 직접 수집

#  Variables in order:
#  CRIM     per capita crime rate by town
#  ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
#  INDUS    proportion of non-retail business acres per town
#  CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
#  NOX      nitric oxides concentration (parts per 10 million)
#  RM       average number of rooms per dwelling
#  AGE      proportion of owner-occupied units built prior to 1940
#  DIS      weighted distances to five Boston employment centres
#  RAD      index of accessibility to radial highways
#  TAX      full-value property-tax rate per $10,000
#  PTRATIO  pupil-teacher ratio by town
#  B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
#  LSTAT    % lower status of the population
#  MEDV     Median value of owner-occupied homes in $1000's

# CRIM: 인구당 마을별 범죄율
# ZN: 25,000 평방피트를 초과하는 주거용 토지 비율
# INDUS: 마을별 비소매업 지역 비율
# CHAS: 찰스강 더미 변수 (강과 접한 지역 = 1, 그렇지 않으면 = 0)
# NOX: 질소 산화물 농도 (1000만 분의 1 단위)
# RM: 주택당 평균 방 개수
# AGE: 1940년 이전에 건축된 자가 소유 주택의 비율
# DIS: 보스턴 주요 고용 센터 5곳까지의 가중 거리
# RAD: 방사형 고속도로 접근성 지수
# TAX: $10,000당 재산세율
# PTRATIO: 마을별 학생-교사 비율
# B: 1000(Bk - 0.63)^2, 여기서 Bk는 마을별 흑인 인구 비율
# LSTAT: 저소득층 인구 비율
# MEDV: 자가 소유 주택의 중간값 ($1000 단위)
```


```python

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+",
                     skiprows=22, header=None)
print(raw_df.head(10))

```

              0      1      2    3      4      5     6       7    8      9     10
    0    0.00632  18.00   2.31  0.0  0.538  6.575  65.2  4.0900  1.0  296.0  15.3
    1  396.90000   4.98  24.00  NaN    NaN    NaN   NaN     NaN  NaN    NaN   NaN
    2    0.02731   0.00   7.07  0.0  0.469  6.421  78.9  4.9671  2.0  242.0  17.8
    3  396.90000   9.14  21.60  NaN    NaN    NaN   NaN     NaN  NaN    NaN   NaN
    4    0.02729   0.00   7.07  0.0  0.469  7.185  61.1  4.9671  2.0  242.0  17.8
    5  392.83000   4.03  34.70  NaN    NaN    NaN   NaN     NaN  NaN    NaN   NaN
    6    0.03237   0.00   2.18  0.0  0.458  6.998  45.8  6.0622  3.0  222.0  18.7
    7  394.63000   2.94  33.40  NaN    NaN    NaN   NaN     NaN  NaN    NaN   NaN
    8    0.06905   0.00   2.18  0.0  0.458  7.147  54.2  6.0622  3.0  222.0  18.7
    9  396.90000   5.33  36.20  NaN    NaN    NaN   NaN     NaN  NaN    NaN   NaN



```python

x_org = np.hstack([raw_df.values[::2, :],
                   raw_df.values[1::2, :2]]) # 짝수줄 전체, 홀 수 줄 [: 2] => Features
# x_org[:10, :5]
```


```python
yt = raw_df.values[1::2, 2] ## Target
feature_names = np.array(['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX',
                          'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO','B', 'LSTAT'])

# 결과 확인
print('원본 데이터', x_org.shape, yt.shape)
print('항목명: ', feature_names)
```

    원본 데이터 (506, 13) (506,)
    항목명:  ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'
     'B' 'LSTAT']



```python
x_org[:5]
feature_names == 'RM'
```




    array([False, False, False, False, False,  True, False, False, False,
           False, False, False, False])




```python
# 데이터 추출(RM 항목)
x = x_org[:,feature_names == 'RM']
print('추출 후', x.shape)
print(x[:5,:])

# 정답 데이터 y 표시
print('정답 데이터')
print(yt[:5])
```

    추출 후 (506, 1)
    [[6.575]
     [6.421]
     [7.185]
     [6.998]
     [7.147]]
    정답 데이터
    [24.  21.6 34.7 33.4 36.2]



```python
# 산포도 출력

plt.scatter(x, yt, s=10, c='b')
plt.xlabel('Room counts')
plt.ylabel('Price')
plt.title(' Scatter plot between Room counts vs Price ')
plt.show()
```


    
*(이미지 없음 / image not preserved: output_37_0.png)*
    


### 단순 선형 회귀


```python
## 회귀모델

# 입력 차원수
n_input= x.shape[1]

# 출력 차원수
n_output = 1

print(f'입력 차원수: {n_input}  출력 차원수: {n_output}')

# 머신러닝 모델(예측 모델)의 클래스 정의

class Net(nn.Module):
    def __init__(self, n_input, n_output):
        #  부모 클래스 nn.Module 초기화
        super().__init__()

        # 출력층 정의
        self.l1 = nn.Linear(n_input, n_output)

        # 초깃값을 모두 1로 설정
        # "딥러닝을 위한 수학"과 조건을 맞추기 위함
        # nn.init.constant_(self.l1.weight, 1.0)
        # nn.init.constant_(self.l1.bias, 1.0)

    # 예측 함수 정의
    def forward(self, x):
        x1 = self.l1(x) # 선형 회귀
        return x1
```

    입력 차원수: 1  출력 차원수: 1



```python
# 인스턴스 생성
# １입력 1출력 선형 모델

net = Net(n_input, n_output)

# 모델 안의 파라미터를 확인
for parameter in net.named_parameters():
    print(f'변수명: {parameter[0]}')
    print(f'변숫값: {parameter[1].data}')
print("="*50)
# 파라미터 리스트를 가져오기 위해 parameters 함수를 사용
for parameter in net.parameters():
    print(parameter)
```

    변수명: l1.weight
    변숫값: tensor([[0.4797]])
    변수명: l1.bias
    변숫값: tensor([-0.5425])
    ==================================================
    Parameter containing:
    tensor([[0.4797]], requires_grad=True)
    Parameter containing:
    tensor([-0.5425], requires_grad=True)



```python
print(net)
```

    Net(
      (l1): Linear(in_features=1, out_features=1, bias=True)
    )



```python
# from torchsummary import summary
from torchinfo import summary

summary(net, (1,), device = 'cpu')
```




    ==========================================================================================
    Layer (type:depth-idx)                   Output Shape              Param #
    ==========================================================================================
    Net                                      [1]                       --
    ├─Linear: 1-1                            [1]                       2
    ==========================================================================================
    Total params: 2
    Trainable params: 2
    Non-trainable params: 0
    Total mult-adds (Units.MEGABYTES): 0.00
    ==========================================================================================
    Input size (MB): 0.00
    Forward/backward pass size (MB): 0.00
    Params size (MB): 0.00
    Estimated Total Size (MB): 0.00
    ==========================================================================================




```python
# 손실 함수： 평균 제곱 오차
criterion = nn.MSELoss()

# 학습률
lr = 0.01

# 최적화 함수: 경사 하강법
optimizer = optim.SGD(net.parameters(), lr=lr)
```


```python
# 입력값 x와 정답 yt의 텐서 변수화
# inputs = torch.tensor(x).float()
# labels = torch.tensor(yt).float()
inputs = torch.tensor(x, dtype = torch.float32)
labels = torch.tensor(yt, dtype = torch.float32)

# 차원 수 확인

print(inputs.shape, inputs.dtype)
print(labels.shape, labels.dtype)
```

    torch.Size([506, 1]) torch.float32
    torch.Size([506]) torch.float32



```python
# 손실 계산을 위해 labels를 (N,1) 차원의 행렬로 변환

labels1 = labels.view((-1, 1))

# 차원 수 확인
print("label shape = ", labels1.shape)

# 예측 계산

outputs = net(inputs)
print(outputs.dtype)
print(outputs.dtype)


# 손실 계산
loss = criterion(outputs, labels1)

# 손실 값 가져오기
print(f'{loss.item():.5f}')
```

    label shape =  torch.Size([506, 1])
    torch.float32
    torch.float32
    482.66223



```python
dict(net.named_parameters())
```




    {'l1.weight': Parameter containing:
     tensor([[0.4797]], requires_grad=True),
     'l1.bias': Parameter containing:
     tensor([-0.5425], requires_grad=True)}




```python
# 손실을 그래프로 나타내기
from torchviz import make_dot

g = make_dot(loss, params=dict(net.named_parameters()))
display(g)
```


    
*(이미지 없음 / image not preserved: output_47_0.svg)*
    



```python
# 예측 계산
outputs = net(inputs)

# 손실 계산
loss = criterion(outputs, labels1)

# 경사 계산
loss.backward()

# 경사 계산 결과를 취득 가능하도록 함
print("net.l1.weight.grad = ", net.l1.weight.grad)
print("net.l1.bias.grad = ", net.l1.bias.grad)

# 파라미터 수정
optimizer.step()

# 파라미터 확인
print("="*50)
print(net.l1.weight)
print(net.l1.bias)

# 경삿값 초기화
optimizer.zero_grad()

# 경삿값을 모두 0으로 함
print("="*50)
print(net.l1.weight.grad)
print(net.l1.bias.grad)
```

    net.l1.weight.grad =  tensor([[-260.6448]])
    net.l1.bias.grad =  tensor([-40.1214])
    ==================================================
    Parameter containing:
    tensor([[3.0861]], requires_grad=True)
    Parameter containing:
    tensor([-0.1413], requires_grad=True)
    ==================================================
    None
    None


### 경사 하강법을 이용한 학습


```python
# 학습률
lr = 0.01

# 인스턴스 생성(파라미터 값 초기화)
net = Net(n_input, n_output)

# 손실 함수：평균 제곱 오차
criterion = nn.MSELoss()

# 최적화 함수 : 경사 하강법
optimizer = optim.SGD(net.parameters(), lr=lr)

# 반복 횟수
num_epochs = 50000

# 평가 결과 기록(손실 값만 기록)
history = np.zeros((0,2))
```


```python
# 반복 계산 메인 루프

for epoch in range(num_epochs):

    # 경삿값 초기화
    optimizer.zero_grad()

    # 예측 계산
    outputs = net(inputs)

    # 손실 계산
    # "딥러닝을 위한 수학"에 나온 결과와 맞추기 위해 2로 나눈 값을 손실로 정의
    loss = criterion(outputs, labels1) / 2.0

    # 경사 계산
    loss.backward()

    # 파라미터 수정
    optimizer.step()

    # 100회 마다 도중 경과를 기록
    if ( epoch % 100 == 0):
        history = np.vstack((history, np.array([epoch, loss.item()])))
        print(f'Epoch {epoch} loss: {loss.item():.5f}')
```

    Epoch 0 loss: 299.93127
    Epoch 100 loss: 28.92968
    Epoch 200 loss: 28.76028
    Epoch 300 loss: 28.59489
    Epoch 400 loss: 28.43344
    Epoch 500 loss: 28.27583
    Epoch 600 loss: 28.12196
    Epoch 700 loss: 27.97174
    Epoch 800 loss: 27.82510
    Epoch 900 loss: 27.68194
    Epoch 1000 loss: 27.54218
    Epoch 1100 loss: 27.40574
    Epoch 1200 loss: 27.27254
    Epoch 1300 loss: 27.14251
    Epoch 1400 loss: 27.01557
    Epoch 1500 loss: 26.89165
    Epoch 1600 loss: 26.77067
    Epoch 1700 loss: 26.65256
    Epoch 1800 loss: 26.53726
    Epoch 1900 loss: 26.42470
    Epoch 2000 loss: 26.31482
    Epoch 2100 loss: 26.20755
    Epoch 2200 loss: 26.10282
    Epoch 2300 loss: 26.00059
    Epoch 2400 loss: 25.90078
    Epoch 2500 loss: 25.80334
    Epoch 2600 loss: 25.70822
    Epoch 2700 loss: 25.61536
    Epoch 2800 loss: 25.52471
    Epoch 2900 loss: 25.43621
    Epoch 3000 loss: 25.34982
    Epoch 3100 loss: 25.26547
    Epoch 3200 loss: 25.18313
    Epoch 3300 loss: 25.10275
    Epoch 3400 loss: 25.02428
    Epoch 3500 loss: 24.94767
    Epoch 3600 loss: 24.87288
    Epoch 3700 loss: 24.79987
    Epoch 3800 loss: 24.72860
    Epoch 3900 loss: 24.65902
    Epoch 4000 loss: 24.59109
    Epoch 4100 loss: 24.52477
    Epoch 4200 loss: 24.46003
    Epoch 4300 loss: 24.39683
    Epoch 4400 loss: 24.33513
    Epoch 4500 loss: 24.27490
    Epoch 4600 loss: 24.21610
    Epoch 4700 loss: 24.15870
    Epoch 4800 loss: 24.10266
    Epoch 4900 loss: 24.04795
    Epoch 5000 loss: 23.99454
    Epoch 5100 loss: 23.94240
    Epoch 5200 loss: 23.89150
    Epoch 5300 loss: 23.84181
    Epoch 5400 loss: 23.79330
    Epoch 5500 loss: 23.74594
    Epoch 5600 loss: 23.69971
    Epoch 5700 loss: 23.65458
    Epoch 5800 loss: 23.61051
    Epoch 5900 loss: 23.56750
    Epoch 6000 loss: 23.52551
    Epoch 6100 loss: 23.48451
    Epoch 6200 loss: 23.44449
    Epoch 6300 loss: 23.40542
    Epoch 6400 loss: 23.36728
    Epoch 6500 loss: 23.33005
    Epoch 6600 loss: 23.29370
    Epoch 6700 loss: 23.25821
    Epoch 6800 loss: 23.22357
    Epoch 6900 loss: 23.18975
    Epoch 7000 loss: 23.15673
    Epoch 7100 loss: 23.12450
    Epoch 7200 loss: 23.09304
    Epoch 7300 loss: 23.06232
    Epoch 7400 loss: 23.03233
    Epoch 7500 loss: 23.00305
    Epoch 7600 loss: 22.97447
    Epoch 7700 loss: 22.94658
    Epoch 7800 loss: 22.91933
    Epoch 7900 loss: 22.89275
    Epoch 8000 loss: 22.86679
    Epoch 8100 loss: 22.84145
    Epoch 8200 loss: 22.81671
    Epoch 8300 loss: 22.79255
    Epoch 8400 loss: 22.76898
    Epoch 8500 loss: 22.74596
    Epoch 8600 loss: 22.72349
    Epoch 8700 loss: 22.70155
    Epoch 8800 loss: 22.68013
    Epoch 8900 loss: 22.65923
    Epoch 9000 loss: 22.63881
    Epoch 9100 loss: 22.61889
    Epoch 9200 loss: 22.59944
    Epoch 9300 loss: 22.58045
    Epoch 9400 loss: 22.56191
    Epoch 9500 loss: 22.54381
    Epoch 9600 loss: 22.52615
    Epoch 9700 loss: 22.50890
    Epoch 9800 loss: 22.49206
    Epoch 9900 loss: 22.47563
    Epoch 10000 loss: 22.45958
    Epoch 10100 loss: 22.44391
    Epoch 10200 loss: 22.42862
    Epoch 10300 loss: 22.41369
    Epoch 10400 loss: 22.39911
    Epoch 10500 loss: 22.38488
    Epoch 10600 loss: 22.37099
    Epoch 10700 loss: 22.35743
    Epoch 10800 loss: 22.34419
    Epoch 10900 loss: 22.33126
    Epoch 11000 loss: 22.31865
    Epoch 11100 loss: 22.30633
    Epoch 11200 loss: 22.29431
    Epoch 11300 loss: 22.28257
    Epoch 11400 loss: 22.27111
    Epoch 11500 loss: 22.25992
    Epoch 11600 loss: 22.24900
    Epoch 11700 loss: 22.23834
    Epoch 11800 loss: 22.22793
    Epoch 11900 loss: 22.21777
    Epoch 12000 loss: 22.20785
    Epoch 12100 loss: 22.19816
    Epoch 12200 loss: 22.18871
    Epoch 12300 loss: 22.17948
    Epoch 12400 loss: 22.17047
    Epoch 12500 loss: 22.16167
    Epoch 12600 loss: 22.15308
    Epoch 12700 loss: 22.14470
    Epoch 12800 loss: 22.13652
    Epoch 12900 loss: 22.12853
    Epoch 13000 loss: 22.12073
    Epoch 13100 loss: 22.11312
    Epoch 13200 loss: 22.10568
    Epoch 13300 loss: 22.09842
    Epoch 13400 loss: 22.09134
    Epoch 13500 loss: 22.08442
    Epoch 13600 loss: 22.07767
    Epoch 13700 loss: 22.07108
    Epoch 13800 loss: 22.06465
    Epoch 13900 loss: 22.05836
    Epoch 14000 loss: 22.05223
    Epoch 14100 loss: 22.04624
    Epoch 14200 loss: 22.04040
    Epoch 14300 loss: 22.03469
    Epoch 14400 loss: 22.02912
    Epoch 14500 loss: 22.02369
    Epoch 14600 loss: 22.01838
    Epoch 14700 loss: 22.01319
    Epoch 14800 loss: 22.00814
    Epoch 14900 loss: 22.00320
    Epoch 15000 loss: 21.99837
    Epoch 15100 loss: 21.99367
    Epoch 15200 loss: 21.98907
    Epoch 15300 loss: 21.98459
    Epoch 15400 loss: 21.98021
    Epoch 15500 loss: 21.97593
    Epoch 15600 loss: 21.97176
    Epoch 15700 loss: 21.96769
    Epoch 15800 loss: 21.96371
    Epoch 15900 loss: 21.95982
    Epoch 16000 loss: 21.95603
    Epoch 16100 loss: 21.95233
    Epoch 16200 loss: 21.94872
    Epoch 16300 loss: 21.94519
    Epoch 16400 loss: 21.94175
    Epoch 16500 loss: 21.93839
    Epoch 16600 loss: 21.93510
    Epoch 16700 loss: 21.93190
    Epoch 16800 loss: 21.92877
    Epoch 16900 loss: 21.92572
    Epoch 17000 loss: 21.92274
    Epoch 17100 loss: 21.91983
    Epoch 17200 loss: 21.91699
    Epoch 17300 loss: 21.91422
    Epoch 17400 loss: 21.91151
    Epoch 17500 loss: 21.90886
    Epoch 17600 loss: 21.90629
    Epoch 17700 loss: 21.90377
    Epoch 17800 loss: 21.90131
    Epoch 17900 loss: 21.89891
    Epoch 18000 loss: 21.89656
    Epoch 18100 loss: 21.89428
    Epoch 18200 loss: 21.89205
    Epoch 18300 loss: 21.88986
    Epoch 18400 loss: 21.88773
    Epoch 18500 loss: 21.88565
    Epoch 18600 loss: 21.88363
    Epoch 18700 loss: 21.88165
    Epoch 18800 loss: 21.87971
    Epoch 18900 loss: 21.87782
    Epoch 19000 loss: 21.87598
    Epoch 19100 loss: 21.87419
    Epoch 19200 loss: 21.87243
    Epoch 19300 loss: 21.87071
    Epoch 19400 loss: 21.86904
    Epoch 19500 loss: 21.86740
    Epoch 19600 loss: 21.86581
    Epoch 19700 loss: 21.86425
    Epoch 19800 loss: 21.86274
    Epoch 19900 loss: 21.86125
    Epoch 20000 loss: 21.85980
    Epoch 20100 loss: 21.85839
    Epoch 20200 loss: 21.85701
    Epoch 20300 loss: 21.85566
    Epoch 20400 loss: 21.85434
    Epoch 20500 loss: 21.85306
    Epoch 20600 loss: 21.85180
    Epoch 20700 loss: 21.85058
    Epoch 20800 loss: 21.84939
    Epoch 20900 loss: 21.84822
    Epoch 21000 loss: 21.84708
    Epoch 21100 loss: 21.84597
    Epoch 21200 loss: 21.84488
    Epoch 21300 loss: 21.84382
    Epoch 21400 loss: 21.84278
    Epoch 21500 loss: 21.84177
    Epoch 21600 loss: 21.84079
    Epoch 21700 loss: 21.83983
    Epoch 21800 loss: 21.83889
    Epoch 21900 loss: 21.83797
    Epoch 22000 loss: 21.83707
    Epoch 22100 loss: 21.83620
    Epoch 22200 loss: 21.83535
    Epoch 22300 loss: 21.83451
    Epoch 22400 loss: 21.83370
    Epoch 22500 loss: 21.83291
    Epoch 22600 loss: 21.83213
    Epoch 22700 loss: 21.83138
    Epoch 22800 loss: 21.83063
    Epoch 22900 loss: 21.82991
    Epoch 23000 loss: 21.82921
    Epoch 23100 loss: 21.82852
    Epoch 23200 loss: 21.82785
    Epoch 23300 loss: 21.82720
    Epoch 23400 loss: 21.82655
    Epoch 23500 loss: 21.82593
    Epoch 23600 loss: 21.82532
    Epoch 23700 loss: 21.82473
    Epoch 23800 loss: 21.82415
    Epoch 23900 loss: 21.82358
    Epoch 24000 loss: 21.82302
    Epoch 24100 loss: 21.82249
    Epoch 24200 loss: 21.82196
    Epoch 24300 loss: 21.82144
    Epoch 24400 loss: 21.82094
    Epoch 24500 loss: 21.82045
    Epoch 24600 loss: 21.81997
    Epoch 24700 loss: 21.81950
    Epoch 24800 loss: 21.81904
    Epoch 24900 loss: 21.81860
    Epoch 25000 loss: 21.81817
    Epoch 25100 loss: 21.81774
    Epoch 25200 loss: 21.81732
    Epoch 25300 loss: 21.81692
    Epoch 25400 loss: 21.81652
    Epoch 25500 loss: 21.81614
    Epoch 25600 loss: 21.81576
    Epoch 25700 loss: 21.81540
    Epoch 25800 loss: 21.81503
    Epoch 25900 loss: 21.81468
    Epoch 26000 loss: 21.81434
    Epoch 26100 loss: 21.81401
    Epoch 26200 loss: 21.81368
    Epoch 26300 loss: 21.81336
    Epoch 26400 loss: 21.81305
    Epoch 26500 loss: 21.81275
    Epoch 26600 loss: 21.81245
    Epoch 26700 loss: 21.81216
    Epoch 26800 loss: 21.81188
    Epoch 26900 loss: 21.81161
    Epoch 27000 loss: 21.81133
    Epoch 27100 loss: 21.81107
    Epoch 27200 loss: 21.81082
    Epoch 27300 loss: 21.81056
    Epoch 27400 loss: 21.81032
    Epoch 27500 loss: 21.81008
    Epoch 27600 loss: 21.80985
    Epoch 27700 loss: 21.80962
    Epoch 27800 loss: 21.80940
    Epoch 27900 loss: 21.80918
    Epoch 28000 loss: 21.80897
    Epoch 28100 loss: 21.80877
    Epoch 28200 loss: 21.80856
    Epoch 28300 loss: 21.80837
    Epoch 28400 loss: 21.80817
    Epoch 28500 loss: 21.80799
    Epoch 28600 loss: 21.80781
    Epoch 28700 loss: 21.80762
    Epoch 28800 loss: 21.80745
    Epoch 28900 loss: 21.80728
    Epoch 29000 loss: 21.80712
    Epoch 29100 loss: 21.80695
    Epoch 29200 loss: 21.80679
    Epoch 29300 loss: 21.80664
    Epoch 29400 loss: 21.80649
    Epoch 29500 loss: 21.80634
    Epoch 29600 loss: 21.80619
    Epoch 29700 loss: 21.80605
    Epoch 29800 loss: 21.80592
    Epoch 29900 loss: 21.80578
    Epoch 30000 loss: 21.80565
    Epoch 30100 loss: 21.80552
    Epoch 30200 loss: 21.80540
    Epoch 30300 loss: 21.80528
    Epoch 30400 loss: 21.80516
    Epoch 30500 loss: 21.80504
    Epoch 30600 loss: 21.80493
    Epoch 30700 loss: 21.80482
    Epoch 30800 loss: 21.80471
    Epoch 30900 loss: 21.80461
    Epoch 31000 loss: 21.80450
    Epoch 31100 loss: 21.80440
    Epoch 31200 loss: 21.80431
    Epoch 31300 loss: 21.80421
    Epoch 31400 loss: 21.80411
    Epoch 31500 loss: 21.80403
    Epoch 31600 loss: 21.80394
    Epoch 31700 loss: 21.80385
    Epoch 31800 loss: 21.80376
    Epoch 31900 loss: 21.80368
    Epoch 32000 loss: 21.80360
    Epoch 32100 loss: 21.80352
    Epoch 32200 loss: 21.80344
    Epoch 32300 loss: 21.80337
    Epoch 32400 loss: 21.80330
    Epoch 32500 loss: 21.80322
    Epoch 32600 loss: 21.80315
    Epoch 32700 loss: 21.80309
    Epoch 32800 loss: 21.80302
    Epoch 32900 loss: 21.80295
    Epoch 33000 loss: 21.80289
    Epoch 33100 loss: 21.80283
    Epoch 33200 loss: 21.80277
    Epoch 33300 loss: 21.80271
    Epoch 33400 loss: 21.80265
    Epoch 33500 loss: 21.80259
    Epoch 33600 loss: 21.80254
    Epoch 33700 loss: 21.80249
    Epoch 33800 loss: 21.80243
    Epoch 33900 loss: 21.80238
    Epoch 34000 loss: 21.80233
    Epoch 34100 loss: 21.80228
    Epoch 34200 loss: 21.80224
    Epoch 34300 loss: 21.80219
    Epoch 34400 loss: 21.80214
    Epoch 34500 loss: 21.80210
    Epoch 34600 loss: 21.80206
    Epoch 34700 loss: 21.80202
    Epoch 34800 loss: 21.80197
    Epoch 34900 loss: 21.80193
    Epoch 35000 loss: 21.80189
    Epoch 35100 loss: 21.80186
    Epoch 35200 loss: 21.80181
    Epoch 35300 loss: 21.80178
    Epoch 35400 loss: 21.80174
    Epoch 35500 loss: 21.80171
    Epoch 35600 loss: 21.80168
    Epoch 35700 loss: 21.80164
    Epoch 35800 loss: 21.80161
    Epoch 35900 loss: 21.80158
    Epoch 36000 loss: 21.80155
    Epoch 36100 loss: 21.80152
    Epoch 36200 loss: 21.80149
    Epoch 36300 loss: 21.80146
    Epoch 36400 loss: 21.80143
    Epoch 36500 loss: 21.80140
    Epoch 36600 loss: 21.80138
    Epoch 36700 loss: 21.80135
    Epoch 36800 loss: 21.80132
    Epoch 36900 loss: 21.80130
    Epoch 37000 loss: 21.80128
    Epoch 37100 loss: 21.80125
    Epoch 37200 loss: 21.80123
    Epoch 37300 loss: 21.80120
    Epoch 37400 loss: 21.80119
    Epoch 37500 loss: 21.80116
    Epoch 37600 loss: 21.80114
    Epoch 37700 loss: 21.80112
    Epoch 37800 loss: 21.80110
    Epoch 37900 loss: 21.80108
    Epoch 38000 loss: 21.80106
    Epoch 38100 loss: 21.80105
    Epoch 38200 loss: 21.80102
    Epoch 38300 loss: 21.80101
    Epoch 38400 loss: 21.80099
    Epoch 38500 loss: 21.80097
    Epoch 38600 loss: 21.80095
    Epoch 38700 loss: 21.80094
    Epoch 38800 loss: 21.80092
    Epoch 38900 loss: 21.80091
    Epoch 39000 loss: 21.80090
    Epoch 39100 loss: 21.80088
    Epoch 39200 loss: 21.80087
    Epoch 39300 loss: 21.80085
    Epoch 39400 loss: 21.80083
    Epoch 39500 loss: 21.80083
    Epoch 39600 loss: 21.80081
    Epoch 39700 loss: 21.80080
    Epoch 39800 loss: 21.80079
    Epoch 39900 loss: 21.80078
    Epoch 40000 loss: 21.80076
    Epoch 40100 loss: 21.80075
    Epoch 40200 loss: 21.80074
    Epoch 40300 loss: 21.80073
    Epoch 40400 loss: 21.80072
    Epoch 40500 loss: 21.80071
    Epoch 40600 loss: 21.80070
    Epoch 40700 loss: 21.80069
    Epoch 40800 loss: 21.80068
    Epoch 40900 loss: 21.80067
    Epoch 41000 loss: 21.80066
    Epoch 41100 loss: 21.80065
    Epoch 41200 loss: 21.80064
    Epoch 41300 loss: 21.80063
    Epoch 41400 loss: 21.80062
    Epoch 41500 loss: 21.80062
    Epoch 41600 loss: 21.80061
    Epoch 41700 loss: 21.80060
    Epoch 41800 loss: 21.80059
    Epoch 41900 loss: 21.80058
    Epoch 42000 loss: 21.80058
    Epoch 42100 loss: 21.80057
    Epoch 42200 loss: 21.80056
    Epoch 42300 loss: 21.80056
    Epoch 42400 loss: 21.80055
    Epoch 42500 loss: 21.80054
    Epoch 42600 loss: 21.80054
    Epoch 42700 loss: 21.80053
    Epoch 42800 loss: 21.80052
    Epoch 42900 loss: 21.80052
    Epoch 43000 loss: 21.80051
    Epoch 43100 loss: 21.80051
    Epoch 43200 loss: 21.80050
    Epoch 43300 loss: 21.80050
    Epoch 43400 loss: 21.80049
    Epoch 43500 loss: 21.80049
    Epoch 43600 loss: 21.80048
    Epoch 43700 loss: 21.80047
    Epoch 43800 loss: 21.80047
    Epoch 43900 loss: 21.80047
    Epoch 44000 loss: 21.80046
    Epoch 44100 loss: 21.80046
    Epoch 44200 loss: 21.80045
    Epoch 44300 loss: 21.80045
    Epoch 44400 loss: 21.80045
    Epoch 44500 loss: 21.80044
    Epoch 44600 loss: 21.80044
    Epoch 44700 loss: 21.80043
    Epoch 44800 loss: 21.80043
    Epoch 44900 loss: 21.80043
    Epoch 45000 loss: 21.80042
    Epoch 45100 loss: 21.80042
    Epoch 45200 loss: 21.80042
    Epoch 45300 loss: 21.80041
    Epoch 45400 loss: 21.80041
    Epoch 45500 loss: 21.80041
    Epoch 45600 loss: 21.80040
    Epoch 45700 loss: 21.80040
    Epoch 45800 loss: 21.80040
    Epoch 45900 loss: 21.80040
    Epoch 46000 loss: 21.80039
    Epoch 46100 loss: 21.80039
    Epoch 46200 loss: 21.80039
    Epoch 46300 loss: 21.80038
    Epoch 46400 loss: 21.80038
    Epoch 46500 loss: 21.80038
    Epoch 46600 loss: 21.80038
    Epoch 46700 loss: 21.80037
    Epoch 46800 loss: 21.80037
    Epoch 46900 loss: 21.80037
    Epoch 47000 loss: 21.80037
    Epoch 47100 loss: 21.80036
    Epoch 47200 loss: 21.80036
    Epoch 47300 loss: 21.80036
    Epoch 47400 loss: 21.80036
    Epoch 47500 loss: 21.80035
    Epoch 47600 loss: 21.80035
    Epoch 47700 loss: 21.80035
    Epoch 47800 loss: 21.80035
    Epoch 47900 loss: 21.80035
    Epoch 48000 loss: 21.80035
    Epoch 48100 loss: 21.80035
    Epoch 48200 loss: 21.80034
    Epoch 48300 loss: 21.80034
    Epoch 48400 loss: 21.80034
    Epoch 48500 loss: 21.80034
    Epoch 48600 loss: 21.80034
    Epoch 48700 loss: 21.80034
    Epoch 48800 loss: 21.80034
    Epoch 48900 loss: 21.80033
    Epoch 49000 loss: 21.80033
    Epoch 49100 loss: 21.80033
    Epoch 49200 loss: 21.80033
    Epoch 49300 loss: 21.80033
    Epoch 49400 loss: 21.80033
    Epoch 49500 loss: 21.80033
    Epoch 49600 loss: 21.80033
    Epoch 49700 loss: 21.80032
    Epoch 49800 loss: 21.80032
    Epoch 49900 loss: 21.80032



```python
# 초기 손실값과 최종 손실값

print(f'초기 손실값: {history[0,1]:.5f}')
print(f'최종 손실값: {history[-1,1]:.5f}')
```

    초기 손실값: 299.93127
    최종 손실값: 21.80032



```python
# 학습 곡선 출력(손실)
# 가장 처음 요소는 제외

plt.plot(history[1:,0], history[1:,1], 'b')
plt.xlabel('반복 횟수')
plt.ylabel('손실')
plt.title('학습 곡선(손실)')
plt.show()
```


    
*(이미지 없음 / image not preserved: output_53_0.png)*
    



```python
# 회귀 직선 산출

# x의 최솟값, 최댓값
xse = np.array((x.min(), x.max())).reshape(-1,1)
Xse = torch.tensor(xse).float()

with torch.no_grad():
  Yse = net(Xse)

print(Yse.numpy())

# 산포도와 회귀 직선 출력

plt.scatter(x, yt, s=10, c='b')
plt.xlabel('방 개수')
plt.ylabel('가격')
plt.plot(Xse.data, Yse.data, c='k')
plt.title('산포도와 회귀 직선')
plt.show()
```

    [[-2.2208]
     [45.2137]]



    
*(이미지 없음 / image not preserved: output_54_1.png)*
    


### 중회귀 모델


```python
# 열(LSTAT: 저소득자 비율) 추가

x_add = x_org[:,feature_names == 'LSTAT']
x2 = np.hstack((x, x_add))

# shape 표시
print(x2.shape)

# 입력 데이터 x 표시
print(x2[:5,:])
```

    (506, 2)
    [[6.575 4.98 ]
     [6.421 9.14 ]
     [7.185 4.03 ]
     [6.998 2.94 ]
     [7.147 5.33 ]]



```python
# 입력 차원수=2

n_input = x2.shape[1]
print(n_input)

# 모델 인스턴스 생성
net = Net(n_input, n_output)
```

    2



```python
# 모델 안의 파라미터 확인
# predict.weight가 2차원으로 바뀜

for parameter in net.named_parameters():
    print(f'변수명: {parameter[0]}')
    print(f'변숫값: {parameter[1].data}')
```

    변수명: l1.weight
    변숫값: tensor([[-0.5142,  0.2712]])
    변수명: l1.bias
    변숫값: tensor([-0.2058])



```python
# 모델의 개요 표시

from torchinfo import summary
summary(net, (2,))
```




    ==========================================================================================
    Layer (type:depth-idx)                   Output Shape              Param #
    ==========================================================================================
    Net                                      [1]                       --
    ├─Linear: 1-1                            [1]                       3
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




```python
# 입력 변수 x2를 텐서로 변환
# labels, labels1은 이전과 같음

# inputs = torch.tensor(x2).float()
inputs = torch.tensor(x2, dtype = torch.float32)

# 초기화 처리

# 학습률
# lr = 0.01
lr = 0.001


# 인스턴스 생성(파라미터 값 초기화)
net = Net(n_input, n_output)

# 손실 함수：평균 제곱 오차
criterion = nn.MSELoss()

# 최적화 함수 : 경사 하강법
optimizer = optim.SGD(net.parameters(), lr=lr)

# 반복 횟수
num_epochs = 50000

# 평가 결과 기록(손실 값만 기록)
history = np.zeros((0,2))

```


```python
# 반복 계산 메인 루프

for epoch in range(num_epochs):

    # 경삿값 초기화
    optimizer.zero_grad()

    # 예측 계산
    outputs = net(inputs)

    # 오차 계산
    # "딥러닝을 위한 수학"에 나온 결과와 맞추기 위해 2로 나눈 값을 손실로 정의
    loss = criterion(outputs, labels1) / 2.0

    # 경사 계산
    loss.backward()

    # 파라미터 수정
    optimizer.step()

    # 100회 마다 도중 경과를 기록
    if ( epoch % 100 == 0):
        history = np.vstack((history, np.array([epoch, loss.item()])))
        print(f'Epoch {epoch} loss: {loss.item():.5f}')
```

    Epoch 0 loss: 529.79584
    Epoch 100 loss: 29.48064
    Epoch 200 loss: 16.83812
    Epoch 300 loss: 15.44524
    Epoch 400 loss: 15.29177
    Epoch 500 loss: 15.27485
    Epoch 600 loss: 15.27296
    Epoch 700 loss: 15.27274
    Epoch 800 loss: 15.27270
    Epoch 900 loss: 15.27268
    Epoch 1000 loss: 15.27265
    Epoch 1100 loss: 15.27264
    Epoch 1200 loss: 15.27262
    Epoch 1300 loss: 15.27260
    Epoch 1400 loss: 15.27258
    Epoch 1500 loss: 15.27256
    Epoch 1600 loss: 15.27254
    Epoch 1700 loss: 15.27252
    Epoch 1800 loss: 15.27250
    Epoch 1900 loss: 15.27248
    Epoch 2000 loss: 15.27246
    Epoch 2100 loss: 15.27244
    Epoch 2200 loss: 15.27242
    Epoch 2300 loss: 15.27240
    Epoch 2400 loss: 15.27239
    Epoch 2500 loss: 15.27237
    Epoch 2600 loss: 15.27235
    Epoch 2700 loss: 15.27233
    Epoch 2800 loss: 15.27231
    Epoch 2900 loss: 15.27229
    Epoch 3000 loss: 15.27227
    Epoch 3100 loss: 15.27225
    Epoch 3200 loss: 15.27223
    Epoch 3300 loss: 15.27222
    Epoch 3400 loss: 15.27220
    Epoch 3500 loss: 15.27217
    Epoch 3600 loss: 15.27216
    Epoch 3700 loss: 15.27214
    Epoch 3800 loss: 15.27212
    Epoch 3900 loss: 15.27210
    Epoch 4000 loss: 15.27208
    Epoch 4100 loss: 15.27207
    Epoch 4200 loss: 15.27205
    Epoch 4300 loss: 15.27203
    Epoch 4400 loss: 15.27201
    Epoch 4500 loss: 15.27199
    Epoch 4600 loss: 15.27197
    Epoch 4700 loss: 15.27195
    Epoch 4800 loss: 15.27193
    Epoch 4900 loss: 15.27192
    Epoch 5000 loss: 15.27189
    Epoch 5100 loss: 15.27188
    Epoch 5200 loss: 15.27186
    Epoch 5300 loss: 15.27184
    Epoch 5400 loss: 15.27182
    Epoch 5500 loss: 15.27180
    Epoch 5600 loss: 15.27179
    Epoch 5700 loss: 15.27177
    Epoch 5800 loss: 15.27175
    Epoch 5900 loss: 15.27173
    Epoch 6000 loss: 15.27171
    Epoch 6100 loss: 15.27170
    Epoch 6200 loss: 15.27168
    Epoch 6300 loss: 15.27166
    Epoch 6400 loss: 15.27164
    Epoch 6500 loss: 15.27162
    Epoch 6600 loss: 15.27160
    Epoch 6700 loss: 15.27158
    Epoch 6800 loss: 15.27157
    Epoch 6900 loss: 15.27155
    Epoch 7000 loss: 15.27153
    Epoch 7100 loss: 15.27151
    Epoch 7200 loss: 15.27149
    Epoch 7300 loss: 15.27148
    Epoch 7400 loss: 15.27146
    Epoch 7500 loss: 15.27144
    Epoch 7600 loss: 15.27142
    Epoch 7700 loss: 15.27140
    Epoch 7800 loss: 15.27139
    Epoch 7900 loss: 15.27137
    Epoch 8000 loss: 15.27135
    Epoch 8100 loss: 15.27133
    Epoch 8200 loss: 15.27131
    Epoch 8300 loss: 15.27129
    Epoch 8400 loss: 15.27128
    Epoch 8500 loss: 15.27126
    Epoch 8600 loss: 15.27124
    Epoch 8700 loss: 15.27122
    Epoch 8800 loss: 15.27121
    Epoch 8900 loss: 15.27119
    Epoch 9000 loss: 15.27117
    Epoch 9100 loss: 15.27115
    Epoch 9200 loss: 15.27114
    Epoch 9300 loss: 15.27112
    Epoch 9400 loss: 15.27110
    Epoch 9500 loss: 15.27109
    Epoch 9600 loss: 15.27107
    Epoch 9700 loss: 15.27105
    Epoch 9800 loss: 15.27103
    Epoch 9900 loss: 15.27101
    Epoch 10000 loss: 15.27100
    Epoch 10100 loss: 15.27098
    Epoch 10200 loss: 15.27096
    Epoch 10300 loss: 15.27094
    Epoch 10400 loss: 15.27093
    Epoch 10500 loss: 15.27091
    Epoch 10600 loss: 15.27089
    Epoch 10700 loss: 15.27087
    Epoch 10800 loss: 15.27085
    Epoch 10900 loss: 15.27084
    Epoch 11000 loss: 15.27082
    Epoch 11100 loss: 15.27080
    Epoch 11200 loss: 15.27079
    Epoch 11300 loss: 15.27077
    Epoch 11400 loss: 15.27075
    Epoch 11500 loss: 15.27073
    Epoch 11600 loss: 15.27072
    Epoch 11700 loss: 15.27070
    Epoch 11800 loss: 15.27068
    Epoch 11900 loss: 15.27067
    Epoch 12000 loss: 15.27065
    Epoch 12100 loss: 15.27063
    Epoch 12200 loss: 15.27061
    Epoch 12300 loss: 15.27060
    Epoch 12400 loss: 15.27058
    Epoch 12500 loss: 15.27057
    Epoch 12600 loss: 15.27055
    Epoch 12700 loss: 15.27053
    Epoch 12800 loss: 15.27051
    Epoch 12900 loss: 15.27050
    Epoch 13000 loss: 15.27048
    Epoch 13100 loss: 15.27047
    Epoch 13200 loss: 15.27045
    Epoch 13300 loss: 15.27043
    Epoch 13400 loss: 15.27041
    Epoch 13500 loss: 15.27040
    Epoch 13600 loss: 15.27038
    Epoch 13700 loss: 15.27036
    Epoch 13800 loss: 15.27035
    Epoch 13900 loss: 15.27033
    Epoch 14000 loss: 15.27032
    Epoch 14100 loss: 15.27029
    Epoch 14200 loss: 15.27028
    Epoch 14300 loss: 15.27026
    Epoch 14400 loss: 15.27025
    Epoch 14500 loss: 15.27023
    Epoch 14600 loss: 15.27021
    Epoch 14700 loss: 15.27020
    Epoch 14800 loss: 15.27018
    Epoch 14900 loss: 15.27017
    Epoch 15000 loss: 15.27015
    Epoch 15100 loss: 15.27013
    Epoch 15200 loss: 15.27011
    Epoch 15300 loss: 15.27010
    Epoch 15400 loss: 15.27008
    Epoch 15500 loss: 15.27007
    Epoch 15600 loss: 15.27005
    Epoch 15700 loss: 15.27003
    Epoch 15800 loss: 15.27002
    Epoch 15900 loss: 15.27000
    Epoch 16000 loss: 15.26999
    Epoch 16100 loss: 15.26997
    Epoch 16200 loss: 15.26995
    Epoch 16300 loss: 15.26993
    Epoch 16400 loss: 15.26992
    Epoch 16500 loss: 15.26990
    Epoch 16600 loss: 15.26989
    Epoch 16700 loss: 15.26987
    Epoch 16800 loss: 15.26986
    Epoch 16900 loss: 15.26984
    Epoch 17000 loss: 15.26982
    Epoch 17100 loss: 15.26981
    Epoch 17200 loss: 15.26979
    Epoch 17300 loss: 15.26977
    Epoch 17400 loss: 15.26976
    Epoch 17500 loss: 15.26974
    Epoch 17600 loss: 15.26973
    Epoch 17700 loss: 15.26971
    Epoch 17800 loss: 15.26969
    Epoch 17900 loss: 15.26968
    Epoch 18000 loss: 15.26966
    Epoch 18100 loss: 15.26965
    Epoch 18200 loss: 15.26963
    Epoch 18300 loss: 15.26962
    Epoch 18400 loss: 15.26960
    Epoch 18500 loss: 15.26958
    Epoch 18600 loss: 15.26957
    Epoch 18700 loss: 15.26955
    Epoch 18800 loss: 15.26954
    Epoch 18900 loss: 15.26952
    Epoch 19000 loss: 15.26951
    Epoch 19100 loss: 15.26949
    Epoch 19200 loss: 15.26947
    Epoch 19300 loss: 15.26946
    Epoch 19400 loss: 15.26944
    Epoch 19500 loss: 15.26943
    Epoch 19600 loss: 15.26941
    Epoch 19700 loss: 15.26939
    Epoch 19800 loss: 15.26938
    Epoch 19900 loss: 15.26936
    Epoch 20000 loss: 15.26935
    Epoch 20100 loss: 15.26933
    Epoch 20200 loss: 15.26932
    Epoch 20300 loss: 15.26930
    Epoch 20400 loss: 15.26929
    Epoch 20500 loss: 15.26927
    Epoch 20600 loss: 15.26926
    Epoch 20700 loss: 15.26924
    Epoch 20800 loss: 15.26923
    Epoch 20900 loss: 15.26921
    Epoch 21000 loss: 15.26919
    Epoch 21100 loss: 15.26918
    Epoch 21200 loss: 15.26916
    Epoch 21300 loss: 15.26915
    Epoch 21400 loss: 15.26913
    Epoch 21500 loss: 15.26912
    Epoch 21600 loss: 15.26910
    Epoch 21700 loss: 15.26909
    Epoch 21800 loss: 15.26907
    Epoch 21900 loss: 15.26906
    Epoch 22000 loss: 15.26904
    Epoch 22100 loss: 15.26903
    Epoch 22200 loss: 15.26901
    Epoch 22300 loss: 15.26900
    Epoch 22400 loss: 15.26898
    Epoch 22500 loss: 15.26897
    Epoch 22600 loss: 15.26895
    Epoch 22700 loss: 15.26894
    Epoch 22800 loss: 15.26892
    Epoch 22900 loss: 15.26891
    Epoch 23000 loss: 15.26889
    Epoch 23100 loss: 15.26888
    Epoch 23200 loss: 15.26886
    Epoch 23300 loss: 15.26885
    Epoch 23400 loss: 15.26883
    Epoch 23500 loss: 15.26881
    Epoch 23600 loss: 15.26880
    Epoch 23700 loss: 15.26879
    Epoch 23800 loss: 15.26877
    Epoch 23900 loss: 15.26875
    Epoch 24000 loss: 15.26874
    Epoch 24100 loss: 15.26873
    Epoch 24200 loss: 15.26871
    Epoch 24300 loss: 15.26870
    Epoch 24400 loss: 15.26868
    Epoch 24500 loss: 15.26867
    Epoch 24600 loss: 15.26865
    Epoch 24700 loss: 15.26864
    Epoch 24800 loss: 15.26863
    Epoch 24900 loss: 15.26861
    Epoch 25000 loss: 15.26859
    Epoch 25100 loss: 15.26858
    Epoch 25200 loss: 15.26857
    Epoch 25300 loss: 15.26855
    Epoch 25400 loss: 15.26854
    Epoch 25500 loss: 15.26852
    Epoch 25600 loss: 15.26851
    Epoch 25700 loss: 15.26849
    Epoch 25800 loss: 15.26848
    Epoch 25900 loss: 15.26846
    Epoch 26000 loss: 15.26845
    Epoch 26100 loss: 15.26844
    Epoch 26200 loss: 15.26842
    Epoch 26300 loss: 15.26840
    Epoch 26400 loss: 15.26839
    Epoch 26500 loss: 15.26838
    Epoch 26600 loss: 15.26836
    Epoch 26700 loss: 15.26835
    Epoch 26800 loss: 15.26833
    Epoch 26900 loss: 15.26832
    Epoch 27000 loss: 15.26830
    Epoch 27100 loss: 15.26829
    Epoch 27200 loss: 15.26828
    Epoch 27300 loss: 15.26826
    Epoch 27400 loss: 15.26825
    Epoch 27500 loss: 15.26824
    Epoch 27600 loss: 15.26822
    Epoch 27700 loss: 15.26821
    Epoch 27800 loss: 15.26819
    Epoch 27900 loss: 15.26818
    Epoch 28000 loss: 15.26816
    Epoch 28100 loss: 15.26815
    Epoch 28200 loss: 15.26814
    Epoch 28300 loss: 15.26812
    Epoch 28400 loss: 15.26811
    Epoch 28500 loss: 15.26809
    Epoch 28600 loss: 15.26808
    Epoch 28700 loss: 15.26806
    Epoch 28800 loss: 15.26805
    Epoch 28900 loss: 15.26804
    Epoch 29000 loss: 15.26802
    Epoch 29100 loss: 15.26801
    Epoch 29200 loss: 15.26800
    Epoch 29300 loss: 15.26798
    Epoch 29400 loss: 15.26797
    Epoch 29500 loss: 15.26795
    Epoch 29600 loss: 15.26794
    Epoch 29700 loss: 15.26793
    Epoch 29800 loss: 15.26791
    Epoch 29900 loss: 15.26790
    Epoch 30000 loss: 15.26789
    Epoch 30100 loss: 15.26787
    Epoch 30200 loss: 15.26786
    Epoch 30300 loss: 15.26785
    Epoch 30400 loss: 15.26783
    Epoch 30500 loss: 15.26782
    Epoch 30600 loss: 15.26780
    Epoch 30700 loss: 15.26779
    Epoch 30800 loss: 15.26778
    Epoch 30900 loss: 15.26776
    Epoch 31000 loss: 15.26775
    Epoch 31100 loss: 15.26774
    Epoch 31200 loss: 15.26772
    Epoch 31300 loss: 15.26771
    Epoch 31400 loss: 15.26769
    Epoch 31500 loss: 15.26768
    Epoch 31600 loss: 15.26767
    Epoch 31700 loss: 15.26766
    Epoch 31800 loss: 15.26764
    Epoch 31900 loss: 15.26763
    Epoch 32000 loss: 15.26761
    Epoch 32100 loss: 15.26760
    Epoch 32200 loss: 15.26759
    Epoch 32300 loss: 15.26757
    Epoch 32400 loss: 15.26756
    Epoch 32500 loss: 15.26755
    Epoch 32600 loss: 15.26753
    Epoch 32700 loss: 15.26752
    Epoch 32800 loss: 15.26751
    Epoch 32900 loss: 15.26749
    Epoch 33000 loss: 15.26748
    Epoch 33100 loss: 15.26747
    Epoch 33200 loss: 15.26745
    Epoch 33300 loss: 15.26744
    Epoch 33400 loss: 15.26743
    Epoch 33500 loss: 15.26741
    Epoch 33600 loss: 15.26740
    Epoch 33700 loss: 15.26739
    Epoch 33800 loss: 15.26738
    Epoch 33900 loss: 15.26736
    Epoch 34000 loss: 15.26735
    Epoch 34100 loss: 15.26733
    Epoch 34200 loss: 15.26732
    Epoch 34300 loss: 15.26731
    Epoch 34400 loss: 15.26730
    Epoch 34500 loss: 15.26728
    Epoch 34600 loss: 15.26727
    Epoch 34700 loss: 15.26726
    Epoch 34800 loss: 15.26724
    Epoch 34900 loss: 15.26723
    Epoch 35000 loss: 15.26722
    Epoch 35100 loss: 15.26720
    Epoch 35200 loss: 15.26719
    Epoch 35300 loss: 15.26718
    Epoch 35400 loss: 15.26717
    Epoch 35500 loss: 15.26715
    Epoch 35600 loss: 15.26714
    Epoch 35700 loss: 15.26713
    Epoch 35800 loss: 15.26712
    Epoch 35900 loss: 15.26710
    Epoch 36000 loss: 15.26709
    Epoch 36100 loss: 15.26707
    Epoch 36200 loss: 15.26707
    Epoch 36300 loss: 15.26705
    Epoch 36400 loss: 15.26704
    Epoch 36500 loss: 15.26702
    Epoch 36600 loss: 15.26701
    Epoch 36700 loss: 15.26700
    Epoch 36800 loss: 15.26699
    Epoch 36900 loss: 15.26697
    Epoch 37000 loss: 15.26696
    Epoch 37100 loss: 15.26695
    Epoch 37200 loss: 15.26694
    Epoch 37300 loss: 15.26692
    Epoch 37400 loss: 15.26691
    Epoch 37500 loss: 15.26690
    Epoch 37600 loss: 15.26689
    Epoch 37700 loss: 15.26687
    Epoch 37800 loss: 15.26686
    Epoch 37900 loss: 15.26685
    Epoch 38000 loss: 15.26684
    Epoch 38100 loss: 15.26682
    Epoch 38200 loss: 15.26681
    Epoch 38300 loss: 15.26680
    Epoch 38400 loss: 15.26679
    Epoch 38500 loss: 15.26677
    Epoch 38600 loss: 15.26676
    Epoch 38700 loss: 15.26675
    Epoch 38800 loss: 15.26674
    Epoch 38900 loss: 15.26673
    Epoch 39000 loss: 15.26671
    Epoch 39100 loss: 15.26670
    Epoch 39200 loss: 15.26669
    Epoch 39300 loss: 15.26667
    Epoch 39400 loss: 15.26666
    Epoch 39500 loss: 15.26665
    Epoch 39600 loss: 15.26664
    Epoch 39700 loss: 15.26663
    Epoch 39800 loss: 15.26661
    Epoch 39900 loss: 15.26660
    Epoch 40000 loss: 15.26659
    Epoch 40100 loss: 15.26657
    Epoch 40200 loss: 15.26656
    Epoch 40300 loss: 15.26655
    Epoch 40400 loss: 15.26654
    Epoch 40500 loss: 15.26653
    Epoch 40600 loss: 15.26651
    Epoch 40700 loss: 15.26650
    Epoch 40800 loss: 15.26649
    Epoch 40900 loss: 15.26648
    Epoch 41000 loss: 15.26647
    Epoch 41100 loss: 15.26645
    Epoch 41200 loss: 15.26644
    Epoch 41300 loss: 15.26643
    Epoch 41400 loss: 15.26642
    Epoch 41500 loss: 15.26641
    Epoch 41600 loss: 15.26639
    Epoch 41700 loss: 15.26638
    Epoch 41800 loss: 15.26637
    Epoch 41900 loss: 15.26636
    Epoch 42000 loss: 15.26635
    Epoch 42100 loss: 15.26633
    Epoch 42200 loss: 15.26632
    Epoch 42300 loss: 15.26631
    Epoch 42400 loss: 15.26630
    Epoch 42500 loss: 15.26629
    Epoch 42600 loss: 15.26627
    Epoch 42700 loss: 15.26626
    Epoch 42800 loss: 15.26625
    Epoch 42900 loss: 15.26624
    Epoch 43000 loss: 15.26623
    Epoch 43100 loss: 15.26621
    Epoch 43200 loss: 15.26620
    Epoch 43300 loss: 15.26619
    Epoch 43400 loss: 15.26618
    Epoch 43500 loss: 15.26617
    Epoch 43600 loss: 15.26616
    Epoch 43700 loss: 15.26615
    Epoch 43800 loss: 15.26613
    Epoch 43900 loss: 15.26612
    Epoch 44000 loss: 15.26611
    Epoch 44100 loss: 15.26610
    Epoch 44200 loss: 15.26609
    Epoch 44300 loss: 15.26608
    Epoch 44400 loss: 15.26606
    Epoch 44500 loss: 15.26605
    Epoch 44600 loss: 15.26604
    Epoch 44700 loss: 15.26603
    Epoch 44800 loss: 15.26602
    Epoch 44900 loss: 15.26601
    Epoch 45000 loss: 15.26600
    Epoch 45100 loss: 15.26598
    Epoch 45200 loss: 15.26597
    Epoch 45300 loss: 15.26596
    Epoch 45400 loss: 15.26595
    Epoch 45500 loss: 15.26594
    Epoch 45600 loss: 15.26592
    Epoch 45700 loss: 15.26592
    Epoch 45800 loss: 15.26590
    Epoch 45900 loss: 15.26589
    Epoch 46000 loss: 15.26588
    Epoch 46100 loss: 15.26587
    Epoch 46200 loss: 15.26586
    Epoch 46300 loss: 15.26585
    Epoch 46400 loss: 15.26584
    Epoch 46500 loss: 15.26582
    Epoch 46600 loss: 15.26581
    Epoch 46700 loss: 15.26580
    Epoch 46800 loss: 15.26579
    Epoch 46900 loss: 15.26578
    Epoch 47000 loss: 15.26577
    Epoch 47100 loss: 15.26576
    Epoch 47200 loss: 15.26574
    Epoch 47300 loss: 15.26573
    Epoch 47400 loss: 15.26572
    Epoch 47500 loss: 15.26571
    Epoch 47600 loss: 15.26570
    Epoch 47700 loss: 15.26569
    Epoch 47800 loss: 15.26568
    Epoch 47900 loss: 15.26567
    Epoch 48000 loss: 15.26566
    Epoch 48100 loss: 15.26564
    Epoch 48200 loss: 15.26563
    Epoch 48300 loss: 15.26562
    Epoch 48400 loss: 15.26561
    Epoch 48500 loss: 15.26560
    Epoch 48600 loss: 15.26559
    Epoch 48700 loss: 15.26558
    Epoch 48800 loss: 15.26557
    Epoch 48900 loss: 15.26556
    Epoch 49000 loss: 15.26554
    Epoch 49100 loss: 15.26554
    Epoch 49200 loss: 15.26552
    Epoch 49300 loss: 15.26551
    Epoch 49400 loss: 15.26550
    Epoch 49500 loss: 15.26549
    Epoch 49600 loss: 15.26548
    Epoch 49700 loss: 15.26547
    Epoch 49800 loss: 15.26546
    Epoch 49900 loss: 15.26545



```python
# 학습 곡선 출력(손실)

plt.plot(history[:,0], history[:,1], 'b')
plt.xlabel('반복 횟수')
plt.ylabel('손실')
plt.title('학습 곡선(손실)')
plt.show()
```


    
*(이미지 없음 / image not preserved: output_62_0.png)*
    

