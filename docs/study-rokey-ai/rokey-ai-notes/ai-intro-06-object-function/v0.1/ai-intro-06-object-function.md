# 6장 예측 함수 정의 하기

> **Migrated from Quartz wiki** (`wiki/study/rokey-ai/basic/ai-intro-06-object_function.md`, updated 2026-09-26)  
> tags: rokey, ai, notebook, basic, study  
> source: raw/2026-09-26-rokey-ai/basic/강의_3기_AI개론_6차시__Object_function_.ipynb  
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
```

    Successfully installed nvidia-cublas-cu12-12.4.5.8 nvidia-cuda-cupti-cu12-12.4.127 nvidia-cuda-nvrtc-cu12-12.4.127 nvidia-cuda-runtime-cu12-12.4.127 nvidia-cudnn-cu12-9.1.0.70 nvidia-cufft-cu12-11.2.1.3 nvidia-curand-cu12-10.3.5.147 nvidia-cusolver-cu12-11.6.1.9 nvidia-cusparse-cu12-12.3.1.170 nvidia-nvjitlink-cu12-12.4.127 torchviz-0.0.3


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

## 선형 회귀 (Linear regression) 손실 함수

### Linear hidden layers


```python
def linear(x, w, b):
  y = torch.matmul(x, w) + b
  return y

x = torch.FloatTensor([[1, 2, 3],
                       [2, 4, 6]])
print("Initial value", "="*50)
print("x: \n", x)

w1 = torch.rand(3, 4)
b1 = torch.ones(1, 4)
print("w1: \n", w1)
print("b1: \n", b1)

```

    Initial value ==================================================
    x: 
     tensor([[1., 2., 3.],
            [2., 4., 6.]])
    w1: 
     tensor([[0.1561, 0.8231, 0.7018, 0.2547],
            [0.0762, 0.3104, 0.7260, 0.5712],
            [0.3995, 0.0515, 0.5970, 0.3863]])
    b1: 
     tensor([[1., 1., 1., 1.]])



```python
## Hidden layer 1
print("Hidden layer 1", "="*50)
h1 = linear(x, w1, b1)
print("linear(x, w1, b1): \n")
print(h1)
# b = torch.tensor([3])

# Hidden layer 2
print("Hidden layer 2", "="*50)
w2 = torch.rand(4, 3)
print("w2: \n", w2)
b2 = torch.ones(1, 3)
print("b2: \n", b2)

print()
h2 = linear(h1, w2, b2)
print("linear(w1, w2, b): \n", "="*50)
print(h2)
```

    Hidden layer 1 ==================================================
    linear(x, w1, b1): 
    
    tensor([[2.5070, 2.5984, 4.9449, 3.5558],
            [4.0141, 4.1967, 8.8898, 6.1117]])
    Hidden layer 2 ==================================================
    w2: 
     tensor([[0.3832, 0.9377, 0.8449],
            [0.5708, 0.0678, 0.6028],
            [0.8122, 0.1830, 0.8957],
            [0.6924, 0.1640, 0.2347]])
    b2: 
     tensor([[1., 1., 1.]])
    
    linear(w1, w2, b): 
     ==================================================
    tensor([[ 9.9220,  5.0146,  9.9480],
            [16.3854,  7.6768, 16.3179]])


### 회귀 분석: Single data


```python
# torch.manual_seed(1)
input = torch.randn(1, requires_grad=True)
target = torch.randn(1)

print("Before SDG","="*50)
print("input: ", input)
print("target: ", target)

loss = nn.MSELoss() # class
optimizer = optim.SGD([input], lr=0.02) # params argument given to the optimizer should be an iterable of Tensors or dicts

for epoch in range(100):

  loss_result = loss(input, target)
  # loss_result = (input - target)**2
  # print("loss_result: ", loss_result)
  optimizer.zero_grad() # Sets the gradients of all optimized torch.Tensors to zero.
  loss_result.backward() # Computes the gradient of current tensor wrt graph leaves
  optimizer.step()

print("After SDG", "="*50)
print("input: ", input)
print("target: ", target)
```

    Before SDG ==================================================
    input:  tensor([-2.2794], requires_grad=True)
    target:  tensor([0.8761])
    After SDG ==================================================
    input:  tensor([0.8228], requires_grad=True)
    target:  tensor([0.8761])


### 회귀 분석: Batch data


```python
x = torch.arange(-3, 3, 0.1)
y = 2 * x + 1
y1 = 2 * x + 1 + torch.randn(x.shape)
print("x = \n", x)
print("y = \n", y1)

plt.plot(x.numpy(), y.numpy())
plt.plot(x.numpy(), y1.numpy(), "o")
plt.show()
```

    x = 
     tensor([-3.0000e+00, -2.9000e+00, -2.8000e+00, -2.7000e+00, -2.6000e+00,
            -2.5000e+00, -2.4000e+00, -2.3000e+00, -2.2000e+00, -2.1000e+00,
            -2.0000e+00, -1.9000e+00, -1.8000e+00, -1.7000e+00, -1.6000e+00,
            -1.5000e+00, -1.4000e+00, -1.3000e+00, -1.2000e+00, -1.1000e+00,
            -1.0000e+00, -9.0000e-01, -8.0000e-01, -7.0000e-01, -6.0000e-01,
            -5.0000e-01, -4.0000e-01, -3.0000e-01, -2.0000e-01, -1.0000e-01,
            -2.3842e-08,  1.0000e-01,  2.0000e-01,  3.0000e-01,  4.0000e-01,
             5.0000e-01,  6.0000e-01,  7.0000e-01,  8.0000e-01,  9.0000e-01,
             1.0000e+00,  1.1000e+00,  1.2000e+00,  1.3000e+00,  1.4000e+00,
             1.5000e+00,  1.6000e+00,  1.7000e+00,  1.8000e+00,  1.9000e+00,
             2.0000e+00,  2.1000e+00,  2.2000e+00,  2.3000e+00,  2.4000e+00,
             2.5000e+00,  2.6000e+00,  2.7000e+00,  2.8000e+00,  2.9000e+00])
    y = 
     tensor([-5.6562, -4.6821, -4.5692, -5.3380, -4.2822, -3.6094, -3.4296, -4.3626,
            -3.4961, -1.8547, -2.4206, -1.5669, -2.2308, -3.1217, -1.5268, -1.1674,
            -1.7885, -0.6554, -0.5419, -1.6530, -1.6600,  0.0740, -0.8599, -1.7846,
            -1.0413, -0.1855,  0.7463, -0.5915,  1.6027,  2.2478,  1.2890, -0.2717,
             1.6807,  1.2525,  0.8120,  2.6701,  0.3058,  3.3220,  1.8301,  2.5123,
             3.9486,  1.9211,  5.9662,  3.7459,  3.6962,  3.2712,  3.6519,  4.3499,
             5.7666,  3.9812,  5.7902,  5.2615,  7.1498,  4.2552,  6.4119,  6.3046,
             5.0495,  5.3742,  6.8492,  7.4082])



    
*(이미지 없음 / image not preserved: output_16_1.png)*
    


### 회귀 분석: batch data


```python
## Simulate data
x = torch.arange(-3, 3, 0.1)
y = 2 * x + 1
y1 = 2 * x + 1 + torch.randn(x.shape)
print("x = \n", x)
print("y = \n", y1)

plt.plot(x, y)
plt.plot(x, y1, "o")
plt.show()
```

    x = 
     tensor([-3.0000e+00, -2.9000e+00, -2.8000e+00, -2.7000e+00, -2.6000e+00,
            -2.5000e+00, -2.4000e+00, -2.3000e+00, -2.2000e+00, -2.1000e+00,
            -2.0000e+00, -1.9000e+00, -1.8000e+00, -1.7000e+00, -1.6000e+00,
            -1.5000e+00, -1.4000e+00, -1.3000e+00, -1.2000e+00, -1.1000e+00,
            -1.0000e+00, -9.0000e-01, -8.0000e-01, -7.0000e-01, -6.0000e-01,
            -5.0000e-01, -4.0000e-01, -3.0000e-01, -2.0000e-01, -1.0000e-01,
            -2.3842e-08,  1.0000e-01,  2.0000e-01,  3.0000e-01,  4.0000e-01,
             5.0000e-01,  6.0000e-01,  7.0000e-01,  8.0000e-01,  9.0000e-01,
             1.0000e+00,  1.1000e+00,  1.2000e+00,  1.3000e+00,  1.4000e+00,
             1.5000e+00,  1.6000e+00,  1.7000e+00,  1.8000e+00,  1.9000e+00,
             2.0000e+00,  2.1000e+00,  2.2000e+00,  2.3000e+00,  2.4000e+00,
             2.5000e+00,  2.6000e+00,  2.7000e+00,  2.8000e+00,  2.9000e+00])
    y = 
     tensor([-5.9167, -3.9447, -5.6223, -4.0863, -4.5880, -5.0063, -3.9763, -2.4459,
            -3.5088, -3.8868, -2.1705, -2.7506, -1.4298, -4.2381, -2.3589, -0.1983,
            -2.0479, -1.8499, -0.9510, -0.8186,  0.7749, -0.9127, -2.3240,  0.5426,
             0.3100, -1.0177, -1.0024,  0.4696, -0.6133,  1.4404,  1.8807,  0.6990,
             1.2501,  2.3735,  2.0950,  1.3549,  1.2450,  2.1464,  2.4091,  3.8207,
             2.1393,  2.6143,  3.0969,  3.2183,  3.5943,  3.1596,  2.0317,  4.2764,
             4.3305,  5.2369,  5.2304,  5.0597,  4.8653,  4.8217,  5.8014,  4.6964,
             5.1506,  6.5247,  6.1717,  5.7490])



    
*(이미지 없음 / image not preserved: output_18_1.png)*
    



```python
# Initial weight and bias
a = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)
print("Initial value = \n", "="*50)
print("a: ", a)
print("b: ", b)

# Define loss and optimizer
lr = 0.01
loss = nn.MSELoss()
optimizer = optim.SGD([a, b], lr=lr)

n_epochs = 100

for epoch in range(n_epochs):
    yhat = a*x + b
    loss_result = loss(yhat, y1)
    if epoch % 10 == 0:
      print("epoch: ", epoch, "a: ", a.item(), "b: ",
            b.item(), "loss: ", loss_result.item())
    optimizer.zero_grad()
    loss_result.backward()
    optimizer.step()

print("After SGD", "="*50)
print(a, b)
```

    Initial value = 
     ==================================================
    a:  tensor([-0.4390], requires_grad=True)
    b:  tensor([0.0206], requires_grad=True)
    epoch:  0 a:  -0.4390020966529846 b:  0.02057533524930477 loss:  17.44143295288086
    epoch:  10 a:  0.6328509449958801 b:  0.14871202409267426 loss:  5.784862041473389
    epoch:  20 a:  1.2108873128890991 b:  0.26087409257888794 loss:  2.317333936691284
    epoch:  30 a:  1.5229146480560303 b:  0.35654592514038086 loss:  1.252461552619934
    epoch:  40 a:  1.6915931701660156 b:  0.43689200282096863 loss:  0.9037297368049622
    epoch:  50 a:  1.7829780578613281 b:  0.5037174224853516 loss:  0.7758422493934631
    epoch:  60 a:  1.8326499462127686 b:  0.558957040309906 loss:  0.7208566665649414
    epoch:  70 a:  1.8597805500030518 b:  0.6044394969940186 loss:  0.6929482817649841
    epoch:  80 a:  1.8747056722640991 b:  0.6417922377586365 loss:  0.6768621802330017
    epoch:  90 a:  1.8830021619796753 b:  0.672417402267456 loss:  0.6668579578399658
    After SGD ==================================================
    tensor([1.8877], requires_grad=True) tensor([0.6975], requires_grad=True)


## 예측 함수의 내부 구조


```python
# 레이어 함수 정의

# 첫번째 선형 함수
# 784 입력 수
# 128 출력 수
l1 = nn.Linear(784, 128)

# 두번째 선형 함수
# 128 입력 수
# 10 출력 수
l2 = nn.Linear(128, 10)

# 활성화 함수
relu = nn.ReLU(inplace=True)
```


```python
# 입력 텐서로부터 출력 텐서를 계산

# 더미 입력 데이터 작성
inputs = torch.randn(100, 784)

# 중간 텐서 1 계산
m1 = l1(inputs)

# 중간 텐서 2 계산
m2 = relu(m1)

# 출력 텐서 계산
outputs = l2(m2)

# 입력 텐서와 출력 텐서 shape 확인
print('입력 텐서', inputs.shape)
print('출력 텐서', outputs.shape)
```

    입력 텐서 torch.Size([100, 784])
    출력 텐서 torch.Size([100, 10])



```python
# nn.Sequential을 사용해 전체를 합성 함수로 정의

net2 = nn.Sequential(
    l1,
    relu,
    l2
)

outputs2 = net2(inputs)

# 입력 텐서와 출력 텐서의 shape 확인
print('입력 텐서', inputs.shape)
print('출력 텐서', outputs2.shape)
```

    입력 텐서 torch.Size([100, 784])
    출력 텐서 torch.Size([100, 10])


###  활성화 함수의 목적
이 절에서는 예측 결과 그래프(그림 4-9에서 그림 4-11까지)가 중요하며, 따라서 지금 시점에서 구현 코드의 의미를 이해하지 못하더라도 상관없다. 아래의 코드는 어디까지나 참고를 위해 작성되었다.


```python
# 훈련 데이터, 검증 데이터 계산
np.random.seed(123)
x = np.random.randn(100,1)

# y는 x^2에 난수를 1/10만큼 더한 값
y = x**2 + np.random.randn(100,1) * 0.2

# 데이터를 50건씩 훈련용과 검증용으로 나눔
x_train = x[:50,:]
x_test = x[50:,:]
y_train = y[:50,:]
y_test = y[50:,:]
```


```python
# 산포도 출력
plt.scatter(x_train, y_train, c='b', label='훈련 데이터')
plt.scatter(x_test, y_test, c='k', marker='x', label='검증 데이터')
plt.legend()
plt.show()
```


    
*(이미지 없음 / image not preserved: output_26_0.png)*
    



```python
# 입력 변수 x와 정답 yt의 텐서화

inputs = torch.tensor(x_train).float()
labels = torch.tensor(y_train).float()

inputs_test = torch.tensor(x_test).float()
labels_test = torch.tensor(y_test).float()
```

### 선형 회귀 모델의 경우


```python
# 모델 정의

class Net(nn.Module):
    def __init__(self):
        #  부모 클래스 nn.Modules 의 초기화
        super().__init__()

        # 출력층 정의
        self.l1 = nn.Linear(1, 1)

    # 예측 함수 정의
    def forward(self, x):
        x1 = self.l1(x) # 선형 회귀
        return x1
```


```python
# 학습률
lr = 0.01

# 인스턴스 생성(파라미터 초기화)
net = Net()

# 최적화 알고리즘 : 경사 하강법
optimizer = optim.SGD(net.parameters(), lr=lr)

# 손실 함수： 평균 제곱 오차
criterion = nn.MSELoss()

# 반복 횟수
num_epochs = 10000

#  history 기록을 위한 배열 초기화(손실 함수 값 만을 기록)
history = np.zeros((0,2))
```


```python
# 반복 계산 메인 루프

for epoch in range(num_epochs):

    # 경사 값 초기화
    optimizer.zero_grad()

    # 예측 계산
    outputs = net(inputs)

    # 오차 계산
    loss = criterion(outputs, labels)

    # 경사 계산
    loss.backward()

    # 경사 하강법 적용
    optimizer.step()

    # 100회 마다 도중 경과를 기록
    if ( epoch % 100 == 0):
        history = np.vstack((history, np.array([epoch, loss.item()])))
        print(f'Epoch {epoch} loss: {loss.item():.5f}')
```

    Epoch 0 loss: 4.07665
    Epoch 100 loss: 3.21111
    Epoch 200 loss: 3.19657
    Epoch 300 loss: 3.19632
    Epoch 400 loss: 3.19631
    Epoch 500 loss: 3.19631
    Epoch 600 loss: 3.19631
    Epoch 700 loss: 3.19631
    Epoch 800 loss: 3.19631
    Epoch 900 loss: 3.19631
    Epoch 1000 loss: 3.19631
    Epoch 1100 loss: 3.19631
    Epoch 1200 loss: 3.19631
    Epoch 1300 loss: 3.19631
    Epoch 1400 loss: 3.19631
    Epoch 1500 loss: 3.19631
    Epoch 1600 loss: 3.19631
    Epoch 1700 loss: 3.19631
    Epoch 1800 loss: 3.19631
    Epoch 1900 loss: 3.19631
    Epoch 2000 loss: 3.19631
    Epoch 2100 loss: 3.19631
    Epoch 2200 loss: 3.19631
    Epoch 2300 loss: 3.19631
    Epoch 2400 loss: 3.19631
    Epoch 2500 loss: 3.19631
    Epoch 2600 loss: 3.19631
    Epoch 2700 loss: 3.19631
    Epoch 2800 loss: 3.19631
    Epoch 2900 loss: 3.19631
    Epoch 3000 loss: 3.19631
    Epoch 3100 loss: 3.19631
    Epoch 3200 loss: 3.19631
    Epoch 3300 loss: 3.19631
    Epoch 3400 loss: 3.19631
    Epoch 3500 loss: 3.19631
    Epoch 3600 loss: 3.19631
    Epoch 3700 loss: 3.19631
    Epoch 3800 loss: 3.19631
    Epoch 3900 loss: 3.19631
    Epoch 4000 loss: 3.19631
    Epoch 4100 loss: 3.19631
    Epoch 4200 loss: 3.19631
    Epoch 4300 loss: 3.19631
    Epoch 4400 loss: 3.19631
    Epoch 4500 loss: 3.19631
    Epoch 4600 loss: 3.19631
    Epoch 4700 loss: 3.19631
    Epoch 4800 loss: 3.19631
    Epoch 4900 loss: 3.19631
    Epoch 5000 loss: 3.19631
    Epoch 5100 loss: 3.19631
    Epoch 5200 loss: 3.19631
    Epoch 5300 loss: 3.19631
    Epoch 5400 loss: 3.19631
    Epoch 5500 loss: 3.19631
    Epoch 5600 loss: 3.19631
    Epoch 5700 loss: 3.19631
    Epoch 5800 loss: 3.19631
    Epoch 5900 loss: 3.19631
    Epoch 6000 loss: 3.19631
    Epoch 6100 loss: 3.19631
    Epoch 6200 loss: 3.19631
    Epoch 6300 loss: 3.19631
    Epoch 6400 loss: 3.19631
    Epoch 6500 loss: 3.19631
    Epoch 6600 loss: 3.19631
    Epoch 6700 loss: 3.19631
    Epoch 6800 loss: 3.19631
    Epoch 6900 loss: 3.19631
    Epoch 7000 loss: 3.19631
    Epoch 7100 loss: 3.19631
    Epoch 7200 loss: 3.19631
    Epoch 7300 loss: 3.19631
    Epoch 7400 loss: 3.19631
    Epoch 7500 loss: 3.19631
    Epoch 7600 loss: 3.19631
    Epoch 7700 loss: 3.19631
    Epoch 7800 loss: 3.19631
    Epoch 7900 loss: 3.19631
    Epoch 8000 loss: 3.19631
    Epoch 8100 loss: 3.19631
    Epoch 8200 loss: 3.19631
    Epoch 8300 loss: 3.19631
    Epoch 8400 loss: 3.19631
    Epoch 8500 loss: 3.19631
    Epoch 8600 loss: 3.19631
    Epoch 8700 loss: 3.19631
    Epoch 8800 loss: 3.19631
    Epoch 8900 loss: 3.19631
    Epoch 9000 loss: 3.19631
    Epoch 9100 loss: 3.19631
    Epoch 9200 loss: 3.19631
    Epoch 9300 loss: 3.19631
    Epoch 9400 loss: 3.19631
    Epoch 9500 loss: 3.19631
    Epoch 9600 loss: 3.19631
    Epoch 9700 loss: 3.19631
    Epoch 9800 loss: 3.19631
    Epoch 9900 loss: 3.19631



```python
# 결과 그래프
labels_pred = net(inputs_test)

plt.title('은닉층 없음,활성화 함수 없음')
plt.scatter(inputs_test[:,0].data, labels_pred[:,0].data, c='b', label='예측값')
plt.scatter(inputs_test[:,0].data, labels_test[:,0].data, c='k', marker='x',label='정답')
plt.legend()
plt.show()
```


    
*(이미지 없음 / image not preserved: output_32_0.png)*
    


### 활성화 함수가 없는 딥러닝 모델의 경우


```python
# 모델 정의

class Net2(nn.Module):
    def __init__(self):
        #  부모 클래스 nn.Modules 초기화
        super().__init__()

        # 출력층 정의
        self.l1 = nn.Linear(1, 10)
        self.l2 = nn.Linear(10, 10)
        self.l3 = nn.Linear(10,1)

    # 예측 함수 정의
    def forward(self, x):
        x1 = self.l1(x)
        x2 = self.l2(x1)
        x3 = self.l3(x2)
        return x3
```


```python
# 학습률
lr = 0.01

# 인스턴스 생성(파라미터 초기화)
net2 = Net2()

# 최적화 알고리즘 : 경사 하강법
optimizer = optim.SGD(net2.parameters(), lr=lr)

# 손실 함수 : 평균 제곱 오차
criterion = nn.MSELoss()

# 반복 횟수
num_epochs = 10000

# history 기록을 위한 배열 초기화(손실 함수 값 만을 기록)
history = np.zeros((0,2))
```


```python
# 반복 계산 메인 루프

for epoch in range(num_epochs):

    # 경사 값 초기화
    optimizer.zero_grad()

    # 예측 계산
    outputs = net2(inputs)

    # 오차 계산
    loss = criterion(outputs, labels)

    # 경사 계산
    loss.backward()

    # 경사 하강법 적용
    optimizer.step()

    # 100회 마다 도중 경과를 기록
    if ( epoch % 100 == 0):
        history = np.vstack((history, np.array([epoch, loss.item()])))
        print(f'Epoch {epoch} loss: {loss.item():.5f}')
```

    Epoch 0 loss: 6.10382
    Epoch 100 loss: 3.19631
    Epoch 200 loss: 3.19631
    Epoch 300 loss: 3.19631
    Epoch 400 loss: 3.19631
    Epoch 500 loss: 3.19631
    Epoch 600 loss: 3.19631
    Epoch 700 loss: 3.19631
    Epoch 800 loss: 3.19631
    Epoch 900 loss: 3.19631
    Epoch 1000 loss: 3.19631
    Epoch 1100 loss: 3.19631
    Epoch 1200 loss: 3.19631
    Epoch 1300 loss: 3.19631
    Epoch 1400 loss: 3.19631
    Epoch 1500 loss: 3.19631
    Epoch 1600 loss: 3.19631
    Epoch 1700 loss: 3.19631
    Epoch 1800 loss: 3.19631
    Epoch 1900 loss: 3.19631
    Epoch 2000 loss: 3.19631
    Epoch 2100 loss: 3.19631
    Epoch 2200 loss: 3.19631
    Epoch 2300 loss: 3.19631
    Epoch 2400 loss: 3.19631
    Epoch 2500 loss: 3.19631
    Epoch 2600 loss: 3.19631
    Epoch 2700 loss: 3.19631
    Epoch 2800 loss: 3.19631
    Epoch 2900 loss: 3.19631
    Epoch 3000 loss: 3.19631
    Epoch 3100 loss: 3.19631
    Epoch 3200 loss: 3.19631
    Epoch 3300 loss: 3.19631
    Epoch 3400 loss: 3.19631
    Epoch 3500 loss: 3.19631
    Epoch 3600 loss: 3.19631
    Epoch 3700 loss: 3.19631
    Epoch 3800 loss: 3.19631
    Epoch 3900 loss: 3.19631
    Epoch 4000 loss: 3.19631
    Epoch 4100 loss: 3.19631
    Epoch 4200 loss: 3.19631
    Epoch 4300 loss: 3.19631
    Epoch 4400 loss: 3.19631
    Epoch 4500 loss: 3.19631
    Epoch 4600 loss: 3.19631
    Epoch 4700 loss: 3.19631
    Epoch 4800 loss: 3.19631
    Epoch 4900 loss: 3.19631
    Epoch 5000 loss: 3.19631
    Epoch 5100 loss: 3.19631
    Epoch 5200 loss: 3.19631
    Epoch 5300 loss: 3.19631
    Epoch 5400 loss: 3.19631
    Epoch 5500 loss: 3.19631
    Epoch 5600 loss: 3.19631
    Epoch 5700 loss: 3.19631
    Epoch 5800 loss: 3.19631
    Epoch 5900 loss: 3.19631
    Epoch 6000 loss: 3.19631
    Epoch 6100 loss: 3.19631
    Epoch 6200 loss: 3.19631
    Epoch 6300 loss: 3.19631
    Epoch 6400 loss: 3.19631
    Epoch 6500 loss: 3.19631
    Epoch 6600 loss: 3.19631
    Epoch 6700 loss: 3.19631
    Epoch 6800 loss: 3.19631
    Epoch 6900 loss: 3.19631
    Epoch 7000 loss: 3.19631
    Epoch 7100 loss: 3.19631
    Epoch 7200 loss: 3.19631
    Epoch 7300 loss: 3.19631
    Epoch 7400 loss: 3.19631
    Epoch 7500 loss: 3.19631
    Epoch 7600 loss: 3.19631
    Epoch 7700 loss: 3.19631
    Epoch 7800 loss: 3.19631
    Epoch 7900 loss: 3.19631
    Epoch 8000 loss: 3.19631
    Epoch 8100 loss: 3.19631
    Epoch 8200 loss: 3.19631
    Epoch 8300 loss: 3.19631
    Epoch 8400 loss: 3.19631
    Epoch 8500 loss: 3.19631
    Epoch 8600 loss: 3.19631
    Epoch 8700 loss: 3.19631
    Epoch 8800 loss: 3.19631
    Epoch 8900 loss: 3.19631
    Epoch 9000 loss: 3.19631
    Epoch 9100 loss: 3.19631
    Epoch 9200 loss: 3.19631
    Epoch 9300 loss: 3.19631
    Epoch 9400 loss: 3.19631
    Epoch 9500 loss: 3.19631
    Epoch 9600 loss: 3.19631
    Epoch 9700 loss: 3.19631
    Epoch 9800 loss: 3.19631
    Epoch 9900 loss: 3.19631



```python
# 결과 그래프
labels_pred2 = net2(inputs_test)

plt.title('은닉층 2개, 활성화 함수 사용하지 않음')
plt.scatter(inputs_test[:,0].data, labels_pred2[:,0].data, c='b', label='예측값')
plt.scatter(inputs_test[:,0].data, labels_test[:,0].data, c='k', marker='x',label='정답')
plt.legend()
plt.show()
```


    
*(이미지 없음 / image not preserved: output_37_0.png)*
    


### 활성화 함수 사용가 있는 딥러닝 모델의 경우


```python
# 모델 정의

class Net3(nn.Module):
    def __init__(self):
        #  부모 클래스 nn.Modules 초기화
        super().__init__()

        # 출력층 정의
        self.l1 = nn.Linear(1, 10)
        self.l2 = nn.Linear(10, 10)
        self.l3 = nn.Linear(10,1)
        self.relu = nn.ReLU(inplace=True)

    # 예측 함수 정의
    def forward(self, x):
        x1 = self.relu(self.l1(x))
        x2 = self.relu(self.l2(x1))
        x3 = self.l3(x2)
        return x3
```


```python
# 학습률
lr = 0.01

# 인스턴스 생성(파라미터 초기화)
net3 = Net3()

# 최적화 알고리즘 : 경사 하강법
optimizer = optim.SGD(net3.parameters(), lr=lr)

# 손실 함수： 평균 제곱 오차
criterion = nn.MSELoss()

# 반복 횟수
num_epochs = 10000

# history 기록을 위한 배열 초기화(손실 함수 값 만을 기록)
history = np.zeros((0,2))
```


```python
# 반복 계산 메인 루프

for epoch in range(num_epochs):

    # 경사 값 초기화
    optimizer.zero_grad()

    # 예측 계산
    outputs = net3(inputs)

    # 오차 계산
    loss = criterion(outputs, labels)

    # 경사 계산
    loss.backward()

    # 경사 하강법 적용
    optimizer.step()

    # 100회 마다 도중 경과를 기록
    if ( epoch % 100 == 0):
        history = np.vstack((history, np.array([epoch, loss.item()])))
        print(f'Epoch {epoch} loss: {loss.item():.5f}')
```

    Epoch 0 loss: 4.68561
    Epoch 100 loss: 0.93972
    Epoch 200 loss: 0.15431
    Epoch 300 loss: 0.09376
    Epoch 400 loss: 0.07852
    Epoch 500 loss: 0.07051
    Epoch 600 loss: 0.06508
    Epoch 700 loss: 0.06159
    Epoch 800 loss: 0.05960
    Epoch 900 loss: 0.05824
    Epoch 1000 loss: 0.05723
    Epoch 1100 loss: 0.05644
    Epoch 1200 loss: 0.05582
    Epoch 1300 loss: 0.05494
    Epoch 1400 loss: 0.05419
    Epoch 1500 loss: 0.05365
    Epoch 1600 loss: 0.05321
    Epoch 1700 loss: 0.05285
    Epoch 1800 loss: 0.05252
    Epoch 1900 loss: 0.05205
    Epoch 2000 loss: 0.05155
    Epoch 2100 loss: 0.05114
    Epoch 2200 loss: 0.05079
    Epoch 2300 loss: 0.05048
    Epoch 2400 loss: 0.05020
    Epoch 2500 loss: 0.04995
    Epoch 2600 loss: 0.04972
    Epoch 2700 loss: 0.04947
    Epoch 2800 loss: 0.04925
    Epoch 2900 loss: 0.04905
    Epoch 3000 loss: 0.04888
    Epoch 3100 loss: 0.04874
    Epoch 3200 loss: 0.04860
    Epoch 3300 loss: 0.04848
    Epoch 3400 loss: 0.04837
    Epoch 3500 loss: 0.04826
    Epoch 3600 loss: 0.04817
    Epoch 3700 loss: 0.04809
    Epoch 3800 loss: 0.04802
    Epoch 3900 loss: 0.04795
    Epoch 4000 loss: 0.04788
    Epoch 4100 loss: 0.04782
    Epoch 4200 loss: 0.04777
    Epoch 4300 loss: 0.04772
    Epoch 4400 loss: 0.04767
    Epoch 4500 loss: 0.04762
    Epoch 4600 loss: 0.04758
    Epoch 4700 loss: 0.04754
    Epoch 4800 loss: 0.04750
    Epoch 4900 loss: 0.04741
    Epoch 5000 loss: 0.04730
    Epoch 5100 loss: 0.04720
    Epoch 5200 loss: 0.04713
    Epoch 5300 loss: 0.04704
    Epoch 5400 loss: 0.04694
    Epoch 5500 loss: 0.04686
    Epoch 5600 loss: 0.04678
    Epoch 5700 loss: 0.04670
    Epoch 5800 loss: 0.04662
    Epoch 5900 loss: 0.04654
    Epoch 6000 loss: 0.04647
    Epoch 6100 loss: 0.04640
    Epoch 6200 loss: 0.04634
    Epoch 6300 loss: 0.04628
    Epoch 6400 loss: 0.04622
    Epoch 6500 loss: 0.04616
    Epoch 6600 loss: 0.04611
    Epoch 6700 loss: 0.04606
    Epoch 6800 loss: 0.04601
    Epoch 6900 loss: 0.04597
    Epoch 7000 loss: 0.04593
    Epoch 7100 loss: 0.04590
    Epoch 7200 loss: 0.04587
    Epoch 7300 loss: 0.04584
    Epoch 7400 loss: 0.04581
    Epoch 7500 loss: 0.04579
    Epoch 7600 loss: 0.04577
    Epoch 7700 loss: 0.04575
    Epoch 7800 loss: 0.04574
    Epoch 7900 loss: 0.04573
    Epoch 8000 loss: 0.04572
    Epoch 8100 loss: 0.04571
    Epoch 8200 loss: 0.04570
    Epoch 8300 loss: 0.04569
    Epoch 8400 loss: 0.04568
    Epoch 8500 loss: 0.04568
    Epoch 8600 loss: 0.04567
    Epoch 8700 loss: 0.04567
    Epoch 8800 loss: 0.04566
    Epoch 8900 loss: 0.04566
    Epoch 9000 loss: 0.04565
    Epoch 9100 loss: 0.04565
    Epoch 9200 loss: 0.04564
    Epoch 9300 loss: 0.04564
    Epoch 9400 loss: 0.04564
    Epoch 9500 loss: 0.04563
    Epoch 9600 loss: 0.04563
    Epoch 9700 loss: 0.04563
    Epoch 9800 loss: 0.04562
    Epoch 9900 loss: 0.04562



```python
# 결과 그래프
labels_pred3 = net3(inputs_test)

plt.title('은닉층 2개, 활성화 함수 사용')
plt.scatter(inputs_test[:,0].data, labels_pred3[:,0].data, c='b', label='예측값')
plt.scatter(inputs_test[:,0].data, labels_test[:,0].data, c='k', marker='x',label='정답')
plt.legend()
plt.show()

```


    
*(이미지 없음 / image not preserved: output_42_0.png)*
    


## 이진 분류 (Binary classification) 비용 함수

### 시그모이드 (Sigmoid) 함수


```python
##
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)

fig, axes = plt.subplots(figsize=(4, 4))
plt.plot(x, y)
plt.grid(linestyle = ":")
plt.hlines(0.5, -5, 5, colors="r", linestyles="dashed")
plt.show()
```


    
*(이미지 없음 / image not preserved: output_45_0.png)*
    



```python
x = np.arange(-5.0, 5.0, 0.1)
y1 = sigmoid(0.5*x)
y2 = sigmoid(x)
y3 = sigmoid(2*x)

fig, axes = plt.subplots(figsize=(4, 4))
plt.plot(x, y1, 'r', linestyle='--') # W의 값이 0.5일때
plt.plot(x, y2, 'g') # W의 값이 1일때
plt.plot(x, y3, 'b', linestyle='--') # W의 값이 2일때
plt.plot([0,0],[1.0,0.0], ':') # 가운데 점선 추가
plt.title('Sigmoid Function')
plt.show()
```


    
*(이미지 없음 / image not preserved: output_46_0.png)*
    


### 이진 분류 함수의 손실 함수와 비용 함수


```python
# torch.manual_seed(1)
x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

print("Data shape", "="*50)
print(x_train.shape)
print(y_train.shape)

# Parameter initialize
W = torch.zeros((2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

print("W: \n", W)
print("b: \n", b)

# sigmoid function
def prob(x, w, b):
    y = 1/(1 + torch.exp(-(torch.matmul(x, w) + b)))
    return y

prob = prob(x_train, W, b)
# prob = torch.sigmoid(x_train.matmul(W) + b)

print("Prob: \n", prob)

# cost function
losses = -(y_train*torch.log(prob) + (1 - y_train)*torch.log(1 - prob))
cost = losses.mean()
print("cost: \n", cost)
```

    Data shape ==================================================
    torch.Size([6, 2])
    torch.Size([6, 1])
    W: 
     tensor([[0.],
            [0.]], requires_grad=True)
    b: 
     tensor([0.], requires_grad=True)
    Prob: 
     tensor([[0.5000],
            [0.5000],
            [0.5000],
            [0.5000],
            [0.5000],
            [0.5000]], grad_fn=<MulBackward0>)
    cost: 
     tensor(0.6931, grad_fn=<MeanBackward0>)



```python
#
x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

loss = nn.BCELoss()
sigmoid = nn.Sigmoid()
W = torch.zeros((2, 1), requires_grad=True) #
b = torch.zeros(1, requires_grad=True)

lr = 1
optimizer = optim.SGD([W, b], lr=1)
# print("optimizer: \n", optimizer)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    prediction = sigmoid(x_train.matmul(W) + b)
    # prediction = torch.sigmoid(x_train.matmul(W) + b) # method
    cost = loss(prediction, y_train)
    # cost = F.binary_cross_entropy(prediction, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))
```

    Epoch    0/1000 Cost: 0.693147
    Epoch  100/1000 Cost: 0.134722
    Epoch  200/1000 Cost: 0.080643
    Epoch  300/1000 Cost: 0.057900
    Epoch  400/1000 Cost: 0.045300
    Epoch  500/1000 Cost: 0.037261
    Epoch  600/1000 Cost: 0.031672
    Epoch  700/1000 Cost: 0.027556
    Epoch  800/1000 Cost: 0.024394
    Epoch  900/1000 Cost: 0.021888
    Epoch 1000/1000 Cost: 0.019852


## 다중 분류 (Multinomial classification) 비용 함수

### Pytorch 소프트맥스 함수


```python
def softmax(x):
    y = np.exp(x)/np.sum(np.exp(x))
    return y


x = torch.rand(1, 4)
print('Tensor x = \n', x)

prob = softmax(x.numpy())
# prob = torch.softmax(x, dim = 1) # dim 0, 1
# prob = F.softmax(x, dim = 1)

print("Softmax = \n", prob)
```

    Tensor x = 
     tensor([[0.5389, 0.7376, 0.4636, 0.6481]])
    Softmax = 
     [[0.2346 0.2862 0.2176 0.2617]]


### Pytorch로 softmax 의 cost 함수 구현하기


```python
### One-hot vector 만들기

x = torch.rand(3, 5, requires_grad=True)
print("x: \n", x)
prob = F.softmax(x, dim = 1)
print("prob: \n", prob)

y = torch.randint(5, (3,)) # torch.int64
print("target y = ", y)

# 모든 원소가 0의 값을 가진 3 × 5 텐서 생성
y_one_hot = torch.zeros_like(x)
print(y_one_hot)
y_one_hot.scatter_(dim = 1, index = y.unsqueeze(dim = 1), value = 1) # Tensor.scatter_(dim, index, src, *, reduce=None)
print("one hot vector = \n", y_one_hot)


# 비용함수
print("Crossentroy cost function = ")
(-y_one_hot * torch.log(F.softmax(x, dim=1))).sum(dim=1).mean()
```

    x: 
     tensor([[0.7282, 0.0839, 0.4163, 0.9972, 0.7081],
            [0.5378, 0.7255, 0.6014, 0.9830, 0.2230],
            [0.8833, 0.5953, 0.5312, 0.8071, 0.5571]], requires_grad=True)
    prob: 
     tensor([[0.2200, 0.1155, 0.1610, 0.2879, 0.2156],
            [0.1798, 0.2169, 0.1916, 0.2806, 0.1312],
            [0.2438, 0.1828, 0.1715, 0.2259, 0.1760]], grad_fn=<SoftmaxBackward0>)
    target y =  tensor([0, 4, 3])
    tensor([[0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.]])
    one hot vector = 
     tensor([[1., 0., 0., 0., 0.],
            [0., 0., 0., 0., 1.],
            [0., 0., 0., 1., 0.]])
    Crossentroy cost function = 





    tensor(1.6776, grad_fn=<MeanBackward0>)




```python
### Crossentropy 비용함수
# Low level
torch.log(F.softmax(x, dim=1))

# High level
F.log_softmax(x, dim=1)

# cost function
(y_one_hot * - F.log_softmax(x, dim=1)).sum(dim=1).mean()

 # High level
# 세번째 수식
F.nll_loss(F.log_softmax(x, dim=1), y) # y =  tensor([1, 1, 3])

# 네번째 수식
F.cross_entropy(x, y) # z = torch.rand(3, 5, requires_grad=True), y = torch.randint(5, (3,))
```




    tensor(1.6776, grad_fn=<NllLossBackward0>)




```python
x_train = [[1, 2, 1, 1],
           [2, 1, 3, 2],
           [3, 1, 3, 4],
           [4, 1, 5, 5],
           [1, 7, 5, 5],
           [1, 2, 5, 6],
           [1, 6, 6, 6],
           [1, 7, 7, 7]]
y_train = [2, 2, 2, 1, 1, 1, 0, 0]
x_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train)

print(x_train.shape)
print(y_train.shape)

#
y_one_hot = torch.zeros(8, 3)
y_one_hot.scatter_(1, y_train.unsqueeze(1), 1)


# 모델 초기화
W = torch.zeros((4, 3), requires_grad=True)
b = torch.zeros((1, 3), requires_grad=True)
# optimizer 설정
lr = 0.1
optimizer = optim.SGD([W, b], lr=lr)

nb_epochs = 100

for epoch in range(nb_epochs + 1):

    # H(x) 계산
    hypothesis = F.softmax(x_train.matmul(W) + b, dim=1)
    cost = (y_one_hot * -torch.log(hypothesis)).sum(dim=1).mean()

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[1], line 10
          1 x_train = [[1, 2, 1, 1],
          2            [2, 1, 3, 2],
          3            [3, 1, 3, 4],
       (...)
          7            [1, 6, 6, 6],
          8            [1, 7, 7, 7]]
          9 y_train = [2, 2, 2, 1, 1, 1, 0, 0]
    ---> 10 x_train = torch.FloatTensor(x_train)
         11 y_train = torch.LongTensor(y_train)
         13 print(x_train.shape)


    NameError: name 'torch' is not defined



```python
import numpy as np
import matplotlib.pyplot as plt

# 예제 3D 포인트 클라우드 데이터 (X, Y, Z)
point_cloud = np.array([
    [1, 2, 0.5], [2, 3, 1.0], [3, 4, 1.5], [4, 5, 2.0], 
    [2, 2, 0.3], [3, 3, 1.2], [5, 5, 2.5], [6, 7, 3.0]
])

# X, Y 좌표만 가져와서 Top-Down Projection 수행
x = point_cloud[:, 0]
y = point_cloud[:, 1]

# 2D Top-Down View 시각화
plt.figure(figsize=(6, 6))
plt.scatter(x, y, c='blue', marker='o', label="Projected Points")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.title("Top-Down Projection of 3D Point Cloud")
plt.grid(True)
plt.legend()
plt.show()

```


    
*(이미지 없음 / image not preserved: output_57_0.png)*
    



```python

```
