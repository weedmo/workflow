# 3장 파이토치의 기본 기능

> **Migrated from Quartz wiki** (`wiki/study/rokey-ai/basic/ai-intro-03-pytorch.md`, updated 2026-09-26)  
> tags: rokey, ai, notebook, basic, study  
> source: raw/2026-09-26-rokey-ai/basic/강의_3기_AI개론_3차시__pytorch_.ipynb  
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

## 텐서

### 라이브러리 임포트


```python
# 파이토치 라이브러리

import torch
```

### 다양한 계수의 텐서 만들기


```python
## Tensor 만들기
r0 = torch.tensor(10) # rank 0
print("r0 = \n", r0)
# print("r0.shape = \n", r0.shape)
print(r0.dtype)

r1 = torch.tensor([1, 2 ,3]) # rank 1
print("r1 = \n",r1)
# print("r1.shape = \n", r1.shape)

r2 = torch.tensor([[1, 2, 3], [3, 4, 5]]) # rank 2
print("r2 = \n",r2)
# print("r2.shape = \n", r2.shape)

r3 = torch.tensor([[[1, 2], [3, 4]], # rank 3
                 [[5, 6], [7, 8]]])
print("r3 = \n",r3)
# print("r3.shape = \n", r3.shape)

```

    r0 = 
     tensor(10)
    torch.int64
    r1 = 
     tensor([1, 2, 3])
    r2 = 
     tensor([[1, 2, 3],
            [3, 4, 5]])
    r3 = 
     tensor([[[1, 2],
             [3, 4]],
    
            [[5, 6],
             [7, 8]]])



```python
# 0계 텐서(스칼라)
r0 = torch.tensor(1.0).float()
# r0 = torch.tensor(1.0).float()

# type 확인
print(type(r0))

# dtype 확인
print(r0.dtype)
```

    <class 'torch.Tensor'>
    torch.float32



```python
# shape 확인
print(r0.shape)

# 데이터 확인
print(r0.data)
```

    torch.Size([])
    tensor(1.)



```python
## tensor attribute
print("int attribute:")
int_atr = torch.tensor([1, 2, 3])
print(int_atr)
print(int_atr.shape)
print(int_atr.dtype)
print(int_atr.device)
print()

print("float attribute:")
float_atr = torch.tensor([[1, 2, 3, 4],
                          [5, 6, 7, 8]], dtype=torch.float32)
print(float_atr)
print(float_atr.shape)
print(float_atr.dtype)
print(float_atr.device)
```

    int attribute:
    tensor([1, 2, 3])
    torch.Size([3])
    torch.int64
    cpu
    
    float attribute:
    tensor([[1., 2., 3., 4.],
            [5., 6., 7., 8.]])
    torch.Size([2, 4])
    torch.float32
    cpu



```python
## gpu 사용확인
# 현재 시스템에서 cuda가 사용가능한지 여부
print("torch.cuda.is_available():")
print(torch.cuda.is_available())
print()
# 현재 pythorch가 사용 중인 gpu 인덱스 번호 반환 여러개시 0반환
# 없거나 사용 불가시 오류
print("torch.cuda.current_device():")
print(torch.cuda.current_device())
print()
# 지정한 gpu의 모델명 반환
print("torch.cuda.get_device_name():")
print(torch.cuda.get_device_name(0))
print()
# 사용가능한 gpu의 개수 반환 없으면 0반환
print("torch.cuda.device_count():")
print(torch.cuda.device_count())
```

    torch.cuda.is_available():
    True
    
    torch.cuda.current_device():
    0
    
    torch.cuda.get_device_name():
    Tesla T4
    
    torch.cuda.device_count():
    1



```python
# 1계 텐서(벡터)

# 1계 넘파이 변수 작성
r1_np = np.array([1, 2, 3, 4, 5])
print(r1_np.shape)

# 넘파이에서 텐서로 변환
r1 = torch.tensor(r1_np).float()

# dtype 확인
print(r1.dtype)

# shape 확인
print(r1.shape)

# 데이터 확인
print(r1.data)
```

    (5,)
    torch.float32
    torch.Size([5])
    tensor([1., 2., 3., 4., 5.])



```python
# 2계 텐서(행렬)

# 2계 넘파이 변수 작성
r2_np = np.array([[1, 5, 6], [4, 3, 2]])
print(r2_np.shape)

# 넘파이에서 텐서로 변환
r2 = torch.tensor(r2_np).float()

# shape 확인
print(r2.shape)

# 데이터 확인
print(r2.data)
```

    (2, 3)
    torch.Size([2, 3])
    tensor([[1., 5., 6.],
            [4., 3., 2.]])



```python
# ３계 텐서

# 난수 seed 초기화
torch.manual_seed(123)

# shape=[3,2,2]의 정규분포 텐서 작성
r3 = torch.randn((3, 2, 2))

# shape 확인
print(r3.shape)

# 데이터 확인
print(r3.data)
```

    torch.Size([3, 2, 2])
    tensor([[[-0.1115,  0.1204],
             [-0.3696, -0.2404]],
    
            [[-1.1969,  0.2093],
             [-0.9724, -0.7550]],
    
            [[ 0.3239, -0.1085],
             [ 0.2103, -0.3908]]])



```python
# 4계 텐서

# 요소가 모두 1인 shape=[2,3,2,2] 텐서 작성
r4 = torch.ones((2, 3, 2, 2))

# shape 확인
print(r4.shape)

# 데이터 확인
print(r4.data)
```

    torch.Size([2, 3, 2, 2])
    tensor([[[[1., 1.],
              [1., 1.]],
    
             [[1., 1.],
              [1., 1.]],
    
             [[1., 1.],
              [1., 1.]]],
    
    
            [[[1., 1.],
              [1., 1.]],
    
             [[1., 1.],
              [1., 1.]],
    
             [[1., 1.],
              [1., 1.]]]])



```python
## list, numpy to Tensor
x = [1, 2, 3]
y = torch.tensor(x)
print(y)

x1 = np.array([1, 2, 3])
# y1 = torch.tensor(x1) # 복사
y1 = torch.from_numpy(x1) # memory 공유

print(y1)

```

    tensor([1, 2, 3])
    tensor([1, 2, 3])



```python
## ones, zeros tensor
print("torch.ones:")
print(torch.ones(2))
print()
print(torch.ones(2, 3))
print()
print(torch.ones(2, 3, 2))

print("torch.zeros:")
print(torch.zeros(2))
print()
print(torch.zeros(2, 3))
print()
print(torch.zeros(2, 3, 2))
```

    torch.ones:
    tensor([1., 1.])
    
    tensor([[1., 1., 1.],
            [1., 1., 1.]])
    
    tensor([[[1., 1.],
             [1., 1.],
             [1., 1.]],
    
            [[1., 1.],
             [1., 1.],
             [1., 1.]]])
    torch.zeros:
    tensor([0., 0.])
    
    tensor([[0., 0., 0.],
            [0., 0., 0.]])
    
    tensor([[[0., 0.],
             [0., 0.],
             [0., 0.]],
    
            [[0., 0.],
             [0., 0.],
             [0., 0.]]])


### Random 텐서 만들기


```python
## tensor random
print("torch.rand: uniform distribution")
print(torch.rand(2, 3))

print()

print("torch.randn: standard normal distribution")
print(torch.randn(2, 3)) #standard normal distribution

print()

print("torch.randint: random integer")
print(torch.randint(1, 10, (2, 3)))

print("torch.normal: normal distribution")
print(torch.normal(10, 1, (2, 3)))

print("torch.randperm: random permutation")
print(torch.randperm(10))
print(torch.randperm(10).dtype)
```

    torch.rand: uniform distribution
    tensor([[0.2745, 0.6584, 0.2775],
            [0.8573, 0.8993, 0.0390]])
    
    torch.randn: standard normal distribution
    tensor([[-1.4168,  0.8429, -0.6307],
            [ 1.2340,  0.3127,  0.6972]])
    
    torch.randint: random integer
    tensor([[8, 8, 9],
            [8, 2, 3]])
    torch.normal: normal distribution
    tensor([[ 9.0399,  9.5913, 11.0764],
            [ 9.5985,  9.2709,  9.8782]])
    torch.randperm: random permutation
    tensor([0, 6, 2, 3, 4, 5, 8, 9, 7, 1])
    torch.int64


### 정숫값을 갖는 텐서 만들기


```python
# # 1계 넘파이 변수 작성
# r1_np = np.array([1, 2, 3, 4, 5])
# print(r1_np.shape)

# # 넘파이에서 텐서로 변환
# r1 = torch.tensor(r1_np).float()

r5 = r1.long()

# dtype 확인
print(r5.dtype)

# 값 확인
print(r5)
```

    torch.int64
    tensor([1, 2, 3, 4, 5])


### view 함수


```python
r3
# r3.shape
```




    tensor([[[-0.1115,  0.1204],
             [-0.3696, -0.2404]],
    
            [[-1.1969,  0.2093],
             [-0.9724, -0.7550]],
    
            [[ 0.3239, -0.1085],
             [ 0.2103, -0.3908]]])




```python
# 2계화
# 남은 자리에 -1을 지정하면 이 수를 자동으로 조정함

r6 = r3.view(3, -1) # r3.shape = (3, 2, 2)

# shape 확인
print(r6.shape)

# 값 확인
print(r6.data)
```

    torch.Size([3, 4])
    tensor([[-0.1115,  0.1204, -0.3696, -0.2404],
            [-1.1969,  0.2093, -0.9724, -0.7550],
            [ 0.3239, -0.1085,  0.2103, -0.3908]])



```python
# 1계화
# 남은 자리에 -1을 지정하면 이 수를 자동으로 조정함

r7 = r3.view(-1)

# shape 확인
print(r7.shape)

# 값 확인
print(r7.data)
```

    torch.Size([12])
    tensor([-0.1115,  0.1204, -0.3696, -0.2404, -1.1969,  0.2093, -0.9724, -0.7550,
             0.3239, -0.1085,  0.2103, -0.3908])


### Tensor reshape


```python
## tensor reshape
tensor_reshape_list=  [[1, 2, 3, 4], [5, 6, 7, 8]]
tensor_reshape = torch.tensor(tensor_reshape_list)

print("tensor_reshape:")
print(tensor_reshape)
print(tensor_reshape.shape)

print()

print("tensor_reshape.reshape(4, 2):")
print(tensor_reshape.reshape(4, 2))
print(tensor_reshape.reshape(4, 2).shape)

print()

print("tensor_reshape.reshape(4, -1):")
print(tensor_reshape.reshape(4, -1))
print(tensor_reshape.reshape(4, -1).shape)

print()

print("tensor_reshape.reshape(-1):")
print(tensor_reshape.reshape(-1))
print(tensor_reshape.reshape(-1).shape)

print()

print("tensor_reshape.reshape(-1, 2):")
print(tensor_reshape.reshape(-1, 2))
print(tensor_reshape.reshape(-1, 2).shape)

print()

print("tensor_reshape.reshape(1, 2, -1)")
print(tensor_reshape.reshape(1, 2, -1))

print()

print("tensor_reshape.reshape(-1, 2, 2)")
print(tensor_reshape.reshape(-1, 2, 2))
print(tensor_reshape.reshape(-1, 2, 2).shape)

```

    tensor_reshape:
    tensor([[1, 2, 3, 4],
            [5, 6, 7, 8]])
    torch.Size([2, 4])
    
    tensor_reshape.reshape(4, 2):
    tensor([[1, 2],
            [3, 4],
            [5, 6],
            [7, 8]])
    torch.Size([4, 2])
    
    tensor_reshape.reshape(4, -1):
    tensor([[1, 2],
            [3, 4],
            [5, 6],
            [7, 8]])
    torch.Size([4, 2])
    
    tensor_reshape.reshape(-1):
    tensor([1, 2, 3, 4, 5, 6, 7, 8])
    torch.Size([8])
    
    tensor_reshape.reshape(-1, 2):
    tensor([[1, 2],
            [3, 4],
            [5, 6],
            [7, 8]])
    torch.Size([4, 2])
    
    tensor_reshape.reshape(1, 2, -1)
    tensor([[[1, 2, 3, 4],
             [5, 6, 7, 8]]])
    
    tensor_reshape.reshape(-1, 2, 2)
    tensor([[[1, 2],
             [3, 4]],
    
            [[5, 6],
             [7, 8]]])
    torch.Size([2, 2, 2])


### Tensor squeeze


```python
tensor_squeeze = torch.tensor([[1], [2], [3]])

print("tensor_squeeze:")
print(tensor_squeeze)
print(tensor_squeeze.shape)


print()

print("tensor_squeeze.squeeze():")
print(tensor_squeeze.squeeze())
print(tensor_squeeze.squeeze().shape)

tensor_squeeze2 = torch.ones(1,2,1,4)
print()

print("tensor_squeeze2:")
print(tensor_squeeze2)
print(tensor_squeeze2.shape )
```

    tensor_squeeze:
    tensor([[1],
            [2],
            [3]])
    torch.Size([3, 1])
    
    tensor_squeeze.squeeze():
    tensor([1, 2, 3])
    torch.Size([3])
    
    tensor_squeeze2:
    tensor([[[[1., 1., 1., 1.]],
    
             [[1., 1., 1., 1.]]]])
    torch.Size([1, 2, 1, 4])



```python
## tensor arange
print(torch.arange(1, 10))
print(torch.arange(1, 10, 2))
print(torch.arange(1, 3, 0.5))
print(torch.arange(10))
print(torch.linspace(1, 10, 5))
```

    tensor([1, 2, 3, 4, 5, 6, 7, 8, 9])
    tensor([1, 3, 5, 7, 9])
    tensor([1.0000, 1.5000, 2.0000, 2.5000])
    tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    tensor([ 1.0000,  3.2500,  5.5000,  7.7500, 10.0000])


### 그 밖의 속성


```python
# requires_grad 속성
print('requires_grad: ', r1.requires_grad)

# device 속성
print('device: ', r1.device)
```

    requires_grad:  False
    device:  cpu


### item 함수


```python
# 스칼라 텐서(0계 텐서)는 item 함수로 값을 꺼낼 수 있음

item = r0.item()

print(type(item))
print(item)
```

    <class 'float'>
    1.0



```python
# 스칼라 이외의 텐서에 item 함수는 무효함

print(r1.item()) # RuntimeError: a Tensor with 5 elements cannot be converted to Scalar
```


    ---------------------------------------------------------------------------

    RuntimeError                              Traceback (most recent call last)

    <ipython-input-25-19b2fc013c51> in <cell line: 0>()
          1 # 스칼라 이외의 텐서에 item 함수는 무효함
          2 
    ----> 3 print(r1.item())
    

    RuntimeError: a Tensor with 5 elements cannot be converted to Scalar



```python
# 요소 수가 하나뿐인 1계 텐서는 OK
# (2계 이상에서도 마찬가지)
t1 = torch.ones(1)

# shape 확인
print(t1.shape)

# item 함수 호출
print(t1.item())
```

    torch.Size([1])
    1.0


### max 함수


```python
# 텐서 r2 확인
print(r2)

# max 함수를 인수 없이 호출하면, 최댓값을 얻음
print(r2.max())
```

    tensor([[1., 5., 6.],
            [4., 3., 2.]])
    tensor(6.)



```python
# torch.max 함수
# 두번째 인수는 기준이 되는 축을 의미함
print(torch.max(r2, 1))
```

    torch.return_types.max(
    values=tensor([6., 4.]),
    indices=tensor([2, 0]))



```python
# 몇 번째 요소가 최댓값인지 indices 확인을 통해 알 수 있음
# 아래 계산은 다중 분류에서 예측 라벨을 구할 때 자주 사용되는 패턴임
print(torch.max(r2, 1)[1])
```

    tensor([2, 0])


### tensor statistic


```python
## tensor statistic
tensor_stat_list = [[1, 3, 5], [2, 4, 8]]
tensor_stat = torch.tensor(tensor_stat_list, dtype=torch.float)

print("tensor_stat:")
print(tensor_stat)

print()

print(  "tensor_stat.sum():")
print(tensor_stat.sum())
print(tensor_stat.median())

print()

print("tensor_stat.sum(dim = 0):")
print(tensor_stat.sum(dim = 0))
print(tensor_stat.median(dim = 0))
print()

print("tensor_stat.sum(dim = 1):")
print(tensor_stat.sum(dim = 1))

print()

print("tensor_stat.sum(dim = -1):")
print(tensor_stat.sum(dim = -1))

print()

print("tensor_stat.mean():")
print(tensor_stat.mean())

print()

print("tensor_stat.mean(dim = 1):")
print(tensor_stat.mean(dim = 1))

print()

print("tensor_stat.std():")
print(tensor_stat.std())

print()

print("tensor_stat.max():")
print(tensor_stat.max())

print()

print("tensor_stat.min():")
print(tensor_stat.min())

print("tensor_stat.argmax()")
print(tensor_stat.argmax(dim = 1))
```

    tensor_stat:
    tensor([[1., 3., 5.],
            [2., 4., 8.]])
    
    tensor_stat.sum():
    tensor(23.)
    tensor(3.)
    
    tensor_stat.sum(dim = 0):
    tensor([ 3.,  7., 13.])
    torch.return_types.median(
    values=tensor([1., 3., 5.]),
    indices=tensor([0, 0, 0]))
    
    tensor_stat.sum(dim = 1):
    tensor([ 9., 14.])
    
    tensor_stat.sum(dim = -1):
    tensor([ 9., 14.])
    
    tensor_stat.mean():
    tensor(3.8333)
    
    tensor_stat.mean(dim = 1):
    tensor([3.0000, 4.6667])
    
    tensor_stat.std():
    tensor(2.4833)
    
    tensor_stat.max():
    tensor(8.)
    
    tensor_stat.min():
    tensor(1.)
    tensor_stat.argmax()
    tensor([2, 2])


### 넘파이 변수로 변환


```python
# 넘파이로 변환
r2_np = r2.data.numpy()

# type 확인
print(type(r2_np))

# 값 확인
print(r2_np)
```

    <class 'numpy.ndarray'>
    [[1. 5. 6.]
     [4. 3. 2.]]


## ２차 함수의 경사 계산

 ### 데이터 준비


```python
# x를 넘파이 배열로 정의
x_np = np.arange(-2, 2.1, 0.25)

# x값 표시
print(x_np)
```

    [-2.   -1.75 -1.5  -1.25 -1.   -0.75 -0.5  -0.25  0.    0.25  0.5   0.75
      1.    1.25  1.5   1.75  2.  ]



```python
# (1) 경사 계산용 변수 정의

x = torch.tensor(x_np, requires_grad=True,
    dtype=torch.float32)

# 결과 확인
print(x)
```

    tensor([-2.0000, -1.7500, -1.5000, -1.2500, -1.0000, -0.7500, -0.5000, -0.2500,
             0.0000,  0.2500,  0.5000,  0.7500,  1.0000,  1.2500,  1.5000,  1.7500,
             2.0000], requires_grad=True)


### ２차 함수 계산


```python
# 2차 함수의 계산
# 계산 그래프는 내부에서 자동 생성됨

y = 2 * x**2 + 2
```

$ y = 2x^2 + 2$를 의미한다.


```python
# y의 계산 결과 확인

print(y)
```

    tensor([10.0000,  8.1250,  6.5000,  5.1250,  4.0000,  3.1250,  2.5000,  2.1250,
             2.0000,  2.1250,  2.5000,  3.1250,  4.0000,  5.1250,  6.5000,  8.1250,
            10.0000], grad_fn=<AddBackward0>)



```python
# 그래프(산포도) 출력

plt.plot(x.data, y.data)
plt.title("Y=2X^2+2")
plt.show()
```


    
*(이미지 없음 / image not preserved: output_57_0.png)*
    



```python
# 경사 계산을 위해 최종 값은 스칼라일 필요가 있으므로, 더미로 sum 함수를 붙임

z = y.sum()
```


```python
# (3) 계산 그래프 시각화

# 필요한 라이브러리 임포트
from torchviz import make_dot

# 시각화 함수 호출
g = make_dot(z, params={'x': x})
display(g)
```


    
*(이미지 없음 / image not preserved: output_59_0.svg)*
    



```python
# (4) 경사 계산

z.backward()
```


```python
# (5) 경삿값 가져오기

print(x.grad)
```

    tensor([-8., -7., -6., -5., -4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.,  5.,
             6.,  7.,  8.])



```python
# 원래 함수와 경사 그래프

plt.plot(x.data, y.data, c='b', label='y')
plt.plot(x.data, x.grad.data, c='k', label='y.grad')
plt.legend()
plt.show()
```


    
*(이미지 없음 / image not preserved: output_62_0.png)*
    


원래 함수가 2차 함수이기 때문에 경사 계산의 결과가 직선인 것은 타당한 결과

경사 계산을 한번 더 해보자.


```python
# 경사를 초기화하지 않고 두번째 경사를 계산

y = 2 * x**2 + 2
z = y.sum()
z.backward()

# x의 경사 확인
print(x.grad)
```

    tensor([-16., -14., -12., -10.,  -8.,  -6.,  -4.,  -2.,   0.,   2.,   4.,   6.,
              8.,  10.,  12.,  14.,  16.])


경삿값은 경사 계산의 값이 점차 합해진 결과다. 새로운 값을 계산하기 위해서 경삿값을 조기화 해야한다.


```python
# (6) 경삿값의 초기화는 zero_() 함수를 사용함

x.grad.zero_()
print(x.grad)
```

    tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])


## 시그모이드 함수의 경사 계산

시그모이드 함수는 수식으로 나타내면 다음과 같지만, 여기서는 파이토치에서 제공하는 함수를 사용한다.

$ y = \dfrac{1}{1 + \exp{(-x)}} $


```python
# 시그모이드 함수의 정의

sigmoid = torch.nn.Sigmoid()
```


```python
# (2) y값의 계산

y = sigmoid(x)
```


```python
# 그래프(산포도) 출력

plt.plot(x.data, y.data)
plt.show()
```


    
*(이미지 없음 / image not preserved: output_72_0.png)*
    



```python
# 경사 계산을 위해 최종 값은 스칼라일 필요가 있으므로, 더미로 sum 함수를 붙임

z = y.sum()
```


```python
# (3) 계산 그래프 시각화

g = make_dot(z, params={'x': x})
display(g)
```


    
*(이미지 없음 / image not preserved: output_74_0.svg)*
    



```python
# (4) 경사 계산
z.backward()

# (5) 경삿값 확인
print(x.grad)
```

    tensor([0.1050, 0.1261, 0.1491, 0.1731, 0.1966, 0.2179, 0.2350, 0.2461, 0.2500,
            0.2461, 0.2350, 0.2179, 0.1966, 0.1731, 0.1491, 0.1261, 0.1050])



```python
# 원래 함수와 경사 그래프

plt.plot(x.data, y.data, c='b', label='y')
plt.plot(x.data, x.grad.data, c='k', label='y.grad')
plt.legend()
plt.show()
```


    
*(이미지 없음 / image not preserved: output_76_0.png)*
    


시그모이드 함수의 경사는 $y(1-y)$ 이다.  
2차 함수이므로, $y=\dfrac{1}{2}$(x=0일때) 최댓값 $\dfrac{1}{4}$을 얻는다. 위의 그래프는 이 계산 결과와 일치한다.


```python
# (6) 경사 초기화는 zero_() 함수를 사용

x.grad.zero_()
print(x.grad)
```

    tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])


### (참고) 시그모이드 함수를 직접 구현한 경우


```python
# 시그모이드 함수의 정의

def sigmoid(x):
    return(1/(1 + torch.exp(-x)))
```


```python
# (2) y값의 계산

y = sigmoid(x)
```


```python
# 그래프(산포도) 출력

plt.plot(x.data, y.data)
plt.xlabel('x')
plt.ylabel('y')
plt.title('시그모이드 함수 그래프')
plt.show()
```


    
*(이미지 없음 / image not preserved: output_82_0.png)*
    



```python
# 경사 계산을 위해 최종 값은 스칼라일 필요가 있으므로, 더미로 sum 함수를 붙임

z = y.sum()
```


```python
# (3) 계산 그래프 시각화

params = {'x': x}
g = make_dot(z, params=params)
display(g)
```


    
*(이미지 없음 / image not preserved: output_84_0.svg)*
    



```python
# (4) 경사 계산
z.backward()

# (5) 경삿값 확인
print(x.grad)
```

    tensor([0.1050, 0.1261, 0.1491, 0.1731, 0.1966, 0.2179, 0.2350, 0.2461, 0.2500,
            0.2461, 0.2350, 0.2179, 0.1966, 0.1731, 0.1491, 0.1261, 0.1050])



```python
# 원래 함수와 경사 그래프

plt.plot(x.data, y.data, c='b', label='y')
plt.plot(x.data, x.grad.data, c='k', label='y.grad')
plt.legend()
plt.show()
```


    
*(이미지 없음 / image not preserved: output_86_0.png)*
    

