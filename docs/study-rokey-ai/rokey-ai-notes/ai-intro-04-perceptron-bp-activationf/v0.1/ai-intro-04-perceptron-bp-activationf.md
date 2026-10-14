# 4장 인공지능 개론

> **Migrated from Quartz wiki** (`wiki/study/rokey-ai/basic/ai-intro-04-perceptron_bp_activationf.md`, updated 2026-09-26)  
> tags: rokey, ai, notebook, basic, study  
> source: raw/2026-09-26-rokey-ai/basic/강의_3기_AI개론_4차시__Perceptron_BP_ActivationF_.ipynb  
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

## 퍼셉트론 (Perceptron)


```python
import torch
from torch import nn
import torch.nn.functional as F
```

### 선형 퍼셉트론 구현


```python
## 선형 퍼셉트론 구현

def perceptron(x, w, b = 0):
    if not isinstance(x, np.ndarray) or not isinstance(x, np.ndarray):
        x = np.array(x)
        w = np.array(w)
    y = np.sum(x*w) + b
    return y

x = [1, 2, 3]
w = [1, 2, 3]
# x = np.array([1, 2, 3])
# w = np.array([1, 2, 3])

y = perceptron(x, w)
print("result =", y)
```

    result = 14



```python
## 선형 퍼셉트론 구현: 랜텀 initial weights

def perceptron(x):
    w = np.random.rand(len(x))
    b = np.random.rand()
    y = np.sum(x*w) + b
    return y, w, b

x = np.array([1, 2, 3])
w = np.array([1, 2, 3])

y, w, b = perceptron(x)
print("result =", y.round(3))
print("weight = ", w)
print("b = ", round(b, 3))

```

    result = 3.872
    weight =  [0.6354 0.9663 0.1028]
    b =  0.996


### Pytorch 를 이용한 perceptron


```python
## Pytorch 를 이용한 perceptron
# torch.manual_seed(0)

input_size = 10
output_size = 1
y = nn.Linear(input_size, output_size)

print("weights = \n", y.weight)
print("bias = \n", y.bias)
print("="*50)

# 난수 생성
x = torch.rand(input_size)
print("x input = \n", x)
print("output = ", y(x))
```

    weights = 
     Parameter containing:
    tensor([[-0.2100, -0.1679, -0.2891,  0.2916,  0.1475,  0.0402, -0.2919, -0.1315,
             -0.0679,  0.0026]], requires_grad=True)
    bias = 
     Parameter containing:
    tensor([-0.0885], requires_grad=True)
    ==================================================
    x input = 
     tensor([0.6266, 0.7196, 0.6680, 0.5833, 0.0691, 0.9878, 0.5864, 0.4230, 0.2783,
            0.4751])
    output =  tensor([-0.5584], grad_fn=<ViewBackward0>)


## 오류 순전파 (Forward propagation)


```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sse(y, x):
    return (y-x)**2

def perceptron(x, w, b = 0):
    if not isinstance(x, np.ndarray) or not isinstance(x, np.ndarray):
        x = np.array(x)
        w = np.array(w)
    y = np.sum(x*w) + b
    return y
```

## 활성화 함수 (Activation function)


```python
## 시그모이드 (sigmoid)
def activation_plot(x, y, title:str):
    plt.plot(x, y)
    plt.title(title)
    # plt.grid(linestyle = ":")
    plt.show()
```


```python
# numpy
x = np.linspace(-5, 5, 100)
def sigmoid(x):
    y = 1/(1 + np.exp(-x))
    return y
y = sigmoid(x)

# torch class
# torch_simoid = nn.Sigmoid()
# y = torch_simoid(torch.tensor(x))
# y = F.sigmoid(torch.tensor(x))

activation_plot(x, y, "Sigmoid")
```

    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.



    
*(이미지 없음 / image not preserved: output_18_1.png)*
    



```python
## Tanh (Hyperbolic Tangent)
def tanh(x):
    y = np.tanh(x)
        # y = (np.exp(x) - np.exp(-x))/ (np.exp(x) + np.exp(-x))
    return y
y = tanh(x)

# torch
torch_tanh = nn.Tanh()
y = torch_tanh(torch.tensor(x))
y = F.tanh(torch.tensor(x))

activation_plot(x, y, "Tanh")
```

    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.



    
*(이미지 없음 / image not preserved: output_19_1.png)*
    



```python
## ReLU (Rectified Linear Unit)
def relu(x):
    y = np.maximum(0, x)
    return y

# torch
# torch_relu = nn.ReLU()
# y = torch_relu(torch.tensor(x))
# y = F.relu(torch.tensor(x))

# y = relu(x)
activation_plot(x, y, "ReLU")
```

    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.



    
*(이미지 없음 / image not preserved: output_20_1.png)*
    



```python
## Leaky ReLU
def leakyRelu(x, alpha:float):
    y = np.maximum(alpha*x, x)
    return y
y = leakyRelu(x, 0.1)

# torch
# torch_lrelu = nn.LeakyReLU(0.1)
# y = torch_lrelu(torch.tensor(x))
# y = F.leaky_relu(torch.tensor(x), 0.1)


activation_plot(x, y, "Leaky ReLU")
```

    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.



    
*(이미지 없음 / image not preserved: output_21_1.png)*
    



```python
## Exponential ReLU
def eRelu(x, alpha):
    y = (x>0)*x + (x <= 0)*(alpha*(np.exp(x) - 1))
    return y

y = eRelu(x, 1.)

# torch
# torch_lrelu = nn.GELU(0.1)
# y = torch_lrelu(torch.tensor(x))
# y = F.leaky_relu(torch.tensor(x), 0.1)

activation_plot(x, y, "Exponential ReLU")
```

    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.
    WARNING:matplotlib.font_manager:findfont: Font family 'NanumGothic' not found.



    
*(이미지 없음 / image not preserved: output_22_1.png)*
    

