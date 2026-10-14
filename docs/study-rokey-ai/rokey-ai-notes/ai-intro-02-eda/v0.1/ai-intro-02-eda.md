# 2장 탐색적 자료 분석 (EDA)

> **Migrated from Quartz wiki** (`wiki/study/rokey-ai/basic/ai-intro-02-eda.md`, updated 2026-09-26)  
> tags: rokey, ai, notebook, basic, study  
> source: raw/2026-09-26-rokey-ai/basic/강의_3기_AI개론_2차시__EDA_.ipynb  
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
# path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
# font_name = fm.FontProperties(fname=path, size=10).get_name()

# Window
font_name = "NanumBarunGothic"

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

## Numpy 를 이용한 행렬 연산


```python
# from IPython.display import Image
# Image('../../../fig/numpy_axis.png')
```

### numpy class 속성


```python
## numpy class 속성
array = np.array([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]], dtype = np.int32)

print(array)
print("array data type =", array.dtype)
print(array.shape)
print(array.ndim)
print(array.size)
print(array.dtype)
array
```

    [[1 2 3]
     [4 5 6]
     [7 8 9]]
    array data type = int32
    (3, 3)
    2
    9
    int32





    array([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]], dtype=int32)



### Indexing


```python
## Indexing
print("array = \n", array)

array[:, :]

## row operation
array[:2]
array[[0, 2]]

## column operation
array[:, :2]
array[:, [0, 1]]

## row x column operation
array[[0, 2], [1, 2]]
```

    array = 
     [[1 2 3]
     [4 5 6]
     [7 8 9]]





    array([2, 9], dtype=int32)



### 특별한 행렬


```python
## 특별한 행렬
print('np.eye(2, 3) = \n', np.eye(2, 3))
print('np.identity(3) = \n', np.identity(3))
print('np.zeros([3,3]) = \n', np.zeros([3,3]))
print('np.ones((3,5) = \n', np.ones((3,5)))
print('np.full((3,5), 5) = \n', np.full((3,5), 5))
```

    np.eye(2, 3) = 
     [[1. 0. 0.]
     [0. 1. 0.]]
    np.identity(3) = 
     [[1. 0. 0.]
     [0. 1. 0.]
     [0. 0. 1.]]
    np.zeros([3,3]) = 
     [[0. 0. 0.]
     [0. 0. 0.]
     [0. 0. 0.]]
    np.ones((3,5) = 
     [[1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1.]]
    np.full((3,5), 5) = 
     [[5 5 5 5 5]
     [5 5 5 5 5]
     [5 5 5 5 5]]


### 행렬 변환


```python
## 행렬 변환
x = np.arange(12)
print(x)
```

    [ 0  1  2  3  4  5  6  7  8  9 10 11]



```python
y = np.linspace(0, 11, 12)
print(y)
```

    [ 0.  1.  2.  3.  4.  5.  6.  7.  8.  9. 10. 11.]



```python
## noarray.flat, Transpose

x = np.arange(1,7).reshape(2, 3)
print('x = \n', x)
print('x.T = \n', x.T)

##
print("matrix multiplication = \n", np.matmul(x, x.T))
```

    x = 
     [[1 2 3]
     [4 5 6]]
    x.T = 
     [[1 4]
     [2 5]
     [3 6]]
    matrix multiplication = 
     [[14 32]
     [32 77]]



```python
## transpose
x = np.linspace(2, 10, 6)
y = x.reshape(2, -1)
print ("y = ",  y)
print('y.transpose(0, 1) = \n', y.transpose(1, 0))
```

    y =  [[ 2.   3.6  5.2]
     [ 6.8  8.4 10. ]]
    y.transpose(0, 1) = 
     [[ 2.   6.8]
     [ 3.6  8.4]
     [ 5.2 10. ]]


### squeeze


```python
## squeeze
x = np.array([[[0],[1],[2]]])
print('x = \n', x)
print('x.shape = ', x.shape)
print('x.squeeze() = \n', x.squeeze())

```

    x = 
     [[[0]
      [1]
      [2]]]
    x.shape =  (1, 3, 1)
    x.squeeze() = 
     [0 1 2]


### 배열객체 연결


```python
# 배열객체 연결
list1 = [[1,2,3]]
list2 = [[4,5,5]]

print(np.shape(list1))
print('concatenate = ')
print(np.concatenate([list1, list2], axis = 0))
print(np.concatenate([list1, list2], axis = 1))

print('vstack = ')
print(np.vstack((list1, list2)))

print('hstack = ')
print(np.hstack((list1, list2)))
```

    (1, 3)
    concatenate = 
    [[1 2 3]
     [4 5 5]]
    [[1 2 3 4 5 5]]
    vstack = 
    [[1 2 3]
     [4 5 5]]
    hstack = 
    [[1 2 3 4 5 5]]


### min, max


```python
## min, max
a = np.arange(4).reshape((2,2))
print(a)

print(np.min(a))
print(np.min(a, axis = 0))
print(np.min(a, axis = 1))

np.min(a,axis = 0)
```

    [[0 1]
     [2 3]]
    0
    [0 1]
    [0 2]





    array([0, 1])



### mean, median, var, std


```python
# mean, median, var, std
a = np.array([[1, 2], [3, 4]])
print(a)

print("mean of a = ")
print(np.mean(a))
print("mean of a along axis = 0 :")
print(np.mean(a, axis = 0))
print("mean of a along axis = 1 : ")
print(np.mean(a, axis = 1))
```

    [[1 2]
     [3 4]]
    mean of a = 
    2.5
    mean of a along axis = 0 :
    [2. 3.]
    mean of a along axis = 1 : 
    [1.5 3.5]


### 랜덤값 생성


```python
# Uniform [0, 1]
print("Uniform random sampling = ")
np.random.rand(3,2)
```

    Uniform random sampling = 





    array([[0.9808, 0.6848],
           [0.4809, 0.3921],
           [0.3432, 0.729 ]])




```python
# Uniform integer between [1, 10]
# randint(low, high=None, size=None, dtype=int)
print("Uniform integer random sampling = ")
np.random.randint(1, 10, (3,2))
```

    Uniform integer random sampling = 





    array([[4, 5],
           [1, 1],
           [5, 2]])




```python
# from the "standard normal" distribution
np.random.seed(123)

print("Standard normal random sampling = ")
np.random.randn(3,2)
```

    Standard normal random sampling = 





    array([[-1.0856,  0.9973],
           [ 0.283 , -1.5063],
           [-0.5786,  1.6514]])



## Matplotlib을 이용한 자료 시각화

### 산포도


```python
# 데이터 준비
import seaborn as sns
df_iris = sns.load_dataset("iris")

# 결과 확인
print(df_iris.head())

# 산포도의 x좌표용 배열
xs = df_iris['sepal_length'].values

# 산포도의 y좌표용 배열
ys = df_iris['sepal_width'].values
```

       sepal_length  sepal_width  petal_length  petal_width species
    0           5.1          3.5           1.4          0.2  setosa
    1           4.9          3.0           1.4          0.2  setosa
    2           4.7          3.2           1.3          0.2  setosa
    3           4.6          3.1           1.5          0.2  setosa
    4           5.0          3.6           1.4          0.2  setosa



```python
# 산포도 그리기
plt.scatter(xs, ys)

# 출력
plt.show()
```


    
*(이미지 없음 / image not preserved: output_34_0.png)*
    


### 함수 그래프


```python
# 데이터 준비

# 시그모이드 함수 정의
def sigmoid(x, a):
    return 1/(1 + np.exp(-a*x))

# 그래프를 그리기 위한 x좌표 리스트
xp = np.linspace(-3, 3, 61)
yp = sigmoid(xp, 1.0)
yp2 = sigmoid(xp, 2.0)
```


```python
# 그래프 그리기
plt.plot(xp, yp)

# 출력
plt.show()
```


    
*(이미지 없음 / image not preserved: output_37_0.png)*
    



```python
# 라벨을 포함한 그래프 출력 #1
plt.plot(xp, yp,
         label='시그모이드 함수 1', lw=3, c='k')

# 라벨을 포함한 그래프 출력 #2
plt.plot(xp, yp2,
         label='시그모이드 함수 2', lw=2, c='b')

# 범례 표시
plt.legend()

# 축 표시
plt.xlabel('x 축')
plt.ylabel('y 축')

# 출력
plt.show()
```


    
*(이미지 없음 / image not preserved: output_38_0.png)*
    


### subplot을 사용한 그래프 동시 출력


```python
# 손글씨 숫자 데이터
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1,as_frame=False)

# 이미지 데이터
image = mnist.data
# 정답 데이터
label = mnist.target
```


```python
# 사이즈 지정
plt.figure(figsize=(10, 3))

# 20개 이미지를 표시
for i in range(20):

    # i번째 ax 변수 취득
    ax = plt.subplot(2, 10, i+1)

    # i번째 이미지 데이터를 취득한 다음 28x28로 변환
    img = image[i].reshape(28,28)

    # img를 이미지로 표시
    ax.imshow(img, cmap='gray_r')

    # 정답 데이터를 타이틀로 표시
    ax.set_title(label[i])

    # x, y 눈금 표시하지 않음
    ax.set_xticks([])
    ax.set_yticks([])

# 인접 객체와 겹치지 않도록 함
plt.tight_layout()

# 출력
plt.show()
```


    
*(이미지 없음 / image not preserved: output_41_0.png)*
    

