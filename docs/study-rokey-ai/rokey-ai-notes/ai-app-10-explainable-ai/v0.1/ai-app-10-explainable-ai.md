# 10장 설명가능한 AI (Explainable AI)

> **Migrated from Quartz wiki** (`wiki/study/rokey-ai/intermediate/ai-app-10-explainable_ai.md`, updated 2026-09-26)  
> tags: rokey, ai, notebook, intermediate, study  
> source: raw/2026-09-26-rokey-ai/intermediate/강의_3기_AI응용_10차시__Explainable_AI_.ipynb  
> source: https://github.com/weedmo/Rokey-AI  

* "부록3 매트플롯립 입문"에서 한글 폰트를 올바르게 출력하기 위한 설치 방법을 설명했다. 설치 방법은 다음과 같다.


```python
# 한글 폰트 설치

!sudo apt-get install -y fonts-nanum* | tail -n 1
!sudo fc-cache -fv
!rm -rf ~/.cache/matplotlib
```

* 모든 설치가 끝나면 한글 폰트를 바르게 출력하기 위해 **[런타임]** -> **[런타임 다시시작]**을 클릭한 다음, 아래 셀부터 코드를 실행해 주십시오.


```python
# 라이브러리 임포트

%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
# from IPython.display import display

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

## Import modules


```python
import os
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import ipdb

```


```python
np.random.seed(123)
torch.manual_seed(123)
```




    <torch._C.Generator at 0x26338c6f8f0>




```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device = {device}")
```

    device = cuda


## Define model architecture


```python
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.cn1 = nn.Conv2d(1, 16, 3, 1)
        self.cn2 = nn.Conv2d(16, 32, 3, 1)
        self.dp1 = nn.Dropout2d(0.10)
        self.dp2 = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(4608, 64) # 4608 is basically 12 X 12 X 32
        self.fc2 = nn.Linear(64, 10)
 
    def forward(self, x):
        x = self.cn1(x)
        x = F.relu(x)
        x = self.cn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dp1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dp2(x)
        x = self.fc2(x)
        op = F.log_softmax(x, dim=1)
        return op
```

## Define training and inference routines


```python
def train(model, device, train_dataloader, optim, epoch):
    model.train()
    for b_i, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)
        
        pred_prob = model(X)
        loss = F.nll_loss(pred_prob, y) # nll is the negative likelihood loss
        
        optim.zero_grad()
        loss.backward()
        optim.step()
        if b_i % 100 == 0:
            print('epoch: {} [{}/{} ({:.0f}%)]\t training loss: {:.6f}'.format(
                epoch, b_i * len(X), len(train_dataloader.dataset),
                100. * b_i / len(train_dataloader), 
                loss.item()))
```


```python

def test(model, device, test_dataloader):
    model.eval()
    loss = 0
    success = 0
    with torch.no_grad():
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)
            pred_prob = model(X)
            loss += F.nll_loss(pred_prob, y).item()  # loss summed across the batch
            pred = pred_prob.argmax(dim=1)  # us argmax to get the most likely prediction
            # ipdb.set_trace()
            # success += pred.eq(y.view_as(pred)).sum().item()
            success += (pred == y).float().mean()

    loss /= len(test_dataloader)
    success /= len(test_dataloader)
    print('\nTest dataset: Overall Loss: {:.4f}, Overall Accuracy: {:.3f}%'.format(
        loss, 100. * success))

```

## Create data loaders


```python
# The mean and standard deviation values are calculated as the mean of all pixel values of all images in the training dataset
path = os.path.join(os.getcwd(), "data")

train_dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(path, train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1302,), (0.3069,))])), # train_X.mean()/256. and train_X.std()/256.
                    batch_size=32, shuffle=True)

test_dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(path, train=False, 
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1302,), (0.3069,)) 
                   ])),
                    batch_size=500, shuffle=True)
```

## Define optimizer and run training epochs


```python
# device = torch.device("cuda")

model = ConvNet().to(device)
optimizer = optim.Adadelta(model.parameters(), lr=0.5)
```

## model training


```python
for epoch in range(1, 10):
    train(model, device, train_dataloader, optimizer, epoch)
    test(model, device, test_dataloader)
```

    c:\Users\user\anaconda3\envs\torchgpu_py3.12\Lib\site-packages\torch\nn\functional.py:1538: UserWarning: dropout2d: Received a 2-D input to dropout2d, which is deprecated and will result in an error in a future release. To retain the behavior and silence this warning, please use dropout instead. Note that dropout2d exists to provide channel-wise dropout on inputs with 2 spatial dimensions, a channel dimension, and an optional batch dimension (i.e. 3D or 4D inputs).
      warnings.warn(warn_msg)


    epoch: 1 [0/60000 (0%)]	 training loss: 2.309232
    epoch: 1 [3200/60000 (5%)]	 training loss: 0.278278
    epoch: 1 [6400/60000 (11%)]	 training loss: 0.121370
    epoch: 1 [9600/60000 (16%)]	 training loss: 0.181576
    epoch: 1 [12800/60000 (21%)]	 training loss: 0.101811
    epoch: 1 [16000/60000 (27%)]	 training loss: 0.068833
    epoch: 1 [19200/60000 (32%)]	 training loss: 0.138396
    epoch: 1 [22400/60000 (37%)]	 training loss: 0.189149
    epoch: 1 [25600/60000 (43%)]	 training loss: 0.251213
    epoch: 1 [28800/60000 (48%)]	 training loss: 0.009341
    epoch: 1 [32000/60000 (53%)]	 training loss: 0.052232
    epoch: 1 [35200/60000 (59%)]	 training loss: 0.160566
    epoch: 1 [38400/60000 (64%)]	 training loss: 0.013284
    epoch: 1 [41600/60000 (69%)]	 training loss: 0.027691
    epoch: 1 [44800/60000 (75%)]	 training loss: 0.081332
    epoch: 1 [48000/60000 (80%)]	 training loss: 0.237036
    epoch: 1 [51200/60000 (85%)]	 training loss: 0.116369
    epoch: 1 [54400/60000 (91%)]	 training loss: 0.025693
    epoch: 1 [57600/60000 (96%)]	 training loss: 0.074641
    
    Test dataset: Overall Loss: 0.0448, Overall Accuracy: 98.510%
    epoch: 2 [0/60000 (0%)]	 training loss: 0.123495
    epoch: 2 [3200/60000 (5%)]	 training loss: 0.045361
    epoch: 2 [6400/60000 (11%)]	 training loss: 0.049744
    epoch: 2 [9600/60000 (16%)]	 training loss: 0.058824
    epoch: 2 [12800/60000 (21%)]	 training loss: 0.249931
    epoch: 2 [16000/60000 (27%)]	 training loss: 0.171683
    epoch: 2 [19200/60000 (32%)]	 training loss: 0.007672
    epoch: 2 [22400/60000 (37%)]	 training loss: 0.005648
    epoch: 2 [25600/60000 (43%)]	 training loss: 0.003113
    epoch: 2 [28800/60000 (48%)]	 training loss: 0.052169
    epoch: 2 [32000/60000 (53%)]	 training loss: 0.011703
    epoch: 2 [35200/60000 (59%)]	 training loss: 0.157394
    epoch: 2 [38400/60000 (64%)]	 training loss: 0.191598
    epoch: 2 [41600/60000 (69%)]	 training loss: 0.048679
    epoch: 2 [44800/60000 (75%)]	 training loss: 0.025116
    epoch: 2 [48000/60000 (80%)]	 training loss: 0.101853
    epoch: 2 [51200/60000 (85%)]	 training loss: 0.054859
    epoch: 2 [54400/60000 (91%)]	 training loss: 0.003429
    epoch: 2 [57600/60000 (96%)]	 training loss: 0.018314
    
    Test dataset: Overall Loss: 0.0362, Overall Accuracy: 98.750%
    epoch: 3 [0/60000 (0%)]	 training loss: 0.013529
    epoch: 3 [3200/60000 (5%)]	 training loss: 0.099732
    epoch: 3 [6400/60000 (11%)]	 training loss: 0.005558
    epoch: 3 [9600/60000 (16%)]	 training loss: 0.015110
    epoch: 3 [12800/60000 (21%)]	 training loss: 0.033571
    epoch: 3 [16000/60000 (27%)]	 training loss: 0.064919
    epoch: 3 [19200/60000 (32%)]	 training loss: 0.008560
    epoch: 3 [22400/60000 (37%)]	 training loss: 0.008617
    epoch: 3 [25600/60000 (43%)]	 training loss: 0.021366
    epoch: 3 [28800/60000 (48%)]	 training loss: 0.058135
    epoch: 3 [32000/60000 (53%)]	 training loss: 0.038140
    epoch: 3 [35200/60000 (59%)]	 training loss: 0.168531
    epoch: 3 [38400/60000 (64%)]	 training loss: 0.005081
    epoch: 3 [41600/60000 (69%)]	 training loss: 0.124696
    epoch: 3 [44800/60000 (75%)]	 training loss: 0.001875
    epoch: 3 [48000/60000 (80%)]	 training loss: 0.017088
    epoch: 3 [51200/60000 (85%)]	 training loss: 0.087384
    epoch: 3 [54400/60000 (91%)]	 training loss: 0.004997
    epoch: 3 [57600/60000 (96%)]	 training loss: 0.001871
    
    Test dataset: Overall Loss: 0.0320, Overall Accuracy: 98.920%
    epoch: 4 [0/60000 (0%)]	 training loss: 0.074035
    epoch: 4 [3200/60000 (5%)]	 training loss: 0.002991
    epoch: 4 [6400/60000 (11%)]	 training loss: 0.009879
    epoch: 4 [9600/60000 (16%)]	 training loss: 0.135356
    epoch: 4 [12800/60000 (21%)]	 training loss: 0.002469
    epoch: 4 [16000/60000 (27%)]	 training loss: 0.241587
    epoch: 4 [19200/60000 (32%)]	 training loss: 0.010140
    epoch: 4 [22400/60000 (37%)]	 training loss: 0.000642
    epoch: 4 [25600/60000 (43%)]	 training loss: 0.060177
    epoch: 4 [28800/60000 (48%)]	 training loss: 0.056297
    epoch: 4 [32000/60000 (53%)]	 training loss: 0.014556
    epoch: 4 [35200/60000 (59%)]	 training loss: 0.001380
    epoch: 4 [38400/60000 (64%)]	 training loss: 0.007528
    epoch: 4 [41600/60000 (69%)]	 training loss: 0.003082
    epoch: 4 [44800/60000 (75%)]	 training loss: 0.010884
    epoch: 4 [48000/60000 (80%)]	 training loss: 0.003054
    epoch: 4 [51200/60000 (85%)]	 training loss: 0.056021
    epoch: 4 [54400/60000 (91%)]	 training loss: 0.002138
    epoch: 4 [57600/60000 (96%)]	 training loss: 0.004467
    
    Test dataset: Overall Loss: 0.0305, Overall Accuracy: 99.040%
    epoch: 5 [0/60000 (0%)]	 training loss: 0.000518
    epoch: 5 [3200/60000 (5%)]	 training loss: 0.001188
    epoch: 5 [6400/60000 (11%)]	 training loss: 0.009993
    epoch: 5 [9600/60000 (16%)]	 training loss: 0.013775
    epoch: 5 [12800/60000 (21%)]	 training loss: 0.014831
    epoch: 5 [16000/60000 (27%)]	 training loss: 0.020766
    epoch: 5 [19200/60000 (32%)]	 training loss: 0.000947
    epoch: 5 [22400/60000 (37%)]	 training loss: 0.001270
    epoch: 5 [25600/60000 (43%)]	 training loss: 0.022234
    epoch: 5 [28800/60000 (48%)]	 training loss: 0.004858
    epoch: 5 [32000/60000 (53%)]	 training loss: 0.002067
    epoch: 5 [35200/60000 (59%)]	 training loss: 0.019483
    epoch: 5 [38400/60000 (64%)]	 training loss: 0.132162
    epoch: 5 [41600/60000 (69%)]	 training loss: 0.000567
    epoch: 5 [44800/60000 (75%)]	 training loss: 0.005990
    epoch: 5 [48000/60000 (80%)]	 training loss: 0.050496
    epoch: 5 [51200/60000 (85%)]	 training loss: 0.025538
    epoch: 5 [54400/60000 (91%)]	 training loss: 0.004219
    epoch: 5 [57600/60000 (96%)]	 training loss: 0.009227
    
    Test dataset: Overall Loss: 0.0328, Overall Accuracy: 99.060%
    epoch: 6 [0/60000 (0%)]	 training loss: 0.007086
    epoch: 6 [3200/60000 (5%)]	 training loss: 0.000884
    epoch: 6 [6400/60000 (11%)]	 training loss: 0.006149
    epoch: 6 [9600/60000 (16%)]	 training loss: 0.000063
    epoch: 6 [12800/60000 (21%)]	 training loss: 0.045278
    epoch: 6 [16000/60000 (27%)]	 training loss: 0.055924
    epoch: 6 [19200/60000 (32%)]	 training loss: 0.004901
    epoch: 6 [22400/60000 (37%)]	 training loss: 0.000259
    epoch: 6 [25600/60000 (43%)]	 training loss: 0.011155
    epoch: 6 [28800/60000 (48%)]	 training loss: 0.120620
    epoch: 6 [32000/60000 (53%)]	 training loss: 0.022840
    epoch: 6 [35200/60000 (59%)]	 training loss: 0.488387
    epoch: 6 [38400/60000 (64%)]	 training loss: 0.012747
    epoch: 6 [41600/60000 (69%)]	 training loss: 0.010363
    epoch: 6 [44800/60000 (75%)]	 training loss: 0.003192
    epoch: 6 [48000/60000 (80%)]	 training loss: 0.001400
    epoch: 6 [51200/60000 (85%)]	 training loss: 0.055191
    epoch: 6 [54400/60000 (91%)]	 training loss: 0.001423
    epoch: 6 [57600/60000 (96%)]	 training loss: 0.006740
    
    Test dataset: Overall Loss: 0.0376, Overall Accuracy: 98.780%
    epoch: 7 [0/60000 (0%)]	 training loss: 0.117586
    epoch: 7 [3200/60000 (5%)]	 training loss: 0.013059
    epoch: 7 [6400/60000 (11%)]	 training loss: 0.035918
    epoch: 7 [9600/60000 (16%)]	 training loss: 0.036338
    epoch: 7 [12800/60000 (21%)]	 training loss: 0.007791
    epoch: 7 [16000/60000 (27%)]	 training loss: 0.000122
    epoch: 7 [19200/60000 (32%)]	 training loss: 0.004934
    epoch: 7 [22400/60000 (37%)]	 training loss: 0.000408
    epoch: 7 [25600/60000 (43%)]	 training loss: 0.004158
    epoch: 7 [28800/60000 (48%)]	 training loss: 0.002284
    epoch: 7 [32000/60000 (53%)]	 training loss: 0.089690
    epoch: 7 [35200/60000 (59%)]	 training loss: 0.000036
    epoch: 7 [38400/60000 (64%)]	 training loss: 0.004212
    epoch: 7 [41600/60000 (69%)]	 training loss: 0.000579
    epoch: 7 [44800/60000 (75%)]	 training loss: 0.002025
    epoch: 7 [48000/60000 (80%)]	 training loss: 0.000779
    epoch: 7 [51200/60000 (85%)]	 training loss: 0.021036
    epoch: 7 [54400/60000 (91%)]	 training loss: 0.034359
    epoch: 7 [57600/60000 (96%)]	 training loss: 0.000122
    
    Test dataset: Overall Loss: 0.0350, Overall Accuracy: 99.070%
    epoch: 8 [0/60000 (0%)]	 training loss: 0.000672
    epoch: 8 [3200/60000 (5%)]	 training loss: 0.010706
    epoch: 8 [6400/60000 (11%)]	 training loss: 0.000801
    epoch: 8 [9600/60000 (16%)]	 training loss: 0.000360
    epoch: 8 [12800/60000 (21%)]	 training loss: 0.034542
    epoch: 8 [16000/60000 (27%)]	 training loss: 0.007012
    epoch: 8 [19200/60000 (32%)]	 training loss: 0.003235
    epoch: 8 [22400/60000 (37%)]	 training loss: 0.012091
    epoch: 8 [25600/60000 (43%)]	 training loss: 0.001134
    epoch: 8 [28800/60000 (48%)]	 training loss: 0.020984
    epoch: 8 [32000/60000 (53%)]	 training loss: 0.038265
    epoch: 8 [35200/60000 (59%)]	 training loss: 0.070304
    epoch: 8 [38400/60000 (64%)]	 training loss: 0.017858
    epoch: 8 [41600/60000 (69%)]	 training loss: 0.012005
    epoch: 8 [44800/60000 (75%)]	 training loss: 0.017090
    epoch: 8 [48000/60000 (80%)]	 training loss: 0.000842
    epoch: 8 [51200/60000 (85%)]	 training loss: 0.000052
    epoch: 8 [54400/60000 (91%)]	 training loss: 0.215633
    epoch: 8 [57600/60000 (96%)]	 training loss: 0.001072
    
    Test dataset: Overall Loss: 0.0299, Overall Accuracy: 99.180%
    epoch: 9 [0/60000 (0%)]	 training loss: 0.000083
    epoch: 9 [3200/60000 (5%)]	 training loss: 0.011790
    epoch: 9 [6400/60000 (11%)]	 training loss: 0.040857
    epoch: 9 [9600/60000 (16%)]	 training loss: 0.009835
    epoch: 9 [12800/60000 (21%)]	 training loss: 0.016525
    epoch: 9 [16000/60000 (27%)]	 training loss: 0.006370
    epoch: 9 [19200/60000 (32%)]	 training loss: 0.013823
    epoch: 9 [22400/60000 (37%)]	 training loss: 0.031687
    epoch: 9 [25600/60000 (43%)]	 training loss: 0.013757
    epoch: 9 [28800/60000 (48%)]	 training loss: 0.004209
    epoch: 9 [32000/60000 (53%)]	 training loss: 0.003744
    epoch: 9 [35200/60000 (59%)]	 training loss: 0.000545
    epoch: 9 [38400/60000 (64%)]	 training loss: 0.034161
    epoch: 9 [41600/60000 (69%)]	 training loss: 0.011542
    epoch: 9 [44800/60000 (75%)]	 training loss: 0.008834
    epoch: 9 [48000/60000 (80%)]	 training loss: 0.003623
    epoch: 9 [51200/60000 (85%)]	 training loss: 0.001592
    epoch: 9 [54400/60000 (91%)]	 training loss: 0.015191
    epoch: 9 [57600/60000 (96%)]	 training loss: 0.000761
    
    Test dataset: Overall Loss: 0.0331, Overall Accuracy: 99.070%


## Run inference on trained model


```python
test_samples = enumerate(test_dataloader)
b_i, (sample_data, sample_targets) = next(test_samples)

idx = np.random.randint(0, len(sample_targets))
print(sample_targets[idx].item())
plt.imshow(sample_data[idx][0], cmap='gray', interpolation='none')
plt.show()
```

    1



    
*(이미지 없음 / image not preserved: output_22_1.png)*
    



```python
idx = np.random.randint(0, len(sample_targets))
sample_data = sample_data.to(device)
print(f"Model prediction is : {model(sample_data).data.max(1)[1][idx]}")
print(f"Ground truth is : {sample_targets[idx]}")
```

    Model prediction is : 1
    Ground truth is : 1


### visualize filters


```python
dict(model.named_parameters()).keys()
```




    dict_keys(['cn1.weight', 'cn1.bias', 'cn2.weight', 'cn2.bias', 'fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias'])




```python
model_children_list = list(model.children())
convolutional_layers = []
model_parameters = []
model_children_list
# len(model_children_list)
print("model_children_list[0] = ", model_children_list[0])
print("type(model_children_list[0]) = ", type(model_children_list[0]))
type(model_children_list[0]) == nn.Conv2d # true

print("model_children_list[2] = ", model_children_list[2])
print("type(model_children_list[2]) = ", type(model_children_list[2]))
type(model_children_list[2]) == nn.Dropout2d # true
```

    model_children_list[0] =  Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1))
    type(model_children_list[0]) =  <class 'torch.nn.modules.conv.Conv2d'>
    model_children_list[2] =  Dropout2d(p=0.1, inplace=False)
    type(model_children_list[2]) =  <class 'torch.nn.modules.dropout.Dropout2d'>





    True




```python
for i in range(len(model_children_list)):
    if type(model_children_list[i]) == nn.Conv2d:
        model_parameters.append(model_children_list[i].weight)
        convolutional_layers.append(model_children_list[i])

# len(model_parameters) # 2
# len(model_parameters[0]) # 16
# len(model_parameters[1]) # 32
```


```python
len(model_parameters) # 2
print(model_parameters[0].shape) # len(model_parameters) # 2
print(model_parameters[1].shape) # torch.Size([32, 16, 3, 3])
```

    torch.Size([16, 1, 3, 3])
    torch.Size([32, 16, 3, 3])



```python
plt.figure(figsize=(5, 4))
for i, flt in enumerate(model_parameters[0]): #
    plt.subplot(4, 4, i+1)
    plt.imshow(flt[0, :, :].cpu().detach().numpy(), cmap='gray')
    plt.axis('off')
plt.show()

```


    
*(이미지 없음 / image not preserved: output_29_0.png)*
    



```python
plt.figure(figsize=(5, 8))
for i, flt in enumerate(model_parameters[1]):
    plt.subplot(8, 4, i+1)
    plt.imshow(flt[0, :, :].cpu().detach(), cmap='gray')
    plt.axis('off')
plt.show()
```


    
*(이미지 없음 / image not preserved: output_30_0.png)*
    


### Visualize feature maps


```python
sample_data.shape
```




    torch.Size([500, 1, 28, 28])




```python
display(convolutional_layers)
per_layer_results = convolutional_layers[0](sample_data)
per_layer_results.shape # torch.Size([500, 16, 26, 26])

plt.imshow(per_layer_results[-1].cpu().data.numpy()[0], cmap = "gray_r")
plt.show()
```


    [Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1)),
     Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1))]



    
*(이미지 없음 / image not preserved: output_33_1.png)*
    



```python
per_layer_results[-1].shape
```




    torch.Size([16, 26, 26])




```python
per_layer_results = [convolutional_layers[0](sample_data)] # [torch.Size([500, 16, 26, 26])]
for i in range(1, len(convolutional_layers)): 
    per_layer_results.append(convolutional_layers[i](per_layer_results[-1])) # last feature

print(len(per_layer_results))
print("per_layer_results[0] = ")
print(per_layer_results[0].shape)
print(per_layer_results[1].shape)
```

    2
    per_layer_results[0] = 
    torch.Size([500, 16, 26, 26])
    torch.Size([500, 32, 24, 24])



```python
per_layer_results[0].shape[0]
```




    500




```python
idx = np.random.randint(0, per_layer_results[0].shape[0]) # (0, 500)

plt.figure(figsize=(5, 4))
layer_visualisation = per_layer_results[0][idx, ...] # torch.Size([16, 26, 26])
layer_visualisation = layer_visualisation.data
print(layer_visualisation.size())
print(type(layer_visualisation))
for i, flt in enumerate(layer_visualisation):
    plt.subplot(4, 4, i + 1)
    plt.imshow(flt.cpu().detach(), cmap='gray')
    plt.axis("off")
plt.show()
```

    torch.Size([16, 26, 26])
    <class 'torch.Tensor'>



    
*(이미지 없음 / image not preserved: output_37_1.png)*
    



```python
idx = np.random.randint(0, per_layer_results[1].shape[0]) # (0, 500)

plt.figure(figsize=(5, 8))
layer_visualisation = per_layer_results[1][idx, :, :, :]
layer_visualisation = layer_visualisation.data
print(layer_visualisation.size())
for i, flt in enumerate(layer_visualisation):
    plt.subplot(8, 4, i + 1)
    plt.imshow(flt.cpu().detach(), cmap='gray')
    plt.axis("off")
plt.show()
```

    torch.Size([32, 24, 24])



    
*(이미지 없음 / image not preserved: output_38_1.png)*
    


# Captum

## import modules


```python
from captum.attr import IntegratedGradients
from captum.attr import Saliency
from captum.attr import DeepLift
from captum.attr import visualization as viz
```


```python
test_samples = enumerate(test_dataloader)
b_i, (sample_data, sample_targets) = next(test_samples)

plt.imshow(sample_data[0][0], cmap='gray', interpolation='none') # bilinear
plt.show()
```


    
*(이미지 없음 / image not preserved: output_42_0.png)*
    



```python
sample_data = sample_data.to(device)
print(f"Model prediction is : {model(sample_data).data.max(1)[1][0]}")
print(f"Ground truth is : {sample_targets[0]}")
```

    Model prediction is : 6
    Ground truth is : 6


## captum tools


```python
print(sample_data.shape)
sample_data[0].unsqueeze(0).shape
```

    torch.Size([500, 1, 28, 28])





    torch.Size([1, 1, 28, 28])




```python
captum_input = sample_data[0].unsqueeze(0) # torch.Size([1, 1, 28, 28])
captum_input.requires_grad = True
```


```python
np.transpose((sample_data[0].cpu().detach().numpy() / 2) + 0.5, (1, 2, 0)).shape
```




    (28, 28, 1)




```python
orig_image = np.tile(np.transpose((sample_data[0].cpu().detach().numpy() / 2) + 0.5, (1, 2, 0)), (1,1,3))
print(orig_image.shape)
# tmp = np.transpose((sample_data[0].cpu().detach().numpy() / 2) + 0.5, (1, 2, 0))
# orig_image = np.concatenate([tmp, np.zeros(tmp.shape), np.zeros(tmp.shape)], axis=2)
_ = viz.visualize_image_attr(None, orig_image, cmap='gray', method="original_image", title="Original Image")  # a function that visualizes attribution maps over an image.
```

    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [0.28787878..1.9106851].


    (28, 28, 3)



    
*(이미지 없음 / image not preserved: output_48_2.png)*
    



```python
sample_targets[0]
```




    tensor(6)




```python
saliency = Saliency(model) # <class 'captum.attr._core.saliency.Saliency'>, gradient-based saliency map
gradients = saliency.attribute(captum_input, target=sample_targets[0].item()) 
# print(gradients.shape) # torch.Size([1, 1, 28, 28])
gradients = np.reshape(gradients.squeeze().cpu().detach().numpy(), (28, 28, 1))
_ = viz.visualize_image_attr(gradients, orig_image, method="blended_heat_map", 
                             sign="absolute_value",
                             show_colorbar=True, 
                             title="Overlayed Gradients")
```

    c:\Users\user\anaconda3\envs\torchgpu_py3.12\Lib\site-packages\torch\nn\functional.py:1538: UserWarning: dropout2d: Received a 2-D input to dropout2d, which is deprecated and will result in an error in a future release. To retain the behavior and silence this warning, please use dropout instead. Note that dropout2d exists to provide channel-wise dropout on inputs with 2 spatial dimensions, a channel dimension, and an optional batch dimension (i.e. 3D or 4D inputs).
      warnings.warn(warn_msg)
    c:\Users\user\anaconda3\envs\torchgpu_py3.12\Lib\site-packages\captum\attr\_utils\visualization.py:51: UserWarning: Attempting to normalize by value approximately 0, visualized resultsmay be misleading. This likely means that attribution values are allclose to 0.
      warnings.warn(



    
*(이미지 없음 / image not preserved: output_50_1.png)*
    



```python
plt.imshow(np.tile(gradients/(np.max(gradients)), (1,1,3)))
```




    <matplotlib.image.AxesImage at 0x2637e2bb4a0>




    
*(이미지 없음 / image not preserved: output_51_1.png)*
    



```python
integ_grads = IntegratedGradients(model) # IG accumulates gradients over multiple interpolated inputs between x and x' making it more reliable than raw gradients.
attributed_ig, delta = integ_grads.attribute(captum_input, 
                                             target=sample_targets[0].item(), 
                                             baselines=captum_input * 0,
                                             return_convergence_delta=True)

attributed_ig = np.reshape(attributed_ig.squeeze().cpu().detach().numpy(), (28, 28, 1))
_ = viz.visualize_image_attr(attributed_ig, orig_image, 
                             method="blended_heat_map",
                             sign="all", 
                             show_colorbar=True, 
                             title="Overlayed Integrated Gradients")
```

    c:\Users\user\anaconda3\envs\opencv_torch_py3.12\Lib\site-packages\torch\nn\functional.py:1538: UserWarning: dropout2d: Received a 2-D input to dropout2d, which is deprecated and will result in an error in a future release. To retain the behavior and silence this warning, please use dropout instead. Note that dropout2d exists to provide channel-wise dropout on inputs with 2 spatial dimensions, a channel dimension, and an optional batch dimension (i.e. 3D or 4D inputs).
      warnings.warn(warn_msg)



    
*(이미지 없음 / image not preserved: output_52_1.png)*
    



```python
deep_lift = DeepLift(model)
attributed_dl = deep_lift.attribute(captum_input, 
                                    target=sample_targets[0].item(), 
                                    baselines=captum_input * 0,
                                    return_convergence_delta=False)
attributed_dl = np.reshape(attributed_dl.squeeze(0).cpu().detach().numpy(), (28, 28, 1))
_ = viz.visualize_image_attr(attributed_dl, 
                             orig_image, 
                             method="blended_heat_map",
                             sign="all",
                             show_colorbar=True,
                             title="Overlayed DeepLift")
```

    c:\Users\user\anaconda3\envs\torchgpu_py3.12\Lib\site-packages\captum\attr\_core\deep_lift.py:304: UserWarning: Setting forward, backward hooks and attributes on non-linear
                   activations. The hooks and attributes will be removed
                after the attribution is finished
      warnings.warn(
    c:\Users\user\anaconda3\envs\torchgpu_py3.12\Lib\site-packages\torch\nn\functional.py:1538: UserWarning: dropout2d: Received a 2-D input to dropout2d, which is deprecated and will result in an error in a future release. To retain the behavior and silence this warning, please use dropout instead. Note that dropout2d exists to provide channel-wise dropout on inputs with 2 spatial dimensions, a channel dimension, and an optional batch dimension (i.e. 3D or 4D inputs).
      warnings.warn(warn_msg)
    c:\Users\user\anaconda3\envs\torchgpu_py3.12\Lib\site-packages\captum\attr\_utils\visualization.py:51: UserWarning: Attempting to normalize by value approximately 0, visualized resultsmay be misleading. This likely means that attribution values are allclose to 0.
      warnings.warn(



    
*(이미지 없음 / image not preserved: output_53_1.png)*
    

