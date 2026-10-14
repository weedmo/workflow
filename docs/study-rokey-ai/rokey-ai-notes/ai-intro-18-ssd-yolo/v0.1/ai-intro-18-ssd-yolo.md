# 18 객체 검출 (One-stage object detection)

> **Migrated from Quartz wiki** (`wiki/study/rokey-ai/basic/ai-intro-18-ssd_yolo.md`, updated 2026-09-26)  
> tags: rokey, ai, notebook, basic, study  
> source: raw/2026-09-26-rokey-ai/basic/강의_3기_AI개론_18차시__SSD_Yolo_.ipynb  
> source: https://github.com/weedmo/Rokey-AI  
- SSD, YOLO

* "부록3 매트플롯립 입문"에서 한글 폰트를 올바르게 출력하기 위한 설치 방법을 설명했다. 설치 방법은 다음과 같다.


```python
!sudo apt-get install -y fonts-nanum* | tail -n 1
!sudo fc-cache -fv
!rm -rf ~/.cache/matplotlib
```


```python
# 필요 라이브러리 설치

!pip install torchviz | tail -n 1
!pip install torchinfo | tail -n 1
# w = !apt install tree
# print(w[-2])
```

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
# warning 표시 끄기
import warnings
warnings.simplefilter('ignore')

import os
import numpy as np
import matplotlib.pyplot as plt
# 폰트 관련 용도
import matplotlib.font_manager as fm
import cv2

import torch
from torch import nn, optim
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from torchvision.io import read_image
from torchvision import models, datasets, transforms
from torchinfo import summary
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
# GPU 디바이스 할당

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
```

    cuda:0


## SSD 모델 사용하기

### 모델 불러 오기


```python
## Pretrained object detection model list
dir(models.detection)
```




    ['FCOS',
     'FCOS_ResNet50_FPN_Weights',
     'FasterRCNN',
     'FasterRCNN_MobileNet_V3_Large_320_FPN_Weights',
     'FasterRCNN_MobileNet_V3_Large_FPN_Weights',
     'FasterRCNN_ResNet50_FPN_V2_Weights',
     'FasterRCNN_ResNet50_FPN_Weights',
     'KeypointRCNN',
     'KeypointRCNN_ResNet50_FPN_Weights',
     'MaskRCNN',
     'MaskRCNN_ResNet50_FPN_V2_Weights',
     'MaskRCNN_ResNet50_FPN_Weights',
     'RetinaNet',
     'RetinaNet_ResNet50_FPN_V2_Weights',
     'RetinaNet_ResNet50_FPN_Weights',
     'SSD300_VGG16_Weights',
     'SSDLite320_MobileNet_V3_Large_Weights',
     '__builtins__',
     '__cached__',
     '__doc__',
     '__file__',
     '__loader__',
     '__name__',
     '__package__',
     '__path__',
     '__spec__',
     '_utils',
     'anchor_utils',
     'backbone_utils',
     'faster_rcnn',
     'fasterrcnn_mobilenet_v3_large_320_fpn',
     'fasterrcnn_mobilenet_v3_large_fpn',
     'fasterrcnn_resnet50_fpn',
     'fasterrcnn_resnet50_fpn_v2',
     'fcos',
     'fcos_resnet50_fpn',
     'generalized_rcnn',
     'image_list',
     'keypoint_rcnn',
     'keypointrcnn_resnet50_fpn',
     'mask_rcnn',
     'maskrcnn_resnet50_fpn',
     'maskrcnn_resnet50_fpn_v2',
     'retinanet',
     'retinanet_resnet50_fpn',
     'retinanet_resnet50_fpn_v2',
     'roi_heads',
     'rpn',
     'ssd',
     'ssd300_vgg16',
     'ssdlite',
     'ssdlite320_mobilenet_v3_large',
     'transform']




```python
## load model
# from torchvision.models.detection import ssd300_vgg16 # input image 300x300, backbone vgg16

weights = models.detection.SSD300_VGG16_Weights.COCO_V1 # 2014
ssd300 = models.detection.ssd300_vgg16(weights = weights)
# weights.transforms()

```

### 모델 확인 하기


```python
print(ssd300)
```

    SSD(
      (backbone): SSDFeatureExtractorVGG(
        (features): Sequential(
          (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (1): ReLU(inplace=True)
          (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (3): ReLU(inplace=True)
          (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
          (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (6): ReLU(inplace=True)
          (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (8): ReLU(inplace=True)
          (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
          (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (11): ReLU(inplace=True)
          (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (13): ReLU(inplace=True)
          (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (15): ReLU(inplace=True)
          (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)
          (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (18): ReLU(inplace=True)
          (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (20): ReLU(inplace=True)
          (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (22): ReLU(inplace=True)
        )
        (extra): ModuleList(
          (0): Sequential(
            (0): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            (1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU(inplace=True)
            (3): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (4): ReLU(inplace=True)
            (5): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (6): ReLU(inplace=True)
            (7): Sequential(
              (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
              (1): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(6, 6), dilation=(6, 6))
              (2): ReLU(inplace=True)
              (3): Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1))
              (4): ReLU(inplace=True)
            )
          )
          (1): Sequential(
            (0): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1))
            (1): ReLU(inplace=True)
            (2): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            (3): ReLU(inplace=True)
          )
          (2): Sequential(
            (0): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1))
            (1): ReLU(inplace=True)
            (2): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            (3): ReLU(inplace=True)
          )
          (3-4): 2 x Sequential(
            (0): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
            (1): ReLU(inplace=True)
            (2): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1))
            (3): ReLU(inplace=True)
          )
        )
      )
      (anchor_generator): DefaultBoxGenerator(aspect_ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]], clip=True, scales=[0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05], steps=[8, 16, 32, 64, 100, 300])
      (head): SSDHead(
        (classification_head): SSDClassificationHead(
          (module_list): ModuleList(
            (0): Conv2d(512, 364, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (1): Conv2d(1024, 546, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): Conv2d(512, 546, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (3): Conv2d(256, 546, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (4-5): 2 x Conv2d(256, 364, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
        )
        (regression_head): SSDRegressionHead(
          (module_list): ModuleList(
            (0): Conv2d(512, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (1): Conv2d(1024, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): Conv2d(512, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (3): Conv2d(256, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (4-5): 2 x Conv2d(256, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
        )
      )
      (transform): GeneralizedRCNNTransform(
          Normalize(mean=[0.48235, 0.45882, 0.40784], std=[0.00392156862745098, 0.00392156862745098, 0.00392156862745098])
          Resize(min_size=(300,), max_size=300, mode='bilinear')
      )
    )



```python
summary(ssd300, (1, 3, 300, 300))
```




    ==========================================================================================
    Layer (type:depth-idx)                   Output Shape              Param #
    ==========================================================================================
    SSD                                      [97, 4]                   --
    ├─GeneralizedRCNNTransform: 1-1          [1, 3, 300, 300]          --
    ├─SSDFeatureExtractorVGG: 1-2            [1, 256, 1, 1]            512
    │    └─Sequential: 2-1                   [1, 512, 38, 38]          --
    │    │    └─Conv2d: 3-1                  [1, 64, 300, 300]         (1,792)
    │    │    └─ReLU: 3-2                    [1, 64, 300, 300]         --
    │    │    └─Conv2d: 3-3                  [1, 64, 300, 300]         (36,928)
    │    │    └─ReLU: 3-4                    [1, 64, 300, 300]         --
    │    │    └─MaxPool2d: 3-5               [1, 64, 150, 150]         --
    │    │    └─Conv2d: 3-6                  [1, 128, 150, 150]        73,856
    │    │    └─ReLU: 3-7                    [1, 128, 150, 150]        --
    │    │    └─Conv2d: 3-8                  [1, 128, 150, 150]        147,584
    │    │    └─ReLU: 3-9                    [1, 128, 150, 150]        --
    │    │    └─MaxPool2d: 3-10              [1, 128, 75, 75]          --
    │    │    └─Conv2d: 3-11                 [1, 256, 75, 75]          295,168
    │    │    └─ReLU: 3-12                   [1, 256, 75, 75]          --
    │    │    └─Conv2d: 3-13                 [1, 256, 75, 75]          590,080
    │    │    └─ReLU: 3-14                   [1, 256, 75, 75]          --
    │    │    └─Conv2d: 3-15                 [1, 256, 75, 75]          590,080
    │    │    └─ReLU: 3-16                   [1, 256, 75, 75]          --
    │    │    └─MaxPool2d: 3-17              [1, 256, 38, 38]          --
    │    │    └─Conv2d: 3-18                 [1, 512, 38, 38]          1,180,160
    │    │    └─ReLU: 3-19                   [1, 512, 38, 38]          --
    │    │    └─Conv2d: 3-20                 [1, 512, 38, 38]          2,359,808
    │    │    └─ReLU: 3-21                   [1, 512, 38, 38]          --
    │    │    └─Conv2d: 3-22                 [1, 512, 38, 38]          2,359,808
    │    │    └─ReLU: 3-23                   [1, 512, 38, 38]          --
    │    └─ModuleList: 2-2                   --                        --
    │    │    └─Sequential: 3-24             [1, 1024, 19, 19]         12,848,640
    │    │    └─Sequential: 3-25             [1, 512, 10, 10]          1,442,560
    │    │    └─Sequential: 3-26             [1, 256, 5, 5]            360,832
    │    │    └─Sequential: 3-27             [1, 256, 3, 3]            328,064
    │    │    └─Sequential: 3-28             [1, 256, 1, 1]            328,064
    ├─SSDHead: 1-3                           [1, 8732, 91]             --
    │    └─SSDRegressionHead: 2-3            [1, 8732, 4]              --
    │    │    └─ModuleList: 3-29             --                        534,648
    │    └─SSDClassificationHead: 2-4        [1, 8732, 91]             --
    │    │    └─ModuleList: 3-30             --                        12,163,242
    ├─DefaultBoxGenerator: 1-4               [8732, 4]                 --
    ==========================================================================================
    Total params: 35,641,826
    Trainable params: 35,603,106
    Non-trainable params: 38,720
    Total mult-adds (G): 34.88
    ==========================================================================================
    Input size (MB): 1.08
    Forward/backward pass size (MB): 208.89
    Params size (MB): 142.57
    Estimated Total Size (MB): 352.53
    ==========================================================================================



### 데이터 준비


```python
## Image preprocessing
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor()
])

# transform = transforms.Compose([
#     transforms.Resize((300, 300)),
#     transforms.ToTensor(),
#     transforms.Normalize(
#         mean=[0.48235, 0.45882, 0.40784],  # Mean values for COCO (normalized to [0, 1])
#         std=[0.00392156862745098, 0.00392156862745098, 0.00392156862745098]  # Scale pixel values to [0, 1]
#     )
# ])
```


```python
# VOCDetection 데이터셋 불러오기
dataset = datasets.VOCDetection(root = "./VOC_dataset/VOC2012", 
                                year = "2012",
                                image_set = "val", # Use "train", "val", or "trainval"
                                download=True,
                                transform = transform # Apply preprocessing pipeline
                                )  

# data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
```

    Using downloaded and verified file: ./VOC_dataset/VOC2012\VOCtrainval_11-May-2012.tar
    Extracting ./VOC_dataset/VOC2012\VOCtrainval_11-May-2012.tar to ./VOC_dataset/VOC2012


### 입력 영상 확인하기


```python
idx = torch.randint(len(dataset), size = (1, ))
image, annotation = dataset[idx.item()]


plt.imshow(image.permute(1, 2, 0))
plt.grid(None)
plt.axis("off")
plt.show()
```


    
*(이미지 없음 / image not preserved: output_20_0.png)*
    


### 모델 예측


```python
import time

idx = torch.randint(len(dataset), size = (1, ))
image_jpg, annotation = dataset[idx.item()]
image = image_jpg.unsqueeze(0).to(device)

ssd300.eval()
start = time.time()
with torch.no_grad():
    prediction = ssd300(image)
stop = time.time()
print(f"estimation time = {(stop - start)*1000:.3f}ms")
prediction = prediction[0] # batch remove
prediction
```

    estimation time = 185.503ms





    {'boxes': tensor([[1.1262e+02, 6.5100e+01, 2.1513e+02, 1.8358e+02],
             [8.8847e+01, 6.2713e+01, 2.4138e+02, 2.5318e+02],
             [8.2444e+00, 1.4865e+02, 2.9779e+02, 2.9904e+02],
             [2.4332e+02, 3.4011e+00, 2.9987e+02, 1.3884e+02],
             [1.8608e+01, 2.8053e+01, 6.6836e+01, 2.1345e+02],
             [6.5991e+00, 1.7761e+02, 1.5280e+02, 2.9850e+02],
             [1.1509e+02, 6.3359e+01, 1.7068e+02, 1.6076e+02],
             [1.5691e+02, 8.5306e+01, 2.3508e+02, 1.8724e+02],
             [7.1447e+01, 7.7665e+01, 1.6359e+02, 2.0069e+02],
             [2.5747e+02, 3.0218e+01, 2.9824e+02, 8.5727e+01],
             [2.4039e+01, 2.4206e+02, 5.2521e+01, 2.7752e+02],
             [2.7608e+01, 2.5449e+02, 4.7232e+01, 2.7788e+02],
             [1.2059e+02, 1.0803e+02, 1.5563e+02, 1.8187e+02],
             [1.4902e+02, 1.0802e+02, 1.9267e+02, 1.8077e+02],
             [1.3566e+01, 5.9837e+01, 2.9061e+02, 2.9515e+02],
             [1.3296e+02, 1.4723e+02, 2.0539e+02, 1.8141e+02],
             [1.2713e+02, 1.3802e+02, 2.1133e+02, 2.0558e+02],
             [3.2951e+01, 2.3434e+02, 2.3441e+02, 2.9992e+02],
             [1.3268e+02, 1.2623e+02, 1.7042e+02, 1.8729e+02],
             [1.1654e+02, 6.5065e+01, 1.6978e+02, 1.0808e+02],
             [2.6553e+01, 4.5842e+01, 5.2569e+01, 1.5028e+02],
             [1.8091e+02, 0.0000e+00, 3.0000e+02, 1.2458e+02],
             [2.0823e+00, 2.0143e+01, 4.6297e+01, 2.0735e+02],
             [2.4797e+02, 3.4914e+01, 2.8237e+02, 9.8553e+01],
             [2.4879e+01, 1.4781e+02, 5.2215e+01, 1.9713e+02],
             [1.5745e+02, 9.3629e+01, 2.0591e+02, 1.6235e+02],
             [2.7077e+01, 9.3283e+01, 5.0884e+01, 1.9975e+02],
             [1.1815e+02, 1.3531e+02, 1.9542e+02, 1.7165e+02],
             [2.3157e+02, 1.2944e+02, 2.6351e+02, 1.9811e+02],
             [2.0102e+01, 2.4127e+02, 6.5305e+01, 2.9298e+02],
             [1.7710e+02, 1.0589e+02, 2.1826e+02, 1.6936e+02],
             [2.2639e+02, 1.2053e+02, 2.9165e+02, 2.0699e+02],
             [1.6301e+02, 1.2347e+02, 2.0437e+02, 1.8570e+02],
             [1.3836e+02, 8.3384e+01, 1.9517e+02, 1.2312e+02],
             [1.1054e+02, 8.4607e+01, 1.7236e+02, 1.1847e+02],
             [2.0724e+02, 6.5498e+01, 2.8422e+02, 2.0185e+02],
             [2.6934e+00, 2.0977e+02, 3.9960e+01, 2.8771e+02],
             [2.6683e+02, 1.4120e+02, 2.9494e+02, 2.2982e+02],
             [1.2180e+02, 9.9138e+01, 1.7816e+02, 1.4173e+02],
             [2.0152e+02, 1.0903e+02, 2.3390e+02, 1.8048e+02],
             [0.0000e+00, 2.2842e+02, 1.1478e+02, 3.0000e+02],
             [2.6364e+02, 0.0000e+00, 2.9973e+02, 7.6078e+01],
             [2.4596e+02, 1.2570e+02, 2.7778e+02, 2.0292e+02],
             [1.9314e+01, 2.5186e+02, 3.6432e+01, 2.7902e+02],
             [1.4996e+02, 1.4922e+02, 1.8758e+02, 2.0706e+02],
             [1.2033e+01, 6.2718e+01, 2.9159e+02, 2.9673e+02],
             [1.1332e+02, 1.1368e+02, 1.6910e+02, 1.5642e+02],
             [1.3049e+02, 1.6908e+02, 2.0806e+02, 1.9631e+02],
             [9.2237e+01, 4.6891e+01, 2.1070e+02, 1.2619e+02],
             [2.0608e+02, 1.1134e+02, 2.3711e+02, 1.4633e+02],
             [1.6260e+02, 1.1726e+02, 2.2606e+02, 1.5392e+02],
             [2.1294e+02, 1.0853e+02, 2.5202e+02, 1.7428e+02],
             [2.3360e+02, 7.7548e+01, 2.9801e+02, 2.5035e+02],
             [7.2506e+00, 4.1083e+01, 3.4351e+01, 1.3887e+02],
             [9.9559e+01, 1.0104e+02, 1.5327e+02, 1.3836e+02],
             [1.0502e+02, 7.4723e+01, 1.4589e+02, 1.2955e+02],
             [2.0783e+02, 3.6051e+01, 2.9338e+02, 1.9183e+02],
             [1.6495e+02, 1.0280e+02, 2.2831e+02, 1.3913e+02],
             [1.0848e+02, 9.8211e+01, 1.3508e+02, 1.7177e+02],
             [5.5352e+01, 7.2755e+01, 1.9565e+02, 1.4870e+02],
             [2.4863e+01, 3.7728e+01, 5.6120e+01, 1.0224e+02],
             [1.4976e+02, 7.4522e+01, 1.8808e+02, 1.4709e+02],
             [1.6498e+02, 1.1065e+02, 2.5745e+02, 2.1685e+02],
             [1.5889e+02, 1.3491e+02, 2.2578e+02, 1.6964e+02],
             [1.9150e+02, 1.0100e+02, 2.4268e+02, 1.4255e+02],
             [3.4547e+01, 2.6295e+02, 5.0502e+01, 2.8572e+02],
             [2.6314e+01, 1.4489e+02, 5.1329e+01, 2.1413e+02],
             [0.0000e+00, 1.3025e+02, 2.6878e+01, 1.7329e+02],
             [6.5246e+01, 1.0600e+00, 1.4453e+02, 5.3497e+01],
             [1.7806e+02, 7.2294e+01, 3.0000e+02, 1.5203e+02],
             [2.2755e+02, 7.5259e-01, 2.9246e+02, 8.6693e+01],
             [4.9127e+01, 0.0000e+00, 1.1652e+02, 8.1133e+01],
             [1.6532e+02, 1.4815e+02, 2.0229e+02, 2.0515e+02],
             [1.7039e+02, 8.4908e+01, 2.2574e+02, 1.2609e+02],
             [4.1368e+01, 4.8671e+01, 7.0007e+01, 1.4585e+02],
             [2.2277e+02, 1.0083e+02, 2.7534e+02, 1.4145e+02],
             [2.6426e+02, 1.1098e+02, 2.9474e+02, 1.9126e+02],
             [3.4430e+00, 2.6317e+02, 3.3318e+01, 3.0000e+02],
             [2.3276e+02, 9.4717e+01, 2.6556e+02, 1.6255e+02],
             [4.3765e+01, 1.6596e+02, 1.1121e+02, 2.9378e+02],
             [1.3776e+02, 1.3873e+02, 1.6569e+02, 2.2108e+02],
             [5.3676e+01, 1.0546e+02, 2.0609e+02, 1.8247e+02],
             [2.2202e+01, 2.5698e+02, 5.1517e+01, 2.9284e+02],
             [9.3237e+01, 2.8046e+02, 1.0717e+02, 3.0000e+02],
             [1.2574e+02, 1.6765e+02, 2.0966e+02, 2.3383e+02],
             [0.0000e+00, 7.6399e+01, 7.8534e+01, 3.0000e+02],
             [3.8097e+01, 4.0502e+01, 7.4172e+01, 9.4964e+01],
             [2.0467e+02, 1.3025e+02, 3.0000e+02, 2.8190e+02],
             [1.7996e+02, 1.4716e+02, 2.1229e+02, 1.8205e+02],
             [2.4392e+01, 2.7371e+02, 5.7802e+01, 3.0000e+02],
             [1.9919e+02, 8.7053e+01, 2.9711e+02, 2.4591e+02],
             [1.2975e+02, 4.3949e+01, 2.7434e+02, 1.2061e+02],
             [6.1990e+00, 7.5520e+01, 3.5306e+01, 1.7568e+02],
             [1.8482e+02, 0.0000e+00, 3.0000e+02, 4.4032e+01],
             [5.2050e+00, 9.1122e+01, 3.6830e+01, 1.7597e+02],
             [1.0655e+02, 1.3738e+02, 1.7566e+02, 2.1477e+02],
             [1.1943e+02, 1.4556e+02, 1.5214e+02, 2.0964e+02],
             [2.2517e+02, 8.4079e+01, 2.7898e+02, 1.2491e+02],
             [2.6137e+02, 5.0318e+01, 2.9895e+02, 1.1613e+02],
             [2.7024e+02, 4.6286e+01, 2.9800e+02, 8.2859e+01],
             [8.6794e+01, 2.8234e+02, 9.8415e+01, 3.0000e+02],
             [7.7143e+01, 2.8459e+02, 8.8786e+01, 3.0000e+02],
             [1.3975e+02, 4.1382e+01, 2.9931e+02, 2.7759e+02],
             [3.4274e+00, 1.8143e+02, 3.7327e+01, 2.5543e+02],
             [2.1211e+02, 1.0421e+02, 2.4400e+02, 1.3496e+02],
             [1.6164e+02, 3.0446e+01, 2.9376e+02, 2.8220e+02],
             [9.7141e+01, 0.0000e+00, 3.0000e+02, 7.2839e+01],
             [9.0797e+00, 2.0628e+02, 1.4903e+02, 3.0000e+02],
             [2.8739e+01, 2.6654e+02, 4.7588e+01, 2.8149e+02],
             [7.6200e+01, 1.9732e+02, 1.4245e+02, 3.0000e+02],
             [1.8015e+02, 1.5688e+02, 1.9724e+02, 1.7510e+02],
             [1.0035e+02, 1.3206e+02, 1.5161e+02, 1.7284e+02],
             [1.2896e+02, 1.8451e+02, 2.0478e+02, 2.1493e+02],
             [1.5009e+02, 1.6989e+02, 1.8568e+02, 2.3226e+02],
             [2.0381e+02, 1.1824e+02, 2.6472e+02, 1.5338e+02],
             [4.5271e+01, 4.8017e+01, 5.9255e+01, 7.5684e+01],
             [2.4736e+02, 1.5845e+02, 2.7825e+02, 2.3631e+02],
             [0.0000e+00, 2.6236e-01, 3.9252e+01, 8.3543e+01],
             [1.8883e+02, 1.4010e+02, 2.0275e+02, 1.6926e+02],
             [2.3475e+02, 6.9016e+01, 2.6614e+02, 1.3673e+02],
             [0.0000e+00, 1.2234e+02, 2.6402e+02, 2.9557e+02],
             [9.0739e+01, 3.4670e+01, 1.5800e+02, 1.0362e+02],
             [7.0053e-01, 1.9191e+02, 2.5623e+01, 2.7361e+02],
             [0.0000e+00, 2.6236e-01, 3.9252e+01, 8.3543e+01],
             [1.8613e+02, 1.5526e+02, 2.0546e+02, 1.7521e+02],
             [1.4005e+01, 0.0000e+00, 8.4360e+01, 6.2809e+01],
             [2.0693e+02, 1.1127e+02, 2.8560e+02, 2.7764e+02],
             [2.1205e+01, 2.5996e+02, 3.5240e+01, 2.8987e+02],
             [3.4355e+01, 4.6169e+01, 5.0254e+01, 7.4532e+01],
             [2.3790e+02, 2.7117e+01, 2.6618e+02, 9.5413e+01],
             [2.1104e+02, 1.1435e+02, 2.3033e+02, 1.3614e+02],
             [2.3719e+01, 2.6946e+02, 3.4947e+01, 2.9880e+02],
             [5.4845e+01, 3.0641e+00, 8.4185e+01, 4.3762e+01],
             [1.2977e+02, 2.6673e+02, 1.6623e+02, 2.9949e+02],
             [7.7951e+01, 0.0000e+00, 1.4502e+02, 7.8170e+01],
             [4.4531e+01, 2.5518e+02, 5.9404e+01, 2.7930e+02],
             [2.7620e+01, 2.7103e+02, 4.5167e+01, 2.9764e+02],
             [2.1352e+02, 8.0281e+01, 2.5375e+02, 1.3187e+02],
             [2.4486e+01, 8.5836e+01, 5.1876e+01, 1.8861e+02],
             [0.0000e+00, 5.5516e+01, 1.0116e+01, 9.3727e+01],
             [1.3284e+00, 1.6610e+02, 5.0052e+01, 3.0000e+02],
             [1.3115e+01, 2.6681e+02, 2.7784e+01, 2.9891e+02],
             [2.7369e+00, 2.0024e+02, 3.1214e+01, 2.4118e+02],
             [1.6958e+02, 1.3649e+02, 2.0467e+02, 1.7422e+02],
             [3.4733e+01, 2.7246e+02, 5.2063e+01, 2.9659e+02],
             [1.1108e+02, 1.6895e+02, 1.7180e+02, 1.9890e+02],
             [2.2348e+01, 2.0286e+02, 5.5345e+01, 2.6822e+02],
             [2.1609e+02, 1.2373e+02, 2.4961e+02, 1.9544e+02],
             [1.8280e+02, 1.4747e+02, 1.9556e+02, 1.6378e+02],
             [4.8337e-01, 2.6548e+02, 1.8968e+01, 3.0000e+02],
             [8.0369e+01, 1.7715e+02, 2.9349e+02, 2.9746e+02],
             [3.6577e+01, 1.0696e+02, 5.1737e+01, 1.4984e+02],
             [0.0000e+00, 1.5419e+02, 2.8511e+01, 1.8973e+02],
             [1.1500e+00, 8.0133e+01, 2.2933e+01, 1.6802e+02],
             [1.2527e+01, 6.0236e+01, 2.8842e+02, 2.9003e+02],
             [9.2530e+00, 1.9297e+01, 3.1482e+01, 7.1072e+01],
             [2.6988e+02, 5.5862e+01, 2.9933e+02, 9.7382e+01],
             [1.6196e+01, 2.6468e+02, 4.6781e+01, 2.9977e+02],
             [1.0586e+00, 2.3826e+02, 6.0510e+01, 3.0000e+02],
             [2.2275e+02, 1.1363e+02, 2.3515e+02, 1.4041e+02],
             [1.4553e+01, 0.0000e+00, 1.4937e+02, 9.3181e+01],
             [1.8520e+01, 2.3740e+02, 3.6157e+01, 2.6781e+02],
             [1.9543e+00, 7.3124e+01, 4.0074e+01, 1.3744e+02],
             [2.5311e+01, 2.7842e+02, 3.4296e+01, 3.0000e+02],
             [6.5246e+01, 1.0600e+00, 1.4453e+02, 5.3497e+01],
             [6.2167e+00, 7.3394e+01, 6.4603e+01, 1.7249e+02],
             [4.3474e+01, 1.1921e+02, 6.7450e+01, 2.1057e+02],
             [0.0000e+00, 4.3247e+01, 1.6312e+02, 2.9152e+02],
             [2.9164e+01, 6.7715e+01, 5.5653e+01, 1.1601e+02],
             [6.0237e+00, 6.0450e+01, 1.9032e+01, 9.1025e+01],
             [2.8704e+02, 5.2506e+01, 2.9997e+02, 8.1833e+01],
             [6.1451e+00, 3.0641e+01, 1.7065e+01, 7.5105e+01],
             [1.0072e+02, 2.7757e+02, 1.1564e+02, 3.0000e+02],
             [2.7481e+02, 4.9030e+01, 2.9209e+02, 7.9942e+01],
             [2.8040e+02, 0.0000e+00, 3.0000e+02, 1.3579e+02],
             [2.6483e+02, 4.1991e+01, 2.9167e+02, 8.4969e+01],
             [1.4296e+02, 7.8226e+00, 2.6265e+02, 9.8022e+01],
             [2.5156e+02, 3.6799e+01, 2.8027e+02, 9.1022e+01],
             [1.1677e+01, 2.0524e+02, 2.7458e+01, 2.3653e+02],
             [2.8534e+02, 1.1165e+02, 2.9798e+02, 1.6020e+02],
             [2.3253e+02, 1.6188e+02, 2.8907e+02, 2.0202e+02],
             [1.4264e+01, 2.3510e+02, 8.2463e+01, 2.9864e+02],
             [2.1017e+00, 0.0000e+00, 3.4675e+01, 8.8614e+01],
             [1.7790e+00, 4.1898e+01, 2.2267e+01, 1.3345e+02],
             [1.1363e+02, 2.6574e+02, 1.4332e+02, 2.9985e+02],
             [2.8704e+02, 5.2506e+01, 2.9997e+02, 8.1833e+01],
             [2.5011e+02, 6.3502e-01, 2.8413e+02, 7.0594e+01],
             [7.0394e+01, 2.7544e+02, 9.9670e+01, 3.0000e+02],
             [1.6091e+02, 1.7158e+02, 2.5516e+02, 2.9511e+02],
             [2.8069e+01, 2.7779e+02, 4.3785e+01, 3.0000e+02],
             [9.8555e+00, 1.1961e+02, 6.6770e+01, 2.0918e+02],
             [2.2301e+02, 1.1039e+02, 2.4986e+02, 1.4078e+02],
             [3.9344e+00, 2.5360e+02, 4.8506e+01, 2.9934e+02],
             [2.7471e+02, 6.5201e+01, 2.9317e+02, 8.5715e+01],
             [2.1740e+01, 1.9924e+02, 3.3347e+01, 2.3876e+02],
             [0.0000e+00, 1.8690e+02, 2.6029e+01, 2.3328e+02],
             [3.4590e+01, 2.2481e+02, 7.2581e+01, 2.8204e+02],
             [0.0000e+00, 2.3549e+02, 2.8934e+01, 2.9230e+02],
             [1.1500e+00, 8.0133e+01, 2.2933e+01, 1.6802e+02],
             [2.9411e+02, 1.1617e+02, 3.0000e+02, 1.3939e+02]], device='cuda:0'),
     'scores': tensor([0.9462, 0.1989, 0.1664, 0.1032, 0.0894, 0.0836, 0.0830, 0.0819, 0.0794,
             0.0723, 0.0723, 0.0720, 0.0700, 0.0675, 0.0648, 0.0641, 0.0613, 0.0606,
             0.0600, 0.0597, 0.0590, 0.0560, 0.0547, 0.0541, 0.0538, 0.0532, 0.0518,
             0.0517, 0.0501, 0.0501, 0.0498, 0.0497, 0.0493, 0.0491, 0.0490, 0.0475,
             0.0475, 0.0471, 0.0465, 0.0463, 0.0462, 0.0459, 0.0454, 0.0445, 0.0434,
             0.0433, 0.0432, 0.0425, 0.0419, 0.0419, 0.0419, 0.0416, 0.0416, 0.0409,
             0.0404, 0.0398, 0.0398, 0.0397, 0.0396, 0.0395, 0.0387, 0.0385, 0.0382,
             0.0369, 0.0369, 0.0366, 0.0363, 0.0354, 0.0352, 0.0350, 0.0349, 0.0345,
             0.0343, 0.0341, 0.0339, 0.0331, 0.0327, 0.0324, 0.0323, 0.0320, 0.0320,
             0.0318, 0.0317, 0.0317, 0.0316, 0.0315, 0.0313, 0.0308, 0.0308, 0.0307,
             0.0304, 0.0302, 0.0302, 0.0301, 0.0300, 0.0297, 0.0297, 0.0297, 0.0297,
             0.0297, 0.0295, 0.0294, 0.0294, 0.0293, 0.0293, 0.0293, 0.0292, 0.0291,
             0.0291, 0.0289, 0.0284, 0.0282, 0.0282, 0.0281, 0.0279, 0.0277, 0.0277,
             0.0277, 0.0276, 0.0273, 0.0272, 0.0272, 0.0271, 0.0271, 0.0270, 0.0270,
             0.0268, 0.0267, 0.0265, 0.0265, 0.0264, 0.0262, 0.0261, 0.0259, 0.0259,
             0.0258, 0.0258, 0.0257, 0.0256, 0.0255, 0.0255, 0.0254, 0.0254, 0.0252,
             0.0251, 0.0251, 0.0250, 0.0249, 0.0249, 0.0247, 0.0245, 0.0243, 0.0241,
             0.0240, 0.0238, 0.0237, 0.0236, 0.0235, 0.0231, 0.0229, 0.0229, 0.0229,
             0.0228, 0.0228, 0.0228, 0.0227, 0.0225, 0.0224, 0.0221, 0.0220, 0.0220,
             0.0220, 0.0220, 0.0219, 0.0218, 0.0217, 0.0216, 0.0216, 0.0216, 0.0214,
             0.0212, 0.0211, 0.0211, 0.0211, 0.0210, 0.0210, 0.0208, 0.0207, 0.0207,
             0.0206, 0.0205, 0.0205, 0.0205, 0.0203, 0.0203, 0.0203, 0.0202, 0.0202,
             0.0201, 0.0201], device='cuda:0'),
     'labels': tensor([16, 16,  2, 64, 44,  2, 16, 16, 16, 86, 44, 44, 16, 16, 64, 16, 16,  2,
             16, 16, 44, 64, 44, 86, 47, 16, 44, 16, 86, 44, 16, 86, 16, 16, 16, 16,
             44, 86, 16, 16,  2, 64, 86, 44, 16, 61, 16, 16, 16, 62, 16, 16, 86, 44,
             16, 16, 64, 16, 16, 16, 44, 16, 16, 16, 16, 44, 44, 62, 31, 16, 64,  1,
             16, 16, 44, 16, 86, 62, 16,  2, 16, 16, 44,  1, 16,  2, 44, 86, 52, 62,
             64, 16, 44, 64, 47, 16, 16, 16, 86, 31,  1,  1, 16, 44, 62, 64, 64,  4,
             44,  2, 52, 16, 16, 16, 16, 44, 86, 31, 52, 16,  4, 16, 44, 27, 52,  1,
             62, 44, 44, 86, 62, 44,  1, 47,  1, 44, 44, 16, 47,  1,  2, 62, 62, 52,
             44, 16, 44, 86, 52, 62, 62, 44, 62, 47, 86,  1, 62, 62,  2, 62,  1, 44,
             47, 62, 27, 47, 44,  2, 47,  1, 62,  1,  1, 62, 64, 62, 64, 62, 62,  1,
             86,  4, 44, 44, 47, 31, 64,  1, 62, 62, 47, 62, 44, 31, 62,  1, 44, 44,
             44,  1], device='cuda:0')}



### Bounding box 그리기


```python

image = image.cpu().data[0]
image = transforms.functional.to_pil_image(image)
image = np.array(image)

threshold=0.5

for box, label, score in zip(prediction["boxes"], prediction["labels"], prediction["scores"]):
    if score > threshold:
        box = list(map(int, box))
        print(box)
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2, cv2.LINE_AA)

fig, axs = plt.subplots(1, 2, figsize = (10, 5))

axs[0].imshow(image_jpg.permute(1, 2, 0))
axs[0].grid(None)
axs[1].imshow(image)
axs[1].grid(None)
plt.show()

```

    [112, 65, 215, 183]



    
*(이미지 없음 / image not preserved: output_24_1.png)*
    


## SSD + CV DNN 얼굴검출
- https://github.com/spmallick/learnopencv/tree/master/AgeGender


```python
import numpy as np
import sys
import cv2
import pandas as pd
img = cv2.imread('./figure/king_face.png')

if img is None:
    print('image read failed')
    sys.exit()

## tensorflow model   
model = './opencv_face_detector/opencv_face_detector_uint8.pb'
config = './opencv_face_detector/opencv_face_detector.pbtxt'

face_net = cv2.dnn.readNet(model, config)
# face_net.getLayerNames()

if face_net.empty():
    print('Net open failed')
    sys.exit()

# blobFromImage(image[, scalefactor[, size[, mean[, swapRB[, crop[, ddepth]]]]]]) -> retval
blob = cv2.dnn.blobFromImage(img, 1, (300, 300), (104, 177, 123),
                            swapRB=False)

face_net.setInput(blob)
out = face_net.forward()

labels = ["img_id", "is_face", "confidence", "left", "top", "right", "bottom"]
out_df = pd.DataFrame(out[0][0], columns = labels)
print(out_df)

```

         img_id  is_face  confidence      left       top     right    bottom
    0       0.0      1.0    0.989431  0.825202  0.502988  0.893821  0.658601
    1       0.0      1.0    0.950553  0.147596  0.511079  0.215222  0.681570
    2       0.0      1.0    0.947276  0.288199  0.444965  0.359195  0.628823
    3       0.0      1.0    0.920967  0.499137  0.392986  0.589955  0.570557
    4       0.0      1.0    0.835135  0.642767  0.463616  0.720524  0.659535
    ..      ...      ...         ...       ...       ...       ...       ...
    195     0.0      0.0    0.000000  0.000000  0.000000  0.000000  0.000000
    196     0.0      0.0    0.000000  0.000000  0.000000  0.000000  0.000000
    197     0.0      0.0    0.000000  0.000000  0.000000  0.000000  0.000000
    198     0.0      0.0    0.000000  0.000000  0.000000  0.000000  0.000000
    199     0.0      0.0    0.000000  0.000000  0.000000  0.000000  0.000000
    
    [200 rows x 7 columns]



```python

detect = out[0, 0, :, :]
h, w = img.shape[:2]

for i in range(detect.shape[0]):
    confidence = detect[i, 2] # (0, 1, confidence, x1, y1, x2, y2)
    
    if confidence > 0.15:
        # out matrix에서 x1, y1, x2, y2 값이 0 ~1로 normalize 되어 있음
        
        x1 = int(detect[i, 3]*w)
        y1 = int(detect[i, 4]*h)
        x2 = int(detect[i, 5]*w)
        y2 = int(detect[i, 6]*h)
        
        cv2.rectangle(img, (x1, y1), (x2, y2),
                     (0, 0, 255), 2)
        
        text = 'Face: {}%'.format(round(confidence*100, 2))
        cv2.putText(img, text, (x1, y1-1), cv2.FONT_HERSHEY_SIMPLEX,
                   0.8, (0, 0, 255), 1, cv2.LINE_AA)
        
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure(figsize = (8,8))
plt.imshow(img)
plt.grid(None)
plt.axis("off")
plt.show()
```


    
*(이미지 없음 / image not preserved: output_27_0.png)*
    


## Yolo v3 객체검출 with opencv dnn


```python
# https://pjreddie.com/darknet/yolo/

# NMSBoxes(bboxes, scores, score_threshold, nms_threshold) -> indices
# nms_threshold: nms_threshold a threshold used in non maximum suppression

# getPerfProfile() -> retval, timings
# .   @brief Returns overall time for inference and timings (in ticks) for layers.

# https://github.com/pjreddie/darknet/blob/master/data/coco.names
```


```python
## Automating with K-Means in Python

import numpy as np
from sklearn.cluster import KMeans

# Example bounding boxes (width, height)
bboxes = np.array([[32, 32], [64, 64], 
                   [128, 128], [256, 256], 
                   [512, 512], [128, 64], 
                   [64, 128], [256, 128]])

# Perform K-means clustering with k=9
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(bboxes)

# Output anchor boxes
anchors = kmeans.cluster_centers_
print("Anchors:", anchors)

```

    Anchors: [[512.  512. ]
     [ 83.2  83.2]
     [256.  192. ]]



```python
import sys
import numpy as np
import cv2

# 모델 & 설정 파일
model = './yolo_v3_pb/yolov3.weights'
config = './yolo_v3_pb/yolov3.cfg'
class_labels = './yolo_v3_pb/coco.names'

# 테스트 이미지 파일
img_files = ['./figure/dog.jpg', 
             './figure/person.jpg', 
             './figure/sheep.jpg', 
             './figure/kite.jpg']


# 네트워크 생성
net = cv2.dnn.readNet(model, config)

if net.empty():
    print('Net open failed!')
    sys.exit()

# 클래스 이름 불러오기
classes = []
with open(class_labels, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# colors = np.random.uniform(0, 255, size=(len(classes), 3))
# colors = np.array([[0, 0, 255], 
#                    [255, 0, 0],
#                    [0, 255, 0],
#                    [0, 255, 255],
#                    [255, 255, 0],
#                    [255, 0, 255]])

# 출력 레이어 이름 받아오기
net.getUnconnectedOutLayers()
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
print(output_layers)
```

    ['yolo_82', 'yolo_94', 'yolo_106']



```python
# outs는 3개의 ndarray 리스트.
# output_layers = ['yolo_82', 'yolo_94', 'yolo_106']
# output_layers[0].shape = (507, 85), 13*13*3
# output_layers[1].shape = (2028, 85), 26*26*3
# output_layers[2].shape = (8112, 85), 52*52*3
import time

confThreshold = 0.5
nmsThreshold = 0.4

# 실행
print(img_files)
for i in img_files:
    img = cv2.imread(i)

    if img is None:
        continue

    # 블롭 생성 & 추론
    blob = cv2.dnn.blobFromImage(img, 1/255., (320, 320), swapRB=True)
    # blob = cv2.dnn.blobFromImage(img, 1/255., (416, 416), swapRB=True)
    # blob = cv2.dnn.blobFromImage(img, 1/255., (608, 608), swapRB=True)

    net.setInput(blob)
    outs = net.forward(output_layers) 

    # outs[0].shape=(507, 85), 13*13*3=507
    # outs[1].shape=(2028, 85), 26*26*3=2028
    # outs[2].shape=(8112, 85), 52*52*3=8112

    h, w = img.shape[:2]

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            # detection: 4(bounding box) + 1(objectness_score) + 80(class confidence)
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confThreshold:
                # 바운딩 박스 중심 좌표 & 박스 크기
                cx = int(detection[0] * w)
                cy = int(detection[1] * h)
                bw = int(detection[2] * w)
                bh = int(detection[3] * h)

                # 바운딩 박스 좌상단 좌표
                sx = int(cx - bw / 2)
                sy = int(cy - bh / 2)

                boxes.append([sx, sy, bw, bh])
                confidences.append(float(confidence))
                class_ids.append(int(class_id))

    # 비최대 억제, Non Max Suppression
# https://deep-learning-study.tistory.com/403
# nmsThreshold: Determines the IoU (Intersection over Union) threshold
# A higher value results in more boxes being retained.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)

    for i in indices:
#         i = i[0]
        sx, sy, bw, bh = boxes[i]
        label = f'{classes[class_ids[i]]}: {confidences[i]:.2}'
        # color = colors[class_ids[i]]
        color = (0, 0, 255)
        cv2.rectangle(img, (sx, sy, bw, bh), color, 2)
        cv2.putText(img, label, (sx, sy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    t, _ = net.getPerfProfile() # Total number of ticks spent during the last forward() call.
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
    
    cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 1, cv2.LINE_AA)
    
    img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.grid(None)
    plt.axis("off")
    plt.show()
```

    ['./figure/dog.jpg', './figure/person.jpg', './figure/sheep.jpg', './figure/kite.jpg']



    
*(이미지 없음 / image not preserved: output_32_1.png)*
    



    
*(이미지 없음 / image not preserved: output_32_2.png)*
    



    
*(이미지 없음 / image not preserved: output_32_3.png)*
    



    
*(이미지 없음 / image not preserved: output_32_4.png)*
    


## Yolo v10 객체검출 with pytorch


```python
!pip install ultralytics
```


```python
from ultralytics import YOLO
```


```python
## coco dataset
# 클래스 이름 불러오기
# classNames = []
with open(class_labels, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
#               "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
#               "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
#               "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
#               "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
#               "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
#               "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
#               "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
#               "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
#               "teddy bear", "hair drier", "toothbrush"
#               ]
```


```python
import time

data_dir = "./figure"

img_path = os.path.join(data_dir, "peoples.jpg")
img = cv2.imread(img_path)

if img is None:
    print("Image read failed")
    sys.exit()

model = YOLO("yolo11x.pt")  # load a pretrained model (recommended for training)

start = time.time()
detection = model(img, verbose=False)[0]
stop = time.time()
on_time = (stop - start)*1000
print(f"estimation time = {on_time:.3f}ms")
fps = f'{1000 / on_time:.4f} fps'
CONFIDENCE_THRESHOLD = 0.6

for data in detection.boxes.data.tolist():
        confidence = data[4]
        if confidence < CONFIDENCE_THRESHOLD:
            continue
        
        xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
        label = int(data[5])
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(img, classNames[label]+ ' ' +str(round(confidence, 2))+'%', 
        (xmin, ymin-5), cv2.FONT_ITALIC, 0.7, (0, 0, 255), 1)        

    
        cv2.putText(img, fps, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)


img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.grid(None)
plt.axis("off")
plt.show()
```

    estimation time = 701.129ms



    
*(이미지 없음 / image not preserved: output_37_1.png)*
    

