# 20장 Vision Transformer (Vit)

> **Migrated from Quartz wiki** (`wiki/study/rokey-ai/basic/ai-intro-20-vit.md`, updated 2026-09-26)  
> tags: rokey, ai, notebook, basic, study  
> source: raw/2026-09-26-rokey-ai/basic/강의_3기_AI개론_20차시__ViT_.ipynb  
> source: https://github.com/weedmo/Rokey-AI  

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
```


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
# 라이브러리 임포트

import torch
from torch import nn, optim
from torchinfo import summary
from torchviz import make_dot
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
```

    c:\Users\user\anaconda3\envs\torchgpu_py3.9\lib\site-packages\google\protobuf\runtime_version.py:112: UserWarning: Protobuf gencode version 5.27.5 is older than the runtime version 5.28.2 at onnx/onnx-ml.proto. Please avoid checked-in Protobuf gencode that can be obsolete.
      warnings.warn(
    c:\Users\user\anaconda3\envs\torchgpu_py3.9\lib\site-packages\google\protobuf\runtime_version.py:112: UserWarning: Protobuf gencode version 5.27.5 is older than the runtime version 5.28.2 at onnx/onnx-operators-ml.proto. Please avoid checked-in Protobuf gencode that can be obsolete.
      warnings.warn(
    c:\Users\user\anaconda3\envs\torchgpu_py3.9\lib\site-packages\google\protobuf\runtime_version.py:112: UserWarning: Protobuf gencode version 5.27.5 is older than the runtime version 5.28.2 at onnx/onnx-data.proto. Please avoid checked-in Protobuf gencode that can be obsolete.
      warnings.warn(



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


# **Transformer**

### 모델 불러 오기


```python
weights = models.ViT_B_16_Weights.IMAGENET1K_V1
vit = models.vit_b_16(weights = weights)
print(weights)
```

    ViT_B_16_Weights.IMAGENET1K_V1



```python
print(vit)
```

    VisionTransformer(
      (conv_proj): Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16))
      (encoder): Encoder(
        (dropout): Dropout(p=0.0, inplace=False)
        (layers): Sequential(
          (encoder_layer_0): EncoderBlock(
            (ln_1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
            (self_attention): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
            )
            (dropout): Dropout(p=0.0, inplace=False)
            (ln_2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
            (mlp): MLPBlock(
              (0): Linear(in_features=768, out_features=3072, bias=True)
              (1): GELU(approximate='none')
              (2): Dropout(p=0.0, inplace=False)
              (3): Linear(in_features=3072, out_features=768, bias=True)
              (4): Dropout(p=0.0, inplace=False)
            )
          )
          (encoder_layer_1): EncoderBlock(
            (ln_1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
            (self_attention): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
            )
            (dropout): Dropout(p=0.0, inplace=False)
            (ln_2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
            (mlp): MLPBlock(
              (0): Linear(in_features=768, out_features=3072, bias=True)
              (1): GELU(approximate='none')
              (2): Dropout(p=0.0, inplace=False)
              (3): Linear(in_features=3072, out_features=768, bias=True)
              (4): Dropout(p=0.0, inplace=False)
            )
          )
          (encoder_layer_2): EncoderBlock(
            (ln_1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
            (self_attention): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
            )
            (dropout): Dropout(p=0.0, inplace=False)
            (ln_2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
            (mlp): MLPBlock(
              (0): Linear(in_features=768, out_features=3072, bias=True)
              (1): GELU(approximate='none')
              (2): Dropout(p=0.0, inplace=False)
              (3): Linear(in_features=3072, out_features=768, bias=True)
              (4): Dropout(p=0.0, inplace=False)
            )
          )
          (encoder_layer_3): EncoderBlock(
            (ln_1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
            (self_attention): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
            )
            (dropout): Dropout(p=0.0, inplace=False)
            (ln_2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
            (mlp): MLPBlock(
              (0): Linear(in_features=768, out_features=3072, bias=True)
              (1): GELU(approximate='none')
              (2): Dropout(p=0.0, inplace=False)
              (3): Linear(in_features=3072, out_features=768, bias=True)
              (4): Dropout(p=0.0, inplace=False)
            )
          )
          (encoder_layer_4): EncoderBlock(
            (ln_1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
            (self_attention): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
            )
            (dropout): Dropout(p=0.0, inplace=False)
            (ln_2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
            (mlp): MLPBlock(
              (0): Linear(in_features=768, out_features=3072, bias=True)
              (1): GELU(approximate='none')
              (2): Dropout(p=0.0, inplace=False)
              (3): Linear(in_features=3072, out_features=768, bias=True)
              (4): Dropout(p=0.0, inplace=False)
            )
          )
          (encoder_layer_5): EncoderBlock(
            (ln_1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
            (self_attention): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
            )
            (dropout): Dropout(p=0.0, inplace=False)
            (ln_2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
            (mlp): MLPBlock(
              (0): Linear(in_features=768, out_features=3072, bias=True)
              (1): GELU(approximate='none')
              (2): Dropout(p=0.0, inplace=False)
              (3): Linear(in_features=3072, out_features=768, bias=True)
              (4): Dropout(p=0.0, inplace=False)
            )
          )
          (encoder_layer_6): EncoderBlock(
            (ln_1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
            (self_attention): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
            )
            (dropout): Dropout(p=0.0, inplace=False)
            (ln_2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
            (mlp): MLPBlock(
              (0): Linear(in_features=768, out_features=3072, bias=True)
              (1): GELU(approximate='none')
              (2): Dropout(p=0.0, inplace=False)
              (3): Linear(in_features=3072, out_features=768, bias=True)
              (4): Dropout(p=0.0, inplace=False)
            )
          )
          (encoder_layer_7): EncoderBlock(
            (ln_1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
            (self_attention): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
            )
            (dropout): Dropout(p=0.0, inplace=False)
            (ln_2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
            (mlp): MLPBlock(
              (0): Linear(in_features=768, out_features=3072, bias=True)
              (1): GELU(approximate='none')
              (2): Dropout(p=0.0, inplace=False)
              (3): Linear(in_features=3072, out_features=768, bias=True)
              (4): Dropout(p=0.0, inplace=False)
            )
          )
          (encoder_layer_8): EncoderBlock(
            (ln_1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
            (self_attention): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
            )
            (dropout): Dropout(p=0.0, inplace=False)
            (ln_2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
            (mlp): MLPBlock(
              (0): Linear(in_features=768, out_features=3072, bias=True)
              (1): GELU(approximate='none')
              (2): Dropout(p=0.0, inplace=False)
              (3): Linear(in_features=3072, out_features=768, bias=True)
              (4): Dropout(p=0.0, inplace=False)
            )
          )
          (encoder_layer_9): EncoderBlock(
            (ln_1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
            (self_attention): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
            )
            (dropout): Dropout(p=0.0, inplace=False)
            (ln_2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
            (mlp): MLPBlock(
              (0): Linear(in_features=768, out_features=3072, bias=True)
              (1): GELU(approximate='none')
              (2): Dropout(p=0.0, inplace=False)
              (3): Linear(in_features=3072, out_features=768, bias=True)
              (4): Dropout(p=0.0, inplace=False)
            )
          )
          (encoder_layer_10): EncoderBlock(
            (ln_1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
            (self_attention): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
            )
            (dropout): Dropout(p=0.0, inplace=False)
            (ln_2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
            (mlp): MLPBlock(
              (0): Linear(in_features=768, out_features=3072, bias=True)
              (1): GELU(approximate='none')
              (2): Dropout(p=0.0, inplace=False)
              (3): Linear(in_features=3072, out_features=768, bias=True)
              (4): Dropout(p=0.0, inplace=False)
            )
          )
          (encoder_layer_11): EncoderBlock(
            (ln_1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
            (self_attention): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
            )
            (dropout): Dropout(p=0.0, inplace=False)
            (ln_2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
            (mlp): MLPBlock(
              (0): Linear(in_features=768, out_features=3072, bias=True)
              (1): GELU(approximate='none')
              (2): Dropout(p=0.0, inplace=False)
              (3): Linear(in_features=3072, out_features=768, bias=True)
              (4): Dropout(p=0.0, inplace=False)
            )
          )
        )
        (ln): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
      )
      (heads): Sequential(
        (head): Linear(in_features=768, out_features=1000, bias=True)
      )
    )



```python
summary(vit,(1, 3, 224, 224))
```

    c:\Users\user\anaconda3\envs\torchgpu_py3.9\lib\site-packages\torch\nn\modules\activation.py:1221: UserWarning: 1Torch was not compiled with flash attention. (Triggered internally at C:\actions-runner\_work\pytorch\pytorch\builder\windows\pytorch\aten\src\ATen\native\transformers\cuda\sdp_utils.cpp:455.)
      return torch._native_multi_head_attention(





    ===============================================================================================
    Layer (type:depth-idx)                        Output Shape              Param #
    ===============================================================================================
    VisionTransformer                             [1, 1000]                 768
    ├─Conv2d: 1-1                                 [1, 768, 14, 14]          590,592
    ├─Encoder: 1-2                                [1, 197, 768]             151,296
    │    └─Dropout: 2-1                           [1, 197, 768]             --
    │    └─Sequential: 2-2                        [1, 197, 768]             --
    │    │    └─EncoderBlock: 3-1                 [1, 197, 768]             7,087,872
    │    │    └─EncoderBlock: 3-2                 [1, 197, 768]             7,087,872
    │    │    └─EncoderBlock: 3-3                 [1, 197, 768]             7,087,872
    │    │    └─EncoderBlock: 3-4                 [1, 197, 768]             7,087,872
    │    │    └─EncoderBlock: 3-5                 [1, 197, 768]             7,087,872
    │    │    └─EncoderBlock: 3-6                 [1, 197, 768]             7,087,872
    │    │    └─EncoderBlock: 3-7                 [1, 197, 768]             7,087,872
    │    │    └─EncoderBlock: 3-8                 [1, 197, 768]             7,087,872
    │    │    └─EncoderBlock: 3-9                 [1, 197, 768]             7,087,872
    │    │    └─EncoderBlock: 3-10                [1, 197, 768]             7,087,872
    │    │    └─EncoderBlock: 3-11                [1, 197, 768]             7,087,872
    │    │    └─EncoderBlock: 3-12                [1, 197, 768]             7,087,872
    │    └─LayerNorm: 2-3                         [1, 197, 768]             1,536
    ├─Sequential: 1-3                             [1, 1000]                 --
    │    └─Linear: 2-4                            [1, 1000]                 769,000
    ===============================================================================================
    Total params: 86,567,656
    Trainable params: 86,567,656
    Non-trainable params: 0
    Total mult-adds (M): 173.23
    ===============================================================================================
    Input size (MB): 0.60
    Forward/backward pass size (MB): 104.09
    Params size (MB): 232.27
    Estimated Total Size (MB): 336.96
    ===============================================================================================




```python
from PIL import Image

filename = "./golden.jpg"
img = Image.open(filename) # torch.Size([3, 366, 640]) 

plt.imshow(img)
plt.grid(None)
plt.axis("off")
plt.show()

```


    
*(이미지 없음 / image not preserved: output_13_0.png)*
    



```python
preprocessed = weights.transforms()
img_tensor = preprocessed(img)

print("Image tensor shape:", img_tensor.shape)  # Should be [3, 224, 224]
```

    Image tensor shape: torch.Size([1, 3, 224, 224])



```python
vit = vit.to(device)

vit.eval()
```




    VisionTransformer(
      (conv_proj): Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16))
      (encoder): Encoder(
        (dropout): Dropout(p=0.0, inplace=False)
        (layers): Sequential(
          (encoder_layer_0): EncoderBlock(
            (ln_1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
            (self_attention): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
            )
            (dropout): Dropout(p=0.0, inplace=False)
            (ln_2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
            (mlp): MLPBlock(
              (0): Linear(in_features=768, out_features=3072, bias=True)
              (1): GELU(approximate='none')
              (2): Dropout(p=0.0, inplace=False)
              (3): Linear(in_features=3072, out_features=768, bias=True)
              (4): Dropout(p=0.0, inplace=False)
            )
          )
          (encoder_layer_1): EncoderBlock(
            (ln_1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
            (self_attention): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
            )
            (dropout): Dropout(p=0.0, inplace=False)
            (ln_2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
            (mlp): MLPBlock(
              (0): Linear(in_features=768, out_features=3072, bias=True)
              (1): GELU(approximate='none')
              (2): Dropout(p=0.0, inplace=False)
              (3): Linear(in_features=3072, out_features=768, bias=True)
              (4): Dropout(p=0.0, inplace=False)
            )
          )
          (encoder_layer_2): EncoderBlock(
            (ln_1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
            (self_attention): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
            )
            (dropout): Dropout(p=0.0, inplace=False)
            (ln_2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
            (mlp): MLPBlock(
              (0): Linear(in_features=768, out_features=3072, bias=True)
              (1): GELU(approximate='none')
              (2): Dropout(p=0.0, inplace=False)
              (3): Linear(in_features=3072, out_features=768, bias=True)
              (4): Dropout(p=0.0, inplace=False)
            )
          )
          (encoder_layer_3): EncoderBlock(
            (ln_1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
            (self_attention): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
            )
            (dropout): Dropout(p=0.0, inplace=False)
            (ln_2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
            (mlp): MLPBlock(
              (0): Linear(in_features=768, out_features=3072, bias=True)
              (1): GELU(approximate='none')
              (2): Dropout(p=0.0, inplace=False)
              (3): Linear(in_features=3072, out_features=768, bias=True)
              (4): Dropout(p=0.0, inplace=False)
            )
          )
          (encoder_layer_4): EncoderBlock(
            (ln_1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
            (self_attention): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
            )
            (dropout): Dropout(p=0.0, inplace=False)
            (ln_2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
            (mlp): MLPBlock(
              (0): Linear(in_features=768, out_features=3072, bias=True)
              (1): GELU(approximate='none')
              (2): Dropout(p=0.0, inplace=False)
              (3): Linear(in_features=3072, out_features=768, bias=True)
              (4): Dropout(p=0.0, inplace=False)
            )
          )
          (encoder_layer_5): EncoderBlock(
            (ln_1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
            (self_attention): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
            )
            (dropout): Dropout(p=0.0, inplace=False)
            (ln_2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
            (mlp): MLPBlock(
              (0): Linear(in_features=768, out_features=3072, bias=True)
              (1): GELU(approximate='none')
              (2): Dropout(p=0.0, inplace=False)
              (3): Linear(in_features=3072, out_features=768, bias=True)
              (4): Dropout(p=0.0, inplace=False)
            )
          )
          (encoder_layer_6): EncoderBlock(
            (ln_1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
            (self_attention): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
            )
            (dropout): Dropout(p=0.0, inplace=False)
            (ln_2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
            (mlp): MLPBlock(
              (0): Linear(in_features=768, out_features=3072, bias=True)
              (1): GELU(approximate='none')
              (2): Dropout(p=0.0, inplace=False)
              (3): Linear(in_features=3072, out_features=768, bias=True)
              (4): Dropout(p=0.0, inplace=False)
            )
          )
          (encoder_layer_7): EncoderBlock(
            (ln_1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
            (self_attention): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
            )
            (dropout): Dropout(p=0.0, inplace=False)
            (ln_2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
            (mlp): MLPBlock(
              (0): Linear(in_features=768, out_features=3072, bias=True)
              (1): GELU(approximate='none')
              (2): Dropout(p=0.0, inplace=False)
              (3): Linear(in_features=3072, out_features=768, bias=True)
              (4): Dropout(p=0.0, inplace=False)
            )
          )
          (encoder_layer_8): EncoderBlock(
            (ln_1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
            (self_attention): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
            )
            (dropout): Dropout(p=0.0, inplace=False)
            (ln_2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
            (mlp): MLPBlock(
              (0): Linear(in_features=768, out_features=3072, bias=True)
              (1): GELU(approximate='none')
              (2): Dropout(p=0.0, inplace=False)
              (3): Linear(in_features=3072, out_features=768, bias=True)
              (4): Dropout(p=0.0, inplace=False)
            )
          )
          (encoder_layer_9): EncoderBlock(
            (ln_1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
            (self_attention): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
            )
            (dropout): Dropout(p=0.0, inplace=False)
            (ln_2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
            (mlp): MLPBlock(
              (0): Linear(in_features=768, out_features=3072, bias=True)
              (1): GELU(approximate='none')
              (2): Dropout(p=0.0, inplace=False)
              (3): Linear(in_features=3072, out_features=768, bias=True)
              (4): Dropout(p=0.0, inplace=False)
            )
          )
          (encoder_layer_10): EncoderBlock(
            (ln_1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
            (self_attention): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
            )
            (dropout): Dropout(p=0.0, inplace=False)
            (ln_2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
            (mlp): MLPBlock(
              (0): Linear(in_features=768, out_features=3072, bias=True)
              (1): GELU(approximate='none')
              (2): Dropout(p=0.0, inplace=False)
              (3): Linear(in_features=3072, out_features=768, bias=True)
              (4): Dropout(p=0.0, inplace=False)
            )
          )
          (encoder_layer_11): EncoderBlock(
            (ln_1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
            (self_attention): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
            )
            (dropout): Dropout(p=0.0, inplace=False)
            (ln_2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
            (mlp): MLPBlock(
              (0): Linear(in_features=768, out_features=3072, bias=True)
              (1): GELU(approximate='none')
              (2): Dropout(p=0.0, inplace=False)
              (3): Linear(in_features=3072, out_features=768, bias=True)
              (4): Dropout(p=0.0, inplace=False)
            )
          )
        )
        (ln): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
      )
      (heads): Sequential(
        (head): Linear(in_features=768, out_features=1000, bias=True)
      )
    )




```python
img = img_tensor.unsqueeze(0).to(device)
output = vit(img)
output
```




    tensor([[ 1.7242e-01, -1.3533e-01, -5.5473e-01, -3.2600e-01, -3.2524e-01,
              6.0302e-02, -1.4193e-01,  3.0145e-03,  5.0543e-02, -4.9414e-02,
             -8.6714e-02, -4.5993e-01, -2.6059e-01,  1.5029e-01, -3.2692e-01,
              3.1221e-01, -1.4011e-01, -3.4145e-01,  1.5873e-01, -2.6251e-01,
             -1.7628e-01, -8.3054e-02,  1.4802e-01,  5.7551e-01, -2.7244e-01,
             -2.0661e-01,  1.2875e-01,  4.3234e-03, -3.0197e-01,  2.5118e-01,
              2.5700e-01, -1.6148e-01,  2.9481e-01,  1.5440e-01,  1.3747e-01,
              2.4419e-01,  1.4160e-01, -3.6953e-01,  3.0247e-01, -9.8280e-02,
              3.1038e-01, -3.0506e-01,  1.8005e-01,  1.4380e-01, -1.4137e-01,
              3.2926e-01, -1.3490e-01,  2.9605e-01, -1.7539e-01, -6.6714e-02,
             -2.6848e-01,  1.9891e-01, -4.4626e-02, -3.6104e-01, -2.6228e-01,
             -7.2178e-02, -1.6355e-01,  2.3819e-01,  1.5466e-01, -5.8613e-01,
             -8.3064e-01, -5.1237e-01, -9.1716e-02,  9.9686e-02,  2.9706e-02,
              8.7632e-02, -8.6670e-02, -4.8205e-01, -3.3912e-01, -4.1230e-01,
              2.6116e-01, -2.0237e-01, -3.4868e-01,  5.0008e-02, -5.0095e-01,
             -1.6590e-01, -1.6842e-01, -7.4002e-02, -1.2656e-01, -1.7148e-01,
             -3.0670e-01, -4.1156e-02,  6.7786e-01, -1.4499e-01,  1.2530e-01,
             -7.1696e-01,  1.8497e-01,  8.9239e-02, -4.1724e-01,  7.1521e-02,
             -7.9664e-02, -2.1433e-01, -4.6403e-01, -2.9512e-01, -7.7186e-02,
             -8.3919e-02,  1.0270e-01, -4.4291e-02,  6.4035e-02, -4.9280e-01,
             -2.0283e-01, -2.8773e-02, -2.7155e-02,  5.7732e-01, -8.7235e-02,
             -5.6568e-01, -4.0754e-01, -2.5758e-01, -6.1545e-01, -2.4592e-01,
              1.8440e-01,  1.6873e-02, -5.3494e-01, -2.3342e-01,  6.8058e-02,
             -4.3753e-01, -1.2484e-01,  1.7283e-02,  4.0450e-01,  2.0901e-01,
             -2.6206e-01, -2.0312e-02,  1.9752e-01, -3.4057e-01, -4.6482e-02,
              3.1208e-02, -1.4594e-01,  5.4763e-01,  1.1905e-01, -4.4474e-01,
             -3.1170e-01,  1.6952e-01, -3.7668e-01, -3.5738e-01,  2.0584e-01,
             -8.2327e-02, -4.8097e-01,  2.6133e-01, -3.1380e-01,  2.2430e-01,
             -1.6384e-01,  4.7528e-02,  1.2883e-01,  3.3141e-01, -4.9040e-01,
             -2.3123e-02, -8.2234e-01,  3.1993e-02, -4.2399e-01,  2.4114e-01,
              1.9886e-01, -2.6285e-01,  5.0362e-01, -4.7001e-01,  1.3830e-01,
              2.7675e-01,  1.2395e+00,  3.0427e-02,  1.1046e+00,  7.0079e-01,
             -3.9479e-01,  4.9518e-01,  4.2247e-01, -9.0414e-01, -1.0252e+00,
             -8.9828e-01,  1.8372e+00,  9.9317e-01, -1.0510e-01, -1.2781e+00,
              5.2464e-01, -1.0490e+00, -1.5512e+00,  1.5879e-01, -3.3093e-01,
              7.9808e-01, -5.5136e-01, -7.6471e-01,  5.0019e-01, -1.0083e+00,
             -1.2429e+00, -1.7660e-01,  1.5703e-02,  1.5307e-01,  7.1253e-01,
              2.4145e-01, -2.9939e-01,  5.2417e-01,  8.4676e-01,  4.7893e-01,
              2.0228e-02,  1.4447e+00,  3.3412e-01,  2.5939e-01,  3.1075e-01,
              9.8168e-01, -4.1121e-02, -3.4375e-01, -3.2826e-01, -3.9243e-01,
              7.5814e-01, -3.4723e-01, -1.2844e-01, -6.1654e-01, -1.9289e-01,
              2.1105e+00,  1.4147e+00,  9.0959e+00,  5.9681e+00,  1.3017e+00,
              4.0787e-01,  3.2535e-01,  1.1311e+00,  1.6051e+00,  1.6167e+00,
              3.6339e+00,  1.9924e+00,  2.8683e-01,  7.4827e-01,  5.7033e-01,
              2.5937e+00, -1.2825e-01,  2.3927e+00, -1.0534e+00, -4.0766e-01,
             -1.5255e+00,  1.2546e-01, -5.2228e-01,  1.0978e+00,  1.5120e-01,
             -3.3116e-01,  8.9133e-01,  1.0068e+00,  4.9555e-02, -2.0533e-01,
             -3.7293e-01,  7.9265e-01,  1.8388e-02,  1.3165e+00,  1.1429e+00,
              2.1104e-01, -2.8398e-01,  2.0138e-01,  4.0801e-01, -5.4816e-01,
             -2.6917e-01, -1.2980e-01,  1.0200e+00,  4.6088e-01, -1.5420e-01,
             -1.3503e+00, -3.8795e-01,  3.5089e-01,  7.7339e-02, -5.8547e-01,
              8.8975e-01,  3.5718e-01,  6.0955e-01, -6.7947e-02,  3.1932e-01,
              4.8629e-01, -4.8666e-01, -4.1725e-01, -2.2962e-02, -4.9043e-01,
              4.9310e-01,  1.1918e+00,  1.3865e+00,  5.5509e-01, -9.9139e-01,
             -8.2322e-01, -4.5467e-01, -4.5557e-02,  1.6933e-01, -4.7560e-01,
             -1.6993e-01,  3.6869e-01, -6.8544e-01,  3.3707e-01, -1.0709e+00,
              1.6660e-01,  9.4918e-01,  7.8144e-01,  2.9784e-01,  5.9820e-01,
              1.6122e-01, -6.4121e-01, -1.0048e-01,  2.0229e-01, -4.6350e-01,
              1.9077e-01,  3.8498e-01, -4.5007e-01,  3.3928e-01,  1.4289e-01,
             -3.1421e-01, -4.3228e-01, -2.0821e-01, -8.6272e-01, -1.4338e-01,
              4.7283e-01,  3.7132e-01,  2.4120e-01,  2.1093e-01,  2.2518e-01,
              1.6119e-01,  3.9178e-01, -5.4066e-01,  1.3744e-01, -6.3491e-02,
             -5.4225e-01,  2.2937e-01,  7.0131e-02, -2.4479e-01, -6.8883e-01,
             -1.0793e-01, -6.3330e-01,  2.5944e-01,  1.2283e-01,  1.5734e-01,
              1.2471e-01,  3.8190e-01,  5.2634e-01, -5.6937e-02, -2.7870e-01,
             -6.4935e-01, -4.4016e-01,  2.9842e-01, -4.6675e-01,  4.3094e-02,
             -1.8817e-01,  5.7068e-01,  5.6752e-01,  2.9789e-02,  3.0812e-01,
             -1.0371e-01,  1.7258e-01, -8.4501e-02, -2.7659e-01, -6.2658e-01,
              4.2016e-03, -5.5366e-01, -3.2028e-01, -6.7992e-01,  1.5414e-01,
             -6.7077e-02, -1.3284e-01, -3.0875e-01, -8.0844e-01, -4.9112e-01,
             -3.4379e-01,  1.1217e-02, -5.2387e-01, -1.9592e-01, -7.8514e-01,
             -8.5312e-02, -1.3523e-01, -9.1780e-02,  3.5169e-01, -2.1364e-01,
             -3.9375e-01, -1.8432e-01,  3.4137e-01,  5.3120e-02, -1.1600e-01,
              2.4752e-01, -2.7064e-01, -3.1753e-01,  2.3848e-01,  5.3228e-01,
              2.0010e-01,  4.4266e-01, -7.1181e-01,  3.7898e-02,  2.3858e-01,
             -9.1807e-01, -2.2641e-01, -3.5029e-01, -5.0150e-01, -1.1581e+00,
              2.5384e-01, -1.3119e-01, -6.1444e-02, -9.3563e-02, -1.1872e-01,
             -1.6478e-01, -4.6231e-01, -6.5962e-01, -4.0729e-01, -1.0928e-01,
             -6.5995e-01, -3.2489e-01,  1.4635e-01, -1.4191e-01,  1.0512e-01,
              3.6864e-02, -4.4372e-01, -4.9659e-01,  1.5549e-01, -3.0470e-01,
              1.1639e-01, -2.9459e-02, -1.0826e-01, -3.7785e-01, -7.7129e-03,
             -1.3621e-01, -5.1815e-01,  1.7771e-01,  2.7807e-01,  3.0155e-01,
             -4.1124e-01, -1.5622e-01, -5.0180e-01, -2.0270e-01, -1.4885e-01,
             -5.4540e-01, -1.4659e-01, -2.3358e-01, -5.5808e-01,  3.1665e-01,
             -3.1822e-01, -3.6972e-01, -2.9316e-01, -2.6343e-01,  4.7082e-02,
             -1.7853e-01, -4.1090e-01, -4.5294e-01, -8.1702e-02, -9.8545e-02,
              9.9474e-02, -5.3942e-01,  5.1590e-02, -1.5806e-01,  8.1521e-01,
              2.8540e-01, -2.9320e-01,  4.9368e-01, -6.6760e-02,  5.5084e-01,
             -1.5213e-01, -3.0106e-01, -1.6635e-01, -6.5563e-01,  2.0963e-01,
             -4.1204e-01, -2.9966e-01, -3.4062e-01, -3.1361e-01, -1.1823e-01,
              1.4229e-01, -3.0244e-01, -4.0364e-01, -1.9384e-01, -5.2968e-01,
             -3.8935e-01, -4.3123e-01, -1.9598e-01,  3.0799e-01, -5.1536e-01,
              2.4130e-01,  1.2957e-01,  3.1814e-01,  4.0341e-01,  1.1551e-01,
             -1.9015e-02, -4.5434e-02, -1.1250e-01, -3.8999e-01, -2.3926e-01,
             -3.0663e-01, -1.9448e-02, -2.1778e-01, -3.0415e-02, -3.1584e-01,
             -2.8635e-01,  1.2385e-01, -2.0676e-01,  3.3003e-01, -9.4973e-02,
             -8.4721e-02, -4.0941e-01, -3.4302e-01, -6.7307e-02, -5.5743e-02,
             -3.2931e-01, -4.1091e-01, -1.4741e-01,  4.8135e-02, -5.7207e-01,
             -7.8918e-01, -4.8658e-01, -8.6676e-01,  1.2072e-01, -3.2996e-01,
             -3.5234e-01, -3.4474e-01, -1.9556e-02, -1.4326e-02, -3.0434e-01,
             -4.2574e-01, -3.2869e-02,  8.7640e-02,  3.9863e-01, -8.9700e-02,
              2.6106e-01, -5.3803e-01, -4.6759e-02, -2.9096e-01,  1.4745e-01,
              3.3900e-01,  1.1369e-01, -3.4371e-01, -2.5259e-01, -5.7716e-01,
              2.2051e-01, -3.8378e-01, -8.8034e-02,  2.0504e-01, -1.7580e-01,
              2.3259e-01, -7.0707e-02,  1.0568e-01, -2.0526e-01, -6.0287e-03,
             -3.8697e-01, -9.1261e-02, -2.5185e-01, -2.0375e-01,  1.9690e-01,
             -5.5242e-02,  5.7298e-03, -1.1830e-01, -7.4495e-01,  3.9882e-01,
             -2.5940e-01, -2.6826e-01, -4.9874e-01, -2.9366e-01,  1.1856e+00,
             -4.4820e-01, -7.3915e-01,  2.9252e-01,  8.0297e-01, -5.5171e-01,
              3.7413e-02, -1.7804e-01,  2.7056e-01, -1.2087e-01,  3.2347e-01,
              2.7676e-01,  9.7123e-02,  2.7505e-01,  4.0591e-01,  2.0201e-01,
             -2.7270e-01, -6.8610e-01, -9.7139e-02,  6.1551e-02,  7.5015e-01,
             -1.9304e-01,  1.3045e-01, -5.3959e-02, -1.1976e-01,  1.9059e-02,
              1.5713e-01,  2.1037e-01, -5.7357e-01, -3.7741e-01,  1.9483e-01,
             -1.6948e-01, -7.7195e-02, -6.1098e-01, -1.1469e-01,  2.6326e-01,
             -2.0089e-01, -4.3447e-02, -1.3429e-01,  2.7541e-01,  1.6660e-01,
             -3.0677e-01, -3.6381e-01,  2.0549e-01, -1.0067e+00, -1.3855e-01,
              8.6831e-02,  3.6047e-01, -9.1658e-02, -1.3885e-01, -1.8596e-01,
             -2.5029e-01,  1.1806e-01, -3.8802e-02, -1.6966e-01, -5.2452e-01,
             -1.6738e-01, -6.4935e-01, -3.4249e-01, -6.8920e-01, -1.5729e-01,
             -3.7415e-01, -1.4640e-01,  3.5636e-01, -8.3016e-02, -7.1431e-03,
              1.2450e-01,  3.4161e-02, -8.2627e-01,  7.6313e-01, -3.4709e-01,
             -3.4772e-01, -1.2747e-01, -2.6818e-01, -5.8576e-01, -5.7772e-01,
              2.4355e-01, -2.9402e-01,  5.7519e-01,  5.7611e-01,  6.3158e-02,
              3.6946e-01, -8.4244e-01,  2.1957e-01,  1.5411e-01, -2.3260e-01,
             -3.0996e-01,  5.8615e-02, -1.2517e-01, -1.8075e-01,  2.5582e-01,
              4.3955e-01, -1.6214e-01,  1.8557e-01, -1.4935e-01,  2.4670e-01,
              7.0228e-01, -7.4643e-02, -8.8025e-03,  2.4561e-01, -1.0284e-01,
              3.1141e-03,  9.4802e-02, -4.2814e-02, -2.2996e-01, -2.5689e-01,
             -6.1418e-01, -4.6437e-02, -8.4782e-02,  7.2430e-02,  1.0570e-01,
              2.1582e-02,  2.1333e-01, -2.2961e-01, -2.0857e-01, -8.4543e-02,
              1.2491e-01,  2.5163e-02, -2.3431e-01, -2.2363e-01, -2.1814e-01,
             -1.0264e-01, -4.3172e-02, -2.9102e-01, -4.9441e-01, -7.2800e-02,
              8.2214e-02, -2.4999e-01,  2.1934e-01,  7.4396e-02, -4.3672e-02,
             -5.9440e-02, -2.5615e-01, -2.7244e-02, -4.9970e-01, -1.0168e-01,
             -3.9909e-01, -3.9892e-01,  2.0453e-01,  4.7146e-01, -2.3963e-01,
             -2.6747e-01, -2.7220e-01, -1.2940e-01,  2.7703e-01,  3.1557e-01,
             -2.8199e-01,  2.0278e-01, -3.2618e-01,  2.4193e-01, -1.2513e-01,
             -4.6505e-01,  2.4490e-02, -1.8161e-01,  1.9471e-03,  8.1418e-03,
             -1.8312e-01, -1.7128e-01,  1.2141e-01,  4.8816e-02,  7.5712e-02,
              2.2285e-01,  1.3848e-01, -3.7429e-01, -6.1453e-01,  3.4228e-01,
             -2.3332e-03, -3.4146e-01, -3.0150e-01,  5.1634e-01,  3.3286e-02,
             -4.7529e-02, -6.9810e-01,  8.5853e-02,  3.2189e-01, -2.0521e-01,
              1.2339e-01, -5.5054e-01, -2.8348e-02,  2.2689e-01, -3.6344e-01,
             -1.0764e-01, -6.3705e-01,  1.1619e-02, -8.1453e-02, -1.5518e-01,
             -4.1930e-01, -2.8809e-01, -1.6053e-01,  2.6224e-01,  8.9127e-02,
              3.9913e-01,  1.1677e-03, -4.3315e-01, -5.3390e-01,  2.3053e-01,
              2.1918e-01, -1.3308e-01, -4.0319e-01, -1.0348e-01,  1.3807e-01,
              9.7878e-02,  5.9253e-01, -2.3690e-01, -3.2694e-01,  2.2409e-02,
             -5.6232e-01, -3.6002e-01, -3.5883e-02, -4.8406e-01, -1.8727e-01,
              4.8229e-01, -3.8691e-01, -4.0583e-01, -9.5624e-01,  4.5627e-02,
             -4.9914e-01, -4.9791e-01,  1.2162e-01,  9.7296e-02, -1.7672e-01,
              4.8241e-02,  4.2538e-02, -2.6486e-01,  1.8208e-01, -6.5736e-02,
             -1.9335e-01,  7.8041e-02, -1.7279e-01,  2.4278e-01, -1.5790e-01,
              3.0745e-03, -3.9149e-01, -1.4596e-01, -1.4837e-01,  1.1907e-01,
             -2.0714e-01, -1.2942e-01, -4.6821e-01, -4.5139e-01, -2.2995e-01,
              1.0909e-01, -8.6097e-02,  3.0634e-01, -1.9069e-01, -6.5564e-01,
              2.4918e-01,  6.9581e-02, -5.5154e-01, -3.9221e-01, -1.8285e-01,
             -2.5633e-01, -6.9683e-01, -2.8287e-01,  2.4451e-01,  2.3096e-01,
             -4.2396e-01,  2.0889e-02, -1.2135e-01,  2.0218e-01, -4.1051e-01,
             -9.4909e-02,  4.0625e-01, -1.5290e-02,  1.1784e-01,  1.2668e-01,
              2.1321e+00, -1.2904e-01,  1.1129e-01,  1.2829e-01, -1.1127e-01,
              2.9810e-01, -1.7866e-02,  1.5382e-02,  3.3549e-01, -1.2782e-01,
              2.1037e-01, -3.7039e-01, -5.8314e-01,  2.0098e-01, -2.7104e-01,
             -2.4185e-01, -3.6742e-01, -2.1659e-01, -6.2800e-01, -3.8423e-01,
              1.5984e-01,  2.6681e-01,  9.4210e-02, -1.8401e-01, -4.5175e-01,
             -4.7724e-01,  4.0087e-01,  3.7280e-03, -1.1641e-01,  2.7488e-01,
             -7.8398e-02,  2.3844e-01, -3.3062e-01, -2.9984e-02, -4.8608e-01,
             -1.7625e-01,  3.9606e-01,  1.8317e-01, -1.8544e-01, -3.3829e-01,
             -4.2248e-01, -2.9587e-01, -1.1692e-02, -8.0564e-02,  2.9799e-01,
             -2.7148e-02,  2.4370e-01,  2.5362e+00, -2.9350e-01,  3.7484e-01,
             -1.5281e-01, -2.2980e-01, -1.8951e-01, -2.5019e-01, -2.6427e-01,
             -1.6784e-01,  7.3725e-02, -3.3012e-01, -2.2280e-01, -7.5307e-03,
             -6.5080e-01,  3.6131e-01, -2.6393e-01,  4.8487e-03, -1.8136e-01,
              1.1616e-01, -5.1273e-01, -1.2963e-01,  1.9467e-01, -3.0255e-01,
              2.4787e-01,  6.7418e-01, -3.9782e-01,  1.5496e-01, -4.4804e-01,
              4.4110e-02, -8.5661e-02, -4.7953e-02, -1.2277e-01, -1.4821e-01,
              5.2790e-01,  4.6331e-02, -7.0609e-02, -5.1341e-01,  2.9867e-01,
             -1.1900e-02,  1.1543e-03, -7.5050e-02, -5.8691e-02,  1.6466e-01,
             -4.5072e-01, -8.0915e-02, -9.7755e-02,  1.7667e-01, -8.2371e-02,
             -7.7544e-02,  1.6065e-01,  2.6157e-01, -7.5106e-01, -1.4866e-01,
             -4.7457e-01,  5.2498e-01, -1.9754e-01, -4.1618e-01, -5.0529e-01,
             -5.6762e-02,  3.4012e-01, -1.5214e-01,  4.9457e-01, -8.7458e-02,
              2.2440e-02, -3.0998e-01,  5.4757e-02, -4.3918e-01,  3.4206e-01,
             -1.6649e-01, -5.9853e-01, -5.7325e-01, -3.1225e-01,  4.9790e-01,
              2.9516e-01, -2.8060e-01,  2.5567e-01, -4.0543e-01,  4.6037e-01,
             -2.5863e-01,  5.3356e-01, -3.8728e-01, -1.4177e-01, -1.1428e-02,
             -1.2759e-01, -3.8310e-01,  2.2130e-01, -4.4342e-01, -1.7307e-01,
             -2.5645e-01, -5.6469e-01,  1.4531e-01, -4.0682e-01, -6.8922e-03,
             -6.0456e-01,  3.6180e-02,  5.6250e-01,  7.4718e-02,  1.3483e-01,
              3.9327e-01,  3.8916e-02,  2.9880e-01, -6.0860e-02,  1.6088e-01,
              8.4129e-02, -2.6258e-01,  2.5040e-01,  3.9916e-01, -3.9546e-01,
             -5.1292e-01, -5.7884e-02, -5.6947e-01, -5.3683e-01,  2.8356e-01,
             -2.5264e-01,  3.9299e-01, -5.3464e-01,  1.6740e-01, -3.4949e-02,
             -1.4565e-01, -7.6740e-02, -1.6362e-01,  6.3027e-02,  3.4604e-03,
              2.5973e-02,  1.9449e-02, -3.7329e-01, -5.3344e-02, -2.3836e-01,
             -4.0919e-01, -3.2514e-01,  2.1826e-01,  3.1947e-01, -5.9651e-02,
             -7.1580e-01, -3.0824e-01, -5.5377e-01, -2.1805e-01, -2.5724e-01,
              2.8294e-01, -5.1901e-01, -5.6830e-01,  2.1898e-01, -4.8628e-01,
              2.1736e-01,  1.1770e-01, -3.2656e-01, -1.1077e-02,  2.0961e-01]],
           device='cuda:0', grad_fn=<AddmmBackward0>)




```python
idx = output.argmax(-1)
print(idx)
```

    tensor([207], device='cuda:0')



```python
weights.meta["categories"][idx]
```




    'golden retriever'


