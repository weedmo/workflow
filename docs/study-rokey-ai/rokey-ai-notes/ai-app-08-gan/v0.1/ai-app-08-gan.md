# 8장 적대적 생성 신경망 (Generative adversarial network)

> **Migrated from Quartz wiki** (`wiki/study/rokey-ai/intermediate/ai-app-08-gan.md`, updated 2026-09-26)  
> tags: rokey, ai, notebook, intermediate, study  
> source: raw/2026-09-26-rokey-ai/intermediate/강의_3기_AI응용_8차시__GAN_.ipynb  
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


```python
import os
import numpy as np
import matplotlib.pyplot as plt


import cv2
import torch
from torch import nn, optim
from torchvision import transforms, datasets
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.autograd import Variable

```


```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device = ", device)
```

    device =  cuda


## Vanilla GAN

### Vanilla GAN hyperparameters


```python
# Hyper-parameters & Variables setting
num_epoch = 1000
batch_size = 100
learning_rate = 0.0002
img_size = 28 * 28
num_channel = 1
dir_name = "GAN_results"

noise_size = 100
hidden_size1 = 256
hidden_size2 = 512
# hidden_size3 = 1024

# Create a directory for saving samples
if not os.path.exists(dir_name):
    os.makedirs(dir_name)
```

### MNIST 데이터 


```python
# Dataset transform setting
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)])
```


```python
# MNIST dataset setting
MNIST_dataset = datasets.MNIST(root='./',
                                train=True,
                                transform=transform,
                                download=True)

# Data loader
data_loader = DataLoader(dataset=MNIST_dataset,
                        batch_size=batch_size,
                        shuffle=True)
```

    100%|██████████| 9.91M/9.91M [00:02<00:00, 4.46MB/s]
    100%|██████████| 28.9k/28.9k [00:00<00:00, 164kB/s]
    100%|██████████| 1.65M/1.65M [00:01<00:00, 1.51MB/s]
    100%|██████████| 4.54k/4.54k [00:00<00:00, 4.55MB/s]


### 판별기 (Discriminator) 모델


```python
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(img_size, hidden_size2)
        self.linear2 = nn.Linear(hidden_size2, hidden_size1)
        self.linear3 = nn.Linear(hidden_size1, 1)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.leaky_relu(self.linear1(x))
        x = self.leaky_relu(self.linear2(x))
        x = self.linear3(x)
        x = self.sigmoid(x)
        return x
```

### 생성기 (Generator) 모델


```python
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.linear1 = nn.Linear(noise_size, hidden_size1)
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.linear3 = nn.Linear(hidden_size2, img_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        x = self.tanh(x)
        return x
```


```python
# Initialize generator/Discriminator
discriminator = Discriminator().to(device)
generator = Generator().to(device)
```

### Vanilla GAN 학습


```python
# Loss function & Optimizer setting
criterion = nn.BCELoss()
d_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate)
g_optimizer = optim.Adam(generator.parameters(), lr=learning_rate)
```


```python
# next(iter(data_loader))[0].shape  # torch.Size([100, 1, 28, 28])
# batch_size # 100
img_test = next(iter(data_loader))[0]
img_test.reshape(100, -1).shape
# print(len(data_loader))
```




    torch.Size([100, 784])




```python
"""
Training part
"""
for epoch in range(num_epoch):
    for i, (images, label) in enumerate(data_loader):

        # make ground truth (labels) -> 1 for real, 0 for fake
        real_label = torch.full((batch_size, 1), 1, dtype=torch.float32).to(device) # batch_size  100
        fake_label = torch.full((batch_size, 1), 0, dtype=torch.float32).to(device)

        # reshape real images from MNIST dataset
        real_images = images.reshape(batch_size, -1).to(device) # torch.Size([100, 784])

        # +---------------------+
        # |   train Generator   |
        # +---------------------+

        # Initialize grad
        g_optimizer.zero_grad()
        # d_optimizer.zero_grad()

        # make fake images with generator & noise vector 'z'
        z = torch.randn(batch_size, noise_size).to(device)
        fake_images = generator(z)
        
        # Compare result of discriminator with fake images & real labels
        # If generator deceives discriminator, g_loss will decrease
        # discriminator should be freezing
        g_loss = criterion(discriminator(fake_images), real_label)

        # Train generator with backpropagation
        g_loss.backward()
        g_optimizer.step()

        # +---------------------+
        # | train Discriminator |
        # +---------------------+

        # Initialize grad
        d_optimizer.zero_grad()
        g_optimizer.zero_grad()

        # make fake images with generator & noise vector 'z'
        z = torch.randn(batch_size, noise_size).to(device)
        fake_images = generator(z)

        # Calculate fake & real loss with generated images above & real images
        fake_loss = criterion(discriminator(fake_images), fake_label)
        real_loss = criterion(discriminator(real_images), real_label)
        d_loss = (fake_loss + real_loss) / 2

        # Train discriminator with backpropagation
        # In this part, we don't train generator
        d_loss.backward()
        d_optimizer.step()

        d_performance = discriminator(real_images).mean()
        g_performance = discriminator(fake_images).mean()

        if (i + 1) % 150 == 0:
            print("Epoch [ {}/{} ]  Step [ {}/{} ]  d_loss : {:.5f}  g_loss : {:.5f}"
                  .format(epoch, num_epoch, i+1, len(data_loader), d_loss.item(), g_loss.item()))

    # print discriminator & generator's performance
    print(" Epock {}'s discriminator performance : {:.2f}  generator performance : {:.2f}"
          .format(epoch, d_performance, g_performance))

    # Save fake images in each epoch
    samples = fake_images.reshape(batch_size, 1, 28, 28)
    save_image(samples, os.path.join(dir_name, 'GAN_fake_samples{}.png'.format(epoch + 1)))
```

## DCGAN

### DCGAN hyperparameters


```python
num_eps=10      # num_epoch
bsize=32        # batch_size 
lrate=0.001     # learning_rate
noise_size=64 # noise_size
img_size=64     # img_size
num_channel=1         # num_channel
```

### DCGAN 생성기


```python
class GANGenerator(nn.Module):
    def __init__(self):
        super(GANGenerator, self).__init__()
        self.inp_sz = img_size // 4
        self.lin = nn.Linear(noise_size, 128 * self.inp_sz ** 2)
        self.bn1 = nn.BatchNorm2d(128)
        self.up1 = nn.Upsample(scale_factor=2, mode ='nearest')
        self.cn1 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128, 0.8)
        self.rl1 = nn.LeakyReLU(0.2, inplace=True)
        self.up2 = nn.Upsample(scale_factor=2)
        self.cn2 = nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64, 0.8)
        self.rl2 = nn.LeakyReLU(0.2, inplace=True)
        self.cn3 = nn.Conv2d(64, num_channel, 3, stride=1, padding=1)
        self.act = nn.Tanh()

    def forward(self, x):
        x = self.lin(x)
        x = x.view(x.shape[0], 128, self.inp_sz, self.inp_sz)
        x = self.bn1(x)
        x = self.up1(x)
        x = self.cn1(x)
        x = self.bn2(x)
        x = self.rl1(x)
        x = self.up2(x)
        x = self.cn2(x)
        x = self.bn3(x)
        x = self.rl2(x)
        x = self.cn3(x)
        out = self.act(x)
        return out
```

### DCGAN 분류기


```python
# num_eps=10
# bsize=32
# lrate=0.001
# lat_dimension=64
# image_sz=64
# chnls=1
# logging_intv=200

class GANDiscriminator(nn.Module):
    def __init__(self):
        super(GANDiscriminator, self).__init__()

        def disc_module(ip_chnls, op_chnls, bnorm=True):
            mod = [nn.Conv2d(ip_chnls, op_chnls, 3, 2, 1), 
                   nn.LeakyReLU(0.2, inplace=True), 
                   nn.Dropout2d(0.25)]
            if bnorm:
                mod += [nn.BatchNorm2d(op_chnls, 0.8)]
            return mod

        self.disc_model = nn.Sequential(
            *disc_module(num_channel, 16, bnorm=False),
            *disc_module(16, 32),
            *disc_module(32, 64),
            *disc_module(64, 128),
        )

        # width and height of the down-sized image
        ds_size = img_size // 2 ** 4
        self.adverse_lyr = nn.Sequential(
            nn.Linear(128 * ds_size ** 2, 1), 
            nn.Sigmoid())

    def forward(self, x):
        x = self.disc_model(x)
        x = x.view(x.shape[0], -1)
        out = self.adverse_lyr(x)
        return out
```


```python
# instantiate the discriminator and generator models
gen = GANGenerator().to(device)
disc = GANDiscriminator().to(device)

# define the loss metric
adv_loss_func = torch.nn.BCELoss()
```


```python
# define the dataset and corresponding dataloader
data_loader = DataLoader(
    datasets.MNIST(
        root="./",
        download=True,
        transform=transforms.Compose(
            [transforms.Resize((img_size, img_size)), 
             transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=bsize,
    shuffle=True,
)

# define the optimization schedule for both G and D
opt_gen = optim.Adam(gen.parameters(), lr=lrate)
opt_disc = optim.Adam(disc.parameters(), lr=lrate)
```


```python
from tqdm import tqdm
os.makedirs("./DCGAN_results", exist_ok=True)

for ep in tqdm(range(num_eps)):
    for idx, (images, _) in enumerate(data_loader):

        # generate grounnd truths for real and fake images
        real_label = torch.full((images.shape[0], 1), 1, dtype = torch.float32).to(device)
        fake_label = torch.full((images.shape[0], 1), 0, dtype = torch.float32).to(device)

        # get a real image
        real_images = images.to(device)

        # train the generator model
        opt_gen.zero_grad()

        # generate a batch of images based on random noise as input
        noise = torch.randn(images.shape[0], noise_size).to(device)
        fake_images = gen(noise)

        # generator model optimization - how well can it fool the discriminator
        generator_loss = adv_loss_func(disc(fake_images), real_label)
        generator_loss.backward()
        opt_gen.step()

        # train the discriminator model
        opt_disc.zero_grad()

        # calculate discriminator loss as average of mistakes(losses) in confusing real images as fake and vice versa
        actual_image_loss = adv_loss_func(disc(real_images), real_label)
        fake_image_loss = adv_loss_func(disc(fake_images.detach()), fake_label)
        discriminator_loss = (actual_image_loss + fake_image_loss) / 2

        # discriminator model optimization
        discriminator_loss.backward()
        opt_disc.step()

        batches_completed = ep * len(dloader) + idx
        if batches_completed % 200 == 0:
            print(f"epoch number {ep} | batch number {idx} | generator loss = {generator_loss.item()} | discriminator loss = {discriminator_loss.item()}")
            save_image(fake_images.data[:25], f"DCGAN_results/{batches_completed}.png", nrow=5, normalize=True)
```

      0%|          | 0/10 [00:00<?, ?it/s]

    epoch number 0 | batch number 0 | generator loss = 3.9545516967773438 | discriminator loss = 0.2782806158065796
    epoch number 0 | batch number 200 | generator loss = 3.554192304611206 | discriminator loss = 0.03740197420120239
    epoch number 0 | batch number 400 | generator loss = 1.8775991201400757 | discriminator loss = 0.13742583990097046
    epoch number 0 | batch number 600 | generator loss = 4.771371841430664 | discriminator loss = 0.0422741174697876
    epoch number 0 | batch number 800 | generator loss = 6.1848835945129395 | discriminator loss = 0.06815836578607559
    epoch number 0 | batch number 1000 | generator loss = 6.0200371742248535 | discriminator loss = 0.012616288848221302
    epoch number 0 | batch number 1200 | generator loss = 3.3388099670410156 | discriminator loss = 0.20030230283737183
    epoch number 0 | batch number 1400 | generator loss = 3.468569755554199 | discriminator loss = 0.05986403673887253
    epoch number 0 | batch number 1600 | generator loss = 4.02747106552124 | discriminator loss = 0.15941214561462402
    epoch number 0 | batch number 1800 | generator loss = 1.9418367147445679 | discriminator loss = 0.19932261109352112


     10%|█         | 1/10 [00:49<07:22, 49.16s/it]

    epoch number 1 | batch number 125 | generator loss = 4.23289680480957 | discriminator loss = 0.04313182458281517
    epoch number 1 | batch number 325 | generator loss = 3.6431376934051514 | discriminator loss = 0.25708135962486267
    epoch number 1 | batch number 525 | generator loss = 3.278818368911743 | discriminator loss = 0.08988559246063232
    epoch number 1 | batch number 725 | generator loss = 2.4317989349365234 | discriminator loss = 0.05262760818004608
    epoch number 1 | batch number 925 | generator loss = 2.1540660858154297 | discriminator loss = 0.19819074869155884
    epoch number 1 | batch number 1125 | generator loss = 4.866243362426758 | discriminator loss = 0.07104020565748215
    epoch number 1 | batch number 1325 | generator loss = 4.817590713500977 | discriminator loss = 0.03069155663251877
    epoch number 1 | batch number 1525 | generator loss = 5.30476188659668 | discriminator loss = 0.1353522092103958
    epoch number 1 | batch number 1725 | generator loss = 3.282376766204834 | discriminator loss = 0.07071726024150848


     20%|██        | 2/10 [01:37<06:30, 48.86s/it]

    epoch number 2 | batch number 50 | generator loss = 5.153894424438477 | discriminator loss = 0.04896470159292221
    epoch number 2 | batch number 250 | generator loss = 4.530649185180664 | discriminator loss = 0.3270239531993866
    epoch number 2 | batch number 450 | generator loss = 8.000043869018555 | discriminator loss = 0.10126665979623795
    epoch number 2 | batch number 650 | generator loss = 6.198918342590332 | discriminator loss = 0.010016540065407753
    epoch number 2 | batch number 850 | generator loss = 5.983379364013672 | discriminator loss = 0.3273736536502838
    epoch number 2 | batch number 1050 | generator loss = 4.024206161499023 | discriminator loss = 0.038508810102939606
    epoch number 2 | batch number 1250 | generator loss = 0.7392721176147461 | discriminator loss = 0.15315017104148865
    epoch number 2 | batch number 1450 | generator loss = 3.7891921997070312 | discriminator loss = 0.1695229858160019
    epoch number 2 | batch number 1650 | generator loss = 4.046806335449219 | discriminator loss = 0.10749366879463196
    epoch number 2 | batch number 1850 | generator loss = 4.57669734954834 | discriminator loss = 0.12455864250659943


     30%|███       | 3/10 [02:27<05:44, 49.16s/it]

    epoch number 3 | batch number 175 | generator loss = 5.730733871459961 | discriminator loss = 0.12807787954807281
    epoch number 3 | batch number 375 | generator loss = 4.793088912963867 | discriminator loss = 0.37881994247436523
    epoch number 3 | batch number 575 | generator loss = 6.7368974685668945 | discriminator loss = 0.016468007117509842
    epoch number 3 | batch number 775 | generator loss = 5.118185997009277 | discriminator loss = 0.034300170838832855
    epoch number 3 | batch number 975 | generator loss = 5.057310581207275 | discriminator loss = 0.10720119625329971
    epoch number 3 | batch number 1175 | generator loss = 6.804218769073486 | discriminator loss = 0.008177373558282852
    epoch number 3 | batch number 1375 | generator loss = 3.3641469478607178 | discriminator loss = 0.10269354283809662
    epoch number 3 | batch number 1575 | generator loss = 5.981612205505371 | discriminator loss = 0.021350819617509842
    epoch number 3 | batch number 1775 | generator loss = 5.107820510864258 | discriminator loss = 0.2083660066127777


     40%|████      | 4/10 [03:16<04:55, 49.24s/it]

    epoch number 4 | batch number 100 | generator loss = 8.087093353271484 | discriminator loss = 0.09759678691625595
    epoch number 4 | batch number 300 | generator loss = 1.7739126682281494 | discriminator loss = 0.6854257583618164
    epoch number 4 | batch number 500 | generator loss = 8.029947280883789 | discriminator loss = 0.045721717178821564
    epoch number 4 | batch number 700 | generator loss = 3.9087915420532227 | discriminator loss = 0.16519811749458313
    epoch number 4 | batch number 900 | generator loss = 5.707446575164795 | discriminator loss = 0.08258886635303497
    epoch number 4 | batch number 1100 | generator loss = 1.5349751710891724 | discriminator loss = 0.53525710105896
    epoch number 4 | batch number 1300 | generator loss = 3.1992595195770264 | discriminator loss = 0.004317648708820343
    epoch number 4 | batch number 1500 | generator loss = 7.143947601318359 | discriminator loss = 0.0162322036921978
    epoch number 4 | batch number 1700 | generator loss = 4.835370063781738 | discriminator loss = 0.33377426862716675


     50%|█████     | 5/10 [04:12<04:18, 51.67s/it]

    epoch number 5 | batch number 25 | generator loss = 2.1899523735046387 | discriminator loss = 0.16276304423809052
    epoch number 5 | batch number 225 | generator loss = 5.253238677978516 | discriminator loss = 0.3563573360443115
    epoch number 5 | batch number 425 | generator loss = 7.449577808380127 | discriminator loss = 0.12209033966064453
    epoch number 5 | batch number 625 | generator loss = 3.859504222869873 | discriminator loss = 0.17049042880535126
    epoch number 5 | batch number 825 | generator loss = 4.514573574066162 | discriminator loss = 0.06178569048643112
    epoch number 5 | batch number 1025 | generator loss = 5.278169631958008 | discriminator loss = 0.027951447293162346
    epoch number 5 | batch number 1225 | generator loss = 6.560893535614014 | discriminator loss = 0.08444608002901077
    epoch number 5 | batch number 1425 | generator loss = 7.850395202636719 | discriminator loss = 0.25208404660224915
    epoch number 5 | batch number 1625 | generator loss = 4.836728096008301 | discriminator loss = 0.0652989074587822
    epoch number 5 | batch number 1825 | generator loss = 8.25395679473877 | discriminator loss = 0.03623552992939949


     60%|██████    | 6/10 [05:01<03:23, 50.85s/it]

    epoch number 6 | batch number 150 | generator loss = 11.474748611450195 | discriminator loss = 0.4834703803062439
    epoch number 6 | batch number 350 | generator loss = 4.103691101074219 | discriminator loss = 0.18094411492347717
    epoch number 6 | batch number 550 | generator loss = 5.099274635314941 | discriminator loss = 0.08729465305805206
    epoch number 6 | batch number 750 | generator loss = 3.8545188903808594 | discriminator loss = 0.06430787593126297
    epoch number 6 | batch number 950 | generator loss = 1.6146601438522339 | discriminator loss = 0.004101771395653486
    epoch number 6 | batch number 1150 | generator loss = 1.69037926197052 | discriminator loss = 0.0950821042060852
    epoch number 6 | batch number 1350 | generator loss = 3.2029476165771484 | discriminator loss = 0.01881629414856434
    epoch number 6 | batch number 1550 | generator loss = 6.5531840324401855 | discriminator loss = 0.05114094167947769
    epoch number 6 | batch number 1750 | generator loss = 8.155012130737305 | discriminator loss = 0.37255942821502686


     70%|███████   | 7/10 [05:51<02:31, 50.50s/it]

    epoch number 7 | batch number 75 | generator loss = 5.89101505279541 | discriminator loss = 0.029376033693552017
    epoch number 7 | batch number 275 | generator loss = 3.817337989807129 | discriminator loss = 0.02690809778869152
    epoch number 7 | batch number 475 | generator loss = 10.73667049407959 | discriminator loss = 0.1453002542257309
    epoch number 7 | batch number 675 | generator loss = 3.355811595916748 | discriminator loss = 0.22888179123401642
    epoch number 7 | batch number 875 | generator loss = 2.42566180229187 | discriminator loss = 0.4170704782009125
    epoch number 7 | batch number 1075 | generator loss = 4.84708833694458 | discriminator loss = 0.16885662078857422
    epoch number 7 | batch number 1275 | generator loss = 5.702351093292236 | discriminator loss = 0.03158178552985191
    epoch number 7 | batch number 1475 | generator loss = 6.083104133605957 | discriminator loss = 0.008439648896455765
    epoch number 7 | batch number 1675 | generator loss = 7.070625305175781 | discriminator loss = 0.25249427556991577


     80%|████████  | 8/10 [06:40<01:39, 49.86s/it]

    epoch number 8 | batch number 0 | generator loss = 4.203941822052002 | discriminator loss = 0.18343289196491241
    epoch number 8 | batch number 200 | generator loss = 2.7548460960388184 | discriminator loss = 0.017191078513860703
    epoch number 8 | batch number 400 | generator loss = 5.153848171234131 | discriminator loss = 0.2126798778772354
    epoch number 8 | batch number 600 | generator loss = 2.4082131385803223 | discriminator loss = 0.7340734004974365
    epoch number 8 | batch number 800 | generator loss = 1.9588088989257812 | discriminator loss = 0.7638360857963562
    epoch number 8 | batch number 1000 | generator loss = 3.1427531242370605 | discriminator loss = 0.04076611250638962
    epoch number 8 | batch number 1200 | generator loss = 9.426302909851074 | discriminator loss = 0.0671580359339714
    epoch number 8 | batch number 1400 | generator loss = 4.568195343017578 | discriminator loss = 0.2781957983970642
    epoch number 8 | batch number 1600 | generator loss = 3.1912214756011963 | discriminator loss = 0.04164225980639458
    epoch number 8 | batch number 1800 | generator loss = 4.669330596923828 | discriminator loss = 0.0010130235459655523


     90%|█████████ | 9/10 [07:28<00:49, 49.38s/it]

    epoch number 9 | batch number 125 | generator loss = 2.847745895385742 | discriminator loss = 0.2367047667503357
    epoch number 9 | batch number 325 | generator loss = 8.497081756591797 | discriminator loss = 0.6592368483543396
    epoch number 9 | batch number 525 | generator loss = 5.124856948852539 | discriminator loss = 0.05494079366326332
    epoch number 9 | batch number 725 | generator loss = 2.808432102203369 | discriminator loss = 0.08962322771549225
    epoch number 9 | batch number 925 | generator loss = 3.5111680030822754 | discriminator loss = 0.177422434091568
    epoch number 9 | batch number 1125 | generator loss = 4.133440017700195 | discriminator loss = 0.12160167098045349
    epoch number 9 | batch number 1325 | generator loss = 6.491473197937012 | discriminator loss = 0.011363385245203972
    epoch number 9 | batch number 1525 | generator loss = 1.895836591720581 | discriminator loss = 0.25524279475212097
    epoch number 9 | batch number 1725 | generator loss = 3.693697690963745 | discriminator loss = 0.03281675651669502


    100%|██████████| 10/10 [08:16<00:00, 49.69s/it]

