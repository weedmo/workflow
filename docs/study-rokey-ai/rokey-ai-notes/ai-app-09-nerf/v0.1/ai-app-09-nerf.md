# 9장 NeRF

> **Migrated from Quartz wiki** (`wiki/study/rokey-ai/intermediate/ai-app-09-nerf.md`, updated 2026-09-26)  
> tags: rokey, ai, notebook, intermediate, study  
> source: raw/2026-09-26-rokey-ai/intermediate/강의_3기_AI응용_9차시__NeRF_.ipynb  
> source: https://github.com/weedmo/Rokey-AI  
This is a PyTorch implementation based on the paper: https://arxiv.org/abs/2003.08934. The code takes mostly after the officially tiny nerf implementation: https://colab.research.google.com/github/bmild/nerf/blob/master/tiny_nerf.ipynb

*(이미지 없음 / image not preserved: image omitted)*

## Imports + Data



```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.init as init
import time
import numpy as np
import torch.nn.functional as F
import os
from tqdm import tqdm
from ipywidgets import interactive, widgets


device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
# if not os.path.exists('tiny_nerf_data.npz'):
#     !wget http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/tiny_nerf_data.npz
```

## Encoding + Model

*(이미지 없음 / image not preserved: image omitted)*

$$\gamma(p) = (\sin(2^0\pi p), \cos(2^0\pi p), \dots, \sin(2^{L-1}\pi p), \cos(2^{L-1}\pi p))$$


```python
def encoding(x, L=10):
  res = [x]
  for i in range(L):
    for fn in [torch.sin, torch.cos]:
      res.append(fn(2 ** i * torch.pi * x))
  return torch.cat(res,dim=-1)
```


```python
x = torch.Tensor([3.1,5.6,7.3]) # x, y, z
y = encoding(x,L=4)
y
```




    tensor([ 3.1000,  5.6000,  7.3000, -0.3090, -0.9511, -0.8090, -0.9511,  0.3090,
            -0.5878,  0.5878, -0.5878,  0.9511,  0.8090, -0.8090, -0.3090,  0.9511,
             0.9511, -0.5878,  0.3090,  0.3090, -0.8090,  0.5878,  0.5878,  0.9511,
            -0.8090, -0.8090,  0.3090])




```python
class NeRF(nn.Module):
  def __init__(self, pos_enc_dim=63, view_enc_dim=27, hidden=256) -> None:
     super().__init__()

     self.linear1 = nn.Sequential(nn.Linear(pos_enc_dim,hidden),nn.ReLU())

     self.pre_skip_linear = nn.Sequential()
     for _ in range(4):
      self.pre_skip_linear.append(nn.Linear(hidden,hidden))
      self.pre_skip_linear.append(nn.ReLU())

     self.linear_skip = nn.Sequential(nn.Linear(pos_enc_dim+hidden,hidden),nn.ReLU())

     self.post_skip_linear = nn.Sequential()
     for _ in range(2):
      self.post_skip_linear.append(nn.Linear(hidden,hidden))
      self.post_skip_linear.append(nn.ReLU())

     self.density_layer = nn.Sequential(nn.Linear(hidden,1),nn.ReLU())

     self.linear2 = nn.Linear(hidden,hidden)

     self.color_linear1 = nn.Sequential(nn.Linear(hidden+view_enc_dim,hidden//2),nn.ReLU())
     self.color_linear2 = nn.Sequential(nn.Linear(hidden//2,3), nn.Sigmoid())

  def forward(self, input):

    positions = input[...,:3]
    view_dirs = input[...,3:]

    # Encode
    pos_enc = encoding(positions,L=10)
    view_enc = encoding(view_dirs,L=4)

    x = self.linear1(pos_enc)
    x = self.pre_skip_linear(x)

    # Skip connection
    x = torch.cat([x,pos_enc],dim=-1)
    x = self.linear_skip(x)

    x = self.post_skip_linear(x)

    # Density
    sigma = self.density_layer(x)

    x = self.linear2(x)

    # View Encoding
    x = torch.cat([x,view_enc],dim=-1)
    x = self.color_linear1(x)

    # Color Prediction
    rgb = self.color_linear2(x)

    return torch.cat([sigma,rgb],dim=-1)

```

## Get Ray + Render


```python
## Meshgrid
W = 3; H = 4; focal = 1
i, j = torch.meshgrid(
      torch.arange(W, dtype=torch.float32, device=device),
      torch.arange(H, dtype=torch.float32, device=device),
      indexing='xy' # ensures that i represents the width and j represents the height, following standard image conventions.
  )

print("i = \n",  i)
print("j = \n",  j)
print('z = ')
print(torch.ones_like(i))

print("+"*50)
print("i = \n",  i)
print("-j = \n",  -j)
print('-z = ')
print(-torch.ones_like(i))

```

    i = 
     tensor([[0., 1., 2.],
            [0., 1., 2.],
            [0., 1., 2.],
            [0., 1., 2.]], device='cuda:0')
    j = 
     tensor([[0., 0., 0.],
            [1., 1., 1.],
            [2., 2., 2.],
            [3., 3., 3.]], device='cuda:0')
    z = 
    tensor([[1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.]], device='cuda:0')
    ++++++++++++++++++++++++++++++++++++++++++++++++++
    i = 
     tensor([[0., 1., 2.],
            [0., 1., 2.],
            [0., 1., 2.],
            [0., 1., 2.]], device='cuda:0')
    -j = 
     tensor([[-0., -0., -0.],
            [-1., -1., -1.],
            [-2., -2., -2.],
            [-3., -3., -3.]], device='cuda:0')
    -z = 
    tensor([[-1., -1., -1.],
            [-1., -1., -1.],
            [-1., -1., -1.],
            [-1., -1., -1.]], device='cuda:0')



```python
## image centering

dirs = torch.stack(
      [(i - W * .5) / focal, -(j - H * .5) / focal, -torch.ones_like(i, device = device)], -1
  )

# dirs = torch.stack(
#       [i, -j, -torch.ones_like(i, device = device)], -1
#   )

print(dirs)
print(dirs.shape)
dirs[..., None, :].shape
```

    
    tensor([[[-1.5000,  2.0000, -1.0000],
             [-0.5000,  2.0000, -1.0000],
             [ 0.5000,  2.0000, -1.0000]],
    
            [[-1.5000,  1.0000, -1.0000],
             [-0.5000,  1.0000, -1.0000],
             [ 0.5000,  1.0000, -1.0000]],
    
            [[-1.5000, -0.0000, -1.0000],
             [-0.5000, -0.0000, -1.0000],
             [ 0.5000, -0.0000, -1.0000]],
    
            [[-1.5000, -1.0000, -1.0000],
             [-0.5000, -1.0000, -1.0000],
             [ 0.5000, -1.0000, -1.0000]]], device='cuda:0')
    torch.Size([4, 3, 3])





    torch.Size([4, 3, 1, 3])




```python
## mat1*mat2
mat1 = torch.tensor([[1, 2, 3]])
mat2 = torch.full((3, 3), 10)
print(mat1)
print(mat2)
print(mat1*mat2)
print(torch.sum(mat1*mat2, -1))
```

    tensor([[1, 2, 3]])
    tensor([[10, 10, 10],
            [10, 10, 10],
            [10, 10, 10]])
    tensor([[10, 20, 30],
            [10, 20, 30],
            [10, 20, 30]])
    tensor([60, 60, 60])



```python
## torch.from_numpy
a =  np.array([1, 2, 4])

b = torch.tensor(a)
c = torch.from_numpy(a)


a[0] = 10

print(a)
print(b)
print(c)
```

    [10  2  4]
    tensor([1, 2, 4])
    tensor([10,  2,  4])



```python
## rand
near = 2; far = 6; N_samples = 6
z_vals = torch.linspace(near, far, steps=N_samples, device=device)
print("z_vals = \n", z_vals)

# torch.rand(*z_vals.shape[:-1], N_samples, device=device) * (far - near) / N_samples
torch.rand(*z_vals.shape[:-1], N_samples, device=device) * (far - near) / N_samples

```

    z_vals = 
     tensor([2.0000, 2.8000, 3.6000, 4.4000, 5.2000, 6.0000], device='cuda:0')





    tensor([0.0192, 0.0702, 0.6064, 0.1827, 0.3709, 0.2194], device='cuda:0')




```python
z_vals.shape[:-1]

scalar_val = torch.tensor(0)
print(scalar_val, scalar_val.shape)
```

    tensor(0) torch.Size([])



```python
## torch.cumprod
a = torch.arange(1, 4)
print('a = ', a)
print(torch.cumprod(a, dim = -1) )
```

    a =  tensor([1, 2, 3])
    tensor([1, 2, 6])



```python
def get_rays(H, W, focal, c2w):
  """
  Generate rays for a given camera configuration.

  Args:
    H: Image height.
    W: Image width.
    focal: Focal length.
    c2w: Camera-to-world transformation matrix (4x4).

  Returns:
    rays_o: Ray origins (H*W, 3).
    rays_d: Ray directions (H*W, 3).
  """
  device = c2w.device  # Get the device of c2w
  focal = torch.from_numpy(focal).to(device) # call by reference
  # print(type(H), type(W), type(focal), type(c2w))

  i, j = torch.meshgrid(
      torch.arange(W, dtype=torch.float32, device=device),
      torch.arange(H, dtype=torch.float32, device=device),
      indexing='xy' # ensures that i represents the width and j represents the height, following standard image conventions.
  )
  dirs = torch.stack(
      [(i - W * .5) / focal, -(j - H * .5) / focal, -torch.ones_like(i, device = device)], -1
  )
 # dirs = [H, W, 1, (x, y, z)] * c2w[:3, :3] (# Intrinsic matrix)
  rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], -1)  # a cone beam model,forming a perspective projection
  # rays_d = torch.Size([100, 100, 3])
  rays_d = rays_d.view(-1, 3)
  # rays_d = torch.Size([10000, 3])
  rays_o = c2w[:3, -1].expand(rays_d.shape)
  # rays_o = torch.Size([10000, 3])
  return rays_o, rays_d

# rgb, depth, acc = render_rays(model, rays_o, rays_d, near=2., far=6., N_samples=n_samples, device=device, rand=True)
def render_rays(network_fn, rays_o, rays_d, near, far, N_samples, device, rand=False, embed_fn=None, chunk=1024*4):
    # This function processes large inputs in chunks to prevent memory overload when calling
    def batchify(fn, chunk):
        return lambda inputs: torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0) # [0, 4096, 8192]

    # Sampling
    z_vals = torch.linspace(near, far, steps=N_samples, device=device)

    if rand:
        z_vals += torch.rand(*z_vals.shape[:-1], N_samples, device=rays_o.device) * (far - near) / N_samples
    # z_vals shape = torch.Size([64])
   
    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # z_vals shape = torch.Size([64, 1])
    # pts.shape = torch.Size([10000, 64, 3])

    # Normalize view directions
    view_dirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    view_dirs = view_dirs[..., None, :].expand(pts.shape)

    input_pts = torch.cat((pts, view_dirs), dim=-1)
    raw = batchify(network_fn, chunk)(input_pts)

    # Apply activations here instead of in network
    sigma_a = raw[...,0]  # Shape: [batch, N_samples]
    rgb = raw[...,1:]    # Shape: [batch, N_samples, 3]

    # Improved volume rendering
    dists = z_vals[..., 1:] - z_vals[..., :-1]  # Shape: [batch, N_samples-1]
    dists = torch.cat([dists, torch.tensor([1e10], device=device)], -1)

    # No need to manually expand dists as broadcasting will handle it
    alpha = 1. - torch.exp(-sigma_a * dists)  # Shape: [batch, N_samples] , cdf of exponetial
    alpha = alpha.unsqueeze(-1)  # Shape: [batch, N_samples, 1] 

    ## Computing transmittance: Transmittance (T) represents how much light reaches each point.
    ones_shape = (alpha.shape[0], 1, 1)
    T = torch.cumprod(
        torch.cat([
            torch.ones(ones_shape, device=device),  # (10000, 1, 1)
            1. - alpha + 1e-10                      # (10000, 64, 1)
        ], dim=1),
        dim=1
    )[:, :-1]  # Shape: [batch, N_samples, 1]

    weights = alpha * T  # Shape: [batch, N_samples, 1]

    # Compute final colors and depths, accumulation
    rgb_map = torch.sum(weights * rgb, dim=1)  # Sum along sample dimension
    # rgb_map.shape = torch.Size([10000, 3])
    depth_map = torch.sum(weights.squeeze(-1) * z_vals, dim=-1)  # Shape: [batch]
    acc_map = torch.sum(weights.squeeze(-1), dim=-1)  # Shape: [batch]

    return rgb_map, depth_map, acc_map
```

## Train Loop


```python
def train(images,poses,H,W,focal,testpose,testimg,device):

    print(f"Using device: {device}")
    model = NeRF().to(device)

    criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(),lr=5e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99) # decays the learning rate (LR) by multiplying it by gamma=0.99 after each epoch
 
    n_iter = 1000
    n_samples = 64
    i_plot = 50
    psnrs = []
    iternums = []
    t = time.time()

    # Convert data to tensors and move to device ONCE
    images_tensor = torch.from_numpy(images).float().to(device)
    poses_tensor = torch.from_numpy(poses).float().to(device)

    for i in range(n_iter):

        img_i = np.random.randint(images.shape[0])

        target = images_tensor[img_i]  # Use the corresponding image
        pose = poses_tensor[img_i]     # Use the corresponding pose

        rays_o, rays_d = get_rays(H, W, focal, pose)

        optimizer.zero_grad()

        rgb, depth, acc = render_rays(model, rays_o, rays_d, near=2., far=6., N_samples=n_samples, device=device, rand=True)

        rgb = rgb.reshape(H,W,3)

        loss = criterion(rgb, target)

        loss.backward()
        optimizer.step()

        if i % i_plot == 0:
            print(f'Iteration: {i}, Loss: {loss.item():.6f}, Time: {(time.time() - t) / i_plot:.2f} secs per iter')
            t = time.time()

            with torch.no_grad():
                rays_o, rays_d = get_rays(H, W, focal, testpose)
                rgb, depth, acc = render_rays(model, rays_o, rays_d, near=2., far=6.,
                                           N_samples=n_samples, device=device)
                rgb = rgb.reshape(H, W, 3)
                loss = criterion(rgb, testimg)
                psnr = -10. * torch.log10(loss)

                psnrs.append(psnr.item())
                iternums.append(i)

                plt.figure(figsize=(10,4))
                plt.subplot(121)
                plt.imshow(rgb.cpu().detach())
                plt.title(f'Iteration: {i}')
                plt.subplot(122)
                plt.plot(iternums, psnrs)
                plt.title('PSNR')
                plt.show()

    return model
```

## Load Data


```python
file_path = os.path.join(os.getcwd(),'data', 'tiny_nerf_data.npz')
data = np.load(file_path)
images = data['images']
poses = data['poses']
focal = data['focal']
H, W = images.shape[1:3]
print("image shape = ", images.shape)
print("pose shape = ", poses.shape)
print("focal length = ", focal)

device = "cuda" if torch.cuda.is_available() else "cpu"

testimg, testpose = images[101], poses[101]
images = images[:100,...,:3]
poses = poses[:100]
plt.imshow(testimg)
plt.show()
testimg =  torch.from_numpy(testimg).float().to("cuda")
testpose = torch.from_numpy(testpose).float().to("cuda")
```

    image shape =  (106, 100, 100, 3)
    pose shape =  (106, 4, 4)
    focal length =  138.88887889922103



    
*(이미지 없음 / image not preserved: output_22_1.png)*
    


## Train


```python
model = train(images,poses,H,W,focal,testpose,testimg,device)
```

    Using device: cuda
    Iteration: 0, Loss: 0.081648, Time: 0.02 secs per iter



    
*(이미지 없음 / image not preserved: output_24_1.png)*
    


    Iteration: 50, Loss: 0.054801, Time: 0.81 secs per iter



    
*(이미지 없음 / image not preserved: output_24_3.png)*
    


    Iteration: 100, Loss: 0.101059, Time: 0.80 secs per iter



    
*(이미지 없음 / image not preserved: output_24_5.png)*
    


    Iteration: 150, Loss: 0.052301, Time: 0.80 secs per iter



    
*(이미지 없음 / image not preserved: output_24_7.png)*
    


    Iteration: 200, Loss: 0.042073, Time: 0.80 secs per iter



    
*(이미지 없음 / image not preserved: output_24_9.png)*
    


    Iteration: 250, Loss: 0.021877, Time: 0.78 secs per iter



    
*(이미지 없음 / image not preserved: output_24_11.png)*
    


    Iteration: 300, Loss: 0.017164, Time: 0.78 secs per iter



    
*(이미지 없음 / image not preserved: output_24_13.png)*
    


    Iteration: 350, Loss: 0.015522, Time: 0.77 secs per iter



    
*(이미지 없음 / image not preserved: output_24_15.png)*
    


    Iteration: 400, Loss: 0.015200, Time: 0.77 secs per iter



    
*(이미지 없음 / image not preserved: output_24_17.png)*
    


    Iteration: 450, Loss: 0.017326, Time: 0.77 secs per iter



    
*(이미지 없음 / image not preserved: output_24_19.png)*
    


    Iteration: 500, Loss: 0.010243, Time: 0.77 secs per iter



    
*(이미지 없음 / image not preserved: output_24_21.png)*
    


    Iteration: 550, Loss: 0.004872, Time: 0.77 secs per iter



    
*(이미지 없음 / image not preserved: output_24_23.png)*
    


    Iteration: 600, Loss: 0.007803, Time: 0.77 secs per iter



    
*(이미지 없음 / image not preserved: output_24_25.png)*
    


    Iteration: 650, Loss: 0.009860, Time: 0.76 secs per iter



    
*(이미지 없음 / image not preserved: output_24_27.png)*
    


    Iteration: 700, Loss: 0.005969, Time: 0.76 secs per iter



    
*(이미지 없음 / image not preserved: output_24_29.png)*
    


    Iteration: 750, Loss: 0.010574, Time: 0.76 secs per iter



    
*(이미지 없음 / image not preserved: output_24_31.png)*
    


    Iteration: 800, Loss: 0.007468, Time: 0.76 secs per iter



    
*(이미지 없음 / image not preserved: output_24_33.png)*
    


    Iteration: 850, Loss: 0.010832, Time: 0.76 secs per iter



    
*(이미지 없음 / image not preserved: output_24_35.png)*
    


    Iteration: 900, Loss: 0.010579, Time: 0.76 secs per iter



    
*(이미지 없음 / image not preserved: output_24_37.png)*
    


    Iteration: 950, Loss: 0.007801, Time: 0.76 secs per iter



    
*(이미지 없음 / image not preserved: output_24_39.png)*
    


## Render Video


```python
# Transformation matrices in PyTorch
trans_t = lambda t: torch.tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, t],
    [0, 0, 0, 1]
], dtype=torch.float32)

rot_phi = lambda phi: torch.tensor([
    [1, 0, 0, 0],
    [0, torch.cos(phi), -torch.sin(phi), 0],
    [0, torch.sin(phi), torch.cos(phi), 0],
    [0, 0, 0, 1]
], dtype=torch.float32)

rot_theta = lambda th: torch.tensor([
    [torch.cos(th), 0, -torch.sin(th), 0],
    [0, 1, 0, 0],
    [torch.sin(th), 0, torch.cos(th), 0],
    [0, 0, 0, 1]
], dtype=torch.float32)

# Pose function with spherical coordinates
def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = torch.matmul(rot_phi(torch.Tensor([phi / 180. * np.pi])), c2w)
    c2w = torch.matmul(rot_theta(torch.Tensor([theta / 180. * np.pi])), c2w)
    c2w = torch.tensor([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=torch.float32) @ c2w
    return c2w

# Function for rendering based on user input
def f(**kwargs):
    c2w = pose_spherical(**kwargs)
    rays_o, rays_d = get_rays(H, W, focal, c2w[:3, :4])  # Get rays (this is a placeholder)
    c2w, rays_o, rays_d = map(lambda t: t.to(device), (c2w, rays_o, rays_d))
    with torch.no_grad():
      rgb, depth, acc = render_rays(model, rays_o, rays_d, near=2., far=6., N_samples=64, device=device)  # Render rays
    rgb = rgb.reshape(H, W, 3).cpu().detach()
    img = torch.clamp(rgb, 0, 1).numpy()  # Clamp RGB values between 0 and 1 and convert to numpy

    plt.figure(2, figsize=(20, 6))
    plt.imshow(img)
    plt.show()

# Interactive slider setup for theta, phi, and radius
sldr = lambda v, mi, ma: widgets.FloatSlider(
    value=v,
    min=mi,
    max=ma,
    step=.01,
)

names = [
    ['theta', [100., 0., 360]],
    ['phi', [-30., -90, 0]],
    ['radius', [4., 3., 5.]],
]

interactive_plot = interactive(f, **{s[0]: sldr(*s[1]) for s in names})
output = interactive_plot.children[-1]
output.layout.height = '350px'
interactive_plot
```


    interactive(children=(FloatSlider(value=100.0, description='theta', max=360.0, step=0.01), FloatSlider(value=-…



```python
frames = []
for th in tqdm(np.linspace(0., 360., 120, endpoint=False)):
    c2w = pose_spherical(th, -30., 4.)
    rays_o, rays_d = get_rays(H, W, focal, c2w[:3,:4])
    c2w, rays_o, rays_d = map(lambda t: t.to(device), (c2w, rays_o, rays_d))
    with torch.no_grad():
      rgb, depth, acc = render_rays(model, rays_o, rays_d, near=2., far=6., N_samples=64, device = device)
    rgb = rgb.reshape(H, W, 3)
    frames.append((255*np.clip(rgb.cpu().detach().numpy(),0,1)).astype(np.uint8))

import imageio
f = 'video.mp4'
imageio.mimwrite(f, frames, fps=30, quality=7)
```

    100%|██████████| 120/120 [00:36<00:00,  3.29it/s]
    WARNING:imageio_ffmpeg:IMAGEIO FFMPEG_WRITER WARNING: input image is not divisible by macro_block_size=16, resizing from (100, 100) to (112, 112) to ensure video compatibility with most codecs and players. To prevent resizing, make your input image divisible by the macro_block_size or set the macro_block_size to 1 (risking incompatibility).



```python
from IPython.display import HTML
from base64 import b64encode
mp4 = open('video.mp4','rb').read()
data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
HTML("""
<video width=400 controls autoplay loop>
      <source src="%s" type="video/mp4">
</video>
""" % data_url)
```





<video width=400 controls autoplay loop>
      <source src="data:video/mp4;base64,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" type="video/mp4">
</video>



