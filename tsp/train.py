import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import math

from unet import UNetModel

import tqdm
import matplotlib.pyplot as plt

class TSPDataset(torch.utils.data.Dataset):
    def __init__(self, data_file, img_size, point_radius=2, point_color=1, point_circle=True, line_thickness=2, line_color=0.5, max_points=100):
        self.data_file = data_file
        self.img_size = img_size
        self.point_radius = point_radius
        self.point_color = point_color
        self.point_circle = point_circle
        self.line_thickness = line_thickness
        self.line_color = line_color
        self.max_points = max_points
        
        self.file_lines = open(data_file).read().splitlines()
        print(f'Loaded "{data_file}" with {len(self.file_lines)} lines')
        
    def __len__(self):
        return len(self.file_lines)
    
    def rasterize(self, idx):
        # Select sample
        line = self.file_lines[idx]
        # Clear leading/trailing characters
        line = line.strip()

        # Extract points
        points = line.split(' output ')[0]
        points = points.split(' ')
        points = np.array([[float(points[i]), float(points[i+1])] for i in range(0,len(points),2)])
        # Extract tour
        tour = line.split(' output ')[1]
        tour = tour.split(' ')
        tour = np.array([int(t) for t in tour])
        
        # Rasterize lines
        img = np.zeros((self.img_size, self.img_size))
        for i in range(tour.shape[0]-1):
            from_idx = tour[i]-1
            to_idx = tour[i+1]-1

            cv2.line(img, 
                     tuple(((img_size-1)*points[from_idx,::-1]).astype(int)), 
                     tuple(((img_size-1)*points[to_idx,::-1]).astype(int)), 
                     color=self.line_color, thickness=self.line_thickness)

        # Rasterize points
        for i in range(points.shape[0]):
            if self.point_circle:
                cv2.circle(img, tuple(((img_size-1)*points[i,::-1]).astype(int)), 
                           radius=self.point_radius, color=self.point_color, thickness=-1)
            else:
                row = round((img_size-1)*points[i,0])
                col = round((img_size-1)*points[i,1])
                img[row,col] = self.point_color
            
        # Rescale image to [-1,1]
        img = 2*(img-0.5)
            
        return img, points, tour

    def __getitem__(self, idx):
        img, points, tour = self.rasterize(idx)
            
        return img[np.newaxis,:,:], idx

device = torch.device('cuda:0')
batch_size = 16
img_size = 64

train_dataset = TSPDataset(data_file='data/tsp50_train_concorde.txt',
                           img_size=img_size)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
print('Created dataset')
print(len(train_dataset))

class GaussianDiffusion():
    '''Gaussian Diffusion process with linear beta scheduling'''
    def __init__(self, T, schedule):
        # Diffusion steps
        self.T = T
    
        # Noise schedule
        if schedule == 'linear':
            b0=1e-4
            bT=2e-2
            self.beta = np.linspace(b0, bT, T)
        elif schedule == 'cosine':
            self.alphabar = self.__cos_noise(np.arange(0, T+1, 1)) / self.__cos_noise(0) # Generate an extra alpha for bT
            self.beta = np.clip(1 - (self.alphabar[1:] / self.alphabar[:-1]), None, 0.999)
            
        self.betabar = np.cumprod(self.beta)
        self.alpha = 1 - self.beta
        self.alphabar = np.cumprod(self.alpha)

    def __cos_noise(self, t):
        offset = 0.008
        return np.cos(math.pi * 0.5 * (t/self.T + offset) / (1+offset)) ** 2
   
    def sample(self, x0, t):        
        # Select noise scales
        noise_dims = (x0.shape[0],) + tuple((1 for _ in x0.shape[1:]))        
        atbar = torch.from_numpy(self.alphabar[t-1]).view(noise_dims).to(x0.device)
        assert len(atbar.shape) == len(x0.shape), 'Shape mismatch'
        
        # Sample noise and add to x0
        epsilon = torch.randn_like(x0)
        xt = torch.sqrt(atbar) * x0 + torch.sqrt(1-atbar) * epsilon        
        return xt, epsilon
    
    def inverse(self, net, shape=(1,32,32), steps=None, x=None):
        if steps is None:
            steps = self.T
            
        # Generate sample starting from xT
        if x is None:
            x = torch.randn((1,) + shape)

        for t in range(self.T, self.T-steps, -1):
            at = self.alpha[t-1]
            atbar = self.alphabar[t-1]
            
            if t > 1:
                z = torch.randn_like(x)
                atbar_prev = self.alphabar[t-2]
                beta_tilde = self.beta[t-1] * (1 - atbar_prev) / (1 - atbar) 
            else:
                z = torch.zeros_like(x)
                beta_tilde = 0

            with torch.no_grad():
                t = torch.tensor([t]).view(1)
                pred = net(x.float().to(device), t.float().to(device)).cpu()

            x = (1 / np.sqrt(at)) * (x - ((1-at) / np.sqrt(1-atbar)) * pred) + np.sqrt(beta_tilde) * z

        return x    

# Train diffusion model
diffusion = GaussianDiffusion(T=1000, schedule='linear')
net = UNetModel(image_size=img_size, in_channels=1, out_channels=1, 
                model_channels=64, num_res_blocks=2, channel_mult=(1,2,3,4),
                attention_resolutions=[16,8], num_heads=4).to(device)
print('Parameters:', sum([p.numel() for p in net.parameters()]))
net.train()
opt = torch.optim.Adam(list(net.parameters()), lr=1e-4)

epochs = 8
update_every = 20
losses = []
for e in range(epochs):
    print(f'Epoch [{e+1:02d}/{epochs:02d}]')
    
    for i, batch in enumerate(train_dataloader):
        # Unwrap batch
        image, _ = batch
        
        # Sample from diffusion
        t = np.random.randint(1, diffusion.T+1, image.shape[0]).astype(int)
        xt, epsilon = diffusion.sample(image, t)
        t = torch.from_numpy(t).float().view(image.shape[0])
        # Denoise
        epsilon_pred = net(xt.float().to(device), t.float().to(device))

        # Compute loss   
        loss = F.mse_loss(epsilon_pred, epsilon.float().to(device))
        # Update parameters
        opt.zero_grad()
        loss.backward()
        opt.step()

        losses.append(loss.item())
        if (i+1) % update_every == 0 or i == 0:
            print(i, np.mean(losses))
            losses = []
    torch.save(net.state_dict(), f'models/unet50_64_{e+1}.pth')
    
    
