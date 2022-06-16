import numpy as np
import torch

class GaussianDiffusion():
    '''Gaussian Diffusion process'''
    
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
    
    def inverse(self, net, shape=(1,32,32), x0=None, start_t=None, steps=None, device='cpu'):
        if x0 is None:
            x = torch.randn((1,) + shape)
        else:
            x = x0
        if start_t is None:
            start_t = self.T
        if steps is None:
            steps = self.T
        
        for t in range(start_t, start_t-steps, -1):
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
                pred = net(x.float().to(device), t.float().to(device))
                pred = pred.cpu()

            x = (1 / np.sqrt(at)) * (x - ((1-at) / np.sqrt(1-atbar)) * pred) + np.sqrt(beta_tilde) * z

        return x
