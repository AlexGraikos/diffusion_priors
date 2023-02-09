import torch as T
from tqdm import trange

def local_avg(data, radius, stride=1):
    if radius==0: return data
    w,h = data.shape[-2:]
    mean = T.nn.functional.avg_pool2d(data.reshape(-1,1,w,h), 2*radius+1, stride=stride, padding=radius, count_include_pad=False)
    if stride>1: mean = T.nn.functional.interpolate(mean, size=(w,h), mode='bilinear')
    return mean.view(data.shape)

def local_moments(data, q, radius, stride=1, var_min=0.0001, mq_min=0.000001):
    mq = local_avg(q, radius, stride)
    mq.clamp(min=mq_min)
    weighted = T.einsum('zij,cij->czij', data, q) #class,channel,x,y
    weighted_sq = T.einsum('zij,cij->czij', data**2, q)
    mean = local_avg(weighted, radius, stride) / mq.unsqueeze(1)
    var = local_avg(weighted_sq, radius, stride) / mq.unsqueeze(1) - mean**2
    var = var.clamp(min=var_min)
    return mean, var

def lp_gaussian(data, mean, var, radius, stride=1):
    #means: c,ch,x,y
    #data: ch,x,y
    #out: c,x,y
    
    m0 = -local_avg(1 / var, radius, stride)
    m1 = local_avg(2 * mean / var, radius, stride)
    m2 = -local_avg(mean**2 / var, radius, stride)
    L = local_avg(T.log(var), radius, stride)
    return (m0*data**2 + m1*data + m2 - 1 * L).sum(1) / 2

def prob_gaussian(data, prior, mean, var, radius, stride=1):
    lp = lp_gaussian(data, mean, var, radius, stride)
    p = lp.softmax(0) * prior
    p /= p.sum(0)
    p+=0.001
    p /= p.sum(0)
    return p

def em(data, p, radius, stride=1):
    prior = local_avg(p, radius, stride)
    mean, var = local_moments(data, p, radius, stride)
    p_new = prob_gaussian(data, prior, mean, var, radius, stride)
    return p_new, mean, var, prior

def run_clustering(image, n_classes, radius, n_iter, stride, warmup_steps, warmup_radius, device):
    
    # Why?
    #T.set_grad_enabled(False)
    
    with T.no_grad():
        data = T.tensor(image).to(device)
        p = T.rand((n_classes,) + image.shape[1:], dtype=T.double).to(device)
        p /= p.sum(0)

        for i in trange(n_iter):
            p.mean().item() # trigger synchronization
            p, mean, var, prior = em(data, p, warmup_radius if i<warmup_steps else radius, stride)# stride if i<n_iter-1 else 1)

    p_ = p.cpu().numpy()
    mean_ = mean.cpu().numpy()
    var_ = var.cpu().numpy()
    prior_ = prior.cpu().numpy()

    return p_, mean_, var_, prior_