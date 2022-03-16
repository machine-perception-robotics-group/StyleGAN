import os 
import sys
import copy
import yaml
import shutil
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import datasets, transforms

from config import config
from models import Generator, Discriminator
from dataset import CelebA_hq_DataLoader, FFHQ_DataLoader

conf = config()

if os.path.exists(conf.log_dir):
    shutil.rmtree(conf.log_dir)

tb = SummaryWriter(log_dir=conf.log_dir)

if not os.path.exists(conf.result_dir):
    os.makedirs(conf.result_dir)

device = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")
G = nn.DataParallel(
    Generator(switch_ch=conf.switch_timing, 
              img_size=conf.last_img_size, 
              n_ch=conf.n_ch_g, 
              n_style=conf.n_style, 
              n_mapping_layers=conf.n_mapping_network).to(device), 
    device_ids=conf.gpu)

G_ema = copy.deepcopy(G.module).cpu()
G_ema.eval()
for p in G_ema.parameters():
    p.requires_grad_(False)

D = nn.DataParallel(
    Discriminator(switch_ch=conf.switch_timing, 
                  img_size=conf.last_img_size, 
                  n_ch=conf.n_ch_d).to(device), 
    device_ids=conf.gpu)

print(G)
print('=' * 50)
print(D)
print('=' * 50)

with open(conf.yaml_path, 'r+') as f:
    training_params = yaml.load(f)
batch_size = training_params['batch_size']
lr = training_params['learning_rate']
epochs = training_params['epochs']
deltas = training_params['delta']
G_opt = optim.Adam([
        {"params": G.module.mapping_network.parameters(), "lr": 0.00001},
        {"params": G.module.style_generator.parameters()}], lr=0.001, betas=(conf.beta1, conf.beta2), eps=conf.eps)
#G_opt = optim.Adam(G.parameters(), lr=0.001, betas=(conf.beta1, conf.beta2), eps=conf.eps)
D_opt = optim.Adam(D.parameters(), lr=0.001, betas=(conf.beta1, conf.beta2), eps=conf.eps)
criterion = nn.BCEWithLogitsLoss()

def update_EMA(src, tgt, strength=0.999):
    with torch.no_grad():
        paramnames = dict(src.module.named_parameters())
        for k, v in tgt.named_parameters():
            param = paramnames[k].detach().cpu()
            v.copy_(strength * v + (1 - strength) * param)

def gradient_penalty(real_img, fake_img, alpha, d):           
    eps = torch.rand(real_img.size(0), 1, 1, 1).cuda().expand_as(real_img)
    interpolated = torch.autograd.Variable(eps * real_img.data + (1 - eps) * fake_img.data, 
                                           requires_grad=True)
    out = D(interpolated, d, alpha=alpha)

    grad = torch.autograd.grad(outputs=out,
                               inputs=interpolated,
                               grad_outputs=torch.ones(out.size()).to(device),
                               retain_graph=True,
                               create_graph=True,
                               only_inputs=True)[0]

    grad = grad.view(grad.size(0), -1)
    grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
    d_loss_gp = torch.mean((grad_l2norm - 1) ** 2)
    
    return d_loss_gp
    
for d in range(conf.depth):
    print('='*50, 'Now network depth is %d' % d, '='*50)
    size = 4 * 2 ** d
    alpha = conf.alpha
    iteration = 0
    lr_val = lr['%dx%d' % (size, size)]
    epoch = epochs['%dx%d' % (size, size)]
    batch = batch_size['%dx%d' % (size, size)]
    transform_train = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])
    #training_data = CelebA_hq_DataLoader(size, conf.h5py_path, conf.data_path, conf.txt_path, transform=transform_train)
    training_data = FFHQ_DataLoader(size, conf.data_path, transform=transform_train)
    training_dataset = DataLoader(dataset=training_data, batch_size=batch,
                                  shuffle=True, num_workers=conf.workers)
    
    if d < conf.depth - 1 and conf.use_pretrain:
        print('Load pretrain model (%d x %d)' % (size, size))
        G.module.load_state_dict(torch.load('./result/gen_%dx%d' % (size, size)))
        G_ema.load_state_dict(torch.load('./result/gen_ema_%dx%d' % (size, size)))
        D.module.load_state_dict(torch.load('./result/dis_%dx%d' % (size, size)))
        G_opt.load_state_dict(torch.load('./result/gen_opt_%dx%d' % (size, size)))
        D_opt.load_state_dict(torch.load('./result/dis_opt_%dx%d' % (size, size)))
        alpha = 1
        epoch = 1
    #print('Delta: %f' % alpha_delta)
    print('Epoch: %d' % epoch)
    print('Mini batch size: %d' % batch)
    print('Learning rate: %f' % lr_val)
    print('='*50, 'Currently, image size is %d x %d' % (size, size), '='*50)
    
    for g_params in G_opt.param_groups:
        if g_params['lr'] != 1e-5:
            g_params['lr'] == lr_val
    for d_params in D_opt.param_groups:
        d_params['lr'] == lr_val
    
    ticker = 0
    fade_in_rate = 0.5
    fade_point = int(fade_in_rate * epoch * len(training_dataset))
    G.train()
    D.train()
    for epoch in range(1, epoch+1):
        Tensor = torch.cuda.FloatTensor
        for idx, img in enumerate(training_dataset):
            flag_real = Tensor(img.size(0)).fill_(1.0)
            flag_fake = Tensor(img.size(0)).fill_(0.0)
            real_img = img.to(device)
            style_noise = torch.randn(real_img.size(0), conf.n_style).to(device)
        
            # ====================== Update Discriminator ======================
            D.zero_grad()
            dis_real_out = D(real_img, d, alpha=alpha)
            fake_img = G(style_noise, d, alpha=alpha, true_size=size)
            dis_fake_out = D(fake_img, d, alpha=alpha)
            
            if conf.loss == 'wgan-gp':
                dis_real = - torch.mean(dis_real_out)
                dis_fake = torch.mean(dis_fake_out)
            elif conf.loss == 'hinge':
                dis_real = nn.ReLU()(1.0 - dis_real_out).mean()
                dis_fake = nn.ReLU()(1.0 + dis_fake_out).mean()
            elif conf.loss == 'relativistic_hinge':
                r_diff = dis_fake_out - torch.mean(dis_real_out)
                f_diff = dis_real_out - torch.mean(dis_fake_out)
                dis_real = nn.ReLU()(1.0 + r_diff).mean()
                dis_fake = nn.ReLU()(1.0 - f_diff).mean()
            elif conf.loss == 'R1':
                dis_real = criterion(dis_real_out.view(-1), flag_real)
                dis_fake = criterion(dis_fake_out.view(-1), flag_fake)
            
            dis_loss = dis_real + dis_fake
            
            if conf.loss == 'R1':
                real_img_detach = torch.autograd.Variable(real_img, requires_grad=True)
                dis_real_out = D(real_img_detach, d, alpha=alpha).view(real_img.size(0), -1)
                dis_grad = torch.autograd.grad(outputs=dis_real_out,
                               inputs=real_img_detach,
                               grad_outputs=torch.ones(dis_real_out.size()).to(device),
                               retain_graph=True,
                               create_graph=True)[0].view(real_img.size(0), -1)
                r1_penalty = torch.sum(torch.mul(dis_grad, dis_grad))
                r1_loss = conf.lambda_r1 / 2 + r1_penalty
                dis_loss += r1_loss
            elif conf.loss == 'wgan-gp':
                loss_gp = gradient_penalty(real_img, fake_img, alpha, d)
                d_drift = conf.lambda_drift * torch.mean(dis_real ** 2)
                dis_loss = dis_loss + conf.lambda_gp * loss_gp + d_drift
            
            dis_loss.backward()
            D_opt.step()
        
            # ====================== Update Generator ======================
            G.zero_grad()
            style_noise = torch.randn(real_img.size(0), conf.n_style)
            fake_img = G(style_noise, d, alpha=alpha, true_size=size)
            gen_out = D(fake_img, d, alpha=alpha)
            if conf.loss == 'wgan-gp' or 'hinge':
                gen_loss = - gen_out.mean()
            elif conf.loss == 'relativistic_hinge':
                r_diff = gen_out - torch.mean(dis_real_out)
                f_diff = dis_real_out - torch.mean(gen_out)
                gen_real_loss = nn.ReLU()(1.0 - r_diff).mean()
                gen_fake_loss = nn.ReLU()(1.0 + f_diff).mean()
                gen_loss = gen_real_loss + gen_fake_loss
            elif conf.loss == 'R1':
                gen_loss = criterion(gen_out.view(-1), flag_real)
                
            gen_loss.backward()
            nn.utils.clip_grad_norm_(G.parameters(), max_norm=10.)
            G_opt.step()
            update_EMA(G, G_ema, strength=0.999)
            
            alpha = ticker / fade_point if ticker <= fade_point else 1
            #alpha = min(1.0, alpha+alpha_delta)
        
            ticker += 1
            iteration += 1
            if idx %  100 == 0:
                print('Training epoch: {} [{}/{} ({:.0f}%)] | D loss : {:.6f} | G loss: {:.6f} | Alpha: {:.6f}|'\
                      .format(epoch, idx * len(img), len(training_dataset.dataset),
                      100. * idx / len(training_dataset), dis_loss.item(), gen_loss.item(), alpha))
                
                tb.add_scalars('prediction loss depth%d' % d,
                               {'D': dis_loss.item(),
                                'G': gen_loss.item()},
                                iteration)
                
        print('Now we will save network parameters and generating image.')
        torch.save(G.module.state_dict(), os.path.join(conf.result_dir, conf.gen + '_%dx%d' % (size, size)))
        torch.save(G_ema.state_dict(), os.path.join(conf.result_dir, 'gen_ema' + '_%dx%d' % (size, size)))
        torch.save(D.module.state_dict(), os.path.join(conf.result_dir, conf.dis + '_%dx%d' % (size, size)))
        torch.save(G_opt.state_dict(), os.path.join(conf.result_dir, conf.G_opt + '_%dx%d' % (size, size)))
        torch.save(D_opt.state_dict(), os.path.join(conf.result_dir, conf.D_opt + '_%dx%d' % (size, size)))
    
        
        G_ema.eval()
        test_style_noise = torch.randn(conf.n_gen_img, conf.n_style)
        #test_mean = torch.tensor([0.5, 0.5, 0.5]).type_as(test_style_noise)[None,:,None,None]
        #test_std  = torch.tensor([0.5, 0.5, 0.5]).type_as(test_style_noise)[None,:,None,None]
    
        with torch.no_grad():
            test_img = G_ema.forward(test_style_noise, d, alpha=1.0, true_size=size)
        #test_img = (test_img * test_std) + test_mean
        test_img = torchvision.utils.make_grid(test_img, nrow=int(conf.n_gen_img ** 0.5))
        tb.add_image('Generated_Images (%d x %d)' % (size, size), test_img, global_step=epoch)