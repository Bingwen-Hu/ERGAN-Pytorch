"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from networks import AdaINGen, MsImageDis, VAEGen
from utils import weights_init, get_model_list, vgg_preprocess, load_vgg16, get_scheduler
from torch.autograd import Variable
import torch
import torch.nn as nn
import os

try:
    import apex
    from apex import amp, optimizers
    from apex.fp16_utils import *
except ImportError:
    print('This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')
    pass

def to_gray(half=False): #simple
    def forward(x):
        x = torch.mean(x, dim=1, keepdim=True)
        if half:
            x = x.half()
        return x
    return forward

class ERGAN_Trainer(nn.Module):
    def __init__(self, hyperparameters):
        super(ERGAN_Trainer, self).__init__()
        lr_G = hyperparameters['lr_G']
        lr_D = hyperparameters['lr_D']
        print(lr_D, lr_G)
        self.fp16 = hyperparameters['fp16']
        # Initiate the networks
        self.gen_a = AdaINGen(hyperparameters['input_dim_a'], hyperparameters['gen'])  # auto-encoder for domain a
        self.gen_b = AdaINGen(hyperparameters['input_dim_b'], hyperparameters['gen'])  # auto-encoder for domain b
        self.gen_b.enc_content = self.gen_a.enc_content  # content share weight
        #self.gen_b.enc_style = self.gen_a.enc_style
        self.dis_a = MsImageDis(hyperparameters['input_dim_a'], hyperparameters['dis'])  # discriminator for domain a
        self.dis_b = MsImageDis(hyperparameters['input_dim_b'], hyperparameters['dis'])  # discriminator for domain b
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)
        self.style_dim = hyperparameters['gen']['style_dim']
        self.a = hyperparameters['gen']['new_size'] / 224
        # fix the noise used in sampling
        display_size = int(hyperparameters['display_size'])
        self.s_a = torch.randn(display_size, self.style_dim, 1, 1).cuda()
        self.s_b = torch.randn(display_size, self.style_dim, 1, 1).cuda()
        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        dis_params = list(self.dis_a.parameters()) + list(self.dis_b.parameters())
        gen_params = list(self.gen_a.parameters()) + list(self.gen_b.parameters())

        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=lr_D, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr_G, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])

        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)

        # Network weight initialization
        self.apply(weights_init(hyperparameters['init']))
        self.dis_a.apply(weights_init('gaussian'))
        self.dis_b.apply(weights_init('gaussian'))

        if  self.fp16:
            self.gen_a = self.gen_a.cuda()
            self.dis_a = self.dis_a.cuda()
            self.gen_b = self.gen_b.cuda()
            self.dis_b = self.dis_b.cuda()
            self.gen_a, self.gen_opt = amp.initialize(self.gen_a, self.gen_opt, opt_level="O1")
            self.dis_a, self.dis_opt = amp.initialize(self.dis_a, self.dis_opt, opt_level="O1")

        # Load VGG model if needed
        if 'vgg_w' in hyperparameters.keys() and hyperparameters['vgg_w'] > 0:
            self.vgg = load_vgg16(hyperparameters['vgg_model_path'] + '/models')
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False

    def recon_criterion(self, input, target):

        input = input.type_as(target)
        return torch.mean(torch.abs(input - target))

    def forward(self, x_a, x_b):
        self.eval()
        s_a = Variable(self.s_a)
        s_b = Variable(self.s_b)
        c_a, s_a_fake = self.gen_a.encode(x_a)
        c_b, s_b_fake = self.gen_b.encode(x_b)
        x_ba = self.gen_a.decode(c_b, s_a, x_b)
        x_ab = self.gen_b.decode(c_a, s_b, x_a)
        self.train()
        return x_ab, x_ba

    def gen_update(self, x_a, x_b, hyperparameters):
        self.gen_opt.zero_grad()
        #mask = torch.ones(x_a.shape).cuda()
        block_a = x_a.clone()
        block_b = x_b.clone()
        block_a[:,:, round(self.a * 92):round(self.a * 144), round(self.a *48):round(self.a * 172)] = 0
        block_b[:,:, round(self.a * 92):round(self.a * 144), round(self.a *48):round(self.a * 172)] = 0

        # encode
        c_a, s_a_prime = self.gen_a.encode(x_a)
        c_b, s_b_prime = self.gen_b.encode(x_b)
        # decode (within domain)
        x_a_recon = self.gen_a.decode(c_a, s_a_prime, x_a)
        x_b_recon = self.gen_b.decode(c_b, s_b_prime, x_b)

        # decode (cross domain)
          # decode random
        # x_ba_randn = self.gen_a.decode(c_b, s_a, x_b)
        # x_ab_randn = self.gen_b.decode(c_a, s_b, x_a)
          # decode real
        x_ba_real = self.gen_a.decode(c_b, s_a_prime, x_b)
        x_ab_real = self.gen_b.decode(c_a, s_b_prime, x_a)

        block_ba_real = x_ba_real.clone()
        block_ab_real = x_ab_real.clone()
        block_ba_real[:,:, round(self.a * 92):round(self.a * 144), round(self.a *48):round(self.a * 172)] = 0
        block_ab_real[:,:, round(self.a * 92):round(self.a * 144), round(self.a *48):round(self.a * 172)] = 0
        # encode again
        # c_b_recon, s_a_recon = self.gen_a.encode(x_ba_randn)
        # c_a_recon, s_b_recon = self.gen_b.encode(x_ab_randn)

        c_b_real_recon, s_a_prime_recon = self.gen_a.encode(x_ba_real)
        c_a_real_recon, s_b_prime_recon = self.gen_b.encode(x_ab_real)
        # decode again (if needed)
        x_aba = self.gen_a.decode(c_a_real_recon, s_a_prime, x_a) if hyperparameters['recon_x_cyc_w'] > 0 else None
        x_bab = self.gen_b.decode(c_b_real_recon, s_b_prime, x_b) if hyperparameters['recon_x_cyc_w'] > 0 else None

        # reconstruction loss
        self.loss_gen_recon_res_a = self.recon_criterion(block_ab_real, block_a)
        self.loss_gen_recon_res_b = self.recon_criterion(block_ba_real, block_b)
        self.loss_gen_recon_x_a_re = self.recon_criterion(
            x_a_recon[:,:, round(self.a * 92):round(self.a * 144), round(self.a *48):round(self.a * 172)],
             x_a[:,:, round(self.a * 92):round(self.a * 144), round(self.a *48):round(self.a * 172)]) # meglass [88:146, 44:178]
        self.loss_gen_recon_x_b_re = self.recon_criterion(
             x_b_recon[:,:, round(self.a * 92):round(self.a * 144), round(self.a *48):round(self.a * 172)],
             x_b[:,:, round(self.a * 92):round(self.a * 144), round(self.a *48):round(self.a * 172)]) # celebA [92:144, 48:172]
        self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a)
        self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b)

        self.loss_gen_recon_s_a_prime = self.recon_criterion(s_a_prime_recon, s_a_prime)
        self.loss_gen_recon_s_b_prime = self.recon_criterion(s_b_prime_recon, s_b_prime)

        self.loss_gen_recon_c_a_real = self.recon_criterion(c_a_real_recon, c_a)
        self.loss_gen_recon_c_b_real = self.recon_criterion(c_b_real_recon, c_b)
        self.loss_gen_cycrecon_x_a = self.recon_criterion(x_aba, x_a) if hyperparameters['recon_x_cyc_w'] > 0 else 0
        self.loss_gen_cycrecon_x_b = self.recon_criterion(x_bab, x_b) if hyperparameters['recon_x_cyc_w'] > 0 else 0
        # GAN loss
        self.loss_gen_adv_a_real = self.dis_a.calc_gen_loss(x_ba_real)
        self.loss_gen_adv_b_real = self.dis_b.calc_gen_loss(x_ab_real)
        # domain-invariant perceptual loss
        # self.loss_gen_vgg_a = self.compute_vgg_loss(self.vgg, x_ba_randn, x_b) if hyperparameters['vgg_w'] > 0 else 0
        # self.loss_gen_vgg_b = self.compute_vgg_loss(self.vgg, x_ab_randn, x_a) if hyperparameters['vgg_w'] > 0 else 0
        # total loss
        self.loss_gen_total = hyperparameters['gan_w'] * self.loss_gen_adv_a_real + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_b + \
                              hyperparameters['recon_x_w_re'] * self.loss_gen_recon_x_b_re + \
                              hyperparameters['recon_s_w'] * self.loss_gen_recon_s_b_prime + \
                              hyperparameters['recon_c_w'] * self.loss_gen_recon_c_b_real + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_b+\
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_a + \
                              hyperparameters['recon_x_w_re'] * self.loss_gen_recon_x_a_re + \
                              hyperparameters['recon_s_w'] * self.loss_gen_recon_s_a_prime + \
                              hyperparameters['recon_c_w'] * self.loss_gen_recon_c_a_real + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_a + \
                              hyperparameters['gan_w'] * self.loss_gen_adv_b_real +\
                              hyperparameters['recon_x_w_res'] * self.loss_gen_recon_res_b + \
                              hyperparameters['recon_x_w_res'] * self.loss_gen_recon_res_b


        if  self.fp16:
            with amp.scale_loss(self.loss_gen_total, self.gen_opt) as scaled_loss:
                scaled_loss.backward()
            self.gen_opt.step()

        else:
            self.loss_gen_total.backward()
            self.gen_opt.step()

        #loss_gan = hyperparameters['gan_w'] * (self.loss_gen_adv_a + self.loss_gen_adv_b)
        loss_gan_real = hyperparameters['gan_w'] * (self.loss_gen_adv_a_real+self.loss_gen_adv_b_real )
        loss_x = hyperparameters['recon_x_w'] * (self.loss_gen_recon_x_a+self.loss_gen_recon_x_b )
        loss_x_re = hyperparameters['recon_x_w_re'] * (self.loss_gen_recon_x_a_re + self.loss_gen_recon_x_b_re)
        #loss_x_res = hyperparameters['recon_x_w_res'] * (self.loss_gen_recon_res_a + self.loss_gen_recon_res_b)
        #loss_s = hyperparameters['recon_s_w'] * (self.loss_gen_recon_s_a + self.loss_gen_recon_s_b)
        #loss_c = hyperparameters['recon_c_w'] * (self.loss_gen_recon_c_a + self.loss_gen_recon_c_b)
        loss_s_prime = hyperparameters['recon_s_w'] * (self.loss_gen_recon_s_a_prime + self.loss_gen_recon_s_b_prime)
        loss_c_real = hyperparameters['recon_c_w'] * (self.loss_gen_recon_c_a_real + self.loss_gen_recon_c_b_real)
        loss_x_cyc = hyperparameters['recon_x_cyc_w'] * ( self.loss_gen_cycrecon_x_a + self.loss_gen_cycrecon_x_b)

        #loss_vgg = hyperparameters['vgg_w'] * (self.loss_gen_vgg_a + self.loss_gen_vgg_b)
        print('||total:%.2f||gan_real:%.2f||x:%.2f||x_re:%.2f||s_prime:%.4f||c_real:%.2f||x_cyc:%.4f||'
              % (self.loss_gen_total,  loss_gan_real, loss_x, loss_x_re, loss_s_prime, loss_c_real, loss_x_cyc) )

    def compute_vgg_loss(self, vgg, img, target):
        img_vgg = vgg_preprocess(img)
        target_vgg = vgg_preprocess(target)
        img_fea = vgg(img_vgg)
        target_fea = vgg(target_vgg)
        return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)

    def sample(self, x_a, x_b):
        self.eval()
        # s_a1 = Variable(self.s_a)
        # s_b1 = Variable(self.s_b)
        # s_a2 = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        # s_b2 = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())

        #x_a_recon, x_b_recon, x_ba1, x_ba2, x_ab1, x_ab2 = [], [], [], [], [], []
        x_a_recon, x_b_recon, x_bab, x_ab, x_ba, x_aba = [], [], [], [], [], []
        for i in range(x_a.size(0)):
            c_a, s_a_fake = self.gen_a.encode(x_a[i].unsqueeze(0))
            c_b, s_b_fake = self.gen_b.encode(x_b[i].unsqueeze(0))

            x_a_recon.append(self.gen_a.decode(c_a, s_a_fake, x_a[i].unsqueeze(0)))
            x_b_recon.append(self.gen_b.decode(c_b, s_b_fake, x_b[i].unsqueeze(0)))

            x_ba_tmp = self.gen_a.decode(c_b, s_a_fake, x_b[i].unsqueeze(0))
            x_ab_tmp = self.gen_b.decode(c_a, s_b_fake, x_a[i].unsqueeze(0))
            x_ba.append(x_ba_tmp)
            x_ab.append(x_ab_tmp)

            c_b_recon, _ = self.gen_a.encode(x_ba_tmp)
            c_a_recon, _ = self.gen_b.encode(x_ab_tmp)

            x_aba.append(self.gen_a.decode(c_a_recon, s_a_fake, x_a[i].unsqueeze(0)))
            x_bab.append(self.gen_b.decode(c_b_recon, s_b_fake, x_b[i].unsqueeze(0)))

        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)

        x_aba, x_bab = torch.cat(x_aba), torch.cat(x_bab)
        x_ab, x_ba = torch.cat(x_ab), torch.cat(x_ba)
        self.train()

        return x_a, x_a_recon,  x_ab, x_ba, x_b, x_aba, x_b, x_b_recon, x_ba,x_ab, x_a, x_bab
    def dis_update(self, x_a, x_b, hyperparameters):
        self.dis_opt.zero_grad()
        # s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        # s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        # # encode
        c_a, s_a_prime = self.gen_a.encode(x_a)
        c_b, s_b_prime = self.gen_b.encode(x_b)
        # decode (cross domain)

        x_ba_real = self.gen_a.decode(c_b, s_a_prime, x_b)
        x_ab_real = self.gen_b.decode(c_a, s_b_prime, x_a)
        # D loss
        # self.loss_dis_a = self.dis_a.calc_dis_loss(x_ba_randn.detach(), x_a)
        # self.loss_dis_b = self.dis_b.calc_dis_loss(x_ab_randn.detach(), x_b)

        self.loss_dis_a_real = self.dis_a.calc_dis_loss(x_ba_real.detach(), x_a)
        self.loss_dis_b_real = self.dis_b.calc_dis_loss(x_ab_real.detach(), x_b)
        self.loss_dis_total = hyperparameters['gan_w'] * (self.loss_dis_a_real + self.loss_dis_b_real )

        if  self.fp16:
            with amp.scale_loss(self.loss_dis_total, self.dis_opt) as scaled_loss:
                scaled_loss.backward()
            self.dis_opt.step()
        else:
            self.loss_dis_total.backward()
            self.dis_opt.step()


    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

    def resume(self, checkpoint_dir, hyperparameters):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        self.gen_a.load_state_dict(state_dict['a'])
        self.gen_b.load_state_dict(state_dict['b'])
        iterations = int(last_model_name[-11:-3])
        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name)
        self.dis_a.load_state_dict(state_dict['a'])
        self.dis_b.load_state_dict(state_dict['b'])
        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self.dis_opt.load_state_dict(state_dict['dis'])
        self.gen_opt.load_state_dict(state_dict['gen'])
        # Reinitilize schedulers
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters, iterations-375000) #fine_tune -370000
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters, iterations-375000)
        print('Resume from iteration %d' % iterations)
        return iterations

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save({'a': self.gen_a.state_dict(), 'b': self.gen_b.state_dict()}, gen_name)
        torch.save({'a': self.dis_a.state_dict(), 'b': self.dis_b.state_dict()}, dis_name)
        torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict()}, opt_name)

################################################################################################
#UNIT
################################################################################################
class UNIT_Trainer(nn.Module):
    def __init__(self, hyperparameters):
        super(UNIT_Trainer, self).__init__()
        lr = hyperparameters['lr']
        # Initiate the networks
        self.gen_a = VAEGen(hyperparameters['input_dim_a'], hyperparameters['gen'])  # auto-encoder for domain a
        self.gen_b = VAEGen(hyperparameters['input_dim_b'], hyperparameters['gen'])  # auto-encoder for domain b
        self.dis_a = MsImageDis(hyperparameters['input_dim_a'], hyperparameters['dis'])  # discriminator for domain a
        self.dis_b = MsImageDis(hyperparameters['input_dim_b'], hyperparameters['dis'])  # discriminator for domain b
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)

        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        dis_params = list(self.dis_a.parameters()) + list(self.dis_b.parameters())
        gen_params = list(self.gen_a.parameters()) + list(self.gen_b.parameters())
        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)

        # Network weight initialization
        self.apply(weights_init(hyperparameters['init']))
        self.dis_a.apply(weights_init('gaussian'))
        self.dis_b.apply(weights_init('gaussian'))

        # Load VGG model if needed
        if 'vgg_w' in hyperparameters.keys() and hyperparameters['vgg_w'] > 0:
            self.vgg = load_vgg16(hyperparameters['vgg_model_path'] + '/models')
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False

    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def forward(self, x_a, x_b):
        self.eval()
        h_a, _ = self.gen_a.encode(x_a)
        h_b, _ = self.gen_b.encode(x_b)
        x_ba = self.gen_a.decode(h_b)
        x_ab = self.gen_b.decode(h_a)
        self.train()
        return x_ab, x_ba

    def __compute_kl(self, mu):
        # def _compute_kl(self, mu, sd):
        # mu_2 = torch.pow(mu, 2)
        # sd_2 = torch.pow(sd, 2)
        # encoding_loss = (mu_2 + sd_2 - torch.log(sd_2)).sum() / mu_2.size(0)
        # return encoding_loss
        mu_2 = torch.pow(mu, 2)
        encoding_loss = torch.mean(mu_2)
        return encoding_loss

    def gen_update(self, x_a, x_b, hyperparameters):
        self.gen_opt.zero_grad()
        # encode
        h_a, n_a = self.gen_a.encode(x_a)
        h_b, n_b = self.gen_b.encode(x_b)
        # decode (within domain)
        x_a_recon = self.gen_a.decode(h_a + n_a)
        x_b_recon = self.gen_b.decode(h_b + n_b)
        # decode (cross domain)
        x_ba = self.gen_a.decode(h_b + n_b)
        x_ab = self.gen_b.decode(h_a + n_a)
        # encode again
        h_b_recon, n_b_recon = self.gen_a.encode(x_ba)
        h_a_recon, n_a_recon = self.gen_b.encode(x_ab)
        # decode again (if needed)
        x_aba = self.gen_a.decode(h_a_recon + n_a_recon) if hyperparameters['recon_x_cyc_w'] > 0 else None
        x_bab = self.gen_b.decode(h_b_recon + n_b_recon) if hyperparameters['recon_x_cyc_w'] > 0 else None

        # reconstruction loss
        self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a)
        self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b)
        self.loss_gen_recon_kl_a = self.__compute_kl(h_a)
        self.loss_gen_recon_kl_b = self.__compute_kl(h_b)
        self.loss_gen_cyc_x_a = self.recon_criterion(x_aba, x_a)
        self.loss_gen_cyc_x_b = self.recon_criterion(x_bab, x_b)
        self.loss_gen_recon_kl_cyc_aba = self.__compute_kl(h_a_recon)
        self.loss_gen_recon_kl_cyc_bab = self.__compute_kl(h_b_recon)
        # GAN loss
        self.loss_gen_adv_a = self.dis_a.calc_gen_loss(x_ba)
        self.loss_gen_adv_b = self.dis_b.calc_gen_loss(x_ab)
        # domain-invariant perceptual loss
        self.loss_gen_vgg_a = self.compute_vgg_loss(self.vgg, x_ba, x_b) if hyperparameters['vgg_w'] > 0 else 0
        self.loss_gen_vgg_b = self.compute_vgg_loss(self.vgg, x_ab, x_a) if hyperparameters['vgg_w'] > 0 else 0
        # total loss
        self.loss_gen_total = hyperparameters['gan_w'] * self.loss_gen_adv_a + \
                              hyperparameters['gan_w'] * self.loss_gen_adv_b + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_a + \
                              hyperparameters['recon_kl_w'] * self.loss_gen_recon_kl_a + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_b + \
                              hyperparameters['recon_kl_w'] * self.loss_gen_recon_kl_b + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cyc_x_a + \
                              hyperparameters['recon_kl_cyc_w'] * self.loss_gen_recon_kl_cyc_aba + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cyc_x_b + \
                              hyperparameters['recon_kl_cyc_w'] * self.loss_gen_recon_kl_cyc_bab + \
                              hyperparameters['vgg_w'] * self.loss_gen_vgg_a + \
                              hyperparameters['vgg_w'] * self.loss_gen_vgg_b
        self.loss_gen_total.backward()
        self.gen_opt.step()

    def compute_vgg_loss(self, vgg, img, target):
        img_vgg = vgg_preprocess(img)
        target_vgg = vgg_preprocess(target)
        img_fea = vgg(img_vgg)
        target_fea = vgg(target_vgg)
        return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)

    def sample(self, x_a, x_b):
        self.eval()
        x_a_recon, x_b_recon, x_ba, x_ab = [], [], [], []
        for i in range(x_a.size(0)):
            h_a, _ = self.gen_a.encode(x_a[i].unsqueeze(0))
            h_b, _ = self.gen_b.encode(x_b[i].unsqueeze(0))
            x_a_recon.append(self.gen_a.decode(h_a))
            x_b_recon.append(self.gen_b.decode(h_b))
            x_ba.append(self.gen_a.decode(h_b))
            x_ab.append(self.gen_b.decode(h_a))
        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_ba = torch.cat(x_ba)
        x_ab = torch.cat(x_ab)
        self.train()
        return x_a, x_a_recon, x_ab, x_b, x_b_recon, x_ba

    def dis_update(self, x_a, x_b, hyperparameters):
        self.dis_opt.zero_grad()
        # encode
        h_a, n_a = self.gen_a.encode(x_a)
        h_b, n_b = self.gen_b.encode(x_b)
        # decode (cross domain)
        x_ba = self.gen_a.decode(h_b + n_b)
        x_ab = self.gen_b.decode(h_a + n_a)
        # D loss
        self.loss_dis_a = self.dis_a.calc_dis_loss(x_ba.detach(), x_a)
        self.loss_dis_b = self.dis_b.calc_dis_loss(x_ab.detach(), x_b)
        self.loss_dis_total = hyperparameters['gan_w'] * self.loss_dis_a + hyperparameters['gan_w'] * self.loss_dis_b
        self.loss_dis_total.backward()
        self.dis_opt.step()

    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

    def resume(self, checkpoint_dir, hyperparameters):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        self.gen_a.load_state_dict(state_dict['a'])
        self.gen_b.load_state_dict(state_dict['b'])
        iterations = int(last_model_name[-11:-3])
        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name)
        self.dis_a.load_state_dict(state_dict['a'])
        self.dis_b.load_state_dict(state_dict['b'])
        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self.dis_opt.load_state_dict(state_dict['dis'])
        self.gen_opt.load_state_dict(state_dict['gen'])
        # Reinitilize schedulers
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters, iterations)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters, iterations)
        print('Resume from iteration %d' % iterations)
        return iterations

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save({'a': self.gen_a.state_dict(), 'b': self.gen_b.state_dict()}, gen_name)
        torch.save({'a': self.dis_a.state_dict(), 'b': self.dis_b.state_dict()}, dis_name)
        torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict()}, opt_name)
