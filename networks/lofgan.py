import random

import numpy as np
from torch import autograd

from networks.blocks import *
from networks.loss import *
from utils import batched_index_select, batched_scatter
import time

use_generator_sc = False # whether or not use skip connection (sc) for generator
use_attention = False # only effective if using generator with sc
use_skip_index = False # unused yet


class LoFGAN(nn.Module):
    def __init__(self, config):
        super(LoFGAN, self).__init__()

        if not use_generator_sc:   
            self.gen = Generator(config['gen'])
        else:
            self.gen = GeneratorWithSC(config['gen'])
        self.dis = Discriminator(config['dis'])

        self.w_adv_g = config['w_adv_g']
        self.w_adv_d = config['w_adv_d']
        self.w_recon = config['w_recon']
        self.w_cls = config['w_cls']
        self.w_gp = config['w_gp']
        self.n_sample = config['n_sample_train']

    def forward(self, xs, y, mode):
        if mode == 'gen_update':
            fake_x, similarity, indices_feat, indices_ref, base_index = self.gen(xs)

            loss_recon = local_recon_criterion(xs, fake_x, similarity, indices_feat, indices_ref, base_index, s=8)

            feat_real, _, _ = self.dis(xs)
            feat_fake, logit_adv_fake, logit_c_fake = self.dis(fake_x)
            loss_adv_gen = torch.mean(-logit_adv_fake)
            loss_cls_gen = F.cross_entropy(logit_c_fake, y.squeeze())

            loss_recon = loss_recon * self.w_recon
            loss_adv_gen = loss_adv_gen * self.w_adv_g
            loss_cls_gen = loss_cls_gen * self.w_cls

            loss_total = loss_recon + loss_adv_gen + loss_cls_gen
            loss_total.backward()

            return {'loss_total': loss_total,
                    'loss_recon': loss_recon,
                    'loss_adv_gen': loss_adv_gen,
                    'loss_cls_gen': loss_cls_gen}

        elif mode == 'dis_update':
            xs.requires_grad_()

            advdissttt = time.time()
            _, logit_adv_real, logit_c_real = self.dis(xs)
            loss_adv_dis_real = torch.nn.ReLU()(1.0 - logit_adv_real).mean()
            loss_adv_dis_real = loss_adv_dis_real * self.w_adv_d
            loss_adv_dis_real.backward(retain_graph=True)
            advdisendt = time.time()

            regdissttt = time.time()
            y_extend = y.repeat(1, self.n_sample).view(-1)
            index = torch.LongTensor(range(y_extend.size(0))).cuda()
            logit_c_real_forgp = logit_c_real[index, y_extend].unsqueeze(1)
            loss_reg_dis = self.calc_grad2(logit_c_real_forgp, xs)

            loss_reg_dis = loss_reg_dis * self.w_gp
            loss_reg_dis.backward(retain_graph=True)
            regdisendt = time.time()

            clsdissttt = time.time()
            loss_cls_dis = F.cross_entropy(logit_c_real, y_extend)
            loss_cls_dis = loss_cls_dis * self.w_cls
            loss_cls_dis.backward()
            clsdisendt = time.time()

            advdisfksttt = time.time()
            with torch.no_grad():
                fake_x = self.gen(xs)[0]

            _, logit_adv_fake, _ = self.dis(fake_x.detach())
            loss_adv_dis_fake = torch.nn.ReLU()(1.0 + logit_adv_fake).mean()
            loss_adv_dis_fake = loss_adv_dis_fake * self.w_adv_d
            loss_adv_dis_fake.backward()
            advdisfkendt = time.time()

#             print(f"loss time for adv_dis {advdisendt-advdissttt} reg_dis {regdisendt-regdissttt} cls_dis {clsdisendt-clsdissttt} adv_dis_fk {advdisfkendt-advdisfksttt}")
            loss_total = loss_adv_dis_real + loss_adv_dis_fake + loss_cls_dis
            return {'loss_total': loss_total,
                    'loss_adv_dis': loss_adv_dis_fake + loss_adv_dis_real,
                    'loss_adv_dis_real': loss_adv_dis_real,
                    'loss_adv_dis_fake': loss_adv_dis_fake,
                    'loss_cls_dis': loss_cls_dis,
                    'loss_reg': loss_reg_dis}

        else:
            assert 0, 'Not support operation'

    def generate(self, xs):
        fake_x = self.gen(xs)[0]
        return fake_x

    def calc_grad2(self, d_out, x_in):
        batch_size = x_in.size(0)
        grad_dout = autograd.grad(outputs=d_out.mean(),
                                  inputs=x_in,
                                  create_graph=True,
                                  retain_graph=True,
                                  only_inputs=True)[0]
        grad_dout2 = grad_dout.pow(2)
        assert (grad_dout2.size() == x_in.size())
        reg = grad_dout2.sum()
        reg /= batch_size
        return reg

class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.soft_label = False
        nf = config['nf']
        n_class = config['num_classes']
        n_res_blks = config['n_res_blks']

        cnn_f = [Conv2dBlock(3, nf, 5, 1, 2,
                             pad_type='reflect',
                             norm='sn',
                             activation='none')]
        for i in range(n_res_blks):
            nf_out = np.min([nf * 2, 1024])
            cnn_f += [ActFirstResBlock(nf, nf_out, fhid=None, activation='lrelu', norm='sn')]
            cnn_f += [nn.ReflectionPad2d(1)]
            cnn_f += [nn.AvgPool2d(kernel_size=3, stride=2)]
            nf = np.min([nf * 2, 1024])

        nf_out = np.min([nf * 2, 1024])
        cnn_f += [ActFirstResBlock(nf, nf_out, fhid=None, activation='lrelu', norm='sn')]
        cnn_adv = [nn.AdaptiveAvgPool2d(1),
                   Conv2dBlock(nf_out, 1, 1, 1,
                               norm='none',
                               activation='none',
                               activation_first=False)]
        cnn_c = [nn.AdaptiveAvgPool2d(1),
                 Conv2dBlock(nf_out, n_class, 1, 1,
                             norm='none',
                             activation='none',
                             activation_first=False)]
        self.cnn_f = nn.Sequential(*cnn_f)
        self.cnn_adv = nn.Sequential(*cnn_adv)
        self.cnn_c = nn.Sequential(*cnn_c)

    def forward(self, x):
        if len(x.size()) == 5:
            B, K, C, H, W = x.size()
            x = x.view(B * K, C, H, W)
        else:
            B, C, H, W = x.size()
            K = 1
        featst = time.time()
        feat = self.cnn_f(x)
        feated = time.time()
        logit_advst = time.time()
        logit_adv = self.cnn_adv(feat).view(B * K, -1)
        logit_adved = time.time()
        logit_cst = time.time()
        logit_c = self.cnn_c(feat).view(B * K, -1)
        logit_ced = time.time()
#         print(f"dis f time {feated-featst} adv time {logit_adved-logit_advst} c time {logit_ced-logit_cst}")
        return feat, logit_adv, logit_c



class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim
        
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)
        
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)
        
        out = self.gamma * out + x
        return out, attention

class GeneratorWithSC(nn.Module):
    def __init__(self, config):
        super(GeneratorWithSC, self).__init__()
        self.encoder = SmallerEncoderWithSC()
        self.decoder = SmallerDecoderWithSC()
        self.fusion = LocalFusionModule(inplanes=64, rate=config['rate'])

#     def forward(self, xs):
#         b, k, C, H, W = xs.size()
#         xs = xs.view(-1, C, H, W)
#         x1, x2, x3, x4, querys = self.encoder(xs)
#         c, h, w = querys.size()[-3:]
#         querys = querys.view(b, k, c, h, w)

#         similarity_total = torch.cat([torch.rand(b, 1) for _ in range(k)], dim=1).cuda()  # b*k
#         similarity_sum = torch.sum(similarity_total, dim=1, keepdim=True).expand(b, k)  # b*k
#         similarity = similarity_total / similarity_sum  # b*k

#         base_index = random.choice(range(k))

#         base_feat = querys[:, base_index, :, :, :]
#         feat_gen, indices_feat, indices_ref = self.fusion(base_feat, querys, base_index, similarity)

#         fake_x = self.decoder([x1, x2, x3, x4, feat_gen])

#         return fake_x, similarity, indices_feat, indices_ref, base_index
    def forward(self, xs):
        b, k, C, H, W = xs.size()
        xs = xs.view(-1, C, H, W)
        querys = self.encoder(xs)
        c, h, w = querys[-1].size()[-3:]
#         for j in querys:
#             print(j.size())
        queryswh = [q.size()[-2:] for q in querys]
        querys = [q.view(b, k, -1, h, w) for q in querys]
#         print("after view")
#         for j in querys:
#             print(j.size())

        similarity_total = torch.cat([torch.rand(b, 1) for _ in range(k)], dim=1).cuda()  # b*k
        similarity_sum = torch.sum(similarity_total, dim=1, keepdim=True).expand(b, k)  # b*k
        similarity = similarity_total / similarity_sum  # b*k

        base_index = random.choice(range(k))
        skip_index = random.choice(range(k))

        base_feat = querys[-1][:, base_index, :, :, :]
        feat_gen, indices_feat, indices_ref = self.fusion(base_feat, querys[-1], base_index, similarity)

        # choose the skip-connection feature to be the base feature correspondance
        if not use_skip_index:
            skip_feats = [q[:, base_index, :, :, :] for q in querys[:-1]]
        else:
            skip_feats = [q[:, skip_index, :, :, :] for q in querys[:-1]]
        # reshape the size of the skip features to match the input w,h dimensions
        b2, _, _, _ = skip_feats[0].size()
        skip_feats = [skip_feats[j].view(b2, -1, queryswh[j][0], queryswh[j][1]) for j in range(len(skip_feats))]
#         print("skip_feats")
#         for j in skip_feats:
#             print(j.size())
#         print(feat_gen.shape)
        fake_x = self.decoder([*skip_feats, feat_gen])

        return fake_x, similarity, indices_feat, indices_ref, base_index

class SmallerEncoderWithSC(nn.Module):
    def __init__(self):
        super(SmallerEncoderWithSC, self).__init__()

        self.layer1 = Conv2dBlock(3, 8, 5, 1, 2, norm='bn', activation='lrelu', pad_type='reflect')
        self.layer2 = Conv2dBlock(8, 16, 3, 2, 1, norm='bn', activation='lrelu', pad_type='reflect')
        self.layer3 = Conv2dBlock(16, 32, 3, 2, 1, norm='bn', activation='lrelu', pad_type='reflect')
        if use_attention:
            self.att1 = SelfAttention(32)
            self.att2 = SelfAttention(64)
        self.layer4 = Conv2dBlock(32, 64, 3, 2, 1, norm='bn', activation='lrelu', pad_type='reflect')
        self.layer5 = Conv2dBlock(64, 64, 3, 2, 1, norm='bn', activation='lrelu', pad_type='reflect')

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        if use_attention:
            x3, _ = self.att1(x3)
        x4 = self.layer4(x3)
        if use_attention:
            x4, _ = self.att2(x4)
        x5 = self.layer5(x4)
        return [x1, x2, x3, x4, x5]

class SmallerDecoderWithSC(nn.Module):
    def __init__(self):
        super(SmallerDecoderWithSC, self).__init__()

        self.upsample1 = nn.Upsample(scale_factor=2)
        self.conv1 = Conv2dBlock(128, 64, 3, 1, 1, norm='bn', activation='lrelu', pad_type='reflect')  # Concatenate with x4
        
        if use_attention:
            self.att1 = SelfAttention(64)
            self.att2 = SelfAttention(32)
        
        self.upsample2 = nn.Upsample(scale_factor=2)
        self.conv2 = Conv2dBlock(128, 32, 3, 1, 1, norm='bn', activation='lrelu', pad_type='reflect')  # Concatenate with x3
        
        self.upsample3 = nn.Upsample(scale_factor=2)
        self.conv3 = Conv2dBlock(32, 16, 3, 1, 1, norm='bn', activation='lrelu', pad_type='reflect')   # Concatenate with x2
#         self.conv3 = Conv2dBlock(64, 16, 3, 1, 1, norm='bn', activation='lrelu', pad_type='reflect')   # Concatenate with x2
        
        self.upsample4 = nn.Upsample(scale_factor=2)
#         self.conv4 = Conv2dBlock(32, 8, 3, 1, 1, norm='bn', activation='lrelu', pad_type='reflect')   # Concatenate with x1
        self.conv4 = Conv2dBlock(16, 8, 3, 1, 1, norm='bn', activation='lrelu', pad_type='reflect')   # Concatenate with x1
        
        self.conv5 = Conv2dBlock(8, 3, 5, 1, 2, norm='none', activation='tanh', pad_type='reflect')
        
        # match skip channels by 1*1 convolution (since skip-connections are done after up-sampling)
        self.skipconv1 = Conv2dBlock(64, 64, 1, 1, norm='bn', activation='lrelu', pad_type='reflect')
        self.skipconv2 = Conv2dBlock(32, 64, 1, 1, norm='bn', activation='lrelu', pad_type='reflect')
#         self.skipconv3 = Conv2dBlock(16, 32, 1, 1, norm='bn', activation='lrelu', pad_type='reflect')
#         self.skipconv4 = Conv2dBlock(8, 16, 1, 1, norm='bn', activation='lrelu', pad_type='reflect')
        

    def forward(self, xs):
        x1, x2, x3, x4, x5 = xs # note that initially x1-x4 are not reshaped

        x = self.upsample1(x5)
#         print(f"{x.shape}, {x4.shape}")
        # reshape skip features since they're connected after 2-times up-sampling
        x4 = self.skipconv1(x4)
        x = torch.cat([x, x4], dim=1)
        x = self.conv1(x)
        if use_attention:
            x, _ = self.att1(x) 
        
        x = self.upsample2(x)
#         print(f"{x.shape}, {x3.shape}")
#         x3 = x3.view(x.size())
        x3 = self.skipconv2(x3)
        x = torch.cat([x, x3], dim=1)
        x = self.conv2(x)
        if use_attention:
            x, _ = self.att2(x) 
        
        x = self.upsample3(x)
#         print(f"{x.shape}, {x2.shape}")
#         x2 = self.skipconv3(x2) 
#         x = torch.cat([x, x2], dim=1)
        x = self.conv3(x)
        
        x = self.upsample4(x)
#         print(f"{x.shape}, {x1.shape}")
#         x1 = self.skipconv4(x1)
#         x = torch.cat([x, x1], dim=1)
        x = self.conv4(x)
        
        x = self.conv5(x)
        return x


    
class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
#         self.encoder = Encoder()
#         self.decoder = Decoder()
        self.encoder = SmallerEncoder()
        self.decoder = SmallerDecoder()
#         self.encoder = EvenSmallerEncoder()
#         self.decoder = EvenSmallerDecoder()
#         self.encoder = EvenMuchSmallerEncoder()
#         self.decoder = EvenMuchSmallerDecoder()
        self.fusion = LocalFusionModule(inplanes=32, rate=config['rate'])

    def forward(self, xs):
        b, k, C, H, W = xs.size()
        xs = xs.view(-1, C, H, W)
        querys = self.encoder(xs)
        c, h, w = querys.size()[-3:]
        querys = querys.view(b, k, c, h, w)

        similarity_total = torch.cat([torch.rand(b, 1) for _ in range(k)], dim=1).cuda()  # b*k
        similarity_sum = torch.sum(similarity_total, dim=1, keepdim=True).expand(b, k)  # b*k
        similarity = similarity_total / similarity_sum  # b*k

        base_index = random.choice(range(k))

        base_feat = querys[:, base_index, :, :, :]
        feat_gen, indices_feat, indices_ref = self.fusion(base_feat, querys, base_index, similarity)

        fake_x = self.decoder(feat_gen)

        return fake_x, similarity, indices_feat, indices_ref, base_index


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        model = [Conv2dBlock(3, 32, 5, 1, 2,
                             norm='bn',
                             activation='lrelu',
                             pad_type='reflect'),
                 Conv2dBlock(32, 64, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='reflect'),
                 Conv2dBlock(64, 128, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='reflect'),
                 Conv2dBlock(128, 128, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='reflect'),
                 Conv2dBlock(128, 128, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='reflect')
                 ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        model = [nn.Upsample(scale_factor=2),
                 Conv2dBlock(128, 128, 3, 1, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='reflect'),
                 nn.Upsample(scale_factor=2),
                 Conv2dBlock(128, 128, 3, 1, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='reflect'),
                 nn.Upsample(scale_factor=2),
                 Conv2dBlock(128, 64, 3, 1, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='reflect'),
                 nn.Upsample(scale_factor=2),
                 Conv2dBlock(64, 32, 3, 1, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='reflect'),
                 Conv2dBlock(32, 3, 5, 1, 2,
                             norm='none',
                             activation='tanh',
                             pad_type='reflect')]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)





class SmallerEncoder(nn.Module):
    def __init__(self):
        super(SmallerEncoder, self).__init__()

        self.layer1 = Conv2dBlock(3, 8, 5, 1, 2, norm='bn', activation='lrelu', pad_type='reflect')
        self.layer2 = Conv2dBlock(8, 16, 3, 2, 1, norm='bn', activation='lrelu', pad_type='reflect')
        self.layer3 = Conv2dBlock(16, 32, 3, 2, 1, norm='bn', activation='lrelu', pad_type='reflect')
        self.layer4 = Conv2dBlock(32, 64, 3, 2, 1, norm='bn', activation='lrelu', pad_type='reflect')
        self.layer5 = Conv2dBlock(64, 64, 3, 2, 1, norm='bn', activation='lrelu', pad_type='reflect')

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x

class SmallerDecoder(nn.Module):
    def __init__(self):
        super(SmallerDecoder, self).__init__()

        self.upsample1 = nn.Upsample(scale_factor=2)
        self.conv1 = Conv2dBlock(64, 64, 3, 1, 1, norm='bn', activation='lrelu', pad_type='reflect')
        
        self.upsample2 = nn.Upsample(scale_factor=2)
        self.conv2 = Conv2dBlock(64, 32, 3, 1, 1, norm='bn', activation='lrelu', pad_type='reflect')
        
        self.upsample3 = nn.Upsample(scale_factor=2)
        self.conv3 = Conv2dBlock(32, 16, 3, 1, 1, norm='bn', activation='lrelu', pad_type='reflect')
        
        self.upsample4 = nn.Upsample(scale_factor=2)
        self.conv4 = Conv2dBlock(16, 8, 3, 1, 1, norm='bn', activation='lrelu', pad_type='reflect')
        
        self.conv5 = Conv2dBlock(8, 3, 5, 1, 2, norm='none', activation='tanh', pad_type='reflect')

    def forward(self, x):
        x = self.upsample1(x)
        x = self.conv1(x)
        
        x = self.upsample2(x)
        x = self.conv2(x)
        
        x = self.upsample3(x)
        x = self.conv3(x)
        
        x = self.upsample4(x)
        x = self.conv4(x)
        
        x = self.conv5(x)
        return x

### Even Smaller Encoder Decoder with Pooling Layers
class EvenSmallerEncoder(nn.Module):
    def __init__(self):
        super(EvenSmallerEncoder, self).__init__()
        model = [
            Conv2dBlock(3, 8, 3, 1, 1, norm='bn', activation='lrelu', pad_type='reflect'),  # 128x128 -> 128x128
            nn.MaxPool2d(2, 2),  # 128x128 -> 64x64
            Conv2dBlock(8, 16, 3, 1, 1, norm='bn', activation='lrelu', pad_type='reflect'),  # 64x64 -> 64x64
            nn.MaxPool2d(2, 2),  # 64x64 -> 32x32
            Conv2dBlock(16, 32, 3, 1, 1, norm='bn', activation='lrelu', pad_type='reflect'),  # 32x32 -> 32x32
            nn.MaxPool2d(2, 2),  # 32x32 -> 16x16
            Conv2dBlock(32, 64, 3, 1, 1, norm='bn', activation='lrelu', pad_type='reflect'),  # 16x16 -> 16x16
            nn.MaxPool2d(2, 2)  # 16x16 -> 8x8
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class EvenSmallerDecoder(nn.Module):
    def __init__(self):
        super(EvenSmallerDecoder, self).__init__()
        model = [
            nn.Upsample(scale_factor=2),  # 8x8 -> 16x16
            Conv2dBlock(64, 32, 3, 1, 1, norm='bn', activation='lrelu', pad_type='reflect'),
            nn.Upsample(scale_factor=2),  # 16x16 -> 32x32
            Conv2dBlock(32, 16, 3, 1, 1, norm='bn', activation='lrelu', pad_type='reflect'),
            nn.Upsample(scale_factor=2),  # 32x32 -> 64x64
            Conv2dBlock(16, 8, 3, 1, 1, norm='bn', activation='lrelu', pad_type='reflect'),
            nn.Upsample(scale_factor=2),  # 64x64 -> 128x128
            Conv2dBlock(8, 3, 3, 1, 1, norm='none', activation='tanh', pad_type='reflect')
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class EvenMuchSmallerEncoder(nn.Module):
    def __init__(self):
        super(EvenMuchSmallerEncoder, self).__init__()
        model = [
            Conv2dBlock(3, 8, 3, 1, 1, norm='bn', activation='lrelu', pad_type='reflect'),  # 128x128 -> 128x128
            nn.MaxPool2d(2, 2),  # 128x128 -> 64x64
            Conv2dBlock(8, 16, 3, 1, 1, norm='bn', activation='lrelu', pad_type='reflect'),  # 64x64 -> 64x64
            nn.MaxPool2d(2, 2),  # 64x64 -> 32x32
            Conv2dBlock(16, 32, 3, 1, 1, norm='bn', activation='lrelu', pad_type='reflect'),  # 32x32 -> 32x32
            nn.MaxPool2d(4, 4),  # 32x32 -> 8x8
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class EvenMuchSmallerDecoder(nn.Module):
    def __init__(self):
        super(EvenMuchSmallerDecoder, self).__init__()
        model = [
            nn.Upsample(scale_factor=4),  # 8x8 -> 32x32
            Conv2dBlock(32, 16, 3, 1, 1, norm='bn', activation='lrelu', pad_type='reflect'),  # 32x32
            nn.Upsample(scale_factor=2),  # 32x32 -> 64x64
            Conv2dBlock(16, 8, 3, 1, 1, norm='bn', activation='lrelu', pad_type='reflect'),  # 64x64
            nn.Upsample(scale_factor=2),  # 64x64 -> 128x128
            Conv2dBlock(8, 3, 3, 1, 1, norm='none', activation='tanh', pad_type='reflect')  # 128x128
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)



class LocalFusionModule(nn.Module):
    def __init__(self, inplanes, rate):
        super(LocalFusionModule, self).__init__()

        self.W = nn.Sequential(
            nn.Conv2d(inplanes, inplanes, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(inplanes)
        )
        self.rate = rate

    def forward(self, feat, refs, index, similarity):
        refs = torch.cat([refs[:, :index, :, :, :], refs[:, (index + 1):, :, :, :]], dim=1)
        base_similarity = similarity[:, index]
        ref_similarities = torch.cat([similarity[:, :index], similarity[:, (index + 1):]], dim=1)

        # take ref:(32, 2, 128, 8, 8) for example
        b, n, c, h, w = refs.size()
        refs = refs.view(b * n, c, h, w)

        w_feat = feat.view(b, c, -1)
        w_feat = w_feat.permute(0, 2, 1).contiguous()
        w_feat = F.normalize(w_feat, dim=2)  # (32*64*128)

        w_refs = refs.view(b, n, c, -1)
        w_refs = w_refs.permute(0, 2, 1, 3).contiguous().view(b, c, -1)
        w_refs = F.normalize(w_refs, dim=1)  # (32*128*128)

        # local selection
        rate = self.rate
        num = int(rate * h * w)
        feat_indices = torch.cat([torch.LongTensor(random.sample(range(h * w), num)).unsqueeze(0) for _ in range(b)],
                                 dim=0).cuda()  # B*num

        feat = feat.view(b, c, -1)  # (32*128*64)
        feat_select = batched_index_select(feat, dim=2, index=feat_indices)  # (32*128*12)

        # local matching
        w_feat_select = batched_index_select(w_feat, dim=1, index=feat_indices)  # (32*12*128)
        w_feat_select = F.normalize(w_feat_select, dim=2)  # (32*12*128)

        refs = refs.view(b, n, c, h * w)
        ref_indices = []
        ref_selects = []
        for j in range(n):
            ref = refs[:, j, :, :]  # (32*128*64)
            w_ref = w_refs.view(b, c, n, h * w)[:, :, j, :]  # (32*128*64)
            fx = torch.matmul(w_feat_select, w_ref)  # (32*12*64)
            _, indice = torch.topk(fx, dim=2, k=1)
            indice = indice.squeeze(0).squeeze(-1)  # (32*10)
            select = batched_index_select(ref, dim=2, index=indice)  # (32*128*12)
            ref_indices.append(indice)
            ref_selects.append(select)
        ref_indices = torch.cat([item.unsqueeze(1) for item in ref_indices], dim=1)  # (32*2*12)
        ref_selects = torch.cat([item.unsqueeze(1) for item in ref_selects], dim=1)  # (32*2*128*12)

        # local replacement
        base_similarity = base_similarity.view(b, 1, 1)  # (32*1*1)
        ref_similarities = ref_similarities.view(b, 1, n)  # (32*1*2)
        feat_select = feat_select.view(b, 1, -1)  # (32*1*(128*12))
        ref_selects = ref_selects.view(b, n, -1)  # (32*2*(128*12))

        feat_fused = torch.matmul(base_similarity, feat_select) \
                     + torch.matmul(ref_similarities, ref_selects)  # (32*1*(128*12))
        feat_fused = feat_fused.view(b, c, num)  # (32*128*12)

        feat = batched_scatter(feat, dim=2, index=feat_indices, src=feat_fused)
        feat = feat.view(b, c, h, w)  # (32*128*8*8)

        return feat, feat_indices, ref_indices  # (32*128*8*8), (32*12), (32*2*12)


if __name__ == '__main__':
    config = {}
    model = Generator(config).cuda()
    x = torch.randn(32, 3, 3, 128, 128).cuda()
    y, sim = model(x)
    print(y.size())
