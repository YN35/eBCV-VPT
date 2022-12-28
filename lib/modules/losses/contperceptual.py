import torch
import torch.nn as nn

from taming.modules.losses.vqperceptual import *  # TODO: taming dependency yes/no?


class LPIPSWithDiscriminator(nn.Module):
    def __init__(self, disc_start, logvar_init=0.0, kl_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_loss="hinge"):

        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm
                                                 ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional
        
    def adopt_weight(self, global_step):
        if global_step < self.discriminator_iter_start:
            return 0.0
        else:
            return self.disc_factor

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, inputs, reconstructions, posteriors, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train",
                weights=None):
        
        b,c,t,h,w = inputs.shape
        inputs_img = inputs.reshape(b*t,c,h,w)
        reconstructions_img = reconstructions.reshape(b*t,c,h,w)
        
        if optimizer_idx == 0:
            
            # L1 loss of reconstruction
            rec_loss = torch.abs(inputs_img.contiguous() - reconstructions_img.contiguous())
            if self.perceptual_weight > 0:
                # mesure the similarity between the input and the reconstruction with LPIPS which trained network
                p_loss = self.perceptual_loss(inputs_img.contiguous(), reconstructions_img.contiguous())
                rec_loss = rec_loss + self.perceptual_weight * p_loss

            nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
            weighted_nll_loss = nll_loss
            if weights is not None:
                weighted_nll_loss = weights*nll_loss
            # 画像ごとの重み付き再構築ロス
            weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
            # weighted_nll_loss = torch.mean(weighted_nll_loss)
            # ただの再構築ロス 重みが無かったら上と下は全く同じ
            nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
            # nll_loss = torch.mean(nll_loss)
            # nll_loss = torch.mean(nll_loss)
            # zが正規分布に従うようにするためのKLロス(正則加工でμとlogvarの発散を防止) encoderを学習
            kl_loss = posteriors.kl()
            kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
            # kl_loss = torch.mean(kl_loss)

            # now the GAN part
            
            # self.discriminator_iter_startステップになるまではGANを使ったVAE学習を行わない
            disc_factor = self.adopt_weight(global_step)
            if self.disc_factor > 0.0 and disc_factor > 0.0:
                
                # generator update
                if cond is None:
                    assert not self.disc_conditional
                    # 画像を小さい領域に分割してdiscriminatorで生成された画像か判定 この場合偽物を入力しているため0に近い値が出るといい
                    logits_fake = self.discriminator(reconstructions_img.contiguous())
                else:
                    assert self.disc_conditional
                    logits_fake = self.discriminator(torch.cat((reconstructions_img.contiguous(), cond), dim=1))
                # discriminatorの出力が1に近い値になる(dicsをできるだけ騙せるようになる)ようにvae全体に更新を入れる
                g_loss = -torch.mean(logits_fake)
                
                try:
                    # discriminatorのを学習しないようにvae -> discriminatorの間の勾配を取り出している？
                    d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                except RuntimeError:# 評価時はrequires_gradがFalseになっているためエラーが出るナノでその時は無視
                    assert not self.training
                    d_weight = torch.tensor(0.0)
            else:
                d_weight = torch.tensor(0.0)
                g_loss = torch.tensor(0.0)

            # 重み付き再構築ロス + 洗剤変数発散防止klロス + GANのロス
            loss = weighted_nll_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss
            if torch.isnan(loss):
                print('detected nan')

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(), "{}/logvar".format(split): self.logvar.detach(),
                   "{}/kl_loss".format(split): kl_loss.detach().mean(), "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   }
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                # discriminatorのみ学習するためvae(generator)をdetachする
                logits_real = self.discriminator(inputs_img.contiguous().detach())
                logits_fake = self.discriminator(reconstructions_img.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs_img.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions_img.contiguous().detach(), cond), dim=1))

            disc_factor = self.adopt_weight(global_step)
            # fakeは-1に近づける realは1に近づける
            # exp)
            # self.disc_loss(torch.tensor(1.0),torch.tensor(-1.0))
            # -> tensor(0.)
            # self.disc_loss(torch.tensor(-1.0),torch.tensor(1.0))
            # -> tensor(2.)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            return d_loss, log

