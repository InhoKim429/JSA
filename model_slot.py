import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import copy
import resnet
import random
from einops import rearrange, repeat, einsum

import matplotlib.pyplot as plt
import cv2
import utils

# https://github.com/lucidrains/slot-attention/blob/master/slot_attention/slot_attention.py
# https://github.com/evelinehong/slot-attention-pytorch/blob/master/model.py
class SlotAttention(nn.Module):
    def __init__(self, num_slots, infer_sharpening, mask_ratio, iters):
        super().__init__()
        self.num_slots = num_slots
        self.infer_sharpening = infer_sharpening
        self.mask_ratio = mask_ratio
        self.iters = iters
        self.eps = 1e-8
        self.scale = 512 ** -0.5

        # Learnable Query
        self.slots = nn.Parameter(torch.randn(1, num_slots, 512))

        # Learnable Mask Tokens
        self.mask_token_img = nn.Parameter(torch.randn(1, 1, 512))
        self.mask_token_aud = nn.Parameter(torch.randn(1, 1, 512))
        
        # Vision
        self.img_to_q = nn.Linear(512, 512)
        self.img_to_k = nn.Linear(512, 512)
        self.img_to_v = nn.Linear(512, 512)
        self.img_gru = nn.GRUCell(512, 512)
        self.img_mlp = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512)
        )
        self.img_norm_input  = nn.LayerNorm(512)
        self.img_norm_slots  = nn.LayerNorm(512)
        self.img_norm_pre_ff = nn.LayerNorm(512)

        # Audio
        self.aud_to_q = nn.Linear(512, 512)
        self.aud_to_k = nn.Linear(512, 512)
        self.aud_to_v = nn.Linear(512, 512)
        self.aud_gru = nn.GRUCell(512, 512)
        self.aud_mlp = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512)
        )

        self.aud_norm_input  = nn.LayerNorm(512)
        self.aud_norm_slots  = nn.LayerNorm(512)
        self.aud_norm_pre_ff = nn.LayerNorm(512)
        return

    def forward(self, img, aud):
        # Masking img patches
        b, n_img_patches, c = img.size()
        _, n_aud_patches, _ = aud.size()
        device = img.device

        if self.training:
            # Masking patches and chunks
            img_mask = torch.rand(b, n_img_patches, 1, device=device) < self.mask_ratio
            img_mask = img_mask.to(dtype=img.dtype)
            mask_token_img = self.mask_token_img.expand(b, n_img_patches, c)
            img = img * (1 - img_mask) + mask_token_img * img_mask

            aud_mask = torch.rand(b, n_aud_patches, 1, device=device) < self.mask_ratio
            aud_mask = aud_mask.to(dtype=aud.dtype)
            mask_token_aud = self.mask_token_aud.expand(b, n_aud_patches, c)
            aud = aud * (1 - aud_mask) + mask_token_aud * aud_mask

        original_slots = self.slots.expand(b, -1, -1)  # b x n x c

        img, aud = self.img_norm_input(img), self.aud_norm_input(aud)
        img_k, img_v = self.img_to_k(img), self.img_to_v(img)
        aud_k, aud_v = self.aud_to_k(aud), self.aud_to_v(aud)

        img_slots = None
        for iter in range(self.iters):
            if img_slots is None:
                img_slots = original_slots
            slots_prev = img_slots
            img_slots = self.img_norm_slots(img_slots)
            img_q = self.img_to_q(img_slots)

            dots = torch.einsum('bid,bjd->bij', img_q, img_k) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)

            cross_dots = torch.einsum('bid,bjd->bij', img_q, aud_k) * self.scale
            cross_attn = cross_dots.softmax(dim=1) + self.eps
            cross_attn = cross_attn / cross_attn.sum(dim=-1, keepdim=True)
            imgq_imgk_attn, imgq_audk_attn = attn, cross_attn

            updates = torch.einsum('bjd,bij->bid', img_v, attn)
            img_slots = self.img_gru(updates.reshape(-1, c), slots_prev.reshape(-1, c))
            img_slots = img_slots.reshape(b, -1, c)
            img_slots = img_slots + self.img_mlp(self.img_norm_pre_ff(img_slots))

        aud_slots = None
        for iter in range(self.iters):
            if aud_slots is None:
                aud_slots = original_slots
            slots_prev = aud_slots
            aud_slots = self.aud_norm_slots(aud_slots)
            aud_q = self.aud_to_q(aud_slots)

            dots = torch.einsum('bid,bjd->bij', aud_q, aud_k) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)
            updates = torch.einsum('bjd,bij->bid', aud_v, attn)

            cross_dots = torch.einsum('bid,bjd->bij', aud_q, img_k) * self.scale
            cross_attn = cross_dots.softmax(dim=1) + self.eps
            cross_attn = cross_attn / cross_attn.sum(dim=-1, keepdim=True)
            audq_audk_attn, audq_imgk_attn = attn, cross_attn

            aud_slots = self.aud_gru(updates.reshape(-1, c), slots_prev.reshape(-1, c))
            aud_slots = aud_slots.reshape(b, -1, c)
            aud_slots = aud_slots + self.aud_mlp(self.aud_norm_pre_ff(aud_slots))
        
        return img_slots, aud_slots, img_q, aud_q, imgq_imgk_attn, imgq_audk_attn, audq_audk_attn, audq_imgk_attn
    
    def get_cross_attn(self, img, aud):
        b, _, c = img.size()
        img = self.img_norm_input(img)
        img_k, img_v = self.img_to_k(img), self.img_to_v(img)
        aud = self.aud_norm_input(aud)
        aud_k, aud_v = self.aud_to_k(aud), self.aud_to_v(aud)

        original_slots = self.slots.expand(b, -1, -1)  # b x n x c
        
        img_slots = None
        for iter in range(self.iters):
            if img_slots is None:
                img_slots = original_slots
            slots_prev = img_slots
            img_slots = self.img_norm_slots(img_slots)
            img_q = self.img_to_q(img_slots)

            img_dots = torch.einsum('bid,bjd->bij', img_q, img_k) * (self.infer_sharpening * self.scale)
            img_attn = img_dots.softmax(dim=1) + self.eps
            img_attn = img_attn / img_attn.sum(dim=-1, keepdim=True)
            
            dots = torch.einsum('bid,bjd->bij', img_q, img_k) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum('bjd,bij->bid', img_v, attn)
            img_slots = self.img_gru(updates.reshape(-1, c), slots_prev.reshape(-1, c))
            img_slots = img_slots.reshape(b, -1, c)
            img_slots = img_slots + self.img_mlp(self.img_norm_pre_ff(img_slots))

        aud_slots = None
        for iter in range(self.iters):
            if aud_slots is None:
                aud_slots = original_slots
            slots_prev = aud_slots
            aud_slots = self.aud_norm_slots(aud_slots)
            aud_q = self.aud_to_q(aud_slots)

            cross_dots = torch.einsum('bid,bjd->bij', aud_q, img_k) * (self.infer_sharpening * self.scale)
            cross_attn = cross_dots.softmax(dim=1) + self.eps
            cross_attn = cross_attn / cross_attn.sum(dim=-1, keepdim=True)

            dots = torch.einsum('bid,bjd->bij', aud_q, aud_k) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum('bjd,bij->bid', aud_v, attn)
            aud_slots = self.aud_gru(updates.reshape(-1, c), slots_prev.reshape(-1, c))
            aud_slots = aud_slots.reshape(b, -1, c)
            aud_slots = aud_slots + self.aud_mlp(self.aud_norm_pre_ff(aud_slots))

        return img_attn, cross_attn, img_q, aud_q, img_k, img_v

class PositionEmbeddingLearned2D(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, w, h, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(h, num_pos_feats)
        self.col_embed = nn.Embedding(w, num_pos_feats)
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x):
        h, w = x.shape[-3:-1]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        pos = rearrange(pos, 'b ... d -> b (...) d')

        return pos

class MlpDecoder(nn.Module):
    def __init__(self, num_patches_h, num_patches_w, slot_dim, feat_dim, normalizer='softmax') -> None:
        super().__init__()
        self.width = num_patches_w
        self.height = num_patches_h
        self.pos_emb = PositionEmbeddingLearned2D(num_patches_w, num_patches_h, slot_dim//2)
        self.alpha_holder = nn.Identity()
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, slot_dim),
            nn.ReLU(),
            nn.Linear(slot_dim, slot_dim),
            nn.ReLU(),
            nn.Linear(slot_dim, slot_dim),
            nn.ReLU(),
            nn.Linear(slot_dim, feat_dim+1)
        )
        self.normalizer = {
            'softmax': F.softmax,
        }[normalizer]

        return

    def forward(self, slots):
        slots = repeat(slots, 'b s d -> b s h w d', h=self.height, w=self.width)
        slots = rearrange(slots, 'b s h w d -> b s (h w) d') + self.pos_emb(slots[:, 0, :, :, :]).unsqueeze(1)
        feat_decode = self.mlp(slots)
        feat, alpha = feat_decode[:, :, :, :-1], feat_decode[:, :, :, -1]
        alpha = self.alpha_holder(self.normalizer(alpha, dim=1))
        """
        idx = torch.argmax(alpha,dim=1, keepdims=True)
        alpha_mask = torch.zeros_like(alpha).scatter_(1,idx,1.0)
        recon = einsum(feat, alpha_mask, 'b s hw d, b s hw -> b hw d')
        """
        recon = einsum(feat, alpha, 'b s hw d, b s hw -> b hw d')
        return recon

class mymodel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.tau = args.tau
        self.k = args.reciprocal_k

        self.num_slots = args.num_slots
        self.infer_sharpening = args.infer_sharpening
        self.mask_ratio = args.mask_ratio
        self.iters = args.iters

        # Student
        self.imgnet = resnet.resnet(True, 'vision', 0.0, args.out_dim, 2)
        self.audnet = resnet.resnet(False, 'audio', 0.0, args.out_dim, 2)
        self.slot_attn = SlotAttention(num_slots=self.num_slots,
                                       infer_sharpening=self.infer_sharpening,
                                       mask_ratio=self.mask_ratio,
                                       iters=self.iters)
        
        # Auxiliary decoder
        self.img_decoder = MlpDecoder(7, 7, 512, 512)
        self.aud_decoder = MlpDecoder(1, 16, 512, 512)

        self.CELoss = nn.CrossEntropyLoss()
        self.L1Loss = nn.L1Loss()
        self.MSELoss = nn.MSELoss()
        self.mode = 'train'
        return

    def get_index(self, label, key):
        for idx, cla in enumerate(label):
            if key in cla:
                print(idx, cla)
        return

    def get_reciprocal_label(self, reciprocal, label, idx):
        for i in range(len(reciprocal[idx])):
            if reciprocal[idx, i] == 1:
                print(i, label[i])
        return
    
    def cosine_loss(self, slots):
        B, i, c = slots.size()
        mask = (1 - torch.eye(i)).unsqueeze(0).expand(B, -1, -1).to(slots.device)
        sim_matrix = torch.einsum('bic,bjc->bij', slots, slots)
        
        diff_slot_sim = sim_matrix * mask
        diff_slot_sim = torch.relu(diff_slot_sim)
        return torch.sum(diff_slot_sim) / (B * i * (i - 1))

    def calculate(self, img_slots, aud_slots):
        B, i, c = img_slots.size()
        aud_B, aud_t, aud_c = aud_slots.size()
        assert (B == aud_B and c == aud_c)

        labels = torch.arange(B).long().to(img_slots.device)

        sim = torch.einsum('nic,mjc->nmij', img_slots, aud_slots)
        first_sim = sim[:, :, 0, 0]

        _, _, reciprocal = utils.get_potential_false_negative(img_slots.detach(), aud_slots.detach(), k=self.k)
        first_sim = first_sim.masked_fill(reciprocal == False, -float('inf'))

        infoNCE = self.CELoss(first_sim/self.tau, labels) + \
                  self.CELoss(first_sim.permute(1,0)/self.tau, labels)

        div_loss = self.cosine_loss(img_slots) + self.cosine_loss(aud_slots)
        return infoNCE, div_loss
    
    def get_sim(self, query, key):
        dots = torch.einsum('bid,bjd->bij', query, key) * (10 * (512 ** -0.5))
        dots = dots[:, 0, :]
        dots = torch.softmax(dots, dim=1)
        return dots
    
    def forward_train(self, frame, spec):
        # encoding
        img = self.imgnet(frame) # b x c x h x w
        aud = self.audnet(spec)  # b x c x t

        b, c, h, w = img.size()
        aud_B, aud_c, aud_t = aud.size()
        assert (b == aud_B and c == aud_c)
        
        # reshaping for slot
        img = torch.reshape(img, (b, c, h*w))
        img = torch.permute(img, (0, 2, 1)) # b x (h x w) x c
        aud = torch.permute(aud, (0, 2, 1)) # b x t x c

        # get slot attn
        img_slots, aud_slots, _, _, imgq_imgk_attn, imgq_audk_attn, audq_audk_attn, audq_imgk_attn = self.slot_attn(img, aud)

        img_recon = self.img_decoder(img_slots)
        aud_recon = self.aud_decoder(aud_slots)
        aud_recon = aud_recon.flatten(start_dim=2)

        img_slots = nn.functional.normalize(img_slots, dim=2)
        aud_slots = nn.functional.normalize(aud_slots, dim=2)
            
        att_loss = self.MSELoss(audq_imgk_attn[:, 0, :], imgq_imgk_attn[:, 0, :].detach()) + \
                   self.MSELoss(imgq_audk_attn[:, 0, :], audq_audk_attn[:, 0, :].detach())
        # audq_imgk_attn = b x num_slots x (h x w) size. (h x w) = 49.
        # imgq_audk_attn = b x num_slots x t size. t = 16

        recon_loss = self.MSELoss(img_recon, img.detach()) + self.MSELoss(aud_recon, aud.detach())
        
        info_loss, div_loss = self.calculate(img_slots, aud_slots)
        return info_loss, recon_loss, div_loss, att_loss
    
    def forward_eval(self, image, audio):
        with torch.no_grad():
            img = self.imgnet(image) # b x c x h x w
            aud = self.audnet(audio) # b x c x t

            b, c, h, w = img.size()
            aud_B, aud_c, aud_t = aud.size()
            assert (b == aud_B and c == aud_c)
            
            img = torch.reshape(img, (b, c, h*w))
            img = torch.permute(img, (0, 2, 1)) # b x (h x w) x c
            aud = torch.permute(aud, (0, 2, 1)) # b x t x c

            _, _, img_q, aud_q, img_k, img_v = self.slot_attn.get_cross_attn(img, aud) # b x 2 x 196

        img_dots = torch.einsum('bid,bjd->bij', img_q, img_k) * (self.infer_sharpening * (512 ** -0.5))
        img_attn = img_dots.softmax(dim=1) + 1e-8
        img_attn = img_attn / img_attn.sum(dim=-1, keepdim=True)

        cross_dots = torch.einsum('bid,bjd->bij', aud_q, img_k) * (self.infer_sharpening * (512 ** -0.5))
        cross_attn = cross_dots.softmax(dim=1) + 1e-8
        cross_attn = cross_attn / cross_attn.sum(dim=-1, keepdim=True)
            
        img_attn = torch.reshape(img_attn, (image.size(0), self.num_slots, 7, 7))
        img_attn = img_attn[:, 0, ...]
        img_attn = torch.unsqueeze(img_attn, dim=1)

        cross_attn = torch.reshape(cross_attn, (image.size(0), self.num_slots, 7, 7))
        cross_attn = cross_attn[:, 0, ...]
        cross_attn = torch.unsqueeze(cross_attn, dim=1)
        return img_attn, cross_attn
        
    def forward(self, frame, spec):
        if self.mode == 'train':
            return self.forward_train(frame, spec)
        else:
            return self.forward_eval(frame, spec)
        
    def get_slot(self, frame, spec):
        with torch.no_grad():
            img = self.imgnet(frame) # b x c x h x w
            aud = self.audnet(spec)  # b x c x t

            b, c, h, w = img.size()
            aud_B, aud_c, aud_t = aud.size()
            assert (b == aud_B and c == aud_c)
            
            img = torch.reshape(img, (b, c, h*w))
            img = torch.permute(img, (0, 2, 1)) # b x (h x w) x c
            aud = torch.permute(aud, (0, 2, 1)) # b x t x c

            img_slots, aud_slots, _, _, _, _, _, _ = self.slot_attn(img, aud)
            return img_slots, aud_slots
    
    
    def train(self, mode=True):
        super().train(mode)
        self.mode = 'train'
        return
    
def eval(self):
        def set_bn_to_eval(m): 
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                m.eval()
        self.apply(set_bn_to_eval)
        super().eval()
        self.mode = 'eval'
        return
