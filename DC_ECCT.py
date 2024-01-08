"""
Deep Coding for Linear Block Error Correction
"""
from torch.nn import LayerNorm
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy

def sign_to_bin(x):
    return 0.5 * (1 - x)

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        if N > 1:
            self.norm2 = LayerNorm(layer.size)

    def forward(self, x, mask):
        for idx, layer in enumerate(self.layers, start=1):
            x = layer(x, mask)
            if idx == len(self.layers)//2 and len(self.layers) > 1:
                x = self.norm2(x)
        return self.norm(x)


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, args, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.args = args
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        ###
        d_hidden = 50
        self.one_d_mapping = nn.Sequential(nn.Linear(1, d_hidden), nn.ReLU(),nn.Linear(d_hidden, 1))        
    
    def get_mask_from_pc_matrix(self, pc_matrix):
        mask_nk_nk = pc_matrix@pc_matrix.T
        mask_n_n = pc_matrix.T@pc_matrix
        tmp1 = torch.cat([mask_n_n,pc_matrix.T],1)
        tmp2 = torch.cat([pc_matrix,mask_nk_nk],1)
        return self.one_d_mapping(torch.cat([tmp1,tmp2],0).unsqueeze(-1)).squeeze().unsqueeze(0).unsqueeze(0)
    
    def forward(self, query, key, value, mask=None):
        nbatches = query.size(0)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = self.attention(query, key, value, mask=mask)

        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

    def attention(self, query, key, value, mask=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        scores = scores*self.get_mask_from_pc_matrix(mask)
        # scores.masked_fill_(torch.eye(scores.size(-1)).unsqueeze(0).unsqueeze(0).bool().to(scores.device),-1e9)
        p_attn = F.softmax(scores, dim=-1)
        if self.dropout is not None:
            p_attn = self.dropout(p_attn)
        return torch.matmul(p_attn, value), None

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.gelu(self.w_1(x))))

############################################################


class DC_ECC_Transformer(nn.Module):
    def __init__(self, args):
        super(DC_ECC_Transformer, self).__init__()
        ####
        self.args = args
        c = copy.deepcopy
        attn = MultiHeadedAttention(args.h, args.d_model, args, dropout=args.dropout_attn)
        ff = PositionwiseFeedForward(args.d_model, args.d_model*4, args.dropout)
        #
        self.src_embed_synd = torch.nn.Embedding(5, args.d_model)
        self.src_embed_magn = torch.nn.Parameter(torch.empty(
            (1, args.d_model)))
        self.decoder = Encoder(EncoderLayer(
            args.d_model, c(attn), c(ff), args.dropout), args.N_dec)
        self.oned_final_embed = torch.nn.Sequential(
            *[nn.Linear(args.d_model, 1)])
        self.synd_to_mag = nn.Linear(args.d_model, args.d_model)
        self.mag_to_mag = nn.Linear(args.d_model, args.d_model)

        for name, p in self.named_parameters():
            if p.dim() > 1 and 'mask_emb' not in name and 'src_embed_synd' not in name:
                nn.init.xavier_uniform_(p)
                

    def forward(self, magnitude, syndrome, mask, z_mul, y, dict_batch):
        emb_magn = self.src_embed_magn.unsqueeze(0) * magnitude.unsqueeze(-1)
        emb_synd = self.src_embed_synd(sign_to_bin(syndrome).long())
        emb = torch.cat([emb_magn, emb_synd], 1)
        emb = self.decoder(emb, mask)
        loss, x_pred = self.loss(-emb, z_mul, y, dict_batch)
        return emb, loss.unsqueeze(-1), x_pred

    def loss(self, z_pred, z2, y, pc_matrix):
        n_max = z2.size(1)
        tmp = z_pred[:,n_max:].unsqueeze(-1)*pc_matrix.unsqueeze(0).unsqueeze(2)
        z_pred = self.mag_to_mag(z_pred[:,:n_max])+self.synd_to_mag(tmp.permute(0,1,3,2)).sum(1)
        z_pred = self.oned_final_embed(z_pred).squeeze()
        #
        z_pred = z_pred[:,:z2.size(1)]

        loss = (F.binary_cross_entropy_with_logits(z_pred, sign_to_bin(torch.sign(z2)),reduction='none')).mean(-1).mean()
        x_pred = sign_to_bin(torch.sign(-z_pred * torch.sign(y)))
        return loss, x_pred
############################################################
############################################################

if __name__ == '__main__':
    pass
