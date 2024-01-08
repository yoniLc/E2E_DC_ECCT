"""
Deep Coding for Linear Block Error Correction
"""
import torch
import torch.nn as nn

def sign_to_bin(x):
    return 0.5 * (1 - x)

def bin_to_sign(x):
    return 1 - 2 * x

def diff_syndrome(H,x):
    tmp = bin_to_sign(H.unsqueeze(0)*x.unsqueeze(1))
    tmp = torch.prod(tmp,2)
    return sign_to_bin(tmp)

def diff_gener(G,m):
    tmp = bin_to_sign(G.unsqueeze(0)*m.unsqueeze(2))
    tmp = torch.prod(tmp,1)
    return sign_to_bin(tmp)

class Binarization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return ((input>=0)*1. - (input<0)*1.).float()
    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        return grad_output*(torch.abs(x)<=1)

class E2E_DC_ECC_Transformer(nn.Module):
    def __init__(self, args, decoder):
        super(E2E_DC_ECC_Transformer, self).__init__()
        ####
        self.args = args
        code = args.code
        self.n = code.n
        self.k = code.k
        self.bin = Binarization.apply
        with torch.no_grad():
            P_matrix = (torch.randint(0,2,(code.n-code.k,code.k))).float()
            P_matrix = bin_to_sign(P_matrix)*0.01
        self.P_matrix = nn.Parameter(P_matrix)
        # self.register_buffer('P_matrix', P_matrix)    
        self.register_buffer('I_matrix_H', torch.eye(code.n-code.k))
        self.register_buffer('I_matrix_G', torch.eye(code.k))
        #
        self.decoder = decoder
        ########
        
    def forward(self, m, z):
        x = diff_gener(self.get_generator_matrix(), m)
        x = bin_to_sign(x)
        z_mul = ((x+z) * x).detach()
        y = x*z_mul
        syndrome = bin_to_sign(diff_syndrome(self.get_pc_matrix(),sign_to_bin(self.bin(y))))
        magnitude = torch.abs(y)
        emb, loss, x_pred = self.decoder(magnitude, syndrome, self.get_pc_matrix(), z_mul, y, self.get_pc_matrix())
        return loss, x_pred, sign_to_bin(x)        
    
    def get_pc_matrix(self):
        bin_P =  sign_to_bin(self.bin(self.P_matrix))
        return torch.cat([self.I_matrix_H,bin_P],1)
    
    def get_generator_matrix(self,):
        bin_P =  sign_to_bin(self.bin(self.P_matrix))
        return torch.cat([bin_P,self.I_matrix_G],0).transpose(0,1)
    
############################################################
############################################################

if __name__ == '__main__':
    from DC_ECCT import DC_ECC_Transformer
    import numpy as np
    class Code():
        pass
    def EbN0_to_std(EbN0, rate):
        snr =  EbN0 + 10. * np.log10(2 * rate)
        return np.sqrt(1. / (10. ** (snr / 10.)))
    code = Code()
    code.k = 16
    code.n = 31
    
    args = Code()
    args.code = code
    args.d_model = 32
    args.h = 8
    args.N_dec = 2
    args.dropout_attn = 0
    args.dropout = 0
    
    bs = 1024
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = E2E_DC_ECC_Transformer(args, DC_ECC_Transformer(args)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


    EbNo_range_train = range(2, 8)
    std_train = torch.tensor([EbN0_to_std(ebn0, code.k / code.n) for ebn0 in EbNo_range_train]).float()

    m = torch.ones((bs, code.k)).long().to(device)
    H0 = model.get_pc_matrix().detach().clone()
    for iter in range(10000):
        model.zero_grad()
        stds = std_train[torch.randperm(bs)%len(std_train)]
        loss, x_pred, x = model(m, (torch.randn(bs,code.n)*stds.unsqueeze(-1)).to(device))
        loss.backward()
        optimizer.step()
        if iter%1000 == 0:
            print(f'iter {iter}: loss = {loss.item()} BER = {torch.mean((x_pred!=x).float()).item()} ||H_t-H0||_1 = {torch.sum((H0-model.get_pc_matrix()).abs())}')

        
