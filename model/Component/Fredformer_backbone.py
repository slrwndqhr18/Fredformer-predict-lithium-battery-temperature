__all__ = ['FT_backbone']

# Cell
import torch
from torch import nn
from model.Component.RevIN import RevIN
from model.Component.cross_Transformer_nys import Trans_C as Trans_C_nys
from model.Component.cross_Transformer import Trans_C
from model.Component.Flatten_head import Flatten_Head

class Fredformer_backbone(nn.Module):
    def __init__(self, ablation:int, use_nys:int, cf_dim:int,cf_depth :int,cf_heads:int,cf_mlp:int,cf_head_dim:int,
                cf_drop:float,c_in:int, context_window:int, target_window:int, patch_len:int, stride:int,  d_model:int, 
                head_dropout = 0, padding_patch = None,individual = False, revin = True, affine = True, subtract_last = False, 
                **kwargs): #output:int
        
        super().__init__()
        self.use_nys = use_nys
        self.ablation = ablation
        # RevIn
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)
        # self.output = output
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.targetwindow=target_window
        self.horizon = self.targetwindow
        patch_num = int((context_window - patch_len)/stride + 1)
        self.norm = nn.LayerNorm(patch_len)
        #print("depth=",cf_depth)
        # Backbone 
        self.re_attn = True
        if self.use_nys==0:
            self.fre_transformer = Trans_C(dim = cf_dim,depth = cf_depth, heads = cf_heads, mlp_dim = cf_mlp, dim_head = cf_head_dim, dropout = cf_drop, patch_dim = patch_len*2 , horizon = self.horizon*2, d_model=d_model*2)
        else:
            self.fre_transformer = Trans_C_nys(dim = cf_dim,depth = cf_depth, heads = cf_heads, mlp_dim = cf_mlp, dim_head = cf_head_dim, dropout = cf_drop, patch_dim = patch_len*2 , horizon = self.horizon*2, d_model=d_model*2)
        
        
        # Head
        self.head_nf_f  = d_model * 2 * patch_num #self.horizon * patch_num#patch_len * patch_num
        self.n_vars = c_in
        self.individual = individual
        self.head_f1 = Flatten_Head(self.individual, self.n_vars, self.head_nf_f, target_window, head_dropout=head_dropout)
        self.head_f2 = Flatten_Head(self.individual, self.n_vars, self.head_nf_f, target_window, head_dropout=head_dropout)
        
        self.ircom = nn.Linear(self.targetwindow*2,self.targetwindow)
        self.rfftlayer = nn.Linear(self.targetwindow*2-2,self.targetwindow)
        self.final = nn.Linear(self.targetwindow*2,self.targetwindow)

        #break up R&I:
        self.get_r = nn.Linear(d_model*2,d_model*2)
        self.get_i = nn.Linear(d_model*2,d_model*2)
        self.output1 = nn.Linear(target_window,target_window)


        #ablation
        self.input = nn.Linear(c_in,patch_len*2)
        self.outpt = nn.Linear(d_model*2,c_in)
        self.abfinal = nn.Linear(patch_len*patch_num,target_window)
    
    def _layer_patch(self, _z):
        z1 = _z.real
        z2 = _z.imag
        # do patching
        z1 = z1.unfold(dimension=-1, size=self.patch_len, step=self.stride)                         # z1: [bs x nvars x patch_num x patch_len]
        z2 = z2.unfold(dimension=-1, size=self.patch_len, step=self.stride)                         # z2: [bs x nvars x patch_num x patch_len]  
        return z1, z2
    
    def _layer_revin(self, _z):
        _z = _z.permute(0,2,1)
        _z = self.revin_layer(_z, 'norm')
        _z = _z.permute(0,2,1)
        return _z

    def __layer_frequency_domain_modeling(self, _z): #handle DFT
        if self.revin: 
            _z = self._layer_revin(_z)
        
        _z = torch.fft.fft(_z)
        z1, z2 = self._layer_patch(_z) #(_z.real, _z.imag)                                                               

        #for channel-wise_1
        z1 = z1.permute(0,2,1,3)
        z2 = z2.permute(0,2,1,3)

        # model shape
        batch_size = z1.shape[0]
        patch_num  = z1.shape[1]
        input_size_encoder       = z1.shape[2]
        #patch_len  = z1.shape[3]
        
        #proposed
        z1 = torch.reshape(z1, (batch_size*patch_num,input_size_encoder,z1.shape[-1]))                            # z: [bs * patch_num,nvars, patch_len]
        z2 = torch.reshape(z2, (batch_size*patch_num,input_size_encoder,z2.shape[-1]))                            # z: [bs * patch_num,nvars, patch_len]
        return z1, z2, batch_size, patch_num, input_size_encoder

    def __layer_frequency_summarization(self, _z, batch_size, patch_num, input_size_encoder):
        z1 = self.get_r(_z)
        z2 = self.get_i(_z)
        
        z1 = torch.reshape(z1, (batch_size,patch_num,input_size_encoder,z1.shape[-1]))
        z2 = torch.reshape(z2, (batch_size,patch_num,input_size_encoder,z2.shape[-1]))
        
        z1 = self.head_f1(z1.permute(0,2,1,3) )                                                                    # z: [bs x nvars x pred_len] 
        z2 = self.head_f2(z2.permute(0,2,1,3))                                                                    # z: [bs x nvars x pred_len]
        
        _z = torch.fft.ifft(torch.complex(z1,z2))
        _z = self.ircom(torch.cat((_z.real,_z.imag),-1))

        # denorm
        if self.revin: 
            _z = _z.permute(0,2,1)
            _z = self.revin_layer(_z, 'denorm')
            _z = _z.permute(0,2,1)
        return _z

    def forward(self, z):                                                                   # z: [bs x nvars x seq_len]
        z1, z2, batch_size, patch_num, input_size_encoder = self.__layer_frequency_domain_modeling(z)
        z = self.fre_transformer(torch.cat((z1,z2),-1))
        z = self.__layer_frequency_summarization(z, batch_size, patch_num, input_size_encoder)
        return z