__all__ = ['PatchTST']

# Cell
from typing import Optional

from torch import nn, Tensor
from model.Component.Fredformer_backbone import Fredformer_backbone

class Fredformer(nn.Module):
    def __init__(self, _hyperParams, **kwargs):
        #Unused params in backbone
        # padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None,res_attention:bool=True, verbose:bool=False,
        # pe:str='zeros', store_attn:bool=False, d_v:Optional[int]=None, d_k:Optional[int]=None, max_seq_len:Optional[int]=1024,
        # pre_norm:bool=False, key_padding_mask:bool='auto', act:str="gelu", attn_dropout:float=0.3, norm:str='BatchNorm',
        super().__init__()
        self.model = Fredformer_backbone(
            c_in=_hyperParams["enc_in"], context_window = _hyperParams["input_len"], 
            target_window = _hyperParams["pred_len"], **_hyperParams)
        #Unused params in backbone
        # padding_var=padding_var, attn_mask=attn_mask, res_attention=res_attention, verbose=verbose, pe=pe, store_attn=store_attn, 
        # pre_norm=pre_norm, key_padding_mask=key_padding_mask, act=act, attn_dropout=attn_dropout, norm=norm, d_v=d_v, d_k=d_k,
        # max_seq_len=max_seq_len, n_layers=_hyperParams["enc_layers"],

    def forward(self, x): # x: [Batch, Input length, Channel]
        #outputs = self.Model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
        x = x.permute(0,2,1) # x: [Batch, Channel, Input length]
        x = self.model(x)
        x = x.permute(0,2,1) # x: [Batch, Input length, Channel]
        return x #,oz,t,attn
