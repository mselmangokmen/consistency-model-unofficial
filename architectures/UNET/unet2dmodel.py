

from architectures.UNET.utils import POSITIONAL_TYPE

from diffusers import UNet2DModel 
import diffusers
import numpy as np 
def get_unet2dmodel(  img_channels=3,mult=[1,2,4,8],base_channels=64,
        num_res_blocks = 3, 
        dropout=0.3, 
        img_size=32,
        #num_classes=10,
        emb_type=POSITIONAL_TYPE,
        #attention_resolution = [4,8,16],
        groupnorm=32,
        #use_conv_in_res=False,
        #use_conv_up=False,
        #use_conv_down=False, 
        #use_new_attention_order=False,
        # use_flash_attention=False,  
        #use_scale_shift_norm=False
        ):
    diffusers
    block_out_channels= np.array(mult)* base_channels 
    return UNet2DModel(
    sample_size=img_size,   
    in_channels=img_channels,    
    out_channels=img_channels,  
    dropout=dropout,
     
    time_embedding_type=emb_type, 
    down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),  # Encoder blokları
    up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
    block_out_channels=block_out_channels,  # Blok çıkış kanalları
    layers_per_block=num_res_blocks,  # Her blok için katman sayısı 
    norm_num_groups=groupnorm,  # Normalizasyon grubu sayısı
)
