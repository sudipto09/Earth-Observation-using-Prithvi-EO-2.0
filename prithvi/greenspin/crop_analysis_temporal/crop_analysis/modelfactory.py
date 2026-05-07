"""
modelfactory.py

Loads Prithvi-EO-2.0-300M-TL weights from the prithvi_300m_tl/ folder.
Handles positional-embedding interpolation when the number of input dates
differs from the checkpoint. Returns (model, decoder) both in eval() mode.
"""

import torch
from torch import nn
from pathlib import Path
from prithvi_mae import PrithviMAE  
from config import TEMPORAL_REPEATS

WEIGHTS_DIR = Path(__file__).parent / "prithvi_300m_tl"


class TemporalDecoder(nn.Module):
    def __init__(self, embed_dim, num_timestamps=4, patch_size=16, img_size=224):
        super().__init__()
        self.T = num_timestamps
        self.h = img_size // patch_size             # 14
        self.P = self.h ** 2                        # 196

        self.temporal_attn = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 128, 4, stride=2, padding=1),  
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),         
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),          
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),          
            nn.ReLU(),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, tokens):
        B, _, D = tokens.shape
        x = tokens[:, 1 : self.T * self.P + 1]                          # skip CLS token
        x = x.reshape(B, self.T, self.P, D).permute(0, 2, 1, 3)        
        x = x.reshape(B * self.P, self.T, D)
        x, _ = self.temporal_attn(x, x, x)
        x = x.mean(dim=1).reshape(B, self.P, D).permute(0, 2, 1)      
        x = x.reshape(B, D, self.h, self.h)                             
        return self.decoder(x)                                         


def load_pipeline(device):
    print("Loading Prithvi-EO-2.0-300M-TL from local weights...")

    # .pt weights file
    weight_files = list(WEIGHTS_DIR.glob("*.pt")) + list(WEIGHTS_DIR.glob("*.pth"))
    if not weight_files:
        raise FileNotFoundError(f"No .pt/.pth weight file found in {WEIGHTS_DIR}")
    weights_path = weight_files[0]
    print(f"Using weights: {weights_path.name}")

    # Prithvi-300M config 
    model = PrithviMAE(
        img_size=224,
        patch_size=(1, 16, 16),
        num_frames=TEMPORAL_REPEATS,    
        in_chans=6,                     
        embed_dim=1024,                 
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        encoder_only=True,          
    )

    # load weights
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("model", checkpoint)  # handle wrapped checkpoints

  
    has_encoder_prefix = any(k.startswith("encoder.") for k in state_dict)
    if not has_encoder_prefix:
        
        state_dict = {"encoder." + k: v for k, v in state_dict.items()}
        print(" Added 'encoder.' prefix to bare-encoder checkpoint keys")

    
    import torch.nn.functional as F
    PE_KEY = "encoder.pos_embed"
    if PE_KEY in state_dict:
        ckpt_pe= state_dict[PE_KEY]              
        model_pe = model.encoder.pos_embed         
        if ckpt_pe.shape != model_pe.shape:
            D = ckpt_pe.shape[-1]
            P  = 196                           
            cls_tok = ckpt_pe[:, :1, :]             
            patches= ckpt_pe[:, 1:, :]              
            T_ckpt = patches.shape[1] // P
            T_model = (model_pe.shape[1] - 1) // P
            
            patches = patches.reshape(1, T_ckpt, P, D).permute(0, 3, 2, 1).float()
            patches = F.interpolate(patches, size=(P, T_model),mode='bilinear', align_corners=False)
            patches = patches.permute(0, 3, 2, 1).reshape(1, T_model * P, D)
            state_dict[PE_KEY] = torch.cat([cls_tok, patches], dim=1)
            

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    
    real_missing = [k for k in missing if not k.startswith("decoder.")]
    if real_missing:
        print(f"  WARNING: {len(real_missing)} encoder keys not in checkpoint: "
              f"{real_missing[:5]}{'...' if len(real_missing)>5 else ''}")
    print("  Weights loaded ")

    model = model.to(device).eval()

    embed_dim = 1024
    decoder = TemporalDecoder(
        embed_dim=embed_dim,
        num_timestamps=TEMPORAL_REPEATS,   
    ).to(device).eval()
    return model, decoder