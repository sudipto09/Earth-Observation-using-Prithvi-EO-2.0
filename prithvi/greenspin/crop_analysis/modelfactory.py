import torch
from torch import nn
from pathlib import Path
from prithvi_mae import PrithviMAE  

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
        x = x.reshape(B, self.T, self.P, D).permute(0, 2, 1, 3)        # (B, P, T, D)
        x = x.reshape(B * self.P, self.T, D)
        x, _ = self.temporal_attn(x, x, x)
        x = x.mean(dim=1).reshape(B, self.P, D).permute(0, 2, 1)       # (B, D, P)
        x = x.reshape(B, D, self.h, self.h)                             # (B, D, 14, 14)
        return self.decoder(x)                                           # (B, 1, 224, 224)


def load_pipeline(device):
    print("Loading Prithvi-EO-2.0-300M-TL from local weights...")

    # .pt weights file
    weight_files = list(WEIGHTS_DIR.glob("*.pt")) + list(WEIGHTS_DIR.glob("*.pth"))
    if not weight_files:
        raise FileNotFoundError(f"No .pt/.pth weight file found in {WEIGHTS_DIR}")
    weights_path = weight_files[0]
    print(f"  Using weights: {weights_path.name}")

    # Prithvi-300M config 
    model = PrithviMAE(
        img_size=224,
        patch_size=(1, 16, 16),
        num_frames=4,               #  4 timestamps
        in_chans=6,                 # 6 S2 bands
        embed_dim=1024,             # 300M model
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

   
    try:
        model.load_state_dict(state_dict, strict=False)
        print("  Weights loaded [OK]")
    except Exception as e:
        print(f"  Direct load failed ({e}), trying prefix strip...")
        state_dict = {k.replace("encoder.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        print("  Weights loaded with prefix strip [OK]")

    model = model.to(device).eval()

    embed_dim = 1024
    decoder = TemporalDecoder(embed_dim=embed_dim, num_timestamps=4).to(device).eval()
    return model, decoder