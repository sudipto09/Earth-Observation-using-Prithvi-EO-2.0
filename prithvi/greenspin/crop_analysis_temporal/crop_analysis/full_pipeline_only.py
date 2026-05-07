"""
full_pipeline.py

Re-runs the full visualisation and export pipeline (dashboard, GeoTIFF, per-date
maps) on specific field IDs without re-running the batch screening step.
Useful for regenerating outputs after parameter or style changes.

"""



import torch

from batch_pipeline import screen_field, run_full_pipeline
from modelfactory import load_pipeline


target_fids = [29]   

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, _ = load_pipeline(device)

for fid in target_fids:
    print(f"\nFID {fid}")
    bundle = screen_field(fid, model, device)
    if bundle is None:
        print(f"  skipped")
        continue
    run_full_pipeline(bundle, device, run_per_date=True)