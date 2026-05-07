"""
screening.py

Minimal batch-run launcher. Edit field_ids and run: python screening.py.
Calls run_batch() from batch_pipeline.py, which handles model loading, two-stage
processing, CSV/PNG summary generation, and pickling of results.

"""
from batch_pipeline import run_batch

field_ids = [357,349,348,344,341,327,321,317,312,310,309,307,289,269,266,259,238,233,224,198,193,190,178,153,127,120,116,115,97,95,92,90,89,88,81,74,73,64,56,52,50,39,36,30,7,5,0,2064,29] 
run_batch(
    field_ids=field_ids,
    run_per_date=False,   
)