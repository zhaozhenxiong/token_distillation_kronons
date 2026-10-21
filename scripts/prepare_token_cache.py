#!/usr/bin/env python3
"""Explicit frozen tokenizer encoding with no optimizer initialization."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from token_distillation.config import load_config
from token_distillation.training.token_cache import prepare_cache
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--dataset',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--device',default='mps');a=p.parse_args()
    print(json.dumps(prepare_cache(load_config(a.config),dataset=a.dataset,output_dir=a.output_dir,device=a.device),indent=2))
