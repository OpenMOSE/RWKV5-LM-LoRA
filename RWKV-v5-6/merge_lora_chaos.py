from collections import OrderedDict
import os
import sys
from typing import Dict
import typing
import torch
from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("--lora_alpha", default=16, type=float)
parser.add_argument("--base_model", type=str)
parser.add_argument("--lora_checkpoint", type=str)
parser.add_argument("--output", type=str)

parser.add_argument("--r",default=1, type=int)
parser.add_argument("--k",default=1, type=int)
parser.add_argument("--v",default=1, type=int)

args = parser.parse_args()


#if sys.argv[1] == '--use-gpu':
#    device = 'cuda'
#    lora_alpha, base_model, lora, output = float(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
#else:
#    device = 'cpu'
#    lora_alpha, base_model, lora, output = float(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
device='cpu'
lora_alpha = args.lora_alpha
base_model = args.base_model
lora = args.lora_checkpoint
output = args.output

if args.r == 0:
    print('Receptance Merge Disabled')
if args.k == 0:
    print('Key Merge Disabled')
if args.v == 0:
    print('Value Merge Disabled')


with torch.no_grad():
    w: Dict[str, torch.Tensor] = torch.load(base_model, map_location='cpu')
    # merge LoRA-only slim checkpoint into the main weights
    w_lora: Dict[str, torch.Tensor] = torch.load(lora, map_location='cpu')
    for k in w_lora.keys():
        w[k] = w_lora[k]
    output_w: typing.OrderedDict[str, torch.Tensor] = OrderedDict()
    # merge LoRA weights
    keys = list(w.keys())
    for k in keys:
        if k.endswith('.weight'):
            prefix = k[:-len('.weight')]
            lora_A = prefix + '.lora_A'
            lora_B = prefix + '.lora_B'
            LoRAMerge = True
            if "receptance" in prefix and args.r == 0:
                LoRAMerge = False
            if "key" in prefix and args.k == 0:
                LoRAMerge = False
            if "value" in prefix and args.v == 0:
                LoRAMerge = False
            if lora_A in keys and LoRAMerge == True:
                assert lora_B in keys
                print(f'merging {lora_A} and {lora_B} into {k}')
                assert w[lora_B].shape[1] == w[lora_A].shape[0]
                lora_r = w[lora_B].shape[1]
                w[k] = w[k].to(device=device)
                w[lora_A] = w[lora_A].to(device=device)
                w[lora_B] = w[lora_B].to(device=device)
                w[k] += w[lora_B] @ w[lora_A] * (lora_alpha / lora_r)
                output_w[k] = w[k].to(device='cpu', copy=True)
                del w[k]
                del w[lora_A]
                del w[lora_B]
                continue
            else:
                output_w[k] = w[k].to(device='cpu', copy=True)

        if 'lora' not in k:
            print(f'retaining {k}')
            output_w[k] = w[k].clone()
            del w[k]

    torch.save(output_w, output)
