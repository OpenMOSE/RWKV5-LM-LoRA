This repo is forked from RWKV-LM-LoRA

# RWKV-5.2 and 6.0 LoRA Experiment Project RWKV5-LM-LoRA

2024.1.12 Added LoRA Trainer with Rocm5.6 Pytorch. 
  I tested training RWKV-5-World 7b-LoRA on 2 x AMD Instinct MI100
2024.03.25 Added v6 model support
  with emb(full params),output,gate LoRA training

We have added LoRA training functionality to the RWKV v5.2 model.

Now you can initiate LoRA training for the RWKV-5-World model.

in 7b model,training can be performed with 24GB of VRAM if less ctx,rank

The basic commands follow those of RWKV-LM-LoRA.

Examples of training commands can be found in lora-training.sh, so please make changes as needed.



# And Thanks to:
RWKV-LM @BlinkDL
RWKV-LM-LoRA @Blealtan



# License
same with RWKV-LM and RWKV-LM-LoRA

Apache 2.0


@ 2024 OpenMOSE
