python merge_lora_chaos.py --base_model "model/RWKV-5-World-3B-v2-20231113-ctx4096.pth"\
 --lora_alpha 16 \
 --lora_checkpoint "output/rwkv-22.pth"\
 --output "LoRAMerged.pth" \
 --r 0\
 --k 1\
 --v 1\

