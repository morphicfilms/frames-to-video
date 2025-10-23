#!/bin/bash

# Image-to-Video Generation Script
# Generates a video from an input image using Wan2.2-I2V-A14B model

# two frame interpolation / transition
torchrun --nproc_per_node=8 generate.py \
    --task i2v-A14B \
    --size 1280*720 \
    --frame_num 81 \
    --ckpt_dir /data/adi_temp/MorphicVideo/Wan2.2-I2V-A14B \
    --high_noise_lora_weights_path lora_interpolation_high_noise_final.safetensors \
    --dit_fsdp \
    --t5_fsdp \
    --ulysses_size 8 \
    --image examples/transition9_1.png \
    --prompt "Aa clown, slowly transforms into a poster." \
    --img_end examples/transition9_2.png \



# multi-frame interpolation / transition
torchrun --nproc_per_node=8 generate.py \
    --task i2v-A14B \
    --size 1280*720 \
    --frame_num 81 \
    --ckpt_dir /data/adi_temp/MorphicVideo/Wan2.2-I2V-A14B \
    --high_noise_lora_weights_path lora_interpolation_high_noise_final.safetensors \
    --dit_fsdp \
    --t5_fsdp \
    --ulysses_size 8 \
    --image examples/pink_1.png \
    --prompt "The animated girl rises up from her chair and waves hi to the camera as the camer zooms in." \
    --img_end examples/pink_4.png \
    --middle_images examples/pink_2.png examples/pink_3.png \
    --middle_images_timestamps 0.4 0.7