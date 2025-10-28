<div align="center">

# 🎬 Morphic Frames to Video

### High-quality video generation from image frames using Wan2.2

<div align="center">
  <a href="https://github.com/morphicfilms/MorphicFrames2Video.git">
    <img src="https://img.shields.io/badge/GitHub-Morphic%20Frames-black?logo=github" alt="GitHub">
  </a>
  <a href="https://huggingface.co/morphicmlteam/Wan2.2-I2V-A14B-frames">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Morphic%20Frames-blue" alt="Hugging Face">
  </a>
  <a href="https://www.morphic.com">
    <img src="https://img.shields.io/badge/Website-morphic.com-purple" alt="Website">
  </a>
</div>

</div>

## Showcase

## Showcase

| Video | Input Images & Prompt |
|-------|----------------------|
| <video src="https://github.com/user-attachments/assets/5d69f1c9-1cd6-4db7-9ab1-1a055e0a0804" controls width="640"></video> | <img src="examples/transition9_1.png" width="200"> → <img src="examples/transition9_2.png" width="200"><br><br>**Prompt:** "A clown, slowly transforms into a poster."<br><br>**Type:** Two-frame transition |
| <video src="https://github.com/user-attachments/assets/49019528-f5fc-4344-84a5-302e856dd770" controls width="640"></video> | <img src="examples/pink_1.png" width="150"> → <img src="examples/pink_2.png" width="150"> → <img src="examples/pink_3.png" width="150"> → <img src="examples/pink_4.png" width="150"><br><br>**Prompt:** "The animated girl rises up from her chair and waves hi to the camera as the camera zooms in."<br><br>**Type:** Multi-frame interpolation |
## Setting up the repository

First clone the Morphic Interpolation repo:

```
git clone https://github.com/morphicfilms/frames-to-video.git
```

To install the environment, we recommend following the [Wan2.2 installation guide](https://github.com/Wan-Video/Wan2.2).

Or you could alternatively run : `bash setup_env.sh` -> we recommend using the flash-attn version listed in the bash file for hassle free install.

## Downloading the weights

First:  download Wan2.2 I2V weights:

```
huggingface-cli download Wan-AI/Wan2.2-I2V-A14B --local-dir ./Wan2.2-I2V-A14B
```

Second : download the Morphic Frames to Video lora weights :

```
huggingface-cli download morphic/Wan2.2-frames-to-video --local-dir ./morphic-frames-lora-weights
```

## Running Frames to Video

For multi node run for 2 frame interpolation : 

```
torchrun --nproc_per_node=8 generate.py \
    --task i2v-A14B \
    --size 1280*720 \
    --frame_num 81 \
    --ckpt_dir ./Wan2.2-I2V-A14B-Interpolation \
    --high_noise_lora_weights_path ./morphic-frames-lora-weights/lora_interpolation_high_noise_final.safetensors \
    --dit_fsdp \
    --t5_fsdp \
    --ulysses_size 8 \
    --image examples/transition9_1.png \
    --prompt "Aa clown, slowly transforms into a poster." \
    --img_end examples/transition9_2.png \
```

For multi node run for multi frame interpolation : 
```
torchrun --nproc_per_node=8 generate.py \
    --task i2v-A14B \
    --size 1280*720 \
    --frame_num 81 \
    --ckpt_dir ./Wan2.2-I2V-A14B-Interpolation \
    --high_noise_lora_weights_path ./morphic-frames-lora-weights/lora_interpolation_high_noise_final.safetensors \
    --dit_fsdp \
    --t5_fsdp \
    --ulysses_size 8 \
    --image examples/pink_1.png \
    --prompt "The animated girl rises up from her chair and waves hi to the camera as the camer zooms in." \
    --img_end examples/pink_4.png \
    --middle_images examples/pink_2.png examples/pink_3.png \
    --middle_images_timestamps 0.4 0.7
```

Note:
1. --middle_images_timestamps : should be used if multiple intermediate frames are provided, the numbers indicate the the location the intermediate frame is provided (0.5 -> midway, 0.33, 0.66 -> 2 equally spaced intermediate frames, 0.25, 0.5, 0.75 -> 3 equally spaced intermediate frames)
2. Number of middle_images must be equal to number of middle_images_timestamps

## Acknowledgements

We would like to thank the Wan2.2 repo authors for the important research and open weights: [Wan2.2](https://github.com/Wan-Video/Wan2.2)

## Contact us 

You can reach out to us via : adithya.iyer@morphic.com
