# Wan2.2 based Transitions and Multi-frame Interpolation

To doenload the weights :

```
huggingface-cli download Morphic/Wan2.2-I2V-A14B-Intepolation --local-dir ./Wan2.2-I2V-A14B-Interpolation
```

For multi node run for 2 frame interpolation : 

```
torchrun --nproc_per_node=8 generate.py \
    --task i2v-A14B \
    --size 1280*720 \
    --frame_num 81 \
    --ckpt_dir ./Wan2.2-I2V-A14B-Interpolation \
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

