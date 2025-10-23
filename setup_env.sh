#!/bin/bash

# install uv, activate the environment, and install the dependencies
uv venv

source .venv/bin/activate

uv pip install torch==2.5.0 torchvision --index-url https://download.pytorch.org/whl/cu121

uv pip install packaging ninja && uv pip install flash-attn==2.7.0.post2 --no-build-isolation 

uv pip install -r requirements.txt