#!/bin/bash

echo "Downloading ASR model weights..."

wget -O nemo_experiments/Conformer-CTC-Char/2024-03-23_16-27-39/checkpoints/Conformer-CTC-Char--val_wer=0.2807-epoch=55-last.ckpt https://api.wandb.ai/artifactsV2/default/abdelrhman-elrawy/QXJ0aWZhY3Q6ODYzNzA1NzE1/8d9795e8a4f2b648d4b355c297384aab/Conformer-CTC-Char--val_wer%3D0.2807-epoch%3D55-last.ckpt

echo "Weights downloaded successfully."
