#!/bin/bash
srun -K --job-name="speech_music_xai" --gpus=1 --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"$(pwd)":"$(pwd)" --container-image=/netscratch/zhazzouri/scripts/speech-music-xai.sqsh --container-workdir="$(pwd)" -p RTXA6000 --mem 64GB --gpus=1 --cpus-per-task=4 bash << 'EOF'
# Run xai.py script
python xai.py \
    --mlflow_dir /netscratch/zhazzouri/experiments/mlflow \
    --audio_dir /netscratch/zhazzouri/dataset/test/ \
    --sample_idx 0 \
    --save_global_path /netscratch/zhazzouri/experiments/cam_outputs/ \
    --path_type cluster
EOF