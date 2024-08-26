#!/bin/bash

srun -K \
--job-name="speech_music_classification" \
--gpus=1 \
--container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`" \
--container-image=/netscratch/zhazzouri/scripts/speech-music-classification.sqsh \
--container-workdir="`pwd`" \
-p RTXA6000 \
--mem 64GB \
--gpus 1 \
bash << EOF &> output.txt

# Define the paths to the Python training scripts
SCRIPTS=(
    "/home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py"
    "/home/zhazzouri/speech-music-classification-unet/Models/AttentionUNet/Code/train.py"
    "/home/zhazzouri/speech-music-classification-unet/Models/R2UNet/Code/train.py"
    "/home/zhazzouri/speech-music-classification-unet/Models/AttentionR2UNet/Code/train.py"
)

# Function to clear Python cache
clear_cache() {
    find . -type d -name "__pycache__" -exec rm -r {} +
    echo "Cache cleared."
}

# Run each script and clear cache afterwards
for script in "\${SCRIPTS[@]}"; do
    echo "Running script: \$script"
    python3 "\$script"
    echo "Script \$script completed."
    clear_cache
done
EOF


# #!/bin/bash srun -K --job-name="speech_music_classification" --gpus=1 --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`" --container-image=/netscratch/zhazzouri/scripts/speech-music-classification.sqsh --container-workdir="`pwd`" -p RTXA6000 --mem 64GB --gpus 1 bash -c 'SCRIPTS=("/home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py" "/home/zhazzouri/speech-music-classification-unet/Models/AttentionUNet/Code/train.py" "/home/zhazzouri/speech-music-classification-unet/Models/R2UNet/Code/train.py" "/home/zhazzouri/speech-music-classification-unet/Models/AttentionR2UNet/Code/train.py"); clear_cache() { find . -type d -name "__pycache__" -exec rm -r {} +; echo "Cache cleared."; }; for script in "${SCRIPTS[@]}"; do echo "Running script: $script"; python3 "$script"; echo "Script $script completed."; clear_cache; done' &> output.txt
# srun -K --job-name="speech_music_classification" --gpus=1 --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`" --container-image=/netscratch/zhazzouri/scripts/speech-music-classification.sqsh --container-workdir="`pwd`" -p RTXA6000 --mem 64GB --gpus 1 bash -c 'SCRIPTS=("/home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py" "/home/zhazzouri/speech-music-classification-unet/Models/R2UNet/Code/train.py"); clear_cache() { find . -type d -name "__pycache__" -exec rm -r {} +; echo "Cache cleared."; }; for script in "${SCRIPTS[@]}"; do echo "Running script: $script"; python3 "$script"; echo "Script $script completed."; clear_cache; done' &> output.txt