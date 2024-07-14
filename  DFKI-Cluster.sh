
srun -K \
--job-name="speech_music_classification" \
--gpus=1 \
--container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`" \ 
--container-image=/enroot/nvcr.io_nvidia_pytorch_23.12-py3.sqsh \
--container-workdir="`pwd`" \

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
for script in "${SCRIPTS[@]}"; do
    echo "Running script: $script"
    srun python3 "$script"
    echo "Script $script completed."
    clear_cache
done