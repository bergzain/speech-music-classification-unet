#!/bin/bash

# make sure only first task per node installs stuff, others wait
DONEFILE="/tmp/install_done_${SLURM_JOBID}"
if [[ $SLURM_LOCALID == 0 ]]; then
  
  # put your install commands here:
    module load python/3.10.11
    pip install -r requirements.txt
  
  # Tell other tasks we are done installing
  touch "${DONEFILE}"
else
  # Wait until packages are installed
  while [[ ! -f "${DONEFILE}" ]]; do sleep 1; done
fi


srun -K \
--job-name="speech_music_classification" \
--gpus=1 \
--container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`" \ 
--container-image=/enroot/nvcr.io_nvidia_pytorch_24.06-py3.sqsh \
--container-workdir="`pwd`" \
-p batch \
--mem 64GB \
--gpus 1 \

# Define the paths to the Python training scripts
SCRIPTS=(
    "/home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py"
    "/home/zhazzouri/speech-music-classification-unet/Models/AttentionUNet/Code/train.py"
    "/home/zhazzouri/speech-music-classification-unet/Models/R2UNet/Code/train.py"
    "/home/zhazzouri/speech-music-classification-unet/Models/AttentionR2UNet/Code/train.py"
)
# Function to clear Python cache
"""
Clears the Python cache by recursively removing all "__pycache__" directories in the current working directory.
"""
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