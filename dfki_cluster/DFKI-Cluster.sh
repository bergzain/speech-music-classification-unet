#!/bin/bash

srun -K \
--job-name="speech_music_classification" \
--gpus=1 \
--container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`" \
--container-image=/netscratch/zhazzouri/scripts/speech-music-xai.sqsh \
--container-workdir="`pwd`" \
-p RTXA6000 \
--mem 64GB \
--gpus 1 \
--cpus-per-task=4 \
bash << EOF &> output.txt
# Define the paths to the Python training scripts and their parameters
SCRIPTS=(
    ##group 1
    "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 5 --batch_size 64 --type_of_transformation MFCC"
    "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 5 --batch_size 64 --type_of_transformation delta"
    "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 5 --batch_size 64 --type_of_transformation delta-delta "
    "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 5 --batch_size 64 --type_of_transformation LFCC "
    "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 5 --batch_size 64 --type_of_transformation lfcc-delta "
    "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 5 --batch_size 64 --type_of_transformation lfcc-delta-delta"
    ##group 2
    "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 30 --batch_size 64 --type_of_transformation MFCC"
    "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 30 --batch_size 64 --type_of_transformation delta"
    "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 30 --batch_size 64 --type_of_transformation delta-delta "
    "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 30 --batch_size 64 --type_of_transformation LFCC "
    "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 30 --batch_size 64 --type_of_transformation lfcc-delta "
    "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 30 --batch_size 64 --type_of_transformation lfcc-delta-delta"
    ##group 3

    "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 80 --length_in_seconds 5 --batch_size 64 --type_of_transformation MFCC "
    "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 80 --length_in_seconds 5 --batch_size 64 --type_of_transformation delta "
    "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 80 --length_in_seconds 5 --batch_size 64 --type_of_transformation delta-delta "
    "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 80 --length_in_seconds 5 --batch_size 64 --type_of_transformation LFCC "
    "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 80 --length_in_seconds 5 --batch_size 64 --type_of_transformation lfcc-delta "
    "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 80 --length_in_seconds 5 --batch_size 64 --type_of_transformation lfcc-delta-delta "
    ##group 4
    "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model AttentionUNet --n_mfcc 32 --length_in_seconds 5 --batch_size 64 --type_of_transformation MFCC "
    "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model AttentionUNet --n_mfcc 32 --length_in_seconds 5 --batch_size 64 --type_of_transformation delta "
    "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model AttentionUNet --n_mfcc 32 --length_in_seconds 5 --batch_size 64 --type_of_transformation delta-delta "
    "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model AttentionUNet --n_mfcc 32 --length_in_seconds 5 --batch_size 64 --type_of_transformation LFCC "
    "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model AttentionUNet --n_mfcc 32 --length_in_seconds 5 --batch_size 64 --type_of_transformation lfcc-delta "
    "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model AttentionUNet --n_mfcc 32 --length_in_seconds 5 --batch_size 64 --type_of_transformation lfcc-delta-delta "
    ##group 5
    "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model R2AttU_Net --n_mfcc 32 --length_in_seconds 5 --batch_size 64 --type_of_transformation MFCC "
    "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model R2AttU_Net --n_mfcc 32 --length_in_seconds 5 --batch_size 64 --type_of_transformation delta "
    "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model R2AttU_Net --n_mfcc 32 --length_in_seconds 5 --batch_size 64 --type_of_transformation delta-delta "
    "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model R2AttU_Net --n_mfcc 32 --length_in_seconds 5 --batch_size 64 --type_of_transformation LFCC "
    "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model R2AttU_Net --n_mfcc 32 --length_in_seconds 5 --batch_size 64 --type_of_transformation lfcc-delta "
    "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model R2AttU_Net --n_mfcc 32 --length_in_seconds 5 --batch_size 64 --type_of_transformation lfcc-delta-delta "
    ##group 6
    "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model R2U_Net --n_mfcc 32 --length_in_seconds 5 --batch_size 64 --type_of_transformation MFCC "
    "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model R2U_Net --n_mfcc 32 --length_in_seconds 5 --batch_size 64 --type_of_transformation delta "
    "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model R2U_Net --n_mfcc 32 --length_in_seconds 5 --batch_size 64 --type_of_transformation delta-delta "
    "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model R2U_Net --n_mfcc 32 --length_in_seconds 5 --batch_size 64 --type_of_transformation LFCC "
    "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model R2U_Net --n_mfcc 32 --length_in_seconds 5 --batch_size 64 --type_of_transformation lfcc-delta "
    "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model R2U_Net --n_mfcc 32 --length_in_seconds 5 --batch_size 64 --type_of_transformation lfcc-delta-delta "

    ## group 7
    python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 10 --batch_size 64 --type_of_transformation MFCC
    python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 10 --batch_size 64 --type_of_transformation delta
    python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 10 --batch_size 64 --type_of_transformation delta-delta
    python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 10 --batch_size 64 --type_of_transformation LFCC
    python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 10 --batch_size 64 --type_of_transformation lfcc-delta
    python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 10 --batch_size 64 --type_of_transformation lfcc-delta-delta

    ## group 8
    python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 15 --batch_size 64 --type_of_transformation MFCC
    python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 15 --batch_size 64 --type_of_transformation delta
    python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 15 --batch_size 64 --type_of_transformation delta-delta
    python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 15 --batch_size 64 --type_of_transformation LFCC    
    python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 15 --batch_size 64 --type_of_transformation lfcc-delta
    python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 15 --batch_size 64 --type_of_transformation lfcc-delta-delta

    ## group 9
    python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 3 --batch_size 64 --type_of_transformation MFCC
    python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 3 --batch_size 64 --type_of_transformation delta
    python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 3 --batch_size 64 --type_of_transformation delta-delta
    python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 3 --batch_size 64 --type_of_transformation LFCC
    python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 3 --batch_size 64 --type_of_transformation lfcc-delta
    python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 3 --batch_size 64 --type_of_transformation lfcc-delta-delta

    ## group 10 , 2 seconds
    python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 2 --batch_size 64 --type_of_transformation MFCC
    python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 2 --batch_size 64 --type_of_transformation delta
    python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 2 --batch_size 64 --type_of_transformation delta-delta
    python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 2 --batch_size 64 --type_
    python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 2 --batch_size 64 --type_of_transformation LFCC
    python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 2 --batch_size 64 --type_of_transformation lfcc-delta
    python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 2 --batch_size 64 --type_of_transformation lfcc-delta-delta

    ## group 11 , 1 seconds
    python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 1 --batch_size 64 --type_of_transformation MFCC
    python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 1 --batch_size 64 --type_of_transformation delta
    python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 1 --batch_size 64 --type_of_transformation delta-delta
    python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 1 --batch_size 64 --type_of_transformation LFCC
    python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 1 --batch_size 64 --type_of_transformation lfcc-delta
    python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 1 --batch_size 64 --type_of_transformation lfcc-delta-delta

    ## group 12 , 0.5 seconds
    python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 0.5 --batch_size 64 --type_of_transformation MFCC
    python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 0.5 --batch_size 64 --type_of_transformation delta
    python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 0.5 --batch_size 64 --type_of_transformation delta-delta
    python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 0.5 --batch_size 64 --type_of_transformation LFCC
    python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 0.5 --batch_size 64 --type_of_transformation lfcc-delta
    python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 0.5 --batch_size 64 --type_of_transformation lfcc-delta-delta




)# Function to clear Python cache
clear_cache() {
    find . -type d -name "__pycache__" -exec rm -r {} +
    echo "Cache cleared."
}

# Run each script and clear cache afterwards \ 
for script in "\${SCRIPTS[@]}"; do
    echo "Running script: \$script"
    eval \$script
    if [ \$? -ne 0 ]; then
        echo "Script \$script failed. Check error.txt for details." >> error.txt
    else
        echo "Script \$script completed successfully." >> output.txt
    fi
    clear_cache
done
EOF


#!/bin/bash

# Group 1
srun -K --job-name="speech_music_classification_group_1" --gpus=1 --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`" --container-image=/netscratch/zhazzouri/scripts/speech-music-xai.sqsh --container-workdir="`pwd`" -p RTXA6000 --mem 64GB --gpus 1 --cpus-per-task=4 bash -c 'SCRIPTS=("python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 5 --batch_size 64 --type_of_transformation MFCC" "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 5 --batch_size 64 --type_of_transformation delta" "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 5 --batch_size 64 --type_of_transformation delta-delta" "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 5 --batch_size 64 --type_of_transformation LFCC" "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 5 --batch_size 64 --type_of_transformation lfcc-delta" "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 5 --batch_size 64 --type_of_transformation lfcc-delta-delta"); for script in "${SCRIPTS[@]}"; do echo "Running script: $script"; eval $script; if [ $? -ne 0 ]; then echo "Script $script failed. Check error_group_1.txt for details." >> error_group_1.txt; else echo "Script $script completed successfully." >> output_group_1.txt; fi; done' 

# Group 2
srun -K --job-name="speech_music_classification_group_2" --gpus=1 --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`" --container-image=/netscratch/zhazzouri/scripts/speech-music-xai.sqsh --container-workdir="`pwd`" -p RTXA6000 --mem 64GB --gpus 1 --cpus-per-task=4 bash -c 'SCRIPTS=("python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 30 --batch_size 32 --type_of_transformation MFCC" "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 30 --batch_size 32 --type_of_transformation delta" "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 30 --batch_size 32 --type_of_transformation delta-delta" "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 30 --batch_size 32 --type_of_transformation LFCC" "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 30 --batch_size 32 --type_of_transformation lfcc-delta" "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 30 --batch_size 32 --type_of_transformation lfcc-delta-delta"); for script in "${SCRIPTS[@]}"; do echo "Running script: $script"; eval $script; if [ $? -ne 0 ]; then echo "Script $script failed. Check error_group_2.txt for details." >> error_group_2.txt; else echo "Script $script completed successfully." >> output_group_2.txt; fi; done' 

# Group 3
srun -K --job-name="speech_music_classification_group_3" --gpus=1 --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`" --container-image=/netscratch/zhazzouri/scripts/speech-music-xai.sqsh --container-workdir="`pwd`" -p RTXA6000 --mem 64GB --gpus 1 --cpus-per-task=4 bash -c 'SCRIPTS=("python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 80 --length_in_seconds 5 --batch_size 32 --type_of_transformation MFCC" "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 80 --length_in_seconds 5 --batch_size 32 --type_of_transformation delta" "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 80 --length_in_seconds 5 --batch_size 32 --type_of_transformation delta-delta" "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 80 --length_in_seconds 5 --batch_size 32 --type_of_transformation LFCC" "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 80 --length_in_seconds 5 --batch_size 32 --type_of_transformation lfcc-delta" "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 80 --length_in_seconds 5 --batch_size 32 --type_of_transformation lfcc-delta-delta"); for script in "${SCRIPTS[@]}"; do echo "Running script: $script"; eval $script; if [ $? -ne 0 ]; then echo "Script $script failed. Check error_group_3.txt for details." >> error_group_3.txt; else echo "Script $script completed successfully." >> output_group_3.txt; fi; done' 

# Group 4
srun -K --job-name="speech_music_classification_group_4" --gpus=1 --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`" --container-image=/netscratch/zhazzouri/scripts/speech-music-xai.sqsh --container-workdir="`pwd`" -p RTXA6000 --mem 64GB --gpus 1 --cpus-per-task=4 bash -c 'SCRIPTS=("python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model AttentionUNet --n_mfcc 32 --length_in_seconds 5 --batch_size 64 --type_of_transformation MFCC" "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model AttentionUNet --n_mfcc 32 --length_in_seconds 5 --batch_size 64 --type_of_transformation delta" "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model AttentionUNet --n_mfcc 32 --length_in_seconds 5 --batch_size 64 --type_of_transformation delta-delta" "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model AttentionUNet --n_mfcc 32 --length_in_seconds 5 --batch_size 64 --type_of_transformation LFCC" "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model AttentionUNet --n_mfcc 32 --length_in_seconds 5 --batch_size 64 --type_of_transformation lfcc-delta" "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model AttentionUNet --n_mfcc 32 --length_in_seconds 5 --batch_size 64 --type_of_transformation lfcc-delta-delta"); for script in "${SCRIPTS[@]}"; do echo "Running script: $script"; eval $script; if [ $? -ne 0 ]; then echo "Script $script failed. Check error_group_4.txt for details." >> error_group_4.txt; else echo "Script $script completed successfully." >> output_group_4.txt; fi; done' 

# Group 5
srun -K --job-name="speech_music_classification_group_5" --gpus=1 --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`" --container-image=/netscratch/zhazzouri/scripts/speech-music-xai.sqsh --container-workdir="`pwd`" -p RTXA6000 --mem 64GB --gpus 1 --cpus-per-task=4 bash -c 'SCRIPTS=("python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model R2AttU_Net --n_mfcc 32 --length_in_seconds 5 --batch_size 64 --type_of_transformation MFCC" "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model R2AttU_Net --n_mfcc 32 --length_in_seconds 5 --batch_size 64 --type_of_transformation delta" "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model R2AttU_Net --n_mfcc 32 --length_in_seconds 5 --batch_size 64 --type_of_transformation delta-delta" "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model R2AttU_Net --n_mfcc 32 --length_in_seconds 5 --batch_size 64 --type_of_transformation LFCC" "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model R2AttU_Net --n_mfcc 32 --length_in_seconds 5 --batch_size 64 --type_of_transformation lfcc-delta" "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model R2AttU_Net --n_mfcc 32 --length_in_seconds 5 --batch_size 64 --type_of_transformation lfcc-delta-delta"); for script in "${SCRIPTS[@]}"; do echo "Running script: $script"; eval $script; if [ $? -ne 0 ]; then echo "Script $script failed. Check error_group_5.txt for details." >> error_group_5.txt; else echo "Script $script completed successfully." >> output_group_5.txt; fi; done' 

# Group 6
srun -K --job-name="speech_music_classification_group_6" --gpus=1 --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`" --container-image=/netscratch/zhazzouri/scripts/speech-music-xai.sqsh --container-workdir="`pwd`" -p RTXA6000 --mem 64GB --gpus 1 --cpus-per-task=4 bash -c 'SCRIPTS=("python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model R2U_Net --n_mfcc 32 --length_in_seconds 5 --batch_size 64 --type_of_transformation MFCC" "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model R2U_Net --n_mfcc 32 --length_in_seconds 5 --batch_size 64 --type_of_transformation delta" "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model R2U_Net --n_mfcc 32 --length_in_seconds 5 --batch_size 64 --type_of_transformation delta-delta" "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model R2U_Net --n_mfcc 32 --length_in_seconds 5 --batch_size 64 --type_of_transformation LFCC" "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model R2U_Net --n_mfcc 32 --length_in_seconds 5 --batch_size 64 --type_of_transformation lfcc-delta" "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model R2U_Net --n_mfcc 32 --length_in_seconds 5 --batch_size 64 --type_of_transformation lfcc-delta-delta"); for script in "${SCRIPTS[@]}"; do echo "Running script: $script"; eval $script; if [ $? -ne 0 ]; then echo "Script $script failed. Check error_group_6.txt for details." >> error_group_6.txt; else echo "Script $script completed successfully." >> output_group_6.txt; fi; done' 

#group 7    
srun -K --job-name="speech_music_classification_group_7" --gpus=1 --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`" --container-image=/netscratch/zhazzouri/scripts/speech-music-xai.sqsh --container-workdir="`pwd`" -p RTXA6000 --mem 64GB --gpus 1 --cpus-per-task=4 bash -c 'SCRIPTS=("python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 10 --batch_size 64 --type_of_transformation MFCC" "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 10 --batch_size 64 --type_of_transformation delta" "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 10 --batch_size 64 --type_of_transformation delta-delta" "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 10 --batch_size 64 --type_of_transformation LFCC" "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 10 --batch_size 64 --type_of_transformation lfcc-delta" "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 10 --batch_size 64 --type_of_transformation lfcc-delta-delta"); for script in "${SCRIPTS[@]}"; do echo "Running script: $script"; eval $script; if [ $? -ne 0 ]; then echo "Script $script failed. Check error_group_7.txt for details." >> error_group_7.txt; else echo "Script $script completed successfully." >> output_group_7.txt; fi; done' 
#group 8
srun -K --job-name="speech_music_classification_group_8" --gpus=1 --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`" --container-image=/netscratch/zhazzouri/scripts/speech-music-xai.sqsh --container-workdir="`pwd`" -p RTXA6000 --mem 64GB --gpus 1 --cpus-per-task=4 bash -c 'SCRIPTS=("python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 15 --batch_size 32 --type_of_transformation MFCC" "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 15 --batch_size 32 --type_of_transformation delta" "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 15 --batch_size 32 --type_of_transformation delta-delta" "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 15 --batch_size 32 --type_of_transformation LFCC" "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 15 --batch_size 32 --type_of_transformation lfcc-delta" "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 15 --batch_size 32 --type_of_transformation lfcc-delta-delta"); for script in "${SCRIPTS[@]}"; do echo "Running script: $script"; eval $script; if [ $? -ne 0 ]; then echo "Script $script failed. Check error_group_8.txt for details." >> error_group_8.txt; else echo "Script $script completed successfully." >> output_group_8.txt; fi; done' 
#group 9
srun -K --job-name="speech_music_classification_group_9" --gpus=1 --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`" --container-image=/netscratch/zhazzouri/scripts/speech-music-xai.sqsh --container-workdir="`pwd`" -p RTXA6000 --mem 64GB --gpus 1 --cpus-per-task=4 bash -c 'SCRIPTS=("python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 3 --batch_size 64 --type_of_transformation MFCC" "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 3 --batch_size 64 --type_of_transformation delta" "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 3 --batch_size 64 --type_of_transformation delta-delta" "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 3 --batch_size 64 --type_of_transformation LFCC" "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 3 --batch_size 64 --type_of_transformation lfcc-delta" "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 3 --batch_size 64 --type_of_transformation lfcc-delta-delta"); for script in "${SCRIPTS[@]}"; do echo "Running script: $script"; eval $script; if [ $? -ne 0 ]; then echo "Script $script failed. Check error_group_9.txt for details." >> error_group_9.txt; else echo "Script $script completed successfully." >> output_group_9.txt; fi; done' 
#group 10
srun -K --job-name="speech_music_classification_group_10" --gpus=1 --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`" --container-image=/netscratch/zhazzouri/scripts/speech-music-xai.sqsh --container-workdir="`pwd`" -p RTXA6000 --mem 64GB --gpus 1 --cpus-per-task=4 bash -c 'SCRIPTS=("python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 2 --batch_size 64 --type_of_transformation MFCC" "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 2 --batch_size 64 --type_of_transformation delta" "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 2 --batch_size 64 --type_of_transformation delta-delta" "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 2 --batch_size 64 --type_of_transformation LFCC" "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 2 --batch_size 64 --type_of_transformation lfcc-delta" "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 2 --batch_size 64 --type_of_transformation lfcc-delta-delta"); for script in "${SCRIPTS[@]}"; do echo "Running script: $script"; eval $script; if [ $? -ne 0 ]; then echo "Script $script failed. Check error_group_10.txt for details." >> error_group_10.txt; else echo "Script $script completed successfully." >> output_group_10.txt; fi; done' 
#group 11
srun -K --job-name="speech_music_classification_group_11" --gpus=1 --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`" --container-image=/netscratch/zhazzouri/scripts/speech-music-xai.sqsh --container-workdir="`pwd`" -p RTXA6000 --mem 64GB --gpus 1 --cpus-per-task=4 bash -c 'SCRIPTS=("python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 1 --batch_size 64 --type_of_transformation MFCC" "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 1 --batch_size 64 --type_of_transformation delta" "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 1 --batch_size 64 --type_of_transformation delta-delta" "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 1 --batch_size 64 --type_of_transformation LFCC" "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 1 --batch_size 64 --type_of_transformation lfcc-delta" "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 1 --batch_size 64 --type_of_transformation lfcc-delta-delta"); for script in "${SCRIPTS[@]}"; do echo "Running script: $script"; eval $script; if [ $? -ne 0 ]; then echo "Script $script failed. Check error_group_11.txt for details." >> error_group_11.txt; else echo "Script $script completed successfully." >> output_group_11.txt; fi; done' 
#group 12
srun -K --job-name="speech_music_classification_group_12" --gpus=1 --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`" --container-image=/netscratch/zhazzouri/scripts/speech-music-xai.sqsh --container-workdir="`pwd`" -p RTXA6000 --mem 64GB --gpus 1 --cpus-per-task=4 bash -c 'SCRIPTS=("python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 0.5 --batch_size 64 --type_of_transformation MFCC" "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 0.5 --batch_size 64 --type_of_transformation delta" "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 0.5 --batch_size 64 --type_of_transformation delta-delta" "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 0.5 --batch_size 64 --type_of_transformation LFCC" "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 0.5 --batch_size 64 --type_of_transformation lfcc-delta" "python /home/zhazzouri/speech-music-classification-unet/Models/UNet/Code/train.py --model U_Net --n_mfcc 32 --length_in_seconds 0.5 --batch_size 64 --type_of_transformation lfcc-delta-delta"); for script in "${SCRIPTS[@]}"; do echo "Running script: $script"; eval $script; if [ $? -ne 0 ]; then echo "Script $script failed. Check error_group_12.txt for details." >> error_group_12.txt; else echo "Script $script completed successfully." >> output_group_12.txt; fi; done' 



