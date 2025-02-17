# ICBHI 2017:       icbhi_disease, icbhi_lungsound
# SPRSound:         sprsound_tenaryrecord, sprsound_multirecord
# HF LUng:          hf_lung_binarylungsound, hf_lung_quaternarylungsound, hf_lung_multilungsound
# Lung Sound:       lung_sound_multilungsound, lung_sound_multilabeldisease
# RD@TR:            RD@TR_multidisease
# Cinc 2016:        cinc2016_binarydisease
# Korean:           korean_multidisease
# Circor 2022:      circor2022_ternarymurmur, circor2022_binarydisease
# HSDReport:        xinhua_labeled_binaryreport
# Bowel Sound:      bowel_sound_binarysound


DATASET_NAME=korean
TASK_NAME=korean_multidisease
INIT_ENCODER=true
TRAIN_ENCODER=true

HEAD_TYPE=linear
MODEL_TYPE=auscultabase
SPEC_DIR=./spectrogram
SPEC_SUBDIR=${SPEC_DIR}/${DATASET_NAME}/${MODEL_TYPE}
CHECKPOINT_DIR=./checkpoints


echo "$TASK_NAME"
source ./scripts/finetune_options.sh

python -u src/finetune/finetune.py \
    --task_name $TASK_NAME \
    --spec_dir $SPEC_SUBDIR \
    --model_type $MODEL_TYPE \
    --model_path $MODEL_PATH \
    --head_type $HEAD_TYPE \
    --init_encoder true \
    --train_encoder false \
    --checkpoint_dir $CHECKPOINT_DIR





