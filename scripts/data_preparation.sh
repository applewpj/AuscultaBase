# Generate training & test split
# icbhi, korean, circor2022, sprsound, hf_lung, cinc2016, bowel_sound, lung_sound, RD@TR
DATASET_NAME=korean
SPEC_DIR=./spectrogram
TRAIN_TEST_PATH=${SPEC_DIR}/${DATASET_NAME}/train_test_split.json
srun --partition medai_llm --time=4-00:00:00 \
    python -u src/data_processing/generate_train_test.py \
        --dataset_name $DATASET_NAME \
        --train_test_path ${TRAIN_TEST_PATH}


# Prepare spectrodram & labels for finetuning
# icbhi:            icbhi_lungsound, icbhi_disease
# sprsound:         sprsound_ternaryrecord, sprsound_multirecord
# hf_lung:          quaternary, hf_lung_quaternarylungsound, hf_lung_multilungsound
# lung_sound:       lung_sound_multilungsound, lung_sound_multilabeldisease
# RD@TR:            RD@TR_multidisease
# cinc2016:         cinc2016_binarydisease
# korean:           korean_multidisease
# circor2022:       circor2022_ternarymurmur, circor2022_binarydisease
# xinhua_labeled:   xinhua_labeled_binarydisease, xinhua_labeled_multidisease
# bowel_sound:      bowel_sound_binarysound
MODEL_TYPE=auscultabase
SPEC_DIR=./spectrogram
DATASET_NAME=korean
TASK_NAME=korean_multidisease
TRAIN_TEST_PATH=${SPEC_DIR}/${DATASET_NAME}/train_test_split.json
srun --partition medai_llm --cpus-per-task=1 --time=4-00:00:00\
    python -u src/data_processing/prepare_label.py \
        --dataset_name ${DATASET_NAME} \
        --task_name ${TASK_NAME} \
        --train_test_path ${TRAIN_TEST_PATH} \
        --save_dir ${SPEC_DIR}/${DATASET_NAME}/${MODEL_TYPE} \
        --model_type ${MODEL_TYPE}
