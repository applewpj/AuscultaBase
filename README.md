# AuscultaBase
Official Repository of **<AuscultaBase: A Foundational Step Towards AI-Powered Body Sound Diagnostics>**

### Model Checkpoint

The model checkpoint for this manuscript can be downloaded at [this link](https://drive.google.com/file/d/1DtQ2SK70lQd_R4Aq6vSBHzbKrZYAs2zn/view?usp=drive_link). The checkpoints for other models have to be downloaded as saved into the corresponding directory in `./models`.

### Data Preparation

1. Download datasets
    
    Please refer to the data availability part in the manuscript or the following table, and then download the recordings and labels into the corresponding directory below `./data` .
    
    | Dataset | `$DATASET_NAME` | Sound Type | Source | License |
    | --- | --- | --- | --- | --- |
    | [SPRSound](https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound) | `sprsound` | Lung | SJTU | CC-BY-4.0 |
    | [HF Lung](https://gitlab.com/techsupportHF/HF_Lung_V1) | `hf_lung` | Lung | NTU | CC-BY-4.0 |
    | [ICBHI 2017](https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge) | `icbhi` | Lung | * | CC0 |
    | [Lung Sound](https://data.mendeley.com/datasets/jwyy9np4gv/3) | `lung_sound` | Lung | JUST&KAUH | CC-BY-4.0 |
    | [RD@TR](https://data.mendeley.com/datasets/p9z4h98s6j/1) | `RD@TR` | Lung | ITU | CC-BY-4.0 |
    | [Korean](https://github.com/yaseen21khan/Classification-of-Heart-Sound-Signal-Using-Multiple-Features-) | `korean` | Heart | SJU | CC-BY-4.0 |
    | [Cinc 2016](https://archive.physionet.org/physiobank/database/challenge/2016/) | `cinc2016` | Heart | * | Custom License |
    | [Circor 2022](https://physionet.org/content/circor-heart-sound/1.0.3/) | `circor2022` | Heart | * | ODC-By |
    | [Bowel Sound](https://www.kaggle.com/robertnowak/bowel-sounds) | `bowel_sound` | Bowel | WTU&TUMS | CC BY-NC 4.0 |
2. Make sure the indexing path for each dataset is correct in `./src/data_processing/data_info.py` .
3. Generate the training and test split for each dataset. The default ratio of the test split is set as 0.1 if there is no specific partition strategy, you can also change this ratio in `./src/data_processing/data_info.py` .
    
    ```bash
    SPEC_DIR=./spectrogram
    DATASET_NAME=korean
    TRAIN_TEST_PATH=${SPEC_DIR}/${DATASET_NAME}/train_test_split.json
    python -u src/data_processing/generate_train_test.py \
        --dataset_name $DATASEt_NAME \
        --train_test_path ${TRAIN_TEST_PATH}
    ```
    
4. Pro-process the recordings for each dataset and generate the label for each task.
    
     
    
    ```bash
    MODEL_TYPE=auscultabase. # auscultabase, clap, pann, audiomae, opera-ct
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
    ```
    
    You can specify `$TASK_NAME` from the following table.
    
    | Task ID | `$DATASET_NAME` | `TASK_NAME`  | Sound Type | Function | Task Type |
    | --- | --- | --- | --- | --- | --- |
    | T1 | `sprsound` | `sprsound_ternaryrecord` | Lung | Abnormality Detection | MC |
    | T2 | `sprsound` | `sprsound_multirecord` | Lung | Abnormality Detection | MC |
    | T3 | `hf_lung` | `hf_lung_binarylungsound` | Lung | Abnormality Detection | BC |
    | T4 | `hf_lung` | `hf_lung_quaternarylungsound` | Lung | Abnormality Detection | MC |
    | T5 | `hf_lung` | `hf_lung_multilungsound` | Lung | Abnormality Detection | MC |
    | T6 | `icbhi` | `icbhi_lungsound` | Lung | Abnormality Detection | MC |
    | T7 | `lung_sound` | `lung_sound_multilungsound` | Lung | Abnormality Detection | MC |
    | T8 | `circor2022` | `circor2022_ternarymurmur` | Heart | Abnormality Detection | MC |
    | T9 | `bowel_sound` | `bowel_sound_binarysound` | Bowel | Abnormality Detection | BC |
    | T10 | `icbhi` | `icbhi_disease` | Lung | Disease Diagnosis | MC |
    | T11 | `lung_sound` | `lung_sound_multilabeldisease` | Lung | Disease Diagnosis | ML |
    | T12 | `RD@TR` | `RD@TR_multidisease` | Lung | Disease Diagnosis | MC |
    | T13 | `korean` | `korean_multidisease` | Heart | Disease Diagnosis | MC |
    | T14 | `cinc2016` | `cinc2016_binarydisease` | Heart | Disease Diagnosis | BC |
    | T15 | `circor2022` | `circor2022_binarydisease` | Heart | Disease Diagnosis | BC |

### Fine-tuning

1. Make sure the path for checkpoints which you have specified as `$MODEL_TYPE` is correct in `./scripts/finetune_options.sh` .
2. Start finetuning
    
    ```bash
    DATASET_NAME=korean
    TASK_NAME=korean_multidisease
    INIT_ENCODER=true
    TRAIN_ENCODER=true
    
    HEAD_TYPE=linear
    MODEL_TYPE=auscultabase
    SPEC_DIR=./spectrogram
    SPEC_SUBDIR=${SPEC_DIR}/${DATASET_NAME}/${MODEL_TYPE}
    CHECKPOINT_DIR=./checkpoints
    
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
    ```
