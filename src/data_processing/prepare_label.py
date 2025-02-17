import glob as gb
import argparse, random
import numpy as np
import pandas as pd
from tqdm import tqdm
import os, sys, pdb, json, torchaudio
import argparse
from str2bool import str2bool

from prepare_spec import preprocess_entire_or_chunk, preprocess_icbhi_lungsound, preprocess_hf_lung_lungsound
from data_info import DATA_DIR_DICT



def generate_label_icbhi_disease(dataset_name, task_name, data_dir, split_path, save_dir, label_json_name, model_type):
    assert task_name == "icbhi_disease"
    print(f"Generating labels for {task_name}...")
    

    with open(split_path, 'r', encoding='utf-8') as f:
        splits = json.load(f)
    with open(os.path.join(data_dir, "ICBHI_Challenge_diagnosis.txt")) as f:
        labels = f.readlines()
    classes = ["URTI", "Healthy", "COPD", "Bronchiectasis", "Pneumonia", "Bronchiolitis"]  # 排除LRTI, Asthma
    
    n_classes = len(classes)
    diagnosis_index = {name: idx for idx, name in enumerate(classes)}
    patient_class_dict = {}
    for raw_text in labels:
        patient_ID, diagnosis = raw_text.split()
        if diagnosis in diagnosis_index.keys():  # 有对应label的patient
            patient_class_dict[patient_ID] = diagnosis_index[diagnosis]
    
    label_dict = {"n_classes": n_classes, "classes": classes, "task": "classification"}
    for spec_path, split in tqdm(splits.items(), total=len(splits)):
        spec_name = os.path.basename(spec_path)
        spec_dirname = os.path.basename(os.path.dirname(spec_path))
        if "chunk" in spec_dirname:
            audio_name = "_".join(spec_name[:-4].split("_"))
        else:
            audio_name = spec_name[:-4]
        patient_ID = spec_name[:3]
        if patient_ID in patient_class_dict.keys():
            # label_dict[audio_name] = patient_class_dict[patient_ID]
            label_dict[spec_path] = [split, patient_class_dict[patient_ID]]
        
    label_dir = os.path.join(save_dir, "label")
    save_path = os.path.join(label_dir, label_json_name)
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    with open(save_path, 'w') as f:
        json.dump(label_dict, f)


def generate_label_sprsound_record(dataset_name, task_name, data_dir, split_path, save_dir, label_json_name, model_type):
    assert task_name in ["sprsound_ternaryrecord", "multirecord"]
    print(f"Generating labels for {task_name}...")
    
    
    with open(split_path, 'r', encoding='utf-8') as f:
        splits = json.load(f)
    json_subdirs = [os.path.join(data_dir, i) for i in ["train_json", "test_json_2023", "test_json_2022/inter_test_json", "test_json_2022/intra_test_json"]]
        
    # classes = ["Normal" , "Poor Quality", "CAS", "DAS", "CAS & DAS"]
    if task_name == "sprsound_ternaryrecord":
        classes = ["Normal" , "Poor Quality", "Adventitious"]
        class_index = {
            "Normal": 0,
            "Poor Quality": 1,
            "CAS": 2,
            "DAS": 2,
            "CAS & DAS": 2
        }
    elif task_name == "sprsound_multirecord":
        classes = ["Normal" , "Poor Quality", "CAS", "DAS", "CAS & DAS"]
        class_index = {
            "Normal": 0,
            "Poor Quality": 1,
            "CAS": 2,
            "DAS": 3,
            "CAS & DAS": 4
        }
    n_classes = len(classes)
    
    label_dict = {"n_classes": n_classes, "classes": classes, "task": "classification"}
    for spec_path, split in tqdm(splits.items(), total=len(splits)):
        spec_name = os.path.basename(spec_path)
        spec_dirname = os.path.basename(os.path.dirname(spec_path))
        if "chunk" in spec_dirname:
            audio_name = "_".join(spec_name[:-4].split("_")[:-1])
        else:
            audio_name = spec_name[:-4]
        json_name = audio_name + ".json"
        
        flag = False
        for json_subdir in json_subdirs:
            # json_list = list(gb.glob(os.path.join(json_subdir, "*.json")))
            json_list = os.listdir(json_subdir)
            if json_name in json_list:
                with open(os.path.join(json_subdir, json_name)) as f:
                    labels = json.load(f)
                record_label = labels["record_annotation"]
                # event_label = labels["event_annotation"]
                label_dict[spec_path] = [split, class_index[record_label]]
                flag = True
                break
        assert flag
    label_dir = os.path.join(save_dir, "label")
    save_path = os.path.join(label_dir, label_json_name)
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    with open(save_path, 'w') as f:
        json.dump(label_dict, f)


def generate_label_icbhi_lungsound(task_name, data_dir, split_path, save_dir, label_json_name, model_type):
    assert task_name == "icbhi_lungsound"
    split_dir = os.path.dirname(split_path)
    print(f"Generating labels for {task_name}...")
    
    
    with open(split_path, 'r', encoding='utf-8') as f:
        splits = json.load(f)
    
    classes = ["Normal", "Crackle", "Wheeze", "Wheeze+Crackle"]
    class_index = {
        "0,0": 0,
        "1,0": 1,
        "0,1": 2,
        "1,1": 3,
    }
    n_classes = len(classes)
    
    label_dict = {"n_classes": n_classes, "classes": classes, "task": "classification"}
    
    label_path_list = list(gb.glob(os.path.join(data_dir, "*.txt")))
    
    if model_type == "cola_md":
        suffix = "npy"
    elif model_type in ["clap", "pann"]:
        suffix = "wav"
    elif model_type == "audiomae":
        suffix = "pt"
    
    for idx, label_path in tqdm(enumerate(label_path_list), total=len(label_path_list)):
        # wav_name = os.path.basename(wav_path)
        # label_name = wav_name[:-4] + ".json"
        # label_path = wav_path[:-4] + ".txt"
        label_name = os.path.basename(label_path)
        if label_name not in ["filename_differences.txt", "filename_format.txt", "ICBHI_Challenge_demographic_information.txt", "ICBHI_Challenge_diagnosis.txt", "ICBHI_challenge_train_test.txt"]:
            with open(label_path, 'r', encoding='utf-8') as f:
                labels = f.readlines()
            
            for segment_idx, segment_label in enumerate(labels):
                segment_label = segment_label.split()
                label_0 = int(segment_label[2])
                label_1 = int(segment_label[3])
                label_indicator = f"{str(label_0)},{str(label_1)}"
                chunk_idx = 0
                while True:
                    spec_path = os.path.join(split_dir, label_name[:-4] + f"_segment{str(segment_idx)}_{str(chunk_idx)}.{suffix}")
                    try:
                        split = splits[spec_path]
                        label_dict[spec_path] = [split, class_index[label_indicator]]
                        chunk_idx += 1
                    except:
                        break
    label_dir = os.path.join(save_dir, "label")
    save_path = os.path.join(label_dir, label_json_name)
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    with open(save_path, 'w') as f:
        json.dump(label_dict, f)




def generate_label_hf_lung_lungsound(task_name, data_dir, split_path, save_dir, label_json_name, model_type):
    assert task_name in ["hf_lung_binarylungsound", "hf_lung_quaternarylungsound", "hf_lung_multilungsound"]
    split_dir = os.path.dirname(split_path)
    print(f"Generating labels for {task_name}...")
    
    
    with open(split_path, 'r', encoding='utf-8') as f:
        splits = json.load(f)
    
    if task_name == "hf_lung_binarylungsound":
        classes = ["Normal", "Abnormal"]
        class_index = {
            "I": 0,
            "E": 0,
            "Wheeze": 1,
            "Stridor": 1,
            "Rhonchi": 1,
            "D": 1,
        }
    elif task_name == "hf_lung_quaternarylungsound":
        classes = ["Inhalation", "Exhalation", "CAS", "DAS"]
        class_index = {
            "I": 0,
            "E": 1,
            "Wheeze": 2,
            "Stridor": 2,
            "Rhonchi": 2,
            "D": 3,
        }
    elif task_name == "hf_lung_multilungsound":
        classes = ["Inhalation", "Exhalation", "Wheeze", "Stridor", "Rhonchi", "Crackle"]
        class_index = {
            "I": 0,
            "E": 1,
            "Wheeze": 2,
            "Stridor": 3,
            "Rhonchi": 4,
            "D": 5,
        }
    n_classes = len(classes)
    
    label_dict = {"n_classes": n_classes, "classes": classes, "task": "classification"}
    
    label_path_list_1 = list(gb.glob(os.path.join(data_dir, "HF_Lung_V1-master/train", "*.txt")))
    label_path_list_2 = list(gb.glob(os.path.join(data_dir, "HF_Lung_V1-master/test", "*.txt")))
    label_path_list = label_path_list_1 + label_path_list_2
    
    if model_type == "cola_md":
        suffix = "npy"
    elif model_type in ["clap", "pann"]:
        suffix = "wav"
    elif model_type == "audiomae":
        suffix = "pt"
    
    for idx, label_path in tqdm(enumerate(label_path_list), total=len(label_path_list)):
        with open(label_path, 'r', encoding='utf-8') as f:
            labels = f.readlines()
        
        label_name = os.path.basename(label_path)
        for segment_idx, segment_label in enumerate(labels):
            segment_label = segment_label.split()
            label = segment_label[0]
            chunk_idx = 0
            while True:
                spec_path = os.path.join(split_dir, label_name[:-10] + f"_segment{str(segment_idx)}_{str(chunk_idx)}.{suffix}")
                try:
                    split = splits[spec_path]
                    label_dict[spec_path] = [split, class_index[label]]
                    chunk_idx += 1
                except:
                    break
    label_dir = os.path.join(save_dir, "label")
    save_path = os.path.join(label_dir, label_json_name)
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    with open(save_path, 'w') as f:
        json.dump(label_dict, f)



def generate_label_lung_sound_multilungsound(task_name, data_dir, split_path, save_dir, label_json_name, model_type):
    assert task_name in ["lung_sound_multilungsound"]
    split_dir = os.path.dirname(split_path)
    print(f"Generating labels for {task_name}...")
    
    
    with open(split_path, 'r', encoding='utf-8') as f:
        splits = json.load(f)
    
    classes = ["Normal", "Crepitation", "Wheeze", "Crackle", "Bronchi", "Wheeze & Crackle", "Bornchi & Crackle"]
    class_index = {
        "N": 0,
        "Crep": 1,
        "I E W": 2, "E W": 2,
        "I C": 3, "C": 3,
        "Bronchial": 4,
        "I C E W": 5,
        "I C B": 6,
    }
    n_classes = len(classes)
    
    label_dict = {"n_classes": n_classes, "classes": classes, "task": "classification"}
    
    
    for idx, (spec_path, split) in tqdm(enumerate(splits.items()), total=len(splits)):
        spec_name = os.path.basename(spec_path)
        spec_name_list = spec_name.split(",")
        # Number, disease_label = spec_name_list[0].split("_")
        lungsound_label = spec_name_list[1]
        label_dict[spec_path] = [split, class_index[lungsound_label]]
    
    label_dir = os.path.join(save_dir, "label")
    save_path = os.path.join(label_dir, label_json_name)
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    with open(save_path, 'w') as f:
        json.dump(label_dict, f)


def generate_label_lung_sound_multilabeldisease(task_name, data_dir, split_path, save_dir, label_json_name, model_type):
    assert task_name in ["lung_sound_multilabeldisease"]
    split_dir = os.path.dirname(split_path)
    print(f"Generating labels for {task_name}...")
    
    
    with open(split_path, 'r', encoding='utf-8') as f:
        splits = json.load(f)
    
    classes = ["Normal", "Asthma", "Heart Failure", "Lung Fibrosis", "Plueral Effusion", "COPD", "Pneumonia", "BRON"]
    class_index = {
        "N": 0,
        "asthma": 1,
        "heart failure": 2,
        "lung fibrosis": 3,
        "plueral effusion": 4,
        "copd": 5,
        "pneumonia": 6,
        "bron": 7
    }
    n_classes = len(classes)
    
    label_dict = {"n_classes": n_classes, "classes": classes, "task": "multilabel"}
    
    
    for idx, (spec_path, split) in tqdm(enumerate(splits.items()), total=len(splits)):
        spec_name = os.path.basename(spec_path)
        disease_label = spec_name.split(",")[0].split("_")[1]
        labels = [0] * n_classes
        for idx, candidate_name in enumerate(class_index.keys()):
            if idx == 0 and disease_label == candidate_name:
                labels[idx] = 1
                break
            elif candidate_name in disease_label.lower():
                labels[idx] = 1
        try:
            assert sum(labels) > 0
        except:
            pdb.set_trace()
        label_dict[spec_path] = [split, labels]
    
    label_dir = os.path.join(save_dir, "label")
    save_path = os.path.join(label_dir, label_json_name)
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    with open(save_path, 'w') as f:
        json.dump(label_dict, f)


def generate_label_RDTR_multidisease(task_name, data_dir, split_path, save_dir, label_json_name, model_type):
    assert task_name in ["RD@TR_multidisease"]
    split_dir = os.path.dirname(split_path)
    data_parent_dir = os.path.dirname(data_dir)
    print(f"Generating labels for {task_name}...")
    
    
    with open(split_path, 'r', encoding='utf-8') as f:
        splits = json.load(f)
    
    classes = ["COPD0", "COPD1", "COPD2", "COPD3", "COPD4"]
    class_index = {
        "COPD0": 0,
        "COPD1": 1,
        "COPD2": 2,
        "COPD3": 3,
        "COPD4": 4,
    }
    n_classes = len(classes)
    
    label_dict = {"n_classes": n_classes, "classes": classes, "task": "classification"}
    
    label_path = os.path.join(data_parent_dir, "Labels.xlsx")
    temp = pd.read_excel(label_path)
    labels = {}
    # pdb.set_trace()
    for idx in range(len(temp)):
        key = temp["Patient ID"][idx]
        val = temp["Diagnosis"][idx]
        labels[key] = val
    
    for idx, (spec_path, split) in tqdm(enumerate(splits.items()), total=len(splits)):
        spec_name = os.path.basename(spec_path)
        patientID = spec_name.split("_")[0]
        disease_label = labels[patientID]
        label_dict[spec_path] = [split, class_index[disease_label]]
    
    label_dir = os.path.join(save_dir, "label")
    save_path = os.path.join(label_dir, label_json_name)
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    with open(save_path, 'w') as f:
        json.dump(label_dict, f)


def generate_label_cinc2016_binarydisease(task_name, data_dir, split_path, save_dir, label_json_name, model_type):
    assert task_name in ["cinc2016_binarydisease"]
    # split_dir = os.path.dirname(split_path)
    print(f"Generating labels for {task_name}...")
    
    
    with open(split_path, 'r', encoding='utf-8') as f:
        splits = json.load(f)
    
    classes = ["Normal", "Abnormal"]
    class_index = {
        "Normal": 0,
        "Abnormal": 1,
    }
    n_classes = len(classes)
    
    label_dict = {"n_classes": n_classes, "classes": classes, "task": "classification"}
    
    for idx, (spec_path, split) in tqdm(enumerate(splits.items()), total=len(splits)):
        spec_name = os.path.basename(spec_path)
        subset = spec_name[0]
        label_path = os.path.join(data_dir, "training-"+subset, spec_name[:-4].split("_")[0]+".hea")
        with open(label_path) as f:
            labels = f.readlines()
        disease_label = labels[-1].split()[-1]
        label_dict[spec_path] = [split, class_index[disease_label]]
    
    label_dir = os.path.join(save_dir, "label")
    save_path = os.path.join(label_dir, label_json_name)
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    with open(save_path, 'w') as f:
        json.dump(label_dict, f)


def generate_label_korean_multidisease(task_name, data_dir, split_path, save_dir, label_json_name, model_type):
    assert task_name in ["korean_multidisease"]
    # split_dir = os.path.dirname(split_path)
    print(f"Generating labels for {task_name}...")
    
    
    with open(split_path, 'r', encoding='utf-8') as f:
        splits = json.load(f)
    
    classes = ["Normal (N)", "Aortic Stenosis (AS)", "Mitral Reguritation (MR)", "Mitral Stenosis (MS)", "Murmur in Systole (MVP)"]
    class_index = {
        "N": 0,
        "AS": 1,
        "MR": 2,
        "MS": 3,
        "MVP": 4,
    }
    n_classes = len(classes)
    
    label_dict = {"n_classes": n_classes, "classes": classes, "task": "classification"}
    
    for idx, (spec_path, split) in tqdm(enumerate(splits.items()), total=len(splits)):
        spec_name = os.path.basename(spec_path)
        disease_label = spec_name.split("_")[1]
        label_dict[spec_path] = [split, class_index[disease_label]]
    
    label_dir = os.path.join(save_dir, "label")
    save_path = os.path.join(label_dir, label_json_name)
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    with open(save_path, 'w') as f:
        json.dump(label_dict, f)
        

def generate_label_circor2022(task_name, data_dir, split_path, save_dir, label_json_name, model_type):
    assert task_name in ["circor2022_ternarymurmur", "circor2022_binarydisease"]
    # split_dir = os.path.dirname(split_path)
    print(f"Generating labels for {task_name}...")
    
    
    with open(split_path, 'r', encoding='utf-8') as f:
        splits = json.load(f)
    
    data_parent_dir = os.path.dirname(data_dir)
    label_csv = pd.read_csv(os.path.join(data_parent_dir, "training_data.csv"))
    ID_label = {}
    for idx in range(len(label_csv)):
        ID = str(label_csv["Patient ID"][idx])
        if task_name == "circor2022_ternarymurmur":
            label = label_csv["Murmur"][idx]
        elif task_name == "circor2022_binarydisease":
            label = label_csv["Outcome"][idx]
        ID_label[ID] = label
    
    if task_name == "circor2022_ternarymurmur":
        classes = ["Present", "Absent", "Unknown"]
        class_index = {
            "Present": 0,
            "Absent": 1,
            "Unknown": 2,
        }
    elif task_name == "circor2022_binarydisease":
        classes = ["Normal", "Abnormal"]
        class_index = {
            "Normal": 0,
            "Abnormal": 1,
        }
    n_classes = len(classes)
    
    label_dict = {"n_classes": n_classes, "classes": classes, "task": "classification"}
    for idx, (spec_path, split) in tqdm(enumerate(splits.items()), total=len(splits)):
        spec_name = os.path.basename(spec_path)
        ID = spec_name.split("_")[0]
        label = ID_label[ID]
        label_dict[spec_path] = [split, class_index[label]]
    
    label_dir = os.path.join(save_dir, "label")
    save_path = os.path.join(label_dir, label_json_name)
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    with open(save_path, 'w') as f:
        json.dump(label_dict, f)


def generate_label_bowel_sound(task_name, data_dir, split_path, save_dir, label_json_name, model_type):
    assert task_name in ["bowel_sound_count"]
    # split_dir = os.path.dirname(split_path)
    print(f"Generating labels for {task_name}...")
    
    
    with open(split_path, 'r', encoding='utf-8') as f:
        splits = json.load(f)
    
    label_csv = pd.read_csv(os.path.join(data_dir, "files.csv"))
    wavname_count = {}
    for idx in range(len(label_csv)):
        wav_name = label_csv["filename"][idx]
        count = label_csv["sounds_amount"][idx]
        wavname_count[wav_name] = int(count)
    
    
    label_dict = {"n_classes": 1, "classes": "Count", "task": "regression"}
    for idx, (spec_path, split) in tqdm(enumerate(splits.items()), total=len(splits)):
        spec_name = os.path.basename(spec_path)
        wav_name = "_".join(spec_name.split("_")[:2]) + ".wav"
        label_dict[spec_path] = [split, wavname_count[wav_name]]
    
    label_dir = os.path.join(save_dir, "label")
    save_path = os.path.join(label_dir, label_json_name)
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    with open(save_path, 'w') as f:
        json.dump(label_dict, f)



def generate_label_bowel_sound_count(task_name, data_dir, split_path, save_dir, label_json_name, model_type):
    assert task_name in ["bowel_sound_count"]
    # split_dir = os.path.dirname(split_path)
    print(f"Generating labels for {task_name}...")
    
    
    with open(split_path, 'r', encoding='utf-8') as f:
        splits = json.load(f)
    
    label_csv = pd.read_csv(os.path.join(data_dir, "files.csv"))
    wavname_count = {}
    for idx in range(len(label_csv)):
        wav_name = label_csv["filename"][idx]
        count = label_csv["sounds_amount"][idx]
        wavname_count[wav_name] = int(count)
    
    
    label_dict = {"n_classes": 1, "classes": "Count", "task": "regression"}
    for idx, (spec_path, split) in tqdm(enumerate(splits.items()), total=len(splits)):
        spec_name = os.path.basename(spec_path)
        wav_name = "_".join(spec_name.split("_")[:2]) + ".wav"
        label_dict[spec_path] = [split, wavname_count[wav_name]]
    
    label_dir = os.path.join(save_dir, "label")
    save_path = os.path.join(label_dir, label_json_name)
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    with open(save_path, 'w') as f:
        json.dump(label_dict, f)


def generate_label_bowel_sound_binarysound(task_name, data_dir, split_path, save_dir, label_json_name, model_type):
    assert task_name in ["bowel_sound_binarysound"]
    print(f"Generating labels for {task_name}...")
    
    
    with open(split_path, 'r', encoding='utf-8') as f:
        splits = json.load(f)
    
    label_csv = pd.read_csv(os.path.join(data_dir, "files.csv"))
    wavname_count = {}
    for idx in range(len(label_csv)):
        wav_name = label_csv["filename"][idx]
        count = label_csv["sounds_amount"][idx]
        wavname_count[wav_name] = int(count)
    
    classes = ['sound absent', 'sound present']
    label_dict = {"n_classes": 2, "classes": classes, "task": "classification"}
    for idx, (spec_path, split) in tqdm(enumerate(splits.items()), total=len(splits)):
        spec_name = os.path.basename(spec_path)
        wav_name = "_".join(spec_name.split("_")[:2]) + ".wav"
        if wavname_count[wav_name] == 0:
            label = 0
        else:
            label = 1
        label_dict[spec_path] = [split, label]
    
    label_dir = os.path.join(save_dir, "label")
    save_path = os.path.join(label_dir, label_json_name)
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    with open(save_path, 'w') as f:
        json.dump(label_dict, f)


def transform_test_split_json(dataset_name, save_dir, label_json_name):
    print(f"Transform the chunked test set into entire for {label_json_name}...")
    label_path = os.path.join(save_dir, "label", label_json_name)
    with open(label_path, "r") as f:
        # pdb.set_trace()
        old_label_dict = json.load(f)
    
    new_label_dict = {"classes": old_label_dict["classes"], "n_classes": old_label_dict["n_classes"], "task": old_label_dict["task"]}
    for key, value in tqdm(old_label_dict.items(), total=len(old_label_dict)):
        if key not in ["classes", "n_classes", "task"]:
            split = value[0]
            if split == "train":
                new_label_dict[key] = value
            else:  # split == "test"
                label = value[1]
                spec_chunk_name = os.path.basename(key)
                spec_parent_dir = os.path.dirname(os.path.dirname(key))
                spec_chunk_dirname = os.path.basename(os.path.dirname(key))
                if len(spec_chunk_dirname) > 8:
                    prefix = spec_chunk_dirname.split("_")[0]
                    spec_entire_dirname = "_".join([prefix, "entire"])
                else:
                    spec_entire_dirname = "entire"
                # spec_entire_dir = os.path.join(spec_parent_dir, spec_entire_dirname)
                suffix = spec_chunk_name.split(".")[-1]
                spec_entire_name = "_".join(spec_chunk_name[:-4].split("_")[:-1]) + "." + suffix
                spec_entire_path = os.path.join(spec_parent_dir, spec_entire_dirname, spec_entire_name)
                new_label_dict[spec_entire_path] = value
    
    with open(label_path, 'w') as f:
        json.dump(new_label_dict, f)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--task_name", type=str)
    parser.add_argument("--train_test_path", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--model_type", type=str, choices=["auscultabase", "pann", "clap", "audiomae", "opera-ct"])
    args = parser.parse_args()
    
    task_name = args.task_name
    dataset_name = args.dataset_name
    train_test_path = args.train_test_path
    save_dir = args.save_dir
    model_type = args.model_type
    label_json_name = "_".join([task_name, "label.json"])
    data_dir = DATA_DIR_DICT[dataset_name]

    if task_name == "icbhi_disease":
        min_entire_len = 0
        max_chunk_len = 8
        # audio processing
        preprocess_entire_or_chunk(dataset_name, save_dir, split_path=train_test_path, valid_name="valid_split.json", chunk_type="chunk", max_chunk_len=max_chunk_len, model_type=model_type)
        preprocess_entire_or_chunk(dataset_name, save_dir, split_path=train_test_path, valid_name="valid_split.json", chunk_type="entire", min_entire_len=min_entire_len, model_type=model_type)
        # label generation
        split_path = os.path.join(save_dir, "chunk_" + str(max_chunk_len), "valid_split.json")
        generate_label_icbhi_disease(dataset_name, task_name, data_dir, split_path, save_dir, label_json_name, model_type=model_type)
        # transform
        transform_test_split_json(dataset_name, save_dir, label_json_name)
    elif task_name == "icbhi_lungsound":
        min_entire_len = 0
        max_chunk_len = 8
        # audio processing
        preprocess_icbhi_lungsound(save_dir, split_path=train_test_path, valid_name="valid_split.json", chunk_type="chunk", max_chunk_len=max_chunk_len, model_type=model_type)
        min_entire_len = 1 if model_type in ["pann", "clap"] else 0
        preprocess_icbhi_lungsound(save_dir, split_path=train_test_path, valid_name="valid_split.json", chunk_type="entire", min_entire_len=min_entire_len, model_type=model_type)
        # label generation
        split_path = os.path.join(save_dir, "_".join(["cycle", "chunk", str(max_chunk_len)]), "valid_split.json")
        generate_label_icbhi_lungsound(task_name, data_dir, split_path, save_dir, label_json_name, model_type=model_type)
        # transform
        transform_test_split_json(dataset_name, save_dir, label_json_name)
    elif task_name in ["sprsound_ternaryrecord", "sprsound_multirecord"]:
        min_entire_len = 0
        max_chunk_len = 8
        # audio processing
        preprocess_entire_or_chunk(dataset_name, save_dir, split_path=train_test_path, valid_name="valid_split.json", chunk_type="chunk", max_chunk_len=max_chunk_len, model_type=model_type)
        preprocess_entire_or_chunk(dataset_name, save_dir, split_path=train_test_path, valid_name="valid_split.json", chunk_type="entire", min_entire_len=0, model_type=model_type)
        # label generation
        split_path = os.path.join(save_dir, "chunk_" + str(max_chunk_len), "valid_split.json")
        generate_label_sprsound_record(dataset_name, task_name, data_dir, split_path, save_dir, label_json_name, model_type=model_type)
        # transform
        transform_test_split_json(dataset_name, save_dir, label_json_name)
    elif task_name in ["hf_lung_binarylungsound", "hf_lung_quaternarylungsound", "hf_lung_multilungsound"]:
        min_entire_len = 0
        max_chunk_len = 2
        # audio processing
        preprocess_hf_lung_lungsound(save_dir, split_path=train_test_path, valid_name="valid_split.json", chunk_type="chunk", max_chunk_len=max_chunk_len, model_type=model_type)
        preprocess_hf_lung_lungsound(save_dir, split_path=train_test_path, valid_name="valid_split.json", chunk_type="entire", min_entire_len=0, model_type=model_type)
        # label generation
        split_path = os.path.join(save_dir, "_".join(["event", "chunk", str(max_chunk_len)]), "valid_split.json")
        generate_label_hf_lung_lungsound(task_name, data_dir, split_path, save_dir, label_json_name, model_type=model_type)
        # transform
        transform_test_split_json(dataset_name, save_dir, label_json_name)
    elif task_name in ["lung_sound_multilungsound"]:
        min_entire_len = 0
        max_chunk_len = 8
        # audio processing
        preprocess_entire_or_chunk(dataset_name, save_dir, split_path=train_test_path, valid_name="valid_split.json", chunk_type="chunk", max_chunk_len=max_chunk_len, model_type=model_type)
        preprocess_entire_or_chunk(dataset_name, save_dir, split_path=train_test_path, valid_name="valid_split.json", chunk_type="entire", min_entire_len=0, model_type=model_type)
        # label generation
        split_path = os.path.join(save_dir, "_".join(["chunk", str(max_chunk_len)]), "valid_split.json")
        generate_label_lung_sound_multilungsound(task_name, data_dir, split_path, save_dir, label_json_name, model_type=model_type)
        # transform
        transform_test_split_json(dataset_name, save_dir, label_json_name)
    elif task_name in ["lung_sound_multilabeldisease"]:
        min_entire_len = 0
        max_chunk_len = 8
        # audio processing
        preprocess_entire_or_chunk(dataset_name, save_dir, split_path=train_test_path, valid_name="valid_split.json", chunk_type="chunk", max_chunk_len=max_chunk_len, model_type=model_type)
        preprocess_entire_or_chunk(dataset_name, save_dir, split_path=train_test_path, valid_name="valid_split.json", chunk_type="entire", min_entire_len=0, model_type=model_type)
        # label generation
        split_path = os.path.join(save_dir, "_".join(["chunk", str(max_chunk_len)]), "valid_split.json")
        generate_label_lung_sound_multilabeldisease(task_name, data_dir, split_path, save_dir, label_json_name, model_type=model_type)
        # transform
        transform_test_split_json(dataset_name, save_dir, label_json_name)
    elif task_name in ["RD@TR_multidisease"]:
        min_entire_len = 0
        max_chunk_len = 8
        # audio processing
        preprocess_entire_or_chunk(dataset_name, save_dir, split_path=train_test_path, valid_name="valid_split.json", chunk_type="chunk", max_chunk_len=max_chunk_len, model_type=model_type)
        preprocess_entire_or_chunk(dataset_name, save_dir, split_path=train_test_path, valid_name="valid_split.json", chunk_type="entire", min_entire_len=0, model_type=model_type)
        # label generation
        split_path = os.path.join(save_dir, "_".join(["chunk", str(max_chunk_len)]), "valid_split.json")
        generate_label_RDTR_multidisease(task_name, data_dir, split_path, save_dir, label_json_name, model_type=model_type)
        # transform
        transform_test_split_json(dataset_name, save_dir, label_json_name)
    elif task_name in ["cinc2016_binarydisease"]:
        min_entire_len = 0
        max_chunk_len = 8
        # audio processing
        preprocess_entire_or_chunk(dataset_name, save_dir, split_path=train_test_path, valid_name="valid_split.json", chunk_type="chunk", max_chunk_len=max_chunk_len, model_type=model_type)
        preprocess_entire_or_chunk(dataset_name, save_dir, split_path=train_test_path, valid_name="valid_split.json", chunk_type="entire", min_entire_len=0, model_type=model_type)
        # label generation
        split_path = os.path.join(save_dir, "_".join(["chunk", str(max_chunk_len)]), "valid_split.json")
        generate_label_cinc2016_binarydisease(task_name, data_dir, split_path, save_dir, label_json_name, model_type=model_type)
        # transform
        transform_test_split_json(dataset_name, save_dir, label_json_name)
    elif task_name in ["korean_multidisease"]:
        min_entire_len = 0
        max_chunk_len = 4
        # audio processing
        preprocess_entire_or_chunk(dataset_name, save_dir, split_path=train_test_path, valid_name="valid_split.json", chunk_type="chunk", max_chunk_len=max_chunk_len, model_type=model_type)
        preprocess_entire_or_chunk(dataset_name, save_dir, split_path=train_test_path, valid_name="valid_split.json", chunk_type="entire", min_entire_len=0, model_type=model_type)
        # label generation
        split_path = os.path.join(save_dir, "_".join(["chunk", str(max_chunk_len)]), "valid_split.json")
        generate_label_korean_multidisease(task_name, data_dir, split_path, save_dir, label_json_name, model_type=model_type)
        # transform
        transform_test_split_json(dataset_name, save_dir, label_json_name)
    elif task_name in ["circor2022_ternarymurmur", "circor2022_binarydisease"]:
        min_entire_len = 0
        max_chunk_len = 8
        # audio processing
        preprocess_entire_or_chunk(dataset_name, save_dir, split_path=train_test_path, valid_name="valid_split.json", chunk_type="chunk", max_chunk_len=max_chunk_len, model_type=model_type)
        preprocess_entire_or_chunk(dataset_name, save_dir, split_path=train_test_path, valid_name="valid_split.json", chunk_type="entire", min_entire_len=0, model_type=model_type)
        # label generation
        split_path = os.path.join(save_dir, "_".join(["chunk", str(max_chunk_len)]), "valid_split.json")
        generate_label_circor2022(task_name, data_dir, split_path, save_dir, label_json_name, model_type=model_type)
        # transform
        transform_test_split_json(dataset_name, save_dir, label_json_name)
    elif task_name in ["bowel_sound_binarysound"]:
        min_entire_len = 0
        max_chunk_len = 2
        # audio processing
        preprocess_entire_or_chunk(dataset_name, save_dir, split_path=train_test_path, valid_name="valid_split.json", chunk_type="chunk", max_chunk_len=max_chunk_len, model_type=model_type)
        preprocess_entire_or_chunk(dataset_name, save_dir, split_path=train_test_path, valid_name="valid_split.json", chunk_type="entire", min_entire_len=0, model_type=model_type)
        # label generation
        split_path = os.path.join(save_dir, "_".join(["chunk", str(max_chunk_len)]), "valid_split.json")
        generate_label_bowel_sound_binarysound(task_name, data_dir, split_path, save_dir, label_json_name, model_type=model_type)
        # transform
        transform_test_split_json(dataset_name, save_dir, label_json_name)