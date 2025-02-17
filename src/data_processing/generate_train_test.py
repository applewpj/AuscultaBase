import glob as gb
import argparse, random
import numpy as np
import pandas as pd
from tqdm import tqdm
import os, json
from data_info import DATA_DIR_DICT, TEST_RATIO_DICT


def generate_train_test(dataset_name, data_dir, train_test_path, test_ratio=None):
    print(f"Generating train & test splits for {dataset_name}...")
    train_test_split = {}
    
    if dataset_name in ["xinhua_report", "xinhua_extra", "xinhua_labeled", "xinhua_unlabeled", "circor2022", "korean", "lung_sound", "RD@TR"]:
        wav_list = list(gb.glob(os.path.join(data_dir, "*.wav")))
        num_sample = len(wav_list)
        test_indices = random.sample(range(0, num_sample), int(num_sample*test_ratio))
        train_test_split = {}
        for i, wav_path in tqdm(enumerate(wav_list), total=len(wav_list)):
            if i in test_indices:
                train_test_split[wav_path] = "test"
            else:
                train_test_split[wav_path] = "train"
    elif dataset_name == "icbhi":
        txt_data = pd.read_csv(os.path.join(data_dir, 'ICBHI_challenge_train_test.txt'), dtype=str, sep='\t', names=['fileID', 'split'])
        wav_list = list(gb.glob(os.path.join(data_dir, "*.wav")))
        for i, wav_path in tqdm(enumerate(wav_list), total=len(wav_list)):
            wav_name = os.path.basename(wav_path)
            fileID = wav_name.split(".")[0]
            split = txt_data["split"][txt_data.fileID == fileID].values[0]
            train_test_split[wav_path] = split
    elif dataset_name == "cinc2016":
        for subname in os.listdir(data_dir):
            data_subdir = os.path.join(data_dir, subname)
            wav_list = list(gb.glob(os.path.join(data_subdir, "*.wav")))
            num_sample = len(wav_list)
            test_indices = random.sample(range(0, num_sample), int(num_sample*test_ratio))
            for i, wav_path in tqdm(enumerate(wav_list), total=len(wav_list)):
                if i in test_indices:
                    train_test_split[wav_path] = "test"
                else:
                    train_test_split[wav_path] = "train"
    elif dataset_name == "sprsound":
        # train
        data_subdir = os.path.join(data_dir, "train_wav")
        wav_list = os.listdir(data_subdir)
        for i, wav_name in tqdm(enumerate(wav_list), total=len(wav_list)):
            wav_path = os.path.join(data_subdir, wav_name)
            train_test_split[wav_path] = "train"
        # test_2022
        splits = ["test_wav_2022", "test_wav_2023"]
        for split in splits:
            data_subdir = os.path.join(data_dir, split)
            wav_list = os.listdir(data_subdir)
            for i, wav_name in tqdm(enumerate(wav_list), total=len(wav_list)):
                wav_path = os.path.join(data_subdir, wav_name)
                train_test_split[wav_path] = "test"
    elif dataset_name == "hf_lung":
        subname_list = ["HF_Lung_V1-master", "HF_Lung_V1_IP-main"]
        for subname in subname_list:
            for split in ["train", "test"]:
                data_subdir = os.path.join(data_dir, subname, split)
                wav_list = list(gb.glob(os.path.join(data_subdir, "*.wav")))
                for i, wav_path in tqdm(enumerate(wav_list), total=len(wav_list)):
                    train_test_split[wav_path] = split
    elif dataset_name == "bowel_sound":
        with open(os.path.join(data_dir, "files.csv")) as f:
            csv_data = pd.read_csv(f)
        filename_list = csv_data["filename"]
        split_list = csv_data["train/test"]
        sounds_amount_list = csv_data["sounds_amount"]
        for i, (wav_name, split) in tqdm(enumerate(zip(filename_list, split_list)), total=len(csv_data)):
            wav_path = os.path.join(data_dir, wav_name)
            train_test_split[wav_path] = split
            
    save_path = train_test_path
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    with open(save_path, 'w') as f:
        json.dump(train_test_split, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--train_test_path", type=str)

    args = parser.parse_args()
    json_name = "train_test_split.json"

    generate_train_test(
            args.dataset_name,
            DATA_DIR_DICT[args.dataset_name],
            train_test_path=args.train_test_path,
            test_ratio=TEST_RATIO_DICT[args.dataset_name]
        )