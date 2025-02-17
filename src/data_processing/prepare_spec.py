import glob as gb
import argparse, random
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import os, sys, pdb, json, torchaudio, librosa, torch

from utils import get_entire_signal_librosa, get_split_signal_librosa, get_split_signal_fbank, get_entire_signal_fbank


def compute_statistics(train_test_splits):
    num_sample, min_len, max_len, total_len = 0, 999999, 0, 0
    sr_list = []
    for idx, wav_path in tqdm(enumerate(train_test_splits.keys()), total=len(train_test_splits)):
        # wav_path = os.path.join(data_dir, wav_name)
        wav, sr = torchaudio.load(wav_path)
        wav_len = wav.shape[-1] / sr
        
        num_sample += 1
        total_len += wav_len
        min_len = wav_len if wav_len < min_len else min_len
        max_len = wav_len if wav_len > max_len else max_len
        if sr not in sr_list:
            sr_list.append(sr)
    avg_len = total_len / num_sample
    # pdb.set_trace()
    return {
        "min_len": min_len,
        "max_len": max_len,
        "total_len": total_len,
        "avg_len": avg_len,
        "num_sample": num_sample,
        "sr_list": sr_list,
    }





# Preprocess audio -> spectrogram for:
# 1. entire audio
# 2. chunked audio
def preprocess_entire_or_chunk(dataset_name, save_dir, split_path, valid_name, chunk_type="entire", min_entire_len=None, max_chunk_len=None, model_type=None):
    assert chunk_type in ["entire", "chunk"]
    print(f"Preprocessing for {dataset_name}...")
    with open(split_path, 'r', encoding='utf-8') as f:
        train_test_splits = json.load(f)
    
    if chunk_type == "chunk":
        chunk_name = "_".join([chunk_type, str(max_chunk_len)])
    elif chunk_type == "entire":
        chunk_name = chunk_type
    save_subdir = os.path.join(save_dir, chunk_name)
    
    if os.path.exists(save_subdir):
        os.system(f"rm -rf {save_subdir}")
        os.makedirs(save_subdir)
    else:
        os.makedirs(save_subdir)
    
    # Statistical characteristics
    # statistics = compute_statistics(train_test_splits)
    
    invalid_data = 0
    valid_json = {}
    if model_type == "pann":
        sr = 32000
    elif model_type == "clap":
        sr = 44100
    else:
        sr = 16000
    for idx, (wav_path, split) in tqdm(enumerate(train_test_splits.items()), total=len(train_test_splits)):
        try:
            data, rate = librosa.load(wav_path, sr=sr)
        except:
            print("Warning: failed to load audio, skipped")
            invalid_data += 1
            continue
        try:
            assert 0 not in data.shape
            assert data.size > 0
        except:
            print(f"Warning: audio shape is {data.shape}, skipped")
            invalid_data += 1
            continue
        
        wav_name = os.path.basename(wav_path)
        fileID = wav_name.rsplit(".pt")[0].rsplit(".wav")[0].rsplit(".npy")[0]
        if model_type in ["auscultabase", "opera-ct"]:
            if chunk_type == "entire":
                data = get_entire_signal_librosa(data, sample_rate=sr, spectrogram=True, input_sec=min_entire_len, pad_type="zero")
                npy_path = os.path.join(save_subdir, fileID + ".npy")
                np.save(npy_path, data)
                valid_json[npy_path] = split
            elif chunk_type == "chunk":
                data = get_split_signal_librosa(data, sample_rate=sr, spectrogram=True, input_sec=max_chunk_len, pad_type="zero")
                num_chunk = len(data)
                for idx in range(num_chunk):
                    npy_path = os.path.join(save_subdir, fileID + "_" + str(idx) + ".npy")
                    np.save(npy_path, data[idx])
                    valid_json[npy_path] = split
        elif model_type in ["pann", "clap"]:
            if chunk_type == "entire":
                # pdb.set_trace()
                data = get_entire_signal_librosa(data, sample_rate=sr, spectrogram=False, input_sec=min_entire_len, pad_type="zero")
                audio_path = os.path.join(save_subdir, fileID + ".wav")
                data_save = torch.tensor(data).unsqueeze(0)
                torchaudio.save(audio_path, data_save, sample_rate=sr)
            elif chunk_type == "chunk":
                data = get_split_signal_librosa(data, sample_rate=sr, spectrogram=False, input_sec=max_chunk_len, pad_type="zero")
                num_chunk = len(data)
                for idx in range(num_chunk):
                    audio_path = os.path.join(save_subdir, fileID + "_" + str(idx) + ".wav")
                    data_save = torch.tensor(data[idx]).unsqueeze(0)
                    torchaudio.save(audio_path, data_save, sample_rate=sr)
                    valid_json[audio_path] = split
        elif model_type == "audiomae":
            if chunk_type == "entire":
                data = get_entire_signal_fbank(data, sample_rate=sr)
                pt_path = os.path.join(save_subdir, fileID + ".pt")
                # pdb.set_trace()
                torch.save(data, pt_path)
                valid_json[pt_path] = split
            elif chunk_type == "chunk":
                data = get_split_signal_fbank(data, sample_rate=sr, input_sec=max_chunk_len)
                num_chunk = len(data)
                for idx in range(num_chunk):
                    pt_path = os.path.join(save_subdir, fileID + "_" + str(idx) + ".pt")
                    torch.save(data[idx], pt_path)
                    valid_json[pt_path] = split
    with open(os.path.join(save_subdir, valid_name), 'w') as f:
        json.dump(valid_json, f)

    print("invalid_data", invalid_data)



# Preprocess audio for sprsound_binarylungsound & sprsound_multilungsound
def preprocess_sprsound_event(save_dir, split_path, valid_name, data_dir, chunk_type="entire", min_entire_len=None, max_chunk_len=None, model_type=None):
    assert chunk_type in ["entire", "chunk"]
    print(f"Preprocessing for sprsound event {chunk_type}...")
    dataset_name = "sprsound"
    with open(split_path, 'r', encoding='utf-8') as f:
        train_test_splits = json.load(f)
    
    if chunk_type == "chunk":
        chunk_name = "_".join(["event", chunk_type, str(max_chunk_len)])
    elif chunk_type == "entire":
        chunk_name = "_".join(["event", chunk_type])
    save_subdir = os.path.join(save_dir, chunk_name)
    
    if os.path.exists(save_subdir):
        os.system(f"rm -rf {save_subdir}")
    os.makedirs(save_subdir)
    
    invalid_data = 0
    valid_json = {}
    if model_type == "pann":
        sr = 32000
    elif model_type == "clap":
        sr = 44100
    else:
        sr = 16000
    for idx, (wav_path, split) in tqdm(enumerate(train_test_splits.items()), total=len(train_test_splits)):
        # Loading audio
        try:
            data, rate = librosa.load(wav_path, sr=sr)
        except:
            print("Warning: failed to load audio, skipped")
            invalid_data += 1
            continue
        try:
            assert 0 not in data.shape
            assert data.size > 0
        except:
            print(f"Warning: audio shape is {data.shape}, skipped")
            invalid_data += 1
            continue
        # Loading label
        wav_name = os.path.basename(wav_path)
        label_name = wav_name.rsplit(".pt")[0].rsplit(".wav")[0].rsplit(".npy")[0] + ".json"
        label_subdirs = [os.path.join(data_dir, i) for i in ["train_json", "test_json_2023", "test_json_2022/inter_test_json", "test_json_2022/intra_test_json"]]
        label_json = None
        for label_subdir in label_subdirs:
            if label_name in os.listdir(label_subdir):
                label_path = os.path.join(label_subdir, label_name)
                with open(label_path, 'r', encoding='utf-8') as f:
                    label_json = json.load(f)
        assert "record_annotation" in label_json.keys() and "event_annotation" in label_json.keys()
        if len(label_json["event_annotation"]) == 0:
            continue
        
        for event_idx, label in enumerate(label_json["event_annotation"]):
            start = int(int(label["start"]) / 1000 * sr)
            end = int(int(label["end"]) / 1000 * sr)
            event_type = label["type"]
            segment = data[start:end]
            fileID = wav_name.rsplit(".pt")[0].rsplit(".wav")[0].rsplit(".npy")[0]
            if model_type in ["auscultabase", "opera-ct"]:
                if chunk_type == "entire":
                    segment = get_entire_signal_librosa(segment, sample_rate=sr, spectrogram=True, input_sec=min_entire_len, pad_type="zero")
                    npy_path = os.path.join(save_subdir, fileID + f"_event{str(event_idx)}" + ".npy")
                    np.save(npy_path, segment)
                    valid_json[npy_path] = split
                elif chunk_type == "chunk":
                    segment = get_split_signal_librosa(segment, sample_rate=sr, spectrogram=True, input_sec=max_chunk_len, pad_type="zero")
                    num_chunk = len(segment)
                    for idx in range(num_chunk):
                        npy_path = os.path.join(save_subdir, fileID + f"_event{str(event_idx)}" + f"_{str(idx)}" + ".npy")
                        np.save(npy_path, segment[idx])
                        valid_json[npy_path] = split
            elif model_type in ["pann", "clap"]:
                if chunk_type == "entire":
                    segment = get_entire_signal_librosa(segment, sample_rate=sr, spectrogram=False, input_sec=min_entire_len, pad=True, pad_type="zero")
                    audio_path = os.path.join(save_subdir, fileID + f"_event{str(event_idx)}" + ".wav")
                    segment_save = torch.tensor(segment).unsqueeze(0)
                    torchaudio.save(audio_path, segment_save, sample_rate=sr)
                    valid_json[audio_path] = split
                elif chunk_type == "chunk":
                    segment = get_split_signal_librosa(segment, sample_rate=sr, spectrogram=False, input_sec=max_chunk_len, pad_type="zero")
                    num_chunk = len(segment)
                    for idx in range(num_chunk):
                        audio_path = os.path.join(save_subdir, fileID + f"_event{str(event_idx)}" + f"_{str(idx)}" + ".wav")
                        segment_save = torch.tensor(segment[idx]).unsqueeze(0)
                        torchaudio.save(audio_path, segment_save, sample_rate=sr)
                        valid_json[audio_path] = split
            elif model_type == "audiomae":
                if chunk_type == "entire":
                    segment = get_entire_signal_fbank(segment, sample_rate=sr)
                    pt_path = os.path.join(save_subdir, fileID + f"_event{str(event_idx)}" + ".pt")
                    torch.save(segment, pt_path)
                    valid_json[pt_path] = split
                elif chunk_type == "chunk":
                    segment = get_split_signal_fbank(segment, sample_rate=sr, input_sec=max_chunk_len)
                    num_chunk = len(segment)
                    for idx in range(num_chunk):
                        pt_path = os.path.join(save_subdir, fileID + f"_event{str(event_idx)}" + f"_{str(idx)}" + ".pt")
                        torch.save(segment[idx], pt_path)
                        valid_json[pt_path] = split

    with open(os.path.join(save_subdir, valid_name), 'w') as f:
        json.dump(valid_json, f)
    # pdb.set_trace()
    print("invalid_data", invalid_data)



def preprocess_icbhi_lungsound(save_dir, split_path, valid_name, chunk_type="entire", min_entire_len=None, max_chunk_len=None, model_type=None):
    assert chunk_type in ["entire", "chunk"]
    print(f"Preprocessing for icbhi_lungsound {chunk_type}...")
    dataset_name = "icbhi"
    # split_path = os.path.join(save_dir, dataset_name, json_name)
    with open(split_path, 'r', encoding='utf-8') as f:
        train_test_splits = json.load(f)
    
    # save_subdir = save_dir
    if chunk_type == "entire":
        save_subdir = os.path.join(save_dir, "_".join(["cycle", "entire"]))
    elif chunk_type =="chunk":
        save_subdir = os.path.join(save_dir, "_".join(["cycle", "chunk", str(max_chunk_len)]))
    
    if os.path.exists(save_subdir):
        os.system(f"rm -rf {save_subdir}")
    os.makedirs(save_subdir)
    
    invalid_data = 0
    valid_json = {}
    if model_type == "pann":
        sr = 32000
    elif model_type == "clap":
        sr = 44100
    else:
        sr = 16000
    for idx, (wav_path, split) in tqdm(enumerate(train_test_splits.items()), total=len(train_test_splits)):
        # Loading audio
        try:
            data, rate = librosa.load(wav_path, sr=sr)
        except:
            print("Warning: failed to load audio, skipped")
            invalid_data += 1
            continue
        try:
            assert 0 not in data.shape
            assert data.size > 0
        except:
            print(f"Warning: audio shape is {data.shape}, skipped")
            invalid_data += 1
            continue
        # Loading label
        wav_name = os.path.basename(wav_path)
        label_path = wav_path.rsplit(".pt")[0].rsplit(".wav")[0].rsplit(".npy")[0] + ".txt"
        with open(label_path, 'r', encoding='utf-8') as f:
            labels = f.readlines()
        
        for segment_idx, segment_label in enumerate(labels):
            segment_label = segment_label.split()
            start = int(float(segment_label[0]) * sr)
            end = int(float(segment_label[1]) * sr)
            label_0 = int(segment_label[2])
            label_1 = int(segment_label[3])
            
            segment = data[start:end]
            fileID = wav_name.rsplit(".pt")[0].rsplit(".wav")[0].rsplit(".npy")[0]
            # aa.append(segment.size)
            if model_type in ["auscultabase", "opera-ct"]:
                if chunk_type == "entire":
                    segment = get_entire_signal_librosa(segment, sample_rate=sr, spectrogram=True, input_sec=min_entire_len, pad_type="zero")
                    npy_path = os.path.join(save_subdir, fileID + f"_segment{str(segment_idx)}" + ".npy")
                    np.save(npy_path, segment)
                    valid_json[npy_path] = split
                elif chunk_type == "chunk":
                    segment = get_split_signal_librosa(segment, sample_rate=sr, spectrogram=True, input_sec=max_chunk_len, pad_type="zero")
                    num_chunk = len(segment)
                    for idx in range(num_chunk):
                        npy_path = os.path.join(save_subdir, fileID + f"_segment{str(segment_idx)}" + f"_{str(idx)}" + ".npy")
                        np.save(npy_path, segment[idx])
                        valid_json[npy_path] = split
            elif model_type in ["pann", "clap"]:
                if chunk_type == "entire":
                    # pdb.set_trace()
                    segment = get_entire_signal_librosa(segment, sample_rate=sr, spectrogram=False, input_sec=min_entire_len, pad=True, pad_type="zero")
                    audio_path = os.path.join(save_subdir, fileID + f"_segment{str(segment_idx)}" + ".wav")
                    data_save = torch.tensor(segment).unsqueeze(0)
                    torchaudio.save(audio_path, data_save, sample_rate=sr)
                    valid_json[audio_path] = split
                elif chunk_type == "chunk":
                    segment = get_split_signal_librosa(segment, sample_rate=sr, spectrogram=False, input_sec=max_chunk_len, pad_type="zero")
                    num_chunk = len(segment)
                    for idx in range(num_chunk):
                        audio_path = os.path.join(save_subdir, fileID + f"_segment{str(segment_idx)}" + f"_{str(idx)}" + ".wav")
                        data_save = torch.tensor(segment[idx]).unsqueeze(0)
                        torchaudio.save(audio_path, data_save, sample_rate=sr)
                        valid_json[audio_path] = split
            elif model_type == "audiomae":
                if chunk_type == "entire":
                    segment = get_entire_signal_fbank(segment, sample_rate=sr)
                    pt_path = os.path.join(save_subdir, fileID + f"_segment{str(segment_idx)}" + ".pt")
                    # pdb.set_trace()
                    torch.save(segment, pt_path)
                    valid_json[pt_path] = split
                elif chunk_type == "chunk":
                    segment = get_split_signal_fbank(segment, sample_rate=sr, input_sec=max_chunk_len)
                    num_chunk = len(segment)
                    for idx in range(num_chunk):
                        pt_path = os.path.join(save_subdir, fileID + f"_segment{str(segment_idx)}" + f"_{str(idx)}" + ".pt")
                        torch.save(segment[idx], pt_path)
                        valid_json[pt_path] = split
    with open(os.path.join(save_subdir, valid_name), 'w') as f:
        json.dump(valid_json, f)
    # pdb.set_trace()
    print("invalid_data", invalid_data)



def preprocess_hf_lung_lungsound(save_dir, split_path, valid_name, chunk_type="entire", min_entire_len=None, max_chunk_len=None, model_type=None):
    assert chunk_type in ["entire", "chunk"]
    print(f"Preprocessing for hf_lung_lungsound {chunk_type}...")
    dataset_name = "hf_lung"
    with open(split_path, 'r', encoding='utf-8') as f:
        train_test_splits = json.load(f)
    
    # save_subdir = save_dir
    if chunk_type == "entire":
        save_subdir = os.path.join(save_dir, "_".join(["event", "entire"]))
    elif chunk_type =="chunk":
        save_subdir = os.path.join(save_dir, "_".join(["event", "chunk", str(max_chunk_len)]))
    
    if os.path.exists(save_subdir):
        os.system(f"rm -rf {save_subdir}")
    os.makedirs(save_subdir)
    
    invalid_data = 0
    valid_json = {}
    if model_type == "pann":
        sr = 32000
    elif model_type == "clap":
        sr = 44100
    else:
        sr = 16000
    for idx, (wav_path, split) in tqdm(enumerate(train_test_splits.items()), total=len(train_test_splits)):
        if "HF_Lung_V1-master" in wav_path:  # 只有这一部分有label
            # Loading audio
            try:
                all_data, rate = librosa.load(wav_path, sr=sr)
            except:
                print("Warning: failed to load audio, skipped")
                invalid_data += 1
                continue
            try:
                assert all_data.size > 0
            except:
                print(f"Warning: audio shape is {all_data.shape}, skipped")
                invalid_data += 1
                continue
            # Loading label
            wav_name = os.path.basename(wav_path)
            label_path = wav_path.rsplit(".pt")[0].rsplit(".wav")[0].rsplit(".npy")[0] + "_label.txt"
            with open(label_path, 'r', encoding='utf-8') as f:
                labels = f.readlines()
            
            for segment_idx, segment_label in enumerate(labels):
                segment_label = segment_label.split()
                start = int(float(segment_label[1][-6:]) * sr)
                end = int(float(segment_label[2][-6:]) * sr)
                
                segment = all_data[start:end]
                fileID = wav_name.rsplit(".pt")[0].rsplit(".wav")[0].rsplit(".npy")[0]
                if model_type in ["auscultabase", "opera-ct"]:
                    if chunk_type == "entire":
                        segment = get_entire_signal_librosa(segment, sample_rate=sr, spectrogram=True, input_sec=min_entire_len, pad_type="zero")
                        
                        npy_path = os.path.join(save_subdir, fileID + f"_segment{str(segment_idx)}" + ".npy")
                        np.save(npy_path, segment)
                        valid_json[npy_path] = split
                    elif chunk_type == "chunk":
                        segment = get_split_signal_librosa(segment, sample_rate=sr, spectrogram=True, input_sec=max_chunk_len, pad_type="zero")
                        
                        num_chunk = len(segment)
                        for idx in range(num_chunk):
                            npy_path = os.path.join(save_subdir, fileID + f"_segment{str(segment_idx)}" + f"_{str(idx)}" + ".npy")
                            np.save(npy_path, segment[idx])
                            valid_json[npy_path] = split
                elif model_type in ["pann", "clap"]:
                    if chunk_type == "entire":
                        data = get_entire_signal_librosa(segment, sample_rate=sr, spectrogram=False, input_sec=min_entire_len, pad_type="zero")
                        audio_path = os.path.join(save_subdir, fileID + f"_segment{str(segment_idx)}" + ".wav")
                        data_save = torch.tensor(data).unsqueeze(0)
                        torchaudio.save(audio_path, data_save, sample_rate=sr)
                        valid_json[audio_path] = split
                    elif chunk_type == "chunk":
                        data = get_split_signal_librosa(segment, sample_rate=sr, spectrogram=False, input_sec=max_chunk_len, pad_type="zero")
                        num_chunk = len(data)
                        for idx in range(num_chunk):
                            audio_path = os.path.join(save_subdir, fileID + f"_segment{str(segment_idx)}" + f"_{str(idx)}" + ".wav")
                            data_save = torch.tensor(data[idx]).unsqueeze(0)
                            torchaudio.save(audio_path, data_save, sample_rate=sr)
                            valid_json[audio_path] = split
                elif model_type == "audiomae":
                    if chunk_type == "entire":
                        data = get_entire_signal_fbank(segment, sample_rate=sr)
                        pt_path = os.path.join(save_subdir, fileID + f"_segment{str(segment_idx)}" + ".pt")
                        torch.save(data, pt_path)
                        valid_json[pt_path] = split
                    elif chunk_type == "chunk":
                        data = get_split_signal_fbank(segment, sample_rate=sr, input_sec=max_chunk_len)
                        num_chunk = len(data)
                        for idx in range(num_chunk):
                            pt_path = os.path.join(save_subdir, fileID + f"_segment{str(segment_idx)}" + f"_{str(idx)}" + ".pt")
                            torch.save(data[idx], pt_path)
                            valid_json[pt_path] = split
    with open(os.path.join(save_subdir, valid_name), 'w') as f:
        json.dump(valid_json, f)
    
    # print("invalid_data", invalid_data)







if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_list", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--chunk_type", type=str, default="entire")
    parser.add_argument("--min_entire_len", type=float, default=0.0)
    parser.add_argument("--max_chunk_len", type=float, default=8.0)
    args = parser.parse_args()
    
    
    dataset_list = args.dataset_list.split(",")
    for dataset_name in dataset_list:
        json_name = "train_test_split.json"
        train_test_path = os.path.join(args.save_dir, dataset_name, json_name)

        preprocess_entire_or_chunk(dataset_name, args.save_dir, train_test_path, valid_name="valid_split.json", chunk_type=args.chunk_type, min_entire_len=args.min_entire_len, max_chunk_len=args.max_chunk_len)