
import torch
import numpy as np
import os
import torchaudio


class FinetuneDataset(torch.utils.data.Dataset):
    def __init__(self, split_label_json, split="train"):
        self.split = split
        # if self.split == "test":
        #     for key, value in split_label_json.items():
        #         if key not in ["n_classes", "classes", "task"] and value[0] == "train":
        #             spec_dirname = os.path.basename(os.path.dirname(key))
        #             self.if_chunk_len_s = int(spec_dirname.split("_")[-1])
        #             break
        #     self.if_interval_len_s = self.if_chunk_len_s / 2
        self.classes = split_label_json["classes"]
        self.n_classes = split_label_json["n_classes"]
        self.task = split_label_json["task"]
        
        self.file_list = []
        self.label_list = []

        for key, value in split_label_json.items():
            if key not in ["n_classes", "classes", "task"]:
                spec_path = key
                spec_split, spec_label = value
                if spec_split == split:
                    self.file_list.append(spec_path)
                    if self.task in ["classification", "regression"]:
                        self.label_list.append(spec_label)
                    elif self.task == "multilabel":
                        self.label_list.append(torch.tensor(spec_label, dtype=torch.float))

                    if self.split == "test":
                        pass

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        
        filepath = self.file_list[idx]
        label = self.label_list[idx]
        
        if filepath.endswith(".npy"):
            x = np.load(filepath)
            x = torch.tensor(x, dtype=torch.float)
        elif filepath.endswith(".wav"):
            x, sr = torchaudio.load(filepath)
            x = x.squeeze()
        elif filepath.endswith(".pt"):
            x = torch.load(filepath)
        
        return x, label



