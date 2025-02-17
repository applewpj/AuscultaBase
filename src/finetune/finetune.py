import json
from glob import glob
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import LearningRateMonitor
import collections
from tqdm import tqdm
import torch.nn as nn
from datetime import datetime
import os, sys, pdb
from str2bool import str2bool
sys.path.append(os.getcwd())
from src.model.cola.cola import Cola, ColaMD
from src.model.pann.pann import Cnn14, Cnn14_no_specaug
from src.model.clap.CLAPWrapper import CLAPWrapper
from src.model.audiomae.audiomae import  vit_base_patch16
from src.model.model_eval import LinearHead
from src.dataset.dataset import FinetuneDataset


def str2none(value, raise_exc=False):
    if isinstance(value, str) or sys.version_info[0] < 3 and isinstance(value, basestring):
        value = value.lower()
        if value == "none":
            return None
        else:
            return value



def finetune(args, logger, l2_strength=1e-4, epochs=64, batch_size=64, label_json_path=None):
    l2_strength = args.l2_strength
    lr = args.lr
    head_type = args.head_type
    model_path = args.model_path
    model_type = args.model_type
    init_encoder = args.init_encoder
    train_encoder = args.train_encoder
    print("*" * 48)

    
    # Checkpoint callback
    save_topk = 1
    exp_name_part1 = " ".join(exp_name.split(" ")[:2])
    exp_name_part2 = " ".join(exp_name.split(" ")[2:])
    checkpoint_dir = os.path.join(args.checkpoint_dir, exp_name_part1, exp_name_part2)
    checkpoint_name = "_".join([head_type, str(batch_size), str(lr), str(epochs), str(l2_strength)]) + "-{epoch:02d}-{" + args.monitor_metric + ":.4f}"
    checkpoint_callback = ModelCheckpoint(
        # monitor="class0_specificity@0.9",
        monitor=args.monitor_metric,
        mode="max",
        save_top_k=save_topk,
        dirpath=checkpoint_dir,
        filename=checkpoint_name
    )

    
    # Dataset & Dataloader
    with open(label_json_path) as p:
        label_json = json.load(p)
    train_dataset = FinetuneDataset(label_json, split="train")
    val_dataset = FinetuneDataset(label_json, split="test")
    # pdb.set_trace()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=2, shuffle=False)

    #  Recording metrics
    recorded_metrics = []
    if train_dataset.task in ["classification", "multilabel"]:
        recorded_metrics.append("macro_f1")
        recorded_metrics.append("micro_f1")
        recorded_metrics.append("precision")
        recorded_metrics.append("recall")
    if train_dataset.n_classes == 2:
        recorded_metrics.append("auroc")
    # recorded_metrics.append("classwise-f1")
    # recorded_metrics.append("classwise-acc")
    # recorded_metrics.append("classwise-precision")
    # recorded_metrics.append("classwise-recall")


    # Model
    if train_dataset.task == "regression":
        loss_func = nn.MSELoss()
    elif train_dataset.task == "classification":
        loss_func = nn.CrossEntropyLoss()
    elif train_dataset.task == "multilabel":
        loss_func = nn.BCELoss()
    if model_type in ["auscultabase", "opera-ct"]:
        ckpt = torch.load(model_path)
        encoder_settings = ckpt["hyper_parameters"]
        encoder = ColaMD(encoder=encoder_settings["encoder"], max_len=encoder_settings["max_len"],
                        dim_fea=encoder_settings["dim_fea"], dim_hidden=encoder_settings["dim_hidden"],
                        dim_out=encoder_settings["dim_out"], num_batch=encoder_settings["num_batch"],
                        )
        model = LinearHead(encoder=encoder, n_classes=train_dataset.n_classes, task=train_dataset.task, feat_dim=768, head_type=head_type, lr=lr, loss_func=loss_func, l2_strength=l2_strength,  checkpoint_dir=checkpoint_dir, save_topk=save_topk, recorded_metrics=recorded_metrics, seed=args.seed, monitor_metric=args.monitor_metric)
        strict = True
    elif model_type == "pann":
        ckpt = torch.load(model_path)
        encoder = Cnn14_no_specaug(sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, fmin=50, fmax=14000, classes_num=527)
        model = LinearHead(encoder=encoder, n_classes=train_dataset.n_classes, task=train_dataset.task, feat_dim=2048, head_type=head_type, lr=lr, loss_func=loss_func, l2_strength=l2_strength,  checkpoint_dir=checkpoint_dir, save_topk=save_topk, recorded_metrics=recorded_metrics, seed=args.seed, monitor_metric=args.monitor_metric)
        strict = True
    elif model_type == "audiomae":
        ckpt = torch.load(model_path)
        encoder = vit_base_patch16(in_chans=1, img_size=(1024,128), drop_path_rate=0.1, global_pool=True, mask_2d=False, use_custom_patch=False)
        model = LinearHead(encoder=encoder, n_classes=train_dataset.n_classes, task=train_dataset.task, feat_dim=768, head_type=head_type, lr=lr, loss_func=loss_func, l2_strength=l2_strength,  checkpoint_dir=checkpoint_dir, save_topk=save_topk, recorded_metrics=recorded_metrics, seed=args.seed, monitor_metric=args.monitor_metric)
        strict = False
    elif model_type == "clap":
        encoder = CLAPWrapper(version="2022", use_cuda=True)
        model = LinearHead(encoder=encoder, n_classes=train_dataset.n_classes, task=train_dataset.task, feat_dim=1024, head_type=head_type, lr=lr, loss_func=loss_func, l2_strength=l2_strength,  checkpoint_dir=checkpoint_dir, save_topk=save_topk, recorded_metrics=recorded_metrics, seed=args.seed, monitor_metric=args.monitor_metric)
    if init_encoder and model_type != "clap":
        encoder_state = ckpt["state_dict"] if "state_dict" in ckpt.keys() else ckpt["model"]
        model.encoder.load_state_dict(encoder_state, strict=strict)
    if not train_encoder:
        if model_type == "clap":
            for p in model.encoder.clap.parameters():  p.requires_grad = False
        else:
            for p in model.encoder.parameters():  p.requires_grad = False
    

    # LR scheduler & Trainer
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu",
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        gradient_clip_val=1.0,
        log_every_n_steps=1,
        enable_progress_bar=False
    )


    # Start training
    trainer.fit(model, train_loader, val_loader)




if __name__ == "__main__":
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser()

    parser.add_argument("--task_name", type=str)
    parser.add_argument("--spec_dir", type=str)
    parser.add_argument("--model_type", type=str, choices=["auscultabase", "pann", "clap", "audiomae", "opera-ct"])
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--head_type", type=str, default="linear", choices=["linear", "tdnn+blstm+mlp"])
    parser.add_argument("--monitor_metric", type=str2none, default="macro_f1")
    parser.add_argument("--init_encoder", type=str2bool, default=True)
    parser.add_argument("--train_encoder", type=str2bool, default=False)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    # you can keep these parameters as default
    parser.add_argument("--lr", type=float, default=1e-4) 
    parser.add_argument("--l2_strength", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epoch", type=int, default=64)
    
    
    
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    
    label_json_path = os.path.join(args.spec_dir, "label", "_".join([args.task_name, "label.json"]))

    kwargs = {
        "epochs": args.epoch,
        "batch_size": 32,
    }
    
    # logger
    exp_name = args.task_name + " " + str(args.train_encoder) + " " + args.model_type + " " + f"seed{args.seed}" + " " + datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger = WandbLogger(
        project="BodySoundFinetune",
        name=exp_name,
    )
    logger.log_table(key="args", columns=["Key", "Value"], data=[[k, str(v)] for k, v in vars(args).items()])
    logger.log_table(key="kwargs", columns=["Key", "Value"], data=[[k, str(v)] for k, v in kwargs.items()])
    
    finetune(args, logger, label_json_path=label_json_path, **kwargs)






