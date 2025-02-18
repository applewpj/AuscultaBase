from networkx import dfs_edges
import pytorch_lightning as pl
import torch, os, sys
from efficientnet_pytorch import EfficientNet
from torch.nn import functional as F
from sklearn import metrics
import torch.nn as nn
import numpy as np
from torchmetrics import AUROC
from src.model.cola.htsat.htsat import HTSATWrapper
import pandas as pd
from sklearn.metrics import f1_score, recall_score, roc_auc_score, confusion_matrix, roc_curve, precision_score
from torch.optim.lr_scheduler import StepLR
import json, time

from src.model.cola.cola import Cola, ColaMD
from src.model.audiomae.audiomae import VisionTransformer
from src.model.pann.pann import Cnn14, Cnn14_no_specaug
from src.model.clap.CLAPWrapper import CLAPWrapper
from src.model.tdnn.TDNN import TDNN
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


def init_weights(module):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class TDNN_BLSTM_MLP_Head(nn.Module):
    def __init__(self, feat_dim, n_classes):
        super().__init__()
        self.tdnn = nn.Sequential(
                    TDNN(64, 512, context_size=5, stride=1,dilation=1),
                    TDNN(512, 512, context_size=3, stride=1,dilation=2), 
                    TDNN(512, 512, context_size=3, stride=1,dilation=3), 
                    TDNN(512, 128, context_size=1, stride=1,dilation=1)
                )
        self.lstm = nn.LSTM(128, 128, batch_first=True,bidirectional=True)
        #self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Sequential(nn.Linear(256, 1024),
                                    # nn.ReLU(),
                                    nn.Linear(1024, n_classes)
                                    )
    
    def forward(self, embedding):
        if len(embedding.shape) == 2:
            embedding = embedding.unsqueeze(0)
        lengths = [embedding.size(1)] * embedding.size(0)
        fea = self.tdnn(embedding)
        packed_input = pack_padded_sequence(fea, lengths, batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_input)
        x, lstm_len = pad_packed_sequence(packed_output, batch_first=True)
        x = x.sum(dim=1) / torch.tensor(lengths, device=x.device).unsqueeze(1)
        x = self.classifier(x).squeeze(1)
        return x


class LinearHead(pl.LightningModule):
    def __init__(self, encoder=None, n_classes=2, task="classification", head_type="linear", feat_dim=1280, lr=1e-4, loss_func=None, l2_strength=0.0005, checkpoint_dir=None, save_topk=0, recorded_metrics=None, seed=None, monitor_metric=None):
        super().__init__()

        self.encoder = encoder
        self.head_type = head_type
        if head_type == 'linear':
            self.head = nn.Sequential(nn.Linear(feat_dim, n_classes))
        elif head_type == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.ReLU(inplace=True),
                nn.Linear(feat_dim, n_classes)
            )
        elif head_type == "tdnn+blstm+mlp":
            self.head = TDNN_BLSTM_MLP_Head(feat_dim, n_classes)
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head_type))
        self.apply(init_weights)
        
        self.lr = lr
        self.l2_strength = l2_strength
        self.loss_func = loss_func if loss_func else nn.CrossEntropyLoss()
        self.n_classes = n_classes
        assert task in ["classification", "multilabel", "regression"]
        self.task = task
        if self.task == "multilabel":
            self.sigmoid = nn.Sigmoid()
        elif self.task == "regression":
            self.relu = nn.ReLU()
        
        self.validation_step_outputs = []

        self.save_topk = save_topk
        if self.save_topk > 0:
            self.checkpoint_dir = checkpoint_dir
            self.topk_dict = {0.00001*i: {"save_path": ""} for i in range(save_topk)}
        
        self.recorded_metrics = recorded_metrics
        if self.recorded_metrics is not None:
            self.current_recorded_metrics = {k:0.0 for k in self.recorded_metrics}
            self.best_recorded_metrics = {k:0.0 for k in self.recorded_metrics}
        self.monitor_metric = monitor_metric
        self.seed = seed
            
            
    def forward(self, x):
        if self.head_type == "linear":
            if isinstance(self.encoder, ColaMD):
                x = self.encoder.extract_feature(x)
            elif isinstance(self.encoder, Cnn14_no_specaug):
                x = self.encoder(x)["embedding"]
            elif isinstance(self.encoder, VisionTransformer):
                x = self.encoder.forward_feature(x)
            elif isinstance(self.encoder, CLAPWrapper):
                x = self.encoder.clap.audio_encoder(x)[0]
        elif self.head_type == "tdnn+blstm+mlp":
            if isinstance(self.encoder, ColaMD):
                x = self.encoder.encoder.encoder.htsat(x.unsqueeze(1))["sequence_output"]

        x = self.head(x)
        

        if self.task == "multilabel":
            x = self.sigmoid(x)
        elif self.task == "regression":
            x = self.relu(x)
        return x
        

    def training_step(self, batch, batch_idx):
        x, y = batch
        if self.task == "regression":
            y = y.unsqueeze(1).float()
        elif self.task == "classification":
            y = torch.tensor(y, dtype=torch.long)
        elif self.task == "multilabel":
            y = torch.tensor(y, dtype=torch.float)
        y_hat = self(x)

        
        loss = self.loss_func(y_hat, y)
        self.log("train_loss", loss)
        # Apply L2 regularization
        l2_regularization = 0
        for param in self.head.parameters():
            l2_regularization += param.pow(2).sum()
        self.log("train_l2", l2_regularization)
        self.log("train_loss", loss)
        loss += self.l2_strength * l2_regularization
        self.log("total_loss", loss)
        if self.task == "classification":
            _, predicted = torch.max(y_hat, 1)
            acc = (predicted == y).double().mean()
        elif self.task == "regression":
            predicted = torch.round(y_hat)
            acc = (predicted == y).double().mean()
        elif self.task == "multilabel":
            probabilities = y_hat
            predicted = torch.where(probabilities < 0.5, 0, 1)
            micro_f1s = []
            for class_idx in range(y.shape[-1]):
                micro_f1s.append(f1_score(y[:, class_idx].cpu(), predicted[:, class_idx].cpu(), average="micro"))
            acc = sum(micro_f1s) / len(micro_f1s)

        self.log("train_acc", acc)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
            
        if self.task == "regression":
            y = y.unsqueeze(1).float()
        elif self.task == "classification":
            y = torch.tensor(y, dtype=torch.long)
        elif self.task == "multilabel":
            y = torch.tensor(y, dtype=torch.float)

        y_hat = self(x)
        
        loss = self.loss_func(y_hat, y)

        if self.task == "classification":
            probabilities = F.softmax(y_hat.cpu(), dim=1)
            _, predicted = torch.max(y_hat.cpu(), 1)
        elif self.task == "regression":
            probabilities = None
            predicted = torch.round(y_hat.cpu())
        elif self.task == "multilabel":
            probabilities = y_hat.cpu()
            predicted = torch.where(probabilities < 0.5, 0, 1)
        # pdb.set_trace()
        self.validation_step_outputs.append((y.cpu(), predicted, probabilities, loss.item()))

    
    def obtain_metric_specificity(self, sen_target_list, todo_probs, y):
        sen_real_list, spe_real_list, th_real_list = [], [], []
        for sen_target in sen_target_list:
            fpr, tpr, thresholds = roc_curve(y, todo_probs, pos_label=1)
            sensitivity_index = np.argmax(tpr >= sen_target)
            th = thresholds[sensitivity_index]
            sen = tpr[sensitivity_index]
            spe = 1 - fpr[sensitivity_index]
            
            sen_real_list.append(sen)
            spe_real_list.append(spe)
            th_real_list.append(th)

        return sen_real_list, spe_real_list, th_real_list

    def record_metrics(self, y, predicted, probs):
        if "precision" in self.recorded_metrics:
            precision = precision_score(y, predicted, average="macro")
            self.current_recorded_metrics["precision"] = precision
            self.best_recorded_metrics["precision"] = max(self.best_recorded_metrics["precision"], self.current_recorded_metrics["precision"])
            self.log("precision", self.current_recorded_metrics["precision"])
            self.log("best_precision", self.best_recorded_metrics["precision"])
        if "recall" in self.recorded_metrics:
            recall = recall_score(y, predicted, average="macro")
            self.current_recorded_metrics["recall"] = recall
            self.best_recorded_metrics["recall"] = max(self.best_recorded_metrics["recall"], self.current_recorded_metrics["recall"] )
            self.log("recall", self.current_recorded_metrics["recall"] )
            self.log("best_recall", self.best_recorded_metrics["recall"])
        if "micro_f1" in self.recorded_metrics:
            micro_f1 = f1_score(y, predicted, average="micro")
            self.current_recorded_metrics["micro_f1"] = micro_f1
            self.best_recorded_metrics["micro_f1"] = max(self.best_recorded_metrics["micro_f1"], self.current_recorded_metrics["micro_f1"])
            self.log("micro_f1", self.current_recorded_metrics["micro_f1"])
            self.log("best_micro_f1", self.best_recorded_metrics["micro_f1"])
        if "macro_f1" in self.recorded_metrics:
            macro_f1 = f1_score(y, predicted, average="macro")
            self.current_recorded_metrics["macro_f1"] = macro_f1
            self.best_recorded_metrics["macro_f1"] = max(self.best_recorded_metrics["macro_f1"], self.current_recorded_metrics["macro_f1"])
            self.log("macro_f1", self.current_recorded_metrics["macro_f1"])
            self.log("best_macro_f1", self.best_recorded_metrics["macro_f1"])
        if "auroc" in self.recorded_metrics:
            auroc = roc_auc_score(y, probs[:, 1])
            self.current_recorded_metrics["auroc"] = auroc
            self.best_recorded_metrics["auroc"] = max(self.best_recorded_metrics["auroc"], self.current_recorded_metrics["auroc"])
            self.log("auroc", self.current_recorded_metrics["auroc"])
            self.log("best_auroc", self.best_recorded_metrics["auroc"])
        if "classwise-f1" in self.recorded_metrics:
            classwise_f1s = f1_score(y, predicted, average=None)
            self.current_recorded_metrics["classwise-f1"] = classwise_f1s
            for i, score in enumerate(self.current_recorded_metrics["classwise-f1"]):
                self.log(f"classwise_f1_class{i}", score)
        if "classwise-precision" in self.recorded_metrics:
            classwise_precisions = precision_score(y, predicted, average=None)
            self.current_recorded_metrics["classwise-precision"] = classwise_precisions
            for i, score in enumerate(self.current_recorded_metrics["classwise-precision"]):
                self.log(f"classwise_precision_class{i}", score)
        if "classwise-recall" in self.recorded_metrics:
            classwise_recalls = recall_score(y, predicted, average=None)
            self.current_recorded_metrics["classwise-recall"] = classwise_recalls
            for i, score in enumerate(self.current_recorded_metrics["classwise-recall"]):
                self.log(f"classwise_recall_class{i}", score)
            
    def on_validation_epoch_end(self):
        all_outputs = self.validation_step_outputs
        y = torch.cat([output[0] for output in all_outputs], dim=0)
        predicted = torch.cat([output[1] for output in all_outputs], dim=0)
        probs = torch.cat([output[2] for output in all_outputs], dim=0) if all_outputs[0][2] is not None else None
        losses = torch.tensor([output[3] for output in all_outputs])
        loss = losses.double().mean()
        auroc = None
        
        if not self.trainer.sanity_checking:
            self.log("valid_loss", loss)
            self.record_metrics(y, predicted, probs)
            
            if self.best_recorded_metrics[self.monitor_metric] == self.current_recorded_metrics[self.monitor_metric]:
                df_list = []
                df_list.append(["epoch", self.current_epoch])
                df_list.append(["step", self.global_step])
                df_list.append(["monitor_metric", self.monitor_metric])
                for key in self.recorded_metrics:
                    df_list.append([key, self.current_recorded_metrics[key]])
                df = pd.DataFrame(df_list)
                save_path = os.path.join(self.checkpoint_dir, "metrics.xlsx")
                if not os.path.exists(self.checkpoint_dir):
                    os.makedirs(self.checkpoint_dir)
                df.to_excel(save_path, index=False)
                # save probs
                save_path = os.path.join(self.checkpoint_dir, "probs.pt")
                torch.save(probs, save_path)
                # save label
                save_path = os.path.join(self.checkpoint_dir, "label.pt")
                torch.save(y, save_path)
                
        self.validation_step_outputs.clear() 


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        
        lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.99)

        return [optimizer], [lr_scheduler]

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, **kwargs):
        checkpoint = torch.load(checkpoint_path)
        state_dict = checkpoint['state_dict']
        model = cls(**kwargs)
        model.load_state_dict(state_dict)
        
        return model
