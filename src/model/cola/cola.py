import pytorch_lightning as pl
import torch
from efficientnet_pytorch import EfficientNet
from torch.nn import functional as F
import numpy as np
from .htsat.htsat import HTSATWrapper
import random
import pdb


class Encoder(torch.nn.Module):
    def __init__(self, drop_connect_rate=0.1):
        super(Encoder, self).__init__()

        self.cnn1 = torch.nn.Conv2d(1, 3, kernel_size=3)
        self.efficientnet = EfficientNet.from_name(
            "efficientnet-b0", include_top=False, drop_connect_rate=drop_connect_rate
        )

    def forward(self, x):
        x = x.unsqueeze(1)

        x = self.cnn1(x)
        x = self.efficientnet(x)

        y = x.squeeze(3).squeeze(2)

        return y


class EncoderHTSAT(torch.nn.Module):
    def __init__(self, drop_connect_rate=0.1):
        super(EncoderHTSAT, self).__init__()
        self.encoder = HTSATWrapper()
        from .htsat import config
        # self.out_emb = 768
        self.out_emb = config.htsat_dim * (2 ** (len(config.htsat_depth) - 1))
        
    def forward(self, x):
        x = x.unsqueeze(1)
        y = self.encoder(x)

        return y


class Cola(pl.LightningModule):
    def __init__(self, p=0.1, dim_fea=1280, dim_hidden=1280, dim_out=512, encoder="efficientnet", max_len=251, out_emb=2048):
        super().__init__()
        self.save_hyperparameters()

        self.p = p
        self.dim_fea, self.dim_hidden, self.dim_out = dim_fea, dim_hidden, dim_out
        self.do = torch.nn.Dropout(p=self.p)
        self.input_length = max_len

        if encoder == "efficientnet":
            self.encoder = Encoder(drop_connect_rate=p)
        elif encoder == "htsat":
            self.encoder = EncoderHTSAT()
            self.dim_fea = self.encoder.out_emb
            if dim_hidden > self.dim_fea : self.dim_hidden = self.dim_fea
        self.encoder_model = encoder

        self.middle_enabled = (self.dim_fea != self.dim_hidden)
        if self.middle_enabled:
            self.middle = torch.nn.Linear(self.dim_fea, self.dim_hidden)

        self.g = torch.nn.Linear(self.dim_hidden, self.dim_out)
        self.layer_norm = torch.nn.LayerNorm(normalized_shape=self.dim_out)
        self.linear = torch.nn.Linear(self.dim_out, self.dim_out, bias=False)

    def forward(self, x):
        # print(x)
        x1, x2 = x

        if self.middle_enabled:
            x1 = self.do(self.middle(self.encoder(x1)))
        else:
            x1 = self.do(self.encoder(x1))
        x1 = self.do(self.g(x1))
        x1 = self.do(torch.tanh(self.layer_norm(x1)))

        if self.middle_enabled:
            x2 = self.do(self.middle(self.encoder(x2)))
        else:
            x2 = self.do(self.encoder(x2))
        x2 = self.do(self.g(x2))
        x2 = self.do(torch.tanh(self.layer_norm(x2)))

        x1 = self.linear(x1)

        return x1, x2

    def extract_feature(self, x, dim=1280):
        if self.encoder_model == "vit":
            return self.extract_feature_vit(x, dim)
        x = self.encoder(x)
        if dim == self.dim_fea:
            return x
        if self.middle_enabled:
            x = self.middle(x)
        if dim == self.dim_hidden:
            return x
        x = self.g(x)
        if dim == self.dim_out:
            return x
        raise NotImplementedError

    
    def training_step(self, x, batch_idx):
        x1, x2 = self(x)

        y = torch.arange(x1.size(0), device=x1.device)

        y_hat = torch.mm(x1, x2.t())

        loss = F.cross_entropy(y_hat, y)

        _, predicted = torch.max(y_hat, 1)
        acc = (predicted == y).double().mean()

        self.log("train_loss", loss)
        self.log("train_acc", acc)

        return loss

    def validation_step(self, x, batch_idx, dataloader_idx=0):
        x1, x2 = self(x)

        y = torch.arange(x1.size(0), device=x1.device)

        y_hat = torch.mm(x1, x2.t())

        loss = F.cross_entropy(y_hat, y)

        _, predicted = torch.max(y_hat, 1)
        acc = (predicted == y).double().mean()

        self.log("valid_loss", loss)
        self.log("valid_acc", acc)

    def test_step(self, x, batch_idx):
        x1, x2 = self(x)

        y = torch.arange(x1.size(0), device=x1.device)

        y_hat = torch.mm(x1, x2.t())

        loss = F.cross_entropy(y_hat, y)

        _, predicted = torch.max(y_hat, 1)
        acc = (predicted == y).double().mean()

        self.log("test_loss", loss)
        self.log("test_acc", acc)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)


class ColaMD(pl.LightningModule):
    def __init__(self, p=0.1, dim_fea=1280, dim_hidden=1280, dim_out=512, encoder="efficientnet", batch_size=128, num_batch=[258.0, 288, 4, 51, 75, 146, 138], out_emb=2048, max_len=251, dataset_names=[]):
        super().__init__()
        self.save_hyperparameters()

        self.p = p
        self.dim_fea, self.dim_hidden, self.dim_out = dim_fea, dim_hidden, dim_out
        self.do = torch.nn.Dropout(p=self.p)
        self.input_length = max_len

        # self.encoder = Encoder(drop_connect_rate=p)
        if encoder == "efficientnet":
            self.encoder = Encoder(drop_connect_rate=p)
        elif encoder == "htsat":
            self.encoder = EncoderHTSAT()
            self.dim_fea = self.encoder.out_emb
            if dim_hidden > self.dim_fea : self.dim_hidden = self.dim_fea
        self.encoder_model = encoder
        if num_batch is not None:
            print(num_batch)
            self.num_batch = [b/np.sum(num_batch) for b in num_batch]
            self.num_batch = torch.tensor(self.num_batch)
            print(self.num_batch)

        self.middle_enabled = (self.dim_fea != self.dim_hidden)
        if self.middle_enabled:
            self.middle = torch.nn.Linear(self.dim_fea, self.dim_hidden)

        self.g = torch.nn.Linear(self.dim_hidden, self.dim_out)
        self.layer_norm = torch.nn.LayerNorm(normalized_shape=dim_out)
        self.linear = torch.nn.Linear(dim_out, dim_out, bias=False)
        self.batch_size = batch_size
        
        self.dataset_names = dataset_names
        
        # self.done_batches = torch.zeros(len(self.num_batch))
        self.val_losses, self.val_accs = [], []

    def forward(self, x):
        x1, x2 = x

        if self.middle_enabled:
            x1 = self.do(self.middle(self.encoder(x1)))
        else:
            x1 = self.do(self.encoder(x1))
        
        x1 = self.do(self.g(x1))
        x1 = self.do(torch.tanh(self.layer_norm(x1)))

        if self.middle_enabled:
            x2 = self.do(self.middle(self.encoder(x2)))
        else:
            x2 = self.do(self.encoder(x2))
            
        x2 = self.do(self.g(x2))
        x2 = self.do(torch.tanh(self.layer_norm(x2)))

        x1 = self.linear(x1)

        return x1, x2

    # def extract_feature(self, x, dim=1280):
    #     x = self.encoder(x)
    #     if dim == self.dim_fea:
    #         return x
    #     if self.middle_enabled:
    #         x = self.middle(x)
    #     if dim == self.dim_hidden:
    #         return x
    #     x = self.g(x)
    #     if dim == self.dim_out:
    #         return x
    #     raise NotImplementedError
    def extract_feature(self, x):
        x = self.encoder(x)
        # x = F.dropout(self.encoder(x), p=0.5, training=self.training)
        # x = self.do(self.encoder(x))
        # x = self.do(self.g(x))
        # x = self.do(torch.tanh(self.layer_norm(x)))
        # x = self.linear(x)
        return x
    
    def _calculate_loss(self, x, mode):
        # pdb.set_trace()
        x1, x2 = self(x)
        # pdb.set_trace()
        y = torch.arange(x1.size(0), device=x1.device)

        y_hat = torch.mm(x1, x2.t())

        loss = F.cross_entropy(y_hat, y)

        _, predicted = torch.max(y_hat, dim=1)
        acc = (predicted == y).double().mean()
        # pdb.set_trace()
        if "train" in mode:
            self.log("{}_loss".format(mode), loss, batch_size=self.batch_size, on_epoch=True)  # , on_epoch=True
            self.log("{}_acc".format(mode), acc, batch_size=self.batch_size, on_epoch=True)
        else:
            self.val_losses.append(loss.item())
            self.val_accs.append(acc.item())
        return loss
    
    def training_step(self, x, batch_idx):
        """
        covidbreath Length of Training, Validation, Testing: 258 29 29
        covidcough Length of Training, Validation, Testing: 288 32 32
        icbhi Length of Training, Validation, Testing: 4 1 1
        coughvid Length of Training, Validation, Testing: 51 6 6
        hf_lung Length of Training, Validation, Testing: 75 9 9
        covidUKexhalation Length of Training, Validation, Testing: 146 17 17
        covidUKcough Length of Training, Validation, Testing: 138 16 16
        """
        # pdb.set_trace()
        batch, batch_idx, dataloader_idx = x
        
        # lst = range(len(batch))
        # pdb.set_trace()
        # s = random.choices(lst, weights=self.num_batch, k=1)[0]
        
        # candidate_index = torch.where(self.num_batch - self.done_batches > 0)[0]
        # s = random.choices(candidate_index, k=1)[0]
        # self.done_batches[s] += 1
        # if torch.sum(self.done_batches) == torch.sum(self.num_batch):
        #     self.done_batches = torch.zeros(len(batch))
        # loss = self._calculate_loss(batch[s], batch_idx, self.dataset_names[s]+"_train")
        
        # pdb.set_trace()
        loss = self._calculate_loss(batch, self.dataset_names[dataloader_idx]+"_train")
        
        return loss

    def validation_step(self, x, batch_idx, dataloader_idx=0):
        batch, batch_idx, dataloader_idx = x
        
        self._calculate_loss(batch, "valid")  # "valid"
    
    def on_validation_epoch_end(self):
        valid_loss = torch.tensor(self.val_losses).mean()
        valid_acc = torch.tensor(self.val_accs).mean()
        self.log("valid_loss", valid_loss)  # 记录整个验证集的平均损失
        self.log("valid_acc", valid_acc) 
        self.val_losses.clear() 
        self.val_accs.clear() 
    

    def test_step(self, x, batch_idx):
        batch, batch_idx, dataloader_idx = x
        self._calculate_loss(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)


def weights_init(network):
    for m in network:
        classname = m.__class__.__name__
        # print(classname)
        if classname.find('Linear') != -1:
            m.weight.data.normal_(mean=0.0, std=0.01)
            m.bias.data.zero_()

