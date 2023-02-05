import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from model import Seq2seq

class LitLM(pl.LightningModule):
    def __init__(self, total_steps, smoothing_prob, decay_rate, 
            loss_weight,**kwargs):
        super().__init__()
        self.total_steps = total_steps
        self.model = Seq2seq(**kwargs).to(self.device)
        self.criterion = nn.CrossEntropyLoss(
            weight=loss_weight, ignore_index=kwargs['pad_token'])
        
        self.smoothing_prob = smoothing_prob
        self.decay_rate = decay_rate

        self.pad_token = kwargs['pad_token']

    def forward(self):
        raise NotImplementedError
    
    def full_accuracy(self, y_hat, y):
        correct_cnt = 0
        all_cnt = 0
        for _yh, _y in zip(y_hat, y):
            all_cnt += 1

            mask = (_y != self.pad_token)
            __yh = _yh.topk(k=1, dim=-1)[1].squeeze(-1)[mask]
            __y = _y[mask]

            correct_cnt += 1 if (__yh == __y).all() else 0

        return correct_cnt / all_cnt

    def accuracy(self, y_hat, y):
        # mask out pad_token
        mask = (y != self.pad_token)
        _y_hat = y_hat[mask]
        _y = y[mask]

        # pick top 1 indicies and calculate
        _y_hat = _y_hat.topk(k=1, dim=-1)[1].squeeze(-1)
        return (_y_hat == _y).type(torch.float32).mean()

    def training_step(self, batch, batch_idx):
        # exponential decay
        self.smoothing_prob *= self.decay_rate
        self.log('smo_prob', self.smoothing_prob, prog_bar=False)

        # X: (batch, seq), y: (batch, seq)
        X, y, _ = batch
        X = X.to(self.device)
        y = y.to(self.device)
        batch_size = y.shape[0]
        seq_len = y.shape[1]

        # y_hat: (batch, seq, d_model)
        if np.random.random() > self.smoothing_prob:
            y_hat = self.model(X.permute(1, 0), None).permute(1, 0, 2)
        else:
            y_hat = self.model(X.permute(1, 0), y.permute(1, 0)).permute(1, 0, 2)

        # calculate loss and acc
        loss = self.criterion(
            y_hat.reshape(batch_size*seq_len, -1),
            y.reshape(-1)
        )
        acc = self.accuracy(y_hat, y)
        facc = self.full_accuracy(y_hat, y)

        self.log('train_loss', loss.item(), prog_bar=False, on_step=True, on_epoch=False)
        self.log('train_acc', acc.item(), prog_bar=True, on_step=True, on_epoch=False)
        self.log('train_facc', facc, prog_bar=True, on_step=True, on_epoch=False)

        return {'loss': loss, 'train_acc': acc}

    def training_epoch_end(self, outputs):
        #pl.utilities.memory.garbage_collection_cuda()
        pass

    def validation_step(self, batch, batch_idx):
        # X: (batch, seq), y: (batch, seq)
        X, y, _ = batch
        X = X.to(self.device)
        y = y.to(self.device)
        batch_size = y.shape[0]
        seq_len = y.shape[1]

        # y_hat: (batch, seq, d_model)
        y_hat = self.model(X.permute(1, 0), None).permute(1, 0, 2)

        # calculate loss and acc
        loss = self.criterion(
            y_hat.reshape(batch_size*seq_len, -1),
            y.reshape(-1)
        )
        acc = self.accuracy(y_hat, y)
        facc = self.full_accuracy(y_hat, y)

        self.log(f'val_loss', loss.item(), prog_bar=False, on_step=False, on_epoch=True)
        self.log(f'val_acc', acc.item(), prog_bar=False, on_step=False, on_epoch=True)
        self.log('val_facc', facc, prog_bar=False, on_step=False, on_epoch=True)

        return {f'val_loss': loss.item(), f'val_acc': acc.item()}
        
    def validation_epoch_end(self, outputs):
        #pl.utilities.memory.garbage_collection_cuda()
        pass

    def predict_step(self, batch, batch_idx):
        # X: (batch, seq), y: (batch, seq)
        X, _ = batch
        X = X.to(self.device)

        y_hat = self.model(X.permute(1, 0)).permute(1, 0, 2)
        y_hat = y_hat.topk(k=1, dim=-1)[1].squeeze(-1)

        return y_hat

    def configure_optimizers(self):
        return {'optimizer': torch.optim.AdamW(self.parameters(), amsgrad=True)}
