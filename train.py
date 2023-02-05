import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from torchinfo import summary
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from sklearn.model_selection import train_test_split

from utils import Tokenizer, Visualize
from lightning import LitLM

MAX_SEQUENCE_LENGTH = 36

def load_data(file_path):
    try:
        # Try to load from cache
        print('Loading cache...', end='', flush=True)
        with open('model_file/cache_tokenizer.pickle', 'rb') as f:
            tokenizer = pickle.load(f)
        with open('model_file/cache_data.pickle', 'rb') as f:
            data = pickle.load(f)
            train_data, train_X, train_y, train_denom = data['train']
            test_data, test_X, test_y, test_denom = data['test']
        print('Success', flush=True)
    except:
        # Failed. Process and store to cache
        print('Failed. Preprocessing...', end='', flush=True)
        with open(file_path, 'r') as f:
            samples = f.read().strip().split('\n')
            train_data, test_data = train_test_split(samples, test_size=0.2, 
                random_state=0)
            tokenizer = Tokenizer(MAX_SEQUENCE_LENGTH)
            train_X, train_y, train_denom = tokenizer.preprocess(train_data)
            test_X, test_y, test_denom = tokenizer.preprocess(test_data)

        with open('model_file/cache_tokenizer.pickle', 'wb') as f:
            pickle.dump(tokenizer, f)
        with open('model_file/cache_data.pickle', 'wb') as f:
            pickle.dump({
                'train': (train_data, train_X, train_y, train_denom),
                'test': (test_data, test_X, test_y, test_denom),
            }, f);
        print('Done', flush=True)

    return tokenizer, train_data, train_X, train_y, train_denom,\
        test_data, test_X, test_y, test_denom

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=240)
    parser.add_argument('--steps_per_epoch', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=50)

    return parser.parse_args()

def main():
    args = parse_args()

    tokenizer, train_data, train_X, train_y, train_denom, \
        test_data, test_X, test_y, test_denom = load_data('data/train.txt')
    print('dictionary size', tokenizer.idcnt)
    print('number of unknown token', len(tokenizer.unk_tokens))
    print('unique number of unknown token', len(set(tokenizer.unk_tokens)))

    # dataset
    train_dataset = TensorDataset(train_X, train_y, train_denom)
    test_dataset = TensorDataset(test_X, test_y, test_denom)

    # sampler
    train_sampler = RandomSampler(train_dataset, num_samples=args.steps_per_epoch * args.batch_size)

    # dataloader
    batch_size = args.batch_size
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=16)
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=16)
    
    # Callbacks
    mc_facc = pl.callbacks.ModelCheckpoint(
            save_top_k=1,
            dirpath='./model_file',
            filename='facc-{epoch:02d}-{train_acc:.2f}-{val_acc:.2f}-{val_facc:.2f}',
            verbose=True,
            mode='max',
            monitor='val_facc',
            save_last=True
        )
    mc_acc = pl.callbacks.ModelCheckpoint(
            save_top_k=1,
            dirpath='./model_file',
            filename='acc-{epoch:02d}-{train_acc:.2f}-{val_acc:.2f}-{val_facc:.2f}',
            verbose=True,
            mode='max',
            monitor='val_acc',
            save_last=False # mc_facc has save_last already
        )
    lrMonitor = pl.callbacks.LearningRateMonitor(
        logging_interval='step', log_momentum=True
    )
    visualize_cb = Visualize(next(iter(test_dataloader)), tokenizer)
    tb_logger = pl_loggers.TensorBoardLogger('./tensorboard')
 
    # trainer & model
    trainer = pl.Trainer(
        devices='1', 
        accelerator='gpu' if torch.cuda.is_available() else 'cpu', 
        callbacks=[mc_facc, mc_acc, lrMonitor, visualize_cb], 
        max_epochs=args.epochs,
        logger=tb_logger
    )

    loss_weight = torch.ones(tokenizer.idcnt)
    for token in tokenizer.unary_operators:
        loss_weight[tokenizer.token2id(token)] = 2
    #for token in tokenizer.binary_operators:
    #    loss_weight[tokenizer.token2id(token)] = 2

    model = LitLM(
        total_steps=args.epochs * args.steps_per_epoch,
        smoothing_prob=0.8,
        decay_rate=np.exp(-0.0005),
        ntoken=tokenizer.idcnt,
        d_model=64,
        nhead=8,
        d_hid=128,
        enc_nlayers=6,
        dec_nlayers=6,
        start_token=tokenizer.token2id('<BOS>'),
        pad_token=tokenizer.token2id('<PAD>'),
        dropout=0.1,
        loss_weight=loss_weight
    )

    # summary model
    summary(model.model, input_data=next(iter(test_dataloader))[0].permute(1, 0))
 
    # Start training!
    trainer.fit(model, train_dataloader, test_dataloader)

if __name__ == "__main__":
    main()
