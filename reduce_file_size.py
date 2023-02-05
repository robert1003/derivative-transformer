import sys
import pickle
import torch
from lightning import LitLM

assert len(sys.argv) > 3

tokenizer_path = sys.argv[1]
model_path = sys.argv[2]
save_model_path = sys.argv[3]

with open(tokenizer_path, 'rb') as f:
	tokenizer = pickle.load(f)

model = LitLM.load_from_checkpoint(
    model_path,
    total_steps=0,
    smoothing_prob=0,
    decay_rate=0,
    ntoken=tokenizer.idcnt,
    d_model=64,
    nhead=8,
    d_hid=128,
    enc_nlayers=6,
    dec_nlayers=6,
    start_token=tokenizer.token2id('<BOS>'),
    pad_token=tokenizer.token2id('<PAD>'),
    dropout=0.1,
    loss_weight=torch.ones(tokenizer.idcnt)
)

torch.save({
	'tokenizer': tokenizer,
	'model_state_dict': model.model.state_dict(),
}, save_model_path)