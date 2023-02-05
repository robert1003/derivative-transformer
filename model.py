import torch
import torch.nn as nn
import math
from torch import Tensor

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class Seq2seq(nn.Module):
    def __init__(self, 
            ntoken: int, 
            d_model: int, 
            nhead: int, 
            d_hid: int,
            enc_nlayers: int,
            dec_nlayers: int,
            pad_token: int, 
            start_token: int, 
            dropout: float = 0.1
        ):
        super().__init__()

        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.embedding = nn.Embedding(ntoken, d_model, padding_idx=pad_token)
        self.transformer = nn.Transformer(
            d_model=d_model, 
            nhead=nhead, 
            num_encoder_layers=enc_nlayers,
            num_decoder_layers=dec_nlayers,
            dim_feedforward=d_hid,
            dropout=dropout)
        self.linear = nn.Linear(d_model, ntoken)

        self.pad_token = pad_token
        self.start_token = start_token
        self.d_model = d_model
        self.nhead = nhead

    def encode(self, x):
        return self.pos_encoder(self.embedding(x) * math.sqrt(self.d_model))

    def forward(self, src, target=None):
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]

        Returns:
            output Tensor of shape [seq_len, batch_size, logits(sz of embedding)]
        """
        batch_size = src.shape[1]
        seq_len = src.shape[0]

        # mask out attention for padding
        _input_mask = (src == self.pad_token)
        _input_mask = _input_mask.permute(1, 0).unsqueeze(1)\
            .repeat(self.nhead, seq_len, 1)
        input_mask = torch.zeros_like(_input_mask, dtype=torch.float)
        input_mask[_input_mask] = -torch.inf

        # encode
        encoder_hidden = self.encode(src)
        encoder_hidden = self.transformer.encoder(encoder_hidden, mask=input_mask)

        # decode
        decoder_input = torch.LongTensor([self.start_token \
            for _ in range(batch_size)]).view(1, -1).to(src.device) # (seq_len=1, batch_size)

        def _gen_target_mask(cur_target):
            target_len = cur_target.shape[0]
            target_mask = torch.ones(target_len, target_len, 
                dtype=torch.bool, device=cur_target.device)\
                .triu(diagonal=1).repeat(self.nhead * batch_size, 1, 1)

            target_mask |= (cur_target == self.pad_token)\
                .permute(1, 0).unsqueeze(1).repeat(self.nhead, target_len, 1)

            _target_mask = torch.zeros_like(target_mask, dtype=torch.float)
            _target_mask[target_mask] = -torch.inf

            return _target_mask

        if target is not None:
            target_mask = _gen_target_mask(target)

            decoder_output = self.transformer.decoder(
                self.encode(target), memory=encoder_hidden, tgt_mask=target_mask)

            decoder_output = self.linear(decoder_output)

            return decoder_output
        else:
            decoder_input = torch.ones(1, batch_size, 
                dtype=torch.int32, device=src.device).fill_(self.start_token)

            result = []
            for i in range(seq_len):
                memory = _gen_target_mask(decoder_input)
                decoder_output = self.transformer.decoder(
                    self.encode(decoder_input), memory=encoder_hidden)
                decoder_output = self.linear(decoder_output)
                topv, topi = decoder_output.topk(k=1, dim=-1)
                result.append(decoder_output[-1, :, :])

                decoder_input = torch.cat([decoder_input, topi[[-1], :, 0]], dim=0)

            return torch.stack(result)
