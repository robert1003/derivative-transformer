from typing import Tuple

import numpy as np

MAX_SEQUENCE_LENGTH = 30
TRAIN_URL = "https://drive.google.com/file/d/1ND_nNV5Lh2_rf3xcHbvwkKLbY2DmOH09/view?usp=sharing"


def load_file(file_path: str) -> Tuple[Tuple[str], Tuple[str]]:
    """loads the test file and extracts all functions/derivatives"""
    data = open(file_path, "r").readlines()
    functions, derivatives = zip(*[line.strip().split("=") for line in data])
    return functions, derivatives


def score(true_derivative: str, predicted_derivative: str) -> int:
    """binary scoring function for model evaluation"""
    return int(true_derivative == predicted_derivative)


# --------- PLEASE FILL THIS IN --------- #
def setup(model_path):
    from lightning import LitLM
    import torch
    saved_info = torch.load(model_path)
    tokenizer = saved_info['tokenizer']

    model = LitLM(
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
        loss_weight=None
    )

    model.model.load_state_dict(saved_info['model_state_dict'])

    return tokenizer, model

def predict(functions: str):
    tokenizer, model = setup('submission_model.ckpt')
    model.eval()

    from torch.utils.data import TensorDataset, DataLoader
    test_X, test_denom = tokenizer.preprocess(functions, test=True)
    test_dataset = TensorDataset(test_X, test_denom)
    test_dataloader = DataLoader(
        test_dataset, batch_size=64, shuffle=False, num_workers=16)

    import pytorch_lightning as pl
    import torch
    trainer = pl.Trainer(
        devices='1', 
        accelerator='gpu' if torch.cuda.is_available() else 'cpu'
    )

    raw_predictions = trainer.predict(model, test_dataloader)

    from tqdm import tqdm
    predictions = []
    for ((X, denom), y_hat) in tqdm(zip(test_dataloader, raw_predictions), total=len(test_dataloader)):
        predictions += tokenizer.postprocess(
            y_hat.tolist(), denom.tolist(), to_infix=True)

    return predictions


# ----------------- END ----------------- #


def main(filepath: str = "test.txt"):
    """load, inference, and evaluate"""
    functions, true_derivatives = load_file(filepath)
    predicted_derivatives = predict(functions)
    scores = [score(td, pd) for td, pd in zip(true_derivatives, predicted_derivatives)]
    print(np.mean(scores))


if __name__ == "__main__":
    main()
