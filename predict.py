import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL = "tanaysingh2512/swedish-grammar-scorer"
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def load_model(model_id=MODEL):
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id).to(device).eval()
    return tok, model


@torch.no_grad()
def score(texts, tok, model, batch_size=32, max_length=128):
    collate = lambda b: tok(b, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    out = []
    for enc in DataLoader(texts, batch_size=batch_size, collate_fn=collate):
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits.squeeze(-1).float().cpu().numpy()
        out.append(logits)
    return np.clip(np.concatenate(out), 0, 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score Swedish texts for grammatical correctness (0 = poor, 1 = good).")
    parser.add_argument("--input", required=True, help="JSONL with one {'id', 'text'} per line")
    parser.add_argument("--output", required=True, help="JSONL to write one {'id', 'score'} per line")
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args()

    rows = [json.loads(line) for line in open(args.input, encoding="utf-8")]
    tok, model = load_model(args.model)
    scores = score([r["text"] for r in rows], tok, model)

    with open(args.output, "w", encoding="utf-8") as f:
        for r, s in zip(rows, scores):
            f.write(json.dumps({"id": r["id"], "score": float(s)}) + "\n")
