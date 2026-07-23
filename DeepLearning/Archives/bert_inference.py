#!/usr/bin/env python3
"""BERT inference examples:
- Masked token prediction (MLM)
- Sequence classification
- Token classification
- Extractive QA
"""

import math
import os
from typing import List

import torch
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = "outputs"

MAX_LEN = 24
EMBED = 64
HEADS = 4
FFN_DIM = 128
LAYERS = 2
DROPOUT = 0.1

SPECIAL = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
WORDS = [
    "king",
    "queen",
    "man",
    "woman",
    "boy",
    "girl",
    "paris",
    "london",
    "france",
    "england",
    "city",
    "country",
    "is",
    "a",
    "in",
    "and",
    "are",
    "royalty",
    "children",
    "capital",
]
VOCAB = SPECIAL + WORDS
STOI = {w: i for i, w in enumerate(VOCAB)}
ITOS = {i: w for w, i in STOI.items()}

PAD_ID = STOI["[PAD]"]
UNK_ID = STOI["[UNK]"]
CLS_ID = STOI["[CLS]"]
SEP_ID = STOI["[SEP]"]
MASK_ID = STOI["[MASK]"]


def tokenize(text: str) -> List[int]:
    return [STOI.get(w, UNK_ID) for w in text.split()]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, heads, dropout=0.0):
        super().__init__()
        assert d_model % heads == 0
        self.d_model = d_model
        self.heads = heads
        self.d_head = d_model // heads
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        b = x.size(0)
        q = self.wq(x).view(b, -1, self.heads, self.d_head).transpose(1, 2)
        k = self.wk(x).view(b, -1, self.heads, self.d_head).transpose(1, 2)
        v = self.wv(x).view(b, -1, self.heads, self.d_head).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(b, -1, self.d_model)
        return self.wo(context)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout=0.0):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn = self.attn(x, mask)
        x = self.norm1(x + self.drop(attn))
        ffn = self.ffn(x)
        x = self.norm2(x + self.drop(ffn))
        return x


class BertEncoder(nn.Module):
    def __init__(self, vocab, d_model, heads, d_ff, layers, dropout=0.0, max_len=512):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = PositionalEncoding(d_model, max_len)
        self.seg = nn.Embedding(2, d_model)
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, heads, d_ff, dropout) for _ in range(layers)]
        )

    def forward(self, input_ids, token_type_ids, attn_mask):
        x = self.tok(input_ids) + self.seg(token_type_ids)
        x = self.drop(self.pos(x))
        mask = attn_mask.unsqueeze(1).unsqueeze(2)
        for layer in self.layers:
            x = layer(x, mask)
        return x


class BertForPretraining(nn.Module):
    def __init__(self, vocab, d_model, heads, d_ff, layers, dropout=0.0, max_len=512):
        super().__init__()
        self.encoder = BertEncoder(vocab, d_model, heads, d_ff, layers, dropout, max_len)
        self.mlm_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.LayerNorm(d_model), nn.Linear(d_model, vocab)
        )
        self.nsp_head = nn.Linear(d_model, 2)

    def forward(self, input_ids, token_type_ids, attn_mask):
        enc = self.encoder(input_ids, token_type_ids, attn_mask)
        cls = enc[:, 0, :]
        mlm_logits = self.mlm_head(enc)
        nsp_logits = self.nsp_head(cls)
        return mlm_logits, nsp_logits


class BertForSequenceClassification(nn.Module):
    def __init__(self, encoder, num_labels=2):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(EMBED, num_labels)

    def forward(self, input_ids, token_type_ids, attn_mask):
        enc = self.encoder(input_ids, token_type_ids, attn_mask)
        cls = enc[:, 0, :]
        return self.classifier(cls)


class BertForTokenClassification(nn.Module):
    def __init__(self, encoder, num_labels=2):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(EMBED, num_labels)

    def forward(self, input_ids, token_type_ids, attn_mask):
        enc = self.encoder(input_ids, token_type_ids, attn_mask)
        return self.classifier(enc)


class BertForQA(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.qa_head = nn.Linear(EMBED, 2)

    def forward(self, input_ids, token_type_ids, attn_mask):
        enc = self.encoder(input_ids, token_type_ids, attn_mask)
        logits = self.qa_head(enc)
        start_logits, end_logits = logits.split(1, dim=-1)
        return start_logits.squeeze(-1), end_logits.squeeze(-1)



def main():
    # MLM inference
    pre_ckpt = os.path.join(OUT_DIR, "bert_pretrain.pt")
    if os.path.exists(pre_ckpt):
        pre = BertForPretraining(len(VOCAB), EMBED, HEADS, FFN_DIM, LAYERS, DROPOUT, MAX_LEN).to(DEVICE)
        pre.load_state_dict(torch.load(pre_ckpt, map_location=DEVICE))
        pre.eval()

        sentence = "paris is [MASK] france"
        ids = [CLS_ID] + [STOI.get(w, UNK_ID) for w in sentence.split()] + [SEP_ID]
        ids = ids[:MAX_LEN]
        seg = [0] * len(ids)
        attn = [1] * len(ids)
        while len(ids) < MAX_LEN:
            ids.append(PAD_ID)
            seg.append(0)
            attn.append(0)

        input_ids = torch.tensor([ids]).to(DEVICE)
        token_type_ids = torch.tensor([seg]).to(DEVICE)
        attn_mask = torch.tensor([attn]).to(DEVICE)

        with torch.no_grad():
            mlm_logits, _ = pre(input_ids, token_type_ids, attn_mask)
            mask_idx = ids.index(MASK_ID)
            pred_id = torch.argmax(mlm_logits[0, mask_idx]).item()
        print("Masked sentence:", sentence)
        print("Predicted token:", ITOS[pred_id])
    else:
        print("Pretrained BERT not found. Run bert_pretrain.py first.")

    # Sequence classification
    seq_ckpt = os.path.join(OUT_DIR, "bert_finetune_sequence.pt")
    if os.path.exists(seq_ckpt):
        encoder = BertEncoder(len(VOCAB), EMBED, HEADS, FFN_DIM, LAYERS, DROPOUT, MAX_LEN)
        clf = BertForSequenceClassification(encoder, num_labels=2).to(DEVICE)
        clf.load_state_dict(torch.load(seq_ckpt, map_location=DEVICE))
        clf.eval()

        test_sent = "queen is a woman"
        ids = [CLS_ID] + tokenize(test_sent) + [SEP_ID]
        ids = ids[:MAX_LEN]
        seg = [0] * len(ids)
        attn = [1] * len(ids)
        while len(ids) < MAX_LEN:
            ids.append(PAD_ID)
            seg.append(0)
            attn.append(0)

        input_ids = torch.tensor([ids]).to(DEVICE)
        token_type_ids = torch.tensor([seg]).to(DEVICE)
        attn_mask = torch.tensor([attn]).to(DEVICE)

        with torch.no_grad():
            logits = clf(input_ids, token_type_ids, attn_mask)
            pred = torch.argmax(logits, dim=1).item()
        print("Sequence classification sentence:", test_sent)
        print("Prediction (1=royalty,0=other):", pred)
    else:
        print("Finetuned sequence classifier not found. Run bert_finetune_sequence.py first.")

    # Token classification
    tok_ckpt = os.path.join(OUT_DIR, "bert_finetune_token.pt")
    if os.path.exists(tok_ckpt):
        encoder = BertEncoder(len(VOCAB), EMBED, HEADS, FFN_DIM, LAYERS, DROPOUT, MAX_LEN)
        tok = BertForTokenClassification(encoder, num_labels=2).to(DEVICE)
        tok.load_state_dict(torch.load(tok_ckpt, map_location=DEVICE))
        tok.eval()

        sent = "paris is in france"
        ids = [CLS_ID] + tokenize(sent) + [SEP_ID]
        ids = ids[:MAX_LEN]
        seg = [0] * len(ids)
        attn = [1] * len(ids)
        while len(ids) < MAX_LEN:
            ids.append(PAD_ID)
            seg.append(0)
            attn.append(0)

        input_ids = torch.tensor([ids]).to(DEVICE)
        token_type_ids = torch.tensor([seg]).to(DEVICE)
        attn_mask = torch.tensor([attn]).to(DEVICE)

        with torch.no_grad():
            logits = tok(input_ids, token_type_ids, attn_mask)
            preds = torch.argmax(logits, dim=-1).cpu().tolist()[0]
        print("Token classification sentence:", sent)
        print("Token preds:", preds)
    else:
        print("Finetuned token classifier not found. Run bert_finetune_token.py first.")

    # QA
    qa_ckpt = os.path.join(OUT_DIR, "bert_finetune_qa.pt")
    if os.path.exists(qa_ckpt):
        encoder = BertEncoder(len(VOCAB), EMBED, HEADS, FFN_DIM, LAYERS, DROPOUT, MAX_LEN)
        qa = BertForQA(encoder).to(DEVICE)
        qa.load_state_dict(torch.load(qa_ckpt, map_location=DEVICE))
        qa.eval()

        question = "what is the capital"
        context = "paris is a city in france"
        q_ids = tokenize(question)
        c_ids = tokenize(context)
        ids = [CLS_ID] + q_ids + [SEP_ID] + c_ids + [SEP_ID]
        ids = ids[:MAX_LEN]
        seg = [0] * (len(q_ids) + 2) + [1] * (len(c_ids) + 1)
        seg = seg[:MAX_LEN]
        attn = [1] * len(ids)
        while len(ids) < MAX_LEN:
            ids.append(PAD_ID)
            seg.append(0)
            attn.append(0)

        input_ids = torch.tensor([ids]).to(DEVICE)
        token_type_ids = torch.tensor([seg]).to(DEVICE)
        attn_mask = torch.tensor([attn]).to(DEVICE)

        with torch.no_grad():
            start_logits, end_logits = qa(input_ids, token_type_ids, attn_mask)
            start = torch.argmax(start_logits, dim=1).item()
            end = torch.argmax(end_logits, dim=1).item()
        answer_tokens = [ITOS.get(t, "?") for t in ids[start : end + 1]]
        print("QA question:", question)
        print("QA context:", context)
        print("QA answer tokens:", answer_tokens)
    else:
        print("Finetuned QA model not found. Run bert_finetune_qa.py first.")


if __name__ == "__main__":
    main()
