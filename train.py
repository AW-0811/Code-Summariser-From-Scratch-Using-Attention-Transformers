import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from datasets import load_from_disk
from tokenizers import ByteLevelBPETokenizer
from model import TransformerCommentGen
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from bert_score import score as bert_score
from tqdm import tqdm
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
EPOCHS = 10
MAX_LEN = 256  # match preprocessing truncation

def collate_fn(batch):
    src_batch = [torch.tensor(item["input_ids"])[:MAX_LEN] for item in batch]
    tgt_batch = [torch.tensor(item["target_ids"])[:MAX_LEN] for item in batch]
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=1)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=1)
    return src_padded, tgt_padded

def generate_square_subsequent_mask(sz):
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1).to(DEVICE)

def evaluate(model, dataloader, tokenizer):
    model.eval()
    bleu_scores, rouge_scores = [], []
    preds, refs = [], []
    smooth_fn = SmoothingFunction().method1
    rouge = Rouge()
    printed_sample = False

    with torch.no_grad():
        for src, tgt in dataloader:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            tgt_input = tgt[:, :-1]
            tgt_mask = generate_square_subsequent_mask(tgt_input.size(1))
            output = model(src, tgt_input, tgt_mask=tgt_mask).argmax(dim=-1)

            pred_strs = [tokenizer.decode(row.tolist(), skip_special_tokens=True).strip() for row in output]
            ref_strs = [tokenizer.decode(row.tolist(), skip_special_tokens=True).strip() for row in tgt[:, 1:]]

            preds.extend(pred_strs)
            refs.extend(ref_strs)

            for p, r in zip(pred_strs, ref_strs):
                bleu_scores.append(sentence_bleu([r.split()], p.split(), smoothing_function=smooth_fn))
                try:
                    rouge_score = rouge.get_scores(p, r)[0]['rouge-l']['f']
                except Exception:
                    rouge_score = 0.0
                rouge_scores.append(rouge_score)

            if not printed_sample:
                input_clean = tokenizer.decode([t for t in src[0].tolist() if t != 1], skip_special_tokens=True).strip()
                print("\n🔎 Sample Prediction")
                print("🔢 Input:", input_clean)
                print("🧠 Target:", ref_strs[0])
                print("📤 Output:", f"[{pred_strs[0]}]")
                printed_sample = True

    # Compute BERTScore
    P, R, F1 = bert_score(preds, refs, lang="en", device=DEVICE)
    avg_bert_f1 = float(F1.mean())

    return sum(bleu_scores)/len(bleu_scores), sum(rouge_scores)/len(rouge_scores), avg_bert_f1

if __name__ == "__main__":
    train_data = load_from_disk("train_tokenized")
    val_data = load_from_disk("valid_tokenized")

    tokenizer = ByteLevelBPETokenizer("tokenizer/vocab.json", "tokenizer/merges.txt")
    vocab_size = tokenizer.get_vocab_size()

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    model = TransformerCommentGen(vocab_size).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for src, tgt in loop:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            tgt_input = tgt[:, :-1]
            tgt_expected = tgt[:, 1:]
            tgt_mask = generate_square_subsequent_mask(tgt_input.size(1))

            output = model(src, tgt_input, tgt_mask=tgt_mask)
            loss = F.cross_entropy(output.view(-1, vocab_size), tgt_expected.reshape(-1), ignore_index=1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        bleu, rouge, bert = evaluate(model, val_loader, tokenizer)
        print(f"\n📊 Epoch {epoch+1} Summary | Avg Loss: {total_loss:.2f} | BLEU: {bleu:.3f} | ROUGE-L: {rouge:.3f} | BERTScore-F1: {bert:.3f}")

        checkpoint_path = f"checkpoints/transformer_epoch{epoch+1}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"💾 Model saved to: {checkpoint_path}\n")