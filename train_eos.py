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
MAX_LEN = 128

def collate_fn(batch):
    src_batch = [torch.tensor(item["input_ids"])[:MAX_LEN] for item in batch]
    tgt_batch = [torch.tensor(item["target_ids"])[:MAX_LEN] for item in batch]
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=1)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=1)
    return src_padded, tgt_padded

def generate_square_subsequent_mask(sz):
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1).to(DEVICE)

def truncate_after_eos(token_ids, eos_id):
    if eos_id in token_ids:
        return token_ids[:token_ids.index(eos_id)+1]
    return token_ids

def batched_greedy_decode(model, src, tokenizer, max_len=MAX_LEN):
    model.eval()
    batch_size = src.size(0)
    generated = torch.full((batch_size, 1), tokenizer.token_to_id("<s>"), dtype=torch.long).to(DEVICE)
    finished = torch.zeros(batch_size, dtype=torch.bool).to(DEVICE)
    eos_token_id = tokenizer.token_to_id("</s>")

    for _ in range(max_len):
        tgt_mask = generate_square_subsequent_mask(generated.size(1))
        output = model(src, generated, tgt_mask=tgt_mask)
        next_token = output[:, -1, :].argmax(dim=-1).unsqueeze(1)
        generated = torch.cat((generated, next_token), dim=1)
        finished |= (next_token.squeeze(1) == eos_token_id)
        if finished.all():
            break

    return generated

def evaluate(model, dataloader, tokenizer):
    model.eval()
    bleu_scores, rouge_scores = [], []
    smooth_fn = SmoothingFunction().method1
    rouge = Rouge()
    printed_sample = False
    eos_id = tokenizer.token_to_id("</s>")
    all_preds, all_refs = [], []

    with torch.no_grad():
        loop = tqdm(dataloader, desc="🔍 Evaluating", leave=False)
        for src, tgt in loop:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            output_ids = batched_greedy_decode(model, src, tokenizer)

            preds = [
                tokenizer.decode(truncate_after_eos(row.tolist(), eos_id), skip_special_tokens=True).strip()
                for row in output_ids
            ]
            refs = [tokenizer.decode(row[1:].tolist(), skip_special_tokens=True).strip() for row in tgt]

            all_preds.extend(preds)
            all_refs.extend(refs)

            for p, r, input_ids in zip(preds, refs, src):
                bleu_scores.append(sentence_bleu([r.split()], p.split(), smoothing_function=smooth_fn))
                try:
                    rouge_score = rouge.get_scores(p, r)[0]['rouge-l']['f']
                except Exception:
                    rouge_score = 0.0
                rouge_scores.append(rouge_score)

                if not printed_sample:
                    input_clean = tokenizer.decode([t for t in input_ids.tolist() if t != 1], skip_special_tokens=True).strip()
                    print("\n🔎 Sample Prediction")
                    print("🔢 Input:", input_clean)
                    print("🧠 Target:", r)
                    print("📄 Output:", f"[{p}]")
                    printed_sample = True

    # Compute BERTScore
    P, R, F1 = bert_score(all_preds, all_refs, lang="en", device=DEVICE)
    bert_f1 = float(F1.mean())

    return sum(bleu_scores)/len(bleu_scores), sum(rouge_scores)/len(rouge_scores), bert_f1

if __name__ == "__main__":
    train_data = load_from_disk("train_tokenized_eos")
    val_data = load_from_disk("valid_tokenized_eos")

    tokenizer = ByteLevelBPETokenizer("tokenizer_eos/vocab.json", "tokenizer_eos/merges.txt")
    vocab_size = tokenizer.get_vocab_size()

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    model = TransformerCommentGen(vocab_size).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    os.makedirs("checkpoints_EOSMODEL", exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        loop = tqdm(train_loader, desc=f"🚂 Epoch {epoch+1}/{EPOCHS}")
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

        checkpoint_path = f"checkpoints_EOSMODEL/transformer_epoch{epoch+1}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"💾 Model saved to: {checkpoint_path}\n")