# preprocess_eos.py

from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
import os

# === Parameters ===
MAX_LINES = 50
MAX_SEQ_LEN = 256
TOKENIZER_DIR = "tokenizer_eos"
os.makedirs(TOKENIZER_DIR, exist_ok=True)

def load_and_filter():
    print("🔍 Loading dataset...")
    dataset = load_dataset("code_search_net", "python", trust_remote_code=True)

    def is_short_fn(example):
        code = example.get("func_code_string", "")
        doc = example.get("func_documentation_string", "")
        return code.strip() and doc.strip() and len(code.split("\n")) <= MAX_LINES

    train_filtered = dataset["train"].filter(is_short_fn)
    valid_filtered = dataset["validation"].filter(is_short_fn)

    print(f"✅ Filtered Train Set: {len(train_filtered)} examples")
    print(f"✅ Filtered Valid Set: {len(valid_filtered)} examples")

    return train_filtered, valid_filtered

def train_tokenizer(code_list):
    print("🧠 Training BPE tokenizer...")
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train_from_iterator(code_list, vocab_size=30000, min_frequency=2, special_tokens=[
        "<s>", "<pad>", "</s>", "<unk>", "<mask>"
    ])
    tokenizer.save_model(TOKENIZER_DIR)
    print(f"✅ Tokenizer saved to {TOKENIZER_DIR}/")
    return tokenizer

def encode_dataset(dataset, tokenizer):
    eos_token_id = tokenizer.token_to_id("</s>")

    def encode(example):
        code_ids = tokenizer.encode(example["func_code_string"]).ids[:MAX_SEQ_LEN]
        doc_ids = tokenizer.encode(example["func_documentation_string"]).ids[:MAX_SEQ_LEN - 1]
        doc_ids.append(eos_token_id)  # Append </s> to target
        return {"input_ids": code_ids, "target_ids": doc_ids}

    return dataset.map(encode, remove_columns=dataset.column_names)

if __name__ == "__main__":
    train_set, valid_set = load_and_filter()

    code_samples = [ex["func_code_string"] for ex in train_set]
    tokenizer = train_tokenizer(code_samples)

    tokenized_train = encode_dataset(train_set, tokenizer)
    tokenized_valid = encode_dataset(valid_set, tokenizer)

    print("💾 Saving tokenized datasets...")
    tokenized_train.save_to_disk("train_tokenized_eos")
    tokenized_valid.save_to_disk("valid_tokenized_eos")
    print("✅ Done! Tokenized datasets saved.")