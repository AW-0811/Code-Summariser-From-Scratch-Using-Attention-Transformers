# Code-Summariser-From-Scratch-Using-Attention-Transformers

This project implements a Transformer-based model from scratch to generate natural language docstrings for Python functions. It uses the [CodeSearchNet dataset](https://huggingface.co/datasets/code_search_net) (Python subset), a custom Byte-Pair Encoding tokenizer, and is trained end-to-end using PyTorch.

---

## 🧠 Model Overview

We implement a sequence-to-sequence transformer model with:
- Learned embeddings for code and documentation
- Positional encoding
- PyTorch’s built-in `nn.Transformer` module
- Cross-entropy loss (ignoring padding)
- Evaluation via BLEU, ROUGE-L, and BERTScore

---

## 🗂 File Structure

```
├── model.py               # Transformer model definition
├── preprocess.py          # Preprocessing and BPE tokenizer (no EOS token)
├── preprocess_eos.py      # Preprocessing with explicit EOS token
├── preprocess_output.log  # Logs of preprocessing execution
├── README.md              # README file
├── requirements.txt       # Requirements file with python dependencies
├── train.py               # Training without EOS tokens
├── train_eos.py           # Training with EOS-aware decoding
├── train_output.log       # Sample logs for non EOS training
├── trainEOS_output.log       # Sample logs for EOS-aware training
```

---

## 🔧 Setup & Installation

1. **Clone the repo**:
   ```bash
   git clone https://github.com/your-username/code-summariser.git
   cd code-summariser
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Preprocess dataset**:
   - Without EOS:
     ```bash
     python preprocess.py
     ```
   - With EOS appended to targets:
     ```bash
     python preprocess_eos.py
     ```

4. **Train the model**:
   - Base model:
     ```bash
     python train.py
     ```
   - EOS-based model (greedy decoding):
     ```bash
     python train_eos.py
     ```

---

## 📊 Evaluation

Each training script prints BLEU, ROUGE-L, and BERTScore after every epoch.

Example:
```
📊 Epoch 3 Summary | Avg Loss: 2.56 | BLEU: 0.431 | ROUGE-L: 0.589 | BERTScore-F1: 0.792
```

---

## ✍️ Sample Prediction Output

```
🔢 Input: def add(a, b): return a + b
🧠 Target: Returns the sum of a and b.
📤 Output: Returns the addition of two values.
```

---

## 📁 Checkpoints

Model checkpoints are saved after each epoch in:
- `checkpoints/` for the base model
- `checkpoints_EOSMODEL/` for the EOS decoding model

---

## 🧪 Notes

- EOS-based decoding can produce more natural sentence endings.
- The tokenizer is trained on only function bodies (`func_code_string`) from the dataset.
- Evaluation includes greedy decoding with optional truncation at EOS.

---

## 📜 License

MIT License.
