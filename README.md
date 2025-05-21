# 🥞 Banhxeo: A Simple, Efficient (Enough), and Educational NLP Library

> WARNING: Banhxeo cannot be used (and will never be used) for production.

**Welcome to Banhxeo!**

Just like its namesake – a delicious, crispy Vietnamese savory pancake that's delightful to make and eat – Banhxeo aims to be an NLP library that's:

*   **Simple & Understandable:** I believe in clarity. Many core NLP concepts and models are implemented from scratch (or close to it!) so you can see exactly what's going on under the hood. No black boxes here!
*   **Efficient (Just Enough):** While our primary goal is learning, we leverage PyTorch for neural models to ensure reasonable performance for your experiments and projects.
*   **Educational at Heart:** Banhxeo is designed for learners, educators, and anyone curious about how NLP models work. We encourage you to dive into the code, experiment, and build your understanding.

Think of Banhxeo as your kitchen for cooking up NLP models. We provide the basic ingredients (core components like tokenizers, vocabularies) and some foundational recipes (N-grams, MLPs, and more to come!). You're encouraged to get your hands dirty, modify the recipes, and even create your own!

## ✨ Philosophy

Our core philosophy revolves around three pillars:

1.  **Simplicity:** We strive for a clean, modular, and Pythonic codebase. Configurations are explicit, and APIs are designed to be intuitive.
2.  **From-Scratch Learning:** Many fundamental algorithms and model architectures are built with minimal reliance on high-level abstractions from other large libraries. This transparency is key for genuine understanding. If you've ever wondered "How does an N-gram model *actually* count things?" or "What are the layers inside a basic MLP for text classification?", Banhxeo is for you.
3.  **Practical Efficiency:** We use PyTorch as the backbone for our neural network models, allowing you to train and run models with decent speed, especially if you have a GPU.

## 🚀 Getting Started

1. **Installation:**

```bash
# Install with pip (we recommend using uv to manage environment)
pip install banhxeo 
# Or build by your self
git clone https://github.com/vietfood/banhxeo.git
cd banhxeo
uv sync
```

2. **A Quick Start**
```python
from banhxeo.dataset import IMDBDataset
from banhxeo.core.tokenizer import NLTKTokenizer, TokenizerConfig
from banhxeo.core.vocabulary import Vocabulary
from banhxeo.models.neural.mlp import MLP, MLPConfig
from banhxeo.training.trainer import Trainer, TrainerConfig

# --- Load Data ---
raw_imdb = IMDBDataset(root_dir="./", split="train")

# -- Create Tokenizer and Vocab ---
tokenizer = NLTKTokenizer()
vocab = Vocabulary.build(corpus=raw_imdb.text_data, tokenizer=tokenizer)

# -- Train MLP ---
```

3. **Use our Examples**

- [x] [N-gram example](./examples/n_gram.ipynb)
- [x] [MLP example](./examples/mlp_train.ipynb)
- [ ] RNN/LSTM example

## 🗺️ Roadmap

Banhxeo is an ongoing project. Here's a glimpse of where we're headed. We welcome contributions and ideas!

- [ ] `Vocabulary` system
    - [x] Basic system
    - [ ] Maybe there is something more
- [ ] `Tokenizer` system
    - [x] Basic system
    - [x] NLTK Tokenizer wrapper
    - [ ] HuggingFace Tokenizer wrapper
    - [ ] BPE from scratch (Greedy and Dynamic version, maybe ?)
- [x] Flexible `TextDataset` base class with Hugging Face integration.
- [x] `TorchDataset` wrapper for PyTorch `DataLoader`.
-  [x] Initial Models:
    - [x] N-gram
    - [x] MLP for text classification
    - [ ] RNN
    - [ ] LSTM
    - [ ] GPT-2
    - [ ] Word2Vec
- [x] `Trainer` class:
   - [x] User-defined training step.
   - [x] Callback system for logging (console, file, W&B).
   - [x] Checkpointing.
- [ ] Comprehensive documentation for all core modules.
- [ ] More examples and tutorials for common NLP tasks.
- [ ] Add test (Important)

## 🤝 Contributing

We'd love for you to contribute to Banhxeo! Whether it's fixing a bug, adding a new model, improving documentation, or suggesting an idea, your help is welcome.

## 📜 License

Banhxeo is licensed under the [MIT License](LICENSE).