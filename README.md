# 🥞 Banhxeo: A Simple, Efficient (Enough), and Educational NLP Library

> [!WARNING] 
> Banhxeo cannot be used (and will never be used) for production.

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
3.  **Practical Efficiency:** We use Jax/Flax (Jax is really cool btw) as the backbone for our neural network models, allowing you to train and run models with decent speed, especially if you have a GPU (and TPU, maybe 😊).

## 🚀 Getting Started

1. **Installation:**

```bash
# Install with pip 
pip install banhxeo 
# Or build by your self (we recommend using uv to manage environment)
git clone https://github.com/vietfood/banhxeo.git
cd banhxeo
uv sync
```
2. **Examples**

- [ ] [N-gram example]()
- [ ] [MLP example]()
- [ ] RNN/LSTM example
- [ ] GPT-2 example

3. **API References**

>[!NOTE]
>In constructions

## 🗺️ Roadmap

Banhxeo is an ongoing project. Here's a glimpse of where we're headed. We welcome contributions and ideas!

- [ ] `Tokenizer` system
    - [x] Basic system (Normalizer -> PreTokenizer -> Model -> PostProcessor)
    - [x] NLTK Tokenizer wrapper
    - [ ] HuggingFace Tokenizer wrapper (we have HuggingFace Toeknizer at home)
    - [x] BPE from scratch (Greedy version)
-  [x] Initial Models:
    - [ ] MLP for text classification
    - [ ] RNN
    - [ ] LSTM
    - [ ] GPT-2
    - [ ] Word2Vec
- [ ] `Trainer` class:
   - [ ] User-defined training step.
   - [ ] Callback system for logging (console, file, W&B).
   - [ ] Checkpointing.
- [ ] More examples and tutorials for common NLP tasks.
- [ ] Add test (Important)
    - [ ] Tokenizer Test (50%)

## 🤝 Contributing

We'd love for you to contribute to Banhxeo! Whether it's fixing a bug, adding a new model, improving documentation, or suggesting an idea, your help is welcome.

## 📜 License

Banhxeo is licensed under the [MIT License](LICENSE).