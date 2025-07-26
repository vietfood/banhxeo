🥞 Banhxeo: A Simple, Efficient (Enough), and Educational NLP Library
======================================================================

.. warning::

   Banhxeo cannot be used (and will never be used) for production.

Welcome to Banhxeo!
-------------------

Just like its namesake – a delicious, crispy Vietnamese savory pancake that's delightful to make and eat – Banhxeo aims to be an NLP library that's:

- **Simple & Understandable:** I believe in clarity. Many core NLP concepts and models are implemented from scratch (or close to it!) so you can see exactly what's going on under the hood. No black boxes here!
- **Efficient (Just Enough):** While our primary goal is learning, we leverage PyTorch for neural models to ensure reasonable performance for your experiments and projects.
- **Educational at Heart:** Banhxeo is designed for learners, educators, and anyone curious about how NLP models work. We encourage you to dive into the code, experiment, and build your understanding.

Think of Banhxeo as your kitchen for cooking up NLP models. We provide the basic ingredients (core components like tokenizers) and some foundational recipes (MLPs, GPT-2, and more to come!). You're encouraged to get your hands dirty, modify the recipes, and even create your own!

✨ Philosophy
------------

Our core philosophy revolves around three pillars:

1. **Simplicity:** We strive for a clean, modular, and Pythonic codebase. Configurations are explicit, and APIs are designed to be intuitive.
2. **From-Scratch Learning:** Many fundamental algorithms and model architectures are built with minimal reliance on high-level abstractions from other large libraries. This transparency is key for genuine understanding.
3. **Practical Efficiency:** We use JAX/Flax (JAX is really cool btw) as the backbone for our neural network models, allowing you to train and run models with decent speed, especially if you have a GPU (and TPU, maybe 😊).

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart

.. toctree::
   :glob:
   :caption: API References
   :maxdepth: 3

   api/modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`