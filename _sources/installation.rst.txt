Installation
============

To install Banhxeo, you can use pip:

.. code-block:: bash

   pip install banhxeo

If you want to install from source:

.. code-block:: bash

   git clone https://github.com/vietfood/banhxeo.git
   cd banhxeo
   uv sync # (we recommend using uv to manage environment)

Dependencies
------------

Banhxeo requires Python 3.9+ and the following dependencies:
    - Pytorch, Numpy
    - Pydantic
    - Tqdm
    - Datasets (Hugging Face) and HF-Xet
    - Einops
    - Gdown (for Google Drive downloads)
    - Jaxtyping (for more strict tensor typing)
    - Polars (data manipulation)

You can install all core dependencies via:

.. code-block:: bash

   pip install banhxeo

Or all dependencies including optional ones:

.. code-block:: bash

    pip install "banhxeo[all]"

Current optionals are:
    - **Extras**: NLTK, Plotly, Rich (`pip install banhxeo[extras]`).
    - **Docs**: sphinx, sphinx-rtd-theme (for building documentation) (`pip install banhxeo[docs]`).
    - **Test**: pytest, pytest-cov (for testing) (`pip install banhxeo[test]`).