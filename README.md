SpecFold is a small JAX implementation of the folded transformer layer from arXiv:2610.04875.
pip install -e .
pytest
This is the folded layer (gate, exact attention correction, residual not folded, FFN reuse), not the Triton kernels and not a diffusion LLM.
