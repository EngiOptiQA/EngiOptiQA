[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10222617.svg)](https://doi.org/10.5281/zenodo.10222617)
[![Test Examples](https://github.com/EngiOptiQA/EngiOptiQA/actions/workflows/test_examples.yml/badge.svg)](https://github.com/EngiOptiQA/EngiOptiQA/actions/workflows/test_examples.yml)
[![PyPI - Version](https://img.shields.io/pypi/v/engioptiqa)](https://pypi.org/project/engioptiqa/)

# EngiOptiQA: Engineering Optimization with Quantum Annealing

**Please note:** _EngiOptiQA_ is currently in a very early stage of development. As the project progresses, documentation, additional features, and enhancements will be added.

## Overview
_EngiOptiQA_ is a Python software library dedicated to **Engi**neering **Opti**mization with **Q**uantum **A**nnealing (QA).
This project provides a set of tools to formulate engineering optimization problems suitable for QA.

A minimal documentation can be found under [https://engioptiqa.github.io/EngiOptiQA/](https://engioptiqa.github.io/EngiOptiQA/). To learn more about the background of _EngiOptiQA_ and the implemented problem formulations, please refer to the corresponding publication [[1]](#pub1).

## Citation

If you use _EngiOptiQA_ in your research or work, please consider citing it using the software's [DOI](https://zenodo.org/doi/10.5281/zenodo.10222617) and the corresponding publication [Key2024](#pub1).

## Quick Example

Run this example for the design optimization of a rod under self-weight loading presented in [Key2024](#pub1), Section 3.2, solved using *simulated annealing (SA)*:

```bash
pip install -r requirements.txt
python3 examples/rod_1d/design_optimization_sa.py
```

The expected $H_1$ error for the best solution is approximately $1.59 \times 10^{-2}$:

```bash
H1 Error 0.015873015873015817 0.015873015873015817
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References
  1. <a name="pub1"></a>Key F, Freinberger L. A Formulation of Structural Design Optimization Problems for Quantum Annealing. Mathematics. 2024; 12(3):482. [https://doi.org/10.3390/math12030482](https://doi.org/10.3390/math12030482)
