# Quantitative Methods in Finance (QMF)

This repository contains the pedagogical and replication code for the lecture notes **Quantitative Methods in Finance**, developed by **Eric Vansteenberghe** over more than ten years of teaching at **Université Paris 1 Panthéon-Sorbonne**, within the Master *Finance, Technology & Data*.

The lecture notes are currently available on **SSRN** and will be released on **arXiv** shortly. This GitHub repository serves as the executable companion, providing transparent and reproducible implementations of the methods introduced in the notes.

## Repository structure

```
QMF/
├── code/
│   └── variables_functions_an_introduction/
│       └── vansteenberghe_types_loops_functions.py
```

- `code/` contains one folder per chapter or major section of the lecture notes.
- Each folder includes self-contained Python scripts illustrating the concepts developed in the corresponding chapter.

## Usage

Create the Python environment:

```bash
conda env create -f environment.yml
```

Activate the environment:

```bash
conda activate qmf
```

Run any script independently, for example:

```bash
python code/variables_functions_an_introduction/vansteenberghe_types_loops_functions.py
```

Some sections (e.g. performance comparisons) are intended for interactive use and are better explored in IPython or Jupyter.

## License

- Code: MIT License
- Lecture notes: academic preprint (SSRN / arXiv forthcoming)

## Citation

If you use this material for teaching or research, please cite the lecture notes:

```bibtex
@article{vansteenberghe2025quantitative,
  title   = {Quantitative Methods in Finance},
  author  = {Vansteenberghe, Eric},
  journal = {Available at SSRN 5178205},
  year    = {2025}
}
```
