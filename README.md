# Quantitative Methods in Finance (QMF) — companion code

Executable companion to the lecture notes **_Quantitative Methods in Finance_** and to the
companion primer **_Mathematics for Finance: A Self-Contained Primer_**, by
**Eric Vansteenberghe** (Banque de France; Université Paris 1 Panthéon-Sorbonne).

The notes support two graduate lectures taught at **Université Paris 1 Panthéon-Sorbonne**
for more than ten years, notably in the Master *Finance, Technology & Data*:

- **Quantitative Methods in Finance** — probability, simulation, numerical methods and their
  implementation in Python for students with heterogeneous programming backgrounds;
- **Financial Econometrics** — stationarity and unit roots, ARIMA/SARIMA, ARCH–GARCH, VAR and
  structural VAR, cointegration, panel data, identification and causal inference in finance
  and macro-finance.

This repository provides transparent, reproducible implementations of the methods introduced in
the notes: one folder per chapter or major section, self-contained scripts, and the data files
needed to run them.

## Reference material

| Document | Identifier | Link |
| --- | --- | --- |
| *Quantitative Methods in Finance* (lecture notes, 600+ pages, 88 chapters) | arXiv:2601.12896 · DOI [10.48550/arXiv.2601.12896](https://doi.org/10.48550/arXiv.2601.12896) | <https://arxiv.org/abs/2601.12896> |
| *Quantitative Methods in Finance* (lecture notes, SSRN copy) | SSRN 5178205 | <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5178205> |
| *Mathematics for Finance: A Self-Contained Primer* (mathematical companion) | SSRN 7442161 | <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7442161> |
| Author's publication list | — | <https://scholar.google.com/citations?user=KInXUlUAAAAJ> |

**How the three documents fit together.** The primer, *Mathematics for Finance: A Self-Contained
Primer*, assumes no prior knowledge and develops the algebra, calculus, trigonometry, complex
numbers, combinatorics, probability, linear algebra and statistical tools (law of large numbers,
central limit theorem, least squares, maximum likelihood, delta method, risk measures, extreme
value analysis) required to follow the lecture. The lecture notes, *Quantitative Methods in
Finance*, build the econometric and empirical material on top of it. This repository is where the
corresponding code lives.


## Getting started

```bash
git clone https://github.com/skimeur/QMF.git
cd QMF
conda env create -f environment.yml
conda activate qmf
```

Run any script independently, for example:

```bash
python code/variables_functions_an_introduction/vansteenberghe_types_loops_functions.py
```

Scripts expect to be run **from the repository root**, so that relative paths to `data/` resolve.
Some sections (performance comparisons, interactive plots) are meant for interactive use and are
better explored in IPython, Spyder or Jupyter.

**Python stack:** `numpy`, `pandas`, `scipy`, `matplotlib`, `statsmodels`, `arch`, `scikit-learn`,
`pandas-datareader`, `pmdarima`.
**R scripts** (`code/SARIMA/`, `code/seasonality_treatment/`) use base R with `forecast` and `tseries`; see
the notes for the R setup chapter.

## Topics covered by the lecture notes

Probability, random variables and distributions · Monte Carlo simulation · maximum likelihood
estimation · law of large numbers and central limit theorem · OLS and multivariate regression ·
instrumental variables, Hausman test and endogeneity · difference-in-differences · quantile
regression · kernel density estimation · bootstrap · GMM · binary choice models · panel data ·
stationarity and unit roots · ARIMA, SARIMA and inflation persistence · ARCH, GARCH and volatility
modelling · cointegration and error-correction models · VAR, structural VAR and monetary policy
shocks · Granger causality · event studies · CAPM and arbitrage pricing theory · portfolio
performance measures and pairs trading · value-at-risk and expected shortfall · extreme value
theory and heavy tails · copulas · Bayesian statistics and insurance applications · networks ·
forecast combination and uncertainty.

Replications of published work (Arrow 1963; Meyers 1996; Kousky 2012; Bleakley 2010; Adrian,
Boyarchenko and Giannone 2019, among others) are part of the material, in line with the
replication-based approach of the lecture.

## Questions this repository answers

**Which lecture notes does this code accompany?** *Quantitative Methods in Finance*
([arXiv:2601.12896](https://arxiv.org/abs/2601.12896),
[SSRN 5178205](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5178205)) by Eric Vansteenberghe.

**What mathematical background is required?** None beyond what is developed in *Mathematics for
Finance: A Self-Contained Primer*
([SSRN 7442161](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7442161)), which is designed to
be read alongside the notes.

**Which language should I use?** Python is the main language; R is used where it is the natural
choice (SARIMA, seasonal adjustment, some panel-data and extreme-value exercises).

**Can I use this material to teach my own course?** Yes — the code is MIT-licensed. Please cite the
lecture notes (see below).

## How to cite

Please cite the lecture notes rather than the repository alone. Preferred citation:

> Vansteenberghe, E. (2026). *Quantitative Methods in Finance*. arXiv:2601.12896.
> https://doi.org/10.48550/arXiv.2601.12896

```bibtex
@article{vansteenberghe2026quantitative,
  title  = {Quantitative Methods in Finance},
  author = {Vansteenberghe, Eric},
  year   = {2026},
  journal = {arXiv preprint arXiv:2601.12896},
  doi    = {10.48550/arXiv.2601.12896},
  url    = {https://arxiv.org/abs/2601.12896},
  note   = {Also available on SSRN, abstract 5178205}
}

@article{vansteenberghe2026mathematics,
  title  = {Mathematics for Finance: A Self-Contained Primer},
  author = {Vansteenberghe, Eric},
  year   = {2026},
  journal = {SSRN Electronic Journal},
  note   = {SSRN working paper 7442161},
  url    = {https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7442161}
}
```

A machine-readable `CITATION.cff` is provided, so GitHub's *Cite this repository* button returns
the same reference.

## Related research by the author

The lecture draws on the author's empirical research; these papers are the natural extensions of
several chapters.

- Vansteenberghe, E. (2025). Insurance supervision under climate change: a pioneer detection
  method. *The Geneva Papers on Risk and Insurance — Issues and Practice*. — extends the insurance,
  extreme value theory and climate-risk chapters; replication code:
  <https://github.com/skimeur/pioneer-detection-method>.
- Vansteenberghe, E. (2025). Monetary policy, uncertainty, and credit supply. *Banque de France
  Working Paper No. 1025*. — extends the VAR, structural VAR and monetary policy chapters.
- Vansteenberghe, E. (2024). Uncertain and asymmetric forecasts. *SSRN working paper*. — extends
  the forecast uncertainty, quantile regression and opinion-pooling chapters.
- Beaumont, P., Tang, H., & Vansteenberghe, E. (2025). Collateral effects: the role of fintech in
  small business lending. *The Review of Financial Studies*. — extends the credit-risk and
  identification chapters.
- Nicolas, T., Ungaro, S., & Vansteenberghe, E. (2025). Public-guaranteed loans, bank risk-taking,
  and the regulatory capital windfall. *Journal of Financial Services Research*. — extends the
  panel-data and policy-evaluation chapters.

## License

- **Code and data preparation scripts:** MIT License (see [`LICENSE`](LICENSE)).
- **Lecture notes and primer:** academic preprints; see the arXiv and SSRN pages for their terms.

## Author

**Eric Vansteenberghe** — Research economist, Banque de France; lecturer, Université Paris 1
Panthéon-Sorbonne.
[Google Scholar](https://scholar.google.com/citations?user=KInXUlUAAAAJ) ·
[IDEAS/RePEc](https://ideas.repec.org/f/pva1090.html) ·
[Banque de France](https://www.banque-france.fr/en/eric-vansteenberghe)

Comments, corrections and pull requests are welcome — including from students of the lecture.

*The views expressed in this material are those of the author and do not necessarily reflect those
of any of his past, present and future employers.*
