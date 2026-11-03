# MozzareLLM

[![Release](https://img.shields.io/github/v/release/cheeseman-lab/mozzarellm)](https://github.com/cheeseman-lab/mozzarellm/releases)
[![Python](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![bioRxiv](https://img.shields.io/badge/bioRxiv-10.1101%2F2025.05.26.656231-b31b1b)](https://doi.org/10.1101/2025.05.26.656231)

MozzareLLM annotates gene clusters from functional genomics screens with a large language model.
For each cluster it names the pathway that explains the cluster (or says none does), classifies
every gene as established, novel role or uncharacterized, and flags the understudied genes for
follow-up. Every call is grounded in per-gene evidence and written to disk for auditing.

## Getting Started

Install into a Python 3.11+ environment:

```bash
python -m pip install "mozzarellm @ git+https://github.com/cheeseman-lab/mozzarellm.git"
```

Put an `ANTHROPIC_API_KEY` in a `.env` file, then open
[`examples/analyze_clusters.ipynb`](examples/analyze_clusters.ipynb). It runs three worked
examples (optical pooled screen, DepMap co-essentiality, proteomics co-abundance) and shows how
to point the same two calls at your own screen.

The benchmark that selected the shipped prompts is in [`benchmarks/`](benchmarks/README.md).

## Citing MozzareLLM

MozzareLLM was created by [Matteo Di Bernardo](https://github.com/mat10d),
[Alexa Mallar](https://github.com/amallar428), and others in the
[Cheeseman Lab](https://cheesemanlab.wi.mit.edu/). It is described in the Brieflow manuscript;
if you use MozzareLLM, please cite:

```
@ARTICLE
author={Di Bernardo, Matteo and Kern, Roshan S. and Mallar, Alexa and Nutter-Upham, Andy and Blainey, Paul C. and Cheeseman, Iain},
title={Brieflow: An Integrated Computational Pipeline for High-Throughput Analysis of Optical Pooled Screening Data},
year={2025},
DOI={10.1101/2025.05.26.656231}
```

## Contributing

- File a [GitHub issue](https://github.com/cheeseman-lab/mozzarellm/issues) with comments or problems.
- Clone the repository, create a branch, and open a
  [pull request](https://github.com/cheeseman-lab/mozzarellm/compare). Run
  `python -m pytest tests/` before you push.

## License

MIT. See [LICENSE](LICENSE).
