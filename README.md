# MozzareLLM

[![Release](https://img.shields.io/github/v/release/cheeseman-lab/mozzarellm)](https://github.com/cheeseman-lab/mozzarellm/releases)
[![Python](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Nature Communications](https://img.shields.io/badge/Nat%20Commun-10.1038%2Fs41467--026--73643--7-b31b1b)](https://doi.org/10.1038/s41467-026-73643-7)

MozzareLLM annotates gene clusters from functional genomics screens with a large language model.
For each cluster it names the pathway that explains the cluster (or says none does), classifies
every gene as established, novel role or uncharacterized, and flags the understudied genes for
follow-up. Every call is grounded in per-gene evidence and written to disk for auditing.

## Getting Started

Install into a Python 3.11+ environment:

```bash
python -m pip install "mozzarellm @ git+https://github.com/cheeseman-lab/mozzarellm.git"
```

Claude models work out of the box; add `[openai]` or `[gemini]` to the requirement for those
providers (`"mozzarellm[openai] @ git+..."`). For development, clone the repo and run
`conda env create -f environment.yml && python -m pip install -e ".[dev]"`.

Put an `ANTHROPIC_API_KEY` in a `.env` file, then open
[`examples/analyze_clusters.ipynb`](examples/analyze_clusters.ipynb). It runs three worked
examples (optical pooled screen, DepMap co-essentiality, proteomics co-abundance) and shows how
to point the same two calls at your own screen.

The benchmark that selected the shipped prompts is in [`benchmarks/`](benchmarks/README.md).

## Citing MozzareLLM

MozzareLLM was created by [Matteo Di Bernardo](https://github.com/mat10d),
[Alexa Mallar](https://github.com/amallar428), and others in the
[Cheeseman Lab](https://cheesemanlab.wi.mit.edu/). It is presented in the Brieflow paper; if you use MozzareLLM, please cite:

```
@article{dibernardo2026brieflow,
  author  = {Di Bernardo, Matteo and Kern, Roshan S. and Cepeda Diaz, Ana Karla and Mallar, Alexa and Choi, Samuel J. and Nutter-Upham, Andrew and Lourido, Sebastian and Blainey, Paul C. and Cheeseman, Iain M.},
  title   = {Brieflow: an integrated computational pipeline for high-throughput analysis of optical pooled screening data},
  journal = {Nature Communications},
  volume  = {17},
  pages   = {6997},
  year    = {2026},
  doi     = {10.1038/s41467-026-73643-7}
}
```

## Contributing

We welcome contributions. If you find MozzareLLM useful, feel free to:

- Give the repo a star to boost MozzareLLM's visibility!
- File a [GitHub issue](https://github.com/cheeseman-lab/mozzarellm/issues) with comments or problems.
- Clone the repository, create a branch, and open a
  [pull request](https://github.com/cheeseman-lab/mozzarellm/compare). Run
  `python -m pytest tests/` before you push.

## License

MIT. See [LICENSE](LICENSE).
