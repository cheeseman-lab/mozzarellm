# mozzarellm

# Gene Set Analysis with LLMs

This repository contains code to analyze gene sets using various LLM APIs (OpenAI, Google Gemini, Anthropic Claude, and custom models).

## Installation

1. Clone this repository
   ```bash
   git clone https://github.com/yourusername/mozzarellm.git
   cd mozzarellm
   ```

2. Create and activate a conda environment
   ```bash
    conda env create -f environment.yml
    conda activate mozzarellm
   ```

3. Install dependencies
   ```bash
   conda install -c conda-forge pandas numpy tqdm
   pip install openai==1.6.1 anthropic==0.9.1 google-generativeai==0.3.2 python-dotenv==1.0.0 requests>=2.32.0
   ```

4. Copy `.env.example` to `.env` and add your API keys:
   ```
   OPENAI_API_KEY=your_openai_key_here
   ANTHROPIC_API_KEY=your_anthropic_key_here
   GOOGLE_API_KEY=your_google_key_here
   PERPLEXITY_API_KEY=your_perplexity_key_here
   ```

5. Modify `config.json` to set your preferred models and parameters

## Usage

```bash
python main.py --config config.json \
               --input data/sample_gene_sets.csv \
               --input_sep "," \
               --gene_column "genes" \
               --gene_sep ";" \
               --start 0 \
               --end 5 \
               --initialize \
               --output_file results/analysis
```

### Arguments

- `--config`: Path to configuration JSON file
- `--input`: Path to input CSV with gene sets
- `--input_sep`: Separator for input CSV
- `--gene_column`: Column name containing gene set
- `--gene_sep`: Separator for genes within a set
- `--start`: Start index for processing
- `--end`: End index for processing
- `--output_file`: Output file path (without extension)
- `--initialize`: Initialize output columns if needed
- `--run_contaminated`: Process contaminated gene sets

## Output

The script produces:
- A TSV file with gene set names, scores, and analyses
- A JSON file with full LLM responses
- A log file tracking all API calls and errors

## Structure

```
mozzarellm/
├── main.py                      # Main script for running gene analysis
├── config.json                  # Configuration for models and settings
├── .env                         # Environment variables with API keys (not committed)
├── .env.example                 # Template for .env file (committed)
├── requirements.txt             # Python dependencies (legacy)
├── .gitignore                   # Git ignore file
├── README.md                    # Project documentation
├── constant.py                  # Global constants
├── data/                        # Directory for input data
│   ├── .gitkeep                 # Keep directory in git even if empty
│   └── sample_gene_sets.csv     # Sample input file
├── prompts/                     # Prompt templates
│   ├── .gitkeep
│   └── custom_prompt.txt        # Custom prompt template
├── results/                     # Output directory
│   └── .gitkeep
├── utils/                       # Utility modules
│   ├── __init__.py
│   ├── openai_query.py          # OpenAI API handler
│   ├── anthropic_query.py       # Anthropic API handler
│   ├── genai_query.py           # Google Gemini API handler
│   ├── perplexity_query.py      # Perplexity API handler
│   ├── server_model_query.py    # Custom server model handler
│   ├── prompt_factory.py        # Functions to create prompts
│   ├── llm_analysis_utils.py    # Processing LLM responses
│   └── logging_utils.py         # Logging configuration
└── tests/                       # Unit tests
    ├── __init__.py
    ├── test_prompt_factory.py
    ├── test_openai_query.py
    ├── test_anthropic_query.py
    └── test_analysis_utils.py
```

## Examples

Here's an example of analyzing the provided sample gene sets:

```bash
# Analyze all gene sets in the sample file
python main.py --config config.json \
               --input data/sample_gene_sets.csv \
               --input_sep "," \
               --gene_column "genes" \
               --gene_sep ";" \
               --start 0 \
               --end 5 \
               --initialize \
               --output_file results/full_analysis
```