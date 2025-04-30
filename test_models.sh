#!/bin/bash
# Genomics Pathway Analysis Pipeline
# This script processes gene cluster data and runs analysis through multiple LLM models

# Set up environment variables
DATA_DIR="data"
RESULTS_DIR="results/sample_gene_sets"
PROCESSED_FILE="${DATA_DIR}/sample_gene_sets.csv"
GENE_FEATURES="${DATA_DIR}/HeLa_essentials/essentials_uniprot.csv"
PROJECT_NAME="sample_gene_sets"

# Step 0: Create results directory if it doesn't exist
mkdir -p ${RESULTS_DIR}

echo "Starting genomics pathway analysis pipeline..."

# Step 2: Run analysis with OpenAI GPT-4o
echo "Running analysis with OpenAI o4-mini..."
python main.py \
  --config config_openai.json \
  --mode cluster \
  --model "o4-mini" \
  --custom_prompt "prompts/top_targets.txt" \
  --input ${PROCESSED_FILE} \
  --input_sep "," \
  --gene_column "genes" \
  --gene_sep ";" \
  --gene_features ${GENE_FEATURES} \
  --batch_size 2 \
  --output_file ${RESULTS_DIR}/${PROJECT_NAME}_openai \
  --screen_info prompts/HeLa_interphase_screen_info.txt \

# Step 3: Run analysis with Anthropic Claude-3-7-Sonnet
echo "Running analysis with Anthropic Claude-3-7-Sonnet..."
python main.py \
  --config config_anthropic.json \
  --mode cluster \
  --model "claude-3-7-sonnet-20250219" \
  --custom_prompt "prompts/top_targets.txt" \
  --input ${PROCESSED_FILE} \
  --input_sep "," \
  --gene_column "genes" \
  --gene_sep ";" \
  --gene_features ${GENE_FEATURES} \
  --batch_size 2 \
  --output_file ${RESULTS_DIR}/${PROJECT_NAME}_anthropic \
  --screen_info prompts/HeLa_interphase_screen_info.txt \

echo "Analysis complete. Results saved to ${RESULTS_DIR}/"