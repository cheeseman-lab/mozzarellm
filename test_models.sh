#!/bin/bash
# Genomics Pathway Analysis Pipeline
# This script processes gene cluster data and runs analysis through multiple LLM models

# Set up environment variables
DATA_DIR="data"
RESULTS_DIR="results/sample_gene_sets"
PROCESSED_FILE="${DATA_DIR}/sample_gene_sets.csv"
PROJECT_NAME="aconcagua_cell"

# Step 0: Create results directory if it doesn't exist
mkdir -p ${RESULTS_DIR}

echo "Starting genomics pathway analysis pipeline..."

# Step 2: Run analysis with OpenAI GPT-4o
echo "Running analysis with OpenAI GPT-4o..."
python main.py \
  --config config_openai.json \
  --mode cluster \
  --model "gpt-4o" \
  --input ${PROCESSED_FILE} \
  --input_sep "," \
  --gene_column "genes" \
  --gene_sep ";" \
  --batch_size 1 \
  --output_file ${RESULTS_DIR}/${PROJECT_NAME}_openai
  # --screen_info path_to_screen_info.txt \

# Step 3: Run analysis with Anthropic Claude-3-7-Sonnet
echo "Running analysis with Anthropic Claude-3-7-Sonnet..."
python main.py \
  --config config_anthropic.json \
  --mode cluster \
  --model "claude-3-7-sonnet-20250219" \
  --input ${PROCESSED_FILE} \
  --input_sep "," \
  --gene_column "genes" \
  --gene_sep ";" \
  --batch_size 2 \
  --output_file ${RESULTS_DIR}/${PROJECT_NAME}_anthropic
  # --screen_info path_to_screen_info.txt \

echo "Analysis complete. Results saved to ${RESULTS_DIR}/"