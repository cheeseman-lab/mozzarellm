#!/bin/bash
# Genomics Pathway Analysis Pipeline
# This script processes gene cluster data and runs analysis through multiple LLM models

# Set up environment variables
DATA_DIR="data"
RESULTS_DIR="results"
INPUT_FILE="${DATA_DIR}/df_phate_i.csv"
PROCESSED_FILE="${DATA_DIR}/luke_clusters.csv"
PROJECT_NAME="aconcagua_cell"

echo "Starting genomics pathway analysis pipeline..."

# Step 0: Create results directory if it doesn't exist
mkdir -p ${RESULTS_DIR}

# Step 1: Reshape clusters for analysis
# echo "Preprocessing data: Converting raw data to cluster format..."
# python reshape_clusters.py \
#   --input ${INPUT_FILE} \
#   --output ${PROCESSED_FILE} \
#   --sep "," \
#   --gene_col "gene_symbol" \
#   --cluster_col "cluster" \
#   --gene_sep ";" \
#   --additional_cols "cluster_group"

# # TODO: remove this testing block
# python main.py \
#   --config config_openai.json \
#   --mode cluster \
#   --input ${PROCESSED_FILE} \
#   --input_sep "," \
#   --set_index "cluster_id" \
#   --gene_column "genes" \
#   --gene_sep ";" \
#   --batch_size 2 \
#   --start 0 \
#   --end 5 \
#   --output_file ${RESULTS_DIR}/${PROJECT_NAME}_openai

# # Step 2: Run analysis with OpenAI GPT-4o
# echo "Running analysis with OpenAI GPT-4o..."
# python main.py \
#   --config config_openai.json \
#   --mode cluster \
#   --input ${PROCESSED_FILE} \
#   --input_sep "," \
#   --set_index "cluster_id" \
#   --gene_column "genes" \
#   --gene_sep ";" \
#   --batch_size 2 \
#   --output_file ${RESULTS_DIR}/${PROJECT_NAME}_openai

# # Step 3: Run analysis with Anthropic Claude-3-7-Sonnet
# echo "Running analysis with Anthropic Claude-3-7-Sonnet..."
# python main.py \
#   --config config_anthropic.json \
#   --mode cluster \
#   --input ${PROCESSED_FILE} \
#   --input_sep "," \
#   --set_index "cluster_id" \
#   --gene_column "genes" \
#   --gene_sep ";" \
#   --batch_size 2 \
#   --output_file ${RESULTS_DIR}/${PROJECT_NAME}_anthropic

# Step 4: Run with DeepSeek-R1
# echo "Running analysis with Perplexity DeepSeek-R1..."
# python main.py \
#   --config config_deepseek.json \
#   --mode cluster \
#   --input ${PROCESSED_FILE} \
#   --input_sep "," \
#   --set_index "cluster_id" \
#   --gene_column "genes" \
#   --gene_sep ";" \
#   --batch_size 2 \
#   --output_file ${RESULTS_DIR}/${PROJECT_NAME}_deepseek

# # Step 5: Run with sonar
# echo "Running analysis with sonar..."
# python main.py \
#   --config config_sonar.json \
#   --mode cluster \
#   --input ${PROCESSED_FILE} \
#   --input_sep "," \
#   --set_index "cluster_id" \
#   --gene_column "genes" \
#   --gene_sep ";" \
#   --batch_size 2 \
#   --output_file ${RESULTS_DIR}/${PROJECT_NAME}_sonar

# Step 6: Run with Gemini
echo "Running analysis with Gemini..."
python main.py \
  --config config_gemini.json \
  --mode cluster \
  --input ${PROCESSED_FILE} \
  --input_sep "," \
  --set_index "cluster_id" \
  --gene_column "genes" \
  --gene_sep ";" \
  --batch_size 2 \
  --output_file ${RESULTS_DIR}/${PROJECT_NAME}_gemini

echo "Analysis complete. Results saved to ${RESULTS_DIR}/"