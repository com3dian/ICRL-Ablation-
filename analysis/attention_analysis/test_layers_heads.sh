#!/bin/bash

# Optimized attention head analysis pipeline for Qwen3-32B
# Uses two GPUs in parallel, extracts all heads per forward pass
# Total: 100 questions × 2 passes = 200 forward passes (split across 2 GPUs)

export MPLBACKEND=Agg

JSON_PATH="${1:?Usage: bash test_layers_heads.sh <path_to_output_list.json>}"
LOG_FILE="attention_correlation_results_qwen3.log"
OUTPUT_DIR_GPU0="results_gpu0"
OUTPUT_DIR_GPU1="results_gpu1"
MERGED_DIR="results_merged"
NUM_HEADS=64  # Qwen3-32B has 64 attention heads

echo "==========================================" > $LOG_FILE
echo "Attention Correlation Analysis - Qwen3-32B" >> $LOG_FILE
echo "100 questions, 4 layers × 64 heads" >> $LOG_FILE
echo "Started at: $(date)" >> $LOG_FILE
echo "==========================================" >> $LOG_FILE

# Step 1: Run attention extraction on both GPUs in parallel
echo "Launching GPU 0 (questions 0-49)..."
CUDA_VISIBLE_DEVICES=0 python analyze_all_questions.py \
    --gpu 0 --start-q 0 --end-q 50 \
    --output-dir $OUTPUT_DIR_GPU0 \
    --json-path "$JSON_PATH" \
    > gpu0.log 2>&1 &
PID_GPU0=$!

echo "Launching GPU 1 (questions 50-99)..."
CUDA_VISIBLE_DEVICES=1 python analyze_all_questions.py \
    --gpu 1 --start-q 50 --end-q 100 \
    --output-dir $OUTPUT_DIR_GPU1 \
    --json-path "$JSON_PATH" \
    > gpu1.log 2>&1 &
PID_GPU1=$!

echo "Waiting for both GPUs to finish..."
echo "  GPU 0 PID: $PID_GPU0"
echo "  GPU 1 PID: $PID_GPU1"
wait $PID_GPU0
echo "GPU 0 done (exit code: $?)"
wait $PID_GPU1
echo "GPU 1 done (exit code: $?)"

# Step 2: Merge results from both GPUs
echo "Merging results..."
mkdir -p $MERGED_DIR
python merge_results.py \
    --dir1 $OUTPUT_DIR_GPU0 \
    --dir2 $OUTPUT_DIR_GPU1 \
    --output-dir $MERGED_DIR

# Step 3: Run correlation analysis on each merged JSON
echo "" >> $LOG_FILE
echo "Running correlation analysis..." | tee -a $LOG_FILE

for layer in -1 -2 -3 -4; do
    echo "" >> $LOG_FILE
    echo "==== LAYER $layer ====" >> $LOG_FILE
    echo "" >> $LOG_FILE

    for head in $(seq 0 $((NUM_HEADS - 1))); do
        json_file="${MERGED_DIR}/attention_analysis_layer_${layer}_head_${head}.json"

        if [ ! -f "$json_file" ]; then
            echo "MISSING: $json_file" >> $LOG_FILE
            continue
        fi

        cp "$json_file" attention_analysis_all_questions.json
        python calculate_correlation.py > temp_correlation_output.txt 2>&1

        echo "----------------------------------------" >> $LOG_FILE
        echo "Layer: $layer, Head: $head" >> $LOG_FILE

        pearson=$(grep "Pearson correlation:" temp_correlation_output.txt | grep -o "r = [0-9.-]*" | cut -d' ' -f3)
        pearson_p=$(grep "Pearson correlation:" temp_correlation_output.txt | grep -o "p = [0-9.e-]*" | cut -d' ' -f3)
        t_stat=$(grep "T-statistic:" temp_correlation_output.txt | grep -o "[0-9.-]*" | head -1)
        t_p=$(grep "P-value:" temp_correlation_output.txt | grep -o "[0-9.e-]*" | head -1)

        echo "  Pearson r=$pearson (p=$pearson_p)" >> $LOG_FILE
        echo "  T-stat=$t_stat (p=$t_p)" >> $LOG_FILE

        grep -E "✓|✗" temp_correlation_output.txt | head -2 >> $LOG_FILE
        echo "" >> $LOG_FILE
    done
done

rm -f temp_correlation_output.txt

echo "==========================================" >> $LOG_FILE
echo "Analysis completed at: $(date)" >> $LOG_FILE
echo "==========================================" >> $LOG_FILE

echo "" >> $LOG_FILE
echo "SUMMARY OF SIGNIFICANT FINDINGS (p < 0.05):" >> $LOG_FILE
echo "----------------------------------------" >> $LOG_FILE
grep -B2 "✓ Significant" $LOG_FILE | grep "Layer:" >> $LOG_FILE

echo "All tests complete! Results saved to $LOG_FILE"
echo ""
echo "Top 10 significant findings:"
grep -B2 "✓ Significant" $LOG_FILE | grep "Layer:" | head -10
