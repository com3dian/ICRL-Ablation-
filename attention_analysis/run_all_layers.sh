#!/bin/bash
# Run attention analysis across many layers in batches
# Each batch handles 8 target layers to stay within GPU memory

export MPLBACKEND=Agg

JSON_PATH="${1:?Usage: bash run_all_layers.sh <path_to_output_list.json>}"

echo "Starting extended layer analysis at: $(date)"

# Batch 1: layers -5 to -8 (already have -1 to -4)
echo "=== BATCH 1: layers -5 to -8 ==="
CUDA_VISIBLE_DEVICES=0 python analyze_all_questions.py \
    --start-q 0 --end-q 50 --output-dir results_b1_gpu0 \
    --json-path "$JSON_PATH" \
    --target-layers -5 -6 -7 -8 \
    > batch1_gpu0.log 2>&1 &
PID1=$!

CUDA_VISIBLE_DEVICES=1 python analyze_all_questions.py \
    --start-q 50 --end-q 100 --output-dir results_b1_gpu1 \
    --json-path "$JSON_PATH" \
    --target-layers -5 -6 -7 -8 \
    > batch1_gpu1.log 2>&1 &
PID2=$!

wait $PID1; echo "Batch1 GPU0 done ($?)"
wait $PID2; echo "Batch1 GPU1 done ($?)"

# Merge batch 1
python merge_results.py --dir1 results_b1_gpu0 --dir2 results_b1_gpu1 --output-dir results_merged_b1

# Batch 2: layers -9 to -16
echo "=== BATCH 2: layers -9 to -16 ==="
CUDA_VISIBLE_DEVICES=0 python analyze_all_questions.py \
    --start-q 0 --end-q 50 --output-dir results_b2_gpu0 \
    --json-path "$JSON_PATH" \
    --target-layers -9 -10 -11 -12 -13 -14 -15 -16 \
    > batch2_gpu0.log 2>&1 &
PID1=$!

CUDA_VISIBLE_DEVICES=1 python analyze_all_questions.py \
    --start-q 50 --end-q 100 --output-dir results_b2_gpu1 \
    --json-path "$JSON_PATH" \
    --target-layers -9 -10 -11 -12 -13 -14 -15 -16 \
    > batch2_gpu1.log 2>&1 &
PID2=$!

wait $PID1; echo "Batch2 GPU0 done ($?)"
wait $PID2; echo "Batch2 GPU1 done ($?)"

python merge_results.py --dir1 results_b2_gpu0 --dir2 results_b2_gpu1 --output-dir results_merged_b2

# Batch 3: layers -17 to -32
echo "=== BATCH 3: layers -17 to -32 ==="
CUDA_VISIBLE_DEVICES=0 python analyze_all_questions.py \
    --start-q 0 --end-q 50 --output-dir results_b3_gpu0 \
    --json-path "$JSON_PATH" \
    --target-layers -17 -18 -19 -20 -21 -22 -23 -24 -25 -26 -27 -28 -29 -30 -31 -32 \
    > batch3_gpu0.log 2>&1 &
PID1=$!

CUDA_VISIBLE_DEVICES=1 python analyze_all_questions.py \
    --start-q 50 --end-q 100 --output-dir results_b3_gpu1 \
    --json-path "$JSON_PATH" \
    --target-layers -17 -18 -19 -20 -21 -22 -23 -24 -25 -26 -27 -28 -29 -30 -31 -32 \
    > batch3_gpu1.log 2>&1 &
PID2=$!

wait $PID1; echo "Batch3 GPU0 done ($?)"
wait $PID2; echo "Batch3 GPU1 done ($?)"

python merge_results.py --dir1 results_b3_gpu0 --dir2 results_b3_gpu1 --output-dir results_merged_b3

# Consolidate all results into one directory
echo "=== Consolidating all results ==="
mkdir -p results_all_layers
cp results_merged/attention_analysis_layer_*.json results_all_layers/      # layers -1 to -4
cp results_merged_b1/attention_analysis_layer_*.json results_all_layers/   # layers -5 to -8
cp results_merged_b2/attention_analysis_layer_*.json results_all_layers/   # layers -9 to -16
cp results_merged_b3/attention_analysis_layer_*.json results_all_layers/   # layers -17 to -32

echo "Total files: $(ls results_all_layers/ | wc -l)"
echo "Completed at: $(date)"
