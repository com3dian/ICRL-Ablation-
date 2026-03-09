#!/usr/bin/env python3
"""
Merge attention analysis results from two GPU runs (split by question range).
Each GPU produces JSONs with a subset of questions; this script concatenates them.
"""

import json
import os
import glob
import argparse


def merge_json_files(file1, file2, output_file):
    """Merge two attention analysis JSONs by concatenating question-level lists."""
    with open(file1, 'r') as f:
        data1 = json.load(f)
    with open(file2, 'r') as f:
        data2 = json.load(f)

    merged = {
        "num_questions_processed": data1["num_questions_processed"] + data2["num_questions_processed"],
        "num_trials_per_question": data1["num_trials_per_question"],
        "test_rewards_per_question": data1["test_rewards_per_question"] + data2["test_rewards_per_question"],
        "baseline_rewards": data1["baseline_rewards"],
        "raw_attentions": data1["raw_attentions"] + data2["raw_attentions"],
        "baseline_attentions": data1["baseline_attentions"] + data2["baseline_attentions"],
        "adjusted_attentions": data1["adjusted_attentions"] + data2["adjusted_attentions"],
    }

    with open(output_file, 'w') as f:
        json.dump(merged, f, indent=2)

    return merged["num_questions_processed"]


def main():
    parser = argparse.ArgumentParser(description='Merge results from two GPU runs')
    parser.add_argument('--dir1', required=True, help='Directory with GPU 0 results')
    parser.add_argument('--dir2', required=True, help='Directory with GPU 1 results')
    parser.add_argument('--output-dir', required=True, help='Directory for merged results')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Find all JSON files from GPU 0
    files1 = glob.glob(os.path.join(args.dir1, "attention_analysis_layer_*.json"))
    print(f"Found {len(files1)} files in {args.dir1}")

    merged_count = 0
    for f1 in sorted(files1):
        basename = os.path.basename(f1)
        f2 = os.path.join(args.dir2, basename)
        output = os.path.join(args.output_dir, basename)

        if os.path.exists(f2):
            n = merge_json_files(f1, f2, output)
            merged_count += 1
            print(f"  Merged {basename}: {n} questions")
        else:
            # Only one GPU produced this file, just copy it
            import shutil
            shutil.copy2(f1, output)
            print(f"  Copied {basename} (only in dir1)")

    # Check for files only in dir2
    files2 = glob.glob(os.path.join(args.dir2, "attention_analysis_layer_*.json"))
    for f2 in sorted(files2):
        basename = os.path.basename(f2)
        f1 = os.path.join(args.dir1, basename)
        if not os.path.exists(f1):
            import shutil
            output = os.path.join(args.output_dir, basename)
            shutil.copy2(f2, output)
            print(f"  Copied {basename} (only in dir2)")

    print(f"\nMerge complete: {merged_count} files merged to {args.output_dir}")


if __name__ == "__main__":
    main()
