#!/bin/bash

# Script to run all creative writing experiments sequentially
# This script will run each experiment one after another, with logging and error handling

# Set up logging directory
LOG_DIR="/sfs/weka/scratch/ks8vf/code_submission/ICRL"
mkdir -p "$LOG_DIR"

# Get current timestamp for log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Function to log messages
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_DIR/run_all_${TIMESTAMP}.log"
}

# Function to run experiment with error handling
run_experiment() {
    local script_name=$1
    local experiment_name=$2
    
    log_message "Starting $experiment_name..."
    
    # Run the experiment and capture output
    python "$script_name" > "$LOG_DIR/${experiment_name}_${TIMESTAMP}.log" 2>&1
    
    # Check exit status
    if [ $? -eq 0 ]; then
        log_message "✓ $experiment_name completed successfully"
    else
        log_message "✗ $experiment_name failed with error code $?"
        log_message "Check log file: $LOG_DIR/${experiment_name}_${TIMESTAMP}.log"
        # Continue with next experiment even if this one failed
    fi
    
    log_message "----------------------------------------"
}

# Main execution
log_message "Starting all creative writing experiments"
log_message "Log directory: $LOG_DIR"
log_message "========================================"

# Run ICRL experiment
# run_experiment "llm_creative_writing_api.py" "ICRL"

# Add a small delay between experiments to avoid resource conflicts
# sleep 30

# Run Self-Refine experiment
run_experiment "llm_creative_writing_api_self-refine.py" "Self-Refine"

# Add delay
sleep 30

# Run Reflexion experiment
run_experiment "llm_creative_writing_api_reflexion.py" "Reflexion"

# Summary
log_message "========================================"
log_message "All experiments completed!"
log_message "Check individual log files in: $LOG_DIR"

# Optional: Send notification (uncomment if you have mail configured)
# echo "Creative writing experiments completed at $(date)" | mail -s "Experiments Complete" your-email@example.com