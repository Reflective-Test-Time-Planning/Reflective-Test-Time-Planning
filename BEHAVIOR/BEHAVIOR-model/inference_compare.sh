#!/bin/bash

# Bash script to run compare task inference on multiple scene configs
# Usage: ./run_compare_batch.sh [TYPE]
# TYPE can be: vanilla, full, wo-roa, wo-ria, wo-value, wo-action

set -e  # Exit on error

# Check if type argument is provided
if [ $# -eq 0 ]; then
    echo "Error: No experiment type provided!"
    echo "Usage: $0 [TYPE]"
    echo "  TYPE can be: vanilla, full, wo-roa, wo-ria, wo-value, wo-action"
    exit 1
fi

EXPERIMENT_TYPE=$1

# Validate experiment type
if [[ ! "$EXPERIMENT_TYPE" =~ ^(vanilla|full|wo-roa|wo-ria|wo-value|wo-action)$ ]]; then
    echo "Error: Invalid experiment type '$EXPERIMENT_TYPE'"
    echo "Valid types: vanilla, full, wo-roa, wo-ria, wo-value, wo-action"
    exit 1
fi

echo "Running experiment type: $EXPERIMENT_TYPE"

# Configuration
MODEL_PATH="evelynhong/rettp"
SCENE_CONFIG_DIR="compare_data_sample"
OUTPUT_DIR="compare_results_final_${EXPERIMENT_TYPE}"
LOG_DIR="compare_logs_final_${EXPERIMENT_TYPE}"

# Default parameters
MAX_ITERATIONS=25
LOAD_4BIT="--load-4bit"

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# Check if scene config directory exists
if [ ! -d "$SCENE_CONFIG_DIR" ]; then
    echo "Error: $SCENE_CONFIG_DIR directory not found!"
    exit 1
fi

# Get all JSON files in the directory and shuffle
echo "Loading scene files from $SCENE_CONFIG_DIR..."
SCENE_FILES=$(find "$SCENE_CONFIG_DIR" -name "*.json" -type f | shuf)

# Count total scenes
TOTAL_SCENES=$(echo "$SCENE_FILES" | wc -l)
echo "Found $TOTAL_SCENES scenes to process (shuffled order)"

# Statistics
COMPLETED=0  # Successfully completed (exit code 42)
FAILED=0     # Failed/incomplete but script ran (exit code 43)
CRASHED=0    # Script crashed/error (other exit codes)
CURRENT=0
TOTAL_VALID=0  # Total scenes that completed normally (exit 42 or 43)

# Iterate through each scene file
while IFS= read -r scene_path; do
    CURRENT=$((CURRENT + 1))
    
    # Extract filename
    SCENE_FILE=$(basename "$scene_path")
    
    echo ""
    echo "=========================================="
    echo "Processing scene $CURRENT/$TOTAL_SCENES"
    echo "Scene: $SCENE_FILE"
    echo "Experiment Type: $EXPERIMENT_TYPE"
    echo "=========================================="
    
    # Extract base name for output files
    BASE_NAME=$(basename "$SCENE_FILE" .json)
    OUTPUT_FILE="$OUTPUT_DIR/${BASE_NAME}_result.json"
    LOG_FILE="$LOG_DIR/${BASE_NAME}.log"
    
    # Check if already processed
    if [ -f "$OUTPUT_FILE" ]; then
        echo "Output file exists: $OUTPUT_FILE"
        
        # Check status field to see if it completed
        if grep -q '"status".*:.*"completed_successfully"' "$OUTPUT_FILE" 2>/dev/null; then
            echo "Already completed successfully. Skipping..."
            COMPLETED=$((COMPLETED + 1))
            TOTAL_VALID=$((TOTAL_VALID + 1))
            continue
        elif grep -q '"status".*:.*"completed_unsuccessfully"' "$OUTPUT_FILE" 2>/dev/null; then
            echo "Already completed unsuccessfully. Skipping..."
            FAILED=$((FAILED + 1))
            TOTAL_VALID=$((TOTAL_VALID + 1))
            continue
        elif grep -q '"status".*:.*"in_progress"' "$OUTPUT_FILE" 2>/dev/null; then
            echo "Found incomplete run (crashed mid-execution). Re-running..."
            CRASHED=$((CRASHED + 1))
            # Continue to re-run this one
        else
            echo "Found output file but couldn't determine status. Re-running..."
            # Continue to re-run
        fi
    fi
    
    echo "Running inference..."
    echo "Config: $scene_path"
    echo "Output: $OUTPUT_FILE"
    echo "Log: $LOG_FILE"
    
    # Run the inference script with experiment type argument
    # Capture exit code: 42 = success, 43 = failure, anything else = crash
    # Use tee to show output on terminal AND save to log file
    set +e  # Temporarily disable exit on error to capture exit code
    
    python inference_compare.py \
        --model-path "$MODEL_PATH" \
        --scene-config "$scene_path" \
        $LOAD_4BIT \
        --max-iterations $MAX_ITERATIONS \
        --output-file "$OUTPUT_FILE" \
        --experiment-type "$EXPERIMENT_TYPE" \
        2>&1 | tee "$LOG_FILE"
    
    EXIT_CODE=${PIPESTATUS[0]}
    set -e  # Re-enable exit on error
    
    # Check exit code
    if [ $EXIT_CODE -eq 42 ]; then
        echo "✓✓✓ Task COMPLETED successfully! ✓✓✓"
        COMPLETED=$((COMPLETED + 1))
        TOTAL_VALID=$((TOTAL_VALID + 1))
    elif [ $EXIT_CODE -eq 43 ]; then
        echo "✗✗✗ Task FAILED or incomplete ✗✗✗"
        FAILED=$((FAILED + 1))
        TOTAL_VALID=$((TOTAL_VALID + 1))
    else
        echo "💥💥💥 Script CRASHED (exit code: $EXIT_CODE) 💥💥💥"
        CRASHED=$((CRASHED + 1))
    fi
    
    # Print current statistics
    echo ""
    echo "--- Statistics ---"
    echo "Processed: $CURRENT/$TOTAL_SCENES"
    echo "Completed (Right): $COMPLETED"
    echo "Failed (Wrong): $FAILED"
    echo "Crashed (Not counted): $CRASHED"
    echo "Valid runs: $TOTAL_VALID"
    if [ $TOTAL_VALID -gt 0 ]; then
        echo "Accuracy: $(awk "BEGIN {printf \"%.1f\", ($COMPLETED/$TOTAL_VALID)*100}")%"
    else
        echo "Accuracy: N/A (no valid runs yet)"
    fi
    echo ""
    
    # Optional: Add a small delay between scenes to allow cleanup
    sleep 2

    rm -rf room_pointclouds_compare/*
    
done <<< "$SCENE_FILES"

# Final summary
echo ""
echo "=========================================="
echo "FINAL SUMMARY"
echo "=========================================="
echo "Experiment Type: $EXPERIMENT_TYPE"
echo "Total Scenes Attempted: $TOTAL_SCENES"
echo "Completed (Right): $COMPLETED"
echo "Failed (Wrong): $FAILED"
echo "Crashed (Not counted): $CRASHED"
echo "Valid Runs: $TOTAL_VALID"
if [ $TOTAL_VALID -gt 0 ]; then
    echo "Accuracy: $(awk "BEGIN {printf \"%.1f\", ($COMPLETED/$TOTAL_VALID)*100}")% ($COMPLETED/$TOTAL_VALID)"
else
    echo "Accuracy: N/A (no valid runs)"
fi
echo ""
echo "Results saved in: $OUTPUT_DIR"
echo "Logs saved in: $LOG_DIR"
echo "=========================================="

# Create summary JSON
SUMMARY_FILE="$OUTPUT_DIR/summary.json"

# Calculate accuracy safely
if [ $TOTAL_VALID -gt 0 ]; then
    ACCURACY=$(awk "BEGIN {printf \"%.2f\", ($COMPLETED/$TOTAL_VALID)*100}")
else
    ACCURACY="null"
fi

cat > "$SUMMARY_FILE" << EOF
{
  "experiment_type": "$EXPERIMENT_TYPE",
  "total_scenes": $TOTAL_SCENES,
  "completed_right": $COMPLETED,
  "failed_wrong": $FAILED,
  "crashed": $CRASHED,
  "valid_runs": $TOTAL_VALID,
  "accuracy": $ACCURACY,
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
}
EOF

echo "Summary saved to: $SUMMARY_FILE"