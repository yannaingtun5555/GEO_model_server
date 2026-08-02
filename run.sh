#!/usr/bin/env bash
# run.sh — Convenience wrapper for all pipeline scripts
# Usage:
#   ./run.sh preprocess
#   ./run.sh label
#   ./run.sh view --list
#   ./run.sh view -r ayeyawaddy -y 2018 -m 01
#   ./run.sh view -r ayeyawaddy -y 2018 -m 01 --distributions
#   ./run.sh view -r ayeyawaddy -y 2018 -m 01 --labels
#   ./run.sh view -r ayeyawaddy -y 2018 -m 01 --summary
#   ./run.sh view -r ayeyawaddy -y 2018 -m 01 --columns
#   ./run.sh all        # run full pipeline: preprocess + label
#   ./run.sh serve      # run the private model API on localhost:8001

if [ -f "$(cd "$(dirname "$0")" && pwd)/.venv/bin/python3" ]; then
    PYTHON="$(cd "$(dirname "$0")" && pwd)/.venv/bin/python3"
else
    PYTHON="python3"
fi
SCRIPTS_DIR="$(cd "$(dirname "$0")/scripts" && pwd)"

if [ -z "$1" ]; then
    echo "Usage: ./run.sh <command> [args]"
    echo ""
    echo "Commands:"
    echo "  preprocess          Merge static + dynamic CSVs into data/processed/"
    echo "  label               Apply 40-column labels to processed CSVs"
    echo "  combine             Drop noise cols + combine all CSVs into data/combined/"
    echo "  train               Train ML models on data/combined/combined_dataset.csv"
    echo "  train --quick       Train with small models (fast, for testing)"
    echo "  train --target X    Train only one target prediction"
    echo "  test                Test model accuracy on data/combined/combined_dataset.csv"
    echo "  test-gp             Test all models in gp_models/ directory one by one"
    echo "  test --mode full    Test accuracy on entire dataset (default: 20% test split)"
    echo "  test --target X     Test accuracy for specific target model"
    echo "  recommend           Rank best crops to plant for each region"
    echo "  pipeline            Test the 40-model end-to-end inference pipeline & BE JSON output"
    echo "  serve               Run the FastAPI model server (default: 127.0.0.1:8001)"
    echo "  all                 Run preprocess + label in sequence"
    echo ""
    echo "View options:"
    echo "  --list                     List all processed files"
    echo "  -r REGION -y YEAR -m MONTH  Select a specific file"
    echo "  --labels                   Show label columns"
    echo "  --features                 Show only feature columns"
    echo "  --distributions            Show label bar charts"
    echo "  --summary                  Show statistics"
    echo "  --columns                  List all columns with dtypes"
    echo "  --cols col1,col2           Show specific columns"
    echo "  --rows N                   Number of rows (default 10)"
    echo "  --all                      Apply to all files"
    exit 0
fi

CMD="$1"
shift

case "$CMD" in
    preprocess)
        $PYTHON "$SCRIPTS_DIR/preprocess.py" "$@"
        ;;
    label)
        $PYTHON "$SCRIPTS_DIR/label.py" "$@"
        ;;
    view)
        $PYTHON "$SCRIPTS_DIR/view_data.py" "$@"
        ;;
    combine)
        $PYTHON "$SCRIPTS_DIR/combine.py" "$@"
        ;;
    train)
        $PYTHON "$SCRIPTS_DIR/train.py" "$@"
        ;;
    test|evaluate)
        $PYTHON "$SCRIPTS_DIR/test_accuracy.py" "$@"
        ;;
    test-gp|test_gp)
        $PYTHON "$SCRIPTS_DIR/test_gp_models.py" "$@"
        ;;
    recommend)
        $PYTHON "$SCRIPTS_DIR/recommend_crops.py" "$@"
        ;;
    pipeline)
        $PYTHON "pipeline/test_pipeline.py" "$@"
        ;;
    serve)
        HOST="${HOST:-127.0.0.1}"
        PORT="${PORT:-8001}"
        exec "$PYTHON" -m uvicorn server.main:app --host "$HOST" --port "$PORT" "$@"
        ;;
    all)
        echo ">>> Running preprocess..."
        $PYTHON "$SCRIPTS_DIR/preprocess.py" && \
        echo ">>> Running label..." && \
        $PYTHON "$SCRIPTS_DIR/label.py" && \
        echo ">>> Combining datasets..." && \
        $PYTHON "$SCRIPTS_DIR/combine.py"
        ;;
    *)
        echo "Unknown command: $CMD"
        echo "Run ./run.sh for help."
        exit 1
        ;;
esac
