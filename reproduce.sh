
#!/usr/bin/env bash

set -euo pipefail

# Move to the repository root
cd "$(dirname "$0")"

# Use the project virtual environment if available
if [[ -x ".venv/bin/python" ]]; then
    PYTHON=".venv/bin/python"
else
    PYTHON="python3"
fi

# Select which experiment to reproduce
MODE="${1:-help}"

run_q1() {
    echo "Running Question 1"

    "$PYTHON" src/decision_tree.py
    "$PYTHON" src/bagging.py
    "$PYTHON" src/bagging_plots.py
    "$PYTHON" src/q1d_comparison.py

    echo "Question 1 completed"
}

run_q2() {
    echo "Running Question 2"

    "$PYTHON" src/q2a_samme_demo.py
    "$PYTHON" src/adaboost.py
    "$PYTHON" src/q2c_comparison.py
    "$PYTHON" src/q2d_comparison.py

    echo "Question 2 completed"
}

run_q3() {
    echo "Running Question 3"

    "$PYTHON" src/bias_variance.py
    "$PYTHON" src/q3b_depth_analysis.py
    "$PYTHON" src/q3_plots.py

    echo "Question 3 completed"
}

run_plots() {
    echo "Regenerating plots from saved results"

    "$PYTHON" src/bagging_plots.py
    "$PYTHON" src/q3_plots.py

    echo "Saved-result plots completed"
}

# Check that the saved dataset split exists
if [[ ! -f "data/splits/olivetti_split.npz" ]]; then
    echo "Error: Saved dataset split is missing."
    echo "The assignment requires the original split."
    echo "Do not regenerate it unless explicitly necessary."
    exit 1
fi

# Execute the requested experiment
case "$MODE" in
    q1)
        run_q1
        ;;
    q2)
        run_q2
        ;;
    q3)
        run_q3
        ;;
    all)
        run_q1
        run_q2
        run_q3
        ;;
    plots)
        run_plots
        ;;
    *)
        echo "Usage: ./reproduce.sh {q1|q2|q3|all|plots}"
        exit 1
        ;;
esac
