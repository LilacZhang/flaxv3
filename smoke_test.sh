#!/bin/bash
# ================================================================
# Smoke Test: verify all 7 experiment configs can launch and train
# Uses Boxing only, runs 200 steps per config (~1-2 min each)
# Total: ~10 min
# ================================================================

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

GAME="Boxing"
SEED=1
DEVICE=0
STEPS=200       # just enough to trigger training + logging

COMPLETED=0
TOTAL=7
FAILED=0
FAILED_NAMES=""

run_test() {
    local tag=$1
    shift
    local overrides=("$@")

    COMPLETED=$((COMPLETED + 1))
    echo ""
    echo "[$COMPLETED/$TOTAL] Testing: $tag"

    local t=$SECONDS

    if python train.py \
        training.env_name="$GAME" \
        training.seed=$SEED \
        training.device=$DEVICE \
        training.total_steps=$STEPS \
        "${overrides[@]}" 2>&1; then
        echo "  PASS  ($tag)  $(( SECONDS - t ))s"
    else
        echo "  FAIL  ($tag)"
        FAILED=$((FAILED + 1))
        FAILED_NAMES="$FAILED_NAMES $tag"
    fi
}

echo "=========================================="
echo " Smoke Test — $GAME, $STEPS steps each"
echo " GPU: $DEVICE"
echo "=========================================="

START=$SECONDS

# Group A: imagine_length
run_test "IL=5"          training.imagine_length=5
run_test "baseline"      # all defaults
run_test "IL=30"         training.imagine_length=30

# Group B: gamma
run_test "gamma=0.99"    agent.gamma=0.99
run_test "gamma=0.999"   agent.gamma=0.999

# Group C: entropy_coef
run_test "ent=1e-4"      agent.entropy_coef=0.0001
run_test "ent=1e-3"      agent.entropy_coef=0.001

ELAPSED=$(( SECONDS - START ))
echo ""
echo "=========================================="
if [ $FAILED -eq 0 ]; then
    echo " ALL $TOTAL PASSED  (${ELAPSED}s)"
    echo " Safe to run: bash run_experiments.sh"
else
    echo " $FAILED/$TOTAL FAILED: $FAILED_NAMES"
    echo " Fix before running full experiments"
fi
echo "=========================================="
