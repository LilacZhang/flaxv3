#!/bin/bash
# ================================================================
# DreamerV3 Hyperparameter Comparison (Control Variable Method)
#
# Games:  Freeway / Breakout / Boxing
# Group A: imagine_length  = 5 / 15* / 30
# Group B: gamma           = 0.99 / 0.997* / 0.999
# Group C: entropy_coef    = 1e-4 / 3e-4* / 1e-3
#                                       (* = baseline, shared)
#
# Per game: 3 + 2 + 2 = 7 unique runs  (baseline not repeated)
# Total:    2 × 7 = 14 experiments      (~3.5 days on single GPU)
# ================================================================

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

# ---- Configuration ----
GAMES=("Freeway" "Boxing")
SEED=1
DEVICE=0
TOTAL_STEPS=100000

# ---- Defaults (baseline) ----
DEFAULT_IL=15
DEFAULT_GAMMA=0.997
DEFAULT_ENT=0.0003

COMPLETED=0
TOTAL=14

run() {
    local game=$1
    local tag=$2
    shift 2
    local overrides=("$@")

    COMPLETED=$((COMPLETED + 1))
    echo ""
    echo "=========================================="
    echo " [$COMPLETED/$TOTAL] $game | $tag"
    echo " $(date '+%H:%M:%S')"
    echo "=========================================="

    local t=$SECONDS

    python train.py \
        training.env_name="$game" \
        training.seed=$SEED \
        training.device=$DEVICE \
        training.total_steps=$TOTAL_STEPS \
        "${overrides[@]}"

    echo " -> done in $(( (SECONDS - t) / 60 ))min"
}

echo "Starting $TOTAL experiments on GPU $DEVICE"
echo "Games: ${GAMES[*]} (Freeway=sparse reward, Boxing=dense reward)"
echo ""

GLOBAL_START=$SECONDS

for game in "${GAMES[@]}"; do

    # ---- Group A: imagine_length ----
    run "$game" "IL=5" \
        training.imagine_length=5

    run "$game" "baseline (IL=15,g=0.997,ent=3e-4)"
        # all defaults

    run "$game" "IL=30" \
        training.imagine_length=30

    # ---- Group B: gamma (baseline already done) ----
    run "$game" "gamma=0.99" \
        agent.gamma=0.99

    run "$game" "gamma=0.999" \
        agent.gamma=0.999

    # ---- Group C: entropy_coef (baseline already done) ----
    run "$game" "ent=1e-4" \
        agent.entropy_coef=0.0001

    run "$game" "ent=1e-3" \
        agent.entropy_coef=0.001

done

GLOBAL_MINS=$(( (SECONDS - GLOBAL_START) / 60 ))
echo ""
echo "=========================================="
echo " All $TOTAL experiments finished in ${GLOBAL_MINS}min"
echo " SwanLab project: DreamerV3"
echo "=========================================="
