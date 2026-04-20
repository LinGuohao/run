#!/bin/bash
#
# 同时在多张 GPU 上运行不同 max_loop_count 的实验 - Qwen3.5-9B
# max_loop_count = 2, 3, 4, 5
# 使用服务器本地模型目录和本地数据集目录
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROXSPARSE_ROOT="${PROXSPARSE_ROOT:-${REPO_ROOT}/ProxSparse-main}"

echo "启动多GPU并行实验 - 不同max_loop_count值 (Qwen3.5-9B)"
echo "=================================================="

# 路径配置
STORAGE_ROOT="${STORAGE_ROOT:-/mnt/si002961ale4}"
DATA_HOME="${DATA_HOME:-${STORAGE_ROOT}/default/lgy/guohao}"

# main_genetic.py 的 --model 支持直接传本地模型目录
MODEL_PATH="${MODEL_PATH:-${DATA_HOME}/checkpoints/Qwen/Qwen3.5-9B}"
MODEL="${MODEL:-${MODEL_PATH}}"
MODEL_SHORT="${MODEL_SHORT:-qwen35_9b}"

# 直接指定本地数据集目录。若是 save_to_disk 导出的数据集目录，可将 DATASET_NAME 置空。
DATASET_PATH="${DATASET_PATH:-${DATA_HOME}/datasets/wikitext}"
DATASET_NAME="${DATASET_NAME:-wikitext-2-raw-v1}"

# 本地模型目录通常不需要缓存；如需从远端仓库名加载，可按需覆盖 CACHE_DIR。
CACHE_DIR="${CACHE_DIR:-}"

PYTHON_BIN="${PYTHON_BIN:-/usr/local/miniconda3/envs/proxsparse/bin/python}"

# 实验配置
POPULATION_SIZE=100
MAX_GENERATIONS=300
MUTATION_RATE=0.05
CROSSOVER_RATE=1.0
CROSSOVER_TYPE="twopoint"
SELECTION_METHOD="top20"
TOP_PERCENT=0.6
MAX_PARAM_RATIO=0.5
FROM_ELITE_PARAM_RATIO=0.8
USE_ELITE_POOL=false
EVAL_SAMPLES=15
CALIBRATION_SAMPLES=128
CTX_LEN=1024
SEED=42

# 精英池路径
FROM_ELITE_PCT=$(awk "BEGIN {print int(${FROM_ELITE_PARAM_RATIO} * 100)}")
ELITE_SEED_POOL_PATH="${ELITE_SEED_POOL_PATH:-elite_seed_pool_qwen35_9b_${FROM_ELITE_PCT}.json}"

# 并行配置
LOOP_COUNTS=(2 3 4 5)
GPU_DEVICES=("cuda:0" "cuda:1" "cuda:2" "cuda:3")

if [ "${#LOOP_COUNTS[@]}" -ne "${#GPU_DEVICES[@]}" ]; then
    echo "Error: LOOP_COUNTS and GPU_DEVICES length mismatch"
    exit 1
fi

if [ ! -d "${PROXSPARSE_ROOT}" ]; then
    echo "Error: ProxSparse root not found: ${PROXSPARSE_ROOT}"
    exit 1
fi

if [ ! -x "${PYTHON_BIN}" ]; then
    echo "Error: python executable not found: ${PYTHON_BIN}"
    exit 1
fi

if [[ "${MODEL_PATH}" = /* ]] && [ ! -d "${MODEL_PATH}" ]; then
    echo "Error: model path not found: ${MODEL_PATH}"
    exit 1
fi

if [[ "${DATASET_PATH}" = /* ]] && [ ! -e "${DATASET_PATH}" ]; then
    echo "Error: dataset path not found: ${DATASET_PATH}"
    exit 1
fi

if [ "${USE_ELITE_POOL}" = "true" ] && [[ "${ELITE_SEED_POOL_PATH}" != /* ]]; then
    ELITE_SEED_POOL_PATH="${PROXSPARSE_ROOT}/${ELITE_SEED_POOL_PATH}"
fi

PARAM_RATIO_PCT=$(awk "BEGIN {print int(${MAX_PARAM_RATIO} * 100)}")
LOG_DIR="${PROXSPARSE_ROOT}/logs/multi_loop_experiments_${MODEL_SHORT}_${PARAM_RATIO_PCT}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

echo "实验配置："
echo "  Model: ${MODEL}"
echo "  Model path: ${MODEL_PATH}"
echo "  Storage root: ${STORAGE_ROOT}"
echo "  Data home: ${DATA_HOME}"
echo "  ProxSparse root: ${PROXSPARSE_ROOT}"
echo "  Python: ${PYTHON_BIN}"
echo "  Dataset path: ${DATASET_PATH}"
echo "  Dataset config: ${DATASET_NAME:-<none>}"
echo "  Target param ratio: ${MAX_PARAM_RATIO} (${PARAM_RATIO_PCT}%)"
echo "  Elite pool from param ratio: ${FROM_ELITE_PARAM_RATIO} (${FROM_ELITE_PCT}%)"
echo "  Elite pool file: ${ELITE_SEED_POOL_PATH}"
echo "  Loop counts: ${LOOP_COUNTS[*]}"
echo "  GPU devices: ${GPU_DEVICES[*]}"
echo "  Log directory: ${LOG_DIR}"
echo ""

for i in "${!LOOP_COUNTS[@]}"; do
    LOOP_COUNT=${LOOP_COUNTS[$i]}
    GPU_DEVICE=${GPU_DEVICES[$i]}
    TOP_PERCENT_INT=$(awk "BEGIN {print int(${TOP_PERCENT} * 100)}")

    OUTPUT_DIR="Qwen3.5-9B-genetic_loop${LOOP_COUNT}_p${POPULATION_SIZE}_g${MAX_GENERATIONS}_${CROSSOVER_TYPE}_${SELECTION_METHOD}${TOP_PERCENT_INT}_ratio${MAX_PARAM_RATIO}"
    LOG_FILE="${LOG_DIR}/genetic_Qwen3.5-9B_loop${LOOP_COUNT}_p${POPULATION_SIZE}_g${MAX_GENERATIONS}_${CROSSOVER_TYPE}_${SELECTION_METHOD}${TOP_PERCENT_INT}_sparse${PARAM_RATIO_PCT}_$(date +%Y%m%d_%H%M%S).log"

    echo "启动实验 $((i+1)): max_loop_count=${LOOP_COUNT}, GPU=${GPU_DEVICE}"
    echo "  输出目录: ${OUTPUT_DIR}"
    echo "  日志文件: ${LOG_FILE}"
    echo ""

    {
        echo "============================================================================"
        echo "Genetic Algorithm Pruning - Loop Count ${LOOP_COUNT} (Qwen3.5-9B)"
        echo "  GPU: ${GPU_DEVICE}"
        echo "  Target: ${MAX_PARAM_RATIO} param ratio (${PARAM_RATIO_PCT}%)"
        echo "  Elite from: ${FROM_ELITE_PARAM_RATIO} param ratio (${FROM_ELITE_PCT}%) - ${ELITE_SEED_POOL_PATH}"
        echo "  Started at: $(date)"
        echo "============================================================================"
        echo ""

        cd "${PROXSPARSE_ROOT}"

        CHECKPOINT_FILE="${OUTPUT_DIR}/checkpoints/checkpoint_latest.json"
        CMD=(
            "${PYTHON_BIN}" end-to-end/main_genetic.py
            --model "${MODEL}"
            --dataset_path "${DATASET_PATH}"
            --population_size "${POPULATION_SIZE}"
            --max_generations "${MAX_GENERATIONS}"
            --mutation_rate "${MUTATION_RATE}"
            --crossover_rate "${CROSSOVER_RATE}"
            --crossover_type "${CROSSOVER_TYPE}"
            --selection_method "${SELECTION_METHOD}"
            --top_percent "${TOP_PERCENT}"
            --max_param_ratio "${MAX_PARAM_RATIO}"
            --max_loop_count "${LOOP_COUNT}"
            --use_elite_pool "${USE_ELITE_POOL}"
            --eval_samples "${EVAL_SAMPLES}"
            --calibration_samples "${CALIBRATION_SAMPLES}"
            --ctx_len "${CTX_LEN}"
            --device "${GPU_DEVICE}"
            --seed "${SEED}"
            --output_dir "${OUTPUT_DIR}"
            --checkpoint_interval 10
        )

        if [ -n "${DATASET_NAME}" ]; then
            CMD+=(--dataset_name "${DATASET_NAME}")
        fi

        if [ -n "${CACHE_DIR}" ]; then
            CMD+=(--cache_dir "${CACHE_DIR}")
        fi

        #if [ -f "${CHECKPOINT_FILE}" ]; then
        #    CMD+=(--resume_from "${CHECKPOINT_FILE}")
        #    echo "Found existing checkpoint, resuming from: ${CHECKPOINT_FILE}"
        #else
        #    echo "Starting new experiment"
        #fi

        if [ "${USE_ELITE_POOL}" = "true" ]; then
            CMD+=(--elite_seed_pool_path "${ELITE_SEED_POOL_PATH}")
        fi

        printf 'Running command:'
        printf ' %q' "${CMD[@]}"
        echo

        "${CMD[@]}" 2>&1 | tee "${LOG_FILE}"

        echo ""
        echo "============================================================================"
        echo "Loop Count ${LOOP_COUNT} Experiment Completed"
        echo "  Finished at: $(date)"
        echo "  Best model saved to: ${OUTPUT_DIR}"
        echo "  Log saved to: ${LOG_FILE}"
        echo "============================================================================"

    } &

    echo $! >> "${LOG_DIR}/pids_${LOOP_COUNT}.txt"
    sleep 2
done

echo ""
echo "所有实验已启动"
echo "  模型: ${MODEL}"
echo "  目标参数比例: ${MAX_PARAM_RATIO} (${PARAM_RATIO_PCT}%)"
echo "  主日志目录: ${LOG_DIR}"
echo "  进程ID文件: ${LOG_DIR}/pids_*.txt"
echo ""
echo "实验状态："
for i in "${!LOOP_COUNTS[@]}"; do
    LOOP_COUNT=${LOOP_COUNTS[$i]}
    GPU_DEVICE=${GPU_DEVICES[$i]}
    echo "  Loop ${LOOP_COUNT}: GPU=${GPU_DEVICE}, PID=$(cat "${LOG_DIR}/pids_${LOOP_COUNT}.txt" 2>/dev/null || echo 'N/A')"
done

echo ""
echo "查看实验进度："
echo "  tail -f ${LOG_DIR}/*.log"
echo ""
echo "停止所有实验："
echo "  kill \$(cat ${LOG_DIR}/pids_*.txt 2>/dev/null | tr '\\n' ' ')"
echo ""
echo "============================================================================"
