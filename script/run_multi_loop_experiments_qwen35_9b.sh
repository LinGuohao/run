#!/bin/bash
#
# 同时在4张GPU上运行不同max_loop_count的实验 - Qwen2.5-7B
# max_loop_count = 2, 3, 4, 5
# 目标稀疏度50%（参数保留率50%）
# 从Qwen2.5-14B的50-90%精英池迁移学习
#

set -e

echo "🚀 启动多GPU并行实验 - 不同max_loop_count值 (Qwen3.5-9B)"
echo "=================================================="

# 基础配置
MODEL="Qwen/Qwen3.5-9B"
MODEL_SHORT="qwen35_9b"
CACHE_DIR="/hy-tmp/huggingface/hub"
POPULATION_SIZE=100
MAX_GENERATIONS=300
MUTATION_RATE=0.05
CROSSOVER_RATE=1.0
CROSSOVER_TYPE="twopoint"
SELECTION_METHOD="top20"
TOP_PERCENT=0.6
MAX_PARAM_RATIO=0.5  # 目标稀疏度50%（参数保留率50%）
FROM_ELITE_PARAM_RATIO=0.8  # 精英池来源参数比例（可以与目标比例不同）
USE_ELITE_POOL=false
EVAL_SAMPLES=15
CALIBRATION_SAMPLES=128
CTX_LEN=1024
SEED=42

# 精英池路径 - 动态根据FROM_ELITE_PARAM_RATIO确定
# 例如：FROM_ELITE_PARAM_RATIO=0.5 -> elite_seed_pool_qwen25_7b_50.json
# 从第一轮300代实验中提取的Top 60精英个体
# 染色体长度：56（7B模型）
FROM_ELITE_PCT=$(awk "BEGIN {print int(${FROM_ELITE_PARAM_RATIO} * 100)}")
ELITE_SEED_POOL_PATH="elite_seed_pool_qwen35_9b_${FROM_ELITE_T360-84UU-X8X9-1LKKPCT}.json"

# 实验配置 - 4个不同的max_loop_count值
LOOP_COUNTS=(2 3 4 5)
GPU_DEVICES=("cuda:0" "cuda:1" "cuda:2" "cuda:3")

# 创建日志目录（包含模型和参数比例信息）
PARAM_RATIO_PCT=$(awk "BEGIN {print int(${MAX_PARAM_RATIO} * 100)}")
LOG_DIR="logs/multi_loop_experiments_${MODEL_SHORT}_${PARAM_RATIO_PCT}_$(date +%Y%m%d_%H%M%S)"
mkdir -p ${LOG_DIR}

echo "实验配置："
echo "  Model: ${MODEL}"
echo "  Target param ratio: ${MAX_PARAM_RATIO} (${PARAM_RATIO_PCT}%)"
echo "  Elite pool from param ratio: ${FROM_ELITE_PARAM_RATIO} (${FROM_ELITE_PCT}%)"
echo "  Elite pool file: ${ELITE_SEED_POOL_PATH}"
echo "  Loop counts: ${LOOP_COUNTS[@]}"
echo "  GPU devices: ${GPU_DEVICES[@]}"
echo "  Log directory: ${LOG_DIR}"
echo ""

# 启动4个并行实验
for i in {0..3}; do
    LOOP_COUNT=${LOOP_COUNTS[$i]}
    GPU_DEVICE=${GPU_DEVICES[$i]}

    # 计算top百分比整数
    TOP_PERCENT_INT=$(awk "BEGIN {print int(${TOP_PERCENT} * 100)}")

    # 生成输出目录名
    OUTPUT_DIR="Qwen3.5-9B-genetic_loop${LOOP_COUNT}_p${POPULATION_SIZE}_g${MAX_GENERATIONS}_${CROSSOVER_TYPE}_${SELECTION_METHOD}${TOP_PERCENT_INT}_ratio${MAX_PARAM_RATIO}"

    # 生成日志文件名
    LOG_FILE="${LOG_DIR}/genetic_Qwen3.5-9B_loop${LOOP_COUNT}_p${POPULATION_SIZE}_g${MAX_GENERATIONS}_${CROSSOVER_TYPE}_${SELECTION_METHOD}${TOP_PERCENT_INT}_sparse${PARAM_RATIO_PCT}_$(date +%Y%m%d_%H%M%S).log"


    echo "✨ 启动实验 $((i+1)): max_loop_count=${LOOP_COUNT}, GPU=${GPU_DEVICE}"
    echo "   输出目录: ${OUTPUT_DIR}"
    echo "   日志文件: ${LOG_FILE}"
    echo ""

    # 启动后台进程
    {
        echo "============================================================================"
        echo "🧬 Genetic Algorithm Pruning - Loop Count ${LOOP_COUNT} (Qwen3.5-2B)"
        echo "   GPU: ${GPU_DEVICE}"
        echo "   Target: ${MAX_PARAM_RATIO} param ratio (${PARAM_RATIO_PCT}%)"
        echo "   Elite from: ${FROM_ELITE_PARAM_RATIO} param ratio (${FROM_ELITE_PCT}%) - ${ELITE_SEED_POOL_PATH}"
        echo "   Started at: $(date)"
        echo "============================================================================"
        echo ""

        cd /hy-tmp/ProxSparse-main

        # 检查是否存在checkpoint，如果存在则resume
        CHECKPOINT_FILE="${OUTPUT_DIR}/checkpoints/checkpoint_latest.json"
        RESUME_ARG=""
        #if [ -f "${CHECKPOINT_FILE}" ]; then
        #    RESUME_ARG="--resume_from ${CHECKPOINT_FILE}"
        #    echo "📌 Found existing checkpoint, resuming from: ${CHECKPOINT_FILE}"
        #else
        #    echo "🆕 Starting new experiment"
        #fi

        /usr/local/miniconda3/envs/proxsparse/bin/python end-to-end/main_genetic.py \
            --model ${MODEL} \
            --cache_dir ${CACHE_DIR} \
            --population_size ${POPULATION_SIZE} \
            --max_generations ${MAX_GENERATIONS} \
            --mutation_rate ${MUTATION_RATE} \
            --crossover_rate ${CROSSOVER_RATE} \
            --crossover_type ${CROSSOVER_TYPE} \
            --selection_method ${SELECTION_METHOD} \
            --top_percent ${TOP_PERCENT} \
            --max_param_ratio ${MAX_PARAM_RATIO} \
            --max_loop_count ${LOOP_COUNT} \
            --use_elite_pool ${USE_ELITE_POOL} \
            --elite_seed_pool_path ${ELITE_SEED_POOL_PATH} \
            --eval_samples ${EVAL_SAMPLES} \
            --calibration_samples ${CALIBRATION_SAMPLES} \
            --ctx_len ${CTX_LEN} \
            --device ${GPU_DEVICE} \
            --seed ${SEED} \
            --output_dir ${OUTPUT_DIR} \
            --checkpoint_interval 10 \
            ${RESUME_ARG} \
            2>&1 | tee ${LOG_FILE}

        echo ""
        echo "============================================================================"
        echo "✅ Loop Count ${LOOP_COUNT} Experiment Completed!"
        echo "   Finished at: $(date)"
        echo "   Best model saved to: ${OUTPUT_DIR}"
        echo "   Log saved to: ${LOG_FILE}"
        echo "============================================================================"

    } &

    # 记录进程ID
    echo $! >> ${LOG_DIR}/pids_${LOOP_COUNT}.txt

    # 稍微延迟，避免同时启动造成资源冲突
    sleep 2
done

echo ""
echo "🎯 所有实验已启动！"
echo "   模型: ${MODEL}"
echo "   目标参数比例: ${MAX_PARAM_RATIO} (${PARAM_RATIO_PCT}%)"
echo "   主日志目录: ${LOG_DIR}"
echo "   进程ID文件: ${LOG_DIR}/pids_*.txt"
echo ""
echo "📊 实验状态："
for i in {0..3}; do
    LOOP_COUNT=${LOOP_COUNTS[$i]}
    GPU_DEVICE=${GPU_DEVICES[$i]}
    echo "   Loop ${LOOP_COUNT}: GPU=${GPU_DEVICE}, PID=$(cat ${LOG_DIR}/pids_${LOOP_COUNT}.txt 2>/dev/null || echo 'N/A')"
done

echo ""
echo "📝 查看实验进度："
echo "   tail -f ${LOG_DIR}/*.log"
echo ""
echo "🛑 停止所有实验："
echo "   kill $(cat ${LOG_DIR}/pids_*.txt 2>/dev/null | tr '\\n' ' ')"
echo ""
echo "============================================================================"
