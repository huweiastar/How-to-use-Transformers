#!/bin/bash

# 生成式问答模型（T5-Base + DuReaderQG）- 训练脚本

# ============ 基础配置 ============
OUTPUT_DIR="./qa_model_output"
DATA_DIR="../../data/DuReaderQG/"

# ============ 模型配置 ============
# 选项: mengzi-t5-base, google-t5-base, google-t5-small, mt5-base, mt5-small, custom
MODEL_CHOICE="mengzi-t5-base"

# ============ 训练配置 ============
NUM_EPOCHS=3
BATCH_SIZE=8
LEARNING_RATE=2e-5
WARMUP_RATIO=0.1

# ============ 其他配置 ============
SEED=42
NUM_WORKERS=0

echo "==========================================="
echo "生成式问答模型训练"
echo "==========================================="
echo ""
echo "配置信息:"
echo "  数据目录: $DATA_DIR"
echo "  输出目录: $OUTPUT_DIR"
echo "  模型选择: $MODEL_CHOICE"
echo "  训练轮数: $NUM_EPOCHS"
echo "  批大小: $BATCH_SIZE"
echo "  学习率: $LEARNING_RATE"
echo ""

# ============ 运行训练 ============
python run_generativeQA.py \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --model_choice $MODEL_CHOICE \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --warmup_ratio $WARMUP_RATIO \
    --seed $SEED \
    --num_workers $NUM_WORKERS \
    --do_all

echo ""
echo "==========================================="
echo "✅ 训练完成！"
echo "==========================================="