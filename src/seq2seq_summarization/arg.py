"""
===============================================================================
参数解析模块：用于文本摘要任务的所有超参数和路径配置
===============================================================================

主要参数说明：
  - 路径类：数据文件、模型、输出目录
  - 长度类：输入文本最大长度、摘要最大长度（Seq2Seq 特有）
  - 生成类：beam search 宽度、n-gram 重复限制（推理解码时的关键参数）
  - 优化类：学习率、warmup、权重衰减等训练超参数
"""
import argparse

def parse_args():
    """
    解析命令行参数

    返回值:
        args: argparse.Namespace 对象，包含所有参数
    """
    parser = argparse.ArgumentParser()

    # ============ 必需参数：数据和模型路径 ============
    parser.add_argument("--output_dir", default=None, type=str, required=True,
        help="输出目录，保存模型检查点和预测结果。训练前必须指定。",
    )
    parser.add_argument("--train_file", default=None, type=str, required=True,
        help="训练数据文件路径（格式：每行一个JSON或制表符分隔的 content\\ttitle）。")
    parser.add_argument("--dev_file", default=None, type=str, required=True,
        help="验证数据文件路径（训练时用来选择最佳模型）。")
    parser.add_argument("--test_file", default=None, type=str, required=True,
        help="测试数据文件路径（最终评估和预测时使用）。")

    # ============ 模型相关参数 ============
    parser.add_argument("--model_type", default="bert", type=str, required=True,
        help="模型类型标识（如 'mt5', 'bart'，仅用于日志记录）。"
    )
    parser.add_argument("--model_checkpoint", default="bert-large-cased/", type=str, required=True,
        help="预训练模型路径或 HuggingFace Hub 模型标识（如 'google/mt5-small'）。"
    )

    # ============ Seq2Seq 长度限制 ============
    # 【初学者注意】这两个参数只作用于 tokenizer，实际生成长度由 model.generate() 的 max_length 控制
    parser.add_argument("--max_input_length", default=256, type=int, required=True,
        help="输入（原文）的最大token数。超出时会被截断。"
    )
    parser.add_argument("--max_target_length", default=256, type=int, required=True,
        help="目标（摘要）的最大token数。用于标签处理和生成时的长度限制。"
    )

    # ============ 三种运行模式：train / test / predict ============
    parser.add_argument("--do_train", action="store_true",
        help="是否执行训练阶段（会在验证集上选择最佳权重）。")
    parser.add_argument("--do_test", action="store_true",
        help="是否在测试集上评估所有保存的权重。")
    parser.add_argument("--do_predict", action="store_true",
        help="是否对测试集进行预测并保存生成的摘要到 JSON 文件。")

    # ============ 训练超参数 ============
    parser.add_argument("--learning_rate", default=1e-5, type=float,
        help="Adam 优化器的初始学习率（Seq2Seq 微调常用 1e-5 ~ 5e-5）。")
    parser.add_argument("--num_train_epochs", default=3, type=int,
        help="总的训练轮数。Seq2Seq 通常 2~3 个 epoch 就收敛。")
    parser.add_argument("--batch_size", default=4, type=int,
        help="批次大小。显存不足可调小（文本摘要对显存要求较高）。")
    parser.add_argument("--seed", type=int, default=42,
        help="随机种子，用于结果复现（固定所有随机源）。")

    # ============ Seq2Seq 生成阶段的参数 ============
    # 这些参数在 model.generate() 时起作用，控制摘要的生成质量
    parser.add_argument("--beam_search_size", default=4, type=int,
        help="Beam Search 的宽度（如 4 = 每步保留 4 个最优候选）。"
                 "值越大生成质量越好但速度越慢，通常 3~5。"
    )
    parser.add_argument("--no_repeat_ngram_size", default=2, type=int,
        help="禁止 n-gram 重复的 n 值（如 2 = 不允许 2-gram 重复）。"
                 "避免生成 \"这是这是这是\" 这样的重复文本，取值通常 2~3。"
    )

    # ============ Adam 优化器参数 ============
    parser.add_argument("--adam_beta1", default=0.9, type=float,
        help="Adam 优化器的 β1 参数（一阶矩估计的指数衰减率）。默认 0.9 通常不需要改。"
    )
    parser.add_argument("--adam_beta2", default=0.98, type=float,
        help="Adam 优化器的 β2 参数（二阶矩估计的指数衰减率）。微调时 0.98 比 0.999 更稳定。"
    )
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
        help="Adam 优化器的 epsilon 值（数值稳定性，防止除以 0）。通常不需要改。"
    )

    # ============ 学习率调度参数 ============
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
        help="学习率预热阶段占总步数的比例（如 0.1 = 前 10% 步数线性升到 lr）。"
                 "预热可以稳定训练初期，通常设置 0.05 ~ 0.1。"
    )

    # ============ 正则化参数 ============
    parser.add_argument("--weight_decay", default=0.01, type=float,
        help="权重衰减系数（L2 正则化），作用在除 bias 和 LayerNorm.weight 外的参数。"
                 "值越大正则化越强，通常 0.01 ~ 0.1。"
    )

    args = parser.parse_args()
    return args
