"""
===============================================================================
参数解析模块：抽取式阅读理解任务的超参数和配置
===============================================================================

【任务背景】
抽取式阅读理解（Extractive QA）：给定文章和问题，从文章中提取答案的起始和结束位置。
本任务使用 CMRC 2018（中文阅读理解）数据集。

【核心参数说明】
  - max_length: 输入长度限制（question + context）
  - max_answer_length: 答案最多包含多少个 token（防止太长的预测）
  - stride: 处理超长文本的关键参数，相邻分割块的重叠 token 数
  - n_best: 推理时保留前 n 个最高概率的答案候选

这不是 sequence classification，而是 token classification：
对序列中的每个位置都要预测"是否是答案的开始 / 结束"。
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
        help="输出目录，保存模型检查点和预测结果。",
    )
    parser.add_argument("--train_file", default=None, type=str, required=True,
        help="训练数据文件路径（SQuAD 或 CMRC 格式的 JSON）。")
    parser.add_argument("--dev_file", default=None, type=str, required=True,
        help="验证数据文件路径（训练时用来选择最佳模型）。")
    parser.add_argument("--test_file", default=None, type=str, required=True,
        help="测试数据文件路径（最终评估和预测）。")

    # ============ 模型相关参数 ============
    parser.add_argument("--model_type", default="bert", type=str, required=True,
        help="模型类型标识（如 'bert'，仅用于日志记录）。"
    )
    parser.add_argument("--model_checkpoint", default="bert-large-cased/", type=str, required=True,
        help="预训练模型路径或 HuggingFace Hub 模型标识（如 'bert-base-chinese'）。"
    )

    # ============ 输入长度相关参数 ============
    parser.add_argument("--max_length", default=512, type=int, required=True,
        help="输入序列的最大长度（question + [SEP] + context）。超出则截断。"
    )
    parser.add_argument("--max_answer_length", default=50, type=int, required=True,
        help="答案的最大长度（token 数）。推理时超过此长度的答案会被过滤掉。"
    )

    # ============ 三种运行模式 ============
    parser.add_argument("--do_train", action="store_true",
        help="是否执行训练阶段（在验证集上选最佳权重）。")
    parser.add_argument("--do_test", action="store_true",
        help="是否在测试集上评估（计算 F1 和 EM）。")
    parser.add_argument("--do_predict", action="store_true",
        help="是否对测试集进行预测并保存结果。")

    # ============ 处理超长文本的参数（Extractive QA 特有） ============
    # 【关键概念】长文本处理：当文章太长时，不能直接截断（会丢失答案）。
    # 解决方案：将文本分割成多个块，相邻块有重叠，推理时合并所有块的预测。
    parser.add_argument("--stride", default=128, type=int,
        help="处理超长文本时，相邻分割块之间的重叠 token 数。"
             "例如：max_length=512, stride=128 → 每个 chunk 之间重叠 128 个 token。"
             "stride 越大，chunks 越多，计算量越大；但对于长范围的答案更友好。"
    )
    parser.add_argument("--n_best", default=20, type=int,
        help="推理时，对每个 chunk 保留前 n_best 个最高概率的答案起点和终点候选。"
             "然后枚举这些候选的组合，选择置信度最高的作为最终答案。"
             "n_best 越大，精度可能越高，但计算开销也越大。"
    )

    # ============ 训练超参数 ============
    parser.add_argument("--learning_rate", default=1e-5, type=float,
        help="Adam 优化器的初始学习率（BERT 微调常用 1e-5 ~ 5e-5）。")
    parser.add_argument("--num_train_epochs", default=3, type=int,
        help="总的训练轮数（Extractive QA 通常 2~3 个 epoch）。")
    parser.add_argument("--batch_size", default=4, type=int,
        help="批次大小。显存不足可调小（含长文本的 batch 对显存要求较高）。")
    parser.add_argument("--seed", type=int, default=42,
        help="随机种子，用于结果复现。")

    # ============ Adam 优化器参数 ============
    parser.add_argument("--adam_beta1", default=0.9, type=float,
        help="Adam 优化器的 β1 参数（一阶矩估计的指数衰减率）。"
    )
    parser.add_argument("--adam_beta2", default=0.98, type=float,
        help="Adam 优化器的 β2 参数（二阶矩估计的指数衰减率）。"
    )
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
        help="Adam 优化器的 epsilon 值（数值稳定性）。"
    )

    # ============ 学习率调度参数 ============
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
        help="学习率预热阶段占总步数的比例（例如 0.1 = 前 10% 步数线性升温）。"
    )

    # ============ 正则化参数 ============
    parser.add_argument("--weight_decay", default=0.01, type=float,
        help="权重衰减系数（L2 正则化），作用在除 bias 和 LayerNorm.weight 外的参数。"
    )

    args = parser.parse_args()
    return args
