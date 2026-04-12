"""
命令行参数解析模块
用于解析运行训练、测试和预测所需的所有超参数和文件路径
"""
import argparse

def parse_args():
    """
    解析命令行参数

    返回值:
        args: argparse.Namespace 对象，包含所有解析的参数
    """
    parser = argparse.ArgumentParser()

    # ============ 必需的参数 ============
    # 输出目录相关参数
    parser.add_argument("--output_dir", default=None, type=str, required=True,
        help="输出目录，用于保存模型检查点和预测结果",
    )

    # 数据文件路径参数
    parser.add_argument("--train_file", default=None, type=str, required=True,
        help="训练数据文件路径")
    parser.add_argument("--dev_file", default=None, type=str, required=True,
        help="验证数据文件路径")
    parser.add_argument("--test_file", default=None, type=str, required=True,
        help="测试数据文件路径")

    # 模型相关参数
    parser.add_argument("--model_type", default="bert", type=str, required=True,
        help="预训练模型类型，如'bert'或'roberta'"
    )
    parser.add_argument("--model_checkpoint", default="bert-large-cased/", type=str, required=True,
        help="预训练模型的路径或在HuggingFace上的模型标识符",
    )
    parser.add_argument("--max_seq_length", default=512, type=int, required=True,
        help="输入序列的最大长度"
    )

    # ============ 执行模式参数 ============
    parser.add_argument("--do_train", action="store_true",
        help="是否执行训练阶段")
    parser.add_argument("--do_test", action="store_true",
        help="是否在测试集上进行评估")
    parser.add_argument("--do_predict", action="store_true",
        help="是否保存预测标签")

    # ============ 训练超参数 ============
    parser.add_argument("--learning_rate", default=1e-5, type=float,
        help="Adam优化器的初始学习率")
    parser.add_argument("--num_train_epochs", default=3, type=int,
        help="总的训练轮数")
    parser.add_argument("--batch_size", default=4, type=int,
        help="批次大小")
    parser.add_argument("--seed", type=int, default=42,
        help="随机种子，用于结果复现")

    # ============ Adam优化器相关参数 ============
    parser.add_argument("--adam_beta1", default=0.9, type=float,
        help="Adam优化器的β1参数（一阶矩估计的指数衰减率）"
    )
    parser.add_argument("--adam_beta2", default=0.98, type=float,
        help="Adam优化器的β2参数（二阶矩估计的指数衰减率）"
    )
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
        help="Adam优化器的epsilon值，用于数值稳定性"
    )

    # ============ 学习率调度参数 ============
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
        help="学习率预热阶段占总训练步数的比例，例如0.1表示前10%的步数进行线性预热"
    )

    # ============ 正则化参数 ============
    parser.add_argument("--weight_decay", default=0.01, type=float,
        help="权重衰减值（L2正则化系数）"
    )

    args = parser.parse_args()
    return args
