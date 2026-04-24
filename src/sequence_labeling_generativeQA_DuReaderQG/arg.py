"""
生成式问答模型（T5-Base + DuReaderQG）- 参数配置模块
"""
import argparse
import json
from pathlib import Path


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='生成式问答模型训练脚本')

    # ============ 数据路径 ============
    parser.add_argument('--data_dir', type=str, default='../../data/DuReaderQG/',
                        help='数据集目录路径')
    parser.add_argument('--train_file', type=str, default='train.json',
                        help='训练数据文件名（相对于data_dir）')
    parser.add_argument('--dev_file', type=str, default='dev.json',
                        help='验证数据文件名（相对于data_dir）')
    parser.add_argument('--output_dir', type=str, default='./qa_model_output/',
                        help='输出目录，保存模型和结果')

    # ============ 模型配置 ============
    parser.add_argument('--model_choice', type=str, default='mengzi-t5-base',
                        choices=['google-t5-base', 'google-t5-small', 'mengzi-t5-base',
                                'mt5-base', 'mt5-small', 'custom'],
                        help='选择预训练模型（默认: Mengzi-T5-Base 中文模型）')
    parser.add_argument('--model_name', type=str, default=None,
                        help='自定义模型路径或HuggingFace模型ID（当model_choice为custom时使用）')
    parser.add_argument('--tokenizer_name', type=str, default=None,
                        help='自定义tokenizer路径（默认与model_name相同）')

    # ============ 输入长度 ============
    parser.add_argument('--max_input_length', type=int, default=512,
                        help='输入序列最大长度')
    parser.add_argument('--max_target_length', type=int, default=64,
                        help='目标序列最大长度')

    # ============ 训练超参数 ============
    parser.add_argument('--num_epochs', type=int, default=3,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='训练批大小')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='权重衰减系数')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                        help='预热步数占比')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='梯度裁剪最大范数')

    # ============ 生成参数 ============
    parser.add_argument('--num_beams', type=int, default=4,
                        help='Beam Search宽度')
    parser.add_argument('--max_gen_length', type=int, default=64,
                        help='生成答案最大长度')
    parser.add_argument('--temperature', type=float, default=0.9,
                        help='生成温度（控制多样性）')
    parser.add_argument('--top_p', type=float, default=0.95,
                        help='Top-p采样的p值')

    # ============ 其他配置 ============
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--device', type=str, default=None,
                        choices=['cuda', 'mps', 'cpu'],
                        help='运行设备（None表示自动检测）')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='数据加载进程数')

    # ============ 执行模式 ============
    parser.add_argument('--do_train', action='store_true',
                        help='是否执行训练')
    parser.add_argument('--do_eval', action='store_true',
                        help='是否执行评估')
    parser.add_argument('--do_predict', action='store_true',
                        help='是否执行预测')
    parser.add_argument('--do_all', action='store_true',
                        help='执行所有阶段（训练+评估+预测）')

    # ============ 保存和加载 ============
    parser.add_argument('--save_best_only', action='store_true', default=True,
                        help='仅保存最佳模型')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                        help='从检查点恢复训练')

    args = parser.parse_args()

    # 如果指定了--do_all，则设置所有标志
    if args.do_all:
        args.do_train = True
        args.do_eval = True
        args.do_predict = True

    # 如果都没有指定，默认执行训练
    if not (args.do_train or args.do_eval or args.do_predict):
        args.do_train = True
        args.do_eval = True

    return args


def get_model_config(model_choice):
    """
    获取模型配置信息

    Args:
        model_choice: 模型选择标识符

    Returns:
        dict: 包含model_name和tokenizer_name的配置字典
    """
    model_config = {
        'google-t5-base': {
            'model_name': 'google-t5/t5-base',
            'tokenizer_name': 'google-t5/t5-base',
            'description': 'Google T5-Base (220M 参数，英文)'
        },
        'google-t5-small': {
            'model_name': 'google-t5/t5-small',
            'tokenizer_name': 'google-t5/t5-small',
            'description': 'Google T5-Small (60M 参数，英文，更快)'
        },
        'mengzi-t5-base': {
            'model_name': 'Langboat/mengzi-t5-base',
            'tokenizer_name': 'Langboat/mengzi-t5-base',
            'description': 'Mengzi T5-Base (中文，推荐)'
        },
        'mt5-base': {
            'model_name': 'google/mt5-base',
            'tokenizer_name': 'google/mt5-base',
            'description': 'mT5-Base (多语言，包含中文)'
        },
        'mt5-small': {
            'model_name': 'google/mt5-small',
            'tokenizer_name': 'google/mt5-small',
            'description': 'mT5-Small (多语言，快速)'
        }
    }

    return model_config.get(model_choice, None)


def setup_args_and_paths(args):
    """
    设置参数和路径

    Args:
        args: 解析的命令行参数

    Returns:
        args: 修改后的参数对象
    """
    # 设置数据文件完整路径
    if not Path(args.train_file).is_absolute():
        args.train_file = str(Path(args.data_dir) / args.train_file)
    if not Path(args.dev_file).is_absolute():
        args.dev_file = str(Path(args.data_dir) / args.dev_file)

    # 创建输出目录
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # 设置模型和tokenizer
    if args.model_choice == 'custom' and args.model_name:
        # 自定义模型
        args.tokenizer_name = args.tokenizer_name or args.model_name
    else:
        # 预定义模型
        config = get_model_config(args.model_choice)
        if config:
            args.model_name = config['model_name']
            args.tokenizer_name = config['tokenizer_name']
        else:
            raise ValueError(f"Unknown model choice: {args.model_choice}")

    return args


if __name__ == '__main__':
    args = parse_args()
    print("命令行参数:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")