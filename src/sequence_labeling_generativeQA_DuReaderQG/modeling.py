"""
生成式问答模型（T5-Base + DuReaderQG）- 模型加载模块
"""
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer


def get_device(device_str=None):
    """
    获取计算设备

    Args:
        device_str: 设备字符串（'cuda'、'mps'、'cpu'或None自动检测）

    Returns:
        torch.device: PyTorch设备对象
    """
    if device_str:
        return torch.device(device_str)

    # 自动检测最优设备
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        device_name = "MPS (Apple Silicon)"
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = f"CUDA ({torch.cuda.get_device_name(0)})"
    else:
        device = torch.device("cpu")
        device_name = "CPU"

    print(f"✓ 检测到设备: {device_name}")
    return device


def load_tokenizer(tokenizer_name):
    """
    加载分词器

    Args:
        tokenizer_name: 分词器标识符或本地路径

    Returns:
        tokenizer: T5Tokenizer实例
    """
    print(f"加载分词器: {tokenizer_name}...")
    tokenizer = T5Tokenizer.from_pretrained(tokenizer_name, model_max_length=512)
    print(f"✓ 分词器加载成功")
    return tokenizer


def load_model(model_name, device):
    """
    加载T5模型

    Args:
        model_name: 模型标识符或本地路径
        device: 计算设备

    Returns:
        model: T5ForConditionalGeneration实例
    """
    print(f"加载模型: {model_name}...")
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model = model.to(device)

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"✓ 模型加载成功")
    print(f"  总参数量: {total_params / 1e6:.1f}M")
    print(f"  可训练参数量: {trainable_params / 1e6:.1f}M")
    print(f"  设备: {next(model.parameters()).device}")

    return model


def setup_optimizer_and_scheduler(model, args, num_training_steps):
    """
    设置优化器和学习率调度器

    Args:
        model: T5模型
        args: 参数对象
        num_training_steps: 总训练步数

    Returns:
        optimizer, scheduler: 优化器和调度器实例
    """
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup

    # 优化器
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # 学习率调度器
    warmup_steps = int(num_training_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps
    )

    print(f"优化器: AdamW")
    print(f"  学习率: {args.learning_rate}")
    print(f"  权重衰减: {args.weight_decay}")
    print(f"\n学习率调度器: Linear with Warmup")
    print(f"  总步数: {num_training_steps}")
    print(f"  预热步数: {warmup_steps}")

    return optimizer, scheduler