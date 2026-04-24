"""
===============================================================================
数据加载模块：Seq2Seq 文本摘要的数据处理
===============================================================================

核心概念（初学者必读）：
  1. LCSTS 数据集格式：每行一个样本，content 是原文，title 是摘要摘要
  2. Seq2Seq 的特殊处理：
     - encoder 处理 input（原文）
     - decoder 处理 target（摘要）
     - decoder_input_ids：目标序列向左移一位（因为 decoder 边生成边消耗）
     - labels：只在 EOS 令牌后的位置计算损失（-100 表示忽略）

思考一下：为什么需要这么多 special handling？因为 Seq2Seq 训练时 decoder 是
"teacher forcing" 模式：输入前一个词，预测当前词。而评估/推理时是自回归
的：上一步的输出做下一步的输入。这种不匹配需要在数据处理层解决。
"""
from torch.utils.data import Dataset, DataLoader
import torch

# 数据集最大样本数（防止数据太大撑爆内存，可按需调整）
MAX_DATASET_SIZE = 200000

class LCSTS(Dataset):
    """
    LCSTS 数据集加载器

    数据格式（两种都支持）：
      1. 分隔符分隔：content != ! title （content 和 title 用 "!=" 分隔）
      2. 每行一个 JSON：{"content": "...", "title": "..."}

    这里使用分隔符格式。目录下的 data 文件通常就是这种格式。
    """

    def __init__(self, data_file):
        """
        参数:
            data_file (str): 数据文件路径
        """
        self.data = self.load_data(data_file)

    def load_data(self, data_file):
        """
        从文件按行加载数据

        参数:
            data_file (str): 文件路径

        返回值:
            dict: {样本索引: {"content": 原文, "title": 摘要}}
        """
        Data = {}
        with open(data_file, 'rt', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                # 防止数据太大
                if idx >= MAX_DATASET_SIZE:
                    break

                # 按 "!=" 分隔符拆分成两部分
                items = line.strip().split('!=!')
                assert len(items) == 2, f"行 {idx} 格式错误，不含两个分隔符"

                # 第一部分是 title（摘要），第二部分是 content（原文）
                # 注意顺序：数据文件中摘要在前、原文在后
                Data[idx] = {
                    'title': items[0],    # 目标摘要
                    'content': items[1]   # 输入原文
                }
        return Data

    def __len__(self):
        """返回数据集大小"""
        return len(self.data)

    def __getitem__(self, idx):
        """获取第 idx 个样本（DataLoader 会调用这个方法）"""
        return self.data[idx]


def get_dataLoader(args, dataset, model, tokenizer, batch_size=None, shuffle=False):
    """
    创建 DataLoader，包括复杂的 Seq2Seq 数据处理逻辑

    参数:
        args: 包含 max_input_length, max_target_length 等参数的对象
        dataset: LCSTS 数据集实例
        model: Seq2Seq 模型（用来初始化 decoder_input_ids）
        tokenizer: 分词器（同时处理 encoder 和 decoder 的分词）
        batch_size: 批次大小（为 None 时使用 args.batch_size）
        shuffle: 是否打乱（训练集应为 True，验证/测试集为 False）

    返回值:
        DataLoader: PyTorch DataLoader，每个 batch 包含：
                    - input_ids: encoder 输入
                    - attention_mask: encoder 注意力掩码
                    - decoder_input_ids: decoder 输入（向左移的目标）
                    - labels: 目标序列，用于计算损失（EOS 后的位置设为 -100 忽略）
    """

    def collote_fn(batch_samples):
        """
        将一个 batch 的样本组装成张量

        这是 Seq2Seq 的核心数据处理逻辑。为什么这么复杂？
        因为训练时用 teacher forcing（知道正确答案），推理时用自回归（逐步生成）。
        数据层需要把两者的数据准备好。

        参数:
            batch_samples: DataLoader 取出的 batch_size 个样本列表

        返回值:
            dict: 包含模型所需的所有张量
        """
        # 分离输入和目标
        batch_inputs, batch_targets = [], []
        for sample in batch_samples:
            batch_inputs.append(sample['content'])    # 原文 → encoder 输入
            batch_targets.append(sample['title'])     # 摘要 → decoder 目标

        # ============ 编码 Encoder 输入（原文） ============
        # 这部分标准处理：分词 + padding + truncation
        batch_data = tokenizer(
            batch_inputs,
            padding=True,                      # 动态 padding 到 batch 内最长长度
            max_length=args.max_input_length,  # 超出则截断
            truncation=True,
            return_tensors="pt"                # 返回 PyTorch 张量
        )

        # ============ 编码 Decoder 目标（摘要）— 需要特殊处理 ============
        # 使用 tokenizer.as_target_tokenizer() 告诉分词器：这是目标序列
        # （有些模型如 mBART 对 encoder/decoder 有不同的令牌词表）
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                batch_targets,
                padding=True,
                max_length=args.max_target_length,
                truncation=True,
                return_tensors="pt"
            )["input_ids"]  # 只取 token IDs，不需要 attention_mask

            # ============ 生成 decoder_input_ids ============
            # 关键概念：Seq2Seq decoder 的 "teacher forcing"
            #
            # 训练时：
            #   input:  [BOS, 中国, 消息, 发布]
            #   output: [中国, 消息, 发布, EOS]  ← 向左移了一位
            #
            # 这样 decoder 在 timestep t 看到 token t-1，预测 token t，
            # 与推理时的自回归流程保持一致（推理时上一步输出→下一步输入）。
            #
            # prepare_decoder_input_ids_from_labels() 自动处理这个：
            # - 取出 labels，在头部插入 BOS，在尾部删除 EOS
            batch_data['decoder_input_ids'] = model.prepare_decoder_input_ids_from_labels(labels)

            # ============ 处理 labels 的 EOS 后部分 ============
            # labels 中的 EOS 之后的 padding token 不应该计算损失
            # 原因：EOS 已经表示序列结束了，EOS 之后都是 padding，
            #      计算它们的损失是浪费，而且可能引入噪声
            #
            # 操作：
            # 1. 找到每个样本的 EOS 位置
            # 2. EOS 之后的所有位置设为 -100（损失函数会忽略 -100 标签）
            end_token_index = torch.where(labels == tokenizer.eos_token_id)[1]
            for idx, end_idx in enumerate(end_token_index):
                labels[idx][end_idx+1:] = -100  # -100 是 PyTorch CE Loss 的"忽略索引"

            batch_data['labels'] = labels

        return batch_data

    # 创建并返回 DataLoader
    return DataLoader(
        dataset,
        batch_size=(batch_size if batch_size else args.batch_size),
        shuffle=shuffle,
        collate_fn=collote_fn
    )
