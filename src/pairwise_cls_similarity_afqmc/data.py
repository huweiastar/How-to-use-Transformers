"""
数据处理模块 —— 加载 AFQMC 数据集并构造 DataLoader
================================================================

AFQMC = Ant Financial Question Matching Corpus（蚂蚁金融语义相似度数据集）。
任务是判断两句话是否同义，标签 0=不同义，1=同义。官方划分为
训练 34334 / 验证 4316 / 测试 3861 条。每行是一个 json：

    {"sentence1": "...", "sentence2": "...", "label": "0"}

本文件做两件事：
  1. AFQMC 类：继承 torch.utils.data.Dataset，把每行 json 读成一个样本 dict。
  2. get_dataLoader 函数：用自定义的 collate_fn 动态分词 + padding，
     返回可以直接喂给模型的 DataLoader。

【初学者重点】
  * 为什么要自定义 collate_fn？
      因为样本是 dict 且句子长度不等，PyTorch 默认的 collate 无法 stack。
  * 动态 padding 的优势？
      padding=True 只补齐到本 batch 最长句，seq_len 随 batch 变化，
      比固定 max_length=512 快很多，也省显存。
  * tokenizer 会返回什么？
      input_ids / attention_mask / token_type_ids 三个 [batch, seq_len] 张量。
      token_type_ids 用 0/1 区分第一句和第二句，是 BERT 句对任务的关键。
================================================================
"""
import json
from torch.utils.data import Dataset, DataLoader, IterableDataset

class AFQMC(Dataset):
    """
    AFQMC数据集类
    用于加载和管理AFQMC格式的数据（每行是一个JSON对象）
    Pytorch 通过 Dataset 类和 DataLoader 类处理数据集和加载样本。
    同样地，这里我们首先继承 Dataset 类构造自定义数据集，以组织样本和标签。
    AFQMC 样本以 json 格式存储，因此我们使用 json 库按行读取样本，并且以行号作为索引构建数据集。
    """

    def __init__(self, data_file):
        """
        初始化数据集

        参数:
            data_file (str): 数据文件路径，JSON Lines格式（每行一个JSON对象）
        """
        self.data = self.load_data(data_file)

    def load_data(self, data_file):
        """
        从JSON Lines格式的文件中加载数据

        参数:
            data_file (str): 数据文件路径

        返回值:
            Data (dict): 以索引为键、数据样本为值的字典
                        每个样本包含: sentence1, sentence2, label
        """
        Data = {}
        with open(data_file, 'rt', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                # 解析每一行JSON数据
                sample = json.loads(line.strip())
                Data[idx] = sample
        return Data

    def __len__(self):
        """返回数据集的样本总数"""
        return len(self.data)


    def __getitem__(self, idx):
        """
        根据索引获取单个样本

        参数:
            idx (int): 样本索引

        返回值:
            sample (dict): 包含sentence1, sentence2, label的字典
        """
        return self.data[idx]


def get_dataLoader(args, dataset, tokenizer, batch_size=None, shuffle=False):
    """
    接下来就需要通过 DataLoader 库按批 (batch) 加载数据，并且将样本转换成模型可以接受的输入格式。
    对于 NLP 任务，这个环节就是将每个 batch 中的文本按照预训练模型的格式进行编码（包括 Padding、截断等操作）。
    我们通过手工编写 DataLoader 的批处理函数 collate_fn 来实现。首先加载分词器，然后对每个 batch 中的所有句子对进行编码，同时把标签转换为张量格式。

    创建数据加载器（DataLoader）
    负责批处理、分词和填充操作

    参数:
        args: 包含配置信息的参数对象（包含max_seq_length, batch_size等）
        dataset: AFQMC数据集对象
        tokenizer: HuggingFace分词器，用于文本分词和编码
        batch_size (int, optional): 批次大小，如果为None则使用args.batch_size
        shuffle (bool): 是否打乱数据顺序，训练集通常为True，验证/测试集为False

    返回值:
        DataLoader: PyTorch数据加载器对象
    """

    def collote_fn(batch_samples):
        """
        自定义batch处理函数
        将多个样本组合成一个batch，进行分词、编码和填充

        参数:
            batch_samples (list): 一个batch的样本列表

        返回值:
            dict: 包含以下键的字典:
                - 'batch_inputs': 分词后的输入张量（包含input_ids, attention_mask, token_type_ids）
                - 'labels': 标签列表
        """
        # 初始化batch中各个成分的列表
        batch_sentence_1, batch_sentence_2, batch_label = [], [], []

        # 从batch中提取第一句、第二句和标签
        for sample in batch_samples:
            batch_sentence_1.append(sample['sentence1'])
            batch_sentence_2.append(sample['sentence2'])
            batch_label.append(int(sample['label']))

        # 使用tokenizer同时处理两个句子
        # tokenizer会自动添加特殊tokens (如[CLS], [SEP])，并处理句子对的编码
        batch_inputs = tokenizer(
            batch_sentence_1,  # 第一个句子列表
            batch_sentence_2,  # 第二个句子列表
            max_length=args.max_seq_length,  # 最大序列长度
            padding=True,      # 自动填充到batch中最长的序列长度
            truncation=True,   # 自动截断超过max_length的序列
            return_tensors="pt"  # 返回PyTorch张量格式
        )

        return {
            'batch_inputs': batch_inputs,  # 分词后的输入
            'labels': batch_label          # 标签列表
        }

    # 创建并返回DataLoader
    return DataLoader(
        dataset,
        batch_size=(batch_size if batch_size else args.batch_size),  # 使用指定的batch_size或默认值
        shuffle=shuffle,      # 是否打乱数据
        collate_fn=collote_fn # 使用自定义的batch处理函数
    )




# 如果数据集非常巨大，难以一次性加载到内存中，我们也可以继承 IterableDataset 类构建迭代型数据集：
# class IterableAFQMC(IterableDataset):
#     def __init__(self, data_file):
#         self.data_file = data_file
#
#     def __iter__(self):
#         with open(self.data_file, 'rt') as f:
#             for line in f:
#                 sample = json.loads(line.strip())
#                 yield sample


if __name__ == "__main__":
    """
    主程序入口
    这是 Python 文件的「主程序入口」
    只有直接运行这个文件时，里面的代码才会执行
    被别的文件导入时，不执行
    """
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    # 加载三个AFQMC数据集（训练、验证、测试）
    train_data = AFQMC('/Users/huwei/PyCharmMiscProject/How-to-use-Transformers/data/afqmc_public/train.json')
    valid_data = AFQMC('/Users/huwei/PyCharmMiscProject/How-to-use-Transformers/data/afqmc_public/dev.json')
    test_data = AFQMC('/Users/huwei/PyCharmMiscProject/How-to-use-Transformers/data/afqmc_public/test.json')

    # 打印各个数据集的第一个样本，用于验证数据加载是否正确
    print("训练集样本：", train_data[0])
    print("验证集样本：", valid_data[0])
    print("测试集样本：", test_data[0])

    # train_data = IterableAFQMC('/Users/huwei/PyCharmMiscProject/How-to-use-Transformers/data/afqmc_public/train.json')
    # print(next(iter(train_data)))

    checkpoint = "bert-base-chinese"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)


    def collote_fn(batch_samples):
        batch_sentence_1, batch_sentence_2 = [], []
        batch_label = []
        for sample in batch_samples:
            batch_sentence_1.append(sample['sentence1'])
            batch_sentence_2.append(sample['sentence2'])
            batch_label.append(int(sample['label']))
        X = tokenizer(
            batch_sentence_1,
            batch_sentence_2,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        y = torch.tensor(batch_label)
        return X, y


    train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True, collate_fn=collote_fn)

    batch_X, batch_y = next(iter(train_dataloader))
    print('batch_X shape:', {k: v.shape for k, v in batch_X.items()})
    print('batch_y shape:', batch_y.shape)

    # 典型输出形如：
    #   batch_X shape: {'input_ids': [4, 39], 'token_type_ids': [4, 39], 'attention_mask': [4, 39]}
    #   batch_y shape: [4]
    # 三个维度的含义：
    #   4  = batch_size（一次取 4 条样本）
    #   39 = 本 batch 内最长那条句对拼接后的 token 数（动态，每个 batch 都可能不同）
    # 想直观确认 token_type_ids 的作用，可以打印一下：
    #   print(batch_X['token_type_ids'][0])
    # 会看到前半部分是 0（第一句 + 第一个 [SEP]），后半部分是 1（第二句 + 第二个 [SEP]）。


