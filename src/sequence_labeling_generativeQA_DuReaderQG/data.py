"""
生成式问答模型（T5-Base + DuReaderQG）- 数据加载模块
"""
import json
from pathlib import Path
from torch.utils.data import Dataset, DataLoader


class QADataset(Dataset):
    """
    生成式问答数据集

    数据格式 (JSON Lines):
    {"context": "...", "question": "...", "answer": "...", "id": 0}

    Args:
        data_file: 数据文件路径
        tokenizer: 分词器
        max_input_length: 输入序列最大长度（默认512）
        max_target_length: 目标序列最大长度（默认64）
    """

    def __init__(self, data_file, tokenizer, max_input_length=512, max_target_length=64):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.data = self.load_data(data_file)

    def load_data(self, data_file):
        """加载 JSON Lines 格式的数据"""
        data = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line.strip())
                data.append(sample)
        print(f"✓ 加载 {len(data)} 条数据 (来自: {Path(data_file).name})")
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        context = sample['context']
        question = sample['question']
        answer = sample['answer']

        # T5 输入格式："question: {Q} context: {C}"
        input_text = f"question: {question} context: {context}"

        # 编码输入
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # 编码目标（答案）
        targets = self.tokenizer(
            answer,
            max_length=self.max_target_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': targets['input_ids'].squeeze(),
            'decoder_attention_mask': targets['attention_mask'].squeeze(),
            'question': question,
            'context': context,
            'answer': answer
        }


def create_dataloaders(train_file, dev_file, tokenizer, batch_size,
                      max_input_length=512, max_target_length=64, num_workers=0):
    """
    创建训练和验证数据加载器

    Args:
        train_file: 训练数据文件路径
        dev_file: 验证数据文件路径
        tokenizer: 分词器
        batch_size: 批大小
        max_input_length: 输入最大长度
        max_target_length: 目标最大长度
        num_workers: 数据加载进程数

    Returns:
        train_loader, dev_loader: 数据加载器
    """
    # 创建数据集
    train_dataset = QADataset(
        train_file, tokenizer,
        max_input_length=max_input_length,
        max_target_length=max_target_length
    )
    dev_dataset = QADataset(
        dev_file, tokenizer,
        max_input_length=max_input_length,
        max_target_length=max_target_length
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    print(f"✓ 训练集批次数: {len(train_loader)}")
    print(f"✓ 验证集批次数: {len(dev_loader)}")

    return train_loader, dev_loader, train_dataset, dev_dataset