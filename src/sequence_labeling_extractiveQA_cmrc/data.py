"""
===============================================================================
数据加载模块：CMRC 2018 阅读理解数据的处理和预处理
===============================================================================

【核心概念】
Extractive QA 数据格式（SQuAD / CMRC 兼容）：
  {
    "data": [
      {
        "title": "文章标题",
        "paragraphs": [
          {
            "context": "这是文章内容...",
            "qas": [
              {
                "id": "question_id_1",
                "question": "问题？",
                "answers": [
                  {"text": "答案", "answer_start": 5}  # 答案在 context 中的起始字符位置
                ]
              }
            ]
          }
        ]
      }
    ]
  }

注意：answer_start 是【字符位置】，不是 token 位置。需要后续转换。
"""
from torch.utils.data import Dataset, DataLoader
import json
import torch

class CMRC2018(Dataset):
    """
    CMRC 2018 数据集加载器

    数据格式：SQuAD 兼容的 JSON 格式
    https://rajpurkar.github.io/SQuAD-explorer/

    每个样本包含：
      - context: 原文（passage）
      - question: 问题
      - answers: 答案列表（训练时可能有多个，推理时预测一个）
      - answer_start: 答案在 context 中的起始【字符位置】（不是 token）
    """

    def __init__(self, data_file):
        """
        参数:
            data_file (str): 数据文件路径（JSON 格式）
        """
        self.data = self.load_data(data_file)

    def load_data(self, data_file):
        """
        从 JSON 文件加载数据

        参数:
            data_file (str): 文件路径

        返回值:
            dict: {样本索引: 样本数据}
        """
        Data = {}
        with open(data_file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            idx = 0

            # 遍历所有文章
            for article in json_data['data']:
                title = article['title']
                # 获取第一个段落的内容和问答对
                context = article['paragraphs'][0]['context']

                # 遍历该段落的所有问答对
                for question in article['paragraphs'][0]['qas']:
                    q_id = question['id']  # 问题的唯一 ID（用于后续追踪）
                    ques = question['question']

                    # 答案可能有多个（例如众包标注）
                    text = [ans['text'] for ans in question['answers']]
                    # answer_start：答案在原文中的【起始字符位置】
                    answer_start = [ans['answer_start'] for ans in question['answers']]

                    Data[idx] = {
                        'id': q_id,
                        'title': title,
                        'context': context,  # 原文（字符串）
                        'question': ques,    # 问题（字符串）
                        'answers': {
                            'text': text,            # 答案文本列表
                            'answer_start': answer_start  # 起始位置列表（字符）
                        }
                    }
                    idx += 1
        return Data

    def __len__(self):
        """返回数据集大小"""
        return len(self.data)

    def __getitem__(self, idx):
        """获取第 idx 个样本"""
        return self.data[idx]

def get_dataLoader(args, dataset, tokenizer, mode='train', batch_size=None, shuffle=False):
    """
    创建 DataLoader，包括训练 / 验证 / 测试阶段的数据预处理

    【Extractive QA 的数据处理核心】
    1. tokenizer：question + context 一起分词，输出两个 sequence ID
    2. stride 和 overflowing tokens：处理超长文本的关键机制
    3. offset_mapping：token 到原始字符的映射
    4. 答案标签转换：character position → token position

    参数:
        args: 包含 max_length, stride 等参数的对象
        dataset: CMRC2018 数据集
        tokenizer: 分词器（支持 stride 参数）
        mode: 'train' / 'valid' / 'test'（决定 collate_fn 的行为）
        batch_size: 批次大小（None 时使用 args.batch_size）
        shuffle: 是否打乱（训练集为 True，验证集为 False）

    返回值:
        DataLoader: PyTorch DataLoader
    """

    assert mode in ['train', 'valid', 'test'], f"mode 必须是 'train' / 'valid' / 'test'，但得到 {mode}"

    def train_collote_fn(batch_samples):
        """
        训练数据的预处理：生成答案的起始和结束位置标签

        【关键步骤】
        1. 分词：question + context，启用 stride（处理超长文本）
        2. 获取 offset_mapping：每个 token 对应原文的字符范围
        3. 找到 context 在分词结果中的位置范围
        4. 将答案的字符位置转换为 token 位置
        5. 如果答案被截断或不在当前 chunk 中，标签为 (0, 0)
        """
        batch_question, batch_context, batch_answers = [], [], []
        for sample in batch_samples:
            batch_question.append(sample['question'])
            batch_context.append(sample['context'])
            batch_answers.append(sample['answers'])

        # ============ 分词：启用 stride 处理超长文本 ============
        # stride 的作用：当 context 超过 max_length 时，分割成多个 chunk，
        # 相邻 chunk 之间有 stride 个 token 的重叠，确保跨越边界的答案不会被遗漏。
        #
        # 返回结果：
        # - input_ids: [batch_size * num_chunks, max_length] 或变长
        # - token_type_ids: 0 = question, 1 = context
        # - offset_mapping: [batch_size * num_chunks, max_length, 2]
        #   即每个 token 在原文中的字符范围 [start_char, end_char]
        # - overflow_to_sample_mapping: 指示每个 chunk 来自原 batch 的哪个样本
        batch_inputs = tokenizer(
            batch_question,
            batch_context,
            max_length=args.max_length,
            truncation="only_second",    # 只截断 context（保留完整的 question）
            stride=args.stride,           # 相邻 chunk 的重叠 token 数
            return_overflowing_tokens=True,  # 返回溢出的部分（分割成多个 chunk）
            return_offsets_mapping=True,     # 返回 token 到字符的映射
            padding='max_length',
            return_tensors="pt"
        )

        # ============ 提取关键信息 ============
        # offset_mapping：token → 原文字符位置的映射
        offset_mapping = batch_inputs.pop('offset_mapping')
        # overflow_to_sample_mapping：每个 feature（chunk）来自原 batch 的哪个样本
        sample_mapping = batch_inputs.pop('overflow_to_sample_mapping')

        start_positions = []
        end_positions = []

        # 遍历每个 chunk
        for i, offset in enumerate(offset_mapping):
            # 获取这个 chunk 对应的原始样本
            sample_idx = sample_mapping[i]
            answer = batch_answers[sample_idx]

            # 答案的【字符】位置
            start_char = answer['answer_start'][0]
            end_char = answer['answer_start'][0] + len(answer['text'][0])

            # ============ 找到 context 在 token 序列中的位置范围 ============
            # sequence_ids(i) 返回第 i 个样本的 sequence IDs：
            # 0 = [CLS], question, [SEP]
            # 1 = context
            # -1 = [PAD]（某些 tokenizer 的实现）
            sequence_ids = batch_inputs.sequence_ids(i)

            # 找 context 的起始 token 位置（第一个 sequence_id == 1 的位置）
            idx = 0
            while idx < len(sequence_ids) and sequence_ids[idx] != 1:
                idx += 1
            context_start = idx

            # 找 context 的结束 token 位置（最后一个 sequence_id == 1 的位置）
            while idx < len(sequence_ids) and sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # ============ 转换：答案的字符位置 → token 位置 ============
            # 检查答案是否完全落在当前 chunk 的 context 范围内
            if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
                # 答案被截断或不在当前 chunk 内，标记为 impossible：(0, 0)
                start_positions.append(0)
                end_positions.append(0)
            else:
                # 答案完全在当前 chunk 内，找其对应的 token 范围

                # 找答案开始对应的 token：第一个 token 包含 start_char
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                # 找答案结束对应的 token：最后一个 token 包含 end_char - 1
                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        return {
            'batch_inputs': batch_inputs,
            'start_positions': start_positions,  # 答案起点的 token 位置
            'end_positions': end_positions       # 答案终点的 token 位置
        }

    def test_collote_fn(batch_samples):
        """
        测试 / 验证数据的预处理：不生成标签，但保留 offset_mapping 用于还原答案

        返回的 offset_mapping 将用于推理阶段，把 token 位置还原为原文字符范围。
        """
        batch_id, batch_question, batch_context = [], [], []
        for sample in batch_samples:
            batch_id.append(sample['id'])
            batch_question.append(sample['question'])
            batch_context.append(sample['context'])

        # ============ 分词（与训练相同，但不计算标签） ============
        batch_inputs = tokenizer(
            batch_question,
            batch_context,
            max_length=args.max_length,
            truncation="only_second",
            stride=args.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
            return_tensors="pt"
        )

        # ============ 处理 offset_mapping ============
        # 转为列表格式（便于后续处理）
        offset_mapping = batch_inputs.pop('offset_mapping').numpy().tolist()
        # 获取映射关系：每个 feature 来自哪个原始样本
        sample_mapping = batch_inputs.pop('overflow_to_sample_mapping')

        example_ids = []

        # 为每个 chunk 保留其样本 ID 和 offset_mapping，
        # 但只保留 context 部分的 offset（question 部分设为 None）
        for i in range(len(batch_inputs['input_ids'])):
            sample_idx = sample_mapping[i]
            example_ids.append(batch_id[sample_idx])

            # 找 context 的位置范围
            sequence_ids = batch_inputs.sequence_ids(i)
            offset = offset_mapping[i]

            # 只保留 context 部分（sequence_id == 1）的 offset，
            # question 和 padding 部分设为 None（用于后续过滤）
            offset_mapping[i] = [
                o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
            ]

        return {
            'batch_inputs': batch_inputs,
            'offset_mapping': offset_mapping,  # 用于推理阶段还原答案
            'example_ids': example_ids         # 样本 ID，用于追踪结果
        }

    # ============ 选择合适的 collate_fn ============
    if mode == 'train':
        collote_fn = train_collote_fn
    else:
        # 'valid' 和 'test' 都使用相同的 collate_fn（不需要标签）
        collote_fn = test_collote_fn

    return DataLoader(
        dataset,
        batch_size=(batch_size if batch_size else args.batch_size),
        shuffle=shuffle,
        collate_fn=collote_fn
    )
