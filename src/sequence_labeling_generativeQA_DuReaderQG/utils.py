"""
生成式问答模型（T5-Base + DuReaderQG）- 工具函数模块
"""
import json
import numpy as np
import torch
from pathlib import Path
from tqdm.auto import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk

# 下载NLTK数据
nltk.download('punkt', quiet=True)


def calculate_bleu(predictions, references, n_gram=4):
    """
    计算BLEU分数

    Args:
        predictions: 预测文本列表
        references: 参考文本列表
        n_gram: 最大n-gram数（默认4）

    Returns:
        dict: 各个BLEU分数的平均值
    """
    smoothing_function = SmoothingFunction().method1
    bleu_scores = {f'BLEU-{i}': [] for i in range(1, n_gram + 1)}

    for pred, ref in zip(predictions, references):
        pred_tokens = pred.split()
        ref_tokens = ref.split()

        for i in range(1, n_gram + 1):
            weights = tuple([1.0 / i] * i)
            score = sentence_bleu(
                [ref_tokens],
                pred_tokens,
                weights=weights,
                smoothing_function=smoothing_function
            )
            bleu_scores[f'BLEU-{i}'].append(score)

    avg_bleu = {k: np.mean(v) for k, v in bleu_scores.items()}
    return avg_bleu


def predict_answer(question, context, model, tokenizer, device,
                   max_length=64, num_beams=4, temperature=0.9, top_p=0.95):
    """
    生成式问答预测函数

    Args:
        question: 问题文本
        context: 上下文/文章文本
        model: T5模型
        tokenizer: 分词器
        device: 计算设备
        max_length: 生成答案最大长度
        num_beams: Beam Search宽度
        temperature: 生成温度
        top_p: Top-p采样的p值

    Returns:
        str: 生成的答案文本
    """
    # 构造输入
    input_text = f"question: {question} context: {context}"

    # 编码输入
    inputs = tokenizer(
        input_text,
        max_length=512,
        truncation=True,
        return_tensors='pt'
    ).to(device)

    # 生成答案
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
            temperature=temperature,
            top_p=top_p
        )

    # 解码答案
    answer = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return answer


def save_predictions(predictions, references, output_file):
    """
    保存预测结果

    Args:
        predictions: 预测列表
        references: 参考答案列表
        output_file: 输出文件路径
    """
    results = []
    for pred, ref in zip(predictions, references):
        bleu_scores = calculate_bleu([pred], [ref])
        results.append({
            'prediction': pred,
            'reference': ref,
            'bleu_scores': bleu_scores
        })

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"✓ 预测结果已保存到: {output_file}")


def print_predictions(predictions, references, num_samples=5):
    """
    打印预测示例

    Args:
        predictions: 预测列表
        references: 参考答案列表
        num_samples: 打印的样本数量
    """
    print("\n" + "="*80)
    print("预测示例")
    print("="*80)

    for i in range(min(num_samples, len(predictions))):
        pred = predictions[i]
        ref = references[i]
        bleu_scores = calculate_bleu([pred], [ref])

        print(f"\n示例 {i + 1}:")
        print(f"✓ 参考答案: {ref}")
        print(f"🤖 预测答案: {pred}")
        print(f"📊 BLEU-1: {bleu_scores['BLEU-1']:.4f} | BLEU-2: {bleu_scores['BLEU-2']:.4f}")
        print("-" * 80)


def setup_seed(seed):
    """
    设置随机种子

    Args:
        seed: 随机种子值
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"✓ 随机种子已设置: {seed}")