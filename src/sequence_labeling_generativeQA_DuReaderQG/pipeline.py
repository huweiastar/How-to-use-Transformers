"""
生成式问答模型训练脚本（T5-Base）
数据集: DuReaderQG（中文阅读理解）

运行方式:
    python pipeline.py
"""

import os
import json
import random

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import T5ForConditionalGeneration, T5Tokenizer, get_linear_schedule_with_warmup
from tqdm.auto import tqdm
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

nltk.download('punkt', quiet=True)

# ============================================================
# 全局路径与模型配置
# ============================================================

DATA_DIR   = '../../data/DuReaderQG/'
OUTPUT_DIR = './qa_model_output/'

# 可选模型（修改 MODEL_CHOICE 切换）：
#   'mengzi-t5-base'  中文专用，推荐 ⭐
#   'mt5-base'        Google 多语言版
#   'google-t5-base'  英文版，仅用于英文任务
MODEL_CHOICE = 'mengzi-t5-base'

MODEL_CONFIG = {
    'mengzi-t5-base': {
        'model_name':    'Langboat/mengzi-t5-base',
        'tokenizer_name':'Langboat/mengzi-t5-base',
        'description':   'Mengzi T5-Base（中文，推荐）',
    },
    'mt5-base': {
        'model_name':    'google/mt5-base',
        'tokenizer_name':'google/mt5-base',
        'description':   'mT5-Base（多语言）',
    },
    'google-t5-base': {
        'model_name':    'google-t5/t5-base',
        'tokenizer_name':'google-t5/t5-base',
        'description':   'Google T5-Base（英文）',
    },
}

HYPERPARAMS = {
    'learning_rate': 2e-5,
    'num_epochs':    3,
    'batch_size':    8,
    'warmup_ratio':  0.1,   # 前 10% 步数线性预热
    'weight_decay':  0.01,  # AdamW 权重衰减，防过拟合
    'max_grad_norm': 1.0,   # 梯度裁剪上限
    'seed':          42,
}

# 中文 matplotlib 字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. 数据集
# ============================================================

class QADataset(Dataset):
    """
    DuReaderQG 生成式问答数据集。

    每条样本格式（JSON Lines）：
        {"context": "...", "question": "...", "answer": "..."}

    __getitem__ 返回：
        input_ids / attention_mask   —— 编码器输入（question + context 拼接）
        labels / decoder_attention_mask —— 解码器目标（answer）
        question / context / answer  —— 原始文本，供打印与 BLEU 计算使用
    """

    def __init__(self, data_file: str, tokenizer, max_input_len: int = 512, max_target_len: int = 64):
        self.tokenizer      = tokenizer
        self.max_input_len  = max_input_len
        self.max_target_len = max_target_len
        self.samples        = self._load(data_file)

    def _load(self, data_file: str):
        samples = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                samples.append(json.loads(line.strip()))
        print(f"  ✓ 加载 {len(samples)} 条数据：{data_file}")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        # T5 输入格式：任务前缀 + 问题 + 文章
        input_text = f"question: {s['question']} context: {s['context']}"

        # tokenizer 返回形状 (1, seq_len)，.squeeze() 去掉 batch 维度变成 (seq_len,)
        # 因为 DataLoader 会自己在外层加 batch 维度
        enc = self.tokenizer(
            input_text, max_length=self.max_input_len,
            padding='max_length', truncation=True, return_tensors='pt'
        )
        dec = self.tokenizer(
            s['answer'], max_length=self.max_target_len,
            padding='max_length', truncation=True, return_tensors='pt'
        )

        return {
            'input_ids':              enc['input_ids'].squeeze(),
            'attention_mask':         enc['attention_mask'].squeeze(),
            'labels':                 dec['input_ids'].squeeze(),
            'decoder_attention_mask': dec['attention_mask'].squeeze(),
            'question': s['question'],
            'context':  s['context'],
            'answer':   s['answer'],
        }


# ============================================================
# 2. 初始化工具函数
# ============================================================

def set_seed(seed: int):
    """固定所有随机种子，保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """按优先级选择：MPS（Apple Silicon）> CUDA > CPU。"""
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def build_dataloaders(train_dataset, dev_dataset, batch_size: int):
    """
    构建 DataLoader。
    训练集 shuffle=True：打乱顺序防止模型学到样本顺序规律。
    验证集 shuffle=False：保证每次评估顺序一致，结果可复现。
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader   = DataLoader(dev_dataset,   batch_size=batch_size, shuffle=False)
    return train_loader, dev_loader


def build_optimizer_and_scheduler(model, train_loader, hyperparams: dict):
    """
    构建 AdamW 优化器 + 线性预热衰减调度器。

    AdamW：为每个参数自适应调整步长，并额外施加权重衰减（防过拟合）。
    调度器曲线：
        前 warmup_ratio 比例的步数：学习率从 0 线性爬升到峰值（平稳起步）
        剩余步数：             学习率从峰值线性降到 0（精细收敛）
    """
    optimizer = AdamW(
        model.parameters(),
        lr=hyperparams['learning_rate'],
        weight_decay=hyperparams['weight_decay'],
    )

    total_steps  = len(train_loader) * hyperparams['num_epochs']
    warmup_steps = int(total_steps * hyperparams['warmup_ratio'])

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    return optimizer, scheduler, total_steps, warmup_steps


# ============================================================
# 3. 训练 / 评估 / 预测
# ============================================================

def compute_bleu(predictions: list[str], references: list[str], max_n: int = 4) -> dict:
    """
    计算 BLEU-1 到 BLEU-n 的平均分。

    BLEU-n 衡量预测与参考答案在 n 连续词（n-gram）上的重合度：
        BLEU-1：单词匹配  → 宽松，反映词汇覆盖
        BLEU-2：双词组匹配 → 兼顾词形与搭配，中文短文本的惯用选择
        BLEU-4：四词组匹配 → 严格，短答案容易全 0

    SmoothingFunction.method1：对 n-gram 计数为 0 时做平滑，
    避免短句子因某阶 n-gram 为 0 导致整体 BLEU 变 0。
    """
    smoother = SmoothingFunction().method1
    scores   = {f'BLEU-{n}': [] for n in range(1, max_n + 1)}

    for pred, ref in zip(predictions, references):
        pred_tokens = pred.split()
        ref_tokens  = ref.split()
        for n in range(1, max_n + 1):
            weights = tuple([1.0 / n] * n)
            s = sentence_bleu([ref_tokens], pred_tokens,
                              weights=weights, smoothing_function=smoother)
            scores[f'BLEU-{n}'].append(s)

    return {k: float(np.mean(v)) for k, v in scores.items()}


def train_one_epoch(model, train_loader, optimizer, scheduler, device, max_grad_norm: float) -> float:
    """
    训练一个 epoch，返回平均训练 loss。

    关键步骤：
        zero_grad()   → 清空上一步的梯度（PyTorch 默认累积梯度）
        loss.backward()→ 反向传播，计算每个参数的梯度
        clip_grad_norm → 梯度裁剪，防止梯度爆炸（把所有参数梯度的范数限制在阈值内）
        optimizer.step → 用梯度更新参数
        scheduler.step → 按调度器曲线更新学习率
    """
    model.train()
    total_loss = 0.0

    for batch in tqdm(train_loader, desc='  Train'):
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels         = batch['labels'].to(device)

        # transformers 内部：自动右移 labels 构造 decoder_input_ids，并计算交叉熵 loss
        loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def evaluate_on_loader(model, dev_loader, device, tokenizer, max_gen_len: int = 64):
    """
    在验证集上评估，返回 (avg_loss, bleu_scores, predictions, references)。

    注意两次前向传播的区别：
        model(...)       → teacher forcing，用标准答案强制喂给解码器，计算 loss（快）
        model.generate() → 自回归生成，解码器逐词输出，计算 BLEU（慢但反映真实生成质量）
    两者都有必要：loss 反映训练稳定性，BLEU 反映生成质量。
    """
    model.eval()
    total_loss  = 0.0
    predictions = []
    references  = []

    with torch.no_grad():
        for batch in tqdm(dev_loader, desc='  Eval '):
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['labels'].to(device)

            # 计算 loss（teacher forcing）
            total_loss += model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            ).loss.item()

            # 自回归生成，num_beams=4 即 Beam Search：每步保留 4 个候选，最后取总概率最大的路径
            generated_ids = model.generate(
                input_ids=input_ids, attention_mask=attention_mask,
                max_length=max_gen_len, num_beams=4, early_stopping=True,
            )
            predictions.extend(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))
            references.extend(batch['answer'])

    avg_loss    = total_loss / len(dev_loader)
    bleu_scores = compute_bleu(predictions, references)
    return avg_loss, bleu_scores, predictions, references


def predict_answer(question: str, context: str, model, tokenizer, device, max_gen_len: int = 64) -> str:
    """
    对单条（问题, 文章）生成答案，供推理/演示使用。

    temperature / top_p 是采样参数，只在 do_sample=True 时生效；
    这里用 Beam Search（num_beams=4），它们实际不起作用，但保留作示意。
    """
    input_text = f"question: {question} context: {context}"
    inputs = tokenizer(
        input_text, max_length=512, truncation=True, return_tensors='pt'
    ).to(device)

    model.eval()
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=max_gen_len,
            num_beams=4,
            early_stopping=True,
        )
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)


# ============================================================
# 4. 可视化 / 保存
# ============================================================

def plot_training_curves(history: dict, save_path: str):
    """绘制 Loss 和 BLEU 两张独立子图并保存为 PNG。"""
    epochs = range(1, len(history['train_loss']) + 1)

    fig, (ax_loss, ax_bleu) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('模型训练收敛曲线', fontsize=15, fontweight='bold')

    # 左图：Loss
    ax_loss.plot(epochs, history['train_loss'], label='训练 Loss', marker='o', linewidth=2)
    ax_loss.plot(epochs, history['dev_loss'],   label='验证 Loss', marker='s', linewidth=2)
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Loss')
    ax_loss.set_title('Loss 曲线', fontweight='bold')
    ax_loss.set_xticks(epochs)
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    # 右图：BLEU-1/2/3/4 全部画在一张图方便对比
    markers = ['o', 's', '^', 'd']
    for n, marker in zip(range(1, 5), markers):
        ax_bleu.plot(epochs, history[f'dev_bleu{n}'],
                     label=f'BLEU-{n}', marker=marker, linewidth=2)
    ax_bleu.set_xlabel('Epoch')
    ax_bleu.set_ylabel('BLEU 分数')
    ax_bleu.set_title('BLEU 曲线', fontweight='bold')
    ax_bleu.set_xticks(epochs)
    ax_bleu.legend()
    ax_bleu.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"  ✓ 曲线已保存：{save_path}")


def save_outputs(model, tokenizer, history: dict, hyperparams: dict, output_dir: str):
    """保存最终模型、分词器、训练历史、超参数到 output_dir。"""
    final_model_dir = os.path.join(output_dir, 'final_model')
    # 微调后 shared.weight 与 lm_head.weight 已各自独立更新，值不再相同，
    # 设为 False 使 config.json 与 safetensors 保持一致，避免加载时出现 tie 警告。
    model.config.tie_word_embeddings = False
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    print(f"  ✓ 模型已保存：{final_model_dir}")

    with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    with open(os.path.join(output_dir, 'hyperparams.json'), 'w') as f:
        json.dump(hyperparams, f, indent=2)

    print(f"  ✓ 训练历史 & 超参数已保存：{output_dir}")


def run_demo_predictions(model, tokenizer, device, dev_dataset, n_samples: int = 5):
    """在验证集前 n 条和自定义样例上打印预测结果，便于快速人工抽查。"""
    print("\n" + "=" * 70)
    print(f"验证集前 {n_samples} 条预测示例")
    print("=" * 70)
    for i in range(min(n_samples, len(dev_dataset))):
        s          = dev_dataset[i]
        prediction = predict_answer(s['question'], s['context'], model, tokenizer, device)
        bleu       = compute_bleu([prediction], [s['answer']])
        print(f"\n[{i+1}] 问题：{s['question']}")
        print(f"    文章：{s['context'][:80]}...")
        print(f"    参考：{s['answer']}")
        print(f"    预测：{prediction}")
        print(f"    BLEU-1={bleu['BLEU-1']:.4f}  BLEU-2={bleu['BLEU-2']:.4f}")
        print("-" * 70)

    custom_examples = [
        ("什么时候清零?",         "淘宝网每年12月31日24:00点会对符合条件的扣分做清零处理。"),
        ("这个产品的主要特点是什么?", "iPhone 15 采用了新一代 A17 Pro 芯片，支持 USB-C 接口，配备了更先进的相机系统，续航能力提升至 26 小时。"),
        ("中国的首都是哪里?",       "北京是中华人民共和国的首都，是全国的政治、文化、国际交往中心。"),
    ]

    print("\n" + "=" * 70)
    print("自定义输入预测")
    print("=" * 70)
    for i, (question, context) in enumerate(custom_examples, 1):
        pred = predict_answer(question, context, model, tokenizer, device)
        print(f"\n[{i}] 问题：{question}")
        print(f"    文章：{context}")
        print(f"    答案：{pred}")


# ============================================================
# 5. 主流程
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 随机种子 ────────────────────────────────────────────
    set_seed(HYPERPARAMS['seed'])
    device = get_device()
    print(f"✓ 设备：{device}")

    # ── 模型 / 分词器名称解析 ────────────────────────────────
    if MODEL_CHOICE.startswith('/') or MODEL_CHOICE.startswith('./'):
        model_name = tokenizer_name = MODEL_CHOICE
    elif MODEL_CHOICE in MODEL_CONFIG:
        cfg            = MODEL_CONFIG[MODEL_CHOICE]
        model_name     = cfg['model_name']
        tokenizer_name = cfg['tokenizer_name']
        print(f"✓ 模型：{cfg['description']}")
    else:
        model_name = tokenizer_name = MODEL_CHOICE

    # ── 分词器 ──────────────────────────────────────────────
    print(f"\n加载分词器：{tokenizer_name}")
    tokenizer = T5Tokenizer.from_pretrained(tokenizer_name, model_max_length=512)
    print("✓ 分词器加载成功")

    # ── 数据集 & DataLoader ──────────────────────────────────
    print("\n加载数据集...")
    train_dataset = QADataset(os.path.join(DATA_DIR, 'train.json'), tokenizer)
    dev_dataset   = QADataset(os.path.join(DATA_DIR, 'dev.json'),   tokenizer)
    train_loader, dev_loader = build_dataloaders(
        train_dataset, dev_dataset, HYPERPARAMS['batch_size']
    )
    print(f"  训练批次：{len(train_loader)}  验证批次：{len(dev_loader)}")

    # 打印第一条样本，帮助确认数据格式正确
    s = train_dataset[0]
    print(f"\n第一条训练样本 —— 问题：{s['question']}  答案：{s['answer']}")

    # ── 模型 ────────────────────────────────────────────────
    print(f"\n加载模型：{model_name}")
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"✓ 模型加载成功，参数量：{n_params:.1f}M")

    # ── 优化器 & 调度器 ──────────────────────────────────────
    optimizer, scheduler, total_steps, warmup_steps = build_optimizer_and_scheduler(
        model, train_loader, HYPERPARAMS
    )
    print(f"\n优化器配置  总步数：{total_steps}  预热步数：{warmup_steps}")
    print("\n超参数：")
    for k, v in HYPERPARAMS.items():
        print(f"  {k}: {v}")

    # ── 训练循环 ─────────────────────────────────────────────
    # 选取 BLEU-2 作为最佳模型判断指标的原因：
    #   DuReaderQG 答案平均 5~8 字，BLEU-1 太宽松，BLEU-4 在短文本上容易归零，
    #   BLEU-2 在词语搭配层面区分度最好，是中文短文本生成任务的惯用选择。
    history = {k: [] for k in ['train_loss', 'dev_loss',
                                'dev_bleu1', 'dev_bleu2', 'dev_bleu3', 'dev_bleu4']}
    best_bleu2      = 0.0
    best_model_path = os.path.join(OUTPUT_DIR, 'best_model.pt')

    print("\n开始训练...\n")
    for epoch in range(HYPERPARAMS['num_epochs']):
        print(f"\n{'='*50}  Epoch {epoch+1}/{HYPERPARAMS['num_epochs']}  {'='*50}")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            device, HYPERPARAMS['max_grad_norm']
        )
        print(f"\n  训练 Loss：{train_loss:.4f}")

        dev_loss, bleu_scores, _, _ = evaluate_on_loader(model, dev_loader, device, tokenizer)
        print(f"  验证 Loss：{dev_loss:.4f}")
        for k, v in bleu_scores.items():
            print(f"    {k}: {v:.4f}")

        # 记录历史
        history['train_loss'].append(train_loss)
        history['dev_loss'].append(dev_loss)
        for n in range(1, 5):
            history[f'dev_bleu{n}'].append(bleu_scores[f'BLEU-{n}'])

        # 按 BLEU-2 保存最佳 checkpoint
        if bleu_scores['BLEU-2'] > best_bleu2:
            best_bleu2 = bleu_scores['BLEU-2']
            torch.save(model.state_dict(), best_model_path)
            print(f"  ✓ 保存最佳模型（BLEU-2: {best_bleu2:.4f}）→ {best_model_path}")

    print("\n" + "=" * 50)
    print("训练完成！")

    # ── 可视化 ───────────────────────────────────────────────
    plot_training_curves(history, os.path.join(OUTPUT_DIR, 'training_curves.png'))

    # ── 加载最佳模型并演示预测 ────────────────────────────────
    print(f"\n加载最佳模型：{best_model_path}")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    run_demo_predictions(model, tokenizer, device, dev_dataset)

    # ── 保存最终模型 & 日志 ───────────────────────────────────
    print("\n保存输出...")
    save_outputs(model, tokenizer, history, HYPERPARAMS, OUTPUT_DIR)

    # ── 最终汇总 ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"  模型：{MODEL_CHOICE}  设备：{device}  参数量：{n_params:.1f}M")
    print(f"  最佳 BLEU-2：{best_bleu2:.4f}")
    print(f"  最终验证 Loss：{history['dev_loss'][-1]:.4f}")
    print(f"  输出目录：{OUTPUT_DIR}")
    print("=" * 70)


if __name__ == '__main__':
    main()
