"""
生成式问答模型训练脚本（T5-Base）
数据集: DuReaderQG（中文阅读理解）
"""
import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)
from tqdm.auto import tqdm
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
nltk.download('punkt', quiet=True)

print("✓ 库导入成功")

# ============ 1. 数据加载 ============

DATA_DIR = '../../data/DuReaderQG/'
OUTPUT_DIR = './qa_model_output/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_FILE = os.path.join(DATA_DIR, 'train.json')
DEV_FILE = os.path.join(DATA_DIR, 'dev.json')

class QADataset(Dataset):
    """生成式问答数据集"""

    def __init__(self, data_file, tokenizer, max_input_length=512, max_target_length=64):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.data = self.load_data(data_file)

    def load_data(self, data_file):
        """加载 JSON Lines 数据"""
        data = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line.strip())
                data.append(sample)
        print(f"✓ 加载 {len(data)} 条数据")
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        context = sample['context']
        question = sample['question']
        answer = sample['answer']

        # T5 输入格式：\"question: {Q} context: {C}\"
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

# 加载分词器和数据
print("加载 T5 分词器...")
tokenizer = T5Tokenizer.from_pretrained('google-t5/t5-base', model_max_length=512)
print("✓ 分词器加载成功")

print("\n加载训练集...")
train_dataset = QADataset(TRAIN_FILE, tokenizer, max_input_length=512, max_target_length=64)
print("\n加载验证集...")
dev_dataset = QADataset(DEV_FILE, tokenizer, max_input_length=512, max_target_length=64)

# 查看数据样本
sample = train_dataset[0]
print("\n=== 第一条训练样本 ===")
print(f"问题: {sample['question']}")
print(f"原文: {sample['context'][:200]}...")
print(f"答案: {sample['answer']}")

# ============ 2. 模型初始化 ============

hyperparams = {
    'learning_rate': 2e-5,
    'num_epochs': 3,
    'batch_size': 8,
    'warmup_ratio': 0.1,
    'weight_decay': 0.01,
    'max_grad_norm': 1.0,
    'seed': 42
}

# 设置随机种子
random.seed(hyperparams['seed'])
np.random.seed(hyperparams['seed'])
torch.manual_seed(hyperparams['seed'])

print("\n超参数配置:")
for k, v in hyperparams.items():
    print(f"  {k}: {v}")

# 设置设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\n✓ 设备: {device}")

# 创建 DataLoader
print("\n创建 DataLoader...")
train_loader = DataLoader(train_dataset, batch_size=hyperparams['batch_size'], shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=hyperparams['batch_size'], shuffle=False)
print(f"✓ 训练集批次数: {len(train_loader)}")
print(f"✓ 验证集批次数: {len(dev_loader)}")

# 加载 T5 模型
print("\n加载 T5-Base 模型...")
model = T5ForConditionalGeneration.from_pretrained('google-t5/t5-base')
model = model.to(device)
print(f"✓ 模型加载成功")
print(f"✓ 模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

# 优化器
optimizer = AdamW(
    model.parameters(),
    lr=hyperparams['learning_rate'],
    weight_decay=hyperparams['weight_decay']
)

# 学习率调度器
total_steps = len(train_loader) * hyperparams['num_epochs']
warmup_steps = int(total_steps * hyperparams['warmup_ratio'])
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

print(f"\n优化器配置:")
print(f"  总步数: {total_steps}")
print(f"  预热步数: {warmup_steps}")

# ============ 3. 训练函数 ============

def calculate_bleu(predictions, references, n_gram=4):
    """计算 BLEU 分数"""
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


def train_epoch(model, train_loader, optimizer, scheduler, device):
    """训练一个 epoch"""
    model.train()
    total_loss = 0

    progress_bar = tqdm(train_loader, desc='Training')
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), hyperparams['max_grad_norm'])
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})

    avg_loss = total_loss / len(train_loader)
    return avg_loss


def evaluate(model, dev_loader, device, tokenizer, max_length=64):
    """在验证集上评估"""
    model.eval()
    total_loss = 0
    predictions = []
    references = []

    with torch.no_grad():
        for batch in tqdm(dev_loader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            total_loss += loss.item()

            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                num_beams=4,
                early_stopping=True
            )

            preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            refs = batch['answer']

            predictions.extend(preds)
            references.extend(refs)

    avg_loss = total_loss / len(dev_loader)
    bleu_scores = calculate_bleu(predictions, references)

    return avg_loss, bleu_scores, predictions, references

# ============ 4. 训练循环 ============

training_history = {
    'train_loss': [],
    'dev_loss': [],
    'dev_bleu1': [],
    'dev_bleu2': [],
    'dev_bleu3': [],
    'dev_bleu4': []
}

best_bleu = 0
best_model_path = os.path.join(OUTPUT_DIR, 'best_model.pt')

print("\n开始训练...\n")
for epoch in range(hyperparams['num_epochs']):
    print(f"\n{'='*50}")
    print(f"Epoch {epoch + 1}/{hyperparams['num_epochs']}")
    print(f"{'='*50}")

    # 训练
    train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
    print(f"\n训练损失: {train_loss:.4f}")
    training_history['train_loss'].append(train_loss)

    # 验证
    dev_loss, bleu_scores, predictions, references = evaluate(model, dev_loader, device, tokenizer)
    print(f"\n验证损失: {dev_loss:.4f}")
    print("验证 BLEU 分数:")
    for k, v in bleu_scores.items():
        print(f"  {k}: {v:.4f}")

    # 记录历史
    training_history['dev_loss'].append(dev_loss)
    training_history['dev_bleu1'].append(bleu_scores['BLEU-1'])
    training_history['dev_bleu2'].append(bleu_scores['BLEU-2'])
    training_history['dev_bleu3'].append(bleu_scores['BLEU-3'])
    training_history['dev_bleu4'].append(bleu_scores['BLEU-4'])

    # 保存最佳模型
    if bleu_scores['BLEU-2'] > best_bleu:
        best_bleu = bleu_scores['BLEU-2']
        torch.save(model.state_dict(), best_model_path)
        print(f"\n✓ 保存最佳模型 (BLEU-2: {best_bleu:.4f})")

print("\n" + "="*50)
print("训练完成！")
print("="*50)

# ============ 5. 绘制收敛曲线 ============

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('模型训练收敛曲线', fontsize=16, fontweight='bold')

# 1. 损失曲线
ax1 = axes[0, 0]
ax1.plot(training_history['train_loss'], label='训练损失', marker='o', linewidth=2)
ax1.plot(training_history['dev_loss'], label='验证损失', marker='s', linewidth=2)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('损失', fontsize=12)
ax1.set_title('损失曲线', fontsize=13, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# 2. BLEU-1/2 曲线
ax2 = axes[0, 1]
ax2.plot(training_history['dev_bleu1'], label='BLEU-1', marker='o', linewidth=2)
ax2.plot(training_history['dev_bleu2'], label='BLEU-2', marker='s', linewidth=2)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('BLEU 分数', fontsize=12)
ax2.set_title('BLEU-1/2 分数', fontsize=13, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

# 3. BLEU-3/4 曲线
ax3 = axes[1, 0]
ax3.plot(training_history['dev_bleu3'], label='BLEU-3', marker='o', linewidth=2)
ax3.plot(training_history['dev_bleu4'], label='BLEU-4', marker='s', linewidth=2)
ax3.set_xlabel('Epoch', fontsize=12)
ax3.set_ylabel('BLEU 分数', fontsize=12)
ax3.set_title('BLEU-3/4 分数', fontsize=13, fontweight='bold')
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3)

# 4. 所有 BLEU 分数
ax4 = axes[1, 1]
ax4.plot(training_history['dev_bleu1'], label='BLEU-1', marker='o', linewidth=2)
ax4.plot(training_history['dev_bleu2'], label='BLEU-2', marker='s', linewidth=2)
ax4.plot(training_history['dev_bleu3'], label='BLEU-3', marker='^', linewidth=2)
ax4.plot(training_history['dev_bleu4'], label='BLEU-4', marker='d', linewidth=2)
ax4.set_xlabel('Epoch', fontsize=12)
ax4.set_ylabel('BLEU 分数', fontsize=12)
ax4.set_title('所有 BLEU 指标', fontsize=13, fontweight='bold')
ax4.legend(fontsize=11)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'training_curves.png'), dpi=300, bbox_inches='tight')
plt.show()

print(f"\n✓ 曲线已保存到 {OUTPUT_DIR}training_curves.png")

# ============ 6. 模型预测 ============

model.load_state_dict(torch.load(best_model_path))
model.eval()
print(f"\n✓ 加载最佳模型")


def predict_answer(question, context, model, tokenizer, device, max_length=64):
    """生成式问答预测函数"""
    input_text = f"question: {question} context: {context}"
    inputs = tokenizer(
        input_text,
        max_length=512,
        truncation=True,
        return_tensors='pt'
    ).to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=max_length,
            num_beams=4,
            early_stopping=True,
            temperature=0.9,
            top_p=0.95
        )

    answer = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return answer


# 测试预测
print("\n" + "="*80)
print("验证集上的预测示例 (前 5 个样本)")
print("="*80)

for i in range(min(5, len(dev_dataset))):
    sample = dev_dataset[i]
    question = sample['question']
    context = sample['context']
    ground_truth = sample['answer']

    prediction = predict_answer(question, context, model, tokenizer, device)
    bleu_scores = calculate_bleu([prediction], [ground_truth])

    print(f"\n示例 {i + 1}:")
    print(f"问题: {question}")
    print(f"文章: {context[:100]}...")
    print(f"参考答案: {ground_truth}")
    print(f"预测答案: {prediction}")
    print(f"BLEU-1: {bleu_scores['BLEU-1']:.4f}")
    print(f"BLEU-2: {bleu_scores['BLEU-2']:.4f}")
    print("-" * 80)

# ============ 7. 自定义输入预测 ============

print("\n" + "="*80)
print("自定义输入预测")
print("="*80)

examples = [
    ("什么时候清零?", "淘宝网每年12月31日24:00点会对符合条件的扣分做清零处理。"),
    ("这个产品的主要特点是什么?", "iPhone 15 采用了新一代 A17 Pro 芯片，支持 USB-C 接口，配备了更先进的相机系统，续航能力提升至 26 小时。"),
    ("中国的首都是哪里?", "北京是中华人民共和国的首都，是全国的政治、文化、国际交往中心，也是中国历史文化名城。"),
]

for i, (question, context) in enumerate(examples, 1):
    print(f"\n示例 {i}:")
    print(f"问题: {question}")
    print(f"文章: {context}")
    pred = predict_answer(question, context, model, tokenizer, device)
    print(f"答案: {pred}")

print("\n" + "="*80)

# ============ 8. 保存模型 ============

model_save_path = os.path.join(OUTPUT_DIR, 'final_model')
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)

print(f"\n✓ 模型已保存到: {model_save_path}")

# 保存超参数
with open(os.path.join(OUTPUT_DIR, 'hyperparams.json'), 'w') as f:
    json.dump(hyperparams, f, indent=2)
print(f"✓ 超参数已保存")

# 保存训练历史
with open(os.path.join(OUTPUT_DIR, 'training_history.json'), 'w') as f:
    json.dump(training_history, f, indent=2)
print(f"✓ 训练历史已保存")

# ============ 9. 最终总结 ============

print("\n" + "="*80)
print("项目总结")
print("="*80)
print(f"""
【模型信息】
- 模型架构: Google T5-Base (220M 参数)
- 任务类型: 生成式问答 (Abstractive QA)
- 数据集: DuReaderQG (中文阅读理解)
- 设备: {device}

【训练配置】
- 批大小: {hyperparams['batch_size']}
- 学习率: {hyperparams['learning_rate']}
- 总 Epoch: {hyperparams['num_epochs']}
- 优化器: AdamW

【最终指标 (第 {hyperparams['num_epochs']} Epoch)】
- 训练损失: {training_history['train_loss'][-1]:.4f}
- 验证损失: {training_history['dev_loss'][-1]:.4f}
- BLEU-1: {training_history['dev_bleu1'][-1]:.4f}
- BLEU-2: {training_history['dev_bleu2'][-1]:.4f}
- BLEU-3: {training_history['dev_bleu3'][-1]:.4f}
- BLEU-4: {training_history['dev_bleu4'][-1]:.4f}

【输出文件】
- 最佳模型: {best_model_path}
- 完整模型: {model_save_path}
- 训练曲线: {os.path.join(OUTPUT_DIR, 'training_curves.png')}
""")
print("="*80)
