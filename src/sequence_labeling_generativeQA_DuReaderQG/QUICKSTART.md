# 快速开始指南

## 30 秒快速开始

### 步骤 1：安装依赖

```bash
cd /Users/huwei/PyCharmMiscProject/How-to-use-Transformers/src/sequence_labeling_generativeQA_DuReaderQG/

pip install -r requirements.txt
```

### 步骤 2：运行训练脚本

```bash
python train_qa_model.py
```

就这么简单！脚本会：
- ✅ 加载 DuReaderQG 数据
- ✅ 训练 T5-Base 模型（3 个 epoch）
- ✅ 绘制训练曲线
- ✅ 进行预测
- ✅ 保存模型

## 输出示例

```
✓ 库导入成功
✓ PyTorch 版本: 2.0.0
✓ GPU 可用: True

✓ 加载 97896 条数据
✓ 加载 10000 条数据
✓ 分词器加载成功

==================================================
Epoch 1/3
==================================================
Training: 100%|██████████| 12237/12237 [45:32<00:00, ...]

训练损失: 2.4532

Evaluating: 100%|██████████| 1250/1250 [15:23<00:00, ...]

验证损失: 1.8765
验证 BLEU 分数:
  BLEU-1: 0.5234
  BLEU-2: 0.3821
  BLEU-3: 0.2847
  BLEU-4: 0.1923

✓ 保存最佳模型 (BLEU-2: 0.3821)
```

## 关键输出文件

| 文件 | 说明 |
| --- | --- |
| `qa_model_output/best_model.pt` | 最佳模型权重 |
| `qa_model_output/final_model/` | 可加载的完整模型 |
| `qa_model_output/training_curves.png` | 训练曲线图 |
| `qa_model_output/training_history.json` | 详细的训练数据 |

## 预测答案

训练完成后，脚本会显示预测示例：

```
================================================================================
验证集上的预测示例 (前 5 个样本)
================================================================================

示例 1:
问题: 淘宝扣分什么时候清零
文章: 违规分为:一般违规扣分、严重违规扣分...
参考答案: 12月31日24:00
预测答案: 12月31日24:00点
BLEU-1: 0.8333
BLEU-2: 0.6667

================================================================================
自定义输入预测
================================================================================

示例 1:
问题: 什么时候清零?
文章: 淘宝网每年12月31日24:00点会对符合条件的扣分做清零处理。
答案: 12月31日24:00点
```

## 完整的预测代码

如果你想在训练完成后单独预测，使用这段代码：

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

# 加载模型
model = T5ForConditionalGeneration.from_pretrained('./qa_model_output/final_model')
tokenizer = T5Tokenizer.from_pretrained('./qa_model_output/final_model')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

def predict_answer(question, context):
    """预测答案"""
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
            max_length=64,
            num_beams=4,
            early_stopping=True
        )
    
    answer = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return answer

# 使用示例
question = "什么是人工智能?"
context = "人工智能（AI）是计算机科学的一个分支，致力于创建能执行通常需要人工智能的任务的计算机。"
answer = predict_answer(question, context)
print(f"问题: {question}")
print(f"答案: {answer}")
```

## 常见问题

### Q: 需要多长时间？

- **数据加载**：~5 分钟
- **第 1 个 Epoch**：~45 分钟（取决于 GPU）
- **第 2、3 个 Epoch**：各 ~45 分钟
- **总计**：约 2.5-3 小时（RTX 3090）

### Q: 显存要求？

| GPU | 推荐 |
| --- | --- |
| A100 (80GB) | ✅ 轻松 |
| V100 (32GB) | ✅ 可以 |
| RTX 3090 (24GB) | ✅ 可以 |
| RTX 3080 (10GB) | ⚠️ 减小 batch_size |
| RTX 2080 (8GB) | ❌ 不推荐 |

如果显存不足，在 `train_qa_model.py` 中改：
```python
hyperparams['batch_size'] = 4  # 改从 8 到 4
```

### Q: 能否使用 CPU？

可以，但会很慢（100 倍）。建议使用 GPU。

### Q: 模型在哪里下载？

- 首次运行时，自动从 HuggingFace 下载 T5-Base (~1GB)
- 下载到 `~/.cache/huggingface/hub/`

## 进阶用法

### 修改超参数

编辑 `train_qa_model.py` 的 `hyperparams` 字典：

```python
hyperparams = {
    'learning_rate': 3e-5,      # 改学习率
    'num_epochs': 5,            # 改轮数
    'batch_size': 16,           # 改批大小
    'warmup_ratio': 0.2,        # 改预热比例
    'weight_decay': 0.01,       # 改权重衰减
    'max_grad_norm': 1.0,       # 改梯度裁剪
    'seed': 42
}
```

### 只进行预测（不训练）

注释掉训练代码，直接加载模型预测：

```python
# 注释掉这一大段
# for epoch in range(hyperparams['num_epochs']):
#     ...

# 直接加载最佳模型
model.load_state_dict(torch.load('./qa_model_output/best_model.pt'))

# 进行预测
question = "你的问题"
context = "你的文章"
answer = predict_answer(question, context, model, tokenizer, device)
print(answer)
```

## 实际应用场景

### 1. 智能客服

```python
def answer_customer_question(customer_question, faq_document):
    return predict_answer(customer_question, faq_document)

# 使用
answer = answer_customer_question(
    "如何退货?",
    "我们支持无理由退货，在收到商品后 30 天内可申请退货..."
)
```

### 2. 自动摘要

```python
def summarize_document(document):
    # 简单地用整个文档作为 context，问一个总结问题
    return predict_answer("这个文档的主要内容是什么?", document)
```

### 3. 信息提取

```python
def extract_info(context, info_type):
    questions = {
        'date': '发生了什么时候?',
        'place': '发生在哪里?',
        'person': '涉及了谁?',
    }
    return predict_answer(questions[info_type], context)

# 使用
date = extract_info(document, 'date')
place = extract_info(document, 'place')
```

## 下一步

- 📖 阅读 `README.md` 了解详细细节
- 🔧 查看 `train_qa_model.py` 的代码注释
- 📊 分析生成的 `training_curves.png`
- 🚀 调整超参数提升效果
- 💾 保存最佳模型用于生产环境

---

祝使用愉快！🎉
