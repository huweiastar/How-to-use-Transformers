# 生成式问答模型（T5-Base）

## 📋 项目说明

这是一个基于 **Google T5-Base** 的生成式问答（Abstractive QA）项目，使用 **DuReaderQG** 数据集进行训练。

### 任务特点

| 特性 | 说明 |
| --- | --- |
| **任务类型** | 生成式问答（Abstractive QA）|
| **模型** | Google T5-Base (220M 参数) |
| **数据集** | DuReaderQG（中文阅读理解） |
| **输入** | 问题 + 文章 |
| **输出** | 自动生成的答案 |
| **评估指标** | BLEU-1/2/3/4 |

## 🔄 和抽取式问答的区别

| 维度 | 抽取式（CMRC） | 生成式（DuReaderQG） |
| --- | --- | --- |
| **输出** | 答案位置（start/end） | 生成的答案文本 |
| **模型** | BERT（序列分类） | T5（编码-解码） |
| **灵活性** | 低（只能提取） | 高（可改写/总结） |
| **应用** | 阅读理解 | 问答、总结、翻译 |

## 📁 文件结构

```
sequence_labeling_generativeQA_DuReaderQG/
├── train_qa_model.py          # 完整的训练脚本
├── pipeline.ipynb             # Jupyter Notebook 版本
└── README.md                  # 本文件
```

## 📊 数据格式

JSON Lines 格式，每行一个样本：

```json
{
  "context": "违规分为:一般违规扣分、严重违规扣分...",
  "answer": "12月31日24:00",
  "question": "淘宝扣分什么时候清零",
  "id": 203
}
```

**字段说明**：
- `context`：参考文章（字符串）
- `question`：问题（字符串）
- `answer`：答案（字符串）
- `id`：样本 ID（整数）

## 🚀 使用方法

### 方式 1：运行 Python 脚本

```bash
cd /Users/huwei/PyCharmMiscProject/How-to-use-Transformers/src/sequence_labeling_generativeQA_DuReaderQG/

# 安装依赖
pip install transformers torch datasets nltk rouge-score matplotlib -q

# 运行训练脚本
python train_qa_model.py
```

### 方式 2：在 Jupyter Notebook 中运行

打开 `pipeline.ipynb` 文件，逐个 Cell 运行。

## 🎯 训练配置

| 参数 | 值 |
| --- | --- |
| 学习率 | 2e-5 |
| 批大小 | 8 |
| Epoch 数 | 3 |
| 优化器 | AdamW |
| 预热比例 | 0.1 |
| 梯度裁剪 | 1.0 |
| 随机种子 | 42 |

## 📈 输出文件

训练完成后，在 `qa_model_output/` 目录会生成：

```
qa_model_output/
├── best_model.pt                  # 最佳模型权重
├── final_model/                   # 完整的可加载模型
│   ├── pytorch_model.bin
│   ├── config.json
│   ├── tokenizer.json
│   └── ...
├── training_curves.png            # 训练曲线图
├── hyperparams.json               # 超参数配置
└── training_history.json          # 训练历史数据
```

## 💡 代码功能详解

### 1. 数据加载（`QADataset`）

```python
# T5 输入格式：\"question: {Q} context: {C}\"
input_text = f"question: {question} context: {context}"

# 编码并处理为张量
inputs = tokenizer(input_text, max_length=512, padding='max_length', truncation=True)
```

### 2. 模型架构

- **Encoder**：处理输入的问题和文章
- **Decoder**：生成答案文本
- **Loss**：CrossEntropyLoss（在解码器输出上）

### 3. BLEU 评估

计算 4 种 n-gram 的 BLEU 分数：
- **BLEU-1**：单词精确匹配
- **BLEU-2**：二元组匹配
- **BLEU-3**：三元组匹配
- **BLEU-4**：四元组匹配

### 4. 预测函数

```python
def predict_answer(question, context, model, tokenizer, device, max_length=64):
    """生成答案"""
    input_text = f"question: {question} context: {context}"
    inputs = tokenizer(input_text, max_length=512, truncation=True, return_tensors='pt')
    
    # Beam Search 生成答案
    generated_ids = model.generate(
        input_ids=inputs['input_ids'],
        max_length=64,
        num_beams=4,  # Beam Search 宽度
        early_stopping=True
    )
    
    answer = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return answer
```

## 🖼️ 训练曲线

训练完成后会生成 4 个子图：

1. **损失曲线**：训练损失 vs 验证损失
2. **BLEU-1/2**：单词和二元组精确度
3. **BLEU-3/4**：三元组和四元组精确度
4. **综合指标**：所有 BLEU 分数对比

## 📝 模型使用示例

### 加载模型

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer

model = T5ForConditionalGeneration.from_pretrained('./qa_model_output/final_model')
tokenizer = T5Tokenizer.from_pretrained('./qa_model_output/final_model')
```

### 预测答案

```python
# 定义预测函数
def predict(question, context):
    input_text = f"question: {question} context: {context}"
    inputs = tokenizer(input_text, max_length=512, truncation=True, return_tensors='pt')
    generated_ids = model.generate(inputs['input_ids'], max_length=64, num_beams=4)
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

# 预测
question = "什么时候清零?"
context = "淘宝网每年12月31日24:00点会对符合条件的扣分做清零处理。"
answer = predict(question, context)
print(f"答案: {answer}")  # 输出: 12月31日24:00
```

## ⚙️ 超参数调整建议

| 场景 | 调整 | 原因 |
| --- | --- | --- |
| **显存不足** | 减小 `batch_size` (4/2) | 降低显存占用 |
| **效果不好** | 增加 `num_epochs` (4/5) | 更充分的训练 |
| **收敛很慢** | 增加 `learning_rate` (3e-5) | 加快收敛 |
| **过拟合** | 减小 `learning_rate` (1e-5) | 防止过度学习 |
| **答案过短** | 增加 `max_target_length` (100) | 允许更长的答案 |

## 🔍 常见问题

### Q1: BLEU 分数很低怎么办？

1. **增加训练数据**：更多数据帮助模型学习
2. **加长训练周期**：增加 `num_epochs`
3. **调整学习率**：试试 1.5e-5 或 2.5e-5
4. **检查数据质量**：确保答案在文章中存在

### Q2: 生成的答案不连贯？

1. 增加 `num_beams`（从 4 到 8）提高搜索质量
2. 调整生成参数：`temperature=0.7, top_p=0.9`
3. 增加训练数据

### Q3: 如何加快训练？

1. 减小 `max_input_length`（从 512 到 256）
2. 增加 `batch_size`（如果显存允许）
3. 减少 `num_epochs`（但会影响效果）

## 📚 参考资源

- [Google T5 论文](https://arxiv.org/abs/1910.10683)
- [HuggingFace T5 文档](https://huggingface.co/docs/transformers/model_doc/t5)
- [DuReaderQG 数据集](https://github.com/PaddlePaddle/PaddleNLP)
- [BLEU 评估指标](https://en.wikipedia.org/wiki/BLEU)

## 📞 技术细节

### 模型选择

选择 **T5-Base** 而不是其他模型的原因：

1. **编码-解码架构**：天然适合生成任务
2. **多任务预训练**：在多种 NLP 任务上预训练过
3. **文本到文本范式**：统一的输入-输出格式
4. **模型大小合适**：220M 参数，显存 8GB 可训练

### Beam Search

生成答案时使用 Beam Search（4-beam）：

- 每步保留 4 个最可能的候选序列
- 找到全局最优答案而非贪心解
- 权衡精度和速度

## 🎓 学习路径

1. **理解数据**：查看 DuReaderQG 数据格式
2. **理解模型**：T5 的编码-解码机制
3. **理解训练**：损失函数、优化器、学习率调度
4. **理解评估**：BLEU 分数含义
5. **实践改进**：调参、增加数据、优化生成

## ✅ 检查清单

- [ ] 已安装 `transformers>=4.10.0`
- [ ] 已安装 `torch`（GPU 版本推荐）
- [ ] 数据文件存在：`../../data/DuReaderQG/`
- [ ] 有足够显存（推荐 12GB+）
- [ ] 已下载 NLTK punkt 数据

## 📄 许可证

MIT License