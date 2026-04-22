# 改造指南：把 CMRC 抽取式问答项目迁移到你自己的数据 / 任务 / 场景

> 配套文件：本目录的 `readme.md`（项目介绍）+ `pipeline.ipynb`（教学笔记本）+ `data.py / modeling.py / arg.py / run_extractiveQA.py`（工程化版本）。
>
> **注意**：这是 **token 级别的序列标注任务**（和句级分类 AFQMC 完全不同）。读完本指南，你应该能在 **1 小时内** 把这套代码改造成任意一个"抽取式问答"类型的微调项目。

---

## 〇、先搞清楚这是什么任务

### 任务定义

**抽取式问答（Extractive QA）**：给定**文章 + 问题**，从文章中**提取答案的起始和结束位置**。

### 数据格式（SQuAD/CMRC 标准）

```json
{
  "data": [
    {
      "title": "文章标题",
      "paragraphs": [
        {
          "context": "这是文章的完整内容，很长很长...",
          "qas": [
            {
              "id": "question_id_001",
              "question": "问题是什么？",
              "answers": [
                {
                  "text": "答案文本",
                  "answer_start": 42
                }
              ]
            }
          ]
        }
      ]
    }
  ]
}
```

**关键字段解释**：
- `context`：原文（长字符串）
- `question`：问题（字符串）
- `answers[].text`：答案文本（可能多个，众包标注）
- `answers[].answer_start`：答案在 context 里的**起始字符位置**（不是 token 位置）

### 和 AFQMC（句对分类）的核心区别

| 维度 | AFQMC（句对分类） | 抽取式问答 |
| --- | --- | --- |
| **输入** | 两个句子 | 文章 + 问题 |
| **输出** | 1 个标签（二分类 0/1） | 2 个位置（答案的起点 + 终点） |
| **任务类型** | Sequence Classification（序列级） | Token Classification（token 级） |
| **数据难点** | 字段提取简单 | 需要 offset_mapping（字符→token 映射）、stride（超长文本处理） |
| **评估指标** | Accuracy | F1 和 EM（Exact Match） |

### 心智模型（抽取式 QA 的 5 个零件）

```
   原始 JSON           context + question
       ↓                      ↓
   [1] 加载数据集    [2] collate_fn          [3] Encoder        [4] Task Head            [5] Loss
   (data.py)      分词+offset_mapping    (BERT encoder)    (Linear 预测位置)      (CrossEntropyLoss)
                  转换answer_start→token      ↓               ↓                        ↓
                  处理超长文本(stride)   [B, L, H]      start_logits [B,L]    L = (L_start+L_end)/2
                                          ↓            end_logits [B,L]
```

改造就是：你换了哪个零件，就只改对应文件里那几行。

---

## 一、场景 A：**换自己的 QA 数据集**（同格式）

最常见的情形。比如你有一个**医学领域的阅读理解数据**，或者**法律条文问答**。

### 前提条件

你的数据已经是 **SQuAD/CMRC 兼容的 JSON 格式**（结构如上所示）。

### 改动步骤

**第 1 步**：准备数据目录
```bash
mkdir -p data/medical_qa
# 把你的 JSON 文件放进去
# medical_qa/
# ├── train.json
# ├── dev.json
# └── test.json
```

**第 2 步**：只改 shell 脚本里的数据路径
```bash
# run_medical_qa.sh（新建）
export OUTPUT_DIR=./medical_qa_results/

python3 run_extractiveQA.py \
    --output_dir=$OUTPUT_DIR \
    --model_type=bert \
    --model_checkpoint=bert-base-chinese \
    --train_file=../../data/medical_qa/train.json \
    --dev_file=../../data/medical_qa/dev.json \
    --test_file=../../data/medical_qa/test.json \
    --max_length=512 \
    --max_answer_length=50 \
    --stride=128 \
    --n_best=20 \
    --learning_rate=1e-5 \
    --num_train_epochs=3 \
    --batch_size=4 \
    --do_train \
    --warmup_proportion=0.1 \
    --seed=42
```

**关键调整**：
- `--max_length`：根据你的医学文档长度调整（医学文本可能很长）
- `--max_answer_length`：医学答案通常较长，可能需要调大（比如 100）
- `--batch_size`：长文本占显存，可能需要调小（甚至 2）

**第 3 步**：不用改任何代码
- `data.py` 的 `CMRC2018` 类能自动加载任何 SQuAD 格式的 JSON
- `modeling.py` 完全不动
- `run_extractiveQA.py` 完全不动

**就这样，一个新的 QA 项目跑起来了。**

---

## 二、场景 B：**自己的数据不是标准 SQuAD 格式**

比如你的数据长这样：

```csv
id,passage,question,answer,char_start
1,"医学文献...",问题1,答案1,42
2,"医学文献...",问题2,答案2,88
```

或者你的数据是自定义 JSON 格式：

```json
{
  "samples": [
    {
      "doc_id": "doc_001",
      "doc_text": "...",
      "q_text": "问题",
      "a_text": "答案",
      "a_start": 42
    }
  ]
}
```

### 改动方案：转换脚本 vs 改代码

**推荐方案**：写个小脚本一次性转成标准 SQuAD 格式，代码一行不用改。

**转换脚本示例** `convert_to_squad.py`：
```python
import json

# 读取你的数据
with open('your_data.csv', 'r', encoding='utf-8') as f:
    lines = f.readlines()[1:]  # 跳过表头
    
# 转成 SQuAD 格式
squad_data = {"data": []}
for idx, line in enumerate(lines):
    parts = line.strip().split(',')
    doc_id, passage, question, answer, char_start = parts
    
    squad_data['data'].append({
        "title": doc_id,
        "paragraphs": [{
            "context": passage,
            "qas": [{
                "id": f"q_{idx}",
                "question": question,
                "answers": [{
                    "text": answer,
                    "answer_start": int(char_start)
                }]
            }]
        }]
    })

# 保存为 SQuAD 格式
with open('converted_data.json', 'w', encoding='utf-8') as f:
    json.dump(squad_data, f, ensure_ascii=False, indent=2)
```

运行转换脚本后，你的数据就变成标准格式了，**后续代码一个字不用改**。

### 如果不想转换，直接改代码

**修改 `data.py` 里的 `load_data` 方法**：

**改之前**（标准 SQuAD）：
```python
def load_data(self, data_file):
    Data = {}
    with open(data_file, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
        idx = 0
        # 遍历 json_data['data'] → articles → paragraphs → qas
        for article in json_data['data']:
            title = article['title']
            context = article['paragraphs'][0]['context']
            for question in article['paragraphs'][0]['qas']:
                # 提取字段
                ...
```

**改之后**（自定义格式）：
```python
def load_data(self, data_file):
    Data = {}
    with open(data_file, 'r', encoding='utf-8') as f:
        json_data = json.load(f)  # 假设你的格式是 {"samples": [...]}
        idx = 0
        for sample in json_data['samples']:  # ← 改这里
            Data[idx] = {
                'id': sample['doc_id'],               # ← 改字段名
                'title': sample['doc_id'],
                'context': sample['doc_text'],         # ← 改字段名
                'question': sample['q_text'],          # ← 改字段名
                'answers': {
                    'text': [sample['a_text']],        # ← 改字段名（包成列表）
                    'answer_start': [sample['a_start']]  # ← 改字段名（包成列表）
                }
            }
            idx += 1
    return Data
```

**改动总结**：只需改 `data.py` 里的 `load_data` 方法，提取你的自定义字段名并映射到标准的 `context / question / answers` 即可。

---

## 三、场景 C：**改预训练模型**

### 3.1 同架构换权重（最常见）

在 shell 脚本里改 `--model_checkpoint`：

```bash
# 从 BERT 换成 RoBERTa
--model_checkpoint=hfl/chinese-roberta-wwm-ext

# 或者用 MacBERT（可能对医学文本更友好）
--model_checkpoint=hfl/chinese-macbert-base

# 或者用 ERNIE（百度模型）
--model_checkpoint=nghuyong/ernie-3.0-base-zh
```

**完全不用改代码**。

### 3.2 换成完全不同的架构（如 RoBERTa）

**第 1 步**：在 `modeling.py` 里加新类

```python
from transformers import RobertaPreTrainedModel, RobertaModel

class RobertaForExtractiveQA(RobertaPreTrainedModel):
    """基于 RoBERTa 的抽取式问答模型"""
    
    def __init__(self, config, args):
        super().__init__(config)
        self.num_labels = args.num_labels
        self.roberta = RobertaModel(config, add_pooling_layer=False)  # ← 改这里
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    def forward(self, batch_inputs, start_positions=None, end_positions=None):
        # 逻辑与 BertForExtractiveQA 完全相同，只是 encoder 换成 RoBERTa
        bert_output = self.roberta(**batch_inputs)  # ← 改这里
        # ... 其他逻辑不变
```

**第 2 步**：在 `run_extractiveQA.py` 里改 `MODEL_CLASSES`

```python
# 改之前
MODEL_CLASSES = {
    'bert': BertForExtractiveQA,
}

# 改之后
MODEL_CLASSES = {
    'bert': BertForExtractiveQA,
    'roberta': RobertaForExtractiveQA,  # ← 加这行
}
```

**第 3 步**：shell 脚本里改

```bash
--model_type=roberta \
--model_checkpoint=facebook/roberta-base
```

---

## 四、场景 D：**调整超长文本处理参数**

这是抽取式 QA 特有的。如果你的文章特别长或特别短，需要调 `--max_length` 和 `--stride`。

### 参数说明

| 参数 | 含义 | 调整建议 |
| --- | --- | --- |
| `--max_length` | 输入序列最大长度 | 文章长 → 加大（512/768）；文章短 → 减小（256） |
| `--stride` | 超长文本时的重叠 token 数 | 大 → 精度高但慢；小 → 快但可能漏掉长范围答案 |
| `--max_answer_length` | 答案最多多少个 token | 通常 30~50；医学答案可能要 100+ |
| `--n_best` | 推理时保留前 n 个候选 | 大 → 精度高但慢；小 → 快但可能漏掉好答案 |

### 三个典型场景

**场景 1：文章很短（< 256 token）**
```bash
--max_length=256 \
--stride=64 \
--max_answer_length=20 \
--n_best=10
```

**场景 2：文章很长（医学论文 / 法律文书）**
```bash
--max_length=512 \
--stride=256 \
--max_answer_length=100 \
--n_best=30
```

**场景 3：显存有限（batch_size 太小）**
```bash
--max_length=256 \
--stride=128 \
--batch_size=2 \
--max_answer_length=30 \
--n_best=15
```

---

## 五、改造模板：动手做一个新项目

### 任务：医学论文问答（Medical QA）

假设你有医学论文的自动抽取式问答数据。

### Step 1：准备数据目录和文件

```bash
mkdir -p data/medical_qa
```

**train.json 格式**（SQuAD 兼容）：
```json
{
  "data": [
    {
      "title": "论文001",
      "paragraphs": [
        {
          "context": "高血压是指体循环动脉血压升高为主要特征，可伴有心、脑、肾等靶器官功能或器质性损害的疾病。高血压是世界范围内的重要公共卫生问题...",
          "qas": [
            {
              "id": "medical_q_001",
              "question": "高血压的主要特征是什么？",
              "answers": [
                {
                  "text": "体循环动脉血压升高",
                  "answer_start": 7
                }
              ]
            },
            {
              "id": "medical_q_002",
              "question": "高血压可能伴有哪些器官功能损害？",
              "answers": [
                {
                  "text": "心、脑、肾等靶器官",
                  "answer_start": 32
                }
              ]
            }
          ]
        }
      ]
    },
    {
      "title": "论文002",
      "paragraphs": [
        {
          "context": "糖尿病是一组以高血糖为特征的代谢性疾病群...",
          "qas": [...]
        }
      ]
    }
  ]
}
```

### Step 2：新建 shell 脚本 `run_medical_qa.sh`

```bash
#!/bin/bash

export OUTPUT_DIR=./medical_qa_results/

python3 run_extractiveQA.py \
    --output_dir=$OUTPUT_DIR \
    --model_type=bert \
    --model_checkpoint=bert-base-chinese \
    --train_file=../../data/medical_qa/train.json \
    --dev_file=../../data/medical_qa/dev.json \
    --test_file=../../data/medical_qa/test.json \
    --max_length=512 \
    --max_answer_length=100 \
    --stride=256 \
    --n_best=25 \
    --learning_rate=2e-5 \
    --num_train_epochs=3 \
    --batch_size=4 \
    --warmup_proportion=0.1 \
    --do_train \
    --do_test \
    --seed=42
```

### Step 3：验证数据加载

创建 `test_medical_data.py`：

```python
import sys
import json
sys.path.append('.')

from src.sequence_labeling_extractiveQA_cmrc.data import CMRC2018

# 加载训练集
dataset = CMRC2018('data/medical_qa/train.json')
print(f"✓ 数据集大小: {len(dataset)}")

# 查看第一条样本
sample = dataset[0]
print(f"✓ 样本ID: {sample['id']}")
print(f"✓ 问题: {sample['question']}")
print(f"✓ 原文长度: {len(sample['context'])} 字符")
print(f"✓ 答案文本: {sample['answers']['text']}")
print(f"✓ 答案位置: {sample['answers']['answer_start']}")

# 验证答案位置正确性
context = sample['context']
answer_start = sample['answers']['answer_start'][0]
answer_text = sample['answers']['text'][0]
extracted_text = context[answer_start:answer_start+len(answer_text)]
assert extracted_text == answer_text, f"答案位置错误！期望: {answer_text}, 实际: {extracted_text}"
print(f"✓ 答案位置验证通过")
```

运行：
```bash
python test_medical_data.py
```

输出应该像：
```
✓ 数据集大小: 2000
✓ 样本ID: medical_q_001
✓ 问题: 高血压的主要特征是什么？
✓ 原文长度: 450 字符
✓ 答案文本: ['体循环动脉血压升高']
✓ 答案位置: [7]
✓ 答案位置验证通过
```

### Step 4：运行训练

```bash
cd /Users/huwei/PyCharmMiscProject/How-to-use-Transformers/src/sequence_labeling_extractiveQA_cmrc/
chmod +x run_medical_qa.sh
bash run_medical_qa.sh
```

### Step 5：查看结果

```bash
ls -la medical_qa_results/
```

输出目录结构：
```
medical_qa_results/
├── epoch_1_dev_f1_xx.xx_em_yy.yy_weights.bin
├── epoch_2_dev_f1_xx.xx_em_yy.yy_weights.bin
├── epoch_3_dev_f1_xx.xx_em_yy.yy_weights.bin
├── train.log
├── test.log
└── args.txt
```

---

## 六、改动清单（快速参考）

### 场景 1：换自己的 QA 数据集（同 SQuAD 格式）

| 文件 | 改什么 | 具体位置 |
| --- | --- | --- |
| shell 脚本 | 数据路径 | `--train_file / --dev_file / --test_file` |
| shell 脚本 | 超参数调整 | `--max_length / --stride / --batch_size / --max_answer_length` |
| 其他 | **不用改** | `data.py / modeling.py / run_extractiveQA.py` 全部原样 |

### 场景 2：自定义数据格式

| 文件 | 改什么 | 具体位置 |
| --- | --- | --- |
| `data.py` | `load_data` 方法 | 修改 JSON 字段解析逻辑 |
| 其他 | **不用改** | 其他代码原样 |

### 场景 3：换预训练模型

| 文件 | 改什么 | 具体位置 | 改法 |
| --- | --- | --- | --- |
| shell 脚本 | checkpoint | `--model_checkpoint` | 直接换（同架构） |
| `modeling.py` | 新增 Model 类 | 文件末尾 | 新增 `RobertaForExtractiveQA` 等 |
| `run_extractiveQA.py` | `MODEL_CLASSES` | 第 XX 行 | 加上新模型的映射 |
| shell 脚本 | model_type | `--model_type=` | 改成新类型 |

### 场景 4：调参数

| 参数 | 改哪里 | 调整建议 |
| --- | --- | --- |
| 序列长度 | `--max_length` | 文章长 → 512+；文章短 → 256 |
| 重叠范围 | `--stride` | 通常 max_length 的 1/4 ~ 1/2 |
| 答案长度 | `--max_answer_length` | 领域相关；医学可能 100+；新闻通常 30 |
| 候选数 | `--n_best` | 精度优先 → 25+；速度优先 → 10 |
| 学习率 | `--learning_rate` | 通常 1e-5 ~ 3e-5；医学领域可能需要 2e-5 |
| batch 大小 | `--batch_size` | 显存 16GB → 4；8GB → 2 |

---

## 七、核心概念：offset_mapping 和 stride

初学者容易迷糊的两个东西。

### offset_mapping：字符 → token 的映射

CMRC 数据里，`answer_start` 是**字符位置**（比如 42 表示第 42 个字符）。但模型需要 **token 位置**（比如第 10 个 token）。

`collate_fn` 里有一段代码就是做这个转换：

```python
# tokenizer 返回 offset_mapping：每个 token 对应原文的字符范围
encodings = tokenizer(question, context, ..., return_offsets_mapping=True)
offset_mapping = encodings['offset_mapping']

# offset_mapping[i] = (char_start, char_end)
# 比如 token 5 对应字符范围 (20, 25)

# 遍历 offset_mapping，找到包含 answer_start 的 token
for idx, (start, end) in enumerate(offset_mapping):
    if start <= answer_char_start < end:
        answer_token_start = idx  # ← 找到了！
        break
```

**你不用手写这个逻辑**——本项目的 `data.py` 已经封装好了。你只需确保 JSON 里的 `answer_start` 是**正确的字符位置**。

### stride：处理超长文本

当文章太长（超过 `max_length`）时，直接截断会丢失答案。解决方案：把文章分成多个"块"（chunk），相邻块有重叠。

```
文章（1000 个 token）：[0 1 2 3 4 5 6 ... 999]

max_length=512, stride=128：

chunk 1: [0 1 2 ... 511]
chunk 2: [384 385 ... 895]  ← 与 chunk 1 重叠 128 个 token
chunk 3: [768 769 ... 1000] ← 与 chunk 2 重叠 128 个 token
```

推理时，对每个 chunk 都预测 start/end logits，然后把所有 chunk 的预测结合起来，选最好的答案。

**你不用手写这个逻辑**——`data.py` 和 `run_extractiveQA.py` 已经处理好了。你只需在 shell 脚本里调 `--stride` 参数：

- `stride` 大（比如 256）→ chunks 多 → 精度高但慢
- `stride` 小（比如 64）→ chunks 少 → 快但可能漏掉答案

---

## 八、常见坑

1. **答案位置错误，training 不工作**。检查 JSON 里的 `answer_start` 是否是**字符位置**（不是 token 位置）。用 `test_medical_data.py` 验证。

2. **跑了 3 个 epoch，F1 还是 0%**。可能原因：
   - 数据格式错误（不是标准 SQuAD）
   - 答案位置都对应不上（collate_fn 里转换失败）
   - 数据太少（< 100 条样本）
   
   排查：在 `run_extractiveQA.py` 的 training loop 里加 `print` 看看 `start_positions` 和 `end_positions` 是否有效数值。

3. **stride 参数搞反了，显存爆炸**。`stride` 太大（比如 512）可能导致 chunks 数量激增，显存压力大。改小（128）试试。

4. **答案跨越了多个 chunks，被截断了**。这是 stride 处理的代价。如果答案经常被截断，加大 `stride`（让重叠更多）。

5. **预测时输出的答案位置不合理**（比如 end < start）。这是推理逻辑加的约束：
   ```python
   if end_pos < start_pos:
       continue  # 跳过不合理的 (start, end) 组合
   if end_pos - start_pos > max_answer_length:
       continue  # 跳过过长的答案
   ```

   如果很多答案都被过滤，增大 `--max_answer_length`。

6. **`--do_predict` 保存的预测结果是什么格式**？结果文件里每行是一个 JSON：
   ```json
   {"id": "question_id", "prediction_text": "提取出的答案", "nbest_predictions": [...]}
   ```

---

## 九、改造全流程 Checklist

照着这个 checklist，30 分钟改好一个新任务：

- [ ] 准备数据目录 `data/my_task/`
- [ ] 把数据转成 SQuAD JSON 格式（或修改 `data.py` 解析自定义格式）
- [ ] 验证 JSON 格式：运行 `test_medical_data.py` 确保 `answer_start` 正确
- [ ] 新建 shell 脚本 `run_my_task.sh`
- [ ] 在 shell 脚本里调整 `--max_length / --stride / --batch_size / --max_answer_length`
- [ ] 运行 `bash run_my_task.sh --do_train`
- [ ] 等待训练完成，查看 `train.log`
- [ ] 运行 `bash run_my_task.sh --do_test`，看 F1 和 EM 分数
- [ ] 如果效果不好，微调超参数重来

---

## 十、下一步可以读什么

- 本项目 `cmrc2018_evaluate.py` —— 自定义的 F1/EM 评估脚本
- 本仓库 `src/sequence_labeling_ner_cpd/` —— token 级任务的另一个例子（NER）
- HuggingFace 官方 [QA examples](https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering) —— 支持更多高级特性

理解了本目录的代码 + 本改造指南，你就能：
1. 改造任何 SQuAD 格式的 QA 数据集
2. 调整长文本处理参数
3. 换不同的预训练模型
4. 排查 token 级任务的常见问题

祝改造顺利！
