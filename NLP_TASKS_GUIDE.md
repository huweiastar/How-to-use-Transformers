# NLP 任务全景指南：从任务理解到代码实现

> 本文档总结了本仓库中 7 个经典 NLP 任务的完整学习路径。读完本文，当你遇到一个新的 NLP 任务时，能够快速判断它属于哪种范式，知道该用什么模型、怎么写代码。

---

## 目录

1. [NLP 任务全景图](#1-nlp-任务全景图)
2. [句对分类——语义相似度](#2-句对分类语义相似度-pairwise-classification)
3. [文本分类（Prompt 范式）——情感分析](#3-文本分类prompt-范式情感分析)
4. [序列标注——命名实体识别（NER）](#4-序列标注命名实体识别-ner)
5. [抽取式问答（MRC）](#5-抽取式问答-extractive-qa--mrc)
6. [生成式问答](#6-生成式问答-generative-qa)
7. [机器翻译（Seq2Seq）](#7-机器翻译-seq2seq)
8. [文本摘要（Seq2Seq）](#8-文本摘要-seq2seq)
9. [横向对比：选模型的决策树](#9-横向对比选模型的决策树)
10. [通用代码框架模板](#10-通用代码框架模板)

---

## 1. NLP 任务全景图

所有 NLP 任务可以按照**输入输出结构**分为三大范式：

```
┌─────────────────────────────────────────────────────────────┐
│                     NLP 任务范式                             │
├───────────────┬────────────────────┬────────────────────────┤
│  分类范式      │   序列标注范式      │   生成范式              │
│  (N → 1)      │   (N → N)          │   (N → M)              │
├───────────────┼────────────────────┼────────────────────────┤
│ • 文本分类     │ • 命名实体识别(NER) │ • 文本摘要              │
│ • 句对分类     │ • 抽取式问答*       │ • 机器翻译              │
│ • 情感分析     │ • 词性标注          │ • 生成式问答            │
│ • 意图识别     │ • 关系抽取          │ • 文本续写              │
└───────────────┴────────────────────┴────────────────────────┘

*抽取式问答本质上是对每个 token 预测 start/end 概率，也属于序列标注
```

**模型选择原则：**
- 分类/序列标注任务 → **Encoder-only 模型**（BERT、RoBERTa）：理解能力强
- 生成/翻译/摘要任务 → **Encoder-Decoder 模型**（T5、mT5、MarianMT）：生成能力强

---

## 2. 句对分类——语义相似度 (Pairwise Classification)

### 📁 项目位置
`src/pairwise_cls_similarity_afqmc/`

### 🎯 任务定义

**输入**：两个句子（sentence1, sentence2）
**输出**：二分类标签（0 = 不相似，1 = 相似）

**典型应用场景**：智能客服去重、搜索引擎问题匹配、FAQ 自动检索

### 📦 数据集：AFQMC（蚂蚁金融语义相似度）

| 切分 | 数量 | 说明 |
|------|------|------|
| 训练集 | 34,334 | 有标签 |
| 验证集 | 4,316  | 有标签 |
| 测试集 | 3,861  | 无标签（CLUE 官方评测） |

```json
{"sentence1": "花呗如何还款", "sentence2": "花呗怎么还款", "label": "1"}
{"sentence1": "双十一花呗提额在哪", "sentence2": "哪里可以提花呗额度", "label": "0"}
```

**难点**：困难负样本——字面高度重叠但语义截然不同，传统 BM25 失效。

### 🏗️ 模型架构

```
[CLS] sentence1 [SEP] sentence2 [SEP]
         ↓  BERT Encoder
last_hidden_state[:, 0, :]   ← 取 [CLS] 位置作为句对整体表示
         ↓  Dropout
         ↓  Linear(768 → 2)
      logits [batch, 2]
         ↓  CrossEntropyLoss
```

**关键：为什么用 [CLS]？**
BERT 的 [CLS] 在预训练时专门被训练成聚合整段输入语义的向量（Next Sentence Prediction 任务），天然适合句子/句对级别的分类。

### 💻 核心代码

**数据加载（句对拼接编码）：**
```python
def collote_fn(batch_samples):
    batch_s1 = [s['sentence1'] for s in batch_samples]
    batch_s2 = [s['sentence2'] for s in batch_samples]
    # tokenizer 自动拼成 [CLS] sent1 [SEP] sent2 [SEP]
    # token_type_ids: sent1 部分=0, sent2 部分=1
    X = tokenizer(batch_s1, batch_s2, padding=True, truncation=True, return_tensors="pt")
    y = torch.tensor([int(s['label']) for s in batch_samples])
    return X, y
```

**模型定义（推荐 BertPreTrainedModel 写法）：**
```python
class BertForPairwiseCLS(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)  # 不用内置 pooler
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 2)
        self.post_init()  # 重要：正确初始化新加的层

    def forward(self, x):
        output = self.bert(**x)
        cls_vec = output.last_hidden_state[:, 0, :]  # [batch, hidden]
        return self.classifier(self.dropout(cls_vec))
```

### 📊 评估指标
- **Accuracy**（准确率）
- **Macro-F1** + **Weighted-F1**（处理类别不均衡）

---

## 3. 文本分类（Prompt 范式）——情感分析

### 📁 项目位置
`src/text_cls_prompt_senti_chnsenticorp/`

### 🎯 任务定义

**输入**：一段评论文本
**输出**：情感极性（正面 / 负面）

**Prompt 范式 vs 传统分类范式的区别：**

| | 传统分类 | Prompt 分类 |
|---|---|---|
| 输入 | 原文 | 原文 + 模板（含 [MASK]） |
| 输出头 | Linear(768→2) | MLM Head（预测词表） |
| 决策依据 | [CLS] 向量 | [MASK] 位置预测的词 |
| 优点 | 简单直接 | 更贴近预训练目标，少样本效果好 |

### 📦 数据集：ChnSentiCorp（中文酒店评论）

```
正面: 房间宽敞，服务很好，早餐种类多，下次还会入住。
负面: 床很硬，隔音很差，服务态度很差，不推荐！
```

格式：TSV 文件，`label\ttext`

### 🏗️ 模型架构

**核心思路：将分类问题转化为完形填空**

```
原始输入:  "这家酒店很不错，推荐！"
Prompt:   "总体上来说很[MASK]。这家酒店很不错，推荐！"
              ↓  BERT MLM Head
         预测 [MASK] 处的词语
              ↓  Verbalizer（词语→类别的映射）
         "好" → 正面(1)
         "差" → 负面(0)
```

**Verbalizer（标签词映射器）：**
```python
def get_verbalizer(tokenizer):
    return {
        'pos': tokenizer.convert_tokens_to_ids('好'),  # 正面 → "好"
        'neg': tokenizer.convert_tokens_to_ids('差'),  # 负面 → "差"
    }
```

**模型（使用 BERT 的 MLM 头）：**
```python
class BertForPrompt(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)  # 完整 MLM 预测头
        self.post_init()

    def forward(self, batch_inputs, batch_mask_idxs, label_word_id, labels=None):
        output = self.bert(**batch_inputs)
        seq = output.last_hidden_state              # [batch, seq_len, hidden]
        # 只取 [MASK] 位置的表示
        mask_reps = seq[range(len(seq)), batch_mask_idxs]  # [batch, hidden]
        pred_scores = self.cls(mask_reps)[:, label_word_id]  # 只保留标签词的分数
        loss = nn.CrossEntropyLoss()(pred_scores, labels) if labels is not None else None
        return loss, pred_scores
```

### 💻 关键技巧：找 [MASK] 的位置

```python
def get_mask_index(encoding):
    # char_to_token 将字符位置映射到 token 位置
    # Prompt 模板 "总体上来说很[MASK]。" 中 [MASK] 在第 6 个字符
    return encoding.char_to_token(6)
```

### 📊 评估指标
Accuracy、Macro-F1、Weighted-F1

---

## 4. 序列标注——命名实体识别 (NER)

### 📁 项目位置
`src/sequence_labeling_ner_cpd/`

### 🎯 任务定义

**输入**：一段文本（逐字）
**输出**：每个字的实体标签

**典型应用**：信息抽取、知识图谱构建、搜索引擎理解

### 📦 数据集：人民日报 NER 语料

**标注格式：IOB2（Inside-Outside-Beginning）**
```
人  B-PER
民  I-PER
日  B-ORG
报  I-ORG
北  B-LOC
京  I-LOC
```

实体类型：`PER`（人名）、`ORG`（机构名）、`LOC`（地名）

**IOB2 规则：**
- `B-X`：实体 X 的首字
- `I-X`：实体 X 的续字
- `O`：不属于任何实体

### 🏗️ 模型架构（两种方案）

**方案 A：Softmax（简单，训练快）**
```
token序列 → BERT Encoder → [batch, seq, hidden]
                         → Linear(hidden → num_labels)
                         → [batch, seq, num_labels]
                         → Softmax → 每个 token 的标签概率
```

**方案 B：CRF（考虑标签间依赖，精度更高）**
```
token序列 → BERT Encoder → [batch, seq, hidden]
                         → Linear → emissions
                         → CRF Layer（学习转移矩阵 B-PER → I-PER 合法）
                         → Viterbi 解码 → 全局最优标签序列
```

**何时用 CRF？** 当你需要避免 `O → I-PER` 这种非法转移时。

```python
class BertForNER(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, args.num_labels)
        self.post_init()

    def forward(self, batch_inputs, labels=None):
        output = self.bert(**batch_inputs)
        seq_out = self.dropout(output.last_hidden_state)  # [batch, seq, hidden]
        logits = self.classifier(seq_out)                 # [batch, seq, num_labels]

        loss = None
        if labels is not None:
            # 只对非 [PAD] 的位置计算损失
            active = batch_inputs['attention_mask'].view(-1) == 1
            loss = CrossEntropyLoss()(logits.view(-1, num_labels)[active],
                                      labels.view(-1)[active])
        return loss, logits
```

### 💻 关键技巧：字符到 token 的位置映射

BERT 的中文分词以字为单位，但原始标注也以字为单位，**不会产生对齐问题**。
但如果是英文（WordPiece 切分），需要特殊处理：

```python
# 只对每个词的第一个 subword 设置标签，其余设 -100（忽略）
word_ids = encoding.word_ids()
previous_word_idx = None
for word_idx in word_ids:
    if word_idx is None or word_idx == previous_word_idx:
        label_ids.append(-100)  # 忽略
    else:
        label_ids.append(labels[word_idx])
    previous_word_idx = word_idx
```

### 📊 评估指标

使用 `seqeval` 库进行**实体级别**评估（不是 token 级别）：
```python
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2
# 输入是完整的实体序列，自动按实体边界计算
report = classification_report(true_labels, pred_labels, mode='strict', scheme=IOB2)
```

| 指标 | 含义 |
|------|------|
| Precision | 预测为实体的里，有多少是对的 |
| Recall | 真实实体里，有多少被找到了 |
| F1 | 两者的调和平均 |

---

## 5. 抽取式问答 (Extractive QA / MRC)

### 📁 项目位置
`src/sequence_labeling_extractiveQA_cmrc/`

### 🎯 任务定义

**输入**：问题（question）+ 文章（context）
**输出**：文章中的一段文字（answer span）—— 答案一定在原文中

**典型应用**：智能搜索、阅读理解测试评分、文档问答系统

### 📦 数据集：CMRC 2018（中文机器阅读理解）

```json
{
  "question": "姚明在哪个球队效力？",
  "context": "...... 姚明于 2002 年加入 NBA 休斯顿火箭队，成为首位在 NBA 首轮第一顺位被选中的亚洲球员 ......",
  "answers": [{"text": "休斯顿火箭队", "answer_start": 15}]
}
```

### 🏗️ 模型架构

```
[CLS] question [SEP] context [SEP]
         ↓  BERT Encoder
sequence_output [batch, seq_len, hidden]
         ↓  Linear(hidden → 2)
[start_logits, end_logits]  各自 [batch, seq_len]
         ↓  argmax
answer = context[start_pos : end_pos+1]
```

### 💻 关键技巧 1：处理超长文档（Stride 滑动窗口）

当 question + context 超过 512 token 时，用**滑动窗口**切分：

```python
tokenizer(
    question, context,
    max_length=512,
    stride=128,          # 相邻窗口重叠 128 token（保证答案不被截断）
    truncation="only_second",   # 只截断 context
    return_overflowing_tokens=True,  # 返回所有窗口
    return_offsets_mapping=True      # 返回字符偏移量（用于还原答案位置）
)
```

### 💻 关键技巧 2：Offset Mapping（token ↔ 字符位置映射）

```python
# offset_mapping[i] = (char_start, char_end) 表示第 i 个 token 对应原文的字符范围
# 用于将预测的 token 位置还原为原文中的字符串
offset = offset_mapping[start_token_idx]
answer_text = context[offset[0] : offset_end[1]]
```

### 💻 关键技巧 3：N-Best 搜索（提高答案质量）

不直接取 argmax，而是遍历所有合法的 (start, end) 组合，取得分最高的：

```python
best_answer = None
best_score = -float('inf')
for start_idx in range(len(context_tokens)):
    for end_idx in range(start_idx, min(start_idx + max_answer_length, len(context_tokens))):
        score = start_logits[start_idx] + end_logits[end_idx]
        if score > best_score:
            best_score = score
            best_answer = context[offset[start_idx][0] : offset[end_idx][1]]
```

### 📊 评估指标

| 指标 | 含义 |
|------|------|
| **F1**（字符级） | 预测答案与真实答案的公共字符比例（允许部分匹配） |
| **EM**（Exact Match） | 完全匹配率（去除标点后完全相同才算对） |

```python
# F1：基于最长公共子序列（LCS）
def compute_f1(prediction, ground_truth):
    pred_chars = list(prediction)
    gold_chars = list(ground_truth)
    common = Counter(pred_chars) & Counter(gold_chars)
    num_same = sum(common.values())
    precision = num_same / len(pred_chars)
    recall    = num_same / len(gold_chars)
    return 2 * precision * recall / (precision + recall)
```

---

## 6. 生成式问答 (Generative QA)

### 📁 项目位置
`src/sequence_labeling_generativeQA_DuReaderQG/`

### 🎯 任务定义

**输入**：问题（question）+ 文章（context）
**输出**：自由生成的答案字符串（不限于原文中的片段）

**与抽取式问答的核心区别：**

| | 抽取式 | 生成式 |
|---|---|---|
| 答案来源 | 必须是原文中的文字片段 | 可以是模型自由生成的文字 |
| 模型类型 | BERT（Encoder-only） | T5（Encoder-Decoder） |
| 任务建模 | Token 分类（start/end 位置） | Seq2Seq 生成 |
| 适用场景 | 答案在原文中有明确边界 | 需要推理、总结、改写 |

### 📦 数据集：DuReaderQG

百度开放的中文阅读理解数据集，包含文章、问题和参考答案三元组。

### 🏗️ 模型架构

```
Encoder:  [question] [SEP] [context]
                ↓  T5 Encoder
          encoder_hidden_states
                ↓
Decoder:  [BOS] → token1 → token2 → ... → [EOS]
                ↑  T5 Decoder (cross-attention 关注 Encoder 输出)
```

**训练（Teacher Forcing）：** Decoder 输入是真实答案右移一位
**推理（Beam Search）：** 从 [BOS] 自回归逐 token 生成，保留前 k 条候选

```python
# 训练时
labels = tokenizer(answer, return_tensors="pt")["input_ids"]
decoder_input_ids = model.prepare_decoder_input_ids_from_labels(labels)
# 推理时
generated = model.generate(
    input_ids=input_ids,
    num_beams=4,           # Beam Search 宽度
    max_length=64
)
```

### 📊 评估指标：BLEU

```python
from nltk.translate.bleu_score import corpus_bleu
# BLEU-1~4：n-gram 精确率
# BLEU-4 是最常用的摘要/生成质量指标
bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))
```

---

## 7. 机器翻译 (Seq2Seq)

### 📁 项目位置
`src/seq2seq_translation/`

### 🎯 任务定义

**输入**：源语言句子（中文）
**输出**：目标语言句子（英文）

### 📦 数据集：Translation2019zh（中英翻译语料）

格式：JSON Lines，每行 `{"chinese": "...", "english": "..."}`

220 万条样本，实验取前 20 万条。

### 🏗️ 模型：MarianMT（专门的机器翻译模型）

```python
from transformers import MarianMTModel, MarianTokenizer
model_name = "Helsinki-NLP/opus-mt-zh-en"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)
```

MarianMT 是专门为翻译任务训练的 Encoder-Decoder 模型，Helsinki NLP 团队维护了数百个语言对。

### 💻 关键技巧：目标语言的编码方式

```python
def collote_fn(batch_samples):
    batch_zh = [s['chinese'] for s in batch_samples]
    batch_en = [s['english'] for s in batch_samples]

    inputs = tokenizer(batch_zh, padding=True, truncation=True, return_tensors="pt")

    # 目标语言用 text_target 参数，而不是 text
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(batch_en, padding=True, truncation=True, return_tensors="pt")["input_ids"]

    inputs['decoder_input_ids'] = model.prepare_decoder_input_ids_from_labels(labels)
    # EOS 之后的 padding 标记为 -100，不计入损失
    eos_positions = torch.where(labels == tokenizer.eos_token_id)[1]
    for i, pos in enumerate(eos_positions):
        labels[i][pos+1:] = -100
    inputs['labels'] = labels
    return inputs
```

### 📊 评估指标：SacreBLEU（语料级）

```python
from sacrebleu.metrics import BLEU
bleu = BLEU()
# 语料级 BLEU：同时传入所有预测和参考，比单句平均更准
score = bleu.corpus_score(predictions, [references])
```

---

## 8. 文本摘要 (Seq2Seq)

### 📁 项目位置
`src/seq2seq_summarization/`

### 🎯 任务定义

**输入**：长文本（微博正文，约 100 字）
**输出**：精炼摘要（约 15 字）

**抽取式 vs 生成式摘要：**
- **抽取式**：从原文中选出最重要的句子拼接（不需要 Seq2Seq）
- **生成式**：理解语义后重新生成（本任务，更难、效果更好）

### 📦 数据集：LCSTS（新浪微博摘要数据集）

| 切分 | 总数 | 说明 |
|------|------|------|
| Part I（训练集） | ~240 万 | 自动配对，有噪声 |
| Part II（验证集） | 10,666 | 人工评分 |
| Part III（测试集） | 1,106 | 高质量精标 |

**⚠️ 重要：** Part II/III 需要过滤 score < 3 的样本（低质数据），否则 ROUGE 指标失真。

格式：`摘要!=!正文`（tsv，感叹号分隔）

### 🏗️ 模型：mT5（多语言 T5）

```python
from transformers import AutoModelForSeq2SeqLM
model = AutoModelForSeq2SeqLM.from_pretrained(
    "csebuetnlp/mT5_multilingual_XLSum",
    tie_word_embeddings=False  # 不共享 encoder/decoder embedding，对中文更好
)
```

**mT5 的分词（SentencePiece Unigram）：**
```python
tokenizer("我叫张三") → ['▁', '我', '叫', '张', '三', '</s>']
# ▁ 表示词首，</s> 表示序列结束
# 无需依赖中文分词工具，多语言通用
```

### 💻 推理：Beam Search 解码

```python
def summarize(text, model, tokenizer, args):
    inputs = tokenizer(text, max_length=512, truncation=True, return_tensors="pt").to(args.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            num_beams=4,          # Beam Search 宽度（越大质量越好，但越慢）
            max_length=64,
            early_stopping=True   # 生成 EOS 时提前停止
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)
```

### 📊 评估指标：ROUGE

```python
from rouge import Rouge
rouge = Rouge()
scores = rouge.get_scores(predictions, references, avg=True)
# ROUGE-1: unigram 重叠率
# ROUGE-2: bigram 重叠率
# ROUGE-L: 最长公共子序列
```

---

## 9. 横向对比：选模型的决策树

```
遇到新任务
    │
    ├── 输入是句子/句对，输出是类别？
    │       ├── 单句 → BERT + [CLS] + Linear → 文本分类
    │       ├── 句对 → BERT(sent1, sent2) + [CLS] + Linear → 句对分类
    │       └── 数据量少？→ Prompt + MLM Head（Few-shot 效果好）
    │
    ├── 输入是句子，输出是每个 token 的标签？
    │       ├── 标签独立 → BERT + Linear + Softmax → NER/词性标注
    │       └── 标签有依赖（如 IOB2）→ BERT + Linear + CRF → NER（更高精度）
    │
    ├── 输入是问题+文章，输出是答案？
    │       ├── 答案必须来自原文 → BERT + start/end 位置预测 → 抽取式 QA
    │       └── 答案可以自由生成 → T5 Seq2Seq → 生成式 QA
    │
    └── 输入是长文本，输出是另一种文本？
            ├── 翻译（语言对固定）→ MarianMT
            ├── 摘要（中文压缩）→ mT5
            └── 通用生成 → T5 / GPT

```

---

## 10. 通用代码框架模板

所有任务的代码结构高度一致，掌握这个框架后，换任务只需替换关键部分。

### 10.1 超参数配置（Args 类）

```python
class Args:
    model_checkpoint = "bert-base-chinese"  # 或其他模型
    batch_size    = 16
    learning_rate = 1e-5
    epoch_num     = 3
    warmup_ratio  = 0.1

    # 设备检测：优先 MPS（Mac GPU）> CUDA > CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

args = Args()
```

### 10.2 优化器配置（AdamW + 权重衰减分组）

```python
# bias 和 LayerNorm 不做 weight decay，其他参数做 weight decay=0.01
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

total_steps  = args.epoch_num * len(train_dataloader)
warmup_steps = int(args.warmup_ratio * total_steps)
lr_scheduler = get_scheduler("linear", optimizer=optimizer,
                              num_warmup_steps=warmup_steps,
                              num_training_steps=total_steps)
```

### 10.3 训练函数模板

```python
def train_loop(dataloader, model, optimizer, lr_scheduler, epoch, total_loss):
    progress_bar = tqdm(range(len(dataloader)))
    finish_batch_num = (epoch - 1) * len(dataloader)

    model.train()
    for step, batch_data in enumerate(dataloader, start=1):
        batch_data = batch_data.to(args.device)
        outputs = model(**batch_data)
        loss = outputs.loss  # 或 outputs[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        progress_bar.set_description(f'loss: {total_loss / (finish_batch_num + step):>7f}')
        progress_bar.update(1)
    return total_loss
```

### 10.4 训练主循环（带最优模型保存）

```python
history = {'train_loss': [], 'valid_metric': []}
best_metric = 0.
best_model_path = 'best_model.bin'

for epoch in range(1, args.epoch_num + 1):
    print(f'Epoch {epoch}/{args.epoch_num}')
    total_loss = train_loop(train_dataloader, model, optimizer, lr_scheduler, epoch, 0)
    history['train_loss'].append(total_loss / len(train_dataloader))

    metric = test_loop(valid_dataloader, model)  # 返回主要评估指标
    history['valid_metric'].append(metric)

    if metric > best_metric:
        best_metric = metric
        torch.save(model.state_dict(), best_model_path)
        print(f'New best! metric: {metric:.4f}')
```

### 10.5 绘制训练曲线

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history['train_loss'], 'o-', label='Train Loss')
axes[0].set_title('Training Loss')
axes[1].plot(history['valid_metric'], 's-', color='orange', label='Valid Metric')
axes[1].set_title('Validation Metric')
plt.tight_layout(); plt.show()
```

### 10.6 加载最优模型并预测

```python
# 加载最优权重
model.load_state_dict(torch.load(best_model_path, map_location=args.device))
model.eval()

# 预测（以分类任务为例）
def predict(text, model, tokenizer, args):
    inputs = tokenizer(text, return_tensors="pt").to(args.device)
    with torch.no_grad():
        logits = model(inputs)
    return logits.argmax(dim=-1).item()
```

---

## 附录：各任务关键依赖库速查

| 库 | 用途 | 安装 |
|---|---|---|
| `transformers` | 加载预训练模型、分词器 | `pip install transformers` |
| `torch` | 深度学习框架 | `pip install torch` |
| `seqeval` | NER 实体级 F1 评估 | `pip install seqeval` |
| `rouge` | 摘要 ROUGE 评估 | `pip install rouge` |
| `sacrebleu` | 翻译 BLEU 评估 | `pip install sacrebleu` |
| `sklearn` | 分类 Accuracy/F1 | `pip install scikit-learn` |
| `nltk` | BLEU（句子级） | `pip install nltk` |
| `tqdm` | 训练进度条 | `pip install tqdm` |
| `matplotlib` | 绘制训练曲线 | `pip install matplotlib` |

---

## 附录：各任务汇总速查表

| 任务 | 数据集 | 预训练模型 | 输出头 | 评估指标 |
|------|--------|-----------|--------|---------|
| 句对分类 | AFQMC | bert-base-chinese | Linear(768→2) | Accuracy, F1 |
| Prompt 情感分析 | ChnSentiCorp | bert-base-chinese | MLM Head（词表预测） | Accuracy, F1 |
| NER（Softmax） | 人民日报 | bert-base-chinese | Linear(768→7) + Softmax | seqeval F1 |
| NER（CRF） | 人民日报 | bert-base-chinese | Linear + CRF | seqeval F1 |
| 抽取式 QA | CMRC 2018 | bert-base-chinese | Linear(768→2)→start/end | F1, EM |
| 生成式 QA | DuReaderQG | Mengzi-T5-Base | Seq2Seq Decoder | BLEU-1~4 |
| 机器翻译 | Translation2019zh | Helsinki-NLP/opus-mt-zh-en | Seq2Seq Decoder | SacreBLEU |
| 文本摘要 | LCSTS | csebuetnlp/mT5_multilingual_XLSum | Seq2Seq Decoder | ROUGE-1/2/L |