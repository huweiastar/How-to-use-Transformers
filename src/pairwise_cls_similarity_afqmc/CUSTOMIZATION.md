# 改造指南：把 AFQMC 项目迁移到你自己的数据 / 任务 / 场景

> 配套文件：本目录的 `readme.md`（项目介绍）+ `pipeline.py`（教学单文件）+ `data.py / modeling.py / arg.py / run_simi_cls.py`（工程化版本）。
>
> 读完本指南，你应该能在 **30 分钟内** 把这套代码改造成任意一个"文本 → 标签"类型的微调项目。

---

## 〇、先把心智模型画出来

任何一个基于 Transformers 的微调项目，都可以被拆成 **5 个可替换的零件**：

```
        ┌───────────────┐   ┌────────────────┐   ┌──────────────┐   ┌───────────┐   ┌──────────┐
 原始   │ Dataset 读文件│   │  collate_fn    │   │ Pretrained    │   │ Task Head │   │  Loss    │
 数据 ─>│  (data.py)    │─> │ 分词+padding   │─> │   (encoder)   │─> │分类/回归头│─> │ CE/MSE.. │
        └───────────────┘   └────────────────┘   └──────────────┘   └───────────┘   └──────────┘
                                                                         ▲                ▲
                                                                       改这里          改这里
```

改造的本质就是：**你换了哪个零件，就只动对应文件里那几行。** 下面每一小节都会精确地告诉你"动哪里、改成什么"。

对照表：

| 零件 | 对应文件 / 函数 | 典型改动 |
| --- | --- | --- |
| Dataset | `data.py: AFQMC.__init__/load_data` | 换数据格式（csv / tsv / jsonl / HuggingFace datasets） |
| collate_fn | `data.py: get_dataLoader.collote_fn` | 单句 vs 句对；标签是 int / float / list |
| Encoder | `modeling.py: self.bert = BertModel(...)` | 换预训练模型（Bert → RoBERTa / ERNIE / mT5 …） |
| Task Head | `modeling.py: self.classifier = nn.Linear(...)` | 分类头维度 / 回归头 / token 级输出头 |
| Loss | `modeling.py: forward` 里的 `CrossEntropyLoss()` | CE / BCE / MSE / 自定义 |

---

## 一、场景 A：**换数据集，任务不变**（还是句对二分类）

最常见的情形。比如你有一个"标题 + 正文是否匹配"的自有数据集。

### 改动步骤

1. **统一数据格式** 把你的数据转成和 AFQMC 一样的 JSON Lines：
   ```json
   {"sentence1": "...", "sentence2": "...", "label": "0"}
   ```
   如果字段名不一样（例如 `query/doc/score`），二选一：

   - **推荐**：写个小脚本一次性转成 AFQMC 格式，代码一行不用改。
   - **或者**：改 `data.py:load_data` 里取字段的两行 + `collote_fn` 里的 `sample['sentence1']` 等 key 名。

2. **替换数据路径**（只动 shell 脚本）：
   ```bash
   --train_file=../../data/你的数据/train.json \
   --dev_file=../../data/你的数据/dev.json \
   --test_file=../../data/你的数据/test.json \
   ```

3. **调 max_seq_length**：如果你的句子比 AFQMC 长很多，要么加大 `--max_seq_length=512`，要么缩小 `--batch_size`（显存和 `seq_len²` 成正比）。

4. **不用改**：模型、分类头、损失、训练循环、保存策略——全部原样复用。

### 如果数据量变大很多（百万级）

`data.py` 里现在是把整份文件读到内存的 `dict`。当数据过大时，把 `AFQMC` 改成 `IterableDataset`（文件末尾有注释版的 `IterableAFQMC` 可以直接启用）。注意：`IterableDataset` 下 `shuffle=True` 无效，需要自己做 shuffle buffer。

---

## 二、场景 B：**单句分类**（情感分析 / 意图识别 / 新闻分类）

比如 ChnSentiCorp 情感分析：输入一段评论，输出"正面/负面"。

### 改动步骤

1. **Dataset**（`data.py`）：把样本字段改成只有 `text + label`。

2. **collate_fn**（`data.py` 里 `get_dataLoader` 内部）：
   ```python
   # 改之前
   batch_inputs = tokenizer(batch_sentence_1, batch_sentence_2, ...)
   # 改之后
   batch_texts = [s['text'] for s in batch_samples]
   batch_inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")
   ```

3. **模型**：**完全不用改**。`BertForPairwiseCLS` 本来就是 `取 [CLS] → Linear → logits`，单句 / 句对对模型都是透明的（差别只在 tokenizer 怎么拼）。

4. **num_labels**：如果类别数不是 2，在 `run_simi_cls.py` 里把 `args.num_labels = 2` 改成对应数字（比如情感三分类就写 3）。或者把它提到 `arg.py` 里变成命令行参数更干净。

5. **改个名字**：`BertForPairwiseCLS` 现在是"句对分类"；单句用同一个类没问题，但想读着舒服可以重命名为 `BertForTextCLS`。

---

## 三、场景 C：**多分类**（N > 2）

几乎零成本。

- `args.num_labels = N`（比如新闻 10 分类就写 10）；
- 标签在 JSON 里依旧是整数字符串 `"0" ~ "N-1"`；
- **其它全不动**：`CrossEntropyLoss` 在任意类别数下都成立，`logits.argmax(dim=-1)` 也不用改。

常见坑：

- 类别不平衡时，给 `CrossEntropyLoss(weight=class_weights)` 传权重。
- 评估只看 accuracy 不够，建议加 `sklearn.metrics.classification_report` 打印 per-class P/R/F1。

---

## 四、场景 D：**回归**（语义相似度打分 0~5）

任务从"离散类别"变成"连续数值"，改 3 处：

1. **Task Head**（`modeling.py`）：
   ```python
   self.classifier = nn.Linear(config.hidden_size, 1)   # 2 → 1
   ```

2. **Loss**（`modeling.py: forward`）：
   ```python
   loss = nn.MSELoss()(logits.squeeze(-1), labels.float())
   ```

3. **标签类型**（`data.py: collote_fn`）：
   ```python
   batch_label.append(float(sample['label']))   # int → float
   ```

4. **评估指标**（`run_simi_cls.py: test_loop`）：准确率失效，改用 Pearson / Spearman：
   ```python
   from scipy.stats import pearsonr, spearmanr
   pearson = pearsonr(preds, labels)[0]
   ```

> 口诀：**分类 → 离散 → CrossEntropyLoss；回归 → 连续 → MSELoss**。

---

## 五、场景 E：**更换预训练模型**

### 5.1 同架构换权重（最常见）

| 换成 | 只改 shell 脚本里的 `--model_checkpoint` |
| --- | --- |
| 哈工大中文 RoBERTa | `hfl/chinese-roberta-wwm-ext` |
| 百度 ERNIE | `nghuyong/ernie-3.0-base-zh` |
| MacBERT | `hfl/chinese-macbert-base` |
| 英文 BERT | `bert-base-uncased` |

注意本项目 `MODEL_CLASSES` 里把 `'roberta'` 也映射到了 `BertForPairwiseCLS`：这是因为哈工大的 "roberta-wwm-ext" 权重实际上是以 BERT 结构发布的，**必须用 BertModel 加载**，否则权重对不上。如果要用 Facebook 真正的 RoBERTa 权重（英文 `roberta-base`），才把映射改回 `RobertaForPairwiseCLS`。

### 5.2 换成完全不同的架构（如 `ELECTRA` / `DeBERTa`）

改两个地方：

1. `modeling.py`：
   ```python
   from transformers import ElectraPreTrainedModel, ElectraModel
   class ElectraForPairwiseCLS(ElectraPreTrainedModel):
       def __init__(self, config, args):
           super().__init__(config)
           self.electra = ElectraModel(config)
           ...
   ```

2. `run_simi_cls.py: MODEL_CLASSES`：
   ```python
   MODEL_CLASSES = {'bert': ..., 'electra': ElectraForPairwiseCLS}
   ```

然后 shell 脚本 `--model_type=electra --model_checkpoint=hfl/chinese-electra-180g-base-discriminator`。

### 5.3 用 `*ForSequenceClassification` 零代码版本

如果你只是想快速跑起来而不关心自定义结构，可以完全绕开 `modeling.py`：

```python
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
```

HuggingFace 已经帮你写好了标准分类头。本项目保留自定义版本的意义是**学习**——让你看清"BERT + 分类头"内部是怎么搭出来的。

---

## 六、场景 F：**换成 token 级任务**（NER / 抽取式 QA）

这个改动比前几种都大，因为输出从"句子级 1 个标签"变成了"每个 token 1 个标签"。

### 改法的关键：`[:, 0, :]` 变成 `[:, :, :]`

```python
# 之前（句级）
cls_vectors = outputs.last_hidden_state[:, 0, :]   # [B, H]
logits = self.classifier(cls_vectors)              # [B, num_labels]

# 现在（token 级）
token_vectors = outputs.last_hidden_state          # [B, L, H]
logits = self.classifier(token_vectors)            # [B, L, num_labels]
```

相应地：
- **Dataset**：每个样本多一个 `labels` 数组，长度等于 token 数；padding 位置用 `-100`（`CrossEntropyLoss` 的 `ignore_index` 默认值）。
- **collate_fn**：用 `tokenizer(..., is_split_into_words=True, return_offsets_mapping=True)`，做 label 和 subword 的对齐。
- **评估**：不是 accuracy 而是 `seqeval` 的 entity-level F1。

本仓库其它目录已有现成示例，不要重造：
- `src/sequence_labeling_ner_cpd/` —— NER
- `src/sequence_labeling_extractiveQA_cmrc/` —— 抽取式 QA

**结论**：要做 token 级任务时，别硬改本目录，**直接 fork 那两个目录**。

---

## 七、"最小改动"清单速查

按照"你想做什么"反查要动的地方：

| 你想做的修改 | `arg.py` | `data.py` | `modeling.py` | `run_simi_cls.py` | shell |
| --- | :---: | :---: | :---: | :---: | :---: |
| 换数据集（同格式） |  |  |  |  | ✅ 路径 |
| 换数据集（不同字段名） |  | ✅ 字段 |  |  | ✅ 路径 |
| 单句分类 |  | ✅ collate_fn |  | `num_labels` |  |
| 多分类 N>2 |  |  |  | ✅ `num_labels` |  |
| 回归 |  | ✅ label 转 float | ✅ 头维度 1 + MSE | ✅ 评估指标 |  |
| 换预训练权重（同架构） |  |  |  |  | ✅ checkpoint |
| 换预训练架构 |  |  | ✅ 新增 Model 类 | ✅ `MODEL_CLASSES` | ✅ `model_type` |
| token 级任务 | —— 建议直接用 NER/QA 目录，不要在这里改 —— |

---

## 八、超详细改造案例：单句三分类

**任务**：电商评论情感分类（正 / 中 / 负）。

### Step 1：准备数据目录和文件

```bash
# 创建目录
mkdir -p data/senti_ecommerce

# 创建文件 data/senti_ecommerce/train.json（JSON Lines 格式）
```

**train.json 内容**（每行一个 JSON，共 3000 条示例）：
```json
{"text": "物流很快，包装精美，非常满意", "label": "2"}
{"text": "产品质量真的没话说，强烈推荐", "label": "2"}
{"text": "收到货很满意，和图片一致", "label": "2"}
{"text": "质量一般，和描述有差距", "label": "1"}
{"text": "产品还可以，没什么特别的", "label": "1"}
{"text": "感觉一般，价格有点贵", "label": "1"}
{"text": "完全是假货，商家欺骗消费者", "label": "0"}
{"text": "质量太差了，钱白花了", "label": "0"}
```

**标签含义**：0=负面 / 1=中立 / 2=正面

### Step 2：改 `data.py`（只改这里）

**改之前**（AFQMC 句对分类）：
```python
"""
data.py —— AFQMC 数据处理模块
"""
import json
from torch.utils.data import Dataset, DataLoader

class AFQMC(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)

    def load_data(self, data_file):
        Data = {}
        with open(data_file, 'rt', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                sample = json.loads(line.strip())
                Data[idx] = sample  # ← 键是 sentence1/sentence2/label
        return Data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_dataLoader(args, dataset, tokenizer, batch_size=None, shuffle=False):
    def collote_fn(batch_samples):
        batch_sentence_1, batch_sentence_2, batch_label = [], [], []
        
        for sample in batch_samples:
            batch_sentence_1.append(sample['sentence1'])  # ← 句对，两个字段
            batch_sentence_2.append(sample['sentence2'])
            batch_label.append(int(sample['label']))
        
        # tokenizer 接收两个参数
        batch_inputs = tokenizer(
            batch_sentence_1,  # ← 第一句
            batch_sentence_2,  # ← 第二句
            max_length=args.max_seq_length,
            padding=True, truncation=True, return_tensors="pt"
        )
        
        return {'batch_inputs': batch_inputs, 'labels': batch_label}

    return DataLoader(dataset, batch_size=batch_size or args.batch_size,
                      shuffle=shuffle, collate_fn=collote_fn)
```

**改之后**（单句三分类）：
```python
"""
data.py —— 电商评论情感分类数据处理模块
"""
import json
from torch.utils.data import Dataset, DataLoader

class SentiDataset(Dataset):  # ← 改了类名
    def __init__(self, data_file):
        self.data = self.load_data(data_file)

    def load_data(self, data_file):
        Data = {}
        with open(data_file, 'rt', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                sample = json.loads(line.strip())
                Data[idx] = sample  # ← 现在是 text/label（不是 sentence1/sentence2）
        return Data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_dataLoader(args, dataset, tokenizer, batch_size=None, shuffle=False):
    def collote_fn(batch_samples):
        batch_texts = []  # ← 改成单个列表
        batch_label = []
        
        for sample in batch_samples:
            batch_texts.append(sample['text'])  # ← 只取 'text'，不是 'sentence1'/'sentence2'
            batch_label.append(int(sample['label']))
        
        # tokenizer 接收一个参数（单句）
        batch_inputs = tokenizer(
            batch_texts,  # ← 只有一个参数
            max_length=args.max_seq_length,
            padding=True, truncation=True, return_tensors="pt"
        )
        
        return {'batch_inputs': batch_inputs, 'labels': batch_label}

    return DataLoader(dataset, batch_size=batch_size or args.batch_size,
                      shuffle=shuffle, collate_fn=collote_fn)
```

**关键改动总结**：
| 地方 | 改之前 | 改之后 |
| --- | --- | --- |
| 类名 | `class AFQMC` | `class SentiDataset` |
| JSON 字段 | `sentence1 / sentence2 / label` | `text / label` |
| collate_fn 里的字段提取 | `sample['sentence1']` 和 `sample['sentence2']` | `sample['text']` |
| tokenizer 参数 | `tokenizer(sent1, sent2, ...)` 两个参数 | `tokenizer(texts, ...)` 一个参数 |

### Step 3：改 `modeling.py`（几乎不用改）

**改之前**（AFQMC）：
```python
class BertForPairwiseCLS(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, args.num_labels)  # ← args.num_labels
        self.post_init()

    def forward(self, batch_inputs, labels=None):
        outputs = self.bert(**batch_inputs)
        cls_vectors = outputs.last_hidden_state[:, 0, :]
        cls_vectors = self.dropout(cls_vectors)
        logits = self.classifier(cls_vectors)  # ← [B, num_labels]
        
        loss = None
        if labels is not None:
            if not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels, device=logits.device)
            loss = nn.CrossEntropyLoss()(logits, labels)  # ← CrossEntropyLoss 支持任意类别数
        
        return loss, logits
```

**改之后**（可选改类名，逻辑完全相同）：
```python
class BertForSentiCLS(BertPreTrainedModel):  # ← 改了类名（可选）
    def __init__(self, config, args):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, args.num_labels)  # ← 当 args.num_labels=3 时，自动变成 Linear(768, 3)
        self.post_init()

    def forward(self, batch_inputs, labels=None):
        outputs = self.bert(**batch_inputs)
        cls_vectors = outputs.last_hidden_state[:, 0, :]
        cls_vectors = self.dropout(cls_vectors)
        logits = self.classifier(cls_vectors)  # ← [B, 3]
        
        loss = None
        if labels is not None:
            if not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels, device=logits.device)
            loss = nn.CrossEntropyLoss()(logits, labels)  # ← 完全没变
        
        return loss, logits
```

**结论**：modeling.py 可以一个字不改，因为它已经是参数化的（`args.num_labels` 决定分类头尺寸）。只需在 run_simi_cls.py 里把 `args.num_labels = 3` 改好即可。

### Step 4：改 `run_simi_cls.py`（5 处改动）

**改动 1**：第 32~33 行，改 import
```python
# 改之前
from src.pairwise_cls_similarity_afqmc.data import AFQMC, get_dataLoader
from src.pairwise_cls_similarity_afqmc.modeling import BertForPairwiseCLS, RobertaForPairwiseCLS

# 改之后
from src.pairwise_cls_similarity_afqmc.data import SentiDataset, get_dataLoader  # ← 改这里
from src.pairwise_cls_similarity_afqmc.modeling import BertForSentiCLS, RobertaForPairwiseCLS  # ← 改这里
```

**改动 2**：第 70~74 行，改 MODEL_CLASSES
```python
# 改之前
MODEL_CLASSES = {
    'bert': BertForPairwiseCLS,
    'roberta': BertForPairwiseCLS
}

# 改之后
MODEL_CLASSES = {
    'bert': BertForSentiCLS,  # ← 改这里
    'roberta': BertForSentiCLS  # ← 改这里
}
```

**改动 3**：第 426 行，改 num_labels
```python
# 改之前
args.num_labels = 2

# 改之后
args.num_labels = 3  # ← 三分类，改这里
```

**改动 4**：第 442~446 行，改数据集类名（3 处）
```python
# 改之前
logger.info("***** Loading training and validation datasets *****")
train_dataset = AFQMC(args.train_file)
dev_dataset = AFQMC(args.dev_file)

# 改之后
logger.info("***** Loading training and validation datasets *****")
train_dataset = SentiDataset(args.train_file)  # ← 改这里
dev_dataset = SentiDataset(args.dev_file)      # ← 改这里
```

**改动 5**：第 481~482 行和第 464 行
```python
# test() 函数里
test_dataset = AFQMC(args.test_file)  # ← 改成 SentiDataset

# predict() 模式里
test_dataset = AFQMC(args.test_file)  # ← 改成 SentiDataset
```

### Step 5：新建 shell 脚本 `run_senti_cls.sh`

在本目录创建 `run_senti_cls.sh`：

```bash
#!/bin/bash

export OUTPUT_DIR=./senti_results/

python3 run_simi_cls.py \
    --output_dir=$OUTPUT_DIR \
    --model_type=bert \
    --model_checkpoint=hfl/chinese-roberta-wwm-ext \
    --train_file=../../data/senti_ecommerce/train.json \
    --dev_file=../../data/senti_ecommerce/dev.json \
    --test_file=../../data/senti_ecommerce/test.json \
    --max_seq_length=256 \
    --learning_rate=2e-5 \
    --num_train_epochs=3 \
    --batch_size=32 \
    --warmup_proportion=0.1 \
    --do_train \
    --do_test \
    --seed=42
```

### Step 6：验证数据加载无误

在项目根目录写个小脚本 `test_senti_data.py`：

```python
import sys
import json
sys.path.append('.')

from src.pairwise_cls_similarity_afqmc.data import SentiDataset

# 加载训练集
dataset = SentiDataset('data/senti_ecommerce/train.json')
print(f"✓ 数据集大小: {len(dataset)}")

# 查看第一条样本
sample = dataset[0]
print(f"✓ 第一条样本: {sample}")

# 确认字段
assert 'text' in sample, "样本里必须有 'text' 字段"
assert 'label' in sample, "样本里必须有 'label' 字段"
print(f"✓ 字段检查通过")
```

运行：
```bash
python test_senti_data.py
```

输出应该像这样：
```
✓ 数据集大小: 3000
✓ 第一条样本: {'text': '物流很快，包装精美，非常满意', 'label': '2'}
✓ 字段检查通过
```

### Step 7：运行训练

```bash
cd /Users/huwei/PyCharmMiscProject/How-to-use-Transformers/src/pairwise_cls_similarity_afqmc/
chmod +x run_senti_cls.sh
bash run_senti_cls.sh
```

训练完成后，输出在 `senti_results/` 目录：
```
senti_results/
├── epoch_1_dev_acc_xx.x_weights.bin    # 第 1 个 epoch 的权重（如果是最好的）
├── epoch_2_dev_acc_yy.y_weights.bin    # 第 2 个 epoch 的权重（如果超过了前面的）
├── epoch_3_dev_acc_zz.z_weights.bin    # 第 3 个 epoch 的权重（如果继续提升）
├── train.log                           # 训练日志
├── test.log                            # 测试日志
└── args.txt                            # 本次运行的参数
```

### 改动清单（快速参考）

| 文件 | 改什么 | 具体位置 | 具体改动 |
| --- | --- | --- | --- |
| `data.py` | 数据加载 | 第 7 行 | `class AFQMC` → `class SentiDataset` |
|  |  | 第 19 行 | 对应改类的初始化 |
|  | collate_fn | 第 35~37 行 | `batch_sentence_1/2` → `batch_texts`；`sample['sentence1']/['sentence2']` → `sample['text']` |
|  |  | 第 40~43 行 | `tokenizer(sent1, sent2, ...)` → `tokenizer(batch_texts, ...)` |
| `modeling.py` | 类名（可选） | 第 26 行 | `BertForPairwiseCLS` → `BertForSentiCLS` |
| `run_simi_cls.py` | import | 第 32~33 行 | `AFQMC` → `SentiDataset`；`BertForPairwiseCLS` → `BertForSentiCLS` |
|  | MODEL_CLASSES | 第 70~74 行 | 映射改成 `BertForSentiCLS` |
|  | num_labels | 第 426 行 | `= 2` → `= 3` |
|  | 加载数据（多处） | 442, 446, 464, 481, 482 行 | `AFQMC(...)` → `SentiDataset(...)` |
| `run_senti_cls.sh` | 新建脚本 | 新文件 | 把数据路径指向 `data/senti_ecommerce/`，`num_labels=3` |

**总计改动**：不超过 10 行代码，5 分钟搞定。

---

## 九、老手才会踩到的坑

1. **`num_labels` 没设对，权重能加载成功但预测乱掉**。`from_pretrained` 只校验 encoder 部分，分类头尺寸是你自己决定的，改 num_labels 的时候要连带检查 `--do_test` 时加载的旧 checkpoint 是否匹配。

2. **哈工大 RoBERTa 必须走 BertModel**。这是 `run_simi_cls.py: MODEL_CLASSES` 里 `'roberta': BertForPairwiseCLS` 的原因。想换成真·Facebook RoBERTa，要同时改映射和 tokenizer。

3. **类别不平衡时 accuracy 会骗人**。`test_loop` 里加一行打印 `confusion_matrix`，几秒钟的事，能省几小时的调试。

4. **`max_seq_length=512` 不是越长越好**。self-attention 是 O(L²)，长度翻倍、显存翻 4 倍；大多数中文任务 128~256 足够。

5. **`--do_train` 时输出目录非空会抛错**（这是 `run_simi_cls.py` 第 395~396 行故意写的"防覆盖"保护）。想重跑同一个实验，要么换目录，要么手动清空——这是刻意设计，不是 bug。

6. **学习率调度器一定要每步 `step()`**。`pipeline.py` 和 `run_simi_cls.py` 都在训练 for 循环里调了 `lr_scheduler.step()`，删掉这一行学习率就冻住了——排查起来很隐蔽。

---

## 十、下一步可以读什么

- 本仓库 `src/text_cls_prompt_senti_chnsenticorp/` —— 情感分类的 prompt 范式版本
- 本仓库 `src/sequence_labeling_ner_cpd/` —— token 级任务的模板
- 本仓库 `src/seq2seq_summarization/` —— 生成式任务（mT5）
- HuggingFace 官方 [examples/pytorch/text-classification](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification) —— 工业级写法对照

把这份目录里的 5 个文件 + 本指南读熟，再去看上面任何一个，都会觉得"哦不过是在改同样的 5 个零件而已"。