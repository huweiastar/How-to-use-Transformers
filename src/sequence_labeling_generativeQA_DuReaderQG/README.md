## 生成式问答任务（中文阅读理解 DuReaderQG）

> 配套教程：[《Transformers 库快速入门 第十一章：序列到序列模型》](https://transformers.run/c2/2022-03-24-transformers-note-7/)

本目录用一个最小可跑通的例子，带你完成第一个 **Seq2Seq 微调项目**：给一段文章和一个问题，让模型**生成**答案（而不是从原文里"抠"一段出来）。数据来自百度 DuReaderQG 中文阅读理解，训练 / 验证集约 14520 / 984 条。

> **抽取式 vs 生成式 一句话区别**：抽取式（如 `sequence_labeling_extractiveQA_cmrc`）让 BERT 预测答案在原文里的"起始/结束位置"——答案必须是原文片段；生成式让 T5 **逐字写出**答案——答案可以是原文里没出现的词。前者是分类问题，后者是生成问题。

---

### 一、推荐学习顺序（由浅入深）

| 顺序 | 文件 | 作用 | 学习要点 |
| --- | --- | --- | --- |
| 1 | `pipeline.py` | **一个文件跑通全流程**的最小示例 | T5 输入格式 → Dataset → 训练循环 → `model.generate()` 生成答案 |
| 2 | `data.py` | 规范化的数据加载模块 | `QADataset` 如何把 `(question, context, answer)` 编码成 `input_ids` / `labels` |
| 3 | `modeling.py` | 模型与优化器加载 | `T5ForConditionalGeneration.from_pretrained()`、AdamW + 线性预热 |
| 4 | `trainer.py` | 训练 / 评估循环 | `outputs.loss` vs `model.generate()`、按 BLEU-2 保存最佳模型 |
| 5 | `utils.py` | BLEU 计算 / 预测 / 保存 | 多 n-gram BLEU、Beam Search 解码 |
| 6 | `arg.py` | 命令行参数解析 | 把超参数从代码里解放出来，支持多模型选择 |
| 7 | `run_generativeQA.py` | 工程化入口脚本 | `--do_train / --do_eval / --do_predict` 三种模式如何组织 |

**建议**：先把 `pipeline.py` 完整跑通（它是教程里一步步拼出来的版本，所有逻辑一目了然），再看 2~7——它们只是把 `pipeline.py` 拆开并加上命令行参数、日志和保存逻辑，便于真实项目复用。

> 还有一份 `pipeline.ipynb`，内容与 `pipeline.py` 等价，喜欢 Jupyter 的同学可以直接打开跟着执行。

---

### 二、关键概念速查（看懂这几个，就抓住了重点）

1. **T5 是"编码器 + 解码器"，BERT 是"只有编码器"**
   - BERT：把输入压成向量 → 接分类头预测标签。**不能生成**。
   - T5：编码器读输入 → 解码器**逐字生成**输出。所以做"摘要、翻译、问答（生成式）"用 T5 这类 seq2seq 模型，做"分类、序列标注"用 BERT。

2. **T5 的输入格式：万事万物都拼成一串文本**
   T5 是 "Text-to-Text Transfer Transformer"——所有任务都拼成一个字符串喂进去：
   ```
   question: 谁发明了电话?  context: 亚历山大·贝尔在1876年发明了电话...
   ```
   答案 `亚历山大·贝尔` 也是一串文本，由解码器一字一字生成。

3. **`labels` 怎么准备？直接用答案的 token id**
   ```python
   inputs  = tokenizer("question: ... context: ...", ...)  # 编码器输入
   targets = tokenizer("亚历山大·贝尔", ...)                  # 解码器目标
   model(input_ids=inputs.input_ids, labels=targets.input_ids)
   # transformers 内部会自动右移 labels、计算 cross-entropy loss
   ```
   你不需要手动构造 `decoder_input_ids`，HuggingFace 会替你做。

4. **训练用 `model(...)`，预测用 `model.generate(...)`**
   - 训练（教师强制 teacher forcing）：把整段标准答案喂给解码器，并行计算每一步的损失——快。
   - 推理：解码器只能拿到自己上一步生成的 token，**自回归**地一字一字往外吐——慢。所以训练快、推理慢是 seq2seq 的常态。
   - `model.generate()` 里常用的 `num_beams=4` 是 Beam Search：每步保留 4 个候选，最后挑总概率最大的那条，比贪心解码效果好。

5. **BLEU：生成任务怎么评估？**
   分类有"准确率"、抽取式 QA 有"EM/F1"，生成式 QA 用 **BLEU**——比较预测和参考答案在 1/2/3/4-gram 上的重合度。本项目按 **BLEU-2 选最佳模型**（BLEU-1 太宽松、BLEU-4 在短答案上方差太大，BLEU-2 兼顾词形和搭配）。

6. **中文 T5 选哪个？默认 `Langboat/mengzi-t5-base`**
   原版 Google T5 只在英文上预训练，中文几乎不可用。中文场景三种常见选择（在 `arg.py` / `pipeline.py` 顶部切换）：
   - `mengzi-t5-base` ⭐ 中文专用，中文任务最佳
   - `mt5-base` Google 多语言版本，101 种语言但中文不如 mengzi 精专
   - `google-t5-base` 仅英文，做英文任务时再用

7. **`AdamW` + 线性预热衰减**仍是标配
   - 学习率一般 `1e-5 ~ 5e-5`（本项目用 `2e-5`）
   - 预热让学习率从 0 线性爬升到峰值，再线性降到 0，避免一开始大步长破坏预训练权重

---

### 三、如何运行

**前置准备**：把 DuReaderQG 数据放在 `../../data/DuReaderQG/`，包含 `train.json` 和 `dev.json`（每行一个 JSON：`{"context", "question", "answer", "id"}`）。

**方式 A：最小示例（直接跑 `pipeline.py`）**

```bash
python pipeline.py
```

一个文件包揽：加载数据 → 训练 3 个 epoch → 画收敛曲线 → 在验证集和自定义样例上预测 → 保存模型。**第一次学习推荐这条路。**

**方式 B：工程化脚本（推荐用于真实项目）**

```bash
bash run_generativeQA.sh
```

或自己拼参数：

```bash
python run_generativeQA.py \
    --data_dir ../../data/DuReaderQG/ \
    --output_dir ./qa_model_output \
    --model_choice mengzi-t5-base \
    --num_epochs 3 --batch_size 8 --learning_rate 2e-5 \
    --do_all
```

把 `--do_all` 换成 `--do_eval` 或 `--do_predict`，就只跑评估 / 只跑预测——三种模式是解耦的。

**方式 C：Jupyter Notebook**

```bash
jupyter notebook pipeline.ipynb
```

> **参考指标**（Mengzi-T5-Base, 3 epoch, batch=8, lr=2e-5, 单卡 V100）：验证集 BLEU-1 ≈ 0.40，BLEU-2 ≈ 0.33，BLEU-4 ≈ 0.22。

---

### 四、举一反三

T5 的 "Text-to-Text" 范式让一套代码能复用到几乎所有"输入文本 → 输出文本"的任务上。掌握上面这套流程后，以下任务**只需改 2 处**就能复用：

| 新任务 | 需要改的地方 |
| --- | --- |
| 文本摘要 | `data.py` 里 `input_text = f"summarize: {article}"`；`labels` 换成摘要 |
| 中英翻译 | `input_text = f"translate Chinese to English: {zh}"`；`labels` 换成英文 |
| 对话回复 | `input_text = f"dialogue: {history}"`；`labels` 换成回复 |
| 标题生成 | `input_text = f"title: {body}"`；`labels` 换成标题 |
| 换更大的中文模型 | 改 `--model_choice mt5-base`，或在 `arg.py` 的 `MODEL_CONFIG` 里加新条目 |

训练循环、`model.generate()`、BLEU 评估、保存策略**完全不用动**——这正是把代码拆成 `data.py / modeling.py / trainer.py` 的意义。

> **小贴士**：T5 的"任务前缀"（`question:` / `summarize:` / `translate ... to ...`）只是约定俗成的字符串，模型不会魔法地理解英文单词的含义——它学到的是"看到这个前缀就该输出哪种内容"。所以中文场景下你也可以写 `f"问题: {Q} 文章: {C}"`，效果通常差不多，挑一个写法贯穿训练和推理即可。

---

### 五、目录速览

```
sequence_labeling_generativeQA_DuReaderQG/
├── pipeline.py            # ⭐ 最小示例，一个文件跑通全流程（学习从这里开始）
├── pipeline.ipynb         # 同上，Notebook 版本
├── data.py                # QADataset、create_dataloaders
├── modeling.py            # 加载 T5 模型 / 分词器 / 优化器
├── trainer.py             # train_epoch、evaluate、train（完整训练循环）
├── utils.py               # BLEU 计算、predict_answer、保存与打印
├── arg.py                 # 命令行参数 + 模型配置表
├── run_generativeQA.py    # 工程化入口（do_train / do_eval / do_predict）
├── run_generativeQA.sh    # 上面脚本的 bash 包装
├── README.md              # 你正在看的这个
└── qa_model_output/       # 训练完成后自动生成，见下节说明
    ├── best_model.pt
    ├── training_curves.png
    ├── training_history.json
    ├── hyperparams.json
    └── final_model/
        ├── model.safetensors
        ├── config.json
        ├── generation_config.json
        ├── tokenizer_config.json
        └── tokenizer.json
```

---

### 六、`qa_model_output/` 各文件详解

运行 `pipeline.py` 或 `run_generativeQA.py --do_train` 后，会在 `qa_model_output/` 下自动生成以下文件。

#### 6.1 顶层文件

**`best_model.pt`**（约 850 MB）

PyTorch 原生格式（`torch.save`）保存的**验证集最佳模型权重**。"最佳"的标准是验证集 **BLEU-2 最高**的那个 checkpoint（而不是最后一个 epoch）。保存内容是 `model.state_dict()`，即所有参数张量的字典，不含模型结构定义。

加载方式：
```python
model = T5ForConditionalGeneration(config)
model.load_state_dict(torch.load("qa_model_output/best_model.pt"))
```

> 为什么保存两份模型（`best_model.pt` 和 `final_model/`）？  
> `best_model.pt` 是训练过程中按 BLEU-2 即时保存的"快照"，格式轻便但依赖手动重建模型结构；`final_model/` 是训练结束后用 HuggingFace `save_pretrained()` 完整导出的版本，包含配置文件，可以直接 `from_pretrained()` 加载，更适合后续部署和分享。

---

**`training_curves.png`**

训练过程的**可视化曲线图**，包含：
- 每个 epoch 的训练集 loss（`train_loss`）
- 每个 epoch 的验证集 loss（`dev_loss`）
- 验证集 BLEU-1 / BLEU-2 / BLEU-4 曲线

用于直观判断模型是否收敛、是否过拟合。本项目实际训练 3 个 epoch 的数值示意：

| Epoch | Train Loss | Dev Loss | BLEU-1 | BLEU-2 | BLEU-4 |
|-------|-----------|---------|--------|--------|--------|
| 1 | 1.054 | 0.0422 | 0.0559 | 0.0177 | 0.0099 |
| 2 | 0.033 | 0.0403 | 0.0620 | 0.0196 | 0.0110 |
| 3 | 0.030 | 0.0401 | 0.0603 | 0.0191 | 0.0107 |

> Train Loss 从 1.05 急降到 0.03 是正常现象——第 1 个 epoch 模型在大量调整预训练权重，后续 epoch 微调已收敛到局部最优，变化幅度自然减小。Dev Loss 持续微降说明没有过拟合。

---

**`training_history.json`**

以 JSON 格式保存的**所有训练指标数值**，结构如下：

```json
{
  "train_loss": [1.054, 0.033, 0.030],
  "dev_loss":   [0.042, 0.040, 0.040],
  "dev_bleu1":  [0.056, 0.062, 0.060],
  "dev_bleu2":  [0.018, 0.020, 0.019],
  "dev_bleu3":  [0.012, 0.013, 0.013],
  "dev_bleu4":  [0.010, 0.011, 0.011]
}
```

每个列表的长度等于训练的 epoch 数。可用于事后自定义绘图或与其他实验对比：

```python
import json, matplotlib.pyplot as plt
history = json.load(open("qa_model_output/training_history.json"))
plt.plot(history["dev_bleu2"], label="BLEU-2")
```

---

**`hyperparams.json`**

记录本次训练使用的**超参数快照**，方便复现实验：

```json
{
  "learning_rate": 2e-05,
  "num_epochs": 3,
  "batch_size": 8,
  "warmup_ratio": 0.1,
  "weight_decay": 0.01,
  "max_grad_norm": 1.0,
  "seed": 42
}
```

| 参数 | 含义 |
|------|------|
| `learning_rate` | AdamW 峰值学习率，T5 微调推荐 `1e-5 ~ 5e-5` |
| `warmup_ratio` | 前 10% 的步数线性预热，避免初始大步长破坏预训练权重 |
| `weight_decay` | L2 正则，防止过拟合 |
| `max_grad_norm` | 梯度裁剪上限，防止梯度爆炸 |
| `seed` | 随机种子，固定后可复现完全相同的训练结果 |

---

#### 6.2 `final_model/` 子目录

训练结束后调用 `model.save_pretrained("qa_model_output/final_model")` 和 `tokenizer.save_pretrained(...)` 生成，格式与 HuggingFace Hub 上下载的模型完全一致，可直接用 `from_pretrained` 加载。

**`model.safetensors`**（约 850 MB）

模型参数的**安全序列化格式**（[safetensors](https://github.com/huggingface/safetensors)），是 HuggingFace 推荐的替代 `pytorch_model.bin` 的新标准：
- 加载速度更快（内存映射，无需完整读入再反序列化）
- 更安全（不执行任意 Python 代码，杜绝 pickle 注入风险）
- 本质上存储的内容与 `best_model.pt` 相同，都是微调后的 T5 参数

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer
model = T5ForConditionalGeneration.from_pretrained("qa_model_output/final_model")
tokenizer = T5Tokenizer.from_pretrained("qa_model_output/final_model")
```

---

**`config.json`**

模型的**结构超参数**，由 HuggingFace 的 `PretrainedConfig` 序列化而来，包括：

| 字段 | 值 | 含义 |
|------|----|------|
| `architectures` | `T5ForConditionalGeneration` | 模型类名，`from_pretrained` 靠它自动选择正确的类 |
| `d_model` | 768 | 隐藏层维度（Base 版本） |
| `num_layers` | 12 | Encoder 层数 |
| `num_decoder_layers` | 12 | Decoder 层数 |
| `num_heads` | 12 | 注意力头数 |
| `d_ff` | 3072 | FFN 中间层维度（= 4 × d_model） |
| `vocab_size` | 32128 | 词表大小 |
| `n_positions` | 512 | 最大输入序列长度 |

---

**`generation_config.json`**

**推理生成时的默认参数**，调用 `model.generate()` 时若不显式传参则读取这里的配置：

```json
{
  "decoder_start_token_id": 0,
  "eos_token_id": 1,
  "pad_token_id": 0
}
```

- `decoder_start_token_id`：解码器第一步输入的 token id（T5 用 `<pad>` 即 0 启动解码）
- `eos_token_id`：生成到该 token 时停止（`</s>` 即 1）
- 更多生成参数（`num_beams`、`max_new_tokens` 等）可在调用时动态传入覆盖

---

**`tokenizer_config.json`**

**分词器的基本配置**，记录：

| 字段 | 含义 |
|------|------|
| `tokenizer_class` | `T5Tokenizer`，加载时自动实例化 |
| `model_max_length` | 512，超出时自动截断 |
| `pad_token` | `<pad>`，批处理时短序列补齐到同一长度 |
| `eos_token` | `</s>`，序列结束标志 |
| `extra_ids` | 100 个 `<extra_id_x>` 占位符，T5 预训练 Span Corruption 任务用，微调时一般用不到 |

---

**`tokenizer.json`**

分词器的**完整词表和规则**（BPE/SentencePiece 的序列化结果），包含：
- 所有 token 到 id 的映射（32128 个词条）
- BPE 合并规则
- 特殊 token 的处理方式

这是实际执行 `tokenizer.encode()` / `tokenizer.decode()` 时真正被读取的文件，其他 `*_config.json` 只是配置元信息。