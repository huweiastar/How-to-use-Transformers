## 句子对分类任务（金融语义相似度 AFQMC）

> 配套教程：[《Transformers 库快速入门 第七章：微调预训练模型》](https://transformers.run/c2/2021-12-17-transformers-note-4/)

本目录用一个最小可跑通的例子，带你完成第一个 Transformers 微调项目：**判断两个句子是否同义**（二分类，标签 0=不同义，1=同义）。数据来自蚂蚁金融语义相似度数据集 AFQMC，训练/验证/测试划分为 34334 / 4316 / 3861 条。

---

### 一、推荐学习顺序（由浅入深）

| 顺序 | 文件 | 作用 | 学习要点 |
| --- | --- | --- | --- |
| 1 | `pipeline.py` | **一个文件跑通全流程**的最小示例 | Dataset → DataLoader → Model → 训练循环 的完整骨架 |
| 2 | `data.py` | 规范化的数据加载模块 | 自定义 Dataset、`collate_fn` 动态 padding |
| 3 | `modeling.py` | 规范化的模型定义 | 继承 `BertPreTrainedModel` 的正确姿势 |
| 4 | `arg.py` | 命令行参数解析 | 如何把超参数从代码里解放出来 |
| 5 | `run_simi_cls.py` | 工程化入口脚本 | 训练 / 测试 / 预测三种模式如何组织 |

**建议**：先完整读懂 `pipeline.py`（它是教程里一步步拼出来的版本），然后再看其它四个文件——它们只是把 `pipeline.py` 拆开并加了命令行参数、日志和保存逻辑，便于真实项目复用。

---

### 二、关键概念速查（看懂这几个，就抓住了重点）

1. **句子对怎么喂进 BERT？**
   `tokenizer(sent1, sent2, ...)` 会编码成 `[CLS] sent1 [SEP] sent2 [SEP]`，并返回三样东西：
   - `input_ids`：token 的整数 id，形状 `[batch, seq_len]`
   - `attention_mask`：1 表示真实 token，0 表示 padding
   - `token_type_ids`：0 表示第一句，1 表示第二句（BERT 用它区分两句）

2. **动态 padding**：`padding=True` 只把一个 batch 内的句子补齐到"本 batch 最长长度"，而不是固定 512。每个 batch 的 seq_len 会变，这样比静态 padding 快很多。

3. **取 `[CLS]` 做分类**：`outputs.last_hidden_state[:, 0, :]` 就是 `[CLS]` 位置的向量，形状 `[batch, hidden]`。接一层 `Linear(hidden, 2)` 就得到二分类 logits。这是 BERT 做句级任务的标准做法。

4. **继承 `BertPreTrainedModel` 而不是包一层 `nn.Module`**：
   - 能直接用 `.from_pretrained(checkpoint)` 自动加载权重
   - 自动走 `post_init()` 正确初始化你新加的层
   - 能被 `save_pretrained` / `AutoConfig` 等 HuggingFace 工具识别

5. **`AdamW` + 线性预热衰减**是 Transformers 微调的标配：
   - 学习率一般用 `1e-5 ~ 5e-5`
   - 预热（warmup）让学习率从 0 线性爬升到峰值，再线性降到 0
   - `bias` 和 `LayerNorm.weight` 通常**不加** weight decay

6. **保存"最好"而不是"最后"**：每个 epoch 在验证集上评估，只有指标提升才保存 `state_dict()`。文件名里带上 epoch 和准确率，方便事后挑选。

---

### 三、如何运行

**方式 A：最小示例（直接跑 `pipeline.py`）**

```bash
python pipeline.py
```

**方式 B：工程化脚本（推荐用于真实项目）**

```bash
bash run_simi_bert.sh       # 训练 BERT
bash run_simi_roberta.sh    # 训练 RoBERTa（哈工大权重）
```

想改成"只测试"或"只预测"，把脚本里的 `--do_train` 改为 `--do_test` 或 `--do_predict` 即可——训练、测试、预测是解耦的。

> 参考指标：3 个 epoch 后 BERT 在验证集约 73.61%，RoBERTa 约 73.84%（V100, batch=16）。

---

### 四、举一反三

掌握上面这套流程后，以下任务**只需改 3 处**就能复用：

| 新任务 | 需要改的地方 |
| --- | --- |
| 单句分类（情感分析等） | `collate_fn` 里 `tokenizer(单句)`；Dataset 字段换成 `text` |
| 多分类（N>2） | `args.num_labels = N`；标签列表保持整数 |
| 回归（语义相似度打分） | 分类头改 `Linear(hidden, 1)`；损失改 `MSELoss` |
| 更换预训练模型 | 换 `checkpoint` 字符串即可（如 `hfl/chinese-roberta-wwm-ext`） |

数据加载、训练循环、保存策略**不需要动**——这正是把代码拆成 `data.py / modeling.py / run_*.py` 的意义。

注意
- 分类（离散类别 0/1/2…）→ CrossEntropyLoss
- 回归（连续数值 3.14 / 5.2 / 0.8…）→ MSELoss
