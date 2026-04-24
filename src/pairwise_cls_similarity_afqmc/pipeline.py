"""
===============================================================================
 pipeline.py —— 一个文件跑通 BERT 微调全流程（AFQMC 句对相似度分类）
===============================================================================

这是本目录里最重要的"教学用"脚本。它对应教程
    https://transformers.run/c2/2021-12-17-transformers-note-4/
一步步拼出来的完整代码，把以下 6 件事塞进了一个文件：

    1. 设定随机种子 & 自动选择设备 (CPU / CUDA / Apple MPS)
    2. 定义 Dataset —— 按行读 json
    3. 定义 DataLoader 的 collate_fn —— 分词 + 动态 padding
    4. 定义模型 —— 继承 BertPreTrainedModel + 自定义分类头
    5. 训练循环 train_loop + 验证循环 test_loop
    6. 每个 epoch 后挑"验证集最好"的权重保存下来

建议的学习路径：从头读到尾，看不懂的地方回头翻 readme.md 的"关键概念速查"。
当你读懂这个文件，再去看 data.py / modeling.py / run_simi_cls.py 就会发现，
它们只是把这里的内容拆成几个模块、加了命令行参数和日志而已。
===============================================================================
"""
import random
import os
import numpy as np
import json
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoConfig
from transformers import BertPreTrainedModel, BertModel
from transformers import get_scheduler
from torch.optim import AdamW  # 与 transformers.AdamW 等价，且无 deprecation 警告
from tqdm.auto import tqdm


# -----------------------------------------------------------------------------
# 1. 可复现性：固定所有随机源的种子
# -----------------------------------------------------------------------------
# 深度学习里有很多随机性：Python random、NumPy、PyTorch、CUDA kernel……
# 想让每次跑出来的结果可复现，必须把它们同时固定。
def seed_everything(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cuDNN 的某些算法即使固定了 seed 仍然是非确定的，需要显式关闭
    torch.backends.cudnn.deterministic = True


# -----------------------------------------------------------------------------
# 2. 自动选择设备：优先 Apple MPS (M 系列芯片) > CUDA > CPU
# -----------------------------------------------------------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f'Using {device} device')
seed_everything(42)

# -----------------------------------------------------------------------------
# 3. 超参数：初学者常改的几个
# -----------------------------------------------------------------------------
learning_rate = 1e-5   # 微调 BERT 常用区间 1e-5 ~ 5e-5
batch_size = 4         # 显存不够就调小
epoch_num = 3          # AFQMC 跑 3 个 epoch 基本就收敛了

# checkpoint 可以是 HuggingFace Hub 上的模型名，也可以是本地目录
checkpoint = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


# -----------------------------------------------------------------------------
# 4. 自定义 Dataset：AFQMC 数据按行存 json，每行一个样本
# -----------------------------------------------------------------------------
# PyTorch 有两种 Dataset：
#   - Dataset（map-style）   ：支持随机索引 data[i]，小数据集首选
#   - IterableDataset（迭代） ：只支持顺序读，适合超大数据流式处理
# AFQMC 只有 3 万多条，用 map-style 完全够。
class AFQMC(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)

    def load_data(self, data_file):
        # 以行号为 key，样本 dict 为 value，便于 __getitem__ 随机访问
        Data = {}
        with open(data_file, 'rt', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                # 每行是一个 json：{"sentence1": ..., "sentence2": ..., "label": "0/1"}
                sample = json.loads(line.strip())
                Data[idx] = sample
        return Data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


train_data = AFQMC('/Users/huwei/PyCharmMiscProject/How-to-use-Transformers/data/afqmc_public/train.json')
valid_data = AFQMC('/Users/huwei/PyCharmMiscProject/How-to-use-Transformers/data/afqmc_public/dev.json')


# -----------------------------------------------------------------------------
# 5. collate_fn：把一个 batch 的 dict 样本组装成模型能吃的张量
# -----------------------------------------------------------------------------
# DataLoader 每次取出 batch_size 个样本，默认它会用 torch.stack 尝试堆叠，
# 但我们的样本是 dict 而且长度不一，所以必须写自定义 collate_fn。

def collote_fn(batch_samples):
    batch_sentence_1, batch_sentence_2 = [], []
    batch_label = []
    for sample in batch_samples:
        batch_sentence_1.append(sample['sentence1'])
        batch_sentence_2.append(sample['sentence2'])
        batch_label.append(int(sample['label']))
    X = tokenizer(
        batch_sentence_1,
        batch_sentence_2,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    y = torch.tensor(batch_label)
    return X, y


# 训练集 shuffle=True 打乱顺序，验证集不用打乱
# 训练集：打乱 → 学得稳、学得好
# 验证集：不打乱 → 评得准、可复现
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collote_fn)
valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, collate_fn=collote_fn)


# -----------------------------------------------------------------------------
# 6. 模型：继承 BertPreTrainedModel，手动加一个分类头
# -----------------------------------------------------------------------------
# 为什么要继承 BertPreTrainedModel 而不是 nn.Module？
#   ✓ 可以直接用 .from_pretrained(checkpoint) 加载 BERT 预训练权重
#   ✓ post_init() 会正确初始化你新加的 classifier
#   ✓ 能被 HuggingFace 生态的 save_pretrained / AutoConfig 等工具识别
class BertForPairwiseCLS(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # add_pooling_layer=False：我们自己取 [CLS]，不要 BERT 自带的 pooler
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 768 是 bert-base 的 hidden_size，2 是二分类
        self.classifier = nn.Linear(768, 2)
        self.post_init()

    def forward(self, x):
        # x 是 tokenizer 返回的 dict：input_ids / attention_mask / token_type_ids
        outputs = self.bert(**x)
        # last_hidden_state 形状 [batch, seq_len, hidden]
        # 取第 0 个位置（[CLS]），就得到 [batch, hidden] 的整句对表示
        cls_vectors = outputs.last_hidden_state[:, 0, :]
        cls_vectors = self.dropout(cls_vectors)
        logits = self.classifier(cls_vectors)
        return logits


# 加载 config + 预训练权重 —— 这一步是"微调"的起点
config = AutoConfig.from_pretrained(checkpoint)
model = BertForPairwiseCLS.from_pretrained(checkpoint, config=config).to(device)


# -----------------------------------------------------------------------------
# 7. 训练循环：每个 epoch 遍历一遍 train_dataloader
# -----------------------------------------------------------------------------
def train_loop(dataloader, model, loss_fn, optimizer, lr_scheduler, epoch, total_loss):
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f'loss: {0:>7f}')
    # finish_step_num：之前的 epoch 累计走了多少步，用于计算"跨 epoch 的平均 loss"
    finish_step_num = (epoch - 1) * len(dataloader)

    model.train()   # 打开 dropout、batchnorm 的训练模式
    for step, (X, y) in enumerate(dataloader, start=1):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        # PyTorch 训练三件套：清零 → 反向 → 更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()   # 每一步都要推进 scheduler，否则学习率不会衰减

        total_loss += loss.item()
        progress_bar.set_description(f'loss: {total_loss / (finish_step_num + step):>7f}')
        progress_bar.update(1)
    return total_loss


# -----------------------------------------------------------------------------
# 8. 评估循环：验证集 / 测试集共用
# -----------------------------------------------------------------------------
def test_loop(dataloader, model, mode='Test'):
    assert mode in ['Valid', 'Test']
    size = len(dataloader.dataset)
    correct = 0

    model.eval()                    # 关闭 dropout
    with torch.no_grad():           # 关闭梯度计算，更快、更省显存
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            # argmax(1) 沿类别维取最大 → 预测标签；与 y 相等的累加到 correct
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    correct /= size
    print(f"{mode} Accuracy: {(100 * correct):>0.1f}%\n")
    return correct


# -----------------------------------------------------------------------------
# 9. 损失 / 优化器 / 学习率调度器
# -----------------------------------------------------------------------------
loss_fn = nn.CrossEntropyLoss()      # 二分类/多分类都用 CE
optimizer = AdamW(model.parameters(), lr=learning_rate)

# 线性衰减：学习率从 lr 线性降到 0，num_warmup_steps=0 表示不做预热
# 如果训练不稳定，可以把 num_warmup_steps 设为总步数的 10%
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=epoch_num * len(train_dataloader),
)


# -----------------------------------------------------------------------------
# 10. 单句对预测：加载好模型后，对任意一对句子直接输出"是否相似 + 概率"
# -----------------------------------------------------------------------------
# 训练/验证走的是"整批数据 → 平均指标"，而真实部署时更常见的是
# "给我两个句子，告诉我它们像不像"。这个函数就是推理阶段的最小封装。
def predict(sent1, sent2, model, tokenizer):
    # 单条样本也要过 tokenizer，拼成 [CLS] sent1 [SEP] sent2 [SEP]
    inputs = tokenizer(
        sent1, sent2,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(inputs)
        # softmax 把 logits 变成 [不相似概率, 相似概率]
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
    pred = int(probs.argmax())
    return pred, float(probs[pred])


# -----------------------------------------------------------------------------
# 11. 主循环：训练 + 按"验证集最佳"保存 + 末尾加载最佳权重做推理演示
# -----------------------------------------------------------------------------
# 训练模型时，策略是"只保留到目前为止最好的 checkpoint"，而不是每个 epoch 都存。
# 这样不会塞满硬盘，事后也能直接取最优权重去做测试/部署。
#
# 把所有产物落到一个 output 目录里，避免污染工作区根目录。
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pipeline_output')
os.makedirs(output_dir, exist_ok=True)

total_loss = 0.
best_acc = 0.
best_weight_path = None   # 记住最佳权重落盘路径，末尾还要加载它
for t in range(epoch_num):
    print(f"Epoch {t + 1}/{epoch_num}\n-------------------------------")
    total_loss = train_loop(train_dataloader, model, loss_fn, optimizer, lr_scheduler, t + 1, total_loss)
    valid_acc = test_loop(valid_dataloader, model, mode='Valid')
    if valid_acc > best_acc:
        best_acc = valid_acc
        print('saving new weights...\n')
        save_name = f'epoch_{t + 1}_valid_acc_{(100 * valid_acc):0.1f}_model_weights.bin'
        best_weight_path = os.path.join(output_dir, save_name)
        # 只存 state_dict()，体积小；加载时需要先实例化相同结构的 model
        torch.save(model.state_dict(), best_weight_path)
print("Done!")

# -----------------------------------------------------------------------------
# 12. 加载训练期间保存下来的"最佳权重"做一次完整评估 + 两条样本的推理演示
# -----------------------------------------------------------------------------
# 这一步模拟真实部署：训练完不直接用内存里的 model，而是从磁盘重新加载权重，
# 以此验证"保存 → 加载"链路没问题，并展示如何在单条样本上调用 predict。
if best_weight_path is not None:
    print(f'loading best weights from {best_weight_path} (valid acc={100 * best_acc:0.1f}%)')
    model.load_state_dict(torch.load(best_weight_path, map_location=device))
    test_loop(valid_dataloader, model, mode='Test')

    # 从验证集随机挑 2 条句对跑一下 predict，直观看到输出形式
    demo_samples = [valid_data[i] for i in range(min(2, len(valid_data)))]
    for sample in demo_samples:
        pred, prob = predict(sample['sentence1'], sample['sentence2'], model, tokenizer)
        print(
            f"sent1: {sample['sentence1']}\n"
            f"sent2: {sample['sentence2']}\n"
            f"gold : {sample['label']}  |  pred: {pred}  |  prob: {prob:.4f}\n"
        )
else:
    print('no checkpoint was saved (best_a'
          'cc never improved), skipping final evaluation.')