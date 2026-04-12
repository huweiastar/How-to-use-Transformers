"""
模型定义模块
包含基于 BERT 和 RoBERTa 的句对相似度分类模型。

======================================================================
【初学者重点】为什么要继承 BertPreTrainedModel，而不是写 nn.Module？
======================================================================
两种写法都能跑，但继承 BertPreTrainedModel 有三个关键好处：
  1) 可以直接用  Model.from_pretrained("bert-base-chinese")  一键加载
     预训练权重——底层的 BERT 权重自动填好，新增的 classifier 保持随机
     初始化，这正是"微调"的标准起点。
  2) self.post_init() 会按照 config 里的方式正确初始化新加的层
     （比如 Linear 的 w/b），避免手写初始化踩坑。
  3) 能被 save_pretrained / AutoConfig / HuggingFace Hub 等生态识别。

句对分类的通用套路（本文件就是这个套路的最小实现）：
    input  →  BERT encoder  →  取 [CLS] 向量  →  Dropout  →  Linear → logits
======================================================================
"""
import torch
from torch import nn
from transformers import BertPreTrainedModel, BertModel
from transformers import RobertaPreTrainedModel, RobertaModel


class BertForPairwiseCLS(BertPreTrainedModel):
    """
    基于 BERT 的句对分类模型（文本相似度判断，0=不相似，1=相似）。

    模型结构（自底向上）：
      1. BertModel         ：把 [CLS] sent1 [SEP] sent2 [SEP] 编码为隐层序列
      2. 取 [CLS] 向量     ：last_hidden_state[:, 0, :]，代表"整句对"的语义 ，(batch_size, seq_len, hidden_size)
      3. Dropout           ：正则化，缓解过拟合
      4. Linear(hidden, N) ：把 CLS 向量线性映射到 num_labels 维 logits
    """

    def __init__(self, config, args):
        """
        初始化BERT分类模型

        参数:
            config: BERT模型的配置对象
            args: 包含num_labels（标签数量）等参数的对象
        """
        super().__init__(config)

        # add_pooling_layer=False：我们自己取 [CLS]，不需要 BERT 自带的
        # pooler（pooler 会在 [CLS] 后多套一个 Tanh，常见微调任务里反而多余）
        self.bert = BertModel(config, add_pooling_layer=False)

        # 用 config 里定义的 dropout 概率，保持与预训练一致（通常是 0.1）
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # 分类头：hidden_size → num_labels
        #   bert-base-chinese 的 hidden_size = 768
        #   bert-large       的 hidden_size = 1024
        # 用 config.hidden_size 而不是硬编码 768，换模型时就不用改代码
        self.classifier = nn.Linear(config.hidden_size, args.num_labels)

        # 必须调用：让新加的 classifier 按 config 正确初始化
        self.post_init()

    def forward(self, batch_inputs, labels=None):
        """
        前向传播

        参数:
            batch_inputs (dict): 分词后的输入，包含：
                - input_ids: token索引
                - attention_mask: 注意力掩码（1表示实际token，0表示填充）
                - token_type_ids: token类型索引（第一句为0，第二句为1）
            labels (torch.Tensor, optional): 目标标签，如果提供则计算损失

        返回值:
            tuple: (loss, logits)
                - loss (torch.Tensor or None): 交叉熵损失，如果labels为None则为None
                - logits (torch.Tensor): 分类logits，形状为[batch_size, num_labels]
        """
        # 把整个 batch_inputs 字典解包成 input_ids / attention_mask /
        # token_type_ids 三个关键字参数传给 BERT
        outputs = self.bert(**batch_inputs)

        # [:, 0, :] 切片的三个维度对应 (batch, seq, hidden)：
        #   - 第 1 维 0：batch 内每一条样本
        #   - 第 2 维 0：取 seq_len 里的第 0 个位置，也就是 [CLS]
        #   - 第 3 维 :：保留整条 hidden 向量
        # 结果 cls_vectors 形状: [batch_size, hidden_size]
        cls_vectors = outputs.last_hidden_state[:, 0, :]

        # 训练时 dropout 随机失活；eval() 模式下会自动失效
        cls_vectors = self.dropout(cls_vectors)

        # logits 形状: [batch_size, num_labels]（还没 softmax，CE loss 内部做）
        logits = self.classifier(cls_vectors)

        # 把损失计算也放进模型里，训练循环就能直接 loss.backward()，
        # 这是 HuggingFace 官方模型的通用做法（看齐 *ForSequenceClassification）。
        loss = None
        if labels is not None:
            if not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels, device=logits.device)
            loss = nn.CrossEntropyLoss()(logits, labels)

        return loss, logits


class RobertaForPairwiseCLS(RobertaPreTrainedModel):
    """
    基于RoBERTa的句对分类模型
    用于文本相似度判断（0表示不相似，1表示相似）

    模型结构与BertForPairwiseCLS类似，但使用RoBERTa预训练模型：
    1. RoBERTa预训练模型：将输入文本转换为隐层表示
    2. Dropout层：防止过拟合
    3. 分类器：线性层将CLS token的表示转换为分类logits

    RoBERTa相比BERT的改进：
    - 更好的预训练策略
    - 更强的性能
    """

    def __init__(self, config, args):
        """
        初始化RoBERTa分类模型

        参数:
            config: RoBERTa模型的配置对象
            args: 包含num_labels（标签数量）等参数的对象
        """
        super().__init__(config)

        # 加载RoBERTa预训练模型（不使用PoolingLayer，我们将使用CLS token）
        self.roberta = RobertaModel(config, add_pooling_layer=False)

        # Dropout层，使用RoBERTa配置中的隐层dropout概率
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # 分类器：线性层将隐层维度映射到标签数量
        self.classifier = nn.Linear(config.hidden_size, args.num_labels)

        # 初始化权重
        self.post_init()

    def forward(self, batch_inputs, labels=None):
        """
        前向传播

        参数:
            batch_inputs (dict): 分词后的输入，包含：
                - input_ids: token索引
                - attention_mask: 注意力掩码（1表示实际token，0表示填充）
                - token_type_ids: token类型索引（第一句为0，第二句为1）
            labels (torch.Tensor, optional): 目标标签，如果提供则计算损失

        返回值:
            tuple: (loss, logits)
                - loss (torch.Tensor or None): 交叉熵损失，如果labels为None则为None
                - logits (torch.Tensor): 分类logits，形状为[batch_size, num_labels]
        """
        # 逻辑与 BertForPairwiseCLS.forward 完全相同，只是底层换成 RoBERTa
        outputs = self.roberta(**batch_inputs)
        cls_vectors = outputs.last_hidden_state[:, 0, :]
        cls_vectors = self.dropout(cls_vectors)
        logits = self.classifier(cls_vectors)

        loss = None
        if labels is not None:
            if not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels, device=logits.device)
            loss = nn.CrossEntropyLoss()(logits, labels)

        return loss, logits


if __name__ == '__main__':
    # 独立运行本文件时的小测试：加载预训练权重并打印模型结构，
    # 帮助初学者看清楚 BERT + 分类头的层次。
    from types import SimpleNamespace
    from transformers import AutoConfig

    # 自动适配 Windows/Linux/Mac
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f'Using {device} device')

    checkpoint = "bert-base-chinese"
    config = AutoConfig.from_pretrained(checkpoint)
    # 用 SimpleNamespace 冒充 args 对象，只为满足 __init__ 需要的 num_labels
    fake_args = SimpleNamespace(num_labels=2)

    # 注意：这里用 from_pretrained 而不是直接 BertForPairwiseCLS(config, fake_args)，
    # 前者会加载预训练权重，后者会得到随机初始化的 BERT。
    model = BertForPairwiseCLS.from_pretrained(
        checkpoint, config=config, args=fake_args
    ).to(device)
    print(model)
