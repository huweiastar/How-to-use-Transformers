"""
===============================================================================
模型定义模块：Token 分类模型用于抽取式阅读理解
===============================================================================

【核心思想】
Extractive QA 是一个 Token Classification 任务：
  - 输入：[CLS] question [SEP] context [SEP]
  - 输出：对 context 中的每个 token 预测
    * start_logits: 是否为答案的起始 token
    * end_logits: 是否为答案的结束 token

模型结构：
  BERT encoder → [batch_size, seq_len, hidden_size]
               → Linear(hidden_size → 2) → [batch_size, seq_len, 2]
               → split → start_logits, end_logits

损失函数：
  使用两个独立的 CrossEntropyLoss：
  L = (L_start + L_end) / 2

推理时：
  找 start_logits 最高的 token 和 end_logits 最高的 token 组成答案。
  但需要：end_index >= start_index 且长度 <= max_answer_length。
"""
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertPreTrainedModel, BertModel

class BertForExtractiveQA(BertPreTrainedModel):
    """
    用于抽取式阅读理解的 BERT 模型

    模型输出答案在 context 中的起始和结束位置（token 级别）。
    """

    def __init__(self, config, args):
        """
        初始化模型

        参数:
            config: BERT 的配置对象
            args: 包含 num_labels 等参数的对象
        """
        super().__init__(config)

        # 标签数：2（start 和 end）
        self.num_labels = args.num_labels

        # BERT 编码器（不使用默认的 pooling 层）
        # add_pooling_layer=False：我们需要完整的序列输出，不需要 [CLS] 的汇总向量
        self.bert = BertModel(config, add_pooling_layer=False)

        # Dropout：正则化，缓解过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # 分类器：将每个 token 的隐藏表示映射到 num_labels（2）维
        # num_labels = 2: logits[:, :, 0] = start_logits, logits[:, :, 1] = end_logits
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化新加的层（classifier）
        self.post_init()

    def forward(self, batch_inputs, start_positions=None, end_positions=None):
        """
        前向传播

        参数:
            batch_inputs (dict): 分词后的输入，包含：
                - input_ids: token 索引
                - attention_mask: 注意力掩码
                - token_type_ids: token 类型 ID（0 = question, 1 = context）
            start_positions (torch.Tensor, optional): 答案起点的 token 位置，形状 [batch_size]
            end_positions (torch.Tensor, optional): 答案终点的 token 位置，形状 [batch_size]

        返回值:
            tuple: (loss, start_logits, end_logits)
                - loss: 训练时的 CrossEntropyLoss（如果提供标签）
                - start_logits: [batch_size, seq_len] 答案起点的 logits
                - end_logits: [batch_size, seq_len] 答案终点的 logits
        """

        # ============ 前向传播：通过 BERT ============
        bert_output = self.bert(**batch_inputs)

        # last_hidden_state: [batch_size, seq_len, hidden_size]
        # 每个 token 的上下文相关的表示
        sequence_output = bert_output.last_hidden_state

        # ============ Dropout ============
        sequence_output = self.dropout(sequence_output)

        # ============ 分类层：每个 token 输出 2 个 logits ============
        # logits: [batch_size, seq_len, 2]
        logits = self.classifier(sequence_output)

        # ============ 拆分 start_logits 和 end_logits ============
        # split(1, dim=-1): 沿最后一维分割，分成两部分各 1 维
        # start_logits: [batch_size, seq_len, 1] → squeeze → [batch_size, seq_len]
        # end_logits: [batch_size, seq_len, 1] → squeeze → [batch_size, seq_len]
        start_logits, end_logits = logits.split(1, dim=-1)

        # squeeze(-1): 移除最后一维（大小为 1 的维度）
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        # ============ 计算损失（仅训练时） ============
        loss = None
        if start_positions is not None and end_positions is not None:
            # 使用两个独立的 CrossEntropyLoss：
            # - L_start: 预测的 start_logits 与真实 start_positions 的交叉熵
            # - L_end: 预测的 end_logits 与真实 end_positions 的交叉熵
            loss_fct = CrossEntropyLoss()

            # start_loss: [batch_size] → scalar
            start_loss = loss_fct(start_logits, start_positions)

            # end_loss: [batch_size] → scalar
            end_loss = loss_fct(end_logits, end_positions)

            # 平均两个损失
            loss = (start_loss + end_loss) / 2

        return loss, start_logits, end_logits
