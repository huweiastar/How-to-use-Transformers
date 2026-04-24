"""
===============================================================================
run_summarization_mt5.py —— 文本摘要完整训练/评估/预测脚本
===============================================================================

本脚本展示 Seq2Seq 文本摘要的完整工程流程：

【工作流程】
  1. 加载预训练 mT5 / BART 等 Seq2Seq 模型
  2. 训练阶段：过训练集，计算损失，每个 epoch 在验证集上评估 ROUGE 指标，保留最佳权重
  3. 测试阶段：加载保存的权重，在测试集上评估 ROUGE（生成摘要后与真实摘要对比）
  4. 预测阶段：对测试集生成摘要，保存为 JSON Lines 格式

【核心概念】
  - Seq2Seq：encoder 读原文，decoder 生成摘要
  - Teacher Forcing：训练时给 decoder 正确答案，推理时自回归生成
  - Beam Search：生成时保留多个候选，选最高概率的
  - ROUGE：文本摘要评估指标，基于 n-gram 重叠度

【初学者易踩的坑】
  1. 训练损失和验证 ROUGE 的数值范围完全不同，不能直观对比
  2. model.generate() 是推理，不计算损失；model() 是训练，计算损失
  3. Beam search 会显著降速，但通常能提升 ROUGE
  4. 标签中的 -100 标记会被损失函数忽略，这是设计特性
"""
import os
import logging
import json
from tqdm.auto import tqdm
import numpy as np
import torch
from transformers import get_scheduler
from torch.optim import AdamW  # PyTorch 原生 AdamW，完全等价于 transformers.AdamW（已弃用）
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rouge import Rouge
import sys
sys.path.append('../../')
from src.tools import seed_everything
from src.seq2seq_summarization.arg import parse_args
from src.seq2seq_summarization.data import LCSTS, get_dataLoader

# ============ 日志配置 ============
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger("Model")

def train_loop(args, dataloader, model, optimizer, lr_scheduler, epoch, total_loss):
    """
    训练一个 epoch

    参数:
        args: 包含 device 等参数的对象
        dataloader: 训练数据加载器
        model: Seq2Seq 模型（mT5 / BART 等）
        optimizer: AdamW 优化器
        lr_scheduler: 学习率调度器
        epoch (int): 当前 epoch 编号（从 0 开始）
        total_loss (float): 累积损失值（用于计算跨 epoch 的平均损失）

    返回值:
        float: 更新后的累积损失值
    """
    # 进度条
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f'loss: {0:>7f}')

    # 之前 epoch 走过的总 batch 数（用于计算"跨 epoch 的平均损失"）
    finish_batch_num = epoch * len(dataloader)

    # 设置模型为训练模式（启用 dropout / batchnorm 等）
    model.train()

    # 遍历每个 batch
    for batch, batch_data in enumerate(dataloader, start=1):
        # 把 batch 数据移到设备上（GPU / CPU）
        batch_data = batch_data.to(args.device)

        # ============ 前向传播 ============
        # model() 计算损失（与 model.generate() 推理不同）
        # 返回的 outputs 包含 loss 和 logits
        outputs = model(**batch_data)
        loss = outputs.loss

        # ============ 反向传播和优化 ============
        # PyTorch 训练三件套：
        # 1. 清零梯度（防止累积）
        # 2. 反向传播（计算梯度）
        # 3. 参数更新（一步）
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 更新学习率（如果使用了 warmup + decay，每一步都要调用）
        lr_scheduler.step()

        # ============ 损失统计 ============
        # 累积这一 batch 的损失
        total_loss += loss.item()

        # 更新进度条，显示截至目前的平均损失
        progress_bar.set_description(f'loss: {total_loss/(finish_batch_num + batch):>7f}')
        progress_bar.update(1)

    return total_loss

def test_loop(args, dataloader, model, tokenizer):
    """
    评估阶段：在验证 / 测试集上评估模型，计算 ROUGE 指标

    【关键概念】
    这里使用的是 model.generate()，不同于训练时的 model()：
      - model(): 计算损失，用于反向传播和参数更新（训练）
      - model.generate(): 自回归生成摘要，返回生成的 token IDs（评估 / 推理）

    ROUGE 是文本摘要的标准评估指标，基于 n-gram 重叠：
      - ROUGE-1: 单词级别重叠（1-gram）
      - ROUGE-2: 词对级别重叠（2-gram）
      - ROUGE-L: 最长公共子序列（考虑句子顺序）
    一般关注 ROUGE-1, ROUGE-2 和 ROUGE-L 中的 F1 分数。

    参数:
        args: 包含 device, max_target_length, beam_search_size 等参数
        dataloader: 验证 / 测试数据加载器
        model: Seq2Seq 模型
        tokenizer: 分词器（用于把 token IDs 解码回文本）

    返回值:
        dict: 包含 ROUGE-1, ROUGE-2, ROUGE-L 和平均分数的字典
              例：{'rouge-1': 45.2, 'rouge-2': 32.1, 'rouge-l': 40.5, 'avg': 39.3}
    """
    # 存储所有样本的预测摘要和真实摘要
    preds, labels = [], []

    # 创建 ROUGE 计算器
    rouge = Rouge()

    # 设置模型为评估模式（关闭 dropout 等）
    model.eval()

    # 不计算梯度，节省显存
    with torch.no_grad():
        # 遍历验证 / 测试集
        for batch_data in tqdm(dataloader):
            # 移到设备上
            batch_data = batch_data.to(args.device)

            # ============ 使用 model.generate() 生成摘要 ============
            # 关键参数：
            # - input_ids: encoder 的输入（原文 tokens）
            # - attention_mask: encoder 的注意力掩码（哪些位置是真实 token，哪些是 padding）
            # - max_length: 生成摘要的最大长度
            # - num_beams: Beam Search 宽度（>1 时启用 Beam Search）
            # - no_repeat_ngram_size: 禁止 n-gram 重复（防止 "的的的" 这样的重复）
            #
            # 返回值：形状为 [batch_size, max_length] 的生成 token IDs
            generated_tokens = model.generate(
                batch_data["input_ids"],
                attention_mask=batch_data["attention_mask"],
                max_length=args.max_target_length,
                num_beams=args.beam_search_size,        # Beam Search 宽度
                no_repeat_ngram_size=args.no_repeat_ngram_size,  # 禁止 n-gram 重复
            ).cpu().numpy()

            # 有些模型 generate() 可能返回 tuple（tokens, scores 等），需要取第一个
            if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]

            # 获取标签（真实摘要）的 token IDs，它们包含 -100 标记（要忽略的位置）
            label_tokens = batch_data["labels"].cpu().numpy()

            # ============ 解码：token IDs → 文本 ============
            # tokenizer.batch_decode() 把 token IDs 转回文本
            # skip_special_tokens=True: 不显示 [CLS], [SEP], <pad> 等特殊令牌
            # clean_up_tokenization_spaces=False: 不自动处理 tokenizer 可能加的空格
            decoded_preds = tokenizer.batch_decode(
                generated_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )

            # 标签中的 -100 不是真实 token，替换为 padding token ID
            # （之前在 data.py 中设置 -100 是为了让损失函数忽略这些位置）
            label_tokens = np.where(label_tokens != -100, label_tokens, tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(
                label_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )

            # 空格处理：把摘要分割成词，再用空格连接
            # 这样可以保证词之间有明确的分界（ROUGE 计算时需要）
            preds += [' '.join(pred.strip()) for pred in decoded_preds]
            labels += [' '.join(label.strip()) for label in decoded_labels]

    # ============ 计算 ROUGE ============
    # rouge.get_scores(hyps=预测, refs=真实, avg=True) 计算所有样本的 ROUGE，然后平均
    scores = rouge.get_scores(hyps=preds, refs=labels, avg=True)

    # 提取 F1 分数（每个 ROUGE 指标都有 precision, recall, f）
    # 乘以 100 转换为百分比
    result = {key: value['f'] * 100 for key, value in scores.items()}

    # 计算三个指标的平均值
    result['avg'] = np.mean(list(result.values()))

    return result

def train(args, train_dataset, dev_dataset, model, tokenizer):
    """
    完整训练流程：包括数据加载、优化器配置、多 epoch 训练循环、验证集评估、权重保存

    参数:
        args: 所有超参数和配置
        train_dataset: 训练数据集
        dev_dataset: 验证数据集
        model: Seq2Seq 模型
        tokenizer: 分词器
    """
    # ============ 创建数据加载器 ============
    # shuffle=True for train（增加随机性），shuffle=False for dev（保证可复现）
    train_dataloader = get_dataLoader(args, train_dataset, model, tokenizer, shuffle=True)
    dev_dataloader = get_dataLoader(args, dev_dataset, model, tokenizer, shuffle=False)

    # ============ 计算总优化步数 ============
    # 总步数 = 每个 epoch 的 batch 数 × epoch 数
    # 用于学习率调度器的 warmup 和 decay 计算
    t_total = len(train_dataloader) * args.num_train_epochs

    # ============ 配置优化器（两组参数，不同的权重衰减） ============
    # 工程实践：bias 和 LayerNorm.weight 不应用权重衰减
    # 原因：这些参数的梯度已经足够稳定，加权重衰减反而可能让优化变得不稳定
    no_decay = ["bias", "LayerNorm.weight"]

    # 把模型参数分成两组
    optimizer_grouped_parameters = [
        {
            # 大部分参数：应用权重衰减（L2 正则化）
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay
        },
        {
            # bias 和 LayerNorm.weight：不应用权重衰减
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        }
    ]

    # 计算预热步数（前 10% 的步数用于线性预热学习率）
    args.warmup_steps = int(t_total * args.warmup_proportion)

    # 创建 AdamW 优化器
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),  # (β1, β2) = (动量, 二阶动量)
        eps=args.adam_epsilon  # 数值稳定性参数
    )

    # ============ 配置学习率调度器 ============
    # "linear": 线性预热 + 线性衰减
    # 前 warmup_steps：学习率从 0 线性升到 lr
    # 后续：学习率线性衰减到 0
    lr_scheduler = get_scheduler(
        'linear',
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=t_total
    )

    # ============ 记录训练信息 ============
    logger.info("***** Running training *****")
    logger.info(f"Num examples - {len(train_dataset)}")
    logger.info(f"Num Epochs - {args.num_train_epochs}")
    logger.info(f"Total optimization steps - {t_total}")
    logger.info(f"Warmup steps - {args.warmup_steps}")

    # 保存参数到文件（便于复现）
    with open(os.path.join(args.output_dir, 'args.txt'), 'wt', encoding='utf-8') as f:
        f.write(str(args))

    # ============ 主训练循环 ============
    total_loss = 0.0       # 累积损失（用于计算跨 epoch 的平均损失）
    best_avg_rouge = 0.0   # 最佳验证 ROUGE 分数（用于保存最佳权重）

    for epoch in range(args.num_train_epochs):
        print(f"Epoch {epoch+1}/{args.num_train_epochs}\n" + 30 * "-")

        # 执行一个 epoch 的训练
        total_loss = train_loop(args, train_dataloader, model, optimizer, lr_scheduler, epoch, total_loss)

        # 在验证集上评估（计算 ROUGE）
        dev_rouges = test_loop(args, dev_dataloader, model, tokenizer)

        # 记录验证结果
        logger.info(
            f"Dev Rouge1: {dev_rouges['rouge-1']:>0.2f} "
            f"Rouge2: {dev_rouges['rouge-2']:>0.2f} "
            f"RougeL: {dev_rouges['rouge-l']:>0.2f} "
            f"Avg: {dev_rouges['avg']:>0.2f}"
        )

        # ============ 按验证集最佳分数保存权重 ============
        # 策略：只保留"最好的"checkpoint，不是每个 epoch 都存
        # 这样既节省磁盘空间，也方便事后直接加载最优权重
        rouge_avg = dev_rouges['avg']
        if rouge_avg > best_avg_rouge:
            best_avg_rouge = rouge_avg
            logger.info(f'saving new best weights to {args.output_dir}...\n')

            # 文件名包含 epoch 和 ROUGE 分数，方便追踪
            save_weight = f'epoch_{epoch+1}_dev_rouge_avg_{rouge_avg:0.4f}_weights.bin'
            torch.save(model.state_dict(), os.path.join(args.output_dir, save_weight))

    logger.info("Done!")

def test(args, test_dataset, model, tokenizer, save_weights: list):
    """
    在测试集上评估所有保存的权重

    参数:
        args: 参数对象
        test_dataset: 测试数据集
        model: 模型（会被加载不同的权重进行重复评估）
        tokenizer: 分词器
        save_weights: 保存的权重文件名列表
    """
    # 创建测试数据加载器
    test_dataloader = get_dataLoader(args, test_dataset, model, tokenizer, shuffle=False)

    logger.info('***** Running testing *****')

    # 遍历每个保存的权重，逐个在测试集上评估
    # 这样可以对比不同 epoch 的模型在测试集上的表现
    for save_weight in save_weights:
        logger.info(f'loading weights from {save_weight}...')

        # 加载权重
        model.load_state_dict(torch.load(os.path.join(args.output_dir, save_weight), map_location=args.device))

        # 评估
        test_rouges = test_loop(args, test_dataloader, model, tokenizer)

        # 记录结果
        logger.info(
            f"Test Rouge1: {test_rouges['rouge-1']:>0.2f} "
            f"Rouge2: {test_rouges['rouge-2']:>0.2f} "
            f"RougeL: {test_rouges['rouge-l']:>0.2f} "
            f"Avg: {test_rouges['avg']:>0.2f}"
        )

def predict(args, document: str, model, tokenizer):
    """
    对单篇文档生成摘要

    这是推理的最小单位：给定一篇原文，输出对应的摘要。

    参数:
        args: 包含 device, max_input_length, max_target_length 等参数
        document (str): 要摘要的原文（中文字符串）
        model: Seq2Seq 模型
        tokenizer: 分词器

    返回值:
        str: 生成的摘要文本
    """
    # ============ 编码输入 ============
    # 把原文分词、padding、转为 token IDs
    inputs = tokenizer(
        document,
        max_length=args.max_input_length,  # 超过长度则截断
        truncation=True,
        return_tensors="pt"  # 返回 PyTorch 张量
    )

    # 移到设备上
    inputs = inputs.to(args.device)

    # ============ 生成摘要 ============
    with torch.no_grad():
        # 模型自回归生成，返回 token IDs
        generated_tokens = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=args.max_target_length,
            num_beams=args.beam_search_size,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
        ).cpu().numpy()

    # 有些模型返回 tuple，取第一个元素
    if isinstance(generated_tokens, tuple):
        generated_tokens = generated_tokens[0]

    # ============ 解码输出 ============
    # 把生成的 token IDs 解码回文本
    # 注意：tokenizer.decode() 是单样本；tokenizer.batch_decode() 是批处理
    # 这里是单样本，用 [0] 取第一个样本的结果
    decoded_preds = tokenizer.decode(
        generated_tokens[0],  # 第一个（唯一）样本的 tokens
        skip_special_tokens=True,  # 不显示 [CLS], <pad> 等
        clean_up_tokenization_spaces=False
    )

    return decoded_preds

if __name__ == '__main__':
    """
    主程序入口：支持三种模式 --do_train / --do_test / --do_predict

    使用示例：
      python run_summarization_mt5.py \\
        --output_dir ./checkpoints \\
        --train_file ../data/train.txt \\
        --dev_file ../data/dev.txt \\
        --test_file ../data/test.txt \\
        --model_checkpoint google/mt5-small \\
        --do_train --do_test --do_predict

    三种模式可单独运行，也可一起运行：
      - --do_train: 执行训练，在验证集上选最佳权重
      - --do_test: 对所有保存的权重在测试集上评估 ROUGE
      - --do_predict: 对测试集生成摘要，保存为 JSON
    """

    # ============ 解析命令行参数 ============
    args = parse_args()

    # ============ 设置输出目录 ============
    # 安全检查：如果目录已存在且非空，且要求训练，则报错（防止覆盖之前的结果）
    if args.do_train and os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError(
            f'Output directory ({args.output_dir}) already exists and is not empty. '
            f'Please use a new directory or remove old files.'
        )

    # 创建输出目录
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # ============ 设置设备和 GPU 数量 ============
    # 优先级：CUDA > CPU
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.n_gpu = torch.cuda.device_count()
    logger.warning(f'Using {args.device} device, n_gpu: {args.n_gpu}')

    # ============ 设置随机种子（用于可复现性） ============
    # 固定所有随机源（Python, NumPy, PyTorch, CUDA）
    seed_everything(args.seed)

    # ============ 加载预训练模型和分词器 ============
    logger.info(f'loading pretrained model and tokenizer of {args.model_type} ...')

    # 从 HuggingFace Hub 或本地路径加载分词器
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)

    # 从 HuggingFace Hub 或本地路径加载 Seq2Seq 模型（mT5, BART 等）
    # AutoModelForSeq2SeqLM 会自动选择合适的模型类
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_checkpoint).to(args.device)

    logger.info(f'Model loaded: {model.__class__.__name__}')
    logger.info(f'Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M')

    # ============ 执行训练 ============
    if args.do_train:
        logger.info("***** Loading training and validation datasets *****")

        # 加载训练和验证数据集
        train_dataset = LCSTS(args.train_file)
        dev_dataset = LCSTS(args.dev_file)

        logger.info(f"Loaded {len(train_dataset)} training examples")
        logger.info(f"Loaded {len(dev_dataset)} validation examples")

        # 执行训练
        train(args, train_dataset, dev_dataset, model, tokenizer)

    # ============ 收集所有保存的权重文件 ============
    # 用于 test 和 predict 阶段
    save_weights = [file for file in os.listdir(args.output_dir) if file.endswith('.bin')]
    logger.info(f'Found {len(save_weights)} saved checkpoints: {save_weights}')

    # ============ 执行测试 ============
    if args.do_test:
        logger.info("***** Loading test dataset *****")

        # 加载测试数据集
        test_dataset = LCSTS(args.test_file)
        logger.info(f"Loaded {len(test_dataset)} test examples")

        # 在测试集上评估所有保存的权重
        test(args, test_dataset, model, tokenizer, save_weights)

    # ============ 执行预测 ============
    if args.do_predict:
        logger.info("***** Running prediction *****")

        # 加载测试数据集（用于生成摘要）
        test_dataset = LCSTS(args.test_file)
        logger.info(f"Loaded {len(test_dataset)} test examples for prediction")

        # 对每个保存的权重进行预测
        for save_weight in save_weights:
            logger.info(f'loading weights from {save_weight}...')

            # 加载模型权重到内存
            model.load_state_dict(torch.load(os.path.join(args.output_dir, save_weight), map_location=args.device))
            logger.info(f'predicting with {save_weight}...')

            # 存储预测结果（每个样本：原文、预测摘要、真实摘要）
            results = []

            # 设置模型为评估模式（关闭 dropout 等）
            model.eval()

            # 对测试集的每个样本进行预测
            for s_idx in tqdm(range(len(test_dataset)), desc='Predicting'):
                # 获取单个样本
                sample = test_dataset[s_idx]

                # 对该样本的原文生成摘要
                pred_summ = predict(args, sample['content'], model, tokenizer)

                # 保存原文、预测摘要、真实摘要
                results.append({
                    "content": sample['content'],         # 原文
                    "prediction": pred_summ,              # 模型生成的摘要
                    "gold_summary": sample['title']       # 真实摘要（用于后续评估）
                })

            # ============ 将预测结果保存到 JSON 文件 ============
            # 文件名格式：{权重文件名}_predictions.json
            output_file = os.path.join(args.output_dir, save_weight + '_predictions.json')

            # 按 JSON Lines 格式保存（每行一个 JSON 对象）
            # 这样可以流式处理大文件
            with open(output_file, 'wt', encoding='utf-8') as f:
                for result in results:
                    # json.dumps() 转为 JSON 字符串，ensure_ascii=False 保证中文正常显示
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')

            logger.info(f'Predictions saved to {output_file}')

    logger.info("All done!")
