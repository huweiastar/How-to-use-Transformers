"""
===============================================================================
run_extractiveQA.py —— 抽取式阅读理解的完整训练 / 评估 / 预测脚本
===============================================================================

【工作流程】
  1. 训练阶段：在训练集上优化模型，每个 epoch 在验证集评估，保留最佳权重
  2. 测试阶段：加载最佳权重，在测试集上计算 F1 和 EM
  3. 预测阶段：对新样本生成答案

【关键特性】
  - 处理超长文本：使用 stride 分割，多个 chunk 的预测合并
  - Offset mapping：token 位置 → 原文字符位置的映射
  - N-best 束搜索：保留前 n_best 个答案候选，枚举组合找最优
  - 评估指标：F1（token 级匹配）+ EM（完全匹配）

【核心关键概念】
  - feature：一个 token 序列（question + context）
  - example：一个原始数据样本（可对应多个 feature，如果被分割了）
  - chunk：当 context 超长时，用 stride 分割出的分块
"""
import os
import json
import logging
import numpy as np
from tqdm.auto import tqdm
import collections
import torch
from transformers import AutoConfig, AutoTokenizer
from transformers import AdamW, get_scheduler
import sys
sys.path.append('../../')
from src.sequence_labeling_extractiveQA_cmrc.data import CMRC2018, get_dataLoader
from src.sequence_labeling_extractiveQA_cmrc.modeling import BertForExtractiveQA
from src.sequence_labeling_extractiveQA_cmrc.arg import parse_args
from src.sequence_labeling_extractiveQA_cmrc.cmrc2018_evaluate import evaluate
from src.tools import seed_everything

# ============ 日志配置 ============
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger("Model")


def to_device(args, batch_data):
    """
    将 batch 数据移到指定的设备（CPU / GPU）

    参数:
        args: 包含 device 信息的对象
        batch_data (dict): 包含以下可能的键：
            - batch_inputs: dict（input_ids, attention_mask 等）
            - start_positions: list（训练时的标签）
            - end_positions: list（训练时的标签）
            - example_ids: list（推理时的样本 ID）
            - offset_mapping: list（推理时的 offset 映射）

    返回值:
        dict: 所有张量都被移到指定设备的字典
    """
    new_batch_data = {}
    for k, v in batch_data.items():
        if k == 'batch_inputs':
            # batch_inputs 是一个字典，需要对其中的每个张量都调用 .to()
            new_batch_data[k] = {
                k_: v_.to(args.device) for k_, v_ in v.items()
            }
        else:
            # 其他字段（start_positions, end_positions 等）先转为张量再移到设备
            new_batch_data[k] = torch.tensor(v).to(args.device)
    return new_batch_data

def train_loop(args, dataloader, model, optimizer, lr_scheduler, epoch, total_loss):
    """
    训练一个 epoch

    参数:
        args: 包含 device 等参数的对象
        dataloader: 训练数据加载器
        model: 抽取式 QA 模型
        optimizer: AdamW 优化器
        lr_scheduler: 学习率调度器
        epoch (int): 当前 epoch 编号（从 0 开始）
        total_loss (float): 累积损失值

    返回值:
        float: 更新后的累积损失值
    """
    # 进度条
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f'loss: {0:>7f}')

    # 计算此前已完成的总 batch 数（用于计算跨 epoch 的平均损失）
    finish_batch_num = epoch * len(dataloader)

    # 设置模型为训练模式
    model.train()

    for step, batch_data in enumerate(dataloader, start=1):
        # 移到设备
        batch_data = to_device(args, batch_data)

        # 前向传播：计算损失
        # outputs = (loss, start_logits, end_logits)
        outputs = model(**batch_data)
        loss = outputs[0]

        # ============ 反向传播三件套 ============
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 更新学习率
        lr_scheduler.step()

        # 记录损失
        total_loss += loss.item()
        progress_bar.set_description(f'loss: {total_loss/(finish_batch_num + step):>7f}')
        progress_bar.update(1)

    return total_loss

def test_loop(args, dataloader, dataset, model):
    """
    评估阶段：在验证 / 测试集上评估，计算 F1 和 EM 指标

    【关键步骤】
    1. 收集所有 feature 的预测 logits 和 offset mapping
    2. 构建 example → features 的映射（因为一个 example 可能对应多个 feature）
    3. 对每个 example，收集所有关联 feature 的候选答案
    4. 执行 n-best 搜索：取 top-n 的 start 和 end token，枚举组合找最优
    5. 用 offset mapping 把 token 位置还原为原文字符范围
    6. 计算 F1 和 EM

    参数:
        args: 包含 device, n_best, max_answer_length 等参数
        dataloader: 验证 / 测试数据加载器
        dataset: 对应的数据集（用于获取 context 和真实答案）
        model: 抽取式 QA 模型

    返回值:
        dict: {'f1': ..., 'em': ..., 'avg': ..., 'total': ..., 'skip': ...}
    """

    # ============ 收集所有 feature 的 ID 和 offset mapping ============
    # feature：一个 token 序列。当原始 context 很长时，用 stride 分割成多个 chunk，
    # 每个 chunk 就是一个 feature。
    all_example_ids = []
    all_offset_mapping = []
    for batch_data in dataloader:
        all_example_ids += batch_data['example_ids']
        all_offset_mapping += batch_data['offset_mapping']

    # ============ 建立 example → features 的映射 ============
    # example：原始的 QA 对（可能被分割成多个 feature）
    # 这个映射用于后续合并同一 example 的多个 feature 的预测
    example_to_features = collections.defaultdict(list)
    for idx, feature_id in enumerate(all_example_ids):
        example_to_features[feature_id].append(idx)

    # ============ 推理：收集所有 logits ============
    start_logits = []
    end_logits = []
    model.eval()
    with torch.no_grad():
        for batch_data in tqdm(dataloader):
            # 移除暂时不需要的信息
            batch_data.pop('example_ids')
            batch_data.pop('offset_mapping')

            # 移到设备，前向传播
            batch_data = to_device(args, batch_data)
            outputs = model(**batch_data)

            # outputs = (loss, start_logits, end_logits)
            # 在 eval 模式下，loss = None
            pred_start_logits, pred_end_logit = outputs[1], outputs[2]

            # 保存 logits（转为 numpy 便于后续处理）
            start_logits.append(pred_start_logits.cpu().numpy())
            end_logits.append(pred_end_logit.cpu().numpy())

    # 拼接所有 logits
    start_logits = np.concatenate(start_logits)
    end_logits = np.concatenate(end_logits)

    # ============ 构建真实答案列表 ============
    theoretical_answers = [
        {"id": dataset[s_idx]["id"], "answers": dataset[s_idx]["answers"]}
        for s_idx in range(len(dataset))
    ]

    # ============ 对每个 example 生成预测答案 ============
    predicted_answers = []
    for s_idx in tqdm(range(len(dataset))):
        example_id = dataset[s_idx]["id"]
        context = dataset[s_idx]["context"]
        answers = []  # 候选答案列表

        # ============ 遍历该 example 对应的所有 feature ============
        # （可能被分割成多个 feature / chunk）
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]  # [seq_len]
            end_logit = end_logits[feature_index]      # [seq_len]
            offsets = all_offset_mapping[feature_index]  # [seq_len] 的 offset

            # ============ N-best 搜索 ============
            # 对 start 和 end 分别找前 n_best 个最高概率的位置
            # np.argsort()[-1 : -args.n_best - 1 : -1] 是个快速取 top-n 的技巧：
            # 从末尾开始倒序，每次取一个，共取 n_best 个
            start_indexes = np.argsort(start_logit)[-1 : -args.n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -args.n_best - 1 : -1].tolist()

            # 枚举 start 和 end 的所有有效组合
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # 检查 offset 是否有效（即该位置是否在 context 范围内）
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue

                    # 检查答案长度是否合法
                    # 条件 1：end_index >= start_index（答案不能倒序）
                    # 条件 2：答案长度 <= max_answer_length（不能太长）
                    if end_index < start_index or end_index - start_index + 1 > args.max_answer_length:
                        continue

                    # 这是一个有效的答案候选
                    # offset 是 [start_char, end_char] 形式，可以直接用于切割原文
                    answers.append({
                        "start": offsets[start_index][0],  # 答案在原文中的起始字符位置
                        "text": context[offsets[start_index][0] : offsets[end_index][1]],  # 从原文切割答案
                        "logit_score": start_logit[start_index] + end_logit[end_index],  # 置信度得分
                    })

        # ============ 从所有候选中选择最优答案 ============
        if len(answers) > 0:
            # 选择置信度最高的答案
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append({
                "id": example_id,
                "prediction_text": best_answer["text"],
                "answer_start": best_answer["start"]
            })
        else:
            # 没有候选答案（极少见），返回空答案
            predicted_answers.append({
                "id": example_id,
                "prediction_text": "",
                "answer_start": 0
            })

    # ============ 计算 F1 和 EM ============
    return evaluate(predicted_answers, theoretical_answers)

def train(args, train_dataset, dev_dataset, model, tokenizer):
    """
    完整的训练流程：包括数据加载、优化器配置、多 epoch 训练、保存最佳权重

    参数:
        args: 所有超参数和配置
        train_dataset: 训练数据集
        dev_dataset: 验证数据集
        model: 抽取式 QA 模型
        tokenizer: 分词器
    """
    # ============ 创建数据加载器 ============
    train_dataloader = get_dataLoader(args, train_dataset, tokenizer, mode='train', shuffle=True)
    dev_dataloader = get_dataLoader(args, dev_dataset, tokenizer, mode='valid', shuffle=False)

    # ============ 计算总优化步数 ============
    t_total = len(train_dataloader) * args.num_train_epochs

    # ============ 配置优化器 ============
    # bias 和 LayerNorm.weight 不应用权重衰减
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        }
    ]

    args.warmup_steps = int(t_total * args.warmup_proportion)
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon
    )

    # ============ 配置学习率调度器 ============
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

    # 保存参数到文件
    with open(os.path.join(args.output_dir, 'args.txt'), 'wt', encoding='utf-8') as f:
        f.write(str(args))

    # ============ 主训练循环 ============
    total_loss = 0.0
    best_avg_score = 0.0

    for epoch in range(args.num_train_epochs):
        print(f"Epoch {epoch+1}/{args.num_train_epochs}\n" + 30 * "-")

        # 执行一个 epoch 的训练
        total_loss = train_loop(args, train_dataloader, model, optimizer, lr_scheduler, epoch, total_loss)

        # 在验证集上评估
        metrics = test_loop(args, dev_dataloader, dev_dataset, model)
        F1_score, EM_score, avg_score = metrics['f1'], metrics['em'], metrics['avg']
        logger.info(f'Dev: F1 - {F1_score:0.4f} EM - {EM_score:0.4f} AVG - {avg_score:0.4f}')

        # ============ 按最佳分数保存权重 ============
        if avg_score > best_avg_score:
            best_avg_score = avg_score
            logger.info(f'saving new weights to {args.output_dir}...\n')
            save_weight = f'epoch_{epoch+1}_dev_f1_{F1_score:0.4f}_em_{EM_score:0.4f}_weights.bin'
            torch.save(model.state_dict(), os.path.join(args.output_dir, save_weight))

    logger.info("Done!")

def test(args, test_dataset, model, tokenizer, save_weights: list):
    """
    在测试集上评估所有保存的权重

    参数:
        args: 参数对象
        test_dataset: 测试数据集
        model: 模型
        tokenizer: 分词器
        save_weights: 保存的权重文件名列表
    """
    # 创建测试数据加载器
    test_dataloader = get_dataLoader(args, test_dataset, tokenizer, mode='test', shuffle=False)

    logger.info('***** Running testing *****')

    # 逐个加载权重并评估
    for save_weight in save_weights:
        logger.info(f'loading weights from {save_weight}...')
        model.load_state_dict(torch.load(os.path.join(args.output_dir, save_weight), map_location=args.device))

        # 评估
        metrics = test_loop(args, test_dataloader, test_dataset, model)
        F1_score, EM_score, avg_score = metrics['f1'], metrics['em'], metrics['avg']
        logger.info(f'Test: F1 - {F1_score:0.4f} EM - {EM_score:0.4f} AVG - {avg_score:0.4f}')

def predict(args, context: str, question: str, model, tokenizer):
    """
    对单个问题进行预测，返回答案

    这是推理的最小单位：给定 context 和 question，输出预测的答案及其位置。

    参数:
        args: 包含 device, max_length, stride, n_best, max_answer_length 等参数
        context (str): 原文
        question (str): 问题
        model: 抽取式 QA 模型
        tokenizer: 分词器

    返回值:
        dict: {
          'prediction_text': 预测的答案文本,
          'answer_start': 答案在原文中的起始字符位置
        }
    """

    # ============ 分词 ============
    # 使用与训练相同的方式进行分词（启用 stride）
    inputs = tokenizer(
        question,
        context,
        max_length=args.max_length,
        truncation="only_second",
        stride=args.stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
        return_tensors="pt"
    )

    # 当 context 很长时，会被分割成多个 chunk
    chunk_num = inputs['input_ids'].shape[0]

    # ============ 处理 offset_mapping ============
    inputs.pop('overflow_to_sample_mapping')  # 不需要这个
    offset_mapping = inputs.pop('offset_mapping').numpy().tolist()

    # 只保留 context 部分的 offset（question 和 padding 部分设为 None）
    for i in range(chunk_num):
        sequence_ids = inputs.sequence_ids(i)
        offset = offset_mapping[i]
        offset_mapping[i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    # ============ 准备输入 ============
    inputs = {'batch_inputs': inputs}
    inputs = to_device(args, inputs)

    # ============ 推理 ============
    with torch.no_grad():
        # outputs = (loss, start_logits, end_logits)
        # 在 eval 模式下，loss = None
        _, pred_start_logits, pred_end_logit = model(**inputs)
        start_logits = pred_start_logits.cpu().numpy()
        end_logits = pred_end_logit.cpu().numpy()

    # ============ 生成候选答案 ============
    answers = []
    for feature_index in range(chunk_num):
        start_logit = start_logits[feature_index]
        end_logit = end_logits[feature_index]
        offsets = offset_mapping[feature_index]

        # N-best 搜索
        start_indexes = np.argsort(start_logit)[-1 : -args.n_best - 1 : -1].tolist()
        end_indexes = np.argsort(end_logit)[-1 : -args.n_best - 1 : -1].tolist()

        for start_index in start_indexes:
            for end_index in end_indexes:
                # 检查有效性
                if offsets[start_index] is None or offsets[end_index] is None:
                    continue
                if end_index < start_index or end_index - start_index + 1 > args.max_answer_length:
                    continue

                answers.append({
                    "start": offsets[start_index][0],
                    "text": context[offsets[start_index][0] : offsets[end_index][1]],
                    "logit_score": start_logit[start_index] + end_logit[end_index],
                })

    # ============ 选择最优答案 ============
    if len(answers) > 0:
        best_answer = max(answers, key=lambda x: x["logit_score"])
        return {
            "prediction_text": best_answer["text"],
            "answer_start": best_answer["start"]
        }
    else:
        # 没有候选答案
        return {
            "prediction_text": "",
            "answer_start": 0
        }
    
if __name__ == '__main__':
    """
    主程序入口：支持三种模式 --do_train / --do_test / --do_predict

    使用示例：
      python run_extractiveQA.py \\
        --output_dir ./checkpoints \\
        --train_file ../data/cmrc_train.json \\
        --dev_file ../data/cmrc_dev.json \\
        --test_file ../data/cmrc_test.json \\
        --model_checkpoint bert-base-chinese \\
        --max_length 512 \\
        --stride 128 \\
        --do_train --do_test --do_predict
    """

    # ============ 解析命令行参数 ============
    args = parse_args()

    # ============ 设置输出目录 ============
    if args.do_train and os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError(f'Output directory ({args.output_dir}) already exists and is not empty.')
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # ============ 设置设备和 GPU 数量 ============
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.n_gpu = torch.cuda.device_count()
    logger.warning(f'Using {args.device} device, n_gpu: {args.n_gpu}')

    # ============ 设置随机种子 ============
    seed_everything(args.seed)

    # ============ 配置标签数量 ============
    # Token classification：每个位置都要分类
    # 标签数 = 2（start 和 end）
    args.num_labels = 2

    # ============ 加载预训练模型和分词器 ============
    logger.info(f'loading pretrained model and tokenizer of {args.model_type} ...')
    config = AutoConfig.from_pretrained(args.model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)

    # 加载自定义的 BertForExtractiveQA 模型
    model = BertForExtractiveQA.from_pretrained(
        args.model_checkpoint,
        config=config,
        args=args
    ).to(args.device)

    logger.info(f'Model: {model.__class__.__name__}')
    logger.info(f'Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M')

    # ============ 执行训练 ============
    if args.do_train:
        logger.info("***** Loading training and validation datasets *****")
        train_dataset = CMRC2018(args.train_file)
        dev_dataset = CMRC2018(args.dev_file)
        logger.info(f"Loaded {len(train_dataset)} training examples")
        logger.info(f"Loaded {len(dev_dataset)} validation examples")

        train(args, train_dataset, dev_dataset, model, tokenizer)

    # ============ 收集所有保存的权重文件 ============
    save_weights = [file for file in os.listdir(args.output_dir) if file.endswith('.bin')]
    logger.info(f'Found {len(save_weights)} saved checkpoints')

    # ============ 执行测试 ============
    if args.do_test:
        logger.info("***** Loading test dataset *****")
        test_dataset = CMRC2018(args.test_file)
        logger.info(f"Loaded {len(test_dataset)} test examples")

        test(args, test_dataset, model, tokenizer, save_weights)

    # ============ 执行预测 ============
    if args.do_predict:
        logger.info("***** Running prediction *****")
        test_dataset = CMRC2018(args.test_file)
        logger.info(f"Loaded {len(test_dataset)} test examples for prediction")

        for save_weight in save_weights:
            logger.info(f'loading weights from {save_weight}...')
            model.load_state_dict(torch.load(os.path.join(args.output_dir, save_weight), map_location=args.device))
            logger.info(f'predicting with {save_weight}...')

            results = []
            model.eval()

            # 对测试集的每个样本进行预测
            for s_idx in tqdm(range(len(test_dataset)), desc='Predicting'):
                sample = test_dataset[s_idx]

                # 生成答案
                pred_answer = predict(args, sample['context'], sample['question'], model, tokenizer)

                # 保存结果
                results.append({
                    "id": sample['id'],
                    "title": sample['title'],
                    "context": sample['context'],
                    "question": sample['question'],
                    "answers": sample['answers'],  # 真实答案（用于评估）
                    "prediction_text": pred_answer['prediction_text'],
                    "answer_start": pred_answer['answer_start']
                })

            # ============ 保存预测结果到 JSON 文件 ============
            output_file = os.path.join(args.output_dir, save_weight + '_predictions.json')
            with open(output_file, 'wt', encoding='utf-8') as f:
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
            logger.info(f'Predictions saved to {output_file}')

    logger.info("All done!")
