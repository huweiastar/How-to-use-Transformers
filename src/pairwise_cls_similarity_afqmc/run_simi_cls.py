"""
run_simi_cls.py —— 工程化的训练/测试/预测入口脚本
================================================================

本文件等价于 pipeline.py 的"生产版"。两者的关系是：
    pipeline.py  = 单文件、硬编码超参数的教学示例
    run_simi_cls.py = 拆成 arg.py / data.py / modeling.py 的工程版

工程版多做的几件事（这是真实项目里该有的样子）：
  1. 所有超参数通过 arg.py 命令行传入（方便跑多组实验）
  2. 支持三种互不耦合的运行模式：--do_train / --do_test / --do_predict
  3. 有日志、有输出目录、保存"最佳权重"而不是"最后一轮"
  4. 优化器对 bias / LayerNorm.weight 单独处理（不加 weight decay，
     这是 BERT 微调的惯例，能轻微提升效果）
  5. 学习率带 warmup 预热阶段（先线性升到峰值，再线性降到 0）

【阅读建议】先读懂 pipeline.py 再读这个文件——你会发现结构完全对应，
只是把 train / test / predict 各自抽成了函数，并把超参数外置而已。
================================================================
"""
import os
import logging
import json
import torch
# from transformers import AdamW, get_scheduler
from transformers import get_scheduler
from torch.optim import AdamW  # PyTorch 原生AdamW，完全等价
from transformers import AutoConfig, AutoTokenizer
import sys

# 添加父目录到路径，以便导入其他模块
sys.path.append('../../')
from src.tools import seed_everything
from src.pairwise_cls_similarity_afqmc.arg import parse_args
from src.pairwise_cls_similarity_afqmc.data import AFQMC, get_dataLoader
from src.pairwise_cls_similarity_afqmc.modeling import BertForPairwiseCLS, RobertaForPairwiseCLS
from tqdm.auto import tqdm

# ============ 日志配置 ============
# 控制台日志：所有阶段都打到 stdout
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
LOG_DATEFMT = '%Y/%m/%d %H:%M:%S'
logging.basicConfig(format=LOG_FORMAT,
                    datefmt=LOG_DATEFMT,
                    level=logging.INFO)
logger = logging.getLogger("Model")


def add_file_logger(output_dir, log_name):
    """
    给 logger 再加一个 FileHandler，让当前阶段的日志同时落盘到
    output_dir/{log_name}.log。
    每次调用都会新建一个独立的文件句柄，因此可以在同一次运行里
    分别为 train / test / predict 三个阶段写三份日志。

    参数:
        output_dir (str): 日志文件输出目录
        log_name (str): 日志文件名（不含 .log 后缀），例如 'train'
    返回值:
        logging.FileHandler: 新增的文件处理器，调用方在阶段结束后可用
                             logger.removeHandler(fh) 卸载，避免下一阶段串日志。
    """
    log_path = os.path.join(output_dir, f'{log_name}.log')
    fh = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(fmt=LOG_FORMAT, datefmt=LOG_DATEFMT))
    logger.addHandler(fh)
    logger.info(f'file logging enabled: {log_path}')
    return fh

# ============ 模型类映射 ============
# 将模型类型字符串映射到实际的模型类
MODEL_CLASSES = {
    'bert': BertForPairwiseCLS,
    # 'roberta': RobertaForPairwiseCLS,
    'roberta': BertForPairwiseCLS  # 哈工大权重必须使用Bert模型加载
}

def to_device(args, batch_data):
    """
    将batch数据移动到指定的设备（CPU或GPU）

    参数:
        args: 包含device信息的参数对象
        batch_data (dict): 包含batch_inputs和labels的字典

    返回值:
        dict: 所有张量都被移动到指定设备的字典
    """
    new_batch_data = {}

    for k, v in batch_data.items():
        if k == 'batch_inputs':
            # batch_inputs是一个字典，包含input_ids, attention_mask, token_type_ids等
            # 需要将这个字典中的所有张量都移动到设备上
            new_batch_data[k] = {
                k_: v_.to(args.device) for k_, v_ in v.items()
            }
        else:
            # labels是一个列表，需要转换为张量并移动到设备上
            new_batch_data[k] = torch.tensor(v).to(args.device)

    return new_batch_data

def train_loop(args, dataloader, model, optimizer, lr_scheduler, epoch, total_loss):
    """
    执行一个epoch的训练

    参数:
        args: 包含device等参数的对象
        dataloader: 训练数据加载器
        model: 要训练的模型
        optimizer: 优化器（AdamW）
        lr_scheduler: 学习率调度器
        epoch (int): 当前epoch数（用于计算总步数）
        total_loss (float): 累积损失值

    返回值:
        float: 更新后的累积损失值
    """
    # 创建进度条，用于可视化训练进度
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f'loss: {0:>7f}')

    # 计算当前epoch之前已完成的总步数
    finish_step_num = epoch * len(dataloader)

    # 设置模型为训练模式（启用dropout等）
    model.train()

    # 遍历数据加载器中的每个batch
    for step, batch_data in enumerate(dataloader, start=1):
        # 将batch数据移动到指定设备
        batch_data = to_device(args, batch_data)

        # 前向传播，获取损失和logits
        outputs = model(**batch_data)
        loss = outputs[0]

        # ============ 反向传播和优化 ============
        # 清除之前的梯度
        optimizer.zero_grad()

        # 反向传播计算梯度
        loss.backward()

        # 使用优化器更新模型参数
        optimizer.step()

        # 更新学习率
        lr_scheduler.step()

        # ============ 损失统计 ============
        # 累积损失值
        total_loss += loss.item()

        # 更新进度条显示的信息（平均损失）
        progress_bar.set_description(f'loss: {total_loss/(finish_step_num + step):>7f}')
        progress_bar.update(1)

    return total_loss

def test_loop(args, dataloader, model, mode='Test'):
    """
    执行验证或测试，计算模型的准确率

    参数:
        args: 包含device等参数的对象
        dataloader: 验证/测试数据加载器
        model: 要评估的模型
        mode (str): 评估模式，'Valid'或'Test'（仅用于日志）

    返回值:
        float: 准确率（0-1之间）
    """
    # 验证mode参数
    assert mode in ['Valid', 'Test'], f"mode必须是'Valid'或'Test'，但得到'{mode}'"

    # 初始化正确预测的样本数
    correct = 0

    # 设置模型为评估模式（关闭dropout等）
    model.eval()

    # 在不计算梯度的情况下进行评估（节省内存）
    with torch.no_grad():
        # 遍历数据加载器中的每个batch
        for batch_data in tqdm(dataloader):
            # 将batch数据移动到指定设备
            batch_data = to_device(args, batch_data)

            # 前向传播
            outputs = model(**batch_data)
            # outputs是(loss, logits)的元组，我们需要logits
            logits = outputs[1]

            # ============ 获取预测结果 ============
            # argmax获取最高logit对应的类别索引
            predictions = logits.argmax(dim=-1).cpu().numpy().tolist()

            # 获取真实标签
            labels = batch_data['labels'].cpu().numpy()

            # 计算正确预测的数量
            correct += (predictions == labels).sum()

    # 计算准确率 = 正确预测数 / 总样本数
    correct /= len(dataloader.dataset)

    return correct

def train(args, train_dataset, dev_dataset, model, tokenizer):
    """
    完整的训练流程
    包括：创建数据加载器、配置优化器、执行多个epoch的训练和验证

    参数:
        args: 包含所有超参数和配置信息的对象
        train_dataset: 训练数据集对象
        dev_dataset: 验证数据集对象
        model: 要训练的模型
        tokenizer: 分词器
    """
    # ============ 创建数据加载器 ============
    # 训练集需要shuffle，以增加随机性
    train_dataloader = get_dataLoader(args, train_dataset, tokenizer, shuffle=True)
    # 验证集不需要shuffle
    dev_dataloader = get_dataLoader(args, dev_dataset, tokenizer, shuffle=False)

    # ============ 计算总的优化步数 ============
    # 总步数 = 每个epoch的步数 × epoch数
    t_total = len(train_dataloader) * args.num_train_epochs

    # ============ 配置优化器 ============
    # 某些参数（如bias和LayerNorm.weight）不应用权重衰减
    no_decay = ["bias", "LayerNorm.weight"]

    # 将模型参数分为两组：有权重衰减和无权重衰减
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

    # 计算预热步数
    args.warmup_steps = int(t_total * args.warmup_proportion)

    # 创建AdamW优化器
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),  # (β1, β2)参数
        eps=args.adam_epsilon  # 数值稳定性参数
    )

    # ============ 配置学习率调度器 ============
    # 使用线性预热和线性衰减的学习率调度
    lr_scheduler = get_scheduler(
        'linear',
        optimizer,
        num_warmup_steps=args.warmup_steps,  # 预热阶段的步数
        num_training_steps=t_total            # 总训练步数
    )

    # ============ 记录训练信息 ============
    logger.info("***** Running training *****")
    logger.info(f"Num examples - {len(train_dataset)}")
    logger.info(f"Num Epochs - {args.num_train_epochs}")
    logger.info(f"Total optimization steps - {t_total}")

    # 保存训练参数到文件
    with open(os.path.join(args.output_dir, 'args.txt'), 'wt', encoding='utf-8') as f:
        f.write(str(args))

    # ============ 执行多epoch训练 ============
    total_loss = 0.0  # 累积损失值
    best_acc = 0.0    # 最佳验证准确率

    for epoch in range(args.num_train_epochs):
        print(f"Epoch {epoch+1}/{args.num_train_epochs}\n-------------------------------")

        # 执行一个epoch的训练
        total_loss = train_loop(args, train_dataloader, model, optimizer, lr_scheduler, epoch, total_loss)

        # 在验证集上评估模型
        valid_acc = test_loop(args, dev_dataloader, model)
        logger.info(f"Dev Accuracy: {(100*valid_acc):>0.2f}%")

        # ============ 模型检查点保存 ============
        # 如果验证准确率改进，保存模型权重
        if valid_acc > best_acc:
            best_acc = valid_acc
            logger.info(f'saving new weights to {args.output_dir}...\n')

            # 生成检查点文件名（包含epoch和准确率信息）
            save_weight = f'epoch_{epoch+1}_dev_acc_{(100*valid_acc):0.1f}_weights.bin'

            # 保存模型参数
            torch.save(model.state_dict(), os.path.join(args.output_dir, save_weight))

    logger.info("Done!")

def test(args, test_dataset, model, tokenizer, save_weights: list):
    """
    在测试集上评估所有保存的模型检查点

    参数:
        args: 包含device等参数的对象
        test_dataset: 测试数据集对象
        model: 模型对象（将被加载不同的权重）
        tokenizer: 分词器
        save_weights (list): 要评估的模型权重文件列表
    """
    # 创建测试数据加载器
    test_dataloader = get_dataLoader(args, test_dataset, tokenizer, shuffle=False)

    logger.info('***** Running testing *****')

    # 遍历每个保存的权重文件
    for save_weight in save_weights:
        logger.info(f'loading weights from {save_weight}...')

        # 加载模型权重
        model.load_state_dict(torch.load(os.path.join(args.output_dir, save_weight)))

        # 在测试集上评估模型
        test_acc = test_loop(args, test_dataloader, model)
        logger.info(f"Test Accuracy: {(100*test_acc):>0.2f}%")

def predict(args, sent_1, sent_2, model, tokenizer):
    """
    对单个句对进行相似度分类预测

    参数:
        args: 包含device等参数的对象
        sent_1 (str): 第一个句子
        sent_2 (str): 第二个句子
        model: 分类模型
        tokenizer: 分词器

    返回值:
        tuple: (pred, prob)
            - pred (int): 预测标签（0或1，1表示相似）
            - prob (float): 预测标签的概率
    """
    # ============ 文本编码 ============
    # 使用tokenizer将两个句子编码为模型能处理的格式
    inputs = tokenizer(
        sent_1,
        sent_2,
        max_length=args.max_seq_length,
        truncation=True,           # 自动截断超过max_length的序列
        return_tensors="pt"        # 返回PyTorch张量
    )

    # 将inputs包装成expected格式
    inputs = {
        'batch_inputs': inputs
    }

    # 将数据移动到指定设备（CPU或GPU）
    inputs = to_device(args, inputs)

    # ============ 模型推理 ============
    # 在不计算梯度的情况下进行推理（节省内存）
    with torch.no_grad():
        # 前向传播
        outputs = model(**inputs)
        # outputs是(loss, logits)的元组，我们需要logits
        logits = outputs[1]

    # ============ 获取预测结果 ============
    # argmax获取最高logit对应的类别索引
    pred = int(logits.argmax(dim=-1)[0].cpu().numpy())

    # 计算softmax概率分布
    prob = torch.nn.functional.softmax(logits, dim=-1)[0].cpu().numpy().tolist()

    # 返回预测标签和该标签的概率
    return pred, prob[pred]

if __name__ == '__main__':
    """
    主程序入口
    支持三种操作：训练、测试、预测
    可以通过命令行参数--do_train, --do_test, --do_predict控制
    """
    # ============ 解析命令行参数 ============
    args = parse_args()

    # ============ 设置输出目录 ============
    # 如果是训练模式，检查输出目录是否已存在且非空
    if args.do_train and os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError(f'Output directory ({args.output_dir}) already exists and is not empty.')

    # 如果输出目录不存在，创建它
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    # ============ 设置设备（CPU或GPU） ============
    # args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.backends.mps.is_available():
        args.device = torch.device("mps")
    elif torch.cuda.is_available():
        args.device = torch.device("cuda")
    else:
        args.device = torch.device("cpu")
    args.n_gpu = torch.cuda.device_count()
    logger.warning(f'Using {args.device} device, n_gpu: {args.n_gpu}')

    # ============ 设置随机种子（用于结果复现） ============
    seed_everything(args.seed)

    # ============ 加载预训练模型和分词器 ============
    logger.info(f'loading pretrained model and tokenizer of {args.model_type} ...')

    # 从预训练模型加载配置
    config = AutoConfig.from_pretrained(args.model_checkpoint)

    # 从预训练模型加载分词器
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)

    # 设置标签数量（二分类问题：0表示不相似，1表示相似）
    args.num_labels = 2

    # 根据model_type选择相应的模型类，加载预训练权重
    model = MODEL_CLASSES[args.model_type].from_pretrained(
        args.model_checkpoint,
        config=config,
        args=args
    ).to(args.device)

    # ============ 执行训练 ============
    if args.do_train:
        # 给训练阶段单独挂一个 FileHandler → output_dir/train.log
        train_fh = add_file_logger(args.output_dir, 'train')
        try:
            logger.info("***** Loading training and validation datasets *****")

            # 加载训练数据集
            train_dataset = AFQMC(args.train_file)

            # 加载验证数据集
            dev_dataset = AFQMC(args.dev_file)

            # 执行训练（训练中每个 epoch 的 loss / valid acc 都会写进 train.log）
            train(args, train_dataset, dev_dataset, model, tokenizer)
        finally:
            # 训练阶段结束后卸掉 FileHandler，避免后续 test/predict 的日志串进 train.log
            logger.removeHandler(train_fh)
            train_fh.close()

    # ============ 获取所有保存的模型权重文件 ============
    save_weights = [file for file in os.listdir(args.output_dir) if file.endswith('.bin')]

    # ============ 执行测试 ============
    if args.do_test:
        # 测试阶段日志 → output_dir/test.log
        test_fh = add_file_logger(args.output_dir, 'test')
        try:
            logger.info("***** Loading test dataset *****")

            # 加载测试数据集
            test_dataset = AFQMC(args.test_file)

            # 在测试集上评估模型（每个权重的 Test Accuracy 都会写进 test.log）
            test(args, test_dataset, model, tokenizer, save_weights)
        finally:
            logger.removeHandler(test_fh)
            test_fh.close()

    # ============ 执行预测 ============
    if args.do_predict:
        # 预测阶段日志 → output_dir/predict.log
        predict_fh = add_file_logger(args.output_dir, 'predict')
        try:
            logger.info("***** Running prediction *****")

            # 加载测试数据集（用于预测）
            test_dataset = AFQMC(args.test_file)

            # 对每个保存的模型权重进行预测
            for save_weight in save_weights:
                logger.info(f'loading weights from {save_weight}...')

                # 加载模型权重
                model.load_state_dict(torch.load(os.path.join(args.output_dir, save_weight)))
                logger.info(f'predicting labels of {save_weight}...')

                # 存储预测结果
                results = []

                # 设置模型为评估模式
                model.eval()

                # 对每个样本进行预测
                for s_idx in tqdm(range(len(test_dataset))):
                    # 获取单个样本
                    sample = test_dataset[s_idx]

                    # 进行预测
                    pred, prob = predict(args, sample['sentence1'], sample['sentence2'], model, tokenizer)

                    # 保存预测结果
                    results.append({
                        "sentence1": sample['sentence1'],      # 第一个句子
                        "sentence2": sample['sentence2'],      # 第二个句子
                        "label": sample['label'],              # 真实标签
                        "pred_label": str(pred),               # 预测标签
                        "pred_prob": prob                      # 预测概率
                    })

                # ============ 保存预测结果到文件 ============
                # 生成输出文件名
                output_file = os.path.join(args.output_dir, save_weight + '_test_pred_simi.json')

                # 以JSON Lines格式保存结果（每行一个JSON对象）
                with open(output_file, 'wt', encoding='utf-8') as f:
                    for sample_result in results:
                        # 将字典转换为JSON字符串并写入文件
                        # ensure_ascii=False 保证中文字符正常显示
                        f.write(json.dumps(sample_result, ensure_ascii=False) + '\n')

                logger.info(f'Predictions saved to {output_file}')
        finally:
            logger.removeHandler(predict_fh)
            predict_fh.close()
