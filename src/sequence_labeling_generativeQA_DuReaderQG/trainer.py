"""
生成式问答（T5 + DuReaderQG）- 训练与评估循环
（被 run_generativeQA.py 调用；pipeline.py 是独立的最小示例，不依赖此模块）
"""
import torch
from tqdm.auto import tqdm
from utils import calculate_bleu, predict_answer


def train_epoch(model, train_loader, optimizer, scheduler, device, max_grad_norm=1.0):
    """
    训练一个epoch

    Args:
        model: T5模型
        train_loader: 训练数据加载器
        optimizer: 优化器
        scheduler: 学习率调度器
        device: 计算设备
        max_grad_norm: 梯度裁剪最大范数

    Returns:
        float: 平均训练损失
    """
    model.train()
    total_loss = 0

    progress_bar = tqdm(train_loader, desc='Training')
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})

    avg_loss = total_loss / len(train_loader)
    return avg_loss


def evaluate(model, dev_loader, device, tokenizer, max_length=64, num_beams=4):
    """
    在验证集上评估

    Args:
        model: T5模型
        dev_loader: 验证数据加载器
        device: 计算设备
        tokenizer: 分词器
        max_length: 生成答案最大长度
        num_beams: Beam Search宽度

    Returns:
        tuple: (平均损失, BLEU分数字典, 预测列表, 参考答案列表)
    """
    model.eval()
    total_loss = 0
    predictions = []
    references = []

    with torch.no_grad():
        for batch in tqdm(dev_loader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            total_loss += loss.item()

            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True
            )

            preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            refs = batch['answer']

            predictions.extend(preds)
            references.extend(refs)

    avg_loss = total_loss / len(dev_loader)
    bleu_scores = calculate_bleu(predictions, references)

    return avg_loss, bleu_scores, predictions, references


def train(model, train_loader, dev_loader, optimizer, scheduler, device,
          args, train_dataset=None):
    """
    完整的训练流程

    Args:
        model: T5模型
        train_loader: 训练数据加载器
        dev_loader: 验证数据加载器
        optimizer: 优化器
        scheduler: 学习率调度器
        device: 计算设备
        args: 参数对象
        train_dataset: 训练数据集（可选）

    Returns:
        dict: 训练历史记录
    """
    training_history = {
        'train_loss': [],
        'dev_loss': [],
        'dev_bleu1': [],
        'dev_bleu2': [],
        'dev_bleu3': [],
        'dev_bleu4': []
    }

    best_bleu = 0
    best_model_path = f"{args.output_dir}/best_model.pt"

    print("\n开始训练...\n")
    for epoch in range(args.num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{args.num_epochs}")
        print(f"{'='*60}")

        # 训练
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device,
            max_grad_norm=args.max_grad_norm
        )
        print(f"\n训练损失: {train_loss:.4f}")
        training_history['train_loss'].append(train_loss)

        # 验证
        dev_loss, bleu_scores, predictions, references = evaluate(
            model, dev_loader, device, args.tokenizer,
            max_length=args.max_target_length,
            num_beams=args.num_beams
        )
        print(f"\n验证损失: {dev_loss:.4f}")
        print("验证 BLEU 分数:")
        for k, v in bleu_scores.items():
            print(f"  {k}: {v:.4f}")

        # 记录历史
        training_history['dev_loss'].append(dev_loss)
        training_history['dev_bleu1'].append(bleu_scores['BLEU-1'])
        training_history['dev_bleu2'].append(bleu_scores['BLEU-2'])
        training_history['dev_bleu3'].append(bleu_scores['BLEU-3'])
        training_history['dev_bleu4'].append(bleu_scores['BLEU-4'])

        # 保存最佳模型
        if bleu_scores['BLEU-2'] > best_bleu:
            best_bleu = bleu_scores['BLEU-2']
            torch.save(model.state_dict(), best_model_path)
            print(f"\n✓ 保存最佳模型 (BLEU-2: {best_bleu:.4f})")

    print("\n" + "="*60)
    print("✅ 训练完成！")
    print("="*60)

    return training_history