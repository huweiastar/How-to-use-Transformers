"""
生成式问答模型（T5-Base + DuReaderQG）- 主训练脚本
"""
import os
import json
import matplotlib.pyplot as plt
from pathlib import Path

import torch
from arg import parse_args, setup_args_and_paths
from data import create_dataloaders
from modeling import get_device, load_tokenizer, load_model, setup_optimizer_and_scheduler
from trainer import train, evaluate
from utils import setup_seed, predict_answer, print_predictions, save_predictions

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def main():
    """主函数"""
    # ============ 1. 解析参数 ============
    args = parse_args()
    args = setup_args_and_paths(args)

    print("="*60)
    print("生成式问答模型（T5 + DuReaderQG）")
    print("="*60)
    print("\n模型配置:")
    print(f"  模型选择: {args.model_choice} (默认: Mengzi-T5-Base 中文模型)")
    print(f"  模型路径: {args.model_name}")
    print(f"  输出目录: {args.output_dir}")

    # ============ 2. 设置随机种子 ============
    setup_seed(args.seed)

    # ============ 3. 获取设备 ============
    device = get_device(args.device)
    args.device_obj = device

    # ============ 4. 加载分词器和数据 ============
    print("\n" + "="*60)
    print("加载数据和分词器")
    print("="*60 + "\n")

    tokenizer = load_tokenizer(args.tokenizer_name)
    args.tokenizer = tokenizer

    # 验证数据文件存在
    if not Path(args.train_file).exists():
        print(f"✗ 训练数据不存在: {args.train_file}")
        print(f"  请确保数据文件路径正确或运行此脚本时位置正确")
        return

    if not Path(args.dev_file).exists():
        print(f"✗ 验证数据不存在: {args.dev_file}")
        return

    train_loader, dev_loader, train_dataset, dev_dataset = create_dataloaders(
        args.train_file, args.dev_file, tokenizer,
        batch_size=args.batch_size,
        max_input_length=args.max_input_length,
        max_target_length=args.max_target_length,
        num_workers=args.num_workers
    )

    # ============ 5. 加载模型 ============
    print("\n" + "="*60)
    print("加载模型")
    print("="*60 + "\n")

    model = load_model(args.model_name, device)

    # ============ 6. 训练 ============
    if args.do_train:
        print("\n" + "="*60)
        print("设置优化器和调度器")
        print("="*60 + "\n")

        total_steps = len(train_loader) * args.num_epochs
        optimizer, scheduler = setup_optimizer_and_scheduler(model, args, total_steps)

        print("\n" + "="*60)
        print("开始训练")
        print("="*60)

        training_history = train(
            model, train_loader, dev_loader, optimizer, scheduler, device,
            args, train_dataset=train_dataset
        )

        # ============ 7. 保存模型和结果 ============
        print("\n" + "="*60)
        print("保存模型和结果")
        print("="*60 + "\n")

        # 加载最佳模型
        best_model_path = f"{args.output_dir}/best_model.pt"
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        model.eval()
        print(f"✓ 加载最佳模型: {best_model_path}")

        # 保存完整模型
        model_save_path = f"{args.output_dir}/final_model"
        model.save_pretrained(model_save_path)
        tokenizer.save_pretrained(model_save_path)
        print(f"✓ 完整模型已保存到: {model_save_path}")

        # 保存超参数
        hyperparams = {k: v for k, v in vars(args).items()
                      if not k.startswith('_') and k not in ['device_obj', 'tokenizer']}
        with open(f"{args.output_dir}/hyperparams.json", 'w') as f:
            json.dump(hyperparams, f, indent=2, ensure_ascii=False, default=str)
        print(f"✓ 超参数已保存到: {args.output_dir}/hyperparams.json")

        # 保存训练历史
        with open(f"{args.output_dir}/training_history.json", 'w') as f:
            json.dump(training_history, f, indent=2)
        print(f"✓ 训练历史已保存到: {args.output_dir}/training_history.json")

        # 绘制训练曲线
        print("\n绘制训练曲线...")
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('模型训练收敛曲线', fontsize=16, fontweight='bold')

        # 1. 损失曲线
        ax1 = axes[0, 0]
        ax1.plot(training_history['train_loss'], label='训练损失', marker='o', linewidth=2, markersize=8)
        ax1.plot(training_history['dev_loss'], label='验证损失', marker='s', linewidth=2, markersize=8)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('损失', fontsize=12)
        ax1.set_title('损失曲线', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(range(args.num_epochs))

        # 2. BLEU-1/2 曲线
        ax2 = axes[0, 1]
        ax2.plot(training_history['dev_bleu1'], label='BLEU-1', marker='o', linewidth=2, markersize=8)
        ax2.plot(training_history['dev_bleu2'], label='BLEU-2', marker='s', linewidth=2, markersize=8)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('BLEU 分数', fontsize=12)
        ax2.set_title('BLEU-1/2 分数', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(range(args.num_epochs))

        # 3. BLEU-3/4 曲线
        ax3 = axes[1, 0]
        ax3.plot(training_history['dev_bleu3'], label='BLEU-3', marker='o', linewidth=2, markersize=8)
        ax3.plot(training_history['dev_bleu4'], label='BLEU-4', marker='s', linewidth=2, markersize=8)
        ax3.set_xlabel('Epoch', fontsize=12)
        ax3.set_ylabel('BLEU 分数', fontsize=12)
        ax3.set_title('BLEU-3/4 分数', fontsize=13, fontweight='bold')
        ax3.legend(fontsize=11)
        ax3.grid(True, alpha=0.3)
        ax3.set_xticks(range(args.num_epochs))

        # 4. 所有 BLEU 分数
        ax4 = axes[1, 1]
        ax4.plot(training_history['dev_bleu1'], label='BLEU-1', marker='o', linewidth=2, markersize=8)
        ax4.plot(training_history['dev_bleu2'], label='BLEU-2', marker='s', linewidth=2, markersize=8)
        ax4.plot(training_history['dev_bleu3'], label='BLEU-3', marker='^', linewidth=2, markersize=8)
        ax4.plot(training_history['dev_bleu4'], label='BLEU-4', marker='d', linewidth=2, markersize=8)
        ax4.set_xlabel('Epoch', fontsize=12)
        ax4.set_ylabel('BLEU 分数', fontsize=12)
        ax4.set_title('所有 BLEU 指标对比', fontsize=13, fontweight='bold')
        ax4.legend(fontsize=11)
        ax4.grid(True, alpha=0.3)
        ax4.set_xticks(range(args.num_epochs))

        plt.tight_layout()
        curves_path = f"{args.output_dir}/training_curves.png"
        plt.savefig(curves_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✓ 训练曲线已保存到: {curves_path}")

    # ============ 8. 评估 ============
    if args.do_eval:
        print("\n" + "="*60)
        print("评估模型")
        print("="*60 + "\n")

        model.eval()
        dev_loss, bleu_scores, predictions, references = evaluate(
            model, dev_loader, device, tokenizer,
            max_length=args.max_target_length,
            num_beams=args.num_beams
        )

        print(f"\n验证损失: {dev_loss:.4f}")
        print("验证 BLEU 分数:")
        for k, v in bleu_scores.items():
            print(f"  {k}: {v:.4f}")

        # 保存预测结果
        save_predictions(predictions, references, f"{args.output_dir}/predictions.json")

    # ============ 9. 预测 ============
    if args.do_predict:
        print("\n" + "="*60)
        print("预测示例")
        print("="*60)

        model.eval()

        # 验证集预测示例
        print("\n验证集上的预测示例 (前 5 个样本):\n")
        val_predictions = []
        val_references = []
        for i in range(min(5, len(dev_dataset))):
            sample = dev_dataset[i]
            question = sample['question']
            context = sample['context']
            ground_truth = sample['answer']

            pred = predict_answer(question, context, model, tokenizer, device,
                                max_length=args.max_target_length,
                                num_beams=args.num_beams)
            val_predictions.append(pred)
            val_references.append(ground_truth)

        print_predictions(val_predictions, val_references, num_samples=5)

        # 自定义输入预测
        print("\n" + "="*80)
        print("自定义输入预测示例")
        print("="*80)

        custom_examples = [
            {
                "question": "什么时候清零?",
                "context": "淘宝网每年12月31日24:00点会对符合条件的扣分做清零处理。"
            },
            {
                "question": "这个产品的主要特点是什么?",
                "context": "iPhone 15 采用了新一代 A17 Pro 芯片，支持 USB-C 接口，配备了更先进的相机系统，续航能力提升至 26 小时。"
            },
            {
                "question": "中国的首都是哪里?",
                "context": "北京是中华人民共和国的首都，是全国的政治、文化、国际交往中心，也是中国历史文化名城。"
            }
        ]

        for i, example in enumerate(custom_examples, 1):
            question = example["question"]
            context = example["context"]
            print(f"\n示例 {i}:")
            print(f"❓ 问题: {question}")
            print(f"📄 文章: {context}")
            pred = predict_answer(question, context, model, tokenizer, device,
                                max_length=args.max_target_length,
                                num_beams=args.num_beams)
            print(f"🤖 答案: {pred}")

        print("\n" + "="*80)

    # ============ 10. 最终总结 ============
    print("\n" + "="*80)
    print("✅ 项目总结")
    print("="*80)
    print(f"""
【模型信息】
  架构: Google T5 ({args.model_choice})
  任务: 生成式问答 (Abstractive QA)
  数据集: DuReaderQG (中文阅读理解)
  设备: {device}

【训练配置】
  批大小: {args.batch_size}
  学习率: {args.learning_rate}
  总 Epoch: {args.num_epochs}
  优化器: AdamW
  预热比例: {args.warmup_ratio}

【输出文件】
  最佳模型: {args.output_dir}/best_model.pt
  完整模型: {args.output_dir}/final_model
  训练曲线: {args.output_dir}/training_curves.png
  超参数: {args.output_dir}/hyperparams.json
  训练历史: {args.output_dir}/training_history.json

【下一步】
  1. 查看 training_curves.png 分析训练情况
  2. 使用模型进行预测：
     python -c "
     from transformers import T5ForConditionalGeneration, T5Tokenizer
     model = T5ForConditionalGeneration.from_pretrained('{args.output_dir}/final_model')
     tokenizer = T5Tokenizer.from_pretrained('{args.output_dir}/final_model')
     "
""")
    print("="*80)
    print("\n🎉 训练完成！祝使用愉快！")


if __name__ == '__main__':
    main()