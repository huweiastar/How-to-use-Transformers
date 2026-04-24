"""
生成式问答推理脚本

从已训练好的模型加载权重，对任意（问题, 文章）输入生成答案。
不依赖训练数据，可单独运行。

用法：
    python predict.py                           # 运行内置示例
    python predict.py --model_dir ./qa_model_output/final_model
"""

import argparse
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer


# ============================================================
# 核心推理函数
# ============================================================

def load_model(model_dir: str, device: torch.device):
    """从 save_pretrained 目录加载模型和分词器。"""
    print(f"加载模型：{model_dir}")
    tokenizer = T5Tokenizer.from_pretrained(model_dir)
    # tie_word_embeddings=False：微调后两个权重矩阵已各自独立，
    # 显式声明避免 from_pretrained 检测到 config 与 checkpoint 不一致时打印警告。
    model = T5ForConditionalGeneration.from_pretrained(
        model_dir, tie_word_embeddings=False
    ).to(device)
    model.eval()
    print(f"✓ 加载成功")
    return model, tokenizer


def predict_answer(
    question:    str,
    context:     str,
    model,
    tokenizer,
    device:      torch.device,
    max_gen_len: int = 64,
    num_beams:   int = 4,
) -> str:
    """
    对单条（问题, 文章）生成答案。

    Args:
        question:    问题文本
        context:     参考文章
        model:       已加载的 T5ForConditionalGeneration
        tokenizer:   对应的 T5Tokenizer
        device:      运行设备
        max_gen_len: 生成答案的最大 token 数
        num_beams:   Beam Search 的候选数，越大越准但越慢；1 = 贪心解码

    Returns:
        生成的答案字符串
    """
    # T5 输入格式与训练时保持一致
    input_text = f"question: {question} context: {context}"
    inputs = tokenizer(
        input_text,
        max_length=512,
        truncation=True,
        return_tensors='pt',
    ).to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=max_gen_len,
            num_beams=num_beams,
            early_stopping=True,
        )

    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)


def predict_batch(
    samples:     list[dict],
    model,
    tokenizer,
    device:      torch.device,
    max_gen_len: int = 64,
    num_beams:   int = 4,
) -> list[str]:
    """
    批量预测，每条 sample 需包含 'question' 和 'context' 键。

    比循环调用 predict_answer 更高效：一次 forward 处理多条。
    """
    input_texts = [
        f"question: {s['question']} context: {s['context']}"
        for s in samples
    ]
    inputs = tokenizer(
        input_texts,
        max_length=512,
        padding=True,
        truncation=True,
        return_tensors='pt',
    ).to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=max_gen_len,
            num_beams=num_beams,
            early_stopping=True,
        )

    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)


# ============================================================
# 命令行入口
# ============================================================

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def main():
    parser = argparse.ArgumentParser(description='生成式问答推理')
    parser.add_argument(
        '--model_dir', type=str,
        default='./qa_model_output/final_model',
        help='save_pretrained 保存的模型目录',
    )
    parser.add_argument('--max_gen_len', type=int, default=64,  help='最大生成长度')
    parser.add_argument('--num_beams',   type=int, default=4,   help='Beam Search 候选数')
    args = parser.parse_args()

    device = get_device()
    print(f"✓ 设备：{device}\n")

    model, tokenizer = load_model(args.model_dir, device)

    # ── 内置演示样例 ─────────────────────────────────────────
    examples = [
        {
            'question': '什么时候清零?',
            'context':  '淘宝网每年12月31日24:00点会对符合条件的扣分做清零处理。',
            'answer':   '12月31日24:00点',
        },
        {
            'question': '这个产品的主要特点是什么?',
            'context':  'iPhone 15 采用了新一代 A17 Pro 芯片，支持 USB-C 接口，配备了更先进的相机系统，续航能力提升至 26 小时。',
            'answer':   'A17 Pro 芯片、USB-C 接口、更先进的相机系统',
        },
        {
            'question': '中国的首都是哪里?',
            'context':  '北京是中国的首都，是全国的政治、文化、国际交往中心，也是中国历史文化名城。',
            'answer':   '北京',
        },
    ]

    # 单条预测演示
    print("=" * 70)
    print("单条预测示例")
    print("=" * 70)
    for i, ex in enumerate(examples, 1):
        pred = predict_answer(
            ex['question'], ex['context'],
            model, tokenizer, device,
            max_gen_len=args.max_gen_len,
            num_beams=args.num_beams,
        )
        print(f"\n[{i}] 问题：{ex['question']}")
        print(f"    文章：{ex['context'][:80]}...")
        print(f"    参考：{ex['answer']}")
        print(f"    预测：{pred}")

    # 批量预测演示
    print("\n" + "=" * 70)
    print("批量预测示例（同样的三条一次性送入）")
    print("=" * 70)
    preds = predict_batch(
        examples, model, tokenizer, device,
        max_gen_len=args.max_gen_len,
        num_beams=args.num_beams,
    )
    for ex, pred in zip(examples, preds):
        print(f"  问题：{ex['question']}  →  预测：{pred}")


if __name__ == '__main__':
    main()