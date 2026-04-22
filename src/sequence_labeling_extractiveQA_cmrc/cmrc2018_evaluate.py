"""
===============================================================================
评估模块：CMRC 2018 的 F1 和 EM 指标计算
===============================================================================

【阅读理解的评估指标】
  - EM (Exact Match): 预测与真实答案完全相同（忽略标点）= 1，否则 = 0
  - F1 (F1 Score): 基于词级别（token 级别）的部分匹配
    * 计算预测答案和真实答案的最长公共子序列（LCS）
    * F1 = 2 * (precision * recall) / (precision + recall)
    * 对有多个参考答案的问题，取最高分

【与英文 SQuAD 的区别】
  - 英文：直接分割 by space
  - 中文：混合分词，中文字符逐字处理，英文词作为单位

【重要概念】
  - mixed_segmentation: 中英混合分割（中文按字，英文按词）
  - remove_punctuation: 忽略标点符号，使评估更公平
  - find_lcs: 最长公共子序列，用于计算精准度和召回率
"""
import re
import sys
from transformers import AutoTokenizer

# ============ 分词器设置 ============
# 使用预训练的 BERT tokenizer 来分割英文单词
# 注意：这里使用的是 "bert-base-cased"，你可以根据需要改为中文 BERT
model_checkpoint = "bert-base-cased"
tokenizer = None

def _get_tokenizer():
    """延迟加载 tokenizer，避免 import 时下载模型"""
    global tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    return tokenizer

# tokenize 函数：使用 BERT tokenizer 分词，去掉 [CLS] 和 [SEP]
# tokenizer(x).tokens()[1:-1] 跳过了第一个 [CLS] 和最后一个 [SEP]
def tokenize(x):
    """分词函数，使用 BERT tokenizer"""
    tok = _get_tokenizer()
    return tok(x).tokens()[1:-1]

# 可选：使用 NLTK 的 word_tokenize（评论掉的部分）
# import nltk
# tokenize = lambda x: nltk.word_tokenize(x)


def mixed_segmentation(in_str, rm_punc=False):
    """
    中英混合分词：中文字符逐字处理，英文单词作为一个单位

    【算法思路】
    1. 逐字遍历输入字符串
    2. 遇到中文字符或标点：直接作为一个单元输出
    3. 遇到英文字符：积累成一个单词，然后用英文 tokenizer 分割
    4. 可选：rm_punc=True 时删除标点符号

    这样的混合分词适合中文文本中混有英文的情况（如科技文章）。

    参数:
        in_str (str): 输入字符串
        rm_punc (bool): 是否删除标点符号

    返回值:
        list: 分割后的 token 列表
    """
    in_str = str(in_str).lower().strip()
    segs_out = []
    temp_str = ""  # 临时字符串，用于积累英文字符

    # 中英文标点符号表
    sp_char = ['-', ':', '_', '*', '^', '/', '\\', '~', '`', '+', '=',
               '，', '。', '：', '？', '！', '"', '"', '；', ''', '《', '》', '……', '·', '、',
               '「', '」', '（', '）', '－', '～', '『', '』']

    for char in in_str:
        # 如果需要删除标点，且当前字符是标点，跳过
        if rm_punc and char in sp_char:
            continue

        # 检查是否是中文字符（Unicode 范围 \u4e00-\u9fa5）或标点
        if re.search(r'[\u4e00-\u9fa5]', char) or char in sp_char:
            # 先处理积累的英文字符
            if temp_str != "":
                # 用 tokenizer 分割英文单词
                ss = tokenize(temp_str)
                segs_out.extend(ss)
                temp_str = ""
            # 中文字符 / 标点直接作为一个单元
            segs_out.append(char)
        else:
            # 英文字符，积累起来
            temp_str += char

    # ============ 处理末尾的英文字符 ============
    if temp_str != "":
        ss = tokenize(temp_str)
        segs_out.extend(ss)

    return segs_out


def remove_punctuation(in_str):
    """
    删除标点符号

    在计算 EM 时，忽略标点符号使评估更公平。
    例如："北京市。" 和 "北京市" 应被认为是相同的答案。

    参数:
        in_str (str): 输入字符串

    返回值:
        str: 删除标点后的字符串
    """
    in_str = str(in_str).lower().strip()
    sp_char = ['-', ':', '_', '*', '^', '/', '\\', '~', '`', '+', '=',
               '，', '。', '：', '？', '！', '"', '"', '；', ''', '《', '》', '……', '·', '、',
               '「', '」', '（', '）', '－', '～', '『', '』']
    out_segs = []
    for char in in_str:
        if char in sp_char:
            continue
        else:
            out_segs.append(char)
    return ''.join(out_segs)


def find_lcs(s1, s2):
    """
    计算两个序列的最长公共子序列（LCS）

    使用动态规划实现：
      m[i][j] = s1[0:i] 和 s2[0:j] 的 LCS 长度

    参数:
        s1 (list): 第一个序列（通常是真实答案的 token 列表）
        s2 (list): 第二个序列（通常是预测答案的 token 列表）

    返回值:
        tuple: (lcs_sequence, lcs_length)
    """
    # dp 表：m[i][j] 表示 s1[0:i] 和 s2[0:j] 的 LCS 长度
    m = [[0 for i in range(len(s2) + 1)] for j in range(len(s1) + 1)]
    mmax = 0  # 最长公共子序列的长度
    p = 0     # LCS 在 s1 中的结束位置

    # 双重循环填充 DP 表
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                # 如果当前字符匹配，长度 +1
                m[i + 1][j + 1] = m[i][j] + 1
                # 更新最长长度和位置
                if m[i + 1][j + 1] > mmax:
                    mmax = m[i + 1][j + 1]
                    p = i + 1

    # 从 s1 中提取 LCS
    return s1[p - mmax:p], mmax


def calc_f1_score(answers, prediction):
    """
    计算单个预测相对于所有参考答案的 F1 分数

    对多个参考答案中的每一个都计算一次 F1，然后取最高值。
    这样可以容纳众包标注中的多个答案。

    参数:
        answers (list): 参考答案列表（可能有多个）
        prediction (str): 预测答案

    返回值:
        float: 最高的 F1 分数（0-1）
    """
    f1_scores = []

    for ans in answers:
        # ============ 分词 ============
        # 对真实答案和预测答案都进行混合分词，删除标点
        ans_segs = mixed_segmentation(ans, rm_punc=True)
        prediction_segs = mixed_segmentation(prediction, rm_punc=True)

        # ============ 计算最长公共子序列 ============
        lcs, lcs_len = find_lcs(ans_segs, prediction_segs)

        # 边界情况：任何一方长度为 0，F1 = 0
        if len(prediction_segs) == 0 or len(ans_segs) == 0:
            if lcs_len == 0:
                f1_scores.append(0)
            continue

        # 如果没有公共部分，F1 = 0
        if lcs_len == 0:
            f1_scores.append(0)
            continue

        # ============ 计算精准度和召回率 ============
        # precision: 预测答案中有多少比例的 token 在真实答案中
        precision = 1.0 * lcs_len / len(prediction_segs)

        # recall: 真实答案中有多少比例的 token 在预测答案中
        recall = 1.0 * lcs_len / len(ans_segs)

        # F1 = 调和平均数
        f1 = (2 * precision * recall) / (precision + recall)
        f1_scores.append(f1)

    # 取所有参考答案中的最高 F1
    # 防御：如果 f1_scores 为空，返回 0
    return max(f1_scores) if f1_scores else 0


def calc_em_score(answers, prediction):
    """
    计算 EM (Exact Match) 分数

    检查预测答案是否与某个参考答案完全相同（忽略标点）。

    参数:
        answers (list): 参考答案列表
        prediction (str): 预测答案

    返回值:
        int: 1 if 完全匹配，0 otherwise
    """
    em = 0

    for ans in answers:
        # 删除标点并转为小写
        ans_ = remove_punctuation(ans)
        prediction_ = remove_punctuation(prediction)

        # 如果与任何一个参考答案相同，EM = 1
        if ans_ == prediction_:
            em = 1
            break

    return em


def evaluate(predictions, references):
    """
    在整个测试集上计算平均 F1 和 EM 分数

    参数:
        predictions (list): 预测结果列表，每个元素是：
            {
              'id': 'question_id',
              'prediction_text': '预测的答案'
            }
        references (list): 参考答案列表，每个元素是：
            {
              'id': 'question_id',
              'answers': {
                'text': ['答案1', '答案2', ...]  # 可能有多个参考答案
              }
            }

    返回值:
        dict: {
          'f1': 平均 F1 分数,
          'em': 平均 EM 分数,
          'avg': (F1 + EM) / 2,
          'total': 总问题数,
          'skip': 未回答的问题数
        }
    """
    f1 = 0
    em = 0
    total_count = 0
    skip_count = 0

    # ============ 构建预测和参考字典，便于通过 ID 查询 ============
    # 预测字典：{question_id → prediction_text}
    pred = dict([(data['id'], data['prediction_text']) for data in predictions])

    # 参考字典：{question_id → [answer_text1, answer_text2, ...]}
    ref = dict([(data['id'], data['answers']['text']) for data in references])

    # ============ 逐个问题计算 F1 和 EM ============
    for query_id, answers in ref.items():
        total_count += 1

        # 检查是否存在预测（某些问题可能未回答）
        if query_id not in pred:
            sys.stderr.write('Unanswered question: {}\n'.format(query_id))
            skip_count += 1
            continue

        prediction = pred[query_id]

        # 累积 F1 和 EM 分数
        f1 += calc_f1_score(answers, prediction)
        em += calc_em_score(answers, prediction)

    # ============ 计算平均分数 ============
    # 边界情况：如果 total_count = 0，避免除以 0
    if total_count == 0:
        return {
            'avg': 0,
            'f1': 0,
            'em': 0,
            'total': 0,
            'skip': 0
        }

    # 转换为百分比
    f1_score = 100.0 * f1 / total_count
    em_score = 100.0 * em / total_count

    return {
        'avg': (em_score + f1_score) * 0.5,  # 平均分
        'f1': f1_score,
        'em': em_score,
        'total': total_count,
        'skip': skip_count
    }