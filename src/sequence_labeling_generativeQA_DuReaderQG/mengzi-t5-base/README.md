---
license: Apache License 2.0

tasks:
- text2text-generation

model-type:
- t5

domain:
- nlp

frameworks:
- pytorch

backbone:
- transformer

metrics:
- BLEU-4
- Rouge-L

language:
- zh

tags:
- transformer
- 澜舟科技
- Langboat
- mengzi

widgets:
- task: text2text-generation
  inputs:
    - type: text
      name: input
      title: 输入文字
      validator:
        max_words: 128
  examples:
      - name: 1
        title: 示例1
        inputs:
          - name: text
            data: 中国的首都位于<extra_id_0>。 
inferencespec:  
      cpu: 2  
      memory: 12000  
      gpu: 0  
      gpu_memory: 16000  
---


# 孟子中文T5预训练生成模型介绍
尽管预训练语言模型在NLP的各个领域里得到了广泛的应用，但是其高昂的时间和算力成本依然是一个亟需解决的问题。这要求我们在一定的算力约束下，研发出各项指标更优的模型。 我们的目标不是追求更大的模型规模，而是轻量级但更强大，同时对部署和工业落地更友好的模型。 基于语言学信息融入和训练加速等方法，我们研发了Mengzi 系列模型。这个模型页面提供了孟子中文T5预训练生成模型，可以用于下游的生成场景。

详细的技术报告请参考: [Mengzi: Towards Lightweight yet Ingenious Pre-trained Models for Chinese](https://arxiv.org/abs/2110.06696)

## 模型描述

针对实际场景中常见的文本生成需求，孟子中文T5预训练生成模型与T5结构相同，不包含下游任务，需要在特定任务上 Finetune 后使用。与 GPT 定位不同，不适合文本续写。适用于文案生成、新闻生成等可控文本生成任务。孟子T5具有以下特点：

- 与 T5 结构相同，不包含下游任务，只有无监督数据训练
- 适应各类生成任务：T5可用于各类不同的生成任务，如摘要、问题生成、paraphrasing等。
- 方便易用：下游使用方便，基于T5的传统encoder-decoder框架。

T5模型的详细介绍见：[Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/pdf/1910.10683.pdf)


## 期望模型使用方式以及适用范围
本模型主要用于给输入文档生成摘要内容。用户可以自行尝试各种输入文档。具体调用方式请参考代码示例。

### 如何使用
在安装完成Modelscope之后即可使用text2text-generation的能力


#### 代码范例
```python
from modelscope.models import Model
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

pipeline_ins = pipeline(task=Tasks.text2text_generation, model='langboat/mengzi-t5-base')
result = pipeline_ins(input='中国的首都位于<extra_id_0>。')
print (result)
#{'text': '北京'}


from modelscope.models.nlp import T5ForConditionalGeneration
from modelscope.preprocessors import TextGenerationTransformersPreprocessor
model = T5ForConditionalGeneration.from_pretrained('langboat/mengzi-t5-base')
preprocessor = TextGenerationTransformersPreprocessor(model.model_dir)
pipeline_ins = pipeline(
    task=Tasks.text2text_generation,
    model=model,
    preprocessor=preprocessor)
print(pipeline_ins('中国的首都位于<extra_id_0>。'))
#{'text': '北京'}

```

### 模型局限性以及可能的偏差
模型在大量无监督数据上训练，没有加入下游任务，所以是通用的预训练模型，不能直接在下游任务场景使用



### 相关论文以及引用信息
如果我们的模型对您有帮助，请您引用我们的文章：
```
@misc{zhang2021mengzi,
      title={Mengzi: Towards Lightweight yet Ingenious Pre-trained Models for Chinese}, 
      author={Zhuosheng Zhang and Hanqing Zhang and Keming Chen and Yuhang Guo and Jingyun Hua and Yulong Wang and Ming Zhou},
      year={2021},
      eprint={2110.06696},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
