# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Taming Pre-trained LLMs for Generalised Time Series Forecasting via Cross-modal Knowledge Distillation](https://arxiv.org/abs/2403.07300) | 通过跨模态知识蒸馏和LLMs对齐框架，该方法利用静态和动态知识，充分释放LLMs在时间序列预测中的潜力 |
| [^2] | [An In-depth Evaluation of GPT-4 in Sentence Simplification with Error-based Human Assessment](https://arxiv.org/abs/2403.04963) | 本文深入评估了GPT-4在句子简化中的表现，指出现有自动评估指标和人类评估方法对于大型语言模型的适用性仍有待进一步研究。 |

# 详细

[^1]: 通过跨模态知识蒸馏控制预训练LLMs进行广义时间序列预测

    Taming Pre-trained LLMs for Generalised Time Series Forecasting via Cross-modal Knowledge Distillation

    [https://arxiv.org/abs/2403.07300](https://arxiv.org/abs/2403.07300)

    通过跨模态知识蒸馏和LLMs对齐框架，该方法利用静态和动态知识，充分释放LLMs在时间序列预测中的潜力

    

    多变量时间序列预测最近随着深度学习模型的快速增长取得了巨大成功。然而，现有方法通常使用有限的时间数据从头开始训练模型，阻碍了它们的泛化。最近，随着大语言模型（LLMs）的激增，一些工作尝试将LLMs引入时间序列预测中。尽管取得了有希望的结果，但这些方法直接将时间序列作为LLMs的输入，忽略了时间和文本数据之间固有的模态差距。在这项工作中，我们提出了一个新颖的大语言模型和时间序列对齐框架，称为LLaTA，以充分发挥LLMs在时间序列预测挑战中的潜力。基于跨模态知识蒸馏，所提出的方法利用了预训练LLMs中的输入无关静态知识和输入相关动态知识。通过这种方式，该方法为预测模型赋能

    arXiv:2403.07300v1 Announce Type: cross  Abstract: Multivariate time series forecasting has recently gained great success with the rapid growth of deep learning models. However, existing approaches usually train models from scratch using limited temporal data, preventing their generalization. Recently, with the surge of the Large Language Models (LLMs), several works have attempted to introduce LLMs into time series forecasting. Despite promising results, these methods directly take time series as the input to LLMs, ignoring the inherent modality gap between temporal and text data. In this work, we propose a novel Large Language Models and time series alignment framework, dubbed LLaTA, to fully unleash the potentials of LLMs in the time series forecasting challenge. Based on cross-modal knowledge distillation, the proposed method exploits both input-agnostic static knowledge and input-dependent dynamic knowledge in pre-trained LLMs. In this way, it empowers the forecasting model with f
    
[^2]: 在基于错误的人类评估中深入评估GPT-4在句子简化中的表现

    An In-depth Evaluation of GPT-4 in Sentence Simplification with Error-based Human Assessment

    [https://arxiv.org/abs/2403.04963](https://arxiv.org/abs/2403.04963)

    本文深入评估了GPT-4在句子简化中的表现，指出现有自动评估指标和人类评估方法对于大型语言模型的适用性仍有待进一步研究。

    

    句子简化是一种重写句子以便更易阅读和理解的方法，对于帮助有各种阅读难题的人来说是一种有前途的技术。随着先进大型语言模型（LLMs）的兴起，评估它们在句子简化中的表现变得迫在眉睫。最近的研究利用自动评估指标和人类评估来评估LLMs的简化能力。然而，现有评估方法对LLMs在简化评估中的适用性仍然存在疑问。首先，现有自动指标在LLMs的简化评估中的适用性仍不确定。其次，当前在句子简化中的人类评估方法通常陷入两个极端：要么过于肤浅，无法清晰理解模型的表现，要么过于详细，使注释过程复杂且容易出现不一致性，从而影响评估的可靠性。

    arXiv:2403.04963v1 Announce Type: cross  Abstract: Sentence simplification, which rewrites a sentence to be easier to read and understand, is a promising technique to help people with various reading difficulties. With the rise of advanced large language models (LLMs), evaluating their performance in sentence simplification has become imperative. Recent studies have used both automatic metrics and human evaluations to assess the simplification abilities of LLMs. However, the suitability of existing evaluation methodologies for LLMs remains in question. First, the suitability of current automatic metrics on LLMs' simplification evaluation is still uncertain. Second, current human evaluation approaches in sentence simplification often fall into two extremes: they are either too superficial, failing to offer a clear understanding of the models' performance, or overly detailed, making the annotation process complex and prone to inconsistency, which in turn affects the evaluation's reliabil
    

