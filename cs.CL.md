# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [I Learn Better If You Speak My Language: Enhancing Large Language Model Fine-Tuning with Style-Aligned Response Adjustments](https://arxiv.org/abs/2402.11192) | 将微调过程中的实际响应风格与大型语言模型固有风格相匹配能够产生更好的学习结果，开发的方法通过最小程度地调整模型响应来避免过拟合。 |

# 详细

[^1]: 如果你讲我的语言，我会更好地学习：使用风格对齐响应调整增强大型语言模型微调

    I Learn Better If You Speak My Language: Enhancing Large Language Model Fine-Tuning with Style-Aligned Response Adjustments

    [https://arxiv.org/abs/2402.11192](https://arxiv.org/abs/2402.11192)

    将微调过程中的实际响应风格与大型语言模型固有风格相匹配能够产生更好的学习结果，开发的方法通过最小程度地调整模型响应来避免过拟合。

    

    使用小数据集为特定任务微调大型语言模型(LLMs)是一个普遍遇到的但复杂的挑战。在有限的示例上过多拟合可能会对模型的泛化能力和保留原始技能产生负面影响。我们的研究探讨了在微调过程中地实际响应风格的影响。我们发现将地实际响应风格与LLM固有风格匹配会产生更好的学习结果。基于这一观点，我们开发了一种方法，最小程度地修改LLM的现有响应以更正错误，使用这些调整后的响应作为训练目标。这种技术能够实现与模型固有响应风格一致的精确更正，维护模型的核心能力，从而避免过多拟合。我们的研究结果表明，这种方法不仅提高了LLM的特定任务准确性，而且关键地

    arXiv:2402.11192v1 Announce Type: cross  Abstract: Fine-tuning large language models (LLMs) with a small data set for particular tasks is a widely encountered yet complex challenge. The potential for overfitting on a limited number of examples can negatively impact the model's ability to generalize and retain its original skills. Our research explores the impact of the style of ground-truth responses during the fine-tuning process. We found that matching the ground-truth response style with the LLM's inherent style results in better learning outcomes. Building on this insight, we developed a method that minimally alters the LLM's pre-existing responses to correct errors, using these adjusted responses as training targets. This technique enables precise corrections in line with the model's native response style, safeguarding the model's core capabilities and thus avoid overfitting. Our findings show that this approach not only improves the LLM's task-specific accuracy but also crucially
    

