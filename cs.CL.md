# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Multi-Level Explanations for Generative Language Models](https://arxiv.org/abs/2403.14459) | 本文提出了一个名为MExGen的通用框架，通过引入标量化概念和多级方法处理生成式语言模型的挑战，证明可以提供更贴近本地的解释。 |

# 详细

[^1]: 生成式语言模型的多级解释

    Multi-Level Explanations for Generative Language Models

    [https://arxiv.org/abs/2403.14459](https://arxiv.org/abs/2403.14459)

    本文提出了一个名为MExGen的通用框架，通过引入标量化概念和多级方法处理生成式语言模型的挑战，证明可以提供更贴近本地的解释。

    

    基于扰动的解释方法，如LIME和SHAP，通常应用于文本分类。本文关注它们如何扩展到生成式语言模型。为了解决文本作为输出和长文本输入的挑战，我们提出了一个名为MExGen的通用框架，可以用不同的归因算法实例化。为了处理文本输出，我们引入了将文本映射到实数的标量化概念，并探讨了多种可能性。为了处理长输入，我们采用多级方法，从粗粒度到细粒度，重点关注具有模型查询线性缩放的算法。我们对基于扰动的归因方法进行了系统评估，包括自动化和人工评估，用于摘要和基于上下文的问答。结果表明，我们的框架可以提供更加贴近本地的生成式输出解释。

    arXiv:2403.14459v1 Announce Type: cross  Abstract: Perturbation-based explanation methods such as LIME and SHAP are commonly applied to text classification. This work focuses on their extension to generative language models. To address the challenges of text as output and long text inputs, we propose a general framework called MExGen that can be instantiated with different attribution algorithms. To handle text output, we introduce the notion of scalarizers for mapping text to real numbers and investigate multiple possibilities. To handle long inputs, we take a multi-level approach, proceeding from coarser levels of granularity to finer ones, and focus on algorithms with linear scaling in model queries. We conduct a systematic evaluation, both automated and human, of perturbation-based attribution methods for summarization and context-grounded question answering. The results show that our framework can provide more locally faithful explanations of generated outputs.
    

