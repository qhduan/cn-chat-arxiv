# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Juru: Legal Brazilian Large Language Model from Reputable Sources](https://arxiv.org/abs/2403.18140) | Juru 模型通过从巴西法律来源提取的19亿个唯一标记，展示了领域专门化可以在减少预训练数据量方面发挥作用，但这种专门化会导致同一语言中其他知识领域性能下降。 |
| [^2] | [Cheap Learning: Maximising Performance of Language Models for Social Data Science Using Minimal Data.](http://arxiv.org/abs/2401.12295) | 本文回顾了“廉价”学习技术在社会科学中的应用，包括弱监督、迁移学习和提示工程。特别地，通过提示大规模语言模型，可以实现高准确性的性能。 |
| [^3] | [Otter: A Multi-Modal Model with In-Context Instruction Tuning.](http://arxiv.org/abs/2305.03726) | Otter是一种多模态模型，引入了指令调整方法，基于OpenFlamingo训练，能够更好地指令跟随和上下文学习。 |

# 详细

[^1]: Juru: 来自可靠来源的巴西法律大语言模型

    Juru: Legal Brazilian Large Language Model from Reputable Sources

    [https://arxiv.org/abs/2403.18140](https://arxiv.org/abs/2403.18140)

    Juru 模型通过从巴西法律来源提取的19亿个唯一标记，展示了领域专门化可以在减少预训练数据量方面发挥作用，但这种专门化会导致同一语言中其他知识领域性能下降。

    

    与预训练大型语言模型相关的高计算成本限制了相关研究。为解决这一问题，出现了两种策略：领域专门化和使用高质量数据进行预训练。为探索这些策略，我们使用来自可靠巴西法律来源的19亿个唯一标记专门化了Sabi\'a-2 Small模型，并在法律和一般知识考试中进行了少样本评估。我们的模型Juru展示了领域专门化在减少预训练数据量方面的优势。然而，这种专门化是以在同一语言中其他知识领域性能下降为代价的。这项研究有助于增加的科学证据，表明预训练数据的选择可能提高大型语言模型的性能，从而能够以较低成本探索这些模型。

    arXiv:2403.18140v1 Announce Type: cross  Abstract: The high computational cost associated with pretraining large language models limits their research. Two strategies have emerged to address this issue: domain specialization and pretraining with high-quality data. To explore these strategies, we specialized the Sabi\'a-2 Small model with 1.9 billion unique tokens from reputable Brazilian legal sources and conducted few-shot evaluations on legal and general knowledge exams. Our model, Juru, demonstrates the benefits of domain specialization with a reduced amount of pretraining data. However, this specialization comes at the expense of degrading performance in other knowledge areas within the same language. This study contributes to the growing body of scientific evidence showing that pretraining data selection may enhance the performance of large language models, enabling the exploration of these models at a lower cost.
    
[^2]: 廉价学习：最大化社会数据科学中语言模型的性能，使用最少的数据。

    Cheap Learning: Maximising Performance of Language Models for Social Data Science Using Minimal Data. (arXiv:2401.12295v1 [cs.CL])

    [http://arxiv.org/abs/2401.12295](http://arxiv.org/abs/2401.12295)

    本文回顾了“廉价”学习技术在社会科学中的应用，包括弱监督、迁移学习和提示工程。特别地，通过提示大规模语言模型，可以实现高准确性的性能。

    

    机器学习领域在构建新模型时，最近取得了降低标注训练数据要求的重要进展。这些“廉价”学习技术在社会科学领域具有巨大潜力，因为开发大型标注训练数据集通常是机器学习用于分析任务的实际障碍。在本文中，我们回顾了最近发展的三种“廉价”技术：弱监督、迁移学习和提示工程。对于后者，我们还回顾了大规模语言模型的零样本提示的特殊情况。针对每种技术，我们提供了工作原理的指南，并展示了它们在六个不同的实际社会科学应用程序中的应用情况（两个不同任务与三种不同数据集的组合）。我们展示了所有技术的良好性能，特别是我们演示了如何通过大规模语言模型的提示可以实现很高的准确性。

    The field of machine learning has recently made significant progress in reducing the requirements for labelled training data when building new models. These `cheaper' learning techniques hold significant potential for the social sciences, where development of large labelled training datasets is often a significant practical impediment to the use of machine learning for analytical tasks. In this article we review three `cheap' techniques that have developed in recent years: weak supervision, transfer learning and prompt engineering. For the latter, we also review the particular case of zero-shot prompting of large language models. For each technique we provide a guide of how it works and demonstrate its application across six different realistic social science applications (two different tasks paired with three different dataset makeups). We show good performance for all techniques, and in particular we demonstrate how prompting of large language models can achieve high accuracy at very
    
[^3]: Otter: 一种多模态模型及其上下文指令调整方法

    Otter: A Multi-Modal Model with In-Context Instruction Tuning. (arXiv:2305.03726v1 [cs.CV])

    [http://arxiv.org/abs/2305.03726](http://arxiv.org/abs/2305.03726)

    Otter是一种多模态模型，引入了指令调整方法，基于OpenFlamingo训练，能够更好地指令跟随和上下文学习。

    

    巨大的语言模型(LLMs)由于预训练了大量文本数据而展示出在各种任务中以零/少数据学习的显著普适能力，例如GPT-3，它推出了InstrctGPT和ChatGPT，能够通过自然语言指令完成真实世界的任务。本文提出了将指令调整引入到多模态模型中的想法，受到Flamingo模型上游交替格式预训练数据集的启发。我们采用类似的方法构建了我们的MultI-Modal In-Context Instruction Tuning (MIMIC-IT)数据集。我们提出了Otter，一种基于OpenFlamingo的多模态模型(DeepMind的Flamingo的开源版本)，它在MIMIC-IT上进行训练，并展示了更好的指令跟随能力和上下文学习能力。我们还针对研究人员优化了OpenFlamingo的实现，将所需的训练资源从1个A100 GPU降至4个RTX-3090 GPU，从而使研究更具民主性。

    Large language models (LLMs) have demonstrated significant universal capabilities as few/zero-shot learners in various tasks due to their pre-training on vast amounts of text data, as exemplified by GPT-3, which boosted to InstrctGPT and ChatGPT, effectively following natural language instructions to accomplish real-world tasks. In this paper, we propose to introduce instruction tuning into multi-modal models, motivated by the Flamingo model's upstream interleaved format pretraining dataset. We adopt a similar approach to construct our MultI-Modal In-Context Instruction Tuning (MIMIC-IT) dataset. We then introduce Otter, a multi-modal model based on OpenFlamingo (open-sourced version of DeepMind's Flamingo), trained on MIMIC-IT and showcasing improved instruction-following ability and in-context learning. We also optimize OpenFlamingo's implementation for researchers, democratizing the required training resources from 1$\times$ A100 GPU to 4$\times$ RTX-3090 GPUs, and integrate both Op
    

