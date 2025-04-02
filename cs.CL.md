# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Three-Phases SFT Hybrid Model Integrated Strong Prior Module and Data Overlap Estimation in the Eduation Context](https://arxiv.org/abs/2403.15426) | 提出了一种在教育领域中应用的三阶段监督微调模型，通过先验和数据重叠估计实现了教育知识的结构拆卸和增量引导输出。 |
| [^2] | [Large Language Models are In-Context Molecule Learners](https://arxiv.org/abs/2403.04197) | 提出了上下文分子适应（ICMA）范式，允许LLMs通过上下文示例学习分子-文本对齐，解决了在分子-标题翻译任务中对LLMs的挑战。 |

# 详细

[^1]: 教育环境下集成强先验模块和数据重叠估计的三阶段SFT混合模型

    A Three-Phases SFT Hybrid Model Integrated Strong Prior Module and Data Overlap Estimation in the Eduation Context

    [https://arxiv.org/abs/2403.15426](https://arxiv.org/abs/2403.15426)

    提出了一种在教育领域中应用的三阶段监督微调模型，通过先验和数据重叠估计实现了教育知识的结构拆卸和增量引导输出。

    

    在本文中，我们提出了一种端到端基于先验的三阶段监督微调模型，证明比传统微调方法更有竞争力。具体而言，我们的模型实现了教育知识的结构拆卸和增量引导输出。为此，我们通过采样器和重叠估计神经网络对三种类型的数据进行了健壮的分类，将预处理数据集分三批注入预训练模型进行LORA微调。然后，我们设计了一个先验模块，将系统提示、向量数据库和抽象语法树任务分割相结合。最后，对基于先验的微调模型应用了压缩方法和正则化约束，随后在输出端进行文本过滤以获得增量引导结果。我们的模型代表了真正以丰富的教育知识、分步指导的特点体现导师角色的第一项研究努力。

    arXiv:2403.15426v1 Announce Type: cross  Abstract: In this paper, we propose an end-to-end prior-based three-phases supervised fine-tuned model, which is proved more competitive than traditional fine-tuning method. More specifically, our model realizes the structural disassembly and incremental guided output of educational knowledge. To this end, we robustify data classification of three types via a sampler and overlap estimation neural network, and inject the preprocessing datasets into pre-trained model in three batches for LORA fine-tuning. Then, we design a prior module couples system prompt, vector databases, and abstract syntax tree task segmentation. Finally, the compression method and regularization constraint are applied to the prior-based fine-tuned model, followed by text filter at the output end to obtain incremental guided results. Our model represents the first research effort to truly embody the tutor role with the features of abundant educational knowledge, step-by-step
    
[^2]: 大规模语言模型是上下文分子学习器

    Large Language Models are In-Context Molecule Learners

    [https://arxiv.org/abs/2403.04197](https://arxiv.org/abs/2403.04197)

    提出了上下文分子适应（ICMA）范式，允许LLMs通过上下文示例学习分子-文本对齐，解决了在分子-标题翻译任务中对LLMs的挑战。

    

    大型语言模型（LLMs）在生物化学任务中表现出色，尤其是分子标题翻译任务，旨在弥合分子和自然语言文本之间的差距。然而，先前在适应LLMs到分子-标题翻译任务中的方法需要额外的领域特定预训练阶段，存在分子和文本空间之间的弱对齐，或对LLMs的规模有严格要求。为了解决这些挑战，我们提出了上下文分子适应（ICMA），作为一种新的范例，允许LLMs通过上下文示例学习分子-文本对齐，通过上下文分子调整。具体而言，ICMA包括以下三个阶段：跨模态检索、检索后排序和上下文分子调整。

    arXiv:2403.04197v1 Announce Type: cross  Abstract: Large Language Models (LLMs) have demonstrated exceptional performance in biochemical tasks, especially the molecule caption translation task, which aims to bridge the gap between molecules and natural language texts. However, previous methods in adapting LLMs to the molecule-caption translation task required extra domain-specific pre-training stages, suffered weak alignment between molecular and textual spaces, or imposed stringent demands on the scale of LLMs. To resolve the challenges, we propose In-Context Molecule Adaptation (ICMA), as a new paradigm allowing LLMs to learn the molecule-text alignment from context examples via In-Context Molecule Tuning. Specifically, ICMA incorporates the following three stages: Cross-modal Retrieval, Post-retrieval Re-ranking, and In-context Molecule Tuning. Initially, Cross-modal Retrieval utilizes BM25 Caption Retrieval and Molecule Graph Retrieval to retrieve informative context examples. Addi
    

