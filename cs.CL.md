# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Can Large Language Models Recall Reference Location Like Humans?](https://arxiv.org/abs/2402.17010) | 本文探讨了大型语言模型如何利用预训练阶段的知识回忆参考段落，提出了一个两阶段框架模拟人类回忆参考的过程。 |
| [^2] | [Scaling Efficient LLMs](https://arxiv.org/abs/2402.14746) | 训练得到的LLM模型通常是稀疏的，为了提高效率，研究了在训练语料上达到所需准确度的参数最少的高效LLM模型，得出了参数数量与自然训练语料规模之间的关系，并指出扩展可以揭示新技能。 |
| [^3] | [Audio Contrastive based Fine-tuning.](http://arxiv.org/abs/2309.11895) | 本论文提出了一种基于音频对比的微调方法（AudioConFit），通过借助对比学习的可转移性，该方法在各种音频分类任务中表现出强大的泛化能力，并在不同设置下实现了最先进的结果。 |
| [^4] | [Alpha-GPT: Human-AI Interactive Alpha Mining for Quantitative Investment.](http://arxiv.org/abs/2308.00016) | 本论文提出了一种通过引入人机交互的新型 alpha 挖掘范式，并利用大型语言模型的能力，通过一种新颖的提示工程算法框架，开发了 Alpha-GPT。通过多个实验，展示了 Alpha-GPT 在量化投资领域的有效性和优势。 |

# 详细

[^1]: 大型语言模型能像人类一样回忆参考位置吗？

    Can Large Language Models Recall Reference Location Like Humans?

    [https://arxiv.org/abs/2402.17010](https://arxiv.org/abs/2402.17010)

    本文探讨了大型语言模型如何利用预训练阶段的知识回忆参考段落，提出了一个两阶段框架模拟人类回忆参考的过程。

    

    在完成知识密集型任务时，人类有时不仅需要一个答案，还需要相应的参考段落供辅助阅读。先前的方法需要通过额外的检索模型获取预分段的文章块。本文探讨了利用大型语言模型（LLMs）的预训练阶段存储的参数化知识，独立于任何起始位置回忆参考段落。我们提出了一个模拟人类回忆易被遗忘参考的情景的两阶段框架。首先，LLM被提示回忆文档标题标识符以获取粗粒度文档集。然后，基于获得的粗粒度文档集，它回忆细粒度段落。在两阶段回忆过程中，我们使用约束解码来确保不生成存储文档之外的内容。为了增加速度，我们只回忆短前缀。

    arXiv:2402.17010v1 Announce Type: cross  Abstract: When completing knowledge-intensive tasks, humans sometimes need not just an answer but also a corresponding reference passage for auxiliary reading. Previous methods required obtaining pre-segmented article chunks through additional retrieval models. This paper explores leveraging the parameterized knowledge stored during the pre-training phase of large language models (LLMs) to independently recall reference passage from any starting position. We propose a two-stage framework that simulates the scenario of humans recalling easily forgotten references. Initially, the LLM is prompted to recall document title identifiers to obtain a coarse-grained document set. Then, based on the acquired coarse-grained document set, it recalls fine-grained passage. In the two-stage recall process, we use constrained decoding to ensure that content outside of the stored documents is not generated. To increase speed, we only recall a short prefix in the 
    
[^2]: 扩展高效的LLM模型

    Scaling Efficient LLMs

    [https://arxiv.org/abs/2402.14746](https://arxiv.org/abs/2402.14746)

    训练得到的LLM模型通常是稀疏的，为了提高效率，研究了在训练语料上达到所需准确度的参数最少的高效LLM模型，得出了参数数量与自然训练语料规模之间的关系，并指出扩展可以揭示新技能。

    

    训练得到的LLM模型通常是稀疏的，即大部分参数为零，这引发了关于效率的问题。为此，我们研究了高效的LLM模型，即那些在训练语料上达到所需准确度的参数最少。具体地，我们比较了当前规模下训练损失的理论和实证估计，以获得自然训练语料中独特序列数量上下界的数量。我们的结果暗示：(1)要在训练语料中表示的技能数量翻倍，需要将语料规模大约扩展三到五倍，(2)对于高效的LLM模型，参数数量$N$和自然训练语料规模$D$满足$N \sim D^{0.58}$的关系，(3)如果一个LLM模型的参数数量小于训练语料中的独特序列数量，扩展可以揭示出新的技能。

    arXiv:2402.14746v1 Announce Type: new  Abstract: Trained LLMs are typically sparse in that most of the parameters are zero, raising questions on efficiency. In response, we inquire into efficient LLMs, i.e. those with the fewest parameters that achieve the desired accuracy on a training corpus. Specifically, we compare theoretical and empirical estimates for training loss at current scale to obtain upper and lower bounds on the number of unique sequences in a natural training corpus as a function of its size. Our result implies (1) to double the number of skills represented in a training corpus, the corpus must scale roughly between three and five fold (2) for efficient LLMs, the number of parameters $N$ and the size $D$ of a natural training corpus scale as $N \sim D^{0.58}$ (3) if the number of parameters of an LLM is smaller than the number of unique sequences in the training corpus, scaling up can uncover emergent skills.
    
[^3]: 基于音频对比的微调方法

    Audio Contrastive based Fine-tuning. (arXiv:2309.11895v1 [cs.SD])

    [http://arxiv.org/abs/2309.11895](http://arxiv.org/abs/2309.11895)

    本论文提出了一种基于音频对比的微调方法（AudioConFit），通过借助对比学习的可转移性，该方法在各种音频分类任务中表现出强大的泛化能力，并在不同设置下实现了最先进的结果。

    

    音频分类在语音和声音处理任务中起着至关重要的作用，具有广泛的应用。在将模型拟合到训练数据（避免过拟合）并使其能够良好地泛化到新领域之间仍然存在着平衡的挑战。借助对比学习的可转移性，我们引入了基于音频对比的微调方法（AudioConFit），这种方法具有强大的泛化能力。对各种音频分类任务的实证实验表明了我们方法的有效性和鲁棒性，在不同设置下取得了最先进的结果。

    Audio classification plays a crucial role in speech and sound processing tasks with a wide range of applications. There still remains a challenge of striking the right balance between fitting the model to the training data (avoiding overfitting) and enabling it to generalise well to a new domain. Leveraging the transferability of contrastive learning, we introduce Audio Contrastive-based Fine-tuning (AudioConFit), an efficient approach characterised by robust generalisability. Empirical experiments on a variety of audio classification tasks demonstrate the effectiveness and robustness of our approach, which achieves state-of-the-art results in various settings.
    
[^4]: Alpha-GPT：人机交互式 Alpha 挖掘在量化投资中的应用

    Alpha-GPT: Human-AI Interactive Alpha Mining for Quantitative Investment. (arXiv:2308.00016v1 [q-fin.CP])

    [http://arxiv.org/abs/2308.00016](http://arxiv.org/abs/2308.00016)

    本论文提出了一种通过引入人机交互的新型 alpha 挖掘范式，并利用大型语言模型的能力，通过一种新颖的提示工程算法框架，开发了 Alpha-GPT。通过多个实验，展示了 Alpha-GPT 在量化投资领域的有效性和优势。

    

    在量化投资研究中，挖掘新的 alpha（有效的交易信号或因子）是其中最重要的任务之一。传统的 alpha 挖掘方法，无论是手工合成因子还是算法挖掘因子（如遗传编程搜索），都存在固有的局限性，尤其在实施量化分析师的想法方面。在本研究中，我们提出了一种新的 alpha 挖掘范式，引入了人机交互，并通过利用大型语言模型的能力，提出了一种新颖的提示工程算法框架来实现这个范式。此外，我们开发了 Alpha-GPT，一种新的交互式 alpha 挖掘系统框架，以一种启发式的方式“理解”量化研究人员的想法，并输出具有创造性、深入洞察力和有效性的 alpha。通过多个 alpha 挖掘实验，我们展示了 Alpha-GPT 的有效性和优势。

    One of the most important tasks in quantitative investment research is mining new alphas (effective trading signals or factors). Traditional alpha mining methods, either hand-crafted factor synthesizing or algorithmic factor mining (e.g., search with genetic programming), have inherent limitations, especially in implementing the ideas of quants. In this work, we propose a new alpha mining paradigm by introducing human-AI interaction, and a novel prompt engineering algorithmic framework to implement this paradigm by leveraging the power of large language models. Moreover, we develop Alpha-GPT, a new interactive alpha mining system framework that provides a heuristic way to ``understand'' the ideas of quant researchers and outputs creative, insightful, and effective alphas. We demonstrate the effectiveness and advantage of Alpha-GPT via a number of alpha mining experiments.
    

