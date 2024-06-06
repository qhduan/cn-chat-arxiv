# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Stealthy Attack on Large Language Model based Recommendation](https://arxiv.org/abs/2402.14836) | 大型语言模型推荐系统容易受到隐秘攻击，攻击者可以通过微调文本内容在不干预模型训练的情况下显著提高物品的曝光度，而这种攻击对整体推荐性能无影响且难以被检测到。 |
| [^2] | [Self-Augmented In-Context Learning for Unsupervised Word Translation](https://arxiv.org/abs/2402.10024) | 通过自学习上下文增强方法，本论文提出一种无监督词汇翻译的方法，在零样本提示的大型语言模型上取得了显著的改进，超过了传统基于映射的方法。 |
| [^3] | [Visualization for Recommendation Explainability: A Survey and New Perspectives.](http://arxiv.org/abs/2305.11755) | 本文回顾了推荐系统中有关可视化解释的研究，提出了一组可能有益于设计解释性可视化推荐系统的指南。 |

# 详细

[^1]: 大型语言模型推荐中的隐秘攻击

    Stealthy Attack on Large Language Model based Recommendation

    [https://arxiv.org/abs/2402.14836](https://arxiv.org/abs/2402.14836)

    大型语言模型推荐系统容易受到隐秘攻击，攻击者可以通过微调文本内容在不干预模型训练的情况下显著提高物品的曝光度，而这种攻击对整体推荐性能无影响且难以被检测到。

    

    最近，强大的大型语言模型(LLMs)在推动推荐系统(RS)的进展方面发挥了重要作用。然而，尽管这些系统蓬勃发展，但它们对安全威胁的敏感性却被大多忽视了。在这项工作中，我们揭示了LLMs引入推荐模型中产生新安全漏洞的情况，这是由于它们注重物品的文本内容。我们证明了攻击者可以在测试阶段仅通过改变物品的文本内容显著增加其曝光度，而无需直接干预模型的训练过程。此外，该攻击具有显著的隐秘性，因为它不会影响整体推荐性能，对文本的修改微妙，使用户和平台难以检测到。我们在四个主流的LLM-based推荐模型上进行了全面的实验。

    arXiv:2402.14836v1 Announce Type: cross  Abstract: Recently, the powerful large language models (LLMs) have been instrumental in propelling the progress of recommender systems (RS). However, while these systems have flourished, their susceptibility to security threats has been largely overlooked. In this work, we reveal that the introduction of LLMs into recommendation models presents new security vulnerabilities due to their emphasis on the textual content of items. We demonstrate that attackers can significantly boost an item's exposure by merely altering its textual content during the testing phase, without requiring direct interference with the model's training process. Additionally, the attack is notably stealthy, as it does not affect the overall recommendation performance and the modifications to the text are subtle, making it difficult for users and platforms to detect. Our comprehensive experiments across four mainstream LLM-based recommendation models demonstrate the superior
    
[^2]: 自学习上下文增强对于无监督词汇翻译的研究

    Self-Augmented In-Context Learning for Unsupervised Word Translation

    [https://arxiv.org/abs/2402.10024](https://arxiv.org/abs/2402.10024)

    通过自学习上下文增强方法，本论文提出一种无监督词汇翻译的方法，在零样本提示的大型语言模型上取得了显著的改进，超过了传统基于映射的方法。

    

    近期的研究表明，尽管大型语言模型在一些小规模的设置中展示出了较强的词汇翻译和双语词典诱导(BLI)的能力，但在无监督的情况下，即没有种子翻译对可用的情况下，尤其是对于资源较少的语言，它们仍然无法达到“传统”的基于映射的方法的性能。为了解决这个挑战，我们提出了一种自学习上下文增强方法 (SAIL) 来进行无监督的BLI：从零样本提示开始，SAIL通过迭代地从LLM中引出一组高置信度的词汇翻译对，然后在ICL的方式下再次应用于同一个LLM中。我们的方法在两个广泛的BLI基准测试中，跨越多种语言对，在零样本提示的LLM上取得了显著的改进，也在各个方面优于基于映射的基线。除了达到最先进的无监督

    arXiv:2402.10024v1 Announce Type: cross  Abstract: Recent work has shown that, while large language models (LLMs) demonstrate strong word translation or bilingual lexicon induction (BLI) capabilities in few-shot setups, they still cannot match the performance of 'traditional' mapping-based approaches in the unsupervised scenario where no seed translation pairs are available, especially for lower-resource languages. To address this challenge with LLMs, we propose self-augmented in-context learning (SAIL) for unsupervised BLI: starting from a zero-shot prompt, SAIL iteratively induces a set of high-confidence word translation pairs for in-context learning (ICL) from an LLM, which it then reapplies to the same LLM in the ICL fashion. Our method shows substantial gains over zero-shot prompting of LLMs on two established BLI benchmarks spanning a wide range of language pairs, also outperforming mapping-based baselines across the board. In addition to achieving state-of-the-art unsupervised 
    
[^3]: 推荐系统解释的可视化：综述和新视角

    Visualization for Recommendation Explainability: A Survey and New Perspectives. (arXiv:2305.11755v1 [cs.IR])

    [http://arxiv.org/abs/2305.11755](http://arxiv.org/abs/2305.11755)

    本文回顾了推荐系统中有关可视化解释的研究，提出了一组可能有益于设计解释性可视化推荐系统的指南。

    

    为推荐提供系统生成的解释是实现透明且值得信赖的推荐系统的重要步骤。可解释的推荐系统为输出提供了人类可理解的基础。在过去的20年中，可解释的推荐引起了推荐系统研究社区的广泛关注。本文旨在全面回顾推荐系统中有关可视化解释的研究工作。更具体地，我们根据解释目标、解释范围、解释样式和解释格式这四个维度系统地审查推荐系统中有关解释的文献。认识到可视化的重要性，我们从解释性视觉方式的角度途径推荐系统文献，即使用可视化作为解释的显示样式。因此，我们得出了一组可能有益于设计解释性可视化推荐系统的指南。

    Providing system-generated explanations for recommendations represents an important step towards transparent and trustworthy recommender systems. Explainable recommender systems provide a human-understandable rationale for their outputs. Over the last two decades, explainable recommendation has attracted much attention in the recommender systems research community. This paper aims to provide a comprehensive review of research efforts on visual explanation in recommender systems. More concretely, we systematically review the literature on explanations in recommender systems based on four dimensions, namely explanation goal, explanation scope, explanation style, and explanation format. Recognizing the importance of visualization, we approach the recommender system literature from the angle of explanatory visualizations, that is using visualizations as a display style of explanation. As a result, we derive a set of guidelines that might be constructive for designing explanatory visualizat
    

