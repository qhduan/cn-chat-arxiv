# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Understanding and Mitigating the Threat of Vec2Text to Dense Retrieval Systems](https://arxiv.org/abs/2402.12784) | 本文研究了Vec2Text对密集检索系统的威胁以及如何缓解，通过对距离度量、池化函数、瓶颈预训练等方面进行深入分析，以获得对密集检索系统中文本可恢复性和检索效果权衡关键元素的更深入理解。 |

# 详细

[^1]: 理解和缓解Vec2Text对密集检索系统的威胁

    Understanding and Mitigating the Threat of Vec2Text to Dense Retrieval Systems

    [https://arxiv.org/abs/2402.12784](https://arxiv.org/abs/2402.12784)

    本文研究了Vec2Text对密集检索系统的威胁以及如何缓解，通过对距离度量、池化函数、瓶颈预训练等方面进行深入分析，以获得对密集检索系统中文本可恢复性和检索效果权衡关键元素的更深入理解。

    

    引入Vec2Text技术，一种用于反转文本嵌入的技术，引发了人们对密集检索系统中存在严重隐私问题的担忧，包括那些使用OpenAI和Cohere提供的文本嵌入的系统。这种威胁来自于一个恶意攻击者通过访问文本嵌入来重构原始文本。本文研究了影响使用Vec2Text恢复文本的嵌入模型的各个方面。我们的探索涉及距离度量、池化函数、瓶颈预训练、加噪声训练、嵌入量化和嵌入维度等因素，这些因素在原始Vec2Text论文中尚未被讨论。通过对这些因素的深入分析，我们旨在更深入地了解影响密集检索系统中文本可恢复性和检索效果之间权衡的关键因素。

    arXiv:2402.12784v1 Announce Type: cross  Abstract: The introduction of Vec2Text, a technique for inverting text embeddings, has raised serious privacy concerns within dense retrieval systems utilizing text embeddings, including those provided by OpenAI and Cohere. This threat comes from the ability for a malicious attacker with access to text embeddings to reconstruct the original text.   In this paper, we investigate various aspects of embedding models that could influence the recoverability of text using Vec2Text. Our exploration involves factors such as distance metrics, pooling functions, bottleneck pre-training, training with noise addition, embedding quantization, and embedding dimensions -- aspects not previously addressed in the original Vec2Text paper. Through a thorough analysis of these factors, our aim is to gain a deeper understanding of the critical elements impacting the trade-offs between text recoverability and retrieval effectiveness in dense retrieval systems. This a
    

