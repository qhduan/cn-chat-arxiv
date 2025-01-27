# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [NLP Verification: Towards a General Methodology for Certifying Robustness](https://arxiv.org/abs/2403.10144) | 本文尝试总结和评估由该领域迄今进展而形成的NLP验证流程的一般组成部分，贡献在于提出了将句子嵌入连续空间得到的可验证子空间的一般描述。 |
| [^2] | [What's in a Name? Auditing Large Language Models for Race and Gender Bias](https://arxiv.org/abs/2402.14875) | 调查发现，大型语言模型存在种族和性别偏见，尤其对与黑人女性相关的名字表现最不利。审计在模型部署和实施时的重要性得到强调。 |

# 详细

[^1]: NLP验证：走向一种通用的用于认证鲁棒性的方法论

    NLP Verification: Towards a General Methodology for Certifying Robustness

    [https://arxiv.org/abs/2403.10144](https://arxiv.org/abs/2403.10144)

    本文尝试总结和评估由该领域迄今进展而形成的NLP验证流程的一般组成部分，贡献在于提出了将句子嵌入连续空间得到的可验证子空间的一般描述。

    

    深度神经网络在自然语言处理（NLP）领域取得了显著成功，确保它们的安全性和可靠性至关重要：在安全关键的情境中，这些模型必须对变化或攻击具有鲁棒性，并能对其输出给出保证。与计算机视觉不同，NLP缺乏一个统一的验证方法论，尽管近年来文献中取得了一些进展，但对于NLP验证的实用问题常常涉及不深。在本文中，我们尝试提炼和评估一个NLP验证流程的一般组成部分，该流程来源于迄今为止该领域的进展。我们的贡献有两方面：首先，我们给出了将句子嵌入连续空间得到的可验证子空间的一般描述。我们确定了可验证子空间的语义泛化技术挑战，并提出了一种有效处理的方法。

    arXiv:2403.10144v1 Announce Type: cross  Abstract: Deep neural networks have exhibited substantial success in the field of Natural Language Processing (NLP) and ensuring their safety and reliability is crucial: there are safety critical contexts where such models must be robust to variability or attack, and give guarantees over their output. Unlike Computer Vision, NLP lacks a unified verification methodology and, despite recent advancements in literature, they are often light on the pragmatical issues of NLP verification. In this paper, we make an attempt to distil and evaluate general components of an NLP verification pipeline, that emerges from the progress in the field to date. Our contributions are two-fold. Firstly, we give a general characterisation of verifiable subspaces that result from embedding sentences into continuous spaces. We identify, and give an effective method to deal with, the technical challenge of semantic generalisability of verified subspaces; and propose it a
    
[^2]: 名字的含义是什么？审计大型语言模型中的种族和性别偏见

    What's in a Name? Auditing Large Language Models for Race and Gender Bias

    [https://arxiv.org/abs/2402.14875](https://arxiv.org/abs/2402.14875)

    调查发现，大型语言模型存在种族和性别偏见，尤其对与黑人女性相关的名字表现最不利。审计在模型部署和实施时的重要性得到强调。

    

    我们采用审计设计来调查最先进的大型语言模型中的偏见，包括GPT-4。在我们的研究中，我们引发模型在各种情景下为个人提供建议，比如在购车谈判或选举结果预测过程中。我们发现该建议系统性地对与种族少数群体和女性常见相关的名字产生不利影响。与黑人女性相关的名字得到的结果最不利。这些偏见在42个提示模板和多个模型中都是一致的，表明这是一个系统性问题，而不是孤立事件。在提示中提供数值、与决策相关的锚点可以成功抵消偏见，而定性细节的影响并不一致，甚至可能会加剧差异。我们的研究结果强调了在语言模型部署和实施时进行审计的重要性，以减轻其潜在影响。

    arXiv:2402.14875v1 Announce Type: cross  Abstract: We employ an audit design to investigate biases in state-of-the-art large language models, including GPT-4. In our study, we elicit prompt the models for advice regarding an individual across a variety of scenarios, such as during car purchase negotiations or election outcome predictions. We find that the advice systematically disadvantages names that are commonly associated with racial minorities and women. Names associated with Black women receive the least advantageous outcomes. The biases are consistent across 42 prompt templates and several models, indicating a systemic issue rather than isolated incidents. While providing numerical, decision-relevant anchors in the prompt can successfully counteract the biases, qualitative details have inconsistent effects and may even increase disparities. Our findings underscore the importance of conducting audits at the point of LLM deployment and implementation to mitigate their potential for
    

