# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Attention is Naturally Sparse with Gaussian Distributed Input](https://arxiv.org/abs/2404.02690) | 通过对高斯输入下注意力得分稀疏性进行理论分析，揭示了注意力机制中稀疏性的特征及其对计算效率的影响。 |
| [^2] | [Investigating Continual Pretraining in Large Language Models: Insights and Implications](https://arxiv.org/abs/2402.17400) | 本研究探讨了大型语言模型中的持续预训练领域，提出了一种能够在不同领域中整合新信息、保留已学知识并增强跨领域知识转移能力的策略，并引入了新的基准来评估模型的适应性。 |
| [^3] | [Evaluating the Performance of ChatGPT for Spam Email Detection](https://arxiv.org/abs/2402.15537) | 该研究评估了ChatGPT在英文和中文电子邮件数据集中用于垃圾邮件检测的性能，并探讨了其在这一领域的潜力。 |

# 详细

[^1]: 注意力机制在高斯分布输入下自然稀疏

    Attention is Naturally Sparse with Gaussian Distributed Input

    [https://arxiv.org/abs/2404.02690](https://arxiv.org/abs/2404.02690)

    通过对高斯输入下注意力得分稀疏性进行理论分析，揭示了注意力机制中稀疏性的特征及其对计算效率的影响。

    

    大型语言模型（LLMs）的计算强度是关键瓶颈，主要是由于transformer架构中注意力机制的$O(n^2)$复杂度。稀疏注意力作为一个关键创新应运而生，旨在减少计算负荷同时保持模型性能。本研究对LLMs内的注意力分数稀疏性进行了严格的理论分析，特别是在高斯输入框架下。通过建立一组基础假设并采用一种系统的理论方法，我们揭示了注意力分数稀疏性的内在特征及其对计算效率的影响。我们的主要贡献在于提供了对注意力机制中稀疏性表现形式的详细理论检查，揭示了在计算节约和模型有效性之间潜在权衡的见解。

    arXiv:2404.02690v1 Announce Type: cross  Abstract: The computational intensity of Large Language Models (LLMs) is a critical bottleneck, primarily due to the $O(n^2)$ complexity of the attention mechanism in transformer architectures. Addressing this, sparse attention emerges as a key innovation, aiming to reduce computational load while maintaining model performance. This study presents a rigorous theoretical analysis of the sparsity in attention scores within LLMs, particularly under the framework of Gaussian inputs. By establishing a set of foundational assumptions and employing a methodical theoretical approach, we unravel the intrinsic characteristics of attention score sparsity and its implications on computational efficiency. Our main contribution lies in providing a detailed theoretical examination of how sparsity manifests in attention mechanisms, offering insights into the potential trade-offs between computational savings and model effectiveness. This work not only advances 
    
[^2]: 探讨大型语言模型中的持续预训练：见解与影响

    Investigating Continual Pretraining in Large Language Models: Insights and Implications

    [https://arxiv.org/abs/2402.17400](https://arxiv.org/abs/2402.17400)

    本研究探讨了大型语言模型中的持续预训练领域，提出了一种能够在不同领域中整合新信息、保留已学知识并增强跨领域知识转移能力的策略，并引入了新的基准来评估模型的适应性。

    

    本文研究了大型语言模型（LLMs）中不断学习（CL）领域的发展，重点是制定高效可持续培训策略。我们的主要重点是持续领域自适应预训练，这是一个旨在使LLMs能够整合来自不同领域的新信息，同时保留以前学到的知识，并增强跨领域知识转移能力而不依赖于特定领域识别的过程。与以往主要集中于有限任务或领域并主要旨在解决遗忘问题的先前研究不同，我们的研究评估了LLMs对实际情景中不断变化的数据景观的适应性和能力。为此，我们引入了一个新的基准，旨在衡量LLMs对这些不断演变的数据环境的适应性，提供了一个全面的评估框架。

    arXiv:2402.17400v1 Announce Type: new  Abstract: This paper studies the evolving domain of Continual Learning (CL) in large language models (LLMs), with a focus on developing strategies for efficient and sustainable training. Our primary emphasis is on continual domain-adaptive pretraining, a process designed to equip LLMs with the ability to integrate new information from various domains while retaining previously learned knowledge and enhancing cross-domain knowledge transfer without relying on domain-specific identification. Unlike previous studies, which mostly concentrate on a limited selection of tasks or domains and primarily aim to address the issue of forgetting, our research evaluates the adaptability and capabilities of LLMs to changing data landscapes in practical scenarios. To this end, we introduce a new benchmark designed to measure the adaptability of LLMs to these evolving data environments, offering a comprehensive framework for evaluation. We examine the impact of mo
    
[^3]: 评估ChatGPT用于垃圾邮件检测的性能

    Evaluating the Performance of ChatGPT for Spam Email Detection

    [https://arxiv.org/abs/2402.15537](https://arxiv.org/abs/2402.15537)

    该研究评估了ChatGPT在英文和中文电子邮件数据集中用于垃圾邮件检测的性能，并探讨了其在这一领域的潜力。

    

    电子邮件继续是专业和商业领域中至关重要且广泛使用的通信媒介。然而，垃圾邮件的普及给用户带来了重大挑战，扰乱了他们的日常工作并降低了生产率。因此，基于内容准确地识别和过滤垃圾邮件对网络安全至关重要。最近自然语言处理领域的发展，特别是大型语言模型如ChatGPT，在诸如问答和文本生成等任务中表现出色。然而，其在垃圾邮件识别方面的潜力尚未得到充分探索。为了填补这一空白，本研究尝试评估ChatGPT在英文和中文电子邮件数据集中用于垃圾邮件识别的能力。我们利用ChatGPT进行垃圾邮件检测，采用上下文学习，需要提示说明和少量示范。

    arXiv:2402.15537v1 Announce Type: cross  Abstract: Email continues to be a pivotal and extensively utilized communication medium within professional and commercial domains. Nonetheless, the prevalence of spam emails poses a significant challenge for users, disrupting their daily routines and diminishing productivity. Consequently, accurately identifying and filtering spam based on content has become crucial for cybersecurity. Recent advancements in natural language processing, particularly with large language models like ChatGPT, have shown remarkable performance in tasks such as question answering and text generation. However, its potential in spam identification remains underexplored. To fill in the gap, this study attempts to evaluate ChatGPT's capabilities for spam identification in both English and Chinese email datasets. We employ ChatGPT for spam email detection using in-context learning, which requires a prompt instruction and a few demonstrations. We also investigate how the t
    

