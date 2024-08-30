# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Helmsman of the Masses? Evaluate the Opinion Leadership of Large Language Models in the Werewolf Game](https://arxiv.org/abs/2404.01602) | 本研究通过狼人游戏模拟平台评估了大语言模型的观点领导作用，并开发了两个新的评估指标。 |
| [^2] | [GEAR: An Efficient KV Cache Compression Recipefor Near-Lossless Generative Inference of LLM](https://arxiv.org/abs/2403.05527) | GEAR提出了一种高效的KV缓存压缩框架，实现几乎无损的高比率压缩，用于解决大型语言模型推断中因缓存需求增长而导致的记忆绑定问题和性能下降。 |
| [^3] | [TEncDM: Understanding the Properties of Diffusion Model in the Space of Language Model Encodings](https://arxiv.org/abs/2402.19097) | 通过在语言模型编码空间中训练模型，并使用基于Transformer的解码器以及自我调节，本文提出了名为TEncDM的文本编码扩散模型，在两个文本生成任务上展示了其优越性 |
| [^4] | [LLM Agents for Psychology: A Study on Gamified Assessments](https://arxiv.org/abs/2402.12326) | 本研究提出了PsychoGAT（心理游戏代理）以实现心理评估的通用游戏化，通过将强大的LLM代理纳入角色，将标准量表转化为个性化且具有吸引力的互动小说游戏。 |
| [^5] | [Exploring the Limitations of Graph Reasoning in Large Language Models](https://arxiv.org/abs/2402.01805) | 本文测试了5种不同的大型语言模型在图推理问题上的推理深度，并发现了LLMs的局限性、偏见和属性。我们发现LLMs对于节点遍历自由度的平均度数呈反向关系，k-shot提示对图推理任务有负面影响，并且LLMs存在积极的回应偏差，无法识别有效解的缺失。我们还提出了一种新的图推理提示技术。 |

# 详细

[^1]: 大语言模型在狼人游戏中的舵手？评估其观点引领作用

    Helmsman of the Masses? Evaluate the Opinion Leadership of Large Language Models in the Werewolf Game

    [https://arxiv.org/abs/2404.01602](https://arxiv.org/abs/2404.01602)

    本研究通过狼人游戏模拟平台评估了大语言模型的观点领导作用，并开发了两个新的评估指标。

    

    大语言模型（LLMs）在社交推理游戏中展现出令人难忘的战略行为。然而，LLM代理所展示的观点领导力的重要性被忽视了，而这对于多智能体和人工智能交互设置中的实际应用至关重要。在此研究中，我们利用狼人游戏作为模拟平台，评估LLMs的观点引领作用。该游戏中有警长角色，负责总结论据并推荐决策选项，因此可作为观点领袖的可信代理。我们开发了一个整合了警长角色的框架，并设计了两个基于观点领袖关键特征的新指标进行评估。第一个度量标准衡量观点领袖的可靠性，第二个评估...

    arXiv:2404.01602v1 Announce Type: cross  Abstract: Large language models (LLMs) have exhibited memorable strategic behaviors in social deductive games. However, the significance of opinion leadership exhibited by LLM-based agents has been overlooked, which is crucial for practical applications in multi-agent and human-AI interaction settings. Opinion leaders are individuals who have a noticeable impact on the beliefs and behaviors of others within a social group. In this work, we employ the Werewolf game as a simulation platform to assess the opinion leadership of LLMs. The game features the role of the Sheriff, tasked with summarizing arguments and recommending decision options, and therefore serves as a credible proxy for an opinion leader. We develop a framework integrating the Sheriff role and devise two novel metrics for evaluation based on the critical characteristics of opinion leaders. The first metric measures the reliability of the opinion leader, and the second assesses the 
    
[^2]: GEAR: 一种用于几乎无损生成推断大型语言模型的高效KV缓存压缩方案

    GEAR: An Efficient KV Cache Compression Recipefor Near-Lossless Generative Inference of LLM

    [https://arxiv.org/abs/2403.05527](https://arxiv.org/abs/2403.05527)

    GEAR提出了一种高效的KV缓存压缩框架，实现几乎无损的高比率压缩，用于解决大型语言模型推断中因缓存需求增长而导致的记忆绑定问题和性能下降。

    

    关键-值（KV）缓存已成为加快大型语言模型（LLMs）推断生成速度的事实标准。然而，随着序列长度增加而增长的缓存需求已将LLM推断转变为一个记忆绑定问题，显著地限制了系统吞吐量。现有方法依赖于丢弃不重要的标记或均匀量化所有条目。然而，这种方法往往会产生较高的近似误差来表示压缩后的矩阵。自回归解码过程进一步增加了每个步骤的误差，导致模型生成中的重大偏差和性能恶化。为了解决这一挑战，我们提出了GEAR，一种高效的KV缓存压缩框架，实现几乎无损的高压缩比。

    arXiv:2403.05527v1 Announce Type: cross  Abstract: Key-value (KV) caching has become the de-facto to accelerate generation speed for large language models (LLMs) inference. However, the growing cache demand with increasing sequence length has transformed LLM inference to be a memory bound problem, significantly constraining the system throughput. Existing methods rely on dropping unimportant tokens or quantizing all entries uniformly. Such methods, however, often incur high approximation errors to represent the compressed matrices. The autoregressive decoding process further compounds the error of each step, resulting in critical deviation in model generation and deterioration of performance. To tackle this challenge, we propose GEAR, an efficient KV cache compression framework that achieves near-lossless high-ratio compression. GEAR first applies quantization to majority of entries of similar magnitudes to ultra-low precision. It then employs a low rank matrix to approximate the quant
    
[^3]: TEncDM: 在语言模型编码空间中理解扩散模型的属性

    TEncDM: Understanding the Properties of Diffusion Model in the Space of Language Model Encodings

    [https://arxiv.org/abs/2402.19097](https://arxiv.org/abs/2402.19097)

    通过在语言模型编码空间中训练模型，并使用基于Transformer的解码器以及自我调节，本文提出了名为TEncDM的文本编码扩散模型，在两个文本生成任务上展示了其优越性

    

    受到扩散模型在各个领域取得成功的启发，许多研究论文提出了将其应用于文本数据的方法。尽管有这些努力，但没有一种方法能够达到大型语言模型的质量。本文对文本扩散模型的关键组件进行了全面分析，并介绍了一种名为Text Encoding Diffusion Model (TEncDM)的新方法。我们在语言模型编码空间中训练我们的模型，而不是通常使用的标记嵌入空间。此外，我们提出使用基于Transformer的解码器，利用上下文信息进行文本重构。我们还分析了自我调节，并发现这会增加模型输出的数量级，从而减少推理阶段的去噪步骤数量。在两个下游文本生成任务QQP和XSum上对TEncDM的评估表明其优越性。

    arXiv:2402.19097v1 Announce Type: new  Abstract: Drawing inspiration from the success of diffusion models in various domains, numerous research papers proposed methods for adapting them to text data. Despite these efforts, none of them has managed to achieve the quality of the large language models. In this paper, we conduct a comprehensive analysis of key components of the text diffusion models and introduce a novel approach named Text Encoding Diffusion Model (TEncDM). Instead of the commonly used token embedding space, we train our model in the space of the language model encodings. Additionally, we propose to use a Transformer-based decoder that utilizes contextual information for text reconstruction. We also analyse self-conditioning and find that it increases the magnitude of the model outputs, allowing the reduction of the number of denoising steps at the inference stage. Evaluation of TEncDM on two downstream text generation tasks, QQP and XSum, demonstrates its superiority ove
    
[^4]: 基于LLM的心理学智能代理：一项关于游戏化评估的研究

    LLM Agents for Psychology: A Study on Gamified Assessments

    [https://arxiv.org/abs/2402.12326](https://arxiv.org/abs/2402.12326)

    本研究提出了PsychoGAT（心理游戏代理）以实现心理评估的通用游戏化，通过将强大的LLM代理纳入角色，将标准量表转化为个性化且具有吸引力的互动小说游戏。

    

    心理测量对于精神健康、自我理解和个人发展至关重要。传统方法，如自我报告量表和心理学家访谈，常常面临参与度和可获得性方面的挑战。虽然已经探讨了基于游戏和LLM的工具来提高用户兴趣并自动化评估，但它们难以平衡参与度和普适性。在这项工作中，我们提出了PsychoGAT（心理游戏代理），以实现心理评估的通用游戏化。主要洞察是强大的LLM既可以充当熟练的心理学家，也可以是创新的游戏设计师。通过将LLM代理纳入指定角色并精心管理它们的互动，PsychoGAT可以将任何标准量表转化为个性化且具有吸引力的互动小说游戏。为验证所提出的方法，我们进行心理度量评估以评估其有效性，并使用人类

    arXiv:2402.12326v1 Announce Type: new  Abstract: Psychological measurement is essential for mental health, self-understanding, and personal development. Traditional methods, such as self-report scales and psychologist interviews, often face challenges with engagement and accessibility. While game-based and LLM-based tools have been explored to improve user interest and automate assessment, they struggle to balance engagement with generalizability. In this work, we propose PsychoGAT (Psychological Game AgenTs) to achieve a generic gamification of psychological assessment. The main insight is that powerful LLMs can function both as adept psychologists and innovative game designers. By incorporating LLM agents into designated roles and carefully managing their interactions, PsychoGAT can transform any standardized scales into personalized and engaging interactive fiction games. To validate the proposed method, we conduct psychometric evaluations to assess its effectiveness and employ huma
    
[^5]: 探索大型语言模型中图推理的局限性

    Exploring the Limitations of Graph Reasoning in Large Language Models

    [https://arxiv.org/abs/2402.01805](https://arxiv.org/abs/2402.01805)

    本文测试了5种不同的大型语言模型在图推理问题上的推理深度，并发现了LLMs的局限性、偏见和属性。我们发现LLMs对于节点遍历自由度的平均度数呈反向关系，k-shot提示对图推理任务有负面影响，并且LLMs存在积极的回应偏差，无法识别有效解的缺失。我们还提出了一种新的图推理提示技术。

    

    预训练的大型语言模型仅通过基于语言的提示就展示了各种类型的推理能力。然而，在本文中，我们通过图推理问题测试了5种不同的大型语言模型（GPT-4，GPT-3.5，Claude-2，Llama-2和Palm-2）的推理深度。特别地，我们设计了10个不同的图遍历问题，每个问题代表着逐步增加的复杂性水平。此外，我们通过对不同图大小以及不同形式的k-shot提示的设置分析了模型的性能。通过这个基准测试过程，我们凸显了LLMs的各种局限性、偏见和属性，比如与每个节点的遍历自由度的平均度数呈反向关系，k-shot提示对图推理任务的整体负面影响，以及积极的回应偏差导致LLMs无法识别有效解的缺失。最后，我们提出一种新的提示技术，专门用于图推理。

    Pretrained Large Language Models have demonstrated various types of reasoning capabilities through language-based prompts alone. However, in this paper, we test the depth of graph reasoning for 5 different LLMs (GPT-4, GPT-3.5, Claude-2, Llama-2 and Palm-2) through the problems of graph reasoning. In particular, we design 10 distinct problems of graph traversal, each representing increasing levels of complexity. Further, we analyze the performance of models across various settings such as varying sizes of graphs as well as different forms of k-shot prompting. We highlight various limitations, biases, and properties of LLMs through this benchmarking process, such as an inverse relation to the average degrees of freedom of traversal per node in graphs, the overall negative impact of k-shot prompting on graph reasoning tasks, and a positive response bias which prevents LLMs from identifying the absence of a valid solution. Finally, we propose a new prompting technique specially designed f
    

