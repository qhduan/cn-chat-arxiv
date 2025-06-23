# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [ALTO: An Efficient Network Orchestrator for Compound AI Systems](https://arxiv.org/abs/2403.04311) | ALTO是一个网络编排器，针对生成语言模型的优化机会，实现了高吞吐量和低延迟，同时解决了流式中间输出的两个新挑战：正确性和负载平衡。 |
| [^2] | [AQA-Bench: An Interactive Benchmark for Evaluating LLMs' Sequential Reasoning Ability](https://arxiv.org/abs/2402.09404) | AQA-Bench是一个交互式基准测试，用于评估大型语言模型在算法上下文中的顺序推理能力。研究发现闭源模型表现出更强的顺序推理能力，显著优于开源模型。 |
| [^3] | [Supercharging academic writing with generative AI: framework, techniques, and caveats.](http://arxiv.org/abs/2310.17143) | 这篇论文介绍了使用生成型人工智能（AI）提高学术写作质量和效率的原则和方法，包括一个人机协作框架、有效的提示技术和两阶段模型，旨在实现认知卸载和想象刺激的AI辅助写作。 |
| [^4] | [Voices of Her: Analyzing Gender Differences in the AI Publication World.](http://arxiv.org/abs/2305.14597) | 通过对AI学术界的78K研究人员的分析，研究发现女性第一作者的论文具有不同的语言风格，例如更长的文本、更多的正面情感词汇和更引人注目的标题；在AI论文的合著中存在很大的性别同质性。我们鼓励未来实现更多的性别平等和多样性。 |

# 详细

[^1]: ALTO：一种用于复合AI系统的高效网络编排器

    ALTO: An Efficient Network Orchestrator for Compound AI Systems

    [https://arxiv.org/abs/2403.04311](https://arxiv.org/abs/2403.04311)

    ALTO是一个网络编排器，针对生成语言模型的优化机会，实现了高吞吐量和低延迟，同时解决了流式中间输出的两个新挑战：正确性和负载平衡。

    

    我们提出了ALTO，一种用于有效为诸如语言模型管道之类的复合AI系统提供服务的网络编排器。ALTO通过利用生成语言模型特有的优化机会：流式中间输出，实现了高吞吐量和低延迟。由于语言模型逐个生成token的输出，ALTO在可能时暴露了在阶段之间流式传输中间输出的机会。我们强调了在跨分布式管道阶段实例之间流式传输中间数据时出现的两个新挑战：正确性和负载平衡。我们还提出了聚合感知路由接口和分布式提示感知调度以应对这些挑战的需求。我们在一个复杂的聊天机器人验证管道上展示了ALTO部分输出流式传输的影响，将吞吐量提高了最多3倍，同时将固定延迟目标设置为4秒/请求，还减少了尾延迟。

    arXiv:2403.04311v1 Announce Type: new  Abstract: We present ALTO, a network orchestrator for efficiently serving compound AI systems such as pipelines of language models. ALTO achieves high throughput and low latency by taking advantage of an optimization opportunity specific to generative language models: streaming intermediate outputs. As language models produce outputs token by token, ALTO exposes opportunities to stream intermediate outputs between stages when possible. We highlight two new challenges of correctness and load balancing which emerge when streaming intermediate data across distributed pipeline stage instances. We also motivate the need for an aggregation-aware routing interface and distributed prompt-aware scheduling to address these challenges. We demonstrate the impact of ALTO's partial output streaming on a complex chatbot verification pipeline, increasing throughput by up to 3x for a fixed latency target of 4 seconds / request while also reducing tail latency by 1
    
[^2]: AQA-Bench：评估LLM顺序推理能力的交互式基准测试

    AQA-Bench: An Interactive Benchmark for Evaluating LLMs' Sequential Reasoning Ability

    [https://arxiv.org/abs/2402.09404](https://arxiv.org/abs/2402.09404)

    AQA-Bench是一个交互式基准测试，用于评估大型语言模型在算法上下文中的顺序推理能力。研究发现闭源模型表现出更强的顺序推理能力，显著优于开源模型。

    

    该论文介绍了一种新的基准测试AQA-Bench，用于评估大型语言模型（LLMs）在算法上下文中，如深度优先搜索（DFS）等的顺序推理能力。我们评估基准测试的关键特点在于其交互式评估协议-例如，在DFS中，每个节点的可用连接边取决于模型对该节点的遍历，因此需要LLM有效地记住已访问节点并策划后续移动的能力。我们使用三种不同的算法构建了AQA-Bench，分别是二分搜索，深度优先搜索和广度优先搜索，并评估了12种不同的LLMs的顺序推理能力。我们的调查揭示了一些有趣的发现：（1）类似GPT-4和Gemini等闭源模型通常显示出强大的顺序推理能力，明显优于开源LLMs。（2）天真地提供互操作性

    arXiv:2402.09404v1 Announce Type: cross Abstract: This paper introduces AQA-Bench, a novel benchmark to assess the sequential reasoning capabilities of large language models (LLMs) in algorithmic contexts, such as depth-first search (DFS). The key feature of our evaluation benchmark lies in its interactive evaluation protocol -- for example, in DFS, the availability of each node's connected edge is contingent upon the model's traversal to that node, thereby necessitating the LLM's ability to effectively remember visited nodes and strategize subsequent moves. We comprehensively build AQA-Bench with three different algorithms, namely binary search, depth-first search, and breadth-first search, and to evaluate the sequential reasoning ability of 12 different LLMs. Our investigations reveal several interesting findings: (1) Closed-source models like GPT-4 and Gemini generally show strong sequential reasoning ability, significantly outperforming open-source LLMs. (2) Naively providing inter
    
[^3]: 使用生成型人工智能推动学术写作：框架、技术和注意事项

    Supercharging academic writing with generative AI: framework, techniques, and caveats. (arXiv:2310.17143v1 [cs.CY])

    [http://arxiv.org/abs/2310.17143](http://arxiv.org/abs/2310.17143)

    这篇论文介绍了使用生成型人工智能（AI）提高学术写作质量和效率的原则和方法，包括一个人机协作框架、有效的提示技术和两阶段模型，旨在实现认知卸载和想象刺激的AI辅助写作。

    

    学术写作是研究项目中不可或缺但费时费力的部分。本文介绍了使用生成型人工智能（AI）特别是大型语言模型（LLMs）提高学术写作质量和效率的原则和方法。我们提出了一个人机协作框架，详细阐述了AI在写作中的理论基础（为什么）、过程（如何）和性质（什么）。该框架指出了短期和长期参与AI写作的原因及其基本机制（如认知卸载和想象刺激）。它揭示了AI在整个写作过程中的作用，通过一个人机协作写作的两阶段模型和写作辅助类型和级别的模型表示了AI在写作中的帮助方式。基于该框架，我们描述了在写作常规中整合AI的有效提示技术（大纲、起草和编辑）。

    Academic writing is an indispensable yet laborious part of the research enterprise. This Perspective maps out principles and methods for using generative artificial intelligence (AI), specifically large language models (LLMs), to elevate the quality and efficiency of academic writing. We introduce a human-AI collaborative framework that delineates the rationale (why), process (how), and nature (what) of AI engagement in writing. The framework pinpoints both short-term and long-term reasons for engagement and their underlying mechanisms (e.g., cognitive offloading and imaginative stimulation). It reveals the role of AI throughout the writing process, conceptualized through a two-stage model for human-AI collaborative writing, and the nature of AI assistance in writing, represented through a model of writing-assistance types and levels. Building on this framework, we describe effective prompting techniques for incorporating AI into the writing routine (outlining, drafting, and editing) a
    
[^4]: 她们的声音：分析人工智能出版领域的性别差异

    Voices of Her: Analyzing Gender Differences in the AI Publication World. (arXiv:2305.14597v1 [cs.CL])

    [http://arxiv.org/abs/2305.14597](http://arxiv.org/abs/2305.14597)

    通过对AI学术界的78K研究人员的分析，研究发现女性第一作者的论文具有不同的语言风格，例如更长的文本、更多的正面情感词汇和更引人注目的标题；在AI论文的合著中存在很大的性别同质性。我们鼓励未来实现更多的性别平等和多样性。

    

    虽然先前的研究已经分析了学术界中的性别偏见，但是我们仍然缺乏一个全面的人工智能社区性别差异的分析，涵盖各种主题和不同的发展趋势。我们使用AI Scholar数据集中的78K位AI领域的研究人员，发现了一些性别差异：（1）虽然女性研究人员的总引用次数比男性少，但这种引用差异并不适用于所有学术年龄组；（2）在AI论文的合著中存在很大的性别同质性；（3）女性第一作者的论文显示出不同的语言风格，例如更长的文本、更多的正面情感词汇和更引人注目的标题。我们的分析为我们的AI社区现有的人口统计趋势提供了一个窗口，并鼓励在未来实现更多的性别平等和多样性。我们的代码和数据可在https://github.com/causalNLP/ai-scholar-gender找到。

    While several previous studies have analyzed gender bias in research, we are still missing a comprehensive analysis of gender differences in the AI community, covering diverse topics and different development trends. Using the AI Scholar dataset of 78K researchers in the field of AI, we identify several gender differences: (1) Although female researchers tend to have fewer overall citations than males, this citation difference does not hold for all academic-age groups; (2) There exist large gender homophily in co-authorship on AI papers; (3) Female first-authored papers show distinct linguistic styles, such as longer text, more positive emotion words, and more catchy titles than male first-authored papers. Our analysis provides a window into the current demographic trends in our AI community, and encourages more gender equality and diversity in the future. Our code and data are at https://github.com/causalNLP/ai-scholar-gender.
    

