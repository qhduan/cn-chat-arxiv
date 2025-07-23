# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [ALTO: An Efficient Network Orchestrator for Compound AI Systems](https://arxiv.org/abs/2403.04311) | ALTO是一个网络编排器，针对生成语言模型的优化机会，实现了高吞吐量和低延迟，同时解决了流式中间输出的两个新挑战：正确性和负载平衡。 |
| [^2] | [Prioritizing Safeguarding Over Autonomy: Risks of LLM Agents for Science](https://arxiv.org/abs/2402.04247) | 本文探讨了科学领域中基于LLM的智能机器人的漏洞与风险，并强调了对安全措施的重要性。 |

# 详细

[^1]: ALTO：一种用于复合AI系统的高效网络编排器

    ALTO: An Efficient Network Orchestrator for Compound AI Systems

    [https://arxiv.org/abs/2403.04311](https://arxiv.org/abs/2403.04311)

    ALTO是一个网络编排器，针对生成语言模型的优化机会，实现了高吞吐量和低延迟，同时解决了流式中间输出的两个新挑战：正确性和负载平衡。

    

    我们提出了ALTO，一种用于有效为诸如语言模型管道之类的复合AI系统提供服务的网络编排器。ALTO通过利用生成语言模型特有的优化机会：流式中间输出，实现了高吞吐量和低延迟。由于语言模型逐个生成token的输出，ALTO在可能时暴露了在阶段之间流式传输中间输出的机会。我们强调了在跨分布式管道阶段实例之间流式传输中间数据时出现的两个新挑战：正确性和负载平衡。我们还提出了聚合感知路由接口和分布式提示感知调度以应对这些挑战的需求。我们在一个复杂的聊天机器人验证管道上展示了ALTO部分输出流式传输的影响，将吞吐量提高了最多3倍，同时将固定延迟目标设置为4秒/请求，还减少了尾延迟。

    arXiv:2403.04311v1 Announce Type: new  Abstract: We present ALTO, a network orchestrator for efficiently serving compound AI systems such as pipelines of language models. ALTO achieves high throughput and low latency by taking advantage of an optimization opportunity specific to generative language models: streaming intermediate outputs. As language models produce outputs token by token, ALTO exposes opportunities to stream intermediate outputs between stages when possible. We highlight two new challenges of correctness and load balancing which emerge when streaming intermediate data across distributed pipeline stage instances. We also motivate the need for an aggregation-aware routing interface and distributed prompt-aware scheduling to address these challenges. We demonstrate the impact of ALTO's partial output streaming on a complex chatbot verification pipeline, increasing throughput by up to 3x for a fixed latency target of 4 seconds / request while also reducing tail latency by 1
    
[^2]: 优先安全保障而非自治：科学中LLM智能机器人的风险

    Prioritizing Safeguarding Over Autonomy: Risks of LLM Agents for Science

    [https://arxiv.org/abs/2402.04247](https://arxiv.org/abs/2402.04247)

    本文探讨了科学领域中基于LLM的智能机器人的漏洞与风险，并强调了对安全措施的重要性。

    

    由大型语言模型（LLMs）驱动的智能机器人在各个学科中自主进行实验和促进科学发现方面展示了巨大的前景。尽管它们的能力非常有前途，但也引入了一些新的漏洞，需要仔细考虑安全性。然而，文献中存在显著的空白，尚未对这些漏洞进行全面探讨。本文通过对科学领域中基于LLM的机器人的漏洞进行深入研究，揭示了它们误用可能带来的潜在风险，并强调了对安全措施的需求，填补了这一空白。我们首先全面概述了科学LLM机器人固有的潜在风险，考虑了用户意图、特定的科学领域以及它们对外部环境可能造成的影响。然后，我们深入探讨了这些漏洞的起源和提供的解决方案。

    Intelligent agents powered by large language models (LLMs) have demonstrated substantial promise in autonomously conducting experiments and facilitating scientific discoveries across various disciplines. While their capabilities are promising, they also introduce novel vulnerabilities that demand careful consideration for safety. However, there exists a notable gap in the literature, as there has been no comprehensive exploration of these vulnerabilities. This position paper fills this gap by conducting a thorough examination of vulnerabilities in LLM-based agents within scientific domains, shedding light on potential risks associated with their misuse and emphasizing the need for safety measures. We begin by providing a comprehensive overview of the potential risks inherent to scientific LLM agents, taking into account user intent, the specific scientific domain, and their potential impact on the external environment. Then, we delve into the origins of these vulnerabilities and provid
    

