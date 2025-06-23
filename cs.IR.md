# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [ALTO: An Efficient Network Orchestrator for Compound AI Systems](https://arxiv.org/abs/2403.04311) | ALTO是一个网络编排器，针对生成语言模型的优化机会，实现了高吞吐量和低延迟，同时解决了流式中间输出的两个新挑战：正确性和负载平衡。 |

# 详细

[^1]: ALTO：一种用于复合AI系统的高效网络编排器

    ALTO: An Efficient Network Orchestrator for Compound AI Systems

    [https://arxiv.org/abs/2403.04311](https://arxiv.org/abs/2403.04311)

    ALTO是一个网络编排器，针对生成语言模型的优化机会，实现了高吞吐量和低延迟，同时解决了流式中间输出的两个新挑战：正确性和负载平衡。

    

    我们提出了ALTO，一种用于有效为诸如语言模型管道之类的复合AI系统提供服务的网络编排器。ALTO通过利用生成语言模型特有的优化机会：流式中间输出，实现了高吞吐量和低延迟。由于语言模型逐个生成token的输出，ALTO在可能时暴露了在阶段之间流式传输中间输出的机会。我们强调了在跨分布式管道阶段实例之间流式传输中间数据时出现的两个新挑战：正确性和负载平衡。我们还提出了聚合感知路由接口和分布式提示感知调度以应对这些挑战的需求。我们在一个复杂的聊天机器人验证管道上展示了ALTO部分输出流式传输的影响，将吞吐量提高了最多3倍，同时将固定延迟目标设置为4秒/请求，还减少了尾延迟。

    arXiv:2403.04311v1 Announce Type: new  Abstract: We present ALTO, a network orchestrator for efficiently serving compound AI systems such as pipelines of language models. ALTO achieves high throughput and low latency by taking advantage of an optimization opportunity specific to generative language models: streaming intermediate outputs. As language models produce outputs token by token, ALTO exposes opportunities to stream intermediate outputs between stages when possible. We highlight two new challenges of correctness and load balancing which emerge when streaming intermediate data across distributed pipeline stage instances. We also motivate the need for an aggregation-aware routing interface and distributed prompt-aware scheduling to address these challenges. We demonstrate the impact of ALTO's partial output streaming on a complex chatbot verification pipeline, increasing throughput by up to 3x for a fixed latency target of 4 seconds / request while also reducing tail latency by 1
    

