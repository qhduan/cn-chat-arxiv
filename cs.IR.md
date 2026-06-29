# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [NOVA: A Verification-Aware Agent Harness for Architecture Evolution in Industrial Recommender Systems](https://arxiv.org/abs/2606.27243) | NOVA 通过引入“架构梯度”这一受SGD启发的不可微更新信号，实现了对工业推荐系统架构演进的验证感知自动化，解决了现有方法仅关注代码可运行性而忽视架构有效性的问题。 |
| [^2] | [AgentX: Towards Agent-Driven Self-Iteration of Industrial Recommender Systems](https://arxiv.org/abs/2606.26859) | AgentX是一个已部署的多智能体系统，通过自主生成、实施、评估和学习推荐实验，实现了工业推荐系统从依赖人工到自我迭代的范式转变，从而打破了创新瓶颈。 |

# 详细

[^1]: NOVA：面向工业推荐系统架构演进的验证感知智能体框架

    NOVA: A Verification-Aware Agent Harness for Architecture Evolution in Industrial Recommender Systems

    [https://arxiv.org/abs/2606.27243](https://arxiv.org/abs/2606.27243)

    NOVA 通过引入“架构梯度”这一受SGD启发的不可微更新信号，实现了对工业推荐系统架构演进的验证感知自动化，解决了现有方法仅关注代码可运行性而忽视架构有效性的问题。

    

    arXiv:2606.27243v1 公告类型：新文 摘要：工业广告推荐模型通过架构演进持续改进。诸如RankMixer、TokenMixer-Large和MixFormer等升级表明，更好的结构仍然是质量和业务收益的关键来源。然而，在生产环境中开发此类升级需要大量专家投入且难以规模化。现有自动化方法存在不足：AutoML主要调优超参数，而有效收益往往需要在严格约束下进行跨模块更改；通用LLM编码智能体优化可运行代码，但可运行代码并不意味着有效的推荐架构。候选方案可能通过局部测试，但会导致静默故障从而降低性能。我们提出了NOVA，一种用于验证感知架构演进的层级感知智能体框架。NOVA使用架构梯度，这是一种受SGD启发的、不可微的更新信号，它聚合了先前的修改、验证诊断和指标反馈。

    arXiv:2606.27243v1 Announce Type: new  Abstract: Industrial advertising recommender models are continuously improved through architecture evolution. Upgrades such as RankMixer, TokenMixer-Large, and MixFormer show that better structures remain a key source of quality and business gains. Yet developing such upgrades in production is expert-intensive and difficult to scale. Existing automation is insufficient: AutoML mainly tunes hyper-parameters, while effective gains often require cross-module changes under strict constraints; generic LLM coding agents optimize for runnable code, but runnable code does not imply a valid recommender architecture. Candidates may pass local tests while causing silent failures that degrade performance.   We present NOVA, a level-aware agent harness for verification-aware architecture evolution. NOVA uses an architecture gradient, an SGD-inspired, non-differentiable update signal that aggregates prior modifications, verification diagnostics, metric feedback
    
[^2]: AgentX：迈向工业推荐系统智能体驱动的自我迭代

    AgentX: Towards Agent-Driven Self-Iteration of Industrial Recommender Systems

    [https://arxiv.org/abs/2606.26859](https://arxiv.org/abs/2606.26859)

    AgentX是一个已部署的多智能体系统，通过自主生成、实施、评估和学习推荐实验，实现了工业推荐系统从依赖人工到自我迭代的范式转变，从而打破了创新瓶颈。

    

    arXiv:2606.26859v1 公告类型：新 摘要：推荐算法的迭代正从依赖工程师的手工过程向工业化研究循环转变，但这种转变仍被结构性执行瓶颈所阻碍：从想法到上线的周期仍然依赖人类工程师提出假设、修改生产代码、启动A/B实验并归因线上结果。因此，创新规模与人力成正比，而非与证据、计算资源和积累的实验知识复合增长。我们提出AgentX，一个已投入生产的、从根本上重构这一生产函数的多智能体系统。AgentX作为一个自我进化的开发引擎运作：它能自主生成、实现、评估并学习推荐实验，其规模和速度远超任何人工工作流所能维持的水平。该系统在一个闭环中编排四个紧密耦合的阶段。一个头脑风暴智能体从历史数据中综合证据。

    arXiv:2606.26859v1 Announce Type: new  Abstract: Recommendation algorithm iteration is moving from an artisanal, engineer-bound process toward an industrialized research loop, but this transition remains blocked by a structural execution bottleneck: the idea-to-launch cycle still depends on human engineers to generate hypotheses, modify production code, launch A/B experiments, and attribute online results. Innovation therefore scales linearly with headcount rather than compounding with evidence, compute, and accumulated experimental knowledge. We present AgentX, a production-deployed multi-agent system that fundamentally restructures this production function. AgentX operates as a self-evolving development engine: it autonomously generates, implements, evaluates, and learns from recommendation experiments at a scale and pace that no manual workflow can sustain.   The system orchestrates four tightly coupled stages in a closed loop. A Brainstorm Agent synthesizes evidence from historical
    

