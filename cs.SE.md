# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [NOVA: A Verification-Aware Agent Harness for Architecture Evolution in Industrial Recommender Systems](https://arxiv.org/abs/2606.27243) | NOVA 通过引入“架构梯度”这一受SGD启发的不可微更新信号，实现了对工业推荐系统架构演进的验证感知自动化，解决了现有方法仅关注代码可运行性而忽视架构有效性的问题。 |

# 详细

[^1]: NOVA：面向工业推荐系统架构演进的验证感知智能体框架

    NOVA: A Verification-Aware Agent Harness for Architecture Evolution in Industrial Recommender Systems

    [https://arxiv.org/abs/2606.27243](https://arxiv.org/abs/2606.27243)

    NOVA 通过引入“架构梯度”这一受SGD启发的不可微更新信号，实现了对工业推荐系统架构演进的验证感知自动化，解决了现有方法仅关注代码可运行性而忽视架构有效性的问题。

    

    arXiv:2606.27243v1 公告类型：新文 摘要：工业广告推荐模型通过架构演进持续改进。诸如RankMixer、TokenMixer-Large和MixFormer等升级表明，更好的结构仍然是质量和业务收益的关键来源。然而，在生产环境中开发此类升级需要大量专家投入且难以规模化。现有自动化方法存在不足：AutoML主要调优超参数，而有效收益往往需要在严格约束下进行跨模块更改；通用LLM编码智能体优化可运行代码，但可运行代码并不意味着有效的推荐架构。候选方案可能通过局部测试，但会导致静默故障从而降低性能。我们提出了NOVA，一种用于验证感知架构演进的层级感知智能体框架。NOVA使用架构梯度，这是一种受SGD启发的、不可微的更新信号，它聚合了先前的修改、验证诊断和指标反馈。

    arXiv:2606.27243v1 Announce Type: new  Abstract: Industrial advertising recommender models are continuously improved through architecture evolution. Upgrades such as RankMixer, TokenMixer-Large, and MixFormer show that better structures remain a key source of quality and business gains. Yet developing such upgrades in production is expert-intensive and difficult to scale. Existing automation is insufficient: AutoML mainly tunes hyper-parameters, while effective gains often require cross-module changes under strict constraints; generic LLM coding agents optimize for runnable code, but runnable code does not imply a valid recommender architecture. Candidates may pass local tests while causing silent failures that degrade performance.   We present NOVA, a level-aware agent harness for verification-aware architecture evolution. NOVA uses an architecture gradient, an SGD-inspired, non-differentiable update signal that aggregates prior modifications, verification diagnostics, metric feedback
    

