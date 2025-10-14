# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [HAMUR: Hyper Adapter for Multi-Domain Recommendation.](http://arxiv.org/abs/2309.06217) | HAMUR是一种超级适配器用于处理多领域推荐的模型，它通过引入领域特定适配器和领域共享超网络来解决当前模型中的相互干扰和静态参数限制的问题。 |

# 详细

[^1]: HAMUR: 超级适配器用于多领域推荐

    HAMUR: Hyper Adapter for Multi-Domain Recommendation. (arXiv:2309.06217v1 [cs.IR])

    [http://arxiv.org/abs/2309.06217](http://arxiv.org/abs/2309.06217)

    HAMUR是一种超级适配器用于处理多领域推荐的模型，它通过引入领域特定适配器和领域共享超网络来解决当前模型中的相互干扰和静态参数限制的问题。

    

    多领域推荐 (MDR) 在近年来受到了显着关注，它利用多个领域的数据同时增强它们的性能。然而，当前的 MDR 模型面临两个限制。首先，大多数模型采用明确共享参数的方法，导致它们之间相互干扰。其次，由于不同领域之间的分布差异，现有方法中静态参数的使用限制了其适应多样领域的灵活性。为了解决这些挑战，我们提出了一种新颖的模型 Hyper Adapter for Multi-Domain Recommendation (HAMUR)。具体来说，HAMUR 包括两个组成部分：(1). 领域特定适配器，作为一个可无缝集成到各种现有的多领域骨干模型的插件模块设计，和 (2). 领域共享超网络，隐含地捕捉领域间的共享信息并动态生成参数。

    Multi-Domain Recommendation (MDR) has gained significant attention in recent years, which leverages data from multiple domains to enhance their performance concurrently.However, current MDR models are confronted with two limitations. Firstly, the majority of these models adopt an approach that explicitly shares parameters between domains, leading to mutual interference among them. Secondly, due to the distribution differences among domains, the utilization of static parameters in existing methods limits their flexibility to adapt to diverse domains. To address these challenges, we propose a novel model Hyper Adapter for Multi-Domain Recommendation (HAMUR). Specifically, HAMUR consists of two components: (1). Domain-specific adapter, designed as a pluggable module that can be seamlessly integrated into various existing multi-domain backbone models, and (2). Domain-shared hyper-network, which implicitly captures shared information among domains and dynamically generates the parameters fo
    

