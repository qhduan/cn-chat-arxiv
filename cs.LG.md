# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Thousands of AI Authors on the Future of AI.](http://arxiv.org/abs/2401.02843) | 数千位AI作者对未来AI的预测显示，到2028年，AI系统有50%的几率实现多个里程碑，包括自主构建全新的付款处理网站、创作一首与知名音乐家的新歌难以区分的歌曲，并自主下载和调整大型语言模型。同时，无需辅助的机器在各种任务上胜过人类的几率估计为10%到2047年为50%。 |
| [^2] | [Contrastive Difference Predictive Coding.](http://arxiv.org/abs/2310.20141) | 本文介绍了一种时间差异版本的对比预测编码，通过将不同时间序列数据的片段组合在一起，来减少学习预测未来事件所需的数据量。实验证明，与先前的方法相比，我们的方法在成功率上提高了2倍，并且对于随机环境有更好的适应能力。 |
| [^3] | [Graph-SCP: Accelerating Set Cover Problems with Graph Neural Networks.](http://arxiv.org/abs/2310.07979) | 图形-SCP是一种使用图神经网络加速集合覆盖问题的方法，通过学习识别包含解空间的较小子问题来提高优化求解器的性能，实验结果表明，图形-SCP能够将问题大小减少30-70%，和商业求解器相比加速高达25倍，并且能够在给定的最优性阈值下改进或实现100%的最优性。 |
| [^4] | [Towards Automated Urban Planning: When Generative and ChatGPT-like AI Meets Urban Planning.](http://arxiv.org/abs/2304.03892) | 本文探讨了城市规划与人工智能的交叉应用，重点是自动化用地配置，通过对抗学习、生成神经网络、深度编码器-解码器网络、对话式 AI 和地理空间和时间机器学习等技术，AI 可以为现代城市规划带来不少创新与贡献。 |
| [^5] | [Stochastic Interpolants: A Unifying Framework for Flows and Diffusions.](http://arxiv.org/abs/2303.08797) | 本文提出了一种统一的生成模型，该模型基于随机插值框架，可以实现流和扩散方法的统一。作者构建了一类广泛的连续时间随机过程，用于将两个任意的密度在有限时间内精确地连接。这种方法可以用于基于概率微分方程的确定性和随机生成模型的构建。 |

# 详细

[^1]: 数千位AI作者对未来AI的预测

    Thousands of AI Authors on the Future of AI. (arXiv:2401.02843v1 [cs.CY])

    [http://arxiv.org/abs/2401.02843](http://arxiv.org/abs/2401.02843)

    数千位AI作者对未来AI的预测显示，到2028年，AI系统有50%的几率实现多个里程碑，包括自主构建全新的付款处理网站、创作一首与知名音乐家的新歌难以区分的歌曲，并自主下载和调整大型语言模型。同时，无需辅助的机器在各种任务上胜过人类的几率估计为10%到2047年为50%。

    

    在迄今为止最大规模的调查中，2778名在顶级人工智能（AI）会议上发表过论文的研究人员对AI进展的速度、高级AI系统的性质和影响进行了预测。总体预测显示，到2028年，AI系统至少有50%的几率实现多个里程碑，包括自主构建一个全新的付款处理网站、创作一首可以与知名音乐家的新歌难以区分的歌曲，并自主下载和调整大型语言模型。如果科学持续不受干扰，2027年无需辅助的机器在各种任务上胜过人类的几率估计为10%，到2047年为50%。后者的估计比我们一年前进行的类似调查[Grace et al., 2022]提前了13年。然而，所有人类职业完全可自动化的几率预计要到2037年达到10%，到2116年才达到50%（与2022年调查中的2164年相比）。

    In the largest survey of its kind, 2,778 researchers who had published in top-tier artificial intelligence (AI) venues gave predictions on the pace of AI progress and the nature and impacts of advanced AI systems The aggregate forecasts give at least a 50% chance of AI systems achieving several milestones by 2028, including autonomously constructing a payment processing site from scratch, creating a song indistinguishable from a new song by a popular musician, and autonomously downloading and fine-tuning a large language model. If science continues undisrupted, the chance of unaided machines outperforming humans in every possible task was estimated at 10% by 2027, and 50% by 2047. The latter estimate is 13 years earlier than that reached in a similar survey we conducted only one year earlier [Grace et al., 2022]. However, the chance of all human occupations becoming fully automatable was forecast to reach 10% by 2037, and 50% as late as 2116 (compared to 2164 in the 2022 survey).  Most
    
[^2]: 对比差异性预测编码

    Contrastive Difference Predictive Coding. (arXiv:2310.20141v1 [cs.LG])

    [http://arxiv.org/abs/2310.20141](http://arxiv.org/abs/2310.20141)

    本文介绍了一种时间差异版本的对比预测编码，通过将不同时间序列数据的片段组合在一起，来减少学习预测未来事件所需的数据量。实验证明，与先前的方法相比，我们的方法在成功率上提高了2倍，并且对于随机环境有更好的适应能力。

    

    预测和推理未来是许多时间序列问题的核心。例如，目标导向的强化学习可以被看作是学习表示以预测未来可能访问的状态。虽然先前的方法已经使用对比性预测编码来建模时间序列数据，但学习编码长期依赖通常需要大量的数据。在本文中，我们引入了一种时间差异版本的对比预测编码，将不同时间序列数据的片段组合在一起，以减少学习未来事件预测所需的数据量。我们将这种表示学习方法应用于导出目标导向的强化学习的离策略算法。实验证明，与先前的强化学习方法相比，我们的方法在成功率上实现了中位数提高2倍，并且可以更好地应对随机环境。在表格设置中，我们展示了我们的方法约为20倍。

    Predicting and reasoning about the future lie at the heart of many time-series questions. For example, goal-conditioned reinforcement learning can be viewed as learning representations to predict which states are likely to be visited in the future. While prior methods have used contrastive predictive coding to model time series data, learning representations that encode long-term dependencies usually requires large amounts of data. In this paper, we introduce a temporal difference version of contrastive predictive coding that stitches together pieces of different time series data to decrease the amount of data required to learn predictions of future events. We apply this representation learning method to derive an off-policy algorithm for goal-conditioned RL. Experiments demonstrate that, compared with prior RL methods, ours achieves $2 \times$ median improvement in success rates and can better cope with stochastic environments. In tabular settings, we show that our method is about $20
    
[^3]: 图形-SCP: 用图神经网络加速集合覆盖问题

    Graph-SCP: Accelerating Set Cover Problems with Graph Neural Networks. (arXiv:2310.07979v1 [cs.LG])

    [http://arxiv.org/abs/2310.07979](http://arxiv.org/abs/2310.07979)

    图形-SCP是一种使用图神经网络加速集合覆盖问题的方法，通过学习识别包含解空间的较小子问题来提高优化求解器的性能，实验结果表明，图形-SCP能够将问题大小减少30-70%，和商业求解器相比加速高达25倍，并且能够在给定的最优性阈值下改进或实现100%的最优性。

    

    机器学习方法越来越多地用于加速组合优化问题。我们特别关注集合覆盖问题（SCP），提出了一种名为图形-SCP的图神经网络方法，可以通过学习识别包含解空间的大大较小的子问题来增强现有的优化求解器。我们在具有不同问题特征和复杂度的合成加权和非加权SCP实例上评估了图形-SCP的性能，并在OR Library的实例上进行了评估，这是SCP的一个经典基准。我们展示了图形-SCP将问题大小减少了30-70%，并且相对于商业求解器（Gurobi）实现了高达25倍的运行时间加速。在给定所需的最优性阈值的情况下，图形-SCP将改进或甚至实现100%的最优性。这与快速贪婪解决方案形成了对比，后者在保证多项式运行时间的同时明显损害了解决方案的质量。图形-SCP可以推广到更大的问题规模。

    Machine learning (ML) approaches are increasingly being used to accelerate combinatorial optimization (CO) problems. We look specifically at the Set Cover Problem (SCP) and propose Graph-SCP, a graph neural network method that can augment existing optimization solvers by learning to identify a much smaller sub-problem that contains the solution space. We evaluate the performance of Graph-SCP on synthetic weighted and unweighted SCP instances with diverse problem characteristics and complexities, and on instances from the OR Library, a canonical benchmark for SCP. We show that Graph-SCP reduces the problem size by 30-70% and achieves run time speedups up to~25x when compared to commercial solvers (Gurobi). Given a desired optimality threshold, Graph-SCP will improve upon it or even achieve 100% optimality. This is in contrast to fast greedy solutions that significantly compromise solution quality to achieve guaranteed polynomial run time. Graph-SCP can generalize to larger problem sizes
    
[^4]: 自动化城市规划：生成式和聊天式 AI 相结合的城市规划探索

    Towards Automated Urban Planning: When Generative and ChatGPT-like AI Meets Urban Planning. (arXiv:2304.03892v1 [cs.AI])

    [http://arxiv.org/abs/2304.03892](http://arxiv.org/abs/2304.03892)

    本文探讨了城市规划与人工智能的交叉应用，重点是自动化用地配置，通过对抗学习、生成神经网络、深度编码器-解码器网络、对话式 AI 和地理空间和时间机器学习等技术，AI 可以为现代城市规划带来不少创新与贡献。

    

    城市规划领域和人工智能领域曾经是独立发展的，但现在两个领域开始交叉汇合，互相借鉴和受益。本文介绍了城市规划从可持续性、生活、经济、灾害和环境等方面的重要性，回顾了城市规划的基本概念，并将这些概念与机器学习的关键开放问题联系起来，包括对抗学习、生成神经网络、深度编码器-解码器网络、对话式 AI 以及地理空间和时间机器学习等，评估了 AI 如何为现代城市规划做出贡献。因此，一个核心问题是自动化用地配置，即从周围的地理空间、人类移动、社交媒体、环境和经济活动中为目标区域生成土地用途和建筑配置。最后，本文勾画了集成 AI 和城市规划面临的一些挑战和潜在解决方案。

    The two fields of urban planning and artificial intelligence (AI) arose and developed separately. However, there is now cross-pollination and increasing interest in both fields to benefit from the advances of the other. In the present paper, we introduce the importance of urban planning from the sustainability, living, economic, disaster, and environmental perspectives. We review the fundamental concepts of urban planning and relate these concepts to crucial open problems of machine learning, including adversarial learning, generative neural networks, deep encoder-decoder networks, conversational AI, and geospatial and temporal machine learning, thereby assaying how AI can contribute to modern urban planning. Thus, a central problem is automated land-use configuration, which is formulated as the generation of land uses and building configuration for a target area from surrounding geospatial, human mobility, social media, environment, and economic activities. Finally, we delineate some 
    
[^5]: 随机插值：流和扩散的统一框架

    Stochastic Interpolants: A Unifying Framework for Flows and Diffusions. (arXiv:2303.08797v1 [cs.LG])

    [http://arxiv.org/abs/2303.08797](http://arxiv.org/abs/2303.08797)

    本文提出了一种统一的生成模型，该模型基于随机插值框架，可以实现流和扩散方法的统一。作者构建了一类广泛的连续时间随机过程，用于将两个任意的密度在有限时间内精确地连接。这种方法可以用于基于概率微分方程的确定性和随机生成模型的构建。

    

    我们介绍了一类建立在随机插值框架上的生成模型，该框架是基于Albergo＆Vanden-Eijnden（2023）提出的，在流和扩散方法上实现统一，我们首先展示了如何构建一类广泛的连续时间随机过程，其时间依赖的概率密度函数在有限时间内精确地连接两个任意的密度。这些“随机插值器”是通过将来自两个密度的数据与其他潜在变量相结合构建的，并且构造的具体细节可以灵活地塑造导致的时间依赖密度。然后我们展示了随机插值器的时间依赖密度满足一阶输运方程以及一系列具有可调扩散的正向和反向Fokker-Planck方程族; 在考虑单个样本的时间演化时，这个观点立即导致了基于概率微分方程的确定性和随机生成模型。

    We introduce a class of generative models based on the stochastic interpolant framework proposed in Albergo & Vanden-Eijnden (2023) that unifies flow-based and diffusion-based methods. We first show how to construct a broad class of continuous-time stochastic processes whose time-dependent probability density function bridges two arbitrary densities exactly in finite time. These `stochastic interpolants' are built by combining data from the two densities with an additional latent variable, and the specific details of the construction can be leveraged to shape the resulting time-dependent density in a flexible way. We then show that the time-dependent density of the stochastic interpolant satisfies a first-order transport equation as well as a family of forward and backward Fokker-Planck equations with tunable diffusion; upon consideration of the time evolution of an individual sample, this viewpoint immediately leads to both deterministic and stochastic generative models based on proba
    

