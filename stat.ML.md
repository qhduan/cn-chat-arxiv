# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Learning Dynamic Mechanisms in Unknown Environments: A Reinforcement Learning Approach](https://arxiv.org/abs/2202.12797) | 通过将无奖励在线强化学习引入到在线机制设计问题中，我们提出了能够在未知环境中学习动态VCG机制且具有上界为$\tilde{\mathcal{O}}(T^{2/3})$的遗憾保证的新颖学习算法。 |
| [^2] | [Flexible and efficient spatial extremes emulation via variational autoencoders.](http://arxiv.org/abs/2307.08079) | 本文提出了一种新的空间极端值模型，通过集成在变分自动编码器的结构中，可以灵活、高效地模拟具有非平稳相关性的极端事件。实验证明，在时间效率和性能上，相对于传统的贝叶斯推断和许多具有平稳相关性的空间极端值模型，我们的方法具有优势。 |
| [^3] | [Diffusion map particle systems for generative modeling.](http://arxiv.org/abs/2304.00200) | 本文提出一种新型扩散映射粒子系统(DMPS)，可以用于高效生成建模，实验表明在包含流形结构的合成数据集上取得了比其他方法更好的效果。 |

# 详细

[^1]: 在未知环境中学习动态机制：一种强化学习方法

    Learning Dynamic Mechanisms in Unknown Environments: A Reinforcement Learning Approach

    [https://arxiv.org/abs/2202.12797](https://arxiv.org/abs/2202.12797)

    通过将无奖励在线强化学习引入到在线机制设计问题中，我们提出了能够在未知环境中学习动态VCG机制且具有上界为$\tilde{\mathcal{O}}(T^{2/3})$的遗憾保证的新颖学习算法。

    

    动态机制设计研究了机制设计者在时变环境中应该如何在代理之间分配资源。我们考虑了一种问题，即代理根据未知的马尔可夫决策过程(MDP)与机制设计者互动，在这个过程中代理的奖励和机制设计者的状态根据一个带有未知奖励函数和转移核的情节MDP演化。我们关注在线设置下的线性函数近似，并提出了新颖的学习算法，在多轮互动中恢复动态Vickrey-Clarke-Grove(VCG)机制。我们方法的一个关键贡献是将无奖励在线强化学习(RL)结合进来，以帮助在丰富的策略空间中进行探索，从而估计动态VCG机制中的价格。我们展示了我们提出的方法的遗憾上界为$\tilde{\mathcal{O}}(T^{2/3})$，并进一步设计了一个下界，以展示我们方法的...

    arXiv:2202.12797v2 Announce Type: replace  Abstract: Dynamic mechanism design studies how mechanism designers should allocate resources among agents in a time-varying environment. We consider the problem where the agents interact with the mechanism designer according to an unknown Markov Decision Process (MDP), where agent rewards and the mechanism designer's state evolve according to an episodic MDP with unknown reward functions and transition kernels. We focus on the online setting with linear function approximation and propose novel learning algorithms to recover the dynamic Vickrey-Clarke-Grove (VCG) mechanism over multiple rounds of interaction. A key contribution of our approach is incorporating reward-free online Reinforcement Learning (RL) to aid exploration over a rich policy space to estimate prices in the dynamic VCG mechanism. We show that the regret of our proposed method is upper bounded by $\tilde{\mathcal{O}}(T^{2/3})$ and further devise a lower bound to show that our a
    
[^2]: 通过变分自动 编码器实现灵活高效的空间极端值模拟

    Flexible and efficient spatial extremes emulation via variational autoencoders. (arXiv:2307.08079v1 [stat.ML])

    [http://arxiv.org/abs/2307.08079](http://arxiv.org/abs/2307.08079)

    本文提出了一种新的空间极端值模型，通过集成在变分自动编码器的结构中，可以灵活、高效地模拟具有非平稳相关性的极端事件。实验证明，在时间效率和性能上，相对于传统的贝叶斯推断和许多具有平稳相关性的空间极端值模型，我们的方法具有优势。

    

    许多现实世界的过程具有复杂的尾依赖结构，这种结构无法使用传统的高斯过程来描述。更灵活的空间极端值模型， 如高斯尺度混合模型和单站点调节模型，具有吸引人的极端依赖性质，但往往难以拟合和模拟。本文中，我们提出了一种新的空间极端值模型，具有灵活和非平稳的相关性属性，并将其集成到变分自动编码器 (extVAE) 的编码-解码结构中。 extVAE 可以作为一个时空模拟器，对潜在的机制模型输出状态的分布进行建模，并产生具有与输入相同属性的输出，尤其是在尾部区域。通过广泛的模拟研究，我们证明我们的extVAE比传统的贝叶斯推断更高效，并且在具有 平稳相关性结构的许多空间极端值模型中表现 更好。

    Many real-world processes have complex tail dependence structures that cannot be characterized using classical Gaussian processes. More flexible spatial extremes models such as Gaussian scale mixtures and single-station conditioning models exhibit appealing extremal dependence properties but are often exceedingly prohibitive to fit and simulate from. In this paper, we develop a new spatial extremes model that has flexible and non-stationary dependence properties, and we integrate it in the encoding-decoding structure of a variational autoencoder (extVAE). The extVAE can be used as a spatio-temporal emulator that characterizes the distribution of potential mechanistic model output states and produces outputs that have the same properties as the inputs, especially in the tail. Through extensive simulation studies, we show that our extVAE is vastly more time-efficient than traditional Bayesian inference while also outperforming many spatial extremes models with a stationary dependence str
    
[^3]: 基于扩散映射的粒子系统用于生成模型

    Diffusion map particle systems for generative modeling. (arXiv:2304.00200v1 [stat.ML])

    [http://arxiv.org/abs/2304.00200](http://arxiv.org/abs/2304.00200)

    本文提出一种新型扩散映射粒子系统(DMPS)，可以用于高效生成建模，实验表明在包含流形结构的合成数据集上取得了比其他方法更好的效果。

    

    本文提出了一种新颖的扩散映射粒子系统(DMPS)，用于生成建模，该方法基于扩散映射和Laplacian调整的Wasserstein梯度下降（LAWGD）。扩散映射被用来从样本中近似Langevin扩散过程的生成器，从而学习潜在的数据生成流形。另一方面，LAWGD能够在合适的核函数选择下高效地从目标分布中抽样，我们在这里通过扩散映射计算生成器的谱逼近来构造核函数。数值实验表明，我们的方法在包括具有流形结构的合成数据集上优于其他方法。

    We propose a novel diffusion map particle system (DMPS) for generative modeling, based on diffusion maps and Laplacian-adjusted Wasserstein gradient descent (LAWGD). Diffusion maps are used to approximate the generator of the Langevin diffusion process from samples, and hence to learn the underlying data-generating manifold. On the other hand, LAWGD enables efficient sampling from the target distribution given a suitable choice of kernel, which we construct here via a spectral approximation of the generator, computed with diffusion maps. Numerical experiments show that our method outperforms others on synthetic datasets, including examples with manifold structure.
    

