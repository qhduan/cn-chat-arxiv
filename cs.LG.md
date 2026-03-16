# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [From Activation to Initialization: Scaling Insights for Optimizing Neural Fields](https://arxiv.org/abs/2403.19205) | 本文研究了神经场的初始化和激活之间的关系，强调了网络初始化、架构选择和优化过程之间的深层联系，为神经场的优化提供了理论基础。 |
| [^2] | [Value-Aided Conditional Supervised Learning for Offline RL](https://arxiv.org/abs/2402.02017) | 该论文提出了一种称为价值增强的条件监督学习方法，通过将RCSL的稳定性与基于价值的方法的连接能力相结合，动态地根据轨迹回报将价值帮助注入损失函数中。实验证明，该方法不仅优于现有方法，而且在各种离线强化学习任务中实现了最高的轨迹回报，推动了离线强化学习的发展。 |
| [^3] | [Sampling and Uniqueness Sets in Graphon Signal Processing.](http://arxiv.org/abs/2401.06279) | 本论文研究了大规模图谱族上的采样集特性，并将“可移除集合”和“唯一性集合”的概念推广到了图谱信号处理中。通过利用图谱表示法，可以比较具有不同节点数和边数以及不同节点标记的图谱之间的采样集，并证明了具有相同图谱表示的采样集序列的收敛性。 |
| [^4] | [Diffusion Variational Inference: Diffusion Models as Expressive Variational Posteriors.](http://arxiv.org/abs/2401.02739) | 本文提出了去噪扩散变分推断（DDVI）算法，该算法使用扩散模型作为表达性变分后验，并通过反转加噪过程在潜空间中进行扩散。该方法易于实现，兼容黑盒变分推断，并在深度潜变量模型中的任务中表现优异。 |
| [^5] | [Superiority of Softmax: Unveiling the Performance Edge Over Linear Attention.](http://arxiv.org/abs/2310.11685) | 通过对softmax和线性注意力机制进行全面比较分析，本论文揭示了softmax注意力在大多数情况下优于线性注意力的潜在原因。 |
| [^6] | [Partially Observable Multi-agent RL with (Quasi-)Efficiency: The Blessing of Information Sharing.](http://arxiv.org/abs/2308.08705) | 本文研究了部分可观测随机博弈的可证明多Agent强化学习。通过信息共享和观测可能性假设，提出了构建近似模型以实现准效率的方法。 |
| [^7] | [Tight Non-asymptotic Inference via Sub-Gaussian Intrinsic Moment Norm.](http://arxiv.org/abs/2303.07287) | 本文提出了一种通过最大化一系列归一化矩来使用子高斯内在矩范实现紧凑的非渐进推断的方法，该方法可以导致更紧的Hoeffding子高斯浓度不等式，并且可以通过子高斯图检查具有有限样本大小的子高斯数据。 |
| [^8] | [Data-Driven Influence Functions for Optimization-Based Causal Inference.](http://arxiv.org/abs/2208.13701) | 本文提出了一种利用有限差分逼近统计泛函Gateaux导数的构造算法，并研究了从数据中进行概率分布估计的情况下的Gateaux导数估计。研究结果为因果推断和动态治疗方案等问题提供了解决方案。 |

# 详细

[^1]: 从激活到初始化：优化神经场的扩展见解

    From Activation to Initialization: Scaling Insights for Optimizing Neural Fields

    [https://arxiv.org/abs/2403.19205](https://arxiv.org/abs/2403.19205)

    本文研究了神经场的初始化和激活之间的关系，强调了网络初始化、架构选择和优化过程之间的深层联系，为神经场的优化提供了理论基础。

    

    在计算机视觉领域，神经场作为一种当代工具，利用神经网络进行信号表示而备受重视。尽管已经在使这些网络适应解决各种问题方面取得了显著进展，但该领域仍然缺乏一个全面的理论框架。本文旨在填补这一空白，深入探讨初始化和激活之间错综复杂的相互作用，为神经场的强化优化提供基础。我们的理论见解揭示了网络初始化、架构选择和优化过程之间的深层联系，强调在设计尖端神经场时需要采用全面的方法。

    arXiv:2403.19205v1 Announce Type: cross  Abstract: In the realm of computer vision, Neural Fields have gained prominence as a contemporary tool harnessing neural networks for signal representation. Despite the remarkable progress in adapting these networks to solve a variety of problems, the field still lacks a comprehensive theoretical framework. This article aims to address this gap by delving into the intricate interplay between initialization and activation, providing a foundational basis for the robust optimization of Neural Fields. Our theoretical insights reveal a deep-seated connection among network initialization, architectural choices, and the optimization process, emphasizing the need for a holistic approach when designing cutting-edge Neural Fields.
    
[^2]: 无需奖励的条件监督学习在离线强化学习中的价值增强

    Value-Aided Conditional Supervised Learning for Offline RL

    [https://arxiv.org/abs/2402.02017](https://arxiv.org/abs/2402.02017)

    该论文提出了一种称为价值增强的条件监督学习方法，通过将RCSL的稳定性与基于价值的方法的连接能力相结合，动态地根据轨迹回报将价值帮助注入损失函数中。实验证明，该方法不仅优于现有方法，而且在各种离线强化学习任务中实现了最高的轨迹回报，推动了离线强化学习的发展。

    

    离线强化学习通过基于回报的条件监督学习（RCSL）和基于价值的方法取得了显著进展，但每种方法都存在一些实际挑战。为了解决这些挑战，我们提出了价值增强的条件监督学习（VCS）方法，该方法将RCSL的稳定性与基于价值的方法的连接能力有效地结合在一起。通过神经切线核分析，VCS可以动态地根据轨迹回报将价值帮助注入RCSL的损失函数中，以区分价值函数可能无法实现稳定连接的实例。我们的实证研究表明，VCS不仅显著优于RCSL和基于价值的方法，而且在各种离线强化学习基准测试中始终实现了或经常超过最高的轨迹回报。这一突破为离线强化学习开辟了新的道路，推动了可实现的极限，并促进了进一步的创新。

    Offline reinforcement learning (RL) has seen notable advancements through return-conditioned supervised learning (RCSL) and value-based methods, yet each approach comes with its own set of practical challenges. Addressing these, we propose Value-Aided Conditional Supervised Learning (VCS), a method that effectively synergizes the stability of RCSL with the stitching ability of value-based methods. Based on the Neural Tangent Kernel analysis to discern instances where value function may not lead to stable stitching, VCS injects the value aid into the RCSL's loss function dynamically according to the trajectory return. Our empirical studies reveal that VCS not only significantly outperforms both RCSL and value-based methods but also consistently achieves, or often surpasses, the highest trajectory returns across diverse offline RL benchmarks. This breakthrough in VCS paves new paths in offline RL, pushing the limits of what can be achieved and fostering further innovations.
    
[^3]: 图谱信号处理中的采样和唯一性集

    Sampling and Uniqueness Sets in Graphon Signal Processing. (arXiv:2401.06279v1 [cs.LG])

    [http://arxiv.org/abs/2401.06279](http://arxiv.org/abs/2401.06279)

    本论文研究了大规模图谱族上的采样集特性，并将“可移除集合”和“唯一性集合”的概念推广到了图谱信号处理中。通过利用图谱表示法，可以比较具有不同节点数和边数以及不同节点标记的图谱之间的采样集，并证明了具有相同图谱表示的采样集序列的收敛性。

    

    在这项工作中，我们通过利用图谱和图极限的理论，研究了大规模图谱族上采样集的特性。为此，我们将“可移除集合”和“唯一性集合”的概念扩展到了图谱信号领域，这些概念最初是用于分析图谱上的信号的。我们给出了$\Lambda-$可移除集合的正式定义，并在得到从图谱中一个给定$\Lambda-$可移除集合的补集中的样本时，给出了一个频带有限的图谱信号可以以唯一方式表示的条件。通过利用这些结果，我们证明了图谱表示法可以作为一种共同的框架来比较具有不同节点数和边数以及不同节点标记的图谱之间的采样集。此外，对于收敛到一个图谱的图序列，我们还证明了具有相同$[0,1]$中图谱表示的采样集序列也是收敛的。我们利用这种收敛性。

    In this work, we study the properties of sampling sets on families of large graphs by leveraging the theory of graphons and graph limits. To this end, we extend to graphon signals the notion of removable and uniqueness sets, which was developed originally for the analysis of signals on graphs. We state the formal definition of a $\Lambda-$removable set and conditions under which a bandlimited graphon signal can be represented in a unique way when its samples are obtained from the complement of a given $\Lambda-$removable set in the graphon. By leveraging such results we show that graphon representations of graphs and graph signals can be used as a common framework to compare sampling sets between graphs with different numbers of nodes and edges, and different node labelings. Additionally, given a sequence of graphs that converges to a graphon, we show that the sequences of sampling sets whose graphon representation is identical in $[0,1]$ are convergent as well. We exploit the converge
    
[^4]: 扩散变分推断：扩散模型作为表达性变分后验

    Diffusion Variational Inference: Diffusion Models as Expressive Variational Posteriors. (arXiv:2401.02739v1 [cs.LG])

    [http://arxiv.org/abs/2401.02739](http://arxiv.org/abs/2401.02739)

    本文提出了去噪扩散变分推断（DDVI）算法，该算法使用扩散模型作为表达性变分后验，并通过反转加噪过程在潜空间中进行扩散。该方法易于实现，兼容黑盒变分推断，并在深度潜变量模型中的任务中表现优异。

    

    我们提出了去噪扩散变分推断（DDVI），一种用扩散模型作为表达性变分后验的潜变量模型的近似推断算法。我们的方法通过辅助潜变量增加了变分后验，从而得到一个表达性的模型类，通过反转用户指定的加噪过程在潜空间中进行扩散。我们通过优化一个受到觉醒-睡眠算法启发的边际似然新下界来拟合这些模型。我们的方法易于实现（它适配了正则化的ELBO扩展），与黑盒变分推断兼容，并且表现优于基于归一化流或对抗网络的替代近似后验类别。将我们的方法应用于深度潜变量模型时，我们的方法得到了去噪扩散变分自动编码器（DD-VAE）算法。我们将该算法应用于生物学中的一个激励任务 -- 从人类基因组中推断潜在血统 -- 超过了强基线模型。

    We propose denoising diffusion variational inference (DDVI), an approximate inference algorithm for latent variable models which relies on diffusion models as expressive variational posteriors. Our method augments variational posteriors with auxiliary latents, which yields an expressive class of models that perform diffusion in latent space by reversing a user-specified noising process. We fit these models by optimizing a novel lower bound on the marginal likelihood inspired by the wake-sleep algorithm. Our method is easy to implement (it fits a regularized extension of the ELBO), is compatible with black-box variational inference, and outperforms alternative classes of approximate posteriors based on normalizing flows or adversarial networks. When applied to deep latent variable models, our method yields the denoising diffusion VAE (DD-VAE) algorithm. We use this algorithm on a motivating task in biology -- inferring latent ancestry from human genomes -- outperforming strong baselines
    
[^5]: Softmax的优越性：揭示其相对于线性注意力的性能优势

    Superiority of Softmax: Unveiling the Performance Edge Over Linear Attention. (arXiv:2310.11685v1 [cs.CL])

    [http://arxiv.org/abs/2310.11685](http://arxiv.org/abs/2310.11685)

    通过对softmax和线性注意力机制进行全面比较分析，本论文揭示了softmax注意力在大多数情况下优于线性注意力的潜在原因。

    

    大型Transformer模型在许多自然语言处理任务中取得了最先进的结果。在Transformer架构的重要组成部分中，注意力机制通过利用softmax函数捕捉序列中的标记交互起着关键作用。相反，线性注意力通过线性复杂度近似softmax操作，提供了一种计算效率更高的替代方法。然而，与传统的softmax注意力机制相比，它在性能上表现出明显的降级。在本文中，我们对这两种注意力机制进行了全面的比较分析，揭示了softmax注意力在大多数情况下优于线性注意力的潜在原因。

    Large transformer models have achieved state-of-the-art results in numerous natural language processing tasks. Among the pivotal components of the transformer architecture, the attention mechanism plays a crucial role in capturing token interactions within sequences through the utilization of softmax function.  Conversely, linear attention presents a more computationally efficient alternative by approximating the softmax operation with linear complexity. However, it exhibits substantial performance degradation when compared to the traditional softmax attention mechanism.  In this paper, we bridge the gap in our theoretical understanding of the reasons behind the practical performance gap between softmax and linear attention. By conducting a comprehensive comparative analysis of these two attention mechanisms, we shed light on the underlying reasons for why softmax attention outperforms linear attention in most scenarios.
    
[^6]: 部分可观测的多Agent强化学习与（准）效率：信息共享的好处。

    Partially Observable Multi-agent RL with (Quasi-)Efficiency: The Blessing of Information Sharing. (arXiv:2308.08705v1 [cs.LG])

    [http://arxiv.org/abs/2308.08705](http://arxiv.org/abs/2308.08705)

    本文研究了部分可观测随机博弈的可证明多Agent强化学习。通过信息共享和观测可能性假设，提出了构建近似模型以实现准效率的方法。

    

    本文研究了部分可观测随机博弈（POSGs）的可证明多Agent强化学习（MARL）。为了规避已知的难度问题和使用计算不可行的预言机，我们倡导利用Agent之间的潜在“信息共享”，这是实证MARL中的常见做法，也是具备通信功能的多Agent控制系统的标准模型。我们首先建立了若干计算复杂性结果，来证明信息共享的必要性，以及观测可能性假设为了求解POSGs中的计算效率已经使得部分可观测的单Agent强化学习具有准效率。然后我们提出进一步“近似”共享的公共信息构建POSG的“近似模型”，在该模型中计划一个近似均衡（从解决原始POSG的角度）可以实现准效率，即准多项式时间，前提是上述假设满足。

    We study provable multi-agent reinforcement learning (MARL) in the general framework of partially observable stochastic games (POSGs). To circumvent the known hardness results and the use of computationally intractable oracles, we advocate leveraging the potential \emph{information-sharing} among agents, a common practice in empirical MARL, and a standard model for multi-agent control systems with communications. We first establish several computation complexity results to justify the necessity of information-sharing, as well as the observability assumption that has enabled quasi-efficient single-agent RL with partial observations, for computational efficiency in solving POSGs. We then propose to further \emph{approximate} the shared common information to construct an {approximate model} of the POSG, in which planning an approximate equilibrium (in terms of solving the original POSG) can be quasi-efficient, i.e., of quasi-polynomial-time, under the aforementioned assumptions. Furthermo
    
[^7]: 通过子高斯内在矩范实现紧凑的非渐进推断

    Tight Non-asymptotic Inference via Sub-Gaussian Intrinsic Moment Norm. (arXiv:2303.07287v1 [stat.ML])

    [http://arxiv.org/abs/2303.07287](http://arxiv.org/abs/2303.07287)

    本文提出了一种通过最大化一系列归一化矩来使用子高斯内在矩范实现紧凑的非渐进推断的方法，该方法可以导致更紧的Hoeffding子高斯浓度不等式，并且可以通过子高斯图检查具有有限样本大小的子高斯数据。

    This paper proposes a method of achieving tight non-asymptotic inference by using sub-Gaussian intrinsic moment norm through maximizing a series of normalized moments, which can lead to tighter Hoeffding's sub-Gaussian concentration inequalities and can be checked with sub-Gaussian plot for sub-Gaussian data with a finite sample size.

    在非渐进统计推断中，子高斯分布的方差类型参数起着至关重要的作用。然而，基于经验矩生成函数（MGF）的直接估计这些参数是不可行的。为此，我们建议通过最大化一系列归一化矩来使用子高斯内在矩范[Buldygin和Kozachenko（2000），定理1.3]。重要的是，推荐的范数不仅可以恢复相应MGF的指数矩界限，而且还可以导致更紧的Hoeffding子高斯浓度不等式。在实践中，我们提出了一种直观的方法，通过子高斯图检查具有有限样本大小的子高斯数据。可以通过简单的插入方法鲁棒地估计内在矩范数。我们的理论结果应用于非渐进分析，包括多臂赌博机。

    In non-asymptotic statistical inferences, variance-type parameters of sub-Gaussian distributions play a crucial role. However, direct estimation of these parameters based on the empirical moment generating function (MGF) is infeasible. To this end, we recommend using a sub-Gaussian intrinsic moment norm [Buldygin and Kozachenko (2000), Theorem 1.3] through maximizing a series of normalized moments. Importantly, the recommended norm can not only recover the exponential moment bounds for the corresponding MGFs, but also lead to tighter Hoeffding's sub-Gaussian concentration inequalities. In practice, {\color{black} we propose an intuitive way of checking sub-Gaussian data with a finite sample size by the sub-Gaussian plot}. Intrinsic moment norm can be robustly estimated via a simple plug-in approach. Our theoretical results are applied to non-asymptotic analysis, including the multi-armed bandit.
    
[^8]: 基于数据驱动的最优化因果推断影响函数

    Data-Driven Influence Functions for Optimization-Based Causal Inference. (arXiv:2208.13701v4 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2208.13701](http://arxiv.org/abs/2208.13701)

    本文提出了一种利用有限差分逼近统计泛函Gateaux导数的构造算法，并研究了从数据中进行概率分布估计的情况下的Gateaux导数估计。研究结果为因果推断和动态治疗方案等问题提供了解决方案。

    

    本研究探讨了一种利用有限差分逼近统计泛函Gateaux导数的构造算法，重点研究了在因果推断中出现的泛函。我们研究了概率分布未知但需要从数据中进行估计的情况。这些估计分布引导了经验Gateaux导数，我们研究了经验、数值和解析Gateaux导数之间的关系。从干预均值（平均潜在结果）的案例入手，我们勾勒了有限差分和解析Gateaux导数之间的关系。然后，我们得出了关于扰动和平滑的数值逼近速率要求，以保持单步调整的统计优势，例如速率双重强健性。接下来，我们研究了更复杂的泛函，如动态治疗方案、无限时段Markov决策中策略优化的线性规划形式。

    We study a constructive algorithm that approximates Gateaux derivatives for statistical functionals by finite differencing, with a focus on functionals that arise in  causal inference. We study the case where probability distributions are not known a priori but need to be estimated from data. These estimated distributions lead to empirical Gateaux derivatives, and we study the relationships between empirical, numerical, and analytical Gateaux derivatives. Starting with a case study of the interventional mean (average potential outcome), we delineate the relationship between finite differences and the analytical Gateaux derivative. We then derive requirements on the rates of numerical approximation in perturbation and smoothing that preserve the statistical benefits of one-step adjustments, such as rate double robustness. We then study more complicated functionals such as dynamic treatment regimes, the linear-programming formulation for policy optimization in infinite-horizon Markov dec
    

