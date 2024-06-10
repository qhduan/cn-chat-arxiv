# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [An Ordering of Divergences for Variational Inference with Factorized Gaussian Approximations](https://arxiv.org/abs/2403.13748) | 不同的散度排序可以通过它们的变分近似误估不确定性的各种度量，并且因子化近似无法同时匹配这些度量中的任意两个 |
| [^2] | [Transformers are Expressive, But Are They Expressive Enough for Regression?](https://arxiv.org/abs/2402.15478) | Transformer在逼近连续函数方面存在困难，是否真正是通用函数逼近器仍有待考证 |
| [^3] | [Dealing with unbounded gradients in stochastic saddle-point optimization](https://arxiv.org/abs/2402.13903) | 提出一种简单而有效的正则化技术，稳定了随机鞍点优化过程中的梯度不断增长的问题，能够在无界梯度和噪声的情况下提供有意义的性能保证 |
| [^4] | [Neural Control System for Continuous Glucose Monitoring and Maintenance](https://arxiv.org/abs/2402.13852) | 引入了一种新颖的神经控制系统，用于连续葡萄糖监测和维护，实时动态调整胰岛素输送，增强葡萄糖优化，最大化效率并确保个性化护理。 |
| [^5] | [Meta-learning the mirror map in policy mirror descent](https://arxiv.org/abs/2402.05187) | 该论文通过实证研究发现，传统的镜像映射选择（NPG）在标准基准环境中常常导致不理想的结果。通过元学习方法，找到了更高效的镜像映射，提升了性能。 |
| [^6] | [$C^*$-Algebraic Machine Learning: Moving in a New Direction](https://arxiv.org/abs/2402.02637) | $C^*$-代数机器学习是将$C^*$-代数与机器学习结合的新研究方向，它通过统一现有的学习策略，并构建更多元化和信息丰富的数据模型的新框架，为机器学习提供了一种新的方法。 |
| [^7] | [Discrete Diffusion Language Modeling by Estimating the Ratios of the Data Distribution.](http://arxiv.org/abs/2310.16834) | 本研究通过引入得分熵这一新颖的离散得分匹配损失，弥补了离散数据领域中现有方法的不足，提出了得分熵离散扩散模型(SEDD)并在GPT-2实验中取得了有竞争力的效果。 |
| [^8] | [Causal Representation Learning Made Identifiable by Grouping of Observational Variables.](http://arxiv.org/abs/2310.15709) | 该论文研究了因果表示学习（CRL），提出了一种基于观测变量分组的新型弱约束的可辨识性方法，该方法不依赖于时间结构、干预或监督，具有实际应用的意义。 |
| [^9] | [GTA: A Geometry-Aware Attention Mechanism for Multi-View Transformers.](http://arxiv.org/abs/2310.10375) | 提出了一种面向几何的注意力机制（GTA），用于将几何结构编码为相对变换，从而改进了多视图Transformer的学习效率和性能。 |
| [^10] | [Beyond Implicit Bias: The Insignificance of SGD Noise in Online Learning.](http://arxiv.org/abs/2306.08590) | 本文通过实际数据的全面实证分析证明了在在线学习中，大学习率和小批量大小并不会带来任何隐性偏差的优势。在线学习中SGD噪音的好处只是计算上的便利，可以促进更大或更具成本效益的梯度步骤。 |
| [^11] | [BOtied: Multi-objective Bayesian optimization with tied multivariate ranks.](http://arxiv.org/abs/2306.00344) | BOtied 是一种带有相关多元等级的多目标贝叶斯优化算法，相较于现有的方法具有更高的样本效率和较好的近似质量。 |
| [^12] | [An inexact linearized proximal algorithm for a class of DC composite optimization problems and applications.](http://arxiv.org/abs/2303.16822) | 本文提出了一种解决非凸非光滑问题的非精确线性化近端算法，并应用于鲁棒分解中的两个问题，得到了有效的数值结果。 |
| [^13] | [PDExplain: Contextual Modeling of PDEs in the Wild.](http://arxiv.org/abs/2303.15827) | 我们提出了PDExplain，一种解释性的方法来解决偏微分方程。该算法能够通过提供少量样本的方式，预测未来时间步的PDE解，极大地协助了建立物理科学中基于数据的现象建模。 |

# 详细

[^1]: 变分推断中因子化高斯近似的差异排序

    An Ordering of Divergences for Variational Inference with Factorized Gaussian Approximations

    [https://arxiv.org/abs/2403.13748](https://arxiv.org/abs/2403.13748)

    不同的散度排序可以通过它们的变分近似误估不确定性的各种度量，并且因子化近似无法同时匹配这些度量中的任意两个

    

    在变分推断（VI）中，给定一个难以处理的分布$p$，问题是从一些更易处理的族$\mathcal{Q}$中计算最佳近似$q$。通常情况下，这种近似是通过最小化Kullback-Leibler (KL)散度来找到的。然而，存在其他有效的散度选择，当$\mathcal{Q}$不包含$p$时，每个散度都支持不同的解决方案。我们分析了在高斯的密集协方差矩阵被对角协方差矩阵的高斯近似所影响的VI结果中，散度选择如何影响VI结果。在这种设置中，我们展示了不同的散度可以通过它们的变分近似误估不确定性的各种度量，如方差、精度和熵，进行\textit{排序}。我们还得出一个不可能定理，表明无法通过因子化近似同时匹配这些度量中的任意两个；因此

    arXiv:2403.13748v1 Announce Type: cross  Abstract: Given an intractable distribution $p$, the problem of variational inference (VI) is to compute the best approximation $q$ from some more tractable family $\mathcal{Q}$. Most commonly the approximation is found by minimizing a Kullback-Leibler (KL) divergence. However, there exist other valid choices of divergences, and when $\mathcal{Q}$ does not contain~$p$, each divergence champions a different solution. We analyze how the choice of divergence affects the outcome of VI when a Gaussian with a dense covariance matrix is approximated by a Gaussian with a diagonal covariance matrix. In this setting we show that different divergences can be \textit{ordered} by the amount that their variational approximations misestimate various measures of uncertainty, such as the variance, precision, and entropy. We also derive an impossibility theorem showing that no two of these measures can be simultaneously matched by a factorized approximation; henc
    
[^2]: Transformer是表现力强大的，但是对于回归任务来说表现力足够吗？

    Transformers are Expressive, But Are They Expressive Enough for Regression?

    [https://arxiv.org/abs/2402.15478](https://arxiv.org/abs/2402.15478)

    Transformer在逼近连续函数方面存在困难，是否真正是通用函数逼近器仍有待考证

    

    Transformer已成为自然语言处理中至关重要的技术，在机器翻译和摘要等应用中表现出色。随着它们的广泛应用，一些研究尝试分析Transformer的表现力。神经网络的表现力指的是它能够逼近的函数类。一个神经网络是完全表现力的，如果它可以充当通用函数逼近器。我们尝试分析Transformer的表现力。与现有观点相反，我们的研究结果表明，Transformer在可靠逼近连续函数方面存在困难，依赖于具有可观区间的分段常数逼近。关键问题是：“Transformer是否真正是通用函数逼近器？”为了解决这个问题，我们进行了彻底的调查，通过实验提供理论见解和支持证据。我们的贡献包括了一个理论分析……（摘要未完整）

    arXiv:2402.15478v1 Announce Type: new  Abstract: Transformers have become pivotal in Natural Language Processing, demonstrating remarkable success in applications like Machine Translation and Summarization. Given their widespread adoption, several works have attempted to analyze the expressivity of Transformers. Expressivity of a neural network is the class of functions it can approximate. A neural network is fully expressive if it can act as a universal function approximator. We attempt to analyze the same for Transformers. Contrary to existing claims, our findings reveal that Transformers struggle to reliably approximate continuous functions, relying on piecewise constant approximations with sizable intervals. The central question emerges as: "\textit{Are Transformers truly Universal Function Approximators}?" To address this, we conduct a thorough investigation, providing theoretical insights and supporting evidence through experiments. Our contributions include a theoretical analysi
    
[^3]: 处理随机鞍点优化中的无界梯度

    Dealing with unbounded gradients in stochastic saddle-point optimization

    [https://arxiv.org/abs/2402.13903](https://arxiv.org/abs/2402.13903)

    提出一种简单而有效的正则化技术，稳定了随机鞍点优化过程中的梯度不断增长的问题，能够在无界梯度和噪声的情况下提供有意义的性能保证

    

    我们研究了用于寻找凸凹函数鞍点的随机一阶方法的性能。这类方法面临的一个举世闻名的挑战是，在优化过程中梯度可能会任意增长，这可能导致不稳定性和发散。在本文中，我们提出了一种简单而有效的正则化技术，稳定了迭代并产生了有意义的性能保证，即使定义域和梯度噪声随迭代的规模线性变化（因此可能是无界的）。除了提供一系列一般性结果外，我们还将我们的算法应用到强化学习中的一个具体问题，该问题导致在不需要有关偏置跨度先验知识的情况下，找到平均奖励MDP中接近最优策略的性能保证。

    arXiv:2402.13903v1 Announce Type: new  Abstract: We study the performance of stochastic first-order methods for finding saddle points of convex-concave functions. A notorious challenge faced by such methods is that the gradients can grow arbitrarily large during optimization, which may result in instability and divergence. In this paper, we propose a simple and effective regularization technique that stabilizes the iterates and yields meaningful performance guarantees even if the domain and the gradient noise scales linearly with the size of the iterates (and is thus potentially unbounded). Besides providing a set of general results, we also apply our algorithm to a specific problem in reinforcement learning, where it leads to performance guarantees for finding near-optimal policies in an average-reward MDP without prior knowledge of the bias span.
    
[^4]: 连续葡萄糖监测和维护的神经控制系统

    Neural Control System for Continuous Glucose Monitoring and Maintenance

    [https://arxiv.org/abs/2402.13852](https://arxiv.org/abs/2402.13852)

    引入了一种新颖的神经控制系统，用于连续葡萄糖监测和维护，实时动态调整胰岛素输送，增强葡萄糖优化，最大化效率并确保个性化护理。

    

    精确的葡萄糖水平管理对于糖尿病患者至关重要，可以避免严重并发症。本研究引入了一种新颖的神经控制系统，用于连续葡萄糖监测和维护，利用微分预测控制。我们的系统受到复杂神经策略和可区分建模的指导，实时动态调整胰岛素输送，增强葡萄糖优化。这种端到端方法最大化效率，确保个性化护理和改善健康结果，如经验发现所证实。

    arXiv:2402.13852v1 Announce Type: cross  Abstract: Precise glucose level management is pivotal for individuals with diabetes, averting severe complications. In this work, we introduce a novel neural control system for continuous glucose monitoring and maintenance, utilizing differential predictive control. Our system, guided by a sophisticated neural policy and differentiable modeling, dynamically adjusts insulin delivery in real-time, enhancing glucose optimization. This end-to-end approach maximizes efficiency, ensuring personalized care and improved health outcomes, as affirmed by empirical findings.
    
[^5]: 在策略镜像下降中元学习镜像映射

    Meta-learning the mirror map in policy mirror descent

    [https://arxiv.org/abs/2402.05187](https://arxiv.org/abs/2402.05187)

    该论文通过实证研究发现，传统的镜像映射选择（NPG）在标准基准环境中常常导致不理想的结果。通过元学习方法，找到了更高效的镜像映射，提升了性能。

    

    策略镜像下降（PMD）是强化学习中的一种流行框架，作为一种统一视角，它包含了许多算法。这些算法是通过选择一个镜像映射而导出的，并且具有有限时间的收敛保证。尽管它很受欢迎，但对PMD的全面潜力的探索是有限的，大部分研究集中在一个特定的镜像映射上，即负熵，从而产生了著名的自然策略梯度（NPG）方法。目前的理论研究还不确定镜像映射的选择是否会对PMD的有效性产生重大影响。在我们的工作中，我们进行了实证研究，证明了传统的镜像映射选择（NPG）在几个标准基准环境中经常产生不理想的结果。通过应用元学习方法，我们确定了更高效的镜像映射，提高了性能，无论是平均性能还是最佳性能。

    Policy Mirror Descent (PMD) is a popular framework in reinforcement learning, serving as a unifying perspective that encompasses numerous algorithms. These algorithms are derived through the selection of a mirror map and enjoy finite-time convergence guarantees. Despite its popularity, the exploration of PMD's full potential is limited, with the majority of research focusing on a particular mirror map -- namely, the negative entropy -- which gives rise to the renowned Natural Policy Gradient (NPG) method. It remains uncertain from existing theoretical studies whether the choice of mirror map significantly influences PMD's efficacy. In our work, we conduct empirical investigations to show that the conventional mirror map choice (NPG) often yields less-than-optimal outcomes across several standard benchmark environments. By applying a meta-learning approach, we identify more efficient mirror maps that enhance performance, both on average and in terms of best performance achieved along th
    
[^6]: $C^*$-代数机器学习：迈向新的方向

    $C^*$-Algebraic Machine Learning: Moving in a New Direction

    [https://arxiv.org/abs/2402.02637](https://arxiv.org/abs/2402.02637)

    $C^*$-代数机器学习是将$C^*$-代数与机器学习结合的新研究方向，它通过统一现有的学习策略，并构建更多元化和信息丰富的数据模型的新框架，为机器学习提供了一种新的方法。

    

    机器学习与数学的几个领域（如统计学、概率论和线性代数）有着长期的合作传统。我们提出了机器学习研究的一个新方向：$C^*$-代数机器学习，这是$C^*$-代数和机器学习之间的交流和相互滋养。$C^*$-代数是复数空间的自然推广的数学概念，它使我们能够统一现有的学习策略，并构建一个更多元化和信息丰富的数据模型的新框架。我们解释了在机器学习中使用$C^*$-代数的原因和方法，并提供了在核方法和神经网络背景下设计$C^*$-代数学习模型的技术考虑。此外，我们讨论了$C^*$-代数机器学习中的开放问题和挑战，并提出了我们对未来发展和应用的思考。

    Machine learning has a long collaborative tradition with several fields of mathematics, such as statistics, probability and linear algebra. We propose a new direction for machine learning research: $C^*$-algebraic ML $-$ a cross-fertilization between $C^*$-algebra and machine learning. The mathematical concept of $C^*$-algebra is a natural generalization of the space of complex numbers. It enables us to unify existing learning strategies, and construct a new framework for more diverse and information-rich data models. We explain why and how to use $C^*$-algebras in machine learning, and provide technical considerations that go into the design of $C^*$-algebraic learning models in the contexts of kernel methods and neural networks. Furthermore, we discuss open questions and challenges in $C^*$-algebraic ML and give our thoughts for future development and applications.
    
[^7]: 通过估计数据分布比例的离散扩散语言建模

    Discrete Diffusion Language Modeling by Estimating the Ratios of the Data Distribution. (arXiv:2310.16834v1 [stat.ML])

    [http://arxiv.org/abs/2310.16834](http://arxiv.org/abs/2310.16834)

    本研究通过引入得分熵这一新颖的离散得分匹配损失，弥补了离散数据领域中现有方法的不足，提出了得分熵离散扩散模型(SEDD)并在GPT-2实验中取得了有竞争力的效果。

    

    尽管扩散模型在许多生成建模任务中具有突破性的性能，但在自然语言等离散数据领域中却表现不佳。关键是，标准的扩散模型依赖于成熟的得分匹配理论，但是将其推广到离散结构并没有取得相同的经验收益。在本文中，我们通过提出得分熵，一种新颖的离散得分匹配损失，来弥补这个差距，它比现有方法更稳定，可以形成最大似然训练的ELBO，并且可以通过去噪变体高效优化。我们将我们的得分熵离散扩散模型（SEDD）扩展到GPT-2的实验设置中，实现了极具竞争力的似然度，同时引入了独特的算法优势。特别是，在比较大小相似的SEDD和GPT-2模型时，SEDD达到了可比较的困惑度（通常在基线的+$10\%$内，并且有时超过基线）。此外，SEDD模型学到了...

    Despite their groundbreaking performance for many generative modeling tasks, diffusion models have fallen short on discrete data domains such as natural language. Crucially, standard diffusion models rely on the well-established theory of score matching, but efforts to generalize this to discrete structures have not yielded the same empirical gains. In this work, we bridge this gap by proposing score entropy, a novel discrete score matching loss that is more stable than existing methods, forms an ELBO for maximum likelihood training, and can be efficiently optimized with a denoising variant. We scale our Score Entropy Discrete Diffusion models (SEDD) to the experimental setting of GPT-2, achieving highly competitive likelihoods while also introducing distinct algorithmic advantages. In particular, when comparing similarly sized SEDD and GPT-2 models, SEDD attains comparable perplexities (normally within $+10\%$ of and sometimes outperforming the baseline). Furthermore, SEDD models lear
    
[^8]: 通过观测变量的分组使因果表示学习可辨识化

    Causal Representation Learning Made Identifiable by Grouping of Observational Variables. (arXiv:2310.15709v1 [stat.ML])

    [http://arxiv.org/abs/2310.15709](http://arxiv.org/abs/2310.15709)

    该论文研究了因果表示学习（CRL），提出了一种基于观测变量分组的新型弱约束的可辨识性方法，该方法不依赖于时间结构、干预或监督，具有实际应用的意义。

    

    当今很有意义的话题是因果表示学习（Causal Representation Learning，简称CRL），其目标是以数据驱动的方式学习隐藏特征的因果模型。不幸的是，CRL存在严重的不适问题，因为它结合了表示学习和因果发现这两个容易不适问题。然而，要找到能保证唯一解的实际可识别性条件对于其实际应用非常重要。目前大多数方法都基于对潜在因果机制的假设，比如时间因果性、监督或干预的存在；在实际应用中，这些假设可能过于限制性。在这里，我们通过对观测混合表现出合适的变量分组的新型弱约束，展示了可辨识性。我们还提出了一种与模型一致的新型自我监督估计框架。

    A topic of great current interest is Causal Representation Learning (CRL), whose goal is to learn a causal model for hidden features in a data-driven manner. Unfortunately, CRL is severely ill-posed since it is a combination of the two notoriously ill-posed problems of representation learning and causal discovery. Yet, finding practical identifiability conditions that guarantee a unique solution is crucial for its practical applicability. Most approaches so far have been based on assumptions on the latent causal mechanisms, such as temporal causality, or existence of supervision or interventions; these can be too restrictive in actual applications. Here, we show identifiability based on novel, weak constraints, which requires no temporal structure, intervention, nor weak supervision. The approach is based assuming the observational mixing exhibits a suitable grouping of the observational variables. We also propose a novel self-supervised estimation framework consistent with the model, 
    
[^9]: GTA：一种面向几何的多视图Transformer的注意力机制

    GTA: A Geometry-Aware Attention Mechanism for Multi-View Transformers. (arXiv:2310.10375v1 [cs.CV])

    [http://arxiv.org/abs/2310.10375](http://arxiv.org/abs/2310.10375)

    提出了一种面向几何的注意力机制（GTA），用于将几何结构编码为相对变换，从而改进了多视图Transformer的学习效率和性能。

    

    随着transformers对输入标记的排列具有等变性，对标记的位置信息进行编码对许多任务是必要的。然而，由于现有的位置编码方案最初是为自然语言处理任务设计的，对于通常在其数据中表现出不同结构特性的视觉任务来说，它们的适用性值得怀疑。我们认为现有的位置编码方案对于3D视觉任务来说是次优的，因为它们不尊重其底层的3D几何结构。基于这个假设，我们提出了一种面向几何的注意力机制，它将标记的几何结构编码为由查询和键值对之间的几何关系所确定的相对变换。通过在稀疏宽基线多视图设置中评估多个新颖视图合成（NVS）数据集，我们展示了我们的注意力机制——几何变换注意力（GTA）如何提高了最先进的Transformer的学习效率和性能。

    As transformers are equivariant to the permutation of input tokens, encoding the positional information of tokens is necessary for many tasks. However, since existing positional encoding schemes have been initially designed for NLP tasks, their suitability for vision tasks, which typically exhibit different structural properties in their data, is questionable. We argue that existing positional encoding schemes are suboptimal for 3D vision tasks, as they do not respect their underlying 3D geometric structure. Based on this hypothesis, we propose a geometry-aware attention mechanism that encodes the geometric structure of tokens as relative transformation determined by the geometric relationship between queries and key-value pairs. By evaluating on multiple novel view synthesis (NVS) datasets in the sparse wide-baseline multi-view setting, we show that our attention, called Geometric Transform Attention (GTA), improves learning efficiency and performance of state-of-the-art transformer-b
    
[^10]: 超越隐性偏差：SGD噪声在在线学习中的不重要性

    Beyond Implicit Bias: The Insignificance of SGD Noise in Online Learning. (arXiv:2306.08590v1 [cs.LG])

    [http://arxiv.org/abs/2306.08590](http://arxiv.org/abs/2306.08590)

    本文通过实际数据的全面实证分析证明了在在线学习中，大学习率和小批量大小并不会带来任何隐性偏差的优势。在线学习中SGD噪音的好处只是计算上的便利，可以促进更大或更具成本效益的梯度步骤。

    

    先前的研究认为，SGD在深度学习中的成功归因于高学习率或小批量大小所引起的隐性偏差（“SGD噪声”）。而我们研究了SGD噪声在在线（即单个epoch）学习中的影响，通过对图像和语言数据的全面实证分析，我们证明了在在线学习中，大学习率和小批量大小并不会带来任何隐性偏差的优势。与离线学习相反，在线学习中SGD噪声的好处严格来说只是计算上的便利，可以促进更大或更具成本效益的梯度步骤。我们的研究表明，SGD在在线模式下可以被视为是在“无噪声梯度流算法”的“黄金路径”上踩踏嘈杂步伐。通过减少训练期间的SGD噪声和测量模型之间的逐点功能距离，我们提供了支持此假设的证据。

    The success of SGD in deep learning has been ascribed by prior works to the implicit bias induced by high learning rate or small batch size ("SGD noise"). While prior works that focused on offline learning (i.e., multiple-epoch training), we study the impact of SGD noise on online (i.e., single epoch) learning. Through an extensive empirical analysis of image and language data, we demonstrate that large learning rate and small batch size do not confer any implicit bias advantages in online learning. In contrast to offline learning, the benefits of SGD noise in online learning are strictly computational, facilitating larger or more cost-effective gradient steps. Our work suggests that SGD in the online regime can be construed as taking noisy steps along the "golden path" of the noiseless gradient flow algorithm. We provide evidence to support this hypothesis by conducting experiments that reduce SGD noise during training and by measuring the pointwise functional distance between models 
    
[^11]: BOtied: 带有相关多元等级的多目标贝叶斯优化

    BOtied: Multi-objective Bayesian optimization with tied multivariate ranks. (arXiv:2306.00344v1 [cs.LG])

    [http://arxiv.org/abs/2306.00344](http://arxiv.org/abs/2306.00344)

    BOtied 是一种带有相关多元等级的多目标贝叶斯优化算法，相较于现有的方法具有更高的样本效率和较好的近似质量。

    

    许多科学和工业应用需要同时优化多个潜在的相互冲突的目标。多目标贝叶斯优化 (MOBO) 是一种高效地识别 Pareto 最优解的框架。我们展示了无支配解和最高多元等级之间的自然联系，它与联合累积分布函数的最外层等高线重合。我们提出了 CDF indicator，这是一种 Pareto 合规的度量，用于评估近似 Pareto 集合的质量，它补充了流行的 hypervolume indicator。MOBO 的核心是采集函数，它通过导航目标之间的最佳折中来确定下一个要评估的候选项。 基于盒子分解目标空间的多目标采集函数（例如期望的 hypervolume 改进（EHVI）和熵搜索）在存在大量目标时的性能缩放很差。我们提出一种采集函数，称为 BOtied，它利用相关多元等级来高效搜索 Pareto frontier。我们的实验表明，BOtied 在样本效率和近似质量方面优于现有方法。

    Many scientific and industrial applications require joint optimization of multiple, potentially competing objectives. Multi-objective Bayesian optimization (MOBO) is a sample-efficient framework for identifying Pareto-optimal solutions. We show a natural connection between non-dominated solutions and the highest multivariate rank, which coincides with the outermost level line of the joint cumulative distribution function (CDF). We propose the CDF indicator, a Pareto-compliant metric for evaluating the quality of approximate Pareto sets that complements the popular hypervolume indicator. At the heart of MOBO is the acquisition function, which determines the next candidate to evaluate by navigating the best compromises among the objectives. Multi-objective acquisition functions that rely on box decomposition of the objective space, such as the expected hypervolume improvement (EHVI) and entropy search, scale poorly to a large number of objectives. We propose an acquisition function, call
    
[^12]: 一类DC复合优化问题的非精确线性近似近端算法及应用

    An inexact linearized proximal algorithm for a class of DC composite optimization problems and applications. (arXiv:2303.16822v1 [math.OC])

    [http://arxiv.org/abs/2303.16822](http://arxiv.org/abs/2303.16822)

    本文提出了一种解决非凸非光滑问题的非精确线性化近端算法，并应用于鲁棒分解中的两个问题，得到了有效的数值结果。

    

    本文研究了一类DC复合优化问题。这类问题通常由低秩矩阵恢复的鲁棒分解模型推导而来，是凸复合优化问题和具有非光滑分量的DC规划的扩展。针对这类非凸和非光滑问题，我们提出了一种非精确线性化近端算法（iLPA）。算法中，我们利用目标函数的部分线性化，计算强凸主导的非精确最小化值。迭代序列的生成收敛于潜在函数的Kurdyka-{\L}ojasiewicz（KL）性质，如果潜在函数在极限点处具有KL指数$1/2$的KL性质，则收敛具有局部R线性速率。对于后一种假设，我们利用复合结构提供了一个可验证的条件，并阐明了与凸复合优化所使用的正则性的关系。最后，我们将所提出的非精确线性近端算法应用于解决鲁棒分解中的两个重要问题：张量鲁棒主成分分析（TRPCA）和张量鲁棒低秩张量完成（TRLRTC）。对合成和真实数据的数值结果证明了我们的算法相对于现有最新算法的有效性。

    This paper is concerned with a class of DC composite optimization problems which, as an extension of the convex composite optimization problem and the DC program with nonsmooth components, often arises from robust factorization models of low-rank matrix recovery. For this class of nonconvex and nonsmooth problems, we propose an inexact linearized proximal algorithm (iLPA) which in each step computes an inexact minimizer of a strongly convex majorization constructed by the partial linearization of their objective functions. The generated iterate sequence is shown to be convergent under the Kurdyka-{\L}ojasiewicz (KL) property of a potential function, and the convergence admits a local R-linear rate if the potential function has the KL property of exponent $1/2$ at the limit point. For the latter assumption, we provide a verifiable condition by leveraging the composite structure, and clarify its relation with the regularity used for the convex composite optimization. Finally, the propose
    
[^13]: PDExplain：PDEs 在实际应用中的情境建模

    PDExplain: Contextual Modeling of PDEs in the Wild. (arXiv:2303.15827v1 [cs.LG])

    [http://arxiv.org/abs/2303.15827](http://arxiv.org/abs/2303.15827)

    我们提出了PDExplain，一种解释性的方法来解决偏微分方程。该算法能够通过提供少量样本的方式，预测未来时间步的PDE解，极大地协助了建立物理科学中基于数据的现象建模。

    

    我们提出了一种解释性的方法PDExplain用于解决偏微分方程。在训练阶段，我们的方法通过一个操作员定义的PDE家族的数据以及这个家族的一般形式进行馈送。在推断阶段，提供了一个从现象中收集到的最小样本，其中样本与 PDE 家族相关，但不一定属于训练阶段看到的具体 PDE 集合。我们展示了算法如何预测未来时间步的PDE解。此外，我们的方法提供了PDE的可解释形式，这种特征可以协助通过物理科学数据来对现象进行建模。为了验证我们的方法，我们进行了大量实验，考察了其在预测误差和可解释性方面的质量。

    We propose an explainable method for solving Partial Differential Equations by using a contextual scheme called PDExplain. During the training phase, our method is fed with data collected from an operator-defined family of PDEs accompanied by the general form of this family. In the inference phase, a minimal sample collected from a phenomenon is provided, where the sample is related to the PDE family but not necessarily to the set of specific PDEs seen in the training phase. We show how our algorithm can predict the PDE solution for future timesteps. Moreover, our method provides an explainable form of the PDE, a trait that can assist in modelling phenomena based on data in physical sciences. To verify our method, we conduct extensive experimentation, examining its quality both in terms of prediction error and explainability.
    

