# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [The Unreasonable Ineffectiveness of the Deeper Layers](https://arxiv.org/abs/2403.17887) | 层剪枝方法可以在流行的预训练语言模型中实现大部分层的移除而保持性能，同时使用参数高效的微调方法可以进一步减少计算资源，提高推断的内存和延迟。 |
| [^2] | [On the Asymptotic Mean Square Error Optimality of Diffusion Probabilistic Models](https://arxiv.org/abs/2403.02957) | 本论文通过严格证明一个特定的DPM去噪策略在大量扩散步数下收敛到均方误差最优条件均值估计器，突出了DPM由渐近最优的去噪器组成，同时具有强大生成器的独特视角。 |
| [^3] | [Linear quadratic control of nonlinear systems with Koopman operator learning and the Nystr\"om method](https://arxiv.org/abs/2403.02811) | 本文将Koopman算子框架与核方法相结合，通过Nyström逼近实现了对非线性动态系统的有效控制，其理论贡献在于推导出Nyström逼近效果的理论保证。 |
| [^4] | [Signature Kernel Conditional Independence Tests in Causal Discovery for Stochastic Processes](https://arxiv.org/abs/2402.18477) | 本文在随机过程中开发了一种基于签名核的条件独立性测试，实现了对因果关系的推断，以及开发了约束条件的因果发现算法用于恢复整个有向图。 |
| [^5] | [Improving LSH via Tensorized Random Projection](https://arxiv.org/abs/2402.07189) | 本文提出了CP-E2LSH和TT-E2LSH两种方法，用于改进局部敏感哈希算法LSH，在处理张量数据的欧几里得距离和余弦相似度时能够提供更快和更空间有效的结果。 |
| [^6] | [Hypergraph Node Classification With Graph Neural Networks](https://arxiv.org/abs/2402.05569) | 本研究提出了一种简单高效的框架，利用加权子图扩展的图神经网络(WCE-GNN)实现了超图节点分类。实验证明，WCE-GNN具有优秀的预测效果和较低的计算复杂度。 |
| [^7] | [Global $\mathcal{L}^2$ minimization at uniform exponential rate via geometrically adapted gradient descent in Deep Learning](https://arxiv.org/abs/2311.15487) | 通过几何调整的梯度下降，在深度学习中以均匀指数速率实现全局$\mathcal{L}^2$最小化，这一方法在过参数化情况下具有明确自然的不变几何含义。 |
| [^8] | [Asymptotic Behavior of Adversarial Training Estimator under $\ell_\infty$-Perturbation.](http://arxiv.org/abs/2401.15262) | 本文研究了在$\ell_\infty$-扰动下的对抗性训练，证明当真实参数为0时，对抗性训练估计器在该扰动下的极限分布可能在0处有一个正概率质量，提供了稀疏性恢复能力的理论保证，并提出了一种两步过程——自适应对抗性训练，可以进一步提高性能。 |
| [^9] | [Assessing Robustness via Score-Based Adversarial Image Generation.](http://arxiv.org/abs/2310.04285) | 本论文介绍了一种基于分数的对抗生成框架（ScoreAG），可以生成超过$\ell_p$-范数约束的对抗性示例，并通过图像转换或新图像合成的方法保持图像的核心语义，大大增强了分类器的鲁棒性。 |
| [^10] | [Online Estimation and Inference for Robust Policy Evaluation in Reinforcement Learning.](http://arxiv.org/abs/2310.02581) | 该论文提出了一种针对鲁棒策略评估的在线估计和推断方法，在解决异常值污染和重尾奖励的问题方面引入了鲁棒统计学的概念。此外，还提出了一种完全在线的统计推断过程，并建立了估计量的极限分布。 |
| [^11] | [Geometry-Aware Approaches for Balancing Performance and Theoretical Guarantees in Linear Bandits.](http://arxiv.org/abs/2306.14872) | 本文提出了一种新的数据驱动技术，跟踪不确定度椭球体的几何形状，为线性赌博机算法建立实例相关的频率后悔界，并实现了平衡算法性能与理论保证的效果。 |
| [^12] | [Accelerating Generalized Random Forests with Fixed-Point Trees.](http://arxiv.org/abs/2306.11908) | 本文提出一种新的树生长规则，使广义随机森林在无梯度优化的情况下大大节省了时间。 |

# 详细

[^1]: 深层神经网络层剪枝的不合理无效性

    The Unreasonable Ineffectiveness of the Deeper Layers

    [https://arxiv.org/abs/2403.17887](https://arxiv.org/abs/2403.17887)

    层剪枝方法可以在流行的预训练语言模型中实现大部分层的移除而保持性能，同时使用参数高效的微调方法可以进一步减少计算资源，提高推断的内存和延迟。

    

    我们在流行的预训练语言模型中进行了简单的层剪枝策略的实证研究，发现在移除大部分层（最高达一半）之前，不同问答基准测试的性能几乎没有受到影响。为了剪枝这些模型，我们通过考虑层间的相似性来确定最佳的剪枝层块；然后，为了“修复”损害，我们进行了少量微调。特别地，我们使用参数高效的微调（PEFT）方法，具体包括量化和低秩适配器（QLoRA），这样我们的每个实验都可以在单个A100 GPU上执行。从实际的角度来看，这些结果表明层剪枝方法可以补充其他PEFT策略，从而进一步减少微调的计算资源，另一方面可以提高推断的内存和延迟。从科学的角度来看，该研究表明深层神经网络在某种程度上具有鲁棒性，并且对模型的剪枝没有太大影响。

    arXiv:2403.17887v1 Announce Type: new  Abstract: We empirically study a simple layer-pruning strategy for popular families of open-weight pretrained LLMs, finding minimal degradation of performance on different question-answering benchmarks until after a large fraction (up to half) of the layers are removed. To prune these models, we identify the optimal block of layers to prune by considering similarity across layers; then, to "heal" the damage, we perform a small amount of finetuning. In particular, we use parameter-efficient finetuning (PEFT) methods, specifically quantization and Low Rank Adapters (QLoRA), such that each of our experiments can be performed on a single A100 GPU. From a practical perspective, these results suggest that layer pruning methods can complement other PEFT strategies to further reduce computational resources of finetuning on the one hand, and can improve the memory and latency of inference on the other hand. From a scientific perspective, the robustness of 
    
[^2]: 关于扩散概率模型渐近均方误差最优性的研究

    On the Asymptotic Mean Square Error Optimality of Diffusion Probabilistic Models

    [https://arxiv.org/abs/2403.02957](https://arxiv.org/abs/2403.02957)

    本论文通过严格证明一个特定的DPM去噪策略在大量扩散步数下收敛到均方误差最优条件均值估计器，突出了DPM由渐近最优的去噪器组成，同时具有强大生成器的独特视角。

    

    最近，扩散概率模型（DPMs）在去噪任务中展现出巨大潜力。尽管它们在实际应用中很有用，但它们的理论理解存在明显的差距。本文通过严格证明特定DPM去噪策略在大量扩散步数下收敛到均方误差（MSE）最优条件均值估计器（CME），为该领域提供了新的理论见解。研究的基于DPM的去噪器在训练过程中与DPMs共享，但在训练后的逆推理过程中仅传递条件均值。我们强调了DPM由渐近最优的去噪器组成的独特视角，同时通过在逆过程中切换重新采样的方式继承了一个强大的生成器。通过数值结果验证了理论发现。

    arXiv:2403.02957v1 Announce Type: new  Abstract: Diffusion probabilistic models (DPMs) have recently shown great potential for denoising tasks. Despite their practical utility, there is a notable gap in their theoretical understanding. This paper contributes novel theoretical insights by rigorously proving the asymptotic convergence of a specific DPM denoising strategy to the mean square error (MSE)-optimal conditional mean estimator (CME) over a large number of diffusion steps. The studied DPM-based denoiser shares the training procedure of DPMs but distinguishes itself by forwarding only the conditional mean during the reverse inference process after training. We highlight the unique perspective that DPMs are composed of an asymptotically optimal denoiser while simultaneously inheriting a powerful generator by switching re-sampling in the reverse process on and off. The theoretical findings are validated by numerical results.
    
[^3]: 具有Koopman算子学习和Nyström方法的非线性系统的线性二次控制

    Linear quadratic control of nonlinear systems with Koopman operator learning and the Nystr\"om method

    [https://arxiv.org/abs/2403.02811](https://arxiv.org/abs/2403.02811)

    本文将Koopman算子框架与核方法相结合，通过Nyström逼近实现了对非线性动态系统的有效控制，其理论贡献在于推导出Nyström逼近效果的理论保证。

    

    在本文中，我们研究了Koopman算子框架如何与核方法相结合以有效控制非线性动力系统。虽然核方法通常具有很大的计算需求，但我们展示了随机子空间（Nyström逼近）如何实现巨大的计算节约，同时保持精度。我们的主要技术贡献在于推导出关于Nyström逼近效果的理论保证。更具体地说，我们研究了线性二次调节器问题，证明了对于最优控制问题的相关解的近似Riccati算子和调节器目标都以$ m^{-1/2} $的速率收敛，其中$ m $是随机子空间大小。理论发现得到了数值实验的支持。

    arXiv:2403.02811v1 Announce Type: cross  Abstract: In this paper, we study how the Koopman operator framework can be combined with kernel methods to effectively control nonlinear dynamical systems. While kernel methods have typically large computational requirements, we show how random subspaces (Nystr\"om approximation) can be used to achieve huge computational savings while preserving accuracy. Our main technical contribution is deriving theoretical guarantees on the effect of the Nystr\"om approximation. More precisely, we study the linear quadratic regulator problem, showing that both the approximated Riccati operator and the regulator objective, for the associated solution of the optimal control problem, converge at the rate $m^{-1/2}$, where $m$ is the random subspace size. Theoretical findings are complemented by numerical experiments corroborating our results.
    
[^4]: 在因果发现中的签名核条件独立性测试用于随机过程

    Signature Kernel Conditional Independence Tests in Causal Discovery for Stochastic Processes

    [https://arxiv.org/abs/2402.18477](https://arxiv.org/abs/2402.18477)

    本文在随机过程中开发了一种基于签名核的条件独立性测试，实现了对因果关系的推断，以及开发了约束条件的因果发现算法用于恢复整个有向图。

    

    从观测数据中推断随机动力系统背后的因果结构在科学、健康和金融等领域具有巨大潜力。本文通过利用最近签名核技术的进展，开发了一种基于内核的“路径空间”上条件独立性（CI）测试，用于随机微分方程的解。我们展示了相较于现有方法，在路径空间上，我们提出的CI测试表现出严格更好的性能。此外，我们还为非循环随机动力系统开发了基于约束的因果发现算法，利用时间信息来恢复整个有向图。在假设忠实性和CI预言机的情况下，我们的算法是完备且正确的。

    arXiv:2402.18477v1 Announce Type: cross  Abstract: Inferring the causal structure underlying stochastic dynamical systems from observational data holds great promise in domains ranging from science and health to finance. Such processes can often be accurately modeled via stochastic differential equations (SDEs), which naturally imply causal relationships via "which variables enter the differential of which other variables". In this paper, we develop a kernel-based test of conditional independence (CI) on "path-space" -- solutions to SDEs -- by leveraging recent advances in signature kernels. We demonstrate strictly superior performance of our proposed CI test compared to existing approaches on path-space. Then, we develop constraint-based causal discovery algorithms for acyclic stochastic dynamical systems (allowing for loops) that leverage temporal information to recover the entire directed graph. Assuming faithfulness and a CI oracle, our algorithm is sound and complete. We empirical
    
[^5]: 通过张量化随机投影改进局部敏感哈希LSH

    Improving LSH via Tensorized Random Projection

    [https://arxiv.org/abs/2402.07189](https://arxiv.org/abs/2402.07189)

    本文提出了CP-E2LSH和TT-E2LSH两种方法，用于改进局部敏感哈希算法LSH，在处理张量数据的欧几里得距离和余弦相似度时能够提供更快和更空间有效的结果。

    

    局部敏感哈希(LSH)是数据科学家用于近似最近邻搜索问题的基本算法工具，已在许多大规模数据处理应用中广泛使用，如近似重复检测、最近邻搜索、聚类等。在本文中，我们旨在提出更快和空间更有效的局部敏感哈希函数，用于张量数据的欧几里得距离和余弦相似度。通常，对于张量数据获得LSH的朴素方法涉及将张量重塑为向量，然后应用现有的向量数据LSH方法(E2LSH和SRP)。然而，对于高阶张量，这种方法变得不切实际，因为重塑向量的大小在张量的阶数中呈指数增长。因此，LSH参数的大小呈指数增加。为解决这个问题，我们提出了两种欧几里得距离和余弦相似度的LSH方法，分别是CP-E2LSH和TT-E2LSH。

    Locality sensitive hashing (LSH) is a fundamental algorithmic toolkit used by data scientists for approximate nearest neighbour search problems that have been used extensively in many large scale data processing applications such as near duplicate detection, nearest neighbour search, clustering, etc. In this work, we aim to propose faster and space efficient locality sensitive hash functions for Euclidean distance and cosine similarity for tensor data. Typically, the naive approach for obtaining LSH for tensor data involves first reshaping the tensor into vectors, followed by applying existing LSH methods for vector data $E2LSH$ and $SRP$. However, this approach becomes impractical for higher order tensors because the size of the reshaped vector becomes exponential in the order of the tensor. Consequently, the size of LSH parameters increases exponentially. To address this problem, we suggest two methods for LSH for Euclidean distance and cosine similarity, namely $CP-E2LSH$, $TT-E2LSH
    
[^6]: 使用图神经网络进行超图节点分类

    Hypergraph Node Classification With Graph Neural Networks

    [https://arxiv.org/abs/2402.05569](https://arxiv.org/abs/2402.05569)

    本研究提出了一种简单高效的框架，利用加权子图扩展的图神经网络(WCE-GNN)实现了超图节点分类。实验证明，WCE-GNN具有优秀的预测效果和较低的计算复杂度。

    

    超图是用来模拟现实世界数据中的高阶相互作用的关键。图神经网络（GNNs）的成功揭示了神经网络处理具有成对交互的数据的能力。这激发了使用神经网络处理具有高阶相互作用的数据的想法，从而导致了超图神经网络（HyperGNNs）的发展。GNNs和HyperGNNs通常被认为是不同的，因为它们被设计用于处理不同几何拓扑的数据。然而，在本文中，我们在理论上证明，在节点分类的上下文中，大多数HyperGNNs可以使用带有超图的加权子图扩展的GNN来近似。这导致了WCE-GNN，一种简单高效的框架，包括一个GNN和一个加权子图扩展（WCE），用于超图节点分类。对于九个真实世界的超图节点分类数据集的实验表明，WCE-GNN不仅具有优秀的预测效果，而且具有较低的计算复杂度。

    Hypergraphs, with hyperedges connecting more than two nodes, are key for modelling higher-order interactions in real-world data. The success of graph neural networks (GNNs) reveals the capability of neural networks to process data with pairwise interactions. This inspires the usage of neural networks for data with higher-order interactions, thereby leading to the development of hypergraph neural networks (HyperGNNs). GNNs and HyperGNNs are typically considered distinct since they are designed for data on different geometric topologies. However, in this paper, we theoretically demonstrate that, in the context of node classification, most HyperGNNs can be approximated using a GNN with a weighted clique expansion of the hypergraph. This leads to WCE-GNN, a simple and efficient framework comprising a GNN and a weighted clique expansion (WCE), for hypergraph node classification. Experiments on nine real-world hypergraph node classification benchmarks showcase that WCE-GNN demonstrates not o
    
[^7]: 深度学习中通过几何调整的梯度下降以均匀指数速率全局$\mathcal{L}^2$最小化

    Global $\mathcal{L}^2$ minimization at uniform exponential rate via geometrically adapted gradient descent in Deep Learning

    [https://arxiv.org/abs/2311.15487](https://arxiv.org/abs/2311.15487)

    通过几何调整的梯度下降，在深度学习中以均匀指数速率实现全局$\mathcal{L}^2$最小化，这一方法在过参数化情况下具有明确自然的不变几何含义。

    

    我们考虑在深度学习网络中广泛使用的用于最小化$\mathcal{L}^2$代价函数的梯度下降流，并引入两个改进版本；一个适用于过参数化设置，另一个适用于欠参数化设置。这两个版本都具有明确自然的不变几何含义，考虑到在过参数化设置中的拉回向量丛结构和在欠参数化设置中的推前向量丛结构。在过参数化情况下，我们证明，只要满足秩条件，改进的梯度下降的所有轨道将以均匀指数收敛速率将$\mathcal{L}^2$代价驱动到全局最小值；因此，对于任何预先指定的接近全局最小值的近似，我们可以得到先验停止时间。我们指出后者与次Riemann几何的关系。

    arXiv:2311.15487v3 Announce Type: replace-cross  Abstract: We consider the gradient descent flow widely used for the minimization of the $\mathcal{L}^2$ cost function in Deep Learning networks, and introduce two modified versions; one adapted for the overparametrized setting, and the other for the underparametrized setting. Both have a clear and natural invariant geometric meaning, taking into account the pullback vector bundle structure in the overparametrized, and the pushforward vector bundle structure in the underparametrized setting. In the overparametrized case, we prove that, provided that a rank condition holds, all orbits of the modified gradient descent drive the $\mathcal{L}^2$ cost to its global minimum at a uniform exponential convergence rate; one thereby obtains an a priori stopping time for any prescribed proximity to the global minimum. We point out relations of the latter to sub-Riemannian geometry.
    
[^8]: 在$\ell_\infty$-扰动下对抗性训练估计器的渐近行为

    Asymptotic Behavior of Adversarial Training Estimator under $\ell_\infty$-Perturbation. (arXiv:2401.15262v1 [math.ST])

    [http://arxiv.org/abs/2401.15262](http://arxiv.org/abs/2401.15262)

    本文研究了在$\ell_\infty$-扰动下的对抗性训练，证明当真实参数为0时，对抗性训练估计器在该扰动下的极限分布可能在0处有一个正概率质量，提供了稀疏性恢复能力的理论保证，并提出了一种两步过程——自适应对抗性训练，可以进一步提高性能。

    

    对抗性训练被提出来抵御机器学习和统计模型中的对抗性攻击。本文重点研究了在$\ell_\infty$-扰动下的对抗性训练，这个问题最近引起了很多研究的关注。在广义线性模型中研究了对抗性训练估计器的渐近行为。结果表明，当真实参数为0时，对抗性训练估计器在$\ell_\infty$-扰动下的极限分布可能在0处有一个正概率质量，为相关的稀疏性恢复能力提供了理论保证。此外，提出了一种两步过程——自适应对抗性训练，可以进一步提高在$\ell_\infty$-扰动下的对抗性训练的性能。具体而言，所提出的过程可以实现渐近无偏性和变量选择一致性。通过数值实验展示了稀疏性恢复的能力。

    Adversarial training has been proposed to hedge against adversarial attacks in machine learning and statistical models. This paper focuses on adversarial training under $\ell_\infty$-perturbation, which has recently attracted much research attention. The asymptotic behavior of the adversarial training estimator is investigated in the generalized linear model. The results imply that the limiting distribution of the adversarial training estimator under $\ell_\infty$-perturbation could put a positive probability mass at $0$ when the true parameter is $0$, providing a theoretical guarantee of the associated sparsity-recovery ability. Alternatively, a two-step procedure is proposed -adaptive adversarial training, which could further improve the performance of adversarial training under $\ell_\infty$-perturbation. Specifically, the proposed procedure could achieve asymptotic unbiasedness and variable-selection consistency. Numerical experiments are conducted to show the sparsity-recovery a
    
[^9]: 通过基于分数的对抗图像生成评估鲁棒性

    Assessing Robustness via Score-Based Adversarial Image Generation. (arXiv:2310.04285v1 [cs.CV])

    [http://arxiv.org/abs/2310.04285](http://arxiv.org/abs/2310.04285)

    本论文介绍了一种基于分数的对抗生成框架（ScoreAG），可以生成超过$\ell_p$-范数约束的对抗性示例，并通过图像转换或新图像合成的方法保持图像的核心语义，大大增强了分类器的鲁棒性。

    

    大多数对抗攻击和防御都集中在小的$\ell_p$-范数约束内的扰动上。然而，$\ell_p$威胁模型无法捕捉到所有相关的保留语义的扰动，因此，鲁棒性评估的范围是有限的。在这项工作中，我们引入了基于分数的对抗生成（ScoreAG），一种利用基于分数的生成模型的进展来生成超过$\ell_p$-范数约束的对抗性示例的新的框架，称为无限制的对抗性示例，克服了它们的局限性。与传统方法不同，ScoreAG在生成逼真的对抗性示例时保持图像的核心语义，可以通过转换现有图像或完全从零开始合成新图像的方式实现。我们进一步利用ScoreAG的生成能力来净化图像，从经验上增强分类器的鲁棒性。我们的大量实证评估表明，ScoreAG与现有最先进的对抗攻击方法的性能相当。

    Most adversarial attacks and defenses focus on perturbations within small $\ell_p$-norm constraints. However, $\ell_p$ threat models cannot capture all relevant semantic-preserving perturbations, and hence, the scope of robustness evaluations is limited. In this work, we introduce Score-Based Adversarial Generation (ScoreAG), a novel framework that leverages the advancements in score-based generative models to generate adversarial examples beyond $\ell_p$-norm constraints, so-called unrestricted adversarial examples, overcoming their limitations. Unlike traditional methods, ScoreAG maintains the core semantics of images while generating realistic adversarial examples, either by transforming existing images or synthesizing new ones entirely from scratch. We further exploit the generative capability of ScoreAG to purify images, empirically enhancing the robustness of classifiers. Our extensive empirical evaluation demonstrates that ScoreAG matches the performance of state-of-the-art atta
    
[^10]: 在强化学习中的鲁棒策略评估的在线估计和推断

    Online Estimation and Inference for Robust Policy Evaluation in Reinforcement Learning. (arXiv:2310.02581v1 [stat.ML])

    [http://arxiv.org/abs/2310.02581](http://arxiv.org/abs/2310.02581)

    该论文提出了一种针对鲁棒策略评估的在线估计和推断方法，在解决异常值污染和重尾奖励的问题方面引入了鲁棒统计学的概念。此外，还提出了一种完全在线的统计推断过程，并建立了估计量的极限分布。

    

    最近，强化学习在现代统计学中备受关注，策略评估是其中一个关键组成部分。与传统机器学习文献上对该主题的研究不同，我们的工作强调使用强化学习算法计算的参数估计的统计推断。尽管大多数现有分析假设随机奖励遵循标准分布，限制了它们的适用性，但我们在统一框架中同时解决了异常值污染和重尾奖励的问题，从而拥抱了鲁棒统计学在强化学习中的概念。在本文中，我们开发了一种在线鲁棒策略评估过程，并根据其Bahadur表示建立了我们估计量的极限分布。此外，我们还开发了一种完全在线的过程，以高效地进行基于渐近分布的统计推断。这篇论文填补了强化学习中鲁棒统计学和统计推断之间的差距。

    Recently, reinforcement learning has gained prominence in modern statistics, with policy evaluation being a key component. Unlike traditional machine learning literature on this topic, our work places emphasis on statistical inference for the parameter estimates computed using reinforcement learning algorithms. While most existing analyses assume random rewards to follow standard distributions, limiting their applicability, we embrace the concept of robust statistics in reinforcement learning by simultaneously addressing issues of outlier contamination and heavy-tailed rewards within a unified framework. In this paper, we develop an online robust policy evaluation procedure, and establish the limiting distribution of our estimator, based on its Bahadur representation. Furthermore, we develop a fully-online procedure to efficiently conduct statistical inference based on the asymptotic distribution. This paper bridges the gap between robust statistics and statistical inference in reinfor
    
[^11]: 线性赌博机中平衡性能与理论保证的几何感知方法

    Geometry-Aware Approaches for Balancing Performance and Theoretical Guarantees in Linear Bandits. (arXiv:2306.14872v1 [cs.LG])

    [http://arxiv.org/abs/2306.14872](http://arxiv.org/abs/2306.14872)

    本文提出了一种新的数据驱动技术，跟踪不确定度椭球体的几何形状，为线性赌博机算法建立实例相关的频率后悔界，并实现了平衡算法性能与理论保证的效果。

    

    本文受线性赌博机算法表现良好的实证性能与悲观理论后悔界之间的不一致性启发，提出一种新的数据驱动技术，跟踪不确定度椭球体的几何形状，为包括贪心、OFUL和汤普森抽样算法在内的广泛算法类建立实例相关的频率后悔界，在保留基本算法大部分优良特性的同时“校正”基本算法在某些实例中表现差的问题，实现了渐近最优后悔界。我们通过仿真实验验证了该方法的有效性。

    This paper is motivated by recent developments in the linear bandit literature, which have revealed a discrepancy between the promising empirical performance of algorithms such as Thompson sampling and Greedy, when compared to their pessimistic theoretical regret bounds. The challenge arises from the fact that while these algorithms may perform poorly in certain problem instances, they generally excel in typical instances. To address this, we propose a new data-driven technique that tracks the geometry of the uncertainty ellipsoid, enabling us to establish an instance-dependent frequentist regret bound for a broad class of algorithms, including Greedy, OFUL, and Thompson sampling. This result empowers us to identify and ``course-correct" instances in which the base algorithms perform poorly. The course-corrected algorithms achieve the minimax optimal regret of order $\tilde{\mathcal{O}}(d\sqrt{T})$, while retaining most of the desirable properties of the base algorithms. We present sim
    
[^12]: 基于定点树的广义随机森林加速

    Accelerating Generalized Random Forests with Fixed-Point Trees. (arXiv:2306.11908v1 [stat.ML])

    [http://arxiv.org/abs/2306.11908](http://arxiv.org/abs/2306.11908)

    本文提出一种新的树生长规则，使广义随机森林在无梯度优化的情况下大大节省了时间。

    

    广义随机森林建立在传统随机森林的基础上，通过将其作为自适应核加权算法来构建估算器，并通过基于梯度的树生长过程来实现。我们提出了一种新的树生长规则，基于定点迭代近似表示梯度近似，实现了无梯度优化，并为此开发了渐近理论。这有效地节省了时间，尤其是在目标量的维度适中时。

    Generalized random forests arXiv:1610.01271 build upon the well-established success of conventional forests (Breiman, 2001) to offer a flexible and powerful non-parametric method for estimating local solutions of heterogeneous estimating equations. Estimators are constructed by leveraging random forests as an adaptive kernel weighting algorithm and implemented through a gradient-based tree-growing procedure. By expressing this gradient-based approximation as being induced from a single Newton-Raphson root-finding iteration, and drawing upon the connection between estimating equations and fixed-point problems arXiv:2110.11074, we propose a new tree-growing rule for generalized random forests induced from a fixed-point iteration type of approximation, enabling gradient-free optimization, and yielding substantial time savings for tasks involving even modest dimensionality of the target quantity (e.g. multiple/multi-level treatment effects). We develop an asymptotic theory for estimators o
    

