# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Statistical Guarantees for Variational Autoencoders using PAC-Bayesian Theory.](http://arxiv.org/abs/2310.04935) | 这项工作利用PAC-Bayesian理论为变分自动编码器提供了统计保证，包括对后验分布、重构损失和输入与生成分布之间距离的上界。 |
| [^2] | [Non-Smooth Weakly-Convex Finite-sum Coupled Compositional Optimization.](http://arxiv.org/abs/2310.03234) | 本文研究了一种新的组合优化问题，称为非光滑弱凸有限和耦合组合优化(NSWC FCCO)，通过扩展已有的研究，我们研究了非光滑弱凸FCCO的问题，并提出了一种单循环算法来找到Moreau环的ε-稳定点。 |
| [^3] | [Depthwise Hyperparameter Transfer in Residual Networks: Dynamics and Scaling Limit.](http://arxiv.org/abs/2309.16620) | 这项研究通过残差分支尺度和$\mu$P参数化的残差网络，实现了深度学习中超参数的跨宽度和深度的转移。 |
| [^4] | [Causal normalizing flows: from theory to practice.](http://arxiv.org/abs/2306.05415) | 本文研究了使用因果归一化流进行因果推论的方法，证明了在给定因果排序情况下，利用自回归归一化流可以恢复因果模型。通过实验和比较研究，证明了因果归一化流可用于解决实际问题。 |
| [^5] | [When Does Optimizing a Proper Loss Yield Calibration?.](http://arxiv.org/abs/2305.18764) | 研究优化适当的损失函数是否能在受限的预测器族中得到校准的模型，使用局部最优条件取代全局最优性条件并在此基础上进行了严格的证明。 |
| [^6] | [Loss minimization yields multicalibration for large neural networks.](http://arxiv.org/abs/2304.09424) | 本文展示了对于大型神经网络大小，最优地最小化损失会导致多校准，以提供公平的预测结果。 |
| [^7] | [Construction of Hierarchical Neural Architecture Search Spaces based on Context-free Grammars.](http://arxiv.org/abs/2211.01842) | 本研究基于无上下文文法提出了一个统一的搜索空间设计框架，可以生成表达力强大的分层搜索空间，实现了对整个体系结构的搜索并促进结构的规律性。 |
| [^8] | [Multi-Frequency Joint Community Detection and Phase Synchronization.](http://arxiv.org/abs/2206.12276) | 本文提出了两种简单而高效的算法，利用MLE公式并从多个频率的信息中受益，用于解决具有相对相位的随机块模型上的联合社区检测和相位同步问题。 |
| [^9] | [Predicting Census Survey Response Rates With Parsimonious Additive Models and Structured Interactions.](http://arxiv.org/abs/2108.11328) | 本文提出了一种可解释的非参数加性模型，使用少量主要和成对交互效应预测调查反应率。该模型可以生成易于可视化和解释的预测面，并取得了 ROAM 数据集上的最先进性能，可以提供改进美国人口普查局和其他调查的反应率议论。 |

# 详细

[^1]: 使用PAC-Bayesian理论给变分自动编码器提供统计保证

    Statistical Guarantees for Variational Autoencoders using PAC-Bayesian Theory. (arXiv:2310.04935v1 [cs.LG])

    [http://arxiv.org/abs/2310.04935](http://arxiv.org/abs/2310.04935)

    这项工作利用PAC-Bayesian理论为变分自动编码器提供了统计保证，包括对后验分布、重构损失和输入与生成分布之间距离的上界。

    

    自从它们的问世以来，变分自动编码器（VAEs）在机器学习中变得非常重要。尽管它们被广泛使用，关于它们的理论性质仍存在许多问题。本文利用PAC-Bayesian理论为VAEs提供统计保证。首先，我们推导出了基于独立样本的后验分布的首个PAC-Bayesian界限。然后，利用这一结果为VAE的重构损失提供了泛化保证，同时提供了输入分布与VAE生成模型定义的分布之间距离的上界。更重要的是，我们提供了输入分布与VAE生成模型定义的分布之间Wasserstein距离的上界。

    Since their inception, Variational Autoencoders (VAEs) have become central in machine learning. Despite their widespread use, numerous questions regarding their theoretical properties remain open. Using PAC-Bayesian theory, this work develops statistical guarantees for VAEs. First, we derive the first PAC-Bayesian bound for posterior distributions conditioned on individual samples from the data-generating distribution. Then, we utilize this result to develop generalization guarantees for the VAE's reconstruction loss, as well as upper bounds on the distance between the input and the regenerated distributions. More importantly, we provide upper bounds on the Wasserstein distance between the input distribution and the distribution defined by the VAE's generative model.
    
[^2]: 非光滑弱凸有限和耦合组合优化

    Non-Smooth Weakly-Convex Finite-sum Coupled Compositional Optimization. (arXiv:2310.03234v1 [math.OC])

    [http://arxiv.org/abs/2310.03234](http://arxiv.org/abs/2310.03234)

    本文研究了一种新的组合优化问题，称为非光滑弱凸有限和耦合组合优化(NSWC FCCO)，通过扩展已有的研究，我们研究了非光滑弱凸FCCO的问题，并提出了一种单循环算法来找到Moreau环的ε-稳定点。

    

    本文研究了一类新的组合优化问题，称为非光滑弱凸有限和耦合组合优化(NSWC FCCO)。由于其在机器学习和人工智能领域的广泛应用以及其解决基于经验风险最小化的随机算法的局限性，FCCO引起了越来越多的关注。然而，目前对于FCCO的研究假设内外函数都是光滑的，限制了其能够解决更多种类的问题的潜力。我们的研究从非光滑弱凸FCCO的角度进行了扩展，其中外函数是弱凸且非递减的，内函数是弱凸的。我们分析了一种单循环算法，并确定其在找到Moreau环的ε-稳定点的复杂度。

    This paper investigates new families of compositional optimization problems, called $\underline{\bf n}$on-$\underline{\bf s}$mooth $\underline{\bf w}$eakly-$\underline{\bf c}$onvex $\underline{\bf f}$inite-sum $\underline{\bf c}$oupled $\underline{\bf c}$ompositional $\underline{\bf o}$ptimization (NSWC FCCO). There has been a growing interest in FCCO due to its wide-ranging applications in machine learning and AI, as well as its ability to address the shortcomings of stochastic algorithms based on empirical risk minimization. However, current research on FCCO presumes that both the inner and outer functions are smooth, limiting their potential to tackle a more diverse set of problems. Our research expands on this area by examining non-smooth weakly-convex FCCO, where the outer function is weakly convex and non-decreasing, and the inner function is weakly-convex. We analyze a single-loop algorithm and establish its complexity for finding an $\epsilon$-stationary point of the Moreau env
    
[^3]: 残差网络中的深度超参数转移：动态和缩放限制

    Depthwise Hyperparameter Transfer in Residual Networks: Dynamics and Scaling Limit. (arXiv:2309.16620v1 [stat.ML])

    [http://arxiv.org/abs/2309.16620](http://arxiv.org/abs/2309.16620)

    这项研究通过残差分支尺度和$\mu$P参数化的残差网络，实现了深度学习中超参数的跨宽度和深度的转移。

    

    随着模型大小的增加，深度学习中超参数调整的成本不断上升，促使从业者寻找使用较小网络的代理方法进行调整。其中一个建议使用$\mu$P参数化网络，其中小宽度网络的最佳超参数转移到任意宽度的网络中。然而，在这个方案中，超参数不会在不同深度之间转移。为了解决这个问题，我们研究了具有$1/\sqrt{\text{depth}}$的残差分支尺度和$\mu$P参数化的残差网络。我们通过实验证明，使用这种参数化训练的残差结构，包括卷积ResNet和Vision Transformer，在CIFAR-10和ImageNet上展示了跨宽度和深度的最佳超参数转移。此外，我们的经验发现得到了理论的支持和动机。利用神经网络学习动力学的动态均场理论（DMFT）描述的最新进展，我们展示了

    The cost of hyperparameter tuning in deep learning has been rising with model sizes, prompting practitioners to find new tuning methods using a proxy of smaller networks. One such proposal uses $\mu$P parameterized networks, where the optimal hyperparameters for small width networks transfer to networks with arbitrarily large width. However, in this scheme, hyperparameters do not transfer across depths. As a remedy, we study residual networks with a residual branch scale of $1/\sqrt{\text{depth}}$ in combination with the $\mu$P parameterization. We provide experiments demonstrating that residual architectures including convolutional ResNets and Vision Transformers trained with this parameterization exhibit transfer of optimal hyperparameters across width and depth on CIFAR-10 and ImageNet. Furthermore, our empirical findings are supported and motivated by theory. Using recent developments in the dynamical mean field theory (DMFT) description of neural network learning dynamics, we show
    
[^4]: 因果归一化流：从理论到实践

    Causal normalizing flows: from theory to practice. (arXiv:2306.05415v1 [cs.LG])

    [http://arxiv.org/abs/2306.05415](http://arxiv.org/abs/2306.05415)

    本文研究了使用因果归一化流进行因果推论的方法，证明了在给定因果排序情况下，利用自回归归一化流可以恢复因果模型。通过实验和比较研究，证明了因果归一化流可用于解决实际问题。

    

    本文深入探讨了利用归一化流进行因果推论的应用。具体来说，我们首先利用非线性ICA的最新结果，显示出在给定因果排序的情况下，因果模型可以从观测数据中鉴别出来，并且可以使用自回归归一化流进行恢复。其次，我们分析了用于捕捉潜在因果数据生成过程的不同设计和学习选择的因果归一化流。第三，我们描述了如何在因果归一化流中实现do-operator，从而回答干预和反事实问题。最后，在我们的实验中，我们通过综合对比研究验证了我们的设计和训练选择；将因果归一化流与其他逼近因果模型的方法进行比较；并通过实证研究证明因果归一化流可用于解决现实世界中存在混合离散连续数据和因果图部分知识的问题。本文的代码可以进行访问。

    In this work, we deepen on the use of normalizing flows for causal reasoning. Specifically, we first leverage recent results on non-linear ICA to show that causal models are identifiable from observational data given a causal ordering, and thus can be recovered using autoregressive normalizing flows (NFs). Second, we analyze different design and learning choices for causal normalizing flows to capture the underlying causal data-generating process. Third, we describe how to implement the do-operator in causal NFs, and thus, how to answer interventional and counterfactual questions. Finally, in our experiments, we validate our design and training choices through a comprehensive ablation study; compare causal NFs to other approaches for approximating causal models; and empirically demonstrate that causal NFs can be used to address real-world problems, where the presence of mixed discrete-continuous data and partial knowledge on the causal graph is the norm. The code for this work can be f
    
[^5]: 优化适当的损失函数是否能得到校准的预测器？

    When Does Optimizing a Proper Loss Yield Calibration?. (arXiv:2305.18764v1 [cs.LG])

    [http://arxiv.org/abs/2305.18764](http://arxiv.org/abs/2305.18764)

    研究优化适当的损失函数是否能在受限的预测器族中得到校准的模型，使用局部最优条件取代全局最优性条件并在此基础上进行了严格的证明。

    

    优化适当的损失函数被广泛认为会得到具有良好校准特性的预测器，这是因为对于这样的损失，全局最优解是预测真实概率，这确实是校准的。但是，典型的机器学习模型是训练来近似地最小化在受限制的预测器族中的损失，这些预测器族不太可能包含真实的概率。在什么情况下，优化受限制的预测器族中适当的损失可以得到校准的模型？它提供了什么精确的校准保证？在这项工作中，我们提供了这些问题的严格答案。我们用局部最优条件替换全局最优性条件，该条件规定了预测器（适当的）损失不能通过使用一定族群的Lipschitz函数后处理其预测而降低太多。我们证明了具有这种局部最优性质的任何预测器都满足Kakade-Foster(2008)、Błasiok等人(2023)中定义的平稳校准。

    Optimizing proper loss functions is popularly believed to yield predictors with good calibration properties; the intuition being that for such losses, the global optimum is to predict the ground-truth probabilities, which is indeed calibrated. However, typical machine learning models are trained to approximately minimize loss over restricted families of predictors, that are unlikely to contain the ground truth. Under what circumstances does optimizing proper loss over a restricted family yield calibrated models? What precise calibration guarantees does it give? In this work, we provide a rigorous answer to these questions. We replace the global optimality with a local optimality condition stipulating that the (proper) loss of the predictor cannot be reduced much by post-processing its predictions with a certain family of Lipschitz functions. We show that any predictor with this local optimality satisfies smooth calibration as defined in Kakade-Foster (2008), B{\l}asiok et al. (2023). L
    
[^6]: 大型神经网络的多校准可最小化损失

    Loss minimization yields multicalibration for large neural networks. (arXiv:2304.09424v1 [cs.LG])

    [http://arxiv.org/abs/2304.09424](http://arxiv.org/abs/2304.09424)

    本文展示了对于大型神经网络大小，最优地最小化损失会导致多校准，以提供公平的预测结果。

    

    多校准是一种公平性概念，旨在提供跨大量团体的准确预测。即使对于简单的预测器，如线性函数，多校准也被认为是与最小化损失不同的目标。在本文中，我们展示了对于（几乎所有的）大型神经网络大小，最优地最小化平方误差会导致多校准。我们的结果关于神经网络的表征方面，而不是关于算法或样本复杂性考虑。以前的这样的结果仅适用于几乎贝叶斯最优的预测器，因此是表征无关的。我们强调，我们的结果不适用于优化神经网络的特定算法，如 SGD，并且不应解释为“公平性从优化神经网络中获得免费的好处”。

    Multicalibration is a notion of fairness that aims to provide accurate predictions across a large set of groups. Multicalibration is known to be a different goal than loss minimization, even for simple predictors such as linear functions. In this note, we show that for (almost all) large neural network sizes, optimally minimizing squared error leads to multicalibration. Our results are about representational aspects of neural networks, and not about algorithmic or sample complexity considerations. Previous such results were known only for predictors that were nearly Bayes-optimal and were therefore representation independent. We emphasize that our results do not apply to specific algorithms for optimizing neural networks, such as SGD, and they should not be interpreted as "fairness comes for free from optimizing neural networks".
    
[^7]: 基于无上下文文法的分层神经架构搜索空间构建

    Construction of Hierarchical Neural Architecture Search Spaces based on Context-free Grammars. (arXiv:2211.01842v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2211.01842](http://arxiv.org/abs/2211.01842)

    本研究基于无上下文文法提出了一个统一的搜索空间设计框架，可以生成表达力强大的分层搜索空间，实现了对整个体系结构的搜索并促进结构的规律性。

    

    从简单的构建块中发现神经结构是神经架构搜索(NAS)的一个长期目标。分层搜索空间是实现这一目标的一个有前途的步骤，但缺乏统一的搜索空间设计框架，并且通常仅搜索一些限定方面的架构。在本研究中，我们介绍了一个基于无上下文文法的统一搜索空间设计框架，它可以自然而紧凑地生成表达力强大的分层搜索空间，比文献中常见的空间大几个数量级。通过增强和利用它们的属性，我们有效地实现了对整个体系结构的搜索，并促进了结构的规律性。此外，我们提出了一种高效的分层核设计用于贝叶斯优化搜索策略，以高效搜索如此庞大的空间。我们展示了我们搜索空间设计框架的多样性，并表明我们的搜索策略可以优于现有的NAS方法。

    The discovery of neural architectures from simple building blocks is a long-standing goal of Neural Architecture Search (NAS). Hierarchical search spaces are a promising step towards this goal but lack a unifying search space design framework and typically only search over some limited aspect of architectures. In this work, we introduce a unifying search space design framework based on context-free grammars that can naturally and compactly generate expressive hierarchical search spaces that are 100s of orders of magnitude larger than common spaces from the literature. By enhancing and using their properties, we effectively enable search over the complete architecture and can foster regularity. Further, we propose an efficient hierarchical kernel design for a Bayesian Optimization search strategy to efficiently search over such huge spaces. We demonstrate the versatility of our search space design framework and show that our search strategy can be superior to existing NAS approaches. Co
    
[^8]: 多频联合社区检测和相位同步

    Multi-Frequency Joint Community Detection and Phase Synchronization. (arXiv:2206.12276v2 [cs.SI] UPDATED)

    [http://arxiv.org/abs/2206.12276](http://arxiv.org/abs/2206.12276)

    本文提出了两种简单而高效的算法，利用MLE公式并从多个频率的信息中受益，用于解决具有相对相位的随机块模型上的联合社区检测和相位同步问题。

    This paper proposes two simple and efficient algorithms that leverage the MLE formulation and benefit from the information across multiple frequencies to solve the joint community detection and phase synchronization problem on the stochastic block model with relative phase.

    本文研究了具有相对相位的随机块模型上的联合社区检测和相位同步问题，其中每个节点都与一个未知的相位角相关联。这个问题具有多种实际应用，旨在同时恢复簇结构和相关的相位角。我们通过仔细研究其最大似然估计（MLE）公式，展示了这个问题呈现出“多频”结构，而现有方法并非源于这个角度。为此，提出了两种简单而高效的算法，利用MLE公式并从多个频率的信息中受益。前者是基于新颖的多频列主元QR分解的谱方法。应用于观测矩阵的前几个特征向量的分解提供了有关簇结构和相关相位角的关键信息。第二种方法是迭代的多频率方法。

    This paper studies the joint community detection and phase synchronization problem on the stochastic block model with relative phase, where each node is associated with an unknown phase angle. This problem, with a variety of real-world applications, aims to recover the cluster structure and associated phase angles simultaneously. We show this problem exhibits a ``multi-frequency'' structure by closely examining its maximum likelihood estimation (MLE) formulation, whereas existing methods are not originated from this perspective. To this end, two simple yet efficient algorithms that leverage the MLE formulation and benefit from the information across multiple frequencies are proposed. The former is a spectral method based on the novel multi-frequency column-pivoted QR factorization. The factorization applied to the top eigenvectors of the observation matrix provides key information about the cluster structure and associated phase angles. The second approach is an iterative multi-frequen
    
[^9]: 用简洁可解释的加性模型和结构交互预测人口普查调查反应率

    Predicting Census Survey Response Rates With Parsimonious Additive Models and Structured Interactions. (arXiv:2108.11328v3 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2108.11328](http://arxiv.org/abs/2108.11328)

    本文提出了一种可解释的非参数加性模型，使用少量主要和成对交互效应预测调查反应率。该模型可以生成易于可视化和解释的预测面，并取得了 ROAM 数据集上的最先进性能，可以提供改进美国人口普查局和其他调查的反应率议论。

    

    本文考虑使用一系列灵活且可解释的非参数模型预测调查反应率。本研究受到美国人口普查局著名的 ROAM 应用的启发，该应用使用在美国人口普查规划数据库数据上训练的线性回归模型来识别难以调查的区域。十年前组织的一场众包竞赛表明，基于回归树集成的机器学习方法在预测调查反应率方面表现最佳；然而，由于它们的黑盒特性，相应的模型不能用于拟定的应用。我们考虑使用 $\ell_0$-based 惩罚的非参数加性模型，它具有少数主要和成对交互效应。从方法论的角度来看，我们研究了我们估计器的计算和统计方面，并讨论了将强层次交互合并的变体。我们的算法（在Github 上开源）允许我们生成易于可视化和解释的预测面，从而获得有关调查反应率的可行见解。我们提出的模型在 ROAM 数据集上实现了最先进的性能，并可以提供有关美国人口普查局和其他调查的改进调查反应率的见解。

    In this paper we consider the problem of predicting survey response rates using a family of flexible and interpretable nonparametric models. The study is motivated by the US Census Bureau's well-known ROAM application which uses a linear regression model trained on the US Census Planning Database data to identify hard-to-survey areas. A crowdsourcing competition organized around ten years ago revealed that machine learning methods based on ensembles of regression trees led to the best performance in predicting survey response rates; however, the corresponding models could not be adopted for the intended application due to their black-box nature. We consider nonparametric additive models with small number of main and pairwise interaction effects using $\ell_0$-based penalization. From a methodological viewpoint, we study both computational and statistical aspects of our estimator; and discuss variants that incorporate strong hierarchical interactions. Our algorithms (opensourced on gith
    

