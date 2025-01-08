# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Generative Modeling for Tabular Data via Penalized Optimal Transport Network](https://arxiv.org/abs/2402.10456) | 提出了一种名为POTNet的生成建模网络，基于边缘惩罚的Wasserstein损失，能够有效地建模同时包含分类和连续特征的表格数据。 |
| [^2] | [Uncertainty-Aware Partial-Label Learning](https://arxiv.org/abs/2402.00592) | 本文提出了一种基于最近邻的部分标签学习算法，利用Dempster-Shafer理论实现对模糊标记的数据的训练。实验结果表明，该算法能够提供良好的不确定性估计，并具有竞争力的预测性能。 |
| [^3] | [Unexpected Improvements to Expected Improvement for Bayesian Optimization.](http://arxiv.org/abs/2310.20708) | 提出了LogEI作为一类新的贝叶斯优化的获得函数，具有与传统的EI函数相同或近似相等的最优解，但数值上更容易进行优化。 |
| [^4] | [On Penalty-based Bilevel Gradient Descent Method.](http://arxiv.org/abs/2302.05185) | 本文提出了基于惩罚的双层梯度下降算法，解决了下层非强凸约束双层问题，实验表明该算法有效。 |

# 详细

[^1]: 通过惩罚最优输运网络对表格数据进行生成建模

    Generative Modeling for Tabular Data via Penalized Optimal Transport Network

    [https://arxiv.org/abs/2402.10456](https://arxiv.org/abs/2402.10456)

    提出了一种名为POTNet的生成建模网络，基于边缘惩罚的Wasserstein损失，能够有效地建模同时包含分类和连续特征的表格数据。

    

    准确学习表格数据中行的概率分布并生成真实的合成样本的任务既关键又非平凡。Wasserstein生成对抗网络(WGAN)在生成建模中取得了显著进展，解决了其前身生成对抗网络所面临的挑战。然而，由于表格数据中存在混合数据类型和多模态性，生成器和鉴别器之间的微妙平衡以及Wasserstein距离在高维度中的固有不稳定性，WGAN通常无法生成高保真样本。因此，我们提出了POTNet（惩罚最优输运网络），这是一种基于新颖、强大且可解释的边际惩罚Wasserstein（MPW）损失的生成深度神经网络。POTNet能够有效地建模包含分类和连续特征的表格数据。

    arXiv:2402.10456v1 Announce Type: cross  Abstract: The task of precisely learning the probability distribution of rows within tabular data and producing authentic synthetic samples is both crucial and non-trivial. Wasserstein generative adversarial network (WGAN) marks a notable improvement in generative modeling, addressing the challenges faced by its predecessor, generative adversarial network. However, due to the mixed data types and multimodalities prevalent in tabular data, the delicate equilibrium between the generator and discriminator, as well as the inherent instability of Wasserstein distance in high dimensions, WGAN often fails to produce high-fidelity samples. To this end, we propose POTNet (Penalized Optimal Transport Network), a generative deep neural network based on a novel, robust, and interpretable marginally-penalized Wasserstein (MPW) loss. POTNet can effectively model tabular data containing both categorical and continuous features. Moreover, it offers the flexibil
    
[^2]: 不确定性感知的部分标签学习

    Uncertainty-Aware Partial-Label Learning

    [https://arxiv.org/abs/2402.00592](https://arxiv.org/abs/2402.00592)

    本文提出了一种基于最近邻的部分标签学习算法，利用Dempster-Shafer理论实现对模糊标记的数据的训练。实验结果表明，该算法能够提供良好的不确定性估计，并具有竞争力的预测性能。

    

    在现实世界的应用中，人们经常遇到标记模糊的数据，即不同的标注者为相同样本分配了冲突的类别标签。部分标签学习允许在这种弱监督的情况下训练分类器。虽然最先进的方法已经具有良好的预测性能，但它们往往受到错误的不确定性估计的影响。然而，在医学和自动驾驶等安全关键领域，具有良好校准的不确定性估计尤为重要。在本文中，我们提出了一种基于最近邻的部分标签学习算法，该算法利用了Dempster-Shafer理论。对人工数据集和实际数据集进行的广泛实验表明，所提出的方法能够提供良好的不确定性估计，并具有竞争力的预测性能。此外，我们还证明了我们的算法具有风险一致性。

    In real-world applications, one often encounters ambiguously labeled data, where different annotators assign conflicting class labels. Partial-label learning allows training classifiers in this weakly supervised setting. While state-of-the-art methods already feature good predictive performance, they often suffer from miscalibrated uncertainty estimates. However, having well-calibrated uncertainty estimates is important, especially in safety-critical domains like medicine and autonomous driving. In this article, we propose a novel nearest-neighbor-based partial-label-learning algorithm that leverages Dempster-Shafer theory. Extensive experiments on artificial and real-world datasets show that the proposed method provides a well-calibrated uncertainty estimate and achieves competitive prediction performance. Additionally, we prove that our algorithm is risk-consistent.
    
[^3]: 对贝叶斯优化的期望改进的意外提升

    Unexpected Improvements to Expected Improvement for Bayesian Optimization. (arXiv:2310.20708v1 [cs.LG])

    [http://arxiv.org/abs/2310.20708](http://arxiv.org/abs/2310.20708)

    提出了LogEI作为一类新的贝叶斯优化的获得函数，具有与传统的EI函数相同或近似相等的最优解，但数值上更容易进行优化。

    

    期望改进（EI）可以说是贝叶斯优化中最流行的获得函数，并且已经在很多成功的应用中得到了应用。但是，EI的性能往往被一些新方法超越。尤其是，EI及其变种在并行和多目标设置中很难进行优化，因为它们的获得值在许多区域中数值上变为零。当观测次数增加、搜索空间的维度增加或约束条件的数量增加时，这种困难通常会增加，导致性能在文献中不一致且大多数情况下亚优化。在本论文中，我们提出了LogEI，这是一类新的采样函数。与标准EI相比，这些LogEI函数的成员要么具有相同的最优解，要么具有近似相等的最优解，但数值上更容易进行优化。我们证明了数值病态在“经典”分析EI、期望超体积改进（EHVI）以及它们的...

    Expected Improvement (EI) is arguably the most popular acquisition function in Bayesian optimization and has found countless successful applications, but its performance is often exceeded by that of more recent methods. Notably, EI and its variants, including for the parallel and multi-objective settings, are challenging to optimize because their acquisition values vanish numerically in many regions. This difficulty generally increases as the number of observations, dimensionality of the search space, or the number of constraints grow, resulting in performance that is inconsistent across the literature and most often sub-optimal. Herein, we propose LogEI, a new family of acquisition functions whose members either have identical or approximately equal optima as their canonical counterparts, but are substantially easier to optimize numerically. We demonstrate that numerical pathologies manifest themselves in "classic" analytic EI, Expected Hypervolume Improvement (EHVI), as well as their
    
[^4]: 基于惩罚的双层梯度下降方法研究

    On Penalty-based Bilevel Gradient Descent Method. (arXiv:2302.05185v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.05185](http://arxiv.org/abs/2302.05185)

    本文提出了基于惩罚的双层梯度下降算法，解决了下层非强凸约束双层问题，实验表明该算法有效。

    This paper proposes a penalty-based bilevel gradient descent algorithm to solve the constrained bilevel problem without lower-level strong convexity, and experiments show its efficiency.

    双层优化在超参数优化、元学习和强化学习等领域有广泛应用，但是双层优化问题难以解决。最近的可扩展双层算法主要集中在下层目标函数是强凸或无约束的双层优化问题上。在本文中，我们通过惩罚方法来解决双层问题。我们证明，在一定条件下，惩罚重构可以恢复原始双层问题的解。此外，我们提出了基于惩罚的双层梯度下降（PBGD）算法，并证明了其在下层非强凸约束双层问题上的有限时间收敛性。实验展示了所提出的PBGD算法的效率。

    Bilevel optimization enjoys a wide range of applications in hyper-parameter optimization, meta-learning and reinforcement learning. However, bilevel optimization problems are difficult to solve. Recent progress on scalable bilevel algorithms mainly focuses on bilevel optimization problems where the lower-level objective is either strongly convex or unconstrained. In this work, we tackle the bilevel problem through the lens of the penalty method. We show that under certain conditions, the penalty reformulation recovers the solutions of the original bilevel problem. Further, we propose the penalty-based bilevel gradient descent (PBGD) algorithm and establish its finite-time convergence for the constrained bilevel problem without lower-level strong convexity. Experiments showcase the efficiency of the proposed PBGD algorithm.
    

