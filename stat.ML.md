# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Self-Play Fine-Tuning of Diffusion Models for Text-to-Image Generation](https://arxiv.org/abs/2402.10210) | 本文介绍了一种创新的技术，称为自我对抗微调扩散模型（SPIN-Diffusion），通过扩散模型与其先前版本的竞争，实现了逐步自我改进过程。 |
| [^2] | [Unlocking the Potential of Transformers in Time Series Forecasting with Sharpness-Aware Minimization and Channel-Wise Attention](https://arxiv.org/abs/2402.10198) | 本文研究了Transformer在时间序列预测中的局限性，发现其注意力机制是泛化能力不足的原因。在此基础上，提出了一个浅层轻量级的Transformer模型SAMformer，通过锐度感知优化避免了陷入坏的局部最小值，并在常用时间序列数据集上超过了当前最先进的模型TSMixer。 |
| [^3] | [Nonlinear spiked covariance matrices and signal propagation in deep neural networks](https://arxiv.org/abs/2402.10127) | 该论文研究了非线性尖峰协方差矩阵与深度神经网络中的信号传播。通过对尖峰特征结构的定量描述，揭示了输入数据中的低维信号结构如何经过神经网络的隐藏层传播。此外，研究了一种表示学习的简单情境，其中权重矩阵发展出一个秩为一的信号分量。 |
| [^4] | [How Much Does Each Datapoint Leak Your Privacy? Quantifying the Per-datum Membership Leakage](https://arxiv.org/abs/2402.10065) | 本论文研究了每个数据点的成员推断攻击，量化了每个数据点的成员泄露，并评估了两种隐私防御措施的效果。 |
| [^5] | [Optimal Parameter and Neuron Pruning for Out-of-Distribution Detection](https://arxiv.org/abs/2402.10062) | 提出了一种用于识别未知分布的最优参数和神经元剪枝方法（OPNP），通过评估模型参数和神经元的敏感性来解决OOD检测的问题。 |
| [^6] | [How to validate average calibration for machine learning regression tasks ?](https://arxiv.org/abs/2402.10043) | 本文提出了两种验证机器学习回归任务平均校准性的方法，将校准误差与平均绝对误差之间的差值和将平均平方z-分数与1进行比较。研究发现，前者对不确定性分布敏感，而后者在该方面提供了最可靠的方法。 |
| [^7] | [Diffusion Models Meet Contextual Bandits with Large Action Spaces](https://arxiv.org/abs/2402.10028) | 本文设计了一种利用预训练扩散模型的扩散汤普森采样方法，用于在大动作空间下进行高效的情境强化学习探索。实证评估结果表明了该方法的优越性能。 |
| [^8] | [Accelerating Parallel Sampling of Diffusion Models](https://arxiv.org/abs/2402.09970) | 本文提出了一种并行化自回归过程来加速扩散模型的采样的方法，并引入了ParaTAA，一种通用的并行采样算法，可以显著减少推理步骤。 |
| [^9] | [FedLion: Faster Adaptive Federated Optimization with Fewer Communication](https://arxiv.org/abs/2402.09941) | FedLion是一种自适应联邦优化算法，通过引入集中式自适应算法Lion的关键元素，实现了更快的收敛速度和更少的通信成本。经过广泛评估，FedLion优于之前的最先进自适应算法，并通过使用有符号梯度在本地训练中减少数据传输要求。 |
| [^10] | [Predictors from causal features do not generalize better to new domains](https://arxiv.org/abs/2402.09891) | 因果特征不能更好地推广到新领域，预测器使用所有特征的效果更好。 |
| [^11] | [Recommendations for Baselines and Benchmarking Approximate Gaussian Processes](https://arxiv.org/abs/2402.09849) | 对于基准线和基准测试近似高斯过程的研究，我们提出了对比方法的建议，并开发了一种训练程序，该程序不需要用户选择，并且证明这是一个符合要求的强大基准。 |
| [^12] | [Two trust region type algorithms for solving nonconvex-strongly concave minimax problems](https://arxiv.org/abs/2402.09807) | 本文提出了两种信赖域类算法，用于解决非凸强凹最小最大问题，并可以在迭代次数为$\mathcal{O}(\epsilon^{-1.5})$内找到二阶稳定点。 |
| [^13] | [Criterion collapse and loss distribution control](https://arxiv.org/abs/2402.09802) | 该论文研究了"准则崩溃"的概念，即优化一个度量指标意味着另一个度量指标的最优性。研究结果发现，对于损失的伯努利分布，CVaR和DRO的结果远超出现有研究，同时发现了一些特定条件下，单调准则如倾斜ERM无法避免崩溃，而非单调的替代方案可以。 |
| [^14] | [Closed-form Filtering for Non-linear Systems](https://arxiv.org/abs/2402.09796) | 提出了一种基于高斯PSD模型的新型滤波器，可以在转换和观测都是高斯PSD模型时以闭式形式高效地进行滤波，并且提出的估计器具有强大的理论保证，适应转换概率的正则性。 |
| [^15] | [Extrapolation-Aware Nonparametric Statistical Inference](https://arxiv.org/abs/2402.09758) | 该论文提出了考虑外推的非参数统计推断方法，并引入了一类外推假设，结合现有推断技术可以得出受外推影响的结论。 |
| [^16] | [Robust SVD Made Easy: A fast and reliable algorithm for large-scale data analysis](https://arxiv.org/abs/2402.09754) | 本研究提出了一种名为球形单位正则化SVD的高效算法，用于鲁棒的SVD逼近，该算法不受异常值干扰，计算可伸缩，并能提供准确的奇异向量逼近。相比竞争算法，该算法仅使用标准降秩SVD算法两次应用于适当缩放的数据，具有显著的计算速度优势。 |
| [^17] | [Best Arm Identification for Prompt Learning under a Limited Budget](https://arxiv.org/abs/2402.09723) | 这项工作提出了一种在提示学习中考虑有限预算约束的方法，通过建立提示学习和多臂赌博机中固定预算最佳臂识别之间的联系，提出了一个通用框架TRIPLE，通过利用聚类和嵌入思想实现了两个增强方法。 |
| [^18] | [Sparse and Faithful Explanations Without Sparse Models](https://arxiv.org/abs/2402.09702) | 引入了稀疏解释值(SEV)，用于衡量机器学习模型的决策稀疏性。即使模型不是稀疏的，许多机器学习模型在SEV的衡量下仍具有低决策稀疏性。 |
| [^19] | [Combining Evidence Across Filtrations](https://arxiv.org/abs/2402.09698) | 这篇论文研究了合并使用不同过滤器计算的e进程的方法，探讨了其在顺序推理中的应用。 |
| [^20] | [Conformalized Adaptive Forecasting of Heterogeneous Trajectories](https://arxiv.org/abs/2402.09623) | 本研究提出了一种新的符合性方法，通过结合在线符合性预测技术和解决回归中异方差性的方法，生成了同时预测边界，并能够可靠地覆盖新随机轨迹的整个路径。这种方法不仅有精确的有限样本保证，而且往往比之前的方法具有更丰富的预测结果。 |
| [^21] | [Exact, Fast and Expressive Poisson Point Processes via Squared Neural Families](https://arxiv.org/abs/2402.09608) | 该论文介绍了使用平方神经网络族的精确、快速和表达性泊松点过程。通过利用两层神经网络的平方范数来参数化强度函数，可以获得更灵活和高效的方法。该方法在计算积分强度函数时具有封闭形式和二次时间复杂度，并且相比于传统方法更节约内存和时间。通过解决凸优化问题，可以获得对强度函数最终层的参数化重参数化的最大似然估计和最大后验估计。 |
| [^22] | [Low-Rank Graph Contrastive Learning for Node Classification](https://arxiv.org/abs/2402.09600) | 本研究提出了一种新颖且鲁棒的低秩图对比学习（LR-GCL）算法，应用于转导节点分类任务。该算法通过低秩正规化的对比学习训练一个编码器，并使用生成的特征进行线性转导分类。 |
| [^23] | [MCMC-driven learning](https://arxiv.org/abs/2402.09598) | 这篇论文旨在统一解决MCMC和机器学习交叉领域的各种问题，包括黑盒变分推断、自适应MCMC、正规流构建和传输辅助MCMC、替代似然MCMC、大数据的MCMC核心集构建等，并提出一个通用的框架。 |
| [^24] | [Distribution-Free Rates in Neyman-Pearson Classification](https://arxiv.org/abs/2402.09560) | 该论文提供了一个关于Neyman-Pearson分类中无分布率的完整特征，通过简单的几何条件，即三点分离条件，刻画了硬分类器和简单分类器之间的二分条件。 |
| [^25] | [Statistical and Machine Learning Models for Predicting Fire and Other Emergency Events](https://arxiv.org/abs/2402.09553) | 本文系统地开发了一种用于预测加拿大埃德蒙顿市不同类型紧急事件的预测模型，并分析了事件类型与邻域层面的社会经济和人口统计数据的关联性。 |
| [^26] | [Oracle-Efficient Differentially Private Learning with Public Data](https://arxiv.org/abs/2402.09483) | 这项研究提出了一种具有公共数据的计算高效算法，可以在满足差分隐私条件的情况下学习私有数据，以提高学习算法性能。 |
| [^27] | [One-for-many Counterfactual Explanations by Column Generation](https://arxiv.org/abs/2402.09473) | 本文提出了一个列生成框架，用于解决一对多反事实解释的问题。框架通过限制每个解释中可集体改变的特征数量，旨在尽可能少地使用解释来解释所有实例。相比于现有的混合整数规划方法，该框架在可扩展性、计算性能和解决方案质量方面具有优势。 |
| [^28] | [Rolling Diffusion Models](https://arxiv.org/abs/2402.09470) | 本文介绍了一种滚动扩散模型，用于处理时间数据，通过滑动窗口去噪并根据帧在序列中的时间先后分配不同的噪声量，更好地捕捉到复杂的时间动态。通过实验证明，在视频预测和混沌流体动力学预测任务中，该模型优于传统扩散方法。 |
| [^29] | [Fourier Circuits in Neural Networks: Unlocking the Potential of Large Language Models in Mathematical Reasoning and Modular Arithmetic](https://arxiv.org/abs/2402.09469) | 本研究探索了神经网络和Transformer在数学推理和模运算中的潜力。我们分析了单隐藏层神经网络和单层Transformer在解决复杂代数学习任务中的特征。阐明了边缘最大化原则对单隐藏层神经网络的影响。 |
| [^30] | [Optimal Thresholding Linear Bandit](https://arxiv.org/abs/2402.09467) | 本论文研究了具有固定置信度的随机线性赌博机的ε-阈值赌博机问题，并提出了一种在渐近意义上是最优的算法。 |
| [^31] | [Optimistic Thompson Sampling for No-Regret Learning in Unknown Games](https://arxiv.org/abs/2402.09456) | 该论文提出了一种在未知博弈中进行无遗憾学习的乐观的汤普森抽样方法，通过利用对手的行动和奖励结构信息，显著减少了实验预算，成功地缓解了多机构问题。此外，研究还引入了乐观-无遗憾框架，将现有算法与提出的方法相结合。 |
| [^32] | [Towards Robust Model-Based Reinforcement Learning Against Adversarial Corruption](https://arxiv.org/abs/2402.08991) | 本研究通过引入对抗性健壮的乐观MLE（CR-OMLE）算法，解决了模型驱动强化学习中对抗性破坏的挑战，实现了对转移模型的健壮估计。 |
| [^33] | [Correction to "Wasserstein distance estimates for the distributions of numerical approximations to ergodic stochastic differential equations"](https://arxiv.org/abs/2402.08711) | 修正了《对于数值逼近遍历SDE的分布的Wasserstein距离估计》中的错误局部误差估计，提出了一种方法来分析数值离散遍历SDE的Wasserstein-2距离的非渐近保证，并解决了实践中维度依赖性的问题。 |
| [^34] | [Sequential Flow Matching for Generative Modeling](https://arxiv.org/abs/2402.06461) | 本文提出了一种称为SeqRF的新方法，用于通过直线化概率流来减小全局截断误差，并以此加速取样和提高综合质量。 |
| [^35] | [On Computational Limits of Modern Hopfield Models: A Fine-Grained Complexity Analysis](https://arxiv.org/abs/2402.04520) | 通过细粒度复杂性分析，我们研究了现代Hopfield模型的记忆检索计算限制，发现了一种基于模式范数的相变行为，并且建立了有效变体的上界条件。使用低秩逼近的方法，我们提供了有效构造的示例，同时证明了计算时间下界、记忆检索误差界和指数记忆容量。 |
| [^36] | [Empirical Comparison between Cross-Validation and Mutation-Validation in Model Selection](https://arxiv.org/abs/2311.14079) | 本研究通过对比基准和实际数据集，实证比较了突变验证（MV）和交叉验证（CV）在模型选择中的表现。结果发现，MV和CV在选择模型的泛化性能方面基本等效，但MV在选择简单模型和计算成本方面具有优势。 |
| [^37] | [Differentially Private Graph Learning via Sensitivity-Bounded Personalized PageRank](https://arxiv.org/abs/2207.06944) | 本论文提出了一种敏感性有界的个性化PageRank算法，能够保护用户隐私。该算法在保持准确性的同时，实现了差分隐私图学习的几种工具。 |
| [^38] | [Fast and explainable clustering based on sorting](https://arxiv.org/abs/2202.01456) | CLASSIX是一种快速可解释的聚类算法，它通过排序后的数据的贪婪聚合和群组合并来进行聚类。该算法具有与最先进的聚类算法相媲美的性能，并且具有线性空间复杂性和近线性时间复杂性。 |
| [^39] | [Structure by Architecture: Structured Representations without Regularization](https://arxiv.org/abs/2006.07796) | 我们提出了一种自我监督的结构化表示学习方法，使用无需正则化的自动编码器架构。通过依赖潜变量的独立性进行采样，我们避免了重构质量和生成性能之间的权衡。我们的模型能够学习出一种有序的结构化表示，改善了生成、解缠和外推等多个下游任务的性能。 |
| [^40] | [Causal Similarity-Based Hierarchical Bayesian Models.](http://arxiv.org/abs/2310.12595) | 本文提出了一种基于因果相似性的分层贝叶斯模型，通过学习如何从具有相似因果机制的训练任务中汇集数据来提高机器学习算法对新任务的泛化能力。 |
| [^41] | [Unlabeled Out-Of-Domain Data Improves Generalization.](http://arxiv.org/abs/2310.00027) | 这个论文提出了一种新的框架，可以将无标记的域外数据纳入半监督分类问题，从而改善泛化能力。该框架结合了分布鲁棒优化与自监督训练，并利用了高效的多项式时间算法。在理论上，该框架在高斯混合分类问题中得到了验证。 |
| [^42] | [Interactive and Concentrated Differential Privacy for Bandits.](http://arxiv.org/abs/2309.00557) | 本文研究了在交互学习和推荐系统中隐私保护的Bandit问题，并引入了集中差分隐私的概念。通过提供关于有限臂和线性Bandit问题遗憾的下界，我们揭示了不同隐私预算下的难度区域，并发现集中差分隐私可以比全局差分隐私更有效地保护隐私，我们提出了两种相应的算法。 |
| [^43] | [Normalization Is All You Need: Understanding Layer-Normalized Federated Learning under Extreme Label Shift.](http://arxiv.org/abs/2308.09565) | 本论文揭示了层归一化和联邦学习中的标签偏移问题之间的深刻联系，通过在联邦学习中应用特征归一化，使得对严重倾斜的数据集进行加速全局训练，从而在极端标签偏移下获得显著改进。 |
| [^44] | [A/B Testing and Best-arm Identification for Linear Bandits with Robustness to Non-stationarity.](http://arxiv.org/abs/2307.15154) | 本文研究了在非稳态环境中的线性赌博机的最佳臂识别问题，提出了一种具有鲁棒性的算法来解决。该算法通过在每个时间步从一个G-最优设计中随机选择臂来实现最佳臂的鲁棒识别。 |
| [^45] | [Privacy Amplification via Importance Sampling.](http://arxiv.org/abs/2307.10187) | 通过重要性采样进行隐私放大，可以同时增强隐私保护和提高效用。我们提供了一个一般的结果来量化选择概率权重对隐私放大的影响，并展示了异质采样概率可以在保持子采样大小不变的情况下获得更好的隐私和效用。 |
| [^46] | [Stabilized Neural Differential Equations for Learning Constrained Dynamics.](http://arxiv.org/abs/2306.09739) | 本文提出了一种稳定神经微分方程（SNDEs）的方法，可以强制使用任意流形约束。该方法通过添加稳定项使约束流形成为渐进稳定的，并且在实验中表现优于现有方法。 |
| [^47] | [Improved Stability and Generalization Analysis of the Decentralized SGD Algorithm.](http://arxiv.org/abs/2306.02939) | 本文提出了新的算法稳定性理论来改进分布式SGD算法的泛化性能分析，推翻了现有技术对通信图负面影响的观点，并展示了D-SGD在凸设置中与经典SGD算法泛化界相同。 |
| [^48] | [Self-Correcting Bayesian Optimization through Bayesian Active Learning.](http://arxiv.org/abs/2304.11005) | 该论文提出了SAL和SCoreBO两种方法，用于提高高斯过程模型的超参数选择和贝叶斯优化的表现。 |
| [^49] | [Bayesian inference on Brain-Computer Interface using the GLASS Model.](http://arxiv.org/abs/2304.07401) | 本文针对P300 BCI问题，开发了一种基于GLASS模型的贝叶斯推断方法，直接解决了脑机接口应用中数据集不平衡问题，具有良好的分类性能和易于解释性。 |
| [^50] | [Inverse Solvability and Security with Applications to Federated Learning.](http://arxiv.org/abs/2211.14115) | 介绍了逆可解性和安全性的概念，以及其在联邦学习中的应用。论文提供了模型示例，展示了如何通过增加用户数量来增加可解性和安全性。 |

# 详细

[^1]: 自我对抗微调扩散模型用于文本到图像生成

    Self-Play Fine-Tuning of Diffusion Models for Text-to-Image Generation

    [https://arxiv.org/abs/2402.10210](https://arxiv.org/abs/2402.10210)

    本文介绍了一种创新的技术，称为自我对抗微调扩散模型（SPIN-Diffusion），通过扩散模型与其先前版本的竞争，实现了逐步自我改进过程。

    

    微调扩散模型在生成人工智能领域仍然是一个未被充分探索的前沿，尤其是与在大型语言模型（LLMs）微调方面取得的显著进展相比。尽管现在的先进扩散模型如稳定扩散（SD）和SDXL依赖于监督微调，但它们的性能在观察到一定数量的数据后必然会达到瓶颈。最近，强化学习（RL）被应用于通过人类偏好数据对扩散模型进行微调，但每个文本提示需要至少两个图像（“获胜者”和“失败者”图像）。本文介绍了一种创新的技术，称为自我对抗微调扩散模型（SPIN-Diffusion），其中扩散模型与其先前版本进行竞争，促进了一个迭代的自我改进过程。我们的方法提供了一种替代传统监督微调和RL策略的选择。

    arXiv:2402.10210v1 Announce Type: cross  Abstract: Fine-tuning Diffusion Models remains an underexplored frontier in generative artificial intelligence (GenAI), especially when compared with the remarkable progress made in fine-tuning Large Language Models (LLMs). While cutting-edge diffusion models such as Stable Diffusion (SD) and SDXL rely on supervised fine-tuning, their performance inevitably plateaus after seeing a certain volume of data. Recently, reinforcement learning (RL) has been employed to fine-tune diffusion models with human preference data, but it requires at least two images ("winner" and "loser" images) for each text prompt. In this paper, we introduce an innovative technique called self-play fine-tuning for diffusion models (SPIN-Diffusion), where the diffusion model engages in competition with its earlier versions, facilitating an iterative self-improvement process. Our approach offers an alternative to conventional supervised fine-tuning and RL strategies, signific
    
[^2]: 使用锐度感知最小化和通道注意力解锁Transformer在时间序列预测中的潜力

    Unlocking the Potential of Transformers in Time Series Forecasting with Sharpness-Aware Minimization and Channel-Wise Attention

    [https://arxiv.org/abs/2402.10198](https://arxiv.org/abs/2402.10198)

    本文研究了Transformer在时间序列预测中的局限性，发现其注意力机制是泛化能力不足的原因。在此基础上，提出了一个浅层轻量级的Transformer模型SAMformer，通过锐度感知优化避免了陷入坏的局部最小值，并在常用时间序列数据集上超过了当前最先进的模型TSMixer。

    

    Transformer架构在自然语言处理和计算机视觉中取得了突破性的性能，但在多元长期预测方面，它们仍然不如更简单的线性基线。为了更好地理解这一现象，我们首先研究了一个玩具线性预测问题，展示了尽管Transformer具有高表达能力，但它们无法收敛到真正的解决方案。我们进一步确定Transformer的注意力是造成其低泛化能力的原因。基于这一认识，我们提出了一个浅层轻量级的Transformer模型，在锐度感知优化的情况下成功避免了坏的局部最小值。我们通过实验证明，这个结果适用于所有常用的实际多元时间序列数据集。特别是，相比当前最先进的模型TSMixer，SAMformer的平均性能提高了14.33%，并且参数数量减少了约4倍。

    arXiv:2402.10198v1 Announce Type: new  Abstract: Transformer-based architectures achieved breakthrough performance in natural language processing and computer vision, yet they remain inferior to simpler linear baselines in multivariate long-term forecasting. To better understand this phenomenon, we start by studying a toy linear forecasting problem for which we show that transformers are incapable of converging to their true solution despite their high expressive power. We further identify the attention of transformers as being responsible for this low generalization capacity. Building upon this insight, we propose a shallow lightweight transformer model that successfully escapes bad local minima when optimized with sharpness-aware optimization. We empirically demonstrate that this result extends to all commonly used real-world multivariate time series datasets. In particular, SAMformer surpasses the current state-of-the-art model TSMixer by 14.33% on average, while having ~4 times few
    
[^3]: 非线性尖峰协方差矩阵与深度神经网络中的信号传播

    Nonlinear spiked covariance matrices and signal propagation in deep neural networks

    [https://arxiv.org/abs/2402.10127](https://arxiv.org/abs/2402.10127)

    该论文研究了非线性尖峰协方差矩阵与深度神经网络中的信号传播。通过对尖峰特征结构的定量描述，揭示了输入数据中的低维信号结构如何经过神经网络的隐藏层传播。此外，研究了一种表示学习的简单情境，其中权重矩阵发展出一个秩为一的信号分量。

    

    许多最近的研究都研究了由前馈神经网络的非线性特征映射定义的共轭核（CK）的特征值谱。然而，现有的结果只能建立经验特征值分布的弱收敛性，并没有提供对通常捕捉学习问题的低维信号结构的“尖峰”特征值和特征向量的精确定量描述。在这项工作中，我们对非线性版本的尖峰协方差模型（包括CK作为特例）进行了这些信号特征值和特征向量的表征。利用这个一般结果，我们定量描述了输入数据中的尖峰特征结构如何通过具有随机权重的神经网络的隐藏层传播。作为第二个应用，我们研究了表示学习的一个简单情境，其中权重矩阵在训练过程中发展出一个秩为一的信号分量。

    arXiv:2402.10127v1 Announce Type: cross  Abstract: Many recent works have studied the eigenvalue spectrum of the Conjugate Kernel (CK) defined by the nonlinear feature map of a feedforward neural network. However, existing results only establish weak convergence of the empirical eigenvalue distribution, and fall short of providing precise quantitative characterizations of the ''spike'' eigenvalues and eigenvectors that often capture the low-dimensional signal structure of the learning problem. In this work, we characterize these signal eigenvalues and eigenvectors for a nonlinear version of the spiked covariance model, including the CK as a special case. Using this general result, we give a quantitative description of how spiked eigenstructure in the input data propagates through the hidden layers of a neural network with random weights. As a second application, we study a simple regime of representation learning where the weight matrix develops a rank-one signal component over trainin
    
[^4]: 每个数据点泄露您隐私的程度有多大？量化每个数据点的成员泄露

    How Much Does Each Datapoint Leak Your Privacy? Quantifying the Per-datum Membership Leakage

    [https://arxiv.org/abs/2402.10065](https://arxiv.org/abs/2402.10065)

    本论文研究了每个数据点的成员推断攻击，量化了每个数据点的成员泄露，并评估了两种隐私防御措施的效果。

    

    我们研究了每个数据点的成员推断攻击（MIAs），其中攻击者旨在推断出一个固定目标数据是否已包含在算法的输入数据集中，从而侵犯隐私。首先，我们定义每个数据点的成员泄露为最优对手辨识它的优势。然后，我们量化了经验均值的每个数据点的成员泄露，并表明它取决于目标数据点和数据生成分布之间的马氏距离。我们进一步评估了两种隐私防御措施的效果，即添加高斯噪声和子采样。我们准确地量化了它们都如何降低每个数据点的成员泄露。我们的分析建立在一个结合了似然比检验的Edgeworth展开和Lindeberg-Feller中心极限定理的新型证明技术上。我们的分析连接了现有的似然比和标量乘积攻击，并对这些攻击进行了论证。

    arXiv:2402.10065v1 Announce Type: new  Abstract: We study the per-datum Membership Inference Attacks (MIAs), where an attacker aims to infer whether a fixed target datum has been included in the input dataset of an algorithm and thus, violates privacy. First, we define the membership leakage of a datum as the advantage of the optimal adversary targeting to identify it. Then, we quantify the per-datum membership leakage for the empirical mean, and show that it depends on the Mahalanobis distance between the target datum and the data-generating distribution. We further assess the effect of two privacy defences, i.e. adding Gaussian noise and sub-sampling. We quantify exactly how both of them decrease the per-datum membership leakage. Our analysis builds on a novel proof technique that combines an Edgeworth expansion of the likelihood ratio test and a Lindeberg-Feller central limit theorem. Our analysis connects the existing likelihood ratio and scalar product attacks, and also justifies 
    
[^5]: 用于识别未知分布的最优参数和神经元剪枝方法

    Optimal Parameter and Neuron Pruning for Out-of-Distribution Detection

    [https://arxiv.org/abs/2402.10062](https://arxiv.org/abs/2402.10062)

    提出了一种用于识别未知分布的最优参数和神经元剪枝方法（OPNP），通过评估模型参数和神经元的敏感性来解决OOD检测的问题。

    

    对于在现实场景中部署的机器学习模型，识别未知分布（OOD）样本的能力是不可或缺且具有挑战性的。大多数已有的OOD检测方法关注于探索高级训练技巧或训练无关的技巧，以防止模型对未知样本产生过于自信的置信度分数。基于训练的方法需要昂贵的训练成本，并且依赖于并非始终可用的OOD样本，而大多数基于训练无关的方法无法有效利用训练数据的先验信息。在这项工作中，我们提出了一种名为OPNP（Optimal Parameter and Neuron Pruning）的方法，旨在识别并删除导致过度拟合的参数和神经元。主要方法分为两个步骤。在第一步中，我们通过对所有训练样本进行梯度平均来评估模型参数和神经元的敏感性。

    arXiv:2402.10062v1 Announce Type: new  Abstract: For a machine learning model deployed in real world scenarios, the ability of detecting out-of-distribution (OOD) samples is indispensable and challenging. Most existing OOD detection methods focused on exploring advanced training skills or training-free tricks to prevent the model from yielding overconfident confidence score for unknown samples. The training-based methods require expensive training cost and rely on OOD samples which are not always available, while most training-free methods can not efficiently utilize the prior information from the training data. In this work, we propose an \textbf{O}ptimal \textbf{P}arameter and \textbf{N}euron \textbf{P}runing (\textbf{OPNP}) approach, which aims to identify and remove those parameters and neurons that lead to over-fitting. The main method is divided into two steps. In the first step, we evaluate the sensitivity of the model parameters and neurons by averaging gradients over all train
    
[^6]: 如何验证机器学习回归任务的平均校准性？

    How to validate average calibration for machine learning regression tasks ?

    [https://arxiv.org/abs/2402.10043](https://arxiv.org/abs/2402.10043)

    本文提出了两种验证机器学习回归任务平均校准性的方法，将校准误差与平均绝对误差之间的差值和将平均平方z-分数与1进行比较。研究发现，前者对不确定性分布敏感，而后者在该方面提供了最可靠的方法。

    

    机器学习回归任务的平均校准性可以通过两种方式进行测试。一种方式是将校准误差（CE）估计为平均绝对误差（MSE）与平均方差（MV）或平均平方不确定性之间的差值。另一种方式是将平均平方z-分数或缩放误差（ZMS）与1进行比较。两种方法可能得出不同的结论，正如来自最近的机器学习不确定性量化文献中的数据集集合所示。研究表明，CE对不确定性分布非常敏感，特别是对于离群不确定性的存在，因此无法可靠地用于校准测试。相比之下，ZMS统计量不具有这种敏感性问题，在这种情况下提供了最可靠的方法。文章还讨论了对条件校准验证的影响。

    arXiv:2402.10043v1 Announce Type: cross  Abstract: Average calibration of the uncertainties of machine learning regression tasks can be tested in two ways. One way is to estimate the calibration error (CE) as the difference between the mean absolute error (MSE) and the mean variance (MV) or mean squared uncertainty. The alternative is to compare the mean squared z-scores or scaled errors (ZMS) to 1. Both approaches might lead to different conclusion, as illustrated on an ensemble of datasets from the recent machine learning uncertainty quantification literature. It is shown here that the CE is very sensitive to the distribution of uncertainties, and notably to the presence of outlying uncertainties, and that it cannot be used reliably for calibration testing. By contrast, the ZMS statistic does not present this sensitivity issue and offers the most reliable approach in this context. Implications for the validation of conditional calibration are discussed.
    
[^7]: 扩散模型与大动作空间情境强化学习的结合

    Diffusion Models Meet Contextual Bandits with Large Action Spaces

    [https://arxiv.org/abs/2402.10028](https://arxiv.org/abs/2402.10028)

    本文设计了一种利用预训练扩散模型的扩散汤普森采样方法，用于在大动作空间下进行高效的情境强化学习探索。实证评估结果表明了该方法的优越性能。

    

    由于动作空间较大，有效的探索是情境强化学习中的一个关键挑战。本文通过利用预训练的扩散模型来捕捉动作之间的相关性，设计了扩散汤普森采样（dTS）方法，实现了高效的探索。我们为dTS方法提供了理论和算法基础，并通过实证评估展示了它的优越性能。

    arXiv:2402.10028v1 Announce Type: cross  Abstract: Efficient exploration is a key challenge in contextual bandits due to the large size of their action space, where uninformed exploration can result in computational and statistical inefficiencies. Fortunately, the rewards of actions are often correlated and this can be leveraged to explore them efficiently. In this work, we capture such correlations using pre-trained diffusion models; upon which we design diffusion Thompson sampling (dTS). Both theoretical and algorithmic foundations are developed for dTS, and empirical evaluation also shows its favorable performance.
    
[^8]: 加速并行采样扩散模型

    Accelerating Parallel Sampling of Diffusion Models

    [https://arxiv.org/abs/2402.09970](https://arxiv.org/abs/2402.09970)

    本文提出了一种并行化自回归过程来加速扩散模型的采样的方法，并引入了ParaTAA，一种通用的并行采样算法，可以显著减少推理步骤。

    

    扩散模型已经成为图像生成的最先进生成模型。然而，由于其采样过程中固有的自回归性质，从扩散模型中进行采样通常耗时。在本文中，我们提出了一种新的方法，通过并行化自回归过程来加速扩散模型的采样。具体而言，我们将采样过程重新构建为通过固定点迭代解决三角非线性方程组的过程。通过这种创新的公式，我们探索了一些系统化的技术，进一步减少了求解过程所需的迭代步骤。应用这些技术，我们引入了ParaTAA，一种通用的、无需训练的并行采样算法，可以利用额外的计算和内存资源来增加采样速度。我们的实验表明，ParaTAA可以减少常见的顺序采样所需的推理步骤。

    arXiv:2402.09970v1 Announce Type: new  Abstract: Diffusion models have emerged as state-of-the-art generative models for image generation. However, sampling from diffusion models is usually time-consuming due to the inherent autoregressive nature of their sampling process. In this work, we propose a novel approach that accelerates the sampling of diffusion models by parallelizing the autoregressive process. Specifically, we reformulate the sampling process as solving a system of triangular nonlinear equations through fixed-point iteration. With this innovative formulation, we explore several systematic techniques to further reduce the iteration steps required by the solving process. Applying these techniques, we introduce ParaTAA, a universal and training-free parallel sampling algorithm that can leverage extra computational and memory resources to increase the sampling speed. Our experiments demonstrate that ParaTAA can decrease the inference steps required by common sequential sampli
    
[^9]: FedLion: 更快的自适应联邦优化算法，通信更少

    FedLion: Faster Adaptive Federated Optimization with Fewer Communication

    [https://arxiv.org/abs/2402.09941](https://arxiv.org/abs/2402.09941)

    FedLion是一种自适应联邦优化算法，通过引入集中式自适应算法Lion的关键元素，实现了更快的收敛速度和更少的通信成本。经过广泛评估，FedLion优于之前的最先进自适应算法，并通过使用有符号梯度在本地训练中减少数据传输要求。

    

    在联邦学习（FL）中，一种跨分布式数据训练机器学习模型的框架中，像FedAvg这样的知名算法往往具有较慢的收敛速度，在训练过程中导致高通信成本。为了解决这个挑战，我们引入了FedLion，一种自适应联邦优化算法，无缝地将最近提出的集中式自适应算法Lion（Chen et al. 2023）的关键元素融入到FL框架中。通过对两个广泛采用的FL基准进行全面评估，我们证明了FedLion优于之前的最先进自适应算法，包括FAFED（Wu et al. 2023）和FedDA。此外，由于在本地训练中使用了有符号梯度，与现有的自适应算法相比，FedLion在上行通信过程中大大降低了数据传输要求，进一步降低了通信成本。

    arXiv:2402.09941v1 Announce Type: cross  Abstract: In Federated Learning (FL), a framework to train machine learning models across distributed data, well-known algorithms like FedAvg tend to have slow convergence rates, resulting in high communication costs during training. To address this challenge, we introduce FedLion, an adaptive federated optimization algorithm that seamlessly incorporates key elements from the recently proposed centralized adaptive algorithm, Lion (Chen et al. 2o23), into the FL framework. Through comprehensive evaluations on two widely adopted FL benchmarks, we demonstrate that FedLion outperforms previous state-of-the-art adaptive algorithms, including FAFED (Wu et al. 2023) and FedDA. Moreover, thanks to the use of signed gradients in local training, FedLion substantially reduces data transmission requirements during uplink communication when compared to existing adaptive algorithms, further reducing communication costs. Last but not least, this work also incl
    
[^10]: 预测因果特征不能更好地推广到新领域

    Predictors from causal features do not generalize better to new domains

    [https://arxiv.org/abs/2402.09891](https://arxiv.org/abs/2402.09891)

    因果特征不能更好地推广到新领域，预测器使用所有特征的效果更好。

    

    我们研究了在不同领域中，基于因果特征训练的机器学习模型的泛化效果。我们考虑了涵盖健康、就业、教育、社会福利和政治等应用的16个表格数据集的预测任务。每个数据集都有多个领域，我们可以测试一个在一个领域训练的模型在另一个领域的表现。对于每个预测任务，我们选择对预测目标有因果影响的特征。我们的目标是测试基于因果特征训练的模型是否在不同领域中更好地泛化。我们发现，无论是否具有因果关系，使用所有可用特征的预测器都比使用因果特征的预测器在领域内外的准确性更高。而且，即使是从一个领域到另一个领域的准确性绝对下降对于因果预测器来说也不比使用所有特征的模型更好。如果目标是在新领域中泛化，实践中使用所有特征的预测器效果更好。

    arXiv:2402.09891v1 Announce Type: new  Abstract: We study how well machine learning models trained on causal features generalize across domains. We consider 16 prediction tasks on tabular datasets covering applications in health, employment, education, social benefits, and politics. Each dataset comes with multiple domains, allowing us to test how well a model trained in one domain performs in another. For each prediction task, we select features that have a causal influence on the target of prediction. Our goal is to test the hypothesis that models trained on causal features generalize better across domains. Without exception, we find that predictors using all available features, regardless of causality, have better in-domain and out-of-domain accuracy than predictors using causal features. Moreover, even the absolute drop in accuracy from one domain to the other is no better for causal predictors than for models that use all features. If the goal is to generalize to new domains, prac
    
[^11]: 对于基准线和基准测试近似高斯过程的建议

    Recommendations for Baselines and Benchmarking Approximate Gaussian Processes

    [https://arxiv.org/abs/2402.09849](https://arxiv.org/abs/2402.09849)

    对于基准线和基准测试近似高斯过程的研究，我们提出了对比方法的建议，并开发了一种训练程序，该程序不需要用户选择，并且证明这是一个符合要求的强大基准。

    

    Gaussian processes (GPs)是机器学习工具箱中成熟且广泛使用的组件。它们具有自动超参数选择的优点，可以实现无需用户干预的训练。然而，在许多现实情况下，通常需要使用近似方法，而这些方法通常需要调整。我们认为，这种调整要求使得评估变得复杂，这导致缺乏对在哪种情况下使用哪种方法的明确建议。为了解决这个问题，我们提出了对比GP近似方法的建议，基于用户对方法的期望的规范。此外，我们开发了一种训练程序，用于Titsias [2009]的变分方法，该方法不需要用户选择，并且证明这是符合我们规范的一个强大基准。我们得出结论，按照我们的建议进行基准测试可以更清晰地了解当前领域的状态，并发现……

    arXiv:2402.09849v1 Announce Type: new  Abstract: Gaussian processes (GPs) are a mature and widely-used component of the ML toolbox. One of their desirable qualities is automatic hyperparameter selection, which allows for training without user intervention. However, in many realistic settings, approximations are typically needed, which typically do require tuning. We argue that this requirement for tuning complicates evaluation, which has led to a lack of a clear recommendations on which method should be used in which situation. To address this, we make recommendations for comparing GP approximations based on a specification of what a user should expect from a method. In addition, we develop a training procedure for the variational method of Titsias [2009] that leaves no choices to the user, and show that this is a strong baseline that meets our specification. We conclude that benchmarking according to our suggestions gives a clearer view of the current state of the field, and uncovers 
    
[^12]: 解决非凸强凹最小最大问题的两种信赖域类算法

    Two trust region type algorithms for solving nonconvex-strongly concave minimax problems

    [https://arxiv.org/abs/2402.09807](https://arxiv.org/abs/2402.09807)

    本文提出了两种信赖域类算法，用于解决非凸强凹最小最大问题，并可以在迭代次数为$\mathcal{O}(\epsilon^{-1.5})$内找到二阶稳定点。

    

    在本文中，我们提出了解决非凸强凹最小最大问题的最小最大信赖域（MINIMAX-TR）算法和具有收缩和扩张的最小最大信赖域算法（MINIMAX-TRACE）。这两种算法可以在$\mathcal{O}(\epsilon^{-1.5})$次迭代内找到$(\epsilon, \sqrt{\epsilon})$-二阶稳定点(SSP)，这与已知最好的迭代复杂度相匹配。

    arXiv:2402.09807v1 Announce Type: cross  Abstract: In this paper, we propose a Minimax Trust Region (MINIMAX-TR) algorithm and a Minimax Trust Region Algorithm with Contractions and Expansions(MINIMAX-TRACE) algorithm for solving nonconvex-strongly concave minimax problems. Both algorithms can find an $(\epsilon, \sqrt{\epsilon})$-second order stationary point(SSP) within $\mathcal{O}(\epsilon^{-1.5})$ iterations, which matches the best well known iteration complexity.
    
[^13]: 准则崩溃和损失分布控制

    Criterion collapse and loss distribution control

    [https://arxiv.org/abs/2402.09802](https://arxiv.org/abs/2402.09802)

    该论文研究了"准则崩溃"的概念，即优化一个度量指标意味着另一个度量指标的最优性。研究结果发现，对于损失的伯努利分布，CVaR和DRO的结果远超出现有研究，同时发现了一些特定条件下，单调准则如倾斜ERM无法避免崩溃，而非单调的替代方案可以。

    

    在这项工作中，我们考虑了"准则崩溃"的概念，即优化一个度量指标意味着另一个度量指标的最优性，特别关注各种学习准则下崩溃成误差概率最小化器的条件，从DRO和OCE风险（CVaR、倾斜ERM）到文献中探索的最新上升-下降算法的非单调准则（洪水、SoftAD）。我们展示了在伯努利分布损失的背景下，CVaR和DRO的现有结果远远超越了崩溃的范围，然后扩大了我们的范围，包括代理损失，展示了像倾斜ERM这样的单调准则无法避免崩溃的条件，而非单调的替代方案可以。

    arXiv:2402.09802v1 Announce Type: cross  Abstract: In this work, we consider the notion of "criterion collapse," in which optimization of one metric implies optimality in another, with a particular focus on conditions for collapse into error probability minimizers under a wide variety of learning criteria, ranging from DRO and OCE risks (CVaR, tilted ERM) to non-monotonic criteria underlying recent ascent-descent algorithms explored in the literature (Flooding, SoftAD). We show how collapse in the context of losses with a Bernoulli distribution goes far beyond existing results for CVaR and DRO, then expand our scope to include surrogate losses, showing conditions where monotonic criteria such as tilted ERM cannot avoid collapse, whereas non-monotonic alternatives can.
    
[^14]: 闭式滤波器在非线性系统中的应用

    Closed-form Filtering for Non-linear Systems

    [https://arxiv.org/abs/2402.09796](https://arxiv.org/abs/2402.09796)

    提出了一种基于高斯PSD模型的新型滤波器，可以在转换和观测都是高斯PSD模型时以闭式形式高效地进行滤波，并且提出的估计器具有强大的理论保证，适应转换概率的正则性。

    

    顺序贝叶斯滤波旨在估计隐藏马尔可夫模型的当前状态分布，给定过去的观测值。对于大多数应用领域来说，这个问题是难以解决的，除了像表格设置或具有高斯噪声的线性动力系统这样的明显情况。在这项工作中，我们提出了一种基于高斯PSD模型的新型滤波器，它在密度近似和计算效率方面具有多个优势。我们展示了当转换和观测都是高斯PSD模型时，滤波可以以闭式形式高效地进行。当转换和观测被高斯PSD模型近似时，我们证明了我们提出的估计器具有强大的理论保证，估计误差取决于近似的质量，并且适应转换概率的正则性。特别是，我们确定了我们的方法在某些情况下的适用范围，其中我们可以以闭式形式高效地进行滤波。

    arXiv:2402.09796v1 Announce Type: cross  Abstract: Sequential Bayesian Filtering aims to estimate the current state distribution of a Hidden Markov Model, given the past observations. The problem is well-known to be intractable for most application domains, except in notable cases such as the tabular setting or for linear dynamical systems with gaussian noise. In this work, we propose a new class of filters based on Gaussian PSD Models, which offer several advantages in terms of density approximation and computational efficiency. We show that filtering can be efficiently performed in closed form when transitions and observations are Gaussian PSD Models. When the transition and observations are approximated by Gaussian PSD Models, we show that our proposed estimator enjoys strong theoretical guarantees, with estimation error that depends on the quality of the approximation and is adaptive to the regularity of the transition probabilities. In particular, we identify regimes in which our 
    
[^15]: 考虑外推的非参数统计推断

    Extrapolation-Aware Nonparametric Statistical Inference

    [https://arxiv.org/abs/2402.09758](https://arxiv.org/abs/2402.09758)

    该论文提出了考虑外推的非参数统计推断方法，并引入了一类外推假设，结合现有推断技术可以得出受外推影响的结论。

    

    我们将外推定义为对超出条件变量支持范围的条件函数（例如条件期望或条件分位数）进行的任何类型的统计推断。这种外推类型在许多数据分析应用中都出现，并且如果不考虑它们可能会使得结果的结论失效。尽管在参数模型中外推是直接的，但在非参数模型中却具有挑战性。在这项工作中，我们将非参数统计模型扩展到明确允许外推，并引入一类可以与现有推断技术结合使用的外推假设，以得出受外推影响的结论。提出的外推假设类规定，条件函数在观察到的支持范围内的每个方向上都达到其最小和最大方向导数。我们演示了该框架如何应用于几个实例。

    arXiv:2402.09758v1 Announce Type: cross  Abstract: We define extrapolation as any type of statistical inference on a conditional function (e.g., a conditional expectation or conditional quantile) evaluated outside of the support of the conditioning variable. This type of extrapolation occurs in many data analysis applications and can invalidate the resulting conclusions if not taken into account. While extrapolating is straightforward in parametric models, it becomes challenging in nonparametric models. In this work, we extend the nonparametric statistical model to explicitly allow for extrapolation and introduce a class of extrapolation assumptions that can be combined with existing inference techniques to draw extrapolation-aware conclusions. The proposed class of extrapolation assumptions stipulate that the conditional function attains its minimal and maximal directional derivative, in each direction, within the observed support. We illustrate how the framework applies to several st
    
[^16]: Robust SVD变得简单：一种用于大规模数据分析的快速可靠算法

    Robust SVD Made Easy: A fast and reliable algorithm for large-scale data analysis

    [https://arxiv.org/abs/2402.09754](https://arxiv.org/abs/2402.09754)

    本研究提出了一种名为球形单位正则化SVD的高效算法，用于鲁棒的SVD逼近，该算法不受异常值干扰，计算可伸缩，并能提供准确的奇异向量逼近。相比竞争算法，该算法仅使用标准降秩SVD算法两次应用于适当缩放的数据，具有显著的计算速度优势。

    

    奇异值分解（SVD）是机器学习和统计数据分析中的重要工具。然而，它对数据矩阵中的异常值非常敏感。现有的鲁棒SVD算法往往在保证鲁棒性方面牺牲了速度，或者在只有少数异常值存在时失效。本研究介绍了一种高度不受异常值干扰，计算可伸缩且提供准确奇异向量逼近的高效算法，称为球形单位正则化SVD。该算法通过仅使用标准降秩SVD算法的两个应用于适当缩放的数据，实现了显著的计算速度优势，明显优于竞争算法的计算时间。为了评估逼近奇异向量及其子空间的抗数据污染能力，我们引入了矩阵值输入的新的失效点概念，包括逐行，c

    arXiv:2402.09754v1 Announce Type: new  Abstract: The singular value decomposition (SVD) is a crucial tool in machine learning and statistical data analysis. However, it is highly susceptible to outliers in the data matrix. Existing robust SVD algorithms often sacrifice speed for robustness or fail in the presence of only a few outliers. This study introduces an efficient algorithm, called Spherically Normalized SVD, for robust SVD approximation that is highly insensitive to outliers, computationally scalable, and provides accurate approximations of singular vectors. The proposed algorithm achieves remarkable speed by utilizing only two applications of a standard reduced-rank SVD algorithm to appropriately scaled data, significantly outperforming competing algorithms in computation times. To assess the robustness of the approximated singular vectors and their subspaces against data contamination, we introduce new notions of breakdown points for matrix-valued input, including row-wise, c
    
[^17]: 有限预算下的迅速学习最佳臂识别

    Best Arm Identification for Prompt Learning under a Limited Budget

    [https://arxiv.org/abs/2402.09723](https://arxiv.org/abs/2402.09723)

    这项工作提出了一种在提示学习中考虑有限预算约束的方法，通过建立提示学习和多臂赌博机中固定预算最佳臂识别之间的联系，提出了一个通用框架TRIPLE，通过利用聚类和嵌入思想实现了两个增强方法。

    

    大型语言模型（LLMs）的显著指令跟随能力引发了对自动学习合适提示的兴趣。然而，虽然提出了许多有效的方法，但在学习过程中产生的成本（例如访问LLM和评估响应）尚未得到考虑。为克服这个限制，本工作在提示学习中明确引入了有限预算约束。为了开发有原则的解决方案，本研究在提示学习和多臂赌博机的固定预算最佳臂识别（BAI-FB）之间建立了一种新的联系。基于这种联系，提出了一个通用框架TRIPLE（用于提示学习的最佳臂识别），以系统地利用BAI-FB在提示学习中的力量。提示学习的独特特点进一步通过利用聚类和嵌入思想提出了TRIPLE的两个基于嵌入的增强方法。

    arXiv:2402.09723v1 Announce Type: cross  Abstract: The remarkable instruction-following capability of large language models (LLMs) has sparked a growing interest in automatically learning suitable prompts. However, while many effective methods have been proposed, the cost incurred during the learning process (e.g., accessing LLM and evaluating the responses) has not been considered. To overcome this limitation, this work explicitly incorporates a finite budget constraint into prompt learning. Towards developing principled solutions, a novel connection is established between prompt learning and fixed-budget best arm identification (BAI-FB) in multi-armed bandits (MAB). Based on this connection, a general framework TRIPLE (besT aRm Identification for Prompt LEarning) is proposed to harness the power of BAI-FB in prompt learning systematically. Unique characteristics of prompt learning further lead to two embedding-based enhancements of TRIPLE by exploiting the ideas of clustering and fun
    
[^18]: 无需稀疏模型的稀疏且准确的解释

    Sparse and Faithful Explanations Without Sparse Models

    [https://arxiv.org/abs/2402.09702](https://arxiv.org/abs/2402.09702)

    引入了稀疏解释值(SEV)，用于衡量机器学习模型的决策稀疏性。即使模型不是稀疏的，许多机器学习模型在SEV的衡量下仍具有低决策稀疏性。

    

    即使模型不满足全局的稀疏性，决策仍然可以用少量的特征准确地描述。例如，对于某人而言，尽管没有信用历史，但申请大笔贷款可能会被拒绝，这就忽视了与其信用价值相关的任何证据。在本论文中，我们引入了稀疏解释值（SEV），这是一种衡量机器学习模型稀疏性的新方法。在以上贷款拒绝的例子中，SEV为1，因为只需要一个因素来解释为什么贷款被拒绝。SEV是对决策稀疏性的衡量，而不是对整体模型稀疏性的衡量，并且我们能够证明许多机器学习模型——即使它们不是稀疏的——实际上在SEV的衡量下具有低决策稀疏性。SEV使用超立方体上的移动进行定义，使得SEV能够在各种模型类别上一致地定义，其中移动限制反映了模型的性质。

    arXiv:2402.09702v1 Announce Type: new  Abstract: Even if a model is not globally sparse, it is possible for decisions made from that model to be accurately and faithfully described by a small number of features. For instance, an application for a large loan might be denied to someone because they have no credit history, which overwhelms any evidence towards their creditworthiness. In this work, we introduce the Sparse Explanation Value (SEV), a new way of measuring sparsity in machine learning models. In the loan denial example above, the SEV is 1 because only one factor is needed to explain why the loan was denied. SEV is a measure of decision sparsity rather than overall model sparsity, and we are able to show that many machine learning models -- even if they are not sparse -- actually have low decision sparsity, as measured by SEV. SEV is defined using movements over a hypercube, allowing SEV to be defined consistently over various model classes, with movement restrictions reflectin
    
[^19]: 合并不同过滤器中的证据

    Combining Evidence Across Filtrations

    [https://arxiv.org/abs/2402.09698](https://arxiv.org/abs/2402.09698)

    这篇论文研究了合并使用不同过滤器计算的e进程的方法，探讨了其在顺序推理中的应用。

    

    在任何时刻有效的顺序推理中，已知任何可接受的推理方法必须基于测试鞅和它们的组合广义化，称为e进程，它们是非负进程，其在任何任意停时的期望上界不超过一。e进程量化了针对复合零假设的一系列结果的累积证据。本文研究了使用不同信息集（即过滤器）计算的e进程的合并方法，针对一个零假设。尽管在相同过滤器上构建的e进程可以轻松地合并（例如，通过平均），但在不同过滤器上构建的e进程不能那么容易地合并，因为它们在较粗的过滤器中的有效性不能转换为在更细的过滤器中的有效性。我们讨论了文献中三个具体例子：可交换性测试，独立性测试等。

    arXiv:2402.09698v1 Announce Type: cross  Abstract: In anytime-valid sequential inference, it is known that any admissible inference procedure must be based on test martingales and their composite generalization, called e-processes, which are nonnegative processes whose expectation at any arbitrary stopping time is upper-bounded by one. An e-process quantifies the accumulated evidence against a composite null hypothesis over a sequence of outcomes. This paper studies methods for combining e-processes that are computed using different information sets, i.e., filtrations, for a null hypothesis. Even though e-processes constructed on the same filtration can be combined effortlessly (e.g., by averaging), e-processes constructed on different filtrations cannot be combined as easily because their validity in a coarser filtration does not translate to validity in a finer filtration. We discuss three concrete examples of such e-processes in the literature: exchangeability tests, independence te
    
[^20]: 多元轨迹的符合性自适应预测方法

    Conformalized Adaptive Forecasting of Heterogeneous Trajectories

    [https://arxiv.org/abs/2402.09623](https://arxiv.org/abs/2402.09623)

    本研究提出了一种新的符合性方法，通过结合在线符合性预测技术和解决回归中异方差性的方法，生成了同时预测边界，并能够可靠地覆盖新随机轨迹的整个路径。这种方法不仅有精确的有限样本保证，而且往往比之前的方法具有更丰富的预测结果。

    

    本文提出了一种新的符合性方法，用于生成同时预测边界，以具有足够高的概率覆盖新随机轨迹的整个路径。鉴于在运动规划应用中需要可靠的不确定性估计，其中不同物体的行为可能更或更少可预测，我们将来自单个和多个时间序列的在线符合性预测技术，以及解决回归中的异方差性的方法进行了融合。该解决方案既有原则性，提供了精确的有限样本保证，又有效，通常比先前的方法具有更丰富的预测结果。

    arXiv:2402.09623v1 Announce Type: cross  Abstract: This paper presents a new conformal method for generating simultaneous forecasting bands guaranteed to cover the entire path of a new random trajectory with sufficiently high probability. Prompted by the need for dependable uncertainty estimates in motion planning applications where the behavior of diverse objects may be more or less unpredictable, we blend different techniques from online conformal prediction of single and multiple time series, as well as ideas for addressing heteroscedasticity in regression. This solution is both principled, providing precise finite-sample guarantees, and effective, often leading to more informative predictions than prior methods.
    
[^21]: 使用平方神经网络族的精确、快速和表达性泊松点过程

    Exact, Fast and Expressive Poisson Point Processes via Squared Neural Families

    [https://arxiv.org/abs/2402.09608](https://arxiv.org/abs/2402.09608)

    该论文介绍了使用平方神经网络族的精确、快速和表达性泊松点过程。通过利用两层神经网络的平方范数来参数化强度函数，可以获得更灵活和高效的方法。该方法在计算积分强度函数时具有封闭形式和二次时间复杂度，并且相比于传统方法更节约内存和时间。通过解决凸优化问题，可以获得对强度函数最终层的参数化重参数化的最大似然估计和最大后验估计。

    

    我们通过将强度函数的参数化为两层神经网络的平方范数引入了平方神经泊松点过程（SNEPPPs）。当隐藏层被固定且第二层只有一个神经元时，我们的方法类似于之前使用平方高斯过程或核方法，但允许隐藏层学习能够提供额外的灵活性。在许多感兴趣的情况下，积分强度函数可以得到封闭形式，并且可以以二次时间相对于隐藏神经元的数量进行计算。我们列举了比以前讨论过的更多这样的情况。我们的方法比简单实现平方或指数核方法或高斯过程更节约内存和时间。最大似然和最大后验估计可以通过解决（严格）凸优化问题来获得强度函数最终层的参数化重参数化。

    arXiv:2402.09608v1 Announce Type: new  Abstract: We introduce squared neural Poisson point processes (SNEPPPs) by parameterising the intensity function by the squared norm of a two layer neural network. When the hidden layer is fixed and the second layer has a single neuron, our approach resembles previous uses of squared Gaussian process or kernel methods, but allowing the hidden layer to be learnt allows for additional flexibility. In many cases of interest, the integrated intensity function admits a closed form and can be computed in quadratic time in the number of hidden neurons. We enumerate a far more extensive number of such cases than has previously been discussed. Our approach is more memory and time efficient than naive implementations of squared or exponentiated kernel methods or Gaussian processes. Maximum likelihood and maximum a posteriori estimates in a reparameterisation of the final layer of the intensity function can be obtained by solving a (strongly) convex optimisa
    
[^22]: 低秩图对比学习用于节点分类

    Low-Rank Graph Contrastive Learning for Node Classification

    [https://arxiv.org/abs/2402.09600](https://arxiv.org/abs/2402.09600)

    本研究提出了一种新颖且鲁棒的低秩图对比学习（LR-GCL）算法，应用于转导节点分类任务。该算法通过低秩正规化的对比学习训练一个编码器，并使用生成的特征进行线性转导分类。

    

    图神经网络（GNNs）广泛应用于学习节点表示，并在节点分类等各种任务中表现出色。然而，最近的研究表明，在现实世界的图数据中不可避免地存在噪声，这会严重降低GNNs的性能。在本文中，我们提出了一种新颖且鲁棒的GNN编码器，即低秩图对比学习（LR-GCL）。我们的方法通过两个步骤进行转导节点分类。首先，通过低秩正常对比学习训练一个名为LR-GCL的低秩GCL编码器。然后，使用LR-GCL生成的特征，使用线性转导分类算法对图中的未标记节点进行分类。我们的LR-GCL受到图数据和其标签的低频性质的启示，并在理论上受到我们关于转导学习的尖锐泛化界限的推动。

    arXiv:2402.09600v1 Announce Type: new  Abstract: Graph Neural Networks (GNNs) have been widely used to learn node representations and with outstanding performance on various tasks such as node classification. However, noise, which inevitably exists in real-world graph data, would considerably degrade the performance of GNNs revealed by recent studies. In this work, we propose a novel and robust GNN encoder, Low-Rank Graph Contrastive Learning (LR-GCL). Our method performs transductive node classification in two steps. First, a low-rank GCL encoder named LR-GCL is trained by prototypical contrastive learning with low-rank regularization. Next, using the features produced by LR-GCL, a linear transductive classification algorithm is used to classify the unlabeled nodes in the graph. Our LR-GCL is inspired by the low frequency property of the graph data and its labels, and it is also theoretically motivated by our sharp generalization bound for transductive learning. To the best of our kno
    
[^23]: 基于MCMC的学习

    MCMC-driven learning

    [https://arxiv.org/abs/2402.09598](https://arxiv.org/abs/2402.09598)

    这篇论文旨在统一解决MCMC和机器学习交叉领域的各种问题，包括黑盒变分推断、自适应MCMC、正规流构建和传输辅助MCMC、替代似然MCMC、大数据的MCMC核心集构建等，并提出一个通用的框架。

    

    这篇论文旨在作为《马尔科夫链蒙特卡罗手册》的一章出现。该章的目标是在马尔科夫链蒙特卡罗（MCMC）和机器学习之间的交叉点上统一各种问题，其中包括黑盒变分推断、自适应MCMC、正规流构建和传输辅助MCMC、替代似然MCMC、用于大数据的MCMC核心集构建、马尔科夫链梯度下降、马尔科夫得分攀爬等。通过这样做，可以将为每个问题开发的理论和方法进行翻译和推广。

    arXiv:2402.09598v1 Announce Type: cross  Abstract: This paper is intended to appear as a chapter for the Handbook of Markov Chain Monte Carlo. The goal of this chapter is to unify various problems at the intersection of Markov chain Monte Carlo (MCMC) and machine learning$\unicode{x2014}$which includes black-box variational inference, adaptive MCMC, normalizing flow construction and transport-assisted MCMC, surrogate-likelihood MCMC, coreset construction for MCMC with big data, Markov chain gradient descent, Markovian score climbing, and more$\unicode{x2014}$within one common framework. By doing so, the theory and methods developed for each may be translated and generalized.
    
[^24]: Neyman-Pearson分类中的无分布率

    Distribution-Free Rates in Neyman-Pearson Classification

    [https://arxiv.org/abs/2402.09560](https://arxiv.org/abs/2402.09560)

    该论文提供了一个关于Neyman-Pearson分类中无分布率的完整特征，通过简单的几何条件，即三点分离条件，刻画了硬分类器和简单分类器之间的二分条件。

    

    我们考虑Neyman-Pearson分类问题，该问题模拟了不平衡分类设置，在这种设置中，最小化与分布$\mu_1$相关的错误，同时保证与另一个分布$\mu_0$相关的错误较低。给定一个固定的VC分类器类$\mathcal{H}$，我们提供了可能的无分布率的完整特征，即所有配对$(\mu_0, \mu_1)$的极小化率。这些速率涉及到了硬分类器和简单分类器之间的二分条件，它们是根据一个简单的几何条件，即三点分离条件来刻画的，与VC维度略有关联。

    arXiv:2402.09560v1 Announce Type: new  Abstract: We consider the problem of Neyman-Pearson classification which models unbalanced classification settings where error w.r.t. a distribution $\mu_1$ is to be minimized subject to low error w.r.t. a different distribution $\mu_0$. Given a fixed VC class $\mathcal{H}$ of classifiers to be minimized over, we provide a full characterization of possible distribution-free rates, i.e., minimax rates over the space of all pairs $(\mu_0, \mu_1)$. The rates involve a dichotomy between hard and easy classes $\mathcal{H}$ as characterized by a simple geometric condition, a three-points-separation condition, loosely related to VC dimension.
    
[^25]: 统计与机器学习模型用于预测火灾和其他紧急事件

    Statistical and Machine Learning Models for Predicting Fire and Other Emergency Events

    [https://arxiv.org/abs/2402.09553](https://arxiv.org/abs/2402.09553)

    本文系统地开发了一种用于预测加拿大埃德蒙顿市不同类型紧急事件的预测模型，并分析了事件类型与邻域层面的社会经济和人口统计数据的关联性。

    

    城市中的紧急事件给个人、家庭和社区都带来了相当大的经济损失。准确和及时地预测事件可以帮助应急消防和救援部门为和减轻紧急事件的后果做好准备。在本文中，我们系统地开发了一种针对加拿大埃德蒙顿市不同类型紧急事件的预测模型。我们提出了以下方法：（i）数据收集和数据集开发；（ii）对不同时空级别的每种事件类型及其特征进行描述性分析；（iii）基于相关系数分析和特征重要性分析的特征分析和选择；（iv）针对不同时空分辨率开发每种事件类型发生可能性的预测模型。我们分析了事件类型与邻域层面的社会经济和人口统计数据的关联性。

    arXiv:2402.09553v1 Announce Type: new  Abstract: Emergency events in a city cause considerable economic loss to individuals, their families, and the community. Accurate and timely prediction of events can help the emergency fire and rescue services in preparing for and mitigating the consequences of emergency events. In this paper, we present a systematic development of predictive models for various types of emergency events in the City of Edmonton, Canada. We present methods for (i) data collection and dataset development; (ii) descriptive analysis of each event type and its characteristics at different spatiotemporal levels; (iii) feature analysis and selection based on correlation coefficient analysis and feature importance analysis; and (iv) development of prediction models for the likelihood of occurrence of each event type at different temporal and spatial resolutions. We analyze the association of event types with socioeconomic and demographic data at the neighborhood level, ide
    
[^26]: 具有公共数据的Oracle-Efficient差分隐私学习

    Oracle-Efficient Differentially Private Learning with Public Data

    [https://arxiv.org/abs/2402.09483](https://arxiv.org/abs/2402.09483)

    这项研究提出了一种具有公共数据的计算高效算法，可以在满足差分隐私条件的情况下学习私有数据，以提高学习算法性能。

    

    由于在隐私约束下许多函数类的可学习性的统计下限，最近出现了利用公共数据提高私有学习算法性能的兴趣。在这种模型中，算法必须始终保证相对于私有样本的差分隐私，并在私有数据分布与公共数据分布足够接近时确保学习保证。先前的研究表明，当有足够的公共非标记数据时，可以使私有学习在统计上可以处理，但得到的算法都是计算效率低下的。在这项工作中，我们提出了第一种可计算高效的算法，可以在函数类可非私有学习时明确利用公共数据进行私有学习，其中我们对计算效率的概念是相对于优化调用次数的。

    arXiv:2402.09483v1 Announce Type: cross  Abstract: Due to statistical lower bounds on the learnability of many function classes under privacy constraints, there has been recent interest in leveraging public data to improve the performance of private learning algorithms. In this model, algorithms must always guarantee differential privacy with respect to the private samples while also ensuring learning guarantees when the private data distribution is sufficiently close to that of the public data. Previous work has demonstrated that when sufficient public, unlabelled data is available, private learning can be made statistically tractable, but the resulting algorithms have all been computationally inefficient. In this work, we present the first computationally efficient, algorithms to provably leverage public data to learn privately whenever a function class is learnable non-privately, where our notion of computational efficiency is with respect to the number of calls to an optimization o
    
[^27]: 列生成的一对多反事实解释

    One-for-many Counterfactual Explanations by Column Generation

    [https://arxiv.org/abs/2402.09473](https://arxiv.org/abs/2402.09473)

    本文提出了一个列生成框架，用于解决一对多反事实解释的问题。框架通过限制每个解释中可集体改变的特征数量，旨在尽可能少地使用解释来解释所有实例。相比于现有的混合整数规划方法，该框架在可扩展性、计算性能和解决方案质量方面具有优势。

    

    在本文中，我们考虑了一个问题，即如何生成一组针对一组实例的反事实解释，采用一对多分配规则，其中一个解释被分配给一个实例子组。我们首次解决了在考虑稀疏性的情况下最小化解释所需数量的问题，通过限制每个解释中允许集体改变的特征数量。我们开发了一个新颖的列生成框架，用于高效搜索解释。我们的框架可以应用于任何黑盒分类器，如神经网络。与文献中的简单混合整数规划公式的简单适应相比，列生成框架在可扩展性、计算性能和解决方案的质量方面占优势。

    arXiv:2402.09473v1 Announce Type: new  Abstract: In this paper, we consider the problem of generating a set of counterfactual explanations for a group of instances, with the one-for-many allocation rule, where one explanation is allocated to a subgroup of the instances. For the first time, we solve the problem of minimizing the number of explanations needed to explain all the instances, while considering sparsity by limiting the number of features allowed to be changed collectively in each explanation. A novel column generation framework is developed to efficiently search for the explanations. Our framework can be applied to any black-box classifier, like neural networks. Compared with a simple adaptation of a mixed-integer programming formulation from the literature, the column generation framework dominates in terms of scalability, computational performance and quality of the solutions.
    
[^28]: 滚动扩散模型

    Rolling Diffusion Models

    [https://arxiv.org/abs/2402.09470](https://arxiv.org/abs/2402.09470)

    本文介绍了一种滚动扩散模型，用于处理时间数据，通过滑动窗口去噪并根据帧在序列中的时间先后分配不同的噪声量，更好地捕捉到复杂的时间动态。通过实验证明，在视频预测和混沌流体动力学预测任务中，该模型优于传统扩散方法。

    

    最近，扩散模型越来越多地应用于时间数据，如视频、流体力学模拟或气候数据。这些方法通常将后续帧在扩散过程中的噪声量视为相等。本文探讨了滚动扩散：一种使用滑动窗口去噪的新方法。它确保扩散过程逐渐通过时间进行破坏，通过将更多的噪声分配给序列中出现较晚的帧，反映出随着生成过程的展开，对未来的不确定性越来越大。通过实证研究，我们表明当时间动态复杂时，滚动扩散优于标准扩散。特别是在使用Kinetics-600视频数据集进行视频预测任务和混沌流体动力学预测实验中证明了这一结果。

    arXiv:2402.09470v1 Announce Type: new  Abstract: Diffusion models have recently been increasingly applied to temporal data such as video, fluid mechanics simulations, or climate data. These methods generally treat subsequent frames equally regarding the amount of noise in the diffusion process. This paper explores Rolling Diffusion: a new approach that uses a sliding window denoising process. It ensures that the diffusion process progressively corrupts through time by assigning more noise to frames that appear later in a sequence, reflecting greater uncertainty about the future as the generation process unfolds. Empirically, we show that when the temporal dynamics are complex, Rolling Diffusion is superior to standard diffusion. In particular, this result is demonstrated in a video prediction task using the Kinetics-600 video dataset and in a chaotic fluid dynamics forecasting experiment.
    
[^29]: 神经网络中的傅立叶电路：解锁大规模语言模型在数学推理和模运算中的潜力

    Fourier Circuits in Neural Networks: Unlocking the Potential of Large Language Models in Mathematical Reasoning and Modular Arithmetic

    [https://arxiv.org/abs/2402.09469](https://arxiv.org/abs/2402.09469)

    本研究探索了神经网络和Transformer在数学推理和模运算中的潜力。我们分析了单隐藏层神经网络和单层Transformer在解决复杂代数学习任务中的特征。阐明了边缘最大化原则对单隐藏层神经网络的影响。

    

    在机器学习不断发展的背景下，理解神经网络和Transformer所利用的内部表示是一个关键挑战。本研究在近期的研究基础上，对网络采用特定计算策略背后的原因进行了探索。我们的研究聚焦于涉及k个输入的复杂代数学习任务，即模运算的加法。我们对单隐藏层神经网络和单层Transformer在解决这一任务中学到的特征进行了深入的分析。我们理论框架的一个关键是阐明边缘最大化原则对单隐藏层神经网络采用的特征的影响。其中，p表示模数，Dp表示k个输入的模运算数据集，m表示网络输出。

    arXiv:2402.09469v1 Announce Type: new  Abstract: In the evolving landscape of machine learning, a pivotal challenge lies in deciphering the internal representations harnessed by neural networks and Transformers. Building on recent progress toward comprehending how networks execute distinct target functions, our study embarks on an exploration of the underlying reasons behind networks adopting specific computational strategies. We direct our focus to the complex algebraic learning task of modular addition involving $k$ inputs. Our research presents a thorough analytical characterization of the features learned by stylized one-hidden layer neural networks and one-layer Transformers in addressing this task.   A cornerstone of our theoretical framework is the elucidation of how the principle of margin maximization shapes the features adopted by one-hidden layer neural networks. Let $p$ denote the modulus, $D_p$ denote the dataset of modular arithmetic with $k$ inputs and $m$ denote the net
    
[^30]: 最优阈值线性赌博机

    Optimal Thresholding Linear Bandit

    [https://arxiv.org/abs/2402.09467](https://arxiv.org/abs/2402.09467)

    本论文研究了具有固定置信度的随机线性赌博机的ε-阈值赌博机问题，并提出了一种在渐近意义上是最优的算法。

    

    我们研究了一种新颖的纯探索问题：具有固定置信度的随机线性赌博机的ε-阈值赌博机问题(TBP)。我们证明了样本复杂性的下界，并将设计用于线性情况下的最佳臂识别算法扩展到了TBP，该算法在渐近意义上是最优的。

    arXiv:2402.09467v1 Announce Type: cross  Abstract: We study a novel pure exploration problem: the $\epsilon$-Thresholding Bandit Problem (TBP) with fixed confidence in stochastic linear bandits. We prove a lower bound for the sample complexity and extend an algorithm designed for Best Arm Identification in the linear case to TBP that is asymptotically optimal.
    
[^31]: 未知博弈中乐观的汤普森抽样方法用于无遗憾学习

    Optimistic Thompson Sampling for No-Regret Learning in Unknown Games

    [https://arxiv.org/abs/2402.09456](https://arxiv.org/abs/2402.09456)

    该论文提出了一种在未知博弈中进行无遗憾学习的乐观的汤普森抽样方法，通过利用对手的行动和奖励结构信息，显著减少了实验预算，成功地缓解了多机构问题。此外，研究还引入了乐观-无遗憾框架，将现有算法与提出的方法相结合。

    

    许多涉及多个决策者的真实世界问题可以建模为一个具有部分观测的未知博弈。为了解决部分信息和多机构的挑战，我们开发了汤普森抽样类型的算法，利用对手的行动和奖励结构的信息。我们的方法在实际应用中，如交通路由和雷达感知中，显著减少了实验预算，与基准算法相比，减少了十倍以上。我们证明，在对奖励结构有一定假设的情况下，遗憾界限仅对总行动空间大小呈对数依赖，有效缓解了多机构问题。此外，本研究引入了乐观-无遗憾框架，该框架将我们提出的方法和领域内现有的算法相结合，是一项新的贡献。

    arXiv:2402.09456v1 Announce Type: cross  Abstract: Many real-world problems involving multiple decision-makers can be modeled as an unknown game characterized by partial observations. Addressing the challenges posed by partial information and the curse of multi-agency, we developed Thompson sampling-type algorithms, leveraging information about opponent's action and reward structures. Our approach significantly reduces experimental budgets, achieving a more than tenfold reduction compared to baseline algorithms in practical applications like traffic routing and radar sensing. We demonstrate that, under certain assumptions about the reward structure, the regret bound exhibits merely a logarithmic dependence on the total action space size, effectively mitigating the curse of multi-agency. Additionally, this research introduces the Optimism-then-NoRegret framework, a novel contribution that integrates both our proposed methodologies and existing algorithms in the field.
    
[^32]: 面向对抗性破坏的健壮模型驱动强化学习

    Towards Robust Model-Based Reinforcement Learning Against Adversarial Corruption

    [https://arxiv.org/abs/2402.08991](https://arxiv.org/abs/2402.08991)

    本研究通过引入对抗性健壮的乐观MLE（CR-OMLE）算法，解决了模型驱动强化学习中对抗性破坏的挑战，实现了对转移模型的健壮估计。

    

    本研究解决了模型驱动强化学习中对抗性破坏的挑战，其中转移动力学可以被对手破坏。现有研究主要集中在模型无关强化学习的情景下，通常采用健壮的最小二乘回归来进行值函数估计。然而，这些技术不能直接应用于模型驱动的强化学习。在本文中，我们专注于模型驱动的强化学习，并采用最大似然估计（MLE）方法来学习转移模型。我们的工作涵盖了在线和离线两种情况。在在线情况下，我们引入了一种名为对抗性健壮的乐观MLE（CR-OMLE）的算法，它利用基于总变差（TV）的信息比率作为MLE的不确定权重。我们证明了CR-OMLE的遗憾度为$ \tilde {\mathcal {O}}（\sqrt {T} + C）$，其中$ C $表示经过$ T $个回合后的累计破坏水平。

    arXiv:2402.08991v1 Announce Type: cross Abstract: This study tackles the challenges of adversarial corruption in model-based reinforcement learning (RL), where the transition dynamics can be corrupted by an adversary. Existing studies on corruption-robust RL mostly focus on the setting of model-free RL, where robust least-square regression is often employed for value function estimation. However, these techniques cannot be directly applied to model-based RL. In this paper, we focus on model-based RL and take the maximum likelihood estimation (MLE) approach to learn transition model. Our work encompasses both online and offline settings. In the online setting, we introduce an algorithm called corruption-robust optimistic MLE (CR-OMLE), which leverages total-variation (TV)-based information ratios as uncertainty weights for MLE. We prove that CR-OMLE achieves a regret of $\tilde{\mathcal{O}}(\sqrt{T} + C)$, where $C$ denotes the cumulative corruption level after $T$ episodes. We also pro
    
[^33]: 《对于数值逼近遍历SDE的分布的Wasserstein距离估计》修正

    Correction to "Wasserstein distance estimates for the distributions of numerical approximations to ergodic stochastic differential equations"

    [https://arxiv.org/abs/2402.08711](https://arxiv.org/abs/2402.08711)

    修正了《对于数值逼近遍历SDE的分布的Wasserstein距离估计》中的错误局部误差估计，提出了一种方法来分析数值离散遍历SDE的Wasserstein-2距离的非渐近保证，并解决了实践中维度依赖性的问题。

    

    本文对San-Serna和Zygalakis的《对于数值逼近遍历SDE的分布的Wasserstein距离估计》中的非渐近保证数值离散分析方法进行了修正。他们分析了UBU积分器，该积分器是二阶强型的，并且每个步骤只需要一次梯度评估，从而得到了理想的非渐近保证，特别是在Wasserstein-2距离中到达离目标分布 $\epsilon > 0$ 的距离仅需 $\mathcal{O}(d^{1/4}\epsilon^{-1/2})$ 步。然而，Sanz-Serna和Zygalakis (2021)中的局部误差估计存在错误，在实践中需要更强的假设才能实现这些复杂度估计。本文解决了理论与实践中观察到的许多应用场景中的维度依赖性。

    arXiv:2402.08711v1 Announce Type: cross Abstract: A method for analyzing non-asymptotic guarantees of numerical discretizations of ergodic SDEs in Wasserstein-2 distance is presented by Sanz-Serna and Zygalakis in ``Wasserstein distance estimates for the distributions of numerical approximations to ergodic stochastic differential equations". They analyze the UBU integrator which is strong order two and only requires one gradient evaluation per step, resulting in desirable non-asymptotic guarantees, in particular $\mathcal{O}(d^{1/4}\epsilon^{-1/2})$ steps to reach a distance of $\epsilon > 0$ in Wasserstein-2 distance away from the target distribution. However, there is a mistake in the local error estimates in Sanz-Serna and Zygalakis (2021), in particular, a stronger assumption is needed to achieve these complexity estimates. This note reconciles the theory with the dimension dependence observed in practice in many applications of interest.
    
[^34]: 顺序流匹配用于生成建模

    Sequential Flow Matching for Generative Modeling

    [https://arxiv.org/abs/2402.06461](https://arxiv.org/abs/2402.06461)

    本文提出了一种称为SeqRF的新方法，用于通过直线化概率流来减小全局截断误差，并以此加速取样和提高综合质量。

    

    直接引导连续时间生成模型（例如扩散模型或基于流的模型）的概率流是通过数值解算器快速取样的关键。现有方法通过直接生成噪声和数据分布之间的联合分布的概率路径来学习线性路径。ODE模型的仿真速度慢的一个重要原因是ODE轨迹的高曲率导致的ODE求解器的全局截断误差，这会在低NFE范围内放大数值解算器的截断误差。为了解决这个挑战，我们提出了一种称为SeqRF的新方法，它是一种学习技术，用于直线化概率流以减小全局截断误差，从而加速取样并提高综合质量。通过理论和实证研究，我们首先观察到了SeqRF的直线化特性。

    Straightening the probability flow of the continuous-time generative models, such as diffusion models or flow-based models, is the key to fast sampling through the numerical solvers, existing methods learn a linear path by directly generating the probability path the joint distribution between the noise and data distribution. One key reason for the slow sampling speed of the ODE-based solvers that simulate these generative models is the global truncation error of the ODE solver, caused by the high curvature of the ODE trajectory, which explodes the truncation error of the numerical solvers in the low-NFE regime. To address this challenge, We propose a novel method called SeqRF, a learning technique that straightens the probability flow to reduce the global truncation error and hence enable acceleration of sampling and improve the synthesis quality. In both theoretical and empirical studies, we first observe the straightening property of our SeqRF. Through empirical evaluations via SeqR
    
[^35]: 关于现代Hopfield模型计算限制的一个细粒度复杂性分析

    On Computational Limits of Modern Hopfield Models: A Fine-Grained Complexity Analysis

    [https://arxiv.org/abs/2402.04520](https://arxiv.org/abs/2402.04520)

    通过细粒度复杂性分析，我们研究了现代Hopfield模型的记忆检索计算限制，发现了一种基于模式范数的相变行为，并且建立了有效变体的上界条件。使用低秩逼近的方法，我们提供了有效构造的示例，同时证明了计算时间下界、记忆检索误差界和指数记忆容量。

    

    我们从细粒度复杂性分析的角度研究了现代Hopfield模型的记忆检索动力学的计算限制。我们的主要贡献是基于模式的范数对所有可能的现代Hopfield模型的效率进行相变行为的刻画。具体来说，我们建立了对输入查询模式和记忆模式的范数的上界标准。仅在这个标准之下，假设满足Strong Exponential Time Hypothesis (SETH)，存在子二次（高效）变体的现代Hopfield模型。为了展示我们的理论，当有效标准成立时，我们提供了现代Hopfield模型使用低秩逼近的有效构造的正式示例。这包括一个计算时间的下界导出，与$\Max\{$存储的记忆模式数量，输入查询序列的长度$\}$线性缩放。此外，我们证明了记忆检索误差界和指数记忆容量。

    We investigate the computational limits of the memory retrieval dynamics of modern Hopfield models from the fine-grained complexity analysis. Our key contribution is the characterization of a phase transition behavior in the efficiency of all possible modern Hopfield models based on the norm of patterns. Specifically, we establish an upper bound criterion for the norm of input query patterns and memory patterns. Only below this criterion, sub-quadratic (efficient) variants of the modern Hopfield model exist, assuming the Strong Exponential Time Hypothesis (SETH). To showcase our theory, we provide a formal example of efficient constructions of modern Hopfield models using low-rank approximation when the efficient criterion holds. This includes a derivation of a lower bound on the computational time, scaling linearly with $\Max\{$# of stored memory patterns, length of input query sequence$\}$. In addition, we prove its memory retrieval error bound and exponential memory capacity.
    
[^36]: 交叉验证和突变验证在模型选择中的实证比较

    Empirical Comparison between Cross-Validation and Mutation-Validation in Model Selection

    [https://arxiv.org/abs/2311.14079](https://arxiv.org/abs/2311.14079)

    本研究通过对比基准和实际数据集，实证比较了突变验证（MV）和交叉验证（CV）在模型选择中的表现。结果发现，MV和CV在选择模型的泛化性能方面基本等效，但MV在选择简单模型和计算成本方面具有优势。

    

    突变验证（MV）是一种近期提出的模型选择方法，因其独特特性和潜在益处而受到广泛关注，与广泛使用的交叉验证（CV）方法相比。在本研究中，我们对MV和k折交叉验证（CV）进行了基准和实际数据集的实证比较。通过使用贝叶斯测试，我们比较了产生三个后验概率的泛化估计：实际等效性、CV优势和MV优势。我们还评估了所选模型的能力差异和计算效率。我们发现，在各种机器学习算法和大多数基准数据集中，MV和CV都选择具有实际等效泛化性能的模型。MV在选择较简单模型和较低计算成本方面具有优势。然而，在某些情况下，MV选择过于简单的模型导致欠拟合。

    arXiv:2311.14079v2 Announce Type: replace  Abstract: Mutation validation (MV) is a recently proposed approach for model selection, garnering significant interest due to its unique characteristics and potential benefits compared to the widely used cross-validation (CV) method. In this study, we empirically compared MV and $k$-fold CV using benchmark and real-world datasets. By employing Bayesian tests, we compared generalization estimates yielding three posterior probabilities: practical equivalence, CV superiority, and MV superiority. We also evaluated the differences in the capacity of the selected models and computational efficiency. We found that both MV and CV select models with practically equivalent generalization performance across various machine learning algorithms and the majority of benchmark datasets. MV exhibited advantages in terms of selecting simpler models and lower computational costs. However, in some cases MV selected overly simplistic models leading to underfitting
    
[^37]: 差分隐私图学习的敏感性有界个性化PageRank算法

    Differentially Private Graph Learning via Sensitivity-Bounded Personalized PageRank

    [https://arxiv.org/abs/2207.06944](https://arxiv.org/abs/2207.06944)

    本论文提出了一种敏感性有界的个性化PageRank算法，能够保护用户隐私。该算法在保持准确性的同时，实现了差分隐私图学习的几种工具。

    

    个性化PageRank(PPR)是一种基本工具，用于无监督学习图表示，如节点排序、标注和图嵌入。然而，随着数据隐私成为最近最重要的关注点之一，现有的PPR算法并未设计用于保护用户隐私。PPR对输入图的边非常敏感：仅差一个边的差异可能会导致PPR向量发生巨大改变，从而可能泄漏用户私密数据。在这篇论文中，我们提出了一种算法，该算法输出近似PPR，并对输入边具有可证明的敏感性边界。此外，我们证明了当输入图具有大度数时，我们的算法达到与非私密算法相似的准确性。我们敏感性有界PPR直接意味着图学习的几种私密算法，如差分隐私(DP)PPR排序、DP节点分类和DP节点嵌入。为了补充我们的理论分析，我们还通过实验证明了算法的实际性能。

    Personalized PageRank (PPR) is a fundamental tool in unsupervised learning of graph representations such as node ranking, labeling, and graph embedding. However, while data privacy is one of the most important recent concerns, existing PPR algorithms are not designed to protect user privacy. PPR is highly sensitive to the input graph edges: the difference of only one edge may cause a big change in the PPR vector, potentially leaking private user data.   In this work, we propose an algorithm which outputs an approximate PPR and has provably bounded sensitivity to input edges. In addition, we prove that our algorithm achieves similar accuracy to non-private algorithms when the input graph has large degrees. Our sensitivity-bounded PPR directly implies private algorithms for several tools of graph learning, such as, differentially private (DP) PPR ranking, DP node classification, and DP node embedding. To complement our theoretical analysis, we also empirically verify the practical perfor
    
[^38]: 基于排序的快速可解释聚类算法

    Fast and explainable clustering based on sorting

    [https://arxiv.org/abs/2202.01456](https://arxiv.org/abs/2202.01456)

    CLASSIX是一种快速可解释的聚类算法，它通过排序后的数据的贪婪聚合和群组合并来进行聚类。该算法具有与最先进的聚类算法相媲美的性能，并且具有线性空间复杂性和近线性时间复杂性。

    

    我们引入了一种快速可解释的聚类方法，称为CLASSIX。它由两个阶段组成，即将排序后的数据聚合成附近数据点组成的群组的贪婪聚合阶段，然后将群组合并成聚类。该算法由两个标量参数控制，一个是聚合的距离参数，另一个是控制最小聚类大小的参数。我们进行了广泛的实验，对合成和真实数据集的聚类性能进行了全面评估，涵盖了各种聚类形状和低到高的特征维度。实验结果表明，CLASSIX可以与最先进的聚类算法竞争。该算法具有线性空间复杂性，在广泛的问题范围内实现了接近线性的时间复杂性。其固有的简单性使得可以生成对计算的聚类的直观解释。

    arXiv:2202.01456v2 Announce Type: replace  Abstract: We introduce a fast and explainable clustering method called CLASSIX. It consists of two phases, namely a greedy aggregation phase of the sorted data into groups of nearby data points, followed by the merging of groups into clusters. The algorithm is controlled by two scalar parameters, namely a distance parameter for the aggregation and another parameter controlling the minimal cluster size. Extensive experiments are conducted to give a comprehensive evaluation of the clustering performance on synthetic and real-world datasets, with various cluster shapes and low to high feature dimensionality. Our experiments demonstrate that CLASSIX competes with state-of-the-art clustering algorithms. The algorithm has linear space complexity and achieves near linear time complexity on a wide range of problems. Its inherent simplicity allows for the generation of intuitive explanations of the computed clusters.
    
[^39]: 结构通过架构：无需正则化的结构化表示

    Structure by Architecture: Structured Representations without Regularization

    [https://arxiv.org/abs/2006.07796](https://arxiv.org/abs/2006.07796)

    我们提出了一种自我监督的结构化表示学习方法，使用无需正则化的自动编码器架构。通过依赖潜变量的独立性进行采样，我们避免了重构质量和生成性能之间的权衡。我们的模型能够学习出一种有序的结构化表示，改善了生成、解缠和外推等多个下游任务的性能。

    

    我们研究了自我监督结构化表示学习的问题，使用自动编码器进行下游任务，如生成模型。与大多数方法依赖于匹配任意的、相对非结构化的先验分布进行采样的情况不同，我们提出了一种仅仅依赖于潜变量的独立性的采样技术，从而避免了在VAE中通常观察到的重构质量和生成性能之间的权衡。我们设计了一种新颖的自动编码器架构，能够学习出一种无需过度正则化的结构化表示。我们的结构解码器学习了一个层次的潜变量，从而无需额外的正则化或监督来对信息进行排序。我们演示了这些模型如何学习出改善各种下游任务的表示，包括生成、解缠和外推，使用了几个具有挑战性的任务。

    arXiv:2006.07796v4 Announce Type: replace  Abstract: We study the problem of self-supervised structured representation learning using autoencoders for downstream tasks such as generative modeling. Unlike most methods which rely on matching an arbitrary, relatively unstructured, prior distribution for sampling, we propose a sampling technique that relies solely on the independence of latent variables, thereby avoiding the trade-off between reconstruction quality and generative performance typically observed in VAEs. We design a novel autoencoder architecture capable of learning a structured representation without the need for aggressive regularization. Our structural decoders learn a hierarchy of latent variables, thereby ordering the information without any additional regularization or supervision. We demonstrate how these models learn a representation that improves results in a variety of downstream tasks including generation, disentanglement, and extrapolation using several challengi
    
[^40]: 基于因果相似性的分层贝叶斯模型

    Causal Similarity-Based Hierarchical Bayesian Models. (arXiv:2310.12595v1 [cs.LG])

    [http://arxiv.org/abs/2310.12595](http://arxiv.org/abs/2310.12595)

    本文提出了一种基于因果相似性的分层贝叶斯模型，通过学习如何从具有相似因果机制的训练任务中汇集数据来提高机器学习算法对新任务的泛化能力。

    

    机器学习的关键挑战是对新数据的泛化能力。本研究探讨了对由相关任务组成的数据集进行泛化的问题，这些任务可能在因果机制上存在差异。例如，复杂疾病的观察性医学数据在不同患者间具有疾病因果机制的异质性，这给需要对训练数据集之外的新患者进行泛化的机器学习算法带来了挑战。常用的处理异质性数据集的方法包括为整个数据集学习一个全局模型，为每个任务的数据学习本地模型，或者利用分层、元学习和多任务学习方法从汇集的多个任务的数据中学习泛化。本文提出了基于因果相似性的分层贝叶斯模型，通过学习如何从具有相似因果机制的训练任务中汇集数据来提高对新任务的泛化能力。我们应用这种通用建模方法

    The key challenge underlying machine learning is generalisation to new data. This work studies generalisation for datasets consisting of related tasks that may differ in causal mechanisms. For example, observational medical data for complex diseases suffers from heterogeneity in causal mechanisms of disease across patients, creating challenges for machine learning algorithms that need to generalise to new patients outside of the training dataset. Common approaches for learning supervised models with heterogeneous datasets include learning a global model for the entire dataset, learning local models for each tasks' data, or utilising hierarchical, meta-learning and multi-task learning approaches to learn how to generalise from data pooled across multiple tasks. In this paper we propose causal similarity-based hierarchical Bayesian models to improve generalisation to new tasks by learning how to pool data from training tasks with similar causal mechanisms. We apply this general modelling
    
[^41]: 无标记的域外数据改善了泛化能力

    Unlabeled Out-Of-Domain Data Improves Generalization. (arXiv:2310.00027v1 [stat.ML])

    [http://arxiv.org/abs/2310.00027](http://arxiv.org/abs/2310.00027)

    这个论文提出了一种新的框架，可以将无标记的域外数据纳入半监督分类问题，从而改善泛化能力。该框架结合了分布鲁棒优化与自监督训练，并利用了高效的多项式时间算法。在理论上，该框架在高斯混合分类问题中得到了验证。

    

    我们提出了一种将无标记数据纳入半监督分类问题的新框架，其中考虑了最小化鲁棒性损失函数或非鲁棒性损失函数的情景。值得注意的是，我们允许无标记样本在总变差意义上略微偏离域内分布。我们的框架的核心思想是将分布鲁棒优化（DRO）与自监督训练相结合。因此，我们还利用了训练阶段的高效多项式时间算法。从理论上讲，我们将我们的框架应用于在$\mathbb{R}^d$中的两个高斯混合分类问题，除了来自真实分布的$m$个独立标记样本之外，还给出了一组$n$个（通常$n\gg m$）域外和无标记样本。已知仅使用标记数据，泛化误差可以通过$\propto\left(d/m\right)$进行界定。

    We propose a novel framework for incorporating unlabeled data into semi-supervised classification problems, where scenarios involving the minimization of either i) adversarially robust or ii) non-robust loss functions have been considered. Notably, we allow the unlabeled samples to deviate slightly (in total variation sense) from the in-domain distribution. The core idea behind our framework is to combine Distributionally Robust Optimization (DRO) with self-supervised training. As a result, we also leverage efficient polynomial-time algorithms for the training stage. From a theoretical standpoint, we apply our framework on the classification problem of a mixture of two Gaussians in $\mathbb{R}^d$, where in addition to the $m$ independent and labeled samples from the true distribution, a set of $n$ (usually with $n\gg m$) out of domain and unlabeled samples are gievn as well. Using only the labeled data, it is known that the generalization error can be bounded by $\propto\left(d/m\right
    
[^42]: 交互式和集中式差分隐私在Bandit问题中的应用

    Interactive and Concentrated Differential Privacy for Bandits. (arXiv:2309.00557v1 [stat.ML])

    [http://arxiv.org/abs/2309.00557](http://arxiv.org/abs/2309.00557)

    本文研究了在交互学习和推荐系统中隐私保护的Bandit问题，并引入了集中差分隐私的概念。通过提供关于有限臂和线性Bandit问题遗憾的下界，我们揭示了不同隐私预算下的难度区域，并发现集中差分隐私可以比全局差分隐私更有效地保护隐私，我们提出了两种相应的算法。

    

    Bandit问题在交互式学习方案和现代推荐系统中起着至关重要的作用。然而，这些系统通常依赖于敏感的用户数据，因此隐私是一个重要问题。本文通过交互式差分隐私的视角研究了基于可信集中式决策者的Bandit问题的隐私性。虽然已经对纯ε-全局差分隐私的Bandit问题进行了广泛研究，但我们在理解零集中差分隐私(zCDP)的Bandit问题方面做出了贡献。针对有限臂和线性Bandit问题，我们提供了关于遗憾的最小最大和问题相关下界，从而量化了这些情况下ρ-全局zCDP的代价。这些下界揭示了基于隐私预算ρ的两个困难区域，并表明ρ-全局zCDP比纯ε-全局差分隐私产生的遗憾更小。我们提出了两种有限臂和线性Bandit问题的ρ-全局zCDP算法，即AdaC-UCB和AdaC-GOPE。这两个算法都使用了高斯机制的共同策略。

    Bandits play a crucial role in interactive learning schemes and modern recommender systems. However, these systems often rely on sensitive user data, making privacy a critical concern. This paper investigates privacy in bandits with a trusted centralized decision-maker through the lens of interactive Differential Privacy (DP). While bandits under pure $\epsilon$-global DP have been well-studied, we contribute to the understanding of bandits under zero Concentrated DP (zCDP). We provide minimax and problem-dependent lower bounds on regret for finite-armed and linear bandits, which quantify the cost of $\rho$-global zCDP in these settings. These lower bounds reveal two hardness regimes based on the privacy budget $\rho$ and suggest that $\rho$-global zCDP incurs less regret than pure $\epsilon$-global DP. We propose two $\rho$-global zCDP bandit algorithms, AdaC-UCB and AdaC-GOPE, for finite-armed and linear bandits respectively. Both algorithms use a common recipe of Gaussian mechanism 
    
[^43]: 规范化就是你所需要的：理解极端标签偏移下的层归一化联邦学习

    Normalization Is All You Need: Understanding Layer-Normalized Federated Learning under Extreme Label Shift. (arXiv:2308.09565v1 [cs.LG])

    [http://arxiv.org/abs/2308.09565](http://arxiv.org/abs/2308.09565)

    本论文揭示了层归一化和联邦学习中的标签偏移问题之间的深刻联系，通过在联邦学习中应用特征归一化，使得对严重倾斜的数据集进行加速全局训练，从而在极端标签偏移下获得显著改进。

    

    层归一化（LN）是一个广泛采用的深度学习技术，特别在基础模型的时代。最近，已经证明LN在非独立同分布数据上的联邦学习（FL）中非常有效。然而，它为什么以及如何起作用仍然是个谜。在这项工作中，我们揭示了层归一化和联邦学习中的标签偏移问题之间的深刻联系。为了更好地理解FL中的层归一化，我们确定了规范化方法在FL中的关键贡献机制，称之为特征归一化（FN），它在分类器头之前将归一化应用于潜在特征表示。虽然LN和FN不会提高表达能力，但它们控制特征崩溃和局部过拟合，使得对严重倾斜的数据集进行加速全局训练。经验证明，规范化在极端标签偏移下可以引起标准基准的显著改进。此外，我们还进行了大量的割除研究。

    Layer normalization (LN) is a widely adopted deep learning technique especially in the era of foundation models. Recently, LN has been shown to be surprisingly effective in federated learning (FL) with non-i.i.d. data. However, exactly why and how it works remains mysterious. In this work, we reveal the profound connection between layer normalization and the label shift problem in federated learning. To understand layer normalization better in FL, we identify the key contributing mechanism of normalization methods in FL, called feature normalization (FN), which applies normalization to the latent feature representation before the classifier head. Although LN and FN do not improve expressive power, they control feature collapse and local overfitting to heavily skewed datasets, and thus accelerates global training. Empirically, we show that normalization leads to drastic improvements on standard benchmarks under extreme label shift. Moreover, we conduct extensive ablation studies to unde
    
[^44]: A/B测试和具有非稳态鲁棒性的线性赌博机最佳臂识别问题

    A/B Testing and Best-arm Identification for Linear Bandits with Robustness to Non-stationarity. (arXiv:2307.15154v1 [cs.LG])

    [http://arxiv.org/abs/2307.15154](http://arxiv.org/abs/2307.15154)

    本文研究了在非稳态环境中的线性赌博机的最佳臂识别问题，提出了一种具有鲁棒性的算法来解决。该算法通过在每个时间步从一个G-最优设计中随机选择臂来实现最佳臂的鲁棒识别。

    

    本文研究了在可能存在非稳态环境下的线性赌博机中的固定预算最佳臂识别问题。给定有限臂集合X，固定预算T以及不可预测的参数序列θ，算法的目标是以尽可能高的概率正确识别最佳臂x*。之前的工作已经在稳态设置下进行了研究，并且证明了错误概率随着预算的增加而指数下降。但在许多现实世界的A/B/n多变量测试场景中，环境是非稳态的，而一个期望稳态的算法很容易失败。为了具有鲁棒的识别能力，众所周知，如果在每个时间步从X的一个G-最优设计中以随机和非自适应的方式选择臂，那么可以实现最佳臂的鲁棒识别。

    We investigate the fixed-budget best-arm identification (BAI) problem for linear bandits in a potentially non-stationary environment. Given a finite arm set $\mathcal{X}\subset\mathbb{R}^d$, a fixed budget $T$, and an unpredictable sequence of parameters $\left\lbrace\theta_t\right\rbrace_{t=1}^{T}$, an algorithm will aim to correctly identify the best arm $x^* := \arg\max_{x\in\mathcal{X}}x^\top\sum_{t=1}^{T}\theta_t$ with probability as high as possible. Prior work has addressed the stationary setting where $\theta_t = \theta_1$ for all $t$ and demonstrated that the error probability decreases as $\exp(-T /\rho^*)$ for a problem-dependent constant $\rho^*$. But in many real-world $A/B/n$ multivariate testing scenarios that motivate our work, the environment is non-stationary and an algorithm expecting a stationary setting can easily fail. For robust identification, it is well-known that if arms are chosen randomly and non-adaptively from a G-optimal design over $\mathcal{X}$ at each 
    
[^45]: 隐私放大通过重要性采样

    Privacy Amplification via Importance Sampling. (arXiv:2307.10187v1 [cs.CR])

    [http://arxiv.org/abs/2307.10187](http://arxiv.org/abs/2307.10187)

    通过重要性采样进行隐私放大，可以同时增强隐私保护和提高效用。我们提供了一个一般的结果来量化选择概率权重对隐私放大的影响，并展示了异质采样概率可以在保持子采样大小不变的情况下获得更好的隐私和效用。

    

    我们研究了通过重要性采样对数据集进行子采样作为差分隐私机制的预处理步骤来增强隐私保护的性质。这扩展了已有的通过子采样进行隐私放大的结果到重要性采样，其中每个数据点的权重为其被选择概率的倒数。每个点的选择概率的权重对隐私的影响并不明显。一方面，较低的选择概率会导致更强的隐私放大。另一方面，权重越高，在点被选择时，点对机制输出的影响就越强。我们提供了一个一般的结果来量化这两个影响之间的权衡。我们展示了异质采样概率可以同时比均匀子采样具有更强的隐私和更好的效用，并保持子采样大小不变。特别地，我们制定和解决了隐私优化采样的问题，即寻找...

    We examine the privacy-enhancing properties of subsampling a data set via importance sampling as a pre-processing step for differentially private mechanisms. This extends the established privacy amplification by subsampling result to importance sampling where each data point is weighted by the reciprocal of its selection probability. The implications for privacy of weighting each point are not obvious. On the one hand, a lower selection probability leads to a stronger privacy amplification. On the other hand, the higher the weight, the stronger the influence of the point on the output of the mechanism in the event that the point does get selected. We provide a general result that quantifies the trade-off between these two effects. We show that heterogeneous sampling probabilities can lead to both stronger privacy and better utility than uniform subsampling while retaining the subsample size. In particular, we formulate and solve the problem of privacy-optimal sampling, that is, finding
    
[^46]: 学习受限动力学的稳定神经微分方程

    Stabilized Neural Differential Equations for Learning Constrained Dynamics. (arXiv:2306.09739v1 [cs.LG])

    [http://arxiv.org/abs/2306.09739](http://arxiv.org/abs/2306.09739)

    本文提出了一种稳定神经微分方程（SNDEs）的方法，可以强制使用任意流形约束。该方法通过添加稳定项使约束流形成为渐进稳定的，并且在实验中表现优于现有方法。

    

    最近出现了许多成功的从数据学习动态系统的方法。然而，确保推断出的动态系统保留已知约束条件（例如守恒定律或对允许的系统状态的限制）仍然具有挑战性。我们提出了稳定神经微分方程（SNDEs）的方法，这是一种用于神经微分方程强制使用任意流形约束的方法。我们的方法基于一个稳定项，当添加到原始动态系统中时，可以将约束流形成为渐进稳定的。由于其简单性，我们的方法与所有常见的神经常微分方程（NODE）模型兼容并广泛适用。在广泛的经验评估中，我们证明SNDE在扩展可纳入NODE训练的约束类型方面胜过现有方法。

    Many successful methods to learn dynamical systems from data have recently been introduced. However, assuring that the inferred dynamics preserve known constraints, such as conservation laws or restrictions on the allowed system states, remains challenging. We propose stabilized neural differential equations (SNDEs), a method to enforce arbitrary manifold constraints for neural differential equations. Our approach is based on a stabilization term that, when added to the original dynamics, renders the constraint manifold provably asymptotically stable. Due to its simplicity, our method is compatible with all common neural ordinary differential equation (NODE) models and broadly applicable. In extensive empirical evaluations, we demonstrate that SNDEs outperform existing methods while extending the scope of which types of constraints can be incorporated into NODE training.
    
[^47]: 分布式SGD算法的稳定性与泛化分析改进

    Improved Stability and Generalization Analysis of the Decentralized SGD Algorithm. (arXiv:2306.02939v1 [cs.LG])

    [http://arxiv.org/abs/2306.02939](http://arxiv.org/abs/2306.02939)

    本文提出了新的算法稳定性理论来改进分布式SGD算法的泛化性能分析，推翻了现有技术对通信图负面影响的观点，并展示了D-SGD在凸设置中与经典SGD算法泛化界相同。

    

    本文基于算法稳定性，提出了分布式随机梯度下降(D-SGD)算法的新的泛化误差分析方法。得到的结果大大改进了现有技术，并推翻了它们关于通信图对泛化的负面影响的观点。例如，在凸设置中，无论图的选择如何，D-SGD具有与经典SGD算法相同的泛化界。我们发现这种反直觉的结果来自于考虑本地参数的平均值，这会隐藏一个与分布式场景不兼容的最终全局平均化步骤。考虑到这一观察结果，我们倡导分析本地参数的上确界，并展示了在这种情况下，图确实对泛化产生影响。与之前的结果不同，我们的分析即使对于非连接图也能产生非平凡边界。

    This paper presents a new generalization error analysis for the Decentralized Stochastic Gradient Descent (D-SGD) algorithm based on algorithmic stability. The obtained results largely improve upon state-of-the-art results, and even invalidate their claims that the communication graph has a detrimental effect on generalization. For instance, we show that in convex settings, D-SGD has the same generalization bounds as the classical SGD algorithm, no matter the choice of graph. We exhibit that this counter-intuitive result comes from considering the average of local parameters, which hides a final global averaging step incompatible with the decentralized scenario. In light of this observation, we advocate to analyze the supremum over local parameters and show that in this case, the graph does have an impact on the generalization. Unlike prior results, our analysis yields non-vacuous bounds even for non-connected graphs.
    
[^48]: 通过贝叶斯主动学习实现自校正贝叶斯优化

    Self-Correcting Bayesian Optimization through Bayesian Active Learning. (arXiv:2304.11005v1 [cs.LG])

    [http://arxiv.org/abs/2304.11005](http://arxiv.org/abs/2304.11005)

    该论文提出了SAL和SCoreBO两种方法，用于提高高斯过程模型的超参数选择和贝叶斯优化的表现。

    

    高斯过程已成为贝叶斯优化和主动学习中的首选模型。然而，高斯过程的完全发挥需要巧妙选择超参数，而在文献中很少有关于找到正确超参数的努力。我们演示了选择好的超参数对于高斯过程的影响，并提出了两个明确优先考虑此目标的收购函数。统计距离主动学习（SAL）考虑后验样本的平均不一致性，由统计距离测量。结果显示，在许多测试函数上，它胜过了贝叶斯主动学习的最新结果。然后，我们引入了自校正贝叶斯优化（SCoreBO），它将SAL扩展到同时执行贝叶斯优化和主动超参数学习。相比传统BO，SCoreBO以改进的速度学习模型超参数，同时在最新的贝叶斯优化搜索中取得更好的表现。

    Gaussian processes are cemented as the model of choice in Bayesian optimization and active learning. Yet, they are severely dependent on cleverly chosen hyperparameters to reach their full potential, and little effort is devoted to finding the right hyperparameters in the literature. We demonstrate the impact of selecting good hyperparameters for GPs and present two acquisition functions that explicitly prioritize this goal. Statistical distance-based Active Learning (SAL) considers the average disagreement among samples from the posterior, as measured by a statistical distance. It is shown to outperform the state-of-the-art in Bayesian active learning on a number of test functions. We then introduce Self-Correcting Bayesian Optimization (SCoreBO), which extends SAL to perform Bayesian optimization and active hyperparameter learning simultaneously. SCoreBO learns the model hyperparameters at improved rates compared to vanilla BO, while outperforming the latest Bayesian optimization met
    
[^49]: 基于GLASS模型的脑机接口贝叶斯推断

    Bayesian inference on Brain-Computer Interface using the GLASS Model. (arXiv:2304.07401v1 [stat.AP])

    [http://arxiv.org/abs/2304.07401](http://arxiv.org/abs/2304.07401)

    本文针对P300 BCI问题，开发了一种基于GLASS模型的贝叶斯推断方法，直接解决了脑机接口应用中数据集不平衡问题，具有良好的分类性能和易于解释性。

    

    脑机接口（BCI）使重度残疾人士与世界进行交流。BCI将实时的脑活动转化为计算机指令，通常被认为是一个分类问题，计算神经科学提供了机遇和挑战。本文集中在使用事件相关电位（ERP）BCI设计的P300 BCI上。我们开发了一种新颖的具有稀疏时变效应的高斯潜在组模型（GLASS），用于在P300 BCI上进行贝叶斯推断。GLASS采用多项式回归框架，直接解决了BCI应用中的数据集不平衡问题。先验规范促进了i）使用软阈值进行特征选择和噪声降低，ii）使用全局收缩对时变效应进行平滑处理，iii）对潜在组进行聚类，以减轻EEG数据的高空间相关性。我们开发了一种有效的图模型算法，用于后验计算和模型选择。所提出的GLASS模型在基准数据集上实现了竞争性的分类性能，并提供了所推断的条件相关时空模式的易于解释性。

    The brain-computer interface (BCI) enables individuals with severe physical impairments to communicate with the world. BCIs offer computational neuroscience opportunities and challenges in converting real-time brain activities to computer commands and are typically framed as a classification problem. This article focuses on the P300 BCI that uses the event-related potential (ERP) BCI design, where the primary challenge is classifying target/non-target stimuli. We develop a novel Gaussian latent group model with sparse time-varying effects (GLASS) for making Bayesian inferences on the P300 BCI. GLASS adopts a multinomial regression framework that directly addresses the dataset imbalance in BCI applications. The prior specifications facilitate i) feature selection and noise reduction using soft-thresholding, ii) smoothing of the time-varying effects using global shrinkage, and iii) clustering of latent groups to alleviate high spatial correlations of EEG data. We develop an efficient gra
    
[^50]: 逆可解性和安全性及其在联邦学习中的应用

    Inverse Solvability and Security with Applications to Federated Learning. (arXiv:2211.14115v3 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2211.14115](http://arxiv.org/abs/2211.14115)

    介绍了逆可解性和安全性的概念，以及其在联邦学习中的应用。论文提供了模型示例，展示了如何通过增加用户数量来增加可解性和安全性。

    

    我们介绍了逆可解性和安全性的概念，适用于一般线性前向模型，并展示了如何将其应用于联邦学习中使用的模型。我们提供了这样的模型的示例，其逆可解性和安全性在本文中得到定义。我们还展示了如何利用参与给定迭代的大量用户来增加可解性和安全性。最后，我们讨论了所提出概念的可能扩展，包括非线性情况。

    We introduce the concepts of inverse solvability and security for a generic linear forward model and demonstrate how they can be applied to models used in federated learning. We provide examples of such models which differ in the resulting inverse solvability and security as defined in this paper. We also show how the large number of users participating in a given iteration of federated learning can be leveraged to increase both solvability and security. Finally, we discuss possible extensions of the presented concepts including the nonlinear case.
    

