# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Benchmarking Uncertainty Disentanglement: Specialized Uncertainties for Specialized Tasks](https://arxiv.org/abs/2402.19460) | 该论文评估了在ImageNet上的多种不确定性估计器，发现虽然有理论努力，但实践中尚未实现不确定性的解开，同时揭示了哪些估计器在特定任务上表现出色，为从业者提供见解并指导未来研究。 |
| [^2] | [Listening to the Noise: Blind Denoising with Gibbs Diffusion](https://arxiv.org/abs/2402.19455) | 引入了Gibbs扩散（GDiff）方法，通过交替采样信号先验和噪声分布族，以及蒙特卡洛采样来推断噪声参数，解决了盲去噪中需要知道噪声水平和协方差的问题。 |
| [^3] | [Heavy-Tailed Class Imbalance and Why Adam Outperforms Gradient Descent on Language Models](https://arxiv.org/abs/2402.19449) | 研究发现语言模型中的重尾类别不平衡问题导致了优化动态上的困难，Adam和基于符号的方法在这种情况下优于梯度下降。 |
| [^4] | [Training Dynamics of Multi-Head Softmax Attention for In-Context Learning: Emergence, Convergence, and Optimality](https://arxiv.org/abs/2402.19442) | 研究了多头softmax注意力模型在上下文学习中的训练动态，证明了全局收敛性，并发现了“任务分配”现象，梯度流动分为热身、涌现和收敛三个阶段，最终证明了梯度流的最优性。 |
| [^5] | [Negative-Binomial Randomized Gamma Markov Processes for Heterogeneous Overdispersed Count Time Series](https://arxiv.org/abs/2402.18995) | 提出了一种负二项随机Gamma马尔可夫过程，用于改进异质过度离散计数时间序列的预测性能，并加快推断算法的收敛速度。 |
| [^6] | [Semi-Supervised U-statistics](https://arxiv.org/abs/2402.18921) | 介绍了一种半监督 U-统计量，利用大量未标记数据，获得了渐近正态分布的性质，并通过有效整合各种强大预测工具实现了明显的效率提升。 |
| [^7] | [Prognostic Covariate Adjustment for Logistic Regression in Randomized Controlled Trials](https://arxiv.org/abs/2402.18900) | 使用预后评分调整可以提高逻辑回归中条件比值的Wald检验的能力 |
| [^8] | [Supervised Contrastive Representation Learning: Landscape Analysis with Unconstrained Features](https://arxiv.org/abs/2402.18884) | 通过监督对比表示学习，在超参数化的深度神经网络中研究解决方案，揭示了最小化SC损失的全局最小值和唯一最小化器。 |
| [^9] | [Applications of 0-1 Neural Networks in Prescription and Prediction](https://arxiv.org/abs/2402.18851) | 引入了处方网络（PNNs）这种新型神经网络，通过混合整数规划训练，结合反事实估计，在医疗决策中展现出优于现有方法的表现，可优化治疗策略，并具有更大的可解释性和更复杂的策略编码能力。 |
| [^10] | [VEC-SBM: Optimal Community Detection with Vectorial Edges Covariates](https://arxiv.org/abs/2402.18805) | 该论文介绍了一种名为VEC-SBM的算法，可以在社交网络中使用向量边缘协变量来最优地检测社区，证明了在社区检测过程中利用边缘信息的附加价值。 |
| [^11] | [BlockEcho: Retaining Long-Range Dependencies for Imputing Block-Wise Missing Data](https://arxiv.org/abs/2402.18800) | 该论文提出了一种名为BlockEcho的新矩阵填充方法，通过将矩阵分解和生成对抗网络相结合，创造性地保留了原始矩阵中的长距离依赖关系，以解决块状缺失数据对数据插值和预测能力的挑战。 |
| [^12] | [Learning Associative Memories with Gradient Descent](https://arxiv.org/abs/2402.18724) | 该论文研究了一个关联记忆模块的训练动态，通过理论和实验揭示了在过参数化和欠参数化情况下的学习动态和误差特性。 |
| [^13] | [Inferring Dynamic Networks from Marginals with Iterative Proportional Fitting](https://arxiv.org/abs/2402.18697) | 通过识别一个生成网络模型，我们建立了一个设置，IPF可以恢复最大似然估计，揭示了关于在这种设置中使用IPF的隐含假设，并可以为IPF的参数估计提供结构相关的误差界。 |
| [^14] | [Arithmetic Control of LLMs for Diverse User Preferences: Directional Preference Alignment with Multi-Objective Rewards](https://arxiv.org/abs/2402.18571) | 提出了方向偏好对齐（DPA）框架，通过多目标奖励模拟不同偏好配置，以实现用户相关的偏好控制。 |
| [^15] | [RNNs are not Transformers (Yet): The Key Bottleneck on In-context Retrieval](https://arxiv.org/abs/2402.18510) | 本文研究了RNNs和Transformer在处理算法问题时的表现能力差距，发现RNNs存在关键瓶颈，即无法完美地从上下文中检索信息，导致无法像Transformer那样轻松解决需要这种能力的任务。 |
| [^16] | [Zeroth-Order Sampling Methods for Non-Log-Concave Distributions: Alleviating Metastability by Denoising Diffusion](https://arxiv.org/abs/2402.17886) | 本文提出了一种基于去噪扩散过程的零阶扩散蒙特卡洛算法，克服了非对数凹分布采样中的亚稳定性问题，并证明其采样精度具有倒多项式依赖。 |
| [^17] | [A Provably Accurate Randomized Sampling Algorithm for Logistic Regression](https://arxiv.org/abs/2402.16326) | 提出了一种逻辑回归问题的简单随机抽样算法，通过随机矩阵乘法实现高质量逼近估计概率和模型整体差异性。 |
| [^18] | [Learning Granger Causality from Instance-wise Self-attentive Hawkes Processes](https://arxiv.org/abs/2402.03726) | 本论文提出了一种名为ISAHP的深度学习框架，可以从异步、相互依赖的多类型事件序列中无监督地学习实例级的格兰杰因果关系。它是第一个满足格兰杰因果关系要求的神经点过程模型，并利用变压器的自我注意机制来实现格兰杰因果关系的推断。 |
| [^19] | [Detecting algorithmic bias in medical AI-models](https://arxiv.org/abs/2312.02959) | 本文提出了一种创新的框架，用于检测医疗AI决策支持系统中的算法偏倚，通过采用CART算法有效地识别医疗AI模型中的潜在偏倚，并在合成数据实验和真实临床环境中验证了其有效性。 |
| [^20] | [Time-Uniform Confidence Spheres for Means of Random Vectors](https://arxiv.org/abs/2311.08168) | 该研究提出了时间均匀置信球序列，可以同时高概率地包含各种样本量下随机向量的均值，并针对不同分布假设进行了扩展和统一分析。 |
| [^21] | [Inference via robust optimal transportation: theory and methods](https://arxiv.org/abs/2301.06297) | 通过鲁棒Wasserstein距离处理输运问题，探讨了与$W_1$的关联，推导了集中不等式，并提出最小距离估计器以及其统计保证。 |
| [^22] | [Neural Galerkin Schemes with Active Learning for High-Dimensional Evolution Equations](https://arxiv.org/abs/2203.01360) | 通过神经Galerkin方案结合深度学习和主动学习，能够自主生成训练数据用于高维偏微分方程的数值求解 |
| [^23] | [Offline detection of change-points in the mean for stationary graph signals](https://arxiv.org/abs/2006.10628) | 提出了一种离线方法用于检测图信号中均值变化点，通过在频谱域解决问题，充分利用了稀疏性，采用模型选择方法自动确定变点的数量，并给出了非渐近oracle不等式的证明。 |
| [^24] | [The committee machine: Computational to statistical gaps in learning a two-layers neural network](https://arxiv.org/abs/1806.05451) | 介绍了对于两层神经网络模型委员会机器的严格理论基础和近似消息传递算法，揭示了计算到统计学差距。 |
| [^25] | [Looping in the Human: Collaborative and Explainable Bayesian Optimization.](http://arxiv.org/abs/2310.17273) | 协作和可解释的贝叶斯优化框架(CoExBO)在贝叶斯优化中引入了循环，平衡了人工智能和人类的合作关系。它利用偏好学习将用户见解融合到优化中，解释每次迭代的候选选择，从而增强用户对优化过程的信任，并提供无害保证。 |
| [^26] | [Extended Deep Adaptive Input Normalization for Preprocessing Time Series Data for Neural Networks.](http://arxiv.org/abs/2310.14720) | 本研究提出了一种名为EDAIn的扩展深度自适应输入规范化层，通过以端到端的方式学习如何适当地规范化时序数据，而不是使用固定的规范化方案，来提高深度神经网络在时序预测和分类任务中的性能。实验证明该方法在不同数据集上都取得了良好效果。 |
| [^27] | [A new high-resolution indoor radon map for Germany using a machine learning based probabilistic exposure model.](http://arxiv.org/abs/2310.11143) | 本研究提出了一种基于机器学习的概率暴露模型，可以更准确地估计德国室内氡气分布，并具有更高的空间分辨率。 |
| [^28] | [Mirror Diffusion Models for Constrained and Watermarked Generation.](http://arxiv.org/abs/2310.01236) | 提出了一种新的镜像扩散模型（MDM），可以在受限制集合上生成数据而不丧失可追溯性。这通过在一个标准的欧几里得空间中学习扩散过程，并利用镜像映射来实现。 |
| [^29] | [Cross-Prediction-Powered Inference.](http://arxiv.org/abs/2309.16598) | 本文介绍了一种基于机器学习的交叉预测方法，可以有效地进行推理。该方法通过使用一个小型标记数据集和一个大型未标记数据集，通过机器学习填补缺失的标签，并采用去偏差方法纠正预测的不准确性。 |
| [^30] | [Statistical Component Separation for Targeted Signal Recovery in Noisy Mixtures.](http://arxiv.org/abs/2306.15012) | 本论文提出了一种用于从噪声混合物中恢复目标信号的统计分量分离方法，并且在图像降噪任务中展示了其优于标准降噪方法的表现。 |
| [^31] | [Neural Likelihood Surfaces for Spatial Processes with Computationally Intensive or Intractable Likelihoods.](http://arxiv.org/abs/2305.04634) | 研究提出了使用CNN学习空间过程的似然函数。即使在没有确切似然函数的情况下，通过分类任务进行的神经网络的训练，可以隐式地学习似然函数。使用Platt缩放可以提高神经似然面的准确性。 |
| [^32] | [Non-asymptotic analysis of Langevin-type Monte Carlo algorithms.](http://arxiv.org/abs/2303.12407) | 本文提出了一种新的Langevin型算法并应用于吉布斯分布。通过提出的2-Wasserstein距离上限，我们发现势函数的耗散性以及梯度 $\alpha>1/3$ 下的 $\alpha$-H\"{o}lder连续性可以保证算法具有接近零的误差。新的Langevin型算法还可以应用于无凸性或连续可微性的势函数。 |
| [^33] | [Imputation of missing values in multi-view data.](http://arxiv.org/abs/2210.14484) | 本文提出了一种基于StaPLR算法的新的多视角数据插补算法，通过在降维空间中执行插补以解决计算挑战，并在模拟数据集中得到了竞争性结果。 |

# 详细

[^1]: 为标准化的任务专门指定不确定性量化：专门的不确定性用于专门的任务

    Benchmarking Uncertainty Disentanglement: Specialized Uncertainties for Specialized Tasks

    [https://arxiv.org/abs/2402.19460](https://arxiv.org/abs/2402.19460)

    该论文评估了在ImageNet上的多种不确定性估计器，发现虽然有理论努力，但实践中尚未实现不确定性的解开，同时揭示了哪些估计器在特定任务上表现出色，为从业者提供见解并指导未来研究。

    

    不确定性量化，曾经是一个独立的任务，已经发展成为一个包含预测抑制、越界检测以及随机不确定性量化在内的任务谱系。最新的目标是解开不确定性：构建多个估计器，每个都专门定制于一个特定任务。因此，最近有大量不同意图的最新进展——这些往往完全偏离实际行为。本文在ImageNet上对多种不确定性估计器进行全面评估。我们发现，尽管有着颇有希望的理论努力，实践中仍未实现解开。此外，我们揭示了哪些不确定性估计器在哪些特定任务上表现出色，为从业者提供见解并引导未来研究朝着基于任务和解开的不确定性估计方法发展。我们的代码可在 https://github.com/bmucsanyi/bud 找到。

    arXiv:2402.19460v1 Announce Type: new  Abstract: Uncertainty quantification, once a singular task, has evolved into a spectrum of tasks, including abstained prediction, out-of-distribution detection, and aleatoric uncertainty quantification. The latest goal is disentanglement: the construction of multiple estimators that are each tailored to one and only one task. Hence, there is a plethora of recent advances with different intentions - that often entirely deviate from practical behavior. This paper conducts a comprehensive evaluation of numerous uncertainty estimators across diverse tasks on ImageNet. We find that, despite promising theoretical endeavors, disentanglement is not yet achieved in practice. Additionally, we reveal which uncertainty estimators excel at which specific tasks, providing insights for practitioners and guiding future research toward task-centric and disentangled uncertainty estimation methods. Our code is available at https://github.com/bmucsanyi/bud.
    
[^2]: 听噪声：使用Gibbs扩散进行盲去噪

    Listening to the Noise: Blind Denoising with Gibbs Diffusion

    [https://arxiv.org/abs/2402.19455](https://arxiv.org/abs/2402.19455)

    引入了Gibbs扩散（GDiff）方法，通过交替采样信号先验和噪声分布族，以及蒙特卡洛采样来推断噪声参数，解决了盲去噪中需要知道噪声水平和协方差的问题。

    

    近年来，去噪问题与深度生成模型的发展密不可分。特别是，扩散模型被训练成去噪器，它们所建模的分布与贝叶斯图像中的去噪先验相符。然而，通过基于扩散的后验采样进行去噪需要知道噪声水平和协方差，这阻碍了盲去噪。我们通过引入 Gibbs扩散（GDiff）克服了这一限制，这是一种通用方法论，可以处理信号和噪声参数的后验采样。假设任意参数化的高斯噪声，我们开发了一种Gibbs算法，交替地从条件扩散模型中进行采样，该模型经过训练将信号先验映射到噪声分布族，以及一个蒙特卡洛采样器来推断噪声参数。我们的理论分析突出了潜在的缺陷，指导了诊断的使用，并量化了Gibbs s中的错误。

    arXiv:2402.19455v1 Announce Type: cross  Abstract: In recent years, denoising problems have become intertwined with the development of deep generative models. In particular, diffusion models are trained like denoisers, and the distribution they model coincide with denoising priors in the Bayesian picture. However, denoising through diffusion-based posterior sampling requires the noise level and covariance to be known, preventing blind denoising. We overcome this limitation by introducing Gibbs Diffusion (GDiff), a general methodology addressing posterior sampling of both the signal and the noise parameters. Assuming arbitrary parametric Gaussian noise, we develop a Gibbs algorithm that alternates sampling steps from a conditional diffusion model trained to map the signal prior to the family of noise distributions, and a Monte Carlo sampler to infer the noise parameters. Our theoretical analysis highlights potential pitfalls, guides diagnostic usage, and quantifies errors in the Gibbs s
    
[^3]: Heavy-Tailed Class Imbalance and Why Adam Outperforms Gradient Descent on Language Models

    Heavy-Tailed Class Imbalance and Why Adam Outperforms Gradient Descent on Language Models

    [https://arxiv.org/abs/2402.19449](https://arxiv.org/abs/2402.19449)

    研究发现语言模型中的重尾类别不平衡问题导致了优化动态上的困难，Adam和基于符号的方法在这种情况下优于梯度下降。

    

    本文研究了在语言建模任务中存在的重尾类别不平衡问题，以及为什么Adam在优化大型语言模型时的表现优于梯度下降方法。我们发现，由于语言建模任务中存在的重尾类别不平衡，使用梯度下降时，与不常见单词相关的损失下降速度比与常见单词相关的损失下降速度慢。由于大多数样本来自相对不常见的单词，平均损失值在梯度下降时下降速度较慢。相比之下，Adam和基于符号的方法却不受此问题影响，并改善了所有类别的预测性能。我们在不同架构和数据类型上进行了实证研究，证明了这种行为确实是由类别不平衡引起的。

    arXiv:2402.19449v1 Announce Type: cross  Abstract: Adam has been shown to outperform gradient descent in optimizing large language transformers empirically, and by a larger margin than on other tasks, but it is unclear why this happens. We show that the heavy-tailed class imbalance found in language modeling tasks leads to difficulties in the optimization dynamics. When training with gradient descent, the loss associated with infrequent words decreases slower than the loss associated with frequent ones. As most samples come from relatively infrequent words, the average loss decreases slowly with gradient descent. On the other hand, Adam and sign-based methods do not suffer from this problem and improve predictions on all classes. To establish that this behavior is indeed caused by class imbalance, we show empirically that it persist through different architectures and data types, on language transformers, vision CNNs, and linear models. We further study this phenomenon on a linear clas
    
[^4]: 多头softmax注意力机制在上下文学习中的训练动态：涌现、收敛和最优性

    Training Dynamics of Multi-Head Softmax Attention for In-Context Learning: Emergence, Convergence, and Optimality

    [https://arxiv.org/abs/2402.19442](https://arxiv.org/abs/2402.19442)

    研究了多头softmax注意力模型在上下文学习中的训练动态，证明了全局收敛性，并发现了“任务分配”现象，梯度流动分为热身、涌现和收敛三个阶段，最终证明了梯度流的最优性。

    

    我们研究了用于上下文学习的多任务线性回归的多头softmax注意力模型的梯度流动力学。我们证明了在适当的初始化选择下，梯度流动的全局收敛性。此外，我们证明了在梯度流动动力学中出现了有趣的“任务分配”现象，每个注意力头都专注于解决多任务模型中的单个任务。具体而言，我们证明了梯度流动动力学可以分为三个阶段——热身阶段，在这个阶段损失减少速度较慢，注意力头逐渐倾向于各自的任务；涌现阶段，在这个阶段，每个头选择一个单独的任务，损失迅速减少；和收敛阶段，在这个阶段，注意力参数收敛到一个极限。此外，我们证明了梯度流在学习极限模型方面的最优性。

    arXiv:2402.19442v1 Announce Type: cross  Abstract: We study the dynamics of gradient flow for training a multi-head softmax attention model for in-context learning of multi-task linear regression. We establish the global convergence of gradient flow under suitable choices of initialization. In addition, we prove that an interesting "task allocation" phenomenon emerges during the gradient flow dynamics, where each attention head focuses on solving a single task of the multi-task model. Specifically, we prove that the gradient flow dynamics can be split into three phases -- a warm-up phase where the loss decreases rather slowly and the attention heads gradually build up their inclination towards individual tasks, an emergence phase where each head selects a single task and the loss rapidly decreases, and a convergence phase where the attention parameters converge to a limit. Furthermore, we prove the optimality of gradient flow in the sense that the limiting model learned by gradient flo
    
[^5]: 用于异质过度离散计数时间序列的负二项随机Gamma马尔可夫过程

    Negative-Binomial Randomized Gamma Markov Processes for Heterogeneous Overdispersed Count Time Series

    [https://arxiv.org/abs/2402.18995](https://arxiv.org/abs/2402.18995)

    提出了一种负二项随机Gamma马尔可夫过程，用于改进异质过度离散计数时间序列的预测性能，并加快推断算法的收敛速度。

    

    对于计数值时间序列的建模自然地在物理和社会领域中引起越来越多的关注。Poisson gamma动态系统（PGDSs）是新开发的方法，可以很好地捕捉计数序列背后表现出的明显的潜在转换结构和突发动态。特别是，与基于经典线性动态系统（LDS）的方法相比，PGDSs在数据填充和预测方面表现出优越性能。尽管具有这些优势，PGDS不能捕捉基础动态过程的异质过度离散行为。为了减轻这一缺陷，我们提出了一种负二项随机Gamma马尔可夫过程，它不仅显著改善了所提出的动态系统的预测性能，还促进了推断算法的快速收敛。此外，我们开发了估计因子结构和图结构的方法。

    arXiv:2402.18995v1 Announce Type: cross  Abstract: Modeling count-valued time series has been receiving increasing attention since count time series naturally arise in physical and social domains. Poisson gamma dynamical systems (PGDSs) are newly-developed methods, which can well capture the expressive latent transition structure and bursty dynamics behind count sequences. In particular, PGDSs demonstrate superior performance in terms of data imputation and prediction, compared with canonical linear dynamical system (LDS) based methods. Despite these advantages, PGDS cannot capture the heterogeneous overdispersed behaviours of the underlying dynamic processes. To mitigate this defect, we propose a negative-binomial-randomized gamma Markov process, which not only significantly improves the predictive performance of the proposed dynamical system, but also facilitates the fast convergence of the inference algorithm. Moreover, we develop methods to estimate both factor-structured and graph
    
[^6]: 半监督 U-统计量

    Semi-Supervised U-statistics

    [https://arxiv.org/abs/2402.18921](https://arxiv.org/abs/2402.18921)

    介绍了一种半监督 U-统计量，利用大量未标记数据，获得了渐近正态分布的性质，并通过有效整合各种强大预测工具实现了明显的效率提升。

    

    arXiv:2402.18921v1 通报类型: 跨领域  摘要: 半监督数据集在多个领域中普遍存在，其中获得完全标记数据成本高昂或耗时。这类数据集的普遍存在一直推动着对利用未标记数据潜力的新工具和方法的需求。为了满足这种需求，我们介绍了受益于大量未标记数据的半监督 U-统计量，并研究了它们的统计特性。我们展示了所提出的方法渐近地服从正态分布，并且通过有效地将各种强大的预测工具整合到框架中，获得了明显的效率提升，超过了经典 U-统计量。为了理解问题的根本困难，我们在半监督设置中推导了极小极大下界，并展示了在正则条件下我们的过程是半参数有效的。此外，针对双变量核函数，我们提出了一种优化的方法，胜过了经典的 U-统计量。

    arXiv:2402.18921v1 Announce Type: cross  Abstract: Semi-supervised datasets are ubiquitous across diverse domains where obtaining fully labeled data is costly or time-consuming. The prevalence of such datasets has consistently driven the demand for new tools and methods that exploit the potential of unlabeled data. Responding to this demand, we introduce semi-supervised U-statistics enhanced by the abundance of unlabeled data, and investigate their statistical properties. We show that the proposed approach is asymptotically Normal and exhibits notable efficiency gains over classical U-statistics by effectively integrating various powerful prediction tools into the framework. To understand the fundamental difficulty of the problem, we derive minimax lower bounds in semi-supervised settings and showcase that our procedure is semi-parametrically efficient under regularity conditions. Moreover, tailored to bivariate kernels, we propose a refined approach that outperforms the classical U-st
    
[^7]: 随机对照试验中逻辑回归的预后辅助校正

    Prognostic Covariate Adjustment for Logistic Regression in Randomized Controlled Trials

    [https://arxiv.org/abs/2402.18900](https://arxiv.org/abs/2402.18900)

    使用预后评分调整可以提高逻辑回归中条件比值的Wald检验的能力

    

    通过使用生成人工智能（AI）算法预测对照结果，即所谓的预后评分，来进行逻辑回归中的预后评分调整，以增加固定样本量条件下条件比值的Wald检验力量。

    arXiv:2402.18900v1 Announce Type: cross  Abstract: Randomized controlled trials (RCTs) with binary primary endpoints introduce novel challenges for inferring the causal effects of treatments. The most significant challenge is non-collapsibility, in which the conditional odds ratio estimand under covariate adjustment differs from the unconditional estimand in the logistic regression analysis of RCT data. This issue gives rise to apparent paradoxes, such as the variance of the estimator for the conditional odds ratio from a covariate-adjusted model being greater than the variance of the estimator from the unadjusted model. We address this challenge in the context of adjustment based on predictions of control outcomes from generative artificial intelligence (AI) algorithms, which are referred to as prognostic scores. We demonstrate that prognostic score adjustment in logistic regression increases the power of the Wald test for the conditional odds ratio under a fixed sample size, or alter
    
[^8]: 监督对比表示学习：具有不受限制特征的景观分析

    Supervised Contrastive Representation Learning: Landscape Analysis with Unconstrained Features

    [https://arxiv.org/abs/2402.18884](https://arxiv.org/abs/2402.18884)

    通过监督对比表示学习，在超参数化的深度神经网络中研究解决方案，揭示了最小化SC损失的全局最小值和唯一最小化器。

    

    最近的研究发现，在超参数化的深度神经网络中，经过零训练误差训练后的网络，在最后一层呈现出严格的结构模式，被称为神经坍塌（NC）。这些结果表明，在这种网络中，最终隐藏层输出在训练集上显示出最小的类内变化。虽然现有研究在交叉熵损失下广泛探讨了这一现象，但关于其对应的对比损失——监督对比（SC）损失的研究较少。本文通过NC的视角，采用分析方法研究了优化SC损失所得解决方案。我们采用不受限制特征模型（UFM）作为代表性代理，揭示了在充分超参数化的深度网络中衍生的与NC相关现象。我们展示了，尽管SC损失最小化是非凸的，但所有局部最小值都是全局最小值。此外，最小化器是唯一的

    arXiv:2402.18884v1 Announce Type: new  Abstract: Recent findings reveal that over-parameterized deep neural networks, trained beyond zero training-error, exhibit a distinctive structural pattern at the final layer, termed as Neural-collapse (NC). These results indicate that the final hidden-layer outputs in such networks display minimal within-class variations over the training set. While existing research extensively investigates this phenomenon under cross-entropy loss, there are fewer studies focusing on its contrastive counterpart, supervised contrastive (SC) loss. Through the lens of NC, this paper employs an analytical approach to study the solutions derived from optimizing the SC loss. We adopt the unconstrained features model (UFM) as a representative proxy for unveiling NC-related phenomena in sufficiently over-parameterized deep networks. We show that, despite the non-convexity of SC loss minimization, all local minima are global minima. Furthermore, the minimizer is unique (
    
[^9]: 在处方和预测中应用0-1神经网络

    Applications of 0-1 Neural Networks in Prescription and Prediction

    [https://arxiv.org/abs/2402.18851](https://arxiv.org/abs/2402.18851)

    引入了处方网络（PNNs）这种新型神经网络，通过混合整数规划训练，结合反事实估计，在医疗决策中展现出优于现有方法的表现，可优化治疗策略，并具有更大的可解释性和更复杂的策略编码能力。

    

    医疗决策中的一个关键挑战是在有限的观察数据下学习针对患者的治疗策略。为了解决这个问题，我们引入了处方网络（PNNs），这是用混合整数规划训练的浅层0-1神经网络，可以与反事实估计一起在中等数据情况下优化策略。这些模型比深度神经网络具有更大的可解释性，并且可以编码比常见模型（如决策树）更复杂的策略。我们展示了PNNs在合成数据实验和产后高血压治疗分配案例研究中表现优于现有方法。特别是，PNNs被证明能够产生可降低高血压峰值的治疗策略。

    arXiv:2402.18851v1 Announce Type: cross  Abstract: A key challenge in medical decision making is learning treatment policies for patients with limited observational data. This challenge is particularly evident in personalized healthcare decision-making, where models need to take into account the intricate relationships between patient characteristics, treatment options, and health outcomes. To address this, we introduce prescriptive networks (PNNs), shallow 0-1 neural networks trained with mixed integer programming that can be used with counterfactual estimation to optimize policies in medium data settings. These models offer greater interpretability than deep neural networks and can encode more complex policies than common models such as decision trees. We show that PNNs can outperform existing methods in both synthetic data experiments and in a case study of assigning treatments for postpartum hypertension. In particular, PNNs are shown to produce policies that could reduce peak bloo
    
[^10]: VEC-SBM：具有向量边缘协变量的最优社区检测

    VEC-SBM: Optimal Community Detection with Vectorial Edges Covariates

    [https://arxiv.org/abs/2402.18805](https://arxiv.org/abs/2402.18805)

    该论文介绍了一种名为VEC-SBM的算法，可以在社交网络中使用向量边缘协变量来最优地检测社区，证明了在社区检测过程中利用边缘信息的附加价值。

    

    社交网络通常与丰富的边缘信息相关联，例如文本和图像。虽然已经开发了许多方法来从成对互动中识别社区，但它们通常忽略了这种边缘信息。在这项工作中，我们研究了随机块模型（SBM）的扩展，这是一种广泛用于社区检测的统计框架，它集成了向量边缘协变量：向量边缘协变量随机块模型（VEC-SBM）。我们提出了一种基于迭代细化技术的新算法，并展示了它在VEC-SBM下的最优恢复潜在社区的能力。此外，我们严格评估了在社区检测过程中利用边缘信息的附加价值。我们通过对合成和半合成数据进行数值实验来补充我们的理论结果。

    arXiv:2402.18805v1 Announce Type: cross  Abstract: Social networks are often associated with rich side information, such as texts and images. While numerous methods have been developed to identify communities from pairwise interactions, they usually ignore such side information. In this work, we study an extension of the Stochastic Block Model (SBM), a widely used statistical framework for community detection, that integrates vectorial edges covariates: the Vectorial Edges Covariates Stochastic Block Model (VEC-SBM). We propose a novel algorithm based on iterative refinement techniques and show that it optimally recovers the latent communities under the VEC-SBM. Furthermore, we rigorously assess the added value of leveraging edge's side information in the community detection process. We complement our theoretical results with numerical experiments on synthetic and semi-synthetic data.
    
[^11]: BlockEcho：保留长距离依赖关系用于填补块状缺失数据

    BlockEcho: Retaining Long-Range Dependencies for Imputing Block-Wise Missing Data

    [https://arxiv.org/abs/2402.18800](https://arxiv.org/abs/2402.18800)

    该论文提出了一种名为BlockEcho的新矩阵填充方法，通过将矩阵分解和生成对抗网络相结合，创造性地保留了原始矩阵中的长距离依赖关系，以解决块状缺失数据对数据插值和预测能力的挑战。

    

    arXiv:2402.18800v1 类型：新  摘要：块状缺失数据在实际数据填补任务中带来了显著挑战。与分散的缺失数据相比，块状缺失数据加剧了对后续分析和机器学习任务的不利影响，因为缺乏局部相邻元素显著降低了插值能力和预测能力。然而，这个问题尚未得到充分关注。由于过度依赖邻近元素进行预测，大多数SOTA矩阵填充方法显示出较低的有效性。我们系统地分析了这个问题，并提出了一种新颖的矩阵填充方法“BlockEcho”以提供更全面的解决方案。该方法创造性地将矩阵分解（MF）与生成对抗网络（GAN）相结合，以明确保留原始矩阵中的长距离元素间关系。此外，我们还为GAN引入了一个额外的鉴别器，比较生成器的中间进程。

    arXiv:2402.18800v1 Announce Type: new  Abstract: Block-wise missing data poses significant challenges in real-world data imputation tasks. Compared to scattered missing data, block-wise gaps exacerbate adverse effects on subsequent analytic and machine learning tasks, as the lack of local neighboring elements significantly reduces the interpolation capability and predictive power. However, this issue has not received adequate attention. Most SOTA matrix completion methods appeared less effective, primarily due to overreliance on neighboring elements for predictions. We systematically analyze the issue and propose a novel matrix completion method ``BlockEcho" for a more comprehensive solution. This method creatively integrates Matrix Factorization (MF) within Generative Adversarial Networks (GAN) to explicitly retain long-distance inter-element relationships in the original matrix. Besides, we incorporate an additional discriminator for GAN, comparing the generator's intermediate progre
    
[^12]: 使用梯度下降学习关联记忆

    Learning Associative Memories with Gradient Descent

    [https://arxiv.org/abs/2402.18724](https://arxiv.org/abs/2402.18724)

    该论文研究了一个关联记忆模块的训练动态，通过理论和实验揭示了在过参数化和欠参数化情况下的学习动态和误差特性。

    

    这项工作主要关注存储标记嵌入的外积的一个关联记忆模块的训练动态。我们将这个问题简化为研究一个粒子系统，这些粒子根据数据分布的特性以及嵌入之间的相关性进行交互。通过理论和实验，我们提供了一些见解。在过参数化的情况下，我们获得了“分类边界”的对数增长。然而，我们表明标记频率的不平衡和由相关嵌入导致的内存干扰会导致振荡的瞬态区域。振荡在步长较大时更为明显，这可能导致良性损失峰，尽管这些学习率加速了动态并加速了渐近收敛。在欠参数化的情况下，我们阐明了交叉熵损失如何导致次优的记忆方案。最后，我们评估了我们发现的在小规模Tr上的有效性。

    arXiv:2402.18724v1 Announce Type: cross  Abstract: This work focuses on the training dynamics of one associative memory module storing outer products of token embeddings. We reduce this problem to the study of a system of particles, which interact according to properties of the data distribution and correlations between embeddings. Through theory and experiments, we provide several insights. In overparameterized regimes, we obtain logarithmic growth of the ``classification margins.'' Yet, we show that imbalance in token frequencies and memory interferences due to correlated embeddings lead to oscillatory transitory regimes. The oscillations are more pronounced with large step sizes, which can create benign loss spikes, although these learning rates speed up the dynamics and accelerate the asymptotic convergence. In underparameterized regimes, we illustrate how the cross-entropy loss can lead to suboptimal memorization schemes. Finally, we assess the validity of our findings on small Tr
    
[^13]: 从边际推断动态网络的方法：迭代比例拟合

    Inferring Dynamic Networks from Marginals with Iterative Proportional Fitting

    [https://arxiv.org/abs/2402.18697](https://arxiv.org/abs/2402.18697)

    通过识别一个生成网络模型，我们建立了一个设置，IPF可以恢复最大似然估计，揭示了关于在这种设置中使用IPF的隐含假设，并可以为IPF的参数估计提供结构相关的误差界。

    

    来自现实数据约束的常见网络推断问题是如何从时间聚合的邻接矩阵和时间变化边际（即行向量和列向量之和）推断动态网络。先前的方法为了解决这个问题重新利用了经典的迭代比例拟合（IPF）过程，也称为Sinkhorn算法，并取得了令人满意的经验结果。然而，使用IPF的统计基础尚未得到很好的理解：在什么情况下，IPF提供了从边际准确估计动态网络的原则性，以及它在多大程度上估计了网络？在这项工作中，我们确定了这样一个设置，通过识别一个生成网络模型，IPF可以恢复其最大似然估计。我们的模型揭示了关于在这种设置中使用IPF的隐含假设，并使得可以进行新的分析，如有关IPF参数估计的结构相关误差界。当IPF失败时

    arXiv:2402.18697v1 Announce Type: cross  Abstract: A common network inference problem, arising from real-world data constraints, is how to infer a dynamic network from its time-aggregated adjacency matrix and time-varying marginals (i.e., row and column sums). Prior approaches to this problem have repurposed the classic iterative proportional fitting (IPF) procedure, also known as Sinkhorn's algorithm, with promising empirical results. However, the statistical foundation for using IPF has not been well understood: under what settings does IPF provide principled estimation of a dynamic network from its marginals, and how well does it estimate the network? In this work, we establish such a setting, by identifying a generative network model whose maximum likelihood estimates are recovered by IPF. Our model both reveals implicit assumptions on the use of IPF in such settings and enables new analyses, such as structure-dependent error bounds on IPF's parameter estimates. When IPF fails to c
    
[^14]: 用于满足多样用户偏好的算术控制LLMs：具有多目标奖励的方向偏好对齐

    Arithmetic Control of LLMs for Diverse User Preferences: Directional Preference Alignment with Multi-Objective Rewards

    [https://arxiv.org/abs/2402.18571](https://arxiv.org/abs/2402.18571)

    提出了方向偏好对齐（DPA）框架，通过多目标奖励模拟不同偏好配置，以实现用户相关的偏好控制。

    

    针对大型语言模型（LLMs）的精细控制仍然是一个重要挑战，阻碍了它们适应各种用户需求。本文提出了方向偏好对齐（DPA）框架，通过多目标奖励建模来表示多样化的偏好配置，将用户偏好建模为奖励空间中的方向（即单位向量）以实现用户相关的偏好控制。

    arXiv:2402.18571v1 Announce Type: cross  Abstract: Fine-grained control over large language models (LLMs) remains a significant challenge, hindering their adaptability to diverse user needs. While Reinforcement Learning from Human Feedback (RLHF) shows promise in aligning LLMs, its reliance on scalar rewards often limits its ability to capture diverse user preferences in real-world applications. To address this limitation, we introduce the Directional Preference Alignment (DPA) framework. Unlike the scalar-reward RLHF, DPA incorporates multi-objective reward modeling to represent diverse preference profiles. Additionally, DPA models user preferences as directions (i.e., unit vectors) in the reward space to achieve user-dependent preference control. Our method involves training a multi-objective reward model and then fine-tuning the LLM with a preference-conditioned variant of Rejection Sampling Finetuning (RSF), an RLHF method adopted by Llama 2. This method enjoys a better performance
    
[^15]: RNNs还不是Transformer：在上下文检索中的关键瓶颈

    RNNs are not Transformers (Yet): The Key Bottleneck on In-context Retrieval

    [https://arxiv.org/abs/2402.18510](https://arxiv.org/abs/2402.18510)

    本文研究了RNNs和Transformer在处理算法问题时的表现能力差距，发现RNNs存在关键瓶颈，即无法完美地从上下文中检索信息，导致无法像Transformer那样轻松解决需要这种能力的任务。

    

    本文探讨循环神经网络（RNNs）和Transformer在解决算法问题时的表示能力差距。我们重点关注RNNs是否能在处理长序列时，通过Chain-of-Thought (CoT)提示，与Transformer的性能相匹配。我们的理论分析显示CoT可以改进RNNs，但无法弥补与Transformer之间的差距。关键瓶颈在于RNNs无法完全从上下文中检索信息，即使经过CoT的增强：对于几个明确或隐式需要这种能力的任务，如联想召回和确定图是否为树，我们证明RNNs表达能力不足以解决这些任务，而Transformer可以轻松解决。相反，我们证明采用增强RNNs上下文检索能力的技术，包括

    arXiv:2402.18510v1 Announce Type: cross  Abstract: This paper investigates the gap in representation powers of Recurrent Neural Networks (RNNs) and Transformers in the context of solving algorithmic problems. We focus on understanding whether RNNs, known for their memory efficiency in handling long sequences, can match the performance of Transformers, particularly when enhanced with Chain-of-Thought (CoT) prompting. Our theoretical analysis reveals that CoT improves RNNs but is insufficient to close the gap with Transformers. A key bottleneck lies in the inability of RNNs to perfectly retrieve information from the context, even with CoT: for several tasks that explicitly or implicitly require this capability, such as associative recall and determining if a graph is a tree, we prove that RNNs are not expressive enough to solve the tasks while Transformers can solve them with ease. Conversely, we prove that adopting techniques to enhance the in-context retrieval capability of RNNs, inclu
    
[^16]: 用于非对数凹分布的零阶采样方法：通过去噪扩散缓解亚稳定性

    Zeroth-Order Sampling Methods for Non-Log-Concave Distributions: Alleviating Metastability by Denoising Diffusion

    [https://arxiv.org/abs/2402.17886](https://arxiv.org/abs/2402.17886)

    本文提出了一种基于去噪扩散过程的零阶扩散蒙特卡洛算法，克服了非对数凹分布采样中的亚稳定性问题，并证明其采样精度具有倒多项式依赖。

    

    这篇论文考虑了基于其非对数凹分布未归一化密度查询的采样问题。首先描述了一个基于模拟去噪扩散过程的框架，即扩散蒙特卡洛（DMC），其得分函数通过通用蒙特卡洛估计器逼近。DMC是一个基于神谕的元算法，其中神谕是假设可以访问生成蒙特卡洛分数估计器的样本的访问。然后，我们提供了一个基于拒绝采样的这个神谕的实现，这将DMC转化为一个真正的算法，称为零阶扩散蒙特卡洛（ZOD-MC）。我们通过首先构建一个通用框架，即DMC的性能保证，而不假设目标分布为对数凹或满足任何等周不等式，提供了收敛分析。然后我们证明ZOD-MC对所需采样精度具有倒多项式依赖，尽管仍然受到...

    arXiv:2402.17886v1 Announce Type: cross  Abstract: This paper considers the problem of sampling from non-logconcave distribution, based on queries of its unnormalized density. It first describes a framework, Diffusion Monte Carlo (DMC), based on the simulation of a denoising diffusion process with its score function approximated by a generic Monte Carlo estimator. DMC is an oracle-based meta-algorithm, where its oracle is the assumed access to samples that generate a Monte Carlo score estimator. Then we provide an implementation of this oracle, based on rejection sampling, and this turns DMC into a true algorithm, termed Zeroth-Order Diffusion Monte Carlo (ZOD-MC). We provide convergence analyses by first constructing a general framework, i.e. a performance guarantee for DMC, without assuming the target distribution to be log-concave or satisfying any isoperimetric inequality. Then we prove that ZOD-MC admits an inverse polynomial dependence on the desired sampling accuracy, albeit sti
    
[^17]: 逻辑回归的可证实准确性随机抽样算法

    A Provably Accurate Randomized Sampling Algorithm for Logistic Regression

    [https://arxiv.org/abs/2402.16326](https://arxiv.org/abs/2402.16326)

    提出了一种逻辑回归问题的简单随机抽样算法，通过随机矩阵乘法实现高质量逼近估计概率和模型整体差异性。

    

    在统计学和机器学习中，逻辑回归是一种广泛应用于二分类任务的监督学习技术。当观测数量远远超过预测变量数量时，我们提出了一种简单的基于随机抽样的逻辑回归问题算法，保证高质量逼近估计概率和模型整体差异性。我们的分析建立在两个简单的结构条件基础上，这两个条件可归结为随机矩阵乘法，是随机化数值线性代数的基本且深入理解的基元。当利用杠杆分数对观测进行抽样时，我们分析了逻辑回归的估计概率属性，并证明准确逼近可以通过远小于总观测数的样本实现。为了进一步验证我们的理论发现，

    arXiv:2402.16326v1 Announce Type: cross  Abstract: In statistics and machine learning, logistic regression is a widely-used supervised learning technique primarily employed for binary classification tasks. When the number of observations greatly exceeds the number of predictor variables, we present a simple, randomized sampling-based algorithm for logistic regression problem that guarantees high-quality approximations to both the estimated probabilities and the overall discrepancy of the model. Our analysis builds upon two simple structural conditions that boil down to randomized matrix multiplication, a fundamental and well-understood primitive of randomized numerical linear algebra. We analyze the properties of estimated probabilities of logistic regression when leverage scores are used to sample observations, and prove that accurate approximations can be achieved with a sample whose size is much smaller than the total number of observations. To further validate our theoretical findi
    
[^18]: 从实例级的自我注意力Hawkes过程中学习格兰杰因果关系

    Learning Granger Causality from Instance-wise Self-attentive Hawkes Processes

    [https://arxiv.org/abs/2402.03726](https://arxiv.org/abs/2402.03726)

    本论文提出了一种名为ISAHP的深度学习框架，可以从异步、相互依赖的多类型事件序列中无监督地学习实例级的格兰杰因果关系。它是第一个满足格兰杰因果关系要求的神经点过程模型，并利用变压器的自我注意机制来实现格兰杰因果关系的推断。

    

    我们解决了从异步、相互依赖的多类型事件序列中学习格兰杰因果关系的问题。特别是，我们对以无监督的方式发现实例级的因果结构感兴趣。实例级因果关系识别单个事件之间的因果关系，为决策提供了更精细化的信息。现有文献中的工作要么需要强加一些假设，比如强加到强度函数中的线性假设，要么启发式地定义模型参数，这些不一定满足格兰杰因果关系的要求。我们提出了一种新颖的深度学习框架，即实例级自我注意力Hawkes过程（ISAHP），可以直接推断事件实例级的格兰杰因果关系。ISAHP是第一个满足格兰杰因果关系要求的神经点过程模型。它利用了变压器的自我注意机制，与格兰杰因果关系的原理相一致。我们通过实验证明了ISAHP的有效性和优越性。

    We address the problem of learning Granger causality from asynchronous, interdependent, multi-type event sequences. In particular, we are interested in discovering instance-level causal structures in an unsupervised manner. Instance-level causality identifies causal relationships among individual events, providing more fine-grained information for decision-making. Existing work in the literature either requires strong assumptions, such as linearity in the intensity function, or heuristically defined model parameters that do not necessarily meet the requirements of Granger causality. We propose Instance-wise Self-Attentive Hawkes Processes (ISAHP), a novel deep learning framework that can directly infer the Granger causality at the event instance level. ISAHP is the first neural point process model that meets the requirements of Granger causality. It leverages the self-attention mechanism of the transformer to align with the principles of Granger causality. We empirically demonstrate th
    
[^19]: 在医疗AI模型中检测算法偏倚

    Detecting algorithmic bias in medical AI-models

    [https://arxiv.org/abs/2312.02959](https://arxiv.org/abs/2312.02959)

    本文提出了一种创新的框架，用于检测医疗AI决策支持系统中的算法偏倚，通过采用CART算法有效地识别医疗AI模型中的潜在偏倚，并在合成数据实验和真实临床环境中验证了其有效性。

    

    随着机器学习和人工智能医疗决策支持系统日益普及，确保这些系统以公平、公正的方式提供患者结果变得同样重要。本文提出了一种创新的框架，用于检测医疗AI决策支持系统中的算法偏倚区域。我们的方法通过采用分类与回归树（CART）算法，在脓毒症预测背景下有效地识别医疗AI模型中的潜在偏倚。我们通过进行一系列合成数据实验验证了我们的方法，展示了其在受控环境中准确估计偏倚区域的能力。这一概念的有效性通过使用亚特兰大乔治亚州格雷迪纪念医院的电子病历进行实验进一步得到验证。这些测试展示了我们策略在临床中的实际应用。

    arXiv:2312.02959v3 Announce Type: replace-cross  Abstract: With the growing prevalence of machine learning and artificial intelligence-based medical decision support systems, it is equally important to ensure that these systems provide patient outcomes in a fair and equitable fashion. This paper presents an innovative framework for detecting areas of algorithmic bias in medical-AI decision support systems. Our approach efficiently identifies potential biases in medical-AI models, specifically in the context of sepsis prediction, by employing the Classification and Regression Trees (CART) algorithm. We verify our methodology by conducting a series of synthetic data experiments, showcasing its ability to estimate areas of bias in controlled settings precisely. The effectiveness of the concept is further validated by experiments using electronic medical records from Grady Memorial Hospital in Atlanta, Georgia. These tests demonstrate the practical implementation of our strategy in a clini
    
[^20]: 随机向量均值的时间均匀置信球

    Time-Uniform Confidence Spheres for Means of Random Vectors

    [https://arxiv.org/abs/2311.08168](https://arxiv.org/abs/2311.08168)

    该研究提出了时间均匀置信球序列，可以同时高概率地包含各种样本量下随机向量的均值，并针对不同分布假设进行了扩展和统一分析。

    

    我们推导并研究了时间均匀置信球——包含随机向量均值并且跨越所有样本量具有很高概率的置信球序列（CSSs）。受Catoni和Giulini原始工作启发，我们统一并扩展了他们的分析，涵盖顺序设置并处理各种分布假设。我们的结果包括有界随机向量的经验伯恩斯坦CSS（导致新颖的经验伯恩斯坦置信区间，渐近宽度按照真实未知方差成比例缩放）、用于子-$\psi$随机向量的CSS（包括子伽马、子泊松和子指数分布）、和用于重尾随机向量（仅有两阶矩）的CSS。最后，我们提供了两个抵抗Huber噪声污染的CSS。第一个是我们经验伯恩斯坦CSS的鲁棒版本，第二个扩展了单变量序列最近的工作。

    arXiv:2311.08168v2 Announce Type: replace-cross  Abstract: We derive and study time-uniform confidence spheres -- confidence sphere sequences (CSSs) -- which contain the mean of random vectors with high probability simultaneously across all sample sizes. Inspired by the original work of Catoni and Giulini, we unify and extend their analysis to cover both the sequential setting and to handle a variety of distributional assumptions. Our results include an empirical-Bernstein CSS for bounded random vectors (resulting in a novel empirical-Bernstein confidence interval with asymptotic width scaling proportionally to the true unknown variance), CSSs for sub-$\psi$ random vectors (which includes sub-gamma, sub-Poisson, and sub-exponential), and CSSs for heavy-tailed random vectors (two moments only). Finally, we provide two CSSs that are robust to contamination by Huber noise. The first is a robust version of our empirical-Bernstein CSS, and the second extends recent work in the univariate se
    
[^21]: 经过鲁棒优化输运的推断：理论与方法

    Inference via robust optimal transportation: theory and methods

    [https://arxiv.org/abs/2301.06297](https://arxiv.org/abs/2301.06297)

    通过鲁棒Wasserstein距离处理输运问题，探讨了与$W_1$的关联，推导了集中不等式，并提出最小距离估计器以及其统计保证。

    

    最优输运理论及相关的$p$-Wasserstein距离（$W_p$，$p\geq 1$）在统计学和机器学习中被广泛应用。尽管它们很受欢迎，但基于这些工具的推断存在一些问题。 为了应对这些问题，首先我们考虑了原始输运问题的鲁棒版本，并展示其定义了依赖于调节参数$\lambda > 0$的{鲁棒Wasserstein距离}，$W^{(\lambda)}$。其次，我们讨论了$W_1$和$W^{(\lambda)}$之间的联系，并研究了其关键的测度论方面。第三，我们推导了$W^{(\lambda)}$的一些集中不等式。第四，我们利用$W^{(\lambda)}$定义了最小距离估计器，提供了它们的统计保证，并说明了如何应用所推导的集中不等式到数据中。

    arXiv:2301.06297v4 Announce Type: replace-cross  Abstract: Optimal transportation theory and the related $p$-Wasserstein distance ($W_p$, $p\geq 1$) are widely-applied in statistics and machine learning. In spite of their popularity, inference based on these tools has some issues. For instance, it is sensitive to outliers and it may not be even defined when the underlying model has infinite moments. To cope with these problems, first we consider a robust version of the primal transportation problem and show that it defines the {robust Wasserstein distance}, $W^{(\lambda)}$, depending on a tuning parameter $\lambda > 0$. Second, we illustrate the link between $W_1$ and $W^{(\lambda)}$ and study its key measure theoretic aspects. Third, we derive some concentration inequalities for $W^{(\lambda)}$. Fourth, we use $W^{(\lambda)}$ to define minimum distance estimators, we provide their statistical guarantees and we illustrate how to apply the derived concentration inequalities for a data d
    
[^22]: 具有主动学习的神经Galerkin方案用于高维演化方程

    Neural Galerkin Schemes with Active Learning for High-Dimensional Evolution Equations

    [https://arxiv.org/abs/2203.01360](https://arxiv.org/abs/2203.01360)

    通过神经Galerkin方案结合深度学习和主动学习，能够自主生成训练数据用于高维偏微分方程的数值求解

    

    深度神经网络已被证明能够在高维度中提供准确的函数逼近。然而，拟合网络参数需要信息丰富的训练数据，在科学和工程应用中往往难以收集。本文提出了基于深度学习的神经Galerkin方案，通过主动学习生成训练数据，用于数值求解高维偏微分方程。神经Galerkin方案基于Dirac-Frenkel变分原理，通过随时间顺序最小化残差来训练网络，这使得能够以自主、动态描述的方式收集新的训练数据，以指导偏微分方程描述的动态。这与其他机器学习方法形成对比，其他方法旨在全局时间内拟合网络参数，而不考虑训练数据获取。我们的发现是主动形式的

    arXiv:2203.01360v4 Announce Type: replace-cross  Abstract: Deep neural networks have been shown to provide accurate function approximations in high dimensions. However, fitting network parameters requires informative training data that are often challenging to collect in science and engineering applications. This work proposes Neural Galerkin schemes based on deep learning that generate training data with active learning for numerically solving high-dimensional partial differential equations. Neural Galerkin schemes build on the Dirac-Frenkel variational principle to train networks by minimizing the residual sequentially over time, which enables adaptively collecting new training data in a self-informed manner that is guided by the dynamics described by the partial differential equations. This is in contrast to other machine learning methods that aim to fit network parameters globally in time without taking into account training data acquisition. Our finding is that the active form of 
    
[^23]: 离线检测平稳图信号均值变化点

    Offline detection of change-points in the mean for stationary graph signals

    [https://arxiv.org/abs/2006.10628](https://arxiv.org/abs/2006.10628)

    提出了一种离线方法用于检测图信号中均值变化点，通过在频谱域解决问题，充分利用了稀疏性，采用模型选择方法自动确定变点的数量，并给出了非渐近oracle不等式的证明。

    

    这篇论文解决了图信号流分割的问题：我们旨在检测已知图上的多变量信号均值变化。我们提出了一种离线方法，依赖于图信号平稳性的概念，并允许将问题从原始顶点域转换到频谱域（图傅里叶变换），在那里更容易解决。虽然在实际应用中获得的频谱表示是稀疏的，但据我们所知，这种特性在现有相关文献中尚未得到充分利用。我们的变点检测方法采用模型选择方法，考虑了频谱表示的稀疏性，并自动确定变点的数量。我们的检测器伴随着非渐近优等性的证明。数值实验展示了该方法的性能。

    arXiv:2006.10628v2 Announce Type: replace  Abstract: This paper addresses the problem of segmenting a stream of graph signals: we aim to detect changes in the mean of a multivariate signal defined over the nodes of a known graph. We propose an offline method that relies on the concept of graph signal stationarity and allows the convenient translation of the problem from the original vertex domain to the spectral domain (Graph Fourier Transform), where it is much easier to solve. Although the obtained spectral representation is sparse in real applications, to the best of our knowledge this property has not been sufficiently exploited in the existing related literature. Our change-point detection method adopts a model selection approach that takes into account the sparsity of the spectral representation and determines automatically the number of change-points. Our detector comes with a proof of a non-asymptotic oracle inequality. Numerical experiments demonstrate the performance of the p
    
[^24]: 委员会机器：学习两层神经网络中计算到统计学差距的研究

    The committee machine: Computational to statistical gaps in learning a two-layers neural network

    [https://arxiv.org/abs/1806.05451](https://arxiv.org/abs/1806.05451)

    介绍了对于两层神经网络模型委员会机器的严格理论基础和近似消息传递算法，揭示了计算到统计学差距。

    

    过去，统计物理学中的启发式工具被用来定位相变并计算多层神经网络中教师-学生场景中的最优学习和泛化错误。在这篇论文中，我们为一个名为委员会机器的两层神经网络模型提供了这些方法的严格理论基础。我们还引入了一个委员会机器的近似消息传递（AMP）算法版本，允许在多种参数下以多项式时间执行最佳学习。我们发现在某些情况下，虽然AMP算法无法实现，但在信息理论上可以实现低泛化错误率，这强烈暗示对于这些情况不存在有效算法，揭示了一个巨大的计算差距。

    arXiv:1806.05451v3 Announce Type: replace  Abstract: Heuristic tools from statistical physics have been used in the past to locate the phase transitions and compute the optimal learning and generalization errors in the teacher-student scenario in multi-layer neural networks. In this contribution, we provide a rigorous justification of these approaches for a two-layers neural network model called the committee machine. We also introduce a version of the approximate message passing (AMP) algorithm for the committee machine that allows to perform optimal learning in polynomial time for a large set of parameters. We find that there are regimes in which a low generalization error is information-theoretically achievable while the AMP algorithm fails to deliver it, strongly suggesting that no efficient algorithm exists for those cases, and unveiling a large computational gap.
    
[^25]: 将循环引入人类：协作和可解释的贝叶斯优化

    Looping in the Human: Collaborative and Explainable Bayesian Optimization. (arXiv:2310.17273v1 [cs.LG])

    [http://arxiv.org/abs/2310.17273](http://arxiv.org/abs/2310.17273)

    协作和可解释的贝叶斯优化框架(CoExBO)在贝叶斯优化中引入了循环，平衡了人工智能和人类的合作关系。它利用偏好学习将用户见解融合到优化中，解释每次迭代的候选选择，从而增强用户对优化过程的信任，并提供无害保证。

    

    像许多优化器一样，贝叶斯优化在获得用户信任方面常常存在不足，因为其不透明性。虽然已经尝试开发面向人类的优化器，但它们通常假设用户知识是明确且无误的，并主要将用户作为优化过程的监督者。我们放宽了这些假设，提出了一种更平衡的人工智能和人类合作伙伴关系，即我们的协作和可解释的贝叶斯优化（CoExBO）框架。CoExBO使用偏好学习来无缝地将人类见解整合到优化中，从而产生与用户使用偏好一致的算法建议。CoExBO解释其每次迭代的候选选择，以培养信任，使用户更清楚地掌握优化的过程。此外，CoExBO提供无害保证，允许用户犯错误；即使在极端对抗性干扰下，算法也会渐进地收敛。

    Like many optimizers, Bayesian optimization often falls short of gaining user trust due to opacity. While attempts have been made to develop human-centric optimizers, they typically assume user knowledge is well-specified and error-free, employing users mainly as supervisors of the optimization process. We relax these assumptions and propose a more balanced human-AI partnership with our Collaborative and Explainable Bayesian Optimization (CoExBO) framework. Instead of explicitly requiring a user to provide a knowledge model, CoExBO employs preference learning to seamlessly integrate human insights into the optimization, resulting in algorithmic suggestions that resonate with user preference. CoExBO explains its candidate selection every iteration to foster trust, empowering users with a clearer grasp of the optimization. Furthermore, CoExBO offers a no-harm guarantee, allowing users to make mistakes; even with extreme adversarial interventions, the algorithm converges asymptotically to
    
[^26]: 扩展深度自适应输入规范化用于神经网络对时序数据的预处理

    Extended Deep Adaptive Input Normalization for Preprocessing Time Series Data for Neural Networks. (arXiv:2310.14720v1 [cs.LG])

    [http://arxiv.org/abs/2310.14720](http://arxiv.org/abs/2310.14720)

    本研究提出了一种名为EDAIn的扩展深度自适应输入规范化层，通过以端到端的方式学习如何适当地规范化时序数据，而不是使用固定的规范化方案，来提高深度神经网络在时序预测和分类任务中的性能。实验证明该方法在不同数据集上都取得了良好效果。

    

    数据预处理是任何机器学习流程中至关重要的一部分，它对性能和训练效率都有重要影响。当使用深度神经网络进行时序预测和分类时，这一点尤为明显：真实世界的时序数据通常表现出多样性、偏斜和异常值等不规则特征，如果不充分处理这些特征，模型性能很快会下降。在本研究中，我们提出了EDAIN（扩展深度自适应输入规范化）层，一种新颖的自适应神经层，它能够以端到端的方式学习如何适当地规范化不规则的时序数据，而不是使用固定的规范化方案。这通过使用反向传播算法，同时优化其未知参数和深度神经网络来实现。我们的实验证明，这一方法在使用合成数据、信用违约预测数据集和大规模限价单簿基准数据集时都取得了良好效果。

    Data preprocessing is a crucial part of any machine learning pipeline, and it can have a significant impact on both performance and training efficiency. This is especially evident when using deep neural networks for time series prediction and classification: real-world time series data often exhibit irregularities such as multi-modality, skewness and outliers, and the model performance can degrade rapidly if these characteristics are not adequately addressed. In this work, we propose the EDAIN (Extended Deep Adaptive Input Normalization) layer, a novel adaptive neural layer that learns how to appropriately normalize irregular time series data for a given task in an end-to-end fashion, instead of using a fixed normalization scheme. This is achieved by optimizing its unknown parameters simultaneously with the deep neural network using back-propagation. Our experiments, conducted using synthetic data, a credit default prediction dataset, and a large-scale limit order book benchmark datase
    
[^27]: 一种基于机器学习的概率暴露模型的德国高分辨率室内氡气地图

    A new high-resolution indoor radon map for Germany using a machine learning based probabilistic exposure model. (arXiv:2310.11143v1 [stat.ML])

    [http://arxiv.org/abs/2310.11143](http://arxiv.org/abs/2310.11143)

    本研究提出了一种基于机器学习的概率暴露模型，可以更准确地估计德国室内氡气分布，并具有更高的空间分辨率。

    

    室内氡气是一种致癌的放射性气体，可以在室内积累。通常情况下，全国范围内的室内氡暴露是基于广泛的测量活动估计得来的。然而，样本的特征往往与人口特征不同，这是由于许多相关因素，如地质源氡气的可用性或楼层水平。此外，样本大小通常不允许以高空间分辨率进行暴露估计。我们提出了一种基于模型的方法，可以比纯数据方法更加现实地估计室内氡分布，并具有更高的空间分辨率。我们采用了两阶段建模方法：1）应用分位数回归森林，使用环境和建筑数据作为预测因子，估计了德国每个住宅楼的每个楼层的室内氡概率分布函数；2）使用概率蒙特卡罗抽样技术使它们组合和。

    Radon is a carcinogenic, radioactive gas that can accumulate indoors. Indoor radon exposure at the national scale is usually estimated on the basis of extensive measurement campaigns. However, characteristics of the sample often differ from the characteristics of the population due to the large number of relevant factors such as the availability of geogenic radon or floor level. Furthermore, the sample size usually does not allow exposure estimation with high spatial resolution. We propose a model-based approach that allows a more realistic estimation of indoor radon distribution with a higher spatial resolution than a purely data-based approach. We applied a two-stage modelling approach: 1) a quantile regression forest using environmental and building data as predictors was applied to estimate the probability distribution function of indoor radon for each floor level of each residential building in Germany; (2) a probabilistic Monte Carlo sampling technique enabled the combination and
    
[^28]: 受限制和带水印生成的镜像扩散模型

    Mirror Diffusion Models for Constrained and Watermarked Generation. (arXiv:2310.01236v1 [stat.ML])

    [http://arxiv.org/abs/2310.01236](http://arxiv.org/abs/2310.01236)

    提出了一种新的镜像扩散模型（MDM），可以在受限制集合上生成数据而不丧失可追溯性。这通过在一个标准的欧几里得空间中学习扩散过程，并利用镜像映射来实现。

    

    现代扩散模型在学习复杂的高维数据分布方面取得了成功，这部分归功于其能够构建具有解析转移核函数和评分函数的扩散过程。这种可追溯性结果在不需要模拟的框架中具有稳定的回归损失，从而可以学习到可以扩展的逆向生成过程。然而，当数据被限制在受限制集合而不是标准的欧几里得空间中时，根据之前的尝试，这些理想的特性似乎丧失了。在这项工作中，我们提出了镜像扩散模型（MDM），一种新的扩散模型类，可以在凸约束集合上生成数据而不丧失任何可追溯性。这是通过在从镜像映射构建的对偶空间中学习扩散过程来实现的，关键的是，这是一个标准的欧几里得空间。我们推导了流行的约束集合（如单纯形和$\ell_2$-球）的镜像映射的有效计算，显示明显的提升。

    Modern successes of diffusion models in learning complex, high-dimensional data distributions are attributed, in part, to their capability to construct diffusion processes with analytic transition kernels and score functions. The tractability results in a simulation-free framework with stable regression losses, from which reversed, generative processes can be learned at scale. However, when data is confined to a constrained set as opposed to a standard Euclidean space, these desirable characteristics appear to be lost based on prior attempts. In this work, we propose Mirror Diffusion Models (MDM), a new class of diffusion models that generate data on convex constrained sets without losing any tractability. This is achieved by learning diffusion processes in a dual space constructed from a mirror map, which, crucially, is a standard Euclidean space. We derive efficient computation of mirror maps for popular constrained sets, such as simplices and $\ell_2$-balls, showing significantly im
    
[^29]: 基于交叉预测的推理

    Cross-Prediction-Powered Inference. (arXiv:2309.16598v1 [stat.ML])

    [http://arxiv.org/abs/2309.16598](http://arxiv.org/abs/2309.16598)

    本文介绍了一种基于机器学习的交叉预测方法，可以有效地进行推理。该方法通过使用一个小型标记数据集和一个大型未标记数据集，通过机器学习填补缺失的标签，并采用去偏差方法纠正预测的不准确性。

    

    可靠的数据驱动决策依赖于高质量的标注数据，然而获取高质量的标注数据经常需要繁琐的人工标注或者缓慢昂贵的科学测量。机器学习作为一种替代方案正变得越来越有吸引力，因为精密的预测技术可以快速、廉价地产生大量预测标签；例如，预测的蛋白质结构被用来补充实验得到的结构，卫星图像预测的社会经济指标被用来补充准确的调查数据等。由于预测具有不完美和潜在偏差的特点，这种做法对下游推理的有效性产生了质疑。我们引入了基于机器学习的交叉预测方法，用于有效的推理。通过一个小的标记数据集和一个大的未标记数据集，交叉预测通过机器学习填补缺失的标签，并应用一种去偏差的方法来纠正预测不准确性。

    While reliable data-driven decision-making hinges on high-quality labeled data, the acquisition of quality labels often involves laborious human annotations or slow and expensive scientific measurements. Machine learning is becoming an appealing alternative as sophisticated predictive techniques are being used to quickly and cheaply produce large amounts of predicted labels; e.g., predicted protein structures are used to supplement experimentally derived structures, predictions of socioeconomic indicators from satellite imagery are used to supplement accurate survey data, and so on. Since predictions are imperfect and potentially biased, this practice brings into question the validity of downstream inferences. We introduce cross-prediction: a method for valid inference powered by machine learning. With a small labeled dataset and a large unlabeled dataset, cross-prediction imputes the missing labels via machine learning and applies a form of debiasing to remedy the prediction inaccurac
    
[^30]: 用于噪声混合物中目标信号恢复的统计分量分离

    Statistical Component Separation for Targeted Signal Recovery in Noisy Mixtures. (arXiv:2306.15012v1 [stat.ML])

    [http://arxiv.org/abs/2306.15012](http://arxiv.org/abs/2306.15012)

    本论文提出了一种用于从噪声混合物中恢复目标信号的统计分量分离方法，并且在图像降噪任务中展示了其优于标准降噪方法的表现。

    

    当只对给定信号的特定属性感兴趣时，从一个加性混合物中分离信号可能是一个不必要地困难的问题。在本工作中，我们解决了更简单的“统计分量分离”问题，该问题专注于从噪声混合物中恢复目标信号的预定义统计描述量。假设可以获得噪声过程的样本，我们研究了一种方法，该方法旨在使受噪声样本污染的解决方案候选的统计特性与观测的混合物的统计特性匹配。首先，我们使用具有解析可追踪计算的简单示例分析了该方法的行为。然后，我们将其应用于图像降噪环境中，使用了1）基于小波的描述符，2）针对天体物理和ImageNet数据的ConvNet-based描述符。在第一种情况下，我们展示了我们的方法在大多数情况下比标准降噪方法更好地恢复了目标数据的描述符。此外，尽管不是为此目的构建的，它也表现出对目标信号描述符恢复的潜力。

    Separating signals from an additive mixture may be an unnecessarily hard problem when one is only interested in specific properties of a given signal. In this work, we tackle simpler "statistical component separation" problems that focus on recovering a predefined set of statistical descriptors of a target signal from a noisy mixture. Assuming access to samples of the noise process, we investigate a method devised to match the statistics of the solution candidate corrupted by noise samples with those of the observed mixture. We first analyze the behavior of this method using simple examples with analytically tractable calculations. Then, we apply it in an image denoising context employing 1) wavelet-based descriptors, 2) ConvNet-based descriptors on astrophysics and ImageNet data. In the case of 1), we show that our method better recovers the descriptors of the target data than a standard denoising method in most situations. Additionally, despite not constructed for this purpose, it pe
    
[^31]: 空间过程的神经似然面

    Neural Likelihood Surfaces for Spatial Processes with Computationally Intensive or Intractable Likelihoods. (arXiv:2305.04634v1 [stat.ME])

    [http://arxiv.org/abs/2305.04634](http://arxiv.org/abs/2305.04634)

    研究提出了使用CNN学习空间过程的似然函数。即使在没有确切似然函数的情况下，通过分类任务进行的神经网络的训练，可以隐式地学习似然函数。使用Platt缩放可以提高神经似然面的准确性。

    

    在空间统计中，当拟合空间过程到真实世界的数据时，快速准确的参数估计和可靠的不确定性量化手段可能是一项具有挑战性的任务，因为似然函数可能评估缓慢或难以处理。 在本研究中，我们提出使用卷积神经网络（CNN）学习空间过程的似然函数。通过特定设计的分类任务，我们的神经网络隐式地学习似然函数，即使在没有显式可用的确切似然函数的情况下也可以实现。一旦在分类任务上进行了训练，我们的神经网络使用Platt缩放进行校准，从而提高了神经似然面的准确性。为了展示我们的方法，我们比较了来自神经似然面的最大似然估计和近似置信区间与两个不同空间过程（高斯过程和对数高斯Cox过程）的相应精确或近似的似然函数构成的等效物。

    In spatial statistics, fast and accurate parameter estimation coupled with a reliable means of uncertainty quantification can be a challenging task when fitting a spatial process to real-world data because the likelihood function might be slow to evaluate or intractable. In this work, we propose using convolutional neural networks (CNNs) to learn the likelihood function of a spatial process. Through a specifically designed classification task, our neural network implicitly learns the likelihood function, even in situations where the exact likelihood is not explicitly available. Once trained on the classification task, our neural network is calibrated using Platt scaling which improves the accuracy of the neural likelihood surfaces. To demonstrate our approach, we compare maximum likelihood estimates and approximate confidence regions constructed from the neural likelihood surface with the equivalent for exact or approximate likelihood for two different spatial processes: a Gaussian Pro
    
[^32]: Langevin型Monte Carlo算法的非渐进分析

    Non-asymptotic analysis of Langevin-type Monte Carlo algorithms. (arXiv:2303.12407v1 [math.ST])

    [http://arxiv.org/abs/2303.12407](http://arxiv.org/abs/2303.12407)

    本文提出了一种新的Langevin型算法并应用于吉布斯分布。通过提出的2-Wasserstein距离上限，我们发现势函数的耗散性以及梯度 $\alpha>1/3$ 下的 $\alpha$-H\"{o}lder连续性可以保证算法具有接近零的误差。新的Langevin型算法还可以应用于无凸性或连续可微性的势函数。

    

    本文研究了Langevin型算法应用于吉布斯分布的情况，其中势函数是耗散的，且其弱梯度具有有限的连续性模量。我们的主要结果是2-Wasserstein距离上限的非渐进性，它衡量了吉布斯分布与基于Liptser-Shiryaev理论和函数不等式的Langevin型算法的一般分布之间的距离。我们应用这个上限来展示势函数的耗散性以及梯度 $\alpha>1/3$ 下的 $\alpha$-H\"{o}lder连续性是充分的，可以通过适当控制参数来获得Langevin Monte Carlo算法的收敛性。我们还针对无凸性或连续可微性的势函数提出了球形平滑技术的Langevin型算法。

    We study the Langevin-type algorithms for Gibbs distributions such that the potentials are dissipative and their weak gradients have the finite moduli of continuity. Our main result is a non-asymptotic upper bound of the 2-Wasserstein distance between the Gibbs distribution and the law of general Langevin-type algorithms based on the Liptser--Shiryaev theory and functional inequalities. We apply this bound to show that the dissipativity of the potential and the $\alpha$-H\"{o}lder continuity of the gradient with $\alpha>1/3$ are sufficient for the convergence of the Langevin Monte Carlo algorithm with appropriate control of the parameters. We also propose Langevin-type algorithms with spherical smoothing for potentials without convexity or continuous differentiability.
    
[^33]: 多视角数据中缺失值的插补问题解决方法

    Imputation of missing values in multi-view data. (arXiv:2210.14484v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2210.14484](http://arxiv.org/abs/2210.14484)

    本文提出了一种基于StaPLR算法的新的多视角数据插补算法，通过在降维空间中执行插补以解决计算挑战，并在模拟数据集中得到了竞争性结果。

    

    多视角数据是指由多个不同特征集描述的数据。在处理多视角数据时，若出现缺失值，则一个视角中的所有特征极有可能同时缺失，因而导致非常大量的缺失数据问题。本文提出了一种新的多视角学习算法中的插补方法，它基于堆叠惩罚逻辑回归(StaPLR)算法，在降维空间中执行插补，以解决固有的多视角计算挑战。实验结果表明，该方法在模拟数据集上具有竞争性结果，而且具有更低的计算成本，从而可以使用先进的插补算法，例如missForest。

    Data for which a set of objects is described by multiple distinct feature sets (called views) is known as multi-view data. When missing values occur in multi-view data, all features in a view are likely to be missing simultaneously. This leads to very large quantities of missing data which, especially when combined with high-dimensionality, makes the application of conditional imputation methods computationally infeasible. We introduce a new imputation method based on the existing stacked penalized logistic regression (StaPLR) algorithm for multi-view learning. It performs imputation in a dimension-reduced space to address computational challenges inherent to the multi-view context. We compare the performance of the new imputation method with several existing imputation algorithms in simulated data sets. The results show that the new imputation method leads to competitive results at a much lower computational cost, and makes the use of advanced imputation algorithms such as missForest 
    

