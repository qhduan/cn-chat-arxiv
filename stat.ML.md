# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [AdaTrans: Feature-wise and Sample-wise Adaptive Transfer Learning for High-dimensional Regression](https://arxiv.org/abs/2403.13565) | 提出了一种针对高维回归的自适应迁移学习方法，可以根据可迁移结构自适应检测和聚合特征和样本的可迁移结构。 |
| [^2] | [Selecting informative conformal prediction sets with false coverage rate control](https://arxiv.org/abs/2403.12295) | 提出了一种新的统一框架，用于构建信息丰富的符合预测集，同时控制所选样本的虚警覆盖率。 |
| [^3] | [Robust Learning of Noisy Time Series Collections Using Stochastic Process Models with Motion Codes](https://arxiv.org/abs/2402.14081) | 使用具有学习谱核的混合高斯过程的潜变量模型方法，针对嘈杂时间序列数据进行鲁棒学习。 |
| [^4] | [Mixed-Output Gaussian Process Latent Variable Models](https://arxiv.org/abs/2402.09122) | 本文提出了一种基于高斯过程潜变量模型的贝叶斯非参数方法，可以用于信号分离，并且能够处理包含纯组分信号加权和的情况，适用于光谱学和其他领域的多种应用。 |
| [^5] | [Rethinking Scaling Laws for Learning in Strategic Environments](https://arxiv.org/abs/2402.07588) | 本文重新思考了在战略环境中学习的比例定律，发现战略互动可以打破传统的观点，即模型越大或表达能力越强并不一定会随之提高性能。通过几个战略环境的例子，我们展示了这种现象的影响。 |
| [^6] | [Practical and Asymptotically Exact Conditional Sampling in Diffusion Models.](http://arxiv.org/abs/2306.17775) | 本论文提出了一种名为TDS的扭转式扩散采样器，它是一种针对扩散模型的顺序蒙特卡洛算法。该方法通过使用扭转技术结合启发式近似，能够在不需要特定训练的情况下在广泛的条件分布上提供精确的样本。 |
| [^7] | [Auditing for Human Expertise.](http://arxiv.org/abs/2306.01646) | 人类专家的价值超出了算法可捕捉范围，我们可以用一个简单的程序测试这个问题。 |
| [^8] | [Constructing Semantics-Aware Adversarial Examples with Probabilistic Perspective.](http://arxiv.org/abs/2306.00353) | 本研究提出了一个基于概率视角的对抗样本构建方法，可以生成语义感知的对抗性样本，并可以有效规避传统对抗性攻击的强化对抗训练方法。 |
| [^9] | [Sample Complexity of Probability Divergences under Group Symmetry.](http://arxiv.org/abs/2302.01915) | 本文研究了具有群不变性的分布变量在变分差异估计中的样本复杂度，发现在群大小维度相关的情况下，样本复杂度会有所降低，并在实验中得到了验证。 |

# 详细

[^1]: AdaTrans：针对高维回归的特征自适应与样本自适应迁移学习

    AdaTrans: Feature-wise and Sample-wise Adaptive Transfer Learning for High-dimensional Regression

    [https://arxiv.org/abs/2403.13565](https://arxiv.org/abs/2403.13565)

    提出了一种针对高维回归的自适应迁移学习方法，可以根据可迁移结构自适应检测和聚合特征和样本的可迁移结构。

    

    我们考虑高维背景下的迁移学习问题，在该问题中，特征维度大于样本大小。为了学习可迁移的信息，该信息可能在特征或源样本之间变化，我们提出一种自适应迁移学习方法，可以检测和聚合特征-wise (F-AdaTrans)或样本-wise (S-AdaTrans)可迁移结构。我们通过采用一种新颖的融合惩罚方法，结合权重，可以根据可迁移结构进行调整。为了选择权重，我们提出了一个在理论上建立，数据驱动的过程，使得 F-AdaTrans 能够选择性地将可迁移的信号与目标融合在一起，同时滤除非可迁移的信号，S-AdaTrans则可以获得每个源样本传递的信息的最佳组合。我们建立了非渐近速率，可以在特殊情况下恢复现有的近最小似乎最优速率。效果证明...

    arXiv:2403.13565v1 Announce Type: cross  Abstract: We consider the transfer learning problem in the high dimensional setting, where the feature dimension is larger than the sample size. To learn transferable information, which may vary across features or the source samples, we propose an adaptive transfer learning method that can detect and aggregate the feature-wise (F-AdaTrans) or sample-wise (S-AdaTrans) transferable structures. We achieve this by employing a novel fused-penalty, coupled with weights that can adapt according to the transferable structure. To choose the weight, we propose a theoretically informed, data-driven procedure, enabling F-AdaTrans to selectively fuse the transferable signals with the target while filtering out non-transferable signals, and S-AdaTrans to obtain the optimal combination of information transferred from each source sample. The non-asymptotic rates are established, which recover existing near-minimax optimal rates in special cases. The effectivene
    
[^2]: 通过控制虚警覆盖率选择信息量丰富的符合预测集

    Selecting informative conformal prediction sets with false coverage rate control

    [https://arxiv.org/abs/2403.12295](https://arxiv.org/abs/2403.12295)

    提出了一种新的统一框架，用于构建信息丰富的符合预测集，同时控制所选样本的虚警覆盖率。

    

    在监督学习中，包括回归和分类，符合方法为任何机器学习预测器提供预测结果/标签的预测集合，具有有限样本覆盖率。在这里我们考虑了这样一种情况，即这种预测集合是经过选择过程得到的。该选择过程要求选择的预测集在某种明确定义的意义上是“信息量丰富的”。我们考虑了分类和回归设置，在这些设置中，分析人员可能只考虑具有预测标签集或预测区间足够小、不包括空值或遵守其他适当的“单调”约束的样本为具有信息量丰富的。虽然这涵盖了各种应用中可能感兴趣的许多设置，我们开发了一个统一的框架，用来构建这样的信息量丰富的符合预测集，同时控制所选样本上的虚警覆盖率（FCR）。

    arXiv:2403.12295v1 Announce Type: cross  Abstract: In supervised learning, including regression and classification, conformal methods provide prediction sets for the outcome/label with finite sample coverage for any machine learning predictors. We consider here the case where such prediction sets come after a selection process. The selection process requires that the selected prediction sets be `informative' in a well defined sense. We consider both the classification and regression settings where the analyst may consider as informative only the sample with prediction label sets or prediction intervals small enough, excluding null values, or obeying other appropriate `monotone' constraints. While this covers many settings of possible interest in various applications, we develop a unified framework for building such informative conformal prediction sets while controlling the false coverage rate (FCR) on the selected sample. While conformal prediction sets after selection have been the f
    
[^3]: 使用具有运动代码的随机过程模型对嘈杂时间序列集合进行鲁棒学习

    Robust Learning of Noisy Time Series Collections Using Stochastic Process Models with Motion Codes

    [https://arxiv.org/abs/2402.14081](https://arxiv.org/abs/2402.14081)

    使用具有学习谱核的混合高斯过程的潜变量模型方法，针对嘈杂时间序列数据进行鲁棒学习。

    

    虽然时间序列分类和预测问题已经得到广泛研究，但具有任意时间序列长度的嘈杂时间序列数据的情况仍具挑战性。每个时间序列实例可以看作是嘈杂动态模型的一个样本实现，其特点是连续随机过程。对于许多应用，数据是混合的，由多个随机过程建模的几种类型的嘈杂时间序列序列组成，使得预测和分类任务变得更具挑战性。我们不是简单地将数据回归到每种时间序列类型，而是采用具有学习谱核的混合高斯过程的潜变量模型方法。更具体地说，我们为每种类型的嘈杂时间序列数据自动分配一个称为其运动代码的签名向量。然后，在每个分配的运动代码的条件下，我们推断出相关性的稀疏近似。

    arXiv:2402.14081v1 Announce Type: cross  Abstract: While time series classification and forecasting problems have been extensively studied, the cases of noisy time series data with arbitrary time sequence lengths have remained challenging. Each time series instance can be thought of as a sample realization of a noisy dynamical model, which is characterized by a continuous stochastic process. For many applications, the data are mixed and consist of several types of noisy time series sequences modeled by multiple stochastic processes, making the forecasting and classification tasks even more challenging. Instead of regressing data naively and individually to each time series type, we take a latent variable model approach using a mixtured Gaussian processes with learned spectral kernels. More specifically, we auto-assign each type of noisy time series data a signature vector called its motion code. Then, conditioned on each assigned motion code, we infer a sparse approximation of the corr
    
[^4]: 混合输出高斯过程潜变量模型

    Mixed-Output Gaussian Process Latent Variable Models

    [https://arxiv.org/abs/2402.09122](https://arxiv.org/abs/2402.09122)

    本文提出了一种基于高斯过程潜变量模型的贝叶斯非参数方法，可以用于信号分离，并且能够处理包含纯组分信号加权和的情况，适用于光谱学和其他领域的多种应用。

    

    本文提出了一种贝叶斯非参数的信号分离方法，其中信号可以根据潜变量变化。我们的主要贡献是增加了高斯过程潜变量模型（GPLVMs），以包括每个数据点由已知数量的纯组分信号的加权和组成的情况，并观察多个输入位置。我们的框架允许使用各种关于每个观测权重的先验。这种灵活性使我们能够表示包括用于估计分数组成的总和为一约束和用于分类的二进制权重的用例。我们的贡献对于光谱学尤其相关，因为改变条件可能导致基础纯组分信号在样本之间变化。为了展示对光谱学和其他领域的适用性，我们考虑了几个应用：一个具有不同温度的近红外光谱数据集。

    arXiv:2402.09122v1 Announce Type: cross Abstract: This work develops a Bayesian non-parametric approach to signal separation where the signals may vary according to latent variables. Our key contribution is to augment Gaussian Process Latent Variable Models (GPLVMs) to incorporate the case where each data point comprises the weighted sum of a known number of pure component signals, observed across several input locations. Our framework allows the use of a range of priors for the weights of each observation. This flexibility enables us to represent use cases including sum-to-one constraints for estimating fractional makeup, and binary weights for classification. Our contributions are particularly relevant to spectroscopy, where changing conditions may cause the underlying pure component signals to vary from sample to sample. To demonstrate the applicability to both spectroscopy and other domains, we consider several applications: a near-infrared spectroscopy data set with varying temper
    
[^5]: 重新思考战略环境中学习的比例定律

    Rethinking Scaling Laws for Learning in Strategic Environments

    [https://arxiv.org/abs/2402.07588](https://arxiv.org/abs/2402.07588)

    本文重新思考了在战略环境中学习的比例定律，发现战略互动可以打破传统的观点，即模型越大或表达能力越强并不一定会随之提高性能。通过几个战略环境的例子，我们展示了这种现象的影响。

    

    越来越大的机器学习模型的部署反映出一个共识：模型越有表达能力，越拥有大量数据，就能改善性能。随着模型在各种真实场景中的部署，它们不可避免地面临着战略环境。本文考虑了模型与战略互动对比例定律的相互作用对性能的影响这个自然问题。我们发现战略互动可以打破传统的比例定律观点，即性能并不一定随着模型的扩大和/或表达能力的增强（即使有无限数据）而单调提高。我们通过战略回归、战略分类和多智能体强化学习的例子展示了这一现象的影响，这些例子展示了战略环境中的限制模型或策略类的表达能力即可。

    The deployment of ever-larger machine learning models reflects a growing consensus that the more expressive the model$\unicode{x2013}$and the more data one has access to$\unicode{x2013}$the more one can improve performance. As models get deployed in a variety of real world scenarios, they inevitably face strategic environments. In this work, we consider the natural question of how the interplay of models and strategic interactions affects scaling laws. We find that strategic interactions can break the conventional view of scaling laws$\unicode{x2013}$meaning that performance does not necessarily monotonically improve as models get larger and/ or more expressive (even with infinite data). We show the implications of this phenomenon in several contexts including strategic regression, strategic classification, and multi-agent reinforcement learning through examples of strategic environments in which$\unicode{x2013}$by simply restricting the expressivity of one's model or policy class$\uni
    
[^6]: 扩散模型中的实用和渐进精确条件采样

    Practical and Asymptotically Exact Conditional Sampling in Diffusion Models. (arXiv:2306.17775v1 [stat.ML])

    [http://arxiv.org/abs/2306.17775](http://arxiv.org/abs/2306.17775)

    本论文提出了一种名为TDS的扭转式扩散采样器，它是一种针对扩散模型的顺序蒙特卡洛算法。该方法通过使用扭转技术结合启发式近似，能够在不需要特定训练的情况下在广泛的条件分布上提供精确的样本。

    

    扩散模型在分子设计和文本到图像生成等条件生成任务中取得了成功。然而，这些成就主要依赖于任务特定的条件训练或容易出错的启发式近似。理想情况下，条件生成方法应该能够在不需要特定训练的情况下为广泛的条件分布提供精确的样本。为此，我们引入了扭转式扩散采样器(TDS)。TDS是一种针对扩散模型的顺序蒙特卡洛(SMC)算法。其主要思想是使用扭转，一种具有良好计算效率的SMC技术，来结合启发式近似而不影响渐进精确性。我们首先在模拟实验和MNIST图像修复以及类条件生成任务中发现，TDS提供了计算统计权衡，使用更多粒子得到更准确的近似结果，但同时需要更多计算资源。

    Diffusion models have been successful on a range of conditional generation tasks including molecular design and text-to-image generation. However, these achievements have primarily depended on task-specific conditional training or error-prone heuristic approximations. Ideally, a conditional generation method should provide exact samples for a broad range of conditional distributions without requiring task-specific training. To this end, we introduce the Twisted Diffusion Sampler, or TDS. TDS is a sequential Monte Carlo (SMC) algorithm that targets the conditional distributions of diffusion models. The main idea is to use twisting, an SMC technique that enjoys good computational efficiency, to incorporate heuristic approximations without compromising asymptotic exactness. We first find in simulation and on MNIST image inpainting and class-conditional generation tasks that TDS provides a computational statistical trade-off, yielding more accurate approximations with many particles but wi
    
[^7]: 人类专家审核研究

    Auditing for Human Expertise. (arXiv:2306.01646v1 [stat.ML])

    [http://arxiv.org/abs/2306.01646](http://arxiv.org/abs/2306.01646)

    人类专家的价值超出了算法可捕捉范围，我们可以用一个简单的程序测试这个问题。

    

    高风险预测任务（例如患者诊断）通常由接受培训的人类专家处理。在这些设置中，自动化的一个常见问题是，专家可能运用很难建模的直觉，并且/或者可以获取信息（例如与患者的交谈），这些信息对于算法来说是不可用的。这引发了一个自然的问题，人类专家是否增加了无法被算法预测器捕捉到的价值。我们开发了一个统计框架，可以将这个问题提出为一个自然的假设检验。正如我们的框架所强调的那样，检测人类专业知识比简单比较专家预测准确性与特定学习算法做出的准确性更加微妙。而是提出了一个简单的程序，测试专家预测是否在“特征”可用而条件下是否与感兴趣的结果统计上独立。因此，我们测试的拒绝表明了人类专业知识确实增加了超出算法可捕捉范围的价值。

    High-stakes prediction tasks (e.g., patient diagnosis) are often handled by trained human experts. A common source of concern about automation in these settings is that experts may exercise intuition that is difficult to model and/or have access to information (e.g., conversations with a patient) that is simply unavailable to a would-be algorithm. This raises a natural question whether human experts add value which could not be captured by an algorithmic predictor. We develop a statistical framework under which we can pose this question as a natural hypothesis test. Indeed, as our framework highlights, detecting human expertise is more subtle than simply comparing the accuracy of expert predictions to those made by a particular learning algorithm. Instead, we propose a simple procedure which tests whether expert predictions are statistically independent from the outcomes of interest after conditioning on the available inputs (`features'). A rejection of our test thus suggests that huma
    
[^8]: 从概率角度构建语义感知的对抗样本

    Constructing Semantics-Aware Adversarial Examples with Probabilistic Perspective. (arXiv:2306.00353v1 [stat.ML])

    [http://arxiv.org/abs/2306.00353](http://arxiv.org/abs/2306.00353)

    本研究提出了一个基于概率视角的对抗样本构建方法，可以生成语义感知的对抗性样本，并可以有效规避传统对抗性攻击的强化对抗训练方法。

    

    本研究提出了一种新颖的概率视角对抗样本构建方法——箱约束 Langevin Monte Carlo（LMC）。从这个角度出发，我们开发了一种创新性的方法，以原则性的方式生成语义感知的对抗性样本。这种方法超越了几何距离所施加的限制，选择了语义约束。我们的方法赋予了个体将其对语义的理解融入到模型中的能力。通过人类评估，我们验证了我们的语义感知的对抗样本保持其固有的含义。在 MNIST 和 SVHN 数据集上的实验结果表明，我们的语义感知的对抗样本可以有效地规避针对传统对抗性攻击的强健性对抗训练方法。

    In this study, we introduce a novel, probabilistic viewpoint on adversarial examples, achieved through box-constrained Langevin Monte Carlo (LMC). Proceeding from this perspective, we develop an innovative approach for generating semantics-aware adversarial examples in a principled manner. This methodology transcends the restriction imposed by geometric distance, instead opting for semantic constraints. Our approach empowers individuals to incorporate their personal comprehension of semantics into the model. Through human evaluation, we validate that our semantics-aware adversarial examples maintain their inherent meaning. Experimental findings on the MNIST and SVHN datasets demonstrate that our semantics-aware adversarial examples can effectively circumvent robust adversarial training methods tailored for traditional adversarial attacks.
    
[^9]: 基于群对称性的概率差异的样本复杂度分析

    Sample Complexity of Probability Divergences under Group Symmetry. (arXiv:2302.01915v2 [math.ST] UPDATED)

    [http://arxiv.org/abs/2302.01915](http://arxiv.org/abs/2302.01915)

    本文研究了具有群不变性的分布变量在变分差异估计中的样本复杂度，发现在群大小维度相关的情况下，样本复杂度会有所降低，并在实验中得到了验证。

    

    我们对于具有群不变性的分布变量在变分差异估计中的样本复杂度进行了严谨的量化分析。在Wasserstein-1距离和Lipschitz正则化α差异的情况下，样本复杂度的降低与群大小的维度相关。对于最大均值差异（MMD），样本复杂度的改进更加复杂，因为它不仅取决于群大小，还取决于内核的选择。 数值模拟验证了我们的理论。

    We rigorously quantify the improvement in the sample complexity of variational divergence estimations for group-invariant distributions. In the cases of the Wasserstein-1 metric and the Lipschitz-regularized $\alpha$-divergences, the reduction of sample complexity is proportional to an ambient-dimension-dependent power of the group size. For the maximum mean discrepancy (MMD), the improvement of sample complexity is more nuanced, as it depends on not only the group size but also the choice of kernel. Numerical simulations verify our theories.
    

