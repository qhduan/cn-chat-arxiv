# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Thermometer: Towards Universal Calibration for Large Language Models](https://arxiv.org/abs/2403.08819) | 提出了一种针对大型语言模型的校准方法THERMOMETER，通过学习来自多个任务数据的辅助模型，实现了计算效率高、准确性保持并产生更好校准响应的目标。 |
| [^2] | [Improving Variational Autoencoder Estimation from Incomplete Data with Mixture Variational Families](https://arxiv.org/abs/2403.03069) | 缺失数据增加了模型对潜在变量后验分布的复杂性，本文提出了两种策略——基于有限变分混合和基于填补的变分混合分布，有效改善了从不完整数据估计VAE的准确性。 |
| [^3] | [The Price of Adaptivity in Stochastic Convex Optimization](https://arxiv.org/abs/2402.10898) | 该论文证明了在非光滑随机凸优化中，适应性的代价是无法避免的，并且给出了关于不确定性参数的次优性乘法增加的下界。 |
| [^4] | [Bayesian identification of nonseparable Hamiltonians with multiplicative noise using deep learning and reduced-order modeling.](http://arxiv.org/abs/2401.12476) | 本文提出了一种用于学习非分离哈密顿系统的结构保持的贝叶斯方法，可以处理统计相关的加性和乘性噪声，并且通过将结构保持方法纳入框架中，提供了对高维系统的高效识别。 |
| [^5] | [MixerFlow for Image Modelling.](http://arxiv.org/abs/2310.16777) | MixerFlow是一种新型的基于MLP-Mixer架构的正则化流模型，通过提供有效的权重共享机制，实现了更好的图像密度估计性能和更丰富的嵌入表示。 |
| [^6] | [Hierarchical Concept Discovery Models: A Concept Pyramid Scheme.](http://arxiv.org/abs/2310.02116) | 本论文提出了一种分层概念发现模型，通过利用图像文本模型和基于数据驱动的贝叶斯参数，实现了基于人类可理解概念的高度可解释的决策过程。 |
| [^7] | [Kernelised Normalising Flows.](http://arxiv.org/abs/2307.14839) | 本文提出了一种新颖的核化归一化流范式，称为Ferumal流，它将核函数集成到归一化流的框架中。相对于基于神经网络的流，核化流可以在低数据环境中产生竞争力或优越的结果，同时保持参数效率。 |
| [^8] | [A Penalized Poisson Likelihood Approach to High-Dimensional Semi-Parametric Inference for Doubly-Stochastic Point Processes.](http://arxiv.org/abs/2306.06756) | 本研究提出了一种对于双随机点过程的估计方法，该方法在进行协变量效应估计时非常高效，不需要强烈的限制性假设，且在理论和实践中均表现出了良好的信度保证和效能。 |
| [^9] | [Expressiveness Remarks for Denoising Diffusion Models and Samplers.](http://arxiv.org/abs/2305.09605) | 本文在漫扩扩散模型和采样器方面进行了表达能力的研究，通过将已知的神经网络逼近结果扩展到漫扩扩散模型和采样器来实现。 |

# 详细

[^1]: 温度计：面向大型语言模型的通用校准

    Thermometer: Towards Universal Calibration for Large Language Models

    [https://arxiv.org/abs/2403.08819](https://arxiv.org/abs/2403.08819)

    提出了一种针对大型语言模型的校准方法THERMOMETER，通过学习来自多个任务数据的辅助模型，实现了计算效率高、准确性保持并产生更好校准响应的目标。

    

    我们考虑大型语言模型（LLM）中的校准问题。最近的研究发现，常见的干预措施如指令调整通常会导致校准不佳的LLMs。尽管校准在传统应用中得到了很好的探讨，但对LLMs进行校准具有独特挑战。这些挑战不仅来自LLMs的严格计算要求，也来自它们的多功能性，使它们可以应用于各种任务。为了解决这些挑战，我们提出了一个针对LLMs的校准方法THERMOMETER。THERMOMETER通过学习来自多个任务的数据的辅助模型，用于校准LLM。它在计算上效率高，保持了LLM的准确性，并为新任务产生了更好的校准响应。对各种基准的广泛实证评估显示了所提方法的有效性。

    arXiv:2403.08819v1 Announce Type: cross  Abstract: We consider the issue of calibration in large language models (LLM). Recent studies have found that common interventions such as instruction tuning often result in poorly calibrated LLMs. Although calibration is well-explored in traditional applications, calibrating LLMs is uniquely challenging. These challenges stem as much from the severe computational requirements of LLMs as from their versatility, which allows them to be applied to diverse tasks. Addressing these challenges, we propose THERMOMETER, a calibration approach tailored to LLMs. THERMOMETER learns an auxiliary model, given data from multiple tasks, for calibrating a LLM. It is computationally efficient, preserves the accuracy of the LLM, and produces better-calibrated responses for new tasks. Extensive empirical evaluations across various benchmarks demonstrate the effectiveness of the proposed method.
    
[^2]: 用混合变分家族改进从不完整数据估计的变分自动编码器

    Improving Variational Autoencoder Estimation from Incomplete Data with Mixture Variational Families

    [https://arxiv.org/abs/2403.03069](https://arxiv.org/abs/2403.03069)

    缺失数据增加了模型对潜在变量后验分布的复杂性，本文提出了两种策略——基于有限变分混合和基于填补的变分混合分布，有效改善了从不完整数据估计VAE的准确性。

    

    我们考虑了在训练数据不完整的情况下估计变分自动编码器（VAEs）的任务。我们证明了缺失数据会增加模型对潜在变量的后验分布的复杂性，与完全观测的情况相比。增加的复杂性可能会由于变分分布和模型后验分布之间的不匹配而对模型拟合产生不利影响。我们引入了两种基于（i）有限变分混合和（ii）基于填补的变分混合分布的策略，以解决增加的后验复杂性。通过对所提出方法的全面评估，我们表明变分混合在改进从不完整数据估计VAE的准确性方面是有效的。

    arXiv:2403.03069v1 Announce Type: new  Abstract: We consider the task of estimating variational autoencoders (VAEs) when the training data is incomplete. We show that missing data increases the complexity of the model's posterior distribution over the latent variables compared to the fully-observed case. The increased complexity may adversely affect the fit of the model due to a mismatch between the variational and model posterior distributions. We introduce two strategies based on (i) finite variational-mixture and (ii) imputation-based variational-mixture distributions to address the increased posterior complexity. Through a comprehensive evaluation of the proposed approaches, we show that variational mixtures are effective at improving the accuracy of VAE estimation from incomplete data.
    
[^3]: 随机凸优化中适应性的代价

    The Price of Adaptivity in Stochastic Convex Optimization

    [https://arxiv.org/abs/2402.10898](https://arxiv.org/abs/2402.10898)

    该论文证明了在非光滑随机凸优化中，适应性的代价是无法避免的，并且给出了关于不确定性参数的次优性乘法增加的下界。

    

    我们证明了在非光滑随机凸优化中适应性的不可能性结果。给定一组我们希望适应的问题参数，我们定义了“适应性的代价”（PoA），粗略地说，它衡量了由于这些参数的不确定性而导致的次优性的乘法增加。当初始距离最优解未知但梯度范数有界时，我们证明PoA至少对于期望次优性是对数级别，对于中位数次优性是双对数级别。当距离和梯度范数都存在不确定性时，我们表明PoA必须是与不确定性水平多项式相关的。我们的下界几乎与现有的上界相匹配，并且确定了没有无参数午餐的结论。

    arXiv:2402.10898v1 Announce Type: cross  Abstract: We prove impossibility results for adaptivity in non-smooth stochastic convex optimization. Given a set of problem parameters we wish to adapt to, we define a "price of adaptivity" (PoA) that, roughly speaking, measures the multiplicative increase in suboptimality due to uncertainty in these parameters. When the initial distance to the optimum is unknown but a gradient norm bound is known, we show that the PoA is at least logarithmic for expected suboptimality, and double-logarithmic for median suboptimality. When there is uncertainty in both distance and gradient norm, we show that the PoA must be polynomial in the level of uncertainty. Our lower bounds nearly match existing upper bounds, and establish that there is no parameter-free lunch.
    
[^4]: 用深度学习和降阶建模进行贝叶斯非分离哈密顿系统的识别和多项式噪声 (arXiv:2401.12476v1 [stat.ML])

    Bayesian identification of nonseparable Hamiltonians with multiplicative noise using deep learning and reduced-order modeling. (arXiv:2401.12476v1 [stat.ML])

    [http://arxiv.org/abs/2401.12476](http://arxiv.org/abs/2401.12476)

    本文提出了一种用于学习非分离哈密顿系统的结构保持的贝叶斯方法，可以处理统计相关的加性和乘性噪声，并且通过将结构保持方法纳入框架中，提供了对高维系统的高效识别。

    

    本文提出了一种结构保持的贝叶斯方法，用于学习使用随机动力模型的非分离哈密顿系统，该系统允许统计相关的，矢量值的加性和乘性测量噪声。该方法由三个主要方面组成。首先，我们推导了一个用于评估贝叶斯后验中的似然函数所需的统计相关的，矢量值的加性和乘性噪声模型的高斯滤波器。其次，我们开发了一种新算法，用于对高维系统进行高效的贝叶斯系统识别。第三，我们演示了如何将结构保持方法纳入所提议的框架中，使用非分离哈密顿系统作为一个举例的系统类别。我们将贝叶斯方法与一种最先进的机器学习方法在一个典型的非分离哈密顿模型和带有小型噪声训练数据集的混沌双摆模型上进行了比较，实验结果表明

    This paper presents a structure-preserving Bayesian approach for learning nonseparable Hamiltonian systems using stochastic dynamic models allowing for statistically-dependent, vector-valued additive and multiplicative measurement noise. The approach is comprised of three main facets. First, we derive a Gaussian filter for a statistically-dependent, vector-valued, additive and multiplicative noise model that is needed to evaluate the likelihood within the Bayesian posterior. Second, we develop a novel algorithm for cost-effective application of Bayesian system identification to high-dimensional systems. Third, we demonstrate how structure-preserving methods can be incorporated into the proposed framework, using nonseparable Hamiltonians as an illustrative system class. We compare the Bayesian method to a state-of-the-art machine learning method on a canonical nonseparable Hamiltonian model and a chaotic double pendulum model with small, noisy training datasets. The results show that us
    
[^5]: 图像建模的MixerFlow

    MixerFlow for Image Modelling. (arXiv:2310.16777v1 [stat.ML])

    [http://arxiv.org/abs/2310.16777](http://arxiv.org/abs/2310.16777)

    MixerFlow是一种新型的基于MLP-Mixer架构的正则化流模型，通过提供有效的权重共享机制，实现了更好的图像密度估计性能和更丰富的嵌入表示。

    

    正则化流是一种统计模型，通过使用双射变换将复杂密度转换为简单密度，实现了密度估计和从单个模型生成数据的功能。在图像建模的背景下，主要选择的是基于Glow的架构，而其他架构在研究界尚未得到广泛探索。在本研究中，我们提出了一种基于MLP-Mixer架构的新型架构MixerFlow，进一步统一了生成性和判别性建模架构。MixerFlow提供了一种有效的权重共享机制，适用于基于流的模型。我们的结果表明，在固定计算预算下，MixerFlow在图像数据集上具有更好的密度估计性能，并且随着图像分辨率的增加，其性能也得到了良好的扩展，使得MixerFlow成为Glow-based架构的一个强大而简单的替代品。我们还展示了MixerFlow提供了比Glow-based架构更丰富的嵌入表示。

    Normalising flows are statistical models that transform a complex density into a simpler density through the use of bijective transformations enabling both density estimation and data generation from a single model. In the context of image modelling, the predominant choice has been the Glow-based architecture, whereas alternative architectures remain largely unexplored in the research community. In this work, we propose a novel architecture called MixerFlow, based on the MLP-Mixer architecture, further unifying the generative and discriminative modelling architectures. MixerFlow offers an effective mechanism for weight sharing for flow-based models. Our results demonstrate better density estimation on image datasets under a fixed computational budget and scales well as the image resolution increases, making MixeFlow a powerful yet simple alternative to the Glow-based architectures. We also show that MixerFlow provides more informative embeddings than Glow-based architectures.
    
[^6]: 分层概念发现模型：一个概念金字塔方案

    Hierarchical Concept Discovery Models: A Concept Pyramid Scheme. (arXiv:2310.02116v1 [cs.LG])

    [http://arxiv.org/abs/2310.02116](http://arxiv.org/abs/2310.02116)

    本论文提出了一种分层概念发现模型，通过利用图像文本模型和基于数据驱动的贝叶斯参数，实现了基于人类可理解概念的高度可解释的决策过程。

    

    最近，深度学习算法因其卓越的性能而引起了大量关注。然而，它们的高复杂性和不可解释的操作方式阻碍了它们在真实世界的安全关键任务中的自信部署。本研究针对的是ante hoc可解释性，具体 说是概念瓶颈模型（CBMs）。我们的目标是设计一个框架，以多个层次粒度上的人类可理解概念为基础，实现高度可解释的决策过程。为此，我们提出了一种新颖的分层概念发现方法，利用：（i）图像文本模型的最新进展，以及（ii）基于数据驱动和稀疏诱导的贝叶斯参数进行多层概念选择的创新公式。在这个框架中，概念信息不仅仅依赖于整体图像与一般非结构化概念之间的相似性；相反，我们引入了概念层次的概念，以揭示和利用更多的细节。

    Deep Learning algorithms have recently gained significant attention due to their impressive performance. However, their high complexity and un-interpretable mode of operation hinders their confident deployment in real-world safety-critical tasks. This work targets ante hoc interpretability, and specifically Concept Bottleneck Models (CBMs). Our goal is to design a framework that admits a highly interpretable decision making process with respect to human understandable concepts, on multiple levels of granularity. To this end, we propose a novel hierarchical concept discovery formulation leveraging: (i) recent advances in image-text models, and (ii) an innovative formulation for multi-level concept selection via data-driven and sparsity inducing Bayesian arguments. Within this framework, concept information does not solely rely on the similarity between the whole image and general unstructured concepts; instead, we introduce the notion of concept hierarchy to uncover and exploit more gra
    
[^7]: 核化归一化流

    Kernelised Normalising Flows. (arXiv:2307.14839v1 [stat.ML])

    [http://arxiv.org/abs/2307.14839](http://arxiv.org/abs/2307.14839)

    本文提出了一种新颖的核化归一化流范式，称为Ferumal流，它将核函数集成到归一化流的框架中。相对于基于神经网络的流，核化流可以在低数据环境中产生竞争力或优越的结果，同时保持参数效率。

    

    归一化流是以其可逆的架构而被描述的生成模型。然而，可逆性要求对其表达能力施加限制，需要大量的参数和创新的架构设计来达到满意的结果。虽然基于流的模型主要依赖于基于神经网络的转换来实现表达能力，但替代的转换方法却受到了有限的关注。在这项工作中，我们提出了一种新颖的核化归一化流范式，称为Ferumal流，它将核函数集成到框架中。我们的结果表明，相比于基于神经网络的流，核化流可以产生有竞争力或优越的结果，同时保持参数效率。核化流在低数据环境中表现出色，可以在数据稀缺的应用中进行灵活的非参数密度估计。

    Normalising Flows are generative models characterised by their invertible architecture. However, the requirement of invertibility imposes constraints on their expressiveness, necessitating a large number of parameters and innovative architectural designs to achieve satisfactory outcomes. Whilst flow-based models predominantly rely on neural-network-based transformations for expressive designs, alternative transformation methods have received limited attention. In this work, we present Ferumal flow, a novel kernelised normalising flow paradigm that integrates kernels into the framework. Our results demonstrate that a kernelised flow can yield competitive or superior results compared to neural network-based flows whilst maintaining parameter efficiency. Kernelised flows excel especially in the low-data regime, enabling flexible non-parametric density estimation in applications with sparse data availability.
    
[^8]: 一种对于双随机点过程的高维半参数推理的惩罚泊松似然方法。

    A Penalized Poisson Likelihood Approach to High-Dimensional Semi-Parametric Inference for Doubly-Stochastic Point Processes. (arXiv:2306.06756v1 [stat.ME])

    [http://arxiv.org/abs/2306.06756](http://arxiv.org/abs/2306.06756)

    本研究提出了一种对于双随机点过程的估计方法，该方法在进行协变量效应估计时非常高效，不需要强烈的限制性假设，且在理论和实践中均表现出了良好的信度保证和效能。

    

    双随机点过程将空间域内事件的发生建模为在实现随机强度函数的条件下，不均匀泊松过程。它们是捕捉空间异质性和依赖性的灵活工具。然而，双随机空间模型的实现在计算上是有要求的，往往具有有限的理论保证和/或依赖于具有限制性假设。我们提出了一种惩罚回归方法，用于估计双随机点过程中的协变量效应，具有计算效率且不需要基础强度的参数形式或平稳性。我们证实了所提出估计器的一致性和渐近正态性，并开发了一个协方差估计器，导致保守的统计推断程序。模拟研究显示了我们的方法在数据生成机制的限制性较小的情况下的有效性，并且在西雅图犯罪事件的应用中证明了我们的方法在实践中的良好性能。

    Doubly-stochastic point processes model the occurrence of events over a spatial domain as an inhomogeneous Poisson process conditioned on the realization of a random intensity function. They are flexible tools for capturing spatial heterogeneity and dependence. However, implementations of doubly-stochastic spatial models are computationally demanding, often have limited theoretical guarantee, and/or rely on restrictive assumptions. We propose a penalized regression method for estimating covariate effects in doubly-stochastic point processes that is computationally efficient and does not require a parametric form or stationarity of the underlying intensity. We establish the consistency and asymptotic normality of the proposed estimator, and develop a covariance estimator that leads to a conservative statistical inference procedure. A simulation study shows the validity of our approach under less restrictive assumptions on the data generating mechanism, and an application to Seattle crim
    
[^9]: 漫扩扩散模型和采样器的表达能力研究

    Expressiveness Remarks for Denoising Diffusion Models and Samplers. (arXiv:2305.09605v1 [stat.ML])

    [http://arxiv.org/abs/2305.09605](http://arxiv.org/abs/2305.09605)

    本文在漫扩扩散模型和采样器方面进行了表达能力的研究，通过将已知的神经网络逼近结果扩展到漫扩扩散模型和采样器来实现。

    

    漫扩扩散模型是一类生成模型，在许多领域最近已经取得了最先进的结果。通过漫扩过程逐渐向数据中添加噪声，将数据分布转化为高斯分布。然后，通过模拟该漫扩的时间反演的逼近来获取生成模型的样本，刚开始这个漫扩模拟的初始值是高斯样本。最近的研究探索了将漫扩模型适应于采样和推断任务。本文基于众所周知的与F\"ollmer漂移类似的随机控制联系，将针对F\"ollmer漂移的已知神经网络逼近结果扩展到漫扩扩散模型和采样器。

    Denoising diffusion models are a class of generative models which have recently achieved state-of-the-art results across many domains. Gradual noise is added to the data using a diffusion process, which transforms the data distribution into a Gaussian. Samples from the generative model are then obtained by simulating an approximation of the time reversal of this diffusion initialized by Gaussian samples. Recent research has explored adapting diffusion models for sampling and inference tasks. In this paper, we leverage known connections to stochastic control akin to the F\"ollmer drift to extend established neural network approximation results for the F\"ollmer drift to denoising diffusion models and samplers.
    

