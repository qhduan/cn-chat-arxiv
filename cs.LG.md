# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Prior-Dependent Allocations for Bayesian Fixed-Budget Best-Arm Identification in Structured Bandits](https://arxiv.org/abs/2402.05878) | 本论文研究了在结构化赌博机中的贝叶斯固定预算最佳臂识别问题，提出了一种基于先验信息的固定分配算法，并引入了新的证明方法，以得到更紧密的多臂BAI界限。该方法在各种情况下展现出一致且稳健的性能，加深了我们对于该问题的理解。 |
| [^2] | [A Bias-Variance Decomposition for Ensembles over Multiple Synthetic Datasets](https://arxiv.org/abs/2402.03985) | 本研究通过对多个合成数据集进行偏差-方差分解，增加了对其理论理解。实验证明多个合成数据集对于高方差的下游预测器特别有益，并提供了一个简单的经验法则用于选择适当的合成数据集数量。 |
| [^3] | [SafEDMD: A certified learning architecture tailored to data-driven control of nonlinear dynamical systems](https://arxiv.org/abs/2402.03145) | SafEDMD是一种基于EDMD的学习架构，通过稳定性和认证导向，生成可靠的数据驱动替代模型，并基于半定规划进行认证控制器设计。它在多个基准示例上展示了优于现有方法的优势。 |
| [^4] | [Tensor Networks for Explainable Machine Learning in Cybersecurity.](http://arxiv.org/abs/2401.00867) | 张量网络可以帮助发展可解释的机器学习算法，并提供丰富的模型可解释性。在网络安全中，我们的无监督聚类算法基于矩阵乘积状态，在性能上与传统的深度学习模型相媲美。我们的方法还能提取特征概率、熵和互信息，提供了分类异常的引人入胜的叙述，并实现了前所未有的透明度和可解释性水平。 |
| [^5] | [Adversarial Machine Learning in Latent Representations of Neural Networks.](http://arxiv.org/abs/2309.17401) | 这项研究通过分析分布式深度神经网络对抗性行为的韧性填补了现有研究空白，并发现潜在特征在相同信息失真水平下比输入表示更加韧性，并且对抗性韧性由特征维度和神经网络的泛化能力共同决定。 |
| [^6] | [Deep Optimal Transport for Domain Adaptation on SPD Manifolds.](http://arxiv.org/abs/2201.05745) | 这项研究介绍了一种基于深度最优传输的方法，用于解决在SPD流形上的领域自适应问题。通过利用最优传输理论和SPD流形的对数欧几里得几何，我们克服了协方差矩阵操作的复杂性挑战。 |

# 详细

[^1]: 基于先验依赖分配的结构化赌博机中贝叶斯固定预算最佳臂识别

    Prior-Dependent Allocations for Bayesian Fixed-Budget Best-Arm Identification in Structured Bandits

    [https://arxiv.org/abs/2402.05878](https://arxiv.org/abs/2402.05878)

    本论文研究了在结构化赌博机中的贝叶斯固定预算最佳臂识别问题，提出了一种基于先验信息的固定分配算法，并引入了新的证明方法，以得到更紧密的多臂BAI界限。该方法在各种情况下展现出一致且稳健的性能，加深了我们对于该问题的理解。

    

    我们研究了在结构化赌博机中的贝叶斯固定预算最佳臂识别（BAI）问题。我们提出了一种算法，该算法基于先验信息和环境结构使用固定分配。我们在多个模型中提供了它在性能上的理论界限，包括线性和分层BAI的首个先验依赖上界。我们的主要贡献是引入了新的证明方法，相比现有方法，它能得到更紧密的多臂BAI界限。我们广泛比较了我们的方法与其他固定预算BAI方法，在各种设置中展示了其一致且稳健的性能。我们的工作改进了对于结构化赌博机中贝叶斯固定预算BAI的理解，并突出了我们的方法在实际场景中的有效性。

    We study the problem of Bayesian fixed-budget best-arm identification (BAI) in structured bandits. We propose an algorithm that uses fixed allocations based on the prior information and the structure of the environment. We provide theoretical bounds on its performance across diverse models, including the first prior-dependent upper bounds for linear and hierarchical BAI. Our key contribution is introducing new proof methods that result in tighter bounds for multi-armed BAI compared to existing methods. We extensively compare our approach to other fixed-budget BAI methods, demonstrating its consistent and robust performance in various settings. Our work improves our understanding of Bayesian fixed-budget BAI in structured bandits and highlights the effectiveness of our approach in practical scenarios.
    
[^2]: 对多个合成数据集的集成进行偏差-方差分解

    A Bias-Variance Decomposition for Ensembles over Multiple Synthetic Datasets

    [https://arxiv.org/abs/2402.03985](https://arxiv.org/abs/2402.03985)

    本研究通过对多个合成数据集进行偏差-方差分解，增加了对其理论理解。实验证明多个合成数据集对于高方差的下游预测器特别有益，并提供了一个简单的经验法则用于选择适当的合成数据集数量。

    

    最近的研究强调了为监督学习生成多个合成数据集的好处，包括增加准确性、更有效的模型选择和不确定性估计。这些好处在经验上有明确的支持，但对它们的理论理解目前非常有限。我们通过推导使用多个合成数据集的几种设置的偏差-方差分解，来增加理论理解。我们的理论预测，对于高方差的下游预测器，多个合成数据集将特别有益，并为均方误差和Brier分数的情况提供了一个简单的经验法则来选择合适的合成数据集数量。我们通过评估一个集成在多个合成数据集和几个真实数据集以及下游预测器上的性能来研究我们的理论在实践中的效果。结果验证了我们的理论，表明我们的洞察也在实践中具有相关性。

    Recent studies have highlighted the benefits of generating multiple synthetic datasets for supervised learning, from increased accuracy to more effective model selection and uncertainty estimation. These benefits have clear empirical support, but the theoretical understanding of them is currently very light. We seek to increase the theoretical understanding by deriving bias-variance decompositions for several settings of using multiple synthetic datasets. Our theory predicts multiple synthetic datasets to be especially beneficial for high-variance downstream predictors, and yields a simple rule of thumb to select the appropriate number of synthetic datasets in the case of mean-squared error and Brier score. We investigate how our theory works in practice by evaluating the performance of an ensemble over many synthetic datasets for several real datasets and downstream predictors. The results follow our theory, showing that our insights are also practically relevant.
    
[^3]: SafEDMD：一种专为非线性动态系统数据驱动控制而设计的认证学习架构

    SafEDMD: A certified learning architecture tailored to data-driven control of nonlinear dynamical systems

    [https://arxiv.org/abs/2402.03145](https://arxiv.org/abs/2402.03145)

    SafEDMD是一种基于EDMD的学习架构，通过稳定性和认证导向，生成可靠的数据驱动替代模型，并基于半定规划进行认证控制器设计。它在多个基准示例上展示了优于现有方法的优势。

    

    Koopman算子作为机器学习动态控制系统的理论基础，其中算子通过扩展动态模态分解（EDMD）启发式近似。在本文中，我们提出了稳定性和认证导向的EDMD（SafEDMD）：一种新颖的基于EDMD的学习架构，它提供了严格的证书，从而以数据驱动的方式生成可靠的替代模型。为了确保SafEDMD的可靠性，我们推导出比例误差界限，这些界限在原点处消失，并且适用于控制任务，从而基于半定规划进行认证控制器设计。我们通过几个基准示例说明了所开发的机制，并强调其相对于现有方法的优势。

    The Koopman operator serves as the theoretical backbone for machine learning of dynamical control systems, where the operator is heuristically approximated by extended dynamic mode decomposition (EDMD). In this paper, we propose Stability- and certificate-oriented EDMD (SafEDMD): a novel EDMD-based learning architecture which comes along with rigorous certificates, resulting in a reliable surrogate model generated in a data-driven fashion. To ensure trustworthiness of SafEDMD, we derive proportional error bounds, which vanish at the origin and are tailored for control tasks, leading to certified controller design based on semi-definite programming. We illustrate the developed machinery by means of several benchmark examples and highlight the advantages over state-of-the-art methods.
    
[^4]: 张量网络在可解释的机器学习中在网络安全中的应用

    Tensor Networks for Explainable Machine Learning in Cybersecurity. (arXiv:2401.00867v1 [cs.LG])

    [http://arxiv.org/abs/2401.00867](http://arxiv.org/abs/2401.00867)

    张量网络可以帮助发展可解释的机器学习算法，并提供丰富的模型可解释性。在网络安全中，我们的无监督聚类算法基于矩阵乘积状态，在性能上与传统的深度学习模型相媲美。我们的方法还能提取特征概率、熵和互信息，提供了分类异常的引人入胜的叙述，并实现了前所未有的透明度和可解释性水平。

    

    本文展示了张量网络如何帮助发展可解释的机器学习算法。具体而言，我们基于矩阵乘积状态（MPS）开发了一种无监督聚类算法，并将其应用于实际使用案例中的对手生成的威胁情报。我们的研究证明，MPS在性能方面可以与传统的深度学习模型如自编码器和生成对抗网络相媲美，同时提供更丰富的模型可解释性。我们的方法自然地促进了特征概率、冯·诺伊曼熵和互信息的提取，为异常分类提供了引人入胜的叙述，并促进了前所未有的透明度和可解释性水平，这对于理解人工智能决策的基本原理至关重要。

    In this paper we show how tensor networks help in developing explainability of machine learning algorithms. Specifically, we develop an unsupervised clustering algorithm based on Matrix Product States (MPS) and apply it in the context of a real use-case of adversary-generated threat intelligence. Our investigation proves that MPS rival traditional deep learning models such as autoencoders and GANs in terms of performance, while providing much richer model interpretability. Our approach naturally facilitates the extraction of feature-wise probabilities, Von Neumann Entropy, and mutual information, offering a compelling narrative for classification of anomalies and fostering an unprecedented level of transparency and interpretability, something fundamental to understand the rationale behind artificial intelligence decisions.
    
[^5]: 神经网络潜在表示中的对抗性机器学习

    Adversarial Machine Learning in Latent Representations of Neural Networks. (arXiv:2309.17401v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2309.17401](http://arxiv.org/abs/2309.17401)

    这项研究通过分析分布式深度神经网络对抗性行为的韧性填补了现有研究空白，并发现潜在特征在相同信息失真水平下比输入表示更加韧性，并且对抗性韧性由特征维度和神经网络的泛化能力共同决定。

    

    分布式深度神经网络已被证明可以减轻移动设备的计算负担，并降低边缘计算场景中的端到端推理延迟。尽管已经对分布式深度神经网络进行了研究，但据我们所知，分布式深度神经网络对于对抗性行为的韧性仍然是一个开放问题。在本文中，我们通过严格分析分布式深度神经网络对抗性行为的韧性来填补现有的研究空白。我们将这个问题置于信息论的背景下，并引入了两个新的衡量指标来衡量失真和韧性。我们的理论发现表明：（i）在假设具有相同信息失真水平的情况下，潜在特征始终比输入表示更加韧性；（ii）对抗性韧性同时由特征维度和深度神经网络的泛化能力决定。为了验证我们的理论发现，我们进行了广泛的实验分析，考虑了6种不同的深度神经网络架构。

    Distributed deep neural networks (DNNs) have been shown to reduce the computational burden of mobile devices and decrease the end-to-end inference latency in edge computing scenarios. While distributed DNNs have been studied, to the best of our knowledge the resilience of distributed DNNs to adversarial action still remains an open problem. In this paper, we fill the existing research gap by rigorously analyzing the robustness of distributed DNNs against adversarial action. We cast this problem in the context of information theory and introduce two new measurements for distortion and robustness. Our theoretical findings indicate that (i) assuming the same level of information distortion, latent features are always more robust than input representations; (ii) the adversarial robustness is jointly determined by the feature dimension and the generalization capability of the DNN. To test our theoretical findings, we perform extensive experimental analysis by considering 6 different DNN arc
    
[^6]: 基于SPD流形的深度最优传输领域自适应

    Deep Optimal Transport for Domain Adaptation on SPD Manifolds. (arXiv:2201.05745v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2201.05745](http://arxiv.org/abs/2201.05745)

    这项研究介绍了一种基于深度最优传输的方法，用于解决在SPD流形上的领域自适应问题。通过利用最优传输理论和SPD流形的对数欧几里得几何，我们克服了协方差矩阵操作的复杂性挑战。

    

    近年来，机器学习界对于在对称正定（SPD）流形上解决领域自适应（DA）问题表现出了很大兴趣。这种兴趣源于医疗设备产生的复杂神经物理数据（如脑电图、脑磁图和扩散张量成像）在不同领域之间存在数据分布的偏移。这些数据表示以信号协方差矩阵的形式表示，并具有对称性和正定性的属性。然而，由于协方差矩阵的复杂操作特性，直接将先前的经验和解决方案应用于DA问题存在挑战。为了解决这个问题，我们的研究引入了一类基于深度学习的迁移学习方法，称为深度最优传输。这一类方法利用最优传输理论，并利用SPD流形的对数欧几里得几何。此外，我们还展示了...

    In recent years, there has been significant interest in solving the domain adaptation (DA) problem on symmetric positive definite (SPD) manifolds within the machine learning community. This interest stems from the fact that complex neurophysiological data generated by medical equipment, such as electroencephalograms, magnetoencephalograms, and diffusion tensor imaging, often exhibit a shift in data distribution across different domains. These data representations, represented by signal covariance matrices, possess properties of symmetry and positive definiteness. However, directly applying previous experiences and solutions to the DA problem poses challenges due to the manipulation complexities of covariance matrices.To address this, our research introduces a category of deep learning-based transfer learning approaches called deep optimal transport. This category utilizes optimal transport theory and leverages the Log-Euclidean geometry for SPD manifolds. Additionally, we present a com
    

