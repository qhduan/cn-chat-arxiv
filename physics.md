# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Learning interpretable causal networks from very large datasets, application to 400,000 medical records of breast cancer patients.](http://arxiv.org/abs/2303.06423) | 本文提出了一种更可靠和可扩展的因果发现方法（iMIIC），并在来自美国监测、流行病学和终末结果计划的396,179名乳腺癌患者的医疗保健数据上展示了其独特能力。超过90％的预测因果效应是正确的，而其余的意外直接和间接因果效应可以解释为诊断程序、治疗时间、患者偏好或社会经济差距。 |
| [^2] | [Generative Adversarial Networks for Scintillation Signal Simulation in EXO-200.](http://arxiv.org/abs/2303.06311) | 本文介绍了一种基于生成对抗网络的新方法，用于从EXO-200实验的时间投影室中模拟光电探测器信号。该方法能够比传统的模拟方法快一个数量级地产生高质量的模拟波形，并且能够从训练样本中推广并识别数据的显著高级特征。 |
| [^3] | [A Machine Learning Tutorial for Operational Meteorology, Part II: Neural Networks and Deep Learning.](http://arxiv.org/abs/2211.00147) | 本文讨论了机器学习在气象学中的应用，特别是神经网络和深度学习。涵盖了感知器、人工神经网络、卷积神经网络和U型网络等方法。 |
| [^4] | [PDEBENCH: An Extensive Benchmark for Scientific Machine Learning.](http://arxiv.org/abs/2210.07182) | PDEBench是一个基于偏微分方程的时间依赖性模拟任务基准套件，包括代码和数据，可用于对新型机器学习模型的性能进行基准测试，同时还可以与经典数值模拟和机器学习基线进行比较。 |
| [^5] | [What shapes the loss landscape of self-supervised learning?.](http://arxiv.org/abs/2210.00638) | 本文通过分析自监督学习的损失函数空间，回答了维度崩溃的原因和影响，以及维度崩溃如何有益，并影响SSL对数据不平衡的鲁棒性。 |
| [^6] | [Physics-Constrained Deep Learning for Climate Downscaling.](http://arxiv.org/abs/2208.05424) | 本文提出了一种物理约束深度学习降尺度模型的方法，以保证模型在预测物理变量时满足守恒定律，并提高其性能。 |
| [^7] | [Intrinsic dimension estimation for discrete metrics.](http://arxiv.org/abs/2207.09688) | 本文介绍了一种算法，用于推断嵌入离散空间的数据集的内在维度（ID），并在物种指纹的代谢组学数据集上展示了其准确性，发现一个令人惊讶的小ID，约为2的数量级。 |
| [^8] | [Learning Similarity Metrics for Volumetric Simulations with Multiscale CNNs.](http://arxiv.org/abs/2202.04109) | 本文提出了一种基于熵的相似度模型，用于评估基于运输和运动的模拟产生的标量和矢量数据的相似度，并提出了一种多尺度CNN架构，用于计算体积相似度度量（VolSiM）。 |
| [^9] | [Landslide Susceptibility Modeling by Interpretable Neural Network.](http://arxiv.org/abs/2201.06837) | 本文介绍了一种可解释神经网络（SNN）优化框架，用于评估滑坡易发性。SNN模型发现坡度和降水的乘积以及坡向是高滑坡易发性的重要主要贡献因素。 |

# 详细

[^1]: 从大型数据集中学习可解释的因果网络，以乳腺癌患者的40万份医疗记录为例

    Learning interpretable causal networks from very large datasets, application to 400,000 medical records of breast cancer patients. (arXiv:2303.06423v1 [q-bio.QM])

    [http://arxiv.org/abs/2303.06423](http://arxiv.org/abs/2303.06423)

    本文提出了一种更可靠和可扩展的因果发现方法（iMIIC），并在来自美国监测、流行病学和终末结果计划的396,179名乳腺癌患者的医疗保健数据上展示了其独特能力。超过90％的预测因果效应是正确的，而其余的意外直接和间接因果效应可以解释为诊断程序、治疗时间、患者偏好或社会经济差距。

    This paper proposes a more reliable and scalable causal discovery method (iMIIC) and showcases its unique capabilities on healthcare data from 396,179 breast cancer patients from the US Surveillance, Epidemiology, and End Results program. Over 90% of predicted causal effects appear correct, while the remaining unexpected direct and indirect causal effects can be interpreted in terms of diagnostic procedures, therapeutic timing, patient preference or socio-economic disparity.

    发现因果效应是科学研究的核心，但当只有观察数据可用时，这仍然具有挑战性。在实践中，因果网络难以学习和解释，并且仅限于相对较小的数据集。我们报告了一种更可靠和可扩展的因果发现方法（iMIIC），基于一般的互信息最大原则，它极大地提高了推断的因果关系的精度，同时区分了真正的原因和假定的和潜在的因果效应。我们展示了iMIIC在来自美国监测、流行病学和终末结果计划的396,179名乳腺癌患者的合成和现实医疗保健数据上的独特能力。超过90％的预测因果效应是正确的，而其余的意外直接和间接因果效应可以解释为诊断程序、治疗时间、患者偏好或社会经济差距。iMIIC的独特能力开辟了发现可靠和可解释的因果网络的新途径。

    Discovering causal effects is at the core of scientific investigation but remains challenging when only observational data is available. In practice, causal networks are difficult to learn and interpret, and limited to relatively small datasets. We report a more reliable and scalable causal discovery method (iMIIC), based on a general mutual information supremum principle, which greatly improves the precision of inferred causal relations while distinguishing genuine causes from putative and latent causal effects. We showcase iMIIC on synthetic and real-life healthcare data from 396,179 breast cancer patients from the US Surveillance, Epidemiology, and End Results program. More than 90\% of predicted causal effects appear correct, while the remaining unexpected direct and indirect causal effects can be interpreted in terms of diagnostic procedures, therapeutic timing, patient preference or socio-economic disparity. iMIIC's unique capabilities open up new avenues to discover reliable and
    
[^2]: 生成对抗网络在EXO-200闪烁信号模拟中的应用

    Generative Adversarial Networks for Scintillation Signal Simulation in EXO-200. (arXiv:2303.06311v1 [hep-ex])

    [http://arxiv.org/abs/2303.06311](http://arxiv.org/abs/2303.06311)

    本文介绍了一种基于生成对抗网络的新方法，用于从EXO-200实验的时间投影室中模拟光电探测器信号。该方法能够比传统的模拟方法快一个数量级地产生高质量的模拟波形，并且能够从训练样本中推广并识别数据的显著高级特征。

    This paper introduces a novel approach using Generative Adversarial Networks to simulate photodetector signals from the time projection chamber of the EXO-200 experiment. The method is able to produce high-quality simulated waveforms an order of magnitude faster than traditional simulation methods and can generalize from the training sample and discern salient high-level features of the data.

    基于模拟或实际事件样本训练的生成对抗网络被提出作为一种以降低计算成本为代价生成大规模模拟数据集的方法。本文展示了一种新的方法，用于从EXO-200实验的时间投影室中模拟光电探测器信号。该方法基于Wasserstein生成对抗网络，这是一种深度学习技术，允许对给定对象集的总体分布进行隐式非参数估计。我们的网络使用原始闪烁波形作为输入，通过对真实校准数据进行训练。我们发现，它能够比传统的模拟方法快一个数量级地产生高质量的模拟波形，并且重要的是，能够从训练样本中推广并识别数据的显著高级特征。特别是，网络正确推断出探测器中闪烁光响应的位置依赖性和相关性。

    Generative Adversarial Networks trained on samples of simulated or actual events have been proposed as a way of generating large simulated datasets at a reduced computational cost. In this work, a novel approach to perform the simulation of photodetector signals from the time projection chamber of the EXO-200 experiment is demonstrated. The method is based on a Wasserstein Generative Adversarial Network - a deep learning technique allowing for implicit non-parametric estimation of the population distribution for a given set of objects. Our network is trained on real calibration data using raw scintillation waveforms as input. We find that it is able to produce high-quality simulated waveforms an order of magnitude faster than the traditional simulation approach and, importantly, generalize from the training sample and discern salient high-level features of the data. In particular, the network correctly deduces position dependency of scintillation light response in the detector and corr
    
[^3]: 操作气象学的机器学习教程，第二部分：神经网络和深度学习

    A Machine Learning Tutorial for Operational Meteorology, Part II: Neural Networks and Deep Learning. (arXiv:2211.00147v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2211.00147](http://arxiv.org/abs/2211.00147)

    本文讨论了机器学习在气象学中的应用，特别是神经网络和深度学习。涵盖了感知器、人工神经网络、卷积神经网络和U型网络等方法。

    This paper discusses the application of machine learning in meteorology, specifically neural networks and deep learning. It covers methods such as perceptrons, artificial neural networks, convolutional neural networks, and U-networks.

    在过去的十年中，机器学习在气象学中的应用迅速增长。特别是神经网络和深度学习的使用率前所未有。为了填补缺乏以气象学视角涵盖神经网络的资源，本文以平易近人的语言格式讨论了机器学习方法，针对操作气象学界。这是一对旨在为气象学家提供机器学习资源的论文中的第二篇。第一篇论文侧重于传统的机器学习方法（例如随机森林），而本文则涵盖了广泛的神经网络和深度学习方法。具体而言，本文涵盖了感知器、人工神经网络、卷积神经网络和U型网络。与第一篇论文一样，本文讨论了与神经网络及其训练相关的术语。然后，本文提供了每种方法背后的一些直觉，并以展示每种方法的实例来结束。

    Over the past decade the use of machine learning in meteorology has grown rapidly. Specifically neural networks and deep learning have been used at an unprecedented rate. In order to fill the dearth of resources covering neural networks with a meteorological lens, this paper discusses machine learning methods in a plain language format that is targeted for the operational meteorological community. This is the second paper in a pair that aim to serve as a machine learning resource for meteorologists. While the first paper focused on traditional machine learning methods (e.g., random forest), here a broad spectrum of neural networks and deep learning methods are discussed. Specifically this paper covers perceptrons, artificial neural networks, convolutional neural networks and U-networks. Like the part 1 paper, this manuscript discusses the terms associated with neural networks and their training. Then the manuscript provides some intuition behind every method and concludes by showing ea
    
[^4]: PDEBENCH：科学机器学习的广泛基准测试

    PDEBENCH: An Extensive Benchmark for Scientific Machine Learning. (arXiv:2210.07182v6 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2210.07182](http://arxiv.org/abs/2210.07182)

    PDEBench是一个基于偏微分方程的时间依赖性模拟任务基准套件，包括代码和数据，可用于对新型机器学习模型的性能进行基准测试，同时还可以与经典数值模拟和机器学习基线进行比较。

    PDEBench is a benchmark suite of time-dependent simulation tasks based on Partial Differential Equations (PDEs), which includes code and data to benchmark the performance of novel machine learning models against both classical numerical simulations and machine learning baselines.

    近年来，基于机器学习的物理系统建模受到了越来越多的关注。尽管取得了一些令人印象深刻的进展，但仍缺乏易于使用但具有挑战性和代表性的科学ML基准测试。我们介绍了PDEBench，这是一个基于偏微分方程（PDE）的时间依赖性模拟任务基准套件。PDEBench包括代码和数据，以对新型机器学习模型的性能进行基准测试，同时还可以与经典数值模拟和机器学习基线进行比较。我们提出的基准问题集具有以下独特特征：（1）与现有基准测试相比，PDE的范围更广，从相对常见的示例到更现实和困难的问题；（2）与先前的工作相比，准备好使用的数据集更大，包括跨更多初始和边界条件以及PDE参数的多个模拟运行；（3）更广泛的基准测试，包括更多的性能指标和评估方法。

    Machine learning-based modeling of physical systems has experienced increased interest in recent years. Despite some impressive progress, there is still a lack of benchmarks for Scientific ML that are easy to use but still challenging and representative of a wide range of problems. We introduce PDEBench, a benchmark suite of time-dependent simulation tasks based on Partial Differential Equations (PDEs). PDEBench comprises both code and data to benchmark the performance of novel machine learning models against both classical numerical simulations and machine learning baselines. Our proposed set of benchmark problems contribute the following unique features: (1) A much wider range of PDEs compared to existing benchmarks, ranging from relatively common examples to more realistic and difficult problems; (2) much larger ready-to-use datasets compared to prior work, comprising multiple simulation runs across a larger number of initial and boundary conditions and PDE parameters; (3) more exte
    
[^5]: 自监督学习的损失函数空间是如何形成的？

    What shapes the loss landscape of self-supervised learning?. (arXiv:2210.00638v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2210.00638](http://arxiv.org/abs/2210.00638)

    本文通过分析自监督学习的损失函数空间，回答了维度崩溃的原因和影响，以及维度崩溃如何有益，并影响SSL对数据不平衡的鲁棒性。

    This paper answers questions about the causes and effects of dimensional collapse in self-supervised learning (SSL) by analyzing the SSL loss landscape, and explores how dimensional collapse can be beneficial and affect the robustness of SSL against data imbalance.

    最近，防止表示完全和维度崩溃已成为自监督学习（SSL）的设计原则。然而，我们对理论的理解仍有疑问：这些崩溃何时发生？机制和原因是什么？我们通过推导和彻底分析SSL损失函数空间的可分析理论来回答这些问题。在这个理论中，我们确定了维度崩溃的原因，并研究了归一化和偏差的影响。最后，我们利用分析理论所提供的可解释性来理解维度崩溃如何有益，并影响SSL对数据不平衡的鲁棒性。

    Prevention of complete and dimensional collapse of representations has recently become a design principle for self-supervised learning (SSL). However, questions remain in our theoretical understanding: When do those collapses occur? What are the mechanisms and causes? We answer these questions by deriving and thoroughly analyzing an analytically tractable theory of SSL loss landscapes. In this theory, we identify the causes of the dimensional collapse and study the effect of normalization and bias. Finally, we leverage the interpretability afforded by the analytical theory to understand how dimensional collapse can be beneficial and what affects the robustness of SSL against data imbalance.
    
[^6]: 物理约束深度学习用于气候降尺度

    Physics-Constrained Deep Learning for Climate Downscaling. (arXiv:2208.05424v6 [physics.ao-ph] UPDATED)

    [http://arxiv.org/abs/2208.05424](http://arxiv.org/abs/2208.05424)

    本文提出了一种物理约束深度学习降尺度模型的方法，以保证模型在预测物理变量时满足守恒定律，并提高其性能。

    This paper proposes a method for physics-constrained deep learning downscaling models to ensure that the models satisfy conservation laws when predicting physical variables, while improving their performance according to traditional metrics.

    可靠的高分辨率气候和天气数据的可用性对于指导气候适应和减缓的长期决策以及指导对极端事件的快速响应至关重要。预测模型受计算成本限制，因此通常生成粗分辨率预测。统计降尺度，包括深度学习的超分辨率方法，可以提供一种有效的方法来上采样低分辨率数据。然而，尽管在某些情况下取得了视觉上令人信服的结果，但这些模型在预测物理变量时经常违反守恒定律。为了保持物理量的守恒，我们开发了一种方法，保证深度学习降尺度模型满足物理约束条件，同时根据传统指标提高其性能。我们比较了不同的约束方法，并展示了它们在不同的神经架构以及各种气候和天气数据上的适用性。

    The availability of reliable, high-resolution climate and weather data is important to inform long-term decisions on climate adaptation and mitigation and to guide rapid responses to extreme events. Forecasting models are limited by computational costs and, therefore, often generate coarse-resolution predictions. Statistical downscaling, including super-resolution methods from deep learning, can provide an efficient method of upsampling low-resolution data. However, despite achieving visually compelling results in some cases, such models frequently violate conservation laws when predicting physical variables. In order to conserve physical quantities, we develop methods that guarantee physical constraints are satisfied by a deep learning downscaling model while also improving their performance according to traditional metrics. We compare different constraining approaches and demonstrate their applicability across different neural architectures as well as a variety of climate and weather
    
[^7]: 离散度量的内在维度估计

    Intrinsic dimension estimation for discrete metrics. (arXiv:2207.09688v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2207.09688](http://arxiv.org/abs/2207.09688)

    本文介绍了一种算法，用于推断嵌入离散空间的数据集的内在维度（ID），并在物种指纹的代谢组学数据集上展示了其准确性，发现一个令人惊讶的小ID，约为2的数量级。

    This paper introduces an algorithm to estimate the intrinsic dimension (ID) of datasets embedded in discrete spaces, and demonstrates its accuracy on a metagenomic dataset for species fingerprinting, finding a surprisingly small ID of order 2, suggesting that evolutive pressure acts on a low-dimensional manifold despite the high-dimensionality of sequences' space.

    具有离散特征的真实世界数据集是无处不在的：从分类调查到临床问卷，从无权网络到DNA序列。然而，最常见的无监督降维方法是为连续空间设计的，它们在离散空间中的使用可能会导致错误和偏差。在本文中，我们介绍了一种算法，用于推断嵌入离散空间的数据集的内在维度（ID）。我们在基准数据集上展示了其准确性，并将其应用于分析用于物种指纹的代谢组学数据集，发现一个令人惊讶的小ID，约为2的数量级。这表明，尽管序列空间的高维度，进化压力仍然作用于低维流形上。

    Real world-datasets characterized by discrete features are ubiquitous: from categorical surveys to clinical questionnaires, from unweighted networks to DNA sequences. Nevertheless, the most common unsupervised dimensional reduction methods are designed for continuous spaces, and their use for discrete spaces can lead to errors and biases. In this letter we introduce an algorithm to infer the intrinsic dimension (ID) of datasets embedded in discrete spaces. We demonstrate its accuracy on benchmark datasets, and we apply it to analyze a metagenomic dataset for species fingerprinting, finding a surprisingly small ID, of order 2. This suggests that evolutive pressure acts on a low-dimensional manifold despite the high-dimensionality of sequences' space.
    
[^8]: 基于多尺度CNN的体积模拟相似度度量学习

    Learning Similarity Metrics for Volumetric Simulations with Multiscale CNNs. (arXiv:2202.04109v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2202.04109](http://arxiv.org/abs/2202.04109)

    本文提出了一种基于熵的相似度模型，用于评估基于运输和运动的模拟产生的标量和矢量数据的相似度，并提出了一种多尺度CNN架构，用于计算体积相似度度量（VolSiM）。

    This paper proposes a similarity model based on entropy for assessing the similarity of scalar and vectorial data produced from transport and motion-based simulations, and a multiscale CNN architecture for computing a volumetric similarity metric (VolSiM).

    三维数据模拟在科学中应用广泛，从流体流动到等离子物理。本文提出了一种基于熵的相似度模型，允许创建物理上有意义的基准距离，用于评估基于运输和运动的模拟产生的标量和矢量数据的相似度。利用从该模型导出的两种数据采集方法，我们创建了从数值PDE求解器和现有模拟数据存储库中收集的场集合。此外，我们提出了一种多尺度CNN架构，用于计算体积相似度度量（VolSiM）。据我们所知，这是第一种天然设计用于解决高维模拟数据相似度评估挑战的学习方法。此外，我们还研究了基于相关损失函数的大批量大小和准确相关计算之间的权衡，并研究了该度量的不变性。

    Simulations that produce three-dimensional data are ubiquitous in science, ranging from fluid flows to plasma physics. We propose a similarity model based on entropy, which allows for the creation of physically meaningful ground truth distances for the similarity assessment of scalar and vectorial data, produced from transport and motion-based simulations. Utilizing two data acquisition methods derived from this model, we create collections of fields from numerical PDE solvers and existing simulation data repositories. Furthermore, a multiscale CNN architecture that computes a volumetric similarity metric (VolSiM) is proposed. To the best of our knowledge this is the first learning method inherently designed to address the challenges arising for the similarity assessment of high-dimensional simulation data. Additionally, the tradeoff between a large batch size and an accurate correlation computation for correlation-based loss functions is investigated, and the metric's invariance with 
    
[^9]: 可解释神经网络在滑坡易发性建模中的应用

    Landslide Susceptibility Modeling by Interpretable Neural Network. (arXiv:2201.06837v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2201.06837](http://arxiv.org/abs/2201.06837)

    本文介绍了一种可解释神经网络（SNN）优化框架，用于评估滑坡易发性。SNN模型发现坡度和降水的乘积以及坡向是高滑坡易发性的重要主要贡献因素。

    This paper introduces an interpretable neural network framework, called superposable neural network (SNN) optimization, for assessing landslide susceptibility. The SNN models found the product of slope and precipitation and hillslope aspect to be important primary contributors to high landslide susceptibility.

    滑坡由于许多时空变化因素影响而难以预测。人工神经网络（ANN）已被证明可以提高预测准确性，但缺乏可解释性。本文介绍了一种可加性ANN优化框架，用于评估滑坡易发性，以及数据集划分和结果解释技术。我们将这种具有完全可解释性、高准确性、高泛化性和低模型复杂度的方法称为可叠加神经网络（SNN）优化。我们通过对来自三个不同东喜马拉雅地区的滑坡清单进行模型训练来验证我们的方法。我们的SNN优于基于物理和统计模型，并实现了类似于最先进的深度神经网络的性能。SNN模型发现坡度和降水的乘积以及坡向是高滑坡易发性的重要主要贡献因素。

    Landslides are notoriously difficult to predict because numerous spatially and temporally varying factors contribute to slope stability. Artificial neural networks (ANN) have been shown to improve prediction accuracy but are largely uninterpretable. Here we introduce an additive ANN optimization framework to assess landslide susceptibility, as well as dataset division and outcome interpretation techniques. We refer to our approach, which features full interpretability, high accuracy, high generalizability and low model complexity, as superposable neural network (SNN) optimization. We validate our approach by training models on landslide inventory from three different easternmost Himalaya regions. Our SNN outperformed physically-based and statistical models and achieved similar performance to state-of-the-art deep neural networks. The SNN models found the product of slope and precipitation and hillslope aspect to be important primary contributors to high landslide susceptibility, which 
    

