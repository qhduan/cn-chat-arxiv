# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Unraveling the Key of Machine Learning Solutions for Android Malware Detection](https://arxiv.org/abs/2402.02953) | 本文通过调查和分析，提出了一个全面研究基于机器学习的Android恶意软件检测的方案，并重新实现了12个代表性方法的评估。 |
| [^2] | [Graph Neural Networks for Graphs with Heterophily: A Survey](https://arxiv.org/abs/2202.07082) | 该论文提出了对具有异质性的图进行图神经网络研究的系统回顾，并提出了系统性分类法以指导现有异质性GNN模型。 |
| [^3] | [Neural Network-Based Score Estimation in Diffusion Models: Optimization and Generalization.](http://arxiv.org/abs/2401.15604) | 本文提出了基于神经网络的扩散模型中分数估计的优化和泛化方法，并建立了对分数估计进行分析的数学框架。 |
| [^4] | [Multimodal Sentiment Analysis with Missing Modality: A Knowledge-Transfer Approach.](http://arxiv.org/abs/2401.10747) | 本文提出了一种知识迁移方法，用于在缺失模态下进行多模态情感分析。通过翻译不同模态之间的内容以重构缺失的音频模态，并利用跨模态注意机制进行情感预测，实验证明了该方法在多个数据集上表现出显著的改进和与完整多模态监督方法相媲美的效果。 |
| [^5] | [Deep Learning Predicts Biomarker Status and Discovers Related Histomorphology Characteristics for Low-Grade Glioma.](http://arxiv.org/abs/2310.07464) | 该研究提出了一种基于深度学习的多生物标志物组织形态学发现者模型，利用全切片图像预测低级别胶质瘤中五个生物标志物的状态。该模型通过将单类分类纳入多实例学习框架，实现了准确的实例级别监督，在提高生物标志物预测性能方面表现出优势。 |
| [^6] | [Neural Operator: Is data all you need to model the world? An insight into the impact of Physics Informed Machine Learning.](http://arxiv.org/abs/2301.13331) | 本文探讨了如何将数据驱动方法与传统技术相结合，以解决工程和物理问题，并指出了机器学习方法的一些主要问题。 |
| [^7] | [PyEPO: A PyTorch-based End-to-End Predict-then-Optimize Library for Linear and Integer Programming.](http://arxiv.org/abs/2206.14234) | PyEPO是一种针对线性规划和整数规划的基于PyTorch的端到端预测优化库，提供了一种新的方法来解决历史数据中存在先验未知参数的问题。它是具有预测目标函数系数的一般性构架，并提供了四种基本算法。 |

# 详细

[^1]: 揭示解决Android恶意软件检测的机器学习方案的关键

    Unraveling the Key of Machine Learning Solutions for Android Malware Detection

    [https://arxiv.org/abs/2402.02953](https://arxiv.org/abs/2402.02953)

    本文通过调查和分析，提出了一个全面研究基于机器学习的Android恶意软件检测的方案，并重新实现了12个代表性方法的评估。

    

    Android恶意软件检测作为对恶意应用程序的第一道防线。随着机器学习（ML）的快速发展，基于ML的Android恶意软件检测因其能够自动捕获Android APK中的恶意模式而受到越来越多的关注。这些基于学习的方法在检测恶意软件方面取得了有希望的结果。然而，缺乏对当前研究进展的深入分析，使得很难对这一领域的最新发展有一个全面的了解。本文对基于ML的Android恶意软件检测进行了全面的实证和定量分析。我们首先对文献进行了调查，并根据Android特征工程和ML建模过程将贡献分类。然后，我们设计了一个通用的ML-based Android恶意软件检测框架，重新实现了来自不同研究社区的12个代表性方法，并从三个方面对其进行了评估。

    Android malware detection serves as the front line against malicious apps. With the rapid advancement of machine learning (ML), ML-based Android malware detection has attracted increasing attention due to its capability of automatically capturing malicious patterns from Android APKs. These learning-driven methods have reported promising results in detecting malware. However, the absence of an in-depth analysis of current research progress makes it difficult to gain a holistic picture of the state of the art in this area.   This paper presents a comprehensive investigation to date into ML-based Android malware detection with empirical and quantitative analysis. We first survey the literature, categorizing contributions into a taxonomy based on the Android feature engineering and ML modeling pipeline. Then, we design a general-propose framework for ML-based Android malware detection, re-implement 12 representative approaches from different research communities, and evaluate them from thr
    
[^2]: 具有异质性的图神经网络：一项调查

    Graph Neural Networks for Graphs with Heterophily: A Survey

    [https://arxiv.org/abs/2202.07082](https://arxiv.org/abs/2202.07082)

    该论文提出了对具有异质性的图进行图神经网络研究的系统回顾，并提出了系统性分类法以指导现有异质性GNN模型。

    

    近年来，图神经网络（GNNs）的快速发展使得许多图分析任务和应用受益。大多数GNNs通常依赖于同质性假设，即属于同一类别的节点更可能相连。然而，在许多真实场景中，作为一种普遍的图属性，即不同标签的节点往往相连，这显著限制了定制的同质性GNNs的性能。因此，针对异质性图的GNNs正受到越来越多的研究关注，以增强对具有异质性的图学习。本文针对具有异质性的图提供了全面的GNNs回顾。具体来说，我们提出了一个系统性分类法，本质上指导着现有的异质性GNN模型，以及一个概括性摘要和详细分析。

    arXiv:2202.07082v2 Announce Type: replace  Abstract: Recent years have witnessed fast developments of graph neural networks (GNNs) that have benefited myriads of graph analytic tasks and applications. In general, most GNNs depend on the homophily assumption that nodes belonging to the same class are more likely to be connected. However, as a ubiquitous graph property in numerous real-world scenarios, heterophily, i.e., nodes with different labels tend to be linked, significantly limits the performance of tailor-made homophilic GNNs. Hence, GNNs for heterophilic graphs are gaining increasing research attention to enhance graph learning with heterophily. In this paper, we provide a comprehensive review of GNNs for heterophilic graphs. Specifically, we propose a systematic taxonomy that essentially governs existing heterophilic GNN models, along with a general summary and detailed analysis. %Furthermore, we summarize the mainstream heterophilic graph benchmarks to facilitate robust and fa
    
[^3]: 基于神经网络的扩散模型中的分数估计：优化和泛化

    Neural Network-Based Score Estimation in Diffusion Models: Optimization and Generalization. (arXiv:2401.15604v1 [cs.LG])

    [http://arxiv.org/abs/2401.15604](http://arxiv.org/abs/2401.15604)

    本文提出了基于神经网络的扩散模型中分数估计的优化和泛化方法，并建立了对分数估计进行分析的数学框架。

    

    扩散模型已经成为与GANs相媲美的强大工具，可以生成具有改进保真度，灵活性和鲁棒性的高质量样本。这些模型的一个关键组成部分是通过分数匹配来学习分数函数。尽管在各种任务上取得了实证成功，但尚不清楚基于梯度的算法是否可以以可证实的准确性学习分数函数。作为回答这个问题的首要步骤，本文建立了一个数学框架，用于分析用梯度下降训练的神经网络来进行分数估计。我们的分析包括学习过程的优化和泛化方面。特别是，我们提出了一个参数化形式来将去噪分数匹配问题制定为带有噪声标签的回归问题。与标准的监督学习设置相比，分数匹配问题引入了独特的挑战，包括无界输入，向量值输出和额外的时间变量。

    Diffusion models have emerged as a powerful tool rivaling GANs in generating high-quality samples with improved fidelity, flexibility, and robustness. A key component of these models is to learn the score function through score matching. Despite empirical success on various tasks, it remains unclear whether gradient-based algorithms can learn the score function with a provable accuracy. As a first step toward answering this question, this paper establishes a mathematical framework for analyzing score estimation using neural networks trained by gradient descent. Our analysis covers both the optimization and the generalization aspects of the learning procedure. In particular, we propose a parametric form to formulate the denoising score-matching problem as a regression with noisy labels. Compared to the standard supervised learning setup, the score-matching problem introduces distinct challenges, including unbounded input, vector-valued output, and an additional time variable, preventing
    
[^4]: 缺失模态下的多模态情感分析:一种知识迁移方法

    Multimodal Sentiment Analysis with Missing Modality: A Knowledge-Transfer Approach. (arXiv:2401.10747v1 [cs.SD])

    [http://arxiv.org/abs/2401.10747](http://arxiv.org/abs/2401.10747)

    本文提出了一种知识迁移方法，用于在缺失模态下进行多模态情感分析。通过翻译不同模态之间的内容以重构缺失的音频模态，并利用跨模态注意机制进行情感预测，实验证明了该方法在多个数据集上表现出显著的改进和与完整多模态监督方法相媲美的效果。

    

    多模态情感分析旨在通过视觉、语言和声音线索来识别个体表达的情绪。然而，现有研究大多假设在训练和测试过程中所有模态都是可用的，这使得它们的算法容易受到缺失模态的影响。在本文中，我们提出了一种新颖的知识迁移网络，用于在不同模态之间进行翻译，以重构缺失的音频模态。此外，我们还开发了一种跨模态注意机制，以保留重构和观察到的模态的最大信息，用于情感预测。在三个公开数据集上进行的大量实验证明了相对于基线算法的显著改进，并实现了与具有完整多模态监督的先前方法相媲美的结果。

    Multimodal sentiment analysis aims to identify the emotions expressed by individuals through visual, language, and acoustic cues. However, most of the existing research efforts assume that all modalities are available during both training and testing, making their algorithms susceptible to the missing modality scenario. In this paper, we propose a novel knowledge-transfer network to translate between different modalities to reconstruct the missing audio modalities. Moreover, we develop a cross-modality attention mechanism to retain the maximal information of the reconstructed and observed modalities for sentiment prediction. Extensive experiments on three publicly available datasets demonstrate significant improvements over baselines and achieve comparable results to the previous methods with complete multi-modality supervision.
    
[^5]: 深度学习预测低级别胶质瘤的生物标志物状态并发现相关的组织形态学特征

    Deep Learning Predicts Biomarker Status and Discovers Related Histomorphology Characteristics for Low-Grade Glioma. (arXiv:2310.07464v1 [eess.IV])

    [http://arxiv.org/abs/2310.07464](http://arxiv.org/abs/2310.07464)

    该研究提出了一种基于深度学习的多生物标志物组织形态学发现者模型，利用全切片图像预测低级别胶质瘤中五个生物标志物的状态。该模型通过将单类分类纳入多实例学习框架，实现了准确的实例级别监督，在提高生物标志物预测性能方面表现出优势。

    

    生物标志物检测是低级别胶质瘤（LGG）诊断和治疗中不可或缺的一部分。然而，当前的LGG生物标志物检测方法依赖于昂贵而复杂的分子遗传学测试，需要专业人员分析结果，且常常存在内部一致性差异。为了克服这些挑战，我们提出了一种可解释的深度学习流程，即基于多实例学习（MIL）框架的多生物标志物组织形态学发现者（Multi-Beholder）模型，仅使用苏木精-伊红染色的全切片图像和切片级生物标志物状态标签预测LGG中五个生物标志物的状态。具体而言，通过将单类分类融入MIL框架，实现了准确的示例伪标签，用于实例级别监督，这极大地补充了切片级标签，并提高了生物标志物预测性能。 Multi-Beholder展示了优越的预测性能。

    Biomarker detection is an indispensable part in the diagnosis and treatment of low-grade glioma (LGG). However, current LGG biomarker detection methods rely on expensive and complex molecular genetic testing, for which professionals are required to analyze the results, and intra-rater variability is often reported. To overcome these challenges, we propose an interpretable deep learning pipeline, a Multi-Biomarker Histomorphology Discoverer (Multi-Beholder) model based on the multiple instance learning (MIL) framework, to predict the status of five biomarkers in LGG using only hematoxylin and eosin-stained whole slide images and slide-level biomarker status labels. Specifically, by incorporating the one-class classification into the MIL framework, accurate instance pseudo-labeling is realized for instance-level supervision, which greatly complements the slide-level labels and improves the biomarker prediction performance. Multi-Beholder demonstrates superior prediction performance and g
    
[^6]: 神经操作员：数据是否足以模拟世界？对物理启示机器学习影响的洞察

    Neural Operator: Is data all you need to model the world? An insight into the impact of Physics Informed Machine Learning. (arXiv:2301.13331v2 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2301.13331](http://arxiv.org/abs/2301.13331)

    本文探讨了如何将数据驱动方法与传统技术相结合，以解决工程和物理问题，并指出了机器学习方法的一些主要问题。

    

    常常使用偏微分方程（PDE）的数值近似来构建解决物理、工程和数学问题的方案，这些问题涉及到多个变量的函数，比如热传导或声音传播、流体流动、弹性、静电学、电动力学等。虽然这在解决许多复杂现象方面发挥了作用，但存在一些限制。常规方法如有限元法（FEM）和有限差分法（FDM）需要大量时间且计算成本高。相比之下，数据驱动的基于神经网络的方法提供了一种更快速、相对准确的替代方案，并具有离散不变性和分辨率不变性等优势。本文旨在深入了解数据驱动方法如何与传统技术相辅相成，解决工程和物理问题，同时指出机器学习方法的一些主要问题。

    Numerical approximations of partial differential equations (PDEs) are routinely employed to formulate the solution of physics, engineering and mathematical problems involving functions of several variables, such as the propagation of heat or sound, fluid flow, elasticity, electrostatics, electrodynamics, and more. While this has led to solving many complex phenomena, there are some limitations. Conventional approaches such as Finite Element Methods (FEMs) and Finite Differential Methods (FDMs) require considerable time and are computationally expensive. In contrast, data driven machine learning-based methods such as neural networks provide a faster, fairly accurate alternative, and have certain advantages such as discretization invariance and resolution invariance. This article aims to provide a comprehensive insight into how data-driven approaches can complement conventional techniques to solve engineering and physics problems, while also noting some of the major pitfalls of machine l
    
[^7]: PyEPO: 一种基于PyTorch的端到端预测优化库，适用于线性规划和整数规划

    PyEPO: A PyTorch-based End-to-End Predict-then-Optimize Library for Linear and Integer Programming. (arXiv:2206.14234v2 [math.OC] UPDATED)

    [http://arxiv.org/abs/2206.14234](http://arxiv.org/abs/2206.14234)

    PyEPO是一种针对线性规划和整数规划的基于PyTorch的端到端预测优化库，提供了一种新的方法来解决历史数据中存在先验未知参数的问题。它是具有预测目标函数系数的一般性构架，并提供了四种基本算法。

    

    在确定性优化中，通常假定所有问题参数都是固定和已知的。然而，在实践中，一些参数可能是先验未知的，但可以从历史数据中估计出来。一种典型的预测优化方法将预测和优化分为两个阶段。最近，端到端预测优化已成为一种有吸引力的替代方案。在本文中，我们介绍了PyEPO包，这是一个基于PyTorch的端到端预测优化库。据我们所知，PyEPO（发音类似于带有一个“n”的菠萝）是第一个具有预测目标函数系数的一般性线性和整数规划工具。它提供了四种基本算法：Elmachtoub和Grigas的半凸估计函数、Pogancic等人的可微黑盒求解器方法、以及Berthet等人的两种可微扰动方法。PyEPO为使用的算法提供了简单的接口。

    In deterministic optimization, it is typically assumed that all problem parameters are fixed and known. In practice, however, some parameters may be a priori unknown but can be estimated from historical data. A typical predict-then-optimize approach separates predictions and optimization into two stages. Recently, end-to-end predict-then-optimize has become an attractive alternative. In this work, we present the PyEPO package, a PyTorchbased end-to-end predict-then-optimize library in Python. To the best of our knowledge, PyEPO (pronounced like pineapple with a silent "n") is the first such generic tool for linear and integer programming with predicted objective function coefficients. It provides four base algorithms: a convex surrogate loss function from the seminal work of Elmachtoub and Grigas [16], a differentiable black-box solver approach of Pogancic et al. [35], and two differentiable perturbation-based methods from Berthet et al. [6]. PyEPO provides a simple interface for the d
    

