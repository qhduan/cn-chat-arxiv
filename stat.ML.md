# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Attentional Graph Neural Networks for Robust Massive Network Localization](https://arxiv.org/abs/2311.16856) | 本文通过将图神经网络与注意机制相结合，提出了一种用于网络定位的新方法。该方法具有出色的精确度，甚至在严重非直视视线条件下也能表现出良好的效果。通过提出的关注图神经网络模型，我们进一步改善了现有方法的灵活性和对超参数的敏感性。 |
| [^2] | [Differentially Private Sliced Inverse Regression: Minimax Optimality and Algorithm.](http://arxiv.org/abs/2401.08150) | 本文提出了针对充足维度减少中的隐私问题的最佳差分隐私算法，并在低维和高维设置下建立了不同ially private 切片逆回归的下界。通过仿真和真实数据分析验证了这些算法的有效性。 |
| [^3] | [Understanding deep neural networks through the lens of their non-linearity.](http://arxiv.org/abs/2310.11439) | 本文提出了一个理论上有效的解决方案，通过亲和度评分追踪深度神经网络中的非线性传播，尤其关注计算机视觉应用。实验证实了所提出方法的实用性和对广泛应用的潜力。 |
| [^4] | [Weighted Averaged Stochastic Gradient Descent: Asymptotic Normality and Optimality.](http://arxiv.org/abs/2307.06915) | 本文探索了一种加权平均随机梯度下降（SGD）方案，并建立了渐近正态性，提供了渐近有效的在线推理方法。此外，我们提出了一种自适应平均方案，具有最优的统计速度和有利的非渐近收敛性。 |
| [^5] | [Predicting Census Survey Response Rates With Parsimonious Additive Models and Structured Interactions.](http://arxiv.org/abs/2108.11328) | 本文提出了一种可解释的非参数加性模型，使用少量主要和成对交互效应预测调查反应率。该模型可以生成易于可视化和解释的预测面，并取得了 ROAM 数据集上的最先进性能，可以提供改进美国人口普查局和其他调查的反应率议论。 |

# 详细

[^1]: 关注图神经网络用于稳健的大规模网络定位

    Attentional Graph Neural Networks for Robust Massive Network Localization

    [https://arxiv.org/abs/2311.16856](https://arxiv.org/abs/2311.16856)

    本文通过将图神经网络与注意机制相结合，提出了一种用于网络定位的新方法。该方法具有出色的精确度，甚至在严重非直视视线条件下也能表现出良好的效果。通过提出的关注图神经网络模型，我们进一步改善了现有方法的灵活性和对超参数的敏感性。

    

    近年来，图神经网络(GNNs)已成为机器学习分类任务中的重要工具。然而，它们在回归任务中的应用仍然未被充分探索。为了发掘GNNs在回归中的潜力，本文将GNNs与注意机制相结合，这是一种通过其适应性和鲁棒性彻底改变了序列学习任务的技术，以解决一个具有挑战性的非线性回归问题：网络定位。我们首先介绍了一种基于图卷积网络(GCN)的新型网络定位方法，即使在严重非直视视线(NLOS)条件下也表现出卓越的精度，从而减少了繁琐的离线校准或NLOS识别的需求。我们进一步提出了一种关注图神经网络(AGNN)模型，旨在改善基于GCN方法的有限灵活性和对超参数的高敏感性。

    arXiv:2311.16856v2 Announce Type: replace Abstract: In recent years, Graph neural networks (GNNs) have emerged as a prominent tool for classification tasks in machine learning. However, their application in regression tasks remains underexplored. To tap the potential of GNNs in regression, this paper integrates GNNs with attention mechanism, a technique that revolutionized sequential learning tasks with its adaptability and robustness, to tackle a challenging nonlinear regression problem: network localization. We first introduce a novel network localization method based on graph convolutional network (GCN), which exhibits exceptional precision even under severe non-line-of-sight (NLOS) conditions, thereby diminishing the need for laborious offline calibration or NLOS identification. We further propose an attentional graph neural network (AGNN) model, aimed at improving the limited flexibility and mitigating the high sensitivity to the hyperparameter of the GCN-based method. The AGNN co
    
[^2]: 差分隐私切片逆回归: 极小极大性和算法

    Differentially Private Sliced Inverse Regression: Minimax Optimality and Algorithm. (arXiv:2401.08150v1 [stat.ML])

    [http://arxiv.org/abs/2401.08150](http://arxiv.org/abs/2401.08150)

    本文提出了针对充足维度减少中的隐私问题的最佳差分隐私算法，并在低维和高维设置下建立了不同ially private 切片逆回归的下界。通过仿真和真实数据分析验证了这些算法的有效性。

    

    随着数据驱动应用的普及，隐私保护已成为高维数据分析中的一个关键问题。切片逆回归是一种广泛应用的统计技术，通过降低协变量的维度，同时保持足够的统计信息。本文提出了针对充足维度减少中的隐私问题的最佳差分隐私算法。我们在低维和高维设置下建立了不同ially private 切片逆回归的下界。此外，我们设计了差分隐私算法，实现了极小极大下界的要求，并在降维空间中同时保护隐私和保存重要信息的有效性。通过一系列的仿真实验和真实数据分析，我们证明了这些差分隐私算法的有效性。

    Privacy preservation has become a critical concern in high-dimensional data analysis due to the growing prevalence of data-driven applications. Proposed by Li (1991), sliced inverse regression has emerged as a widely utilized statistical technique for reducing covariate dimensionality while maintaining sufficient statistical information. In this paper, we propose optimally differentially private algorithms specifically designed to address privacy concerns in the context of sufficient dimension reduction. We proceed to establish lower bounds for differentially private sliced inverse regression in both the low and high-dimensional settings. Moreover, we develop differentially private algorithms that achieve the minimax lower bounds up to logarithmic factors. Through a combination of simulations and real data analysis, we illustrate the efficacy of these differentially private algorithms in safeguarding privacy while preserving vital information within the reduced dimension space. As a na
    
[^3]: 通过非线性研究深度神经网络的理解

    Understanding deep neural networks through the lens of their non-linearity. (arXiv:2310.11439v1 [cs.LG])

    [http://arxiv.org/abs/2310.11439](http://arxiv.org/abs/2310.11439)

    本文提出了一个理论上有效的解决方案，通过亲和度评分追踪深度神经网络中的非线性传播，尤其关注计算机视觉应用。实验证实了所提出方法的实用性和对广泛应用的潜力。

    

    深度神经网络(DNN)的显著成功常常归因于它们的高表达能力和近似任意复杂函数的能力。事实上，DNN是高度非线性的模型，其中引入的激活函数在其中起到了重要作用。然而，尽管许多研究通过近似能力的视角研究了DNN的表达能力，但量化DNN或个别激活函数的非线性仍然是一个开放性问题。在本文中，我们提出了第一个在具体关注计算机视觉应用中追踪非线性传播的理论有效解决方案。我们提出的亲和度评分允许我们深入了解各种不同体系结构和学习范式的内部工作原理。我们提供了大量的实验结果，突出了所提出的亲和度评分的实际效用和潜在应用的可能性。

    The remarkable success of deep neural networks (DNN) is often attributed to their high expressive power and their ability to approximate functions of arbitrary complexity. Indeed, DNNs are highly non-linear models, and activation functions introduced into them are largely responsible for this. While many works studied the expressive power of DNNs through the lens of their approximation capabilities, quantifying the non-linearity of DNNs or of individual activation functions remains an open problem. In this paper, we propose the first theoretically sound solution to track non-linearity propagation in deep neural networks with a specific focus on computer vision applications. Our proposed affinity score allows us to gain insights into the inner workings of a wide range of different architectures and learning paradigms. We provide extensive experimental results that highlight the practical utility of the proposed affinity score and its potential for long-reaching applications.
    
[^4]: 加权平均随机梯度下降: 渐近正态性和最优性

    Weighted Averaged Stochastic Gradient Descent: Asymptotic Normality and Optimality. (arXiv:2307.06915v1 [stat.ML])

    [http://arxiv.org/abs/2307.06915](http://arxiv.org/abs/2307.06915)

    本文探索了一种加权平均随机梯度下降（SGD）方案，并建立了渐近正态性，提供了渐近有效的在线推理方法。此外，我们提出了一种自适应平均方案，具有最优的统计速度和有利的非渐近收敛性。

    

    随机梯度下降（SGD）是现代统计和机器学习中最简单和最流行的算法之一，由于其计算和内存效率而受到青睐。在不同的情境下，已经提出了各种平均方案来加速SGD的收敛。在本文中，我们探讨了一种用于SGD的通用平均方案。具体而言，我们建立了一类加权平均SGD解的渐近正态性，并提供了渐近有效的在线推理方法。此外，我们提出了一种自适应平均方案，展现出最优的统计速度和有利的非渐近收敛性，借鉴了线性模型的非渐近均方误差（MSE）的最优权重的见解。

    Stochastic Gradient Descent (SGD) is one of the simplest and most popular algorithms in modern statistical and machine learning due to its computational and memory efficiency. Various averaging schemes have been proposed to accelerate the convergence of SGD in different settings. In this paper, we explore a general averaging scheme for SGD. Specifically, we establish the asymptotic normality of a broad range of weighted averaged SGD solutions and provide asymptotically valid online inference approaches. Furthermore, we propose an adaptive averaging scheme that exhibits both optimal statistical rate and favorable non-asymptotic convergence, drawing insights from the optimal weight for the linear model in terms of non-asymptotic mean squared error (MSE).
    
[^5]: 用简洁可解释的加性模型和结构交互预测人口普查调查反应率

    Predicting Census Survey Response Rates With Parsimonious Additive Models and Structured Interactions. (arXiv:2108.11328v3 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2108.11328](http://arxiv.org/abs/2108.11328)

    本文提出了一种可解释的非参数加性模型，使用少量主要和成对交互效应预测调查反应率。该模型可以生成易于可视化和解释的预测面，并取得了 ROAM 数据集上的最先进性能，可以提供改进美国人口普查局和其他调查的反应率议论。

    

    本文考虑使用一系列灵活且可解释的非参数模型预测调查反应率。本研究受到美国人口普查局著名的 ROAM 应用的启发，该应用使用在美国人口普查规划数据库数据上训练的线性回归模型来识别难以调查的区域。十年前组织的一场众包竞赛表明，基于回归树集成的机器学习方法在预测调查反应率方面表现最佳；然而，由于它们的黑盒特性，相应的模型不能用于拟定的应用。我们考虑使用 $\ell_0$-based 惩罚的非参数加性模型，它具有少数主要和成对交互效应。从方法论的角度来看，我们研究了我们估计器的计算和统计方面，并讨论了将强层次交互合并的变体。我们的算法（在Github 上开源）允许我们生成易于可视化和解释的预测面，从而获得有关调查反应率的可行见解。我们提出的模型在 ROAM 数据集上实现了最先进的性能，并可以提供有关美国人口普查局和其他调查的改进调查反应率的见解。

    In this paper we consider the problem of predicting survey response rates using a family of flexible and interpretable nonparametric models. The study is motivated by the US Census Bureau's well-known ROAM application which uses a linear regression model trained on the US Census Planning Database data to identify hard-to-survey areas. A crowdsourcing competition organized around ten years ago revealed that machine learning methods based on ensembles of regression trees led to the best performance in predicting survey response rates; however, the corresponding models could not be adopted for the intended application due to their black-box nature. We consider nonparametric additive models with small number of main and pairwise interaction effects using $\ell_0$-based penalization. From a methodological viewpoint, we study both computational and statistical aspects of our estimator; and discuss variants that incorporate strong hierarchical interactions. Our algorithms (opensourced on gith
    

