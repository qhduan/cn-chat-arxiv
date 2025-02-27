# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Sampling-based Framework for Hypothesis Testing on Large Attributed Graphs](https://arxiv.org/abs/2403.13286) | 本论文提出了一个基于抽样的假设检验框架，能够在大属性图中处理节点、边和路径假设，通过提出路径假设感知采样器 PHASE 以及 PHASEopt，实现了准确且高效的抽样，实验证明了其在假设检验上的优势。 |
| [^2] | [Mathematical Introduction to Deep Learning: Methods, Implementations, and Theory.](http://arxiv.org/abs/2310.20360) | 本书提供了对深度学习算法的数学介绍，包括不同的神经网络架构和优化算法，并涵盖了深度学习算法的理论方面。此外，还介绍了深度学习逼近偏微分方程的方法。希望对学生和科学家们有所帮助。 |
| [^3] | [DCSI -- An improved measure of cluster separability based on separation and connectedness.](http://arxiv.org/abs/2310.12806) | 这篇论文提出了一种改进的聚类可分离性度量方法，旨在量化类间分离和类内连通性，对于密度聚类具有较好的性能表现。 |
| [^4] | [Confounding-Robust Policy Improvement with Human-AI Teams.](http://arxiv.org/abs/2310.08824) | 本文提出了一种通过采用边际灵敏度模型来解决人工智能与人类合作中未被观察到的混淆问题的新方法。该方法结合了领域专业知识和基于人工智能的统计建模，以控制潜在的混淆因素的影响，并通过推迟合作系统来利用不同决策者的专业知识。 |
| [^5] | [p$^3$VAE: a physics-integrated generative model. Application to the semantic segmentation of optical remote sensing images.](http://arxiv.org/abs/2210.10418) | 本文介绍了p$^3$VAE生成模型，它将一个完美的物理模型集成到模型中，并应用于高分辨率高光谱遥感图像的语义分割。模型具有更好的外推能力和可解释性，同时具有高度解缕能力。 |

# 详细

[^1]: 基于抽样的大属性图假设检验框架

    A Sampling-based Framework for Hypothesis Testing on Large Attributed Graphs

    [https://arxiv.org/abs/2403.13286](https://arxiv.org/abs/2403.13286)

    本论文提出了一个基于抽样的假设检验框架，能够在大属性图中处理节点、边和路径假设，通过提出路径假设感知采样器 PHASE 以及 PHASEopt，实现了准确且高效的抽样，实验证明了其在假设检验上的优势。

    

    假设检验是一种用于从样本数据中得出关于总体的结论的统计方法，通常用表格表示。随着现实应用中图表示的普及，图中的假设检验变得越来越重要。本文对属性图中的节点、边和路径假设进行了形式化。我们开发了一个基于抽样的假设检验框架，可以容纳现有的假设不可知的图抽样方法。为了实现准确和高效的抽样，我们提出了一种路径假设感知采样器 PHASE，它是一种考虑假设中指定路径的 m-维随机游走。我们进一步优化了其时间效率并提出了 PHASEopt。对真实数据集的实验表明，我们的框架能够利用常见的图抽样方法进行假设检验，并且在准确性和时间效率方面假设感知抽样具有优势。

    arXiv:2403.13286v1 Announce Type: cross  Abstract: Hypothesis testing is a statistical method used to draw conclusions about populations from sample data, typically represented in tables. With the prevalence of graph representations in real-life applications, hypothesis testing in graphs is gaining importance. In this work, we formalize node, edge, and path hypotheses in attributed graphs. We develop a sampling-based hypothesis testing framework, which can accommodate existing hypothesis-agnostic graph sampling methods. To achieve accurate and efficient sampling, we then propose a Path-Hypothesis-Aware SamplEr, PHASE, an m- dimensional random walk that accounts for the paths specified in a hypothesis. We further optimize its time efficiency and propose PHASEopt. Experiments on real datasets demonstrate the ability of our framework to leverage common graph sampling methods for hypothesis testing, and the superiority of hypothesis-aware sampling in terms of accuracy and time efficiency.
    
[^2]: 深度学习的数学介绍：方法、实现和理论

    Mathematical Introduction to Deep Learning: Methods, Implementations, and Theory. (arXiv:2310.20360v1 [cs.LG])

    [http://arxiv.org/abs/2310.20360](http://arxiv.org/abs/2310.20360)

    本书提供了对深度学习算法的数学介绍，包括不同的神经网络架构和优化算法，并涵盖了深度学习算法的理论方面。此外，还介绍了深度学习逼近偏微分方程的方法。希望对学生和科学家们有所帮助。

    

    本书旨在介绍深度学习算法的主题。我们详细介绍了深度学习算法的基本组成部分，包括不同的人工神经网络架构（如全连接前馈神经网络、卷积神经网络、循环神经网络、残差神经网络和带有批归一化的神经网络）以及不同的优化算法（如基本的随机梯度下降法、加速方法和自适应方法）。我们还涵盖了深度学习算法的几个理论方面，如人工神经网络的逼近能力（包括神经网络的微积分）、优化理论（包括Kurdyka-Lojasiewicz不等式）和泛化误差。在本书的最后一部分，我们还回顾了一些用于偏微分方程的深度学习逼近方法，包括物理信息神经网络（PINNs）和深度Galerkin方法。希望本书能对学生和科学家们有所帮助。

    This book aims to provide an introduction to the topic of deep learning algorithms. We review essential components of deep learning algorithms in full mathematical detail including different artificial neural network (ANN) architectures (such as fully-connected feedforward ANNs, convolutional ANNs, recurrent ANNs, residual ANNs, and ANNs with batch normalization) and different optimization algorithms (such as the basic stochastic gradient descent (SGD) method, accelerated methods, and adaptive methods). We also cover several theoretical aspects of deep learning algorithms such as approximation capacities of ANNs (including a calculus for ANNs), optimization theory (including Kurdyka-{\L}ojasiewicz inequalities), and generalization errors. In the last part of the book some deep learning approximation methods for PDEs are reviewed including physics-informed neural networks (PINNs) and deep Galerkin methods. We hope that this book will be useful for students and scientists who do not yet 
    
[^3]: DCSI -- 基于分离和连通性的改进的聚类可分离性度量

    DCSI -- An improved measure of cluster separability based on separation and connectedness. (arXiv:2310.12806v1 [stat.ML])

    [http://arxiv.org/abs/2310.12806](http://arxiv.org/abs/2310.12806)

    这篇论文提出了一种改进的聚类可分离性度量方法，旨在量化类间分离和类内连通性，对于密度聚类具有较好的性能表现。

    

    确定给定数据集中的类别标签是否对应于有意义的聚类对于使用真实数据集评估聚类算法至关重要。这个特性可以通过可分离性度量来量化。现有文献的综述显示，既有的基于分类的复杂性度量方法和聚类有效性指标 (CVIs) 都没有充分融入基于密度的聚类的核心特征：类间分离和类内连通性。一种新开发的度量方法 (密度聚类可分离性指数, DCSI) 旨在量化这两个特征，并且也可用作 CVI。对合成数据的广泛实验表明，DCSI 与通过调整兰德指数 (ARI) 测量的DBSCAN的性能之间有很强的相关性，但在对多类数据集进行密度聚类不适当的重叠类别时缺乏鲁棒性。对经常使用的真实数据集进行详细评估显示，DCSI 能够更好地区分密度聚类的可分离性。

    Whether class labels in a given data set correspond to meaningful clusters is crucial for the evaluation of clustering algorithms using real-world data sets. This property can be quantified by separability measures. A review of the existing literature shows that neither classification-based complexity measures nor cluster validity indices (CVIs) adequately incorporate the central aspects of separability for density-based clustering: between-class separation and within-class connectedness. A newly developed measure (density cluster separability index, DCSI) aims to quantify these two characteristics and can also be used as a CVI. Extensive experiments on synthetic data indicate that DCSI correlates strongly with the performance of DBSCAN measured via the adjusted rand index (ARI) but lacks robustness when it comes to multi-class data sets with overlapping classes that are ill-suited for density-based hard clustering. Detailed evaluation on frequently used real-world data sets shows that
    
[^4]: 带有人工智能团队的混淆鲁棒的策略改进

    Confounding-Robust Policy Improvement with Human-AI Teams. (arXiv:2310.08824v1 [cs.HC])

    [http://arxiv.org/abs/2310.08824](http://arxiv.org/abs/2310.08824)

    本文提出了一种通过采用边际灵敏度模型来解决人工智能与人类合作中未被观察到的混淆问题的新方法。该方法结合了领域专业知识和基于人工智能的统计建模，以控制潜在的混淆因素的影响，并通过推迟合作系统来利用不同决策者的专业知识。

    

    人工智能与人类的合作有可能通过充分发挥人类专家和人工智能系统的相互补充优势来改变各个领域。然而，未被观察到的混淆可能会破坏这种合作的有效性，导致偏见和不可靠的结果。本文提出了一种解决人工智能与人类合作中未被观察到的混淆问题的新方法，即采用边际灵敏度模型（MSM）。我们的方法将领域专业知识与基于人工智能的统计建模相结合，以考虑潜在的可能会隐藏的混淆因素。我们提出了一个推迟合作框架，将边际灵敏度模型纳入观测数据中的策略学习，使系统能够控制未被观察到的混淆因素的影响。此外，我们提出了一个个性化的推迟合作系统，以利用不同人类决策者的多样化专业知识。通过调整潜在的偏见，我们的解决方案能够提高合作结果的可靠性。

    Human-AI collaboration has the potential to transform various domains by leveraging the complementary strengths of human experts and Artificial Intelligence (AI) systems. However, unobserved confounding can undermine the effectiveness of this collaboration, leading to biased and unreliable outcomes. In this paper, we propose a novel solution to address unobserved confounding in human-AI collaboration by employing the marginal sensitivity model (MSM). Our approach combines domain expertise with AI-driven statistical modeling to account for potential confounders that may otherwise remain hidden. We present a deferral collaboration framework for incorporating the MSM into policy learning from observational data, enabling the system to control for the influence of unobserved confounding factors. In addition, we propose a personalized deferral collaboration system to leverage the diverse expertise of different human decision-makers. By adjusting for potential biases, our proposed solution e
    
[^5]: p$^3$VAE：一个物理集成的生成模型，应用于光学遥感图像的语义分割

    p$^3$VAE: a physics-integrated generative model. Application to the semantic segmentation of optical remote sensing images. (arXiv:2210.10418v3 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2210.10418](http://arxiv.org/abs/2210.10418)

    本文介绍了p$^3$VAE生成模型，它将一个完美的物理模型集成到模型中，并应用于高分辨率高光谱遥感图像的语义分割。模型具有更好的外推能力和可解释性，同时具有高度解缕能力。

    

    将机器学习模型与物理模型相结合是学习强大数据表示的最新研究方向。本文介绍了p$^3$VAE，这是一个生成模型，它集成了一个完美的物理模型，部分解释了数据中真实的变化因素。为了充分利用我们的混合设计，我们提出了一种半监督优化过程和一种推断方案，同时伴随着有意义的不确定性估计。我们将p$^3$VAE应用于高分辨率高光谱遥感图像的语义分割。我们在一个模拟数据集上的实验表明，与传统的机器学习模型相比，我们的混合模型具有更好的外推能力和可解释性。特别是，我们展示了p$^3$VAE自然具有高度解缕能力。我们的代码和数据已在https://github.com/Romain3Ch216/p3VAE上公开发布。

    The combination of machine learning models with physical models is a recent research path to learn robust data representations. In this paper, we introduce p$^3$VAE, a generative model that integrates a perfect physical model which partially explains the true underlying factors of variation in the data. To fully leverage our hybrid design, we propose a semi-supervised optimization procedure and an inference scheme that comes along meaningful uncertainty estimates. We apply p$^3$VAE to the semantic segmentation of high-resolution hyperspectral remote sensing images. Our experiments on a simulated data set demonstrated the benefits of our hybrid model against conventional machine learning models in terms of extrapolation capabilities and interpretability. In particular, we show that p$^3$VAE naturally has high disentanglement capabilities. Our code and data have been made publicly available at https://github.com/Romain3Ch216/p3VAE.
    

