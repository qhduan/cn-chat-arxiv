# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Conformal Predictive Programming for Chance Constrained Optimization](https://arxiv.org/abs/2402.07407) | 可容许预测规划（CPP）是一种解决受任意随机参数影响的优化问题的方法，通过利用样本和量子引理将机遇受限优化（CCO）问题转化为确定性优化问题，并具备边际概率可行性保证。 |
| [^2] | [Precise localization within the GI tract by combining classification of CNNs and time-series analysis of HMMs.](http://arxiv.org/abs/2310.07895) | 本文提出了一种通过使用CNN进行分类和HMM的时间序列分析，高效地将胃肠造影的图像进行分类，并通过连续的时间序列分析纠正CNN输出来实现精确定位。研究结果表明，该方法在Rhode Island胃肠病学数据集上达到了98.04％的准确率，可以在低功耗设备上使用。 |
| [^3] | [Efficient Partitioning Method of Large-Scale Public Safety Spatio-Temporal Data based on Information Loss Constraints.](http://arxiv.org/abs/2306.12857) | 本文提出了一种基于信息丢失约束的大规模公共安全时空数据高效划分方法(IFL-LSTP)，可以显著减小数据规模，同时保持模型的准确性，确保分布式存储的负载平衡，同时保持数据划分的时空接近性。 |
| [^4] | [Calibrated Stackelberg Games: Learning Optimal Commitments Against Calibrated Agents.](http://arxiv.org/abs/2306.02704) | 本文提出了一种新的校准史塔克伯格博弈（CSG）框架，其中智能体根据校准预测进行最佳响应。同时引入了自适应校准概念，提供精细的任何时候校准保证。在有限CSG中，主体可以获得最优解。 |

# 详细

[^1]: 可容许预测规划用于机遇受限优化

    Conformal Predictive Programming for Chance Constrained Optimization

    [https://arxiv.org/abs/2402.07407](https://arxiv.org/abs/2402.07407)

    可容许预测规划（CPP）是一种解决受任意随机参数影响的优化问题的方法，通过利用样本和量子引理将机遇受限优化（CCO）问题转化为确定性优化问题，并具备边际概率可行性保证。

    

    在对预测规划（CP）的进展的激励下，我们提出了可容许预测规划（CPP），一种解决机遇受限优化（CCO）问题的方法，即受任意随机参数影响的非线性约束函数的优化问题。CPP利用这些随机参数的样本以及量子引理（CP的核心）将CCO问题转化为确定性优化问题。然后，我们通过：（1）将量子表示为线性规划以及其KKT条件（CPP-KKT）；（2）使用混合整数规划（CPP-MIP）来呈现CPP的两种易于处理的改进。CPP具备对CCO问题进行边际概率可行性保证，这与现有方法（例如样本逼近和场景方法）在概念上有所不同。尽管我们探讨了与样本逼近方法的算法相似之处，但我们强调CPP的优势在于易于扩展。

    Motivated by the advances in conformal prediction (CP), we propose conformal predictive programming (CPP), an approach to solve chance constrained optimization (CCO) problems, i.e., optimization problems with nonlinear constraint functions affected by arbitrary random parameters. CPP utilizes samples from these random parameters along with the quantile lemma -- which is central to CP -- to transform the CCO problem into a deterministic optimization problem. We then present two tractable reformulations of CPP by: (1) writing the quantile as a linear program along with its KKT conditions (CPP-KKT), and (2) using mixed integer programming (CPP-MIP). CPP comes with marginal probabilistic feasibility guarantees for the CCO problem that are conceptually different from existing approaches, e.g., the sample approximation and the scenario approach. While we explore algorithmic similarities with the sample approximation approach, we emphasize that the strength of CPP is that it can easily be ext
    
[^2]: 通过结合CNN的分类和HMM的时间序列分析在GI道中进行精确定位的方法

    Precise localization within the GI tract by combining classification of CNNs and time-series analysis of HMMs. (arXiv:2310.07895v1 [cs.LG])

    [http://arxiv.org/abs/2310.07895](http://arxiv.org/abs/2310.07895)

    本文提出了一种通过使用CNN进行分类和HMM的时间序列分析，高效地将胃肠造影的图像进行分类，并通过连续的时间序列分析纠正CNN输出来实现精确定位。研究结果表明，该方法在Rhode Island胃肠病学数据集上达到了98.04％的准确率，可以在低功耗设备上使用。

    

    本文介绍了一种通过探索卷积神经网络（CNN）进行分类和隐马尔可夫模型（HMM）的时间序列分析属性的组合来高效地对视频胶囊内镜（VCE）研究中基于图像的胃肠学部分进行分类的方法。实验证明连续的时间序列分析可以识别和纠正CNN输出中的错误。我们的方法在Rhode Island（RI）胃肠病学数据集上实现了98.04％的准确率。这使得在胃肠道内实现精确定位成为可能，同时只需要约1M个参数，因此适用于低功耗设备。

    This paper presents a method to efficiently classify the gastroenterologic section of images derived from Video Capsule Endoscopy (VCE) studies by exploring the combination of a Convolutional Neural Network (CNN) for classification with the time-series analysis properties of a Hidden Markov Model (HMM). It is demonstrated that successive time-series analysis identifies and corrects errors in the CNN output. Our approach achieves an accuracy of $98.04\%$ on the Rhode Island (RI) Gastroenterology dataset. This allows for precise localization within the gastrointestinal (GI) tract while requiring only approximately 1M parameters and thus, provides a method suitable for low power devices
    
[^3]: 基于信息丢失约束的大规模公共安全时空数据高效划分方法

    Efficient Partitioning Method of Large-Scale Public Safety Spatio-Temporal Data based on Information Loss Constraints. (arXiv:2306.12857v1 [cs.LG])

    [http://arxiv.org/abs/2306.12857](http://arxiv.org/abs/2306.12857)

    本文提出了一种基于信息丢失约束的大规模公共安全时空数据高效划分方法(IFL-LSTP)，可以显著减小数据规模，同时保持模型的准确性，确保分布式存储的负载平衡，同时保持数据划分的时空接近性。

    

    大规模时空数据的存储、管理和应用在各种实际场景中广泛应用，包括公共安全。然而，由于现实世界数据的独特时空分布特征，大多数现有方法在数据时空接近度和分布式存储负载平衡方面存在限制。因此，本文提出了一种基于信息丢失约束的大规模公共安全时空数据高效划分方法(IFL-LSTP)。该IFL-LSTP模型针对大规模时空点数据，将时空划分模块(STPM)和图划分模块(GPM)相结合。该方法可以显著减小数据规模，同时保持模型的准确性，以提高划分效率。它还可以确保分布式存储的负载平衡，同时保持数据划分的时空接近性。

    The storage, management, and application of massive spatio-temporal data are widely applied in various practical scenarios, including public safety. However, due to the unique spatio-temporal distribution characteristics of re-al-world data, most existing methods have limitations in terms of the spatio-temporal proximity of data and load balancing in distributed storage. There-fore, this paper proposes an efficient partitioning method of large-scale public safety spatio-temporal data based on information loss constraints (IFL-LSTP). The IFL-LSTP model specifically targets large-scale spatio-temporal point da-ta by combining the spatio-temporal partitioning module (STPM) with the graph partitioning module (GPM). This approach can significantly reduce the scale of data while maintaining the model's accuracy, in order to improve the partitioning efficiency. It can also ensure the load balancing of distributed storage while maintaining spatio-temporal proximity of the data partitioning res
    
[^4]: 校准史塔克伯格博弈：学习对抗校准智能体的最优承诺

    Calibrated Stackelberg Games: Learning Optimal Commitments Against Calibrated Agents. (arXiv:2306.02704v1 [cs.GT] CROSS LISTED)

    [http://arxiv.org/abs/2306.02704](http://arxiv.org/abs/2306.02704)

    本文提出了一种新的校准史塔克伯格博弈（CSG）框架，其中智能体根据校准预测进行最佳响应。同时引入了自适应校准概念，提供精细的任何时候校准保证。在有限CSG中，主体可以获得最优解。

    

    本文提出了标准史塔克伯格博弈（SG）框架的一种推广：校准史塔克伯格博弈（CSG）。在CSG中，一个主体与一个智能体反复交互，后者不像标准SG一样直接访问主体的动作，而是对其进行校准预测，以达到最佳响应。CSG是一个强大的建模工具，超越了假定代理使用特定算法进行战略交互的做法，因此更加鲁棒地应对了SG最初旨在捕捉的现实应用。除了CSG外，本文还介绍了更强的校准概念，称为自适应校准，可针对敌对序列提供精细的任何时候校准保证。本文给出了获得自适应校准算法的一般方法，并将其专门用于有限CSG。在我们的主要技术结果中，我们证明在CSG中，主体可以获得收敛于最优解的效用。

    In this paper, we introduce a generalization of the standard Stackelberg Games (SGs) framework: Calibrated Stackelberg Games (CSGs). In CSGs, a principal repeatedly interacts with an agent who (contrary to standard SGs) does not have direct access to the principal's action but instead best-responds to calibrated forecasts about it. CSG is a powerful modeling tool that goes beyond assuming that agents use ad hoc and highly specified algorithms for interacting in strategic settings and thus more robustly addresses real-life applications that SGs were originally intended to capture. Along with CSGs, we also introduce a stronger notion of calibration, termed adaptive calibration, that provides fine-grained any-time calibration guarantees against adversarial sequences. We give a general approach for obtaining adaptive calibration algorithms and specialize them for finite CSGs. In our main technical result, we show that in CSGs, the principal can achieve utility that converges to the optimum
    

