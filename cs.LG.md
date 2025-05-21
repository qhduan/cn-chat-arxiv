# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Note on High-Probability Analysis of Algorithms with Exponential, Sub-Gaussian, and General Light Tails](https://arxiv.org/abs/2403.02873) | 这种技术可以简化分析依赖轻尾随机源的算法，通过对较简单的算法变体进行分析，避免使用专门的集中不等式，并且适用于指数、亚高斯和更一般的快速衰减分布。 |
| [^2] | [Towards Explaining Deep Neural Network Compression Through a Probabilistic Latent Space](https://arxiv.org/abs/2403.00155) | 通过概率潜在空间提出了一个新的理论框架，解释了深度神经网络压缩的优化网络稀疏度，并探讨了网络层的AP3/AP2属性与性能之间的关系。 |
| [^3] | [Self Supervised Correlation-based Permutations for Multi-View Clustering](https://arxiv.org/abs/2402.16383) | 提出了一种基于深度学习的多视图聚类框架，利用新颖的基于置换的规范相关性目标学习融合数据表示，并通过识别多个视图的一致伪标签来学习聚类分配，实验结果表明模型有效性，理论上证明逼近监督线性判别分析（LDA）表示，提供了由错误伪标签注释引起的误差界限。 |
| [^4] | [Conformal Monte Carlo Meta-learners for Predictive Inference of Individual Treatment Effects](https://arxiv.org/abs/2402.04906) | 本研究提出了一种新方法，即一致性蒙特卡洛元学习模型，用于预测个体治疗效果。通过利用一致性预测系统、蒙特卡洛采样和CATE元学习模型，该方法生成可用于个性化决策的预测分布。实验结果显示，该方法在保持较小区间宽度的情况下具有强大的实验覆盖范围，可以提供真实个体治疗效果的估计。 |
| [^5] | [Deep Learning for Multivariate Time Series Imputation: A Survey](https://arxiv.org/abs/2402.04059) | 本文调查了深度学习在多变量时间序列插补中的应用。通过综述不同的方法以及它们的优点和限制，研究了它们对下游任务性能的改进，并指出了未来研究的开放问题。 |
| [^6] | [Diversity-aware clustering: Computational Complexity and Approximation Algorithms.](http://arxiv.org/abs/2401.05502) | 本研究讨论了多样性感知聚类问题，在选择聚类中心时要考虑多个属性，同时最小化聚类目标。我们提出了针对不同聚类目标的参数化近似算法，这些算法在保证聚类质量的同时，具有紧确的近似比。 |
| [^7] | [Empathy Detection Using Machine Learning on Text, Audiovisual, Audio or Physiological Signals.](http://arxiv.org/abs/2311.00721) | 本论文对共情检测领域的机器学习研究进行了综述和分析，包括文本、视听、音频和生理信号四种输入模态的处理和网络设计，以及评估协议和数据集的描述。 |
| [^8] | [Performative Prediction: Past and Future.](http://arxiv.org/abs/2310.16608) | 表演性预测是机器学习中一个新兴领域，通过定义和研究预测对目标的影响，提供了对于分布变化和优化挑战的解决方法。 |
| [^9] | [Nonlinear Meta-Learning Can Guarantee Faster Rates.](http://arxiv.org/abs/2307.10870) | 非线性元学习可以保证更快的收敛速度。 |
| [^10] | [Gradient Leakage Defense with Key-Lock Module for Federated Learning.](http://arxiv.org/abs/2305.04095) | 本研究提出了一种新的联邦学习梯度泄露防御技术，使用私钥锁模块保护任意模型体系结构，并可确保无法从共享的梯度中重建私有训练数据。 |
| [^11] | [Towards Model-Agnostic Federated Learning over Networks.](http://arxiv.org/abs/2302.04363) | 该论文提出了一种适用于网络环境中多种数据和模型的模型无关关联学习方法，旨在通过网络结构反映本地数据集的相似性并保证本地模型产生一致的预测结果。 |
| [^12] | [Sequential Kernelized Independence Testing.](http://arxiv.org/abs/2212.07383) | 该论文介绍了顺序核独立性测试的方法，以解决传统批量测试在流数据上的问题，实现了根据任务复杂性自适应调整样本大小，并在收集新数据后持续监测和控制误报率。 |

# 详细

[^1]: 关于具有指数、亚高斯和一般轻尾的高概率分析算法的注解

    A Note on High-Probability Analysis of Algorithms with Exponential, Sub-Gaussian, and General Light Tails

    [https://arxiv.org/abs/2403.02873](https://arxiv.org/abs/2403.02873)

    这种技术可以简化分析依赖轻尾随机源的算法，通过对较简单的算法变体进行分析，避免使用专门的集中不等式，并且适用于指数、亚高斯和更一般的快速衰减分布。

    

    这篇简短的注解描述了一种分析概率算法的简单技术，该算法依赖于一个轻尾（但不一定有界）的随机化来源。我们展示了这样一个算法的分析可以通过黑盒方式减少，只在对数因子中有小量损失，转化为分析同一算法的一个更简单变体，该变体使用有界随机变量，通常更容易分析。这种方法同时适用于任何轻尾随机化，包括指数、亚高斯和更一般的快速衰减分布，而不需要调用专门的集中不等式。提供了对一般化Azuma不等式和具有一般轻尾噪声的随机优化的分析，以说明该技术。

    arXiv:2403.02873v1 Announce Type: new  Abstract: This short note describes a simple technique for analyzing probabilistic algorithms that rely on a light-tailed (but not necessarily bounded) source of randomization. We show that the analysis of such an algorithm can be reduced, in a black-box manner and with only a small loss in logarithmic factors, to an analysis of a simpler variant of the same algorithm that uses bounded random variables and often easier to analyze. This approach simultaneously applies to any light-tailed randomization, including exponential, sub-Gaussian, and more general fast-decaying distributions, without needing to appeal to specialized concentration inequalities. Analyses of a generalized Azuma inequality and stochastic optimization with general light-tailed noise are provided to illustrate the technique.
    
[^2]: 通过概率潜在空间解释深度神经网络压缩

    Towards Explaining Deep Neural Network Compression Through a Probabilistic Latent Space

    [https://arxiv.org/abs/2403.00155](https://arxiv.org/abs/2403.00155)

    通过概率潜在空间提出了一个新的理论框架，解释了深度神经网络压缩的优化网络稀疏度，并探讨了网络层的AP3/AP2属性与性能之间的关系。

    

    尽管深度神经网络（DNNs）表现出色，但它们的计算复杂性和存储空间消耗导致了网络压缩的概念。尽管已广泛研究了诸如修剪和低秩分解等DNN压缩技术，但对它们的理论解释仍未受到足够关注。本文提出了一个利用DNN权重的概率潜在空间并利用信息理论分歧度量解释最佳网络稀疏性的新理论框架。我们为DNN引入了新的类比投影模式（AP2）和概率中的类比投影模式（AP3）概念，并证明网络中层的AP3/AP2特性与其性能之间存在关系。此外，我们提供了一个理论分析，解释了压缩网络的训练过程。这些理论结果是从实证实验

    arXiv:2403.00155v1 Announce Type: new  Abstract: Despite the impressive performance of deep neural networks (DNNs), their computational complexity and storage space consumption have led to the concept of network compression. While DNN compression techniques such as pruning and low-rank decomposition have been extensively studied, there has been insufficient attention paid to their theoretical explanation. In this paper, we propose a novel theoretical framework that leverages a probabilistic latent space of DNN weights and explains the optimal network sparsity by using the information-theoretic divergence measures. We introduce new analogous projected patterns (AP2) and analogous-in-probability projected patterns (AP3) notions for DNNs and prove that there exists a relationship between AP3/AP2 property of layers in the network and its performance. Further, we provide a theoretical analysis that explains the training process of the compressed network. The theoretical results are empirica
    
[^3]: 自监督基于相关性的多视图聚类排序

    Self Supervised Correlation-based Permutations for Multi-View Clustering

    [https://arxiv.org/abs/2402.16383](https://arxiv.org/abs/2402.16383)

    提出了一种基于深度学习的多视图聚类框架，利用新颖的基于置换的规范相关性目标学习融合数据表示，并通过识别多个视图的一致伪标签来学习聚类分配，实验结果表明模型有效性，理论上证明逼近监督线性判别分析（LDA）表示，提供了由错误伪标签注释引起的误差界限。

    

    融合来自不同模态的信息可以增强数据分析任务，包括聚类。然而，现有的多视图聚类（MVC）解决方案仅限于特定领域，或者依赖于次优的且计算需求高的表示和聚类两阶段程序。我们提出了一个基于端到端深度学习的通用数据（图像、表格等）的MVC框架。我们的方法涉及使用基于新颖置换的规范相关性目标来学习有意义的融合数据表示。同时，我们通过识别跨多个视图的一致伪标签来学习聚类分配。我们使用十个MVC基准数据集展示了我们模型的有效性。在理论上，我们证明了我们的模型逼近了监督线性判别分析（LDA）表示。另外，我们提供了由错误伪标签注释引起的误差界限。

    arXiv:2402.16383v1 Announce Type: new  Abstract: Fusing information from different modalities can enhance data analysis tasks, including clustering. However, existing multi-view clustering (MVC) solutions are limited to specific domains or rely on a suboptimal and computationally demanding two-stage procedure of representation and clustering. We propose an end-to-end deep learning-based MVC framework for general data (image, tabular, etc.). Our approach involves learning meaningful fused data representations with a novel permutation-based canonical correlation objective. Concurrently, we learn cluster assignments by identifying consistent pseudo-labels across multiple views. We demonstrate the effectiveness of our model using ten MVC benchmark datasets. Theoretically, we show that our model approximates the supervised linear discrimination analysis (LDA) representation. Additionally, we provide an error bound induced by false-pseudo label annotations.
    
[^4]: 预测个体治疗效果的一致性蒙特卡洛元学习模型

    Conformal Monte Carlo Meta-learners for Predictive Inference of Individual Treatment Effects

    [https://arxiv.org/abs/2402.04906](https://arxiv.org/abs/2402.04906)

    本研究提出了一种新方法，即一致性蒙特卡洛元学习模型，用于预测个体治疗效果。通过利用一致性预测系统、蒙特卡洛采样和CATE元学习模型，该方法生成可用于个性化决策的预测分布。实验结果显示，该方法在保持较小区间宽度的情况下具有强大的实验覆盖范围，可以提供真实个体治疗效果的估计。

    

    认识干预效果，即治疗效果，对于决策至关重要。用条件平均治疗效果 (CATE) 估计等方法通常只提供治疗效果的点估计，而常常需要额外的不确定性量化。因此，我们提出了一个新方法，即一致性蒙特卡洛 (CMC) 元学习模型，利用一致性预测系统、蒙特卡洛采样和 CATE 元学习模型，来产生可用于个性化决策的预测分布。此外，我们展示了结果噪声分布的特定假设如何严重影响这些不确定性预测。尽管如此，CMC框架展示了强大的实验覆盖范围，同时保持较小的区间宽度，以提供真实个体治疗效果的估计。

    Knowledge of the effect of interventions, called the treatment effect, is paramount for decision-making. Approaches to estimating this treatment effect, e.g. by using Conditional Average Treatment Effect (CATE) estimators, often only provide a point estimate of this treatment effect, while additional uncertainty quantification is frequently desired instead. Therefore, we present a novel method, the Conformal Monte Carlo (CMC) meta-learners, leveraging conformal predictive systems, Monte Carlo sampling, and CATE meta-learners, to instead produce a predictive distribution usable in individualized decision-making. Furthermore, we show how specific assumptions on the noise distribution of the outcome heavily affect these uncertainty predictions. Nonetheless, the CMC framework shows strong experimental coverage while retaining small interval widths to provide estimates of the true individual treatment effect.
    
[^5]: 深度学习在多变量时间序列插补中的应用：一项调查

    Deep Learning for Multivariate Time Series Imputation: A Survey

    [https://arxiv.org/abs/2402.04059](https://arxiv.org/abs/2402.04059)

    本文调查了深度学习在多变量时间序列插补中的应用。通过综述不同的方法以及它们的优点和限制，研究了它们对下游任务性能的改进，并指出了未来研究的开放问题。

    

    普遍存在的缺失值导致多变量时间序列数据部分观测，破坏了时间序列的完整性，阻碍了有效的时间序列数据分析。最近，深度学习插补方法在提高损坏的时间序列数据质量方面取得了显著的成功，进而提高了下游任务的性能。本文对最近提出的深度学习插补方法进行了全面的调查。首先，我们提出了对这些方法进行分类的方法，并通过强调它们的优点和限制来进行了结构化的综述。我们还进行了实证实验，研究了不同方法，并比较了它们对下游任务的改进。最后，我们指出了多变量时间序列插补未来研究的开放问题。本文的所有代码和配置，包括定期维护的多变量时间序列插补论文列表，可以在以下位置找到。

    The ubiquitous missing values cause the multivariate time series data to be partially observed, destroying the integrity of time series and hindering the effective time series data analysis. Recently deep learning imputation methods have demonstrated remarkable success in elevating the quality of corrupted time series data, subsequently enhancing performance in downstream tasks. In this paper, we conduct a comprehensive survey on the recently proposed deep learning imputation methods. First, we propose a taxonomy for the reviewed methods, and then provide a structured review of these methods by highlighting their strengths and limitations. We also conduct empirical experiments to study different methods and compare their enhancement for downstream tasks. Finally, the open issues for future research on multivariate time series imputation are pointed out. All code and configurations of this work, including a regularly maintained multivariate time series imputation paper list, can be foun
    
[^6]: 多样性感知聚类：计算复杂性和近似算法

    Diversity-aware clustering: Computational Complexity and Approximation Algorithms. (arXiv:2401.05502v1 [cs.DS])

    [http://arxiv.org/abs/2401.05502](http://arxiv.org/abs/2401.05502)

    本研究讨论了多样性感知聚类问题，在选择聚类中心时要考虑多个属性，同时最小化聚类目标。我们提出了针对不同聚类目标的参数化近似算法，这些算法在保证聚类质量的同时，具有紧确的近似比。

    

    在这项工作中，我们研究了多样性感知聚类问题，其中数据点与多个属性相关联，形成交叉的组。聚类解决方案需要确保从每个组中选择最少数量的聚类中心，同时最小化聚类目标，可以是$k$-中位数，$k$-均值或$k$-供应商。我们提出了参数化近似算法，近似比分别为$1+\frac{2}{e}$，$1+\frac{8}{e}$和$3$，用于多样性感知$k$-中位数，多样性感知$k$-均值和多样性感知$k$-供应商。这些近似比在假设Gap-ETH和FPT $\neq$ W[2]的情况下是紧确的。对于公平$k$-中位数和公平$k$-均值的不相交工厂组，我们提出了参数化近似算法，近似比分别为$1+\frac{2}{e}$和$1+\frac{8}{e}$。对于具有不相交工厂组的公平$k$-供应商，我们提出了一个多项式时间近似算法，因子为$3$。

    In this work, we study diversity-aware clustering problems where the data points are associated with multiple attributes resulting in intersecting groups. A clustering solution need to ensure that a minimum number of cluster centers are chosen from each group while simultaneously minimizing the clustering objective, which can be either $k$-median, $k$-means or $k$-supplier. We present parameterized approximation algorithms with approximation ratios $1+ \frac{2}{e}$, $1+\frac{8}{e}$ and $3$ for diversity-aware $k$-median, diversity-aware $k$-means and diversity-aware $k$-supplier, respectively. The approximation ratios are tight assuming Gap-ETH and FPT $\neq$ W[2]. For fair $k$-median and fair $k$-means with disjoint faicility groups, we present parameterized approximation algorithm with approximation ratios $1+\frac{2}{e}$ and $1+\frac{8}{e}$, respectively. For fair $k$-supplier with disjoint facility groups, we present a polynomial-time approximation algorithm with factor $3$, improv
    
[^7]: 使用机器学习在文本、视听、音频或生理信号上进行共情检测

    Empathy Detection Using Machine Learning on Text, Audiovisual, Audio or Physiological Signals. (arXiv:2311.00721v1 [cs.HC])

    [http://arxiv.org/abs/2311.00721](http://arxiv.org/abs/2311.00721)

    本论文对共情检测领域的机器学习研究进行了综述和分析，包括文本、视听、音频和生理信号四种输入模态的处理和网络设计，以及评估协议和数据集的描述。

    

    共情是一个社交技能，表明一个个体理解他人的能力。近年来，共情引起了包括情感计算、认知科学和心理学在内的各个学科的关注。共情是一个依赖于上下文的术语，因此检测或识别共情在社会、医疗和教育等领域具有潜在的应用。尽管共情检测领域涉及范围广泛且有重叠，但从整体文献角度来看，利用机器学习的共情检测研究仍然相对较少。为此，我们系统收集和筛选了来自10个知名数据库的801篇论文，并分析了选定的54篇论文。我们根据共情检测系统的输入模态，即文本、视听、音频和生理信号，对论文进行分组。我们分别研究了特定模态的预处理和网络架构设计协议、常见数据集的描述和可用性详情，以及评估协议。

    Empathy is a social skill that indicates an individual's ability to understand others. Over the past few years, empathy has drawn attention from various disciplines, including but not limited to Affective Computing, Cognitive Science and Psychology. Empathy is a context-dependent term; thus, detecting or recognising empathy has potential applications in society, healthcare and education. Despite being a broad and overlapping topic, the avenue of empathy detection studies leveraging Machine Learning remains underexplored from a holistic literature perspective. To this end, we systematically collect and screen 801 papers from 10 well-known databases and analyse the selected 54 papers. We group the papers based on input modalities of empathy detection systems, i.e., text, audiovisual, audio and physiological signals. We examine modality-specific pre-processing and network architecture design protocols, popular dataset descriptions and availability details, and evaluation protocols. We fur
    
[^8]: 表演性预测：过去与未来

    Performative Prediction: Past and Future. (arXiv:2310.16608v1 [cs.LG])

    [http://arxiv.org/abs/2310.16608](http://arxiv.org/abs/2310.16608)

    表演性预测是机器学习中一个新兴领域，通过定义和研究预测对目标的影响，提供了对于分布变化和优化挑战的解决方法。

    

    在社会世界中，预测通常会影响预测的目标，这一现象被称为表演性。自我实现和自我否定的预测是表演性的例子。对经济学、金融学和社会科学至关重要的概念在机器学习的发展中一直缺失。在机器学习应用中，表演性通常表现为分布变化。例如，在数字平台上部署的预测模型会影响消费，从而改变数据生成的分布。我们调查了最近成立的表演性预测领域，该领域提供了一个定义和概念框架来研究机器学习中的表演性。表演性预测的一个结果是自然均衡概念的产生，从而产生了新的优化挑战。另一个结果是学习和操控之间的区别，这是表演性预测中的两种机制。

    Predictions in the social world generally influence the target of prediction, a phenomenon known as performativity. Self-fulfilling and self-negating predictions are examples of performativity. Of fundamental importance to economics, finance, and the social sciences, the notion has been absent from the development of machine learning. In machine learning applications, performativity often surfaces as distribution shift. A predictive model deployed on a digital platform, for example, influences consumption and thereby changes the data-generating distribution. We survey the recently founded area of performative prediction that provides a definition and conceptual framework to study performativity in machine learning. A consequence of performative prediction is a natural equilibrium notion that gives rise to new optimization challenges. Another consequence is a distinction between learning and steering, two mechanisms at play in performative prediction. The notion of steering is in turn i
    
[^9]: 非线性元学习可以保证更快的收敛速度

    Nonlinear Meta-Learning Can Guarantee Faster Rates. (arXiv:2307.10870v1 [stat.ML])

    [http://arxiv.org/abs/2307.10870](http://arxiv.org/abs/2307.10870)

    非线性元学习可以保证更快的收敛速度。

    

    最近许多关于元学习的理论研究旨在利用相关任务中的相似表示结构来简化目标任务，并实现收敛速率的保证。然而，在实践中，表示往往是高度非线性的，引入了每个任务中不可简单平均的非平凡偏差。本研究通过非线性表示推导出元学习的理论保证。

    Many recent theoretical works on \emph{meta-learning} aim to achieve guarantees in leveraging similar representational structures from related tasks towards simplifying a target task. Importantly, the main aim in theory works on the subject is to understand the extent to which convergence rates -- in learning a common representation -- \emph{may scale with the number $N$ of tasks} (as well as the number of samples per task). First steps in this setting demonstrate this property when both the shared representation amongst tasks, and task-specific regression functions, are linear. This linear setting readily reveals the benefits of aggregating tasks, e.g., via averaging arguments. In practice, however, the representation is often highly nonlinear, introducing nontrivial biases in each task that cannot easily be averaged out as in the linear case. In the present work, we derive theoretical guarantees for meta-learning with nonlinear representations. In particular, assuming the shared nonl
    
[^10]: 基于密钥锁模块的联邦学习梯度泄露防御

    Gradient Leakage Defense with Key-Lock Module for Federated Learning. (arXiv:2305.04095v1 [cs.LG])

    [http://arxiv.org/abs/2305.04095](http://arxiv.org/abs/2305.04095)

    本研究提出了一种新的联邦学习梯度泄露防御技术，使用私钥锁模块保护任意模型体系结构，并可确保无法从共享的梯度中重建私有训练数据。

    

    联邦学习是一种广泛采用的隐私保护机器学习方法，其中私有数据保持本地，允许安全计算和本地模型梯度与第三方参数服务器之间的交换。然而，最近的研究发现，通过共享的梯度可能会危及隐私并恢复敏感信息。本研究提供了详细的分析和对梯度泄漏问题的新视角。这些理论工作导致了一种新的梯度泄露防御技术，使用私钥锁模块保护任意模型体系结构。只有锁定的梯度被传输到参数服务器进行全局模型聚合。我们提出的学习方法对梯度泄露攻击具有抵抗力，并且所设计和训练的密钥锁模块可以确保，没有密钥锁模块的私有信息：a) 无法从共享的梯度中重建私有训练数据。

    Federated Learning (FL) is a widely adopted privacy-preserving machine learning approach where private data remains local, enabling secure computations and the exchange of local model gradients between local clients and third-party parameter servers. However, recent findings reveal that privacy may be compromised and sensitive information potentially recovered from shared gradients. In this study, we offer detailed analysis and a novel perspective on understanding the gradient leakage problem. These theoretical works lead to a new gradient leakage defense technique that secures arbitrary model architectures using a private key-lock module. Only the locked gradient is transmitted to the parameter server for global model aggregation. Our proposed learning method is resistant to gradient leakage attacks, and the key-lock module is designed and trained to ensure that, without the private information of the key-lock module: a) reconstructing private training data from the shared gradient is
    
[^11]: 探索网络上的模型无关联合学习

    Towards Model-Agnostic Federated Learning over Networks. (arXiv:2302.04363v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.04363](http://arxiv.org/abs/2302.04363)

    该论文提出了一种适用于网络环境中多种数据和模型的模型无关关联学习方法，旨在通过网络结构反映本地数据集的相似性并保证本地模型产生一致的预测结果。

    

    我们提出了一种适用于异构数据和模型网络的模型无关联合学习方法。网络结构反映了本地数据集（统计数据）和它们相关的本地模型之间的相似性。我们的方法是经验风险最小化的一种实例，其中正则化项是从数据的网络结构导出的。特别地，我们要求良好连接的本地模型形成聚类，在一个公共测试集上产生相似的预测结果。所提出的方法允许使用各种各样的本地模型。 对这些本地模型唯一的限制是它们允许有效实现正则化的经验风险最小化（训练）。对于各种模型，这样的实现都可以在高级编程库（包括scikit-learn、Keras或PyTorch）中找到。

    We present a model-agnostic federated learning method for networks of heterogeneous data and models. The network structure reflects similarities between the (statistics of) local datasets and, in turn, their associated local("personal") models. Our method is an instance of empirical risk minimization, with the regularization term derived from the network structure of data. In particular, we require well-connected local models, forming clusters, to yield similar predictions on a common test set. The proposed method allows for a wide range of local models. The only restriction on these local models is that they allow for efficient implementation of regularized empirical risk minimization (training). For a wide range of models, such implementations are available in high-level programming libraries including scikit-learn, Keras or PyTorch.
    
[^12]: 顺序核独立性测试

    Sequential Kernelized Independence Testing. (arXiv:2212.07383v3 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2212.07383](http://arxiv.org/abs/2212.07383)

    该论文介绍了顺序核独立性测试的方法，以解决传统批量测试在流数据上的问题，实现了根据任务复杂性自适应调整样本大小，并在收集新数据后持续监测和控制误报率。

    

    独立性测试是一个经典的统计问题，在固定采集数据之前的批量设置中得到了广泛研究。然而，实践者们往往更喜欢能够根据问题的复杂性进行自适应的程序，而不是事先设定样本大小。理想情况下，这样的程序应该（a）在简单任务上尽早停止（在困难任务上稍后停止），因此更好地利用可用资源，以及（b）在收集新数据之后，持续监测数据并高效地整合统计证据，同时控制误报率。经典的批量测试不适用于流数据：在数据观察后进行有效推断需要对多重测试进行校正，这导致了低功率。遵循通过投注进行测试的原则，我们设计了顺序核独立性测试，克服了这些缺点。我们通过采用由核相关性测度（如Hilbert-）启发的投注来说明我们的广泛框架。

    Independence testing is a classical statistical problem that has been extensively studied in the batch setting when one fixes the sample size before collecting data. However, practitioners often prefer procedures that adapt to the complexity of a problem at hand instead of setting sample size in advance. Ideally, such procedures should (a) stop earlier on easy tasks (and later on harder tasks), hence making better use of available resources, and (b) continuously monitor the data and efficiently incorporate statistical evidence after collecting new data, while controlling the false alarm rate. Classical batch tests are not tailored for streaming data: valid inference after data peeking requires correcting for multiple testing which results in low power. Following the principle of testing by betting, we design sequential kernelized independence tests that overcome such shortcomings. We exemplify our broad framework using bets inspired by kernelized dependence measures, e.g., the Hilbert-
    

