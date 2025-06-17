# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Sample complexity of quantum hypothesis testing](https://arxiv.org/abs/2403.17868) | 本文研究了量子假设检验的样本复杂度，得出了对称和非对称设置中的二进制量子假设检验的样本复杂度与反错误概率的对数和保真度的负对数的关系。 |
| [^2] | [Unifying Lane-Level Traffic Prediction from a Graph Structural Perspective: Benchmark and Baseline](https://arxiv.org/abs/2403.14941) | 本文提出了一个简单的基线模型GraphMLP，基于图结构和MLP网络，在车道级交通预测中建立了统一的空间拓扑结构和预测任务，帮助突破了现有评估标准和数据公开性的限制。 |
| [^3] | [Inference for an Algorithmic Fairness-Accuracy Frontier](https://arxiv.org/abs/2402.08879) | 本文提供了一个算法公平性和准确性推理的方法。我们提出了一种一致的估计器，并进行了一些检验假设的推理。同时，我们还给出了一个估计器来计算一个给定算法与前沿上最公平点之间的距离，并描述了它的渐近性质。 |
| [^4] | [On the Completeness of Invariant Geometric Deep Learning Models](https://arxiv.org/abs/2402.04836) | 这项研究集中于不变模型的理论表达能力，通过引入完备的设计GeoNGNN，并利用其作为理论工具，首次证明了E(3)-完备性。 |
| [^5] | [Zeroth-Order primal-dual Alternating Projection Gradient Algorithms for Nonconvex Minimax Problems with Coupled linear Constraints](https://arxiv.org/abs/2402.03352) | 本文研究了具有耦合线性约束的非凸极小极大问题的零阶算法，提出了两个单循环算法用于求解这些问题，并证明了它们的迭代复杂度分别为O(ε^(-2))和O(ε^(-4))。 |
| [^6] | [Statistics without Interpretation: A Sober Look at Explainable Machine Learning](https://arxiv.org/abs/2402.02870) | 解释算法往往数学上复杂且难以解释，这导致解释错误。为了向前推进，解释算法需要明确其输出的解释方式，并澄清可以和不能回答的问题。这一论点基于统计学和解释之间的区别，以及可解释机器学习和应用统计学之间的相似性。 |
| [^7] | [Computing the Distance between unbalanced Distributions -- The flat Metric.](http://arxiv.org/abs/2308.01039) | 该论文提出了一种计算不平衡分布之间距离的方法，基于平坦度量，该方法可以推广到分布总质量不平衡的情况下，特别适用于不平衡最优输运和数据分布分析。论文实现了一个基于神经网络的方法，可以计算出两个给定测度之间距离的最佳测试函数，并通过多个实验验证了方法的准确性。 |
| [^8] | [How Sparse Can We Prune A Deep Network: A Geometric Viewpoint.](http://arxiv.org/abs/2306.05857) | 本文从高维几何的角度，通过在原始损失函数中强制施加稀疏性约束，描述了深度网络剪枝比率的相变点，该点等于某些凸体的平方高斯宽度除以参数的原始维度。 |
| [^9] | [CLCIFAR: CIFAR-Derived Benchmark Datasets with Human Annotated Complementary Labels.](http://arxiv.org/abs/2305.08295) | 本研究开发了由人类标注的互补标签，创造了两个真实世界的CLL数据集，进一步揭示了现实表现下CLL算法的性能，为这一领域的研究提供了更实际的评估标准。 |
| [^10] | [Geometry-Aware Latent Representation Learning for Modeling Disease Progression of Barrett's Esophagus.](http://arxiv.org/abs/2303.12711) | 本文提出了一种基于几何思想的潜在表示学习方法，用于建模Barrett食管疾病进程，与传统方法相比，具有更好的重建损失。 |
| [^11] | [Online Learning for Equilibrium Pricing in Markets under Incomplete Information.](http://arxiv.org/abs/2303.11522) | 该论文研究了在不完全信息的市场中使用在线学习进行平衡定价的问题，提出了解决选择性谎言问题的新方法。 |
| [^12] | [Compressing Transformer-based self-supervised models for speech processing.](http://arxiv.org/abs/2211.09949) | 本文研究了对基于Transformer的自监督模型进行压缩的方法，包括权重修剪、头部修剪、低秩逼近和知识蒸馏。结果发现，基本的压缩技术是强大的基准，可以改善模型的压缩效果。 |
| [^13] | [Conformal Risk Control.](http://arxiv.org/abs/2208.02814) | 该论文提出了一种符合保序的风险控制方法，可以控制任何单调损失函数的期望值，示例证明其在计算机视觉和自然语言处理领域具有控制误报率、图形距离和令牌级F1得分的能力。 |

# 详细

[^1]: 量子假设检验的样本复杂度

    Sample complexity of quantum hypothesis testing

    [https://arxiv.org/abs/2403.17868](https://arxiv.org/abs/2403.17868)

    本文研究了量子假设检验的样本复杂度，得出了对称和非对称设置中的二进制量子假设检验的样本复杂度与反错误概率的对数和保真度的负对数的关系。

    

    传统上，人们从信息论的角度研究量子假设检验，在这种情况下，人们对错误概率的最优衰减速率感兴趣，这个速率是未知状态的样本数量函数。本文研究了量子假设检验的样本复杂度，旨在确定达到所需错误概率所需的最少样本数量。通过利用已有文献中关于量子假设检验的丰富知识，我们表征了对称和非对称设置中的二进制量子假设检验的样本复杂度，并提供了多个量子假设检验的样本复杂度的界限。更详细地说，我们证明了对称二进制量子假设检验的样本复杂度对反错误概率的对数和保真度的负对数的对数。

    arXiv:2403.17868v1 Announce Type: cross  Abstract: Quantum hypothesis testing has been traditionally studied from the information-theoretic perspective, wherein one is interested in the optimal decay rate of error probabilities as a function of the number of samples of an unknown state. In this paper, we study the sample complexity of quantum hypothesis testing, wherein the goal is to determine the minimum number of samples needed to reach a desired error probability. By making use of the wealth of knowledge that already exists in the literature on quantum hypothesis testing, we characterize the sample complexity of binary quantum hypothesis testing in the symmetric and asymmetric settings, and we provide bounds on the sample complexity of multiple quantum hypothesis testing. In more detail, we prove that the sample complexity of symmetric binary quantum hypothesis testing depends logarithmically on the inverse error probability and inversely on the negative logarithm of the fidelity. 
    
[^2]: 从图结构角度统一车道级交通预测：基准和基线

    Unifying Lane-Level Traffic Prediction from a Graph Structural Perspective: Benchmark and Baseline

    [https://arxiv.org/abs/2403.14941](https://arxiv.org/abs/2403.14941)

    本文提出了一个简单的基线模型GraphMLP，基于图结构和MLP网络，在车道级交通预测中建立了统一的空间拓扑结构和预测任务，帮助突破了现有评估标准和数据公开性的限制。

    

    交通预测长期以来一直是研究中的一个焦点和关键领域，在过去几年里，既见证了从城市级到道路级预测取得的重大进展。随着车辆对一切（V2X）技术、自动驾驶和交通领域的大规模模型的进步，道路级交通预测已经成为一个不可或缺的方向。然而，这一领域的进一步进展受到了全面和统一的评估标准的缺乏以及有限的公开数据和代码的阻碍。本文对车道级交通预测中现有研究进行了广泛的分析和分类，建立了统一的空间拓扑结构和预测任务，并介绍了一个基于图结构和MLP网络的简单基线模型GraphMLP。我们复制了现有研究中尚不公开的代码，并基于此充分而公正地评估了各种模型。

    arXiv:2403.14941v1 Announce Type: cross  Abstract: Traffic prediction has long been a focal and pivotal area in research, witnessing both significant strides from city-level to road-level predictions in recent years. With the advancement of Vehicle-to-Everything (V2X) technologies, autonomous driving, and large-scale models in the traffic domain, lane-level traffic prediction has emerged as an indispensable direction. However, further progress in this field is hindered by the absence of comprehensive and unified evaluation standards, coupled with limited public availability of data and code. This paper extensively analyzes and categorizes existing research in lane-level traffic prediction, establishes a unified spatial topology structure and prediction tasks, and introduces a simple baseline model, GraphMLP, based on graph structure and MLP networks. We have replicated codes not publicly available in existing studies and, based on this, thoroughly and fairly assessed various models in 
    
[^3]: 一个算法公平性和准确性的推理

    Inference for an Algorithmic Fairness-Accuracy Frontier

    [https://arxiv.org/abs/2402.08879](https://arxiv.org/abs/2402.08879)

    本文提供了一个算法公平性和准确性推理的方法。我们提出了一种一致的估计器，并进行了一些检验假设的推理。同时，我们还给出了一个估计器来计算一个给定算法与前沿上最公平点之间的距离，并描述了它的渐近性质。

    

    决策过程越来越依赖于算法的使用。然而，算法的预测能力在人口的不同子群体中经常出现系统性变化。虽然公平性和准确性都是算法的期望特性，但它们常常是相互牺牲的。那么，当面对有限的数据时，一个注重公平性的决策者应该怎么做呢?在本文中，我们为Liang，Lu和Mu（2023）提出的一个理论公平性-准确性前沿提供了一致的估计器，并提出了检验假设的推理方法。这些假设在公平性文献中引起了很多关注，例如(i)全面排除在算法训练中使用一个协变量是否是最优的，(ii)是否存在对现有算法更少歧视性的替代方案。我们还为给定算法与前沿上最公平点之间的距离提供了一个估计器，并描述了它的渐近性质。

    arXiv:2402.08879v1 Announce Type: cross Abstract: Decision-making processes increasingly rely on the use of algorithms. Yet, algorithms' predictive ability frequently exhibit systematic variation across subgroups of the population. While both fairness and accuracy are desirable properties of an algorithm, they often come at the cost of one another. What should a fairness-minded policymaker do then, when confronted with finite data? In this paper, we provide a consistent estimator for a theoretical fairness-accuracy frontier put forward by Liang, Lu and Mu (2023) and propose inference methods to test hypotheses that have received much attention in the fairness literature, such as (i) whether fully excluding a covariate from use in training the algorithm is optimal and (ii) whether there are less discriminatory alternatives to an existing algorithm. We also provide an estimator for the distance between a given algorithm and the fairest point on the frontier, and characterize its asymptot
    
[^4]: 关于不变几何深度学习模型的完备性

    On the Completeness of Invariant Geometric Deep Learning Models

    [https://arxiv.org/abs/2402.04836](https://arxiv.org/abs/2402.04836)

    这项研究集中于不变模型的理论表达能力，通过引入完备的设计GeoNGNN，并利用其作为理论工具，首次证明了E(3)-完备性。

    

    不变模型是一类重要的几何深度学习模型，通过利用信息丰富的几何特征生成有意义的几何表示。这些模型以其简单性、良好的实验结果和计算效率而闻名。然而，它们的理论表达能力仍然不清楚，限制了对这种模型潜力的深入理解。在这项工作中，我们集中讨论不变模型的理论表达能力。我们首先严格限制了最经典的不变模型Vanilla DisGNN（结合距离的消息传递神经网络）的表达能力，将其不可识别的情况仅限于高度对称的几何图形。为了打破这些特殊情况的对称性，我们引入了一个简单而完备的不变设计，即嵌套Vanilla DisGNN的GeoNGNN。利用GeoNGNN作为理论工具，我们首次证明了E(3)-完备性。

    Invariant models, one important class of geometric deep learning models, are capable of generating meaningful geometric representations by leveraging informative geometric features. These models are characterized by their simplicity, good experimental results and computational efficiency. However, their theoretical expressive power still remains unclear, restricting a deeper understanding of the potential of such models. In this work, we concentrate on characterizing the theoretical expressiveness of invariant models. We first rigorously bound the expressiveness of the most classical invariant model, Vanilla DisGNN (message passing neural networks incorporating distance), restricting its unidentifiable cases to be only those highly symmetric geometric graphs. To break these corner cases' symmetry, we introduce a simple yet E(3)-complete invariant design by nesting Vanilla DisGNN, named GeoNGNN. Leveraging GeoNGNN as a theoretical tool, we for the first time prove the E(3)-completeness 
    
[^5]: 面向具有耦合线性约束的非凸极小极大问题的零阶原始对偶交替投影梯度算法

    Zeroth-Order primal-dual Alternating Projection Gradient Algorithms for Nonconvex Minimax Problems with Coupled linear Constraints

    [https://arxiv.org/abs/2402.03352](https://arxiv.org/abs/2402.03352)

    本文研究了具有耦合线性约束的非凸极小极大问题的零阶算法，提出了两个单循环算法用于求解这些问题，并证明了它们的迭代复杂度分别为O(ε^(-2))和O(ε^(-4))。

    

    本文研究了确定性和随机设置下具有耦合线性约束的非凸极小极大问题的零阶算法，这在机器学习、信号处理和其他领域中近年来引起了广泛关注，例如资源分配问题和网络流问题中的对抗攻击等。我们提出了两个单循环算法，分别是零阶原始对偶交替投影梯度（ZO-PDAPG）算法和零阶正则动量原始对偶投影梯度算法（ZO-RMPDPG），用于解决具有耦合线性约束的确定性和随机非凸-(强)凹极小极大问题。证明了这两个算法获得一个ε-稳定点的迭代复杂度分别为O(ε^(-2))（对于求解非凸-凹极小极大问题）和O(ε^(-4))（对于求解非凸-凹极小极大问题）。

    In this paper, we study zeroth-order algorithms for nonconvex minimax problems with coupled linear constraints under the deterministic and stochastic settings, which have attracted wide attention in machine learning, signal processing and many other fields in recent years, e.g., adversarial attacks in resource allocation problems and network flow problems etc. We propose two single-loop algorithms, namely the zero-order primal-dual alternating projected gradient (ZO-PDAPG) algorithm and the zero-order regularized momentum primal-dual projected gradient algorithm (ZO-RMPDPG), for solving deterministic and stochastic nonconvex-(strongly) concave minimax problems with coupled linear constraints. The iteration complexity of the two proposed algorithms to obtain an $\varepsilon$-stationary point are proved to be $\mathcal{O}(\varepsilon ^{-2})$ (resp. $\mathcal{O}(\varepsilon ^{-4})$) for solving nonconvex-strongly concave (resp. nonconvex-concave) minimax problems with coupled linear const
    
[^6]: 没有解释的统计学：对可解释机器学习的冷静观察

    Statistics without Interpretation: A Sober Look at Explainable Machine Learning

    [https://arxiv.org/abs/2402.02870](https://arxiv.org/abs/2402.02870)

    解释算法往往数学上复杂且难以解释，这导致解释错误。为了向前推进，解释算法需要明确其输出的解释方式，并澄清可以和不能回答的问题。这一论点基于统计学和解释之间的区别，以及可解释机器学习和应用统计学之间的相似性。

    

    在关于解释算法的快速发展的文献中，这些算法往往不清楚所用于何处及其使用方式。我们认为这是因为解释算法往往在数学上复杂且难以解释。然而，没有清晰解释的复杂统计方法很可能导致解释的错误，这一事实在文献中越来越明显。为了向前推进，关于解释算法的论文应明确解释算法的输出如何解释。他们还应澄清在给出解释的情况下可以回答哪些关于函数的问题，以及哪些问题无法回答。我们的论点基于统计学和它们的解释之间的区别。它还依赖于可解释机器学习和应用统计学之间的相似之处。

    In the rapidly growing literature on explanation algorithms, it often remains unclear what precisely these algorithms are for and how they should be used. We argue that this is because explanation algorithms are often mathematically complex but don't admit a clear interpretation. Unfortunately, complex statistical methods that don't have a clear interpretation are bound to lead to errors in interpretation, a fact that has become increasingly apparent in the literature. In order to move forward, papers on explanation algorithms should make clear how precisely the output of the algorithms should be interpreted. They should also clarify what questions about the function can and cannot be answered given the explanations. Our argument is based on the distinction between statistics and their interpretation. It also relies on parallels between explainable machine learning and applied statistics.
    
[^7]: 计算不平衡分布之间的距离 - 平坦度量

    Computing the Distance between unbalanced Distributions -- The flat Metric. (arXiv:2308.01039v1 [cs.LG])

    [http://arxiv.org/abs/2308.01039](http://arxiv.org/abs/2308.01039)

    该论文提出了一种计算不平衡分布之间距离的方法，基于平坦度量，该方法可以推广到分布总质量不平衡的情况下，特别适用于不平衡最优输运和数据分布分析。论文实现了一个基于神经网络的方法，可以计算出两个给定测度之间距离的最佳测试函数，并通过多个实验验证了方法的准确性。

    

    我们提供了一个在任意维度计算平坦度量的实现。平坦度量，也称为双边界Lipschitz距离，将众所周知的Wasserstein距离W1推广到了分布总质量不平衡的情况。这对于不平衡最优输运任务和数据分布分析中，样本大小重要或者归一化不可能的情况具有特殊的意义。该方法的核心是基于神经网络来确定实现两个给定测度之间距离的最佳测试函数。我们特别注重实现了从独立训练的网络计算出的成对距离的可比性。我们通过几个实验证明了输出的质量，其中包括了一些有实际真值的实验以及使用模拟数据的实验。

    We provide an implementation to compute the flat metric in any dimension. The flat metric, also called dual bounded Lipschitz distance, generalizes the well-known Wasserstein distance W1 to the case that the distributions are of unequal total mass. This is of particular interest for unbalanced optimal transport tasks and for the analysis of data distributions where the sample size is important or normalization is not possible. The core of the method is based on a neural network to determine on optimal test function realizing the distance between two given measures. Special focus was put on achieving comparability of pairwise computed distances from independently trained networks. We tested the quality of the output in several experiments where ground truth was available as well as with simulated data.
    
[^8]: 深度网络可以被剪枝到多么稀疏：几何视角下的研究

    How Sparse Can We Prune A Deep Network: A Geometric Viewpoint. (arXiv:2306.05857v1 [stat.ML])

    [http://arxiv.org/abs/2306.05857](http://arxiv.org/abs/2306.05857)

    本文从高维几何的角度，通过在原始损失函数中强制施加稀疏性约束，描述了深度网络剪枝比率的相变点，该点等于某些凸体的平方高斯宽度除以参数的原始维度。

    

    过度参数化是深度神经网络最重要的特征之一。虽然它可以提供出色的泛化性能，但同时也强加了重大的存储负担，因此有必要研究网络剪枝。一个自然而基本的问题是：我们能剪枝一个深度网络到多么稀疏（几乎不影响性能）？为了解决这个问题，本文采用了第一原理方法，具体地，只通过在原始损失函数中强制施加稀疏性约束，我们能够从高维几何的角度描述剪枝比率的尖锐相变点，该点对应于可行和不可行之间的边界。结果表明，剪枝比率的相变点等于某些凸体的平方高斯宽度，这些凸体是由$l_1$-规则化损失函数得出的，除以参数的原始维度。作为副产品，我们证明了剪枝过程中参数的分布性质。

    Overparameterization constitutes one of the most significant hallmarks of deep neural networks. Though it can offer the advantage of outstanding generalization performance, it meanwhile imposes substantial storage burden, thus necessitating the study of network pruning. A natural and fundamental question is: How sparse can we prune a deep network (with almost no hurt on the performance)? To address this problem, in this work we take a first principles approach, specifically, by merely enforcing the sparsity constraint on the original loss function, we're able to characterize the sharp phase transition point of pruning ratio, which corresponds to the boundary between the feasible and the infeasible, from the perspective of high-dimensional geometry. It turns out that the phase transition point of pruning ratio equals the squared Gaussian width of some convex body resulting from the $l_1$-regularized loss function, normalized by the original dimension of parameters. As a byproduct, we pr
    
[^9]: CLCIFAR：带人类标注互补标签的CIFAR派生基准数据集

    CLCIFAR: CIFAR-Derived Benchmark Datasets with Human Annotated Complementary Labels. (arXiv:2305.08295v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2305.08295](http://arxiv.org/abs/2305.08295)

    本研究开发了由人类标注的互补标签，创造了两个真实世界的CLL数据集，进一步揭示了现实表现下CLL算法的性能，为这一领域的研究提供了更实际的评估标准。

    

    互补标签学习（CLL）是一种弱监督学习范式，旨在仅使用互补标签（标示实例不属于哪些类别）来训练多类分类器。尽管已经提出了多种CLL算法，但由于两个原因，它们的实际表现仍不清楚。首先，这些算法通常依赖于对互补标签生成的假设。其次，它们的评估仅限于合成数据集。为了获取有关CLL算法的真实世界表现的见解，我们开发了一种协议来收集由人类注释者注释的互补标签。这一努力导致创建了两个数据集，CLCIFAR10和CLCIFAR20，分别由CIFAR10和CIFAR100派生而来。这些数据集在https://github.com/ntucllab/complementary_cifar上公开发布，代表了第一个真实世界的CLL数据集。通过广泛的基准实验，我们发现相较于合成数据集，当使用人类注释的互补标签时，性能有明显下降。但是，我们也观察到，真实世界的CLL数据集使得在更接近实际应用条件下评估算法成为可能，从而更真实地评估其性能。

    Complementary-label learning (CLL) is a weakly-supervised learning paradigm that aims to train a multi-class classifier using only complementary labels, which indicate classes to which an instance does not belong. Despite numerous algorithmic proposals for CLL, their practical performance remains unclear for two reasons. Firstly, these algorithms often rely on assumptions about the generation of complementary labels. Secondly, their evaluation has been limited to synthetic datasets. To gain insights into the real-world performance of CLL algorithms, we developed a protocol to collect complementary labels annotated by human annotators. This effort resulted in the creation of two datasets, CLCIFAR10 and CLCIFAR20, derived from CIFAR10 and CIFAR100, respectively. These datasets, publicly released at https://github.com/ntucllab/complementary_cifar, represent the very first real-world CLL datasets. Through extensive benchmark experiments, we discovered a notable decline in performance when 
    
[^10]: 基于几何感知的潜在表示学习用于建模Barrett食管疾病进程

    Geometry-Aware Latent Representation Learning for Modeling Disease Progression of Barrett's Esophagus. (arXiv:2303.12711v1 [eess.IV])

    [http://arxiv.org/abs/2303.12711](http://arxiv.org/abs/2303.12711)

    本文提出了一种基于几何思想的潜在表示学习方法，用于建模Barrett食管疾病进程，与传统方法相比，具有更好的重建损失。

    

    Barrett食管是食管腺癌的唯一先驱，这是一种在诊断时预后不良的食管癌症。因此，诊断Barrett食管对于预防和治疗食管癌至关重要。监督机器学习支持Barrett食管诊断，但组织病理学训练数据的高观察者变异限制了这些方法。用变分自动编码器(VAEs)进行无监督表示学习显示出潜在优势，因为它们将输入数据映射到具有仅有用特征的低维流形，为改进下游任务和见解将Barrett食管病程表征。然而，VAE的欧几里得潜在空间扭曲了点之间的关系，从而阻碍了疾病进展建模。几何VAEs为潜在空间提供附加几何结构，RHVAE假设为黎曼流形，$\mathcal{S}$-VAE假设为超球面流形。我们的研究表明，$\mathcal{S}$-VAE优于常规VAE，具有更好的重建损失。

    Barrett's Esophagus (BE) is the only precursor known to Esophageal Adenocarcinoma (EAC), a type of esophageal cancer with poor prognosis upon diagnosis. Therefore, diagnosing BE is crucial in preventing and treating esophageal cancer. While supervised machine learning supports BE diagnosis, high interobserver variability in histopathological training data limits these methods. Unsupervised representation learning via Variational Autoencoders (VAEs) shows promise, as they map input data to a lower-dimensional manifold with only useful features, characterizing BE progression for improved downstream tasks and insights. However, the VAE's Euclidean latent space distorts point relationships, hindering disease progression modeling. Geometric VAEs provide additional geometric structure to the latent space, with RHVAE assuming a Riemannian manifold and $\mathcal{S}$-VAE a hyperspherical manifold. Our study shows that $\mathcal{S}$-VAE outperforms vanilla VAE with better reconstruction losses, 
    
[^11]: 在不完全信息的市场中使用在线学习进行平衡定价

    Online Learning for Equilibrium Pricing in Markets under Incomplete Information. (arXiv:2303.11522v1 [cs.GT])

    [http://arxiv.org/abs/2303.11522](http://arxiv.org/abs/2303.11522)

    该论文研究了在不完全信息的市场中使用在线学习进行平衡定价的问题，提出了解决选择性谎言问题的新方法。

    

    市场平衡的研究是经济理论的核心，特别是在有效配置稀缺资源方面。然而，定价均衡的计算通常依赖于完整的个体属性信息，如供应商的成本函数等，这在实践中往往不可用。因此，我们考虑了在不完全信息的情况下解决定价均衡的问题。在这种情况下，市场经营者寻求通过从成本函数未知的竞争供应商购买所需数量来满足客户需求。在这种不完整信息的情况下，我们考虑了在线学习问题，即学习随时间变化的平衡价格，同时联合优化三个性能指标——未满足的需求、成本失误和付款失误——这是在定价均衡的情况下相关的。

    The study of market equilibria is central to economic theory, particularly in efficiently allocating scarce resources. However, the computation of equilibrium prices at which the supply of goods matches their demand typically relies on having access to complete information on private attributes of agents, e.g., suppliers' cost functions, which are often unavailable in practice. Motivated by this practical consideration, we consider the problem of setting equilibrium prices in the incomplete information setting wherein a market operator seeks to satisfy the customer demand for a commodity by purchasing the required amount from competing suppliers with privately known cost functions unknown to the market operator. In this incomplete information setting, we consider the online learning problem of learning equilibrium prices over time while jointly optimizing three performance metrics -- unmet demand, cost regret, and payment regret -- pertinent in the context of equilibrium pricing over a
    
[^12]: 对基于Transformer的自监督模型在语音处理中进行压缩

    Compressing Transformer-based self-supervised models for speech processing. (arXiv:2211.09949v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2211.09949](http://arxiv.org/abs/2211.09949)

    本文研究了对基于Transformer的自监督模型进行压缩的方法，包括权重修剪、头部修剪、低秩逼近和知识蒸馏。结果发现，基本的压缩技术是强大的基准，可以改善模型的压缩效果。

    

    尽管Transformer在自监督学习中取得了成功，并应用于各种下游任务，但是训练和推断的计算成本仍然是将这些模型应用于各种设备的主要挑战。目前已有一些孤立的尝试来压缩Transformer，但研究中的设置和指标各不相同。此前的工作很少涉及不同压缩率之间的权衡，这使得比较压缩技术变得困难。在这项工作中，我们旨在为这些孤立结果提供背景，研究几种常用的压缩技术，包括权重修剪、头部修剪、低秩逼近和知识蒸馏。我们报告了在不同压缩率下的权衡，包括墙钟时间、参数数量和乘加操作数量。我们的结果表明，与最近的方法相比，基本的压缩技术是强大的基准。我们进一步提出了几种压缩方法来改进模型的压缩效果。

    Despite the success of Transformers in self- supervised learning with applications to various downstream tasks, the computational cost of training and inference remains a major challenge for applying these models to a wide spectrum of devices. Several isolated attempts have been made to compress Transformers, but the settings and metrics are different across studies. Trade-off at various compression rates are also largely missing in prior work, making it difficult to compare compression techniques. In this work, we aim to provide context for the isolated results, studying several commonly used compression techniques, including weight pruning, head pruning, low-rank approximation, and knowledge distillation. We report trade- off at various compression rate, including wall-clock time, the number of parameters, and the number of multiply-accumulate operations. Our results show that compared to recent approaches, basic compression techniques are strong baselines. We further present several
    
[^13]: 一种符合保序的风险控制方法

    Conformal Risk Control. (arXiv:2208.02814v2 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2208.02814](http://arxiv.org/abs/2208.02814)

    该论文提出了一种符合保序的风险控制方法，可以控制任何单调损失函数的期望值，示例证明其在计算机视觉和自然语言处理领域具有控制误报率、图形距离和令牌级F1得分的能力。

    

    我们将符合性预测推广至控制任何单调损失函数的期望值。该算法将分裂符合性预测及其覆盖保证进行了泛化。类似于符合性预测，符合保序的风险控制方法在$\mathcal{O}(1/n)$因子内保持紧密性。计算机视觉和自然语言处理领域的示例证明了我们算法在控制误报率、图形距离和令牌级F1得分方面的应用。

    We extend conformal prediction to control the expected value of any monotone loss function. The algorithm generalizes split conformal prediction together with its coverage guarantee. Like conformal prediction, the conformal risk control procedure is tight up to an $\mathcal{O}(1/n)$ factor. Worked examples from computer vision and natural language processing demonstrate the usage of our algorithm to bound the false negative rate, graph distance, and token-level F1-score.
    

