# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Nowcasting with mixed frequency data using Gaussian processes](https://arxiv.org/abs/2402.10574) | 使用高斯过程和贝叶斯添加回归树作为线性惩罚估计的灵活扩展，解决了混合频率数据中的频率不匹配问题，提高了现场预测的准确性。 |
| [^2] | [Optimal Regularization for a Data Source](https://arxiv.org/abs/2212.13597) | 本文研究了基于优化的方法中凸正则化的能力和局限性问题，通过研究给定分布情况下，对数据采用何种正则化方法是最优的。 |
| [^3] | [Optimal Differentially Private Learning with Public Data.](http://arxiv.org/abs/2306.15056) | 本论文研究了具有公共数据的最优差分隐私学习，并解决了在训练差分隐私模型时如何利用公共数据提高准确性的问题。 |
| [^4] | [Memory of recurrent networks: Do we compute it right?.](http://arxiv.org/abs/2305.01457) | 本文研究了线性回声状态网络的记忆容量计算问题。通过发现数值评估的不准确性主要源于数值方面的问题，提出了基于掩码矩阵MC相对于中立性的稳健数值方法，该方法可以解决数值评估中的误差问题。 |

# 详细

[^1]: 使用高斯过程进行混合频率数据的现场预测

    Nowcasting with mixed frequency data using Gaussian processes

    [https://arxiv.org/abs/2402.10574](https://arxiv.org/abs/2402.10574)

    使用高斯过程和贝叶斯添加回归树作为线性惩罚估计的灵活扩展，解决了混合频率数据中的频率不匹配问题，提高了现场预测的准确性。

    

    我们提出并讨论了用于混合数据采样（MIDAS）回归的贝叶斯机器学习方法。这涉及使用受限和非受限的MIDAS变体处理频率不匹配，并指定许多预测变量与因变量之间的函数关系。我们使用高斯过程（GP）和贝叶斯添加回归树（BART）作为线性惩罚估计的灵活扩展。在现场预测和预测练习中，我们专注于季度美国产出增长和GDP价格指数的通货膨胀。这些新模型以计算效率的方式利用宏观经济大数据，并在多个维度上提供了预测准确度的增益。

    arXiv:2402.10574v1 Announce Type: new  Abstract: We propose and discuss Bayesian machine learning methods for mixed data sampling (MIDAS) regressions. This involves handling frequency mismatches with restricted and unrestricted MIDAS variants and specifying functional relationships between many predictors and the dependent variable. We use Gaussian processes (GP) and Bayesian additive regression trees (BART) as flexible extensions to linear penalized estimation. In a nowcasting and forecasting exercise we focus on quarterly US output growth and inflation in the GDP deflator. The new models leverage macroeconomic Big Data in a computationally efficient way and offer gains in predictive accuracy along several dimensions.
    
[^2]: 适用于数据源的最优正则化方法

    Optimal Regularization for a Data Source

    [https://arxiv.org/abs/2212.13597](https://arxiv.org/abs/2212.13597)

    本文研究了基于优化的方法中凸正则化的能力和局限性问题，通过研究给定分布情况下，对数据采用何种正则化方法是最优的。

    

    在基于优化的逆问题和统计估计中，常常通过加入促使数据保真性的准则和促进解的所需结构性质的正则化项来解决问题。选择适当的正则化约束通常由前领域知识和计算考虑共同驱动。凸正则化项在计算上具有吸引力，但在提升结构类型方面存在局限性。另一方面，非凸正则化项在促进结构类型方面更具灵活性，并且在某些应用中展示出了强大的实证性能，但同时也带来了解决相关优化问题的计算挑战。本文通过研究以下问题，寻求对凸正则化在效能和局限性方面的系统理解：给定一个分布，对于从该分布中抽取的数据，什么是最优的正则化方法？

    In optimization-based approaches to inverse problems and to statistical estimation, it is common to augment criteria that enforce data fidelity with a regularizer that promotes desired structural properties in the solution. The choice of a suitable regularizer is typically driven by a combination of prior domain information and computational considerations. Convex regularizers are attractive computationally but they are limited in the types of structure they can promote. On the other hand, nonconvex regularizers are more flexible in the forms of structure they can promote and they have showcased strong empirical performance in some applications, but they come with the computational challenge of solving the associated optimization problems. In this paper, we seek a systematic understanding of the power and the limitations of convex regularization by investigating the following questions: Given a distribution, what is the optimal regularizer for data drawn from the distribution? What pro
    
[^3]: 具有公共数据的最优差分隐私学习

    Optimal Differentially Private Learning with Public Data. (arXiv:2306.15056v1 [cs.LG])

    [http://arxiv.org/abs/2306.15056](http://arxiv.org/abs/2306.15056)

    本论文研究了具有公共数据的最优差分隐私学习，并解决了在训练差分隐私模型时如何利用公共数据提高准确性的问题。

    

    差分隐私能够确保训练机器学习模型不泄漏私密数据。然而，差分隐私的代价是模型的准确性降低或样本复杂度增加。在实践中，我们可能可以访问不涉及隐私问题的辅助公共数据。这促使了最近研究公共数据在提高差分隐私模型准确性方面的作用。在本研究中，我们假设有一定数量的公共数据，并解决以下基本开放问题：1.在有公共数据的情况下，训练基于私有数据集的差分隐私模型的最优（最坏情况）误差是多少？哪些算法是最优的？2.如何利用公共数据在实践中改进差分隐私模型训练？我们在本地模型和中心模型的差分隐私问题下考虑这些问题。为了回答第一个问题，我们证明了对三个基本问题的最优误差率的紧密（最高常数因子）下界和上界。这三个问题是：均值估计，经验风险最小化和凸奇化。

    Differential Privacy (DP) ensures that training a machine learning model does not leak private data. However, the cost of DP is lower model accuracy or higher sample complexity. In practice, we may have access to auxiliary public data that is free of privacy concerns. This has motivated the recent study of what role public data might play in improving the accuracy of DP models. In this work, we assume access to a given amount of public data and settle the following fundamental open questions: 1. What is the optimal (worst-case) error of a DP model trained over a private data set while having access to side public data? What algorithms are optimal? 2. How can we harness public data to improve DP model training in practice? We consider these questions in both the local and central models of DP. To answer the first question, we prove tight (up to constant factors) lower and upper bounds that characterize the optimal error rates of three fundamental problems: mean estimation, empirical ris
    
[^4]: 循环神经网络的记忆：我们计算得对吗？

    Memory of recurrent networks: Do we compute it right?. (arXiv:2305.01457v1 [cs.LG])

    [http://arxiv.org/abs/2305.01457](http://arxiv.org/abs/2305.01457)

    本文研究了线性回声状态网络的记忆容量计算问题。通过发现数值评估的不准确性主要源于数值方面的问题，提出了基于掩码矩阵MC相对于中立性的稳健数值方法，该方法可以解决数值评估中的误差问题。

    

    文献中对于循环神经网络的记忆容量（MC）的数值评估常常与已经建立的理论界限相矛盾。本文研究了线性回声状态网络的情况，对应的Kalman可控矩阵的秩已被证明等于总记忆容量。我们揭示了关于记忆不准确的数值评估的各种原因，并表明这些问题是纯粹数值方面上的，往往在近期文献中被忽视。更明确地说，我们证明了当线性MC的Krylov结构被忽略时，理论MC和它的经验值之间会存在差距。解决这一问题的方法是，利用MC相对于输入掩码矩阵的中立性，开发出稳健的数值方法。模拟结果显示，我们提出的方法得到的记忆曲线与理论完全一致。

    Numerical evaluations of the memory capacity (MC) of recurrent neural networks reported in the literature often contradict well-established theoretical bounds. In this paper, we study the case of linear echo state networks, for which the total memory capacity has been proven to be equal to the rank of the corresponding Kalman controllability matrix. We shed light on various reasons for the inaccurate numerical estimations of the memory, and we show that these issues, often overlooked in the recent literature, are of an exclusively numerical nature. More explicitly, we prove that when the Krylov structure of the linear MC is ignored, a gap between the theoretical MC and its empirical counterpart is introduced. As a solution, we develop robust numerical approaches by exploiting a result of MC neutrality with respect to the input mask matrix. Simulations show that the memory curves that are recovered using the proposed methods fully agree with the theory.
    

