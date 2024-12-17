# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [PyTorch Frame: A Modular Framework for Multi-Modal Tabular Learning](https://arxiv.org/abs/2404.00776) | PyTorch Frame是一个用于处理多模态表格数据的PyTorch框架，通过提供数据结构、模型抽象和外部基础模型整合等功能，实现了模块化的表格模型实现，并成功将这些模型应用于复杂的数据集。 |
| [^2] | [Robust Learning for Optimal Dynamic Treatment Regimes with Observational Data](https://arxiv.org/abs/2404.00221) | 学习利用观测数据提出了一种逐步双重强健方法，通过向后归纳解决了最佳动态治疗方案的问题 |
| [^3] | [Estimating the history of a random recursive tree](https://arxiv.org/abs/2403.09755) | 本文研究了估计随机递归树中顶点到达顺序的问题，提出了基于Jordan中心性度量的顺序估计器，并证明其几乎是最优的。 |
| [^4] | [Optimal Top-Two Method for Best Arm Identification and Fluid Analysis](https://arxiv.org/abs/2403.09123) | 提出了解决最佳臂识别问题中的样本复杂度挑战的最优Top-Two算法。 |
| [^5] | [Simulation Based Bayesian Optimization.](http://arxiv.org/abs/2401.10811) | 本文介绍了基于仿真的贝叶斯优化（SBBO）作为一种新方法，用于通过仅需基于采样的访问来优化获取函数。 |
| [^6] | [xVal: A Continuous Number Encoding for Large Language Models.](http://arxiv.org/abs/2310.02989) | xVal是一种连续数字编码方案，通过使用单个标记来表示任何实数。与现有的数字编码方案相比，xVal更加高效，并且在泛化性能上表现更好。 |

# 详细

[^1]: PyTorch Frame: 一个用于多模态表格学习的模块化框架

    PyTorch Frame: A Modular Framework for Multi-Modal Tabular Learning

    [https://arxiv.org/abs/2404.00776](https://arxiv.org/abs/2404.00776)

    PyTorch Frame是一个用于处理多模态表格数据的PyTorch框架，通过提供数据结构、模型抽象和外部基础模型整合等功能，实现了模块化的表格模型实现，并成功将这些模型应用于复杂的数据集。

    

    我们提出了PyTorch Frame，这是一个基于PyTorch的框架，用于处理多模态表格数据的深度学习。PyTorch Frame通过提供基于PyTorch的数据结构来处理复杂的表格数据，引入模型抽象以实现表格模型的模块化实现，并允许整合外部基础模型来处理复杂列（例如，用于文本列的LLMs）。我们通过以模块化方式实现多样的表格模型，成功将这些模型应用于复杂的多模态表格数据，并将我们的框架与PyTorch Geometric集成，PyTorch Geometric是一个用于图神经网络（GNNs）的PyTorch库，以实现对关系数据库的端到端学习。

    arXiv:2404.00776v1 Announce Type: new  Abstract: We present PyTorch Frame, a PyTorch-based framework for deep learning over multi-modal tabular data. PyTorch Frame makes tabular deep learning easy by providing a PyTorch-based data structure to handle complex tabular data, introducing a model abstraction to enable modular implementation of tabular models, and allowing external foundation models to be incorporated to handle complex columns (e.g., LLMs for text columns). We demonstrate the usefulness of PyTorch Frame by implementing diverse tabular models in a modular way, successfully applying these models to complex multi-modal tabular data, and integrating our framework with PyTorch Geometric, a PyTorch library for Graph Neural Networks (GNNs), to perform end-to-end learning over relational databases.
    
[^2]: 利用观测数据进行强健学习以获得最佳动态治疗方案

    Robust Learning for Optimal Dynamic Treatment Regimes with Observational Data

    [https://arxiv.org/abs/2404.00221](https://arxiv.org/abs/2404.00221)

    学习利用观测数据提出了一种逐步双重强健方法，通过向后归纳解决了最佳动态治疗方案的问题

    

    许多公共政策和医疗干预涉及其治疗分配中的动态性，治疗通常依据先前治疗的历史和相关特征对每个阶段的效果具有异质性。本文研究了统计学习最佳动态治疗方案(DTR)，根据个体的历史指导每个阶段的最佳治疗分配。我们提出了一种基于观测数据的逐步双重强健方法，在顺序可忽略性假设下学习最佳DTR。该方法通过向后归纳解决了顺序治疗分配问题，在每一步中，我们结合倾向评分和行动值函数(Q函数)的估计量，构建了政策价值的增强反向概率加权估计量。

    arXiv:2404.00221v1 Announce Type: cross  Abstract: Many public policies and medical interventions involve dynamics in their treatment assignments, where treatments are sequentially assigned to the same individuals across multiple stages, and the effect of treatment at each stage is usually heterogeneous with respect to the history of prior treatments and associated characteristics. We study statistical learning of optimal dynamic treatment regimes (DTRs) that guide the optimal treatment assignment for each individual at each stage based on the individual's history. We propose a step-wise doubly-robust approach to learn the optimal DTR using observational data under the assumption of sequential ignorability. The approach solves the sequential treatment assignment problem through backward induction, where, at each step, we combine estimators of propensity scores and action-value functions (Q-functions) to construct augmented inverse probability weighting estimators of values of policies 
    
[^3]: 估计随机递归树的历史

    Estimating the history of a random recursive tree

    [https://arxiv.org/abs/2403.09755](https://arxiv.org/abs/2403.09755)

    本文研究了估计随机递归树中顶点到达顺序的问题，提出了基于Jordan中心性度量的顺序估计器，并证明其几乎是最优的。

    

    本文研究了估计随机递归树中顶点到达顺序的问题。具体来说，我们研究了两个基本模型：均匀连接模型和线性优先连接模型。我们提出了一种基于Jordan中心性度量的顺序估计器，并定义了一族风险度量来量化排序过程的质量。此外，我们为这个问题建立了极小-最大下界，并证明所提出的估计器几乎是最优的。最后，我们通过数值实验表明所提出的估计器优于基于度数和谱排序程序。

    arXiv:2403.09755v1 Announce Type: cross  Abstract: This paper studies the problem of estimating the order of arrival of the vertices in a random recursive tree. Specifically, we study two fundamental models: the uniform attachment model and the linear preferential attachment model. We propose an order estimator based on the Jordan centrality measure and define a family of risk measures to quantify the quality of the ordering procedure. Moreover, we establish a minimax lower bound for this problem, and prove that the proposed estimator is nearly optimal. Finally, we numerically demonstrate that the proposed estimator outperforms degree-based and spectral ordering procedures.
    
[^4]: 最佳臂识别和流体分析的最优Top-Two方法

    Optimal Top-Two Method for Best Arm Identification and Fluid Analysis

    [https://arxiv.org/abs/2403.09123](https://arxiv.org/abs/2403.09123)

    提出了解决最佳臂识别问题中的样本复杂度挑战的最优Top-Two算法。

    

    Top-2方法在解决最佳臂识别（BAI）问题中变得流行。该方法通过一个算法识别最佳臂，即在有限数量臂中具有最大均值的臂，该算法在任何顺序步骤中独立地以固定概率 β 拉动经验最佳臂，并在其他情况下拉动最佳挑战者臂。选择错误的概率保证在指定的δ >0以下。对于BAI问题，已知信息理论下界的样本复杂度，并在δ → 0时与计算要求高的插件方法渐近匹配。 对于任何 β ∈（0,1）的上述Top 2算法的样本复杂度始终保持在下界的常数范围内。然而，确定与下界匹配的最佳 β 已被证明困难。在本文中，我们解决了这个问题并提出了一个最优的Top-2类型算法。我们考虑分配锚点的一个函数。

    arXiv:2403.09123v1 Announce Type: new  Abstract: Top-$2$ methods have become popular in solving the best arm identification (BAI) problem. The best arm, or the arm with the largest mean amongst finitely many, is identified through an algorithm that at any sequential step independently pulls the empirical best arm, with a fixed probability $\beta$, and pulls the best challenger arm otherwise. The probability of incorrect selection is guaranteed to lie below a specified $\delta >0$. Information theoretic lower bounds on sample complexity are well known for BAI problem and are matched asymptotically as $\delta \rightarrow 0$ by computationally demanding plug-in methods. The above top 2 algorithm for any $\beta \in (0,1)$ has sample complexity within a constant of the lower bound. However, determining the optimal $\beta$ that matches the lower bound has proven difficult. In this paper, we address this and propose an optimal top-2 type algorithm. We consider a function of allocations anchor
    
[^5]: 基于仿真的贝叶斯优化

    Simulation Based Bayesian Optimization. (arXiv:2401.10811v1 [stat.ML])

    [http://arxiv.org/abs/2401.10811](http://arxiv.org/abs/2401.10811)

    本文介绍了基于仿真的贝叶斯优化（SBBO）作为一种新方法，用于通过仅需基于采样的访问来优化获取函数。

    

    贝叶斯优化是一种将先验知识与持续函数评估相结合的强大方法，用于优化黑盒函数。贝叶斯优化通过构建与协变量相关的目标函数的概率代理模型来指导未来评估点的选择。对于平滑连续的搜索空间，高斯过程经常被用作代理模型，因为它们提供对后验预测分布的解析访问，从而便于计算和优化获取函数。然而，在涉及对分类或混合协变量空间进行优化的复杂情况下，高斯过程可能不是理想的选择。本文介绍了一种名为基于仿真的贝叶斯优化（SBBO）的新方法，该方法仅需要对后验预测分布进行基于采样的访问，以优化获取函数。

    Bayesian Optimization (BO) is a powerful method for optimizing black-box functions by combining prior knowledge with ongoing function evaluations. BO constructs a probabilistic surrogate model of the objective function given the covariates, which is in turn used to inform the selection of future evaluation points through an acquisition function. For smooth continuous search spaces, Gaussian Processes (GPs) are commonly used as the surrogate model as they offer analytical access to posterior predictive distributions, thus facilitating the computation and optimization of acquisition functions. However, in complex scenarios involving optimizations over categorical or mixed covariate spaces, GPs may not be ideal.  This paper introduces Simulation Based Bayesian Optimization (SBBO) as a novel approach to optimizing acquisition functions that only requires \emph{sampling-based} access to posterior predictive distributions. SBBO allows the use of surrogate probabilistic models tailored for co
    
[^6]: xVal: 大型语言模型的连续数字编码

    xVal: A Continuous Number Encoding for Large Language Models. (arXiv:2310.02989v1 [stat.ML])

    [http://arxiv.org/abs/2310.02989](http://arxiv.org/abs/2310.02989)

    xVal是一种连续数字编码方案，通过使用单个标记来表示任何实数。与现有的数字编码方案相比，xVal更加高效，并且在泛化性能上表现更好。

    

    由于数字令牌化的独特困难，大型语言模型尚未广泛用于科学数据集的分析。我们提出了xVal，一种数字编码方案，可以使用单个标记来表示任何实数。xVal通过将专用嵌入向量按数字值进行缩放来表示给定的实数。结合修改后的数字推断方法，该策略使模型在考虑作为从输入字符串的数字到输出字符串的数字的映射时成为端到端连续的。这导致了一种更适用于科学领域应用的归纳偏差。我们在许多合成和现实世界数据集上进行了实证评估。与现有的数字编码方案相比，我们发现xVal在令牌效率和泛化性能上表现更好。

    Large Language Models have not yet been broadly adapted for the analysis of scientific datasets due in part to the unique difficulties of tokenizing numbers. We propose xVal, a numerical encoding scheme that represents any real number using just a single token. xVal represents a given real number by scaling a dedicated embedding vector by the number value. Combined with a modified number-inference approach, this strategy renders the model end-to-end continuous when considered as a map from the numbers of the input string to those of the output string. This leads to an inductive bias that is generally more suitable for applications in scientific domains. We empirically evaluate our proposal on a number of synthetic and real-world datasets. Compared with existing number encoding schemes, we find that xVal is more token-efficient and demonstrates improved generalization.
    

