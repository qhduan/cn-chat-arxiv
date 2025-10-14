# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Deep Conditional Generative Learning: Model and Error Analysis](https://rss.arxiv.org/abs/2402.01460) | 提出了一种基于ODE的深度生成方法，通过条件Follmer流来学习条件分布，通过离散化和深度神经网络实现高效转化。同时，通过Wasserstein距离的非渐近收敛速率，提供了第一个端到端误差分析，数值实验证明其在不同场景下的优越性。 |
| [^2] | [Human-in-the-Loop Causal Discovery under Latent Confounding using Ancestral GFlowNets.](http://arxiv.org/abs/2309.12032) | 该论文提出了一种人机协同的因果发现方法，通过使用生成流网按照基于评分函数的信念分布采样祖先图，并引入最佳实验设计与专家互动，以提供专家可验证的不确定性估计并迭代改进因果推断。 |

# 详细

[^1]: 深度条件生成学习：模型与误差分析

    Deep Conditional Generative Learning: Model and Error Analysis

    [https://rss.arxiv.org/abs/2402.01460](https://rss.arxiv.org/abs/2402.01460)

    提出了一种基于ODE的深度生成方法，通过条件Follmer流来学习条件分布，通过离散化和深度神经网络实现高效转化。同时，通过Wasserstein距离的非渐近收敛速率，提供了第一个端到端误差分析，数值实验证明其在不同场景下的优越性。

    

    我们介绍了一种基于普通微分方程（ODE）的深度生成方法，用于学习条件分布，称为条件Follmer流。从标准高斯分布开始，所提出的流能够以高效的方式将其转化为目标条件分布，在时间1处达到稳定。为了有效实现，我们使用欧拉方法对流进行离散化，使用深度神经网络非参数化估计速度场。此外，我们导出了学习样本的分布与目标分布之间的Wasserstein距离的非渐近收敛速率，在条件分布学习中提供了第一个全面的端到端误差分析。我们的数值实验展示了它在一系列情况下的有效性，从标准的非参数化条件密度估计问题到涉及图像数据的更复杂的挑战，说明它优于各种现有方法。

    We introduce an Ordinary Differential Equation (ODE) based deep generative method for learning a conditional distribution, named the Conditional Follmer Flow. Starting from a standard Gaussian distribution, the proposed flow could efficiently transform it into the target conditional distribution at time 1. For effective implementation, we discretize the flow with Euler's method where we estimate the velocity field nonparametrically using a deep neural network. Furthermore, we derive a non-asymptotic convergence rate in the Wasserstein distance between the distribution of the learned samples and the target distribution, providing the first comprehensive end-to-end error analysis for conditional distribution learning via ODE flow. Our numerical experiments showcase its effectiveness across a range of scenarios, from standard nonparametric conditional density estimation problems to more intricate challenges involving image data, illustrating its superiority over various existing condition
    
[^2]: 人机协同下使用祖先GFlowNets进行潜在混淆的因果发现

    Human-in-the-Loop Causal Discovery under Latent Confounding using Ancestral GFlowNets. (arXiv:2309.12032v1 [cs.LG])

    [http://arxiv.org/abs/2309.12032](http://arxiv.org/abs/2309.12032)

    该论文提出了一种人机协同的因果发现方法，通过使用生成流网按照基于评分函数的信念分布采样祖先图，并引入最佳实验设计与专家互动，以提供专家可验证的不确定性估计并迭代改进因果推断。

    

    结构学习是因果推断的关键。值得注意的是，当数据稀缺时，因果发现（CD）算法很脆弱，可能推断出与专家知识相矛盾的不准确因果关系，尤其是考虑到潜在混淆因素时更是如此。为了加重这个问题，大多数CD方法并不提供不确定性估计，这使得用户难以解释结果和改进推断过程。令人惊讶的是，尽管CD是一个以人为中心的事务，但没有任何研究专注于构建既能输出专家可验证的不确定性估计又能与专家进行交互迭代改进的方法。为了解决这些问题，我们首先提出使用生成流网，根据基于评分函数（如贝叶斯信息准则）的信念分布，按比例对（因果）祖先图进行采样。然后，我们利用候选图的多样性并引入最佳实验设计，以迭代性地探索实验来与专家互动。

    Structure learning is the crux of causal inference. Notably, causal discovery (CD) algorithms are brittle when data is scarce, possibly inferring imprecise causal relations that contradict expert knowledge -- especially when considering latent confounders. To aggravate the issue, most CD methods do not provide uncertainty estimates, making it hard for users to interpret results and improve the inference process. Surprisingly, while CD is a human-centered affair, no works have focused on building methods that both 1) output uncertainty estimates that can be verified by experts and 2) interact with those experts to iteratively refine CD. To solve these issues, we start by proposing to sample (causal) ancestral graphs proportionally to a belief distribution based on a score function, such as the Bayesian information criterion (BIC), using generative flow networks. Then, we leverage the diversity in candidate graphs and introduce an optimal experimental design to iteratively probe the expe
    

