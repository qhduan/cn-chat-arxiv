# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Proofs that the Gerber Statistic is Positive Semidefinite.](http://arxiv.org/abs/2305.05663) | 本文证明了Gerber统计量是正半定的。 |
| [^2] | [GPT Agents in Game Theory Experiments.](http://arxiv.org/abs/2305.05516) | 本文探讨了使用GPT代理作为战略游戏实验参与者的潜力。GPT代理可以生成逼真的结果，并在某些方面表现出与人类行为一致的行为。处理效果在两个游戏中都很明显。这项初步探索表明，基于GPT的代理在博弈论实验中具有潜在的应用。 |
| [^3] | [UQ for Credit Risk Management: A deep evidence regression approach.](http://arxiv.org/abs/2305.04967) | 本文扩展了Deep Evidence Regression方法，将其应用于预测信用风险中的违约损失；我们提供了相关的学习框架，并在模拟和实际数据上进行了验证。 |
| [^4] | [BloombergGPT: A Large Language Model for Finance.](http://arxiv.org/abs/2303.17564) | 本文提出了BloombergGPT，一个500亿参数的金融领域的大型语言模型，其基于Bloomberg的广泛数据来源和通用数据集进行训练。通过混合数据集训练，该模型在金融任务上表现出色，并且不会牺牲在普通任务上的性能。 |
| [^5] | [The quintic Ornstein-Uhlenbeck volatility model that jointly calibrates SPX & VIX smiles.](http://arxiv.org/abs/2212.10917) | 五次Ornstein-Uhlenbeck波动率模型可以联合拟合SPX和VIX smiles，并且只需使用6个参数和一个输入曲线即可实现。模型非常简单和易于处理，能够实现精确的波动率过程模拟和高效的衍生品定价。 |
| [^6] | [Designing Universal Causal Deep Learning Models: The Case of Infinite-Dimensional Dynamical Systems from Stochastic Analysis.](http://arxiv.org/abs/2210.13300) | 设计了一个DL模型框架，名为因果神经算子（CNO），以逼近因果算子（CO），并证明了CNO模型可以在紧致集上一致逼近Hölder或平滑迹类算子。 |
| [^7] | [Before and after default: information and optimal portfolio via anticipating calculus.](http://arxiv.org/abs/2208.07163) | 本研究提出了一种使用前向积分的替代方法来避免传统随机控制技术不适用于违约前后情景下的限制性假设，并证明了较弱的强度假设是最优性的适当条件，为应对违约风险的风险管理策略的制定提供了重要思路。 |
| [^8] | [Joint mixability and notions of negative dependence.](http://arxiv.org/abs/2204.11438) | 本文研究了联合可混性的概念和统计学中流行的负相关性概念之间的联系，推导了其成为负相关的条件，证明了在新的不确定性条件下解决了二次成本的多边形最优输运问题。 |
| [^9] | [Simplified calculus for semimartingales: Multiplicative compensators and changes of measure.](http://arxiv.org/abs/2006.12765) | 本文介绍了适用于复值半鞅的乘法补偿，填补了文献中的一个空白，并简化了测度变换的处理。可以计算符号随机指数的梅林变换，有实际应用于均值方差组合理论中。 |

# 详细

[^1]: 证明Gerber统计量是正半定的

    Proofs that the Gerber Statistic is Positive Semidefinite. (arXiv:2305.05663v1 [q-fin.PM])

    [http://arxiv.org/abs/2305.05663](http://arxiv.org/abs/2305.05663)

    本文证明了Gerber统计量是正半定的。

    

    在这篇简短的文章中，我们证明了Gerber统计量在Gerber等人（2022）中引入的两种形式都是正半定的。

    In this brief note, we prove that both forms of the Gerber statistic introduced in Gerber et al. (2022) are positive semi-definite.
    
[^2]: GPT Agent在博弈论实验中的应用

    GPT Agents in Game Theory Experiments. (arXiv:2305.05516v1 [econ.GN])

    [http://arxiv.org/abs/2305.05516](http://arxiv.org/abs/2305.05516)

    本文探讨了使用GPT代理作为战略游戏实验参与者的潜力。GPT代理可以生成逼真的结果，并在某些方面表现出与人类行为一致的行为。处理效果在两个游戏中都很明显。这项初步探索表明，基于GPT的代理在博弈论实验中具有潜在的应用。

    

    本文探讨了使用基于生成预训练转换器（GPT）的代理作为战略游戏实验参与者的潜力。具体而言，作者关注了在经济学中广受研究的有限重复严肃和囚徒困境两个游戏。作者设计了提示，使GPT代理能够理解游戏规则并参与其中。结果表明，在经过精心设计的提示后，GPT可以生成逼真的结果，并在某些方面表现出与人类行为一致的行为，例如在严肃游戏中，接受率与提供金额之间的正相关关系以及在囚徒困境游戏中的合作率。在一些方面，例如在多轮选择的演化方面，GPT行为与人类会有所不同。作者还研究了两种处理方式，在这两种处理方式中通过提示，GPT代理可以具有或没有社会偏好。处理效果在两个游戏中都很明显。这项初步探索表明，基于GPT的代理在博弈论实验中具有潜在的应用，为研究者研究战略行为提供了一个新的工具。

    This paper explores the potential of using Generative Pre-trained Transformer (GPT)-based agents as participants in strategic game experiments. Specifically, I focus on the finitely repeated ultimatum and prisoner's dilemma games, two well-studied games in economics. I develop prompts to enable GPT agents to understand the game rules and play the games. The results indicate that, given well-crafted prompts, GPT can generate realistic outcomes and exhibit behavior consistent with human behavior in certain important aspects, such as positive relationship between acceptance rates and offered amounts in the ultimatum game and positive cooperation rates in the prisoner's dilemma game. Some differences between the behavior of GPT and humans are observed in aspects like the evolution of choices over rounds. I also study two treatments in which the GPT agents are prompted to either have social preferences or not. The treatment effects are evident in both games. This preliminary exploration ind
    
[^3]: 信用风险管理中的量化不确定性：一种深度证据回归方法

    UQ for Credit Risk Management: A deep evidence regression approach. (arXiv:2305.04967v1 [q-fin.RM])

    [http://arxiv.org/abs/2305.04967](http://arxiv.org/abs/2305.04967)

    本文扩展了Deep Evidence Regression方法，将其应用于预测信用风险中的违约损失；我们提供了相关的学习框架，并在模拟和实际数据上进行了验证。

    

    机器学习已经广泛应用于各种信用风险应用程序中。由于信用风险的固有性质，量化预测风险指标的不确定性是必要的，将考虑不确定性的深度学习模型应用于信用风险设置中非常有帮助。在本项工作中，我们探索了一种可扩展的UQ感知深度学习技术，Deep Evidence Regression，并将其应用于预测违约损失。我们通过将Deep Evidence Regression方法扩展到通过Weibull过程生成的目标变量的学习来为文献做出了贡献，并提供了相关的学习框架。我们展示了我们的方法在模拟和实际数据上的应用。

    Machine Learning has invariantly found its way into various Credit Risk applications. Due to the intrinsic nature of Credit Risk, quantifying the uncertainty of the predicted risk metrics is essential, and applying uncertainty-aware deep learning models to credit risk settings can be very helpful. In this work, we have explored the application of a scalable UQ-aware deep learning technique, Deep Evidence Regression and applied it to predicting Loss Given Default. We contribute to the literature by extending the Deep Evidence Regression methodology to learning target variables generated by a Weibull process and provide the relevant learning framework. We demonstrate the application of our approach to both simulated and real-world data.
    
[^4]: BloombergGPT：金融领域的大型语言模型

    BloombergGPT: A Large Language Model for Finance. (arXiv:2303.17564v1 [cs.LG])

    [http://arxiv.org/abs/2303.17564](http://arxiv.org/abs/2303.17564)

    本文提出了BloombergGPT，一个500亿参数的金融领域的大型语言模型，其基于Bloomberg的广泛数据来源和通用数据集进行训练。通过混合数据集训练，该模型在金融任务上表现出色，并且不会牺牲在普通任务上的性能。

    

    自然语言处理在金融技术领域有着广泛而复杂的应用，从情感分析和命名实体识别到问答。大型语言模型（LLM）已被证明在各种任务上非常有效；然而，专为金融领域设计的LLM尚未在文献中报告。在本文中，我们提出了BloombergGPT，一个拥有500亿个参数的语言模型，它是基于广泛的金融数据进行训练的。我们构建了一种3630亿个标记的数据集，该数据集基于彭博社的广泛数据来源，可能是迄今最大的领域特定数据集，同时又增加了来自通用数据集的3450亿个标记。我们在标准LLM基准、开放式金融基准和一套最能准确反映我们预期用途的内部基准上验证了BloombergGPT。我们的混合数据集训练产生了一个在金融任务上明显优于现有模型的模型，同时不会牺牲普通任务的性能。

    The use of NLP in the realm of financial technology is broad and complex, with applications ranging from sentiment analysis and named entity recognition to question answering. Large Language Models (LLMs) have been shown to be effective on a variety of tasks; however, no LLM specialized for the financial domain has been reported in literature. In this work, we present BloombergGPT, a 50 billion parameter language model that is trained on a wide range of financial data. We construct a 363 billion token dataset based on Bloomberg's extensive data sources, perhaps the largest domain-specific dataset yet, augmented with 345 billion tokens from general purpose datasets. We validate BloombergGPT on standard LLM benchmarks, open financial benchmarks, and a suite of internal benchmarks that most accurately reflect our intended usage. Our mixed dataset training leads to a model that outperforms existing models on financial tasks by significant margins without sacrificing performance on general 
    
[^5]: 联合校准SPX和VIX smiles的五次Ornstein-Uhlenbeck波动率模型

    The quintic Ornstein-Uhlenbeck volatility model that jointly calibrates SPX & VIX smiles. (arXiv:2212.10917v2 [q-fin.MF] UPDATED)

    [http://arxiv.org/abs/2212.10917](http://arxiv.org/abs/2212.10917)

    五次Ornstein-Uhlenbeck波动率模型可以联合拟合SPX和VIX smiles，并且只需使用6个参数和一个输入曲线即可实现。模型非常简单和易于处理，能够实现精确的波动率过程模拟和高效的衍生品定价。

    

    五次Ornstein-Uhlenbeck波动率模型是一种随机波动率模型，其中波动率过程是单个具有快速均值回归和大波动率的五次代数函数的Ornstein-Uhlenbeck过程。该模型只使用6个有效参数和一个输入曲线即可实现SPX-VIX smiles的显著联合拟合，该输入曲线允许匹配某些期限结构。我们提供了输入曲线的几个实际规格，研究了它们对联合校准问题的影响，并考虑时间相关参数以帮助更好地拟合超过1年的较长到期日。更好的是，该模型仍然非常简单和易于处理：VIX平方再次是Ornstein-Uhlenbeck过程的多项式，通过对高斯密度的简单积分实现了高效的VIX衍生品定价；波动性过程的模拟是精确的；并且SPX产品衍生品的定价可以有效地完成。

    The quintic Ornstein-Uhlenbeck volatility model is a stochastic volatility model where the volatility process is a polynomial function of degree five of a single Ornstein-Uhlenbeck process with fast mean reversion and large vol-of-vol. The model is able to achieve remarkable joint fits of the SPX-VIX smiles with only 6 effective parameters and an input curve that allows to match certain term structures. We provide several practical specifications of the input curve, study their impact on the joint calibration problem and consider additionally time-dependent parameters to help achieve better fits for longer maturities going beyond 1 year. Even better, the model remains very simple and tractable for pricing and calibration: the VIX squared is again polynomial in the Ornstein-Uhlenbeck process, leading to efficient VIX derivative pricing by a simple integration against a Gaussian density; simulation of the volatility process is exact; and pricing SPX products derivatives can be done effic
    
[^6]: 设计通用因果深度学习模型：以随机分析中的无限维动态系统为例

    Designing Universal Causal Deep Learning Models: The Case of Infinite-Dimensional Dynamical Systems from Stochastic Analysis. (arXiv:2210.13300v2 [math.DS] UPDATED)

    [http://arxiv.org/abs/2210.13300](http://arxiv.org/abs/2210.13300)

    设计了一个DL模型框架，名为因果神经算子（CNO），以逼近因果算子（CO），并证明了CNO模型可以在紧致集上一致逼近Hölder或平滑迹类算子。

    

    因果算子（CO）在当代随机分析中扮演着重要角色，例如各种随机微分方程的解算子。然而，目前还没有一个能够逼近CO的深度学习（DL）模型的规范框架。本文通过引入一个DL模型设计框架来提出一个“几何感知”的解决方案，该框架以合适的无限维线性度量空间为输入，并返回适应这些线性几何的通用连续序列DL模型。我们称这些模型为因果神经算子（CNO）。我们的主要结果表明，我们的框架所产生的模型可以在紧致集上和跨任意有限时间视野上一致逼近Hölder或平滑迹类算子，这些算子因果地映射给定线性度量空间之间的序列。我们的分析揭示了关于CNO的潜在状态空间维度的新定量关系，甚至对于（经典的）有限维DL模型也有新的影响。

    Causal operators (CO), such as various solution operators to stochastic differential equations, play a central role in contemporary stochastic analysis; however, there is still no canonical framework for designing Deep Learning (DL) models capable of approximating COs. This paper proposes a "geometry-aware'" solution to this open problem by introducing a DL model-design framework that takes suitable infinite-dimensional linear metric spaces as inputs and returns a universal sequential DL model adapted to these linear geometries. We call these models Causal Neural Operators (CNOs). Our main result states that the models produced by our framework can uniformly approximate on compact sets and across arbitrarily finite-time horizons H\"older or smooth trace class operators, which causally map sequences between given linear metric spaces. Our analysis uncovers new quantitative relationships on the latent state-space dimension of CNOs which even have new implications for (classical) finite-d
    
[^7]: 违约前后：基于预期计算的信息和最优投资组合

    Before and after default: information and optimal portfolio via anticipating calculus. (arXiv:2208.07163v2 [q-fin.PM] UPDATED)

    [http://arxiv.org/abs/2208.07163](http://arxiv.org/abs/2208.07163)

    本研究提出了一种使用前向积分的替代方法来避免传统随机控制技术不适用于违约前后情景下的限制性假设，并证明了较弱的强度假设是最优性的适当条件，为应对违约风险的风险管理策略的制定提供了重要思路。

    

    在风险资产受到破产威胁时，违约风险计算在投资组合优化中起着至关重要的作用。然而，在这种情况下，传统的随机控制技术并不适用，需要额外的假设才能得到违约前后情景下的最优解。我们提出了一种使用前向积分的替代方法，可以避免一种限制性假设，即Jacod密度假设。我们证明，在对数效用的情况下，较弱的强度假设是最优性的适当条件。此外，我们在假定存在最优投资组合的情况下，建立了风险资产在筛选器中的半鞅分解，该筛选器将逐步扩大来适应违约过程。本研究旨在为应对违约风险的风险管理策略的制定提供有价值的见解。

    Default risk calculus plays a crucial role in portfolio optimization when the risky asset is under threat of bankruptcy. However, traditional stochastic control techniques are not applicable in this scenario, and additional assumptions are required to obtain the optimal solution in a before-and-after default context. We propose an alternative approach using forward integration, which allows to avoid one of the restrictive assumptions, the Jacod density hypothesis. We demonstrate that, in the case of logarithmic utility, the weaker intensity hypothesis is the appropriate condition for optimality. Furthermore, we establish the semimartingale decomposition of the risky asset in the filtration that is progressively enlarged to accommodate the default process, under the assumption of the existence of the optimal portfolio. This work aims to provide valueable insights for developing effective risk management strategies when facing default risk.
    
[^8]: 联合可混性和负相关性概念

    Joint mixability and notions of negative dependence. (arXiv:2204.11438v3 [math.ST] UPDATED)

    [http://arxiv.org/abs/2204.11438](http://arxiv.org/abs/2204.11438)

    本文研究了联合可混性的概念和统计学中流行的负相关性概念之间的联系，推导了其成为负相关的条件，证明了在新的不确定性条件下解决了二次成本的多边形最优输运问题。

    

    联合可混是一个具有恒定分量总和的随机向量。 联合混合的相关结构使一些常见的目标最小化，例如分量总和的方差，并被认为是极端负相关的概念。 本文探讨了联合混合结构与统计学中流行的负相关性概念之间的联系，例如负相关依赖、负半轴依赖和负关联。 联合混合不总是以任何上述意义负相关，但是一些自然类的联合混合是。 我们推导了联合混合成为负相关的各种必要和充分条件，并研究了这些概念的兼容性。 对于相同的边缘分布，我们证明了负相关联合混合在新的不确定性设置下解决了二次成本的多边形最优输运问题。 对不同边缘分布的最优输运问题的分析

    A joint mix is a random vector with a constant component-wise sum. The dependence structure of a joint mix minimizes some common objectives such as the variance of the component-wise sum, and it is regarded as a concept of extremal negative dependence. In this paper, we explore the connection between the joint mix structure and popular notions of negative dependence in statistics, such as negative correlation dependence, negative orthant dependence and negative association. A joint mix is not always negatively dependent in any of the above senses, but some natural classes of joint mixes are. We derive various necessary and sufficient conditions for a joint mix to be negatively dependent, and study the compatibility of these notions. For identical marginal distributions, we show that a negatively dependent joint mix solves a multi-marginal optimal transport problem for quadratic cost under a novel setting of uncertainty. Analysis of this optimal transport problem with heterogeneous marg
    
[^9]: 半鞅的简化微积分：乘法补偿和测度变换

    Simplified calculus for semimartingales: Multiplicative compensators and changes of measure. (arXiv:2006.12765v4 [math.PR] UPDATED)

    [http://arxiv.org/abs/2006.12765](http://arxiv.org/abs/2006.12765)

    本文介绍了适用于复值半鞅的乘法补偿，填补了文献中的一个空白，并简化了测度变换的处理。可以计算符号随机指数的梅林变换，有实际应用于均值方差组合理论中。

    

    本文开发了适用于复值半鞅的乘法补偿，并研究了其一些结果。当有意义的时候，它表明具有独立增量的任何复值半鞅的随机指数成为真鞅。这种L\'evy--Khintchin公式的推广填补了文献中的一个现有空白。例如，它允许计算符号随机指数的梅林变换，进而在均值方差组合理论中具有实际应用。基于乘法补偿的半鞅Girsanov类型的结果简化了绝对连续测度变换的处理。作为例子，我们在L\'evy设置中获得了一类最小极小测度的对数回报的特征函数。

    The paper develops multiplicative compensation for complex-valued semimartingales and studies some of its consequences. It is shown that the stochastic exponential of any complex-valued semimartingale with independent increments becomes a true martingale after multiplicative compensation when such compensation is meaningful. This generalization of the L\'evy--Khintchin formula fills an existing gap in the literature. It allows, for example, the computation of the Mellin transform of a signed stochastic exponential, which in turn has practical applications in mean--variance portfolio theory. Girsanov-type results based on multiplicatively compensated semimartingales simplify treatment of absolutely continuous measure changes. As an example, we obtain the characteristic function of log returns for a popular class of minimax measures in a L\'evy setting.
    

