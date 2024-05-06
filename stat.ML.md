# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Credal Learning Theory](https://rss.arxiv.org/abs/2402.00957) | 本文提出了一种信任学习理论，通过使用凸集的概率来建模数据生成分布的变异性，从有限样本的训练集中推断出信任集，并推导出bounds。 |
| [^2] | [Learning Action-based Representations Using Invariance](https://arxiv.org/abs/2403.16369) | 提出了一种新的方法，动作双模拟编码，通过递归不变性约束扩展了单步控制性，学习了一个可以平滑折扣远期元素的多步控制度量 |
| [^3] | [Contraction of Locally Differentially Private Mechanisms](https://arxiv.org/abs/2210.13386) | 本文研究了局部差分隐私机制的收缩性质, 并给出了输出分布与输入分布之间差异的上界，这对于界定极小极大估计风险非常有用。 |
| [^4] | [The Impact of Differential Feature Under-reporting on Algorithmic Fairness.](http://arxiv.org/abs/2401.08788) | 本文研究了差异特征未报告对算法公平性的影响，并提出了一个可分析的模型进行刻画。 |
| [^5] | [Decolonial AI Alignment: Vi\'{s}esadharma, Argument, and Artistic Expression.](http://arxiv.org/abs/2309.05030) | 本文提出了去殖民化人工智能对齐的三个建议：改变基本道德哲学为达尔玛哲学，允许多元主义的论证传统存在于对齐技术中，以及将价值认识论扩展到超越自然语言中的指令。 |
| [^6] | [A General Framework for Learning under Corruption: Label Noise, Attribute Noise, and Beyond.](http://arxiv.org/abs/2307.08643) | 该研究提出了一个通用框架，在分布层面上对不同类型的数据污染模型进行了形式化分析，并通过分析贝叶斯风险的变化展示了这些污染对标准监督学习的影响。这些发现为进一步研究提供了新的方向和基础。 |
| [^7] | [A Black-box Approach for Non-stationary Multi-agent Reinforcement Learning.](http://arxiv.org/abs/2306.07465) | 本文提出了一种通用的黑盒方法，适用于多种多智能体强化学习问题，可以在非平稳环境下实现低遗憾率的学习。 |
| [^8] | [Adaptive deep learning for nonlinear time series models.](http://arxiv.org/abs/2207.02546) | 本文提出了一种自适应深度学习方法用于非线性时间序列模型的均值函数估计，证明了稀疏惩罚的深度神经网络估计器在许多非线性自回归模型中达到了极小下界的最优速率。 |

# 详细

[^1]: 信任学习理论

    Credal Learning Theory

    [https://rss.arxiv.org/abs/2402.00957](https://rss.arxiv.org/abs/2402.00957)

    本文提出了一种信任学习理论，通过使用凸集的概率来建模数据生成分布的变异性，从有限样本的训练集中推断出信任集，并推导出bounds。

    

    统计学习理论是机器学习的基础，为从未知概率分布中学习到的模型的风险提供理论边界。然而，在实际部署中，数据分布可能会变化，导致领域适应/泛化问题。在本文中，我们建立了一个“信任”学习理论的基础，使用概率的凸集（信任集）来建模数据生成分布的变异性。我们认为，这样的信任集可以从有限样本的训练集中推断出来。对于有限假设空间（无论是否可实现）和无限模型空间，推导出界限，这直接推广了经典结果。

    Statistical learning theory is the foundation of machine learning, providing theoretical bounds for the risk of models learnt from a (single) training set, assumed to issue from an unknown probability distribution. In actual deployment, however, the data distribution may (and often does) vary, causing domain adaptation/generalization issues. In this paper we lay the foundations for a `credal' theory of learning, using convex sets of probabilities (credal sets) to model the variability in the data-generating distribution. Such credal sets, we argue, may be inferred from a finite sample of training sets. Bounds are derived for the case of finite hypotheses spaces (both assuming realizability or not) as well as infinite model spaces, which directly generalize classical results.
    
[^2]: 使用不变性学习基于动作的表示

    Learning Action-based Representations Using Invariance

    [https://arxiv.org/abs/2403.16369](https://arxiv.org/abs/2403.16369)

    提出了一种新的方法，动作双模拟编码，通过递归不变性约束扩展了单步控制性，学习了一个可以平滑折扣远期元素的多步控制度量

    

    强化学习代理使用高维度观测必须能够在许多外源性干扰中识别相关状态特征。一个能够捕捉可控性的表示通过确定影响代理控制的因素来识别这些状态元素。虽然诸如逆动力学和互信息等方法可以捕捉有限数量的时间步的可控性，但捕获长时间元素仍然是一个具有挑战性的问题。短视的可控性可以捕捉代理即将撞向墙壁的瞬间，但不能在代理还有一定距离之时捕捉墙壁的控制相关性。为解决这个问题，我们提出了动作双模拟编码，这是一种受到双模拟不变量假度量启发的方法，它通过递归不变性约束扩展了单步控制性。通过这种方式，动作双模拟学习了一个平滑折扣远期元素的多步控制度量。

    arXiv:2403.16369v1 Announce Type: cross  Abstract: Robust reinforcement learning agents using high-dimensional observations must be able to identify relevant state features amidst many exogeneous distractors. A representation that captures controllability identifies these state elements by determining what affects agent control. While methods such as inverse dynamics and mutual information capture controllability for a limited number of timesteps, capturing long-horizon elements remains a challenging problem. Myopic controllability can capture the moment right before an agent crashes into a wall, but not the control-relevance of the wall while the agent is still some distance away. To address this we introduce action-bisimulation encoding, a method inspired by the bisimulation invariance pseudometric, that extends single-step controllability with a recursive invariance constraint. By doing this, action-bisimulation learns a multi-step controllability metric that smoothly discounts dist
    
[^3]: 局部差分隐私机制的收缩性分析

    Contraction of Locally Differentially Private Mechanisms

    [https://arxiv.org/abs/2210.13386](https://arxiv.org/abs/2210.13386)

    本文研究了局部差分隐私机制的收缩性质, 并给出了输出分布与输入分布之间差异的上界，这对于界定极小极大估计风险非常有用。

    

    我们研究了局部差分隐私机制的收缩性质。具体来说，我们给出了一个关于输入分布之间的差异和输出分布之间的差异的上界，用于度量ϵ-局部差分隐私机制K的输出分布PK和QK之间的差异，分别对应于输入分布P和Q之间的差异。我们的第一个主要技术结果给出了一个关于χ^2-距离χ^2(PK}∥QK)的尖锐上界，该上界与χ^2(P∥Q)和ϵ有关。我们还展示了该结果对一大类距离的上界成立，包括KL-距离和平方Hellinger距离。第二个主要技术结果给出了一个关于χ^2(PK∥QK)的上界，该上界与总变差距离TV(P,Q)和ϵ有关。然后我们利用这些上界建立了van Trees不等式、Le Cam氏不等式、Assouad不等式和互信息方法的局部隐私版本，这些方法对于界定极小极大估计风险非常有用。

    We investigate the contraction properties of locally differentially private mechanisms. More specifically, we derive tight upper bounds on the divergence between $PK$ and $QK$ output distributions of an $\epsilon$-LDP mechanism $K$ in terms of a divergence between the corresponding input distributions $P$ and $Q$, respectively. Our first main technical result presents a sharp upper bound on the $\chi^2$-divergence $\chi^2(PK}\|QK)$ in terms of $\chi^2(P\|Q)$ and $\varepsilon$. We also show that the same result holds for a large family of divergences, including KL-divergence and squared Hellinger distance. The second main technical result gives an upper bound on $\chi^2(PK\|QK)$ in terms of total variation distance $\mathsf{TV}(P, Q)$ and $\epsilon$. We then utilize these bounds to establish locally private versions of the van Trees inequality, Le Cam's, Assouad's, and the mutual information methods, which are powerful tools for bounding minimax estimation risks. These results are shown
    
[^4]: 差异特征未报告对算法公平性的影响

    The Impact of Differential Feature Under-reporting on Algorithmic Fairness. (arXiv:2401.08788v1 [cs.LG])

    [http://arxiv.org/abs/2401.08788](http://arxiv.org/abs/2401.08788)

    本文研究了差异特征未报告对算法公平性的影响，并提出了一个可分析的模型进行刻画。

    

    公共部门的预测风险模型通常使用更完整的行政数据来开发，这些数据对于更大程度依赖公共服务的亚群体更为完整。例如，在美国，对于由医疗补助和医疗保险支持的个人，政府机构常常可以获得有关医疗保健利用的信息，但对于私人保险的人则没有。对公共部门算法的批评指出，差异特征未报告导致算法决策中的不公平。然而，这种数据偏见在技术视角下仍然研究不足。虽然以前的研究已经考察了添加特征噪声和明确标记为缺失的特征对公平性的影响，但缺失指标的数据缺失情况（即差异特征未报告）尚未得到研究的关注。在本研究中，我们提出了一个可分析的差异特征未报告模型，并将其应用于特征未报告对算法公平性的刻画。

    Predictive risk models in the public sector are commonly developed using administrative data that is more complete for subpopulations that more greatly rely on public services. In the United States, for instance, information on health care utilization is routinely available to government agencies for individuals supported by Medicaid and Medicare, but not for the privately insured. Critiques of public sector algorithms have identified such differential feature under-reporting as a driver of disparities in algorithmic decision-making. Yet this form of data bias remains understudied from a technical viewpoint. While prior work has examined the fairness impacts of additive feature noise and features that are clearly marked as missing, the setting of data missingness absent indicators (i.e. differential feature under-reporting) has been lacking in research attention. In this work, we present an analytically tractable model of differential feature under-reporting which we then use to charac
    
[^5]: 去殖民化的人工智能对齐：威色达尔玛、论证和艺术表达

    Decolonial AI Alignment: Vi\'{s}esadharma, Argument, and Artistic Expression. (arXiv:2309.05030v1 [cs.CY])

    [http://arxiv.org/abs/2309.05030](http://arxiv.org/abs/2309.05030)

    本文提出了去殖民化人工智能对齐的三个建议：改变基本道德哲学为达尔玛哲学，允许多元主义的论证传统存在于对齐技术中，以及将价值认识论扩展到超越自然语言中的指令。

    

    先前的研究已经阐明了人工智能（AI）开发和部署的殖民性。然而，这些研究很少涉及到对齐：即基于细致的人类反馈，调整大型语言模型（LLM）的行为与期望值一致。除了其他实践，殖民主义还有一部分是改变被殖民民族的信仰和价值观的历史；而当前的LLM对齐实践正是这一历史的复制。我们建议通过三个提议对AI对齐进行去殖民化：（a）将基本道德哲学从西方哲学转变为达尔玛哲学，（b）在对齐技术中允许论证和多元主义的传统，以及（c）将价值的认识论扩展到超越自然语言中的指令或命令。

    Prior work has explicated the coloniality of artificial intelligence (AI) development and deployment. One process that that work has not engaged with much is alignment: the tuning of large language model (LLM) behavior to be in line with desired values based on fine-grained human feedback. In addition to other practices, colonialism has a history of altering the beliefs and values of colonized peoples; this history is recapitulated in current LLM alignment practices. We suggest that AI alignment be decolonialized using three proposals: (a) changing the base moral philosophy from Western philosophy to dharma, (b) permitting traditions of argument and pluralism in alignment technologies, and (c) expanding the epistemology of values beyond instructions or commandments given in natural language.
    
[^6]: 一个学习受到污染的通用框架：标签噪声、属性噪声等等

    A General Framework for Learning under Corruption: Label Noise, Attribute Noise, and Beyond. (arXiv:2307.08643v1 [cs.LG])

    [http://arxiv.org/abs/2307.08643](http://arxiv.org/abs/2307.08643)

    该研究提出了一个通用框架，在分布层面上对不同类型的数据污染模型进行了形式化分析，并通过分析贝叶斯风险的变化展示了这些污染对标准监督学习的影响。这些发现为进一步研究提供了新的方向和基础。

    

    数据中的污染现象很常见，并且已经在不同的污染模型下进行了广泛研究。尽管如此，对于这些模型之间的关系仍然了解有限，缺乏对污染及其对学习的影响的统一视角。在本研究中，我们通过基于马尔可夫核的一般性和详尽的框架，在分布层面上正式分析了污染模型。我们强调了标签和属性上存在的复杂联合和依赖性污染，这在现有研究中很少触及。此外，我们通过分析贝叶斯风险变化来展示这些污染如何影响标准的监督学习。我们的发现提供了对于“更复杂”污染对学习问题影响的定性洞察，并为未来的定量比较提供了基础。该框架的应用包括污染校正学习，其中包含一个子案例。

    Corruption is frequently observed in collected data and has been extensively studied in machine learning under different corruption models. Despite this, there remains a limited understanding of how these models relate such that a unified view of corruptions and their consequences on learning is still lacking. In this work, we formally analyze corruption models at the distribution level through a general, exhaustive framework based on Markov kernels. We highlight the existence of intricate joint and dependent corruptions on both labels and attributes, which are rarely touched by existing research. Further, we show how these corruptions affect standard supervised learning by analyzing the resulting changes in Bayes Risk. Our findings offer qualitative insights into the consequences of "more complex" corruptions on the learning problem, and provide a foundation for future quantitative comparisons. Applications of the framework include corruption-corrected learning, a subcase of which we 
    
[^7]: 面向非平稳多智能体强化学习的黑盒方法

    A Black-box Approach for Non-stationary Multi-agent Reinforcement Learning. (arXiv:2306.07465v1 [cs.LG])

    [http://arxiv.org/abs/2306.07465](http://arxiv.org/abs/2306.07465)

    本文提出了一种通用的黑盒方法，适用于多种多智能体强化学习问题，可以在非平稳环境下实现低遗憾率的学习。

    

    本文研究了在非平稳多智能体系统中学习均衡的方法，并解决了区别于单智能体学习的挑战。我们重点关注带有赌徒反馈的游戏，其中即使待测试的差距很小，测试一个均衡也可能导致大量的遗憾，并且在静态游戏中存在多个最优解（均衡）会带来额外的难题。为了克服这些障碍，我们提出了一种通用的黑盒方法，适用于广泛的问题，如一般和博弈、潜在博弈和马尔可夫博弈，只要在静态环境下配备适当的学习和测试神谕。当非平稳程度（通过总变化量 $\Delta$ 测量）已知时，我们的算法可以实现 $\widetilde{O}\left(\Delta^{1/4}T^{3/4}\right)$ 的遗憾，当 $\Delta$ 未知时，可以实现 $\widetilde{O}\left(\Delta^{1/5}T^{4/5}\right)$ 的遗憾。

    We investigate learning the equilibria in non-stationary multi-agent systems and address the challenges that differentiate multi-agent learning from single-agent learning. Specifically, we focus on games with bandit feedback, where testing an equilibrium can result in substantial regret even when the gap to be tested is small, and the existence of multiple optimal solutions (equilibria) in stationary games poses extra challenges. To overcome these obstacles, we propose a versatile black-box approach applicable to a broad spectrum of problems, such as general-sum games, potential games, and Markov games, when equipped with appropriate learning and testing oracles for stationary environments. Our algorithms can achieve $\widetilde{O}\left(\Delta^{1/4}T^{3/4}\right)$ regret when the degree of nonstationarity, as measured by total variation $\Delta$, is known, and $\widetilde{O}\left(\Delta^{1/5}T^{4/5}\right)$ regret when $\Delta$ is unknown, where $T$ is the number of rounds. Meanwhile, 
    
[^8]: 自适应深度学习用于非线性时间序列模型

    Adaptive deep learning for nonlinear time series models. (arXiv:2207.02546v2 [math.ST] UPDATED)

    [http://arxiv.org/abs/2207.02546](http://arxiv.org/abs/2207.02546)

    本文提出了一种自适应深度学习方法用于非线性时间序列模型的均值函数估计，证明了稀疏惩罚的深度神经网络估计器在许多非线性自回归模型中达到了极小下界的最优速率。

    

    本文提出了一种基于深度神经网络的非参数自适应估计理论，用于非平稳和非线性的时间序列模型的均值函数估计。我们首先考虑了两种类型的深度神经网络估计器，即非惩罚和稀疏惩罚的估计器，并为一般的非平稳时间序列建立了它们的泛化误差界限。然后，我们推导了估计属于广泛类别的非线性自回归模型（包括非线性广义可加自回归、单指数和阈值自回归模型）均值函数的极小下界。在这些结果的基础上，我们展示了稀疏惩罚的深度神经网络估计器在许多非线性自回归模型中是自适应的，并达到极小下界的最优速率，仅有多对数因子的差距。通过数值模拟，我们证明了深度神经网络方法在估计具有内在低维结构和不连续或粗糙均值函数的非线性自回归模型中的实用性。

    In this paper, we develop a general theory for adaptive nonparametric estimation of the mean function of a non-stationary and nonlinear time series model using deep neural networks (DNNs). We first consider two types of DNN estimators, non-penalized and sparse-penalized DNN estimators, and establish their generalization error bounds for general non-stationary time series. We then derive minimax lower bounds for estimating mean functions belonging to a wide class of nonlinear autoregressive (AR) models that include nonlinear generalized additive AR, single index, and threshold AR models. Building upon the results, we show that the sparse-penalized DNN estimator is adaptive and attains the minimax optimal rates up to a poly-logarithmic factor for many nonlinear AR models. Through numerical simulations, we demonstrate the usefulness of the DNN methods for estimating nonlinear AR models with intrinsic low-dimensional structures and discontinuous or rough mean functions, which is consistent
    

