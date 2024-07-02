# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Enhancing Educational Outcome with Machine Learning: Modeling Friendship Formation, Measuring Peer Effect and Optimizing Class Assignment](https://arxiv.org/abs/2404.02497) | 该论文利用机器学习解决学校班级分配问题，通过友谊预测、同侪影响估计和班级分配优化，发现将学生分成有性别特征的班级能够提高平均同侪影响，并且极端混合的班级分配方法可以改善底部四分之一学生的同侪影响。 |
| [^2] | [The Value of Context: Human versus Black Box Evaluators](https://arxiv.org/abs/2402.11157) | 机器学习算法是标准化的，评估所有个体时通过固定的共变量，而人类评估者通过定制共变量的获取对每个个体进行评估，我们展示在高维数据环境中，上下文的定制化优势。 |
| [^3] | [Bundling Demand in K-12 Broadband Procurement](https://arxiv.org/abs/2402.07277) | 该研究评估了K-12学校在宽带互联网采购中通过捆绑需求的效果。研究发现，参与者的价格平均下降了三分之一，购买的宽带速度增加了六倍。参与学校节省的金额至少等于联邦政府的补贴金额。根据弱假设，参与学校获得了巨大的福利提升。 |
| [^4] | [Entrywise Inference for Causal Panel Data: A Simple and Instance-Optimal Approach.](http://arxiv.org/abs/2401.13665) | 本研究提出了一种在面板数据中进行因果推断的简单且最佳化的方法，通过简单的矩阵代数和奇异值分解来实现高效计算。通过导出非渐近界限，我们证明了逐项误差与高斯变量的适当缩放具有接近性。同时，我们还开发了一个数据驱动程序，用于构建具有预先指定覆盖保证的逐项置信区间。 |
| [^5] | [Analyzing the Reporting Error of Public Transport Trips in the Danish National Travel Survey Using Smart Card Data.](http://arxiv.org/abs/2308.01198) | 本研究使用丹麦智能卡数据和全国出行调查数据，发现在公共交通用户的时间报告中存在中位数为11.34分钟的报告误差。 |
| [^6] | [A Robust Characterization of Nash Equilibrium.](http://arxiv.org/abs/2307.03079) | 本论文通过假设在不同游戏中具有一致的行为，给出了一种鲁棒的纳什均衡特征化方法，并证明了纳什均衡是唯一满足结果主义、一致性和合理性的解概念。该结果适用于各种自然子类的游戏。 |
| [^7] | [Investigating Emergent Goal-Like Behaviour in Large Language Models Using Experimental Economics.](http://arxiv.org/abs/2305.07970) | 本研究探讨了大型语言模型的能力，发现其可以将自然语言描述转化为适当的行为，但在区分细微的合作和竞争水平方面的能力受到限制，为使用LLMs在人类决策制定背景下的伦理意义和局限性做出了贡献。 |
| [^8] | [Strategic Ambiguity in Global Games.](http://arxiv.org/abs/2303.12263) | 论文研究全球博弈中的策略模糊性对玩家行为的影响，发现模糊质量信息下更多玩家选择恒定收益行动，而低质量信息下更多玩家选择前期最优反应。在金融危机应用中，更模糊质量的新闻会引发债务危机，而质量低的新闻会引发货币危机。 |
| [^9] | [Binary response model with many weak instruments.](http://arxiv.org/abs/2201.04811) | 本文提供了一种使用控制函数方法和正则化方案来获得更好内生二进制响应模型估计结果的方法，并应用于研究家庭收入对大学完成的影响。 |
| [^10] | [Overinference from Weak Signals and Underinference from Strong Signals.](http://arxiv.org/abs/2109.09871) | 研究表明，人们在面对弱信号时会存在过度反应，在面对强信号时会存在欠反应，这与信号信息量的认知不精确有关。 |

# 详细

[^1]: 用机器学习提升教育成果：建模友谊形成，衡量同侪影响和优化班级分配

    Enhancing Educational Outcome with Machine Learning: Modeling Friendship Formation, Measuring Peer Effect and Optimizing Class Assignment

    [https://arxiv.org/abs/2404.02497](https://arxiv.org/abs/2404.02497)

    该论文利用机器学习解决学校班级分配问题，通过友谊预测、同侪影响估计和班级分配优化，发现将学生分成有性别特征的班级能够提高平均同侪影响，并且极端混合的班级分配方法可以改善底部四分之一学生的同侪影响。

    

    在这篇论文中，我们研究了学校校长的班级分配问题。我们将问题分为三个阶段：友谊预测、同侪影响评估和班级分配优化。我们建立了一个微观基础模型来模拟友谊形成，并将该模型逼近为一个神经网络。利用预测的友谊概率邻接矩阵，我们改进了传统的线性均值模型并估计了同侪影响。我们提出了一种新的工具以解决友谊选择的内生性问题。估计的同侪影响略大于线性均值模型的估计。利用友谊预测和同侪影响估计结果，我们模拟了所有学生的反事实同侪影响。我们发现将学生分成有性别特征的班级可以将平均同侪影响提高0.02分（在5分制中）。我们还发现极端混合的班级分配方法可以提高底部四分之一学生的同侪影响。

    arXiv:2404.02497v1 Announce Type: new  Abstract: In this paper, we look at a school principal's class assignment problem. We break the problem into three stages (1) friendship prediction (2) peer effect estimation (3) class assignment optimization. We build a micro-founded model for friendship formation and approximate the model as a neural network. Leveraging on the predicted friendship probability adjacent matrix, we improve the traditional linear-in-means model and estimate peer effect. We propose a new instrument to address the friendship selection endogeneity. The estimated peer effect is slightly larger than the linear-in-means model estimate. Using the friendship prediction and peer effect estimation results, we simulate counterfactual peer effects for all students. We find that dividing students into gendered classrooms increases average peer effect by 0.02 point on a scale of 5. We also find that extreme mixing class assignment method improves bottom quartile students' peer ef
    
[^2]: 上下文的价值：人类评估者与黑匣子评估者

    The Value of Context: Human versus Black Box Evaluators

    [https://arxiv.org/abs/2402.11157](https://arxiv.org/abs/2402.11157)

    机器学习算法是标准化的，评估所有个体时通过固定的共变量，而人类评估者通过定制共变量的获取对每个个体进行评估，我们展示在高维数据环境中，上下文的定制化优势。

    

    评估曾经只在人类专家领域内进行（例如医生进行的医学诊断），现在也可以通过机器学习算法进行。这引发了一个新的概念问题：被人类和算法评估之间有什么区别，在什么时候个人应该更喜欢其中一种形式的评估？我们提出了一个理论框架，形式化了这两种评估形式之间的一个关键区别：机器学习算法是标准化的，通过固定的共变量来评估所有个体，而人类评估者则根据个体定制获取哪些共变量。我们的框架定义并分析了这种定制化的优势——上下文的价值，在具有非常高维数据的环境中。我们表明，除非代理人对共变量的联合分布有精确的知识，更多共变量的价值超过了上下文的价值。

    arXiv:2402.11157v1 Announce Type: new  Abstract: Evaluations once solely within the domain of human experts (e.g., medical diagnosis by doctors) can now also be carried out by machine learning algorithms. This raises a new conceptual question: what is the difference between being evaluated by humans and algorithms, and when should an individual prefer one form of evaluation over the other? We propose a theoretical framework that formalizes one key distinction between the two forms of evaluation: Machine learning algorithms are standardized, fixing a common set of covariates by which to assess all individuals, while human evaluators customize which covariates are acquired to each individual. Our framework defines and analyzes the advantage of this customization -- the value of context -- in environments with very high-dimensional data. We show that unless the agent has precise knowledge about the joint distribution of covariates, the value of more covariates exceeds the value of context
    
[^3]: K-12宽带采购中的需求捆绑效应

    Bundling Demand in K-12 Broadband Procurement

    [https://arxiv.org/abs/2402.07277](https://arxiv.org/abs/2402.07277)

    该研究评估了K-12学校在宽带互联网采购中通过捆绑需求的效果。研究发现，参与者的价格平均下降了三分之一，购买的宽带速度增加了六倍。参与学校节省的金额至少等于联邦政府的补贴金额。根据弱假设，参与学校获得了巨大的福利提升。

    

    我们评估了K-12学校通过捆绑需求获得宽带互联网的效果。2014年，新泽西州从分散的采购方式转变为将学校分为四个区域组合的新采购系统。采用事件研究方法，我们发现参与者的价格平均下降了三分之一，购买的宽带速度增加了六倍。我们对该计划导致的学校支出变化进行了界定，发现参与者节省的金额至少等于他们从联邦政府获得的"E-rate"补贴总额。在对需求进行弱假设的情况下，我们表明参与学校获得了巨大的福利提升。

    We evaluate the effects of bundling demand for broadband internet by K-12 schools. In 2014, New Jersey switched from decentralized procurements to a new procurement system that bundled schools into four regional groups. Using an event study approach, we find that, on average, prices for participants decreased by one-third, and broadband speed purchased increased sixfold. We bound the change in school expenditures due to the program and find that participants saved at least as much as their total "E-rate" subsidy from the federal government. Under weak assumptions on demand, we show that participating schools experienced large welfare gains.
    
[^4]: 《面板数据因果推断的逐项推理方法：一种简单且最佳化的方法》

    Entrywise Inference for Causal Panel Data: A Simple and Instance-Optimal Approach. (arXiv:2401.13665v1 [math.ST])

    [http://arxiv.org/abs/2401.13665](http://arxiv.org/abs/2401.13665)

    本研究提出了一种在面板数据中进行因果推断的简单且最佳化的方法，通过简单的矩阵代数和奇异值分解来实现高效计算。通过导出非渐近界限，我们证明了逐项误差与高斯变量的适当缩放具有接近性。同时，我们还开发了一个数据驱动程序，用于构建具有预先指定覆盖保证的逐项置信区间。

    

    在分阶段采用的面板数据中的因果推断中，目标是估计和推导出潜在结果和处理效应的置信区间。我们提出了一种计算效率高的程序，仅涉及简单的矩阵代数和奇异值分解。我们导出了逐项误差的非渐近界限，证明其接近于适当缩放的高斯变量。尽管我们的程序简单，但却是局部最佳化的，因为我们的理论缩放与通过贝叶斯Cram\'{e}r-Rao论证得出的局部实例下界相匹配。利用我们的见解，我们开发了一种数据驱动的程序，用于构建具有预先指定覆盖保证的逐项置信区间。我们的分析基于对矩阵去噪模型应用SVD算法的一般推理工具箱，这可能具有独立的兴趣。

    In causal inference with panel data under staggered adoption, the goal is to estimate and derive confidence intervals for potential outcomes and treatment effects. We propose a computationally efficient procedure, involving only simple matrix algebra and singular value decomposition. We derive non-asymptotic bounds on the entrywise error, establishing its proximity to a suitably scaled Gaussian variable. Despite its simplicity, our procedure turns out to be instance-optimal, in that our theoretical scaling matches a local instance-wise lower bound derived via a Bayesian Cram\'{e}r-Rao argument. Using our insights, we develop a data-driven procedure for constructing entrywise confidence intervals with pre-specified coverage guarantees. Our analysis is based on a general inferential toolbox for the SVD algorithm applied to the matrix denoising model, which might be of independent interest.
    
[^5]: 使用智能卡数据分析丹麦全国出行调查中公共交通的报告误差

    Analyzing the Reporting Error of Public Transport Trips in the Danish National Travel Survey Using Smart Card Data. (arXiv:2308.01198v1 [stat.AP])

    [http://arxiv.org/abs/2308.01198](http://arxiv.org/abs/2308.01198)

    本研究使用丹麦智能卡数据和全国出行调查数据，发现在公共交通用户的时间报告中存在中位数为11.34分钟的报告误差。

    

    家庭出行调查已经用于数十年来收集个人和家庭的出行行为。然而，自我报告的调查存在回忆偏差，因为受访者可能难以准确回忆和报告他们的活动。本研究通过将两个数据源的连续五年数据在个体层面进行匹配（即丹麦国家出行调查和丹麦智能卡系统），来研究全国范围内家庭出行调查中公共交通用户的时间报告误差。调查受访者与智能卡数据中的旅行卡进行匹配，匹配仅基于受访者声明的时空旅行行为。大约70%的受访者成功与智能卡进行匹配。研究结果显示，中位数的时间报告误差为11.34分钟，四分位范围为28.14分钟。此外，进行了统计分析来探索调查响应者特征和报告误差之间的关系。

    Household travel surveys have been used for decades to collect individuals and households' travel behavior. However, self-reported surveys are subject to recall bias, as respondents might struggle to recall and report their activities accurately. This study addresses examines the time reporting error of public transit users in a nationwide household travel survey by matching, at the individual level, five consecutive years of data from two sources, namely the Danish National Travel Survey (TU) and the Danish Smart Card system (Rejsekort). Survey respondents are matched with travel cards from the Rejsekort data solely based on the respondents' declared spatiotemporal travel behavior. Approximately, 70% of the respondents were successfully matched with Rejsekort travel cards. The findings reveal a median time reporting error of 11.34 minutes, with an Interquartile Range of 28.14 minutes. Furthermore, a statistical analysis was performed to explore the relationships between the survey res
    
[^6]: 一种鲁棒的纳什均衡特征化方法

    A Robust Characterization of Nash Equilibrium. (arXiv:2307.03079v1 [econ.TH])

    [http://arxiv.org/abs/2307.03079](http://arxiv.org/abs/2307.03079)

    本论文通过假设在不同游戏中具有一致的行为，给出了一种鲁棒的纳什均衡特征化方法，并证明了纳什均衡是唯一满足结果主义、一致性和合理性的解概念。该结果适用于各种自然子类的游戏。

    

    我们通过假设在不同游戏中具有一致的行为来给出纳什均衡的鲁棒特征化方法：纳什均衡是唯一满足结果主义、一致性和合理性的解概念。因此，每个均衡改进方法都至少违反其中一种性质。我们还证明，每个近似满足结果主义、一致性和合理性的解概念都会产生近似的纳什均衡。通过增加公理的逼近程度，后者的逼近程度可以任意提高。该结果适用于两人零和游戏、势函数游戏和图形游戏等各种自然子类的游戏。

    We give a robust characterization of Nash equilibrium by postulating coherent behavior across varying games: Nash equilibrium is the only solution concept that satisfies consequentialism, consistency, and rationality. As a consequence, every equilibrium refinement violates at least one of these properties. We moreover show that every solution concept that approximately satisfies consequentialism, consistency, and rationality returns approximate Nash equilibria. The latter approximation can be made arbitrarily good by increasing the approximation of the axioms. This result extends to various natural subclasses of games such as two-player zero-sum games, potential games, and graphical games.
    
[^7]: 利用实验经济学研究大型语言模型中出现的类似目标行为

    Investigating Emergent Goal-Like Behaviour in Large Language Models Using Experimental Economics. (arXiv:2305.07970v1 [cs.GT])

    [http://arxiv.org/abs/2305.07970](http://arxiv.org/abs/2305.07970)

    本研究探讨了大型语言模型的能力，发现其可以将自然语言描述转化为适当的行为，但在区分细微的合作和竞争水平方面的能力受到限制，为使用LLMs在人类决策制定背景下的伦理意义和局限性做出了贡献。

    

    本研究探讨了大型语言模型（LLMs），特别是GPT-3.5，实现合作、竞争、利他和自私行为的自然语言描述在社会困境下的能力。我们聚焦于迭代囚徒困境，这是一个非零和互动的经典例子，但我们的更广泛研究计划包括一系列实验经济学场景，包括最后通牒博弈、独裁者博弈和公共物品游戏。使用被试内实验设计，我们运用不同的提示信息实例化由LLM生成的智能体，表达不同的合作和竞争立场。我们评估了智能体在迭代囚徒困境中的合作水平，同时考虑到它们对合作或出尔反尔的伙伴行动的响应。我们的结果表明，LLMs在某种程度上可以将利他和自私的自然语言描述转化为适当的行为，但展示出区分合作和竞争水平的能力有限。总体而言，我们的研究为在人类决策制定的背景下使用LLMs的伦理意义和局限性提供了证据。

    In this study, we investigate the capacity of large language models (LLMs), specifically GPT-3.5, to operationalise natural language descriptions of cooperative, competitive, altruistic, and self-interested behavior in social dilemmas. Our focus is on the iterated Prisoner's Dilemma, a classic example of a non-zero-sum interaction, but our broader research program encompasses a range of experimental economics scenarios, including the ultimatum game, dictator game, and public goods game. Using a within-subject experimental design, we instantiated LLM-generated agents with various prompts that conveyed different cooperative and competitive stances. We then assessed the agents' level of cooperation in the iterated Prisoner's Dilemma, taking into account their responsiveness to the cooperative or defection actions of their partners. Our results provide evidence that LLMs can translate natural language descriptions of altruism and selfishness into appropriate behaviour to some extent, but e
    
[^8]: 全球博弈中的策略模糊性

    Strategic Ambiguity in Global Games. (arXiv:2303.12263v1 [econ.TH])

    [http://arxiv.org/abs/2303.12263](http://arxiv.org/abs/2303.12263)

    论文研究全球博弈中的策略模糊性对玩家行为的影响，发现模糊质量信息下更多玩家选择恒定收益行动，而低质量信息下更多玩家选择前期最优反应。在金融危机应用中，更模糊质量的新闻会引发债务危机，而质量低的新闻会引发货币危机。

    

    在具有不完全和模糊信息的博弈中，理性行为不仅取决于基本模糊（关于状态的模糊性）而且取决于策略模糊（关于他人行为的模糊性）。我们研究了策略模糊在全球博弈中的影响。模糊质量信息使更多的玩家选择产生恒定收益的行动，而（明确的）低质量信息使更多的玩家选择对对手行动的统一信念做出前期最优反应。如果前期最优反应的行动产生恒定收益，则足够模糊的质量信息会诱导出一种唯一的均衡，而足够低质量的信息会产生多个均衡。在金融危机的应用中，我们证明了更模糊质量的新闻会引发债务展期危机，而质量较低的新闻会引发货币危机。

    In games with incomplete and ambiguous information, rational behavior depends not only on fundamental ambiguity (ambiguity about states) but also on strategic ambiguity (ambiguity about others' actions). We study the impact of strategic ambiguity in global games. Ambiguous-quality information makes more players choose an action yielding a constant payoff, whereas (unambiguous) low-quality information makes more players choose an ex-ante best response to the uniform belief over the opponents' actions. If the ex-ante best-response action yields a constant payoff, sufficiently ambiguous-quality information induces a unique equilibrium, whereas sufficiently low-quality information generates multiple equilibria. In applications to financial crises, we demonstrate that news of more ambiguous quality triggers a debt rollover crisis, whereas news of less ambiguous quality triggers a currency crisis.
    
[^9]: 多个弱工具的二进制响应模型

    Binary response model with many weak instruments. (arXiv:2201.04811v3 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2201.04811](http://arxiv.org/abs/2201.04811)

    本文提供了一种使用控制函数方法和正则化方案来获得更好内生二进制响应模型估计结果的方法，并应用于研究家庭收入对大学完成的影响。

    

    本文考虑了具有许多弱工具的内生二进制响应模型。我们采用控制函数方法和正则化方案，在存在许多弱工具的情况下获得更好的内生二进制响应模型估计结果。提供了两个一致且渐近正态分布的估计器，分别称为正则化条件最大似然估计器（RCMLE）和正则化非线性最小二乘估计器（RNLSE）。Monte Carlo模拟表明，所提出的估计量在存在许多弱工具时优于现有的估计量。我们应用估计方法研究家庭收入对大学完成的影响。

    This paper considers an endogenous binary response model with many weak instruments. We in the current paper employ a control function approach and a regularization scheme to obtain better estimation results for the endogenous binary response model in the presence of many weak instruments. Two consistent and asymptotically normally distributed estimators are provided, each of which is called a regularized conditional maximum likelihood estimator (RCMLE) and a regularized nonlinear least square estimator (RNLSE) respectively. Monte Carlo simulations show that the proposed estimators outperform the existing estimators when many weak instruments are present. We apply our estimation method to study the effect of family income on college completion.
    
[^10]: 弱信号的过度推断及强信号的欠推断

    Overinference from Weak Signals and Underinference from Strong Signals. (arXiv:2109.09871v4 [econ.GN] UPDATED)

    [http://arxiv.org/abs/2109.09871](http://arxiv.org/abs/2109.09871)

    研究表明，人们在面对弱信号时会存在过度反应，在面对强信号时会存在欠反应，这与信号信息量的认知不精确有关。

    

    本文研究了关于信息量大小对信号更新的过度反应和欠反应的影响。尽管大量文献研究了高信息量信号的信念更新，但在现实世界中，人们往往面对一大堆弱信号。我们使用了严密的实验和新的来自博彩和金融市场的经验证据证明，信号强度对更新行为有着实质性的影响：在各领域中，我们得到了一致且稳健的结果，即对于弱信号存在过度反应，而对于强信号存在欠反应。这两种结果都与关于信号信息量的认知不精确的简单理论相吻合。我们的框架和发现可帮助协调实验和经验文献中表现出的明显矛盾之处。

    We study how overreaction and underreaction to signals depend on their informativeness. While a large literature has studied belief updating in response to highly informative signals, people in important real-world settings are often faced with a steady stream of weak signals. We use a tightly controlled experiment and new empirical evidence from betting and financial markets to demonstrate that updating behavior differs meaningfully by signal strength: across domains, our consistent and robust finding is overreaction to weak signals and underreaction to strong signals. Both sets of results align well with a simple theory of cognitive imprecision about signal informativeness. Our framework and findings can help harmonize apparently contradictory results from the experimental and empirical literatures.
    

