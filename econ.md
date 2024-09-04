# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Enhancing Educational Outcome with Machine Learning: Modeling Friendship Formation, Measuring Peer Effect and Optimizing Class Assignment](https://arxiv.org/abs/2404.02497) | 该论文利用机器学习解决学校班级分配问题，通过友谊预测、同侪影响估计和班级分配优化，发现将学生分成有性别特征的班级能够提高平均同侪影响，并且极端混合的班级分配方法可以改善底部四分之一学生的同侪影响。 |
| [^2] | [GPT has become financially literate: Insights from financial literacy tests of GPT and a preliminary test of how people use it as a source of advice.](http://arxiv.org/abs/2309.00649) | GPT通过金融素养测试显示出具备成为大众金融机器顾问的能力，其中基于GPT-4的ChatGPT几乎完美地得分99%，揭示了金融素养正在成为最先进模型的新兴能力。 |
| [^3] | [Scalable Estimation of Multinomial Response Models with Uncertain Consideration Sets.](http://arxiv.org/abs/2308.12470) | 这篇论文提出了一种克服估计多项式响应模型中指数级支持问题的方法，通过使用基于列联表概率分布的考虑集概率模型。 |
| [^4] | [Should the Timing of Inspections be Predictable?.](http://arxiv.org/abs/2304.01385) | 该论文研究了长期项目中雇佣代理人的检查策略。周期性检查适用于代理人的行为主要加速突破，随机检查适用于代理人的行为主要延迟崩溃。代理人的行为决定了他在惩罚时间方面的风险态度。 |
| [^5] | [Long-term Causal Inference Under Persistent Confounding via Data Combination.](http://arxiv.org/abs/2202.07234) | 本研究通过数据组合解决了长期治疗效果识别和估计中的持续未测量混淆因素挑战，并提出了三种新的识别策略和估计器。 |

# 详细

[^1]: 用机器学习提升教育成果：建模友谊形成，衡量同侪影响和优化班级分配

    Enhancing Educational Outcome with Machine Learning: Modeling Friendship Formation, Measuring Peer Effect and Optimizing Class Assignment

    [https://arxiv.org/abs/2404.02497](https://arxiv.org/abs/2404.02497)

    该论文利用机器学习解决学校班级分配问题，通过友谊预测、同侪影响估计和班级分配优化，发现将学生分成有性别特征的班级能够提高平均同侪影响，并且极端混合的班级分配方法可以改善底部四分之一学生的同侪影响。

    

    在这篇论文中，我们研究了学校校长的班级分配问题。我们将问题分为三个阶段：友谊预测、同侪影响评估和班级分配优化。我们建立了一个微观基础模型来模拟友谊形成，并将该模型逼近为一个神经网络。利用预测的友谊概率邻接矩阵，我们改进了传统的线性均值模型并估计了同侪影响。我们提出了一种新的工具以解决友谊选择的内生性问题。估计的同侪影响略大于线性均值模型的估计。利用友谊预测和同侪影响估计结果，我们模拟了所有学生的反事实同侪影响。我们发现将学生分成有性别特征的班级可以将平均同侪影响提高0.02分（在5分制中）。我们还发现极端混合的班级分配方法可以提高底部四分之一学生的同侪影响。

    arXiv:2404.02497v1 Announce Type: new  Abstract: In this paper, we look at a school principal's class assignment problem. We break the problem into three stages (1) friendship prediction (2) peer effect estimation (3) class assignment optimization. We build a micro-founded model for friendship formation and approximate the model as a neural network. Leveraging on the predicted friendship probability adjacent matrix, we improve the traditional linear-in-means model and estimate peer effect. We propose a new instrument to address the friendship selection endogeneity. The estimated peer effect is slightly larger than the linear-in-means model estimate. Using the friendship prediction and peer effect estimation results, we simulate counterfactual peer effects for all students. We find that dividing students into gendered classrooms increases average peer effect by 0.02 point on a scale of 5. We also find that extreme mixing class assignment method improves bottom quartile students' peer ef
    
[^2]: GPT已经具备了金融素养：来自GPT金融素养测试的见解以及人们使用其作为咨询来源的初步测试

    GPT has become financially literate: Insights from financial literacy tests of GPT and a preliminary test of how people use it as a source of advice. (arXiv:2309.00649v1 [cs.CL])

    [http://arxiv.org/abs/2309.00649](http://arxiv.org/abs/2309.00649)

    GPT通过金融素养测试显示出具备成为大众金融机器顾问的能力，其中基于GPT-4的ChatGPT几乎完美地得分99%，揭示了金融素养正在成为最先进模型的新兴能力。

    

    通过使用金融素养测试，我们评估了GPT（一种大型语言模型）作为大众金融机器顾问的能力。基于GPT-3.5的Davinci和ChatGPT分别在金融素养测试中得分为66%和65%，而基于GPT-4的ChatGPT几乎完美地得到了99%的分数，这表明金融素养正在成为最先进模型的新兴能力。我们使用Judge-Advisor系统和一个储蓄困境来说明研究人员如何评估大型语言模型提供的建议利用情况。我们还提出了一些未来研究的方向。

    We assess the ability of GPT -- a large language model -- to serve as a financial robo-advisor for the masses, by using a financial literacy test. Davinci and ChatGPT based on GPT-3.5 score 66% and 65% on the financial literacy test, respectively, compared to a baseline of 33%. However, ChatGPT based on GPT-4 achieves a near-perfect 99% score, pointing to financial literacy becoming an emergent ability of state-of-the-art models. We use the Judge-Advisor System and a savings dilemma to illustrate how researchers might assess advice-utilization from large language models. We also present a number of directions for future research.
    
[^3]: 可伸缩估计具有不确定的选项集的多项式响应模型

    Scalable Estimation of Multinomial Response Models with Uncertain Consideration Sets. (arXiv:2308.12470v1 [stat.ME])

    [http://arxiv.org/abs/2308.12470](http://arxiv.org/abs/2308.12470)

    这篇论文提出了一种克服估计多项式响应模型中指数级支持问题的方法，通过使用基于列联表概率分布的考虑集概率模型。

    

    在交叉或纵向数据的无序多项式响应模型拟合中的一个标准假设是，响应来自于相同的J个类别集合。然而，当响应度量主体做出的选择时，更适合假设多项式响应的分布是在主体特定的考虑集条件下，其中这个考虑集是从{1,2, ..., J}的幂集中抽取的。由于这个幂集的基数在J中是指数级的，一般来说估计是无法实现的。在本文中，我们提供了一种克服这个问题的方法。这种方法中的一个关键步骤是基于在列联表上的概率分布的一般表示的考虑集的概率模型。尽管这个分布的支持是指数级大的，但给定参数的考虑集的后验分布通常是稀疏的。

    A standard assumption in the fitting of unordered multinomial response models for J mutually exclusive nominal categories, on cross-sectional or longitudinal data, is that the responses arise from the same set of J categories between subjects. However, when responses measure a choice made by the subject, it is more appropriate to assume that the distribution of multinomial responses is conditioned on a subject-specific consideration set, where this consideration set is drawn from the power set of {1,2,...,J}. Because the cardinality of this power set is exponential in J, estimation is infeasible in general. In this paper, we provide an approach to overcoming this problem. A key step in the approach is a probability model over consideration sets, based on a general representation of probability distributions on contingency tables. Although the support of this distribution is exponentially large, the posterior distribution over consideration sets given parameters is typically sparse, and
    
[^4]: 检查时间可预测吗？

    Should the Timing of Inspections be Predictable?. (arXiv:2304.01385v1 [econ.TH])

    [http://arxiv.org/abs/2304.01385](http://arxiv.org/abs/2304.01385)

    该论文研究了长期项目中雇佣代理人的检查策略。周期性检查适用于代理人的行为主要加速突破，随机检查适用于代理人的行为主要延迟崩溃。代理人的行为决定了他在惩罚时间方面的风险态度。

    

    一位委托人聘请一个代理人长期从事工作，该工作会在某个时刻突破或者崩溃。在每个时刻，代理人会私下选择工作或逃避责任。工作可以增加突破到来的速度并减少崩溃到来的速度。为了激励代理人工作，委托人会进行昂贵的检查，如果发现代理人在逃避责任，就会解雇他。我们确定了委托人的最佳检查策略。当工作主要加速突破时，周期性检查是最优的。当工作主要延迟崩溃时，随机检查是最优的。至关重要的是，代理人的行为决定了他在惩罚时间方面的风险态度。

    A principal hires an agent to work on a long-term project that culminates in a breakthrough or a breakdown. At each time, the agent privately chooses to work or shirk. Working increases the arrival rate of breakthroughs and decreases the arrival rate of breakdowns. To motivate the agent to work, the principal conducts costly inspections. She fires the agent if shirking is detected. We characterize the principal's optimal inspection policy. Periodic inspections are optimal if work primarily speeds up breakthroughs. Random inspections are optimal if work primarily delays breakdowns. Crucially, the agent's actions determine his risk-attitude over the timing of punishments.
    
[^5]: 长期持续混淆情况下的因果推断与数据组合研究

    Long-term Causal Inference Under Persistent Confounding via Data Combination. (arXiv:2202.07234v3 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2202.07234](http://arxiv.org/abs/2202.07234)

    本研究通过数据组合解决了长期治疗效果识别和估计中的持续未测量混淆因素挑战，并提出了三种新的识别策略和估计器。

    

    我们研究了当实验数据和观察数据同时存在时，长期治疗效果的识别和估计问题。由于长期结果仅在长时间延迟后才观察到，在实验数据中无法测量，但在观察数据中有记录。然而，这两种类型的数据都包含对一些短期结果的观察。在本文中，我们独特地解决了持续未测量混淆因素的挑战，即一些未测量混淆因素可以同时影响治疗、短期结果和长期结果，而这会使得之前文献中的识别策略无效。为了解决这个挑战，我们利用多个短期结果的连续结构，为平均长期治疗效果提出了三种新的识别策略。我们进一步提出了三种对应的估计器，并证明了它们的渐近一致性和渐近正态性。最后，我们将我们的方法应用于估计长期治疗效果。

    We study the identification and estimation of long-term treatment effects when both experimental and observational data are available. Since the long-term outcome is observed only after a long delay, it is not measured in the experimental data, but only recorded in the observational data. However, both types of data include observations of some short-term outcomes. In this paper, we uniquely tackle the challenge of persistent unmeasured confounders, i.e., some unmeasured confounders that can simultaneously affect the treatment, short-term outcomes and the long-term outcome, noting that they invalidate identification strategies in previous literature. To address this challenge, we exploit the sequential structure of multiple short-term outcomes, and develop three novel identification strategies for the average long-term treatment effect. We further propose three corresponding estimators and prove their asymptotic consistency and asymptotic normality. We finally apply our methods to esti
    

