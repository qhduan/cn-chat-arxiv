# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Enhancing Educational Outcome with Machine Learning: Modeling Friendship Formation, Measuring Peer Effect and Optimizing Class Assignment](https://arxiv.org/abs/2404.02497) | 该论文利用机器学习解决学校班级分配问题，通过友谊预测、同侪影响估计和班级分配优化，发现将学生分成有性别特征的班级能够提高平均同侪影响，并且极端混合的班级分配方法可以改善底部四分之一学生的同侪影响。 |
| [^2] | [Extract Mechanisms from Heterogeneous Effects: Identification Strategy for Mediation Analysis](https://arxiv.org/abs/2403.04131) | 该论文开发了一种新的识别策略，通过将总处理效应进行分解，将复杂的中介问题转化为简单的线性回归问题，实现了因果效应和中介效应的同时估计，建立了新的因果中介和因果调节之间的联系。 |
| [^3] | [The Gerber-Shiu Expected Discounted Penalty Function: An Application to Poverty Trapping](https://arxiv.org/abs/2402.11715) | 本文引入了类似于经典Gerber-Shiu预期折扣罚函数的函数，用于分析家庭资本陷入贫困陷阱时的特征，并获得了一些经典风险理论中通常研究的量的闭合形式表达式。 |
| [^4] | [Climate, Crops, and Postharvest Conflict.](http://arxiv.org/abs/2311.16370) | 本研究提供了关于气候冲击对冲突的新证据，表明El Niño事件对农作物生长季和采摘后期的农田中的政治暴力具有负面影响。具体而言，中强度El Niño事件导致的海面温度上升1°C将至少减少三个百分点的政治暴力与针对平民的冲突。这一发现支持了农业作为连接气候冲击和政治暴力之间关键途径的观点。 |
| [^5] | [On the Efficiency of Finely Stratified Experiments.](http://arxiv.org/abs/2307.15181) | 本文研究了在实验分析中涉及的大类处理效应参数的有效估计，包括平均处理效应、分位数处理效应、局部平均处理效应等。 |

# 详细

[^1]: 用机器学习提升教育成果：建模友谊形成，衡量同侪影响和优化班级分配

    Enhancing Educational Outcome with Machine Learning: Modeling Friendship Formation, Measuring Peer Effect and Optimizing Class Assignment

    [https://arxiv.org/abs/2404.02497](https://arxiv.org/abs/2404.02497)

    该论文利用机器学习解决学校班级分配问题，通过友谊预测、同侪影响估计和班级分配优化，发现将学生分成有性别特征的班级能够提高平均同侪影响，并且极端混合的班级分配方法可以改善底部四分之一学生的同侪影响。

    

    在这篇论文中，我们研究了学校校长的班级分配问题。我们将问题分为三个阶段：友谊预测、同侪影响评估和班级分配优化。我们建立了一个微观基础模型来模拟友谊形成，并将该模型逼近为一个神经网络。利用预测的友谊概率邻接矩阵，我们改进了传统的线性均值模型并估计了同侪影响。我们提出了一种新的工具以解决友谊选择的内生性问题。估计的同侪影响略大于线性均值模型的估计。利用友谊预测和同侪影响估计结果，我们模拟了所有学生的反事实同侪影响。我们发现将学生分成有性别特征的班级可以将平均同侪影响提高0.02分（在5分制中）。我们还发现极端混合的班级分配方法可以提高底部四分之一学生的同侪影响。

    arXiv:2404.02497v1 Announce Type: new  Abstract: In this paper, we look at a school principal's class assignment problem. We break the problem into three stages (1) friendship prediction (2) peer effect estimation (3) class assignment optimization. We build a micro-founded model for friendship formation and approximate the model as a neural network. Leveraging on the predicted friendship probability adjacent matrix, we improve the traditional linear-in-means model and estimate peer effect. We propose a new instrument to address the friendship selection endogeneity. The estimated peer effect is slightly larger than the linear-in-means model estimate. Using the friendship prediction and peer effect estimation results, we simulate counterfactual peer effects for all students. We find that dividing students into gendered classrooms increases average peer effect by 0.02 point on a scale of 5. We also find that extreme mixing class assignment method improves bottom quartile students' peer ef
    
[^2]: 从异质效应中提取机制：中介分析的识别策略

    Extract Mechanisms from Heterogeneous Effects: Identification Strategy for Mediation Analysis

    [https://arxiv.org/abs/2403.04131](https://arxiv.org/abs/2403.04131)

    该论文开发了一种新的识别策略，通过将总处理效应进行分解，将复杂的中介问题转化为简单的线性回归问题，实现了因果效应和中介效应的同时估计，建立了新的因果中介和因果调节之间的联系。

    

    理解因果机制对于解释和概括经验现象至关重要。因果中介分析提供了量化中介效应的统计技术。然而，现有方法通常需要强大的识别假设或复杂的研究设计。我们开发了一种新的识别策略，简化了这些假设，实现了因果效应和中介效应的同时估计。该策略基于总处理效应的新型分解，将具有挑战性的中介问题转化为简单的线性回归问题。新方法建立了因果中介和因果调节之间的新联系。我们讨论了几种研究设计和估计器，以增加我们的识别策略在各种实证研究中的可用性。我们通过在实验中估计因果中介效应来演示我们方法的应用。

    arXiv:2403.04131v1 Announce Type: cross  Abstract: Understanding causal mechanisms is essential for explaining and generalizing empirical phenomena. Causal mediation analysis offers statistical techniques to quantify mediation effects. However, existing methods typically require strong identification assumptions or sophisticated research designs. We develop a new identification strategy that simplifies these assumptions, enabling the simultaneous estimation of causal and mediation effects. The strategy is based on a novel decomposition of total treatment effects, which transforms the challenging mediation problem into a simple linear regression problem. The new method establishes a new link between causal mediation and causal moderation. We discuss several research designs and estimators to increase the usability of our identification strategy for a variety of empirical studies. We demonstrate the application of our method by estimating the causal mediation effect in experiments concer
    
[^3]: 《Gerber-Shiu预期折扣罚函数：贫困陷阱的应用》

    The Gerber-Shiu Expected Discounted Penalty Function: An Application to Poverty Trapping

    [https://arxiv.org/abs/2402.11715](https://arxiv.org/abs/2402.11715)

    本文引入了类似于经典Gerber-Shiu预期折扣罚函数的函数，用于分析家庭资本陷入贫困陷阱时的特征，并获得了一些经典风险理论中通常研究的量的闭合形式表达式。

    

    在这篇文章中，我们考虑了一个具有确定性增长和按比例损失的风险过程，以模拟一个家庭的资本。我们的工作集中在分析这样一个过程的陷阱时间，当一个家庭的资本水平陷入贫困区域时就会发生陷阱，这是一个很难在没有外部帮助的情况下摆脱的区域。我们引入了一个类似于经典的Gerber-Shiu预期折扣罚函数的函数，它包含了关于陷阱时间、陷入陷阱前立即剩余资本和陷阱时的资本赤字的信息。鉴于在经历资本损失时的剩余资本比例为$Beta(\alpha,1)$分布，我们得到了经典风险理论中通常研究的量的闭合形式表达式，包括陷阱时间的拉普拉斯变换和陷阱时的资本赤字的分布。特别地，我们得到了一个模型。

    arXiv:2402.11715v1 Announce Type: new  Abstract: In this article, we consider a risk process with deterministic growth and prorated losses to model the capital of a household. Our work focuses on the analysis of the trapping time of such a process, where trapping occurs when a household's capital level falls into the poverty area, a region from which it is difficult to escape without external help. A function analogous to the classical Gerber-Shiu expected discounted penalty function is introduced, which incorporates information on the trapping time, the capital surplus immediately before trapping and the capital deficit at trapping. Given that the remaining proportion of capital upon experiencing a capital loss is $Beta(\alpha,1)-$distributed, closed-form expressions are obtained for quantities typically studied in classical risk theory, including the Laplace transform of the trapping time and the distribution of the capital deficit at trapping. In particular, we derive a model belong
    
[^4]: 气候、农作物与采摘后冲突

    Climate, Crops, and Postharvest Conflict. (arXiv:2311.16370v3 [econ.GN] UPDATED)

    [http://arxiv.org/abs/2311.16370](http://arxiv.org/abs/2311.16370)

    本研究提供了关于气候冲击对冲突的新证据，表明El Niño事件对农作物生长季和采摘后期的农田中的政治暴力具有负面影响。具体而言，中强度El Niño事件导致的海面温度上升1°C将至少减少三个百分点的政治暴力与针对平民的冲突。这一发现支持了农业作为连接气候冲击和政治暴力之间关键途径的观点。

    

    我提出了气候冲击对冲突的影响的新证据。利用覆盖1997年至2023年整个非洲大陆的详细冲突和天气数据，我发现在农作物生长季遭受El Niño事件对夏季采摘后期农田的政治暴力具有负差异效应。相对于长期均值，热带太平洋海面温度上升1°C（典型的中强度El Niño事件的代表）至少导致政治暴力与针对平民的冲突减少三个百分点。这一主要发现通过一系列稳健性检验得到支持，支持了农业是连接气候冲击与政治暴力之间的关键途径，而贪婪是关键动机。令人放心的是，当我使用更适合揭示提出的机制的数据子集时，估计效应的幅度几乎翻了一番，而这种效应仅仅存在于农作物生长季和夏季采摘后期的农田中。

    I present new evidence of the effects of climate shocks on conflict. Using granular conflict and weather data covering the entire continent of Africa from 1997 to 2023, I find that exposure to El Ni\~no events during the crop-growing season has a negative differential effect on political violence against civilians in croplands during the early postharvest season. A 1{\deg}C warming of the sea surface temperature in the tropical Pacific Ocean relative to its long-run mean, a typical proxy for a moderate-strength El Ni\~no event, results in at least a three percent reduction in political violence with civilian targeting. This main finding, backed by a series of robustness checks, supports the idea that agriculture is the key channel and rapacity is the key motive connecting climatic shocks and political violence. Reassuringly, the magnitude of the estimated effect nearly doubles when I use subsets of data that are better suited to unveiling the proposed mechanism, and the effect only man
    
[^5]: 关于细分实验效率的研究

    On the Efficiency of Finely Stratified Experiments. (arXiv:2307.15181v1 [econ.EM])

    [http://arxiv.org/abs/2307.15181](http://arxiv.org/abs/2307.15181)

    本文研究了在实验分析中涉及的大类处理效应参数的有效估计，包括平均处理效应、分位数处理效应、局部平均处理效应等。

    

    本文研究了在实验分析中涉及的大类处理效应参数的有效估计。在这里，效率是指对于一类广泛的处理分配方案而言的，其中任何单位被分配到处理的边际概率等于预先指定的值，例如一半。重要的是，我们不要求处理状态是以i.i.d.的方式分配的，因此可以适应实践中使用的复杂处理分配方案，如分层随机化和匹配对。所考虑的参数类别是可以表示为已知观测数据的一个已知函数的期望的约束的解的那些参数，其中可能包括处理分配边际概率的预先指定值。我们证明了这类参数包括平均处理效应、分位数处理效应、局部平均处理效应等。

    This paper studies the efficient estimation of a large class of treatment effect parameters that arise in the analysis of experiments. Here, efficiency is understood to be with respect to a broad class of treatment assignment schemes for which the marginal probability that any unit is assigned to treatment equals a pre-specified value, e.g., one half. Importantly, we do not require that treatment status is assigned in an i.i.d. fashion, thereby accommodating complicated treatment assignment schemes that are used in practice, such as stratified block randomization and matched pairs. The class of parameters considered are those that can be expressed as the solution to a restriction on the expectation of a known function of the observed data, including possibly the pre-specified value for the marginal probability of treatment assignment. We show that this class of parameters includes, among other things, average treatment effects, quantile treatment effects, local average treatment effect
    

