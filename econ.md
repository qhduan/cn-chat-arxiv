# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Linking Representations with Multimodal Contrastive Learning.](http://arxiv.org/abs/2304.03464) | 本文提出了一种名为CLIPPINGS的多模态框架，用于记录链接。该框架利用深度学习和对比学习的方法，通过端到端训练对称的视觉和语言编码器，在度量空间中学习相近或不同类别的表示方法，用于多个应用场景，如构建全面的补充专利注册表和识别不同社交媒体平台上的个人。 |
| [^2] | [Individual Welfare Analysis: Random Quasilinear Utility, Independence, and Confidence Bounds.](http://arxiv.org/abs/2304.01921) | 本论文提出了一种基于随机准线性效用函数的个体福利分析框架，通过新的置信区间约束求解任意置信水平下的个人消费者福利损失下限，该求解方法简单高效，且适用范围广泛，具有很强的实际应用价值。 |
| [^3] | [Difference-in-Differences with Time-Varying Covariates in the Parallel Trends Assumption.](http://arxiv.org/abs/2202.02903) | 本文研究了有时间变量的异质性对比中的平行趋势假设的识别和估计策略，并提出了一种利用局部投影得到的高维残差之间的差异的新的识别和估计策略，能够在更多情况下具有鲁棒性和高效性，可用于估计存在异质性治疗效应和/或干扰的环境中的因果效应。 |
| [^4] | [Game Transformations That Preserve Nash Equilibria or Best Response Sets.](http://arxiv.org/abs/2111.00076) | 本研究探讨了对N人博弈应用的游戏变换中，哪些变换可以保持最佳反应集或纳什均衡集。我们证明了正仿射变换可以保持最佳反应集。这个研究提供了一个明确的描述，说明哪些游戏变换可以保持最佳反应集或纳什均衡集。 |

# 详细

[^1]: 用多模态对比学习连接表示

    Linking Representations with Multimodal Contrastive Learning. (arXiv:2304.03464v1 [cs.CV])

    [http://arxiv.org/abs/2304.03464](http://arxiv.org/abs/2304.03464)

    本文提出了一种名为CLIPPINGS的多模态框架，用于记录链接。该框架利用深度学习和对比学习的方法，通过端到端训练对称的视觉和语言编码器，在度量空间中学习相近或不同类别的表示方法，用于多个应用场景，如构建全面的补充专利注册表和识别不同社交媒体平台上的个人。

    

    许多应用需要将包含在各种文档数据集中的实例分组成类。最广泛使用的方法不使用深度学习，也不利用文档固有的多模态性质。值得注意的是，记录链接通常被概念化为字符串匹配问题。本研究开发了 CLIPPINGS，一种用于记录链接的多模态框架。CLIPPINGS 采用端到端训练对称的视觉和语言双编码器，通过对比语言-图像预训练进行对齐，学习一个度量空间，其中给定实例的汇总图像-文本表示靠近同一类中的表示，并远离不同类中的表示。在推理时，可以通过从离线示例嵌入索引中检索它们最近的邻居或聚类它们的表示来链接实例。本研究研究了两个具有挑战性的应用：通过将专利与其对应的监管文件链接来构建全面的补充专利注册表，以及在不同的社交媒体平台上识别个人。

    Many applications require grouping instances contained in diverse document datasets into classes. Most widely used methods do not employ deep learning and do not exploit the inherently multimodal nature of documents. Notably, record linkage is typically conceptualized as a string-matching problem. This study develops CLIPPINGS, (Contrastively Linking Pooled Pre-trained Embeddings), a multimodal framework for record linkage. CLIPPINGS employs end-to-end training of symmetric vision and language bi-encoders, aligned through contrastive language-image pre-training, to learn a metric space where the pooled image-text representation for a given instance is close to representations in the same class and distant from representations in different classes. At inference time, instances can be linked by retrieving their nearest neighbor from an offline exemplar embedding index or by clustering their representations. The study examines two challenging applications: constructing comprehensive suppl
    
[^2]: 个体福利分析：随机准线性效用、独立性和置信区间

    Individual Welfare Analysis: Random Quasilinear Utility, Independence, and Confidence Bounds. (arXiv:2304.01921v1 [econ.EM])

    [http://arxiv.org/abs/2304.01921](http://arxiv.org/abs/2304.01921)

    本论文提出了一种基于随机准线性效用函数的个体福利分析框架，通过新的置信区间约束求解任意置信水平下的个人消费者福利损失下限，该求解方法简单高效，且适用范围广泛，具有很强的实际应用价值。

    

    我们介绍了一种新的个体水平福利分析框架。它建立在一个具有准线性效用函数的连续需求参数模型上，允许存在未观测到的个体-产品水平的偏好冲击。我们得出了由于假想价格上涨导致的个人级别消费者福利损失的任何置信水平的下限，并在独立性限制下受新的置信区间约束的可扩展优化问题中求解。这个置信区间非常简单，鲁棒性强，对于弱工具和非线性模型敏感度都很低，并且可能适用于福利分析之外的其他领域。Monte Carlo模拟和两个关于汽油和食品需求的实证应用证明了我们的方法的实效性。

    We introduce a novel framework for individual-level welfare analysis. It builds on a parametric model for continuous demand with a quasilinear utility function, allowing for unobserved individual-product-level preference shocks. We obtain bounds on the individual-level consumer welfare loss at any confidence level due to a hypothetical price increase, solving a scalable optimization problem constrained by a new confidence set under an independence restriction. This confidence set is computationally simple, robust to weak instruments and nonlinearity, and may have applications beyond welfare analysis. Monte Carlo simulations and two empirical applications on gasoline and food demand demonstrate the effectiveness of our method.
    
[^3]: 有时间变量的异质性对比中的平行趋势假设问题

    Difference-in-Differences with Time-Varying Covariates in the Parallel Trends Assumption. (arXiv:2202.02903v2 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2202.02903](http://arxiv.org/abs/2202.02903)

    本文研究了有时间变量的异质性对比中的平行趋势假设的识别和估计策略，并提出了一种利用局部投影得到的高维残差之间的差异的新的识别和估计策略，能够在更多情况下具有鲁棒性和高效性，可用于估计存在异质性治疗效应和/或干扰的环境中的因果效应。

    

    本文研究了异质性对比的识别和估计策略，其中平行趋势假设是在对时间变量和/或时间不变协变量进行条件化后成立的。我们的第一个主要贡献是指出了常用的双向固定效应（TWFE）回归在这种情况下存在一些弱点。除了文献中强调的涉及多个时期和治疗时机变化的问题之外，我们还表明，即使在只有两个时间段的情况下，TWFE回归也不一定能够抵御（i）未接受处理的潜在结果路径取决于时间变量的水平（而不仅仅是时间内变量的变化），（ii）未接受处理的潜在结果路径取决于时间不变的协变量，以及（iii）结果随时间和/或倾向得分的线性条件违反的情况。即使在前三个问题都不成立的情况下，我们还是表明，TWFE回归可能会受到显著的效率损失。我们的第二个主要贡献是提出了一种利用局部投影得到的高维残差之间的差异的新的识别和估计策略。由于构造方式，这种新策略通常对上述所有问题都具有鲁棒性，包括结果或治疗分配规则中的潜在非线性。重要的是，我们的方法可以用现成的软件包实现，也可以用于估计存在异质性治疗效应和/或干扰的环境中的因果效应。

    In this paper, we study difference-in-differences identification and estimation strategies where the parallel trends assumption holds after conditioning on time-varying covariates and/or time-invariant covariates. Our first main contribution is to point out a number of weaknesses of commonly used two-way fixed effects (TWFE) regressions in this context. In addition to issues related to multiple periods and variation in treatment timing that have been emphasized in the literature, we show that, even in the case with only two time periods, TWFE regressions are not generally robust to (i) paths of untreated potential outcomes depending on the level of time-varying covariates (as opposed to only the change in the covariates over time), (ii) paths of untreated potential outcomes depending on time-invariant covariates, and (iii) violations of linearity conditions for outcomes over time and/or the propensity score. Even in cases where none of the previous three issues hold, we show that TWFE 
    
[^4]: 保持纳什均衡或最佳反应集的游戏变换

    Game Transformations That Preserve Nash Equilibria or Best Response Sets. (arXiv:2111.00076v2 [cs.GT] UPDATED)

    [http://arxiv.org/abs/2111.00076](http://arxiv.org/abs/2111.00076)

    本研究探讨了对N人博弈应用的游戏变换中，哪些变换可以保持最佳反应集或纳什均衡集。我们证明了正仿射变换可以保持最佳反应集。这个研究提供了一个明确的描述，说明哪些游戏变换可以保持最佳反应集或纳什均衡集。

    

    在同时非合作博弈的文献中，广泛使用的事实是，效用收益的正仿射（线性）变换既不改变最佳反应集，也不改变纳什均衡集。我们研究了哪些其他游戏变换在应用于任意N人游戏（N≥2）时也具有这两种属性之一：（i）纳什均衡集保持不变；（ii）最佳反应集保持不变。对于以玩家和策略为基础的游戏变换，我们证明（i）意味着（ii），具有属性（ii）的变换必须是正仿射的。得到的等价链明确描述了那些总是保持纳什均衡集（或最佳反应集）的游戏变换。同时，我们获得了正仿射变换类的两个新特征描述。

    In the literature on simultaneous non-cooperative games, it is a widely used fact that a positive affine (linear) transformation of the utility payoffs neither changes the best response sets nor the Nash equilibrium set. We investigate which other game transformations also possess one of these two properties when being applied to an arbitrary N-player game (N >= 2):  (i) The Nash equilibrium set stays the same.  (ii) The best response sets stay the same.  For game transformations that operate player-wise and strategy-wise, we prove that (i) implies (ii) and that transformations with property (ii) must be positive affine. The resulting equivalence chain gives an explicit description of all those game transformations that always preserve the Nash equilibrium set (or, respectively, the best response sets). Simultaneously, we obtain two new characterizations of the class of positive affine transformations.
    

