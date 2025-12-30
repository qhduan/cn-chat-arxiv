# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Algorithmic Collusion and Price Discrimination: The Over-Usage of Data](https://arxiv.org/abs/2403.06150) | 本文通过模拟研究探讨了算法性暗中勾结和价格歧视之间的相互作用，发现了导致共谋结果的两种新机制。 |
| [^2] | [Estimating Effects of Long-Term Treatments.](http://arxiv.org/abs/2308.08152) | 本论文介绍了一个纵向替代框架，用于准确估计长期治疗的效果。论文通过分解长期治疗效果为一系列函数，考虑用户属性、短期中间指标和治疗分配等因素。 |
| [^3] | [Unconfoundedness with Network Interference.](http://arxiv.org/abs/2211.07823) | 本文研究使用单个大型网络的观测数据进行非参数估计，利用图神经网络估计高维烦恼函数，研究治疗效应和溢出效应。 |
| [^4] | [Nonparametric Treatment Effect Identification in School Choice.](http://arxiv.org/abs/2112.03872) | 本论文研究了集中学校分配中的非参数化治疗效果识别和估计方法，通过识别原子治疗效应，该研究揭示了在回归不连续和抽签驱动的变异下的学校选择的异质性和重要性。 |

# 详细

[^1]: 算法共谋与价格歧视：数据过度使用

    Algorithmic Collusion and Price Discrimination: The Over-Usage of Data

    [https://arxiv.org/abs/2403.06150](https://arxiv.org/abs/2403.06150)

    本文通过模拟研究探讨了算法性暗中勾结和价格歧视之间的相互作用，发现了导致共谋结果的两种新机制。

    

    随着企业定价策略越来越依赖算法，两个问题引起了人们的关注：算法性暗中勾结和价格歧视。本文通过模拟研究了这两个问题之间的互动。每个时期，一个新的买家带着独立同分布的支付意愿到来，每家公司观察到关于支付意愿的私人信号，采用Q学习算法来设定价格。我们记录下两种导致共谋结果的新机制。在信息不对称的情况下，具有信息优势的算法采取了“诱饵和限制-利用”策略，通过设置更高的价格牺牲了一些信号上的利润，同时通过设置更低的价格在其余信号上利用有限的利润。在对称信息结构下，对一些信号的竞争促进了对其余信号上的超竞争价格的收敛。算法往往会导致

    arXiv:2403.06150v1 Announce Type: new  Abstract: As firms' pricing strategies increasingly rely on algorithms, two concerns have received much attention: algorithmic tacit collusion and price discrimination. This paper investigates the interaction between these two issues through simulations. In each period, a new buyer arrives with independently and identically distributed willingness to pay (WTP), and each firm, observing private signals about WTP, adopts Q-learning algorithms to set prices. We document two novel mechanisms that lead to collusive outcomes. Under asymmetric information, the algorithm with information advantage adopts a Bait-and-Restrained-Exploit strategy, surrendering profits on some signals by setting higher prices, while exploiting limited profits on the remaining signals by setting much lower prices. Under a symmetric information structure, competition on some signals facilitates convergence to supra-competitive prices on the remaining signals. Algorithms tend to 
    
[^2]: 估计长期治疗效果

    Estimating Effects of Long-Term Treatments. (arXiv:2308.08152v1 [econ.EM])

    [http://arxiv.org/abs/2308.08152](http://arxiv.org/abs/2308.08152)

    本论文介绍了一个纵向替代框架，用于准确估计长期治疗的效果。论文通过分解长期治疗效果为一系列函数，考虑用户属性、短期中间指标和治疗分配等因素。

    

    在A/B测试中，估计长期治疗的效果是一个巨大的挑战。这种治疗措施包括产品功能的更新、用户界面设计和推荐算法等，旨在在其发布后长期存在系统中。然而，由于长期试验的限制，从业者通常依赖短期实验结果来做产品发布决策。如何使用短期实验数据准确估计长期治疗效果仍然是一个未解决的问题。为了解决这个问题，我们引入了一个纵向替代框架。我们展示了，在标准假设下，长期治疗效果可以分解为一系列函数，这些函数依赖于用户属性、短期中间指标和治疗分配。我们描述了识别假设、估计策略和推理技术。

    Estimating the effects of long-term treatments in A/B testing presents a significant challenge. Such treatments -- including updates to product functions, user interface designs, and recommendation algorithms -- are intended to remain in the system for a long period after their launches. On the other hand, given the constraints of conducting long-term experiments, practitioners often rely on short-term experimental results to make product launch decisions. It remains an open question how to accurately estimate the effects of long-term treatments using short-term experimental data. To address this question, we introduce a longitudinal surrogate framework. We show that, under standard assumptions, the effects of long-term treatments can be decomposed into a series of functions, which depend on the user attributes, the short-term intermediate metrics, and the treatment assignments. We describe the identification assumptions, the estimation strategies, and the inference technique under thi
    
[^3]: 使用网络干扰进行无偏性估计研究

    Unconfoundedness with Network Interference. (arXiv:2211.07823v2 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2211.07823](http://arxiv.org/abs/2211.07823)

    本文研究使用单个大型网络的观测数据进行非参数估计，利用图神经网络估计高维烦恼函数，研究治疗效应和溢出效应。

    

    本文研究使用来自单个大型网络的观测数据对治疗效应和溢出效应进行非参数估计。我们考虑一个模型，其中干扰随着网络距离的增加而衰减，这允许对结果和治疗选择的同伴影响进行建模。在这个模型下，总网络和所有单位的协变量构成了混淆的来源，与现有工作不同的是，现有工作假设混淆可以由这些对象的已知低维函数总结。我们提出使用图神经网络来估计双重稳健估计量的高维烦恼函数。我们建立了近似稀疏的网络类比，以证明使用浅层结构的合理性。

    This paper studies nonparametric estimation of treatment and spillover effects using observational data from a single large network. We consider a model in which interference decays with network distance, which allows for peer influence in both outcomes and selection into treatment. Under this model, the total network and covariates of all units constitute sources of confounding, in contrast to existing work that assumes confounding can be summarized by a known, low-dimensional function of these objects. We propose to use graph neural networks to estimate the high-dimensional nuisance functions of a doubly robust estimator. We establish a network analog of approximate sparsity to justify the use of shallow architectures.
    
[^4]: 学校选择中的非参数化治疗效果识别

    Nonparametric Treatment Effect Identification in School Choice. (arXiv:2112.03872v3 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2112.03872](http://arxiv.org/abs/2112.03872)

    本论文研究了集中学校分配中的非参数化治疗效果识别和估计方法，通过识别原子治疗效应，该研究揭示了在回归不连续和抽签驱动的变异下的学校选择的异质性和重要性。

    

    本论文研究集中学校分配中因果效应的非参数化识别和估计。在许多集中分配设置中，学生既受到抽签驱动的变异，也受到回归不连续（RD）驱动的变异。我们刻画了被识别的原子治疗效应（aTEs）的完整集合，定义为在给定学生特征时，一对学校之间的条件平均治疗效应。原子治疗效应是治疗对比的基础，常见的估计方法会掩盖重要的异质性。特别地，许多原子治疗效应的聚合将在RD变异驱动下置零权重，并且这种聚合的估计器在渐进下将对RD驱动的原子治疗效应放置逐渐消失的权重。我们开发了一种用于经验评估RD变异驱动下的原子治疗效应权重的诊断工具。最后，我们提供了估计器和相应的渐进结果以进行推断。

    This paper studies nonparametric identification and estimation of causal effects in centralized school assignment. In many centralized assignment settings, students are subjected to both lottery-driven variation and regression discontinuity (RD) driven variation. We characterize the full set of identified atomic treatment effects (aTEs), defined as the conditional average treatment effect between a pair of schools, given student characteristics. Atomic treatment effects are the building blocks of more aggregated notions of treatment contrasts, and common approaches estimating aggregations of aTEs can mask important heterogeneity. In particular, many aggregations of aTEs put zero weight on aTEs driven by RD variation, and estimators of such aggregations put asymptotically vanishing weight on the RD-driven aTEs. We develop a diagnostic tool for empirically assessing the weight put on aTEs driven by RD variation. Lastly, we provide estimators and accompanying asymptotic results for infere
    

