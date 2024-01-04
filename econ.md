# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Efficient Computation of Confidence Sets Using Classification on Equidistributed Grids.](http://arxiv.org/abs/2401.01804) | 本文提出了一种在均匀分布网格上使用分类法高效计算置信区间的方法。通过使用支持向量机分类器，将参数空间划分为两个区域，并通过训练分类器快速确定点是否在置信区间内。实验结果表明该方法具有高效和准确的特点。 |
| [^2] | [Model Averaging and Double Machine Learning.](http://arxiv.org/abs/2401.01645) | 本文介绍了一种将双机器学习和模型平均化相结合的方法，用于估计结构参数。研究表明，这种方法比起常见的基于单一学习器的替代方法更加鲁棒，适用于处理部分未知的函数形式。 |
| [^3] | [Classification and Treatment Learning with Constraints via Composite Heaviside Optimization: a Progressive MIP Method.](http://arxiv.org/abs/2401.01565) | 本文提出了一种使用Heaviside复合优化的渐进MIP方法，用于解决带有约束的多类别分类和多动作治疗问题。 |
| [^4] | [Urban Street Network Design and Transport-Related Greenhouse Gas Emissions around the World.](http://arxiv.org/abs/2401.01411) | 本研究通过全球调查发现，更直线、更连通、更不过度建造的街道网络与较低的交通排放相关。这些关系在不同发展水平和设计范式间存在差异，为规划者提供了更好的实证基础。 |
| [^5] | [Digital interventions and habit formation in educational technology.](http://arxiv.org/abs/2310.10850) | 该研究对印度儿童学习英语的教育应用程序进行了基于竞赛的干预，以帮助他们建立学习习惯。研究发现，在竞赛结束后的12周，治疗组继续保持对应用程序的参与。 |
| [^6] | [Difference-in-Differences with a Continuous Treatment.](http://arxiv.org/abs/2107.02637) | 本文研究了带有连续治疗的差异法设置。在广义平行趋势假设下，可以确定对待处理类型参数的治疗效应。然而，由于治疗效应的异质性，解释不同治疗值下参数差异具有挑战性。在框架之外引入了替代估计程序来克服传统方法的缺点，并在应用中展示它们可以得出不同的结论。 |
| [^7] | [Bias correction for quantile regression estimators.](http://arxiv.org/abs/2011.03073) | 本文研究了经典的分位数回归和工具变量分位数回归估计器的偏差，并通过推导高阶随机展开式，提出了一个可行的偏差修正方法。使用有限差分估计器对偏差分量进行估计，该方法在模拟实验中表现良好。 |

# 详细

[^1]: 在均匀分布网格上使用分类法高效计算置信区间

    Efficient Computation of Confidence Sets Using Classification on Equidistributed Grids. (arXiv:2401.01804v1 [econ.EM])

    [http://arxiv.org/abs/2401.01804](http://arxiv.org/abs/2401.01804)

    本文提出了一种在均匀分布网格上使用分类法高效计算置信区间的方法。通过使用支持向量机分类器，将参数空间划分为两个区域，并通过训练分类器快速确定点是否在置信区间内。实验结果表明该方法具有高效和准确的特点。

    

    经济模型产生的矩不等式可以用来形成对真实参数的检验，通过对这些检验进行反演可以得出真实参数的置信区间。然而，这些置信区间通常没有解析表达式，需要通过保留通过检验的网格点来数值计算得出置信区间。当统计量不具有渐近关键性时，在参数空间的每个网格点上构建临界值增加了计算负担。本文通过使用支持向量机（SVM）分类器，将计算问题转化为分类问题。其决策函数为将参数空间划分为两个区域（置信区间内部与外部）提供了更快速和更系统的方式。我们将置信区间内部的点标记为1，将外部的点标记为-1。研究人员可以在可管理的网格上训练SVM分类器，并使用该分类器确定密度更高的网格上的点是否在置信区间内。我们做出了一系列实验，证明了这种方法的效率和准确性。

    Economic models produce moment inequalities, which can be used to form tests of the true parameters. Confidence sets (CS) of the true parameters are derived by inverting these tests. However, they often lack analytical expressions, necessitating a grid search to obtain the CS numerically by retaining the grid points that pass the test. When the statistic is not asymptotically pivotal, constructing the critical value for each grid point in the parameter space adds to the computational burden. In this paper, we convert the computational issue into a classification problem by using a support vector machine (SVM) classifier. Its decision function provides a faster and more systematic way of dividing the parameter space into two regions: inside vs. outside of the confidence set. We label those points in the CS as 1 and those outside as -1. Researchers can train the SVM classifier on a grid of manageable size and use it to determine whether points on denser grids are in the CS or not. We est
    
[^2]: 模型平均化和双机器学习

    Model Averaging and Double Machine Learning. (arXiv:2401.01645v1 [econ.EM])

    [http://arxiv.org/abs/2401.01645](http://arxiv.org/abs/2401.01645)

    本文介绍了一种将双机器学习和模型平均化相结合的方法，用于估计结构参数。研究表明，这种方法比起常见的基于单一学习器的替代方法更加鲁棒，适用于处理部分未知的函数形式。

    

    本文讨论了将双重/无偏机器学习（DDML）与stacking（一种模型平均化方法，用于结合多个候选学习器）相结合，用于估计结构参数。我们引入了两种新的DDML stacking方法：短stacking利用DDML的交叉拟合步骤大大减少了计算负担，而汇总stacking可以在交叉拟合的折叠上强制执行通用 stacking权重。通过经过校准的模拟研究和两个应用程序，即估计引用和工资中的性别差距，我们展示了DDML与stacking相比基于单个预选学习器的常见替代方法对于部分未知的函数形式更加鲁棒。我们提供了实现我们方案的Stata和R软件。

    This paper discusses pairing double/debiased machine learning (DDML) with stacking, a model averaging method for combining multiple candidate learners, to estimate structural parameters. We introduce two new stacking approaches for DDML: short-stacking exploits the cross-fitting step of DDML to substantially reduce the computational burden and pooled stacking enforces common stacking weights over cross-fitting folds. Using calibrated simulation studies and two applications estimating gender gaps in citations and wages, we show that DDML with stacking is more robust to partially unknown functional forms than common alternative approaches based on single pre-selected learners. We provide Stata and R software implementing our proposals.
    
[^3]: 通过复合Heaviside优化实现约束的分类和治疗学习：一种渐进MIP方法

    Classification and Treatment Learning with Constraints via Composite Heaviside Optimization: a Progressive MIP Method. (arXiv:2401.01565v1 [math.OC])

    [http://arxiv.org/abs/2401.01565](http://arxiv.org/abs/2401.01565)

    本文提出了一种使用Heaviside复合优化的渐进MIP方法，用于解决带有约束的多类别分类和多动作治疗问题。

    

    本文提出了一种Heaviside复合优化方法，并提出了一种渐进（混合）整数规划（PIP）方法，用于解决带有约束的多类别分类和多动作治疗问题。Heaviside复合函数是Heaviside函数（即开区间$(0, \infty)$或闭区间$[0, \infty)$的指示函数）与可能非可微函数的复合。在建模方面，我们展示了Heaviside复合优化如何为学习最优多类别分类和多动作治疗规则提供统一的表达，这些规则受到规则依赖约束的限制，规定了各种领域限制。Heaviside复合函数具有等效的离散表达形式，可以通过整数规划（IP）方法解决。然而，对于具有大数据集的约束学习问题，

    This paper proposes a Heaviside composite optimization approach and presents a progressive (mixed) integer programming (PIP) method for solving multi-class classification and multi-action treatment problems with constraints. A Heaviside composite function is a composite of a Heaviside function (i.e., the indicator function of either the open $( \, 0,\infty )$ or closed $[ \, 0,\infty \, )$ interval) with a possibly nondifferentiable function. Modeling-wise, we show how Heaviside composite optimization provides a unified formulation for learning the optimal multi-class classification and multi-action treatment rules, subject to rule-dependent constraints stipulating a variety of domain restrictions. A Heaviside composite function has an equivalent discrete formulation %in terms of integer variables, and the resulting optimization problem can in principle be solved by integer programming (IP) methods. Nevertheless, for constrained learning problems with large data sets, %of modest or lar
    
[^4]: 城市街道网络设计与全球交通相关温室气体排放

    Urban Street Network Design and Transport-Related Greenhouse Gas Emissions around the World. (arXiv:2401.01411v1 [physics.soc-ph])

    [http://arxiv.org/abs/2401.01411](http://arxiv.org/abs/2401.01411)

    本研究通过全球调查发现，更直线、更连通、更不过度建造的街道网络与较低的交通排放相关。这些关系在不同发展水平和设计范式间存在差异，为规划者提供了更好的实证基础。

    

    本研究估计了街道网络特征和交通部门二氧化碳排放之间的关系，覆盖了世界上每个城市地区，并调查了这些关系是否在不同发展水平和城市设计范式中相同。以往的研究主要通过针对特定世界地区或相对较小的城市样本的案例研究来估计街道网络设计和交通排放之间的关系，这使得其推广性和适用性受到了限制。我们的全球研究发现，相较于其他因素，更直线、更连通、更不过度建造的街道网络与较低的交通排放相关。重要的是，这些关系在不同的发展水平和设计范式之间存在差异，然而大部分以往的文献报告的结果并不符合全球标准。规划者需要一个更好的实证基础来指导实践。

    This study estimates the relationships between street network characteristics and transport-sector CO2 emissions across every urban area in the world and investigates whether they are the same across development levels and urban design paradigms. The prior literature has estimated relationships between street network design and transport emissions -- including greenhouse gases implicated in climate change -- primarily through case studies focusing on certain world regions or relatively small samples of cities, complicating generalizability and applicability for evidence-informed practice. Our worldwide study finds that straighter, more-connected, and less-overbuilt street networks are associated with lower transport emissions, all else equal. Importantly, these relationships vary across development levels and design paradigms -- yet most prior literature reports findings from urban areas that are outliers by global standards. Planners need a better empirical base for evidence-informed 
    
[^5]: 在教育技术中的数字干预和习惯形成

    Digital interventions and habit formation in educational technology. (arXiv:2310.10850v1 [econ.GN])

    [http://arxiv.org/abs/2310.10850](http://arxiv.org/abs/2310.10850)

    该研究对印度儿童学习英语的教育应用程序进行了基于竞赛的干预，以帮助他们建立学习习惯。研究发现，在竞赛结束后的12周，治疗组继续保持对应用程序的参与。

    

    随着在线教育技术产品越来越常见，大量证据指出，学习者往往很难建立规律的学习习惯并完成他们的课程。与此同时，面向娱乐和社交互动的在线产品有时在增加用户参与度和创造频繁使用习惯方面非常有效，却无意中导致数字上瘾，特别是在青少年中。在这个项目中，我们在一个用于印度儿童学习英语的教育应用程序上进行了基于竞赛的干预，这在娱乐环境中很常见。大约一万名随机选择的学习者参加了一个为期100天的阅读竞赛。如果他们在基于学习内容消耗量的排行榜上排名足够高，他们将赢得一套实体书籍作为奖品。比赛结束后的12周，当治疗组没有额外的激励使用该应用程序时，他们继续参与其中。

    As online educational technology products have become increasingly prevalent, rich evidence indicates that learners often find it challenging to establish regular learning habits and complete their programs. Concurrently, online products geared towards entertainment and social interactions are sometimes so effective in increasing user engagement and creating frequent usage habits that they inadvertently lead to digital addiction, especially among youth. In this project, we carry out a contest-based intervention, common in the entertainment context, on an educational app for Indian children learning English. Approximately ten thousand randomly selected learners entered a 100-day reading contest. They would win a set of physical books if they ranked sufficiently high on a leaderboard based on the amount of educational content consumed. Twelve weeks after the end of the contest, when the treatment group had no additional incentives to use the app, they continued their engagement with it a
    
[^6]: 带连续治疗的差异法分析

    Difference-in-Differences with a Continuous Treatment. (arXiv:2107.02637v3 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2107.02637](http://arxiv.org/abs/2107.02637)

    本文研究了带有连续治疗的差异法设置。在广义平行趋势假设下，可以确定对待处理类型参数的治疗效应。然而，由于治疗效应的异质性，解释不同治疗值下参数差异具有挑战性。在框架之外引入了替代估计程序来克服传统方法的缺点，并在应用中展示它们可以得出不同的结论。

    

    本文分析了带有连续治疗的差异法设置。我们展示了在一种类似于二元治疗设置的广义平行趋势假设下，可以确定对待处理类型参数的治疗效应。然而，由于治疗效应异质性，解释不同治疗值下这些参数的差异可能特别具有挑战性。我们讨论了解决这些挑战的替代假设，通常是更强的假设。我们还提供了各种治疗效应分解结果，强调与流行的线性双向固定效应（TWFE）规格相关的参数可能很难解释，即使只有两个时间段。我们引入了不受TWFE缺点困扰的替代估计程序，并在一个应用中展示它们可能导致不同的结论。

    This paper analyzes difference-in-differences setups with a continuous treatment. We show that treatment effect on the treated-type parameters can be identified under a generalized parallel trends assumption that is similar to the binary treatment setup. However, interpreting differences in these parameters across different values of the treatment can be particularly challenging due to treatment effect heterogeneity. We discuss alternative, typically stronger, assumptions that alleviate these challenges. We also provide a variety of treatment effect decomposition results, highlighting that parameters associated with popular linear two-way fixed-effect (TWFE) specifications can be hard to interpret, \emph{even} when there are only two time periods. We introduce alternative estimation procedures that do not suffer from these TWFE drawbacks, and show in an application that they can lead to different conclusions.
    
[^7]: 偏差修正对分位数回归估计器的影响

    Bias correction for quantile regression estimators. (arXiv:2011.03073v6 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2011.03073](http://arxiv.org/abs/2011.03073)

    本文研究了经典的分位数回归和工具变量分位数回归估计器的偏差，并通过推导高阶随机展开式，提出了一个可行的偏差修正方法。使用有限差分估计器对偏差分量进行估计，该方法在模拟实验中表现良好。

    

    本文研究了经典的分位数回归和工具变量分位数回归估计器的偏差。虽然这些估计器在渐近意义下是一阶无偏的，但它们可能存在显著的二阶偏差。我们利用经验过程理论推导了这些估计器的高阶随机展开式。基于该展开式，我们得到了二阶偏差的显式公式，并提出了一个可行的偏差修正方法，该方法使用了偏差分量的有限差分估计器。在模拟实验中，提出的偏差修正方法表现良好。我们使用Engel家庭支出的经典数据提供了一个实证说明。

    We study the bias of classical quantile regression and instrumental variable quantile regression estimators. While being asymptotically first-order unbiased, these estimators can have non-negligible second-order biases. We derive a higher-order stochastic expansion of these estimators using empirical process theory. Based on this expansion, we derive an explicit formula for the second-order bias and propose a feasible bias correction procedure that uses finite-difference estimators of the bias components. The proposed bias correction method performs well in simulations. We provide an empirical illustration using Engel's classical data on household expenditure.
    

