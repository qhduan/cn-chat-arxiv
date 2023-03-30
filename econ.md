# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Peer Prediction for Peer Review: Designing a Marketplace for Ideas.](http://arxiv.org/abs/2303.16855) | 本文提出了一个平台，利用同行预测算法奖励评审人员，旨在改善早期研究的同行评审，解决研究问题与出版偏见之间的不匹配问题。 |
| [^2] | [Power sector effects of alternative options for electrifying heavy-duty vehicles: go electric, and charge smartly.](http://arxiv.org/abs/2303.16629) | 研究了电动公路系统和电池电动车的替代方案对于电力部门的影响，发现可灵活充电的车辆共享BEV的电力部门成本最低，而使用电力燃料的重型车辆的成本最高。 |
| [^3] | [A general equilibrium model for multi-passenger ridesharing systems with stable matching.](http://arxiv.org/abs/2303.16595) | 本文提出了一个通用均衡模型以解决多乘客拼车系统的问题，并提出了一个序列-树算法用于求解问题。 |
| [^4] | [Critical Thinking Via Storytelling: Theory and Social Media Experiment.](http://arxiv.org/abs/2303.16422) | 通过社交媒体实验，研究人员发现不同的数字叙事格式会影响人们的批判性思维，中等长度的设计最有效，需要认知程度高的个体影响最大。 |
| [^5] | [Complexity of Equilibria in First-Price Auctions under General Tie-Breaking Rules.](http://arxiv.org/abs/2303.16388) | 本研究探讨了一种常见信息分布下一价拍卖中关于贝叶斯纳什均衡的复杂性，发现即使在三方平局规则下，该问题的复杂度都是PPAD完全的。同时提出了在采用均匀平局规则情况下的部分近似算法。 |
| [^6] | [Inflation forecasting with attention based transformer neural networks.](http://arxiv.org/abs/2303.15364) | 本文研究了基于注意力机制变压器神经网络用于预测不同通货膨胀率的潜力，结果表明其可以超越传统模型，成为金融决策中的有用工具。 |
| [^7] | [Incorporating Prior Knowledge of Latent Group Structure in Panel Data Models.](http://arxiv.org/abs/2211.16714) | 本文提出了一种受限制的贝叶斯分组估计器，通过利用研究人员对组的先验信念以两两约束的形式进行表达，将潜在组结构的先验知识纳入到面板数据模型中。蒙特卡罗实验显示，加入先验知识可以得到更准确的系数估计，并且比替代估计器获得更多的预测增益分数。我们应用了我们的方法到两个经验应用中。 |
| [^8] | [Sparse Quantile Regression.](http://arxiv.org/abs/2006.11201) | 本文研究了稀疏分位回归估计器，通过指数不等式得到了均方误差和回归函数估计误差的非渐近上界，并使用混合整数线性规划和一阶近似算法实现，能在实际数据应用中使用。 |

# 详细

[^1]: 同行评审的同行预测：设计一个想法市场

    Peer Prediction for Peer Review: Designing a Marketplace for Ideas. (arXiv:2303.16855v1 [cs.DL])

    [http://arxiv.org/abs/2303.16855](http://arxiv.org/abs/2303.16855)

    本文提出了一个平台，利用同行预测算法奖励评审人员，旨在改善早期研究的同行评审，解决研究问题与出版偏见之间的不匹配问题。

    

    本文描述了一个潜在的平台，旨在促进早期研究的学术同行评审。该平台旨在通过基于同行预测算法对评审人员进行奖励，使同行评审更加准确和及时。该算法使用对众包进行的Peer Truth Serum的变体（Radanovic等人，2016），其中人类评分者与机器学习基准竞争。我们解释了我们的方法如何解决科学中的两个大的低效问题：研究问题与出版偏见之间的不匹配。更好的早期研究同行评审为分享研究成果创造了额外的激励，简化了将想法匹配到团队，并使负面结果和p-hacking更加明显。

    The paper describes a potential platform to facilitate academic peer review with emphasis on early-stage research. This platform aims to make peer review more accurate and timely by rewarding reviewers on the basis of peer prediction algorithms. The algorithm uses a variation of Peer Truth Serum for Crowdsourcing (Radanovic et al., 2016) with human raters competing against a machine learning benchmark. We explain how our approach addresses two large productive inefficiencies in science: mismatch between research questions and publication bias. Better peer review for early research creates additional incentives for sharing it, which simplifies matching ideas to teams and makes negative results and p-hacking more visible.
    
[^2]: 重型车辆电气化的替代方案的电力部门影响:电气化和智能充电

    Power sector effects of alternative options for electrifying heavy-duty vehicles: go electric, and charge smartly. (arXiv:2303.16629v1 [econ.GN])

    [http://arxiv.org/abs/2303.16629](http://arxiv.org/abs/2303.16629)

    研究了电动公路系统和电池电动车的替代方案对于电力部门的影响，发现可灵活充电的车辆共享BEV的电力部门成本最低，而使用电力燃料的重型车辆的成本最高。

    

    在乘用车领域，电池电动车(BEV)已成为去碳化交通的最有前途的选择。对于重型车辆(HDV)，技术领域似乎更为开放。除了BEV外，还讨论了用于动态供电的电动公路系统(ERS)，以及使用氢燃料电池或电力燃料的卡车间接电气化。在这里，我们研究了这些替代方案的电力部门影响。我们将基于未来德国高可再生能源份额的情景，应用一个开源的容量扩展模型，利用详细的以路线为基础的卡车交通数据。结果表明，可灵活充电的车辆共享BEV的电力部门成本最低，而使用电力燃料的重型车辆的成本最高。如果BEV和ERS-BEV没有以优化的方式充电，电力部门成本会增加，但仍远低于使用氢或电力燃料的情景。这是相对较小电池、高度灵活的BEV在短途和中途步骤转移和超出道路广泛使用的优势的结果。

    In the passenger car segment, battery-electric vehicles (BEV) have emerged as the most promising option to decarbonize transportation. For heavy-duty vehicles (HDV), the technology space still appears to be more open. Aside from BEV, electric road systems (ERS) for dynamic power transfer are discussed, as well as indirect electrification with trucks that use hydrogen fuel cells or e-fuels. Here we investigate the power sector implications of these alternative options. We apply an open-source capacity expansion model to future scenarios of Germany with high renewable energy shares, drawing on detailed route-based truck traffic data. Results show that power sector costs are lowest for flexibly charged BEV that also carry out vehicle-to-grid operations, and highest for HDV using e-fuels. If BEV and ERS-BEV are not charged in an optimized way, power sector costs increase, but are still substantially lower than in scenarios with hydrogen or e-fuels. This is a consequence of the relatively p
    
[^3]: 带稳定匹配的多乘客拼车系统的一般均衡模型研究

    A general equilibrium model for multi-passenger ridesharing systems with stable matching. (arXiv:2303.16595v1 [econ.GN])

    [http://arxiv.org/abs/2303.16595](http://arxiv.org/abs/2303.16595)

    本文提出了一个通用均衡模型以解决多乘客拼车系统的问题，并提出了一个序列-树算法用于求解问题。

    

    本文提出了一个通用均衡模型，用于捕捉多乘客拼车系统中乘客、司机、平台和交通网络之间的内生性互动。稳定匹配被建模为一个均衡问题，其中没有拼车司机或乘客能够通过单方面切换另一个匹配序列来降低拼车的不满意程度。本文是首批将拼车平台多乘客匹配问题明确融入模型的研究之一。通过将匹配序列与超网络相结合，避免了多乘客拼车系统中的拼车-乘客转移。此外，本论文将拼车司机和乘客之间的匹配稳定性扩展到以匹配序列为基础的多OD多乘客情况。本文提供了所提出一般均衡模型的存在性证明。针对求解多乘客拼车问题，还提出了一个序列-树算法。

    This paper proposes a general equilibrium model for multi-passenger ridesharing systems, in which interactions between ridesharing drivers, passengers, platforms, and transportation networks are endogenously captured. Stable matching is modeled as an equilibrium problem in which no ridesharing driver or passenger can reduce ridesharing disutility by unilaterally switching to another matching sequence. This paper is one of the first studies that explicitly integrates the ridesharing platform multi-passenger matching problem into the model. By integrating matching sequence with hyper-network, ridesharing-passenger transfers are avoided in a multi-passenger ridesharing system. Moreover, the matching stability between the ridesharing drivers and passengers is extended to address the multi-OD multi-passenger case in terms of matching sequence. The paper provides a proof for the existence of the proposed general equilibrium. A sequence-bush algorithm is developed for solving the multi-passen
    
[^4]: 通过讲故事培养批判性思维：理论与社交媒体实验

    Critical Thinking Via Storytelling: Theory and Social Media Experiment. (arXiv:2303.16422v1 [econ.TH])

    [http://arxiv.org/abs/2303.16422](http://arxiv.org/abs/2303.16422)

    通过社交媒体实验，研究人员发现不同的数字叙事格式会影响人们的批判性思维，中等长度的设计最有效，需要认知程度高的个体影响最大。

    

    在一个简化的投票模型中，我们证明了增加具有意识到某一问题模棱两可性质的批判性思维者的比例会提高调查（选举）的效率，但可能会增加调查的偏见。在针对代表性美国人口的激励在线社交媒体实验（N = 706）中，我们证明了不同的数字叙事格式 – 不同的设计来呈现同一组事实 – 影响个体成为批判性思维者的强度。中等长度的设计（Facebook帖子）最有效地激发个体的批判性思维。需要认知的程度高的个体主要驱动了治疗效果的差异。

    In a stylized voting model, we establish that increasing the share of critical thinkers -- individuals who are aware of the ambivalent nature of a certain issue -- in the population increases the efficiency of surveys (elections) but might increase surveys' bias. In an incentivized online social media experiment on a representative US population (N = 706), we show that different digital storytelling formats -- different designs to present the same set of facts -- affect the intensity at which individuals become critical thinkers. Intermediate-length designs (Facebook posts) are most effective at triggering individuals into critical thinking. Individuals with a high need for cognition mostly drive the differential effects of the treatments.
    
[^5]: 一般性平局规则下一价拍卖均衡的复杂度

    Complexity of Equilibria in First-Price Auctions under General Tie-Breaking Rules. (arXiv:2303.16388v1 [cs.GT])

    [http://arxiv.org/abs/2303.16388](http://arxiv.org/abs/2303.16388)

    本研究探讨了一种常见信息分布下一价拍卖中关于贝叶斯纳什均衡的复杂性，发现即使在三方平局规则下，该问题的复杂度都是PPAD完全的。同时提出了在采用均匀平局规则情况下的部分近似算法。

    

    本研究探讨了一种常见信息分布下一价拍卖中关于贝叶斯纳什均衡的复杂性，考虑了平局规则作为输入的一部分。结果表明即使在三方平局规则（即表示当出现不超过三名竞标人员并列第一时，将物品分配给这些人员，否则采用均匀平局规则）的情况下，该问题的复杂度也是PPAD完全的。这是关于带有平局规则下一价拍卖均衡计算的首个困难结果。在积极方面，我们对于采用均匀平局规则的问题提出了一种部分近似算法。

    We study the complexity of finding an approximate (pure) Bayesian Nash equilibrium in a first-price auction with common priors when the tie-breaking rule is part of the input. We show that the problem is PPAD-complete even when the tie-breaking rule is trilateral (i.e., it specifies item allocations when no more than three bidders are in tie, and adopts the uniform tie-breaking rule otherwise). This is the first hardness result for equilibrium computation in first-price auctions with common priors. On the positive side, we give a PTAS for the problem under the uniform tie-breaking rule.
    
[^6]: 基于注意力机制变压器神经网络的通胀预测

    Inflation forecasting with attention based transformer neural networks. (arXiv:2303.15364v1 [econ.EM])

    [http://arxiv.org/abs/2303.15364](http://arxiv.org/abs/2303.15364)

    本文研究了基于注意力机制变压器神经网络用于预测不同通货膨胀率的潜力，结果表明其可以超越传统模型，成为金融决策中的有用工具。

    

    通胀是资金配置决策的重要因素，其预测是政府和中央银行的基本目标。然而，由于其预测取决于低频高波动数据且缺乏清晰的解释变量，因此预测通胀并不是一项简单的任务。最近，（深度）神经网络在许多应用中已显示出惊人的结果，逐渐成为新的技术水平标杆。本文研究了变压器深度神经网络架构用于预测不同通胀率的潜力。结果与经典时间序列和机器学习模型进行了比较。我们发现，我们改进后的变压器模型在16个实验中平均超过基准模型6个实验，在研究的4个通货膨胀率中表现最佳。我们的结果表明，基于变压器的通胀预测模型有超越传统模型的潜力，并可以成为金融决策中的有用工具。

    Inflation is a major determinant for allocation decisions and its forecast is a fundamental aim of governments and central banks. However, forecasting inflation is not a trivial task, as its prediction relies on low frequency, highly fluctuating data with unclear explanatory variables. While classical models show some possibility of predicting inflation, reliably beating the random walk benchmark remains difficult. Recently, (deep) neural networks have shown impressive results in a multitude of applications, increasingly setting the new state-of-the-art. This paper investigates the potential of the transformer deep neural network architecture to forecast different inflation rates. The results are compared to a study on classical time series and machine learning models. We show that our adapted transformer, on average, outperforms the baseline in 6 out of 16 experiments, showing best scores in two out of four investigated inflation rates. Our results demonstrate that a transformer based
    
[^7]: 将潜在组结构的先验知识纳入面板数据模型中

    Incorporating Prior Knowledge of Latent Group Structure in Panel Data Models. (arXiv:2211.16714v2 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2211.16714](http://arxiv.org/abs/2211.16714)

    本文提出了一种受限制的贝叶斯分组估计器，通过利用研究人员对组的先验信念以两两约束的形式进行表达，将潜在组结构的先验知识纳入到面板数据模型中。蒙特卡罗实验显示，加入先验知识可以得到更准确的系数估计，并且比替代估计器获得更多的预测增益分数。我们应用了我们的方法到两个经验应用中。

    

    组异质性假设已经成为面板数据模型中的研究热点。本文提出了一种受限制的贝叶斯分组估计器，利用研究人员对组的先验信念以两两约束的形式进行表达，表明一对单位是否可能属于同一组或不同组。我们提出了一种先验方法，用不同程度的置信度来纳入两两约束。整个框架建立在非参数贝叶斯方法上，隐含地指定了对组分区的分布，因此后验分析考虑了潜在的组结构的不确定性。蒙特卡罗实验显示，加入先验知识可以得到更准确的系数估计，并且比替代估计器获得更多的预测增益分数。我们应用了我们的方法到两个经验应用中。在第一个预测美国CPI通货膨胀率的应用中，我们证明了组的先验知识可以在数据不充足时提高密度预测。

    The assumption of group heterogeneity has become popular in panel data models. We develop a constrained Bayesian grouped estimator that exploits researchers' prior beliefs on groups in a form of pairwise constraints, indicating whether a pair of units is likely to belong to a same group or different groups. We propose a prior to incorporate the pairwise constraints with varying degrees of confidence. The whole framework is built on the nonparametric Bayesian method, which implicitly specifies a distribution over the group partitions, and so the posterior analysis takes the uncertainty of the latent group structure into account. Monte Carlo experiments reveal that adding prior knowledge yields more accurate estimates of coefficient and scores predictive gains over alternative estimators. We apply our method to two empirical applications. In a first application to forecasting U.S. CPI inflation, we illustrate that prior knowledge of groups improves density forecasts when the data is not 
    
[^8]: 稀疏分位回归

    Sparse Quantile Regression. (arXiv:2006.11201v4 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2006.11201](http://arxiv.org/abs/2006.11201)

    本文研究了稀疏分位回归估计器，通过指数不等式得到了均方误差和回归函数估计误差的非渐近上界，并使用混合整数线性规划和一阶近似算法实现，能在实际数据应用中使用。

    

    本文考虑了$\ell_0$惩罚和约束下的分位回归估计器。对于$\ell_0$惩罚的估计器，我们推导出超额分位预测风险尾部概率的指数不等式，并将其应用于获得非渐近上界的均方误差和回归函数估计误差。我们还为$\ell_0$约束的估计器推导了类似的结果。得到的收敛速率几乎是极小最优的，并且与$\ell_1$惩罚和非凸惩罚估计器的速率相同。此外，我们还表征了$\ell_0$惩罚估计器的期望汉明损失。我们通过混合整数线性规划和一个更可扩展的一阶近似算法实现了所提出的过程。我们通过蒙特卡罗实验展示了我们方法在有限样本情况下的性能，并在涉及婴儿出生体重的实际数据应用中展示了它的实用性。

    We consider both $\ell _{0}$-penalized and $\ell _{0}$-constrained quantile regression estimators. For the $\ell _{0}$-penalized estimator, we derive an exponential inequality on the tail probability of excess quantile prediction risk and apply it to obtain non-asymptotic upper bounds on the mean-square parameter and regression function estimation errors. We also derive analogous results for the $\ell _{0}$-constrained estimator. The resulting rates of convergence are nearly minimax-optimal and the same as those for $\ell _{1}$-penalized and non-convex penalized estimators. Further, we characterize expected Hamming loss for the $\ell _{0}$-penalized estimator. We implement the proposed procedure via mixed integer linear programming and also a more scalable first-order approximation algorithm. We illustrate the finite-sample performance of our approach in Monte Carlo experiments and its usefulness in a real data application concerning conformal prediction of infant birth weights (with $
    

