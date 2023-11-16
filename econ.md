# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [On propensity score matching with a diverging number of matches.](http://arxiv.org/abs/2310.14142) | 本文重新审视了倾向得分匹配方法在估计平均处理效应时的应用。我们研究了最近邻居数量随样本大小增长时的估计结果，发现改进后的估计量在效率方面优于原始的固定数量匹配方法。此外，我们还展示了在倾向得分实现“充分”降维时，可以达到半参数效率下界。 |
| [^2] | [Semidiscrete optimal transport with unknown costs.](http://arxiv.org/abs/2310.00786) | 本文研究了具有未知成本的半离散最优输运问题，提出了一种采用在线学习和随机逼近相结合的半局部算法，并证明其具有最优的收敛速度。 |
| [^3] | [Beyond Citations: Text-Based Metrics for Assessing Novelty and its Impact in Scientific Publications.](http://arxiv.org/abs/2309.16437) | 本研究使用文本挖掘技术，验证了其对于确定科学论文中新科学思想的来源和影响的有效性，并显示出其相对于传统的基于引用的度量指标的显著改进。 |
| [^4] | [Strategic Budget Selection in a Competitive Autobidding World.](http://arxiv.org/abs/2307.07374) | 该论文研究了在线广告平台上的广告商之间的战略预算选择问题。通过自动竞价算法，广告商根据预算和线性价值最大化其总价值。尽管广告商的真实偏好与自动竞价器的约束存在差距，但在均衡状态下的分配是近似的。 |
| [^5] | [Valuing Pharmaceutical Drug Innovations.](http://arxiv.org/abs/2212.07384) | 通过市场价值和研发成本，估计了制药药品创新的平均价值。成功药品的平均价值为 16.2 亿美元，临床试验前阶段的平均价值为 6430 万美元，成本为 5850 万美元。 |
| [^6] | [Filtered and Unfiltered Treatment Effects with Targeting Instruments.](http://arxiv.org/abs/2007.10432) | 本文研究如何使用有目标工具来控制多值处理中的选择偏差，并建立了组合编译器群体的条件来确定反事实平均值和处理效果。 |
| [^7] | [Difference-in-Differences Estimators of Intertemporal Treatment Effects.](http://arxiv.org/abs/2007.04267) | 本研究提出了使用面板数据进行治疗效应估计的方法，不限制治疗效应的异质性，并介绍了简化形式和归一化事件研究估计器用于衡量不同治疗剂量对效应的影响。同时，还展示了如何将简化形式估计器组合成经济可解释的成本效益比。 |

# 详细

[^1]: 关于倾向得分匹配和不同数量匹配的研究

    On propensity score matching with a diverging number of matches. (arXiv:2310.14142v1 [math.ST])

    [http://arxiv.org/abs/2310.14142](http://arxiv.org/abs/2310.14142)

    本文重新审视了倾向得分匹配方法在估计平均处理效应时的应用。我们研究了最近邻居数量随样本大小增长时的估计结果，发现改进后的估计量在效率方面优于原始的固定数量匹配方法。此外，我们还展示了在倾向得分实现“充分”降维时，可以达到半参数效率下界。

    

    本文重新审视了Abadie和Imbens(2016)关于使用倾向得分匹配估计平均处理效应的工作。我们探讨了当最近邻居数量$M$随样本大小增长时，这些估计量的渐近行为。结果表明，在效率方面，改进后的估计量能够优于原始的固定$M$估计量，这不足为奇但从技术上来说是非平凡的。此外，我们还展示了在倾向得分实现“充分”降维时，达到半参数效率下界的潜力，这与Hahn(1998)关于倾向得分方法基于降维的因果推断中降维的作用的见解相呼应。

    This paper reexamines Abadie and Imbens (2016)'s work on propensity score matching for average treatment effect estimation. We explore the asymptotic behavior of these estimators when the number of nearest neighbors, $M$, grows with the sample size. It is shown, hardly surprising but technically nontrivial, that the modified estimators can improve upon the original fixed-$M$ estimators in terms of efficiency. Additionally, we demonstrate the potential to attain the semiparametric efficiency lower bound when the propensity score achieves "sufficient" dimension reduction, echoing Hahn (1998)'s insight about the role of dimension reduction in propensity score-based causal inference.
    
[^2]: 具有未知成本的半离散最优输运

    Semidiscrete optimal transport with unknown costs. (arXiv:2310.00786v1 [econ.EM])

    [http://arxiv.org/abs/2310.00786](http://arxiv.org/abs/2310.00786)

    本文研究了具有未知成本的半离散最优输运问题，提出了一种采用在线学习和随机逼近相结合的半局部算法，并证明其具有最优的收敛速度。

    

    半离散最优输运是线性规划中经典输运问题的一种有挑战性的推广。其目标是以固定边际分布的方式设计两个随机变量（一个连续，一个离散）的联合分布，以最小化期望成本。我们提出了这个问题的一个新型变体，其中成本函数是未知的，但可以通过噪声观测学习；然而，每次只能采样一个函数。我们开发了一种半局部算法，将在线学习与随机逼近相结合，并证明其实现了最优的收敛速度，尽管随机梯度的非光滑性和目标函数的缺乏强凹性。

    Semidiscrete optimal transport is a challenging generalization of the classical transportation problem in linear programming. The goal is to design a joint distribution for two random variables (one continuous, one discrete) with fixed marginals, in a way that minimizes expected cost. We formulate a novel variant of this problem in which the cost functions are unknown, but can be learned through noisy observations; however, only one function can be sampled at a time. We develop a semi-myopic algorithm that couples online learning with stochastic approximation, and prove that it achieves optimal convergence rates, despite the non-smoothness of the stochastic gradient and the lack of strong concavity in the objective function.
    
[^3]: 超越引用：用于评估科学出版物中的新颖性及其影响力的基于文本的指标

    Beyond Citations: Text-Based Metrics for Assessing Novelty and its Impact in Scientific Publications. (arXiv:2309.16437v1 [econ.GN])

    [http://arxiv.org/abs/2309.16437](http://arxiv.org/abs/2309.16437)

    本研究使用文本挖掘技术，验证了其对于确定科学论文中新科学思想的来源和影响的有效性，并显示出其相对于传统的基于引用的度量指标的显著改进。

    

    我们使用文本挖掘来确定来自Microsoft Academic Graph (MAG)科学论文群体中新科学思想的来源和影响。我们验证了新技术及其相对于基于引用的传统度量指标的改进。首先，我们收集与诺贝尔奖联系的科学论文。这些论文可以说引入了对科学进展具有重大影响的全新科学思想。其次，我们确定文献综述论文，这些论文通常总结之前的科学发现而不是引领新的科学见解。最后，我们证明引领新的科学思想的论文更有可能被高度引用。我们的研究结果支持使用文本挖掘来测量发表时的新颖科学思想以及这些新思想对后续科学工作的影响。此外，研究结果还表明，相比基于论文引用的传统指标，新的文本指标有显著的改进。

    We use text mining to identify the origin and impact of new scientific ideas in the population of scientific papers from Microsoft Academic Graph (MAG). We validate the new techniques and their improvement over the traditional metrics based on citations. First, we collect scientific papers linked to Nobel prizes. These papers arguably introduced fundamentally new scientific ideas with a major impact on scientific progress. Second, we identify literature review papers which typically summarize prior scientific findings rather than pioneer new scientific insights. Finally, we illustrate that papers pioneering new scientific ideas are more likely to become highly cited. Our findings support the use of text mining both to measure novel scientific ideas at the time of publication and to measure the impact of these new ideas on later scientific work. Moreover, the results illustrate the significant improvement of the new text metrics over the traditional metrics based on paper citations. We 
    
[^4]: 竞争性自动竞价世界中的战略预算选择

    Strategic Budget Selection in a Competitive Autobidding World. (arXiv:2307.07374v1 [cs.GT])

    [http://arxiv.org/abs/2307.07374](http://arxiv.org/abs/2307.07374)

    该论文研究了在线广告平台上的广告商之间的战略预算选择问题。通过自动竞价算法，广告商根据预算和线性价值最大化其总价值。尽管广告商的真实偏好与自动竞价器的约束存在差距，但在均衡状态下的分配是近似的。

    

    我们研究了在线广告平台上广告商之间的一种游戏。该平台通过一次价格拍卖销售广告展示，并提供优化广告商投标的自动竞价算法。每个广告商都对其自动竞价器声明预算约束（可能还有最大竞价）。所选择的约束为自动竞价器定义了一个“内部”预算控制游戏，竞争者在竞争条件下最大化总价值。广告商在约束选择的“元游戏”中的获利取决于自动竞价器达到的均衡状态。广告商只向其自动竞价器指定预算和线性价值，但其真实偏好可以更加一般化：我们仅假设他们对点击具有递减边际价值并且对花费资金具有递增边际不便。我们的主要结果是，尽管存在广告商一般偏好和简单自动竞价器约束之间的差距，但在均衡状态下的分配是近似的。

    We study a game played between advertisers in an online ad platform. The platform sells ad impressions by first-price auction and provides autobidding algorithms that optimize bids on each advertiser's behalf. Each advertiser strategically declares a budget constraint (and possibly a maximum bid) to their autobidder. The chosen constraints define an "inner" budget-pacing game for the autobidders, who compete to maximize the total value received subject to the constraints. Advertiser payoffs in the constraint-choosing "metagame" are determined by the equilibrium reached by the autobidders.  Advertisers only specify budgets and linear values to their autobidders, but their true preferences can be more general: we assume only that they have weakly decreasing marginal value for clicks and weakly increasing marginal disutility for spending money. Our main result is that despite this gap between general preferences and simple autobidder constraints, the allocations at equilibrium are approxi
    
[^5]: 评估制药药品创新价值

    Valuing Pharmaceutical Drug Innovations. (arXiv:2212.07384v2 [econ.GN] UPDATED)

    [http://arxiv.org/abs/2212.07384](http://arxiv.org/abs/2212.07384)

    通过市场价值和研发成本，估计了制药药品创新的平均价值。成功药品的平均价值为 16.2 亿美元，临床试验前阶段的平均价值为 6430 万美元，成本为 5850 万美元。

    

    我们通过估计药品市场价值和研发成本来衡量制药药品创新的价值。我们依靠市场对药品研发公告的反应来确定价值和成本。利用公司公告数据和日收益率，我们估计成功药品的平均价值为16.2亿美元。平均而言，发现阶段的药品评估为6430万美元，成本为5850万美元。三个临床试验阶段的平均成本分别为600万美元、3000万美元和4100万美元。我们还研究了将这些估计应用于支持药品研发的政策的可能性。

    We measure the $\textit{value of pharmaceutical drug innovations}$ by estimating the market values of drugs and their development costs. We rely on market responses to drug development announcements to identify the values and costs. Using data on announcements by firms and their daily stock returns, we estimate the average value of successful drugs at \$1.62 billion. At the discovery stage, on average, drugs are valued at \$64.3 million and cost \$58.5 million. The average costs of the three phases of clinical trials are \$0.6, \$30, and \$41 million, respectively. We also investigate applying these estimates to policies supporting drug development.
    
[^6]: 有目标工具的过滤与未过滤处理效果

    Filtered and Unfiltered Treatment Effects with Targeting Instruments. (arXiv:2007.10432v3 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2007.10432](http://arxiv.org/abs/2007.10432)

    本文研究如何使用有目标工具来控制多值处理中的选择偏差，并建立了组合编译器群体的条件来确定反事实平均值和处理效果。

    

    在应用中，多值处理是很常见的。我们探讨了在这种情况下使用离散工具来控制选择偏差的方法。我们强调了有关定位（工具定位于哪些处理）和过滤（限制分析师对给定观测的处理分配的知识）的假设作用。这允许我们建立条件，使得针对组合编译器群体，可以确定反事实平均值和处理效果。我们通过将其应用于Head Start Impact Study和Student Achievement and Retention Project的数据来说明我们框架的实用性。

    Multivalued treatments are commonplace in applications. We explore the use of discrete-valued instruments to control for selection bias in this setting. Our discussion stresses the role of assumptions on targeting (which instruments target which treatments) and filtering (limits on the analyst's knowledge of the treatment assigned to a given observation). It allows us to establish conditions under which counterfactual averages and treatment effects are identified for composite complier groups. We illustrate the usefulness of our framework by applying it to data from the Head Start Impact Study and the Student Achievement and Retention Project.
    
[^7]: Difference-in-Differences估计器对时间跨期治疗效应的研究

    Difference-in-Differences Estimators of Intertemporal Treatment Effects. (arXiv:2007.04267v11 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2007.04267](http://arxiv.org/abs/2007.04267)

    本研究提出了使用面板数据进行治疗效应估计的方法，不限制治疗效应的异质性，并介绍了简化形式和归一化事件研究估计器用于衡量不同治疗剂量对效应的影响。同时，还展示了如何将简化形式估计器组合成经济可解释的成本效益比。

    

    我们研究了使用面板数据进行治疗效应估计。治疗可能是非二进制的，非吸收性的，并且结果可能受到治疗滞后的影响。我们做出了平行趋势的假设，但不限制治疗效应的异质性，与常用的双向固定效应回归不同。我们提出了事件研究的简化形式估计器，用于衡量暴露于弱治疗剂量的$\ell$期间的效应。我们还提出了归一化事件研究估计器，估计当前治疗和滞后治疗效应的加权平均值。最后，我们展示了简化形式估计器可以组合成一个经济可解释的成本效益比。

    We study treatment-effect estimation using panel data. The treatment may be non-binary, non-absorbing, and the outcome may be affected by the treatment lags. We make parallel-trends assumptions, but do not restrict treatment effect heterogeneity, unlike commonly-used two-way-fixed-effects regressions. We propose reduced-form event-study estimators of the effect of being exposed to a weakly higher treatment dose for $\ell$ periods. We also propose normalized event-study estimators, that estimate a weighted average of the effects of the current treatment and its lags. Finally, we show that the reduced-form estimators can be combined into an economically interpretable cost-benefit ratio.
    

