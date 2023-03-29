# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Behavioral Machine Learning? Computer Predictions of Corporate Earnings also Overreact.](http://arxiv.org/abs/2303.16158) | 本文研究发现，机器学习算法可以更准确地预测公司盈利，但同样存在过度反应的问题，而传统培训的股市分析师和经过机器学习方法培训的分析师相比会产生较少的过度反应。 |
| [^2] | [Forecasting Large Realized Covariance Matrices: The Benefits of Factor Models and Shrinkage.](http://arxiv.org/abs/2303.16151) | 本论文介绍了一种用因子模型和收缩的方法预测大型实现协方差矩阵的模型。这种方法通过分解回报协方差矩阵并使用向量异质自回归模型进行估计，相对于标准基准提高了预测精度，并导致对最小方差组合的更好估计。 |
| [^3] | [The Value of Information and Circular Settings.](http://arxiv.org/abs/2303.16126) | 本文提出了一种基于克劳德·香农的信息和Ruslan Stratonovich的工作的信息价值（VoI）的通用概念，并将其应用于环形设置的经济应用，具有重要意义。 |
| [^4] | [Study on the risk-informed heuristic of decision-making on the restoration of defaulted corporation networks.](http://arxiv.org/abs/2303.15863) | 本文提出了可行的启发式决策制定方案，用于驱动违约公司网络恢复工作，在两个真实的DCN案例中进行了实验验证，研究结果表明启发式决策制定方案的实用性和性能。 |
| [^5] | [Redeeming Falsifiability?.](http://arxiv.org/abs/2303.15723) | 波普尔的可证伪性标准仍然有价值，因为如果知情的专家可以获得额外的信息，那么它能够识别出没有价值的理论。 |
| [^6] | [Endogenous Labour Flow Networks.](http://arxiv.org/abs/2301.07979) | 本文提出一种新型模型，从代理层面出发形成劳动力流动网络(LFNs)，消除了历史路径假设。该模型使用英国的微观数据为基础，生成了具有高精度的实证LFNs。 |
| [^7] | [Generalized Difference-in-differences Models: Robust Bounds.](http://arxiv.org/abs/2211.06710) | 该论文发展了一个鲁棒的广义DID方法，可以利用多个数据来源的信息来推断治疗效果。这种方法通过选择偏差的概念重新解释了平行趋势假设，适用性更广泛。 |
| [^8] | [Revealed preference characterization of marital stability under mutual consent divorce.](http://arxiv.org/abs/2110.10781) | 该论文提出了一个基于揭示性偏好的婚姻稳定性刻画，明确考虑了调节婚姻解散的离婚法律下的互意离婚，为了评估该刻画在确定家庭内消费方面的潜力，还进行了模拟实验。 |
| [^9] | [Equilibrium Selection in Data Markets: Multiple-Principal, Multiple-Agent Problems with Non-Rivalrous Goods.](http://arxiv.org/abs/2004.00196) | 数据市场的均衡选择是多元化的，现有的均衡概念不能预测结果。需要修改数据市场的制度框架或引入外部力量来解决根本问题。 |

# 详细

[^1]: 机器学习准确预测财报，但同样存在过度反应

    Behavioral Machine Learning? Computer Predictions of Corporate Earnings also Overreact. (arXiv:2303.16158v1 [q-fin.ST])

    [http://arxiv.org/abs/2303.16158](http://arxiv.org/abs/2303.16158)

    本文研究发现，机器学习算法可以更准确地预测公司盈利，但同样存在过度反应的问题，而传统培训的股市分析师和经过机器学习方法培训的分析师相比会产生较少的过度反应。

    

    大量证据表明，在金融领域中，机器学习算法的预测能力比人类更为准确。但是，文献并未测试算法预测是否更为理性。本文研究了几个算法（包括线性回归和一种名为Gradient Boosted Regression Trees的流行算法）对于公司盈利的预测结果。结果发现，GBRT平均胜过线性回归和人类股市分析师，但仍存在过度反应且无法满足理性预期标准。通过降低学习率，可最小程度上减少过度反应程度，但这会牺牲预测准确性。通过机器学习方法培训过的股市分析师比传统训练的分析师产生的过度反应较少。此外，股市分析师的预测反映出机器算法没有捕捉到的信息。

    There is considerable evidence that machine learning algorithms have better predictive abilities than humans in various financial settings. But, the literature has not tested whether these algorithmic predictions are more rational than human predictions. We study the predictions of corporate earnings from several algorithms, notably linear regressions and a popular algorithm called Gradient Boosted Regression Trees (GBRT). On average, GBRT outperformed both linear regressions and human stock analysts, but it still overreacted to news and did not satisfy rational expectation as normally defined. By reducing the learning rate, the magnitude of overreaction can be minimized, but it comes with the cost of poorer out-of-sample prediction accuracy. Human stock analysts who have been trained in machine learning methods overreact less than traditionally trained analysts. Additionally, stock analyst predictions reflect information not otherwise available to machine algorithms.
    
[^2]: 预测大型实现协方差矩阵:因子模型和收缩的好处。

    Forecasting Large Realized Covariance Matrices: The Benefits of Factor Models and Shrinkage. (arXiv:2303.16151v1 [q-fin.ST])

    [http://arxiv.org/abs/2303.16151](http://arxiv.org/abs/2303.16151)

    本论文介绍了一种用因子模型和收缩的方法预测大型实现协方差矩阵的模型。这种方法通过分解回报协方差矩阵并使用向量异质自回归模型进行估计，相对于标准基准提高了预测精度，并导致对最小方差组合的更好估计。

    

    我们提出了一种模型来预测收益的大型实现协方差矩阵，并对S&P 500的成分股进行了应用。为了解决维数灾难，我们使用标准企业级别因子（如大小、价值和盈利能力）分解回报协方差矩阵，并在残差协方差矩阵中使用部门限制。然后，使用最小绝对收缩和选择运算符（LASSO）的向量异质自回归（VHAR）模型对该限制模型进行估计。相对于标准基准，我们的方法提高了预测精度，并导致对最小方差组合的更好估计。

    We propose a model to forecast large realized covariance matrices of returns, applying it to the constituents of the S\&P 500 daily. To address the curse of dimensionality, we decompose the return covariance matrix using standard firm-level factors (e.g., size, value, and profitability) and use sectoral restrictions in the residual covariance matrix. This restricted model is then estimated using vector heterogeneous autoregressive (VHAR) models with the least absolute shrinkage and selection operator (LASSO). Our methodology improves forecasting precision relative to standard benchmarks and leads to better estimates of minimum variance portfolios.
    
[^3]: 信息价值和环形设置的研究

    The Value of Information and Circular Settings. (arXiv:2303.16126v1 [econ.TH])

    [http://arxiv.org/abs/2303.16126](http://arxiv.org/abs/2303.16126)

    本文提出了一种基于克劳德·香农的信息和Ruslan Stratonovich的工作的信息价值（VoI）的通用概念，并将其应用于环形设置的经济应用，具有重要意义。

    

    本文提出了一种基于克劳德·香农的信息和Ruslan Stratonovich的工作的信息价值（VoI）的通用概念，该概念具有贝叶斯决策理论和需求分析所需的理想属性。将Shannon / Stratonovich VoI概念与Hartley VoI概念进行比较，并应用于环形设置的典型经济应用，该设置概括了Ruslan Stratonovich的示例，并允许网络结构和不同经济运输成本的调查。

    We present a universal concept for the Value of Information (VoI) based on Claude Shannon's information and work of Ruslan Stratonovich that has desirable properties for Bayesian decision theory and demand analysis. The Shannon/Stratonovich VoI concept is compared to the concept of Hartley VoI and applied to an epitome economic application of a circular setting generalizing an example of Ruslan Stratonovich and allowing for a network structure and an investigation of various economic transport costs.
    
[^4]: 对违约公司网络恢复的风险知情启发式决策研究

    Study on the risk-informed heuristic of decision-making on the restoration of defaulted corporation networks. (arXiv:2303.15863v1 [econ.TH])

    [http://arxiv.org/abs/2303.15863](http://arxiv.org/abs/2303.15863)

    本文提出了可行的启发式决策制定方案，用于驱动违约公司网络恢复工作，在两个真实的DCN案例中进行了实验验证，研究结果表明启发式决策制定方案的实用性和性能。

    

    由政府主导的恢复已成为减轻因公司信用违约引发的金融风险的常见且有效方法。然而，在实践中，由于违约公司网络（DCNs）中存在大量搜索空间以及个体公司间的动态和循环相互依赖关系，往往难以提出最优方案。为了解决这一挑战，本文提出了一系列可行的决策启发式，驱动这些恢复工作的决策制定。为了检查其适用性和测量其性能，将这些启发式应用于由100家上市中国A股公司组成的两个真实DCN，并基于2021年财务数据，模拟随机生成的违约情况下的恢复。相应的案例研究模拟结果表明，DCN的恢复将会是...

    Government-run (Government-led) restoration has become a common and effective approach to the mitigation of financial risks triggered by corporation credit defaults. However, in practice, it is often challenging to come up with the optimal plan of those restorations, due to the massive search space associated with defaulted corporation networks (DCNs), as well as the dynamic and looped interdependence among the recovery of those individual corporations. To address such a challenge, this paper proposes an array of viable heuristics of the decision-making that drives those restoration campaigns. To examine their applicability and measure their performance, those heuristics have been applied to two real-work DCNs that consists of 100 listed Chinese A-share companies, whose restoration has been modelled based on the 2021 financial data, in the wake of randomly generated default scenarios. The corresponding simulation outcome of the case-study shows that the restoration of the DCNs would be
    
[^5]: 重温可证伪性？

    Redeeming Falsifiability?. (arXiv:2303.15723v1 [econ.TH])

    [http://arxiv.org/abs/2303.15723](http://arxiv.org/abs/2303.15723)

    波普尔的可证伪性标准仍然有价值，因为如果知情的专家可以获得额外的信息，那么它能够识别出没有价值的理论。

    

    我们重新审视了波普尔的可证伪性标准。一个测试者雇佣一个潜在的专家来提出一种理论，根据理论的表现，向专家提供有条件的付款。我们认为，如果知情的专家可以获得额外的信息，可证伪性确实有能力识别出没有价值的理论。

    We revisit Popper's falsifiability criterion. A tester hires a potential expert to produce a theory, offering payments contingent on the observed performance of the theory. We argue that if the informed expert can acquire additional information, falsifiability does have the power to identify worthless theories.
    
[^6]: 内生化劳动力流动网络

    Endogenous Labour Flow Networks. (arXiv:2301.07979v2 [econ.GN] UPDATED)

    [http://arxiv.org/abs/2301.07979](http://arxiv.org/abs/2301.07979)

    本文提出一种新型模型，从代理层面出发形成劳动力流动网络(LFNs)，消除了历史路径假设。该模型使用英国的微观数据为基础，生成了具有高精度的实证LFNs。

    

    在过去十年中，劳动动力学的研究引入了劳动力流动网络（LFN）来概括工作转移，并开发了数学模型来探索这些网络流的动态性。到目前为止，LFN模型一直依赖于静态网络结构的假设。然而，正如最近的事件（工作场所的自动化增加，COVID-19大流行，对编程技能的需求激增等）所显示的，我们正在经历改变劳动力市场中个人导航方式的重大变革。在这里，我们开发了一种新型模型，从代理层面出发形成LFNs，消除了假定未来工作转移流将沿着它们历史观察到的相同轨迹流动的必要性。该模型以英国的微观数据为基础，生成了具有高精度的实证LFNs。我们使用该模型探索了影响基础的冲击。

    In the last decade, the study of labour dynamics has led to the introduction of labour flow networks (LFNs) as a way to conceptualise job-to-job transitions, and to the development of mathematical models to explore the dynamics of these networked flows. To date, LFN models have relied upon an assumption of static network structure. However, as recent events (increasing automation in the workplace, the COVID-19 pandemic, a surge in the demand for programming skills, etc.) have shown, we are experiencing drastic shifts to the job landscape that are altering the ways individuals navigate the labour market. Here we develop a novel model that emerges LFNs from agent-level behaviour, removing the necessity of assuming that future job-to-job flows will be along the same paths where they have been historically observed. This model, informed by microdata for the United Kingdom, generates empirical LFNs with a high level of accuracy. We use the model to explore how shocks impacting the underlyin
    
[^7]: 广义差异-in-差异模型：稳健的界限

    Generalized Difference-in-differences Models: Robust Bounds. (arXiv:2211.06710v4 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2211.06710](http://arxiv.org/abs/2211.06710)

    该论文发展了一个鲁棒的广义DID方法，可以利用多个数据来源的信息来推断治疗效果。这种方法通过选择偏差的概念重新解释了平行趋势假设，适用性更广泛。

    

    差异-in-差异（DID）方法主要基于平行趋势（PT）假设，确定了被治疗对象的平均治疗效果（ATT）。通常，预处理期检验是证明PT假设的最常用方法。如果预处理期内治疗组和对照组的结果平均值趋势假设被拒绝，研究人员则对PT和DID结果的可信程度降低。本文发展了一个鲁棒的广义DID方法，利用所有可用的信息，不仅包括来自预处理期的信息，还包括来自多个数据来源的信息。我们采用一种不同于常规DID方法的方法来解释PT，利用了选择偏差的概念，使我们能够通过定义可能包含多个预处理期或其他基线协变量的信息集来推广标准DID估计量。我们的主要假设是，后处理期的选择偏差在所有可能结果的凸包内。

    The difference-in-differences (DID) method identifies the average treatment effects on the treated (ATT) under mainly the so-called parallel trends (PT) assumption. The most common and widely used approach to justify the PT assumption is the pre-treatment period examination. If a null hypothesis of the same trend in the outcome means for both treatment and control groups in the pre-treatment periods is rejected, researchers believe less in PT and the DID results. This paper develops a robust generalized DID method that utilizes all the information available not only from the pre-treatment periods but also from multiple data sources. Our approach interprets PT in a different way using a notion of selection bias, which enables us to generalize the standard DID estimand by defining an information set that may contain multiple pre-treatment periods or other baseline covariates. Our main assumption states that the selection bias in the post-treatment period lies within the convex hull of al
    
[^8]: 基于揭示性偏好的互意离婚下婚姻稳定性的刻画

    Revealed preference characterization of marital stability under mutual consent divorce. (arXiv:2110.10781v3 [econ.GN] UPDATED)

    [http://arxiv.org/abs/2110.10781](http://arxiv.org/abs/2110.10781)

    该论文提出了一个基于揭示性偏好的婚姻稳定性刻画，明确考虑了调节婚姻解散的离婚法律下的互意离婚，为了评估该刻画在确定家庭内消费方面的潜力，还进行了模拟实验。

    

    我们提出了一个基于揭示性偏好的婚姻稳定性刻画，明确考虑了调节婚姻解散的离婚法律。我们重点研究了互意离婚，即当事人只有在获得了对方的同意时才能离婚。我们提供理论见解，探讨了该刻画在确定家庭内消费方面的潜力。我们使用从“社会科学的网络化纵向研究”（LISS）小组抽取的家庭数据进行模拟实验，结果支持了我们的理论发现。

    We present a revealed preference characterization of marital stability explicitly accounting for the divorce law governing marital dissolution. We focus on mutual consent divorce, where individuals can divorce their partner only if they can obtain their consent. We provide theoretical insights into the potential of the characterization for identifying intrahousehold consumption. Simulation exercises using household data drawn from the Longitudinal Internet Studies for the Social Sciences (LISS) panel support our theoretical findings.
    
[^9]: 数据市场中的均衡选择：多主体、多代理问题中的非竞争性商品

    Equilibrium Selection in Data Markets: Multiple-Principal, Multiple-Agent Problems with Non-Rivalrous Goods. (arXiv:2004.00196v2 [cs.GT] UPDATED)

    [http://arxiv.org/abs/2004.00196](http://arxiv.org/abs/2004.00196)

    数据市场的均衡选择是多元化的，现有的均衡概念不能预测结果。需要修改数据市场的制度框架或引入外部力量来解决根本问题。

    

    数据市场与典型的商品市场存在几个不同之处，如信息不对称、数据的非竞争性和信息外部性。正式地，这些特点引出了一类新的博弈问题，我们称之为多主体、多代理非竞争性商品问题。假设主体的收益是支付给代理商的收益的拟线性函数，我们发现非竞争性商品市场存在根本的退化。这种均衡的多样性也影响到了均衡定义的普遍反复修正：变分均衡和归一化均衡在一般情况下都是不唯一的。这意味着大多数现有的均衡概念不能对当今出现的数据市场结果进行预测。研究结果支持了这样一个想法：即对支付合同本身的修改不太可能产生唯一的均衡，而需要调整数据市场的制度框架或引入外部力量来解决根本的退化问题。

    There are several aspects of data markets that distinguish them from a typical commodity market: asymmetric information, the non-rivalrous nature of data, and informational externalities. Formally, this gives rise to a new class of games which we call multiple-principal, multiple-agent problem with non-rivalrous goods. Under the assumption that the principal's payoff is quasilinear in the payments given to agents, we show that there is a fundamental degeneracy in the market of non-rivalrous goods. This multiplicity of equilibria also affects common refinements of equilibrium definitions intended to uniquely select an equilibrium: both variational equilibria and normalized equilibria will be non-unique in general. This implies that most existing equilibrium concepts cannot provide predictions on the outcomes of data markets emerging today. The results support the idea that modifications to payment contracts themselves are unlikely to yield a unique equilibrium, and either changes to the
    

