# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Robust and Efficient Optimization Model for Electric Vehicle Charging Stations in Developing Countries under Electricity Uncertainty.](http://arxiv.org/abs/2307.05470) | 提出了一种鲁棒的、高效的模拟优化模型，用于解决发展中国家电动车充电站的电力不确定性问题。通过估计候选站点的服务可靠性，并运用控制变量技术，我们提供了一个高度鲁棒的解决方案，有效应对电力中断的不确定性。 |
| [^2] | [Harnessing the Potential of Volatility: Advancing GDP Prediction.](http://arxiv.org/abs/2307.05391) | 本文介绍了一种新的机器学习方法，通过将波动性作为模型权重，准确预测国内生产总值（GDP）。该方法考虑了意外冲击和事件对经济的影响，通过测试和比较表明，波动性加权的Lasso方法在准确性和鲁棒性方面优于其他方法，为决策者提供了有价值的决策工具。 |
| [^3] | [Synthetic Decomposition for Counterfactual Predictions.](http://arxiv.org/abs/2307.05122) | 本论文提出了一种使用“源”地区数据进行目标地区政策预测的方法。通过制定可转移条件并构建合成的结果-政策关系来满足条件。我们开发了通用过程来构建反事实预测的置信区间，并证明其有效性。本研究应用该方法预测了德克萨斯州青少年就业率。 |
| [^4] | [Selling Data to a Competitor (Extended Abstract).](http://arxiv.org/abs/2307.05078) | 本研究探讨了向竞争对手出售数据的成本与效益，并确定了利润最大化和帕累托改进机制。结果显示，利润最大化机制对消费者不利，而帕累托改进机制能提高消费者福利和公司利润。 |
| [^5] | [Resilient Information Aggregation.](http://arxiv.org/abs/2307.05054) | 这篇论文研究了在信息聚合游戏中，通过一个中介将多个发送者的信息聚合并向接收者推荐行动的问题。研究目标是找到一个最优的中介/平台，以在选择行动时最大化用户的福利，并且要求该中介具有激励兼容性和弹性。 |
| [^6] | [Epidemic Modeling with Generative Agents.](http://arxiv.org/abs/2307.04986) | 本研究利用生成型智能体在流行病模型中模拟了人类行为，通过模拟实验展示了智能体的行为与真实世界相似，并成功实现了流行病曲线的平坦化。该研究创造了改进动态系统建模的潜力，为表示人类思维、推理和决策提供了一种途径。 |
| [^7] | [Modeling evidential cooperation in large worlds.](http://arxiv.org/abs/2307.04879) | 该论文研究了大规模世界中的证据合作（ECL）的问题，并提出了一个不完全信息的谈判问题的博弈论模型。通过模型，作者发现所有合作者必须最大化相同的加权效用函数之和才能达到帕累托最优结果。 |
| [^8] | [Adapting to Misspecification.](http://arxiv.org/abs/2305.14265) | 研究提出了一种自适应收缩估计量，通过最小化最坏风险相对于已知偏差上限的理论最优估计量的最大风险的百分比增加。研究尝试解决经验研究中的鲁棒性和效率之间的权衡问题，避免了模型检验的复杂性。 |
| [^9] | [Don't (fully) exclude me, it's not necessary! Identification with semi-IVs.](http://arxiv.org/abs/2303.12667) | 本文提出了一种利用半工具变量实现离散内生变量模型识别的方法，对劳动、健康、教育等领域具有潜在应用价值。 |
| [^10] | [Optimal Mechanism Design for Agents with DSL Strategies: The Case of Sybil Attacks in Combinatorial Auctions.](http://arxiv.org/abs/2210.15181) | 本论文考虑了基于DSL策略的代理人优化机制设计，并在组合拍卖中的Sybil攻击案例中进行了研究。我们引入了DSL和Leximin概念，探讨了它们与其他鲁棒概念的关系，并展示了它们在拍卖和其他场景中的应用结果。 |
| [^11] | [SoK: Blockchain Decentralization.](http://arxiv.org/abs/2205.04256) | 该论文为区块链去中心化领域的知识系统化，提出了分类法并建议使用指数来衡量和量化区块链的去中心化水平。除了共识去中心化外，其他方面的研究相对较少。这项工作为未来的研究提供了基准和指导。 |

# 详细

[^1]: 发展中国家电动车充电站的鲁棒高效优化模型——基于电力不确定性的研究

    A Robust and Efficient Optimization Model for Electric Vehicle Charging Stations in Developing Countries under Electricity Uncertainty. (arXiv:2307.05470v1 [math.OC])

    [http://arxiv.org/abs/2307.05470](http://arxiv.org/abs/2307.05470)

    提出了一种鲁棒的、高效的模拟优化模型，用于解决发展中国家电动车充电站的电力不确定性问题。通过估计候选站点的服务可靠性，并运用控制变量技术，我们提供了一个高度鲁棒的解决方案，有效应对电力中断的不确定性。

    

    全球电动车（EV）的需求日益增长，这需要发展鲁棒且可访问的充电基础设施，尤其是在电力不稳定的发展中国家。早期的充电基础设施优化研究未能严格考虑此类服务中断特征，导致基础设施设计不佳。为解决这个问题，我们提出了一个高效的基于模拟的优化模型，该模型估计候选站点的服务可靠性，并将其纳入目标函数和约束中。我们采用了控制变量（CV）方差减小技术来提高模拟效率。我们的模型提供了一个高度鲁棒的解决方案，即使在候选站点服务可靠性被低估或高估的情况下，也能缓冲不确定的电力中断。使用印度尼西亚苏拉巴亚的数据集，我们的数值实验证明了所提出模型的有效性。

    The rising demand for electric vehicles (EVs) worldwide necessitates the development of robust and accessible charging infrastructure, particularly in developing countries where electricity disruptions pose a significant challenge. Earlier charging infrastructure optimization studies do not rigorously address such service disruption characteristics, resulting in suboptimal infrastructure designs. To address this issue, we propose an efficient simulation-based optimization model that estimates candidate stations' service reliability and incorporates it into the objective function and constraints. We employ the control variates (CV) variance reduction technique to enhance simulation efficiency. Our model provides a highly robust solution that buffers against uncertain electricity disruptions, even when candidate station service reliability is subject to underestimation or overestimation. Using a dataset from Surabaya, Indonesia, our numerical experiment demonstrates that the proposed mod
    
[^2]: 发挥波动性潜力：推进国内生产总值预测

    Harnessing the Potential of Volatility: Advancing GDP Prediction. (arXiv:2307.05391v1 [econ.GN])

    [http://arxiv.org/abs/2307.05391](http://arxiv.org/abs/2307.05391)

    本文介绍了一种新的机器学习方法，通过将波动性作为模型权重，准确预测国内生产总值（GDP）。该方法考虑了意外冲击和事件对经济的影响，通过测试和比较表明，波动性加权的Lasso方法在准确性和鲁棒性方面优于其他方法，为决策者提供了有价值的决策工具。

    

    本文提出了一种新颖的机器学习方法，将波动性作为模型权重引入国内生产总值（GDP）预测中。该方法专门设计用于准确预测GDP，同时考虑到可能影响经济的意外冲击或事件。通过对实际数据进行测试，并与以往用于GDP预测的技术（如Lasso和自适应Lasso）进行比较，表明波动性加权Lasso方法在准确性和鲁棒性方面优于其他方法，为决策者和分析师在快速变化的经济环境中提供了有价值的决策工具。本研究展示了数据驱动方法如何帮助我们更好地理解经济波动，并支持更有效的经济政策制定。

    This paper presents a novel machine learning approach to GDP prediction that incorporates volatility as a model weight. The proposed method is specifically designed to identify and select the most relevant macroeconomic variables for accurate GDP prediction, while taking into account unexpected shocks or events that may impact the economy. The proposed method's effectiveness is tested on real-world data and compared to previous techniques used for GDP forecasting, such as Lasso and Adaptive Lasso. The findings show that the Volatility-weighted Lasso method outperforms other methods in terms of accuracy and robustness, providing policymakers and analysts with a valuable tool for making informed decisions in a rapidly changing economic environment. This study demonstrates how data-driven approaches can help us better understand economic fluctuations and support more effective economic policymaking.  Keywords: GDP prediction, Lasso, Volatility, Regularization, Macroeconomics Variable Sele
    
[^3]: 模拟分解进行反事实预测

    Synthetic Decomposition for Counterfactual Predictions. (arXiv:2307.05122v1 [econ.EM])

    [http://arxiv.org/abs/2307.05122](http://arxiv.org/abs/2307.05122)

    本论文提出了一种使用“源”地区数据进行目标地区政策预测的方法。通过制定可转移条件并构建合成的结果-政策关系来满足条件。我们开发了通用过程来构建反事实预测的置信区间，并证明其有效性。本研究应用该方法预测了德克萨斯州青少年就业率。

    

    当政策变量超出先前政策支持范围时，反事实预测是具有挑战性的。然而，在许多情况下，关于感兴趣政策的信息可以从不同的“源”地区得到，这些地区已经实施了类似的政策。在本论文中，我们提出了一种新的方法，利用来自源地区的数据来预测目标地区的新政策。我们不依赖于使用参数化规范的结构关系的外推，而是制定一个可转移条件，并构建一个合成的结果-政策关系，使其尽可能接近满足条件。合成关系考虑了可观测数据和结构关系的相似性。我们开发了一个通用过程来构建反事实预测的渐进置信区间，并证明了其渐进有效性。然后，我们将我们的提议应用于预测德克萨斯州青少年就业率。

    Counterfactual predictions are challenging when the policy variable goes beyond its pre-policy support. However, in many cases, information about the policy of interest is available from different ("source") regions where a similar policy has already been implemented. In this paper, we propose a novel method of using such data from source regions to predict a new policy in a target region. Instead of relying on extrapolation of a structural relationship using a parametric specification, we formulate a transferability condition and construct a synthetic outcome-policy relationship such that it is as close as possible to meeting the condition. The synthetic relationship weighs both the similarity in distributions of observables and in structural relationships. We develop a general procedure to construct asymptotic confidence intervals for counterfactual predictions and prove its asymptotic validity. We then apply our proposal to predict average teenage employment in Texas following a cou
    
[^4]: 向竞争对手出售数据的成本与效益的研究

    Selling Data to a Competitor (Extended Abstract). (arXiv:2307.05078v1 [cs.GT])

    [http://arxiv.org/abs/2307.05078](http://arxiv.org/abs/2307.05078)

    本研究探讨了向竞争对手出售数据的成本与效益，并确定了利润最大化和帕累托改进机制。结果显示，利润最大化机制对消费者不利，而帕累托改进机制能提高消费者福利和公司利润。

    

    我们研究了向竞争对手出售数据的成本与效益。尽管出售所有消费者的数据可能会降低公司的总利润，但存在其他出售机制——只销售部分消费者的数据——使得两家公司都受益。我们确定了利润最大化的机制，并证明了公司的受益是以消费者的成本为代价的。然后，我们构建了帕累托改进机制，其中每个消费者的福利以及两家公司的利润均增加。最后，我们证明消费者选择加入可以作为一种手段，促使公司选择帕累托改进机制而不是利润最大化机制。

    We study the costs and benefits of selling data to a competitor. Although selling all consumers' data may decrease total firm profits, there exist other selling mechanisms -- in which only some consumers' data is sold -- that render both firms better off. We identify the profit-maximizing mechanism, and show that the benefit to firms comes at a cost to consumers. We then construct Pareto-improving mechanisms, in which each consumers' welfare, as well as both firms' profits, increase. Finally, we show that consumer opt-in can serve as an instrument to induce firms to choose a Pareto-improving mechanism over a profit-maximizing one.
    
[^5]: 弹性信息聚合

    Resilient Information Aggregation. (arXiv:2307.05054v1 [econ.TH])

    [http://arxiv.org/abs/2307.05054](http://arxiv.org/abs/2307.05054)

    这篇论文研究了在信息聚合游戏中，通过一个中介将多个发送者的信息聚合并向接收者推荐行动的问题。研究目标是找到一个最优的中介/平台，以在选择行动时最大化用户的福利，并且要求该中介具有激励兼容性和弹性。

    

    在信息聚合游戏中，一组发送者通过一个中介与接收者进行互动。每个发送者观察到世界的状态，并向中介传达一个信息，中介根据接收到的信息向接收者推荐一个行动。发送者和接收者的回报取决于世界的状态和接收者选择的行动。这个设置在两个方面扩展了著名的廉价交谈模型：有很多发送者（而不是只有一个）和一个中介。从实践角度来看，这个设置捕捉到了将战略专家的建议在服务行动推荐给用户时进行聚合的平台。我们的目标是找到一个最优的中介/平台，以在选择均衡时最大化用户的福利，并且对于接收者/用户在选择推荐行动时保持高度的激励兼容性要求，并且对于被推荐的行动选择也具有弹性。

    In an information aggregation game, a set of senders interact with a receiver through a mediator. Each sender observes the state of the world and communicates a message to the mediator, who recommends an action to the receiver based on the messages received. The payoff of the senders and of the receiver depend on both the state of the world and the action selected by the receiver. This setting extends the celebrated cheap talk model in two aspects: there are many senders (as opposed to just one) and there is a mediator. From a practical perspective, this setting captures platforms in which strategic experts advice is aggregated in service of action recommendations to the user. We aim at finding an optimal mediator/platform that maximizes the users' welfare given highly resilient incentive compatibility requirements on the equilibrium selected: we want the platform to be incentive compatible for the receiver/user when selecting the recommended action, and we want it to be resilient agai
    
[^6]: 用生成型智能体进行流行病建模

    Epidemic Modeling with Generative Agents. (arXiv:2307.04986v1 [cs.AI])

    [http://arxiv.org/abs/2307.04986](http://arxiv.org/abs/2307.04986)

    本研究利用生成型智能体在流行病模型中模拟了人类行为，通过模拟实验展示了智能体的行为与真实世界相似，并成功实现了流行病曲线的平坦化。该研究创造了改进动态系统建模的潜力，为表示人类思维、推理和决策提供了一种途径。

    

    本研究提供了一种新的个体层面建模范式，以解决将人类行为纳入流行病模型的重大挑战。通过在基于智能体的流行病模型中利用生成型人工智能，每个智能体都能够通过连接到大型语言模型（如ChatGPT）进行自主推理和决策。通过各种模拟实验，我们呈现了令人信服的证据，表明生成型智能体模仿了现实世界的行为，如生病时进行隔离，病例增加时进行自我隔离。总体而言，智能体展示了类似于近期流行病观察到的多次波动，然后是一段流行期。此外，智能体成功地使流行病曲线平坦化。该研究提供了一种改进动态系统建模的潜力，通过提供一种表示人类大脑、推理和决策的方法。

    This study offers a new paradigm of individual-level modeling to address the grand challenge of incorporating human behavior in epidemic models. Using generative artificial intelligence in an agent-based epidemic model, each agent is empowered to make its own reasonings and decisions via connecting to a large language model such as ChatGPT. Through various simulation experiments, we present compelling evidence that generative agents mimic real-world behaviors such as quarantining when sick and self-isolation when cases rise. Collectively, the agents demonstrate patterns akin to multiple waves observed in recent pandemics followed by an endemic period. Moreover, the agents successfully flatten the epidemic curve. This study creates potential to improve dynamic system modeling by offering a way to represent human brain, reasoning, and decision making.
    
[^7]: 在大规模世界中建模证据合作

    Modeling evidential cooperation in large worlds. (arXiv:2307.04879v1 [econ.GN])

    [http://arxiv.org/abs/2307.04879](http://arxiv.org/abs/2307.04879)

    该论文研究了大规模世界中的证据合作（ECL）的问题，并提出了一个不完全信息的谈判问题的博弈论模型。通过模型，作者发现所有合作者必须最大化相同的加权效用函数之和才能达到帕累托最优结果。

    

    大规模世界中的证据合作（ECL）指的是人类和其他代理人可以通过与具有不同价值观的相似代理人在一个大宇宙中的因果断开部分合作而获益。合作为代理人提供了其他相似代理人可能会合作的证据，从而使所有人都从交易中获益。对于利他主义者来说，这可能是一个关键的考虑因素。我将ECL发展为一个不完全信息的谈判问题的博弈论模型。该模型融入了对他人价值观和实证情况的不确定性，并解决了选择妥协结果的问题。使用该模型，我调查了ECL存在的问题，并概述了开放的技术和哲学问题。我展示了所有合作者必须最大化相同的加权效用函数之和才能达到帕累托最优结果。然而，我反对通过对效用函数进行归一化来隐式选择妥协结果。我回顾了谈判理论和

    Evidential cooperation in large worlds (ECL) refers to the idea that humans and other agents can benefit by cooperating with similar agents with differing values in causally disconnected parts of a large universe. Cooperating provides agents with evidence that other similar agents are likely to cooperate too, resulting in gains from trade for all. This could be a crucial consideration for altruists.  I develop a game-theoretic model of ECL as an incomplete information bargaining problem. The model incorporates uncertainty about others' value systems and empirical situations, and addresses the problem of selecting a compromise outcome. Using the model, I investigate issues with ECL and outline open technical and philosophical questions.  I show that all cooperators must maximize the same weighted sum of utility functions to reach a Pareto optimal outcome. However, I argue against selecting a compromise outcome implicitly by normalizing utility functions. I review bargaining theory and a
    
[^8]: 适应模型错误的估计

    Adapting to Misspecification. (arXiv:2305.14265v1 [econ.EM])

    [http://arxiv.org/abs/2305.14265](http://arxiv.org/abs/2305.14265)

    研究提出了一种自适应收缩估计量，通过最小化最坏风险相对于已知偏差上限的理论最优估计量的最大风险的百分比增加。研究尝试解决经验研究中的鲁棒性和效率之间的权衡问题，避免了模型检验的复杂性。

    

    经验研究通常涉及到鲁棒性和效率之间的权衡。研究人员想要估计一个标量参数，可以使用强假设来设计一个精准但可能存在严重偏差的局限估计量，也可以放松一些假设并设计一个更加鲁棒但变量较大的估计量。当局限估计量的偏差上限已知时，将无限制估计量收缩到局限估计量是最优的。对于局限估计量偏差上限未知的情况，我们提出了自适应收缩估计量，该估计量最小化最坏风险相对于已知偏差上限的理论最优估计量的最大风险的百分比增加。我们证明自适应估计量是一个加权凸最小化最大问题，并提供查找表以便于快速计算。重新审视了五项存在模型规范问题的经验研究，我们研究了适应错误的模型的优势而不是检验。

    Empirical research typically involves a robustness-efficiency tradeoff. A researcher seeking to estimate a scalar parameter can invoke strong assumptions to motivate a restricted estimator that is precise but may be heavily biased, or they can relax some of these assumptions to motivate a more robust, but variable, unrestricted estimator. When a bound on the bias of the restricted estimator is available, it is optimal to shrink the unrestricted estimator towards the restricted estimator. For settings where a bound on the bias of the restricted estimator is unknown, we propose adaptive shrinkage estimators that minimize the percentage increase in worst case risk relative to an oracle that knows the bound. We show that adaptive estimators solve a weighted convex minimax problem and provide lookup tables facilitating their rapid computation. Revisiting five empirical studies where questions of model specification arise, we examine the advantages of adapting to -- rather than testing for -
    
[^9]: 不要完全排除我，这是不必要的! 半工具变量的识别

    Don't (fully) exclude me, it's not necessary! Identification with semi-IVs. (arXiv:2303.12667v1 [econ.EM])

    [http://arxiv.org/abs/2303.12667](http://arxiv.org/abs/2303.12667)

    本文提出了一种利用半工具变量实现离散内生变量模型识别的方法，对劳动、健康、教育等领域具有潜在应用价值。

    

    本文提出了一种识别离散内生变量模型的新方法，将其应用于连续潜在结果的不可分离模型的一般情况下进行研究。我们采用半工具变量（semi-IVs) 来实现潜在结果的非参数识别以及选择方程式的识别，因此也能够识别个体治疗效应。与标准工具变量 （IVs）需要强制性完全排除不同，半工具变量仅在一些潜在结果方程式中部分排除，而不是全部排除。实践中，需要在强化排除约束和找到支持范围更广、相关性假设更强的半工具变量之间权衡。我们的方法为识别、估计和反事实预测开辟了新的途径，并在许多领域，如劳动，健康和教育等方面具有潜在应用。

    This paper proposes a novel approach to identify models with a discrete endogenous variable, that I study in the general context of nonseparable models with continuous potential outcomes. I show that nonparametric identification of the potential outcome and selection equations, and thus of the individual treatment effects, can be obtained with semi-instrumental variables (semi-IVs), which are relevant but only partially excluded from the potential outcomes, i.e., excluded from one or more potential outcome equations, but not necessarily all. This contrasts with the full exclusion restriction imposed on standard instrumental variables (IVs), which is stronger than necessary for identification: IVs are only a special case of valid semi-IVs. In practice, there is a trade-off between imposing stronger exclusion restrictions, and finding semi-IVs with a larger support and stronger relevance assumptions. Since, in empirical work, the main obstacle for finding a valid IV is often the full exc
    
[^10]: 基于DSL策略的代理人优化机制设计：组合拍卖中的Sybil攻击案例

    Optimal Mechanism Design for Agents with DSL Strategies: The Case of Sybil Attacks in Combinatorial Auctions. (arXiv:2210.15181v3 [econ.TH] UPDATED)

    [http://arxiv.org/abs/2210.15181](http://arxiv.org/abs/2210.15181)

    本论文考虑了基于DSL策略的代理人优化机制设计，并在组合拍卖中的Sybil攻击案例中进行了研究。我们引入了DSL和Leximin概念，探讨了它们与其他鲁棒概念的关系，并展示了它们在拍卖和其他场景中的应用结果。

    

    在不确定性下的鲁棒决策中，一个自然的选择是采用安全（或称为安全性）级别的策略。然而，在许多重要情况下，特别是拍卖中，存在大量的安全级别策略，因此选择变得不明确。我们考虑了两个细化的概念：（i）我们称之为DSL（可区分安全级别），它基于“区分”概念，使用一对一比较来去除平凡的等价性。这捕捉到了当比较两种行为时，代理人不应关心导致相同回报的情况下的回报情况。（ii）社会选择理论中众所周知的Leximin概念，我们将其应用于鲁棒决策。特别是，leximin总是DSL，但反之不成立。我们研究了这些概念与其他鲁棒概念的关系，并展示了它们在拍卖和其他场景中的应用结果。经济设计旨在在面对自我激励的情况下最大化社会福利。

    In robust decision making under uncertainty, a natural choice is to go with safety (aka security) level strategies. However, in many important cases, most notably auctions, there is a large multitude of safety level strategies, thus making the choice unclear. We consider two refined notions:  (i) a term we call DSL (distinguishable safety level), and is based on the notion of ``discrimin'', which uses a pairwise comparison of actions while removing trivial equivalencies. This captures the fact that when comparing two actions an agent should not care about payoffs in situations where they lead to identical payoffs.  (ii) The well-known Leximin notion from social choice theory, which we apply for robust decision-making. In particular, the leximin is always DSL but not vice-versa.  We study the relations of these notions to other robust notions, and illustrate the results of their use in auctions and other settings. Economic design aims to maximize social welfare when facing self-motivate
    
[^11]: SoK：区块链去中心化

    SoK: Blockchain Decentralization. (arXiv:2205.04256v4 [econ.GN] UPDATED)

    [http://arxiv.org/abs/2205.04256](http://arxiv.org/abs/2205.04256)

    该论文为区块链去中心化领域的知识系统化，提出了分类法并建议使用指数来衡量和量化区块链的去中心化水平。除了共识去中心化外，其他方面的研究相对较少。这项工作为未来的研究提供了基准和指导。

    

    区块链通过在点对点网络中实现分布式信任，为去中心化经济提供了支持。然而，令人惊讶的是，目前还缺乏广泛接受的去中心化定义或度量标准。我们通过全面分析现有研究，探索了区块链去中心化的知识系统化（SoK）。首先，我们通过对现有研究的定性分析，在共识、网络、治理、财富和交易等五个方面建立了用于分析区块链去中心化的分类法。我们发现，除了共识去中心化以外，其他方面的研究相对较少。其次，我们提出了一种指数，通过转换香农熵来衡量和量化区块链在不同方面的去中心化水平。我们通过比较静态模拟验证了该指数的可解释性。我们还提供了其他指数的定义和讨论，包括基尼系数、中本聪系数和赫尔曼-赫尔东指数等。我们的工作概述了当前区块链去中心化的景象，并提出了一个量化的度量标准，为未来的研究提供基准。

    Blockchain empowers a decentralized economy by enabling distributed trust in a peer-to-peer network. However, surprisingly, a widely accepted definition or measurement of decentralization is still lacking. We explore a systematization of knowledge (SoK) on blockchain decentralization by comprehensively analyzing existing studies in various aspects. First, we establish a taxonomy for analyzing blockchain decentralization in the five facets of consensus, network, governance, wealth, and transaction bu qualitative analysis of existing research. We find relatively little research on aspects other than consensus decentralization. Second, we propose an index that measures and quantifies the decentralization level of blockchain across different facets by transforming Shannon entropy. We verify the explainability of the index via comparative static simulations. We also provide the definition and discussion of alternative indices including the Gini Coefficient, Nakamoto Coefficient, and Herfind
    

