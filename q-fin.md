# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Applying Deep Learning to Calibrate Stochastic Volatility Models.](http://arxiv.org/abs/2309.07843) | 本研究将深度学习技术应用于校准随机波动性模型，通过训练神经网络对基于Heston模型的标的资产进行定价，并且在校准方面取得了快速和准确的结果。 |
| [^2] | [Market-GAN: Adding Control to Financial Market Data Generation with Semantic Context.](http://arxiv.org/abs/2309.07708) | 本研究通过提出具有市场动态、股票代码和历史状态作为上下文的上下文市场数据集以及使用条件生成对抗网络（GAN）来实现对金融数据生成的控制。 |
| [^3] | [An empirical study of profit and loss allocations.](http://arxiv.org/abs/2309.07667) | 本研究通过实证研究了利润和损失分配中的三种分解原则，发现一次一个分解会产生无法解释的损失，而顺序更新和平均顺序更新可以完全解释利润和损失，且顺序更新与更新顺序有关。此外，分解原则的结果也受子区间大小的影响。 |
| [^4] | [Computer says 'no': Exploring systemic hiring bias in ChatGPT using an audit approach.](http://arxiv.org/abs/2309.07664) | 本研究使用审计方法探索了ChatGPT在求职者筛选中的系统偏见，研究发现语言提示和申请者姓名中的种族身份线索会影响ChatGPT的评估。 |
| [^5] | [Long-Term Mean-Variance Optimization Under Mean-Reverting Equity Returns.](http://arxiv.org/abs/2309.07488) | 本文研究了长期均方差优化在均值回归股票收益下的应用，结果发现如果股票风险溢价缓慢回归，承诺长期确定性投资策略的投资者将实现更好的风险收益平衡。为求解该问题，应用了变分法原理，推导出了描述最优投资策略的欧拉-拉格朗日方程，问题等价于谱问题，可以得到最优策略的显式解。 |
| [^6] | [Measuring Higher-Order Rationality with Belief Control.](http://arxiv.org/abs/2309.07427) | 本研究通过将人类参与者与机器人玩家配对，发现当与机器人配对时，个体表现出较高的理性水平，并在不同游戏中保持稳定水平。这为评估个体战略能力提供了一种新方法。 |
| [^7] | [The Fiscal Cost of Public Debt and Government Spending Shocks.](http://arxiv.org/abs/2309.07371) | 本研究通过使用美国历史数据，发现当债务成为财政负担时，政府通过限制债务发行来应对支出冲击，导致财政政策失去了刺激经济活动的能力。 |
| [^8] | [The effect of housewife labor on gdp calculations.](http://arxiv.org/abs/2309.07160) | 本研究通过理论分析和经验建模，探讨了家庭主妇劳动对GDP计算的影响。研究发现，人类特征的变化导致了有限理性个体特征的形成，女性更倾向于成为家庭主妇的选择。实证分析揭示了家庭主妇劳动的经济价值。 |
| [^9] | [Finite Difference Solution Ansatz approach in Least-Squares Monte Carlo.](http://arxiv.org/abs/2305.09166) | 本文提出了一种通用的数值方案，使用低维有限差分法的精确解来构建条件期望继续支付的假设，并将其用于线性回归，以提高在美式期权定价中最小二乘蒙特卡罗方法的精确性。 |

# 详细

[^1]: 将深度学习应用于校准随机波动性模型

    Applying Deep Learning to Calibrate Stochastic Volatility Models. (arXiv:2309.07843v1 [q-fin.CP])

    [http://arxiv.org/abs/2309.07843](http://arxiv.org/abs/2309.07843)

    本研究将深度学习技术应用于校准随机波动性模型，通过训练神经网络对基于Heston模型的标的资产进行定价，并且在校准方面取得了快速和准确的结果。

    

    随机波动性模型是一种波动率是随机过程的模型，可以捕捉到隐含波动率曲面的大部分基本特征，并提供更真实的波动率笑曲线或偏斜动态。然而，它们存在一个重要问题，即校准时间过长。最近，基于深度学习（DL）技术的替代校准方法已被用于构建快速且准确的校准解决方案。Huge和Savine开发了一种差分深度学习（DDL）方法，该方法在样本中训练了机器学习模型，其中样本不仅包括特征和标签，还包括标签对特征的微分。本研究旨在将DDL技术应用于定价基本欧洲期权（即校准工具），具体而言，是在基于Heston模型的标的资产上定价看涨期权，并使用训练好的网络对模型进行校准。DDL可以实现快速训练和准确定价。训练好的神经网络戏剧性地

    Stochastic volatility models, where the volatility is a stochastic process, can capture most of the essential stylized facts of implied volatility surfaces and give more realistic dynamics of the volatility smile or skew. However, they come with the significant issue that they take too long to calibrate.  Alternative calibration methods based on Deep Learning (DL) techniques have been recently used to build fast and accurate solutions to the calibration problem. Huge and Savine developed a Differential Deep Learning (DDL) approach, where Machine Learning models are trained on samples of not only features and labels but also differentials of labels to features. The present work aims to apply the DDL technique to price vanilla European options (i.e. the calibration instruments), more specifically, puts when the underlying asset follows a Heston model and then calibrate the model on the trained network. DDL allows for fast training and accurate pricing. The trained neural network dramatic
    
[^2]: 添加语义上下文的控制力量，为金融市场数据生成引入Market-GAN

    Market-GAN: Adding Control to Financial Market Data Generation with Semantic Context. (arXiv:2309.07708v1 [cs.LG])

    [http://arxiv.org/abs/2309.07708](http://arxiv.org/abs/2309.07708)

    本研究通过提出具有市场动态、股票代码和历史状态作为上下文的上下文市场数据集以及使用条件生成对抗网络（GAN）来实现对金融数据生成的控制。

    

    金融模拟器在提升预测准确性、管理风险和促进战略金融决策方面发挥着重要作用。尽管已经开发了金融市场模拟方法，但现有的框架常常难以适应专门的模拟上下文。我们将挑战归结为：i）当前的金融数据集不包含上下文标签；ii）当前的技术没有设计用于生成具有上下文控制的金融数据，与其他模态相比，这要求更高的精度；iii）由于金融数据的非平稳、噪声性质，生成与上下文对齐、高保真度的数据存在困难。为了解决这些挑战，我们的贡献是：i）提出了具有市场动态、股票代码和历史状态作为上下文的上下文市场数据集，利用线性回归和动态时间扭曲聚类结合的市场动态建模方法提取市场动态；ii）我们预先准备了Market-GAN模型，该模型通过使用条件生成对抗网络（GAN）的方法以及编码上下文向量的方式来实现对金融数据生成的控制。

    Financial simulators play an important role in enhancing forecasting accuracy, managing risks, and fostering strategic financial decision-making. Despite the development of financial market simulation methodologies, existing frameworks often struggle with adapting to specialized simulation context. We pinpoint the challenges as i) current financial datasets do not contain context labels; ii) current techniques are not designed to generate financial data with context as control, which demands greater precision compared to other modalities; iii) the inherent difficulties in generating context-aligned, high-fidelity data given the non-stationary, noisy nature of financial data. To address these challenges, our contributions are: i) we proposed the Contextual Market Dataset with market dynamics, stock ticker, and history state as context, leveraging a market dynamics modeling method that combines linear regression and Dynamic Time Warping clustering to extract market dynamics; ii) we prese
    
[^3]: 利润和损失分配的实证研究

    An empirical study of profit and loss allocations. (arXiv:2309.07667v1 [q-fin.PM])

    [http://arxiv.org/abs/2309.07667](http://arxiv.org/abs/2309.07667)

    本研究通过实证研究了利润和损失分配中的三种分解原则，发现一次一个分解会产生无法解释的损失，而顺序更新和平均顺序更新可以完全解释利润和损失，且顺序更新与更新顺序有关。此外，分解原则的结果也受子区间大小的影响。

    

    每个业务年度的投资利润和损失（p＆l）被分解为不同的风险因素（如利率、信用利差、外汇汇率等）是一项监管要求的任务，例如 Solvency 2。目前有三种常见的分解原则：一次一个（OAT）、顺序更新（SU）和平均顺序更新（ASU）分解。SU和ASU分解完全解释了p＆l。然而，OAT分解会产生一些无法解释的p＆l。SU分解取决于风险因素的顺序或标签。这三种分解可以使用年度、季度、月度或日常数据在不同的子区间定义。在本研究中，我们经验性地量化了OAT分解的无法解释的p＆l，SU分解对更新顺序的依赖以及三种分解原则在子区间大小上的依赖程度。

    The decomposition of the investment profit and loss (p&l) for each business year into different risk factors (e.g., interest rates, credit spreads, foreign exchange rate etc.) is a task that is regulatory required, e.g., by Solvency 2. Three different decomposition principles are prevalent: one-at-a-time (OAT), sequential updating (SU) and average sequential updating (ASU) decompositions. The SU and the ASU decompositions explain the p&l fully. However, the OAT decomposition generates some unexplained p&l. The SU decomposition depends on the order or labeling of the risk factors. The three decompositions can be defined on different sub-intervals using annually, quarterly, monthly or daily data. In this research, we empirically quantify: the unexplained p\&l of the OAT decomposition; the dependence of the SU decomposition on the update order; and how much the three decomposition principles depend on the size of the sub-intervals.
    
[^4]: 计算机说“不行”: 使用审计方法探索ChatGPT中的系统招聘偏见

    Computer says 'no': Exploring systemic hiring bias in ChatGPT using an audit approach. (arXiv:2309.07664v1 [econ.GN])

    [http://arxiv.org/abs/2309.07664](http://arxiv.org/abs/2309.07664)

    本研究使用审计方法探索了ChatGPT在求职者筛选中的系统偏见，研究发现语言提示和申请者姓名中的种族身份线索会影响ChatGPT的评估。

    

    大型语言模型在优化职业活动方面具有重要潜力，如简化人员选拔程序。然而，人们担心这些模型会延续预训练数据中嵌入的系统偏见。这项研究探讨了ChatGPT在求职者筛选方面是否表现出种族或性别偏见，ChatGPT是一个能够生成类似人类回应的聊天机器人。通过使用一种通信审计方法，我模拟了一个简历筛选任务，指示聊天机器人评价虚构的申请者简历，这些简历只有名字不同，以暗示种族和性别身份。通过比较阿拉伯、亚洲、美国黑人、中非、荷兰、东欧、西班牙裔、土耳其和美国白人男性和女性申请者的评分，我发现种族和性别身份会影响ChatGPT的评估。种族偏见似乎部分源于提示语言，部分源于申请者姓名中的种族身份线索。

    Large language models offer significant potential for optimising professional activities, such as streamlining personnel selection procedures. However, concerns exist about these models perpetuating systemic biases embedded into their pre-training data. This study explores whether ChatGPT, a chatbot producing human-like responses to language tasks, displays ethnic or gender bias in job applicant screening. Using a correspondence audit approach, I simulated a CV screening task in which I instructed the chatbot to rate fictitious applicant profiles only differing in names, signalling ethnic and gender identity. Comparing ratings of Arab, Asian, Black American, Central African, Dutch, Eastern European, Hispanic, Turkish, and White American male and female applicants, I show that ethnic and gender identity influence ChatGPT's evaluations. The ethnic bias appears to arise partly from the prompts' language and partly from ethnic identity cues in applicants' names. Although ChatGPT produces n
    
[^5]: 长期均方差优化在均值回归股票收益下的应用

    Long-Term Mean-Variance Optimization Under Mean-Reverting Equity Returns. (arXiv:2309.07488v1 [q-fin.MF])

    [http://arxiv.org/abs/2309.07488](http://arxiv.org/abs/2309.07488)

    本文研究了长期均方差优化在均值回归股票收益下的应用，结果发现如果股票风险溢价缓慢回归，承诺长期确定性投资策略的投资者将实现更好的风险收益平衡。为求解该问题，应用了变分法原理，推导出了描述最优投资策略的欧拉-拉格朗日方程，问题等价于谱问题，可以得到最优策略的显式解。

    

    成为长期投资者已经成为支持投资更大风险资产的一个论据，但是尽管直观上具有吸引力，很少有人确切说明为什么资本市场会为长期投资者提供比其他投资者更好的机会。本文显示，如果实际上股票风险溢价是缓慢回归的，那么承诺长期确定性投资策略的投资者在均值方差优化中将实现更好的风险收益平衡比短期投资期限的投资者。众所周知，均值方差优化问题不能通过动态规划来求解。相反，应用变分法原理推导出了描述最优投资策略的欧拉-拉格朗日方程。主要结果是优化问题等价于谱问题，通过它可以得到最优投资策略的显式解。

    Being a long-term investor has become an argument by itself to sustain larger allocations to risky assets, but - although intuitively appealing - it is rarely stated exactly why capital markets would provide a better opportunity set to investors with long investment horizons than to other investors. In this paper, it is shown that if in fact the equity risk-premium is slowly mean-reverting then an investor committing to a long-term deterministic investment strategy would realize a better risk-return trade-off in a mean-variance optimization than investors with shorter investment horizons. It is well known that the problem of mean-variance optimization cannot be solved by dynamic programming. Instead, the principle of Calculus of Variations is applied to derive an Euler-Lagrange equation characterizing the optimal investment strategy. It is a main result that the optimization problem is equivalent to a spectral problem by which explicit solutions to the optimal investment strategy can b
    
[^6]: 用信念控制来衡量高阶理性

    Measuring Higher-Order Rationality with Belief Control. (arXiv:2309.07427v1 [econ.GN])

    [http://arxiv.org/abs/2309.07427](http://arxiv.org/abs/2309.07427)

    本研究通过将人类参与者与机器人玩家配对，发现当与机器人配对时，个体表现出较高的理性水平，并在不同游戏中保持稳定水平。这为评估个体战略能力提供了一种新方法。

    

    仅基于选择数据确定个体的战略推理能力是一项复杂的任务。这种复杂性源于复杂的玩家可能对其他人有非均衡的信念，导致非均衡的行为。在我们的研究中，我们将人类参与者与已知完全理性的计算机玩家配对。通过使用机器人玩家，我们能够将有限的推理能力与信念形成和社会偏差相区分开来。我们的研究结果表明，当与机器人配对时，被试表现出始终较高的理性水平，并在不同游戏中保持稳定的理性水平，相比之下与人类配对时则不然。这表明战略推理可能的确是个体的一种一贯特征。此外，确定的理性限制可以作为评估个体对他人信念适当控制时的战略能力的指标。

    Determining an individual's strategic reasoning capability based solely on choice data is a complex task. This complexity arises because sophisticated players might have non-equilibrium beliefs about others, leading to non-equilibrium actions. In our study, we pair human participants with computer players known to be fully rational. This use of robot players allows us to disentangle limited reasoning capacity from belief formation and social biases. Our results show that, when paired with robots, subjects consistently demonstrate higher levels of rationality and maintain stable rationality levels across different games compared to when paired with humans. This suggests that strategic reasoning might indeed be a consistent trait in individuals. Furthermore, the identified rationality limits could serve as a measure for evaluating an individual's strategic capacity when their beliefs about others are adequately controlled.
    
[^7]: 公共债务的财政成本和政府支出冲击

    The Fiscal Cost of Public Debt and Government Spending Shocks. (arXiv:2309.07371v1 [econ.GN])

    [http://arxiv.org/abs/2309.07371](http://arxiv.org/abs/2309.07371)

    本研究通过使用美国历史数据，发现当债务成为财政负担时，政府通过限制债务发行来应对支出冲击，导致财政政策失去了刺激经济活动的能力。

    

    本文研究了公共债务成本如何塑造财政政策及其对经济的影响。利用美国的历史数据，我表明当债务服务成为财政负担时，政府通过限制债务发行来应对支出冲击。因此，初始冲击只在短期内引发对公共支出的有限增加，并且甚至导致长期内的支出逆转。在这种情况下，财政政策失去了刺激经济活动的能力。这一结果是由于财政当局限制自身借款能力以确保公共债务可持续性。这些发现在多个识别和估计策略下都具有鲁棒性。

    This paper investigates how the cost of public debt shapes fiscal policy and its effect on the economy. Using U.S. historical data, I show that when servicing the debt creates a fiscal burden, the government responds to spending shocks by limiting debt issuance. As a result, the initial shock triggers only a limited increase in public spending in the short run, and even leads to spending reversal in the long run. Under these conditions, fiscal policy loses its ability to stimulate economic activity. This outcome arises as the fiscal authority limits its own ability to borrow to ensure public debt sustainability. These findings are robust to several identification and estimation strategies.
    
[^8]: 主妇劳动对GDP计算的影响

    The effect of housewife labor on gdp calculations. (arXiv:2309.07160v1 [econ.GN])

    [http://arxiv.org/abs/2309.07160](http://arxiv.org/abs/2309.07160)

    本研究通过理论分析和经验建模，探讨了家庭主妇劳动对GDP计算的影响。研究发现，人类特征的变化导致了有限理性个体特征的形成，女性更倾向于成为家庭主妇的选择。实证分析揭示了家庭主妇劳动的经济价值。

    

    本研究试图通过理论分析揭示劳动力的演化发展。以衡量社会福利的GDP为例，试图通过经验建模来衡量家庭主妇劳动的经济价值。为此，首先质疑了正统（主流）经济理论中劳动的概念；然后，摒弃劳动就业关系，考察了在资本主义体制下失业的无偿家庭主妇劳动对GDP的影响。在理论分析中，确定了人类特征的变化使其远离理性，形成了有限理性和异质个体特征的限制理性个体特征。将女性定义为异质个体的新例子，因为她们更适合有限理性个体的定义，即更倾向于成为家庭主妇。在本研究的实证分析中，对家庭主妇劳动进行了研究。

    In this study, the evolutionary development of labor has been tried to be revealed based on theoretical analysis. Using the example of gdp, which is an indicator of social welfare, the economic value of the labor of housewives was tried to be measured with an empirical modeling. To this end; first of all, the concept of labor was questioned in orthodox (mainstream) economic theories; then, by abstracting from the labor-employment relationship, it was examined what effect the labor of unpaid housewives who are unemployed in the capitalist system could have on gdp. In theoretical analysis; It has been determined that the changing human profile moves away from rationality and creates limited rationality and, accordingly, a heterogeneous individual profile. Women were defined as the new example of heterogeneous individuals, as those who best fit the definition of limited rational individuals because they prefer to be housewives. In the empirical analysis of the study, housewife labor was t
    
[^9]: 最小二乘蒙特卡罗中的有限差分解法

    Finite Difference Solution Ansatz approach in Least-Squares Monte Carlo. (arXiv:2305.09166v1 [q-fin.GN])

    [http://arxiv.org/abs/2305.09166](http://arxiv.org/abs/2305.09166)

    本文提出了一种通用的数值方案，使用低维有限差分法的精确解来构建条件期望继续支付的假设，并将其用于线性回归，以提高在美式期权定价中最小二乘蒙特卡罗方法的精确性。

    

    本文提出了一种简单而有效的方法，以提高在美式期权定价中最小二乘蒙特卡罗方法的精确性。关键思想是使用低维有限差分法的精确解来构建条件期望继续支付的假设，用于线性回归。该方法在解决向后偏微分方程和蒙特卡罗模拟方面建立了桥梁，旨在实现两者的最佳结合。我们通过实际示例说明该技术，包括百慕大期权和最差发行人可赎回票据。该方法可被视为跨越各种资产类别的通用数值方案，特别是在任意维度下，作为定价美式衍生产品的准确方法。

    This article presents a simple but effective approach to improve the accuracy of Least-Squares Monte Carlo for American-style options. The key idea is to construct the ansatz of conditional expected continuation payoff using the exact solution from low dimensional finite difference methods, to be used in linear regression. This approach builds a bridge between solving backward partial differential equations and a Monte Carlo simulation, aiming at achieving the best of both worlds. We illustrate the technique with realistic examples including Bermuda options and worst of issuer callable notes. The method can be considered as a generic numerical scheme across various asset classes, in particular, as an accurate method for pricing American-style derivatives under arbitrary dimensions.
    

