# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Adaptive Student's t-distribution with method of moments moving estimator for nonstationary time series.](http://arxiv.org/abs/2304.03069) | 本文提出了一种适用于非平稳时间序列的自适应学生t分布方法，基于方法的一般自适应矩可以使用廉价的指数移动平均值（EMA）来估计参数。 |
| [^2] | [Covert learning and disclosure.](http://arxiv.org/abs/2304.02989) | 本研究研究了一个信息获取和传递的模型，在该模型中，发送者选择有选择性地忽视信息，而不是欺骗接收者。本文阐明了欺骗可能性如何决定发送者选择获取和传递的信息，并确定了发送者和接收者最优的伪造环境。 |
| [^3] | [Efficient OCR for Building a Diverse Digital History.](http://arxiv.org/abs/2304.02737) | 本研究使用对比训练的视觉编码器，将OCR建模为字符级图像检索问题，相比于已有架构更具样本效率和可扩展性，从而使数字历史更具代表性的文献史料得以更好地参与社区。 |
| [^4] | [Measuring Discrete Risks on Infinite Domains: Theoretical Foundations, Conditional Five Number Summaries, and Data Analyses.](http://arxiv.org/abs/2304.02723) | 本文提出了一种新的离散损失分布截断方法，将平滑分位数估计的统计推断从有限域扩展到无限域。并提出一种基于自助法的灵活方法，用于实际中的应用。此外，我们计算了构成条件五数摘要（C5NS）的五个分位数的置信区间，并计算尾部概率，结果表明，平滑分位数方法不仅分类投资组合的尾部风险性，还可对其进行分类。 |
| [^5] | [Prolonged Learning and Hasty Stopping: the Wald Problem with Ambiguity.](http://arxiv.org/abs/2208.14121) | 本文研究了风险厌恶的决策者的信息收集行为，发现与贝叶斯决策者不同，会在面对适度的不确定性时过度实验，并在面对高度不确定性时过早停止实验 |
| [^6] | [Treatment Choice with Nonlinear Regret.](http://arxiv.org/abs/2205.08586) | 该论文针对治疗选择中的不确定性问题，提出了最小化非线性悔恨的方法，并导出了闭合分数形式的有限样本贝叶斯和极小化最优规则，可视为支持治疗的证据强度。 |
| [^7] | [Sharp Bounds in the Latent Index Selection Model.](http://arxiv.org/abs/2012.02390) | 本文研究如何对于一个潜在变量选择模型中相关政策的处理效应，提供了尖锐边界和解析刻画，使用了已识别的边际分布的全部内容，推导依赖于随机顺序理论，方法还能将新的辅助分布假设纳入框架中，并且实证结果表明，这些方法应用到健康保险实验中的推断效果令人满意。 |
| [^8] | [Functional Principal Component Analysis for Cointegrated Functional Time Series.](http://arxiv.org/abs/2011.12781) | 本研究提出了一种改进的FPCA，可用于分析协整的函数时间序列并提供更有效的估计，以及FPCA-based的测试，以检查协整函数时间序列的重要属性。 |
| [^9] | [Distribution Regression with Sample Selection, with an Application to Wage Decompositions in the UK.](http://arxiv.org/abs/1811.11603) | 该论文提出了一种带有样本选择的分布回归模型，旨在研究男女工资差距。研究结果表明，即使在控制就业选择后，仍存在大量无法解释的性别工资差距。 |

# 详细

[^1]: 自适应学生t分布与方法矩移动估计器用于非平稳时间序列

    Adaptive Student's t-distribution with method of moments moving estimator for nonstationary time series. (arXiv:2304.03069v1 [stat.ME])

    [http://arxiv.org/abs/2304.03069](http://arxiv.org/abs/2304.03069)

    本文提出了一种适用于非平稳时间序列的自适应学生t分布方法，基于方法的一般自适应矩可以使用廉价的指数移动平均值（EMA）来估计参数。

    

    真实的时间序列通常是非平稳的，这带来了模型适应的难题。传统方法如GARCH假定任意类型的依赖性。为了避免这种偏差，我们将着眼于最近提出的不可知的移动估计器哲学：在时间$t$找到优化$F_t=\sum_{\tau<t} (1-\eta)^{t-\tau} \ln(\rho_\theta (x_\tau))$移动对数似然的参数，随时间演化。例如，它允许使用廉价的指数移动平均值（EMA）来估计参数，例如绝对中心矩$E[|x-\mu|^p]$随$p\in\mathbb{R}^+$的变化而演化$m_{p,t+1} = m_{p,t} + \eta (|x_t-\mu_t|^p-m_{p,t})$。这种基于方法的一般自适应矩的应用将呈现在学生t分布上，尤其是在经济应用中流行，这里应用于DJIA公司的对数收益率。

    The real life time series are usually nonstationary, bringing a difficult question of model adaptation. Classical approaches like GARCH assume arbitrary type of dependence. To prevent such bias, we will focus on recently proposed agnostic philosophy of moving estimator: in time $t$ finding parameters optimizing e.g. $F_t=\sum_{\tau<t} (1-\eta)^{t-\tau} \ln(\rho_\theta (x_\tau))$ moving log-likelihood, evolving in time. It allows for example to estimate parameters using inexpensive exponential moving averages (EMA), like absolute central moments $E[|x-\mu|^p]$ evolving with $m_{p,t+1} = m_{p,t} + \eta (|x_t-\mu_t|^p-m_{p,t})$ for one or multiple powers $p\in\mathbb{R}^+$. Application of such general adaptive methods of moments will be presented on Student's t-distribution, popular especially in economical applications, here applied to log-returns of DJIA companies.
    
[^2]: 隐秘的学习和披露

    Covert learning and disclosure. (arXiv:2304.02989v1 [econ.TH])

    [http://arxiv.org/abs/2304.02989](http://arxiv.org/abs/2304.02989)

    本研究研究了一个信息获取和传递的模型，在该模型中，发送者选择有选择性地忽视信息，而不是欺骗接收者。本文阐明了欺骗可能性如何决定发送者选择获取和传递的信息，并确定了发送者和接收者最优的伪造环境。

    

    本研究研究了一个信息获取和传递的模型，在该模型中，发送者误报其发现的能力受到限制。在均衡状态下，发送者选择有选择性地忽视信息，而不是欺骗接收者。虽然不会产生欺骗，但我强调了欺骗可能性如何决定发送者选择获取和传递的信息。然后，本文转向比较静态分析，阐明了发送者如何从其声明更可验证中受益，并表明这类似于增加其承诺能力。最后，本文确定了发送者和接收者最优的伪造环境。

    I study a model of information acquisition and transmission in which the sender's ability to misreport her findings is limited. In equilibrium, the sender only influences the receiver by choosing to remain selectively ignorant, rather than by deceiving her about the discoveries. Although deception does not occur, I highlight how deception possibilities determine what information the sender chooses to acquire and transmit. I then turn to comparative statics, characterizing in which sense the sender benefits from her claims being more verifiable, showing this is akin to increasing her commitment power. Finally, I characterize sender- and receiver-optimal falsification environments.
    
[^3]: 建设多样化数字历史的高效OCR

    Efficient OCR for Building a Diverse Digital History. (arXiv:2304.02737v1 [cs.CV])

    [http://arxiv.org/abs/2304.02737](http://arxiv.org/abs/2304.02737)

    本研究使用对比训练的视觉编码器，将OCR建模为字符级图像检索问题，相比于已有架构更具样本效率和可扩展性，从而使数字历史更具代表性的文献史料得以更好地参与社区。

    

    每天有成千上万的用户查阅数字档案，但他们可以使用的信息并不能代表各种文献史料的多样性。典型用于光学字符识别（OCR）的序列到序列架构——联合学习视觉和语言模型——在低资源文献集合中很难扩展，因为学习语言-视觉模型需要大量标记的序列和计算。本研究将OCR建模为字符级图像检索问题，使用对比训练的视觉编码器。因为该模型只学习字符的视觉特征，它比现有架构更具样本效率和可扩展性，能够在现有解决方案失败的情况下实现准确的OCR。关键是，该模型为社区参与在使数字历史更具代表性的文献史料方面开辟了新的途径。

    Thousands of users consult digital archives daily, but the information they can access is unrepresentative of the diversity of documentary history. The sequence-to-sequence architecture typically used for optical character recognition (OCR) - which jointly learns a vision and language model - is poorly extensible to low-resource document collections, as learning a language-vision model requires extensive labeled sequences and compute. This study models OCR as a character level image retrieval problem, using a contrastively trained vision encoder. Because the model only learns characters' visual features, it is more sample efficient and extensible than existing architectures, enabling accurate OCR in settings where existing solutions fail. Crucially, the model opens new avenues for community engagement in making digital history more representative of documentary history.
    
[^4]: 无限域上离散风险的测量：理论基础、条件五数摘要和数据分析

    Measuring Discrete Risks on Infinite Domains: Theoretical Foundations, Conditional Five Number Summaries, and Data Analyses. (arXiv:2304.02723v1 [stat.AP])

    [http://arxiv.org/abs/2304.02723](http://arxiv.org/abs/2304.02723)

    本文提出了一种新的离散损失分布截断方法，将平滑分位数估计的统计推断从有限域扩展到无限域。并提出一种基于自助法的灵活方法，用于实际中的应用。此外，我们计算了构成条件五数摘要（C5NS）的五个分位数的置信区间，并计算尾部概率，结果表明，平滑分位数方法不仅分类投资组合的尾部风险性，还可对其进行分类。

    

    为了适应众多实际情况，本文将平滑分位数估计的统计推断从有限域扩展到无限域。我们通过新设计的离散损失分布截断方法完成此任务。模拟研究展示了这种方法在几种分布（如泊松分布、负二项分布及其零膨胀版本）中的应用，这些分布通常用于模拟保险业中的索赔频率。此外，我们提出了一种非常灵活的基于自助法的方法，用于实际中的应用。我们使用汽车事故数据及其修改，计算构成条件五数摘要（C5NS）的五个分位数的置信区间，并计算尾部概率。结果表明，平滑分位数方法不仅分类投资组合的尾部风险性，还可对其进行分类。

    To accommodate numerous practical scenarios, in this paper we extend statistical inference for smoothed quantile estimators from finite domains to infinite domains. We accomplish the task with the help of a newly designed truncation methodology for discrete loss distributions with infinite domains. A simulation study illustrates the methodology in the case of several distributions, such as Poisson, negative binomial, and their zero inflated versions, which are commonly used in insurance industry to model claim frequencies. Additionally, we propose a very flexible bootstrap-based approach for the use in practice. Using automobile accident data and their modifications, we compute what we have termed the conditional five number summary (C5NS) for the tail risk and construct confidence intervals for each of the five quantiles making up C5NS, and then calculate the tail probabilities. The results show that the smoothed quantile approach classifies the tail riskiness of portfolios not only m
    
[^5]: 风险厌恶的决策者的信息收集行为研究

    Prolonged Learning and Hasty Stopping: the Wald Problem with Ambiguity. (arXiv:2208.14121v2 [econ.TH] UPDATED)

    [http://arxiv.org/abs/2208.14121](http://arxiv.org/abs/2208.14121)

    本文研究了风险厌恶的决策者的信息收集行为，发现与贝叶斯决策者不同，会在面对适度的不确定性时过度实验，并在面对高度不确定性时过早停止实验

    

    本文研究了风险厌恶的决策者的信息收集行为，探讨了在何种情况下他们决定进行可逆性行为之前需要多长时间收集信息。我们发现与贝叶斯决策者相比，考虑到不确定性，非理性决策者会在面对适度的不确定性时过分实验，而在面对高度不确定性时可能会过早停止实验。在后一种情况下，决策者的停止规则在信念上是非单增的，并具有随机停止的特征。

    This paper studies sequential information acquisition by an ambiguity-averse decision maker (DM), who decides how long to collect information before taking an irreversible action. The agent optimizes against the worst-case belief and updates prior by prior. We show that the consideration of ambiguity gives rise to rich dynamics: compared to the Bayesian DM, the DM here tends to experiment excessively when facing modest uncertainty and, to counteract it, may stop experimenting prematurely when facing high uncertainty. In the latter case, the DM's stopping rule is non-monotonic in beliefs and features randomized stopping.
    
[^6]: 非线性悔恨下的治疗选择

    Treatment Choice with Nonlinear Regret. (arXiv:2205.08586v3 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2205.08586](http://arxiv.org/abs/2205.08586)

    该论文针对治疗选择中的不确定性问题，提出了最小化非线性悔恨的方法，并导出了闭合分数形式的有限样本贝叶斯和极小化最优规则，可视为支持治疗的证据强度。

    

    文献一般侧重于最小化福利悔恨的平均值，由于采样不确定性，这可能导致不良的治疗选择。我们建议最小化非线性悔恨的平均值，并表明可接受的规则为非线性悔恨的分数规则。针对均方悔恨，我们推导出有限样本贝叶斯和极小化最优规则的闭合分数形式。我们的方法基于决策理论，扩展到极限实验。治疗分数可以看作是支持治疗的证据强度。我们将我们的框架应用于正常回归模型和随机实验的样本量计算。

    The literature focuses on minimizing the mean of welfare regret, which can lead to undesirable treatment choice due to sampling uncertainty. We propose to minimize the mean of a nonlinear transformation of regret and show that admissible rules are fractional for nonlinear regret. Focusing on mean square regret, we derive closed-form fractions for finite-sample Bayes and minimax optimal rules. Our approach is grounded in decision theory and extends to limit experiments. The treatment fractions can be viewed as the strength of evidence favoring treatment. We apply our framework to a normal regression model and sample size calculations in randomized experiments.
    
[^7]: 潜在变量选择模型中的尖锐边界

    Sharp Bounds in the Latent Index Selection Model. (arXiv:2012.02390v2 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2012.02390](http://arxiv.org/abs/2012.02390)

    本文研究如何对于一个潜在变量选择模型中相关政策的处理效应，提供了尖锐边界和解析刻画，使用了已识别的边际分布的全部内容，推导依赖于随机顺序理论，方法还能将新的辅助分布假设纳入框架中，并且实证结果表明，这些方法应用到健康保险实验中的推断效果令人满意。

    

    偏识别文献中一个基本问题是：关于政策相关的参数，我们能从可观测的外生变化中学到什么，而这些参数并不一定通过外生变化被点识别。本文通过对 Vytlacil(2002) 中的潜在变量选择模型进行尖锐、解析的刻画和边界提出了一个答案，涉及到一类重要的政策相关的处理效应，包括边际处理效应和其中的线性泛函。这些尖锐边界利用了已识别的边际分布的全部内容，解析推导依赖于随机顺序理论。所提出的方法还使得可以利用新的辅助分布假设尖锐地将其纳入潜在变量选择框架中。在实证方面，本文应用这些方法来研究医疗补助对俄勒冈州健康保险实验中急诊室利用率的影响，显示基于外推的预测与随机分配实验的结果有很好的吻合性。

    A fundamental question underlying the literature on partial identification is: what can we learn about parameters that are relevant for policy but not necessarily point-identified by the exogenous variation we observe? This paper provides an answer in terms of sharp, analytic characterizations and bounds for an important class of policy-relevant treatment effects, consisting of marginal treatment effects and linear functionals thereof, in the latent index selection model as formalized in Vytlacil (2002). The sharp bounds use the full content of identified marginal distributions, and analytic derivations rely on the theory of stochastic orders. The proposed methods also make it possible to sharply incorporate new auxiliary assumptions on distributions into the latent index selection framework. Empirically, I apply the methods to study the effects of Medicaid on emergency room utilization in the Oregon Health Insurance Experiment, showing that the predictions from extrapolations based on
    
[^8]: Cointegrated Functional Time Series的函数主成分分析

    Functional Principal Component Analysis for Cointegrated Functional Time Series. (arXiv:2011.12781v6 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2011.12781](http://arxiv.org/abs/2011.12781)

    本研究提出了一种改进的FPCA，可用于分析协整的函数时间序列并提供更有效的估计，以及FPCA-based的测试，以检查协整函数时间序列的重要属性。

    

    函数主成分分析（FPCA）在发展函数时间序列分析方面发挥了重要作用。本文研究了如何利用FPCA分析协整的函数时间序列，并提出了一种改进的FPCA作为一种新的统计工具。我们改进后的FPCA不仅提供了协整向量的渐近更有效的估计，还引导了基于FPCA的测试，以检查协整函数时间序列的重要属性。

    Functional principal component analysis (FPCA) has played an important role in the development of functional time series analysis. This note investigates how FPCA can be used to analyze cointegrated functional time series and proposes a modification of FPCA as a novel statistical tool. Our modified FPCA not only provides an asymptotically more efficient estimator of the cointegrating vectors, but also leads to novel FPCA-based tests for examining essential properties of cointegrated functional time series.
    
[^9]: 带有样本选择的分布回归：应用于英国工资分解的研究

    Distribution Regression with Sample Selection, with an Application to Wage Decompositions in the UK. (arXiv:1811.11603v5 [econ.EM] UPDATED)

    [http://arxiv.org/abs/1811.11603](http://arxiv.org/abs/1811.11603)

    该论文提出了一种带有样本选择的分布回归模型，旨在研究男女工资差距。研究结果表明，即使在控制就业选择后，仍存在大量无法解释的性别工资差距。

    

    我们在内生样本选择的情况下开发了一个分布回归模型。该模型是Heckman选择模型的半参数推广，可以适应更丰富的协变量对结果分布和选择过程异质性模式的影响，同时允许与高斯误差结构有显著偏差的情况，而仍然保持与经典模型同样的可处理性。该模型适用于连续、离散和混合结果。我们提供了识别、估计和推断方法，并将其应用于获得英国工资分解。我们将男女工资分布差异分解为成分、工资结构、选择结构和选择排序效应。在控制内生就业选择后，我们仍然发现显著的性别工资差距-在未解释组成成分的情况下，从21％到40％在（潜在的）提供工资分布中。

    We develop a distribution regression model under endogenous sample selection. This model is a semi-parametric generalization of the Heckman selection model. It accommodates much richer effects of the covariates on outcome distribution and patterns of heterogeneity in the selection process, and allows for drastic departures from the Gaussian error structure, while maintaining the same level tractability as the classical model. The model applies to continuous, discrete and mixed outcomes. We provide identification, estimation, and inference methods, and apply them to obtain wage decomposition for the UK. Here we decompose the difference between the male and female wage distributions into composition, wage structure, selection structure, and selection sorting effects. After controlling for endogenous employment selection, we still find substantial gender wage gap -- ranging from 21\% to 40\% throughout the (latent) offered wage distribution that is not explained by composition. We also un
    

