# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Efficient Variational Inference for Large Skew-t Copulas with Application to Intraday Equity Returns.](http://arxiv.org/abs/2308.05564) | 本研究提出一种快速而准确的贝叶斯变分推理方法，用于估计大规模偏t乌鸦因子勾结模型。该方法能够捕捉到金融数据中的不对称和极端尾部相关性，以及股票对之间的异质性非对称依赖。 |
| [^2] | [Financial Fraud Detection: A Comparative Study of Quantum Machine Learning Models.](http://arxiv.org/abs/2308.05237) | 该研究通过比较研究四种量子机器学习模型，证明了量子支持向量分类器模型在金融欺诈检测方面具有最高性能，并为量子机器学习在欺诈检测领域的未来发展提供了重要见解。 |
| [^3] | ["Generate" the Future of Work through AI: Empirical Evidence from Online Labor Markets.](http://arxiv.org/abs/2308.05201) | 这项研究通过利用ChatGPT作为外生冲击，揭示了其对在线劳动市场的影响。结果显示，直接接触ChatGPT的任务和自由职业者的交易量显著下降，但适应新技术并提供增强人工智能的服务的自由职业者仍能获得利益。 |
| [^4] | [SmartDCA superiority.](http://arxiv.org/abs/2308.05200) | 本文介绍了一种新的投资策略SmartDCA，它通过根据价格水平调整资产购买，相比传统的DCA方法，能够提高投资的效率。在数学分析中，证明了SmartDCA的优越性。此外，在引入有界版本的SmartDCA时，作者提出了两个新的均值定义，以解决可能导致无限投资的问题。 |
| [^5] | [Empirical Evidence for the New Definitions in Financial Markets.](http://arxiv.org/abs/2305.03468) | 研究证实了新的金融市场定义准确反映了投资者行为，并提供了投资策略方面的建议。 |
| [^6] | [Conditional Generative Models for Learning Stochastic Processes.](http://arxiv.org/abs/2304.10382) | 提出了一种称为 C-qGAN 的框架，利用量子电路结构实现了有效的状态准备过程，可以利用该方法加速蒙特卡罗分析等算法，并将其应用于亚式期权衍生品定价的任务中。 |
| [^7] | [Understanding Model Complexity for temporal tabular and multi-variate time series, case study with Numerai data science tournament.](http://arxiv.org/abs/2303.07925) | 本文采用 Numerai 数据科学竞赛的数据，探究了多变量时间序列建模中不同特征工程和降维方法的应用；提出了一种新的集成方法，用于高维时间序列建模，该方法在通用性、鲁棒性和效率上优于一些深度学习模型。 |
| [^8] | [Dynamic Feature Engineering and model selection methods for temporal tabular datasets with regime changes.](http://arxiv.org/abs/2301.00790) | 本文提出了一种新的机器学习管道，用于在数据制度变化下对时序面板数据集的预测进行排名。使用梯度提升决策树（GBDT）并结合dropout技术的模型具有良好的性能和泛化能力，而动态特征中和则是一种高效而不需要重新训练模型就可以应用于任何机器学习模型中的后处理技术。 |
| [^9] | [A stochastic volatility model for the valuation of temperature derivatives.](http://arxiv.org/abs/2209.05918) | 本文提出了一种新的随机波动模型，用于温度的估值，该模型在处理极端事件时更加谨慎，同时保持可追踪性，并提供了一种有效计算天气衍生品平均支付的方法。 |
| [^10] | [Cost-efficient Payoffs under Model Ambiguity.](http://arxiv.org/abs/2207.02948) | 该论文研究了在模型不确定性下寻找成本效益收益的问题，并确定了鲁棒成本效益收益的解决方案。研究表明最大最小鲁棒期望效用的解也是鲁棒成本效益收益的解。通过示例说明了不确定性对收益的影响。 |
| [^11] | [Polarization and Quid Pro Quo: The Role of Party Cohesiveness.](http://arxiv.org/abs/2205.07486) | 研究表明，一个利益集团是否能够利用不断增加的意识形态和情感极化获利取决于两个政党社交网络相对凝聚度的聚合度量，而在没有意识形态 polarization 的情况下，两个政党社交网络的相对凝聚度则变得无关紧要。 |

# 详细

[^1]: 大规模偏t乌鸦勾结的高效变分推理及其在股票收益率中的应用

    Efficient Variational Inference for Large Skew-t Copulas with Application to Intraday Equity Returns. (arXiv:2308.05564v1 [econ.EM])

    [http://arxiv.org/abs/2308.05564](http://arxiv.org/abs/2308.05564)

    本研究提出一种快速而准确的贝叶斯变分推理方法，用于估计大规模偏t乌鸦因子勾结模型。该方法能够捕捉到金融数据中的不对称和极端尾部相关性，以及股票对之间的异质性非对称依赖。

    

    大规模偏t乌鸦因子勾结模型对金融数据建模具有吸引力，因为它们允许不对称和极端的尾部相关性。我们展示了Azzalini和Capitanio（2003）所隐含的乌鸦勾结在成对非对称依赖性方面比两种流行的乌鸦勾结更高。在高维情况下，对该乌鸦勾结的估计具有挑战性，我们提出了一种快速而准确的贝叶斯变分推理方法来解决这个问题。该方法使用条件高斯生成表示法定义了一个可以准确近似的附加后验。使用快速随机梯度上升算法来解决变分优化。这种新的方法被用来估计2017年至2021年间93个美国股票的股票收益率的勾结模型。除了成对相关性的变化外，该勾结还捕捉到了股票对之间的非对称依赖的大量异质性。

    Large skew-t factor copula models are attractive for the modeling of financial data because they allow for asymmetric and extreme tail dependence. We show that the copula implicit in the skew-t distribution of Azzalini and Capitanio (2003) allows for a higher level of pairwise asymmetric dependence than two popular alternative skew-t copulas. Estimation of this copula in high dimensions is challenging, and we propose a fast and accurate Bayesian variational inference (VI) approach to do so. The method uses a conditionally Gaussian generative representation of the skew-t distribution to define an augmented posterior that can be approximated accurately. A fast stochastic gradient ascent algorithm is used to solve the variational optimization. The new methodology is used to estimate copula models for intraday returns from 2017 to 2021 on 93 U.S. equities. The copula captures substantial heterogeneity in asymmetric dependence over equity pairs, in addition to the variability in pairwise co
    
[^2]: 金融欺诈检测：量子机器学习模型的比较研究

    Financial Fraud Detection: A Comparative Study of Quantum Machine Learning Models. (arXiv:2308.05237v1 [quant-ph])

    [http://arxiv.org/abs/2308.05237](http://arxiv.org/abs/2308.05237)

    该研究通过比较研究四种量子机器学习模型，证明了量子支持向量分类器模型在金融欺诈检测方面具有最高性能，并为量子机器学习在欺诈检测领域的未来发展提供了重要见解。

    

    在本研究中，针对金融欺诈检测进行了四种量子机器学习（QML）模型的比较研究。我们证明了量子支持向量分类器模型的性能最高，欺诈和非欺诈类别的F1分数均为0.98。其他模型如变分量子分类器、估计量子神经网络和采样器量子神经网络展示了有希望的结果，推动了QML在金融应用中的潜力。虽然它们存在一定的限制，但所得到的洞察为未来的增强和优化策略铺平了道路。然而，挑战存在，包括需要更高效的量子算法和更大更复杂的数据集。该文章提供了克服当前限制的解决方案，并为量子机器学习在欺诈检测领域提供了新的见解，对其未来发展有重要影响。

    In this research, a comparative study of four Quantum Machine Learning (QML) models was conducted for fraud detection in finance. We proved that the Quantum Support Vector Classifier model achieved the highest performance, with F1 scores of 0.98 for fraud and non-fraud classes. Other models like the Variational Quantum Classifier, Estimator Quantum Neural Network (QNN), and Sampler QNN demonstrate promising results, propelling the potential of QML classification for financial applications. While they exhibit certain limitations, the insights attained pave the way for future enhancements and optimisation strategies. However, challenges exist, including the need for more efficient Quantum algorithms and larger and more complex datasets. The article provides solutions to overcome current limitations and contributes new insights to the field of Quantum Machine Learning in fraud detection, with important implications for its future development.
    
[^3]: 通过人工智能"生成"工作：在线劳动市场的经验证据

    "Generate" the Future of Work through AI: Empirical Evidence from Online Labor Markets. (arXiv:2308.05201v1 [cs.AI])

    [http://arxiv.org/abs/2308.05201](http://arxiv.org/abs/2308.05201)

    这项研究通过利用ChatGPT作为外生冲击，揭示了其对在线劳动市场的影响。结果显示，直接接触ChatGPT的任务和自由职业者的交易量显著下降，但适应新技术并提供增强人工智能的服务的自由职业者仍能获得利益。

    

    随着通用生成式人工智能的出现，对其对劳动市场的影响的兴趣不断增加。为了填补现有的实证空白，我们将ChatGPT的推出解释为一种外生冲击，并采用差异法来量化其对在线劳动市场中与文本相关的工作和自由职业者的影响。我们的结果显示，直接接触ChatGPT的任务和自由职业者的交易量显著下降。此外，这种下降在相对较高的过去交易量或较低的质量标准下尤为显著。然而，并非所有服务提供商都普遍经历了负面影响。随后的分析表明，在这个转型期间，能够适应新进展并提供增强人工智能技术的服务的自由职业者可以获得可观的利益。因此，虽然ChatGPT的出现有可能替代人力劳动

    With the advent of general-purpose Generative AI, the interest in discerning its impact on the labor market escalates. In an attempt to bridge the extant empirical void, we interpret the launch of ChatGPT as an exogenous shock, and implement a Difference-in-Differences (DID) approach to quantify its influence on text-related jobs and freelancers within an online labor marketplace. Our results reveal a significant decrease in transaction volume for gigs and freelancers directly exposed to ChatGPT. Additionally, this decline is particularly marked in units of relatively higher past transaction volume or lower quality standards. Yet, the negative effect is not universally experienced among service providers. Subsequent analyses illustrate that freelancers proficiently adapting to novel advancements and offering services that augment AI technologies can yield substantial benefits amidst this transformative period. Consequently, even though the advent of ChatGPT could conceivably substitute
    
[^4]: SmartDCA优势

    SmartDCA superiority. (arXiv:2308.05200v1 [q-fin.PM])

    [http://arxiv.org/abs/2308.05200](http://arxiv.org/abs/2308.05200)

    本文介绍了一种新的投资策略SmartDCA，它通过根据价格水平调整资产购买，相比传统的DCA方法，能够提高投资的效率。在数学分析中，证明了SmartDCA的优越性。此外，在引入有界版本的SmartDCA时，作者提出了两个新的均值定义，以解决可能导致无限投资的问题。

    

    美元成本平均法(DCA)是一种广泛应用的技术，用于减轻长期投资中对升值资产的波动性。然而，DCA的低效性来源于在不考虑市场状况的情况下固定投资金额。本文介绍了一种更高效的方法，我们称之为SmartDCA，该方法基于价格水平来调整资产购买。SmartDCA的简单性使得我们能够进行严格的数学分析，通过应用Cauchy-Schwartz不等式和Lehmer平均数，我们证明了它的优越性。我们进一步将我们的分析扩展到了我们所称的ρ-SmartDCA，在这种情况下，投资金额被提升到ρ的幂。我们证明了较高的ρ值可以带来更好的性能。然而，这种方法可能导致无限的投资。为了解决这个问题，我们引入了SmartDCA的有界版本，利用了我们称之为准Lehmer平均数的两个新的均值定义。

    Dollar-Cost Averaging (DCA) is a widely used technique to mitigate volatility in long-term investments of appreciating assets. However, the inefficiency of DCA arises from fixing the investment amount regardless of market conditions. In this paper, we present a more efficient approach that we name SmartDCA, which consists in adjusting asset purchases based on price levels. The simplicity of SmartDCA allows for rigorous mathematical analysis, enabling us to establish its superiority through the application of Cauchy-Schwartz inequality and Lehmer means. We further extend our analysis to what we refer to as $\rho$-SmartDCA, where the invested amount is raised to the power of $\rho$. We demonstrate that higher values of $\rho$ lead to enhanced performance. However, this approach may result in unbounded investments. To address this concern, we introduce a bounded version of SmartDCA, taking advantage of two novel mean definitions that we name quasi-Lehmer means. The bounded SmartDCA is spe
    
[^5]: 金融市场新定义的经验证据

    Empirical Evidence for the New Definitions in Financial Markets. (arXiv:2305.03468v1 [q-fin.GN])

    [http://arxiv.org/abs/2305.03468](http://arxiv.org/abs/2305.03468)

    研究证实了新的金融市场定义准确反映了投资者行为，并提供了投资策略方面的建议。

    

    本研究给出了支持金融市场新定义的经验证据。分析了1889-1978年美国金融市场投资者的风险态度，结果表明，1977年在投资综合S＆P 500指数的股票投资者是风险规避者。相反，投资美国国债的无风险资产投资者则表现出不足的风险偏爱，这可以被认为是一种风险规避行为。这些发现表明，金融市场新定义准确反映了投资者的行为，应考虑在投资策略中。

    This study presents empirical evidence to support the validity of new definitions in financial markets. The risk attitudes of investors in US financial markets from 1889-1978 are analyzed and the results indicate that equity investors who invested in the composite S&P 500 index were risk-averse in 1977. Conversely, risk-free asset investors who invested in US Treasury bills were found to exhibit not enough risk-loving behavior, which can be considered a type of risk-averse behavior. These findings suggest that the new definitions in financial markets accurately reflect the behavior of investors and should be considered in investment strategies.
    
[^6]: 学习随机过程的有条件生成模型

    Conditional Generative Models for Learning Stochastic Processes. (arXiv:2304.10382v1 [quant-ph])

    [http://arxiv.org/abs/2304.10382](http://arxiv.org/abs/2304.10382)

    提出了一种称为 C-qGAN 的框架，利用量子电路结构实现了有效的状态准备过程，可以利用该方法加速蒙特卡罗分析等算法，并将其应用于亚式期权衍生品定价的任务中。

    

    提出了一种学习多模态分布的框架，称为条件量子生成对抗网络（C-qGAN）。神经网络结构严格采用量子电路，因此被证明能够比当前的方法更有效地表示状态准备过程。这种方法有潜力加速蒙特卡罗分析等算法。特别地，在展示了网络在学习任务中的有效性后，将该技术应用于定价亚式期权衍生品，为未来研究其他路径相关期权打下基础。

    A framework to learn a multi-modal distribution is proposed, denoted as the Conditional Quantum Generative Adversarial Network (C-qGAN). The neural network structure is strictly within a quantum circuit and, as a consequence, is shown to represents a more efficient state preparation procedure than current methods. This methodology has the potential to speed-up algorithms, such as Monte Carlo analysis. In particular, after demonstrating the effectiveness of the network in the learning task, the technique is applied to price Asian option derivatives, providing the foundation for further research on other path-dependent options.
    
[^7]: 通过 Numerai 数据科学竞赛案例，理解时间表格和多变量时间序列的模型复杂度

    Understanding Model Complexity for temporal tabular and multi-variate time series, case study with Numerai data science tournament. (arXiv:2303.07925v1 [cs.LG])

    [http://arxiv.org/abs/2303.07925](http://arxiv.org/abs/2303.07925)

    本文采用 Numerai 数据科学竞赛的数据，探究了多变量时间序列建模中不同特征工程和降维方法的应用；提出了一种新的集成方法，用于高维时间序列建模，该方法在通用性、鲁棒性和效率上优于一些深度学习模型。

    

    本文探究了在多变量时间序列建模中使用不同特征工程和降维方法的应用。利用从 Numerai 数据竞赛创建的特征目标交叉相关时间序列数据集，我们证明在过度参数化的情况下，不同特征工程方法的性能与预测会收敛到可由再生核希尔伯特空间刻画的相同平衡态。我们提出了一种新的集成方法，该方法结合了不同的随机非线性变换，随后采用岭回归模型进行高维时间序列建模。与一些常用的用于序列建模的深度学习模型（如 LSTM 和 transformer）相比，我们的方法更加鲁棒（在不同的随机种子下具有较低的模型方差，且对架构的选择不太敏感），并且更有效率。我们方法的另一个优势在于模型的简单性，因为没有必要使用复杂的深度学习框架。

    In this paper, we explore the use of different feature engineering and dimensionality reduction methods in multi-variate time-series modelling. Using a feature-target cross correlation time series dataset created from Numerai tournament, we demonstrate under over-parameterised regime, both the performance and predictions from different feature engineering methods converge to the same equilibrium, which can be characterised by the reproducing kernel Hilbert space. We suggest a new Ensemble method, which combines different random non-linear transforms followed by ridge regression for modelling high dimensional time-series. Compared to some commonly used deep learning models for sequence modelling, such as LSTM and transformers, our method is more robust (lower model variance over different random seeds and less sensitive to the choice of architecture) and more efficient. An additional advantage of our method is model simplicity as there is no need to use sophisticated deep learning frame
    
[^8]: 面向时序表格数据的动态特征工程和模型选择方法在制度变化下的应用

    Dynamic Feature Engineering and model selection methods for temporal tabular datasets with regime changes. (arXiv:2301.00790v2 [q-fin.CP] UPDATED)

    [http://arxiv.org/abs/2301.00790](http://arxiv.org/abs/2301.00790)

    本文提出了一种新的机器学习管道，用于在数据制度变化下对时序面板数据集的预测进行排名。使用梯度提升决策树（GBDT）并结合dropout技术的模型具有良好的性能和泛化能力，而动态特征中和则是一种高效而不需要重新训练模型就可以应用于任何机器学习模型中的后处理技术。

    

    由于严重的非平稳性，将深度学习算法应用于时序面板数据集是困难的，这可能导致过度拟合的模型在制度变化下性能不佳。在本文中，我们提出了一种新的机器学习管道，用于在数据制度变化下对时序面板数据集的预测进行排名。管道评估不同的机器学习模型，包括梯度提升决策树（GBDT）和具有和不具有简单特征工程的神经网络。我们发现，具有dropout的GBDT模型具有高性能、稳健性和泛化能力，而且相对复杂度较低、计算成本较低。然后，我们展示了在线学习技术可以在预测后处理中用于增强结果。特别地，我们提出了动态特征中和，这是一种无需重新训练模型就可以应用于任何机器学习模型的高效过程。

    The application of deep learning algorithms to temporal panel datasets is difficult due to heavy non-stationarities which can lead to over-fitted models that under-perform under regime changes. In this work we propose a new machine learning pipeline for ranking predictions on temporal panel datasets which is robust under regime changes of data. Different machine-learning models, including Gradient Boosting Decision Trees (GBDTs) and Neural Networks with and without simple feature engineering are evaluated in the pipeline with different settings. We find that GBDT models with dropout display high performance, robustness and generalisability with relatively low complexity and reduced computational cost. We then show that online learning techniques can be used in post-prediction processing to enhance the results. In particular, dynamic feature neutralisation, an efficient procedure that requires no retraining of models and can be applied post-prediction to any machine learning model, impr
    
[^9]: 用于温度衍生品定价的随机波动模型

    A stochastic volatility model for the valuation of temperature derivatives. (arXiv:2209.05918v2 [q-fin.RM] UPDATED)

    [http://arxiv.org/abs/2209.05918](http://arxiv.org/abs/2209.05918)

    本文提出了一种新的随机波动模型，用于温度的估值，该模型在处理极端事件时更加谨慎，同时保持可追踪性，并提供了一种有效计算天气衍生品平均支付的方法。

    

    本文提出了一种新的随机波动模型，用于温度的估值，这是Benth和Benth（2007）提出的Ornstein-Uhlenbeck模型的自然延伸。该模型在保持可追踪性的同时，更加谨慎地处理极端事件。我们提供了一种基于条件最小二乘法的方法，在每日数据上估计参数，并在八个主要的欧洲城市中估计我们的模型。然后，我们展示了如何通过蒙特卡罗和傅里叶变换技术有效地计算天气衍生品的平均支付。这种新模型可以更好地评估与温度波动相关的风险。

    This paper develops a new stochastic volatility model for the temperature that is a natural extension of the Ornstein-Uhlenbeck model proposed by Benth and Benth (2007). This model allows to be more conservative regarding extreme events while keeping tractability. We give a method based on Conditional Least Squares to estimate the parameters on daily data and estimate our model on eight major European cities. We then show how to calculate efficiently the average payoff of weather derivatives both by Monte-Carlo and Fourier transform techniques. This new model allows to better assess the risk related to temperature volatility.
    
[^10]: 模型不确定性下的成本效益收益

    Cost-efficient Payoffs under Model Ambiguity. (arXiv:2207.02948v2 [q-fin.PM] UPDATED)

    [http://arxiv.org/abs/2207.02948](http://arxiv.org/abs/2207.02948)

    该论文研究了在模型不确定性下寻找成本效益收益的问题，并确定了鲁棒成本效益收益的解决方案。研究表明最大最小鲁棒期望效用的解也是鲁棒成本效益收益的解。通过示例说明了不确定性对收益的影响。

    

    Dybvig（1988a，b）在完全市场的条件下解决了寻找最便宜的达成给定目标分布的收益（“成本效益收益”）的问题。然而，在不确定性存在的情况下，收益的分布不再确定。我们研究了在满足一定条件下找到最便宜的收益，其最差情况下的分布随机优于给定目标分布（“鲁棒成本效益收益”）的问题，并确定了解决方案。我们还研究了Gilboa和Schmeidler的最大最小期望效用设置和在可能的非期望效用设置中的鲁棒偏好与“鲁棒成本效益收益”之间的联系。具体来说，我们证明了最大最小鲁棒期望效用的解必然是鲁棒成本效益收益。我们通过包括风险资产的漂移和波动性的不确定性的示例来说明我们的研究。

    Dybvig (1988a,b) solves in a complete market setting the problem of finding a payoff that is cheapest possible in reaching a given target distribution ("cost-efficient payoff"). In the presence of ambiguity, the distribution of a payoff is, however, no longer known with certainty. We study the problem of finding the cheapest possible payoff whose worst-case distribution stochastically dominates a given target distribution ("robust cost-efficient payoff") and determine solutions under certain conditions. We study the link between "robust cost-efficiency" and the maxmin expected utility setting of Gilboa and Schmeidler, as well as more generally with robust preferences in a possibly non-expected utility setting. Specifically, we show that solutions to maxmin robust expected utility are necessarily robust cost-efficient. We illustrate our study with examples involving uncertainty both on the drift and on the volatility of the risky asset.
    
[^11]: 极化和交换：政党凝聚力的作用。

    Polarization and Quid Pro Quo: The Role of Party Cohesiveness. (arXiv:2205.07486v2 [econ.GN] UPDATED)

    [http://arxiv.org/abs/2205.07486](http://arxiv.org/abs/2205.07486)

    研究表明，一个利益集团是否能够利用不断增加的意识形态和情感极化获利取决于两个政党社交网络相对凝聚度的聚合度量，而在没有意识形态 polarization 的情况下，两个政党社交网络的相对凝聚度则变得无关紧要。

    

    什么情况下，利益集团才能利用政党之间的意识形态和情感极化来获取利益？我们研究了一个模型，其中利益集团可信地承诺向立法者支付款项，条件是他们投票支持其支持的政策。议员们看重在党内的朋友投票，如果违反其党派的意识形态优先政策投票，则会受到一种意识形态上的不利。情感上的极化，由于其人际关系的本质，在模型中假设议员重视将其投票决策与反对党议员区分开来。我们的主要发现是，两个政党社交网络相对凝聚度的聚合度量确定了利益集团能否利用不断增加的极化获利。然而，如果两个政党之间没有意识形态 polarization，则相对凝聚度的意义将消失。

    When can an interest group exploit ideological and affective polarization between political parties to its advantage? We study a model where an interest group credibly promises payments to legislators conditional on voting for its favored policy. Legislators value voting as their friends within their party, and suffer an ideological-disutility upon voting against their party's ideologically preferred policy. Affective polarization, owing to its interpersonal nature, is modeled by assuming a legislator values distinguishing her voting decision from legislators in the opposite party. Our main finding is that an aggregate measure of relative cohesiveness of social networks in the two parties determines whether the interest group can profitably exploit increasing polarization. However, the significance of relative cohesiveness vanishes if there is no ideological polarization between the two parties.
    

