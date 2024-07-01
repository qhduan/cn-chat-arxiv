# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [FishNet: Deep Neural Networks for Low-Cost Fish Stock Estimation](https://arxiv.org/abs/2403.10916) | 提出了FishNet，一个自动化计算机视觉系统，利用低成本数码相机图像执行鱼类分类和大小估算，具有高准确性和精度。 |
| [^2] | [Generative Probabilistic Forecasting with Applications in Market Operations](https://arxiv.org/abs/2403.05743) | 提出了一种基于Wiener-Kallianpur创新表示的生成式概率预测方法，包括自编码器和新颖的深度学习算法，具有渐近最优性和结构收敛性质，适用于实时市场运营中的高动态和波动时间序列。 |
| [^3] | [Uncovering the Sino-US dynamic risk spillovers effects: Evidence from agricultural futures markets](https://arxiv.org/abs/2403.01745) | 通过TVP-VAR-DY模型和分位数方法，研究发现CBOT玉米、大豆和小麦是主要的风险传播者，DCE玉米和大豆是主要的风险接受者，并且突发事件或增加的经济不确定性可能导致整体风险溢出。 |
| [^4] | [FinAgent: A Multimodal Foundation Agent for Financial Trading: Tool-Augmented, Diversified, and Generalist](https://arxiv.org/abs/2402.18485) | FinAgent是一个多模态基础代理，通过工具增强用于金融交易，具有独特的双重反射模块，可以处理多样化的数据并快速适应市场动态。 |
| [^5] | [On Finding Bi-objective Pareto-optimal Fraud Prevention Rule Sets for Fintech Applications.](http://arxiv.org/abs/2311.00964) | 本文研究了在金融科技应用中寻找高质量的双目标 Pareto 最优欺诈预防规则集的问题。通过采用 Pareto 最优性概念和启发式框架 PORS，我们成功提出了一组非支配的规则子集，并通过实证评估证明了其有效性。 |
| [^6] | [Portfolio Optimization in a Market with Hidden Gaussian Drift and Randomly Arriving Expert Opinions: Modeling and Theoretical Results.](http://arxiv.org/abs/2308.02049) | 本研究分析了金融市场中投资组合优化的问题，考虑了股票回报的隐藏高斯漂移以及随机到达的专家意见，应用卡尔曼滤波技术获得了隐藏漂移的估计，并通过动态规划方法解决了功用最大化问题。 |
| [^7] | [Socioeconomic agents as active matter in nonequilibrium Sakoda-Schelling models.](http://arxiv.org/abs/2307.14270) | 该研究通过考虑Sakoda-Schelling模型中的职业模型，揭示了社会经济代理人模型中的非平衡动力学，并在平均场近似下将其映射为主动物质描述。通过研究非互惠性互动，展示了非稳态的宏观行为。这一研究为地理相关的基于代理人的模型提供了统一的框架，有助于同时考虑人口和价格动态。 |

# 详细

[^1]: FishNet:用于低成本鱼类存栏估算的深度神经网络

    FishNet: Deep Neural Networks for Low-Cost Fish Stock Estimation

    [https://arxiv.org/abs/2403.10916](https://arxiv.org/abs/2403.10916)

    提出了FishNet，一个自动化计算机视觉系统，利用低成本数码相机图像执行鱼类分类和大小估算，具有高准确性和精度。

    

    鱼类库存评估通常需要由分类专家进行手工鱼类计数，这既耗时又昂贵。我们提出了一个自动化的计算机视觉系统，可以从使用低成本数码相机拍摄的图像中执行分类和鱼类大小估算。该系统首先利用Mask R-CNN执行目标检测和分割，以识别包含多条鱼的图像中的单条鱼，这些鱼可能由不同物种组成。然后，每个鱼类被分类并使用单独的机器学习模型预测长度。这些模型训练于包含50,000张手工注释图像的数据集，其中包含163种不同长度从10厘米到250厘米的鱼类。在保留的测试数据上评估，我们的系统在鱼类分割任务上达到了92%的交并比，单一鱼类分类准确率为89%，平均误差为2.3厘米。

    arXiv:2403.10916v1 Announce Type: cross  Abstract: Fish stock assessment often involves manual fish counting by taxonomy specialists, which is both time-consuming and costly. We propose an automated computer vision system that performs both taxonomic classification and fish size estimation from images taken with a low-cost digital camera. The system first performs object detection and segmentation using a Mask R-CNN to identify individual fish from images containing multiple fish, possibly consisting of different species. Then each fish species is classified and the predicted length using separate machine learning models. These models are trained on a dataset of 50,000 hand-annotated images containing 163 different fish species, ranging in length from 10cm to 250cm. Evaluated on held-out test data, our system achieves a $92\%$ intersection over union on the fish segmentation task, a $89\%$ top-1 classification accuracy on single fish species classification, and a $2.3$~cm mean error on
    
[^2]: 具有市场运营应用的生成式概率预测

    Generative Probabilistic Forecasting with Applications in Market Operations

    [https://arxiv.org/abs/2403.05743](https://arxiv.org/abs/2403.05743)

    提出了一种基于Wiener-Kallianpur创新表示的生成式概率预测方法，包括自编码器和新颖的深度学习算法，具有渐近最优性和结构收敛性质，适用于实时市场运营中的高动态和波动时间序列。

    

    本文提出了一种新颖的生成式概率预测方法，该方法源自于非参数时间序列的Wiener-Kallianpur创新表示。在生成人工智能的范式下，所提出的预测架构包括一个自编码器，将非参数多变量随机过程转化为规范的创新序列，从中根据过去样本生成未来时间序列样本，条件是它们的概率分布取决于过去样本。提出了一种新的深度学习算法，将潜在过程限制为具有匹配自编码器输入-输出条件概率分布的独立同分布序列。建立了所提出的生成式预测方法的渐近最优性和结构收敛性质。该方法在实时市场运营中涉及高度动态和波动时间序列的三个应用方面。

    arXiv:2403.05743v1 Announce Type: cross  Abstract: This paper presents a novel generative probabilistic forecasting approach derived from the Wiener-Kallianpur innovation representation of nonparametric time series. Under the paradigm of generative artificial intelligence, the proposed forecasting architecture includes an autoencoder that transforms nonparametric multivariate random processes into canonical innovation sequences, from which future time series samples are generated according to their probability distributions conditioned on past samples. A novel deep-learning algorithm is proposed that constrains the latent process to be an independent and identically distributed sequence with matching autoencoder input-output conditional probability distributions. Asymptotic optimality and structural convergence properties of the proposed generative forecasting approach are established. Three applications involving highly dynamic and volatile time series in real-time market operations a
    
[^3]: 揭示中美动态风险溢出效应：来自农业期货市场的证据

    Uncovering the Sino-US dynamic risk spillovers effects: Evidence from agricultural futures markets

    [https://arxiv.org/abs/2403.01745](https://arxiv.org/abs/2403.01745)

    通过TVP-VAR-DY模型和分位数方法，研究发现CBOT玉米、大豆和小麦是主要的风险传播者，DCE玉米和大豆是主要的风险接受者，并且突发事件或增加的经济不确定性可能导致整体风险溢出。

    

    农产品在人类发展中扮演着至关重要的角色。随着经济全球化和农产品金融化的不断推进，不同农产品期货之间的相互关联变得更加紧密。我们利用TVP-VAR-DY模型结合分位数方法，从2014年7月9日至2022年12月31日度量了美国和中国期货交易所的11种农产品期货之间的风险溢出情况。研究得出了几个重大发现。首先，CBOT玉米、大豆和小麦被确定为主要的风险传播者，DCE玉米和大豆则是主要的风险接受者。其次，突发事件或增加的经济不确定性可能会增加整体风险溢出。第三，基于动态方向溢出结果，农产品期货之间存在风险溢出的聚合情况。最后，中央农产品期货在条件均值下是

    arXiv:2403.01745v1 Announce Type: new  Abstract: Agricultural products play a critical role in human development. With economic globalization and the financialization of agricultural products continuing to advance, the interconnections between different agricultural futures have become closer. We utilize a TVP-VAR-DY model combined with the quantile method to measure the risk spillover between 11 agricultural futures on the futures exchanges of US and China from July 9,2014, to December 31,2022. This study yielded several significant findings. Firstly, CBOT corn, soybean, and wheat were identified as the primary risk transmitters, with DCE corn and soybean as the main risk receivers. Secondly, sudden events or increased eco- nomic uncertainty can increase the overall risk spillovers. Thirdly, there is an aggregation of risk spillovers amongst agricultural futures based on the dynamic directional spillover results. Lastly, the central agricultural futures under the conditional mean are 
    
[^4]: FinAgent: 用于金融交易的多模态基础代理：工具增强、多样化和通用

    FinAgent: A Multimodal Foundation Agent for Financial Trading: Tool-Augmented, Diversified, and Generalist

    [https://arxiv.org/abs/2402.18485](https://arxiv.org/abs/2402.18485)

    FinAgent是一个多模态基础代理，通过工具增强用于金融交易，具有独特的双重反射模块，可以处理多样化的数据并快速适应市场动态。

    

    金融交易是市场的重要组成部分，受到新闻、价格和K线图等多模态信息构成的信息景观的影响，涵盖了诸如量化交易和不同资产的高频交易等多样化任务。尽管深度学习和强化学习等先进AI技术在金融领域得到广泛应用，但它们在金融交易任务中的应用往往面临着多模态数据处理不足和跨不同任务有限泛化能力的挑战。为了解决这些挑战，我们提出了FinAgent，一个具有工具增强功能的多模态基础代理，用于金融交易。FinAgent的市场智能模块处理各种数据-数值、文本和图像-以准确分析金融市场。其独特的双重反射模块不仅能够快速适应市场动态，还融合了多样化的记忆检索。

    arXiv:2402.18485v1 Announce Type: cross  Abstract: Financial trading is a crucial component of the markets, informed by a multimodal information landscape encompassing news, prices, and Kline charts, and encompasses diverse tasks such as quantitative trading and high-frequency trading with various assets. While advanced AI techniques like deep learning and reinforcement learning are extensively utilized in finance, their application in financial trading tasks often faces challenges due to inadequate handling of multimodal data and limited generalizability across various tasks. To address these challenges, we present FinAgent, a multimodal foundational agent with tool augmentation for financial trading. FinAgent's market intelligence module processes a diverse range of data-numerical, textual, and visual-to accurately analyze the financial market. Its unique dual-level reflection module not only enables rapid adaptation to market dynamics but also incorporates a diversified memory retri
    
[^5]: 在金融科技应用中寻找双目标 Pareto 最优欺诈预防规则集

    On Finding Bi-objective Pareto-optimal Fraud Prevention Rule Sets for Fintech Applications. (arXiv:2311.00964v1 [cs.LG])

    [http://arxiv.org/abs/2311.00964](http://arxiv.org/abs/2311.00964)

    本文研究了在金融科技应用中寻找高质量的双目标 Pareto 最优欺诈预防规则集的问题。通过采用 Pareto 最优性概念和启发式框架 PORS，我们成功提出了一组非支配的规则子集，并通过实证评估证明了其有效性。

    

    规则在金融科技机构中被广泛用于进行欺诈预防决策，因为规则具有直观的 if-then 结构，易于理解。在实践中，大型金融科技机构通常采用两阶段欺诈预防决策规则集挖掘框架。本文关注于从初始规则集中找到高质量的规则子集，以双目标空间（如精确率和召回率）为基础。为此，我们采用 Pareto 最优性概念，旨在找到一组非支配的规则子集，构成一个 Pareto 前沿。我们提出了一个基于启发式的框架 PORS，并确定了 PORS 的核心问题是前沿解决方案选择（SSF）问题。我们对 SSF 问题进行了系统分类，并在公开和专有数据集上进行了全面的实证评估。我们还引入了一种名为 SpectralRules 的新颖变体的顺序覆盖算法，以鼓励规则的多样性。

    Rules are widely used in Fintech institutions to make fraud prevention decisions, since rules are highly interpretable thanks to their intuitive if-then structure. In practice, a two-stage framework of fraud prevention decision rule set mining is usually employed in large Fintech institutions. This paper is concerned with finding high-quality rule subsets in a bi-objective space (such as precision and recall) from an initial pool of rules. To this end, we adopt the concept of Pareto optimality and aim to find a set of non-dominated rule subsets, which constitutes a Pareto front. We propose a heuristic-based framework called PORS and we identify that the core of PORS is the problem of solution selection on the front (SSF). We provide a systematic categorization of the SSF problem and a thorough empirical evaluation of various SSF methods on both public and proprietary datasets. We also introduce a novel variant of sequential covering algorithm called SpectralRules to encourage the diver
    
[^6]: 一个具有隐藏高斯漂移和随机到达专家意见的市场中的组合优化：建模和理论结果

    Portfolio Optimization in a Market with Hidden Gaussian Drift and Randomly Arriving Expert Opinions: Modeling and Theoretical Results. (arXiv:2308.02049v1 [q-fin.PM])

    [http://arxiv.org/abs/2308.02049](http://arxiv.org/abs/2308.02049)

    本研究分析了金融市场中投资组合优化的问题，考虑了股票回报的隐藏高斯漂移以及随机到达的专家意见，应用卡尔曼滤波技术获得了隐藏漂移的估计，并通过动态规划方法解决了功用最大化问题。

    

    本文研究了在一个金融市场中，股票回报依赖于一个隐藏的高斯均值回归漂移过程的情况下，功用最大化投资者的组合选择问题。漂移的信息是通过回报和专家意见的噪声信号获得的，这些信号随机地随时间到达。到达日期被建模为一个齐次泊松过程的跳跃时间。应用卡尔曼滤波技术，我们计算出了关于观测值的条件均值和协方差的隐藏漂移的估计值。利用动态规划方法解决了功用最大化问题。我们推导出了相关的动态规划方程，并对严谨的数学证明进行了正则化论证。

    This paper investigates the optimal selection of portfolios for power utility maximizing investors in a financial market where stock returns depend on a hidden Gaussian mean reverting drift process. Information on the drift is obtained from returns and expert opinions in the form of noisy signals about the current state of the drift arriving randomly over time. The arrival dates are modeled as the jump times of a homogeneous Poisson process. Applying Kalman filter techniques we derive estimates of the hidden drift which are described by the conditional mean and covariance of the drift given the observations. The utility maximization problem is solved with dynamic programming methods. We derive the associated dynamic programming equation and study regularization arguments for a rigorous mathematical justification.
    
[^7]: 非平衡的Sakoda-Schelling模型中的社会经济代理人作为主动物质

    Socioeconomic agents as active matter in nonequilibrium Sakoda-Schelling models. (arXiv:2307.14270v1 [cond-mat.stat-mech])

    [http://arxiv.org/abs/2307.14270](http://arxiv.org/abs/2307.14270)

    该研究通过考虑Sakoda-Schelling模型中的职业模型，揭示了社会经济代理人模型中的非平衡动力学，并在平均场近似下将其映射为主动物质描述。通过研究非互惠性互动，展示了非稳态的宏观行为。这一研究为地理相关的基于代理人的模型提供了统一的框架，有助于同时考虑人口和价格动态。

    

    代理人的决策规则对于社会经济代理人模型有多么稳健？我们通过考虑一种类似Sakoda-Schelling模型的职业模型来解决这个问题，该模型在历史上被引入以揭示人类群体之间的隔离动力学。对于大类的效用函数和决策规则，我们确定了代理人动力学的非平衡性，同时恢复了类似平衡相分离的现象学。在平均场近似下，我们展示了该模型在一定程度上可以被映射为主动物质场描述（Active Model B）。最后，我们考虑了两个人群之间的非互惠性互动，并展示了它们如何导致非稳态的宏观行为。我们相信我们的方法提供了一个统一的框架，进一步研究地理相关的基于代理人的模型，尤其是在场论方法中同时考虑人口和价格动态的研究。

    How robust are socioeconomic agent-based models with respect to the details of the agents' decision rule? We tackle this question by considering an occupation model in the spirit of the Sakoda-Schelling model, historically introduced to shed light on segregation dynamics among human groups. For a large class of utility functions and decision rules, we pinpoint the nonequilibrium nature of the agent dynamics, while recovering the equilibrium-like phase separation phenomenology. Within the mean field approximation we show how the model can be mapped, to some extent, onto an active matter field description (Active Model B). Finally, we consider non-reciprocal interactions between two populations, and show how they can lead to non-steady macroscopic behavior. We believe our approach provides a unifying framework to further study geography-dependent agent-based models, notably paving the way for joint consideration of population and price dynamics within a field theoretic approach.
    

