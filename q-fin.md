# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Swing Contract Pricing: A Parametric Approach with Adjoint Automatic Differentiation and Neural Networks.](http://arxiv.org/abs/2306.03822) | 本文提出了一种参数化方法来定价具有限制的摇摆合约，并尝试使用神经网络取代一些参数。数值实验表明，相比于现有方法，该方法能在短时间内提供更好的价格。 |
| [^2] | [ChatGPT Informed Graph Neural Network for Stock Movement Prediction.](http://arxiv.org/abs/2306.03763) | 该研究介绍了一种新的框架，利用ChatGPT技术增强图神经网络，能够从财经新闻中提取出不断变化的网络结构，并用于股票价格预测，获得了超过基于深度学习的最新基准的表现，提示了ChatGPT在文本推断和金融预测方面的潜力。 |
| [^3] | [Global universal approximation of functional input maps on weighted spaces.](http://arxiv.org/abs/2306.03303) | 本文提出了功能性输入神经网络，可以在带权重空间上完成全局函数逼近。这一方法适用于连续函数的推广，还可用于路径空间函数的逼近，同时也可以逼近线性函数签名。 |
| [^4] | ['Ergodicity Economics' is Pseudoscience.](http://arxiv.org/abs/2306.03275) | 本文指出，"遍历性经济学" 是伪科学，因为它缺乏可验证的实证影响。 |
| [^5] | [Gauge symmetries and the Higgs mechanism in Quantum Finance.](http://arxiv.org/abs/2306.03237) | 本文利用规范对称性引入了随机波动率，从而从 Black-Scholes 方程中推导出了 Merton-Garman 方程，最终通过 Higgs 机制将随机波动率转化为有质量的。 |
| [^6] | [Time-Consistent Asset Allocation for Risk Measures in a L\'evy Market.](http://arxiv.org/abs/2305.09471) | 本文研究了使用风险度量进行资产配置的问题，通过使用广义的风险度量，最大化时间一致的平均风险回报函数。作者模拟了市场，并在符合条件的情况下证明了该问题的最优解是确定性且唯一的。 |
| [^7] | [The self-exciting nature of the bid-ask spread dynamics.](http://arxiv.org/abs/2303.02038) | 本研究提出了一种基于状态的"Spread Hawkes模型"，可以预测Cac40 Euronext市场期货中的Spread值，并捕捉了一些统计属性。 |
| [^8] | [Measuring Cognitive Abilities in the Wild: Validating a Population-Scale Game-Based Cognitive Assessment.](http://arxiv.org/abs/2009.05274) | 该论文提出了一种基于游戏的认知评估方法Skill Lab，利用一个流行的公民科学平台进行全面验证，在真实环境中测量了广泛的认知能力，可以同时预测8种认知能力。 |
| [^9] | [Denise: Deep Robust Principal Component Analysis for Positive Semidefinite Matrices.](http://arxiv.org/abs/2004.13612) | Denise是一种基于深度学习的算法，用于对协方差矩阵进行低秩加稀疏分解，达到了与最先进技术相当的性能而且近乎接近20倍的加速。 |

# 详细

[^1]: 摇摆合约定价: 一种带有自动微分和神经网络的参数化方法

    Swing Contract Pricing: A Parametric Approach with Adjoint Automatic Differentiation and Neural Networks. (arXiv:2306.03822v1 [q-fin.MF])

    [http://arxiv.org/abs/2306.03822](http://arxiv.org/abs/2306.03822)

    本文提出了一种参数化方法来定价具有限制的摇摆合约，并尝试使用神经网络取代一些参数。数值实验表明，相比于现有方法，该方法能在短时间内提供更好的价格。

    

    我们提出了两种参数化方法来定价带有强制性限制的摇摆合约。我们的目标是创建近似最优控制的函数，其代表合约期内购买能源的数量。第一种方法涉及明确地定义一个参数化函数来建模最优控制并使用基于随机梯度下降的算法来确定参数。第二种方法基于第一种方法，将参数替换为神经网络。我们的数值实验表明，通过使用Langevin算法，这两种参数化方法都在短时间内提供了比现有方法(如Longstaff和Schwartz提出的方法)更好的价格。

    We propose two parametric approaches to price swing contracts with firm constraints. Our objective is to create approximations for the optimal control, which represents the amounts of energy purchased throughout the contract. The first approach involves explicitly defining a parametric function to model the optimal control, and the parameters using stochastic gradient descent-based algorithms. The second approach builds on the first one, replacing the parameters with neural networks. Our numerical experiments demonstrate that by using Langevin-based algorithms, both parameterizations provide, in a short computation time, better prices compared to state-of-the-art methods (like the one given by Longstaff and Schwartz).
    
[^2]: ChatGPT信息的图神经网络用于股票价格预测

    ChatGPT Informed Graph Neural Network for Stock Movement Prediction. (arXiv:2306.03763v1 [q-fin.ST])

    [http://arxiv.org/abs/2306.03763](http://arxiv.org/abs/2306.03763)

    该研究介绍了一种新的框架，利用ChatGPT技术增强图神经网络，能够从财经新闻中提取出不断变化的网络结构，并用于股票价格预测，获得了超过基于深度学习的最新基准的表现，提示了ChatGPT在文本推断和金融预测方面的潜力。

    

    ChatGPT已在各种自然语言处理（NLP）任务中展示了出色的能力。然而，它从时间文本数据（尤其是财经新闻）推断动态网络结构的潜力仍是一个未开发的领域。在这项研究中，我们介绍了一个新的框架，利用ChatGPT的图推断能力来增强图神经网络（GNN）。我们的框架巧妙地从文本数据中提取出不断变化的网络结构，并将这些网络结构融合到图神经网络中，进行后续的预测任务。股票价格预测的实验结果表明，我们的模型始终优于基于深度学习的最新基准。此外，基于我们模型的产出构建的组合展示出更高的年化累计回报、更低的波动性和最大回撤。这种卓越表现突显了ChatGPT用于基于文本的网络推断和金融预测应用的潜力。

    ChatGPT has demonstrated remarkable capabilities across various natural language processing (NLP) tasks. However, its potential for inferring dynamic network structures from temporal textual data, specifically financial news, remains an unexplored frontier. In this research, we introduce a novel framework that leverages ChatGPT's graph inference capabilities to enhance Graph Neural Networks (GNN). Our framework adeptly extracts evolving network structures from textual data, and incorporates these networks into graph neural networks for subsequent predictive tasks. The experimental results from stock movement forecasting indicate our model has consistently outperformed the state-of-the-art Deep Learning-based benchmarks. Furthermore, the portfolios constructed based on our model's outputs demonstrate higher annualized cumulative returns, alongside reduced volatility and maximum drawdown. This superior performance highlights the potential of ChatGPT for text-based network inferences and 
    
[^3]: 带权重空间上功能性输入映射的全局普适逼近

    Global universal approximation of functional input maps on weighted spaces. (arXiv:2306.03303v1 [stat.ML])

    [http://arxiv.org/abs/2306.03303](http://arxiv.org/abs/2306.03303)

    本文提出了功能性输入神经网络，可以在带权重空间上完成全局函数逼近。这一方法适用于连续函数的推广，还可用于路径空间函数的逼近，同时也可以逼近线性函数签名。

    

    我们引入了所谓的功能性输入神经网络，定义在可能是无限维带权重空间上，其值也在可能是无限维的输出空间中。为此，我们使用一个加性族作为隐藏层映射，以及一个非线性激活函数应用于每个隐藏层。依靠带权重空间上的Stone-Weierstrass定理，我们可以证明连续函数的推广的全局普适逼近结果，超越了常规紧集逼近。这特别适用于通过功能性输入神经网络逼近（非先见之明的）路径空间函数。作为带权Stone-Weierstrass定理的进一步应用，我们证明了线性函数签名的全局普适逼近结果。我们还在这个设置中引入了高斯过程回归的观点，并展示了签名内核的再生核希尔伯特空间是某些高斯过程的Cameron-Martin空间。

    We introduce so-called functional input neural networks defined on a possibly infinite dimensional weighted space with values also in a possibly infinite dimensional output space. To this end, we use an additive family as hidden layer maps and a non-linear activation function applied to each hidden layer. Relying on Stone-Weierstrass theorems on weighted spaces, we can prove a global universal approximation result for generalizations of continuous functions going beyond the usual approximation on compact sets. This then applies in particular to approximation of (non-anticipative) path space functionals via functional input neural networks. As a further application of the weighted Stone-Weierstrass theorem we prove a global universal approximation result for linear functions of the signature. We also introduce the viewpoint of Gaussian process regression in this setting and show that the reproducing kernel Hilbert space of the signature kernels are Cameron-Martin spaces of certain Gauss
    
[^4]: "「遍历性经济学」是伪科学"

    'Ergodicity Economics' is Pseudoscience. (arXiv:2306.03275v1 [q-fin.GN])

    [http://arxiv.org/abs/2306.03275](http://arxiv.org/abs/2306.03275)

    本文指出，"遍历性经济学" 是伪科学，因为它缺乏可验证的实证影响。

    

    Ole Peters 和他的合作者在一系列论文中声称，"主流经济理论的概念基础"是"有缺陷的"，他们所称的 "遍历性经济学" 的方法则为"希望未来经济科学更加简明、概念更清晰、更少主观性"提供了理由(Peters, 2019)。本文认为 "遍历性经济学" 是伪科学，因为它没有产生可证伪的影响，应以怀疑的态度看待。

    In a series of papers, Ole Peters and his collaborators claim that the 'conceptual basis of mainstream economic theory' is 'flawed' and that the approach they call 'ergodicity economics' gives 'reason to hope for a future economic science that is more parsimonious, conceptually clearer and less subjective' (Peters, 2019). This paper argues that 'ergodicity economics' is pseudoscience because it has not produced falsifiable implications and should be taken with skepticism.
    
[^5]: 量子金融中的规范对称性和 Higgs 机制

    Gauge symmetries and the Higgs mechanism in Quantum Finance. (arXiv:2306.03237v1 [q-fin.GN])

    [http://arxiv.org/abs/2306.03237](http://arxiv.org/abs/2306.03237)

    本文利用规范对称性引入了随机波动率，从而从 Black-Scholes 方程中推导出了 Merton-Garman 方程，最终通过 Higgs 机制将随机波动率转化为有质量的。

    

    本文利用哈密顿形式，证明了在对股票价格的局部（规范）变换下，强加不变性（对称性）后，Merton-Garman 方程会自然地从 Black-Scholes 方程中出现。这是因为强加规范对称性会导致额外场的出现，相应于随机波动率。然后，规范对称性会对 Merton-Garman 哈密顿量的自由参数施加约束。最后，我们分析了随机波动率如何通过 Higgs 机制动态地转化为有质量的。

    By using the Hamiltonian formulation, we demonstrate that the Merton-Garman equation emerges naturally from the Black-Scholes equation after imposing invariance (symmetry) under local (gauge) transformations over changes in the stock price. This is the case because imposing gauge symmetry implies the appearance of an additional field, which corresponds to the stochastic volatility. The gauge symmetry then imposes some constraints over the free-parameters of the Merton-Garman Hamiltonian. Finally, we analyze how the stochastic volatility gets massive dynamically via Higgs mechanism.
    
[^6]: 在 L\'evy 市场中基于风险度量的时一致资产配置

    Time-Consistent Asset Allocation for Risk Measures in a L\'evy Market. (arXiv:2305.09471v1 [q-fin.MF])

    [http://arxiv.org/abs/2305.09471](http://arxiv.org/abs/2305.09471)

    本文研究了使用风险度量进行资产配置的问题，通过使用广义的风险度量，最大化时间一致的平均风险回报函数。作者模拟了市场，并在符合条件的情况下证明了该问题的最优解是确定性且唯一的。

    

    本文针对增益而非终止财富，考虑了一种资产配置问题，旨在通过使用一种广义的风险度量（满足 i）具有律变性，ii）具有现金或平移不变性，及 iii）可能嵌入到一种通用函数中）最大限度地一致化时间来扩大风险收益函数。我们使用 $\alpha$ 稳定的 L\'evy 过程来模拟市场，并为经典的 Black-Scholes 模型提供了补充结果。该问题的最优解是一个 Nash 子游戏均衡，其由扩展的 Hamilton-Jacobi-Bellman 方程的解决方案给出。此外，在适当的假设下，我们证明最优解是确定性且唯一的。

    Focusing on gains instead of terminal wealth, we consider an asset allocation problem to maximize time-consistently a mean-risk reward function with a general risk measure which is i) law-invariant, ii) cash- or shift-invariant, and iii) positively homogeneous, and possibly plugged into a general function. We model the market via a generalized version of the multi-dimensional Black-Scholes model using $\alpha$-stable L\'evy processes and give supplementary results for the classical Black-Scholes model. The optimal solution to this problem is a Nash subgame equilibrium given by the solution of an extended Hamilton-Jacobi-Bellman equation. Moreover, we show that the optimal solution is deterministic and unique under appropriate assumptions.
    
[^7]: 基于状态的Spread Hawkes模型对Cac40 Euronext市场期货的分析

    The self-exciting nature of the bid-ask spread dynamics. (arXiv:2303.02038v2 [q-fin.TR] UPDATED)

    [http://arxiv.org/abs/2303.02038](http://arxiv.org/abs/2303.02038)

    本研究提出了一种基于状态的"Spread Hawkes模型"，可以预测Cac40 Euronext市场期货中的Spread值，并捕捉了一些统计属性。

    

    在限价委托簿中，最佳卖价和最佳买价的差价是金融证券分析中至关重要的因素。本研究提出了一种“基于状态的Spread Hawkes模型”（SDSH），它考虑了各种Spread跳跃大小，并将当前Spread状态对其强度函数的影响纳入模型中。我们将此模型应用于Cac40 Euronext市场的高频数据，捕捉了一些统计属性，如Spread分布、事件间隔分布和Spread自相关函数。我们说明了SDSH模型在短期内预测Spread值的能力。

    The bid-ask spread, which is defined by the difference between the best selling price and the best buying price in a Limit Order Book at a given time, is a crucial factor in the analysis of financial securities. In this study, we propose a "State-dependent Spread Hawkes model" (SDSH) that accounts for various spread jump sizes and incorporates the impact of the current spread state on its intensity functions. We apply this model to the high-frequency data from the Cac40 Euronext market and capture several statistical properties, such as the spread distributions, inter-event time distributions, and spread autocorrelation functions. We illustrate the ability of the SDSH model to forecast spread values at short-term horizons.
    
[^8]: 在真实环境中测量认知能力：验证一种面向人群的基于游戏的认知评估方法

    Measuring Cognitive Abilities in the Wild: Validating a Population-Scale Game-Based Cognitive Assessment. (arXiv:2009.05274v5 [physics.soc-ph] UPDATED)

    [http://arxiv.org/abs/2009.05274](http://arxiv.org/abs/2009.05274)

    该论文提出了一种基于游戏的认知评估方法Skill Lab，利用一个流行的公民科学平台进行全面验证，在真实环境中测量了广泛的认知能力，可以同时预测8种认知能力。

    

    个体认知表型的快速测量具有革命性的潜力，可在个性化学习、就业实践和精准精神病学等广泛领域得到应用。为了超越传统实验室实验所带来的限制，人们正在努力增加生态效度和参与者多样性，以捕捉普通人群中认知能力和行为的个体差异的全部范围。基于此，我们开发了Skill Lab，一种新型的基于游戏的工具，它在提供引人入胜的故事情节的同时评估广泛的认知能力。Skill Lab由六个小游戏和14个已知的认知能力任务组成。利用一个流行的公民科学平台（N = 10725），我们在真实环境中进行了一项全面的基于游戏的认知评估。基于游戏和验证任务的数据，我们构建了可靠的模型，同时预测八种认知能力。

    Rapid individual cognitive phenotyping holds the potential to revolutionize domains as wide-ranging as personalized learning, employment practices, and precision psychiatry. Going beyond limitations imposed by traditional lab-based experiments, new efforts have been underway towards greater ecological validity and participant diversity to capture the full range of individual differences in cognitive abilities and behaviors across the general population. Building on this, we developed Skill Lab, a novel game-based tool that simultaneously assesses a broad suite of cognitive abilities while providing an engaging narrative. Skill Lab consists of six mini-games as well as 14 established cognitive ability tasks. Using a popular citizen science platform (N = 10725), we conducted a comprehensive validation in the wild of a game-based cognitive assessment suite. Based on the game and validation task data, we constructed reliable models to simultaneously predict eight cognitive abilities based 
    
[^9]: Denise: 面向半正定矩阵的深度健壮主成分分析

    Denise: Deep Robust Principal Component Analysis for Positive Semidefinite Matrices. (arXiv:2004.13612v4 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2004.13612](http://arxiv.org/abs/2004.13612)

    Denise是一种基于深度学习的算法，用于对协方差矩阵进行低秩加稀疏分解，达到了与最先进技术相当的性能而且近乎接近20倍的加速。

    

    协方差矩阵的健壮主成分分析在隔离关键解释特征方面发挥着至关重要的作用。目前可用的执行低秩加稀疏分解的方法是针对特定矩阵的，也就是说，这些算法必须针对每个新的矩阵重新运行。由于这些算法的计算成本很高，因此最好学习和存储一个函数，在评估时几乎立即执行此分解。因此，我们引入了 Denise，一种基于深度学习的协方差矩阵的健壮主成分分析算法，或更一般地说，对称半正定矩阵，它学习到了这样一个函数。我们提供了 Denise 的理论保证。这些包括一个新的通用逼近定理，适用于我们的几何深度学习问题，并趋于学习问题的最优解。我们的实验表明，Denise 在分解质量方面与最先进的性能相匹配，同时近乎接近20倍的加速。

    The robust PCA of covariance matrices plays an essential role when isolating key explanatory features. The currently available methods for performing such a low-rank plus sparse decomposition are matrix specific, meaning, those algorithms must re-run for every new matrix. Since these algorithms are computationally expensive, it is preferable to learn and store a function that nearly instantaneously performs this decomposition when evaluated. Therefore, we introduce Denise, a deep learning-based algorithm for robust PCA of covariance matrices, or more generally, of symmetric positive semidefinite matrices, which learns precisely such a function. Theoretical guarantees for Denise are provided. These include a novel universal approximation theorem adapted to our geometric deep learning problem and convergence to an optimal solution to the learning problem. Our experiments show that Denise matches state-of-the-art performance in terms of decomposition quality, while being approximately $20
    

