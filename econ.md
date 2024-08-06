# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Costly Persuasion by a Partially Informed Sender.](http://arxiv.org/abs/2401.14087) | 本研究探讨了具有高昂成本的贝叶斯说服模型，研究对象是一位私人且部分信息知情的发送者在进行公共实验。研究发现实验中好消息和坏消息的成本差异对均衡结果具有重要影响，坏消息成本高时，存在唯一的分离均衡，接收者受益于发送者的私有信息；而好消息成本高时，均衡情况可能出现汇集和部分汇集均衡，接收者可能会因为发送者私有信息而受到损害。 |
| [^2] | [Deterministic Impartial Selection with Weights.](http://arxiv.org/abs/2310.14991) | 该研究提供了带权力的确定性公正机制，通过引入权重，改进了以前无权重设置下的逼近比率，并展示了该机制可以适用于公正分配问题。 |
| [^3] | [The Inflation Attention Threshold and Inflation Surges.](http://arxiv.org/abs/2308.09480) | 本论文研究了通胀关注度与通胀激增之间的关系，发现在通胀率低稳定时，人们关注度较低，但一旦通胀率超过4%，关注度会明显增加，高关注区域的关注度是低关注区域的两倍。这种关注阈值的存在导致了状态依赖效应，即在宽松货币政策时，成本推动冲击对通胀的影响更大。这些结论有助于理解最近美国的通胀激增现象。 |
| [^4] | [School Choice with Multiple Priorities.](http://arxiv.org/abs/2308.04780) | 本研究提出了一个具有多个优先级的学校选择模型，引入了一种名为M-fairness的公平性概念，并介绍了一种利用效率调整延迟接受算法的机制，该机制是学生最优M稳定的，改进群体最优M稳定的，并且对改进是有响应的。 |
| [^5] | [Adaptive Principal Component Regression with Applications to Panel Data.](http://arxiv.org/abs/2307.01357) | 本文提出了自适应主成分回归方法，并在面板数据中的应用中获得了均匀有限样本保证。该方法可以用于面板数据中的实验设计，特别是当干预方案是自适应分配的情况。 |
| [^6] | [Inference in Cluster Randomized Trials with Matched Pairs.](http://arxiv.org/abs/2211.14903) | 本文研究了在匹配对簇随机试验中进行统计推断的问题，提出了加权均值差估计量和方差估计量，建立了基于这些估计量的渐近精确性，并探讨了常用的两种测试程序的特性。 |

# 详细

[^1]: 高昂的说服成本与部分信息的发送者

    Costly Persuasion by a Partially Informed Sender. (arXiv:2401.14087v1 [econ.TH])

    [http://arxiv.org/abs/2401.14087](http://arxiv.org/abs/2401.14087)

    本研究探讨了具有高昂成本的贝叶斯说服模型，研究对象是一位私人且部分信息知情的发送者在进行公共实验。研究发现实验中好消息和坏消息的成本差异对均衡结果具有重要影响，坏消息成本高时，存在唯一的分离均衡，接收者受益于发送者的私有信息；而好消息成本高时，均衡情况可能出现汇集和部分汇集均衡，接收者可能会因为发送者私有信息而受到损害。

    

    本文研究了由一个拥有私有且部分信息的发送者进行的昂贵的贝叶斯说服模型，该发送者进行了一个公共实验。实验的成本是发送者信念的加权对数似然比函数的期望减少。这个模型通过一个沃尔德的顺序抽样问题得到微基础，其中好消息和坏消息的成本不同。我们关注满足D1准则的均衡。均衡结果取决于实验中获得好消息和坏消息的相对成本。如果坏消息的成本更高，则存在唯一的分离均衡，并且接收者明确受益于发送者的私有信息。如果好消息的成本更高，则单点交叉特性不成立。可能存在汇集和部分汇集均衡，在某些均衡中，接收者会明确受到发送者私有信息的伤害。

    I study a model of costly Bayesian persuasion by a privately and partially informed sender who conducts a public experiment. The cost of running an experiment is the expected reduction of a weighted log-likelihood ratio function of the sender's belief. This is microfounded by a Wald's sequential sampling problem where good news and bad news cost differently. I focus on equilibria that satisfy the D1 criterion. The equilibrium outcome depends on the relative costs of drawing good and bad news in the experiment. If bad news is more costly, there exists a unique separating equilibrium, and the receiver unambiguously benefits from the sender's private information. If good news is more costly, the single-crossing property fails. There may exist pooling and partial pooling equilibria, and in some equilibria, the receiver strictly suffers from sender private information.
    
[^2]: 带权力的确定性公正选择

    Deterministic Impartial Selection with Weights. (arXiv:2310.14991v1 [cs.GT])

    [http://arxiv.org/abs/2310.14991](http://arxiv.org/abs/2310.14991)

    该研究提供了带权力的确定性公正机制，通过引入权重，改进了以前无权重设置下的逼近比率，并展示了该机制可以适用于公正分配问题。

    

    在公正选择问题中，基于代理人投票选取一个固定大小为$k$的代理人子集。如果没有代理人可以通过改变自己的投票来影响自己被选择的机会，则选择机制是公正的。如果对于每个实例，所选子集所接收的票数与获得最高票数的大小为$k$的子集所接收的票数的比率至少为$\alpha$的一部分，则该机制是$\alpha$-最优的。我们在一个更一般的设置中研究了带权力的确定性公正机制，并提供了首个逼近保证，大约为$1/\lceil 2n/k\rceil$。当要选择的代理人数量相对于总代理人数量足够大时，这比以前已知的无权重设置的逼近比率$1/k$有所改进。我们进一步证明了我们的机制可以适应公正分配问题，即在其中有多个集合需要选出代理人。

    In the impartial selection problem, a subset of agents up to a fixed size $k$ among a group of $n$ is to be chosen based on votes cast by the agents themselves. A selection mechanism is impartial if no agent can influence its own chance of being selected by changing its vote. It is $\alpha$-optimal if, for every instance, the ratio between the votes received by the selected subset is at least a fraction of $\alpha$ of the votes received by the subset of size $k$ with the highest number of votes. We study deterministic impartial mechanisms in a more general setting with arbitrarily weighted votes and provide the first approximation guarantee, roughly $1/\lceil 2n/k\rceil$. When the number of agents to select is large enough compared to the total number of agents, this yields an improvement on the previously best known approximation ratio of $1/k$ for the unweighted setting. We further show that our mechanism can be adapted to the impartial assignment problem, in which multiple sets of u
    
[^3]: 通胀关注阈值与通胀激增

    The Inflation Attention Threshold and Inflation Surges. (arXiv:2308.09480v1 [econ.GN])

    [http://arxiv.org/abs/2308.09480](http://arxiv.org/abs/2308.09480)

    本论文研究了通胀关注度与通胀激增之间的关系，发现在通胀率低稳定时，人们关注度较低，但一旦通胀率超过4%，关注度会明显增加，高关注区域的关注度是低关注区域的两倍。这种关注阈值的存在导致了状态依赖效应，即在宽松货币政策时，成本推动冲击对通胀的影响更大。这些结论有助于理解最近美国的通胀激增现象。

    

    在最近通胀激增爆发时，公众对通胀的关注度很低，但一旦通胀开始上升，关注度迅速增加。本文构建了一个一般均衡货币模型，该模型在通胀低而稳定时，最优化的策略是对通胀关注较少，但一旦通胀超过某个阈值，就会增加关注度。利用调查问卷中的通胀预期，我估计关注阈值在4%通胀率，高关注区域的关注度是低关注区域的两倍。当校准到这些发现时，该模型产生与美国最近通胀激增一致的通胀和通胀预期动态。关注阈值导致状态依赖性：成本推动冲击在货币宽松政策时更加通胀。这些状态依赖性效应在恒定关注或理性预期模型中是不存在的。

    At the outbreak of the recent inflation surge, the public's attention to inflation was low but increased rapidly once inflation started to rise. In this paper, I develop a general equilibrium monetary model where it is optimal for agents to pay little attention to inflation when inflation is low and stable, but in which they increase their attention once inflation exceeds a certain threshold. Using survey inflation expectations, I estimate the attention threshold to be at an inflation rate of 4%, with attention in the high-attention regime being twice as high as in the low-attention regime. When calibrated to match these findings, the model generates inflation and inflation expectation dynamics consistent with the recent inflation surge in the US. The attention threshold induces a state dependency: cost-push shocks become more inflationary in times of loose monetary policy. These state-dependent effects are absent in the model with constant attention or under rational expectations. Fol
    
[^4]: 具有多个优先级的学校选择模型

    School Choice with Multiple Priorities. (arXiv:2308.04780v1 [econ.TH])

    [http://arxiv.org/abs/2308.04780](http://arxiv.org/abs/2308.04780)

    本研究提出了一个具有多个优先级的学校选择模型，引入了一种名为M-fairness的公平性概念，并介绍了一种利用效率调整延迟接受算法的机制，该机制是学生最优M稳定的，改进群体最优M稳定的，并且对改进是有响应的。

    

    本研究考虑了一种模型，在这种模型中，学校可能对学生有多个优先级顺序，这些顺序可能相互矛盾。例如，在学校选择系统中，由于兄弟姐妹优先级和步行区域优先级并存，基于它们的优先级顺序可能存在冲突。在这种情况下，可能找不到满足所有优先级顺序的匹配。我们引入了一种名为M-fairness的新颖公平性概念来研究这样的市场。此外，我们重点研究了一种更具体的情况，即所有学校都有两个优先级顺序，并且对于某个学生群体，每所学校的一个优先级顺序是另一优先级顺序的改进。一个说明性例子是具有基于优先级的积极行动政策的学校选择匹配市场。我们引入了一种利用效率调整延迟接受算法的机制，并证明该机制是学生最优M稳定的，改进群体最优M稳定的，并且对改进是有响应的。

    This study considers a model where schools may have multiple priority orders on students, which may be inconsistent with each other. For example, in school choice systems, since the sibling priority and the walk zone priority coexist, the priority orders based on them would be conflicting. In that case, there may be no matching that respect to all priority orders. We introduce a novel fairness notion called M-fairness to examine such markets. Further, we focus on a more specific situation where all schools have two priority orders, and for a certain group of students, a priority order of each school is an improvement of the other priority order of the school. An illustrative example is the school choice matching market with a priority-based affirmative action policy. We introduce a mechanism that utilizes the efficiency adjusted deferred acceptance algorithm and show that the mechanism is student optimally M-stable, improved-group optimally M-stable and responsive to improvements.
    
[^5]: 自适应主成分回归在面板数据中的应用

    Adaptive Principal Component Regression with Applications to Panel Data. (arXiv:2307.01357v1 [cs.LG])

    [http://arxiv.org/abs/2307.01357](http://arxiv.org/abs/2307.01357)

    本文提出了自适应主成分回归方法，并在面板数据中的应用中获得了均匀有限样本保证。该方法可以用于面板数据中的实验设计，特别是当干预方案是自适应分配的情况。

    

    主成分回归(PCR)是一种流行的固定设计误差变量回归技术，它是线性回归的推广，观测的协变量受到随机噪声的污染。我们在数据收集时提供了在线（正则化）PCR的第一次均匀有限样本保证。由于分析固定设计中PCR的证明技术无法很容易地扩展到在线设置，我们的结果依赖于将现代鞅浓度的工具适应到误差变量设置中。作为我们界限的应用，我们在面板数据设置中提供了实验设计框架，当干预被自适应地分配时。我们的框架可以被认为是合成控制和合成干预框架的泛化，其中数据是通过自适应干预分配策略收集的。

    Principal component regression (PCR) is a popular technique for fixed-design error-in-variables regression, a generalization of the linear regression setting in which the observed covariates are corrupted with random noise. We provide the first time-uniform finite sample guarantees for online (regularized) PCR whenever data is collected adaptively. Since the proof techniques for analyzing PCR in the fixed design setting do not readily extend to the online setting, our results rely on adapting tools from modern martingale concentration to the error-in-variables setting. As an application of our bounds, we provide a framework for experiment design in panel data settings when interventions are assigned adaptively. Our framework may be thought of as a generalization of the synthetic control and synthetic interventions frameworks, where data is collected via an adaptive intervention assignment policy.
    
[^6]: 匹配对簇随机试验中的推断

    Inference in Cluster Randomized Trials with Matched Pairs. (arXiv:2211.14903v2 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2211.14903](http://arxiv.org/abs/2211.14903)

    本文研究了在匹配对簇随机试验中进行统计推断的问题，提出了加权均值差估计量和方差估计量，建立了基于这些估计量的渐近精确性，并探讨了常用的两种测试程序的特性。

    

    本文研究了在匹配对簇随机试验中进行推断的问题。匹配对设计意味着根据基线簇级协变量对一组簇进行匹配，然后在每对簇中随机选择一个簇进行处理。我们研究了加权均值差估计量的大样本行为，并根据匹配过程是否匹配簇大小，导出了两组不同的结果。然后，我们提出了一个方差估计量，无论匹配过程是匹配簇大小还是不匹配簇大小，都是一致的。结合这些结果，建立了基于这些估计量的检验的渐近精确性。接下来，我们考虑了基于线性回归构造的$t$测试的两种常见测试程序的特性，并声称这两种程序通常是保守的。

    This paper considers the problem of inference in cluster randomized trials where treatment status is determined according to a "matched pairs'' design. Here, by a cluster randomized experiment, we mean one in which treatment is assigned at the level of the cluster; by a "matched pairs'' design we mean that a sample of clusters is paired according to baseline, cluster-level covariates and, within each pair, one cluster is selected at random for treatment. We study the large sample behavior of a weighted difference-in-means estimator and derive two distinct sets of results depending on if the matching procedure does or does not match on cluster size. We then propose a variance estimator which is consistent in either case. Combining these results establishes the asymptotic exactness of tests based on these estimators. Next, we consider the properties of two common testing procedures based on $t$-tests constructed from linear regressions, and argue that both are generally conservative in o
    

