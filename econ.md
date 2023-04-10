# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Distributed VC Firms: The Next Iteration of Venture Capital.](http://arxiv.org/abs/2304.03525) | 本文提出了一种由软件自动化推动的“分布式创业投资公司”，并批评了传统风险投资公司的激励机制。可以通过进一步研究来验证该模型并发现实施的可能路径。 |
| [^2] | [Linking Representations with Multimodal Contrastive Learning.](http://arxiv.org/abs/2304.03464) | 本文提出了一种名为CLIPPINGS的多模态框架，用于记录链接。该框架利用深度学习和对比学习的方法，通过端到端训练对称的视觉和语言编码器，在度量空间中学习相近或不同类别的表示方法，用于多个应用场景，如构建全面的补充专利注册表和识别不同社交媒体平台上的个人。 |
| [^3] | [Echo disappears: momentum term structure and cyclic information in turnover.](http://arxiv.org/abs/2304.03437) | 通过提取换手中的循环信息，研究发现动量回声可以通过近期动量反转进行解释。 |
| [^4] | [The Dynamics of Leverage and the Belief Distribution of Wealth.](http://arxiv.org/abs/2304.03436) | 本文中，通过建模经济体和资产，探讨了信仰分布和风险类型对借贷的影响。在好的状态之后，总借贷会增加，债务面额也会增加。当然，若借款违约，则悲观主义者会更被看好。即使没有噪声，财富也会收敛到与实际概率不同的信仰，风险越大，乐观主义就越强。 |
| [^5] | [Leveraging policy instruments and financial incentives to reduce embodied carbon in energy retrofits.](http://arxiv.org/abs/2304.03403) | 本研究旨在发展政策策略来减少翻新中的体积碳排放，基于对文献的综述, 确定了不列颠哥伦比亚省（BC）的政策和财政激励来减少现有建筑部门总体GHG排放。 |
| [^6] | [How to sample and when to stop sampling: The generalized Wald problem and minimax policies.](http://arxiv.org/abs/2210.15841) | 本文研究了引入信息成本的实验设计技术，特别是在顺序实验中，通过自适应地分配样本和在福利最大时停止采样来确定最佳处理方法。最小极大遗憾标准下，我们描述了最优策略，最小极大优化采样规则只是Neyman分配，独立于采样成本和先前的结果。 |
| [^7] | [Sequential Veto Bargaining with Incomplete Information.](http://arxiv.org/abs/2202.02462) | 研究了有关顺序谈判的新模型，发现在提议者和否决者具有不确定性和单峰偏好的情况下，提议者可以在没有承诺权的情况下获得高收益。 |

# 详细

[^1]: 分布式创业投资公司：风险投资的下一次革新

    Distributed VC Firms: The Next Iteration of Venture Capital. (arXiv:2304.03525v1 [econ.GN])

    [http://arxiv.org/abs/2304.03525](http://arxiv.org/abs/2304.03525)

    本文提出了一种由软件自动化推动的“分布式创业投资公司”，并批评了传统风险投资公司的激励机制。可以通过进一步研究来验证该模型并发现实施的可能路径。

    

    本文结合激励模型和实证元分析，对驱动风险投资公司优化其实践以增加GP效用的激励系统提出了尖锐的批评，这些实践与改善初创企业股权的基础资产不相关。我们提出了一种由软件自动化推动的“分布式创业投资公司”，由一组名为“Pods”的功能小组管理，他们在交易-交易的基础上执行特定任务，并获得即时和长期回报。提供了进一步研究的途径，以验证这个模型并发现实施的可能路径。

    Using a combination of incentive modeling and empirical meta-analyses, this paper provides a pointed critique at the incentive systems that drive venture capital firms to optimize their practices towards activities that increase General Partner utility yet are disjoint from improving the underlying asset of startup equity. We propose a "distributed venture firm" powered by software automations and governed by a set of functional teams called "Pods" that carry out specific tasks with immediate and long-term payouts given on a deal-by-deal basis. Avenues are provided for further research to validate this model and discover likely paths to implementation.
    
[^2]: 用多模态对比学习连接表示

    Linking Representations with Multimodal Contrastive Learning. (arXiv:2304.03464v1 [cs.CV])

    [http://arxiv.org/abs/2304.03464](http://arxiv.org/abs/2304.03464)

    本文提出了一种名为CLIPPINGS的多模态框架，用于记录链接。该框架利用深度学习和对比学习的方法，通过端到端训练对称的视觉和语言编码器，在度量空间中学习相近或不同类别的表示方法，用于多个应用场景，如构建全面的补充专利注册表和识别不同社交媒体平台上的个人。

    

    许多应用需要将包含在各种文档数据集中的实例分组成类。最广泛使用的方法不使用深度学习，也不利用文档固有的多模态性质。值得注意的是，记录链接通常被概念化为字符串匹配问题。本研究开发了 CLIPPINGS，一种用于记录链接的多模态框架。CLIPPINGS 采用端到端训练对称的视觉和语言双编码器，通过对比语言-图像预训练进行对齐，学习一个度量空间，其中给定实例的汇总图像-文本表示靠近同一类中的表示，并远离不同类中的表示。在推理时，可以通过从离线示例嵌入索引中检索它们最近的邻居或聚类它们的表示来链接实例。本研究研究了两个具有挑战性的应用：通过将专利与其对应的监管文件链接来构建全面的补充专利注册表，以及在不同的社交媒体平台上识别个人。

    Many applications require grouping instances contained in diverse document datasets into classes. Most widely used methods do not employ deep learning and do not exploit the inherently multimodal nature of documents. Notably, record linkage is typically conceptualized as a string-matching problem. This study develops CLIPPINGS, (Contrastively Linking Pooled Pre-trained Embeddings), a multimodal framework for record linkage. CLIPPINGS employs end-to-end training of symmetric vision and language bi-encoders, aligned through contrastive language-image pre-training, to learn a metric space where the pooled image-text representation for a given instance is close to representations in the same class and distant from representations in different classes. At inference time, instances can be linked by retrieving their nearest neighbor from an offline exemplar embedding index or by clustering their representations. The study examines two challenging applications: constructing comprehensive suppl
    
[^3]: Echo消失：动量期限结构和换手中的循环信息

    Echo disappears: momentum term structure and cyclic information in turnover. (arXiv:2304.03437v1 [econ.GN])

    [http://arxiv.org/abs/2304.03437](http://arxiv.org/abs/2304.03437)

    通过提取换手中的循环信息，研究发现动量回声可以通过近期动量反转进行解释。

    

    我们提取了换手中的循环信息，并发现它可以解释动量回声。近期动量反转是取消近期动量的关键因素，将其排除后，回声会回归到阻尼形状。理性和行为理论均可解释反转现象。这项研究是对美国股市动量回声的首次解释。

    We extract cyclic information in turnover and find it can explain the momentum echo. The reversal in recent month momentum is the key factor that cancels out the recent month momentum and excluding it makes the echo regress to a damped shape. Both rational and behavioral theories can explain the reversal. This study is the first explanation of the momentum echo in U.S. stock markets.
    
[^4]: 杠杆的动态和财富信仰分布

    The Dynamics of Leverage and the Belief Distribution of Wealth. (arXiv:2304.03436v1 [econ.GN])

    [http://arxiv.org/abs/2304.03436](http://arxiv.org/abs/2304.03436)

    本文中，通过建模经济体和资产，探讨了信仰分布和风险类型对借贷的影响。在好的状态之后，总借贷会增加，债务面额也会增加。当然，若借款违约，则悲观主义者会更被看好。即使没有噪声，财富也会收敛到与实际概率不同的信仰，风险越大，乐观主义就越强。

    

    经济中的总体借贷规模和条款取决于财富如何在对未来持有不同信仰的潜在债权人中分配。如果贷款得到全额偿还，则不确定性逐渐解决，使乐观主义者更加被青睐。反之，如果出现大规模违约，则悲观主义者更被看好。我们在一个拥有两种资产——风险债券和无风险现金的经济体中建模这个过程。在每个时期内，考虑到相继经历的信仰类型，借贷的规模和条款都是内生确定的。好的状态之后，总借贷和债务面额都会上升，利率会下降。在没有噪声的情况下，财富会收敛到与实际概率不同的信仰，风险厌恶越大，乐观主义越强。在存在噪声的情况下，经济体表现出高表现期和低表现期的交替。

    The scale and terms of aggregate borrowing in an economy depend on the manner in which wealth is distributed across potential creditors with heterogeneous beliefs about the future. This distribution evolves over time as uncertainty is resolved, in favour of optimists if loans are repaid in full, and in favour of pessimists if there is widespread default. We model this process in an economy with two assets - risky bonds and risk-free cash. Within periods, given the inherited distribution of wealth across belief types, the scale and terms of borrowing are endogenously determined. Following good states, aggregate borrowing and the face value of debt both rise, and the interest rate falls. In the absence of noise, wealth converges to beliefs that differ systematically from the objective probability governing state realisations, with greater risk-aversion associated with greater optimism. In the presence of noise, the economy exhibits periods of high performance, punctuated by periods of cr
    
[^5]: 利用政策工具和财政激励减少能源翻新中的碳排放

    Leveraging policy instruments and financial incentives to reduce embodied carbon in energy retrofits. (arXiv:2304.03403v1 [econ.GN])

    [http://arxiv.org/abs/2304.03403](http://arxiv.org/abs/2304.03403)

    本研究旨在发展政策策略来减少翻新中的体积碳排放，基于对文献的综述, 确定了不列颠哥伦比亚省（BC）的政策和财政激励来减少现有建筑部门总体GHG排放。

    

    建筑行业和建筑施工行业占全球总能源消耗的三分之一以上，近40％的总温室气体（GHG）排放量。建筑部门的GHG排放由体积排放和经营排放组成。为了降低建筑部门的能耗和排放，许多政府推出了政策、标准和设计方针，以改善建筑能源性能并减少与建筑经营相关的GHG排放。然而，减少现有建筑部门中的体积排放的政策倡议还不足。本研究旨在制定减少翻新中体积碳排放的政策策略。为了实现这一目标，本研究对文献进行了综述，并确定了在不列颠哥伦比亚省（BC）减少现有建筑部门总体GHG排放的政策和财政激励措施。

    The existing buildings and building construction sectors together are responsible for over one-third of the total global energy consumption and nearly 40% of total greenhouse gas (GHG) emissions. GHG emissions from the building sector are made up of embodied emissions and operational emissions. Recognizing the importance of reducing energy use and emissions associated with the building sector, governments have introduced policies, standards, and design guidelines to improve building energy performance and reduce GHG emissions associated with operating buildings. However, policy initiatives that reduce embodied emissions of the existing building sector are lacking. This research aims to develop policy strategies to reduce embodied carbon emissions in retrofits. In order to achieve this goal, this research conducted a literature review and identification of policies and financial incentives in British Columbia (BC) for reducing overall GHG emissions from the existing building sector. The
    
[^6]: 如何进行样本采集以及何时停止采样：广义Wald问题和最小极大决策

    How to sample and when to stop sampling: The generalized Wald problem and minimax policies. (arXiv:2210.15841v4 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2210.15841](http://arxiv.org/abs/2210.15841)

    本文研究了引入信息成本的实验设计技术，特别是在顺序实验中，通过自适应地分配样本和在福利最大时停止采样来确定最佳处理方法。最小极大遗憾标准下，我们描述了最优策略，最小极大优化采样规则只是Neyman分配，独立于采样成本和先前的结果。

    

    获取信息是昂贵的。实验者需要仔细选择每种处理的样本数量以及何时停止采样。本文的目的是开发将信息成本纳入实验设计的技术。特别是，我们研究了一种顺序实验，其中采样具有成本，决策者通过（1）自适应地分配单位到两种可能的处理方法，以及（2）当实施所选择的处理的预期福利（包括采样成本）达到最大时停止实验，来确定最佳处理方法以进行全面实施。在扩散极限下工作，我们在最小极大遗憾标准下描述了最优策略。在小成本渐近下，相同的策略在参数化和非参数化结果分布下也是最优的。最小极大优化采样规则只是Neyman分配；它独立于采样成本，也不适应于先前的结果。

    Acquiring information is expensive. Experimenters need to carefully choose how many units of each treatment to sample and when to stop sampling. The aim of this paper is to develop techniques for incorporating the cost of information into experimental design. In particular, we study sequential experiments where sampling is costly and a decision-maker aims to determine the best treatment for full scale implementation by (1) adaptively allocating units to two possible treatments, and (2) stopping the experiment when the expected welfare (inclusive of sampling costs) from implementing the chosen treatment is maximized. Working under the diffusion limit, we describe the optimal policies under the minimax regret criterion. Under small cost asymptotics, the same policies are also optimal under parametric and non-parametric distributions of outcomes. The minimax optimal sampling rule is just the Neyman allocation; it is independent of sampling costs and does not adapt to previous outcomes. Th
    
[^7]: 不完全信息下的顺序否决谈判

    Sequential Veto Bargaining with Incomplete Information. (arXiv:2202.02462v3 [econ.TH] UPDATED)

    [http://arxiv.org/abs/2202.02462](http://arxiv.org/abs/2202.02462)

    研究了有关顺序谈判的新模型，发现在提议者和否决者具有不确定性和单峰偏好的情况下，提议者可以在没有承诺权的情况下获得高收益。

    

    我们研究了提议者和否决者之间的顺序谈判。两者都具有单峰偏好，但提议者对否决者的理想点不确定。当玩家有耐心时，可以出现带有Coasian动态的均衡：否决者的私人信息可以在很大程度上抵消提议者的谈判能力。然而，我们的主要结果是，在某些条件下，也存在使提议者获得承诺权高收益的均衡。驱动力在于否决者的单峰偏好为提议者提供了一种“跨越”的选择，即提议者可以通过先从低剩余类型处获得协议以可信地从高类型处提取剩余来实现。

    We study sequential bargaining between a proposer and a veto player. Both have single-peaked preferences, but the proposer is uncertain about the veto player's ideal point. The proposer cannot commit to future proposals. When players are patient, there can be equilibria with Coasian dynamics: the veto player's private information can largely nullify proposer's bargaining power. Our main result, however, is that under some conditions there are also equilibria in which the proposer obtains the high payoff that he would with commitment power. The driving force is that the veto player's single-peaked preferences give the proposer an option to "leapfrog", i.e., to secure agreement from only low-surplus types early on to credibly extract surplus from high types later. Methodologically, we exploit the connection between sequential bargaining and static mechanism design.
    

