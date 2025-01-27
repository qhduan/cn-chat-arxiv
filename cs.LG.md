# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Faster Convergence of Stochastic Accelerated Gradient Descent under Interpolation](https://arxiv.org/abs/2404.02378) | 该论文证明了在插值条件下对随机加速的一般化版本的新收敛速度，在强增长条件下的加速SGD中取得了显著改进。 |
| [^2] | [NLP Verification: Towards a General Methodology for Certifying Robustness](https://arxiv.org/abs/2403.10144) | 本文尝试总结和评估由该领域迄今进展而形成的NLP验证流程的一般组成部分，贡献在于提出了将句子嵌入连续空间得到的可验证子空间的一般描述。 |
| [^3] | [Graph Regularized Encoder Training for Extreme Classification](https://arxiv.org/abs/2402.18434) | 本文提出了一种图正则化编码器训练方法用于极端分类，在实践中发现使用图数据来规范编码器训练比实施 GCN 效果更好。 |
| [^4] | [What's in a Name? Auditing Large Language Models for Race and Gender Bias](https://arxiv.org/abs/2402.14875) | 调查发现，大型语言模型存在种族和性别偏见，尤其对与黑人女性相关的名字表现最不利。审计在模型部署和实施时的重要性得到强调。 |
| [^5] | [Hierarchical Bayes Approach to Personalized Federated Unsupervised Learning](https://arxiv.org/abs/2402.12537) | 该论文提出了基于分层贝叶斯统计框架的算法，用于个性化无监督学习，其中开发了适应性算法来平衡利用有限本地数据和协作信息。 |
| [^6] | [Toward Green and Human-Like Artificial Intelligence: A Complete Survey on Contemporary Few-Shot Learning Approaches](https://arxiv.org/abs/2402.03017) | 本文全面调查了少样本学习领域的最新进展，探讨了该方法在解决深度学习在实际应用中的限制方面的潜力和挑战。 |
| [^7] | [Disentangled Condensation for Large-scale Graphs.](http://arxiv.org/abs/2401.12231) | 本文提出了用于大规模图的解缠结凝聚方法DisCo，通过节点和边的凝聚模块实现了对大规模图的高效缩凝，提高了可扩展性和压缩图的保真度。 |
| [^8] | [Input Convex Lipschitz RNN: A Fast and Robust Approach for Engineering Tasks.](http://arxiv.org/abs/2401.07494) | 通过结合输入凸性和Lipschitz连续性的优势，我们开发了一种名为输入凸性Lipschitz循环神经网络的新型网络结构，在计算效率和对抗鲁棒性方面优于现有的循环单元，并适用于多种工程任务。 |
| [^9] | [Motif-aware Attribute Masking for Molecular Graph Pre-training.](http://arxiv.org/abs/2309.04589) | 本研究提出并研究了一种模式感知的属性屏蔽策略，通过利用相邻模式中的原子信息来捕捉模式间的结构，从而提高分子图预训练的效果。 |
| [^10] | [Learning Personalized Decision Support Policies.](http://arxiv.org/abs/2304.06701) | 本文提出了一种学习个性化决策支持策略的算法 $\texttt{THREAD}$，可以为决策者提供不同形式的支持。同时，引入了 $\texttt{Modiste}$ 工具来提供个性化的医学诊断决策支持，使用 $\texttt{THREAD}$ 学习个性化决策支持策略，有效提高了预期的诊断正确性，并减少了严重并发症的风险，同时推荐了更少和更便宜的研究。 |
| [^11] | [Coupled Multiwavelet Neural Operator Learning for Coupled Partial Differential Equations.](http://arxiv.org/abs/2303.02304) | 本论文提出一种耦合多小波神经算子学习的方案，解决了处理耦合多变量映射问题的难点，能够显著提高解决耦合偏微分方程的准确性，并在实验中得到了验证。 |
| [^12] | [A Comparative Evaluation of Quantification Methods.](http://arxiv.org/abs/2103.03223) | 本研究通过对24种不同量化方法在超过40个数据集上进行全面实证比较，填补了量化方法比较研究的空白。我们发现在二分类设置中，基于阈值选择的Median Sweep和TSMax方法、DyS框架和弗里德曼的方法表现最佳；而在多分类设置中，Generaliz方法表现良好。 |

# 详细

[^1]: 针对插值条件下随机加速梯度下降的更快收敛速度

    Faster Convergence of Stochastic Accelerated Gradient Descent under Interpolation

    [https://arxiv.org/abs/2404.02378](https://arxiv.org/abs/2404.02378)

    该论文证明了在插值条件下对随机加速的一般化版本的新收敛速度，在强增长条件下的加速SGD中取得了显著改进。

    

    我们证明了在插值条件下对随机Nesterov加速的一般化版本的新收敛速度。与先前的分析不同，我们的方法加速了任何在期望中取得足够进展的随机梯度方法。证明使用估计序列框架进行，适用于凸函数和强凸函数，并可轻松专门用于强增长条件下的加速SGD。在这种特殊情况下，与先前的工作相比，我们的分析将强增长常数的依赖性从$\rho$减少到$\sqrt{\rho}$。这种改进在最坏情况下相当于条件数的平方根，并解决了关于随机加速的保证可能比SGD更差的批评。

    arXiv:2404.02378v1 Announce Type: cross  Abstract: We prove new convergence rates for a generalized version of stochastic Nesterov acceleration under interpolation conditions. Unlike previous analyses, our approach accelerates any stochastic gradient method which makes sufficient progress in expectation. The proof, which proceeds using the estimating sequences framework, applies to both convex and strongly convex functions and is easily specialized to accelerated SGD under the strong growth condition. In this special case, our analysis reduces the dependence on the strong growth constant from $\rho$ to $\sqrt{\rho}$ as compared to prior work. This improvement is comparable to a square-root of the condition number in the worst case and address criticism that guarantees for stochastic acceleration could be worse than those for SGD.
    
[^2]: NLP验证：走向一种通用的用于认证鲁棒性的方法论

    NLP Verification: Towards a General Methodology for Certifying Robustness

    [https://arxiv.org/abs/2403.10144](https://arxiv.org/abs/2403.10144)

    本文尝试总结和评估由该领域迄今进展而形成的NLP验证流程的一般组成部分，贡献在于提出了将句子嵌入连续空间得到的可验证子空间的一般描述。

    

    深度神经网络在自然语言处理（NLP）领域取得了显著成功，确保它们的安全性和可靠性至关重要：在安全关键的情境中，这些模型必须对变化或攻击具有鲁棒性，并能对其输出给出保证。与计算机视觉不同，NLP缺乏一个统一的验证方法论，尽管近年来文献中取得了一些进展，但对于NLP验证的实用问题常常涉及不深。在本文中，我们尝试提炼和评估一个NLP验证流程的一般组成部分，该流程来源于迄今为止该领域的进展。我们的贡献有两方面：首先，我们给出了将句子嵌入连续空间得到的可验证子空间的一般描述。我们确定了可验证子空间的语义泛化技术挑战，并提出了一种有效处理的方法。

    arXiv:2403.10144v1 Announce Type: cross  Abstract: Deep neural networks have exhibited substantial success in the field of Natural Language Processing (NLP) and ensuring their safety and reliability is crucial: there are safety critical contexts where such models must be robust to variability or attack, and give guarantees over their output. Unlike Computer Vision, NLP lacks a unified verification methodology and, despite recent advancements in literature, they are often light on the pragmatical issues of NLP verification. In this paper, we make an attempt to distil and evaluate general components of an NLP verification pipeline, that emerges from the progress in the field to date. Our contributions are two-fold. Firstly, we give a general characterisation of verifiable subspaces that result from embedding sentences into continuous spaces. We identify, and give an effective method to deal with, the technical challenge of semantic generalisability of verified subspaces; and propose it a
    
[^3]: 图正则化编码器训练用于极端分类

    Graph Regularized Encoder Training for Extreme Classification

    [https://arxiv.org/abs/2402.18434](https://arxiv.org/abs/2402.18434)

    本文提出了一种图正则化编码器训练方法用于极端分类，在实践中发现使用图数据来规范编码器训练比实施 GCN 效果更好。

    

    arXiv:2402.18434v1 通告类型: 新的 摘要: 深度极端分类（XC）旨在训练编码器架构和配套的分类器架构，以从一个非常庞大的标签集合中为数据点打上最相关的子标签集合。在排名、推荐和标记中常见的XC应用中，通常会遇到训练数据极少的尾标签。图卷积网络（GCN）提供了一个方便但计算代价高昂的方法，可利用任务元数据并增强模型在这些设置中的准确性。本文正式确定了在若干用例中，通过用非GCN架构替换GCNs，完全可以避免GCNs的巨大计算成本。本文指出，在这些设置中，使用图数据来规范编码器训练比实施GCN更加有效。基于这些见解，提出了一种替代范式RAMEN，用于利用XC设置中的图元数据。

    arXiv:2402.18434v1 Announce Type: new  Abstract: Deep extreme classification (XC) aims to train an encoder architecture and an accompanying classifier architecture to tag a data point with the most relevant subset of labels from a very large universe of labels. XC applications in ranking, recommendation and tagging routinely encounter tail labels for which the amount of training data is exceedingly small. Graph convolutional networks (GCN) present a convenient but computationally expensive way to leverage task metadata and enhance model accuracies in these settings. This paper formally establishes that in several use cases, the steep computational cost of GCNs is entirely avoidable by replacing GCNs with non-GCN architectures. The paper notices that in these settings, it is much more effective to use graph data to regularize encoder training than to implement a GCN. Based on these insights, an alternative paradigm RAMEN is presented to utilize graph metadata in XC settings that offers 
    
[^4]: 名字的含义是什么？审计大型语言模型中的种族和性别偏见

    What's in a Name? Auditing Large Language Models for Race and Gender Bias

    [https://arxiv.org/abs/2402.14875](https://arxiv.org/abs/2402.14875)

    调查发现，大型语言模型存在种族和性别偏见，尤其对与黑人女性相关的名字表现最不利。审计在模型部署和实施时的重要性得到强调。

    

    我们采用审计设计来调查最先进的大型语言模型中的偏见，包括GPT-4。在我们的研究中，我们引发模型在各种情景下为个人提供建议，比如在购车谈判或选举结果预测过程中。我们发现该建议系统性地对与种族少数群体和女性常见相关的名字产生不利影响。与黑人女性相关的名字得到的结果最不利。这些偏见在42个提示模板和多个模型中都是一致的，表明这是一个系统性问题，而不是孤立事件。在提示中提供数值、与决策相关的锚点可以成功抵消偏见，而定性细节的影响并不一致，甚至可能会加剧差异。我们的研究结果强调了在语言模型部署和实施时进行审计的重要性，以减轻其潜在影响。

    arXiv:2402.14875v1 Announce Type: cross  Abstract: We employ an audit design to investigate biases in state-of-the-art large language models, including GPT-4. In our study, we elicit prompt the models for advice regarding an individual across a variety of scenarios, such as during car purchase negotiations or election outcome predictions. We find that the advice systematically disadvantages names that are commonly associated with racial minorities and women. Names associated with Black women receive the least advantageous outcomes. The biases are consistent across 42 prompt templates and several models, indicating a systemic issue rather than isolated incidents. While providing numerical, decision-relevant anchors in the prompt can successfully counteract the biases, qualitative details have inconsistent effects and may even increase disparities. Our findings underscore the importance of conducting audits at the point of LLM deployment and implementation to mitigate their potential for
    
[^5]: 针对个性化联邦无监督学习的分层贝叶斯方法

    Hierarchical Bayes Approach to Personalized Federated Unsupervised Learning

    [https://arxiv.org/abs/2402.12537](https://arxiv.org/abs/2402.12537)

    该论文提出了基于分层贝叶斯统计框架的算法，用于个性化无监督学习，其中开发了适应性算法来平衡利用有限本地数据和协作信息。

    

    客户本地数据的统计异质性是联邦学习中的重要特征，其促使个性化算法针对本地数据统计量进行定制。尽管已经提出了大量针对个性化监督学习的算法，但通过个性化无监督学习发现本地数据的结构却很少被探索。我们通过基于层次贝叶斯统计框架启动了对这种个性化无监督学习的系统研究。我们开发了基于优化标准的算法，这些算法受启发于层次贝叶斯统计框架。我们开发了适应性算法，发现了利用有限本地数据和协作信息之间的平衡。我们在两个无监督学习任务的背景下进行了这项工作：个性化降维和个性化扩散模型。我们为我们的自适应算法开发了收敛分析，这些分析展示了对问题参数（例如，异质性）的依赖性。

    arXiv:2402.12537v1 Announce Type: new  Abstract: Statistical heterogeneity of clients' local data is an important characteristic in federated learning, motivating personalized algorithms tailored to the local data statistics. Though there has been a plethora of algorithms proposed for personalized supervised learning, discovering the structure of local data through personalized unsupervised learning is less explored. We initiate a systematic study of such personalized unsupervised learning by developing algorithms based on optimization criteria inspired by a hierarchical Bayesian statistical framework. We develop adaptive algorithms that discover the balance between using limited local data and collaborative information. We do this in the context of two unsupervised learning tasks: personalized dimensionality reduction and personalized diffusion models. We develop convergence analyses for our adaptive algorithms which illustrate the dependence on problem parameters (e.g., heterogeneity
    
[^6]: 向绿色且类人的人工智能迈进：当代少样本学习方法的全面调查

    Toward Green and Human-Like Artificial Intelligence: A Complete Survey on Contemporary Few-Shot Learning Approaches

    [https://arxiv.org/abs/2402.03017](https://arxiv.org/abs/2402.03017)

    本文全面调查了少样本学习领域的最新进展，探讨了该方法在解决深度学习在实际应用中的限制方面的潜力和挑战。

    

    尽管深度学习取得了广泛的成功，但其对数据的需求和计算的昂贵性使其在许多数据受限的真实应用中不实用。少样本学习（FSL）旨在通过实现对新学习任务的快速适应来解决这些限制，并在近年来取得了显著发展。本调查提供了该领域最新进展的全面概述。首先，正式定义了FSL，并介绍了它与不同学习领域的关系。引入了一种新的分类法，扩展了以前提出的方法，并对经典和新领域中的实际应用进行了描述。最后，讨论了塑造该领域的最新趋势、突出挑战和有前途的未来研究方向。

    Despite deep learning's widespread success, its data-hungry and computationally expensive nature makes it impractical for many data-constrained real-world applications. Few-Shot Learning (FSL) aims to address these limitations by enabling rapid adaptation to novel learning tasks, seeing significant growth in recent years. This survey provides a comprehensive overview of the field's latest advancements. Initially, FSL is formally defined, and its relationship with different learning fields is presented. A novel taxonomy is introduced, extending previously proposed ones, and real-world applications in classic and novel fields are described. Finally, recent trends shaping the field, outstanding challenges, and promising future research directions are discussed.
    
[^7]: 大规模图的解缠结凝聚

    Disentangled Condensation for Large-scale Graphs. (arXiv:2401.12231v1 [cs.SI])

    [http://arxiv.org/abs/2401.12231](http://arxiv.org/abs/2401.12231)

    本文提出了用于大规模图的解缠结凝聚方法DisCo，通过节点和边的凝聚模块实现了对大规模图的高效缩凝，提高了可扩展性和压缩图的保真度。

    

    图解缠结已经成为一种有趣的技术，为大规模图提供了一种更紧凑但信息丰富的小图，以节省大规模图学习的昂贵成本。尽管取得了有前途的结果，但先前的图解缠结方法常常采用纠缠的缩凝策略，同时涉及节点和边的缩凝，导致大量的GPU内存需求。这种纠缠的策略极大地阻碍了图解缠结的可扩展性，削弱了它对极大规模图的缩凝和高保真度压缩图的能力。因此，本文提出了用于大规模图的解缠结凝聚，简称为DisCo，以提供可扩展的图解缠结，适用于不同规模的图。DisCo的核心是两个互补的组件，即节点和边的凝聚模块，在解缠的方式下实现节点和边的凝聚。

    Graph condensation has emerged as an intriguing technique to provide Graph Neural Networks for large-scale graphs with a more compact yet informative small graph to save the expensive costs of large-scale graph learning. Despite the promising results achieved, previous graph condensation methods often employ an entangled condensation strategy that involves condensing nodes and edges simultaneously, leading to substantial GPU memory demands. This entangled strategy has considerably impeded the scalability of graph condensation, impairing its capability to condense extremely large-scale graphs and produce condensed graphs with high fidelity. Therefore, this paper presents Disentangled Condensation for large-scale graphs, abbreviated as DisCo, to provide scalable graph condensation for graphs of varying sizes. At the heart of DisCo are two complementary components, namely node and edge condensation modules, that realize the condensation of nodes and edges in a disentangled manner. In the 
    
[^8]: 输入凸性Lipschitz RNN: 一种用于工程任务的快速和鲁棒的方法

    Input Convex Lipschitz RNN: A Fast and Robust Approach for Engineering Tasks. (arXiv:2401.07494v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2401.07494](http://arxiv.org/abs/2401.07494)

    通过结合输入凸性和Lipschitz连续性的优势，我们开发了一种名为输入凸性Lipschitz循环神经网络的新型网络结构，在计算效率和对抗鲁棒性方面优于现有的循环单元，并适用于多种工程任务。

    

    计算效率和对抗鲁棒性是真实世界工程应用中的关键因素。然而，传统的神经网络往往在同时或分别解决这两个问题方面存在不足。通过从自然物理系统和现有文献中获取的见解，已知输入凸性结构增强了计算效率，而Lipschitz约束结构增强了对抗鲁棒性。通过利用凸性和Lipschitz连续性的优点，我们开发了一种新的网络结构，称为输入凸性Lipschitz循环神经网络。该模型在计算效率和对抗鲁棒性方面表现优于现有的循环单元，适用于一系列工程任务，包括基准MNIST图像分类、新加坡LHT Holdings公司的实际太阳能光伏系统规划中的实时太阳辐射预测，以及化学反应器的实时模型预测控制优化等。

    Computational efficiency and adversarial robustness are critical factors in real-world engineering applications. Yet, conventional neural networks often fall short in addressing both simultaneously, or even separately. Drawing insights from natural physical systems and existing literature, it is known that an input convex architecture enhances computational efficiency, while a Lipschitz-constrained architecture bolsters adversarial robustness. By leveraging the strengths of convexity and Lipschitz continuity, we develop a novel network architecture, termed Input Convex Lipschitz Recurrent Neural Networks. This model outperforms existing recurrent units across a spectrum of engineering tasks in terms of computational efficiency and adversarial robustness. These tasks encompass a benchmark MNIST image classification, real-world solar irradiance prediction for Solar PV system planning at LHT Holdings in Singapore, and real-time Model Predictive Control optimization for a chemical reactor.
    
[^9]: 面向分子图预训练的模式感知属性屏蔽

    Motif-aware Attribute Masking for Molecular Graph Pre-training. (arXiv:2309.04589v1 [cs.LG])

    [http://arxiv.org/abs/2309.04589](http://arxiv.org/abs/2309.04589)

    本研究提出并研究了一种模式感知的属性屏蔽策略，通过利用相邻模式中的原子信息来捕捉模式间的结构，从而提高分子图预训练的效果。

    

    在图神经网络的预训练中，属性重构用于预测节点或边的特征。通过给定大量的分子，它们学习捕捉结构知识，这对于各种下游属性预测任务在化学、生物医学和材料科学中至关重要。先前的策略是随机选择节点进行属性屏蔽，利用局部邻居的信息。然而，对这些邻居的过度依赖抑制了模型从更高级的亚结构中学习。例如，模型从预测苯环中的三个碳原子中学到的信息很少，但是可以从功能基团之间的相互连接中学到更多信息，也可以称为化学模式。在这项工作中，我们提出并研究了模式感知的属性屏蔽策略，通过利用相邻模式中的原子信息来捕捉模式间的结构。一旦每个图被分解为不相交的

    Attribute reconstruction is used to predict node or edge features in the pre-training of graph neural networks. Given a large number of molecules, they learn to capture structural knowledge, which is transferable for various downstream property prediction tasks and vital in chemistry, biomedicine, and material science. Previous strategies that randomly select nodes to do attribute masking leverage the information of local neighbors However, the over-reliance of these neighbors inhibits the model's ability to learn from higher-level substructures. For example, the model would learn little from predicting three carbon atoms in a benzene ring based on the other three but could learn more from the inter-connections between the functional groups, or called chemical motifs. In this work, we propose and investigate motif-aware attribute masking strategies to capture inter-motif structures by leveraging the information of atoms in neighboring motifs. Once each graph is decomposed into disjoint
    
[^10]: 学习个性化决策支持策略

    Learning Personalized Decision Support Policies. (arXiv:2304.06701v1 [cs.LG])

    [http://arxiv.org/abs/2304.06701](http://arxiv.org/abs/2304.06701)

    本文提出了一种学习个性化决策支持策略的算法 $\texttt{THREAD}$，可以为决策者提供不同形式的支持。同时，引入了 $\texttt{Modiste}$ 工具来提供个性化的医学诊断决策支持，使用 $\texttt{THREAD}$ 学习个性化决策支持策略，有效提高了预期的诊断正确性，并减少了严重并发症的风险，同时推荐了更少和更便宜的研究。

    

    个体决策者可能需要不同形式的支持来提高决策结果，但重要的问题是，哪种形式的支持会在低成本下导致准确的决策。本文提出了学习决策支持策略的方法，它在给定输入时选择是否以及如何提供支持。我们考虑没有先验信息的决策者，并将学习各自的策略形式化为一个多目标优化问题，这个问题权衡了准确性和成本。使用随机环境的技术，我们提出了 $\texttt{THREAD}$，这是一种个性化决策支持策略的在线算法，并设计了一种超参数调整策略，以利用模拟人类行为来确定成本-性能权衡。我们提供计算实验来证明 $\texttt{THREAD}$ 相对于线下基线的优势。然后，我们推出了一个交互式工具 $\texttt{Modiste}$，它为现实中的医学诊断提供个性化决策支持。$\texttt{Modiste}$ 使用 $\texttt{THREAD}$ 为每位医生学习个性化的决策支持策略，并推荐个性化研究以优化患者的预期结果并将严重并发症的风险降至最低。使用电子健康记录数据，我们展示了 $\texttt{Modiste}$ 显著提高了预期的诊断正确性，并减少了严重并发症的风险，同时推荐了更少和更便宜的研究。

    Individual human decision-makers may benefit from different forms of support to improve decision outcomes. However, a key question is which form of support will lead to accurate decisions at a low cost. In this work, we propose learning a decision support policy that, for a given input, chooses which form of support, if any, to provide. We consider decision-makers for whom we have no prior information and formalize learning their respective policies as a multi-objective optimization problem that trades off accuracy and cost. Using techniques from stochastic contextual bandits, we propose $\texttt{THREAD}$, an online algorithm to personalize a decision support policy for each decision-maker, and devise a hyper-parameter tuning strategy to identify a cost-performance trade-off using simulated human behavior. We provide computational experiments to demonstrate the benefits of $\texttt{THREAD}$ compared to offline baselines. We then introduce $\texttt{Modiste}$, an interactive tool that pr
    
[^11]: 针对耦合偏微分方程的耦合多小波神经算子学习

    Coupled Multiwavelet Neural Operator Learning for Coupled Partial Differential Equations. (arXiv:2303.02304v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2303.02304](http://arxiv.org/abs/2303.02304)

    本论文提出一种耦合多小波神经算子学习的方案，解决了处理耦合多变量映射问题的难点，能够显著提高解决耦合偏微分方程的准确性，并在实验中得到了验证。

    

    耦合偏微分方程是描述许多物理过程复杂动态的关键任务。最近，神经算子已经展示出通过在傅里叶/小波空间直接学习积分核来解决PDE的能力。对于耦合PDE的解决方法，难点在于处理函数之间的耦合映射。为此，我们提出了一种耦合多小波神经算子（CMWNO）学习方案，通过在小波空间中进行多小波分解和重构过程中解耦合积分核。在解决Gray-Scott（GS）方程和非局部均场博弈（MFG）问题等耦合PDE方面，所提出的模型相对于先前基于学习的求解器实现了显著提高的准确性。根据我们的实验结果，所提出的模型相对于最先进模型的$L^2$误差表现出了$2\times \sim 4\times$的改进。

    Coupled partial differential equations (PDEs) are key tasks in modeling the complex dynamics of many physical processes. Recently, neural operators have shown the ability to solve PDEs by learning the integral kernel directly in Fourier/Wavelet space, so the difficulty for solving the coupled PDEs depends on dealing with the coupled mappings between the functions. Towards this end, we propose a \textit{coupled multiwavelets neural operator} (CMWNO) learning scheme by decoupling the coupled integral kernels during the multiwavelet decomposition and reconstruction procedures in the Wavelet space. The proposed model achieves significantly higher accuracy compared to previous learning-based solvers in solving the coupled PDEs including Gray-Scott (GS) equations and the non-local mean field game (MFG) problem. According to our experimental results, the proposed model exhibits a $2\times \sim 4\times$ improvement relative $L$2 error compared to the best results from the state-of-the-art mode
    
[^12]: 量化方法的比较评估

    A Comparative Evaluation of Quantification Methods. (arXiv:2103.03223v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2103.03223](http://arxiv.org/abs/2103.03223)

    本研究通过对24种不同量化方法在超过40个数据集上进行全面实证比较，填补了量化方法比较研究的空白。我们发现在二分类设置中，基于阈值选择的Median Sweep和TSMax方法、DyS框架和弗里德曼的方法表现最佳；而在多分类设置中，Generaliz方法表现良好。

    

    量化是指在数据集中预测类别分布的问题。它也代表着一个在监督式机器学习中不断发展的研究领域，近年来提出了大量不同的算法。然而，目前还没有一份全面的实证比较量化方法的研究，以支持算法选择。在本研究中，我们通过对超过40个数据集进行了24种不同量化方法的彻底实证性性能比较，包括二分类和多分类量化设置，填补了这一研究空白。我们观察到没有单一算法能够在所有竞争对手中始终表现最佳，但我们确定了一组在二分类设置中表现最佳的方法，包括基于阈值选择的Median Sweep和TSMax方法、DyS框架和弗里德曼的方法。对于多分类设置，我们观察到另一组算法表现良好，包括Generaliz方法。

    Quantification represents the problem of predicting class distributions in a dataset. It also represents a growing research field in supervised machine learning, for which a large variety of different algorithms has been proposed in recent years. However, a comprehensive empirical comparison of quantification methods that supports algorithm selection is not available yet. In this work, we close this research gap by conducting a thorough empirical performance comparison of 24 different quantification methods on overall more than 40 data sets, considering binary as well as multiclass quantification settings. We observe that no single algorithm generally outperforms all competitors, but identify a group of methods including the threshold selection-based Median Sweep and TSMax methods, the DyS framework, and Friedman's method that performs best in the binary setting. For the multiclass setting, we observe that a different group of algorithms yields good performance, including the Generaliz
    

