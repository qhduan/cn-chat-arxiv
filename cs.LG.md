# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [GuideGen: A Text-guided Framework for Joint CT Volume and Anatomical structure Generation](https://arxiv.org/abs/2403.07247) | 该论文提出了一种名为GuideGen的框架，可以根据文本提示联合生成CT图像和腹部器官以及结直肠癌组织掩膜，为医学图像分析领域提供了一种生成数据集的新途径。 |
| [^2] | [OCD-FL: A Novel Communication-Efficient Peer Selection-based Decentralized Federated Learning](https://arxiv.org/abs/2403.04037) | 提出了一种名为OCD-FL的新方案，通过系统化的FL对等选择进行协作，旨在在减少能耗的同时实现最大的FL知识增益 |
| [^3] | [Stable Neural Stochastic Differential Equations in Analyzing Irregular Time Series Data](https://arxiv.org/abs/2402.14989) | 神经常微分方程（Neural ODEs）的扩展——神经随机微分方程（Neural SDEs）在处理不规则时间序列数据中的稳定性和性能方面提出了重要指导，需要谨慎设计漂移和扩散函数以保持稳定性。 |
| [^4] | [AdaFlow: Imitation Learning with Variance-Adaptive Flow-Based Policies](https://arxiv.org/abs/2402.04292) | AdaFlow是一个基于流模型的模仿学习框架，通过使用变异自适应ODE求解器，在保持多样性的同时，提供快速推理能力。 |
| [^5] | [Do DL models and training environments have an impact on energy consumption?.](http://arxiv.org/abs/2307.05520) | 本研究分析了模型架构和训练环境对训练更环保的计算机视觉模型的影响，并找出了能源效率和模型正确性之间的权衡关系。 |
| [^6] | [NeuroGraph: Benchmarks for Graph Machine Learning in Brain Connectomics.](http://arxiv.org/abs/2306.06202) | 本文介绍了神经成像领域的图机器学习基准测试NeuroGraph，并探讨了数据集生成的搜索空间。 |
| [^7] | [How Sparse Can We Prune A Deep Network: A Geometric Viewpoint.](http://arxiv.org/abs/2306.05857) | 本文从高维几何的角度，通过在原始损失函数中强制施加稀疏性约束，描述了深度网络剪枝比率的相变点，该点等于某些凸体的平方高斯宽度除以参数的原始维度。 |
| [^8] | [Deep ReLU Networks Have Surprisingly Simple Polytopes.](http://arxiv.org/abs/2305.09145) | 本文通过计算和分析ReLU网络多面体的单纯形直方图，发现在初始化和梯度下降时它们结构相对简单，这说明了一种新的隐式偏见。 |
| [^9] | [Spectrum Breathing: Protecting Over-the-Air Federated Learning Against Interference.](http://arxiv.org/abs/2305.05933) | Spectrum Breathing是一种保护空中联合学习免受干扰的实际方法，通过将随机梯度剪枝和扩频级联起来，以压制干扰而无需扩展带宽。代价是增加的学习延迟。 |
| [^10] | [Algorithms for Social Justice: Affirmative Action in Social Networks.](http://arxiv.org/abs/2305.03223) | 本文介绍了一个新的基于谱图理论的链接推荐算法ERA-Link，旨在缓解现有推荐算法带来的信息孤岛和社会成见，实现社交网络平台的社会正义目标。 |
| [^11] | [What Do GNNs Actually Learn? Towards Understanding their Representations.](http://arxiv.org/abs/2304.10851) | 本文研究了四种GNN模型，指出其中两种将所有节点嵌入同一特征向量中，而另外两种模型生成的表示与输入图中的步长数量相关。在一定条件下，不同结构的节点可能有相似的表示。 |
| [^12] | [Huber-energy measure quantization.](http://arxiv.org/abs/2212.08162) | 该论文提出了一种Huber能量量化的算法，用于找到目标概率定律的最佳逼近，通过最小化原测度与量化版本之间的统计距离来实现。该算法已在多维高斯混合物、维纳空间魔方等几个数据库上进行了测试。 |
| [^13] | [Continuous Generative Neural Networks.](http://arxiv.org/abs/2205.14627) | 本文介绍了一种连续生成神经网络(CGNN)的模型，使用条件保证CGNN是单射的，其生成流形被用于求解反问题，并证明了其方法的有效性和稳健性。 |

# 详细

[^1]: GuideGen：一种用于联合CT体积和解剖结构生成的文本引导框架

    GuideGen: A Text-guided Framework for Joint CT Volume and Anatomical structure Generation

    [https://arxiv.org/abs/2403.07247](https://arxiv.org/abs/2403.07247)

    该论文提出了一种名为GuideGen的框架，可以根据文本提示联合生成CT图像和腹部器官以及结直肠癌组织掩膜，为医学图像分析领域提供了一种生成数据集的新途径。

    

    arXiv:2403.07247v1 公告类型：交叉 摘要：为了收集带有图像和相应标签的大型医学数据集而进行的注释负担和大量工作很少是划算且令人望而生畏的。这导致了缺乏丰富的训练数据，削弱了下游任务，并在一定程度上加剧了医学领域面临的图像分析挑战。作为一种权宜之计，鉴于生成性神经模型的最近成功，现在可以在外部约束的引导下以高保真度合成图像数据集。本文探讨了这种可能性，并提出了GuideGen：一种联合生成腹部器官和结直肠癌CT图像和组织掩膜的管线，其受文本提示条件约束。首先，我们介绍了体积掩膜采样器，以适应掩膜标签的离散分布并生成低分辨率3D组织掩膜。其次，我们的条件图像生成器会在收到相应文本提示的情况下自回归生成CT切片。

    arXiv:2403.07247v1 Announce Type: cross  Abstract: The annotation burden and extensive labor for gathering a large medical dataset with images and corresponding labels are rarely cost-effective and highly intimidating. This results in a lack of abundant training data that undermines downstream tasks and partially contributes to the challenge image analysis faces in the medical field. As a workaround, given the recent success of generative neural models, it is now possible to synthesize image datasets at a high fidelity guided by external constraints. This paper explores this possibility and presents \textbf{GuideGen}: a pipeline that jointly generates CT images and tissue masks for abdominal organs and colorectal cancer conditioned on a text prompt. Firstly, we introduce Volumetric Mask Sampler to fit the discrete distribution of mask labels and generate low-resolution 3D tissue masks. Secondly, our Conditional Image Generator autoregressively generates CT slices conditioned on a corre
    
[^2]: OCD-FL: 一种基于点对点选择的通信高效去中心化联邦学习

    OCD-FL: A Novel Communication-Efficient Peer Selection-based Decentralized Federated Learning

    [https://arxiv.org/abs/2403.04037](https://arxiv.org/abs/2403.04037)

    提出了一种名为OCD-FL的新方案，通过系统化的FL对等选择进行协作，旨在在减少能耗的同时实现最大的FL知识增益

    

    边缘智能和不断增长的物联网网络的结合开创了协作机器学习的新时代，联邦学习(FL)作为最突出的范式出现。随着人们对这些学习方案越来越感兴趣，研究人员开始解决它们最基本的一些限制。事实上，具有中心聚合器的传统FL存在单点故障和网络瓶颈。为了规避这个问题，提出了节点在点对点网络中协作的去中心化FL。尽管后者效率高，但在去中心化FL中，通信成本和数据异质性仍然是关键挑战。在这种背景下，我们提出了一种名为机会主义通信高效的去中心化联邦学习(OCD-FL)的新方案，其中包括系统化的FL对等选择以进行协作，旨在实现最大的FL知识增益同时减少能耗。

    arXiv:2403.04037v1 Announce Type: new  Abstract: The conjunction of edge intelligence and the ever-growing Internet-of-Things (IoT) network heralds a new era of collaborative machine learning, with federated learning (FL) emerging as the most prominent paradigm. With the growing interest in these learning schemes, researchers started addressing some of their most fundamental limitations. Indeed, conventional FL with a central aggregator presents a single point of failure and a network bottleneck. To bypass this issue, decentralized FL where nodes collaborate in a peer-to-peer network has been proposed. Despite the latter's efficiency, communication costs and data heterogeneity remain key challenges in decentralized FL. In this context, we propose a novel scheme, called opportunistic communication-efficient decentralized federated learning, a.k.a., OCD-FL, consisting of a systematic FL peer selection for collaboration, aiming to achieve maximum FL knowledge gain while reducing energy co
    
[^3]: 分析不规则时间序列数据中的稳定神经随机微分方程

    Stable Neural Stochastic Differential Equations in Analyzing Irregular Time Series Data

    [https://arxiv.org/abs/2402.14989](https://arxiv.org/abs/2402.14989)

    神经常微分方程（Neural ODEs）的扩展——神经随机微分方程（Neural SDEs）在处理不规则时间序列数据中的稳定性和性能方面提出了重要指导，需要谨慎设计漂移和扩散函数以保持稳定性。

    

    实际时间序列数据中的不规则采样间隔和缺失值对于假设一致间隔和完整数据的传统方法构成挑战。神经常微分方程（Neural ODEs）提供了一种替代方法，利用神经网络与常微分方程求解器结合，通过参数化向量场学习连续潜在表示。神经随机微分方程（Neural SDEs）通过引入扩散项扩展了神经常微分方程，然而在处理不规则间隔和缺失值时，这种添加并不是微不足道的。因此，仔细设计漂移和扩散函数对于保持稳定性和增强性能至关重要，而粗心的选择可能导致出现没有强解、随机破坏或不稳定的Euler离散化等不利的性质，显著影响神经随机微分方程的性能。

    arXiv:2402.14989v1 Announce Type: cross  Abstract: Irregular sampling intervals and missing values in real-world time series data present challenges for conventional methods that assume consistent intervals and complete data. Neural Ordinary Differential Equations (Neural ODEs) offer an alternative approach, utilizing neural networks combined with ODE solvers to learn continuous latent representations through parameterized vector fields. Neural Stochastic Differential Equations (Neural SDEs) extend Neural ODEs by incorporating a diffusion term, although this addition is not trivial, particularly when addressing irregular intervals and missing values. Consequently, careful design of drift and diffusion functions is crucial for maintaining stability and enhancing performance, while incautious choices can result in adverse properties such as the absence of strong solutions, stochastic destabilization, or unstable Euler discretizations, significantly affecting Neural SDEs' performance. In 
    
[^4]: AdaFlow: 变异自适应流策略的模仿学习

    AdaFlow: Imitation Learning with Variance-Adaptive Flow-Based Policies

    [https://arxiv.org/abs/2402.04292](https://arxiv.org/abs/2402.04292)

    AdaFlow是一个基于流模型的模仿学习框架，通过使用变异自适应ODE求解器，在保持多样性的同时，提供快速推理能力。

    

    基于扩散的模仿学习在多模态决策中改进了行为克隆（BC），但由于扩散过程中的递归而导致推理速度显著减慢。这促使我们设计高效的策略生成器，同时保持生成多样化动作的能力。为了解决这个挑战，我们提出了AdaFlow，这是一个基于流模型的模仿学习框架。AdaFlow使用状态条件的常微分方程（ODE）表示策略，这被称为概率流。我们揭示了它们训练损失的条件方差与ODE的离散化误差之间的有趣关系。基于这个观察，我们提出了一个变异自适应ODE求解器，在推理阶段可以调整步长，使AdaFlow成为一个自适应决策者，能够快速推理而不牺牲多样性。有趣的是，当动作分布被降低到一步生成器时，它自动退化到一个一步生成器。

    Diffusion-based imitation learning improves Behavioral Cloning (BC) on multi-modal decision-making, but comes at the cost of significantly slower inference due to the recursion in the diffusion process. It urges us to design efficient policy generators while keeping the ability to generate diverse actions. To address this challenge, we propose AdaFlow, an imitation learning framework based on flow-based generative modeling. AdaFlow represents the policy with state-conditioned ordinary differential equations (ODEs), which are known as probability flows. We reveal an intriguing connection between the conditional variance of their training loss and the discretization error of the ODEs. With this insight, we propose a variance-adaptive ODE solver that can adjust its step size in the inference stage, making AdaFlow an adaptive decision-maker, offering rapid inference without sacrificing diversity. Interestingly, it automatically reduces to a one-step generator when the action distribution i
    
[^5]: DL模型和训练环境对能源消耗有影响吗？

    Do DL models and training environments have an impact on energy consumption?. (arXiv:2307.05520v1 [cs.LG])

    [http://arxiv.org/abs/2307.05520](http://arxiv.org/abs/2307.05520)

    本研究分析了模型架构和训练环境对训练更环保的计算机视觉模型的影响，并找出了能源效率和模型正确性之间的权衡关系。

    

    当前计算机视觉领域的研究主要集中在提高深度学习（DL）的正确性和推理时间性能上。然而，目前很少有关于训练DL模型带来巨大碳足迹的研究。本研究旨在分析模型架构和训练环境对训练更环保的计算机视觉模型的影响。我们将这个目标分为两个研究问题。首先，我们分析模型架构对实现更环保模型同时保持正确性在最佳水平的影响。其次，我们研究训练环境对生成更环保模型的影响。为了调查这些关系，我们在模型训练过程中收集了与能源效率和模型正确性相关的多个指标。然后，我们描述了模型架构在测量能源效率和模型正确性方面的权衡，以及它们与训练环境的关系。我们在一个实验平台上进行了这项研究。

    Current research in the computer vision field mainly focuses on improving Deep Learning (DL) correctness and inference time performance. However, there is still little work on the huge carbon footprint that has training DL models. This study aims to analyze the impact of the model architecture and training environment when training greener computer vision models. We divide this goal into two research questions. First, we analyze the effects of model architecture on achieving greener models while keeping correctness at optimal levels. Second, we study the influence of the training environment on producing greener models. To investigate these relationships, we collect multiple metrics related to energy efficiency and model correctness during the models' training. Then, we outline the trade-offs between the measured energy efficiency and the models' correctness regarding model architecture, and their relationship with the training environment. We conduct this research in the context of a 
    
[^6]: NeuroGraph:面向脑连接组学的图机器学习基准测试

    NeuroGraph: Benchmarks for Graph Machine Learning in Brain Connectomics. (arXiv:2306.06202v1 [cs.LG])

    [http://arxiv.org/abs/2306.06202](http://arxiv.org/abs/2306.06202)

    本文介绍了神经成像领域的图机器学习基准测试NeuroGraph，并探讨了数据集生成的搜索空间。

    

    机器学习为分析高维功能性神经成像数据提供了有价值的工具，已被证明对预测各种神经疾病、精神障碍和认知模式有效。在功能磁共振成像研究中，大脑区域之间的相互作用通常使用基于图的表示进行建模。图机器学习方法的有效性已在多个领域得到证实，标志着数据解释和预测建模中的一个转变步骤。然而，尽管有前景，但由于图形数据集构建的广泛预处理流水线和大参数搜索空间，在神经成像领域中应用这些技术的转换仍然受到意外的限制。本文介绍了NeuroGraph(一个基于图的神经成像数据集)，它涵盖了多个行为和认知特征类别。我们深入探讨了数据集生成搜索空间

    Machine learning provides a valuable tool for analyzing high-dimensional functional neuroimaging data, and is proving effective in predicting various neurological conditions, psychiatric disorders, and cognitive patterns. In functional Magnetic Resonance Imaging (MRI) research, interactions between brain regions are commonly modeled using graph-based representations. The potency of graph machine learning methods has been established across myriad domains, marking a transformative step in data interpretation and predictive modeling. Yet, despite their promise, the transposition of these techniques to the neuroimaging domain remains surprisingly under-explored due to the expansive preprocessing pipeline and large parameter search space for graph-based datasets construction. In this paper, we introduce NeuroGraph, a collection of graph-based neuroimaging datasets that span multiple categories of behavioral and cognitive traits. We delve deeply into the dataset generation search space by c
    
[^7]: 深度网络可以被剪枝到多么稀疏：几何视角下的研究

    How Sparse Can We Prune A Deep Network: A Geometric Viewpoint. (arXiv:2306.05857v1 [stat.ML])

    [http://arxiv.org/abs/2306.05857](http://arxiv.org/abs/2306.05857)

    本文从高维几何的角度，通过在原始损失函数中强制施加稀疏性约束，描述了深度网络剪枝比率的相变点，该点等于某些凸体的平方高斯宽度除以参数的原始维度。

    

    过度参数化是深度神经网络最重要的特征之一。虽然它可以提供出色的泛化性能，但同时也强加了重大的存储负担，因此有必要研究网络剪枝。一个自然而基本的问题是：我们能剪枝一个深度网络到多么稀疏（几乎不影响性能）？为了解决这个问题，本文采用了第一原理方法，具体地，只通过在原始损失函数中强制施加稀疏性约束，我们能够从高维几何的角度描述剪枝比率的尖锐相变点，该点对应于可行和不可行之间的边界。结果表明，剪枝比率的相变点等于某些凸体的平方高斯宽度，这些凸体是由$l_1$-规则化损失函数得出的，除以参数的原始维度。作为副产品，我们证明了剪枝过程中参数的分布性质。

    Overparameterization constitutes one of the most significant hallmarks of deep neural networks. Though it can offer the advantage of outstanding generalization performance, it meanwhile imposes substantial storage burden, thus necessitating the study of network pruning. A natural and fundamental question is: How sparse can we prune a deep network (with almost no hurt on the performance)? To address this problem, in this work we take a first principles approach, specifically, by merely enforcing the sparsity constraint on the original loss function, we're able to characterize the sharp phase transition point of pruning ratio, which corresponds to the boundary between the feasible and the infeasible, from the perspective of high-dimensional geometry. It turns out that the phase transition point of pruning ratio equals the squared Gaussian width of some convex body resulting from the $l_1$-regularized loss function, normalized by the original dimension of parameters. As a byproduct, we pr
    
[^8]: 深层ReLU网络的多面体异常简单

    Deep ReLU Networks Have Surprisingly Simple Polytopes. (arXiv:2305.09145v1 [cs.LG])

    [http://arxiv.org/abs/2305.09145](http://arxiv.org/abs/2305.09145)

    本文通过计算和分析ReLU网络多面体的单纯形直方图，发现在初始化和梯度下降时它们结构相对简单，这说明了一种新的隐式偏见。

    

    ReLU网络是一种多面体上的分段线性函数。研究这种多面体的性质对于神经网络的研究和发展至关重要。目前，对于多面体的理论和实证研究仅停留在计算数量的水平，这远远不能完整地描述多面体。为了将特征提升到一个新的水平，我们提出通过三角剖分多面体得出多面体的形状。通过计算和分析不同多面体的单纯形直方图，我们发现ReLU网络在初始化和梯度下降时具有相对简单的多面体结构，尽管这些多面体从理论上来说可以非常丰富和复杂。这一发现可以被认为是一种新的隐式偏见。随后，我们使用非平凡的组合推导来理论上解释为什么增加深度不会创建更复杂的多面体，通过限制每个维度的平均单纯形数量。

    A ReLU network is a piecewise linear function over polytopes. Figuring out the properties of such polytopes is of fundamental importance for the research and development of neural networks. So far, either theoretical or empirical studies on polytopes only stay at the level of counting their number, which is far from a complete characterization of polytopes. To upgrade the characterization to a new level, here we propose to study the shapes of polytopes via the number of simplices obtained by triangulating the polytope. Then, by computing and analyzing the histogram of simplices across polytopes, we find that a ReLU network has relatively simple polytopes under both initialization and gradient descent, although these polytopes theoretically can be rather diverse and complicated. This finding can be appreciated as a novel implicit bias. Next, we use nontrivial combinatorial derivation to theoretically explain why adding depth does not create a more complicated polytope by bounding the av
    
[^9]: Spectrum Breathing：保护空中联合学习免受干扰

    Spectrum Breathing: Protecting Over-the-Air Federated Learning Against Interference. (arXiv:2305.05933v1 [cs.LG])

    [http://arxiv.org/abs/2305.05933](http://arxiv.org/abs/2305.05933)

    Spectrum Breathing是一种保护空中联合学习免受干扰的实际方法，通过将随机梯度剪枝和扩频级联起来，以压制干扰而无需扩展带宽。代价是增加的学习延迟。

    

    联合学习是一种从分布式移动数据中蒸馏人工智能的广泛应用范例。但联合学习在移动网络中的部署可能会受到邻近单元或干扰源的干扰而受损。现有的干扰抑制技术需要多单元合作或至少需要昂贵的干扰通道状态信息。另一方面，将干扰视为噪声进行功率控制可能并不有效，由于预算限制，也由于这种机制可能会触发干扰源的反制措施。作为保护空中联合学习免受干扰的实际方法，我们提出了Spectrum Breathing，它将随机梯度剪枝和扩频级联起来，以压制干扰而无需扩展带宽。代价是通过利用剪枝导致学习速度优雅降低而增加的学习延迟。我们将两个操作同步，以保证它们的级别是相互对应的。

    Federated Learning (FL) is a widely embraced paradigm for distilling artificial intelligence from distributed mobile data. However, the deployment of FL in mobile networks can be compromised by exposure to interference from neighboring cells or jammers. Existing interference mitigation techniques require multi-cell cooperation or at least interference channel state information, which is expensive in practice. On the other hand, power control that treats interference as noise may not be effective due to limited power budgets, and also that this mechanism can trigger countermeasures by interference sources. As a practical approach for protecting FL against interference, we propose Spectrum Breathing, which cascades stochastic-gradient pruning and spread spectrum to suppress interference without bandwidth expansion. The cost is higher learning latency by exploiting the graceful degradation of learning speed due to pruning. We synchronize the two operations such that their levels are contr
    
[^10]: 社会正义算法：社交网络中的平权行动

    Algorithms for Social Justice: Affirmative Action in Social Networks. (arXiv:2305.03223v1 [cs.SI])

    [http://arxiv.org/abs/2305.03223](http://arxiv.org/abs/2305.03223)

    本文介绍了一个新的基于谱图理论的链接推荐算法ERA-Link，旨在缓解现有推荐算法带来的信息孤岛和社会成见，实现社交网络平台的社会正义目标。

    

    链接推荐算法对于世界各地数十亿用户的人际关系产生了影响。为了最大化相关性，它们通常建议连接相互相似的用户。然而，这被发现会产生信息孤岛，加剧弱势突出群体所遭受的孤立，并延续社会成见。为了缓解这些限制，大量研究致力于实现公平的链接推荐方法。然而，大多数方法并不质疑链接推荐算法的最终目标，即数据交易的复杂商业模型中用户参与的货币化。本文主张实现社交网络平台玩家和目的的多样化，以实现社会正义。为了说明这一概念目标，我们提出了ERA-Link，这是一种基于谱图理论的新型链接推荐算法，可以抵消系统性的社会歧视。

    Link recommendation algorithms contribute to shaping human relations of billions of users worldwide in social networks. To maximize relevance, they typically propose connecting users that are similar to each other. This has been found to create information silos, exacerbating the isolation suffered by vulnerable salient groups and perpetuating societal stereotypes. To mitigate these limitations, a significant body of work has been devoted to the implementation of fair link recommendation methods. However, most approaches do not question the ultimate goal of link recommendation algorithms, namely the monetization of users' engagement in intricate business models of data trade. This paper advocates for a diversification of players and purposes of social network platforms, aligned with the pursue of social justice. To illustrate this conceptual goal, we present ERA-Link, a novel link recommendation algorithm based on spectral graph theory that counteracts the systemic societal discriminat
    
[^11]: GNNs到底在学什么？——理解它们的表示方法

    What Do GNNs Actually Learn? Towards Understanding their Representations. (arXiv:2304.10851v1 [cs.LG])

    [http://arxiv.org/abs/2304.10851](http://arxiv.org/abs/2304.10851)

    本文研究了四种GNN模型，指出其中两种将所有节点嵌入同一特征向量中，而另外两种模型生成的表示与输入图中的步长数量相关。在一定条件下，不同结构的节点可能有相似的表示。

    

    最近几年，图神经网络（GNNs）在图嵌入学习领域取得了巨大成功。尽管以往的研究揭示了这些模型的表达能力（即它们是否能区分非同构图对），但仍不清楚这些模型所学习的节点表示中编码了哪些结构信息。本文研究了四种流行的GNN模型，并展示了其中两种将所有节点嵌入同一特征向量中，而另外两种模型生成的表示与输入图中的步长数量相关。令人惊讶的是，如果两个不同结构的节点在某一层$k>1$ 中的步长相同，则它们的表示可能相似。我们在真实数据集上进行了实证验证，从而验证了我们的理论发现。

    In recent years, graph neural networks (GNNs) have achieved great success in the field of graph representation learning. Although prior work has shed light into the expressiveness of those models (\ie whether they can distinguish pairs of non-isomorphic graphs), it is still not clear what structural information is encoded into the node representations that are learned by those models. In this paper, we investigate which properties of graphs are captured purely by these models, when no node attributes are available. Specifically, we study four popular GNN models, and we show that two of them embed all nodes into the same feature vector, while the other two models generate representations that are related to the number of walks over the input graph. Strikingly, structurally dissimilar nodes can have similar representations at some layer $k>1$, if they have the same number of walks of length $k$. We empirically verify our theoretical findings on real datasets.
    
[^12]: Huber能量量化

    Huber-energy measure quantization. (arXiv:2212.08162v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2212.08162](http://arxiv.org/abs/2212.08162)

    该论文提出了一种Huber能量量化的算法，用于找到目标概率定律的最佳逼近，通过最小化原测度与量化版本之间的统计距离来实现。该算法已在多维高斯混合物、维纳空间魔方等几个数据库上进行了测试。

    

    我们描述了一种测量量化过程，即一种算法，它通过$Q$个狄拉克函数的总和（$Q$为量化参数），找到目标概率定律（更一般地，为有限变差测度）的最佳逼近。该过程通过将原测度与其量化版本之间的统计距离最小化来实现；该距离基于负定核构建，并且如果必要，可以实时计算并输入随机优化算法（如SGD，Adam等）。我们在理论上研究了最优测量量化器的存在的基本问题，并确定了需要保证合适行为的核属性。我们提出了两个最佳线性无偏（BLUE）估计器，用于平方统计距离，并将它们用于无偏程序HEMQ中，以找到最佳量化。我们在多维高斯混合物、维纳空间魔方等几个数据库上测试了HEMQ

    We describe a measure quantization procedure i.e., an algorithm which finds the best approximation of a target probability law (and more generally signed finite variation measure) by a sum of $Q$ Dirac masses ($Q$ being the quantization parameter). The procedure is implemented by minimizing the statistical distance between the original measure and its quantized version; the distance is built from a negative definite kernel and, if necessary, can be computed on the fly and feed to a stochastic optimization algorithm (such as SGD, Adam, ...). We investigate theoretically the fundamental questions of existence of the optimal measure quantizer and identify what are the required kernel properties that guarantee suitable behavior. We propose two best linear unbiased (BLUE) estimators for the squared statistical distance and use them in an unbiased procedure, called HEMQ, to find the optimal quantization. We test HEMQ on several databases: multi-dimensional Gaussian mixtures, Wiener space cub
    
[^13]: 连续生成神经网络

    Continuous Generative Neural Networks. (arXiv:2205.14627v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2205.14627](http://arxiv.org/abs/2205.14627)

    本文介绍了一种连续生成神经网络(CGNN)的模型，使用条件保证CGNN是单射的，其生成流形被用于求解反问题，并证明了其方法的有效性和稳健性。

    

    本文介绍了并研究了一种连续生成神经网络（CGNN），即连续情境下的生成模型：CGNN的输出属于无限维函数空间。该架构受DCGAN的启发，采用一个全连接层，多个卷积层和非线性激活函数。在连续的$L^2$情境下，每层空间的维度被紧支小波的多重分辨率分析的尺度所代替。我们提出了关于卷积滤波器和非线性的条件，保证CGNN是单射的。该理论应用于反问题，并允许导出一个CGNN生成流形的（可能非线性的）无限维反问题的Lipschitz稳定性估计。包括信号去模糊在内的多个数值模拟证明并验证了这一方法。

    In this work, we present and study Continuous Generative Neural Networks (CGNNs), namely, generative models in the continuous setting: the output of a CGNN belongs to an infinite-dimensional function space. The architecture is inspired by DCGAN, with one fully connected layer, several convolutional layers and nonlinear activation functions. In the continuous $L^2$ setting, the dimensions of the spaces of each layer are replaced by the scales of a multiresolution analysis of a compactly supported wavelet. We present conditions on the convolutional filters and on the nonlinearity that guarantee that a CGNN is injective. This theory finds applications to inverse problems, and allows for deriving Lipschitz stability estimates for (possibly nonlinear) infinite-dimensional inverse problems with unknowns belonging to the manifold generated by a CGNN. Several numerical simulations, including signal deblurring, illustrate and validate this approach.
    

