# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Juru: Legal Brazilian Large Language Model from Reputable Sources](https://arxiv.org/abs/2403.18140) | Juru 模型通过从巴西法律来源提取的19亿个唯一标记，展示了领域专门化可以在减少预训练数据量方面发挥作用，但这种专门化会导致同一语言中其他知识领域性能下降。 |
| [^2] | [GLC++: Source-Free Universal Domain Adaptation through Global-Local Clustering and Contrastive Affinity Learning](https://arxiv.org/abs/2403.14410) | 该论文提出了GLC++方法，通过全局和局部聚类以及对比关联学习实现了无源通用域自适应，能够准确分类已知数据并将其从未知数据中分离。 |
| [^3] | [The Effect of Data Poisoning on Counterfactual Explanations](https://arxiv.org/abs/2402.08290) | 本研究研究了反事实解释在数据污染方面的脆弱性，发现最先进的反事实生成方法和工具包容易受到数据污染的影响。 |
| [^4] | [Large-scale Generative AI Models Lack Visual Number Sense](https://arxiv.org/abs/2402.03328) | 本研究调查了基于大规模Transformer架构的生成性AI模型是否能够准确命名物体数量或生成包含目标数量物品的图像，结果发现这些模型都没有以类似人类的方式表现，并且即使对于小数量的物体也会出现显著的错误。 |
| [^5] | [Uncertainty-Aware Testing-Time Optimization for 3D Human Pose Estimation](https://arxiv.org/abs/2402.02339) | 本文提出了一种不确定性感知的测试时间优化（UAO）框架，通过量化关节点的不确定性来缓解过拟合问题，提高3D人体姿势估计的性能。 |
| [^6] | [ShaRP: Explaining Rankings with Shapley Values.](http://arxiv.org/abs/2401.16744) | ShaRP是一个基于Shapley值的框架，用于解释排名结果中各个特征的贡献。即使使用线性评分函数，特征的权重也不一定对应其Shapley值的贡献，而是取决于特征分布和评分特征之间的局部相互作用。 |
| [^7] | [Contrastive learning-based agent modeling for deep reinforcement learning.](http://arxiv.org/abs/2401.00132) | 本研究提出了一种基于对比学习的深度强化学习代理建模方法，该方法可以在仅利用自我代理的本地观测的情况下，提取其他代理的有意义策略表示，以改进自我代理的自适应策略。 |
| [^8] | [Algebras of actions in an agent's representations of the world.](http://arxiv.org/abs/2310.01536) | 本文提出了一个框架，用于从代理的视角提取世界转换的代数，并研究了在简单强化学习场景中出现的世界转换的代数。 |
| [^9] | [Geometric structure of shallow neural networks and constructive ${\mathcal L}^2$ cost minimization.](http://arxiv.org/abs/2309.10370) | 本文提供了浅层神经网络的几何结构解释，并通过基于${\mathcal L}^2$代价最小化的构造方法获得了一个具有优越性能的网络。 |
| [^10] | [Benchmarking and Analyzing Generative Data for Visual Recognition.](http://arxiv.org/abs/2307.13697) | 这篇论文通过实验研究了利用生成数据在视觉识别中的应用，并提出了一个用于评估生成数据的基准，一个训练-free的度量指标以及与检索数据进行比较揭示生成数据的独特特征的新方法。 |
| [^11] | [An automated end-to-end deep learning-based framework for lung cancer diagnosis by detecting and classifying the lung nodules.](http://arxiv.org/abs/2305.00046) | 本文提出了一种基于深度学习的智能诊断框架，针对低资源环境实现早期检测和分类肺部结节，并在公共数据集上取得了较好的表现。 |

# 详细

[^1]: Juru: 来自可靠来源的巴西法律大语言模型

    Juru: Legal Brazilian Large Language Model from Reputable Sources

    [https://arxiv.org/abs/2403.18140](https://arxiv.org/abs/2403.18140)

    Juru 模型通过从巴西法律来源提取的19亿个唯一标记，展示了领域专门化可以在减少预训练数据量方面发挥作用，但这种专门化会导致同一语言中其他知识领域性能下降。

    

    与预训练大型语言模型相关的高计算成本限制了相关研究。为解决这一问题，出现了两种策略：领域专门化和使用高质量数据进行预训练。为探索这些策略，我们使用来自可靠巴西法律来源的19亿个唯一标记专门化了Sabi\'a-2 Small模型，并在法律和一般知识考试中进行了少样本评估。我们的模型Juru展示了领域专门化在减少预训练数据量方面的优势。然而，这种专门化是以在同一语言中其他知识领域性能下降为代价的。这项研究有助于增加的科学证据，表明预训练数据的选择可能提高大型语言模型的性能，从而能够以较低成本探索这些模型。

    arXiv:2403.18140v1 Announce Type: cross  Abstract: The high computational cost associated with pretraining large language models limits their research. Two strategies have emerged to address this issue: domain specialization and pretraining with high-quality data. To explore these strategies, we specialized the Sabi\'a-2 Small model with 1.9 billion unique tokens from reputable Brazilian legal sources and conducted few-shot evaluations on legal and general knowledge exams. Our model, Juru, demonstrates the benefits of domain specialization with a reduced amount of pretraining data. However, this specialization comes at the expense of degrading performance in other knowledge areas within the same language. This study contributes to the growing body of scientific evidence showing that pretraining data selection may enhance the performance of large language models, enabling the exploration of these models at a lower cost.
    
[^2]: GLC++: 全局局部聚类和对比关联学习的无源通用域自适应

    GLC++: Source-Free Universal Domain Adaptation through Global-Local Clustering and Contrastive Affinity Learning

    [https://arxiv.org/abs/2403.14410](https://arxiv.org/abs/2403.14410)

    该论文提出了GLC++方法，通过全局和局部聚类以及对比关联学习实现了无源通用域自适应，能够准确分类已知数据并将其从未知数据中分离。

    

    深度神经网络经常在协变量和类别转移下表现出次优性能。无源域自适应（SFDA）为这一困境提供了一个有希望的解决方案，然而大多数SFDA方法局限于封闭集场景。在本文中，我们探讨了旨在准确分类属于常见类别的“已知”数据并将其与目标专有“未知”数据隔离开来的无源通用域自适应（SF-UniDA）。我们提出了一种新颖的全球和局部聚类（GLC）技术，其中包括自适应的一对全局聚类算法来区分目标类别，辅以本地k-NN聚类策略以减轻负面转移。尽管有效，但固有的封闭源架构导致对“未知”数据的统一处理，阻碍了对不同“未知”类别的识别。为了解决这个问题，我们将GLC发展到GLC++，整合了对比亲和性。

    arXiv:2403.14410v1 Announce Type: cross  Abstract: Deep neural networks often exhibit sub-optimal performance under covariate and category shifts. Source-Free Domain Adaptation (SFDA) presents a promising solution to this dilemma, yet most SFDA approaches are restricted to closed-set scenarios. In this paper, we explore Source-Free Universal Domain Adaptation (SF-UniDA) aiming to accurately classify "known" data belonging to common categories and segregate them from target-private "unknown" data. We propose a novel Global and Local Clustering (GLC) technique, which comprises an adaptive one-vs-all global clustering algorithm to discern between target classes, complemented by a local k-NN clustering strategy to mitigate negative transfer. Despite the effectiveness, the inherent closed-set source architecture leads to uniform treatment of "unknown" data, impeding the identification of distinct "unknown" categories. To address this, we evolve GLC to GLC++, integrating a contrastive affini
    
[^3]: 数据污染对反事实解释的影响

    The Effect of Data Poisoning on Counterfactual Explanations

    [https://arxiv.org/abs/2402.08290](https://arxiv.org/abs/2402.08290)

    本研究研究了反事实解释在数据污染方面的脆弱性，发现最先进的反事实生成方法和工具包容易受到数据污染的影响。

    

    反事实解释是分析黑盒系统预测的一种流行方法，它们提供了根据不同情况建议改变输入以获得不同（更有利）系统输出的计算补救机会。然而，最近的研究突显了它们对不同类型操纵的脆弱性。本研究研究了反事实解释对数据污染的脆弱性。我们在增加三个不同层次的补救成本方面，形式化地研究了反事实解释在单个实例、某个子组或所有实例上的数据污染。我们证明了最先进的反事实生成方法和工具包对此类数据污染是脆弱的。

    Counterfactual explanations provide a popular method for analyzing the predictions of black-box systems, and they can offer the opportunity for computational recourse by suggesting actionable changes on how to change the input to obtain a different (i.e. more favorable) system output. However, recent work highlighted their vulnerability to different types of manipulations. This work studies the vulnerability of counterfactual explanations to data poisoning. We formalize data poisoning in the context of counterfactual explanations for increasing the cost of recourse on three different levels: locally for a single instance, or a sub-group of instances, or globally for all instances. We demonstrate that state-of-the-art counterfactual generation methods \& toolboxes are vulnerable to such data poisoning.
    
[^4]: 大规模生成AI模型缺乏视觉数字感知能力

    Large-scale Generative AI Models Lack Visual Number Sense

    [https://arxiv.org/abs/2402.03328](https://arxiv.org/abs/2402.03328)

    本研究调查了基于大规模Transformer架构的生成性AI模型是否能够准确命名物体数量或生成包含目标数量物品的图像，结果发现这些模型都没有以类似人类的方式表现，并且即使对于小数量的物体也会出现显著的错误。

    

    人类能够在视觉场景中轻松判断物体的数量，即使不进行计数，而且这种技能在各种动物物种和语言发展和正式学校教育之前的婴儿中都有记录。对于小的物体集，数字判断是无误的，而对于更大的集合，回应变得近似，并且变异性与目标数字成比例增加。尽管物体特征（如颜色或形状）存在差异，但这种回应模式在所有类型的物体上观察到，这表明我们的视觉数字感知依赖于数字数量的抽象表示。在本研究中，我们调查了基于大规模Transformer架构的生成性人工智能（AI）模型是否可以可靠地命名简单视觉刺激中的物体数量或生成包含目标物品数量的图像（1-10范围内）。令人惊讶的是，所考虑的所有基础模型都没有以类似人类一样的方式表现出来：即使是具有较小数量的物体也会犯下显著的错误。

    Humans can readily judge the number of objects in a visual scene, even without counting, and such a skill has been documented in a variety of animal species and in babies prior to language development and formal schooling. Numerical judgments are error-free for small sets, while for larger collections responses become approximate, with variability increasing proportionally to the target number. This response pattern is observed for items of all kinds, despite variation in object features (such as color or shape), suggesting that our visual number sense relies on abstract representations of numerosity. Here, we investigated whether generative Artificial Intelligence (AI) models based on large-scale transformer architectures can reliably name the number of objects in simple visual stimuli or generate images containing a target number of items in the 1-10 range. Surprisingly, none of the foundation models considered performed in a human-like way: They all made striking errors even with sm
    
[^5]: 不确定性感知的3D人体姿势估计测试时间优化

    Uncertainty-Aware Testing-Time Optimization for 3D Human Pose Estimation

    [https://arxiv.org/abs/2402.02339](https://arxiv.org/abs/2402.02339)

    本文提出了一种不确定性感知的测试时间优化（UAO）框架，通过量化关节点的不确定性来缓解过拟合问题，提高3D人体姿势估计的性能。

    

    尽管数据驱动方法在3D人体姿势估计方面取得了成功，但它们常常受到域间差异的限制，表现出有限的泛化能力。相比之下，基于优化的方法在特定情况下进行微调方面表现优秀，但整体表现通常不如数据驱动方法。我们观察到先前的基于优化的方法通常依赖于投影约束，这仅仅确保了在2D空间中的对齐，可能导致过拟合问题。为了解决这个问题，我们提出了一种不确定性感知的测试时间优化 (UAO) 框架，它保留了预训练模型的先验信息，并利用关节点的不确定性来缓解过拟合问题。具体而言，在训练阶段，我们设计了一个有效的2D到3D网络，用于估计相应的3D姿势，并量化每个3D关节点的不确定性。对于测试时的优化，所提出的优化框架冻结预训练模型，并仅优化少量关键参数，以提高性能。

    Although data-driven methods have achieved success in 3D human pose estimation, they often suffer from domain gaps and exhibit limited generalization. In contrast, optimization-based methods excel in fine-tuning for specific cases but are generally inferior to data-driven methods in overall performance. We observe that previous optimization-based methods commonly rely on projection constraint, which only ensures alignment in 2D space, potentially leading to the overfitting problem. To address this, we propose an Uncertainty-Aware testing-time Optimization (UAO) framework, which keeps the prior information of pre-trained model and alleviates the overfitting problem using the uncertainty of joints. Specifically, during the training phase, we design an effective 2D-to-3D network for estimating the corresponding 3D pose while quantifying the uncertainty of each 3D joint. For optimization during testing, the proposed optimization framework freezes the pre-trained model and optimizes only a 
    
[^6]: ShaRP：用Shapley值解释排名

    ShaRP: Explaining Rankings with Shapley Values. (arXiv:2401.16744v1 [cs.AI])

    [http://arxiv.org/abs/2401.16744](http://arxiv.org/abs/2401.16744)

    ShaRP是一个基于Shapley值的框架，用于解释排名结果中各个特征的贡献。即使使用线性评分函数，特征的权重也不一定对应其Shapley值的贡献，而是取决于特征分布和评分特征之间的局部相互作用。

    

    在招聘、大学招生和贷款等重要领域的算法决策常常是基于排名的。由于这些决策对个人、组织和人群的影响，有必要了解它们：了解决策是否遵守法律，帮助个人提高他们的排名，并设计更好的排名程序。本文提出了ShaRP（Shapley for Rankings and Preferences），这是一个基于Shapley值的框架，用于解释特征对排名结果不同方面的贡献。使用ShaRP，我们展示了即使算法排名器使用的评分函数是已知的且是线性的，每个特征的权重也不一定对应其Shapley值的贡献。贡献取决于特征的分布以及评分特征之间微妙的局部相互作用。ShaRP基于量化输入影响框架，并可以计算贡献。

    Algorithmic decisions in critical domains such as hiring, college admissions, and lending are often based on rankings. Because of the impact these decisions have on individuals, organizations, and population groups, there is a need to understand them: to know whether the decisions are abiding by the law, to help individuals improve their rankings, and to design better ranking procedures.  In this paper, we present ShaRP (Shapley for Rankings and Preferences), a framework that explains the contributions of features to different aspects of a ranked outcome, and is based on Shapley values. Using ShaRP, we show that even when the scoring function used by an algorithmic ranker is known and linear, the weight of each feature does not correspond to its Shapley value contribution. The contributions instead depend on the feature distributions, and on the subtle local interactions between the scoring features. ShaRP builds on the Quantitative Input Influence framework, and can compute the contri
    
[^7]: 基于对比学习的深度强化学习代理建模

    Contrastive learning-based agent modeling for deep reinforcement learning. (arXiv:2401.00132v2 [cs.MA] UPDATED)

    [http://arxiv.org/abs/2401.00132](http://arxiv.org/abs/2401.00132)

    本研究提出了一种基于对比学习的深度强化学习代理建模方法，该方法可以在仅利用自我代理的本地观测的情况下，提取其他代理的有意义策略表示，以改进自我代理的自适应策略。

    

    多智能体系统经常需要代理与具有不同目标、行为或策略的其他代理合作或竞争。在多智能体系统中设计自适应策略时，代理建模是必不可少的，因为这是自我代理理解其他代理行为并提取有意义的策略表示的方式。这些表示可以用来增强自我代理的自适应策略，该策略通过强化学习进行训练。然而，现有的代理建模方法通常假设在训练或长时间观察轨迹的策略适应过程中可以使用来自其他代理（建模代理）的本地观测。为了消除这些限制性假设并提高代理建模性能，我们设计了一种基于对比学习的代理建模（CLAM）方法，该方法仅依赖于自我代理在训练和执行过程中的本地观测。

    Multi-agent systems often require agents to collaborate with or compete against other agents with diverse goals, behaviors, or strategies. Agent modeling is essential when designing adaptive policies for intelligent machine agents in multiagent systems, as this is the means by which the ego agent understands other agents' behavior and extracts their meaningful policy representations. These representations can be used to enhance the ego agent's adaptive policy which is trained by reinforcement learning. However, existing agent modeling approaches typically assume the availability of local observations from other agents (modeled agents) during training or a long observation trajectory for policy adaption. To remove these constrictive assumptions and improve agent modeling performance, we devised a Contrastive Learning-based Agent Modeling (CLAM) method that relies only on the local observations from the ego agent during training and execution. With these observations, CLAM is capable of 
    
[^8]: 一个代理在世界表示中行动的代数

    Algebras of actions in an agent's representations of the world. (arXiv:2310.01536v1 [cs.AI])

    [http://arxiv.org/abs/2310.01536](http://arxiv.org/abs/2310.01536)

    本文提出了一个框架，用于从代理的视角提取世界转换的代数，并研究了在简单强化学习场景中出现的世界转换的代数。

    

    本文提出了一个框架，从一个代理的视角提取世界转换的代数。首先，我们使用我们的框架从对称性分解表示学习(SBDRL)的角度复现了对称性基础表示的工作[1]，只有形成群的世界转换代数才能用对称性基础表示描述。然后，我们研究在简单强化学习场景中出现的具有特征的世界转换的代数。我们使用我们开发的计算方法提取了这些世界转换的代数，并根据它们的属性进行分类。最后，我们将SBDRL的两个重要结果 - 等变条件和分离定义 - 从仅适用于对称性基础表示扩展到适用于捕捉世界转换特性的表示。

    In this paper, we propose a framework to extract the algebra of the transformations of worlds from the perspective of an agent. As a starting point, we use our framework to reproduce the symmetry-based representations from the symmetry-based disentangled representation learning (SBDRL) formalism proposed by [1]; only the algebra of transformations of worlds that form groups can be described using symmetry-based representations. We then study the algebras of the transformations of worlds with features that occur in simple reinforcement learning scenarios. Using computational methods, that we developed, we extract the algebras of the transformations of these worlds and classify them according to their properties. Finally, we generalise two important results of SBDRL - the equivariance condition and the disentangling definition - from only working with symmetry-based representations to working with representations capturing the transformation properties of worlds with transformations for 
    
[^9]: 浅层神经网络的几何结构和基于${\mathcal L}^2$代价最小化的构造方法

    Geometric structure of shallow neural networks and constructive ${\mathcal L}^2$ cost minimization. (arXiv:2309.10370v1 [cs.LG])

    [http://arxiv.org/abs/2309.10370](http://arxiv.org/abs/2309.10370)

    本文提供了浅层神经网络的几何结构解释，并通过基于${\mathcal L}^2$代价最小化的构造方法获得了一个具有优越性能的网络。

    

    本文给出了一个几何解释：浅层神经网络的结构由一个隐藏层、一个斜坡激活函数、一个${\mathcal L}^2$谱范类（或者Hilbert-Schmidt）的代价函数、输入空间${\mathbb R}^M$、输出空间${\mathbb R}^Q$（其中$Q\leq M$），以及训练输入样本数量$N>QM$所特征。我们证明了代价函数的最小值具有$O(\delta_P)$的上界，其中$\delta_P$衡量了训练输入的信噪比。我们使用适应于属于同一输出向量$y_j$的训练输入向量$\overline{x_{0,j}}$的投影来获得近似的优化器，其中$j=1,\dots,Q$。在特殊情况$M=Q$下，我们明确确定了代价函数的一个确切退化局部最小值；这个尖锐的值与对于$Q\leq M$所获得的上界之间有一个相对误差$O(\delta_P^2)$。上界证明的方法提供了一个构造性训练的网络；我们证明它测度了$Q$维空间中的给定输出。

    In this paper, we provide a geometric interpretation of the structure of shallow neural networks characterized by one hidden layer, a ramp activation function, an ${\mathcal L}^2$ Schatten class (or Hilbert-Schmidt) cost function, input space ${\mathbb R}^M$, output space ${\mathbb R}^Q$ with $Q\leq M$, and training input sample size $N>QM$. We prove an upper bound on the minimum of the cost function of order $O(\delta_P$ where $\delta_P$ measures the signal to noise ratio of training inputs. We obtain an approximate optimizer using projections adapted to the averages $\overline{x_{0,j}}$ of training input vectors belonging to the same output vector $y_j$, $j=1,\dots,Q$. In the special case $M=Q$, we explicitly determine an exact degenerate local minimum of the cost function; the sharp value differs from the upper bound obtained for $Q\leq M$ by a relative error $O(\delta_P^2)$. The proof of the upper bound yields a constructively trained network; we show that it metrizes the $Q$-dimen
    
[^10]: 基于视觉识别的生成数据的基准测试与分析

    Benchmarking and Analyzing Generative Data for Visual Recognition. (arXiv:2307.13697v1 [cs.CV])

    [http://arxiv.org/abs/2307.13697](http://arxiv.org/abs/2307.13697)

    这篇论文通过实验研究了利用生成数据在视觉识别中的应用，并提出了一个用于评估生成数据的基准，一个训练-free的度量指标以及与检索数据进行比较揭示生成数据的独特特征的新方法。

    

    大型预训练生成模型的进步扩大了它们作为有效数据生成器在视觉识别中的潜力。本研究深入探讨了生成图像的影响，主要比较了利用外部数据（如生成数据、检索数据、原始数据）的范例。我们的主要贡献包括：1) GenBench构建：我们设计了GenBench，一个包含22个数据集和2548个类别的广泛基准，用于评估不同视觉识别任务中的生成数据。2) CLER分数：为了解决现有度量指标（如FID、CLIP分数）与下游识别性能之间的不足相关性问题，我们提出了CLER，一种无需训练的度量指标，用于指示识别任务训练之前生成数据的效率。3) 新的基准线：将生成数据与来自相同外部池的检索数据进行比较，有助于阐明生成数据的独特特征。4) 外部知识注入：通过注入外部知识，提高生成数据在视觉识别任务中的性能。

    Advancements in large pre-trained generative models have expanded their potential as effective data generators in visual recognition. This work delves into the impact of generative images, primarily comparing paradigms that harness external data (\ie generative \vs retrieval \vs original).  Our key contributions are: \textbf{1) GenBench Construction:} We devise \textbf{GenBench}, a broad benchmark comprising 22 datasets with 2548 categories, to appraise generative data across various visual recognition tasks. \textbf{2) CLER Score:} To address the insufficient correlation of existing metrics (\eg, FID, CLIP score) with downstream recognition performance, we propose \textbf{CLER}, a training-free metric indicating generative data's efficiency for recognition tasks prior to training. \textbf{3) New Baselines:} Comparisons of generative data with retrieved data from the same external pool help to elucidate the unique traits of generative data. \textbf{4) External Knowledge Injection:} By 
    
[^11]: 一种基于深度学习技术的肺癌诊断自动化端到端框架，用于检测和分类肺部结节

    An automated end-to-end deep learning-based framework for lung cancer diagnosis by detecting and classifying the lung nodules. (arXiv:2305.00046v1 [eess.IV])

    [http://arxiv.org/abs/2305.00046](http://arxiv.org/abs/2305.00046)

    本文提出了一种基于深度学习的智能诊断框架，针对低资源环境实现早期检测和分类肺部结节，并在公共数据集上取得了较好的表现。

    

    肺癌是全球癌症相关死亡的主要原因，在低资源环境中早期诊断对于改善患者疗效至关重要。本研究的目的是提出一种基于深度学习技术的自动化端到端框架，用于早期检测和分类肺部结节，特别是针对低资源环境。该框架由三个阶段组成：使用改进的3D Res-U-Net进行肺分割、使用YOLO-v5进行结节检测、使用基于Vision Transformer的架构进行分类。我们在开放的数据集LUNA16上对该框架进行了评估。所提出的框架的性能是使用各领域的评估指标进行衡量的。该框架在肺部分割dice系数上达到了98.82％，同时检测肺结节的平均准确度为0.76 mAP。

    Lung cancer is a leading cause of cancer-related deaths worldwide, and early detection is crucial for improving patient outcomes. Nevertheless, early diagnosis of cancer is a major challenge, particularly in low-resource settings where access to medical resources and trained radiologists is limited. The objective of this study is to propose an automated end-to-end deep learning-based framework for the early detection and classification of lung nodules, specifically for low-resource settings. The proposed framework consists of three stages: lung segmentation using a modified 3D U-Net named 3D Res-U-Net, nodule detection using YOLO-v5, and classification with a Vision Transformer-based architecture. We evaluated the proposed framework on a publicly available dataset, LUNA16. The proposed framework's performance was measured using the respective domain's evaluation matrices. The proposed framework achieved a 98.82% lung segmentation dice score while detecting the lung nodule with 0.76 mAP
    

