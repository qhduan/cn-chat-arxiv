# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Benchmarking Counterfactual Image Generation](https://arxiv.org/abs/2403.20287) | 提出了一个针对对照图像生成方法的基准测试框架，包含评估对照多个方面的度量标准以及评估三种不同类型的条件图像生成模型性能。 |
| [^2] | [Scalable Spatiotemporal Prediction with Bayesian Neural Fields](https://arxiv.org/abs/2403.07657) | 该论文提出了贝叶斯神经场（BayesNF），结合了深度神经网络和分层贝叶斯推断，用于处理大规模时空预测问题。 |
| [^3] | [Bayesian Hierarchical Probabilistic Forecasting of Intraday Electricity Prices](https://arxiv.org/abs/2403.05441) | 该研究首次提出了为德国连续日内市场交易的电力价格进行贝叶斯预测，考虑了参数不确定性，并在2022年的电力价格验证中取得了统计显著的改进。 |
| [^4] | [CaT-GNN: Enhancing Credit Card Fraud Detection via Causal Temporal Graph Neural Networks](https://arxiv.org/abs/2402.14708) | 该论文提出了一种名为CaT-GNN的新型信用卡欺诈检测方法，通过因果不变性学习揭示交易数据中的固有相关性，并引入因果混合策略来增强模型的鲁棒性和可解释性。 |
| [^5] | [MENTOR: Guiding Hierarchical Reinforcement Learning with Human Feedback and Dynamic Distance Constraint](https://arxiv.org/abs/2402.14244) | 使用人类反馈和动态距离约束对层次化强化学习进行引导，解决了找到适当子目标的问题，并设计了双策略以稳定训练。 |
| [^6] | [Multiscale Hodge Scattering Networks for Data Analysis](https://arxiv.org/abs/2311.10270) | 提出了多尺度霍奇散射网络（MHSNs），利用多尺度基础词典和卷积结构，生成对节点排列不变的特征。 |
| [^7] | [cedar: Composable and Optimized Machine Learning Input Data Pipelines.](http://arxiv.org/abs/2401.08895) | cedar是一个编程模型和框架，可以轻松构建、优化和执行机器学习输入数据管道。它提供了易于使用的编程接口和可组合运算符，支持任意ML框架和库。通过解决当前输入数据系统无法充分利用性能优化的问题，cedar提高了资源利用效率，满足了庞大数据量和高训练吞吐量的需求。 |
| [^8] | [CSG: Curriculum Representation Learning for Signed Graph.](http://arxiv.org/abs/2310.11083) | 本文提出了一种用于有符号图的课程表示学习框架（CSG），通过引入课程化训练方法和轻量级机制，实现了按难易程度优化样本展示顺序，从而提高有符号图神经网络（SGNN）模型的准确性和稳定性。 |
| [^9] | [Simplifying GNN Performance with Low Rank Kernel Models.](http://arxiv.org/abs/2310.05250) | 本文提出了一种用于简化GNN性能的低秩内核模型，通过应用传统的非参数估计方法在谱域中取代过于复杂的GNN架构，并在多个图类型的半监督节点分类基准测试中取得了最先进的性能。 |
| [^10] | [A 3D deep learning classifier and its explainability when assessing coronary artery disease.](http://arxiv.org/abs/2308.00009) | 本文提出了一种3D深度学习模型，用于直接分类冠心病患者和正常受试者，相较于2D模型提高了23.65%的性能，并通过Grad-GAM提供了可解释性。此外，通过与2D语义分割相结合，实现了更好的解释性和准确的异常定位。 |
| [^11] | [Atlas-Based Interpretable Age Prediction.](http://arxiv.org/abs/2307.07439) | 本研究提出了一种基于图谱的可解释年龄预测方法，利用全身图像研究了各个身体部位的年龄相关变化。通过使用解释性方法和配准技术，确定了最能预测年龄的身体区域，并创下了整个身体年龄预测的最新水平。研究结果表明，脊柱、本原性背部肌肉和心脏区域是最重要的关注领域。 |
| [^12] | [Referential communication in heterogeneous communities of pre-trained visual deep networks.](http://arxiv.org/abs/2302.08913) | 异构视觉深度网络社区中的预训练网络可以自我监督地开发出共享协议，以指代一组目标中的目标对象，并可用于沟通不同粒度的未知对象类别。 |

# 详细

[^1]: 基准对照图像生成

    Benchmarking Counterfactual Image Generation

    [https://arxiv.org/abs/2403.20287](https://arxiv.org/abs/2403.20287)

    提出了一个针对对照图像生成方法的基准测试框架，包含评估对照多个方面的度量标准以及评估三种不同类型的条件图像生成模型性能。

    

    对照图像生成在理解变量因果关系方面具有关键作用，在解释性和生成无偏合成数据方面有应用。然而，评估图像生成本身就是一个长期存在的挑战。对于评估对照生成的需求进一步加剧了这一挑战，因为根据定义，对照情景是没有可观测基准事实的假设情况。本文提出了一个旨在对照图像生成方法进行基准测试的新颖综合框架。我们结合了侧重于评估对照的不同方面的度量标准，例如组成、有效性、干预的最小性和图像逼真度。我们评估了基于结构因果模型范式的三种不同条件图像生成模型类型的性能。我们的工作还配备了一个用户友好的Python软件包，可以进一步评估。

    arXiv:2403.20287v1 Announce Type: cross  Abstract: Counterfactual image generation is pivotal for understanding the causal relations of variables, with applications in interpretability and generation of unbiased synthetic data. However, evaluating image generation is a long-standing challenge in itself. The need to evaluate counterfactual generation compounds on this challenge, precisely because counterfactuals, by definition, are hypothetical scenarios without observable ground truths. In this paper, we present a novel comprehensive framework aimed at benchmarking counterfactual image generation methods. We incorporate metrics that focus on evaluating diverse aspects of counterfactuals, such as composition, effectiveness, minimality of interventions, and image realism. We assess the performance of three distinct conditional image generation model types, based on the Structural Causal Model paradigm. Our work is accompanied by a user-friendly Python package which allows to further eval
    
[^2]: 使用贝叶斯神经场进行可扩展的时空预测

    Scalable Spatiotemporal Prediction with Bayesian Neural Fields

    [https://arxiv.org/abs/2403.07657](https://arxiv.org/abs/2403.07657)

    该论文提出了贝叶斯神经场（BayesNF），结合了深度神经网络和分层贝叶斯推断，用于处理大规模时空预测问题。

    

    时空数据集由空间参考的时间序列表示，广泛应用于许多科学和商业智能领域，例如空气污染监测，疾病跟踪和云需求预测。随着现代数据集规模和复杂性的不断增加，需要新的统计方法来捕捉复杂的时空动态并处理大规模预测问题。本研究介绍了Bayesian Neural Field (BayesNF)，这是一个用于推断时空域上丰富概率分布的通用领域统计模型，可用于包括预测、插值和变异分析在内的数据分析任务。BayesNF将用于高容量函数估计的新型深度神经网络架构与用于鲁棒不确定性量化的分层贝叶斯推断相结合。通过在定义先验分布方面进行序列化

    arXiv:2403.07657v1 Announce Type: cross  Abstract: Spatiotemporal datasets, which consist of spatially-referenced time series, are ubiquitous in many scientific and business-intelligence applications, such as air pollution monitoring, disease tracking, and cloud-demand forecasting. As modern datasets continue to increase in size and complexity, there is a growing need for new statistical methods that are flexible enough to capture complex spatiotemporal dynamics and scalable enough to handle large prediction problems. This work presents the Bayesian Neural Field (BayesNF), a domain-general statistical model for inferring rich probability distributions over a spatiotemporal domain, which can be used for data-analysis tasks including forecasting, interpolation, and variography. BayesNF integrates a novel deep neural network architecture for high-capacity function estimation with hierarchical Bayesian inference for robust uncertainty quantification. By defining the prior through a sequenc
    
[^3]: 基于贝叶斯层次概率的日内电力价格预测

    Bayesian Hierarchical Probabilistic Forecasting of Intraday Electricity Prices

    [https://arxiv.org/abs/2403.05441](https://arxiv.org/abs/2403.05441)

    该研究首次提出了为德国连续日内市场交易的电力价格进行贝叶斯预测，考虑了参数不确定性，并在2022年的电力价格验证中取得了统计显著的改进。

    

    我们首次提出了对德国连续日内市场交易的电力价格进行贝叶斯预测的研究，充分考虑参数不确定性。我们的目标变量是IDFull价格指数，预测以后验预测分布的形式给出。我们使用了2022年极度波动的电力价格进行验证，在之前几乎没有成为预测研究对象。作为基准模型，我们使用了预测创建时的所有可用日内交易来计算IDFull的当前值。根据弱式有效假设，从最后价格信息建立的基准无法显著改善。然而，我们观察到在点度量和概率评分方面存在着统计显著的改进。最后，我们挑战了在电力价格预测中使用LASSO进行特征选择的宣布的黄金标准。

    arXiv:2403.05441v1 Announce Type: cross  Abstract: We present a first study of Bayesian forecasting of electricity prices traded on the German continuous intraday market which fully incorporates parameter uncertainty. Our target variable is the IDFull price index, forecasts are given in terms of posterior predictive distributions. For validation we use the exceedingly volatile electricity prices of 2022, which have hardly been the subject of forecasting studies before. As a benchmark model, we use all available intraday transactions at the time of forecast creation to compute a current value for the IDFull. According to the weak-form efficiency hypothesis, it would not be possible to significantly improve this benchmark built from last price information. We do, however, observe statistically significant improvement in terms of both point measures and probability scores. Finally, we challenge the declared gold standard of using LASSO for feature selection in electricity price forecastin
    
[^4]: 通过因果时间图神经网络增强信用卡欺诈检测

    CaT-GNN: Enhancing Credit Card Fraud Detection via Causal Temporal Graph Neural Networks

    [https://arxiv.org/abs/2402.14708](https://arxiv.org/abs/2402.14708)

    该论文提出了一种名为CaT-GNN的新型信用卡欺诈检测方法，通过因果不变性学习揭示交易数据中的固有相关性，并引入因果混合策略来增强模型的鲁棒性和可解释性。

    

    信用卡欺诈对经济构成重大威胁。尽管基于图神经网络（GNN）的欺诈检测方法表现良好，但它们经常忽视节点的本地结构对预测的因果效应。本文引入了一种新颖的信用卡欺诈检测方法——CaT-GNN（Causal Temporal Graph Neural Networks），利用因果不变性学习来揭示交易数据中的固有相关性。通过将问题分解为发现和干预阶段，CaT-GNN确定交易图中的因果节点，并应用因果混合策略来增强模型的鲁棒性和可解释性。CaT-GNN由两个关键组件组成：Causal-Inspector和Causal-Intervener。Causal-Inspector利用时间注意力机制中的注意力权重来识别因果和环境

    arXiv:2402.14708v1 Announce Type: cross  Abstract: Credit card fraud poses a significant threat to the economy. While Graph Neural Network (GNN)-based fraud detection methods perform well, they often overlook the causal effect of a node's local structure on predictions. This paper introduces a novel method for credit card fraud detection, the \textbf{\underline{Ca}}usal \textbf{\underline{T}}emporal \textbf{\underline{G}}raph \textbf{\underline{N}}eural \textbf{N}etwork (CaT-GNN), which leverages causal invariant learning to reveal inherent correlations within transaction data. By decomposing the problem into discovery and intervention phases, CaT-GNN identifies causal nodes within the transaction graph and applies a causal mixup strategy to enhance the model's robustness and interpretability. CaT-GNN consists of two key components: Causal-Inspector and Causal-Intervener. The Causal-Inspector utilizes attention weights in the temporal attention mechanism to identify causal and environm
    
[^5]: MENTOR：在层次化强化学习中引导人类反馈和动态距离约束

    MENTOR: Guiding Hierarchical Reinforcement Learning with Human Feedback and Dynamic Distance Constraint

    [https://arxiv.org/abs/2402.14244](https://arxiv.org/abs/2402.14244)

    使用人类反馈和动态距离约束对层次化强化学习进行引导，解决了找到适当子目标的问题，并设计了双策略以稳定训练。

    

    层次化强化学习（HRL）为智能体的复杂任务提供了一种有前途的解决方案，其中使用了将任务分解为子目标并依次完成的层次框架。然而，当前的方法难以找到适当的子目标来确保稳定的学习过程。为了解决这个问题，我们提出了一个通用的层次强化学习框架，将人类反馈和动态距离约束整合到其中（MENTOR）。MENTOR充当“导师”，将人类反馈纳入高层策略学习中，以找到更好的子目标。至于低层策略，MENTOR设计了一个双策略以分别进行探索-开发解耦，以稳定训练。此外，尽管人类可以简单地将任务拆分成...

    arXiv:2402.14244v1 Announce Type: new  Abstract: Hierarchical reinforcement learning (HRL) provides a promising solution for complex tasks with sparse rewards of intelligent agents, which uses a hierarchical framework that divides tasks into subgoals and completes them sequentially. However, current methods struggle to find suitable subgoals for ensuring a stable learning process. Without additional guidance, it is impractical to rely solely on exploration or heuristics methods to determine subgoals in a large goal space. To address the issue, We propose a general hierarchical reinforcement learning framework incorporating human feedback and dynamic distance constraints (MENTOR). MENTOR acts as a "mentor", incorporating human feedback into high-level policy learning, to find better subgoals. As for low-level policy, MENTOR designs a dual policy for exploration-exploitation decoupling respectively to stabilize the training. Furthermore, although humans can simply break down tasks into s
    
[^6]: 用于数据分析的多尺度霍奇散射网络

    Multiscale Hodge Scattering Networks for Data Analysis

    [https://arxiv.org/abs/2311.10270](https://arxiv.org/abs/2311.10270)

    提出了多尺度霍奇散射网络（MHSNs），利用多尺度基础词典和卷积结构，生成对节点排列不变的特征。

    

    我们提出了一种新的散射网络，用于在单纯复合仿射上测量的信号，称为\emph{多尺度霍奇散射网络}（MHSNs）。我们的构造基于单纯复合仿射上的多尺度基础词典，即$\kappa$-GHWT和$\kappa$-HGLET，我们最近为给定单纯复合仿射中的维度$\kappa \in \mathbb{N}$推广了基于节点的广义哈-沃什变换（GHWT）和分层图拉普拉斯特征变换（HGLET）。$\kappa$-GHWT和$\kappa$-HGLET都形成冗余集合（即词典）的多尺度基础向量和给定信号的相应扩展系数。我们的MHSNs使用类似于卷积神经网络（CNN）的分层结构来级联词典系数模的矩。所得特征对单纯复合仿射的重新排序不变（即节点排列的置换

    arXiv:2311.10270v2 Announce Type: replace  Abstract: We propose new scattering networks for signals measured on simplicial complexes, which we call \emph{Multiscale Hodge Scattering Networks} (MHSNs). Our construction is based on multiscale basis dictionaries on simplicial complexes, i.e., the $\kappa$-GHWT and $\kappa$-HGLET, which we recently developed for simplices of dimension $\kappa \in \mathbb{N}$ in a given simplicial complex by generalizing the node-based Generalized Haar-Walsh Transform (GHWT) and Hierarchical Graph Laplacian Eigen Transform (HGLET). The $\kappa$-GHWT and the $\kappa$-HGLET both form redundant sets (i.e., dictionaries) of multiscale basis vectors and the corresponding expansion coefficients of a given signal. Our MHSNs use a layered structure analogous to a convolutional neural network (CNN) to cascade the moments of the modulus of the dictionary coefficients. The resulting features are invariant to reordering of the simplices (i.e., node permutation of the u
    
[^7]: cedar：可组合和优化的机器学习输入数据管道

    cedar: Composable and Optimized Machine Learning Input Data Pipelines. (arXiv:2401.08895v1 [cs.LG])

    [http://arxiv.org/abs/2401.08895](http://arxiv.org/abs/2401.08895)

    cedar是一个编程模型和框架，可以轻松构建、优化和执行机器学习输入数据管道。它提供了易于使用的编程接口和可组合运算符，支持任意ML框架和库。通过解决当前输入数据系统无法充分利用性能优化的问题，cedar提高了资源利用效率，满足了庞大数据量和高训练吞吐量的需求。

    

    输入数据管道是每个机器学习（ML）训练任务的重要组成部分。它负责读取大量的训练数据，使用复杂的变换处理样本批次，并以低延迟和高吞吐量将其加载到训练节点上。高性能的输入数据系统变得越来越关键，原因是数据量急剧增加和训练吞吐量的要求。然而，当前的输入数据系统无法充分利用关键的性能优化，导致资源利用效率极低的基础设施，或者更糟糕地，浪费昂贵的加速器。为了满足这些需求，我们提出了cedar，一个编程模型和框架，允许用户轻松构建、优化和执行输入数据管道。cedar提供了易于使用的编程接口，允许用户使用可组合运算符来定义支持任意ML框架和库的输入数据管道。

    The input data pipeline is an essential component of each machine learning (ML) training job. It is responsible for reading massive amounts of training data, processing batches of samples using complex of transformations, and loading them onto training nodes at low latency and high throughput. Performant input data systems are becoming increasingly critical, driven by skyrocketing data volumes and training throughput demands. Unfortunately, current input data systems cannot fully leverage key performance optimizations, resulting in hugely inefficient infrastructures that require significant resources -- or worse -- underutilize expensive accelerators.  To address these demands, we present cedar, a programming model and framework that allows users to easily build, optimize, and execute input data pipelines. cedar presents an easy-to-use programming interface, allowing users to define input data pipelines using composable operators that support arbitrary ML frameworks and libraries. Mean
    
[^8]: CSG: 用于有符号图的课程表示学习

    CSG: Curriculum Representation Learning for Signed Graph. (arXiv:2310.11083v1 [cs.LG])

    [http://arxiv.org/abs/2310.11083](http://arxiv.org/abs/2310.11083)

    本文提出了一种用于有符号图的课程表示学习框架（CSG），通过引入课程化训练方法和轻量级机制，实现了按难易程度优化样本展示顺序，从而提高有符号图神经网络（SGNN）模型的准确性和稳定性。

    

    有符号图对于建模具有正负连接的复杂关系非常有价值，有符号图神经网络已成为其分析的重要工具。然而，在我们的工作之前，没有针对有符号图神经网络的特定训练方案，并且传统的随机抽样方法没有解决图结构中不同学习困难的问题。我们提出了一种基于课程的训练方法，其中样本从易到难，灵感来自于人类学习。为了衡量学习困难，我们引入了一个轻量级机制，并创建了用于有符号图的课程表示学习框架（CSG）。通过对六个真实数据集的实证验证，我们取得了令人印象深刻的结果，在链接符号预测（AUC）方面将SGNN模型的准确性提高了高达23.7％，并且在AUC的标准差方面显著提高了稳定性，最多减少了8.4。

    Signed graphs are valuable for modeling complex relationships with positive and negative connections, and Signed Graph Neural Networks (SGNNs) have become crucial tools for their analysis. However, prior to our work, no specific training plan existed for SGNNs, and the conventional random sampling approach did not address varying learning difficulties within the graph's structure. We proposed a curriculum-based training approach, where samples progress from easy to complex, inspired by human learning. To measure learning difficulty, we introduced a lightweight mechanism and created the Curriculum representation learning framework for Signed Graphs (CSG). This framework optimizes the order in which samples are presented to the SGNN model. Empirical validation across six real-world datasets showed impressive results, enhancing SGNN model accuracy by up to 23.7% in link sign prediction (AUC) and significantly improving stability with an up to 8.4 reduction in the standard deviation of AUC
    
[^9]: 用低秩内核模型简化GNN性能

    Simplifying GNN Performance with Low Rank Kernel Models. (arXiv:2310.05250v1 [cs.LG])

    [http://arxiv.org/abs/2310.05250](http://arxiv.org/abs/2310.05250)

    本文提出了一种用于简化GNN性能的低秩内核模型，通过应用传统的非参数估计方法在谱域中取代过于复杂的GNN架构，并在多个图类型的半监督节点分类基准测试中取得了最先进的性能。

    

    我们重新审视了最近的谱GNN方法对半监督节点分类（SSNC）的应用。我们认为许多当前的GNN架构可能过于精细设计。相反，简单的非参数估计传统方法，在谱域中应用，可以取代许多受深度学习启发的GNN设计。这些传统技术似乎非常适合各种图类型，在许多常见的SSNC基准测试中达到了最先进的性能。此外，我们还展示了最近在GNN方法方面的性能改进可能部分归因于评估惯例的变化。最后，我们对与GNN谱过滤技术相关的各种超参数进行了消融研究。

    We revisit recent spectral GNN approaches to semi-supervised node classification (SSNC). We posit that many of the current GNN architectures may be over-engineered. Instead, simpler, traditional methods from nonparametric estimation, applied in the spectral domain, could replace many deep-learning inspired GNN designs. These conventional techniques appear to be well suited for a variety of graph types reaching state-of-the-art performance on many of the common SSNC benchmarks. Additionally, we show that recent performance improvements in GNN approaches may be partially attributed to shifts in evaluation conventions. Lastly, an ablative study is conducted on the various hyperparameters associated with GNN spectral filtering techniques. Code available at: https://github.com/lucianoAvinas/lowrank-gnn-kernels
    
[^10]: 一种用于评估冠心病的3D深度学习分类器及其可解释性

    A 3D deep learning classifier and its explainability when assessing coronary artery disease. (arXiv:2308.00009v1 [eess.IV])

    [http://arxiv.org/abs/2308.00009](http://arxiv.org/abs/2308.00009)

    本文提出了一种3D深度学习模型，用于直接分类冠心病患者和正常受试者，相较于2D模型提高了23.65%的性能，并通过Grad-GAM提供了可解释性。此外，通过与2D语义分割相结合，实现了更好的解释性和准确的异常定位。

    

    早期发现和诊断冠心病（CAD）可挽救生命并降低医疗成本。在本研究中，我们提出了一个3D Resnet-50深度学习模型，可以直接对计算机断层扫描冠状动脉造影图像上的正常受试者和冠心病患者进行分类。我们的方法比2D Resnet-50模型提高了23.65%。通过使用Grad-GAM提供了可解释性。此外，我们将3D冠心病分类与2D二类语义分割相结合，以提高解释性和准确的异常定位。

    Early detection and diagnosis of coronary artery disease (CAD) could save lives and reduce healthcare costs. In this study, we propose a 3D Resnet-50 deep learning model to directly classify normal subjects and CAD patients on computed tomography coronary angiography images. Our proposed method outperforms a 2D Resnet-50 model by 23.65%. Explainability is also provided by using a Grad-GAM. Furthermore, we link the 3D CAD classification to a 2D two-class semantic segmentation for improved explainability and accurate abnormality localisation.
    
[^11]: 基于图谱的可解释年龄预测

    Atlas-Based Interpretable Age Prediction. (arXiv:2307.07439v1 [eess.IV])

    [http://arxiv.org/abs/2307.07439](http://arxiv.org/abs/2307.07439)

    本研究提出了一种基于图谱的可解释年龄预测方法，利用全身图像研究了各个身体部位的年龄相关变化。通过使用解释性方法和配准技术，确定了最能预测年龄的身体区域，并创下了整个身体年龄预测的最新水平。研究结果表明，脊柱、本原性背部肌肉和心脏区域是最重要的关注领域。

    

    年龄预测是医学评估和研究的重要部分，可以通过突出实际年龄和生物年龄之间的差异来帮助检测疾病和异常衰老。为了全面了解各个身体部位的年龄相关变化，我们使用了全身图像进行研究。我们利用Grad-CAM解释性方法确定最能预测一个人年龄的身体区域。通过使用配准技术生成整个人群的解释性图，我们将分析扩展到个体之外。此外，我们以一个平均绝对误差为2.76年的模型，创下了整个身体年龄预测的最新水平。我们的研究结果揭示了三个主要的关注领域：脊柱、本原性背部肌肉和心脏区域，其中心脏区域具有最重要的作用。

    Age prediction is an important part of medical assessments and research. It can aid in detecting diseases as well as abnormal ageing by highlighting the discrepancy between chronological and biological age. To gain a comprehensive understanding of age-related changes observed in various body parts, we investigate them on a larger scale by using whole-body images. We utilise the Grad-CAM interpretability method to determine the body areas most predictive of a person's age. We expand our analysis beyond individual subjects by employing registration techniques to generate population-wide interpretability maps. Furthermore, we set state-of-the-art whole-body age prediction with a model that achieves a mean absolute error of 2.76 years. Our findings reveal three primary areas of interest: the spine, the autochthonous back muscles, and the cardiac region, which exhibits the highest importance.
    
[^12]: 异构视觉深度网络社区中的指代性沟通

    Referential communication in heterogeneous communities of pre-trained visual deep networks. (arXiv:2302.08913v3 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2302.08913](http://arxiv.org/abs/2302.08913)

    异构视觉深度网络社区中的预训练网络可以自我监督地开发出共享协议，以指代一组目标中的目标对象，并可用于沟通不同粒度的未知对象类别。

    

    随着大型预训练图像处理神经网络被嵌入自动驾驶汽车或机器人等自主代理中，一个问题出现了：在它们具有不同架构和训练方式的情况下，这些系统如何相互之间进行沟通以了解周围的世界。作为朝着这个方向的第一步，我们系统地探索了在一组异构最先进的预训练视觉网络社区中进行"指代性沟通"的任务，结果表明它们可以自我监督地发展一种共享协议来指代一组候选目标中的目标对象。在某种程度上，这种共享协议也可以用来沟通不同粒度的先前未见过的对象类别。此外，一个最初不属于现有社区的视觉网络可以轻松地学习到社区的协议。最后，我们定性和定量地研究了这种新产生的协议的属性，提供了一些证据。

    As large pre-trained image-processing neural networks are being embedded in autonomous agents such as self-driving cars or robots, the question arises of how such systems can communicate with each other about the surrounding world, despite their different architectures and training regimes. As a first step in this direction, we systematically explore the task of \textit{referential communication} in a community of heterogeneous state-of-the-art pre-trained visual networks, showing that they can develop, in a self-supervised way, a shared protocol to refer to a target object among a set of candidates. This shared protocol can also be used, to some extent, to communicate about previously unseen object categories of different granularity. Moreover, a visual network that was not initially part of an existing community can learn the community's protocol with remarkable ease. Finally, we study, both qualitatively and quantitatively, the properties of the emergent protocol, providing some evi
    

