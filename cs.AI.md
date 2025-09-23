# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Breast Cancer Classification Using Gradient Boosting Algorithms Focusing on Reducing the False Negative and SHAP for Explainability](https://arxiv.org/abs/2403.09548) | 本研究使用梯度提升算法对乳腺癌进行分类，关注提高召回率，以实现更好的检测和预测效果。 |
| [^2] | [Can Large Language Models Recall Reference Location Like Humans?](https://arxiv.org/abs/2402.17010) | 本文探讨了大型语言模型如何利用预训练阶段的知识回忆参考段落，提出了一个两阶段框架模拟人类回忆参考的过程。 |
| [^3] | [Faster and Accurate Neural Networks with Semantic Inference.](http://arxiv.org/abs/2310.01259) | 本研究提出了一种名为语义推理（SINF）的新框架，在减少计算负载的同时，通过聚类语义相似的类来提取子图，从而减少深度神经网络的计算负担，并在性能上有限损失。 |
| [^4] | [Audio Contrastive based Fine-tuning.](http://arxiv.org/abs/2309.11895) | 本论文提出了一种基于音频对比的微调方法（AudioConFit），通过借助对比学习的可转移性，该方法在各种音频分类任务中表现出强大的泛化能力，并在不同设置下实现了最先进的结果。 |
| [^5] | [Asynchronous Perception-Action-Communication with Graph Neural Networks.](http://arxiv.org/abs/2309.10164) | 该论文提出了使用图神经网络实现异步感知-动作-通信的方法，解决了在大型机器人群体中协作和通信的挑战。现有的框架假设顺序执行，该方法是完全分散的，但在评估和部署方面仍存在一些限制。 |
| [^6] | [Alpha-GPT: Human-AI Interactive Alpha Mining for Quantitative Investment.](http://arxiv.org/abs/2308.00016) | 本论文提出了一种通过引入人机交互的新型 alpha 挖掘范式，并利用大型语言模型的能力，通过一种新颖的提示工程算法框架，开发了 Alpha-GPT。通过多个实验，展示了 Alpha-GPT 在量化投资领域的有效性和优势。 |
| [^7] | [Semantic Technologies in Sensor-Based Personal Health Monitoring Systems: A Systematic Mapping Study.](http://arxiv.org/abs/2306.04335) | 这项研究评估了传感器个人健康监测系统中语义技术的应用现状，发现此类系统必须克服的关键挑战为互操作性、上下文感知、情境检测、情境预测、决策支持和知识表示。 |
| [^8] | [Exploiting Symmetry and Heuristic Demonstrations in Off-policy Reinforcement Learning for Robotic Manipulation.](http://arxiv.org/abs/2304.06055) | 本文提出了一个离线强化学习方法，该方法利用对称环境中的专家演示来进行机器人操作的策略训练，从而提高了学习效率和样本效率。 |
| [^9] | [Unsupervised Interpretable Basis Extraction for Concept-Based Visual Explanations.](http://arxiv.org/abs/2303.10523) | 本文提出了一种无监督的方法，通过对CNN进行转换，从而更好地解释中间层的表示，提取了一个可解释性欠完备基础，并证明该方法在各种网络结构和训练数据集上都很有效。 |

# 详细

[^1]: 使用梯度提升算法对乳腺癌进行分类，重点减少假阴性和使用 SHAP 进行解释性研究

    Breast Cancer Classification Using Gradient Boosting Algorithms Focusing on Reducing the False Negative and SHAP for Explainability

    [https://arxiv.org/abs/2403.09548](https://arxiv.org/abs/2403.09548)

    本研究使用梯度提升算法对乳腺癌进行分类，关注提高召回率，以实现更好的检测和预测效果。

    

    癌症是世界上夺走最多女性生命的疾病之一，其中乳腺癌占据了癌症病例和死亡人数最高的位置。然而，通过早期检测可以预防乳腺癌，从而进行早期治疗。许多研究关注的是在癌症预测中具有高准确性的模型，但有时仅依靠准确性可能并非始终可靠。本研究对使用提升技术基于不同机器学习算法预测乳腺癌的性能进行了调查性研究，重点关注召回率指标。提升机器学习算法已被证明是检测医学疾病的有效工具。利用加州大学尔湾分校 (UCI) 数据集对训练和测试模型分类器进行训练，其中包含各自属性。

    arXiv:2403.09548v1 Announce Type: new  Abstract: Cancer is one of the diseases that kill the most women in the world, with breast cancer being responsible for the highest number of cancer cases and consequently deaths. However, it can be prevented by early detection and, consequently, early treatment. Any development for detection or perdition this kind of cancer is important for a better healthy life. Many studies focus on a model with high accuracy in cancer prediction, but sometimes accuracy alone may not always be a reliable metric. This study implies an investigative approach to studying the performance of different machine learning algorithms based on boosting to predict breast cancer focusing on the recall metric. Boosting machine learning algorithms has been proven to be an effective tool for detecting medical diseases. The dataset of the University of California, Irvine (UCI) repository has been utilized to train and test the model classifier that contains their attributes. Th
    
[^2]: 大型语言模型能像人类一样回忆参考位置吗？

    Can Large Language Models Recall Reference Location Like Humans?

    [https://arxiv.org/abs/2402.17010](https://arxiv.org/abs/2402.17010)

    本文探讨了大型语言模型如何利用预训练阶段的知识回忆参考段落，提出了一个两阶段框架模拟人类回忆参考的过程。

    

    在完成知识密集型任务时，人类有时不仅需要一个答案，还需要相应的参考段落供辅助阅读。先前的方法需要通过额外的检索模型获取预分段的文章块。本文探讨了利用大型语言模型（LLMs）的预训练阶段存储的参数化知识，独立于任何起始位置回忆参考段落。我们提出了一个模拟人类回忆易被遗忘参考的情景的两阶段框架。首先，LLM被提示回忆文档标题标识符以获取粗粒度文档集。然后，基于获得的粗粒度文档集，它回忆细粒度段落。在两阶段回忆过程中，我们使用约束解码来确保不生成存储文档之外的内容。为了增加速度，我们只回忆短前缀。

    arXiv:2402.17010v1 Announce Type: cross  Abstract: When completing knowledge-intensive tasks, humans sometimes need not just an answer but also a corresponding reference passage for auxiliary reading. Previous methods required obtaining pre-segmented article chunks through additional retrieval models. This paper explores leveraging the parameterized knowledge stored during the pre-training phase of large language models (LLMs) to independently recall reference passage from any starting position. We propose a two-stage framework that simulates the scenario of humans recalling easily forgotten references. Initially, the LLM is prompted to recall document title identifiers to obtain a coarse-grained document set. Then, based on the acquired coarse-grained document set, it recalls fine-grained passage. In the two-stage recall process, we use constrained decoding to ensure that content outside of the stored documents is not generated. To increase speed, we only recall a short prefix in the 
    
[^3]: 使用语义推理实现更快更准确的神经网络

    Faster and Accurate Neural Networks with Semantic Inference. (arXiv:2310.01259v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2310.01259](http://arxiv.org/abs/2310.01259)

    本研究提出了一种名为语义推理（SINF）的新框架，在减少计算负载的同时，通过聚类语义相似的类来提取子图，从而减少深度神经网络的计算负担，并在性能上有限损失。

    

    深度神经网络通常具有显著的计算负担。虽然提出了结构化剪枝和专门用于移动设备的神经网络的方法，但它们会导致明显的准确率损失。在本文中，我们利用潜在表示中的内在冗余来减少计算负载，并在性能上有限损失。我们证明，语义上相似的输入共享许多滤波器，尤其是在较早的层次上。因此，可以对语义上相似的类进行聚类，以创建特定于聚类的子图。为此，我们提出了一个名为语义推理（SINF）的新框架。简而言之，SINF（i）使用一个小的附加分类器来识别对象属于的语义聚类，并（ii）执行与该语义聚类相关的基本DNN提取的子图进行推理。为了提取每个特定于聚类的子图，我们提出了一个名为区分能力得分（DCS）的新方法，用于找到具有区分能力的子图。

    Deep neural networks (DNN) usually come with a significant computational burden. While approaches such as structured pruning and mobile-specific DNNs have been proposed, they incur drastic accuracy loss. In this paper we leverage the intrinsic redundancy in latent representations to reduce the computational load with limited loss in performance. We show that semantically similar inputs share many filters, especially in the earlier layers. Thus, semantically similar classes can be clustered to create cluster-specific subgraphs. To this end, we propose a new framework called Semantic Inference (SINF). In short, SINF (i) identifies the semantic cluster the object belongs to using a small additional classifier and (ii) executes the subgraph extracted from the base DNN related to that semantic cluster for inference. To extract each cluster-specific subgraph, we propose a new approach named Discriminative Capability Score (DCS) that finds the subgraph with the capability to discriminate amon
    
[^4]: 基于音频对比的微调方法

    Audio Contrastive based Fine-tuning. (arXiv:2309.11895v1 [cs.SD])

    [http://arxiv.org/abs/2309.11895](http://arxiv.org/abs/2309.11895)

    本论文提出了一种基于音频对比的微调方法（AudioConFit），通过借助对比学习的可转移性，该方法在各种音频分类任务中表现出强大的泛化能力，并在不同设置下实现了最先进的结果。

    

    音频分类在语音和声音处理任务中起着至关重要的作用，具有广泛的应用。在将模型拟合到训练数据（避免过拟合）并使其能够良好地泛化到新领域之间仍然存在着平衡的挑战。借助对比学习的可转移性，我们引入了基于音频对比的微调方法（AudioConFit），这种方法具有强大的泛化能力。对各种音频分类任务的实证实验表明了我们方法的有效性和鲁棒性，在不同设置下取得了最先进的结果。

    Audio classification plays a crucial role in speech and sound processing tasks with a wide range of applications. There still remains a challenge of striking the right balance between fitting the model to the training data (avoiding overfitting) and enabling it to generalise well to a new domain. Leveraging the transferability of contrastive learning, we introduce Audio Contrastive-based Fine-tuning (AudioConFit), an efficient approach characterised by robust generalisability. Empirical experiments on a variety of audio classification tasks demonstrate the effectiveness and robustness of our approach, which achieves state-of-the-art results in various settings.
    
[^5]: 异步感知-动作-通信与图神经网络

    Asynchronous Perception-Action-Communication with Graph Neural Networks. (arXiv:2309.10164v1 [cs.RO])

    [http://arxiv.org/abs/2309.10164](http://arxiv.org/abs/2309.10164)

    该论文提出了使用图神经网络实现异步感知-动作-通信的方法，解决了在大型机器人群体中协作和通信的挑战。现有的框架假设顺序执行，该方法是完全分散的，但在评估和部署方面仍存在一些限制。

    

    在大型机器人群体中实现共同的全局目标的协作是一个具有挑战性的问题，因为机器人的感知和通信能力有限。机器人必须执行感知-动作-通信（PAC）循环-它们感知局部环境，与其他机器人通信，并实时采取行动。分散的PAC系统面临的一个基本挑战是决定与相邻机器人通信的信息以及如何在利用邻居共享的信息的同时采取行动。最近，使用图神经网络（GNNs）来解决这个问题已经取得了一些进展，比如在群集和覆盖控制等应用中。虽然在概念上，GNN策略是完全分散的，但评估和部署这样的策略主要仍然是集中式的或具有限制性的分散式。此外，现有的框架假设感知和动作推理的顺序执行，这在现实世界的应用中非常限制性。

    Collaboration in large robot swarms to achieve a common global objective is a challenging problem in large environments due to limited sensing and communication capabilities. The robots must execute a Perception-Action-Communication (PAC) loop -- they perceive their local environment, communicate with other robots, and take actions in real time. A fundamental challenge in decentralized PAC systems is to decide what information to communicate with the neighboring robots and how to take actions while utilizing the information shared by the neighbors. Recently, this has been addressed using Graph Neural Networks (GNNs) for applications such as flocking and coverage control. Although conceptually, GNN policies are fully decentralized, the evaluation and deployment of such policies have primarily remained centralized or restrictively decentralized. Furthermore, existing frameworks assume sequential execution of perception and action inference, which is very restrictive in real-world applica
    
[^6]: Alpha-GPT：人机交互式 Alpha 挖掘在量化投资中的应用

    Alpha-GPT: Human-AI Interactive Alpha Mining for Quantitative Investment. (arXiv:2308.00016v1 [q-fin.CP])

    [http://arxiv.org/abs/2308.00016](http://arxiv.org/abs/2308.00016)

    本论文提出了一种通过引入人机交互的新型 alpha 挖掘范式，并利用大型语言模型的能力，通过一种新颖的提示工程算法框架，开发了 Alpha-GPT。通过多个实验，展示了 Alpha-GPT 在量化投资领域的有效性和优势。

    

    在量化投资研究中，挖掘新的 alpha（有效的交易信号或因子）是其中最重要的任务之一。传统的 alpha 挖掘方法，无论是手工合成因子还是算法挖掘因子（如遗传编程搜索），都存在固有的局限性，尤其在实施量化分析师的想法方面。在本研究中，我们提出了一种新的 alpha 挖掘范式，引入了人机交互，并通过利用大型语言模型的能力，提出了一种新颖的提示工程算法框架来实现这个范式。此外，我们开发了 Alpha-GPT，一种新的交互式 alpha 挖掘系统框架，以一种启发式的方式“理解”量化研究人员的想法，并输出具有创造性、深入洞察力和有效性的 alpha。通过多个 alpha 挖掘实验，我们展示了 Alpha-GPT 的有效性和优势。

    One of the most important tasks in quantitative investment research is mining new alphas (effective trading signals or factors). Traditional alpha mining methods, either hand-crafted factor synthesizing or algorithmic factor mining (e.g., search with genetic programming), have inherent limitations, especially in implementing the ideas of quants. In this work, we propose a new alpha mining paradigm by introducing human-AI interaction, and a novel prompt engineering algorithmic framework to implement this paradigm by leveraging the power of large language models. Moreover, we develop Alpha-GPT, a new interactive alpha mining system framework that provides a heuristic way to ``understand'' the ideas of quant researchers and outputs creative, insightful, and effective alphas. We demonstrate the effectiveness and advantage of Alpha-GPT via a number of alpha mining experiments.
    
[^7]: 传感器个人健康监测系统中的语义技术：一项系统性映射研究

    Semantic Technologies in Sensor-Based Personal Health Monitoring Systems: A Systematic Mapping Study. (arXiv:2306.04335v1 [cs.AI])

    [http://arxiv.org/abs/2306.04335](http://arxiv.org/abs/2306.04335)

    这项研究评估了传感器个人健康监测系统中语义技术的应用现状，发现此类系统必须克服的关键挑战为互操作性、上下文感知、情境检测、情境预测、决策支持和知识表示。

    

    近年来，人们对于疾病的早期检测、预防和预测越来越重视。此外，传感器技术和物联网技术的不断进步也推动了个人健康监测系统的发展。语义技术作为一种有效的方法，不仅可以处理异构健康传感器数据的互操作性问题，还可以表示专家健康知识以支持决策所需的复杂推理。本研究评估了传感器个人健康监测系统中语义技术的应用现状。使用系统方法对40个代表该领域最新技术水平的系统进行了分析。通过这项分析，确定了此类系统必须克服的六个关键挑战：互操作性、上下文感知、情境检测、情境预测、决策支持和知识表示。

    In recent years, there has been an increased focus on early detection, prevention, and prediction of diseases. This, together with advances in sensor technology and the Internet of Things, has led to accelerated efforts in the development of personal health monitoring systems. Semantic technologies have emerged as an effective way to not only deal with the issue of interoperability associated with heterogeneous health sensor data, but also to represent expert health knowledge to support complex reasoning required for decision-making. This study evaluates the state of the art in the use of semantic technologies in sensor-based personal health monitoring systems. Using a systematic approach, a total of 40 systems representing the state of the art in the field are analysed. Through this analysis, six key challenges that such systems must overcome for optimal and effective health monitoring are identified: interoperability, context awareness, situation detection, situation prediction, deci
    
[^8]: 利用对称性和启发式演示来进行机器人操作的离线强化学习

    Exploiting Symmetry and Heuristic Demonstrations in Off-policy Reinforcement Learning for Robotic Manipulation. (arXiv:2304.06055v1 [cs.RO])

    [http://arxiv.org/abs/2304.06055](http://arxiv.org/abs/2304.06055)

    本文提出了一个离线强化学习方法，该方法利用对称环境中的专家演示来进行机器人操作的策略训练，从而提高了学习效率和样本效率。

    

    强化学习在许多领域中自动构建控制策略具有显著潜力，但在应用于机器人操作任务时由于维度的问题，效率较低。为了促进这些任务的学习，先前的知识或启发式方法可以有效地提高学习性能。本文旨在定义和结合物理机器环境中存在的自然对称性，利用对称环境中的专家演示通过强化学习和行为克隆的融合来训练具有高样本效率的策略，从而给离线强化学习过程提供多样化而紧凑的启动。此外，本文提出了一个最近概念的严格框架，并探索了它在机器人操作任务中的范围。该方法通过在模拟环境中进行两个点对点的工业臂到达任务（有障碍和无障碍）的验证。

    Reinforcement learning demonstrates significant potential in automatically building control policies in numerous domains, but shows low efficiency when applied to robot manipulation tasks due to the curse of dimensionality. To facilitate the learning of such tasks, prior knowledge or heuristics that incorporate inherent simplification can effectively improve the learning performance. This paper aims to define and incorporate the natural symmetry present in physical robotic environments. Then, sample-efficient policies are trained by exploiting the expert demonstrations in symmetrical environments through an amalgamation of reinforcement and behavior cloning, which gives the off-policy learning process a diverse yet compact initiation. Furthermore, it presents a rigorous framework for a recent concept and explores its scope for robot manipulation tasks. The proposed method is validated via two point-to-point reaching tasks of an industrial arm, with and without an obstacle, in a simulat
    
[^9]: 无监督解释性基础抽取用于基于概念的视觉解释

    Unsupervised Interpretable Basis Extraction for Concept-Based Visual Explanations. (arXiv:2303.10523v1 [cs.CV])

    [http://arxiv.org/abs/2303.10523](http://arxiv.org/abs/2303.10523)

    本文提出了一种无监督的方法，通过对CNN进行转换，从而更好地解释中间层的表示，提取了一个可解释性欠完备基础，并证明该方法在各种网络结构和训练数据集上都很有效。

    

    研究人员尝试用人类可以理解的概念来解释CNN图像分类器预测和中间层表示。本文提出了一种无监督后处理方法，通过查找解释像素激活的稀疏二值化转换表示的特征空间旋转来提取解释性欠完备基础。我们对现有的流行CNN进行了实验，并证明了我们方法在网络架构和训练数据集上提取解释性基础的有效性。最后，我们扩展了文献中的基础可解释性度量，并表明，当中间层表示被转换为我们方法提取的基础时，它们变得更易解释。

    An important line of research attempts to explain CNN image classifier predictions and intermediate layer representations in terms of human understandable concepts. In this work, we expand on previous works in the literature that use annotated concept datasets to extract interpretable feature space directions and propose an unsupervised post-hoc method to extract a disentangling interpretable basis by looking for the rotation of the feature space that explains sparse one-hot thresholded transformed representations of pixel activations. We do experimentation with existing popular CNNs and demonstrate the effectiveness of our method in extracting an interpretable basis across network architectures and training datasets. We make extensions to the existing basis interpretability metrics found in the literature and show that, intermediate layer representations become more interpretable when transformed to the bases extracted with our method. Finally, using the basis interpretability metrics
    

