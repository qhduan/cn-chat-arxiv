# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Bridging Language, Vision and Action: Multimodal VAEs in Robotic Manipulation Tasks](https://arxiv.org/abs/2404.01932) | 本研究探索了多模态VAE如何在无监督机器人操作任务中实现，提出了一个新的模型训练方法，可以使模拟器中的模型性能提高55%。 |
| [^2] | [LC-Tsalis-INF: Generalized Best-of-Both-Worlds Linear Contextual Bandits](https://arxiv.org/abs/2403.03219) | 该研究提出了一种广义最佳双赢线性背景强化型赌博机算法，能够在次优性差距受到下界限制时遗憾为$O(\log(T))$。同时引入了边缘条件来描述次优性差距对问题难度的影响。 |
| [^3] | [FedStruct: Federated Decoupled Learning over Interconnected Graphs](https://arxiv.org/abs/2402.19163) | FedStruct提出了一种新的框架，利用深层结构依赖关系在互联图上进行联合解耦学习，有效地维护隐私并捕捉节点间的依赖关系。 |
| [^4] | [InterpretCC: Conditional Computation for Inherently Interpretable Neural Networks](https://arxiv.org/abs/2402.02933) | InterpretCC是一种新的解释性神经网络模型，通过条件计算和稀疏激活特征，在保持性能的同时实现了人类中心的解释能力。该模型适用于需要可信解释、可操作解释和准确预测的人类面向领域。 |
| [^5] | [Meta Co-Training: Two Views are Better than One](https://arxiv.org/abs/2311.18083) | 元共训练通过在数据上构建不同的视角，并利用未标记数据进行共同训练，提高了半监督学习的性能。 |
| [^6] | [LMM-Assisted Breast Cancer Treatment Target Segmentation with Consistency Embedding](https://arxiv.org/abs/2311.15876) | RO-LMM是一个针对放射肿瘤学领域设计的多功能大型多模型，提出了一种Consistency Embedding Fine-Tuning（CEFTune）技术，使其能够在保持处理干净输入能力的同时提升对嘈杂输入的鲁棒性，用于放射治疗计划和目标体积分割。 |
| [^7] | [Beyond RMSE and MAE: Introducing EAUC to unmask hidden bias and unfairness in dyadic regression models.](http://arxiv.org/abs/2401.10690) | 本研究引入EAUC作为一种新的度量标准，用以揭示迪亚德回归模型中隐藏的偏见和不公平问题。传统的全局错误度量标准如RMSE和MAE无法捕捉到这种问题。 |
| [^8] | [A Simple Way to Incorporate Novelty Detection in World Models.](http://arxiv.org/abs/2310.08731) | 本文提出了一个简单的方法，通过利用世界模型幻觉状态和真实观察状态的不匹配性作为异常分数，将新颖性检测纳入世界模型强化学习代理中。在新环境中与传统方法相比，我们的工作具有优势。 |
| [^9] | [Causal Inference with Differentially Private (Clustered) Outcomes.](http://arxiv.org/abs/2308.00957) | 本文提出了一种新的差分隐私机制"Cluster-DP"，它在保证隐私的同时利用数据的聚类结构，从而实现了更强的隐私保证和较低的方差，可以用于进行因果分析。 |
| [^10] | [Variational Positive-incentive Noise: How Noise Benefits Models.](http://arxiv.org/abs/2306.07651) | 本文研究了如何通过正激励噪声框架下的随机噪声使经典模型受益，并提出了变分Pi-Noise，它可以在不改变原始模型结构的情况下增强和简化模型。 |
| [^11] | [Is Rewiring Actually Helpful in Graph Neural Networks?.](http://arxiv.org/abs/2305.19717) | 本文研究了图神经网络中的改连方法是否有用，提出了一种新的评估设置，并在真实世界的节点和图分类任务上进行了系统的实验比较。 |
| [^12] | [Inferring Traffic Models in Terminal Airspace from Flight Tracks and Procedures.](http://arxiv.org/abs/2303.09981) | 本文提出了一种通过收集飞行轨迹和程序数据学习飞行器行为变异性的概率模型，并且可以生成涉及任意数量飞行器的交通模型。 |

# 详细

[^1]: 跨越语言、视觉和行动：多模态VAE在机器人操作任务中的应用

    Bridging Language, Vision and Action: Multimodal VAEs in Robotic Manipulation Tasks

    [https://arxiv.org/abs/2404.01932](https://arxiv.org/abs/2404.01932)

    本研究探索了多模态VAE如何在无监督机器人操作任务中实现，提出了一个新的模型训练方法，可以使模拟器中的模型性能提高55%。

    

    在这项工作中，我们关注机器人操作领域中无监督的视觉-语言-动作映射。我们探讨了多模态变分自动编码器（VAE）在模拟环境中如何被应用于无监督机器人操作任务中，并提出了一个改进模型性能的模型不变式训练方法，可以使模拟器中的模型性能提高高达55%。

    arXiv:2404.01932v1 Announce Type: cross  Abstract: In this work, we focus on unsupervised vision-language-action mapping in the area of robotic manipulation. Recently, multiple approaches employing pre-trained large language and vision models have been proposed for this task. However, they are computationally demanding and require careful fine-tuning of the produced outputs. A more lightweight alternative would be the implementation of multimodal Variational Autoencoders (VAEs) which can extract the latent features of the data and integrate them into a joint representation, as has been demonstrated mostly on image-image or image-text data for the state-of-the-art models. Here we explore whether and how can multimodal VAEs be employed in unsupervised robotic manipulation tasks in a simulated environment. Based on the obtained results, we propose a model-invariant training alternative that improves the models' performance in a simulator by up to 55%. Moreover, we systematically evaluate 
    
[^2]: LC-Tsalis-INF: 广义最佳双赢线性背景强化型赌博机

    LC-Tsalis-INF: Generalized Best-of-Both-Worlds Linear Contextual Bandits

    [https://arxiv.org/abs/2403.03219](https://arxiv.org/abs/2403.03219)

    该研究提出了一种广义最佳双赢线性背景强化型赌博机算法，能够在次优性差距受到下界限制时遗憾为$O(\log(T))$。同时引入了边缘条件来描述次优性差距对问题难度的影响。

    

    本研究考虑具有独立同分布（i.i.d.）背景的线性背景强化型赌博机问题。在这个问题中，现有研究提出了最佳双赢（BoBW）算法，其遗憾在随机区域中满足$O(\log^2(T))$，其中$T$为回合数，其次优性差距由正常数下界，同时在对抗性区域中满足$O(\sqrt{T})$。然而，对$T$的依赖仍有改进空间，并且次优性差距的假设可以放宽。针对这个问题，本研究提出了一个算法，当次优性差距受到下界限制时，其遗憾满足$O(\log(T))$。此外，我们引入了一个边缘条件，即对次优性差距的一个更温和的假设。该条件使用参数$\beta \in (0, \infty]$表征与次优性差距相关的问题难度。然后我们证明该算法的遗憾满足$O\left(\

    arXiv:2403.03219v1 Announce Type: new  Abstract: This study considers the linear contextual bandit problem with independent and identically distributed (i.i.d.) contexts. In this problem, existing studies have proposed Best-of-Both-Worlds (BoBW) algorithms whose regrets satisfy $O(\log^2(T))$ for the number of rounds $T$ in a stochastic regime with a suboptimality gap lower-bounded by a positive constant, while satisfying $O(\sqrt{T})$ in an adversarial regime. However, the dependency on $T$ has room for improvement, and the suboptimality-gap assumption can be relaxed. For this issue, this study proposes an algorithm whose regret satisfies $O(\log(T))$ in the setting when the suboptimality gap is lower-bounded. Furthermore, we introduce a margin condition, a milder assumption on the suboptimality gap. That condition characterizes the problem difficulty linked to the suboptimality gap using a parameter $\beta \in (0, \infty]$. We then show that the algorithm's regret satisfies $O\left(\
    
[^3]: FedStruct：联合解耦学习在互联图上

    FedStruct: Federated Decoupled Learning over Interconnected Graphs

    [https://arxiv.org/abs/2402.19163](https://arxiv.org/abs/2402.19163)

    FedStruct提出了一种新的框架，利用深层结构依赖关系在互联图上进行联合解耦学习，有效地维护隐私并捕捉节点间的依赖关系。

    

    我们解决了分布在多个客户端上的图结构数据上的联合学习挑战。具体来说，我们关注互联子图的普遍情况，其中不同客户端之间的相互连接起着关键作用。我们提出了针对这种情况的一种新颖框架，名为FedStruct，它利用深层结构依赖关系。为了维护隐私，与现有方法不同，FedStruct消除了在客户端之间共享或生成敏感节点特征或嵌入的必要性。相反，它利用显式全局图结构信息来捕捉节点间的依赖关系。我们通过在六个数据集上进行的实验结果验证了FedStruct的有效性，展示了在各种情况下（包括不同数据分区方法、不同标签可用性以及客户个数的）接近于集中式方法的性能。

    arXiv:2402.19163v1 Announce Type: new  Abstract: We address the challenge of federated learning on graph-structured data distributed across multiple clients. Specifically, we focus on the prevalent scenario of interconnected subgraphs, where inter-connections between different clients play a critical role. We present a novel framework for this scenario, named FedStruct, that harnesses deep structural dependencies. To uphold privacy, unlike existing methods, FedStruct eliminates the necessity of sharing or generating sensitive node features or embeddings among clients. Instead, it leverages explicit global graph structure information to capture inter-node dependencies. We validate the effectiveness of FedStruct through experimental results conducted on six datasets for semi-supervised node classification, showcasing performance close to the centralized approach across various scenarios, including different data partitioning methods, varying levels of label availability, and number of cl
    
[^4]: InterpretCC: 适于解释的神经网络的条件计算

    InterpretCC: Conditional Computation for Inherently Interpretable Neural Networks

    [https://arxiv.org/abs/2402.02933](https://arxiv.org/abs/2402.02933)

    InterpretCC是一种新的解释性神经网络模型，通过条件计算和稀疏激活特征，在保持性能的同时实现了人类中心的解释能力。该模型适用于需要可信解释、可操作解释和准确预测的人类面向领域。

    

    神经网络的真实世界解释性在三个方面之间存在权衡：1）需要人类信任解释的近似（例如事后方法）；2）削弱了解释的可理解性（例如自动识别的特征掩码）；3）削弱了模型性能（例如决策树）。这些缺点对于面向人类的领域（如教育、医疗保健或自然语言）是不可接受的，这些领域需要可信的解释、可操作的解释和准确的预测。在这项工作中，我们提出了InterpretCC（可解释的条件计算），这是一种可解释性的设计神经网络系列，通过在预测之前自适应和稀疏地激活特征，确保人类中心的可解释性，同时保持与最先进模型相当的性能。我们将这个思想扩展为可解释的专家混合模型，允许人们离散地指定兴趣话题。

    Real-world interpretability for neural networks is a tradeoff between three concerns: 1) it requires humans to trust the explanation approximation (e.g. post-hoc approaches), 2) it compromises the understandability of the explanation (e.g. automatically identified feature masks), and 3) it compromises the model performance (e.g. decision trees). These shortcomings are unacceptable for human-facing domains, like education, healthcare, or natural language, which require trustworthy explanations, actionable interpretations, and accurate predictions. In this work, we present InterpretCC (interpretable conditional computation), a family of interpretable-by-design neural networks that guarantee human-centric interpretability while maintaining comparable performance to state-of-the-art models by adaptively and sparsely activating features before prediction. We extend this idea into an interpretable mixture-of-experts model, that allows humans to specify topics of interest, discretely separate
    
[^5]: 元共训练：两种视角优于一种

    Meta Co-Training: Two Views are Better than One

    [https://arxiv.org/abs/2311.18083](https://arxiv.org/abs/2311.18083)

    元共训练通过在数据上构建不同的视角，并利用未标记数据进行共同训练，提高了半监督学习的性能。

    

    在许多实际的计算机视觉场景中，未标记的数据很多，但标签却稀缺且难以获得。因此，半监督学习利用未标记的数据提升监督分类器的性能已经在最近的文献中引起了重要的关注。其中一种主要的半监督算法是共训练。在共训练中，两种不同的模型利用数据的不同独立和足够的“视角”来共同进行更好的预测。在共训练过程中，每个模型在未标记的数据点上创建伪标签，用于改进另一个模型的性能。我们展示了在常见情况下，当独立视角不可用时，我们可以使用预训练模型来廉价地构建这些视角。在构建的视角上进行共训练可以提高性能，优于我们构建的任何单个视角，并且与半监督学习中的最新方法性能相当，但具有一些不可取之处。

    In many practical computer vision scenarios unlabeled data is plentiful, but labels are scarce and difficult to obtain. As a result, semi-supervised learning which leverages unlabeled data to boost the performance of supervised classifiers have received significant attention in recent literature. One major class of semi-supervised algorithms is co-training. In co-training two different models leverage different independent and sufficient "views" of the data to jointly make better predictions. During co-training each model creates pseudo labels on unlabeled points which are used to improve the other model. We show that in the common case when independent views are not available we can construct such views inexpensively using pre-trained models. Co-training on the constructed views yields a performance improvement over any of the individual views we construct and performance comparable with recent approaches in semi-supervised learning, but has some undesirable properties. To alleviate t
    
[^6]: LMM辅助的一致性嵌入下乳腺癌治疗目标分割

    LMM-Assisted Breast Cancer Treatment Target Segmentation with Consistency Embedding

    [https://arxiv.org/abs/2311.15876](https://arxiv.org/abs/2311.15876)

    RO-LMM是一个针对放射肿瘤学领域设计的多功能大型多模型，提出了一种Consistency Embedding Fine-Tuning（CEFTune）技术，使其能够在保持处理干净输入能力的同时提升对嘈杂输入的鲁棒性，用于放射治疗计划和目标体积分割。

    

    人工智能的最新进展深刻影响了医学领域，为降低临床工作量提供了工具。然而，大多数人工智能模型受限于执行单模式任务，与医学专业人员所使用的综合方法形成鲜明对比。为解决这一问题，本文介绍了RO-LMM，一个专为放射肿瘤学领域设计的多功能大型多模型（LMM）。该模型涵盖了临床工作流中的一系列任务，擅长临床报告摘要、放疗治疗计划建议和计划引导的目标体积分割。为了执行连续的临床任务，我们进一步提出了一种新颖的一致性嵌入微调（CEFTune）技术，提升了LMM对嘈杂输入的鲁棒性，同时保持了处理干净输入的能力，并将该概念转化为LMM驱动的分割框架，即一致性嵌入S。

    arXiv:2311.15876v2 Announce Type: replace-cross  Abstract: Recent advancements in Artificial Intelligence (AI) have profoundly influenced medical fields, by providing tools to reduce clinical workloads. However, most AI models are constrained to execute unimodal tasks, in stark contrast to the comprehensive approaches utilized by medical professionals. To address this, here we present RO-LMM, a multi-purpose large multimodal model (LMM) tailored for the field of radiation oncology. This model covers series of tasks within clinical workflow, adept at clinical report summarization, radiation treatment plan suggestion, and plan-guided target volume segmentation. In particular, to perform consecutive clinical tasks, we further present a novel Consistency Embedding Fine-Tuning (CEFTune) technique, which boosts LMM's robustness to noisy inputs while preserving the capability of handling clean inputs, and transform this concept into LMM-driven segmentation framework as Consistency Embedding S
    
[^7]: 超越RMSE和MAE：引入EAUC来揭示偏见和不公平的迪亚德回归模型中的隐藏因素

    Beyond RMSE and MAE: Introducing EAUC to unmask hidden bias and unfairness in dyadic regression models. (arXiv:2401.10690v1 [cs.LG])

    [http://arxiv.org/abs/2401.10690](http://arxiv.org/abs/2401.10690)

    本研究引入EAUC作为一种新的度量标准，用以揭示迪亚德回归模型中隐藏的偏见和不公平问题。传统的全局错误度量标准如RMSE和MAE无法捕捉到这种问题。

    

    迪亚德回归模型用于预测一对实体的实值结果，在许多领域中都是基础的（例如，在推荐系统中预测用户对产品的评分），在许多其他领域中也有许多潜力但尚未深入探索（例如，在个性化药理学中近似确定患者的适当剂量）。本研究中，我们证明个体实体观察值分布的非均匀性导致了最先进模型中的严重偏见预测，偏向于实体的观察过去值的平均值，并在另类但同样重要的情况下提供比随机预测更差的预测能力。我们表明，全局错误度量标准如均方根误差（RMSE）和平均绝对误差（MAE）不足以捕捉到这种现象，我们将其命名为另类偏见，并引入另类-曲线下面积（EAUC）作为一个新的补充度量，可以在所有研究的模型中量化它。

    Dyadic regression models, which predict real-valued outcomes for pairs of entities, are fundamental in many domains (e.g. predicting the rating of a user to a product in Recommender Systems) and promising and under exploration in many others (e.g. approximating the adequate dosage of a drug for a patient in personalized pharmacology). In this work, we demonstrate that non-uniformity in the observed value distributions of individual entities leads to severely biased predictions in state-of-the-art models, skewing predictions towards the average of observed past values for the entity and providing worse-than-random predictive power in eccentric yet equally important cases. We show that the usage of global error metrics like Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE) is insufficient to capture this phenomenon, which we name eccentricity bias, and we introduce Eccentricity-Area Under the Curve (EAUC) as a new complementary metric that can quantify it in all studied models
    
[^8]: 一个简单的方法在世界模型中实现新颖性检测

    A Simple Way to Incorporate Novelty Detection in World Models. (arXiv:2310.08731v1 [cs.AI])

    [http://arxiv.org/abs/2310.08731](http://arxiv.org/abs/2310.08731)

    本文提出了一个简单的方法，通过利用世界模型幻觉状态和真实观察状态的不匹配性作为异常分数，将新颖性检测纳入世界模型强化学习代理中。在新环境中与传统方法相比，我们的工作具有优势。

    

    使用世界模型进行强化学习已经取得了显著的成功。然而，当世界机制或属性发生突然变化时，代理的性能和可靠性可能会显著下降。我们将视觉属性或状态转换的突变称为“新颖性”。在生成的世界模型框架中实施新颖性检测是保护部署时代理的关键任务。在本文中，我们提出了一种简单的边界方法，用于将新颖性检测纳入世界模型强化学习代理中，通过利用世界模型幻觉状态和真实观察状态的不匹配性作为异常分数。首先，我们提供了与序列决策相关的新颖性检测本体论，然后我们提供了在代理在世界模型中学习的转换分布中检测新颖性的有效方法。最后，我们展示了我们的工作在新环境中与传统方法相比的优势。

    Reinforcement learning (RL) using world models has found significant recent successes. However, when a sudden change to world mechanics or properties occurs then agent performance and reliability can dramatically decline. We refer to the sudden change in visual properties or state transitions as {\em novelties}. Implementing novelty detection within generated world model frameworks is a crucial task for protecting the agent when deployed. In this paper, we propose straightforward bounding approaches to incorporate novelty detection into world model RL agents, by utilizing the misalignment of the world model's hallucinated states and the true observed states as an anomaly score. We first provide an ontology of novelty detection relevant to sequential decision making, then we provide effective approaches to detecting novelties in a distribution of transitions learned by an agent in a world model. Finally, we show the advantage of our work in a novel environment compared to traditional ma
    
[^9]: 具有差分隐私(分组)结果的因果推断

    Causal Inference with Differentially Private (Clustered) Outcomes. (arXiv:2308.00957v1 [stat.ML])

    [http://arxiv.org/abs/2308.00957](http://arxiv.org/abs/2308.00957)

    本文提出了一种新的差分隐私机制"Cluster-DP"，它在保证隐私的同时利用数据的聚类结构，从而实现了更强的隐私保证和较低的方差，可以用于进行因果分析。

    

    从随机实验中估计因果效应只有在参与者同意透露他们可能敏感的响应时才可行。在确保隐私的许多方法中，标签差分隐私是一种广泛使用的算法隐私保证度量，可以鼓励参与者分享响应而不会面临去匿名化的风险。许多差分隐私机制会向原始数据集中注入噪音来实现这种隐私保证，这会增加大多数统计估计量的方差，使得精确测量因果效应变得困难：从差分隐私数据进行因果分析存在着固有的隐私-方差权衡。为了实现更强隐私保证的较低方差，我们提出了一种新的差分隐私机制"Cluster-DP"，它利用数据的任何给定的聚类结构，同时仍然允许对因果效应进行估计。

    Estimating causal effects from randomized experiments is only feasible if participants agree to reveal their potentially sensitive responses. Of the many ways of ensuring privacy, label differential privacy is a widely used measure of an algorithm's privacy guarantee, which might encourage participants to share responses without running the risk of de-anonymization. Many differentially private mechanisms inject noise into the original data-set to achieve this privacy guarantee, which increases the variance of most statistical estimators and makes the precise measurement of causal effects difficult: there exists a fundamental privacy-variance trade-off to performing causal analyses from differentially private data. With the aim of achieving lower variance for stronger privacy guarantees, we suggest a new differential privacy mechanism, "Cluster-DP", which leverages any given cluster structure of the data while still allowing for the estimation of causal effects. We show that, depending 
    
[^10]: 变分激励噪声：噪声如何改进模型

    Variational Positive-incentive Noise: How Noise Benefits Models. (arXiv:2306.07651v1 [cs.LG])

    [http://arxiv.org/abs/2306.07651](http://arxiv.org/abs/2306.07651)

    本文研究了如何通过正激励噪声框架下的随机噪声使经典模型受益，并提出了变分Pi-Noise，它可以在不改变原始模型结构的情况下增强和简化模型。

    

    大量研究旨在减轻由于负面噪声的基本假设而导致的噪声影响。但是，一些现有的研究表明，这种假设并不总是成立的。本文研究了如何在正激励噪声（Pi-Noise）框架下通过随机噪声使经典模型受益。由于Pi-Noise的理想目标是难以实现的，我们提出了对其变分下界进行优化的变分Pi-Noise（VPN），通过变分推断，设计了一个VPN生成器来增强基础模型并简化基础模型的推断，而不改变基础模型的架构。由于基础模型和VPN生成器的独立设计， VPN生成器可以与大多数现有模型一起使用。从实验结果来看，所提出的VPN生成器可以改进基本模型。值得称赞的是，训练有素的变分VPN生成器更喜欢独立密集型噪声。（翻译有删减）

    A large number of works aim to alleviate the impact of noise due to an underlying conventional assumption of the negative role of noise. However, some existing works show that the assumption does not always hold. In this paper, we investigate how to benefit the classical models by random noise under the framework of Positive-incentive Noise (Pi-Noise). Since the ideal objective of Pi-Noise is intractable, we propose to optimize its variational bound instead, namely variational Pi-Noise (VPN). With the variational inference, a VPN generator implemented by neural networks is designed for enhancing base models and simplifying the inference of base models, without changing the architecture of base models. Benefiting from the independent design of base models and VPN generators, the VPN generator can work with most existing models. From the experiments, it is shown that the proposed VPN generator can improve the base models. It is appealing that the trained variational VPN generator prefers
    
[^11]: 图神经网络中的改连是否真正有用？

    Is Rewiring Actually Helpful in Graph Neural Networks?. (arXiv:2305.19717v1 [cs.LG])

    [http://arxiv.org/abs/2305.19717](http://arxiv.org/abs/2305.19717)

    本文研究了图神经网络中的改连方法是否有用，提出了一种新的评估设置，并在真实世界的节点和图分类任务上进行了系统的实验比较。

    

    图神经网络通过执行多个消息传递步骤来计算节点表示，这些步骤包括节点特征的本地聚合。但是，深层模型能够利用节点之间更长距离的交互的问题受到了过度平滑和过度压缩的影响。而后者归因于指导消息传递的图拓扑，导致节点表示对包含在远程节点上的信息不敏感。许多改连方法已被提出来解决或减轻这个问题。然而，由于过度压缩与其他与模型训练密切相关的问题（如消失的梯度）相耦合，所以正确评估这些方法的好处是困难的。因此，我们提出了一种基于消息传递模型的评估设置，这些模型不需要训练即可计算节点和图表示。我们在真实世界的节点和图分类任务上进行了系统的实验比较。

    Graph neural networks compute node representations by performing multiple message-passing steps that consist in local aggregations of node features. Having deep models that can leverage longer-range interactions between nodes is hindered by the issues of over-smoothing and over-squashing. In particular, the latter is attributed to the graph topology which guides the message-passing, causing a node representation to become insensitive to information contained at distant nodes. Many graph rewiring methods have been proposed to remedy or mitigate this problem. However, properly evaluating the benefits of these methods is made difficult by the coupling of over-squashing with other issues strictly related to model training, such as vanishing gradients. Therefore, we propose an evaluation setting based on message-passing models that do not require training to compute node and graph representations. We perform a systematic experimental comparison on real-world node and graph classification ta
    
[^12]: 从飞行轨迹和程序中推断终端空域交通模型

    Inferring Traffic Models in Terminal Airspace from Flight Tracks and Procedures. (arXiv:2303.09981v1 [cs.LG])

    [http://arxiv.org/abs/2303.09981](http://arxiv.org/abs/2303.09981)

    本文提出了一种通过收集飞行轨迹和程序数据学习飞行器行为变异性的概率模型，并且可以生成涉及任意数量飞行器的交通模型。

    

    真实的航空器轨迹模型对于空中交通管理（ATM）系统设计和验证很有用。仪表飞行规则（IFR）下操作的飞行器模型需要捕捉飞行器按照标准飞行程序的固有变异性。飞行器行为的变异性在不同的飞行阶段之间各不相同。本文提出了一种概率模型，可以从程序数据和从雷达监视数据收集的飞行轨迹中学习变异性。 对于每个段落，使用高斯混合模型学习飞行器轨迹与其程序之间的偏差。给定新的程序，我们可以通过从经过训练的高斯分布中抽样一系列偏差，并使用偏差和程序重构飞行器轨迹来生成合成轨迹。我们将这种方法扩展到捕捉飞行器之间的成对相关性，并展示如何使用成对模型来生成涉及任意数量飞行器的交通模型。

    Realistic aircraft trajectory models are useful in the design and validation of air traffic management (ATM) systems. Models of aircraft operated under instrument flight rules (IFR) require capturing the variability inherent in how aircraft follow standard flight procedures. The variability in aircraft behavior varies among flight stages. In this paper, we propose a probabilistic model that can learn the variability from the procedural data and flight tracks collected from radar surveillance data. For each segment, a Gaussian mixture model is used to learn the deviations of aircraft trajectories from their procedures. Given new procedures, we can generate synthetic trajectories by sampling a series of deviations from the trained Gaussian distributions and reconstructing the aircraft trajectory using the deviations and the procedures. We extend this method to capture pairwise correlations between aircraft and show how a pairwise model can be used to generate traffic involving an arbitra
    

