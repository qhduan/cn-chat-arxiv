# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Byzantine-resilient Federated Learning With Adaptivity to Data Heterogeneity](https://arxiv.org/abs/2403.13374) | 通过提出新的Robust Average Gradient Algorithm（RAGA），本研究在联邦学习中解决了恶意拜占庭攻击和数据异构性的问题，实现了在非凸损失函数和异构数据集上的收敛性分析，并展示了RAGA的良好收敛性能。 |
| [^2] | [Spurious Correlations in Machine Learning: A Survey](https://arxiv.org/abs/2402.12715) | 机器学习系统对输入中偏见特征与标签之间的虚假相关性敏感，本文回顾了解决这一问题的最新方法，同时总结了数据集、基准和度量标准，并讨论了未来研究挑战。 |
| [^3] | [Off-Policy Evaluation in Markov Decision Processes under Weak Distributional Overlap](https://arxiv.org/abs/2402.08201) | 本文研究了弱分布重叠下马尔可夫决策过程中的离策略评估问题，并提出了一种截断双重稳健（TDR）估计器，在这种情况下表现良好。 |
| [^4] | [EUGENE: Explainable Unsupervised Approximation of Graph Edit Distance](https://arxiv.org/abs/2402.05885) | EUGENE是一种可解释的无监督图编辑距离近似方法，可以通过生成编辑路径来近似计算图编辑距离，同时消除了ground-truth生成和数据特定训练的需求。 |
| [^5] | [Machines Do See Color: A Guideline to Classify Different Forms of Racist Discourse in Large Corpora.](http://arxiv.org/abs/2401.09333) | 本文提供了一个逐步可推广的准则，用于在大规模语料库中识别和分类不同形式的种族主义言论。通过对种族主义的概念化和上下文化，以及使用XLM-R和XLM-R-Racismo模型，我们展示了在大规模语料库中进行种族主义分类的优势。 |
| [^6] | [Symbolic Imitation Learning: From Black-Box to Explainable Driving Policies.](http://arxiv.org/abs/2309.16025) | 本文介绍了一种名为符号化模仿学习（SIL）的方法，通过引入归纳逻辑编程（ILP）来学习从现有数据集中获取透明、可解释和泛化的驾驶策略。与传统的基于深度神经网络的模仿学习方法相比，SIL不仅提高了驾驶策略的可解释性，还显著改进了它们在各种驾驶情况下的适用性。 |
| [^7] | [Few-Shot Personalized Saliency Prediction Using Tensor Regression for Preserving Structural Global Information.](http://arxiv.org/abs/2307.02799) | 本文提出了一种使用张量回归进行少样本个性化显著性预测的方法，以保留个性化显著性图的结构全局信息。 |
| [^8] | [A Double Machine Learning Approach to Combining Experimental and Observational Data.](http://arxiv.org/abs/2307.01449) | 这种双机器学习方法将实验和观测研究结合起来，能够测试假设的违反情况并一致估计处理效应。它提供了半参数高效的处理效应估计器。这种方法在实际环境中是可行的。 |
| [^9] | [Policy Gradient Algorithms for Robust MDPs with Non-Rectangular Uncertainty Sets.](http://arxiv.org/abs/2305.19004) | 本文提出了针对具有非矩形不确定性集的强健MDP的策略梯度算法，并开发了投射Langevin动力学算法和确定性策略梯度方法。数值实验展示了这些算法的有效性。 |
| [^10] | [Improving Multi-task Learning via Seeking Task-based Flat Regions.](http://arxiv.org/abs/2211.13723) | 通过寻找基于任务的平坦区域，可以改进多任务学习并提高模型性能，但需要正确使用正则化技术以避免次优解。 |
| [^11] | [Explicit Second-Order Min-Max Optimization Methods with Optimal Convergence Guarantee.](http://arxiv.org/abs/2210.12860) | 本文提出了一种具有最优收敛保证的显式二阶最小最大优化方法，用于解决凸凹无约束最小最大优化问题。该方法利用二阶信息加速额外梯度方法，并且在迭代过程中保持在有界集内，达到了与理论下界相匹配的收敛速度。 |
| [^12] | [TabText: A Flexible and Contextual Approach to Tabular Data Representation.](http://arxiv.org/abs/2206.10381) | TabText是一种处理和特征提取框架，通过转换内容为语言并利用预训练的大型语言模型，从表格数据中提取上下文信息。通过应用TabText框架可以生成高性能且简单的机器学习基准模型，减少数据预处理的工作量。该框架在医疗预测任务中展现出良好的效果。 |

# 详细

[^1]: 具有对数据异构性的自适应的拜占庭弹性联邦学习

    Byzantine-resilient Federated Learning With Adaptivity to Data Heterogeneity

    [https://arxiv.org/abs/2403.13374](https://arxiv.org/abs/2403.13374)

    通过提出新的Robust Average Gradient Algorithm（RAGA），本研究在联邦学习中解决了恶意拜占庭攻击和数据异构性的问题，实现了在非凸损失函数和异构数据集上的收敛性分析，并展示了RAGA的良好收敛性能。

    

    本文处理了在存在恶意拜占庭攻击和数据异构性的情况下的联邦学习（FL）。提出了一种新颖的鲁棒平均梯度算法（RAGA），该算法利用几何中位数进行聚合，并可以自由选择本地更新的轮数。与大多数现有的弹性方法不同，这些方法基于强凸损失函数或均匀分布的数据集进行收敛分析，我们进行了对强凸和非凸损失函数在异构数据集上的收敛分析。根据我们的理论分析，只要恶意用户数据集的比例小于一半，RAGA就可以以$\mathcal{O}({1}/{T^{2/3- \delta}})$的速度实现非凸损失函数的收敛，其中$T$为迭代次数，$\delta \in (0, 2/3)$，对于强凸损失函数则呈线性收敛。此外，稳定点或全局最优解

    arXiv:2403.13374v1 Announce Type: new  Abstract: This paper deals with federated learning (FL) in the presence of malicious Byzantine attacks and data heterogeneity. A novel Robust Average Gradient Algorithm (RAGA) is proposed, which leverages the geometric median for aggregation and can freely select the round number for local updating. Different from most existing resilient approaches, which perform convergence analysis based on strongly-convex loss function or homogeneously distributed dataset, we conduct convergence analysis for not only strongly-convex but also non-convex loss function over heterogeneous dataset. According to our theoretical analysis, as long as the fraction of dataset from malicious users is less than half, RAGA can achieve convergence at rate $\mathcal{O}({1}/{T^{2/3- \delta}})$ where $T$ is the iteration number and $\delta \in (0, 2/3)$ for non-convex loss function, and at linear rate for strongly-convex loss function. Moreover, stationary point or global optim
    
[^2]: 机器学习中的虚假相关性：一项调查

    Spurious Correlations in Machine Learning: A Survey

    [https://arxiv.org/abs/2402.12715](https://arxiv.org/abs/2402.12715)

    机器学习系统对输入中偏见特征与标签之间的虚假相关性敏感，本文回顾了解决这一问题的最新方法，同时总结了数据集、基准和度量标准，并讨论了未来研究挑战。

    

    众所周知，机器学习系统对输入中偏见特征（例如背景、纹理和次要对象）与相应标签之间的虚假相关性敏感。这些特征及其与标签的相关性被称为“虚假”，因为它们往往随着真实世界数据分布的变化而改变，这可能对模型的泛化能力和鲁棒性产生负面影响。在这项调查中，我们全面审查了这一问题，提供了一个关于解决机器学习模型中虚假相关性的当前最先进方法的分类法。此外，我们总结了现有的数据集、基准和度量标准，以帮助未来的研究。本文最后讨论了这一领域的最新进展和未来研究挑战，旨在为相关领域的研究人员提供宝贵的见解。

    arXiv:2402.12715v1 Announce Type: new  Abstract: Machine learning systems are known to be sensitive to spurious correlations between biased features of the inputs (e.g., background, texture, and secondary objects) and the corresponding labels. These features and their correlations with the labels are known as "spurious" because they tend to change with shifts in real-world data distributions, which can negatively impact the model's generalization and robustness. In this survey, we provide a comprehensive review of this issue, along with a taxonomy of current state-of-the-art methods for addressing spurious correlations in machine learning models. Additionally, we summarize existing datasets, benchmarks, and metrics to aid future research. The paper concludes with a discussion of the recent advancements and future research challenges in this field, aiming to provide valuable insights for researchers in the related domains.
    
[^3]: 弱分布重叠下马尔可夫决策过程中的离策略评估

    Off-Policy Evaluation in Markov Decision Processes under Weak Distributional Overlap

    [https://arxiv.org/abs/2402.08201](https://arxiv.org/abs/2402.08201)

    本文研究了弱分布重叠下马尔可夫决策过程中的离策略评估问题，并提出了一种截断双重稳健（TDR）估计器，在这种情况下表现良好。

    

    在马尔可夫决策过程（MDP）中，双重稳健方法在序列可忽略性下对离策略评估具有很大的潜力：它们已经证明了随着时长T的收敛速度为$1/\sqrt{T}$，在大样本中具有统计效率，并且可以通过标准强化学习技术执行预估任务，具有模块化实现的能力。然而，现有结果在很大程度上使用了强分布重叠假设，即目标政策和数据收集政策的稳态分布相差在有限因子内，而这个假设通常只在MDP的状态空间有界时才可信。在本文中，我们重新审视了在弱分布重叠概念下的MDP离策略评估任务，并引入了一类截断双重稳健（TDR）估计器，在这种情况下表现良好。当目标和数据收集的分布比率有界时，我们证明了这些估计器的一致性。

    Doubly robust methods hold considerable promise for off-policy evaluation in Markov decision processes (MDPs) under sequential ignorability: They have been shown to converge as $1/\sqrt{T}$ with the horizon $T$, to be statistically efficient in large samples, and to allow for modular implementation where preliminary estimation tasks can be executed using standard reinforcement learning techniques. Existing results, however, make heavy use of a strong distributional overlap assumption whereby the stationary distributions of the target policy and the data-collection policy are within a bounded factor of each other -- and this assumption is typically only credible when the state space of the MDP is bounded. In this paper, we re-visit the task of off-policy evaluation in MDPs under a weaker notion of distributional overlap, and introduce a class of truncated doubly robust (TDR) estimators which we find to perform well in this setting. When the distribution ratio of the target and data-coll
    
[^4]: EUGENE: 可解释的无监督图编辑距离近似方法

    EUGENE: Explainable Unsupervised Approximation of Graph Edit Distance

    [https://arxiv.org/abs/2402.05885](https://arxiv.org/abs/2402.05885)

    EUGENE是一种可解释的无监督图编辑距离近似方法，可以通过生成编辑路径来近似计算图编辑距离，同时消除了ground-truth生成和数据特定训练的需求。

    

    在生物学、化学、推荐系统和社交网络分析等领域，需要识别与查询图结构距离较小的图形。在多种测量图间距离的方法中，图编辑距离（GED）因其可理解性而被认为是首选，但其计算的NP难度限制了其应用。目前最先进的GED近似方法主要采用神经方法，然而，这些方法（i）缺少与近似的GED对应的解释性编辑路径；（ii）需要通过NP难问题生成ground-truth GED进行训练；（iii）需要在每个数据集上进行独立训练。本文提出了一种高效的代数无监督方法EUGENE，它近似计算GED并生成与近似成本对应的编辑路径，同时消除了生成ground-truth和数据特定训练的需求。广泛的实验评估表明，EUGENE的上述优点并不以效力为代价。

    The need to identify graphs having small structural distance from a query arises in biology, chemistry, recommender systems, and social network analysis. Among several methods to measure inter graph distance, Graph Edit Distance (GED) is preferred for its comprehensibility, yet hindered by the NP-hardness of its computation. State-of-the-art GED approximations predominantly employ neural methods, which, however, (i) lack an explanatory edit path corresponding to the approximated GED; (ii) require the NP-hard generation of ground-truth GEDs for training; and (iii) necessitate separate training on each dataset. In this paper, we propose an efficient algebraic unsuper vised method, EUGENE, that approximates GED and yields edit paths corresponding to the approx imated cost, while eliminating the need for ground truth generation and data-specific training. Extensive experimental evaluation demonstrates that the aforementioned benefits of EUGENE do not come at the cost of efficacy. Specifica
    
[^5]: 机器能够看到颜色：大规模语料库中分类不同形式种族主义言论的准则

    Machines Do See Color: A Guideline to Classify Different Forms of Racist Discourse in Large Corpora. (arXiv:2401.09333v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2401.09333](http://arxiv.org/abs/2401.09333)

    本文提供了一个逐步可推广的准则，用于在大规模语料库中识别和分类不同形式的种族主义言论。通过对种族主义的概念化和上下文化，以及使用XLM-R和XLM-R-Racismo模型，我们展示了在大规模语料库中进行种族主义分类的优势。

    

    目前识别和分类文本中的种族主义语言的方法主要依赖小规模的质性方法或大规模的方法，专注于明显的种族主义言论。本文提供了一个逐步可推广的准则，用于在大规模语料库中识别和分类不同形式的种族主义言论。在我们的方法中，我们首先将种族主义及其不同表现形式进行概念化。然后，我们将这些种族主义表现形式置于感兴趣的时间和地点背景下，以便研究人员能够识别它们的话语形式。最后，我们应用了XLM-RoBERTa（XLM-R），这是一个具有先进上下文理解能力的跨语言监督文本分类模型。我们展示了XLM-R和XLM-R-Racismo（我们的预训练模型）在大规模语料库中对种族主义进行分类的性能优于其他最先进的方法。我们通过使用涉及2018年至2021年厄瓜多尔本土群体的推文语料库来说明我们的方法。

    Current methods to identify and classify racist language in text rely on small-n qualitative approaches or large-n approaches focusing exclusively on overt forms of racist discourse. This article provides a step-by-step generalizable guideline to identify and classify different forms of racist discourse in large corpora. In our approach, we start by conceptualizing racism and its different manifestations. We then contextualize these racist manifestations to the time and place of interest, which allows researchers to identify their discursive form. Finally, we apply XLM-RoBERTa (XLM-R), a cross-lingual model for supervised text classification with a cutting-edge contextual understanding of text. We show that XLM-R and XLM-R-Racismo, our pretrained model, outperform other state-of-the-art approaches in classifying racism in large corpora. We illustrate our approach using a corpus of tweets relating to the Ecuadorian ind\'igena community between 2018 and 2021.
    
[^6]: 符号化模仿学习：从黑盒到可解释的驾驶策略

    Symbolic Imitation Learning: From Black-Box to Explainable Driving Policies. (arXiv:2309.16025v1 [cs.LG])

    [http://arxiv.org/abs/2309.16025](http://arxiv.org/abs/2309.16025)

    本文介绍了一种名为符号化模仿学习（SIL）的方法，通过引入归纳逻辑编程（ILP）来学习从现有数据集中获取透明、可解释和泛化的驾驶策略。与传统的基于深度神经网络的模仿学习方法相比，SIL不仅提高了驾驶策略的可解释性，还显著改进了它们在各种驾驶情况下的适用性。

    

    当前的模仿学习方法主要基于深度神经网络，提供了从现实世界数据中获取驾驶策略的有效手段，但在可解释性和泛化性方面存在显著局限性。这些缺点在自动驾驶等安全关键应用中尤为令人担忧。本文通过引入符号化模仿学习（SIL），一种使用归纳逻辑编程（ILP）学习从可用数据集中获取透明、可解释和泛化的驾驶策略的创新方法，来解决这些局限性。利用真实世界的highD数据集，我们对我们的方法进行了严格的比较分析，与当前的基于神经网络的模仿学习方法进行了对比。我们的结果表明，SIL不仅提高了驾驶策略的可解释性，还显著提高了它们在各种驾驶情况下的适用性。因此，这项工作为实现更可靠和可解释的驾驶策略打开了一条新的途径。

    Current methods of imitation learning (IL), primarily based on deep neural networks, offer efficient means for obtaining driving policies from real-world data but suffer from significant limitations in interpretability and generalizability. These shortcomings are particularly concerning in safety-critical applications like autonomous driving. In this paper, we address these limitations by introducing Symbolic Imitation Learning (SIL), a groundbreaking method that employs Inductive Logic Programming (ILP) to learn driving policies which are transparent, explainable and generalisable from available datasets. Utilizing the real-world highD dataset, we subject our method to a rigorous comparative analysis against prevailing neural-network-based IL methods. Our results demonstrate that SIL not only enhances the interpretability of driving policies but also significantly improves their applicability across varied driving situations. Hence, this work offers a novel pathway to more reliable an
    
[^7]: 用张量回归进行少样本个性化显著性预测，保留结构全局信息。

    Few-Shot Personalized Saliency Prediction Using Tensor Regression for Preserving Structural Global Information. (arXiv:2307.02799v1 [eess.IV])

    [http://arxiv.org/abs/2307.02799](http://arxiv.org/abs/2307.02799)

    本文提出了一种使用张量回归进行少样本个性化显著性预测的方法，以保留个性化显著性图的结构全局信息。

    

    本文提出了一种使用张量到矩阵回归进行少样本个性化显著性预测的方法，以保留个性化显著性图（PSM）的结构全局信息。与一般的显著性图相比，PSM具有巨大的潜力，因为它的映射指示了个体特定的视觉注意力，对于从凝视区域的异质性中获取个体视觉偏好非常有用。PSM的预测是为了获取未见图像的PSM，但由于个体凝视模式的复杂性，其预测仍然是一项具有挑战性的任务。为了从有限的眼动数据中识别个体凝视模式，先前的方法采用个体之间凝视趋势的相似性。然而，在先前的方法中，PSMs被向量化以适应预测模型，从而忽视了与图像对应的PSMs的结构全局信息。为了自动揭示PSMs之间的关系，我们聚焦于...

    This paper presents a few-shot personalized saliency prediction using tensor-to-matrix regression for preserving the structural global information of personalized saliency maps (PSMs). In contrast to a general saliency map, a PSM has been great potential since its map indicates the person-specific visual attention that is useful for obtaining individual visual preferences from heterogeneity of gazed areas. The PSM prediction is needed for acquiring the PSM for the unseen image, but its prediction is still a challenging task due to the complexity of individual gaze patterns. For recognizing individual gaze patterns from the limited amount of eye-tracking data, the previous methods adopt the similarity of gaze tendency between persons. However, in the previous methods, the PSMs are vectorized for the prediction model. In this way, the structural global information of the PSMs corresponding to the image is ignored. For automatically revealing the relationship between PSMs, we focus on the
    
[^8]: 将实验数据与观测数据结合的双机器学习方法

    A Double Machine Learning Approach to Combining Experimental and Observational Data. (arXiv:2307.01449v1 [stat.ME])

    [http://arxiv.org/abs/2307.01449](http://arxiv.org/abs/2307.01449)

    这种双机器学习方法将实验和观测研究结合起来，能够测试假设的违反情况并一致估计处理效应。它提供了半参数高效的处理效应估计器。这种方法在实际环境中是可行的。

    

    实验和观测研究通常由于无法测试的假设而缺乏有效性。我们提出了一种双机器学习方法，将实验和观测研究结合起来，使从业人员能够测试假设违反情况并一致估计处理效应。我们的框架在较轻的假设下测试外部效度和可忽视性的违反情况。当只有一个假设被违反时，我们提供半参数高效的处理效应估计器。然而，我们的无免费午餐定理强调了准确识别违反的假设对一致的处理效应估计的必要性。我们通过三个实际案例研究展示了我们方法的适用性，并突出了其在实际环境中的相关性。

    Experimental and observational studies often lack validity due to untestable assumptions. We propose a double machine learning approach to combine experimental and observational studies, allowing practitioners to test for assumption violations and estimate treatment effects consistently. Our framework tests for violations of external validity and ignorability under milder assumptions. When only one assumption is violated, we provide semi-parametrically efficient treatment effect estimators. However, our no-free-lunch theorem highlights the necessity of accurately identifying the violated assumption for consistent treatment effect estimation. We demonstrate the applicability of our approach in three real-world case studies, highlighting its relevance for practical settings.
    
[^9]: 非矩形不确定性集的强健MDP的策略梯度算法

    Policy Gradient Algorithms for Robust MDPs with Non-Rectangular Uncertainty Sets. (arXiv:2305.19004v2 [math.OC] UPDATED)

    [http://arxiv.org/abs/2305.19004](http://arxiv.org/abs/2305.19004)

    本文提出了针对具有非矩形不确定性集的强健MDP的策略梯度算法，并开发了投射Langevin动力学算法和确定性策略梯度方法。数值实验展示了这些算法的有效性。

    

    我们为具有非矩形不确定性集的强健无限时域马尔可夫决策过程（MDP）提出了一个策略梯度算法，从而解决了强健MDP文献中的一个开放性挑战。确实，显示统计最优性质并充分利用有限数据的不确定性集往往不是矩形的。不幸的是，对应的强健MDPs不能用动态规划技术解决，并且实际上是可证明的不可解决的。这促使我们开发一个针对强健策略评估问题量身定制的投射Langevin动力学算法，该算法提供全局最优性保证。我们还提出了一种确定性策略梯度方法，该方法近似解决了强健策略评估问题，并证明了近似误差与不确定性集的非矩形度量成比例。数值实验展示了我们的投影Langevin动力学算法可以避免局部最优，而算法是量身定制的。

    We propose a policy gradient algorithm for robust infinite-horizon Markov Decision Processes (MDPs) with non-rectangular uncertainty sets, thereby addressing an open challenge in the robust MDP literature. Indeed, uncertainty sets that display statistical optimality properties and make optimal use of limited data often fail to be rectangular. Unfortunately, the corresponding robust MDPs cannot be solved with dynamic programming techniques and are in fact provably intractable. This prompts us to develop a projected Langevin dynamics algorithm tailored to the robust policy evaluation problem, which offers global optimality guarantees. We also propose a deterministic policy gradient method that solves the robust policy evaluation problem approximately, and we prove that the approximation error scales with a new measure of non-rectangularity of the uncertainty set. Numerical experiments showcase that our projected Langevin dynamics algorithm can escape local optima, while algorithms tailor
    
[^10]: 通过寻找基于任务的平坦区域来改进多任务学习

    Improving Multi-task Learning via Seeking Task-based Flat Regions. (arXiv:2211.13723v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2211.13723](http://arxiv.org/abs/2211.13723)

    通过寻找基于任务的平坦区域，可以改进多任务学习并提高模型性能，但需要正确使用正则化技术以避免次优解。

    

    多任务学习（MTL）是一种广泛使用且强大的学习范式，用于训练深度神经网络，可以通过单个骨干学习多个目标。与单独训练任务相比，MTL显着降低了计算成本，提高了数据效率，并通过利用任务之间的知识来潜在地提高模型性能。因此，它已经被应用于各种应用领域，从计算机视觉到自然语言处理和语音识别。其中，MTL的一个新兴研究方向集中在操纵任务梯度以推导出对所有任务有益的最终梯度下降方向。尽管在许多基准测试上取得了令人印象深刻的结果，但是在实际问题上直接应用这些方法而不使用适当的正则化技术可能会导致次优解。特别是，标准训练在训练数据上最小化经验损失，很容易遭受过拟合问题。

    Multi-Task Learning (MTL) is a widely-used and powerful learning paradigm for training deep neural networks that allows learning more than one objective by a single backbone. Compared to training tasks separately, MTL significantly reduces computational costs, improves data efficiency, and potentially enhances model performance by leveraging knowledge across tasks. Hence, it has been adopted in a variety of applications, ranging from computer vision to natural language processing and speech recognition. Among them, there is an emerging line of work in MTL that focuses on manipulating the task gradient to derive an ultimate gradient descent direction to benefit all tasks. Despite achieving impressive results on many benchmarks, directly applying these approaches without using appropriate regularization techniques might lead to suboptimal solutions on real-world problems. In particular, standard training that minimizes the empirical loss on the training data can easily suffer from overfi
    
[^11]: 具有最优收敛保证的显式二阶最小最大优化方法

    Explicit Second-Order Min-Max Optimization Methods with Optimal Convergence Guarantee. (arXiv:2210.12860v3 [math.OC] UPDATED)

    [http://arxiv.org/abs/2210.12860](http://arxiv.org/abs/2210.12860)

    本文提出了一种具有最优收敛保证的显式二阶最小最大优化方法，用于解决凸凹无约束最小最大优化问题。该方法利用二阶信息加速额外梯度方法，并且在迭代过程中保持在有界集内，达到了与理论下界相匹配的收敛速度。

    

    本文提出并分析了一种精确和不精确正则化牛顿型方法，用于求解凸凹无约束最小最大优化问题的全局鞍点。与一阶方法相比，我们对于二阶最小最大优化方法的理解相对较少，因为利用二阶信息获得全局收敛速度更加复杂。在本文中，我们研究了如何利用二阶信息加速额外梯度方法，即使在不精确的情况下也能实现。具体而言，我们证明了所提出的算法生成的迭代保持在有界集内，并且平均迭代收敛到一个 $\epsilon$-鞍点，所需迭代次数为 $O(\epsilon^{-2/3})$，其中使用了受限间隙函数。我们的算法与该领域已经建立的理论下界相匹配，而且我们的分析提供了一种简单直观的二阶方法收敛分析，不需要任何有界性要求。最后，我们提出了一个

    We propose and analyze exact and inexact regularized Newton-type methods for finding a global saddle point of \emph{convex-concave} unconstrained min-max optimization problems. Compared to first-order methods, our understanding of second-order methods for min-max optimization is relatively limited, as obtaining global rates of convergence with second-order information is much more involved. In this paper, we examine how second-order information can be used to speed up extra-gradient methods, even under inexactness. Specifically, we show that the proposed algorithms generate iterates that remain within a bounded set and the averaged iterates converge to an $\epsilon$-saddle point within $O(\epsilon^{-2/3})$ iterations in terms of a restricted gap function. Our algorithms match the theoretically established lower bound in this context and our analysis provides a simple and intuitive convergence analysis for second-order methods without any boundedness requirements. Finally, we present a 
    
[^12]: TabText:一种灵活和上下文化的表格数据表示方法

    TabText: A Flexible and Contextual Approach to Tabular Data Representation. (arXiv:2206.10381v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2206.10381](http://arxiv.org/abs/2206.10381)

    TabText是一种处理和特征提取框架，通过转换内容为语言并利用预训练的大型语言模型，从表格数据中提取上下文信息。通过应用TabText框架可以生成高性能且简单的机器学习基准模型，减少数据预处理的工作量。该框架在医疗预测任务中展现出良好的效果。

    

    表格数据对于在各个行业中应用机器学习任务至关重要。然而，传统的数据处理方法并没有充分利用表格中所有可用的信息，忽视了重要的上下文信息，如列标题描述。此外，将数据预处理成表格格式仍然是模型开发中一项耗时的瓶颈。本工作引入了TabText，一种处理和特征提取框架，将上下文信息从表格数据结构中提取出来。TabText通过将内容转换为语言，并利用预训练的大型语言模型(LLMs)来解决处理困难。我们在涵盖患者出院、ICU入院和死亡等九个医疗预测任务上评估了我们的框架。我们展示了：1) 应用我们的TabText框架可以生成性能优秀且简单的机器学习基准模型，只需最少的数据预处理；2) 增强预处理后的数据利用预训练语言模型能够提升模型效果

    Tabular data is essential for applying machine learning tasks across various industries. However, traditional data processing methods do not fully utilize all the information available in the tables, ignoring important contextual information such as column header descriptions. In addition, pre-processing data into a tabular format can remain a labor-intensive bottleneck in model development. This work introduces TabText, a processing and feature extraction framework that extracts contextual information from tabular data structures. TabText addresses processing difficulties by converting the content into language and utilizing pre-trained large language models (LLMs). We evaluate our framework on nine healthcare prediction tasks ranging from patient discharge, ICU admission, and mortality. We show that 1) applying our TabText framework enables the generation of high-performing and simple machine learning baseline models with minimal data pre-processing, and 2) augmenting pre-processed t
    

