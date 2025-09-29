# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Biospheric AI](https://arxiv.org/abs/2401.17805) | 生物圈人工智能是一种新的范式，以生态中心主义为基础，旨在捕捉生物圈的复杂性并确保人工智能不会对其造成损害。 |
| [^2] | [Multi-View Hypercomplex Learning for Breast Cancer Screening](https://arxiv.org/abs/2204.05798) | 本文提出了一种基于参数化超复数神经网络的多视图乳腺癌分类方法，能够模拟并利用乳房X光检查的不同视图之间的相关性，从而提高肿瘤识别效果。 |
| [^3] | [Leveraging World Model Disentanglement in Value-Based Multi-Agent Reinforcement Learning.](http://arxiv.org/abs/2309.04615) | 本文提出了一种新的基于模型的多智能体强化学习方法，通过使用模块化的世界模型，减少了多智能体系统中训练的样本复杂性，并成功预测了联合动作价值函数。 |
| [^4] | [Escaping the Sample Trap: Fast and Accurate Epistemic Uncertainty Estimation with Pairwise-Distance Estimators.](http://arxiv.org/abs/2308.13498) | 本文介绍了使用配对距离估计器对集成模型进行认识不确定性估计的新方法，相比于常用的深度学习方法，该方法能够更快速、更准确地在更大的空间和更高维度上估计认识不确定性。 |

# 详细

[^1]: 生物圈人工智能

    Biospheric AI

    [https://arxiv.org/abs/2401.17805](https://arxiv.org/abs/2401.17805)

    生物圈人工智能是一种新的范式，以生态中心主义为基础，旨在捕捉生物圈的复杂性并确保人工智能不会对其造成损害。

    

    在人工智能伦理和价值调整领域中，人类中心主义占据主导地位。这些学科的关注重点仅限于人类价值观，导致其洞察力的深度和广度受限。最近，一些学者已开始尝试扩大到以感知者为中心的视角。我们认为，这两种观点都不足以捕捉生物圈的实际复杂性，并确保人工智能不会对其造成损害。因此，我们提出了一种新的范式——生物圈人工智能，它采用了一个生态中心主义的视角。我们讨论了这种人工智能可能被设计的假设性方式。此外，我们还提出了与生物圈利益一致的现代人工智能模型研究和应用的方向。总的来说，这项工作试图迈出首要步骤，研究人工智能与生物圈之间的相互作用。

    The dominant paradigm in AI ethics and value alignment is highly anthropocentric. The focus of these disciplines is strictly on human values which limits the depth and breadth of their insights. Recently, attempts to expand to a sentientist perspective have been initiated. We argue that neither of these outlooks is sufficient to capture the actual complexity of the biosphere and ensure that AI does not damage it. Thus, we propose a new paradigm -- Biospheric AI that assumes an ecocentric perspective. We discuss hypothetical ways in which such an AI might be designed. Moreover, we give directions for research and application of the modern AI models that would be consistent with the biospheric interests. All in all, this work attempts to take first steps towards a comprehensive program of research that focuses on the interactions between AI and the biosphere.
    
[^2]: 多视图超复数学习用于乳腺癌筛查

    Multi-View Hypercomplex Learning for Breast Cancer Screening

    [https://arxiv.org/abs/2204.05798](https://arxiv.org/abs/2204.05798)

    本文提出了一种基于参数化超复数神经网络的多视图乳腺癌分类方法，能够模拟并利用乳房X光检查的不同视图之间的相关性，从而提高肿瘤识别效果。

    

    传统上，用于乳腺癌分类的深度学习方法执行单视图分析。然而，由于乳腺X-ray图像中包含的相关性，放射科医生同时分析组成乳房X光摄影检查的所有四个视图，这为识别肿瘤提供了关键信息。鉴于此，一些研究已经开始提出多视图方法。然而，在这样的现有架构中，乳房X光图像被独立的卷积分支处理为独立的图像，从而失去了它们之间的相关性。为了克服这些局限性，在本文中，我们提出了一种基于参数化超复数神经网络的多视图乳腺癌分类方法。由于超复数代数特性，我们的网络能够建模并利用组成乳房X光检查的不同视图之间的现有相关性，从而模拟阅片过程。

    arXiv:2204.05798v3 Announce Type: replace-cross  Abstract: Traditionally, deep learning methods for breast cancer classification perform a single-view analysis. However, radiologists simultaneously analyze all four views that compose a mammography exam, owing to the correlations contained in mammography views, which present crucial information for identifying tumors. In light of this, some studies have started to propose multi-view methods. Nevertheless, in such existing architectures, mammogram views are processed as independent images by separate convolutional branches, thus losing correlations among them. To overcome such limitations, in this paper, we propose a methodological approach for multi-view breast cancer classification based on parameterized hypercomplex neural networks. Thanks to hypercomplex algebra properties, our networks are able to model, and thus leverage, existing correlations between the different views that comprise a mammogram, thus mimicking the reading process
    
[^3]: 利用世界模型分解在基于值的多智能体强化学习中的应用

    Leveraging World Model Disentanglement in Value-Based Multi-Agent Reinforcement Learning. (arXiv:2309.04615v1 [cs.LG])

    [http://arxiv.org/abs/2309.04615](http://arxiv.org/abs/2309.04615)

    本文提出了一种新的基于模型的多智能体强化学习方法，通过使用模块化的世界模型，减少了多智能体系统中训练的样本复杂性，并成功预测了联合动作价值函数。

    

    本文提出了一种新颖的基于模型的多智能体强化学习方法，名为Value Decomposition Framework with Disentangled World Model，旨在解决在相同环境中多个智能体达成共同目标时的样本复杂性问题。由于多智能体系统的可扩展性和非平稳性问题，无模型方法依赖于大量样本进行训练。相反地，我们使用模块化的世界模型，包括动作条件、无动作和静态分支，来解开环境动态并根据过去的经验产生想象中的结果，而不是直接从真实环境中采样。我们使用变分自动编码器和变分图自动编码器来学习世界模型的潜在表示，将其与基于值的框架合并，以预测联合动作价值函数并优化整体训练目标。我们提供实验结果。

    In this paper, we propose a novel model-based multi-agent reinforcement learning approach named Value Decomposition Framework with Disentangled World Model to address the challenge of achieving a common goal of multiple agents interacting in the same environment with reduced sample complexity. Due to scalability and non-stationarity problems posed by multi-agent systems, model-free methods rely on a considerable number of samples for training. In contrast, we use a modularized world model, composed of action-conditioned, action-free, and static branches, to unravel the environment dynamics and produce imagined outcomes based on past experience, without sampling directly from the real environment. We employ variational auto-encoders and variational graph auto-encoders to learn the latent representations for the world model, which is merged with a value-based framework to predict the joint action-value function and optimize the overall training objective. We present experimental results 
    
[^4]: 逃离样本陷阱：使用配对距离估计器快速准确地估计认识不确定性

    Escaping the Sample Trap: Fast and Accurate Epistemic Uncertainty Estimation with Pairwise-Distance Estimators. (arXiv:2308.13498v1 [cs.LG])

    [http://arxiv.org/abs/2308.13498](http://arxiv.org/abs/2308.13498)

    本文介绍了使用配对距离估计器对集成模型进行认识不确定性估计的新方法，相比于常用的深度学习方法，该方法能够更快速、更准确地在更大的空间和更高维度上估计认识不确定性。

    

    本文介绍了一种使用配对距离估计器（PaiDEs）对集成模型进行认识不确定性估计的新方法。这些估计器利用模型组件之间的配对距离来建立熵的边界，并将这些边界作为基于信息准则的估计值。与最近基于样本的蒙特卡洛估计器用于认识不确定性估计的深度学习方法不同，PaiDEs能够在更大的空间（最多100倍）上以更快的速度（最多100倍）估计认识不确定性，并在更高维度上具有更准确的性能。为了验证我们的方法，我们进行了一系列用于评估认识不确定性估计的实验：一维正弦数据，摆动物体（Pendulum-v0），跳跃机器人（Hopper-v2），蚂蚁机器人（Ant-v2）和人形机器人（Humanoid-v2）。对于每个实验设置，我们应用了主动学习框架来展示PaiDEs在认识不确定性估计中的优势。

    This work introduces a novel approach for epistemic uncertainty estimation for ensemble models using pairwise-distance estimators (PaiDEs). These estimators utilize the pairwise-distance between model components to establish bounds on entropy and uses said bounds as estimates for information-based criterion. Unlike recent deep learning methods for epistemic uncertainty estimation, which rely on sample-based Monte Carlo estimators, PaiDEs are able to estimate epistemic uncertainty up to 100$\times$ faster, over a larger space (up to 100$\times$) and perform more accurately in higher dimensions. To validate our approach, we conducted a series of experiments commonly used to evaluate epistemic uncertainty estimation: 1D sinusoidal data, Pendulum-v0, Hopper-v2, Ant-v2 and Humanoid-v2. For each experimental setting, an Active Learning framework was applied to demonstrate the advantages of PaiDEs for epistemic uncertainty estimation.
    

