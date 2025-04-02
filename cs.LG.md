# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Temporal and Semantic Evaluation Metrics for Foundation Models in Post-Hoc Analysis of Robotic Sub-tasks](https://arxiv.org/abs/2403.17238) | 提出了一种基于基础模型的自动化框架，通过新颖的提示策略将轨迹数据分解为时间和语言描述的子任务，同时引入了时间相似性和语义相似性两种新的评估指标。 |
| [^2] | [A Three-Phases SFT Hybrid Model Integrated Strong Prior Module and Data Overlap Estimation in the Eduation Context](https://arxiv.org/abs/2403.15426) | 提出了一种在教育领域中应用的三阶段监督微调模型，通过先验和数据重叠估计实现了教育知识的结构拆卸和增量引导输出。 |
| [^3] | [A Clustering Method with Graph Maximum Decoding Information](https://arxiv.org/abs/2403.13846) | CMDI聚类方法创新性地将二维结构信息理论融入聚类过程中，弥补了基于图的模型聚类方法中忽略的随机游走访问节点和数据中嵌入的结构信息的不确定性。 |
| [^4] | [VL-ICL Bench: The Devil in the Details of Benchmarking Multimodal In-Context Learning](https://arxiv.org/abs/2403.13164) | 大型语言模型的视觉变种在识别、推理和基准确定等领域取得了显著进展，但多模态上下文学习的广泛能力和限制仍未得到充分探讨。 |
| [^5] | [Towards Adversarially Robust Dataset Distillation by Curvature Regularization](https://arxiv.org/abs/2403.10045) | 本文探讨了如何通过曲率正则化方法在精炼数据集中嵌入对抗鲁棒性，以保持模型高准确性并获得更好的对抗鲁棒性。 |
| [^6] | [Over-The-Air Double-Threshold Deep Learner for Jamming Detection in 5G RF domain](https://arxiv.org/abs/2403.02645) | 本文提出了一种在5G网络中检测干扰者的新型深度学习技术，通过引入双阈值深度学习干扰检测器，专注于SSB的RF领域特征，提高了网络的鲁棒性。 |
| [^7] | [Sharp Rates in Dependent Learning Theory: Avoiding Sample Size Deflation for the Square Loss](https://arxiv.org/abs/2402.05928) | 本文研究了依赖学习理论中的尖锐率，主要是为了避免样本大小缩减对方差产生影响。当假设类别的拓扑结构符合某些条件时，经验风险最小化者的性能与类别的复杂性和二阶统计量有关。 |
| [^8] | [Individualized Policy Evaluation and Learning under Clustered Network Interference](https://arxiv.org/abs/2311.02467) | 本文研究了集群网络干扰下个体化策略评估与学习的问题，提出了一种只假设半参数结构模型的方法，能够更准确地评估和学习最优的个体化处理规则。 |
| [^9] | [Explainable Bayesian Optimization.](http://arxiv.org/abs/2401.13334) | 本论文介绍了一种可解释性贝叶斯优化的方法，通过TNTRules生成高质量的解释，填补了贝叶斯优化和可解释人工智能之间的间隙。 |
| [^10] | [Causal Dynamic Variational Autoencoder for Counterfactual Regression in Longitudinal Data.](http://arxiv.org/abs/2310.10559) | 本论文提出了一种因果动态变分自编码器（CDVAE）来解决纵向数据中的反事实回归问题。该方法假设存在未观察到的调整变量，并通过结合动态变分自编码器（DVAE）框架和使用倾向得分的加权策略来估计反事实响应。 |
| [^11] | [Graph Reinforcement Learning for Radio Resource Allocation.](http://arxiv.org/abs/2203.03906) | 该论文介绍了一种利用图强化学习方法进行无线资源分配的方法，通过利用拓扑信息和排列特性，降低了深度强化学习的训练复杂性，并通过优化预测功率分配问题来验证方法的有效性。 |

# 详细

[^1]: 基于时间和语义评估指标的基础模型在机器人子任务事后分析中的应用

    Temporal and Semantic Evaluation Metrics for Foundation Models in Post-Hoc Analysis of Robotic Sub-tasks

    [https://arxiv.org/abs/2403.17238](https://arxiv.org/abs/2403.17238)

    提出了一种基于基础模型的自动化框架，通过新颖的提示策略将轨迹数据分解为时间和语言描述的子任务，同时引入了时间相似性和语义相似性两种新的评估指标。

    

    最近在任务和运动规划（TAMP）领域的研究表明，在使用带有质量标记数据的语言监督机器人轨迹进行控制策略训练可以显着提高代理任务成功率。然而，这类数据的稀缺性对将这些方法扩展到一般用例构成重大障碍。为了解决这一问题，我们提出了一种自动化框架，通过利用最近的基础模型（FMs）的提示策略，包括大型语言模型（LLMs）和视觉语言模型（VLMs），将轨迹数据分解为基于时间和自然语言的描述性子任务。我们的框架为构成完整轨迹的底层子任务提供了基于时间和语言的描述。为了严格评估我们的自动标记框架的质量，我们提出了一种算法 SIMILARITY 来生成两种新颖的指标，即时间相似性和语义相似性。

    arXiv:2403.17238v1 Announce Type: cross  Abstract: Recent works in Task and Motion Planning (TAMP) show that training control policies on language-supervised robot trajectories with quality labeled data markedly improves agent task success rates. However, the scarcity of such data presents a significant hurdle to extending these methods to general use cases. To address this concern, we present an automated framework to decompose trajectory data into temporally bounded and natural language-based descriptive sub-tasks by leveraging recent prompting strategies for Foundation Models (FMs) including both Large Language Models (LLMs) and Vision Language Models (VLMs). Our framework provides both time-based and language-based descriptions for lower-level sub-tasks that comprise full trajectories. To rigorously evaluate the quality of our automatic labeling framework, we contribute an algorithm SIMILARITY to produce two novel metrics, temporal similarity and semantic similarity. The metrics me
    
[^2]: 教育环境下集成强先验模块和数据重叠估计的三阶段SFT混合模型

    A Three-Phases SFT Hybrid Model Integrated Strong Prior Module and Data Overlap Estimation in the Eduation Context

    [https://arxiv.org/abs/2403.15426](https://arxiv.org/abs/2403.15426)

    提出了一种在教育领域中应用的三阶段监督微调模型，通过先验和数据重叠估计实现了教育知识的结构拆卸和增量引导输出。

    

    在本文中，我们提出了一种端到端基于先验的三阶段监督微调模型，证明比传统微调方法更有竞争力。具体而言，我们的模型实现了教育知识的结构拆卸和增量引导输出。为此，我们通过采样器和重叠估计神经网络对三种类型的数据进行了健壮的分类，将预处理数据集分三批注入预训练模型进行LORA微调。然后，我们设计了一个先验模块，将系统提示、向量数据库和抽象语法树任务分割相结合。最后，对基于先验的微调模型应用了压缩方法和正则化约束，随后在输出端进行文本过滤以获得增量引导结果。我们的模型代表了真正以丰富的教育知识、分步指导的特点体现导师角色的第一项研究努力。

    arXiv:2403.15426v1 Announce Type: cross  Abstract: In this paper, we propose an end-to-end prior-based three-phases supervised fine-tuned model, which is proved more competitive than traditional fine-tuning method. More specifically, our model realizes the structural disassembly and incremental guided output of educational knowledge. To this end, we robustify data classification of three types via a sampler and overlap estimation neural network, and inject the preprocessing datasets into pre-trained model in three batches for LORA fine-tuning. Then, we design a prior module couples system prompt, vector databases, and abstract syntax tree task segmentation. Finally, the compression method and regularization constraint are applied to the prior-based fine-tuned model, followed by text filter at the output end to obtain incremental guided results. Our model represents the first research effort to truly embody the tutor role with the features of abundant educational knowledge, step-by-step
    
[^3]: 一种具有图最大解码信息的聚类方法

    A Clustering Method with Graph Maximum Decoding Information

    [https://arxiv.org/abs/2403.13846](https://arxiv.org/abs/2403.13846)

    CMDI聚类方法创新性地将二维结构信息理论融入聚类过程中，弥补了基于图的模型聚类方法中忽略的随机游走访问节点和数据中嵌入的结构信息的不确定性。

    

    基于图模型的聚类方法因其在各种知识领域中的广泛适用性而备受关注。其能够与其他相关应用无缝集成的适应性赋予了基于图模型的聚类分析能力，可以强大地从数据集中提取“自然关联”或“图结构”，有助于建模数据点之间的关系。尽管这种方法效果显著，但当前利用基于图的模型的聚类方法忽略了节点之间随机游走访问以及数据中嵌入的结构信息所带来的不确定性。为填补这一空白，我们提出了一种新颖的基于图的模型内最大化解码信息的聚类方法，命名为CMDI。CMDI创新地将二维结构信息理论纳入到聚类过程中，包括两个阶段：图结构提取和图顶点

    arXiv:2403.13846v1 Announce Type: cross  Abstract: The clustering method based on graph models has garnered increased attention for its widespread applicability across various knowledge domains. Its adaptability to integrate seamlessly with other relevant applications endows the graph model-based clustering analysis with the ability to robustly extract "natural associations" or "graph structures" within datasets, facilitating the modelling of relationships between data points. Despite its efficacy, the current clustering method utilizing the graph-based model overlooks the uncertainty associated with random walk access between nodes and the embedded structural information in the data. To address this gap, we present a novel Clustering method for Maximizing Decoding Information within graph-based models, named CMDI. CMDI innovatively incorporates two-dimensional structural information theory into the clustering process, consisting of two phases: graph structure extraction and graph vert
    
[^4]: VL-ICL Bench: 基于细节的多模态上下文学习基准测试中的细节之魔

    VL-ICL Bench: The Devil in the Details of Benchmarking Multimodal In-Context Learning

    [https://arxiv.org/abs/2403.13164](https://arxiv.org/abs/2403.13164)

    大型语言模型的视觉变种在识别、推理和基准确定等领域取得了显著进展，但多模态上下文学习的广泛能力和限制仍未得到充分探讨。

    

    大型语言模型（LLMs）以其著名的出现式上下文学习（ICL）而闻名——即在仅提供几个示例作为提示的情况下，快速适应新任务的能力，而无需更新模型的权重。构建在LLMs之上的视觉大型语言模型（VLLMs）在识别、推理和基准确定等领域取得了显著进展。然而，对于\emph{多模态ICL}的研究主要集中在少样本视觉问题回答（VQA）和图像字幕上，我们将展示二者既没有充分利用ICL的优势，也没有测试其限制。对多模态ICL的更广泛能力和局限性尚未得到充分探讨。在本研究中，我们引入了一个全面的多模态上下文学习基准测试 VL-ICL Bench，涵盖了涉及图像和文本作为输入和输出的广泛任务范围，并涵盖了从{感知到推理和长期上下文长度}的不同类型挑战。

    arXiv:2403.13164v1 Announce Type: new  Abstract: Large language models (LLMs) famously exhibit emergent in-context learning (ICL) -- the ability to rapidly adapt to new tasks using few-shot examples provided as a prompt, without updating the model's weights. Built on top of LLMs, vision large language models (VLLMs) have advanced significantly in areas such as recognition, reasoning, and grounding. However, investigations into \emph{multimodal ICL} have predominantly focused on few-shot visual question answering (VQA), and image captioning, which we will show neither exploit the strengths of ICL, nor test its limitations. The broader capabilities and limitations of multimodal ICL remain under-explored. In this study, we introduce a comprehensive benchmark VL-ICL Bench for multimodal in-context learning, encompassing a broad spectrum of tasks that involve both images and text as inputs and outputs, and different types of challenges, from {perception to reasoning and long context length}
    
[^5]: 通过曲率正则化实现对抗鲁棒性数据集精炼

    Towards Adversarially Robust Dataset Distillation by Curvature Regularization

    [https://arxiv.org/abs/2403.10045](https://arxiv.org/abs/2403.10045)

    本文探讨了如何通过曲率正则化方法在精炼数据集中嵌入对抗鲁棒性，以保持模型高准确性并获得更好的对抗鲁棒性。

    

    数据集精炼（DD）允许将数据集精炼为原始大小的分数，同时保留丰富的分布信息，使得在精炼数据集上训练的模型可以在节省显著计算负载的同时达到可比的准确性。最近在这一领域的研究集中在提高在精炼数据集上训练的模型的准确性。在本文中，我们旨在探索DD的一种新视角。我们研究如何在精炼数据集中嵌入对抗鲁棒性，以使在这些数据集上训练的模型保持高精度的同时获得更好的对抗鲁棒性。我们提出了一种通过将曲率正则化纳入到精炼过程中来实现这一目标的新方法，而这种方法的计算开销比标准的对抗训练要少得多。大量的实证实验表明，我们的方法不仅在准确性上优于标准对抗训练，同时在对抗性能方面也取得了显著改进。

    arXiv:2403.10045v1 Announce Type: new  Abstract: Dataset distillation (DD) allows datasets to be distilled to fractions of their original size while preserving the rich distributional information so that models trained on the distilled datasets can achieve a comparable accuracy while saving significant computational loads. Recent research in this area has been focusing on improving the accuracy of models trained on distilled datasets. In this paper, we aim to explore a new perspective of DD. We study how to embed adversarial robustness in distilled datasets, so that models trained on these datasets maintain the high accuracy and meanwhile acquire better adversarial robustness. We propose a new method that achieves this goal by incorporating curvature regularization into the distillation process with much less computational overhead than standard adversarial training. Extensive empirical experiments suggest that our method not only outperforms standard adversarial training on both accur
    
[^6]: 在5G RF领域，用于干扰检测的空中双阈值深度学习器

    Over-The-Air Double-Threshold Deep Learner for Jamming Detection in 5G RF domain

    [https://arxiv.org/abs/2403.02645](https://arxiv.org/abs/2403.02645)

    本文提出了一种在5G网络中检测干扰者的新型深度学习技术，通过引入双阈值深度学习干扰检测器，专注于SSB的RF领域特征，提高了网络的鲁棒性。

    

    随着5G无线通信的发展，同步信号块（SSB）在设备同步和服务可访问性中起着关键作用。然而，由于SSB传输具有可预测性，包括主要同步信号（PSS）和次要同步信号（SSS），干扰攻击是重要威胁。本文利用RF领域知识，提出了一种新颖的基于深度学习的5G网络干扰检测技术。与现有的大多依赖网络参数的干扰检测算法不同，我们通过专注于SSB引入了双阈值深度学习干扰检测器。该检测方法侧重于RF领域特征，提高了网络的鲁棒性，无需与现有网络基础设施集成。通过集成一个预处理块来提取PSS相关性和每个空闲资源元素的能量（EPNRE）

    arXiv:2403.02645v1 Announce Type: cross  Abstract: With the evolution of 5G wireless communications, the Synchronization Signal Block (SSB) plays a critical role in the synchronization of devices and accessibility of services. However, due to the predictable nature of SSB transmission, including the Primary and Secondary Synchronization Signals (PSS and SSS), jamming attacks are critical threats. By leveraging RF domain knowledge, this work presents a novel deep learning-based technique for detecting jammers in 5G networks. Unlike the existing jamming detection algorithms that mostly rely on network parameters, we introduce a double threshold deep learning jamming detector by focusing on the SSB. The detection method is focused on RF domain features and improves the robustness of the network without requiring integration with the pre-existing network infrastructure. By integrating a preprocessing block that extracts PSS correlation and energy per null resource elements (EPNRE) characte
    
[^7]: 依赖学习理论中的尖锐率：避免样本大小缩减的平方损失

    Sharp Rates in Dependent Learning Theory: Avoiding Sample Size Deflation for the Square Loss

    [https://arxiv.org/abs/2402.05928](https://arxiv.org/abs/2402.05928)

    本文研究了依赖学习理论中的尖锐率，主要是为了避免样本大小缩减对方差产生影响。当假设类别的拓扑结构符合某些条件时，经验风险最小化者的性能与类别的复杂性和二阶统计量有关。

    

    本文研究了具有依赖性（β-混合）数据和平方损失的统计学习，在一个假设类别Φ_p的子集F中，其中Φ_p是范数∥f∥_Φ_p≡sup_m≥1 m^{-1/p}∥f∥_L^m，其中p∈[2，∞]。我们的研究动机是在具有依赖性数据的学习中寻找尖锐的噪声交互项或方差代理。在没有任何可实现性假设的情况下，典型的非渐近结果显示出方差代理通过底层协变量过程的混合时间进行了乘积缩减。我们证明，只要在我们的假设类别F上，L^2和Φ_p的拓扑是可比较的，即Φ_p是一个弱亚高斯类别：∥f∥_Φ_p≲∥f∥_L^2^η，其中η∈(0，1]，经验风险最小化者在其主导项中只实现了一种只依赖于类别复杂性和二阶统计量的速率。我们的结果适用于许多依赖性数据模型。

    In this work, we study statistical learning with dependent ($\beta$-mixing) data and square loss in a hypothesis class $\mathscr{F}\subset L_{\Psi_p}$ where $\Psi_p$ is the norm $\|f\|_{\Psi_p} \triangleq \sup_{m\geq 1} m^{-1/p} \|f\|_{L^m} $ for some $p\in [2,\infty]$. Our inquiry is motivated by the search for a sharp noise interaction term, or variance proxy, in learning with dependent data. Absent any realizability assumption, typical non-asymptotic results exhibit variance proxies that are deflated \emph{multiplicatively} by the mixing time of the underlying covariates process. We show that whenever the topologies of $L^2$ and $\Psi_p$ are comparable on our hypothesis class $\mathscr{F}$ -- that is, $\mathscr{F}$ is a weakly sub-Gaussian class: $\|f\|_{\Psi_p} \lesssim \|f\|_{L^2}^\eta$ for some $\eta\in (0,1]$ -- the empirical risk minimizer achieves a rate that only depends on the complexity of the class and second order statistics in its leading term. Our result holds whether t
    
[^8]: 集群网络干扰下的个体化策略评估与学习

    Individualized Policy Evaluation and Learning under Clustered Network Interference

    [https://arxiv.org/abs/2311.02467](https://arxiv.org/abs/2311.02467)

    本文研究了集群网络干扰下个体化策略评估与学习的问题，提出了一种只假设半参数结构模型的方法，能够更准确地评估和学习最优的个体化处理规则。

    

    尽管现在有很多关于政策评估和学习的文献，但大部分之前的工作都假设一个个体的处理分配不会影响另一个个体的结果。不幸的是，忽视干扰可能导致评估偏误和无效的学习策略。例如，处理有很多朋友的有影响力的个体可能产生正向溢出效应，从而改善个体化处理规则（ITR）的整体性能。我们考虑在集群网络干扰（也称为部分干扰）下评估和学习最优ITR的问题，在该问题中，单位聚类从一个总体中抽样，并且在每个聚类中单位之间可能互相影响。与以前的方法强制限制溢出效应不同，所提出的方法只假设半参数结构模型，每个单位的结果是聚类中的个体处理的加法函数。

    While there now exists a large literature on policy evaluation and learning, much of prior work assumes that the treatment assignment of one unit does not affect the outcome of another unit. Unfortunately, ignoring interference may lead to biased policy evaluation and ineffective learned policies. For example, treating influential individuals who have many friends can generate positive spillover effects, thereby improving the overall performance of an individualized treatment rule (ITR). We consider the problem of evaluating and learning an optimal ITR under clustered network interference (also known as partial interference) where clusters of units are sampled from a population and units may influence one another within each cluster. Unlike previous methods that impose strong restrictions on spillover effects, the proposed methodology only assumes a semiparametric structural model where each unit's outcome is an additive function of individual treatments within the cluster. Under this 
    
[^9]: 可解释性贝叶斯优化

    Explainable Bayesian Optimization. (arXiv:2401.13334v1 [cs.LG])

    [http://arxiv.org/abs/2401.13334](http://arxiv.org/abs/2401.13334)

    本论文介绍了一种可解释性贝叶斯优化的方法，通过TNTRules生成高质量的解释，填补了贝叶斯优化和可解释人工智能之间的间隙。

    

    在工业领域，贝叶斯优化（BO）被广泛应用于人工智能协作参数调优的控制系统中。然而，由于近似误差和简化目标，BO的解决方案可能偏离人类专家的真实目标，需要后续调整。BO的黑盒特性限制了协作调优过程，因为专家不信任BO的建议。目前的可解释人工智能（XAI）方法不适用于优化问题，因此无法解决此间隙。为了填补这一间隙，我们提出了TNTRules（TUNE-NOTUNE规则），一种事后基于规则的可解释性方法，通过多目标优化生成高质量的解释。我们对基准优化问题和实际超参数优化任务的评估表明，TNTRules在生成高质量解释方面优于最先进的XAI方法。这项工作对BO和XAI的交叉领域做出了贡献，提供了可解释的优化方法。

    In industry, Bayesian optimization (BO) is widely applied in the human-AI collaborative parameter tuning of cyber-physical systems. However, BO's solutions may deviate from human experts' actual goal due to approximation errors and simplified objectives, requiring subsequent tuning. The black-box nature of BO limits the collaborative tuning process because the expert does not trust the BO recommendations. Current explainable AI (XAI) methods are not tailored for optimization and thus fall short of addressing this gap. To bridge this gap, we propose TNTRules (TUNE-NOTUNE Rules), a post-hoc, rule-based explainability method that produces high quality explanations through multiobjective optimization. Our evaluation of benchmark optimization problems and real-world hyperparameter optimization tasks demonstrates TNTRules' superiority over state-of-the-art XAI methods in generating high quality explanations. This work contributes to the intersection of BO and XAI, providing interpretable opt
    
[^10]: 因果动态变分自编码器用于纵向数据中的反事实回归

    Causal Dynamic Variational Autoencoder for Counterfactual Regression in Longitudinal Data. (arXiv:2310.10559v1 [stat.ML])

    [http://arxiv.org/abs/2310.10559](http://arxiv.org/abs/2310.10559)

    本论文提出了一种因果动态变分自编码器（CDVAE）来解决纵向数据中的反事实回归问题。该方法假设存在未观察到的调整变量，并通过结合动态变分自编码器（DVAE）框架和使用倾向得分的加权策略来估计反事实响应。

    

    在很多实际应用中，如精准医学、流行病学、经济和市场营销中，估计随时间变化的治疗效果是相关的。许多最先进的方法要么假设了所有混杂变量的观测结果，要么试图推断未观察到的混杂变量。我们采取了不同的观点，假设存在未观察到的风险因素，即仅影响结果序列的调整变量。在无混杂性的情况下，我们以未观测到的风险因素导致的治疗反应中的未知异质性为目标，估计个体治疗效果（ITE）。我们应对了时变效应和未观察到的调整变量所带来的挑战。在学习到的调整变量的有效性和治疗效果的一般化界限的理论结果指导下，我们设计了因果DVAE（CDVAE）。该模型将动态变分自编码器（DVAE）框架与使用倾向得分的加权策略相结合，用于估计反事实响应。

    Estimating treatment effects over time is relevant in many real-world applications, such as precision medicine, epidemiology, economy, and marketing. Many state-of-the-art methods either assume the observations of all confounders or seek to infer the unobserved ones. We take a different perspective by assuming unobserved risk factors, i.e., adjustment variables that affect only the sequence of outcomes. Under unconfoundedness, we target the Individual Treatment Effect (ITE) estimation with unobserved heterogeneity in the treatment response due to missing risk factors. We address the challenges posed by time-varying effects and unobserved adjustment variables. Led by theoretical results over the validity of the learned adjustment variables and generalization bounds over the treatment effect, we devise Causal DVAE (CDVAE). This model combines a Dynamic Variational Autoencoder (DVAE) framework with a weighting strategy using propensity scores to estimate counterfactual responses. The CDVA
    
[^11]: 图强化学习用于无线资源分配

    Graph Reinforcement Learning for Radio Resource Allocation. (arXiv:2203.03906v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2203.03906](http://arxiv.org/abs/2203.03906)

    该论文介绍了一种利用图强化学习方法进行无线资源分配的方法，通过利用拓扑信息和排列特性，降低了深度强化学习的训练复杂性，并通过优化预测功率分配问题来验证方法的有效性。

    

    由于其处理无模型和端到端问题的能力，深度强化学习(DRL)在资源分配方面得到了广泛的研究。然而，DRL的高训练复杂性限制了它在动态无线系统中的实际应用。为了降低训练成本，我们采用图强化学习来利用无线通信中许多问题固有的两种关系先验：拓扑信息和排列特性。为了系统地设计图强化学习框架来利用这两个先验，我们首先构思了一种将状态矩阵转换为状态图的方法，然后提出了一种通用的图神经网络方法来满足理想的排列特性。为了展示如何应用所提出的方法，我们以深度确定性策略梯度(DDPG)为例，优化了两个代表性的资源分配问题。一个是预测功率分配，旨在最小化能耗。

    Deep reinforcement learning (DRL) for resource allocation has been investigated extensively owing to its ability of handling model-free and end-to-end problems. Yet the high training complexity of DRL hinders its practical use in dynamic wireless systems. To reduce the training cost, we resort to graph reinforcement learning for exploiting two kinds of relational priors inherent in many problems in wireless communications: topology information and permutation properties. To design graph reinforcement learning framework systematically for harnessing the two priors, we first conceive a method to transform state matrix into state graph, and then propose a general method for graph neural networks to satisfy desirable permutation properties. To demonstrate how to apply the proposed methods, we take deep deterministic policy gradient (DDPG) as an example for optimizing two representative resource allocation problems. One is predictive power allocation that minimizes the energy consumed for e
    

