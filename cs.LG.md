# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Checkmating One, by Using Many: Combining Mixture of Experts with MCTS to Improve in Chess.](http://arxiv.org/abs/2401.16852) | 通过将混合专家方法和MCTS相结合，本研究在国际象棋中显著提升了下棋水平，验证了集成方法的有效性并展示了融入专家知识和战略原则到神经网络中的潜力。 |
| [^2] | [Strategic Client Selection to Address Non-IIDness in HAPS-enabled FL Networks.](http://arxiv.org/abs/2401.05308) | 该研究介绍了一种针对高空平台站（HAPS）使能的垂直异构网络中数据分布不均问题的战略客户选择策略，通过利用用户的网络流量行为预测和分类，优先选择数据呈现相似模式的客户参与，以提高联合学习（FL）模型的训练效果。 |
| [^3] | [Evolution of ESG-focused DLT Research: An NLP Analysis of the Literature.](http://arxiv.org/abs/2308.12420) | 本研究通过NLP分析了ESG主导的DLT研究的演化，通过构建引用网络和命名实体识别任务，对DLT在ESG背景下的发展进行了文献综述。 |
| [^4] | [FigCaps-HF: A Figure-to-Caption Generative Framework and Benchmark with Human Feedback.](http://arxiv.org/abs/2307.10867) | FigCaps-HF是一个图像生成标题的框架，可以通过融入领域专家的反馈意见，生成符合读者偏好的高质量图像标题。将自动评估和强化学习与人类反馈相结合，可以改善生成的标题与读者偏好的一致性。 |
| [^5] | [Accelerating Generalized Random Forests with Fixed-Point Trees.](http://arxiv.org/abs/2306.11908) | 本文提出一种新的树生长规则，使广义随机森林在无梯度优化的情况下大大节省了时间。 |
| [^6] | [Efficient Online Decision Tree Learning with Active Feature Acquisition.](http://arxiv.org/abs/2305.02093) | 论文提出了一种在线决策树学习的新方法，通过主动采集特征值来降低成本，同时使用后验抽样方案来保持在线预测的低遗憾度，该方法在多个基准测试中实现了最先进的性能。 |

# 详细

[^1]: 通过多种方式，将混合专家与MCTS相结合以提高国际象棋中的校验

    Checkmating One, by Using Many: Combining Mixture of Experts with MCTS to Improve in Chess. (arXiv:2401.16852v1 [cs.LG])

    [http://arxiv.org/abs/2401.16852](http://arxiv.org/abs/2401.16852)

    通过将混合专家方法和MCTS相结合，本研究在国际象棋中显著提升了下棋水平，验证了集成方法的有效性并展示了融入专家知识和战略原则到神经网络中的潜力。

    

    本文提出了一种新的方法，将深度学习与计算机棋盘相结合，同时使用混合专家方法和蒙特卡罗树搜索方法。我们的方法采用一套专门设计的模型，每个模型都针对游戏输入数据的特定变化做出响应。这导致了一个稀疏激活模型的框架，提供了显著的计算优势。我们的框架将混合专家方法与蒙特卡罗树搜索方法结合起来，以使其与国际象棋的战略阶段相一致，从而摆脱传统的“一刀切”的模型。相反，我们利用不同的游戏阶段定义，将计算任务有效地分配给多个专家神经网络。我们的实证研究显示，在游戏实力方面有了显著改进，超过了传统的单模型框架。这证实了我们集成方法的功效，并凸显了将专家知识和战略原则纳入神经网络中的潜力。

    This paper presents a new approach that integrates deep learning with computational chess, using both the Mixture of Experts (MoE) method and Monte-Carlo Tree Search (MCTS). Our methodology employs a suite of specialized models, each designed to respond to specific changes in the game's input data. This results in a framework with sparsely activated models, which provides significant computational benefits. Our framework combines the MoE method with MCTS, in order to align it with the strategic phases of chess, thus departing from the conventional ``one-for-all'' model. Instead, we utilize distinct game phase definitions to effectively distribute computational tasks across multiple expert neural networks. Our empirical research shows a substantial improvement in playing strength, surpassing the traditional single-model framework. This validates the efficacy of our integrated approach and highlights the potential of incorporating expert knowledge and strategic principles into neural net
    
[^2]: 面对HAPS使能的FL网络中的非独立同分布问题，战略客户选择的研究

    Strategic Client Selection to Address Non-IIDness in HAPS-enabled FL Networks. (arXiv:2401.05308v1 [cs.NI])

    [http://arxiv.org/abs/2401.05308](http://arxiv.org/abs/2401.05308)

    该研究介绍了一种针对高空平台站（HAPS）使能的垂直异构网络中数据分布不均问题的战略客户选择策略，通过利用用户的网络流量行为预测和分类，优先选择数据呈现相似模式的客户参与，以提高联合学习（FL）模型的训练效果。

    

    在由高空平台站（HAPS）使能的垂直异构网络中部署联合学习（FL）为各种不同通信和计算能力的客户提供了参与的机会。这种多样性不仅提高了FL模型的训练精度，还加快了其收敛速度。然而，在这些广阔的网络中应用FL存在显著的非独立同分布问题。这种数据异质性往往导致收敛速度较慢和模型训练性能的降低。我们的研究引入了一种针对此问题的客户选择策略，利用用户网络流量行为进行预测和分类。该策略通过战略性选择数据呈现相似模式的客户参与，同时优先考虑用户隐私。

    The deployment of federated learning (FL) within vertical heterogeneous networks, such as those enabled by high-altitude platform station (HAPS), offers the opportunity to engage a wide array of clients, each endowed with distinct communication and computational capabilities. This diversity not only enhances the training accuracy of FL models but also hastens their convergence. Yet, applying FL in these expansive networks presents notable challenges, particularly the significant non-IIDness in client data distributions. Such data heterogeneity often results in slower convergence rates and reduced effectiveness in model training performance. Our study introduces a client selection strategy tailored to address this issue, leveraging user network traffic behaviour. This strategy involves the prediction and classification of clients based on their network usage patterns while prioritizing user privacy. By strategically selecting clients whose data exhibit similar patterns for participation
    
[^3]: ESG主导的DLT研究的演化：对文献进行NLP分析

    Evolution of ESG-focused DLT Research: An NLP Analysis of the Literature. (arXiv:2308.12420v1 [cs.IR])

    [http://arxiv.org/abs/2308.12420](http://arxiv.org/abs/2308.12420)

    本研究通过NLP分析了ESG主导的DLT研究的演化，通过构建引用网络和命名实体识别任务，对DLT在ESG背景下的发展进行了文献综述。

    

    分布式账本技术(DLT)迅速发展，需要全面了解其各个组成部分。然而，针对DLT的环境、可持续性和治理(ESG)组成部分的系统文献综述还不足。为填补这一空白，我们选择了107篇种子文献，构建了一个包含63,083个参考文献的引用网络，并将其精炼为24,539篇文献的语料库进行分析。然后，我们根据一个已建立的技术分类法从46篇论文中标记了命名实体，并通过找出DLT的ESG要素来完善这个分类法。利用基于transformer的语言模型，我们对一个预先训练的语言模型进行了细化调整，用于命名实体识别任务，使用我们标记的数据集。我们利用我们调整后的语言模型对语料库进行了精简，得到了505篇关键论文，通过命名实体和时间图分析，促进了对DLT在ESG背景下的演化的文献综述。

    Distributed Ledger Technologies (DLTs) have rapidly evolved, necessitating comprehensive insights into their diverse components. However, a systematic literature review that emphasizes the Environmental, Sustainability, and Governance (ESG) components of DLT remains lacking. To bridge this gap, we selected 107 seed papers to build a citation network of 63,083 references and refined it to a corpus of 24,539 publications for analysis. Then, we labeled the named entities in 46 papers according to twelve top-level categories derived from an established technology taxonomy and enhanced the taxonomy by pinpointing DLT's ESG elements. Leveraging transformer-based language models, we fine-tuned a pre-trained language model for a Named Entity Recognition (NER) task using our labeled dataset. We used our fine-tuned language model to distill the corpus to 505 key papers, facilitating a literature review via named entities and temporal graph analysis on DLT evolution in the context of ESG. Our con
    
[^4]: FigCaps-HF:一个基于人类反馈的图像生成标题框架和基准测试

    FigCaps-HF: A Figure-to-Caption Generative Framework and Benchmark with Human Feedback. (arXiv:2307.10867v1 [cs.CL])

    [http://arxiv.org/abs/2307.10867](http://arxiv.org/abs/2307.10867)

    FigCaps-HF是一个图像生成标题的框架，可以通过融入领域专家的反馈意见，生成符合读者偏好的高质量图像标题。将自动评估和强化学习与人类反馈相结合，可以改善生成的标题与读者偏好的一致性。

    

    标题对于理解科学可视化和文档至关重要。现有的科学图像生成标题方法依赖于从文档中提取的图像-标题配对进行训练，但其中许多配对在帮助性、解释性和视觉描述性等指标上存在不足，导致生成的标题与读者偏好不一致。为了能够生成高质量的图像标题，我们引入了FigCaps-HF，这是一个新的图像生成标题框架，可以融入领域专家的反馈意见，以生成优化了读者偏好的标题。我们的框架包含1）一种评估图像-标题配对质量的自动方法，2）一种基于人类反馈的强化学习（RLHF）方法，用于优化生成式图像生成标题模型以符合读者偏好。我们通过在不同类型的模型上改进性能，证明了我们简单的学习框架的有效性。

    Captions are crucial for understanding scientific visualizations and documents. Existing captioning methods for scientific figures rely on figure-caption pairs extracted from documents for training, many of which fall short with respect to metrics like helpfulness, explainability, and visual-descriptiveness [15] leading to generated captions being misaligned with reader preferences. To enable the generation of high-quality figure captions, we introduce FigCaps-HF a new framework for figure-caption generation that can incorporate domain expert feedback in generating captions optimized for reader preferences. Our framework comprises of 1) an automatic method for evaluating quality of figure-caption pairs, 2) a novel reinforcement learning with human feedback (RLHF) method to optimize a generative figure-to-caption model for reader preferences. We demonstrate the effectiveness of our simple learning framework by improving performance over standard fine-tuning across different types of mod
    
[^5]: 基于定点树的广义随机森林加速

    Accelerating Generalized Random Forests with Fixed-Point Trees. (arXiv:2306.11908v1 [stat.ML])

    [http://arxiv.org/abs/2306.11908](http://arxiv.org/abs/2306.11908)

    本文提出一种新的树生长规则，使广义随机森林在无梯度优化的情况下大大节省了时间。

    

    广义随机森林建立在传统随机森林的基础上，通过将其作为自适应核加权算法来构建估算器，并通过基于梯度的树生长过程来实现。我们提出了一种新的树生长规则，基于定点迭代近似表示梯度近似，实现了无梯度优化，并为此开发了渐近理论。这有效地节省了时间，尤其是在目标量的维度适中时。

    Generalized random forests arXiv:1610.01271 build upon the well-established success of conventional forests (Breiman, 2001) to offer a flexible and powerful non-parametric method for estimating local solutions of heterogeneous estimating equations. Estimators are constructed by leveraging random forests as an adaptive kernel weighting algorithm and implemented through a gradient-based tree-growing procedure. By expressing this gradient-based approximation as being induced from a single Newton-Raphson root-finding iteration, and drawing upon the connection between estimating equations and fixed-point problems arXiv:2110.11074, we propose a new tree-growing rule for generalized random forests induced from a fixed-point iteration type of approximation, enabling gradient-free optimization, and yielding substantial time savings for tasks involving even modest dimensionality of the target quantity (e.g. multiple/multi-level treatment effects). We develop an asymptotic theory for estimators o
    
[^6]: 带有主动特征获取的高效在线决策树学习

    Efficient Online Decision Tree Learning with Active Feature Acquisition. (arXiv:2305.02093v1 [cs.LG])

    [http://arxiv.org/abs/2305.02093](http://arxiv.org/abs/2305.02093)

    论文提出了一种在线决策树学习的新方法，通过主动采集特征值来降低成本，同时使用后验抽样方案来保持在线预测的低遗憾度，该方法在多个基准测试中实现了最先进的性能。

    

    构建决策树在线是一个经典的机器学习问题。现有工作通常假设每个进入的数据点的特征已经准备好了。然而，在许多实际应用中，特征值和标签都是未知的，只能以一定的成本获取。例如，在医学诊断中，医生必须选择对病人进行哪些测试（即进行昂贵的特征查询），以便做出诊断决策（即预测标签）。我们提供了一个新的视角来解决这个实际难题。我们的框架包括一个嵌入在线学习方案的主动计划预测器，我们研究了几个信息收集功能。具体而言，我们采用了一种基于自适应子模性的代理信息获取函数，以最小的成本主动查询特征值，同时使用后验抽样方案来保持在线预测的低遗憾度。我们在合成和真实数据集上展示了我们提出的方法的效率和有效性，在几个基准测试上实现了最先进的性能。

    Constructing decision trees online is a classical machine learning problem. Existing works often assume that features are readily available for each incoming data point. However, in many real world applications, both feature values and the labels are unknown a priori and can only be obtained at a cost. For example, in medical diagnosis, doctors have to choose which tests to perform (i.e., making costly feature queries) on a patient in order to make a diagnosis decision (i.e., predicting labels). We provide a fresh perspective to tackle this practical challenge. Our framework consists of an active planning oracle embedded in an online learning scheme for which we investigate several information acquisition functions. Specifically, we employ a surrogate information acquisition function based on adaptive submodularity to actively query feature values with a minimal cost, while using a posterior sampling scheme to maintain a low regret for online prediction. We demonstrate the efficiency a
    

