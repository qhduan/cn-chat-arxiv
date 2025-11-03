# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [RObotic MAnipulation Network (ROMAN) -- Hybrid Hierarchical Learning for Solving Complex Sequential Tasks.](http://arxiv.org/abs/2307.00125) | RObotic MAnipulation Network（ROMAN）通过混合层次学习框架解决复杂的连续操作任务，实现了任务多样性和鲁棒的失败恢复。 |
| [^2] | [Adversarially robust clustering with optimality guarantees.](http://arxiv.org/abs/2306.09977) | 本文提出了一种简单的算法，即使在存在对抗性的异常值的情况下，也能获得最优的错标率。在没有异常值的情况下，该算法能够实现与洛伊德算法类似的理论保证. |
| [^3] | [Patient Dropout Prediction in Virtual Health: A Multimodal Dynamic Knowledge Graph and Text Mining Approach.](http://arxiv.org/abs/2306.03833) | 本研究提出了一种多模态动态知识驱动退出预测（MDKDP）框架，能够解决虚拟健康中不同利益相关者之间和医疗保健交付系统之间的信息不对称问题，提高了退出预测的性能。 |

# 详细

[^1]: RObotic MAnipulation Network（ROMAN）--混合层次学习解决复杂的连续任务

    RObotic MAnipulation Network (ROMAN) -- Hybrid Hierarchical Learning for Solving Complex Sequential Tasks. (arXiv:2307.00125v1 [cs.RO])

    [http://arxiv.org/abs/2307.00125](http://arxiv.org/abs/2307.00125)

    RObotic MAnipulation Network（ROMAN）通过混合层次学习框架解决复杂的连续操作任务，实现了任务多样性和鲁棒的失败恢复。

    

    在实体人工智能中，解决长序列任务面临着重大挑战。使机器人系统能够执行多样化的连续任务，并具备广泛的操作技能是一个活跃的研究领域。在本文中，我们提出了一种混合层次学习框架，即ROBOTIC Manipulation Network（ROMAN），以解决机器人操作中的多个复杂任务的长时间任务。ROMAN通过集成行为克隆、模仿学习和强化学习来实现任务的多样性以及鲁棒的失败恢复。它包括一个中央操作网络，协调一组不同的神经网络，每个网络专注于不同的可重组子任务，生成它们在复杂的长时间操作任务中的正确连续动作。实验结果表明，通过协调和激活这些专门的操作专家，ROMAN生成了正确的顺序激活。

    Solving long sequential tasks poses a significant challenge in embodied artificial intelligence. Enabling a robotic system to perform diverse sequential tasks with a broad range of manipulation skills is an active area of research. In this work, we present a Hybrid Hierarchical Learning framework, the Robotic Manipulation Network (ROMAN), to address the challenge of solving multiple complex tasks over long time horizons in robotic manipulation. ROMAN achieves task versatility and robust failure recovery by integrating behavioural cloning, imitation learning, and reinforcement learning. It consists of a central manipulation network that coordinates an ensemble of various neural networks, each specialising in distinct re-combinable sub-tasks to generate their correct in-sequence actions for solving complex long-horizon manipulation tasks. Experimental results show that by orchestrating and activating these specialised manipulation experts, ROMAN generates correct sequential activations f
    
[^2]: 带有最优性保证的对抗鲁棒聚类

    Adversarially robust clustering with optimality guarantees. (arXiv:2306.09977v1 [math.ST])

    [http://arxiv.org/abs/2306.09977](http://arxiv.org/abs/2306.09977)

    本文提出了一种简单的算法，即使在存在对抗性的异常值的情况下，也能获得最优的错标率。在没有异常值的情况下，该算法能够实现与洛伊德算法类似的理论保证.

    

    我们考虑对来自亚高斯混合的数据点进行聚类的问题。现有的可证明达到最优错标率的方法，如洛伊德算法，通常容易受到异常值的影响。相反，似乎对对抗性扰动具有鲁棒性的聚类方法不知道是否满足最优的统计保证。我们提出了一种简单的算法，即使允许出现对抗性的异常值，也能获得最优的错标率。当满足弱初始化条件时，我们的算法在常数次迭代中实现最优误差率。在没有异常值的情况下，在固定维度上，我们的理论保证与洛伊德算法类似。在各种模拟数据集上进行了广泛的实验，以支持我们的方法的理论保证。

    We consider the problem of clustering data points coming from sub-Gaussian mixtures. Existing methods that provably achieve the optimal mislabeling error, such as the Lloyd algorithm, are usually vulnerable to outliers. In contrast, clustering methods seemingly robust to adversarial perturbations are not known to satisfy the optimal statistical guarantees. We propose a simple algorithm that obtains the optimal mislabeling rate even when we allow adversarial outliers to be present. Our algorithm achieves the optimal error rate in constant iterations when a weak initialization condition is satisfied. In the absence of outliers, in fixed dimensions, our theoretical guarantees are similar to that of the Lloyd algorithm. Extensive experiments on various simulated data sets are conducted to support the theoretical guarantees of our method.
    
[^3]: 虚拟健康中的患者预测：一种多模态动态知识图谱和文本挖掘方法

    Patient Dropout Prediction in Virtual Health: A Multimodal Dynamic Knowledge Graph and Text Mining Approach. (arXiv:2306.03833v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2306.03833](http://arxiv.org/abs/2306.03833)

    本研究提出了一种多模态动态知识驱动退出预测（MDKDP）框架，能够解决虚拟健康中不同利益相关者之间和医疗保健交付系统之间的信息不对称问题，提高了退出预测的性能。

    

    虚拟健康被誉为医疗保健交付中的改变性力量。然而，它的退出问题是至关重要的，会导致较差的健康结果，增加健康、社会和经济成本。及时预测患者的退出使股东能够采取积极的步骤，解决患者的问题，可能提高保留率。为了解决这些信息不对称问题，我们提出了一种多模态动态知识驱动退出预测（MDKDP）框架，该框架从在线和离线医疗保健交付系统的医生患者对话、各个股东的动态和复杂网络中学习隐式和显式知识。我们通过与中国最大的虚拟健康平台之一合作来评估MDKDP。MDKDP提高了退出预测的性能。

    Virtual health has been acclaimed as a transformative force in healthcare delivery. Yet, its dropout issue is critical that leads to poor health outcomes, increased health, societal, and economic costs. Timely prediction of patient dropout enables stakeholders to take proactive steps to address patients' concerns, potentially improving retention rates. In virtual health, the information asymmetries inherent in its delivery format, between different stakeholders, and across different healthcare delivery systems hinder the performance of existing predictive methods. To resolve those information asymmetries, we propose a Multimodal Dynamic Knowledge-driven Dropout Prediction (MDKDP) framework that learns implicit and explicit knowledge from doctor-patient dialogues and the dynamic and complex networks of various stakeholders in both online and offline healthcare delivery systems. We evaluate MDKDP by partnering with one of the largest virtual health platforms in China. MDKDP improves the 
    

