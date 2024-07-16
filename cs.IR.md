# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Scaling Laws For Dense Retrieval](https://arxiv.org/abs/2403.18684) | 该研究探究了密集检索模型的性能是否遵循其他神经模型的缩放规律，并提出使用对比对数似然作为评估指标进行了广泛实验。 |
| [^2] | [Leveraging Large Language Models for Relevance Judgments in Legal Case Retrieval](https://arxiv.org/abs/2403.18405) | 设计一种新颖的几轮工作流程，专门用于法律案例的相关判断，能够通过模仿人类注释者的过程并整合专家推理，提高相关性判断的准确性。 |
| [^3] | [LLM-Assisted Multi-Teacher Continual Learning for Visual Question Answering in Robotic Surgery](https://arxiv.org/abs/2402.16664) | LLM辅助的多教师持续学习为机器人手术中的视觉问答系统更新提供了解决新任务需求的方法，同时解决了外科领域中的大领域转变和数据不平衡问题。 |
| [^4] | [A Roadmap to Pluralistic Alignment](https://arxiv.org/abs/2402.05070) | 这篇论文提出了一条通向多元对齐的路线图，以解决设计AI系统能够服务于人们具有不同价值观和观点的需求。论文介绍了对齐定义和实现多元主义的三种方式，并提出了三种多元基准类别来评估和测试多元对齐的效果。 |
| [^5] | [Uncertainty Calibration for Counterfactual Propensity Estimation in Recommendation.](http://arxiv.org/abs/2303.12973) | 本文提出了多种不确定性校准技术，以改进推荐系统中倾向性估计的效果。经过实验验证，校准后的IPS估计器在Coat和yahoo数据集上表现更好。 |

# 详细

[^1]: 密集检索的扩展规律

    Scaling Laws For Dense Retrieval

    [https://arxiv.org/abs/2403.18684](https://arxiv.org/abs/2403.18684)

    该研究探究了密集检索模型的性能是否遵循其他神经模型的缩放规律，并提出使用对比对数似然作为评估指标进行了广泛实验。

    

    将神经模型扩展到更大规模已经在多项任务中取得了显著进展，特别是在语言生成方面。先前的研究发现，神经模型的性能常遵循可预测的扩展规律，与训练集大小和模型大小等因素相关。这一洞察力非常宝贵，尤其是随着大规模实验变得越来越耗费资源。然而，由于检索指标的离散性以及检索任务中训练数据和模型大小之间的复杂关系，密集检索中的这种扩展规律尚未得到充分探讨。在本研究中，我们调查了密集检索模型的性能是否遵循其他神经模型的缩放规律。我们建议使用对比对数似然作为评估指标，并对实现了不同参数数量并使用不同数量的数据训练的密集检索模型进行了广泛实验。

    arXiv:2403.18684v1 Announce Type: cross  Abstract: Scaling up neural models has yielded significant advancements in a wide array of tasks, particularly in language generation. Previous studies have found that the performance of neural models frequently adheres to predictable scaling laws, correlated with factors such as training set size and model size. This insight is invaluable, especially as large-scale experiments grow increasingly resource-intensive. Yet, such scaling law has not been fully explored in dense retrieval due to the discrete nature of retrieval metrics and complex relationships between training data and model sizes in retrieval tasks. In this study, we investigate whether the performance of dense retrieval models follows the scaling law as other neural models. We propose to use contrastive log-likelihood as the evaluation metric and conduct extensive experiments with dense retrieval models implemented with different numbers of parameters and trained with different amo
    
[^2]: 利用大型语言模型进行法律案例检索中的相关性判断

    Leveraging Large Language Models for Relevance Judgments in Legal Case Retrieval

    [https://arxiv.org/abs/2403.18405](https://arxiv.org/abs/2403.18405)

    设计一种新颖的几轮工作流程，专门用于法律案例的相关判断，能够通过模仿人类注释者的过程并整合专家推理，提高相关性判断的准确性。

    

    收集法律案例检索的相关判决是一项具有挑战性且耗时的任务。准确判断两个法律案例之间的相关性需要阅读冗长的文本并具备高水平的领域专业知识以提取法律事实并作出司法判断。随着先进的大型语言模型的出现，一些最近的研究表明使用LLM（Large Language Models）进行相关性判断是有前途的。然而，将一般性大型语言模型应用于法律案例检索中可靠的相关性判断的方法尚未得到充分探讨。为了填补这一研究空白，我们设计了一种新颖的几轮工作流程，专门用于法律案例的相关判断。所提出的工作流程将注释过程分解为一系列阶段，模仿人类注释者所使用的过程，并使专家推理能够灵活地整合以增强相关性判断的准确性。

    arXiv:2403.18405v1 Announce Type: new  Abstract: Collecting relevant judgments for legal case retrieval is a challenging and time-consuming task. Accurately judging the relevance between two legal cases requires a considerable effort to read the lengthy text and a high level of domain expertise to extract Legal Facts and make juridical judgments. With the advent of advanced large language models, some recent studies have suggested that it is promising to use LLMs for relevance judgment. Nonetheless, the method of employing a general large language model for reliable relevance judgments in legal case retrieval is yet to be thoroughly explored. To fill this research gap, we devise a novel few-shot workflow tailored to the relevant judgment of legal cases. The proposed workflow breaks down the annotation process into a series of stages, imitating the process employed by human annotators and enabling a flexible integration of expert reasoning to enhance the accuracy of relevance judgments.
    
[^3]: LLM辅助的多教师持续学习在机器人手术中的视觉问答

    LLM-Assisted Multi-Teacher Continual Learning for Visual Question Answering in Robotic Surgery

    [https://arxiv.org/abs/2402.16664](https://arxiv.org/abs/2402.16664)

    LLM辅助的多教师持续学习为机器人手术中的视觉问答系统更新提供了解决新任务需求的方法，同时解决了外科领域中的大领域转变和数据不平衡问题。

    

    视觉问答(VQA)在促进机器人辅助手术教育方面可能至关重要。在实践中，学员的需求不断发展，比如学习更多种类的手术，适应不同的机器人，以及为一种手术学习新的外科器械和技术。因此，在机器人手术中需要通过多个资源的顺序数据流持续更新VQA系统，以解决新任务。在外科场景中，存储成本和患者数据隐私通常限制了在更新模型时旧数据的可用性，这需要一个无样本的持续学习(CL)设置。然而，先前的研究忽视了外科领域的两个重要问题：i)来自不同科室或临床中心收集的各种外科手术的大领域转变，ii)由于外科器械或活动的不均匀出现而导致的严重数据不平衡。

    arXiv:2402.16664v1 Announce Type: new  Abstract: Visual question answering (VQA) can be fundamentally crucial for promoting robotic-assisted surgical education. In practice, the needs of trainees are constantly evolving, such as learning more surgical types, adapting to different robots, and learning new surgical instruments and techniques for one surgery. Therefore, continually updating the VQA system by a sequential data stream from multiple resources is demanded in robotic surgery to address new tasks. In surgical scenarios, the storage cost and patient data privacy often restrict the availability of old data when updating the model, necessitating an exemplar-free continual learning (CL) setup. However, prior studies overlooked two vital problems of the surgical domain: i) large domain shifts from diverse surgical operations collected from multiple departments or clinical centers, and ii) severe data imbalance arising from the uneven presence of surgical instruments or activities du
    
[^4]: 通往多元对齐的路线图

    A Roadmap to Pluralistic Alignment

    [https://arxiv.org/abs/2402.05070](https://arxiv.org/abs/2402.05070)

    这篇论文提出了一条通向多元对齐的路线图，以解决设计AI系统能够服务于人们具有不同价值观和观点的需求。论文介绍了对齐定义和实现多元主义的三种方式，并提出了三种多元基准类别来评估和测试多元对齐的效果。

    

    随着人工智能系统的权力和普及程度的增加，设计能够为不同价值观和观点的人服务的人工智能系统变得愈发重要。然而，将模型对齐以服务多元人类价值观仍然是一个待解决的研究问题。在本文中，我们提出了一条通向多元对齐的路线图，具体使用语言模型作为测试平台。我们确定和形式化了三种可能的方式来定义和实现人工智能系统中的多元主义：1）Overton多元模型，展示合理反应的光谱；2）可操控的多元模型，可以调整以反映特定的观点；3）分布多元模型，在分布中很好地校准给定人群的模型。我们还提出和形式化了三种可能的多元基准类别：1）多目标基准；2）权衡可操控基准，鼓励模型对任意权衡进行调整；3）陪审团多元基准，明确地模拟了不同陪审团的意见。

    With increased power and prevalence of AI systems, it is ever more critical that AI systems are designed to serve all, i.e., people with diverse values and perspectives. However, aligning models to serve pluralistic human values remains an open research question. In this piece, we propose a roadmap to pluralistic alignment, specifically using language models as a test bed. We identify and formalize three possible ways to define and operationalize pluralism in AI systems: 1) Overton pluralistic models that present a spectrum of reasonable responses; 2) Steerably pluralistic models that can steer to reflect certain perspectives; and 3) Distributionally pluralistic models that are well-calibrated to a given population in distribution. We also propose and formalize three possible classes of pluralistic benchmarks: 1) Multi-objective benchmarks, 2) Trade-off steerable benchmarks, which incentivize models to steer to arbitrary trade-offs, and 3) Jury-pluralistic benchmarks which explicitly m
    
[^5]: 推荐系统中反事实倾向估计的不确定性校准

    Uncertainty Calibration for Counterfactual Propensity Estimation in Recommendation. (arXiv:2303.12973v1 [cs.AI])

    [http://arxiv.org/abs/2303.12973](http://arxiv.org/abs/2303.12973)

    本文提出了多种不确定性校准技术，以改进推荐系统中倾向性估计的效果。经过实验验证，校准后的IPS估计器在Coat和yahoo数据集上表现更好。

    

    在推荐系统中，由于选择偏差，许多评分信息都丢失了，这被称为非随机缺失。反事实逆倾向评分（IPS）被用于衡量每个观察到的评分的填充错误。虽然在多种情况下有效，但我们认为IPS估计的性能受到倾向性估计不确定性的限制。本文提出了多种代表性的不确定性校准技术，以改进推荐系统中倾向性估计的不确定性校准。通过对偏误和推广界限的理论分析表明，经过校准的IPS估计器优于未校准的IPS估计器。 Coat和yahoo数据集上的实验结果表明，不确定性校准得到改进，从而使推荐结果更好。

    In recommendation systems, a large portion of the ratings are missing due to the selection biases, which is known as Missing Not At Random. The counterfactual inverse propensity scoring (IPS) was used to weight the imputation error of every observed rating. Although effective in multiple scenarios, we argue that the performance of IPS estimation is limited due to the uncertainty miscalibration of propensity estimation. In this paper, we propose the uncertainty calibration for the propensity estimation in recommendation systems with multiple representative uncertainty calibration techniques. Theoretical analysis on the bias and generalization bound shows the superiority of the calibrated IPS estimator over the uncalibrated one. Experimental results on the coat and yahoo datasets shows that the uncertainty calibration is improved and hence brings the better recommendation results.
    

