# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Learning General Policies for Classical Planning Domains: Getting Beyond C$_2$](https://arxiv.org/abs/2403.11734) | 该研究提出了一种参数化版本的关系GNNs，通过在$t$为无穷大时仅使用二次空间的嵌入来近似$3$-GNNs，对于较低的$t$值，通过交换较少的消息实现弱的近似，同时通常产生了几个规划领域中所需的$C_3$特征。 |
| [^2] | [AutoMathText: Autonomous Data Selection with Language Models for Mathematical Texts](https://arxiv.org/abs/2402.07625) | 本论文介绍了一种自主数据选择策略，利用语言模型进行数学文本的自动评估和选择，并通过连续预训练显著提高了数学推理能力。主要创新包括利用元提示语言模型作为验证器，发布了高质量的AutoMathText数据集，并实现了预训练令牌效率的提升。 |
| [^3] | [A Learning-based Declarative Privacy-Preserving Framework for Federated Data Management.](http://arxiv.org/abs/2401.12393) | 本论文提出了一个基于学习的声明性隐私保护框架，通过使用Differentially-Private Stochastic Gradient Descent（DP-SGD）算法训练的深度学习模型替代部分实际数据来回答查询，并允许用户指定要保护的私人信息。此框架还可以自动选择转换计划和超参数，并允许人工专家审核和调整隐私保护机制。 |
| [^4] | [Multimodal Sentiment Analysis with Missing Modality: A Knowledge-Transfer Approach.](http://arxiv.org/abs/2401.10747) | 本文提出了一种知识迁移方法，用于在缺失模态下进行多模态情感分析。通过翻译不同模态之间的内容以重构缺失的音频模态，并利用跨模态注意机制进行情感预测，实验证明了该方法在多个数据集上表现出显著的改进和与完整多模态监督方法相媲美的效果。 |
| [^5] | [Approximate Multiagent Reinforcement Learning for On-Demand Urban Mobility Problem on a Large Map (extended version).](http://arxiv.org/abs/2311.01534) | 本文研究了大型城市环境下的自主多智能体出租车路径问题，提出了一个近似滚动为基础的两阶段算法来减少计算量。 |
| [^6] | [Evaluating LLMs for Privilege-Escalation Scenarios.](http://arxiv.org/abs/2310.11409) | 本研究评估了在特权升级场景中利用语言模型（LLMs）进行渗透测试的应用。通过创建一个自动化的Linux特权升级基准和一个LLM-guided特权升级工具，我们分析了LLMs的不同提示设计、上下文学习和高级指导对测试的影响，并讨论了LLMs面临的挑战。 |
| [^7] | [A Framework for Responsible Development of Automated Student Feedback with Generative AI.](http://arxiv.org/abs/2308.15334) | 一种基于生成AI的自动学生反馈框架可以提供丰富的反馈，但引入了伦理问题，并需要解决“多数人的暴政”和忽视长尾中少数群体需求的挑战。 |
| [^8] | [Do humans and machines have the same eyes? Human-machine perceptual differences on image classification.](http://arxiv.org/abs/2304.08733) | 本文研究通过图像分类探究了人机感知差异，发现即使准确率相似，人类和机器的答案分布也可能不同，并提出了一种后期人机合作来提高任务表现。 |
| [^9] | [Granular-ball Optimization Algorithm.](http://arxiv.org/abs/2303.12807) | 粒球优化算法(GBO)是一种新的多粒度优化算法，可以通过引入粒球计算来提高全局搜索能力和收敛速度，实验结果表明，在这些方面它比现有的最先进的算法表现更优。 |

# 详细

[^1]: 学习古典规划领域的通用策略：超越$C_2$

    Learning General Policies for Classical Planning Domains: Getting Beyond C$_2$

    [https://arxiv.org/abs/2403.11734](https://arxiv.org/abs/2403.11734)

    该研究提出了一种参数化版本的关系GNNs，通过在$t$为无穷大时仅使用二次空间的嵌入来近似$3$-GNNs，对于较低的$t$值，通过交换较少的消息实现弱的近似，同时通常产生了几个规划领域中所需的$C_3$特征。

    

    基于GNN的方法用于学习跨规划领域的通用策略受到$C_2$表达能力的限制，即一阶逻辑只能包含两个变量和计数。这种限制可以通过转向$k$-GNNs，其中$k=3$，其中物体嵌入被三元组嵌入所替换，来克服。然而，尽管$3$-GNNs具有$C_3$的表达能力，但不同于受限于$C_2$的$1$-和$2$-GNNs，它们需要四次时间进行消息交换和三次空间进行嵌入，使它们变得不切实际。在这项工作中，我们引入了一个参数化版本的关系GNNs。当$t$为无穷大时，R-GNN[$t$]仅使用二次空间的嵌入来近似$3$-GNNs。对于较低的$t$值，例如$t=1$和$t=2$，R-GNN[$t$]通过交换较少的消息实现了更弱的近似，但有趣的是，通常产生了在几个规划领域中所需的$C_3$特征。此外，新的R-GNN[$t$] ar

    arXiv:2403.11734v1 Announce Type: new  Abstract: GNN-based approaches for learning general policies across planning domains are limited by the expressive power of $C_2$, namely; first-order logic with two variables and counting. This limitation can be overcomed by transitioning to $k$-GNNs, for $k=3$, wherein object embeddings are substituted with triplet embeddings. Yet, while $3$-GNNs have the expressive power of $C_3$, unlike $1$- and $2$-GNNs that are confined to $C_2$, they require quartic time for message exchange and cubic space for embeddings, rendering them impractical. In this work, we introduce a parameterized version of relational GNNs. When $t$ is infinity, R-GNN[$t$] approximates $3$-GNNs using only quadratic space for embeddings. For lower values of $t$, such as $t=1$ and $t=2$, R-GNN[$t$] achieves a weaker approximation by exchanging fewer messages, yet interestingly, often yield the $C_3$ features required in several planning domains. Furthermore, the new R-GNN[$t$] ar
    
[^2]: AutoMathText：使用语言模型进行数学文本的自主数据选择

    AutoMathText: Autonomous Data Selection with Language Models for Mathematical Texts

    [https://arxiv.org/abs/2402.07625](https://arxiv.org/abs/2402.07625)

    本论文介绍了一种自主数据选择策略，利用语言模型进行数学文本的自动评估和选择，并通过连续预训练显著提高了数学推理能力。主要创新包括利用元提示语言模型作为验证器，发布了高质量的AutoMathText数据集，并实现了预训练令牌效率的提升。

    

    为了通过持续的预训练改善语言模型在数学推理方面的能力，我们引入了一种新颖的策略，利用基础语言模型进行自主数据选择。与传统的有人工标注数据的监督微调或训练过的分类器不同，我们的方法利用元提示语言模型作为零样本验证器，自主评估和选择高质量的数学内容，并发布了经过策划的开源AutoMathText数据集，其中包含超过200GB的数据。为了证明我们方法的有效性，我们对AutoMathText数据集进行了连续预训练，使得7B参数的Mistral语言模型在MATH数据集上的下游性能大幅提升，而令牌数量比之前的连续预训练工作减少了几个数量级。我们的方法展示了基准的预训练令牌效率提高了2倍，突显了我们方法在增强中的潜力。

    To improve language models' proficiency in mathematical reasoning via continual pretraining, we introduce a novel strategy that leverages base language models for autonomous data selection. Departing from conventional supervised fine-tuning or trained classifiers with human-annotated data, our approach utilizes meta-prompted language models as zero-shot verifiers to autonomously evaluate and select high-quality mathematical content, and we release the curated open-source AutoMathText dataset encompassing over 200GB of data. To demonstrate the efficacy of our method, we continuously pretrained a 7B-parameter Mistral language model on the AutoMathText dataset, achieving substantial improvements in downstream performance on the MATH dataset with a token amount reduced by orders of magnitude compared to previous continuous pretraining works. Our method showcases a 2 times increase in pretraining token efficiency compared to baselines, underscoring the potential of our approach in enhancing
    
[^3]: 基于学习的声明性隐私保护数据联邦管理框架

    A Learning-based Declarative Privacy-Preserving Framework for Federated Data Management. (arXiv:2401.12393v1 [cs.DB])

    [http://arxiv.org/abs/2401.12393](http://arxiv.org/abs/2401.12393)

    本论文提出了一个基于学习的声明性隐私保护框架，通过使用Differentially-Private Stochastic Gradient Descent（DP-SGD）算法训练的深度学习模型替代部分实际数据来回答查询，并允许用户指定要保护的私人信息。此框架还可以自动选择转换计划和超参数，并允许人工专家审核和调整隐私保护机制。

    

    在多个私有数据孤岛上进行联邦查询处理时，平衡隐私和准确性是一项具有挑战性的任务。在这项工作中，我们将演示一种自动化新兴隐私保护技术的端到端工作流，该技术使用使用差分隐私随机梯度下降（DP-SGD）算法训练的深度学习模型替换实际数据的部分来回答查询。我们提出的新颖声明性隐私保护工作流允许用户指定“要保护的私人信息”而不是“如何保护”。在底层，系统自动选择查询-模型转换计划以及超参数。同时，所提出的工作流还允许人工专家审核和调整选择的隐私保护机制，用于审计/合规和优化目的。

    It is challenging to balance the privacy and accuracy for federated query processing over multiple private data silos. In this work, we will demonstrate an end-to-end workflow for automating an emerging privacy-preserving technique that uses a deep learning model trained using the Differentially-Private Stochastic Gradient Descent (DP-SGD) algorithm to replace portions of actual data to answer a query. Our proposed novel declarative privacy-preserving workflow allows users to specify "what private information to protect" rather than "how to protect". Under the hood, the system automatically chooses query-model transformation plans as well as hyper-parameters. At the same time, the proposed workflow also allows human experts to review and tune the selected privacy-preserving mechanism for audit/compliance, and optimization purposes.
    
[^4]: 缺失模态下的多模态情感分析:一种知识迁移方法

    Multimodal Sentiment Analysis with Missing Modality: A Knowledge-Transfer Approach. (arXiv:2401.10747v1 [cs.SD])

    [http://arxiv.org/abs/2401.10747](http://arxiv.org/abs/2401.10747)

    本文提出了一种知识迁移方法，用于在缺失模态下进行多模态情感分析。通过翻译不同模态之间的内容以重构缺失的音频模态，并利用跨模态注意机制进行情感预测，实验证明了该方法在多个数据集上表现出显著的改进和与完整多模态监督方法相媲美的效果。

    

    多模态情感分析旨在通过视觉、语言和声音线索来识别个体表达的情绪。然而，现有研究大多假设在训练和测试过程中所有模态都是可用的，这使得它们的算法容易受到缺失模态的影响。在本文中，我们提出了一种新颖的知识迁移网络，用于在不同模态之间进行翻译，以重构缺失的音频模态。此外，我们还开发了一种跨模态注意机制，以保留重构和观察到的模态的最大信息，用于情感预测。在三个公开数据集上进行的大量实验证明了相对于基线算法的显著改进，并实现了与具有完整多模态监督的先前方法相媲美的结果。

    Multimodal sentiment analysis aims to identify the emotions expressed by individuals through visual, language, and acoustic cues. However, most of the existing research efforts assume that all modalities are available during both training and testing, making their algorithms susceptible to the missing modality scenario. In this paper, we propose a novel knowledge-transfer network to translate between different modalities to reconstruct the missing audio modalities. Moreover, we develop a cross-modality attention mechanism to retain the maximal information of the reconstructed and observed modalities for sentiment prediction. Extensive experiments on three publicly available datasets demonstrate significant improvements over baselines and achieve comparable results to the previous methods with complete multi-modality supervision.
    
[^5]: 大型地图上的按需城市出行问题的近似多智能体强化学习（扩展版）

    Approximate Multiagent Reinforcement Learning for On-Demand Urban Mobility Problem on a Large Map (extended version). (arXiv:2311.01534v1 [cs.MA])

    [http://arxiv.org/abs/2311.01534](http://arxiv.org/abs/2311.01534)

    本文研究了大型城市环境下的自主多智能体出租车路径问题，提出了一个近似滚动为基础的两阶段算法来减少计算量。

    

    本文关注大型城市环境下的自主多智能体出租车路径问题，未来乘车请求的位置和数量事先未知，但遵循估计的经验分布。最近的理论表明，如果基础策略是稳定的，那么基于滚动的算法与这样的基础策略产生接近最优的稳定策略。尽管基于滚动的方法非常适合学习具有对未来需求考虑的合作多智能体策略，但将这些方法应用于大型城市环境可能计算上很昂贵。大型环境往往有大量请求，因此需要大型的出租车队保证稳定性。本文旨在解决多智能体（逐一）滚动的计算瓶颈问题，其中计算复杂性随代理数量线性增长。我们提出了一种近似逐一滚动为基础的两阶段算法，减少计算量

    In this paper, we focus on the autonomous multiagent taxi routing problem for a large urban environment where the location and number of future ride requests are unknown a-priori, but follow an estimated empirical distribution. Recent theory has shown that if a base policy is stable then a rollout-based algorithm with such a base policy produces a near-optimal stable policy. Although, rollout-based approaches are well-suited for learning cooperative multiagent policies with considerations for future demand, applying such methods to a large urban environment can be computationally expensive. Large environments tend to have a large volume of requests, and hence require a large fleet of taxis to guarantee stability. In this paper, we aim to address the computational bottleneck of multiagent (one-at-a-time) rollout, where the computational complexity grows linearly in the number of agents. We propose an approximate one-at-a-time rollout-based two-phase algorithm that reduces the computatio
    
[^6]: 评估LLMs在特权升级场景中的应用

    Evaluating LLMs for Privilege-Escalation Scenarios. (arXiv:2310.11409v2 [cs.CR] UPDATED)

    [http://arxiv.org/abs/2310.11409](http://arxiv.org/abs/2310.11409)

    本研究评估了在特权升级场景中利用语言模型（LLMs）进行渗透测试的应用。通过创建一个自动化的Linux特权升级基准和一个LLM-guided特权升级工具，我们分析了LLMs的不同提示设计、上下文学习和高级指导对测试的影响，并讨论了LLMs面临的挑战。

    

    渗透测试是网络安全的一个重要组成部分，它允许组织主动识别和修复系统中的漏洞，从而增强其对潜在网络攻击的防御机制。在渗透测试领域，最近的一个进展是利用语言模型（LLMs）。我们探索LLMs与渗透测试的交叉领域，以了解它们在特权升级场景中的能力和挑战。我们使用本地虚拟机创建了一个自动化的Linux特权升级基准。我们引入了一种基于LLMs的特权升级工具，用于评估不同的LLMs和提示策略在我们的基准测试中的表现。我们分析了不同提示设计的影响，上下文学习的好处，以及向LLMs提供高级指导的优势。我们讨论了LLMs面临的挑战领域，包括在测试过程中保持专注、处理错误以及与传统方法进行比较。

    Penetration testing, an essential component of cybersecurity, allows organizations to proactively identify and remediate vulnerabilities in their systems, thus bolstering their defense mechanisms against potential cyberattacks. One recent advancement in the realm of penetration testing is the utilization of Language Models (LLMs). We explore the intersection of LLMs and penetration testing to gain insight into their capabilities and challenges in the context of privilige escalation. We create an automated Linux privilege-escalation benchmark utilizing local virtual machines. We introduce an LLM-guided privilege-escalation tool designed for evaluating different LLMs and prompt strategies against our benchmark. We analyze the impact of different prompt designs, the benefits of in-context learning, and the advantages of offering high-level guidance to LLMs. We discuss challenging areas for LLMs, including maintaining focus during testing, coping with errors, and finally comparing them wit
    
[^7]: 一种负责任开发基于生成AI的自动学生反馈框架

    A Framework for Responsible Development of Automated Student Feedback with Generative AI. (arXiv:2308.15334v1 [cs.CY])

    [http://arxiv.org/abs/2308.15334](http://arxiv.org/abs/2308.15334)

    一种基于生成AI的自动学生反馈框架可以提供丰富的反馈，但引入了伦理问题，并需要解决“多数人的暴政”和忽视长尾中少数群体需求的挑战。

    

    提供丰富的反馈对于支持学生学习至关重要。最近生成AI尤其是大规模语言模型的进展，为向学生提供可重复、可扩展和即时生成的自动反馈提供了机会，使得之前稀缺且昂贵的学习资源变得丰富起来。从技术角度而言，这种方法是可行的，得益于最近人工智能和自然语言处理的进步；然而，采用这些技术也引入了一系列潜在的伦理问题，需要认真考虑。人工智能系统的吸引力在于它们可以有效地自动化最乏味的任务；但是这也可能导致“多数人的暴政”，即忽视了长尾中少数群体的需求，因为这些需求很难自动化。因此，开发能够产生有价值和真实的机器学习模型变得至关重要。

    Providing rich feedback to students is essential for supporting student learning. Recent advances in generative AI, particularly within large language modelling (LLM), provide the opportunity to deliver repeatable, scalable and instant automatically generated feedback to students, making abundant a previously scarce and expensive learning resource. Such an approach is feasible from a technical perspective due to these recent advances in Artificial Intelligence (AI) and Natural Language Processing (NLP); while the potential upside is a strong motivator, doing so introduces a range of potential ethical issues that must be considered as we apply these technologies. The attractiveness of AI systems is that they can effectively automate the most mundane tasks; but this risks introducing a "tyranny of the majority", where the needs of minorities in the long tail are overlooked because they are difficult to automate.  Developing machine learning models that can generate valuable and authentic
    
[^8]: 人类和机器有相同的眼睛吗？基于图像分类的人机感知差异研究

    Do humans and machines have the same eyes? Human-machine perceptual differences on image classification. (arXiv:2304.08733v1 [cs.CV])

    [http://arxiv.org/abs/2304.08733](http://arxiv.org/abs/2304.08733)

    本文研究通过图像分类探究了人机感知差异，发现即使准确率相似，人类和机器的答案分布也可能不同，并提出了一种后期人机合作来提高任务表现。

    

    训练良好的计算机视觉模型通常通过模仿从训练标签中学到的人类行为来解决视觉任务。近期视觉研究的大部分努力集中在使用标准化基准来测量模型任务性能。然而，了解人与机器之间的感知差异方面的工作还很有限。为了填补这一空白，我们的研究首先量化并分析了两种来源错误的统计分布。然后我们通过难度级别对任务进行排序，探讨人类与机器专业知识的差异。即使人类和机器的整体准确性相似，答案的分布也可能会有所不同。利用人类和机器之间的感知差异，我们通过实证研究表明了一种后期人机合作，其表现比单独的人或机器更好。

    Trained computer vision models are assumed to solve vision tasks by imitating human behavior learned from training labels. Most efforts in recent vision research focus on measuring the model task performance using standardized benchmarks. Limited work has been done to understand the perceptual difference between humans and machines. To fill this gap, our study first quantifies and analyzes the statistical distributions of mistakes from the two sources. We then explore human vs. machine expertise after ranking tasks by difficulty levels. Even when humans and machines have similar overall accuracies, the distribution of answers may vary. Leveraging the perceptual difference between humans and machines, we empirically demonstrate a post-hoc human-machine collaboration that outperforms humans or machines alone.
    
[^9]: 粒球优化算法

    Granular-ball Optimization Algorithm. (arXiv:2303.12807v1 [cs.LG])

    [http://arxiv.org/abs/2303.12807](http://arxiv.org/abs/2303.12807)

    粒球优化算法(GBO)是一种新的多粒度优化算法，可以通过引入粒球计算来提高全局搜索能力和收敛速度，实验结果表明，在这些方面它比现有的最先进的算法表现更优。

    

    现有的智能优化算法都是基于最小粒度即点的设计，导致全局搜索能力较弱且效率低下。为了解决这个问题，我们提出了一种新的多粒度优化算法，即粒球优化算法(GBO)，通过引入粒球计算来实现。GBO使用多个粒球来覆盖解空间，使用许多细小的细粒度粒球来描述重要部分，使用少量的大粗粒度粒球来描述不重要的部分，精细的多粒度数据描述能力提高了全局搜索能力和收敛速度。针对二十个基准函数的实验结果表明，与最流行的最先进的算法相比，GBO具有更好的性能和更快的速度，更接近最优解，没有超参数，设计更简单。

    The existing intelligent optimization algorithms are designed based on the finest granularity, i.e., a point. This leads to weak global search ability and inefficiency. To address this problem, we proposed a novel multi-granularity optimization algorithm, namely granular-ball optimization algorithm (GBO), by introducing granular-ball computing. GBO uses many granular-balls to cover the solution space. Quite a lot of small and fine-grained granular-balls are used to depict the important parts, and a little number of large and coarse-grained granular-balls are used to depict the inessential parts. Fine multi-granularity data description ability results in a higher global search capability and faster convergence speed. In comparison with the most popular and state-of-the-art algorithms, the experiments on twenty benchmark functions demonstrate its better performance. The faster speed, higher approximation ability of optimal solution, no hyper-parameters, and simpler design of GBO make it 
    

