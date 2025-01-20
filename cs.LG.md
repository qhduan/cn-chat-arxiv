# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [ELITR-Bench: A Meeting Assistant Benchmark for Long-Context Language Models](https://arxiv.org/abs/2403.20262) | 该论文提出了一个新的基准 ELITR-Bench，专注于长上下文语言模型的实际会议助理场景，通过在现有 ELITR 语料库的转录中添加手工制作的问题和真实答案，揭示了开源模型和专有模型之间的差距。 |
| [^2] | [Aligning with Human Judgement: The Role of Pairwise Preference in Large Language Model Evaluators](https://arxiv.org/abs/2403.16950) | 在大型语言模型评估中，通过引入成对偏好搜索方法PAIRS，成功解决了LLMs与人类判断不一致的问题，并取得了优于直接打分的最先进性能。 |
| [^3] | [A Fairness-Oriented Reinforcement Learning Approach for the Operation and Control of Shared Micromobility Services](https://arxiv.org/abs/2403.15780) | 本研究介绍了一种在共享微移动服务运营与控制中实现性能优化和算法公平性平衡的前沿调查，利用Q-Learning算法确保方法稳健，能够实现各种站点类别之间的公平结果。 |
| [^4] | [Bridging Diversity and Uncertainty in Active learning with Self-Supervised Pre-Training](https://arxiv.org/abs/2403.03728) | 通过引入TCM启发式方法，本研究在主动学习中成功结合了多样性采样和不确定性采样策略，解决了冷启动问题并在各种数据水平上表现出色。 |
| [^5] | [Stochastic gradient descent for streaming linear and rectified linear systems with Massart noise](https://arxiv.org/abs/2403.01204) | 我们提出了一种针对具有Massart噪声的线性和ReLU回归问题的随机梯度下降方法，具有新颖的近乎线性收敛保证，首次在流式设置中为鲁棒ReLU回归提供了收敛保证，并展示了其相比于以前的方法有改进的收敛速率。 |
| [^6] | [Keeping LLMs Aligned After Fine-tuning: The Crucial Role of Prompt Templates](https://arxiv.org/abs/2402.18540) | 提出了“纯粹调优，安全测试”（PTST）原则，即在微调时不包含安全提示，但在测试时加入，可以显著减少LLMs中不安全行为的出现。 |
| [^7] | [Anfinsen Goes Neural: a Graphical Model for Conditional Antibody Design](https://arxiv.org/abs/2402.05982) | Anfinsen Goes Neural (AGN) is a graphical model for conditional antibody design that combines a pre-trained protein language model with a graph neural network. It outperforms existing methods and addresses the limitation of generating unrealistic sequences. |
| [^8] | [Two Types of AI Existential Risk: Decisive and Accumulative](https://arxiv.org/abs/2401.07836) | 本文对比了传统的“决定性AI x-risk假设”与“累积性AI x-risk假设”，指出人工智能可能带来的灭绝性灾难有两种可能路径：一种是突然发生的AI接管，另一种是逐渐积累的威胁。 |
| [^9] | [Neural networks for insurance pricing with frequency and severity data: a benchmark study from data preprocessing to technical tariff.](http://arxiv.org/abs/2310.12671) | 本研究通过深度学习结构的神经网络对频率-严重性保险定价进行了基准研究，比较了不同模型的性能，并提出了一种联合精算神经网络(CANN)的方法。 |
| [^10] | [Differentially Private Secure Multiplication: Hiding Information in the Rubble of Noise.](http://arxiv.org/abs/2309.16105) | 本文研究了在分布式计算中，允许信息泄漏和近似乘法的情况下，当诚实节点数量为少数时，差分隐私和准确性之间的权衡关系。 |
| [^11] | [Multi-stage Deep Learning Artifact Reduction for Computed Tomography.](http://arxiv.org/abs/2309.00494) | 本论文提出了一种多阶段深度学习伪影减少方法，用于提高计算机断层扫描的图像质量。传统方法通常在重建之后进行处理，而本方法能够根据不同的图像域进行多步骤去伪影，使得相对困难去除的伪影也能够有效消除。 |
| [^12] | [Automated Machine Learning for Remaining Useful Life Predictions.](http://arxiv.org/abs/2306.12215) | 本文介绍了一种自动化的机器学习方法，名为AutoRUL，用于自动预测工程系统的剩余使用寿命（RUL）。该方法将微调的标准回归方法与高预测能力的集成相结合，并通过八个真实世界的和合成数据集的评估，证明AutoML提供了一种可行的选择。 |
| [^13] | [ThreatCrawl: A BERT-based Focused Crawler for the Cybersecurity Domain.](http://arxiv.org/abs/2304.11960) | 本文提出了一种基于BERT的焦点爬虫ThreatCrawl，使用主题建模和关键词提取技术来筛选出最可能包含有价值CTI信息的网页。 |
| [^14] | [Can AI-Generated Text be Reliably Detected?.](http://arxiv.org/abs/2303.11156) | 本研究通过实证和理论分析表明，在实际场景中，几种AI文本检测器不可靠。改写攻击可以破解多种检测器，包括水印方案、神经网络检测器和零样本分类器。即使是最好的检测器，随着语言模型的进一步提升，性能也会下降。因此，AI生成的文本的可靠检测仍然是一个挑战。 |

# 详细

[^1]: ELITR-Bench: 面向长上下文语言模型的会议助理基准

    ELITR-Bench: A Meeting Assistant Benchmark for Long-Context Language Models

    [https://arxiv.org/abs/2403.20262](https://arxiv.org/abs/2403.20262)

    该论文提出了一个新的基准 ELITR-Bench，专注于长上下文语言模型的实际会议助理场景，通过在现有 ELITR 语料库的转录中添加手工制作的问题和真实答案，揭示了开源模型和专有模型之间的差距。

    

    最近，对大型语言模型（LLMs）的研究越来越受到关注，主要致力于扩展模型的上下文大小，以更好地捕捉长文档内部的依赖关系。尽管已经提出了用于评估长距离能力的基准，但现有的努力主要考虑的是不一定与现实应用相关的通用任务。相反，我们的工作提出了一个针对实际会议助理场景的长上下文LLMs的新基准。在这种情景下，长上下文由自动语音识别获得的转录组成，由于这些数据的固有嘈杂性和口语特性，这为LLMs提出了独特的挑战。我们的基准，名为ELITR-Bench，通过271个手工制作的问题及其真实答案来增强现有的ELITR语料库的转录。我们在ELITR-Bench上对最新的长上下文LLMs进行的实验凸显了开源模型和专有模型之间的差距。

    arXiv:2403.20262v1 Announce Type: cross  Abstract: Research on Large Language Models (LLMs) has recently witnessed an increasing interest in extending models' context size to better capture dependencies within long documents. While benchmarks have been proposed to assess long-range abilities, existing efforts primarily considered generic tasks that are not necessarily aligned with real-world applications. In contrast, our work proposes a new benchmark for long-context LLMs focused on a practical meeting assistant scenario. In this scenario, the long contexts consist of transcripts obtained by automatic speech recognition, presenting unique challenges for LLMs due to the inherent noisiness and oral nature of such data. Our benchmark, named ELITR-Bench, augments the existing ELITR corpus' transcripts with 271 manually crafted questions and their ground-truth answers. Our experiments with recent long-context LLMs on ELITR-Bench highlight a gap between open-source and proprietary models, e
    
[^2]: 与人类判断相一致：大型语言模型评估中成对偏好的作用

    Aligning with Human Judgement: The Role of Pairwise Preference in Large Language Model Evaluators

    [https://arxiv.org/abs/2403.16950](https://arxiv.org/abs/2403.16950)

    在大型语言模型评估中，通过引入成对偏好搜索方法PAIRS，成功解决了LLMs与人类判断不一致的问题，并取得了优于直接打分的最先进性能。

    

    大型语言模型（LLMs）作为自动评估器在评估生成的自然语言质量方面表现出有希望的能力。然而，LLMs在评估中仍存在偏见，常常难以生成与人类评估一致的连贯评估。在这项工作中，我们首先对LLM评估器与人类判断之间的不一致进行系统研究，揭示现有旨在减轻偏见的校准方法不足以有效将LLM评估器对齐。受到RLHF中对偏好数据的使用的启发，我们将评估形式化为一个排序问题，并引入Pairwise-preference Search（PAIRS），这是一种以LLMs进行成对比较并有效对候选文本进行排序的基于不确定性引导的搜索方法。PAIRS在代表性评估任务上实现了最先进的性能，并且显示出比直接打分有显著改进。

    arXiv:2403.16950v1 Announce Type: cross  Abstract: Large Language Models (LLMs) have demonstrated promising capabilities as automatic evaluators in assessing the quality of generated natural language. However, LLMs still exhibit biases in evaluation and often struggle to generate coherent evaluations that align with human assessments. In this work, we first conduct a systematic study of the misalignment between LLM evaluators and human judgement, revealing that existing calibration methods aimed at mitigating biases are insufficient for effectively aligning LLM evaluators. Inspired by the use of preference data in RLHF, we formulate the evaluation as a ranking problem and introduce Pairwise-preference Search (PAIRS), an uncertainty-guided search method that employs LLMs to conduct pairwise comparisons and efficiently ranks candidate texts. PAIRS achieves state-of-the-art performance on representative evaluation tasks and demonstrates significant improvements over direct scoring. Furthe
    
[^3]: 面向公平性的共享微移动服务运营与控制的强化学习方法

    A Fairness-Oriented Reinforcement Learning Approach for the Operation and Control of Shared Micromobility Services

    [https://arxiv.org/abs/2403.15780](https://arxiv.org/abs/2403.15780)

    本研究介绍了一种在共享微移动服务运营与控制中实现性能优化和算法公平性平衡的前沿调查，利用Q-Learning算法确保方法稳健，能够实现各种站点类别之间的公平结果。

    

    随着机器学习系统在各种应用领域变得日益普遍，包括那些直接涉及人类的领域，平等和算法公平性的必要性在人工智能界愈发突出。另一方面，在共享微移动系统的背景下，公平性导向方法的探索仍然有限。为填补这一空白，我们引入了一项探讨性研究，探讨了共享微移动服务运营与控制中性能优化与算法公平性之间的平衡。我们的研究运用强化学习中的Q-Learning算法，利用其收敛保证来确保我们提出的方法的稳健性。值得注意的是，我们的方法在不同站点类别（中心、边缘和远程）之间能够实现公平的结果，这是通过基尼系数来衡量的。

    arXiv:2403.15780v1 Announce Type: cross  Abstract: As Machine Learning systems become increasingly popular across diverse application domains, including those with direct human implications, the imperative of equity and algorithmic fairness has risen to prominence in the Artificial Intelligence community. On the other hand, in the context of Shared Micromobility Systems, the exploration of fairness-oriented approaches remains limited. Addressing this gap, we introduce a pioneering investigation into the balance between performance optimization and algorithmic fairness in the operation and control of Shared Micromobility Services. Our study leverages the Q-Learning algorithm in Reinforcement Learning, benefiting from its convergence guarantees to ensure the robustness of our proposed approach. Notably, our methodology stands out for its ability to achieve equitable outcomes, as measured by the Gini index, across different station categories--central, peripheral, and remote. Through stra
    
[^4]: 通过自监督预训练在主动学习中弥合多样性与不确定性

    Bridging Diversity and Uncertainty in Active learning with Self-Supervised Pre-Training

    [https://arxiv.org/abs/2403.03728](https://arxiv.org/abs/2403.03728)

    通过引入TCM启发式方法，本研究在主动学习中成功结合了多样性采样和不确定性采样策略，解决了冷启动问题并在各种数据水平上表现出色。

    

    本研究探讨了在主动学习中集成基于多样性和基于不确定性的采样策略，特别是在自监督预训练模型的背景下。我们引入了一个称为TCM的简单启发式方法，可以缓解冷启动问题，同时在各种数据水平上保持强大性能。通过首先应用TypiClust进行多样性采样，随后过渡到使用Margin进行不确定性采样，我们的方法有效地结合了两种策略的优势。我们的实验表明，TCM在低数据和高数据情况下始终优于现有方法。

    arXiv:2403.03728v1 Announce Type: cross  Abstract: This study addresses the integration of diversity-based and uncertainty-based sampling strategies in active learning, particularly within the context of self-supervised pre-trained models. We introduce a straightforward heuristic called TCM that mitigates the cold start problem while maintaining strong performance across various data levels. By initially applying TypiClust for diversity sampling and subsequently transitioning to uncertainty sampling with Margin, our approach effectively combines the strengths of both strategies. Our experiments demonstrate that TCM consistently outperforms existing methods across various datasets in both low and high data regimes.
    
[^5]: 具有Massart噪声的流式线性和修正线性系统的随机梯度下降

    Stochastic gradient descent for streaming linear and rectified linear systems with Massart noise

    [https://arxiv.org/abs/2403.01204](https://arxiv.org/abs/2403.01204)

    我们提出了一种针对具有Massart噪声的线性和ReLU回归问题的随机梯度下降方法，具有新颖的近乎线性收敛保证，首次在流式设置中为鲁棒ReLU回归提供了收敛保证，并展示了其相比于以前的方法有改进的收敛速率。

    

    我们提出了SGD-exp，一种用于线性和ReLU回归的随机梯度下降方法，在Massart噪声（对抗性半随机破坏模型）下，完全流式设置下。我们展示了SGD-exp对真实参数的近乎线性收敛保证，最高可达50%的Massart破坏率，在对称无忧破坏情况下，任意破坏率也有保证。这是流式设置中鲁棒ReLU回归的第一个收敛保证结果，它显示了相比于以前的鲁棒方法对于L1线性回归具有改进的收敛速率，这是由于选择了指数衰减步长，这在实践中已被证明是有效的。我们的分析基于离散随机过程的漂移分析，这本身也可能是有趣的。

    arXiv:2403.01204v1 Announce Type: new  Abstract: We propose SGD-exp, a stochastic gradient descent approach for linear and ReLU regressions under Massart noise (adversarial semi-random corruption model) for the fully streaming setting. We show novel nearly linear convergence guarantees of SGD-exp to the true parameter with up to $50\%$ Massart corruption rate, and with any corruption rate in the case of symmetric oblivious corruptions. This is the first convergence guarantee result for robust ReLU regression in the streaming setting, and it shows the improved convergence rate over previous robust methods for $L_1$ linear regression due to a choice of an exponentially decaying step size, known for its efficiency in practice. Our analysis is based on the drift analysis of a discrete stochastic process, which could also be interesting on its own.
    
[^6]: 在微调后保持LLMs的对齐性:提示模板的关键作用

    Keeping LLMs Aligned After Fine-tuning: The Crucial Role of Prompt Templates

    [https://arxiv.org/abs/2402.18540](https://arxiv.org/abs/2402.18540)

    提出了“纯粹调优，安全测试”（PTST）原则，即在微调时不包含安全提示，但在测试时加入，可以显著减少LLMs中不安全行为的出现。

    

    公共LLMs，如Llama 2-Chat，推动了LLM研究的巨大活动。这些模型经历了对齐性训练，被认为是安全的。最近，齐等人（2023年）报告称，即使是良性的微调（例如，在看似安全的数据集上）也可能导致模型产生不安全的行为。本文介绍了减轻这种对齐性丢失的方法和最佳实践。通过对几个聊天模型（Meta的Llama 2-Chat，Mistral AI的Mistral 7B Instruct v0.2和OpenAI的GPT-3.5 Turbo）进行广泛实验，本文发现微调和推理过程中使用的提示模板在保持安全对齐性方面起着至关重要的作用，并提出了“纯粹调优，安全测试”（PTST）原则 - 在测试时不使用安全提示进行模型微调，但在测试时包含它。对GSM8K，ChatDoctor和OpenOrca进行的微调实验表明，PTST显着减少了不安全行为的增加，甚至几乎消除了它们。

    arXiv:2402.18540v1 Announce Type: cross  Abstract: Public LLMs such as the Llama 2-Chat have driven huge activity in LLM research. These models underwent alignment training and were considered safe. Recently Qi et al. (2023) reported that even benign fine-tuning (e.g., on seemingly safe datasets) can give rise to unsafe behaviors in the models. The current paper is about methods and best practices to mitigate such loss of alignment. Through extensive experiments on several chat models (Meta's Llama 2-Chat, Mistral AI's Mistral 7B Instruct v0.2, and OpenAI's GPT-3.5 Turbo), this paper uncovers that the prompt templates used during fine-tuning and inference play a crucial role in preserving safety alignment, and proposes the "Pure Tuning, Safe Testing" (PTST) principle -- fine-tune models without a safety prompt, but include it at test time. Fine-tuning experiments on GSM8K, ChatDoctor, and OpenOrca show that PTST significantly reduces the rise of unsafe behaviors, and even almost elimin
    
[^7]: Anfinsen Goes Neural: 一种用于条件抗体设计的图模型

    Anfinsen Goes Neural: a Graphical Model for Conditional Antibody Design

    [https://arxiv.org/abs/2402.05982](https://arxiv.org/abs/2402.05982)

    Anfinsen Goes Neural (AGN) is a graphical model for conditional antibody design that combines a pre-trained protein language model with a graph neural network. It outperforms existing methods and addresses the limitation of generating unrealistic sequences.

    

    抗体设计在推动治疗学方面起着关键作用。尽管深度学习在这个领域取得了快速进展，但现有方法对一般蛋白质知识的利用有限，并假设图模型违反蛋白质的经验发现。为了解决这些限制，我们提出了Anfinsen Goes Neural (AGN)，这是一个使用预训练的蛋白质语言模型(pLM)并编码了一种关于蛋白质的重要发现，即Anfinsen's dogma的图模型。我们的框架遵循序列生成和图神经网络(GNN)进行结构预测的两步过程。实验证明，我们的方法在基准实验中优于现有方法的结果。我们还解决了非自回归模型的一个关键限制，即它们倾向于生成具有过多重复标记的不现实序列。为了解决这个问题，我们引入了基于组合的正则化项到交叉熵目标中，可以实现有效的权衡。

    Antibody design plays a pivotal role in advancing therapeutics. Although deep learning has made rapid progress in this field, existing methods make limited use of general protein knowledge and assume a graphical model (GM) that violates empirical findings on proteins. To address these limitations, we present Anfinsen Goes Neural (AGN), a graphical model that uses a pre-trained protein language model (pLM) and encodes a seminal finding on proteins called Anfinsen's dogma. Our framework follows a two-step process of sequence generation with pLM and structure prediction with graph neural network (GNN). Experiments show that our approach outperforms state-of-the-art results on benchmark experiments. We also address a critical limitation of non-autoregressive models -- namely, that they tend to generate unrealistic sequences with overly repeating tokens. To resolve this, we introduce a composition-based regularization term to the cross-entropy objective that allows an efficient trade-off be
    
[^8]: 两种类型的人工智能存在风险：决定性和累积性

    Two Types of AI Existential Risk: Decisive and Accumulative

    [https://arxiv.org/abs/2401.07836](https://arxiv.org/abs/2401.07836)

    本文对比了传统的“决定性AI x-risk假设”与“累积性AI x-risk假设”，指出人工智能可能带来的灭绝性灾难有两种可能路径：一种是突然发生的AI接管，另一种是逐渐积累的威胁。

    

    传统上对人工智能(AI)引起的存在风险(x-risks)的讨论通常集中在由先进的AI系统引起的突然、严重事件上，尤其是那些可能达到或超过人类水平智能的系统。这些事件将带来严重后果，要么导致人类灭绝，要么无法逆转地使人类文明陷入无法恢复的状态。然而，这种讨论经常忽视AI x-risk逐渐通过一系列较小但相互关联的中断逐渐显现出来的严重可能性，随着时间的推移逐渐跨越关键阈值。该论文将传统的“决定性AI x-risk假设”与“累积性AI x-risk假设”进行对比。前者描绘了一种明显的AI接管路径，其特征是无法控制的超级智能等情景，而后者则提出了另一种导致灭绝性灾难的因果路径。这涉及到由AI引起的严重威胁的逐渐累积，例如严重的漏洞和系统性问题

    The conventional discourse on existential risks (x-risks) from AI typically focuses on abrupt, dire events caused by advanced AI systems, particularly those that might achieve or surpass human-level intelligence. These events have severe consequences that either lead to human extinction or irreversibly cripple human civilization to a point beyond recovery. This discourse, however, often neglects the serious possibility of AI x-risks manifesting incrementally through a series of smaller yet interconnected disruptions, gradually crossing critical thresholds over time. This paper contrasts the conventional "decisive AI x-risk hypothesis" with an "accumulative AI x-risk hypothesis." While the former envisions an overt AI takeover pathway, characterized by scenarios like uncontrollable superintelligence, the latter suggests a different causal pathway to existential catastrophes. This involves a gradual accumulation of critical AI-induced threats such as severe vulnerabilities and systemic e
    
[^9]: 利用频率和严重性数据进行保险定价的神经网络：从数据预处理到技术定价的基准研究

    Neural networks for insurance pricing with frequency and severity data: a benchmark study from data preprocessing to technical tariff. (arXiv:2310.12671v1 [cs.LG])

    [http://arxiv.org/abs/2310.12671](http://arxiv.org/abs/2310.12671)

    本研究通过深度学习结构的神经网络对频率-严重性保险定价进行了基准研究，比较了不同模型的性能，并提出了一种联合精算神经网络(CANN)的方法。

    

    保险公司通常使用广义线性模型来建模索赔的频率和严重性数据。由于其在其他领域的成功，机器学习技术在精算工具箱中越来越受欢迎。本文通过深度学习结构为频率-严重性保险定价与机器学习相关的文献做出了贡献。我们在四个保险数据集上进行了基准研究，这些数据集包含有多种类型的输入特征和频率-严重性目标。我们详细比较了广义线性模型在分箱输入数据、梯度提升树模型、前馈神经网络（FFNN）和联合精算神经网络（CANN）上的性能。我们的CANN将通过GLM和GBM分别建立的基线预测与神经网络校正相结合。我们解释了数据预处理步骤，特别关注通常存在于表格保险数据集中的多种类型的输入特征，比如邮编和数字编码。

    Insurers usually turn to generalized linear models for modelling claim frequency and severity data. Due to their success in other fields, machine learning techniques are gaining popularity within the actuarial toolbox. Our paper contributes to the literature on frequency-severity insurance pricing with machine learning via deep learning structures. We present a benchmark study on four insurance data sets with frequency and severity targets in the presence of multiple types of input features. We compare in detail the performance of: a generalized linear model on binned input data, a gradient-boosted tree model, a feed-forward neural network (FFNN), and the combined actuarial neural network (CANN). Our CANNs combine a baseline prediction established with a GLM and GBM, respectively, with a neural network correction. We explain the data preprocessing steps with specific focus on the multiple types of input features typically present in tabular insurance data sets, such as postal codes, nu
    
[^10]: 差分隐私安全乘法：在噪声中隐藏信息

    Differentially Private Secure Multiplication: Hiding Information in the Rubble of Noise. (arXiv:2309.16105v1 [cs.IT])

    [http://arxiv.org/abs/2309.16105](http://arxiv.org/abs/2309.16105)

    本文研究了在分布式计算中，允许信息泄漏和近似乘法的情况下，当诚实节点数量为少数时，差分隐私和准确性之间的权衡关系。

    

    我们考虑私密分布式多方乘法的问题。已经确认，Shamir秘密共享编码策略可以通过Ben Or，Goldwasser，Wigderson算法（“BGW算法”）在分布式计算中实现完美的信息理论隐私。然而，完美的隐私和准确性需要一个诚实的多数，即需要$N \geq 2t+1$个计算节点以确保对抗性节点的隐私。我们通过允许一定量的信息泄漏和近似乘法来研究在诚实节点数量为少数时的编码方案，即$N< 2t+1$。我们通过使用差分隐私而不是完美隐私来测量信息泄漏，并使用均方误差度量准确性，对$N < 2t+1$的情况下的隐私-准确性权衡进行了紧密的刻画。一个新颖的技术方面是复杂地控制信息泄漏的细节。

    We consider the problem of private distributed multi-party multiplication. It is well-established that Shamir secret-sharing coding strategies can enable perfect information-theoretic privacy in distributed computation via the celebrated algorithm of Ben Or, Goldwasser and Wigderson (the "BGW algorithm"). However, perfect privacy and accuracy require an honest majority, that is, $N \geq 2t+1$ compute nodes are required to ensure privacy against any $t$ colluding adversarial nodes. By allowing for some controlled amount of information leakage and approximate multiplication instead of exact multiplication, we study coding schemes for the setting where the number of honest nodes can be a minority, that is $N< 2t+1.$ We develop a tight characterization privacy-accuracy trade-off for cases where $N < 2t+1$ by measuring information leakage using {differential} privacy instead of perfect privacy, and using the mean squared error metric for accuracy. A novel technical aspect is an intricately 
    
[^11]: 计算机断层扫描的多阶段深度学习伪影减少

    Multi-stage Deep Learning Artifact Reduction for Computed Tomography. (arXiv:2309.00494v1 [eess.IV])

    [http://arxiv.org/abs/2309.00494](http://arxiv.org/abs/2309.00494)

    本论文提出了一种多阶段深度学习伪影减少方法，用于提高计算机断层扫描的图像质量。传统方法通常在重建之后进行处理，而本方法能够根据不同的图像域进行多步骤去伪影，使得相对困难去除的伪影也能够有效消除。

    

    在计算机断层扫描中，通过一系列获取的投影图像计算出物体内部结构的图像。这些重建图像的质量对于准确分析至关重要，但是这种质量可能会被各种成像伪影降低。为了提高重建质量，获取的投影图像通常通过由多个去伪影步骤组成的流程进行处理，这些步骤应用于不同的图像域（例如，投影图像的异常值去除和重建图像的去噪）。这些伪影去除方法利用了某些伪影在特定域相对于其他域更容易去除的事实。最近，深度学习方法在计算机断层扫描伪影去除方面取得了有希望的结果。然而，大多数现有的计算机断层扫描深度学习方法都是在重建之后作为后处理方法应用的。因此，在重建域相对困难去除的伪影可能无法有效去除。

    In Computed Tomography (CT), an image of the interior structure of an object is computed from a set of acquired projection images. The quality of these reconstructed images is essential for accurate analysis, but this quality can be degraded by a variety of imaging artifacts. To improve reconstruction quality, the acquired projection images are often processed by a pipeline consisting of multiple artifact-removal steps applied in various image domains (e.g., outlier removal on projection images and denoising of reconstruction images). These artifact-removal methods exploit the fact that certain artifacts are easier to remove in a certain domain compared with other domains.  Recently, deep learning methods have shown promising results for artifact removal for CT images. However, most existing deep learning methods for CT are applied as a post-processing method after reconstruction. Therefore, artifacts that are relatively difficult to remove in the reconstruction domain may not be effec
    
[^12]: 面向剩余使用寿命预测的自动化机器学习

    Automated Machine Learning for Remaining Useful Life Predictions. (arXiv:2306.12215v1 [cs.LG])

    [http://arxiv.org/abs/2306.12215](http://arxiv.org/abs/2306.12215)

    本文介绍了一种自动化的机器学习方法，名为AutoRUL，用于自动预测工程系统的剩余使用寿命（RUL）。该方法将微调的标准回归方法与高预测能力的集成相结合，并通过八个真实世界的和合成数据集的评估，证明AutoML提供了一种可行的选择。

    

    预测工程系统的剩余使用寿命（RUL）是预测与健康管理中的重要任务。最近，数据驱动的方法在RUL预测中普及，相比模型驱动的方法不需要工程系统的物理知识。但是，这只是将需要的物理专业知识替换成机器学习（ML）专业知识，而这种专业知识通常也不可得。自动化机器学习（AutoML）承诺自动构建端到端的ML管道，使领域专家而非ML专家能够创建自己的模型。本文介绍了AutoRUL，一种AutoML驱动的端到端方法，用于自动RUL预测。AutoRUL将微调的标准回归方法与高预测能力的集成相结合。通过将所提出的方法用于八个真实世界的和合成数据集，与最先进的手工模型进行比较，我们表明AutoML提供了一种可行的选择。

    Being able to predict the remaining useful life (RUL) of an engineering system is an important task in prognostics and health management. Recently, data-driven approaches to RUL predictions are becoming prevalent over model-based approaches since no underlying physical knowledge of the engineering system is required. Yet, this just replaces required expertise of the underlying physics with machine learning (ML) expertise, which is often also not available. Automated machine learning (AutoML) promises to build end-to-end ML pipelines automatically enabling domain experts without ML expertise to create their own models. This paper introduces AutoRUL, an AutoML-driven end-to-end approach for automatic RUL predictions. AutoRUL combines fine-tuned standard regression methods to an ensemble with high predictive power. By evaluating the proposed method on eight real-world and synthetic datasets against state-of-the-art hand-crafted models, we show that AutoML provides a viable alternative to 
    
[^13]: ThreatCrawl：基于BERT的网络安全焦点爬虫

    ThreatCrawl: A BERT-based Focused Crawler for the Cybersecurity Domain. (arXiv:2304.11960v2 [cs.CR] UPDATED)

    [http://arxiv.org/abs/2304.11960](http://arxiv.org/abs/2304.11960)

    本文提出了一种基于BERT的焦点爬虫ThreatCrawl，使用主题建模和关键词提取技术来筛选出最可能包含有价值CTI信息的网页。

    

    可公开获取的信息对于网络威胁情报（CTI）来说包含有价值的信息。这可以用于预防已经在其他系统上发生的攻击。但是，虽然有不同的标准来交流这些信息，但很多信息是以非标准化的方式在文章或博客帖子中共享的。手动浏览多个在线门户和新闻页面以发现新威胁并提取它们是一项耗时的任务。为了自动化这个扫描过程的一部分，多篇论文提出了使用自然语言处理（NLP）从文档中提取威胁指示器（IOCs）的提取器。然而，虽然这已经解决了从文档中提取信息的问题，但很少考虑搜索这些文档。本文提出了一种新的焦点爬虫ThreatCrawl，它使用双向编码器表示（BERT）搜索网络安全领域中的相关文档。ThreatCrawl使用主题建模和关键词提取技术来识别相关网站和网页，然后应用基于BERT的分类器来优先考虑最可能包含有价值CTI信息的网页。

    Publicly available information contains valuable information for Cyber Threat Intelligence (CTI). This can be used to prevent attacks that have already taken place on other systems. Ideally, only the initial attack succeeds and all subsequent ones are detected and stopped. But while there are different standards to exchange this information, a lot of it is shared in articles or blog posts in non-standardized ways. Manually scanning through multiple online portals and news pages to discover new threats and extracting them is a time-consuming task. To automize parts of this scanning process, multiple papers propose extractors that use Natural Language Processing (NLP) to extract Indicators of Compromise (IOCs) from documents. However, while this already solves the problem of extracting the information out of documents, the search for these documents is rarely considered. In this paper, a new focused crawler is proposed called ThreatCrawl, which uses Bidirectional Encoder Representations 
    
[^14]: AI生成的文本是否可靠地检测出来？

    Can AI-Generated Text be Reliably Detected?. (arXiv:2303.11156v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2303.11156](http://arxiv.org/abs/2303.11156)

    本研究通过实证和理论分析表明，在实际场景中，几种AI文本检测器不可靠。改写攻击可以破解多种检测器，包括水印方案、神经网络检测器和零样本分类器。即使是最好的检测器，随着语言模型的进一步提升，性能也会下降。因此，AI生成的文本的可靠检测仍然是一个挑战。

    

    本文从实证和理论两个方面表明，在实际场景中，几种AI文本检测器并不可靠。从实践上来说，我们证明了轻量级的改写器应用在大型语言模型（LLM）上可以破解一系列的检测器，包括使用水印方案、神经网络检测器和零样本分类器。我们的实验表明，旨在躲避改写攻击的基于检索的检测器仍然容易受到递归改写的攻击。然后，我们提出了一个理论上的不可能结果，指出随着语言模型变得越来越复杂和更擅长模仿人类文本，在最好的检测器性能会下降。对于一个足够先进的语言模型来模仿人类文本，即使最佳的检测器的表现只比随机分类器好上一点点。我们的结果足够概括特定的场景，如改写攻击。

    In this paper, both empirically and theoretically, we show that several AI-text detectors are not reliable in practical scenarios. Empirically, we show that paraphrasing attacks, where a light paraphraser is applied on top of a large language model (LLM), can break a whole range of detectors, including ones using watermarking schemes as well as neural network-based detectors and zero-shot classifiers. Our experiments demonstrate that retrieval-based detectors, designed to evade paraphrasing attacks, are still vulnerable to recursive paraphrasing. We then provide a theoretical impossibility result indicating that as language models become more sophisticated and better at emulating human text, the performance of even the best-possible detector decreases. For a sufficiently advanced language model seeking to imitate human text, even the best-possible detector may only perform marginally better than a random classifier. Our result is general enough to capture specific scenarios such as par
    

