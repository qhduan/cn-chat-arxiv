# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [How many views does your deep neural network use for prediction?](https://rss.arxiv.org/abs/2402.01095) | 本文提出了最小有效视图（MSVs）的概念，该概念类似于多视图，但适用于实际图像，并且通过实证研究表明，MSV的数量与模型的预测准确性之间存在关系。 |
| [^2] | [An Analysis of Switchback Designs in Reinforcement Learning](https://arxiv.org/abs/2403.17285) | 本文通过提出“弱信号分析”框架，研究了强化学习中往返设计对平均处理效应估计准确性的影响，发现在大部分奖励误差为正相关时，往返设计比每日切换策略更有效，增加政策切换频率可以降低平均处理效应估计器的均方误差。 |
| [^3] | [Efficient exploration of high-Tc superconductors by a gradient-based composition design](https://arxiv.org/abs/2403.13627) | 通过基于成分的梯度优化，使模型与目标性质密切对齐，实现高效、广泛的搜索，并能适应新的约束条件，显著进展了材料设计。 |
| [^4] | [Accelerated Inference and Reduced Forgetting: The Dual Benefits of Early-Exit Networks in Continual Learning](https://arxiv.org/abs/2403.07404) | 早期退出网络在持续学习中展现出降低遗忘和在资源利用上表现优异的特点 |
| [^5] | [Robustness-Congruent Adversarial Training for Secure Machine Learning Model Updates](https://arxiv.org/abs/2402.17390) | 通过鲁棒一致对抗训练技术，解决了更新机器学习模型时对抗性鲁棒性和系统安全性的问题。 |
| [^6] | [Building Flexible Machine Learning Models for Scientific Computing at Scale](https://arxiv.org/abs/2402.16014) | OmniArch通过多物理学时空数据处理、可扩展的自回归任务和物理信息增强学习技术，在科学计算领域构建灵活的基础模型，并在性能、适应性和逆问题求解方面取得突破，展现了AI对科学计算的潜力。 |
| [^7] | [Learning to Poison Large Language Models During Instruction Tuning](https://arxiv.org/abs/2402.13459) | 通过设计新的数据注入攻击攻击LLMs，并提出一种梯度引导后门触发器学习方法，通过实验验证表明成功地破坏模型输出，仅改变1%的指导调优样本即可导致性能下降率达到约80％。 |
| [^8] | [Language Agents with Reinforcement Learning for Strategic Play in the Werewolf Game](https://arxiv.org/abs/2310.18940) | 使用强化学习为基础，提出了一种框架，用于在狼人杀游戏中开发具有灵活语言行为和强大决策能力的战略语言代理 |
| [^9] | [NACHOS: Neural Architecture Search for Hardware Constrained Early Exit Neural Networks.](http://arxiv.org/abs/2401.13330) | NACHOS 是一种面向硬件受限的早期退出神经网络的神经架构搜索方法，可以自动化设计早期退出神经网络并考虑骨干和早期退出分类器之间的关系。 |
| [^10] | [Theoretical guarantees on the best-of-n alignment policy.](http://arxiv.org/abs/2401.01879) | 该论文研究了对齐生成模型的最佳n对齐策略，并证明了之前文献中的某个分析表达式是错误的。研究者们提出了一个新的KL散度估计方法，并通过实验证明其有效性。 |
| [^11] | [Ensuring User-side Fairness in Dynamic Recommender Systems.](http://arxiv.org/abs/2308.15651) | 本文提出了一种名为FADE的端到端框架，通过微调策略动态减轻推荐系统中用户群体之间的性能差异。 |
| [^12] | [Towards Model-Agnostic Federated Learning over Networks.](http://arxiv.org/abs/2302.04363) | 该论文提出了一种适用于网络环境中多种数据和模型的模型无关关联学习方法，旨在通过网络结构反映本地数据集的相似性并保证本地模型产生一致的预测结果。 |

# 详细

[^1]: 深度神经网络预测使用多少个视图？

    How many views does your deep neural network use for prediction?

    [https://rss.arxiv.org/abs/2402.01095](https://rss.arxiv.org/abs/2402.01095)

    本文提出了最小有效视图（MSVs）的概念，该概念类似于多视图，但适用于实际图像，并且通过实证研究表明，MSV的数量与模型的预测准确性之间存在关系。

    

    尽管进行了许多理论和实证分析，但深度神经网络（DNN）的泛化能力仍未完全理解。最近，Allen-Zhu和Li（2023）引入了多视图的概念来解释DNN的泛化能力，但他们的主要目标是集成或蒸馏模型，并未讨论用于特定输入预测的多视图估计方法。在本文中，我们提出了最小有效视图（MSVs），它类似于多视图，但可以高效地计算真实图像。MSVs是输入中的一组最小且不同的特征，每个特征保留了模型对该输入的预测。我们通过实证研究表明，不同模型（包括卷积和转换模型）的MSV数量与预测准确性之间存在明确的关系，这表明多视图的角度对于理解（非集成或非蒸馏）DNN的泛化能力也很重要。

    The generalization ability of Deep Neural Networks (DNNs) is still not fully understood, despite numerous theoretical and empirical analyses. Recently, Allen-Zhu & Li (2023) introduced the concept of multi-views to explain the generalization ability of DNNs, but their main target is ensemble or distilled models, and no method for estimating multi-views used in a prediction of a specific input is discussed. In this paper, we propose Minimal Sufficient Views (MSVs), which is similar to multi-views but can be efficiently computed for real images. MSVs is a set of minimal and distinct features in an input, each of which preserves a model's prediction for the input. We empirically show that there is a clear relationship between the number of MSVs and prediction accuracy across models, including convolutional and transformer models, suggesting that a multi-view like perspective is also important for understanding the generalization ability of (non-ensemble or non-distilled) DNNs.
    
[^2]: 对强化学习中的往返设计进行的分析

    An Analysis of Switchback Designs in Reinforcement Learning

    [https://arxiv.org/abs/2403.17285](https://arxiv.org/abs/2403.17285)

    本文通过提出“弱信号分析”框架，研究了强化学习中往返设计对平均处理效应估计准确性的影响，发现在大部分奖励误差为正相关时，往返设计比每日切换策略更有效，增加政策切换频率可以降低平均处理效应估计器的均方误差。

    

    本文提供了对A/B测试中往返设计的详细研究，这些设计随时间在基准和新策略之间交替。我们的目标是全面评估这些设计对其产生的平均处理效应（ATE）估计器准确性的影响。我们提出了一个新颖的“弱信号分析”框架，大大简化了这些ATE的均方误差（MSE）在马尔科夫决策过程环境中的计算。我们的研究结果表明：(i) 当大部分奖励误差呈正相关时，往返设计比每日切换策略的交替设计更有效。此外，增加政策切换的频率往往会降低ATE估计器的MSE。(ii) 然而，当误差不相关时，所有这些设计变得渐近等效。(iii) 在大多数误差为负相关时

    arXiv:2403.17285v1 Announce Type: cross  Abstract: This paper offers a detailed investigation of switchback designs in A/B testing, which alternate between baseline and new policies over time. Our aim is to thoroughly evaluate the effects of these designs on the accuracy of their resulting average treatment effect (ATE) estimators. We propose a novel "weak signal analysis" framework, which substantially simplifies the calculations of the mean squared errors (MSEs) of these ATEs in Markov decision process environments. Our findings suggest that (i) when the majority of reward errors are positively correlated, the switchback design is more efficient than the alternating-day design which switches policies in a daily basis. Additionally, increasing the frequency of policy switches tends to reduce the MSE of the ATE estimator. (ii) When the errors are uncorrelated, however, all these designs become asymptotically equivalent. (iii) In cases where the majority of errors are negative correlate
    
[^3]: 基于梯度的成分设计方法在高临界温度超导体的高效探索中的应用

    Efficient exploration of high-Tc superconductors by a gradient-based composition design

    [https://arxiv.org/abs/2403.13627](https://arxiv.org/abs/2403.13627)

    通过基于成分的梯度优化，使模型与目标性质密切对齐，实现高效、广泛的搜索，并能适应新的约束条件，显著进展了材料设计。

    

    我们提出了一种材料设计方法，通过基于成分的梯度优化，克服了传统方法的局限性：穷举数据库搜索和条件生成模型。它通过反向传播优化输入，使模型的输出与目标性质密切对齐，有助于发现未列出的材料和精确性质确定。我们的方法还能够在新条件下进行自适应优化，无需重新训练。应用于探索高临界温度超导体时，我们识别出超过现有数据库的潜在成分，并通过条件优化发现了新的氢超导体。这种方法是多功能的，通过实现高效、广泛的搜索和对新约束条件的适应性，显着推进了材料设计。

    arXiv:2403.13627v1 Announce Type: cross  Abstract: We propose a material design method via gradient-based optimization on compositions, overcoming the limitations of traditional methods: exhaustive database searches and conditional generation models. It optimizes inputs via backpropagation, aligning the model's output closely with the target property and facilitating the discovery of unlisted materials and precise property determination. Our method is also capable of adaptive optimization under new conditions without retraining. Applying to exploring high-Tc superconductors, we identified potential compositions beyond existing databases and discovered new hydrogen superconductors via conditional optimization. This method is versatile and significantly advances material design by enabling efficient, extensive searches and adaptability to new constraints.
    
[^4]: 提高推理速度和减少遗忘：早期退出网络在持续学习中的双重好处

    Accelerated Inference and Reduced Forgetting: The Dual Benefits of Early-Exit Networks in Continual Learning

    [https://arxiv.org/abs/2403.07404](https://arxiv.org/abs/2403.07404)

    早期退出网络在持续学习中展现出降低遗忘和在资源利用上表现优异的特点

    

    arXiv:2403.07404v1 公告类型: 跨界 摘要: 受深度神经网络能源高效利用需求驱动，早期退出方法备受关注。这些策略通过在网络早期做出决定，实现快速预测，从而节省计算时间和资源。然而，迄今为止，早期退出网络仅针对静态数据分布进行了开发，限制了它们在具有持续非静态数据的实际场景中的应用。本研究旨在探讨早期退出网络的持续学习。我们改编现有的持续学习方法以适应早期退出架构，并研究它们在持续设置中的行为。我们注意到，早期网络层表现出减少遗忘，即使使用的资源显著更少，也能胜过标准网络。此外，我们分析任务最近性偏差对早期退出推理的影响，并提出任务...

    arXiv:2403.07404v1 Announce Type: cross  Abstract: Driven by the demand for energy-efficient employment of deep neural networks, early-exit methods have experienced a notable increase in research attention. These strategies allow for swift predictions by making decisions early in the network, thereby conserving computation time and resources. However, so far the early-exit networks have only been developed for stationary data distributions, which restricts their application in real-world scenarios with continuous non-stationary data. This study aims to explore the continual learning of the early-exit networks. We adapt existing continual learning methods to fit with early-exit architectures and investigate their behavior in the continual setting. We notice that early network layers exhibit reduced forgetting and can outperform standard networks even when using significantly fewer resources. Furthermore, we analyze the impact of task-recency bias on early-exit inference and propose Task
    
[^5]: 针对安全机器学习模型更新的鲁棒一致对抗训练

    Robustness-Congruent Adversarial Training for Secure Machine Learning Model Updates

    [https://arxiv.org/abs/2402.17390](https://arxiv.org/abs/2402.17390)

    通过鲁棒一致对抗训练技术，解决了更新机器学习模型时对抗性鲁棒性和系统安全性的问题。

    

    机器学习模型需要定期更新以提高其平均准确度，利用新颖的架构和额外的数据。然而，新更新的模型可能会犯以前模型未曾犯过的错误。这种误分类被称为负翻转，并被用户体验为性能的退化。在本文中，我们展示了这个问题也影响对抗性样本的鲁棒性，从而阻碍了安全模型更新实践的发展。特别是，当更新模型以提高其对抗性鲁棒性时，一些先前无效的对抗性样本可能会被错误分类，导致系统安全性的认知退化。我们提出了一种名为鲁棒一致对抗训练的新技术来解决这个问题。它涉及使用对抗训练对模型进行微调，同时约束其在对抗性示例上保持更高的鲁棒性。

    arXiv:2402.17390v1 Announce Type: new  Abstract: Machine-learning models demand for periodic updates to improve their average accuracy, exploiting novel architectures and additional data. However, a newly-updated model may commit mistakes that the previous model did not make. Such misclassifications are referred to as negative flips, and experienced by users as a regression of performance. In this work, we show that this problem also affects robustness to adversarial examples, thereby hindering the development of secure model update practices. In particular, when updating a model to improve its adversarial robustness, some previously-ineffective adversarial examples may become misclassified, causing a regression in the perceived security of the system. We propose a novel technique, named robustness-congruent adversarial training, to address this issue. It amounts to fine-tuning a model with adversarial training, while constraining it to retain higher robustness on the adversarial examp
    
[^6]: 在科学计算规模上构建灵活的机器学习模型

    Building Flexible Machine Learning Models for Scientific Computing at Scale

    [https://arxiv.org/abs/2402.16014](https://arxiv.org/abs/2402.16014)

    OmniArch通过多物理学时空数据处理、可扩展的自回归任务和物理信息增强学习技术，在科学计算领域构建灵活的基础模型，并在性能、适应性和逆问题求解方面取得突破，展现了AI对科学计算的潜力。

    

    arXiv:2402.16014v1

    arXiv:2402.16014v1 Announce Type: cross  Abstract: Foundation models have revolutionized knowledge acquisition across domains, and our study introduces OmniArch, a paradigm-shifting approach designed for building foundation models in multi-physics scientific computing. OmniArch's pre-training involves a versatile pipeline that processes multi-physics spatio-temporal data, casting forward problem learning into scalable auto-regressive tasks, while our novel Physics-Informed Reinforcement Learning (PIRL) technique during fine-tuning ensures alignment with physical laws. Pre-trained on the comprehensive PDEBench dataset, OmniArch not only sets new performance benchmarks for 1D, 2D and 3D PDEs but also demonstrates exceptional adaptability to new physics via few-shot and zero-shot learning approaches. The model's representations further extend to inverse problem-solving, highlighting the transformative potential of AI-enabled Scientific Computing(AI4SC) foundation models for engineering ap
    
[^7]: 学习在指导调优期间操纵大型语言模型

    Learning to Poison Large Language Models During Instruction Tuning

    [https://arxiv.org/abs/2402.13459](https://arxiv.org/abs/2402.13459)

    通过设计新的数据注入攻击攻击LLMs，并提出一种梯度引导后门触发器学习方法，通过实验验证表明成功地破坏模型输出，仅改变1%的指导调优样本即可导致性能下降率达到约80％。

    

    大型语言模型（LLMs）的出现标志着语言处理和推理能力方面的重大突破。虽然它们取得了显著进展，但LLMs面临着数据注入攻击的漏洞，其中对手将后门触发器插入训练数据，以操纵输出以进行恶意行为。本研究通过设计一种新的数据注入攻击，旨在利用指导调优过程，进一步识别LLMs中的额外安全风险。我们提出了一种新颖的梯度引导后门触发器学习方法，以有效识别敌对触发器，确保对传统防御手段的规避，同时保持内容的完整性。通过对各种LLMs和任务的实验验证，我们的策略表明在破坏模型输出方面取得了很高的成功率；仅对4,000个指导调优样本中的1％进行注入就导致性能降低率（PDR）约为80％。我们的工作高

    arXiv:2402.13459v1 Announce Type: cross  Abstract: The advent of Large Language Models (LLMs) has marked significant achievements in language processing and reasoning capabilities. Despite their advancements, LLMs face vulnerabilities to data poisoning attacks, where adversaries insert backdoor triggers into training data to manipulate outputs for malicious purposes. This work further identifies additional security risks in LLMs by designing a new data poisoning attack tailored to exploit the instruction tuning process. We propose a novel gradient-guided backdoor trigger learning approach to identify adversarial triggers efficiently, ensuring an evasion of detection by conventional defenses while maintaining content integrity. Through experimental validation across various LLMs and tasks, our strategy demonstrates a high success rate in compromising model outputs; poisoning only 1\% of 4,000 instruction tuning samples leads to a Performance Drop Rate (PDR) of around 80\%. Our work high
    
[^8]: 使用强化学习的语言代理在狼人杀游戏中进行战略对战

    Language Agents with Reinforcement Learning for Strategic Play in the Werewolf Game

    [https://arxiv.org/abs/2310.18940](https://arxiv.org/abs/2310.18940)

    使用强化学习为基础，提出了一种框架，用于在狼人杀游戏中开发具有灵活语言行为和强大决策能力的战略语言代理

    

    基于大型语言模型（LLMs）构建的代理在各领域展现出巨大潜力。然而，在复杂决策任务中，纯LLM代理往往表现出固有偏见，这些偏见来源于模型的训练数据，导致性能不佳。为了开发具有灵活语言行为和强大决策能力的战略语言代理，我们提出了一种新颖的框架，通过强化学习（RL）提升LLM代理的能力。我们选择狼人杀作为具有多样沟通和战略游戏玩法的挑战测试平台。为了减轻语言行为中的固有偏见，我们的代理使用LLM进行演绎推理并生成多样行为候选集。然后，经过训练以优化决策能力的RL策略从候选集中选择行为。

    arXiv:2310.18940v3 Announce Type: replace  Abstract: Agents built with large language models (LLMs) have shown great potential across a wide range of domains. However, in complex decision-making tasks, pure LLM-based agents tend to exhibit intrinsic bias in their choice of actions, which is inherited from the model's training data and results in suboptimal performance. To develop strategic language agents, i.e., agents that generate flexible language actions and possess strong decision-making abilities, we propose a novel framework that powers LLM-based agents with reinforcement learning (RL). We consider Werewolf, a popular social deduction game, as a challenging testbed that emphasizes versatile communication and strategic gameplay. To mitigate the intrinsic bias in language actions, our agents use an LLM to perform deductive reasoning and generate a diverse set of action candidates. Then an RL policy trained to optimize the decision-making ability chooses an action from the candidat
    
[^9]: NACHOS: 硬件受限的早期退出神经网络的神经架构搜索

    NACHOS: Neural Architecture Search for Hardware Constrained Early Exit Neural Networks. (arXiv:2401.13330v1 [cs.LG])

    [http://arxiv.org/abs/2401.13330](http://arxiv.org/abs/2401.13330)

    NACHOS 是一种面向硬件受限的早期退出神经网络的神经架构搜索方法，可以自动化设计早期退出神经网络并考虑骨干和早期退出分类器之间的关系。

    

    早期退出神经网络（EENNs）为标准的深度神经网络（DNN）配备早期退出分类器（EECs），在处理的中间点上提供足够的分类置信度时进行预测。这在效果和效率方面带来了许多好处。目前，EENNs的设计是由专家手动完成的，这是一项复杂和耗时的任务，需要考虑许多方面，包括正确的放置、阈值设置和EECs的计算开销。因此，研究正在探索使用神经架构搜索（NAS）自动化设计EENNs。目前，文献中提出了几个完整的NAS解决方案用于EENNs，并且一个完全自动化的综合设计策略，同时考虑骨干和EECs仍然是一个未解决的问题。为此，本研究呈现了面向硬件受限的早期退出神经网络的神经架构搜索（NACHOS）。

    Early Exit Neural Networks (EENNs) endow astandard Deep Neural Network (DNN) with Early Exit Classifiers (EECs), to provide predictions at intermediate points of the processing when enough confidence in classification is achieved. This leads to many benefits in terms of effectiveness and efficiency. Currently, the design of EENNs is carried out manually by experts, a complex and time-consuming task that requires accounting for many aspects, including the correct placement, the thresholding, and the computational overhead of the EECs. For this reason, the research is exploring the use of Neural Architecture Search (NAS) to automatize the design of EENNs. Currently, few comprehensive NAS solutions for EENNs have been proposed in the literature, and a fully automated, joint design strategy taking into consideration both the backbone and the EECs remains an open problem. To this end, this work presents Neural Architecture Search for Hardware Constrained Early Exit Neural Networks (NACHOS),
    
[^10]: 关于最佳n对齐策略的理论保证

    Theoretical guarantees on the best-of-n alignment policy. (arXiv:2401.01879v1 [cs.LG])

    [http://arxiv.org/abs/2401.01879](http://arxiv.org/abs/2401.01879)

    该论文研究了对齐生成模型的最佳n对齐策略，并证明了之前文献中的某个分析表达式是错误的。研究者们提出了一个新的KL散度估计方法，并通过实验证明其有效性。

    

    一个简单有效的生成模型对齐方法是最佳n对齐策略，该策略从一个基本策略中抽取n个样本，并根据奖励函数对它们进行排序，选择排名最高的样本。文献中常用的分析表达式声称最佳n对齐策略与基本策略之间的KL散度等于$\log (n) (n-1)/n$。我们证明了该论断的不正确性，并展示了它只是实际KL散度的一个上界。我们还研究了在不同情况下该上界的紧致性。最后，我们提出了一种新的KL散度估计方法，并通过几个例子的实验证明它能提供一个紧致的近似。

    A simple and effective method for the alignment of generative models is the best-of-$n$ policy, where $n$ samples are drawn from a base policy, and ranked based on a reward function, and the highest ranking one is selected. A commonly used analytical expression in the literature claims that the KL divergence between the best-of-$n$ policy and the base policy is equal to $\log (n) (n-1)/n.$ We disprove the validity of this claim, and show that it is an upper bound on the actual KL divergence. We also explore the tightness of this upper bound in different regimes. Finally, we propose a new estimator for the KL divergence and empirically show that it provides a tight approximation through a few examples.
    
[^11]: 在动态推荐系统中确保用户侧公平性

    Ensuring User-side Fairness in Dynamic Recommender Systems. (arXiv:2308.15651v1 [cs.IR])

    [http://arxiv.org/abs/2308.15651](http://arxiv.org/abs/2308.15651)

    本文提出了一种名为FADE的端到端框架，通过微调策略动态减轻推荐系统中用户群体之间的性能差异。

    

    用户侧群体公平性对现代推荐系统至关重要，它旨在减轻由敏感属性（如性别、种族或年龄）定义的用户群体之间的性能差异。我们发现这种差异往往会随着时间的推移而持续存在甚至增加。这需要在动态环境中有效解决用户侧公平性的方法，然而这在文献中很少被探讨。然而，用于确保用户侧公平性（即减少性能差异）的典型方法——公平约束重新排名，在动态设定中面临两个基本挑战：（1）基于排名的公平约束的非可微性，阻碍了端到端训练范式；（2）时间效率低下，阻碍了对用户偏好变化的快速适应。在本文中，我们提出了一种名为FADE的端到端框架，通过微调策略动态减轻性能差异。为了解决上述挑战，FADE提出了一种 fine-tuning 策略。

    User-side group fairness is crucial for modern recommender systems, as it aims to alleviate performance disparity between groups of users defined by sensitive attributes such as gender, race, or age. We find that the disparity tends to persist or even increase over time. This calls for effective ways to address user-side fairness in a dynamic environment, which has been infrequently explored in the literature. However, fairness-constrained re-ranking, a typical method to ensure user-side fairness (i.e., reducing performance disparity), faces two fundamental challenges in the dynamic setting: (1) non-differentiability of the ranking-based fairness constraint, which hinders the end-to-end training paradigm, and (2) time-inefficiency, which impedes quick adaptation to changes in user preferences. In this paper, we propose FAir Dynamic rEcommender (FADE), an end-to-end framework with fine-tuning strategy to dynamically alleviate performance disparity. To tackle the above challenges, FADE u
    
[^12]: 探索网络上的模型无关联合学习

    Towards Model-Agnostic Federated Learning over Networks. (arXiv:2302.04363v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.04363](http://arxiv.org/abs/2302.04363)

    该论文提出了一种适用于网络环境中多种数据和模型的模型无关关联学习方法，旨在通过网络结构反映本地数据集的相似性并保证本地模型产生一致的预测结果。

    

    我们提出了一种适用于异构数据和模型网络的模型无关联合学习方法。网络结构反映了本地数据集（统计数据）和它们相关的本地模型之间的相似性。我们的方法是经验风险最小化的一种实例，其中正则化项是从数据的网络结构导出的。特别地，我们要求良好连接的本地模型形成聚类，在一个公共测试集上产生相似的预测结果。所提出的方法允许使用各种各样的本地模型。 对这些本地模型唯一的限制是它们允许有效实现正则化的经验风险最小化（训练）。对于各种模型，这样的实现都可以在高级编程库（包括scikit-learn、Keras或PyTorch）中找到。

    We present a model-agnostic federated learning method for networks of heterogeneous data and models. The network structure reflects similarities between the (statistics of) local datasets and, in turn, their associated local("personal") models. Our method is an instance of empirical risk minimization, with the regularization term derived from the network structure of data. In particular, we require well-connected local models, forming clusters, to yield similar predictions on a common test set. The proposed method allows for a wide range of local models. The only restriction on these local models is that they allow for efficient implementation of regularized empirical risk minimization (training). For a wide range of models, such implementations are available in high-level programming libraries including scikit-learn, Keras or PyTorch.
    

