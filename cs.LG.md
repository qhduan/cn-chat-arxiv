# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Climbing the Ladder of Interpretability with Counterfactual Concept Bottleneck Models](https://rss.arxiv.org/abs/2402.01408) | 本论文提出了一种新的模型 CF-CBMs，可以同时解决深度学习模型的预测、解释和想象能力的不足，为部署可靠的AI代理、校准人类信任和加深人机交互提供了一种有效的解决方法。 |
| [^2] | [Understanding Why Label Smoothing Degrades Selective Classification and How to Fix It](https://arxiv.org/abs/2403.14715) | LS方法在深度神经网络分类器训练中的标签平滑效果被发现会负面影响选择性分类，通过影响模型预测不确定性，此研究阐明了这一现象。 |
| [^3] | [Electioneering the Network: Dynamic Multi-Step Adversarial Attacks for Community Canvassing](https://arxiv.org/abs/2403.12399) | 本论文提出了用于社区拉票的动态多步对抗性攻击，使得对手能够利用基于梯度的攻击来策划目标选民的操纵。 |
| [^4] | [Substrate Scope Contrastive Learning: Repurposing Human Bias to Learn Atomic Representations](https://arxiv.org/abs/2402.16882) | 提出了一种新颖的预训练策略，底物范围对比学习，以学习适合化学反应性的原子表示。 |
| [^5] | [Uncertainty Quantification in Anomaly Detection with Cross-Conformal $p$-Values](https://arxiv.org/abs/2402.16388) | 针对异常检测系统中不确定性量化的需求，提出了一种新颖的框架，称为交叉一致异常检测，通过校准模型的不确定性提供统计保证。 |
| [^6] | [DP-SGD for non-decomposable objective functions.](http://arxiv.org/abs/2310.03104) | 本论文提出了一种针对非可分的目标函数的DP-SGD方法，解决了使用差分隐私进行训练时，相似性损失函数的$L_2$敏感度增长随着批量大小增加的问题。 |
| [^7] | [On Memorization in Diffusion Models.](http://arxiv.org/abs/2310.02664) | 本论文研究了扩散模型的记忆化行为，发现记忆化倾向于在较小的数据集上发生。通过定义有效模型记忆化 (EMM) 这一指标，量化了数据分布和模型配置对记忆化行为的影响。 |
| [^8] | [SEA: Shareable and Explainable Attribution for Query-based Black-box Attacks.](http://arxiv.org/abs/2308.11845) | 本论文提出了SEA，一种用于归因基于查询的黑盒攻击的机器学习安全系统，通过利用隐藏马尔可夫模型框架来理解攻击的演变过程，并有效归因攻击，即使是对于第二次出现的攻击，具有鲁棒性，旨在实现取证和人类可解释的情报共享。 |
| [^9] | [On the Effective Horizon of Inverse Reinforcement Learning.](http://arxiv.org/abs/2307.06541) | 本研究分析了逆强化学习中时间视野的重要性，发现短于实际值的有效时间视野可以更快且更准确地估计奖励函数，减轻过拟合问题。此外，研究还呼吁在IRL中同时学习奖励和有效时间视野。 |
| [^10] | [Robust Classification of High-Dimensional Data using Data-Adaptive Energy Distance.](http://arxiv.org/abs/2306.13985) | 该论文提出了一种用于高维低样本量数据分类的稳健的数据自适应能量距离分类器，该分类器无需调参且在一定条件下可以实现完美分类，已在模拟研究和实际数据分析中得到证明比其他方法表现更优。 |
| [^11] | [UniASM: Binary Code Similarity Detection without Fine-tuning.](http://arxiv.org/abs/2211.01144) | 提出了一种新的二进制代码嵌入模型UniASM，并设计了两个新的训练任务，使得生成向量的空间分布更加均匀，直接可以在无需任何微调的情况下用于二进制代码相似性检测。此外，提出了一种新的二进制函数tokenization方法，缓解了词汇外的问题，并通过消融实验得到了一些新的有价值的发现，实验证明UniASM优于其他模型。 |

# 详细

[^1]: 通过反事实概念瓶颈模型攀登解释性的阶梯

    Climbing the Ladder of Interpretability with Counterfactual Concept Bottleneck Models

    [https://rss.arxiv.org/abs/2402.01408](https://rss.arxiv.org/abs/2402.01408)

    本论文提出了一种新的模型 CF-CBMs，可以同时解决深度学习模型的预测、解释和想象能力的不足，为部署可靠的AI代理、校准人类信任和加深人机交互提供了一种有效的解决方法。

    

    当前的深度学习模型没有同时解决三个基本问题的设计：预测类别标签以解决给定的分类任务（“是什么？”），解释任务预测（“为什么？”），并想象可能导致不同预测的替代情景（“如果怎样？”）。无法回答这些问题代表了部署可靠的AI代理、校准人类信任和加深人机交互的关键差距。为了弥合这一差距，我们引入了反事实概念瓶颈模型（CF-CBMs），这是一类能够高效同时解决上述查询而无需进行事后搜索的模型。我们的结果表明，CF-CBMs能够产生准确的预测（“是什么？”），对任务预测提供简单的解释（“为什么？”），以及可解释的反事实情况（“如果怎样？”）。CF-CBMs还可以对概念干预的影响进行采样或估计最可能的反事实情况，以解释事件，并优化产生多样化的反事实。

    Current deep learning models are not designed to simultaneously address three fundamental questions: predict class labels to solve a given classification task (the "What?"), explain task predictions (the "Why?"), and imagine alternative scenarios that could result in different predictions (the "What if?"). The inability to answer these questions represents a crucial gap in deploying reliable AI agents, calibrating human trust, and deepening human-machine interaction. To bridge this gap, we introduce CounterFactual Concept Bottleneck Models (CF-CBMs), a class of models designed to efficiently address the above queries all at once without the need to run post-hoc searches. Our results show that CF-CBMs produce: accurate predictions (the "What?"), simple explanations for task predictions (the "Why?"), and interpretable counterfactuals (the "What if?"). CF-CBMs can also sample or estimate the most probable counterfactual to: (i) explain the effect of concept interventions on tasks, (ii) sh
    
[^2]: 理解为何标签平滑会降低选择性分类的效果以及如何解决这个问题

    Understanding Why Label Smoothing Degrades Selective Classification and How to Fix It

    [https://arxiv.org/abs/2403.14715](https://arxiv.org/abs/2403.14715)

    LS方法在深度神经网络分类器训练中的标签平滑效果被发现会负面影响选择性分类，通过影响模型预测不确定性，此研究阐明了这一现象。

    

    标签平滑（LS）是一种流行的深度神经网络分类器训练的正则化方法，因为它在提高测试准确性方面效果显著，并且实现简单。"硬"的one-hot标签通过将概率质量均匀分配给其他类别来进行"平滑化"，从而减少过度拟合。在这项工作中，我们揭示了LS如何负面影响选择性分类（SC）- 其目标是利用模型的预测不确定性来拒绝错误分类。我们首先在一系列任务和架构中从经验上证明LS会导致SC的一致性降级。然后，我们通过分析logit级别的梯度来解释这一点，表明LS通过在错误概率低时更加正则化最大logit，而在错误概率高时更少正则化，加剧了过度自信和低自信。这阐明了以前报道的强分类器在SC中性能不佳的实验结果。

    arXiv:2403.14715v1 Announce Type: cross  Abstract: Label smoothing (LS) is a popular regularisation method for training deep neural network classifiers due to its effectiveness in improving test accuracy and its simplicity in implementation. "Hard" one-hot labels are "smoothed" by uniformly distributing probability mass to other classes, reducing overfitting. In this work, we reveal that LS negatively affects selective classification (SC) - where the aim is to reject misclassifications using a model's predictive uncertainty. We first demonstrate empirically across a range of tasks and architectures that LS leads to a consistent degradation in SC. We then explain this by analysing logit-level gradients, showing that LS exacerbates overconfidence and underconfidence by regularising the max logit more when the probability of error is low, and less when the probability of error is high. This elucidates previously reported experimental results where strong classifiers underperform in SC. We
    
[^3]: 将网络选举化：用于社区拉票的动态多步对抗性攻击

    Electioneering the Network: Dynamic Multi-Step Adversarial Attacks for Community Canvassing

    [https://arxiv.org/abs/2403.12399](https://arxiv.org/abs/2403.12399)

    本论文提出了用于社区拉票的动态多步对抗性攻击，使得对手能够利用基于梯度的攻击来策划目标选民的操纵。

    

    在今天的世界中，对于社区拉票的在线社交网络操纵问题是一个真正关注的问题。受选民模型、网络上的观点和极化动态的研究启发，我们将社区拉票建模为一个通过对GNN进行基于梯度的攻击而在网络上进行的动态过程。现有的GNN攻击都是单步的，没有考虑网络中信息传播的动态级联特性。我们考虑了一个现实的场景，即对手使用GNN作为代理来预测和操纵选民偏好，特别是不确定的选民。对GNN的基于梯度的攻击通知对手可以进行战略操纵，以使得目标选民入教。具体而言，我们探讨了$\textit{社区拉票的最小预算攻击}$（MBACC）。我们证明了MBACC问题是NP困难的，并提出了动态多步对抗性社区拉票（MAC）来解决这一问题。MAC m

    arXiv:2403.12399v1 Announce Type: new  Abstract: The problem of online social network manipulation for community canvassing is of real concern in today's world. Motivated by the study of voter models, opinion and polarization dynamics on networks, we model community canvassing as a dynamic process over a network enabled via gradient-based attacks on GNNs. Existing attacks on GNNs are all single-step and do not account for the dynamic cascading nature of information diffusion in networks. We consider the realistic scenario where an adversary uses a GNN as a proxy to predict and manipulate voter preferences, especially uncertain voters. Gradient-based attacks on the GNN inform the adversary of strategic manipulations that can be made to proselytize targeted voters. In particular, we explore $\textit{minimum budget attacks for community canvassing}$ (MBACC). We show that the MBACC problem is NP-Hard and propose Dynamic Multi-Step Adversarial Community Canvassing (MAC) to address it. MAC m
    
[^4]: 底物范围对比学习：重新利用人类偏见学习原子表示

    Substrate Scope Contrastive Learning: Repurposing Human Bias to Learn Atomic Representations

    [https://arxiv.org/abs/2402.16882](https://arxiv.org/abs/2402.16882)

    提出了一种新颖的预训练策略，底物范围对比学习，以学习适合化学反应性的原子表示。

    

    学习分子表示是分子机器学习中的关键步骤，对建模成功产生显著影响，尤其在数据稀缺情况下。广义预训练神经网络的概念推动了计算机视觉、自然语言处理和蛋白质工程等领域的发展。然而，类似的方法在小有机分子方面并未取得类似的成功。在这项工作中，我们引入一种新颖的预训练策略，即底物范围对比学习，它学习适合化学反应性的原子表示。这种方法以已发表的底物范围表中底物的分组和产物收率作为化学反应性相似性或不相似性的衡量。我们关注 CAS Content Collection 中的 20,798 个芳香卤代烃，涵盖数千篇出版物，以学习芳香卤代烃的反应性表示。我们验证了我们的预训练方法。

    arXiv:2402.16882v1 Announce Type: cross  Abstract: Learning molecular representation is a critical step in molecular machine learning that significantly influences modeling success, particularly in data-scarce situations. The concept of broadly pre-training neural networks has advanced fields such as computer vision, natural language processing, and protein engineering. However, similar approaches for small organic molecules have not achieved comparable success. In this work, we introduce a novel pre-training strategy, substrate scope contrastive learning, which learns atomic representations tailored to chemical reactivity. This method considers the grouping of substrates and their yields in published substrate scope tables as a measure of their similarity or dissimilarity in terms of chemical reactivity. We focus on 20,798 aryl halides in the CAS Content Collection spanning thousands of publications to learn a representation of aryl halide reactivity. We validate our pre-training appr
    
[^5]: 具有交叉一致$p$-值的异常检测中的不确定性量化

    Uncertainty Quantification in Anomaly Detection with Cross-Conformal $p$-Values

    [https://arxiv.org/abs/2402.16388](https://arxiv.org/abs/2402.16388)

    针对异常检测系统中不确定性量化的需求，提出了一种新颖的框架，称为交叉一致异常检测，通过校准模型的不确定性提供统计保证。

    

    随着可靠、可信和可解释机器学习的重要性日益增加，对异常检测系统进行不确定性量化的要求变得愈发重要。在这种情况下，有效控制类型I错误率($\alpha$)而又不损害系统的统计功率($1-\beta$)可以建立信任，并减少与假发现相关的成本，特别是当后续程序昂贵时。利用符合预测原则的方法有望通过校准模型的不确定性为异常检测提供相应的统计保证。该工作引入了一个新颖的异常检测框架，称为交叉一致异常检测，建立在为预测任务设计的著名交叉一致方法之上。通过这种方法，他填补了在归纳一致异常检测环境中扩展先前研究的自然研究空白

    arXiv:2402.16388v1 Announce Type: cross  Abstract: Given the growing significance of reliable, trustworthy, and explainable machine learning, the requirement of uncertainty quantification for anomaly detection systems has become increasingly important. In this context, effectively controlling Type I error rates ($\alpha$) without compromising the statistical power ($1-\beta$) of these systems can build trust and reduce costs related to false discoveries, particularly when follow-up procedures are expensive. Leveraging the principles of conformal prediction emerges as a promising approach for providing respective statistical guarantees by calibrating a model's uncertainty. This work introduces a novel framework for anomaly detection, termed cross-conformal anomaly detection, building upon well-known cross-conformal methods designed for prediction tasks. With that, it addresses a natural research gap by extending previous works in the context of inductive conformal anomaly detection, rel
    
[^6]: 非可分的目标函数的DP-SGD方法

    DP-SGD for non-decomposable objective functions. (arXiv:2310.03104v1 [cs.LG])

    [http://arxiv.org/abs/2310.03104](http://arxiv.org/abs/2310.03104)

    本论文提出了一种针对非可分的目标函数的DP-SGD方法，解决了使用差分隐私进行训练时，相似性损失函数的$L_2$敏感度增长随着批量大小增加的问题。

    

    无监督预训练是开发计算机视觉模型和大型语言模型的常见步骤。在这种情况下，由于缺少标签，需要使用基于相似性的损失函数，如对比损失，来优化相似输入之间的距离并最大化不同输入之间的距离。随着隐私问题的增多，使用差分隐私来训练这些模型变得更加重要。然而，由于这些损失函数生成输入的方式，它们的$L_2$敏感度会随着批量大小的增加而增加，这对于差分隐私训练方法（如DP-SGD）特别不利。为了解决这个问题，我们开发了一种新的DP-SGD变体，用于基于相似性的损失函数，特别是常用的对比损失，通过一种新颖的方式处理目标函数的梯度，使得梯度的敏感度对于批量大小是$O(1)$。

    Unsupervised pre-training is a common step in developing computer vision models and large language models. In this setting, the absence of labels requires the use of similarity-based loss functions, such as contrastive loss, that favor minimizing the distance between similar inputs and maximizing the distance between distinct inputs. As privacy concerns mount, training these models using differential privacy has become more important. However, due to how inputs are generated for these losses, one of their undesirable properties is that their $L_2$ sensitivity can grow with increasing batch size. This property is particularly disadvantageous for differentially private training methods, such as DP-SGD. To overcome this issue, we develop a new DP-SGD variant for similarity based loss functions -- in particular the commonly used contrastive loss -- that manipulates gradients of the objective function in a novel way to obtain a senstivity of the summed gradient that is $O(1)$ for batch size
    
[^7]: 关于扩散模型记忆化的研究

    On Memorization in Diffusion Models. (arXiv:2310.02664v1 [cs.LG])

    [http://arxiv.org/abs/2310.02664](http://arxiv.org/abs/2310.02664)

    本论文研究了扩散模型的记忆化行为，发现记忆化倾向于在较小的数据集上发生。通过定义有效模型记忆化 (EMM) 这一指标，量化了数据分布和模型配置对记忆化行为的影响。

    

    近年来，由于其生成新颖高质量样本的能力，扩散模型引起了广泛的研究兴趣。然而，通过典型的训练目标，即去噪得分匹配，扩散模型只能生成复制训练数据的样本，这表明在理论上会出现记忆化的行为，这与现有先进扩散模型的普遍泛化能力相矛盾，因此需要深入理解。我们观察到记忆化行为倾向于在较小的数据集上发生，我们提出了有效模型记忆化(EMM)的定义，这是一种衡量学习的扩散模型在最大数据集上近似其理论最优点的度量标准。然后，我们量化了影响这些记忆化行为的重要因素，重点关注数据分布和模型配置。

    Due to their capacity to generate novel and high-quality samples, diffusion models have attracted significant research interest in recent years. Notably, the typical training objective of diffusion models, i.e., denoising score matching, has a closed-form optimal solution that can only generate training data replicating samples. This indicates that a memorization behavior is theoretically expected, which contradicts the common generalization ability of state-of-the-art diffusion models, and thus calls for a deeper understanding. Looking into this, we first observe that memorization behaviors tend to occur on smaller-sized datasets, which motivates our definition of effective model memorization (EMM), a metric measuring the maximum size of training data at which a learned diffusion model approximates its theoretical optimum. Then, we quantify the impact of the influential factors on these memorization behaviors in terms of EMM, focusing primarily on data distribution, model configuratio
    
[^8]: SEA：可共享和可解释的基于查询的黑盒攻击归因

    SEA: Shareable and Explainable Attribution for Query-based Black-box Attacks. (arXiv:2308.11845v1 [cs.LG])

    [http://arxiv.org/abs/2308.11845](http://arxiv.org/abs/2308.11845)

    本论文提出了SEA，一种用于归因基于查询的黑盒攻击的机器学习安全系统，通过利用隐藏马尔可夫模型框架来理解攻击的演变过程，并有效归因攻击，即使是对于第二次出现的攻击，具有鲁棒性，旨在实现取证和人类可解释的情报共享。

    

    机器学习系统容易受到来自基于查询的黑盒攻击的敌对样本的攻击。尽管有各种努力来检测和防止这些攻击，但仍然需要一种更全面的方法来记录、分析和分享攻击证据。虽然经典安全领域受益于成熟的取证和情报共享技术，但机器学习领域尚未找到一种方式来对攻击者进行画像，并分享关于他们的信息。为此，本论文引入了SEA，一种新颖的机器学习安全系统，用于为取证目的表征对机器学习系统的黑盒攻击，并促进可解释的情报共享。SEA利用隐藏马尔可夫模型框架将观察到的查询序列归因于已知的攻击，因此它能够理解攻击的演变过程而不仅仅关注最终的敌对样本。我们的评估结果显示，SEA能够有效进行攻击归因，即使是对于第二次出现的攻击，也具有鲁棒性。

    Machine Learning (ML) systems are vulnerable to adversarial examples, particularly those from query-based black-box attacks. Despite various efforts to detect and prevent such attacks, there is a need for a more comprehensive approach to logging, analyzing, and sharing evidence of attacks. While classic security benefits from well-established forensics and intelligence sharing, Machine Learning is yet to find a way to profile its attackers and share information about them. In response, this paper introduces SEA, a novel ML security system to characterize black-box attacks on ML systems for forensic purposes and to facilitate human-explainable intelligence sharing. SEA leverages the Hidden Markov Models framework to attribute the observed query sequence to known attacks. It thus understands the attack's progression rather than just focusing on the final adversarial examples. Our evaluations reveal that SEA is effective at attack attribution, even on their second occurrence, and is robus
    
[^9]: 逆强化学习中的有效时间视野研究

    On the Effective Horizon of Inverse Reinforcement Learning. (arXiv:2307.06541v1 [cs.LG])

    [http://arxiv.org/abs/2307.06541](http://arxiv.org/abs/2307.06541)

    本研究分析了逆强化学习中时间视野的重要性，发现短于实际值的有效时间视野可以更快且更准确地估计奖励函数，减轻过拟合问题。此外，研究还呼吁在IRL中同时学习奖励和有效时间视野。

    

    逆强化学习（IRL）算法通常依赖于基于给定时间视野的（前向）强化学习或规划来计算一个近似最优策略，然后将该策略与专家演示匹配。时间视野在确定奖励估计的准确性和IRL算法的计算效率方面起着关键作用。有趣的是，比地面实际值更短的有效时间视野通常能更快地产生更好的结果。本文对此现象进行了正式分析并给出了解释：时间视野控制了引发策略类的复杂性，并在有限数据下减轻过拟合。这一分析为IRL的有效视野选择提供了原则性指导。它也促使我们重新审视经典的IRL公式：与仅具有给定视野的奖励相比，共同学习奖励和有效视野更加自然。我们的实验进一步验证了这一观点。

    Inverse reinforcement learning (IRL) algorithms often rely on (forward) reinforcement learning or planning over a given time horizon to compute an approximately optimal policy for a hypothesized reward function and then match this policy with expert demonstrations. The time horizon plays a critical role in determining both the accuracy of reward estimate and the computational efficiency of IRL algorithms. Interestingly, an effective time horizon shorter than the ground-truth value often produces better results faster. This work formally analyzes this phenomenon and provides an explanation: the time horizon controls the complexity of an induced policy class and mitigates overfitting with limited data. This analysis leads to a principled choice of the effective horizon for IRL. It also prompts us to reexamine the classic IRL formulation: it is more natural to learn jointly the reward and the effective horizon together rather than the reward alone with a given horizon. Our experimental re
    
[^10]: 使用数据自适应能量距离的高维数据稳健分类

    Robust Classification of High-Dimensional Data using Data-Adaptive Energy Distance. (arXiv:2306.13985v1 [stat.ML])

    [http://arxiv.org/abs/2306.13985](http://arxiv.org/abs/2306.13985)

    该论文提出了一种用于高维低样本量数据分类的稳健的数据自适应能量距离分类器，该分类器无需调参且在一定条件下可以实现完美分类，已在模拟研究和实际数据分析中得到证明比其他方法表现更优。

    

    在真实世界中，高维低样本量（HDLSS）数据的分类面临挑战，例如基因表达研究、癌症研究和医学成像等领域。本文提出了一些专门为HDLSS数据设计的分类器的开发和分析。这些分类器没有调节参数，并且是稳健的，因为它们不受底层数据分布的任何矩条件的影响。研究表明，在一些相当普遍的条件下，它们在HDLSS渐近区域内可以实现完美分类。还比较了所提出分类器的性能。我们的理论结果得到了广泛的模拟研究和实际数据分析的支持，证明了所提出分类技术优于几种广泛认可的方法的有希望优势。

    Classification of high-dimensional low sample size (HDLSS) data poses a challenge in a variety of real-world situations, such as gene expression studies, cancer research, and medical imaging. This article presents the development and analysis of some classifiers that are specifically designed for HDLSS data. These classifiers are free of tuning parameters and are robust, in the sense that they are devoid of any moment conditions of the underlying data distributions. It is shown that they yield perfect classification in the HDLSS asymptotic regime, under some fairly general conditions. The comparative performance of the proposed classifiers is also investigated. Our theoretical results are supported by extensive simulation studies and real data analysis, which demonstrate promising advantages of the proposed classification techniques over several widely recognized methods.
    
[^11]: UniASM：无需微调的二进制代码相似性检测

    UniASM: Binary Code Similarity Detection without Fine-tuning. (arXiv:2211.01144v3 [cs.CR] UPDATED)

    [http://arxiv.org/abs/2211.01144](http://arxiv.org/abs/2211.01144)

    提出了一种新的二进制代码嵌入模型UniASM，并设计了两个新的训练任务，使得生成向量的空间分布更加均匀，直接可以在无需任何微调的情况下用于二进制代码相似性检测。此外，提出了一种新的二进制函数tokenization方法，缓解了词汇外的问题，并通过消融实验得到了一些新的有价值的发现，实验证明UniASM优于其他模型。

    

    二进制代码相似性检测被广泛用于各种二进制分析任务，如漏洞搜索、恶意软件检测、克隆检测和补丁分析。最近的研究表明，基于学习的二进制代码嵌入模型比传统的基于特征的方法更好。本文提出了一种新的基于transformer的二进制代码嵌入模型UniASM，用于学习二进制函数的表示。我们设计了两个新的训练任务，使得生成向量的空间分布更加均匀，直接可以在无需任何微调的情况下用于二进制代码相似性检测。此外，我们提出了一种新的二进制函数tokenization方法，增加了tokens的语义信息并缓解了词汇外的问题。通过消融实验进行了深入分析，得到了一些新的有价值的发现，实验证明UniASM优于其他模型。

    Binary code similarity detection (BCSD) is widely used in various binary analysis tasks such as vulnerability search, malware detection, clone detection, and patch analysis. Recent studies have shown that the learning-based binary code embedding models perform better than the traditional feature-based approaches. In this paper, we propose a novel transformer-based binary code embedding model named UniASM to learn representations of the binary functions. We design two new training tasks to make the spatial distribution of the generated vectors more uniform, which can be used directly in BCSD without any fine-tuning. In addition, we present a new tokenization approach for binary functions, which increases the token's semantic information and mitigates the out-of-vocabulary (OOV) problem. We conduct an in-depth analysis of the factors affecting model performance through ablation experiments and obtain some new and valuable findings. The experimental results show that UniASM outperforms th
    

