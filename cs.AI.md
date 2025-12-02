# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Recurrent networks recognize patterns with low-dimensional oscillations.](http://arxiv.org/abs/2310.07908) | 本研究提出了一种通过相位变化识别模式的循环神经网络机制，并通过验证手工制作的振荡模型证实了这一解释。该研究不仅提供了一种潜在的动力学机制用于模式识别，还暗示了有限状态自动机的神经实现方式，并且对深度学习模型的可解释性进行了贡献。 |
| [^2] | [Combining Survival Analysis and Machine Learning for Mass Cancer Risk Prediction using EHR data.](http://arxiv.org/abs/2309.15039) | 该论文介绍了一种利用 EHR 数据进行大规模肿瘤风险预测的新方法，其创新之处在于只需利用历史的医疗服务代码和诊断信息来实现最小化的数据需求，通过将存活分析和机器学习相结合，可以在大规模应用中实现对患者癌症风险的个性化评估。 |
| [^3] | [RPLKG: Robust Prompt Learning with Knowledge Graph.](http://arxiv.org/abs/2304.10805) | 本研究提出了一种基于知识图谱的鲁棒提示学习方法，通过自动设计有意义和可解释的提示集，提高小样本学习的泛化性能。 |

# 详细

[^1]: 循环网络通过低维振荡识别模式

    Recurrent networks recognize patterns with low-dimensional oscillations. (arXiv:2310.07908v1 [q-bio.NC])

    [http://arxiv.org/abs/2310.07908](http://arxiv.org/abs/2310.07908)

    本研究提出了一种通过相位变化识别模式的循环神经网络机制，并通过验证手工制作的振荡模型证实了这一解释。该研究不仅提供了一种潜在的动力学机制用于模式识别，还暗示了有限状态自动机的神经实现方式，并且对深度学习模型的可解释性进行了贡献。

    

    本研究提出了一种通过解释在SET卡牌游戏启发下进行训练的循环神经网络(RNN)在简单任务上的动力学机制来识别模式。我们将训练后的RNN解释为通过低维极限环中的相位变化进行模式识别，类似于有限状态自动机(FSA)中的转换。我们进一步通过手工制作一个简单的振荡模型来验证了这一解释，该模型复制了训练后的RNN的动力学特性。我们的发现不仅暗示了一种潜在的动力学机制能够实现模式识别，还暗示了一种有限状态自动机的潜在神经实现。最重要的是，这项工作有助于关于深度学习模型可解释性的讨论。

    This study proposes a novel dynamical mechanism for pattern recognition discovered by interpreting a recurrent neural network (RNN) trained on a simple task inspired by the SET card game. We interpreted the trained RNN as recognizing patterns via phase shifts in a low-dimensional limit cycle in a manner analogous to transitions in a finite state automaton (FSA). We further validated this interpretation by handcrafting a simple oscillatory model that reproduces the dynamics of the trained RNN. Our findings not only suggest of a potential dynamical mechanism capable of pattern recognition, but also suggest of a potential neural implementation of FSA. Above all, this work contributes to the growing discourse on deep learning model interpretability.
    
[^2]: 结合存活分析和机器学习利用电子健康记录数据进行肿瘤风险预测

    Combining Survival Analysis and Machine Learning for Mass Cancer Risk Prediction using EHR data. (arXiv:2309.15039v1 [cs.LG])

    [http://arxiv.org/abs/2309.15039](http://arxiv.org/abs/2309.15039)

    该论文介绍了一种利用 EHR 数据进行大规模肿瘤风险预测的新方法，其创新之处在于只需利用历史的医疗服务代码和诊断信息来实现最小化的数据需求，通过将存活分析和机器学习相结合，可以在大规模应用中实现对患者癌症风险的个性化评估。

    

    纯粹的医学肿瘤筛查方法通常费用高昂、耗时长，并且仅适用于大规模应用。先进的人工智能（AI）方法在癌症检测方面发挥了巨大作用，但需要特定或深入的医学数据。这些方面影响了癌症筛查方法的大规模实施。因此，基于已有的电子健康记录（EHR）数据对患者进行大规模个性化癌症风险评估应用AI方法是一种颠覆性的改变。本文提出了一种利用EHR数据进行大规模肿瘤风险预测的新方法。与其他方法相比，我们的方法通过最小的数据贪婪策略脱颖而出，仅需要来自EHR的医疗服务代码和诊断历史。我们将问题形式化为二分类问题。该数据集包含了175441名不记名的患者（其中2861名被诊断为癌症）。作为基准，我们实现了一个基于循环神经网络（RNN）的解决方案。我们提出了一种方法，将存活分析和机器学习相结合，

    Purely medical cancer screening methods are often costly, time-consuming, and weakly applicable on a large scale. Advanced Artificial Intelligence (AI) methods greatly help cancer detection but require specific or deep medical data. These aspects affect the mass implementation of cancer screening methods. For these reasons, it is a disruptive change for healthcare to apply AI methods for mass personalized assessment of the cancer risk among patients based on the existing Electronic Health Records (EHR) volume.  This paper presents a novel method for mass cancer risk prediction using EHR data. Among other methods, our one stands out by the minimum data greedy policy, requiring only a history of medical service codes and diagnoses from EHR. We formulate the problem as a binary classification. This dataset contains 175 441 de-identified patients (2 861 diagnosed with cancer). As a baseline, we implement a solution based on a recurrent neural network (RNN). We propose a method that combine
    
[^3]: RPLKG: 基于知识图谱的鲁棒提示学习

    RPLKG: Robust Prompt Learning with Knowledge Graph. (arXiv:2304.10805v1 [cs.AI])

    [http://arxiv.org/abs/2304.10805](http://arxiv.org/abs/2304.10805)

    本研究提出了一种基于知识图谱的鲁棒提示学习方法，通过自动设计有意义和可解释的提示集，提高小样本学习的泛化性能。

    

    大规模预训练模型已经被证明是可迁移的，并且对未知数据集具有很好的泛化性能。最近，诸如CLIP之类的多模态预训练模型在各种实验中表现出显着的性能提升。然而，当标记数据集有限时，新数据集或领域的泛化仍然具有挑战性。为了提高小样本学习的泛化性能，已经进行了各种努力，如提示学习和适配器。然而，当前的少样本自适应方法不具备可解释性，并且需要高计算成本来进行自适应。在本研究中，我们提出了一种新的方法，即基于知识图谱的鲁棒提示学习（RPLKG）。基于知识图谱，我们自动设计出各种可解释和有意义的提示集。我们的模型在大型预训练模型的一次正向传递后获得提示集的缓存嵌入。之后，模型使用GumbelSoftmax优化提示选择过程。

    Large-scale pre-trained models have been known that they are transferable, and they generalize well on the unseen dataset. Recently, multimodal pre-trained models such as CLIP show significant performance improvement in diverse experiments. However, when the labeled dataset is limited, the generalization of a new dataset or domain is still challenging. To improve the generalization performance on few-shot learning, there have been diverse efforts, such as prompt learning and adapter. However, the current few-shot adaptation methods are not interpretable, and they require a high computation cost for adaptation. In this study, we propose a new method, robust prompt learning with knowledge graph (RPLKG). Based on the knowledge graph, we automatically design diverse interpretable and meaningful prompt sets. Our model obtains cached embeddings of prompt sets after one forwarding from a large pre-trained model. After that, model optimizes the prompt selection processes with GumbelSoftmax. In
    

