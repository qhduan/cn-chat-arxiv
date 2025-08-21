# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Enhancing Depression-Diagnosis-Oriented Chat with Psychological State Tracking](https://arxiv.org/abs/2403.09717) | 本文提出将心理状态跟踪集成到大型语言模型中，以明确引导抑郁症诊断导向的聊天，从而能更好地捕捉患者在对话过程中的信息、情绪或症状变化。 |
| [^2] | [Energy-efficiency Limits on Training AI Systems using Learning-in-Memory](https://arxiv.org/abs/2402.14878) | 该论文提出了使用内存中学习的方法训练AI系统时的能效限制，并推导了新的理论下限。 |
| [^3] | [Don't Push the Button! Exploring Data Leakage Risks in Machine Learning and Transfer Learning.](http://arxiv.org/abs/2401.13796) | 本文讨论了机器学习中的数据泄露问题，即未预期的信息污染训练数据，影响模型性能评估，用户可能由于缺乏理解而忽视关键步骤，导致乐观的性能估计在实际场景中不成立。 |

# 详细

[^1]: 通过心理状态跟踪提升面向抑郁症诊断的聊天系统

    Enhancing Depression-Diagnosis-Oriented Chat with Psychological State Tracking

    [https://arxiv.org/abs/2403.09717](https://arxiv.org/abs/2403.09717)

    本文提出将心理状态跟踪集成到大型语言模型中，以明确引导抑郁症诊断导向的聊天，从而能更好地捕捉患者在对话过程中的信息、情绪或症状变化。

    

    抑郁症诊断导向的聊天旨在引导患者进行自我表达，收集抑郁症检测的关键症状。最近的工作集中于结合任务导向对话和闲聊，模拟基于访谈的抑郁症诊断。然而，这些方法无法很好地捕捉患者在对话过程中的信息、情绪或症状的变化。此外，还没有明确的框架用于引导对话，这导致一些无用的交流影响了体验。本文提出将心理状态跟踪（POST）集成到大型语言模型（LLM）中，以明确引导抑郁症诊断导向的聊天。具体而言，该状态源自于心理理论模型，包括四个组件，即阶段、信息、总结和下一步。我们微调了一个LLM模型以生成动态心理状态，进而用于辅助r

    arXiv:2403.09717v1 Announce Type: cross  Abstract: Depression-diagnosis-oriented chat aims to guide patients in self-expression to collect key symptoms for depression detection. Recent work focuses on combining task-oriented dialogue and chitchat to simulate the interview-based depression diagnosis. Whereas, these methods can not well capture the changing information, feelings, or symptoms of the patient during dialogues. Moreover, no explicit framework has been explored to guide the dialogue, which results in some useless communications that affect the experience. In this paper, we propose to integrate Psychological State Tracking (POST) within the large language model (LLM) to explicitly guide depression-diagnosis-oriented chat. Specifically, the state is adapted from a psychological theoretical model, which consists of four components, namely Stage, Information, Summary and Next. We fine-tune an LLM model to generate the dynamic psychological state, which is further used to assist r
    
[^2]: 使用内存中学习的方法训练AI系统的能效限制

    Energy-efficiency Limits on Training AI Systems using Learning-in-Memory

    [https://arxiv.org/abs/2402.14878](https://arxiv.org/abs/2402.14878)

    该论文提出了使用内存中学习的方法训练AI系统时的能效限制，并推导了新的理论下限。

    

    arXiv:2402.14878v1 公告类型: cross 摘要: 内存中学习（LIM）是一种最近提出的范Paradigm，旨在克服训练机器学习系统中的基本内存瓶颈。虽然计算于内存（CIM）方法可以解决所谓的内存墙问题（即由于重复内存读取访问而消耗的能量），但它们对于以训练所需的精度重复内存写入时消耗的能量（更新墙）是不可知的，并且它们不考虑在短期和长期记忆之间传输信息时所消耗的能量（整合墙）。LIM范式提出，如果物理内存的能量屏障被自适应调制，使得存储器更新和整合的动态与梯度下降训练AI模型的Lyapunov动态相匹配，那么这些瓶颈也可以被克服。在本文中，我们推导了使用不同LIM应用程序训练AI系统时的能耗的新理论下限。

    arXiv:2402.14878v1 Announce Type: cross  Abstract: Learning-in-memory (LIM) is a recently proposed paradigm to overcome fundamental memory bottlenecks in training machine learning systems. While compute-in-memory (CIM) approaches can address the so-called memory-wall (i.e. energy dissipated due to repeated memory read access) they are agnostic to the energy dissipated due to repeated memory writes at the precision required for training (the update-wall), and they don't account for the energy dissipated when transferring information between short-term and long-term memories (the consolidation-wall). The LIM paradigm proposes that these bottlenecks, too, can be overcome if the energy barrier of physical memories is adaptively modulated such that the dynamics of memory updates and consolidation match the Lyapunov dynamics of gradient-descent training of an AI model. In this paper, we derive new theoretical lower bounds on energy dissipation when training AI systems using different LIM app
    
[^3]: 不要按按钮！探索机器学习和迁移学习中的数据泄露风险

    Don't Push the Button! Exploring Data Leakage Risks in Machine Learning and Transfer Learning. (arXiv:2401.13796v1 [cs.LG])

    [http://arxiv.org/abs/2401.13796](http://arxiv.org/abs/2401.13796)

    本文讨论了机器学习中的数据泄露问题，即未预期的信息污染训练数据，影响模型性能评估，用户可能由于缺乏理解而忽视关键步骤，导致乐观的性能估计在实际场景中不成立。

    

    机器学习（ML）在各个领域取得了革命性的进展，为多个领域提供了预测能力。然而，随着ML工具的日益可获得性，许多从业者缺乏深入的ML专业知识，采用了“按按钮”方法，利用用户友好的界面而忽视了底层算法的深入理解。虽然这种方法提供了便利，但它引发了对结果可靠性的担忧，导致了错误的性能评估等挑战。本文解决了ML中的一个关键问题，即数据泄露，其中未预期的信息污染了训练数据，影响了模型的性能评估。由于缺乏理解，用户可能会无意中忽视关键步骤，从而导致在现实场景中可能不成立的乐观性能估计。评估性能与实际在新数据上的性能的差异是一个重要的关注点。本文特别将ML中的数据泄露分为不同类别，并讨论了相关解决方法。

    Machine Learning (ML) has revolutionized various domains, offering predictive capabilities in several areas. However, with the increasing accessibility of ML tools, many practitioners, lacking deep ML expertise, adopt a "push the button" approach, utilizing user-friendly interfaces without a thorough understanding of underlying algorithms. While this approach provides convenience, it raises concerns about the reliability of outcomes, leading to challenges such as incorrect performance evaluation. This paper addresses a critical issue in ML, known as data leakage, where unintended information contaminates the training data, impacting model performance evaluation. Users, due to a lack of understanding, may inadvertently overlook crucial steps, leading to optimistic performance estimates that may not hold in real-world scenarios. The discrepancy between evaluated and actual performance on new data is a significant concern. In particular, this paper categorizes data leakage in ML, discussi
    

