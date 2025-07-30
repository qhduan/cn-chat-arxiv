# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [The pitfalls of next-token prediction](https://arxiv.org/abs/2403.06963) | 论文揭示了在某些任务类别中，教师强制方法可能无法在第一时间学习到准确的下一个标记预测器，进而导致模型失败的一般机制。 |
| [^2] | [Adversarial Attacks and Defenses in Explainable Artificial Intelligence: A Survey.](http://arxiv.org/abs/2306.06123) | 本文总结了对抗性攻击和防御在可解释人工智能中的研究。列出了现有的不安全因素，并表明了本领域的新兴研究方向。 |
| [^3] | [Global-QSGD: Practical Floatless Quantization for Distributed Learning with Theoretical Guarantees.](http://arxiv.org/abs/2305.18627) | Global-QSGD是一种新颖的全局缩放量化机制，可以提高分布式学习的效率，并且不需要昂贵的误差反馈，并提供了高达$O(\ sqrt{n})$的额外压缩比。 |
| [^4] | [Demystifying Misconceptions in Social Bots Research.](http://arxiv.org/abs/2303.17251) | 这篇文章揭示了关于社交机器人研究的普遍误解，强调需要以严谨、公正和负责任的方式讨论虚假信息研究。 |

# 详细

[^1]: 下一个标记预测的陷阱

    The pitfalls of next-token prediction

    [https://arxiv.org/abs/2403.06963](https://arxiv.org/abs/2403.06963)

    论文揭示了在某些任务类别中，教师强制方法可能无法在第一时间学习到准确的下一个标记预测器，进而导致模型失败的一般机制。

    

    一篇关于下一个标记预测的论文。我们提出了一个直观的担忧：一个仅仅基于下一个标记预测的模型是否能忠实地模拟人类智能。我们认为下一个标记预测中经常混淆的两个阶段 -- 自回归推断和教师强制训练 -- 必须被区别对待。我们描述了一个一般机制，展示了教师强制如何失败，并设计了一个最小化计划任务，在这个任务中Transformer和Mamba架构在实践中以这种方式失败 -- 尽管任务本身很容易学习。

    arXiv:2403.06963v1 Announce Type: cross  Abstract: Can a mere next-token predictor faithfully model human intelligence? We crystallize this intuitive concern, which is fragmented in the literature. As a starting point, we argue that the two often-conflated phases of next-token prediction -- autoregressive inference and teacher-forced training -- must be treated distinctly. The popular criticism that errors can compound during autoregressive inference, crucially assumes that teacher-forcing has learned an accurate next-token predictor. This assumption sidesteps a more deep-rooted problem we expose: in certain classes of tasks, teacher-forcing can simply fail to learn an accurate next-token predictor in the first place. We describe a general mechanism of how teacher-forcing can fail, and design a minimal planning task where both the Transformer and the Mamba architecture empirically fail in that manner -- remarkably, despite the task being straightforward to learn. We provide preliminary
    
[^2]: 《可解释人工智能中的对抗性攻击和防御：调查报告》

    Adversarial Attacks and Defenses in Explainable Artificial Intelligence: A Survey. (arXiv:2306.06123v1 [cs.CR])

    [http://arxiv.org/abs/2306.06123](http://arxiv.org/abs/2306.06123)

    本文总结了对抗性攻击和防御在可解释人工智能中的研究。列出了现有的不安全因素，并表明了本领域的新兴研究方向。

    

    可解释人工智能（XAI）方法被描绘为调试和信任统计和深度学习模型的治疗方式，以及解释它们的预测。然而，对抗机器学习的最新进展突出了最新解释的局限性和漏洞，这些进展令人对其安全性和可信度产生质疑。操纵、欺骗或洗白模型推理证据的可能性在高风险决策和知识发现中产生不利后果。本文总结了50多篇论文的研究，概述了针对机器学习模型解释的对抗攻击以及公平度量的研究。我们讨论了如何防御攻击并设计鲁棒的解释方法。我们列出XAI中现有的不安全因素，并概述了对抗性XAI（AdvXAI）的新兴研究方向。

    Explainable artificial intelligence (XAI) methods are portrayed as a remedy for debugging and trusting statistical and deep learning models, as well as interpreting their predictions. However, recent advances in adversarial machine learning highlight the limitations and vulnerabilities of state-of-the-art explanations, putting their security and trustworthiness into question. The possibility of manipulating, fooling or fairwashing evidence of the model's reasoning has detrimental consequences when applied in high-stakes decision-making and knowledge discovery. This concise survey of over 50 papers summarizes research concerning adversarial attacks on explanations of machine learning models, as well as fairness metrics. We discuss how to defend against attacks and design robust interpretation methods. We contribute a list of existing insecurities in XAI and outline the emerging research directions in adversarial XAI (AdvXAI).
    
[^3]: 全局缩放量化：具有理论保证的分布式学习实用的无浮点量化

    Global-QSGD: Practical Floatless Quantization for Distributed Learning with Theoretical Guarantees. (arXiv:2305.18627v1 [cs.LG])

    [http://arxiv.org/abs/2305.18627](http://arxiv.org/abs/2305.18627)

    Global-QSGD是一种新颖的全局缩放量化机制，可以提高分布式学习的效率，并且不需要昂贵的误差反馈，并提供了高达$O(\ sqrt{n})$的额外压缩比。

    

    高效的分布式训练是推动深度学习近期进展的主要驱动力。然而，通信常常是系统的主要瓶颈并具有高昂的代价。因此，需要设计高效的通信机制，既能在经验上提高吞吐量，又能提供理论保证。在这项工作中，我们介绍了全局-QSGD，一种新颖的量化运算符，通过全局缩放设计来加速基于分布式学习。我们证明Global-QSGD是第一个理论上严格的Allreduce兼容压缩机制，通过在压缩误差和通信节省之间取得平衡来实现可证明的加速。重要的是，由于其固有的无偏性，Global-QSGD不依赖昂贵的误差反馈，并且相对于流行的QSGD量化能提供高达$O(\sqrt{n})$ 的额外压缩比（其中$n$表示工作者的数量）。为了获得理论保证，我们采用了信息论和凸分析技术。

    Efficient distributed training is a principal driver of recent advances in deep learning. However, communication often proves costly and becomes the primary bottleneck in these systems. As a result, there is a demand for the design of efficient communication mechanisms that can empirically boost throughput while providing theoretical guarantees. In this work, we introduce Global-QSGD, a novel family of quantization operators, engineered to accelerate distributed training based on global scaling. We demonstrate that Global-QSGD is the first theoretically rigorous Allreduce-compatible compression mechanism that achieves a provable speed-up by striking a balance between compression error and communication savings. Importantly, Global-QSGD does not rely on costly error feedback due to its inherent unbiasedness and offers up to $O(\sqrt{n})$ additional compression ratio compared to the popular QSGD quantization ($n$ represents the number of workers). To obtain theoretical guarantees, we gen
    
[^4]: 揭开对社交机器人研究的误解

    Demystifying Misconceptions in Social Bots Research. (arXiv:2303.17251v1 [cs.SI])

    [http://arxiv.org/abs/2303.17251](http://arxiv.org/abs/2303.17251)

    这篇文章揭示了关于社交机器人研究的普遍误解，强调需要以严谨、公正和负责任的方式讨论虚假信息研究。

    

    社交机器人科学寻求解决网络虚假信息最受争议的形式之一的知识和解决方案。然而，社交机器人研究受到普遍的偏见、夸大的结果和误解的困扰，这些都为歧义、不切实际的期望和看似无法调和的发现打下了基础。克服这些问题对于确保可靠的解决方案和重申科学方法的有效性至关重要。在这篇文章中，我们修订了社交机器人研究中的一些最新结果，强调和纠正了事实错误以及方法论和概念问题。更重要的是，我们揭开了普遍的误解，解决了有关如何讨论社交机器人研究的基本问题。我们的分析揭示了以严谨、公正和负责任的方式讨论虚假信息研究的必要性。本文通过确定并驳斥社交机器人研究的支持者和反对者常用的谬误论证，支持这种努力。

    The science of social bots seeks knowledge and solutions to one of the most debated forms of online misinformation. Yet, social bots research is plagued by widespread biases, hyped results, and misconceptions that set the stage for ambiguities, unrealistic expectations, and seemingly irreconcilable findings. Overcoming such issues is instrumental towards ensuring reliable solutions and reaffirming the validity of the scientific method. In this contribution we revise some recent results in social bots research, highlighting and correcting factual errors as well as methodological and conceptual issues. More importantly, we demystify common misconceptions, addressing fundamental points on how social bots research is discussed. Our analysis surfaces the need to discuss misinformation research in a rigorous, unbiased, and responsible way. This article bolsters such effort by identifying and refuting common fallacious arguments used by both proponents and opponents of social bots research as
    

