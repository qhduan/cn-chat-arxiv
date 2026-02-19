# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Evaluating Language Model Agency through Negotiations.](http://arxiv.org/abs/2401.04536) | 本研究通过谈判游戏的视角，提出共同评估语言模型（LM）的性能和对齐，以更好地反映真实世界的部署条件，并避免数据泄漏。通过评估多轮次和跨模型交互，我们发现了LM的自我对弈和交叉对弈性能。 |
| [^2] | [Monaural Multi-Speaker Speech Separation Using Efficient Transformer Model.](http://arxiv.org/abs/2308.00010) | 该论文介绍了一种基于Transformer模型的单声道多人演讲分离方法，旨在减少计算复杂性，并在准确性和性能之间取得平衡。 |

# 详细

[^1]: 通过谈判评估语言模型的代理能力

    Evaluating Language Model Agency through Negotiations. (arXiv:2401.04536v1 [cs.CL])

    [http://arxiv.org/abs/2401.04536](http://arxiv.org/abs/2401.04536)

    本研究通过谈判游戏的视角，提出共同评估语言模型（LM）的性能和对齐，以更好地反映真实世界的部署条件，并避免数据泄漏。通过评估多轮次和跨模型交互，我们发现了LM的自我对弈和交叉对弈性能。

    

    公司、组织和政府越来越多地利用语言模型（LM）展示类似代理行为的出色能力。随着LM被采用来执行越来越具有自主性的任务，迫切需要可靠且可扩展的评估基准。当前主要是静态的LM基准无法很好地评估此类动态应用。因此，我们提议通过谈判游戏的视角来共同评估LM的性能和对齐。我们认为这个共同任务更好地反映了真实世界的部署条件，并提供了关于LM决策过程的见解。至关重要的是，谈判游戏使我们能够研究多轮次和跨模型交互，调整复杂性，并避免评估中的意外数据泄漏。我们报告了来自几个主要供应商的六个公开可访问的LM在各种谈判游戏上的结果，评估了自我对弈和交叉对弈性能。值得注意的发现包括：（i）开源模式

    Companies, organizations, and governments increasingly exploit Language Models' (LM) remarkable capability to display agent-like behavior. As LMs are adopted to perform tasks with growing autonomy, there exists an urgent need for reliable and scalable evaluation benchmarks. Current, predominantly static LM benchmarks are ill-suited to evaluate such dynamic applications. Thus, we propose jointly evaluating LM performance and alignment through the lenses of negotiation games. We argue that this common task better reflects real-world deployment conditions while offering insights into LMs' decision-making processes. Crucially, negotiation games allow us to study multi-turn, and cross-model interactions, modulate complexity, and side-step accidental data leakage in evaluation. We report results for six publicly accessible LMs from several major providers on a variety of negotiation games, evaluating both self-play and cross-play performance. Noteworthy findings include: (i) open-source mode
    
[^2]: 单声道多人演讲分离使用高效Transformer模型

    Monaural Multi-Speaker Speech Separation Using Efficient Transformer Model. (arXiv:2308.00010v1 [cs.SD])

    [http://arxiv.org/abs/2308.00010](http://arxiv.org/abs/2308.00010)

    该论文介绍了一种基于Transformer模型的单声道多人演讲分离方法，旨在减少计算复杂性，并在准确性和性能之间取得平衡。

    

    多人聚会问题是一个难以分离或区分来自几个说话者的混合语音中的个别说话者的场景。在这个领域已经进行了几项研究，但模型的大小和复杂性正在与语音分离的准确性和鲁棒性进行权衡。"单声道多人演讲分离"提出了一种基于Transformer架构及其高效形式的演讲分离模型。该模型使用包含多样化说话者话语的LibriMix数据集进行训练。该模型可以从混合音频输入中分离出2个不同的说话者源。该模型的开发目标是减少语音分离模型的计算复杂性，并在与现有语音分离模型性能最小化权衡的同时，实现显著的改进。该项目预计将为语音分离领域的持续研究做出贡献，同时降低计算成本。

    Cocktail party problem is the scenario where it is difficult to separate or distinguish individual speaker from a mixed speech from several speakers. There have been several researches going on in this field but the size and complexity of the model is being traded off with the accuracy and robustness of speech separation. "Monaural multi-speaker speech separation" presents a speech-separation model based on the Transformer architecture and its efficient forms. The model has been trained with the LibriMix dataset containing diverse speakers' utterances. The model separates 2 distinct speaker sources from a mixed audio input. The developed model approaches the reduction in computational complexity of the speech separation model, with minimum tradeoff with the performance of prevalent speech separation model and it has shown significant movement towards that goal. This project foresees, a rise in contribution towards the ongoing research in the field of speech separation with computationa
    

