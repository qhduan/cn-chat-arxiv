# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Data-Efficient Sleep Staging with Synthetic Time Series Pretraining](https://arxiv.org/abs/2403.08592) | 通过预测合成时间序列的频率内容进行预训练，实现了在有限数据和少受试者情况下超越完全监督学习的方法 |
| [^2] | [Predicting O-GlcNAcylation Sites in Mammalian Proteins with Transformers and RNNs Trained with a New Loss Function](https://arxiv.org/abs/2402.17131) | 本研究提出了一种新的损失函数，称为加权焦点可微MCC，用于改善分类模型的性能，并在预测哺乳动物蛋白质中的O-GlcNAcylation位点方面取得了进展 |
| [^3] | [Tabdoor: Backdoor Vulnerabilities in Transformer-based Neural Networks for Tabular Data.](http://arxiv.org/abs/2311.07550) | 这项研究全面分析了使用DNNs对表格数据进行后门攻击，揭示了基于转换器的DNNs对表格数据非常容易受到后门攻击，甚至只需最小的特征值修改。该攻击还可以推广到其他模型。 |
| [^4] | [Realtime Motion Generation with Active Perception Using Attention Mechanism for Cooking Robot.](http://arxiv.org/abs/2309.14837) | 该论文介绍了一种使用注意机制的预测性递归神经网络，能够实现实时感知和动作生成，以支持烹饪机器人在煮鸡蛋过程中对鸡蛋状态的感知和搅拌动作的调整。 |

# 详细

[^1]: 用合成时间序列预训练实现高效的睡眠分期

    Data-Efficient Sleep Staging with Synthetic Time Series Pretraining

    [https://arxiv.org/abs/2403.08592](https://arxiv.org/abs/2403.08592)

    通过预测合成时间序列的频率内容进行预训练，实现了在有限数据和少受试者情况下超越完全监督学习的方法

    

    分析脑电图（EEG）时间序列可能具有挑战性，特别是在深度神经网络中，由于人类受试者之间的大量变异和通常规模较小的数据集。为了解决这些挑战，提出了各种策略，例如自监督学习，但它们通常依赖于广泛的实证数据集。受计算机视觉最新进展的启发，我们提出了一种预训练任务，称为“频率预训练”，通过预测随机生成的合成时间序列的频率内容来为睡眠分期预训练神经网络。我们的实验表明，我们的方法在有限数据和少受试者的情况下优于完全监督学习，并在许多受试者的情境中表现相匹配。此外，我们的结果强调了频率信息对于睡眠分期评分的相关性，同时表明深度神经网络利用了超出频率信息的信息。

    arXiv:2403.08592v1 Announce Type: new  Abstract: Analyzing electroencephalographic (EEG) time series can be challenging, especially with deep neural networks, due to the large variability among human subjects and often small datasets. To address these challenges, various strategies, such as self-supervised learning, have been suggested, but they typically rely on extensive empirical datasets. Inspired by recent advances in computer vision, we propose a pretraining task termed "frequency pretraining" to pretrain a neural network for sleep staging by predicting the frequency content of randomly generated synthetic time series. Our experiments demonstrate that our method surpasses fully supervised learning in scenarios with limited data and few subjects, and matches its performance in regimes with many subjects. Furthermore, our results underline the relevance of frequency information for sleep stage scoring, while also demonstrating that deep neural networks utilize information beyond fr
    
[^2]: 使用Transformer和RNN在经过训练的新损失函数下预测哺乳动物蛋白质中的O-GlcNAcylation位点

    Predicting O-GlcNAcylation Sites in Mammalian Proteins with Transformers and RNNs Trained with a New Loss Function

    [https://arxiv.org/abs/2402.17131](https://arxiv.org/abs/2402.17131)

    本研究提出了一种新的损失函数，称为加权焦点可微MCC，用于改善分类模型的性能，并在预测哺乳动物蛋白质中的O-GlcNAcylation位点方面取得了进展

    

    糖基化是一种蛋白质修饰，在功能和结构上起着多种重要作用。O-GlcNAcylation是糖基化的一种亚型，有潜力成为治疗的重要靶点，但在2023年之前尚未有可靠预测O-GlcNAcylation位点的方法；2021年的一篇评论正确指出已发表的模型不足，并且未能泛化。此外，许多模型已不再可用。2023年，一篇具有F$_1$分数36.17%和MCC分数34.57%的大型数据集上的显着更好的RNN模型被发表。本文首次试图通过Transformer编码器提高这些指标。尽管Transformer在该数据集上表现出色，但其性能仍不及先前发表的RNN。然后我们创建了一种新的损失函数，称为加权焦点可微MCC，以提高分类模型的性能。

    arXiv:2402.17131v1 Announce Type: new  Abstract: Glycosylation, a protein modification, has multiple essential functional and structural roles. O-GlcNAcylation, a subtype of glycosylation, has the potential to be an important target for therapeutics, but methods to reliably predict O-GlcNAcylation sites had not been available until 2023; a 2021 review correctly noted that published models were insufficient and failed to generalize. Moreover, many are no longer usable. In 2023, a considerably better RNN model with an F$_1$ score of 36.17% and an MCC of 34.57% on a large dataset was published. This article first sought to improve these metrics using transformer encoders. While transformers displayed high performance on this dataset, their performance was inferior to that of the previously published RNN. We then created a new loss function, which we call the weighted focal differentiable MCC, to improve the performance of classification models. RNN models trained with this new function di
    
[^3]: Tabdoor：基于转换器的表格数据神经网络存在后门漏洞

    Tabdoor: Backdoor Vulnerabilities in Transformer-based Neural Networks for Tabular Data. (arXiv:2311.07550v2 [cs.CR] UPDATED)

    [http://arxiv.org/abs/2311.07550](http://arxiv.org/abs/2311.07550)

    这项研究全面分析了使用DNNs对表格数据进行后门攻击，揭示了基于转换器的DNNs对表格数据非常容易受到后门攻击，甚至只需最小的特征值修改。该攻击还可以推广到其他模型。

    

    深度神经网络(DNNs)在各个领域都显示出巨大的潜力。与这些发展同时，与DNN训练相关的漏洞，如后门攻击，是一个重大关切。这些攻击涉及在模型训练过程中微妙地插入触发器，从而允许操纵预测。最近，由于转换器模型的崛起，DNNs用于表格数据越来越受关注。我们的研究对使用DNNs对表格数据进行后门攻击进行了全面分析，特别关注转换器。鉴于表格数据的固有复杂性，我们探究了嵌入后门的挑战。通过对基准数据集进行系统实验，我们发现基于转换器的DNNs对表格数据非常容易受到后门攻击，即使只有最小的特征值修改。我们还验证了我们的攻击可以推广到其他模型，如XGBoost和DeepFM。我们的研究结果几乎表明后门攻击可以完美实现。

    Deep Neural Networks (DNNs) have shown great promise in various domains. Alongside these developments, vulnerabilities associated with DNN training, such as backdoor attacks, are a significant concern. These attacks involve the subtle insertion of triggers during model training, allowing for manipulated predictions.More recently, DNNs for tabular data have gained increasing attention due to the rise of transformer models.  Our research presents a comprehensive analysis of backdoor attacks on tabular data using DNNs, particularly focusing on transformers. Given the inherent complexities of tabular data, we explore the challenges of embedding backdoors. Through systematic experimentation across benchmark datasets, we uncover that transformer-based DNNs for tabular data are highly susceptible to backdoor attacks, even with minimal feature value alterations. We also verify that our attack can be generalized to other models, like XGBoost and DeepFM. Our results indicate nearly perfect attac
    
[^4]: 使用注意机制进行实时动作生成和主动感知的烹饪机器人

    Realtime Motion Generation with Active Perception Using Attention Mechanism for Cooking Robot. (arXiv:2309.14837v1 [cs.RO])

    [http://arxiv.org/abs/2309.14837](http://arxiv.org/abs/2309.14837)

    该论文介绍了一种使用注意机制的预测性递归神经网络，能够实现实时感知和动作生成，以支持烹饪机器人在煮鸡蛋过程中对鸡蛋状态的感知和搅拌动作的调整。

    

    为了支持人类的日常生活，机器人需要自主学习，适应物体和环境，并执行适当的动作。我们尝试使用真实的食材煮炒鸡蛋的任务，其中机器人需要实时感知鸡蛋的状态并调整搅拌动作，同时鸡蛋被加热且状态不断变化。在以前的研究中，处理变化的物体被发现是具有挑战性的，因为感知信息包括动态的、重要或嘈杂的信息，而且每次应该关注的模态不断变化，这使得实现实时感知和动作生成变得困难。我们提出了一个带有注意机制的预测性递归神经网络，可以权衡传感器输入，区分每种模态的重要性和可靠性，实现快速和高效的感知和动作生成。模型通过示范学习进行训练，并允许不断更新。

    To support humans in their daily lives, robots are required to autonomously learn, adapt to objects and environments, and perform the appropriate actions. We tackled on the task of cooking scrambled eggs using real ingredients, in which the robot needs to perceive the states of the egg and adjust stirring movement in real time, while the egg is heated and the state changes continuously. In previous works, handling changing objects was found to be challenging because sensory information includes dynamical, both important or noisy information, and the modality which should be focused on changes every time, making it difficult to realize both perception and motion generation in real time. We propose a predictive recurrent neural network with an attention mechanism that can weigh the sensor input, distinguishing how important and reliable each modality is, that realize quick and efficient perception and motion generation. The model is trained with learning from the demonstration, and allow
    

