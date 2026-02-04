# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Variance Reduction Based Experience Replay for Policy Optimization](https://arxiv.org/abs/2110.08902) | 引入了基于方差减少的经验回放框架，实现了选择性重复利用相关样本来改善策略梯度估计，并构建了高效的离策略算法PG-VRER。 |
| [^2] | [AlignAtt: Using Attention-based Audio-Translation Alignments as a Guide for Simultaneous Speech Translation.](http://arxiv.org/abs/2305.11408) | AlignAtt是一种新型的SimulST策略，使用基于注意力的音频翻译对齐来指导模型，在BLEU和延迟方面均优于之前的策略。 |

# 详细

[^1]: 基于方差减少的经验回放用于策略优化

    Variance Reduction Based Experience Replay for Policy Optimization

    [https://arxiv.org/abs/2110.08902](https://arxiv.org/abs/2110.08902)

    引入了基于方差减少的经验回放框架，实现了选择性重复利用相关样本来改善策略梯度估计，并构建了高效的离策略算法PG-VRER。

    

    在复杂随机系统上进行强化学习时，有效利用历史样本中的信息以加速策略优化是很有必要的。传统的经验回放虽然有效，但是将所有观测都视为相同，忽略了它们的相对重要性。为了解决这一限制，我们引入了一种新颖的方差减少经验回放（VRER）框架，实现对相关样本的选择性重复利用，从而改善策略梯度估计。VRER作为一种适应性方法，可以无缝集成到不同的策略优化算法中，构建了我们高效的离策略算法Policy Optimization with VRER (PG-VRER)。此外，文献中对经验回放方法缺乏严格的理论理解，这促使我们引入一个新颖的理论框架，考虑样本依赖性。

    arXiv:2110.08902v3 Announce Type: replace-cross  Abstract: For reinforcement learning on complex stochastic systems, it is desirable to effectively leverage the information from historical samples collected in previous iterations to accelerate policy optimization. Classical experience replay, while effective, treats all observations uniformly, neglecting their relative importance. To address this limitation, we introduce a novel Variance Reduction Experience Replay (VRER) framework, enabling the selective reuse of relevant samples to improve policy gradient estimation. VRER, as an adaptable method that can seamlessly integrate with different policy optimization algorithms, forms the foundation of our sample-efficient off-policy algorithm known as Policy Optimization with VRER (PG-VRER). Furthermore, the lack of a rigorous theoretical understanding of the experience replay method in the literature motivates us to introduce a novel theoretical framework that accounts for sample dependenc
    
[^2]: AlignAtt：使用基于注意力的音频翻译对齐作为同时语音翻译的指导

    AlignAtt: Using Attention-based Audio-Translation Alignments as a Guide for Simultaneous Speech Translation. (arXiv:2305.11408v1 [cs.CL])

    [http://arxiv.org/abs/2305.11408](http://arxiv.org/abs/2305.11408)

    AlignAtt是一种新型的SimulST策略，使用基于注意力的音频翻译对齐来指导模型，在BLEU和延迟方面均优于之前的策略。

    

    注意力是当今自然语言处理中最常用的架构的核心机制，并已从许多角度进行分析，包括其在机器翻译相关任务中的有效性。在这些研究中，注意力在输入文本被替换为音频片段的情况下，也是获取有关单词对齐的有用信息的一种方式，例如语音翻译（ST）任务。在本文中，我们提出了AlignAtt，一种新颖的同时ST（SimulST）策略，它利用注意力信息来生成源-目标对齐，以在推理过程中指导模型。通过对MuST-C v1.0的8种语言对的实验，我们发现，在线下训练的模型上应用先前的最新SimulST策略，AlignAtt在BLEU方面获得了2个分数的提高，并且8种语言的延迟缩减在0.5秒到0.8秒之间。

    Attention is the core mechanism of today's most used architectures for natural language processing and has been analyzed from many perspectives, including its effectiveness for machine translation-related tasks. Among these studies, attention resulted to be a useful source of information to get insights about word alignment also when the input text is substituted with audio segments, as in the case of the speech translation (ST) task. In this paper, we propose AlignAtt, a novel policy for simultaneous ST (SimulST) that exploits the attention information to generate source-target alignments that guide the model during inference. Through experiments on the 8 language pairs of MuST-C v1.0, we show that AlignAtt outperforms previous state-of-the-art SimulST policies applied to offline-trained models with gains in terms of BLEU of 2 points and latency reductions ranging from 0.5s to 0.8s across the 8 languages.
    

