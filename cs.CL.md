# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [AlignAtt: Using Attention-based Audio-Translation Alignments as a Guide for Simultaneous Speech Translation.](http://arxiv.org/abs/2305.11408) | AlignAtt是一种新型的SimulST策略，使用基于注意力的音频翻译对齐来指导模型，在BLEU和延迟方面均优于之前的策略。 |

# 详细

[^1]: AlignAtt：使用基于注意力的音频翻译对齐作为同时语音翻译的指导

    AlignAtt: Using Attention-based Audio-Translation Alignments as a Guide for Simultaneous Speech Translation. (arXiv:2305.11408v1 [cs.CL])

    [http://arxiv.org/abs/2305.11408](http://arxiv.org/abs/2305.11408)

    AlignAtt是一种新型的SimulST策略，使用基于注意力的音频翻译对齐来指导模型，在BLEU和延迟方面均优于之前的策略。

    

    注意力是当今自然语言处理中最常用的架构的核心机制，并已从许多角度进行分析，包括其在机器翻译相关任务中的有效性。在这些研究中，注意力在输入文本被替换为音频片段的情况下，也是获取有关单词对齐的有用信息的一种方式，例如语音翻译（ST）任务。在本文中，我们提出了AlignAtt，一种新颖的同时ST（SimulST）策略，它利用注意力信息来生成源-目标对齐，以在推理过程中指导模型。通过对MuST-C v1.0的8种语言对的实验，我们发现，在线下训练的模型上应用先前的最新SimulST策略，AlignAtt在BLEU方面获得了2个分数的提高，并且8种语言的延迟缩减在0.5秒到0.8秒之间。

    Attention is the core mechanism of today's most used architectures for natural language processing and has been analyzed from many perspectives, including its effectiveness for machine translation-related tasks. Among these studies, attention resulted to be a useful source of information to get insights about word alignment also when the input text is substituted with audio segments, as in the case of the speech translation (ST) task. In this paper, we propose AlignAtt, a novel policy for simultaneous ST (SimulST) that exploits the attention information to generate source-target alignments that guide the model during inference. Through experiments on the 8 language pairs of MuST-C v1.0, we show that AlignAtt outperforms previous state-of-the-art SimulST policies applied to offline-trained models with gains in terms of BLEU of 2 points and latency reductions ranging from 0.5s to 0.8s across the 8 languages.
    

