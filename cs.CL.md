# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Unsupervised Chunking with Hierarchical RNN.](http://arxiv.org/abs/2309.04919) | 本论文提出了一种无监督的句块化方法，使用分层循环神经网络来建模单词到句块和句块到句子的组合。在实验中取得了显著的改进，将短语F1得分提高了6个百分点。 |

# 详细

[^1]: 无监督句块化与分层循环神经网络

    Unsupervised Chunking with Hierarchical RNN. (arXiv:2309.04919v1 [cs.CL])

    [http://arxiv.org/abs/2309.04919](http://arxiv.org/abs/2309.04919)

    本论文提出了一种无监督的句块化方法，使用分层循环神经网络来建模单词到句块和句块到句子的组合。在实验中取得了显著的改进，将短语F1得分提高了6个百分点。

    

    在自然语言处理（NLP）中，预测语言结构，如解析和句块化，主要依赖于人工标注的句法结构。本文介绍了一种无监督的句块化方法，这是一种以非层次化方式对单词进行分组的句法任务。我们提出了一个两层分层循环神经网络（HRNN）来建模单词到句块和句块到句子的组合。我们的方法包括两个阶段的训练过程：使用无监督解析器进行预训练，然后在下游NLP任务上进行微调。在CoNLL-2000数据集上的实验显示，与现有的无监督方法相比，我们取得了显著的改进，将短语F1得分提高了6个百分点。此外，与下游任务的微调还带来了额外的性能提升。有趣的是，我们观察到句块结构在神经模型的下游任务训练过程中是短暂的。本研究对于推动无监督句块化的进展起到了重要作用。

    In Natural Language Processing (NLP), predicting linguistic structures, such as parsing and chunking, has mostly relied on manual annotations of syntactic structures. This paper introduces an unsupervised approach to chunking, a syntactic task that involves grouping words in a non-hierarchical manner. We present a two-layer Hierarchical Recurrent Neural Network (HRNN) designed to model word-to-chunk and chunk-to-sentence compositions. Our approach involves a two-stage training process: pretraining with an unsupervised parser and finetuning on downstream NLP tasks. Experiments on the CoNLL-2000 dataset reveal a notable improvement over existing unsupervised methods, enhancing phrase F1 score by up to 6 percentage points. Further, finetuning with downstream tasks results in an additional performance improvement. Interestingly, we observe that the emergence of the chunking structure is transient during the neural model's downstream-task training. This study contributes to the advancement 
    

