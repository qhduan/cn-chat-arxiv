# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Multi-Task Media-Bias Analysis Generalization for Pre-Trained Identification of Expressions](https://arxiv.org/abs/2403.07910) | MAGPIE是第一个为媒体偏见检测定制的大规模多任务预训练方法，在媒体偏见检测方面表现优异，并且相对于单一任务方法需要更少的微调步骤。 |

# 详细

[^1]: 多任务媒体偏见分析通用化的预训练表达识别

    Multi-Task Media-Bias Analysis Generalization for Pre-Trained Identification of Expressions

    [https://arxiv.org/abs/2403.07910](https://arxiv.org/abs/2403.07910)

    MAGPIE是第一个为媒体偏见检测定制的大规模多任务预训练方法，在媒体偏见检测方面表现优异，并且相对于单一任务方法需要更少的微调步骤。

    

    媒体偏见检测是一个复杂的、多方面的问题，传统上通过使用单一任务模型和小型领域内数据集来解决，因此缺乏泛化能力。为了解决这一问题，我们介绍了MAGPIE，这是第一个专门为媒体偏见检测定制的大规模多任务预训练方法。为了实现规模化的预训练，我们提出了大偏见混合（LBM），这是一个包含59个与偏见相关的任务的编译。MAGPIE在Bias Annotation By Experts (BABE)数据集上的媒体偏见检测方面优于先前的方法，F1分数相对提高了3.3%。MAGPIE在Media Bias Identification Benchmark (MBIB)中的8个任务中有5个方面表现优于先前的模型。使用RoBERTa编码器，MAGPIE仅需要相对于单一任务方法的15%的微调步骤。我们的评估表明，比如任务如情感和情绪会增强所有学习，所有任务会增强假新闻检测，

    arXiv:2403.07910v1 Announce Type: cross  Abstract: Media bias detection poses a complex, multifaceted problem traditionally tackled using single-task models and small in-domain datasets, consequently lacking generalizability. To address this, we introduce MAGPIE, the first large-scale multi-task pre-training approach explicitly tailored for media bias detection. To enable pre-training at scale, we present Large Bias Mixture (LBM), a compilation of 59 bias-related tasks. MAGPIE outperforms previous approaches in media bias detection on the Bias Annotation By Experts (BABE) dataset, with a relative improvement of 3.3% F1-score. MAGPIE also performs better than previous models on 5 out of 8 tasks in the Media Bias Identification Benchmark (MBIB). Using a RoBERTa encoder, MAGPIE needs only 15% of finetuning steps compared to single-task approaches. Our evaluation shows, for instance, that tasks like sentiment and emotionality boost all learning, all tasks enhance fake news detection, and s
    

