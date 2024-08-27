# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [SpeechDPR: End-to-End Spoken Passage Retrieval for Open-Domain Spoken Question Answering.](http://arxiv.org/abs/2401.13463) | SpeechDPR是第一个用于开放领域口语问答的端到端框架，能够从口语存档中检索可能包含答案的段落。通过融合无监督ASR和文本密集检索器的知识，SpeechDPR能够获得较好的性能，并且在UASR性能较差时表现更加鲁棒。 |
| [^2] | [A transformer-based method for zero and few-shot biomedical named entity recognition.](http://arxiv.org/abs/2305.04928) | 本文提出了一种基于Transformer的生物医学领域零样本和少样本NER方法。此方法利用预训练学习给定和潜在类别之间的语义关系，将多类标记分类任务转换为二元标记分类，能够在不同数量的样本情况下达到良好的识别效果。 |

# 详细

[^1]: SpeechDPR: 开放领域口语问答的端到端口语段落检索

    SpeechDPR: End-to-End Spoken Passage Retrieval for Open-Domain Spoken Question Answering. (arXiv:2401.13463v1 [cs.CL])

    [http://arxiv.org/abs/2401.13463](http://arxiv.org/abs/2401.13463)

    SpeechDPR是第一个用于开放领域口语问答的端到端框架，能够从口语存档中检索可能包含答案的段落。通过融合无监督ASR和文本密集检索器的知识，SpeechDPR能够获得较好的性能，并且在UASR性能较差时表现更加鲁棒。

    

    口语问答(SQA)是机器通过在给定口语段落中找到答案范围来回答用户问题的关键。过去的SQA方法没有使用ASR，以避免识别错误和词汇外问题。然而，实际的开放领域SQA(openSQA)问题中，机器需要首先从口语存档中检索可能包含答案的段落。本文提出了第一个已知的用于openSQA问题检索组件的端到端框架SpeechDPR。SpeechDPR通过从无监督ASR(UASR)和文本密集检索器(TDR)的级联模型中提炼知识，学习句子级语义表示。不需要手动转录的语音数据。初步实验表明，与级联的UASR和TDR模型相比，性能相当，并且在UASR性能较差时显著提高，验证了这种方法更加鲁棒。

    Spoken Question Answering (SQA) is essential for machines to reply to user's question by finding the answer span within a given spoken passage. SQA has been previously achieved without ASR to avoid recognition errors and Out-of-Vocabulary (OOV) problems. However, the real-world problem of Open-domain SQA (openSQA), in which the machine needs to first retrieve passages that possibly contain the answer from a spoken archive in addition, was never considered. This paper proposes the first known end-to-end framework, Speech Dense Passage Retriever (SpeechDPR), for the retrieval component of the openSQA problem. SpeechDPR learns a sentence-level semantic representation by distilling knowledge from the cascading model of unsupervised ASR (UASR) and text dense retriever (TDR). No manually transcribed speech data is needed. Initial experiments showed performance comparable to the cascading model of UASR and TDR, and significantly better when UASR was poor, verifying this approach is more robus
    
[^2]: 基于Transformer的零样本和少样本生物医学命名实体识别方法

    A transformer-based method for zero and few-shot biomedical named entity recognition. (arXiv:2305.04928v1 [cs.CL])

    [http://arxiv.org/abs/2305.04928](http://arxiv.org/abs/2305.04928)

    本文提出了一种基于Transformer的生物医学领域零样本和少样本NER方法。此方法利用预训练学习给定和潜在类别之间的语义关系，将多类标记分类任务转换为二元标记分类，能够在不同数量的样本情况下达到良好的识别效果。

    

    在生物医学领域中，有监督的命名实体识别（NER）依赖于具有给定命名实体的大量注释文本，其创建可能耗时且昂贵。此外，提取新实体通常需要进行额外的注释任务和重新训练模型。为解决这些挑战，本文提出了一种基于Transformer的生物医学领域零样本和少样本NER方法。该方法基于将多类标记分类任务转换为二元标记分类（标记包含搜索的实体或不包含搜索的实体），并在更多的数据集和生物医学实体上进行预训练，从而可学习到给定和潜在类别之间的语义关系。在9种不同的生物医学实体上，我们在零样本NER、一次样本NER、10次样本NER和100次样本NER上实现了平均F1得分分别为35.44％、50.10％、69.94％和79.51％。

    Supervised named entity recognition (NER) in the biomedical domain is dependent on large sets of annotated texts with the given named entities, whose creation can be time-consuming and expensive. Furthermore, the extraction of new entities often requires conducting additional annotation tasks and retraining the model. To address these challenges, this paper proposes a transformer-based method for zero- and few-shot NER in the biomedical domain. The method is based on transforming the task of multi-class token classification into binary token classification (token contains the searched entity or does not contain the searched entity) and pre-training on a larger amount of datasets and biomedical entities, from where the method can learn semantic relations between the given and potential classes. We have achieved average F1 scores of 35.44% for zero-shot NER, 50.10% for one-shot NER, 69.94% for 10-shot NER, and 79.51% for 100-shot NER on 9 diverse evaluated biomedical entities with PubMed
    

