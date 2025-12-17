# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Question Answering Over Spatio-Temporal Knowledge Graph](https://arxiv.org/abs/2402.11542) | 介绍了一个新的基于时空知识图的问答系统STQAD，以解决问答系统在涵盖时空信息的问题上的挑战，提出了一种新的STComplEx嵌入方法STCQA来实现此目标 |
| [^2] | [MIMIR: Masked Image Modeling for Mutual Information-based Adversarial Robustness.](http://arxiv.org/abs/2312.04960) | MIMIR提出了一种新颖的防御方法，通过在预训练中利用遮罩图像建模，构建了一个不同的对抗性训练方法。该方法旨在增强Vision Transformers（ViTs）对抗攻击的鲁棒性。 |
| [^3] | [What Symptoms and How Long? An Interpretable AI Approach for Depression Detection in Social Media.](http://arxiv.org/abs/2305.13127) | 本论文介绍了一种使用可解释的人工智能方法在社交媒体中检测抑郁症的方法。该方法创新地检测和解释抑郁症状及其持续时间，并通过大规模数据集的实证分析表明优于其他方法，发现了新的未注意到的症状。 |

# 详细

[^1]: 基于时空知识图的问答系统

    Question Answering Over Spatio-Temporal Knowledge Graph

    [https://arxiv.org/abs/2402.11542](https://arxiv.org/abs/2402.11542)

    介绍了一个新的基于时空知识图的问答系统STQAD，以解决问答系统在涵盖时空信息的问题上的挑战，提出了一种新的STComplEx嵌入方法STCQA来实现此目标

    

    时空知识图（STKG）通过整合时间和位置信息扩展了知识图（KG）的概念。尽管研究界关注知识图问答（KGQA），但基于STKG的涵盖时空信息的问题回答领域仍未被广泛探讨。此外，缺乏全面的数据集也阻碍了该领域的进展。为解决这一问题，我们提出了STQAD，这是一个包括10,000个自然语言问题的面向时空知识图问答（STKGQA）数据集。不幸的是，各种最先进的KGQA方法在我们的数据集上远未达到令人满意的性能。为此，我们提出了STCQA，一种新的时空KGQA方法，利用了一种名为STComplEx的新型STKG嵌入方法。通过从问题中提取时间和空间信息，我们的问答模型可以更好地理解问题。

    arXiv:2402.11542v1 Announce Type: cross  Abstract: Spatio-temporal knowledge graphs (STKGs) extend the concept of knowledge graphs (KGs) by incorporating time and location information. While the research community's focus on Knowledge Graph Question Answering (KGQA), the field of answering questions incorporating both spatio-temporal information based on STKGs remains largely unexplored. Furthermore, a lack of comprehensive datasets also has hindered progress in this area. To address this issue, we present STQAD, a dataset comprising 10,000 natural language questions for spatio-temporal knowledge graph question answering (STKGQA). Unfortunately, various state-of-the-art KGQA approaches fall far short of achieving satisfactory performance on our dataset. In response, we propose STCQA, a new spatio-temporal KGQA approach that utilizes a novel STKG embedding method named STComplEx. By extracting temporal and spatial information from a question, our QA model can better comprehend the quest
    
[^2]: MIMIR: 基于互信息的对抗鲁棒性的遮罩图像建模

    MIMIR: Masked Image Modeling for Mutual Information-based Adversarial Robustness. (arXiv:2312.04960v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2312.04960](http://arxiv.org/abs/2312.04960)

    MIMIR提出了一种新颖的防御方法，通过在预训练中利用遮罩图像建模，构建了一个不同的对抗性训练方法。该方法旨在增强Vision Transformers（ViTs）对抗攻击的鲁棒性。

    

    视觉变压器（ViTs）相对于卷积神经网络（CNNs）在各种任务上实现了卓越的性能，但ViTs也容易受到对抗性攻击。对抗性训练是建立强大的CNN模型的最成功方法之一。因此，最近的研究探索了基于ViTs和CNNs之间的差异的对抗性训练的新方法，如更好的训练策略，防止注意力集中在单个块上，或丢弃低注意力的嵌入。然而，这些方法仍然遵循传统监督对抗训练的设计，限制了对ViTs的对抗训练的潜力。本文提出了一种新颖的防御方法MIMIR，旨在通过利用预训练中的遮罩图像建模构建不同的对抗性训练方法。我们创建了一个自编码器，它接受对抗性例子作为输入，但将干净的例子作为建模目标。然后，我们创建了一个互信息（MI）

    Vision Transformers (ViTs) achieve superior performance on various tasks compared to convolutional neural networks (CNNs), but ViTs are also vulnerable to adversarial attacks. Adversarial training is one of the most successful methods to build robust CNN models. Thus, recent works explored new methodologies for adversarial training of ViTs based on the differences between ViTs and CNNs, such as better training strategies, preventing attention from focusing on a single block, or discarding low-attention embeddings. However, these methods still follow the design of traditional supervised adversarial training, limiting the potential of adversarial training on ViTs. This paper proposes a novel defense method, MIMIR, which aims to build a different adversarial training methodology by utilizing Masked Image Modeling at pre-training. We create an autoencoder that accepts adversarial examples as input but takes the clean examples as the modeling target. Then, we create a mutual information (MI
    
[^3]: 什么症状以及持续多久？一种可解释的人工智能方法用于社交媒体中的抑郁症检测。

    What Symptoms and How Long? An Interpretable AI Approach for Depression Detection in Social Media. (arXiv:2305.13127v2 [q-bio.QM] UPDATED)

    [http://arxiv.org/abs/2305.13127](http://arxiv.org/abs/2305.13127)

    本论文介绍了一种使用可解释的人工智能方法在社交媒体中检测抑郁症的方法。该方法创新地检测和解释抑郁症状及其持续时间，并通过大规模数据集的实证分析表明优于其他方法，发现了新的未注意到的症状。

    

    抑郁症是最常见和严重的精神疾病，引发了严重的经济和社会影响。抑郁症的检测对于早期干预以减轻这些后果至关重要。这种高风险的决策本质上需要可解释性。虽然有一些抑郁症检测研究试图基于重要性分数或关注权重解释决策，但这些解释与临床抑郁症诊断标准不一致，后者基于抑郁症状。为了填补这一空白，我们遵循计算设计科学范式，开发了一种新颖的多尺度时间原型网络(MSTPNet)。MSTPNet创新地检测和解释抑郁症状及其持续时间。使用大规模数据集进行深入实证分析表明，MSTPNet在F1分数0.851上优于最先进的抑郁症检测方法。这个结果还揭示了未在调查方法中注意到的新症状，例如分享。

    Depression is the most prevalent and serious mental illness, which induces grave financial and societal ramifications. Depression detection is key for early intervention to mitigate those consequences. Such a high-stake decision inherently necessitates interpretability. Although a few depression detection studies attempt to explain the decision based on the importance score or attention weights, these explanations misalign with the clinical depression diagnosis criterion that is based on depressive symptoms. To fill this gap, we follow the computational design science paradigm to develop a novel Multi-Scale Temporal Prototype Network (MSTPNet). MSTPNet innovatively detects and interprets depressive symptoms as well as how long they last. Extensive empirical analyses using a large-scale dataset show that MSTPNet outperforms state-of-the-art depression detection methods with an F1-score of 0.851. This result also reveals new symptoms that are unnoted in the survey approach, such as shari
    

