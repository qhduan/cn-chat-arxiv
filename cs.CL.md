# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Quality-Aware Image-Text Alignment for Real-World Image Quality Assessment](https://arxiv.org/abs/2403.11176) | 提出了一种基于CLIP的自监督方法QualiCLIP，通过质量感知的图像-文本对齐策略，实现了图像质量评估不需要标记MOS的问题 |
| [^2] | [Detecting mental disorder on social media: a ChatGPT-augmented explainable approach](https://arxiv.org/abs/2401.17477) | 本文提出了一种利用大型语言模型，可解释人工智能和对话代理器ChatGPT相结合的新方法，以解决通过社交媒体检测抑郁症的可解释性挑战。通过将Twitter特定变体BERTweet与自解释模型BERT-XDD相结合，并借助ChatGPT将技术解释转化为人类可读的评论，实现了解释能力的同时提高了可解释性。这种方法可以为发展社会负责任的数字平台，促进早期干预做出贡献。 |
| [^3] | [Location Sensitive Embedding for Knowledge Graph Embedding.](http://arxiv.org/abs/2401.10893) | 这篇论文介绍了一种新颖的位置敏感嵌入（LSE）方法，该方法通过关系特定的映射来修改头实体，将关系概念化为线性变换。LSE在知识图谱嵌入领域具有理论基础，同时提出了更高效的变体LSEd。实验证明LSEd在链接预测任务上具有竞争力。 |
| [^4] | [Why can neural language models solve next-word prediction? A mathematical perspective.](http://arxiv.org/abs/2306.17184) | 本文研究了神经语言模型在下一个词预测任务中的成功，在形式语言理论背景下，提出了一种为什么神经语言模型能够学习到组合规则的解释，并在一个现实世界的英语句子示例中提供了零错误的证明。 |
| [^5] | [BUCA: A Binary Classification Approach to Unsupervised Commonsense Question Answering.](http://arxiv.org/abs/2305.15932) | 本文提出了一种更简单的二分类方法，将下游的多项选择题回答任务转换为二分类任务，根据合理性对所有候选答案进行排名，以实现无监督常识问题回答，相较于现有使用知识图谱的UCR方法，我们的方法更为节省数据。 |

# 详细

[^1]: 面向现实世界图像质量评估的质量感知图像-文本对齐

    Quality-Aware Image-Text Alignment for Real-World Image Quality Assessment

    [https://arxiv.org/abs/2403.11176](https://arxiv.org/abs/2403.11176)

    提出了一种基于CLIP的自监督方法QualiCLIP，通过质量感知的图像-文本对齐策略，实现了图像质量评估不需要标记MOS的问题

    

    无参考图像质量评估（NR-IQA）致力于设计一种在没有高质量参考图像的情况下测量图像质量的方法，以符合人类感知，大部分最先进的NR-IQA方法中依赖标注的主观评分（MOS）限制了它们在真实场景中的可扩展性和广泛适用性。为了克服这一限制，我们提出了QualiCLIP（Quality-aware CLIP），这是一种基于CLIP的自监督不需要标记MOS的方法。具体来说，我们引入了一种质量感知的图像-文本对齐策略，使得CLIP生成的表示与图像固有质量相关。从原始图像开始，我们使用不断增加的强度合成地劣化它们。然后，我们训练CLIP根据其与质量相关的反义文本提示的相似性对这些降解图像进行排名，同时保证一致的表达

    arXiv:2403.11176v1 Announce Type: cross  Abstract: No-Reference Image Quality Assessment (NR-IQA) focuses on designing methods to measure image quality in alignment with human perception when a high-quality reference image is unavailable. The reliance on annotated Mean Opinion Scores (MOS) in the majority of state-of-the-art NR-IQA approaches limits their scalability and broader applicability to real-world scenarios. To overcome this limitation, we propose QualiCLIP (Quality-aware CLIP), a CLIP-based self-supervised opinion-unaware method that does not require labeled MOS. In particular, we introduce a quality-aware image-text alignment strategy to make CLIP generate representations that correlate with the inherent quality of the images. Starting from pristine images, we synthetically degrade them with increasing levels of intensity. Then, we train CLIP to rank these degraded images based on their similarity to quality-related antonym text prompts, while guaranteeing consistent represe
    
[^2]: 在社交媒体上检测心理障碍：基于ChatGPT的可解释方法

    Detecting mental disorder on social media: a ChatGPT-augmented explainable approach

    [https://arxiv.org/abs/2401.17477](https://arxiv.org/abs/2401.17477)

    本文提出了一种利用大型语言模型，可解释人工智能和对话代理器ChatGPT相结合的新方法，以解决通过社交媒体检测抑郁症的可解释性挑战。通过将Twitter特定变体BERTweet与自解释模型BERT-XDD相结合，并借助ChatGPT将技术解释转化为人类可读的评论，实现了解释能力的同时提高了可解释性。这种方法可以为发展社会负责任的数字平台，促进早期干预做出贡献。

    

    在数字时代，社交媒体上表达的抑郁症状的频率引起了严重关注，迫切需要先进的方法来及时检测。本文通过提出一种新颖的方法，将大型语言模型（LLM）与可解释的人工智能（XAI）和ChatGPT等对话代理器有效地结合起来，以应对可解释性抑郁症检测的挑战。在我们的方法中，通过将Twitter特定变体BERTweet与一种新型的自解释模型BERT-XDD相结合，实现了解释能力，该模型能够通过掩码注意力提供分类和解释。使用ChatGPT将技术解释转化为可读性强的评论，进一步增强了可解释性。通过引入一种有效且模块化的可解释抑郁症检测方法，我们的方法可以为发展社会负责任的数字平台做出贡献，促进早期干预。

    In the digital era, the prevalence of depressive symptoms expressed on social media has raised serious concerns, necessitating advanced methodologies for timely detection. This paper addresses the challenge of interpretable depression detection by proposing a novel methodology that effectively combines Large Language Models (LLMs) with eXplainable Artificial Intelligence (XAI) and conversational agents like ChatGPT. In our methodology, explanations are achieved by integrating BERTweet, a Twitter-specific variant of BERT, into a novel self-explanatory model, namely BERT-XDD, capable of providing both classification and explanations via masked attention. The interpretability is further enhanced using ChatGPT to transform technical explanations into human-readable commentaries. By introducing an effective and modular approach for interpretable depression detection, our methodology can contribute to the development of socially responsible digital platforms, fostering early intervention and
    
[^3]: 知识图谱嵌入的位置敏感嵌入

    Location Sensitive Embedding for Knowledge Graph Embedding. (arXiv:2401.10893v1 [cs.IR])

    [http://arxiv.org/abs/2401.10893](http://arxiv.org/abs/2401.10893)

    这篇论文介绍了一种新颖的位置敏感嵌入（LSE）方法，该方法通过关系特定的映射来修改头实体，将关系概念化为线性变换。LSE在知识图谱嵌入领域具有理论基础，同时提出了更高效的变体LSEd。实验证明LSEd在链接预测任务上具有竞争力。

    

    知识图谱嵌入将知识图谱转化为连续的、低维度的空间，有助于推理和补全任务。该领域主要分为传统的距离模型和语义匹配模型。传统的距离模型面临的关键挑战是无法有效区分图谱中的“头实体”和“尾实体”。为了解决这个问题，提出了新颖的位置敏感嵌入（LSE）方法。LSE通过关系特定的映射修改头实体，将关系概念化为线性变换而不仅仅是平移。LSE的理论基础，包括其表示能力和与现有模型的联系，都进行了详细研究。一种更简化的变体LSEd利用对角矩阵进行变换以提高实用性能。在对四个大规模数据集进行链接预测的测试中，LSEd要么表现更好，要么具有竞争力。

    Knowledge graph embedding transforms knowledge graphs into a continuous, low-dimensional space, facilitating inference and completion tasks. This field is mainly divided into translational distance models and semantic matching models. A key challenge in translational distance models is their inability to effectively differentiate between 'head' and 'tail' entities in graphs. To address this, the novel location-sensitive embedding (LSE) method has been developed. LSE innovatively modifies the head entity using relation-specific mappings, conceptualizing relations as linear transformations rather than mere translations. The theoretical foundations of LSE, including its representational capabilities and its connections to existing models, have been thoroughly examined. A more streamlined variant, LSEd, employs a diagonal matrix for transformations to enhance practical efficiency. In tests conducted on four large-scale datasets for link prediction, LSEd either outperforms or is competitive
    
[^4]: 为什么神经语言模型可以解决下一个词预测问题？数学视角

    Why can neural language models solve next-word prediction? A mathematical perspective. (arXiv:2306.17184v1 [cs.CL])

    [http://arxiv.org/abs/2306.17184](http://arxiv.org/abs/2306.17184)

    本文研究了神经语言模型在下一个词预测任务中的成功，在形式语言理论背景下，提出了一种为什么神经语言模型能够学习到组合规则的解释，并在一个现实世界的英语句子示例中提供了零错误的证明。

    

    最近，深度学习在自然语言处理领域引起了革命，神经语言模型在下一个词预测方面证明了非常有效。然而，在形式语言理论的背景下，关于神经语言模型在此任务中可以学习到组合规则的成功的严格理论解释尚未被提出，因为尚不清楚为什么神经语言模型可以学习到控制下一个词预测任务的组合规则。在本文中，我们研究了一类可以用来模拟英语句子的现实世界示例的形式语言。我们构建了神经语言模型来解决这种情况下的下一个词预测任务，且错误率为零。我们的证明突出了嵌入层和全连接组件在神经语言模型中的不同作用。

    Recently, deep learning has revolutionized the field of natural language processing, with neural language models proving to be very effective for next-word prediction. However, a rigorous theoretical explanation for their success in the context of formal language theory has not yet been developed, as it is unclear why neural language models can learn the combinatorial rules that govern the next-word prediction task. In this paper, we study a class of formal languages that can be used to model real-world examples of English sentences. We construct neural language models can solve the next-word prediction task in this context with zero error. Our proof highlights the different roles of the embedding layer and the fully connected component within the neural language model.
    
[^5]: BUCA：一种用于无监督常识问题回答的二分类方法

    BUCA: A Binary Classification Approach to Unsupervised Commonsense Question Answering. (arXiv:2305.15932v1 [cs.CL])

    [http://arxiv.org/abs/2305.15932](http://arxiv.org/abs/2305.15932)

    本文提出了一种更简单的二分类方法，将下游的多项选择题回答任务转换为二分类任务，根据合理性对所有候选答案进行排名，以实现无监督常识问题回答，相较于现有使用知识图谱的UCR方法，我们的方法更为节省数据。

    

    随着常识推理数据集的构建变得越来越昂贵且在范围上不可避免地受限，无监督的常识推理(UCR)变得越来越流行。UCR的一种流行方法是利用外部知识将语言模型进行微调(例如，知识图谱)，但这通常需要大量的训练样例。在本文中，我们提出将下游的多项选择题回答任务转换为一个更简单的二分类任务，通过对所有候选答案的合理性进行排名来完成。为了训练模型，我们将知识图谱三元组转换为合理和不合理的文本。广泛的实验结果显示了我们的方法在各种多项选择问题回答基准测试中的有效性。此外，与使用KG的现有UCR方法相比，我们的方法更节省数据。我们的代码可在https://github.com/probe2/BUCA上获取。

    Unsupervised commonsense reasoning (UCR) is becoming increasingly popular as the construction of commonsense reasoning datasets is expensive, and they are inevitably limited in their scope. A popular approach to UCR is to fine-tune language models with external knowledge (e.g., knowledge graphs), but this usually requires a large number of training examples. In this paper, we propose to transform the downstream multiple choice question answering task into a simpler binary classification task by ranking all candidate answers according to their reasonableness. To this end, for training the model, we convert the knowledge graph triples into reasonable and unreasonable texts. Extensive experimental results show the effectiveness of our approach on various multiple choice question answering benchmarks. Furthermore, compared with existing UCR approaches using KGs, ours is less data hungry. Our code is available at https://github.com/probe2/BUCA.
    

