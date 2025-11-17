# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [MoPE: Parameter-Efficient and Scalable Multimodal Fusion via Mixture of Prompt Experts](https://arxiv.org/abs/2403.10568) | 本文提出了MoPE技术，通过解开提示以自适应捕获数据集级和实例级特征，引入了混合Prompt专家来增强表达能力，并且在多模态融合中表现出更大的表达能力和可扩展性。 |
| [^2] | [Survey in Characterization of Semantic Change](https://arxiv.org/abs/2402.19088) | 语义变化对计算语言学算法的结果质量可能会产生影响，因此重要性日益凸显。 |
| [^3] | [GreatSplicing: A Semantically Rich Splicing Dataset.](http://arxiv.org/abs/2310.10070) | 本文提出了一个语义丰富的拼接数据集GreatSplicing，通过包括大量不同语义类别的拼接区域，训练的模型在拼接痕迹检测上表现出较低的误识率和更好的跨数据集检测能力。 |
| [^4] | [Towards Efficient and Trustworthy AI Through Hardware-Algorithm-Communication Co-Design.](http://arxiv.org/abs/2309.15942) | 通过硬件-算法-通信协同设计，本论文提出了一种实现高效可信的人工智能的研究方法，即通过结合物理洞见、高效信息处理原则、最优不确定度量结果和分布式处理准则，来提高神经网络算法的效率和可信度。 |

# 详细

[^1]: MoPE：通过Prompt专家混合实现参数高效和可扩展的多模态融合

    MoPE: Parameter-Efficient and Scalable Multimodal Fusion via Mixture of Prompt Experts

    [https://arxiv.org/abs/2403.10568](https://arxiv.org/abs/2403.10568)

    本文提出了MoPE技术，通过解开提示以自适应捕获数据集级和实例级特征，引入了混合Prompt专家来增强表达能力，并且在多模态融合中表现出更大的表达能力和可扩展性。

    

    Prompt调整已经证明在融合多模态任务的单模基础模型时具有参数效率性。然而，其有限的适应性和表达能力导致性能不佳与其他调整方法相比。本文通过将简单提示解开以自适应地捕获数据集级和实例级特征来解决这个问题。建立在这种解开的基础上，我们引入了Prompt专家的混合（MoPE）技术来增强表达能力。MoPE利用多模态配对先验在每个实例基础上路由最有效的提示。与简单提示相比，我们基于MoPE的条件提示对多模态融合具有更大的表达能力，在训练数据和可训练参数总数上具有更好的扩展性。我们还研究了一个专家路由的正则化项，导致专家的不断发展专长，不同专家专注于不同的特征。

    arXiv:2403.10568v1 Announce Type: cross  Abstract: Prompt-tuning has demonstrated parameter-efficiency in fusing unimodal foundation models for multimodal tasks. However, its limited adaptivity and expressiveness lead to suboptimal performance when compared with other tuning methods. In this paper, we address this issue by disentangling the vanilla prompts to adaptively capture dataset-level and instance-level features. Building upon this disentanglement, we introduce the mixture of prompt experts (MoPE) technique to enhance expressiveness. MoPE leverages multimodal pairing priors to route the most effective prompt on a per-instance basis. Compared to vanilla prompting, our MoPE-based conditional prompting exhibits greater expressiveness for multimodal fusion, scaling better with the training data and the overall number of trainable parameters. We also study a regularization term for expert routing, leading to emergent expert specialization, where different experts focus on different c
    
[^2]: 对语义变化特征的调查

    Survey in Characterization of Semantic Change

    [https://arxiv.org/abs/2402.19088](https://arxiv.org/abs/2402.19088)

    语义变化对计算语言学算法的结果质量可能会产生影响，因此重要性日益凸显。

    

    活语言不断发展，以吸纳人类社会的文化变化。这种演变通过新词语（新单词）或单词的语义变化（赋予已有单词新的含义）来体现。理解单词的含义对解释来自不同文化（地方用语或俚语）、领域（例如技术术语）或时代的文本至关重要。在计算机科学中，这些单词与计算语言学算法相关，例如翻译、信息检索、问答等。语义变化可能会影响这些算法的结果质量。因此，了解和形式化表征这些变化是很重要的。研究这种影响是计算语言学界近期引起关注的问题。几种方法提出了检测语义变化的方法，具有较高的精度，但需要更多努力来对其进行表征。

    arXiv:2402.19088v1 Announce Type: cross  Abstract: Live languages continuously evolve to integrate the cultural change of human societies. This evolution manifests through neologisms (new words) or \textbf{semantic changes} of words (new meaning to existing words). Understanding the meaning of words is vital for interpreting texts coming from different cultures (regionalism or slang), domains (e.g., technical terms), or periods. In computer science, these words are relevant to computational linguistics algorithms such as translation, information retrieval, question answering, etc. Semantic changes can potentially impact the quality of the outcomes of these algorithms. Therefore, it is important to understand and characterize these changes formally. The study of this impact is a recent problem that has attracted the attention of the computational linguistics community. Several approaches propose methods to detect semantic changes with good precision, but more effort is needed to charact
    
[^3]: GreatSplicing: 一个语义丰富的拼接数据集

    GreatSplicing: A Semantically Rich Splicing Dataset. (arXiv:2310.10070v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2310.10070](http://arxiv.org/abs/2310.10070)

    本文提出了一个语义丰富的拼接数据集GreatSplicing，通过包括大量不同语义类别的拼接区域，训练的模型在拼接痕迹检测上表现出较低的误识率和更好的跨数据集检测能力。

    

    在现有的拼接伪造数据集中，拼接区域的语义变化不足导致训练的检测模型对语义特征的过拟合。同时，由于缺乏合理的数据集，不同的检测方法在实验设置上无法达成一致。为了解决这些紧迫的问题，本文提出了GreatSplicing，一个手动创建的具有大量和高质量的拼接数据集。GreatSplicing包括5000张拼接图像，并涵盖了335个不同的语义类别的拼接区域，让神经网络更好地抓住拼接痕迹。大量实验证明，使用GreatSplicing训练的模型相较于现有数据集表现出较低的误识率和更好的跨数据集检测能力。此外，GreatSplicing可供所有研究目的使用，并可从www.greatsplicing.net下载。

    In existing splicing forgery datasets, the insufficient semantic varieties of spliced regions cause a problem that trained detection models overfit semantic features rather than splicing traces. Meanwhile, because of the absence of a reasonable dataset, different detection methods proposed cannot reach a consensus on experimental settings. To address these urgent issues, GreatSplicing, a manually created splicing dataset with a considerable amount and high quality, is proposed in this paper. GreatSplicing comprises 5,000 spliced images and covers spliced regions with 335 distinct semantic categories, allowing neural networks to grasp splicing traces better. Extensive experiments demonstrate that models trained on GreatSplicing exhibit minimal misidentification rates and superior cross-dataset detection capabilities compared to existing datasets. Furthermore, GreatSplicing is available for all research purposes and can be downloaded from www.greatsplicing.net.
    
[^4]: 通过硬件-算法-通信协同设计实现高效可信的人工智能

    Towards Efficient and Trustworthy AI Through Hardware-Algorithm-Communication Co-Design. (arXiv:2309.15942v1 [cs.AI])

    [http://arxiv.org/abs/2309.15942](http://arxiv.org/abs/2309.15942)

    通过硬件-算法-通信协同设计，本论文提出了一种实现高效可信的人工智能的研究方法，即通过结合物理洞见、高效信息处理原则、最优不确定度量结果和分布式处理准则，来提高神经网络算法的效率和可信度。

    

    基于神经网络的人工智能算法已经被设计了几十年，目标是最大化某种准确性度量。这导致了两个不希望出现的结果。首先，以计算和内存需求为衡量标准，模型复杂性呈指数增长。第二，最新的人工智能模型在提供可信度量方面很难实现，可能出现"幻觉"问题，从而阻碍了其在敏感应用的决策制定中的应用。为了实现高效可信的人工智能，本文强调硬件和软件设计交叉的研究方向，将物理洞见与计算基础、神经科学有关的高效信息处理原则、信息论中关于最优不确定度量的结果以及通信理论中关于分布式处理的准则相结合。总体而言，本文提倡新的硬件-算法-通信协同设计方法，以实现高效可信的人工智能。

    Artificial intelligence (AI) algorithms based on neural networks have been designed for decades with the goal of maximising some measure of accuracy. This has led to two undesired effects. First, model complexity has risen exponentially when measured in terms of computation and memory requirements. Second, state-of-the-art AI models are largely incapable of providing trustworthy measures of their uncertainty, possibly `hallucinating' their answers and discouraging their adoption for decision-making in sensitive applications.  With the goal of realising efficient and trustworthy AI, in this paper we highlight research directions at the intersection of hardware and software design that integrate physical insights into computational substrates, neuroscientific principles concerning efficient information processing, information-theoretic results on optimal uncertainty quantification, and communication-theoretic guidelines for distributed processing. Overall, the paper advocates for novel d
    

