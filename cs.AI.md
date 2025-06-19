# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [An Effective Incorporating Heterogeneous Knowledge Curriculum Learning for Sequence Labeling](https://arxiv.org/abs/2402.13534) | 提出了一个专为序列标注任务设计的两阶段课程学习（TCL）框架，逐渐引入数据实例从简单到困难，旨在提高性能和训练速度，并且对六个中文分词和词性标注数据集进行了广泛实验，证明了模型的有效性。 |
| [^2] | [Dynamic ASR Pathways: An Adaptive Masking Approach Towards Efficient Pruning of A Multilingual ASR Model.](http://arxiv.org/abs/2309.13018) | 本研究提出了一种自适应掩蔽方法，用于高效地压缩多语种ASR模型。该方法通过动态适应子网络结构，能够在减少性能损失的情况下得到稀疏的单语种模型或稀疏的多语种模型。实验证明，与现有的修剪方法相比，该方法在针对稀疏的单语种模型时表现更好，并且减少了对特定语言进行修剪的需求。 |
| [^3] | ["Generate" the Future of Work through AI: Empirical Evidence from Online Labor Markets.](http://arxiv.org/abs/2308.05201) | 这项研究通过利用ChatGPT作为外生冲击，揭示了其对在线劳动市场的影响。结果显示，直接接触ChatGPT的任务和自由职业者的交易量显著下降，但适应新技术并提供增强人工智能的服务的自由职业者仍能获得利益。 |

# 详细

[^1]: 一种有效融合异构知识的课程学习方法用于序列标注

    An Effective Incorporating Heterogeneous Knowledge Curriculum Learning for Sequence Labeling

    [https://arxiv.org/abs/2402.13534](https://arxiv.org/abs/2402.13534)

    提出了一个专为序列标注任务设计的两阶段课程学习（TCL）框架，逐渐引入数据实例从简单到困难，旨在提高性能和训练速度，并且对六个中文分词和词性标注数据集进行了广泛实验，证明了模型的有效性。

    

    序列标注模型常常受益于整合外部知识。然而，这一做法引入了数据异构性，并通过额外模块使模型变得复杂，导致训练高性能模型的成本增加。为了应对这一挑战，我们提出了一个专为序列标注任务设计的两阶段课程学习（TCL）框架。TCL框架通过逐渐引入从简单到困难的数据实例来增强训练，旨在提高性能和训练速度。此外，我们还探索了用于评估序列标注任务难度级别的不同指标。通过在六个中文分词（CWS）和词性标注（POS）数据集上进行大量实验，我们展示了我们的模型在提高序列标注模型性能方面的有效性。此外，我们的分析表明TCL加速了训练并缓解了

    arXiv:2402.13534v1 Announce Type: cross  Abstract: Sequence labeling models often benefit from incorporating external knowledge. However, this practice introduces data heterogeneity and complicates the model with additional modules, leading to increased expenses for training a high-performing model. To address this challenge, we propose a two-stage curriculum learning (TCL) framework specifically designed for sequence labeling tasks. The TCL framework enhances training by gradually introducing data instances from easy to hard, aiming to improve both performance and training speed. Furthermore, we explore different metrics for assessing the difficulty levels of sequence labeling tasks. Through extensive experimentation on six Chinese word segmentation (CWS) and Part-of-speech tagging (POS) datasets, we demonstrate the effectiveness of our model in enhancing the performance of sequence labeling models. Additionally, our analysis indicates that TCL accelerates training and alleviates the 
    
[^2]: 动态ASR路径：一种自适应掩蔽方法用于压缩多语种ASR模型的高效修剪

    Dynamic ASR Pathways: An Adaptive Masking Approach Towards Efficient Pruning of A Multilingual ASR Model. (arXiv:2309.13018v1 [eess.AS])

    [http://arxiv.org/abs/2309.13018](http://arxiv.org/abs/2309.13018)

    本研究提出了一种自适应掩蔽方法，用于高效地压缩多语种ASR模型。该方法通过动态适应子网络结构，能够在减少性能损失的情况下得到稀疏的单语种模型或稀疏的多语种模型。实验证明，与现有的修剪方法相比，该方法在针对稀疏的单语种模型时表现更好，并且减少了对特定语言进行修剪的需求。

    

    神经网络修剪是一种有效的方法，可以在性能损失最小的情况下压缩多语种自动语音识别（ASR）模型。然而，这需要对每种语言运行多轮修剪和重新训练。在这项工作中，我们提出了一种自适应掩蔽方法，以两种场景高效地修剪多语种ASR模型，分别得到了稀疏的单语种模型或稀疏的多语种模型（称为动态ASR路径）。我们的方法动态地适应子网络，避免对固定的子网络结构进行过早决策。我们证明了我们的方法在针对稀疏的单语种模型时优于现有的修剪方法。此外，我们还说明了动态ASR路径通过自不同的子网络初始化进行调整，共同发现和训练更好的单一多语种模型的子网络（路径），从而减少了对特定语言进行修剪的需求。

    Neural network pruning offers an effective method for compressing a multilingual automatic speech recognition (ASR) model with minimal performance loss. However, it entails several rounds of pruning and re-training needed to be run for each language. In this work, we propose the use of an adaptive masking approach in two scenarios for pruning a multilingual ASR model efficiently, each resulting in sparse monolingual models or a sparse multilingual model (named as Dynamic ASR Pathways). Our approach dynamically adapts the sub-network, avoiding premature decisions about a fixed sub-network structure. We show that our approach outperforms existing pruning methods when targeting sparse monolingual models. Further, we illustrate that Dynamic ASR Pathways jointly discovers and trains better sub-networks (pathways) of a single multilingual model by adapting from different sub-network initializations, thereby reducing the need for language-specific pruning.
    
[^3]: 通过人工智能"生成"工作：在线劳动市场的经验证据

    "Generate" the Future of Work through AI: Empirical Evidence from Online Labor Markets. (arXiv:2308.05201v1 [cs.AI])

    [http://arxiv.org/abs/2308.05201](http://arxiv.org/abs/2308.05201)

    这项研究通过利用ChatGPT作为外生冲击，揭示了其对在线劳动市场的影响。结果显示，直接接触ChatGPT的任务和自由职业者的交易量显著下降，但适应新技术并提供增强人工智能的服务的自由职业者仍能获得利益。

    

    随着通用生成式人工智能的出现，对其对劳动市场的影响的兴趣不断增加。为了填补现有的实证空白，我们将ChatGPT的推出解释为一种外生冲击，并采用差异法来量化其对在线劳动市场中与文本相关的工作和自由职业者的影响。我们的结果显示，直接接触ChatGPT的任务和自由职业者的交易量显著下降。此外，这种下降在相对较高的过去交易量或较低的质量标准下尤为显著。然而，并非所有服务提供商都普遍经历了负面影响。随后的分析表明，在这个转型期间，能够适应新进展并提供增强人工智能技术的服务的自由职业者可以获得可观的利益。因此，虽然ChatGPT的出现有可能替代人力劳动

    With the advent of general-purpose Generative AI, the interest in discerning its impact on the labor market escalates. In an attempt to bridge the extant empirical void, we interpret the launch of ChatGPT as an exogenous shock, and implement a Difference-in-Differences (DID) approach to quantify its influence on text-related jobs and freelancers within an online labor marketplace. Our results reveal a significant decrease in transaction volume for gigs and freelancers directly exposed to ChatGPT. Additionally, this decline is particularly marked in units of relatively higher past transaction volume or lower quality standards. Yet, the negative effect is not universally experienced among service providers. Subsequent analyses illustrate that freelancers proficiently adapting to novel advancements and offering services that augment AI technologies can yield substantial benefits amidst this transformative period. Consequently, even though the advent of ChatGPT could conceivably substitute
    

