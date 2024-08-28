# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Explain then Rank: Scale Calibration of Neural Rankers Using Natural Language Explanations from Large Language Models](https://arxiv.org/abs/2402.12276) | 本研究探讨了大型语言模型（LLMs）利用提供与规模校准相关的查询和文档对的不确定性测量的潜力，以解决神经排序器的规模校准问题。 |
| [^2] | [From Variability to Stability: Advancing RecSys Benchmarking Practices](https://arxiv.org/abs/2402.09766) | 本论文提出了一种新的基准测试方法，通过使用多样化的开放数据集，并在多个度量指标上评估多种协同过滤算法，来研究数据集特征对算法性能的影响。这一方法填补了推荐系统算法比较中的不足之处，推进了评估实践。 |

# 详细

[^1]: 用大型语言模型的自然语言解释进行神经排序器的规模校准解释和排名

    Explain then Rank: Scale Calibration of Neural Rankers Using Natural Language Explanations from Large Language Models

    [https://arxiv.org/abs/2402.12276](https://arxiv.org/abs/2402.12276)

    本研究探讨了大型语言模型（LLMs）利用提供与规模校准相关的查询和文档对的不确定性测量的潜力，以解决神经排序器的规模校准问题。

    

    排名系统中的规模校准过程涉及调整排序器的输出，以使其与重要品质（如点击率或相关性）相对应，这对于反映现实价值以及提高系统的效果和可靠性至关重要。虽然已经研究了学习排序模型中的校准排序损失，但调整神经排序器的规模的特定问题，这些模型擅长处理文本信息，尚未得到充分研究。神经排序模型擅长处理文本数据，但将现有规模校准技术应用到这些模型会面临重大挑战，因为它们的复杂性和需要大量训练，往往导致次优结果。

    arXiv:2402.12276v1 Announce Type: new  Abstract: The process of scale calibration in ranking systems involves adjusting the outputs of rankers to correspond with significant qualities like click-through rates or relevance, crucial for mirroring real-world value and thereby boosting the system's effectiveness and reliability. Although there has been research on calibrated ranking losses within learning-to-rank models, the particular issue of adjusting the scale for neural rankers, which excel in handling textual information, has not been thoroughly examined. Neural ranking models are adept at processing text data, yet the application of existing scale calibration techniques to these models poses significant challenges due to their complexity and the intensive training they require, often resulting in suboptimal outcomes.   This study delves into the potential of large language models (LLMs) to provide uncertainty measurements for a query and document pair that correlate with the scale-c
    
[^2]: 从变动性到稳定性：推荐系统基准化实践的进展

    From Variability to Stability: Advancing RecSys Benchmarking Practices

    [https://arxiv.org/abs/2402.09766](https://arxiv.org/abs/2402.09766)

    本论文提出了一种新的基准测试方法，通过使用多样化的开放数据集，并在多个度量指标上评估多种协同过滤算法，来研究数据集特征对算法性能的影响。这一方法填补了推荐系统算法比较中的不足之处，推进了评估实践。

    

    在快速发展的推荐系统领域中，新的算法经常通过对一组有限的任意选择的数据集进行评估来声称自己具有最先进的性能。然而，由于数据集特征对算法性能有重大影响，这种方法可能无法全面反映它们的有效性。为了解决这个问题，本文引入了一种新的基准测试方法，以促进公平和稳健的推荐系统算法比较，从而推进评估实践。通过利用包括本文介绍的两个数据集在内的30个开放数据集，并在9个度量指标上评估11种协同过滤算法，我们对数据集特征对算法性能的影响进行了重要的研究。我们进一步研究了将多个数据集的结果聚合成一个统一排名的可行性。通过严格的实验分析，我们发现......

    arXiv:2402.09766v1 Announce Type: cross  Abstract: In the rapidly evolving domain of Recommender Systems (RecSys), new algorithms frequently claim state-of-the-art performance based on evaluations over a limited set of arbitrarily selected datasets. However, this approach may fail to holistically reflect their effectiveness due to the significant impact of dataset characteristics on algorithm performance. Addressing this deficiency, this paper introduces a novel benchmarking methodology to facilitate a fair and robust comparison of RecSys algorithms, thereby advancing evaluation practices. By utilizing a diverse set of $30$ open datasets, including two introduced in this work, and evaluating $11$ collaborative filtering algorithms across $9$ metrics, we critically examine the influence of dataset characteristics on algorithm performance. We further investigate the feasibility of aggregating outcomes from multiple datasets into a unified ranking. Through rigorous experimental analysis, 
    

