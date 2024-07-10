# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Sparse high-dimensional linear mixed modeling with a partitioned empirical Bayes ECM algorithm.](http://arxiv.org/abs/2310.12285) | 提出了一种高维线性混合模型的分区经验Bayes ECM算法，具有快速可扩展的计算能力和更高的灵活性。 |
| [^2] | [TSGM: A Flexible Framework for Generative Modeling of Synthetic Time Series.](http://arxiv.org/abs/2305.11567) | TSGM提供了一种生成合成时间序列数据的灵活框架，使研究人员能够快速实现自己的方法并在可共享的环境中进行比较，从而有助于生成大规模的合成时间序列数据集，以用于训练和验证各种机器学习模型。 |
| [^3] | [Calibrating Transformers via Sparse Gaussian Processes.](http://arxiv.org/abs/2303.02444) | 提出了一种通过Sparse Gaussian Process attention (SGPA)来校准Transformer模型不确定性的方法。在文本、图像和图形的预测任务中，SGPA-based Transformers在预测准确性上表现出竞争力，并显著改善了内分布校准和外分布的鲁棒性和检测能力。 |

# 详细

[^1]: 稀疏高维线性混合模型的分区经验Bayes ECM算法

    Sparse high-dimensional linear mixed modeling with a partitioned empirical Bayes ECM algorithm. (arXiv:2310.12285v1 [stat.ME])

    [http://arxiv.org/abs/2310.12285](http://arxiv.org/abs/2310.12285)

    提出了一种高维线性混合模型的分区经验Bayes ECM算法，具有快速可扩展的计算能力和更高的灵活性。

    

    高维纵向数据在各种科学研究中的应用日益增多。然而，对于高维线性混合模型(LMMs)，目前只有少数统计方法可用，因为大多数贝叶斯变量选择或罚函数方法是针对独立观测设计的。此外，目前少数可用的高维LMMs软件包存在可扩展性问题。本文提出了一种高效准确的高维LMMs贝叶斯框架。我们使用经验Bayes估计器的超参数来增加灵活性，并使用Expectation-Conditional-Minimization (ECM)算法来计算参数的最大后验概率(MAP)估计，从而实现高效的计算。这种方法的创新之处在于其分区和参数扩展，以及其快速和可扩展的计算。我们通过模拟研究中的固定效应和随机效应估计展示了线性混合模型结合分区经验Bayes ECM (LMM-PROBE)的效果。

    High-dimensional longitudinal data is increasingly used in a wide range of scientific studies. However, there are few statistical methods for high-dimensional linear mixed models (LMMs), as most Bayesian variable selection or penalization methods are designed for independent observations. Additionally, the few available software packages for high-dimensional LMMs suffer from scalability issues. This work presents an efficient and accurate Bayesian framework for high-dimensional LMMs. We use empirical Bayes estimators of hyperparameters for increased flexibility and an Expectation-Conditional-Minimization (ECM) algorithm for computationally efficient maximum a posteriori probability (MAP) estimation of parameters. The novelty of the approach lies in its partitioning and parameter expansion as well as its fast and scalable computation. We illustrate Linear Mixed Modeling with PaRtitiOned empirical Bayes ECM (LMM-PROBE) in simulation studies evaluating fixed and random effects estimation 
    
[^2]: TSGM：一种生成合成时间序列数据的灵活框架

    TSGM: A Flexible Framework for Generative Modeling of Synthetic Time Series. (arXiv:2305.11567v1 [cs.LG])

    [http://arxiv.org/abs/2305.11567](http://arxiv.org/abs/2305.11567)

    TSGM提供了一种生成合成时间序列数据的灵活框架，使研究人员能够快速实现自己的方法并在可共享的环境中进行比较，从而有助于生成大规模的合成时间序列数据集，以用于训练和验证各种机器学习模型。

    

    时间序列数据在各个领域中非常重要，对机器学习研究者也很有兴趣。然而，时间序列数据通常很少或高度敏感，这使得数据在研究者和工业组织之间的共享以及现有和新的数据密集型 ML 方法的应用受到限制。解决这一难题的可能方法是生成合成数据。在这项工作中，我们介绍了时间序列生成模型（TSGM），这是一种用于生成合成时间序列数据的开源框架。TSGM包括广泛的机器学习方法：生成模型、概率模型和基于模拟器的方法。该框架使用户能够从不同的角度评估生成的数据的质量：相似性、下游效果、预测一致性、多样性和隐私。该框架是可扩展的，这使得研究人员能够快速实现自己的方法并在可共享的环境中进行比较。TSGM将有助于生成大规模的合成时间序列数据集，这些数据集可以用于训练和验证各种机器学习模型。

    Temporally indexed data are essential in a wide range of fields and of interest to machine learning researchers. Time series data, however, are often scarce or highly sensitive, which precludes the sharing of data between researchers and industrial organizations and the application of existing and new data-intensive ML methods. A possible solution to this bottleneck is to generate synthetic data. In this work, we introduce Time Series Generative Modeling (TSGM), an open-source framework for the generative modeling of synthetic time series. TSGM includes a broad repertoire of machine learning methods: generative models, probabilistic, and simulator-based approaches. The framework enables users to evaluate the quality of the produced data from different angles: similarity, downstream effectiveness, predictive consistency, diversity, and privacy. The framework is extensible, which allows researchers to rapidly implement their own methods and compare them in a shareable environment. TSGM w
    
[^3]: 通过稀疏高斯过程校准Transformer

    Calibrating Transformers via Sparse Gaussian Processes. (arXiv:2303.02444v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2303.02444](http://arxiv.org/abs/2303.02444)

    提出了一种通过Sparse Gaussian Process attention (SGPA)来校准Transformer模型不确定性的方法。在文本、图像和图形的预测任务中，SGPA-based Transformers在预测准确性上表现出竞争力，并显著改善了内分布校准和外分布的鲁棒性和检测能力。

    

    Transformer模型在自然语言处理、语音识别和计算机视觉等广泛应用中取得了巨大成功。将Transformer的成功扩展到安全关键领域需要准确估计的不确定性，这方面的研究较少。为了解决这个问题，我们提出了稀疏高斯过程注意力（SGPA），它直接在Transformer的多头自注意力块（MHA）的输出空间中进行贝叶斯推断，以校准其不确定性。它用一个有效的对称核替代了缩放点积操作，并使用稀疏高斯过程（SGP）技术来近似MHA输出的后验过程。经验上，在文本、图像和图形的一系列预测任务中，基于SGPA的Transformer模型实现了有竞争力的预测准确性，同时显著改善了内分布校准和外分布的鲁棒性和检测能力。

    Transformer models have achieved profound success in prediction tasks in a wide range of applications in natural language processing, speech recognition and computer vision. Extending Transformer's success to safety-critical domains requires calibrated uncertainty estimation which remains under-explored. To address this, we propose Sparse Gaussian Process attention (SGPA), which performs Bayesian inference directly in the output space of multi-head attention blocks (MHAs) in transformer to calibrate its uncertainty. It replaces the scaled dot-product operation with a valid symmetric kernel and uses sparse Gaussian processes (SGP) techniques to approximate the posterior processes of MHA outputs. Empirically, on a suite of prediction tasks on text, images and graphs, SGPA-based Transformers achieve competitive predictive accuracy, while noticeably improving both in-distribution calibration and out-of-distribution robustness and detection.
    

