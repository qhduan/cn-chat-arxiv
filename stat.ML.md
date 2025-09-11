# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [PQMass: Probabilistic Assessment of the Quality of Generative Models using Probability Mass Estimation](https://arxiv.org/abs/2402.04355) | PQMass是一种使用概率质量估计来评估生成模型质量的全面方法，能够直接处理高维数据，不依赖于假设或训练其他模型。 |
| [^2] | [Calibrating Transformers via Sparse Gaussian Processes.](http://arxiv.org/abs/2303.02444) | 提出了一种通过Sparse Gaussian Process attention (SGPA)来校准Transformer模型不确定性的方法。在文本、图像和图形的预测任务中，SGPA-based Transformers在预测准确性上表现出竞争力，并显著改善了内分布校准和外分布的鲁棒性和检测能力。 |

# 详细

[^1]: PQMass: 使用概率质量估计的生成模型质量的概率评估

    PQMass: Probabilistic Assessment of the Quality of Generative Models using Probability Mass Estimation

    [https://arxiv.org/abs/2402.04355](https://arxiv.org/abs/2402.04355)

    PQMass是一种使用概率质量估计来评估生成模型质量的全面方法，能够直接处理高维数据，不依赖于假设或训练其他模型。

    

    我们提出了一种全面的基于样本的方法来评估生成模型的质量。所提出的方法能够估计两个样本集合来自同一分布的概率，为评估单个生成模型的性能或比较在同一数据集上训练的多个竞争模型提供了一个统计上严格的方法。该比较可以通过将空间划分为非重叠的区域并比较每个区域中的数据样本数量来进行。该方法仅需要生成模型和测试数据的样本。它能够直接处理高维数据，无需降维。显著的是，该方法不依赖于关于真实分布密度的假设，并且不依赖于训练或拟合任何辅助模型。相反，它着重于近似计算密度的积分（概率质量）。

    We propose a comprehensive sample-based method for assessing the quality of generative models. The proposed approach enables the estimation of the probability that two sets of samples are drawn from the same distribution, providing a statistically rigorous method for assessing the performance of a single generative model or the comparison of multiple competing models trained on the same dataset. This comparison can be conducted by dividing the space into non-overlapping regions and comparing the number of data samples in each region. The method only requires samples from the generative model and the test data. It is capable of functioning directly on high-dimensional data, obviating the need for dimensionality reduction. Significantly, the proposed method does not depend on assumptions regarding the density of the true distribution, and it does not rely on training or fitting any auxiliary models. Instead, it focuses on approximating the integral of the density (probability mass) acros
    
[^2]: 通过稀疏高斯过程校准Transformer

    Calibrating Transformers via Sparse Gaussian Processes. (arXiv:2303.02444v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2303.02444](http://arxiv.org/abs/2303.02444)

    提出了一种通过Sparse Gaussian Process attention (SGPA)来校准Transformer模型不确定性的方法。在文本、图像和图形的预测任务中，SGPA-based Transformers在预测准确性上表现出竞争力，并显著改善了内分布校准和外分布的鲁棒性和检测能力。

    

    Transformer模型在自然语言处理、语音识别和计算机视觉等广泛应用中取得了巨大成功。将Transformer的成功扩展到安全关键领域需要准确估计的不确定性，这方面的研究较少。为了解决这个问题，我们提出了稀疏高斯过程注意力（SGPA），它直接在Transformer的多头自注意力块（MHA）的输出空间中进行贝叶斯推断，以校准其不确定性。它用一个有效的对称核替代了缩放点积操作，并使用稀疏高斯过程（SGP）技术来近似MHA输出的后验过程。经验上，在文本、图像和图形的一系列预测任务中，基于SGPA的Transformer模型实现了有竞争力的预测准确性，同时显著改善了内分布校准和外分布的鲁棒性和检测能力。

    Transformer models have achieved profound success in prediction tasks in a wide range of applications in natural language processing, speech recognition and computer vision. Extending Transformer's success to safety-critical domains requires calibrated uncertainty estimation which remains under-explored. To address this, we propose Sparse Gaussian Process attention (SGPA), which performs Bayesian inference directly in the output space of multi-head attention blocks (MHAs) in transformer to calibrate its uncertainty. It replaces the scaled dot-product operation with a valid symmetric kernel and uses sparse Gaussian processes (SGP) techniques to approximate the posterior processes of MHA outputs. Empirically, on a suite of prediction tasks on text, images and graphs, SGPA-based Transformers achieve competitive predictive accuracy, while noticeably improving both in-distribution calibration and out-of-distribution robustness and detection.
    

