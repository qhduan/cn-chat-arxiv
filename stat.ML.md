# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Metric Learning from Limited Pairwise Preference Comparisons](https://arxiv.org/abs/2403.19629) | 在有限成对偏好比较下研究度量学习，表明虽然无法学习单个理想项目，但当比较对象表现出低维结构时，每个用户可以帮助学习限制在低维子空间中的度量。 |
| [^2] | [Semi-Supervised Learning for Deep Causal Generative Models](https://arxiv.org/abs/2403.18717) | 首次开发了一种利用变量之间因果关系的半监督深度因果生成模型，以最大限度地利用所有可用数据。 |
| [^3] | [OTSeg: Multi-prompt Sinkhorn Attention for Zero-Shot Semantic Segmentation](https://arxiv.org/abs/2403.14183) | 通过引入OTSeg中的Multi-Prompts Sinkhorn Attention机制，能够更好地利用多个文本提示来匹配相关像素嵌入，从而提升零样本语义分割性能。 |
| [^4] | [Heavy-Tailed Class Imbalance and Why Adam Outperforms Gradient Descent on Language Models](https://arxiv.org/abs/2402.19449) | 研究发现语言模型中的重尾类别不平衡问题导致了优化动态上的困难，Adam和基于符号的方法在这种情况下优于梯度下降。 |
| [^5] | [Loss Shaping Constraints for Long-Term Time Series Forecasting](https://arxiv.org/abs/2402.09373) | 该论文提出了一种用于长期时间序列预测的受限学习方法，通过在每个时间步骤上设置损失上限来寻找最佳模型，以解决平均性能优化导致特定时间步骤上误差过大的问题。 |
| [^6] | [Tree Ensembles for Contextual Bandits](https://arxiv.org/abs/2402.06963) | 本论文提出了一种基于树集成的情境多臂老虎机新框架，通过整合两种广泛使用的老虎机方法，在标准和组合设置中实现了优于基于神经网络的方法的性能，在减少后悔和计算时间方面表现出更出色的性能。 |
| [^7] | [Calibrating dimension reduction hyperparameters in the presence of noise](https://arxiv.org/abs/2312.02946) | 本文提出了一个框架，用于在噪声存在的情况下校准降维超参数，探索了困惑度和维度数量的作用。 |
| [^8] | [Compressed Sensing: A Discrete Optimization Approach.](http://arxiv.org/abs/2306.04647) | 本文中提出了一种离散优化方法来解决压缩感知问题，该方法在二次锥松弛下，可以找到最稀疏的向量，得到了可靠的最优解。 |

# 详细

[^1]: 有限成对偏好比较下的度量学习

    Metric Learning from Limited Pairwise Preference Comparisons

    [https://arxiv.org/abs/2403.19629](https://arxiv.org/abs/2403.19629)

    在有限成对偏好比较下研究度量学习，表明虽然无法学习单个理想项目，但当比较对象表现出低维结构时，每个用户可以帮助学习限制在低维子空间中的度量。

    

    我们研究了在理想点模型下的偏好比较中的度量学习，其中用户如果一个项目比其潜在理想项目更接近，则更喜欢该项目。这些项目嵌入到具有未知马氏距离的$\mathbb{R}^d$中，该距离在用户间共享。尽管最近的工作表明，通过每个用户$\mathcal{O}(d)$个成对比较可以同时恢复度量和理想项目，但在实践中，我们经常有$o(d)$的有限比较预算。我们研究了即使已知学习单个理想项目现在不再可能，度量是否仍然可以恢复。我们发现一般来说，$o(d)$比较不会揭示有关度量的信息，即使用户数量无限。然而，当比较的项目表现出低维结构时，每个用户都可以有助于学习限制在低维子空间中的度量，这样度量就可以被恢复。

    arXiv:2403.19629v1 Announce Type: new  Abstract: We study metric learning from preference comparisons under the ideal point model, in which a user prefers an item over another if it is closer to their latent ideal item. These items are embedded into $\mathbb{R}^d$ equipped with an unknown Mahalanobis distance shared across users. While recent work shows that it is possible to simultaneously recover the metric and ideal items given $\mathcal{O}(d)$ pairwise comparisons per user, in practice we often have a limited budget of $o(d)$ comparisons. We study whether the metric can still be recovered, even though it is known that learning individual ideal items is now no longer possible. We show that in general, $o(d)$ comparisons reveals no information about the metric, even with infinitely many users. However, when comparisons are made over items that exhibit low-dimensional structure, each user can contribute to learning the metric restricted to a low-dimensional subspace so that the metric
    
[^2]: 深度因果生成模型的半监督学习

    Semi-Supervised Learning for Deep Causal Generative Models

    [https://arxiv.org/abs/2403.18717](https://arxiv.org/abs/2403.18717)

    首次开发了一种利用变量之间因果关系的半监督深度因果生成模型，以最大限度地利用所有可用数据。

    

    开发能够回答“如果$y$变为$z$，$x$会如何变化？”这类问题的模型对于推动医学图像分析至关重要。然而，训练能够解决这类反事实问题的因果生成模型目前要求所有相关变量均已被观察到，并且相应的标签在训练数据中可用。我们首次开发了一种利用变量之间因果关系的半监督深度因果生成模型，以最大限度地利用所有可用数据。

    arXiv:2403.18717v1 Announce Type: cross  Abstract: Developing models that can answer questions of the form "How would $x$ change if $y$ had been $z$?" is fundamental for advancing medical image analysis. Training causal generative models that address such counterfactual questions, though, currently requires that all relevant variables have been observed and that corresponding labels are available in training data. However, clinical data may not have complete records for all patients and state of the art causal generative models are unable to take full advantage of this. We thus develop, for the first time, a semi-supervised deep causal generative model that exploits the causal relationships between variables to maximise the use of all available data. We explore this in the setting where each sample is either fully labelled or fully unlabelled, as well as the more clinically realistic case of having different labels missing for each sample. We leverage techniques from causal inference t
    
[^3]: OTSeg：多提示Sinkhorn注意力用于零样本语义分割

    OTSeg: Multi-prompt Sinkhorn Attention for Zero-Shot Semantic Segmentation

    [https://arxiv.org/abs/2403.14183](https://arxiv.org/abs/2403.14183)

    通过引入OTSeg中的Multi-Prompts Sinkhorn Attention机制，能够更好地利用多个文本提示来匹配相关像素嵌入，从而提升零样本语义分割性能。

    

    CLIP的最新成功证明了通过将多模态知识转移到像素级分类来进行零样本语义分割的有希望的结果。然而，在现有方法中，利用预先训练的CLIP知识来紧密对齐文本嵌入和像素嵌入仍然存在局限性。为了解决这个问题，我们提出了OTSeg，这是一种新颖的多模态注意力机制，旨在增强多个文本提示匹配相关像素嵌入的潜力。我们首先提出了基于最优输运（OT）算法的多提示Sinkhorn（MPS），这使得多个文本提示可以有选择地关注图像像素内的各种语义特征。此外，受到Sinkformers在单模态设置中的成功启发，我们引入了MPS的扩展，称为多提示Sinkhorn注意力（MPSA），它有效地取代了Transformer框架中多模态设置中的交叉注意力机制。

    arXiv:2403.14183v1 Announce Type: cross  Abstract: The recent success of CLIP has demonstrated promising results in zero-shot semantic segmentation by transferring muiltimodal knowledge to pixel-level classification. However, leveraging pre-trained CLIP knowledge to closely align text embeddings with pixel embeddings still has limitations in existing approaches. To address this issue, we propose OTSeg, a novel multimodal attention mechanism aimed at enhancing the potential of multiple text prompts for matching associated pixel embeddings. We first propose Multi-Prompts Sinkhorn (MPS) based on the Optimal Transport (OT) algorithm, which leads multiple text prompts to selectively focus on various semantic features within image pixels. Moreover, inspired by the success of Sinkformers in unimodal settings, we introduce the extension of MPS, called Multi-Prompts Sinkhorn Attention (MPSA), which effectively replaces cross-attention mechanisms within Transformer framework in multimodal settin
    
[^4]: Heavy-Tailed Class Imbalance and Why Adam Outperforms Gradient Descent on Language Models

    Heavy-Tailed Class Imbalance and Why Adam Outperforms Gradient Descent on Language Models

    [https://arxiv.org/abs/2402.19449](https://arxiv.org/abs/2402.19449)

    研究发现语言模型中的重尾类别不平衡问题导致了优化动态上的困难，Adam和基于符号的方法在这种情况下优于梯度下降。

    

    本文研究了在语言建模任务中存在的重尾类别不平衡问题，以及为什么Adam在优化大型语言模型时的表现优于梯度下降方法。我们发现，由于语言建模任务中存在的重尾类别不平衡，使用梯度下降时，与不常见单词相关的损失下降速度比与常见单词相关的损失下降速度慢。由于大多数样本来自相对不常见的单词，平均损失值在梯度下降时下降速度较慢。相比之下，Adam和基于符号的方法却不受此问题影响，并改善了所有类别的预测性能。我们在不同架构和数据类型上进行了实证研究，证明了这种行为确实是由类别不平衡引起的。

    arXiv:2402.19449v1 Announce Type: cross  Abstract: Adam has been shown to outperform gradient descent in optimizing large language transformers empirically, and by a larger margin than on other tasks, but it is unclear why this happens. We show that the heavy-tailed class imbalance found in language modeling tasks leads to difficulties in the optimization dynamics. When training with gradient descent, the loss associated with infrequent words decreases slower than the loss associated with frequent ones. As most samples come from relatively infrequent words, the average loss decreases slowly with gradient descent. On the other hand, Adam and sign-based methods do not suffer from this problem and improve predictions on all classes. To establish that this behavior is indeed caused by class imbalance, we show empirically that it persist through different architectures and data types, on language transformers, vision CNNs, and linear models. We further study this phenomenon on a linear clas
    
[^5]: 长期时间序列预测的损失塑造约束

    Loss Shaping Constraints for Long-Term Time Series Forecasting

    [https://arxiv.org/abs/2402.09373](https://arxiv.org/abs/2402.09373)

    该论文提出了一种用于长期时间序列预测的受限学习方法，通过在每个时间步骤上设置损失上限来寻找最佳模型，以解决平均性能优化导致特定时间步骤上误差过大的问题。

    

    许多时间序列预测应用程序需要预测多个步骤。尽管在这个主题上有大量的文献，但经典和最近的基于深度学习的方法主要集中在最小化预测窗口上的性能平均值。我们观察到，这可能导致在预测步骤之间存在不同的错误分布，尤其是对于在常见预测基准上训练的最近的变换器架构。也就是说，平均性能优化可能导致特定时间步骤上的错误过大。在这项工作中，我们提出了一种长期时间序列预测的受限学习方法，旨在找到在平均性能上最好的模型，并且在每个时间步骤上保持用户定义的损失上限。我们称这种方法为损失塑造约束，因为它对每个时间步骤的损失施加约束，并利用最近的对偶性结果展示了...

    arXiv:2402.09373v1 Announce Type: new Abstract: Several applications in time series forecasting require predicting multiple steps ahead. Despite the vast amount of literature in the topic, both classical and recent deep learning based approaches have mostly focused on minimising performance averaged over the predicted window. We observe that this can lead to disparate distributions of errors across forecasting steps, especially for recent transformer architectures trained on popular forecasting benchmarks. That is, optimising performance on average can lead to undesirably large errors at specific time-steps. In this work, we present a Constrained Learning approach for long-term time series forecasting that aims to find the best model in terms of average performance that respects a user-defined upper bound on the loss at each time-step. We call our approach loss shaping constraints because it imposes constraints on the loss at each time step, and leverage recent duality results to show 
    
[^6]: 基于树集成的情境多臂老虎机

    Tree Ensembles for Contextual Bandits

    [https://arxiv.org/abs/2402.06963](https://arxiv.org/abs/2402.06963)

    本论文提出了一种基于树集成的情境多臂老虎机新框架，通过整合两种广泛使用的老虎机方法，在标准和组合设置中实现了优于基于神经网络的方法的性能，在减少后悔和计算时间方面表现出更出色的性能。

    

    我们提出了一个基于树集成的情境多臂老虎机的新框架。我们的框架将两种广泛使用的老虎机方法，上信心界和汤普森抽样，整合到标准和组合设置中。通过使用流行的树集成方法XGBoost进行多次实验研究，我们展示了我们框架的有效性。当应用于基准数据集和道路网络导航的真实世界应用时，与基于神经网络的最先进方法相比，我们的方法在减少后悔和计算时间方面表现出更好的性能。

    We propose a novel framework for contextual multi-armed bandits based on tree ensembles. Our framework integrates two widely used bandit methods, Upper Confidence Bound and Thompson Sampling, for both standard and combinatorial settings. We demonstrate the effectiveness of our framework via several experimental studies, employing XGBoost, a popular tree ensemble method. Compared to state-of-the-art methods based on neural networks, our methods exhibit superior performance in terms of both regret minimization and computational runtime, when applied to benchmark datasets and the real-world application of navigation over road networks.
    
[^7]: 在噪声存在的情况下校准降维超参数

    Calibrating dimension reduction hyperparameters in the presence of noise

    [https://arxiv.org/abs/2312.02946](https://arxiv.org/abs/2312.02946)

    本文提出了一个框架，用于在噪声存在的情况下校准降维超参数，探索了困惑度和维度数量的作用。

    

    降维工具的目标是构建高维数据的低维表示。这些工具被用于噪声降低、可视化和降低计算成本等各种原因。然而，在降维文献中几乎没有讨论过的一个基本问题是过拟合，而在其他建模问题中这个问题已经被广泛讨论。如果我们将数据解释为信号和噪声的组合，先前的研究对降维技术的评判是其是否能够捕捉到数据的全部内容，即信号和噪声。在其他建模问题的背景下，我们会采用特征选择、交叉验证和正则化等技术来防止过拟合，但在进行降维时却没有采取类似的预防措施。本文提出了一个框架，用于在噪声存在的情况下建模降维问题，并利用该框架探索了困惑度和维度数量的作用。

    The goal of dimension reduction tools is to construct a low-dimensional representation of high-dimensional data. These tools are employed for a variety of reasons such as noise reduction, visualization, and to lower computational costs. However, there is a fundamental issue that is highly discussed in other modeling problems, but almost entirely ignored in the dimension reduction literature: overfitting. If we interpret data as a combination of signal and noise, prior works judge dimension reduction techniques on their ability to capture the entirety of the data, i.e. both the signal and the noise. In the context of other modeling problems, techniques such as feature-selection, cross-validation, and regularization are employed to combat overfitting, but no such precautions are taken when performing dimension reduction. In this paper, we present a framework that models dimension reduction problems in the presence of noise and use this framework to explore the role perplexity and number 
    
[^8]: 压缩感知：离散优化方法

    Compressed Sensing: A Discrete Optimization Approach. (arXiv:2306.04647v1 [eess.SP])

    [http://arxiv.org/abs/2306.04647](http://arxiv.org/abs/2306.04647)

    本文中提出了一种离散优化方法来解决压缩感知问题，该方法在二次锥松弛下，可以找到最稀疏的向量，得到了可靠的最优解。

    

    本文研究了压缩感知问题，即找到最稀疏的向量，该向量满足一组线性测量，同时达到一定的数值容限。压缩感知是统计学、运筹学和机器学习中的核心问题，应用于信号处理、数据压缩和图像重建等领域。我们引入了一个带有$\ell_2$正则化的压缩感知问题，将其作为混合整数二次锥规划来重新定义。我们推导出此问题的二次锥松弛，并展示了在正则化参数的温和限制下，得到的松弛等价于深入研究的基础追踪去噪问题。我们提出了一个半定松弛来加强二次锥松弛，开发了一种定制的分支定界算法，利用我们的二次锥松弛来解决压缩感知问题的实例，以确证的最优解。我们的数值结果表明，我们的方法产生的解决方案是精确的，并且优于其他方法。

    We study the Compressed Sensing (CS) problem, which is the problem of finding the most sparse vector that satisfies a set of linear measurements up to some numerical tolerance. CS is a central problem in Statistics, Operations Research and Machine Learning which arises in applications such as signal processing, data compression and image reconstruction. We introduce an $\ell_2$ regularized formulation of CS which we reformulate as a mixed integer second order cone program. We derive a second order cone relaxation of this problem and show that under mild conditions on the regularization parameter, the resulting relaxation is equivalent to the well studied basis pursuit denoising problem. We present a semidefinite relaxation that strengthens the second order cone relaxation and develop a custom branch-and-bound algorithm that leverages our second order cone relaxation to solve instances of CS to certifiable optimality. Our numerical results show that our approach produces solutions that 
    

