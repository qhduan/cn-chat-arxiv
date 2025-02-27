# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Sampling-based Framework for Hypothesis Testing on Large Attributed Graphs](https://arxiv.org/abs/2403.13286) | 本论文提出了一个基于抽样的假设检验框架，能够在大属性图中处理节点、边和路径假设，通过提出路径假设感知采样器 PHASE 以及 PHASEopt，实现了准确且高效的抽样，实验证明了其在假设检验上的优势。 |
| [^2] | [Fast, Scale-Adaptive, and Uncertainty-Aware Downscaling of Earth System Model Fields with Generative Foundation Models](https://arxiv.org/abs/2403.02774) | 通过学习一致性模型，在不需要重新训练的情况下高效、准确地降尺度任意地球系统模型模拟，并产生概率性降尺度场。 |
| [^3] | [Parameter-Efficient Tuning of Large Convolutional Models](https://arxiv.org/abs/2403.00269) | 通过引入滤波器子空间和滤波器原子的概念，本研究提出了一种在微调大型卷积模型时仅调整少量参数来提取任务特定表示的方法。 |
| [^4] | [Not All Weights Are Created Equal: Enhancing Energy Efficiency in On-Device Streaming Speech Recognition](https://arxiv.org/abs/2402.13076) | 权重参数在设备上的流式语音识别模型中的功耗影响有所不同，作者提出了基于权重参数敏感性的有针对性压缩方法，将能源使用减少高达47%而维持模型准确性 |
| [^5] | [Permute-and-Flip: An optimally robust and watermarkable decoder for LLMs](https://arxiv.org/abs/2402.05864) | 提出了一种名为Permute-and-Flip（PF）解码器，其具有最佳的鲁棒性和质量-鲁棒性的 tradeoff，且比采样方法更好。还设计了一种针对PF解码器的水印方案，能够保持样本的分布不变，并实现任意低的假阳性率和高的召回率。实验证明PF解码器在困惑度方面明显优于朴素采样，为LLM解码提供了一种有希望的新方法。 |
| [^6] | [Large Language Model Agent for Hyper-Parameter Optimization](https://arxiv.org/abs/2402.01881) | 基于大规模语言模型的AgentHPO技术通过自动化超参数优化，在机器学习任务中大大减少了试验次数，简化了设置过程，提升了解释性和用户信任。 |
| [^7] | [SMOOTHIE: A Theory of Hyper-parameter Optimization for Software Analytics.](http://arxiv.org/abs/2401.09622) | SMOOTHIE是一种通过考虑损失函数的“光滑度”来引导超参数优化的新型方法，在软件分析中应用可以带来显著的性能改进。 |
| [^8] | [Mathematical Introduction to Deep Learning: Methods, Implementations, and Theory.](http://arxiv.org/abs/2310.20360) | 本书提供了对深度学习算法的数学介绍，包括不同的神经网络架构和优化算法，并涵盖了深度学习算法的理论方面。此外，还介绍了深度学习逼近偏微分方程的方法。希望对学生和科学家们有所帮助。 |
| [^9] | [DCSI -- An improved measure of cluster separability based on separation and connectedness.](http://arxiv.org/abs/2310.12806) | 这篇论文提出了一种改进的聚类可分离性度量方法，旨在量化类间分离和类内连通性，对于密度聚类具有较好的性能表现。 |
| [^10] | [LGL-BCI: A Lightweight Geometric Learning Framework for Motor Imagery-Based Brain-Computer Interfaces.](http://arxiv.org/abs/2310.08051) | LGL-BCI是一种轻量级几何学习框架，通过处理EEG数据在非欧几里德度量空间中捕捉运动想象任务的空间相关性，并通过特征分解算法进行EEG通道选择以提高推断速度。实验证明LGL-BCI相比现有解决方案具有更高的准确性和效率。 |
| [^11] | [AdaRec: Adaptive Sequential Recommendation for Reinforcing Long-term User Engagement.](http://arxiv.org/abs/2310.03984) | AdaRec是一种自适应顺序推荐算法，通过引入基于距离的表示损失来提取潜在信息，以适应大规模在线推荐系统中用户行为模式的变化。 |
| [^12] | [Mixture Encoder Supporting Continuous Speech Separation for Meeting Recognition.](http://arxiv.org/abs/2309.08454) | 本研究将混合编码器方法从两个说话人情况扩展到了更自然的会议环境，包括任意数量的说话人和动态重叠。实验证明，该方法在LibriCSS数据集上达到了最先进的性能，并凸显了混合编码器的优势。 |
| [^13] | [Implementing Quantum Generative Adversarial Network (qGAN) and QCBM in Finance.](http://arxiv.org/abs/2308.08448) | 这项研究讨论了在金融领域中应用量子机器学习的新研究方向，通过比较qGAN和QCBM等模型，展示了在金融领域中实现量子优势的潜力。 |
| [^14] | [Towards an AI Accountability Policy.](http://arxiv.org/abs/2307.13658) | 这份白皮书是对美国国家电信和信息管理局的“AI问责政策评论请求”的回应，提出了一组相互关联的AI问责政策建议。 |
| [^15] | [Diagrammatization: Rationalizing with diagrammatic AI explanations for abductive-deductive reasoning on hypotheses.](http://arxiv.org/abs/2302.01241) | 本文提出了一种图解化的方法，以支持可解释的人工智能，通过图解型和假设性推理，缩小可解释性差距。通过临床应用研究和建模研究，我们发现DiagramNet不仅能提供忠实的杂音形状解释，还具有较好的预测性能，而且图解型解释在临床相关的情况下更受推崇。 |

# 详细

[^1]: 基于抽样的大属性图假设检验框架

    A Sampling-based Framework for Hypothesis Testing on Large Attributed Graphs

    [https://arxiv.org/abs/2403.13286](https://arxiv.org/abs/2403.13286)

    本论文提出了一个基于抽样的假设检验框架，能够在大属性图中处理节点、边和路径假设，通过提出路径假设感知采样器 PHASE 以及 PHASEopt，实现了准确且高效的抽样，实验证明了其在假设检验上的优势。

    

    假设检验是一种用于从样本数据中得出关于总体的结论的统计方法，通常用表格表示。随着现实应用中图表示的普及，图中的假设检验变得越来越重要。本文对属性图中的节点、边和路径假设进行了形式化。我们开发了一个基于抽样的假设检验框架，可以容纳现有的假设不可知的图抽样方法。为了实现准确和高效的抽样，我们提出了一种路径假设感知采样器 PHASE，它是一种考虑假设中指定路径的 m-维随机游走。我们进一步优化了其时间效率并提出了 PHASEopt。对真实数据集的实验表明，我们的框架能够利用常见的图抽样方法进行假设检验，并且在准确性和时间效率方面假设感知抽样具有优势。

    arXiv:2403.13286v1 Announce Type: cross  Abstract: Hypothesis testing is a statistical method used to draw conclusions about populations from sample data, typically represented in tables. With the prevalence of graph representations in real-life applications, hypothesis testing in graphs is gaining importance. In this work, we formalize node, edge, and path hypotheses in attributed graphs. We develop a sampling-based hypothesis testing framework, which can accommodate existing hypothesis-agnostic graph sampling methods. To achieve accurate and efficient sampling, we then propose a Path-Hypothesis-Aware SamplEr, PHASE, an m- dimensional random walk that accounts for the paths specified in a hypothesis. We further optimize its time efficiency and propose PHASEopt. Experiments on real datasets demonstrate the ability of our framework to leverage common graph sampling methods for hypothesis testing, and the superiority of hypothesis-aware sampling in terms of accuracy and time efficiency.
    
[^2]: 快速、自适应尺度和具有不确定性意识的地球系统模型场降尺度与生成基础模型

    Fast, Scale-Adaptive, and Uncertainty-Aware Downscaling of Earth System Model Fields with Generative Foundation Models

    [https://arxiv.org/abs/2403.02774](https://arxiv.org/abs/2403.02774)

    通过学习一致性模型，在不需要重新训练的情况下高效、准确地降尺度任意地球系统模型模拟，并产生概率性降尺度场。

    

    精确和高分辨率的地球系统模型(ESM)模拟对于评估人为气候变化对生态和社会经济影响至关重要，但计算成本过高。最近的机器学习方法在ESM模拟的降尺度中表现出色，优于最先进的统计方法。然而，现有方法对每个ESM都需要计算昂贵的重新训练，并且在训练期间未见过的气候预测效果差。我们通过学习一个一致性模型(CM)，以零样本方式高效准确地降尺度任意ESM模拟来解决这些缺点。我们的基础模型方法以只受观测参考数据限制的分辨率产生概率性降尺度场。我们展示了CM在维持高可控性的同时以较低的计算成本优于最先进的扩散模型。

    arXiv:2403.02774v1 Announce Type: cross  Abstract: Accurate and high-resolution Earth system model (ESM) simulations are essential to assess the ecological and socio-economic impacts of anthropogenic climate change, but are computationally too expensive. Recent machine learning approaches have shown promising results in downscaling ESM simulations, outperforming state-of-the-art statistical approaches. However, existing methods require computationally costly retraining for each ESM and extrapolate poorly to climates unseen during training. We address these shortcomings by learning a consistency model (CM) that efficiently and accurately downscales arbitrary ESM simulations without retraining in a zero-shot manner. Our foundation model approach yields probabilistic downscaled fields at resolution only limited by the observational reference data. We show that the CM outperforms state-of-the-art diffusion models at a fraction of computational cost while maintaining high controllability on
    
[^3]: 大型卷积模型的参数高效调整

    Parameter-Efficient Tuning of Large Convolutional Models

    [https://arxiv.org/abs/2403.00269](https://arxiv.org/abs/2403.00269)

    通过引入滤波器子空间和滤波器原子的概念，本研究提出了一种在微调大型卷积模型时仅调整少量参数来提取任务特定表示的方法。

    

    为了解决微调大型预训练模型所需的高计算和参数复杂性，研究人员开发了参数高效的方法，仅更新下游任务的部分参数。然而，这些工作通常忽视了卷积核的独特属性，而卷积核仍然是许多大型模型的基本元素，比如Stable Diffusion。在本研究中，我们首先通过在每个网络层内分解卷积核到一小组滤波器子空间元素，即滤波器原子，引入了滤波器子空间。然后，我们通过仅调整滤波器原子（通常为几百个参数）对这些模型进行微调，以提取任务特定的表示。为了潜在地扩展调整的参数空间，我们进一步展示了一种简单的方法，通过递归地将每个筛选原子分解到另一组筛选原子来生成一个过完备的滤波器子空间。

    arXiv:2403.00269v1 Announce Type: cross  Abstract: To address the high computational and parameter complexity associated with fine-tuning large pre-trained models, researchers have developed parameter-efficient methods, where only partial parameters are updated for downstream tasks. However, these works often overlook the distinct properties of convolutional kernels, which still remain essential elements in many large models, such as Stable Diffusion. In this study, we first introduce filter subspace by decomposing convolutional kernels within each network layer over a small set of filter subspace elements, referred to as filter atoms. We then fine-tune these models to extract task-specific representation by only adapting the filter atoms, a few hundred parameters typically. To potentially expand the parameter space for tuning, we further show a simple approach to generate an overcomplete filter subspace by recursively decomposing each filter atom over another set of filter atoms. The 
    
[^4]: 不是所有的权重都是平等的: 在设备上增强能效的流式语音识别

    Not All Weights Are Created Equal: Enhancing Energy Efficiency in On-Device Streaming Speech Recognition

    [https://arxiv.org/abs/2402.13076](https://arxiv.org/abs/2402.13076)

    权重参数在设备上的流式语音识别模型中的功耗影响有所不同，作者提出了基于权重参数敏感性的有针对性压缩方法，将能源使用减少高达47%而维持模型准确性

    

    电力消耗在设备上的流式语音识别中起着重要作用，因为它直接影响用户体验。本研究深入探讨了语音识别模型中的权重参数如何影响这些模型的总体功耗。我们发现权重参数对功耗的影响因多种因素而异，受到调用频率及其在内存中的位置等因素的影响。凭借这一洞察力，我们制定了旨在优化设备上语音识别模型的设计指南。这些指南侧重于在尽量不显著影响准确性的情况下最小化功耗。我们的方法，基于权重参数变化敏感性的有针对性压缩，表现出优越性能，相比最先进的压缩方法，可以实现高达47%的能源使用减少，同时保持类似的模型准确性，并改善实时流

    arXiv:2402.13076v1 Announce Type: cross  Abstract: Power consumption plays an important role in on-device streaming speech recognition, as it has a direct impact on the user experience. This study delves into how weight parameters in speech recognition models influence the overall power consumption of these models. We discovered that the impact of weight parameters on power consumption varies, influenced by factors including how often they are invoked and their placement in memory. Armed with this insight, we developed design guidelines aimed at optimizing on-device speech recognition models. These guidelines focus on minimizing power use without substantially affecting accuracy. Our method, which employs targeted compression based on the varying sensitivities of weight parameters, demonstrates superior performance compared to state-of-the-art compression methods. It achieves a reduction in energy usage of up to 47% while maintaining similar model accuracy and improving the real-time f
    
[^5]: Permute-and-Flip：一种具有最佳鲁棒性和可加水印的LLMs解码器

    Permute-and-Flip: An optimally robust and watermarkable decoder for LLMs

    [https://arxiv.org/abs/2402.05864](https://arxiv.org/abs/2402.05864)

    提出了一种名为Permute-and-Flip（PF）解码器，其具有最佳的鲁棒性和质量-鲁棒性的 tradeoff，且比采样方法更好。还设计了一种针对PF解码器的水印方案，能够保持样本的分布不变，并实现任意低的假阳性率和高的召回率。实验证明PF解码器在困惑度方面明显优于朴素采样，为LLM解码提供了一种有希望的新方法。

    

    在本文中，我们提出了一种名为Permute-and-Flip（PF）解码器的新解码方法。它具有与标准采样解码器相似的鲁棒性特性，但在质量和鲁棒性的 tradeoff 上证明比采样方法更好，且永远不会差于任何其他解码器。同时，我们还设计了一种类似于Aaronson的Gumbel水印的加密水印方案，但是针对PF解码器而自然量身定制。该水印方案不改变样本的分布，同时允许任意低的假阳性率和高的召回率，只要生成的文本具有高熵。我们的实验证明，PF解码器（及其带有水印的对应物）在困惑度方面明显优于朴素采样（及其带有Gumbel水印的对应物），同时保持相同的鲁棒性（和可检测性），因此为LLM解码提供了一个有希望的新方法。代码可在https://github.com/XuandongZhao/pf-decoding找到。

    In this paper, we propose a new decoding method called Permute-and-Flip (PF) decoder. It enjoys robustness properties similar to the standard sampling decoder, but is provably up to 2x better in its quality-robustness tradeoff than sampling and never worse than any other decoder. We also design a cryptographic watermarking scheme analogous to Aaronson's Gumbel watermark, but naturally tailored for PF decoder. The watermarking scheme does not change the distribution to sample, while allowing arbitrarily low false positive rate and high recall whenever the generated text has high entropy. Our experiments show that the PF decoder (and its watermarked counterpart) significantly outperform(s) naive sampling (and it's Gumbel watermarked counterpart) in terms of perplexity, while retaining the same robustness (and detectability), hence making it a promising new approach for LLM decoding. The code is available at https://github.com/XuandongZhao/pf-decoding
    
[^6]: 基于大规模语言模型的超参数优化的技术

    Large Language Model Agent for Hyper-Parameter Optimization

    [https://arxiv.org/abs/2402.01881](https://arxiv.org/abs/2402.01881)

    基于大规模语言模型的AgentHPO技术通过自动化超参数优化，在机器学习任务中大大减少了试验次数，简化了设置过程，提升了解释性和用户信任。

    

    超参数优化在现代机器学习中至关重要，需要专业知识、大量实验以及高计算和人力资源。尽管自动化机器学习（AutoML）取得了一些进展，但试验效率、设置复杂性和互操作性方面仍存在挑战。为了解决这些问题，我们引入了一种新的范式，利用大规模语言模型（LLMs）来自动化不同机器学习任务的超参数优化，称为AgentHPO（LLM Agent-based Hyperparameter Optimization）。具体来说，AgentHPO自主处理任务信息，根据历史试验对特定超参数（HPs）进行实验，并进行迭代优化。与传统的AutoML方法相比，这种类似人类的优化过程极大地减少了所需的试验次数，简化了设置过程，并提升了解释性和用户信任。

    Hyperparameter optimization is critical in modern machine learning, requiring expert knowledge, numerous trials, and high computational and human resources. Despite the advancements in Automated Machine Learning (AutoML), challenges in terms of trial efficiency, setup complexity, and interoperability still persist. To address these issues, we introduce a novel paradigm leveraging Large Language Models (LLMs) to automate hyperparameter optimization across diverse machine learning tasks, which is named AgentHPO (short for LLM Agent-based Hyperparameter Optimization). Specifically, AgentHPO processes the task information autonomously, conducts experiments with specific hyperparameters (HPs), and iteratively optimizes them based on historical trials. This human-like optimization process largely reduces the number of required trials, simplifies the setup process, and enhances interpretability and user trust, compared to traditional AutoML methods. Extensive empirical experiments conducted o
    
[^7]: SMOOTHIE: 软件分析的超参数优化理论

    SMOOTHIE: A Theory of Hyper-parameter Optimization for Software Analytics. (arXiv:2401.09622v1 [cs.SE])

    [http://arxiv.org/abs/2401.09622](http://arxiv.org/abs/2401.09622)

    SMOOTHIE是一种通过考虑损失函数的“光滑度”来引导超参数优化的新型方法，在软件分析中应用可以带来显著的性能改进。

    

    超参数优化是调整学习器控制参数的黑魔法。在软件分析中，经常发现调优可以带来显著的性能改进。尽管如此，超参数优化在软件分析中通常被很少或很差地应用，可能是因为探索所有参数选项的CPU成本太高。我们假设当损失函数的“光滑度”更好时，学习器的泛化能力更强。这个理论非常有用，因为可以很快测试不同超参数选择对“光滑度”的影响（例如，对于深度学习器，在一个epoch之后就可以进行测试）。为了测试这个理论，本文实现和测试了SMOOTHIE，一种通过考虑“光滑度”来引导优化的新型超参数优化器。本文的实验将SMOOTHIE应用于多个软件工程任务，包括（a）GitHub问题寿命预测；（b）静态代码警告中错误警报的检测；（c）缺陷预测。

    Hyper-parameter optimization is the black art of tuning a learner's control parameters. In software analytics, a repeated result is that such tuning can result in dramatic performance improvements. Despite this, hyper-parameter optimization is often applied rarely or poorly in software analytics--perhaps due to the CPU cost of exploring all those parameter options can be prohibitive.  We theorize that learners generalize better when the loss landscape is ``smooth''. This theory is useful since the influence on ``smoothness'' of different hyper-parameter choices can be tested very quickly (e.g. for a deep learner, after just one epoch).  To test this theory, this paper implements and tests SMOOTHIE, a novel hyper-parameter optimizer that guides its optimizations via considerations of ``smothness''. The experiments of this paper test SMOOTHIE on numerous SE tasks including (a) GitHub issue lifetime prediction; (b) detecting false alarms in static code warnings; (c) defect prediction, and
    
[^8]: 深度学习的数学介绍：方法、实现和理论

    Mathematical Introduction to Deep Learning: Methods, Implementations, and Theory. (arXiv:2310.20360v1 [cs.LG])

    [http://arxiv.org/abs/2310.20360](http://arxiv.org/abs/2310.20360)

    本书提供了对深度学习算法的数学介绍，包括不同的神经网络架构和优化算法，并涵盖了深度学习算法的理论方面。此外，还介绍了深度学习逼近偏微分方程的方法。希望对学生和科学家们有所帮助。

    

    本书旨在介绍深度学习算法的主题。我们详细介绍了深度学习算法的基本组成部分，包括不同的人工神经网络架构（如全连接前馈神经网络、卷积神经网络、循环神经网络、残差神经网络和带有批归一化的神经网络）以及不同的优化算法（如基本的随机梯度下降法、加速方法和自适应方法）。我们还涵盖了深度学习算法的几个理论方面，如人工神经网络的逼近能力（包括神经网络的微积分）、优化理论（包括Kurdyka-Lojasiewicz不等式）和泛化误差。在本书的最后一部分，我们还回顾了一些用于偏微分方程的深度学习逼近方法，包括物理信息神经网络（PINNs）和深度Galerkin方法。希望本书能对学生和科学家们有所帮助。

    This book aims to provide an introduction to the topic of deep learning algorithms. We review essential components of deep learning algorithms in full mathematical detail including different artificial neural network (ANN) architectures (such as fully-connected feedforward ANNs, convolutional ANNs, recurrent ANNs, residual ANNs, and ANNs with batch normalization) and different optimization algorithms (such as the basic stochastic gradient descent (SGD) method, accelerated methods, and adaptive methods). We also cover several theoretical aspects of deep learning algorithms such as approximation capacities of ANNs (including a calculus for ANNs), optimization theory (including Kurdyka-{\L}ojasiewicz inequalities), and generalization errors. In the last part of the book some deep learning approximation methods for PDEs are reviewed including physics-informed neural networks (PINNs) and deep Galerkin methods. We hope that this book will be useful for students and scientists who do not yet 
    
[^9]: DCSI -- 基于分离和连通性的改进的聚类可分离性度量

    DCSI -- An improved measure of cluster separability based on separation and connectedness. (arXiv:2310.12806v1 [stat.ML])

    [http://arxiv.org/abs/2310.12806](http://arxiv.org/abs/2310.12806)

    这篇论文提出了一种改进的聚类可分离性度量方法，旨在量化类间分离和类内连通性，对于密度聚类具有较好的性能表现。

    

    确定给定数据集中的类别标签是否对应于有意义的聚类对于使用真实数据集评估聚类算法至关重要。这个特性可以通过可分离性度量来量化。现有文献的综述显示，既有的基于分类的复杂性度量方法和聚类有效性指标 (CVIs) 都没有充分融入基于密度的聚类的核心特征：类间分离和类内连通性。一种新开发的度量方法 (密度聚类可分离性指数, DCSI) 旨在量化这两个特征，并且也可用作 CVI。对合成数据的广泛实验表明，DCSI 与通过调整兰德指数 (ARI) 测量的DBSCAN的性能之间有很强的相关性，但在对多类数据集进行密度聚类不适当的重叠类别时缺乏鲁棒性。对经常使用的真实数据集进行详细评估显示，DCSI 能够更好地区分密度聚类的可分离性。

    Whether class labels in a given data set correspond to meaningful clusters is crucial for the evaluation of clustering algorithms using real-world data sets. This property can be quantified by separability measures. A review of the existing literature shows that neither classification-based complexity measures nor cluster validity indices (CVIs) adequately incorporate the central aspects of separability for density-based clustering: between-class separation and within-class connectedness. A newly developed measure (density cluster separability index, DCSI) aims to quantify these two characteristics and can also be used as a CVI. Extensive experiments on synthetic data indicate that DCSI correlates strongly with the performance of DBSCAN measured via the adjusted rand index (ARI) but lacks robustness when it comes to multi-class data sets with overlapping classes that are ill-suited for density-based hard clustering. Detailed evaluation on frequently used real-world data sets shows that
    
[^10]: LGL-BCI：一种轻量级几何学习框架用于基于运动想象的脑机接口

    LGL-BCI: A Lightweight Geometric Learning Framework for Motor Imagery-Based Brain-Computer Interfaces. (arXiv:2310.08051v1 [cs.LG])

    [http://arxiv.org/abs/2310.08051](http://arxiv.org/abs/2310.08051)

    LGL-BCI是一种轻量级几何学习框架，通过处理EEG数据在非欧几里德度量空间中捕捉运动想象任务的空间相关性，并通过特征分解算法进行EEG通道选择以提高推断速度。实验证明LGL-BCI相比现有解决方案具有更高的准确性和效率。

    

    脑机接口是一种使用脑信号与外部设备进行交互的开创性技术。尽管有所进展，基于脑电图（EEG）的运动想象任务面临挑战，如幅度和相位变异，以及复杂的空间相关性，需要更小的模型大小和更快的推断。本研究介绍了LGL-BCI框架，采用几何深度学习框架处理非欧几里德度量空间中的EEG，特别是对称正定（SPD）流形空间。LGL-BCI提供了稳健的EEG数据表示，并捕捉了空间相关性。我们提出了一种通过特征分解算法进行EEG通道选择的解决方案，以减少SPD矩阵的维度，同时提高了推断速度。广泛的实验显示，与当前解决方案相比，LGL-BCI具有更高的准确性和效率，突出了几何深度学习在运动想象-脑机接口应用中的潜力。

    Brain-Computer Interfaces (BCIs) are a groundbreaking technology for interacting with external devices using brain signals. Despite advancements, electroencephalogram (EEG)-based Motor Imagery (MI) tasks face challenges like amplitude and phase variability, and complex spatial correlations, with a need for smaller model size and faster inference. This study introduces the LGL-BCI framework, employing a Geometric Deep Learning Framework for EEG processing in non-Euclidean metric spaces, particularly the Symmetric Positive Definite (SPD) Manifold space. LGL-BCI offers robust EEG data representation and captures spatial correlations. We propose an EEG channel selection solution via a feature decomposition algorithm to reduce SPD matrix dimensionality, with a lossless transformation boosting inference speed. Extensive experiments show LGL-BCI's superior accuracy and efficiency compared to current solutions, highlighting geometric deep learning's potential in MI-BCI applications. The effici
    
[^11]: AdaRec：用于增强用户长期参与度的自适应顺序推荐算法

    AdaRec: Adaptive Sequential Recommendation for Reinforcing Long-term User Engagement. (arXiv:2310.03984v1 [cs.IR])

    [http://arxiv.org/abs/2310.03984](http://arxiv.org/abs/2310.03984)

    AdaRec是一种自适应顺序推荐算法，通过引入基于距离的表示损失来提取潜在信息，以适应大规模在线推荐系统中用户行为模式的变化。

    

    在顺序推荐任务中，人们越来越关注使用强化学习算法来优化用户的长期参与度。大规模在线推荐系统面临的一个挑战是用户行为模式（如互动频率和保留倾向）的不断复杂变化。当将问题建模为马尔科夫决策过程时，推荐系统的动态和奖励函数会不断受到这些变化的影响。现有的推荐系统强化学习算法会受到分布偏移问题的困扰，并难以适应这种马尔科夫决策过程。本文介绍了一种新的范式，称为自适应顺序推荐（AdaRec），来解决这个问题。AdaRec提出了一种基于距离的表示损失，从用户的互动轨迹中提取潜在信息。这些信息反映了强化学习策略与当前用户行为模式的匹配程度，并帮助策略识别推荐系统中的细微变化。

    Growing attention has been paid to Reinforcement Learning (RL) algorithms when optimizing long-term user engagement in sequential recommendation tasks. One challenge in large-scale online recommendation systems is the constant and complicated changes in users' behavior patterns, such as interaction rates and retention tendencies. When formulated as a Markov Decision Process (MDP), the dynamics and reward functions of the recommendation system are continuously affected by these changes. Existing RL algorithms for recommendation systems will suffer from distribution shift and struggle to adapt in such an MDP. In this paper, we introduce a novel paradigm called Adaptive Sequential Recommendation (AdaRec) to address this issue. AdaRec proposes a new distance-based representation loss to extract latent information from users' interaction trajectories. Such information reflects how RL policy fits to current user behavior patterns, and helps the policy to identify subtle changes in the recomm
    
[^12]: 混合编码器支持连续语音分离用于会议识别

    Mixture Encoder Supporting Continuous Speech Separation for Meeting Recognition. (arXiv:2309.08454v1 [eess.AS])

    [http://arxiv.org/abs/2309.08454](http://arxiv.org/abs/2309.08454)

    本研究将混合编码器方法从两个说话人情况扩展到了更自然的会议环境，包括任意数量的说话人和动态重叠。实验证明，该方法在LibriCSS数据集上达到了最先进的性能，并凸显了混合编码器的优势。

    

    自动语音识别（ASR）的许多实际应用需要处理重叠的语音。一种常见的方法是首先将语音分离成无重叠的流，然后对生成的信号进行ASR。最近，提出了在ASR模型中包含混合编码器的方法。该混合编码器利用原始重叠的语音来减轻语音分离引入的伪影效果。然而，先前的方法仅针对两个说话人的情况。在这项工作中，我们将这种方法扩展到更自然的会议环境，包括任意数量的说话人和动态重叠。我们使用不同的语音分离器（包括强大的TF-GridNet模型）评估性能。实验证明，在LibriCSS数据集上达到了最先进的性能，并凸显了混合编码器的优势。此外，实验还展示了TF-GridNet的强大分离能力，大大缩小了先前方法的差距。

    Many real-life applications of automatic speech recognition (ASR) require processing of overlapped speech. A commonmethod involves first separating the speech into overlap-free streams and then performing ASR on the resulting signals. Recently, the inclusion of a mixture encoder in the ASR model has been proposed. This mixture encoder leverages the original overlapped speech to mitigate the effect of artifacts introduced by the speech separation. Previously, however, the method only addressed two-speaker scenarios. In this work, we extend this approach to more natural meeting contexts featuring an arbitrary number of speakers and dynamic overlaps. We evaluate the performance using different speech separators, including the powerful TF-GridNet model. Our experiments show state-of-the-art performance on the LibriCSS dataset and highlight the advantages of the mixture encoder. Furthermore, they demonstrate the strong separation of TF-GridNet which largely closes the gap between previous m
    
[^13]: 在金融领域中实现量子生成对抗网络（qGAN）和QCBM

    Implementing Quantum Generative Adversarial Network (qGAN) and QCBM in Finance. (arXiv:2308.08448v1 [quant-ph])

    [http://arxiv.org/abs/2308.08448](http://arxiv.org/abs/2308.08448)

    这项研究讨论了在金融领域中应用量子机器学习的新研究方向，通过比较qGAN和QCBM等模型，展示了在金融领域中实现量子优势的潜力。

    

    量子机器学习（QML）是一个跨学科的领域，由两个最具创新性的研究领域组成：量子计算和经典机器学习（ML），ML和人工智能（AI）被认为是将受到量子计算机兴起影响的第一个领域。这项工作讨论了在金融中应用量子机器学习（QML）的一些新研究领域，我们讨论了一些已在金融界引起关注的QML模型，以及使用模拟环境中的真实金融数据集对qGAN（量子生成对抗网络）和QCBM（量子电路Born机）等模型进行比较。对于qGAN，我们定义了鉴别器和生成器的量子电路，并展示了未来在金融领域中通过QML实现量子优势的潜力。

    Quantum machine learning (QML) is a cross-disciplinary subject made up of two of the most exciting research areas: quantum computing and classical machine learning (ML), with ML and artificial intelligence (AI) being projected as the first fields that will be impacted by the rise of quantum machines. Quantum computers are being used today in drug discovery, material & molecular modelling and finance. In this work, we discuss some upcoming active new research areas in application of quantum machine learning (QML) in finance. We discuss certain QML models that has become areas of active interest in the financial world for various applications. We use real world financial dataset and compare models such as qGAN (quantum generative adversarial networks) and QCBM (quantum circuit Born machine) among others, using simulated environments. For the qGAN, we define quantum circuits for discriminators and generators and show promises of future quantum advantage via QML in finance.
    
[^14]: 关于AI问责政策的探索

    Towards an AI Accountability Policy. (arXiv:2307.13658v1 [cs.CY])

    [http://arxiv.org/abs/2307.13658](http://arxiv.org/abs/2307.13658)

    这份白皮书是对美国国家电信和信息管理局的“AI问责政策评论请求”的回应，提出了一组相互关联的AI问责政策建议。

    

    这份白皮书是对美国国家电信和信息管理局的“AI问责政策评论请求”作出的回应。在回答相关问题的关键句子末尾，提供了要求评论的问题编号的上标。该白皮书提出了一组相互关联的AI问责政策建议。

    This white paper is a response to the "AI Accountability Policy Request for Comments" by the National Telecommunications and Information Administration of the United States. The question numbers for which comments were requested are provided in superscripts at the end of key sentences answering the respective questions. The white paper offers a set of interconnected recommendations for an AI accountability policy.
    
[^15]: 图解化：利用图解型AI解释对假设性演绎推理的理性化

    Diagrammatization: Rationalizing with diagrammatic AI explanations for abductive-deductive reasoning on hypotheses. (arXiv:2302.01241v2 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2302.01241](http://arxiv.org/abs/2302.01241)

    本文提出了一种图解化的方法，以支持可解释的人工智能，通过图解型和假设性推理，缩小可解释性差距。通过临床应用研究和建模研究，我们发现DiagramNet不仅能提供忠实的杂音形状解释，还具有较好的预测性能，而且图解型解释在临床相关的情况下更受推崇。

    

    许多可解释的人工智能（XAI）可视化工具已经被开发出来，但它们通常需要用户进一步推理来解释。我们认为，XAI应该支持图解型和假设性推理，以便AI能够进行假设生成和评估，从而减少可解释性差距。我们提出了图解化方法，以i)进行Peircean推导-演绎推理，ii)遵循领域惯例，和iii)用图示或语言进行解释。我们在临床应用领域实现了DiagramNet，以预测心脏听诊中的心脏诊断，并用基于形状的杂音图解进行解释。在建模研究中，我们发现DiagramNet不仅提供了忠实的杂音形状解释，而且比基线模型具有更好的预测性能。我们进一步通过医学生的定性用户研究展示了图解型解释的可理解性和可信度，并表明在临床相关的情况下，图解式解释比其他方式更受推崇。

    Many visualizations have been developed for explainable AI (XAI), but they often require further reasoning by users to interpret. We argue that XAI should support diagrammatic and abductive reasoning for the AI to perform hypothesis generation and evaluation to reduce the interpretability gap. We propose Diagrammatization to i) perform Peircean abductive-deductive reasoning, ii) follow domain conventions, and iii) explain with diagrams visually or verbally. We implemented DiagramNet for a clinical application to predict cardiac diagnoses from heart auscultation, and explain with shape-based murmur diagrams. In modeling studies, we found that DiagramNet not only provides faithful murmur shape explanations, but also has better prediction performance than baseline models. We further demonstrate the interpretability and trustworthiness of diagrammatic explanations in a qualitative user study with medical students, showing that clinically-relevant, diagrammatic explanations are preferred ov
    

