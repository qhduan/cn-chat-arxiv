# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Phase-driven Domain Generalizable Learning for Nonstationary Time Series](https://arxiv.org/abs/2402.05960) | 该论文提出了一个基于相位驱动的时间序列学习框架PhASER，通过相位增强、分离特征编码和特征广播的方法，实现了对非平稳数据的通用学习能力。 |
| [^2] | [Towards Context-Aware Domain Generalization: Understanding the Benefits and Limits of Marginal Transfer Learning](https://arxiv.org/abs/2312.10107) | 分析了上下文感知领域泛化的条件，提出了理论分析和实证分析所需的标准，并展示了该方法可以检测非常数域的场景。 |
| [^3] | [AtomSurf : Surface Representation for Learning on Protein Structures.](http://arxiv.org/abs/2309.16519) | 本文研究了将蛋白质作为3D网格的表面表示，并提出了一种结合图表面的协同方法，既有竞争优势，又有实际应用潜力。 |
| [^4] | [DR-VIDAL -- Doubly Robust Variational Information-theoretic Deep Adversarial Learning for Counterfactual Prediction and Treatment Effect Estimation on Real World Data.](http://arxiv.org/abs/2303.04201) | DR-VIDAL是一个新型的生成框架，可用于处理真实世界数据中的干预措施对结果的因果效应估计，并具有处理混淆偏差和模型不良的能力。 |

# 详细

[^1]: 基于相位驱动的非平稳时间序列通用学习

    Phase-driven Domain Generalizable Learning for Nonstationary Time Series

    [https://arxiv.org/abs/2402.05960](https://arxiv.org/abs/2402.05960)

    该论文提出了一个基于相位驱动的时间序列学习框架PhASER，通过相位增强、分离特征编码和特征广播的方法，实现了对非平稳数据的通用学习能力。

    

    监测和识别连续感知数据中的模式对许多实际应用至关重要。这些实际时间序列数据通常是非平稳的，其统计和谱特性随时间变化。这在开发能够有效泛化不同分布的学习模型方面提出了重大挑战。在本工作中，我们观察到非平稳统计与相位信息内在相关，提出了一个时间序列学习框架PhASER。它包括三个新颖的元素：1）相位增强，使非平稳性多样化同时保留有区别性的语义；2）将时变幅度和相位视为独立模态进行单独特征编码；3）利用新颖的残差连接将相位与特征结合，以强化分布不变性学习的固有正则化作用。通过在5个人体活动识别数据集上进行广泛评估，

    Monitoring and recognizing patterns in continuous sensing data is crucial for many practical applications. These real-world time-series data are often nonstationary, characterized by varying statistical and spectral properties over time. This poses a significant challenge in developing learning models that can effectively generalize across different distributions. In this work, based on our observation that nonstationary statistics are intrinsically linked to the phase information, we propose a time-series learning framework, PhASER. It consists of three novel elements: 1) phase augmentation that diversifies non-stationarity while preserving discriminatory semantics, 2) separate feature encoding by viewing time-varying magnitude and phase as independent modalities, and 3) feature broadcasting by incorporating phase with a novel residual connection for inherent regularization to enhance distribution invariant learning. Upon extensive evaluation on 5 datasets from human activity recognit
    
[^2]: 迈向面向上下文感知领域泛化：理解边缘传递学习的好处和限制

    Towards Context-Aware Domain Generalization: Understanding the Benefits and Limits of Marginal Transfer Learning

    [https://arxiv.org/abs/2312.10107](https://arxiv.org/abs/2312.10107)

    分析了上下文感知领域泛化的条件，提出了理论分析和实证分析所需的标准，并展示了该方法可以检测非常数域的场景。

    

    在这项工作中，我们分析了关于输入$X$的上下文信息如何改善深度学习模型在新领域中的预测的条件。在领域泛化中边缘传递学习的研究基础上，我们将上下文的概念形式化为一组数据点的排列不变表示，这些数据点来自于与输入本身相同的域。我们对这种方法在原则上可以产生好处的条件进行了理论分析，并制定了两个在实践中可以轻松验证的必要标准。此外，我们提供了关于边缘传递学习方法有望具有稳健性的分布变化类型的见解。实证分析表明我们的标准有效地区分了有利和不利的场景。最后，我们证明可以可靠地检测模型面临非常数域的场景。

    arXiv:2312.10107v2 Announce Type: replace-cross  Abstract: In this work, we analyze the conditions under which information about the context of an input $X$ can improve the predictions of deep learning models in new domains. Following work in marginal transfer learning in Domain Generalization (DG), we formalize the notion of context as a permutation-invariant representation of a set of data points that originate from the same domain as the input itself. We offer a theoretical analysis of the conditions under which this approach can, in principle, yield benefits, and formulate two necessary criteria that can be easily verified in practice. Additionally, we contribute insights into the kind of distribution shifts for which the marginal transfer learning approach promises robustness. Empirical analysis shows that our criteria are effective in discerning both favorable and unfavorable scenarios. Finally, we demonstrate that we can reliably detect scenarios where a model is tasked with unw
    
[^3]: AtomSurf：蛋白质结构上的学习的表面表示

    AtomSurf : Surface Representation for Learning on Protein Structures. (arXiv:2309.16519v1 [cs.LG])

    [http://arxiv.org/abs/2309.16519](http://arxiv.org/abs/2309.16519)

    本文研究了将蛋白质作为3D网格的表面表示，并提出了一种结合图表面的协同方法，既有竞争优势，又有实际应用潜力。

    

    近期Cryo-EM和蛋白质结构预测算法的进展使得大规模蛋白质结构可获得，为基于机器学习的功能注释铺平了道路。几何深度学习领域关注创建适用于几何数据的方法。从蛋白质结构中学习的一个重要方面是将这些结构表示为几何对象（如网格、图或表面）并应用适合这种表示形式的学习方法。给定方法的性能将取决于表示和相应的学习方法。在本文中，我们研究将蛋白质表示为$\textit{3D mesh surfaces}$并将其纳入已建立的表示基准中。我们的第一个发现是，尽管有着有希望的初步结果，但仅单独表面表示似乎无法与3D网格竞争。在此基础上，我们提出了一种协同方法，将表面表示与图表面结合起来。

    Recent advancements in Cryo-EM and protein structure prediction algorithms have made large-scale protein structures accessible, paving the way for machine learning-based functional annotations.The field of geometric deep learning focuses on creating methods working on geometric data. An essential aspect of learning from protein structures is representing these structures as a geometric object (be it a grid, graph, or surface) and applying a learning method tailored to this representation. The performance of a given approach will then depend on both the representation and its corresponding learning method.  In this paper, we investigate representing proteins as $\textit{3D mesh surfaces}$ and incorporate them into an established representation benchmark. Our first finding is that despite promising preliminary results, the surface representation alone does not seem competitive with 3D grids. Building on this, we introduce a synergistic approach, combining surface representations with gra
    
[^4]: DR-VIDAL--双重稳健变分信息论深度对抗学习用于真实世界数据的反事实预测和治疗效果估计

    DR-VIDAL -- Doubly Robust Variational Information-theoretic Deep Adversarial Learning for Counterfactual Prediction and Treatment Effect Estimation on Real World Data. (arXiv:2303.04201v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2303.04201](http://arxiv.org/abs/2303.04201)

    DR-VIDAL是一个新型的生成框架，可用于处理真实世界数据中的干预措施对结果的因果效应估计，并具有处理混淆偏差和模型不良的能力。

    

    从真实世界的观察性（非随机化）数据中确定干预措施对结果的因果效应，例如使用电子健康记录的治疗重用，由于潜在偏差而具有挑战性。因果深度学习已经改进了传统技术，用于估计个性化治疗效果（ITE）。我们提出了双重稳健变分信息论深度对抗学习（DR-VIDAL），这是一个结合了治疗和结果两个联合模型的新型生成框架，确保无偏的ITE估计，即使其中一个模型设定不正确。DR-VIDAL整合了： （i）变分自编码器（VAE）根据因果假设将混淆变量分解为潜在变量; （ii）基于信息论的生成对抗网络（Info-GAN）用于生成反事实情况; （iii）一个双重稳健块，其中包括治疗倾向于预测结果。在合成和真实数据集（Infant Health和Development Program，Transforming Clinical Practice Initiative [TCPI]）中进行实验，我们证明了DR-VIDAL在估计ITE方面优于现有的最先进方法，因为它具有处理混淆偏差和模型不正确的能力。

    Determining causal effects of interventions onto outcomes from real-world, observational (non-randomized) data, e.g., treatment repurposing using electronic health records, is challenging due to underlying bias. Causal deep learning has improved over traditional techniques for estimating individualized treatment effects (ITE). We present the Doubly Robust Variational Information-theoretic Deep Adversarial Learning (DR-VIDAL), a novel generative framework that combines two joint models of treatment and outcome, ensuring an unbiased ITE estimation even when one of the two is misspecified. DR-VIDAL integrates: (i) a variational autoencoder (VAE) to factorize confounders into latent variables according to causal assumptions; (ii) an information-theoretic generative adversarial network (Info-GAN) to generate counterfactuals; (iii) a doubly robust block incorporating treatment propensities for outcome predictions. On synthetic and real-world datasets (Infant Health and Development Program, T
    

