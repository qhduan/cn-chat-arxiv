# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Bipartite Graph Variational Auto-Encoder with Fair Latent Representation to Account for Sampling Bias in Ecological Networks](https://arxiv.org/abs/2403.02011) | 本研究提出了一种公平潜在表示的二分图变分自动编码器方法，以解决生态网络中的抽样偏差问题，通过在损失函数中引入额外的HSIC惩罚项，确保了潜在空间结构与连续变量的独立性。 |
| [^2] | [Decentralized Bilevel Optimization over Graphs: Loopless Algorithmic Update and Transient Iteration Complexity](https://arxiv.org/abs/2402.03167) | 本文提出了一种单循环的去中心化双级优化算法（D-SOBA），首次阐明了网络拓扑和数据异构性对去中心化双级算法的共同影响。D-SOBA在渐近速率、渐近梯度/海森复杂性和瞬态梯度/海森复杂性方面达到了最先进水平。 |
| [^3] | [Spatial-and-Frequency-aware Restoration method for Images based on Diffusion Models](https://arxiv.org/abs/2401.17629) | 本文提出了一种名为SaFaRI的基于空间和频率感知的扩散模型，用于图像恢复。在各种噪声逆问题上，SaFaRI在ImageNet数据集和FFHQ数据集上实现了最先进的性能。 |
| [^4] | [Can Large Language Models Replace Economic Choice Prediction Labs?](https://arxiv.org/abs/2401.17435) | 该论文研究大型语言模型是否能够取代经济实验室进行选择预测，并通过相关实验证明了其可行性。 |
| [^5] | [LSAP: Rethinking Inversion Fidelity, Perception and Editability in GAN Latent Space.](http://arxiv.org/abs/2209.12746) | LSAP通过对潜空间实现对齐解决了反演和编辑结果中保真度、感知和可编辑性的问题，使得在保留重建保真度的前提下具有更好的感知和可编辑性。 |

# 详细

[^1]: 公平潜在表示的二分图变分自动编码器，以解决生态网络中的抽样偏差问题

    Bipartite Graph Variational Auto-Encoder with Fair Latent Representation to Account for Sampling Bias in Ecological Networks

    [https://arxiv.org/abs/2403.02011](https://arxiv.org/abs/2403.02011)

    本研究提出了一种公平潜在表示的二分图变分自动编码器方法，以解决生态网络中的抽样偏差问题，通过在损失函数中引入额外的HSIC惩罚项，确保了潜在空间结构与连续变量的独立性。

    

    我们提出一种方法，使用图嵌入来表示二分网络，以解决研究生态网络所面临的挑战，比如连接植物和传粉者等网络，需考虑许多协变量，尤其要控制抽样偏差。我们将变分图自动编码器方法调整为二分情况，从而能够在潜在空间中生成嵌入，其中两组节点的位置基于它们的连接概率。我们将在社会学中常考虑的公平性框架转化为生态学中的抽样偏差问题。通过在损失函数中添加Hilbert-Schmidt独立准则（HSIC）作为额外惩罚项，我们确保潜在空间结构与连续变量（与抽样过程相关）无关。最后，我们展示了我们的方法如何改变我们对生态网络的理解。

    arXiv:2403.02011v1 Announce Type: cross  Abstract: We propose a method to represent bipartite networks using graph embeddings tailored to tackle the challenges of studying ecological networks, such as the ones linking plants and pollinators, where many covariates need to be accounted for, in particular to control for sampling bias. We adapt the variational graph auto-encoder approach to the bipartite case, which enables us to generate embeddings in a latent space where the two sets of nodes are positioned based on their probability of connection. We translate the fairness framework commonly considered in sociology in order to address sampling bias in ecology. By incorporating the Hilbert-Schmidt independence criterion (HSIC) as an additional penalty term in the loss we optimize, we ensure that the structure of the latent space is independent of continuous variables, which are related to the sampling process. Finally, we show how our approach can change our understanding of ecological n
    
[^2]: 图上的去中心化双级优化: 无环算法更新和瞬态迭代复杂性

    Decentralized Bilevel Optimization over Graphs: Loopless Algorithmic Update and Transient Iteration Complexity

    [https://arxiv.org/abs/2402.03167](https://arxiv.org/abs/2402.03167)

    本文提出了一种单循环的去中心化双级优化算法（D-SOBA），首次阐明了网络拓扑和数据异构性对去中心化双级算法的共同影响。D-SOBA在渐近速率、渐近梯度/海森复杂性和瞬态梯度/海森复杂性方面达到了最先进水平。

    

    随机双级优化（SBO）在处理嵌套结构方面的多样性使其在机器学习中变得越来越重要。为了解决大规模SBO，去中心化方法作为有效的范例出现，其中节点与直接相邻节点进行通信，无需中央服务器，从而提高通信效率和增强算法的稳健性。然而，当前的去中心化SBO算法面临挑战，包括昂贵的内部循环更新和对网络拓扑、数据异构性和嵌套双级算法结构的影响不明确。在本文中，我们引入了一种单循环的去中心化SBO（D-SOBA）算法，并建立了其瞬态迭代复杂性，首次澄清了网络拓扑和数据异构性对去中心化双级算法的共同影响。D-SOBA实现了最先进的渐近速率、渐近梯度/海森复杂性和瞬态梯度/海森复杂性。

    Stochastic bilevel optimization (SBO) is becoming increasingly essential in machine learning due to its versatility in handling nested structures. To address large-scale SBO, decentralized approaches have emerged as effective paradigms in which nodes communicate with immediate neighbors without a central server, thereby improving communication efficiency and enhancing algorithmic robustness. However, current decentralized SBO algorithms face challenges, including expensive inner-loop updates and unclear understanding of the influence of network topology, data heterogeneity, and the nested bilevel algorithmic structures. In this paper, we introduce a single-loop decentralized SBO (D-SOBA) algorithm and establish its transient iteration complexity, which, for the first time, clarifies the joint influence of network topology and data heterogeneity on decentralized bilevel algorithms. D-SOBA achieves the state-of-the-art asymptotic rate, asymptotic gradient/Hessian complexity, and transien
    
[^3]: 基于扩散模型的空间和频率感知图像恢复方法

    Spatial-and-Frequency-aware Restoration method for Images based on Diffusion Models

    [https://arxiv.org/abs/2401.17629](https://arxiv.org/abs/2401.17629)

    本文提出了一种名为SaFaRI的基于空间和频率感知的扩散模型，用于图像恢复。在各种噪声逆问题上，SaFaRI在ImageNet数据集和FFHQ数据集上实现了最先进的性能。

    

    扩散模型最近成为一种有前途的图像恢复（IR）框架，因为它们能够产生高质量的重建结果并且与现有方法兼容。现有的解决IR中噪声逆问题的方法通常仅考虑像素级的数据保真度。在本文中，我们提出了一种名为SaFaRI的基于空间和频率感知的扩散模型，用于处理带有高斯噪声的IR问题。我们的模型鼓励图像在空间和频率域中保持数据保真度，从而提高重建质量。我们全面评估了我们的模型在各种噪声逆问题上的性能，包括修复、降噪和超分辨率。我们的细致评估表明，SaFaRI在ImageNet数据集和FFHQ数据集上实现了最先进的性能，以LPIPS和FID指标超过了现有的零样本IR方法。

    Diffusion models have recently emerged as a promising framework for Image Restoration (IR), owing to their ability to produce high-quality reconstructions and their compatibility with established methods. Existing methods for solving noisy inverse problems in IR, considers the pixel-wise data-fidelity. In this paper, we propose SaFaRI, a spatial-and-frequency-aware diffusion model for IR with Gaussian noise. Our model encourages images to preserve data-fidelity in both the spatial and frequency domains, resulting in enhanced reconstruction quality. We comprehensively evaluate the performance of our model on a variety of noisy inverse problems, including inpainting, denoising, and super-resolution. Our thorough evaluation demonstrates that SaFaRI achieves state-of-the-art performance on both the ImageNet datasets and FFHQ datasets, outperforming existing zero-shot IR methods in terms of LPIPS and FID metrics.
    
[^4]: 大型语言模型能否取代经济选择预测实验室？

    Can Large Language Models Replace Economic Choice Prediction Labs?

    [https://arxiv.org/abs/2401.17435](https://arxiv.org/abs/2401.17435)

    该论文研究大型语言模型是否能够取代经济实验室进行选择预测，并通过相关实验证明了其可行性。

    

    经济选择预测是一项具有挑战性的重要任务，往往受限于获取人类选择数据的困难。实验经济学研究在很大程度上专注于简单的选择环境。最近，人工智能界以两种方式为该努力做出了贡献：考虑大型语言模型是否可以代替人类在上述简单选择预测环境中，以及通过机器学习视角研究更复杂但仍严格的实验经济学环境，包括不完全信息、重复博弈和基于自然语言交流的说服游戏。这引发了一个重要的灵感：大型语言模型是否能够完全模拟经济环境，并生成用于高效人类选择预测的数据，替代复杂的经济实验室研究？我们在这个主题上开创了研究，并展示了其可行性。特别是，我们表明仅在大型语言模型生成的数据上训练的模型可以有效地进行预测。

    Economic choice prediction is an essential challenging task, often constrained by the difficulties in acquiring human choice data. Indeed, experimental economics studies had focused mostly on simple choice settings. The AI community has recently contributed to that effort in two ways: considering whether LLMs can substitute for humans in the above-mentioned simple choice prediction settings, and the study through ML lens of more elaborated but still rigorous experimental economics settings, employing incomplete information, repetitive play, and natural language communication, notably language-based persuasion games. This leaves us with a major inspiration: can LLMs be used to fully simulate the economic environment and generate data for efficient human choice prediction, substituting for the elaborated economic lab studies? We pioneer the study of this subject, demonstrating its feasibility. In particular, we show that a model trained solely on LLM-generated data can effectively predic
    
[^5]: LSAP: 重新思考GAN潜空间中反演的保真度、感知和可编辑性

    LSAP: Rethinking Inversion Fidelity, Perception and Editability in GAN Latent Space. (arXiv:2209.12746v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2209.12746](http://arxiv.org/abs/2209.12746)

    LSAP通过对潜空间实现对齐解决了反演和编辑结果中保真度、感知和可编辑性的问题，使得在保留重建保真度的前提下具有更好的感知和可编辑性。

    

    随着方法的发展，反演主要分为两个步骤。第一步是图像嵌入，在这个步骤中，编码器或者优化过程嵌入图像以获取相应的潜在码。之后，第二步旨在改善反演和编辑结果，我们称之为结果细化。尽管第二步显著提高了保真度，但感知和可编辑性几乎没有改变，深度依赖于在第一步中获得的反向潜在码。因此，重要的问题是在保留重建保真度的同时获得具有更好感知和可编辑性的潜在码。在这项工作中，我们首先指出这两个特征与反向码与合成分布的对齐（或不对齐）程度有关。然后，我们提出了潜空间对齐反演范例（LSAP），其中包括评估指标和解决此问题的解决方法。具体而言，我们引入了标准化风格空间（$\mathcal{S^N}$）和标准化内容空间（$\mathcal{C^N}$），分别在风格和内容上对齐正向和负向潜在码和合成分布。 LSAP在各种任务中都取得了最先进的结果，例如图像编辑、图像转换和图像合成。此外，我们证明了LSAP具有比以前方法更好的特性，如改进的可编辑性、视觉质量和更少的模式崩塌。

    As the methods evolve, inversion is mainly divided into two steps. The first step is Image Embedding, in which an encoder or optimization process embeds images to get the corresponding latent codes. Afterward, the second step aims to refine the inversion and editing results, which we named Result Refinement. Although the second step significantly improves fidelity, perception and editability are almost unchanged, deeply dependent on inverse latent codes attained in the first step. Therefore, a crucial problem is gaining the latent codes with better perception and editability while retaining the reconstruction fidelity. In this work, we first point out that these two characteristics are related to the degree of alignment (or disalignment) of the inverse codes with the synthetic distribution. Then, we propose Latent Space Alignment Inversion Paradigm (LSAP), which consists of evaluation metric and solution for this problem. Specifically, we introduce Normalized Style Space ($\mathcal{S^N
    

