# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Attribution Regularization for Multimodal Paradigms](https://arxiv.org/abs/2404.02359) | 提出一种新的正则化项，鼓励多模态模型有效利用所有模态信息，以解决多模态学习中单模态模型优于多模态模型的问题。 |
| [^2] | [Semantic Augmentation in Images using Language](https://arxiv.org/abs/2404.02353) | 深度学习模型需要大规模标记数据集，本文提出利用生成图像增强数据集以改进模型跨领域泛化能力。 |
| [^3] | [Geometry and Stability of Supervised Learning Problems](https://arxiv.org/abs/2403.01660) | 引入了监督学习问题之间的风险距离概念，通过风险距离可以量化问题的稳定性变化，并探索了监督学习问题空间的几何结构。 |

# 详细

[^1]: 多模态范式的归因正则化

    Attribution Regularization for Multimodal Paradigms

    [https://arxiv.org/abs/2404.02359](https://arxiv.org/abs/2404.02359)

    提出一种新的正则化项，鼓励多模态模型有效利用所有模态信息，以解决多模态学习中单模态模型优于多模态模型的问题。

    

    多模态机器学习近年来受到广泛关注，因为它能整合多个模态的信息以增强学习和决策过程。然而，通常观察到单模态模型优于多模态模型，尽管后者可以访问更丰富的信息。此外，单个模态的影响常常主导决策过程，导致性能不佳。这个研究项目旨在通过提出一种新颖的正则化项来解决这些挑战，该项鼓励多模态模型在做出决策时有效利用所有模态的信息。该项目的重点在于视频-音频领域，尽管所提出的正则化技术在涉及多个模态的体现AI研究中具有广泛应用前景。通过利用这种正则化项，提出的方法

    arXiv:2404.02359v1 Announce Type: new  Abstract: Multimodal machine learning has gained significant attention in recent years due to its potential for integrating information from multiple modalities to enhance learning and decision-making processes. However, it is commonly observed that unimodal models outperform multimodal models, despite the latter having access to richer information. Additionally, the influence of a single modality often dominates the decision-making process, resulting in suboptimal performance. This research project aims to address these challenges by proposing a novel regularization term that encourages multimodal models to effectively utilize information from all modalities when making decisions. The focus of this project lies in the video-audio domain, although the proposed regularization technique holds promise for broader applications in embodied AI research, where multiple modalities are involved. By leveraging this regularization term, the proposed approach
    
[^2]: 利用语言在图像中进行语义增强

    Semantic Augmentation in Images using Language

    [https://arxiv.org/abs/2404.02353](https://arxiv.org/abs/2404.02353)

    深度学习模型需要大规模标记数据集，本文提出利用生成图像增强数据集以改进模型跨领域泛化能力。

    

    深度学习模型需要非常庞大的标记数据集进行监督学习，缺乏这些数据集会导致过拟合并限制其泛化到现实世界示例的能力。最近扩散模型的进展使得能够基于文本输入生成逼真的图像。利用用于训练这些扩散模型的大规模数据集，我们提出一种利用生成的图像来增强现有数据集的技术。本文探讨了各种有效数据增强策略，以提高深度学习模型的跨领域泛化能力。

    arXiv:2404.02353v1 Announce Type: cross  Abstract: Deep Learning models are incredibly data-hungry and require very large labeled datasets for supervised learning. As a consequence, these models often suffer from overfitting, limiting their ability to generalize to real-world examples. Recent advancements in diffusion models have enabled the generation of photorealistic images based on textual inputs. Leveraging the substantial datasets used to train these diffusion models, we propose a technique to utilize generated images to augment existing datasets. This paper explores various strategies for effective data augmentation to improve the out-of-domain generalization capabilities of deep learning models.
    
[^3]: 监督学习问题的几何和稳定性

    Geometry and Stability of Supervised Learning Problems

    [https://arxiv.org/abs/2403.01660](https://arxiv.org/abs/2403.01660)

    引入了监督学习问题之间的风险距离概念，通过风险距离可以量化问题的稳定性变化，并探索了监督学习问题空间的几何结构。

    

    我们引入了一种监督学习问题之间的距离概念，我们称之为风险距离。这种以最优传输为灵感的距离促进了稳定性结果；我们可以通过限制这些修改可以将问题移动多少来量化诸如采样偏差、噪声、有限数据和逼近等问题在风险距离下如何改变给定问题。在建立了距离之后，我们探索了产生的监督学习问题空间的几何结构，提供了明确的测地线并证明分类问题集在更大类的问题中是密集的。我们还提供了风险距离的两个变体：一个在问题的预测变量上结合了指定的权重，另一个对问题的风险景观轮廓更为敏感。

    arXiv:2403.01660v1 Announce Type: new  Abstract: We introduce a notion of distance between supervised learning problems, which we call the Risk distance. This optimal-transport-inspired distance facilitates stability results; one can quantify how seriously issues like sampling bias, noise, limited data, and approximations might change a given problem by bounding how much these modifications can move the problem under the Risk distance. With the distance established, we explore the geometry of the resulting space of supervised learning problems, providing explicit geodesics and proving that the set of classification problems is dense in a larger class of problems. We also provide two variants of the Risk distance: one that incorporates specified weights on a problem's predictors, and one that is more sensitive to the contours of a problem's risk landscape.
    

