# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Plug-in Diffusion Model for Sequential Recommendation.](http://arxiv.org/abs/2401.02913) | 本论文提出了一个称为PDRec的插件扩散模型，用于顺序推荐。该模型利用扩散模型来充分利用用户对所有项目的偏好，解决数据稀疏问题。 |
| [^2] | [Let's Get It Started: Fostering the Discoverability of New Releases on Deezer.](http://arxiv.org/abs/2401.02827) | 本文介绍了在Deezer音乐服务上促进新发布内容发现能力的最新举措，包括个性化推荐、冷启动嵌入和情境强化学习算法等。通过在线实验的支持，我们展示了这些举措在提高推荐质量和新发布内容曝光方面的优势。 |
| [^3] | [DocGraphLM: Documental Graph Language Model for Information Extraction.](http://arxiv.org/abs/2401.02823) | 本文介绍了一个名为DocGraphLM的框架，它将预训练的语言模型与图语义相结合，通过联合编码器架构表示文档，并使用一种新颖的链接预测方法重构文档图。实验证明，采用图特征可以在信息抽取和问题回答任务上取得一致的改进，并且在训练过程中加速了收敛过程。 |

# 详细

[^1]: 为顺序推荐设计的插件扩散模型

    Plug-in Diffusion Model for Sequential Recommendation. (arXiv:2401.02913v1 [cs.IR])

    [http://arxiv.org/abs/2401.02913](http://arxiv.org/abs/2401.02913)

    本论文提出了一个称为PDRec的插件扩散模型，用于顺序推荐。该模型利用扩散模型来充分利用用户对所有项目的偏好，解决数据稀疏问题。

    

    先驱性的研究已经验证了扩散模型在探索推荐中信息不确定性方面的有效性。考虑到推荐和图像合成任务之间的差异，现有方法已经针对扩散和反向过程进行了定制优化。然而，这些方法通常使用语料库中最高分的项目来预测用户的兴趣，导致忽视了其他项目中包含的用户广义偏好，从而仍然受限于数据稀疏问题。为了解决这个问题，本文提出了一种新颖的插件扩散模型用于推荐（PDRec）框架，该框架将扩散模型作为灵活的插件，共同充分利用扩散生成的用户对所有项目的偏好。具体而言，PDRec首先通过时间间隔扩散模型推断用户对所有项目的动态偏好，并提出了一种历史行为重新加权（HBR）机制来识别用户的广义偏好。

    Pioneering efforts have verified the effectiveness of the diffusion models in exploring the informative uncertainty for recommendation. Considering the difference between recommendation and image synthesis tasks, existing methods have undertaken tailored refinements to the diffusion and reverse process. However, these approaches typically use the highest-score item in corpus for user interest prediction, leading to the ignorance of the user's generalized preference contained within other items, thereby remaining constrained by the data sparsity issue. To address this issue, this paper presents a novel Plug-in Diffusion Model for Recommendation (PDRec) framework, which employs the diffusion model as a flexible plugin to jointly take full advantage of the diffusion-generating user preferences on all items. Specifically, PDRec first infers the users' dynamic preferences on all items via a time-interval diffusion model and proposes a Historical Behavior Reweighting (HBR) mechanism to ident
    
[^2]: 让我们开始吧：促进Deezer音乐服务上新发布内容的发现能力

    Let's Get It Started: Fostering the Discoverability of New Releases on Deezer. (arXiv:2401.02827v1 [cs.IR])

    [http://arxiv.org/abs/2401.02827](http://arxiv.org/abs/2401.02827)

    本文介绍了在Deezer音乐服务上促进新发布内容发现能力的最新举措，包括个性化推荐、冷启动嵌入和情境强化学习算法等。通过在线实验的支持，我们展示了这些举措在提高推荐质量和新发布内容曝光方面的优势。

    

    本文介绍了我们最近在音乐流媒体服务Deezer上促进新发布内容发现能力方面的举措。在介绍了我们针对新发布内容的搜索和推荐功能之后，我们概述了我们从编辑推荐向个性化推荐的转变，包括使用冷启动嵌入和情境强化学习算法。通过在线实验的支持，我们讨论了这一转变在推荐质量和新发布内容的曝光方面的优势。

    This paper presents our recent initiatives to foster the discoverability of new releases on the music streaming service Deezer. After introducing our search and recommendation features dedicated to new releases, we outline our shift from editorial to personalized release suggestions using cold start embeddings and contextual bandits. Backed by online experiments, we discuss the advantages of this shift in terms of recommendation quality and exposure of new releases on the service.
    
[^3]: DocGraphLM：信息抽取的文档图语言模型

    DocGraphLM: Documental Graph Language Model for Information Extraction. (arXiv:2401.02823v1 [cs.CL])

    [http://arxiv.org/abs/2401.02823](http://arxiv.org/abs/2401.02823)

    本文介绍了一个名为DocGraphLM的框架，它将预训练的语言模型与图语义相结合，通过联合编码器架构表示文档，并使用一种新颖的链接预测方法重构文档图。实验证明，采用图特征可以在信息抽取和问题回答任务上取得一致的改进，并且在训练过程中加速了收敛过程。

    

    在视觉丰富的文档理解(VrDU)方面取得的进展使得可以对具有复杂布局的文档进行信息抽取和问题回答成为可能。出现了两种架构的模式-受LLM启发的基于transformer的模型和图神经网络。在本文中，我们介绍了一种新颖的框架DocGraphLM，它将预训练的语言模型与图语义相结合。为了实现这一目标，我们提出了1)一种联合编码器架构来表示文档，以及2)一种新颖的链接预测方法来重构文档图。DocGraphLM使用一个收敛的联合损失函数来预测节点之间的方向和距离，该损失函数优先考虑邻域恢复并减轻远程节点检测。我们在三个最先进的数据集上的实验证明，采用图特征可以在信息抽取和问题回答任务上保持一致的改进。此外，我们报告说，尽管仅由构建而来，在训练过程中采用图特征可以加快收敛过程。

    Advances in Visually Rich Document Understanding (VrDU) have enabled information extraction and question answering over documents with complex layouts. Two tropes of architectures have emerged -- transformer-based models inspired by LLMs, and Graph Neural Networks. In this paper, we introduce DocGraphLM, a novel framework that combines pre-trained language models with graph semantics. To achieve this, we propose 1) a joint encoder architecture to represent documents, and 2) a novel link prediction approach to reconstruct document graphs. DocGraphLM predicts both directions and distances between nodes using a convergent joint loss function that prioritizes neighborhood restoration and downweighs distant node detection. Our experiments on three SotA datasets show consistent improvement on IE and QA tasks with the adoption of graph features. Moreover, we report that adopting the graph features accelerates convergence in the learning process during training, despite being solely constructe
    

