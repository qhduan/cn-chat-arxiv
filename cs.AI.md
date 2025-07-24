# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Multi-Level Explanations for Generative Language Models](https://arxiv.org/abs/2403.14459) | 本文提出了一个名为MExGen的通用框架，通过引入标量化概念和多级方法处理生成式语言模型的挑战，证明可以提供更贴近本地的解释。 |
| [^2] | [A Large Language Model Enhanced Sequential Recommender for Joint Video and Comment Recommendation](https://arxiv.org/abs/2403.13574) | 提出了一个新颖的推荐方法LSVCR，通过融合用户与视频和评论的交互历史，联合进行个性化视频和评论推荐 |
| [^3] | [From DDMs to DNNs: Using process data and models of decision-making to improve human-AI interactions.](http://arxiv.org/abs/2308.15225) | 本论文提出利用决策过程数据和模型改善人工智能与人类之间的交互。通过详细描述决策过程和建立决策演变模型，可以揭示潜在的偏好，同时追踪决策过程的数据可以提供重要信息，从而改善人工智能的预测能力。 |
| [^4] | [ACMP: Allen-Cahn Message Passing for Graph Neural Networks with Particle Phase Transition.](http://arxiv.org/abs/2206.05437) | 本文提出了一种基于ACMP的图神经网络模型，它可以通过具有吸引力和排斥力的相互作用粒子系统进行消息传递传播，克服了GNN过度平滑问题，将网络深度推到100层，并在基准数据集上实现了最先进的节点分类和图匹配性能。 |

# 详细

[^1]: 生成式语言模型的多级解释

    Multi-Level Explanations for Generative Language Models

    [https://arxiv.org/abs/2403.14459](https://arxiv.org/abs/2403.14459)

    本文提出了一个名为MExGen的通用框架，通过引入标量化概念和多级方法处理生成式语言模型的挑战，证明可以提供更贴近本地的解释。

    

    基于扰动的解释方法，如LIME和SHAP，通常应用于文本分类。本文关注它们如何扩展到生成式语言模型。为了解决文本作为输出和长文本输入的挑战，我们提出了一个名为MExGen的通用框架，可以用不同的归因算法实例化。为了处理文本输出，我们引入了将文本映射到实数的标量化概念，并探讨了多种可能性。为了处理长输入，我们采用多级方法，从粗粒度到细粒度，重点关注具有模型查询线性缩放的算法。我们对基于扰动的归因方法进行了系统评估，包括自动化和人工评估，用于摘要和基于上下文的问答。结果表明，我们的框架可以提供更加贴近本地的生成式输出解释。

    arXiv:2403.14459v1 Announce Type: cross  Abstract: Perturbation-based explanation methods such as LIME and SHAP are commonly applied to text classification. This work focuses on their extension to generative language models. To address the challenges of text as output and long text inputs, we propose a general framework called MExGen that can be instantiated with different attribution algorithms. To handle text output, we introduce the notion of scalarizers for mapping text to real numbers and investigate multiple possibilities. To handle long inputs, we take a multi-level approach, proceeding from coarser levels of granularity to finer ones, and focus on algorithms with linear scaling in model queries. We conduct a systematic evaluation, both automated and human, of perturbation-based attribution methods for summarization and context-grounded question answering. The results show that our framework can provide more locally faithful explanations of generated outputs.
    
[^2]: 一个大型语言模型增强的序列推荐器，用于联合视频和评论推荐

    A Large Language Model Enhanced Sequential Recommender for Joint Video and Comment Recommendation

    [https://arxiv.org/abs/2403.13574](https://arxiv.org/abs/2403.13574)

    提出了一个新颖的推荐方法LSVCR，通过融合用户与视频和评论的交互历史，联合进行个性化视频和评论推荐

    

    在在线视频平台上，阅读或撰写有趣视频的评论已经成为视频观看体验中不可或缺的一部分。然而，现有视频推荐系统主要对用户与视频的交互行为进行建模，缺乏对评论在用户行为建模中的考虑。本文提出了一种名为LSVCR的新颖推荐方法，通过利用用户与视频和评论的交互历史，共同进行个性化视频和评论推荐。具体而言，我们的方法由两个关键组件组成，即序列推荐（SR）模型和补充大型语言模型（LLM）推荐器。SR模型作为我们方法的主要推荐骨干（在部署中保留），可实现高效的用户偏好建模。与此同时，我们利用LLM推荐器作为一个补充组件（在部署中丢弃），以更好地捕捉潜在

    arXiv:2403.13574v1 Announce Type: new  Abstract: In online video platforms, reading or writing comments on interesting videos has become an essential part of the video watching experience. However, existing video recommender systems mainly model users' interaction behaviors with videos, lacking consideration of comments in user behavior modeling. In this paper, we propose a novel recommendation approach called LSVCR by leveraging user interaction histories with both videos and comments, so as to jointly conduct personalized video and comment recommendation. Specifically, our approach consists of two key components, namely sequential recommendation (SR) model and supplemental large language model (LLM) recommender. The SR model serves as the primary recommendation backbone (retained in deployment) of our approach, allowing for efficient user preference modeling. Meanwhile, we leverage the LLM recommender as a supplemental component (discarded in deployment) to better capture underlying 
    
[^3]: 从DDMs到DNNs：利用决策过程的数据和模型来改善人工智能与人类之间的交互

    From DDMs to DNNs: Using process data and models of decision-making to improve human-AI interactions. (arXiv:2308.15225v1 [q-bio.NC])

    [http://arxiv.org/abs/2308.15225](http://arxiv.org/abs/2308.15225)

    本论文提出利用决策过程数据和模型改善人工智能与人类之间的交互。通过详细描述决策过程和建立决策演变模型，可以揭示潜在的偏好，同时追踪决策过程的数据可以提供重要信息，从而改善人工智能的预测能力。

    

    在过去的几十年中，认知神经科学家和行为经济学家已经认识到详细描述决策过程和建立决策随时间演变的模型的价值。例如，决策所需的时间可以揭示一个个体真正的潜在偏好，而不仅仅是决策本身。类似地，追踪决策过程的数据，如眼动或神经记录，包含了关键的信息，即使没有达成决策也可以被利用。在这里，我们认为人工智能研究应更加关注决策如何随时间演变以及如何融入相关的过程数据来改善人工智能的预测，特别是在人与人工智能之间的交互中。首先，我们介绍了一个非常成熟的计算框架，该框架认为决策是从杂音累积的证据中产生的，并介绍了相关的心理学、神经科学和经济学的实证研究。

    Over the past decades, cognitive neuroscientists and behavioral economists have recognized the value of describing the process of decision making in detail and modeling the emergence of decisions over time. For example, the time it takes to decide can reveal more about an agents true hidden preferences than only the decision itself. Similarly, data that track the ongoing decision process such as eye movements or neural recordings contain critical information that can be exploited, even if no decision is made. Here, we argue that artificial intelligence (AI) research would benefit from a stronger focus on insights about how decisions emerge over time and incorporate related process data to improve AI predictions in general and human-AI interactions in particular. First, we introduce a highly established computational framework that assumes decisions to emerge from the noisy accumulation of evidence, and we present related empirical work in psychology, neuroscience, and economics. Next, 
    
[^4]: ACMP: Allen-Cahn信息传递用于带有物质相变的图神经网络

    ACMP: Allen-Cahn Message Passing for Graph Neural Networks with Particle Phase Transition. (arXiv:2206.05437v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2206.05437](http://arxiv.org/abs/2206.05437)

    本文提出了一种基于ACMP的图神经网络模型，它可以通过具有吸引力和排斥力的相互作用粒子系统进行消息传递传播，克服了GNN过度平滑问题，将网络深度推到100层，并在基准数据集上实现了最先进的节点分类和图匹配性能。

    

    神经消息传递是基于图结构数据的特征提取单元，考虑从一层到下一层的网络传播中的相邻节点特征。我们通过具有吸引力和排斥力的相互作用粒子系统来建模这种过程，并在相变建模中引入Allen-Cahn力。系统的动力学是一种反应扩散过程，可以将粒子分离而不会扩散。这引出了一种Allen-Cahn信息传递(ACMP)用于图神经网络，其中粒子系统解的数值迭代构成了消息传递传播。ACMP具有简单的实现和神经ODE求解器，可以将网络深度推到100层，并具有理论上证明的Dirichlet能量严格正下界。因此，它提供了一种深度模型的GNN，避免了常见的GNN过度平滑问题。使用ACMP的GNN在基准数据集上实现了实际节点分类和图匹配任务的最先进性能。

    Neural message passing is a basic feature extraction unit for graph-structured data considering neighboring node features in network propagation from one layer to the next. We model such process by an interacting particle system with attractive and repulsive forces and the Allen-Cahn force arising in the modeling of phase transition. The dynamics of the system is a reaction-diffusion process which can separate particles without blowing up. This induces an Allen-Cahn message passing (ACMP) for graph neural networks where the numerical iteration for the particle system solution constitutes the message passing propagation. ACMP which has a simple implementation with a neural ODE solver can propel the network depth up to one hundred of layers with theoretically proven strictly positive lower bound of the Dirichlet energy. It thus provides a deep model of GNNs circumventing the common GNN problem of oversmoothing. GNNs with ACMP achieve state of the art performance for real-world node class
    

