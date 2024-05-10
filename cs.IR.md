# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Cobweb: An Incremental and Hierarchical Model of Human-Like Category Learning](https://arxiv.org/abs/2403.03835) | Cobweb是一种类似人类类别学习系统，采用类别效用度量构建分层组织的类似树状结构，能够捕捉心理效应并在单一模型中展现出实例和原型学习的灵活性，为将来研究人类类别学习提供了基础。 |
| [^2] | [REFORM: Removing False Correlation in Multi-level Interaction for CTR Prediction.](http://arxiv.org/abs/2309.14891) | REFORM是一个CTR预测框架，通过两个流式叠加的循环结构利用了多级高阶特征表示，并消除了误关联。 |
| [^3] | [Multi-behavior Recommendation with SVD Graph Neural Networks.](http://arxiv.org/abs/2309.06912) | 本研究提出了一种使用SVD图神经网络进行多行为推荐的模型MB-SVD，通过考虑用户在不同行为下的偏好，改善了推荐效果，同时更好地解决了冷启动问题。 |

# 详细

[^1]: Cobweb：一种增量和分层式的人类类别学习模型

    Cobweb: An Incremental and Hierarchical Model of Human-Like Category Learning

    [https://arxiv.org/abs/2403.03835](https://arxiv.org/abs/2403.03835)

    Cobweb是一种类似人类类别学习系统，采用类别效用度量构建分层组织的类似树状结构，能够捕捉心理效应并在单一模型中展现出实例和原型学习的灵活性，为将来研究人类类别学习提供了基础。

    

    Cobweb是一种类似人类的类别学习系统，与其他增量分类模型不同的是，它利用类别效用度量构建分层组织的类似树状结构。先前的研究表明，Cobweb能够捕捉心理效应，如基本水平、典型性和扇形效应。然而，对Cobweb作为人类分类模型的更广泛评估仍然缺乏。本研究填补了这一空白。它确定了Cobweb与经典的人类类别学习效应的一致性。还探讨了Cobweb展现出在单一模型中既有实例又有原型学习的灵活性。这些发现为将来研究Cobweb作为人类类别学习的综合模型奠定了基础。

    arXiv:2403.03835v1 Announce Type: cross  Abstract: Cobweb, a human like category learning system, differs from other incremental categorization models in constructing hierarchically organized cognitive tree-like structures using the category utility measure. Prior studies have shown that Cobweb can capture psychological effects such as the basic level, typicality, and fan effects. However, a broader evaluation of Cobweb as a model of human categorization remains lacking. The current study addresses this gap. It establishes Cobweb's alignment with classical human category learning effects. It also explores Cobweb's flexibility to exhibit both exemplar and prototype like learning within a single model. These findings set the stage for future research on Cobweb as a comprehensive model of human category learning.
    
[^2]: REFORM: 移除CTR预测中的误关联的多级交互

    REFORM: Removing False Correlation in Multi-level Interaction for CTR Prediction. (arXiv:2309.14891v1 [cs.IR])

    [http://arxiv.org/abs/2309.14891](http://arxiv.org/abs/2309.14891)

    REFORM是一个CTR预测框架，通过两个流式叠加的循环结构利用了多级高阶特征表示，并消除了误关联。

    

    点击率（CTR）预测是在线广告和推荐系统中的关键任务，准确的预测对于用户定位和个性化推荐至关重要。最近的一些前沿方法主要关注复杂的隐式和显式特征交互。然而，这些方法忽视了由混淆因子或选择偏差引起的误关联问题。这个问题在这些交互的复杂性和冗余性下变得更加严重。我们提出了一种CTR预测框架，称为REFORM，在多级特征交互中移除了误关联。所提出的REFORM框架通过两个流式叠加的循环结构利用了大量的多级高阶特征表示，并消除了误关联。该框架有两个关键组成部分：I. 多级叠加循环（MSR）结构使模型能够高效地捕捉到来自特征空间的多样非线性交互。

    Click-through rate (CTR) prediction is a critical task in online advertising and recommendation systems, as accurate predictions are essential for user targeting and personalized recommendations. Most recent cutting-edge methods primarily focus on investigating complex implicit and explicit feature interactions. However, these methods neglect the issue of false correlations caused by confounding factors or selection bias. This problem is further magnified by the complexity and redundancy of these interactions. We propose a CTR prediction framework that removes false correlation in multi-level feature interaction, termed REFORM. The proposed REFORM framework exploits a wide range of multi-level high-order feature representations via a two-stream stacked recurrent structure while eliminating false correlations. The framework has two key components: I. The multi-level stacked recurrent (MSR) structure enables the model to efficiently capture diverse nonlinear interactions from feature spa
    
[^3]: 用SVD图神经网络进行多行为推荐

    Multi-behavior Recommendation with SVD Graph Neural Networks. (arXiv:2309.06912v1 [cs.IR])

    [http://arxiv.org/abs/2309.06912](http://arxiv.org/abs/2309.06912)

    本研究提出了一种使用SVD图神经网络进行多行为推荐的模型MB-SVD，通过考虑用户在不同行为下的偏好，改善了推荐效果，同时更好地解决了冷启动问题。

    

    图神经网络(GNNs)广泛应用于推荐系统领域，为用户提供个性化推荐并取得显著成果。最近，融入对比学习的GNNs在处理推荐系统的稀疏数据问题方面表现出了很大的潜力。然而，现有的对比学习方法在解决冷启动问题和抵抗噪声干扰方面仍然存在限制，尤其是对于多行为推荐。为了缓解上述问题，本研究提出了一种基于GNNs的多行为推荐模型MB-SVD，利用奇异值分解(SVD)图来提高模型性能。具体而言，MB-SVD考虑了用户在不同行为下的偏好，改善了推荐效果，同时更好地解决了冷启动问题。我们的模型引入了一种创新的方法论，将多行为对比学习范式融入到模型中，以提高模型的性能。

    Graph Neural Networks (GNNs) has been extensively employed in the field of recommender systems, offering users personalized recommendations and yielding remarkable outcomes. Recently, GNNs incorporating contrastive learning have demonstrated promising performance in handling sparse data problem of recommendation system. However, existing contrastive learning methods still have limitations in addressing the cold-start problem and resisting noise interference especially for multi-behavior recommendation. To mitigate the aforementioned issues, the present research posits a GNNs based multi-behavior recommendation model MB-SVD that utilizes Singular Value Decomposition (SVD) graphs to enhance model performance. In particular, MB-SVD considers user preferences under different behaviors, improving recommendation effectiveness while better addressing the cold-start problem. Our model introduces an innovative methodology, which subsume multi-behavior contrastive learning paradigm to proficient
    

