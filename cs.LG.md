# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Towards Leveraging AutoML for Sustainable Deep Learning: A Multi-Objective HPO Approach on Deep Shift Neural Networks](https://arxiv.org/abs/2404.01965) | 该研究旨在利用AutoML技术最大化Deep Shift神经网络性能并最小化资源消耗，提出了结合多保真度HPO和多目标优化的方法，实验证明该方法在提高准确率的同时降低了计算复杂性。 |
| [^2] | [Moderating Illicit Online Image Promotion for Unsafe User-Generated Content Games Using Large Vision-Language Models](https://arxiv.org/abs/2403.18957) | 该研究旨在调查不安全用户生成内容游戏中的违法推广威胁，收集了一组包含性暴力和暴力内容的真实图像数据集。 |
| [^3] | [Manifold GCN: Diffusion-based Convolutional Neural Network for Manifold-valued Graphs.](http://arxiv.org/abs/2401.14381) | 本研究提出了两个用于具有流形值特征的图的神经网络层。这些层具有对节点排列和特征流形的等变性，并在深度学习任务中显示出有益的归纳偏差。 |

# 详细

[^1]: 旨在利用AutoML实现可持续深度学习：基于Deep Shift神经网络的多目标HPO方法

    Towards Leveraging AutoML for Sustainable Deep Learning: A Multi-Objective HPO Approach on Deep Shift Neural Networks

    [https://arxiv.org/abs/2404.01965](https://arxiv.org/abs/2404.01965)

    该研究旨在利用AutoML技术最大化Deep Shift神经网络性能并最小化资源消耗，提出了结合多保真度HPO和多目标优化的方法，实验证明该方法在提高准确率的同时降低了计算复杂性。

    

    深度学习（DL）通过从大型数据集中提取复杂模式推动了各个领域的发展。然而，DL模型的计算需求带来了环境和资源挑战。Deep Shift神经网络（DSNN）利用shift操作减少推理时的计算复杂性，为此提供了解决方案。通过借鉴标准DNN的见解，我们有兴趣通过AutoML技术充分发挥DSNN的潜力。我们研究了超参数优化（HPO）对于最大化DSNN性能同时最小化资源消耗的影响。由于将准确性和能耗作为可能互补目标结合的多目标（MO）优化，我们建议将最先进的多保真度（MF）HPO与多目标优化相结合。实验结果证明了我们方法的有效性，得到了准确率超过80％且计算低耗的模型。

    arXiv:2404.01965v1 Announce Type: cross  Abstract: Deep Learning (DL) has advanced various fields by extracting complex patterns from large datasets. However, the computational demands of DL models pose environmental and resource challenges. Deep shift neural networks (DSNNs) offer a solution by leveraging shift operations to reduce computational complexity at inference. Following the insights from standard DNNs, we are interested in leveraging the full potential of DSNNs by means of AutoML techniques. We study the impact of hyperparameter optimization (HPO) to maximize DSNN performance while minimizing resource consumption. Since this combines multi-objective (MO) optimization with accuracy and energy consumption as potentially complementary objectives, we propose to combine state-of-the-art multi-fidelity (MF) HPO with multi-objective optimization. Experimental results demonstrate the effectiveness of our approach, resulting in models with over 80\% in accuracy and low computational 
    
[^2]: 利用大规模视觉语言模型调节不安全用户生成内容游戏中的违法在线图片推广

    Moderating Illicit Online Image Promotion for Unsafe User-Generated Content Games Using Large Vision-Language Models

    [https://arxiv.org/abs/2403.18957](https://arxiv.org/abs/2403.18957)

    该研究旨在调查不安全用户生成内容游戏中的违法推广威胁，收集了一组包含性暴力和暴力内容的真实图像数据集。

    

    在线用户生成内容游戏（UGCGs）在儿童和青少年中越来越受欢迎，用于社交互动和更有创意的在线娱乐。然而，它们存在着更高的暴露不良内容的风险，引发了人们对儿童和青少年在线安全的日益关注。我们采取了第一步研究对不安全UGCGs的违法推广进行威胁性分析。我们收集了一组现实世界数据集，包括2,924张展示不同性暴力和暴力内容的图像，这些内容被游戏创建者用于推广UGCGs。

    arXiv:2403.18957v1 Announce Type: cross  Abstract: Online user-generated content games (UGCGs) are increasingly popular among children and adolescents for social interaction and more creative online entertainment. However, they pose a heightened risk of exposure to explicit content, raising growing concerns for the online safety of children and adolescents. Despite these concerns, few studies have addressed the issue of illicit image-based promotions of unsafe UGCGs on social media, which can inadvertently attract young users. This challenge arises from the difficulty of obtaining comprehensive training data for UGCG images and the unique nature of these images, which differ from traditional unsafe content. In this work, we take the first step towards studying the threat of illicit promotions of unsafe UGCGs. We collect a real-world dataset comprising 2,924 images that display diverse sexually explicit and violent content used to promote UGCGs by their game creators. Our in-depth studi
    
[^3]: 面向流形值图的扩散卷积神经网络：多重难题图神经网络层

    Manifold GCN: Diffusion-based Convolutional Neural Network for Manifold-valued Graphs. (arXiv:2401.14381v1 [cs.LG])

    [http://arxiv.org/abs/2401.14381](http://arxiv.org/abs/2401.14381)

    本研究提出了两个用于具有流形值特征的图的神经网络层。这些层具有对节点排列和特征流形的等变性，并在深度学习任务中显示出有益的归纳偏差。

    

    我们提出了两种用于具有Riemannian流形特征的图上的图神经网络层。第一，基于流形值图的扩散方程，我们构建了一个扩散层，可以应用于任意数量的节点和图连接模式。第二，我们通过将向量神经元框架的思想转化到我们的一般设置中，建立了一个切线多层感知器。这两个层对节点排列和特征流形的等变具有响应，这些特性在许多深度学习任务中已被证明具有有益的归纳偏差。我们在合成数据上以及在右侧海马三角网格上分类阿尔茨海默病的数值实例表明我们建立的层具有非常好的性能。

    We propose two graph neural network layers for graphs with features in a Riemannian manifold. First, based on a manifold-valued graph diffusion equation, we construct a diffusion layer that can be applied to an arbitrary number of nodes and graph connectivity patterns. Second, we model a tangent multilayer perceptron by transferring ideas from the vector neuron framework to our general setting. Both layers are equivariant with respect to node permutations and isometries of the feature manifold. These properties have been shown to lead to a beneficial inductive bias in many deep learning tasks. Numerical examples on synthetic data as well as on triangle meshes of the right hippocampus to classify Alzheimer's disease demonstrate the very good performance of our layers.
    

