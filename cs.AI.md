# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Is my Data in your AI Model? Membership Inference Test with Application to Face Images](https://arxiv.org/abs/2402.09225) | This paper introduces a novel approach called Membership Inference Test (MINT) to empirically assess if specific data was used during the training of AI models. Two MINT architectures based on MLP and CNN are proposed and evaluated on a challenging face recognition task, achieving promising results with up to 90% accuracy. |
| [^2] | [PuriDefense: Randomized Local Implicit Adversarial Purification for Defending Black-box Query-based Attacks.](http://arxiv.org/abs/2401.10586) | PuriDefense是一种高效的防御机制，通过使用轻量级净化模型进行随机路径净化，减缓基于查询的攻击的收敛速度，并有效防御黑盒基于查询的攻击。 |
| [^3] | [Continual Learning as Computationally Constrained Reinforcement Learning.](http://arxiv.org/abs/2307.04345) | 本文研究了连续学习作为计算受限的强化学习的主题，提出了一个框架和一套工具来解决人工智能领域长期以来的挑战并促进进一步的研究。 |
| [^4] | [StyleNAT: Giving Each Head a New Perspective.](http://arxiv.org/abs/2211.05770) | StyleNAT是一个新的基于transformer的图像生成框架，通过使用邻域注意力（NA）来捕捉局部和全局信息，能够高效灵活地适应不同的数据集，并在FFHQ-256上取得了新的最佳结果。 |

# 详细

[^1]: 我的数据在你的AI模型中吗？通过应用于人脸图像的成员推断测试

    Is my Data in your AI Model? Membership Inference Test with Application to Face Images

    [https://arxiv.org/abs/2402.09225](https://arxiv.org/abs/2402.09225)

    This paper introduces a novel approach called Membership Inference Test (MINT) to empirically assess if specific data was used during the training of AI models. Two MINT architectures based on MLP and CNN are proposed and evaluated on a challenging face recognition task, achieving promising results with up to 90% accuracy.

    

    这篇论文介绍了成员推断测试（MINT），一种用于经验性评估特定数据是否被用于训练人工智能（AI）模型的新方法。具体而言，我们提出了两种新颖的MINT架构，旨在学习在经过审计的模型暴露于其训练过程中使用的数据时出现的不同激活模式。第一个架构基于多层感知机（MLP）网络，第二个基于卷积神经网络（CNN）。所提出的MINT架构在具有挑战性的人脸识别任务上进行评估，考虑了三种最先进的人脸识别模型。使用六个公开可用的数据库进行实验，总共包含超过2200万张人脸图像。根据可用的AI模型测试的上下文，考虑了不同的实验场景。有希望的结果达到了90%的准确率。

    arXiv:2402.09225v1 Announce Type: cross Abstract: This paper introduces the Membership Inference Test (MINT), a novel approach that aims to empirically assess if specific data was used during the training of Artificial Intelligence (AI) models. Specifically, we propose two novel MINT architectures designed to learn the distinct activation patterns that emerge when an audited model is exposed to data used during its training process. The first architecture is based on a Multilayer Perceptron (MLP) network and the second one is based on Convolutional Neural Networks (CNNs). The proposed MINT architectures are evaluated on a challenging face recognition task, considering three state-of-the-art face recognition models. Experiments are carried out using six publicly available databases, comprising over 22 million face images in total. Also, different experimental scenarios are considered depending on the context available of the AI model to test. Promising results, up to 90% accuracy, are a
    
[^2]: PuriDefense：用于防御黑盒基于查询的攻击的随机局部隐式对抗净化

    PuriDefense: Randomized Local Implicit Adversarial Purification for Defending Black-box Query-based Attacks. (arXiv:2401.10586v1 [cs.CR])

    [http://arxiv.org/abs/2401.10586](http://arxiv.org/abs/2401.10586)

    PuriDefense是一种高效的防御机制，通过使用轻量级净化模型进行随机路径净化，减缓基于查询的攻击的收敛速度，并有效防御黑盒基于查询的攻击。

    

    黑盒基于查询的攻击对机器学习作为服务系统构成重大威胁，因为它们可以生成对抗样本而不需要访问目标模型的架构和参数。传统的防御机制，如对抗训练、梯度掩盖和输入转换，要么带来巨大的计算成本，要么损害非对抗输入的测试准确性。为了应对这些挑战，我们提出了一种高效的防御机制PuriDefense，在低推理成本的级别上使用轻量级净化模型的随机路径净化。这些模型利用局部隐式函数并重建自然图像流形。我们的理论分析表明，这种方法通过将随机性纳入净化过程来减缓基于查询的攻击的收敛速度。对CIFAR-10和ImageNet的大量实验验证了我们提出的净化器防御的有效性。

    Black-box query-based attacks constitute significant threats to Machine Learning as a Service (MLaaS) systems since they can generate adversarial examples without accessing the target model's architecture and parameters. Traditional defense mechanisms, such as adversarial training, gradient masking, and input transformations, either impose substantial computational costs or compromise the test accuracy of non-adversarial inputs. To address these challenges, we propose an efficient defense mechanism, PuriDefense, that employs random patch-wise purifications with an ensemble of lightweight purification models at a low level of inference cost. These models leverage the local implicit function and rebuild the natural image manifold. Our theoretical analysis suggests that this approach slows down the convergence of query-based attacks by incorporating randomness into purifications. Extensive experiments on CIFAR-10 and ImageNet validate the effectiveness of our proposed purifier-based defen
    
[^3]: 连续学习作为计算受限的强化学习

    Continual Learning as Computationally Constrained Reinforcement Learning. (arXiv:2307.04345v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2307.04345](http://arxiv.org/abs/2307.04345)

    本文研究了连续学习作为计算受限的强化学习的主题，提出了一个框架和一套工具来解决人工智能领域长期以来的挑战并促进进一步的研究。

    

    一种能够在漫长的生命周期内高效积累知识并发展越来越复杂技能的智能体可以推动人工智能能力的前沿。连续学习这一长期以来一直是人工智能领域的挑战，本文介绍了关于连续学习的概念并提出了一个框架和一套工具，以促进进一步的研究。

    An agent that efficiently accumulates knowledge to develop increasingly sophisticated skills over a long lifetime could advance the frontier of artificial intelligence capabilities. The design of such agents, which remains a long-standing challenge of artificial intelligence, is addressed by the subject of continual learning. This monograph clarifies and formalizes concepts of continual learning, introducing a framework and set of tools to stimulate further research.
    
[^4]: StyleNAT：给每个头部一个新的视角

    StyleNAT: Giving Each Head a New Perspective. (arXiv:2211.05770v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2211.05770](http://arxiv.org/abs/2211.05770)

    StyleNAT是一个新的基于transformer的图像生成框架，通过使用邻域注意力（NA）来捕捉局部和全局信息，能够高效灵活地适应不同的数据集，并在FFHQ-256上取得了新的最佳结果。

    

    图像生成一直是一个既期望又具有挑战性的任务，以高效的方式执行生成任务同样困难。通常，研究人员试图创建一个“一刀切”的生成器，在参数空间中，即使是截然不同的数据集，也有很少的差异。在这里，我们提出了一种新的基于transformer的框架，称为StyleNAT，旨在实现高质量的图像生成，并具有卓越的效率和灵活性。在我们的模型核心是一个精心设计的框架，它将注意力头部划分为捕捉局部和全局信息的方式，这是通过使用邻域注意力（NA）实现的。由于不同的头部能够关注不同的感受野，模型能够更好地结合这些信息，并以高度灵活的方式适应手头的数据。StyleNAT在FFHQ-256上获得了新的SOTA FID得分2.046 ，击败了以卷积模型（如StyleGAN-XL）和transformer模型（如HIT）为基础的先前方法。

    Image generation has been a long sought-after but challenging task, and performing the generation task in an efficient manner is similarly difficult. Often researchers attempt to create a "one size fits all" generator, where there are few differences in the parameter space for drastically different datasets. Herein, we present a new transformer-based framework, dubbed StyleNAT, targeting high-quality image generation with superior efficiency and flexibility. At the core of our model, is a carefully designed framework that partitions attention heads to capture local and global information, which is achieved through using Neighborhood Attention (NA). With different heads able to pay attention to varying receptive fields, the model is able to better combine this information, and adapt, in a highly flexible manner, to the data at hand. StyleNAT attains a new SOTA FID score on FFHQ-256 with 2.046, beating prior arts with convolutional models such as StyleGAN-XL and transformers such as HIT 
    

