# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Feynman Diagrams as Computational Graphs](https://arxiv.org/abs/2403.18840) | 该论文提出了一种用于量子场论中高阶费曼图的计算图表示方法，通过组织成张量操作的分形结构显著减少计算冗余，集成了Taylor-mode自动微分技术，开发了费曼图编译器以优化计算图。 |
| [^2] | [Uncertainty-Aware Explanations Through Probabilistic Self-Explainable Neural Networks](https://arxiv.org/abs/2403.13740) | 本文引入了概率自解释神经网络（Prob-PSENN），通过概率分布取代点估计，实现了更灵活的原型学习，提供了实用的对不确定性的解释。 |
| [^3] | [XpertAI: uncovering model strategies for sub-manifolds](https://arxiv.org/abs/2403.07486) | XpertAI是一个框架，可以将预测策略解开为多个特定范围的子策略，并允许将模型的查询制定为这些子策略的线性组合。 |
| [^4] | [Geometry-Informed Neural Networks](https://arxiv.org/abs/2402.14009) | GINNs提出了一种新颖的几何信息神经网络范式，可以在几何任务中生成多样的解决方案，无需训练数据，采用显式多样性损失以及可微损失来减轻模态坍缩，并在实验中展示了其在各种复杂性场景中的高效性。 |
| [^5] | [Interpretable Generative Adversarial Imitation Learning](https://arxiv.org/abs/2402.10310) | 提出了一种结合了信号时序逻辑（STL）推断和控制合成的新颖仿真学习方法，可以明确表示任务为STL公式，同时通过人为调整STL公式实现对人类知识的纳入和新场景的适应，还采用了生成对抗网络（GAN）启发的训练方法，有效缩小了专家策略和学习策略之间的差距 |
| [^6] | [Improved DDIM Sampling with Moment Matching Gaussian Mixtures.](http://arxiv.org/abs/2311.04938) | 在DDIM框架中使用GMM作为反向转移算子，通过矩匹配可以获得质量更高的样本。在无条件模型和类条件模型上进行了实验，并通过FID和IS指标证明了我们的方法的改进效果。 |
| [^7] | [Exact and Cost-Effective Automated Transformation of Neural Network Controllers to Decision Tree Controllers.](http://arxiv.org/abs/2304.06049) | 本文研究了将基于神经网络的控制器转换为等效软决策树控制器并提出了一种自动且节约成本的转换算法。该方法适用于包括ReLU激活函数在内的离散输出NN控制器，并能够提高形式验证的运行效率。 |

# 详细

[^1]: 费曼图作为计算图的表示

    Feynman Diagrams as Computational Graphs

    [https://arxiv.org/abs/2403.18840](https://arxiv.org/abs/2403.18840)

    该论文提出了一种用于量子场论中高阶费曼图的计算图表示方法，通过组织成张量操作的分形结构显著减少计算冗余，集成了Taylor-mode自动微分技术，开发了费曼图编译器以优化计算图。

    

    我们提出了一种适用于空间、时间、动量和频率领域的高阶费曼图的计算图表示，适用于量子场论（QFT）。利用戴森-施温格方程和树图方程，我们的方法有效地将这些图组织成张量操作的分形结构，显著减少了计算冗余。这种方法不仅简化了复杂图的评估，还促进了场论重整化方案的高效实施，对增强微扰QFT计算至关重要。这一进展的关键在于集成了Taylor-mode自动微分，这是机器学习包中用于在计算图上高效计算高阶导数的关键技术。为了操作化这些概念，我们开发了一个费曼图编译器，优化了各种计算图。

    arXiv:2403.18840v1 Announce Type: cross  Abstract: We propose a computational graph representation of high-order Feynman diagrams in Quantum Field Theory (QFT), applicable to any combination of spatial, temporal, momentum, and frequency domains. Utilizing the Dyson-Schwinger and parquet equations, our approach effectively organizes these diagrams into a fractal structure of tensor operations, significantly reducing computational redundancy. This approach not only streamlines the evaluation of complex diagrams but also facilitates an efficient implementation of the field-theoretic renormalization scheme, crucial for enhancing perturbative QFT calculations. Key to this advancement is the integration of Taylor-mode automatic differentiation, a key technique employed in machine learning packages to compute higher-order derivatives efficiently on computational graphs. To operationalize these concepts, we develop a Feynman diagram compiler that optimizes diagrams for various computational pl
    
[^2]: 通过概率自解释神经网络实现对不确定性的认知

    Uncertainty-Aware Explanations Through Probabilistic Self-Explainable Neural Networks

    [https://arxiv.org/abs/2403.13740](https://arxiv.org/abs/2403.13740)

    本文引入了概率自解释神经网络（Prob-PSENN），通过概率分布取代点估计，实现了更灵活的原型学习，提供了实用的对不确定性的解释。

    

    深度神经网络的不透明性持续限制其可靠性和在高风险应用中的使用。本文介绍了概率自解释神经网络（Prob-PSENN），采用概率分布代替原型的点估计，提供了一种更灵活的原型端到端学习框架。

    arXiv:2403.13740v1 Announce Type: new  Abstract: The lack of transparency of Deep Neural Networks continues to be a limitation that severely undermines their reliability and usage in high-stakes applications. Promising approaches to overcome such limitations are Prototype-Based Self-Explainable Neural Networks (PSENNs), whose predictions rely on the similarity between the input at hand and a set of prototypical representations of the output classes, offering therefore a deep, yet transparent-by-design, architecture. So far, such models have been designed by considering pointwise estimates for the prototypes, which remain fixed after the learning phase of the model. In this paper, we introduce a probabilistic reformulation of PSENNs, called Prob-PSENN, which replaces point estimates for the prototypes with probability distributions over their values. This provides not only a more flexible framework for an end-to-end learning of prototypes, but can also capture the explanatory uncertaint
    
[^3]: XpertAI：揭示子流形的模型策略

    XpertAI: uncovering model strategies for sub-manifolds

    [https://arxiv.org/abs/2403.07486](https://arxiv.org/abs/2403.07486)

    XpertAI是一个框架，可以将预测策略解开为多个特定范围的子策略，并允许将模型的查询制定为这些子策略的线性组合。

    

    近年来，可解释人工智能（XAI）方法已经促进了深入验证和知识提取机器学习模型。尽管针对分类进行了广泛研究，但很少有XAI解决方案解决了特定于回归模型的挑战。在回归中，解释需要精确制定以应对特定用户查询（例如区分“为什么输出大于0？”和“为什么输出大于50？”）。此外，它们应反映模型在相关数据子流形上的行为。在本文中，我们介绍了XpertAI，这是一个将预测策略解开为多个范围特定的子策略，并允许将对模型的精准查询（“被解释物”）的制定为这些子策略的线性组合的框架。XpertAI通常制定可以与基于遮挡、梯度集成或反向传播的流行XAI归因技术一起使用。

    arXiv:2403.07486v1 Announce Type: new  Abstract: In recent years, Explainable AI (XAI) methods have facilitated profound validation and knowledge extraction from ML models. While extensively studied for classification, few XAI solutions have addressed the challenges specific to regression models. In regression, explanations need to be precisely formulated to address specific user queries (e.g.\ distinguishing between `Why is the output above 0?' and `Why is the output above 50?'). They should furthermore reflect the model's behavior on the relevant data sub-manifold. In this paper, we introduce XpertAI, a framework that disentangles the prediction strategy into multiple range-specific sub-strategies and allows the formulation of precise queries about the model (the `explanandum') as a linear combination of those sub-strategies. XpertAI is formulated generally to work alongside popular XAI attribution techniques, based on occlusion, gradient integration, or reverse propagation. Qualitat
    
[^4]: 几何信息神经网络

    Geometry-Informed Neural Networks

    [https://arxiv.org/abs/2402.14009](https://arxiv.org/abs/2402.14009)

    GINNs提出了一种新颖的几何信息神经网络范式，可以在几何任务中生成多样的解决方案，无需训练数据，采用显式多样性损失以及可微损失来减轻模态坍缩，并在实验中展示了其在各种复杂性场景中的高效性。

    

    我们引入了几何信息神经网络（GINNs）的概念，涵盖了（i）在几何约束下学习，（ii）神经场作为合适的表示，（iii）生成在几何任务中经常遇到的欠定系统的多样解决方案。值得注意的是，GINN的构建不需要训练数据，因此可以被纯约束驱动地视为生成建模。我们增加了显式的多样性损失来减轻模态坍缩。我们考虑了几种约束，特别是组件的连通性，我们通过莫尔斯理论将其转化为可微损失。在实验中，我们展示了在不断增加复杂性的二维和三维场景中，GINN学习范式的高效性。

    arXiv:2402.14009v1 Announce Type: new  Abstract: We introduce the concept of geometry-informed neural networks (GINNs), which encompass (i) learning under geometric constraints, (ii) neural fields as a suitable representation, and (iii) generating diverse solutions to under-determined systems often encountered in geometric tasks. Notably, the GINN formulation does not require training data, and as such can be considered generative modeling driven purely by constraints. We add an explicit diversity loss to mitigate mode collapse. We consider several constraints, in particular, the connectedness of components which we convert to a differentiable loss through Morse theory. Experimentally, we demonstrate the efficacy of the GINN learning paradigm across a range of two and three-dimensional scenarios with increasing levels of complexity.
    
[^5]: 可解释的生成对抗模仿学习

    Interpretable Generative Adversarial Imitation Learning

    [https://arxiv.org/abs/2402.10310](https://arxiv.org/abs/2402.10310)

    提出了一种结合了信号时序逻辑（STL）推断和控制合成的新颖仿真学习方法，可以明确表示任务为STL公式，同时通过人为调整STL公式实现对人类知识的纳入和新场景的适应，还采用了生成对抗网络（GAN）启发的训练方法，有效缩小了专家策略和学习策略之间的差距

    

    仿真学习方法已经通过专家演示在教授自主系统复杂任务方面取得了相当大的成功。然而，这些方法的局限性在于它们缺乏可解释性，特别是在理解学习代理试图完成的具体任务方面。在本文中，我们提出了一种结合了信号时序逻辑（STL）推断和控制合成的新颖仿真学习方法，使任务可以明确表示为STL公式。这种方法不仅可以清晰地理解任务，还可以通过手动调整STL公式来将人类知识纳入并适应新场景。此外，我们采用了受生成对抗网络（GAN）启发的训练方法进行推断和控制策略，有效地缩小了专家策略和学习策略之间的差距。我们算法的有效性

    arXiv:2402.10310v1 Announce Type: new  Abstract: Imitation learning methods have demonstrated considerable success in teaching autonomous systems complex tasks through expert demonstrations. However, a limitation of these methods is their lack of interpretability, particularly in understanding the specific task the learning agent aims to accomplish. In this paper, we propose a novel imitation learning method that combines Signal Temporal Logic (STL) inference and control synthesis, enabling the explicit representation of the task as an STL formula. This approach not only provides a clear understanding of the task but also allows for the incorporation of human knowledge and adaptation to new scenarios through manual adjustments of the STL formulae. Additionally, we employ a Generative Adversarial Network (GAN)-inspired training approach for both the inference and the control policy, effectively narrowing the gap between the expert and learned policies. The effectiveness of our algorithm
    
[^6]: 使用矩匹配高斯混合模型改进了DDIM采样

    Improved DDIM Sampling with Moment Matching Gaussian Mixtures. (arXiv:2311.04938v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2311.04938](http://arxiv.org/abs/2311.04938)

    在DDIM框架中使用GMM作为反向转移算子，通过矩匹配可以获得质量更高的样本。在无条件模型和类条件模型上进行了实验，并通过FID和IS指标证明了我们的方法的改进效果。

    

    我们提出在Denoising Diffusion Implicit Models (DDIM)框架中使用高斯混合模型（GMM）作为反向转移算子（内核），这是一种从预训练的Denoising Diffusion Probabilistic Models (DDPM)中加速采样的广泛应用方法之一。具体而言，我们通过约束GMM的参数，匹配DDPM前向边际的一阶和二阶中心矩。我们发现，通过矩匹配，可以获得与使用高斯核的原始DDIM相同或更好质量的样本。我们在CelebAHQ和FFHQ的无条件模型以及ImageNet数据集的类条件模型上提供了实验结果。我们的结果表明，在采样步骤较少的情况下，使用GMM内核可以显著改善生成样本的质量，这是通过FID和IS指标衡量的。例如，在ImageNet 256x256上，使用10个采样步骤，我们实现了一个FID值为...

    We propose using a Gaussian Mixture Model (GMM) as reverse transition operator (kernel) within the Denoising Diffusion Implicit Models (DDIM) framework, which is one of the most widely used approaches for accelerated sampling from pre-trained Denoising Diffusion Probabilistic Models (DDPM). Specifically we match the first and second order central moments of the DDPM forward marginals by constraining the parameters of the GMM. We see that moment matching is sufficient to obtain samples with equal or better quality than the original DDIM with Gaussian kernels. We provide experimental results with unconditional models trained on CelebAHQ and FFHQ and class-conditional models trained on ImageNet datasets respectively. Our results suggest that using the GMM kernel leads to significant improvements in the quality of the generated samples when the number of sampling steps is small, as measured by FID and IS metrics. For example on ImageNet 256x256, using 10 sampling steps, we achieve a FID of
    
[^7]: 神经网络控制器到决策树控制器的精确且节约成本的自动转换

    Exact and Cost-Effective Automated Transformation of Neural Network Controllers to Decision Tree Controllers. (arXiv:2304.06049v1 [cs.LG])

    [http://arxiv.org/abs/2304.06049](http://arxiv.org/abs/2304.06049)

    本文研究了将基于神经网络的控制器转换为等效软决策树控制器并提出了一种自动且节约成本的转换算法。该方法适用于包括ReLU激活函数在内的离散输出NN控制器，并能够提高形式验证的运行效率。

    

    在过去的十年中，基于神经网络（NN）的控制器在各种决策任务中表现出了显着的功效。然而，它们的黑盒特性和意外行为和令人惊讶的结果的风险对于在具有正确性和安全性强保证的真实世界系统中的部署构成了挑战。我们通过调查将基于NN的控制器转换为等效的软决策树（SDT）控制器及其对可验证性的影响来解决这些限制。与以前的方法不同，我们专注于离散输出NN控制器，包括整流线性单元（ReLU）激活函数以及argmax操作。然后，我们设计了一种精确但节省成本的转换算法，因为它可以自动删除多余的分支。我们使用OpenAI Gym环境的两个基准测试来评估我们的方法。我们的结果表明，SDT转换可以使形式验证受益，显示运行时改进。

    Over the past decade, neural network (NN)-based controllers have demonstrated remarkable efficacy in a variety of decision-making tasks. However, their black-box nature and the risk of unexpected behaviors and surprising results pose a challenge to their deployment in real-world systems with strong guarantees of correctness and safety. We address these limitations by investigating the transformation of NN-based controllers into equivalent soft decision tree (SDT)-based controllers and its impact on verifiability. Differently from previous approaches, we focus on discrete-output NN controllers including rectified linear unit (ReLU) activation functions as well as argmax operations. We then devise an exact but cost-effective transformation algorithm, in that it can automatically prune redundant branches. We evaluate our approach using two benchmarks from the OpenAI Gym environment. Our results indicate that the SDT transformation can benefit formal verification, showing runtime improveme
    

