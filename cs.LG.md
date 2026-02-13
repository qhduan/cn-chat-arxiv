# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Efficient compilation of expressive problem space specifications to neural network solvers](https://rss.arxiv.org/abs/2402.01353) | 本文提出了一种算法，可以将高级的问题空间规范编译为适合神经网络求解器的满足性查询，以解决神经网络验证中存在的嵌入间隙问题。 |
| [^2] | [Optimizing Sampling Patterns for Compressed Sensing MRI with Diffusion Generative Models.](http://arxiv.org/abs/2306.03284) | 本论文提出了一种基于扩散生成模型的MRI压缩感知采样模式优化方法，可以只用五张训练图像来学习有效的采样模式，并在不同的解剖结构、加速因子和模式类型下获得具有竞争性的重建效果。 |

# 详细

[^1]: 将表达丰富的问题空间规范高效编译为神经网络求解器

    Efficient compilation of expressive problem space specifications to neural network solvers

    [https://rss.arxiv.org/abs/2402.01353](https://rss.arxiv.org/abs/2402.01353)

    本文提出了一种算法，可以将高级的问题空间规范编译为适合神经网络求解器的满足性查询，以解决神经网络验证中存在的嵌入间隙问题。

    

    最近的研究揭示了神经网络验证中存在的嵌入间隙。在间隙的一侧是一个关于网络行为的高级规范，由领域专家根据可解释的问题空间编写。在另一侧是一组逻辑上等价的可满足性查询，以适合神经网络求解器的形式表达在不可理解的嵌入空间中。在本文中，我们描述了一种将前者编译为后者的算法。我们探索和克服了针对神经网络求解器而不是标准SMT求解器所出现的问题。

    Recent work has described the presence of the embedding gap in neural network verification. On one side of the gap is a high-level specification about the network's behaviour, written by a domain expert in terms of the interpretable problem space. On the other side are a logically-equivalent set of satisfiability queries, expressed in the uninterpretable embedding space in a form suitable for neural network solvers. In this paper we describe an algorithm for compiling the former to the latter. We explore and overcome complications that arise from targeting neural network solvers as opposed to standard SMT solvers.
    
[^2]: 基于扩散生成模型的MRI压缩感知采样模式优化

    Optimizing Sampling Patterns for Compressed Sensing MRI with Diffusion Generative Models. (arXiv:2306.03284v1 [cs.LG])

    [http://arxiv.org/abs/2306.03284](http://arxiv.org/abs/2306.03284)

    本论文提出了一种基于扩散生成模型的MRI压缩感知采样模式优化方法，可以只用五张训练图像来学习有效的采样模式，并在不同的解剖结构、加速因子和模式类型下获得具有竞争性的重建效果。

    

    基于扩散生成模型已被用作磁共振成像(MRI)重建的强大先验。我们提出了一种学习方法，利用预先训练的扩散生成模型来优化压缩感知多线圈MRI的子采样模式。在训练过程中，我们使用基于扩散模型和MRI测量过程的后验平均估计的单步重建。在不同解剖结构、加速因子和模式类型的实验中，我们学习到的采样运算符比基线模式具有竞争性，而在2D模式的情况下，重建效果得到了改善。我们的方法只需要五个训练图像就可以学习到有效的采样模式。

    Diffusion-based generative models have been used as powerful priors for magnetic resonance imaging (MRI) reconstruction. We present a learning method to optimize sub-sampling patterns for compressed sensing multi-coil MRI that leverages pre-trained diffusion generative models. Crucially, during training we use a single-step reconstruction based on the posterior mean estimate given by the diffusion model and the MRI measurement process. Experiments across varying anatomies, acceleration factors, and pattern types show that sampling operators learned with our method lead to competitive, and in the case of 2D patterns, improved reconstructions compared to baseline patterns. Our method requires as few as five training images to learn effective sampling patterns.
    

