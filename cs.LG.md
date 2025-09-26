# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Reinforcement Learning in Categorical Cybernetics](https://arxiv.org/abs/2404.02688) | 强化学习算法可以被归纳到分类控制原理框架中，通过参数化的光学相互作用，展示了新的构造方法。 |
| [^2] | [Generalized Gradient Descent is a Hypergraph Functor](https://arxiv.org/abs/2403.19845) | 广义梯度下降相对于Cartesian reverse derivative categories (CRDCs)的通用客观函数诱导出一个超图函子，将优化问题映射到动力系统，为分布式优化算法提供了新途径。 |
| [^3] | [Can ChatGPT predict article retraction based on Twitter mentions?](https://arxiv.org/abs/2403.16851) | 本研究探讨了ChatGPT是否能够基于Twitter提及来预测文章的撤回，研究发现在预测未来被撤回的有问题文章方面是具有一定潜力的。 |
| [^4] | [Analyzing Male Domestic Violence through Exploratory Data Analysis and Explainable Machine Learning Insights](https://arxiv.org/abs/2403.15594) | 该研究是关于在孟加拉国背景下对男性家庭暴力进行开创性探索，揭示了男性受害者的存在、模式和潜在因素，填补了现有文献对男性受害者研究空白的重要性。 |
| [^5] | [Energy based diffusion generator for efficient sampling of Boltzmann distributions.](http://arxiv.org/abs/2401.02080) | 介绍了一种称为基于能量的扩散生成器的新型采样器，用于从任意目标分布中生成样本，并通过扩散模型和广义哈密顿动力学提高采样性能。在各种复杂分布函数上的实证评估中表现出优越性。 |
| [^6] | [Do DL models and training environments have an impact on energy consumption?.](http://arxiv.org/abs/2307.05520) | 本研究分析了模型架构和训练环境对训练更环保的计算机视觉模型的影响，并找出了能源效率和模型正确性之间的权衡关系。 |
| [^7] | [Towards the Identifiability in Noisy Label Learning: A Multinomial Mixture Approach.](http://arxiv.org/abs/2301.01405) | 本文使用多项式混合模型研究了在有噪声标签学习过程中如何识别出干净标签样本，发现每个实例有至少 $2C-1$ 个有噪声标签时，该问题才是可识别的。为了满足这个要求，提出了一种方法，通过估计噪声标签分布自动生成额外的噪声标签以提高可识别性，无需额外的假设。 |

# 详细

[^1]: 在分类控制原理中的强化学习

    Reinforcement Learning in Categorical Cybernetics

    [https://arxiv.org/abs/2404.02688](https://arxiv.org/abs/2404.02688)

    强化学习算法可以被归纳到分类控制原理框架中，通过参数化的光学相互作用，展示了新的构造方法。

    

    我们展示了几种主要的强化学习（RL）算法适用于分类控制原理框架，即参数化的双向过程。我们在此前的工作基础上展开，其中我们展示了价值迭代可以通过预合成特定的光学表示。本文的主要构造概述如下：（1）我们将Bellman算子扩展到适用于动作值函数并依赖于样本的参数化光学。 （2）我们应用一个可表示的逆变子函子，得到一个应用Bellman迭代的参数化函数。（3）该参数化函数成为另一个代表模型的参数化光学的反向传递，通过代理与环境进行交互。因此，在我们的构造中，参数化光学以两种不同的方式出现，其中一种成为另一种的一部分。

    arXiv:2404.02688v1 Announce Type: new  Abstract: We show that several major algorithms of reinforcement learning (RL) fit into the framework of categorical cybernetics, that is to say, parametrised bidirectional processes. We build on our previous work in which we show that value iteration can be represented by precomposition with a certain optic. The outline of the main construction in this paper is: (1) We extend the Bellman operators to parametrised optics that apply to action-value functions and depend on a sample. (2) We apply a representable contravariant functor, obtaining a parametrised function that applies the Bellman iteration. (3) This parametrised function becomes the backward pass of another parametrised optic that represents the model, which interacts with an environment via an agent. Thus, parametrised optics appear in two different ways in our construction, with one becoming part of the other. As we show, many of the major classes of algorithms in RL can be seen as dif
    
[^2]: 广义梯度下降是一个超图函子

    Generalized Gradient Descent is a Hypergraph Functor

    [https://arxiv.org/abs/2403.19845](https://arxiv.org/abs/2403.19845)

    广义梯度下降相对于Cartesian reverse derivative categories (CRDCs)的通用客观函数诱导出一个超图函子，将优化问题映射到动力系统，为分布式优化算法提供了新途径。

    

    Cartesian reverse derivative categories (CRDCs)提供了对反向导数的公理化泛化，这使得可以将相对于广泛类问题的梯度下降的广义类比应用于经典优化算法。本文展示了相对于给定CRDC的广义梯度下降诱导出一个从优化问题的超图范畴到动力系统的超图函子。该函子的定义域由客观函数组成，这些客观函数在任意CRDC下都是通用的，并且是开放的，可以通过变量共享与其他这样的客观函数组合。对映域类似地被指定为基础CRDC的通用和开放动态系统类别。我们描述了超图函子如何诱导出一个针对任意问题的分布式优化算法。

    arXiv:2403.19845v1 Announce Type: cross  Abstract: Cartesian reverse derivative categories (CRDCs) provide an axiomatic generalization of the reverse derivative, which allows generalized analogues of classic optimization algorithms such as gradient descent to be applied to a broad class of problems. In this paper, we show that generalized gradient descent with respect to a given CRDC induces a hypergraph functor from a hypergraph category of optimization problems to a hypergraph category of dynamical systems. The domain of this functor consists of objective functions that are 1) general in the sense that they are defined with respect to an arbitrary CRDC, and 2) open in that they are decorated spans that can be composed with other such objective functions via variable sharing. The codomain is specified analogously as a category of general and open dynamical systems for the underlying CRDC. We describe how the hypergraph functor induces a distributed optimization algorithm for arbitrary
    
[^3]: ChatGPT是否能够基于Twitter提及来预测文章的撤回？

    Can ChatGPT predict article retraction based on Twitter mentions?

    [https://arxiv.org/abs/2403.16851](https://arxiv.org/abs/2403.16851)

    本研究探讨了ChatGPT是否能够基于Twitter提及来预测文章的撤回，研究发现在预测未来被撤回的有问题文章方面是具有一定潜力的。

    

    检测有问题的研究文章具有重要意义，本研究探讨了根据被撤回文章在Twitter上的提及是否能够在文章被撤回前发出信号，从而在预测未来被撤回的有问题文章方面发挥作用。分析了包括3,505篇已撤回文章及其相关Twitter提及在内的数据集，以及使用粗糙精确匹配方法获取的具有类似特征的3,505篇未撤回文章。通过四种预测方法评估了Twitter提及在预测文章撤回方面的有效性，包括手动标注、关键词识别、机器学习模型和ChatGPT。手动标注的结果表明，的确有被撤回的文章，其Twitter提及包含在撤回前发出信号的可识别证据，尽管它们只占所有被撤回文章的一小部分。

    arXiv:2403.16851v1 Announce Type: cross  Abstract: Detecting problematic research articles timely is a vital task. This study explores whether Twitter mentions of retracted articles can signal potential problems with the articles prior to retraction, thereby playing a role in predicting future retraction of problematic articles. A dataset comprising 3,505 retracted articles and their associated Twitter mentions is analyzed, alongside 3,505 non-retracted articles with similar characteristics obtained using the Coarsened Exact Matching method. The effectiveness of Twitter mentions in predicting article retraction is evaluated by four prediction methods, including manual labelling, keyword identification, machine learning models, and ChatGPT. Manual labelling results indicate that there are indeed retracted articles with their Twitter mentions containing recognizable evidence signaling problems before retraction, although they represent only a limited share of all retracted articles with 
    
[^4]: 通过探索性数据分析和可解释的机器学习洞见分析男性家庭暴力

    Analyzing Male Domestic Violence through Exploratory Data Analysis and Explainable Machine Learning Insights

    [https://arxiv.org/abs/2403.15594](https://arxiv.org/abs/2403.15594)

    该研究是关于在孟加拉国背景下对男性家庭暴力进行开创性探索，揭示了男性受害者的存在、模式和潜在因素，填补了现有文献对男性受害者研究空白的重要性。

    

    家庭暴力通常被视为一个关于女性受害者的性别问题，在近年来越来越受到关注。尽管有这种关注，孟加拉国特别是男性受害者仍然主要被忽视。我们的研究代表了在孟加拉国背景下对男性家庭暴力（MDV）这一未被充分探讨领域的开创性探索，揭示了其普遍性、模式和潜在因素。现有文献主要强调家庭暴力情境中女性的受害，导致对男性受害者的研究空白。我们从孟加拉国主要城市收集了数据，并进行了探索性数据分析以了解潜在动态。我们使用了11种传统机器学习模型（包括默认和优化的超参数）、2种深度学习和4种集成模型。尽管采用了各种方法，CatBoost由于其...

    arXiv:2403.15594v1 Announce Type: cross  Abstract: Domestic violence, which is often perceived as a gendered issue among female victims, has gained increasing attention in recent years. Despite this focus, male victims of domestic abuse remain primarily overlooked, particularly in Bangladesh. Our study represents a pioneering exploration of the underexplored realm of male domestic violence (MDV) within the Bangladeshi context, shedding light on its prevalence, patterns, and underlying factors. Existing literature predominantly emphasizes female victimization in domestic violence scenarios, leading to an absence of research on male victims. We collected data from the major cities of Bangladesh and conducted exploratory data analysis to understand the underlying dynamics. We implemented 11 traditional machine learning models with default and optimized hyperparameters, 2 deep learning, and 4 ensemble models. Despite various approaches, CatBoost has emerged as the top performer due to its 
    
[^5]: 基于能量的扩散生成器用于高效采样Boltzmann分布

    Energy based diffusion generator for efficient sampling of Boltzmann distributions. (arXiv:2401.02080v1 [cs.LG])

    [http://arxiv.org/abs/2401.02080](http://arxiv.org/abs/2401.02080)

    介绍了一种称为基于能量的扩散生成器的新型采样器，用于从任意目标分布中生成样本，并通过扩散模型和广义哈密顿动力学提高采样性能。在各种复杂分布函数上的实证评估中表现出优越性。

    

    我们介绍了一种称为基于能量的扩散生成器的新型采样器，用于从任意目标分布中生成样本。采样模型采用类似变分自编码器的结构，利用解码器将来自简单分布的潜在变量转换为逼近目标分布的随机变量，并设计了基于扩散模型的编码器。利用扩散模型对复杂分布的强大建模能力，我们可以获得生成样本和目标分布之间的Kullback-Leibler散度的准确变分估计。此外，我们提出了基于广义哈密顿动力学的解码器，进一步提高采样性能。通过实证评估，我们展示了我们的方法在各种复杂分布函数上的有效性，展示了其相对于现有方法的优越性。

    We introduce a novel sampler called the energy based diffusion generator for generating samples from arbitrary target distributions. The sampling model employs a structure similar to a variational autoencoder, utilizing a decoder to transform latent variables from a simple distribution into random variables approximating the target distribution, and we design an encoder based on the diffusion model. Leveraging the powerful modeling capacity of the diffusion model for complex distributions, we can obtain an accurate variational estimate of the Kullback-Leibler divergence between the distributions of the generated samples and the target. Moreover, we propose a decoder based on generalized Hamiltonian dynamics to further enhance sampling performance. Through empirical evaluation, we demonstrate the effectiveness of our method across various complex distribution functions, showcasing its superiority compared to existing methods.
    
[^6]: DL模型和训练环境对能源消耗有影响吗？

    Do DL models and training environments have an impact on energy consumption?. (arXiv:2307.05520v1 [cs.LG])

    [http://arxiv.org/abs/2307.05520](http://arxiv.org/abs/2307.05520)

    本研究分析了模型架构和训练环境对训练更环保的计算机视觉模型的影响，并找出了能源效率和模型正确性之间的权衡关系。

    

    当前计算机视觉领域的研究主要集中在提高深度学习（DL）的正确性和推理时间性能上。然而，目前很少有关于训练DL模型带来巨大碳足迹的研究。本研究旨在分析模型架构和训练环境对训练更环保的计算机视觉模型的影响。我们将这个目标分为两个研究问题。首先，我们分析模型架构对实现更环保模型同时保持正确性在最佳水平的影响。其次，我们研究训练环境对生成更环保模型的影响。为了调查这些关系，我们在模型训练过程中收集了与能源效率和模型正确性相关的多个指标。然后，我们描述了模型架构在测量能源效率和模型正确性方面的权衡，以及它们与训练环境的关系。我们在一个实验平台上进行了这项研究。

    Current research in the computer vision field mainly focuses on improving Deep Learning (DL) correctness and inference time performance. However, there is still little work on the huge carbon footprint that has training DL models. This study aims to analyze the impact of the model architecture and training environment when training greener computer vision models. We divide this goal into two research questions. First, we analyze the effects of model architecture on achieving greener models while keeping correctness at optimal levels. Second, we study the influence of the training environment on producing greener models. To investigate these relationships, we collect multiple metrics related to energy efficiency and model correctness during the models' training. Then, we outline the trade-offs between the measured energy efficiency and the models' correctness regarding model architecture, and their relationship with the training environment. We conduct this research in the context of a 
    
[^7]: 面向有噪声标签学习的可识别性：多项式混合方法研究

    Towards the Identifiability in Noisy Label Learning: A Multinomial Mixture Approach. (arXiv:2301.01405v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2301.01405](http://arxiv.org/abs/2301.01405)

    本文使用多项式混合模型研究了在有噪声标签学习过程中如何识别出干净标签样本，发现每个实例有至少 $2C-1$ 个有噪声标签时，该问题才是可识别的。为了满足这个要求，提出了一种方法，通过估计噪声标签分布自动生成额外的噪声标签以提高可识别性，无需额外的假设。

    

    从有噪声标签中进行学习在深度学习中扮演着至关重要的角色。最有前途的有噪声标签学习方法依赖于从带有噪声注释的数据集中识别出干净标签样本。这种识别具有挑战性，因为传统的有噪声标签学习问题假定每个实例只有一个有噪声标签，是不可识别的，也就是说，没有附加的启发式方法理论上无法估计出干净标签。在本文中，我们旨在使用多项式混合模型正式调查这个可识别性问题，以确定使问题可识别的约束条件。具体来说，我们发现，如果每个实例有至少 $2C-1$ 个有噪声标签，其中 C 是类的数量，则该有噪声标签学习问题就变得可识别。为了满足这个要求，而不依赖于每个实例额外的 $2C-2$ 手动注释，我们提出了一种方法，通过估计基于最近邻的噪声标签分布来自动生成额外的噪声标签。这些额外的噪声标签提高了可识别性，使得可以无需任何其他假设来估计干净标签。我们在各种基准和应用程序上验证了我们的方法的有效性。

    Learning from noisy labels (LNL) plays a crucial role in deep learning. The most promising LNL methods rely on identifying clean-label samples from a dataset with noisy annotations. Such an identification is challenging because the conventional LNL problem, which assumes a single noisy label per instance, is non-identifiable, i.e., clean labels cannot be estimated theoretically without additional heuristics. In this paper, we aim to formally investigate this identifiability issue using multinomial mixture models to determine the constraints that make the problem identifiable. Specifically, we discover that the LNL problem becomes identifiable if there are at least $2C - 1$ noisy labels per instance, where $C$ is the number of classes. To meet this requirement without relying on additional $2C - 2$ manual annotations per instance, we propose a method that automatically generates additional noisy labels by estimating the noisy label distribution based on nearest neighbours. These additio
    

