# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Application-Driven Innovation in Machine Learning](https://arxiv.org/abs/2403.17381) | 应用驱动研究在机器学习领域具有重要影响，可以与方法驱动研究有益地协同，但目前审查、招聘和教学实践往往阻碍了这种创新。 |
| [^2] | [LOOPer: A Learned Automatic Code Optimizer For Polyhedral Compilers](https://arxiv.org/abs/2403.11522) | LOOPer是针对多面体编译器的学习型自动代码优化器，通过机器学习建立成本模型来指导多面体优化搜索，突破了传统编译器在选择代码转换方面的限制。 |
| [^3] | [Investigation of the Impact of Synthetic Training Data in the Industrial Application of Terminal Strip Object Detection](https://arxiv.org/abs/2403.04809) | 本文研究了标准物体检测器在复杂的工业终端条对象检测应用中的模拟到真实泛化性能。 |
| [^4] | [Predicting large scale cosmological structure evolution with GAN-based autoencoders](https://arxiv.org/abs/2403.02171) | 使用基于GAN的自动编码器在宇宙学模拟中尝试预测结构演化，发现在2D模拟中能够很好地预测暗物质场的结构演化，但在3D模拟中表现更差，提供速度场作为输入后结果显著改善。 |
| [^5] | [DE$^3$-BERT: Distance-Enhanced Early Exiting for BERT based on Prototypical Networks](https://arxiv.org/abs/2402.05948) | DE$^3$-BERT是一种基于原型网络和距离度量的增强距离早期停止框架，用于提高BERT等预训练语言模型的推断速度和准确性。 |
| [^6] | [Revisiting the Last-Iterate Convergence of Stochastic Gradient Methods](https://arxiv.org/abs/2312.08531) | 研究了随机梯度方法的最终迭代收敛性，并提出了不需要限制性假设的最优收敛速率问题。 |
| [^7] | [Decentralised, Scalable and Privacy-Preserving Synthetic Data Generation.](http://arxiv.org/abs/2310.20062) | 这篇论文介绍了一种去中心化、可扩展且保护隐私的合成数据生成系统，使真实数据的贡献者能够参与差分隐私合成数据生成，从而提供更好的隐私和统计保证，并在机器学习流程中更好地利用合成数据。 |
| [^8] | [A Survey on Generative Modeling with Limited Data, Few Shots, and Zero Shot.](http://arxiv.org/abs/2307.14397) | 本文调查了在有限数据、少样本和零样本条件下学习生成模型的情况，并提出了关于任务和方法的分类体系，研究了它们之间的相互作用，并探讨了未来的研究方向。 |
| [^9] | [Towards Global Optimality in Cooperative MARL with the Transformation And Distillation Framework.](http://arxiv.org/abs/2207.11143) | 本文研究了采用分散策略的MARL算法在梯度下降优化器下的次最优性，并提出了转化与蒸馏框架，该框架可以将多智能体MDP转化为单智能体MDP以实现分散执行。 |

# 详细

[^1]: 机器学习中的应用驱动创新

    Application-Driven Innovation in Machine Learning

    [https://arxiv.org/abs/2403.17381](https://arxiv.org/abs/2403.17381)

    应用驱动研究在机器学习领域具有重要影响，可以与方法驱动研究有益地协同，但目前审查、招聘和教学实践往往阻碍了这种创新。

    

    随着机器学习应用的不断增长，受特定现实挑战启发的创新算法变得日益重要。这样的工作不仅在应用领域具有重要影响，也在机器学习本身具有重要影响。本文描述了机器学习中应用驱动研究的范式，将其与更标准的方法驱动研究进行了对比。我们阐明了应用驱动机器学习的好处，以及这种方法如何可以与方法驱动工作有益地协同。尽管具有这些好处，我们发现机器学习中的审查、招聘和教学实践往往阻碍了应用驱动创新。我们概述了如何改进这些流程。

    arXiv:2403.17381v1 Announce Type: cross  Abstract: As applications of machine learning proliferate, innovative algorithms inspired by specific real-world challenges have become increasingly important. Such work offers the potential for significant impact not merely in domains of application but also in machine learning itself. In this paper, we describe the paradigm of application-driven research in machine learning, contrasting it with the more standard paradigm of methods-driven research. We illustrate the benefits of application-driven machine learning and how this approach can productively synergize with methods-driven work. Despite these benefits, we find that reviewing, hiring, and teaching practices in machine learning often hold back application-driven innovation. We outline how these processes may be improved.
    
[^2]: LOOPer: 一个针对多面体编译器的学习型自动代码优化器

    LOOPer: A Learned Automatic Code Optimizer For Polyhedral Compilers

    [https://arxiv.org/abs/2403.11522](https://arxiv.org/abs/2403.11522)

    LOOPer是针对多面体编译器的学习型自动代码优化器，通过机器学习建立成本模型来指导多面体优化搜索，突破了传统编译器在选择代码转换方面的限制。

    

    虽然多面体编译器在实现高级代码转换方面已经取得成功，但在选择能够带来最佳加速的最有利转换方面仍然面临挑战。这促使使用机器学习构建成本模型来引导多面体优化的搜索。最先进的多面体编译器已经展示了这种方法的可行性概念验证。虽然这种概念验证显示出了希望，但仍然存在显著限制。使用深度学习成本模型的最先进多面体编译器只支持少量仿射变换的子集，限制了它们应用复杂代码变换的能力。它们还只支持具有单个循环嵌套和矩形迭代域的简单程序，限制了它们对许多程序的适用性。这些限制显著影响了这样的编译器和自动调度器的通用性

    arXiv:2403.11522v1 Announce Type: cross  Abstract: While polyhedral compilers have shown success in implementing advanced code transformations, they still have challenges in selecting the most profitable transformations that lead to the best speedups. This has motivated the use of machine learning to build cost models to guide the search for polyhedral optimizations. State-of-the-art polyhedral compilers have demonstrated a viable proof-of-concept of this approach. While such a proof-of-concept has shown promise, it still has significant limitations. State-of-the-art polyhedral compilers that use a deep-learning cost model only support a small subset of affine transformations, limiting their ability to apply complex code transformations. They also only support simple programs that have a single loop nest and a rectangular iteration domain, limiting their applicability to many programs. These limitations significantly impact the generality of such compilers and autoschedulers and put in
    
[^3]: 研究合成训练数据对终端条对象检测工业应用的影响

    Investigation of the Impact of Synthetic Training Data in the Industrial Application of Terminal Strip Object Detection

    [https://arxiv.org/abs/2403.04809](https://arxiv.org/abs/2403.04809)

    本文研究了标准物体检测器在复杂的工业终端条对象检测应用中的模拟到真实泛化性能。

    

    在工业制造中，存在许多检查或检测特定对象的任务，目前这些任务通常由人工或经典图像处理方法执行。因此，在工业环境引入最新的深度学习模型有可能提高生产效率并实现新的应用。然而，收集和标记足够的数据通常是困难的，这使得这类项目的实施变得复杂。因此，图像合成方法通常用于从3D模型生成合成训练数据，并自动标注这些数据，尽管这会导致一个模拟到真实领域差距。本文研究了标准物体检测器在复杂的工业终端条对象检测应用中的模拟到真实泛化性能。通过结合领域随机化和领域知识，我们创建了一个图像合成流水线，用于自动生成训练数据。

    arXiv:2403.04809v1 Announce Type: cross  Abstract: In industrial manufacturing, numerous tasks of visually inspecting or detecting specific objects exist that are currently performed manually or by classical image processing methods. Therefore, introducing recent deep learning models to industrial environments holds the potential to increase productivity and enable new applications. However, gathering and labeling sufficient data is often intractable, complicating the implementation of such projects. Hence, image synthesis methods are commonly used to generate synthetic training data from 3D models and annotate them automatically, although it results in a sim-to-real domain gap. In this paper, we investigate the sim-to-real generalization performance of standard object detectors on the complex industrial application of terminal strip object detection. Combining domain randomization and domain knowledge, we created an image synthesis pipeline for automatically generating the training da
    
[^4]: 使用基于GAN的自动编码器预测宇宙大尺度结构演化

    Predicting large scale cosmological structure evolution with GAN-based autoencoders

    [https://arxiv.org/abs/2403.02171](https://arxiv.org/abs/2403.02171)

    使用基于GAN的自动编码器在宇宙学模拟中尝试预测结构演化，发现在2D模拟中能够很好地预测暗物质场的结构演化，但在3D模拟中表现更差，提供速度场作为输入后结果显著改善。

    

    宇宙学模拟在从初始条件预测和理解大尺度结构形成中起着关键作用。我们利用基于GAN的自动编码器尝试预测模拟中的结构演化。自动编码器是在描述暗物质场演化的2D和3D N体模拟生成的图像和立方体上进行训练的。我们发现，虽然自动编码器可以很好地预测2D模拟暗物质场的结构演化，但在类似条件下，仅使用密度场作为输入情况下，在3D模拟中表现明显更差。然而，提供速度场作为输入能够大大改善结果，预测类似，而无论输入和目标之间的时间差异如何。

    arXiv:2403.02171v1 Announce Type: cross  Abstract: Cosmological simulations play a key role in the prediction and understanding of large scale structure formation from initial conditions. We make use of GAN-based Autoencoders (AEs) in an attempt to predict structure evolution within simulations. The AEs are trained on images and cubes issued from respectively 2D and 3D N-body simulations describing the evolution of the dark matter (DM) field. We find that while the AEs can predict structure evolution for 2D simulations of DM fields well, using only the density fields as input, they perform significantly more poorly in similar conditions for 3D simulations. However, additionally providing velocity fields as inputs greatly improves results, with similar predictions regardless of time-difference between input and target.
    
[^5]: DE$^3$-BERT: 基于原型网络的增强距离早期停止方法，用于BERT

    DE$^3$-BERT: Distance-Enhanced Early Exiting for BERT based on Prototypical Networks

    [https://arxiv.org/abs/2402.05948](https://arxiv.org/abs/2402.05948)

    DE$^3$-BERT是一种基于原型网络和距离度量的增强距离早期停止框架，用于提高BERT等预训练语言模型的推断速度和准确性。

    

    早期停止方法通过动态调整执行的层数，提高了像BERT这样的预训练语言模型的推断速度。然而，大多数早期停止方法仅考虑了来自单个测试样本的局部信息来确定早期停止的指标，而未利用样本群体提供的全局信息。这导致对预测正确性的估计不够准确，从而产生错误的早期停止决策。为了弥合这个差距，我们探索了有效结合局部和全局信息以确保可靠的早期停止的必要性。为此，我们利用原型网络学习类别原型，并设计了样本和类别原型之间的距离度量。这使我们能够利用全局信息来估计早期预测的正确性。基于此，我们提出了一种新颖的DE$^3$-BERT增强距离早期停止框架。

    Early exiting has demonstrated its effectiveness in accelerating the inference of pre-trained language models like BERT by dynamically adjusting the number of layers executed. However, most existing early exiting methods only consider local information from an individual test sample to determine their exiting indicators, failing to leverage the global information offered by sample population. This leads to suboptimal estimation of prediction correctness, resulting in erroneous exiting decisions. To bridge the gap, we explore the necessity of effectively combining both local and global information to ensure reliable early exiting during inference. Purposefully, we leverage prototypical networks to learn class prototypes and devise a distance metric between samples and class prototypes. This enables us to utilize global information for estimating the correctness of early predictions. On this basis, we propose a novel Distance-Enhanced Early Exiting framework for BERT (DE$^3$-BERT). DE$^3
    
[^6]: 重新审视随机梯度方法的最终迭代收敛性

    Revisiting the Last-Iterate Convergence of Stochastic Gradient Methods

    [https://arxiv.org/abs/2312.08531](https://arxiv.org/abs/2312.08531)

    研究了随机梯度方法的最终迭代收敛性，并提出了不需要限制性假设的最优收敛速率问题。

    

    在过去几年里，随机梯度下降（SGD）算法的最终迭代收敛引起了人们的兴趣，因为它在实践中表现良好但缺乏理论理解。对于Lipschitz凸函数，不同的研究建立了最佳的$O(\log(1/\delta)\log T/\sqrt{T})$或$O(\sqrt{\log(1/\delta)/T})$最终迭代的高概率收敛速率，其中$T$是时间跨度，$\delta$是失败概率。然而，为了证明这些界限，所有现有的工作要么局限于紧致域，要么需要几乎肯定有界的噪声。很自然地会问，不需要这两个限制性假设的情况下，SGD的最终迭代是否仍然可以保证最佳的收敛速率。除了这个重要问题外，还有很多理论问题仍然没有答案。

    arXiv:2312.08531v2 Announce Type: replace  Abstract: In the past several years, the last-iterate convergence of the Stochastic Gradient Descent (SGD) algorithm has triggered people's interest due to its good performance in practice but lack of theoretical understanding. For Lipschitz convex functions, different works have established the optimal $O(\log(1/\delta)\log T/\sqrt{T})$ or $O(\sqrt{\log(1/\delta)/T})$ high-probability convergence rates for the final iterate, where $T$ is the time horizon and $\delta$ is the failure probability. However, to prove these bounds, all the existing works are either limited to compact domains or require almost surely bounded noises. It is natural to ask whether the last iterate of SGD can still guarantee the optimal convergence rate but without these two restrictive assumptions. Besides this important question, there are still lots of theoretical problems lacking an answer. For example, compared with the last-iterate convergence of SGD for non-smoot
    
[^7]: 去中心化、可扩展且保护隐私的合成数据生成

    Decentralised, Scalable and Privacy-Preserving Synthetic Data Generation. (arXiv:2310.20062v1 [cs.CR])

    [http://arxiv.org/abs/2310.20062](http://arxiv.org/abs/2310.20062)

    这篇论文介绍了一种去中心化、可扩展且保护隐私的合成数据生成系统，使真实数据的贡献者能够参与差分隐私合成数据生成，从而提供更好的隐私和统计保证，并在机器学习流程中更好地利用合成数据。

    

    合成数据作为一种有潜力的方式在降低隐私风险的同时发挥数据价值。合成数据的潜力不仅局限于隐私友好的数据发布，还包括在培训机器学习算法等使用案例中补充真实数据，使其更公平、更能抵抗分布转变等。对于提供更好的隐私和统计保证以及更好地在机器学习流程中利用合成数据的算法进展引起了广泛兴趣。然而，对于负责任和值得信赖的合成数据生成来说，仅关注这些算法方面是不够的，而应该考虑合成数据生成流程的整体视角。我们构建了一个新的系统，允许真实数据的贡献者在没有依赖于值得信赖的中心的情况下自主参与差分隐私合成数据生成。我们的模块化、通用化和可扩展的解决方案基于...

    Synthetic data is emerging as a promising way to harness the value of data, while reducing privacy risks. The potential of synthetic data is not limited to privacy-friendly data release, but also includes complementing real data in use-cases such as training machine learning algorithms that are more fair and robust to distribution shifts etc. There is a lot of interest in algorithmic advances in synthetic data generation for providing better privacy and statistical guarantees and for its better utilisation in machine learning pipelines. However, for responsible and trustworthy synthetic data generation, it is not sufficient to focus only on these algorithmic aspects and instead, a holistic view of the synthetic data generation pipeline must be considered. We build a novel system that allows the contributors of real data to autonomously participate in differentially private synthetic data generation without relying on a trusted centre. Our modular, general and scalable solution is based
    
[^8]: 关于有限数据、少样本和零样本情况下生成建模的调查

    A Survey on Generative Modeling with Limited Data, Few Shots, and Zero Shot. (arXiv:2307.14397v1 [cs.CV])

    [http://arxiv.org/abs/2307.14397](http://arxiv.org/abs/2307.14397)

    本文调查了在有限数据、少样本和零样本条件下学习生成模型的情况，并提出了关于任务和方法的分类体系，研究了它们之间的相互作用，并探讨了未来的研究方向。

    

    在机器学习中，生成建模旨在学习生成与训练数据分布统计相似的新数据。本文调查了在有限数据、少样本和零样本条件下学习生成模型的情况，称为数据约束下的生成建模（GM-DC）。这是一个重要的主题，当数据获取具有挑战性时，例如医疗应用。我们讨论了背景、挑战，并提出了两个分类体系：一个是GM-DC任务分类，另一个是GM-DC方法分类。重要的是，我们研究了不同GM-DC任务和方法之间的相互作用。此外，我们还强调了研究空白、研究趋势和未来探索的潜在途径。项目网站：https://gmdc-survey.github.io。

    In machine learning, generative modeling aims to learn to generate new data statistically similar to the training data distribution. In this paper, we survey learning generative models under limited data, few shots and zero shot, referred to as Generative Modeling under Data Constraint (GM-DC). This is an important topic when data acquisition is challenging, e.g. healthcare applications. We discuss background, challenges, and propose two taxonomies: one on GM-DC tasks and another on GM-DC approaches. Importantly, we study interactions between different GM-DC tasks and approaches. Furthermore, we highlight research gaps, research trends, and potential avenues for future exploration. Project website: https://gmdc-survey.github.io.
    
[^9]: 《采用转化与蒸馏框架实现合作MARL全局最优性》

    Towards Global Optimality in Cooperative MARL with the Transformation And Distillation Framework. (arXiv:2207.11143v3 [cs.MA] UPDATED)

    [http://arxiv.org/abs/2207.11143](http://arxiv.org/abs/2207.11143)

    本文研究了采用分散策略的MARL算法在梯度下降优化器下的次最优性，并提出了转化与蒸馏框架，该框架可以将多智能体MDP转化为单智能体MDP以实现分散执行。

    

    在合作多智能体强化学习中，分散执行是一项核心需求。目前，大多数流行的MARL算法采用分散策略来实现分散执行，并使用梯度下降作为优化器。然而，在考虑到优化方法的情况下，这些算法几乎没有任何理论分析，我们发现当梯度下降被选为优化方法时，各种流行的分散策略MARL算法在玩具任务中都是次最优的。本文在理论上分析了两种常见的采用分散策略的算法——多智能体策略梯度方法和值分解方法，证明了它们在使用梯度下降时的次最优性。此外，我们提出了转化与蒸馏（TAD）框架，它将多智能体MDP重新制定为一种具有连续结构的特殊单智能体MDP，并通过蒸馏实现分散执行。

    Decentralized execution is one core demand in cooperative multi-agent reinforcement learning (MARL). Recently, most popular MARL algorithms have adopted decentralized policies to enable decentralized execution and use gradient descent as their optimizer. However, there is hardly any theoretical analysis of these algorithms taking the optimization method into consideration, and we find that various popular MARL algorithms with decentralized policies are suboptimal in toy tasks when gradient descent is chosen as their optimization method. In this paper, we theoretically analyze two common classes of algorithms with decentralized policies -- multi-agent policy gradient methods and value-decomposition methods to prove their suboptimality when gradient descent is used. In addition, we propose the Transformation And Distillation (TAD) framework, which reformulates a multi-agent MDP as a special single-agent MDP with a sequential structure and enables decentralized execution by distilling the
    

