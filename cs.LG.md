# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [PLOT-TAL -- Prompt Learning with Optimal Transport for Few-Shot Temporal Action Localization](https://arxiv.org/abs/2403.18915) | 提出了使用最优传输进行少样本时序动作定位的提示学习方法，通过多提示学习框架和最优传输理论的结合，有效地捕捉通用特征和减轻过拟合风险 |
| [^2] | [From Zero to Hero: How local curvature at artless initial conditions leads away from bad minima](https://arxiv.org/abs/2403.02418) | 局部曲率变化导致系统从良性且富有信息的局部景观逐渐陷入无信息的迷宫，关键转变与时间相关的Hessian的阈值有关。 |
| [^3] | [DualView: Data Attribution from the Dual Perspective](https://arxiv.org/abs/2402.12118) | 提出了DualView，一种基于替代建模的后期数据归因方法，具有高效计算和优质评估结果。 |
| [^4] | [Fine-Tuned Language Models Generate Stable Inorganic Materials as Text](https://arxiv.org/abs/2402.04379) | 细调语言模型用于生成稳定材料，具有可靠性高和灵活性强的优势，能以较高的速率生成被预测为亚稳态的材料。 |
| [^5] | [Classify and Generate Reciprocally: Simultaneous Positive-Unlabelled Learning and Conditional Generation with Extra Data](https://arxiv.org/abs/2006.07841) | 本论文提出了一种同时利用正数据-无标签学习和有条件生成的训练框架，以及额外无标签数据的方法。通过使用一个对噪声标签具有鲁棒性的分类器噪声不变有条件生成对抗网络来提高PU分类器的性能，并利用PU分类器预测的标签和额外数据来帮助生成。实验结果证明了该方法的有效性。 |
| [^6] | [Learning Concepts Definable in First-Order Logic with Counting](https://arxiv.org/abs/1909.03820) | 该研究将一阶逻辑与计数符号相结合，证明了可以在多对数度结构下以次线性时间一致学习可定义的分类器，为包含数值方面的机器学习扩展学习框架迈出了第一步。 |
| [^7] | [I-CEE: Tailoring Explanations of Image Classification Models to User Expertise.](http://arxiv.org/abs/2312.12102) | I-CEE是一个人为中心的框架，为用户专业知识定制了图像分类模型的解释，通过提供信息丰富的示例图像、局部解释和模型决策来帮助用户理解模型的决策。 |
| [^8] | [Adversarially Robust Deep Learning with Optimal-Transport-Regularized Divergences.](http://arxiv.org/abs/2309.03791) | 本论文介绍了一种新的方法ARMOR_D来加强深度学习模型的对抗鲁棒性，该方法基于最优传输正则化差异，通过在分布的邻域上进行最大化期望损失来实现。实验证明，ARMOR_D方法在恶意软件检测和图像识别应用中能够优于现有方法，在对抗攻击下的鲁棒性方面具有较好的效果。 |
| [^9] | [Compressed and distributed least-squares regression: convergence rates with applications to Federated Learning.](http://arxiv.org/abs/2308.01358) | 本文研究了压缩对分布式和联邦学习中随机梯度算法的影响，通过比较不同的无偏压缩操作符的收敛速度，超越了经典的最坏情况分析。针对最小二乘回归，我们提出了一个随机逼近算法，并考虑了随机场的一般假设和噪声协方差的限制，以分析各种随机化机制。 |
| [^10] | [Pseudo-Labeling for Kernel Ridge Regression under Covariate Shift.](http://arxiv.org/abs/2302.10160) | 该论文提出了一种关于核岭回归的协变量转移策略，通过使用伪标签进行模型选择，能够适应不同特征分布下的学习，实现均方误差最小化。 |

# 详细

[^1]: 使用最优传输进行少样本时序动作定位的提示学习

    PLOT-TAL -- Prompt Learning with Optimal Transport for Few-Shot Temporal Action Localization

    [https://arxiv.org/abs/2403.18915](https://arxiv.org/abs/2403.18915)

    提出了使用最优传输进行少样本时序动作定位的提示学习方法，通过多提示学习框架和最优传输理论的结合，有效地捕捉通用特征和减轻过拟合风险

    

    本文介绍了一种新的方法来处理少样本学习中的时序动作定位（TAL）。我们的工作解决了传统单样本学习方法的固有局限性，这些方法往往由于无法在真实世界的视频中跨不同上下文进行泛化而导致过拟合。鉴于视频中摄像机视角、背景和物体的多样性，我们提出了一个增强了最优传输的多提示学习框架。这个设计允许模型为每个动作学习一组多样的提示，更有效地捕捉通用特征并分布表示以减轻过拟合的风险。此外，通过采用最优传输理论，我们可以有效地将这些提示与动作特征进行对齐，优化以获得适应视频数据多面性的综合表示。我们的实验证明了动作定位方面的显著改进。

    arXiv:2403.18915v1 Announce Type: cross  Abstract: This paper introduces a novel approach to temporal action localization (TAL) in few-shot learning. Our work addresses the inherent limitations of conventional single-prompt learning methods that often lead to overfitting due to the inability to generalize across varying contexts in real-world videos. Recognizing the diversity of camera views, backgrounds, and objects in videos, we propose a multi-prompt learning framework enhanced with optimal transport. This design allows the model to learn a set of diverse prompts for each action, capturing general characteristics more effectively and distributing the representation to mitigate the risk of overfitting. Furthermore, by employing optimal transport theory, we efficiently align these prompts with action features, optimizing for a comprehensive representation that adapts to the multifaceted nature of video data. Our experiments demonstrate significant improvements in action localization a
    
[^2]: 从零到英雄：无知初值处的局部曲率如何远离糟糕的极小值

    From Zero to Hero: How local curvature at artless initial conditions leads away from bad minima

    [https://arxiv.org/abs/2403.02418](https://arxiv.org/abs/2403.02418)

    局部曲率变化导致系统从良性且富有信息的局部景观逐渐陷入无信息的迷宫，关键转变与时间相关的Hessian的阈值有关。

    

    我们研究了梯度下降在非凸和高维设置中的优化动力学，重点关注相位恢复问题作为复杂损失景观的案例研究。通过分析局部曲率在优化过程中的变化，我们发现在中间信噪比下，Hessian在下降的第一个阶段显示出指向好极小值的下降方向，然后在结束时被困在糟糕的极小值中。因此，局部景观起初是良性且富有信息的，然后梯度下降将系统带入无信息的迷宫。两个阶段之间的转变与时间相关的Hessian的BBP类型阈值相关联。

    arXiv:2403.02418v1 Announce Type: new  Abstract: We investigate the optimization dynamics of gradient descent in a non-convex and high-dimensional setting, with a focus on the phase retrieval problem as a case study for complex loss landscapes. We first study the high-dimensional limit where both the number $M$ and the dimension $N$ of the data are going to infinity at fixed signal-to-noise ratio $\alpha = M/N$. By analyzing how the local curvature changes during optimization, we uncover that for intermediate $\alpha$, the Hessian displays a downward direction pointing towards good minima in the first regime of the descent, before being trapped in bad minima at the end. Hence, the local landscape is benign and informative at first, before gradient descent brings the system into a uninformative maze. The transition between the two regimes is associated to a BBP-type threshold in the time-dependent Hessian. Through both theoretical analysis and numerical experiments, we show that in prac
    
[^3]: DualView：双重视角下的数据归因

    DualView: Data Attribution from the Dual Perspective

    [https://arxiv.org/abs/2402.12118](https://arxiv.org/abs/2402.12118)

    提出了DualView，一种基于替代建模的后期数据归因方法，具有高效计算和优质评估结果。

    

    本文提出了DualView，这是一种基于替代建模的后期数据归因方法，展示了高计算效率和良好的评估结果。我们专注于神经网络，在与文献相关的适当定量评估策略下评估了我们提出的技术，比较了与相关主要本地数据归因方法的性能。

    arXiv:2402.12118v1 Announce Type: cross  Abstract: Local data attribution (or influence estimation) techniques aim at estimating the impact that individual data points seen during training have on particular predictions of an already trained Machine Learning model during test time. Previous methods either do not perform well consistently across different evaluation criteria from literature, are characterized by a high computational demand, or suffer from both. In this work we present DualView, a novel method for post-hoc data attribution based on surrogate modelling, demonstrating both high computational efficiency, as well as good evaluation results. With a focus on neural networks, we evaluate our proposed technique using suitable quantitative evaluation strategies from the literature against related principal local data attribution methods. We find that DualView requires considerably lower computational resources than other methods, while demonstrating comparable performance to comp
    
[^4]: 细调语言模型生成稳定的无机材料文本

    Fine-Tuned Language Models Generate Stable Inorganic Materials as Text

    [https://arxiv.org/abs/2402.04379](https://arxiv.org/abs/2402.04379)

    细调语言模型用于生成稳定材料，具有可靠性高和灵活性强的优势，能以较高的速率生成被预测为亚稳态的材料。

    

    我们提出了对大型语言模型进行细调，以生成稳定材料。虽然非传统，但在文本编码的原子数据上细调大型语言模型非常简单易行，同时可靠性高，约90%的采样结构遵守原子位置和电荷的物理约束。通过来自学习的机器学习势和金标准DFT计算的能量以上的计算，我们表明我们的最强模型（细调LLaMA-2 70B）可以以CDVAE竞争扩散模型的约两倍速率（49% vs 28%）生成被预测为亚稳态的材料。由于文本提示的固有灵活性，我们的模型可以同时用于稳定材料的无条件生成、部分结构的填充和文本条件生成。最后，我们表明语言模型捕捉晶体结构的关键对称性的能力随模型规模的增大而改善，这表明预训练的LLM的偏差出奇地适合原子性的应用。

    We propose fine-tuning large language models for generation of stable materials. While unorthodox, fine-tuning large language models on text-encoded atomistic data is simple to implement yet reliable, with around 90% of sampled structures obeying physical constraints on atom positions and charges. Using energy above hull calculations from both learned ML potentials and gold-standard DFT calculations, we show that our strongest model (fine-tuned LLaMA-2 70B) can generate materials predicted to be metastable at about twice the rate (49% vs 28%) of CDVAE, a competing diffusion model. Because of text prompting's inherent flexibility, our models can simultaneously be used for unconditional generation of stable material, infilling of partial structures and text-conditional generation. Finally, we show that language models' ability to capture key symmetries of crystal structures improves with model scale, suggesting that the biases of pretrained LLMs are surprisingly well-suited for atomistic
    
[^5]: 同时进行正数据-无标签学习和有条件生成，利用额外数据来分类和生成

    Classify and Generate Reciprocally: Simultaneous Positive-Unlabelled Learning and Conditional Generation with Extra Data

    [https://arxiv.org/abs/2006.07841](https://arxiv.org/abs/2006.07841)

    本论文提出了一种同时利用正数据-无标签学习和有条件生成的训练框架，以及额外无标签数据的方法。通过使用一个对噪声标签具有鲁棒性的分类器噪声不变有条件生成对抗网络来提高PU分类器的性能，并利用PU分类器预测的标签和额外数据来帮助生成。实验结果证明了该方法的有效性。

    

    在许多机器学习问题中，标记类别数据的稀缺性是一个普遍存在的瓶颈。虽然存在丰富的无标签数据并提供潜在的解决方案，但利用它们是非常具有挑战性的。本文通过同时利用正数据-无标签（Positive-Unlabeled，PU）分类和有条件生成，以及额外的无标签数据，解决了这个问题。特别地，我们提出了一个新的训练框架，使得在面对额外数据（尤其是分布外的无标签数据）时，同时进行PU分类和有条件生成成为可能，通过探索它们之间的相互作用：1）通过一个对噪声标签具有鲁棒性的新型分类器噪声不变有条件生成对抗网络（Classifier-Noise-Invariant Conditional GAN，CNI-CGAN）来提高PU分类器的性能，2）利用PU分类器预测的标签和额外数据来帮助生成。从理论上，我们证明了CNI-CGAN的最优条件，并在实验中通过广泛的评估来验证了我们的方法。

    The scarcity of class-labeled data is a ubiquitous bottleneck in many machine learning problems. While abundant unlabeled data typically exist and provide a potential solution, it is highly challenging to exploit them. In this paper, we address this problem by leveraging Positive-Unlabeled~(PU) classification and the conditional generation with extra unlabeled data \emph{simultaneously}. In particular, we present a novel training framework to jointly target both PU classification and conditional generation when exposed to extra data, especially out-of-distribution unlabeled data, by exploring the interplay between them: 1) enhancing the performance of PU classifiers with the assistance of a novel Classifier-Noise-Invariant Conditional GAN~(CNI-CGAN) that is robust to noisy labels, 2) leveraging extra data with predicted labels from a PU classifier to help the generation. Theoretically, we prove the optimal condition of CNI-CGAN, and experimentally, we conducted extensive evaluations on
    
[^6]: 用计数符号的一阶逻辑定义的概念的学习

    Learning Concepts Definable in First-Order Logic with Counting

    [https://arxiv.org/abs/1909.03820](https://arxiv.org/abs/1909.03820)

    该研究将一阶逻辑与计数符号相结合，证明了可以在多对数度结构下以次线性时间一致学习可定义的分类器，为包含数值方面的机器学习扩展学习框架迈出了第一步。

    

    我们研究了在Grohe和Tur\'an引入的逻辑框架下的关系背景结构上的布尔分类问题。众所周知(Grohe和Ritzert, LICS 2017)，在多对数度结构上的一阶逻辑可定义的分类器可以在次线性时间内学习，其中结构的度和运行时间是以结构的大小为单位来衡量的。我们将结果推广到了由Kuske和Schweikardt(LICS 2017)引入的带计数的一阶逻辑FOCN，它作为一个广泛推广各种计数逻辑的表现逻辑。具体来说，我们证明了可以在多对数度结构类上定义的FOCN中的分类器可以在次线性时间内一致地学习。这可以看作是将学习框架扩展以包含机器学习的数值方面的第一步。我们将这一结果扩展到了无视的概率

    arXiv:1909.03820v2 Announce Type: replace-cross  Abstract: We study Boolean classification problems over relational background structures in the logical framework introduced by Grohe and Tur\'an (TOCS 2004). It is known (Grohe and Ritzert, LICS 2017) that classifiers definable in first-order logic over structures of polylogarithmic degree can be learned in sublinear time, where the degree of the structure and the running time are measured in terms of the size of the structure. We generalise the results to the first-order logic with counting FOCN, which was introduced by Kuske and Schweikardt (LICS 2017) as an expressive logic generalising various other counting logics. Specifically, we prove that classifiers definable in FOCN over classes of structures of polylogarithmic degree can be consistently learned in sublinear time. This can be seen as a first step towards extending the learning framework to include numerical aspects of machine learning. We extend the result to agnostic probabl
    
[^7]: I-CEE: 将图像分类模型的解释定制为用户专业知识

    I-CEE: Tailoring Explanations of Image Classification Models to User Expertise. (arXiv:2312.12102v2 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2312.12102](http://arxiv.org/abs/2312.12102)

    I-CEE是一个人为中心的框架，为用户专业知识定制了图像分类模型的解释，通过提供信息丰富的示例图像、局部解释和模型决策来帮助用户理解模型的决策。

    

    有效解释黑盒机器学习模型的决策对于依赖它们的人工智能系统的负责任部署至关重要。识别到其重要性，可以生成这些解释的可解释人工智能（XAI）领域提供了几种技术。然而，在这一不断发展的工作中，对用户（解释对象）的关注相对较少，大多数XAI技术产生的是“一刀切”的解释。为了弥合这一差距，实现更加以人为中心的XAI，我们提出了I-CEE，这是一个为用户专业知识定制图像分类解释的框架。受到现有工作的启发，I-CEE通过为用户提供信息丰富的训练数据子集（即示例图像）、相应的局部解释和模型决策来解释图像分类模型的决策。然而，与此前的工作不同的是，I-CEE模拟了示例图像的信息量依赖于用户专业知识的情况，从而为不同的用户提供不同的示例。

    Effectively explaining decisions of black-box machine learning models is critical to responsible deployment of AI systems that rely on them. Recognizing their importance, the field of explainable AI (XAI) provides several techniques to generate these explanations. Yet, there is relatively little emphasis on the user (the explainee) in this growing body of work and most XAI techniques generate "one-size-fits-all" explanations. To bridge this gap and achieve a step closer towards human-centered XAI, we present I-CEE, a framework that provides Image Classification Explanations tailored to User Expertise. Informed by existing work, I-CEE explains the decisions of image classification models by providing the user with an informative subset of training data (i.e., example images), corresponding local explanations, and model decisions. However, unlike prior work, I-CEE models the informativeness of the example images to depend on user expertise, resulting in different examples for different u
    
[^8]: 使用最优传输正则化差异来提高对抗性鲁棒深度学习

    Adversarially Robust Deep Learning with Optimal-Transport-Regularized Divergences. (arXiv:2309.03791v1 [cs.LG])

    [http://arxiv.org/abs/2309.03791](http://arxiv.org/abs/2309.03791)

    本论文介绍了一种新的方法ARMOR_D来加强深度学习模型的对抗鲁棒性，该方法基于最优传输正则化差异，通过在分布的邻域上进行最大化期望损失来实现。实验证明，ARMOR_D方法在恶意软件检测和图像识别应用中能够优于现有方法，在对抗攻击下的鲁棒性方面具有较好的效果。

    

    我们引入了ARMOR_D方法作为增强深度学习模型对抗性鲁棒性的创新方法。这些方法基于一种新的最优传输正则化差异类，通过信息差异和最优传输成本之间的infimal卷积构建。我们使用这些方法来增强对抗性鲁棒性，通过在分布的邻域上最大化期望损失，这被称为分布鲁棒优化技术。作为构建对抗样本的工具，我们的方法允许样本根据最优传输成本进行传输，并根据信息差异进行重新加权。我们在恶意软件检测和图像识别应用上证明了我们方法的有效性，并发现在增强对抗攻击鲁棒性方面，据我们所知，它优于现有方法。ARMOR_D在FGSM攻击下的robustified准确率达到98.29%，在其他攻击下达到98.18%。

    We introduce the $ARMOR_D$ methods as novel approaches to enhancing the adversarial robustness of deep learning models. These methods are based on a new class of optimal-transport-regularized divergences, constructed via an infimal convolution between an information divergence and an optimal-transport (OT) cost. We use these as tools to enhance adversarial robustness by maximizing the expected loss over a neighborhood of distributions, a technique known as distributionally robust optimization. Viewed as a tool for constructing adversarial samples, our method allows samples to be both transported, according to the OT cost, and re-weighted, according to the information divergence. We demonstrate the effectiveness of our method on malware detection and image recognition applications and find that, to our knowledge, it outperforms existing methods at enhancing the robustness against adversarial attacks. $ARMOR_D$ yields the robustified accuracy of $98.29\%$ against $FGSM$ and $98.18\%$ aga
    
[^9]: 压缩和分布式最小二乘回归：收敛速度及其在联邦学习中的应用

    Compressed and distributed least-squares regression: convergence rates with applications to Federated Learning. (arXiv:2308.01358v1 [cs.LG])

    [http://arxiv.org/abs/2308.01358](http://arxiv.org/abs/2308.01358)

    本文研究了压缩对分布式和联邦学习中随机梯度算法的影响，通过比较不同的无偏压缩操作符的收敛速度，超越了经典的最坏情况分析。针对最小二乘回归，我们提出了一个随机逼近算法，并考虑了随机场的一般假设和噪声协方差的限制，以分析各种随机化机制。

    

    本文研究了在机器学习中广泛应用的分布式和联邦学习中，压缩对随机梯度算法的影响。我们强调了几种无偏压缩操作符之间的收敛速度差异，这些操作符都满足相同的方差条件，从而超越了经典的最坏情况分析。为此，我们专注于最小二乘回归（LSR）的情况，并分析了一个依赖于随机场的最小二乘回归的随机逼近算法。我们对随机场的一般性假设进行了详细分析（特别是期望的Hölder正则性）并对噪声协方差进行了限制，以便分析各种随机化机制，包括压缩。然后，我们将结果扩展到联邦学习的情况下。具体而言，我们强调了对加性噪声的协方差𝖢𝖠𝖭𝖨𝖠对收敛性的影响。

    In this paper, we investigate the impact of compression on stochastic gradient algorithms for machine learning, a technique widely used in distributed and federated learning. We underline differences in terms of convergence rates between several unbiased compression operators, that all satisfy the same condition on their variance, thus going beyond the classical worst-case analysis. To do so, we focus on the case of least-squares regression (LSR) and analyze a general stochastic approximation algorithm for minimizing quadratic functions relying on a random field. We consider weak assumptions on the random field, tailored to the analysis (specifically, expected H\"older regularity), and on the noise covariance, enabling the analysis of various randomizing mechanisms, including compression. We then extend our results to the case of federated learning.  More formally, we highlight the impact on the convergence of the covariance $\mathfrak{C}_{\mathrm{ania}}$ of the additive noise induced 
    
[^10]: 核岭回归下伪标签的协变量转移策略

    Pseudo-Labeling for Kernel Ridge Regression under Covariate Shift. (arXiv:2302.10160v2 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2302.10160](http://arxiv.org/abs/2302.10160)

    该论文提出了一种关于核岭回归的协变量转移策略，通过使用伪标签进行模型选择，能够适应不同特征分布下的学习，实现均方误差最小化。

    

    我们提出并分析了一种基于协变量转移的核岭回归方法。我们的目标是在目标分布上学习一个均方误差最小的回归函数，基于从目标分布采样的未标记数据和可能具有不同特征分布的已标记数据。我们将已标记数据分成两个子集，并分别进行核岭回归，以获得候选模型集合和一个填充模型。我们使用后者填充缺失的标签，然后相应地选择最佳的候选模型。我们的非渐近性过量风险界表明，在相当一般的情况下，我们的估计器能够适应目标分布以及协变量转移的结构。它能够实现渐近正态误差率直到对数因子的最小极限优化。在模型选择中使用伪标签不会产生主要负面影响。

    We develop and analyze a principled approach to kernel ridge regression under covariate shift. The goal is to learn a regression function with small mean squared error over a target distribution, based on unlabeled data from there and labeled data that may have a different feature distribution. We propose to split the labeled data into two subsets and conduct kernel ridge regression on them separately to obtain a collection of candidate models and an imputation model. We use the latter to fill the missing labels and then select the best candidate model accordingly. Our non-asymptotic excess risk bounds show that in quite general scenarios, our estimator adapts to the structure of the target distribution as well as the covariate shift. It achieves the minimax optimal error rate up to a logarithmic factor. The use of pseudo-labels in model selection does not have major negative impacts.
    

