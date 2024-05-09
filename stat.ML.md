# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Sensitivity-Aware Amortized Bayesian Inference.](http://arxiv.org/abs/2310.11122) | 本文提出了一种敏感性感知的摊销贝叶斯推断方法，通过权重共享和神经网络来进行似然和先验规范的训练，以及对数据扰动和预处理程序的敏感性评估。 |
| [^2] | [Contextualized Policy Recovery: Modeling and Interpreting Medical Decisions with Adaptive Imitation Learning.](http://arxiv.org/abs/2310.07918) | 本论文提出了一种上下文化政策恢复方法用于建模复杂的医疗决策过程，以解决现有模型在准确性和可解释性之间的权衡问题。该方法将决策策略拆分为上下文特定策略，通过多任务学习来实现建模，并提供复杂行为的简洁描述。 |
| [^3] | [Bayesian taut splines for estimating the number of modes.](http://arxiv.org/abs/2307.05825) | 本研究提出了一种贝叶斯紧系数样条方法，用于估计概率密度函数中模式的数量。该方法结合了核估计器和组合样条，实现了特征探索、模型选择和模式检验，并允许引入专家判断。通过在体育分析中的案例研究中的验证，证明了该方法的实用性。 |
| [^4] | [Entropic covariance models.](http://arxiv.org/abs/2306.03590) | 本文提出了一个通用的线性约束协方差矩阵变换的框架，并提出了一种估计方法，解决了一个凸问题，允许相对简单的渐近性和有限样本分析。研究的重点是关于建模相关矩阵和稀疏性方面的内容。 |
| [^5] | [A Study of Bayesian Neural Network Surrogates for Bayesian Optimization.](http://arxiv.org/abs/2305.20028) | 本文研究贝叶斯神经网络替代高斯过程模型作为贝叶斯优化中的代理模型，并在多个基准问题上证明了其优于标准GP代理的能力。 |
| [^6] | [StepMix: A Python Package for Pseudo-Likelihood Estimation of Generalized Mixture Models with External Variables.](http://arxiv.org/abs/2304.03853) | StepMix是一个用于外部变量广义混合模型的伪似然估计的Python包，提供了单步和逐步估计方法，帮助从业人员进行模型估计、选择和解释。 |
| [^7] | [Investigating the Impact of Model Width and Density on Generalization in Presence of Label Noise.](http://arxiv.org/abs/2208.08003) | 本文研究发现标签噪声会导致双丘降曲线出现“最终上升”，即在足够大的噪声样本比率下，中等宽度下实现最佳泛化性能。随机丢弃可训练参数来减少密度可在标签噪声下改善泛化性能。 |

# 详细

[^1]: 敏感性感知的摊销贝叶斯推断

    Sensitivity-Aware Amortized Bayesian Inference. (arXiv:2310.11122v1 [stat.ML])

    [http://arxiv.org/abs/2310.11122](http://arxiv.org/abs/2310.11122)

    本文提出了一种敏感性感知的摊销贝叶斯推断方法，通过权重共享和神经网络来进行似然和先验规范的训练，以及对数据扰动和预处理程序的敏感性评估。

    

    贝叶斯推断是在不确定性下进行概率推理和决策的强大框架。现代贝叶斯工作流程中的基本选择涉及似然函数和先验分布的规范、后验逼近器和数据。每个选择都可以显着影响基于模型的推断和后续决策，因此需要进行敏感性分析。在这项工作中，我们提出了一种多方面的方法，将敏感性分析整合到摊销贝叶斯推断（ABI，即基于神经网络的模拟推断）中。首先，我们利用权重共享在训练过程中编码替代似然和先验规范之间的结构相似性，以最小的计算开销。其次，我们利用神经网络的快速推断来评估对各种数据扰动或预处理程序的敏感性。与大多数其他贝叶斯方法相比，这两个步骤都避免了昂贵的计算。

    Bayesian inference is a powerful framework for making probabilistic inferences and decisions under uncertainty. Fundamental choices in modern Bayesian workflows concern the specification of the likelihood function and prior distributions, the posterior approximator, and the data. Each choice can significantly influence model-based inference and subsequent decisions, thereby necessitating sensitivity analysis. In this work, we propose a multifaceted approach to integrate sensitivity analyses into amortized Bayesian inference (ABI, i.e., simulation-based inference with neural networks). First, we utilize weight sharing to encode the structural similarities between alternative likelihood and prior specifications in the training process with minimal computational overhead. Second, we leverage the rapid inference of neural networks to assess sensitivity to various data perturbations or pre-processing procedures. In contrast to most other Bayesian approaches, both steps circumvent the costly
    
[^2]: 上下文化政策恢复：通过自适应模仿学习对医疗决策进行建模和解释

    Contextualized Policy Recovery: Modeling and Interpreting Medical Decisions with Adaptive Imitation Learning. (arXiv:2310.07918v1 [cs.LG])

    [http://arxiv.org/abs/2310.07918](http://arxiv.org/abs/2310.07918)

    本论文提出了一种上下文化政策恢复方法用于建模复杂的医疗决策过程，以解决现有模型在准确性和可解释性之间的权衡问题。该方法将决策策略拆分为上下文特定策略，通过多任务学习来实现建模，并提供复杂行为的简洁描述。

    

    可解释的策略学习旨在从观察到的行为中估计可理解的决策策略；然而，现有模型在准确性和可解释性之间存在权衡。这种权衡限制了基于数据驱动的对人类决策过程的解释，例如，审计医疗决策的偏见和次优实践，我们需要决策过程的模型，能够提供复杂行为的简洁描述。现有方法基本上由于将潜在决策过程表示为通用策略而负担了这种权衡，而实际上人类决策是动态的，可以随上下文信息而大幅改变。因此，我们提出了上下文化政策恢复（CPR），将建模复杂决策过程的问题重新定义为多任务学习问题，其中复杂决策策略由特定上下文的策略组成。CPR将每个上下文特定策略建模为线性的观察-动作映射

    Interpretable policy learning seeks to estimate intelligible decision policies from observed actions; however, existing models fall short by forcing a tradeoff between accuracy and interpretability. This tradeoff limits data-driven interpretations of human decision-making process. e.g. to audit medical decisions for biases and suboptimal practices, we require models of decision processes which provide concise descriptions of complex behaviors. Fundamentally, existing approaches are burdened by this tradeoff because they represent the underlying decision process as a universal policy, when in fact human decisions are dynamic and can change drastically with contextual information. Thus, we propose Contextualized Policy Recovery (CPR), which re-frames the problem of modeling complex decision processes as a multi-task learning problem in which complex decision policies are comprised of context-specific policies. CPR models each context-specific policy as a linear observation-to-action mapp
    
[^3]: 贝叶斯紧系数样条估计模式的数量

    Bayesian taut splines for estimating the number of modes. (arXiv:2307.05825v1 [stat.ME])

    [http://arxiv.org/abs/2307.05825](http://arxiv.org/abs/2307.05825)

    本研究提出了一种贝叶斯紧系数样条方法，用于估计概率密度函数中模式的数量。该方法结合了核估计器和组合样条，实现了特征探索、模型选择和模式检验，并允许引入专家判断。通过在体育分析中的案例研究中的验证，证明了该方法的实用性。

    

    概率密度函数中模式的数量代表模型的复杂性，也可以看作现有亚群体的数量。尽管其相关性，对其估计的研究非常有限。我们针对单变量情况提出一个新颖的方法，致力于预测准确性，受到了问题的一些被忽视的方面的启发。我们认为解决方案需要结构，模式的主观且不确定性，以及融合全局和局部密度特性的整体视图的便利性。我们的方法结合了灵活的核估计器和简洁的组合样条。特征探索、模型选择和模式检验都在贝叶斯推理范式中实现，为软解决方案提供了便利，并允许在过程中引入专家判断。我们的提议的实用性通过在体育分析中的案例研究中进行了验证，并展示了多个陪伴的可视化。

    The number of modes in a probability density function is representative of the model's complexity and can also be viewed as the number of existing subpopulations. Despite its relevance, little research has been devoted to its estimation. Focusing on the univariate setting, we propose a novel approach targeting prediction accuracy inspired by some overlooked aspects of the problem. We argue for the need for structure in the solutions, the subjective and uncertain nature of modes, and the convenience of a holistic view blending global and local density properties. Our method builds upon a combination of flexible kernel estimators and parsimonious compositional splines. Feature exploration, model selection and mode testing are implemented in the Bayesian inference paradigm, providing soft solutions and allowing to incorporate expert judgement in the process. The usefulness of our proposal is illustrated through a case study in sports analytics, showcasing multiple companion visualisation 
    
[^4]: 熵协方差模型

    Entropic covariance models. (arXiv:2306.03590v1 [math.ST])

    [http://arxiv.org/abs/2306.03590](http://arxiv.org/abs/2306.03590)

    本文提出了一个通用的线性约束协方差矩阵变换的框架，并提出了一种估计方法，解决了一个凸问题，允许相对简单的渐近性和有限样本分析。研究的重点是关于建模相关矩阵和稀疏性方面的内容。

    

    在协方差矩阵估计中，找到合适的模型和有效的估计方法是一项挑战。文献中通常采用两种方法，一种是对协方差矩阵或其逆施加线性约束，另一种是考虑施加在协方差矩阵的矩阵对数上的线性约束。本文提出了一个通用的线性约束协方差矩阵变换的框架，包括上述例子。我们提出的估计方法解决了一个凸问题，并产生了一个M估计量，允许相对简单的渐近性和有限样本分析。在开发了一般理论之后，我们集中在建模相关矩阵和稀疏性方面。我们的几何洞察力允许我们扩展协方差矩阵建模中的一些最新结果。这包括提供相关矩阵空间的无限制参数化，这是一种替代利用变换的最新结果。我们还展示了如何对协方差矩阵的Cholesky因子施加稀疏性限制，这与现有方法不同。

    In covariance matrix estimation, one of the challenges lies in finding a suitable model and an efficient estimation method. Two commonly used approaches in the literature involve imposing linear restrictions on the covariance matrix or its inverse. Another approach considers linear restrictions on the matrix logarithm of the covariance matrix. In this paper, we present a general framework for linear restrictions on different transformations of the covariance matrix, including the mentioned examples. Our proposed estimation method solves a convex problem and yields an M-estimator, allowing for relatively straightforward asymptotic and finite sample analysis. After developing the general theory, we focus on modelling correlation matrices and on sparsity. Our geometric insights allow to extend various recent results in covariance matrix modelling. This includes providing unrestricted parametrizations of the space of correlation matrices, which is alternative to a recent result utilizing t
    
[^5]: 贝叶斯神经网络替代贝叶斯优化中的高斯过程模型

    A Study of Bayesian Neural Network Surrogates for Bayesian Optimization. (arXiv:2305.20028v1 [cs.LG])

    [http://arxiv.org/abs/2305.20028](http://arxiv.org/abs/2305.20028)

    本文研究贝叶斯神经网络替代高斯过程模型作为贝叶斯优化中的代理模型，并在多个基准问题上证明了其优于标准GP代理的能力。

    

    贝叶斯优化是一种高效的优化方法，适用于难以查询的目标函数。这些目标函数通常由高斯过程（GP）代理模型表示，其易于优化并支持精确推理。虽然标准的GP代理已经在贝叶斯优化中被广泛应用，但贝叶斯神经网络（BNNs）最近成为了一个实用的函数逼近器，与标准的GP相比具有许多优点，例如天然处理非平稳性以及学习高维数据的表示。在本文中，我们研究了BNN作为标准GP代理的替代品。我们考虑了各种有限宽度BNN的近似推理过程，包括高质量Hamiltonian Monte Carlo，低成本的随机MCMC和启发式方法（如深度集成）。我们还考虑了无限宽度BNN和部分随机模型，例如深度核学习。我们评估了这些代理模型在多个基准问题上的表现，并证明它们在某些情况下可以优于标准GP代理。我们的结果表明，BNN是传统代理模型在贝叶斯优化中的一个很有前途的替代选择。

    Bayesian optimization is a highly efficient approach to optimizing objective functions which are expensive to query. These objectives are typically represented by Gaussian process (GP) surrogate models which are easy to optimize and support exact inference. While standard GP surrogates have been well-established in Bayesian optimization, Bayesian neural networks (BNNs) have recently become practical function approximators, with many benefits over standard GPs such as the ability to naturally handle non-stationarity and learn representations for high-dimensional data. In this paper, we study BNNs as alternatives to standard GP surrogates for optimization. We consider a variety of approximate inference procedures for finite-width BNNs, including high-quality Hamiltonian Monte Carlo, low-cost stochastic MCMC, and heuristics such as deep ensembles. We also consider infinite-width BNNs and partially stochastic models such as deep kernel learning. We evaluate this collection of surrogate mod
    
[^6]: StepMix: 一个用于外部变量广义混合模型的伪似然估计的Python包

    StepMix: A Python Package for Pseudo-Likelihood Estimation of Generalized Mixture Models with External Variables. (arXiv:2304.03853v1 [stat.ME])

    [http://arxiv.org/abs/2304.03853](http://arxiv.org/abs/2304.03853)

    StepMix是一个用于外部变量广义混合模型的伪似然估计的Python包，提供了单步和逐步估计方法，帮助从业人员进行模型估计、选择和解释。

    

    StepMix是一个用于广义有限混合模型(潜在剖面和潜在类分析)与外部变量(协变量和远程结果)的伪似然估计(单步、两步和三步方法)的开源软件包。在许多社会科学的应用中，主要目标不仅是将个体聚类成潜在类别，还包括使用这些类别来开发更复杂的统计模型。这些模型通常分为一个将潜在类别与观察指标相关联的测量模型和一个将协变量和结果变量与潜在类别相关联的结构模型。测量和结构模型可以使用所谓的一步法共同估计，也可以使用逐步方法逐步估计，对于从业人员来说，这些方法在估计潜在类别的可解释性方面具有显著优势。除了一步法，StepMix还实现了文献中提出的最重要的逐步估计方法，提供了用户友好的界面，方便模型的估计、选择和解释。

    StepMix is an open-source software package for the pseudo-likelihood estimation (one-, two- and three-step approaches) of generalized finite mixture models (latent profile and latent class analysis) with external variables (covariates and distal outcomes). In many applications in social sciences, the main objective is not only to cluster individuals into latent classes, but also to use these classes to develop more complex statistical models. These models generally divide into a measurement model that relates the latent classes to observed indicators, and a structural model that relates covariates and outcome variables to the latent classes. The measurement and structural models can be estimated jointly using the so-called one-step approach or sequentially using stepwise methods, which present significant advantages for practitioners regarding the interpretability of the estimated latent classes. In addition to the one-step approach, StepMix implements the most important stepwise estim
    
[^7]: 研究模型宽度和密度对标签噪声下泛化性能的影响

    Investigating the Impact of Model Width and Density on Generalization in Presence of Label Noise. (arXiv:2208.08003v4 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2208.08003](http://arxiv.org/abs/2208.08003)

    本文研究发现标签噪声会导致双丘降曲线出现“最终上升”，即在足够大的噪声样本比率下，中等宽度下实现最佳泛化性能。随机丢弃可训练参数来减少密度可在标签噪声下改善泛化性能。

    

    扩大过参数化神经网络的规模是实现最先进性能的关键。这是通过双丘降现象捕捉的，其中测试损失随着模型宽度的增加呈现出降低-增加-降低的模式。然而，标签噪声对测试损失曲线的影响尚未被充分探索。在本文中，我们揭示了一个有趣的现象，即标签噪声导致原本观察到的双丘降曲线出现了“最终上升”。具体而言，在足够大的噪声样本比率下，中等宽度下实现最佳泛化性能。通过理论分析，我们将这种现象归因于标签噪声引起的测试损失方差形状转换。此外，我们将最终上升现象扩展到模型密度，并提供了第一个理论表征，表明随机丢弃可训练参数来减少密度可在标签噪声下改善泛化性能。

    Increasing the size of overparameterized neural networks has been a key in achieving state-of-the-art performance. This is captured by the double descent phenomenon, where the test loss follows a decreasing-increasing-decreasing pattern as model width increases. However, the effect of label noise on the test loss curve has not been fully explored. In this work, we uncover an intriguing phenomenon where label noise leads to a \textit{final ascent} in the originally observed double descent curve. Specifically, under a sufficiently large noise-to-sample-size ratio, optimal generalization is achieved at intermediate widths. Through theoretical analysis, we attribute this phenomenon to the shape transition of test loss variance induced by label noise. Furthermore, we extend the final ascent phenomenon to model density and provide the first theoretical characterization showing that reducing density by randomly dropping trainable parameters improves generalization under label noise. We also t
    

