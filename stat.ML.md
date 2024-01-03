# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Double Normalizing Flows: Flexible Bayesian Gaussian Process ODEs Learning.](http://arxiv.org/abs/2309.09222) | 这项研究将标准化流引入高斯过程常微分方程(ODE)模型，使其具备更灵活和表达性强的先验分布和非高斯的后验推断，从而提高了贝叶斯高斯过程ODE的准确性和不确定性估计。 |
| [^2] | [SLEM: Machine Learning for Path Modeling and Causal Inference with Super Learner Equation Modeling.](http://arxiv.org/abs/2308.04365) | SLEM是一种路径建模技术，通过集成机器学习超级学习者，实现了一致且无偏的因果效应估计，并在处理非线性关系时超过了传统的结构方程模型。 |
| [^3] | [Adaptive learning of density ratios in RKHS.](http://arxiv.org/abs/2307.16164) | 该论文研究在再生核希尔伯特空间中的一类密度比率估计方法，提出了一种自适应学习的参数选择原则，并在有限样本情况下推导出新的误差界。其方法在二次损失的情况下实现了极小化最优误差率。 |
| [^4] | [The contextual lasso: Sparse linear models via deep neural networks.](http://arxiv.org/abs/2302.00878) | 本论文提出了一种新的统计估计器——上下文套索，可以通过深度神经网络的方法解决解释性和拟合能力的矛盾问题，实现对可解释特征的稀疏拟合，并且稀疏模式和系数会随着上下文特征的变化而发生变化。 |
| [^5] | [Lossy Image Compression with Conditional Diffusion Models.](http://arxiv.org/abs/2209.06950) | 本文提出了一种利用条件扩散模型进行有损图像压缩的优化框架。通过引入额外的内容潜变量以及合成纹理变量，该方法在图像质量评估指标上表现出更强的性能。 |

# 详细

[^1]: 双重标准化流：灵活的贝叶斯高斯过程ODE学习

    Double Normalizing Flows: Flexible Bayesian Gaussian Process ODEs Learning. (arXiv:2309.09222v1 [cs.LG])

    [http://arxiv.org/abs/2309.09222](http://arxiv.org/abs/2309.09222)

    这项研究将标准化流引入高斯过程常微分方程(ODE)模型，使其具备更灵活和表达性强的先验分布和非高斯的后验推断，从而提高了贝叶斯高斯过程ODE的准确性和不确定性估计。

    

    最近，高斯过程被用来建模连续动力系统的向量场。对于这样的模型，贝叶斯推断已经得到了广泛研究，并应用于时间序列预测等任务，提供不确定性估计。然而，先前的高斯过程常微分方程(ODE)模型在具有非高斯过程先验的数据集上可能表现不佳，因为它们的约束先验和均值场后验可能缺乏灵活性。为了解决这个限制，我们引入了标准化流来重新参数化ODE的向量场，从而得到一个更灵活、更表达性的先验分布。此外，由于标准化流的解析可计算的概率密度函数，我们将它们应用于GP ODE的后验推断，生成一个非高斯的后验。通过这些标准化流的双重应用，我们的模型在贝叶斯高斯过程ODE中提高了准确性和不确定性估计。

    Recently, Gaussian processes have been utilized to model the vector field of continuous dynamical systems. Bayesian inference for such models \cite{hegde2022variational} has been extensively studied and has been applied in tasks such as time series prediction, providing uncertain estimates. However, previous Gaussian Process Ordinary Differential Equation (ODE) models may underperform on datasets with non-Gaussian process priors, as their constrained priors and mean-field posteriors may lack flexibility. To address this limitation, we incorporate normalizing flows to reparameterize the vector field of ODEs, resulting in a more flexible and expressive prior distribution. Additionally, due to the analytically tractable probability density functions of normalizing flows, we apply them to the posterior inference of GP ODEs, generating a non-Gaussian posterior. Through these dual applications of normalizing flows, our model improves accuracy and uncertainty estimates for Bayesian Gaussian P
    
[^2]: SLEM：机器学习用于路径建模和因果推断的超级学习者方程模型

    SLEM: Machine Learning for Path Modeling and Causal Inference with Super Learner Equation Modeling. (arXiv:2308.04365v1 [stat.ML])

    [http://arxiv.org/abs/2308.04365](http://arxiv.org/abs/2308.04365)

    SLEM是一种路径建模技术，通过集成机器学习超级学习者，实现了一致且无偏的因果效应估计，并在处理非线性关系时超过了传统的结构方程模型。

    

    因果推断是科学的关键目标，使研究人员能够通过观察数据得出关于对假定干预的预测的有意义的结论。路径模型、结构方程模型(SEMs)以及更一般的有向无环图(DAGs)能够明确地指定关于现象背后的因果结构的假设。与DAGs不同，SEMs假设线性关系，这可能导致函数错误规范，从而阻碍研究人员进行可靠的效果大小估计。相反，我们提出了超级学习者方程模型（SLEM），一种集成了机器学习超级学习者集成的路径建模技术。我们通过实证研究，证明了SLEM能够提供一致且无偏的因果效应估计，在与SEMs进行线性模型比较时表现出竞争力，并且在处理非线性关系时优于SEMs。

    Causal inference is a crucial goal of science, enabling researchers to arrive at meaningful conclusions regarding the predictions of hypothetical interventions using observational data. Path models, Structural Equation Models (SEMs), and, more generally, Directed Acyclic Graphs (DAGs), provide a means to unambiguously specify assumptions regarding the causal structure underlying a phenomenon. Unlike DAGs, which make very few assumptions about the functional and parametric form, SEM assumes linearity. This can result in functional misspecification which prevents researchers from undertaking reliable effect size estimation. In contrast, we propose Super Learner Equation Modeling, a path modeling technique integrating machine learning Super Learner ensembles. We empirically demonstrate its ability to provide consistent and unbiased estimates of causal effects, its competitive performance for linear models when compared with SEM, and highlight its superiority over SEM when dealing with non
    
[^3]: 在RKHS中自适应学习密度比率

    Adaptive learning of density ratios in RKHS. (arXiv:2307.16164v1 [cs.LG])

    [http://arxiv.org/abs/2307.16164](http://arxiv.org/abs/2307.16164)

    该论文研究在再生核希尔伯特空间中的一类密度比率估计方法，提出了一种自适应学习的参数选择原则，并在有限样本情况下推导出新的误差界。其方法在二次损失的情况下实现了极小化最优误差率。

    

    从有限数量的密度观测中估计两个概率密度的比率是机器学习和统计学中的一个核心问题，应用包括双样本检验、分歧估计、生成建模、协变量转移适应、条件密度估计和新颖性检测。本研究分析了一大类密度比率估计方法，它们通过在再生核希尔伯特空间（RKHS）中最小化真实密度比率与模型之间的正则Bregman距离。我们推导出新的有限样本误差界，并提出了一种Lepskii类型的参数选择原则，在不知道密度比率的正则性的情况下最小化误差界。在二次损失的特殊情况下，我们的方法自适应地实现了极小化最优误差率。提供了一个数值示例。

    Estimating the ratio of two probability densities from finitely many observations of the densities is a central problem in machine learning and statistics with applications in two-sample testing, divergence estimation, generative modeling, covariate shift adaptation, conditional density estimation, and novelty detection. In this work, we analyze a large class of density ratio estimation methods that minimize a regularized Bregman divergence between the true density ratio and a model in a reproducing kernel Hilbert space (RKHS). We derive new finite-sample error bounds, and we propose a Lepskii type parameter choice principle that minimizes the bounds without knowledge of the regularity of the density ratio. In the special case of quadratic loss, our method adaptively achieves a minimax optimal error rate. A numerical illustration is provided.
    
[^4]: 上下文套索：通过深度神经网络的方法实现稀疏线性模型

    The contextual lasso: Sparse linear models via deep neural networks. (arXiv:2302.00878v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2302.00878](http://arxiv.org/abs/2302.00878)

    本论文提出了一种新的统计估计器——上下文套索，可以通过深度神经网络的方法解决解释性和拟合能力的矛盾问题，实现对可解释特征的稀疏拟合，并且稀疏模式和系数会随着上下文特征的变化而发生变化。

    

    稀疏线性模型是可解释机器学习的黄金标准工具，本论文通过使用深度神经网络对稀疏线性模型进行改进，实现了可解释性和强大的拟合能力。上下文套索是一种新的统计估计器，它将输入特征分成可解释特征和上下文特征两组，并对可解释特征进行稀疏拟合，同时其稀疏模式和系数会随着上下文特征的变化而发生变化，这个过程通过深度神经网络无需参数地进行学习。

    Sparse linear models are a gold standard tool for interpretable machine learning, a field of emerging importance as predictive models permeate decision-making in many domains. Unfortunately, sparse linear models are far less flexible as functions of their input features than black-box models like deep neural networks. With this capability gap in mind, we study a not-uncommon situation where the input features dichotomize into two groups: explanatory features, which are candidates for inclusion as variables in an interpretable model, and contextual features, which select from the candidate variables and determine their effects. This dichotomy leads us to the contextual lasso, a new statistical estimator that fits a sparse linear model to the explanatory features such that the sparsity pattern and coefficients vary as a function of the contextual features. The fitting process learns this function nonparametrically via a deep neural network. To attain sparse coefficients, we train the net
    
[^5]: 基于条件扩散模型的有损图像压缩

    Lossy Image Compression with Conditional Diffusion Models. (arXiv:2209.06950v5 [eess.IV] UPDATED)

    [http://arxiv.org/abs/2209.06950](http://arxiv.org/abs/2209.06950)

    本文提出了一种利用条件扩散模型进行有损图像压缩的优化框架。通过引入额外的内容潜变量以及合成纹理变量，该方法在图像质量评估指标上表现出更强的性能。

    

    本文提出了一种利用扩散生成模型的端到端优化的有损图像压缩框架。该方法基于变换编码范式，将图像映射到潜在空间进行信息熵编码，然后再映射回数据空间进行重构。与基于变分自编码器(VAE)的神经压缩方法不同，我们的解码器是一个条件扩散模型。因此，我们的方法引入了一个额外的“内容”潜变量，反向扩散过程会对其进行条件化，并利用该变量存储图像信息。决定扩散过程的剩余“纹理”变量会在解码时合成。通过实验，我们展示了模型的性能可以根据感知度量进行调整。我们广泛的实验涉及了多个数据集和图像质量评估指标，结果表明我们的方法相较于基于生成对抗网络的方法能够得到更好的FID分数。

    This paper outlines an end-to-end optimized lossy image compression framework using diffusion generative models. The approach relies on the transform coding paradigm, where an image is mapped into a latent space for entropy coding and, from there, mapped back to the data space for reconstruction. In contrast to VAE-based neural compression, where the (mean) decoder is a deterministic neural network, our decoder is a conditional diffusion model. Our approach thus introduces an additional "content" latent variable on which the reverse diffusion process is conditioned and uses this variable to store information about the image. The remaining "texture" variables characterizing the diffusion process are synthesized at decoding time. We show that the model's performance can be tuned toward perceptual metrics of interest. Our extensive experiments involving multiple datasets and image quality assessment metrics show that our approach yields stronger reported FID scores than the GAN-based mode
    

