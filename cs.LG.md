# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Mercer Large-Scale Kernel Machines from Ridge Function Perspective.](http://arxiv.org/abs/2307.11925) | 本文从岭函数的角度介绍了Mercer大规模核机器，通过研究哪些核函数可以被余弦函数乘积逼近，解决了大规模核机器中的难题。这些结果对深度学习，特别是图像处理有潜在应用。 |
| [^2] | [AI-Augmented Surveys: Leveraging Large Language Models for Opinion Prediction in Nationally Representative Surveys.](http://arxiv.org/abs/2305.09620) | 本论文研究了利用经过全国代表性调查微调的大语言模型（LLMs）来增强调查的观点预测，取得了在遗漏数据插值和回溯推理方面优秀的成果，在零次预测方面仍需进一步研究。 |
| [^3] | [The Score-Difference Flow for Implicit Generative Modeling.](http://arxiv.org/abs/2304.12906) | 本文提出了一种新的评分差异流模型(SD flow)，它可以最优地减少两个分布之间的散度，同时解决Schr​​ödinger桥问题。与去噪扩散模型不同，它没有对先验分布施加任何限制，在一些基准数据集中优于其他方法。 |

# 详细

[^1]: 基于岭函数角度的Mercer大规模核机器

    Mercer Large-Scale Kernel Machines from Ridge Function Perspective. (arXiv:2307.11925v1 [cs.LG])

    [http://arxiv.org/abs/2307.11925](http://arxiv.org/abs/2307.11925)

    本文从岭函数的角度介绍了Mercer大规模核机器，通过研究哪些核函数可以被余弦函数乘积逼近，解决了大规模核机器中的难题。这些结果对深度学习，特别是图像处理有潜在应用。

    

    为了从岭函数的角度介绍Mercer大规模核机器，我们回顾了Lin和Pinkus在岭函数的基本性上的结果。我们考虑了Rachimi和Recht在近似理论中的最近一篇论文的主要定理，即大规模核机器的随机特征。我们研究了哪些核函数可以被$x$和$y$的余弦函数乘积的和逼近，并提出了这种方法的障碍。本文的结果可能在深度学习中有各种应用，尤其是与图像处理相关的问题。

    To present Mercer large-scale kernel machines from a ridge function perspective, we recall the results by Lin and Pinkus from Fundamentality of ridge functions. We consider the main theorem of the recent paper by Rachimi and Recht, 2008, Random features for large-scale kernel machines in terms of the Approximation Theory. We study which kernels can be approximated by a sum of cosine function products with arguments depending on $x$ and $y$ and present the obstacles of such an approach. The results of this article may have various applications in Deep Learning, especially in problems related to Image Processing.
    
[^2]: AI增强的调查：利用大语言模型进行全国代表性调查的观点预测

    AI-Augmented Surveys: Leveraging Large Language Models for Opinion Prediction in Nationally Representative Surveys. (arXiv:2305.09620v1 [cs.CL])

    [http://arxiv.org/abs/2305.09620](http://arxiv.org/abs/2305.09620)

    本论文研究了利用经过全国代表性调查微调的大语言模型（LLMs）来增强调查的观点预测，取得了在遗漏数据插值和回溯推理方面优秀的成果，在零次预测方面仍需进一步研究。

    

    本论文研究了如何使用经过全国代表性调查微调的大语言模型（LLMs）来增强调查。本文探讨了LLMs在观点预测中，遗漏数据插值，回溯推理和零次预测三个不同应用。我们提出了一种新的方法论框架，将调查问题、个人信念和时间背景的神经嵌入引入到观点预测的个性化LLMs中。在1972年到2021年的“常规社会调查”中，我们从68,846名美国人中获得了3,110个二进制观点，在Alpaca-7b模型的基础上取得了最好的成果，在缺失数据插值（AUC=0.87，公开观点预测为$\rho$=0.99）和回溯推理（AUC=0.86，$\rho$=0.98）方面表现出色。这些显著的预测能力能够以高置信度填补缺失的趋势，并标明公众态度何时发生变化，如同性婚姻的获取支持。然而，在零次预测的情况下，模型的表现受到限制，需要进一步研究。

    How can we use large language models (LLMs) to augment surveys? This paper investigates three distinct applications of LLMs fine-tuned by nationally representative surveys for opinion prediction -- missing data imputation, retrodiction, and zero-shot prediction. We present a new methodological framework that incorporates neural embeddings of survey questions, individual beliefs, and temporal contexts to personalize LLMs in opinion prediction. Among 3,110 binarized opinions from 68,846 Americans in the General Social Survey from 1972 to 2021, our best models based on Alpaca-7b excels in missing data imputation (AUC = 0.87 for personal opinion prediction and $\rho$ = 0.99 for public opinion prediction) and retrodiction (AUC = 0.86, $\rho$ = 0.98). These remarkable prediction capabilities allow us to fill in missing trends with high confidence and pinpoint when public attitudes changed, such as the rising support for same-sex marriage. However, the models show limited performance in a zer
    
[^3]: 评分差值流模型用于隐式生成建模

    The Score-Difference Flow for Implicit Generative Modeling. (arXiv:2304.12906v1 [cs.LG])

    [http://arxiv.org/abs/2304.12906](http://arxiv.org/abs/2304.12906)

    本文提出了一种新的评分差异流模型(SD flow)，它可以最优地减少两个分布之间的散度，同时解决Schr​​ödinger桥问题。与去噪扩散模型不同，它没有对先验分布施加任何限制，在一些基准数据集中优于其他方法。

    

    隐式生成建模(IGM)旨在生成符合目标数据分布特征的合成数据样本。最近的研究(例如评分匹配网络、扩散模型)从通过环境空间中的动态扰动或流将合成源数据推向目标分布的角度解决了IGM问题。我们引入了任意目标和源分布之间的评分差异(SD)作为流，它可以最优地减少它们之间的Kullback-Leibler散度，同时解决Schr​​ödinger桥问题。我们将SD流应用于方便的代理分布，当且仅当原始分布对齐时，它们是对齐的。我们在某些条件下展示了这种公式与去噪扩散模型的形式一致性。然而，与扩散模型不同，SD流没有对先验分布施加任何限制。我们还表明，在无限辨别器能力的极限下，生成对抗网络的训练包含SD流。我们的实验表明，SD流在几个基准数据集上优于先前的最新技术。

    Implicit generative modeling (IGM) aims to produce samples of synthetic data matching the characteristics of a target data distribution. Recent work (e.g. score-matching networks, diffusion models) has approached the IGM problem from the perspective of pushing synthetic source data toward the target distribution via dynamical perturbations or flows in the ambient space. We introduce the score difference (SD) between arbitrary target and source distributions as a flow that optimally reduces the Kullback-Leibler divergence between them while also solving the Schr\"odinger bridge problem. We apply the SD flow to convenient proxy distributions, which are aligned if and only if the original distributions are aligned. We demonstrate the formal equivalence of this formulation to denoising diffusion models under certain conditions. However, unlike diffusion models, SD flow places no restrictions on the prior distribution. We also show that the training of generative adversarial networks includ
    

