# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Universal representations for financial transactional data: embracing local, global, and external contexts](https://arxiv.org/abs/2404.02047) | 提出了一个金融交易数据通用表示的学习框架，结合了本地、全局和外部语境，提出了新颖的生成模型和整合外部信息的方法，并在本地任务中表现出超越性能。 |
| [^2] | [Can LLMs Separate Instructions From Data? And What Do We Even Mean By That?](https://arxiv.org/abs/2403.06833) | 本研究提出了一种形式化的度量来量化指令与数据分离现象，以及一种可以从模型黑盒输出计算的经验变量，并引入了新数据集SEP，用于评估 |
| [^3] | [SecGPT: An Execution Isolation Architecture for LLM-Based Systems](https://arxiv.org/abs/2403.04960) | 提出了一种面向LLM系统的执行隔离架构SecGPT，旨在解决第三方应用程序执行所引发的安全和隐私问题 |
| [^4] | [Random features and polynomial rules](https://arxiv.org/abs/2402.10164) | 本论文分析了随机特征模型在具有高斯数据的一般监督学习问题中的泛化性能，并将随机特征模型映射到等效的多项式模型，得到了与严格界限和数值实验一致的结果。 |
| [^5] | [Statistics without Interpretation: A Sober Look at Explainable Machine Learning](https://arxiv.org/abs/2402.02870) | 解释算法往往数学上复杂且难以解释，这导致解释错误。为了向前推进，解释算法需要明确其输出的解释方式，并澄清可以和不能回答的问题。这一论点基于统计学和解释之间的区别，以及可解释机器学习和应用统计学之间的相似性。 |
| [^6] | [Provably Stable Feature Rankings with SHAP and LIME.](http://arxiv.org/abs/2401.15800) | 这项研究提出了一种通过利用多重假设检验的思想，来设计可靠地排名机器学习模型中最重要特征的特征归因方法，旨在解决SHAP和LIME等常用方法由于随机采样导致的高度不稳定性问题。实验证明了该方法的有效性和计算效率。 |
| [^7] | [Private Fine-tuning of Large Language Models with Zeroth-order Optimization.](http://arxiv.org/abs/2401.04343) | 引入了DP-ZO，一种通过私有化零阶优化来保护大型语言模型训练数据隐私的方法。 |
| [^8] | [Theoretical guarantees on the best-of-n alignment policy.](http://arxiv.org/abs/2401.01879) | 该论文研究了对齐生成模型的最佳n对齐策略，并证明了之前文献中的某个分析表达式是错误的。研究者们提出了一个新的KL散度估计方法，并通过实验证明其有效性。 |
| [^9] | [Maximum Weight Entropy.](http://arxiv.org/abs/2309.15704) | 本文提出了在深度学习中使用最大熵原理的最大权重熵方法，通过最大化权重多样性来解决标准方法在超出分布情况下预测多样性缺乏的问题。 |
| [^10] | [Potential and limitations of random Fourier features for dequantizing quantum machine learning.](http://arxiv.org/abs/2309.11647) | 本文研究了随机傅里叶特征在去量化量子机器学习中的潜力与局限性，并在回归问题上确立了其高效去量化的必要和充分条件，并提出了PQC架构设计建议和识别了潜在量子优势的必要结构。 |
| [^11] | [Enhancing Cell Tracking with a Time-Symmetric Deep Learning Approach.](http://arxiv.org/abs/2308.03887) | 本论文提出了一种使用时间对称的深度学习方法来提升细胞跟踪的准确性。该方法不依赖于连续帧跟踪，而是基于细胞的时空邻域进行跟踪，具有学习细胞运动模式的能力，并能处理具有严重伪影的大量视频帧。 |
| [^12] | [Equivariant Spherical CNN for Data Efficient and High-Performance Medical Image Processing.](http://arxiv.org/abs/2307.03298) | 等变球卷积神经网络能够提高医学图像处理的效率和性能，通过降低对特定训练集的依赖性，并且在去噪和重建方面表现出卓越的质量和计算效率。 |

# 详细

[^1]: 金融交易数据的通用表示：融合本地、全局和外部语境

    Universal representations for financial transactional data: embracing local, global, and external contexts

    [https://arxiv.org/abs/2404.02047](https://arxiv.org/abs/2404.02047)

    提出了一个金融交易数据通用表示的学习框架，结合了本地、全局和外部语境，提出了新颖的生成模型和整合外部信息的方法，并在本地任务中表现出超越性能。

    

    金融交易的有效处理对银行数据分析至关重要。然而，在这一领域中，大多数方法专注于为独立问题提供专门化解决方案，而不是构建适用于许多问题的通用表示。我们提出了一个表示学习框架，旨在解决各种企业挑战。我们还提出了考虑数据特定性的新颖生成模型，并提出了一种整合外部信息到客户表示的方式，借鉴其他客户行动的见解。最后，我们提供了一个基准，描述了全球范围内的表示质量，涉及整个交易历史；本地范围内，反映客户当前状态；动态范围内，捕捉表示随时间演变的情况。我们的生成方法在本地任务中表现出色，对于下一个MCC预测任务的ROC-AUC提升高达14％，对于dow...

    arXiv:2404.02047v1 Announce Type: cross  Abstract: Effective processing of financial transactions is essential for banking data analysis. However, in this domain, most methods focus on specialized solutions to stand-alone problems instead of constructing universal representations suitable for many problems. We present a representation learning framework that addresses diverse business challenges. We also suggest novel generative models that account for data specifics, and a way to integrate external information into a client's representation, leveraging insights from other customers' actions. Finally, we offer a benchmark, describing representation quality globally, concerning the entire transaction history; locally, reflecting the client's current state; and dynamically, capturing representation evolution over time. Our generative approach demonstrates superior performance in local tasks, with an increase in ROC-AUC of up to 14\% for the next MCC prediction task and up to 46\% for dow
    
[^2]: LLMs能够将指令与数据分离吗？我们具体指的是什么？

    Can LLMs Separate Instructions From Data? And What Do We Even Mean By That?

    [https://arxiv.org/abs/2403.06833](https://arxiv.org/abs/2403.06833)

    本研究提出了一种形式化的度量来量化指令与数据分离现象，以及一种可以从模型黑盒输出计算的经验变量，并引入了新数据集SEP，用于评估

    

    arXiv:2403.06833v1 公告类型: 跨 针对大型语言模型（LLMs）进行调节指令的技术取得了突破性的成果，为许多实际应用打开了无数新可能。然而，LLMs缺乏其他计算机科学领域已建立为规范的基本安全特性，比如指令与数据之间的分离，导致它们发生故障或易受第三方操控和干扰（例如通过间接提示/命令注入）。更糟糕的是，迄今为止，甚至没有确切定义这种分离究竟意味着什么以及如何测试其违反情况。本研究旨在填补这一空白。我们引入了一个正式的指标来量化指令与数据分离现象，以及一个可以从模型的黑盒输出计算的经验变量。我们还介绍了一个新的数据集SEP（应该执行还是处理？），该数据集允许评估

    arXiv:2403.06833v1 Announce Type: cross  Abstract: Instruction-tuned Large Language Models (LLMs) have achieved breakthrough results, opening countless new possibilities for many practical applications. However, LLMs lack elementary safety features that are established norms in other areas of computer science, such as the separation between instructions and data, causing them to malfunction or rendering them vulnerable to manipulation and interference by third parties e.g., via indirect prompt/command injection. Even worse, so far, there is not even an established definition of what precisely such a separation would mean and how its violation could be tested. In this work, we aim to close this gap. We introduce a formal measure to quantify the phenomenon of instruction-data separation as well as an empirical variant of the measure that can be computed from a model`s black-box outputs. We also introduce a new dataset, SEP (Should it be Executed or Processed?), which allows estimating th
    
[^3]: SecGPT：一种面向基于LLM系统的执行隔离架构

    SecGPT: An Execution Isolation Architecture for LLM-Based Systems

    [https://arxiv.org/abs/2403.04960](https://arxiv.org/abs/2403.04960)

    提出了一种面向LLM系统的执行隔离架构SecGPT，旨在解决第三方应用程序执行所引发的安全和隐私问题

    

    大型语言模型（LLMs）被扩展为系统，如ChatGPT，已经开始支持第三方应用程序。这些LLM应用程序利用LLMs的事实上基于自然语言的自动执行范式：即，应用程序及其交互是用自然语言定义的，提供对用户数据的访问，并被允许自由地相互交互以及与系统互动。这些LLM应用程序生态系统类似于早期计算平台的设置，在那里应用程序和系统之间缺乏足够的隔离。由于第三方应用程序可能不可信，并且受自然语言界面的不精确性加剧，当前的设计会为用户带来安全和隐私风险。在本文中，我们提出了SecGPT，一种面向LLM系统的架构，旨在缓解由第三方应用程序执行引起的安全性和隐私问题。SecGPT的关键思想是隔离应用程序的执行和更多的预

    arXiv:2403.04960v1 Announce Type: cross  Abstract: Large language models (LLMs) extended as systems, such as ChatGPT, have begun supporting third-party applications. These LLM apps leverage the de facto natural language-based automated execution paradigm of LLMs: that is, apps and their interactions are defined in natural language, provided access to user data, and allowed to freely interact with each other and the system. These LLM app ecosystems resemble the settings of earlier computing platforms, where there was insufficient isolation between apps and the system. Because third-party apps may not be trustworthy, and exacerbated by the imprecision of the natural language interfaces, the current designs pose security and privacy risks for users. In this paper, we propose SecGPT, an architecture for LLM-based systems that aims to mitigate the security and privacy issues that arise with the execution of third-party apps. SecGPT's key idea is to isolate the execution of apps and more pre
    
[^4]: 随机特征和多项式规则

    Random features and polynomial rules

    [https://arxiv.org/abs/2402.10164](https://arxiv.org/abs/2402.10164)

    本论文分析了随机特征模型在具有高斯数据的一般监督学习问题中的泛化性能，并将随机特征模型映射到等效的多项式模型，得到了与严格界限和数值实验一致的结果。

    

    随机特征模型在深度学习理论中起着重要的作用，描述了神经网络接近无限宽度极限时的行为。在本工作中，我们对具有高斯数据的一般监督学习问题的随机特征模型的泛化性能进行了详细分析。我们利用无序系统的统计力学工具将随机特征模型映射到等效的多项式模型，并绘制了平均泛化曲线作为问题的两个主要控制参数的函数：随机特征的数量N和训练集的大小P，假设它们都按照输入维度D的幂进行缩放。我们的结果扩展了N，P和D之间比例缩放的情况。它们与特定学习任务已知的严格界限一致，并与数值实验定量一致。

    arXiv:2402.10164v1 Announce Type: cross  Abstract: Random features models play a distinguished role in the theory of deep learning, describing the behavior of neural networks close to their infinite-width limit. In this work, we present a thorough analysis of the generalization performance of random features models for generic supervised learning problems with Gaussian data. Our approach, built with tools from the statistical mechanics of disordered systems, maps the random features model to an equivalent polynomial model, and allows us to plot average generalization curves as functions of the two main control parameters of the problem: the number of random features $N$ and the size $P$ of the training set, both assumed to scale as powers in the input dimension $D$. Our results extend the case of proportional scaling between $N$, $P$ and $D$. They are in accordance with rigorous bounds known for certain particular learning tasks and are in quantitative agreement with numerical experime
    
[^5]: 没有解释的统计学：对可解释机器学习的冷静观察

    Statistics without Interpretation: A Sober Look at Explainable Machine Learning

    [https://arxiv.org/abs/2402.02870](https://arxiv.org/abs/2402.02870)

    解释算法往往数学上复杂且难以解释，这导致解释错误。为了向前推进，解释算法需要明确其输出的解释方式，并澄清可以和不能回答的问题。这一论点基于统计学和解释之间的区别，以及可解释机器学习和应用统计学之间的相似性。

    

    在关于解释算法的快速发展的文献中，这些算法往往不清楚所用于何处及其使用方式。我们认为这是因为解释算法往往在数学上复杂且难以解释。然而，没有清晰解释的复杂统计方法很可能导致解释的错误，这一事实在文献中越来越明显。为了向前推进，关于解释算法的论文应明确解释算法的输出如何解释。他们还应澄清在给出解释的情况下可以回答哪些关于函数的问题，以及哪些问题无法回答。我们的论点基于统计学和它们的解释之间的区别。它还依赖于可解释机器学习和应用统计学之间的相似之处。

    In the rapidly growing literature on explanation algorithms, it often remains unclear what precisely these algorithms are for and how they should be used. We argue that this is because explanation algorithms are often mathematically complex but don't admit a clear interpretation. Unfortunately, complex statistical methods that don't have a clear interpretation are bound to lead to errors in interpretation, a fact that has become increasingly apparent in the literature. In order to move forward, papers on explanation algorithms should make clear how precisely the output of the algorithms should be interpreted. They should also clarify what questions about the function can and cannot be answered given the explanations. Our argument is based on the distinction between statistics and their interpretation. It also relies on parallels between explainable machine learning and applied statistics.
    
[^6]: 使用SHAP和LIME进行可证明稳定的特征排名

    Provably Stable Feature Rankings with SHAP and LIME. (arXiv:2401.15800v1 [stat.ML])

    [http://arxiv.org/abs/2401.15800](http://arxiv.org/abs/2401.15800)

    这项研究提出了一种通过利用多重假设检验的思想，来设计可靠地排名机器学习模型中最重要特征的特征归因方法，旨在解决SHAP和LIME等常用方法由于随机采样导致的高度不稳定性问题。实验证明了该方法的有效性和计算效率。

    

    特征归因是了解机器学习模型预测的普遍工具。然而，用于评分输入变量的常用方法，如SHAP和LIME，由于随机采样而具有高度不稳定性。借鉴多重假设检验的思想，我们设计了能够以高概率正确排名最重要特征的归因方法。我们的算法RankSHAP保证$K$个最高Shapley值具有超过$1-\alpha$的正确排序概率。实证结果证明了其有效性和令人印象深刻的计算效率。我们还在之前的工作基础上为LIME提供了类似的结果，确保以正确顺序选择最重要的特征。

    Feature attributions are ubiquitous tools for understanding the predictions of machine learning models. However, popular methods for scoring input variables such as SHAP and LIME suffer from high instability due to random sampling. Leveraging ideas from multiple hypothesis testing, we devise attribution methods that correctly rank the most important features with high probability. Our algorithm RankSHAP guarantees that the $K$ highest Shapley values have the proper ordering with probability exceeding $1-\alpha$. Empirical results demonstrate its validity and impressive computational efficiency. We also build on previous work to yield similar results for LIME, ensuring the most important features are selected in the right order.
    
[^7]: 私有零阶优化的大型语言模型的私有微调

    Private Fine-tuning of Large Language Models with Zeroth-order Optimization. (arXiv:2401.04343v1 [cs.LG])

    [http://arxiv.org/abs/2401.04343](http://arxiv.org/abs/2401.04343)

    引入了DP-ZO，一种通过私有化零阶优化来保护大型语言模型训练数据隐私的方法。

    

    在私有数据集上对大型预训练模型进行微调可能会存在违反隐私的风险。差分隐私是一种通过强制算法稳定性来减轻隐私风险的框架。DP-SGD可以以保护隐私的方式训练具有私有数据的模型，但会带来性能损失和重大工程挑战。我们引入了DP-ZO，一种通过私有化零阶优化来保护训练数据隐私的大型语言模型微调方法。我们的方法设计的一个关键见解是，我们使用的零阶算法SPSA中的梯度方向始终是随机的，而仅依赖于私有数据的信息是步长，即一个标量。因此，我们只需要对标量步长进行隐私处理，这是存储效率高的方法。DP-ZO可以使用拉普拉斯噪声或高斯噪声来实现，在不同任务之间提供了隐私和效用之间的强大权衡。

    Fine-tuning large pretrained models on private datasets may run the risk of violating privacy. Differential privacy is a framework for mitigating privacy risks by enforcing algorithmic stability. DP-SGD enables training models with private data in a privacy-preserving manner, but raises new obstacles in the form of performance loss and significant engineering challenges. We introduce DP-ZO, a new method for fine-tuning large language models that preserves the privacy of training data by privatizing zeroth-order optimization. A key insight into the design of our method is that the direction of the gradient in SPSA, the zeroth-order algorithm we use, is always random and the only information that depends on private data is the step size, i.e., a scalar. Therefore, we only need to privatize the scalar step size, which is memory-efficient. DP-ZO, which can be instantiated with either Laplace or Gaussian noise, provides a strong privacy-utility trade-off across different tasks, and model si
    
[^8]: 关于最佳n对齐策略的理论保证

    Theoretical guarantees on the best-of-n alignment policy. (arXiv:2401.01879v1 [cs.LG])

    [http://arxiv.org/abs/2401.01879](http://arxiv.org/abs/2401.01879)

    该论文研究了对齐生成模型的最佳n对齐策略，并证明了之前文献中的某个分析表达式是错误的。研究者们提出了一个新的KL散度估计方法，并通过实验证明其有效性。

    

    一个简单有效的生成模型对齐方法是最佳n对齐策略，该策略从一个基本策略中抽取n个样本，并根据奖励函数对它们进行排序，选择排名最高的样本。文献中常用的分析表达式声称最佳n对齐策略与基本策略之间的KL散度等于$\log (n) (n-1)/n$。我们证明了该论断的不正确性，并展示了它只是实际KL散度的一个上界。我们还研究了在不同情况下该上界的紧致性。最后，我们提出了一种新的KL散度估计方法，并通过几个例子的实验证明它能提供一个紧致的近似。

    A simple and effective method for the alignment of generative models is the best-of-$n$ policy, where $n$ samples are drawn from a base policy, and ranked based on a reward function, and the highest ranking one is selected. A commonly used analytical expression in the literature claims that the KL divergence between the best-of-$n$ policy and the base policy is equal to $\log (n) (n-1)/n.$ We disprove the validity of this claim, and show that it is an upper bound on the actual KL divergence. We also explore the tightness of this upper bound in different regimes. Finally, we propose a new estimator for the KL divergence and empirically show that it provides a tight approximation through a few examples.
    
[^9]: 最大权重熵

    Maximum Weight Entropy. (arXiv:2309.15704v1 [cs.LG])

    [http://arxiv.org/abs/2309.15704](http://arxiv.org/abs/2309.15704)

    本文提出了在深度学习中使用最大熵原理的最大权重熵方法，通过最大化权重多样性来解决标准方法在超出分布情况下预测多样性缺乏的问题。

    

    本文研究了在深度学习中使用贝叶斯和集成方法进行不确定性量化和超出分布检测的问题。当在超出分布的情况下使用标准方法时，我们观察到预测多样性的缺乏。针对这个问题，本文提出了一个实用的解决方案，认为标准方法在权重空间中采样的“过度约束”导致了权重多样性的缺乏。本文建议采用最大熵原理来解决这个问题，通过最大化权重多样性，描述最大熵权重分布来表示认知不确定性，从而产生与训练观察一致的神经网络。

    This paper deals with uncertainty quantification and out-of-distribution detection in deep learning using Bayesian and ensemble methods. It proposes a practical solution to the lack of prediction diversity observed recently for standard approaches when used out-of-distribution (Ovadia et al., 2019; Liu et al., 2021). Considering that this issue is mainly related to a lack of weight diversity, we claim that standard methods sample in "over-restricted" regions of the weight space due to the use of "over-regularization" processes, such as weight decay and zero-mean centered Gaussian priors. We propose to solve the problem by adopting the maximum entropy principle for the weight distribution, with the underlying idea to maximize the weight diversity. Under this paradigm, the epistemic uncertainty is described by the weight distribution of maximal entropy that produces neural networks "consistent" with the training observations. Considering stochastic neural networks, a practical optimizati
    
[^10]: 随机傅里叶特征在去量化量子机器学习中的潜力与局限性

    Potential and limitations of random Fourier features for dequantizing quantum machine learning. (arXiv:2309.11647v1 [quant-ph])

    [http://arxiv.org/abs/2309.11647](http://arxiv.org/abs/2309.11647)

    本文研究了随机傅里叶特征在去量化量子机器学习中的潜力与局限性，并在回归问题上确立了其高效去量化的必要和充分条件，并提出了PQC架构设计建议和识别了潜在量子优势的必要结构。

    

    量子机器学习是近期量子设备最广泛探索的应用之一。目前主要关注的是变分量子机器学习，其中参数化量子电路被用作学习模型。这些参数化量子电路模型具有丰富的结构，因此可能通过随机傅里叶特征进行高效的去量化。本文在回归问题上确立了随机傅里叶特征在变分量子机器学习中提供高效去量化的必要和充分条件。利用这些结果，我们提出了具体的参数化量子电路架构设计建议，以及识别了在回归问题中取得潜在量子优势的必要结构。

    Quantum machine learning is arguably one of the most explored applications of near-term quantum devices. Much focus has been put on notions of variational quantum machine learning where parameterized quantum circuits (PQCs) are used as learning models. These PQC models have a rich structure which suggests that they might be amenable to efficient dequantization via random Fourier features (RFF). In this work, we establish necessary and sufficient conditions under which RFF does indeed provide an efficient dequantization of variational quantum machine learning for regression. We build on these insights to make concrete suggestions for PQC architecture design, and to identify structures which are necessary for a regression problem to admit a potential quantum advantage via PQC based optimization.
    
[^11]: 用一种时间对称的深度学习方法提升细胞跟踪能力

    Enhancing Cell Tracking with a Time-Symmetric Deep Learning Approach. (arXiv:2308.03887v1 [eess.IV])

    [http://arxiv.org/abs/2308.03887](http://arxiv.org/abs/2308.03887)

    本论文提出了一种使用时间对称的深度学习方法来提升细胞跟踪的准确性。该方法不依赖于连续帧跟踪，而是基于细胞的时空邻域进行跟踪，具有学习细胞运动模式的能力，并能处理具有严重伪影的大量视频帧。

    

    使用视频显微镜记录准确跟踪活细胞仍然是目前流行的最先进图像处理技术方法的一个具有挑战性的任务。近年来，已有几个现有和新的应用尝试将基于深度学习的框架整合到该任务中，但大部分仍然严重依赖于嵌入其架构或其他前提条件中的连续帧跟踪，从而限制了广义学习。为了解决这个问题，我们旨在开发一种新的基于深度学习的跟踪方法，该方法仅依赖于细胞可以根据其时空邻域进行跟踪的假设，而非仅限于连续帧。所提出的方法的额外优点是细胞的运动模式可以完全由预测器在没有任何先验假设的情况下学习，并且具有处理大量具有严重伪影的视频帧的潜力。

    The accurate tracking of live cells using video microscopy recordings remains a challenging task for popular state-of-the-art image processing based object tracking methods. In recent years, several existing and new applications have attempted to integrate deep-learning based frameworks for this task, but most of them still heavily rely on consecutive frame based tracking embedded in their architecture or other premises that hinder generalized learning. To address this issue, we aimed to develop a new deep-learning based tracking method that relies solely on the assumption that cells can be tracked based on their spatio-temporal neighborhood, without restricting it to consecutive frames. The proposed method has the additional benefit that the motion patterns of the cells can be learned completely by the predictor without any prior assumptions, and it has the potential to handle a large number of video frames with heavy artifacts. The efficacy of the proposed method is demonstrated thro
    
[^12]: 数据高效和高性能医学图像处理的等变球卷积神经网络

    Equivariant Spherical CNN for Data Efficient and High-Performance Medical Image Processing. (arXiv:2307.03298v1 [eess.IV])

    [http://arxiv.org/abs/2307.03298](http://arxiv.org/abs/2307.03298)

    等变球卷积神经网络能够提高医学图像处理的效率和性能，通过降低对特定训练集的依赖性，并且在去噪和重建方面表现出卓越的质量和计算效率。

    

    本研究突出了等变网络作为高效和高性能途径在断层扫描应用中的重要性。我们的研究建立在卷积神经网络（CNN）的局限性之上，CNN已经在各种医学影像系统的后处理中显示出了潜力。然而，传统CNN的效率严重依赖于一个不变和适当的训练集。为了解决这个问题，在本研究中，我们引入了一个等变网络，旨在减少CNN对特定训练集的依赖性。我们评估了等变CNN在球信号上在断层扫描医学成像问题中的有效性。我们的结果表明，球形CNN（SCNN）在去噪和重建基准问题上具有优越的质量和计算效率。此外，我们提出了一种新颖的方法，利用SCNN作为传统图像重建工具的补充，增强结果同时减少对训练集的依赖性。在所有案例中，我们观察到...

    This work highlights the significance of equivariant networks as efficient and high-performance approaches for tomography applications. Our study builds upon the limitations of Convolutional Neural Networks (CNNs), which have shown promise in post-processing various medical imaging systems. However, the efficiency of conventional CNNs heavily relies on an undiminished and proper training set. To tackle this issue, in this study, we introduce an equivariant network, aiming to reduce CNN's dependency on specific training sets. We evaluate the efficacy of equivariant CNNs on spherical signals for tomographic medical imaging problems. Our results demonstrate superior quality and computational efficiency of spherical CNNs (SCNNs) in denoising and reconstructing benchmark problems. Furthermore, we propose a novel approach to employ SCNNs as a complement to conventional image reconstruction tools, enhancing the outcomes while reducing reliance on the training set. Across all cases, we observe
    

