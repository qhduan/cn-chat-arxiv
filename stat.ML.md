# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [On the rates of convergence for learning with convolutional neural networks](https://arxiv.org/abs/2403.16459) | 该研究提出了对具有一定权重约束的CNNs的新逼近上界，以及对前馈神经网络的覆盖数做了新的分析，为基于CNNs的学习问题推导了收敛速率，并在学习平滑函数和二元分类方面取得了极小最优的结果。 |
| [^2] | [Selecting informative conformal prediction sets with false coverage rate control](https://arxiv.org/abs/2403.12295) | 提出了一种新的统一框架，用于构建信息丰富的符合预测集，同时控制所选样本的虚警覆盖率。 |
| [^3] | [Federated Transfer Learning with Differential Privacy](https://arxiv.org/abs/2403.11343) | 本文提出了具有差分隐私的联邦迁移学习框架，通过利用多个异构源数据集的信息来增强对目标数据集的学习，同时考虑隐私约束。 |
| [^4] | [Extremal graphical modeling with latent variables](https://arxiv.org/abs/2403.09604) | 提出了一种针对混合变量的极端图模型的学习方法，能够有效恢复条件图和潜变量数量。 |
| [^5] | [Minimizing the Thompson Sampling Regret-to-Sigma Ratio (TS-RSR): a provably efficient algorithm for batch Bayesian Optimization](https://arxiv.org/abs/2403.04764) | 该论文提出了一种用于批量贝叶斯优化的高效算法，通过最小化Thompson抽样近似的遗憾与不确定性比率，成功协调每个批次的动作选择，同时实现高概率的理论保证，并在非凸测试函数上表现出色. |
| [^6] | [Misspecification uncertainties in near-deterministic regression](https://arxiv.org/abs/2402.01810) | 该论文研究了近确定性回归中错误规范化的不确定性问题，并提出了一种组合模型，以准确预测和控制参数不确定性。 |
| [^7] | [Faithful and Robust Local Interpretability for Textual Predictions.](http://arxiv.org/abs/2311.01605) | 提出了一种名为FRED的新颖方法，用于解释文本预测。FRED可以识别文档中的关键词，并且通过与最先进的方法进行的实证评估证明了其在提供对文本模型的深入见解方面的有效性。 |
| [^8] | [ExIFFI and EIF+: Interpretability and Enhanced Generalizability to Extend the Extended Isolation Forest.](http://arxiv.org/abs/2310.05468) | 本研究介绍了EIF+和ExIFFI两种改进了扩展孤立森林的方法，分别增强了模型的推广能力和解释性能，实验结果表明其在异常检测任务中具有优势。 |
| [^9] | [On the expressivity of embedding quantum kernels.](http://arxiv.org/abs/2309.14419) | 量子核方法是量子和经典机器学习之间最自然的联系之一。本文探讨了嵌入式量子核的表达能力，并得出结论：通过引入计算普适性，任何核函数都可以表示为量子特征映射和嵌入式量子核。 |
| [^10] | [Sharpness-Aware Minimization and the Edge of Stability.](http://arxiv.org/abs/2309.12488) | 本研究通过类似的计算方法，为锐度感知最小化(SAM)，一种改进泛化性能的梯度下降变种，确定了一个稳定性边界，该边界取决于梯度的范数。 |
| [^11] | [Stochastic Controlled Averaging for Federated Learning with Communication Compression.](http://arxiv.org/abs/2308.08165) | 本文提出了两种压缩联邦学习算法(SCALLION和SCAFCOM)，通过重新审视经典的随机控制平均法并提出了等价但更高效/简化的形式，减少了上行通信成本。 |
| [^12] | [Linear convergence of Nesterov-1983 with the strong convexity.](http://arxiv.org/abs/2306.09694) | 本文使用高分辨率微分方程框架回答了Nesterov-1983和FISTA是否在强凸函数上线性收敛的问题，并指出线性收敛性不依赖于强凸性条件。 |

# 详细

[^1]: 关于使用卷积神经网络进行学习收敛速率的研究

    On the rates of convergence for learning with convolutional neural networks

    [https://arxiv.org/abs/2403.16459](https://arxiv.org/abs/2403.16459)

    该研究提出了对具有一定权重约束的CNNs的新逼近上界，以及对前馈神经网络的覆盖数做了新的分析，为基于CNNs的学习问题推导了收敛速率，并在学习平滑函数和二元分类方面取得了极小最优的结果。

    

    我们研究了卷积神经网络（CNNs）的逼近和学习能力。第一个结果证明了在权重上有一定约束条件下CNNs的新逼近上界。第二个结果给出了对前馈神经网络的覆盖数的新分析，其中CNNs是其特例。该分析详细考虑了权重的大小，在某些情况下给出了比现有文献更好的上界。利用这两个结果，我们能够推导基于CNNs的估计器在许多学习问题中的收敛速率。特别地，我们在非参数回归设置中为基于CNNs的最小二乘学习平滑函数建立了极小最优的收敛速率。对于二元分类，我们推导了具有铰链损失和逻辑损失的CNN分类器的收敛速度。同时还表明所得到的速率在几种情况下是极小最优的。

    arXiv:2403.16459v1 Announce Type: new  Abstract: We study the approximation and learning capacities of convolutional neural networks (CNNs). Our first result proves a new approximation bound for CNNs with certain constraint on the weights. Our second result gives a new analysis on the covering number of feed-forward neural networks, which include CNNs as special cases. The analysis carefully takes into account the size of the weights and hence gives better bounds than existing literature in some situations. Using these two results, we are able to derive rates of convergence for estimators based on CNNs in many learning problems. In particular, we establish minimax optimal convergence rates of the least squares based on CNNs for learning smooth functions in the nonparametric regression setting. For binary classification, we derive convergence rates for CNN classifiers with hinge loss and logistic loss. It is also shown that the obtained rates are minimax optimal in several settings.
    
[^2]: 通过控制虚警覆盖率选择信息量丰富的符合预测集

    Selecting informative conformal prediction sets with false coverage rate control

    [https://arxiv.org/abs/2403.12295](https://arxiv.org/abs/2403.12295)

    提出了一种新的统一框架，用于构建信息丰富的符合预测集，同时控制所选样本的虚警覆盖率。

    

    在监督学习中，包括回归和分类，符合方法为任何机器学习预测器提供预测结果/标签的预测集合，具有有限样本覆盖率。在这里我们考虑了这样一种情况，即这种预测集合是经过选择过程得到的。该选择过程要求选择的预测集在某种明确定义的意义上是“信息量丰富的”。我们考虑了分类和回归设置，在这些设置中，分析人员可能只考虑具有预测标签集或预测区间足够小、不包括空值或遵守其他适当的“单调”约束的样本为具有信息量丰富的。虽然这涵盖了各种应用中可能感兴趣的许多设置，我们开发了一个统一的框架，用来构建这样的信息量丰富的符合预测集，同时控制所选样本上的虚警覆盖率（FCR）。

    arXiv:2403.12295v1 Announce Type: cross  Abstract: In supervised learning, including regression and classification, conformal methods provide prediction sets for the outcome/label with finite sample coverage for any machine learning predictors. We consider here the case where such prediction sets come after a selection process. The selection process requires that the selected prediction sets be `informative' in a well defined sense. We consider both the classification and regression settings where the analyst may consider as informative only the sample with prediction label sets or prediction intervals small enough, excluding null values, or obeying other appropriate `monotone' constraints. While this covers many settings of possible interest in various applications, we develop a unified framework for building such informative conformal prediction sets while controlling the false coverage rate (FCR) on the selected sample. While conformal prediction sets after selection have been the f
    
[^3]: 具有差分隐私的联邦迁移学习

    Federated Transfer Learning with Differential Privacy

    [https://arxiv.org/abs/2403.11343](https://arxiv.org/abs/2403.11343)

    本文提出了具有差分隐私的联邦迁移学习框架，通过利用多个异构源数据集的信息来增强对目标数据集的学习，同时考虑隐私约束。

    

    联邦学习越来越受到欢迎，数据异构性和隐私性是两个突出的挑战。在本文中，我们在联邦迁移学习框架内解决了这两个问题，旨在通过利用来自多个异构源数据集的信息来增强对目标数据集的学习，同时遵守隐私约束。我们严格制定了\textit{联邦差分隐私}的概念，为每个数据集提供隐私保证，而无需假设有一个受信任的中央服务器。在这个隐私约束下，我们研究了三个经典的统计问题，即单变量均值估计、低维线性回归和高维线性回归。通过研究极小值率并确定这些问题的隐私成本，我们展示了联邦差分隐私是已建立的局部和中央模型之间的一种中间隐私模型。

    arXiv:2403.11343v1 Announce Type: new  Abstract: Federated learning is gaining increasing popularity, with data heterogeneity and privacy being two prominent challenges. In this paper, we address both issues within a federated transfer learning framework, aiming to enhance learning on a target data set by leveraging information from multiple heterogeneous source data sets while adhering to privacy constraints. We rigorously formulate the notion of \textit{federated differential privacy}, which offers privacy guarantees for each data set without assuming a trusted central server. Under this privacy constraint, we study three classical statistical problems, namely univariate mean estimation, low-dimensional linear regression, and high-dimensional linear regression. By investigating the minimax rates and identifying the costs of privacy for these problems, we show that federated differential privacy is an intermediate privacy model between the well-established local and central models of 
    
[^4]: 混合变量的极端图模型

    Extremal graphical modeling with latent variables

    [https://arxiv.org/abs/2403.09604](https://arxiv.org/abs/2403.09604)

    提出了一种针对混合变量的极端图模型的学习方法，能够有效恢复条件图和潜变量数量。

    

    极端图模型编码多变量极端条件独立结构，并为量化罕见事件风险提供强大工具。我们提出了面向潜变量的可延伸图模型的可行凸规划方法，将 H\"usler-Reiss 精度矩阵分解为编码观察变量之间的图结构的稀疏部分和编码少量潜变量对观察变量的影响的低秩部分。我们提供了\texttt{eglatent}的有限样本保证，并展示它能一致地恢复条件图以及潜变量的数量。

    arXiv:2403.09604v1 Announce Type: cross  Abstract: Extremal graphical models encode the conditional independence structure of multivariate extremes and provide a powerful tool for quantifying the risk of rare events. Prior work on learning these graphs from data has focused on the setting where all relevant variables are observed. For the popular class of H\"usler-Reiss models, we propose the \texttt{eglatent} method, a tractable convex program for learning extremal graphical models in the presence of latent variables. Our approach decomposes the H\"usler-Reiss precision matrix into a sparse component encoding the graphical structure among the observed variables after conditioning on the latent variables, and a low-rank component encoding the effect of a few latent variables on the observed variables. We provide finite-sample guarantees of \texttt{eglatent} and show that it consistently recovers the conditional graph as well as the number of latent variables. We highlight the improved 
    
[^5]: 将Thompson抽样遗憾与Sigma比率（TS-RSR）最小化：一种用于批量贝叶斯优化的经过证明的高效算法

    Minimizing the Thompson Sampling Regret-to-Sigma Ratio (TS-RSR): a provably efficient algorithm for batch Bayesian Optimization

    [https://arxiv.org/abs/2403.04764](https://arxiv.org/abs/2403.04764)

    该论文提出了一种用于批量贝叶斯优化的高效算法，通过最小化Thompson抽样近似的遗憾与不确定性比率，成功协调每个批次的动作选择，同时实现高概率的理论保证，并在非凸测试函数上表现出色.

    

    本文提出了一个新的方法，用于批量贝叶斯优化（BO），其中抽样通过最小化Thompson抽样方法的遗憾与不确定性比率来进行。我们的目标是能够协调每个批次中选择的动作，以最小化点之间的冗余，同时关注具有高预测均值或高不确定性的点。我们对算法的遗憾提供了高概率的理论保证。最后，从数字上看，我们证明了我们的方法在一系列非凸测试函数上达到了最先进的性能，在平均值上比几个竞争对手的基准批量BO算法表现提高了一个数量级。

    arXiv:2403.04764v1 Announce Type: new  Abstract: This paper presents a new approach for batch Bayesian Optimization (BO), where the sampling takes place by minimizing a Thompson Sampling approximation of a regret to uncertainty ratio. Our objective is able to coordinate the actions chosen in each batch in a way that minimizes redundancy between points whilst focusing on points with high predictive means or high uncertainty. We provide high-probability theoretical guarantees on the regret of our algorithm. Finally, numerically, we demonstrate that our method attains state-of-the-art performance on a range of nonconvex test functions, where it outperforms several competitive benchmark batch BO algorithms by an order of magnitude on average.
    
[^6]: 近确定性回归中的错误规范化不确定性

    Misspecification uncertainties in near-deterministic regression

    [https://arxiv.org/abs/2402.01810](https://arxiv.org/abs/2402.01810)

    该论文研究了近确定性回归中错误规范化的不确定性问题，并提出了一种组合模型，以准确预测和控制参数不确定性。

    

    期望损失是模型泛化误差的上界，可用于学习的鲁棒PAC-Bayes边界。然而，损失最小化被认为忽略了错误规范化，即模型不能完全复制观测结果。这导致大数据或欠参数化极限下对参数不确定性的显著低估。我们分析近确定性、错误规范化和欠参数化替代模型的泛化误差，这是科学和工程中广泛相关的一个领域。我们证明后验分布必须覆盖每个训练点，以避免发散的泛化误差，并导出一个符合这个约束的组合模型。对于线性模型，这种高效的方法产生的额外开销最小。这种高效方法在模型问题上进行了演示，然后应用于原子尺度机器学习中的高维数据集。

    The expected loss is an upper bound to the model generalization error which admits robust PAC-Bayes bounds for learning. However, loss minimization is known to ignore misspecification, where models cannot exactly reproduce observations. This leads to significant underestimates of parameter uncertainties in the large data, or underparameterized, limit. We analyze the generalization error of near-deterministic, misspecified and underparametrized surrogate models, a regime of broad relevance in science and engineering. We show posterior distributions must cover every training point to avoid a divergent generalization error and derive an ensemble {ansatz} that respects this constraint, which for linear models incurs minimal overhead. The efficient approach is demonstrated on model problems before application to high dimensional datasets in atomistic machine learning. Parameter uncertainties from misspecification survive in the underparametrized limit, giving accurate prediction and boundin
    
[^7]: 对于文本预测的忠实和稳健的本地可解释性

    Faithful and Robust Local Interpretability for Textual Predictions. (arXiv:2311.01605v1 [cs.CL])

    [http://arxiv.org/abs/2311.01605](http://arxiv.org/abs/2311.01605)

    提出了一种名为FRED的新颖方法，用于解释文本预测。FRED可以识别文档中的关键词，并且通过与最先进的方法进行的实证评估证明了其在提供对文本模型的深入见解方面的有效性。

    

    可解释性对于机器学习模型在关键领域中得到信任和部署是至关重要的。然而，现有的用于解释文本模型的方法通常复杂，并且缺乏坚实的数学基础，它们的性能也不能保证。在本文中，我们提出了一种新颖的方法FRED（Faithful and Robust Explainer for textual Documents），用于解释文本预测。FRED可以识别文档中的关键词，当这些词被移除时对预测结果产生重大影响。我们通过正式的定义和对可解释分类器的理论分析，确立了FRED的可靠性。此外，我们还通过与最先进的方法进行的实证评估，证明了FRED在提供对文本模型的深入见解方面的有效性。

    Interpretability is essential for machine learning models to be trusted and deployed in critical domains. However, existing methods for interpreting text models are often complex, lack solid mathematical foundations, and their performance is not guaranteed. In this paper, we propose FRED (Faithful and Robust Explainer for textual Documents), a novel method for interpreting predictions over text. FRED identifies key words in a document that significantly impact the prediction when removed. We establish the reliability of FRED through formal definitions and theoretical analyses on interpretable classifiers. Additionally, our empirical evaluation against state-of-the-art methods demonstrates the effectiveness of FRED in providing insights into text models.
    
[^8]: ExIFFI和EIF+：解释性和增强的推广能力以扩展扩展孤立森林

    ExIFFI and EIF+: Interpretability and Enhanced Generalizability to Extend the Extended Isolation Forest. (arXiv:2310.05468v1 [stat.ML])

    [http://arxiv.org/abs/2310.05468](http://arxiv.org/abs/2310.05468)

    本研究介绍了EIF+和ExIFFI两种改进了扩展孤立森林的方法，分别增强了模型的推广能力和解释性能，实验结果表明其在异常检测任务中具有优势。

    

    异常检测是一种重要的无监督机器学习任务，涉及在复杂数据集和系统中识别异常行为。虽然机器学习算法和决策支持系统（DSS）提供了有效的解决方案，但仅仅定位异常往往在实际应用中不足。这些系统的用户通常需要了解预测背后的原因，以便进行根本原因分析并增强对模型的信任。然而，由于异常检测的无监督性质，创建可解释的工具是具有挑战性的。本文介绍了EIF+，这是扩展孤立森林（EIF）的增强变体，旨在增强泛化能力。此外，我们提出了ExIFFI，一种将扩展孤立森林与解释性功能（特征排名）相结合的新方法。实验结果提供了以孤立基于方法进行异常检测的综合比较分析。

    Anomaly detection, an essential unsupervised machine learning task, involves identifying unusual behaviors within complex datasets and systems. While Machine Learning algorithms and decision support systems (DSSs) offer effective solutions for this task, simply pinpointing anomalies often falls short in real-world applications. Users of these systems often require insight into the underlying reasons behind predictions to facilitate Root Cause Analysis and foster trust in the model. However, due to the unsupervised nature of anomaly detection, creating interpretable tools is challenging. This work introduces EIF+, an enhanced variant of Extended Isolation Forest (EIF), designed to enhance generalization capabilities. Additionally, we present ExIFFI, a novel approach that equips Extended Isolation Forest with interpretability features, specifically feature rankings. Experimental results provide a comprehensive comparative analysis of Isolation-based approaches for Anomaly Detection, incl
    
[^9]: 关于嵌入式量子核的表达能力

    On the expressivity of embedding quantum kernels. (arXiv:2309.14419v1 [quant-ph])

    [http://arxiv.org/abs/2309.14419](http://arxiv.org/abs/2309.14419)

    量子核方法是量子和经典机器学习之间最自然的联系之一。本文探讨了嵌入式量子核的表达能力，并得出结论：通过引入计算普适性，任何核函数都可以表示为量子特征映射和嵌入式量子核。

    

    在核方法的背景下，量子核与经典机器学习之间建立了最自然的联系。核方法依赖于内积特征向量，这些特征向量存在于大型特征空间中。量子核通常通过显式构造量子特征态并计算它们的内积来评估，这里称为嵌入式量子核。由于经典核通常在不使用特征向量的情况下进行评估，我们想知道嵌入式量子核的表达能力如何。在这项工作中，我们提出了一个基本问题：是否所有的量子核都可以表达为量子特征态的内积？我们的第一个结果是肯定的：通过调用计算普适性，我们发现对于任何核函数，总是存在对应的量子特征映射和嵌入式量子核。然而，问题更关注的是有效的构造方式。在第二部分中

    One of the most natural connections between quantum and classical machine learning has been established in the context of kernel methods. Kernel methods rely on kernels, which are inner products of feature vectors living in large feature spaces. Quantum kernels are typically evaluated by explicitly constructing quantum feature states and then taking their inner product, here called embedding quantum kernels. Since classical kernels are usually evaluated without using the feature vectors explicitly, we wonder how expressive embedding quantum kernels are. In this work, we raise the fundamental question: can all quantum kernels be expressed as the inner product of quantum feature states? Our first result is positive: Invoking computational universality, we find that for any kernel function there always exists a corresponding quantum feature map and an embedding quantum kernel. The more operational reading of the question is concerned with efficient constructions, however. In a second part
    
[^10]: 锐度感知最小化和稳定性边界。

    Sharpness-Aware Minimization and the Edge of Stability. (arXiv:2309.12488v1 [cs.LG])

    [http://arxiv.org/abs/2309.12488](http://arxiv.org/abs/2309.12488)

    本研究通过类似的计算方法，为锐度感知最小化(SAM)，一种改进泛化性能的梯度下降变种，确定了一个稳定性边界，该边界取决于梯度的范数。

    

    最近的实验表明，当使用梯度下降(GD)训练神经网络时，损失函数的Hessian矩阵的操作符范数会增长，直到接近$2/\eta$，之后会在该值周围波动。根据对损失函数的局部二次逼近，$2/\eta$被称为“稳定性边界”。我们使用类似的计算方法，为锐度感知最小化(SAM)确定了一个“稳定性边界”，SAM是一种改进泛化性能的GD变种。与GD不同，SAM的稳定性边界取决于梯度的范数。通过三个深度学习任务的实证，我们观察到SAM在这个分析中确定的稳定性边界上运行。

    Recent experiments have shown that, often, when training a neural network with gradient descent (GD) with a step size $\eta$, the operator norm of the Hessian of the loss grows until it approximately reaches $2/\eta$, after which it fluctuates around this value.  The quantity $2/\eta$ has been called the "edge of stability" based on consideration of a local quadratic approximation of the loss. We perform a similar calculation to arrive at an "edge of stability" for Sharpness-Aware Minimization (SAM), a variant of GD which has been shown to improve its generalization. Unlike the case for GD, the resulting SAM-edge depends on the norm of the gradient. Using three deep learning training tasks, we see empirically that SAM operates on the edge of stability identified by this analysis.
    
[^11]: 带有通信压缩的随机控制平均法在联邦学习中的应用

    Stochastic Controlled Averaging for Federated Learning with Communication Compression. (arXiv:2308.08165v1 [math.OC] CROSS LISTED)

    [http://arxiv.org/abs/2308.08165](http://arxiv.org/abs/2308.08165)

    本文提出了两种压缩联邦学习算法(SCALLION和SCAFCOM)，通过重新审视经典的随机控制平均法并提出了等价但更高效/简化的形式，减少了上行通信成本。

    

    通信压缩是一种旨在减少通过无线传输的信息量的技术，在联邦学习中引起了极大的关注，因为它有潜力减轻通信开销。然而，通信压缩在联邦学习中带来了新的挑战，包括压缩引起的信息失真以及联邦学习的特性，如部分参与和数据异构性。尽管近年来有所发展，压缩联邦学习方法的性能尚未充分利用。现有方法要么不能适应任意的数据异构性或部分参与，要么要求对压缩有严格的条件。在本文中，我们重新审视了具有开销减半的上行通信成本的经典随机控制平均法，并提出了两种压缩联邦学习算法，SCALLION和SCAFCOM。

    Communication compression, a technique aiming to reduce the information volume to be transmitted over the air, has gained great interests in Federated Learning (FL) for the potential of alleviating its communication overhead. However, communication compression brings forth new challenges in FL due to the interplay of compression-incurred information distortion and inherent characteristics of FL such as partial participation and data heterogeneity. Despite the recent development, the performance of compressed FL approaches has not been fully exploited. The existing approaches either cannot accommodate arbitrary data heterogeneity or partial participation, or require stringent conditions on compression.  In this paper, we revisit the seminal stochastic controlled averaging method by proposing an equivalent but more efficient/simplified formulation with halved uplink communication costs. Building upon this implementation, we propose two compressed FL algorithms, SCALLION and SCAFCOM, to s
    
[^12]: 具有强凸性的 Nesterov-1983 的线性收敛性

    Linear convergence of Nesterov-1983 with the strong convexity. (arXiv:2306.09694v1 [math.OC])

    [http://arxiv.org/abs/2306.09694](http://arxiv.org/abs/2306.09694)

    本文使用高分辨率微分方程框架回答了Nesterov-1983和FISTA是否在强凸函数上线性收敛的问题，并指出线性收敛性不依赖于强凸性条件。

    

    对于现代基于梯度的优化，Nesterov 的加速梯度下降法是一个开创性里程碑，该方法在[Nesterov，1983]中提出，简称为Nesterov-1983。此后，重要的进展之一是它的近端推广，名为快速迭代收缩阈值算法（FISTA），广泛应用于图像科学和工程。然而，目前仍未知道Nesterov-1983和FISTA是否在强凸函数上线性收敛，而这已被列为综合评审[Chambolle和Pock，2016，附录B]中的未解决问题。本文通过使用高分辨率微分方程框架来回答这个问题。与先前采用的相空间表示一起，构造Lyapunov函数的关键区别在于动能的系数随迭代而变化。此外，我们指出，上述两种算法的线性收敛性没有依赖于强凸函数的条件。

    For modern gradient-based optimization, a developmental landmark is Nesterov's accelerated gradient descent method, which is proposed in [Nesterov, 1983], so shorten as Nesterov-1983. Afterward, one of the important progresses is its proximal generalization, named the fast iterative shrinkage-thresholding algorithm (FISTA), which is widely used in image science and engineering. However, it is unknown whether both Nesterov-1983 and FISTA converge linearly on the strongly convex function, which has been listed as the open problem in the comprehensive review [Chambolle and Pock, 2016, Appendix B]. In this paper, we answer this question by the use of the high-resolution differential equation framework. Along with the phase-space representation previously adopted, the key difference here in constructing the Lyapunov function is that the coefficient of the kinetic energy varies with the iteration. Furthermore, we point out that the linear convergence of both the two algorithms above has no d
    

