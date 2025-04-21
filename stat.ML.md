# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Jailbreaking Leading Safety-Aligned LLMs with Simple Adaptive Attacks](https://arxiv.org/abs/2404.02151) | 展示了对齐的LLM对简单自适应越狱攻击不具有鲁棒性，并成功实现了在多个模型上几乎100%的攻击成功率，同时还介绍了对于不公开logprobs的模型如何进行越狱以及如何在受污染的模型中查找木马字符串的方法。 |
| [^2] | [Confidence on the Focal: Conformal Prediction with Selection-Conditional Coverage](https://arxiv.org/abs/2403.03868) | 该论文提出了一种构建具有有限样本精确覆盖的预测集的通用框架，可以解决在数据驱动情境中由于选择偏差导致的边缘有效预测区间误导问题。 |
| [^3] | [Deep Huber quantile regression networks.](http://arxiv.org/abs/2306.10306) | DHQRN可以预测更一般的Huber分位数，并且在预测分布的尾部提供更好的预测。 |
| [^4] | [Approximation-Generalization Trade-offs under (Approximate) Group Equivariance.](http://arxiv.org/abs/2305.17592) | 本论文详细研究了通过对称性明确地引入任务特定的归纳偏差所导致的逼近-泛化权衡，并且证明了这种模型在捕获任务特定对称性的同时会改进泛化。这一结果对于提高机器学习领域的性能具有非常大的帮助。 |
| [^5] | [On the Shift Invariance of Max Pooling Feature Maps in Convolutional Neural Networks.](http://arxiv.org/abs/2209.11740) | 本文研究了卷积神经网络中最大池化特征图的位移不变性问题，并提出了一种近似复数模的条件，实现了位移稳定性。实验证实了理论的有效性。 |
| [^6] | [Nuclear Norm Regularized Estimation of Panel Regression Models.](http://arxiv.org/abs/1810.10987) | 本文提出两种最小化凸目标函数的新估计方法，其中核范数罚项有助于解决低秩回归器下的交互固定效应模型的潜在识别问题，并且具有很重要的计算优势。 |

# 详细

[^1]: 用简单自适应攻击越狱功能对齐的LLM

    Jailbreaking Leading Safety-Aligned LLMs with Simple Adaptive Attacks

    [https://arxiv.org/abs/2404.02151](https://arxiv.org/abs/2404.02151)

    展示了对齐的LLM对简单自适应越狱攻击不具有鲁棒性，并成功实现了在多个模型上几乎100%的攻击成功率，同时还介绍了对于不公开logprobs的模型如何进行越狱以及如何在受污染的模型中查找木马字符串的方法。

    

    我们展示了即使是最新的安全对齐的LLM也不具有抵抗简单自适应越狱攻击的稳健性。首先，我们展示了如何成功利用对logprobs的访问进行越狱：我们最初设计了一个对抗性提示模板（有时会适应目标LLM），然后我们在后缀上应用随机搜索以最大化目标logprob（例如token“Sure”），可能会进行多次重启。通过这种方式，我们实现了对GPT-3.5/4、Llama-2-Chat-7B/13B/70B、Gemma-7B和针对GCG攻击进行对抗训练的HarmBench上的R2D2等几乎100%的攻击成功率--根据GPT-4的评判。我们还展示了如何通过转移或预填充攻击以100%的成功率对所有不暴露logprobs的Claude模型进行越狱。此外，我们展示了如何在受污染的模型中使用对一组受限制的token执行随机搜索以查找木马字符串的方法--这项任务与许多其他任务共享相同的属性。

    arXiv:2404.02151v1 Announce Type: cross  Abstract: We show that even the most recent safety-aligned LLMs are not robust to simple adaptive jailbreaking attacks. First, we demonstrate how to successfully leverage access to logprobs for jailbreaking: we initially design an adversarial prompt template (sometimes adapted to the target LLM), and then we apply random search on a suffix to maximize the target logprob (e.g., of the token "Sure"), potentially with multiple restarts. In this way, we achieve nearly 100\% attack success rate -- according to GPT-4 as a judge -- on GPT-3.5/4, Llama-2-Chat-7B/13B/70B, Gemma-7B, and R2D2 from HarmBench that was adversarially trained against the GCG attack. We also show how to jailbreak all Claude models -- that do not expose logprobs -- via either a transfer or prefilling attack with 100\% success rate. In addition, we show how to use random search on a restricted set of tokens for finding trojan strings in poisoned models -- a task that shares many s
    
[^2]: 焦点置信: 带有选择条件覆盖的整体预测

    Confidence on the Focal: Conformal Prediction with Selection-Conditional Coverage

    [https://arxiv.org/abs/2403.03868](https://arxiv.org/abs/2403.03868)

    该论文提出了一种构建具有有限样本精确覆盖的预测集的通用框架，可以解决在数据驱动情境中由于选择偏差导致的边缘有效预测区间误导问题。

    

    整体预测建立在边缘有效的预测区间上，该区间以某种规定的概率覆盖了随机抽取的新测试点的未知结果。在实践中，常见情况是，在看到测试单元后，从业者以数据驱动的方式决定关注哪些测试单元，并希望量化焦点单元的不确定性。在这种情况下，对于这些焦点单元的边缘有效预测区间可能会因选择偏差而具有误导性。本文提出了一个构建具有有限样本精确覆盖的预测集的通用框架，该覆盖是有条件于所选单元的。其一般形式适用于任意选择规则，并将Mondrian整体预测推广到多个测试单元和非等变分类器。然后，我们为多个现实的选择规则计算了适用于我们框架的计算效率实现，包括top-K选择、优化等。

    arXiv:2403.03868v1 Announce Type: cross  Abstract: Conformal prediction builds marginally valid prediction intervals which cover the unknown outcome of a randomly drawn new test point with a prescribed probability. In practice, a common scenario is that, after seeing the test unit(s), practitioners decide which test unit(s) to focus on in a data-driven manner, and wish to quantify the uncertainty for the focal unit(s). In such cases, marginally valid prediction intervals for these focal units can be misleading due to selection bias. This paper presents a general framework for constructing a prediction set with finite-sample exact coverage conditional on the unit being selected. Its general form works for arbitrary selection rules, and generalizes Mondrian Conformal Prediction to multiple test units and non-equivariant classifiers. We then work out computationally efficient implementation of our framework for a number of realistic selection rules, including top-K selection, optimization
    
[^3]: 深度Huber分位数回归网络

    Deep Huber quantile regression networks. (arXiv:2306.10306v1 [stat.ML])

    [http://arxiv.org/abs/2306.10306](http://arxiv.org/abs/2306.10306)

    DHQRN可以预测更一般的Huber分位数，并且在预测分布的尾部提供更好的预测。

    

    典型的机器学习回归应用旨在通过使用平方误差或绝对误差评分函数来报告预测概率分布的均值或中位数。发出更多预测概率分布的函数（分位数和期望值）的重要性已被认为是量化预测不确定性的手段。在深度学习（DL）应用程序中，通过分位数和期望值回归神经网络（QRNN和ERNN）可以实现这一点。在这里，我们介绍了深度Huber分位数回归网络（DHQRN），它将QRNN和ERNN嵌套为边缘情况。 DHQRN可以预测Huber分位数，这是更一般的函数，因为它们将分位数和期望值作为极限情况嵌套起来。主要思想是使用Huber分位数回归函数训练深度学习算法，这与Huber分位数功能一致。作为概念验证，DHQRN被应用于预测房价的真实数据集，并与其他回归技术进行比较。我们观察到，在几个误差指标中，DHQRN胜过其他技术，在预测分布的尾部提供更好的预测。

    Typical machine learning regression applications aim to report the mean or the median of the predictive probability distribution, via training with a squared or an absolute error scoring function. The importance of issuing predictions of more functionals of the predictive probability distribution (quantiles and expectiles) has been recognized as a means to quantify the uncertainty of the prediction. In deep learning (DL) applications, that is possible through quantile and expectile regression neural networks (QRNN and ERNN respectively). Here we introduce deep Huber quantile regression networks (DHQRN) that nest QRNNs and ERNNs as edge cases. DHQRN can predict Huber quantiles, which are more general functionals in the sense that they nest quantiles and expectiles as limiting cases. The main idea is to train a deep learning algorithm with the Huber quantile regression function, which is consistent for the Huber quantile functional. As a proof of concept, DHQRN are applied to predict hou
    
[^4]: (近似)群等变性下的逼近-泛化权衡

    Approximation-Generalization Trade-offs under (Approximate) Group Equivariance. (arXiv:2305.17592v1 [cs.LG])

    [http://arxiv.org/abs/2305.17592](http://arxiv.org/abs/2305.17592)

    本论文详细研究了通过对称性明确地引入任务特定的归纳偏差所导致的逼近-泛化权衡，并且证明了这种模型在捕获任务特定对称性的同时会改进泛化。这一结果对于提高机器学习领域的性能具有非常大的帮助。

    

    通过对称性明确地引入任务特定的归纳偏差已成为高性能机器学习模型开发中的常规设计准则。例如，群等变神经网络在蛋白质和药物设计等各个领域和应用中展现了卓越的性能。这种模型的普遍感觉是，将相关对称性整合到模型中会增强泛化能力。此外，有人认为，当数据和/或模型只能表现出$\textit{近似}$或$\textit{部分}$对称性时，最优或最好性能的模型是一个模型对齐于数据对称性的模型。在本文中，我们对这些直觉进行了正式的统一研究。首先，我们提出一般的数量界限，证明捕获任务特定对称性的模型将导致改进的泛化。事实上，我们的结果不要求变换是有限的，甚至不需要形成完整的....

    The explicit incorporation of task-specific inductive biases through symmetry has emerged as a general design precept in the development of high-performance machine learning models. For example, group equivariant neural networks have demonstrated impressive performance across various domains and applications such as protein and drug design. A prevalent intuition about such models is that the integration of relevant symmetry results in enhanced generalization. Moreover, it is posited that when the data and/or the model may only exhibit $\textit{approximate}$ or $\textit{partial}$ symmetry, the optimal or best-performing model is one where the model symmetry aligns with the data symmetry. In this paper, we conduct a formal unified investigation of these intuitions. To begin, we present general quantitative bounds that demonstrate how models capturing task-specific symmetries lead to improved generalization. In fact, our results do not require the transformations to be finite or even form
    
[^5]: 关于卷积神经网络中最大池化特征图的位移不变性

    On the Shift Invariance of Max Pooling Feature Maps in Convolutional Neural Networks. (arXiv:2209.11740v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2209.11740](http://arxiv.org/abs/2209.11740)

    本文研究了卷积神经网络中最大池化特征图的位移不变性问题，并提出了一种近似复数模的条件，实现了位移稳定性。实验证实了理论的有效性。

    

    本文致力于改善卷积神经网络（CNN）在图像分类领域中的数学可解释性。具体而言，我们解决了在其第一层中出现的不稳定性问题。当在像ImageNet这样的数据集上进行训练时，其第一层往往学习到与方向边通滤波器非常相似的参数。使用这样的Gabor滤波器进行子采样卷积容易出现混叠问题，导致对输入的小偏移敏感。在这个背景下，我们建立了最大池化算子近似复数模的条件，使其几乎具有位移不变性。然后，我们推导了子采样卷积后最大池化的位移稳定性度量。特别地，我们强调了滤波器的频率和方向在实现稳定性方面的关键作用。通过考虑基于双树复小波包变换的确定性特征提取器，即离散Gabor的一种特殊情况，我们通过实验证实了我们的理论。

    This paper focuses on improving the mathematical interpretability of convolutional neural networks (CNNs) in the context of image classification. Specifically, we tackle the instability issue arising in their first layer, which tends to learn parameters that closely resemble oriented band-pass filters when trained on datasets like ImageNet. Subsampled convolutions with such Gabor-like filters are prone to aliasing, causing sensitivity to small input shifts. In this context, we establish conditions under which the max pooling operator approximates a complex modulus, which is nearly shift invariant. We then derive a measure of shift invariance for subsampled convolutions followed by max pooling. In particular, we highlight the crucial role played by the filter's frequency and orientation in achieving stability. We experimentally validate our theory by considering a deterministic feature extractor based on the dual-tree complex wavelet packet transform, a particular case of discrete Gabor
    
[^6]: 面板回归模型的核范数规则化估计

    Nuclear Norm Regularized Estimation of Panel Regression Models. (arXiv:1810.10987v3 [econ.EM] UPDATED)

    [http://arxiv.org/abs/1810.10987](http://arxiv.org/abs/1810.10987)

    本文提出两种最小化凸目标函数的新估计方法，其中核范数罚项有助于解决低秩回归器下的交互固定效应模型的潜在识别问题，并且具有很重要的计算优势。

    

    本文研究具有交互固定效应的面板回归模型。我们提出了两种基于最小化凸目标函数的新估计方法。第一种方法最小化残差平方和，带有核（迹）范数规则化。第二种方法最小化残差的核范数。我们建立了两个估计器的一致性。这些估计器与现有的最小二乘（LS）估计器相比具有非常重要的计算优势，因为它们被定义为凸目标函数的最小化器。此外，核范数罚项有助于解决交互固定效应模型的潜在识别问题，尤其是当回归器是低秩的且因素数量未知时。我们还展示了如何通过使用我们的核范数规则化估计器构造渐近等效于Bai（2009年）和Moon和Weidner（2017年）最小二乘（LS）估计器的估计器。

    In this paper we investigate panel regression models with interactive fixed effects. We propose two new estimation methods that are based on minimizing convex objective functions. The first method minimizes the sum of squared residuals with a nuclear (trace) norm regularization. The second method minimizes the nuclear norm of the residuals. We establish the consistency of the two resulting estimators. Those estimators have a very important computational advantage compared to the existing least squares (LS) estimator, in that they are defined as minimizers of a convex objective function. In addition, the nuclear norm penalization helps to resolve a potential identification problem for interactive fixed effect models, in particular when the regressors are low-rank and the number of the factors is unknown. We also show how to construct estimators that are asymptotically equivalent to the least squares (LS) estimator in Bai (2009) and Moon and Weidner (2017) by using our nuclear norm regul
    

