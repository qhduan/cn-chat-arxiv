# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Verifix: Post-Training Correction to Improve Label Noise Robustness with Verified Samples](https://arxiv.org/abs/2403.08618) | 提出了后训练校正的新范式，通过奇异值分解算法Verifix在初始训练后校正模型权重以减轻标签噪声，避免了重新训练的需求 |
| [^2] | [SPDE priors for uncertainty quantification of end-to-end neural data assimilation schemes](https://arxiv.org/abs/2402.01855) | SPDE先验在最优插值中的应用及其与神经网络的联合学习问题，为大规模地球物理数据集的时空插值提供了一种新的方法。 |
| [^3] | [Energy-Based Concept Bottleneck Models: Unifying Prediction, Concept Intervention, and Conditional Interpretations.](http://arxiv.org/abs/2401.14142) | 基于能量的概念瓶颈模型统一了预测、概念干预和条件解释的功能，解决了现有方法在高阶非线性相互作用和复杂条件依赖关系上的限制。 |
| [^4] | [Nested Elimination: A Simple Algorithm for Best-Item Identification from Choice-Based Feedback.](http://arxiv.org/abs/2307.09295) | 嵌套消除是一种简单易实现的算法，通过利用创新的消除准则和嵌套结构，能够以最少的样本数量和高置信水平识别出最受欢迎的项目。 |
| [^5] | [Exploiting Observation Bias to Improve Matrix Completion.](http://arxiv.org/abs/2306.04775) | 本研究利用观测偏差来改进矩阵补全问题，提出一个简单的两阶段算法，实现了与对未观测协变量的监督学习性能相当的结果。 |
| [^6] | [Basis Function Encoding of Numerical Features in Factorization Machines for Improved Accuracy.](http://arxiv.org/abs/2305.14528) | 本文提供了一种能够将数字特征编码为基函数向量的方法，通过在因子机中将该方法应用于因子机中，可以改善推荐系统的准确性。 |
| [^7] | [Bayesian approach to Gaussian process regression with uncertain inputs.](http://arxiv.org/abs/2305.11586) | 本文提出了一种新的高斯过程回归技术，通过贝叶斯方法将输入数据的不确定性纳入回归模型预测中。在数值实验中展示了该方法具有普适性和不错的表现。 |

# 详细

[^1]: Verifix: 后训练校正以改善具有经过验证样本的标签噪声鲁棒性

    Verifix: Post-Training Correction to Improve Label Noise Robustness with Verified Samples

    [https://arxiv.org/abs/2403.08618](https://arxiv.org/abs/2403.08618)

    提出了后训练校正的新范式，通过奇异值分解算法Verifix在初始训练后校正模型权重以减轻标签噪声，避免了重新训练的需求

    

    标签错误，即训练样本具有不正确的标签，可能严重损害机器学习模型的性能。这种错误往往来自非专家标注或敌对攻击。获取大型、完全标记的数据集成本高，当有干净的数据集可用时，重新训练大型模型就变得计算昂贵。为了解决这一挑战，我们提出了后训练校正，这是一种在初始训练后调整模型参数以减轻标签噪声的新范式，消除了重新训练的需要。我们引入了Verifix，这是一种基于奇异值分解（SVD）的新算法，利用一个小的、经过验证的数据集，通过单个更新校正模型权重。Verifix使用SVD估计干净激活空间，然后将模型的权重投影到这个空间上，以抑制对应于损坏数据的激活。我们展示了Verifix的有效性。

    arXiv:2403.08618v1 Announce Type: cross  Abstract: Label corruption, where training samples have incorrect labels, can significantly degrade the performance of machine learning models. This corruption often arises from non-expert labeling or adversarial attacks. Acquiring large, perfectly labeled datasets is costly, and retraining large models from scratch when a clean dataset becomes available is computationally expensive. To address this challenge, we propose Post-Training Correction, a new paradigm that adjusts model parameters after initial training to mitigate label noise, eliminating the need for retraining. We introduce Verifix, a novel Singular Value Decomposition (SVD) based algorithm that leverages a small, verified dataset to correct the model weights using a single update. Verifix uses SVD to estimate a Clean Activation Space and then projects the model's weights onto this space to suppress activations corresponding to corrupted data. We demonstrate Verifix's effectiveness 
    
[^2]: SPDE先验在端到端神经数据同化方案的不确定性量化中的应用

    SPDE priors for uncertainty quantification of end-to-end neural data assimilation schemes

    [https://arxiv.org/abs/2402.01855](https://arxiv.org/abs/2402.01855)

    SPDE先验在最优插值中的应用及其与神经网络的联合学习问题，为大规模地球物理数据集的时空插值提供了一种新的方法。

    

    大规模地球物理数据集的时空插值通常通过最优插值(Optimal Interpolation，OI)和更复杂的基于模型或数据驱动的数据同化技术来处理。在过去的十年中，随机偏微分方程(Spatio-temporal Partial Differential Equations，SPDE)和高斯马尔科夫随机场(Gaussian Markov Random Fields，GMRF)之间的联系开辟了一条新的途径，用于处理最优插值中的大数据集和物理诱导协方差矩阵。深度学习社区的最新进展也使得可以将这个问题视为嵌入数据同化变分框架的神经网络体系结构的联合学习问题。重建任务被视为一个包含在变分内部成本中的先验学习问题和后者的基于梯度的最小化：先验模型和求解器都被表示为具有自动微分的神经网络，可以通过最小化损失函数来训练，该损失函数通常被表示为一些真实值和重建值之间的均方误差。

    The spatio-temporal interpolation of large geophysical datasets has historically been adressed by Optimal Interpolation (OI) and more sophisticated model-based or data-driven DA techniques. In the last ten years, the link established between Stochastic Partial Differential Equations (SPDE) and Gaussian Markov Random Fields (GMRF) opened a new way of handling both large datasets and physically-induced covariance matrix in Optimal Interpolation. Recent advances in the deep learning community also enables to adress this problem as neural architecture embedding data assimilation variational framework. The reconstruction task is seen as a joint learning problem of the prior involved in the variational inner cost and the gradient-based minimization of the latter: both prior models and solvers are stated as neural networks with automatic differentiation which can be trained by minimizing a loss function, typically stated as the mean squared error between some ground truth and the reconstructi
    
[^3]: 基于能量的概念瓶颈模型：统一预测、概念干预和条件解释

    Energy-Based Concept Bottleneck Models: Unifying Prediction, Concept Intervention, and Conditional Interpretations. (arXiv:2401.14142v1 [cs.CV])

    [http://arxiv.org/abs/2401.14142](http://arxiv.org/abs/2401.14142)

    基于能量的概念瓶颈模型统一了预测、概念干预和条件解释的功能，解决了现有方法在高阶非线性相互作用和复杂条件依赖关系上的限制。

    

    现有方法，如概念瓶颈模型 (CBM)，在为黑盒深度学习模型提供基于概念的解释方面取得了成功。它们通常通过在给定输入的情况下预测概念，然后在给定预测的概念的情况下预测最终的类别标签。然而，它们经常无法捕捉到概念之间的高阶非线性相互作用，例如纠正一个预测的概念（例如“黄色胸部”）无法帮助纠正高度相关的概念（例如“黄色腹部”），导致最终准确率不理想；它们无法自然地量化不同概念和类别标签之间的复杂条件依赖关系（例如对于一个带有类别标签“Kentucky Warbler”和概念“黑色嘴巴”的图像，模型能够正确预测另一个概念“黑色冠”的概率是多少），因此无法提供关于黑盒模型工作原理更深层次的洞察。针对这些限制，我们提出了基于能量的概念瓶颈模型（Energy-based Concept Bottleneck Models）。

    Existing methods, such as concept bottleneck models (CBMs), have been successful in providing concept-based interpretations for black-box deep learning models. They typically work by predicting concepts given the input and then predicting the final class label given the predicted concepts. However, (1) they often fail to capture the high-order, nonlinear interaction between concepts, e.g., correcting a predicted concept (e.g., "yellow breast") does not help correct highly correlated concepts (e.g., "yellow belly"), leading to suboptimal final accuracy; (2) they cannot naturally quantify the complex conditional dependencies between different concepts and class labels (e.g., for an image with the class label "Kentucky Warbler" and a concept "black bill", what is the probability that the model correctly predicts another concept "black crown"), therefore failing to provide deeper insight into how a black-box model works. In response to these limitations, we propose Energy-based Concept Bot
    
[^4]: 嵌套消除：一种从基于选择的反馈中识别最佳项目的简单算法

    Nested Elimination: A Simple Algorithm for Best-Item Identification from Choice-Based Feedback. (arXiv:2307.09295v1 [cs.LG])

    [http://arxiv.org/abs/2307.09295](http://arxiv.org/abs/2307.09295)

    嵌套消除是一种简单易实现的算法，通过利用创新的消除准则和嵌套结构，能够以最少的样本数量和高置信水平识别出最受欢迎的项目。

    

    我们研究了基于选择的反馈中识别最佳项目的问题。在这个问题中，公司依次向一群顾客展示显示集，并收集他们的选择。目标是以最少的样本数量和高置信水平识别出最受欢迎的项目。我们提出了一种基于消除的算法，即嵌套消除(Nested Elimination，NE)，它受到信息理论下界所暗示的嵌套结构的启发。NE的结构简单，易于实施，具有对样本复杂度的强大理论保证。具体而言，NE利用了一种创新的消除准则，并避免了解决任何复杂的组合优化问题的需要。我们提供了NE的特定实例和非渐近性的样本复杂度的上界。我们还展示了NE实现了高阶最坏情况渐近最优性。最后，来自合成和真实数据的数值实验验证了我们的理论。

    We study the problem of best-item identification from choice-based feedback. In this problem, a company sequentially and adaptively shows display sets to a population of customers and collects their choices. The objective is to identify the most preferred item with the least number of samples and at a high confidence level. We propose an elimination-based algorithm, namely Nested Elimination (NE), which is inspired by the nested structure implied by the information-theoretic lower bound. NE is simple in structure, easy to implement, and has a strong theoretical guarantee for sample complexity. Specifically, NE utilizes an innovative elimination criterion and circumvents the need to solve any complex combinatorial optimization problem. We provide an instance-specific and non-asymptotic bound on the expected sample complexity of NE. We also show NE achieves high-order worst-case asymptotic optimality. Finally, numerical experiments from both synthetic and real data corroborate our theore
    
[^5]: 利用观测偏差提高矩阵补全的方法研究

    Exploiting Observation Bias to Improve Matrix Completion. (arXiv:2306.04775v1 [cs.LG])

    [http://arxiv.org/abs/2306.04775](http://arxiv.org/abs/2306.04775)

    本研究利用观测偏差来改进矩阵补全问题，提出一个简单的两阶段算法，实现了与对未观测协变量的监督学习性能相当的结果。

    

    我们考虑了一种变形的矩阵补全问题，其中输入数据以偏差的方式呈现，类似于Ma和Chen所引入的模型。我们的目标是利用偏差与感兴趣的结果之间的共享信息来改进预测。为此，我们提出了一个简单的两阶段算法：（i）将观测模式解释为完全观测的噪声矩阵，我们对观测模式应用传统的矩阵补全方法来估计潜在因素之间的距离； (ii)我们对恢复的特征应用监督学习来填补缺失观察。我们建立了有限样本误差率，这些误差率与相应的监督学习参数率相竞争，这表明我们的学习性能与使用未观测协变量相当。实证评估使用真实世界数据集反映了类似的表现。

    We consider a variant of matrix completion where entries are revealed in a biased manner, adopting a model akin to that introduced by Ma and Chen. Instead of treating this observation bias as a disadvantage, as is typically the case, our goal is to exploit the shared information between the bias and the outcome of interest to improve predictions. Towards this, we propose a simple two-stage algorithm: (i) interpreting the observation pattern as a fully observed noisy matrix, we apply traditional matrix completion methods to the observation pattern to estimate the distances between the latent factors; (ii) we apply supervised learning on the recovered features to impute missing observations. We establish finite-sample error rates that are competitive with the corresponding supervised learning parametric rates, suggesting that our learning performance is comparable to having access to the unobserved covariates. Empirical evaluation using a real-world dataset reflects similar performance g
    
[^6]: 基函数编码改善因子机中数字特征的准确性

    Basis Function Encoding of Numerical Features in Factorization Machines for Improved Accuracy. (arXiv:2305.14528v1 [cs.LG])

    [http://arxiv.org/abs/2305.14528](http://arxiv.org/abs/2305.14528)

    本文提供了一种能够将数字特征编码为基函数向量的方法，通过在因子机中将该方法应用于因子机中，可以改善推荐系统的准确性。

    

    因子机(FM)变体被广泛用于大规模实时内容推荐系统，因为它们在模型准确性和训练推理的低计算成本之间提供了出色的平衡。本文提供了一种系统、理论上合理的方法，通过将数值特征编码为所选函数集的函数值向量将数值特征纳入FM变体。

    Factorization machine (FM) variants are widely used for large scale real-time content recommendation systems, since they offer an excellent balance between model accuracy and low computational costs for training and inference. These systems are trained on tabular data with both numerical and categorical columns. Incorporating numerical columns poses a challenge, and they are typically incorporated using a scalar transformation or binning, which can be either learned or chosen a-priori. In this work, we provide a systematic and theoretically-justified way to incorporate numerical features into FM variants by encoding them into a vector of function values for a set of functions of one's choice.  We view factorization machines as approximators of segmentized functions, namely, functions from a field's value to the real numbers, assuming the remaining fields are assigned some given constants, which we refer to as the segment. From this perspective, we show that our technique yields a model
    
[^7]: 高斯过程回归的贝叶斯方法中融入不确定输入

    Bayesian approach to Gaussian process regression with uncertain inputs. (arXiv:2305.11586v1 [cs.LG])

    [http://arxiv.org/abs/2305.11586](http://arxiv.org/abs/2305.11586)

    本文提出了一种新的高斯过程回归技术，通过贝叶斯方法将输入数据的不确定性纳入回归模型预测中。在数值实验中展示了该方法具有普适性和不错的表现。

    

    传统高斯过程回归仅假设模型观测数据的输出具有噪声。然而，在许多科学和工程应用中，由于建模假设、测量误差等因素，观测数据的输入位置可能也存在不确定性。在本文中，我们提出了一种贝叶斯方法，将输入数据的可变性融入到高斯过程回归中。考虑两种可观测量——具有固定输入的噪声污染输出和具有先验分布定义的不确定输入，通过贝叶斯框架估计后验分布以推断不确定的数据位置。然后，利用边际化方法将这些输入的量化不确定性纳入高斯过程预测中。通过几个数值实验，展示了这种新回归技术的有效性，在其中观察到不同水平输入数据不确定性下的普适良好表现。

    Conventional Gaussian process regression exclusively assumes the existence of noise in the output data of model observations. In many scientific and engineering applications, however, the input locations of observational data may also be compromised with uncertainties owing to modeling assumptions, measurement errors, etc. In this work, we propose a Bayesian method that integrates the variability of input data into Gaussian process regression. Considering two types of observables -- noise-corrupted outputs with fixed inputs and those with prior-distribution-defined uncertain inputs, a posterior distribution is estimated via a Bayesian framework to infer the uncertain data locations. Thereafter, such quantified uncertainties of inputs are incorporated into Gaussian process predictions by means of marginalization. The effectiveness of this new regression technique is demonstrated through several numerical examples, in which a consistently good performance of generalization is observed, w
    

