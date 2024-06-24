# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Provable Privacy with Non-Private Pre-Processing](https://arxiv.org/abs/2403.13041) | 提出了一个框架，能够评估非私密数据相关预处理算法引起的额外隐私成本，并利用平滑DP和预处理算法的有界敏感性建立整体隐私保证的上限 |
| [^2] | [Testing Calibration in Subquadratic Time](https://arxiv.org/abs/2402.13187) | 该论文通过属性测试算法方面的研究，提出了一种基于近似线性规划的算法，可以在信息理论上最优地解决校准性测试问题（最多一个常数倍数）。 |
| [^3] | [Directional Convergence Near Small Initializations and Saddles in Two-Homogeneous Neural Networks](https://arxiv.org/abs/2402.09226) | 本文研究了两次齐次神经网络在小初值附近的梯度流动，发现权重会在方向上收敛到神经相关函数的KKT点和某些鞍点附近。 |
| [^4] | [Fast sampling from constrained spaces using the Metropolis-adjusted Mirror Langevin algorithm](https://arxiv.org/abs/2312.08823) | 该论文提出了一种名为Metropolis-adjusted Mirror Langevin算法的方法，用于从约束空间中进行快速采样。这种算法是对Mirror Langevin算法的改进，通过添加接受-拒绝过滤器来消除渐近偏差，并具有指数优化依赖。 |
| [^5] | [Metric Space Magnitude for Evaluating the Diversity of Latent Representations](https://arxiv.org/abs/2311.16054) | 基于度量空间大小的潜在表示多样性度量，可稳定计算，能够进行多尺度比较，在多个领域和任务中展现出优越性能。 |
| [^6] | [One-Shot Strategic Classification Under Unknown Costs](https://arxiv.org/abs/2311.02761) | 本研究首次研究了在未知响应下一次性策略分类的情景，针对用户成本函数不确定性，提出解决方案并将任务定义为极小-极大问题。 |
| [^7] | [Accelerating Approximate Thompson Sampling with Underdamped Langevin Monte Carlo.](http://arxiv.org/abs/2401.11665) | 本文提出了一种使用欠阻尼 Langevin Monte Carlo 加速的近似 Thompson 采样策略，通过特定势函数的设计改善了高维问题中的样本复杂度，并在高维赌博机问题中进行了验证。 |
| [^8] | [Exponential Quantum Communication Advantage in Distributed Learning.](http://arxiv.org/abs/2310.07136) | 在分布式学习中，我们提出了一个基于量子网络的框架，可以使用指数级较少的通信和相对较小的时间和空间复杂度开销进行推理和训练。这是第一个展示了具有密集经典数据的通用机器学习问题具有指数量子优势的例子。 |
| [^9] | [Incentivizing High-Quality Content in Online Recommender Systems.](http://arxiv.org/abs/2306.07479) | 本文研究了在线推荐系统中激励高质量内容的算法问题，经典的在线学习算法会激励生产者创建低质量的内容，但本文提出的一种算法通过惩罚低质量内容的创建者，成功地激励了生产者创造高质量的内容。 |
| [^10] | [Hinge-Wasserstein: Mitigating Overconfidence in Regression by Classification.](http://arxiv.org/abs/2306.00560) | 该论文提出了一种基于Wasserstein距离的损失函数hinge-Wasserstein，用于缓解回归任务中由于过度自信导致的不确定性问题。这种损失函数有效提高了aleatoric和epistemic不确定性的质量。 |
| [^11] | [Optimal Estimates for Pairwise Learning with Deep ReLU Networks.](http://arxiv.org/abs/2305.19640) | 本文研究了深度ReLU网络中的成对学习，提出了一个针对一般损失函数的误差估计的尖锐界限，并基于成对最小二乘损失得出几乎最优的过度泛化误差界限。 |

# 详细

[^1]: 具有非私密预处理的可证明隐私

    Provable Privacy with Non-Private Pre-Processing

    [https://arxiv.org/abs/2403.13041](https://arxiv.org/abs/2403.13041)

    提出了一个框架，能够评估非私密数据相关预处理算法引起的额外隐私成本，并利用平滑DP和预处理算法的有界敏感性建立整体隐私保证的上限

    

    当分析差分私密（DP）机器学习管道时，通常会忽略数据相关的预处理的潜在隐私成本。在这项工作中，我们提出了一个通用框架，用于评估由非私密数据相关预处理算法引起的额外隐私成本。我们的框架通过利用两个新的技术概念建立了整体隐私保证的上限：一种称为平滑DP的DP变体以及预处理算法的有界敏感性。

    arXiv:2403.13041v1 Announce Type: cross  Abstract: When analysing Differentially Private (DP) machine learning pipelines, the potential privacy cost of data-dependent pre-processing is frequently overlooked in privacy accounting. In this work, we propose a general framework to evaluate the additional privacy cost incurred by non-private data-dependent pre-processing algorithms. Our framework establishes upper bounds on the overall privacy guarantees by utilising two new technical notions: a variant of DP termed Smooth DP and the bounded sensitivity of the pre-processing algorithms. In addition to the generic framework, we provide explicit overall privacy guarantees for multiple data-dependent pre-processing algorithms, such as data imputation, quantization, deduplication and PCA, when used in combination with several DP algorithms. Notably, this framework is also simple to implement, allowing direct integration into existing DP pipelines.
    
[^2]: 在次线性时间内测试校准性

    Testing Calibration in Subquadratic Time

    [https://arxiv.org/abs/2402.13187](https://arxiv.org/abs/2402.13187)

    该论文通过属性测试算法方面的研究，提出了一种基于近似线性规划的算法，可以在信息理论上最优地解决校准性测试问题（最多一个常数倍数）。

    

    在最近的机器学习和决策制定文献中，校准性已经成为二元预测模型输出的一个值得期望和广泛研究的统计性质。然而，测量模型校准性的算法方面仍然相对较少被探索。在论文 [BGHN23] 的启发下，该论文提出了一个严格的框架来衡量到校准性的距离，我们通过属性测试的视角引入了校准性研究的算法方面。我们定义了从样本中进行校准性测试的问题，其中从分布 $\mathcal{D}$（预测，二元结果）中给出 $n$ 次抽样，我们的目标是区分 $\mathcal{D}$ 完全校准和 $\mathcal{D}$ 距离校准性为 $\varepsilon$ 的情况。

    arXiv:2402.13187v1 Announce Type: new  Abstract: In the recent literature on machine learning and decision making, calibration has emerged as a desirable and widely-studied statistical property of the outputs of binary prediction models. However, the algorithmic aspects of measuring model calibration have remained relatively less well-explored. Motivated by [BGHN23], which proposed a rigorous framework for measuring distances to calibration, we initiate the algorithmic study of calibration through the lens of property testing. We define the problem of calibration testing from samples where given $n$ draws from a distribution $\mathcal{D}$ on (predictions, binary outcomes), our goal is to distinguish between the case where $\mathcal{D}$ is perfectly calibrated, and the case where $\mathcal{D}$ is $\varepsilon$-far from calibration.   We design an algorithm based on approximate linear programming, which solves calibration testing information-theoretically optimally (up to constant factor
    
[^3]: 在两次齐次神经网络的小初值和鞍点附近的方向收敛

    Directional Convergence Near Small Initializations and Saddles in Two-Homogeneous Neural Networks

    [https://arxiv.org/abs/2402.09226](https://arxiv.org/abs/2402.09226)

    本文研究了两次齐次神经网络在小初值附近的梯度流动，发现权重会在方向上收敛到神经相关函数的KKT点和某些鞍点附近。

    

    本文研究了两次齐次神经网络在小初值附近的梯度流动力学，其中所有权重都初始化在原点附近。针对平方误差和逻辑损失，论文证明，对于足够小的初始值，梯度流动动态在原点附近花费足够的时间，使得神经网络的权重可以近似地在方向上收敛到神经相关函数的Karush-Kuhn-Tucker（KKT）点，该函数量化了神经网络输出与训练数据集中相应标签之间的关联性。

    arXiv:2402.09226v1 Announce Type: new Abstract: This paper examines gradient flow dynamics of two-homogeneous neural networks for small initializations, where all weights are initialized near the origin. For both square and logistic losses, it is shown that for sufficiently small initializations, the gradient flow dynamics spend sufficient time in the neighborhood of the origin to allow the weights of the neural network to approximately converge in direction to the Karush-Kuhn-Tucker (KKT) points of a neural correlation function that quantifies the correlation between the output of the neural network and corresponding labels in the training data set. For square loss, it has been observed that neural networks undergo saddle-to-saddle dynamics when initialized close to the origin. Motivated by this, this paper also shows a similar directional convergence among weights of small magnitude in the neighborhood of certain saddle points.
    
[^4]: 使用Metropolis-adjusted Mirror Langevin算法从约束空间中快速采样

    Fast sampling from constrained spaces using the Metropolis-adjusted Mirror Langevin algorithm

    [https://arxiv.org/abs/2312.08823](https://arxiv.org/abs/2312.08823)

    该论文提出了一种名为Metropolis-adjusted Mirror Langevin算法的方法，用于从约束空间中进行快速采样。这种算法是对Mirror Langevin算法的改进，通过添加接受-拒绝过滤器来消除渐近偏差，并具有指数优化依赖。

    

    我们提出了一种新的方法，称为Metropolis-adjusted Mirror Langevin算法，用于从其支持是紧凸集的分布中进行近似采样。该算法在Mirror Langevin算法（Zhang et al., 2020）的单步马尔科夫链中添加了一个接受-拒绝过滤器，Mirror Langevin算法是Mirror Langevin动力学的基本离散化。由于包含了这个过滤器，我们的方法相对于目标是无偏的，而已知的Mirror Langevin算法等Mirror Langevin动力学的离散化具有渐近偏差。对于该算法，我们还给出了混合到一个相对平滑、凸性好且与自共轭镜像函数相关的约束分布所需迭代次数的上界。由于包含Metropolis-Hastings过滤器导致的马尔科夫链是可逆的，我们得到了对误差的指数优化依赖。

    We propose a new method called the Metropolis-adjusted Mirror Langevin algorithm for approximate sampling from distributions whose support is a compact and convex set. This algorithm adds an accept-reject filter to the Markov chain induced by a single step of the Mirror Langevin algorithm (Zhang et al., 2020), which is a basic discretisation of the Mirror Langevin dynamics. Due to the inclusion of this filter, our method is unbiased relative to the target, while known discretisations of the Mirror Langevin dynamics including the Mirror Langevin algorithm have an asymptotic bias. For this algorithm, we also give upper bounds for the number of iterations taken to mix to a constrained distribution whose potential is relatively smooth, convex, and Lipschitz continuous with respect to a self-concordant mirror function. As a consequence of the reversibility of the Markov chain induced by the inclusion of the Metropolis-Hastings filter, we obtain an exponentially better dependence on the erro
    
[^5]: 用于评估潜在表示多样性的度量空间大小

    Metric Space Magnitude for Evaluating the Diversity of Latent Representations

    [https://arxiv.org/abs/2311.16054](https://arxiv.org/abs/2311.16054)

    基于度量空间大小的潜在表示多样性度量，可稳定计算，能够进行多尺度比较，在多个领域和任务中展现出优越性能。

    

    度量空间的大小是一种近期建立的不变性，能够在多个尺度上提供空间的“有效大小”的衡量，并捕捉到许多几何属性。我们发展了一系列基于大小的潜在表示内在多样性度量，形式化了有限度量空间大小函数之间的新颖不相似性概念。我们的度量在数据扰动下保证稳定，可以高效计算，并且能够对潜在表示进行严格的多尺度比较。我们展示了我们的度量在实验套件中的实用性和卓越性能，包括不同领域和任务的多样性评估、模式崩溃检测以及用于文本、图像和图形数据的生成模型评估。

    The magnitude of a metric space is a recently-established invariant, providing a measure of the 'effective size' of a space across multiple scales while also capturing numerous geometrical properties. We develop a family of magnitude-based measures of the intrinsic diversity of latent representations, formalising a novel notion of dissimilarity between magnitude functions of finite metric spaces. Our measures are provably stable under perturbations of the data, can be efficiently calculated, and enable a rigorous multi-scale comparison of latent representations. We show the utility and superior performance of our measures in an experimental suite that comprises different domains and tasks, including the evaluation of diversity, the detection of mode collapse, and the evaluation of generative models for text, image, and graph data.
    
[^6]: 一次性策略分类在未知成本下的研究

    One-Shot Strategic Classification Under Unknown Costs

    [https://arxiv.org/abs/2311.02761](https://arxiv.org/abs/2311.02761)

    本研究首次研究了在未知响应下一次性策略分类的情景，针对用户成本函数不确定性，提出解决方案并将任务定义为极小-极大问题。

    

    策略分类的目标是学习对策略输入操纵具有鲁棒性的决策规则。之前的研究假设这些响应是已知的；而最近的一些研究处理未知响应，但它们专门研究重复模型部署的在线设置。然而，在许多领域，特别是在公共政策中，一个常见的激励用例中，多次部署是不可行的，甚至一个糟糕的轮次都是不可接受的。为了填补这一空白，我们首次引入了在未知响应下的一次性策略分类的正式研究，这需要在一次性选择一个分类器。着重关注用户成本函数中的不确定性，我们首先证明对于一类广泛的成本，即使对真实成本的小误差也可能在最坏情况下导致准确性降至极低水平。鉴于此，我们将任务框定为极小-极大问题，目标是识别

    arXiv:2311.02761v2 Announce Type: replace  Abstract: The goal of strategic classification is to learn decision rules which are robust to strategic input manipulation. Earlier works assume that these responses are known; while some recent works handle unknown responses, they exclusively study online settings with repeated model deployments. But there are many domains$\unicode{x2014}$particularly in public policy, a common motivating use case$\unicode{x2014}$where multiple deployments are infeasible, or where even one bad round is unacceptable. To address this gap, we initiate the formal study of one-shot strategic classification under unknown responses, which requires committing to a single classifier once. Focusing on uncertainty in the users' cost function, we begin by proving that for a broad class of costs, even a small mis-estimation of the true cost can entail trivial accuracy in the worst case. In light of this, we frame the task as a minimax problem, with the goal of identifying
    
[^7]: 使用欠阻尼 Langevin Monte Carlo 加速近似 Thompson 采样

    Accelerating Approximate Thompson Sampling with Underdamped Langevin Monte Carlo. (arXiv:2401.11665v1 [stat.ML])

    [http://arxiv.org/abs/2401.11665](http://arxiv.org/abs/2401.11665)

    本文提出了一种使用欠阻尼 Langevin Monte Carlo 加速的近似 Thompson 采样策略，通过特定势函数的设计改善了高维问题中的样本复杂度，并在高维赌博机问题中进行了验证。

    

    使用欠阻尼 Langevin Monte Carlo 的近似 Thompson 采样方法扩展了其适用范围，从高斯后验采样扩展到更一般的平滑后验。然而，在高维问题中要求高准确性时，仍然面临可扩展性问题。为了解决这个问题，我们提出了一种近似 Thompson 采样策略，利用欠阻尼 Langevin Monte Carlo，后者是模拟高维后验的通用工具。基于标准的平滑性和对数凹性条件，我们研究了使用特定势函数的加速后验集中和采样。该设计改进了实现对数遗憾的样本复杂度，从$\mathcal{\tilde O}(d)$改进到$\mathcal{\tilde O}(\sqrt{d})$。我们还通过合成实验在高维赌博机问题中经验验证了我们算法的可扩展性和鲁棒性。

    Approximate Thompson sampling with Langevin Monte Carlo broadens its reach from Gaussian posterior sampling to encompass more general smooth posteriors. However, it still encounters scalability issues in high-dimensional problems when demanding high accuracy. To address this, we propose an approximate Thompson sampling strategy, utilizing underdamped Langevin Monte Carlo, where the latter is the go-to workhorse for simulations of high-dimensional posteriors. Based on the standard smoothness and log-concavity conditions, we study the accelerated posterior concentration and sampling using a specific potential function. This design improves the sample complexity for realizing logarithmic regrets from $\mathcal{\tilde O}(d)$ to $\mathcal{\tilde O}(\sqrt{d})$. The scalability and robustness of our algorithm are also empirically validated through synthetic experiments in high-dimensional bandit problems.
    
[^8]: 分布式学习中的指数量子通信优势

    Exponential Quantum Communication Advantage in Distributed Learning. (arXiv:2310.07136v1 [quant-ph])

    [http://arxiv.org/abs/2310.07136](http://arxiv.org/abs/2310.07136)

    在分布式学习中，我们提出了一个基于量子网络的框架，可以使用指数级较少的通信和相对较小的时间和空间复杂度开销进行推理和训练。这是第一个展示了具有密集经典数据的通用机器学习问题具有指数量子优势的例子。

    

    使用超过单个设备内存容量的大型机器学习模型进行训练和推理需要设计分布式架构，必须考虑通信限制。我们提出了一种在量子网络上进行分布式计算的框架，其中数据被编码为特殊的量子态。我们证明，在该框架内的某些模型中，使用梯度下降进行推理和训练的通信开销相对于其经典对应模型可以指数级降低，并且相对于标准基于梯度的方法，时间和空间复杂性开销相对较小。据我们所知，这是第一个在具有密集经典数据的通用机器学习问题的情况下，无论数据编码成本如何，都具有指数量子优势的示例。此外，我们还展示了该类模型可以编码输入的高度非线性特征，并且它们的表达能力呈指数增加。

    Training and inference with large machine learning models that far exceed the memory capacity of individual devices necessitates the design of distributed architectures, forcing one to contend with communication constraints. We present a framework for distributed computation over a quantum network in which data is encoded into specialized quantum states. We prove that for certain models within this framework, inference and training using gradient descent can be performed with exponentially less communication compared to their classical analogs, and with relatively modest time and space complexity overheads relative to standard gradient-based methods. To our knowledge, this is the first example of exponential quantum advantage for a generic class of machine learning problems with dense classical data that holds regardless of the data encoding cost. Moreover, we show that models in this class can encode highly nonlinear features of their inputs, and their expressivity increases exponenti
    
[^9]: 在在线推荐系统中激励高质量内容

    Incentivizing High-Quality Content in Online Recommender Systems. (arXiv:2306.07479v1 [cs.GT])

    [http://arxiv.org/abs/2306.07479](http://arxiv.org/abs/2306.07479)

    本文研究了在线推荐系统中激励高质量内容的算法问题，经典的在线学习算法会激励生产者创建低质量的内容，但本文提出的一种算法通过惩罚低质量内容的创建者，成功地激励了生产者创造高质量的内容。

    

    对于像TikTok和YouTube这样的内容推荐系统，平台的决策算法塑造了内容生产者的激励，包括生产者在内容质量上投入多少努力。许多平台采用在线学习，这会产生跨时间的激励，因为今天生产的内容会影响未来内容的推荐。在本文中，我们研究了在线学习产生的激励，分析了在纳什均衡下生产的内容质量。我们发现，像Hedge和EXP3这样的经典在线学习算法会激励生产者创建低质量的内容。特别地，内容质量在学习率方面有上限，并且随着典型学习率进展而趋近于零。在这一负面结果的基础上，我们设计了一种不同的学习算法——基于惩罚创建低质量内容的生产者——正确激励生产者创建高质量内容。我们的算法依赖于新颖的策略性赌博机问题，并克服了在组合设置中应用对抗性技术的挑战。在模拟和真实数据的实验中，我们的算法成功地激励生产者创建高质量内容。

    For content recommender systems such as TikTok and YouTube, the platform's decision algorithm shapes the incentives of content producers, including how much effort the content producers invest in the quality of their content. Many platforms employ online learning, which creates intertemporal incentives, since content produced today affects recommendations of future content. In this paper, we study the incentives arising from online learning, analyzing the quality of content produced at a Nash equilibrium. We show that classical online learning algorithms, such as Hedge and EXP3, unfortunately incentivize producers to create low-quality content. In particular, the quality of content is upper bounded in terms of the learning rate and approaches zero for typical learning rate schedules. Motivated by this negative result, we design a different learning algorithm -- based on punishing producers who create low-quality content -- that correctly incentivizes producers to create high-quality co
    
[^10]: Hinge-Wasserstein: 通过分类避免回归中的过度自信

    Hinge-Wasserstein: Mitigating Overconfidence in Regression by Classification. (arXiv:2306.00560v1 [cs.LG])

    [http://arxiv.org/abs/2306.00560](http://arxiv.org/abs/2306.00560)

    该论文提出了一种基于Wasserstein距离的损失函数hinge-Wasserstein，用于缓解回归任务中由于过度自信导致的不确定性问题。这种损失函数有效提高了aleatoric和epistemic不确定性的质量。

    

    现代深度神经网络在性能方面得到了巨大的提高，但它们容易产生过度自信。在模糊甚至不可预测的现实世界场景中，这种过度自信可能对应用程序的安全性构成重大风险。针对回归任务，采用回归-分类方法有潜力缓解这些歧义，因为它可以预测所需输出的离散概率密度。然而，密度估计仍然倾向于过度自信，尤其是在使用常见的NLL损失函数训练时。为了缓解这种过度自信的问题，我们提出了一种基于Wasserstein距离的损失函数，即hinge-Wasserstein。与以前的工作相比，此损失显着提高了两种不确定性的质量： aleatoric不确定性和epistemic不确定性。我们在合成数据集上展示了新损失的能力，其中两种类型的不确定性可以分别控制。此外，作为现实世界场景的演示，我们在基准数据集上评估了我们的方法。

    Modern deep neural networks are prone to being overconfident despite their drastically improved performance. In ambiguous or even unpredictable real-world scenarios, this overconfidence can pose a major risk to the safety of applications. For regression tasks, the regression-by-classification approach has the potential to alleviate these ambiguities by instead predicting a discrete probability density over the desired output. However, a density estimator still tends to be overconfident when trained with the common NLL loss. To mitigate the overconfidence problem, we propose a loss function, hinge-Wasserstein, based on the Wasserstein Distance. This loss significantly improves the quality of both aleatoric and epistemic uncertainty, compared to previous work. We demonstrate the capabilities of the new loss on a synthetic dataset, where both types of uncertainty are controlled separately. Moreover, as a demonstration for real-world scenarios, we evaluate our approach on the benchmark dat
    
[^11]: 深度ReLU网络中的成对学习最优估计

    Optimal Estimates for Pairwise Learning with Deep ReLU Networks. (arXiv:2305.19640v1 [stat.ML])

    [http://arxiv.org/abs/2305.19640](http://arxiv.org/abs/2305.19640)

    本文研究了深度ReLU网络中的成对学习，提出了一个针对一般损失函数的误差估计的尖锐界限，并基于成对最小二乘损失得出几乎最优的过度泛化误差界限。

    

    成对学习指的是在损失函数中考虑一对样本的学习任务。本文研究了深度ReLU网络中的成对学习，并估计了过度泛化误差。对于满足某些温和条件的一般损失函数，建立了误差估计的尖锐界限，其误差估计的阶数为O（（Vlog（n）/ n）1 /（2-β））。特别地，对于成对最小二乘损失，我们得到了过度泛化误差的几乎最优界限，在真实的预测器满足某些光滑性正则性时，最优界限达到了最小化界限，差距仅为对数项。

    Pairwise learning refers to learning tasks where a loss takes a pair of samples into consideration. In this paper, we study pairwise learning with deep ReLU networks and estimate the excess generalization error. For a general loss satisfying some mild conditions, a sharp bound for the estimation error of order $O((V\log(n) /n)^{1/(2-\beta)})$ is established. In particular, with the pairwise least squares loss, we derive a nearly optimal bound of the excess generalization error which achieves the minimax lower bound up to a logrithmic term when the true predictor satisfies some smoothness regularities.
    

