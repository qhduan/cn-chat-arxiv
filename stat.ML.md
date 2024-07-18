# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [On Stronger Computational Separations Between Multimodal and Unimodal Machine Learning](https://arxiv.org/abs/2404.02254) | 提出了更强的平均情况计算分离，对于“典型”情况下的学习任务实例，单模态学习在计算上是困难的，但多模态学习却很容易。 |
| [^2] | [Minimum-Norm Interpolation Under Covariate Shift](https://arxiv.org/abs/2404.00522) | 本研究首次证明了在转移学习设置下，良性过拟合线性插值器的非渐近超额风险界，并提出了一种新的分类方法。 |
| [^3] | [Universal Lower Bounds and Optimal Rates: Achieving Minimax Clustering Error in Sub-Exponential Mixture Models](https://arxiv.org/abs/2402.15432) | 本文在混合模型中建立了一个通用下界，通过Chernoff散度来表达，将其拓展到具有次指数尾部的混合模型，并证明了迭代算法在这些混合模型中实现了最佳误差率 |
| [^4] | [Nearest Neighbour Score Estimators for Diffusion Generative Models](https://arxiv.org/abs/2402.08018) | 本论文提出了一种新颖的最近邻评分函数估计器，通过利用训练集中的多个样本大大降低了估计器的方差，可用于训练一致性模型和扩散模型，提高收敛速度、样本质量，并为进一步的研究提供了新的可能性。 |
| [^5] | [Non-Vacuous Generalization Bounds for Large Language Models](https://arxiv.org/abs/2312.17173) | 这项研究提供了首个针对预训练大语言模型的非平凡泛化界限，表明语言模型能够发现适用于未见数据的规律性。建立了有效的压缩界限，证明较大的模型具有更好的泛化界限并更易压缩。 |
| [^6] | [Learning in Deep Factor Graphs with Gaussian Belief Propagation](https://arxiv.org/abs/2311.14649) | 提出了一种在高斯因子图中进行学习的方法，利用置信传播解决训练和预测问题，支持分布式和异步训练，可扩展至深度网络，提供持续学习的自然方式，并展示了在视频去噪和图像分类任务中的优势。 |
| [^7] | [Bayesian Optimization with Noise-Free Observations: Improved Regret Bounds via Random Exploration.](http://arxiv.org/abs/2401.17037) | 该论文研究了基于无噪声观测的贝叶斯优化问题，提出了一种基于散乱数据逼近的新算法，并引入随机探索步骤以实现接近最优填充距离的速率衰减。该算法在实现的易用性和累积遗憾边界的性能上超过了传统的GP-UCB算法，并在多个示例中优于其他贝叶斯优化策略。 |
| [^8] | [IW-GAE: Importance weighted group accuracy estimation for improved calibration and model selection in unsupervised domain adaptation.](http://arxiv.org/abs/2310.10611) | 本文提出了一种名为IW-GAE的方法，通过开发一种新颖的加权群准确率估计器来解决非监督领域适应中的校准和模型选择问题。经过理论分析和实验验证，该方法在处理数据分布偏移方面表现出有效性。 |
| [^9] | [Relating Implicit Bias and Adversarial Attacks through Intrinsic Dimension.](http://arxiv.org/abs/2305.15203) | 本文通过研究神经网络的隐性偏差，着眼于其中涉及的傅里叶频率与图像分类和对抗性攻击之间的关系。研究提出了一种新方法，可以发现这些频率之间的非线性相关性。 |
| [^10] | [Mode Connectivity in Auction Design.](http://arxiv.org/abs/2305.11005) | 该论文研究了拍卖设计领域的一个基本问题，即最优拍卖设计。在研究中，作者证明了神经网络在一定条件下可以通过简单的分段线性路径连接不同的局部最优解，并取得了成功。 |
| [^11] | [Distributed Gradient Descent for Functional Learning.](http://arxiv.org/abs/2305.07408) | 该论文提出了一种针对函数数据的分布式梯度下降函数学习算法，在再生核希尔伯特空间框架下通过积分算子方法得到了该算法的理论理解，并取得了不饱和边界的置信度最优学习率。 |
| [^12] | [Towards Lower Bounds on the Depth of ReLU Neural Networks.](http://arxiv.org/abs/2105.14835) | 该研究运用数学和优化理论方法，就 ReLU 神经网络的深度下界做了探究，有助于更好地理解这种网络所能表示的函数类的性质。此外，该研究还肯定了一项旧的分段线性函数猜想。 |

# 详细

[^1]: 关于多模态与单模态机器学习之间更强的计算分离

    On Stronger Computational Separations Between Multimodal and Unimodal Machine Learning

    [https://arxiv.org/abs/2404.02254](https://arxiv.org/abs/2404.02254)

    提出了更强的平均情况计算分离，对于“典型”情况下的学习任务实例，单模态学习在计算上是困难的，但多模态学习却很容易。

    

    在多模态机器学习中，将多种数据模态（例如文本和图像）结合起来以促进更好的机器学习模型的学习，这仍然适用于相应的单模态任务（例如文本生成）。最近，多模态机器学习取得了巨大的经验成功（例如GPT-4）。受到为这种经验成功开发理论基础的动机，Lu（NeurIPS '23，ALT '24）提出了一种多模态学习理论，并考虑了多模态和单模态学习的理论模型之间可能的分离。特别是Lu（ALT '24）展示了一种计算分离，这对学习任务的最坏情况实例是相关的。

    arXiv:2404.02254v1 Announce Type: cross  Abstract: In multimodal machine learning, multiple modalities of data (e.g., text and images) are combined to facilitate the learning of a better machine learning model, which remains applicable to a corresponding unimodal task (e.g., text generation). Recently, multimodal machine learning has enjoyed huge empirical success (e.g. GPT-4). Motivated to develop theoretical justification for this empirical success, Lu (NeurIPS '23, ALT '24) introduces a theory of multimodal learning, and considers possible separations between theoretical models of multimodal and unimodal learning. In particular, Lu (ALT '24) shows a computational separation, which is relevant to worst-case instances of the learning task.   In this paper, we give a stronger average-case computational separation, where for "typical" instances of the learning task, unimodal learning is computationally hard, but multimodal learning is easy. We then question how "organic" the average-cas
    
[^2]: 最小范数插值在协变量转移下的应用

    Minimum-Norm Interpolation Under Covariate Shift

    [https://arxiv.org/abs/2404.00522](https://arxiv.org/abs/2404.00522)

    本研究首次证明了在转移学习设置下，良性过拟合线性插值器的非渐近超额风险界，并提出了一种新的分类方法。

    

    转移学习是现实世界机器学习部署的关键组成部分，并在过参数化神经网络的实验研究中得到广泛研究。然而，即使在线性回归的最简单设置中，在对转移学习的理论理解仍存在显著差距。在高维线性回归的分布研究中，已经发现了一种被称为“良性过拟合”现象的现象，即线性插值器会对噪声训练标签过拟合，但仍然能很好地泛化。这种行为发生在源协方差矩阵和输入数据维度上的特定条件下。因此，自然而然地想知道这样的高维线性模型在转移学习下如何行为。我们证明了在转移学习设置中良性过拟合线性插值器的第一个非渐近超额风险界。通过我们的分析，我们提出了一个对转移学习中的\textit {b进行分类}}的方法

    arXiv:2404.00522v1 Announce Type: new  Abstract: Transfer learning is a critical part of real-world machine learning deployments and has been extensively studied in experimental works with overparameterized neural networks. However, even in the simplest setting of linear regression a notable gap still exists in the theoretical understanding of transfer learning. In-distribution research on high-dimensional linear regression has led to the identification of a phenomenon known as \textit{benign overfitting}, in which linear interpolators overfit to noisy training labels and yet still generalize well. This behavior occurs under specific conditions on the source covariance matrix and input data dimension. Therefore, it is natural to wonder how such high-dimensional linear models behave under transfer learning. We prove the first non-asymptotic excess risk bounds for benignly-overfit linear interpolators in the transfer learning setting. From our analysis, we propose a taxonomy of \textit{b
    
[^3]: 在次指数混合模型中实现极小化聚类误差：通用下界和最佳速率

    Universal Lower Bounds and Optimal Rates: Achieving Minimax Clustering Error in Sub-Exponential Mixture Models

    [https://arxiv.org/abs/2402.15432](https://arxiv.org/abs/2402.15432)

    本文在混合模型中建立了一个通用下界，通过Chernoff散度来表达，将其拓展到具有次指数尾部的混合模型，并证明了迭代算法在这些混合模型中实现了最佳误差率

    

    聚类是无监督机器学习中的一个关键挑战，通常通过混合模型的视角来研究。在高斯和次高斯混合模型中恢复聚类标签的最佳误差率涉及到特定的信噪比。简单的迭代算法，如Lloyd算法，可以达到这个最佳误差率。在本文中，我们首先为任何混合模型中的误差率建立了一个通用下界，通过Chernoff散度来表达，这是一个比信噪比更通用的模型信息度量。然后我们证明了迭代算法在混合模型中实现了这个下界，特别强调了具有拉普拉斯分布误差的位置-尺度混合。此外，针对更适合由泊松或负二项混合模型建模的数据集，我们研究了其分布属于指数族的混合模型。

    arXiv:2402.15432v1 Announce Type: cross  Abstract: Clustering is a pivotal challenge in unsupervised machine learning and is often investigated through the lens of mixture models. The optimal error rate for recovering cluster labels in Gaussian and sub-Gaussian mixture models involves ad hoc signal-to-noise ratios. Simple iterative algorithms, such as Lloyd's algorithm, attain this optimal error rate. In this paper, we first establish a universal lower bound for the error rate in clustering any mixture model, expressed through a Chernoff divergence, a more versatile measure of model information than signal-to-noise ratios. We then demonstrate that iterative algorithms attain this lower bound in mixture models with sub-exponential tails, notably emphasizing location-scale mixtures featuring Laplace-distributed errors. Additionally, for datasets better modelled by Poisson or Negative Binomial mixtures, we study mixture models whose distributions belong to an exponential family. In such m
    
[^4]: 扩散生成模型的最近邻评分估计器

    Nearest Neighbour Score Estimators for Diffusion Generative Models

    [https://arxiv.org/abs/2402.08018](https://arxiv.org/abs/2402.08018)

    本论文提出了一种新颖的最近邻评分函数估计器，通过利用训练集中的多个样本大大降低了估计器的方差，可用于训练一致性模型和扩散模型，提高收敛速度、样本质量，并为进一步的研究提供了新的可能性。

    

    评分函数估计是训练和采样扩散生成模型的基础。尽管如此，最常用的估计器要么是有偏的神经网络逼近，要么是基于条件评分的高方差蒙特卡洛估计器。我们引入了一种创新的最近邻评分函数估计器，利用训练集中的多个样本大大降低了估计器的方差。我们在两个引人注目的应用中利用了低方差估计器。在使用我们的估计器进行训练一致性模型时，我们报告了收敛速度和样本质量显著提高。在扩散模型中，我们展示了我们的估计器可以替代学习网络进行概率流ODE积分，为未来研究开辟了有前景的新方向。

    Score function estimation is the cornerstone of both training and sampling from diffusion generative models. Despite this fact, the most commonly used estimators are either biased neural network approximations or high variance Monte Carlo estimators based on the conditional score. We introduce a novel nearest neighbour score function estimator which utilizes multiple samples from the training set to dramatically decrease estimator variance. We leverage our low variance estimator in two compelling applications. Training consistency models with our estimator, we report a significant increase in both convergence speed and sample quality. In diffusion models, we show that our estimator can replace a learned network for probability-flow ODE integration, opening promising new avenues of future research.
    
[^5]: 大语言模型的非平凡泛化界限

    Non-Vacuous Generalization Bounds for Large Language Models

    [https://arxiv.org/abs/2312.17173](https://arxiv.org/abs/2312.17173)

    这项研究提供了首个针对预训练大语言模型的非平凡泛化界限，表明语言模型能够发现适用于未见数据的规律性。建立了有效的压缩界限，证明较大的模型具有更好的泛化界限并更易压缩。

    

    现代语言模型可以包含数十亿个参数，这引发了一个问题，它们是否可以在训练数据之外进行泛化，或者只是重复它们的训练语料库。我们提供了首个针对预训练大语言模型（LLM）的非平凡泛化界限，表明语言模型能够发现适用于未见数据的规律性。具体而言，我们使用预测平滑导出了一个适用于无界对数似然损失的压缩界限，并且我们扩展了该界限以处理子采样，加速对大规模数据集的界限计算。为了实现非平凡泛化界限所需的极端压缩程度，我们设计了SubLoRA，这是一种低维非线性参数化方法。使用这种方法，我们发现较大的模型具有更好的泛化界限，并且比较小的模型更易压缩。

    Modern language models can contain billions of parameters, raising the question of whether they can generalize beyond the training data or simply regurgitate their training corpora. We provide the first non-vacuous generalization bounds for pretrained large language models (LLMs), indicating that language models are capable of discovering regularities that generalize to unseen data. In particular, we derive a compression bound that is valid for the unbounded log-likelihood loss using prediction smoothing, and we extend the bound to handle subsampling, accelerating bound computation on massive datasets. To achieve the extreme level of compression required for non-vacuous generalization bounds, we devise SubLoRA, a low-dimensional non-linear parameterization. Using this approach, we find that larger models have better generalization bounds and are more compressible than smaller models.
    
[^6]: 在具有高斯置信传播的深度因子图中进行学习

    Learning in Deep Factor Graphs with Gaussian Belief Propagation

    [https://arxiv.org/abs/2311.14649](https://arxiv.org/abs/2311.14649)

    提出了一种在高斯因子图中进行学习的方法，利用置信传播解决训练和预测问题，支持分布式和异步训练，可扩展至深度网络，提供持续学习的自然方式，并展示了在视频去噪和图像分类任务中的优势。

    

    我们提出了一种在高斯因子图中进行学习的方法。我们将所有相关数量（输入、输出、参数、潜变量）视为图模型中的随机变量，并将训练和预测都视为具有不同观察节点的推理问题。我们的实验表明，这些问题可以通过置信传播（BP）有效地解决，其更新本质上是本地的，为分布式和异步训练提供了令人兴奋的机会。我们的方法可以扩展到深层网络，并提供了一种自然的持续学习方式：使用当前任务的BP估计参数边际作为下一个任务的参数先验。在视频去噪任务上，我们展示了可学习参数相对于传统因子图方法的优势，同时展示了深度因子图在持续图像分类方面的鼓舞人心的性能。

    arXiv:2311.14649v2 Announce Type: replace  Abstract: We propose an approach to do learning in Gaussian factor graphs. We treat all relevant quantities (inputs, outputs, parameters, latents) as random variables in a graphical model, and view both training and prediction as inference problems with different observed nodes. Our experiments show that these problems can be efficiently solved with belief propagation (BP), whose updates are inherently local, presenting exciting opportunities for distributed and asynchronous training. Our approach can be scaled to deep networks and provides a natural means to do continual learning: use the BP-estimated parameter marginals of the current task as parameter priors for the next. On a video denoising task we demonstrate the benefit of learnable parameters over a classical factor graph approach and we show encouraging performance of deep factor graphs for continual image classification.
    
[^7]: 基于无噪声观测的贝叶斯优化：通过随机探索改善遗憾边界

    Bayesian Optimization with Noise-Free Observations: Improved Regret Bounds via Random Exploration. (arXiv:2401.17037v1 [cs.LG])

    [http://arxiv.org/abs/2401.17037](http://arxiv.org/abs/2401.17037)

    该论文研究了基于无噪声观测的贝叶斯优化问题，提出了一种基于散乱数据逼近的新算法，并引入随机探索步骤以实现接近最优填充距离的速率衰减。该算法在实现的易用性和累积遗憾边界的性能上超过了传统的GP-UCB算法，并在多个示例中优于其他贝叶斯优化策略。

    

    本文研究了基于无噪声观测的贝叶斯优化。我们引入了新的基于散乱数据逼近的算法，并通过随机探索步骤确保查询点的填充距离以接近最优的速率衰减。我们的算法保留了经典的GP-UCB算法的易实现性，并满足了几乎与arXiv:2002.05096中的猜想相匹配的累积遗憾边界，从而解决了COLT的一个开放问题。此外，新算法在几个示例中优于GP-UCB和其他流行的贝叶斯优化策略。

    This paper studies Bayesian optimization with noise-free observations. We introduce new algorithms rooted in scattered data approximation that rely on a random exploration step to ensure that the fill-distance of query points decays at a near-optimal rate. Our algorithms retain the ease of implementation of the classical GP-UCB algorithm and satisfy cumulative regret bounds that nearly match those conjectured in arXiv:2002.05096, hence solving a COLT open problem. Furthermore, the new algorithms outperform GP-UCB and other popular Bayesian optimization strategies in several examples.
    
[^8]: IW-GAE: 用于提高非监督领域适应中的校准和模型选择的加权群准确率估计

    IW-GAE: Importance weighted group accuracy estimation for improved calibration and model selection in unsupervised domain adaptation. (arXiv:2310.10611v1 [cs.LG])

    [http://arxiv.org/abs/2310.10611](http://arxiv.org/abs/2310.10611)

    本文提出了一种名为IW-GAE的方法，通过开发一种新颖的加权群准确率估计器来解决非监督领域适应中的校准和模型选择问题。经过理论分析和实验验证，该方法在处理数据分布偏移方面表现出有效性。

    

    计算模型在测试样本上的准确率并从中推断其置信度是机器学习中的一个核心问题，与不确定性表示、模型选择和探索等重要应用密切相关。虽然这些连接在独立同分布设置中已经被广泛研究，但数据分布的偏移给传统方法带来了重大挑战。因此，在非监督领域适应问题中，模型校准和模型选择仍然具有挑战性，这是一种在没有标签的情况下在数据分布发生偏移的领域中表现良好的场景。在本文中，我们通过开发一种新颖的加权群准确率估计器来解决由于数据分布的偏移而带来的困难。具体而言，我们制定了一个优化问题，找到导致在数据分布偏移的领域中准确估计群准确率的重要权重，并进行了理论分析。大量实验结果表明了群准确率估计在模型上的有效性。

    Reasoning about a model's accuracy on a test sample from its confidence is a central problem in machine learning, being connected to important applications such as uncertainty representation, model selection, and exploration. While these connections have been well-studied in the i.i.d. settings, distribution shifts pose significant challenges to the traditional methods. Therefore, model calibration and model selection remain challenging in the unsupervised domain adaptation problem--a scenario where the goal is to perform well in a distribution shifted domain without labels. In this work, we tackle difficulties coming from distribution shifts by developing a novel importance weighted group accuracy estimator. Specifically, we formulate an optimization problem for finding an importance weight that leads to an accurate group accuracy estimation in the distribution shifted domain with theoretical analyses. Extensive experiments show the effectiveness of group accuracy estimation on model 
    
[^9]: 通过内在维度将隐性偏见和对抗性攻击相关联

    Relating Implicit Bias and Adversarial Attacks through Intrinsic Dimension. (arXiv:2305.15203v1 [cs.LG])

    [http://arxiv.org/abs/2305.15203](http://arxiv.org/abs/2305.15203)

    本文通过研究神经网络的隐性偏差，着眼于其中涉及的傅里叶频率与图像分类和对抗性攻击之间的关系。研究提出了一种新方法，可以发现这些频率之间的非线性相关性。

    

    尽管神经网络在分类方面表现出色，但众所周知它们易受对抗性攻击的影响。这些攻击是针对模型的输入数据进行的小干扰，旨在欺骗模型。自然而然的问题是，模型的结构、设置或属性与攻击的性质之间可能存在潜在联系。在本文中，我们旨在通过关注神经网络的隐性偏差来解决这个问题，这指的是其固有倾向于支持特定模式或结果。具体而言，我们研究了隐性偏差的一个方面，其中包括进行准确图像分类所需的基本傅里叶频率。我们进行测试以评估这些频率与成功攻击所需的频率之间的统计关系。为了深入探讨这种关系，我们提出了一种新的方法，可以揭示坐标集之间的非线性相关性，在我们的情况下，这些坐标集就是前述的傅里叶频率。

    Despite their impressive performance in classification, neural networks are known to be vulnerable to adversarial attacks. These attacks are small perturbations of the input data designed to fool the model. Naturally, a question arises regarding the potential connection between the architecture, settings, or properties of the model and the nature of the attack. In this work, we aim to shed light on this problem by focusing on the implicit bias of the neural network, which refers to its inherent inclination to favor specific patterns or outcomes. Specifically, we investigate one aspect of the implicit bias, which involves the essential Fourier frequencies required for accurate image classification. We conduct tests to assess the statistical relationship between these frequencies and those necessary for a successful attack. To delve into this relationship, we propose a new method that can uncover non-linear correlations between sets of coordinates, which, in our case, are the aforementio
    
[^10]: 拍卖设计中的模式连通性

    Mode Connectivity in Auction Design. (arXiv:2305.11005v1 [cs.GT])

    [http://arxiv.org/abs/2305.11005](http://arxiv.org/abs/2305.11005)

    该论文研究了拍卖设计领域的一个基本问题，即最优拍卖设计。在研究中，作者证明了神经网络在一定条件下可以通过简单的分段线性路径连接不同的局部最优解，并取得了成功。

    

    最优拍卖设计是算法博弈论中的一个基本问题，即使在非常简单的情况下，这个问题也很难。最近不同的经济学可微分理论表明，神经网络可以有效地学习已知的最优拍卖机制，发现有趣的新机制。为了理论上证明它们的实证成功，我们聚焦于第一个这样的网络，RochetNet，并研究所谓的仿射极大化拍卖的广义版本。我们证明它们满足模式连通性，即局部最优解通过一个简单的分段线性路径连接，路径上的每个解都几乎和两个局部最优解之一一样好。模式连通性最近被证明是神经网络用于预测问题的一个有趣的经验和理论的属性。我们的结果是对可微分经济学领域中神经网络用于解决非线性设计问题的第一个这样的分析。

    Optimal auction design is a fundamental problem in algorithmic game theory. This problem is notoriously difficult already in very simple settings. Recent work in differentiable economics showed that neural networks can efficiently learn known optimal auction mechanisms and discover interesting new ones. In an attempt to theoretically justify their empirical success, we focus on one of the first such networks, RochetNet, and a generalized version for affine maximizer auctions. We prove that they satisfy mode connectivity, i.e., locally optimal solutions are connected by a simple, piecewise linear path such that every solution on the path is almost as good as one of the two local optima. Mode connectivity has been recently investigated as an intriguing empirical and theoretically justifiable property of neural networks used for prediction problems. Our results give the first such analysis in the context of differentiable economics, where neural networks are used directly for solving non-
    
[^11]: 面向函数学习的分布式梯度下降算法

    Distributed Gradient Descent for Functional Learning. (arXiv:2305.07408v1 [stat.ML])

    [http://arxiv.org/abs/2305.07408](http://arxiv.org/abs/2305.07408)

    该论文提出了一种针对函数数据的分布式梯度下降函数学习算法，在再生核希尔伯特空间框架下通过积分算子方法得到了该算法的理论理解，并取得了不饱和边界的置信度最优学习率。

    

    近年来，不同类型的分布式学习方案因其在处理大规模数据信息方面的巨大优势而受到越来越多的关注。针对最近从函数数据分析中产生的大数据挑战，我们在再生核希尔伯特空间框架下提出了一种新颖的分布式梯度下降函数学习（DGDFL）算法，用于处理来自众多本地机器（处理器）的函数数据。基于积分算子方法，我们提供了DGDFL算法在文献中的许多方面的第一个理论理解。在理解DGDFL的过程中，首先，我们提出并全面研究了基于数据的渐进式下降函数学习（GDFL）算法与单机模型相关联。在温和的条件下，得到了DGDFL的置信度最优学习率，避免了先前在正则性索引上遭受的饱和边界。

    In recent years, different types of distributed learning schemes have received increasing attention for their strong advantages in handling large-scale data information. In the information era, to face the big data challenges which stem from functional data analysis very recently, we propose a novel distributed gradient descent functional learning (DGDFL) algorithm to tackle functional data across numerous local machines (processors) in the framework of reproducing kernel Hilbert space. Based on integral operator approaches, we provide the first theoretical understanding of the DGDFL algorithm in many different aspects in the literature. On the way of understanding DGDFL, firstly, a data-based gradient descent functional learning (GDFL) algorithm associated with a single-machine model is proposed and comprehensively studied. Under mild conditions, confidence-based optimal learning rates of DGDFL are obtained without the saturation boundary on the regularity index suffered in previous w
    
[^12]: 关于 ReLU 神经网络深度下界的探究

    Towards Lower Bounds on the Depth of ReLU Neural Networks. (arXiv:2105.14835v4 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2105.14835](http://arxiv.org/abs/2105.14835)

    该研究运用数学和优化理论方法，就 ReLU 神经网络的深度下界做了探究，有助于更好地理解这种网络所能表示的函数类的性质。此外，该研究还肯定了一项旧的分段线性函数猜想。

    

    我们运用混合整数优化、多面体理论和热带几何学等技术，为理解具有 ReLU 激活和给定结构的神经网络所能表示的函数类做出了更好的贡献。尽管普适逼近定理认为单层隐藏层就足以学习任何函数，但我们提供了一个数学的对称性，并详细探讨了添加更多层（无大小限制）时是否严格增加了可表示函数的类。作为研究副产品，我们肯定了 Wang 和 Sun（2005）有关分段线性函数的一个旧猜想。我们还给出了表示具有对数深度函数所需的神经网络大小上界。

    We contribute to a better understanding of the class of functions that can be represented by a neural network with ReLU activations and a given architecture. Using techniques from mixed-integer optimization, polyhedral theory, and tropical geometry, we provide a mathematical counterbalance to the universal approximation theorems which suggest that a single hidden layer is sufficient for learning any function. In particular, we investigate whether the class of exactly representable functions strictly increases by adding more layers (with no restrictions on size). As a by-product of our investigations, we settle an old conjecture about piecewise linear functions by Wang and Sun (2005) in the affirmative. We also present upper bounds on the sizes of neural networks required to represent functions with logarithmic depth.
    

