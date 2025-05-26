# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Federated Learning on Transcriptomic Data: Model Quality and Performance Trade-Offs](https://arxiv.org/abs/2402.14527) | 本文研究了基因组学或转录组数据上的联邦学习，使用 TensorFlow Federated 和 Flower 框架进行实验，以培训疾病预后和细胞类型分类模型。 |
| [^2] | [Statistical Test for Generated Hypotheses by Diffusion Models](https://arxiv.org/abs/2402.11789) | 本研究提出了一种统计检验方法，通过选择性推断框架，在考虑生成图像是由训练的扩散模型产生的条件下，量化医学图像诊断结果的可靠性。 |
| [^3] | [Preconditioners for the Stochastic Training of Implicit Neural Representations](https://arxiv.org/abs/2402.08784) | 本论文提出了一种新的随机训练方法，通过使用曲率感知对角预处理器，在不损失准确性的情况下加速了隐式神经表示的训练过程，适用于多个信号模态。 |
| [^4] | [Understanding Practical Membership Privacy of Deep Learning](https://arxiv.org/abs/2402.06674) | 该论文利用最先进的成员推理攻击方法系统地测试了细调大型图像分类模型的实际隐私漏洞，并发现数据集中每个类别的示例数量以及训练结束时的大梯度与成员推理攻击的漏洞之间存在关联。 |
| [^5] | [Function Aligned Regression: A Method Explicitly Learns Functional Derivatives from Data](https://arxiv.org/abs/2402.06104) | 该论文提出了一种名为FAR的方法，通过捕捉函数导数来更好、更高效地拟合底层真实函数。在合成数据集和八个真实世界任务中证明了该方法的有效性。 |
| [^6] | [RandCom: Random Communication Skipping Method for Decentralized Stochastic Optimization.](http://arxiv.org/abs/2310.07983) | RandCom是一种去中心化的随机通信跳跃方法，能够在分布式优化中通过概率性本地更新减少通信开销，并在不同的设置中实现线性加速。 |
| [^7] | [QFT: Quantized Full-parameter Tuning of LLMs with Affordable Resources.](http://arxiv.org/abs/2310.07147) | 我们提出了一种新的QFT框架，可以对LLMs进行内存高效的全参数微调，而不损害性能。 |
| [^8] | [Comprehensive Assessment of the Performance of Deep Learning Classifiers Reveals a Surprising Lack of Robustness.](http://arxiv.org/abs/2308.04137) | 通过综合评估深度学习分类器的性能，发现它们缺乏稳定性和可靠性，并建议采用广泛的数据类型和统一的评估指标进行性能基准测试。 |
| [^9] | [Learning Formal Specifications from Membership and Preference Queries.](http://arxiv.org/abs/2307.10434) | 该论文提出了一种新的框架，通过请求成员标签和成对偏好来扩展主动规范学习，提高学习形式规范的灵活性。在两个不同领域的实验中，结果表明通过学习成员和偏好的组合可以稳定和方便地识别规范。 |
| [^10] | [MDI+: A Flexible Random Forest-Based Feature Importance Framework.](http://arxiv.org/abs/2307.01932) | MDI+是一种灵活的基于随机森林的特征重要性框架，通过替换线性回归模型和度量，利用正则化的广义线性模型和更适合数据结构的度量来推广MDI。此外，MDI+还引入了其他特征来减轻决策树对加法或平滑模型的已知偏差。 |
| [^11] | [Improving Multi-task Learning via Seeking Task-based Flat Regions.](http://arxiv.org/abs/2211.13723) | 通过寻找基于任务的平坦区域，可以改进多任务学习并提高模型性能，但需要正确使用正则化技术以避免次优解。 |

# 详细

[^1]: 基因组学或转录组数据上的联邦学习：模型质量和性能权衡

    Federated Learning on Transcriptomic Data: Model Quality and Performance Trade-Offs

    [https://arxiv.org/abs/2402.14527](https://arxiv.org/abs/2402.14527)

    本文研究了基因组学或转录组数据上的联邦学习，使用 TensorFlow Federated 和 Flower 框架进行实验，以培训疾病预后和细胞类型分类模型。

    

    在大规模基因组学或转录组数据上进行机器学习对许多新颖的健康应用至关重要。例如，精准医学可以根据个体生物标志物、细胞和分子状态等个体信息来量身定制医学治疗。然而，所需数据敏感、庞大、异质，并且通常分布在无法使用专门的机器学习硬件的地点。由于隐私和监管原因，在可信任的第三方处聚合所有数据也存在问题。联邦学习是这一困境的一个有前途的解决方案，因为它实现了在不交换原始数据的情况下进行分散、协作的机器学习。在本文中，我们使用联邦学习框架 TensorFlow Federated 和 Flower 进行比较实验。我们的测试案例是培训疾病预后和细胞类型分类模型。我们使用分布式转录组对模型进行训练

    arXiv:2402.14527v1 Announce Type: new  Abstract: Machine learning on large-scale genomic or transcriptomic data is important for many novel health applications. For example, precision medicine tailors medical treatments to patients on the basis of individual biomarkers, cellular and molecular states, etc. However, the data required is sensitive, voluminous, heterogeneous, and typically distributed across locations where dedicated machine learning hardware is not available. Due to privacy and regulatory reasons, it is also problematic to aggregate all data at a trusted third party.Federated learning is a promising solution to this dilemma, because it enables decentralized, collaborative machine learning without exchanging raw data. In this paper, we perform comparative experiments with the federated learning frameworks TensorFlow Federated and Flower. Our test case is the training of disease prognosis and cell type classification models. We train the models with distributed transcriptom
    
[^2]: 通过扩散模型生成的假设的统计检验

    Statistical Test for Generated Hypotheses by Diffusion Models

    [https://arxiv.org/abs/2402.11789](https://arxiv.org/abs/2402.11789)

    本研究提出了一种统计检验方法，通过选择性推断框架，在考虑生成图像是由训练的扩散模型产生的条件下，量化医学图像诊断结果的可靠性。

    

    AI的增强性能加速了其融入科学研究。特别是，利用生成式AI创建科学假设是很有前途的，并且正在越来越多地应用于各个领域。然而，当使用AI生成的假设进行关键决策（如医学诊断）时，验证它们的可靠性至关重要。在本研究中，我们考虑使用扩散模型生成的图像进行医学诊断任务，并提出了一种统计检验来量化其可靠性。所提出的统计检验的基本思想是使用选择性推断框架，我们考虑在生成的图像是由经过训练的扩散模型产生的这一事实条件下的统计检验。利用所提出的方法，医学图像诊断结果的统计可靠性可以以p值的形式量化，从而实现在控制错误率的情况下进行决策。

    arXiv:2402.11789v1 Announce Type: cross  Abstract: The enhanced performance of AI has accelerated its integration into scientific research. In particular, the use of generative AI to create scientific hypotheses is promising and is increasingly being applied across various fields. However, when employing AI-generated hypotheses for critical decisions, such as medical diagnoses, verifying their reliability is crucial. In this study, we consider a medical diagnostic task using generated images by diffusion models, and propose a statistical test to quantify its reliability. The basic idea behind the proposed statistical test is to employ a selective inference framework, where we consider a statistical test conditional on the fact that the generated images are produced by a trained diffusion model. Using the proposed method, the statistical reliability of medical image diagnostic results can be quantified in the form of a p-value, allowing for decision-making with a controlled error rate. 
    
[^3]: 隐式神经表示的随机训练的预处理器

    Preconditioners for the Stochastic Training of Implicit Neural Representations

    [https://arxiv.org/abs/2402.08784](https://arxiv.org/abs/2402.08784)

    本论文提出了一种新的随机训练方法，通过使用曲率感知对角预处理器，在不损失准确性的情况下加速了隐式神经表示的训练过程，适用于多个信号模态。

    

    隐式神经表示已经成为一种强大的技术，用于将复杂连续多维信号编码为神经网络，从而实现计算机视觉、机器人学和几何学等广泛应用。尽管Adam由于其随机的高效性而被广泛应用于训练中，但其训练时间往往较长。为了解决这个问题，我们探索了在加速训练的同时不损失准确性的替代优化技术。传统的二阶优化器如L-BFGS在随机环境中效果不佳，因此不适用于大规模数据集。相反，我们提出了使用曲率感知对角预处理器进行随机训练，展示了它们在图像、形状重建和神经辐射场等各种信号模态中的有效性。

    arXiv:2402.08784v1 Announce Type: cross Abstract: Implicit neural representations have emerged as a powerful technique for encoding complex continuous multidimensional signals as neural networks, enabling a wide range of applications in computer vision, robotics, and geometry. While Adam is commonly used for training due to its stochastic proficiency, it entails lengthy training durations. To address this, we explore alternative optimization techniques for accelerated training without sacrificing accuracy. Traditional second-order optimizers like L-BFGS are suboptimal in stochastic settings, making them unsuitable for large-scale data sets. Instead, we propose stochastic training using curvature-aware diagonal preconditioners, showcasing their effectiveness across various signal modalities such as images, shape reconstruction, and Neural Radiance Fields (NeRF).
    
[^4]: 理解深度学习的实际成员隐私

    Understanding Practical Membership Privacy of Deep Learning

    [https://arxiv.org/abs/2402.06674](https://arxiv.org/abs/2402.06674)

    该论文利用最先进的成员推理攻击方法系统地测试了细调大型图像分类模型的实际隐私漏洞，并发现数据集中每个类别的示例数量以及训练结束时的大梯度与成员推理攻击的漏洞之间存在关联。

    

    我们应用最先进的成员推理攻击（MIA）来系统地测试细调大型图像分类模型的实际隐私漏洞。我们的重点是理解使数据集和样本容易受到成员推理攻击的特性。在数据集特性方面，我们发现数据中每个类别的示例数量与成员推理攻击的漏洞之间存在强烈的幂律依赖关系，这是以攻击的真阳性率（在低假阳性率下测量）来衡量的。对于个别样本而言，在训练结束时产生的大梯度与成员推理攻击的漏洞之间存在很强的相关性。

    We apply a state-of-the-art membership inference attack (MIA) to systematically test the practical privacy vulnerability of fine-tuning large image classification models.We focus on understanding the properties of data sets and samples that make them vulnerable to membership inference. In terms of data set properties, we find a strong power law dependence between the number of examples per class in the data and the MIA vulnerability, as measured by true positive rate of the attack at a low false positive rate. For an individual sample, large gradients at the end of training are strongly correlated with MIA vulnerability.
    
[^5]: 功能对齐回归：一种从数据中明确学习函数导数的方法

    Function Aligned Regression: A Method Explicitly Learns Functional Derivatives from Data

    [https://arxiv.org/abs/2402.06104](https://arxiv.org/abs/2402.06104)

    该论文提出了一种名为FAR的方法，通过捕捉函数导数来更好、更高效地拟合底层真实函数。在合成数据集和八个真实世界任务中证明了该方法的有效性。

    

    回归是机器学习中的一个基本任务，在过去几十年中引起了广泛关注。传统的回归方法主要通过使用损失函数来将模型预测与每个个体数据样本的真实值对齐，然而，我们发现这种方法可能导致在不同样本之间关系的预测不够优化。近期的研究工作引入了标签相似性信息来改进回归方法，但在完全捕捉底层真实函数的复杂性方面仍存在明显的差距。在本文中，我们提出了FAR（功能对齐回归）作为一种更好、更高效的解决方案，通过捕捉函数导数来拟合底层真实函数。我们在两个合成数据集和六个领域的八个大规模真实世界任务中验证了该方法的有效性。

    Regression is a fundamental task in machine learning that has garnered extensive attention over the past decades. The conventional approach for regression involves employing loss functions that primarily concentrate on aligning model prediction with the ground truth for each individual data sample, which, as we show, can result in sub-optimal prediction of the relationships between the different samples. Recent research endeavors have introduced novel perspectives by incorporating label similarity information to regression. However, a notable gap persists in these approaches when it comes to fully capturing the intricacies of the underlying ground truth function. In this work, we propose FAR (Function Aligned Regression) as a arguably better and more efficient solution to fit the underlying function of ground truth by capturing functional derivatives. We demonstrate the effectiveness of the proposed method practically on 2 synthetic datasets and on 8 extensive real-world tasks from 6 b
    
[^6]: RandCom：去中心化随机通信跳跃方法用于分布式随机优化

    RandCom: Random Communication Skipping Method for Decentralized Stochastic Optimization. (arXiv:2310.07983v1 [cs.LG])

    [http://arxiv.org/abs/2310.07983](http://arxiv.org/abs/2310.07983)

    RandCom是一种去中心化的随机通信跳跃方法，能够在分布式优化中通过概率性本地更新减少通信开销，并在不同的设置中实现线性加速。

    

    具有随机通信跳过的分布式优化方法因其在加速通信复杂性方面具有的优势而受到越来越多的关注。然而，现有的研究主要集中在强凸确定性设置的集中式通信协议上。在本研究中，我们提出了一种名为RandCom的分布式优化方法，它采用了概率性的本地更新。我们分析了RandCom在随机非凸、凸和强凸设置中的性能，并证明了它能够通过通信概率来渐近地减少通信开销。此外，我们证明当节点数量增加时，RandCom能够实现线性加速。在随机强凸设置中，我们进一步证明了RandCom可以通过独立于网络的步长实现线性加速。此外，我们将RandCom应用于联邦学习，并提供了关于实现线性加速的潜力的积极结果。

    Distributed optimization methods with random communication skips are gaining increasing attention due to their proven benefits in accelerating communication complexity. Nevertheless, existing research mainly focuses on centralized communication protocols for strongly convex deterministic settings. In this work, we provide a decentralized optimization method called RandCom, which incorporates probabilistic local updates. We analyze the performance of RandCom in stochastic non-convex, convex, and strongly convex settings and demonstrate its ability to asymptotically reduce communication overhead by the probability of communication. Additionally, we prove that RandCom achieves linear speedup as the number of nodes increases. In stochastic strongly convex settings, we further prove that RandCom can achieve linear speedup with network-independent stepsizes. Moreover, we apply RandCom to federated learning and provide positive results concerning the potential for achieving linear speedup and
    
[^7]: QFT: 使用可承担资源对LLMs进行量化全参数调整

    QFT: Quantized Full-parameter Tuning of LLMs with Affordable Resources. (arXiv:2310.07147v1 [cs.CL])

    [http://arxiv.org/abs/2310.07147](http://arxiv.org/abs/2310.07147)

    我们提出了一种新的QFT框架，可以对LLMs进行内存高效的全参数微调，而不损害性能。

    

    大型语言模型（LLMs）在自然语言处理任务中展示出了显著的影响。对这些预训练模型进行微调可以进一步提高性能，但由于其巨大的资源需求，这一过程具有挑战性。为此，现有的努力都集中在参数高效的微调上，不幸的是，它们没有充分发挥全参数微调的潜力。在这项工作中，我们提出了QFT，一种新颖的用于LLMs的量化全参数调整框架，可以在不损害性能的情况下实现高效的内存微调。我们的框架包括两个新颖的思想：（i）我们采用高效的Lion优化器，仅跟踪动量并具有每个参数一致的更新幅度，这对于稳健的量化是一种内在优势；（ii）我们将所有模型状态进行量化，并以整数值存储，同时提供梯度流和参数更新的方法。

    Large Language Models (LLMs) have showcased remarkable impacts across a wide spectrum of natural language processing tasks. Fine-tuning these pre-trained models on downstream datasets provides further significant performance gains, but this process has been challenging due to its extraordinary resource requirements. To this end, existing efforts focus on parameter-efficient fine-tuning, which, unfortunately, fail to capitalize on the powerful potential of full-parameter fine-tuning. In this work, we propose QFT, a novel Quantized Full-parameter Tuning framework for LLMs that enables memory-efficient fine-tuning without harming performance. Our framework incorporates two novel ideas: (i) we adopt the efficient Lion optimizer, which only keeps track of the momentum and has consistent update magnitudes for each parameter, an inherent advantage for robust quantization; and (ii) we quantize all model states and store them as integer values, and present a gradient flow and parameter update s
    
[^8]: 深度学习分类器性能的综合评估揭示出惊人的缺乏稳定性

    Comprehensive Assessment of the Performance of Deep Learning Classifiers Reveals a Surprising Lack of Robustness. (arXiv:2308.04137v1 [cs.LG])

    [http://arxiv.org/abs/2308.04137](http://arxiv.org/abs/2308.04137)

    通过综合评估深度学习分类器的性能，发现它们缺乏稳定性和可靠性，并建议采用广泛的数据类型和统一的评估指标进行性能基准测试。

    

    可靠而稳健的评估方法是开发本身稳健可靠的机器学习模型的必要第一步。然而，目前用于评估分类器的常规评估协议在综合评估性能方面存在不足，因为它们往往依赖于有限类型的测试数据，忽视其他类型的数据。例如，使用标准测试数据无法评估分类器对于未经训练的类别样本的预测。另一方面，使用包含未知类别样本的数据进行测试无法评估分类器对于已知类别标签的预测能力。本文提倡使用各种不同类型的数据进行性能基准测试，并使用一种可应用于所有这些数据类型的单一指标，以产生一致的性能评估结果。通过这样的基准测试发现，目前的深度神经网络，包括使用认为是全面的方法进行训练的网络，也存在缺乏稳定性的问题。

    Reliable and robust evaluation methods are a necessary first step towards developing machine learning models that are themselves robust and reliable. Unfortunately, current evaluation protocols typically used to assess classifiers fail to comprehensively evaluate performance as they tend to rely on limited types of test data, and ignore others. For example, using the standard test data fails to evaluate the predictions made by the classifier to samples from classes it was not trained on. On the other hand, testing with data containing samples from unknown classes fails to evaluate how well the classifier can predict the labels for known classes. This article advocates bench-marking performance using a wide range of different types of data and using a single metric that can be applied to all such data types to produce a consistent evaluation of performance. Using such a benchmark it is found that current deep neural networks, including those trained with methods that are believed to pro
    
[^9]: 从成员和偏好查询中学习形式规范

    Learning Formal Specifications from Membership and Preference Queries. (arXiv:2307.10434v1 [cs.FL])

    [http://arxiv.org/abs/2307.10434](http://arxiv.org/abs/2307.10434)

    该论文提出了一种新的框架，通过请求成员标签和成对偏好来扩展主动规范学习，提高学习形式规范的灵活性。在两个不同领域的实验中，结果表明通过学习成员和偏好的组合可以稳定和方便地识别规范。

    

    主动学习是一种研究广泛的学习形式规范的方法，例如自动机。在这项工作中，我们通过提出一种新颖的框架，将主动规范学习扩展到请求组合成员标签和成对偏好（对成员标签的一种流行替代方式）。成对偏好和成员标签的组合允许更灵活的主动规范学习方法，它先前仅依赖成员标签。我们将我们的框架应用于两个不同的领域，证明了我们方法的广泛性。我们的结果表明，从两种模式学习可以通过成员和偏好来稳健和方便地识别规范。

    Active learning is a well-studied approach to learning formal specifications, such as automata. In this work, we extend active specification learning by proposing a novel framework that strategically requests a combination of membership labels and pair-wise preferences, a popular alternative to membership labels. The combination of pair-wise preferences and membership labels allows for a more flexible approach to active specification learning, which previously relied on membership labels only. We instantiate our framework in two different domains, demonstrating the generality of our approach. Our results suggest that learning from both modalities allows us to robustly and conveniently identify specifications via membership and preferences.
    
[^10]: MDI+:一种灵活的基于随机森林的特征重要性框架

    MDI+: A Flexible Random Forest-Based Feature Importance Framework. (arXiv:2307.01932v1 [stat.ME])

    [http://arxiv.org/abs/2307.01932](http://arxiv.org/abs/2307.01932)

    MDI+是一种灵活的基于随机森林的特征重要性框架，通过替换线性回归模型和度量，利用正则化的广义线性模型和更适合数据结构的度量来推广MDI。此外，MDI+还引入了其他特征来减轻决策树对加法或平滑模型的已知偏差。

    

    以不纯度减少的平均值(MDI)是随机森林(RF)中一种流行的特征重要性评估方法。我们展示了在RF中每个树的特征$X_k$的MDI等价于响应变量在决策树集合上的线性回归的未归一化$R^2$值。我们利用这种解释提出了一种灵活的特征重要性框架MDI+，MDI+通过允许分析人员将线性回归模型和$R^2$度量替换为正则化的广义线性模型(GLM)和更适合给定数据结构的度量来推广MDI。此外，MDI+还引入了其他特征来减轻决策树对加法或平滑模型的已知偏差。我们进一步提供了关于如何基于可预测性、可计算性和稳定性框架选择适当的GLM和度量的指导，以进行真实数据科学研究。大量基于数据的模拟结果显示，MDI+在性能上显著优于传统的MDI。

    Mean decrease in impurity (MDI) is a popular feature importance measure for random forests (RFs). We show that the MDI for a feature $X_k$ in each tree in an RF is equivalent to the unnormalized $R^2$ value in a linear regression of the response on the collection of decision stumps that split on $X_k$. We use this interpretation to propose a flexible feature importance framework called MDI+. Specifically, MDI+ generalizes MDI by allowing the analyst to replace the linear regression model and $R^2$ metric with regularized generalized linear models (GLMs) and metrics better suited for the given data structure. Moreover, MDI+ incorporates additional features to mitigate known biases of decision trees against additive or smooth models. We further provide guidance on how practitioners can choose an appropriate GLM and metric based upon the Predictability, Computability, Stability framework for veridical data science. Extensive data-inspired simulations show that MDI+ significantly outperfor
    
[^11]: 通过寻找基于任务的平坦区域来改进多任务学习

    Improving Multi-task Learning via Seeking Task-based Flat Regions. (arXiv:2211.13723v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2211.13723](http://arxiv.org/abs/2211.13723)

    通过寻找基于任务的平坦区域，可以改进多任务学习并提高模型性能，但需要正确使用正则化技术以避免次优解。

    

    多任务学习（MTL）是一种广泛使用且强大的学习范式，用于训练深度神经网络，可以通过单个骨干学习多个目标。与单独训练任务相比，MTL显着降低了计算成本，提高了数据效率，并通过利用任务之间的知识来潜在地提高模型性能。因此，它已经被应用于各种应用领域，从计算机视觉到自然语言处理和语音识别。其中，MTL的一个新兴研究方向集中在操纵任务梯度以推导出对所有任务有益的最终梯度下降方向。尽管在许多基准测试上取得了令人印象深刻的结果，但是在实际问题上直接应用这些方法而不使用适当的正则化技术可能会导致次优解。特别是，标准训练在训练数据上最小化经验损失，很容易遭受过拟合问题。

    Multi-Task Learning (MTL) is a widely-used and powerful learning paradigm for training deep neural networks that allows learning more than one objective by a single backbone. Compared to training tasks separately, MTL significantly reduces computational costs, improves data efficiency, and potentially enhances model performance by leveraging knowledge across tasks. Hence, it has been adopted in a variety of applications, ranging from computer vision to natural language processing and speech recognition. Among them, there is an emerging line of work in MTL that focuses on manipulating the task gradient to derive an ultimate gradient descent direction to benefit all tasks. Despite achieving impressive results on many benchmarks, directly applying these approaches without using appropriate regularization techniques might lead to suboptimal solutions on real-world problems. In particular, standard training that minimizes the empirical loss on the training data can easily suffer from overfi
    

