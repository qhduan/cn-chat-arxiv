# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Temporal graph models fail to capture global temporal dynamics.](http://arxiv.org/abs/2309.15730) | 时间图模型无法捕捉全局时间动态，我们提出了一种"最近流行节点"的基线方法，在时间图基准的中等和大规模数据集上胜过其他方法。我们提出了两个基于Wasserstein距离的度量来量化全局动态。我们展示了标准的负采样评估方法在具有强烈时间动态的数据集上可能不适用，我们还展示了简单的负采样方法可能导致模型退化。我们提出了改进的负采样方案，并证明了它们的有效性。我们还将其与无负采样的非对比训练模型进行了比较。 |
| [^2] | [Cold & Warm Net: Addressing Cold-Start Users in Recommender Systems.](http://arxiv.org/abs/2309.15646) | 本文提出了一种名为冷启动和热启动网络的方法(Cold & Warm Net)，用于解决推荐系统中的冷启动用户问题。该方法利用专家模型分别建模冷启动和热启动用户，并引入门控网络和动态知识蒸馏来提高用户表示的学习效果。通过选择与用户行为高度相关的特征，并建立偏差网络来显式建模用户行为偏差。实验证实了该方法的有效性。 |
| [^3] | [Identifiability Matters: Revealing the Hidden Recoverable Condition in Unbiased Learning to Rank.](http://arxiv.org/abs/2309.15560) | 研究揭示在无偏学习排名中，当点击数据不能完全拟合时，无法恢复真实相关性，导致排名性能显著降低，提出了可识别性图模型作为解决方案。 |
| [^4] | [Automatic Feature Fairness in Recommendation via Adversaries.](http://arxiv.org/abs/2309.15418) | 通过对手训练实现推荐系统中的特征公平性，提高整体准确性和泛化能力 |
| [^5] | [Frequency and cardinality recovery from sketched data: a novel approach bridging Bayesian and frequentist views.](http://arxiv.org/abs/2309.15408) | 该论文研究了如何仅使用压缩表示来恢复大规模数据集中符号的频率，并引入了新的估计方法，将贝叶斯和频率论观点结合起来，提供了更好的解决方案。此外，还扩展了该方法以解决基数恢复问题。 |
| [^6] | [A Content-Driven Micro-Video Recommendation Dataset at Scale.](http://arxiv.org/abs/2309.15379) | 该论文介绍了一个名为"MicroLens"的大规模微视频推荐数据集，包括十亿个用户-项目交互行为和各种原始模态信息，为研究人员开发内容驱动的微视频推荐系统提供了基准。 |
| [^7] | [LD4MRec: Simplifying and Powering Diffusion Model for Multimedia Recommendation.](http://arxiv.org/abs/2309.15363) | LD4MRec是一种简化和加强多媒体推荐的扩散模型，解决了行为数据噪声对推荐性能的负面影响、经典扩散模型计算量过大以及现有反向过程不适用于离散行为数据的挑战。 |

# 详细

[^1]: 时间图模型无法捕捉全局时间动态

    Temporal graph models fail to capture global temporal dynamics. (arXiv:2309.15730v1 [cs.IR])

    [http://arxiv.org/abs/2309.15730](http://arxiv.org/abs/2309.15730)

    时间图模型无法捕捉全局时间动态，我们提出了一种"最近流行节点"的基线方法，在时间图基准的中等和大规模数据集上胜过其他方法。我们提出了两个基于Wasserstein距离的度量来量化全局动态。我们展示了标准的负采样评估方法在具有强烈时间动态的数据集上可能不适用，我们还展示了简单的负采样方法可能导致模型退化。我们提出了改进的负采样方案，并证明了它们的有效性。我们还将其与无负采样的非对比训练模型进行了比较。

    

    在动态链接属性预测的背景下，我们分析了最近发布的时间图基准，并提出了一种"最近流行节点"的基线方法，在时间图基准的中等和大规模数据集上胜过其他方法。我们提出了基于Wasserstein距离的两个度量，可以量化数据集的短期和长期全局动态的强度。通过分析我们出乎意料的强大基线，我们展示了标准的负采样评估方法在具有强烈时间动态的数据集上可能不适用。我们还展示了简单的负采样方法在训练过程中可能导致模型退化，导致无法对时间图网络进行排序的预测完全饱和。我们提出了改进的负采样方案用于训练和评估，并证明了它们的有效性。我们还将其与无负采样的非对比训练模型进行了比较。我们的结果表明...

    A recently released Temporal Graph Benchmark is analyzed in the context of Dynamic Link Property Prediction. We outline our observations and propose a trivial optimization-free baseline of "recently popular nodes" outperforming other methods on all medium and large-size datasets in the Temporal Graph Benchmark. We propose two measures based on Wasserstein distance which can quantify the strength of short-term and long-term global dynamics of datasets. By analyzing our unexpectedly strong baseline, we show how standard negative sampling evaluation can be unsuitable for datasets with strong temporal dynamics. We also show how simple negative-sampling can lead to model degeneration during training, resulting in impossible to rank, fully saturated predictions of temporal graph networks. We propose improved negative sampling schemes for both training and evaluation and prove their usefulness. We conduct a comparison with a model trained non-contrastively without negative sampling. Our resul
    
[^2]: 冷启动和热启动网络：解决推荐系统中的冷启动用户问题

    Cold & Warm Net: Addressing Cold-Start Users in Recommender Systems. (arXiv:2309.15646v1 [cs.IR])

    [http://arxiv.org/abs/2309.15646](http://arxiv.org/abs/2309.15646)

    本文提出了一种名为冷启动和热启动网络的方法(Cold & Warm Net)，用于解决推荐系统中的冷启动用户问题。该方法利用专家模型分别建模冷启动和热启动用户，并引入门控网络和动态知识蒸馏来提高用户表示的学习效果。通过选择与用户行为高度相关的特征，并建立偏差网络来显式建模用户行为偏差。实验证实了该方法的有效性。

    

    冷启动推荐是推荐系统面临的重大挑战之一。本文主要关注用户冷启动问题。最近，使用侧信息或元学习的方法被用来建模冷启动用户。然而，将这些方法应用于工业级推荐系统仍然存在困难。目前对于匹配阶段中的用户冷启动问题的研究还不多。本文提出了基于专家模型的冷启动和热启动用户建模方法：Cold & Warm Net。通过引入门控网络来结合两个专家的结果。此外，还引入了动态知识蒸馏作为一个教师选择器，帮助专家更好地学习用户表示。通过全面的互信息选择与用户行为高度相关的特征，用于显式建模用户行为偏差的偏差网络。最后，在公共数据集上对Cold & Warm Net进行评估。

    Cold-start recommendation is one of the major challenges faced by recommender systems (RS). Herein, we focus on the user cold-start problem. Recently, methods utilizing side information or meta-learning have been used to model cold-start users. However, it is difficult to deploy these methods to industrial RS. There has not been much research that pays attention to the user cold-start problem in the matching stage. In this paper, we propose Cold & Warm Net based on expert models who are responsible for modeling cold-start and warm-up users respectively. A gate network is applied to incorporate the results from two experts. Furthermore, dynamic knowledge distillation acting as a teacher selector is introduced to assist experts in better learning user representation. With comprehensive mutual information, features highly relevant to user behavior are selected for the bias net which explicitly models user behavior bias. Finally, we evaluate our Cold & Warm Net on public datasets in compar
    
[^3]: 识别性很重要：揭示无偏学习排名中隐藏的可恢复条件

    Identifiability Matters: Revealing the Hidden Recoverable Condition in Unbiased Learning to Rank. (arXiv:2309.15560v1 [cs.IR])

    [http://arxiv.org/abs/2309.15560](http://arxiv.org/abs/2309.15560)

    研究揭示在无偏学习排名中，当点击数据不能完全拟合时，无法恢复真实相关性，导致排名性能显著降低，提出了可识别性图模型作为解决方案。

    

    无偏学习排名(Unbiased Learning to Rank, ULTR)在从有偏点击日志训练无偏排名模型的现代系统中被广泛应用。关键在于明确地建模用户行为的生成过程，并基于检验假设对点击数据进行拟合。先前的研究经验性地发现只要点击完全拟合，大多数情况下可以恢复出真实潜在相关性。然而，我们证明并非总是能够实现这一点，从而导致排名性能显著降低。在本工作中，我们旨在回答真实相关性是否能够从点击数据恢复出来的问题，这是ULTR领域的一个基本问题。我们首先将一个排名模型定义为可识别的，如果它可以恢复出真实相关性，最多只有一个缩放变换，这对于成对排名目标来说已足够。然后，我们探讨了一个等价的可识别条件，可以新颖地表达为一个图连通性测试问题：当且仅当一个图（即可识别性图）连通时，该排名模型是可识别的。

    The application of Unbiased Learning to Rank (ULTR) is widespread in modern systems for training unbiased ranking models from biased click logs. The key is to explicitly model a generation process for user behavior and fit click data based on examination hypothesis. Previous research found empirically that the true latent relevance can be recovered in most cases as long as the clicks are perfectly fitted. However, we demonstrate that this is not always achievable, resulting in a significant reduction in ranking performance. In this work, we aim to answer if or when the true relevance can be recovered from click data, which is a foundation issue for ULTR field. We first define a ranking model as identifiable if it can recover the true relevance up to a scaling transformation, which is enough for pairwise ranking objective. Then we explore an equivalent condition for identifiability that can be novely expressed as a graph connectivity test problem: if and only if a graph (namely identifi
    
[^4]: 通过对手对推荐系统中的特征公平性的自动处理

    Automatic Feature Fairness in Recommendation via Adversaries. (arXiv:2309.15418v1 [cs.IR])

    [http://arxiv.org/abs/2309.15418](http://arxiv.org/abs/2309.15418)

    通过对手训练实现推荐系统中的特征公平性，提高整体准确性和泛化能力

    

    公平性是推荐系统中广泛讨论的一个主题，但其实践实现在定义敏感特征的同时保持推荐准确性方面面临挑战。我们提出将特征公平性作为实现各个由不同特征组合定义的多样群体之间的公平待遇的基础。通过平衡特征的泛化能力，可以提高整体准确性。我们通过对手训练引入了无偏特征学习，使用对手扰动增强特征表示。对手改进了模型对少数特征的泛化能力。我们根据特征偏差的两种形式自动适应对手：特征值的频率和组合多样性。这使我们能够动态调整扰动强度和对手训练权重。更强的扰动适用于组合变化少的特征值，以改善泛化能力，而对于低频特征，较高的权重可以解决...

    Fairness is a widely discussed topic in recommender systems, but its practical implementation faces challenges in defining sensitive features while maintaining recommendation accuracy. We propose feature fairness as the foundation to achieve equitable treatment across diverse groups defined by various feature combinations. This improves overall accuracy through balanced feature generalizability. We introduce unbiased feature learning through adversarial training, using adversarial perturbation to enhance feature representation. The adversaries improve model generalization for under-represented features. We adapt adversaries automatically based on two forms of feature biases: frequency and combination variety of feature values. This allows us to dynamically adjust perturbation strengths and adversarial training weights. Stronger perturbations are applied to feature values with fewer combination varieties to improve generalization, while higher weights for low-frequency features address 
    
[^5]: 从压缩数据中恢复频率和基数：一种将贝叶斯和频率论观点连接起来的新方法

    Frequency and cardinality recovery from sketched data: a novel approach bridging Bayesian and frequentist views. (arXiv:2309.15408v1 [stat.ME])

    [http://arxiv.org/abs/2309.15408](http://arxiv.org/abs/2309.15408)

    该论文研究了如何仅使用压缩表示来恢复大规模数据集中符号的频率，并引入了新的估计方法，将贝叶斯和频率论观点结合起来，提供了更好的解决方案。此外，还扩展了该方法以解决基数恢复问题。

    

    我们研究如何仅使用通过随机哈希获得的对数据进行压缩表示或草图来恢复大规模离散数据集中符号的频率。这是一个在计算机科学中的经典问题，有各种算法可用，如计数最小草图。然而，这些算法通常假设数据是固定的，处理随机采样数据时估计过于保守且可能不准确。在本文中，我们将草图数据视为未知分布的随机样本，然后引入改进现有方法的新估计器。我们的方法结合了贝叶斯非参数和经典（频率论）观点，解决了它们独特的限制，提供了一个有原则且实用的解决方案。此外，我们扩展了我们的方法以解决相关但不同的基数恢复问题，该问题涉及估计数据集中不同对象的总数。

    We study how to recover the frequency of a symbol in a large discrete data set, using only a compressed representation, or sketch, of those data obtained via random hashing. This is a classical problem in computer science, with various algorithms available, such as the count-min sketch. However, these algorithms often assume that the data are fixed, leading to overly conservative and potentially inaccurate estimates when dealing with randomly sampled data. In this paper, we consider the sketched data as a random sample from an unknown distribution, and then we introduce novel estimators that improve upon existing approaches. Our method combines Bayesian nonparametric and classical (frequentist) perspectives, addressing their unique limitations to provide a principled and practical solution. Additionally, we extend our method to address the related but distinct problem of cardinality recovery, which consists of estimating the total number of distinct objects in the data set. We validate
    
[^6]: 一个规模庞大的内容驱动的微视频推荐数据集

    A Content-Driven Micro-Video Recommendation Dataset at Scale. (arXiv:2309.15379v1 [cs.IR])

    [http://arxiv.org/abs/2309.15379](http://arxiv.org/abs/2309.15379)

    该论文介绍了一个名为"MicroLens"的大规模微视频推荐数据集，包括十亿个用户-项目交互行为和各种原始模态信息，为研究人员开发内容驱动的微视频推荐系统提供了基准。

    

    微视频最近变得非常受欢迎，引发了对微视频推荐的重要研究，对娱乐、广告和电子商务行业具有重要影响。然而，缺乏大规模的公共微视频数据集为开发有效的推荐系统提供了挑战。为了解决这个问题，我们介绍了一个非常庞大的微视频推荐数据集，名为"MicroLens"，包括十亿个用户-项目交互行为，3400万个用户和100万个微视频。该数据集还包含有关视频的各种原始模态信息，包括标题、封面图像、音频和完整视频。MicroLens作为内容驱动的微视频推荐的基准，使研究人员能够利用各种视频信息的模态进行推荐，而不仅仅依赖于项目ID或从预训练网络中提取的现成视频特征。

    Micro-videos have recently gained immense popularity, sparking critical research in micro-video recommendation with significant implications for the entertainment, advertising, and e-commerce industries. However, the lack of large-scale public micro-video datasets poses a major challenge for developing effective recommender systems. To address this challenge, we introduce a very large micro-video recommendation dataset, named "MicroLens", consisting of one billion user-item interaction behaviors, 34 million users, and one million micro-videos. This dataset also contains various raw modality information about videos, including titles, cover images, audio, and full-length videos. MicroLens serves as a benchmark for content-driven micro-video recommendation, enabling researchers to utilize various modalities of video information for recommendation, rather than relying solely on item IDs or off-the-shelf video features extracted from a pre-trained network. Our benchmarking of multiple reco
    
[^7]: LD4MRec:简化和加强多媒体推荐的扩散模型

    LD4MRec: Simplifying and Powering Diffusion Model for Multimedia Recommendation. (arXiv:2309.15363v1 [cs.IR])

    [http://arxiv.org/abs/2309.15363](http://arxiv.org/abs/2309.15363)

    LD4MRec是一种简化和加强多媒体推荐的扩散模型，解决了行为数据噪声对推荐性能的负面影响、经典扩散模型计算量过大以及现有反向过程不适用于离散行为数据的挑战。

    

    多媒体推荐旨在根据历史行为数据和项目的多模态信息预测用户的未来行为。然而，行为数据中的噪声，产生于与不感兴趣的项目的非预期用户交互，对推荐性能产生不利影响。最近，扩散模型实现了高质量的信息生成，其中反向过程根据受损状态迭代地推断未来信息。它满足了在嘈杂条件下的预测任务需求，并激发了对其在预测用户行为方面的应用的探索。然而，还需要解决几个挑战：1）经典扩散模型需要过多的计算，这不符合推荐系统的效率要求。2）现有的反向过程主要设计用于连续型数据，而行为信息是离散型的。因此，需要有效的方法来生成离散行为。

    Multimedia recommendation aims to predict users' future behaviors based on historical behavioral data and item's multimodal information. However, noise inherent in behavioral data, arising from unintended user interactions with uninteresting items, detrimentally impacts recommendation performance. Recently, diffusion models have achieved high-quality information generation, in which the reverse process iteratively infers future information based on the corrupted state. It meets the need of predictive tasks under noisy conditions, and inspires exploring their application to predicting user behaviors. Nonetheless, several challenges must be addressed: 1) Classical diffusion models require excessive computation, which does not meet the efficiency requirements of recommendation systems. 2) Existing reverse processes are mainly designed for continuous data, whereas behavioral information is discrete in nature. Therefore, an effective method is needed for the generation of discrete behaviora
    

