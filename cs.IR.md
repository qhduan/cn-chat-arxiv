# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Toward a Better Understanding of Loss Functions for Collaborative Filtering.](http://arxiv.org/abs/2308.06091) | 现有研究已经表明，通过改进对齐和均匀性设计的损失函数可以实现显著的性能提升。本文提出了一种新的损失函数，称为MAWU，它考虑了数据集的独特模式。 |
| [^2] | [Toward Transparent Sequence Models with Model-Based Tree Markov Model.](http://arxiv.org/abs/2307.15367) | 本研究引入了基于模型的树马尔可夫模型（MOB-HSMM），用于解决复杂黑盒机器学习模型应用于序列数据时的可解释性问题。通过从深度神经网络中蒸馏的知识，实现了提高预测性能的同时提供清晰解释的目标。实验结果表明通过将LSTM学习到的顺序模式转移到MOB树中，可以进一步提高MOB树的性能，并利用MOB-HSMM将MOB树与隐马尔可夫模型（HSMM）整合，实现了潜在和可解释的序列的发现。 |
| [^3] | [Towards Populating Generalizable Engineering Design Knowledge.](http://arxiv.org/abs/2307.06985) | 这项研究提出了一种从专利文件中提取工程设计知识的方法，通过构建知识图来填充通用设计知识，并与现有方法进行了比较。 |
| [^4] | [Unified Off-Policy Learning to Rank: a Reinforcement Learning Perspective.](http://arxiv.org/abs/2306.07528) | 本文提出了点击模型不可知的统一非同策略学习排序（CUOLR）方法，通过离线强化学习（RL）直接学习最优排名，可以轻松地应用于各种点击模型。 |
| [^5] | [Computational Technologies for Fashion Recommendation: A Survey.](http://arxiv.org/abs/2306.03395) | 本文全面综述了时尚推荐的计算技术研究。研究者从宏观层面介绍了该领域，分析了其特点和不同之处，并将时尚推荐任务分为几个子任务进行重点讨论。 |
| [^6] | [PUNR: Pre-training with User Behavior Modeling for News Recommendation.](http://arxiv.org/abs/2304.12633) | 本论文提出了一种无监督的预训练方法，它可以通过两个任务实现有效的用户行为建模，以提高新闻推荐系统的准确性和性能表现。 |
| [^7] | [A Diffusion model for POI recommendation.](http://arxiv.org/abs/2304.07041) | 本文提出了一种基于扩散算法采样用户空间偏好的POI推荐模型，解决了现有方法只基于用户先前访问位置聚合的缺点，适用于推荐新颖区域的POI。 |
| [^8] | [DiffuRec: A Diffusion Model for Sequential Recommendation.](http://arxiv.org/abs/2304.00686) | 本文提出了一种名为DiffuRec 的扩散模型，其将物品表示为分布而不是固定向量，从而更好地反映了用户的多种偏好和物品的多个方面，并成功地应用于顺序推荐。 |
| [^9] | [Pattern reconstruction with restricted Boltzmann machines.](http://arxiv.org/abs/2205.07087) | 该论文研究了限制玻尔兹曼机在模式重构中的能力，发现隐藏层先验分布的尾部行为对于恢复随机模式的效果有关键影响。 |
| [^10] | [Geodesic Multi-Modal Mixup for Robust Fine-Tuning.](http://arxiv.org/abs/2203.03897) | 本文研究了CLIP模型的多模态嵌入质量，并发现其统一性和对齐性不足，限制了嵌入的传递性和鲁棒性。为了解决这个问题，我们提出了一种新的鲁棒微调方法，通过高度几何多模型混合生成难负样本，并对模型进行微调。 |

# 详细

[^1]: 对协同过滤丢失函数的更好理解

    Toward a Better Understanding of Loss Functions for Collaborative Filtering. (arXiv:2308.06091v1 [cs.IR])

    [http://arxiv.org/abs/2308.06091](http://arxiv.org/abs/2308.06091)

    现有研究已经表明，通过改进对齐和均匀性设计的损失函数可以实现显著的性能提升。本文提出了一种新的损失函数，称为MAWU，它考虑了数据集的独特模式。

    

    协同过滤（CF）是现代推荐系统中的关键技术。CF模型的学习过程通常由三个组件组成：交互编码器、损失函数和负采样。尽管许多现有研究已经提出了各种CF模型来设计复杂的交互编码器，但最近的工作表明，简单地重新制定损失函数可以实现显著的性能提升。本文深入分析了现有损失函数之间的关系。我们的数学分析揭示了先前的损失函数可以解释为对齐和均匀性函数：（i）对齐匹配用户和物品表示，（ii）均匀性分散用户和物品分布。受到这个分析的启示，我们提出了一种改进对齐和均匀性设计的损失函数，考虑到数据集的独特模式，称为Margin-aware Alignment and Weighted Uniformity（MAWU）。MAWU的关键创新是

    Collaborative filtering (CF) is a pivotal technique in modern recommender systems. The learning process of CF models typically consists of three components: interaction encoder, loss function, and negative sampling. Although many existing studies have proposed various CF models to design sophisticated interaction encoders, recent work shows that simply reformulating the loss functions can achieve significant performance gains. This paper delves into analyzing the relationship among existing loss functions. Our mathematical analysis reveals that the previous loss functions can be interpreted as alignment and uniformity functions: (i) the alignment matches user and item representations, and (ii) the uniformity disperses user and item distributions. Inspired by this analysis, we propose a novel loss function that improves the design of alignment and uniformity considering the unique patterns of datasets called Margin-aware Alignment and Weighted Uniformity (MAWU). The key novelty of MAWU 
    
[^2]: 通过基于模型的树马尔可夫模型，实现透明的序列模型

    Toward Transparent Sequence Models with Model-Based Tree Markov Model. (arXiv:2307.15367v1 [cs.LG])

    [http://arxiv.org/abs/2307.15367](http://arxiv.org/abs/2307.15367)

    本研究引入了基于模型的树马尔可夫模型（MOB-HSMM），用于解决复杂黑盒机器学习模型应用于序列数据时的可解释性问题。通过从深度神经网络中蒸馏的知识，实现了提高预测性能的同时提供清晰解释的目标。实验结果表明通过将LSTM学习到的顺序模式转移到MOB树中，可以进一步提高MOB树的性能，并利用MOB-HSMM将MOB树与隐马尔可夫模型（HSMM）整合，实现了潜在和可解释的序列的发现。

    

    本研究解决了应用于序列数据的复杂、黑盒机器学习模型的可解释性问题。我们引入了基于模型的树隐马尔可夫模型（MOB-HSMM），这是一个固有可解释性的模型，旨在检测高死亡风险事件，并发现与死亡风险相关的隐藏模式。该模型利用从深度神经网络（DNN）中蒸馏的知识，提高预测性能的同时提供清晰的解释。我们的实验结果表明，通过使用LSTM学习顺序模式，进而将其转移给MOB树，可以提高基于模型的树（MOB树）的性能。将MOB树与基于模型的隐马尔可夫模型（HSMM）集成在MOB-HSMM中，可以使用可用信息揭示潜在的和可解释的序列。

    In this study, we address the interpretability issue in complex, black-box Machine Learning models applied to sequence data. We introduce the Model-Based tree Hidden Semi-Markov Model (MOB-HSMM), an inherently interpretable model aimed at detecting high mortality risk events and discovering hidden patterns associated with the mortality risk in Intensive Care Units (ICU). This model leverages knowledge distilled from Deep Neural Networks (DNN) to enhance predictive performance while offering clear explanations. Our experimental results indicate the improved performance of Model-Based trees (MOB trees) via employing LSTM for learning sequential patterns, which are then transferred to MOB trees. Integrating MOB trees with the Hidden Semi-Markov Model (HSMM) in the MOB-HSMM enables uncovering potential and explainable sequences using available information.
    
[^3]: 迈向填充通用工程设计知识的方法

    Towards Populating Generalizable Engineering Design Knowledge. (arXiv:2307.06985v1 [cs.CL])

    [http://arxiv.org/abs/2307.06985](http://arxiv.org/abs/2307.06985)

    这项研究提出了一种从专利文件中提取工程设计知识的方法，通过构建知识图来填充通用设计知识，并与现有方法进行了比较。

    

    为了填充通用工程设计知识，我们提出了一种从专利文件中提取head entity :: relationship :: tail entity形式事实的方法。这些事实可以在专利文件内部和跨文件之间组合形成知识图，用作表示和存储设计知识的方案。现有的工程设计文献中的方法通常利用一组预定义的关系来填充统计近似而非事实的三元组。在我们的方法中，我们训练一个标记器来识别句子中的实体和关系。在确定了一对实体后，我们训练另一个标记器来识别特定表示这对实体之间关系的关系标记。为了训练这些标记器，我们手动构建了一个包含44,227个句子和相应事实的数据集。我们还将该方法的性能与通常推荐的方法进行了比较，其中我们预.

    Aiming to populate generalizable engineering design knowledge, we propose a method to extract facts of the form head entity :: relationship :: tail entity from sentences found in patent documents. These facts could be combined within and across patent documents to form knowledge graphs that serve as schemes for representing as well as storing design knowledge. Existing methods in engineering design literature often utilise a set of predefined relationships to populate triples that are statistical approximations rather than facts. In our method, we train a tagger to identify both entities and relationships from a sentence. Given a pair of entities thus identified, we train another tagger to identify the relationship tokens that specifically denote the relationship between the pair. For training these taggers, we manually construct a dataset of 44,227 sentences and corresponding facts. We also compare the performance of the method against typically recommended approaches, wherein, we pre
    
[^4]: 统一的非同策略学习排序：强化学习视角

    Unified Off-Policy Learning to Rank: a Reinforcement Learning Perspective. (arXiv:2306.07528v1 [cs.LG])

    [http://arxiv.org/abs/2306.07528](http://arxiv.org/abs/2306.07528)

    本文提出了点击模型不可知的统一非同策略学习排序（CUOLR）方法，通过离线强化学习（RL）直接学习最优排名，可以轻松地应用于各种点击模型。

    

    非同策略学习排序（LTR）旨在通过已部署的记录策略收集的数据优化排名器。然而，现有的非同策略学习排序方法经常对用户如何生成点击数据即点击模型进行假设，因此需要根据不同的点击模型专门调整他们的方法。在本文中，我们将排名过程在一般随机点击模型下统一为马尔可夫决策过程（MDP），通过离线强化学习（RL），可以直接学习最优排名。在此基础上，我们利用离线RL技术进行非同策略LTR，并提出点击模型不可知的统一非同策略学习排序（CUOLR）方法，该方法可以轻松地应用于各种点击模型。通过对MDP的专门制定，我们证明了离线RL算法可以适应各种点击模型，而无需复杂的去偏倚技术和先验知识。在各种大规模数据集上的实验结果都证明了我们方法的有效性。

    Off-policy Learning to Rank (LTR) aims to optimize a ranker from data collected by a deployed logging policy. However, existing off-policy learning to rank methods often make strong assumptions about how users generate the click data, i.e., the click model, and hence need to tailor their methods specifically under different click models. In this paper, we unified the ranking process under general stochastic click models as a Markov Decision Process (MDP), and the optimal ranking could be learned with offline reinforcement learning (RL) directly. Building upon this, we leverage offline RL techniques for off-policy LTR and propose the Click Model-Agnostic Unified Off-policy Learning to Rank (CUOLR) method, which could be easily applied to a wide range of click models. Through a dedicated formulation of the MDP, we show that offline RL algorithms can adapt to various click models without complex debiasing techniques and prior knowledge of the model. Results on various large-scale datasets
    
[^5]: 时尚推荐的计算技术：一项综述

    Computational Technologies for Fashion Recommendation: A Survey. (arXiv:2306.03395v1 [cs.MM])

    [http://arxiv.org/abs/2306.03395](http://arxiv.org/abs/2306.03395)

    本文全面综述了时尚推荐的计算技术研究。研究者从宏观层面介绍了该领域，分析了其特点和不同之处，并将时尚推荐任务分为几个子任务进行重点讨论。

    

    时尚推荐是计算时尚研究中的一个关键研究领域，近年来在计算机视觉、多媒体和信息检索社区引起了相当大的兴趣。由于应用的巨大需求，文献中提出并探索了各种时尚推荐任务，如个性化时尚产品推荐、相互补充（搭配）推荐和搭配推荐。持续的研究注意和进展促使我们回顾并深入了解这个领域。本文从技术角度全面回顾了近年来关于时尚推荐的研究工作。我们首先从宏观层面介绍了时尚推荐，并分析了它的特点和与一般推荐任务的区别。然后，我们将不同的时尚推荐任务清晰地归类为几个子任务，并从问题解决的角度重点关注每个子任务。

    Fashion recommendation is a key research field in computational fashion research and has attracted considerable interest in the computer vision, multimedia, and information retrieval communities in recent years. Due to the great demand for applications, various fashion recommendation tasks, such as personalized fashion product recommendation, complementary (mix-and-match) recommendation, and outfit recommendation, have been posed and explored in the literature. The continuing research attention and advances impel us to look back and in-depth into the field for a better understanding. In this paper, we comprehensively review recent research efforts on fashion recommendation from a technological perspective. We first introduce fashion recommendation at a macro level and analyse its characteristics and differences with general recommendation tasks. We then clearly categorize different fashion recommendation efforts into several sub-tasks and focus on each sub-task in terms of its problem 
    
[^6]: PUNR: 用户行为建模的新闻推荐预训练

    PUNR: Pre-training with User Behavior Modeling for News Recommendation. (arXiv:2304.12633v1 [cs.IR])

    [http://arxiv.org/abs/2304.12633](http://arxiv.org/abs/2304.12633)

    本论文提出了一种无监督的预训练方法，它可以通过两个任务实现有效的用户行为建模，以提高新闻推荐系统的准确性和性能表现。

    

    新闻推荐旨在基于用户行为预测点击行为。如何有效地建模用户表示是推荐首选新闻的关键。现有方法大多集中在监督微调阶段的改进上。然而，还缺乏针对用户表示优化的基于PLM的无监督预训练方法。在本文中，我们提出了一种具有两个任务的无监督预训练范例，即用户行为掩蔽和用户行为生成，均致力于有效的用户行为建模。首先，我们引入了用户行为掩蔽预训练任务，以恢复基于上下文行为的掩蔽用户行为。通过这种方式，模型可以捕捉到更强大、更全面的用户新闻阅读模式。此外，我们还结合了一种新颖的辅助用户行为生成预训练任务，以增强从用户编码器派生出的用户表示向量。我们使用上述预训练的用户建模来进行新闻推荐，实验结果表明，我们的模型在多个数据集上取得了显著的性能提升。

    News recommendation aims to predict click behaviors based on user behaviors. How to effectively model the user representations is the key to recommending preferred news. Existing works are mostly focused on improvements in the supervised fine-tuning stage. However, there is still a lack of PLM-based unsupervised pre-training methods optimized for user representations. In this work, we propose an unsupervised pre-training paradigm with two tasks, i.e. user behavior masking and user behavior generation, both towards effective user behavior modeling. Firstly, we introduce the user behavior masking pre-training task to recover the masked user behaviors based on their contextual behaviors. In this way, the model could capture a much stronger and more comprehensive user news reading pattern. Besides, we incorporate a novel auxiliary user behavior generation pre-training task to enhance the user representation vector derived from the user encoder. We use the above pre-trained user modeling en
    
[^7]: 一种POI推荐的扩散模型

    A Diffusion model for POI recommendation. (arXiv:2304.07041v1 [cs.IR])

    [http://arxiv.org/abs/2304.07041](http://arxiv.org/abs/2304.07041)

    本文提出了一种基于扩散算法采样用户空间偏好的POI推荐模型，解决了现有方法只基于用户先前访问位置聚合的缺点，适用于推荐新颖区域的POI。

    

    下一个兴趣点（POI）的推荐是定位服务中的关键任务，旨在为用户的下一个目的地提供个性化建议。先前关于POI推荐的工作侧重于对用户空间偏好的建模。然而，现有的利用空间信息的方法仅基于用户先前访问位置的聚合，这会使模型不会推荐新颖区域的POI，从而损害其在许多情况下的性能。此外，将时间顺序信息融入用户的空间偏好仍是一个挑战。在本文中，我们提出了Diff-POI：一种基于扩散的模型，用于采样用户的空间偏好，以进行下一步POI推荐。在扩散算法在从分布中进行采样方面的广泛应用的启发下，Diff-POI使用两个量身定制的图编码模块对用户的访问序列和空间特性进行编码。

    Next Point-of-Interest (POI) recommendation is a critical task in location-based services that aim to provide personalized suggestions for the user's next destination. Previous works on POI recommendation have laid focused on modeling the user's spatial preference. However, existing works that leverage spatial information are only based on the aggregation of users' previous visited positions, which discourages the model from recommending POIs in novel areas. This trait of position-based methods will harm the model's performance in many situations. Additionally, incorporating sequential information into the user's spatial preference remains a challenge. In this paper, we propose Diff-POI: a Diffusion-based model that samples the user's spatial preference for the next POI recommendation. Inspired by the wide application of diffusion algorithm in sampling from distributions, Diff-POI encodes the user's visiting sequence and spatial character with two tailor-designed graph encoding modules
    
[^8]: DiffuRec: 一种用于顺序推荐的扩散模型

    DiffuRec: A Diffusion Model for Sequential Recommendation. (arXiv:2304.00686v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2304.00686](http://arxiv.org/abs/2304.00686)

    本文提出了一种名为DiffuRec 的扩散模型，其将物品表示为分布而不是固定向量，从而更好地反映了用户的多种偏好和物品的多个方面，并成功地应用于顺序推荐。

    

    解决顺序推荐的主流方法是使用固定向量来表示物品。这些向量在捕捉物品的潜在方面和用户的多样化偏好方面方面的能力有限。扩散模型作为一种新的生成范式，在计算机视觉和自然语言处理等领域取得了很好的性能。在我们看来，其在表征生成方面的独特优势很好地适应了顺序推荐的问题设置。本文首次尝试将扩散模型应用于顺序推荐，并提出了DiffuRec，用于物品表示构建和不确定性注入。与将物品表示建模为固定向量不同，我们在DiffuRec中将其表示为分布，这反映了用户的多重兴趣和物品的各个方面的适应性。在扩散阶段，DiffuRec通过添加噪声将目标物品嵌入成高斯分布，进一步应用于顺序物品分布表示。

    Mainstream solutions to Sequential Recommendation (SR) represent items with fixed vectors. These vectors have limited capability in capturing items' latent aspects and users' diverse preferences. As a new generative paradigm, Diffusion models have achieved excellent performance in areas like computer vision and natural language processing. To our understanding, its unique merit in representation generation well fits the problem setting of sequential recommendation. In this paper, we make the very first attempt to adapt Diffusion model to SR and propose DiffuRec, for item representation construction and uncertainty injection. Rather than modeling item representations as fixed vectors, we represent them as distributions in DiffuRec, which reflect user's multiple interests and item's various aspects adaptively. In diffusion phase, DiffuRec corrupts the target item embedding into a Gaussian distribution via noise adding, which is further applied for sequential item distribution representat
    
[^9]: 限制玻尔兹曼机的模式重构

    Pattern reconstruction with restricted Boltzmann machines. (arXiv:2205.07087v3 [math.PR] UPDATED)

    [http://arxiv.org/abs/2205.07087](http://arxiv.org/abs/2205.07087)

    该论文研究了限制玻尔兹曼机在模式重构中的能力，发现隐藏层先验分布的尾部行为对于恢复随机模式的效果有关键影响。

    

    限制玻尔兹曼机是由可见层和隐藏层组成的能量模型。我们找到了描述可见单元上零温度状态的有效能量函数，该函数只依赖于隐藏层先验分布的尾部行为。通过研究该能量函数的局部极小值的位置，我们表明限制玻尔兹曼机重构随机模式的能力确实只取决于隐藏先验分布的尾部。我们发现，具有严格超高斯尾部的隐藏先验仅导致对模式恢复的对数损失，而具有严格次高斯尾部的隐藏单元则导致更难进行有效的恢复；如果隐藏先验具有高斯尾部，恢复能力取决于隐藏单元的数量（与霍普菲尔德模型类似）。

    Restricted Boltzmann machines are energy models made of a visible and a hidden layer. We identify an effective energy function describing the zero-temperature landscape on the visible units and depending only on the tail behaviour of the hidden layer prior distribution. Studying the location of the local minima of such an energy function, we show that the ability of a restricted Boltzmann machine to reconstruct a random pattern depends indeed only on the tail of the hidden prior distribution. We find that hidden priors with strictly super-Gaussian tails give only a logarithmic loss in pattern retrieval, while an efficient retrieval is much harder with hidden units with strictly sub-Gaussian tails; if the hidden prior has Gaussian tails, the retrieval capability is determined by the number of hidden units (as in the Hopfield model).
    
[^10]: 高度几何多模型混合用于鲁棒微调

    Geodesic Multi-Modal Mixup for Robust Fine-Tuning. (arXiv:2203.03897v3 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2203.03897](http://arxiv.org/abs/2203.03897)

    本文研究了CLIP模型的多模态嵌入质量，并发现其统一性和对齐性不足，限制了嵌入的传递性和鲁棒性。为了解决这个问题，我们提出了一种新的鲁棒微调方法，通过高度几何多模型混合生成难负样本，并对模型进行微调。

    

    预训练的多模型模型，如CLIP，在各种应用中提供可转移的嵌入，并显示出有希望的结果。然而，对学习到的多模型嵌入的分析相对较少，嵌入的可转移性有待改进。在这项工作中，我们观察到CLIP为两种不同的模态保留了分离的嵌入子空间，并通过统一对齐的视角对其进行了调查，以衡量学习表示的质量。理论上和实证上，我们展示了即使在微调之后，CLIP仍然保持着较差的统一性和对齐性。这种缺乏对齐和统一性可能限制了嵌入的传递性和鲁棒性。为此，我们设计了一种新的用于鲁棒表示的微调方法，提供更好的对齐和统一性。首先，我们提出了一种高度几何多模型混合方法，将图像和文本的嵌入混合在一起，在超球面上生成难负样本。然后，我们对模型进行鲁棒微调。

    Pre-trained multi-modal models, such as CLIP, provide transferable embeddings and show promising results in diverse applications. However, the analysis of learned multi-modal embeddings is relatively unexplored, and the embedding transferability can be improved. In this work, we observe that CLIP holds separated embedding subspaces for two different modalities, and then we investigate it through the lens of uniformity-alignment to measure the quality of learned representation. Both theoretically and empirically, we show that CLIP retains poor uniformity and alignment even after fine-tuning. Such a lack of alignment and uniformity might restrict the transferability and robustness of embeddings. To this end, we devise a new fine-tuning method for robust representation equipping better alignment and uniformity. First, we propose a Geodesic Multi-Modal Mixup that mixes the embeddings of image and text to generate hard negative samples on the hypersphere. Then, we fine-tune the model on har
    

