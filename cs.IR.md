# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [HiPrompt: Few-Shot Biomedical Knowledge Fusion via Hierarchy-Oriented Prompting.](http://arxiv.org/abs/2304.05973) | HiPrompt是一个监督效率高的知识融合框架，通过层次导向提示和少样本推理能力，弥补了生物医学知识融合和神经嵌入模型之间的差距。 |
| [^2] | [Dynamic Mixed Membership Stochastic Block Model for Weighted Labeled Networks.](http://arxiv.org/abs/2304.05894) | 本论文提出了一种扩展混合成员随机块模型来推断动态标签网络的方法，具有很好的鲁棒性和良好的性能，相对于静态标签网络，对数据的训练需求较少。 |
| [^3] | [Edge-cloud Collaborative Learning with Federated and Centralized Features.](http://arxiv.org/abs/2304.05871) | 本文提出了一种边缘云协作知识转移框架（ECCT），使得共享特征嵌入和预测日志的双向知识传输成为可能，从而实现了个性化增强、模型的异构性、容忍训练的异步性和缓解通信负担的功能。 |
| [^4] | [Deep Stable Multi-Interest Learning for Out-of-distribution Sequential Recommendation.](http://arxiv.org/abs/2304.05615) | 提出了DESMIL，一个新的多兴趣网络，用于序列推荐模型中解决跨领域泛化问题，通过去相关提取的多个兴趣向量，消除虚假相关性，实验结果证明其优于现有的最先进方法。 |
| [^5] | [FALQU: Finding Answers to Legal Questions.](http://arxiv.org/abs/2304.05611) | 本文介绍了一个基于“Law Stack Exchange”网站的法律信息检索数据集 FALQU，包含真实世界用户的多领域信息需求，是当前首个使用 LawSE 数据的测试集。 |
| [^6] | [Towards More Robust and Accurate Sequential Recommendation with Cascade-guided Adversarial Training.](http://arxiv.org/abs/2304.05492) | 本研究利用级联指导下的对抗训练方法，增强了串联推荐模型的鲁棒性和准确性，取得了比已有方法更好的结果。 |
| [^7] | [Audience Expansion for Multi-show Release Based on an Edge-prompted Heterogeneous Graph Network.](http://arxiv.org/abs/2304.05474) | 本文提出了一种基于边缘触发的异构图网络的两阶段受众扩展方案，可以考虑不同的双面交互和特征。 |
| [^8] | [Combat AI With AI: Counteract Machine-Generated Fake Restaurant Reviews on Social Media.](http://arxiv.org/abs/2302.07731) | 本文针对机器生成的虚假评论提出了一种用高质量餐厅评论生成虚假评论并微调GPT输出检测器的方法，该方法预测虚假评论的性能优于现有解决方案。同时，我们还探索了预测非精英评论的模型，并在几个维度上对这些评论进行分析，此类机器生成的虚假评论是社交媒体平台面临的持续挑战。 |
| [^9] | [Multimodal Matching-aware Co-attention Networks with Mutual Knowledge Distillation for Fake News Detection.](http://arxiv.org/abs/2212.05699) | 提出了一种基于相互知识蒸馏的多模匹配感知协同注意力网络用于改进假新闻检测，通过图像-文本匹配感知协同机制捕获图像和文本的对齐以实现更好的多模态融合，同时利用两个中心分别为文本和图像的协同注意力网络进行相互知识蒸馏。 |
| [^10] | [Tensor Completion with Provable Consistency and Fairness Guarantees for Recommender Systems.](http://arxiv.org/abs/2204.01815) | 本文介绍了一种新的一致性方法来解决矩阵和张量补全问题，在推荐系统应用中，我们证明了通过保留单位比例和一致性两个约束条件可以实现解的存在性与唯一性。 |
| [^11] | [Review-Based Domain Disentanglement without Duplicate Users or Contexts for Cross-Domain Recommendation.](http://arxiv.org/abs/2110.12648) | 本文提出了一种使用评论文本来进行领域解缠的方法，使用三个文本分析模块，由单一领域判别器指引，并采用一种新的优化策略，提高了领域解缠的质量，并且扩展了编码网络从单个领域到多个领域。实验证明，该方法比现有方法更高效、稳健和可扩展。 |
| [^12] | [Vec2GC -- A Graph Based Clustering Method for Text Representations.](http://arxiv.org/abs/2104.09439) | 本文介绍了一种文本表示聚类方法Vec2GC，将聚类算法与基于文本表示学习创建的术语或文档加权图的社区检测相结合，可以用于无监督的文档处理。 |

# 详细

[^1]: HiPrompt: 层次导向提示的少样本生物医学知识融合

    HiPrompt: Few-Shot Biomedical Knowledge Fusion via Hierarchy-Oriented Prompting. (arXiv:2304.05973v1 [cs.IR])

    [http://arxiv.org/abs/2304.05973](http://arxiv.org/abs/2304.05973)

    HiPrompt是一个监督效率高的知识融合框架，通过层次导向提示和少样本推理能力，弥补了生物医学知识融合和神经嵌入模型之间的差距。

    

    综合的生物医学知识库可以增强医学决策过程，需要通过统一的索引系统融合来自不同来源的知识图谱。索引系统通常以层次结构组织生物医学术语，以提供细粒度的对齐实体。为了解决生物医学知识融合 (BKF) 任务中监督不足的挑战，研究人员提出了各种无监督方法。然而，这些方法严重依赖于特定的词汇和结构匹配算法，无法捕捉生物医学实体和术语所传达的丰富语义。最近，神经嵌入模型在语义丰富的任务中被证明是有效的，但它们依赖于充足标记数据进行充分训练。为了弥补稀缺标记 BKF 和神经嵌入模型之间的差距，我们提出了 HiPrompt，一个监督效率高的知识融合框架，可以引发大规模语义推理的少样本推理能力。

    Medical decision-making processes can be enhanced by comprehensive biomedical knowledge bases, which require fusing knowledge graphs constructed from different sources via a uniform index system. The index system often organizes biomedical terms in a hierarchy to provide the aligned entities with fine-grained granularity. To address the challenge of scarce supervision in the biomedical knowledge fusion (BKF) task, researchers have proposed various unsupervised methods. However, these methods heavily rely on ad-hoc lexical and structural matching algorithms, which fail to capture the rich semantics conveyed by biomedical entities and terms. Recently, neural embedding models have proved effective in semantic-rich tasks, but they rely on sufficient labeled data to be adequately trained. To bridge the gap between the scarce-labeled BKF and neural embedding models, we propose HiPrompt, a supervision-efficient knowledge fusion framework that elicits the few-shot reasoning ability of large la
    
[^2]: 带权标签网络的动态混合成员随机块模型

    Dynamic Mixed Membership Stochastic Block Model for Weighted Labeled Networks. (arXiv:2304.05894v1 [cs.LG])

    [http://arxiv.org/abs/2304.05894](http://arxiv.org/abs/2304.05894)

    本论文提出了一种扩展混合成员随机块模型来推断动态标签网络的方法，具有很好的鲁棒性和良好的性能，相对于静态标签网络，对数据的训练需求较少。

    

    大多数现实中的网络都是随时间变化的。现有的动态网络模型要么没有标签，要么假定只有一个成员结构。另一方面，一种新的混合成员随机块模型（MMSBM）家族允许在混合成员聚类的假设下模拟静态标签网络。在本文中，我们提出将这种模型扩展到在混合成员假设下推断动态标签网络的模型类。我们的方法采用模型参数的时间先验形式，并依赖于动力学不是突然的单一假设。我们展示了我们的方法与现有方法显著不同，并且可以模拟更复杂的系统——动态标记网络。我们在合成和现实数据集上进行了几个实验，证明了我们方法的鲁棒性。我们方法的一个关键优势是，它只需要很少的训练数据就能产生良好的结果。与静态标签网络相比，我们方法在动态标签网络下的性能提升显著。

    Most real-world networks evolve over time. Existing literature proposes models for dynamic networks that are either unlabeled or assumed to have a single membership structure. On the other hand, a new family of Mixed Membership Stochastic Block Models (MMSBM) allows to model static labeled networks under the assumption of mixed-membership clustering. In this work, we propose to extend this later class of models to infer dynamic labeled networks under a mixed membership assumption. Our approach takes the form of a temporal prior on the model's parameters. It relies on the single assumption that dynamics are not abrupt. We show that our method significantly differs from existing approaches, and allows to model more complex systems --dynamic labeled networks. We demonstrate the robustness of our method with several experiments on both synthetic and real-world datasets. A key interest of our approach is that it needs very few training data to yield good results. The performance gain under 
    
[^3]: 带有联邦和集中特征的边缘云协作学习

    Edge-cloud Collaborative Learning with Federated and Centralized Features. (arXiv:2304.05871v1 [cs.LG])

    [http://arxiv.org/abs/2304.05871](http://arxiv.org/abs/2304.05871)

    本文提出了一种边缘云协作知识转移框架（ECCT），使得共享特征嵌入和预测日志的双向知识传输成为可能，从而实现了个性化增强、模型的异构性、容忍训练的异步性和缓解通信负担的功能。

    

    联邦学习（FL）是一种受欢迎的边缘计算方式，不会危及用户的隐私。目前的FL范例假定数据仅驻留在边缘，而云服务器仅执行模型平均。但是，在诸如推荐系统之类的实际情况下，云服务器具有存储历史和交互特征的能力。本文提出的Edge-Cloud Collaborative Knowledge Transfer Framework（ECCT）弥合了边缘和云之间的差距，使其能够在两者之间进行双向知识传输，共享特征嵌入和预测日志。 ECCT巩固了各种好处，包括增强个性化，实现模型异构性，容忍培训异步性和缓解通信负担。对公共和工业数据集的广泛实验表明ECCT的有效性和学术和工业使用的潜力。

    Federated learning (FL) is a popular way of edge computing that doesn't compromise users' privacy. Current FL paradigms assume that data only resides on the edge, while cloud servers only perform model averaging. However, in real-life situations such as recommender systems, the cloud server has the ability to store historical and interactive features. In this paper, our proposed Edge-Cloud Collaborative Knowledge Transfer Framework (ECCT) bridges the gap between the edge and cloud, enabling bi-directional knowledge transfer between both, sharing feature embeddings and prediction logits. ECCT consolidates various benefits, including enhancing personalization, enabling model heterogeneity, tolerating training asynchronization, and relieving communication burdens. Extensive experiments on public and industrial datasets demonstrate ECCT's effectiveness and potential for use in academia and industry.
    
[^4]: 多兴趣深度稳定学习用于跨领域序列推荐模型

    Deep Stable Multi-Interest Learning for Out-of-distribution Sequential Recommendation. (arXiv:2304.05615v1 [cs.IR])

    [http://arxiv.org/abs/2304.05615](http://arxiv.org/abs/2304.05615)

    提出了DESMIL，一个新的多兴趣网络，用于序列推荐模型中解决跨领域泛化问题，通过去相关提取的多个兴趣向量，消除虚假相关性，实验结果证明其优于现有的最先进方法。

    

    最近，多利益模型被用作提取用户多个表示向量的兴趣， 对于序列推荐表现良好。然而，目前存在的多兴趣推荐模型都没有考虑兴趣分布可能改变带来的跨领域泛化问题。考虑到用户多个兴趣通常高度相关，模型有机会学习到嘈杂兴趣和目标项之间的虚假相关性。数据分布发生变化，兴趣之间的相关性也会发生变化，虚假相关性会误导模型进行错误预测。为了解决上述跨领域泛化问题，我们提出了一个新的多利益网络，名为DESMIL，该网络试图在模型中去相关提取的利益，从而可以消除虚假的相关性。DESMIL应用一个注意力模块来提取多个利益，一个基于Transformer的编码器来对它们进行编码，一个去相关模块来去除相关性。在两个真实世界数据集上的实验证明了DESMIL在in-distribution和out-of-distribution方面都优于现有最先进的方法。

    Recently, multi-interest models, which extract interests of a user as multiple representation vectors, have shown promising performances for sequential recommendation. However, none of existing multi-interest recommendation models consider the Out-Of-Distribution (OOD) generalization problem, in which interest distribution may change. Considering multiple interests of a user are usually highly correlated, the model has chance to learn spurious correlations between noisy interests and target items. Once the data distribution changes, the correlations among interests may also change, and the spurious correlations will mislead the model to make wrong predictions. To tackle with above OOD generalization problem, we propose a novel multi-interest network, named DEep Stable Multi-Interest Learning (DESMIL), which attempts to de-correlate the extracted interests in the model, and thus spurious correlations can be eliminated. DESMIL applies an attentive module to extract multiple interests, an
    
[^5]: FALQU: 寻找法律问题答案

    FALQU: Finding Answers to Legal Questions. (arXiv:2304.05611v1 [cs.IR])

    [http://arxiv.org/abs/2304.05611](http://arxiv.org/abs/2304.05611)

    本文介绍了一个基于“Law Stack Exchange”网站的法律信息检索数据集 FALQU，包含真实世界用户的多领域信息需求，是当前首个使用 LawSE 数据的测试集。

    

    本文介绍了一个新的法律信息检索数据集 - FALQU，该数据集包括了来自“Law Stack Exchange”问答网站的法律问题和答案，涉及版权、知识产权、刑法等多个领域，数据的多样性代表了真实世界用户的信息需求。

    This paper presents a new test collection for Legal IR, FALQU: Finding Answers to Legal Questions, where questions and answers were obtained from Law Stack Exchange (LawSE), a Q&A website for legal professionals, and others with experience in law. Much in line with Stack overflow, Law Stack Exchange has a variety of questions on different topics such as copyright, intellectual property, and criminal laws, making it an interesting source for dataset construction. Questions are also not limited to one country. Often, users of different nationalities may ask questions about laws in different countries and expertise. Therefore, questions in FALQU represent real-world users' information needs thus helping to avoid lab-generated questions. Answers on the other side are given by experts in the field. FALQU is the first test collection, to the best of our knowledge, to use LawSE, considering more diverse questions than the questions from the standard legal bar and judicial exams. It contains 9
    
[^6]: 改进串联推荐的鲁棒性和准确性: 伴随级联指导的对抗训练方法

    Towards More Robust and Accurate Sequential Recommendation with Cascade-guided Adversarial Training. (arXiv:2304.05492v1 [cs.IR])

    [http://arxiv.org/abs/2304.05492](http://arxiv.org/abs/2304.05492)

    本研究利用级联指导下的对抗训练方法，增强了串联推荐模型的鲁棒性和准确性，取得了比已有方法更好的结果。

    

    串联推荐模型是一种通过学习用户与物品间的时间顺序互动来进行推荐的模型，其已经在许多领域中展现出了良好的表现。然而，近期串联推荐模型的鲁棒性备受质疑。这种模型的两个特性使其容易受到攻击 - 在训练中会产生级联效应，在模型过度依赖时间信息的同时会忽略其他特征。为了解决这些问题，本文提出了一种针对串联推荐模型的级联指导下的对抗训练的方法。我们的方法利用串联建模中的内在级联效应，在训练过程中产生战略性的对抗性扰动来影响物品嵌入。在使用不同的公共数据集训练四种最先进的串联模型实验中，我们的训练方法产生了比现有方法更高的模型鲁棒性，并获得了更好的性能。

    Sequential recommendation models, models that learn from chronological user-item interactions, outperform traditional recommendation models in many settings. Despite the success of sequential recommendation models, their robustness has recently come into question. Two properties unique to the nature of sequential recommendation models may impair their robustness - the cascade effects induced during training and the model's tendency to rely too heavily on temporal information. To address these vulnerabilities, we propose Cascade-guided Adversarial training, a new adversarial training procedure that is specifically designed for sequential recommendation models. Our approach harnesses the intrinsic cascade effects present in sequential modeling to produce strategic adversarial perturbations to item embeddings during training. Experiments on training state-of-the-art sequential models on four public datasets from different domains show that our training approach produces superior model ran
    
[^7]: 基于边触发异构图网络的多节目发行的受众扩展

    Audience Expansion for Multi-show Release Based on an Edge-prompted Heterogeneous Graph Network. (arXiv:2304.05474v1 [cs.SI])

    [http://arxiv.org/abs/2304.05474](http://arxiv.org/abs/2304.05474)

    本文提出了一种基于边缘触发的异构图网络的两阶段受众扩展方案，可以考虑不同的双面交互和特征。

    

    在视频平台上，针对新节目进行受众定位和扩展的关键在于如何生成它们的嵌入。应该从用户和节目的角度进行个性化处理。此外，为了追求即时（点击）和长期（观看时间）奖励，以及新节目的冷启动问题，带来了额外的挑战。这种问题适合通过异构图模型进行处理，因为数据具有自然的图结构。但是现实世界中的网络通常具有数十亿个节点和各种类型的边缘。很少有现有方法专注于处理大规模数据并利用不同类型的边缘，特别是后者。本文提出了一种基于边缘触发的异构图网络的两阶段受众扩展方案，可以考虑不同的双面交互和特征。在离线阶段，选择用户ID和显示器的特定边缘信息组合来构建图形。

    In the user targeting and expanding of new shows on a video platform, the key point is how their embeddings are generated. It's supposed to be personalized from the perspective of both users and shows. Furthermore, the pursue of both instant (click) and long-time (view time) rewards, and the cold-start problem for new shows bring additional challenges. Such a problem is suitable for processing by heterogeneous graph models, because of the natural graph structure of data. But real-world networks usually have billions of nodes and various types of edges. Few existing methods focus on handling large-scale data and exploiting different types of edges, especially the latter. In this paper, we propose a two-stage audience expansion scheme based on an edge-prompted heterogeneous graph network which can take different double-sided interactions and features into account. In the offline stage, to construct the graph, user IDs and specific side information combinations of the shows are chosen to 
    
[^8]: AI对抗AI：在社交媒体上打击机器生成的虚假餐厅评论

    Combat AI With AI: Counteract Machine-Generated Fake Restaurant Reviews on Social Media. (arXiv:2302.07731v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2302.07731](http://arxiv.org/abs/2302.07731)

    本文针对机器生成的虚假评论提出了一种用高质量餐厅评论生成虚假评论并微调GPT输出检测器的方法，该方法预测虚假评论的性能优于现有解决方案。同时，我们还探索了预测非精英评论的模型，并在几个维度上对这些评论进行分析，此类机器生成的虚假评论是社交媒体平台面临的持续挑战。

    

    最近生成模型（如GPT）的发展使得以更低的成本制造出难以区分的虚假顾客评论，从而对社交媒体平台检测这些机器生成的虚假评论造成挑战。本文提出利用Yelp验证的高质量的精英餐厅评论来生成OpenAI GPT评论生成器的虚假评论，并最终微调GPT输出检测器来预测明显优于现有解决方案的虚假评论。我们进一步将模型应用于预测非精英评论，并在几个维度（如评论、用户和餐厅特征以及写作风格）上识别模式。我们展示了社交媒体平台正在不断面临机器生成的虚假评论的挑战，尽管他们可能实施检测系统以过滤出可疑的评论。

    Recent advances in generative models such as GPT may be used to fabricate indistinguishable fake customer reviews at a much lower cost, thus posing challenges for social media platforms to detect these machine-generated fake reviews. We propose to leverage the high-quality elite restaurant reviews verified by Yelp to generate fake reviews from the OpenAI GPT review creator and ultimately fine-tune a GPT output detector to predict fake reviews that significantly outperform existing solutions. We further apply the model to predict non-elite reviews and identify the patterns across several dimensions, such as review, user and restaurant characteristics, and writing style. We show that social media platforms are continuously challenged by machine-generated fake reviews, although they may implement detection systems to filter out suspicious reviews.
    
[^9]: 基于相互知识蒸馏的多模匹配感知协同注意力网络用于假新闻检测

    Multimodal Matching-aware Co-attention Networks with Mutual Knowledge Distillation for Fake News Detection. (arXiv:2212.05699v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2212.05699](http://arxiv.org/abs/2212.05699)

    提出了一种基于相互知识蒸馏的多模匹配感知协同注意力网络用于改进假新闻检测，通过图像-文本匹配感知协同机制捕获图像和文本的对齐以实现更好的多模态融合，同时利用两个中心分别为文本和图像的协同注意力网络进行相互知识蒸馏。

    

    假新闻常常包含文本和图像等多媒体信息来误导读者，从而扩大其影响力。目前大多数的假新闻检测方法使用协同注意力机制来融合多模态特征而忽略了协同注意力中图像和文本的一致性。本文提出了一种基于相互知识蒸馏的多模匹配感知协同注意力网络来改进假新闻检测。具体而言，我们设计了一种图像-文本匹配感知协同机制，用于捕获图像和文本的对齐以实现更好的多模态融合。图像-文本匹配表示可以通过视觉语言预训练模型获得。此外，基于设计的图像-文本匹配感知协同机制，我们提出构建两个分别以文本和图像为中心的协同注意力网络以进行相互知识蒸馏，以提高假新闻检测的性能。在三个基准数据集上的广泛实验表明：

    Fake news often involves multimedia information such as text and image to mislead readers, proliferating and expanding its influence. Most existing fake news detection methods apply the co-attention mechanism to fuse multimodal features while ignoring the consistency of image and text in co-attention. In this paper, we propose multimodal matching-aware co-attention networks with mutual knowledge distillation for improving fake news detection. Specifically, we design an image-text matching-aware co-attention mechanism which captures the alignment of image and text for better multimodal fusion. The image-text matching representation can be obtained via a vision-language pre-trained model. Additionally, based on the designed image-text matching-aware co-attention mechanism, we propose to build two co-attention networks respectively centered on text and image for mutual knowledge distillation to improve fake news detection. Extensive experiments on three benchmark datasets demonstrate that
    
[^10]: 具有可证明的一致性和公平保证的推荐系统张量补全

    Tensor Completion with Provable Consistency and Fairness Guarantees for Recommender Systems. (arXiv:2204.01815v3 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2204.01815](http://arxiv.org/abs/2204.01815)

    本文介绍了一种新的一致性方法来解决矩阵和张量补全问题，在推荐系统应用中，我们证明了通过保留单位比例和一致性两个约束条件可以实现解的存在性与唯一性。

    

    我们引入了一种新的基于一致性的方法来定义和解决非负/正矩阵和张量补全问题。该框架的新颖之处在于，我们不是人为地将问题形式化为任意优化问题，例如，最小化一个结构量，如秩或范数，而是展示了一个单一的属性/约束：保留单位比例一致性，保证了解的存在，并在相对较弱的支持假设下保证了解的唯一性。该框架和解算法也直接推广到任意维度的张量中，同时保持了固定维度 d 的问题规模的线性计算复杂性。在推荐系统应用中，我们证明了两个合理的性质，这些性质应该适用于任何 RS 问题的解，足以允许在我们的框架内建立唯一性保证。关键理论贡献是展示了这些约束下解的存在性与唯一性。

    We introduce a new consistency-based approach for defining and solving nonnegative/positive matrix and tensor completion problems. The novelty of the framework is that instead of artificially making the problem well-posed in the form of an application-arbitrary optimization problem, e.g., minimizing a bulk structural measure such as rank or norm, we show that a single property/constraint: preserving unit-scale consistency, guarantees the existence of both a solution and, under relatively weak support assumptions, uniqueness. The framework and solution algorithms also generalize directly to tensors of arbitrary dimensions while maintaining computational complexity that is linear in problem size for fixed dimension d. In the context of recommender system (RS) applications, we prove that two reasonable properties that should be expected to hold for any solution to the RS problem are sufficient to permit uniqueness guarantees to be established within our framework. Key theoretical contribu
    
[^11]: 无重复用户或上下文的基于评论的跨领域推荐中的领域解缠方法

    Review-Based Domain Disentanglement without Duplicate Users or Contexts for Cross-Domain Recommendation. (arXiv:2110.12648v3 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2110.12648](http://arxiv.org/abs/2110.12648)

    本文提出了一种使用评论文本来进行领域解缠的方法，使用三个文本分析模块，由单一领域判别器指引，并采用一种新的优化策略，提高了领域解缠的质量，并且扩展了编码网络从单个领域到多个领域。实验证明，该方法比现有方法更高效、稳健和可扩展。

    

    跨领域推荐在解决数据稀疏性和冷启动问题方面已经取得了良好的结果。尽管如此，现有方法仅专注于领域可共享信息（重叠用户或相同的上下文）用于知识转移，并且没有这样的要求就很难进行良好的泛化。为了解决这些问题，我们建议利用对大多数电子商务系统通用的评论文本。我们的模型（名为SER）使用三个文本分析模块，由单一领域判别器指导进行解缠表示学习。在这里，我们提出了一种新的优化策略，可以提高领域解缠的质量，并削弱源领域的不良信息。此外，我们将编码网络从单个领域扩展到多个领域，这已被证明对于基于评论的推荐系统非常强大。广泛的实验和消融研究表明，与现有方法相比，我们的方法高效、稳健且可扩展。

    A cross-domain recommendation has shown promising results in solving data-sparsity and cold-start problems. Despite such progress, existing methods focus on domain-shareable information (overlapped users or same contexts) for a knowledge transfer, and they fail to generalize well without such requirements. To deal with these problems, we suggest utilizing review texts that are general to most e-commerce systems. Our model (named SER) uses three text analysis modules, guided by a single domain discriminator for disentangled representation learning. Here, we suggest a novel optimization strategy that can enhance the quality of domain disentanglement, and also debilitates detrimental information of a source domain. Also, we extend the encoding network from a single to multiple domains, which has proven to be powerful for review-based recommender systems. Extensive experiments and ablation studies demonstrate that our method is efficient, robust, and scalable compared to the state-of-the-a
    
[^12]: Vec2GC -- 基于图的文本表示聚类方法

    Vec2GC -- A Graph Based Clustering Method for Text Representations. (arXiv:2104.09439v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2104.09439](http://arxiv.org/abs/2104.09439)

    本文介绍了一种文本表示聚类方法Vec2GC，将聚类算法与基于文本表示学习创建的术语或文档加权图的社区检测相结合，可以用于无监督的文档处理。

    

    在有限或没有标签数据的NLP流水线中，需要依赖无监督方法进行文档处理。无监督方法通常依赖于术语或文档的聚类。本文介绍了一种新的聚类算法，Vec2GC (Vector to Graph Communities)，它是一个端到端的流水线，可以针对任何给定的文本语料库聚类术语或文档。我们的方法使用基于文本表示学习创建的术语或文档加权图的社区检测。Vec2GC聚类算法是一种基于密度的方法，同时支持层次聚类。

    NLP pipelines with limited or no labeled data, rely on unsupervised methods for document processing. Unsupervised approaches typically depend on clustering of terms or documents. In this paper, we introduce a novel clustering algorithm, Vec2GC (Vector to Graph Communities), an end-to-end pipeline to cluster terms or documents for any given text corpus. Our method uses community detection on a weighted graph of the terms or documents, created using text representation learning. Vec2GC clustering algorithm is a density based approach, that supports hierarchical clustering as well.
    

