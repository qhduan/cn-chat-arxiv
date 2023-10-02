# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Toward Robust Recommendation via Real-time Vicinal Defense.](http://arxiv.org/abs/2309.17278) | 本研究提出了一种实时邻近防御方法（RVD），通过在推荐之前利用邻近的训练数据微调模型，以提高推荐系统对毒化攻击的防御能力。 |
| [^2] | [SAppKG: Mobile App Recommendation Using Knowledge Graph and Side Information-A Secure Framework.](http://arxiv.org/abs/2309.17115) | SAppKG是一个用户隐私保护的移动应用推荐系统，通过利用知识图谱模型和侧面信息，避免了对用户数据的访问，并在真实数据上进行了验证。 |
| [^3] | [Aligning the Capabilities of Large Language Models with the Context of Information Retrieval via Contrastive Feedback.](http://arxiv.org/abs/2309.17078) | 通过对比反馈强化学习框架，有效提升大型语言模型在信息检索中的应用，使其能够生成更具特定性和上下文适应性的回复。 |
| [^4] | [Beyond Co-occurrence: Multi-modal Session-based Recommendation.](http://arxiv.org/abs/2309.17037) | 本文提出了一种基于多模态信息的会话推荐方法，主要解决了从异构描述性信息中提取相关语义、综合利用这些信息推断用户兴趣以及处理数值信息概率影响等问题。 |
| [^5] | [Hallucination Reduction in Long Input Text Summarization.](http://arxiv.org/abs/2309.16781) | 本文旨在减少长篇文本摘要中的幻觉输出，通过在Longformer Encoder-Decoder模型的微调中采用数据过滤和联合实体和摘要生成技术，我们成功提高了生成摘要的质量。 |
| [^6] | [Temporal graph models fail to capture global temporal dynamics.](http://arxiv.org/abs/2309.15730) | 时间图模型无法捕捉全局时间动态，我们提出了一种"最近流行节点"的基线方法，在时间图基准的中等和大规模数据集上胜过其他方法。我们提出了两个基于Wasserstein距离的度量来量化全局动态。我们展示了标准的负采样评估方法在具有强烈时间动态的数据集上可能不适用，我们还展示了简单的负采样方法可能导致模型退化。我们提出了改进的负采样方案，并证明了它们的有效性。我们还将其与无负采样的非对比训练模型进行了比较。 |
| [^7] | [Pre-trained Neural Recommenders: A Transferable Zero-Shot Framework for Recommendation Systems.](http://arxiv.org/abs/2309.01188) | 本论文提出了一种预训练的神经推荐系统框架，可以在新领域中构建推荐系统，减少或无需重新训练，并且不需要使用任何辅助信息。通过利用用户-项目交互矩阵的统计特征，实现了零样本推荐的挑战。 |
| [^8] | [Efficiently Enabling Block Semantics and Data Updates in DNA Storage.](http://arxiv.org/abs/2212.13447) | 提出了一种新颖而灵活的DNA存储架构，可以独立且高效地访问和更新存储空间中的块数据，同时允许对连续数据块进行顺序访问。 |

# 详细

[^1]: 实时邻近防御方法在健壮推荐系统中的应用

    Toward Robust Recommendation via Real-time Vicinal Defense. (arXiv:2309.17278v1 [cs.LG])

    [http://arxiv.org/abs/2309.17278](http://arxiv.org/abs/2309.17278)

    本研究提出了一种实时邻近防御方法（RVD），通过在推荐之前利用邻近的训练数据微调模型，以提高推荐系统对毒化攻击的防御能力。

    

    推荐系统容易受到攻击，恶意数据插入可以导致系统提供有偏见的推荐。为了应对这种攻击，提出了各种健壮的学习方法。然而，大多数方法都是特定于模型或特定于攻击的，缺乏广泛性，而其他方法（如对抗训练）侧重于逃逸攻击，在毒化攻击上有很弱的防御能力。本文提出了一种通用方法——实时邻近防御（RVD），它利用邻近的训练数据在为每个用户进行推荐之前对模型进行微调。RVD在推断阶段工作，以确保实时性样本的健壮性，因此无需更改模型结构和训练过程，更加实用。广泛的实验结果表明，RVD有效地减轻了有针对性的毒化攻击。

    Recommender systems have been shown to be vulnerable to poisoning attacks, where malicious data is injected into the dataset to cause the recommender system to provide biased recommendations. To defend against such attacks, various robust learning methods have been proposed. However, most methods are model-specific or attack-specific, making them lack generality, while other methods, such as adversarial training, are oriented towards evasion attacks and thus have a weak defense strength in poisoning attacks.  In this paper, we propose a general method, Real-time Vicinal Defense (RVD), which leverages neighboring training data to fine-tune the model before making a recommendation for each user. RVD works in the inference phase to ensure the robustness of the specific sample in real-time, so there is no need to change the model structure and training process, making it more practical. Extensive experimental results demonstrate that RVD effectively mitigates targeted poisoning attacks acr
    
[^2]: SAppKG：使用知识图谱和侧面信息的移动应用推荐-一个安全框架

    SAppKG: Mobile App Recommendation Using Knowledge Graph and Side Information-A Secure Framework. (arXiv:2309.17115v1 [cs.SI])

    [http://arxiv.org/abs/2309.17115](http://arxiv.org/abs/2309.17115)

    SAppKG是一个用户隐私保护的移动应用推荐系统，通过利用知识图谱模型和侧面信息，避免了对用户数据的访问，并在真实数据上进行了验证。

    

    随着技术的快速发展和智能手机的广泛使用，移动应用的数量呈指数级增长。找到一组符合用户需求和偏好的合适应用可能具有挑战性。然而，移动应用推荐系统已经成为简化这一过程的有用工具。但是，使用应用推荐系统存在一个缺点，即这些系统需要访问用户数据，这是一种严重的安全违规。虽然用户寻求准确的意见，但不希望在此过程中损害自己的隐私。我们通过开发SAppKG来解决这个问题，这是一个端到端的用户隐私保护知识图谱架构，基于知识图谱模型，如SAppKG-S和SAppKG-D，利用应用交互数据和侧面信息进行移动应用推荐。我们在来自谷歌Play应用商店的真实数据上对所提出的模型进行了测试，使用了精确度、召回率、平均绝对精确度和。均方根误差等指标进行评估。

    Due to the rapid development of technology and the widespread usage of smartphones, the number of mobile applications is exponentially growing. Finding a suitable collection of apps that aligns with users needs and preferences can be challenging. However, mobile app recommender systems have emerged as a helpful tool in simplifying this process. But there is a drawback to employing app recommender systems. These systems need access to user data, which is a serious security violation. While users seek accurate opinions, they do not want to compromise their privacy in the process. We address this issue by developing SAppKG, an end-to-end user privacy-preserving knowledge graph architecture for mobile app recommendation based on knowledge graph models such as SAppKG-S and SAppKG-D, that utilized the interaction data and side information of app attributes. We tested the proposed model on real-world data from the Google Play app store, using precision, recall, mean absolute precision, and me
    
[^3]: 通过对比反馈将大型语言模型的能力与信息检索上下文对齐

    Aligning the Capabilities of Large Language Models with the Context of Information Retrieval via Contrastive Feedback. (arXiv:2309.17078v1 [cs.IR])

    [http://arxiv.org/abs/2309.17078](http://arxiv.org/abs/2309.17078)

    通过对比反馈强化学习框架，有效提升大型语言模型在信息检索中的应用，使其能够生成更具特定性和上下文适应性的回复。

    

    信息检索(IR)是寻找满足用户信息需求的过程，在现代人的生活中起着重要作用。近年来，大型语言模型(LLMs)在各种任务中展示了杰出的能力，其中一些任务对于IR来说非常重要。然而，LLMs经常面临生成缺乏特定性回复的问题。这在很多情况下限制了LLMs在IR中的整体效果。为了解决这些问题，我们提出了一种无监督对齐框架，称为对比反馈强化学习(RLCF)，它赋予LLMs生成既具有高质量又与IR任务需求相符的上下文特定回复的能力。具体而言，我们通过将每个文档与其相似文档进行比较构建对比反馈，然后提出了一个名为Batched-MRR的奖励函数，教导LLMs生成能够捕捉区分文档与其相似文档的细粒度信息的回复。

    Information Retrieval (IR), the process of finding information to satisfy user's information needs, plays an essential role in modern people's lives. Recently, large language models (LLMs) have demonstrated remarkable capabilities across various tasks, some of which are important for IR. Nonetheless, LLMs frequently confront the issue of generating responses that lack specificity. This has limited the overall effectiveness of LLMs for IR in many cases. To address these issues, we present an unsupervised alignment framework called Reinforcement Learning from Contrastive Feedback (RLCF), which empowers LLMs to generate both high-quality and context-specific responses that suit the needs of IR tasks. Specifically, we construct contrastive feedback by comparing each document with its similar documents, and then propose a reward function named Batched-MRR to teach LLMs to generate responses that captures the fine-grained information that distinguish documents from their similar ones. To dem
    
[^4]: 超越共现: 多模态的基于会话的推荐系统

    Beyond Co-occurrence: Multi-modal Session-based Recommendation. (arXiv:2309.17037v1 [cs.IR])

    [http://arxiv.org/abs/2309.17037](http://arxiv.org/abs/2309.17037)

    本文提出了一种基于多模态信息的会话推荐方法，主要解决了从异构描述性信息中提取相关语义、综合利用这些信息推断用户兴趣以及处理数值信息概率影响等问题。

    

    基于会话的推荐旨在根据短会话来揭示匿名用户的偏好。现有的方法主要关注在会话中通过物品ID展示的有限物品共现模式的挖掘，而忽视了用户对特定物品的吸引力是多模态页面上丰富的多模态信息。一般来说，多模态信息可以分为两类: 描述性信息(例如物品图片和描述文本)和数值信息(例如价格)。本文旨在通过整体建模上述多模态信息来改进基于会话的推荐系统。从多模态信息中揭示用户意图主要存在三个问题: (1) 如何从具有不同噪声的异构描述性信息中提取相关语义? (2) 如何综合利用这些异构描述性信息全面推断用户兴趣? (3) 如何处理数值信息的概率影响？

    Session-based recommendation is devoted to characterizing preferences of anonymous users based on short sessions. Existing methods mostly focus on mining limited item co-occurrence patterns exposed by item ID within sessions, while ignoring what attracts users to engage with certain items is rich multi-modal information displayed on pages. Generally, the multi-modal information can be classified into two categories: descriptive information (e.g., item images and description text) and numerical information (e.g., price). In this paper, we aim to improve session-based recommendation by modeling the above multi-modal information holistically. There are mainly three issues to reveal user intent from multi-modal information: (1) How to extract relevant semantics from heterogeneous descriptive information with different noise? (2) How to fuse these heterogeneous descriptive information to comprehensively infer user interests? (3) How to handle probabilistic influence of numerical information
    
[^5]: 长输入文本摘要中的幻觉减少

    Hallucination Reduction in Long Input Text Summarization. (arXiv:2309.16781v1 [cs.CL])

    [http://arxiv.org/abs/2309.16781](http://arxiv.org/abs/2309.16781)

    本文旨在减少长篇文本摘要中的幻觉输出，通过在Longformer Encoder-Decoder模型的微调中采用数据过滤和联合实体和摘要生成技术，我们成功提高了生成摘要的质量。

    

    文本摘要中的幻觉是指模型生成不被输入源文档支持的信息的现象。幻觉给生成的摘要的准确性和可靠性带来了重大障碍。本文旨在减少长篇文本摘要中的幻觉输出。我们使用了包含长篇科学研究文档及其摘要的PubMed数据集。我们在Longformer Encoder-Decoder (LED)模型的微调中加入了数据过滤和联合实体和摘要生成（JAENS）技术，以最小化幻觉，从而提高生成摘要的质量。我们使用以下指标来衡量实体级别的事实一致性：源精确度和目标F1。实验证明，经过微调的LED模型在生成文章摘要方面表现良好。数据过滤技术基于一些预处理步骤。

    Hallucination in text summarization refers to the phenomenon where the model generates information that is not supported by the input source document. Hallucination poses significant obstacles to the accuracy and reliability of the generated summaries. In this paper, we aim to reduce hallucinated outputs or hallucinations in summaries of long-form text documents. We have used the PubMed dataset, which contains long scientific research documents and their abstracts. We have incorporated the techniques of data filtering and joint entity and summary generation (JAENS) in the fine-tuning of the Longformer Encoder-Decoder (LED) model to minimize hallucinations and thereby improve the quality of the generated summary. We have used the following metrics to measure factual consistency at the entity level: precision-source, and F1-target. Our experiments show that the fine-tuned LED model performs well in generating the paper abstract. Data filtering techniques based on some preprocessing steps
    
[^6]: 时间图模型无法捕捉全局时间动态

    Temporal graph models fail to capture global temporal dynamics. (arXiv:2309.15730v1 [cs.IR])

    [http://arxiv.org/abs/2309.15730](http://arxiv.org/abs/2309.15730)

    时间图模型无法捕捉全局时间动态，我们提出了一种"最近流行节点"的基线方法，在时间图基准的中等和大规模数据集上胜过其他方法。我们提出了两个基于Wasserstein距离的度量来量化全局动态。我们展示了标准的负采样评估方法在具有强烈时间动态的数据集上可能不适用，我们还展示了简单的负采样方法可能导致模型退化。我们提出了改进的负采样方案，并证明了它们的有效性。我们还将其与无负采样的非对比训练模型进行了比较。

    

    在动态链接属性预测的背景下，我们分析了最近发布的时间图基准，并提出了一种"最近流行节点"的基线方法，在时间图基准的中等和大规模数据集上胜过其他方法。我们提出了基于Wasserstein距离的两个度量，可以量化数据集的短期和长期全局动态的强度。通过分析我们出乎意料的强大基线，我们展示了标准的负采样评估方法在具有强烈时间动态的数据集上可能不适用。我们还展示了简单的负采样方法在训练过程中可能导致模型退化，导致无法对时间图网络进行排序的预测完全饱和。我们提出了改进的负采样方案用于训练和评估，并证明了它们的有效性。我们还将其与无负采样的非对比训练模型进行了比较。我们的结果表明...

    A recently released Temporal Graph Benchmark is analyzed in the context of Dynamic Link Property Prediction. We outline our observations and propose a trivial optimization-free baseline of "recently popular nodes" outperforming other methods on all medium and large-size datasets in the Temporal Graph Benchmark. We propose two measures based on Wasserstein distance which can quantify the strength of short-term and long-term global dynamics of datasets. By analyzing our unexpectedly strong baseline, we show how standard negative sampling evaluation can be unsuitable for datasets with strong temporal dynamics. We also show how simple negative-sampling can lead to model degeneration during training, resulting in impossible to rank, fully saturated predictions of temporal graph networks. We propose improved negative sampling schemes for both training and evaluation and prove their usefulness. We conduct a comparison with a model trained non-contrastively without negative sampling. Our resul
    
[^7]: 预训练的神经推荐系统：一种可迁移的零样本框架

    Pre-trained Neural Recommenders: A Transferable Zero-Shot Framework for Recommendation Systems. (arXiv:2309.01188v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2309.01188](http://arxiv.org/abs/2309.01188)

    本论文提出了一种预训练的神经推荐系统框架，可以在新领域中构建推荐系统，减少或无需重新训练，并且不需要使用任何辅助信息。通过利用用户-项目交互矩阵的统计特征，实现了零样本推荐的挑战。

    

    现代神经协同过滤技术对于电子商务、社交媒体和内容共享平台的成功至关重要。然而，尽管技术有所进步，但对于每个新的应用领域，我们仍需要从头开始训练一个NCF模型。相反，预训练的视觉和语言模型通常直接应用于各种应用程序，要么是零样本情况，要么是有限的微调。受到预训练模型的影响，我们探索了在新领域中支持构建推荐系统的预训练模型的可能性，只需最少或不需要重新训练，而无需使用任何辅助用户或项目信息。零样本推荐在没有辅助信息的情况下是具有挑战性的，因为当没有重叠的用户或项目时，我们无法在数据集之间建立用户和项目之间的关联。我们的基本见解是用户-项目交互矩阵的统计特征在不同领域中普遍可用。

    Modern neural collaborative filtering techniques are critical to the success of e-commerce, social media, and content-sharing platforms. However, despite technical advances -- for every new application domain, we need to train an NCF model from scratch. In contrast, pre-trained vision and language models are routinely applied to diverse applications directly (zero-shot) or with limited fine-tuning. Inspired by the impact of pre-trained models, we explore the possibility of pre-trained recommender models that support building recommender systems in new domains, with minimal or no retraining, without the use of any auxiliary user or item information. Zero-shot recommendation without auxiliary information is challenging because we cannot form associations between users and items across datasets when there are no overlapping users or items. Our fundamental insight is that the statistical characteristics of the user-item interaction matrix are universally available across different domains 
    
[^8]: 高效实现DNA存储中的块语义和数据更新

    Efficiently Enabling Block Semantics and Data Updates in DNA Storage. (arXiv:2212.13447v2 [cs.ET] UPDATED)

    [http://arxiv.org/abs/2212.13447](http://arxiv.org/abs/2212.13447)

    提出了一种新颖而灵活的DNA存储架构，可以独立且高效地访问和更新存储空间中的块数据，同时允许对连续数据块进行顺序访问。

    

    我们提出了一种新颖而灵活的DNA存储架构，将存储空间分割为固定大小的单位（块），可以独立且高效地随机访问进行读写操作，并且还允许对连续数据块进行有效的顺序访问。与现有工作相比，在我们的架构中，长度为20的随机访问PCR引物并不定义一个单独的对象，而是定义一个独立的存储分区，该分区在内部进行块划分和独立管理。我们展示了每个分区的内部地址空间的灵活性和约束，并将其纳入我们的设计中，以提供丰富和功能齐全的存储语义，如块存储组织、高效的数据更新实现和顺序访问。为了充分利用PCR寻址的前缀特性，我们定义了一种方法来转换分区的内部寻址方案。

    We propose a novel and flexible DNA-storage architecture, which divides the storage space into fixed-size units (blocks) that can be independently and efficiently accessed at random for both read and write operations, and further allows efficient sequential access to consecutive data blocks. In contrast to prior work, in our architecture a pair of random-access PCR primers of length 20 does not define a single object, but an independent storage partition, which is internally blocked and managed independently of other partitions. We expose the flexibility and constraints with which the internal address space of each partition can be managed, and incorporate them into our design to provide rich and functional storage semantics, such as block-storage organization, efficient implementation of data updates, and sequential access. To leverage the full power of the prefix-based nature of PCR addressing, we define a methodology for transforming the internal addressing scheme of a partition int
    

