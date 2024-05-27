# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [KELLMRec: Knowledge-Enhanced Large Language Models for Recommendation](https://arxiv.org/abs/2403.06642) | 提出了一种知识增强的大型语言模型用于推荐的方法，通过使用外部知识来帮助生成真实可用的文本，并包括知识为基础的对比学习方案进行训练。 |
| [^2] | [EasyRL4Rec: A User-Friendly Code Library for Reinforcement Learning Based Recommender Systems](https://arxiv.org/abs/2402.15164) | EasyRL4Rec是一个面向基于强化学习的推荐系统的用户友好和高效库，提供了多样化的RL环境、全面的核心模块、一致的评估标准和定制解决方案，旨在帮助简化模型开发并改善长期用户参与度。 |
| [^3] | [Artificial Intelligence Model for Tumoral Clinical Decision Support Systems](https://arxiv.org/abs/2301.03701) | 该研究提出的人工智能模型利用二进制信息生成丰富的图像描述符，从而实现了在肿瘤临床决策支持系统中检测患者特征并推荐最相似病例的目标。 |
| [^4] | [Identifiability Matters: Revealing the Hidden Recoverable Condition in Unbiased Learning to Rank.](http://arxiv.org/abs/2309.15560) | 研究揭示在无偏学习排名中，当点击数据不能完全拟合时，无法恢复真实相关性，导致排名性能显著降低，提出了可识别性图模型作为解决方案。 |

# 详细

[^1]: KELLMRec: 知识增强大型语言模型用于推荐

    KELLMRec: Knowledge-Enhanced Large Language Models for Recommendation

    [https://arxiv.org/abs/2403.06642](https://arxiv.org/abs/2403.06642)

    提出了一种知识增强的大型语言模型用于推荐的方法，通过使用外部知识来帮助生成真实可用的文本，并包括知识为基础的对比学习方案进行训练。

    

    在推荐系统领域，利用语义信息是一个重要的研究问题，旨在补充主流基于ID的方法的缺失部分。随着LLM的兴起，它作为知识库的能力和推理能力为这一研究领域开辟了新的可能性，使基于LLM的推荐成为新兴研究方向。然而，直接使用LLM来处理推荐场景中的语义信息是不可靠和次优的，由于存在幻觉等问题。应对这一问题的一种有前途的方法是利用外部知识来帮助LLM生成真实可用的文本。受以上动机的启发，我们提出了一种知识增强的LLMRec方法。除了在提示中使用外部知识外，所提出的方法还包括一个基于知识的对比学习方案用于训练。在公共数据集和企业中进行的实验

    arXiv:2403.06642v1 Announce Type: cross  Abstract: The utilization of semantic information is an important research problem in the field of recommender systems, which aims to complement the missing parts of mainstream ID-based approaches. With the rise of LLM, its ability to act as a knowledge base and its reasoning capability have opened up new possibilities for this research area, making LLM-based recommendation an emerging research direction. However, directly using LLM to process semantic information for recommendation scenarios is unreliable and sub-optimal due to several problems such as hallucination. A promising way to cope with this is to use external knowledge to aid LLM in generating truthful and usable text. Inspired by the above motivation, we propose a Knowledge-Enhanced LLMRec method. In addition to using external knowledge in prompts, the proposed method also includes a knowledge-based contrastive learning scheme for training. Experiments on public datasets and in-enter
    
[^2]: EasyRL4Rec：面向基于强化学习的推荐系统的用户友好代码库

    EasyRL4Rec: A User-Friendly Code Library for Reinforcement Learning Based Recommender Systems

    [https://arxiv.org/abs/2402.15164](https://arxiv.org/abs/2402.15164)

    EasyRL4Rec是一个面向基于强化学习的推荐系统的用户友好和高效库，提供了多样化的RL环境、全面的核心模块、一致的评估标准和定制解决方案，旨在帮助简化模型开发并改善长期用户参与度。

    

    强化学习（RL）-基础的推荐系统（RSs）越来越被认可其提高长期用户参与度的能力。然而，这个领域面临挑战，如缺乏易用的框架、评估标准不一致以及复制以前的工作的复杂性。为解决这些障碍，我们提出了EasyRL4Rec，一个专为基于RL的RSs量身定制的用户友好和高效的库。EasyRL4Rec具有基于五个广泛使用的公共数据集构建的轻量级、多样化的RL环境，并配备了全面的核心模块，提供丰富的选项来简化模型的开发。它建立了一致的评估标准，重点关注长期影响，并引入了针对推荐系统定制的状态建模和行为表示的定制解决方案。此外，我们分享了通过与当前方法进行的大量实验获得的宝贵见解。EasyRL4Rec旨在促进

    arXiv:2402.15164v1 Announce Type: cross  Abstract: Reinforcement Learning (RL)-Based Recommender Systems (RSs) are increasingly recognized for their ability to improve long-term user engagement. Yet, the field grapples with challenges such as the absence of accessible frameworks, inconsistent evaluation standards, and the complexity of replicating prior work. Addressing these obstacles, we present EasyRL4Rec, a user-friendly and efficient library tailored for RL-based RSs. EasyRL4Rec features lightweight, diverse RL environments built on five widely-used public datasets, and is equipped with comprehensive core modules that offer rich options to ease the development of models. It establishes consistent evaluation criteria with a focus on long-term impacts and introduces customized solutions for state modeling and action representation tailored to recommender systems. Additionally, we share valuable insights gained from extensive experiments with current methods. EasyRL4Rec aims to facil
    
[^3]: 肿瘤临床决策支持系统的人工智能模型

    Artificial Intelligence Model for Tumoral Clinical Decision Support Systems

    [https://arxiv.org/abs/2301.03701](https://arxiv.org/abs/2301.03701)

    该研究提出的人工智能模型利用二进制信息生成丰富的图像描述符，从而实现了在肿瘤临床决策支持系统中检测患者特征并推荐最相似病例的目标。

    

    通过比较诊断性脑瘤评估，利用医疗中心的信息来比较类似病例，提出的系统能够利用人工智能模型检索给定查询的最相似的脑瘤病例。主要目标是通过生成更准确的医学图像表示来改善诊断过程，特别关注患者特定正常特征和病理。与先前模型的关键区别在于，它能够仅从二进制信息生成丰富的图像描述符，消除了昂贵且难以获得的肿瘤分割的需求。

    arXiv:2301.03701v2 Announce Type: replace-cross  Abstract: Comparative diagnostic in brain tumor evaluation makes possible to use the available information of a medical center to compare similar cases when a new patient is evaluated. By leveraging Artificial Intelligence models, the proposed system is able of retrieving the most similar cases of brain tumors for a given query. The primary objective is to enhance the diagnostic process by generating more accurate representations of medical images, with a particular focus on patient-specific normal features and pathologies. A key distinction from previous models lies in its ability to produce enriched image descriptors solely from binary information, eliminating the need for costly and difficult to obtain tumor segmentation.   The proposed model uses Artificial Intelligence to detect patient features to recommend the most similar cases from a database. The system not only suggests similar cases but also balances the representation of hea
    
[^4]: 识别性很重要：揭示无偏学习排名中隐藏的可恢复条件

    Identifiability Matters: Revealing the Hidden Recoverable Condition in Unbiased Learning to Rank. (arXiv:2309.15560v1 [cs.IR])

    [http://arxiv.org/abs/2309.15560](http://arxiv.org/abs/2309.15560)

    研究揭示在无偏学习排名中，当点击数据不能完全拟合时，无法恢复真实相关性，导致排名性能显著降低，提出了可识别性图模型作为解决方案。

    

    无偏学习排名(Unbiased Learning to Rank, ULTR)在从有偏点击日志训练无偏排名模型的现代系统中被广泛应用。关键在于明确地建模用户行为的生成过程，并基于检验假设对点击数据进行拟合。先前的研究经验性地发现只要点击完全拟合，大多数情况下可以恢复出真实潜在相关性。然而，我们证明并非总是能够实现这一点，从而导致排名性能显著降低。在本工作中，我们旨在回答真实相关性是否能够从点击数据恢复出来的问题，这是ULTR领域的一个基本问题。我们首先将一个排名模型定义为可识别的，如果它可以恢复出真实相关性，最多只有一个缩放变换，这对于成对排名目标来说已足够。然后，我们探讨了一个等价的可识别条件，可以新颖地表达为一个图连通性测试问题：当且仅当一个图（即可识别性图）连通时，该排名模型是可识别的。

    The application of Unbiased Learning to Rank (ULTR) is widespread in modern systems for training unbiased ranking models from biased click logs. The key is to explicitly model a generation process for user behavior and fit click data based on examination hypothesis. Previous research found empirically that the true latent relevance can be recovered in most cases as long as the clicks are perfectly fitted. However, we demonstrate that this is not always achievable, resulting in a significant reduction in ranking performance. In this work, we aim to answer if or when the true relevance can be recovered from click data, which is a foundation issue for ULTR field. We first define a ranking model as identifiable if it can recover the true relevance up to a scaling transformation, which is enough for pairwise ranking objective. Then we explore an equivalent condition for identifiability that can be novely expressed as a graph connectivity test problem: if and only if a graph (namely identifi
    

