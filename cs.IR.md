# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [MerRec: A Large-scale Multipurpose Mercari Dataset for Consumer-to-Consumer Recommendation Systems](https://arxiv.org/abs/2402.14230) | 提出了MerRec，这是首个专门针对C2C推荐而提出的大规模数据集，填补了C2C推荐数据集中物品属性、用户多样性和规模等方面的缺失。 |
| [^2] | [Unlocking the `Why' of Buying: Introducing a New Dataset and Benchmark for Purchase Reason and Post-Purchase Experience](https://arxiv.org/abs/2402.13417) | 引入了一个新的数据集和基准，旨在揭示用户购买决策背后的原因，提出了一个有效的基于LLM的方法来生成高质量、个性化的购买原因解释。 |
| [^3] | [GPT4Table: Can Large Language Models Understand Structured Table Data? A Benchmark and Empirical Study.](http://arxiv.org/abs/2305.13062) | 本文设计了一个基准测试来评估大型语言模型（LLMs）对结构化表格数据的理解能力，并发现不同的输入选择会对性能产生影响。在基准测试的基础上，提出了“自我增强”技术以改善理解能力。 |

# 详细

[^1]: MerRec：用于消费者对消费者推荐系统的大规模多功能Mercari数据集

    MerRec: A Large-scale Multipurpose Mercari Dataset for Consumer-to-Consumer Recommendation Systems

    [https://arxiv.org/abs/2402.14230](https://arxiv.org/abs/2402.14230)

    提出了MerRec，这是首个专门针对C2C推荐而提出的大规模数据集，填补了C2C推荐数据集中物品属性、用户多样性和规模等方面的缺失。

    

    在不断发展的电子商务领域中，推荐系统至关重要地塑造了用户体验和参与度。消费者对消费者（C2C）推荐系统的崛起，以其灵活性和为客户供应商提供易于访问的特点，标志着一个重要趋势。然而，学术关注主要集中在商家对消费者（B2C）模型上，留下了一个空白，即缺乏物品属性、用户多样性和规模的C2C推荐数据集。C2C推荐系统的复杂性进一步突出了用户扮演卖家和买家两种角色的双重性质，引入了一系列不那么统一和多样化的输入。为解决这一问题，我们引入了MerRec，这是第一个专门用于C2C推荐的大规模数据集，源自Mercari电子商务平台，覆盖了2023年6个月内数百万用户和产品。MerRec不仅包括标准特征，如user_id、item_id和session_id

    arXiv:2402.14230v1 Announce Type: cross  Abstract: In the evolving e-commerce field, recommendation systems crucially shape user experience and engagement. The rise of Consumer-to-Consumer (C2C) recommendation systems, noted for their flexibility and ease of access for customer vendors, marks a significant trend. However, the academic focus remains largely on Business-to-Consumer (B2C) models, leaving a gap filled by the limited C2C recommendation datasets that lack in item attributes, user diversity, and scale. The intricacy of C2C recommendation systems is further accentuated by the dual roles users assume as both sellers and buyers, introducing a spectrum of less uniform and varied inputs. Addressing this, we introduce MerRec, the first large-scale dataset specifically for C2C recommendations, sourced from the Mercari e-commerce platform, covering millions of users and products over 6 months in 2023. MerRec not only includes standard features such as user_id, item_id, and session_id
    
[^2]: 解锁购买的“为何”：引入一个新的数据集和购买原因与后购买体验的基准

    Unlocking the `Why' of Buying: Introducing a New Dataset and Benchmark for Purchase Reason and Post-Purchase Experience

    [https://arxiv.org/abs/2402.13417](https://arxiv.org/abs/2402.13417)

    引入了一个新的数据集和基准，旨在揭示用户购买决策背后的原因，提出了一个有效的基于LLM的方法来生成高质量、个性化的购买原因解释。

    

    解释对于提高现代推荐系统中用户信任和理解至关重要。为了构建真正可解释的系统，我们需要能阐明用户为何做出选择的高质量数据集。我们提出了一个新颖的购买原因解释任务。为此，我们引入了一种基于LLM的方法来生成一个由真实用户解释为何做出某些购买决策的文本解释的数据集。我们诱导LLM明确区分用户评论中购买产品背后的原因和购买后的体验。自动化的LLM驱动评估以及小规模人工评估证实了我们方法获取高质量、个性化解释的有效性。我们在两个个性化数据集上对该数据集进行基准测试。

    arXiv:2402.13417v1 Announce Type: new  Abstract: Explanations are crucial for enhancing user trust and understanding within modern recommendation systems. To build truly explainable systems, we need high-quality datasets that elucidate why users make choices. While previous efforts have focused on extracting users' post-purchase sentiment in reviews, they ignore the reasons behind the decision to buy.   In our work, we propose a novel purchase reason explanation task. To this end, we introduce an LLM-based approach to generate a dataset that consists of textual explanations of why real users make certain purchase decisions. We induce LLMs to explicitly distinguish between the reasons behind purchasing a product and the experience after the purchase in a user review. An automated, LLM-driven evaluation, as well as a small scale human evaluation, confirms the effectiveness of our approach to obtaining high-quality, personalized explanations. We benchmark this dataset on two personalized 
    
[^3]: GPT4Table：大型语言模型能理解结构化表格数据吗？一项基准测试和实证研究

    GPT4Table: Can Large Language Models Understand Structured Table Data? A Benchmark and Empirical Study. (arXiv:2305.13062v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2305.13062](http://arxiv.org/abs/2305.13062)

    本文设计了一个基准测试来评估大型语言模型（LLMs）对结构化表格数据的理解能力，并发现不同的输入选择会对性能产生影响。在基准测试的基础上，提出了“自我增强”技术以改善理解能力。

    

    大型语言模型（LLMs）作为少样本推理器来解决与自然语言相关的任务越来越具吸引力。然而，关于LLMs对结构化数据（例如表格）的理解程度还有很多需要学习的地方。尽管可以使用表格序列化作为LLMs的输入，但目前还缺乏对LLMs是否真正能够理解这类数据的全面研究。本文通过设计一个基准测试来评估LLMs的结构理解能力（SUC）来解决这个问题。我们创建的基准测试包括七个任务，每个任务都有其独特的挑战，例如单元格查找、行检索和大小检测。我们对GPT-3.5和GPT-4进行了一系列评估。我们发现性能因多种输入选择而异，包括表格输入格式、内容顺序、角色提示和分区标记等。根据基准测试评估所得的见解，我们提出了“自我增强”技术以改善性能。

    Large language models (LLMs) are becoming attractive as few-shot reasoners to solve Natural Language (NL)-related tasks. However, there is still much to learn about how well LLMs understand structured data, such as tables. While it is true that tables can be used as inputs to LLMs with serialization, there lack of comprehensive studies examining whether LLMs can truly comprehend such data. In this paper, we try to understand this by designing a benchmark to evaluate the structural understanding capabilities (SUC) of LLMs. The benchmark we create includes seven tasks, each with its own unique challenges, \eg, cell lookup, row retrieval, and size detection. We run a series of evaluations on GPT-3.5 and GPT-4. We discover that the performance varied depending on a number of input choices, including table input format, content order, role prompting, and partition marks. Drawing from the insights gained through the benchmark evaluations, we then propose \textit{self-augmentation} for effect
    

