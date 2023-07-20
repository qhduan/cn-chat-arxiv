# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [UniMatch: A Unified User-Item Matching Framework for the Multi-purpose Merchant Marketing.](http://arxiv.org/abs/2307.09989) | UniMatch是一个统一的用户-物品匹配框架，通过使用一个模型同时进行物品推荐和用户定位，减少了商家购买多个机器学习模型的成本。利用多项分布建模用户-物品交互矩阵，并通过双向偏差校正的损失函数指导模型学习用户-物品联合概率，实现了优化。 |
| [^2] | [Our Model Achieves Excellent Performance on MovieLens: What Does it Mean?.](http://arxiv.org/abs/2307.09985) | 该论文通过对MovieLens数据集的分析，发现用户与该平台的交互在不同阶段存在显著差异，并且用户交互受到平台推荐算法推荐的候选电影的影响。 |
| [^3] | [Who Provides the Largest Megaphone? The Role of Google News in Promoting Russian State-Affiliated News Sources.](http://arxiv.org/abs/2307.09834) | 本研究调查了谷歌新闻与俄罗斯国有附属新闻来源的关系，发现在全球范围内，谷歌在传播国家赞助信息方面起到了重要的作用。 |
| [^4] | [DisCover: Disentangled Music Representation Learning for Cover Song Identification.](http://arxiv.org/abs/2307.09775) | DisCover是一个针对Cover歌曲识别的解缠绕音乐表示学习框架，旨在解缠特定版本因素和无版本因素，从而使模型可以学习到适用于未见查询歌曲的不变音乐表示。 |
| [^5] | [Information Retrieval Meets Large Language Models: A Strategic Report from Chinese IR Community.](http://arxiv.org/abs/2307.09751) | 本论文总结了中国信息检索界关于信息检索与大型语言模型相结合的战略报告。大型语言模型在文本理解、生成和知识推理方面具有出色能力，为信息检索研究开辟了新的方向。此外，IR模型、LLM和人类之间的协同关系形成了一种更强大的信息寻求技术范式。然而，该领域仍面临计算成本、可信度、领域特定限制和伦理考虑等挑战。 |
| [^6] | [Amazon-M2: A Multilingual Multi-locale Shopping Session Dataset for Recommendation and Text Generation.](http://arxiv.org/abs/2307.09688) | Amazon-M2是一个多语言多区域购物会话数据集，可以增强个性化推荐和理解用户偏好能力。 |
| [^7] | [PubMed and Beyond: Recent Advances and Best Practices in Biomedical Literature Search.](http://arxiv.org/abs/2307.09683) | 本论文总结了生物医学文献检索领域的最新进展和最佳实践，介绍了针对不同生物医学信息需求的文献检索工具，并旨在帮助读者高效满足其信息需求。 |
| [^8] | [Retrieving Continuous Time Event Sequences using Neural Temporal Point Processes with Learnable Hashing.](http://arxiv.org/abs/2307.09613) | 提出了一个名为NeuroSeqRet的模型，用于解决大规模检索连续时间事件序列的任务。通过使用可学习的哈希和神经时间点过程，该模型能够为输入的查询序列返回一个相关序列的排序列表。 |
| [^9] | [Algorithmic neutrality.](http://arxiv.org/abs/2303.05103) | 研究算法中立性以及与算法偏见的关系，以搜索引擎为案例研究，得出搜索中立性是不可能的结论。 |
| [^10] | [Trustworthy Recommender Systems.](http://arxiv.org/abs/2208.06265) | 可信度推荐系统研究已经从以准确性为导向转变为以透明、公正、稳健性为特点的可信度推荐系统。本文提供了可信度推荐系统领域的文献综述和讨论。 |

# 详细

[^1]: UniMatch:一个统一的用户-物品匹配框架，用于多用途商家营销

    UniMatch: A Unified User-Item Matching Framework for the Multi-purpose Merchant Marketing. (arXiv:2307.09989v1 [cs.IR])

    [http://arxiv.org/abs/2307.09989](http://arxiv.org/abs/2307.09989)

    UniMatch是一个统一的用户-物品匹配框架，通过使用一个模型同时进行物品推荐和用户定位，减少了商家购买多个机器学习模型的成本。利用多项分布建模用户-物品交互矩阵，并通过双向偏差校正的损失函数指导模型学习用户-物品联合概率，实现了优化。

    

    在使用云服务进行私有领域营销时，商家通常需要为多个营销目的购买不同的机器学习模型，导致成本非常高。我们提出了一个统一的用户-物品匹配框架，可以通过一个模型同时进行物品推荐和用户定位。我们通过对用户-物品交互矩阵进行多项分布建模的实验验证了上述并发建模的可行性，并提出了一个双向偏差校正的NCE loss来实现。提出的损失函数通过纠正由于批次内负采样引起的用户和物品偏差，引导模型学习用户-物品联合概率p(u,i)，而不是条件概率p(i|u)或p(u|i)。此外，我们的框架对模型架构具有灵活的适应性。广泛的实验证明，我们的框架可以显著提高性能。

    When doing private domain marketing with cloud services, the merchants usually have to purchase different machine learning models for the multiple marketing purposes, leading to a very high cost. We present a unified user-item matching framework to simultaneously conduct item recommendation and user targeting with just one model. We empirically demonstrate that the above concurrent modeling is viable via modeling the user-item interaction matrix with the multinomial distribution, and propose a bidirectional bias-corrected NCE loss for the implementation. The proposed loss function guides the model to learn the user-item joint probability $p(u,i)$ instead of the conditional probability $p(i|u)$ or $p(u|i)$ through correcting both the users and items' biases caused by the in-batch negative sampling. In addition, our framework is model-agnostic enabling a flexible adaptation of different model architectures. Extensive experiments demonstrate that our framework results in significant perfo
    
[^2]: 我们的模型在MovieLens上取得了出色的表现：这意味着什么？

    Our Model Achieves Excellent Performance on MovieLens: What Does it Mean?. (arXiv:2307.09985v1 [cs.IR])

    [http://arxiv.org/abs/2307.09985](http://arxiv.org/abs/2307.09985)

    该论文通过对MovieLens数据集的分析，发现用户与该平台的交互在不同阶段存在显著差异，并且用户交互受到平台推荐算法推荐的候选电影的影响。

    

    推荐系统评估的典型基准数据集是在某一时间段内在平台上生成的用户-物品交互数据。交互生成机制部分解释了为什么用户与物品进行交互（如喜欢、购买、评分）以及特定交互发生的背景。在本研究中，我们对MovieLens数据集进行了细致的分析，并解释了使用该数据集进行评估推荐算法时可能的影响。我们从分析中得出了一些主要发现。首先，在用户与MovieLens平台交互的不同阶段存在显著差异。早期交互在很大程度上定义了用户画像，影响了后续的交互。其次，用户交互受到平台内部推荐算法推荐的候选电影的很大影响。删除靠近最后几次交互的交互会对结果产生较大影响。

    A typical benchmark dataset for recommender system (RecSys) evaluation consists of user-item interactions generated on a platform within a time period. The interaction generation mechanism partially explains why a user interacts with (e.g.,like, purchase, rate) an item, and the context of when a particular interaction happened. In this study, we conduct a meticulous analysis on the MovieLens dataset and explain the potential impact on using the dataset for evaluating recommendation algorithms. We make a few main findings from our analysis. First, there are significant differences in user interactions at the different stages when a user interacts with the MovieLens platform. The early interactions largely define the user portrait which affect the subsequent interactions. Second, user interactions are highly affected by the candidate movies that are recommended by the platform's internal recommendation algorithm(s). Removal of interactions that happen nearer to the last few interactions 
    
[^3]: 谁提供了最大的扩音器？谷歌新闻在促进俄罗斯国有附属新闻来源中的角色。

    Who Provides the Largest Megaphone? The Role of Google News in Promoting Russian State-Affiliated News Sources. (arXiv:2307.09834v1 [cs.IR])

    [http://arxiv.org/abs/2307.09834](http://arxiv.org/abs/2307.09834)

    本研究调查了谷歌新闻与俄罗斯国有附属新闻来源的关系，发现在全球范围内，谷歌在传播国家赞助信息方面起到了重要的作用。

    

    互联网不仅数字化了信息，还使全球范围内的信息获取变得民主化。这种逐步但具有突破性的在线信息传播方式导致搜索引擎在塑造人类知识获取方面发挥越来越重要的作用。当互联网用户输入查询时，搜索引擎会对数百亿个可能的网页进行排序以确定要显示的内容。谷歌在搜索引擎市场占据主导地位，在过去十年中，谷歌搜索在全球范围内的市场份额超过80％。只有在俄罗斯和中国，竞争对手们在市场份额方面超过了谷歌，大约60％的俄罗斯互联网用户更喜欢使用Yandex（而不是Google），截至2022年，80％以上的中国互联网用户使用Baidu。尽管在互联网搜索提供商方面存在长期的地区差异，但有限的研究显示了这些提供商在传播国家赞助信息方面的比较情况。

    The Internet has not only digitized but also democratized information access across the globe. This gradual but path-breaking move to online information propagation has resulted in search engines playing an increasingly prominent role in shaping access to human knowledge. When an Internet user enters a query, the search engine sorts through the hundreds of billions of possible webpages to determine what to show. Google dominates the search engine market, with Google Search surpassing 80% market share globally every year of the last decade. Only in Russia and China do Google competitors claim more market share, with approximately 60% of Internet users in Russia preferring Yandex (compared to 40% in favor of Google) and more than 80% of China's Internet users accessing Baidu as of 2022. Notwithstanding this long-standing regional variation in Internet search providers, there is limited research showing how these providers compare in terms of propagating state-sponsored information. Our s
    
[^4]: DisCover: 针对Cover歌曲识别的解缠绕音乐表示学习

    DisCover: Disentangled Music Representation Learning for Cover Song Identification. (arXiv:2307.09775v1 [cs.IR])

    [http://arxiv.org/abs/2307.09775](http://arxiv.org/abs/2307.09775)

    DisCover是一个针对Cover歌曲识别的解缠绕音乐表示学习框架，旨在解缠特定版本因素和无版本因素，从而使模型可以学习到适用于未见查询歌曲的不变音乐表示。

    

    在音乐信息检索（MIR）领域中，Cover歌曲识别（CSI）是一项具有挑战性的任务，旨在从大量的歌曲库中识别出查询歌曲的Cover版本。现有的方法由于建模中特定版本因素和无版本因素的缠绕性，仍然存在高内部和外部歌曲之间的变异性和相关性。在本研究中，我们的目标是解缠出特定版本因素和无版本因素，这能够使模型更容易学习到适用于未见查询歌曲的不变音乐表示。我们使用因果图技术以解缠的视角分析了CSI任务，并确定了影响不变学习的内部版本和外部版本效应。为了阻止这些效应，我们提出了用于CSI的解缠绕音乐表示学习框架（DisCover）。DisCover包括两个关键组件：（1）知识引导的解缠模块（KDM）和（2）基于梯度的对抗解缠。

    In the field of music information retrieval (MIR), cover song identification (CSI) is a challenging task that aims to identify cover versions of a query song from a massive collection. Existing works still suffer from high intra-song variances and inter-song correlations, due to the entangled nature of version-specific and version-invariant factors in their modeling. In this work, we set the goal of disentangling version-specific and version-invariant factors, which could make it easier for the model to learn invariant music representations for unseen query songs. We analyze the CSI task in a disentanglement view with the causal graph technique, and identify the intra-version and inter-version effects biasing the invariant learning. To block these effects, we propose the disentangled music representation learning framework (DisCover) for CSI. DisCover consists of two critical components: (1) Knowledge-guided Disentanglement Module (KDM) and (2) Gradient-based Adversarial Disentanglemen
    
[^5]: 信息检索遇上大型语言模型：中国信息检索界的战略报告

    Information Retrieval Meets Large Language Models: A Strategic Report from Chinese IR Community. (arXiv:2307.09751v1 [cs.IR])

    [http://arxiv.org/abs/2307.09751](http://arxiv.org/abs/2307.09751)

    本论文总结了中国信息检索界关于信息检索与大型语言模型相结合的战略报告。大型语言模型在文本理解、生成和知识推理方面具有出色能力，为信息检索研究开辟了新的方向。此外，IR模型、LLM和人类之间的协同关系形成了一种更强大的信息寻求技术范式。然而，该领域仍面临计算成本、可信度、领域特定限制和伦理考虑等挑战。

    

    信息检索（IR）领域已经取得了显著的发展，超越了传统搜索，以满足多样化的用户信息需求。最近，大型语言模型（LLM）在文本理解、生成和知识推理方面展示了出色的能力，为IR研究开辟了新的契机。LLM不仅能够促进生成式检索，还提供了改进的用户理解、模型评估和用户系统交互方案。更重要的是，IR模型、LLM和人类之间的协同关系构成了一种更强大的信息寻求技术范式。IR模型提供实时和相关的信息，LLM贡献内部知识，而人类在信息服务的可靠性方面起着需求者和评估者的中心作用。然而，仍然存在着一些重要挑战，包括计算成本、可信度问题、领域特定限制和伦理考虑。

    The research field of Information Retrieval (IR) has evolved significantly, expanding beyond traditional search to meet diverse user information needs. Recently, Large Language Models (LLMs) have demonstrated exceptional capabilities in text understanding, generation, and knowledge inference, opening up exciting avenues for IR research. LLMs not only facilitate generative retrieval but also offer improved solutions for user understanding, model evaluation, and user-system interactions. More importantly, the synergistic relationship among IR models, LLMs, and humans forms a new technical paradigm that is more powerful for information seeking. IR models provide real-time and relevant information, LLMs contribute internal knowledge, and humans play a central role of demanders and evaluators to the reliability of information services. Nevertheless, significant challenges exist, including computational costs, credibility concerns, domain-specific limitations, and ethical considerations. To 
    
[^6]: Amazon-M2: 一个用于推荐和文本生成的多语言多区域购物会话数据集

    Amazon-M2: A Multilingual Multi-locale Shopping Session Dataset for Recommendation and Text Generation. (arXiv:2307.09688v1 [cs.IR])

    [http://arxiv.org/abs/2307.09688](http://arxiv.org/abs/2307.09688)

    Amazon-M2是一个多语言多区域购物会话数据集，可以增强个性化推荐和理解用户偏好能力。

    

    对于电子商务来说，建模客户购物意图是一个重要的任务，因为它直接影响用户体验和参与度。因此，准确理解客户的偏好对于提供个性化推荐至关重要。基于会话的推荐技术利用客户会话数据来预测他们的下一次互动，已经越来越受到欢迎。然而，现有的会话数据集在项目属性、用户多样性和数据集规模方面存在局限性。因此，它们不能全面地捕捉用户行为和偏好的谱系。为了弥补这一差距，我们提出了Amazon Multilingual Multi-locale Shopping Session Dataset，即Amazon-M2。它是第一个由来自六个不同区域的数百万用户会话组成的多语言数据集，其中产品的主要语言是英语、德语、日语、法语、意大利语和西班牙语。值得注意的是，这个数据集可以帮助我们增强个性化和理解用户偏好能力。

    Modeling customer shopping intentions is a crucial task for e-commerce, as it directly impacts user experience and engagement. Thus, accurately understanding customer preferences is essential for providing personalized recommendations. Session-based recommendation, which utilizes customer session data to predict their next interaction, has become increasingly popular. However, existing session datasets have limitations in terms of item attributes, user diversity, and dataset scale. As a result, they cannot comprehensively capture the spectrum of user behaviors and preferences. To bridge this gap, we present the Amazon Multilingual Multi-locale Shopping Session Dataset, namely Amazon-M2. It is the first multilingual dataset consisting of millions of user sessions from six different locales, where the major languages of products are English, German, Japanese, French, Italian, and Spanish. Remarkably, the dataset can help us enhance personalization and understanding of user preferences, w
    
[^7]: PubMed及其他：生物医学文献检索的最新进展和最佳实践

    PubMed and Beyond: Recent Advances and Best Practices in Biomedical Literature Search. (arXiv:2307.09683v1 [cs.IR])

    [http://arxiv.org/abs/2307.09683](http://arxiv.org/abs/2307.09683)

    本论文总结了生物医学文献检索领域的最新进展和最佳实践，介绍了针对不同生物医学信息需求的文献检索工具，并旨在帮助读者高效满足其信息需求。

    

    生物医学研究产生了丰富的信息，其中很多只能通过文献获取。因此，文献检索是临床和生物医学研究中建立在先前知识基础上的重要工具。尽管人工智能的最新进展已经将功能扩展到了超越基于关键字的搜索，但这些进展可能对临床医生和研究人员来说还比较陌生。为了解决这个问题，本文介绍了一些特定于生物医学领域信息需求的文献检索工具，旨在帮助读者高效地满足他们的信息需求。我们首先对广泛使用的PubMed搜索引擎进行了讨论，包括最新的改进和仍然存在的挑战。然后，我们描述了五种特定信息需求的文献检索工具：1.为循证医学寻找高质量临床研究。2.为精准医学和基因组学检索基因相关信息。3.根据意义搜索。

    Biomedical research yields a wealth of information, much of which is only accessible through the literature. Consequently, literature search is an essential tool for building on prior knowledge in clinical and biomedical research. Although recent improvements in artificial intelligence have expanded functionality beyond keyword-based search, these advances may be unfamiliar to clinicians and researchers. In response, we present a survey of literature search tools tailored to both general and specific information needs in biomedicine, with the objective of helping readers efficiently fulfill their information needs. We first examine the widely used PubMed search engine, discussing recent improvements and continued challenges. We then describe literature search tools catering to five specific information needs: 1. Identifying high-quality clinical research for evidence-based medicine. 2. Retrieving gene-related information for precision medicine and genomics. 3. Searching by meaning, inc
    
[^8]: 使用可学习的哈希的神经时间点过程检索连续时间事件序列

    Retrieving Continuous Time Event Sequences using Neural Temporal Point Processes with Learnable Hashing. (arXiv:2307.09613v1 [cs.LG])

    [http://arxiv.org/abs/2307.09613](http://arxiv.org/abs/2307.09613)

    提出了一个名为NeuroSeqRet的模型，用于解决大规模检索连续时间事件序列的任务。通过使用可学习的哈希和神经时间点过程，该模型能够为输入的查询序列返回一个相关序列的排序列表。

    

    时间序列已经在各种实际应用中变得普遍。因此，以连续时间事件序列（CTES）形式产生的数据量在过去几年中呈指数增长。因此，针对CTES数据集的当前研究的重要部分是设计用于解决下游任务（如下一个事件预测、长期预测、序列分类等）的模型。使用标记的时间点过程（MTPP）进行预测建模的最新发展使得能够准确地对涉及CTES的几个实际应用进行表征。然而，由于这些CTES数据集的复杂性，过去的文献忽视了大规模检索时间序列的任务。具体而言，通过CTES检索，我们指的是对于一个输入查询序列，检索系统必须从一个大的语料库中返回一个相关序列的排序列表。为了解决这个问题，我们提出了NeuroSeqRet，这是一个典型的

    Temporal sequences have become pervasive in various real-world applications. Consequently, the volume of data generated in the form of continuous time-event sequence(s) or CTES(s) has increased exponentially in the past few years. Thus, a significant fraction of the ongoing research on CTES datasets involves designing models to address downstream tasks such as next-event prediction, long-term forecasting, sequence classification etc. The recent developments in predictive modeling using marked temporal point processes (MTPP) have enabled an accurate characterization of several real-world applications involving the CTESs. However, due to the complex nature of these CTES datasets, the task of large-scale retrieval of temporal sequences has been overlooked by the past literature. In detail, by CTES retrieval we mean that for an input query sequence, a retrieval system must return a ranked list of relevant sequences from a large corpus. To tackle this, we propose NeuroSeqRet, a first-of-its
    
[^9]: 算法中立性

    Algorithmic neutrality. (arXiv:2303.05103v2 [cs.CY] UPDATED)

    [http://arxiv.org/abs/2303.05103](http://arxiv.org/abs/2303.05103)

    研究算法中立性以及与算法偏见的关系，以搜索引擎为案例研究，得出搜索中立性是不可能的结论。

    

    偏见影响着越来越多掌控我们生活的算法。预测性警务系统错误地高估有色人种社区的犯罪率；招聘算法削弱了合格的女性候选人的机会；人脸识别软件难以识别黑皮肤的面部。算法偏见已经受到了重视，相比之下，算法中立性却基本被忽视了。算法中立性是我的研究主题。我提出了三个问题。算法中立性是什么？算法中立性是否可能？当我们考虑算法中立性时，我们可以从算法偏见中学到什么？为了具体回答这些问题，我选择了一个案例研究：搜索引擎。借鉴关于科学中立性的研究，我认为只有当搜索引擎的排名不受某些价值观的影响时，搜索引擎才是中立的，比如政治意识形态或搜索引擎运营商的经济利益。我认为搜索中立性是不可能的。

    Bias infects the algorithms that wield increasing control over our lives. Predictive policing systems overestimate crime in communities of color; hiring algorithms dock qualified female candidates; and facial recognition software struggles to recognize dark-skinned faces. Algorithmic bias has received significant attention. Algorithmic neutrality, in contrast, has been largely neglected. Algorithmic neutrality is my topic. I take up three questions. What is algorithmic neutrality? Is algorithmic neutrality possible? When we have algorithmic neutrality in mind, what can we learn about algorithmic bias? To answer these questions in concrete terms, I work with a case study: search engines. Drawing on work about neutrality in science, I say that a search engine is neutral only if certain values -- like political ideologies or the financial interests of the search engine operator -- play no role in how the search engine ranks pages. Search neutrality, I argue, is impossible. Its impossibili
    
[^10]: 可信推荐系统

    Trustworthy Recommender Systems. (arXiv:2208.06265v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2208.06265](http://arxiv.org/abs/2208.06265)

    可信度推荐系统研究已经从以准确性为导向转变为以透明、公正、稳健性为特点的可信度推荐系统。本文提供了可信度推荐系统领域的文献综述和讨论。

    

    推荐系统旨在帮助用户从庞大的目录中有效地检索感兴趣的物品。长期以来，研究人员一直致力于开发准确的推荐系统。然而，近年来，推荐系统面临越来越多的威胁，包括来自攻击、系统和用户产生的干扰以及系统的偏见。因此，仅仅关注准确性已经不够，研究必须考虑其他重要因素，如可信度。对于终端用户来说，一个值得信赖的推荐系统不仅要准确，而且还要透明、无偏见、公正，并且对干扰或攻击具有稳健性。这些观察实际上导致了推荐系统研究的范式转变: 从以准确性为导向的推荐系统转向了以可信度为导向的推荐系统。然而，研究人员缺乏对可信度推荐系统领域的文献的系统概述和讨论。因此，本文提供了可信度推荐系统的概述，包括对该新兴且快速发展领域的文献的讨论。

    Recommender systems (RSs) aim to help users to effectively retrieve items of their interests from a large catalogue. For a quite long period of time, researchers and practitioners have been focusing on developing accurate RSs. Recent years have witnessed an increasing number of threats to RSs, coming from attacks, system and user generated noise, system bias. As a result, it has become clear that a strict focus on RS accuracy is limited and the research must consider other important factors, e.g., trustworthiness. For end users, a trustworthy RS (TRS) should not only be accurate, but also transparent, unbiased and fair as well as robust to noise or attacks. These observations actually led to a paradigm shift of the research on RSs: from accuracy-oriented RSs to TRSs. However, researchers lack a systematic overview and discussion of the literature in this novel and fast developing field of TRSs. To this end, in this paper, we provide an overview of TRSs, including a discussion of the mo
    

