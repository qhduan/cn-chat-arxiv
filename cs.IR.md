# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Accelerating Retrieval-Augmented Language Model Serving with Speculation.](http://arxiv.org/abs/2401.14021) | 提出了RaLMSpec，这是一个使用推测加速检索增强型语言模型服务的框架，通过推测式检索和批量验证提供了通用的加速效果，并通过进一步优化和并发处理，提高了性能。 |
| [^2] | [Towards 3D Molecule-Text Interpretation in Language Models.](http://arxiv.org/abs/2401.13923) | 提出了一个名为3D-MoLM的模型，通过给语言模型配备一个3D分子编码器，实现了对3D分子-文本的解释和分析，此模型在下游任务上显著优于现有基线。 |
| [^3] | [Integrating Large Language Models into Recommendation via Mutual Augmentation and Adaptive Aggregation.](http://arxiv.org/abs/2401.13870) | 本论文提出了一种将大型语言模型融入推荐系统的新框架，通过互补增强和自适应聚合，充分发挥它们各自的优势，以提升推荐性能。 |
| [^4] | [Algorithmically Curated Lies: How Search Engines Handle Misinformation about US Biolabs in Ukraine.](http://arxiv.org/abs/2401.13832) | 本研究以乌克兰美国生物实验室的虚假信息宣传运动为例，调查了网络搜索引擎如何处理与谣言相关的内容，揭示了算法策划系统容易受到操纵的风险。 |
| [^5] | [Robustness in Fairness against Edge-level Perturbations in GNN-based Recommendation.](http://arxiv.org/abs/2401.13823) | 本文评估了基于图的推荐系统在面对边级扰动攻击时的公平性鲁棒性，发现现有评估协议存在关键缺点。 |
| [^6] | [Longitudinal Sentiment Topic Modelling of Reddit Posts.](http://arxiv.org/abs/2401.13805) | 该研究通过纵向主题建模分析了加拿大四所主要大学学生在Reddit上发布的帖子。研究结果显示，随着时间推移，与心理健康相关的讨论逐渐增加。 |
| [^7] | [Transforming Agriculture with Intelligent Data Management and Insights.](http://arxiv.org/abs/2401.13672) | 使用智能数据管理和洞察力可以解决现代农业面临的增长需求和气候变化等挑战，通过数据创新可以提高农业生产力、可持续性和适应性。 |
| [^8] | [Decentralized Collaborative Learning with Adaptive Reference Data for On-Device POI Recommendation.](http://arxiv.org/abs/2401.13448) | 这项研究提出了一种使用自适应引用数据的去中心化协作学习方法，用于设备上的兴趣点推荐，以解决使用同一引用数据对不同用户产生负面影响的问题。 |
| [^9] | [It's About Time: Incorporating Temporality in Retrieval Augmented Language Models.](http://arxiv.org/abs/2401.13222) | 在大型语言模型中引入时间性是信息检索的关键挑战，目前的检索增强语言模型无法很好地处理时间查询。 |
| [^10] | [SynthTab: Leveraging Synthesized Data for Guitar Tablature Transcription.](http://arxiv.org/abs/2309.09085) | SynthTab是一个利用合成数据的大规模吉他谱转录数据集，解决了现有数据集规模有限的问题，并通过合成音频保持了原始指法、风格和技巧的相符性。 |
| [^11] | [Enhancing Recommender Systems with Large Language Model Reasoning Graphs.](http://arxiv.org/abs/2308.10835) | 本文提出了一种使用大型语言模型（LLMs）构建个性化推理图的方法，通过因果和逻辑推理链接用户的个人资料和行为序列，在提升推荐系统性能的同时实现了更多的逻辑性和可解释性。 |
| [^12] | [The Search for Stability: Learning Dynamics of Strategic Publishers with Initial Documents.](http://arxiv.org/abs/2305.16695) | 本研究在信息检索博弈论模型中提出了相对排名原则（RRP）作为替代排名原则，以达成更稳定的搜索生态系统，并提供了理论和实证证据证明其学习动力学收敛性，同时展示了可能的出版商-用户权衡。 |

# 详细

[^1]: 使用推测加速检索增强型语言模型服务

    Accelerating Retrieval-Augmented Language Model Serving with Speculation. (arXiv:2401.14021v1 [cs.LG])

    [http://arxiv.org/abs/2401.14021](http://arxiv.org/abs/2401.14021)

    提出了RaLMSpec，这是一个使用推测加速检索增强型语言模型服务的框架，通过推测式检索和批量验证提供了通用的加速效果，并通过进一步优化和并发处理，提高了性能。

    

    检索增强型语言模型（RaLM）通过将非参数的知识库与参数化的语言模型相结合，已经展示出解决知识密集型自然语言处理（NLP）任务的潜力。与对完全参数化模型进行微调不同，RaLM在适应最新数据和更好的来源归属机制方面具有低成本的优势。在众多的RaLM方法中，迭代式RaLM由于检索器与语言模型之间更频繁的互动而具有更好的生成质量。尽管有这些好处，迭代式RaLM通常会因为频繁的检索步骤而遇到高开销。为此，我们提出了RaLMSpec，这是一个基于推测的框架，通过推测式检索和批量验证，能够在保持相同模型输出的同时，提供通用加速的效果。通过进一步结合预取、最佳推测步幅调度器和异步验证，RaLMSpec能够自动利用并发性和并行性来最大程度地提高性能。

    Retrieval-augmented language models (RaLM) have demonstrated the potential to solve knowledge-intensive natural language processing (NLP) tasks by combining a non-parametric knowledge base with a parametric language model. Instead of fine-tuning a fully parametric model, RaLM excels at its low-cost adaptation to the latest data and better source attribution mechanisms. Among various RaLM approaches, iterative RaLM delivers a better generation quality due to a more frequent interaction between the retriever and the language model. Despite the benefits, iterative RaLM usually encounters high overheads due to the frequent retrieval step. To this end, we propose RaLMSpec, a speculation-inspired framework that provides generic speed-up over iterative RaLM while preserving the same model outputs through speculative retrieval and batched verification. By further incorporating prefetching, optimal speculation stride scheduler, and asynchronous verification, RaLMSpec can automatically exploit t
    
[^2]: 在语言模型中实现对3D分子-文本的解释

    Towards 3D Molecule-Text Interpretation in Language Models. (arXiv:2401.13923v1 [cs.LG])

    [http://arxiv.org/abs/2401.13923](http://arxiv.org/abs/2401.13923)

    提出了一个名为3D-MoLM的模型，通过给语言模型配备一个3D分子编码器，实现了对3D分子-文本的解释和分析，此模型在下游任务上显著优于现有基线。

    

    语言模型（LMs）在各个领域有着很大的影响。然而，它们对于理解3D分子结构的固有限制极大地限制了它们在生物分子领域的潜力。为了弥补这一差距，我们关注于3D分子-文本解释，并提出3D-MoLM：3D分子语言模型。具体而言，3D-MoLM通过为LM配备一个3D分子编码器，使得LM能够解释和分析3D分子。这种集成是通过一个3D分子-文本投影器实现的，它连接了3D分子编码器的表示空间和LM的输入空间。此外，为了增强3D-MoLM在跨模态分子理解和指令跟随方面的能力，我们精心策划了一个以3D分子为中心的指引调整数据集--3D-MoIT。通过3D分子-文本对齐和3D分子中心的指引调整，3D-MoLM建立了3D分子编码器和LM的集成。它在下游任务上显著超过了现有的基线。

    Language Models (LMs) have greatly influenced diverse domains. However, their inherent limitation in comprehending 3D molecular structures has considerably constrained their potential in the biomolecular domain. To bridge this gap, we focus on 3D molecule-text interpretation, and propose 3D-MoLM: 3D-Molecular Language Modeling. Specifically, 3D-MoLM enables an LM to interpret and analyze 3D molecules by equipping the LM with a 3D molecular encoder. This integration is achieved by a 3D molecule-text projector, bridging the 3D molecular encoder's representation space and the LM's input space. Moreover, to enhance 3D-MoLM's ability of cross-modal molecular understanding and instruction following, we meticulously curated a 3D molecule-centric instruction tuning dataset -- 3D-MoIT. Through 3D molecule-text alignment and 3D molecule-centric instruction tuning, 3D-MoLM establishes an integration of 3D molecular encoder and LM. It significantly surpasses existing baselines on downstream tasks,
    
[^3]: 将大型语言模型融入推荐系统的互补增强和自适应聚合

    Integrating Large Language Models into Recommendation via Mutual Augmentation and Adaptive Aggregation. (arXiv:2401.13870v1 [cs.IR])

    [http://arxiv.org/abs/2401.13870](http://arxiv.org/abs/2401.13870)

    本论文提出了一种将大型语言模型融入推荐系统的新框架，通过互补增强和自适应聚合，充分发挥它们各自的优势，以提升推荐性能。

    

    传统的推荐方法通过利用用户行为中的协同或连续信息取得了显著的进展。最近，由于其在理解和推理文本语义方面的能力，大型语言模型（LLMs）在各个领域中得到了重视，并在推荐系统中发现了其实用性。传统的推荐方法和LLMs各自具有各自的优势和局限性。传统方法擅长挖掘协同信息和建模连续行为，但在数据稀疏和长尾问题方面存在困难。而LLMs则擅长利用丰富的文本上下文，但在挖掘协同或连续信息方面面临挑战。尽管它们各自取得了成功，但在利用它们的联合潜力来提升推荐性能方面存在着显著差距。在本文中，我们介绍了一个通用的、与模型无关的框架，称为大型语言模型互补增强和自适应聚合。

    Conventional recommendation methods have achieved notable advancements by harnessing collaborative or sequential information from user behavior. Recently, large language models (LLMs) have gained prominence for their capabilities in understanding and reasoning over textual semantics, and have found utility in various domains, including recommendation. Conventional recommendation methods and LLMs each have their strengths and weaknesses. While conventional methods excel at mining collaborative information and modeling sequential behavior, they struggle with data sparsity and the long-tail problem. LLMs, on the other hand, are proficient at utilizing rich textual contexts but face challenges in mining collaborative or sequential information. Despite their individual successes, there is a significant gap in leveraging their combined potential to enhance recommendation performance.  In this paper, we introduce a general and model-agnostic framework known as \textbf{L}arge \textbf{la}nguage
    
[^4]: 算法策划的谎言：搜索引擎如何处理有关乌克兰美国生物实验室的虚假信息

    Algorithmically Curated Lies: How Search Engines Handle Misinformation about US Biolabs in Ukraine. (arXiv:2401.13832v1 [cs.IR])

    [http://arxiv.org/abs/2401.13832](http://arxiv.org/abs/2401.13832)

    本研究以乌克兰美国生物实验室的虚假信息宣传运动为例，调查了网络搜索引擎如何处理与谣言相关的内容，揭示了算法策划系统容易受到操纵的风险。

    

    在线内容的增长促使采用算法系统进行信息策划。这些系统包括网络搜索引擎和推荐系统，对帮助用户了解重要社会发展具有重要作用。然而，与新闻编辑不同，算法信息策划系统（AICSs）容易受到各种形式的失效的影响，这使它们容易受到操纵。操纵的风险在于，AICSs必须处理有关虚假主张的信息，这些主张是专制政权的宣传活动的基础。通过以俄罗斯有关乌克兰美国生物实验室的虚假信息宣传运动为案例研究，我们调查了最常用的AICSs形式之一——即网络搜索引擎如何进行谣言相关内容的策划。为此，我们在2020年6月使用虚拟基于代理的算法进行了对Google、Bing和Yandex搜索结果的审核。

    The growing volume of online content prompts the need for adopting algorithmic systems of information curation. These systems range from web search engines to recommender systems and are integral for helping users stay informed about important societal developments. However, unlike journalistic editing the algorithmic information curation systems (AICSs) are known to be subject to different forms of malperformance which make them vulnerable to possible manipulation. The risk of manipulation is particularly prominent in the case when AICSs have to deal with information about false claims that underpin propaganda campaigns of authoritarian regimes. Using as a case study of the Russian disinformation campaign concerning the US biolabs in Ukraine, we investigate how one of the most commonly used forms of AICSs - i.e. web search engines - curate misinformation-related content. For this aim, we conduct virtual agent-based algorithm audits of Google, Bing, and Yandex search outputs in June 20
    
[^5]: GNN-based推荐中对边级扰动的公平性的鲁棒性

    Robustness in Fairness against Edge-level Perturbations in GNN-based Recommendation. (arXiv:2401.13823v1 [cs.IR])

    [http://arxiv.org/abs/2401.13823](http://arxiv.org/abs/2401.13823)

    本文评估了基于图的推荐系统在面对边级扰动攻击时的公平性鲁棒性，发现现有评估协议存在关键缺点。

    

    推荐领域的努力从单纯强调效用转向考虑超出效用因素，如公平性和鲁棒性。推荐模型的鲁棒性通常与其在遭受攻击时维持原始效用的能力相关。有限的研究探讨了在攻击场景下推荐模型在公平性方面的鲁棒性，例如跨组绩效的平等。本文旨在评估基于图的推荐系统在面对基于边级扰动攻击时，与公平性相关的鲁棒性。为此，我们考虑了四种不同的公平性运作模式，包括消费者和提供者的观点。对三个数据集进行的实验揭示了扰动对目标公平性概念的影响，揭示了现有鲁棒性评估协议中的关键缺点。例如，我们观察到扰动会更大地影响消费者公平性。

    Efforts in the recommendation community are shifting from the sole emphasis on utility to considering beyond-utility factors, such as fairness and robustness. Robustness of recommendation models is typically linked to their ability to maintain the original utility when subjected to attacks. Limited research has explored the robustness of a recommendation model in terms of fairness, e.g., the parity in performance across groups, under attack scenarios. In this paper, we aim to assess the robustness of graph-based recommender systems concerning fairness, when exposed to attacks based on edge-level perturbations. To this end, we considered four different fairness operationalizations, including both consumer and provider perspectives. Experiments on three datasets shed light on the impact of perturbations on the targeted fairness notion, uncovering key shortcomings in existing evaluation protocols for robustness. As an example, we observed perturbations affect consumer fairness on a higher
    
[^6]: Reddit帖子的纵向情绪主题建模研究

    Longitudinal Sentiment Topic Modelling of Reddit Posts. (arXiv:2401.13805v1 [cs.SI])

    [http://arxiv.org/abs/2401.13805](http://arxiv.org/abs/2401.13805)

    该研究通过纵向主题建模分析了加拿大四所主要大学学生在Reddit上发布的帖子。研究结果显示，随着时间推移，与心理健康相关的讨论逐渐增加。

    

    在这项研究中，我们分析了四所加拿大主要大学学生撰写的Reddit帖子。通过对帖子文本数据进行纵向主题建模，我们评估情绪调性并揭示主要主题和讨论。我们的研究重点关注2020年至2023年的四年时间段，涵盖了COVID-19大流行及其后续年份。我们的结果突出了与心理健康相关的讨论逐渐增加。

    In this study, we analyze texts of Reddit posts written by students of four major Canadian universities. We gauge the emotional tone and uncover prevailing themes and discussions through longitudinal topic modeling of posts textual data. Our study focuses on four years, 2020-2023, covering COVID-19 pandemic and after pandemic years. Our results highlight a gradual uptick in discussions related to mental health.
    
[^7]: 用智能数据管理和洞察力改变农业

    Transforming Agriculture with Intelligent Data Management and Insights. (arXiv:2401.13672v1 [cs.DB])

    [http://arxiv.org/abs/2401.13672](http://arxiv.org/abs/2401.13672)

    使用智能数据管理和洞察力可以解决现代农业面临的增长需求和气候变化等挑战，通过数据创新可以提高农业生产力、可持续性和适应性。

    

    现代农业面临着在气候变化和自然资源减少的约束下，在人口增长的情况下满足粮食、燃料、饲料和纤维的增加需求的巨大挑战。急需数据创新来确保和提高农业生态系统的生产力、可持续性和适应性。随着各种传感器和物联网设备变得越来越可用、可负担得起、可靠、稳定，可以进行多时空尺度、实时和高分辨率的数据收集、整合和分析。与此同时，庞大的数据量对数据存储和分析构成了巨大挑战，科学家们常规的数据管理和分析实践越来越低效。此外，来自不同学科的数据，如基因组学、表型学、环境学、农学和社会经济学，可能高度异质。

    Modern agriculture faces grand challenges to meet increased demands for food, fuel, feed, and fiber with population growth under the constraints of climate change and dwindling natural resources. Data innovation is urgently required to secure and improve the productivity, sustainability, and resilience of our agroecosystems. As various sensors and Internet of Things (IoT) instrumentation become more available, affordable, reliable, and stable, it has become possible to conduct data collection, integration, and analysis at multiple temporal and spatial scales, in real-time, and with high resolutions. At the same time, the sheer amount of data poses a great challenge to data storage and analysis, and the \textit{de facto} data management and analysis practices adopted by scientists have become increasingly inefficient. Additionally, the data generated from different disciplines, such as genomics, phenomics, environment, agronomy, and socioeconomic, can be highly heterogeneous. That is, d
    
[^8]: 使用自适应引用数据的去中心化协作学习进行设备上的兴趣点推荐

    Decentralized Collaborative Learning with Adaptive Reference Data for On-Device POI Recommendation. (arXiv:2401.13448v1 [cs.IR])

    [http://arxiv.org/abs/2401.13448](http://arxiv.org/abs/2401.13448)

    这项研究提出了一种使用自适应引用数据的去中心化协作学习方法，用于设备上的兴趣点推荐，以解决使用同一引用数据对不同用户产生负面影响的问题。

    

    在基于位置的社交网络中，兴趣点（POI）推荐帮助用户发现有趣的地方。为了保护隐私和减少服务器依赖，从基于云的模型转向设备上的推荐是一个趋势。由于个别设备上的本地用户-项目交互数据稀缺，仅依赖本地数据是不足够的。协作学习（CL）兴起，促进用户之间的模型共享，其中引用数据作为中介，使用户能够在不直接共享私有数据或参数的情况下交换他们的软决策，确保隐私并从协作中受益。然而，现有的基于协作学习的推荐通常为所有用户使用同一个引用数据。对一个用户有价值的引用数据可能对另一个用户有害，鉴于用户偏好的多样性。用户可能不会对他们兴趣范围之外的项目提供有意义的软决策。因此，为所有协作使用相同的引用数据可能会阻碍协作推荐的效果。

    In Location-based Social Networks, Point-of-Interest (POI) recommendation helps users discover interesting places. There is a trend to move from the cloud-based model to on-device recommendations for privacy protection and reduced server reliance. Due to the scarcity of local user-item interactions on individual devices, solely relying on local instances is not adequate. Collaborative Learning (CL) emerges to promote model sharing among users, where reference data is an intermediary that allows users to exchange their soft decisions without directly sharing their private data or parameters, ensuring privacy and benefiting from collaboration. However, existing CL-based recommendations typically use a single reference for all users. Reference data valuable for one user might be harmful to another, given diverse user preferences. Users may not offer meaningful soft decisions on items outside their interest scope. Consequently, using the same reference data for all collaborations can imped
    
[^9]: 关于时间的重要性：在检索增强语言模型中引入时间性

    It's About Time: Incorporating Temporality in Retrieval Augmented Language Models. (arXiv:2401.13222v1 [cs.IR])

    [http://arxiv.org/abs/2401.13222](http://arxiv.org/abs/2401.13222)

    在大型语言模型中引入时间性是信息检索的关键挑战，目前的检索增强语言模型无法很好地处理时间查询。

    

    网络作为全球的知识存储库，被数十亿人用于搜索信息。确保用户能够获得最相关和最新的信息是信息检索面临的关键挑战，尤其是在存在来自不同时间点的多个版本的网络内容的情况下。最近，这个挑战变得更加复杂，原因是对维基百科或网络内容进行训练的问答工具的增加使用，这些工具由大型语言模型（LLM）驱动，而这些模型被发现会虚构信息，且在处理时间信息方面存在困难。即使是引入文档数据库以减少LLM虚构的检索增强语言模型（RALM）也无法正确处理时间查询。这导致RALM在回答类似“谁赢得了温网冠军？”的查询时，只会检索与温网相关的文档内容，而不完整。

    The web serves as a global repository of knowledge, used by billions of people to search for information. Ensuring that users receive the most relevant and up-to-date information, especially in the presence of multiple versions of web content from different time points remains a critical challenge for information retrieval. This challenge has recently been compounded by the increased use of question answering tools trained on Wikipedia or web content and powered by large language models (LLMs) \citep{chatgpt} which have been found to make up information (or hallucinate), and in addition have been shown to struggle with the temporal dimensions of information. Even Retriever Augmented Language Models (RALMs) which incorporate a document database to reduce LLM hallucination are unable to handle temporal queries correctly. This leads to instances where RALMs respond to queries such as "Who won the Wimbledon Championship?", by retrieving document passages related to Wimbledon but without th
    
[^10]: SynthTab: 利用合成数据进行吉他谱转录

    SynthTab: Leveraging Synthesized Data for Guitar Tablature Transcription. (arXiv:2309.09085v2 [cs.SD] UPDATED)

    [http://arxiv.org/abs/2309.09085](http://arxiv.org/abs/2309.09085)

    SynthTab是一个利用合成数据的大规模吉他谱转录数据集，解决了现有数据集规模有限的问题，并通过合成音频保持了原始指法、风格和技巧的相符性。

    

    吉他谱是吉他手广泛使用的一种音乐符号。它不仅捕捉了一首乐曲的音乐内容，还包括了在乐器上的实施和装饰。吉他谱转录（GTT）是一项重要的任务，在音乐教育和娱乐领域有广泛应用。现有的数据集在规模和范围上都有限，导致基于这些数据集训练的最先进的GTT模型容易过拟合，并且在跨数据集的泛化中失败。为解决这个问题，我们开发了一种方法来合成SynthTab，这是一个利用多个商用吉他插件合成的大规模吉他谱转录数据集。该数据集基于DadaGP的吉他谱构建，DadaGP提供了我们希望转录的吉他谱的庞大收藏和特定程度。所提出的合成流程可产生与原始指法、风格和技巧在音色上相符的音频。

    Guitar tablature is a form of music notation widely used among guitarists. It captures not only the musical content of a piece, but also its implementation and ornamentation on the instrument. Guitar Tablature Transcription (GTT) is an important task with broad applications in music education and entertainment. Existing datasets are limited in size and scope, causing state-of-the-art GTT models trained on such datasets to suffer from overfitting and to fail in generalization across datasets. To address this issue, we developed a methodology for synthesizing SynthTab, a large-scale guitar tablature transcription dataset using multiple commercial acoustic and electric guitar plugins. This dataset is built on tablatures from DadaGP, which offers a vast collection and the degree of specificity we wish to transcribe. The proposed synthesis pipeline produces audio which faithfully adheres to the original fingerings, styles, and techniques specified in the tablature with diverse timbre. Exper
    
[^11]: 使用大型语言模型推理图提升推荐系统

    Enhancing Recommender Systems with Large Language Model Reasoning Graphs. (arXiv:2308.10835v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2308.10835](http://arxiv.org/abs/2308.10835)

    本文提出了一种使用大型语言模型（LLMs）构建个性化推理图的方法，通过因果和逻辑推理链接用户的个人资料和行为序列，在提升推荐系统性能的同时实现了更多的逻辑性和可解释性。

    

    推荐系统旨在为用户提供相关建议，但通常缺乏可解释性，并且无法捕捉用户行为和个人资料之间的更高级语义关系。本文提出了一种新颖的方法，利用大型语言模型（LLMs）构建个性化推理图。这些图通过因果和逻辑推理将用户的个人资料和行为序列链接起来，以可解释的方式表示用户的兴趣。我们的方法，LLM推理图（LLMRG），包括四个组件：链接图推理、发散扩展、自我验证与评分，以及知识库自我改进。最终得到的推理图被编码为图神经网络，作为额外的输入来改进传统的推荐系统，而无需额外的用户或项目信息。我们的方法展示了LLMs如何通过个性化推理图实现更具逻辑性和可解释性的推荐系统。

    Recommendation systems aim to provide users with relevant suggestions, but often lack interpretability and fail to capture higher-level semantic relationships between user behaviors and profiles. In this paper, we propose a novel approach that leverages large language models (LLMs) to construct personalized reasoning graphs. These graphs link a user's profile and behavioral sequences through causal and logical inferences, representing the user's interests in an interpretable way. Our approach, LLM reasoning graphs (LLMRG), has four components: chained graph reasoning, divergent extension, self-verification and scoring, and knowledge base self-improvement. The resulting reasoning graph is encoded using graph neural networks, which serves as additional input to improve conventional recommender systems, without requiring extra user or item information. Our approach demonstrates how LLMs can enable more logical and interpretable recommender systems through personalized reasoning graphs. LL
    
[^12]: 寻求稳定性：具有初始文件的战略出版商的学习动态的研究

    The Search for Stability: Learning Dynamics of Strategic Publishers with Initial Documents. (arXiv:2305.16695v1 [cs.GT])

    [http://arxiv.org/abs/2305.16695](http://arxiv.org/abs/2305.16695)

    本研究在信息检索博弈论模型中提出了相对排名原则（RRP）作为替代排名原则，以达成更稳定的搜索生态系统，并提供了理论和实证证据证明其学习动力学收敛性，同时展示了可能的出版商-用户权衡。

    

    我们研究了一种信息检索的博弈论模型，其中战略出版商旨在在保持原始文档完整性的同时最大化自己排名第一的机会。我们表明，常用的PRP排名方案导致环境不稳定，游戏经常无法达到纯纳什均衡。我们将相对排名原则（RRP）作为替代排名原则，并介绍两个排名函数，它们是RRP的实例。我们提供了理论和实证证据，表明这些方法导致稳定的搜索生态系统，通过提供关于学习动力学收敛的积极结果。我们还定义出版商和用户的福利，并展示了可能的出版商-用户权衡，突显了确定搜索引擎设计师应选择哪种排名函数的复杂性。

    We study a game-theoretic model of information retrieval, in which strategic publishers aim to maximize their chances of being ranked first by the search engine, while maintaining the integrity of their original documents. We show that the commonly used PRP ranking scheme results in an unstable environment where games often fail to reach pure Nash equilibrium. We propose the Relative Ranking Principle (RRP) as an alternative ranking principle, and introduce two ranking functions that are instances of the RRP. We provide both theoretical and empirical evidence that these methods lead to a stable search ecosystem, by providing positive results on the learning dynamics convergence. We also define the publishers' and users' welfare, and demonstrate a possible publisher-user trade-off, which highlights the complexity of determining which ranking function should be selected by the search engine designer.
    

