# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Fusion-in-T5: Unifying Document Ranking Signals for Improved Information Retrieval.](http://arxiv.org/abs/2305.14685) | 本文提出了一种新的重新排名器FiT5，它将文档文本信息、检索特征和全局文档信息统一到一个单一的模型中，通过全局注意力使得FiT5能够共同利用排名特征，从而改善检测微妙差别的能力，在实验表现上显著提高了排名表现。 |
| [^2] | [Revisit and Outstrip Entity Alignment: A Perspective of Generative Models.](http://arxiv.org/abs/2305.14651) | 本文重新从生成模型的角度研究了基于嵌入的实体对齐（EEA）问题，引入基于生成对抗网络的EEA方法及提出的生成的EEA（GEEA）框架，通过互相变分自动编码器（M-VAE）实现实体从一个KG转换到另一个KG，并且从随机噪声向量生成新的实体，具有较好的效果。 |
| [^3] | [Enabling Large Language Models to Generate Text with Citations.](http://arxiv.org/abs/2305.14627) | 本文提出ALCE，是首个自动LLMs引文评估基准，实现大型语言模型生成带引文的文本，提高其事实正确性和可验证性；提示LLMs特定的关键词或利用外部知识源可以显著提高其引文准确性。 |
| [^4] | [Contextualized Topic Coherence Metrics.](http://arxiv.org/abs/2305.14587) | 本研究提出了一种上下文化主题相干性评估指标（CTC），该指标不仅在短文档上运作良好，而且在相对于其他五个指标评估上具有更高的性能表现。 |
| [^5] | [Extracting Shopping Interest-Related Product Types from the Web.](http://arxiv.org/abs/2305.14549) | 本文提出了一种从包含手工PT推荐的Web页面中提取PTs的方法，以建立购物兴趣（SI）和产品类型（PT）之间的连接，并引入了TrENC来改进内部节点之间的依赖建模。 |
| [^6] | [NAIL: Lexical Retrieval Indices with Efficient Non-Autoregressive Decoders.](http://arxiv.org/abs/2305.14499) | NAIL是一种带有高效非自回归解码器的词汇检索指数模型，可与现有的预训练模型兼容，并且使用商品CPU提供服务。它可以捕捉Transformer交叉关注模型收益高达86％的方法，与BM25检索器结合使用匹配当前最先进的双编码器检索器的质量。 |
| [^7] | [Knowledge Graphs Querying.](http://arxiv.org/abs/2305.14485) | 该论文调查了知识图谱查询的研究进展和最新技术和方法。 |
| [^8] | [Graph Meets LLM: A Novel Approach to Collaborative Filtering for Robust Conversational Understanding.](http://arxiv.org/abs/2305.14449) | 一种协同过滤新方法用于稳健对话理解，在历史用户-实体交互的基础上，利用多跳客户亲和力丰富每个用户的索引，并使用有限内存BFGS算法调整每个索引的权重，实验结果显示其明显优于最先进的个性化查询重写方法。 |
| [^9] | [Anchor Prediction: Automatic Refinement of Internet Links.](http://arxiv.org/abs/2305.14337) | 本研究介绍了锚点预测的任务，通过对链接目标网页的特定部分进行识别，帮助读者更有效地在链接网页中找到相关信息。 |
| [^10] | [Term-Sets Can Be Strong Document Identifiers For Auto-Regressive Search Engines.](http://arxiv.org/abs/2305.13859) | 本论文提出了一种新的自回归搜索引擎框架AutoTSG，其特点是基于无序术语集的文档标识符和基于集合的生成管道，大大放松了对标识符精确生成的要求。 |
| [^11] | [What Are You Token About? Dense Retrieval as Distributions Over the Vocabulary.](http://arxiv.org/abs/2212.10380) | 本文探讨了双编码器用于稠密检索的机制，通过将向量表示投影到模型的词汇表空间来解释它们，进一步解释了一些失败案例，提出了一种简单的方法在推理时丰富查询和段落表示与词汇信息，显著提高了性能。 |
| [^12] | [COVID-19 Activity Risk Calculator as a Gamified Public Health Intervention Tool.](http://arxiv.org/abs/2212.05035) | CovARC是一种游戏化的公共卫生干预工具，通过风险评估和行为影响来降低个人在日常生活中感染新冠病毒的风险。 |
| [^13] | [SciRepEval: A Multi-Format Benchmark for Scientific Document Representations.](http://arxiv.org/abs/2211.13308) | SciRepEval是第一个综合评估科学文献表示的全面基准，其中包括四种格式的 25 个任务。通过使用格式特定的控制代码和适配器，可以改进科学文献表示模型的泛化能力。 |
| [^14] | [Deep Exploration for Recommendation Systems.](http://arxiv.org/abs/2109.12509) | 本文提出了一种深度探索方法以解决推荐系统中奖励稀少时的问题，并在高保真度的工业级模拟器下进行了实验，证明了该算法相比现有算法有很大的提升。 |

# 详细

[^1]: Fusion-in-T5: 将文档排名信号统一起来以改进信息检索

    Fusion-in-T5: Unifying Document Ranking Signals for Improved Information Retrieval. (arXiv:2305.14685v1 [cs.IR])

    [http://arxiv.org/abs/2305.14685](http://arxiv.org/abs/2305.14685)

    本文提出了一种新的重新排名器FiT5，它将文档文本信息、检索特征和全局文档信息统一到一个单一的模型中，通过全局注意力使得FiT5能够共同利用排名特征，从而改善检测微妙差别的能力，在实验表现上显著提高了排名表现。

    

    常见的信息检索流程通常采用级联系统，可能涉及多个排名器和/或融合模型逐步整合不同的信息。在本文中，我们提出了一种称为Fusion-in-T5（FiT5）的新型重新排名器，它使用基于模板的输入和全局注意力将文档文本信息、检索特征和全局文档信息统一到一个单一的模型中。在MS MARCO和TREC DL的段落排名基准测试中，实验表明FiT5在先前的流水线性能上显著提高了排名表现。分析发现，通过全局注意力，FiT5能够逐渐关注相关文档，从而共同利用排名特征，改善检测它们之间微妙差别的能力。我们的代码将开源。

    Common IR pipelines are typically cascade systems that may involve multiple rankers and/or fusion models to integrate different information step-by-step. In this paper, we propose a novel re-ranker named Fusion-in-T5 (FiT5), which integrates document text information, retrieval features, and global document information into a single unified model using templated-based input and global attention. Experiments on passage ranking benchmarks MS MARCO and TREC DL show that FiT5 significantly improves ranking performance over prior pipelines. Analyses find that through global attention, FiT5 is able to jointly utilize the ranking features via gradually attending to related documents, and thus improve the detection of subtle nuances between them. Our code will be open-sourced.
    
[^2]: 从生成模型的角度重新审视实体对齐及超越：一个视角

    Revisit and Outstrip Entity Alignment: A Perspective of Generative Models. (arXiv:2305.14651v1 [cs.CL])

    [http://arxiv.org/abs/2305.14651](http://arxiv.org/abs/2305.14651)

    本文重新从生成模型的角度研究了基于嵌入的实体对齐（EEA）问题，引入基于生成对抗网络的EEA方法及提出的生成的EEA（GEEA）框架，通过互相变分自动编码器（M-VAE）实现实体从一个KG转换到另一个KG，并且从随机噪声向量生成新的实体，具有较好的效果。

    

    最近，基于嵌入的方法在利用多模态知识图谱（KG）嵌入的实体对齐方面取得了巨大成功。在本文中，我们从生成模型的角度研究了基于嵌入的实体对齐（EEA）。我们表明EEA是一个特殊的问题，其主要目标类似于典型生成模型中的目标，基于这个目标，我们从理论上证明了最近发展的基于生成对抗网络（GAN）的EEA方法的有效性。然后，我们揭示了他们不完整的目标限制了实体对齐和实体合成（即生成新实体）的能力。我们通过引入生成的EEA（abbr.，GEEA）框架和提出的互相变分自动编码器（M-VAE）作为生成模型来缓解这个问题。M-VAE可以将一个实体从一个KG转换到另一个KG，并从随机噪声向量生成新实体。我们通过理论分析和实证实验展示了GEEA的优势。

    Recent embedding-based methods have achieved great successes on exploiting entity alignment from knowledge graph (KG) embeddings of multiple modals. In this paper, we study embedding-based entity alignment (EEA) from a perspective of generative models. We show that EEA is a special problem where the main objective is analogous to that in a typical generative model, based on which we theoretically prove the effectiveness of the recently developed generative adversarial network (GAN)-based EEA methods. We then reveal that their incomplete objective limits the capacity on both entity alignment and entity synthesis (i.e., generating new entities). We mitigate this problem by introducing a generative EEA (abbr., GEEA) framework with the proposed mutual variational autoencoder (M-VAE) as the generative model. M-VAE can convert an entity from one KG to another and generate new entities from random noise vectors. We demonstrate the power of GEEA with theoretical analysis and empirical experime
    
[^3]: 实现大型语言模型生成带引文的文本

    Enabling Large Language Models to Generate Text with Citations. (arXiv:2305.14627v1 [cs.CL])

    [http://arxiv.org/abs/2305.14627](http://arxiv.org/abs/2305.14627)

    本文提出ALCE，是首个自动LLMs引文评估基准，实现大型语言模型生成带引文的文本，提高其事实正确性和可验证性；提示LLMs特定的关键词或利用外部知识源可以显著提高其引文准确性。

    

    大型语言模型（LLMs）已成为广泛使用的信息寻找工具，但生成的输出容易出现幻觉。本文旨在实现LLMs生成带引文的文本，提高其事实正确性和可验证性。我们提出了ALCE，这是首个自动LLMs引文评估基准。ALCE收集了各种问题和检索语料库，并要求建立端到端系统以检索支持证据并生成带有引文的答案。我们沿着流畅性、正确性和引文质量三个维度构建自动指标，并展示了它们与人类判断的强相关性。我们使用最先进的LLMs和新的提示策略进行实验，结果表明当前系统仍有相当大的提升空间--例如，提示LLMs特定的关键词或利用外部知识源可以显著提高其引文准确性。我们的工作为未来研究发展能够生成可验证和可信赖输出的LLMs提供了坚实基础。

    Large language models (LLMs) have emerged as a widely-used tool for information seeking, but their generated outputs are prone to hallucination. In this work, we aim to enable LLMs to generate text with citations, improving their factual correctness and verifiability. Existing work mainly relies on commercial search engines and human evaluation, making it challenging to reproduce and compare with different modeling approaches. We propose ALCE, the first benchmark for Automatic LLMs' Citation Evaluation. ALCE collects a diverse set of questions and retrieval corpora and requires building end-to-end systems to retrieve supporting evidence and generate answers with citations. We build automatic metrics along three dimensions -- fluency, correctness, and citation quality -- and demonstrate their strong correlation with human judgements. Our experiments with state-of-the-art LLMs and novel prompting strategies show that current systems have considerable room for improvements -for example,
    
[^4]: 上下文化主题相干性评估指标

    Contextualized Topic Coherence Metrics. (arXiv:2305.14587v1 [cs.CL])

    [http://arxiv.org/abs/2305.14587](http://arxiv.org/abs/2305.14587)

    本研究提出了一种上下文化主题相干性评估指标（CTC），该指标不仅在短文档上运作良好，而且在相对于其他五个指标评估上具有更高的性能表现。

    

    近年来，神经主题建模的大量研究被指责在优化自动化主题评估指标的同时牺牲了实质性的主题识别。但是，人工标注的成本高昂且耗时。本文提出了一种基于LLM的方法，受到人类主题评估的启发，提出了一种度量系列，称为上下文化主题相干性（CTC）。我们评估了一个全自动的版本以及一个半自动化的CTC，该版本针对人工中心的相干性评估，同时保持了自动化方法的效率。我们在六个主题模型上相对于五个其他指标评估CTC，并发现它优于自动主题一致性方法，在短文档上运作良好，并且不容易受到评分高但毫无意义的主题的影响。

    The recent explosion in work on neural topic modeling has been criticized for optimizing automated topic evaluation metrics at the expense of actual meaningful topic identification. But human annotation remains expensive and time-consuming. We propose LLM-based methods inspired by standard human topic evaluations, in a family of metrics called Contextualized Topic Coherence (CTC). We evaluate both a fully automated version as well as a semi-automated CTC that allows human-centered evaluation of coherence while maintaining the efficiency of automated methods. We evaluate CTC relative to five other metrics on six topic models and find that it outperforms automated topic coherence methods, works well on short documents, and is not susceptible to meaningless but high-scoring topics.
    
[^5]: 从网络中提取与购物兴趣相关的产品类型

    Extracting Shopping Interest-Related Product Types from the Web. (arXiv:2305.14549v1 [cs.IR])

    [http://arxiv.org/abs/2305.14549](http://arxiv.org/abs/2305.14549)

    本文提出了一种从包含手工PT推荐的Web页面中提取PTs的方法，以建立购物兴趣（SI）和产品类型（PT）之间的连接，并引入了TrENC来改进内部节点之间的依赖建模。

    

    当客户在寻找其高级购物兴趣（如徒步旅行等）的产品时，推荐多样化的产品类型（PTs）对于提供良好的购物体验至关重要。然而，电子商务产品目录中通常缺乏SI-PT的连接，并且由于潜在的SI数量巨大，手动构建这种连接也是非常昂贵的，这使得我们无法建立易于访问的知识系统。为了建立这样的连接，我们提出从包含手工PT推荐的Web页面中提取PTs的方法。将提取任务设置为二进制HTML节点分类，因为我们的目标Web页面中的HTML节点可以呈现一个且仅一个PT短语的普遍观察。为此，我们引入了TrENC，即Tree-Transformer编码器用于节点分类。它改进了内部节点之间的依赖建模，并保留了长期的兄弟姐妹和祖先-后代关系的注意机制。

    Recommending a diversity of product types (PTs) is important for a good shopping experience when customers are looking for products around their high-level shopping interests (SIs) such as hiking. However, the SI-PT connection is typically absent in e-commerce product catalogs and expensive to construct manually due to the volume of potential SIs, which prevents us from establishing a recommender with easily accessible knowledge systems. To establish such connections, we propose to extract PTs from the Web pages containing hand-crafted PT recommendations for SIs. The extraction task is formulated as binary HTML node classification given the general observation that an HTML node in our target Web pages can present one and only one PT phrase. Accordingly, we introduce TrENC, which stands for Tree-Transformer Encoders for Node Classification. It improves the inter-node dependency modeling with modified attention mechanisms that preserve the long-term sibling and ancestor-descendant relati
    
[^6]: NAIL: 带高效非自回归解码器的词汇检索指数

    NAIL: Lexical Retrieval Indices with Efficient Non-Autoregressive Decoders. (arXiv:2305.14499v1 [cs.CL])

    [http://arxiv.org/abs/2305.14499](http://arxiv.org/abs/2305.14499)

    NAIL是一种带有高效非自回归解码器的词汇检索指数模型，可与现有的预训练模型兼容，并且使用商品CPU提供服务。它可以捕捉Transformer交叉关注模型收益高达86％的方法，与BM25检索器结合使用匹配当前最先进的双编码器检索器的质量。

    

    神经文档重新排名器在精度方面非常有效。然而，最好的模型需要专用硬件进行服务，这是昂贵并且通常是不可行的。为了避免这种服务时间要求，我们提出一种捕捉Transformer交叉关注模型收益高达86％的方法，该方法使用只需要每个文档转换器FLOP的10-6％的词汇得分功能，并且可以使用商品CPU提供服务。当与BM25检索器结合使用时，此方法可以匹配现有的最先进的双编码器检索器的质量，该检索器仍需要加速器进行查询编码。我们将NAIL（带有语言模型的非自回归索引）引入为与最近的编码器-解码器和仅解码器大型语言模型（例如T5、GPT-3和PaLM）兼容的模型体系结构。该模型体系结构可以利用现有的预训练检查点，并可以微调以有效地构建不需要n

    Neural document rerankers are extremely effective in terms of accuracy. However, the best models require dedicated hardware for serving, which is costly and often not feasible. To avoid this serving-time requirement, we present a method of capturing up to 86% of the gains of a Transformer cross-attention model with a lexicalized scoring function that only requires 10-6% of the Transformer's FLOPs per document and can be served using commodity CPUs. When combined with a BM25 retriever, this approach matches the quality of a state-of-the art dual encoder retriever, that still requires an accelerator for query encoding. We introduce NAIL (Non-Autoregressive Indexing with Language models) as a model architecture that is compatible with recent encoder-decoder and decoder-only large language models, such as T5, GPT-3 and PaLM. This model architecture can leverage existing pre-trained checkpoints and can be fine-tuned for efficiently constructing document representations that do not require n
    
[^7]: 知识图谱查询

    Knowledge Graphs Querying. (arXiv:2305.14485v1 [cs.DB])

    [http://arxiv.org/abs/2305.14485](http://arxiv.org/abs/2305.14485)

    该论文调查了知识图谱查询的研究进展和最新技术和方法。

    

    知识图谱（KG）如DBpedia、Freebase、YAGO、Wikidata和NELL等被构建用来存储大规模、现实世界中的事实（主题、谓语、对象）三元组，可以被建模为一个图形，其中一个节点（主题或对象）代表具有属性的实体，并且有向边（谓词）是两个实体之间的关系。在Web搜索、问答、语义搜索、个人助手、事实检查和推荐中，查询 KG 是至关重要的。尽管在 KG 构建和维护方面已经取得了重大进展，但由于深度学习的影响，我们最近看到了KG 查询和问答研究方面的激增。我们调查的目的是双重的。首先，KG查询的研究由多个社群进行了研究，如数据库、数据挖掘、语义网、机器学习、信息检索和自然语言处理（NLP），关注点和术语不同。其次，我们还调查了最近的KG查询研究中使用的最新技术和方法。

    Knowledge graphs (KGs) such as DBpedia, Freebase, YAGO, Wikidata, and NELL were constructed to store large-scale, real-world facts as (subject, predicate, object) triples -- that can also be modeled as a graph, where a node (a subject or an object) represents an entity with attributes, and a directed edge (a predicate) is a relationship between two entities. Querying KGs is critical in web search, question answering (QA), semantic search, personal assistants, fact checking, and recommendation. While significant progress has been made on KG construction and curation, thanks to deep learning recently we have seen a surge of research on KG querying and QA. The objectives of our survey are two-fold. First, research on KG querying has been conducted by several communities, such as databases, data mining, semantic web, machine learning, information retrieval, and natural language processing (NLP), with different focus and terminologies; and also in diverse topics ranging from graph databases
    
[^8]: 图谱遇见LLM：一种用于稳健对话理解的协同过滤新方法

    Graph Meets LLM: A Novel Approach to Collaborative Filtering for Robust Conversational Understanding. (arXiv:2305.14449v1 [cs.AI])

    [http://arxiv.org/abs/2305.14449](http://arxiv.org/abs/2305.14449)

    一种协同过滤新方法用于稳健对话理解，在历史用户-实体交互的基础上，利用多跳客户亲和力丰富每个用户的索引，并使用有限内存BFGS算法调整每个索引的权重，实验结果显示其明显优于最先进的个性化查询重写方法。

    

    会话式人工智能系统（例如Alexa，Siri，Google Assistant等）需要理解存在缺陷的查询以确保稳健的会话理解并减少用户摩擦。这些有缺陷的查询通常是由用户的歧义和错误，自动语音识别（ASR）和自然语言理解（NLU）中的错误引起的。个性化查询重写（个性化QR）旨在减少身体和尾部用户查询流量中的缺陷，通常依赖于与对话式人工智能的过去成功的用户交互的索引。本文提出我们的“协同查询重写”方法，专注于重写用户历史中没有出现过的新型用户交互。该方法构建了一个“用户反馈交互图”（FIG），由历史用户-实体交互组成，并利用多跳客户亲和力来丰富每个用户的索引（即协同用户索引），从而帮助覆盖未来未曾见过的存在缺陷的查询。为了防止这些新的丰富索引被噪声反馈交互所支配，我们采用了有限内存BFGS（LLM）算法和回退方案来调整每个索引的权重。实验结果表明，我们的方法明显优于最先进的个性化QR方法，并在未看到的用户交互上取得了近乎完美的性能。

    Conversational AI systems (e.g. Alexa, Siri, Google Assistant, etc.) need to understand queries with defects to ensure robust conversational understanding and reduce user frictions. The defective queries are often induced by user ambiguities and mistakes, or errors in the automatic speech recognition (ASR) and natural language understanding (NLU).  Personalized query rewriting (personalized QR) targets reducing defects in the torso and tail user query traffic, and it typically relies on an index of past successful user interactions with the conversational AI. This paper presents our "Collaborative Query Rewriting" approach that focuses on rewriting novel user interactions unseen in the user history. This approach builds a "user Feedback Interaction Graph" (FIG) consisting of historical user-entity interactions, and leverages multi-hop customer affinity to enrich each user's index (i.e. the Collaborative User Index) that would help cover future unseen defective queries. To counteract th
    
[^9]: Anchor Prediction: 自动完善互联网链接

    Anchor Prediction: Automatic Refinement of Internet Links. (arXiv:2305.14337v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2305.14337](http://arxiv.org/abs/2305.14337)

    本研究介绍了锚点预测的任务，通过对链接目标网页的特定部分进行识别，帮助读者更有效地在链接网页中找到相关信息。

    

    互联网链接是通过提供便捷的访问相关信息，帮助用户深入了解一个主题。然而，大部分链接都是无锚点的 - 它们将目标网页作为一个整体链接，读者可能会花费大量的精力定位目标网页中丰富他们理解链接源上下文的特定部分。为了帮助读者有效地在链接的网页中找到信息，我们引入了锚点预测的任务，目标是确定与源链接上下文最相关的目标网页的特定部分。我们发布了作者锚点数据集，其中包括 34K 个自然出现的带锚点链接，反映了源文章作者的相关判断。为了模拟读者相关判断，我们注释并发布了读者锚点数据集，这是读者发现有用的锚点的评估集。我们的分析表明，有效的锚点预测通常需要联合推理。

    Internet links enable users to deepen their understanding of a topic by providing convenient access to related information. However, the majority of links are unanchored -- they link to a target webpage as a whole, and readers may expend considerable effort localizing the specific parts of the target webpage that enrich their understanding of the link's source context. To help readers effectively find information in linked webpages, we introduce the task of anchor prediction, where the goal is to identify the specific part of the linked target webpage that is most related to the source linking context. We release the AuthorAnchors dataset, a collection of 34K naturally-occurring anchored links, which reflect relevance judgments by the authors of the source article. To model reader relevance judgments, we annotate and release ReaderAnchors, an evaluation set of anchors that readers find useful. Our analysis shows that effective anchor prediction often requires jointly reasoning over len
    
[^10]: 术语集可以成为自回归搜索引擎的强文档标识符

    Term-Sets Can Be Strong Document Identifiers For Auto-Regressive Search Engines. (arXiv:2305.13859v1 [cs.IR])

    [http://arxiv.org/abs/2305.13859](http://arxiv.org/abs/2305.13859)

    本论文提出了一种新的自回归搜索引擎框架AutoTSG，其特点是基于无序术语集的文档标识符和基于集合的生成管道，大大放松了对标识符精确生成的要求。

    

    自回归搜索引擎是下一代信息检索系统的有前途的范例。这些方法利用Seq2Seq模型，其中每个查询可以直接映射到其相关文档的标识符。因此，它们因具有端到端可微分性等优点而受到赞扬。然而，自回归搜索引擎在检索质量上也面临着挑战，因为其需要对文档标识符进行精确生成。也就是说，如果在生成过程的任何一步中对其标识符做出了错误的预测，则目标文档将从检索结果中漏失。在这项工作中，我们提出了一种新的框架，即AutoTSG(自回归搜索引擎与术语集生成)，其特点是1)无序基于术语的文档标识符和2)基于集合的生成管道。利用AutoTSG，术语集标识符的任何排列都将导致相应文档的检索，从而大大放松了对标识符精确生成的要求。

    Auto-regressive search engines emerge as a promising paradigm for next-gen information retrieval systems. These methods work with Seq2Seq models, where each query can be directly mapped to the identifier of its relevant document. As such, they are praised for merits like being end-to-end differentiable. However, auto-regressive search engines also confront challenges in retrieval quality, given the requirement for the exact generation of the document identifier. That's to say, the targeted document will be missed from the retrieval result if a false prediction about its identifier is made in any step of the generation process. In this work, we propose a novel framework, namely AutoTSG (Auto-regressive Search Engine with Term-Set Generation), which is featured by 1) the unordered term-based document identifier and 2) the set-oriented generation pipeline. With AutoTSG, any permutation of the term-set identifier will lead to the retrieval of the corresponding document, thus largely relaxi
    
[^11]: 你所谓的令牌是关于什么的？稠密检索作为词汇表上的分布。

    What Are You Token About? Dense Retrieval as Distributions Over the Vocabulary. (arXiv:2212.10380v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2212.10380](http://arxiv.org/abs/2212.10380)

    本文探讨了双编码器用于稠密检索的机制，通过将向量表示投影到模型的词汇表空间来解释它们，进一步解释了一些失败案例，提出了一种简单的方法在推理时丰富查询和段落表示与词汇信息，显著提高了性能。

    

    双编码器现在是稠密检索的主要架构。然而，我们对它们如何表示文本以及为什么会导致良好性能知之甚少。在本文中，我们通过词汇表上的分布来阐明这个问题。我们建议通过将双编码器产生的向量表示投影到模型的词汇表空间中来解释它们，我们展示了产生的投影包含丰富的语义信息，并将它们与稀疏检索之间进行联系。我们发现，这种观点可以解释稠密检索器的一些失败案例。例如，我们观察到模型无法处理尾部实体与令牌分布倾向于忘记这些实体的某些令牌之间存在相关性。我们利用了这一洞察，并提出了一种在推理时丰富查询和段落表示与词汇信息的简单方法，并展示了这相比于常规的双编码器有显著的性能提升。

    Dual encoders are now the dominant architecture for dense retrieval. Yet, we have little understanding of how they represent text, and why this leads to good performance. In this work, we shed light on this question via distributions over the vocabulary. We propose to interpret the vector representations produced by dual encoders by projecting them into the model's vocabulary space. We show that the resulting projections contain rich semantic information, and draw connection between them and sparse retrieval. We find that this view can offer an explanation for some of the failure cases of dense retrievers. For example, we observe that the inability of models to handle tail entities is correlated with a tendency of the token distributions to forget some of the tokens of those entities. We leverage this insight and propose a simple way to enrich query and passage representations with lexical information at inference time, and show that this significantly improves performance compared to 
    
[^12]: 作为一种游戏化的公共卫生干预工具的COVID-19活动风险计算器。

    COVID-19 Activity Risk Calculator as a Gamified Public Health Intervention Tool. (arXiv:2212.05035v4 [cs.CY] UPDATED)

    [http://arxiv.org/abs/2212.05035](http://arxiv.org/abs/2212.05035)

    CovARC是一种游戏化的公共卫生干预工具，通过风险评估和行为影响来降低个人在日常生活中感染新冠病毒的风险。

    

    由严重急性呼吸综合症冠状病毒2 (SARS-CoV-2) 引起的新型冠状病毒肺炎疫情已经影响到200多个国家，在全球造成了数百万人的住院和死亡。公共卫生干预，如风险评估工具，可以通过影响个人行为降低暴发流行病传播风险和感染率。然而，由于社区传播水平和病毒变异等因素的快速演变，目前公开可用的COVID-19风险评估工具的效果参差不齐。此外，关于某些个人防护策略，如口罩佩戴和接种疫苗的风险降低问题也存在困惑。为了创建一个用于评估日常活动所涉及的各种个人风险的简单易用工具，我们开发了COVID-19活动风险计算器（CovARC）。CovARC是一种游戏化的公共卫生干预工具。

    The Coronavirus disease 2019 (COVID-19) pandemic, caused by the virus severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2), has impacted over 200 countries leading to hospitalizations and deaths of millions of people. Public health interventions, such as risk estimators, can reduce the spread of pandemics and epidemics through influencing behavior, which impacts risk of exposure and infection. Current publicly available COVID-19 risk estimation tools have had variable effectiveness during the pandemic due to their dependency on rapidly evolving factors such as community transmission levels and variants. There has also been confusion surrounding certain personal protective strategies such as risk reduction by mask-wearing and vaccination. In order to create a simple easy-to-use tool for estimating different individual risks associated with carrying out daily-life activity, we developed COVID-19 Activity Risk Calculator (CovARC). CovARC is a gamified public health intervention as
    
[^13]: SciRepEval：一个用于科学文献表示的多格式基准

    SciRepEval: A Multi-Format Benchmark for Scientific Document Representations. (arXiv:2211.13308v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2211.13308](http://arxiv.org/abs/2211.13308)

    SciRepEval是第一个综合评估科学文献表示的全面基准，其中包括四种格式的 25 个任务。通过使用格式特定的控制代码和适配器，可以改进科学文献表示模型的泛化能力。

    

    学习的科学文献表示可以作为下游任务的有价值输入特征，无需进一步微调。然而，用于评估这些表示的现有基准未能捕捉到相关任务的多样性。为此，我们介绍了 SciRepEval，第一个用于训练和评估科学文献表示的全面基准。它包括四种格式的 25 个具有挑战性和现实性的任务，其中 11 个是新任务：分类、回归、排名和搜索。我们使用该基准来研究和改进科学文档表示模型的泛化能力。我们展示了最先进的模型如何在任务格式方面缺乏泛化性能，简单的多任务训练也不能改进它们。然而，一种新的方法，学习每个文档的多个嵌入，每个嵌入专门针对不同的格式，可以提高性能。我们尝试使用任务格式特定的控制代码和适配器。

    Learned representations of scientific documents can serve as valuable input features for downstream tasks, without the need for further fine-tuning. However, existing benchmarks for evaluating these representations fail to capture the diversity of relevant tasks. In response, we introduce SciRepEval, the first comprehensive benchmark for training and evaluating scientific document representations. It includes 25 challenging and realistic tasks, 11 of which are new, across four formats: classification, regression, ranking and search. We then use the benchmark to study and improve the generalization ability of scientific document representation models. We show how state-of-the-art models struggle to generalize across task formats, and that simple multi-task training fails to improve them. However, a new approach that learns multiple embeddings per document, each tailored to a different format, can improve performance. We experiment with task-format-specific control codes and adapters in 
    
[^14]: 推荐系统的深度探索

    Deep Exploration for Recommendation Systems. (arXiv:2109.12509v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2109.12509](http://arxiv.org/abs/2109.12509)

    本文提出了一种深度探索方法以解决推荐系统中奖励稀少时的问题，并在高保真度的工业级模拟器下进行了实验，证明了该算法相比现有算法有很大的提升。

    

    现代推荐系统应从延迟反馈中探索和学习。过去的研究往往侧重于从用户对单个推荐的响应中学习。这些工作利用了监督学习和强化学习的方法，但放弃了学习用户之后的行为。在过去的工作中，虽然致力于从随后的行为中学习，但缺乏有效的方法来引导并获取有意义的延迟反馈。当奖励较少时，通过引导探索有意义的延迟反馈变得特别具有挑战性。为了解决这个问题，我们为推荐系统开发了深度探索方法。具体而言，我们将推荐系统形式化为一个序列决策问题，并证明了深度探索方法在单步探索方面的优势。我们的实验是在高保真度的工业级模拟器下进行的，并且证明了该算法相比现有算法有很大的提升。

    Modern recommendation systems ought to benefit by probing for and learning from delayed feedback. Research has tended to focus on learning from a user's response to a single recommendation. Such work, which leverages methods of supervised and bandit learning, forgoes learning from the user's subsequent behavior. Where past work has aimed to learn from subsequent behavior, there has been a lack of effective methods for probing to elicit informative delayed feedback. Effective exploration through probing for delayed feedback becomes particularly challenging when rewards are sparse. To address this, we develop deep exploration methods for recommendation systems. In particular, we formulate recommendation as a sequential decision problem and demonstrate benefits of deep exploration over single-step exploration. Our experiments are carried out with high-fidelity industrial-grade simulators and establish large improvements over existing algorithms.
    

