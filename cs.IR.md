# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [ILCiteR: Evidence-grounded Interpretable Local Citation Recommendation](https://arxiv.org/abs/2403.08737) | 介绍了一个名为ILCiteR的系统，通过基于证据的局部引文推荐任务，从现有研究文献中提取相似证据范围来推荐引用的论文，提高了推荐的可解释性。 |
| [^2] | [NLQxform-UI: A Natural Language Interface for Querying DBLP Interactively](https://arxiv.org/abs/2403.08475) | NLQxform-UI是一个基于自然语言的交互式查询界面，可以自动将复杂自然语言问题转换为SPARQL查询，并在DBLP知识图上执行，提高了系统的可用性和准确性 |
| [^3] | [Knowledge Conflicts for LLMs: A Survey](https://arxiv.org/abs/2403.08319) | 这项调查深入分析了LLMs在融合上下文和参数化知识时所面临的知识冲突，探讨了三类知识冲突对其可信度和性能的重要影响，并提出改进LLMs稳健性策略的策略。 |
| [^4] | [Towards Unified Modeling for Positive and Negative Preferences in Sign-Aware Recommendation](https://arxiv.org/abs/2403.08246) | 提出了一种面向推荐的轻量级符号图卷积网络LSGRec，采用统一建模方法同时对高阶用户的正负偏好进行建模 |
| [^5] | [Discrete Semantic Tokenization for Deep CTR Prediction](https://arxiv.org/abs/2403.08206) | 提出了一种新型的语义标记范式并引入离散语义标记化方法UIST，用于用户和项目表示，旨在将项目内容信息整合到点击率（CTR）预测模型中，实现快速训练和推断，并在保持内存占用的同时提高效率。 |
| [^6] | [MetaSplit: Meta-Split Network for Limited-Stock Product Recommendation](https://arxiv.org/abs/2403.06747) | 提出了Meta-Split网络（MSN）来解决消费者之间电子商务平台中限量库存产品推荐中的独特挑战，通过分割用户历史序列来有效利用用户历史信息。 |
| [^7] | [GPT-4V(ision) is a Generalist Web Agent, if Grounded.](http://arxiv.org/abs/2401.01614) | GPT-4V(ision)是一个通用的网络代理，具有综合视觉理解和网页操作的能力。实验证明，如果将文本计划转化为实际行动，GPT-4V可以在50%的任务上取得成功。这一结果显著优于传统方法。 |
| [^8] | [Improving Detection of ChatGPT-Generated Fake Science Using Real Publication Text: Introducing xFakeBibs a Supervised-Learning Network Algorithm.](http://arxiv.org/abs/2308.11767) | 本文介绍了一种能够提高对ChatGPT生成的假科学进行检测的算法。通过使用一种新设计的监督机器学习算法，该算法能够准确地将机器生成的出版物与科学家生成的出版物区分开来。结果表明，ChatGPT在技术术语方面与真实科学存在显著差异。算法在分类过程中取得了较高的准确率。 |

# 详细

[^1]: ILCiteR：基于证据的可解释局部引文推荐

    ILCiteR: Evidence-grounded Interpretable Local Citation Recommendation

    [https://arxiv.org/abs/2403.08737](https://arxiv.org/abs/2403.08737)

    介绍了一个名为ILCiteR的系统，通过基于证据的局部引文推荐任务，从现有研究文献中提取相似证据范围来推荐引用的论文，提高了推荐的可解释性。

    

    现有的用于局部引文推荐的机器学习方法直接将查询（通常是声明或实体提及）映射或翻译为值得引用的研究论文。在这种表述中，很难确定为什么应该为特定查询引用特定研究论文，从而导致推荐的可解释性受限。为了缓解这一问题，我们引入了基于证据的局部引文推荐任务，其中目标潜在空间包括用于推荐特定论文的证据范围。通过使用远程监督证据检索和多步重新排序框架，我们提出的系统ILCiteR基于从现有研究文献中提取的相似证据范围向查询推荐应引用的论文。与过去简单输出推荐的形式不同，ILCiteR检索出经过排名的证据范围和推荐的论文对列表。

    arXiv:2403.08737v1 Announce Type: cross  Abstract: Existing Machine Learning approaches for local citation recommendation directly map or translate a query, which is typically a claim or an entity mention, to citation-worthy research papers. Within such a formulation, it is challenging to pinpoint why one should cite a specific research paper for a particular query, leading to limited recommendation interpretability. To alleviate this, we introduce the evidence-grounded local citation recommendation task, where the target latent space comprises evidence spans for recommending specific papers. Using a distantly-supervised evidence retrieval and multi-step re-ranking framework, our proposed system, ILCiteR, recommends papers to cite for a query grounded on similar evidence spans extracted from the existing research literature. Unlike past formulations that simply output recommendations, ILCiteR retrieves ranked lists of evidence span and recommended paper pairs. Secondly, previously prop
    
[^2]: NLQxform-UI：用于交互式查询DBLP的自然语言接口

    NLQxform-UI: A Natural Language Interface for Querying DBLP Interactively

    [https://arxiv.org/abs/2403.08475](https://arxiv.org/abs/2403.08475)

    NLQxform-UI是一个基于自然语言的交互式查询界面，可以自动将复杂自然语言问题转换为SPARQL查询，并在DBLP知识图上执行，提高了系统的可用性和准确性

    

    近年来，DBLP计算机科学文献目录已被广泛用于搜索学术信息，如出版物、学者和会议。然而，其当前的搜索服务缺乏处理复杂查询的能力，这限制了DBLP的可用性。本文提出了NLQxform-UI，这是一个基于web的自然语言接口，允许用户直接用复杂自然语言问题查询DBLP。NLQxform-UI会自动将给定问题转换为SPARQL查询，并在DBLP知识图上执行查询以检索答案。查询过程以交互方式呈现给用户，提高了系统的透明度并有助于检查返回的答案。此外，查询过程中的中间结果可以预览和手动修改以提高系统的准确性。NLQxform-UI已完全开源：https://github.com/ruijie-wan

    arXiv:2403.08475v1 Announce Type: new  Abstract: In recent years, the DBLP computer science bibliography has been prominently used for searching scholarly information, such as publications, scholars, and venues. However, its current search service lacks the capability to handle complex queries, which limits the usability of DBLP. In this paper, we present NLQxform-UI, a web-based natural language interface that enables users to query DBLP directly with complex natural language questions. NLQxform-UI automatically translates given questions into SPARQL queries and executes the queries over the DBLP knowledge graph to retrieve answers. The querying process is presented to users in an interactive manner, which improves the transparency of the system and helps examine the returned answers. Also, intermediate results in the querying process can be previewed and manually altered to improve the accuracy of the system. NLQxform-UI has been completely open-sourced: https://github.com/ruijie-wan
    
[^3]: LLMs的知识冲突：一项调查

    Knowledge Conflicts for LLMs: A Survey

    [https://arxiv.org/abs/2403.08319](https://arxiv.org/abs/2403.08319)

    这项调查深入分析了LLMs在融合上下文和参数化知识时所面临的知识冲突，探讨了三类知识冲突对其可信度和性能的重要影响，并提出改进LLMs稳健性策略的策略。

    

    这项调查对大型语言模型（LLMs）的知识冲突进行了深入分析，突出了当它们融合上下文和参数化知识时所遇到的复杂挑战。我们关注三类知识冲突：上下文-记忆冲突、跨上下文冲突和内部记忆冲突。这些冲突可能会显著影响LLMs的可信度和性能，特别是在现实世界应用中，噪音和错误信息很常见。通过对这些冲突进行分类，探讨其原因，研究LLMs在这些冲突下的行为，并回顾可用的解决方案，本调查旨在为改进LLMs的稳健性策略提供启示，从而成为推动这一不断发展领域研究的宝贵资源。

    arXiv:2403.08319v1 Announce Type: cross  Abstract: This survey provides an in-depth analysis of knowledge conflicts for large language models (LLMs), highlighting the complex challenges they encounter when blending contextual and parametric knowledge. Our focus is on three categories of knowledge conflicts: context-memory, inter-context, and intra-memory conflict. These conflicts can significantly impact the trustworthiness and performance of LLMs, especially in real-world applications where noise and misinformation are common. By categorizing these conflicts, exploring the causes, examining the behaviors of LLMs under such conflicts, and reviewing available solutions, this survey aims to shed light on strategies for improving the robustness of LLMs, thereby serving as a valuable resource for advancing research in this evolving area.
    
[^4]: 面向正负偏好的统一建模的符号感知推荐

    Towards Unified Modeling for Positive and Negative Preferences in Sign-Aware Recommendation

    [https://arxiv.org/abs/2403.08246](https://arxiv.org/abs/2403.08246)

    提出了一种面向推荐的轻量级符号图卷积网络LSGRec，采用统一建模方法同时对高阶用户的正负偏好进行建模

    

    最近，符号感知图推荐引起了广泛关注，因为它将从用户与项目之间的正负交互（即，图中的链接）中学习用户的负偏好，除了正偏好。为了适应负链接和正链接的不同语义，现有作品利用两个独立的编码器分别建模用户的正负偏好。然而，这些方法无法从由多个带有不同符号的链接形成的高阶异构交互中学习负偏好，导致负用户偏好不准确和不完整。为了应对这些棘手的问题，我们提出了一种新颖的面向推荐的轻量级符号图卷积网络LSGRec，采用统一建模方法同时对高阶用户的正负偏好进行建模。

    arXiv:2403.08246v1 Announce Type: cross  Abstract: Recently, sign-aware graph recommendation has drawn much attention as it will learn users' negative preferences besides positive ones from both positive and negative interactions (i.e., links in a graph) with items. To accommodate the different semantics of negative and positive links, existing works utilize two independent encoders to model users' positive and negative preferences, respectively. However, these approaches cannot learn the negative preferences from high-order heterogeneous interactions between users and items formed by multiple links with different signs, resulting in inaccurate and incomplete negative user preferences. To cope with these intractable issues, we propose a novel \textbf{L}ight \textbf{S}igned \textbf{G}raph Convolution Network specifically for \textbf{Rec}ommendation (\textbf{LSGRec}), which adopts a unified modeling approach to simultaneously model high-order users' positive and negative preferences on a
    
[^5]: 用于深度CTR预测的离散语义标记化

    Discrete Semantic Tokenization for Deep CTR Prediction

    [https://arxiv.org/abs/2403.08206](https://arxiv.org/abs/2403.08206)

    提出了一种新型的语义标记范式并引入离散语义标记化方法UIST，用于用户和项目表示，旨在将项目内容信息整合到点击率（CTR）预测模型中，实现快速训练和推断，并在保持内存占用的同时提高效率。

    

    将项目内容信息整合到点击率（CTR）预测模型中仍然是一个挑战，尤其是在工业场景下的时间和空间约束下。传统的内容编码范式将用户和项目编码器直接整合到CTR模型中，优先考虑空间而非时间。相反，基于嵌入的范式将项目和用户语义转换为潜在嵌入，然后对其进行缓存，优先考虑空间而非时间。本文介绍了一种新型的语义标记范式，并提出了一种用于用户和项目表示的离散语义标记化方法，即UIST。UIST实现了快速的训练和推断，同时保持了保守的内存占用。具体而言，UIST将密集嵌入向量量化为较短的离散标记，并采用分层混合推断模块来衡量每个用户-项目标记对的贡献。我们在新闻数据集上的实验结果表明，UIST在提高效率的同时降低了内存消耗。

    arXiv:2403.08206v1 Announce Type: new  Abstract: Incorporating item content information into click-through rate (CTR) prediction models remains a challenge, especially with the time and space constraints of industrial scenarios. The content-encoding paradigm, which integrates user and item encoders directly into CTR models, prioritizes space over time. In contrast, the embedding-based paradigm transforms item and user semantics into latent embeddings and then caches them, prioritizes space over time. In this paper, we introduce a new semantic-token paradigm and propose a discrete semantic tokenization approach, namely UIST, for user and item representation. UIST facilitates swift training and inference while maintaining a conservative memory footprint. Specifically, UIST quantizes dense embedding vectors into discrete tokens with shorter lengths and employs a hierarchical mixture inference module to weigh the contribution of each user--item token pair. Our experimental results on news 
    
[^6]: MetaSplit: 用于限量产品推荐的Meta-Split网络

    MetaSplit: Meta-Split Network for Limited-Stock Product Recommendation

    [https://arxiv.org/abs/2403.06747](https://arxiv.org/abs/2403.06747)

    提出了Meta-Split网络（MSN）来解决消费者之间电子商务平台中限量库存产品推荐中的独特挑战，通过分割用户历史序列来有效利用用户历史信息。

    

    相对于面向消费者的电子商务系统，消费者之间的电子商务平台通常会遇到限量库存问题，即产品在C2C系统中只能销售一次。这为点击率（CTR）预测带来了几个独特的挑战。鉴于每个产品（即商品）的有限用户交互，CTR模型中对应的商品嵌入可能不容易收敛。这使得传统基于序列建模的方法无法有效利用用户历史信息，因为历史用户行为包含了不同库存量的商品混合。特别是，序列模型中的注意力机制倾向于将更多累积用户交互的产品分配更高的分数，导致限量产品被忽视且对最终输出的贡献较少。为此，我们提出了Meta-Split网络（MSN）来分割用户历史序列...

    arXiv:2403.06747v1 Announce Type: new  Abstract: Compared to business-to-consumer (B2C) e-commerce systems, consumer-to-consumer (C2C) e-commerce platforms usually encounter the limited-stock problem, that is, a product can only be sold one time in a C2C system. This poses several unique challenges for click-through rate (CTR) prediction. Due to limited user interactions for each product (i.e. item), the corresponding item embedding in the CTR model may not easily converge. This makes the conventional sequence modeling based approaches cannot effectively utilize user history information since historical user behaviors contain a mixture of items with different volume of stocks. Particularly, the attention mechanism in a sequence model tends to assign higher score to products with more accumulated user interactions, making limited-stock products being ignored and contribute less to the final output. To this end, we propose the Meta-Split Network (MSN) to split user history sequence regar
    
[^7]: GPT-4V(ision)是一个通用的网络代理，如果有基础的话。

    GPT-4V(ision) is a Generalist Web Agent, if Grounded. (arXiv:2401.01614v1 [cs.IR])

    [http://arxiv.org/abs/2401.01614](http://arxiv.org/abs/2401.01614)

    GPT-4V(ision)是一个通用的网络代理，具有综合视觉理解和网页操作的能力。实验证明，如果将文本计划转化为实际行动，GPT-4V可以在50%的任务上取得成功。这一结果显著优于传统方法。

    

    最近对大型多模型（LMM）的研究，特别是GPT-4V(ision)和Gemini，快速推动了多模型的能力边界超越传统任务，如图像字幕和视觉问答。在这项工作中，我们探索了像GPT-4V这样的LMM作为通用网络代理的潜力，可以根据自然语言指令在任何给定的网站上完成任务。我们提出了SEEACT，一种利用LMM的力量进行综合视觉理解和网页操作的通用网络代理。我们在最新的MIND2WEB基准上进行评估。除了对缓存网站的标准离线评估外，我们还通过开发一个允许在实时网站上运行网络代理的工具，实现了一种新的在线评估设置。我们展示了GPT-4V在网页代理方面表现出巨大的潜力-如果我们将其文本计划手动地实施为网站上的行动，它可以成功地完成50%的任务。此结果明显超过了传统方法。

    The recent development on large multimodal models (LMMs), especially GPT-4V(ision) and Gemini, has been quickly expanding the capability boundaries of multimodal models beyond traditional tasks like image captioning and visual question answering. In this work, we explore the potential of LMMs like GPT-4V as a generalist web agent that can follow natural language instructions to complete tasks on any given website. We propose SEEACT, a generalist web agent that harnesses the power of LMMs for integrated visual understanding and acting on the web. We evaluate on the recent MIND2WEB benchmark. In addition to standard offline evaluation on cached websites, we enable a new online evaluation setting by developing a tool that allows running web agents on live websites. We show that GPT-4V presents a great potential for web agents - it can successfully complete 50% of the tasks on live websites if we manually ground its textual plans into actions on the websites. This substantially outperforms
    
[^8]: 提高ChatGPT生成的假科学检测的方法：引入xFakeBibs监督学习网络算法

    Improving Detection of ChatGPT-Generated Fake Science Using Real Publication Text: Introducing xFakeBibs a Supervised-Learning Network Algorithm. (arXiv:2308.11767v1 [cs.CL])

    [http://arxiv.org/abs/2308.11767](http://arxiv.org/abs/2308.11767)

    本文介绍了一种能够提高对ChatGPT生成的假科学进行检测的算法。通过使用一种新设计的监督机器学习算法，该算法能够准确地将机器生成的出版物与科学家生成的出版物区分开来。结果表明，ChatGPT在技术术语方面与真实科学存在显著差异。算法在分类过程中取得了较高的准确率。

    

    ChatGPT正在成为现实。本文展示了如何区分ChatGPT生成的出版物与科学家生成的出版物。通过使用一种新设计的监督机器学习算法，我们演示了如何检测机器生成的出版物和科学家生成的出版物。该算法使用100个真实出版物摘要进行训练，然后采用10倍交叉验证方法建立了一个接受范围的下限和上限。与ChatGPT内容进行比较，明显可见ChatGPT仅贡献了23\%的二元组内容，这比其他10个交叉验证中的任何一个都少50\%。这个分析凸显了ChatGPT在技术术语上与真实科学的明显差异。在对每篇文章进行分类时，xFakeBibs算法准确地将98篇出版物识别为假的，有2篇文献错误地分类为真实出版物。尽管这项工作引入了一种算法应用

    ChatGPT is becoming a new reality. In this paper, we show how to distinguish ChatGPT-generated publications from counterparts produced by scientists. Using a newly designed supervised Machine Learning algorithm, we demonstrate how to detect machine-generated publications from those produced by scientists. The algorithm was trained using 100 real publication abstracts, followed by a 10-fold calibration approach to establish a lower-upper bound range of acceptance. In the comparison with ChatGPT content, it was evident that ChatGPT contributed merely 23\% of the bigram content, which is less than 50\% of any of the other 10 calibrating folds. This analysis highlights a significant disparity in technical terms where ChatGPT fell short of matching real science. When categorizing the individual articles, the xFakeBibs algorithm accurately identified 98 out of 100 publications as fake, with 2 articles incorrectly classified as real publications. Though this work introduced an algorithmic app
    

