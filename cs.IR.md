# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Person Re-ID through Unsupervised Hypergraph Rank Selection and Fusion.](http://arxiv.org/abs/2304.14321) | 本研究提出了一种完全无监督的方法，通过流形排名聚合来选择和融合不同鲁棒的人员再识别排名器，以填补使用标记数据困难的空白。 |
| [^2] | [Large Language Models are Strong Zero-Shot Retriever.](http://arxiv.org/abs/2304.14233) | 本文提出了一种在零-shot场景下利用大型语言模型（LLM）进行大规模检索的方法。该方法通过使用查询和查询的候选答案的组合作为提示，使LLM生成更精确的答案。由于自监督检索器在零-shot场景中性能较差，因此LameR优于自监督检索器。 |
| [^3] | [Deeply-Coupled Convolution-Transformer with Spatial-temporal Complementary Learning for Video-based Person Re-identification.](http://arxiv.org/abs/2304.14122) | 本论文结合卷积神经网络和Transformer，提出了一种名为Deeply-Coupled Convolution-Transformer的新型空时互补学习框架，用于高性能的基于视频的人员再识别，并通过互补内容注意和分层时间聚合，实验验证了其优越性能。 |
| [^4] | [Prediction then Correction: An Abductive Prediction Correction Method for Sequential Recommendation.](http://arxiv.org/abs/2304.14050) | 一种用于顺序推荐的归纳预测修正方法，通过模拟归纳推理来校正预测，从而提高推荐的准确性。 |
| [^5] | [Boosting Big Brother: Attacking Search Engines with Encodings.](http://arxiv.org/abs/2304.14031) | 通过编码方式攻击搜索引擎，以微不可见的方式扭曲文本，攻击者可以控制搜索结果。该攻击成功地影响了Google、Bing和Elasticsearch等多个搜索引擎。此外，还可以将该攻击针对搜索相关的任务如文本摘要和抄袭检测模型。需要提供一套有效的防御措施来应对这些技术带来的潜在威胁。 |
| [^6] | [Towards Explainable Collaborative Filtering with Taste Clusters Learning.](http://arxiv.org/abs/2304.13937) | 本文提出了一种利用品味聚类学习实现可解释性协同过滤的模型，在保证高准确性的同时为用户和项目提供可解释的聚类解释。 |
| [^7] | [Neural Keyphrase Generation: Analysis and Evaluation.](http://arxiv.org/abs/2304.13883) | 本文分析了三种神经网络模型（T5、CatSeq-Transformer、ExHiRD）在关键词生成任务中的性能和行为，并提出了一个新的评估框架SoftKeyScore来衡量两组关键词的相似度。 |
| [^8] | [Exploiting Simulated User Feedback for Conversational Search: Ranking, Rewriting, and Beyond.](http://arxiv.org/abs/2304.13874) | 本研究利用一个名为ConvSim的用户模拟器来评估用户反馈，从而提高会话式搜索的性能，实验结果显示有效利用用户反馈可以大幅提高检索性能。 |
| [^9] | [Extracting Structured Seed-Mediated Gold Nanorod Growth Procedures from Literature with GPT-3.](http://arxiv.org/abs/2304.13846) | 该论文提出了一种通过利用GPT-3语言模型从科学文献中自动化地提取金纳米棒合成信息的方法。这种方法可以实现高通量的探索金纳米棒的种子介导生长过程以及结果。 |
| [^10] | [STIR: Siamese Transformer for Image Retrieval Postprocessing.](http://arxiv.org/abs/2304.13393) | 这项工作提出了两部分内容。首先，他们构建了一个基于三元组损失的简单模型，性能达到了最先进水平，但没有复杂模型的缩放问题。其次，他们提出了一种新颖的后处理方法STIR，可在单个前向传递中重新排列多个顶部输出，而不依赖于全局/局部特征提取。 |
| [^11] | [Extreme Classification for Answer Type Prediction in Question Answering.](http://arxiv.org/abs/2304.12395) | 本文提出了使用Transformer模型（XBERT）进行极端多标签分类，通过将KG类型基于问题文本使用结构和语义特征进行聚类，以提高问题回答（QA）系统中语义答案类型预测（SMART）任务的性能，并获得最先进的结果。 |
| [^12] | [LongEval-Retrieval: French-English Dynamic Test Collection for Continuous Web Search Evaluation.](http://arxiv.org/abs/2303.03229) | LongEval-Retrieval是一个面向持续Web搜索评估的动态测试集合，旨在研究信息检索系统的时间持久性。每个子集合包含一组查询、文档和基于点击模型构建的软关联性评估，数据来自Qwant，一个隐私保护的Web搜索引擎。 |
| [^13] | [Local Policy Improvement for Recommender Systems.](http://arxiv.org/abs/2212.11431) | 该论文介绍了一种针对推荐系统的本地策略改进方法，不需要现场校正，易于从数据中估计，适用于以前的策略质量较高但数量较少的情况。 |

# 详细

[^1]: 通过无监督超图排名选择和融合进行人员再识别

    Person Re-ID through Unsupervised Hypergraph Rank Selection and Fusion. (arXiv:2304.14321v1 [cs.CV])

    [http://arxiv.org/abs/2304.14321](http://arxiv.org/abs/2304.14321)

    本研究提出了一种完全无监督的方法，通过流形排名聚合来选择和融合不同鲁棒的人员再识别排名器，以填补使用标记数据困难的空白。

    

    人员再识别已经引起了广泛的关注，现在在许多监控应用中具有基本重要性。该任务包括在没有重叠视图的多个摄像头之间识别个人。大多数方法都需要标记数据，但由于需求量庞大且手动为每个人指定类别的难度较大，标记数据并不总是可用的。最近的研究表明，重新排名方法能够实现显著的收益，特别是在没有标记数据的情况下。此外，特征提取器的融合和多源训练也是另一个非常有前景的研究方向，但并未被广泛利用。我们旨在通过一种流形排名聚合方法，利用不同的人员再识别排名器的互补性填补这一空白。在本工作中，我们对从多个和多样化的特征提取器获得的不同排名列表进行了完全无监督的选择和融合。

    Person Re-ID has been gaining a lot of attention and nowadays is of fundamental importance in many camera surveillance applications. The task consists of identifying individuals across multiple cameras that have no overlapping views. Most of the approaches require labeled data, which is not always available, given the huge amount of demanded data and the difficulty of manually assigning a class for each individual. Recently, studies have shown that re-ranking methods are capable of achieving significant gains, especially in the absence of labeled data. Besides that, the fusion of feature extractors and multiple-source training is another promising research direction not extensively exploited. We aim to fill this gap through a manifold rank aggregation approach capable of exploiting the complementarity of different person Re-ID rankers. In this work, we perform a completely unsupervised selection and fusion of diverse ranked lists obtained from multiple and diverse feature extractors. A
    
[^2]: 大型语言模型在零-shot检索中具有较强的表现力。

    Large Language Models are Strong Zero-Shot Retriever. (arXiv:2304.14233v1 [cs.CL])

    [http://arxiv.org/abs/2304.14233](http://arxiv.org/abs/2304.14233)

    本文提出了一种在零-shot场景下利用大型语言模型（LLM）进行大规模检索的方法。该方法通过使用查询和查询的候选答案的组合作为提示，使LLM生成更精确的答案。由于自监督检索器在零-shot场景中性能较差，因此LameR优于自监督检索器。

    

    本文提出了一种简单的方法，在零-shot场景下应用大型语言模型（LLM）进行大规模检索。我们的方法，Language Model作为检索器（LameR）仅基于大语言模型而不是其他神经模型，通过将LLM与检索器的暴力组合进行分解，将零-shot检索的性能提高到在基准数据集上具有很强的竞争力。本文主要提出通过使用查询和查询的候选答案的组合作为提示，使LLM生成更精确的答案。无论候选答案是否正确，都可以通过模式模仿或候选摘要来帮助LLM产生更精确的答案。此外，由于自监督检索器在零-shot场景中性能较差，因此通过利用LLM对文本模式的强大表现能力，LameR可以优于自监督检索器。

    In this work, we propose a simple method that applies a large language model (LLM) to large-scale retrieval in zero-shot scenarios. Our method, Language language model as Retriever (LameR) is built upon no other neural models but an LLM, while breaking up brute-force combinations of retrievers with LLMs and lifting the performance of zero-shot retrieval to be very competitive on benchmark datasets. Essentially, we propose to augment a query with its potential answers by prompting LLMs with a composition of the query and the query's in-domain candidates. The candidates, regardless of correct or wrong, are obtained by a vanilla retrieval procedure on the target collection. Such candidates, as a part of prompts, are likely to help LLM generate more precise answers by pattern imitation or candidate summarization. Even if all the candidates are wrong, the prompts at least make LLM aware of in-collection patterns and genres. Moreover, due to the low performance of a self-supervised retriever
    
[^3]: 带有空时互补学习的深度耦合卷积Transformer用于基于视频的人员再识别

    Deeply-Coupled Convolution-Transformer with Spatial-temporal Complementary Learning for Video-based Person Re-identification. (arXiv:2304.14122v1 [cs.CV])

    [http://arxiv.org/abs/2304.14122](http://arxiv.org/abs/2304.14122)

    本论文结合卷积神经网络和Transformer，提出了一种名为Deeply-Coupled Convolution-Transformer的新型空时互补学习框架，用于高性能的基于视频的人员再识别，并通过互补内容注意和分层时间聚合，实验验证了其优越性能。

    

    先进的深度卷积神经网络在基于视频的人员再识别中取得了巨大成功。然而，它们通常专注于人员的最明显的区域，具有有限的全局表示能力。最近，Transformer探索了全局观察下的补丁间关系以提高性能。在这项工作中，我们结合了卷积神经网络和Transformer，提出了一种名为Deeply-Coupled Convolution-Transformer (DCCT)的新型空时互补学习框架，用于高性能的基于视频的人员再识别。首先，我们将CNN和Transformer组合起来提取两种视觉特征，并通过实验证明了它们的互补性。进一步在空间上，我们提出了互补内容注意(CCA)来利用耦合结构，并指导独立特征进行空间互补学习。在时间上，我们提出了分层时间聚合(HTA)来逐步捕捉帧间的依赖性。

    Advanced deep Convolutional Neural Networks (CNNs) have shown great success in video-based person Re-Identification (Re-ID). However, they usually focus on the most obvious regions of persons with a limited global representation ability. Recently, it witnesses that Transformers explore the inter-patch relations with global observations for performance improvements. In this work, we take both sides and propose a novel spatial-temporal complementary learning framework named Deeply-Coupled Convolution-Transformer (DCCT) for high-performance video-based person Re-ID. Firstly, we couple CNNs and Transformers to extract two kinds of visual features and experimentally verify their complementarity. Further, in spatial, we propose a Complementary Content Attention (CCA) to take advantages of the coupled structure and guide independent features for spatial complementary learning. In temporal, a Hierarchical Temporal Aggregation (HTA) is proposed to progressively capture the inter-frame dependenc
    
[^4]: 预测再修正：一种用于顺序推荐的归纳预测修正方法

    Prediction then Correction: An Abductive Prediction Correction Method for Sequential Recommendation. (arXiv:2304.14050v1 [cs.IR])

    [http://arxiv.org/abs/2304.14050](http://arxiv.org/abs/2304.14050)

    一种用于顺序推荐的归纳预测修正方法，通过模拟归纳推理来校正预测，从而提高推荐的准确性。

    

    顺序推荐模型通常会在测试过程中一步生成预测，而不考虑额外的预测修正以提高性能，这影响了模型的准确性。为了解决这个问题，我们提出了一种称为“Abductive Prediction Correction”（APC）的框架，该框架通过归纳推理校正预测，从而提高推荐的准确性。

    Sequential recommender models typically generate predictions in a single step during testing, without considering additional prediction correction to enhance performance as humans would. To improve the accuracy of these models, some researchers have attempted to simulate human analogical reasoning to correct predictions for testing data by drawing analogies with the prediction errors of similar training data. However, there are inherent gaps between testing and training data, which can make this approach unreliable. To address this issue, we propose an \textit{Abductive Prediction Correction} (APC) framework for sequential recommendation. Our approach simulates abductive reasoning to correct predictions. Specifically, we design an abductive reasoning task that infers the most probable historical interactions from the future interactions predicted by a recommender, and minimizes the discrepancy between the inferred and true historical interactions to adjust the predictions.We perform th
    
[^5]: 提升老大哥：采用编码方式攻击搜索引擎

    Boosting Big Brother: Attacking Search Engines with Encodings. (arXiv:2304.14031v1 [cs.CR])

    [http://arxiv.org/abs/2304.14031](http://arxiv.org/abs/2304.14031)

    通过编码方式攻击搜索引擎，以微不可见的方式扭曲文本，攻击者可以控制搜索结果。该攻击成功地影响了Google、Bing和Elasticsearch等多个搜索引擎。此外，还可以将该攻击针对搜索相关的任务如文本摘要和抄袭检测模型。需要提供一套有效的防御措施来应对这些技术带来的潜在威胁。

    

    搜索引擎对于文本编码操纵的索引和搜索存在漏洞。通过以不常见的编码表示形式微不可见地扭曲文本，攻击者可以控制特定搜索查询在多个搜索引擎上的结果。我们演示了这种攻击成功地针对了两个主要的商业搜索引擎——Google和Bing——以及一个开源搜索引擎——Elasticsearch。我们进一步展示了这种攻击成功地针对了包括Bing的GPT-4聊天机器人和Google的Bard聊天机器人在内的LLM聊天搜索。我们还提出了一种变体攻击，针对与搜索密切相关的两个ML任务——文本摘要和抄袭检测模型。我们提供了一套针对这些技术的防御措施，并警告攻击者可以利用这些攻击启动反信息争夺战。这促使搜索引擎维护人员修补已部署的系统。

    Search engines are vulnerable to attacks against indexing and searching via text encoding manipulation. By imperceptibly perturbing text using uncommon encoded representations, adversaries can control results across search engines for specific search queries. We demonstrate that this attack is successful against two major commercial search engines - Google and Bing - and one open source search engine - Elasticsearch. We further demonstrate that this attack is successful against LLM chat search including Bing's GPT-4 chatbot and Google's Bard chatbot. We also present a variant of the attack targeting text summarization and plagiarism detection models, two ML tasks closely tied to search. We provide a set of defenses against these techniques and warn that adversaries can leverage these attacks to launch disinformation campaigns against unsuspecting users, motivating the need for search engine maintainers to patch deployed systems.
    
[^6]: 用品味聚类学习实现可解释性协同过滤

    Towards Explainable Collaborative Filtering with Taste Clusters Learning. (arXiv:2304.13937v1 [cs.IR])

    [http://arxiv.org/abs/2304.13937](http://arxiv.org/abs/2304.13937)

    本文提出了一种利用品味聚类学习实现可解释性协同过滤的模型，在保证高准确性的同时为用户和项目提供可解释的聚类解释。

    

    协同过滤是推荐系统中广泛使用且有效的技术。近年来，基于潜在嵌入的协同过滤方法（如矩阵分解、神经协同过滤和LightGCN）已经有了显著的进展，以提高准确性。但是，这些模型的可解释性尚未得到充分探索。给推荐模型添加解释性，不仅可以增加人们对决策过程的信任，而且还有多个好处，如为项目推荐提供有说服力的解释、为用户和项目创建明确的文件、为项目制造商提供设计改进的协助。在本文中，我们提出了一种清晰有效的可解释性协同过滤模型，利用可解释的聚类学习来实现两个最苛刻的目标：（1）精确——模型在追求可解释性时不应妥协准确性；（2）自我解释——模型的解释应易于人们理解。引入品味聚类学习来构成用户和项目的解释，并在四个真实数据集上进行实验，结果证实了我们提出的方法在提供人类可理解的解释的同时保证了高准确性。

    Collaborative Filtering (CF) is a widely used and effective technique for recommender systems. In recent decades, there have been significant advancements in latent embedding-based CF methods for improved accuracy, such as matrix factorization, neural collaborative filtering, and LightGCN. However, the explainability of these models has not been fully explored. Adding explainability to recommendation models can not only increase trust in the decisionmaking process, but also have multiple benefits such as providing persuasive explanations for item recommendations, creating explicit profiles for users and items, and assisting item producers in design improvements.  In this paper, we propose a neat and effective Explainable Collaborative Filtering (ECF) model that leverages interpretable cluster learning to achieve the two most demanding objectives: (1) Precise - the model should not compromise accuracy in the pursuit of explainability; and (2) Self-explainable - the model's explanations 
    
[^7]: 神经关键词生成：分析与评估

    Neural Keyphrase Generation: Analysis and Evaluation. (arXiv:2304.13883v1 [cs.CL])

    [http://arxiv.org/abs/2304.13883](http://arxiv.org/abs/2304.13883)

    本文分析了三种神经网络模型（T5、CatSeq-Transformer、ExHiRD）在关键词生成任务中的性能和行为，并提出了一个新的评估框架SoftKeyScore来衡量两组关键词的相似度。

    

    关键词生成旨在通过从原始文本中复制（现有关键词）或生成捕捉文本语义意义的新关键词（缺失关键词）来生成话题短语。编码器-解码器模型在此任务中最广泛使用，因为它们具有生成缺失关键词的能力。然而，目前几乎没有对此类模型在关键词生成中的性能和行为进行分析。本文研究了三种强劲模型 T5（基于预训练变压器）、CatSeq-Transformer（非预训练变压器）和 ExHiRD（基于循环神经网络）所展示的各种趋势。我们分析了预测置信度分数、模型校准以及词元位置对关键词生成的影响。此外，我们提出并推动一个新的度量框架SoftKeyScore，通过使用 softscores 来计算部分匹配的相似度来评估两组关键词的相似性。

    Keyphrase generation aims at generating topical phrases from a given text either by copying from the original text (present keyphrases) or by producing new keyphrases (absent keyphrases) that capture the semantic meaning of the text. Encoder-decoder models are most widely used for this task because of their capabilities for absent keyphrase generation. However, there has been little to no analysis on the performance and behavior of such models for keyphrase generation. In this paper, we study various tendencies exhibited by three strong models: T5 (based on a pre-trained transformer), CatSeq-Transformer (a non-pretrained Transformer), and ExHiRD (based on a recurrent neural network). We analyze prediction confidence scores, model calibration, and the effect of token position on keyphrases generation. Moreover, we motivate and propose a novel metric framework, SoftKeyScore, to evaluate the similarity between two sets of keyphrases by using softscores to account for partial matching and 
    
[^8]: 利用模拟用户反馈的方式来优化会话式搜索

    Exploiting Simulated User Feedback for Conversational Search: Ranking, Rewriting, and Beyond. (arXiv:2304.13874v1 [cs.IR])

    [http://arxiv.org/abs/2304.13874](http://arxiv.org/abs/2304.13874)

    本研究利用一个名为ConvSim的用户模拟器来评估用户反馈，从而提高会话式搜索的性能，实验结果显示有效利用用户反馈可以大幅提高检索性能。

    

    本研究旨在探索评估用户反馈在混合倡议的会话式搜索系统中的各种方法。虽然会话式搜索系统在多个方面都取得了重大进展，但最近的研究未能成功地将用户反馈纳入系统中。其中一个主要原因是缺乏系统-用户对话交互数据。为此，我们提出了一种基于用户模拟器的框架，可用于与各种混合倡议的会话式搜索系统进行多轮交互。具体来说，我们开发了一个名为ConvSim的用户模拟器，一旦初始化了信息需求描述，就能够对系统的响应提供反馈，并回答潜在的澄清问题。我们对各种最先进的段落检索和神经重新排序模型进行的实验表明，有效利用用户反馈可以导致在nDCG@3方面16%的检索性能提高。此外，我们观察到随着n的增加，一致的改进。

    This research aims to explore various methods for assessing user feedback in mixed-initiative conversational search (CS) systems. While CS systems enjoy profuse advancements across multiple aspects, recent research fails to successfully incorporate feedback from the users. One of the main reasons for that is the lack of system-user conversational interaction data. To this end, we propose a user simulator-based framework for multi-turn interactions with a variety of mixed-initiative CS systems. Specifically, we develop a user simulator, dubbed ConvSim, that, once initialized with an information need description, is capable of providing feedback to a system's responses, as well as answering potential clarifying questions. Our experiments on a wide variety of state-of-the-art passage retrieval and neural re-ranking models show that effective utilization of user feedback can lead to 16% retrieval performance increase in terms of nDCG@3. Moreover, we observe consistent improvements as the n
    
[^9]: 从文献中提取结构化的种子介导金纳米棒生长方法：基于GPT-3的研究

    Extracting Structured Seed-Mediated Gold Nanorod Growth Procedures from Literature with GPT-3. (arXiv:2304.13846v1 [physics.app-ph])

    [http://arxiv.org/abs/2304.13846](http://arxiv.org/abs/2304.13846)

    该论文提出了一种通过利用GPT-3语言模型从科学文献中自动化地提取金纳米棒合成信息的方法。这种方法可以实现高通量的探索金纳米棒的种子介导生长过程以及结果。

    

    尽管金纳米棒已经成为研究的热点，但控制它们的形状，从而控制它们的光学特性的途径仍然很大程度上是基于经验的。尽管合成过程中不同试剂物之间的共存和相互作用控制着这些特性，但在实践中，探索合成空间的计算和实验方法可能会极其繁琐或耗费过多时间。因此，我们提出了一种利用科学文献中已经包含的大量合成信息来自动化提取相关结构化数据的方法，以高通量方式探寻金纳米棒种子介导生长过程以及结果。为此，我们使用强大的GPT-3语言模型提出了一种方法，来从非结构化科学文本中提取金纳米棒的结构化多步种子介导生长过程和结果。将GPT-3的提示完成进行微调，以预测JSON文档形式的合成模板。

    Although gold nanorods have been the subject of much research, the pathways for controlling their shape and thereby their optical properties remain largely heuristically understood. Although it is apparent that the simultaneous presence of and interaction between various reagents during synthesis control these properties, computational and experimental approaches for exploring the synthesis space can be either intractable or too time-consuming in practice. This motivates an alternative approach leveraging the wealth of synthesis information already embedded in the body of scientific literature by developing tools to extract relevant structured data in an automated, high-throughput manner. To that end, we present an approach using the powerful GPT-3 language model to extract structured multi-step seed-mediated growth procedures and outcomes for gold nanorods from unstructured scientific text. GPT-3 prompt completions are fine-tuned to predict synthesis templates in the form of JSON docu
    
[^10]: STIR：用于图像检索后处理的Siamese Transformer（arXiv：2304.13393v1 [cs.IR]）

    STIR: Siamese Transformer for Image Retrieval Postprocessing. (arXiv:2304.13393v1 [cs.IR])

    [http://arxiv.org/abs/2304.13393](http://arxiv.org/abs/2304.13393)

    这项工作提出了两部分内容。首先，他们构建了一个基于三元组损失的简单模型，性能达到了最先进水平，但没有复杂模型的缩放问题。其次，他们提出了一种新颖的后处理方法STIR，可在单个前向传递中重新排列多个顶部输出，而不依赖于全局/局部特征提取。

    

    当前，图像检索的度量学习方法通常基于学习具有信息的潜在表示空间，其中简单的方法如余弦距离将表现良好。最近的最先进方法（如HypViT）转向更复杂的嵌入空间，可能会产生更好的结果，但更难以扩展到生产环境中。在这项工作中，我们首先构建了一个基于三元组损失的简单模型，具有硬负例挖掘，性能达到了最先进水平，但没有这些缺点。其次，我们引入了一种新颖的图像检索后处理方法，称为用于图像检索的Siamese Transformer（STIR），可在单个前向传递中重新排列多个顶部输出。与先前提出的重排变压器不同，STIR不依赖于全局/局部特征提取，而是借助注意机制直接在像素级别比较查询图像和检索到的候选图像。由此得出的方法定义了一个新的最先进水平。

    Current metric learning approaches for image retrieval are usually based on learning a space of informative latent representations where simple approaches such as the cosine distance will work well. Recent state of the art methods such as HypViT move to more complex embedding spaces that may yield better results but are harder to scale to production environments. In this work, we first construct a simpler model based on triplet loss with hard negatives mining that performs at the state of the art level but does not have these drawbacks. Second, we introduce a novel approach for image retrieval postprocessing called Siamese Transformer for Image Retrieval (STIR) that reranks several top outputs in a single forward pass. Unlike previously proposed Reranking Transformers, STIR does not rely on global/local feature extraction and directly compares a query image and a retrieved candidate on pixel level with the usage of attention mechanism. The resulting approach defines a new state of the 
    
[^11]: 问题回答中的答案类型预测的极限分类

    Extreme Classification for Answer Type Prediction in Question Answering. (arXiv:2304.12395v1 [cs.CL])

    [http://arxiv.org/abs/2304.12395](http://arxiv.org/abs/2304.12395)

    本文提出了使用Transformer模型（XBERT）进行极端多标签分类，通过将KG类型基于问题文本使用结构和语义特征进行聚类，以提高问题回答（QA）系统中语义答案类型预测（SMART）任务的性能，并获得最先进的结果。

    

    语义答案类型预测（SMART）已被证明是有效的问题回答（QA）系统的有用步骤。 SMART任务涉及预测给定自然语言问题的前k个知识图（KG）类型。由于KG中存在大量类型，这是具有挑战性的。在本文中，我们提出使用Transformer模型（XBERT）进行极端多标签分类，通过将KG类型基于问题文本使用结构和语义特征进行聚类。我们具体地改善了XBERT流程的聚类阶段，利用从KG中派生的文本和结构特征。我们表明，这些特征可以提高SMART任务的端到端性能，并产生最先进的结果。

    Semantic answer type prediction (SMART) is known to be a useful step towards effective question answering (QA) systems. The SMART task involves predicting the top-$k$ knowledge graph (KG) types for a given natural language question. This is challenging due to the large number of types in KGs. In this paper, we propose use of extreme multi-label classification using Transformer models (XBERT) by clustering KG types using structural and semantic features based on question text. We specifically improve the clustering stage of the XBERT pipeline using textual and structural features derived from KGs. We show that these features can improve end-to-end performance for the SMART task, and yield state-of-the-art results.
    
[^12]: LongEval-Retrieval: 面向持续Web搜索评估的法英动态测试集合

    LongEval-Retrieval: French-English Dynamic Test Collection for Continuous Web Search Evaluation. (arXiv:2303.03229v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2303.03229](http://arxiv.org/abs/2303.03229)

    LongEval-Retrieval是一个面向持续Web搜索评估的动态测试集合，旨在研究信息检索系统的时间持久性。每个子集合包含一组查询、文档和基于点击模型构建的软关联性评估，数据来自Qwant，一个隐私保护的Web搜索引擎。

    

    LongEval-Retrieval是一个Web文档检索基准，专注于持续检索评估。该测试集合旨在用于研究信息检索系统的时间持久性，并将用作CLEF 2023的Longitudinal Evaluation of Model Performance Track (LongEval)的测试集合。该基准模拟了一个不断演变的信息系统环境，例如Web搜索引擎所处的环境，在遵循离线评估的Cranfield范例的同时，文档集合、查询分布和相关性都在不断移动。为此，我们引入了动态测试集合的概念，由连续的子集合组成，每个子集合表示信息系统在给定时间步骤的状态。在LongEval-Retrieval中，每个子集合包含一组查询、文档和基于点击模型构建的软关联性评估。这些数据来自Qwant，一个隐私保护的Web搜索引擎。

    LongEval-Retrieval is a Web document retrieval benchmark that focuses on continuous retrieval evaluation. This test collection is intended to be used to study the temporal persistence of Information Retrieval systems and will be used as the test collection in the Longitudinal Evaluation of Model Performance Track (LongEval) at CLEF 2023. This benchmark simulates an evolving information system environment - such as the one a Web search engine operates in - where the document collection, the query distribution, and relevance all move continuously, while following the Cranfield paradigm for offline evaluation. To do that, we introduce the concept of a dynamic test collection that is composed of successive sub-collections each representing the state of an information system at a given time step. In LongEval-Retrieval, each sub-collection contains a set of queries, documents, and soft relevance assessments built from click models. The data comes from Qwant, a privacy-preserving Web search e
    
[^13]: 推荐系统的本地策略改进

    Local Policy Improvement for Recommender Systems. (arXiv:2212.11431v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2212.11431](http://arxiv.org/abs/2212.11431)

    该论文介绍了一种针对推荐系统的本地策略改进方法，不需要现场校正，易于从数据中估计，适用于以前的策略质量较高但数量较少的情况。

    

    推荐系统基于用户过去的互动行为而预测他们可能会与哪些项目交互。解决该问题的常用方法是通过监督学习，但最近的进展转向了基于奖励（例如用户参与度）的策略优化。后者面临的挑战之一是策略不匹配：我们只能基于以前部署策略收集到的数据来训练新的策略。传统的方法是通过重要性采样校正来解决这个问题，但这种方法存在实际限制。我们建议一种不需要现场校正的本地策略改进方法。我们的方法计算和优化目标策略预期奖励的下限，这易于从数据中估计并且不涉及密度比（例如在重要性采样校正中出现的比率）。这种本地策略改进范例非常适用于推荐系统，因为以前的策略通常质量较高，策略的数量也很少。

    Recommender systems predict what items a user will interact with next, based on their past interactions. The problem is often approached through supervised learning, but recent advancements have shifted towards policy optimization of rewards (e.g., user engagement). One challenge with the latter is policy mismatch: we are only able to train a new policy given data collected from a previously-deployed policy. The conventional way to address this problem is through importance sampling correction, but this comes with practical limitations. We suggest an alternative approach of local policy improvement without off-policy correction. Our method computes and optimizes a lower bound of expected reward of the target policy, which is easy to estimate from data and does not involve density ratios (such as those appearing in importance sampling correction). This local policy improvement paradigm is ideal for recommender systems, as previous policies are typically of decent quality and policies ar
    

