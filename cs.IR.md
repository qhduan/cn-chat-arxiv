# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Retrieval Augmented Chest X-Ray Report Generation using OpenAI GPT models.](http://arxiv.org/abs/2305.03660) | 该研究提出了一种检索增强的方法，利用对比预训练的视觉语言模型的多模态对齐嵌入来检索相应的候选放射学文本，并使用通用领域生成模型来生成报告，可抑制虚构的生成，实现更好的临床指标。 |
| [^2] | [Query Expansion by Prompting Large Language Models.](http://arxiv.org/abs/2305.03653) | 本文提出了一种利用大型语言模型进行查询扩展的方法，相比传统方法具有更好的表现，特别是Chain-of-Thought提示对于查询扩展有着重要的作用。 |
| [^3] | [Retraining A Graph-based Recommender with Interests Disentanglement.](http://arxiv.org/abs/2305.03624) | 本文提出了Disentangled Incremental Learning (DIL)框架，可以通过兴趣解耦的方式重新训练基于图的推荐系统，具有较高的准确性和效率。 |
| [^4] | [Read it Twice: Towards Faithfully Interpretable Fact Verification by Revisiting Evidence.](http://arxiv.org/abs/2305.03507) | 本研究提出了一种名为“ ReRead”的事实验证模型，其以两个阶段（证据检索和声明验证）实现准确且可解释的事实验证，并在FEVER数据集上实现了最先进的结果。 |
| [^5] | [Think Rationally about What You See: Continuous Rationale Extraction for Relation Extraction.](http://arxiv.org/abs/2305.03503) | 本研究提出了一种新颖的关系提取的理据抽取框架RE2，利用两个连续性和稀疏性因素从句子中获取相关而连贯的理据，解决了如何保留相关内容并从句子中去掉噪音段落的问题。实验结果显示我们的模型优于现有的最先进方法，并且提取出的理据对于推理和解释是有用的。 |
| [^6] | [Augmenting Low-Resource Text Classification with Graph-Grounded Pre-training and Prompting.](http://arxiv.org/abs/2305.03324) | 本文提出一种名为G2P2的模型，使用图谱预训练和提示的方式增强低资源文本分类，实验证明该模型优于现有的最先进方法。 |
| [^7] | [Influence of various text embeddings on clustering performance in NLP.](http://arxiv.org/abs/2305.03144) | 研究探索了不同文本嵌入对聚类算法（KMeans、单链接聚合等级、DBSCAN和HDBSCAN）性能的影响，并应用于评论聚类领域。 |
| [^8] | [Optimizing SMS Reminder Campaigns for Pre- and Post-Diagnosis Cancer Check-Ups using Socio-Demographics: An In-Silco Investigation Into Bladder Cancer.](http://arxiv.org/abs/2305.03126) | 本研究提出一个框架，基于社会人口学的特点，对癌症检查的短信提醒活动进行了优化，研究结果表明，光基于这些特征进行短信提醒活动可以将死亡率的统计数量降低5.8％。 |
| [^9] | [An Analysis of Fusion Functions for Hybrid Retrieval.](http://arxiv.org/abs/2210.11934) | 本文研究了混合搜索的方法，将词汇和语义搜索融合在一起，在此基础上分析了利用凸组合和互惠排名融合两种方法的优缺点。我们发现凸组合比互惠排名融合更加优秀，具有样本效率，并且只需要少量训练集即可调整参数以适应目标领域。 |
| [^10] | [LOGEN: Few-shot Logical Knowledge-Conditioned Text Generation with Self-training.](http://arxiv.org/abs/2112.01404) | 本文提出了一种基于少样本的逻辑知识条件下文本生成的统一框架LOGEN，通过自训练和基于内容和结构一致性抽样伪逻辑形式，实现了在少量样本下的文本生成。 |

# 详细

[^1]: 利用OpenAI GPT模型的检索增强的胸部X射线报告生成

    Retrieval Augmented Chest X-Ray Report Generation using OpenAI GPT models. (arXiv:2305.03660v1 [cs.CL])

    [http://arxiv.org/abs/2305.03660](http://arxiv.org/abs/2305.03660)

    该研究提出了一种检索增强的方法，利用对比预训练的视觉语言模型的多模态对齐嵌入来检索相应的候选放射学文本，并使用通用领域生成模型来生成报告，可抑制虚构的生成，实现更好的临床指标。

    

    我们提出了一种名为Retrieval Augmented Generation (RAG) 的方法来自动生成放射学报告，该方法利用对比预训练的视觉语言模型的多模态对齐嵌入来检索相应的候选放射学文本，并使用像OpenAI text-davinci-003、gpt-3.5-turbo和gpt-4这样的通用领域生成模型来生成报告。该方法可以抑制虚构的生成并提供指令跟随能力，以我们所需的格式生成报告内容。我们的方法实现了更好的临床指标，BERTScore为0.2865（Δ+25.88%），Semb Score为0.4026（Δ+6.31%）。我们的方法可以广泛应用于不同的临床设置，因为它允许增强自动生成的放射学报告过程，同时具备适合该设置的相关内容的能力。

    We propose Retrieval Augmented Generation (RAG) as an approach for automated radiology report writing that leverages multimodally aligned embeddings from a contrastively pretrained vision language model for retrieval of relevant candidate radiology text for an input radiology image and a general domain generative model like OpenAI text-davinci-003, gpt-3.5-turbo and gpt-4 for report generation using the relevant radiology text retrieved. This approach keeps hallucinated generations under check and provides capabilities to generate report content in the format we desire leveraging the instruction following capabilities of these generative models. Our approach achieves better clinical metrics with a BERTScore of 0.2865 ({\Delta}+ 25.88%) and Semb score of 0.4026 ({\Delta}+ 6.31%). Our approach can be broadly relevant for different clinical settings as it allows to augment the automated radiology report generation process with content relevant for that setting while also having the abilit
    
[^2]: 利用大语言模型促进查询扩展

    Query Expansion by Prompting Large Language Models. (arXiv:2305.03653v1 [cs.IR])

    [http://arxiv.org/abs/2305.03653](http://arxiv.org/abs/2305.03653)

    本文提出了一种利用大型语言模型进行查询扩展的方法，相比传统方法具有更好的表现，特别是Chain-of-Thought提示对于查询扩展有着重要的作用。

    

    查询扩展是提高搜索系统召回率的常用技术。本文提出了一种利用大型语言模型（LLM）的生成能力进行查询扩展的方法。与传统的查询扩展方法如“伪相关反馈”（PRF）依赖于检索一组好的伪相关文档来扩展查询相比，我们依赖LLM的生成和创造能力，并利用模型固有的知识。我们研究了各种不同的提示，包括零-shot、few-shot和Chain-of-Thought（CoT）。我们发现CoT提示对于查询扩展特别有用，因为这些提示指示模型逐步分解查询，并可以提供与原始查询相关的大量术语。在MS-MARCO和BEIR上的实验结果表明，LLM生成的查询扩展可以比传统的查询扩展方法更具优势。

    Query expansion is a widely used technique to improve the recall of search systems. In this paper, we propose an approach to query expansion that leverages the generative abilities of Large Language Models (LLMs). Unlike traditional query expansion approaches such as Pseudo-Relevance Feedback (PRF) that relies on retrieving a good set of pseudo-relevant documents to expand queries, we rely on the generative and creative abilities of an LLM and leverage the knowledge inherent in the model. We study a variety of different prompts, including zero-shot, few-shot and Chain-of-Thought (CoT). We find that CoT prompts are especially useful for query expansion as these prompts instruct the model to break queries down step-by-step and can provide a large number of terms related to the original query. Experimental results on MS-MARCO and BEIR demonstrate that query expansions generated by LLMs can be more powerful than traditional query expansion methods.
    
[^3]: 通过兴趣解耦重新训练基于图的推荐系统

    Retraining A Graph-based Recommender with Interests Disentanglement. (arXiv:2305.03624v1 [cs.IR])

    [http://arxiv.org/abs/2305.03624](http://arxiv.org/abs/2305.03624)

    本文提出了Disentangled Incremental Learning (DIL)框架，可以通过兴趣解耦的方式重新训练基于图的推荐系统，具有较高的准确性和效率。

    

    在实际的推荐系统中，会不断观察到新的交互。一些交互是预期的，因为它们大部分遵循用户的长期偏好。其他一些交互则表明用户偏好的最新趋势或新物品的营销立场。因此，推荐算法需要周期性地重新训练或更新，以捕捉新的趋势，同时不要忘记长期偏好。本文提出了一种称为Disentangled Incremental Learning（DIL）的新型通用重新训练框架，用于基于图的推荐系统。假设长期偏好已经以学习自过去交互的模型参数的形式在现有模型中得到良好捕捉。新偏好可以通过使用新观察到的交互构建的用户-物品二分图来学习。在Disentangled Incremental Learning（DIL）中，设计了信息提取模块来从现有模型中提取历史偏好。然后，我们通过基于新设计的正则化项的Disentangled Information Embedding（解耦信息嵌入）来混合历史和新偏好。解耦的嵌入可以直接用于推荐或下游任务。我们在三个基准数据集上进行了广泛的实验。实验结果表明，DIL在推荐准确性和效率方面均优于多个最新的方法。

    In a practical recommender system, new interactions are continuously observed. Some interactions are expected, because they largely follow users' long-term preferences. Some other interactions are indications of recent trends in user preference changes or marketing positions of new items. Accordingly, the recommender needs to be periodically retrained or updated to capture the new trends, and yet not to forget the long-term preferences. In this paper, we propose a novel and generic retraining framework called Disentangled Incremental Learning (DIL) for graph-based recommenders. We assume that long-term preferences are well captured in the existing model, in the form of model parameters learned from past interactions. New preferences can be learned from the user-item bipartite graph constructed using the newly observed interactions. In DIL, we design an Information Extraction Module to extract historical preferences from the existing model. Then we blend the historical and new preferenc
    
[^4]: 读两遍：通过重新审视证据实现准确且可解释的事实验证

    Read it Twice: Towards Faithfully Interpretable Fact Verification by Revisiting Evidence. (arXiv:2305.03507v1 [cs.CL])

    [http://arxiv.org/abs/2305.03507](http://arxiv.org/abs/2305.03507)

    本研究提出了一种名为“ ReRead”的事实验证模型，其以两个阶段（证据检索和声明验证）实现准确且可解释的事实验证，并在FEVER数据集上实现了最先进的结果。

    

    现实世界中的事实验证任务旨在通过从原始文档中检索证据来验证声明的事实性。 检索到的证据的质量在该任务中起着重要作用。 理想情况下，检索到的证据应该是可信的（反映了模型在声明验证中的决策过程）且合理的（对人类有说服力），并能提高验证任务的准确性。 尽管现有的方法利用声明和文档之间的语义或表面形式的相似性度量来检索证据，但它们都依赖于某些启发式方法，这些方法会阻止它们满足所有三个要求。 鉴于此，我们提出了一种名为“ ReRead”的事实验证模型，以检索证据并验证声明，该模型具有以下两个阶段：1）证据检索阶段，该阶段通过使用忠实且合理的证据取回器来获取可解释的证据；2）声明验证阶段，该阶段重新审视检索到的证据以验证声明。 我们在广泛使用的FEVER数据集上验证了所提出的模型，实验结果表明，我们的模型取得了最先进的结果。

    Real-world fact verification task aims to verify the factuality of a claim by retrieving evidence from the source document. The quality of the retrieved evidence plays an important role in claim verification. Ideally, the retrieved evidence should be faithful (reflecting the model's decision-making process in claim verification) and plausible (convincing to humans), and can improve the accuracy of verification task. Although existing approaches leverage the similarity measure of semantic or surface form between claims and documents to retrieve evidence, they all rely on certain heuristics that prevent them from satisfying all three requirements. In light of this, we propose a fact verification model named ReRead to retrieve evidence and verify claim that: (1) Train the evidence retriever to obtain interpretable evidence (i.e., faithfulness and plausibility criteria); (2) Train the claim verifier to revisit the evidence retrieved by the optimized evidence retriever to improve the accura
    
[^5]: 合理看待所看到的：关系提取的连续理据抽取

    Think Rationally about What You See: Continuous Rationale Extraction for Relation Extraction. (arXiv:2305.03503v1 [cs.CL])

    [http://arxiv.org/abs/2305.03503](http://arxiv.org/abs/2305.03503)

    本研究提出了一种新颖的关系提取的理据抽取框架RE2，利用两个连续性和稀疏性因素从句子中获取相关而连贯的理据，解决了如何保留相关内容并从句子中去掉噪音段落的问题。实验结果显示我们的模型优于现有的最先进方法，并且提取出的理据对于推理和解释是有用的。

    

    关系提取旨在根据两个实体的语境提取潜在关系，因此，从句子中推导出合理的语境非常重要。以往的研究要么专注于如何利用实体信息（例如，实体类型，实体用语）来推断关系，但忽略了以语境为重点的内容，要么使用反事实思维来消除模型对实体潜在关系的偏见，但关系推理过程仍会受到无关内容的干扰。因此，如何保留有关内容并从句子中去掉噪音段落是一项关键任务。此外，保留的内容需要足够流畅，以保持语义的连贯性和可解释性。在这项工作中，我们提出了一种新颖的理据抽取框架RE2，它利用两个连续性和稀疏性因素从句子中获取相关而连贯的理据。为了解决黄金理据未标记的问题，RE2应用一种无监督方法生成候选理据，并选择最相关和连贯的理据来指导RE模型。两个基准数据集上的实验结果表明，我们的模型优于现有的最先进方法，并且提取出的理据对于推理和解释是有用的。

    Relation extraction (RE) aims to extract potential relations according to the context of two entities, thus, deriving rational contexts from sentences plays an important role. Previous works either focus on how to leverage the entity information (e.g., entity types, entity verbalization) to inference relations, but ignore context-focused content, or use counterfactual thinking to remove the model's bias of potential relations in entities, but the relation reasoning process will still be hindered by irrelevant content. Therefore, how to preserve relevant content and remove noisy segments from sentences is a crucial task. In addition, retained content needs to be fluent enough to maintain semantic coherence and interpretability. In this work, we propose a novel rationale extraction framework named RE2, which leverages two continuity and sparsity factors to obtain relevant and coherent rationales from sentences. To solve the problem that the gold rationales are not labeled, RE2 applies an
    
[^6]: 用图谱预训练和提示增强低资源文本分类

    Augmenting Low-Resource Text Classification with Graph-Grounded Pre-training and Prompting. (arXiv:2305.03324v1 [cs.IR])

    [http://arxiv.org/abs/2305.03324](http://arxiv.org/abs/2305.03324)

    本文提出一种名为G2P2的模型，使用图谱预训练和提示的方式增强低资源文本分类，实验证明该模型优于现有的最先进方法。

    

    文本分类是信息检索中的一个基本问题，具有许多实际应用，例如预测在线文章的主题和电子商务产品描述的类别。然而，低资源文本分类，没有或只有很少标记样本，对于监督学习来说是一个严重的问题。同时，许多文本数据本质上基于网络结构，例如在线文章的超链接/引用网络和电子商务产品的用户-项目购买网络。这些图形结构捕捉了丰富的语义关系，可以潜在地增强低资源文本分类。本文提出了一种新颖的模型，称为图形基础预训练和提示（G2P2），以两个方面解决低资源文本分类。在预训练期间，我们提出了三种基于图形交互的对比策略，以联合预训练图形-文本模型；在下游分类过程中，我们探索提示进行从高资源到低资源任务的迁移学习。在四个低资源基准测试上的实验表明，G2P2显着优于先前的最先进方法，我们的分析表明，图形接地和提示策略对于利用辅助知识进行低资源文本分类是有效的。

    Text classification is a fundamental problem in information retrieval with many real-world applications, such as predicting the topics of online articles and the categories of e-commerce product descriptions. However, low-resource text classification, with few or no labeled samples, poses a serious concern for supervised learning. Meanwhile, many text data are inherently grounded on a network structure, such as a hyperlink/citation network for online articles, and a user-item purchase network for e-commerce products. These graph structures capture rich semantic relationships, which can potentially augment low-resource text classification. In this paper, we propose a novel model called Graph-Grounded Pre-training and Prompting (G2P2) to address low-resource text classification in a two-pronged approach. During pre-training, we propose three graph interaction-based contrastive strategies to jointly pre-train a graph-text model; during downstream classification, we explore prompting for t
    
[^7]: 文本嵌入对NLP聚类性能的影响。

    Influence of various text embeddings on clustering performance in NLP. (arXiv:2305.03144v1 [cs.LG])

    [http://arxiv.org/abs/2305.03144](http://arxiv.org/abs/2305.03144)

    研究探索了不同文本嵌入对聚类算法（KMeans、单链接聚合等级、DBSCAN和HDBSCAN）性能的影响，并应用于评论聚类领域。

    

    随着电子商务平台的出现，评论对于顾客评估产品的可信度至关重要。但是，星级评分并不总是与顾客编写的评论文本相匹配。在本研究中，我们探索了选择不同文本嵌入来表示这些评论的任务，并探究了嵌入选择对各种类型聚类算法性能的影响。我们使用上下文（BERT）和非上下文（Word2Vec）文本嵌入来表示文本，并测量它们对三种聚类算法（基于分区的KMeans、单链接聚合等级和密度基础的DBSCAN和HDBSCAN）在不同实验设置下的影响。

    With the advent of e-commerce platforms, reviews are crucial for customers to assess the credibility of a product. The star ratings do not always match the review text written by the customer. For example, a three star rating (out of five) may be incongruous with the review text, which may be more suitable for a five star review. A clustering approach can be used to relabel the correct star ratings by grouping the text reviews into individual groups. In this work, we explore the task of choosing different text embeddings to represent these reviews and also explore the impact the embedding choice has on the performance of various classes of clustering algorithms. We use contextual (BERT) and non-contextual (Word2Vec) text embeddings to represent the text and measure their impact of three classes on clustering algorithms - partitioning based (KMeans), single linkage agglomerative hierarchical, and density based (DBSCAN and HDBSCAN), each with various experimental settings. We use the sil
    
[^8]: 利用社会人口学优化前后癌症检查的短信提醒活动：对膀胱癌的计算机模拟研究

    Optimizing SMS Reminder Campaigns for Pre- and Post-Diagnosis Cancer Check-Ups using Socio-Demographics: An In-Silco Investigation Into Bladder Cancer. (arXiv:2305.03126v1 [stat.AP])

    [http://arxiv.org/abs/2305.03126](http://arxiv.org/abs/2305.03126)

    本研究提出一个框架，基于社会人口学的特点，对癌症检查的短信提醒活动进行了优化，研究结果表明，光基于这些特征进行短信提醒活动可以将死亡率的统计数量降低5.8％。

    

    及时进行癌症的预后和诊断检查对于各类型的癌症患者来说都至关重要，这通常会带来更好的治疗效果。不幸的是，现有的检查政策只考虑与癌症的临床动力学密切相关的属性。本研究提出了一种新的框架和高分辨率计算机模拟，以调查和优化基于社会人口学的癌症检查短信提醒活动。我们利用大量的真实数据对膀胱癌进行了框架和模拟实例化。研究结果表明，仅基于简单的社会人口学特征优化短信提醒活动可以将死亡率的统计显著降低5.8％，与其他活动相比。

    Timely pre- and post-diagnosis check-ups are critical for cancer patients, across all cancer types, as these often lead to better outcomes. Several socio-demographic properties have been identified as strongly connected with both cancer's clinical dynamics and (indirectly) with different individual check-up behaviors. Unfortunately, existing check-up policies typically consider only the former association explicitly. In this work, we propose a novel framework, accompanied by a high-resolution computer simulation, to investigate and optimize socio-demographic-based SMS reminder campaigns for cancer check-ups. We instantiate our framework and simulation for the case of bladder cancer, the 10th most prevalent cancer today, using extensive real-world data. Our results indicate that optimizing an SMS reminder campaign based solely on simple socio-demographic features can bring about a statistically significant reduction in mortality rate compared to alternative campaigns by up to 5.8%.
    
[^9]: 混合检索中的融合函数分析

    An Analysis of Fusion Functions for Hybrid Retrieval. (arXiv:2210.11934v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2210.11934](http://arxiv.org/abs/2210.11934)

    本文研究了混合搜索的方法，将词汇和语义搜索融合在一起，在此基础上分析了利用凸组合和互惠排名融合两种方法的优缺点。我们发现凸组合比互惠排名融合更加优秀，具有样本效率，并且只需要少量训练集即可调整参数以适应目标领域。

    

    本文研究文本检索中混合搜索的方法，将词汇和语义搜索融合在一起，因为它们对于模型相关性的建模具有互补性。特别地，我们研究了通过词汇和语义评分的凸组合（CC）进行融合，以及互惠排名融合（RRF）方法，并确定它们的优点和潜在缺陷。与现有研究相反，我们发现RRF对其参数很敏感；CC融合的学习通常不关心评分规范的选择；CC在域内和域外设置中优于RRF；最后，CC具有样本效率，只需要少量训练示例即可调整其唯一参数以适应目标领域。

    We study hybrid search in text retrieval where lexical and semantic search are fused together with the intuition that the two are complementary in how they model relevance. In particular, we examine fusion by a convex combination (CC) of lexical and semantic scores, as well as the Reciprocal Rank Fusion (RRF) method, and identify their advantages and potential pitfalls. Contrary to existing studies, we find RRF to be sensitive to its parameters; that the learning of a CC fusion is generally agnostic to the choice of score normalization; that CC outperforms RRF in in-domain and out-of-domain settings; and finally, that CC is sample efficient, requiring only a small set of training examples to tune its only parameter to a target domain.
    
[^10]: LOGEN：基于逻辑知识条件的自训练文本生成在少样本下的应用

    LOGEN: Few-shot Logical Knowledge-Conditioned Text Generation with Self-training. (arXiv:2112.01404v3 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2112.01404](http://arxiv.org/abs/2112.01404)

    本文提出了一种基于少样本的逻辑知识条件下文本生成的统一框架LOGEN，通过自训练和基于内容和结构一致性抽样伪逻辑形式，实现了在少量样本下的文本生成。

    

    结构化数据的自然语言生成主要集中在表面层面描述，其存在控制内容选择困难和低保真度的问题。先前的研究利用逻辑形式来促进逻辑知识条件下的文本生成。虽然取得了显著进展，但是它们对数据的需求量较大，这使得在有限数据情况下应用于现实世界应用变得具有挑战性。为此，本文提出了一种基于少样本的逻辑知识条件下文本生成的统一框架。我们的方法只使用少量种子逻辑形式（如20/100种子） ，并利用自训练和基于内容和结构一致性抽样伪逻辑形式。实验结果表明，我们的方法可以比基准方法获得更好的少样本性能。

    Natural language generation from structured data mainly focuses on surface-level descriptions, suffering from uncontrollable content selection and low fidelity. Previous works leverage logical forms to facilitate logical knowledge-conditioned text generation. Though achieving remarkable progress, they are data-hungry, which makes the adoption for real-world applications challenging with limited data. To this end, this paper proposes a unified framework for logical knowledge-conditioned text generation in the few-shot setting. With only a few seeds logical forms (e.g., 20/100 shot), our approach leverages self-training and samples pseudo logical forms based on content and structure consistency. Experimental results demonstrate that our approach can obtain better few-shot performance than baselines.
    

