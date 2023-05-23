# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Paragraph-level Citation Recommendation based on Topic Sentences as Queries.](http://arxiv.org/abs/2305.12190) | 该论文提出了一种中间层级的引用建议任务——段落级引用建议，即以段落的主题句为输入，输出在段落中引用的建议，并提出了用于解决此任务的模型。 |
| [^2] | [Unified Embedding: Battle-Tested Feature Representations for Web-Scale ML Systems.](http://arxiv.org/abs/2305.12102) | 本文介绍了一种名为“特征复用”的框架，它使用单一的表示空间 能够高效有效地学习高质量的特征嵌入，同时区分不同的分类特征。通过在多个公共数据集和新数据集“Web-Available Image Search (WAIS)”上的测试，我们展示了这种方法的优于现有技术的表现。 |
| [^3] | [UP5: Unbiased Foundation Model for Fairness-aware Recommendation.](http://arxiv.org/abs/2305.12090) | 本研究提出了一种新颖的基础模型UP5，它采用反事实公平促进技术来消除大型语言模型中的偏见，从而实现面向公平性的推荐。 |
| [^4] | [DADIN: Domain Adversarial Deep Interest Network for Cross Domain Recommender Systems.](http://arxiv.org/abs/2305.12058) | 论文提出了一种创新性的深度跨领域点击率预测模型——领域对抗深度兴趣网络（DADIN），该模型通过引入领域不可知层和特别设计的损失，创新地实现了两个领域的联合分布对齐，并采用对抗训练的方式与点击率预测损失一起进行优化，相比竞争基线算法提升明显。 |
| [^5] | [Exploring the Viability of Synthetic Query Generation for Relevance Prediction.](http://arxiv.org/abs/2305.11944) | 本文研究在电子商务和医疗保健等专业领域中，利用强大的模型生成高质量特定任务和领域的合成数据，探索用于预测对文档的查询分级相关性的方法，并尝试使用无监督聚类技术进一步改进对数据中相关性模式的理解。 |
| [^6] | [Knowledge Refinement via Interaction Between Search Engines and Large Language Models.](http://arxiv.org/abs/2305.07402) | 本文介绍了一种新的框架InteR，通过搜索引擎和大型语言模型之间的交互促进知识精炼，从而提高检索准确性。 |
| [^7] | [UDAPDR: Unsupervised Domain Adaptation via LLM Prompting and Distillation of Rerankers.](http://arxiv.org/abs/2303.00807) | 该论文提出了一种无监督领域自适应方法，利用大型语言模型(LLMs)生成大量合成查询和reranker模型，蒸馏为高效的检索器，适用于长尾领域。 |
| [^8] | [Improving Sequential Recommendation Models with an Enhanced Loss Function.](http://arxiv.org/abs/2301.00979) | 本文研究了顺序推荐模型常用损失函数的优劣，提出了一种改进的损失函数。实验表明，这种改进的损失函数可以显著提升多种顺序推荐模型的性能。 |
| [^9] | [Towards Adversarially Robust Recommendation from Adaptive Fraudster Detection.](http://arxiv.org/abs/2211.11534) | 本文提出了一种针对推荐系统的MetaC恶意攻击，并设计了一种自适应欺诈者检测模块PDR，明确考虑标签的不确定性，提高了推荐系统的鲁棒性。 |
| [^10] | [Leveraging End-to-End Speech Recognition with Neural Architecture Search.](http://arxiv.org/abs/1912.05946) | 本文研究表明，通过神经架构搜索可以在非常低的计算成本情况下显著提高深度语音模型的准确性，取得了与最先进结果相当的水平。 |
| [^11] | [NLPExplorer: Exploring the Universe of NLP Papers.](http://arxiv.org/abs/1910.07351) | NLPExplorer是一个自动化门户网站，用于索引、搜索和可视化NLP研究文献，手动策划五类主题类别。提供了年轻热门作者、热门URL和数据集列表、不同主题的论文列表，以及最近热门的论文等。 |
| [^12] | [Network Capacity Bound for Personalized PageRank in Multimodal Networks.](http://arxiv.org/abs/1706.00178) | 本文推广双分图PageRank的想法，提出了一种用于多模网络的超图类型，证明了多模网络中个性化PageRank的网络容量界限。 |

# 详细

[^1]: 基于主题句作为查询的段落级引用建议

    Paragraph-level Citation Recommendation based on Topic Sentences as Queries. (arXiv:2305.12190v1 [cs.IR])

    [http://arxiv.org/abs/2305.12190](http://arxiv.org/abs/2305.12190)

    该论文提出了一种中间层级的引用建议任务——段落级引用建议，即以段落的主题句为输入，输出在段落中引用的建议，并提出了用于解决此任务的模型。

    

    引用建议(CR)模型可帮助作者在论文写作过程中的各个阶段找到相关文章。大多数研究处理全局CR，该全局CR适用于初始写作阶段的一般建议。本研究提出了段落级CR任务，作为两种方法之间的一种中间地带，其中段落的主题句作为输入，生成段落内引用的建议作为输出。我们提出了一个模型来解决这个任务，并使用ACL论文数据集上的四元组损失进行了微调，结果显示相对于基线有所改善。

    Citation recommendation (CR) models may help authors find relevant articles at various stages of the paper writing process. Most research has dealt with either global CR, which produces general recommendations suitable for the initial writing stage, or local CR, which produces specific recommendations more fitting for the final writing stages. We propose the task of paragraph-level CR as a middle ground between the two approaches, where the paragraph's topic sentence is taken as input and recommendations for citing within the paragraph are produced at the output. We propose a model for this task, fine-tune it using the quadruplet loss on the dataset of ACL papers, and show improvements over the baselines.
    
[^2]: 统一嵌入：面向 Web 规模 ML 系统的经过验证的特征表示

    Unified Embedding: Battle-Tested Feature Representations for Web-Scale ML Systems. (arXiv:2305.12102v1 [cs.LG])

    [http://arxiv.org/abs/2305.12102](http://arxiv.org/abs/2305.12102)

    本文介绍了一种名为“特征复用”的框架，它使用单一的表示空间 能够高效有效地学习高质量的特征嵌入，同时区分不同的分类特征。通过在多个公共数据集和新数据集“Web-Available Image Search (WAIS)”上的测试，我们展示了这种方法的优于现有技术的表现。

    

    高效、有效地学习高质量的特征嵌入对于 Web 规模的机器学习系统的性能至关重要。标准方法是将每个特征值表示为一个 d 维嵌入，引入数百亿个参数，而这些特征的基数非常高。这个瓶颈导致了备选嵌入算法的重大进展。本文介绍了一个简单但非常有效的框架，即“特征复用”，在许多不同的分类特征之间使用一个单一的表示空间。我们的理论和实证分析表明，复用的嵌入可以分解为每个组成特征的组件，使得模型可以区分特征。我们展示了复用的嵌入在几个公共数据集上优于现有技术。此外，我们引入了一个名为“Web-Available Image Search (WAIS)”的新数据集，以严格评估 Web 规模下的新嵌入算法。我们邀请社区通过提出可以准确、高效地将数百万张图像嵌入和分类到成千上万个类别的新模型来贡献 WAIS 挑战。

    Learning high-quality feature embeddings efficiently and effectively is critical for the performance of web-scale machine learning systems. A typical model ingests hundreds of features with vocabularies on the order of millions to billions of tokens. The standard approach is to represent each feature value as a d-dimensional embedding, introducing hundreds of billions of parameters for extremely high-cardinality features. This bottleneck has led to substantial progress in alternative embedding algorithms. Many of these methods, however, make the assumption that each feature uses an independent embedding table. This work introduces a simple yet highly effective framework, Feature Multiplexing, where one single representation space is used across many different categorical features. Our theoretical and empirical analysis reveals that multiplexed embeddings can be decomposed into components from each constituent feature, allowing models to distinguish between features. We show that multip
    
[^3]: UP5: 面向公平性推荐的无偏基础模型

    UP5: Unbiased Foundation Model for Fairness-aware Recommendation. (arXiv:2305.12090v1 [cs.IR])

    [http://arxiv.org/abs/2305.12090](http://arxiv.org/abs/2305.12090)

    本研究提出了一种新颖的基础模型UP5，它采用反事实公平促进技术来消除大型语言模型中的偏见，从而实现面向公平性的推荐。

    

    基于大型语言模型（LLM）等基础模型的最新进展，已将它们推到了推荐系统（RS）的前沿。此外，RS中的公平性很关键，因为许多用户将其用于决策和需求履行。然而，目前尚缺乏对推荐基础模型展示公平性水平和公平处理不同用户群组的适当方法的理解。本文侧重于用户方面的不公平问题，并通过彻底检查表明，LLMs中存在不公平性，导致不公平的推荐结果。为了消除LLM中的偏差以实现面向公平性的推荐，我们引入了一种基于反事实公平促进技术的新型无偏P5（UP5）基础模型。CFP包括两个子模块：个性化前缀提示和Prompt混合，从而增强了个体敏感属性的公平性。

    Recent advancements in foundation models such as large language models (LLM) have propelled them to the forefront of recommender systems (RS). Moreover, fairness in RS is critical since many users apply it for decision-making and demand fulfillment. However, at present, there is a lack of understanding regarding the level of fairness exhibited by recommendation foundation models and the appropriate methods for equitably treating different groups of users in foundation models. In this paper, we focus on user-side unfairness problem and show through a thorough examination that there is unfairness involved in LLMs that lead to unfair recommendation results. To eliminate bias from LLM for fairness-aware recommendation, we introduce a novel Unbiased P5 (UP5) foundation model based on Counterfactually-Fair-Prompting (CFP) techniques. CFP includes two sub-modules: a personalized prefix prompt that enhances fairness with respect to individual sensitive attributes, and a Prompt Mixture that int
    
[^4]: DADIN: 面向跨域推荐系统的领域对抗深度兴趣网络

    DADIN: Domain Adversarial Deep Interest Network for Cross Domain Recommender Systems. (arXiv:2305.12058v1 [cs.IR])

    [http://arxiv.org/abs/2305.12058](http://arxiv.org/abs/2305.12058)

    论文提出了一种创新性的深度跨领域点击率预测模型——领域对抗深度兴趣网络（DADIN），该模型通过引入领域不可知层和特别设计的损失，创新地实现了两个领域的联合分布对齐，并采用对抗训练的方式与点击率预测损失一起进行优化，相比竞争基线算法提升明显。

    

    点击率预测是推荐系统的主要任务之一，用户针对不同项目进行点击以获取推荐结果。针对数据稀疏性、用户-项目交互的长尾分布和项目或用户的冷启动等问题，提出了跨领域点击率预测模型。为了使源域到目标域的知识转移更加顺畅，提出了创新性的深度跨领域点击率预测模型——领域对抗深度兴趣网络 (DADIN)，将跨域推荐任务转化为领域适应问题。通过引入领域不可知层和特别设计的损失，创新地实现了两个领域的联合分布对齐，并采用对抗训练的方式与点击率预测损失一起进行优化。实验结果表明，在华为数据集上，DADIN 的曲线下面积 (AUC) 比最具竞争力的基线高出0.08％，高出0.7％。

    Click-Through Rate (CTR) prediction is one of the main tasks of the recommendation system, which is conducted by a user for different items to give the recommendation results. Cross-domain CTR prediction models have been proposed to overcome problems of data sparsity, long tail distribution of user-item interactions, and cold start of items or users. In order to make knowledge transfer from source domain to target domain more smoothly, an innovative deep learning cross-domain CTR prediction model, Domain Adversarial Deep Interest Network (DADIN) is proposed to convert the cross-domain recommendation task into a domain adaptation problem. The joint distribution alignment of two domains is innovatively realized by introducing domain agnostic layers and specially designed loss, and optimized together with CTR prediction loss in a way of adversarial training. It is found that the Area Under Curve (AUC) of DADIN is 0.08% higher than the most competitive baseline on Huawei dataset and is 0.7
    
[^5]: 探索用于相关性预测的合成查询生成的可行性

    Exploring the Viability of Synthetic Query Generation for Relevance Prediction. (arXiv:2305.11944v1 [cs.IR])

    [http://arxiv.org/abs/2305.11944](http://arxiv.org/abs/2305.11944)

    本文研究在电子商务和医疗保健等专业领域中，利用强大的模型生成高质量特定任务和领域的合成数据，探索用于预测对文档的查询分级相关性的方法，并尝试使用无监督聚类技术进一步改进对数据中相关性模式的理解。

    

    查询-文档相关性预测是信息检索系统中的一个关键问题。这个问题越来越多地使用（预先训练的）基于转换器的模型来解决，这些模型使用大量标记数据进行微调。然而，在电子商务和医疗保健等专业领域，这种方法的可行性受到领域内大规模数据的匮乏限制。为了解决这个问题，最近的方法利用这些强大的模型生成高质量的特定任务和领域的合成数据。先前的工作主要探索了合成数据生成或用于问答和二元（是/否）相关性预测的查询生成（QGen）, 其中例如，QGen模型给出一个文档，并训练生成一个与该文档相关的查询。然而，在许多问题中，我们对相关性有一个更细粒度的概念，而不是一个简单的是/否标签。因此，在这项工作中，我们进行了详细的研究，探讨了如何利用QGen方法实现细微的相关性预测。具体而言，我们研究了使用合成查询来预测对文档的查询分级相关性的有效性，并探索使用无监督聚类技术进一步改进对数据中相关性模式的理解。

    Query-document relevance prediction is a critical problem in Information Retrieval systems. This problem has increasingly been tackled using (pretrained) transformer-based models which are finetuned using large collections of labeled data. However, in specialized domains such as e-commerce and healthcare, the viability of this approach is limited by the dearth of large in-domain data. To address this paucity, recent methods leverage these powerful models to generate high-quality task and domain-specific synthetic data. Prior work has largely explored synthetic data generation or query generation (QGen) for Question-Answering (QA) and binary (yes/no) relevance prediction, where for instance, the QGen models are given a document, and trained to generate a query relevant to that document. However in many problems, we have a more fine-grained notion of relevance than a simple yes/no label. Thus, in this work, we conduct a detailed study into how QGen approaches can be leveraged for nuanced
    
[^6]: 搜索引擎与大型语言模型间的交互优化知识精炼

    Knowledge Refinement via Interaction Between Search Engines and Large Language Models. (arXiv:2305.07402v1 [cs.CL])

    [http://arxiv.org/abs/2305.07402](http://arxiv.org/abs/2305.07402)

    本文介绍了一种新的框架InteR，通过搜索引擎和大型语言模型之间的交互促进知识精炼，从而提高检索准确性。

    

    信息检索在从大量数据中定位相关资源方面具有重要作用，其应用已从传统知识库发展至现代搜索引擎（SEs）。大型语言模型（LLMs）的出现进一步通过使用自然语言与搜索系统交互革命性地改变了该领域。本文探索了LLMs和SEs的优缺点，强调它们在理解用户查询和检索最新信息方面的各自优势。为了利用两种范例的优势并避免其限制，我们提出了InteR，这是一个通过SEs和LLMs之间的交互促进知识精炼的新框架。 InteR使SEs能够使用LLM生成的摘要来调整查询，同时使LLMs能够使用SE检索到的文档来增强提示。这种迭代的精炼过程增强了SEs和LLMs的输入，从而导致更准确的检索结果。

    Information retrieval (IR) plays a crucial role in locating relevant resources from vast amounts of data, and its applications have evolved from traditional knowledge bases to modern search engines (SEs). The emergence of large language models (LLMs) has further revolutionized the field by enabling users to interact with search systems in natural language. In this paper, we explore the advantages and disadvantages of LLMs and SEs, highlighting their respective strengths in understanding user-issued queries and retrieving up-to-date information. To leverage the benefits of both paradigms while circumventing their limitations, we propose InteR, a novel framework that facilitates knowledge refinement through interaction between SEs and LLMs. InteR allows SEs to refine knowledge in query using LLM-generated summaries and enables LLMs to enhance prompts using SE-retrieved documents. This iterative refinement process augments the inputs of SEs and LLMs, leading to more accurate retrieval. Ex
    
[^7]: UDAPDR: 基于LLM提示与reranker蒸馏的无监督领域自适应

    UDAPDR: Unsupervised Domain Adaptation via LLM Prompting and Distillation of Rerankers. (arXiv:2303.00807v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2303.00807](http://arxiv.org/abs/2303.00807)

    该论文提出了一种无监督领域自适应方法，利用大型语言模型(LLMs)生成大量合成查询和reranker模型，蒸馏为高效的检索器，适用于长尾领域。

    

    很多信息检索任务需要大型标注数据集进行微调，但这样的数据集通常不可用，且在应用于真实场景中时可能会因为领域漂移而迅速失去效用。为了解决这个问题，我们提出一种使用大型语言模型(LLMs)廉价生成大量合成查询的方法。该方法首先利用昂贵的LLM生成少量合成查询，然后再利用成本较低的LLM生成大量的合成查询以微调一组reranker模型。最后，这些reranker会被蒸 distill 成一个高效的检索器，用于目标领域中的检索。实验证明，这种技术可以提高长尾领域中的零样本准确性，即使只使用2K个合成查询进行微调，并且比标准的reranking方法具有更低的延迟。我们提供完整的端到端方案，包括合成数据集等。

    Many information retrieval tasks require large labeled datasets for fine-tuning. However, such datasets are often unavailable, and their utility for real-world applications can diminish quickly due to domain shifts. To address this challenge, we develop and motivate a method for using large language models (LLMs) to generate large numbers of synthetic queries cheaply. The method begins by generating a small number of synthetic queries using an expensive LLM. After that, a much less expensive one is used to create large numbers of synthetic queries, which are used to fine-tune a family of reranker models. These rerankers are then distilled into a single efficient retriever for use in the target domain. We show that this technique boosts zero-shot accuracy in long-tail domains, even where only 2K synthetic queries are used for fine-tuning, and that it achieves substantially lower latency than standard reranking methods. We make our end-to-end approach, including our synthetic datasets an
    
[^8]: 一种改进的损失函数提升顺序推荐模型

    Improving Sequential Recommendation Models with an Enhanced Loss Function. (arXiv:2301.00979v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2301.00979](http://arxiv.org/abs/2301.00979)

    本文研究了顺序推荐模型常用损失函数的优劣，提出了一种改进的损失函数。实验表明，这种改进的损失函数可以显著提升多种顺序推荐模型的性能。

    

    最近，人们对于顺序推荐模型进行了大量的基准测试和复现/改进现有模型的工作。本文通过分析常用的顺序推荐损失函数的优劣，提出了一种改进的损失函数来充分利用它们的优点。实验结果表明，这种改进的损失函数显著提升了 GRU4Rec，SASRec，SR-GNN和 S3Rec等模型的性能。

    There has been a growing interest in benchmarking sequential recommendation models and reproducing/improving existing models. For example, Rendle et al. improved matrix factorization models by tuning their parameters and hyperparameters. Petrov and Macdonald developed a more efficient and effective implementation of BERT4Rec, which resolved inconsistencies in performance comparison between BERT4Rec and SASRec in previous works. In particular, BERT4Rec and SASRec share a similar network structure, with the main difference lying in their training objective/loss function. Therefore, we analyzed the advantages and disadvantages of commonly used loss functions in sequential recommendation and proposed an improved loss function that leverages their strengths. We conduct extensive experiments on two influential open-source libraries, and the results demonstrate that our improved loss function significantly enhances the performance of GRU4Rec, SASRec, SR-GNN, and S3Rec models, improving their 
    
[^9]: 从自适应欺诈者检测探究对抗鲁棒的推荐系统

    Towards Adversarially Robust Recommendation from Adaptive Fraudster Detection. (arXiv:2211.11534v3 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2211.11534](http://arxiv.org/abs/2211.11534)

    本文提出了一种针对推荐系统的MetaC恶意攻击，并设计了一种自适应欺诈者检测模块PDR，明确考虑标签的不确定性，提高了推荐系统的鲁棒性。

    

    推荐系统在节点注入攻击下的鲁棒性备受关注。最近，提出了基于GNN的推荐系统GraphRfi，它有效减轻了注入的虚假用户的影响。但是，我们展示了GraphRfi仍然容易受到攻击，因为其欺诈者检测组件的监督性质，在实践中很难获得干净的标签。我们提出了一个强大的MetaC恶意攻击，针对GNN-based和MF-based推荐系统。根据我们从易受攻击性分析中得到的见解，我们设计了一种自适应欺诈者检测模块，明确考虑了标签不确定性。该模块可以作为不同推荐系统的插件，形成一个稳健的框架（PDR）。全面的实验表明，我们的防御方法在攻击下优于其他基准方法。总体而言，我们的工作强调了在构建欺诈者检测模块时考虑标签不确定性的重要性，并提供了改善推荐系统对节点注入攻击鲁棒性的实用解决方案。

    The robustness of recommender systems under node injection attacks has garnered significant attention. Recently, GraphRfi, a GNN-based recommender system, was proposed and shown to effectively mitigate the impact of injected fake users. However, we demonstrate that GraphRfi remains vulnerable to attacks due to the supervised nature of its fraudster detection component, where obtaining clean labels is challenging in practice. In particular, we propose a powerful poisoning attack, MetaC, against both GNN-based and MF-based recommender systems. Furthermore, we analyze why GraphRfi fails under such an attack. Then, based on our insights obtained from vulnerability analysis, we design an adaptive fraudster detection module that explicitly considers label uncertainty. This module can serve as a plug-in for different recommender systems, resulting in a robust framework named PDR. Comprehensive experiments show that our defense approach outperforms other benchmark methods under attacks. Overal
    
[^10]: 利用神经架构搜索提升端到端语音识别效果

    Leveraging End-to-End Speech Recognition with Neural Architecture Search. (arXiv:1912.05946v2 [eess.AS] UPDATED)

    [http://arxiv.org/abs/1912.05946](http://arxiv.org/abs/1912.05946)

    本文研究表明，通过神经架构搜索可以在非常低的计算成本情况下显著提高深度语音模型的准确性，取得了与最先进结果相当的水平。

    

    深度神经网络已经被证实在自动语音识别方面优于许多传统机器学习算法。本文研究表明，通过有效实施神经架构搜索可以在非常低的计算成本情况下显著提高深度语音模型的准确性。在使用流行的LibriSpeech和TIMIT基准测试中进行的音素识别测试证明了这一事实，该方法能够在几个小时之内（不到一天），比基于注意力机制的seq2seq模型快多次，探测和训练新的候选模型。我们的方法在LibriSpeech语料库上的测试误差率（WER）为7％，在TIMIT语料库上的音素误差率（PER）为13％，达到了与最先进结果相当的水平。

    Deep neural networks (DNNs) have been demonstrated to outperform many traditional machine learning algorithms in Automatic Speech Recognition (ASR). In this paper, we show that a large improvement in the accuracy of deep speech models can be achieved with effective Neural Architecture Optimization at a very low computational cost. Phone recognition tests with the popular LibriSpeech and TIMIT benchmarks proved this fact by displaying the ability to discover and train novel candidate models within a few hours (less than a day) many times faster than the attention-based seq2seq models. Our method achieves test error of 7% Word Error Rate (WER) on the LibriSpeech corpus and 13% Phone Error Rate (PER) on the TIMIT corpus, on par with state-of-the-art results.
    
[^11]: NLPExplorer：探索自然语言处理论文的宇宙

    NLPExplorer: Exploring the Universe of NLP Papers. (arXiv:1910.07351v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/1910.07351](http://arxiv.org/abs/1910.07351)

    NLPExplorer是一个自动化门户网站，用于索引、搜索和可视化NLP研究文献，手动策划五类主题类别。提供了年轻热门作者、热门URL和数据集列表、不同主题的论文列表，以及最近热门的论文等。

    

    随着科学文章数量的不断增加，了解当前研究趋势、问题及其创新解决方案仍然是一个瓶颈。本文提出了NLPExplorer，这是一个完全自动化的门户网站，用于索引、搜索和可视化自然语言处理（NLP）研究体量。与之前基于主题建模的方法不同，NLPExplorer手动策划了五个粗略的、非排他的主题类别，即语言目标（语法、语篇等）、任务（标注、摘要等）、方法（无监督、监督等）、语言（英语、中文等）和数据集类型（新闻、临床笔记等）。其中一些新颖的功能包括年轻热门作者、热门URL和数据集列表、不同主题的论文列表以及最近热门的论文。此外，它还提供了诸如按年度热度的主题、数据集的统计信息等。

    Understanding the current research trends, problems, and their innovative solutions remains a bottleneck due to the ever-increasing volume of scientific articles. In this paper, we propose NLPExplorer, a completely automatic portal for indexing, searching, and visualizing Natural Language Processing (NLP) research volume. NLPExplorer presents interesting insights from papers, authors, venues, and topics. In contrast to previous topic modelling based approaches, we manually curate five course-grained non-exclusive topical categories namely Linguistic Target (Syntax, Discourse, etc.), Tasks (Tagging, Summarization, etc.), Approaches (unsupervised, supervised, etc.), Languages (English, Chinese,etc.) and Dataset types (news, clinical notes, etc.). Some of the novel features include a list of young popular authors, popular URLs, and datasets, a list of topically diverse papers and recent popular papers. Also, it provides temporal statistics such as yearwise popularity of topics, datasets, 
    
[^12]: 多模网络中个性化PageRank的网络容量界限

    Network Capacity Bound for Personalized PageRank in Multimodal Networks. (arXiv:1706.00178v3 [cs.SI] UPDATED)

    [http://arxiv.org/abs/1706.00178](http://arxiv.org/abs/1706.00178)

    本文推广双分图PageRank的想法，提出了一种用于多模网络的超图类型，证明了多模网络中个性化PageRank的网络容量界限。

    

    在一篇先前的论文中，介绍了双分图PageRank的概念，并推广了个性化PageRank中节点之间授权流限制的定理。本文将这些结果推广到多模网络中。我们特别处理了一种用于描述多模网络的超图类型，其中超链接将每个模态的节点连接起来。我们引入了这种图的PageRank概率分布，并定义了相应的随机游走模型，可用于计算。我们对具有相同和不同阻尼因子的情况下授权流出量的极限情况进行了陈述和证明。

    In a former paper the concept of Bipartite PageRank was introduced and a theorem on the limit of authority flowing between nodes for personalized PageRank has been generalized. In this paper we want to extend those results to multimodal networks. In particular we deal with a hypergraph type that may be used for describing multimodal network where a hyperlink connects nodes from each of the modalities. We introduce a generalisation of PageRank for such graphs and define the respective random walk model that can be used for computations. We state and prove theorems on the limit of outflow of authority for cases where individual modalities have identical and distinct damping factors.
    

