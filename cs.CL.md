# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Interpretation of Intracardiac Electrograms Through Textual Representations](https://rss.arxiv.org/abs/2402.01115) | 本研究首次利用预训练的语言模型，通过文本表示的方式对心内电图进行插值和房颤分类。相比其他表示方法，我们的方法在房颤分类上表现出竞争性的性能。 |
| [^2] | [Make Large Language Model a Better Ranker](https://arxiv.org/abs/2403.19181) | 本文介绍了一种具有对齐列表排名目标的语言模型框架（ALRO），旨在弥合大型语言模型的能力与推荐系统排名任务的要求之间的差距。 |
| [^3] | [Think Twice Before Assure: Confidence Estimation for Large Language Models through Reflection on Multiple Answers](https://arxiv.org/abs/2403.09972) | 提出了一种新的评估大型语言模型置信度的方法，通过反思和提供多个候选答案的理由来解决对不正确答案的过度自信问题 |
| [^4] | [Self-Evaluation of Large Language Model based on Glass-box Features](https://arxiv.org/abs/2403.04222) | 研究探讨了大型语言模型在自我评估中利用玻璃箱特征的实用性，发现softmax分布在质量评估中可靠，提出了通过引入参考特征增强评估的策略，并验证了使用玻璃箱特征进行大型语言模型自我评估的可行性。 |
| [^5] | [Fine Tuning vs. Retrieval Augmented Generation for Less Popular Knowledge](https://arxiv.org/abs/2403.01432) | 本文研究了微调和检索增强生成两种方法对大型语言模型在处理低频实体问题回答任务中的影响，发现微调显著提高了各种受欢迎程度的实体的性能，而检索增强生成方法则超过了其他方法。 |
| [^6] | [SKT5SciSumm - A Hybrid Generative Approach for Multi-Document Scientific Summarization](https://arxiv.org/abs/2402.17311) | 提出了一种名为SKT5SciSumm的混合框架，结合了基于引文信息的变换器和T5系列模型，在多文档科学摘要任务上取得了最先进的性能。 |
| [^7] | [MATHWELL: Generating Educational Math Word Problems at Scale](https://arxiv.org/abs/2402.15861) | 使用MATHWELL模型生成了迄今为止最大的英文数学应用题数据集，其中包含20,490个问题，经领域专家评分结果显示，MATHWELL的问题中具有可执行解决方案并符合所有标准的份额比其他选择高出40%，其中74%的可解决问题同时做到了准确和适当。 |
| [^8] | [PromptKD: Distilling Student-Friendly Knowledge for Generative Language Models via Prompt Tuning](https://arxiv.org/abs/2402.12842) | 提出了PromptKD方法，通过提示调整实现了生成语言模型提取学生友好知识的蒸馏，无需微调整整个教师模型。 |
| [^9] | [HyperBERT: Mixing Hypergraph-Aware Layers with Language Models for Node Classification on Text-Attributed Hypergraphs](https://arxiv.org/abs/2402.07309) | 本文提出了HyperBERT模型，通过在预训练的BERT模型中引入超图感知层，克服了现有方法在节点分类任务上难以捕捉超图结构信息和文本属性的局限性，提高了模型的效果和泛化能力。 |
| [^10] | [SEER: Facilitating Structured Reasoning and Explanation via Reinforcement Learning.](http://arxiv.org/abs/2401.13246) | SEER是一种通过最大化基于结构的回报来促进结构化推理和解释的新方法。 |
| [^11] | [SPEER: Sentence-Level Planning of Long Clinical Summaries via Embedded Entity Retrieval.](http://arxiv.org/abs/2401.02369) | 本研究提出了一种在临床摘要中使用句子级规划并通过嵌入式实体检索的方法，以提高摘要的准确性和实用性。 |
| [^12] | [A Survey on In-context Learning.](http://arxiv.org/abs/2301.00234) | 本文调查和总结了上下文学习(ICL)的进展和挑战，ICL已成为自然语言处理(NLP)的新范式，探索ICL以评估和推广大型语言模型(LLM)的能力已成为一种新趋势。本文提出了ICL的正式定义，并总结了高级技术，最后讨论了ICL的挑战以及进一步研究的潜在方向。 |
| [^13] | [DICTDIS: Dictionary Constrained Disambiguation for Improved NMT.](http://arxiv.org/abs/2210.06996) | DICTDIS是一种新颖有词典约束的NMT系统，其利用多个字典候选项进行训练，实现了从多义词中消除翻译歧义的目的，提高了翻译质量。 |

# 详细

[^1]: 通过文本表示解读心内电图

    Interpretation of Intracardiac Electrograms Through Textual Representations

    [https://rss.arxiv.org/abs/2402.01115](https://rss.arxiv.org/abs/2402.01115)

    本研究首次利用预训练的语言模型，通过文本表示的方式对心内电图进行插值和房颤分类。相比其他表示方法，我们的方法在房颤分类上表现出竞争性的性能。

    

    理解房颤(AFib)的不规则电活动一直是心电图学中的一个重要挑战。对于严重的房颤病例，进行导管消融以获取心内电图(EGMs)。EGMs提供了心脏电活动的复杂细节和局部化信息，是可解释的心脏研究的理想模式。近年来，人工智能(AI)的进展使得一些研究可以利用深度学习框架来解释房颤中的EGMs。此外，语言模型(LMs)在能够推广到未见过的领域方面表现出了出色的性能，尤其在医疗领域。在本研究中，我们首次利用预训练的LMs来通过掩码语言建模对EGM插值和房颤分类进行微调。我们将EGM形式化为文本序列，并与其他表示方法相比，在房颤分类方面展示了竞争性的性能。最后，我们提供了全面的解释性分析。

    Understanding the irregular electrical activity of atrial fibrillation (AFib) has been a key challenge in electrocardiography. For serious cases of AFib, catheter ablations are performed to collect intracardiac electrograms (EGMs). EGMs offer intricately detailed and localized electrical activity of the heart and are an ideal modality for interpretable cardiac studies. Recent advancements in artificial intelligence (AI) has allowed some works to utilize deep learning frameworks to interpret EGMs during AFib. Additionally, language models (LMs) have shown exceptional performance in being able to generalize to unseen domains, especially in healthcare. In this study, we are the first to leverage pretrained LMs for finetuning of EGM interpolation and AFib classification via masked language modeling. We formulate the EGM as a textual sequence and present competitive performances on AFib classification compared against other representations. Lastly, we provide a comprehensive interpretabilit
    
[^2]: 让大型语言模型成为更好的排名器

    Make Large Language Model a Better Ranker

    [https://arxiv.org/abs/2403.19181](https://arxiv.org/abs/2403.19181)

    本文介绍了一种具有对齐列表排名目标的语言模型框架（ALRO），旨在弥合大型语言模型的能力与推荐系统排名任务的要求之间的差距。

    

    大型语言模型（LLMs）的发展显著增强了各个领域的能力，导致推荐系统（RSs）概念和开发方式发生了转变。然而，现有研究主要集中在点对点和成对推荐范式上。这些方法在基于LLM的推荐器中效率低下，因为利用大型语言模型的计算成本很高。一些研究虽然深入研究了列表型方法，但在排名任务中表现不佳。这一不足归因于排名和语言生成目标之间的不匹配。为此，本文介绍了具有对齐列表排名目标的语言模型框架（ALRO）。ALRO旨在弥合LLMs的能力与推荐系统排名任务的微妙要求之间的差距。ALRO的一个关键特性是引入了软lambda值lo

    arXiv:2403.19181v1 Announce Type: cross  Abstract: The evolution of Large Language Models (LLMs) has significantly enhanced capabilities across various fields, leading to a paradigm shift in how Recommender Systems (RSs) are conceptualized and developed. However, existing research primarily focuses on point-wise and pair-wise recommendation paradigms. These approaches prove inefficient in LLM-based recommenders due to the high computational cost of utilizing Large Language Models. While some studies have delved into list-wise approaches, they fall short in ranking tasks. This shortfall is attributed to the misalignment between the objectives of ranking and language generation. To this end, this paper introduces the Language Model Framework with Aligned Listwise Ranking Objectives (ALRO). ALRO is designed to bridge the gap between the capabilities of LLMs and the nuanced requirements of ranking tasks within recommender systems. A key feature of ALRO is the introduction of soft lambda lo
    
[^3]: 在承诺之前三思：通过反思多个答案评估大型语言模型的置信度

    Think Twice Before Assure: Confidence Estimation for Large Language Models through Reflection on Multiple Answers

    [https://arxiv.org/abs/2403.09972](https://arxiv.org/abs/2403.09972)

    提出了一种新的评估大型语言模型置信度的方法，通过反思和提供多个候选答案的理由来解决对不正确答案的过度自信问题

    

    置信度估计旨在评估输出的可信度，在应用大型语言模型（LLM）时至关重要，尤其是黑盒模型。由于LLM在生成不正确答案时的过度自信，现有对LLM的置信度估计通常不可校准。解决这个问题的现有方法通常受到一个显著限制的阻碍，即它们仅考虑LLM生成的一个答案的置信度。为了解决这一限制，我们提出了一种全新的范式，彻底评估多个候选答案的可信度，以减轻对不正确答案的过度自信。基于这一范式，我们引入了一个两步框架，首先指导LLM反思并为每个答案提供理由，然后汇总这些理由进行综合的置信度估计。这一框架可以与现有的置信度估计方法相结合

    arXiv:2403.09972v1 Announce Type: new  Abstract: Confidence estimation aiming to evaluate output trustability is crucial for the application of large language models (LLM), especially the black-box ones. Existing confidence estimation of LLM is typically not calibrated due to the overconfidence of LLM on its generated incorrect answers. Existing approaches addressing the overconfidence issue are hindered by a significant limitation that they merely consider the confidence of one answer generated by LLM. To tackle this limitation, we propose a novel paradigm that thoroughly evaluates the trustability of multiple candidate answers to mitigate the overconfidence on incorrect answers. Building upon this paradigm, we introduce a two-step framework, which firstly instructs LLM to reflect and provide justifications for each answer, and then aggregates the justifications for comprehensive confidence estimation. This framework can be integrated with existing confidence estimation approaches for
    
[^4]: 基于玻璃箱特征的大型语言模型的自我评估

    Self-Evaluation of Large Language Model based on Glass-box Features

    [https://arxiv.org/abs/2403.04222](https://arxiv.org/abs/2403.04222)

    研究探讨了大型语言模型在自我评估中利用玻璃箱特征的实用性，发现softmax分布在质量评估中可靠，提出了通过引入参考特征增强评估的策略，并验证了使用玻璃箱特征进行大型语言模型自我评估的可行性。

    

    arXiv:2403.04222v1 公告类型：新摘要：开源大型语言模型（LLMs）的蓬勃发展凸显了对评估方法的迫切需求。现有作品主要依赖于外部评估者，侧重于训练和提示策略。然而，一个关键方面——模型感知的玻璃箱特征——被忽视了。在这项研究中，我们探讨了在自我评估情境下使用玻璃箱特征的效用，即应用LLM评估其自身输出。我们研究了各种玻璃箱特征组，并发现softmax分布作为质量评估的可靠指标。此外，我们提出了通过合并从参考文献中提取的特征来增强评估的两种策略。在公共基准测试上的实验结果验证了使用玻璃箱特征进行LLMs的自我评估的可行性。

    arXiv:2403.04222v1 Announce Type: new  Abstract: The proliferation of open-source Large Language Models (LLMs) underscores the pressing need for evaluation methods. Existing works primarily rely on external evaluators, focusing on training and prompting strategies. However, a crucial aspect - model-aware glass-box features - is overlooked. In this study, we explore the utility of glass-box features under the scenario of self-evaluation, namely applying an LLM to evaluate its own output. We investigate various glass-box feature groups and discovered that the softmax distribution serves as a reliable indicator for quality evaluation. Furthermore, we propose two strategies to enhance the evaluation by incorporating features derived from references. Experimental results on public benchmarks validate the feasibility of self-evaluation of LLMs using glass-box features.
    
[^5]: 微调与检索增强生成用于不太流行知识的比较

    Fine Tuning vs. Retrieval Augmented Generation for Less Popular Knowledge

    [https://arxiv.org/abs/2403.01432](https://arxiv.org/abs/2403.01432)

    本文研究了微调和检索增强生成两种方法对大型语言模型在处理低频实体问题回答任务中的影响，发现微调显著提高了各种受欢迎程度的实体的性能，而检索增强生成方法则超过了其他方法。

    

    大型语言模型（LLMs）记忆了大量的事实知识，在各种任务和领域表现出色。然而，观察到当处理不太流行或低频概念和实体时，性能会下降，例如在领域特定应用中。本文探讨和评估了检索增强生成（RAG）和通过合成数据进行微调（FT）对定制LLMs处理低频实体问题回答任务的影响。研究结果表明，FT显著提升了各种受欢迎程度的实体的性能，特别是在最受欢迎和最不受欢迎的群体中，而RAG超越了其他方法。另外，检索和数据增强技术的进步加强了RAG和FT方法的成功。

    arXiv:2403.01432v1 Announce Type: new  Abstract: Large language models (LLMs) memorize a vast amount of factual knowledge, exhibiting strong performance across diverse tasks and domains. However, it has been observed that the performance diminishes when dealing with less-popular or low-frequency concepts and entities, for example in domain specific applications. The two prominent approaches to enhance the performance of LLMs on low-frequent topics are: Retrieval Augmented Generation (RAG) and fine-tuning (FT) over synthetic data. This paper explores and evaluates the impact of RAG and FT on customizing LLMs in handling low-frequency entities on question answering task. Our findings indicate that FT significantly boosts the performance across entities of varying popularity, especially in the most and least popular groups, while RAG surpasses other methods. Additionally, the success of both RAG and FT approaches is amplified by advancements in retrieval and data augmentation techniques. 
    
[^6]: SKT5SciSumm - 一种用于多文档科学摘要的混合生成方法

    SKT5SciSumm - A Hybrid Generative Approach for Multi-Document Scientific Summarization

    [https://arxiv.org/abs/2402.17311](https://arxiv.org/abs/2402.17311)

    提出了一种名为SKT5SciSumm的混合框架，结合了基于引文信息的变换器和T5系列模型，在多文档科学摘要任务上取得了最先进的性能。

    

    arXiv:2402.17311v1 公告类型：新 摘要：科学文本摘要对于研究界和人类社会都显示出明显的益处。考虑到科学文本的特殊性以及多文档摘要任务的输入实质上很长，该任务需要足够的嵌入生成和文本截断，同时又不能丢失重要信息。为了解决这些问题，本文提出了SKT5SciSumm - 一种用于多文档科学摘要的混合框架（MDSS）。我们利用基于引文信息的变换器(SPECTER)的科学文献嵌入的句子-变换器版本来编码和表示文本句子，从而实现使用k-means聚类进行高效摘要提取。我们使用T5系列模型使用提取的句子生成抽象摘要。SKT5SciSumm在Multi-XScience数据集上实现了最先进的性能。通过大量实验和

    arXiv:2402.17311v1 Announce Type: new  Abstract: Summarization for scientific text has shown significant benefits both for the research community and human society. Given the fact that the nature of scientific text is distinctive and the input of the multi-document summarization task is substantially long, the task requires sufficient embedding generation and text truncation without losing important information. To tackle these issues, in this paper, we propose SKT5SciSumm - a hybrid framework for multi-document scientific summarization (MDSS). We leverage the Sentence-Transformer version of Scientific Paper Embeddings using Citation-Informed Transformers (SPECTER) to encode and represent textual sentences, allowing for efficient extractive summarization using k-means clustering. We employ the T5 family of models to generate abstractive summaries using extracted sentences. SKT5SciSumm achieves state-of-the-art performance on the Multi-XScience dataset. Through extensive experiments and
    
[^7]: MATHWELL: 在规模上生成教育数学应用题

    MATHWELL: Generating Educational Math Word Problems at Scale

    [https://arxiv.org/abs/2402.15861](https://arxiv.org/abs/2402.15861)

    使用MATHWELL模型生成了迄今为止最大的英文数学应用题数据集，其中包含20,490个问题，经领域专家评分结果显示，MATHWELL的问题中具有可执行解决方案并符合所有标准的份额比其他选择高出40%，其中74%的可解决问题同时做到了准确和适当。

    

    数学应用题在K-8教育中至关重要，但编写它们耗时且需要领域专业知识。我们认为语言模型可以通过自动生成规模化问题来支持K-8数学教育。为了教育性，生成的问题必须是1）可解决的，2）准确的，3）适当的。现有数据集未标记这些标准，因此不适合训练问题生成器。我们引入了MATHWELL，这是一个经过专家注释数据进行迭代微调的70B Llama-2模型，用于生成K-8数学应用题。借助MATHWELL，我们生成了迄今为止最大的英文应用题数据集，其中包含20,490个问题。经领域专家评分的3,484个问题发现，MATHWELL拥有比其他选择更高的可执行解决方案和满足所有标准的问题份额高出40％，其中74％的问题具有可解的、准确的和适当的解决方案。

    arXiv:2402.15861v1 Announce Type: new  Abstract: Math word problems are critical K-8 educational tools, but writing them is time-consuming and requires domain expertise. We suggest that language models can support K-8 math education by automatically generating problems at scale. To be educational, generated problems must be 1) solvable, 2) accurate, and 3) appropriate. Existing datasets are unlabeled for these criteria, making them ill-suited for training problem generators. We introduce MATHWELL, a Llama-2 (70B) model iteratively finetuned to generate K-8 math word problems using data from expert annotation. Using MATHWELL, we generate the largest English word problem dataset to date, containing 20,490 problems. 3,484 are scored by domain experts who find MATHWELL has a 40% higher share of problems that have executable solutions and meet all criteria than alternatives, with 74% of its problems with executable solutions being solvable, accurate, and appropriate.
    
[^8]: PromptKD：通过提示调整为生成语言模型提取学生友好知识的蒸馏方法

    PromptKD: Distilling Student-Friendly Knowledge for Generative Language Models via Prompt Tuning

    [https://arxiv.org/abs/2402.12842](https://arxiv.org/abs/2402.12842)

    提出了PromptKD方法，通过提示调整实现了生成语言模型提取学生友好知识的蒸馏，无需微调整整个教师模型。

    

    近期大型语言模型（LLMs）的发展引起了对推理成本的担忧，进一步增加了对模型压缩研究的需求。尽管知识蒸馏（KD）是一种突出的方法，但是针对LLMs这样的生成语言模型的KD研究相对较少，而提取适合学生的知识的方法，在分类模型的KD中表现出了良好性能，在生成语言模型中尚未被探索。为了探索这种方法，我们提出了PromptKD，一种简单而有效的方法，它利用提示调整 - 在KD中首次出现 - 使生成语言模型能够传递适合学生的知识。与先前分类工作不同，先前那些需要微调整整个教师模型以提取适合学生的知识，PromptKD通过添加少量提示标记，并仅通过学生指导调整提示来达到类似效果。

    arXiv:2402.12842v1 Announce Type: cross  Abstract: Recent advancements in large language models (LLMs) have raised concerns about inference costs, increasing the need for research into model compression. While knowledge distillation (KD) is a prominent method for this, research on KD for generative language models like LLMs is relatively sparse, and the approach of distilling student-friendly knowledge, which has shown promising performance in KD for classification models, remains unexplored in generative language models. To explore this approach, we propose PromptKD, a simple yet effective method that utilizes prompt tuning - for the first time in KD - to enable generative language models to transfer student-friendly knowledge. Unlike previous works in classification that require fine-tuning the entire teacher model for extracting student-friendly knowledge, PromptKD achieves similar effects by adding a small number of prompt tokens and tuning only the prompt with student guidance. Ex
    
[^9]: HyperBERT:将混合超图感知层与语言模型用于文本属性超图上的节点分类

    HyperBERT: Mixing Hypergraph-Aware Layers with Language Models for Node Classification on Text-Attributed Hypergraphs

    [https://arxiv.org/abs/2402.07309](https://arxiv.org/abs/2402.07309)

    本文提出了HyperBERT模型，通过在预训练的BERT模型中引入超图感知层，克服了现有方法在节点分类任务上难以捕捉超图结构信息和文本属性的局限性，提高了模型的效果和泛化能力。

    

    超图通过复杂的拓扑结构标记，表达多个实体之间的高阶相互作用，其中超边扮演重要角色。最近，基于超图的深度学习方法在学习文本属性超图上的节点分类问题中引起了越来越多的研究关注。然而，现有方法往往难以同时捕捉超图结构信息的全部内容和节点属性中的丰富语言属性，这在很大程度上影响了它们的效果和泛化能力。为了克服这些挑战，我们探索了如何通过为节点分类任务进一步增强预训练的BERT模型，引入专门的超图感知层。这些层将高阶结构归纳偏差引入语言模型中，从而提高模型利用超图结构中的高阶上下文信息和文本中的语义信息的能力。

    Hypergraphs are marked by complex topology, expressing higher-order interactions among multiple entities with hyperedges. Lately, hypergraph-based deep learning methods to learn informative data representations for the problem of node classification on text-attributed hypergraphs have garnered increasing research attention. However, existing methods struggle to simultaneously capture the full extent of hypergraph structural information and the rich linguistic attributes inherent in the nodes attributes, which largely hampers their effectiveness and generalizability. To overcome these challenges, we explore ways to further augment a pretrained BERT model with specialized hypergraph-aware layers for the task of node classification. Such layers introduce higher-order structural inductive bias into the language model, thus improving the model's capacity to harness both higher-order context information from the hypergraph structure and semantic information present in text. In this paper, we
    
[^10]: SEER: 通过强化学习促进结构化推理和解释

    SEER: Facilitating Structured Reasoning and Explanation via Reinforcement Learning. (arXiv:2401.13246v1 [cs.CL])

    [http://arxiv.org/abs/2401.13246](http://arxiv.org/abs/2401.13246)

    SEER是一种通过最大化基于结构的回报来促进结构化推理和解释的新方法。

    

    阐明从问题到答案的推理过程，通过结构化解释是根本重要的，因为它显著增强了问答系统的解释性和可信度。然而，结构化解释要求模型进行复杂的结构化推理，这带来了巨大的挑战。大多数现有方法集中在通过监督学习进行单步推理，忽视步骤之间的逻辑依赖关系。同时，现有的基于强化学习（RL）的方法忽视了结构化关系，阻碍了RL在结构化推理中的潜力。在本文中，我们提出了一种名为SEER的新方法，通过最大化基于结构的回报，以促进结构化推理和解释。我们提出的基于结构的回报准确描述了结构化推理中固有的分层和分支结构，有效地捕捉了状态之间的复杂关系。我们还引入了一种细粒度的奖励函数。

    Elucidating the reasoning process with structured explanations from question to answer is fundamentally crucial, as it significantly enhances the interpretability and trustworthiness of question-answering (QA) systems. However, structured explanations demand models to perform intricate structured reasoning, which poses great challenges. Most existing methods focus on single-step reasoning through supervised learning, ignoring logical dependencies between steps. Meanwhile, existing reinforcement learning (RL)-based methods overlook the structured relationships, impeding RL's potential in structured reasoning. In this paper, we propose SEER, a novel method that maximizes a structure-based return to facilitate structured reasoning and explanation. Our proposed structure-based return precisely describes the hierarchical and branching structure inherent in structured reasoning, effectively capturing the intricate relationships between states. We also introduce a fine-grained reward function
    
[^11]: SPEER: Embedded Entity Retrieval下的长临床摘要句子级规划

    SPEER: Sentence-Level Planning of Long Clinical Summaries via Embedded Entity Retrieval. (arXiv:2401.02369v1 [cs.CL])

    [http://arxiv.org/abs/2401.02369](http://arxiv.org/abs/2401.02369)

    本研究提出了一种在临床摘要中使用句子级规划并通过嵌入式实体检索的方法，以提高摘要的准确性和实用性。

    

    临床医生在每次病人出院时必须写一份冗长的摘要。由于涵盖的临床概念数量庞大，这项任务非常耗时。识别和涵盖显著实体对于摘要的临床实用性至关重要。我们在该任务上微调了开源的LLM模型（Mistral-7B-Instruct和Zephyr-7B-η），发现它们生成的摘要不完整且不准确。为了增加实体覆盖范围，我们训练了一个较小的仅编码器模型来预测显著实体，并将其作为内容计划来指导LLM。为了鼓励LLM关注源笔记中的特定提及，我们提出了SPEER：Embedded Entity Retrieval下的句子级规划。具体而言，我们使用特殊的"{{ }}"边界标签标记每个显著实体跨度，并要求LLM在生成每个句子之前检索标记的跨度。句子级规划相当于一种状态追踪，模型明确记录下每个句子的信息。

    Clinician must write a lengthy summary each time a patient is discharged from the hospital. This task is time-consuming due to the sheer number of unique clinical concepts covered in the admission. Identifying and covering salient entities is vital for the summary to be clinically useful. We fine-tune open-source LLMs (Mistral-7B-Instruct and Zephyr-7B-\b{eta}) on the task and find that they generate incomplete and unfaithful summaries. To increase entity coverage, we train a smaller, encoder-only model to predict salient entities, which are treated as content-plans to guide the LLM. To encourage the LLM to focus on specific mentions in the source notes, we propose SPEER: Sentence-level Planning via Embedded Entity Retrieval. Specifically, we mark each salient entity span with special "{{ }}" boundary tags and instruct the LLM to retrieve marked spans before generating each sentence. Sentence-level planning acts as a form of state tracking in that the model is explicitly recording the 
    
[^12]: 关于上下文学习的综述

    A Survey on In-context Learning. (arXiv:2301.00234v3 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2301.00234](http://arxiv.org/abs/2301.00234)

    本文调查和总结了上下文学习(ICL)的进展和挑战，ICL已成为自然语言处理(NLP)的新范式，探索ICL以评估和推广大型语言模型(LLM)的能力已成为一种新趋势。本文提出了ICL的正式定义，并总结了高级技术，最后讨论了ICL的挑战以及进一步研究的潜在方向。

    

    随着大型语言模型（LLM）的能力不断增强，上下文学习（ICL）已成为自然语言处理（NLP）的新范式，在其中LLM仅基于加入少量示例的上下文进行预测。探索ICL以评估和推广LLM的能力已成为一种新趋势。本文旨在调查和总结ICL的进展和挑战。我们首先提出ICL的正式定义，并澄清其与相关研究的关系。然后，我们组织和讨论高级技术，包括训练策略、演示设计策略以及相关分析。最后，我们讨论了ICL的挑战，并提供了进一步研究的潜在方向。我们希望我们的工作可以鼓励更多的研究，揭示ICL的工作原理并改进ICL。

    With the increasing ability of large language models (LLMs), in-context learning (ICL) has become a new paradigm for natural language processing (NLP), where LLMs make predictions only based on contexts augmented with a few examples. It has been a new trend to explore ICL to evaluate and extrapolate the ability of LLMs. In this paper, we aim to survey and summarize the progress and challenges of ICL. We first present a formal definition of ICL and clarify its correlation to related studies. Then, we organize and discuss advanced techniques, including training strategies, demonstration designing strategies, as well as related analysis. Finally, we discuss the challenges of ICL and provide potential directions for further research. We hope that our work can encourage more research on uncovering how ICL works and improving ICL.
    
[^13]: DICTDIS：基于词典约束的神经机器翻译消歧方法对 NMT 的改进

    DICTDIS: Dictionary Constrained Disambiguation for Improved NMT. (arXiv:2210.06996v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2210.06996](http://arxiv.org/abs/2210.06996)

    DICTDIS是一种新颖有词典约束的NMT系统，其利用多个字典候选项进行训练，实现了从多义词中消除翻译歧义的目的，提高了翻译质量。

    

    领域特定的神经机器翻译系统（例如教育应用程序）在多语言社会中帮助使信息对一组多样化的用户可访问是具有社会意义的。这种 NMT 系统应该具有词汇约束并从领域特定的词典中汲取。由于单词的多义性，词典中可能会为源单词或短语呈现多个候选翻译。这时，NMT 模型需要选择与语境最相关的候选翻译。先前的工作主要忽略了这个问题，而侧重于单个候选约束设置，其中目标词或短语被单个约束替换。在本文中，我们提出了一种名为DICTDIS的词典约束 NMT 系统，该系统消除了从字典中得出的多个候选翻译的歧义。我们通过将训练数据与多个字典候选项进行增量来实现这一点，从而在训练期间积极鼓励消除歧义。

    Domain-specific neural machine translation (NMT) systems (\eg, in educational applications) are socially significant with the potential to help make information accessible to a diverse set of users in multilingual societies. It is desirable that such NMT systems be lexically constrained and draw from domain-specific dictionaries. Dictionaries could present multiple candidate translations for a source word/phrase due to the polysemous nature of words. The onus is then on the NMT model to choose the contextually most appropriate candidate. Prior work has largely ignored this problem and focused on the single candidate constraint setting wherein the target word or phrase is replaced by a single constraint. In this work we present \dictdis, a lexically constrained NMT system that disambiguates between multiple candidate translations derived from dictionaries. We achieve this by augmenting training data with multiple dictionary candidates to actively encourage disambiguation during training
    

