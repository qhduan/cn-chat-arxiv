# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Data Mixing Laws: Optimizing Data Mixtures by Predicting Language Modeling Performance](https://arxiv.org/abs/2403.16952) | 该研究发现了数据混合规律，可以量化地预测模型性能与数据混合比例之间的关系，并提出了一种方法来通过拟合函数形式来引导理想的数据混合选择，从而优化大型语言模型的训练混合。 |
| [^2] | [Socratic Reasoning Improves Positive Text Rewriting](https://arxiv.org/abs/2403.03029) | 使用"SocraticReframe"框架，通过引入苏格拉底式的理性化论证，增强了积极文本重写的数据集，显著提高了各种开源LLM的表现。 |
| [^3] | [CREMA: Multimodal Compositional Video Reasoning via Efficient Modular Adaptation and Fusion](https://arxiv.org/abs/2402.05889) | 该论文提出了一种名为CREMA的高效且模块化的模态融合框架，用于将任意新的模态注入视频推理。通过利用预训练模型增强多种信息模态，并引入查询转换器和融合模块，实现了灵活且有效的多模态组合推理。 |
| [^4] | [ULTRA: Unleash LLMs' Potential for Event Argument Extraction through Hierarchical Modeling and Pair-wise Refinement.](http://arxiv.org/abs/2401.13218) | ULTRA是一种层级框架，利用大型语言模型在事件论证提取中进行经济高效的处理，通过自我优化和候选论证集合的生成，解决了位置偏差问题。 |

# 详细

[^1]: 数据混合规律：通过预测语言建模性能来优化数据混合

    Data Mixing Laws: Optimizing Data Mixtures by Predicting Language Modeling Performance

    [https://arxiv.org/abs/2403.16952](https://arxiv.org/abs/2403.16952)

    该研究发现了数据混合规律，可以量化地预测模型性能与数据混合比例之间的关系，并提出了一种方法来通过拟合函数形式来引导理想的数据混合选择，从而优化大型语言模型的训练混合。

    

    大型语言模型的预训练数据包括多个领域（例如网络文本、学术论文、代码），其混合比例对结果模型的能力至关重要。现有的工作通常依赖于启发式方法或定性策略来调整比例，我们发现了模型性能与混合比例之间的函数形式的定量可预测性，我们称之为数据混合规律。在样本混合上拟合这种函数揭示了未见混合的模型性能，从而引导选择理想的数据混合。此外，我们提出了训练步骤、模型大小和我们的数据混合规律的缩放规律的嵌套使用，以使得仅通过小规模训练就能够预测在各种混合数据下训练的大模型的性能。此外，实验结果验证了我们的方法有效地优化了训练混合。

    arXiv:2403.16952v1 Announce Type: cross  Abstract: Pretraining data of large language models composes multiple domains (e.g., web texts, academic papers, codes), whose mixture proportions crucially impact the competence of outcome models. While existing endeavors rely on heuristics or qualitative strategies to tune the proportions, we discover the quantitative predictability of model performance regarding the mixture proportions in function forms, which we refer to as the data mixing laws. Fitting such functions on sample mixtures unveils model performance on unseen mixtures before actual runs, thus guiding the selection of an ideal data mixture. Furthermore, we propose nested use of the scaling laws of training steps, model sizes, and our data mixing law to enable predicting the performance of large models trained on massive data under various mixtures with only small-scale training. Moreover, experimental results verify that our method effectively optimizes the training mixture of a 
    
[^2]: 苏格拉底推理改善积极文本重写

    Socratic Reasoning Improves Positive Text Rewriting

    [https://arxiv.org/abs/2403.03029](https://arxiv.org/abs/2403.03029)

    使用"SocraticReframe"框架，通过引入苏格拉底式的理性化论证，增强了积极文本重写的数据集，显著提高了各种开源LLM的表现。

    

    将负面情绪重塑为积极思维是几种认知方法到心理健康和心理治疗的核心，大型语言模型解决方案可以使这种重塑更易实现。这种重塑通常并不简单，需要多个理性化步骤来揭示负面思维的潜在问题并使其变得更加积极。然而，目前该理性化过程被数据集和模型忽略，这些数据集和模型在一步中重塑思维。本研究填补了这一差距，通过使用一种名为"SocraticReframe"的新框架，通过合成生成的苏格拉底论证，扩充了用于积极文本重写的开源数据集。"SocraticReframe"使用一系列问答对来理性化思维重写过程。我们展示了这种苏格拉底论证显著改善了不同开源LLM的积极文本重写。

    arXiv:2403.03029v1 Announce Type: new  Abstract: Reframing a negative into a positive thought is at the crux of several cognitive approaches to mental health and psychotherapy that could be made more accessible by large language model-based solutions. Such reframing is typically non-trivial and requires multiple rationalization steps to uncover the underlying issue of a negative thought and transform it to be more positive. However, this rationalization process is currently neglected by both datasets and models which reframe thoughts in one step. In this work, we address this gap by augmenting open-source datasets for positive text rewriting with synthetically-generated Socratic rationales using a novel framework called \textsc{SocraticReframe}. \textsc{SocraticReframe} uses a sequence of question-answer pairs to rationalize the thought rewriting process. We show that such Socratic rationales significantly improve positive text rewriting for different open-source LLMs according to both
    
[^3]: CREMA: 通过有效的模块化适应和融合进行多模态组合视频推理

    CREMA: Multimodal Compositional Video Reasoning via Efficient Modular Adaptation and Fusion

    [https://arxiv.org/abs/2402.05889](https://arxiv.org/abs/2402.05889)

    该论文提出了一种名为CREMA的高效且模块化的模态融合框架，用于将任意新的模态注入视频推理。通过利用预训练模型增强多种信息模态，并引入查询转换器和融合模块，实现了灵活且有效的多模态组合推理。

    

    尽管在多模态组合推理方法方面取得了令人瞩目的进展，但由于处理固定模态输入并更新许多模型参数，仍然存在灵活性和效率方面的限制。本文解决了这些关键挑战，提出了CREMA，一种用于将任何新的模态注入视频推理的高效且模块化的模态融合框架。我们首先利用现有的预训练模型从给定的视频中增强多种信息模态（如光流、3D点云、音频），而无需额外的人工注释。接下来，我们引入了一个查询转换器，该转换器与每个可以访问的模态相关联，并具有多个参数高效的模块。它将多种模态特征投影到LLM令牌嵌入空间，使模型能够整合不同的数据类型以进行响应生成。此外，我们提出了一个融合模块，用于压缩多模态查询，在LLM中保持计算效率的同时进行融合组合。

    Despite impressive advancements in multimodal compositional reasoning approaches, they are still limited in their flexibility and efficiency by processing fixed modality inputs while updating a lot of model parameters. This paper tackles these critical challenges and proposes CREMA, an efficient and modular modality-fusion framework for injecting any new modality into video reasoning. We first augment multiple informative modalities (such as optical flow, 3D point cloud, audio) from given videos without extra human annotation by leveraging existing pre-trained models. Next, we introduce a query transformer with multiple parameter-efficient modules associated with each accessible modality. It projects diverse modality features to the LLM token embedding space, allowing the model to integrate different data types for response generation. Furthermore, we propose a fusion module designed to compress multimodal queries, maintaining computational efficiency in the LLM while combining additio
    
[^4]: ULTRA:通过层级建模和逐对优化释放LLMs在事件论证提取中的潜力

    ULTRA: Unleash LLMs' Potential for Event Argument Extraction through Hierarchical Modeling and Pair-wise Refinement. (arXiv:2401.13218v1 [cs.CL])

    [http://arxiv.org/abs/2401.13218](http://arxiv.org/abs/2401.13218)

    ULTRA是一种层级框架，利用大型语言模型在事件论证提取中进行经济高效的处理，通过自我优化和候选论证集合的生成，解决了位置偏差问题。

    

    将事件在话语中进行结构化提取是至关重要的，因为它可以更深入地理解交流模式和行为趋势。事件论证提取（EAE）是事件中心理解的核心任务，其任务是为给定事件识别特定角色的文本范围（即论证）。文档级EAE（DocEAE）侧重于散布在整个文档中的论证。在这项工作中，我们探索了开源的大型语言模型（LLMs，例如Flan-UL2）在DocEAE任务中的能力。为此，我们提出了ULTRA，一种层级框架，通过更加经济高效地提取事件论证，从而在方法中只需要少于50个注释，并且不需要访问昂贵的API端点。此外，它缓解了LLMs固有的位置偏差问题。ULTRA首先顺序阅读文档的文本块以生成候选论证集合，随后通过自我优化学习放弃非相关的候选。我们进一步介绍了...

    Structural extraction of events within discourse is critical since it avails a deeper understanding of communication patterns and behavior trends. Event argument extraction (EAE), at the core of event-centric understanding, is the task of identifying role-specific text spans (i.e., arguments) for a given event. Document-level EAE (DocEAE) focuses on arguments that are scattered across an entire document. In this work, we explore the capabilities of open source Large Language Models (LLMs), i.e., Flan-UL2, for the DocEAE task. To this end, we propose ULTRA, a hierarchical framework that extracts event arguments more cost-effectively -- the method needs as few as 50 annotations and doesn't require hitting costly API endpoints. Further, it alleviates the positional bias issue intrinsic to LLMs. ULTRA first sequentially reads text chunks of a document to generate a candidate argument set, upon which ULTRA learns to drop non-pertinent candidates through self-refinement. We further introduce
    

