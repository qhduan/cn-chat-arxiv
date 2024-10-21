# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Multi-Conditional Ranking with Large Language Models](https://arxiv.org/abs/2404.00211) | 该论文提出了一种新颖的分解推理方法(MCRank)，用于解决大型语言模型在多条件排序任务中性能下降的问题。 |
| [^2] | [Train & Constrain: Phonologically Informed Tongue-Twister Generation from Topics and Paraphrases](https://arxiv.org/abs/2403.13901) | 本文提出了一种从主题和释义生成基于音韵学的绕口令的新方法，生成了迄今为止最大的绕口令数据集TwistList 2.0，并进行了自动和人工评估。 |
| [^3] | [The Missing Piece in Model Editing: A Deep Dive into the Hidden Damage Brought By Model Editing](https://arxiv.org/abs/2403.07825) | 本文提出了一种新的评估方法 GORA 和一种模型编辑方法 SORA，用以解决模型编辑中的隐藏空间中的涟漪效应问题。 |
| [^4] | ['One size doesn't fit all': Learning how many Examples to use for In-Context Learning for Improved Text Classification](https://arxiv.org/abs/2403.06402) | 本文提出了自适应上下文学习（AICL）的工作流程，通过动态调整示例数量来提高文本分类的性能，类似于k最近邻（k-NN）中的可变大小邻域。 |
| [^5] | [MaiBaam Annotation Guidelines](https://arxiv.org/abs/2403.05902) | 该论文提供了MaiBaam语料库的注释准则，详细介绍了如何处理和标记巴伐利亚数据，说明了词性标记和依赖关系的使用，以及对德语等相关语言适用的注释决策和对巴伐利亚语法特定决策的介绍和推动。 |
| [^6] | [WaterMax: breaking the LLM watermark detectability-robustness-quality trade-off](https://arxiv.org/abs/2403.04808) | WaterMax提出了一种新的水印方案，能够在保持生成文本质量的同时实现高检测性能，打破了水印技术中质量和稳健性之间的传统平衡。 |
| [^7] | [SciAssess: Benchmarking LLM Proficiency in Scientific Literature Analysis](https://arxiv.org/abs/2403.01976) | SciAssess介绍了一个专为深度分析科学文献而设计的基准测试，旨在全面评估LLMs在科学领域记忆、理解和分析能力的有效性。 |
| [^8] | [Framing in the Presence of Supporting Data: A Case Study in U.S. Economic News](https://arxiv.org/abs/2402.14224) | 本文提出了一个计算框架，旨在分析主流媒体在报道经济消息时的编辑选择，通过对经济指标的报道进行框架分析，我们可以理解出版物选择和构架的方式。 |
| [^9] | [A Comprehensive Study of Multilingual Confidence Estimation on Large Language Models](https://arxiv.org/abs/2402.13606) | 该论文介绍了对大型语言模型的多语言置信度评估的全面研究，提出了一个专业多语言问答数据集，并研究了这些置信度分数如何增强模型性能，最终提出了一种跨语言置信度估计方法。 |
| [^10] | ["We Demand Justice!": Towards Social Context Grounding of Political Texts](https://arxiv.org/abs/2311.09106) | 该论文提出了定义计算环境中理解政治文本中模棱两可陈述所需背景的框架，并提出了挑战性的数据集，以此来分析和预测文本的真实世界背景。 |
| [^11] | [Towards Verifiable Text Generation with Evolving Memory and Self-Reflection.](http://arxiv.org/abs/2312.09075) | 本研究提出了一种名为VTG的创新框架，用于实现具有进化记忆和自我反思的可验证文本生成。通过引入进化型长短期记忆和两层验证器，VTG解决了大型语言模型在生成过程中出现的信息错误和准确性问题。 |
| [^12] | [Entity Matching using Large Language Models.](http://arxiv.org/abs/2310.11244) | 这项研究探讨了使用大型语言模型（LLMs）作为实体匹配的替代方法，相较于预训练的语言模型（PLMs），LLMs对训练数据需求较少且更具鲁棒性。 |
| [^13] | [Zero-shot Query Reformulation for Conversational Search.](http://arxiv.org/abs/2307.09384) | 提出了一种零样本查询重构（ZeQR）框架，通过利用机器阅读理解任务的语言模型来解决对话搜索中的数据稀疏性、解释性不足和歧义的问题。 |
| [^14] | [WinoQueer: A Community-in-the-Loop Benchmark for Anti-LGBTQ+ Bias in Large Language Models.](http://arxiv.org/abs/2306.15087) | WinoQueer是一个社区协同基准，旨在衡量大型语言模型是否存在对LGBTQ+社区有害的偏见。研究发现现成模型普遍存在相当大的反同偏见，通过在该社区撰写或由该社区成员撰写的数据上进行微调，可以在一定程度上减轻偏见。 |

# 详细

[^1]: 大型语言模型下的多条件排序

    Multi-Conditional Ranking with Large Language Models

    [https://arxiv.org/abs/2404.00211](https://arxiv.org/abs/2404.00211)

    该论文提出了一种新颖的分解推理方法(MCRank)，用于解决大型语言模型在多条件排序任务中性能下降的问题。

    

    利用大型语言模型(LLMs)对一组项目进行排序已成为推荐和检索系统中的常见方法。在这篇论文中，我们定义并探讨了多条件排序的任务，引入了一个名为MCRank的基准，旨在评估跨不同项目类型和条件进行多条件排序。我们使用MCRank对LLMs进行分析表明，随着项目和条件数量以及复杂性的增长，性能显著下降。为了克服这一限制，我们提出了一种新颖的分解推理方法，包括提取和排序条件，然后迭代地对条件进行排序。

    arXiv:2404.00211v1 Announce Type: new  Abstract: Utilizing large language models (LLMs) to rank a set of items has become a common approach in recommendation and retrieval systems. Typically, these systems focus on ordering a substantial number of documents in a monotonic order based on a given query. However, real-world scenarios often present a different challenge: ranking a comparatively smaller set of items, but according to a variety of diverse and occasionally conflicting conditions. In this paper, we define and explore the task of multi-conditional ranking by introducing MCRank, a benchmark tailored for assessing multi-conditional ranking across various item types and conditions. Our analysis of LLMs using MCRank indicates a significant decrease in performance as the number and complexity of items and conditions grow. To overcome this limitation, we propose a novel decomposed reasoning method, consisting of EXtracting and Sorting the conditions, and then Iterativly Ranking the i
    
[^2]: 训练与限制：从主题和释义生成基于音韵学的绕口令

    Train & Constrain: Phonologically Informed Tongue-Twister Generation from Topics and Paraphrases

    [https://arxiv.org/abs/2403.13901](https://arxiv.org/abs/2403.13901)

    本文提出了一种从主题和释义生成基于音韵学的绕口令的新方法，生成了迄今为止最大的绕口令数据集TwistList 2.0，并进行了自动和人工评估。

    

    过去在音韵和语音基础的语言生成方面的工作主要集中在领域，如双关语和诗歌。在本文中，我们提出了产生绕口令的新工作-这种语言形式需要在音素级别上进行条件约束，以最大程度地实现声音重叠，同时与输入主题保持语义一致，仍然保持语法正确。我们提出了TwisterLister，这是一个从大型语言模型（LLMs）中生成基于音韵学的绕口令的流程，我们用它来生成TwistList 2.0，到目前为止最大的一个已标记数据集，包含来自人类和LLM作者合作的超过17K个例子。我们的生成流程涉及使用音韵受限词汇以及LLM提示来生成新颖的、非衍生的绕口令实例。此外，我们还提出了对较小规模的自动和人工评估结果。

    arXiv:2403.13901v1 Announce Type: new  Abstract: Previous work in phonologically and phonetically grounded language generation has mainly focused on domains such as puns and poetry. In this article, we present new work on the generation of tongue-twisters - a form of language that is required to be conditioned on a phoneme level to maximize sound overlap, whilst maintaining semantic consistency with an input topic and still being grammatically correct. We present TwisterLister, a pipeline for generating phonologically informed tongue-twisters from Large Language Models (LLMs) that we use to generate TwistList 2.0, the largest annotated dataset of tongue-twisters to date, consisting of 17K+ examples from a combination of human and LLM authors. Our generation pipeline involves the use of a phonologically constrained vocabulary alongside LLM prompting to generate novel, non-derivative tongue-twister examples. We additionally present the results of automatic and human evaluation of smaller
    
[^3]: 模型编辑中的遗漏之处：深入探讨模型编辑带来的隐藏损害

    The Missing Piece in Model Editing: A Deep Dive into the Hidden Damage Brought By Model Editing

    [https://arxiv.org/abs/2403.07825](https://arxiv.org/abs/2403.07825)

    本文提出了一种新的评估方法 GORA 和一种模型编辑方法 SORA，用以解决模型编辑中的隐藏空间中的涟漪效应问题。

    

    大型语言模型以其卓越的效果彻底改变了许多任务。然而，对这些模型进行编辑，以修改过时或错误信息的关键性工作，往往会导致一个称为“隐藏空间中的涟漪效应”的复杂问题。这种效应虽然难以检测，但却会显著阻碍模型编辑任务的效果，并恶化模型性能。本文通过提出一种新颖的评估方法，基于图形特异值关系的评估(GORA)，来应对这一科学挑战，量化评估模型的适应性和编辑的后续影响。此外，我们引入了一种旨在减轻这种涟漪效应的模型编辑方法——选择性异常值重新编辑方法(SORA)。我们的全面评估揭示了隐藏空间中的涟漪效应在所有当前模型编辑方法中都是一个重要问题。然而，我们提出的方法，G

    arXiv:2403.07825v1 Announce Type: new  Abstract: Large Language Models have revolutionized numerous tasks with their remarkable efficacy.However, the editing of these models, crucial for rectifying outdated or erroneous information, often leads to a complex issue known as the ripple effect in the hidden space. This effect, while difficult to detect, can significantly impede the efficacy of model editing tasks and deteriorate model performance.This paper addresses this scientific challenge by proposing a novel evaluation methodology, Graphical Outlier Relation based Assessment(GORA), which quantitatively evaluates the adaptations of the model and the subsequent impact of editing. Furthermore, we introduce the Selective Outlier Re-Editing Approach(SORA), a model editing method designed to mitigate this ripple effect. Our comprehensive evaluations reveal that the ripple effect in the hidden space is a significant issue in all current model editing methods. However, our proposed methods, G
    
[^4]: 一刀切不适用：学习在文本分类中使用多少例为了改进上下文学习

    'One size doesn't fit all': Learning how many Examples to use for In-Context Learning for Improved Text Classification

    [https://arxiv.org/abs/2403.06402](https://arxiv.org/abs/2403.06402)

    本文提出了自适应上下文学习（AICL）的工作流程，通过动态调整示例数量来提高文本分类的性能，类似于k最近邻（k-NN）中的可变大小邻域。

    

    arXiv:2403.06402v1 发表类型：新 Abstract: 自然语言处理（NLP）中的预测模型已经从从头训练模型发展到使用标记数据微调预训练模型。这种微调的极端形式涉及到上下文学习（ICL），其中一个预先训练的生成模型的输出（冻结的解码器参数）只受到输入字符串的变化（称为指令或提示）的控制。ICL的一个重要组成部分是在提示中使用少量标记数据实例作为示例。尽管现有工作在推理过程中为每个数据实例使用静态数量的示例，但在本文中，我们提出了一种动态调整示例数量的新方法。这类似于k最近邻（k-NN）分类器中使用可变大小邻域的方法。在我们提出的自适应ICL（AICL）的工作流程中，对于特定数据实例进行推理时使用的演示数量是动态调整的。

    arXiv:2403.06402v1 Announce Type: new  Abstract: Predictive models in natural language processing (NLP) have evolved from training models from scratch to fine-tuning pre-trained models with labelled data. An extreme form of this fine-tuning involves in-context learning (ICL), where the output of a pre-trained generative model (frozen decoder parameters) is controlled only with variations in the input strings (called instructions or prompts). An important component of ICL is the use of a small number of labelled data instances as examples in the prompt. While existing work uses a static number of examples during inference for each data instance, in this paper we propose a novel methodology of dynamically adapting the number of examples as per the data. This is analogous to the use of a variable-sized neighborhood in k-nearest neighbors (k-NN) classifier. In our proposed workflow of adaptive ICL (AICL), the number of demonstrations to employ during the inference on a particular data inst
    
[^5]: MaiBaam注释准则

    MaiBaam Annotation Guidelines

    [https://arxiv.org/abs/2403.05902](https://arxiv.org/abs/2403.05902)

    该论文提供了MaiBaam语料库的注释准则，详细介绍了如何处理和标记巴伐利亚数据，说明了词性标记和依赖关系的使用，以及对德语等相关语言适用的注释决策和对巴伐利亚语法特定决策的介绍和推动。

    

    本文提供了MaiBaam的注释准则，这是一个注释了词性标记和句法依赖关系的巴伐利亚语语料库。MaiBaam属于通用依存关系项目（UD），我们的注释详细说明了一般和德国UD第2版指南。在本文中，我们详细介绍了如何预处理和标记巴伐利亚数据，概述了我们使用的词性标记和依赖关系，解释了也适用于德语等密切相关语言的注释决策，最后介绍并推动了适用于巴伐利亚语法的决策。

    arXiv:2403.05902v1 Announce Type: new  Abstract: This document provides the annotation guidelines for MaiBaam, a Bavarian corpus annotated with part-of-speech (POS) tags and syntactic dependencies. MaiBaam belongs to the Universal Dependencies (UD) project, and our annotations elaborate on the general and German UD version 2 guidelines. In this document, we detail how to preprocess and tokenize Bavarian data, provide an overview of the POS tags and dependencies we use, explain annotation decisions that would also apply to closely related languages like German, and lastly we introduce and motivate decisions that are specific to Bavarian grammar.
    
[^6]: WaterMax: 打破LLM水印可检测性-稳健性-质量的平衡

    WaterMax: breaking the LLM watermark detectability-robustness-quality trade-off

    [https://arxiv.org/abs/2403.04808](https://arxiv.org/abs/2403.04808)

    WaterMax提出了一种新的水印方案，能够在保持生成文本质量的同时实现高检测性能，打破了水印技术中质量和稳健性之间的传统平衡。

    

    水印是阻止大型语言模型被恶意使用的技术手段。本文提出了一种称为WaterMax的新颖水印方案，具有高检测性能，同时保持原始LLM生成文本的质量。其新设计不会对LLM进行任何修改（不调整权重、对数、温度或采样技术）。WaterMax平衡了稳健性和复杂性，与文献中的水印技术相反，从根本上引发了质量和稳健性之间的平衡。其性能在理论上得到证明并经过实验证实。在最全面的基准测试套件下，它胜过所有的最先进技术。

    arXiv:2403.04808v1 Announce Type: cross  Abstract: Watermarking is a technical means to dissuade malfeasant usage of Large Language Models. This paper proposes a novel watermarking scheme, so-called WaterMax, that enjoys high detectability while sustaining the quality of the generated text of the original LLM. Its new design leaves the LLM untouched (no modification of the weights, logits, temperature, or sampling technique). WaterMax balances robustness and complexity contrary to the watermarking techniques of the literature inherently provoking a trade-off between quality and robustness. Its performance is both theoretically proven and experimentally validated. It outperforms all the SotA techniques under the most complete benchmark suite.
    
[^7]: SciAssess：基准测试LLM在科学文献分析中的熟练程度

    SciAssess: Benchmarking LLM Proficiency in Scientific Literature Analysis

    [https://arxiv.org/abs/2403.01976](https://arxiv.org/abs/2403.01976)

    SciAssess介绍了一个专为深度分析科学文献而设计的基准测试，旨在全面评估LLMs在科学领域记忆、理解和分析能力的有效性。

    

    arXiv:2403.01976v1 公告类型：新 抽象：大型语言模型（LLMs）的最新突破已经彻底改变了自然语言理解和生成，引发了人们对利用这些技术进行细致科学文献分析的兴趣激增。然而，现有的基准测试未能充分评估LLMs在科学领域的熟练程度，特别是在涉及复杂理解和多模态数据的情况下。为此，我们引入了SciAssess，一个专为深度分析科学文献而设计的基准测试，旨在全面评估LLMs的有效性。SciAssess专注于评估LLMs在科学背景下记忆、理解和分析的能力。它包括来自不同科学领域的代表性任务，如一般化学、有机材料和合金材料。严格的质量控制措施确保了其在正确性、匿名化和复制方面的可靠性。

    arXiv:2403.01976v1 Announce Type: new  Abstract: Recent breakthroughs in Large Language Models (LLMs) have revolutionized natural language understanding and generation, igniting a surge of interest in leveraging these technologies for the nuanced field of scientific literature analysis. Existing benchmarks, however, inadequately evaluate the proficiency of LLMs in the scientific domain, especially in scenarios involving complex comprehension and multimodal data. In response, we introduced SciAssess, a benchmark tailored for the in-depth analysis of scientific literature, crafted to provide a thorough assessment of LLMs' efficacy. SciAssess focuses on evaluating LLMs' abilities in memorization, comprehension, and analysis within scientific contexts. It includes representative tasks from diverse scientific fields, such as general chemistry, organic materials, and alloy materials. And rigorous quality control measures ensure its reliability in terms of correctness, anonymization, and copy
    
[^8]: 在支持数据存在的情况下进行框架构建：以美国经济新闻为例

    Framing in the Presence of Supporting Data: A Case Study in U.S. Economic News

    [https://arxiv.org/abs/2402.14224](https://arxiv.org/abs/2402.14224)

    本文提出了一个计算框架，旨在分析主流媒体在报道经济消息时的编辑选择，通过对经济指标的报道进行框架分析，我们可以理解出版物选择和构架的方式。

    

    主流媒体在选择何事物进行报道以及如何进行报道方面有很大的自由裁量权。这些选择会对人们所了解的信息和随后的行为产生真实世界的影响。然而，缺乏客观的评估编辑选择的度量使得这一领域的研究特别困难。本文认为在一些有支持数据存在的值得报道的话题中，可以提出一个计算框架来分析编辑选择。我们选择经济作为研究重点，因为经济指标的报道为我们提供了一个相对容易确定各种出版物选择和构架的方式。这些指标为我们提供了一个有关经济表现的真实情况，相对于出版物对其进行报道的方式。为了实现这一目标，我们将框架预测定义为一组相互依赖的任务。

    arXiv:2402.14224v1 Announce Type: new  Abstract: The mainstream media has much leeway in what it chooses to cover and how it covers it. These choices have real-world consequences on what people know and their subsequent behaviors. However, the lack of objective measures to evaluate editorial choices makes research in this area particularly difficult. In this paper, we argue that there are newsworthy topics where objective measures exist in the form of supporting data and propose a computational framework to analyze editorial choices in this setup. We focus on the economy because the reporting of economic indicators presents us with a relatively easy way to determine both the selection and framing of various publications. Their values provide a ground truth of how the economy is doing relative to how the publications choose to cover it. To do this, we define frame prediction as a set of interdependent tasks. At the article level, we learn to identify the reported stance towards the gene
    
[^9]: 对大型语言模型的多语言置信度评估进行全面研究

    A Comprehensive Study of Multilingual Confidence Estimation on Large Language Models

    [https://arxiv.org/abs/2402.13606](https://arxiv.org/abs/2402.13606)

    该论文介绍了对大型语言模型的多语言置信度评估的全面研究，提出了一个专业多语言问答数据集，并研究了这些置信度分数如何增强模型性能，最终提出了一种跨语言置信度估计方法。

    

    大型语言模型生成幻觉并在预测中表现过于自信的倾向引发了人们对其可靠性的担忧。表明模型响应的可信度或不确定性估计对于开发可靠的人工智能系统至关重要。目前的研究主要集中在英语中LLM的置信度估计上，在其他广泛使用的语言方面仍存在空白，阻碍了可靠AI应用的全球发展。本文介绍了对LLM上的多语言置信度评估（MlingConf）的全面调查。首先，我们引入了一个经过详细检查的专业多语言问答数据集。其次，我们深入研究置信度估计的性能，并研究这些置信度分数如何通过跨不同语言的自我完善来增强LLM的性能。最后，我们提出了一种跨语言置信度估计方法，以实现更精确的估计。

    arXiv:2402.13606v1 Announce Type: new  Abstract: The tendency of Large Language Models to generate hallucinations and exhibit overconfidence in predictions raises concerns regarding their reliability. Confidence or uncertainty estimations indicating the extent of trustworthiness of a model's response are essential to developing reliable AI systems. Current research primarily focuses on LLM confidence estimations in English, remaining a void for other widely used languages and impeding the global development of reliable AI applications. This paper introduces a comprehensive investigation of Multi-lingual confidence estimation (MlingConf) on LLMs. First, we introduce an elaborated and expert-checked multilingual QA dataset. Second, we delve into the performance of confidence estimations and examine how these confidence scores can enhance LLM performance through self-refinement across diverse languages. Finally, we propose a cross-lingual confidence estimation method to achieve more preci
    
[^10]: “我们要求公正！”：政治文本的社会背景奠基

    "We Demand Justice!": Towards Social Context Grounding of Political Texts

    [https://arxiv.org/abs/2311.09106](https://arxiv.org/abs/2311.09106)

    该论文提出了定义计算环境中理解政治文本中模棱两可陈述所需背景的框架，并提出了挑战性的数据集，以此来分析和预测文本的真实世界背景。

    

    社交媒体话语经常包括“看似相似的语言被政治光谱两端使用”，往往转化为截然不同的观点。例如，“思念与祈祷”可能表达对大规模枪击受害者的同情，也可能批评对该问题缺乏立法行动。本文在计算环境中定义了完全理解此类模棱两可陈述所需的背景，并使其基于现实世界的实体、行动和态度。我们提出了两个需要理解文本实际背景的具有挑战性的数据集。我们将这些数据集与基于大型预训练模型（如RoBERTa和GPT-3）构建的模型进行了基准测试。此外，我们开发并对现有的“议语情境化框架”和“政治角色表征”模型进行了基准测试。我们分析了数据集和预测以获得更多洞见。

    arXiv:2311.09106v2 Announce Type: replace  Abstract: Social media discourse frequently consists of 'seemingly similar language used by opposing sides of the political spectrum', often translating to starkly contrasting perspectives. E.g., 'thoughts and prayers', could express sympathy for mass-shooting victims, or criticize the lack of legislative action on the issue. This paper defines the context required to fully understand such ambiguous statements in a computational setting and ground them in real-world entities, actions, and attitudes. We propose two challenging datasets that require an understanding of the real-world context of the text. We benchmark these datasets against models built upon large pre-trained models, such as RoBERTa and GPT-3. Additionally, we develop and benchmark more structured models building upon existing Discourse Contextualization Framework and Political Actor Representation models. We analyze the datasets and the predictions to obtain further insights int
    
[^11]: 实现具有进化记忆和自我反思的可验证文本生成

    Towards Verifiable Text Generation with Evolving Memory and Self-Reflection. (arXiv:2312.09075v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2312.09075](http://arxiv.org/abs/2312.09075)

    本研究提出了一种名为VTG的创新框架，用于实现具有进化记忆和自我反思的可验证文本生成。通过引入进化型长短期记忆和两层验证器，VTG解决了大型语言模型在生成过程中出现的信息错误和准确性问题。

    

    尽管大型语言模型（LLMs）在语言理解和生成方面具有出色能力，但它们往往会产生错误的信息，也被称为幻觉。解决这个问题的一个有希望的方法是可验证的文本生成，它促使LLMs生成具有引用以进行准确性验证的内容。然而，可验证的文本生成并不简单，因为存在焦点转移现象，需要复杂的推理来与正确的引文对齐，而且在检索文档的精确性和广度之间存在着两难。在本文中，我们提出了一种创新的具有进化记忆和自我反思的可验证文本生成框架VTG。VTG引入了进化型长短期记忆以保留有价值的文档和最近的文档。我们提出了一个配备证据发现器的两层验证器，用于重新思考和反思主张与引文之间的关系。此外，还采用主动检索和多样化的方式来提高论证的质量和广度。

    Despite the remarkable ability of large language models (LLMs) in language comprehension and generation, they often suffer from producing factually incorrect information, also known as hallucination. A promising solution to this issue is verifiable text generation, which prompts LLMs to generate content with citations for accuracy verification. However, verifiable text generation is non-trivial due to the focus-shifting phenomenon, the intricate reasoning needed to align the claim with correct citations, and the dilemma between the precision and breadth of retrieved documents. In this paper, we present VTG, an innovative framework for Verifiable Text Generation with evolving memory and self-reflection. VTG introduces evolving long short-term memory to retain both valuable documents and recent documents. A two-tier verifier equipped with an evidence finder is proposed to rethink and reflect on the relationship between the claim and citations. Furthermore, active retrieval and diverse qu
    
[^12]: 使用大型语言模型进行实体匹配

    Entity Matching using Large Language Models. (arXiv:2310.11244v1 [cs.CL])

    [http://arxiv.org/abs/2310.11244](http://arxiv.org/abs/2310.11244)

    这项研究探讨了使用大型语言模型（LLMs）作为实体匹配的替代方法，相较于预训练的语言模型（PLMs），LLMs对训练数据需求较少且更具鲁棒性。

    

    实体匹配是判断两个实体描述是否指的是同一个真实世界实体的任务。实体匹配是大多数数据集成流程中的核心步骤，也是许多电子商务应用的重要组成部分，这些应用需要将来自不同供应商的产品匹配起来。目前最先进的实体匹配方法通常依赖于预训练的语言模型（PLMs），如BERT或RoBERTa。然而，这些模型在实体匹配中存在两个主要缺点：（i）模型需要大量特定任务的训练数据；（ii）微调后的模型对于超出分布范围的实体不够健壮。本文研究了使用大型语言模型（LLMs）作为基于PLMs的匹配器的备选方案，相比之下，LLMs对领域特定训练数据需求较少且更具鲁棒性。我们的研究涵盖了托管的LLMs，如GPT3.5和GPT4，以及基于Llama2的开源LLMs，可以在本地运行。我们在零样本场景和…

    Entity Matching is the task of deciding whether two entity descriptions refer to the same real-world entity. Entity Matching is a central step in most data integration pipelines and an enabler for many e-commerce applications which require to match products offers from different vendors. State-of-the-art entity matching methods often rely on pre-trained language models (PLMs) such as BERT or RoBERTa. Two major drawbacks of these models for entity matching are that (i) the models require significant amounts of task-specific training data and (ii) the fine-tuned models are not robust concerning out-of-distribution entities. In this paper, we investigate using large language models (LLMs) for entity matching as a less domain-specific training data reliant and more robust alternative to PLM-based matchers. Our study covers hosted LLMs, such as GPT3.5 and GPT4, as well as open source LLMs based on Llama2 which can be run locally. We evaluate these models in a zero-shot scenario as well as a
    
[^13]: 零样本对话搜索中的查询重构

    Zero-shot Query Reformulation for Conversational Search. (arXiv:2307.09384v1 [cs.IR])

    [http://arxiv.org/abs/2307.09384](http://arxiv.org/abs/2307.09384)

    提出了一种零样本查询重构（ZeQR）框架，通过利用机器阅读理解任务的语言模型来解决对话搜索中的数据稀疏性、解释性不足和歧义的问题。

    

    随着语音助手的普及，对话搜索在信息检索领域引起了更多的关注。然而，对话搜索中的数据稀疏性问题严重阻碍了监督式对话搜索方法的进展。因此，研究人员更加关注零样本对话搜索方法。然而，现有的零样本方法存在三个主要限制：它们不适用于所有的检索器，它们的有效性缺乏足够的解释性，并且他们无法解决因省略而导致的常见对话歧义。为了解决这些限制，我们引入了一种新颖的零样本查询重构（ZeQR）框架，该框架根据先前的对话上下文重构查询，而无需对话搜索数据的监督。具体来说，我们的框架利用了设计用于机器阅读理解任务的语言模型来明确解决两个常见的歧义：协调和省略。

    As the popularity of voice assistants continues to surge, conversational search has gained increased attention in Information Retrieval. However, data sparsity issues in conversational search significantly hinder the progress of supervised conversational search methods. Consequently, researchers are focusing more on zero-shot conversational search approaches. Nevertheless, existing zero-shot methods face three primary limitations: they are not universally applicable to all retrievers, their effectiveness lacks sufficient explainability, and they struggle to resolve common conversational ambiguities caused by omission. To address these limitations, we introduce a novel Zero-shot Query Reformulation (ZeQR) framework that reformulates queries based on previous dialogue contexts without requiring supervision from conversational search data. Specifically, our framework utilizes language models designed for machine reading comprehension tasks to explicitly resolve two common ambiguities: cor
    
[^14]: WinoQueer：针对大型语言模型中反LGBTQ+偏见的社区协同基准

    WinoQueer: A Community-in-the-Loop Benchmark for Anti-LGBTQ+ Bias in Large Language Models. (arXiv:2306.15087v1 [cs.CL])

    [http://arxiv.org/abs/2306.15087](http://arxiv.org/abs/2306.15087)

    WinoQueer是一个社区协同基准，旨在衡量大型语言模型是否存在对LGBTQ+社区有害的偏见。研究发现现成模型普遍存在相当大的反同偏见，通过在该社区撰写或由该社区成员撰写的数据上进行微调，可以在一定程度上减轻偏见。

    

    我们提出了WinoQueer：一个专门设计用来测试大型语言模型（LLMs）是否存在对LGBTQ+社区有害的偏见的基准。该基准是通过一种新颖的方法从社区调查中生成的偏见基准。我们将该基准应用于几个流行的LLMs，并发现现成模型普遍存在相当大的反同偏见。最后，我们展示了通过在该社区撰写或由该社区成员撰写的数据上进行微调，可以在一定程度上减轻LLM对边缘化社区的偏见，并且社区成员撰写的社交媒体文本比非社区成员撰写的新闻文本更有效。我们的社区协同基准开发方法为未来的研究人员提供了一个蓝图，以开发面向其他边缘化社区的、以社区为中心的、基于伤害的LLM基准。

    We present WinoQueer: a benchmark specifically designed to measure whether large language models (LLMs) encode biases that are harmful to the LGBTQ+ community. The benchmark is community-sourced, via application of a novel method that generates a bias benchmark from a community survey. We apply our benchmark to several popular LLMs and find that off-the-shelf models generally do exhibit considerable anti-queer bias. Finally, we show that LLM bias against a marginalized community can be somewhat mitigated by finetuning on data written about or by members of that community, and that social media text written by community members is more effective than news text written about the community by non-members. Our method for community-in-the-loop benchmark development provides a blueprint for future researchers to develop community-driven, harms-grounded LLM benchmarks for other marginalized communities.
    

