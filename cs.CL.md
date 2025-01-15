# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [MoPE: Parameter-Efficient and Scalable Multimodal Fusion via Mixture of Prompt Experts](https://arxiv.org/abs/2403.10568) | 本文提出了MoPE技术，通过解开提示以自适应捕获数据集级和实例级特征，引入了混合Prompt专家来增强表达能力，并且在多模态融合中表现出更大的表达能力和可扩展性。 |
| [^2] | [Divergences between Language Models and Human Brains](https://arxiv.org/abs/2311.09308) | 该论文系统地探索了语言模型（LMs）和人类大脑在语言处理方面的差异，发现在社交/情感智能和物理常识领域，LMs无法很好地捕捉到人类的表现，但在这些领域对LMs进行微调可以提高其性能。 |
| [^3] | [JsonTuning: Towards Generalizable, Robust, and Controllable Instruction Tuning.](http://arxiv.org/abs/2310.02953) | JsonTuning是一种面向通用、强大和可控的指令调优方法，通过利用JSON的结构化特性，帮助模型理解任务要素及其关系，从而扩展了通用性、提高了稳健性，并增强了对输出的控制。 |
| [^4] | [FLM-101B: An Open LLM and How to Train It with $100K Budget.](http://arxiv.org/abs/2309.03852) | 本文介绍了一种开放的LLM模型（FLM-101B）以及如何用10万美元的预算来训练它。通过采用增长策略，可以显著降低LLM训练的成本。同时，引入了一种系统的评估方法，以评估LLM的智能能力。 |
| [^5] | [Unleashing the Imagination of Text: A Novel Framework for Text-to-image Person Retrieval via Exploring the Power of Words.](http://arxiv.org/abs/2307.09059) | 本研究提出了一个新的框架，通过探索文本中的文字的力量，实现了准确地将抽象的文本描述映射到具体的图像，从而实现了文本到图像的人物检索。 |

# 详细

[^1]: MoPE：通过Prompt专家混合实现参数高效和可扩展的多模态融合

    MoPE: Parameter-Efficient and Scalable Multimodal Fusion via Mixture of Prompt Experts

    [https://arxiv.org/abs/2403.10568](https://arxiv.org/abs/2403.10568)

    本文提出了MoPE技术，通过解开提示以自适应捕获数据集级和实例级特征，引入了混合Prompt专家来增强表达能力，并且在多模态融合中表现出更大的表达能力和可扩展性。

    

    Prompt调整已经证明在融合多模态任务的单模基础模型时具有参数效率性。然而，其有限的适应性和表达能力导致性能不佳与其他调整方法相比。本文通过将简单提示解开以自适应地捕获数据集级和实例级特征来解决这个问题。建立在这种解开的基础上，我们引入了Prompt专家的混合（MoPE）技术来增强表达能力。MoPE利用多模态配对先验在每个实例基础上路由最有效的提示。与简单提示相比，我们基于MoPE的条件提示对多模态融合具有更大的表达能力，在训练数据和可训练参数总数上具有更好的扩展性。我们还研究了一个专家路由的正则化项，导致专家的不断发展专长，不同专家专注于不同的特征。

    arXiv:2403.10568v1 Announce Type: cross  Abstract: Prompt-tuning has demonstrated parameter-efficiency in fusing unimodal foundation models for multimodal tasks. However, its limited adaptivity and expressiveness lead to suboptimal performance when compared with other tuning methods. In this paper, we address this issue by disentangling the vanilla prompts to adaptively capture dataset-level and instance-level features. Building upon this disentanglement, we introduce the mixture of prompt experts (MoPE) technique to enhance expressiveness. MoPE leverages multimodal pairing priors to route the most effective prompt on a per-instance basis. Compared to vanilla prompting, our MoPE-based conditional prompting exhibits greater expressiveness for multimodal fusion, scaling better with the training data and the overall number of trainable parameters. We also study a regularization term for expert routing, leading to emergent expert specialization, where different experts focus on different c
    
[^2]: 语言模型与人脑的差异

    Divergences between Language Models and Human Brains

    [https://arxiv.org/abs/2311.09308](https://arxiv.org/abs/2311.09308)

    该论文系统地探索了语言模型（LMs）和人类大脑在语言处理方面的差异，发现在社交/情感智能和物理常识领域，LMs无法很好地捕捉到人类的表现，但在这些领域对LMs进行微调可以提高其性能。

    

    机器和人类是否以相似的方式处理语言？最近的研究暗示肯定，发现大脑信号可以通过语言模型（LMs）的内部表示有效地进行预测。尽管这样的结果被认为反映了LMs和人类大脑之间的共享计算原理，但LMs和人类在语言表示和使用上也存在明显的差异。在这项工作中，我们通过检查LM表示和人类大脑对语言的响应之间的差异，通过采用两个数据集对受试者阅读和听叙述故事的方式，系统地探索了人类和机器语言处理之间的分歧。通过数据驱动的方法，我们确定了两个领域，即社交/情感智能和物理常识，这些领域在LMs中无法很好地捕捉到。然后，我们使用人类行为实验验证了这些领域，并证明在这些领域对LMs进行微调可以改善其性能。

    Do machines and humans process language in similar ways? Recent research has hinted in the affirmative, finding that brain signals can be effectively predicted using the internal representations of language models (LMs). Although such results are thought to reflect shared computational principles between LMs and human brains, there are also clear differences in how LMs and humans represent and use language. In this work, we systematically explore the divergences between human and machine language processing by examining the differences between LM representations and human brain responses to language as measured by Magnetoencephalography (MEG) across two datasets in which subjects read and listened to narrative stories. Using a data-driven approach, we identify two domains that are not captured well by LMs: social/emotional intelligence and physical commonsense. We then validate these domains with human behavioral experiments and show that fine-tuning LMs on these domains can improve th
    
[^3]: JsonTuning：面向通用、强大和可控的指令调优

    JsonTuning: Towards Generalizable, Robust, and Controllable Instruction Tuning. (arXiv:2310.02953v1 [cs.CL])

    [http://arxiv.org/abs/2310.02953](http://arxiv.org/abs/2310.02953)

    JsonTuning是一种面向通用、强大和可控的指令调优方法，通过利用JSON的结构化特性，帮助模型理解任务要素及其关系，从而扩展了通用性、提高了稳健性，并增强了对输出的控制。

    

    指令调优已成为利用大型语言模型（LLM）能力的关键过程，通过提供明确的任务指令，从而在各种任务中提高性能。然而，目前的文本-文本指令调优（TextTuning）方法由于任务的模糊性和缺乏明确的结构而存在通用性、稳健性和可控性的限制。在本文中，我们提出了JsonTuning，这是一种新的结构到结构的指令调优方法。通过利用JSON的多功能和结构化特性来表示任务，JsonTuning通过帮助模型理解关键任务要素及其关系，扩展了通用性，通过最小化歧义性提高了稳健性，并通过提供对输出的显式控制增强了可控性。我们对不同的语言模型和评估基准进行了全面的比较研究。实验结果表明，JsonTuning在性能上优于TextTuning。

    Instruction tuning has emerged as a crucial process for harnessing the capabilities of large language models (LLMs) by providing explicit task instructions, leading to improved performance in various tasks. However, prevalent text-to-text instruction tuning (TextTuning) methods suffer from limitations in generalization, robustness, and controllability due to the ambiguity and lack of explicit structure in tasks. In this paper, we propose JsonTuning, a novel structure-to-structure approach for instruction tuning. By leveraging the versatility and structured nature of JSON to represent tasks, JsonTuning enhances generalization by helping the model understand essential task elements and their relations, improves robustness by minimizing ambiguity, and increases controllability by providing explicit control over the output. We conduct a comprehensive comparative study with diverse language models and evaluation benchmarks. Experimental results show that JsonTuning outperforms TextTuning in
    
[^4]: FLM-101B：一种开放的LLM和如何用10万美元预算来训练它

    FLM-101B: An Open LLM and How to Train It with $100K Budget. (arXiv:2309.03852v1 [cs.CL])

    [http://arxiv.org/abs/2309.03852](http://arxiv.org/abs/2309.03852)

    本文介绍了一种开放的LLM模型（FLM-101B）以及如何用10万美元的预算来训练它。通过采用增长策略，可以显著降低LLM训练的成本。同时，引入了一种系统的评估方法，以评估LLM的智能能力。

    

    大型语言模型（LLMs）在自然语言处理和多模态任务中取得了显著的成功。然而，它们的发展面临两个主要挑战：（i）高计算成本；（ii）难以进行公平客观的评估。LLMs的价格昂贵，只有少数几家主要参与者有能力进行训练，从而限制了研究和应用机会。这凸显了成本效益的LLM训练的重要性。在本文中，我们采用了一种增长策略，显著降低LLM训练成本。我们证明了可以在10万美元的预算下训练具有101B参数和0.31TB令牌的LLM。我们还采用了一种系统的评估范式，用于对LLMs进行智能的智商评估，这是针对现有评估更注重知识能力的补充。我们引入了包括符号映射、规则理解、模式挖掘在内的重要智能方面的评估基准。

    Large language models (LLMs) have achieved remarkable success in NLP and multimodal tasks. Despite these successes, their development faces two main challenges: (i) high computational cost; and (ii) difficulty in conducting fair and objective evaluations. LLMs are prohibitively expensive, making it feasible for only a few major players to undertake their training, thereby constraining both research and application opportunities. This underscores the importance of cost-effective LLM training. In this paper, we utilize a growth strategy to significantly reduce LLM training cost. We demonstrate that an LLM with 101B parameters and 0.31TB tokens can be trained on a $100K budget. We also adopt a systematic evaluation paradigm for the IQ evaluation of LLMs, in complement to existing evaluations that focus more on knowledge-oriented abilities. We introduce our benchmark including evaluations on important aspects of intelligence including symbolic mapping, itrule understanding, pattern mining,
    
[^5]: 文字想象的释放：通过探索文字的力量实现文本到图像的人物检索的新框架

    Unleashing the Imagination of Text: A Novel Framework for Text-to-image Person Retrieval via Exploring the Power of Words. (arXiv:2307.09059v1 [cs.CL])

    [http://arxiv.org/abs/2307.09059](http://arxiv.org/abs/2307.09059)

    本研究提出了一个新的框架，通过探索文本中的文字的力量，实现了准确地将抽象的文本描述映射到具体的图像，从而实现了文本到图像的人物检索。

    

    文本到图像的人物检索的目标是从大型图库中检索与给定文本描述相匹配的人物图像。这个任务的主要挑战在于视觉和文本模态之间信息表示的显著差异。文本模态通过词汇和语法结构传递抽象和精确的信息，而视觉模态通过图像传递具体和直观的信息。为了充分利用文字表示的表达力，准确地将抽象的文本描述映射到具体图像是至关重要的。为了解决这个问题，我们提出了一个新的框架，通过探索句子中的文字的力量，释放了文本到图像人物检索中的文字想象力。具体来说，该框架使用预训练的全面CLIP模型作为图像和文本的双编码器，利用先前的跨模态对齐知识。

    The goal of Text-to-image person retrieval is to retrieve person images from a large gallery that match the given textual descriptions. The main challenge of this task lies in the significant differences in information representation between the visual and textual modalities. The textual modality conveys abstract and precise information through vocabulary and grammatical structures, while the visual modality conveys concrete and intuitive information through images. To fully leverage the expressive power of textual representations, it is essential to accurately map abstract textual descriptions to specific images.  To address this issue, we propose a novel framework to Unleash the Imagination of Text (UIT) in text-to-image person retrieval, aiming to fully explore the power of words in sentences. Specifically, the framework employs the pre-trained full CLIP model as a dual encoder for the images and texts , taking advantage of prior cross-modal alignment knowledge. The Text-guided Imag
    

