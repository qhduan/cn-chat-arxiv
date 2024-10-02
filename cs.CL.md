# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Design as Desired: Utilizing Visual Question Answering for Multimodal Pre-training](https://arxiv.org/abs/2404.00226) | 本研究是首次利用视觉问答（VQA）进行多模态预训练，专注于引导模型学习所需病理特征，并提出了一种无需额外专家注释的问题-答案对设计方法，以及一种准文本特征转换器模块。 |
| [^2] | [Large Language Models Are Unconscious of Unreasonability in Math Problems](https://arxiv.org/abs/2403.19346) | 本文研究了大型语言模型在解决数学问题中对不合理性的反应，设计了不合理数学问题基准以及关键计算和结论提示模板，提升了它们在错误检测和修正方面的能力。 |
| [^3] | [Don't Listen To Me: Understanding and Exploring Jailbreak Prompts of Large Language Models](https://arxiv.org/abs/2403.17336) | 该论文系统化了关于大型语言模型越狱提示的存在形式，并衡量了它们的越狱潜力，以更好地理解语义上具有意义的越狱提示的威胁格局。 |
| [^4] | [Outcome-Constrained Large Language Models for Countering Hate Speech](https://arxiv.org/abs/2403.17146) | 该研究探索了利用大型语言模型生成受潜在对话结果限制的对话，以应对在线仇恨言论，通过构建对话结果分类器和提出整合方法，为在线环境中生成对抗性对话提供了新途径 |
| [^5] | [How Far Are We on the Decision-Making of LLMs? Evaluating LLMs' Gaming Ability in Multi-Agent Environments](https://arxiv.org/abs/2403.11807) | 通过博弈论视角评估LLMs的决策能力，结果表明GPT-3.5在稳健性方面表现良好，但泛化能力有限，而GPT-4则优于其他模型。 |
| [^6] | [GEAR: An Efficient KV Cache Compression Recipefor Near-Lossless Generative Inference of LLM](https://arxiv.org/abs/2403.05527) | GEAR提出了一种高效的KV缓存压缩框架，实现几乎无损的高比率压缩，用于解决大型语言模型推断中因缓存需求增长而导致的记忆绑定问题和性能下降。 |
| [^7] | [How Far Are We from Intelligent Visual Deductive Reasoning?](https://arxiv.org/abs/2403.04732) | 目前的视觉语言模型在文本推理方面表现出色，但在视觉演绎推理方面仍存在较大差距和盲点。 |
| [^8] | [DiffuCOMET: Contextual Commonsense Knowledge Diffusion](https://arxiv.org/abs/2402.17011) | DiffuCOMET是一种利用扩散学习来重构叙述上下文与相关常识知识之间语义连接的知识模型，生成的知识在常识多样性、上下文相关性和对已知参考文献的对齐方面达到更好的平衡。 |
| [^9] | [Developments in Sheaf-Theoretic Models of Natural Language Ambiguities](https://arxiv.org/abs/2402.04505) | 本论文扩展了层理论模型从词汇歧义到篇章歧义，通过计算新的上下文性度量，发现上下文模型的比例大幅增加，并通过将Winograd Schema建模为Bell-CHSH场景，展示了层理论模型在处理具有指代歧义的自然语言挑战上的应用。 |
| [^10] | [How You Prompt Matters! Even Task-Oriented Constraints in Instructions Affect LLM-Generated Text Detection](https://arxiv.org/abs/2311.08369) | 即使是任务约束也会影响LLM生成文本的检测性能，本研究发现即使这些约束与规避无关，也会导致现有检测器性能具有显著差异 |
| [^11] | [Universal Vulnerabilities in Large Language Models: In-context Learning Backdoor Attacks.](http://arxiv.org/abs/2401.05949) | 本研究发现上下文学习范式在大型语言模型中存在漏洞，攻击者可以通过污染示范上下文来操控模型行为，而无需进行微调。这项研究设计了一种名为ICLAttack的后门攻击方法，可以通过污染示范样本和提示来使模型按照预定义的意图行事。 |
| [^12] | [Think Twice: A Human-like Two-stage Conversational Agent for Emotional Response Generation.](http://arxiv.org/abs/2301.04907) | 根据 “三思而后语” 行为启发，提出一种两阶段对话代理用于生成情感对话，该代理在情感生成方面优于其他模型，并保持了语义表现。 |

# 详细

[^1]: 想要的设计：利用视觉问答进行多模态预训练

    Design as Desired: Utilizing Visual Question Answering for Multimodal Pre-training

    [https://arxiv.org/abs/2404.00226](https://arxiv.org/abs/2404.00226)

    本研究是首次利用视觉问答（VQA）进行多模态预训练，专注于引导模型学习所需病理特征，并提出了一种无需额外专家注释的问题-答案对设计方法，以及一种准文本特征转换器模块。

    

    多模态预训练在医疗领域展示了其潜力，从成对的医疗报告中学习医学视觉表示。然而，许多预训练任务需要临床医生额外的注释，大多数任务未能明确引导模型学习不同病理特征。据我们所知，我们是第一个利用视觉问答（VQA）进行多模态预训练的团队，以引导框架专注于目标病理特征。在这项工作中，我们利用医疗报告中的描述设计了与不同疾病相关的多粒度问题-答案对，这有助于框架在预训练中无需专家额外的注释。我们还提出了一种新颖的预训练框架，其中包括一种准文本特征转换器模块，旨在通过将视觉特征转换到接近文本领域的准文本空间来辅助预训练过程。

    arXiv:2404.00226v1 Announce Type: cross  Abstract: Multimodal pre-training demonstrates its potential in the medical domain, which learns medical visual representations from paired medical reports. However, many pre-training tasks require extra annotations from clinicians, and most of them fail to explicitly guide the model to learn the desired features of different pathologies. To the best of our knowledge, we are the first to utilize Visual Question Answering (VQA) for multimodal pre-training to guide the framework focusing on targeted pathological features. In this work, we leverage descriptions in medical reports to design multi-granular question-answer pairs associated with different diseases, which assist the framework in pre-training without requiring extra annotations from experts. We also propose a novel pre-training framework with a quasi-textual feature transformer, a module designed to transform visual features into a quasi-textual space closer to the textual domain via a c
    
[^2]: 大型语言模型在数学问题中对不合理性毫无意识

    Large Language Models Are Unconscious of Unreasonability in Math Problems

    [https://arxiv.org/abs/2403.19346](https://arxiv.org/abs/2403.19346)

    本文研究了大型语言模型在解决数学问题中对不合理性的反应，设计了不合理数学问题基准以及关键计算和结论提示模板，提升了它们在错误检测和修正方面的能力。

    

    大型语言模型(LLMs)展示了在解决数学问题方面的巨大能力。然而，当给出包含不合理错误的问题时，它们倾向于产生幻觉。在本文中，我们研究了LLMs在面对不合理数学问题时的行为，并进一步探讨了它们解决这些问题的潜力。首先，我们构建了不合理数学问题(UMP)基准来检查LLMs的错误检测能力。实验证明，LLMs能够检测到不合理错误，但仍然在生成非幻觉内容方面失败。为了改善它们的错误检测和修正能力，我们进一步设计了一种称为关键计算和结论(CCC)的战略提示模板。通过CCC，LLMs可以更好地自我评估并检测数学问题中的不合理错误，使它们在实际应用场景中更可靠和安全。

    arXiv:2403.19346v1 Announce Type: new  Abstract: Large language models (LLMs) demonstrate substantial capabilities in solving math problems. However, they tend to produce hallucinations when given questions containing unreasonable errors. In this paper, we study the behavior of LLMs when faced with unreasonable math problems and further explore their potential to address these problems. First, we construct the Unreasonable Math Problem (UMP) benchmark to examine the error detection ability of LLMs. Experiments show that LLMs are able to detect unreasonable errors, but still fail in generating non-hallucinatory content. In order to improve their ability of error detection and correction, we further design a strategic prompt template called Critical Calculation and Conclusion(CCC). With CCC, LLMs can better self-evaluate and detect unreasonable errors in math questions, making them more reliable and safe in practical application scenarios.
    
[^3]: 不要听我的话：理解和探索大型语言模型的越狱提示

    Don't Listen To Me: Understanding and Exploring Jailbreak Prompts of Large Language Models

    [https://arxiv.org/abs/2403.17336](https://arxiv.org/abs/2403.17336)

    该论文系统化了关于大型语言模型越狱提示的存在形式，并衡量了它们的越狱潜力，以更好地理解语义上具有意义的越狱提示的威胁格局。

    

    生成式人工智能的最新进展使得大型语言模型（LLMs）能够无处不在地被访问。凭借其出色的理解和生成类似人类文本的能力，这些模型正日益融入我们的社会。与此同时，人们也对这种强大技术的潜在滥用表示担忧，并促使服务提供商采取了防御措施。为了克服这种保护机制，越狱提示最近已成为规避安全限制和引诱最初设计为被禁止的有害内容的最有效机制之一。由于LLM的快速发展及通过自然语言轻松获取的便利性，越狱提示的前沿主要出现在在线论坛和爱好者中。为了更好地了解语义上具有意义的越狱提示的威胁格局，我们系统化了现有提示并测量它们的越狱

    arXiv:2403.17336v1 Announce Type: cross  Abstract: Recent advancements in generative AI have enabled ubiquitous access to large language models (LLMs). Empowered by their exceptional capabilities to understand and generate human-like text, these models are being increasingly integrated into our society. At the same time, there are also concerns on the potential misuse of this powerful technology, prompting defensive measures from service providers. To overcome such protection, jailbreaking prompts have recently emerged as one of the most effective mechanisms to circumvent security restrictions and elicit harmful content originally designed to be prohibited.   Due to the rapid development of LLMs and their ease of access via natural languages, the frontline of jailbreak prompts is largely seen in online forums and among hobbyists. To gain a better understanding of the threat landscape of semantically meaningful jailbreak prompts, we systemized existing prompts and measured their jailbre
    
[^4]: 用于抵制仇恨言论的结果受限大型语言模型

    Outcome-Constrained Large Language Models for Countering Hate Speech

    [https://arxiv.org/abs/2403.17146](https://arxiv.org/abs/2403.17146)

    该研究探索了利用大型语言模型生成受潜在对话结果限制的对话，以应对在线仇恨言论，通过构建对话结果分类器和提出整合方法，为在线环境中生成对抗性对话提供了新途径

    

    挑战或回应仇恨言论的对话被视为缓解仇恨言论的负面影响并促进在线交流的替代方法。研究已致力于使用语言模型自动生成对话以协助打击在线仇恨言论。现有研究侧重于生成具有特定语言属性（如礼貌、信息丰富和意图驱动）的对话。然而，对话可能对在线环境产生什么影响仍不明确。我们首先探讨利用大型语言模型（LLM）生成受潜在对话结果限制的对话的方法。我们构建了两个对话结果分类器，用Reddit数据预测应对仇恨言论后的不文明程度和仇恨者重新关注行为，然后提出了四种方法来整合所需的结果，即低礼貌

    arXiv:2403.17146v1 Announce Type: new  Abstract: Counterspeech that challenges or responds to hate speech has been seen as an alternative to mitigate the negative impact of hate speech and foster productive online communications. Research endeavors have been directed to using language models for the automatic generation of counterspeech to assist efforts in combating online hate. Existing research focuses on the generation of counterspeech with certain linguistic attributes, such as being polite, informative, and intent-driven. However, it remains unclear what impact the counterspeech might have in an online environment. We first explore methods that utilize large language models (LLM) to generate counterspeech constrained by potential conversation outcomes. We build two conversation outcome classifiers that predict the incivility level and the hater reentry behavior following replies to hate with Reddit data, then propose four methods to incorporate the desired outcomes, i.e., low con
    
[^5]: LLM的决策水平在多智能体环境中的评估究竟如何？

    How Far Are We on the Decision-Making of LLMs? Evaluating LLMs' Gaming Ability in Multi-Agent Environments

    [https://arxiv.org/abs/2403.11807](https://arxiv.org/abs/2403.11807)

    通过博弈论视角评估LLMs的决策能力，结果表明GPT-3.5在稳健性方面表现良好，但泛化能力有限，而GPT-4则优于其他模型。

    

    决策是一个复杂的任务，需要各种能力，为评估大型语言模型（LLMs）提供了一个极好的框架。我们的研究通过博弈论的视角探究LLMs的决策能力。我们专注于支持多个智能体同时参与的游戏，引入了我们的框架GAMA-Bench，包括八个经典的多智能体游戏。我们设计了一个评分方案，定量评估模型在这些游戏中的表现。通过GAMA-Bench，我们研究了LLMs的稳健性、泛化能力和增强策略。结果显示，虽然GPT-3.5表现出令人满意的稳健性，但其泛化能力相对有限。然而，通过一些方法如“思维链”，其性能可以得到提高。此外，我们对各种LLMs进行评估，发现GPT-4胜过其他模型。

    arXiv:2403.11807v1 Announce Type: new  Abstract: Decision-making, a complicated task requiring various types of abilities, presents an excellent framework for assessing Large Language Models (LLMs). Our research investigates LLMs' decision-making capabilities through the lens of a well-established field, Game Theory. We focus specifically on games that support the participation of more than two agents simultaneously. Subsequently, we introduce our framework, GAMA-Bench, including eight classical multi-agent games. We design a scoring scheme to assess a model's performance in these games quantitatively. Through GAMA-Bench, we investigate LLMs' robustness, generalizability, and enhancement strategies. Results reveal that while GPT-3.5 shows satisfying robustness, its generalizability is relatively limited. However, its performance can be improved through approaches such as Chain-of-Thought. Additionally, we conduct evaluations across various LLMs and find that GPT-4 outperforms other mod
    
[^6]: GEAR: 一种用于几乎无损生成推断大型语言模型的高效KV缓存压缩方案

    GEAR: An Efficient KV Cache Compression Recipefor Near-Lossless Generative Inference of LLM

    [https://arxiv.org/abs/2403.05527](https://arxiv.org/abs/2403.05527)

    GEAR提出了一种高效的KV缓存压缩框架，实现几乎无损的高比率压缩，用于解决大型语言模型推断中因缓存需求增长而导致的记忆绑定问题和性能下降。

    

    关键-值（KV）缓存已成为加快大型语言模型（LLMs）推断生成速度的事实标准。然而，随着序列长度增加而增长的缓存需求已将LLM推断转变为一个记忆绑定问题，显著地限制了系统吞吐量。现有方法依赖于丢弃不重要的标记或均匀量化所有条目。然而，这种方法往往会产生较高的近似误差来表示压缩后的矩阵。自回归解码过程进一步增加了每个步骤的误差，导致模型生成中的重大偏差和性能恶化。为了解决这一挑战，我们提出了GEAR，一种高效的KV缓存压缩框架，实现几乎无损的高压缩比。

    arXiv:2403.05527v1 Announce Type: cross  Abstract: Key-value (KV) caching has become the de-facto to accelerate generation speed for large language models (LLMs) inference. However, the growing cache demand with increasing sequence length has transformed LLM inference to be a memory bound problem, significantly constraining the system throughput. Existing methods rely on dropping unimportant tokens or quantizing all entries uniformly. Such methods, however, often incur high approximation errors to represent the compressed matrices. The autoregressive decoding process further compounds the error of each step, resulting in critical deviation in model generation and deterioration of performance. To tackle this challenge, we propose GEAR, an efficient KV cache compression framework that achieves near-lossless high-ratio compression. GEAR first applies quantization to majority of entries of similar magnitudes to ultra-low precision. It then employs a low rank matrix to approximate the quant
    
[^7]: 我们距离智能视觉演绎推理还有多远？

    How Far Are We from Intelligent Visual Deductive Reasoning?

    [https://arxiv.org/abs/2403.04732](https://arxiv.org/abs/2403.04732)

    目前的视觉语言模型在文本推理方面表现出色，但在视觉演绎推理方面仍存在较大差距和盲点。

    

    最近，诸如GPT-4V之类的视觉语言模型（VLM）在各种视觉语言任务上取得了巨大进展。我们深入探讨了基于视觉的演绎推理，这是一个更复杂但不太被探索的领域，并发现了当前领先的VLM中以前未暴露的盲点。具体来说，我们利用瑞文渐进矩阵（RPM）来评估VLM在仅依靠视觉线索进行多跳关系和演绎推理的能力。我们对几种流行的VLM进行了全面评估，采用了标准策略，如上下文学习、自我一致性和思维链（CoT），在三个不同的数据集上进行了评估，包括Mensa智商测试、智商测试和RAVEN。结果表明，尽管LLM在基于文本的推理方面具有令人印象深刻的能力，但我们在视觉演绎推理方面仍有很大的差距。

    arXiv:2403.04732v1 Announce Type: new  Abstract: Vision-Language Models (VLMs) such as GPT-4V have recently demonstrated incredible strides on diverse vision language tasks. We dig into vision-based deductive reasoning, a more sophisticated but less explored realm, and find previously unexposed blindspots in the current SOTA VLMs. Specifically, we leverage Raven's Progressive Matrices (RPMs), to assess VLMs' abilities to perform multi-hop relational and deductive reasoning relying solely on visual clues. We perform comprehensive evaluations of several popular VLMs employing standard strategies such as in-context learning, self-consistency, and Chain-of-thoughts (CoT) on three diverse datasets, including the Mensa IQ test, IntelligenceTest, and RAVEN. The results reveal that despite the impressive capabilities of LLMs in text-based reasoning, we are still far from achieving comparable proficiency in visual deductive reasoning. We found that certain standard strategies that are effective
    
[^8]: DiffuCOMET: 上下文常识知识扩散

    DiffuCOMET: Contextual Commonsense Knowledge Diffusion

    [https://arxiv.org/abs/2402.17011](https://arxiv.org/abs/2402.17011)

    DiffuCOMET是一种利用扩散学习来重构叙述上下文与相关常识知识之间语义连接的知识模型，生成的知识在常识多样性、上下文相关性和对已知参考文献的对齐方面达到更好的平衡。

    

    推理上下文相关且多样化的常识以理解叙述故事对于知识模型仍然具有挑战性。在这项工作中，我们开发了一系列利用扩散的知识模型DiffuCOMET，以学习重构叙述上下文与相关常识知识之间的隐式语义连接。通过多次扩散步骤，我们的方法逐步完善了与叙述锚定的常识事实表示，为输入上下文生成上下文相关且多样化的常识推断。为了评估DiffuCOMET，我们引入了衡量常识推断的新指标，更密切地衡量知识多样性和上下文相关性。我们在两个不同的基准数据集，ComFact和WebNLG+上的结果显示，DiffuCOMET生成的知识在常识多样性、上下文相关性以及与已知黄金参考文献的对齐之间实现了更好的权衡，与基线方法相比。

    arXiv:2402.17011v1 Announce Type: new  Abstract: Inferring contextually-relevant and diverse commonsense to understand narratives remains challenging for knowledge models. In this work, we develop a series of knowledge models, DiffuCOMET, that leverage diffusion to learn to reconstruct the implicit semantic connections between narrative contexts and relevant commonsense knowledge. Across multiple diffusion steps, our method progressively refines a representation of commonsense facts that is anchored to a narrative, producing contextually-relevant and diverse commonsense inferences for an input context. To evaluate DiffuCOMET, we introduce new metrics for commonsense inference that more closely measure knowledge diversity and contextual relevance. Our results on two different benchmarks, ComFact and WebNLG+, show that knowledge generated by DiffuCOMET achieves a better trade-off between commonsense diversity, contextual relevance and alignment to known gold references, compared to basel
    
[^9]: 自然语言歧义的层理论模型的发展

    Developments in Sheaf-Theoretic Models of Natural Language Ambiguities

    [https://arxiv.org/abs/2402.04505](https://arxiv.org/abs/2402.04505)

    本论文扩展了层理论模型从词汇歧义到篇章歧义，通过计算新的上下文性度量，发现上下文模型的比例大幅增加，并通过将Winograd Schema建模为Bell-CHSH场景，展示了层理论模型在处理具有指代歧义的自然语言挑战上的应用。

    

    层是数学对象，由基础构成的拓扑空间和与之相关的数据组成，例如定义在开集上的连续函数。层最初用于代数拓扑和逻辑中。最近，它们也用于建模物理实验和自然语言消岐过程等事件。我们将这些模型从词汇歧义扩展到由指代产生的篇章歧义。首先，对一组基本的指代篇章数据计算了一个新的上下文性度量，结果表明上下文模型的比例更高，为82.9%，而之前的工作只有3.17%的上下文模型。随后，我们展示了如何将包含指代歧义的自然语言处理挑战——Winograd Schema建模为Bell-CHSH场景，其上下文比例为0.096。

    Sheaves are mathematical objects consisting of a base which constitutes a topological space and the data associated with each open set thereof, e.g. continuous functions defined on the open sets. Sheaves have originally been used in algebraic topology and logic. Recently, they have also modelled events such as physical experiments and natural language disambiguation processes. We extend the latter models from lexical ambiguities to discourse ambiguities arising from anaphora. To begin, we calculated a new measure of contextuality for a dataset of basic anaphoric discourses, resulting in a higher proportion of contextual models--82.9%--compared to previous work which only yielded 3.17% contextual models. Then, we show how an extension of the natural language processing challenge, known as the Winograd Schema, which involves anaphoric ambiguities can be modelled on the Bell-CHSH scenario with a contextual fraction of 0.096.
    
[^10]: 指令方式的重要性：即使任务约束也会影响LLM生成文本的检测

    How You Prompt Matters! Even Task-Oriented Constraints in Instructions Affect LLM-Generated Text Detection

    [https://arxiv.org/abs/2311.08369](https://arxiv.org/abs/2311.08369)

    即使是任务约束也会影响LLM生成文本的检测性能，本研究发现即使这些约束与规避无关，也会导致现有检测器性能具有显著差异

    

    为了对抗大型语言模型（LLMs）的滥用，许多最近的研究提出了性能可靠的LLM生成文本检测器。当用户指示LLMs生成文本时，指令可以根据用户需求包含不同的约束。然而，大多数最近的研究在为LLM检测创建数据集时并没有涵盖这种多样化的指令模式。本文发现，即使是任务导向的约束——这些约束自然会包含在指令中，并且与检测规避无关——也会导致现有的检测器在检测性能上具有较大的方差。我们以学生作文写作为现实领域，并根据几个因素手动创建基于作文质量的任务约束。我们的实验表明，在带有这种约束的指令生成的文本上，当前检测器性能的标准偏差（SD）显著较大。

    arXiv:2311.08369v2 Announce Type: replace  Abstract: To combat the misuse of Large Language Models (LLMs), many recent studies have presented LLM-generated-text detectors with promising performance. When users instruct LLMs to generate texts, the instruction can include different constraints depending on the user's need. However, most recent studies do not cover such diverse instruction patterns when creating datasets for LLM detection. In this paper, we find that even task-oriented constraints -- constraints that would naturally be included in an instruction and are not related to detection-evasion -- cause existing detectors to have a large variance in detection performance. We focus on student essay writing as a realistic domain and manually create task-oriented constraints based on several factors for essay quality. Our experiments show that the standard deviation (SD) of current detector performance on texts generated by an instruction with such a constraint is significantly large
    
[^11]: 大型语言模型中的通用漏洞：上下文学习后门攻击

    Universal Vulnerabilities in Large Language Models: In-context Learning Backdoor Attacks. (arXiv:2401.05949v1 [cs.CL])

    [http://arxiv.org/abs/2401.05949](http://arxiv.org/abs/2401.05949)

    本研究发现上下文学习范式在大型语言模型中存在漏洞，攻击者可以通过污染示范上下文来操控模型行为，而无需进行微调。这项研究设计了一种名为ICLAttack的后门攻击方法，可以通过污染示范样本和提示来使模型按照预定义的意图行事。

    

    上下文学习是一种在预训练和微调之间弥合差距的范式，在几个自然语言处理任务中展现了高效性，特别是在少样本设置中。与传统的微调方法不同，上下文学习能够适应未见过的任务而无需更新任何参数。尽管被广泛应用，上下文学习仍然容易受到恶意攻击。本研究提出了对这一范式的安全性问题的关切。我们的研究表明，攻击者可以通过污染示范上下文来操控大型语言模型的行为，而无需对模型进行微调。具体来说，我们设计了一种新的后门攻击方法，命名为ICLAttack，针对基于上下文学习的大型语言模型。我们的方法包括两种类型的攻击：污染示范样本和污染提示，可以使模型按照预定义的意图行事。ICLAttack不需要额外的微调。

    In-context learning, a paradigm bridging the gap between pre-training and fine-tuning, has demonstrated high efficacy in several NLP tasks, especially in few-shot settings. Unlike traditional fine-tuning methods, in-context learning adapts pre-trained models to unseen tasks without updating any parameters. Despite being widely applied, in-context learning is vulnerable to malicious attacks. In this work, we raise security concerns regarding this paradigm. Our studies demonstrate that an attacker can manipulate the behavior of large language models by poisoning the demonstration context, without the need for fine-tuning the model. Specifically, we have designed a new backdoor attack method, named ICLAttack, to target large language models based on in-context learning. Our method encompasses two types of attacks: poisoning demonstration examples and poisoning prompts, which can make models behave in accordance with predefined intentions. ICLAttack does not require additional fine-tuning 
    
[^12]: Think Twice：一种人类化的两阶段对话代理用于生成情感响应

    Think Twice: A Human-like Two-stage Conversational Agent for Emotional Response Generation. (arXiv:2301.04907v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2301.04907](http://arxiv.org/abs/2301.04907)

    根据 “三思而后语” 行为启发，提出一种两阶段对话代理用于生成情感对话，该代理在情感生成方面优于其他模型，并保持了语义表现。

    

    针对人类化的对话系统，目前情感对话方法采用统一的神经网络联合模型情感和语义。这种策略由于情感和语义之间的相互限制往往会产生安全的响应，并且需要罕见的情感标注大规模对话语料库。受到人类对话中“三思而后语”的行为启发，我们提出了一种用于生成情感对话的两阶段对话代理。首先，一个没有使用情感标注对话语料库训练的对话模型生成符合上下文语义的原型响应。其次，第一阶段原型将通过一个可控的情感优化器与共情假设进行修改。在DailyDialog和EmpatheticDialogues数据集上的实验结果表明，我们提出的对话代理在情感生成方面优于比较模型，并在自动和人类评估中保持了语义表现。

    Towards human-like dialogue systems, current emotional dialogue approaches jointly model emotion and semantics with a unified neural network. This strategy tends to generate safe responses due to the mutual restriction between emotion and semantics, and requires rare emotion-annotated large-scale dialogue corpus. Inspired by the "think twice" behavior in human dialogue, we propose a two-stage conversational agent for the generation of emotional dialogue. Firstly, a dialogue model trained without the emotion-annotated dialogue corpus generates a prototype response that meets the contextual semantics. Secondly, the first-stage prototype is modified by a controllable emotion refiner with the empathy hypothesis. Experimental results on the DailyDialog and EmpatheticDialogues datasets demonstrate that the proposed conversational outperforms the comparison models in emotion generation and maintains the semantic performance in automatic and human evaluations.
    

