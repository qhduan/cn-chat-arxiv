# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [AMOR: A Recipe for Building Adaptable Modular Knowledge Agents Through Process Feedback](https://rss.arxiv.org/abs/2402.01469) | AMOR是一个基于开源LLM的代理框架，通过与外部知识库进行推理和人类监督来适应特定领域的推理过程。通过两阶段的微调，AMOR能够在不同知识环境中泛化，并且可以根据过程反馈进行领域定制。 |
| [^2] | [Skews in the Phenomenon Space Hinder Generalization in Text-to-Image Generation](https://arxiv.org/abs/2403.16394) | 文本到图像生成领域的泛化问题源于现象空间中的偏差，需要量化和解决语言和视觉偏差，以提高泛化性能 |
| [^3] | [Improving Low-Resource Knowledge Tracing Tasks by Supervised Pre-training and Importance Mechanism Fine-tuning](https://arxiv.org/abs/2403.06725) | 本文提出了名为LoReKT的低资源知识追踪框架，通过监督预训练和微调重要性机制，旨在从丰富资源的KT数据集中学习可转移的参数和表示来改进低资源知识追踪任务。 |
| [^4] | [AutoRD: An Automatic and End-to-End System for Rare Disease Knowledge Graph Construction Based on Ontologies-enhanced Large Language Models](https://arxiv.org/abs/2403.00953) | AutoRD是一个自动化端到端系统，使用大型语言模型和医学知识图构建罕见疾病知识图，实现了整体F1得分47.3%，相对于基础LLM有14.4%的提升。 |
| [^5] | [Can GPT Improve the State of Prior Authorization via Guideline Based Automated Question Answering?](https://arxiv.org/abs/2402.18419) | 通过问答任务，GPT能够验证医疗领域患者的PA请求，帮助卫生计划更快地做出决策。 |
| [^6] | [Grasping the Essentials: Tailoring Large Language Models for Zero-Shot Relation Extraction](https://arxiv.org/abs/2402.11142) | 通过使用自然语言表达的关系定义来训练关系抽取模型的零-shot学习设置，从而为模型提供准确和明确的关系类型描述，并同时最小化注释要求。 |
| [^7] | [Is it Possible to Edit Large Language Models Robustly?](https://arxiv.org/abs/2402.05827) | 本研究旨在了解大型语言模型的稳健编辑方法的优势和局限性，从而促进对交流型人工智能的稳健、现实应用。 |
| [^8] | [Assessing the Brittleness of Safety Alignment via Pruning and Low-Rank Modifications](https://arxiv.org/abs/2402.05162) | 本研究通过修剪和低秩修改，发现大型语言模型（LLMs）的安全机制固有易碎性，去除安全关键区域会损害安全性，但对效用影响不大，需要更强健的安全策略。 |
| [^9] | [Democratizing Large Language Models via Personalized Parameter-Efficient Fine-tuning](https://arxiv.org/abs/2402.04401) | 这项研究通过个性化参数高效调整模块（PEFT）实现了大规模语言模型（LLM）的民主化，使用户能够拥有和使用他们自己的LLM，解决了传统方法中的定制能力和隐私问题。 |
| [^10] | [LLMRefine: Pinpointing and Refining Large Language Models via Fine-Grained Actionable Feedback](https://arxiv.org/abs/2311.09336) | LLMRefine提出了一种细粒度反馈模型来指导大型语言模型定位缺陷并进行优化，在机器翻译、长篇问答和主题总结等任务中取得显著的改进。 |
| [^11] | [Investigating Hallucinations in Pruned Large Language Models for Abstractive Summarization.](http://arxiv.org/abs/2311.09335) | 本文通过广泛的实证研究发现，修剪后的大型语言模型在抽象摘要任务中产生幻觉的情况较原始模型要少，表现更可靠，具有更高的效率和稀疏推理能力。 |
| [^12] | [Large Language Models Can Be Good Privacy Protection Learners.](http://arxiv.org/abs/2310.02469) | 本论文介绍了一种名为隐私保护语言模型（PPLM）的新范式，可以在保护数据隐私的同时有效注入领域特定知识。通过对模型设计的理论分析和不同技术的研究，我们验证了使用正向和负向示例进行指令微调的方法具有很大的潜力。 |
| [^13] | [Hate speech detection in algerian dialect using deep learning.](http://arxiv.org/abs/2309.11611) | 本研究提出了一个完整的方法，通过深度学习来检测在线阿尔及利亚信息中的仇恨言论。通过对阿尔及利亚社交网络上的语料库进行评估，我们获得了令人鼓舞的结果。 |
| [^14] | [Demonstration-based learning for few-shot biomedical named entity recognition under machine reading comprehension.](http://arxiv.org/abs/2308.06454) | 本研究提出了一种基于演示的学习方法，通过将生物医学命名实体识别重新定义为机器阅读理解问题，来解决少样本学习场景下的生物医学实体识别问题。实验证明，在少样本学习中，该方法比其他先进方法平均提高了1.1%的F1分数。 |

# 详细

[^1]: AMOR:通过进程反馈构建适应性模块化知识代理的方法

    AMOR: A Recipe for Building Adaptable Modular Knowledge Agents Through Process Feedback

    [https://rss.arxiv.org/abs/2402.01469](https://rss.arxiv.org/abs/2402.01469)

    AMOR是一个基于开源LLM的代理框架，通过与外部知识库进行推理和人类监督来适应特定领域的推理过程。通过两阶段的微调，AMOR能够在不同知识环境中泛化，并且可以根据过程反馈进行领域定制。

    

    大型语言模型（LLM）的显著成功引发了构建语言代理完成各种复杂任务的高潮。我们提出了基于开源LLM的代理框架AMOR，通过与外部知识库进行推理并通过人类监督来适应特定领域的推理过程。AMOR在有限状态机（FSM）上构建推理逻辑，通过自主执行和模块间转换解决问题。这使人们能够直接为单个模块提供反馈，从而自然形成了过程监督。基于这个推理和反馈框架，我们通过两阶段的微调开发了AMOR：预热和适应。前者使用从各种公共数据集自动构建的示例对LLM进行微调，使AMOR能够在不同的知识环境中泛化，后者使用过程反馈将AMOR量身定制到特定领域。在多个领域上进行了广泛的实验。

    The notable success of large language models (LLMs) has sparked an upsurge in building language agents to complete various complex tasks. We present AMOR, an agent framework based on open-source LLMs, which reasons with external knowledge bases and adapts to specific domains through human supervision to the reasoning process. AMOR builds reasoning logic over a finite state machine (FSM) that solves problems through autonomous executions and transitions over disentangled modules. This allows humans to provide direct feedback to the individual modules, and thus naturally forms process supervision. Based on this reasoning and feedback framework, we develop AMOR through two-stage fine-tuning: warm-up and adaptation. The former fine-tunes the LLM with examples automatically constructed from various public datasets and enables AMOR to generalize across different knowledge environments, while the latter tailors AMOR to specific domains using process feedback. Extensive experiments across mult
    
[^2]: 文本到图像生成中现象空间中的偏差阻碍了泛化

    Skews in the Phenomenon Space Hinder Generalization in Text-to-Image Generation

    [https://arxiv.org/abs/2403.16394](https://arxiv.org/abs/2403.16394)

    文本到图像生成领域的泛化问题源于现象空间中的偏差，需要量化和解决语言和视觉偏差，以提高泛化性能

    

    文本到图像生成领域的文献存在着关于如何忠实地组合实体与关系的问题。然而，缺乏对实体-关系组合如何有效学习的形式化理解。此外，反映问题结构的基础现象空间并不明确定义，导致为了希望泛化在大规模预训练中得以展现而不断追求更多数据。我们猜测基础现象学覆盖范围并未按比例扩展，导致所呈现现象的偏差对泛化造成了伤害。我们引入了统计度量标准来量化数据集中的语言和视觉偏差，用于关系学习，并表明文本到图像生成的泛化失败直接源于现象学覆盖不完整或不平衡。我们首先在合成领域进行实验和演示

    arXiv:2403.16394v1 Announce Type: cross  Abstract: The literature on text-to-image generation is plagued by issues of faithfully composing entities with relations. But there lacks a formal understanding of how entity-relation compositions can be effectively learned. Moreover, the underlying phenomenon space that meaningfully reflects the problem structure is not well-defined, leading to an arms race for larger quantities of data in the hope that generalization emerges out of large-scale pretraining. We hypothesize that the underlying phenomenological coverage has not been proportionally scaled up, leading to a skew of the presented phenomenon which harms generalization. We introduce statistical metrics that quantify both the linguistic and visual skew of a dataset for relational learning, and show that generalization failures of text-to-image generation are a direct result of incomplete or unbalanced phenomenological coverage. We first perform experiments in a synthetic domain and demo
    
[^3]: 通过监督预训练和重要性机制微调改进低资源知识追踪任务

    Improving Low-Resource Knowledge Tracing Tasks by Supervised Pre-training and Importance Mechanism Fine-tuning

    [https://arxiv.org/abs/2403.06725](https://arxiv.org/abs/2403.06725)

    本文提出了名为LoReKT的低资源知识追踪框架，通过监督预训练和微调重要性机制，旨在从丰富资源的KT数据集中学习可转移的参数和表示来改进低资源知识追踪任务。

    

    知识追踪（KT）旨在基于学生的历史互动来估计他们的知识掌握程度。最近，基于深度学习的KT（DLKT）方法在KT任务中取得了令人印象深刻的表现。然而，由于各种原因，如预算限制和隐私问题，许多实际场景中观察到的互动非常有限，即低资源KT数据集。直接在低资源KT数据集上训练DLKT模型可能会导致过拟合，并且很难选择适当的深度神经架构。因此，在本文中，我们提出了一个名为LoReKT的低资源KT框架来应对上述挑战。受盛行的“预训练和微调”范式的启发，我们旨在在预训练阶段从丰富资源的KT数据集中学习可转移的参数和表示。

    arXiv:2403.06725v1 Announce Type: cross  Abstract: Knowledge tracing (KT) aims to estimate student's knowledge mastery based on their historical interactions. Recently, the deep learning based KT (DLKT) approaches have achieved impressive performance in the KT task. These DLKT models heavily rely on the large number of available student interactions. However, due to various reasons such as budget constraints and privacy concerns, observed interactions are very limited in many real-world scenarios, a.k.a, low-resource KT datasets. Directly training a DLKT model on a low-resource KT dataset may lead to overfitting and it is difficult to choose the appropriate deep neural architecture. Therefore, in this paper, we propose a low-resource KT framework called LoReKT to address above challenges. Inspired by the prevalent "pre-training and fine-tuning" paradigm, we aim to learn transferable parameters and representations from rich-resource KT datasets during the pre-training stage and subseque
    
[^4]: AutoRD：一种基于本体增强的大型语言模型的罕见疾病知识图构建的自动化端到端系统

    AutoRD: An Automatic and End-to-End System for Rare Disease Knowledge Graph Construction Based on Ontologies-enhanced Large Language Models

    [https://arxiv.org/abs/2403.00953](https://arxiv.org/abs/2403.00953)

    AutoRD是一个自动化端到端系统，使用大型语言模型和医学知识图构建罕见疾病知识图，实现了整体F1得分47.3%，相对于基础LLM有14.4%的提升。

    

    目标：我们的目标是创建一个名为AutoRD的端到端系统，该系统自动从临床文本中提取有关罕见疾病的信息。我们进行了各种测试来评估AutoRD的性能，并在本文中强调了其优势和局限性。方法：我们的系统AutoRD是一个软件流水线，涉及数据预处理、实体提取、关系提取、实体校准和知识图构建。我们使用大型语言模型和由开源医学本体发展而来的医学知识图来实现这一目标。我们通过实体提取、关系提取以及知识图构建性能对系统进行定量评估。结果：AutoRD取得了47.3%的整体F1分数，较基础LLM提高了14.4%。具体来说，AutoRD实现了56.1%的整体实体提取F1分数（罕见疾病：83.5%，疾病：35.8%，s

    arXiv:2403.00953v1 Announce Type: cross  Abstract: Objectives: Our objective is to create an end-to-end system called AutoRD, which automates extracting information from clinical text about rare diseases. We have conducted various tests to evaluate the performance of AutoRD and highlighted its strengths and limitations in this paper.   Materials and Methods: Our system, AutoRD, is a software pipeline involving data preprocessing, entity extraction, relation extraction, entity calibration, and knowledge graph construction. We implement this using large language models and medical knowledge graphs developed from open-source medical ontologies. We quantitatively evaluate our system on entity extraction, relation extraction, and the performance of knowledge graph construction.   Results: AutoRD achieves an overall F1 score of 47.3%, a 14.4% improvement compared to the base LLM. In detail, AutoRD achieves an overall entity extraction F1 score of 56.1% (rare_disease: 83.5%, disease: 35.8%, s
    
[^5]: 能否通过基于指南的自动问答来改善GPT的先前授权状态？

    Can GPT Improve the State of Prior Authorization via Guideline Based Automated Question Answering?

    [https://arxiv.org/abs/2402.18419](https://arxiv.org/abs/2402.18419)

    通过问答任务，GPT能够验证医疗领域患者的PA请求，帮助卫生计划更快地做出决策。

    

    卫生保险公司有一个被称为先前授权（PA）的流程，这是一种卫生计划成本控制流程，要求医生和其他医疗专业人员在对患者执行特定程序之前必须事先获得卫生计划的批准，以便有资格获得支付覆盖。对卫生保险公司来说，批准医疗领域患者的PA请求是一项耗时且具有挑战性的任务。其中的一项关键挑战是验证请求是否符合某些标准，如年龄、性别等。在这项工作中，我们评估了GPT是否能验证大量关键因素，从而帮助卫生计划更快地做出决策。我们将其构建为一个问答任务，促使GPT从患者的电子健康记录中回答问题。我们尝试了不同的传统提示技术，同时还引入了我们自己的新颖提示技术。

    arXiv:2402.18419v1 Announce Type: cross  Abstract: Health insurance companies have a defined process called prior authorization (PA) which is a health plan cost-control process that requires doctors and other healthcare professionals to get clearance in advance from a health plan before performing a particular procedure on a patient in order to be eligible for payment coverage. For health insurance companies, approving PA requests for patients in the medical domain is a time-consuming and challenging task. One of those key challenges is validating if a request matches up to certain criteria such as age, gender, etc. In this work, we evaluate whether GPT can validate numerous key factors, in turn helping health plans reach a decision drastically faster. We frame it as a question answering task, prompting GPT to answer a question from patient electronic health record. We experiment with different conventional prompting techniques as well as introduce our own novel prompting technique. Mo
    
[^6]: 把握要点：定制大型语言模型进行零-shot关系抽取

    Grasping the Essentials: Tailoring Large Language Models for Zero-Shot Relation Extraction

    [https://arxiv.org/abs/2402.11142](https://arxiv.org/abs/2402.11142)

    通过使用自然语言表达的关系定义来训练关系抽取模型的零-shot学习设置，从而为模型提供准确和明确的关系类型描述，并同时最小化注释要求。

    

    关系抽取（RE）是自然语言处理中的一个关键任务，旨在识别文本中提及的实体之间的语义关系。尽管这一领域取得了显著进展，但现有模型通常依赖于大量的注释数据进行训练，获取这些数据可能既昂贵又耗时。此外，这些模型通常难以适应新的或未见过的关系。相比之下，少样本学习设置旨在减少注释要求，对于理解目标关系语义提供了不完整且有偏见的监督，导致性能下降且不稳定。为了为模型提供准确和明确的关系类型描述，同时最小化注释要求，我们研究了仅使用自然语言中表示的关系定义来训练RE模型的仅零-shot RE设置。受LLM（大型语言模型）强大的合成数据生成能力的启发，我们提出了一种

    arXiv:2402.11142v1 Announce Type: new  Abstract: Relation extraction (RE), a crucial task in NLP, aims to identify semantic relationships between entities mentioned in texts. Despite significant advancements in this field, existing models typically rely on extensive annotated data for training, which can be both costly and time-consuming to acquire. Moreover, these models often struggle to adapt to new or unseen relationships. In contrast, few-shot learning settings, which aim to reduce annotation requirements, may offer incomplete and biased supervision for understanding target relation semantics, leading to degraded and unstable performance. To provide the model with accurate and explicit descriptions of the relations types and meanwhile minimize the annotation requirements, we study the definition only zero-shot RE setting where only relation definitions expressed in natural language are used to train a RE model. Motivated by the strong synthetic data generation power of LLMs, we pr
    
[^7]: 是否可以稳健地编辑大型语言模型？

    Is it Possible to Edit Large Language Models Robustly?

    [https://arxiv.org/abs/2402.05827](https://arxiv.org/abs/2402.05827)

    本研究旨在了解大型语言模型的稳健编辑方法的优势和局限性，从而促进对交流型人工智能的稳健、现实应用。

    

    大型语言模型（LLM）在构建能模仿人类行为的交流型人工智能方面发挥了关键作用，但也面临着高效定制的挑战。为了解决这个挑战，最近的研究涉及到了模型编辑的领域，通过操纵语言模型的特定记忆并改变相关的语言生成来进行编辑。然而，模型编辑的稳健性仍然是一个悬而未决的问题。本研究旨在了解编辑方法的优势和局限性，从而促进对交流型人工智能的稳健、现实应用。具体而言，我们进行了广泛的分析以回答三个关键的研究问题。Q1：编辑后的LLM是否能在现实情境中一致地表现出类似于交流型人工智能的行为？Q2：改写提示在多大程度上导致LLM偏离编辑的知识记忆？Q3：哪些知识特征与编辑的性能和稳健性相关？我们的实验结果揭示了显著的差异。

    Large language models (LLMs) have played a pivotal role in building communicative AI to imitate human behaviors but face the challenge of efficient customization. To tackle this challenge, recent studies have delved into the realm of model editing, which manipulates specific memories of language models and changes the related language generation. However, the robustness of model editing remains an open question. This work seeks to understand the strengths and limitations of editing methods, thus facilitating robust, realistic applications of communicative AI. Concretely, we conduct extensive analysis to address the three key research questions. Q1: Can edited LLMs behave consistently resembling communicative AI in realistic situations? Q2: To what extent does the rephrasing of prompts lead LLMs to deviate from the edited knowledge memory? Q3: Which knowledge features are correlated with the performance and robustness of editing? Our experimental results uncover a substantial disparity 
    
[^8]: 通过修剪和低秩修改评估安全对齐的易碎性

    Assessing the Brittleness of Safety Alignment via Pruning and Low-Rank Modifications

    [https://arxiv.org/abs/2402.05162](https://arxiv.org/abs/2402.05162)

    本研究通过修剪和低秩修改，发现大型语言模型（LLMs）的安全机制固有易碎性，去除安全关键区域会损害安全性，但对效用影响不大，需要更强健的安全策略。

    

    大型语言模型（LLMs）在其安全机制方面表现出固有的易碎性，这可从它们易受越狱和即使是非恶意微调也易受影响来说明。本研究通过利用修剪和低秩修改探讨了安全对齐的易碎性。我们开发了方法，能够识别对于安全防护至关重要，且在神经元和秩级别上与效用相关的区域。令人惊讶的是，我们发现的孤立区域是稀疏的，约占参数级别的$3\%$和排名级别的$2.5\%$。去除这些区域会损害安全性，而对效用的影响不大，从而证实了该模型安全机制的固有易碎性。此外，我们还表明，即使限制对安全关键区域进行修改，LLMs仍然容易受到低成本的微调攻击。这些发现强调了在LLMs中更强大的安全策略的紧迫性需求。

    Large language models (LLMs) show inherent brittleness in their safety mechanisms, as evidenced by their susceptibility to jailbreaking and even non-malicious fine-tuning. This study explores this brittleness of safety alignment by leveraging pruning and low-rank modifications. We develop methods to identify critical regions that are vital for safety guardrails, and that are disentangled from utility-relevant regions at both the neuron and rank levels. Surprisingly, the isolated regions we find are sparse, comprising about $3\%$ at the parameter level and $2.5\%$ at the rank level. Removing these regions compromises safety without significantly impacting utility, corroborating the inherent brittleness of the model's safety mechanisms. Moreover, we show that LLMs remain vulnerable to low-cost fine-tuning attacks even when modifications to the safety-critical regions are restricted. These findings underscore the urgent need for more robust safety strategies in LLMs.
    
[^9]: 通过个性化参数高效调整实现大规模语言模型的民主化

    Democratizing Large Language Models via Personalized Parameter-Efficient Fine-tuning

    [https://arxiv.org/abs/2402.04401](https://arxiv.org/abs/2402.04401)

    这项研究通过个性化参数高效调整模块（PEFT）实现了大规模语言模型（LLM）的民主化，使用户能够拥有和使用他们自己的LLM，解决了传统方法中的定制能力和隐私问题。

    

    大规模语言模型（LLM）中的个性化越来越重要，旨在使LLM的交互、内容和推荐与个体用户偏好相一致。最近LLM个性化的进展聚焦于有效的提示设计，通过使用行为历史检索和文本概要等非参数化知识丰富用户查询。然而，由于缺乏模型所有权，这些方法受到了一定的限制，导致定制能力和隐私问题。此外，在复杂和动态用户数据的情况下，它们通常无法准确捕捉用户行为模式。为了解决这些缺点，我们引入了一种名为OPPU的方法，它采用个性化参数高效调整（PEFT）模块来存储用户特定的行为模式和偏好。通过插入用户的个人PEFT参数，他们可以拥有和使用他们的LLM。

    Personalization in large language models (LLMs) is increasingly important, aiming to align LLM's interactions, content, and recommendations with individual user preferences. Recent advances in LLM personalization have spotlighted effective prompt design, by enriching user queries with non-parametric knowledge through behavior history retrieval and textual profiles. However, these approaches were limited due to a lack of model ownership, resulting in constrained customization and privacy issues. Moreover, they often failed to accurately capture user behavior patterns, especially in cases where user data were complex and dynamic. To address these shortcomings, we introduce One PEFT Per User (OPPU), which employs personalized parameter-efficient fine-tuning (PEFT) modules, to store user-specific behavior patterns and preferences. By plugging in users' personal PEFT parameters, they can own and use their LLMs personally. OPPU integrates parametric user knowledge in the personal PEFT parame
    
[^10]: LLMRefine：通过细粒度可操作反馈精确定位和优化大型语言模型

    LLMRefine: Pinpointing and Refining Large Language Models via Fine-Grained Actionable Feedback

    [https://arxiv.org/abs/2311.09336](https://arxiv.org/abs/2311.09336)

    LLMRefine提出了一种细粒度反馈模型来指导大型语言模型定位缺陷并进行优化，在机器翻译、长篇问答和主题总结等任务中取得显著的改进。

    

    最近，大型语言模型（LLM）正在利用人类反馈来提高生成质量。然而，在推断过程中获取人类反馈成本高昂。在这项工作中，我们提出了LLMRefine，一种用于优化推理时间的方法，以改进LLM的输出。其核心思想是利用学习的细粒度反馈模型来准确定位缺陷，并引导LLM进行迭代优化。通过将原始LLM作为编辑建议，LLMRefine通过模拟退火搜索无缺陷文本，权衡探索和开发。我们在三个文本生成任务上进行实验，包括机器翻译，长篇问答（QA）和主题总结。LLMRefine在所有基线方法上一贯表现优异，在翻译任务上取得了高达1.7 MetricX点的改进，在ASQA上为8.1 ROUGE-L，在主题总结上为2.2 ROUGE-L。

    arXiv:2311.09336v2 Announce Type: replace  Abstract: Recent large language models (LLM) are leveraging human feedback to improve their generation quality. However, human feedback is costly to obtain, especially during inference. In this work, we propose LLMRefine, an inference time optimization method to refine LLM's output. The core idea is to use a learned fine-grained feedback model to pinpoint defects and guide LLM to refine them iteratively. Using original LLM as a proposal of edits, LLMRefine searches for defect-less text via simulated annealing, trading off the exploration and exploitation. We conduct experiments on three text generation tasks, including machine translation, long-form question answering (QA), and topical summarization. LLMRefine consistently outperforms all baseline approaches, achieving improvements up to 1.7 MetricX points on translation tasks, 8.1 ROUGE-L on ASQA, 2.2 ROUGE-L on topical summarization.
    
[^11]: 通过修剪大型语言模型调查幻觉在抽象摘要中的应用

    Investigating Hallucinations in Pruned Large Language Models for Abstractive Summarization. (arXiv:2311.09335v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2311.09335](http://arxiv.org/abs/2311.09335)

    本文通过广泛的实证研究发现，修剪后的大型语言模型在抽象摘要任务中产生幻觉的情况较原始模型要少，表现更可靠，具有更高的效率和稀疏推理能力。

    

    尽管生成型的大型语言模型在抽象摘要任务中表现出色，但它们面临两个重要挑战：模型庞大和易产生幻觉。幻觉是令人担忧的，因为它们降低了可靠性并引发安全问题。修剪是一种通过去除冗余权重来减小模型大小，实现更高效稀疏推理的技术。修剪后的模型在下游任务性能上与原始模型相当，因此在预算有限的情况下成为理想的替代选择。然而，修剪对语言模型在抽象摘要中产生幻觉的影响尚未被探索。本文通过对五个摘要数据集、两种最先进的修剪方法和五个经调试的语言模型进行了广泛的实证研究。令人惊讶的是，我们发现修剪后的语言模型产生幻觉的情况较原始模型要少。我们的分析表明，修剪后的模型更倾向于依赖指导信息。

    Despite the remarkable performance of generative large language models (LLMs) on abstractive summarization, they face two significant challenges: their considerable size and tendency to hallucinate. Hallucinations are concerning because they erode reliability and raise safety issues. Pruning is a technique that reduces model size by removing redundant weights, enabling more efficient sparse inference. Pruned models yield downstream task performance comparable to the original, making them ideal alternatives when operating on a limited budget. However, the effect that pruning has upon hallucinations in abstractive summarization with LLMs has yet to be explored. In this paper, we provide an extensive empirical study across five summarization datasets, two state-of-the-art pruning methods, and five instruction-tuned LLMs. Surprisingly, we find that hallucinations from pruned LLMs are less prevalent than the original models. Our analysis suggests that pruned models tend to depend more on th
    
[^12]: 大型语言模型可以成为良好的隐私保护学习者

    Large Language Models Can Be Good Privacy Protection Learners. (arXiv:2310.02469v1 [cs.CL])

    [http://arxiv.org/abs/2310.02469](http://arxiv.org/abs/2310.02469)

    本论文介绍了一种名为隐私保护语言模型（PPLM）的新范式，可以在保护数据隐私的同时有效注入领域特定知识。通过对模型设计的理论分析和不同技术的研究，我们验证了使用正向和负向示例进行指令微调的方法具有很大的潜力。

    

    大型语言模型（LLMs）的普及引发了人们对使用特定领域数据对其进行微调，创建专门的语言模型的兴趣。然而，这种特定领域的微调数据通常包含敏感的个人身份信息（PII）。在没有隐私保护的情况下直接微调 LLMs 会存在信息泄露的风险。为了解决这个挑战，我们引入了隐私保护语言模型（PPLM），这是一种在有效注入领域特定知识的同时保护数据隐私的新范式。我们的工作提供了模型设计的理论分析，并深入研究了各种技术，比如语料库策展、基于惩罚的非概然性训练损失以及基于指令的微调等等。广泛的实验在不同的数据集和场景中验证了我们的方法的有效性。特别是，使用正向和负向示例进行指令微调，显示出很有希望的方法。

    The proliferation of Large Language Models (LLMs) has driven considerable interest in fine-tuning them with domain-specific data to create specialized language models. Nevertheless, such domain-specific fine-tuning data often contains sensitive personally identifiable information (PII). Direct fine-tuning LLMs on this data without privacy protection poses a risk of leakage. To address this challenge, we introduce Privacy Protection Language Models (PPLM), a novel paradigm for fine-tuning LLMs that effectively injects domain-specific knowledge while safeguarding data privacy. Our work offers a theoretical analysis for model design and delves into various techniques such as corpus curation, penalty-based unlikelihood in training loss, and instruction-based tuning, etc. Extensive experiments across diverse datasets and scenarios demonstrate the effectiveness of our approaches. In particular, instruction tuning with both positive and negative examples, stands out as a promising method, eff
    
[^13]: 使用深度学习在阿尔及利亚方言中检测仇恨言论

    Hate speech detection in algerian dialect using deep learning. (arXiv:2309.11611v1 [cs.CL])

    [http://arxiv.org/abs/2309.11611](http://arxiv.org/abs/2309.11611)

    本研究提出了一个完整的方法，通过深度学习来检测在线阿尔及利亚信息中的仇恨言论。通过对阿尔及利亚社交网络上的语料库进行评估，我们获得了令人鼓舞的结果。

    

    随着社交网络上仇恨言论以不同的形式蔓延，如辱骂语言、网络欺凌和暴力等，人们在暴力方面经历了显著增加，使他们处于不适和威胁的境地。在过去几年中，人们已经投入大量的努力来克服这一现象，以检测不同结构语言（如英语、法语、阿拉伯语等）中的仇恨言论，并为阿拉伯方言（如突尼斯、埃及和海湾）进行了较少的研究。为了填补这一空白，我们在本文中提出了一个完整的方法，用于检测在线阿尔及利亚信息中的仇恨言论。我们评估了许多基于深度学习的架构，这些架构是从一些阿尔及利亚社交网络（Facebook、YouTube和Twitter）中创建的语料库上进行的。该语料库包含13.5K多篇阿拉伯语的阿尔及利亚方言文档，被标记为仇恨或非仇恨。我们获得了令人鼓舞的结果，显示出了该方法的有效性。

    With the proliferation of hate speech on social networks under different formats, such as abusive language, cyberbullying, and violence, etc., people have experienced a significant increase in violence, putting them in uncomfortable situations and threats. Plenty of efforts have been dedicated in the last few years to overcome this phenomenon to detect hate speech in different structured languages like English, French, Arabic, and others. However, a reduced number of works deal with Arabic dialects like Tunisian, Egyptian, and Gulf, mainly the Algerian ones. To fill in the gap, we propose in this work a complete approach for detecting hate speech on online Algerian messages. Many deep learning architectures have been evaluated on the corpus we created from some Algerian social networks (Facebook, YouTube, and Twitter). This corpus contains more than 13.5K documents in Algerian dialect written in Arabic, labeled as hateful or non-hateful. Promising results are obtained, which show the e
    
[^14]: 基于演示的学习方法用于少样本生物医学命名实体识别中的机器阅读理解

    Demonstration-based learning for few-shot biomedical named entity recognition under machine reading comprehension. (arXiv:2308.06454v1 [cs.CL])

    [http://arxiv.org/abs/2308.06454](http://arxiv.org/abs/2308.06454)

    本研究提出了一种基于演示的学习方法，通过将生物医学命名实体识别重新定义为机器阅读理解问题，来解决少样本学习场景下的生物医学实体识别问题。实验证明，在少样本学习中，该方法比其他先进方法平均提高了1.1%的F1分数。

    

    虽然深度学习技术在许多领域已经取得了显著的成就，但它们通常依赖大量手工标注的数据，并且在少样本场景下表现不佳。本研究的目标是设计一种能够改进模型在少样本学习场景下识别生物医学实体的能力的策略。通过将生物医学命名实体识别（BioNER）重新定义为机器阅读理解（MRC）问题，我们提出了一种基于演示的学习方法来解决少样本BioNER问题，该方法涉及构建适当的任务演示。在评估我们提出的方法时，我们使用了包括BC4CHEMD、BC5CDR-Chemical、BC5CDR-Disease、NCBI-Disease、BC2GM和JNLPBA在内的六个基准数据集，将所提出的方法与现有的先进方法进行了比较。我们通过报告25样本和50样本学习实验的F1分数来检查模型的效果。在25样本学习中，我们观察到平均F1分数提高了1.1%。

    Although deep learning techniques have shown significant achievements, they frequently depend on extensive amounts of hand-labeled data and tend to perform inadequately in few-shot scenarios. The objective of this study is to devise a strategy that can improve the model's capability to recognize biomedical entities in scenarios of few-shot learning. By redefining biomedical named entity recognition (BioNER) as a machine reading comprehension (MRC) problem, we propose a demonstration-based learning method to address few-shot BioNER, which involves constructing appropriate task demonstrations. In assessing our proposed method, we compared the proposed method with existing advanced methods using six benchmark datasets, including BC4CHEMD, BC5CDR-Chemical, BC5CDR-Disease, NCBI-Disease, BC2GM, and JNLPBA. We examined the models' efficacy by reporting F1 scores from both the 25-shot and 50-shot learning experiments. In 25-shot learning, we observed 1.1% improvements in the average F1 scores 
    

