# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [If CLIP Could Talk: Understanding Vision-Language Model Representations Through Their Preferred Concept Descriptions](https://arxiv.org/abs/2403.16442) | 通过新颖的Extract and Explore（EX2）方法，研究发现在视觉-语言模型（VLM）中，重要的特征描述包括非视觉属性，虚假描述影响VLM表示，不同的VLM优先考虑不同的内容。 |
| [^2] | [From Pixels to Insights: A Survey on Automatic Chart Understanding in the Era of Large Foundation Models](https://arxiv.org/abs/2403.12027) | 近年来，随着大型基础模型的兴起，自动图表理解取得了显著进展，本调查论文概述了在这些基础模型背景下图表理解领域的最新发展、挑战和未来方向 |
| [^3] | [SmallToLarge (S2L): Scalable Data Selection for Fine-tuning Large Language Models by Summarizing Training Trajectories of Small Models](https://arxiv.org/abs/2403.07384) | S2L提出了一种通过总结小模型的训练轨迹，来指导大型语言模型数据选择的方法，显著提高了数学问题解决中监督微调的数据效率，并在数据集性能上表现优异。 |
| [^4] | [Towards a Psychology of Machines: Large Language Models Predict Human Memory](https://arxiv.org/abs/2403.05152) | 这项研究探索了大型语言模型在预测基于语言的记忆任务中的表现，并通过其对模棱两可句子的处理能力增进了对人类认知机制的理解。 |
| [^5] | [PreAct: Predicting Future in ReAct Enhances Agent's Planning Ability](https://arxiv.org/abs/2402.11534) | PreAct是一个整合了预测、推理和行动的智能体框架，利用预测信息可以帮助智能体进行更多样化和策略性的推理，导致更有效的行动，提升任务完成效率。 |
| [^6] | [Universal Prompt Optimizer for Safe Text-to-Image Generation](https://arxiv.org/abs/2402.10882) | 提出了第一个通用提示优化器，用于在黑盒场景中安全生成文本到图像，通过构建毒素-清洁提示对数据集，设计奖励函数，并通过 Proximal Policy Optimization 训练优化器，成功降低各种 T2I 模型生成不安全内容的可能性。 |
| [^7] | [Network Formation and Dynamics Among Multi-LLMs](https://arxiv.org/abs/2402.10659) | 分析了多个LLM在社交网络中的行为，发现它们在给定网络结构并被询问形成网络偏好时表现出与人类社交动态一致的原则。 |
| [^8] | [Diffusion of Thoughts: Chain-of-Thought Reasoning in Diffusion Language Models](https://arxiv.org/abs/2402.07754) | 本文介绍了一种将扩散模型与思维链推理集成的方法，通过扩散传播推理步骤，提供了更大的灵活性和推理能力。实验证明了该方法在数学问题中的有效性，并展示了自我纠正能力和推理技术的潜力。 |
| [^9] | [CREMA: Multimodal Compositional Video Reasoning via Efficient Modular Adaptation and Fusion](https://arxiv.org/abs/2402.05889) | 该论文提出了一种名为CREMA的高效且模块化的模态融合框架，用于将任意新的模态注入视频推理。通过利用预训练模型增强多种信息模态，并引入查询转换器和融合模块，实现了灵活且有效的多模态组合推理。 |
| [^10] | [Agent-OM: Leveraging LLM Agents for Ontology Matching](https://arxiv.org/abs/2312.00326) | 本研究提出了Agent-OM，利用LLM代理为本体匹配系统引入了新的设计范式。 |
| [^11] | [ToolEyes: Fine-Grained Evaluation for Tool Learning Capabilities of Large Language Models in Real-world Scenarios.](http://arxiv.org/abs/2401.00741) | ToolEyes是一个专门用于评估大型语言模型在真实情景中的工具学习能力的细粒度系统，通过对七个真实情景的详细分析，评估了LLMs在工具学习的五个关键维度，并提供了一个拥有600种工具的工具库作为中介。 |

# 详细

[^1]: 如果CLIP能说话: 通过它们的首选概念描述理解视觉-语言模型的表示

    If CLIP Could Talk: Understanding Vision-Language Model Representations Through Their Preferred Concept Descriptions

    [https://arxiv.org/abs/2403.16442](https://arxiv.org/abs/2403.16442)

    通过新颖的Extract and Explore（EX2）方法，研究发现在视觉-语言模型（VLM）中，重要的特征描述包括非视觉属性，虚假描述影响VLM表示，不同的VLM优先考虑不同的内容。

    

    最近的研究常常假设视觉-语言模型（VLM）的表示是基于形状等视觉属性。然而，目前尚不清楚VLM在表示概念时在多大程度上将这些信息作为优先考虑对象。我们提出了一种新颖的方法，称为Extract and Explore（EX2），用于刻画VLM的重要文本特征。EX2使用强化学习将一个大型语言模型与VLM首选项对齐，并生成包含VLM重要特征的描述。然后，我们检查这些描述以确定对VLM表示有贡献的特征。我们发现，虽然提供了没有帮助信息的虚假描述（例如，单击放大概念的照片），但在VLM表示中起着重要作用。更重要的是，在信息丰富的描述中，VLM在表示视觉概念时显著依赖非视觉属性（如栖息地）。此外，我们的分析揭示了不同的VLM优先考虑不同的内容。

    arXiv:2403.16442v1 Announce Type: new  Abstract: Recent works often assume that Vision-Language Model (VLM) representations are based on visual attributes like shape. However, it is unclear to what extent VLMs prioritize this information to represent concepts. We propose Extract and Explore (EX2), a novel approach to characterize important textual features for VLMs. EX2 uses reinforcement learning to align a large language model with VLM preferences and generates descriptions that incorporate the important features for the VLM. Then, we inspect the descriptions to identify the features that contribute to VLM representations. We find that spurious descriptions have a major role in VLM representations despite providing no helpful information, e.g., Click to enlarge photo of CONCEPT. More importantly, among informative descriptions, VLMs rely significantly on non-visual attributes like habitat to represent visual concepts. Also, our analysis reveals that different VLMs prioritize differen
    
[^2]: 从像素到洞察: 在大型基础模型时代自动图表理解的调查

    From Pixels to Insights: A Survey on Automatic Chart Understanding in the Era of Large Foundation Models

    [https://arxiv.org/abs/2403.12027](https://arxiv.org/abs/2403.12027)

    近年来，随着大型基础模型的兴起，自动图表理解取得了显著进展，本调查论文概述了在这些基础模型背景下图表理解领域的最新发展、挑战和未来方向

    

    数据可视化以图表形式在数据分析中扮演着关键角色，提供关键洞察并帮助做出明智决策。随着近年大型基础模型的崛起，自动图表理解取得了显著进展。基础模型，如大型语言模型(LLMs)，已经在各种自然语言处理（NLP）任务中实现了革命，并越来越多地应用于图表理解任务。本调查论文全面介绍了最新进展、挑战和未来方向，探讨了这些基础模型背景下图表理解的内容。

    arXiv:2403.12027v1 Announce Type: cross  Abstract: Data visualization in the form of charts plays a pivotal role in data analysis, offering critical insights and aiding in informed decision-making. Automatic chart understanding has witnessed significant advancements with the rise of large foundation models in recent years. Foundation models, such as large language models (LLMs), have revolutionized various natural language processing (NLP) tasks and are increasingly being applied to chart understanding tasks. This survey paper provides a comprehensive overview of the recent developments, challenges, and future directions in chart understanding within the context of these foundation models. The paper begins by defining chart understanding, outlining problem formulations, and discussing fundamental building blocks crucial for studying chart understanding tasks. In the section on tasks and datasets, we explore various tasks within chart understanding and discuss their evaluation metrics a
    
[^3]: SmallToLarge (S2L): 通过总结小模型的训练轨迹，为大型语言模型的微调提供可伸缩的数据选择

    SmallToLarge (S2L): Scalable Data Selection for Fine-tuning Large Language Models by Summarizing Training Trajectories of Small Models

    [https://arxiv.org/abs/2403.07384](https://arxiv.org/abs/2403.07384)

    S2L提出了一种通过总结小模型的训练轨迹，来指导大型语言模型数据选择的方法，显著提高了数学问题解决中监督微调的数据效率，并在数据集性能上表现优异。

    

    尽管数据选择在大型语言模型（LLMs）的预训练和指导微调阶段非常有效，但在专业领域的监督微调（SFT）中改善数据效率面临着重大挑战，原因是微调数据的复杂性。为弥合这一差距，我们引入了一种有效且可伸缩的数据选择方法S2L（SmallToLarge），它利用小模型的训练轨迹来指导更大模型的数据选择。通过大量实验，我们证明了S2L显著提高了数学问题解决的SFT数据效率，将训练数据缩减到原始MathInstruct数据集（Yue等人，2023）的仅11%，以达到全数据集的性能，并在6个领域内外评估数据集中平均优于最先进的数据选择算法4.7%。值得注意的是，仅选择50K数据进行SFT，S2L实现...

    arXiv:2403.07384v1 Announce Type: cross  Abstract: Despite the effectiveness of data selection for large language models (LLMs) during pretraining and instruction fine-tuning phases, improving data efficiency in supervised fine-tuning (SFT) for specialized domains poses significant challenges due to the complexity of fine-tuning data. To bridge this gap, we introduce an effective and scalable data selection method for SFT, SmallToLarge (S2L), which leverages training trajectories from small models to guide the data selection for larger models. We demonstrate through extensive experiments that S2L significantly improves data efficiency in SFT for mathematical problem-solving, reducing the training data to just 11% of the original MathInstruct dataset (Yue et al., 2023) to match full dataset performance while outperforming state-of-the-art data selection algorithms by an average of 4.7% across 6 in- and out-domain evaluation datasets. Remarkably, selecting only 50K data for SFT, S2L achi
    
[^4]: 朝向机器心理学：大型语言模型预测人类记忆

    Towards a Psychology of Machines: Large Language Models Predict Human Memory

    [https://arxiv.org/abs/2403.05152](https://arxiv.org/abs/2403.05152)

    这项研究探索了大型语言模型在预测基于语言的记忆任务中的表现，并通过其对模棱两可句子的处理能力增进了对人类认知机制的理解。

    

    大型语言模型（LLMs）在各种任务中展示出了非凡的能力，尽管缺乏人类认知基础。这引发了一个问题：除了简单模仿人类语言模式，这些模型能否提供关于人类认知机制的洞见？本研究探讨了ChatGPT在预测基于语言的记忆任务中人类表现的能力。基于文本理解理论，我们假设识别模棱两可的句子（例如，“因为比尔喝酒，所以酒从未留在房子里”）在前面提供与上下文相关信息的情况下会得到促进。参与者，无论是人类还是ChatGPT，都被呈现成对的句子。第二个句子总是一个旨在固有地模棱两可的花园路径句，而第一个句子则提供了合适的（例如，“比尔患有慢性酒精中毒”）或不合适的上下文（例如，“比尔喜欢打高尔夫”）。

    arXiv:2403.05152v1 Announce Type: cross  Abstract: Large language models (LLMs) are demonstrating remarkable capabilities across various tasks despite lacking a foundation in human cognition. This raises the question: can these models, beyond simply mimicking human language patterns, offer insights into the mechanisms underlying human cognition? This study explores the ability of ChatGPT to predict human performance in a language-based memory task. Building upon theories of text comprehension, we hypothesize that recognizing ambiguous sentences (e.g., "Because Bill drinks wine is never kept in the house") is facilitated by preceding them with contextually relevant information. Participants, both human and ChatGPT, were presented with pairs of sentences. The second sentence was always a garden-path sentence designed to be inherently ambiguous, while the first sentence either provided a fitting (e.g., "Bill has chronic alcoholism") or an unfitting context (e.g., "Bill likes to play golf"
    
[^5]: PreAct: 在 ReAct 中预测未来增强智能体的规划能力

    PreAct: Predicting Future in ReAct Enhances Agent's Planning Ability

    [https://arxiv.org/abs/2402.11534](https://arxiv.org/abs/2402.11534)

    PreAct是一个整合了预测、推理和行动的智能体框架，利用预测信息可以帮助智能体进行更多样化和策略性的推理，导致更有效的行动，提升任务完成效率。

    

    处理预测与实际结果之间的差异常常有助于个体拓展思维过程并进行反思，从而促进朝正确方向推理。本文介绍了一种名为 PreAct 的智能体框架，它将预测、推理和行动整合在一起。利用预测提供的信息，基于大语言模型（LLM）的智能体能够提供更多样化和策略导向的推理，进而导致更有效的行动，帮助智能体完成复杂任务。我们的实验表明，PreAct 在完成复杂任务方面优于 ReAct 方法，并且当与反思方法结合时，PreAct 的效果可以得到提升。我们对模型提供不同数量的历史预测，并发现历史预测对LLM规划有持续积极影响。

    arXiv:2402.11534v1 Announce Type: cross  Abstract: Addressing the discrepancies between predictions and actual outcomes often aids individuals in expanding their thought processes and engaging in reflection, thereby facilitating reasoning in the correct direction. In this paper, we introduce $\textbf{PreAct}$, an agent framework that integrates $\textbf{pre}$diction with $\textbf{rea}$soning and $\textbf{act}$ion. Leveraging the information provided by predictions, a large language model (LLM) based agent can offer more diversified and strategically oriented reasoning, which in turn leads to more effective actions that help the agent complete complex tasks. Our experiments demonstrate that PreAct outperforms the ReAct approach in accomplishing complex tasks and that PreAct can be co-enhanced when combined with Reflexion methods. We prompt the model with different numbers of historical predictions and find that historical predictions have a sustained positive effect on LLM planning. The
    
[^6]: 通用提示优化器用于安全文本到图像生成

    Universal Prompt Optimizer for Safe Text-to-Image Generation

    [https://arxiv.org/abs/2402.10882](https://arxiv.org/abs/2402.10882)

    提出了第一个通用提示优化器，用于在黑盒场景中安全生成文本到图像，通过构建毒素-清洁提示对数据集，设计奖励函数，并通过 Proximal Policy Optimization 训练优化器，成功降低各种 T2I 模型生成不安全内容的可能性。

    

    文本到图像（T2I）模型在根据文字提示生成图像方面表现出色。然而，这些模型容易受到不安全输入的影响，从而生成不安全内容，如色情、骚扰和非法活动图像。基于图像检查器、模型微调和嵌入式阻止的现有研究在真实世界应用中不可行。因此，我们提出了第一个用于黑盒场景中安全 T2I 生成的通用提示优化器。

    arXiv:2402.10882v1 Announce Type: cross  Abstract: Text-to-Image (T2I) models have shown great performance in generating images based on textual prompts. However, these models are vulnerable to unsafe input to generate unsafe content like sexual, harassment and illegal-activity images. Existing studies based on image checker, model fine-tuning and embedding blocking are impractical in real-world applications. Hence, \textit{we propose the first universal prompt optimizer for safe T2I generation in black-box scenario}. We first construct a dataset consisting of toxic-clean prompt pairs by GPT-3.5 Turbo. To guide the optimizer to have the ability of converting toxic prompt to clean prompt while preserving semantic information, we design a novel reward function measuring toxicity and text alignment of generated images and train the optimizer through Proximal Policy Optimization. Experiments show that our approach can effectively reduce the likelihood of various T2I models in generating in
    
[^7]: 多个LLM之间的网络形成与动态

    Network Formation and Dynamics Among Multi-LLMs

    [https://arxiv.org/abs/2402.10659](https://arxiv.org/abs/2402.10659)

    分析了多个LLM在社交网络中的行为，发现它们在给定网络结构并被询问形成网络偏好时表现出与人类社交动态一致的原则。

    

    社交网络影响行为、偏好和关系，在人类社会中对信息和规范的传播起着至关重要的作用。随着大型语言模型（LLMs）越来越多地融入社交和专业环境中，理解它们在社交网络和互动背景下的行为变得至关重要。我们的研究分析了标准网络结构和现实世界网络的行为，以确定多个LLMs的动态是否与人类社交动态一致。我们探讨了各种社交网络原则，包括微观层面的概念，如偏爱附着、三角闭合和同似性，以及宏观层面的概念，如社区结构和小世界现象。我们的研究发现表明，当向LLMs提供网络结构并询问它们对网络形成的偏好时，它们表现出所有这些原则。

    arXiv:2402.10659v1 Announce Type: cross  Abstract: Social networks influence behaviors, preferences, and relationships and play a crucial role in the dissemination of information and norms within human societies. As large language models (LLMs) increasingly integrate into social and professional environments, understanding their behavior within the context of social networks and interactions becomes essential. Our study analyzes the behaviors of standard network structures and real-world networks to determine whether the dynamics of multiple LLMs align with human social dynamics. We explore various social network principles, including micro-level concepts such as preferential attachment, triadic closure, and homophily, as well as macro-level concepts like community structure and the small-world phenomenon. Our findings suggest that LLMs demonstrate all these principles when they are provided with network structures and asked about their preferences regarding network formation. Furtherm
    
[^8]: 思想传播：扩散语言模型中的思维链推理

    Diffusion of Thoughts: Chain-of-Thought Reasoning in Diffusion Language Models

    [https://arxiv.org/abs/2402.07754](https://arxiv.org/abs/2402.07754)

    本文介绍了一种将扩散模型与思维链推理集成的方法，通过扩散传播推理步骤，提供了更大的灵活性和推理能力。实验证明了该方法在数学问题中的有效性，并展示了自我纠正能力和推理技术的潜力。

    

    扩散模型在文本处理中引起了关注，相对传统的自回归模型具有许多潜在优势。本文探讨了将扩散模型与思维链（CoT）集成的方法，CoT是一种在自回归语言模型中改进推理能力的成熟技术。我们提出了思维扩散（DoT）模型，允许推理步骤通过扩散过程在时间上传播。与传统的自回归语言模型逐个token从左到右做出决策的方式相比，DoT在计算和推理性能之间具有更大的灵活性。我们的实验证明了DoT在多位数乘法和小学数学问题中的有效性。此外，DoT展示了有希望的自我纠正能力，并从现有的增强推理技术（如自一致解码）中受益。我们的发现有助于理解和发展推理能力。

    Diffusion models have gained attention in text processing, offering many potential advantages over traditional autoregressive models. This work explores the integration of diffusion models and Chain-of-Thought (CoT), a well-established technique to improve the reasoning ability in autoregressive language models. We propose Diffusion-of-Thought (DoT), allowing reasoning steps to diffuse over time through the diffusion process. In contrast to traditional autoregressive language models that make decisions in a left-to-right, token-by-token manner, DoT offers more flexibility in the trade-off between computation and reasoning performance. Our experimental results demonstrate the effectiveness of DoT in multi-digit multiplication and grade school math problems. Additionally, DoT showcases promising self-correction abilities and benefits from existing reasoning-enhancing techniques like self-consistency decoding. Our findings contribute to the understanding and development of reasoning capab
    
[^9]: CREMA: 通过有效的模块化适应和融合进行多模态组合视频推理

    CREMA: Multimodal Compositional Video Reasoning via Efficient Modular Adaptation and Fusion

    [https://arxiv.org/abs/2402.05889](https://arxiv.org/abs/2402.05889)

    该论文提出了一种名为CREMA的高效且模块化的模态融合框架，用于将任意新的模态注入视频推理。通过利用预训练模型增强多种信息模态，并引入查询转换器和融合模块，实现了灵活且有效的多模态组合推理。

    

    尽管在多模态组合推理方法方面取得了令人瞩目的进展，但由于处理固定模态输入并更新许多模型参数，仍然存在灵活性和效率方面的限制。本文解决了这些关键挑战，提出了CREMA，一种用于将任何新的模态注入视频推理的高效且模块化的模态融合框架。我们首先利用现有的预训练模型从给定的视频中增强多种信息模态（如光流、3D点云、音频），而无需额外的人工注释。接下来，我们引入了一个查询转换器，该转换器与每个可以访问的模态相关联，并具有多个参数高效的模块。它将多种模态特征投影到LLM令牌嵌入空间，使模型能够整合不同的数据类型以进行响应生成。此外，我们提出了一个融合模块，用于压缩多模态查询，在LLM中保持计算效率的同时进行融合组合。

    Despite impressive advancements in multimodal compositional reasoning approaches, they are still limited in their flexibility and efficiency by processing fixed modality inputs while updating a lot of model parameters. This paper tackles these critical challenges and proposes CREMA, an efficient and modular modality-fusion framework for injecting any new modality into video reasoning. We first augment multiple informative modalities (such as optical flow, 3D point cloud, audio) from given videos without extra human annotation by leveraging existing pre-trained models. Next, we introduce a query transformer with multiple parameter-efficient modules associated with each accessible modality. It projects diverse modality features to the LLM token embedding space, allowing the model to integrate different data types for response generation. Furthermore, we propose a fusion module designed to compress multimodal queries, maintaining computational efficiency in the LLM while combining additio
    
[^10]: Agent-OM：利用LLM代理进行本体匹配

    Agent-OM: Leveraging LLM Agents for Ontology Matching

    [https://arxiv.org/abs/2312.00326](https://arxiv.org/abs/2312.00326)

    本研究提出了Agent-OM，利用LLM代理为本体匹配系统引入了新的设计范式。

    

    本体匹配（OM）能够实现不同本体之间的语义互操作性，通过对齐相关实体来解决其概念异构性。本研究引入了一种新颖的基于代理的LLM设计范式，命名为Agent-OM，包括两个用于检索和匹配的同体代理以及一组基于提示的简单OM工具。

    arXiv:2312.00326v2 Announce Type: replace  Abstract: Ontology matching (OM) enables semantic interoperability between different ontologies and resolves their conceptual heterogeneity by aligning related entities. OM systems currently have two prevailing design paradigms: conventional knowledge-based expert systems and newer machine learning-based predictive systems. While large language models (LLMs) and LLM agents have revolutionised data engineering and have been applied creatively in many domains, their potential for OM remains underexplored. This study introduces a novel agent-powered LLM-based design paradigm for OM systems. With consideration of several specific challenges in leveraging LLM agents for OM, we propose a generic framework, namely Agent-OM, consisting of two Siamese agents for retrieval and matching, with a set of simple prompt-based OM tools. Our framework is implemented in a proof-of-concept system. Evaluations of three Ontology Alignment Evaluation Initiative (OAE
    
[^11]: ToolEyes：大型语言模型在实际情景中的工具学习能力的细粒度评估

    ToolEyes: Fine-Grained Evaluation for Tool Learning Capabilities of Large Language Models in Real-world Scenarios. (arXiv:2401.00741v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2401.00741](http://arxiv.org/abs/2401.00741)

    ToolEyes是一个专门用于评估大型语言模型在真实情景中的工具学习能力的细粒度系统，通过对七个真实情景的详细分析，评估了LLMs在工具学习的五个关键维度，并提供了一个拥有600种工具的工具库作为中介。

    

    现有的工具学习评估主要集中于验证大型语言模型（LLMs）选择的工具与期望结果的一致性。然而，这些方法依赖于一组有限的情景，在这些情景中答案可以事先确定，与真实需求背道而驰。此外，仅关注结果忽视了LLMs有效利用工具所需的复杂能力。为解决这个问题，我们提出了ToolEyes，这是一个特别针对LLMs工具学习能力在真实情景中评估的细粒度系统。该系统详细分析了七个真实情景，分析了对LLMs在工具学习中至关重要的五个维度：格式对齐，意图理解，行为规划，工具选择和答案组织。此外，ToolEyes还包含一个拥有约600种工具的工具库，作为LLMs与物理世界之间的中介。在涉及三个类别的十个LLMs的评估中，ToolEyes取得了如下的创新与贡献。

    Existing evaluations of tool learning primarily focus on validating the alignment of selected tools for large language models (LLMs) with expected outcomes. However, these approaches rely on a limited set of scenarios where answers can be pre-determined, diverging from genuine needs. Furthermore, a sole emphasis on outcomes disregards the intricate capabilities essential for LLMs to effectively utilize tools. To tackle this issue, we propose ToolEyes, a fine-grained system tailored for the evaluation of the LLMs' tool learning capabilities in authentic scenarios. The system meticulously examines seven real-world scenarios, analyzing five dimensions crucial to LLMs in tool learning: format alignment, intent comprehension, behavior planning, tool selection, and answer organization. Additionally, ToolEyes incorporates a tool library boasting approximately 600 tools, serving as an intermediary between LLMs and the physical world. Evaluations involving ten LLMs across three categories revea
    

