# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [AutoRE: Document-Level Relation Extraction with Large Language Models](https://arxiv.org/abs/2403.14888) | AutoRE 是一种端到端的文档级关系抽取模型，采用了一种名为RHF的新颖关系抽取范式，可有效处理分布在文档中的多个关系和三元组事实。 |
| [^2] | [Exploring Continual Learning of Compositional Generalization in NLI](https://arxiv.org/abs/2403.04400) | 本文介绍了连续组合泛化挑战，探讨了模型如何持续获取原始推理任务的知识并进行组合推理，研究了不断学习对NLI中组合泛化的影响，并提出了解决方案。 |
| [^3] | [Grounding Language Models for Visual Entity Recognition](https://arxiv.org/abs/2402.18695) | 通过AutoVER模型，我们提出了一种在视觉实体识别中应用自回归模型的方法，通过检索增强的约束生成，成功区分巨大标签空间中相似的实体，并在Oven-Wiki基准测试上取得显著进展。 |
| [^4] | [M3-VRD: Multimodal Multi-task Multi-teacher Visually-Rich Form Document Understanding](https://arxiv.org/abs/2402.17983) | 这个模型是一个多模态、多任务、多教师的联合细粒度知识蒸馏模型，通过微妙协作令牌和实体表示，处理复杂的表单文档，引入新的损失函数改进知识蒸馏过程，在处理视觉复杂表单文档的结构和内容上表现出色。 |
| [^5] | [Model Composition for Multimodal Large Language Models](https://arxiv.org/abs/2402.12750) | 通过模型组合现有的多模态大型语言模型，提出了一种新范式，有效地保留了每个原始模型的模态理解能力，并引入了一种用于解决合并参数干扰和不匹配问题的方法。 |
| [^6] | [Measuring and Controlling Persona Drift in Language Model Dialogs](https://arxiv.org/abs/2402.10962) | 提出了一种量化基准来测量语言模型对话中的“人设”漂移，并提出了一种称为split-softmax的轻量级方法来对抗注意力衰减和“人设”漂移 |
| [^7] | [SafeDecoding: Defending against Jailbreak Attacks via Safety-Aware Decoding](https://arxiv.org/abs/2402.08983) | 本文提出了一种名为SafeDecoding的解决方案，通过安全感知解码策略来防御大型语言模型（LLMs）的越狱攻击。该策略可以生成对用户查询有益且无害的响应，有效缓解了LLMs安全性威胁。 |
| [^8] | [AIR-Bench: Benchmarking Large Audio-Language Models via Generative Comprehension](https://arxiv.org/abs/2402.07729) | 这个论文介绍了AIR-Bench，这是第一个用于评估大型音频语言模型理解和生成音频信号的基准。 |
| [^9] | [Debating with More Persuasive LLMs Leads to More Truthful Answers](https://arxiv.org/abs/2402.06782) | 本文研究了更弱的语言模型是否能评估更强的模型的正确性。研究发现，通过进行辩论，非专家模型和人类回答问题的准确性都有所提高。 |
| [^10] | [Tag-LLM: Repurposing General-Purpose LLMs for Specialized Domains](https://arxiv.org/abs/2402.05140) | 本文介绍了一种将通用的LLMs应用于专业领域的方法，通过学习自定义的输入标签来对LLMs进行条件约束。通过明确将任务领域与任务功能分离，这种方法能够改善在专业领域中的任务求解能力。 |
| [^11] | [Large Language Model for Table Processing: A Survey](https://arxiv.org/abs/2402.05121) | 该调查综述了大型语言模型在表格处理中的应用，包括传统的表格问题回答和事实验证，以及新兴的表格操作和高级表格数据分析。还讨论了LLMs的最新范例，特别关注了指导调整、提示和基于代理的方法。 |
| [^12] | [Multi-Scale Contrastive Knowledge Co-Distillation for Event Temporal Relation Extraction](https://arxiv.org/abs/2209.00568) | 提出了MulCo：多尺度对比知识共同蒸馏用于全面提高所有类型时间数据集性能 |
| [^13] | [RCAgent: Cloud Root Cause Analysis by Autonomous Agents with Tool-Augmented Large Language Models.](http://arxiv.org/abs/2310.16340) | RCAgent是一个工具增强的LLM自主代理框架，用于云根本原因分析，能够实现自由格式的数据收集和全面的分析，并在各个方面优于当前方法。 |
| [^14] | [The Janus Interface: How Fine-Tuning in Large Language Models Amplifies the Privacy Risks.](http://arxiv.org/abs/2310.15469) | 《Janus接口：大型语言模型微调如何放大隐私风险》研究了大型语言模型的微调对个人信息泄露的风险，发现了一种新的LLM利用途径。 |

# 详细

[^1]: AutoRE：使用大型语言模型进行文档级关系抽取

    AutoRE: Document-Level Relation Extraction with Large Language Models

    [https://arxiv.org/abs/2403.14888](https://arxiv.org/abs/2403.14888)

    AutoRE 是一种端到端的文档级关系抽取模型，采用了一种名为RHF的新颖关系抽取范式，可有效处理分布在文档中的多个关系和三元组事实。

    

    大型语言模型(LLMs)展示了在理解和生成文本方面的异常能力，这激励着许多研究人员利用它们进行信息抽取(IE)任务，包括关系抽取(RE)。然而，大多数现有方法主要设计用于句子级关系抽取(SentRE)任务，这通常涵盖了单个句子内的一组关系和三元组事实。此外，一些方法采用将关系作为候选选择集成到提示模板中的方式，导致在处理分布在给定文档中的多个关系和三元组事实时效率低下，性能亚优，并在处理文档级关系抽取(DocRE)任务时面临独特挑战。为了克服这些限制，我们介绍了AutoRE，这是一个端到端的DocRE模型，采用了一种名为RHF(Re

    arXiv:2403.14888v1 Announce Type: cross  Abstract: Large Language Models (LLMs) have demonstrated exceptional abilities in comprehending and generating text, motivating numerous researchers to utilize them for Information Extraction (IE) purposes, including Relation Extraction (RE). Nonetheless, most existing methods are predominantly designed for Sentence-level Relation Extraction (SentRE) tasks, which typically encompass a restricted set of relations and triplet facts within a single sentence. Furthermore, certain approaches resort to treating relations as candidate choices integrated into prompt templates, leading to inefficient processing and suboptimal performance when tackling Document-Level Relation Extraction (DocRE) tasks, which entail handling multiple relations and triplet facts distributed across a given document, posing distinct challenges. To overcome these limitations, we introduce AutoRE, an end-to-end DocRE model that adopts a novel RE extraction paradigm named RHF (Re
    
[^2]: 在自然语言推理中探索组合泛化的不间断学习

    Exploring Continual Learning of Compositional Generalization in NLI

    [https://arxiv.org/abs/2403.04400](https://arxiv.org/abs/2403.04400)

    本文介绍了连续组合泛化挑战，探讨了模型如何持续获取原始推理任务的知识并进行组合推理，研究了不断学习对NLI中组合泛化的影响，并提出了解决方案。

    

    组合自然语言推理已被用来评估神经模型执行NLI的真实能力。然而，当前的评估假设模型事先完全访问所有原始推理，与人类不断获得推理知识的方式相反。在本文中，我们介绍了推理中的连续组合泛化（C2Gen NLI）挑战，其中模型持续获取构成原始推理任务的知识作为组合推理的基础。我们研究了不断学习如何影响NLI中的组合泛化，通过为组合NLI推理任务设计了一个不断学习设置。我们的实验表明，模型在不断学习的情况下无法组合泛化。为了解决这个问题，我们首先对各种不断学习算法进行基准测试并验证它们的功效。然后，我们进一步分析了C2Gen，重点关注如何排序

    arXiv:2403.04400v1 Announce Type: new  Abstract: Compositional Natural Language Inference has been explored to assess the true abilities of neural models to perform NLI. Yet, current evaluations assume models to have full access to all primitive inferences in advance, in contrast to humans that continuously acquire inference knowledge. In this paper, we introduce the Continual Compositional Generalization in Inference (C2Gen NLI) challenge, where a model continuously acquires knowledge of constituting primitive inference tasks as a basis for compositional inferences. We explore how continual learning affects compositional generalization in NLI, by designing a continual learning setup for compositional NLI inference tasks. Our experiments demonstrate that models fail to compositionally generalize in a continual scenario. To address this problem, we first benchmark various continual learning algorithms and verify their efficacy. We then further analyze C2Gen, focusing on how to order pri
    
[^3]: 将语言模型应用在视觉实体识别上

    Grounding Language Models for Visual Entity Recognition

    [https://arxiv.org/abs/2402.18695](https://arxiv.org/abs/2402.18695)

    通过AutoVER模型，我们提出了一种在视觉实体识别中应用自回归模型的方法，通过检索增强的约束生成，成功区分巨大标签空间中相似的实体，并在Oven-Wiki基准测试上取得显著进展。

    

    我们引入AutoVER，一种用于视觉实体识别的自回归模型。我们的模型通过使用检索增强的约束生成，扩展了自回归多模式大型语言模型。它在处理跨领域实体时减轻了低性能，在需要视觉推理的查询中表现出色。我们的方法通过在硬负对上进行对比训练，并在序列-序列目标中并行进行训练，学习在巨大的标签空间中区分相似的实体。在推断过程中，一系列检索的候选答案明确指导语言生成，通过消除无效的解码路径。所提出的方法在最近提出的Oven-Wiki基准测试的不同数据集拆分中实现了显著的改进。在已知实体拆分上的准确率从32.7%提高到61.5%。该方法还通过大幅度提升在未知和查询拆分上的性能，表现出卓越的表现。

    arXiv:2402.18695v1 Announce Type: cross  Abstract: We introduce AutoVER, an Autoregressive model for Visual Entity Recognition. Our model extends an autoregressive Multi-modal Large Language Model by employing retrieval augmented constrained generation. It mitigates low performance on out-of-domain entities while excelling in queries that require visually-situated reasoning. Our method learns to distinguish similar entities within a vast label space by contrastively training on hard negative pairs in parallel with a sequence-to-sequence objective without an external retriever. During inference, a list of retrieved candidate answers explicitly guides language generation by removing invalid decoding paths. The proposed method achieves significant improvements across different dataset splits in the recently proposed Oven-Wiki benchmark. Accuracy on the Entity seen split rises from 32.7% to 61.5%. It also demonstrates superior performance on the unseen and query splits by a substantial dou
    
[^4]: M3-VRD: 多模态多任务多教师视觉丰富表单文档理解

    M3-VRD: Multimodal Multi-task Multi-teacher Visually-Rich Form Document Understanding

    [https://arxiv.org/abs/2402.17983](https://arxiv.org/abs/2402.17983)

    这个模型是一个多模态、多任务、多教师的联合细粒度知识蒸馏模型，通过微妙协作令牌和实体表示，处理复杂的表单文档，引入新的损失函数改进知识蒸馏过程，在处理视觉复杂表单文档的结构和内容上表现出色。

    

    这篇论文提出了一个突破性的多模态、多任务、多教师联合细粒度知识蒸馏模型，用于视觉丰富的表单文档理解。该模型旨在通过促进令牌和实体表示之间的微妙相关性来利用细粒度和粗粒度级别的见解，解决表单文档固有的复杂性。此外，我们引入了新的跨细粒度和跨粗粒度损失函数，以进一步改进多教师知识蒸馏传递过程，呈现分布差距和对表单文档的统一理解。通过在公开可用的表单文档理解数据集上进行全面评估，我们提出的模型始终表现出色地优于现有基线，展示了其在处理复杂视觉表单文档的复杂结构和内容方面的功效。

    arXiv:2402.17983v1 Announce Type: new  Abstract: This paper presents a groundbreaking multimodal, multi-task, multi-teacher joint-grained knowledge distillation model for visually-rich form document understanding. The model is designed to leverage insights from both fine-grained and coarse-grained levels by facilitating a nuanced correlation between token and entity representations, addressing the complexities inherent in form documents. Additionally, we introduce new inter-grained and cross-grained loss functions to further refine diverse multi-teacher knowledge distillation transfer process, presenting distribution gaps and a harmonised understanding of form documents. Through a comprehensive evaluation across publicly available form document understanding datasets, our proposed model consistently outperforms existing baselines, showcasing its efficacy in handling the intricate structures and content of visually complex form documents.
    
[^5]: 多模态大型语言模型的模型组合

    Model Composition for Multimodal Large Language Models

    [https://arxiv.org/abs/2402.12750](https://arxiv.org/abs/2402.12750)

    通过模型组合现有的多模态大型语言模型，提出了一种新范式，有效地保留了每个原始模型的模态理解能力，并引入了一种用于解决合并参数干扰和不匹配问题的方法。

    

    近期对多模态大型语言模型（MLLMs）的发展显示出了快速进展，朝着创建能够理解各种模态输入的多功能MLLMs的目标迈进。然而，现有方法通常依赖于与配对的多模态指令数据进行联合训练，这对资源要求高且难以扩展到新的模态。在本文中，我们提出了一种通过现有MLLMs的模型组合来创建一个新模型的新范式，该新模型保留了每个原始模型的模态理解能力。我们的基本实现NaiveMC通过重用模态编码器和合并LLM参数展示了这一范式的有效性。此外，我们引入了DAMC来解决在合并过程中的参数干扰和不匹配问题，从而提升了模型的性能。为促进该领域的研究，我们提出了MCUB，一个用于评估MLLMs理解能力的基准测试。

    arXiv:2402.12750v1 Announce Type: cross  Abstract: Recent developments in Multimodal Large Language Models (MLLMs) have shown rapid progress, moving towards the goal of creating versatile MLLMs that understand inputs from various modalities. However, existing methods typically rely on joint training with paired multimodal instruction data, which is resource-intensive and challenging to extend to new modalities. In this paper, we propose a new paradigm through the model composition of existing MLLMs to create a new model that retains the modal understanding capabilities of each original model. Our basic implementation, NaiveMC, demonstrates the effectiveness of this paradigm by reusing modality encoders and merging LLM parameters. Furthermore, we introduce DAMC to address parameter interference and mismatch issues during the merging process, thereby enhancing the model performance. To facilitate research in this area, we propose MCUB, a benchmark for assessing ability of MLLMs to unders
    
[^6]: 在语言模型对话中测量和控制“人设”漂移

    Measuring and Controlling Persona Drift in Language Model Dialogs

    [https://arxiv.org/abs/2402.10962](https://arxiv.org/abs/2402.10962)

    提出了一种量化基准来测量语言模型对话中的“人设”漂移，并提出了一种称为split-softmax的轻量级方法来对抗注意力衰减和“人设”漂移

    

    提示是定制语言模型聊天机器人的标准工具，使其能够承担特定的“人设”。在使用提示时的一个隐含假设是，它们将是稳定的，因此聊天机器人将在整个对话过程中继续根据规定的“人设”生成文本。我们提出了一个量化基准来测试这一假设，通过两个个性化聊天机器人之间的自我对话来评估“人设”的稳定性。我们对流行模型如LLaMA2-chat-70B进行测试，发现在八轮对话中存在显著的“人设”漂移。对这一现象的实证和理论分析表明，由于长对话中的注意力衰减，变压器注意力机制起到了一定作用。为了对抗注意力衰减和“人设”漂移，我们提出了一种称为split-softmax的轻量级方法，与两个强基线方法相比表现优异。

    arXiv:2402.10962v1 Announce Type: cross  Abstract: Prompting is a standard tool for customizing language-model chatbots, enabling them to take on a specific "persona". An implicit assumption in the use of prompts is that they will be stable, so the chatbot will continue to generate text according to the stipulated persona for the duration of a conversation. We propose a quantitative benchmark to test this assumption, evaluating persona stability via self-chats between two personalized chatbots. Testing popular models like LLaMA2-chat-70B, we reveal a significant persona drift within eight rounds of conversations. An empirical and theoretical analysis of this phenomenon suggests the transformer attention mechanism plays a role, due to attention decay over long exchanges. To combat attention decay and persona drift, we propose a lightweight method called split-softmax, which compares favorably against two strong baselines.
    
[^7]: SafeDecoding: 通过安全感知解码防御越狱攻击

    SafeDecoding: Defending against Jailbreak Attacks via Safety-Aware Decoding

    [https://arxiv.org/abs/2402.08983](https://arxiv.org/abs/2402.08983)

    本文提出了一种名为SafeDecoding的解决方案，通过安全感知解码策略来防御大型语言模型（LLMs）的越狱攻击。该策略可以生成对用户查询有益且无害的响应，有效缓解了LLMs安全性威胁。

    

    随着大型语言模型（LLMs）越来越多地应用于代码生成和聊天机器人辅助等现实应用中，人们为了使LLM的行为与人类价值观保持一致，包括安全性在内做出了大量努力。越狱攻击旨在引发LLM的非预期和不安全行为，仍然是LLM安全性的重要威胁。本文旨在通过引入SafeDecoding来防御LLM的越狱攻击，这是一种安全感知的解码策略，用于生成对用户查询有益且无害的响应。我们在开发SafeDecoding时的洞察力基于观察到，即使代表有害内容的标记的概率超过代表无害响应的标记的概率，安全免责声明仍然出现在按概率降序排序的标记中的前几个。这使我们能够通过识别安全免责声明并增强其良性影响力来减轻越狱攻击。

    arXiv:2402.08983v1 Announce Type: cross Abstract: As large language models (LLMs) become increasingly integrated into real-world applications such as code generation and chatbot assistance, extensive efforts have been made to align LLM behavior with human values, including safety. Jailbreak attacks, aiming to provoke unintended and unsafe behaviors from LLMs, remain a significant/leading LLM safety threat. In this paper, we aim to defend LLMs against jailbreak attacks by introducing SafeDecoding, a safety-aware decoding strategy for LLMs to generate helpful and harmless responses to user queries. Our insight in developing SafeDecoding is based on the observation that, even though probabilities of tokens representing harmful contents outweigh those representing harmless responses, safety disclaimers still appear among the top tokens after sorting tokens by probability in descending order. This allows us to mitigate jailbreak attacks by identifying safety disclaimers and amplifying their
    
[^8]: AIR-Bench: 通过生成性理解评估大型音频语言模型的基准

    AIR-Bench: Benchmarking Large Audio-Language Models via Generative Comprehension

    [https://arxiv.org/abs/2402.07729](https://arxiv.org/abs/2402.07729)

    这个论文介绍了AIR-Bench，这是第一个用于评估大型音频语言模型理解和生成音频信号的基准。

    

    最近，指导性的音频语言模型因其对人与音频的互动能力而受到广泛关注。然而，缺乏能够评估以音频为中心的互动能力的基准已经阻碍了该领域的进展。以往的模型主要关注评估不同的基本任务，如自动语音识别（ASR），缺乏对围绕音频的开放式生成能力的评估。因此，追踪大型音频语言模型（LALMs）领域的进展并为未来的改进提供指导是具有挑战性的。在本文中，我们介绍了AIR-Bench（音频指导基准），这是第一个旨在评估LALMs理解各种类型音频信号（包括人类语音、自然声音和音乐）以及与人以文本形式进行交互能力的基准。AIR-Bench包含两个维度：基础和生成理解。

    Recently, instruction-following audio-language models have received broad attention for human-audio interaction. However, the absence of benchmarks capable of evaluating audio-centric interaction capabilities has impeded advancements in this field. Previous models primarily focus on assessing different fundamental tasks, such as Automatic Speech Recognition (ASR), and lack an assessment of the open-ended generative capabilities centered around audio. Thus, it is challenging to track the progression in the Large Audio-Language Models (LALMs) domain and to provide guidance for future improvement. In this paper, we introduce AIR-Bench (\textbf{A}udio \textbf{I}nst\textbf{R}uction \textbf{Bench}mark), the first benchmark designed to evaluate the ability of LALMs to understand various types of audio signals (including human speech, natural sounds, and music), and furthermore, to interact with humans in the textual format. AIR-Bench encompasses two dimensions: \textit{foundation} and \textit
    
[^9]: 与更有说服力的LLMs辩论会导致更真实的回答

    Debating with More Persuasive LLMs Leads to More Truthful Answers

    [https://arxiv.org/abs/2402.06782](https://arxiv.org/abs/2402.06782)

    本文研究了更弱的语言模型是否能评估更强的模型的正确性。研究发现，通过进行辩论，非专家模型和人类回答问题的准确性都有所提高。

    

    与所需行为一致的大型语言模型（LLM）的常见方法主要依赖于人工标注的数据。然而，随着模型变得越来越复杂，它们将超过人类专业知识，人类评估的角色将演变为非专家监督专家。在此之前，我们问：更弱的模型能评估更强的模型的正确性吗？我们在类似的环境中调查了这个问题，其中更强的模型（专家）拥有回答问题所需的信息，而更弱的模型（非专家）缺乏这些信息。我们评估的方法是\textit{辩论}，其中两个LLM专家分别支持不同的答案，一个非专家选择答案。我们发现辩论 consistently帮助非专家模型和人类回答问题，分别达到76%和88%的准确性（朴素基准分别为48%和60%）。此外，以无监督方式优化专家辩论者的说服力会提高非专家的能力。

    Common methods for aligning large language models (LLMs) with desired behaviour heavily rely on human-labelled data. However, as models grow increasingly sophisticated, they will surpass human expertise, and the role of human evaluation will evolve into non-experts overseeing experts. In anticipation of this, we ask: can weaker models assess the correctness of stronger models? We investigate this question in an analogous setting, where stronger models (experts) possess the necessary information to answer questions and weaker models (non-experts) lack this information. The method we evaluate is \textit{debate}, where two LLM experts each argue for a different answer, and a non-expert selects the answer. We find that debate consistently helps both non-expert models and humans answer questions, achieving 76\% and 88\% accuracy respectively (naive baselines obtain 48\% and 60\%). Furthermore, optimising expert debaters for persuasiveness in an unsupervised manner improves non-expert abilit
    
[^10]: Tag-LLM: 将通用的LLM应用于专业领域的再利用

    Tag-LLM: Repurposing General-Purpose LLMs for Specialized Domains

    [https://arxiv.org/abs/2402.05140](https://arxiv.org/abs/2402.05140)

    本文介绍了一种将通用的LLMs应用于专业领域的方法，通过学习自定义的输入标签来对LLMs进行条件约束。通过明确将任务领域与任务功能分离，这种方法能够改善在专业领域中的任务求解能力。

    

    大型语言模型（LLMs）在理解和生成自然语言方面表现出了非凡的能力。然而，在专门领域中，如物理学和生物医学科学这样的预训练语料库中未充分涵盖的领域，它们的能力下降。本文探讨了如何将通用LLMs重新用于专业领域的有效任务解决方案。我们介绍了一种新颖的、与模型无关的框架，用于学习自定义的输入标签，这些标签被参数化为连续向量并附加到LLMs的嵌入层，以对LLMs进行条件约束。我们设计了两种类型的输入标签：领域标签用于限定专业表示（例如化学式）并提供领域相关的上下文；功能标签用于表示特定的功能（例如预测分子性质）并压缩功能解决指令。我们使用辅助数据和领域知识开发了一个包括三个阶段的学习这些标签的协议。通过明确将任务领域与任务功能分离，我们的方法能够改善在专业领域中的任务求解能力。

    Large Language Models (LLMs) have demonstrated remarkable proficiency in understanding and generating natural language. However, their capabilities wane in highly specialized domains underrepresented in the pretraining corpus, such as physical and biomedical sciences. This work explores how to repurpose general LLMs into effective task solvers for specialized domains. We introduce a novel, model-agnostic framework for learning custom input tags, which are parameterized as continuous vectors appended to the LLM's embedding layer, to condition the LLM. We design two types of input tags: domain tags are used to delimit specialized representations (e.g., chemical formulas) and provide domain-relevant context; function tags are used to represent specific functions (e.g., predicting molecular properties) and compress function-solving instructions. We develop a three-stage protocol to learn these tags using auxiliary data and domain knowledge. By explicitly disentangling task domains from tas
    
[^11]: 大型语言模型在表格处理中的应用：一项调查

    Large Language Model for Table Processing: A Survey

    [https://arxiv.org/abs/2402.05121](https://arxiv.org/abs/2402.05121)

    该调查综述了大型语言模型在表格处理中的应用，包括传统的表格问题回答和事实验证，以及新兴的表格操作和高级表格数据分析。还讨论了LLMs的最新范例，特别关注了指导调整、提示和基于代理的方法。

    

    表格通常是二维结构化的，用于存储大量数据，在数据库查询、电子表格计算和从网络表格生成报告等日常活动中起着重要作用。利用大型语言模型（LLMs）自动化这些以表格为中心的任务可以带来重大的公众利益，引起了学术界和工业界的兴趣。该调查对表格任务进行了广泛的概述，不仅涵盖传统领域如表格问题回答（Table QA）和事实验证，还包括最近强调的方面，如表格操作和高级表格数据分析。此外，它还超越了早期的预训练和微调小型语言模型的策略，包括LLM使用中的最新范例。重点是LLMs领域内的指导调整、提示和基于代理的方法。最后，我们重点介绍了几个挑战，涵盖私有部署、高效推断和 LLMS 发展等方面。

    Tables, typically two-dimensional and structured to store large amounts of data, are essential in daily activities like database queries, spreadsheet calculations, and generating reports from web tables. Automating these table-centric tasks with Large Language Models (LLMs) offers significant public benefits, garnering interest from academia and industry. This survey provides an extensive overview of table tasks, encompassing not only the traditional areas like table question answering (Table QA) and fact verification, but also newly emphasized aspects such as table manipulation and advanced table data analysis. Additionally, it goes beyond the early strategies of pre-training and fine-tuning small language models, to include recent paradigms in LLM usage. The focus here is particularly on instruction-tuning, prompting, and agent-based approaches within the realm of LLMs. Finally, we highlight several challenges, ranging from private deployment and efficient inference to the developmen
    
[^12]: 多尺度对比知识共同蒸馏用于事件时间关系抽取

    Multi-Scale Contrastive Knowledge Co-Distillation for Event Temporal Relation Extraction

    [https://arxiv.org/abs/2209.00568](https://arxiv.org/abs/2209.00568)

    提出了MulCo：多尺度对比知识共同蒸馏用于全面提高所有类型时间数据集性能

    

    事件时间关系抽取（ETRE）是一个关键但具有挑战性的问题。事件对位于不同距离的话语中，我们称之为接近性带。关于位于更远（即“长”）或更近（即“短”）接近性带的事件对的时间顺序传达方式不同。目前ETRE模型往往在位于短或长接近性带的事件上表现良好，但不能同时表现良好。然而，现实世界中的自然文本包含所有类型的时间事件对。在本文中，我们提出了MulCo：多尺度对比知识共同蒸馏，这是一种融合方法，可以跨多个事件对接近性带共享知识，以提高对所有类型时间数据集的性能。我们的实验结果表明MulCo成功地整合了跨短和长接近性带的与时间推理相关的语言线索。

    arXiv:2209.00568v2 Announce Type: replace-cross  Abstract: Event Temporal Relation Extraction (ETRE) is a crucial yet challenging problem. Event pairs are situated within a discourse at different distances, which we refer to as proximity bands. The temporal ordering communicated about event pairs situated at more remote (i.e., ``long'') or less remote (i.e., ``short'') proximity bands is encoded differently. SOTA ETRE models have tended to perform well on events situated at either short or long proximity bands, but not both. Yet, real-world, natural texts contain all types of temporal event-pairs. In this paper, we present MulCo: Multi-Scale Contrastive Knowledge Co-Distillation, a fusion approach that shares knowledge across multiple event pair proximity bands in order to improve performance on all types of temporal datasets. Our experimental results show that MulCo successfully integrates linguistic cues pertaining to temporal reasoning across both short and long proximity bands and 
    
[^13]: RCAgent：基于自主代理和增强的大型语言模型的云根本原因分析

    RCAgent: Cloud Root Cause Analysis by Autonomous Agents with Tool-Augmented Large Language Models. (arXiv:2310.16340v1 [cs.SE])

    [http://arxiv.org/abs/2310.16340](http://arxiv.org/abs/2310.16340)

    RCAgent是一个工具增强的LLM自主代理框架，用于云根本原因分析，能够实现自由格式的数据收集和全面的分析，并在各个方面优于当前方法。

    

    最近，云根本原因分析中的大型语言模型（LLM）应用受到了积极的关注。然而，当前方法仍然依赖于手动工作流设置，并没有充分发挥LLMs的决策和环境交互能力。我们提出了RCAgent，这是一个实用和注重隐私的工具增强LLM自主代理框架，用于实际的工业RCA使用。RCAgent在内部部署的模型上运行，而不是GPT系列，能够进行自由格式的数据收集和全面的分析，并结合各种增强功能，包括独特的行动轨迹自一致性和一套用于上下文管理、稳定化和导入领域知识的方法。我们的实验证明RCAgent在RCA的各个方面（预测根本原因、解决方案、证据和责任）以及当前规则未涵盖的任务上都明显优于ReAct，得到了自动化和人工验证的确认。

    Large language model (LLM) applications in cloud root cause analysis (RCA) have been actively explored recently. However, current methods are still reliant on manual workflow settings and do not unleash LLMs' decision-making and environment interaction capabilities. We present RCAgent, a tool-augmented LLM autonomous agent framework for practical and privacy-aware industrial RCA usage. Running on an internally deployed model rather than GPT families, RCAgent is capable of free-form data collection and comprehensive analysis with tools. Our framework combines a variety of enhancements, including a unique Self-Consistency for action trajectories, and a suite of methods for context management, stabilization, and importing domain knowledge. Our experiments show RCAgent's evident and consistent superiority over ReAct across all aspects of RCA -- predicting root causes, solutions, evidence, and responsibilities -- and tasks covered or uncovered by current rules, as validated by both automate
    
[^14]: 《Janus接口：大型语言模型微调如何放大隐私风险》

    The Janus Interface: How Fine-Tuning in Large Language Models Amplifies the Privacy Risks. (arXiv:2310.15469v1 [cs.CR])

    [http://arxiv.org/abs/2310.15469](http://arxiv.org/abs/2310.15469)

    《Janus接口：大型语言模型微调如何放大隐私风险》研究了大型语言模型的微调对个人信息泄露的风险，发现了一种新的LLM利用途径。

    

    2018年后的时代标志着大型语言模型（LLM）的出现，OpenAI的ChatGPT等创新展示了惊人的语言能力。随着行业在增加模型参数并利用大量的人类语言数据方面的努力，安全和隐私挑战也出现了。其中最重要的是在基于网络的数据获取过程中，可能会意外积累个人可识别信息（PII），从而导致意外的PII泄露风险。虽然像RLHF和灾难性遗忘这样的策略已被用来控制隐私侵权的风险，但LLM的最新进展（以OpenAI的GPT-3.5的微调界面为代表）重新引发了关注。有人可能会问：LLM的微调是否会导致训练数据集中嵌入的个人信息泄漏？本文报道了首次尝试寻求答案的努力，重点是我们发现了一种新的LLM利用途径。

    The era post-2018 marked the advent of Large Language Models (LLMs), with innovations such as OpenAI's ChatGPT showcasing prodigious linguistic prowess. As the industry galloped toward augmenting model parameters and capitalizing on vast swaths of human language data, security and privacy challenges also emerged. Foremost among these is the potential inadvertent accrual of Personal Identifiable Information (PII) during web-based data acquisition, posing risks of unintended PII disclosure. While strategies like RLHF during training and Catastrophic Forgetting have been marshaled to control the risk of privacy infringements, recent advancements in LLMs, epitomized by OpenAI's fine-tuning interface for GPT-3.5, have reignited concerns. One may ask: can the fine-tuning of LLMs precipitate the leakage of personal information embedded within training datasets? This paper reports the first endeavor to seek the answer to the question, particularly our discovery of a new LLM exploitation avenue
    

