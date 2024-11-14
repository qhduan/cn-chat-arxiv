# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Quantifying and Mitigating Unimodal Biases in Multimodal Large Language Models: A Causal Perspective](https://arxiv.org/abs/2403.18346) | 提出了一个因果框架用于解释多模态大型语言模型在视觉问答问题中的偏差，并引入了一个新的挑战性数据集MORE，同时提出两种减轻单模态偏差的策略。 |
| [^2] | [LLMs Instruct LLMs:An Extraction and Editing Method](https://arxiv.org/abs/2403.15736) | 提出了一种顺序融合方法，将复杂环境中的知识融入LLMs中，用于更新大型语言模型。 |
| [^3] | [GlossLM: Multilingual Pretraining for Low-Resource Interlinear Glossing](https://arxiv.org/abs/2403.06399) | 该论文提出了GlossLM模型，通过利用跨语言转移和大规模多语言预训练，实现了低资源语言文字间注释的有效生成。 |
| [^4] | [Attacking LLM Watermarks by Exploiting Their Strengths](https://arxiv.org/abs/2402.16187) | 现有的LLM水印系统虽然具有质量保留、鲁棒性和公开检测API等优点，但也因此容易受到各种攻击，研究者提出了一套实用指南以缓解这些攻击。 |
| [^5] | [Uncertainty of Thoughts: Uncertainty-Aware Planning Enhances Information Seeking in Large Language Models](https://arxiv.org/abs/2402.03271) | 通过引入不确定性感知规划（UoT）算法，我们实现了增强大型语言模型的主动寻求信息的能力，通过模拟未来场景、基于不确定性的奖励机制和奖励传播方案，优化问题提问方式。 |
| [^6] | [Are Large Language Models Table-based Fact-Checkers?](https://arxiv.org/abs/2402.02549) | 本研究初步探讨了大型语言模型在基于表格的事实检查方面的潜力。实验结果表明，通过提示工程，大型语言模型在零样本和少样本的情况下可以实现可接受的表现。 |
| [^7] | [Toxicity Detection is NOT all you Need: Measuring the Gaps to Supporting Volunteer Content Moderators](https://arxiv.org/abs/2311.07879) | 本研究揭示了人工智能模型在识别有毒、冒犯和令人讨厌的内容方面的进展，并探讨了这些改进是否真正满足了志愿内容管理员在工作中的需求。 |

# 详细

[^1]: 在多模态大型语言模型中量化和减轻单模态偏差：因果关系视角

    Quantifying and Mitigating Unimodal Biases in Multimodal Large Language Models: A Causal Perspective

    [https://arxiv.org/abs/2403.18346](https://arxiv.org/abs/2403.18346)

    提出了一个因果框架用于解释多模态大型语言模型在视觉问答问题中的偏差，并引入了一个新的挑战性数据集MORE，同时提出两种减轻单模态偏差的策略。

    

    大型语言模型（LLMs）的最新进展促进了多模态LLMs（MLLMs）的发展。尽管它们具有令人印象深刻的能力，但MLLMs通常过度依赖单模态偏差（例如语言偏差和视觉偏差），导致在复杂多模态任务中给出不正确答案。为了调查这个问题，我们提出了一个因果框架来解释视觉问答（VQA）问题中的偏差。在我们的框架内，我们设计了一个因果图来阐明MLLMs对VQA问题的预测，并通过深入的因果分析评估偏差的因果效果。受因果图的启发，我们引入了一个新颖的MORE数据集，包含12,000个VQA实例。该数据集旨在挑战MLLMs的能力，需要多跳推理和克服单模态偏差。此外，我们提出了两种策略来减轻单模态偏差并增强MLLMs的推理能力。

    arXiv:2403.18346v1 Announce Type: new  Abstract: Recent advancements in Large Language Models (LLMs) have facilitated the development of Multimodal LLMs (MLLMs). Despite their impressive capabilities, MLLMs often suffer from an over-reliance on unimodal biases (e.g., language bias and vision bias), leading to incorrect answers in complex multimodal tasks. To investigate this issue, we propose a causal framework to interpret the biases in Visual Question Answering (VQA) problems. Within our framework, we devise a causal graph to elucidate the predictions of MLLMs on VQA problems, and assess the causal effect of biases through an in-depth causal analysis. Motivated by the causal graph, we introduce a novel MORE dataset, consisting of 12,000 VQA instances. This dataset is designed to challenge MLLMs' abilities, necessitating multi-hop reasoning and the surmounting of unimodal biases. Furthermore, we propose two strategies to mitigate unimodal biases and enhance MLLMs' reasoning capabiliti
    
[^2]: LLMs指导LLMs：一种提取和编辑方法

    LLMs Instruct LLMs:An Extraction and Editing Method

    [https://arxiv.org/abs/2403.15736](https://arxiv.org/abs/2403.15736)

    提出了一种顺序融合方法，将复杂环境中的知识融入LLMs中，用于更新大型语言模型。

    

    arXiv:2403.15736v1 公告类型：新 兴趣点在于无需从头开始训练即可更新大型语言模型（LLMs），但是这也带来了一些挑战。尤其是对于需要用有限样本进行复杂推理的情况来说，我们称之为适用于LLMs的贫乏约束复杂推理（PCRA-LLM）的情况。传统方法如低秩适应（LoRA）和检索增强生成（RAG）对这一关键问题是不足够的，尤其在我们探索特定医学背景时尤为明显，这体现了PCRA-LLM的独特需求。为了解决这个问题，我们提出了一种顺序融合方法，将复杂环境中的知识融入LLMs中。该方法采用两阶段框架：首先，利用通用LLMs构建知识图谱（KGs）来从复杂文本中提取知识；随后，通过知识编辑来更新领域LLMs。根据我们的方法，领域

    arXiv:2403.15736v1 Announce Type: new  Abstract: The interest in updating Large Language Models (LLMs) without retraining from scratch is substantial, yet it comes with some challenges.This is especially true for situations demanding complex reasoning with limited samples, a scenario we refer to as the Paucity-Constrained Complex Reasoning Adaptation for LLMs (PCRA-LLM).Traditional methods like Low-Rank Adaptation (LoRA) and Retrieval-Augmented Generation (RAG) are inadequate for this critical issue, particularly evident in our exploration of a specific medical context that epitomize the PCRA-LLM's distinct needs.To address the issue, we propose a Sequential Fusion method to incorporate knowledge from complex context into LLMs. This method employs a two-stage framework: initially, it leverages general LLMs to construct knowledge graphs (KGs) for extracting knowledge from complex texts; subsequently, it updates the domain LLMs through knowledge edit. According to our method, the domain 
    
[^3]: GlossLM: 低资源语言文字间注释的多语言预训练

    GlossLM: Multilingual Pretraining for Low-Resource Interlinear Glossing

    [https://arxiv.org/abs/2403.06399](https://arxiv.org/abs/2403.06399)

    该论文提出了GlossLM模型，通过利用跨语言转移和大规模多语言预训练，实现了低资源语言文字间注释的有效生成。

    

    语言文献学的一个关键方面是以形式如文字间注释文本（IGT）的方式创建带注释的文本，IGT以逐词素的格式捕捉了精细的形态句法分析。先前的研究已探索了自动生成IGT的方法，以减少语言分析的时间成本。然而，许多语言（尤其是需要保护的语言）缺乏足够的IGT数据来训练有效的模型，跨语言转移被提出作为克服这一局限的方法。我们编制了来自各种来源的最大已有IGT数据语料库，涵盖了来自1.8k种语言的超过45万个例子，以便进行跨语言转移和IGT生成方面的研究。然后，我们在部分语料库上对一个大型多语言模型进行预训练，并进一步对特定语言进行微调。我们的模型在分割数据和大型单语数据方面与最先进的方法相竞争。

    arXiv:2403.06399v1 Announce Type: new  Abstract: A key aspect of language documentation is the creation of annotated text in a format such as interlinear glossed text (IGT), which captures fine-grained morphosyntactic analyses in a morpheme-by-morpheme format. Prior work has explored methods to automatically generate IGT in order to reduce the time cost of language analysis. However, many languages (particularly those requiring preservation) lack sufficient IGT data to train effective models, and crosslingual transfer has been proposed as a method to overcome this limitation.   We compile the largest existing corpus of IGT data from a variety of sources, covering over 450k examples across 1.8k languages, to enable research on crosslingual transfer and IGT generation. Then, we pretrain a large multilingual model on a portion of this corpus, and further finetune it to specific languages. Our model is competitive with state-of-the-art methods for segmented data and large monolingual datas
    
[^4]: 利用其优势攻击LLM水印

    Attacking LLM Watermarks by Exploiting Their Strengths

    [https://arxiv.org/abs/2402.16187](https://arxiv.org/abs/2402.16187)

    现有的LLM水印系统虽然具有质量保留、鲁棒性和公开检测API等优点，但也因此容易受到各种攻击，研究者提出了一套实用指南以缓解这些攻击。

    

    生成模型的进展使得人工智能生成的文本、代码和图片能够在许多应用中模仿人类生成的内容。水印技术旨在将信息嵌入模型的输出中以验证其来源，对于减少对这些人工智能生成内容的滥用非常有用。然而，现有的水印方案仍然令人意外地容易受到攻击。具体而言，我们展示了现有的LLM水印系统共享的可取特性，例如质量保留、鲁棒性和公开检测API，反过来却使这些系统容易遭受各种攻击。我们在常见水印设计选择方面严格研究潜在攻击，并提出了缓解攻击的最佳实践和防御措施——建立了一套嵌入和检测LLM水印的实用指南。

    arXiv:2402.16187v1 Announce Type: cross  Abstract: Advances in generative models have made it possible for AI-generated text, code, and images to mirror human-generated content in many applications. Watermarking, a technique that aims to embed information in the output of a model to verify its source, is useful for mitigating misuse of such AI-generated content. However, existing watermarking schemes remain surprisingly susceptible to attack. In particular, we show that desirable properties shared by existing LLM watermarking systems such as quality preservation, robustness, and public detection APIs can in turn make these systems vulnerable to various attacks. We rigorously study potential attacks in terms of common watermark design choices, and propose best practices and defenses for mitigation -- establishing a set of practical guidelines for embedding and detection of LLM watermarks.
    
[^5]: 想法的不确定性：不确定性感知规划增强大型语言模型的信息搜索能力

    Uncertainty of Thoughts: Uncertainty-Aware Planning Enhances Information Seeking in Large Language Models

    [https://arxiv.org/abs/2402.03271](https://arxiv.org/abs/2402.03271)

    通过引入不确定性感知规划（UoT）算法，我们实现了增强大型语言模型的主动寻求信息的能力，通过模拟未来场景、基于不确定性的奖励机制和奖励传播方案，优化问题提问方式。

    

    在面对不确定性时，寻求信息的能力至关重要。在许多实际应用中，比如医学诊断和故障排除，解决任务所需的信息不是初始给定的，而需要通过询问后续问题来主动寻求（例如，医生向患者询问症状的更多细节）。在这项工作中，我们引入了思想的不确定性（UoT），一种算法将大型语言模型的能力与主动提问信息的能力相结合。UoT结合了1）不确定性感知仿真方法，使模型能够模拟可能的未来场景，并估计其发生的可能性；2）基于不确定性的奖励机制，激励模型寻求信息；3）奖励传播方案，以最大化预期奖励的方式选择最佳的问题提问方式。在医学诊断、故障排除和'20的实验中。

    In the face of uncertainty, the ability to seek information is of fundamental importance. In many practical applications, such as medical diagnosis and troubleshooting, the information needed to solve the task is not initially given, and has to be actively sought by asking follow-up questions (for example, a doctor asking a patient for more details about their symptoms). In this work, we introduce Uncertainty of Thoughts (UoT), an algorithm to augment large language models with the ability to actively seek information by asking effective questions. UoT combines 1) an uncertainty-aware simulation approach which enables the model to simulate possible future scenarios and how likely they are to occur, 2) uncertainty-based rewards motivated by information gain which incentivizes the model to seek information, and 3) a reward propagation scheme to select the optimal question to ask in a way that maximizes the expected reward. In experiments on medical diagnosis, troubleshooting and the '20 
    
[^6]: 大型语言模型是否适合基于表格的事实检查？

    Are Large Language Models Table-based Fact-Checkers?

    [https://arxiv.org/abs/2402.02549](https://arxiv.org/abs/2402.02549)

    本研究初步探讨了大型语言模型在基于表格的事实检查方面的潜力。实验结果表明，通过提示工程，大型语言模型在零样本和少样本的情况下可以实现可接受的表现。

    

    基于表格的事实验证（TFV）旨在提取语句和结构化表格之间的蕴涵关系。现有基于小规模模型的TFV方法在标注数据不足和零样本能力薄弱方面存在问题。近年来，大型语言模型（LLMs）在研究领域引起了广泛关注。它们在几个自然语言处理任务上展示了强大的零样本和上下文学习能力，但它们在TFV领域的潜力还不清楚。在本文中，我们进行了关于LLMs是否适合作为基于表格的事实检查器的初步研究。具体来说，我们设计了多样化的提示语来探索上下文学习如何帮助LLMs在TFV方面，即零样本和少样本TFV能力。此外，我们精心设计和构建了TFV指导以研究LLMs的指导调整带来的性能改进。实验结果表明，通过提示工程，LLMs在零样本和少样本TFV方面可以达到可接受的结果，而指导调整则进一步提升了性能。

    Table-based Fact Verification (TFV) aims to extract the entailment relation between statements and structured tables. Existing TFV methods based on small-scaled models suffer from insufficient labeled data and weak zero-shot ability. Recently, the appearance of Large Language Models (LLMs) has gained lots of attraction in research fields. They have shown powerful zero-shot and in-context learning abilities on several NLP tasks, but their potential on TFV is still unknown. In this work, we implement a preliminary study about whether LLMs are table-based fact-checkers. In detail, we design diverse prompts to explore how the in-context learning can help LLMs in TFV, i.e., zero-shot and few-shot TFV capability. Besides, we carefully design and construct TFV instructions to study the performance gain brought by the instruction tuning of LLMs. Experimental results demonstrate that LLMs can achieve acceptable results on zero-shot and few-shot TFV with prompt engineering, while instruction-tun
    
[^7]: 毒性检测并不是你所需要的全部：弥合支持志愿内容管理员的差距

    Toxicity Detection is NOT all you Need: Measuring the Gaps to Supporting Volunteer Content Moderators

    [https://arxiv.org/abs/2311.07879](https://arxiv.org/abs/2311.07879)

    本研究揭示了人工智能模型在识别有毒、冒犯和令人讨厌的内容方面的进展，并探讨了这些改进是否真正满足了志愿内容管理员在工作中的需求。

    

    人工智能模型在识别有毒、冒犯和令人讨厌的内容方面取得了长足的进展，旨在减轻管理员的工作负担。然而，目前尚不清楚这些任务的改进是否真正满足了管理员在工作中的需求。本文揭示了过去研究努力致力于为内容管理的各个方面提供自动化支持与志愿内容管理员的需求之间存在的差距，尤其是在识别违反各种管理规则方面。为此，我们在Hugging Face上对模型进行了调查，以揭示涵盖三个示范论坛的各种管理规则和指南的模型的可用性。我们进一步对最先进的LLM进行了测试，评估这些模型在标记某个特定论坛的平台规则违规方面的表现。最后，我们进行了用户调查研究。

    arXiv:2311.07879v2 Announce Type: replace-cross  Abstract: Extensive efforts in automated approaches for content moderation have been focused on developing models to identify toxic, offensive, and hateful content with the aim of lightening the load for moderators. Yet, it remains uncertain whether improvements on those tasks have truly addressed moderators' needs in accomplishing their work. In this paper, we surface gaps between past research efforts that have aimed to provide automation for aspects of content moderation and the needs of volunteer content moderators, regarding identifying violations of various moderation rules. To do so, we conduct a model review on Hugging Face to reveal the availability of models to cover various moderation rules and guidelines from three exemplar forums. We further put state-of-the-art LLMs to the test, evaluating how well these models perform in flagging violations of platform rules from one particular forum. Finally, we conduct a user survey stud
    

