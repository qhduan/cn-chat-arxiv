# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Jamba: A Hybrid Transformer-Mamba Language Model](https://arxiv.org/abs/2403.19887) | Jamba是一个基于混合Transformer-Mamba架构的语言模型，在单个80GB GPU上实现了强大的性能，对标准语言模型基准和长上下文评估具有state-of-the-art的表现。 |
| [^2] | [Evaluating Large Language Models with Runtime Behavior of Program Execution](https://arxiv.org/abs/2403.16437) | 本文提出了一个名为REval的框架，用于评估代码LLMs的代码推理能力以及与程序执行的一致性。 |
| [^3] | [Call Me When Necessary: LLMs can Efficiently and Faithfully Reason over Structured Environments](https://arxiv.org/abs/2403.08593) | LLMs借助Reasoning-Path-Editing (Readi)框架，可以在结构化环境中高效且忠实地推理，显著提升了多个KGQA和TableQA数据集上的表现。 |
| [^4] | [Fine-tuning Large Language Models with Sequential Instructions](https://arxiv.org/abs/2403.07794) | 通过顺序指令微调，研究提出了一种简单且有效的策略，可以使大型语言模型具备执行多个顺序指令的能力，优于传统指令微调模型。 |
| [^5] | [GraphWiz: An Instruction-Following Language Model for Graph Problems](https://arxiv.org/abs/2402.16029) | GraphWiz是一个开源语言模型，通过引入指令调优数据集和直接偏好优化框架，能够高效解决各种图问题类型，平均准确率达到65%，超过了GPT-4的43.8%。 |
| [^6] | [ProSparse: Introducing and Enhancing Intrinsic Activation Sparsity within Large Language Models](https://arxiv.org/abs/2402.13516) | 本文介绍了一种名为"ProSparse"的有效稀疏化方法，以推动大型语言模型实现更高的激活稀疏性而不降低模型性能 |
| [^7] | [Smaug: Fixing Failure Modes of Preference Optimisation with DPO-Positive](https://arxiv.org/abs/2402.13228) | 在这项工作中，我们提出了一种新的损失函数和训练过程DPO-Positive（DPOP），以避免直接偏好优化（DPO）中潜在的失败模式，并发现DPOP明显优于DPO。 |
| [^8] | [MRKE: The Multi-hop Reasoning Evaluation of LLMs by Knowledge Edition](https://arxiv.org/abs/2402.11924) | 通过编辑HotpotQA数据集中的新知识，我们引入了一个LLM MHQA评估基准，同时注释和评估了推理链，揭示了当前MHQA基准存在数据污染的潜在风险。 |
| [^9] | [Decomposition for Enhancing Attention: Improving LLM-based Text-to-SQL through Workflow Paradigm](https://arxiv.org/abs/2402.10671) | 提出了一种通过工作流范式方法来改善LLMs在文本到SQL中的上下文学习能力，通过分解提高了模型的注意力和问题解决范围，进一步提高了基于LLM的方法的上限。 |
| [^10] | [Noise Contrastive Alignment of Language Models with Explicit Rewards](https://arxiv.org/abs/2402.05369) | 本文提出了一个基于噪声对比估计的通用LM对齐框架，能够处理明确注释的奖励数据，并且扩展了当前的对齐理论。 |
| [^11] | [Hierarchical Tree-structured Knowledge Graph For Academic Insight Survey](https://arxiv.org/abs/2402.04854) | 该论文提出了一种分层树状知识图谱和推荐系统，帮助初学者研究者进行研究调研，填补了现有导航知识图谱的不足，并解决了学术论文推荐系统中高文本相似性带来的困惑。 |
| [^12] | [DistiLLM: Towards Streamlined Distillation for Large Language Models](https://arxiv.org/abs/2402.03898) | DistiLLM是一个更有效和高效的自回归语言模型蒸馏框架，通过引入新颖的偏斜Kullback-Leibler散度损失和自适应的离策略方法，解决了当前针对大语言模型的知识蒸馏方法缺乏标准化目标函数和计算成本过高的问题。 |
| [^13] | [When Benchmarks are Targets: Revealing the Sensitivity of Large Language Model Leaderboards](https://arxiv.org/abs/2402.01781) | 依赖基准排行榜的大型语言模型评估存在较高敏感性，微小的扰动会导致排名的显著变化。研究结果提供了几个最佳实践建议，包括选择混合评分方法来提高答案选择的性能。 |
| [^14] | [Injecting linguistic knowledge into BERT for Dialogue State Tracking](https://arxiv.org/abs/2311.15623) | 本文提出了一种方法，在对话状态跟踪任务中，通过无监督的知识提取方法将语言知识注入到BERT中，以提高性能和可解释性。这种方法无需额外的训练数据，通过简单的神经模块实现。该方法使用的特征提取工具与对话的句法和语义模式相关，有助于理解DST模型的决策过程。 |
| [^15] | [Prompt Engineering a Prompt Engineer](https://arxiv.org/abs/2311.05661) | 提示工程任务对于优化大型语言模型在定制任务上的表现至关重要，PE2方法通过详细描述、上下文规范和逐步推理模板的注入，在各种语言任务中展现出出色的适用性和效果。 |
| [^16] | [Who Wrote this Code? Watermarking for Code Generation](https://arxiv.org/abs/2305.15060) | 基于代码独特的句法和语义特征，提出了一种新的水印方法SWEET，通过在具有高熵的位置仅放置“绿色”令牌来确保生成代码的正确性，并通过统计测试和Z分数进行检测。 |
| [^17] | [Transformers and Cortical Waves: Encoders for Pulling In Context Across Time.](http://arxiv.org/abs/2401.14267) | 这项研究探讨了transformer网络和大脑皮层波之间的相似性，并指出了皮层波在提取感觉输入序列中的时间上下文方面的潜在应用。 |
| [^18] | [Sketch-Guided Constrained Decoding for Boosting Blackbox Large Language Models without Logit Access.](http://arxiv.org/abs/2401.09967) | 本文介绍了一种无需访问逻辑回归的黑盒大型语言模型的草图引导约束解码的方法，通过利用本地辅助模型优化黑盒语言模型的输出，以初步输出作为进一步扩展的 "草图"，从而提高了有限约束解码的应用能力。 |
| [^19] | [GlotLID: Language Identification for Low-Resource Languages.](http://arxiv.org/abs/2310.16248) | GlotLID-M是一个满足广泛覆盖、可靠性和效率要求的语言识别模型，具有1665个可识别语言，并在实验中表现出色。它解决了低资源LID面临的挑战，并有望提高数据集质量和增强访问能力。 |
| [^20] | [Three Ways to Improve Verbo-visual Fusion for Dense 3D Visual Grounding.](http://arxiv.org/abs/2309.04561) | 提出了一个稠密三维引用网络ConcreteNet，包含三个新模块，旨在改善具有相同语义类别干扰因素的重复实例的引用性能。 |
| [^21] | [Towards Semantically Enriched Embeddings for Knowledge Graph Completion.](http://arxiv.org/abs/2308.00081) | 本论文讨论了知识图谱补全算法以及利用嵌入模型捕捉知识图谱中语义的不同方法，并提出知识图谱和语言模型相互受益的观点。 |
| [^22] | [PRD: Peer Rank and Discussion Improve Large Language Model based Evaluations.](http://arxiv.org/abs/2307.02762) | 本研究提出了PRD算法，利用同行评级和讨论改善了基于大型语言模型的评估方法，解决了自我提升和位置偏见等问题。 |
| [^23] | [Using Natural Language Processing and Networks to Automate Structured Literature Reviews: An Application to Farmers Climate Change Adaptation.](http://arxiv.org/abs/2306.09737) | 本文利用自然语言处理和网络，自动化结构化文献综述，以农民应对气候变化的分析为例，提取变量关系并使用网络综合其发现。 |
| [^24] | [A Framework For Refining Text Classification and Object Recognition from Academic Articles.](http://arxiv.org/abs/2305.17401) | 本文提出了一种结合基于规则的方法和机器学习的框架，旨在解决从学术论文中提炼文本分类和对象识别的问题。 |

# 详细

[^1]: Jamba: 一个混合Transformer-Mamba语言模型

    Jamba: A Hybrid Transformer-Mamba Language Model

    [https://arxiv.org/abs/2403.19887](https://arxiv.org/abs/2403.19887)

    Jamba是一个基于混合Transformer-Mamba架构的语言模型，在单个80GB GPU上实现了强大的性能，对标准语言模型基准和长上下文评估具有state-of-the-art的表现。

    

    我们提出了Jamba，这是一个基于新颖的混合Transformer-Mamba混合专家(MoE)架构的新基础大型语言模型。具体来说，Jamba交错使用Transformer和Mamba层，从两种模型家族中获益。MoE被添加在其中一些层中，以增加模型容量，同时保持活跃参数的可控性。这种灵活的架构允许特定资源和目标的配置。

    arXiv:2403.19887v1 Announce Type: new  Abstract: We present Jamba, a new base large language model based on a novel hybrid Transformer-Mamba mixture-of-experts (MoE) architecture. Specifically, Jamba interleaves blocks of Transformer and Mamba layers, enjoying the benefits of both model families. MoE is added in some of these layers to increase model capacity while keeping active parameter usage manageable. This flexible architecture allows resource- and objective-specific configurations. In the particular configuration we have implemented, we end up with a powerful model that fits in a single 80GB GPU. Built at large scale, Jamba provides high throughput and small memory footprint compared to vanilla Transformers, and at the same time state-of-the-art performance on standard language model benchmarks and long-context evaluations. Remarkably, the model presents strong results for up to 256K tokens context length. We study various architectural decisions, such as how to combine Transfor
    
[^2]: 使用程序执行运行时行为评估大型语言模型

    Evaluating Large Language Models with Runtime Behavior of Program Execution

    [https://arxiv.org/abs/2403.16437](https://arxiv.org/abs/2403.16437)

    本文提出了一个名为REval的框架，用于评估代码LLMs的代码推理能力以及与程序执行的一致性。

    

    大型代码语言模型（即代码LLMs）展示了强大的代码理解和生成能力。为了评估代码LLMs在各个方面的能力，已经提出了许多基准（如HumanEval和ClassEval）。代码推理是代码LLMs最重要的能力之一，但现有的代码推理基准不足。通常，它们重点预测程序的输入和输出，忽略了程序执行过程中的中间行为评估，以及逻辑一致性（例如，如果执行路径预测错误，则模型不应该给出正确的输出）在执行推理时。为了解决这些问题，本文提出了一个名为REval的框架，用于评估代码LLMs的代码推理能力以及与程序执行的一致性。我们利用现有的代码基准，并将它们适应到我们的框架中的新基准中。

    arXiv:2403.16437v1 Announce Type: cross  Abstract: Large language models for code (i.e., code LLMs) have shown strong code understanding and generation capabilities. To evaluate the capabilities of code LLMs in various aspects, many benchmarks have been proposed (e.g., HumanEval and ClassEval). Code reasoning is one of the most essential abilities of code LLMs, but existing benchmarks for code reasoning are not sufficient. Typically, they focus on predicting the input and output of a program, ignoring the evaluation of the intermediate behavior during program execution, as well as the logical consistency (e.g., the model should not give the correct output if the prediction of execution path is wrong) when performing the reasoning. To address these problems, in this paper, we propose a framework, namely REval, for evaluating code reasoning abilities and consistency of code LLMs with program execution. We utilize existing code benchmarks and adapt them to new benchmarks within our framew
    
[^3]: 当需要时给我打电话：LLM可以高效而忠实地推理结构化环境

    Call Me When Necessary: LLMs can Efficiently and Faithfully Reason over Structured Environments

    [https://arxiv.org/abs/2403.08593](https://arxiv.org/abs/2403.08593)

    LLMs借助Reasoning-Path-Editing (Readi)框架，可以在结构化环境中高效且忠实地推理，显著提升了多个KGQA和TableQA数据集上的表现。

    

    大型语言模型（LLMs）已经展示出在推理结构化环境中的潜力，例如知识图谱和表格。这些任务通常需要多跳推理，即将自然语言话语与环境中的实例匹配。以往的方法利用LLMs逐步构建推理路径，其中LLMs通过与环境逐步交互来调用工具或选择模式。我们提出了一种新颖的框架Reasoning-Path-Editing（Readi），在其中LLMs可以高效且忠实地在结构化环境中进行推理。在Readi中，LLMs在给定查询时最初生成一个推理路径，只有在必要时才编辑路径。我们将路径实例化到结构化环境上，并在出现问题时提供反馈以编辑路径。对三个KGQA数据集和两个TableQA数据集的实验结果显示，Readi的有效性，显著超越了所有基于LLM的方法（在WebQ上提高了9.1％）。

    arXiv:2403.08593v1 Announce Type: cross  Abstract: Large Language Models (LLMs) have shown potential in reasoning over structured environments, e.g., knowledge graph and table. Such tasks typically require multi-hop reasoning, i.e., match natural language utterance with instances in the environment. Previous methods leverage LLMs to incrementally build a reasoning path, where the LLMs either invoke tools or pick up schemas by step-by-step interacting with the environment. We propose Reasoning-Path-Editing (Readi), a novel framework where LLMs can efficiently and faithfully reason over structured environments. In Readi, LLMs initially generate a reasoning path given a query, and edit the path only when necessary. We instantiate the path on structured environments and provide feedback to edit the path if anything goes wrong. Experimental results on three KGQA datasets and two TableQA datasets show the effectiveness of Readi, significantly surpassing all LLM-based methods (by 9.1% on WebQ
    
[^4]: 使用顺序指令对大型语言模型进行微调

    Fine-tuning Large Language Models with Sequential Instructions

    [https://arxiv.org/abs/2403.07794](https://arxiv.org/abs/2403.07794)

    通过顺序指令微调，研究提出了一种简单且有效的策略，可以使大型语言模型具备执行多个顺序指令的能力，优于传统指令微调模型。

    

    大型语言模型（LLMs）在单个查询中遵循一系列指令时往往会忽略或误解其中的一部分，这影响了它们在解决需要多个中间步骤的复杂问题中的性能，例如多语言（先翻译再回答）和多模态（标题后回答）任务。我们通过开源LLMs（如LLaMA-2 70B和Mixtral-8x7B）的实证验证了这一点。针对当前数据中顺序指令稀缺的问题，我们提出了顺序指令微调，这是一种简单而有效的策略，可以自动增加指令调整数据，使LLMs具备执行多个顺序指令的能力。在探索现有数据集（如Alpaca）中插入指令并进行一系列中间任务后，我们发现，顺序指令微调的模型在下游任务中始终优于传统的指令微调基线。

    arXiv:2403.07794v1 Announce Type: new  Abstract: Large language models (LLMs) struggle to follow a sequence of instructions in a single query as they may ignore or misinterpret part of it. This impairs their performance in complex problems whose solution requires multiple intermediate steps, such as multilingual (translate then answer) and multimodal (caption then answer) tasks. We empirically verify this with open-source LLMs as large as LLaMA-2 70B and Mixtral-8x7B. Targeting the scarcity of sequential instructions in present-day data, we propose sequential instruction tuning, a simple yet effective strategy to automatically augment instruction tuning data and equip LLMs with the ability to execute multiple sequential instructions. After exploring interleaving instructions in existing datasets, such as Alpaca, with a wide range of intermediate tasks, we find that sequential instruction-tuned models consistently outperform the conventional instruction-tuned baselines in downstream tas
    
[^5]: GraphWiz：用于图问题的指令跟随语言模型

    GraphWiz: An Instruction-Following Language Model for Graph Problems

    [https://arxiv.org/abs/2402.16029](https://arxiv.org/abs/2402.16029)

    GraphWiz是一个开源语言模型，通过引入指令调优数据集和直接偏好优化框架，能够高效解决各种图问题类型，平均准确率达到65%，超过了GPT-4的43.8%。

    

    大型语言模型（LLMs）在多个领域取得了令人印象深刻的成功，但它们在理解和解决复杂图问题方面的能力尚未得到充分探索。为弥合这一差距，我们引入了GraphInstruct，这是一个新颖而全面的指令调优数据集，旨在为语言模型提供处理各种图问题的能力，利用明确的推理路径。利用GraphInstruct，我们构建了GraphWiz，这是一个能够解决各种图问题类型并生成清晰推理过程的开源语言模型。为增强模型的能力和可靠性，我们将直接偏好优化（DPO）框架纳入图问题求解环境中。增强模型GraphWiz-DPO在九个具有不同复杂性水平的任务中取得了65%的平均准确率，超过了平均准确率为43.8%的GPT-4。此外，我们的研究深入探讨了...

    arXiv:2402.16029v1 Announce Type: new  Abstract: Large language models (LLMs) have achieved impressive success across several fields, but their proficiency in understanding and resolving complex graph problems is less explored. To bridge this gap, we introduce GraphInstruct, a novel and comprehensive instruction-tuning dataset designed to equip language models with the ability to tackle a broad spectrum of graph problems using explicit reasoning paths. Utilizing GraphInstruct, we build GraphWiz, an open-source language model capable of resolving various graph problem types while generating clear reasoning processes. To enhance the model's capability and reliability, we incorporate the Direct Preference Optimization (DPO) framework into the graph problem-solving context. The enhanced model, GraphWiz-DPO, achieves an average accuracy of 65% across nine tasks with different complexity levels, surpassing GPT-4 which has an average accuracy of 43.8%. Moreover, our research delves into the d
    
[^6]: ProSparse: 引入和增强大型语言模型内部激活稀疏性

    ProSparse: Introducing and Enhancing Intrinsic Activation Sparsity within Large Language Models

    [https://arxiv.org/abs/2402.13516](https://arxiv.org/abs/2402.13516)

    本文介绍了一种名为"ProSparse"的有效稀疏化方法，以推动大型语言模型实现更高的激活稀疏性而不降低模型性能

    

    Activation sparsity指的是激活输出中存在许多弱贡献元素。作为使用ReLU激活函数的模型的普遍属性，已被证明是提高模型推理效率的一种有前途的范例。然而，大多数大型语言模型（LLMs）采用了没有内在激活稀疏性的激活函数（例如GELU和Swish）。一些最近的努力尝试引入ReLU或其变体作为替代激活函数，以帮助LLMs实现激活稀疏性和推理加速，但很少能同时获得高稀疏度和可比较的模型性能。本文介绍了一种名为"ProSparse"的有效稀疏化方法，以推动LLMs实现更高的激活稀疏性而不降低模型性能。具体来说，将LLMs的激活函数替换为ReLU后，ProSparse采用渐进稀疏正则化

    arXiv:2402.13516v1 Announce Type: cross  Abstract: Activation sparsity refers to the existence of considerable weakly-contributed elements among activation outputs. As a prevalent property of the models using the ReLU activation function, it has been proven a promising paradigm to boost model inference efficiency. Nevertheless, most large language models (LLMs) adopt activation functions without intrinsic activation sparsity (e.g., GELU and Swish). Some recent efforts have explored introducing ReLU or its variants as the substitutive activation function to help LLMs achieve activation sparsity and inference acceleration, but few can simultaneously obtain high sparsity and comparable model performance. This paper introduces an effective sparsification method named "ProSparse" to push LLMs for higher activation sparsity without decreasing model performance. Specifically, after substituting the activation function of LLMs with ReLU, ProSparse adopts progressive sparsity regularization wit
    
[^7]: Smaug：使用DPO-Positive修复偏好优化的失败模式

    Smaug: Fixing Failure Modes of Preference Optimisation with DPO-Positive

    [https://arxiv.org/abs/2402.13228](https://arxiv.org/abs/2402.13228)

    在这项工作中，我们提出了一种新的损失函数和训练过程DPO-Positive（DPOP），以避免直接偏好优化（DPO）中潜在的失败模式，并发现DPOP明显优于DPO。

    

    直接偏好优化（DPO）在显著改善大型语言模型（LLMs）在推理、总结和对齐等下游任务上的性能方面是有效的。 DPO使用首选和非首选数据对模型选择一个响应而不是另一个的“相对”概率进行建模。在这项工作中，我们首先从理论上表明，只要首选和非首选类别之间的相对概率增加，标准DPO损失就可能导致模型对首选示例的可能性降低。然后，我们在实证上展示了当在常见数据集上微调LLMs时，尤其是在完成之间的编辑距离较短的数据集上，会出现这种现象。利用这些见解，我们设计了DPO-Positive（DPOP），一种新的损失函数和训练过程，避免了这种失败模式。令人惊讶的是，我们还发现DPOP明显优于DPO。

    arXiv:2402.13228v1 Announce Type: cross  Abstract: Direct Preference Optimisation (DPO) is effective at significantly improving the performance of large language models (LLMs) on downstream tasks such as reasoning, summarisation, and alignment. Using pairs of preferred and dispreferred data, DPO models the \textit{relative} probability of picking one response over another. In this work, first we show theoretically that the standard DPO loss can lead to a \textit{reduction} of the model's likelihood of the preferred examples, as long as the relative probability between the preferred and dispreferred classes increases. We then show empirically that this phenomenon occurs when fine-tuning LLMs on common datasets, especially datasets in which the edit distance between pairs of completions is low. Using these insights, we design DPO-Positive (DPOP), a new loss function and training procedure which avoids this failure mode. Surprisingly, we also find that DPOP significantly outperforms DPO a
    
[^8]: MRKE：通过知识编辑对LLMs进行多跳推理评估

    MRKE: The Multi-hop Reasoning Evaluation of LLMs by Knowledge Edition

    [https://arxiv.org/abs/2402.11924](https://arxiv.org/abs/2402.11924)

    通过编辑HotpotQA数据集中的新知识，我们引入了一个LLM MHQA评估基准，同时注释和评估了推理链，揭示了当前MHQA基准存在数据污染的潜在风险。

    

    虽然大型语言模型（LLMs）在多跳问题回答（MHQA）任务中表现出色，但它们真正的推理能力仍有待探讨。目前的LLM QA评估基准存在一些限制，包括1）数据污染，评估数据可能在预训练阶段暴露给LLMs；以及2）忽视推理链评估。因此，我们引入了一种LLM MHQA评估基准，这是基于编辑现成HotpotQA数据集上的新、前所未有的知识的第一个QA基准；此外，我们还注释和评估了推理链，以子问题和中间答案的形式对应于多跳问题。具体来说，根据观察结果，1）LLMs在原始HotpotQA和我们编辑的数据之间显示性能差距，认为当前的MHQA基准可能存在数据污染的潜在风险，难以评估LLMs的性能。

    arXiv:2402.11924v1 Announce Type: new  Abstract: Although Large Language Models (LLMs) have shown strong performance in Multi-hop Question Answering (MHQA) tasks, their real reasoning ability remains exploration. Current LLM QA evaluation benchmarks have shown limitations, including 1) data contamination, the evaluation data are potentially exposed to LLMs during the pretraining stage; and 2) ignoration of the reasoning chain evaluation. Thus we introduce an LLM MHQA evaluation benchmark, the first QA benchmark based on the new, unprecedented knowledge by editing the off-the-shelf HotpotQA dataset; Besides, we also annotate and evaluate the reasoning chain in the form of sub-questions and intermediate answers corresponding to the multi-hop questions. Specifically, based on the observation, 1) LLMs show a performance gap between the original HotpotQA and our edited data, deeming that current MHQA benchmarks have the potential risk of data contamination that hard to evaluate LLMs' perfor
    
[^9]: 通过分解来增强注意力：通过工作流范式改进基于LLM的文本到SQL转换

    Decomposition for Enhancing Attention: Improving LLM-based Text-to-SQL through Workflow Paradigm

    [https://arxiv.org/abs/2402.10671](https://arxiv.org/abs/2402.10671)

    提出了一种通过工作流范式方法来改善LLMs在文本到SQL中的上下文学习能力，通过分解提高了模型的注意力和问题解决范围，进一步提高了基于LLM的方法的上限。

    

    大语言模型（LLMs）的上下文学习在自然语言处理领域取得了显著成功，而广泛的案例研究表明，单步链式思维提示方法在复杂任务（如文本到SQL）中面临注意力扩散和性能不足等挑战。为了改善LLMs在文本到SQL中的上下文学习能力，提出了一种工作流范式方法，旨在通过分解增强LLMs的注意力和问题解决范围。具体来说，用于消除冗余信息的信息确定模块和基于问题分类的全新提示结构极大增强了模型的注意力。此外，引入自校正和主动学习模块极大扩展了LLMs的问题解决范围，从而提高了基于LLM方法的上限。在三个数据集上进行了大量实验。

    arXiv:2402.10671v1 Announce Type: new  Abstract: In-context learning of large-language models (LLMs) has achieved remarkable success in the field of natural language processing, while extensive case studies reveal that the single-step chain-of-thought prompting approach faces challenges such as attention diffusion and inadequate performance in complex tasks like text-to-SQL. To improve the contextual learning capabilities of LLMs in text-to-SQL, a workflow paradigm method is proposed, aiming to enhance the attention and problem-solving scope of LLMs through decomposition. Specifically, the information determination module for eliminating redundant information and the brand-new prompt structure based on problem classification greatly enhance the model's attention. Additionally, the inclusion of self-correcting and active learning modules greatly expands the problem-solving scope of LLMs, hence improving the upper limit of LLM-based approaches. Extensive experiments conducted on three da
    
[^10]: 以显式奖励的噪声对比对齐语言模型

    Noise Contrastive Alignment of Language Models with Explicit Rewards

    [https://arxiv.org/abs/2402.05369](https://arxiv.org/abs/2402.05369)

    本文提出了一个基于噪声对比估计的通用LM对齐框架，能够处理明确注释的奖励数据，并且扩展了当前的对齐理论。

    

    用户意图通常被形式化为需要在微调语言模型时最大化的评估奖励。现有的对齐方法，如直接优化偏好（DPO），主要适用于隐含定义而非明确给定奖励的两两偏好数据。在本文中，我们引入了一个通用的LM对齐框架，利用噪声对比估计（NCE）来解决明确注释有标量评估的奖励数据处理的差距。我们的框架包括两个并行算法，NCA和InfoNCA，两者都能从奖励数据和偏好数据中直接提取LM策略。值得注意的是，我们证明了DPO损失是我们提出的InfoNCA目标在两两偏好设置下的特殊情况，从而集成和扩展了当前的对齐理论。通过对比NCA和InfoNCA，我们展示了InfoNCA和DPO如何在不同响应对于单个指令的相对可能性上进行调整。

    User intentions are typically formalized as evaluation rewards to be maximized when fine-tuning language models (LMs). Existing alignment methods, such as Direct Preference Optimization (DPO), are mainly tailored for pairwise preference data where rewards are implicitly defined rather than explicitly given. In this paper, we introduce a general framework for LM alignment, leveraging Noise Contrastive Estimation (NCE) to bridge the gap in handling reward datasets explicitly annotated with scalar evaluations. Our framework comprises two parallel algorithms, NCA and InfoNCA, both enabling the direct extraction of an LM policy from reward data as well as preference data. Notably, we show that the DPO loss is a special case of our proposed InfoNCA objective under pairwise preference settings, thereby integrating and extending current alignment theories. By contrasting NCA and InfoNCA, we show that InfoNCA and DPO adjust relative likelihood across different responses to a single instruction,
    
[^11]: 分层树状知识图谱用于学术调研

    Hierarchical Tree-structured Knowledge Graph For Academic Insight Survey

    [https://arxiv.org/abs/2402.04854](https://arxiv.org/abs/2402.04854)

    该论文提出了一种分层树状知识图谱和推荐系统，帮助初学者研究者进行研究调研，填补了现有导航知识图谱的不足，并解决了学术论文推荐系统中高文本相似性带来的困惑。

    

    对于缺乏研究培训的初学者研究者来说，研究调查一直是一个挑战。这些研究者在短时间内很难理解他们研究主题内的方向，以及发现新的研究发现。为初学者研究者提供直观的帮助的一种方式是提供相关的知识图谱(KG)并推荐相关的学术论文。然而，现有的导航知识图谱主要依赖于研究领域的关键字，常常无法清楚地呈现多个相关论文之间的逻辑层次关系。此外，大多数学术论文推荐系统仅仅依赖于高文本相似性，这可能会让研究人员困惑为什么推荐了特定的文章。他们可能缺乏了解关于他们希望获得的"问题解决"和"问题发现"之间的见解连接的重要信息。为解决这些问题，本研究旨在支持初学者研究者进行研究调研。

    Research surveys have always posed a challenge for beginner researchers who lack of research training. These researchers struggle to understand the directions within their research topic, and the discovery of new research findings within a short time. One way to provide intuitive assistance to beginner researchers is by offering relevant knowledge graphs(KG) and recommending related academic papers. However, existing navigation knowledge graphs primarily rely on keywords in the research field and often fail to present the logical hierarchy among multiple related papers clearly. Moreover, most recommendation systems for academic papers simply rely on high text similarity, which can leave researchers confused as to why a particular article is being recommended. They may lack of grasp important information about the insight connection between "Issue resolved" and "Issue finding" that they hope to obtain. To address these issues, this study aims to support research insight surveys for begi
    
[^12]: DistiLLM: 面向大型语言模型的简化蒸馏方法

    DistiLLM: Towards Streamlined Distillation for Large Language Models

    [https://arxiv.org/abs/2402.03898](https://arxiv.org/abs/2402.03898)

    DistiLLM是一个更有效和高效的自回归语言模型蒸馏框架，通过引入新颖的偏斜Kullback-Leibler散度损失和自适应的离策略方法，解决了当前针对大语言模型的知识蒸馏方法缺乏标准化目标函数和计算成本过高的问题。

    

    知识蒸馏（KD）被广泛用于将教师模型压缩为更小的学生模型，降低推理成本和内存占用，同时保持模型能力。然而，当前针对自回归序列模型（例如大型语言模型）的KD方法存在缺乏标准化目标函数的问题。此外，最近使用学生生成的输出来解决训练-推理不匹配问题的做法显著增加了计算成本。为了解决这些问题，我们引入了DistiLLM，这是一个更有效和高效的自回归语言模型蒸馏框架。DistiLLM由两个组成部分组成：（1）一种新颖的偏斜Kullback-Leibler散度损失，我们揭示并利用了它的理论属性；（2）一种自适应的离策略方法，旨在提高利用学生生成的输出的效率。包括指令跟随任务在内的大量实验验证了DistiLLM在构建高性能模型方面的有效性。

    Knowledge distillation (KD) is widely used for compressing a teacher model to a smaller student model, reducing its inference cost and memory footprint while preserving model capabilities. However, current KD methods for auto-regressive sequence models (e.g., large language models) suffer from missing a standardized objective function. Moreover, the recent use of student-generated outputs to address training-inference mismatches has significantly escalated computational costs. To tackle these issues, we introduce DistiLLM, a more effective and efficient KD framework for auto-regressive language models. DistiLLM comprises two components: (1) a novel skew Kullback-Leibler divergence loss, where we unveil and leverage its theoretical properties, and (2) an adaptive off-policy approach designed to enhance the efficiency in utilizing student-generated outputs. Extensive experiments, including instruction-following tasks, demonstrate the effectiveness of DistiLLM in building high-performing 
    
[^13]: 当基准成为目标：揭示大型语言模型排行榜的敏感性

    When Benchmarks are Targets: Revealing the Sensitivity of Large Language Model Leaderboards

    [https://arxiv.org/abs/2402.01781](https://arxiv.org/abs/2402.01781)

    依赖基准排行榜的大型语言模型评估存在较高敏感性，微小的扰动会导致排名的显著变化。研究结果提供了几个最佳实践建议，包括选择混合评分方法来提高答案选择的性能。

    

    基于基准排名的大型语言模型(LLM)排行榜经常被用来指导实践者在模型选择中。通常，发布的排行榜排名被直接接受 - 我们表明这是一个（潜在昂贵的）错误。在现有的排行榜下，LLM的相对性能对（通常微小的）细节非常敏感。我们展示了对于流行的多项选择题基准（例如MMLU），对基准的微小扰动，如改变选项顺序或答案选择方法，会导致排名变化达到8个位置。我们通过对三个广泛的基准扰动类别进行系统实验并确定这一行为的来源来解释这一现象。我们的分析得出了几个最佳实践建议，包括选择优化的混合评分方法来进行答案选择。我们的研究强调了依赖简单基准评估的风险，并为更健壮的模型评估提供了指导道路。

    Large Language Model (LLM) leaderboards based on benchmark rankings are regularly used to guide practitioners in model selection. Often, the published leaderboard rankings are taken at face value - we show this is a (potentially costly) mistake. Under existing leaderboards, the relative performance of LLMs is highly sensitive to (often minute) details. We show that for popular multiple choice question benchmarks (e.g. MMLU) minor perturbations to the benchmark, such as changing the order of choices or the method of answer selection, result in changes in rankings up to 8 positions. We explain this phenomenon by conducting systematic experiments over three broad categories of benchmark perturbations and identifying the sources of this behavior. Our analysis results in several best-practice recommendations, including the advantage of a hybrid scoring method for answer selection. Our study highlights the dangers of relying on simple benchmark evaluations and charts the path for more robust
    
[^14]: 将语言知识注入到BERT中用于对话状态跟踪

    Injecting linguistic knowledge into BERT for Dialogue State Tracking

    [https://arxiv.org/abs/2311.15623](https://arxiv.org/abs/2311.15623)

    本文提出了一种方法，在对话状态跟踪任务中，通过无监督的知识提取方法将语言知识注入到BERT中，以提高性能和可解释性。这种方法无需额外的训练数据，通过简单的神经模块实现。该方法使用的特征提取工具与对话的句法和语义模式相关，有助于理解DST模型的决策过程。

    

    对话状态跟踪(DST)模型通常采用复杂的神经网络架构，需要大量的训练数据，其推理过程缺乏透明性。本文提出了一种方法，通过无监督框架提取语言知识，然后利用这些知识来增强BERT在DST任务中的性能和可解释性。知识提取过程计算经济高效，不需要注释或额外的训练数据。注入提取的知识只需要添加简单的神经模块。我们使用凸多面体模型(CPM)作为DST任务的特征提取工具，并表明所获取的特征与对话中的句法和语义模式相关。这种相关性有助于全面理解影响DST模型决策过程的语言特征。我们在不同的DST任务上对这个框架进行基准测试，并展示了其效果。

    Dialogue State Tracking (DST) models often employ intricate neural network architectures, necessitating substantial training data, and their inference processes lack transparency. This paper proposes a method that extracts linguistic knowledge via an unsupervised framework and subsequently utilizes this knowledge to augment BERT's performance and interpretability in DST tasks. The knowledge extraction procedure is computationally economical and does not necessitate annotations or additional training data. The injection of the extracted knowledge necessitates the addition of only simple neural modules. We employ the Convex Polytopic Model (CPM) as a feature extraction tool for DST tasks and illustrate that the acquired features correlate with the syntactic and semantic patterns in the dialogues. This correlation facilitates a comprehensive understanding of the linguistic features influencing the DST model's decision-making process. We benchmark this framework on various DST tasks and ob
    
[^15]: Prompt Engineering a Prompt Engineer

    Prompt Engineering a Prompt Engineer

    [https://arxiv.org/abs/2311.05661](https://arxiv.org/abs/2311.05661)

    提示工程任务对于优化大型语言模型在定制任务上的表现至关重要，PE2方法通过详细描述、上下文规范和逐步推理模板的注入，在各种语言任务中展现出出色的适用性和效果。

    

    提示工程是优化大型语言模型在定制任务上表现的一项具有挑战性但至关重要的任务。为了检查模型的错误，假设当前提示中缺少或误导了什么，并清晰地传达任务，需要复杂的推理。尽管最近的研究表明，大型语言模型可以被元提示来执行自动提示工程，但我们认为由于元提示中缺乏复杂推理的充分指导，它们的潜力受到限制。我们通过将详细描述、上下文规范和逐步推理模板注入到元提示中来填补这一空白。所得到的方法称为PE2，展示了在不同语言任务中出色的适用性。它找到的提示在MultiArith上比“按步骤思考”高出6.3%，在GSM8K上高出3.1%，并在对立任务上优于竞争基线

    arXiv:2311.05661v2 Announce Type: replace-cross  Abstract: Prompt engineering is a challenging yet crucial task for optimizing the performance of large language models on customized tasks. It requires complex reasoning to examine the model's errors, hypothesize what is missing or misleading in the current prompt, and communicate the task with clarity. While recent works indicate that large language models can be meta-prompted to perform automatic prompt engineering, we argue that their potential is limited due to insufficient guidance for complex reasoning in the meta-prompt. We fill this gap by infusing into the meta-prompt three key components: detailed descriptions, context specification, and a step-by-step reasoning template. The resulting method, named PE2, showcases remarkable versatility across diverse language tasks. It finds prompts that outperform "let's think step by step" by 6.3% on MultiArith and 3.1% on GSM8K, and outperforms competitive baselines on counterfactual tasks 
    
[^16]: 谁编写了这段代码？用于代码生成的水印技术

    Who Wrote this Code? Watermarking for Code Generation

    [https://arxiv.org/abs/2305.15060](https://arxiv.org/abs/2305.15060)

    基于代码独特的句法和语义特征，提出了一种新的水印方法SWEET，通过在具有高熵的位置仅放置“绿色”令牌来确保生成代码的正确性，并通过统计测试和Z分数进行检测。

    

    随着大型语言模型的出色生成性能，关于使用它们的道德和法律问题日益受到关注，如抄袭和版权问题。为了应对这些问题，最近提出了几种用于水印和检测LLM生成文本的方法。然而，我们发现先前的方法由于代码的句法和语义特征，无法有效地应用于代码生成任务。基于Kirchenbauer等人的研究，我们提出了一种新的水印方法，名为Selective WatErmarking via Entropy Thresholding（SWEET），该方法仅在生成期间将“绿色”令牌放置在具有高熵的令牌分布位置，从而保留生成代码的正确性。水印代码通过基于熵信息的统计测试和Z分数进行检测。我们在HumanEval和MBPP上的实验表明，SWEET显著改善了生成代码的质量。

    arXiv:2305.15060v3 Announce Type: replace  Abstract: With the remarkable generation performance of large language models, ethical and legal concerns about using them have been raised, such as plagiarism and copyright issues. For such concerns, several approaches to watermark and detect LLM-generated text have been proposed very recently. However, we discover that the previous methods fail to function appropriately with code generation tasks because of the syntactic and semantic characteristics of code. Based on \citet{Kirchenbauer2023watermark}, we propose a new watermarking method, Selective WatErmarking via Entropy Thresholding (SWEET), that promotes "green" tokens only at the position with high entropy of the token distribution during generation, thereby preserving the correctness of the generated code. The watermarked code is detected by the statistical test and Z-score based on the entropy information. Our experiments on HumanEval and MBPP show that SWEET significantly improves th
    
[^17]: Transformers和大脑皮层波：在时间上传递上下文的编码器

    Transformers and Cortical Waves: Encoders for Pulling In Context Across Time. (arXiv:2401.14267v1 [cs.CL])

    [http://arxiv.org/abs/2401.14267](http://arxiv.org/abs/2401.14267)

    这项研究探讨了transformer网络和大脑皮层波之间的相似性，并指出了皮层波在提取感觉输入序列中的时间上下文方面的潜在应用。

    

    类似ChatGPT和其他大语言模型（LLM）的transformer网络的能力已经引起了世界的关注。它们的性能依赖于将完整的输入序列（例如句子中的所有单词）转化为一个长的“编码向量”，使得transformer能够学习自然序列中的长程时间依赖关系。具体而言，“自注意力”应用于这个编码向量，通过计算输入序列中单词对之间的关联，增强了transformer中的时间上下文。我们认为神经活动在单个皮层区域内或整个大脑范围内传播的波可以实现类似的编码原理。通过在每个时刻将最近的输入历史封装为单个空间模式，皮层波可以从感觉输入序列中提取时间上下文，这与计算原理相同。

    The capabilities of transformer networks such as ChatGPT and other Large Language Models (LLMs) have captured the world's attention. The crucial computational mechanism underlying their performance relies on transforming a complete input sequence - for example, all the words in a sentence into a long "encoding vector" - that allows transformers to learn long-range temporal dependencies in naturalistic sequences. Specifically, "self-attention" applied to this encoding vector enhances temporal context in transformers by computing associations between pairs of words in the input sequence. We suggest that waves of neural activity, traveling across single cortical regions or across multiple regions at the whole-brain scale, could implement a similar encoding principle. By encapsulating recent input history into a single spatial pattern at each moment in time, cortical waves may enable temporal context to be extracted from sequences of sensory inputs, the same computational principle used in
    
[^18]: 无需访问逻辑回归的黑盒大型语言模型的草图引导约束解码

    Sketch-Guided Constrained Decoding for Boosting Blackbox Large Language Models without Logit Access. (arXiv:2401.09967v1 [cs.CL])

    [http://arxiv.org/abs/2401.09967](http://arxiv.org/abs/2401.09967)

    本文介绍了一种无需访问逻辑回归的黑盒大型语言模型的草图引导约束解码的方法，通过利用本地辅助模型优化黑盒语言模型的输出，以初步输出作为进一步扩展的 "草图"，从而提高了有限约束解码的应用能力。

    

    有限约束在语言模型输出的控制上提供了一种不需要重新训练或架构修改的方式，但通常只适用于拥有逻辑回归访问权限的模型，这对于黑盒大型语言模型存在限制。本文引入了一种新颖的基于草图引导的黑盒大型语言模型约束解码（SGCD）方法，无需访问黑盒语言模型的逻辑回归。SGCD利用本地辅助模型来优化无约束黑盒语言模型的输出，将其作为进一步扩展的“草图”。此方法可与传统的基于逻辑回归的技术相互补充，使有限约束解码在无法完全透明的模型环境中应用。通过实验展示了SGCD的有效性。

    Constrained decoding, a technique for enforcing constraints on language model outputs, offers a way to control text generation without retraining or architectural modifications. Its application is, however, typically restricted to models that give users access to next-token distributions (usually via softmax logits), which poses a limitation with blackbox large language models (LLMs). This paper introduces sketch-guided constrained decoding (SGCD), a novel approach to constrained decoding for blackbox LLMs, which operates without access to the logits of the blackbox LLM. SGCD utilizes a locally hosted auxiliary model to refine the output of an unconstrained blackbox LLM, effectively treating this initial output as a "sketch" for further elaboration. This approach is complementary to traditional logit-based techniques and enables the application of constrained decoding in settings where full model transparency is unavailable. We demonstrate the efficacy of SGCD through experiments in cl
    
[^19]: GlotLID: 低资源语言的语言识别

    GlotLID: Language Identification for Low-Resource Languages. (arXiv:2310.16248v1 [cs.CL])

    [http://arxiv.org/abs/2310.16248](http://arxiv.org/abs/2310.16248)

    GlotLID-M是一个满足广泛覆盖、可靠性和效率要求的语言识别模型，具有1665个可识别语言，并在实验中表现出色。它解决了低资源LID面临的挑战，并有望提高数据集质量和增强访问能力。

    

    最近有几篇论文发表了针对约300种高资源和中资源语言的语言识别（LID）的良好解决方案。然而，目前没有可用的LID满足以下要求：（i）涵盖广泛的低资源语言，（ii）经过严格评估且可靠，（iii）高效易用。在这里，我们发布了GlotLID-M，一个满足广泛覆盖、可靠性和效率要求的LID模型。它可以识别1665种语言，在覆盖范围上相比之前的工作有了大幅增加。在我们的实验中，GlotLID-M在平衡F1分数和假阳性率（FPR）方面优于四个基准模型（CLD3，FT176，OpenLID和NLLB）。我们分析了低资源LID面临的独特挑战：不正确的语料库元数据，来自高资源语言的泄漏，难以区分密切相关的语言，处理宏语言与方言，以及一般的噪声数据。我们希望将GlotLID-M集成到数据集创建流程中，以提高质量和增强访问能力。

    Several recent papers have published good solutions for language identification (LID) for about 300 high-resource and medium-resource languages. However, there is no LID available that (i) covers a wide range of low-resource languages, (ii) is rigorously evaluated and reliable and (iii) efficient and easy to use. Here, we publish GlotLID-M, an LID model that satisfies the desiderata of wide coverage, reliability and efficiency. It identifies 1665 languages, a large increase in coverage compared to prior work. In our experiments, GlotLID-M outperforms four baselines (CLD3, FT176, OpenLID and NLLB) when balancing F1 and false positive rate (FPR). We analyze the unique challenges that low-resource LID poses: incorrect corpus metadata, leakage from high-resource languages, difficulty separating closely related languages, handling of macrolanguage vs varieties and in general noisy data. We hope that integrating GlotLID-M into dataset creation pipelines will improve quality and enhance acces
    
[^20]: 改进稠密三维视觉引用的三种方法

    Three Ways to Improve Verbo-visual Fusion for Dense 3D Visual Grounding. (arXiv:2309.04561v1 [cs.CV])

    [http://arxiv.org/abs/2309.04561](http://arxiv.org/abs/2309.04561)

    提出了一个稠密三维引用网络ConcreteNet，包含三个新模块，旨在改善具有相同语义类别干扰因素的重复实例的引用性能。

    

    三维视觉引用是指通过自然语言描述来定位三维场景中被引用的物体的任务。该任务在自主室内机器人到AR/VR等各种应用中广泛应用。目前一种常见的解决方案是通过检测来完成三维视觉引用，即通过边界框来定位。然而，在需要进行物理交互的实际应用中，边界框不足以描述物体的几何属性。因此，我们解决了稠密三维视觉引用的问题，即基于引用的三维实例分割。我们提出了一个稠密三维引用网络ConcreteNet，其中包含三个独立的新模块，旨在改进具有相同语义类别干扰因素的具有挑战性的重复实例的引用性能。首先，我们引入了一个自下而上的注意力融合模块，旨在消除实例间关系线索的歧义性。接下来，我们构造一个cont

    3D visual grounding is the task of localizing the object in a 3D scene which is referred by a description in natural language. With a wide range of applications ranging from autonomous indoor robotics to AR/VR, the task has recently risen in popularity. A common formulation to tackle 3D visual grounding is grounding-by-detection, where localization is done via bounding boxes. However, for real-life applications that require physical interactions, a bounding box insufficiently describes the geometry of an object. We therefore tackle the problem of dense 3D visual grounding, i.e. referral-based 3D instance segmentation. We propose a dense 3D grounding network ConcreteNet, featuring three novel stand-alone modules which aim to improve grounding performance for challenging repetitive instances, i.e. instances with distractors of the same semantic class. First, we introduce a bottom-up attentive fusion module that aims to disambiguate inter-instance relational cues, next we construct a cont
    
[^21]: 为知识图谱补全构建语义丰富的嵌入模型

    Towards Semantically Enriched Embeddings for Knowledge Graph Completion. (arXiv:2308.00081v1 [cs.AI])

    [http://arxiv.org/abs/2308.00081](http://arxiv.org/abs/2308.00081)

    本论文讨论了知识图谱补全算法以及利用嵌入模型捕捉知识图谱中语义的不同方法，并提出知识图谱和语言模型相互受益的观点。

    

    基于嵌入模型的知识图谱补全在过去几年中越来越受关注。目前的大多数算法将知识图谱视为一个多向标记图，缺乏捕捉底层语义的能力。与此同时，大型语言模型（LLMs）已经捕获了大量信息，这一捕获对人工智能领域产生了革命性影响。知识图谱可以从LLMs中受益，反之亦然。本文讨论了基于不同生成嵌入模型变体的知识图谱补全算法。首先讨论了各种知识图谱补全算法，如转导和归纳链接预测以及实体类型预测算法。然后，介绍了利用知识图谱中的类型信息、LLMs以及捕捉不同描述逻辑公理中的语义的算法。最后，通过对现有算法的关键反思对论文进行总结。

    Embedding based Knowledge Graph (KG) Completion has gained much attention over the past few years. Most of the current algorithms consider a KG as a multidirectional labeled graph and lack the ability to capture the semantics underlying the schematic information. In a separate development, a vast amount of information has been captured within the Large Language Models (LLMs) which has revolutionized the field of Artificial Intelligence. KGs could benefit from these LLMs and vice versa. This vision paper discusses the existing algorithms for KG completion based on the variations for generating KG embeddings. It starts with discussing various KG completion algorithms such as transductive and inductive link prediction and entity type prediction algorithms. It then moves on to the algorithms utilizing type information within the KGs, LLMs, and finally to algorithms capturing the semantics represented in different description logic axioms. We conclude the paper with a critical reflection on
    
[^22]: PRD: 同行评级和讨论改善基于大型语言模型的评估

    PRD: Peer Rank and Discussion Improve Large Language Model based Evaluations. (arXiv:2307.02762v1 [cs.CL])

    [http://arxiv.org/abs/2307.02762](http://arxiv.org/abs/2307.02762)

    本研究提出了PRD算法，利用同行评级和讨论改善了基于大型语言模型的评估方法，解决了自我提升和位置偏见等问题。

    

    如今，评估和比较不同现代大型语言模型（LLMs）生成的回答质量在自动化方面很难。最近的研究建议并主要使用LLMs作为无参考度量衡开放式问题回答的参考指标。更具体地说，他们以被认为是“最强”的LLM作为评估器，对候选模型的答案进行两两比较并提供排名分数。然而，这种直观的方法存在多个问题，例如带来自我提升（青睐自己的答案）和位置偏见。我们从教育领域（Cho and MacArthur, 2011；Walsh, 2014）中汲取见解和教训，改进了基于LLM的评估。具体而言，我们提出了（1）同行评级（PR）算法，该算法考虑每个同行LLM对所有答案对的两两偏好，并输出模型的最终排名；以及（2）同行讨论（PD），在其中我们促使两个LLMs进行讨论并尝试就两个偏好达成共识。

    Nowadays, the quality of responses generated by different modern large language models (LLMs) are hard to evaluate and compare automatically. Recent studies suggest and predominantly use LLMs as a reference-free metric for open-ended question answering. More specifically, they use the recognized "strongest" LLM as the evaluator, which conducts pairwise comparisons of candidate models' answers and provides a ranking score. However, this intuitive method has multiple problems, such as bringing in self-enhancement (favoring its own answers) and positional bias. We draw insights and lessons from the educational domain (Cho and MacArthur, 2011; Walsh, 2014) to improve LLM-based evaluations. Specifically, we propose the (1) peer rank (PR) algorithm that takes into account each peer LLM's pairwise preferences of all answer pairs, and outputs a final ranking of models; and (2) peer discussion (PD), where we prompt two LLMs to discuss and try to reach a mutual agreement on preferences of two an
    
[^23]: 使用自然语言处理和网络自动化结构化文献综述：以农民应对气候变化为例

    Using Natural Language Processing and Networks to Automate Structured Literature Reviews: An Application to Farmers Climate Change Adaptation. (arXiv:2306.09737v1 [cs.CL])

    [http://arxiv.org/abs/2306.09737](http://arxiv.org/abs/2306.09737)

    本文利用自然语言处理和网络，自动化结构化文献综述，以农民应对气候变化的分析为例，提取变量关系并使用网络综合其发现。

    

    随着研究文章数量的增加，学者们很难跟上与自己专业领域相关的新发现。此外，对于需要跨学科解决方案的复杂主题，如气候变化，跨学科研究之间的知识链接也变得具有挑战性。同时，黑匣子类型的文本摘要的兴起使得理解文本关系的建立变得困难，更不用说与已有理论概念化因果关系并进行假设的相关性了。本文旨在合理地利用自然语言处理，通过提取变量关系并使用网络综合其发现，同时与相关学科中占主导地位的关键概念相关联。我们以农民应对气候变化的分析为例。为此，我们对Scopus于2022年8月返回的出版物进行自然语言处理分析。结果展示...

    The fast-growing number of research articles makes it problematic for scholars to keep track of the new findings related to their areas of expertise. Furthermore, linking knowledge across disciplines in rapidly developing fields becomes challenging for complex topics like climate change that demand interdisciplinary solutions. At the same time, the rise of Black Box types of text summarization makes it difficult to understand how text relationships are built, let alone relate to existing theories conceptualizing cause-effect relationships and permitting hypothesizing. This work aims to sensibly use Natural Language Processing by extracting variables relations and synthesizing their findings using networks while relating to key concepts dominant in relevant disciplines. As an example, we apply our methodology to the analysis of farmers' adaptation to climate change. For this, we perform a Natural Language Processing analysis of publications returned by Scopus in August 2022. Results sho
    
[^24]: 一种从学术论文中提炼文本分类和对象识别的框架

    A Framework For Refining Text Classification and Object Recognition from Academic Articles. (arXiv:2305.17401v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2305.17401](http://arxiv.org/abs/2305.17401)

    本文提出了一种结合基于规则的方法和机器学习的框架，旨在解决从学术论文中提炼文本分类和对象识别的问题。

    

    随着互联网的广泛使用，高效地从大量学术论文中提取特定信息变得越来越重要。数据挖掘技术通常用于解决这个问题。然而，挖掘学术论文的数据具有挑战性，因为它需要自动从复杂的非结构化布局文档中提取特定模式。当前的学术论文数据挖掘方法使用基于规则的（RB）或机器学习（ML）方法。然而，使用基于规则的方法需要编写复杂排版论文的高昂成本。另一方面，仅使用机器学习方法需要对文章中复杂内容类型进行注释工作，这可能成本高昂。此外，仅使用机器学习可能会导致基于规则的方法容易识别的模式被错误提取的情况。为了解决这些问题，本文从分析指定著作中使用的标准布局和排版角度出发，提出了一种结合基于规则的方法和机器学习的框架。

    With the widespread use of the internet, it has become increasingly crucial to extract specific information from vast amounts of academic articles efficiently. Data mining techniques are generally employed to solve this issue. However, data mining for academic articles is challenging since it requires automatically extracting specific patterns in complex and unstructured layout documents. Current data mining methods for academic articles employ rule-based(RB) or machine learning(ML) approaches. However, using rule-based methods incurs a high coding cost for complex typesetting articles. On the other hand, simply using machine learning methods requires annotation work for complex content types within the paper, which can be costly. Furthermore, only using machine learning can lead to cases where patterns easily recognized by rule-based methods are mistakenly extracted. To overcome these issues, from the perspective of analyzing the standard layout and typesetting used in the specified p
    

