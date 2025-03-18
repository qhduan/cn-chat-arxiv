# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [CheckEval: Robust Evaluation Framework using Large Language Model via Checklist](https://arxiv.org/abs/2403.18771) | CheckEval是一种使用大型语言模型构建的评估框架，通过详细的子方面和布尔问题清单简化了评估过程，增强了评估结果的稳健性和可靠性，通过SummEval基准验证其有效性。 |
| [^2] | [SVD-LLM: Truncation-aware Singular Value Decomposition for Large Language Model Compression](https://arxiv.org/abs/2403.07378) | SVD-LLM是一种新的基于SVD的LLM压缩方法，通过截断感知数据白化策略和逐层闭式模型参数更新策略，解决了现有方法的限制，实现了直接映射奇异值和压缩损失之间的关系。 |
| [^3] | [Learning with Noisy Foundation Models](https://arxiv.org/abs/2403.06869) | 本文首次全面了解和分析了预训练数据集中的噪声性质，有效减轻其对下游任务影响。 |
| [^4] | [Jailbreaking Proprietary Large Language Models using Word Substitution Cipher](https://arxiv.org/abs/2402.10601) | 本文使用密码技术编码了越狱提示，成功地绕过了大型语言模型对有害问题的检测，实验结果显示攻击成功率高达59.42%。 |
| [^5] | [A Systematic Survey of Prompt Engineering in Large Language Models: Techniques and Applications](https://arxiv.org/abs/2402.07927) | 这篇调查论文系统概述了大型语言模型中提示工程的最新进展，探讨了提示工程的方法和技术，并说明了其在各种应用中的重要作用。 |
| [^6] | [Permute-and-Flip: An optimally robust and watermarkable decoder for LLMs](https://arxiv.org/abs/2402.05864) | 提出了一种名为Permute-and-Flip（PF）解码器，其具有最佳的鲁棒性和质量-鲁棒性的 tradeoff，且比采样方法更好。还设计了一种针对PF解码器的水印方案，能够保持样本的分布不变，并实现任意低的假阳性率和高的召回率。实验证明PF解码器在困惑度方面明显优于朴素采样，为LLM解码提供了一种有希望的新方法。 |
| [^7] | [Evidence to Generate (E2G): A Single-agent Two-step Prompting for Context Grounded and Retrieval Augmented Reasoning.](http://arxiv.org/abs/2401.05787) | 本研究提出了Evidence to Generate（E2G）框架，采用单代理、两步提示的方法来解决目前链式思维提示存在的限制，通过利用上下文中明确提及的思维序列作为证据，以更高的精确度和效率引导LLM的输出生成过程，实现更快、更可靠和更具上下文意识的推理。 |
| [^8] | [Leveraging Large Language Models for Collective Decision-Making.](http://arxiv.org/abs/2311.04928) | 本论文提出了一种利用大型语言模型（LLM）促进集体决策的系统，通过管理对话和平衡个人偏好来提供满足成员需求的选项，实现高效协调并不断优化系统性能。 |
| [^9] | [Thrust: Adaptively Propels Large Language Models with External Knowledge.](http://arxiv.org/abs/2307.10442) | 本论文提出了一种实例级的自适应推动外部知识的方法，通过衡量大型语言模型的知识水平，并利用Thrust指标进行信息检索，实现更高的成本效益。 |
| [^10] | [Evaluating the Capability of Large-scale Language Models on Chinese Grammatical Error Correction Task.](http://arxiv.org/abs/2307.03972) | 本研究评估了大规模语言模型在中文语法错误修正任务上的表现，并发现存在过度修正的问题。此外，我们还发现在评估不同数据分布时，大型语言模型的性能有显著变化。 |
| [^11] | [Valley: Video Assistant with Large Language model Enhanced abilitY.](http://arxiv.org/abs/2306.07207) | 本文介绍了一个名为Valley的视频助手，它是一个以大型语言模型增强的多模态基础模型，能够在一个通用框架内理解视频、图像和语言。 |

# 详细

[^1]: CheckEval: 使用大型语言模型通过清单构建健壮评估框架

    CheckEval: Robust Evaluation Framework using Large Language Model via Checklist

    [https://arxiv.org/abs/2403.18771](https://arxiv.org/abs/2403.18771)

    CheckEval是一种使用大型语言模型构建的评估框架，通过详细的子方面和布尔问题清单简化了评估过程，增强了评估结果的稳健性和可靠性，通过SummEval基准验证其有效性。

    

    我们介绍了CheckEval，一种使用大型语言模型的新型评估框架，解决了当前评估方法中的歧义和不一致性挑战。CheckEval通过将评估标准分解为详细的子方面，并为每个构建布尔问题清单，简化了评估过程。这种方法不仅使过程更具可解释性，还通过专注于特定的评估维度显着增强了结果的稳健性和可靠性。通过使用SummEval基准进行的专注案例研究验证，CheckEval显示出与人类判断的强相关性。此外，它展示了高度一致的互注者一致性。这些发现突显了CheckEval在客观、灵活和精确评估方面的有效性。通过提供可定制和互动的框架，CheckEval为LL的使用设立了新的标准。

    arXiv:2403.18771v1 Announce Type: new  Abstract: We introduce CheckEval, a novel evaluation framework using Large Language Models, addressing the challenges of ambiguity and inconsistency in current evaluation methods. CheckEval addresses these challenges by dividing evaluation criteria into detailed sub-aspects and constructing a checklist of Boolean questions for each, simplifying the evaluation. This approach not only renders the process more interpretable but also significantly enhances the robustness and reliability of results by focusing on specific evaluation dimensions. Validated through a focused case study using the SummEval benchmark, CheckEval indicates a strong correlation with human judgments. Furthermore, it demonstrates a highly consistent Inter-Annotator Agreement. These findings highlight the effectiveness of CheckEval for objective, flexible, and precise evaluations. By offering a customizable and interactive framework, CheckEval sets a new standard for the use of LL
    
[^2]: SVD-LLM: 针对大型语言模型压缩的截断感知奇异值分解

    SVD-LLM: Truncation-aware Singular Value Decomposition for Large Language Model Compression

    [https://arxiv.org/abs/2403.07378](https://arxiv.org/abs/2403.07378)

    SVD-LLM是一种新的基于SVD的LLM压缩方法，通过截断感知数据白化策略和逐层闭式模型参数更新策略，解决了现有方法的限制，实现了直接映射奇异值和压缩损失之间的关系。

    

    大型语言模型（LLMs）的进展受到其庞大尺寸的限制，这需要LLM压缩方法以实现实际部署。奇异值分解（SVD）为LLM压缩提供了一个有希望的解决方案。然而，现有的基于SVD的LLM压缩方法存在两个关键限制：截断较小的奇异值可能导致更高的压缩损失，并且在SVD截断后剩余模型参数的更新缺失。在这项工作中，我们提出了SVD-LLM，一种新的基于SVD的LLM压缩方法，解决了现有方法的限制。SVD-LLM采用了一种截断感知的数据白化策略，以确保奇异值和压缩损失之间的直接映射。此外，SVD-LLM采用一种逐层闭式模型参数更新策略，以弥补SVD截断引起的准确性降低。我们在总共11个数据集和七个m上评估了SVD-LLM。

    arXiv:2403.07378v1 Announce Type: new  Abstract: The advancements in Large Language Models (LLMs) have been hindered by their substantial sizes, which necessitate LLM compression methods for practical deployment. Singular Value Decomposition (SVD) offers a promising solution for LLM compression. However, state-of-the-art SVD-based LLM compression methods have two key limitations: truncating smaller singular values may lead to higher compression loss, and the lack of update on the remaining model parameters after SVD truncation. In this work, we propose SVD-LLM, a new SVD-based LLM compression method that addresses the limitations of existing methods. SVD-LLM incorporates a truncation-aware data whitening strategy to ensure a direct mapping between singular values and compression loss. Moreover, SVD-LLM adopts a layer-wise closed-form model parameter update strategy to compensate for accuracy degradation caused by SVD truncation. We evaluate SVD-LLM on a total of 11 datasets and seven m
    
[^3]: 在有噪声基础模型中学习

    Learning with Noisy Foundation Models

    [https://arxiv.org/abs/2403.06869](https://arxiv.org/abs/2403.06869)

    本文首次全面了解和分析了预训练数据集中的噪声性质，有效减轻其对下游任务影响。

    

    基础模型通常是在大规模数据集上进行预训练，然后通过调整来适应下游任务。然而，大规模预训练数据集往往无法获取或成本过高，可能包含标签噪声，这可能会对模型的泛化能力造成不利影响，并带来意想不到的风险。本文是首个全面了解和分析预训练数据集中噪声性质，并有效减轻其对下游任务影响的工作。具体而言，通过在合成有噪声的ImageNet-1K、YFCC15M和CC12M数据集上进行完全监督和图像-文本对比预训练的广泛实验，我们证明了，尽管预训练中的轻微噪声可以使同领域（ID）性能受益，即训练和测试数据共享类似分布，但它总是会破坏跨领域（OOD）性能，在那里训练和测试分布明显不同。

    arXiv:2403.06869v1 Announce Type: cross  Abstract: Foundation models are usually pre-trained on large-scale datasets and then adapted to downstream tasks through tuning. However, the large-scale pre-training datasets, often inaccessible or too expensive to handle, can contain label noise that may adversely affect the generalization of the model and pose unexpected risks. This paper stands out as the first work to comprehensively understand and analyze the nature of noise in pre-training datasets and then effectively mitigate its impacts on downstream tasks. Specifically, through extensive experiments of fully-supervised and image-text contrastive pre-training on synthetic noisy ImageNet-1K, YFCC15M, and CC12M datasets, we demonstrate that, while slight noise in pre-training can benefit in-domain (ID) performance, where the training and testing data share a similar distribution, it always deteriorates out-of-domain (OOD) performance, where training and testing distributions are signific
    
[^4]: 使用单词替换密码来越狱专有的大型语言模型

    Jailbreaking Proprietary Large Language Models using Word Substitution Cipher

    [https://arxiv.org/abs/2402.10601](https://arxiv.org/abs/2402.10601)

    本文使用密码技术编码了越狱提示，成功地绕过了大型语言模型对有害问题的检测，实验结果显示攻击成功率高达59.42%。

    

    大型语言模型（LLMs）遵循道德和伦理准则，但仍然容易受到名为Jailbreak的创意提示的影响，这些提示可以绕过对齐过程。然而，大多数越狱提示包含自然语言（主要是英语）中的有害问题，可以被LLMs自身检测到。本文提出了使用密码技术编码的越狱提示。我们首先在最先进的LLM，GPT-4上进行了一个试点研究，解码了使用各种密码技术加密的几个安全句子，发现简单的单词替换密码可以被最有效地解码。受此结果启发，我们使用这种编码技术来编写越狱提示。我们提供了将不安全单词映射到安全单词，并使用这些映射的单词提出不安全问题的映射。实验结果显示，我们提出的越狱攻击成功率（高达59.42%）。

    arXiv:2402.10601v1 Announce Type: cross  Abstract: Large Language Models (LLMs) are aligned to moral and ethical guidelines but remain susceptible to creative prompts called Jailbreak that can bypass the alignment process. However, most jailbreaking prompts contain harmful questions in the natural language (mainly English), which can be detected by the LLM themselves. In this paper, we present jailbreaking prompts encoded using cryptographic techniques. We first present a pilot study on the state-of-the-art LLM, GPT-4, in decoding several safe sentences that have been encrypted using various cryptographic techniques and find that a straightforward word substitution cipher can be decoded most effectively. Motivated by this result, we use this encoding technique for writing jailbreaking prompts. We present a mapping of unsafe words with safe words and ask the unsafe question using these mapped words. Experimental results show an attack success rate (up to 59.42%) of our proposed jailbrea
    
[^5]: 大型语言模型中提示工程的系统调查：技术和应用

    A Systematic Survey of Prompt Engineering in Large Language Models: Techniques and Applications

    [https://arxiv.org/abs/2402.07927](https://arxiv.org/abs/2402.07927)

    这篇调查论文系统概述了大型语言模型中提示工程的最新进展，探讨了提示工程的方法和技术，并说明了其在各种应用中的重要作用。

    

    提示工程已成为扩展大型语言模型（LLM）和视觉语言模型（VLM）能力的不可或缺的技术。该方法利用任务特定的指令（称为提示）在不修改核心模型参数的情况下增强模型的效果。提示允许将预训练模型无缝集成到下游任务中，仅根据给定的提示引发所需的模型行为，而不是更新模型参数。提示可以是提供上下文以指导模型的自然语言指令，也可以是调用相关知识的学习向量表示。这个新兴领域在各种应用中取得了成功，从问答到常识推理都有涉及。然而，对于多样的提示工程方法和技术缺乏系统的组织和理解。本调查论文通过提供对最近进展的结构化概述来填补这一空白。

    Prompt engineering has emerged as an indispensable technique for extending the capabilities of large language models (LLMs) and vision-language models (VLMs). This approach leverages task-specific instructions, known as prompts, to enhance model efficacy without modifying the core model parameters. Rather than updating the model parameters, prompts allow seamless integration of pre-trained models into downstream tasks by eliciting desired model behaviors solely based on the given prompt. Prompts can be natural language instructions that provide context to guide the model or learned vector representations that activate relevant knowledge. This burgeoning field has enabled success across various applications, from question-answering to commonsense reasoning. However, there remains a lack of systematic organization and understanding of the diverse prompt engineering methods and techniques. This survey paper addresses the gap by providing a structured overview of recent advancements in pro
    
[^6]: Permute-and-Flip：一种具有最佳鲁棒性和可加水印的LLMs解码器

    Permute-and-Flip: An optimally robust and watermarkable decoder for LLMs

    [https://arxiv.org/abs/2402.05864](https://arxiv.org/abs/2402.05864)

    提出了一种名为Permute-and-Flip（PF）解码器，其具有最佳的鲁棒性和质量-鲁棒性的 tradeoff，且比采样方法更好。还设计了一种针对PF解码器的水印方案，能够保持样本的分布不变，并实现任意低的假阳性率和高的召回率。实验证明PF解码器在困惑度方面明显优于朴素采样，为LLM解码提供了一种有希望的新方法。

    

    在本文中，我们提出了一种名为Permute-and-Flip（PF）解码器的新解码方法。它具有与标准采样解码器相似的鲁棒性特性，但在质量和鲁棒性的 tradeoff 上证明比采样方法更好，且永远不会差于任何其他解码器。同时，我们还设计了一种类似于Aaronson的Gumbel水印的加密水印方案，但是针对PF解码器而自然量身定制。该水印方案不改变样本的分布，同时允许任意低的假阳性率和高的召回率，只要生成的文本具有高熵。我们的实验证明，PF解码器（及其带有水印的对应物）在困惑度方面明显优于朴素采样（及其带有Gumbel水印的对应物），同时保持相同的鲁棒性（和可检测性），因此为LLM解码提供了一个有希望的新方法。代码可在https://github.com/XuandongZhao/pf-decoding找到。

    In this paper, we propose a new decoding method called Permute-and-Flip (PF) decoder. It enjoys robustness properties similar to the standard sampling decoder, but is provably up to 2x better in its quality-robustness tradeoff than sampling and never worse than any other decoder. We also design a cryptographic watermarking scheme analogous to Aaronson's Gumbel watermark, but naturally tailored for PF decoder. The watermarking scheme does not change the distribution to sample, while allowing arbitrarily low false positive rate and high recall whenever the generated text has high entropy. Our experiments show that the PF decoder (and its watermarked counterpart) significantly outperform(s) naive sampling (and it's Gumbel watermarked counterpart) in terms of perplexity, while retaining the same robustness (and detectability), hence making it a promising new approach for LLM decoding. The code is available at https://github.com/XuandongZhao/pf-decoding
    
[^7]: 生成证据（E2G）：一种单代理的两步提示用于上下文辅助和检索增强推理

    Evidence to Generate (E2G): A Single-agent Two-step Prompting for Context Grounded and Retrieval Augmented Reasoning. (arXiv:2401.05787v1 [cs.CL])

    [http://arxiv.org/abs/2401.05787](http://arxiv.org/abs/2401.05787)

    本研究提出了Evidence to Generate（E2G）框架，采用单代理、两步提示的方法来解决目前链式思维提示存在的限制，通过利用上下文中明确提及的思维序列作为证据，以更高的精确度和效率引导LLM的输出生成过程，实现更快、更可靠和更具上下文意识的推理。

    

    虽然思维链（CoT）提示革新了LLMs执行推理任务的方式，但其当前的方法和变体（例如，自一致性，反应，反射，思维树（ToT），累积推理（CR））存在缓慢、有限的上下文接地、幻象和不一致的输出等限制。为了克服这些挑战，我们引入了Evidence to Generate（E2G）这一新颖的单代理、两步提示框架。这种创新的方法利用“决策的证据”的力量，而不是未经验证的推理主张，首先专注于在上下文中明确提及的思维序列（中间步骤的系列），然后将其作为提取的证据，以更高的精确度和效率引导LLM的输出生成过程。这种简单而强大的方法解锁了像链式思维提示这样的潜力，为LLM中更快、更可靠和更具上下文意识的推理铺平了道路。

    While chain-of-thought (CoT) prompting has revolutionized how LLMs perform reasoning tasks, its current methods and variations (e.g, Self-consistency, ReACT, Reflexion, Tree-of-Thoughts (ToT), Cumulative Reasoning (CR)) suffer from limitations like slowness, limited context grounding, hallucination and inconsistent outputs. To overcome these challenges, we introduce Evidence to Generate (E2G), a novel single-agent, two-step prompting framework. Instead of unverified reasoning claims, this innovative approach leverages the power of "evidence for decision making" by first focusing exclusively on the thought sequences (the series of intermediate steps) explicitly mentioned in the context which then serve as extracted evidence, guiding the LLM's output generation process with greater precision and efficiency. This simple yet powerful approach unlocks the true potential of chain-of-thought like prompting, paving the way for faster, more reliable, and more contextually aware reasoning in LLM
    
[^8]: 利用大型语言模型进行集体决策

    Leveraging Large Language Models for Collective Decision-Making. (arXiv:2311.04928v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2311.04928](http://arxiv.org/abs/2311.04928)

    本论文提出了一种利用大型语言模型（LLM）促进集体决策的系统，通过管理对话和平衡个人偏好来提供满足成员需求的选项，实现高效协调并不断优化系统性能。

    

    在各种工作环境中，如会议安排、合作和项目规划中，集体决策是必不可少的，但由于个体偏好多样性、工作焦点不同和成员之间的权力动态等因素，常常具有挑战性。为了解决这个问题，我们提出了一种利用大型语言模型（LLM）来促进群体决策的系统，通过管理对话和平衡个人偏好来实现。我们的系统旨在从对话中提取个体偏好，并提出满足成员偏好的选项。我们特别将此系统应用于企业会议安排。我们利用LLM创建了合成员工配置文件，并模拟了大规模的对话，通过利用LLM评估系统表现来作为开展用户研究的新方法。我们的结果表明，系统能实现成员与LLM系统之间的高效协调，并随着时间的推移对其提出的选项进行改进和完善，确保优化系统性能。

    In various work contexts, such as meeting scheduling, collaborating, and project planning, collective decision-making is essential but often challenging due to diverse individual preferences, varying work focuses, and power dynamics among members. To address this, we propose a system leveraging Large Language Models (LLMs) to facilitate group decision-making by managing conversations and balancing preferences among individuals. Our system aims to extract individual preferences from conversations and suggest options that satisfy the preferences of the members. We specifically apply this system to corporate meeting scheduling. We create synthetic employee profiles and simulate conversations at scale, leveraging LLMs to evaluate the system performance as a novel approach to conducting a user study. Our results indicate efficient coordination with reduced interactions between the members and the LLM-based system. The system refines and improves its proposed options over time, ensuring that
    
[^9]: Thrust: 用外部知识自适应推动大型语言模型

    Thrust: Adaptively Propels Large Language Models with External Knowledge. (arXiv:2307.10442v1 [cs.CL])

    [http://arxiv.org/abs/2307.10442](http://arxiv.org/abs/2307.10442)

    本论文提出了一种实例级的自适应推动外部知识的方法，通过衡量大型语言模型的知识水平，并利用Thrust指标进行信息检索，实现更高的成本效益。

    

    尽管大规模预训练语言模型（PTLM）已被证明在其模型参数中编码了丰富的知识，但PTLM中的内在知识可能是不透明或静态的，因此需要外部知识。然而，现有的信息检索技术可能成本高昂，甚至可能引入噪音和误导性知识。为了解决这些挑战，我们提出了实例级的自适应推动外部知识（IAPEK），只有在必要时才进行检索。为了实现这一目标，我们提出了一种新的度量标准Thrust，利用少量已见实例的表示分布来衡量PTLM是否包含足够的知识来解决一个实例。大量实验证明，Thrust是衡量PTLM模型实例级知识能力的良好指标。此外，利用Thrust分数作为检索指标可以实现显著的成本效益，高于对外部知识的朴素使用。

    Although large-scale pre-trained language models (PTLMs) are shown to encode rich knowledge in their model parameters, the inherent knowledge in PTLMs can be opaque or static, making external knowledge necessary. However, the existing information retrieval techniques could be costly and may even introduce noisy and sometimes misleading knowledge. To address these challenges, we propose the instance-level adaptive propulsion of external knowledge (IAPEK), where we only conduct the retrieval when necessary. To achieve this goal, we propose measuring whether a PTLM contains enough knowledge to solve an instance with a novel metric, Thrust, which leverages the representation distribution of a small number of seen instances. Extensive experiments demonstrate that thrust is a good measurement of PTLM models' instance-level knowledgeability. Moreover, we can achieve significantly higher cost-efficiency with the Thrust score as the retrieval indicator than the naive usage of external knowledge
    
[^10]: 评估大规模语言模型在中文语法错误修正任务上的能力

    Evaluating the Capability of Large-scale Language Models on Chinese Grammatical Error Correction Task. (arXiv:2307.03972v1 [cs.CL])

    [http://arxiv.org/abs/2307.03972](http://arxiv.org/abs/2307.03972)

    本研究评估了大规模语言模型在中文语法错误修正任务上的表现，并发现存在过度修正的问题。此外，我们还发现在评估不同数据分布时，大型语言模型的性能有显著变化。

    

    大规模语言模型（LLMs）在各种自然语言处理（NLP）任务中表现出了令人瞩目的能力，并在最近受到了广泛关注。然而，一些研究表明，大型语言模型在英文语法错误修正任务中未能达到超越最先进模型的良好结果。本报告旨在探究大型语言模型在中文语法错误修正任务中的表现，并为未来的工作提供指导。我们在4个中文语法错误修正数据集上使用了3个不同模型规模的LLMs进行实验。我们的实验结果表明，LLMs在自动评估指标上的表现不及之前的最佳模型，因为存在过度修正的问题。此外，我们还发现LLMs在评估不同数据分布时的性能存在显著变化。我们的发现表明，需要进一步研究LLMs在中文语法错误修正任务上的应用。

    Large-scale language models (LLMs) has shown remarkable capability in various of Natural Language Processing (NLP) tasks and attracted lots of attention recently. However, some studies indicated that large language models fail to achieve promising result beyond the state-of-the-art models in English grammatical error correction (GEC) tasks. In this report, we aim to explore the how large language models perform on Chinese grammatical error correction tasks and provide guidance for future work. We conduct experiments with 3 different LLMs of different model scale on 4 Chinese GEC dataset. Our experimental results indicate that the performances of LLMs on automatic evaluation metrics falls short of the previous sota models because of the problem of over-correction. Furthermore, we also discover notable variations in the performance of LLMs when evaluated on different data distributions. Our findings demonstrates that further investigation is required for the application of LLMs on Chines
    
[^11]: Valley: 大型语言模型增强视频助手

    Valley: Video Assistant with Large Language model Enhanced abilitY. (arXiv:2306.07207v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2306.07207](http://arxiv.org/abs/2306.07207)

    本文介绍了一个名为Valley的视频助手，它是一个以大型语言模型增强的多模态基础模型，能够在一个通用框架内理解视频、图像和语言。

    

    大型语言模型(LLMs)以其卓越的会话能力，在各种应用中表现出色，并成为强大的AI助手。鉴于此，一个直观的问题是：我们能否利用LLMs的能力构建多模态的视觉应用AI助手？最近，已经开发了几个多模态模型来实现这个目的。它们通常预先训练一个适应模块来对齐视觉编码器和语言模型的语义，然后在指令跟随数据上进行微调。然而，尽管这个流程在图像和语言理解方面取得了成功，在视频和语言理解方面的有效性还没有得到广泛探索。在本文中，我们旨在开发一个能够在一个通用框架内理解视频、图像和语言的新型多模态基础模型。为了实现这一目标，我们引入了Valley，一个以大型语言模型增强的视频助手。

    Large language models (LLMs), with their remarkable conversational capabilities, have demonstrated impressive performance across various applications and have emerged as formidable AI assistants. In view of this, it raises an intuitive question: Can we harness the power of LLMs to build multimodal AI assistants for visual applications? Recently, several multi-modal models have been developed for this purpose. They typically pre-train an adaptation module to align the semantics of the vision encoder and language model, followed by fine-tuning on instruction-following data. However, despite the success of this pipeline in image and language understanding, its effectiveness in joint video and language understanding has not been widely explored. In this paper, we aim to develop a novel multi-modal foundation model capable of comprehending video, image, and language within a general framework. To achieve this goal, we introduce Valley, a Video Assistant with Large Language model Enhanced ab
    

