# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Understanding Emergent Abilities of Language Models from the Loss Perspective](https://arxiv.org/abs/2403.15796) | 本文从损失角度重新定义了语言模型的突现能力，发现具有相同预训练损失的模型在不同任务上表现相似，而当预训练损失低于特定阈值时，模型将展现出突现能力。 |
| [^2] | [Mind the Error! Detection and Localization of Instruction Errors in Vision-and-Language Navigation](https://arxiv.org/abs/2403.10700) | 提出了一个新的基准数据集，首次引入了各种类型的指令错误，考虑到潜在的人类原因，以评估连续环境中 VLN 系统的健壮性 |
| [^3] | [The Power of Noise: Toward a Unified Multi-modal Knowledge Graph Representation Framework](https://arxiv.org/abs/2403.06832) | 提出了一种利用噪声掩模的Transformer-based架构SNAG方法，实现了多模态知识图表示中实体嵌入的最先进性能 |
| [^4] | [Do Large Language Models Mirror Cognitive Language Processing?](https://arxiv.org/abs/2402.18023) | 本文提出了一种新颖方法，通过将大型语言模型（LLMs）的表示与人类认知信号联系起来，评估LLMs模拟认知语言处理的效果。 |
| [^5] | [SelectIT: Selective Instruction Tuning for Large Language Models via Uncertainty-Aware Self-Reflection](https://arxiv.org/abs/2402.16705) | SelectIT通过利用大型语言模型本身的能力和基于不确定性的方法，提出了一种无需额外资源的高效选择指导调整数据集的方法，进而提升了模型的能力。 |

# 详细

[^1]: 从损失角度理解语言模型的突现能力

    Understanding Emergent Abilities of Language Models from the Loss Perspective

    [https://arxiv.org/abs/2403.15796](https://arxiv.org/abs/2403.15796)

    本文从损失角度重新定义了语言模型的突现能力，发现具有相同预训练损失的模型在不同任务上表现相似，而当预训练损失低于特定阈值时，模型将展现出突现能力。

    

    近期研究质疑了传统认为语言模型的突现能力仅存在于大模型中的观点。这种怀疑源自两点观察：1）较小的模型也能展现出对突现能力的高性能；2）质疑用于测量这些能力的不连续性指标。本文提议从预训练损失的角度研究突现能力，而非模型大小或训练计算。我们展示了具有相同预训练损失但不同模型和数据大小的模型，在各种下游任务上表现相同。我们还发现，当某一模型的预训练损失低于特定阈值时，在某些任务上表现出突现能力，而不论指标的连续性如何；而在达到该阈值之前，其性能仍保持在随机猜测水平。这启发我们重新定义突现能力为那些......

    arXiv:2403.15796v1 Announce Type: cross  Abstract: Recent studies have put into question the belief that emergent abilities in language models are exclusive to large models. This skepticism arises from two observations: 1) smaller models can also exhibit high performance on emergent abilities and 2) there is doubt on the discontinuous metrics used to measure these abilities. In this paper, we propose to study emergent abilities in the lens of pre-training loss, instead of model size or training compute. We demonstrate that the models with the same pre-training loss, but different model and data sizes, generate the same performance on various downstream tasks. We also discover that a model exhibits emergent abilities on certain tasks -- regardless of the continuity of metrics -- when its pre-training loss falls below a specific threshold. Before reaching this threshold, its performance remains at the level of random guessing. This inspires us to redefine emergent abilities as those that
    
[^2]: 注意错误！检测和定位视觉与语言导航中的指令错误

    Mind the Error! Detection and Localization of Instruction Errors in Vision-and-Language Navigation

    [https://arxiv.org/abs/2403.10700](https://arxiv.org/abs/2403.10700)

    提出了一个新的基准数据集，首次引入了各种类型的指令错误，考虑到潜在的人类原因，以评估连续环境中 VLN 系统的健壮性

    

    Vision-and-Language Navigation in Continuous Environments (VLN-CE) 是一项直观且具有挑战性的体验智能任务。代理人被要求通过执行一系列低级动作、遵循一系列自然语言指令来导航到目标目标。所有文献中的 VLN-CE 方法都假设语言指令是准确的。然而，在实践中，人类给出的指令可能由于不准确的记忆或混淆而包含空间环境描述中的错误。当前 VLN-CE 基准没有解决这种情况，使得 VLN-CE 中的最新方法在面对来自人类用户的错误指令时变得脆弱。我们首次提出了一个引入各种类型指令错误考虑潜在人类原因的新型基准数据集。该基准数据集为连续环境中的 VLN 系统的健壮性提供了宝贵的见解。我们观察到 noticeable...

    arXiv:2403.10700v1 Announce Type: cross  Abstract: Vision-and-Language Navigation in Continuous Environments (VLN-CE) is one of the most intuitive yet challenging embodied AI tasks. Agents are tasked to navigate towards a target goal by executing a set of low-level actions, following a series of natural language instructions. All VLN-CE methods in the literature assume that language instructions are exact. However, in practice, instructions given by humans can contain errors when describing a spatial environment due to inaccurate memory or confusion. Current VLN-CE benchmarks do not address this scenario, making the state-of-the-art methods in VLN-CE fragile in the presence of erroneous instructions from human users. For the first time, we propose a novel benchmark dataset that introduces various types of instruction errors considering potential human causes. This benchmark provides valuable insight into the robustness of VLN systems in continuous environments. We observe a noticeable 
    
[^3]: 噪声的力量：朝着统一的多模态知识图表示框架

    The Power of Noise: Toward a Unified Multi-modal Knowledge Graph Representation Framework

    [https://arxiv.org/abs/2403.06832](https://arxiv.org/abs/2403.06832)

    提出了一种利用噪声掩模的Transformer-based架构SNAG方法，实现了多模态知识图表示中实体嵌入的最先进性能

    

    多模态预训练的进展凸显出鲁棒的多模态知识图（MMKG）表示学习框架的必要性。此框架对于在规模上将结构化知识整合到多模态大型语言模型（LLMs）中至关重要，旨在减轻知识误解和多模态幻觉等问题。在这项工作中，为了评估模型准确嵌入MMKG中的实体的能力，我们专注于两个广泛研究的任务：多模态知识图完成（MKGC）和多模态实体对齐（MMEA）。在此基础上，我们提出了一种新颖的SNAG方法，该方法利用基于Transformer的架构，并配备了模态级噪声掩模，以在知识图中鲁棒地集成多模态实体特征。通过为MKGC和MMEA都引入特定的训练目标，我们的方法在总共十个数据集上（三个用于MKGC和...

    arXiv:2403.06832v1 Announce Type: cross  Abstract: The advancement of Multi-modal Pre-training highlights the necessity for a robust Multi-Modal Knowledge Graph (MMKG) representation learning framework. This framework is crucial for integrating structured knowledge into multi-modal Large Language Models (LLMs) at scale, aiming to alleviate issues like knowledge misconceptions and multi-modal hallucinations. In this work, to evaluate models' ability to accurately embed entities within MMKGs, we focus on two widely researched tasks: Multi-modal Knowledge Graph Completion (MKGC) and Multi-modal Entity Alignment (MMEA). Building on this foundation, we propose a novel SNAG method that utilizes a Transformer-based architecture equipped with modality-level noise masking for the robust integration of multi-modal entity features in KGs. By incorporating specific training objectives for both MKGC and MMEA, our approach achieves SOTA performance across a total of ten datasets (three for MKGC and 
    
[^4]: 大型语言模型是否反映认知语言处理？

    Do Large Language Models Mirror Cognitive Language Processing?

    [https://arxiv.org/abs/2402.18023](https://arxiv.org/abs/2402.18023)

    本文提出了一种新颖方法，通过将大型语言模型（LLMs）的表示与人类认知信号联系起来，评估LLMs模拟认知语言处理的效果。

    

    大型语言模型（LLMs）在文本理解和逻辑推理方面展现出卓越能力，甚至在许多认知任务中实现甚至超越人类水平的表现。由于LLMs是从人类语言认知的大量文本产出中训练出来的，自然而然地会问LLMs是否反映认知语言处理，或LLMs在多大程度上类似于认知语言处理。本文提出了一种新颖的方法，用于连接LLMs表征和人类认知信号，以评估LLMs如何有效地模拟认知语言处理。我们采用表征相似性分析（RSA）来衡量16种主流LLMs与大脑fMRI信号之间的对齐程度。我们在实验中探讨了各种因素（例如模型规模、对齐训练、指导附加）对LLM-大脑对齐的影响。实验结果表明，模型规模与正相关

    arXiv:2402.18023v1 Announce Type: new  Abstract: Large language models (LLMs) have demonstrated remarkable capabilities in text comprehension and logical reasoning, achiving or even surpassing human-level performance in numerous cognition tasks. As LLMs are trained from massive textual outputs of human language cognition, it is natural to ask whether LLMs mirror cognitive language processing. Or to what extend LLMs resemble cognitive language processing? In this paper, we propose a novel method that bridge between LLM representations and human cognition signals to evaluate how effectively LLMs simulate cognitive language processing. We employ Representational Similarity Analysis (RSA) to mearsure the alignment between 16 mainstream LLMs and fMRI signals of the brain. We empirically investigate the impact of a variety of factors (e.g., model scaling, alignment training, instruction appending) on such LLM-brain alignment. Experimental results indicate that model scaling is positively cor
    
[^5]: SelectIT: 通过基于不确定性的自我反思实现大型语言模型的选择性指导调整

    SelectIT: Selective Instruction Tuning for Large Language Models via Uncertainty-Aware Self-Reflection

    [https://arxiv.org/abs/2402.16705](https://arxiv.org/abs/2402.16705)

    SelectIT通过利用大型语言模型本身的能力和基于不确定性的方法，提出了一种无需额外资源的高效选择指导调整数据集的方法，进而提升了模型的能力。

    

    指导调整（IT）对于调整大型语言模型（LLMs）以适应人类中心交互至关重要。最近的进展表明，精心选择一小部分高质量的IT数据可以显着提高LLMs的性能。尽管如此，常见方法通常依赖于额外的模型或数据集，这增加了成本并限制了广泛采用。在这项工作中，我们提出了一种新颖的方法，称为SelectIT，它利用LLM本身的基本能力。具体来说，我们利用LLMs中固有的不确定性，更有效地选择高质量的IT数据，而无需额外资源。此外，我们介绍了一种新颖的IT数据集，名为选择性羊驼（Selective Alpaca），通过将SelectIT应用于Alpaca-GPT4数据集而创建。实证结果表明，使用选择性羊驼进行IT可以极大地提升模型性能。SelectIT的稳健性也得到了验证。

    arXiv:2402.16705v1 Announce Type: new  Abstract: Instruction tuning (IT) is crucial to tailoring large language models (LLMs) towards human-centric interactions. Recent advancements have shown that the careful selection of a small, high-quality subset of IT data can significantly enhance the performance of LLMs. Despite this, common approaches often rely on additional models or data sets, which increases costs and limits widespread adoption. In this work, we propose a novel approach, termed SelectIT, that capitalizes on the foundational capabilities of the LLM itself. Specifically, we exploit the intrinsic uncertainty present in LLMs to more effectively select high-quality IT data, without the need for extra resources. Furthermore, we introduce a novel IT dataset, the Selective Alpaca, created by applying SelectIT to the Alpaca-GPT4 dataset. Empirical results demonstrate that IT using Selective Alpaca leads to substantial model ability enhancement. The robustness of SelectIT has also b
    

