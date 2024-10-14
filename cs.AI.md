# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Soaring from 4K to 400K: Extending LLM's Context with Activation Beacon](https://rss.arxiv.org/abs/2401.03462) | 本论文提出了一种称为激活标志的新方法，它通过压缩LLM的激活状态，使其能够以有限的上下文窗口感知更长的上下文，同时保留了LLM在短上下文中的原始能力。这种方法具有竞争力的内存和时间效率，并通过多样化训练有效地支持不同上下文长度。 |
| [^2] | [Understanding Why Label Smoothing Degrades Selective Classification and How to Fix It](https://arxiv.org/abs/2403.14715) | LS方法在深度神经网络分类器训练中的标签平滑效果被发现会负面影响选择性分类，通过影响模型预测不确定性，此研究阐明了这一现象。 |
| [^3] | [Editing Massive Concepts in Text-to-Image Diffusion Models](https://arxiv.org/abs/2403.13807) | 提出了一种两阶段方法EMCID，用于在文本到图像扩散模型中进行海量概念编辑，通过内存优化和多层模型编辑来同时处理生成内容中的过时、受版权保护、不正确和有偏见的问题 |
| [^4] | [Customizing Segmentation Foundation Model via Prompt Learning for Instance Segmentation](https://arxiv.org/abs/2403.09199) | 提出了一种针对Segment Anything Model (SAM)的新颖方法，通过提示学习定制化实例分割，解决了在应用于定制化实例分割时面临的输入提示模糊性和额外训练需求的挑战 |
| [^5] | [Controllable Preference Optimization: Toward Controllable Multi-Objective Alignment](https://arxiv.org/abs/2402.19085) | 引入了可控偏好优化（CPO）方法，明确为不同目标指定偏好分数，从而引导模型生成符合需求的响应。 |
| [^6] | [Mitigating the Linguistic Gap with Phonemic Representations for Robust Multilingual Language Understanding](https://arxiv.org/abs/2402.14279) | 通过使用音素表示，本文提出了一种新颖的解决方案来减缓高资源语言和低资源语言之间的性能差距，并通过实证研究和理论分析证明了其有效性。 |
| [^7] | [SubIQ: Inverse Soft-Q Learning for Offline Imitation with Suboptimal Demonstrations](https://arxiv.org/abs/2402.13147) | 逆向软 Q 学习用于获得次优演示的离线模仿挑战了离线 IL 中有限支持专家演示的问题，并提出了一种解决方案以匹配次优演示集合的占用分布 |
| [^8] | [I Learn Better If You Speak My Language: Enhancing Large Language Model Fine-Tuning with Style-Aligned Response Adjustments](https://arxiv.org/abs/2402.11192) | 将微调过程中的实际响应风格与大型语言模型固有风格相匹配能够产生更好的学习结果，开发的方法通过最小程度地调整模型响应来避免过拟合。 |
| [^9] | [Exploiting Estimation Bias in Deep Double Q-Learning for Actor-Critic Methods](https://arxiv.org/abs/2402.09078) | 本文提出了两种创新方法，ExpD3 和 BE-TD3，用于解决和利用 Actor-Critic 方法中的估计偏差问题。实验证明这些算法在连续控制任务中比现有方法更高效。 |
| [^10] | [More Agents Is All You Need](https://arxiv.org/abs/2402.05120) | 大型语言模型的性能与代理数量成比例，通过简单的采样和投票方法可以进一步增强性能，这种方法与现有的复杂方法正交。 |
| [^11] | [DeLLMa: A Framework for Decision Making Under Uncertainty with Large Language Models](https://arxiv.org/abs/2402.02392) | DeLLMa是一个旨在提高不确定环境下决策精度的框架，通过多步骤的脚手架程序，借鉴决策理论和效用理论的原则，可以显著提高大型语言模型的决策性能。 |
| [^12] | [Explainable Attention for Few-shot Learning and Beyond.](http://arxiv.org/abs/2310.07800) | 本研究介绍了一种用于少样本学习的解释性注意力框架，旨在通过识别输入数据的关键部分来提高模型性能。 |
| [^13] | [Beyond Myopia: Learning from Positive and Unlabeled Data through Holistic Predictive Trends.](http://arxiv.org/abs/2310.04078) | 本文发现了从正样本和无标签数据中学习的一个重要观察：通过在每个训练迭代中重新采样正样本，可以获得强大的早期性能，并且正类和负类的预测趋势显示出不同的模式。 |
| [^14] | [Foundation Reinforcement Learning: towards Embodied Generalist Agents with Foundation Prior Assistance.](http://arxiv.org/abs/2310.02635) | 本研究提出了一种基于具身基础先验的基础强化学习框架，通过加速训练过程来提高样本效率。 |
| [^15] | [CDAN: Convolutional Dense Attention-guided Network for Low-light Image Enhancement.](http://arxiv.org/abs/2308.12902) | 本研究提出了一种名为CDAN的卷积稠密注意力引导网络，用于低光图像增强。该网络结合了自编码器架构、卷积和稠密块、注意力机制和跳跃连接，通过专门的后处理阶段进一步改善色彩平衡和对比度。与现有方法相比，在低光图像增强方面取得了显著的进展，展示了在各种具有挑战性的场景中的稳健性。 |
| [^16] | [Multimodal Machine Learning for Extraction of Theorems and Proofs in the Scientific Literature.](http://arxiv.org/abs/2307.09047) | 这篇论文提出了一种使用多模态机器学习的方法，通过对文本、字体特征和PDF的位图图像渲染进行融合分类，成功实现了从科学文献中提取定理和证明的目标。在文本模态方面，通过在11 GB的科学语料库上预训练一个新的语言模型，得到了与在160 GB预训练的模型相似的性能，同时具有更快的收敛速度和更少的微调数据要求。 |
| [^17] | [In Defense of Pure 16-bit Floating-Point Neural Networks.](http://arxiv.org/abs/2305.10947) | 本文探讨了纯16位浮点神经网络的被忽视的效率，提供了理论分析来探讨16位和32位模型的差异，并可以定量解释16位模型与其32位对应物之间的条件。 |
| [^18] | [PI-FL: Personalized and Incentivized Federated Learning.](http://arxiv.org/abs/2304.07514) | PI-FL是一种个性化的联邦学习解决方案，使用基于令牌的激励机制奖励个性化训练，可以在尊重客户端隐私的同时生成高质量的个性化模型。 |

# 详细

[^1]: 从4K到400K的飞跃：利用激活标志扩展LLM的上下文

    Soaring from 4K to 400K: Extending LLM's Context with Activation Beacon

    [https://rss.arxiv.org/abs/2401.03462](https://rss.arxiv.org/abs/2401.03462)

    本论文提出了一种称为激活标志的新方法，它通过压缩LLM的激活状态，使其能够以有限的上下文窗口感知更长的上下文，同时保留了LLM在短上下文中的原始能力。这种方法具有竞争力的内存和时间效率，并通过多样化训练有效地支持不同上下文长度。

    

    长上下文的利用对于LLM来说是一个巨大的挑战，因为它们有限的上下文窗口大小。尽管通过微调可以扩展上下文窗口，但这会导致训练和推理时间的显著成本，并对LLM的原始能力产生不利影响。在这项工作中，我们提出了一种名为激活标志的新方法，它将LLM的原始激活压缩成紧凑的形式，使LLM能够以有限的上下文窗口感知更长的上下文。激活标志被引入为插件模块，完全保留了LLM在短上下文中的原始能力。它与滑动窗口一起实时处理长的上下文，从而在训练和推理中实现了竞争力的内存和时间效率。激活标志是通过多样化压缩比的短序列数据进行训练的。得益于这种处理，它可以有效地学习支持不同上下文长度，实现小规模的训练。

    The utilization of long contexts poses a big challenge for LLMs due to their limited context window size. Although the context window can be extended through fine-tuning, it will result in a considerable cost at both training and inference time, and exert an unfavorable impact to the LLM's original capabilities. In this work, we propose a new method called Activation Beacon, which condenses LLM's raw activations into compact forms such that the LLM can perceive a longer context with a limited context window. Activation Beacon is introduced as a plug-in module, which fully preserves the LLM's original capability in short contexts. It works with the sliding window to streamingly process the long context, which leads to a competitive memory and time efficiency in both training and inference. Activation Beacon is trained with short-sequence data of diversified condensing ratios. Thanks to such a treatment, it can be effectively learned to support different context lengths with a small trai
    
[^2]: 理解为何标签平滑会降低选择性分类的效果以及如何解决这个问题

    Understanding Why Label Smoothing Degrades Selective Classification and How to Fix It

    [https://arxiv.org/abs/2403.14715](https://arxiv.org/abs/2403.14715)

    LS方法在深度神经网络分类器训练中的标签平滑效果被发现会负面影响选择性分类，通过影响模型预测不确定性，此研究阐明了这一现象。

    

    标签平滑（LS）是一种流行的深度神经网络分类器训练的正则化方法，因为它在提高测试准确性方面效果显著，并且实现简单。"硬"的one-hot标签通过将概率质量均匀分配给其他类别来进行"平滑化"，从而减少过度拟合。在这项工作中，我们揭示了LS如何负面影响选择性分类（SC）- 其目标是利用模型的预测不确定性来拒绝错误分类。我们首先在一系列任务和架构中从经验上证明LS会导致SC的一致性降级。然后，我们通过分析logit级别的梯度来解释这一点，表明LS通过在错误概率低时更加正则化最大logit，而在错误概率高时更少正则化，加剧了过度自信和低自信。这阐明了以前报道的强分类器在SC中性能不佳的实验结果。

    arXiv:2403.14715v1 Announce Type: cross  Abstract: Label smoothing (LS) is a popular regularisation method for training deep neural network classifiers due to its effectiveness in improving test accuracy and its simplicity in implementation. "Hard" one-hot labels are "smoothed" by uniformly distributing probability mass to other classes, reducing overfitting. In this work, we reveal that LS negatively affects selective classification (SC) - where the aim is to reject misclassifications using a model's predictive uncertainty. We first demonstrate empirically across a range of tasks and architectures that LS leads to a consistent degradation in SC. We then explain this by analysing logit-level gradients, showing that LS exacerbates overconfidence and underconfidence by regularising the max logit more when the probability of error is low, and less when the probability of error is high. This elucidates previously reported experimental results where strong classifiers underperform in SC. We
    
[^3]: 在文本到图像扩散模型中编辑海量概念

    Editing Massive Concepts in Text-to-Image Diffusion Models

    [https://arxiv.org/abs/2403.13807](https://arxiv.org/abs/2403.13807)

    提出了一种两阶段方法EMCID，用于在文本到图像扩散模型中进行海量概念编辑，通过内存优化和多层模型编辑来同时处理生成内容中的过时、受版权保护、不正确和有偏见的问题

    

    文本到图像扩散模型存在生成过时的、受版权保护的、不正确的和有偏见的内容的风险。虽然先前的方法已在小规模上缓解了这些问题，但在更大规模的实际场景中同时处理它们是至关重要的。我们提出了一个两阶段方法，即编辑海量概念扩散模型（EMCID）。第一阶段通过文本对齐损失和扩散噪声预测损失实现对每个单独概念的内存优化的双自蒸馏。第二阶段进行海量概念编辑，采用多层、封闭形式的模型编辑。我们进一步提出了一个名为ImageNet概念编辑基准（ICEB）的综合基准，用于评估针对T2I模型的海量概念编辑，包括两个子任务：自由形式提示和海量概念类别，以及广泛的评估指标。在我们提出的基准和先前的基准上进行了大量实验展示

    arXiv:2403.13807v1 Announce Type: cross  Abstract: Text-to-image diffusion models suffer from the risk of generating outdated, copyrighted, incorrect, and biased content. While previous methods have mitigated the issues on a small scale, it is essential to handle them simultaneously in larger-scale real-world scenarios. We propose a two-stage method, Editing Massive Concepts In Diffusion Models (EMCID). The first stage performs memory optimization for each individual concept with dual self-distillation from text alignment loss and diffusion noise prediction loss. The second stage conducts massive concept editing with multi-layer, closed form model editing. We further propose a comprehensive benchmark, named ImageNet Concept Editing Benchmark (ICEB), for evaluating massive concept editing for T2I models with two subtasks, free-form prompts, massive concept categories, and extensive evaluation metrics. Extensive experiments conducted on our proposed benchmark and previous benchmarks demo
    
[^4]: 通过提示学习定制化分割基础模型进行实例分割

    Customizing Segmentation Foundation Model via Prompt Learning for Instance Segmentation

    [https://arxiv.org/abs/2403.09199](https://arxiv.org/abs/2403.09199)

    提出了一种针对Segment Anything Model (SAM)的新颖方法，通过提示学习定制化实例分割，解决了在应用于定制化实例分割时面临的输入提示模糊性和额外训练需求的挑战

    

    最近，通过大规模数据集训练的基础模型吸引了相当多的关注，并在计算机视觉领域得到积极探讨。在这些模型中，Segment Anything Model (SAM)因其在图像分割任务中的泛化能力和灵活性而脱颖而出，通过基于提示的对象掩模生成取得了显著进展。然而，尽管SAM具有这些优势，在应用于定制化实例分割时（对特定对象或在训练数据中通常不存在的独特环境中进行分割），SAM面临两个关键限制：1）输入提示中固有的模糊性，2）为实现最佳分割需要大量额外训练。为解决这些挑战，我们提出了一种新颖的方法，即通过提示学习定制化实例分割，针对SAM进行了定制。我们的方法包含一个提示学习模块（PLM），可以调整输入。

    arXiv:2403.09199v1 Announce Type: cross  Abstract: Recently, foundation models trained on massive datasets to adapt to a wide range of domains have attracted considerable attention and are actively being explored within the computer vision community. Among these, the Segment Anything Model (SAM) stands out for its remarkable progress in generalizability and flexibility for image segmentation tasks, achieved through prompt-based object mask generation. However, despite its strength, SAM faces two key limitations when applied to customized instance segmentation that segments specific objects or those in unique environments not typically present in the training data: 1) the ambiguity inherent in input prompts and 2) the necessity for extensive additional training to achieve optimal segmentation. To address these challenges, we propose a novel method, customized instance segmentation via prompt learning tailored to SAM. Our method involves a prompt learning module (PLM), which adjusts inpu
    
[^5]: 可控偏好优化：朝着可控多目标对齐方向发展

    Controllable Preference Optimization: Toward Controllable Multi-Objective Alignment

    [https://arxiv.org/abs/2402.19085](https://arxiv.org/abs/2402.19085)

    引入了可控偏好优化（CPO）方法，明确为不同目标指定偏好分数，从而引导模型生成符合需求的响应。

    

    人工智能中的对齐工作旨在追求模型响应与人类偏好和价值的一致性。本文引入了可控偏好优化（CPO）方法，明确为不同目标指定偏好分数，从而引导模型生成符合需求的响应。实验分析表明，经过对齐的模型可以提供符合各种偏好的响应。

    arXiv:2402.19085v1 Announce Type: new  Abstract: Alignment in artificial intelligence pursues the consistency between model responses and human preferences as well as values. In practice, the multifaceted nature of human preferences inadvertently introduces what is known as the "alignment tax" -a compromise where enhancements in alignment within one objective (e.g.,harmlessness) can diminish performance in others (e.g.,helpfulness). However, existing alignment techniques are mostly unidirectional, leading to suboptimal trade-offs and poor flexibility over various objectives. To navigate this challenge, we argue the prominence of grounding LLMs with evident preferences. We introduce controllable preference optimization (CPO), which explicitly specifies preference scores for different objectives, thereby guiding the model to generate responses that meet the requirements. Our experimental analysis reveals that the aligned models can provide responses that match various preferences among t
    
[^6]: 使用音素表示减缓语言差异，实现稳健的多语言理解

    Mitigating the Linguistic Gap with Phonemic Representations for Robust Multilingual Language Understanding

    [https://arxiv.org/abs/2402.14279](https://arxiv.org/abs/2402.14279)

    通过使用音素表示，本文提出了一种新颖的解决方案来减缓高资源语言和低资源语言之间的性能差距，并通过实证研究和理论分析证明了其有效性。

    

    为了改善多语言理解，通常需要在训练阶段使用多种语言，依赖复杂的训练技术，并且在高资源语言和低资源语言之间存在显著的性能差距。我们假设语言之间的性能差距受到这些语言之间的语言差异的影响，并通过使用音素表示（具体来说，将音素作为输入标记输入到语言模型中，而不是子词）提供了一种新颖的解决方案，以实现稳健的多语言建模。我们通过三个跨语言任务的定量证据展示了音素表示的有效性，这进一步得到了对跨语言性能差距的理论分析的证明。

    arXiv:2402.14279v1 Announce Type: cross  Abstract: Approaches to improving multilingual language understanding often require multiple languages during the training phase, rely on complicated training techniques, and -- importantly -- struggle with significant performance gaps between high-resource and low-resource languages. We hypothesize that the performance gaps between languages are affected by linguistic gaps between those languages and provide a novel solution for robust multilingual language modeling by employing phonemic representations (specifically, using phonemes as input tokens to LMs rather than subwords). We present quantitative evidence from three cross-lingual tasks that demonstrate the effectiveness of phonemic representation, which is further justified by a theoretical analysis of the cross-lingual performance gap.
    
[^7]: SubIQ: 逆向软 Q 学习用于获得次优演示的离线模仿

    SubIQ: Inverse Soft-Q Learning for Offline Imitation with Suboptimal Demonstrations

    [https://arxiv.org/abs/2402.13147](https://arxiv.org/abs/2402.13147)

    逆向软 Q 学习用于获得次优演示的离线模仿挑战了离线 IL 中有限支持专家演示的问题，并提出了一种解决方案以匹配次优演示集合的占用分布

    

    我们考虑了离线模仿学习（IL），旨在从专家演示中模仿专家的行为，而无需与环境进行进一步交互。在离线 IL 中的一个主要挑战是处理仅涵盖状态-动作空间的一小部分的专家演示的有限支持。我们考虑离线 IL，其中专家演示受到限制，但是由更大规模的次优演示集合补充。大部分现有的用于此设置的离线 IL 方法基于行为克隆或分布匹配，其目的是将模仿策略的占用分布与专家策略的占用分布匹配。这种方法往往存在过拟合问题，因为专家演示有限，无法准确表示任何占用分布。另一方面，由于次优演示集合规模更大，有很高的可能性

    arXiv:2402.13147v1 Announce Type: cross  Abstract: We consider offline imitation learning (IL), which aims to mimic the expert's behavior from its demonstration without further interaction with the environment. One of the main challenges in offline IL is dealing with the limited support of expert demonstrations that cover only a small fraction of the state-action spaces. In this work, we consider offline IL, where expert demonstrations are limited but complemented by a larger set of sub-optimal demonstrations of lower expertise levels. Most of the existing offline IL methods developed for this setting are based on behavior cloning or distribution matching, where the aim is to match the occupancy distribution of the imitation policy with that of the expert policy. Such an approach often suffers from over-fitting, as expert demonstrations are limited to accurately represent any occupancy distribution. On the other hand, since sub-optimal sets are much larger, there is a high chance that 
    
[^8]: 如果你讲我的语言，我会更好地学习：使用风格对齐响应调整增强大型语言模型微调

    I Learn Better If You Speak My Language: Enhancing Large Language Model Fine-Tuning with Style-Aligned Response Adjustments

    [https://arxiv.org/abs/2402.11192](https://arxiv.org/abs/2402.11192)

    将微调过程中的实际响应风格与大型语言模型固有风格相匹配能够产生更好的学习结果，开发的方法通过最小程度地调整模型响应来避免过拟合。

    

    使用小数据集为特定任务微调大型语言模型(LLMs)是一个普遍遇到的但复杂的挑战。在有限的示例上过多拟合可能会对模型的泛化能力和保留原始技能产生负面影响。我们的研究探讨了在微调过程中地实际响应风格的影响。我们发现将地实际响应风格与LLM固有风格匹配会产生更好的学习结果。基于这一观点，我们开发了一种方法，最小程度地修改LLM的现有响应以更正错误，使用这些调整后的响应作为训练目标。这种技术能够实现与模型固有响应风格一致的精确更正，维护模型的核心能力，从而避免过多拟合。我们的研究结果表明，这种方法不仅提高了LLM的特定任务准确性，而且关键地

    arXiv:2402.11192v1 Announce Type: cross  Abstract: Fine-tuning large language models (LLMs) with a small data set for particular tasks is a widely encountered yet complex challenge. The potential for overfitting on a limited number of examples can negatively impact the model's ability to generalize and retain its original skills. Our research explores the impact of the style of ground-truth responses during the fine-tuning process. We found that matching the ground-truth response style with the LLM's inherent style results in better learning outcomes. Building on this insight, we developed a method that minimally alters the LLM's pre-existing responses to correct errors, using these adjusted responses as training targets. This technique enables precise corrections in line with the model's native response style, safeguarding the model's core capabilities and thus avoid overfitting. Our findings show that this approach not only improves the LLM's task-specific accuracy but also crucially
    
[^9]: 在 Actor-Critic 方法中利用估计偏差的深度双 Q-Learning 的探索

    Exploiting Estimation Bias in Deep Double Q-Learning for Actor-Critic Methods

    [https://arxiv.org/abs/2402.09078](https://arxiv.org/abs/2402.09078)

    本文提出了两种创新方法，ExpD3 和 BE-TD3，用于解决和利用 Actor-Critic 方法中的估计偏差问题。实验证明这些算法在连续控制任务中比现有方法更高效。

    

    本文介绍了在强化学习领域中创新的方法，重点是解决和利用连续控制任务中 Actor-Critic 方法中的估计偏差问题，使用了深度双 Q-Learning。我们提出了两种新的算法：Expectile Delayed Deep Deterministic Policy Gradient (ExpD3) 和 Bias Exploiting - Twin Delayed Deep Deterministic Policy Gradient (BE-TD3)。ExpD3 旨在通过单一的 Q 估计来减少过度估计偏差，并在计算效率和性能之间提供平衡，而 BE-TD3 则旨在在训练过程中动态选择最有利的估计偏差。我们进行了大量的实验，在各种连续控制任务中展示了我们方法的有效性。我们展示了这些算法在与 TD3 等现有方法相比，尤其是在估计偏差显著影响学习的环境中，可以匹敌或超越它们。这些结果凸显了估计偏差对学习的重要性。

    arXiv:2402.09078v1 Announce Type: cross Abstract: This paper introduces innovative methods in Reinforcement Learning (RL), focusing on addressing and exploiting estimation biases in Actor-Critic methods for continuous control tasks, using Deep Double Q-Learning. We propose two novel algorithms: Expectile Delayed Deep Deterministic Policy Gradient (ExpD3) and Bias Exploiting - Twin Delayed Deep Deterministic Policy Gradient (BE-TD3). ExpD3 aims to reduce overestimation bias with a single $Q$ estimate, offering a balance between computational efficiency and performance, while BE-TD3 is designed to dynamically select the most advantageous estimation bias during training. Our extensive experiments across various continuous control tasks demonstrate the effectiveness of our approaches. We show that these algorithms can either match or surpass existing methods like TD3, particularly in environments where estimation biases significantly impact learning. The results underline the importance of
    
[^10]: 更多的代理就是你所需要的

    More Agents Is All You Need

    [https://arxiv.org/abs/2402.05120](https://arxiv.org/abs/2402.05120)

    大型语言模型的性能与代理数量成比例，通过简单的采样和投票方法可以进一步增强性能，这种方法与现有的复杂方法正交。

    

    我们发现，仅通过一种采样和投票的方法，大型语言模型(Large Language Models, LLMs)的性能与实例化的代理数量成比例。此外，这种方法对已有的复杂方法进一步增强LLMs是正交的，而增强的程度与任务的困难程度相关。我们进行了广泛的实验，验证了我们的发现，并研究了能够促进其发生的属性。我们的代码公开在以下网址: \url{https://anonymous.4open.science/r/more_agent_is_all_you_need}

    We find that, simply via a sampling-and-voting method, the performance of large language models (LLMs) scales with the number of agents instantiated. Also, this method is orthogonal to existing complicated methods to further enhance LLMs, while the degree of enhancement is correlated to the task difficulty. We conduct comprehensive experiments on a wide range of LLM benchmarks to verify the presence of our finding, and to study the properties that can facilitate its occurrence. Our code is publicly available at: \url{https://anonymous.4open.science/r/more_agent_is_all_you_need}.
    
[^11]: DeLLMa:一个用于大型语言模型下决策的框架

    DeLLMa: A Framework for Decision Making Under Uncertainty with Large Language Models

    [https://arxiv.org/abs/2402.02392](https://arxiv.org/abs/2402.02392)

    DeLLMa是一个旨在提高不确定环境下决策精度的框架，通过多步骤的脚手架程序，借鉴决策理论和效用理论的原则，可以显著提高大型语言模型的决策性能。

    

    大型语言模型（LLMs）在商业、工程和医学等领域被广泛应用，这些领域往往面临决策不确定性的问题，这是一个关键但具有挑战性的任务。本文表明，在决策问题上直接使用LLMs往往效果较差，尤其是在问题复杂性增加时。为了克服这个限制，我们提出了DeLLMa（Decision-making Large Language Model assistant）框架，旨在提高不确定环境下的决策精度。DeLLMa包括一个多步骤的脚手架程序，借鉴了决策理论和效用理论的原则，提供了一个最优的、可审计的决策过程。我们在涉及真实农业和金融数据的决策环境中验证了我们的框架。结果表明，DeLLMa可以显著提高LLMs的决策性能，准确性可提高高达40%以上。

    Large language models (LLMs) are increasingly used across society, including in domains like business, engineering, and medicine. These fields often grapple with decision-making under uncertainty, a critical yet challenging task. In this paper, we show that directly prompting LLMs on these types of decision-making problems yields poor results, especially as the problem complexity increases. To overcome this limitation, we propose DeLLMa (Decision-making Large Language Model assistant), a framework designed to enhance decision-making accuracy in uncertain environments. DeLLMa involves a multi-step scaffolding procedure, drawing upon principles from decision theory and utility theory, to provide an optimal and human-auditable decision-making process. We validate our framework on decision-making environments involving real agriculture and finance data. Our results show that DeLLMa can significantly improve LLM decision-making performance, achieving up to a 40% increase in accuracy over co
    
[^12]: 解释性注意力用于少样本学习及其它领域

    Explainable Attention for Few-shot Learning and Beyond. (arXiv:2310.07800v1 [cs.AI])

    [http://arxiv.org/abs/2310.07800](http://arxiv.org/abs/2310.07800)

    本研究介绍了一种用于少样本学习的解释性注意力框架，旨在通过识别输入数据的关键部分来提高模型性能。

    

    注意力机制在增强学习模型中显示出了有希望的潜力，它可以通过识别输入数据中显著的部分来提高模型的性能。这在数据收集和标记方面存在挑战，导致训练样本有限的情况下尤其有价值。受人类认知过程启发，我们认为，如果将AI基线暴露于原始数据的关键部分而不是整个输入数据集，类似于人类的感知，那么它的性能可能会更准确、更可靠。然而，选择这些信息性数据部分的任务，即硬注意力寻找，是一个巨大的挑战。在少量训练样本的情况下，现有的研究很难找到这些信息性区域，原因是大量的训练参数无法从有限的样本中有效学习。在本研究中，我们介绍了一种实用的框架，用于实现可解释的硬注意力寻找。

    Attention mechanisms have exhibited promising potential in enhancing learning models by identifying salient portions of input data. This is particularly valuable in scenarios where limited training samples are accessible due to challenges in data collection and labeling. Drawing inspiration from human recognition processes, we posit that an AI baseline's performance could be more accurate and dependable if it is exposed to essential segments of raw data rather than the entire input dataset, akin to human perception. However, the task of selecting these informative data segments, referred to as hard attention finding, presents a formidable challenge. In situations with few training samples, existing studies struggle to locate such informative regions due to the large number of training parameters that cannot be effectively learned from the available limited samples. In this study, we introduce a novel and practical framework for achieving explainable hard attention finding, specifically
    
[^13]: 超越近视：通过整体预测趋势从正样本和无标签数据中学习

    Beyond Myopia: Learning from Positive and Unlabeled Data through Holistic Predictive Trends. (arXiv:2310.04078v1 [cs.LG])

    [http://arxiv.org/abs/2310.04078](http://arxiv.org/abs/2310.04078)

    本文发现了从正样本和无标签数据中学习的一个重要观察：通过在每个训练迭代中重新采样正样本，可以获得强大的早期性能，并且正类和负类的预测趋势显示出不同的模式。

    

    在许多现实世界的应用中，从正样本和无标签数据(PUL)中学习二元分类器是至关重要的，特别是在验证负样本困难的情况下。尽管最近的PUL方法在实验性能方面表现出色，但由于缺乏负标签，仍然存在积累误差和增加估计偏差等挑战。在本文中，我们揭示了PUL中一个引人注目但长期被忽视的观察结果：\textit{在每个训练迭代中重新采样正样本以确保正样本和无标签样本之间的平衡分布会导致强的早期性能。此外，正类和负类的预测趋势显示出不同的模式。}具体而言，无标签负样本的得分(输出概率)持续下降，而无标签正样本的得分显示出大致混乱的趋势。我们创新地采用了一种整体方法，而不是在个别时间段内进行分类。

    Learning binary classifiers from positive and unlabeled data (PUL) is vital in many real-world applications, especially when verifying negative examples is difficult. Despite the impressive empirical performance of recent PUL methods, challenges like accumulated errors and increased estimation bias persist due to the absence of negative labels. In this paper, we unveil an intriguing yet long-overlooked observation in PUL: \textit{resampling the positive data in each training iteration to ensure a balanced distribution between positive and unlabeled examples results in strong early-stage performance. Furthermore, predictive trends for positive and negative classes display distinctly different patterns.} Specifically, the scores (output probability) of unlabeled negative examples consistently decrease, while those of unlabeled positive examples show largely chaotic trends. Instead of focusing on classification within individual time frames, we innovatively adopt a holistic approach, inte
    
[^14]: 强化学习基础：朝向具有基础先验辅助的具身通用智能体

    Foundation Reinforcement Learning: towards Embodied Generalist Agents with Foundation Prior Assistance. (arXiv:2310.02635v1 [cs.RO])

    [http://arxiv.org/abs/2310.02635](http://arxiv.org/abs/2310.02635)

    本研究提出了一种基于具身基础先验的基础强化学习框架，通过加速训练过程来提高样本效率。

    

    最近人们已经表明，从互联网规模的数据中进行大规模预训练是构建通用模型的关键，正如在NLP中所见。为了构建具身通用智能体，我们和许多其他研究者假设这种基础先验也是不可或缺的组成部分。然而，目前尚不清楚如何以适当的具体形式表示这些具身基础先验，以及它们应该如何在下游任务中使用。在本文中，我们提出了一组直观有效的具身先验，包括基础策略、价值和成功奖励。所提出的先验是基于目标条件的MDP。为了验证其有效性，我们实例化了一个由这些先验辅助的演员-评论家方法，称之为基础演员-评论家（FAC）。我们将我们的框架命名为基础强化学习（FRL），因为它完全依赖于具身基础先验来进行探索、学习和强化。FRL的好处有三个。(1)样本效率高。通过基础先验加速训练过程，减少样本使用量。

    Recently, people have shown that large-scale pre-training from internet-scale data is the key to building generalist models, as witnessed in NLP. To build embodied generalist agents, we and many other researchers hypothesize that such foundation prior is also an indispensable component. However, it is unclear what is the proper concrete form to represent those embodied foundation priors and how they should be used in the downstream task. In this paper, we propose an intuitive and effective set of embodied priors that consist of foundation policy, value, and success reward. The proposed priors are based on the goal-conditioned MDP. To verify their effectiveness, we instantiate an actor-critic method assisted by the priors, called Foundation Actor-Critic (FAC). We name our framework as Foundation Reinforcement Learning (FRL), since it completely relies on embodied foundation priors to explore, learn and reinforce. The benefits of FRL are threefold. (1) Sample efficient. With foundation p
    
[^15]: CDAN: 用于低光图像增强的卷积稠密注意力引导网络

    CDAN: Convolutional Dense Attention-guided Network for Low-light Image Enhancement. (arXiv:2308.12902v1 [cs.CV])

    [http://arxiv.org/abs/2308.12902](http://arxiv.org/abs/2308.12902)

    本研究提出了一种名为CDAN的卷积稠密注意力引导网络，用于低光图像增强。该网络结合了自编码器架构、卷积和稠密块、注意力机制和跳跃连接，通过专门的后处理阶段进一步改善色彩平衡和对比度。与现有方法相比，在低光图像增强方面取得了显著的进展，展示了在各种具有挑战性的场景中的稳健性。

    

    低光图像以不足的照明为特征，面临清晰度减弱、颜色暗淡和细节减少的挑战。低光图像增强是计算机视觉中的一个重要任务，旨在通过改善亮度、对比度和整体感知质量来纠正这些问题，从而促进准确的分析和解释。本文介绍了一种新颖的解决方案：卷积稠密注意力引导网络（CDAN），用于增强低光图像。CDAN将自编码器架构与卷积和稠密块相结合，配合注意力机制和跳跃连接。该架构确保了有效的信息传递和特征学习。此外，专门的后处理阶段可以进一步改善色彩平衡和对比度。与低光图像增强领域的最新成果相比，我们的方法取得了显著的进展，并展示了在各种具有挑战性的场景中的稳健性。

    Low-light images, characterized by inadequate illumination, pose challenges of diminished clarity, muted colors, and reduced details. Low-light image enhancement, an essential task in computer vision, aims to rectify these issues by improving brightness, contrast, and overall perceptual quality, thereby facilitating accurate analysis and interpretation. This paper introduces the Convolutional Dense Attention-guided Network (CDAN), a novel solution for enhancing low-light images. CDAN integrates an autoencoder-based architecture with convolutional and dense blocks, complemented by an attention mechanism and skip connections. This architecture ensures efficient information propagation and feature learning. Furthermore, a dedicated post-processing phase refines color balance and contrast. Our approach demonstrates notable progress compared to state-of-the-art results in low-light image enhancement, showcasing its robustness across a wide range of challenging scenarios. Our model performs 
    
[^16]: 用于从科学文献中提取定理和证明的多模态机器学习

    Multimodal Machine Learning for Extraction of Theorems and Proofs in the Scientific Literature. (arXiv:2307.09047v1 [cs.AI])

    [http://arxiv.org/abs/2307.09047](http://arxiv.org/abs/2307.09047)

    这篇论文提出了一种使用多模态机器学习的方法，通过对文本、字体特征和PDF的位图图像渲染进行融合分类，成功实现了从科学文献中提取定理和证明的目标。在文本模态方面，通过在11 GB的科学语料库上预训练一个新的语言模型，得到了与在160 GB预训练的模型相似的性能，同时具有更快的收敛速度和更少的微调数据要求。

    

    数学领域的学术文章中包含定理、命题等数学陈述及其证明。从文章的PDF表示中提取它们需要理解科学文本以及视觉和基于字体的指示符。我们将这个问题作为一种多模态分类问题，使用文本、字体特征和PDF的位图图像渲染作为不同的模态。在本文中，我们提出了一种基于多模态机器学习的方法，用于提取类定理环境和证明，该方法基于通过单一单模态分类器提取的特征的后期融合，考虑文档中块的顺序。对于文本模态，我们在一个11 GB的科学语料库上预训练了一个新的语言模型；实验证明，我们的任务与一个在160 GB上预训练的（RoBERTa）模型相比拥有类似的性能，在要求更少的微调数据的同时收敛更快。

    Scholarly articles in mathematical fields feature mathematical statements such as theorems, propositions, etc., as well as their proofs. Extracting them from the PDF representation of the articles requires understanding of scientific text along with visual and font-based indicators. We pose this problem as a multimodal classification problem using text, font features, and bitmap image rendering of the PDF as different modalities. In this paper we propose a multimodal machine learning approach for extraction of theorem-like environments and proofs, based on late fusion of features extracted by individual unimodal classifiers, taking into account the sequential succession of blocks in the document. For the text modality, we pretrain a new language model on a 11 GB scientific corpus; experiments shows similar performance for our task than a model (RoBERTa) pretrained on 160 GB, with faster convergence while requiring much less fine-tuning data. Font-based information relies on training a 
    
[^17]: 关于纯16位浮点神经网络的辩护

    In Defense of Pure 16-bit Floating-Point Neural Networks. (arXiv:2305.10947v1 [cs.LG])

    [http://arxiv.org/abs/2305.10947](http://arxiv.org/abs/2305.10947)

    本文探讨了纯16位浮点神经网络的被忽视的效率，提供了理论分析来探讨16位和32位模型的差异，并可以定量解释16位模型与其32位对应物之间的条件。

    

    减少编码神经网络权重和激活所需的位数是非常可取的，因为它可以加快神经网络的训练和推理时间，同时减少内存消耗。因此，这一领域的研究引起了广泛关注，以开发利用更低精度计算的神经网络，比如混合精度训练。有趣的是，目前不存在纯16位浮点设置的方法。本文揭示了纯16位浮点神经网络被忽视的效率。我们通过提供全面的理论分析来探讨造成16位和32位模型的差异的因素。我们规范化了浮点误差和容忍度的概念，从而可以定量解释16位模型与其32位对应物之间密切逼近结果的条件。这种理论探索提供了新的视角。

    Reducing the number of bits needed to encode the weights and activations of neural networks is highly desirable as it speeds up their training and inference time while reducing memory consumption. For these reasons, research in this area has attracted significant attention toward developing neural networks that leverage lower-precision computing, such as mixed-precision training. Interestingly, none of the existing approaches has investigated pure 16-bit floating-point settings. In this paper, we shed light on the overlooked efficiency of pure 16-bit floating-point neural networks. As such, we provide a comprehensive theoretical analysis to investigate the factors contributing to the differences observed between 16-bit and 32-bit models. We formalize the concepts of floating-point error and tolerance, enabling us to quantitatively explain the conditions under which a 16-bit model can closely approximate the results of its 32-bit counterpart. This theoretical exploration offers perspect
    
[^18]: PI-FL：个性化和激励联邦学习

    PI-FL: Personalized and Incentivized Federated Learning. (arXiv:2304.07514v1 [cs.LG])

    [http://arxiv.org/abs/2304.07514](http://arxiv.org/abs/2304.07514)

    PI-FL是一种个性化的联邦学习解决方案，使用基于令牌的激励机制奖励个性化训练，可以在尊重客户端隐私的同时生成高质量的个性化模型。

    

    个性化联邦学习已被广泛应用于应对非独立同分布数据异质性的挑战。主要问题是考虑来自客户端的个性化过程以保护其自治权。允许客户端参与个性化联邦学习决策变得重要，因为存在隐私和安全问题，客户端可能无法自由共享生成良好质量个性化模型所必需的私人信息。此外，具有高质量数据和资源的客户端不愿意在没有合理激励的情况下参与联邦学习过程。在本文中，我们提出了PI-FL，这是一个一次性个性化解决方案，配合一个基于令牌的激励机制，奖励个性化训练。PI-FL优于其他最先进的方法，并且可以在尊重客户端隐私的同时生成高质量的个性化模型。

    Personalized FL has been widely used to cater to heterogeneity challenges with non-IID data. A primary obstacle is considering the personalization process from the client's perspective to preserve their autonomy. Allowing the clients to participate in personalized FL decisions becomes significant due to privacy and security concerns, where the clients may not be at liberty to share private information necessary for producing good quality personalized models. Moreover, clients with high-quality data and resources are reluctant to participate in the FL process without reasonable incentive. In this paper, we propose PI-FL, a one-shot personalization solution complemented by a token-based incentive mechanism that rewards personalized training. PI-FL outperforms other state-of-the-art approaches and can generate good-quality personalized models while respecting clients' privacy.
    

