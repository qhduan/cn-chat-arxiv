# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Towards Explainable Adjudicative Variance: Quantifying Judicial Discretion via Gated Multi-Task Learning](https://arxiv.org/abs/2606.27069) | 提出了一种法官感知的门控多任务学习架构，通过细粒度结果分类和动态门控机制，有效分离案件事实与司法自由裁量权，显著提升法律结果预测的可解释性和准确性。 |
| [^2] | [AgentX: Towards Agent-Driven Self-Iteration of Industrial Recommender Systems](https://arxiv.org/abs/2606.26859) | AgentX是一个已部署的多智能体系统，通过自主生成、实施、评估和学习推荐实验，实现了工业推荐系统从依赖人工到自我迭代的范式转变，从而打破了创新瓶颈。 |
| [^3] | [\textsc{DiARC}: Distinguishing Positive and Negative Samples Helps Improving ARC-like Reasoning Ability of Large Language Models](https://arxiv.org/abs/2606.26530) | 本文提出DiARC方法，通过构建正负样本偏好对来提升大型语言模型在类ARC任务中的推理能力，弥补了仅依赖正样本监督的不足。 |
| [^4] | [Assert, don't describe: Linguistic features that shift LLM reasoning about animal welfare](https://arxiv.org/abs/2606.26104) | 研究发现，在微调语言模型时，使用断言式确定性、道德词汇和情感词汇等特征会显著增强模型对动物福利的支持，而模糊语言和具体感官描述则会削弱这一立场。 |
| [^5] | [Helpfulness Hurts: Domain-Dependent Degradation of Mid-Trained Compassion Values Under Post-Training](https://arxiv.org/abs/2606.26102) | 论文发现，后训练中的有益性训练（如SFT和GRPO）会显著削弱语言模型在中期训练中获得的动物同情价值观，而编码训练的影响较小，且这一现象在多个数据集和训练范式上得到验证。 |

# 详细

[^1]: 迈向可解释的裁量差异：通过门控多任务学习量化司法自由裁量权

    Towards Explainable Adjudicative Variance: Quantifying Judicial Discretion via Gated Multi-Task Learning

    [https://arxiv.org/abs/2606.27069](https://arxiv.org/abs/2606.27069)

    提出了一种法官感知的门控多任务学习架构，通过细粒度结果分类和动态门控机制，有效分离案件事实与司法自由裁量权，显著提升法律结果预测的可解释性和准确性。

    

    法律结果预测必须将客观案件事实与裁判背景区分开来。基于案情的裁决依赖于事实证据，而程序性处理则可能取决于司法自由裁量权。我们提出了一种法官感知的门控多任务学习架构，明确建模了这一区别。我们引入了一个细粒度的结果分类法来监督编码器，强制执行一种结构正则化，从而分离出不同的语义路径。这种精细化的法律课程使我们的门控融合机制能够动态调整对法官身份的依赖。我们在13,937份英国就业法庭判决上评估了该方法。我们将我们的设计与基于Gemma-4 26B-A4B骨干网络的监督微调（SFT）进行了基准对比，其中法官身份和分类法作为提示令牌或自回归输出目标注入。当通过单一自回归通道强制组合时，这两个上下文信号只能产生微弱的协同效应。相比之下，我们的方法显著提升了预测性能，并提供了关于法官自由裁量权如何影响判决的可解释性见解。

    arXiv:2606.27069v1 Announce Type: new  Abstract: Legal outcome prediction must disentangle objective case facts from adjudicative context. Merit-based rulings rely on factual evidence while technical disposals may hinge on judicial discretion. We propose a Judge-Aware Gated Multi-Task Learning architecture that explicitly models this distinction. We introduce a fine-grained outcome taxonomy to supervise the encoder, enforcing a structural regularization that disentangles distinct semantic pathways. This granular legal curriculum enables our Gated Fusion mechanism to dynamically modulate reliance on judge identity. We evaluate our approach on 13,937 UK Employment Tribunal decisions. We benchmark our design against supervised fine-tuning (SFT) of a Gemma-4 26B-A4B backbone, in which judge identity and the taxonomy are injected as prompt tokens or autoregressive output targets. The two contextual signals compose only weakly when forced through a single autoregressive channel. In contrast,
    
[^2]: AgentX：迈向工业推荐系统智能体驱动的自我迭代

    AgentX: Towards Agent-Driven Self-Iteration of Industrial Recommender Systems

    [https://arxiv.org/abs/2606.26859](https://arxiv.org/abs/2606.26859)

    AgentX是一个已部署的多智能体系统，通过自主生成、实施、评估和学习推荐实验，实现了工业推荐系统从依赖人工到自我迭代的范式转变，从而打破了创新瓶颈。

    

    arXiv:2606.26859v1 公告类型：新 摘要：推荐算法的迭代正从依赖工程师的手工过程向工业化研究循环转变，但这种转变仍被结构性执行瓶颈所阻碍：从想法到上线的周期仍然依赖人类工程师提出假设、修改生产代码、启动A/B实验并归因线上结果。因此，创新规模与人力成正比，而非与证据、计算资源和积累的实验知识复合增长。我们提出AgentX，一个已投入生产的、从根本上重构这一生产函数的多智能体系统。AgentX作为一个自我进化的开发引擎运作：它能自主生成、实现、评估并学习推荐实验，其规模和速度远超任何人工工作流所能维持的水平。该系统在一个闭环中编排四个紧密耦合的阶段。一个头脑风暴智能体从历史数据中综合证据。

    arXiv:2606.26859v1 Announce Type: new  Abstract: Recommendation algorithm iteration is moving from an artisanal, engineer-bound process toward an industrialized research loop, but this transition remains blocked by a structural execution bottleneck: the idea-to-launch cycle still depends on human engineers to generate hypotheses, modify production code, launch A/B experiments, and attribute online results. Innovation therefore scales linearly with headcount rather than compounding with evidence, compute, and accumulated experimental knowledge. We present AgentX, a production-deployed multi-agent system that fundamentally restructures this production function. AgentX operates as a self-evolving development engine: it autonomously generates, implements, evaluates, and learns from recommendation experiments at a scale and pace that no manual workflow can sustain.   The system orchestrates four tightly coupled stages in a closed loop. A Brainstorm Agent synthesizes evidence from historical
    
[^3]: \textsc{DiARC}：区分正负样本有助于提升大型语言模型的类ARC推理能力

    \textsc{DiARC}: Distinguishing Positive and Negative Samples Helps Improving ARC-like Reasoning Ability of Large Language Models

    [https://arxiv.org/abs/2606.26530](https://arxiv.org/abs/2606.26530)

    本文提出DiARC方法，通过构建正负样本偏好对来提升大型语言模型在类ARC任务中的推理能力，弥补了仅依赖正样本监督的不足。

    

    抽象与推理语料库（ARC；\citealp{chollet2019measure}）包含需要从有限网格样本中总结模式并预测输出网格的任务。近年来，许多基于大型语言模型的方法尝试将其转化为基于文本的推理任务。然而，基于开源模型的方法通常效果不佳，而依赖闭源模型的方法成本过高。当前工作主要集中在数据增强上，即构建类ARC数据以进行更全面的监督微调。在本研究中，我们认为解决类ARC问题不仅需要正样本监督，还需要通过区分负样本来提升模型的推理能力。为此，我们借鉴偏好对齐的思想，提出了\textsc{DiARC}方法，该方法构建偏好对，使模型能够区分正负样本。

    arXiv:2606.26530v1 Announce Type: cross  Abstract: The Abstraction and Reasoning Corpus (ARC;~\citealp{chollet2019measure}) contains tasks that require summarizing patterns from limited grid samples and predicting output grids. Recently, many large language model based approaches have attempted to transform it into a text-based reasoning task. However, methods based on open-source models have generally yielded unsatisfactory results, while those relying on closed-source models are too costly. Current efforts mainly focus on data augmentation, constructing ARC-like data for more comprehensive supervised fine-tuning. In this work, we argue that solving ARC-like problems requires not only \textit{positive} sample supervision but also the ability to improve model reasoning by distinguishing \textit{negative} samples. To this end, we draw on the idea of preference alignment and propose \textsc{DiARC}, a method that constructs preference pairs to enable the model to distinguish between them.
    
[^4]: 断言，而非描述：改变大语言模型对动物福利推理的语言特征

    Assert, don't describe: Linguistic features that shift LLM reasoning about animal welfare

    [https://arxiv.org/abs/2606.26104](https://arxiv.org/abs/2606.26104)

    研究发现，在微调语言模型时，使用断言式确定性、道德词汇和情感词汇等特征会显著增强模型对动物福利的支持，而模糊语言和具体感官描述则会削弱这一立场。

    

    arXiv:2606.26104v1 公告类型：交叉 摘要：动物福利倡导者撰写了大量文本，而这些文本越来越多地用于训练语言模型，随后数百万用户会向这些模型询问动物福利问题。通过在一个预留的动物福利基准测试上使用词汇匹配的立场对比探针，我们测量了十种语言特征分别作为微调数据使用时，如何改变Llama-3.2-1B对支持动物福利推理的偏好。其中八种特征产生了统计上显著的转变。七种特征使模型朝向更强的支持动物福利推理方向移动：断言式确定性、明确的道德词汇、情感词汇、评价性主张、叙事结构、描绘的伤害严重程度以及即时时间框架。两种特征使模型朝相反方向移动：模糊语言和具体感官描述都削弱了支持动物福利的立场。第一人称视角没有统计上显著的影响。对于任何撰写旨在影响语言模型的动物福利文本的人来说，实际建议是使用断言式确定性、明确的道德词汇和情感词汇，同时避免模糊语言和具体感官描述。

    arXiv:2606.26104v1 Announce Type: cross  Abstract: Animal-welfare advocates produce a lot of writing, and increasingly that writing trains the language models that millions of people then ask about animal welfare. Using vocabulary-matched stance-contrast probes on a held-out animal-welfare benchmark, we measure how each of ten linguistic features changes Llama-3.2-1B's preference for pro-animal-welfare reasoning when used as fine-tuning data. Eight of the ten features produce statistically significant shifts. Seven move the model toward stronger pro-animal-welfare reasoning: assertive certainty, explicit moral vocabulary, emotion words, evaluative claims, narrative structure, depicted harm severity, and immediate temporal framing. Two move it the other way: hedged language and concrete sensory description both dilute the pro-animal-welfare stance. First-person perspective has no statistically significant effect. The practical recommendation for anyone writing animal-welfare text that m
    
[^5]: 有益性有害：后训练中基于领域的中期训练同情价值观退化

    Helpfulness Hurts: Domain-Dependent Degradation of Mid-Trained Compassion Values Under Post-Training

    [https://arxiv.org/abs/2606.26102](https://arxiv.org/abs/2606.26102)

    论文发现，后训练中的有益性训练（如SFT和GRPO）会显著削弱语言模型在中期训练中获得的动物同情价值观，而编码训练的影响较小，且这一现象在多个数据集和训练范式上得到验证。

    

    摘要：标准后训练流程采用监督微调（SFT）和强化学习（RL）来使语言模型具有有益性，但这些过程可能会无意中削弱预训练期间灌输的价值观。我们研究了后训练数据的领域是否会对基于同情导向的合成数据进行中期训练的Llama 3.1 8B模型中的动物同情价值观保持产生差异化影响，使用了SFT（通过Dolly-15k的有益性与通过Magicoder-110K的编码）和GRPO（通过RLHFlow的有益性与通过Magicoder的编码），并在动物伤害基准（AHB 2.2）和不确定性下的道德推理基准（MORU）上进行评估。与编码训练相比，有益性训练在AHB上显著降低了动物同情（SFT：35.7%对65.2%；GRPO：18.7%对32.0%），这一结果在两个独立的有益性数据集和两种训练范式上得到复现。在英文MORU项目中，有益性训练也降低了通用道德推理。

    arXiv:2606.26102v1 Announce Type: cross  Abstract: Standard post-training pipelines apply supervised fine-tuning (SFT) and reinforcement learning (RL) to make language models helpful, but these processes may inadvertently degrade values instilled during pre-training. We investigate whether the domain of post-training data differentially affects the retention of animal compassion values in a Llama 3.1 8B model mid-trained on compassion-oriented synthetic data, using both SFT (helpfulness via Dolly-15k vs. coding via Magicoder-110K) and GRPO (helpfulness via RLHFlow vs. coding via Magicoder), evaluated on the Animal Harm Benchmark (AHB 2.2) and MORU benchmark (Moral Reasoning Under Uncertainty). Helpfulness training significantly degrades animal compassion relative to coding training on AHB (SFT: 35.7% vs. 65.2%; GRPO: 18.7% vs. 32.0%), replicating across two independent helpfulness datasets and two training paradigms. On English MORU items, helpfulness training degrades general moral re
    

