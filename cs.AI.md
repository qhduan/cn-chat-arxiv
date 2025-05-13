# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Integrating Large Language Models in Causal Discovery: A Statistical Causal Approach](https://rss.arxiv.org/abs/2402.01454) | 本文提出了一种在因果发现中集成大型语言模型的方法，通过将统计因果提示与知识增强相结合，可以使统计因果发现结果接近真实情况并进一步改进结果。 |
| [^2] | [LLM Agent Operating System](https://arxiv.org/abs/2403.16971) | 提出了一种将大型语言模型嵌入操作系统中的LLM代理操作系统，旨在优化资源分配、促进代理间上下文切换、实现并发执行以及为代理提供工具服务。 |
| [^3] | [Can Interpretability Layouts Influence Human Perception of Offensive Sentences?](https://arxiv.org/abs/2403.05581) | 本文通过用户研究探讨了机器学习解释布局是否会影响参与者对包含仇恨言论句子的评价，结果表明解释布局在触发参与者提供纠正性反馈和评估模型方面具有优势 |
| [^4] | [Gemini: A Family of Highly Capable Multimodal Models](https://arxiv.org/abs/2312.11805) | Gemini家族是一系列在图像、音频、视频和文本理解方面表现出色的多模态模型，其中最具能力的Gemini Ultra模型在30个基准测试中推进了技术前沿，并改进了所有20个多模态基准测试的技术状态。 |
| [^5] | [Fundamental Limits of Membership Inference Attacks on Machine Learning Models.](http://arxiv.org/abs/2310.13786) | 本文探讨了机器学习模型上成员推断攻击的基本限制，包括推导了效果和成功率的统计量，并提供了几种情况下的界限。这使得我们能够根据样本数量和其他结构参数推断潜在攻击的准确性。 |
| [^6] | [Towards Understanding Sycophancy in Language Models.](http://arxiv.org/abs/2310.13548) | 这项研究探讨了强化学习从人类反馈中训练高质量AI助手的技术，发现这种方法可能导致模型在回答问题时过于谄媚，而不是坦诚，通过分析人类偏好数据得出了这一结论。 |
| [^7] | [Review helps learn better: Temporal Supervised Knowledge Distillation.](http://arxiv.org/abs/2307.00811) | 本文提出了一种基于时间的监督知识蒸馏方法，利用评论来帮助学生网络的学习。通过提取学生网络在不同训练阶段的时空特征，并通过动态目标进行训练，实现了对学生网络中旧知识的优化和利用，从而提高了网络的训练性能。 |
| [^8] | [Clickbait Detection via Large Language Models.](http://arxiv.org/abs/2306.09597) | 本文研究了大型语言模型在点击诱骗检测上的性能，结果表明LLM无法取得最佳结果且不能仅通过标题实现满意的检测。 |
| [^9] | [Decentralized Adversarial Training over Graphs.](http://arxiv.org/abs/2303.13326) | 本文研究了在图上的去中心化对抗性训练，利用扩散学习的方法，开发了一种对抗性训练框架，增强了多个代理的鲁棒性以对抗攻击。 |

# 详细

[^1]: 在因果发现中集成大型语言模型: 一种统计因果方法

    Integrating Large Language Models in Causal Discovery: A Statistical Causal Approach

    [https://rss.arxiv.org/abs/2402.01454](https://rss.arxiv.org/abs/2402.01454)

    本文提出了一种在因果发现中集成大型语言模型的方法，通过将统计因果提示与知识增强相结合，可以使统计因果发现结果接近真实情况并进一步改进结果。

    

    在实际的统计因果发现（SCD）中，将领域专家知识作为约束嵌入到算法中被广泛接受，因为这对于创建一致有意义的因果模型是重要的，尽管识别背景知识的挑战被认可。为了克服这些挑战，本文提出了一种新的因果推断方法，即通过将LLM的“统计因果提示（SCP）”与SCD方法和基于知识的因果推断（KBCI）相结合，对SCD进行先验知识增强。实验证明，GPT-4可以使LLM-KBCI的输出与带有LLM-KBCI的先验知识的SCD结果接近真实情况，如果GPT-4经历了SCP，那么SCD的结果还可以进一步改善。而且，即使LLM不含有数据集的信息，LLM仍然可以通过其背景知识来改进SCD。

    In practical statistical causal discovery (SCD), embedding domain expert knowledge as constraints into the algorithm is widely accepted as significant for creating consistent meaningful causal models, despite the recognized challenges in systematic acquisition of the background knowledge. To overcome these challenges, this paper proposes a novel methodology for causal inference, in which SCD methods and knowledge based causal inference (KBCI) with a large language model (LLM) are synthesized through "statistical causal prompting (SCP)" for LLMs and prior knowledge augmentation for SCD. Experiments have revealed that GPT-4 can cause the output of the LLM-KBCI and the SCD result with prior knowledge from LLM-KBCI to approach the ground truth, and that the SCD result can be further improved, if GPT-4 undergoes SCP. Furthermore, it has been clarified that an LLM can improve SCD with its background knowledge, even if the LLM does not contain information on the dataset. The proposed approach
    
[^2]: LLM Agent Operating System

    LLM Agent Operating System

    [https://arxiv.org/abs/2403.16971](https://arxiv.org/abs/2403.16971)

    提出了一种将大型语言模型嵌入操作系统中的LLM代理操作系统，旨在优化资源分配、促进代理间上下文切换、实现并发执行以及为代理提供工具服务。

    

    arXiv:2403.16971v1 公告类型: 跨领域 摘要: 部署大型语言模型（LLM）智能代理存在诸多挑战，会损害它们的效率和功效。其中包括代理请求在LLM上的次优调度和资源分配、在代理和LLM之间交互时保持上下文的困难，以及将具有不同能力和专业化的异构代理集成在一起的复杂性。代理数量和复杂性的快速增加进一步加剧了这些问题，通常会导致资源瓶颈和次优资源利用。受到这些挑战的启发，本文提出了AIOS，一种LLM代理操作系统，它将大型语言模型嵌入操作系统（OS）中。具体地，AIOS旨在优化资源分配，促进代理之间的上下文切换，实现代理的并发执行，为代理提供工具服务。

    arXiv:2403.16971v1 Announce Type: cross  Abstract: The integration and deployment of large language model (LLM)-based intelligent agents have been fraught with challenges that compromise their efficiency and efficacy. Among these issues are sub-optimal scheduling and resource allocation of agent requests over the LLM, the difficulties in maintaining context during interactions between agent and LLM, and the complexities inherent in integrating heterogeneous agents with different capabilities and specializations. The rapid increase of agent quantity and complexity further exacerbates these issues, often leading to bottlenecks and sub-optimal utilization of resources. Inspired by these challenges, this paper presents AIOS, an LLM agent operating system, which embeds large language model into operating systems (OS). Specifically, AIOS is designed to optimize resource allocation, facilitate context switch across agents, enable concurrent execution of agents, provide tool service for agents
    
[^3]: 解释布局可以影响人对冒犯性句子的感知吗？

    Can Interpretability Layouts Influence Human Perception of Offensive Sentences?

    [https://arxiv.org/abs/2403.05581](https://arxiv.org/abs/2403.05581)

    本文通过用户研究探讨了机器学习解释布局是否会影响参与者对包含仇恨言论句子的评价，结果表明解释布局在触发参与者提供纠正性反馈和评估模型方面具有优势

    

    本文进行了一项用户研究，评估三种机器学习（ML）解释布局是否会影响参与者评估包含仇恨言论的句子时的观点，重点关注“厌恶女性”和“种族主义”两类。鉴于文献中存在分歧的结论，我们通过统计和定性分析问卷调查回应的实证证据，探讨在在线社区中使用ML解释性的优势。广义可加模型估计参与者的评级，融合了组内设计和组间设计。尽管我们的统计分析表明，没有任何解释布局显著影响参与者的观点，但我们的定性分析表明ML解释性的优势：1）触发参与者在他们的观点与模型之间存在差异时提供纠正性反馈，2）提供评估模型的见解

    arXiv:2403.05581v1 Announce Type: cross  Abstract: This paper conducts a user study to assess whether three machine learning (ML) interpretability layouts can influence participants' views when evaluating sentences containing hate speech, focusing on the "Misogyny" and "Racism" classes. Given the existence of divergent conclusions in the literature, we provide empirical evidence on using ML interpretability in online communities through statistical and qualitative analyses of questionnaire responses. The Generalized Additive Model estimates participants' ratings, incorporating within-subject and between-subject designs. While our statistical analysis indicates that none of the interpretability layouts significantly influences participants' views, our qualitative analysis demonstrates the advantages of ML interpretability: 1) triggering participants to provide corrective feedback in case of discrepancies between their views and the model, and 2) providing insights to evaluate a model's 
    
[^4]: Gemini：一系列高性能多模态模型

    Gemini: A Family of Highly Capable Multimodal Models

    [https://arxiv.org/abs/2312.11805](https://arxiv.org/abs/2312.11805)

    Gemini家族是一系列在图像、音频、视频和文本理解方面表现出色的多模态模型，其中最具能力的Gemini Ultra模型在30个基准测试中推进了技术前沿，并改进了所有20个多模态基准测试的技术状态。

    

    本报告介绍了一种新的多模态模型系列Gemini，展示出在图像、音频、视频和文本理解方面的显著能力。Gemini系列包括Ultra、Pro和Nano尺寸，适用于从复杂推理任务到设备内存受限应用的各种应用场景。在广泛的基准测试中，我们最具能力的Gemini Ultra模型在32个基准测试中的30个中推进了技术前沿 - 显著地是第一个在被广泛研究的考试基准测试MMLU上实现人类专家水平表现的模型，并在我们研究的每一个20个多模态基准测试中改进了技术前沿。我们相信Gemini系列在跨模态推理和语言理解方面的新能力将能够支持各种用例。我们讨论了负责任地向用户提供Gemini模型的训练后和部署方法，包括使用服务。

    arXiv:2312.11805v2 Announce Type: replace-cross  Abstract: This report introduces a new family of multimodal models, Gemini, that exhibit remarkable capabilities across image, audio, video, and text understanding. The Gemini family consists of Ultra, Pro, and Nano sizes, suitable for applications ranging from complex reasoning tasks to on-device memory-constrained use-cases. Evaluation on a broad range of benchmarks shows that our most-capable Gemini Ultra model advances the state of the art in 30 of 32 of these benchmarks - notably being the first model to achieve human-expert performance on the well-studied exam benchmark MMLU, and improving the state of the art in every one of the 20 multimodal benchmarks we examined. We believe that the new capabilities of the Gemini family in cross-modal reasoning and language understanding will enable a wide variety of use cases. We discuss our approach toward post-training and deploying Gemini models responsibly to users through services includi
    
[^5]: 机器学习模型的成员推断攻击的基本限制

    Fundamental Limits of Membership Inference Attacks on Machine Learning Models. (arXiv:2310.13786v1 [stat.ML])

    [http://arxiv.org/abs/2310.13786](http://arxiv.org/abs/2310.13786)

    本文探讨了机器学习模型上成员推断攻击的基本限制，包括推导了效果和成功率的统计量，并提供了几种情况下的界限。这使得我们能够根据样本数量和其他结构参数推断潜在攻击的准确性。

    

    成员推断攻击（MIA）可以揭示特定数据点是否是训练数据集的一部分，可能暴露个人的敏感信息。本文探讨了关于机器学习模型上MIA的基本统计限制。具体而言，我们首先推导了统计量，该统计量决定了这种攻击的有效性和成功率。然后，我们研究了几种情况，并对这个感兴趣的统计量提供了界限。这使我们能够根据样本数量和学习模型的其他结构参数推断潜在攻击的准确性，在某些情况下可以直接从数据集中估计。

    Membership inference attacks (MIA) can reveal whether a particular data point was part of the training dataset, potentially exposing sensitive information about individuals. This article explores the fundamental statistical limitations associated with MIAs on machine learning models. More precisely, we first derive the statistical quantity that governs the effectiveness and success of such attacks. Then, we investigate several situations for which we provide bounds on this quantity of interest. This allows us to infer the accuracy of potential attacks as a function of the number of samples and other structural parameters of learning models, which in some cases can be directly estimated from the dataset.
    
[^6]: 探索语言模型中谄媚行为的理解

    Towards Understanding Sycophancy in Language Models. (arXiv:2310.13548v1 [cs.CL])

    [http://arxiv.org/abs/2310.13548](http://arxiv.org/abs/2310.13548)

    这项研究探讨了强化学习从人类反馈中训练高质量AI助手的技术，发现这种方法可能导致模型在回答问题时过于谄媚，而不是坦诚，通过分析人类偏好数据得出了这一结论。

    

    「从人类反馈中进行强化学习（RLHF）」是训练高质量AI助手的一种流行技术。然而，RLHF可能会鼓励模型通过与用户信念相符的回答来代替真实回答，这种行为被称为谄媚行为。我们研究了RLHF训练模型中谄媚行为的普遍性以及人类偏好判断是否起到了作用。首先，我们证明了五个最先进的AI助手在四个不同的自由文本生成任务中一贯表现出谄媚行为。为了理解人类偏好是否驱动了RLHF模型的这种广泛行为，我们分析了现有的人类偏好数据。我们发现，当回答与用户的观点相符时，它更有可能被选中。此外，人类和偏好模型（PMs）将有说服力的谄媚回答与正确回答相比，有时几乎可以忽略不计地选择了谄媚回答。优化模型输出以满足PMs有时也会在真实性和谄媚行为之间做出取舍。

    Reinforcement learning from human feedback (RLHF) is a popular technique for training high-quality AI assistants. However, RLHF may also encourage model responses that match user beliefs over truthful responses, a behavior known as sycophancy. We investigate the prevalence of sycophancy in RLHF-trained models and whether human preference judgements are responsible. We first demonstrate that five state-of-the-art AI assistants consistently exhibit sycophantic behavior across four varied free-form text-generation tasks. To understand if human preferences drive this broadly observed behavior of RLHF models, we analyze existing human preference data. We find that when a response matches a user's views, it is more likely to be preferred. Moreover, both humans and preference models (PMs) prefer convincingly-written sycophantic responses over correct ones a negligible fraction of the time. Optimizing model outputs against PMs also sometimes sacrifices truthfulness in favor of sycophancy. Over
    
[^7]: 评论帮助更好地学习：基于时间的监督知识蒸馏

    Review helps learn better: Temporal Supervised Knowledge Distillation. (arXiv:2307.00811v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2307.00811](http://arxiv.org/abs/2307.00811)

    本文提出了一种基于时间的监督知识蒸馏方法，利用评论来帮助学生网络的学习。通过提取学生网络在不同训练阶段的时空特征，并通过动态目标进行训练，实现了对学生网络中旧知识的优化和利用，从而提高了网络的训练性能。

    

    在学习知识时，评论发挥了重要作用。在某个时间点获取的知识可能在之前的经验帮助下得到极大的启发。因此，知识增长过程应该在时间维度上展现出强烈的关联性。在我们的研究中，我们发现在网络训练过程中，特征图的演化遵循时间序列特性。适当的时间监督可以进一步提高网络训练性能。受到这一观察的启发，我们提出了基于时间的监督知识蒸馏（TSKD）。具体而言，我们通过卷积长短期记忆网络（Conv-LSTM）提取学生网络在不同训练阶段的时空特征。然后，我们通过动态目标训练学生网络，而不是静态的教师网络特征。这个过程实现了学生网络中旧知识的优化，并将其用于辅助当前的学习。广泛的实验证实了该方法的有效性。

    Reviewing plays an important role when learning knowledge. The knowledge acquisition at a certain time point may be strongly inspired with the help of previous experience. Thus the knowledge growing procedure should show strong relationship along the temporal dimension. In our research, we find that during the network training, the evolution of feature map follows temporal sequence property. A proper temporal supervision may further improve the network training performance. Inspired by this observation, we propose Temporal Supervised Knowledge Distillation (TSKD). Specifically, we extract the spatiotemporal features in the different training phases of student by convolutional Long Short-term memory network (Conv-LSTM). Then, we train the student net through a dynamic target, rather than static teacher network features. This process realizes the refinement of old knowledge in student network, and utilizes it to assist current learning. Extensive experiments verify the effectiveness and 
    
[^8]: 基于大型语言模型的点击诱骗检测

    Clickbait Detection via Large Language Models. (arXiv:2306.09597v1 [cs.CL])

    [http://arxiv.org/abs/2306.09597](http://arxiv.org/abs/2306.09597)

    本文研究了大型语言模型在点击诱骗检测上的性能，结果表明LLM无法取得最佳结果且不能仅通过标题实现满意的检测。

    

    点击诱骗（Clickbait）会通过一些令人惊讶甚至引人入胜的标题来诱导用户进行点击，几乎渗透到所有在线内容发布者，如新闻门户和社交媒体。最近，大型语言模型 (LLM)已成为一种强大的工具，并在一系列NLP下游任务中取得了巨大成功。但是，LLM是否可以作为高质量的点击诱骗检测系统还不为人所知。本文分析了LLM在多个英文和中文基准数据集的少样本场景下的性能。实验结果表明，与最先进的深度和微调PLM方法相比，LLM无法达到最佳结果。与人类直觉不同，实验表明LLM不能仅通过标题实现满意的点击诱骗检测。

    Clickbait, which aims to induce users with some surprising and even thrilling headlines for increasing click-through rates, permeates almost all online content publishers, such as news portals and social media. Recently, Large Language Models (LLMs) have emerged as a powerful instrument and achieved tremendous success in a serious of NLP downstream tasks. However, it is not yet known whether LLMs can be served as a high-quality clickbait detection system. In this paper, we analyze the performance of LLMs in the few-shot scenarios on a number of English and Chinese benchmark datasets. Experimental results show that LLMs cannot achieve the best results compared to the state-of-the-art deep and fine-tuning PLMs methods. Different from the human intuition, the experiments demonstrated that LLMs cannot make satisfied clickbait detection just by the headlines.
    
[^9]: 基于图的去中心化对抗性训练

    Decentralized Adversarial Training over Graphs. (arXiv:2303.13326v1 [cs.LG])

    [http://arxiv.org/abs/2303.13326](http://arxiv.org/abs/2303.13326)

    本文研究了在图上的去中心化对抗性训练，利用扩散学习的方法，开发了一种对抗性训练框架，增强了多个代理的鲁棒性以对抗攻击。

    

    近年来，机器学习模型对抗攻击的漏洞引起了广泛关注。大多数现有研究都集中在独立单一代理学习者的行为上。相比之下，本文研究了在图上的对抗性训练，其中各个单独的代理会受到空间中不同强度的扰动。预期通过链接代理和可能在图上实现的攻击模型的异质性，协调整个团队的强大协同作用可以帮助增强鲁棒性。本文使用扩散学习的极小-极大公式，为多代理系统开发了一种去中心化的对抗性训练框架。我们分析了该方案在凸和非凸环境下的收敛特性，并说明了增强的鲁棒性对抗攻击。

    The vulnerability of machine learning models to adversarial attacks has been attracting considerable attention in recent years. Most existing studies focus on the behavior of stand-alone single-agent learners. In comparison, this work studies adversarial training over graphs, where individual agents are subjected to perturbations of varied strength levels across space. It is expected that interactions by linked agents, and the heterogeneity of the attack models that are possible over the graph, can help enhance robustness in view of the coordination power of the group. Using a min-max formulation of diffusion learning, we develop a decentralized adversarial training framework for multi-agent systems. We analyze the convergence properties of the proposed scheme for both convex and non-convex environments, and illustrate the enhanced robustness to adversarial attacks.
    

