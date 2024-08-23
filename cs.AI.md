# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Domain Generalization through Meta-Learning: A Survey](https://arxiv.org/abs/2404.02785) | 元学习是一种有前景的方法，通过获取可转移知识实现在各种任务之间快速适应，为解决深度神经网络在面对分布变化和有限标记数据时泛化能力不佳提供了新途径。 |
| [^2] | [Larimar: Large Language Models with Episodic Memory Control](https://arxiv.org/abs/2403.11901) | Larimar提出了一种大脑启发的架构，通过分布式情节记忆增强LLMs，实现了动态、一次性的知识更新，无需昂贵的重新训练或微调，且在速度和灵活性上表现出色。 |
| [^3] | [Defending Against Unforeseen Failure Modes with Latent Adversarial Training](https://arxiv.org/abs/2403.05030) | 本研究利用潜在对抗训练（LAT）来防御AI系统中未预见的故障模式，通过利用网络实际用于预测的压缩、抽象和结构化概念的潜在表示，有效清除了恶意软件和对抗性攻击。 |
| [^4] | [Beyond Specialization: Assessing the Capabilities of MLLMs in Age and Gender Estimation](https://arxiv.org/abs/2403.02302) | 本研究评估了多模态大型语言模型（MLLMs）在年龄和性别估计中的能力，对不同模型进行了比较，揭示了它们在特定任务上的优势和劣势。 |
| [^5] | [Robust Policy Learning via Offline Skill Diffusion](https://arxiv.org/abs/2403.00225) | 提出了一种新颖的离线技能学习框架DuSkill，通过引导扩散模型生成通用技能，从而增强不同领域任务的策略学习鲁棒性。 |
| [^6] | [Language Agents as Optimizable Graphs](https://arxiv.org/abs/2402.16823) | 将基于LLM的代理统一描述为计算图，提出新颖的自动图优化器来改进节点和边，实现了代理之间的自动协作和改进。 |
| [^7] | [AI-Augmented Predictions: LLM Assistants Improve Human Forecasting Accuracy](https://arxiv.org/abs/2402.07862) | 本研究发现，使用LLMs助手可以显著提高预测准确性，不仅仅是由于模型预测准确性的提升。 |
| [^8] | [Studious Bob Fight Back Against Jailbreaking via Prompt Adversarial Tuning](https://arxiv.org/abs/2402.06255) | 本文提出了一种名为Prompt Adversarial Tuning (PAT)的方法，通过训练一个防御控制机制并将其作为前缀嵌入到用户提示中，实现对大型语言模型（LLMs）的越狱行为的防御。实验证明该方法在保护LLMs免受产生有害信息的影响方面效果显著。 |
| [^9] | [Clarify: Improving Model Robustness With Natural Language Corrections](https://arxiv.org/abs/2402.03715) | 论文提出了Clarify，一种通过自然语言纠正模型错误概念的方法，该方法通过用户提供简短的文本描述来纠正模型的一致失败模式，从而提高模型的鲁棒性。 |
| [^10] | [SymbolicAI: A framework for logic-based approaches combining generative models and solvers](https://arxiv.org/abs/2402.00854) | SymbolicAI是一个基于逻辑的框架，将生成模型与多种求解器无缝集成，通过将大型语言模型作为语义解析器，实现了符号推理与生成式人工智能的融合。 |
| [^11] | [Generative Large Language Models are autonomous practitioners of evidence-based medicine.](http://arxiv.org/abs/2401.02851) | 生成式大语言模型被应用于基于证据的医学实践，能够帮助管理临床医生面临的信息过载挑战。 |
| [^12] | [Adversarial Examples in the Physical World: A Survey.](http://arxiv.org/abs/2311.01473) | 本综述系统地研究了物理世界中的对抗样本（PAEs）的特点，并提出了基于其特征的全面分析和分类框架，涵盖了100多个研究，以填补对PAEs独特特征的现有研究不足。 |
| [^13] | [KLoB: a Benchmark for Assessing Knowledge Locating Methods in Language Models.](http://arxiv.org/abs/2309.16535) | KLoB是一个评估语言模型中知识定位方法的基准，旨在解决现有定位方法的准确性和事实知识局部性假设的问题。 |
| [^14] | [Deep Reinforcement Learning for Efficient and Fair Allocation of Health Care Resources.](http://arxiv.org/abs/2309.08560) | 本研究使用强化学习方法，通过整合个体患者的疾病进展和患者间的相互作用效应，来优化医疗资源的分配策略，旨在提高分配的公平性和整体患者结果。 |
| [^15] | [Can we trust the evaluation on ChatGPT?.](http://arxiv.org/abs/2303.12767) | 本文讨论了ChatGPT评估中面临的数据污染挑战，通过倾向性检测任务阐述了这一问题，并探讨了如何在闭合且持续训练模型的时代确保模型评估的公平性。 |
| [^16] | [Self-supervised Learning for Clustering of Wireless Spectrum Activity.](http://arxiv.org/abs/2210.02899) | 本研究使用无人监督学习技术探索无线电频谱活动，比较了两种不同的无人监督学习模型和一种混合模型，实现了精准的频谱活动聚类。 |

# 详细

[^1]: 通过元学习实现领域泛化：一项调查

    Domain Generalization through Meta-Learning: A Survey

    [https://arxiv.org/abs/2404.02785](https://arxiv.org/abs/2404.02785)

    元学习是一种有前景的方法，通过获取可转移知识实现在各种任务之间快速适应，为解决深度神经网络在面对分布变化和有限标记数据时泛化能力不佳提供了新途径。

    

    深度神经网络(DNNs)已经彻底改变了人工智能，但是当面对分布之外(out-of-distribution, OOD)数据时往往表现不佳，这是因为在现实世界应用中由于领域转移不可避免，训练和测试数据被假定为共享相同分布的常见情况。尽管DNNs在大量数据和计算能力方面非常有效，但它们很难应对分布变化和有限标记数据，导致过拟合和跨不同任务和领域的泛化能力不佳。元学习提供了一种有前途的方法，通过采用能够在各种任务之间获取可转移知识的算法进行快速适应，从而消除了需要从头学习每个任务的必要性。本调查论文深入探讨了元学习领域，重点关注其对领域泛化的贡献。

    arXiv:2404.02785v1 Announce Type: cross  Abstract: Deep neural networks (DNNs) have revolutionized artificial intelligence but often lack performance when faced with out-of-distribution (OOD) data, a common scenario due to the inevitable domain shifts in real-world applications. This limitation stems from the common assumption that training and testing data share the same distribution-an assumption frequently violated in practice. Despite their effectiveness with large amounts of data and computational power, DNNs struggle with distributional shifts and limited labeled data, leading to overfitting and poor generalization across various tasks and domains. Meta-learning presents a promising approach by employing algorithms that acquire transferable knowledge across various tasks for fast adaptation, eliminating the need to learn each task from scratch. This survey paper delves into the realm of meta-learning with a focus on its contribution to domain generalization. We first clarify the 
    
[^2]: Larimar: 具有情节记忆控制的大型语言模型

    Larimar: Large Language Models with Episodic Memory Control

    [https://arxiv.org/abs/2403.11901](https://arxiv.org/abs/2403.11901)

    Larimar提出了一种大脑启发的架构，通过分布式情节记忆增强LLMs，实现了动态、一次性的知识更新，无需昂贵的重新训练或微调，且在速度和灵活性上表现出色。

    

    本文提出了Larimar - 一种新颖的、受大脑启发的架构，用于增强大型语言模型(LLMs)的分布式情节记忆。 Larimar的记忆允许动态、一次性更新知识，无需进行计算昂贵的重新训练或微调。在多个事实编辑基准测试上的实验结果表明，Larimar在速度方面表现优异 - 根据基础LLM的不同，速度提升为4-10倍，并且由于提出的架构简单、不依赖于LLM，因此具有良好的灵活性和通用性。我们进一步提供了选择性事实遗忘和输入上下文长度概括机制，并展示了它们的有效性。

    arXiv:2403.11901v1 Announce Type: cross  Abstract: Efficient and accurate updating of knowledge stored in Large Language Models (LLMs) is one of the most pressing research challenges today. This paper presents Larimar - a novel, brain-inspired architecture for enhancing LLMs with a distributed episodic memory. Larimar's memory allows for dynamic, one-shot updates of knowledge without the need for computationally expensive re-training or fine-tuning. Experimental results on multiple fact editing benchmarks demonstrate that Larimar attains accuracy comparable to most competitive baselines, even in the challenging sequential editing setup, but also excels in speed - yielding speed-ups of 4-10x depending on the base LLM - as well as flexibility due to the proposed architecture being simple, LLM-agnostic, and hence general. We further provide mechanisms for selective fact forgetting and input context length generalization with Larimar and show their effectiveness.
    
[^3]: 利用潜在对抗训练防御未预见的故障模式

    Defending Against Unforeseen Failure Modes with Latent Adversarial Training

    [https://arxiv.org/abs/2403.05030](https://arxiv.org/abs/2403.05030)

    本研究利用潜在对抗训练（LAT）来防御AI系统中未预见的故障模式，通过利用网络实际用于预测的压缩、抽象和结构化概念的潜在表示，有效清除了恶意软件和对抗性攻击。

    

    人工智能系统有时在部署后会展示出有害的意外行为。尽管开发人员进行了大量诊断和调试，这种情况经常发生。由于攻击面非常广泛，从模型中减少风险具有挑战性。耗尽地搜索可能导致模型失败的输入是不可行的。红队和对抗训练（AT）通常用于使人工智能系统更加健壮。然而，它们并不足以避免许多与对抗训练不同的真实世界故障模式。在这项工作中，我们利用潜在对抗训练（LAT）来防御漏洞，而无需生成引发这些漏洞的输入。LAT利用网络实际用于预测的压缩、抽象和结构化概念的潜在表示。我们使用LAT来清除恶意软件并防御针对保留类别的对抗性攻击。我们展示在图像分类、文本分类

    arXiv:2403.05030v1 Announce Type: cross  Abstract: AI systems sometimes exhibit harmful unintended behaviors post-deployment. This is often despite extensive diagnostics and debugging by developers. Minimizing risks from models is challenging because the attack surface is so large. It is not tractable to exhaustively search for inputs that may cause a model to fail. Red-teaming and adversarial training (AT) are commonly used to make AI systems more robust. However, they have not been sufficient to avoid many real-world failure modes that differ from the ones adversarially trained on. In this work, we utilize latent adversarial training (LAT) to defend against vulnerabilities without generating inputs that elicit them. LAT leverages the compressed, abstract, and structured latent representations of concepts that the network actually uses for prediction. We use LAT to remove trojans and defend against held-out classes of adversarial attacks. We show in image classification, text classifi
    
[^4]: 超越专业化：评估MLLMs在年龄和性别估计中的能力

    Beyond Specialization: Assessing the Capabilities of MLLMs in Age and Gender Estimation

    [https://arxiv.org/abs/2403.02302](https://arxiv.org/abs/2403.02302)

    本研究评估了多模态大型语言模型（MLLMs）在年龄和性别估计中的能力，对不同模型进行了比较，揭示了它们在特定任务上的优势和劣势。

    

    最近，多模态大型语言模型（MLLMs）变得异常流行。像ChatGPT-4V和Gemini这样功能强大的商用模型，以及像LLaVA这样的开源模型，本质上都是通用模型，应用于解决各种各样的任务，包括计算机视觉中的任务。这些神经网络具有如此强大的通用知识和推理能力，以至于它们已被证明能够处理甚至未经专门训练的任务。我们将迄今为止最强大的MLLMs的能力进行了比较：ShareGPT4V、ChatGPT、LLaVA-Next 进行了专门任务的年龄和性别估计，与我们的最新专业化模型MiVOLO进行了比较。我们还更新了MiVOLO，并在本文中提供了详细信息和新的指标。这种比较产生了一些有趣的结果和关于参与模型的优点和缺点的见解。此外，我们尝试了各种微调方法

    arXiv:2403.02302v1 Announce Type: cross  Abstract: Multimodal Large Language Models (MLLMs) have recently gained immense popularity. Powerful commercial models like ChatGPT-4V and Gemini, as well as open-source ones such as LLaVA, are essentially general-purpose models and are applied to solve a wide variety of tasks, including those in computer vision. These neural networks possess such strong general knowledge and reasoning abilities that they have proven capable of working even on tasks for which they were not specifically trained. We compared the capabilities of the most powerful MLLMs to date: ShareGPT4V, ChatGPT, LLaVA-Next in a specialized task of age and gender estimation with our state-of-the-art specialized model, MiVOLO. We also updated MiVOLO and provide details and new metrics in this article. This comparison has yielded some interesting results and insights about the strengths and weaknesses of the participating models. Furthermore, we attempted various ways to fine-tune 
    
[^5]: 通过离线技能扩散实现稳健策略学习

    Robust Policy Learning via Offline Skill Diffusion

    [https://arxiv.org/abs/2403.00225](https://arxiv.org/abs/2403.00225)

    提出了一种新颖的离线技能学习框架DuSkill，通过引导扩散模型生成通用技能，从而增强不同领域任务的策略学习鲁棒性。

    

    基于技能的强化学习方法在解决长时域任务中表现出了相当大的潜力，尤其是通过分层结构。这些技能是从离线数据集中无关任务地学习的，可以加快针对新任务的策略学习过程。然而，由于这些技能在不同领域中的应用仍受限于对数据集的固有依赖，当尝试通过强化学习为不同于数据集领域的目标领域学习基于技能的策略时，这一挑战就变得困难。在本文中，我们提出了一个新颖的离线技能学习框架DuSkill，它采用了引导扩散模型来生成从数据集中有限技能扩展出的通用技能，从而增强了不同领域任务的策略学习鲁棒性。具体来说，我们设计了一个引导扩散技能解码器，结合分层编码，以解开技能嵌入。

    arXiv:2403.00225v1 Announce Type: new  Abstract: Skill-based reinforcement learning (RL) approaches have shown considerable promise, especially in solving long-horizon tasks via hierarchical structures. These skills, learned task-agnostically from offline datasets, can accelerate the policy learning process for new tasks. Yet, the application of these skills in different domains remains restricted due to their inherent dependency on the datasets, which poses a challenge when attempting to learn a skill-based policy via RL for a target domain different from the datasets' domains. In this paper, we present a novel offline skill learning framework DuSkill which employs a guided Diffusion model to generate versatile skills extended from the limited skills in datasets, thereby enhancing the robustness of policy learning for tasks in different domains. Specifically, we devise a guided diffusion-based skill decoder in conjunction with the hierarchical encoding to disentangle the skill embeddi
    
[^6]: 作为可优化图的语言代理

    Language Agents as Optimizable Graphs

    [https://arxiv.org/abs/2402.16823](https://arxiv.org/abs/2402.16823)

    将基于LLM的代理统一描述为计算图，提出新颖的自动图优化器来改进节点和边，实现了代理之间的自动协作和改进。

    

    多种人类设计的提升技术被提出，用于改进基于大型语言模型（LLMs）的问题求解器，产生了许多不同的代码库。我们通过将LLM代理描述为计算图来统一这些方法。节点实现处理多模态数据或查询LLMs的功能，并且边描述操作之间的信息流动。图形可以递归地组合成代表不同代理之间协作层次的更大组合图（其中边连接不同代理的操作）。我们的新颖自动图优化器（1）优化节点级LLM提示（节点优化）并（2）通过改变图连接性来改善代理协调（边缘优化）。实验证明我们的框架可用于高效开发、集成和自动改进各种LLM代理。代码可在https://github.com/metauto-ai/gptswarm找到。

    arXiv:2402.16823v1 Announce Type: cross  Abstract: Various human-designed prompt engineering techniques have been proposed to improve problem solvers based on Large Language Models (LLMs), yielding many disparate code bases. We unify these approaches by describing LLM-based agents as computational graphs. The nodes implement functions to process multimodal data or query LLMs, and the edges describe the information flow between operations. Graphs can be recursively combined into larger composite graphs representing hierarchies of inter-agent collaboration (where edges connect operations of different agents). Our novel automatic graph optimizers (1) refine node-level LLM prompts (node optimization) and (2) improve agent orchestration by changing graph connectivity (edge optimization). Experiments demonstrate that our framework can be used to efficiently develop, integrate, and automatically improve various LLM agents. The code can be found at https://github.com/metauto-ai/gptswarm.
    
[^7]: AI增强预测：LLM助手提高人类预测准确性

    AI-Augmented Predictions: LLM Assistants Improve Human Forecasting Accuracy

    [https://arxiv.org/abs/2402.07862](https://arxiv.org/abs/2402.07862)

    本研究发现，使用LLMs助手可以显著提高预测准确性，不仅仅是由于模型预测准确性的提升。

    

    大型语言模型(LLMs)展现出令人印象深刻的能力，在许多领域与甚至超过人类表现。本研究探讨了LLMs在预测任务中增强判断力的潜力。我们评估了两个GPT-4-Turbo助手对预测准确性的影响：一个旨在提供高质量建议（超级预测），另一个旨在过于自信和基本概率忽视。参与者（N = 991）可以在整个研究过程中咨询他们被分配的LLM助手，而对照组则使用一个较低级别的模型（DaVinci-003），不提供直接的预测支持。我们的注册分析显示，LLM增强显著提高了23%的预测准确性，无论是对于任何一种助手类型，相比于对照组。这种改进发生在超级预测助手在预测中更高的准确性的情况下，表明增强的效益不仅仅是由于模型预测准确性。

    Large language models (LLMs) show impressive capabilities, matching and sometimes exceeding human performance in many domains. This study explores the potential of LLMs to augment judgement in forecasting tasks. We evaluated the impact on forecasting accuracy of two GPT-4-Turbo assistants: one designed to provide high-quality advice ('superforecasting'), and the other designed to be overconfident and base-rate-neglecting. Participants (N = 991) had the option to consult their assigned LLM assistant throughout the study, in contrast to a control group that used a less advanced model (DaVinci-003) without direct forecasting support. Our preregistered analyses reveal that LLM augmentation significantly enhances forecasting accuracy by 23% across both types of assistants, compared to the control group. This improvement occurs despite the superforecasting assistant's higher accuracy in predictions, indicating the augmentation's benefit is not solely due to model prediction accuracy. Explora
    
[^8]: 进取的鲍勃通过提示对抗调整抵制越狱行为

    Studious Bob Fight Back Against Jailbreaking via Prompt Adversarial Tuning

    [https://arxiv.org/abs/2402.06255](https://arxiv.org/abs/2402.06255)

    本文提出了一种名为Prompt Adversarial Tuning (PAT)的方法，通过训练一个防御控制机制并将其作为前缀嵌入到用户提示中，实现对大型语言模型（LLMs）的越狱行为的防御。实验证明该方法在保护LLMs免受产生有害信息的影响方面效果显著。

    

    尽管大型语言模型（LLM）在各种应用中取得了巨大的成功，但它们也容易受到特定提示的影响，从而绕过内置的安全措施并提供危险或非法内容，这种现象被称为越狱行为。为了保护LLMs免受产生有害信息的影响，提出了各种防御策略，其中大多数集中在内容过滤或模型的对抗训练方面。在本文中，我们提出了一种名为Prompt Adversarial Tuning（PAT）的方法，通过训练一个防御控制机制并将其作为前缀嵌入到用户提示中来实现我们的防御策略。我们设计了一个类似对抗训练的训练过程，以实现我们的优化目标，交替更新攻击和防御控制机制。据我们所知，我们是第一个从提示调整的角度实施防御的人。一旦应用，我们的方法几乎不会影响LLMs的操作效率。实验表明我们的方法在抵御越狱行为方面具有良好的效果。

    Although Large Language Models (LLMs) have achieved tremendous success in various applications, they are also susceptible to certain prompts that can induce them to bypass built-in safety measures and provide dangerous or illegal content, a phenomenon known as jailbreak. To protect LLMs from producing harmful information, various defense strategies are proposed, with most focusing on content filtering or adversarial training of models. In this paper, we propose an approach named Prompt Adversarial Tuning (PAT) to train a defense control mechanism, which is then embedded as a prefix to user prompts to implement our defense strategy. We design a training process similar to adversarial training to achieve our optimized goal, alternating between updating attack and defense controls. To our knowledge, we are the first to implement defense from the perspective of prompt tuning. Once employed, our method will hardly impact the operational efficiency of LLMs. Experiments show that our method i
    
[^9]: 澄清：通过自然语言纠正提高模型的鲁棒性

    Clarify: Improving Model Robustness With Natural Language Corrections

    [https://arxiv.org/abs/2402.03715](https://arxiv.org/abs/2402.03715)

    论文提出了Clarify，一种通过自然语言纠正模型错误概念的方法，该方法通过用户提供简短的文本描述来纠正模型的一致失败模式，从而提高模型的鲁棒性。

    

    在监督学习中，模型被训练从静态数据集中提取相关性。这通常会导致模型依赖于高级错误概念。为了防止这种错误概念，我们必须提供额外的信息。现有的方法包括一些额外的实例级监督形式，例如标记虚假特征或来自平衡分布的额外标记数据。对于大规模数据集来说，这些策略可能会变得昂贵，因为它们需要以接近原始训练数据的规模进行额外注释。我们假设有针对性的关于模型错误概念的自然语言反馈是一种更有效的额外监督形式。我们引入了Clarify，一种新型界面和方法来交互式地纠正模型的错误概念。通过Clarify，用户只需要提供一个简短的文本描述来描述模型的一致性失败模式。然后，我们完全自动化地使用s

    In supervised learning, models are trained to extract correlations from a static dataset. This often leads to models that rely on high-level misconceptions. To prevent such misconceptions, we must necessarily provide additional information beyond the training data. Existing methods incorporate forms of additional instance-level supervision, such as labels for spurious features or additional labeled data from a balanced distribution. Such strategies can become prohibitively costly for large-scale datasets since they require additional annotation at a scale close to the original training data. We hypothesize that targeted natural language feedback about a model's misconceptions is a more efficient form of additional supervision. We introduce Clarify, a novel interface and method for interactively correcting model misconceptions. Through Clarify, users need only provide a short text description to describe a model's consistent failure patterns. Then, in an entirely automated way, we use s
    
[^10]: SymbolicAI: 一个结合生成模型和求解器的基于逻辑的方法的框架

    SymbolicAI: A framework for logic-based approaches combining generative models and solvers

    [https://arxiv.org/abs/2402.00854](https://arxiv.org/abs/2402.00854)

    SymbolicAI是一个基于逻辑的框架，将生成模型与多种求解器无缝集成，通过将大型语言模型作为语义解析器，实现了符号推理与生成式人工智能的融合。

    

    我们介绍了SymbolicAI，这是一个多功能且模块化的框架，采用基于逻辑的方法来处理生成过程中的概念学习和流程管理。SymbolicAI通过将大型语言模型（LLM）作为语义解析器来执行基于自然语言和形式语言指令的任务，从而弥合了符号推理和生成式人工智能之间的差距，使生成模型与各种求解器无缝集成。我们利用概率编程原理来处理复杂任务，并利用可微分和经典编程范 paradigms 的各自优势。该框架引入了一系列多态的、组合的和自指的数据流操作，将LLM的输出与用户的目标对齐。因此，我们可以在具有零次和少次学习能力的各种基础模型之间进行过渡，并与擅长解决特定问题的专业化调优模型或求解器配合使用。

    We introduce SymbolicAI, a versatile and modular framework employing a logic-based approach to concept learning and flow management in generative processes. SymbolicAI enables the seamless integration of generative models with a diverse range of solvers by treating large language models (LLMs) as semantic parsers that execute tasks based on both natural and formal language instructions, thus bridging the gap between symbolic reasoning and generative AI. We leverage probabilistic programming principles to tackle complex tasks, and utilize differentiable and classical programming paradigms with their respective strengths. The framework introduces a set of polymorphic, compositional, and self-referential operations for data stream manipulation, aligning LLM outputs with user objectives. As a result, we can transition between the capabilities of various foundation models endowed with zero- and few-shot learning capabilities and specialized, fine-tuned models or solvers proficient in addres
    
[^11]: 生成式大语言模型是基于证据的医学实践的自主从业者。 (arXiv:2401.02851v1 [cs.AI])

    Generative Large Language Models are autonomous practitioners of evidence-based medicine. (arXiv:2401.02851v1 [cs.AI])

    [http://arxiv.org/abs/2401.02851](http://arxiv.org/abs/2401.02851)

    生成式大语言模型被应用于基于证据的医学实践，能够帮助管理临床医生面临的信息过载挑战。

    

    背景：基于证据的医学 (EBM) 是现代临床实践的基础，要求临床医生不断更新知识并在患者护理中应用最佳临床证据。EBM的实践面临着医学研究的快速进展带来的信息过载挑战。人工智能 (AI)，特别是生成式大语言模型 (LLMs)，提供了管理这种复杂性的有希望的解决方案。方法：本研究涉及在各个专业领域收集现实世界的临床案例，将其转化为.json文件进行分析。使用了LLMs，包括ChatGPT 3.5和4等专有模型，Gemini Pro等，以及开源模型，如LLaMA v2和Mixtral-8x7B。这些模型配备了从案例文件中检索信息并做出临床决策的工具，类似于临床医生在现实世界中的操作方式。模型表现根据正确性进行评估。

    Background: Evidence-based medicine (EBM) is fundamental to modern clinical practice, requiring clinicians to continually update their knowledge and apply the best clinical evidence in patient care. The practice of EBM faces challenges due to rapid advancements in medical research, leading to information overload for clinicians. The integration of artificial intelligence (AI), specifically Generative Large Language Models (LLMs), offers a promising solution towards managing this complexity.  Methods: This study involved the curation of real-world clinical cases across various specialties, converting them into .json files for analysis. LLMs, including proprietary models like ChatGPT 3.5 and 4, Gemini Pro, and open-source models like LLaMA v2 and Mixtral-8x7B, were employed. These models were equipped with tools to retrieve information from case files and make clinical decisions similar to how clinicians must operate in the real world. Model performance was evaluated based on correctness
    
[^12]: 物理世界中的对抗样本：一项综述

    Adversarial Examples in the Physical World: A Survey. (arXiv:2311.01473v1 [cs.CV])

    [http://arxiv.org/abs/2311.01473](http://arxiv.org/abs/2311.01473)

    本综述系统地研究了物理世界中的对抗样本（PAEs）的特点，并提出了基于其特征的全面分析和分类框架，涵盖了100多个研究，以填补对PAEs独特特征的现有研究不足。

    

    深度神经网络（DNNs）对对抗样本表现出高度的脆弱性。除了在数字世界中的攻击外，对抗样本在物理世界中的实际影响提出了重大挑战和安全性问题。然而，当前对物理对抗样本（PAEs）的研究缺乏对其独特特征的全面理解，导致其重要性和理解的局限性。本文通过在训练、制造和重采样过程中全面考察PAEs的特点来弥补这一差距。通过分析物理对抗攻击之间的联系，我们确定制造和重采样是PAEs中独特属性和特殊性的主要来源。利用这一知识，我们基于其特定特征开发了一个全面的PAEs分析和分类框架，涵盖了100多个物理对抗世界研究的研究。

    Deep neural networks (DNNs) have demonstrated high vulnerability to adversarial examples. Besides the attacks in the digital world, the practical implications of adversarial examples in the physical world present significant challenges and safety concerns. However, current research on physical adversarial examples (PAEs) lacks a comprehensive understanding of their unique characteristics, leading to limited significance and understanding. In this paper, we address this gap by thoroughly examining the characteristics of PAEs within a practical workflow encompassing training, manufacturing, and re-sampling processes. By analyzing the links between physical adversarial attacks, we identify manufacturing and re-sampling as the primary sources of distinct attributes and particularities in PAEs. Leveraging this knowledge, we develop a comprehensive analysis and classification framework for PAEs based on their specific characteristics, covering over 100 studies on physical-world adversarial e
    
[^13]: KLoB: 一种评估语言模型中知识定位方法的基准

    KLoB: a Benchmark for Assessing Knowledge Locating Methods in Language Models. (arXiv:2309.16535v1 [cs.CL])

    [http://arxiv.org/abs/2309.16535](http://arxiv.org/abs/2309.16535)

    KLoB是一个评估语言模型中知识定位方法的基准，旨在解决现有定位方法的准确性和事实知识局部性假设的问题。

    

    最近，定位然后编辑的范式已经成为改变语言模型中存储的事实知识的主要方法之一。然而，目前的定位方法是否能够准确地找到嵌入所需知识的确切参数还缺乏研究。此外，尽管许多研究人员对事实知识的局部性假设的有效性提出了质疑，但没有提供一种测试假设的方法以进行更深入的讨论和研究。因此，我们引入了KLoB，一个评估可靠的知识定位方法应满足的三个基本属性的基准。KLoB可作为评估语言模型中现有定位方法的基准，并为重新评估事实知识的局部性假设提供了一种方法。我们的代码公开可用于\url{https://github.com/juyiming/KLoB}。

    Recently, Locate-Then-Edit paradigm has emerged as one of the main approaches in changing factual knowledge stored in the Language models. However, there is a lack of research on whether present locating methods can pinpoint the exact parameters embedding the desired knowledge. Moreover, although many researchers have questioned the validity of locality hypothesis of factual knowledge, no method is provided to test the a hypothesis for more in-depth discussion and research. Therefore, we introduce KLoB, a benchmark examining three essential properties that a reliable knowledge locating method should satisfy. KLoB can serve as a benchmark for evaluating existing locating methods in language models, and can contributes a method to reassessing the validity of locality hypothesis of factual knowledge. Our is publicly available at \url{https://github.com/juyiming/KLoB}.
    
[^14]: 用于高效且公平分配医疗资源的深度强化学习

    Deep Reinforcement Learning for Efficient and Fair Allocation of Health Care Resources. (arXiv:2309.08560v1 [cs.LG])

    [http://arxiv.org/abs/2309.08560](http://arxiv.org/abs/2309.08560)

    本研究使用强化学习方法，通过整合个体患者的疾病进展和患者间的相互作用效应，来优化医疗资源的分配策略，旨在提高分配的公平性和整体患者结果。

    

    医疗资源的稀缺性可能导致不可避免的配给问题。例如，通气机的供应通常有限，特别是在公共卫生紧急情况或资源有限的医疗环境中，如COVID-19大流行期间。目前，针对医疗资源分配的协议并没有普遍接受的标准，导致各国政府根据不同的标准和基于启发式协议来优先考虑患者。在本研究中，我们研究了使用强化学习来优化重症护理资源分配策略，以公平有效地配给资源。我们提出了基于变换器的深度Q网络，用于将个体患者的病情进展和患者间的相互作用效应整合到重症护理资源分配中。我们的目标是提高分配的公平性和整体患者结果。我们的实验表明，我们的方法显著减少了过度配给资源的情况。

    Scarcity of health care resources could result in the unavoidable consequence of rationing. For example, ventilators are often limited in supply, especially during public health emergencies or in resource-constrained health care settings, such as amid the pandemic of COVID-19. Currently, there is no universally accepted standard for health care resource allocation protocols, resulting in different governments prioritizing patients based on various criteria and heuristic-based protocols. In this study, we investigate the use of reinforcement learning for critical care resource allocation policy optimization to fairly and effectively ration resources. We propose a transformer-based deep Q-network to integrate the disease progression of individual patients and the interaction effects among patients during the critical care resource allocation. We aim to improve both fairness of allocation and overall patient outcomes. Our experiments demonstrate that our method significantly reduces exces
    
[^15]: 我们能相信ChatGPT的评估吗？

    Can we trust the evaluation on ChatGPT?. (arXiv:2303.12767v1 [cs.CL])

    [http://arxiv.org/abs/2303.12767](http://arxiv.org/abs/2303.12767)

    本文讨论了ChatGPT评估中面临的数据污染挑战，通过倾向性检测任务阐述了这一问题，并探讨了如何在闭合且持续训练模型的时代确保模型评估的公平性。

    

    ChatGPT是第一个被广泛采纳的大型语言模型，展示出在多项自然语言任务中卓越的表现。但是，由于模型的闭合性以及通过强化学习和人类反馈不断更新，评估ChatGPT在不同问题领域的表现仍然具有挑战性。本文重点讨论了在ChatGPT的评估中存在的数据污染问题，并使用倾向性检测任务作为案例进行了说明。我们还讨论了如何在闭合和持续训练模型的时代，避免数据污染和确保公平的模型评估的挑战。

    ChatGPT, the first large language model (LLM) with mass adoption, has demonstrated remarkable performance in numerous natural language tasks. Despite its evident usefulness, evaluating ChatGPT's performance in diverse problem domains remains challenging due to the closed nature of the model and its continuous updates via Reinforcement Learning from Human Feedback (RLHF). We highlight the issue of data contamination in ChatGPT evaluations, with a case study of the task of stance detection. We discuss the challenge of preventing data contamination and ensuring fair model evaluation in the age of closed and continuously trained models.
    
[^16]: 无人监督学习用于无线电频谱活动聚类

    Self-supervised Learning for Clustering of Wireless Spectrum Activity. (arXiv:2210.02899v2 [cs.NI] UPDATED)

    [http://arxiv.org/abs/2210.02899](http://arxiv.org/abs/2210.02899)

    本研究使用无人监督学习技术探索无线电频谱活动，比较了两种不同的无人监督学习模型和一种混合模型，实现了精准的频谱活动聚类。

    

    近年来，通过机器学习技术处理无线电频谱数据以解决认知无线电网络相关问题，如异常检测、调制分类、技术分类和设备指纹等领域，取得了很大进展。大多数解决方案都是基于受控制的、带标签的数据，并使用监督式学习方法进行处理。然而，在现实世界环境下测量的频谱数据高度不确定，其标记是一项费时且昂贵的过程，需要领域专业知识，因此成为在该领域使用监督式学习方法的主要缺点之一。本文研究了在现实世界未标记数据中利用无人监督学习（SSL）来探索频谱活动的使用。具体来说，我们比较了两种 SSL 模型的性能，一种基于 DeepCluster 参考体系结构，另一种适用于频谱活动识别和聚类，以及一种新型的混合 SSL 方法，结合了两种模型。我们的实验评估表明，这三种方法都可以有效地对频谱活动数据进行聚类，混合方法的性能最佳。

    In recent years, much work has been done on processing of wireless spectrum data involving machine learning techniques in domain-related problems for cognitive radio networks, such as anomaly detection, modulation classification, technology classification and device fingerprinting. Most of the solutions are based on labeled data, created in a controlled manner and processed with supervised learning approaches. However, spectrum data measured in real-world environment is highly nondeterministic, making its labeling a laborious and expensive process, requiring domain expertise, thus being one of the main drawbacks of using supervised learning approaches in this domain. In this paper, we investigate the use of self-supervised learning (SSL) for exploring spectrum activities in a real-world unlabeled data. In particular, we compare the performance of two SSL models, one based on a reference DeepCluster architecture and one adapted for spectrum activity identification and clustering, and a 
    

