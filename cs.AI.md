# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [MoPE: Parameter-Efficient and Scalable Multimodal Fusion via Mixture of Prompt Experts](https://arxiv.org/abs/2403.10568) | 本文提出了MoPE技术，通过解开提示以自适应捕获数据集级和实例级特征，引入了混合Prompt专家来增强表达能力，并且在多模态融合中表现出更大的表达能力和可扩展性。 |
| [^2] | [MathGenie: Generating Synthetic Data with Question Back-translation for Enhancing Mathematical Reasoning of LLMs](https://arxiv.org/abs/2402.16352) | MathGenie通过问题反向翻译生成合成数据，用于增强LLMs的数学推理能力，并创造了一个家族化的模型系列MathGenieLM。 |
| [^3] | [CriticBench: Evaluating Large Language Models as Critic](https://arxiv.org/abs/2402.13764) | CriticBench是一个旨在全面和可靠地评估大型语言模型的评论能力的新型基准，展示了评论能力与任务、响应质量和模型规模之间的关系。 |
| [^4] | [Explaining Learned Reward Functions with Counterfactual Trajectories](https://arxiv.org/abs/2402.04856) | 通过对比原始轨迹和反事实部分轨迹的奖励，我们提出了一种解释强化学习中奖励函数的方法。通过生成符合质量标准的反事实轨迹解释（CTEs），我们的实验表明，CTEs对于代理人模型具有明显的信息性，能提高其预测与奖励函数的一致性。 |
| [^5] | [With Greater Text Comes Greater Necessity: Inference-Time Training Helps Long Text Generation](https://arxiv.org/abs/2401.11504) | Temp-Lora方法通过在长文本生成过程中逐步训练临时Lora模块，有效保留上下文知识并避免对模型参数的永久性改变。 |
| [^6] | [Moderating Model Marketplaces: Platform Governance Puzzles for AI Intermediaries](https://arxiv.org/abs/2311.12573) | 本论文研究了模型市场的调节问题，分析了AI中介平台面临的平台治理挑战，并总结了业界的相关实践，包括许可、访问和使用限制、自动内容调节以及公开政策制定。 |
| [^7] | [Explainable Identification of Hate Speech towards Islam using Graph Neural Networks](https://arxiv.org/abs/2311.04916) | 使用图神经网络解释和识别伊斯兰教仇恨言论，模型在保持出色性能的同时能够解释相关性和因果关系。 |
| [^8] | [Decision Theoretic Foundations for Experiments Evaluating Human Decisions.](http://arxiv.org/abs/2401.15106) | 该论文通过综合统计决策理论和信息经济学，提出了决策问题的广泛适用定义。为了将人类决策的下降归咎于偏见形式，实验必须向参与者提供足够的信息来识别规范决策。然而，根据作者对AI辅助决策的研究的评估，只有17%的研究提供了足够的信息来描述参与者的行为偏离了良好的决策。 |
| [^9] | [Unveiling the General Intelligence Factor in Language Models: A Psychometric Approach.](http://arxiv.org/abs/2310.11616) | 本研究利用心理测量理论揭示了语言模型中的普遍智能因子g的存在，并发现了该因子解释模型性能方差的85%，为模型评估和开发提供了统一的指标。 |
| [^10] | [Rethinking Integration of Prediction and Planning in Deep Learning-Based Automated Driving Systems: A Review.](http://arxiv.org/abs/2308.05731) | 这项综述重新思考了基于深度学习的自动驾驶系统中预测和规划的整合问题，提出了将其作为相互依赖的联合步骤来提高安全性、效率性和舒适性的必要性。 |
| [^11] | [RRWKV: Capturing Long-range Dependencies in RWKV.](http://arxiv.org/abs/2306.05176) | 本文介绍了一种新的RRWKV架构，它在保持记忆和计算效率的同时，通过加入回顾能力有效地捕捉长距离依赖关系。 |

# 详细

[^1]: MoPE：通过Prompt专家混合实现参数高效和可扩展的多模态融合

    MoPE: Parameter-Efficient and Scalable Multimodal Fusion via Mixture of Prompt Experts

    [https://arxiv.org/abs/2403.10568](https://arxiv.org/abs/2403.10568)

    本文提出了MoPE技术，通过解开提示以自适应捕获数据集级和实例级特征，引入了混合Prompt专家来增强表达能力，并且在多模态融合中表现出更大的表达能力和可扩展性。

    

    Prompt调整已经证明在融合多模态任务的单模基础模型时具有参数效率性。然而，其有限的适应性和表达能力导致性能不佳与其他调整方法相比。本文通过将简单提示解开以自适应地捕获数据集级和实例级特征来解决这个问题。建立在这种解开的基础上，我们引入了Prompt专家的混合（MoPE）技术来增强表达能力。MoPE利用多模态配对先验在每个实例基础上路由最有效的提示。与简单提示相比，我们基于MoPE的条件提示对多模态融合具有更大的表达能力，在训练数据和可训练参数总数上具有更好的扩展性。我们还研究了一个专家路由的正则化项，导致专家的不断发展专长，不同专家专注于不同的特征。

    arXiv:2403.10568v1 Announce Type: cross  Abstract: Prompt-tuning has demonstrated parameter-efficiency in fusing unimodal foundation models for multimodal tasks. However, its limited adaptivity and expressiveness lead to suboptimal performance when compared with other tuning methods. In this paper, we address this issue by disentangling the vanilla prompts to adaptively capture dataset-level and instance-level features. Building upon this disentanglement, we introduce the mixture of prompt experts (MoPE) technique to enhance expressiveness. MoPE leverages multimodal pairing priors to route the most effective prompt on a per-instance basis. Compared to vanilla prompting, our MoPE-based conditional prompting exhibits greater expressiveness for multimodal fusion, scaling better with the training data and the overall number of trainable parameters. We also study a regularization term for expert routing, leading to emergent expert specialization, where different experts focus on different c
    
[^2]: MathGenie: 使用问题反向翻译生成合成数据，以增强LLMs的数学推理能力

    MathGenie: Generating Synthetic Data with Question Back-translation for Enhancing Mathematical Reasoning of LLMs

    [https://arxiv.org/abs/2402.16352](https://arxiv.org/abs/2402.16352)

    MathGenie通过问题反向翻译生成合成数据，用于增强LLMs的数学推理能力，并创造了一个家族化的模型系列MathGenieLM。

    

    大型语言模型(LLMs)在数学推理方面展现出巨大潜力。然而，目前开源模型和GPT-4等闭源模型之间在这一领域仍存在性能差距。本文介绍了一种新颖的方法MathGenie，用于从小规模问题-解决方案数据集（称为种子数据）中生成多样且可靠的数学问题。我们扩充了种子数据的真实解决方案，并训练了一个反向翻译模型，将扩充的解决方案翻译回新问题。随后，我们为新问题生成了集成代码解决方案。为确保集成代码解决方案的正确性，我们采用了基于原理的解决方案验证策略。我们在新筛选的数据上对从7B到70B不等的各种预训练模型进行训练，以测试所提出的增强技术的有效性，从而产生了一个称为MathGenieLM的模型系列。

    arXiv:2402.16352v1 Announce Type: cross  Abstract: Large language models (LLMs) have exhibited great potential in mathematical reasoning. However, there remains a performance gap in this area between existing open-source models and closed-source models such as GPT-4. In this paper, we introduce MathGenie, a novel method for generating diverse and reliable math problems from a small-scale problem-solution dataset (denoted as seed data). We augment the ground-truth solutions of our seed data and train a back-translation model to translate the augmented solutions back into new questions. Subsequently, we generate code-integrated solutions for the new questions. To ensure the correctness of the code-integrated solutions, we employ rationale-based strategy for solution verification. Various pretrained models, ranging from 7B to 70B, are trained on the newly curated data to test the effectiveness of the proposed augmentation technique, resulting in a family of models known as MathGenieLM. Th
    
[^3]: CriticBench: 将大型语言模型作为评论家进行评估

    CriticBench: Evaluating Large Language Models as Critic

    [https://arxiv.org/abs/2402.13764](https://arxiv.org/abs/2402.13764)

    CriticBench是一个旨在全面和可靠地评估大型语言模型的评论能力的新型基准，展示了评论能力与任务、响应质量和模型规模之间的关系。

    

    论文提出了 CriticBench，这是一个旨在全面和可靠地评估大型语言模型（LLMs）的四个关键评论能力维度（反馈、比较、改进和元反馈）的新型基准。CriticBench包含九个不同的任务，每个任务评估LLMs在不同质量细粒度水平上评论响应的能力。对开源和闭源LLMs进行的广泛评估揭示了评论能力与任务、响应质量和模型规模之间有趣的关系。CriticBench的数据集、资源和评估工具包将在https://github.com/gmftbyGMFTBY/Cri上公开发布。

    arXiv:2402.13764v1 Announce Type: cross  Abstract: Critique ability are crucial in the scalable oversight and self-improvement of Large Language Models (LLMs). While many recent studies explore the critique ability of LLMs to judge and refine flaws in generations, how to comprehensively and reliably measure the critique abilities of LLMs is under-explored. This paper introduces \shortname, a novel benchmark designed to comprehensively and reliably evaluate four key critique ability dimensions of LLMs: feedback, comparison, refinement and meta-feedback. \shortname~encompasses nine diverse tasks, each assessing the LLMs' ability to critique responses at varying levels of quality granularity. Our extensive evaluations of open-source and closed-source LLMs reveal intriguing relationships between the critique ability and tasks, response qualities, and model scales. Datasets, resources and evaluation toolkit for \shortname~will be publicly released at \url{https://github.com/gmftbyGMFTBY/Cri
    
[^4]: 通过反事实轨迹解释学习到的奖励函数

    Explaining Learned Reward Functions with Counterfactual Trajectories

    [https://arxiv.org/abs/2402.04856](https://arxiv.org/abs/2402.04856)

    通过对比原始轨迹和反事实部分轨迹的奖励，我们提出了一种解释强化学习中奖励函数的方法。通过生成符合质量标准的反事实轨迹解释（CTEs），我们的实验表明，CTEs对于代理人模型具有明显的信息性，能提高其预测与奖励函数的一致性。

    

    从人类行为或反馈中学习奖励是将AI系统与人类价值观一致的一种有希望的方法，但无法始终提取正确的奖励函数。可解释性工具可以帮助用户理解和评估学习到的奖励函数中可能存在的缺陷。我们提出了反事实轨迹解释（CTEs），通过对比原始轨迹和反事实部分轨迹以及它们各自接收的奖励来解释强化学习中的奖励函数。我们为CTEs制定了六个质量标准，并提出了一种基于Monte-Carlo的新算法来生成优化这些质量标准的CTEs。最后，我们通过训练代理人模型来衡量生成的解释对其的信息性。CTEs对于代理人模型具有明显的信息性，增加了其预测与未见轨迹上的奖励函数的相似性。此外，它学会了准确判断轨迹之间的奖励差异。

    Learning rewards from human behaviour or feedback is a promising approach to aligning AI systems with human values but fails to consistently extract correct reward functions. Interpretability tools could enable users to understand and evaluate possible flaws in learned reward functions. We propose Counterfactual Trajectory Explanations (CTEs) to interpret reward functions in reinforcement learning by contrasting an original with a counterfactual partial trajectory and the rewards they each receive. We derive six quality criteria for CTEs and propose a novel Monte-Carlo-based algorithm for generating CTEs that optimises these quality criteria. Finally, we measure how informative the generated explanations are to a proxy-human model by training it on CTEs. CTEs are demonstrably informative for the proxy-human model, increasing the similarity between its predictions and the reward function on unseen trajectories. Further, it learns to accurately judge differences in rewards between trajec
    
[^5]: 随着文本量增加，推断训练有助于长文本生成

    With Greater Text Comes Greater Necessity: Inference-Time Training Helps Long Text Generation

    [https://arxiv.org/abs/2401.11504](https://arxiv.org/abs/2401.11504)

    Temp-Lora方法通过在长文本生成过程中逐步训练临时Lora模块，有效保留上下文知识并避免对模型参数的永久性改变。

    

    长文本生成，如小说创作和具有极长上下文的篇章级翻译，对当前的语言模型提出了重大挑战。现有方法主要集中在通过长度外推等策略扩展模型的上下文窗口。然而，这些方法在训练和/或推断阶段要求大量硬件资源。我们提出的方法Temp-Lora引入了一个替代概念。我们不依赖于KV缓存存储所有上下文信息，而是将这些信息直接嵌入临时Lora模块中。在长文本生成过程中，这个模块会随着先前生成的文本逐渐进行训练。这种方法不仅有效地保留上下文知识，还防止了对模型参数的任何永久性改变，因为模块在生成后被丢弃。在PG19语言建模上进行了大量实验。

    arXiv:2401.11504v2 Announce Type: replace-cross  Abstract: Long text generation, such as novel writing and discourse-level translation with extremely long contexts, presents significant challenges to current language models. Existing methods mainly focus on extending the model's context window through strategies like length extrapolation. However, these approaches demand substantial hardware resources during the training and/or inference phases. Our proposed method, Temp-Lora, introduces an alternative concept. Instead of relying on the KV cache to store all context information, we embeds this information directly into a temporary Lora module. In the process of long text generation, this module is progressively trained with text generated previously. This approach not only efficiently preserves contextual knowledge but also prevents any permanent alteration to the model's parameters given that the module is discarded post-generation. Extensive experiments on the PG19 language modeling 
    
[^6]: 模型市场的调节: AI中介平台的平台治理难题

    Moderating Model Marketplaces: Platform Governance Puzzles for AI Intermediaries

    [https://arxiv.org/abs/2311.12573](https://arxiv.org/abs/2311.12573)

    本论文研究了模型市场的调节问题，分析了AI中介平台面临的平台治理挑战，并总结了业界的相关实践，包括许可、访问和使用限制、自动内容调节以及公开政策制定。

    

    arXiv: 2311.12573v2 公告类型: replace-cross 摘要: AI开发社区越来越多地利用托管中介平台，如Hugging Face，为用户上传的模型和训练数据提供便捷访问。这些模型市场降低了成千上万用户的技术部署门槛，但也可能被用于许多潜在有害和非法的方式。在本文中，我们解释了AI系统如何既能“包含”内容又能是开放式工具，从而成为迄今为止最棘手的平台治理挑战之一。我们提供了几个案例研究来分析模型市场如何管理模型，这些案例跨越了三个具有代表性的平台，即Hugging Face、GitHub和Civitai。基于这些分析，我们总结了业界正在制定的重要（但仍然有限）应对调节需求的做法：许可、访问和使用限制、自动内容调节以及公开政策制定。

    arXiv:2311.12573v2 Announce Type: replace-cross  Abstract: The AI development community is increasingly making use of hosting intermediaries such as Hugging Face provide easy access to user-uploaded models and training data. These model marketplaces lower technical deployment barriers for hundreds of thousands of users, yet can be used in numerous potentially harmful and illegal ways. In this article, we explain ways in which AI systems, which can both `contain' content and be open-ended tools, present one of the trickiest platform governance challenges seen to date. We provide case studies of several incidents across three illustrative platforms -- Hugging Face, GitHub and Civitai -- to examine how model marketplaces moderate models. Building on this analysis, we outline important (and yet nevertheless limited) practices that industry has been developing to respond to moderation demands: licensing, access and use restrictions, automated content moderation, and open policy development.
    
[^7]: 使用图神经网络解释伊斯兰教仇恨言论的研究

    Explainable Identification of Hate Speech towards Islam using Graph Neural Networks

    [https://arxiv.org/abs/2311.04916](https://arxiv.org/abs/2311.04916)

    使用图神经网络解释和识别伊斯兰教仇恨言论，模型在保持出色性能的同时能够解释相关性和因果关系。

    

    伊斯兰教仇恨言论在在线社交互动平台上是一个普遍存在的挑战。识别和消除这种仇恨是迈向和谐与和平未来的关键一步。本研究提出了一种新的范例，利用图神经网络来识别和解释针对伊斯兰教的仇恨言论。利用图神经网络发现、提取并利用不同数据点之间的关系的内在能力，我们的模型始终能够在保持出色性能的同时提供对潜在相关性和因果关系的解释。

    arXiv:2311.04916v2 Announce Type: cross  Abstract: Islamophobic language is a prevalent challenge on online social interaction platforms. Identifying and eliminating such hatred is a crucial step towards a future of harmony and peace. This study presents a novel paradigm for identifying and explaining hate speech towards Islam using graph neural networks. Utilizing the intrinsic ability of graph neural networks to find, extract, and use relationships across disparate data points, our model consistently achieves outstanding performance while offering explanations for the underlying correlations and causation.
    
[^8]: 决策理论基础对评估人类决策的实验的影响

    Decision Theoretic Foundations for Experiments Evaluating Human Decisions. (arXiv:2401.15106v1 [cs.HC])

    [http://arxiv.org/abs/2401.15106](http://arxiv.org/abs/2401.15106)

    该论文通过综合统计决策理论和信息经济学，提出了决策问题的广泛适用定义。为了将人类决策的下降归咎于偏见形式，实验必须向参与者提供足够的信息来识别规范决策。然而，根据作者对AI辅助决策的研究的评估，只有17%的研究提供了足够的信息来描述参与者的行为偏离了良好的决策。

    

    信息展示的决策是可解释AI、人工智能与人类的合作以及数据可视化等领域研究的重点。然而，决策问题的定义以及实验必须具备的条件以得出人类决策存在缺陷的结论仍然存在争议。我们提出了一个广泛适用的决策问题定义，该定义是从统计决策理论和信息经济学中综合提炼而来的。我们认为，要将人类绩效下降归咎于某种偏见形式，实验必须向参与者提供足够的信息，以便合理的代理能够识别规范决策。我们评估了最近有关AI辅助决策的文献中对决策制定进行的评估在多大程度上达到了这一标准。我们发现，只有35项声称确定了有偏差行为的研究中的6项（17%）向参与者提供了足够信息来描述其行为偏离良好决策

    Decision-making with information displays is a key focus of research in areas like explainable AI, human-AI teaming, and data visualization. However, what constitutes a decision problem, and what is required for an experiment to be capable of concluding that human decisions are flawed in some way, remain open to speculation. We present a widely applicable definition of a decision problem synthesized from statistical decision theory and information economics. We argue that to attribute loss in human performance to forms of bias, an experiment must provide participants with the information that a rational agent would need to identify the normative decision. We evaluate the extent to which recent evaluations of decision-making from the literature on AI-assisted decisions achieve this criteria. We find that only 6 (17\%) of 35 studies that claim to identify biased behavior present participants with sufficient information to characterize their behavior as deviating from good decision-making
    
[^9]: 揭示语言模型中的普遍智能因子：一种心理测量方法

    Unveiling the General Intelligence Factor in Language Models: A Psychometric Approach. (arXiv:2310.11616v1 [cs.CL])

    [http://arxiv.org/abs/2310.11616](http://arxiv.org/abs/2310.11616)

    本研究利用心理测量理论揭示了语言模型中的普遍智能因子g的存在，并发现了该因子解释模型性能方差的85%，为模型评估和开发提供了统一的指标。

    

    本研究采用心理测量理论，揭示了语言模型中普遍智能因子g的存在，并扩展了该理论在人类和某些动物物种中的应用。通过对两个大型数据集Open LLM Leaderboard（包含1,232个模型）和General Language Understanding Evaluation（GLUE）Leaderboard（包含88个模型）进行因子分析，我们发现了一个具有一维性和高度稳定性的g因子，可以解释模型性能方差的85%。研究还发现模型大小和g之间的中度相关性为0.48。在语言模型中发现g因子为模型评估提供了统一的指标，为更强大、基于g因子的模型能力评估开辟了新的途径。这些发现为从心理测量的角度理解和未来研究人工智能提供了基础，并对模型评估和开发具有实际意义。

    This study uncovers the factor of general intelligence, or g, in language models, extending the psychometric theory traditionally applied to humans and certain animal species. Utilizing factor analysis on two extensive datasets Open LLM Leaderboard with 1,232 models and General Language Understanding Evaluation (GLUE) Leaderboard with 88 models - we find compelling evidence for a unidimensional, highly stable g factor that accounts for 85% of the variance in model performance. The study also finds a moderate correlation of .48 between model size and g. The discovery of g in language models offers a unified metric for model evaluation and opens new avenues for more robust, g-based model ability assessment. These findings lay the foundation for understanding and future research on artificial general intelligence from a psychometric perspective and have practical implications for model evaluation and development.
    
[^10]: 重新思考基于深度学习的自动驾驶系统中的预测和规划的整合：一项综述

    Rethinking Integration of Prediction and Planning in Deep Learning-Based Automated Driving Systems: A Review. (arXiv:2308.05731v1 [cs.RO])

    [http://arxiv.org/abs/2308.05731](http://arxiv.org/abs/2308.05731)

    这项综述重新思考了基于深度学习的自动驾驶系统中预测和规划的整合问题，提出了将其作为相互依赖的联合步骤来提高安全性、效率性和舒适性的必要性。

    

    自动驾驶有可能彻底改变个人、公共和货运交通的方式。除了感知环境的巨大挑战外，即准确地使用可用的传感器数据感知环境，自动驾驶还包括规划一个安全、舒适和高效的运动轨迹。为了促进安全和进步，许多工作依赖于模块化的交通未来运动的预测。模块化的自动驾驶系统通常将预测和规划作为顺序的独立任务处理。虽然这考虑了周围交通对自车的影响，但它未能预测交通参与者对自车行为的反应。最近的研究表明，将预测和规划整合为相互依赖的联合步骤是实现安全、高效和舒适驾驶所必需的。虽然有各种模型实现了这种集成系统，但对不同原理的全面概述和理论理解仍然缺乏。

    Automated driving has the potential to revolutionize personal, public, and freight mobility. Besides the enormous challenge of perception, i.e. accurately perceiving the environment using available sensor data, automated driving comprises planning a safe, comfortable, and efficient motion trajectory. To promote safety and progress, many works rely on modules that predict the future motion of surrounding traffic. Modular automated driving systems commonly handle prediction and planning as sequential separate tasks. While this accounts for the influence of surrounding traffic on the ego-vehicle, it fails to anticipate the reactions of traffic participants to the ego-vehicle's behavior. Recent works suggest that integrating prediction and planning in an interdependent joint step is necessary to achieve safe, efficient, and comfortable driving. While various models implement such integrated systems, a comprehensive overview and theoretical understanding of different principles are lacking.
    
[^11]: RRWKV：在RWKV中捕捉长距离依赖关系

    RRWKV: Capturing Long-range Dependencies in RWKV. (arXiv:2306.05176v1 [cs.CL])

    [http://arxiv.org/abs/2306.05176](http://arxiv.org/abs/2306.05176)

    本文介绍了一种新的RRWKV架构，它在保持记忆和计算效率的同时，通过加入回顾能力有效地捕捉长距离依赖关系。

    

    由于Transformer惊人的点积注意力，它已经成为各种自然语言处理（NLP）任务中的主要架构。最近，Receptance Weighted Key Value（RWKV）架构遵循非Transformer架构，消除了点积注意力的缺点，其中存储和计算复杂度随着序列长度呈二次扩展。尽管RWKV利用了线性张量积注意机制并通过部署时间序列模式实现了并行计算，但与标准Transformer中直接交互获得的完整信息相比，它无法捕捉长距离依赖关系，因为其受限于向后查看先前信息的能力。因此，本文通过将回顾能力纳入RWKV中来设计Retrospected Receptance Weighted Key Value（RRWKV）架构，以有效地吸收信息，同时保持记忆和计算效率。

    Owing to the impressive dot-product attention, the Transformers have been the dominant architectures in various natural language processing (NLP) tasks. Recently, the Receptance Weighted Key Value (RWKV) architecture follows a non-transformer architecture to eliminate the drawbacks of dot-product attention, where memory and computational complexity exhibits quadratic scaling with sequence length. Although RWKV has exploited a linearly tensor-product attention mechanism and achieved parallelized computations by deploying the time-sequential mode, it fails to capture long-range dependencies because of its limitation on looking back at previous information, compared with full information obtained by direct interactions in the standard transformer. Therefore, the paper devises the Retrospected Receptance Weighted Key Value (RRWKV) architecture via incorporating the retrospecting ability into the RWKV to effectively absorb information, which maintains memory and computational efficiency as 
    

