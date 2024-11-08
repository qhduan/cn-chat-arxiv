# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Grid-Mapping Pseudo-Count Constraint for Offline Reinforcement Learning](https://arxiv.org/abs/2404.02545) | 提出了一种用于连续领域的新的计数方法，称为格点映射伪计数方法（GPC），以适应离线环境中的强化学习问题，并在惩罚Q值的同时减少计算成本。 |
| [^2] | [Long-form factuality in large language models](https://arxiv.org/abs/2403.18802) | 该论文提出了一种通过使用大型语言模型将长篇回应分解为单个事实，并通过发送搜索查询到Google搜索，评估事实准确性的方法，并扩展了F1分数作为长篇事实性的聚合度量。 |
| [^3] | [Gradient Cuff: Detecting Jailbreak Attacks on Large Language Models by Exploring Refusal Loss Landscapes](https://arxiv.org/abs/2403.00867) | 本文提出了一种名为Gradient Cuff的方法，通过探索拒绝损失地形图来检测对大语言模型的越狱攻击，成功设计了一种有效的两步检测策略。 |
| [^4] | [Combinatorial Client-Master Multiagent Deep Reinforcement Learning for Task Offloading in Mobile Edge Computing](https://arxiv.org/abs/2402.11653) | 深度强化学习在移动边缘计算中的任务卸载问题中的应用面临着连续和离散资源约束的挑战，但有望实现高效的任务分配。 |
| [^5] | [Personalized Large Language Models](https://arxiv.org/abs/2402.09269) | 本文研究了个性化大型语言模型的方法，通过比较微调和零样本推理的方法，在主观任务中发现个性化微调能提高模型的推理能力，在情感识别和仇恨言论检测方面也获得了一致的性能提升。 |
| [^6] | [Robust multimodal models have outlier features and encode more concepts.](http://arxiv.org/abs/2310.13040) | 健壮的多模态模型展示了异常特征和更多概念的编码方式。 |
| [^7] | [Language Aligned Visual Representations Predict Human Behavior in Naturalistic Learning Tasks.](http://arxiv.org/abs/2306.09377) | 语言对齐的视觉表示方式比纯视觉表示方式更有效地预测人类在自然学习任务中的行为。 |
| [^8] | [Learning to Communicate and Collaborate in a Competitive Multi-Agent Setup to Clean the Ocean from Macroplastics.](http://arxiv.org/abs/2304.05872) | 本文提出了一种基于图神经网络（GNN）的、用于多智能体交互式海洋废弃物清理的通信机制，使得不同代理之间可以协作竞争并实现收集废弃物的最大化。 |
| [^9] | [Evaluating explainability for machine learning predictions using model-agnostic metrics.](http://arxiv.org/abs/2302.12094) | 本文提出了一种使用模型无关的度量标准，用于评估机器学习模型的预测结果的可解释性。这些度量标准将各个解释能力方面总结成标量，提供全面的理解并促进决策者和利益相关者之间的沟通，从而提高整体的透明度。 |

# 详细

[^1]: 离线强化学习的格点映射伪计数约束

    Grid-Mapping Pseudo-Count Constraint for Offline Reinforcement Learning

    [https://arxiv.org/abs/2404.02545](https://arxiv.org/abs/2404.02545)

    提出了一种用于连续领域的新的计数方法，称为格点映射伪计数方法（GPC），以适应离线环境中的强化学习问题，并在惩罚Q值的同时减少计算成本。

    

    离线强化学习是从静态数据集中学习而不与环境进行交互的方法，这确保了安全性并因此具有良好的应用前景。然而，直接应用朴素的强化学习方法通常在离线环境中失败，因为由于超出分布（OOD）行为引起的函数逼近误差。为了解决这个问题，现有算法主要惩罚OOD行为的Q值，其约束的质量也很重要。不精确的约束可能导致次优解，而精确的约束则需要显著的计算成本。在本文中，我们提出了一种新颖的连续领域计数方法，称为格点映射伪计数方法（GPC），以适当地惩罚Q值并减少计算成本。所提出的方法将状态和动作空间映射到离散空间，并通过伪计数约束它们的Q值。这是一个理论性

    arXiv:2404.02545v1 Announce Type: cross  Abstract: Offline reinforcement learning learns from a static dataset without interacting with the environment, which ensures security and thus owns a good prospect of application. However, directly applying naive reinforcement learning methods usually fails in an offline environment due to function approximation errors caused by out-of-distribution(OOD) actions. To solve this problem, existing algorithms mainly penalize the Q-value of OOD actions, the quality of whose constraints also matter. Imprecise constraints may lead to suboptimal solutions, while precise constraints require significant computational costs. In this paper, we propose a novel count-based method for continuous domains, called Grid-Mapping Pseudo-Count method(GPC), to penalize the Q-value appropriately and reduce the computational cost. The proposed method maps the state and action space to discrete space and constrains their Q-values through the pseudo-count. It is theoretic
    
[^2]: 大型语言模型中的长篇事实性

    Long-form factuality in large language models

    [https://arxiv.org/abs/2403.18802](https://arxiv.org/abs/2403.18802)

    该论文提出了一种通过使用大型语言模型将长篇回应分解为单个事实，并通过发送搜索查询到Google搜索，评估事实准确性的方法，并扩展了F1分数作为长篇事实性的聚合度量。

    

    大型语言模型（LLMs）在回答开放性主题的事实性提示时，经常生成包含事实错误的内容。为了在开放领域中对模型的长篇事实性进行基准测试，我们首先使用GPT-4生成了一个名为LongFact的提示集，其中包含数千个囊括38个主题的问题。然后，我们提出LLM代理可以通过一种名为Search-Augmented Factuality Evaluator（SAFE）的方法作为长篇事实性的自动评估器。SAFE利用LLM将长篇回应分解为一组单独的事实，并通过发送搜索查询到Google搜索以及确定一个事实是否得到搜索结果支持的多步推理过程来评估每个事实的准确性。此外，我们还提议将F1分数扩展为长篇事实性的聚合度量。为此，我们平衡了回应中支持事实的百分比（精度）与

    arXiv:2403.18802v1 Announce Type: cross  Abstract: Large language models (LLMs) often generate content that contains factual errors when responding to fact-seeking prompts on open-ended topics. To benchmark a model's long-form factuality in open domains, we first use GPT-4 to generate LongFact, a prompt set comprising thousands of questions spanning 38 topics. We then propose that LLM agents can be used as automated evaluators for long-form factuality through a method which we call Search-Augmented Factuality Evaluator (SAFE). SAFE utilizes an LLM to break down a long-form response into a set of individual facts and to evaluate the accuracy of each fact using a multi-step reasoning process comprising sending search queries to Google Search and determining whether a fact is supported by the search results. Furthermore, we propose extending F1 score as an aggregated metric for long-form factuality. To do so, we balance the percentage of supported facts in a response (precision) with the 
    
[^3]: 梯度被罚：通过探索拒绝损失地形图来检测针对大语言模型的越狱攻击

    Gradient Cuff: Detecting Jailbreak Attacks on Large Language Models by Exploring Refusal Loss Landscapes

    [https://arxiv.org/abs/2403.00867](https://arxiv.org/abs/2403.00867)

    本文提出了一种名为Gradient Cuff的方法，通过探索拒绝损失地形图来检测对大语言模型的越狱攻击，成功设计了一种有效的两步检测策略。

    

    大型语言模型（LLMs）正成为一种突出的生成式AI工具，用户输入查询，LLM生成答案。为了减少伤害和滥用，人们通过使用先进的训练技术如来自人类反馈的强化学习（RLHF）来将这些LLMs与人类价值观保持一致。然而，最近的研究突显了LLMs对于试图颠覆嵌入的安全防护措施的对抗性越狱尝试的脆弱性。为了解决这一挑战，本文定义并调查了LLMs的拒绝损失，然后提出了一种名为Gradient Cuff的方法来检测越狱尝试。Gradient Cuff利用拒绝损失地形图中观察到的独特特性，包括功能值及其光滑性，设计了一种有效的两步检测策略。

    arXiv:2403.00867v1 Announce Type: cross  Abstract: Large Language Models (LLMs) are becoming a prominent generative AI tool, where the user enters a query and the LLM generates an answer. To reduce harm and misuse, efforts have been made to align these LLMs to human values using advanced training techniques such as Reinforcement Learning from Human Feedback (RLHF). However, recent studies have highlighted the vulnerability of LLMs to adversarial jailbreak attempts aiming at subverting the embedded safety guardrails. To address this challenge, this paper defines and investigates the Refusal Loss of LLMs and then proposes a method called Gradient Cuff to detect jailbreak attempts. Gradient Cuff exploits the unique properties observed in the refusal loss landscape, including functional values and its smoothness, to design an effective two-step detection strategy. Experimental results on two aligned LLMs (LLaMA-2-7B-Chat and Vicuna-7B-V1.5) and six types of jailbreak attacks (GCG, AutoDAN,
    
[^4]: 移动边缘计算中组合式客户端-主控多智能体深度强化学习用于任务卸载

    Combinatorial Client-Master Multiagent Deep Reinforcement Learning for Task Offloading in Mobile Edge Computing

    [https://arxiv.org/abs/2402.11653](https://arxiv.org/abs/2402.11653)

    深度强化学习在移动边缘计算中的任务卸载问题中的应用面临着连续和离散资源约束的挑战，但有望实现高效的任务分配。

    

    最近，出现了大量执行计算密集型任务的移动应用程序，如视频流媒体、数据挖掘、虚拟现实、增强现实、图像处理、视频处理、人脸识别和在线游戏。移动边缘计算（MEC）已经成为一种满足用户设备（UDs）日益增长的计算需求的有前途的技术。MEC中的任务卸载是一种策略，通过在UDs和MEC服务器之间分配任务来满足UDs的需求。深度强化学习（DRL）在任务卸载问题中受到关注，因为它可以适应动态变化并最小化在线计算复杂性。然而，UDs和MEC服务器上的各种类型的连续和离散资源约束对有效的基于DRL的任务卸载设计构成挑战。

    arXiv:2402.11653v1 Announce Type: new  Abstract: Recently, there has been an explosion of mobile applications that perform computationally intensive tasks such as video streaming, data mining, virtual reality, augmented reality, image processing, video processing, face recognition, and online gaming. However, user devices (UDs), such as tablets and smartphones, have a limited ability to perform the computation needs of the tasks. Mobile edge computing (MEC) has emerged as a promising technology to meet the increasing computing demands of UDs. Task offloading in MEC is a strategy that meets the demands of UDs by distributing tasks between UDs and MEC servers. Deep reinforcement learning (DRL) is gaining attention in task-offloading problems because it can adapt to dynamic changes and minimize online computational complexity. However, the various types of continuous and discrete resource constraints on UDs and MEC servers pose challenges to the design of an efficient DRL-based task-offlo
    
[^5]: 个性化的大型语言模型

    Personalized Large Language Models

    [https://arxiv.org/abs/2402.09269](https://arxiv.org/abs/2402.09269)

    本文研究了个性化大型语言模型的方法，通过比较微调和零样本推理的方法，在主观任务中发现个性化微调能提高模型的推理能力，在情感识别和仇恨言论检测方面也获得了一致的性能提升。

    

    近年来，大型语言模型（LLM）在自然语言处理（NLP）任务中取得了显著的进展。然而，它们的通用性在需要个性化回应的场景（如推荐系统和聊天机器人）中存在一定的局限性。本文研究了个性化LLM的方法，比较了微调和零样本推理方法在主观任务中的效果。结果表明，与非个性化模型相比，个性化微调改善了模型的推理能力。在情感识别和仇恨言论检测的数据集上进行的实验表明，个性化方法在不同的LLM架构上获得了一致的性能提升。这些发现强调了在主观文本理解任务中提升LLM能力的个性化的重要性。

    arXiv:2402.09269v1 Announce Type: cross Abstract: Large language models (LLMs) have significantly advanced Natural Language Processing (NLP) tasks in recent years. However, their universal nature poses limitations in scenarios requiring personalized responses, such as recommendation systems and chatbots. This paper investigates methods to personalize LLMs, comparing fine-tuning and zero-shot reasoning approaches on subjective tasks. Results demonstrate that personalized fine-tuning improves model reasoning compared to non-personalized models. Experiments on datasets for emotion recognition and hate speech detection show consistent performance gains with personalized methods across different LLM architectures. These findings underscore the importance of personalization for enhancing LLM capabilities in subjective text perception tasks.
    
[^6]: 健壮的多模态模型具有异常特征并编码更多概念

    Robust multimodal models have outlier features and encode more concepts. (arXiv:2310.13040v1 [cs.LG])

    [http://arxiv.org/abs/2310.13040](http://arxiv.org/abs/2310.13040)

    健壮的多模态模型展示了异常特征和更多概念的编码方式。

    

    什么区分健壮模型与非健壮模型？随着大规模多模态模型（如CLIP）的出现，这个问题引起了人们的关注。这些模型在自然分布转变方面表现出了前所未有的健壮性。尽管已经证明了健壮性的差异可以追溯到训练数据上的差异，但迄今为止还不清楚这对于模型学习到了什么意味着。在这项工作中，我们通过探测12个具有不同骨干（ResNets和ViTs）和预训练集（OpenAI，LAION-400M，LAION-2B，YFCC15M，CC12M和DataComp）的健壮多模态模型的表示空间来填补这一空白。我们发现这些模型的表示空间中存在两个健壮性的特征：（1）健壮模型具有由其激活特征表征的异常特征，其中一些特征值比平均值高几个数量级。这些异常特征在模型的表示空间中引入了特权方向。我们证明了...

    What distinguishes robust models from non-robust ones? This question has gained traction with the appearance of large-scale multimodal models, such as CLIP. These models have demonstrated unprecedented robustness with respect to natural distribution shifts. While it has been shown that such differences in robustness can be traced back to differences in training data, so far it is not known what that translates to in terms of what the model has learned. In this work, we bridge this gap by probing the representation spaces of 12 robust multimodal models with various backbones (ResNets and ViTs) and pretraining sets (OpenAI, LAION-400M, LAION-2B, YFCC15M, CC12M and DataComp). We find two signatures of robustness in the representation spaces of these models: (1) Robust models exhibit outlier features characterized by their activations, with some being several orders of magnitude above average. These outlier features induce privileged directions in the model's representation space. We demon
    
[^7]: 对齐语言的视觉表示预测人类在自然学习任务中的行为

    Language Aligned Visual Representations Predict Human Behavior in Naturalistic Learning Tasks. (arXiv:2306.09377v1 [cs.LG])

    [http://arxiv.org/abs/2306.09377](http://arxiv.org/abs/2306.09377)

    语言对齐的视觉表示方式比纯视觉表示方式更有效地预测人类在自然学习任务中的行为。

    

    人类具备识别和概括自然物体相关特征的能力，在各种情境中有所帮助。为了研究这种现象并确定最有效的表示方式以预测人类行为，我们进行了两个涉及类别学习和奖励学习的实验。我们的实验使用逼真的图像作为刺激物，并要求参与者基于所有试验的新型刺激物作出准确的决策，因此需要泛化。在两个任务中，底层规则是使用人类相似性判断提取的刺激维度生成的简单线性函数。值得注意的是，参与者在几次试验内就成功地确定了相关的刺激特征，证明了有效的泛化。我们进行了广泛的模型比较，评估了各种深度学习模型的表示对人类选择的逐次预测准确性。有趣的是，自然语言处理任务（如语言建模和机器翻译）训练的模型表示优于视觉任务训练的模型表示，表明对齐语言的视觉表示可能更有效地预测人类在自然学习任务中的行为。

    Humans possess the ability to identify and generalize relevant features of natural objects, which aids them in various situations. To investigate this phenomenon and determine the most effective representations for predicting human behavior, we conducted two experiments involving category learning and reward learning. Our experiments used realistic images as stimuli, and participants were tasked with making accurate decisions based on novel stimuli for all trials, thereby necessitating generalization. In both tasks, the underlying rules were generated as simple linear functions using stimulus dimensions extracted from human similarity judgments. Notably, participants successfully identified the relevant stimulus features within a few trials, demonstrating effective generalization. We performed an extensive model comparison, evaluating the trial-by-trial predictive accuracy of diverse deep learning models' representations of human choices. Intriguingly, representations from models train
    
[^8]: 在竞争性多智能体环境中学习沟通和协作以清理海洋废弃塑料

    Learning to Communicate and Collaborate in a Competitive Multi-Agent Setup to Clean the Ocean from Macroplastics. (arXiv:2304.05872v1 [cs.AI])

    [http://arxiv.org/abs/2304.05872](http://arxiv.org/abs/2304.05872)

    本文提出了一种基于图神经网络（GNN）的、用于多智能体交互式海洋废弃物清理的通信机制，使得不同代理之间可以协作竞争并实现收集废弃物的最大化。

    

    在许多实际应用中，协作与竞争之间的平衡对于人工智能代理至关重要。本文使用多智能体强化学习（MARL）建立在一个高影响问题上，通过对海洋废弃塑料的收集实现了协作与竞争的平衡。我们提出了一种基于图神经网络（GNN）的通信机制，它增加了代理的观察空间。在我们自定义的环境中，代理控制着收集塑料的船只。这种通信机制使代理能够使用二进制信号来开发通信协议。虽然代理的集体目标是尽可能地清理海洋废弃塑料，但代理会因个人收集到的废弃塑料数量而获得奖励。因此，代理必须学会有效地沟通并保持竞争关系。

    Finding a balance between collaboration and competition is crucial for artificial agents in many real-world applications. We investigate this using a Multi-Agent Reinforcement Learning (MARL) setup on the back of a high-impact problem. The accumulation and yearly growth of plastic in the ocean cause irreparable damage to many aspects of oceanic health and the marina system. To prevent further damage, we need to find ways to reduce macroplastics from known plastic patches in the ocean. Here we propose a Graph Neural Network (GNN) based communication mechanism that increases the agents' observation space. In our custom environment, agents control a plastic collecting vessel. The communication mechanism enables agents to develop a communication protocol using a binary signal. While the goal of the agent collective is to clean up as much as possible, agents are rewarded for the individual amount of macroplastics collected. Hence agents have to learn to communicate effectively while maintai
    
[^9]: 评估使用模型无关的度量标准解释机器学习预测的可解释性

    Evaluating explainability for machine learning predictions using model-agnostic metrics. (arXiv:2302.12094v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.12094](http://arxiv.org/abs/2302.12094)

    本文提出了一种使用模型无关的度量标准，用于评估机器学习模型的预测结果的可解释性。这些度量标准将各个解释能力方面总结成标量，提供全面的理解并促进决策者和利益相关者之间的沟通，从而提高整体的透明度。

    

    人工智能技术的快速发展带来了管理和监管方面的众多挑战。人工智能系统正在被整合到各个行业和领域，决策者需全面细致地了解这些系统的能力和限制。这个需求的一个关键方面是能够解释机器学习模型的结果，这对于提高透明度和信任度以及帮助模型在道德上进行训练至关重要。本文提出了新颖的度量标准，用于量化AI模型预测结果是否可以通过其特征进行易于解释。我们的度量标准将解释能力的不同方面总结为标量，提供对模型预测的更全面的理解，促进决策者和利益相关者之间的沟通，从而提高整体的透明度。

    Rapid advancements in artificial intelligence (AI) technology have brought about a plethora of new challenges in terms of governance and regulation. AI systems are being integrated into various industries and sectors, creating a demand from decision-makers to possess a comprehensive and nuanced understanding of the capabilities and limitations of these systems. One critical aspect of this demand is the ability to explain the results of machine learning models, which is crucial to promoting transparency and trust in AI systems, as well as fundamental in helping machine learning models to be trained ethically. In this paper, we present novel metrics to quantify the degree of which AI model predictions can be easily explainable by its features. Our metrics summarize different aspects of explainability into scalars, providing a more comprehensive understanding of model predictions and facilitating communication between decision-makers and stakeholders, thereby increasing the overall transp
    

