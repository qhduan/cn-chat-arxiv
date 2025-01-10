# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Synergizing Spatial Optimization with Large Language Models for Open-Domain Urban Itinerary Planning](https://arxiv.org/abs/2402.07204) | 本文提出了Open-domain Urban Itinerary Planning (OUIP)任务，用于根据用户以自然语言描述的请求直接生成行程，通过结合空间优化和大型语言模型(LLM)，提供个性化的城市行程定制服务。 |
| [^2] | [Driving Towards Inclusion: Revisiting In-Vehicle Interaction in Autonomous Vehicles.](http://arxiv.org/abs/2401.14571) | 本文综述了自动驾驶车辆中车内人机交互的现状和新兴技术，并提出了以用户为中心的包容性HCI设计原则，旨在增强乘客体验。 |
| [^3] | [RAPGen: An Approach for Fixing Code Inefficiencies in Zero-Shot.](http://arxiv.org/abs/2306.17077) | RAPGen是一种新方法，通过在零样本情况下使用Retrieval-Augmented Prompt Generation（RAPGen）方法，即从预先构建的性能Bug修复知识库中检索提示指令并生成提示，然后在大型语言模型上生成修复方案，可以有效地解决代码低效问题。实验结果显示，在专家验证的数据集中，RAPGen在60%的情况下可以生成与开发者等效或更好的性能改进建议，其中约39%的建议完全相同。 |
| [^4] | [Explaining the Behavior of Black-Box Prediction Algorithms with Causal Learning.](http://arxiv.org/abs/2006.02482) | 本文提出了一种用于解释黑箱预测算法行为的因果学习方法，通过学习因果图表示来提供因果解释，弥补了现有方法的缺点，即解释单元更加可解释且考虑了宏观级特征和未测量的混淆。 |

# 详细

[^1]: 结合空间优化和大型语言模型的开放领域城市行程规划

    Synergizing Spatial Optimization with Large Language Models for Open-Domain Urban Itinerary Planning

    [https://arxiv.org/abs/2402.07204](https://arxiv.org/abs/2402.07204)

    本文提出了Open-domain Urban Itinerary Planning (OUIP)任务，用于根据用户以自然语言描述的请求直接生成行程，通过结合空间优化和大型语言模型(LLM)，提供个性化的城市行程定制服务。

    

    本文首次提出了Open-domain Urban Itinerary Planning (OUIP)任务，用于根据用户以自然语言描述的请求直接生成行程。OUIP与传统行程规划不同，传统规划限制了用户表达更详细的需求，阻碍了真正的个性化。最近，大型语言模型(LLM)在处理多样化任务方面表现出潜力。然而，由于非实时信息、不完整的知识和不足的空间意识，它们无法独立地提供满意的用户体验。鉴于此，我们提出了一个名为ItiNera的OUIP系统，将空间优化与大型语言模型(LLM)相结合，根据用户需求提供个性化的城市行程定制服务。具体来说，我们开发了一个基于LLM的流水线，用于提取和更新兴趣点特征，以创建用户自己的个性化兴趣点数据库。对于每个用户请求，我们利用LLM进行协同实现优化。

    In this paper, we for the first time propose the task of Open-domain Urban Itinerary Planning (OUIP) for citywalk, which directly generates itineraries based on users' requests described in natural language. OUIP is different from conventional itinerary planning, which limits users from expressing more detailed needs and hinders true personalization. Recently, large language models (LLMs) have shown potential in handling diverse tasks. However, due to non-real-time information, incomplete knowledge, and insufficient spatial awareness, they are unable to independently deliver a satisfactory user experience in OUIP. Given this, we present ItiNera, an OUIP system that synergizes spatial optimization with Large Language Models (LLMs) to provide services that customize urban itineraries based on users' needs. Specifically, we develop an LLM-based pipeline for extracting and updating POI features to create a user-owned personalized POI database. For each user request, we leverage LLM in coop
    
[^2]: 向包容性驱动：重新审视自动驾驶车辆中的车内交互

    Driving Towards Inclusion: Revisiting In-Vehicle Interaction in Autonomous Vehicles. (arXiv:2401.14571v1 [cs.HC])

    [http://arxiv.org/abs/2401.14571](http://arxiv.org/abs/2401.14571)

    本文综述了自动驾驶车辆中车内人机交互的现状和新兴技术，并提出了以用户为中心的包容性HCI设计原则，旨在增强乘客体验。

    

    本文综述了目前自动驾驶车辆中车内人机交互（HCI）的现状，特别关注包容性和可访问性。本研究旨在考察自动驾驶车辆中包容性HCI的以用户为中心的设计原则，评估现有HCI系统，并确定可能增强乘客体验的新兴技术。本文首先概述了自动驾驶车辆技术的现状，然后对这一背景下HCI的重要性进行了分析。接下来，本文综述了包容性HCI设计原则的现有文献，并评估了当前自动驾驶车辆中HCI系统的有效性。本文还确定了可能增强乘客体验的新兴技术，如语音激活界面、触觉反馈系统和增强现实显示。最后，本文总结了研究的重要发现，并讨论了未来的研究方向。

    This paper presents a comprehensive literature review of the current state of in-vehicle human-computer interaction (HCI) in the context of self-driving vehicles, with a specific focus on inclusion and accessibility. This study's aim is to examine the user-centered design principles for inclusive HCI in self-driving vehicles, evaluate existing HCI systems, and identify emerging technologies that have the potential to enhance the passenger experience. The paper begins by providing an overview of the current state of self-driving vehicle technology, followed by an examination of the importance of HCI in this context. Next, the paper reviews the existing literature on inclusive HCI design principles and evaluates the effectiveness of current HCI systems in self-driving vehicles. The paper also identifies emerging technologies that have the potential to enhance the passenger experience, such as voice-activated interfaces, haptic feedback systems, and augmented reality displays. Finally, th
    
[^3]: RAPGen: 一种解决零样本代码低效问题的方法

    RAPGen: An Approach for Fixing Code Inefficiencies in Zero-Shot. (arXiv:2306.17077v1 [cs.SE])

    [http://arxiv.org/abs/2306.17077](http://arxiv.org/abs/2306.17077)

    RAPGen是一种新方法，通过在零样本情况下使用Retrieval-Augmented Prompt Generation（RAPGen）方法，即从预先构建的性能Bug修复知识库中检索提示指令并生成提示，然后在大型语言模型上生成修复方案，可以有效地解决代码低效问题。实验结果显示，在专家验证的数据集中，RAPGen在60%的情况下可以生成与开发者等效或更好的性能改进建议，其中约39%的建议完全相同。

    

    性能Bug是一种即使在经过充分测试的商业产品中也可能出现的非功能性问题。修复这些性能Bug是一个重要但具有挑战性的问题。在这项工作中，我们解决了这个挑战，并提出了一种名为Retrieval-Augmented Prompt Generation（RAPGen）的新方法。给定一个存在性能问题的代码片段，RAPGen首先从预先构建的之前性能Bug修复知识库中检索一个提示指令，然后使用检索到的指令生成一个提示。然后，它在零样本情况下使用这个提示在大型语言模型（如Codex）上生成一个修复方案。我们将我们的方法与各种提示变体和现有方法在性能Bug修复任务中进行了比较。我们的评估结果显示，RAPGen在60%的情况下可以生成与开发者等效或更好的性能改进建议，在经过专家验证的过去C#开发者所做的性能更改数据集中有约39%的建议完全相同。

    Performance bugs are non-functional bugs that can even manifest in well-tested commercial products. Fixing these performance bugs is an important yet challenging problem. In this work, we address this challenge and present a new approach called Retrieval-Augmented Prompt Generation (RAPGen). Given a code snippet with a performance issue, RAPGen first retrieves a prompt instruction from a pre-constructed knowledge-base of previous performance bug fixes and then generates a prompt using the retrieved instruction. It then uses this prompt on a Large Language Model (such as Codex) in zero-shot to generate a fix. We compare our approach with the various prompt variations and state of the art methods in the task of performance bug fixing. Our evaluation shows that RAPGen can generate performance improvement suggestions equivalent or better than a developer in ~60% of the cases, getting ~39% of them verbatim, in an expert-verified dataset of past performance changes made by C# developers.
    
[^4]: 用因果学习解释黑箱预测算法的行为

    Explaining the Behavior of Black-Box Prediction Algorithms with Causal Learning. (arXiv:2006.02482v4 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2006.02482](http://arxiv.org/abs/2006.02482)

    本文提出了一种用于解释黑箱预测算法行为的因果学习方法，通过学习因果图表示来提供因果解释，弥补了现有方法的缺点，即解释单元更加可解释且考虑了宏观级特征和未测量的混淆。

    

    因果学方法在解释黑箱预测模型（例如基于图像像素数据训练的深度神经网络）方面越来越受欢迎。然而，现有方法存在两个重要缺点：（i）“解释单元”是相关预测模型的微观级输入，例如图像像素，而不是更有用于理解如何可能改变算法行为的可解释的宏观级特征；（ii）现有方法假设特征与目标模型预测之间不存在未测量的混淆，这在解释单元是宏观级变量时不成立。我们关注的是在分析人员无法访问目标预测算法内部工作原理的重要情况，而只能根据特定输入查询模型输出的能力。为了在这种情况下提供因果解释，我们提出学习因果图表示，允许更好地理解算法的行为。

    Causal approaches to post-hoc explainability for black-box prediction models (e.g., deep neural networks trained on image pixel data) have become increasingly popular. However, existing approaches have two important shortcomings: (i) the "explanatory units" are micro-level inputs into the relevant prediction model, e.g., image pixels, rather than interpretable macro-level features that are more useful for understanding how to possibly change the algorithm's behavior, and (ii) existing approaches assume there exists no unmeasured confounding between features and target model predictions, which fails to hold when the explanatory units are macro-level variables. Our focus is on the important setting where the analyst has no access to the inner workings of the target prediction algorithm, rather only the ability to query the output of the model in response to a particular input. To provide causal explanations in such a setting, we propose to learn causal graphical representations that allo
    

