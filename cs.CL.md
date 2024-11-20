# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Divide-or-Conquer? Which Part Should You Distill Your LLM?](https://arxiv.org/abs/2402.15000) | 本文提出了一种将推理任务分解为问题分解阶段和问题解决阶段的策略，发现问题分解阶段相比问题解决更容易提炼为较小模型，并证实该策略胜过单阶段解决方案。 |
| [^2] | [Findings of the First Workshop on Simulating Conversational Intelligence in Chat](https://arxiv.org/abs/2402.06420) | 第一届模拟对话智能研讨会的目标是汇集对开放领域对话研究进行实时人类评估的模拟智能对话模型。论文主要提供了共享任务的概述，并附上了一个将在研讨会后发布的深入分析共享任务结果的链接。 |
| [^3] | [Re-Reading Improves Reasoning in Language Models.](http://arxiv.org/abs/2309.06275) | 许多研究关注于如何引导和结构化大型语言模型的推理过程，但很少有研究关注于输入问题本身。本研究引入了一种称为“重新阅读”的提示策略，通过深入阅读输入提示中的问题信息，提供了更深入的洞察、更准确的模式识别和更有效的推理能力。 |
| [^4] | [Investigating the Factual Knowledge Boundary of Large Language Models with Retrieval Augmentation.](http://arxiv.org/abs/2307.11019) | 本研究初步分析了大型语言模型的事实知识边界，并研究了检索增强对开放域问答任务中大型语言模型的影响。结果显示大型语言模型在回答问题时表现出自信，并且回答准确。 |
| [^5] | [How to Choose How to Choose Your Chatbot: A Massively Multi-System MultiReference Data Set for Dialog Metric Evaluation.](http://arxiv.org/abs/2305.14533) | 该研究发布了MMSMR数据集，该数据集包含8个参考对话，旨在促进对话度量和评估的未来工作。该研究使用1750个系统对其进行了评估，以了解稳健相关性并了解测试集中所需的内容。 |

# 详细

[^1]: 划分还是征服？你应该提炼LLM的哪一部分？

    Divide-or-Conquer? Which Part Should You Distill Your LLM?

    [https://arxiv.org/abs/2402.15000](https://arxiv.org/abs/2402.15000)

    本文提出了一种将推理任务分解为问题分解阶段和问题解决阶段的策略，发现问题分解阶段相比问题解决更容易提炼为较小模型，并证实该策略胜过单阶段解决方案。

    

    最近的研究表明，大型语言模型（LLMs）在被鼓励先解决主要任务的子任务时可以更好地解决推理任务。本文设计了一种类似的策略，将推理任务分解为问题分解阶段和问题解决阶段，并展示该策略能够胜过单阶段解决方案。此外，我们假设与解决问题相比，分解阶段更容易被提炼为较小的模型，因为后者需要大量的领域知识，而前者只需要学习一般的问题解决策略。我们提出了提炼这两种能力的方法，并评估了它们对推理结果和推理成本的影响。我们发现我们可以提炼问题分解阶段，并同时在任务、数据集和模型之间实现良好的泛化。然而，要提炼问题解决阶段就更困难了。

    arXiv:2402.15000v1 Announce Type: new  Abstract: Recent methods have demonstrated that Large Language Models (LLMs) can solve reasoning tasks better when they are encouraged to solve subtasks of the main task first. In this paper we devise a similar strategy that breaks down reasoning tasks into a problem decomposition phase and a problem solving phase and show that the strategy is able to outperform a single stage solution. Further, we hypothesize that the decomposition should be easier to distill into a smaller model compared to the problem solving because the latter requires large amounts of domain knowledge while the former only requires learning general problem solving strategies. We propose methods to distill these two capabilities and evaluate their impact on reasoning outcomes and inference cost. We find that we can distill the problem decomposition phase and at the same time achieve good generalization across tasks, datasets, and models. However, it is harder to distill the pr
    
[^2]: 第一届模拟对话智能研讨会的研究结果

    Findings of the First Workshop on Simulating Conversational Intelligence in Chat

    [https://arxiv.org/abs/2402.06420](https://arxiv.org/abs/2402.06420)

    第一届模拟对话智能研讨会的目标是汇集对开放领域对话研究进行实时人类评估的模拟智能对话模型。论文主要提供了共享任务的概述，并附上了一个将在研讨会后发布的深入分析共享任务结果的链接。

    

    本研讨会旨在汇集从事开放领域对话研究的专家。在这个快速发展的研究领域中仍然存在许多挑战，如从对话中学习信息、进行真实和令人信服的人工智能和推理模拟。SCI-CHAT是之前关于开放领域对话的研讨会的延续，但着重于模拟智能对话，并通过人类评估来判断其质量。模型的目标是在多轮对话中能够跟随一个具有挑战性的主题，同时提出、反驳和推理论证。该研讨会包括研究路径和共享任务。本文的主要目标是概述共享任务，并提供一个链接，链接将包含在研讨会上展示后对共享任务结果进行深入分析的另一篇论文。

    The aim of this workshop is to bring together experts working on open-domain dialogue research. In this speedily advancing research area many challenges still exist, such as learning information from conversations, engaging in realistic and convincing simulation of human intelligence and reasoning. SCI-CHAT follows previous workshops on open domain dialogue but with a focus on the simulation of intelligent conversation as judged in a live human evaluation. Models aim to include the ability to follow a challenging topic over a multi-turn conversation, while positing, refuting and reasoning over arguments. The workshop included both a research track and shared task. The main goal of this paper is to provide an overview of the shared task and a link to an additional paper that will include an in depth analysis of the shared task results following presentation at the workshop.
    
[^3]: 重新阅读改善语言模型的推理能力

    Re-Reading Improves Reasoning in Language Models. (arXiv:2309.06275v1 [cs.CL])

    [http://arxiv.org/abs/2309.06275](http://arxiv.org/abs/2309.06275)

    许多研究关注于如何引导和结构化大型语言模型的推理过程，但很少有研究关注于输入问题本身。本研究引入了一种称为“重新阅读”的提示策略，通过深入阅读输入提示中的问题信息，提供了更深入的洞察、更准确的模式识别和更有效的推理能力。

    

    推理对于大型语言模型（LLM）是一个重要而具有挑战性的问题。目前的研究主要集中在开发多样化的提示策略，以引导和结构化LLM的推理过程。然而，这些基于仅解码的因果语言模型的方法通常在单个前向传递中操作输入问题，可能会忽略人类推理中丰富的前后交互。对于嵌入在提示中的输入问题这一关键维度，目前关注较少。为此，我们引入了一种简单但高效的提示策略，称为“重新阅读”。从人类学习和问题解决中汲取灵感，重新阅读意味着重访嵌在输入提示中的问题信息。这种方法与认知增强的原则完美契合，使LLM能够深入洞察、识别复杂的模式、建立 mor

    Reasoning presents a significant and challenging issue for Large Language Models (LLMs). The predominant focus of research has revolved around developing diverse prompting strategies to guide and structure the reasoning processes of LLMs. However, these approaches based on decoder-only causal language models often operate the input question in a single forward pass, potentially missing the rich, back-and-forth interactions inherent in human reasoning. Scant attention has been paid to a critical dimension, i.e., the input question itself embedded within the prompts. In response, we introduce a deceptively simple yet highly effective prompting strategy, termed question "re-reading". Drawing inspiration from human learning and problem-solving, re-reading entails revisiting the question information embedded within input prompts. This approach aligns seamlessly with the cognitive principle of reinforcement, enabling LLMs to extract deeper insights, identify intricate patterns, establish mor
    
[^4]: 用检索增强研究大型语言模型的事实知识边界

    Investigating the Factual Knowledge Boundary of Large Language Models with Retrieval Augmentation. (arXiv:2307.11019v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2307.11019](http://arxiv.org/abs/2307.11019)

    本研究初步分析了大型语言模型的事实知识边界，并研究了检索增强对开放域问答任务中大型语言模型的影响。结果显示大型语言模型在回答问题时表现出自信，并且回答准确。

    

    知识密集型任务（例如，开放域问答（QA））需要大量的事实知识，并经常依赖外部信息进行协助。最近，大型语言模型（LLMs）（例如，ChatGPT）在解决包括知识密集型任务在内的各种任务上展现出了惊人的能力。然而，目前尚不清楚LLMs在感知其事实知识边界方面表现如何，特别是在使用检索增强时的行为。在本研究中，我们对LLMs的事实知识边界进行了初步分析，并研究了检索增强对LLMs在开放域QA上的影响。具体而言，我们关注了三个主要研究问题，并通过检查LLMs的QA性能、先验判断和后验判断来进行分析。我们提供了证据表明LLMs对于自己回答问题的能力和回答的准确性充满了自信。

    Knowledge-intensive tasks (e.g., open-domain question answering (QA)) require a substantial amount of factual knowledge and often rely on external information for assistance. Recently, large language models (LLMs) (e.g., ChatGPT), have demonstrated impressive prowess in solving a wide range of tasks with world knowledge, including knowledge-intensive tasks. However, it remains unclear how well LLMs are able to perceive their factual knowledge boundaries, particularly how they behave when incorporating retrieval augmentation. In this study, we present an initial analysis of the factual knowledge boundaries of LLMs and how retrieval augmentation affects LLMs on open-domain QA. Specially, we focus on three primary research questions and analyze them by examining QA performance, priori judgement and posteriori judgement of LLMs. We show evidence that LLMs possess unwavering confidence in their capabilities to respond to questions and the accuracy of their responses. Furthermore, retrieval 
    
[^5]: 如何选择您的聊天机器人：用于对话指标评估的大规模多系统多参考数据集

    How to Choose How to Choose Your Chatbot: A Massively Multi-System MultiReference Data Set for Dialog Metric Evaluation. (arXiv:2305.14533v1 [cs.CL])

    [http://arxiv.org/abs/2305.14533](http://arxiv.org/abs/2305.14533)

    该研究发布了MMSMR数据集，该数据集包含8个参考对话，旨在促进对话度量和评估的未来工作。该研究使用1750个系统对其进行了评估，以了解稳健相关性并了解测试集中所需的内容。

    

    我们发布了MMSMR，这是一个大规模多系统多参考数据集，旨在促进对话的度量和评估的未来工作。用于对话评估的自动指标应该是人类判断的可靠代理；然而，目前对其稳健性的验证还远远不够令人满意。为了量化稳健性相关性并了解测试集中所需的内容，我们扩展了单参考评估集，推出了一个包含8个参考对话的数据集，并介绍了这个新的语言学习对话数据集。然后我们训练了1750个系统，并在我们的新测试集和DailyDialog数据集上对它们进行了评估。我们发布了这个新的测试集，以及每个系统在各种数据集上的模型超参数、推理输出和指标分数。

    We release MMSMR, a Massively Multi-System MultiReference dataset to enable future work on metrics and evaluation for dialog. Automatic metrics for dialogue evaluation should be robust proxies for human judgments; however, the verification of robustness is currently far from satisfactory. To quantify the robustness correlation and understand what is necessary in a test set, we create and release an 8-reference dialog dataset by extending single-reference evaluation sets and introduce this new language learning conversation dataset. We then train 1750 systems and evaluate them on our novel test set and the DailyDialog dataset. We release the novel test set, and model hyper parameters, inference outputs, and metric scores for each system on a variety of datasets.
    

