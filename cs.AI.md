# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Dynamics-Guided Diffusion Model for Robot Manipulator Design](https://arxiv.org/abs/2402.15038) | 该论文提出了动态引导扩散模型，利用共享的动力学网络为不同操作任务生成 manipulator 几何设计，通过设计目标构建的梯度引导手指几何设计的完善过程。 |
| [^2] | [WildfireGPT: Tailored Large Language Model for Wildfire Analysis](https://arxiv.org/abs/2402.07877) | WildfireGPT是一个针对野火分析的定制化大型语言模型，通过提供领域特定的上下文信息和科学准确性，将用户查询转化为关于野火风险的可操作见解。 |
| [^3] | [Self-Rewarding Language Models.](http://arxiv.org/abs/2401.10020) | 该论文提出了自奖励语言模型的概念，通过LLM作为评判者，使用语言模型自己提供训练过程中的奖励。研究表明，该方法不仅可以提高指令遵循能力，还可以为自己提供高质量的奖励。通过对Llama 2 70B模型的三次迭代微调，结果在AlpacaEval 2.0排行榜上超过了其他现有系统。这项工作为实现能够不断自我改进的模型开辟了新的可能性。 |
| [^4] | [Sherlock Holmes Doesn't Play Dice: The significance of Evidence Theory for the Social and Life Sciences.](http://arxiv.org/abs/2309.03222) | 本文强调了Evidence Theory在社会与生命科学中的潜力，可以表达来自未确定事件可能实现的不确定性，而与之对比的Probability Theory只能限于决策者当前设想的可能性。本文还讨论了Evidence Theory与Probability Theory的关系以及Evidence Theory在信息论中的应用增强效果，并通过审计练习案例进一步说明了Evidence Theory的应用。 |

# 详细

[^1]: 动态引导扩散模型用于机器人 manipulator 设计

    Dynamics-Guided Diffusion Model for Robot Manipulator Design

    [https://arxiv.org/abs/2402.15038](https://arxiv.org/abs/2402.15038)

    该论文提出了动态引导扩散模型，利用共享的动力学网络为不同操作任务生成 manipulator 几何设计，通过设计目标构建的梯度引导手指几何设计的完善过程。

    

    我们提出了一个名为动态引导扩散模型的数据驱动框架，用于为给定操作任务生成 manipulator 几何设计。与为每个任务训练不同的设计模型不同，我们的方法采用一个跨任务共享的学习动力学网络。对于新的操作任务，我们首先将其分解为一组称为目标相互作用配置文件的个别运动目标，其中每个个别运动可以由共享的动力学网络建模。从目标和预测的相互作用配置文件构建的设计目标为任务的手指几何设计提供了梯度引导。这个设计过程被执行为一种分类器引导的扩散过程，其中设计目标作为分类器引导。我们在只使用开环平行夹爪运动的无传感器设置下，在各种操作任务上评估了我们的框架。

    arXiv:2402.15038v1 Announce Type: cross  Abstract: We present Dynamics-Guided Diffusion Model, a data-driven framework for generating manipulator geometry designs for a given manipulation task. Instead of training different design models for each task, our approach employs a learned dynamics network shared across tasks. For a new manipulation task, we first decompose it into a collection of individual motion targets which we call target interaction profile, where each individual motion can be modeled by the shared dynamics network. The design objective constructed from the target and predicted interaction profiles provides a gradient to guide the refinement of finger geometry for the task. This refinement process is executed as a classifier-guided diffusion process, where the design objective acts as the classifier guidance. We evaluate our framework on various manipulation tasks, under the sensor-less setting using only an open-loop parallel jaw motion. Our generated designs outperfor
    
[^2]: WildfireGPT：针对野火分析的定制化大型语言模型

    WildfireGPT: Tailored Large Language Model for Wildfire Analysis

    [https://arxiv.org/abs/2402.07877](https://arxiv.org/abs/2402.07877)

    WildfireGPT是一个针对野火分析的定制化大型语言模型，通过提供领域特定的上下文信息和科学准确性，将用户查询转化为关于野火风险的可操作见解。

    

    大型语言模型（LLMs）的最新进展代表了人工智能（AI）和机器学习（ML）领域的一种变革性能力。然而，LLMs是通用模型，训练于广泛的文本语料库，往往难以提供特定上下文信息，尤其是在需要专业知识的领域，比如野火细节在更广泛的气候变化背景下。对于关注野火弹性和适应性的决策者和政策制定者来说，获取不仅准确而且领域特定的响应至关重要，而不是泛泛而谈。为此，我们开发了WildfireGPT，一个原型LLM代理，旨在将用户查询转化为关于野火风险的可操作见解。我们通过提供气候预测和科学文献等额外上下文信息来丰富WildfireGPT，以确保其信息具有时效性、相关性和科学准确性。这使得WildfireGPT成为一个有效的工具来解决实际问题。

    The recent advancement of large language models (LLMs) represents a transformational capability at the frontier of artificial intelligence (AI) and machine learning (ML). However, LLMs are generalized models, trained on extensive text corpus, and often struggle to provide context-specific information, particularly in areas requiring specialized knowledge such as wildfire details within the broader context of climate change. For decision-makers and policymakers focused on wildfire resilience and adaptation, it is crucial to obtain responses that are not only precise but also domain-specific, rather than generic. To that end, we developed WildfireGPT, a prototype LLM agent designed to transform user queries into actionable insights on wildfire risks. We enrich WildfireGPT by providing additional context such as climate projections and scientific literature to ensure its information is current, relevant, and scientifically accurate. This enables WildfireGPT to be an effective tool for del
    
[^3]: 自奖励语言模型

    Self-Rewarding Language Models. (arXiv:2401.10020v1 [cs.CL])

    [http://arxiv.org/abs/2401.10020](http://arxiv.org/abs/2401.10020)

    该论文提出了自奖励语言模型的概念，通过LLM作为评判者，使用语言模型自己提供训练过程中的奖励。研究表明，该方法不仅可以提高指令遵循能力，还可以为自己提供高质量的奖励。通过对Llama 2 70B模型的三次迭代微调，结果在AlpacaEval 2.0排行榜上超过了其他现有系统。这项工作为实现能够不断自我改进的模型开辟了新的可能性。

    

    我们假设要实现超人级的智能体，未来的模型需要超人级的反馈，以提供足够的训练信号。目前的方法通常是从人类偏好中训练奖励模型，这可能会受到人类表现水平的限制，而且这些独立的冻结奖励模型在LLM训练过程中无法学习改进。在这项工作中，我们研究了自奖励语言模型，其中语言模型本身通过LLM作为评判者的提示在训练过程中提供自己的奖励。我们表明，在迭代DPO训练中，不仅指令遵循能力得到了提高，而且能够为自己提供高质量的奖励。通过对Llama 2 70B进行我们方法的三次迭代的微调，得到的模型在AlpacaEval 2.0排行榜上胜过许多现有系统，包括Claude 2、Gemini Pro和GPT-4 0613。虽然这只是一项初步研究，但这项工作为可能实现能够不断自我改进的模型打开了大门。

    We posit that to achieve superhuman agents, future models require superhuman feedback in order to provide an adequate training signal. Current approaches commonly train reward models from human preferences, which may then be bottlenecked by human performance level, and secondly these separate frozen reward models cannot then learn to improve during LLM training. In this work, we study Self-Rewarding Language Models, where the language model itself is used via LLM-as-a-Judge prompting to provide its own rewards during training. We show that during Iterative DPO training that not only does instruction following ability improve, but also the ability to provide high-quality rewards to itself. Fine-tuning Llama 2 70B on three iterations of our approach yields a model that outperforms many existing systems on the AlpacaEval 2.0 leaderboard, including Claude 2, Gemini Pro, and GPT-4 0613. While only a preliminary study, this work opens the door to the possibility of models that can continuall
    
[^4]: Sherlock Holmes不玩骰子：Evidence Theory对社会与生命科学的意义

    Sherlock Holmes Doesn't Play Dice: The significance of Evidence Theory for the Social and Life Sciences. (arXiv:2309.03222v1 [cs.AI])

    [http://arxiv.org/abs/2309.03222](http://arxiv.org/abs/2309.03222)

    本文强调了Evidence Theory在社会与生命科学中的潜力，可以表达来自未确定事件可能实现的不确定性，而与之对比的Probability Theory只能限于决策者当前设想的可能性。本文还讨论了Evidence Theory与Probability Theory的关系以及Evidence Theory在信息论中的应用增强效果，并通过审计练习案例进一步说明了Evidence Theory的应用。

    

    虽然Evidence Theory（Demster-Shafer Theory，Belief Functions Theory）在数据融合中得到越来越多的应用，但其在社会与生命科学中的潜力常常被人们对其独特特点的缺乏认识所掩盖。本文强调，Evidence Theory可以表达来自于人们未能确定的事件可能实现的不确定性，而Probability Theory必须仅限于决策者当前所设想的可能性。随后，我们说明了Dempster-Shafer的组合规则如何与各种版本的Probability Theory的贝叶斯定理相关，并讨论了哪些信息论的应用可以通过Evidence Theory得到增强。最后，我们通过一个案例阐述了使用Evidence Theory来理解审计练习中出现的部分重叠、部分相互矛盾的解决方案的情况。

    While Evidence Theory (Demster-Shafer Theory, Belief Functions Theory) is being increasingly used in data fusion, its potentialities in the Social and Life Sciences are often obscured by lack of awareness of its distinctive features. With this paper we stress that Evidence Theory can express the uncertainty deriving from the fear that events may materialize, that one has not been able to figure out. By contrast, Probability Theory must limit itself to the possibilities that a decision-maker is currently envisaging.  Subsequently, we illustrate how Dempster-Shafer's combination rule relates to Bayes' Theorem for various versions of Probability Theory and discuss which applications of Information Theory can be enhanced by Evidence Theory. Finally, we illustrate our claims with an example where Evidence Theory is used to make sense of the partially overlapping, partially contradictory solutions that appear in an auditing exercise.
    

