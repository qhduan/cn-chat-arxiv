# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [BAT: Learning to Reason about Spatial Sounds with Large Language Models](https://rss.arxiv.org/abs/2402.01591) | 本文提出了BAT，它结合了双耳声音场景分析模型的空间声音感知能力和大规模语言模型的自然语言推理能力，以复制人类的空间声音推理能力。通过使用合成的双耳音频数据集和基于空间声音的问答数据集进行训练，BAT在空间声音感知和推理方面取得了强大的性能。 |
| [^2] | [Streaming Sequence Transduction through Dynamic Compression](https://rss.arxiv.org/abs/2402.01172) | STAR是一种新型的Transformer模型，通过动态压缩和优化延迟、内存占用和质量，实现对流的高效序列转导，并在自动语音识别领域表现出色。 |
| [^3] | [Controlled Training Data Generation with Diffusion Models](https://arxiv.org/abs/2403.15309) | 提出了一种使用扩散模型生成控制训练数据的方法，通过两个反馈机制，一方面使用监督模型反馈找到对抗性提示词实现图像生成，另一方面通过引导使生成过程朝向特定目标分布。 |
| [^4] | [Usable XAI: 10 Strategies Towards Exploiting Explainability in the LLM Era](https://arxiv.org/abs/2403.08946) | 在大型语言模型时代，为了适应其复杂性和先进能力，我们引入了可用的XAI概念，通过积极增强LLMs在实际环境中的生产力和适用性，实现XAI方法论的重大转变。 |
| [^5] | [Fine-Tuning, Prompting, In-Context Learning and Instruction-Tuning: How Many Labelled Samples Do We Need?](https://arxiv.org/abs/2402.12819) | 专门模型通常只需少量标记样本（100-1000个）就能与通用模型持平甚至更好，取决于任务的复杂性和结果的变化。 |
| [^6] | [FormulaQA: A Question Answering Dataset for Formula-Based Numerical Reasoning](https://arxiv.org/abs/2402.12692) | FormulaQA是一个基于初中物理考试的公式驱动数值推理问题问答数据集，通过评估LLMs的不同方法和使用检索增强型LLMs以及对小型模型进行微调，揭示了现有模型在应对复杂、基于公式的FormulaQA时的潜在改进空间。 |
| [^7] | [Can We Verify Step by Step for Incorrect Answer Detection?](https://arxiv.org/abs/2402.10528) | 通过推理链来预测大型语言模型输出的准确性，我们引入了一个新的基准R2PE，并提出了处理可辨识性评分（PDS）框架。 |
| [^8] | [An Approach to Automatically generating Riddles aiding Concept Attainment.](http://arxiv.org/abs/2310.18290) | 这个论文介绍了一种自动生成概念谜题的方法，以促进在线学习环境中的学习者参与度。通过应用概念达成模型和生成谜题，该方法可以帮助学习者更好地理解概念。 |
| [^9] | [Cross-Lingual Consistency of Factual Knowledge in Multilingual Language Models.](http://arxiv.org/abs/2310.10378) | 本论文研究了多语言预训练语言模型中事实知识的跨语言一致性，提出了一种新的度量方法，并通过分析模型大小、语言配对等因素发现了影响一致性的因素。实验结果表明，增加模型大小可以提高准确性，但不会改善跨语言一致性。 |
| [^10] | [PlanFitting: Tailoring Personalized Exercise Plans with Large Language Models.](http://arxiv.org/abs/2309.12555) | PlanFitting是一个对话型人工智能，利用大型语言模型的生成能力帮助用户定制个性化的运动计划，并在用户研究中证明了它生成个性化、可操作和有据可依的运动计划的潜力。 |
| [^11] | [Physics of Language Models: Part 1, Context-Free Grammar.](http://arxiv.org/abs/2305.13673) | 本研究探究了生成式语言模型如何学习上下文无关文法（CFG），并通过构造人造数据证明了预训练transformers可以学会生成具有接近完美准确度和显着多样性的句子。研究发现transformer内部的隐藏状态隐含而精确地编码了CFG结构，学会形成类似动态规划的“边界到边界”的注意力。此外，还研究了标准CFG的扩展，例如概率CFG和线性CFG，并证明transformers也可以学会这些扩展语法结构。 |
| [^12] | [Large Linguistic Models: Analyzing theoretical linguistic abilities of LLMs.](http://arxiv.org/abs/2305.00948) | 本研究展示了大型语言模型(LLMs)在语言任务上性能不断提高，且首次展示了它们能够生成连贯和有效的语言数据分析。分析和评估它们的元语言能力有助于我们理解它们的一般能力并对语言学理论模型提供新的认识。 |

# 详细

[^1]: BAT: 使用大规模语言模型学习关于空间声音的推理能力

    BAT: Learning to Reason about Spatial Sounds with Large Language Models

    [https://rss.arxiv.org/abs/2402.01591](https://rss.arxiv.org/abs/2402.01591)

    本文提出了BAT，它结合了双耳声音场景分析模型的空间声音感知能力和大规模语言模型的自然语言推理能力，以复制人类的空间声音推理能力。通过使用合成的双耳音频数据集和基于空间声音的问答数据集进行训练，BAT在空间声音感知和推理方面取得了强大的性能。

    

    空间声音推理是一种基本的人类技能，它使我们能够根据声音来导航和解释我们的周围环境。本文提出了BAT，它将双耳声音场景分析模型的空间声音感知能力与大规模语言模型（LLM）的自然语言推理能力相结合，以复制这种固有能力。为了解决现有野外空间声音数据集的缺乏，我们使用AudioSet和SoundSpaces 2.0合成了一个双耳音频数据集。接下来，我们开发了一种基于空间声音的问答数据集SpatialSoundQA，提供了一系列QA任务，以训练BAT在空间声音感知和推理的各个方面。BAT的声学前端编码器是一种名为Spatial Audio Spectrogram Transformer（Spatial-AST）的创新空间音频编码器，它本身在声音事件检测、空间定位和距离估计等方面具有强大的性能。通过将Spatial-AST与LLaMA-2 7B集成，

    Spatial sound reasoning is a fundamental human skill, enabling us to navigate and interpret our surroundings based on sound. In this paper we present BAT, which combines the spatial sound perception ability of a binaural acoustic scene analysis model with the natural language reasoning capabilities of a large language model (LLM) to replicate this innate ability. To address the lack of existing datasets of in-the-wild spatial sounds, we synthesized a binaural audio dataset using AudioSet and SoundSpaces 2.0. Next, we developed SpatialSoundQA, a spatial sound-based question-answering dataset, offering a range of QA tasks that train BAT in various aspects of spatial sound perception and reasoning. The acoustic front end encoder of BAT is a novel spatial audio encoder named Spatial Audio Spectrogram Transformer, or Spatial-AST, which by itself achieves strong performance across sound event detection, spatial localization, and distance estimation. By integrating Spatial-AST with LLaMA-2 7B
    
[^2]: 流式序列转导通过动态压缩

    Streaming Sequence Transduction through Dynamic Compression

    [https://rss.arxiv.org/abs/2402.01172](https://rss.arxiv.org/abs/2402.01172)

    STAR是一种新型的Transformer模型，通过动态压缩和优化延迟、内存占用和质量，实现对流的高效序列转导，并在自动语音识别领域表现出色。

    

    我们引入了STAR（带有锚定表示的流式转导），这是一种基于Transformer的新型模型，旨在实现对流的高效序列转导。STAR动态地对输入流进行分段，创建压缩的锚定表示，实现近乎无损的压缩（12倍）在自动语音识别（ASR）中，并优于现有方法。此外，STAR在同时进行语音到文本任务中展示出优越的分割和延迟-质量折衷，优化延迟、内存占用和质量。

    We introduce STAR (Stream Transduction with Anchor Representations), a novel Transformer-based model designed for efficient sequence-to-sequence transduction over streams. STAR dynamically segments input streams to create compressed anchor representations, achieving nearly lossless compression (12x) in Automatic Speech Recognition (ASR) and outperforming existing methods. Moreover, STAR demonstrates superior segmentation and latency-quality trade-offs in simultaneous speech-to-text tasks, optimizing latency, memory footprint, and quality.
    
[^3]: 使用扩散模型生成控制训练数据

    Controlled Training Data Generation with Diffusion Models

    [https://arxiv.org/abs/2403.15309](https://arxiv.org/abs/2403.15309)

    提出了一种使用扩散模型生成控制训练数据的方法，通过两个反馈机制，一方面使用监督模型反馈找到对抗性提示词实现图像生成，另一方面通过引导使生成过程朝向特定目标分布。

    

    在这项工作中，我们提出了一种方法，可以控制文本到图像生成模型以生成训练数据，专门用于监督学习。与之前那些采用开环方法并预先定义提示词来使用语言模型或人类专业知识生成新数据的作品不同，我们开发了一种自动闭环系统，其中包括两个反馈机制。第一个机制使用来自给定监督模型的反馈，并找到导致图像生成最大化模型损失的对抗提示词。虽然这些对抗提示词导致了经过模型训练的多样化数据生成，但它们并不知道目标分布，这可能效率低下。因此，我们引入第二个反馈机制，将生成过程引导到特定目标分布。我们称将这两个机制结合起来的方法为引导对抗提示词。我们在不同任务上进行评估。

    arXiv:2403.15309v1 Announce Type: cross  Abstract: In this work, we present a method to control a text-to-image generative model to produce training data specifically "useful" for supervised learning. Unlike previous works that employ an open-loop approach and pre-define prompts to generate new data using either a language model or human expertise, we develop an automated closed-loop system which involves two feedback mechanisms. The first mechanism uses feedback from a given supervised model and finds adversarial prompts that result in image generations that maximize the model loss. While these adversarial prompts result in diverse data informed by the model, they are not informed of the target distribution, which can be inefficient. Therefore, we introduce the second feedback mechanism that guides the generation process towards a certain target distribution. We call the method combining these two mechanisms Guided Adversarial Prompts. We perform our evaluations on different tasks, da
    
[^4]: 可用的XAI：在LLM时代利用可解释性的10个策略

    Usable XAI: 10 Strategies Towards Exploiting Explainability in the LLM Era

    [https://arxiv.org/abs/2403.08946](https://arxiv.org/abs/2403.08946)

    在大型语言模型时代，为了适应其复杂性和先进能力，我们引入了可用的XAI概念，通过积极增强LLMs在实际环境中的生产力和适用性，实现XAI方法论的重大转变。

    

    可解释人工智能（XAI）指的是提供人类可理解的洞见，揭示人工智能模型的运作方式的技术。最近，XAI的重点正被扩展到常常因为不透明而备受批评的大型语言模型（LLMs）。这一拓展需要对XAI方法论进行显著转变，因为有两个原因。首先，许多现有的XAI方法无法直接应用于LLMs，因为它们的复杂性和先进能力。其次，随着LLMs越来越广泛地应用于不同行业应用中，XAI的角色从仅仅打开“黑匣子”转变为积极增强LLMs在实际环境中的生产力和适用性。与此同时，不同于传统机器学习模型仅作为XAI洞见的被动接受者，LLMs的独特能力能够相互增强XAI。因此，在本文中，我们通过分析（1）...

    arXiv:2403.08946v1 Announce Type: cross  Abstract: Explainable AI (XAI) refers to techniques that provide human-understandable insights into the workings of AI models. Recently, the focus of XAI is being extended towards Large Language Models (LLMs) which are often criticized for their lack of transparency. This extension calls for a significant transformation in XAI methodologies because of two reasons. First, many existing XAI methods cannot be directly applied to LLMs due to their complexity advanced capabilities. Second, as LLMs are increasingly deployed across diverse industry applications, the role of XAI shifts from merely opening the "black box" to actively enhancing the productivity and applicability of LLMs in real-world settings. Meanwhile, unlike traditional machine learning models that are passive recipients of XAI insights, the distinct abilities of LLMs can reciprocally enhance XAI. Therefore, in this paper, we introduce Usable XAI in the context of LLMs by analyzing (1)
    
[^5]: 微调、提示、上下文学习和指导微调：我们需要多少标记样本？

    Fine-Tuning, Prompting, In-Context Learning and Instruction-Tuning: How Many Labelled Samples Do We Need?

    [https://arxiv.org/abs/2402.12819](https://arxiv.org/abs/2402.12819)

    专门模型通常只需少量标记样本（100-1000个）就能与通用模型持平甚至更好，取决于任务的复杂性和结果的变化。

    

    当解决具有有限标记数据的任务时，研究人员可以选择使用通用的大型语言模型而不进行进一步更新，或者使用少量示例来调整专门的较小模型。 当有足够的标记可用时，专门的模型在许多自然语言处理任务上表现优于通用模型。 在这项工作中，我们旨在调查专门模型需要多少标记样本才能实现这种出色的性能，同时考虑结果的变化。观察提示、上下文学习、微调和指导微调的行为，识别它们在增加不同复杂性任务的标记训练样本数量时的收支平衡点，我们发现专门模型通常只需少量样本（100-1000个）就能与通用模型持平甚至更好。 同时，所需的标记数据量强烈依赖于任务的复杂性和结果的变化。

    arXiv:2402.12819v1 Announce Type: cross  Abstract: When solving a task with limited labelled data, researchers can either use a general large language model without further update, or use the few examples to tune a specialised smaller model. When enough labels are available, the specialised models outperform the general ones on many NLP tasks. In this work, we aim to investigate how many labelled samples are required for the specialised models to achieve this superior performance, while taking the results variance into consideration. Observing the behaviour of prompting, in-context learning, fine-tuning and instruction-tuning, identifying their break-even points when increasing number of labelled training samples across three tasks of varying complexity, we find that the specialised models often need only few samples ($100-1000$) to be on par or better than the general ones. At the same time, the amount of required labelled data strongly depends on the task complexity and results varia
    
[^6]: FormulaQA：一个基于公式的数值推理问题问答数据集

    FormulaQA: A Question Answering Dataset for Formula-Based Numerical Reasoning

    [https://arxiv.org/abs/2402.12692](https://arxiv.org/abs/2402.12692)

    FormulaQA是一个基于初中物理考试的公式驱动数值推理问题问答数据集，通过评估LLMs的不同方法和使用检索增强型LLMs以及对小型模型进行微调，揭示了现有模型在应对复杂、基于公式的FormulaQA时的潜在改进空间。

    

    应用公式是人类在解决数值推理问题时的基本能力。然而，现有的数值推理数据集很少明确指出推理步骤中使用的公式。为了弥补这一差距，我们提出了一个基于初中物理考试的公式驱动数值推理问题问答数据集FormulaQA。我们还使用大小从7B到超过100B参数的LLMs进行了零样本和少样本思维链方法的评估，并探索了在提供外部公式数据库时使用检索增强型LLMs的方法。我们还对大小不超过2B的较小模型进行了微调。我们的实证研究强调了当应用于我们复杂、基于公式的FormulaQA时，现有模型在改进方面具有显著潜力。

    arXiv:2402.12692v1 Announce Type: new  Abstract: The application of formulas is a fundamental ability of humans when addressing numerical reasoning problems. However, existing numerical reasoning datasets seldom explicitly indicate the formulas employed during the reasoning steps. To bridge this gap, we propose a question answering dataset for formula-based numerical reasoning called FormulaQA, from junior high school physics examinations. We further conduct evaluations on LLMs with size ranging from 7B to over 100B parameters utilizing zero-shot and few-shot chain-of-thoughts methods and we explored the approach of using retrieval-augmented LLMs when providing an external formula database. We also fine-tune on smaller models with size not exceeding 2B. Our empirical findings underscore the significant potential for improvement in existing models when applied to our complex, formula-driven FormulaQA.
    
[^7]: 我们能否逐步验证错误答案检测？

    Can We Verify Step by Step for Incorrect Answer Detection?

    [https://arxiv.org/abs/2402.10528](https://arxiv.org/abs/2402.10528)

    通过推理链来预测大型语言模型输出的准确性，我们引入了一个新的基准R2PE，并提出了处理可辨识性评分（PDS）框架。

    

    Chain-of-Thought（CoT）提示在增强大型语言模型（LLMs）的推理能力方面取得了重大进展。先前的研究开发了各种扩展的CoT，主要集中在增强最终任务的性能上。此外，已经有研究评估了CoT中推理链的质量。这引发了一个有趣的问题：通过仔细审查它们生成的推理链，是否可以预测LLMs输出的准确性？为了回答这个研究问题，我们引入了一个基准，R2PE，专门设计用于探究不同领域涵盖五个不同推理任务中推理链与性能之间的关系。该基准旨在基于推理步骤衡量LLMs最终输出的虚假性。为了充分利用多个推理链中的信息，我们提出了打败常识分数（PDS）框架。

    arXiv:2402.10528v1 Announce Type: cross  Abstract: Chain-of-Thought (CoT) prompting has marked a significant advancement in enhancing the reasoning capabilities of large language models (LLMs). Previous studies have developed various extensions of CoT, which focus primarily on enhancing end-task performance. In addition, there has been research on assessing the quality of reasoning chains in CoT. This raises an intriguing question: Is it possible to predict the accuracy of LLM outputs by scrutinizing the reasoning chains they generate? To answer this research question, we introduce a benchmark, R2PE, designed specifically to explore the relationship between reasoning chains and performance in various reasoning tasks spanning five different domains. This benchmark aims to measure the falsehood of the final output of LLMs based on the reasoning steps. To make full use of information in multiple reasoning chains, we propose the process discernibility score (PDS) framework that beats the a
    
[^8]: 一种自动生成谜题以辅助概念理解的方法

    An Approach to Automatically generating Riddles aiding Concept Attainment. (arXiv:2310.18290v1 [cs.CL])

    [http://arxiv.org/abs/2310.18290](http://arxiv.org/abs/2310.18290)

    这个论文介绍了一种自动生成概念谜题的方法，以促进在线学习环境中的学习者参与度。通过应用概念达成模型和生成谜题，该方法可以帮助学习者更好地理解概念。

    

    在在线学习环境中，保持学习者的积极参与是一个主要的挑战。为增强学习者的参与度，提出了各种不同的教学策略，无论是在线还是离线环境中。概念达成模型就是一种教学策略，它着重于学习者对概念的深入理解，而不仅仅是对概念的字典定义。通过搜索和列举用于区分各种概念的实例和非实例之间的属性，来达到这一目的。我们的工作试图将概念达成模型应用于构建概念谜题，以在在线学习环境中使用。该方法涉及从学习资源中创建事实三元组，根据其对概念的唯一性进行分类为“主题标记”和“共同”，然后根据概念达成模型的格式生成谜题，并捕获这些谜题的所有可能解。从人类评估中获得的结果显示...

    One of the primary challenges in online learning environments, is to retain learner engagement. Several different instructional strategies are proposed both in online and offline environments to enhance learner engagement. The Concept Attainment Model is one such instructional strategy that focuses on learners acquiring a deeper understanding of a concept rather than just its dictionary definition. This is done by searching and listing the properties used to distinguish examples from non-examples of various concepts. Our work attempts to apply the Concept Attainment Model to build conceptual riddles, to deploy over online learning environments. The approach involves creating factual triples from learning resources, classifying them based on their uniqueness to a concept into `Topic Markers' and `Common', followed by generating riddles based on the Concept Attainment Model's format and capturing all possible solutions to those riddles. The results obtained from the human evaluation of r
    
[^9]: 跨语言多语言模型中事实知识的跨语言一致性

    Cross-Lingual Consistency of Factual Knowledge in Multilingual Language Models. (arXiv:2310.10378v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2310.10378](http://arxiv.org/abs/2310.10378)

    本论文研究了多语言预训练语言模型中事实知识的跨语言一致性，提出了一种新的度量方法，并通过分析模型大小、语言配对等因素发现了影响一致性的因素。实验结果表明，增加模型大小可以提高准确性，但不会改善跨语言一致性。

    

    多语言大规模预训练语言模型（PLM）显示存储了大量的事实知识，但在不同语言之间存在较大的变化。为了确保不同语言背景的用户从同一个模型中获得一致的反馈，我们研究了各种多语言PLM中事实知识的跨语言一致性（CLC）。为此，我们提出了一种基于排序的一致性（RankC）度量，用于独立于准确性评估跨语言间的知识一致性。利用这个度量方法，我们对决定CLC的因素进行了深入分析，包括模型层面和语言对层面。在其他结果中，我们发现增加模型大小可以提高大多数语言中的事实探测准确性，但不能改善跨语言一致性。最后，我们通过模型编辑在PLMs中插入新的事实关联进行了一个CLC的案例研究。对一小部分事实进行了实验。

    Multilingual large-scale Pretrained Language Models (PLMs) have been shown to store considerable amounts of factual knowledge, but large variations are observed across languages. With the ultimate goal of ensuring that users with different language backgrounds obtain consistent feedback from the same model, we study the cross-lingual consistency (CLC) of factual knowledge in various multilingual PLMs. To this end, we propose a Ranking-based Consistency (RankC) metric to evaluate knowledge consistency across languages independently from accuracy. Using this metric, we conduct an in-depth analysis of the determining factors for CLC, both at model level and at language-pair level. Among other results, we find that increasing model size leads to higher factual probing accuracy in most languages, but does not improve cross-lingual consistency. Finally, we conduct a case study on CLC when new factual associations are inserted in the PLMs via model editing. Results on a small sample of facts 
    
[^10]: PlanFitting：利用大型语言模型定制个性化的运动计划

    PlanFitting: Tailoring Personalized Exercise Plans with Large Language Models. (arXiv:2309.12555v1 [cs.HC])

    [http://arxiv.org/abs/2309.12555](http://arxiv.org/abs/2309.12555)

    PlanFitting是一个对话型人工智能，利用大型语言模型的生成能力帮助用户定制个性化的运动计划，并在用户研究中证明了它生成个性化、可操作和有据可依的运动计划的潜力。

    

    个性化的运动计划对于确保足够的体育活动至关重要，但由于人们的复杂日程和考虑因素以及计划的创建通常需要与专家的反复沟通，这一过程变得具有挑战性。我们提出了PlanFitting，它是一个对话型人工智能，可以辅助个性化的运动计划。通过利用大型语言模型的生成能力，PlanFitting使用户能够用自然语言描述各种约束和查询，从而便于创建和优化适合其特定情况的每周运动计划，并保持基本原则的扎根。通过一项用户研究，参与者（N=18）使用PlanFitting生成个性化的运动计划，而专家规划者（N=3）对这些计划进行评估，我们确定了PlanFitting在生成个性化、可操作和有据可依的运动计划方面的潜力。我们还讨论了AI助手在创建计划方面的未来设计机遇。

    A personally tailored exercise regimen is crucial to ensuring sufficient physical activities, yet challenging to create as people have complex schedules and considerations and the creation of plans often requires iterations with experts. We present PlanFitting, a conversational AI that assists in personalized exercise planning. Leveraging generative capabilities of large language models, PlanFitting enables users to describe various constraints and queries in natural language, thereby facilitating the creation and refinement of their weekly exercise plan to suit their specific circumstances while staying grounded in foundational principles. Through a user study where participants (N=18) generated a personalized exercise plan using PlanFitting and expert planners (N=3) evaluated these plans, we identified the potential of PlanFitting in generating personalized, actionable, and evidence-based exercise plans. We discuss future design opportunities for AI assistants in creating plans that 
    
[^11]: 语言模型的物理学：第一部分，上下文无关文法。

    Physics of Language Models: Part 1, Context-Free Grammar. (arXiv:2305.13673v1 [cs.CL])

    [http://arxiv.org/abs/2305.13673](http://arxiv.org/abs/2305.13673)

    本研究探究了生成式语言模型如何学习上下文无关文法（CFG），并通过构造人造数据证明了预训练transformers可以学会生成具有接近完美准确度和显着多样性的句子。研究发现transformer内部的隐藏状态隐含而精确地编码了CFG结构，学会形成类似动态规划的“边界到边界”的注意力。此外，还研究了标准CFG的扩展，例如概率CFG和线性CFG，并证明transformers也可以学会这些扩展语法结构。

    

    我们设计了实验来研究生成式语言模型（例如GPT）如何学习上下文无关文法（CFG）-具有树状结构的多样化语言系统，可捕捉许多自然语言，程序和人类逻辑的方面。CFG与下推自动机一样困难，可能是模棱两可的，因此验证字符串是否满足规则需要动态规划。我们构造了人造数据，并证明即使对于非常具有挑战性的CFG，预训练transformers也可以学会生成具有接近完美准确度和显着多样性的句子。更重要的是，我们深入探讨了transformers学习CFG背后的物理原理。我们发现transformer内部的隐藏状态隐含而精确地编码了CFG结构（如在子树边界上精确定位树节点信息），并学会形成类似动态规划的“边界到边界”的注意力。我们还涵盖了一些标准CFG的扩展，例如概率CFG和线性CFG，并展示transformers也可以学会这些扩展语法结构。我们的工作揭示了语言模型的内部工作原理，并为未来的模型设计和分析提供了启示。

    We design experiments to study $\textit{how}$ generative language models, like GPT, learn context-free grammars (CFGs) -- diverse language systems with a tree-like structure capturing many aspects of natural languages, programs, and human logics. CFGs are as hard as pushdown automata, and can be ambiguous so that verifying if a string satisfies the rules requires dynamic programming. We construct synthetic data and demonstrate that even for very challenging CFGs, pre-trained transformers can learn to generate sentences with near-perfect accuracy and remarkable $\textit{diversity}$.  More importantly, we delve into the $\textit{physical principles}$ behind how transformers learns CFGs. We discover that the hidden states within the transformer implicitly and $\textit{precisely}$ encode the CFG structure (such as putting tree node information exactly on the subtree boundary), and learn to form "boundary to boundary" attentions that resemble dynamic programming. We also cover some extensio
    
[^12]: 大型语言模型：分析LLM的理论语言能力

    Large Linguistic Models: Analyzing theoretical linguistic abilities of LLMs. (arXiv:2305.00948v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2305.00948](http://arxiv.org/abs/2305.00948)

    本研究展示了大型语言模型(LLMs)在语言任务上性能不断提高，且首次展示了它们能够生成连贯和有效的语言数据分析。分析和评估它们的元语言能力有助于我们理解它们的一般能力并对语言学理论模型提供新的认识。

    

    大型语言模型(LLMs)的性能最近已经提高到了能够在许多语言任务上表现良好的程度。我们在这里展示了，这些模型也可以生成连贯和有效的语言数据的形式分析，展示了大型语言模型对其元语言能力分析的巨大潜力。LLMs主要是通过文本形式的语言数据进行训练；分析和评估它们的元语言能力改进了我们对它们的一般能力的理解，并对语言学中的理论模型提供了新的认识。在本文中，我们通过专注于形式语言学的三个子领域：句法、音韵学和语义学，探究了GPT-4的元语言能力。我们提出了一个关于大型语言模型元语言分析的研究计划，提出了实验设计，提供了一般指导方针，讨论了限制，并为这个研究方向提供了未来的方向。这个研究还有助于揭示大型语言模型的潜在能力和理论模型的新视角。

    The performance of large language models (LLMs) has recently improved to the point where the models can perform well on many language tasks. We show here that for the first time, the models can also generate coherent and valid formal analyses of linguistic data and illustrate the vast potential of large language models for analyses of their metalinguistic abilities. LLMs are primarily trained on language data in the form of text; analyzing and evaluating their metalinguistic abilities improves our understanding of their general capabilities and sheds new light on theoretical models in linguistics. In this paper, we probe into GPT-4's metalinguistic capabilities by focusing on three subfields of formal linguistics: syntax, phonology, and semantics. We outline a research program for metalinguistic analyses of large language models, propose experimental designs, provide general guidelines, discuss limitations, and offer future directions for this line of research. This line of inquiry als
    

