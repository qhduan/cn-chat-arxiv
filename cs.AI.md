# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [How Far Are We on the Decision-Making of LLMs? Evaluating LLMs' Gaming Ability in Multi-Agent Environments](https://arxiv.org/abs/2403.11807) | 通过博弈论视角评估LLMs的决策能力，结果表明GPT-3.5在稳健性方面表现良好，但泛化能力有限，而GPT-4则优于其他模型。 |
| [^2] | [CleanAgent: Automating Data Standardization with LLM-based Agents](https://arxiv.org/abs/2403.08291) | 提出了一个具有声明性、统一API的Python库，通过简洁的API调用简化LLM的代码生成流程 |
| [^3] | [Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context](https://arxiv.org/abs/2403.05530) | Gemini 1.5 Pro是一种高效计算的多模态混合模型，能在数百万标记的上下文中回忆和推理信息，达到近乎完美的长上下文检索任务召回率，改进了长文档问答、长视频问答和长上下文ASR的最新技术水平。 |
| [^4] | [The WMDP Benchmark: Measuring and Reducing Malicious Use With Unlearning](https://arxiv.org/abs/2403.03218) | WMDP基准是一个公开发布的数据集，包含4157个多项选择问题，用作生物安全、网络安全和化学安全危险知识的代理测量。 |
| [^5] | [Even-Ifs From If-Onlys: Are the Best Semi-Factual Explanations Found Using Counterfactuals As Guides?](https://arxiv.org/abs/2403.00980) | 这里是中文总结出的一句话要点：研究对8种半事实方法进行了全面测试，发现反事实指导并非必要，而是... (由于篇幅限制，若有省略，请见谅) |
| [^6] | [ChemLLM: A Chemical Large Language Model](https://arxiv.org/abs/2402.06852) | ChemLLM是第一个专门用于化学领域的大型语言模型，利用新颖的指令构建方法将结构化知识转化为对话形式，具有平滑对话交互的能力，并在化学的三个主要任务中击败了GPT-3.5。 |
| [^7] | [VerAs: Verify then Assess STEM Lab Reports](https://arxiv.org/abs/2402.05224) | VerAs是一个端到端的神经架构，用于验证和评估STEM实验报告。它通过利用多个维度的分析评估标准，以及针对学生提供详细反馈，帮助他们提高科学写作技巧。 |
| [^8] | [Genixer: Empowering Multimodal Large Language Models as a Powerful Data Generator](https://arxiv.org/abs/2312.06731) | Genixer是一种为生成高质量指导数据而设计的创新数据生成管道，包括九个代表性任务，提供了四个关键步骤来改善数据生成的难度。 |
| [^9] | [The Anatomy Spread of Online Opinion Polarization: The Pivotal Role of Super-Spreaders in Social Networks.](http://arxiv.org/abs/2401.01349) | 该研究揭示了在社交网络中塑造意见的超级传播者的重要角色，通过对超级传播者行为的调查和分析，提供了对群体动态和意见形成的条件的理解，对改进在线交流安全和社交影响力的认识具有启示作用。 |
| [^10] | [Large Language Models can Learn Rules.](http://arxiv.org/abs/2310.07064) | 大型语言模型(LLMs)在各种推理任务中展示了令人印象深刻的性能。为了提高提示方法的准确性和一致性，我们提出了Hypotheses-to-Theories (HtT)框架，用于学习LLMs推理的规则库，从而改进了现有的提示方法。 |
| [^11] | [Lemur: Integrating Large Language Models in Automated Program Verification.](http://arxiv.org/abs/2310.04870) | 本论文提出了一种将LLMs和自动推理器结合起来进行自动程序验证的通用方法，并证明了其完备性。这个方法在一些合成和竞争基准上取得了实际的改进。 |
| [^12] | [Game-Theoretic Robust Reinforcement Learning Handles Temporally-Coupled Perturbations.](http://arxiv.org/abs/2307.12062) | 这篇论文提出了一种新的鲁棒强化学习方法，通过将时间耦合的鲁棒强化学习问题视为两人零和游戏来处理问题，并通过找到近似均衡来确保代理对时间耦合干扰的鲁棒性。实验结果显示，该方法在各种连续控制任务中相比基准方法表现出显著的鲁棒性优势。 |
| [^13] | [Structure in Reinforcement Learning: A Survey and Open Problems.](http://arxiv.org/abs/2306.16021) | 这项调查研究了强化学习中结构的角色和重要性，并介绍了各个子领域在提高强化学习的性能方面所做的工作。 |
| [^14] | [Digital Over-the-Air Federated Learning in Multi-Antenna Systems.](http://arxiv.org/abs/2302.14648) | 本文研究了在多天线系统中采用数字调制和空中计算的情况下，联邦学习的性能优化问题。通过结合数字调制和AirComp，提出了一种改进的联邦平均算法，以解决无线信道衰落导致的总体失真问题。 |

# 详细

[^1]: LLM的决策水平在多智能体环境中的评估究竟如何？

    How Far Are We on the Decision-Making of LLMs? Evaluating LLMs' Gaming Ability in Multi-Agent Environments

    [https://arxiv.org/abs/2403.11807](https://arxiv.org/abs/2403.11807)

    通过博弈论视角评估LLMs的决策能力，结果表明GPT-3.5在稳健性方面表现良好，但泛化能力有限，而GPT-4则优于其他模型。

    

    决策是一个复杂的任务，需要各种能力，为评估大型语言模型（LLMs）提供了一个极好的框架。我们的研究通过博弈论的视角探究LLMs的决策能力。我们专注于支持多个智能体同时参与的游戏，引入了我们的框架GAMA-Bench，包括八个经典的多智能体游戏。我们设计了一个评分方案，定量评估模型在这些游戏中的表现。通过GAMA-Bench，我们研究了LLMs的稳健性、泛化能力和增强策略。结果显示，虽然GPT-3.5表现出令人满意的稳健性，但其泛化能力相对有限。然而，通过一些方法如“思维链”，其性能可以得到提高。此外，我们对各种LLMs进行评估，发现GPT-4胜过其他模型。

    arXiv:2403.11807v1 Announce Type: new  Abstract: Decision-making, a complicated task requiring various types of abilities, presents an excellent framework for assessing Large Language Models (LLMs). Our research investigates LLMs' decision-making capabilities through the lens of a well-established field, Game Theory. We focus specifically on games that support the participation of more than two agents simultaneously. Subsequently, we introduce our framework, GAMA-Bench, including eight classical multi-agent games. We design a scoring scheme to assess a model's performance in these games quantitatively. Through GAMA-Bench, we investigate LLMs' robustness, generalizability, and enhancement strategies. Results reveal that while GPT-3.5 shows satisfying robustness, its generalizability is relatively limited. However, its performance can be improved through approaches such as Chain-of-Thought. Additionally, we conduct evaluations across various LLMs and find that GPT-4 outperforms other mod
    
[^2]: CleanAgent：基于LLM代理自动化数据标准化

    CleanAgent: Automating Data Standardization with LLM-based Agents

    [https://arxiv.org/abs/2403.08291](https://arxiv.org/abs/2403.08291)

    提出了一个具有声明性、统一API的Python库，通过简洁的API调用简化LLM的代码生成流程

    

    数据标准化是数据科学生命周期中至关重要的一部分。虽然诸如Pandas之类的工具提供了强大的功能，但它们的复杂性以及需要定制代码以适应不同列类型的手动操作带来了重大挑战。尽管大型语言模型（LLMs）如ChatGPT已经展现出通过自然语言理解和代码生成自动化此过程的潜力，但仍需要专业程度的编程知识和持续互动以进行及时的完善。为了解决这些挑战，我们的关键想法是提出一个具有声明性、统一API的Python库，用于标准化列类型，通过简洁的API调用简化LLM的代码生成流程。我们首先提出了Dataprep.Clean，作为Dataprep库的一个组件，通过一行代码实现特定列类型的标准化，极大降低了复杂性。然后我们介绍了CleanAgen

    arXiv:2403.08291v1 Announce Type: cross  Abstract: Data standardization is a crucial part in data science life cycle. While tools like Pandas offer robust functionalities, their complexity and the manual effort required for customizing code to diverse column types pose significant challenges. Although large language models (LLMs) like ChatGPT have shown promise in automating this process through natural language understanding and code generation, it still demands expert-level programming knowledge and continuous interaction for prompt refinement. To solve these challenges, our key idea is to propose a Python library with declarative, unified APIs for standardizing column types, simplifying the code generation of LLM with concise API calls. We first propose Dataprep.Clean which is written as a component of the Dataprep Library, offers a significant reduction in complexity by enabling the standardization of specific column types with a single line of code. Then we introduce the CleanAgen
    
[^3]: Gemini 1.5：解锁跨数百万标记上下文的多模态理解

    Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context

    [https://arxiv.org/abs/2403.05530](https://arxiv.org/abs/2403.05530)

    Gemini 1.5 Pro是一种高效计算的多模态混合模型，能在数百万标记的上下文中回忆和推理信息，达到近乎完美的长上下文检索任务召回率，改进了长文档问答、长视频问答和长上下文ASR的最新技术水平。

    

    在这份报告中，我们介绍了Gemini家族的最新模型Gemini 1.5 Pro，这是一个高效计算的多模态专家混合模型，能够回忆和推理数百万标记上下文中的细粒度信息，包括多个长文档和几小时的视频和音频。Gemini 1.5 Pro在各种形式的长上下文检索任务中实现了近乎完美的召回率，改进了长文档问答、长视频问答和长上下文ASR的最新技术水平，并在广泛一系列基准测试中与Gemini 1.0 Ultra的最新技术水平相匹敌甚至超过。在研究Gemini 1.5 Pro长上下文能力的极限时，我们发现在至少10M标记的范围内继续改进下一个标记的预测，并且几乎完美地达到了超过99%的检索率，这是对现有模型如Claude 2.1（200k）和GPT-4 Turbo（128k）的世代性飞跃。最后，我们突出了大型语言模型在新领域的令人惊讶的新能力。

    arXiv:2403.05530v1 Announce Type: cross  Abstract: In this report, we present the latest model of the Gemini family, Gemini 1.5 Pro, a highly compute-efficient multimodal mixture-of-experts model capable of recalling and reasoning over fine-grained information from millions of tokens of context, including multiple long documents and hours of video and audio. Gemini 1.5 Pro achieves near-perfect recall on long-context retrieval tasks across modalities, improves the state-of-the-art in long-document QA, long-video QA and long-context ASR, and matches or surpasses Gemini 1.0 Ultra's state-of-the-art performance across a broad set of benchmarks. Studying the limits of Gemini 1.5 Pro's long-context ability, we find continued improvement in next-token prediction and near-perfect retrieval (>99%) up to at least 10M tokens, a generational leap over existing models such as Claude 2.1 (200k) and GPT-4 Turbo (128k). Finally, we highlight surprising new capabilities of large language models at the
    
[^4]: WMDP基准：通过遗忘测量和减少恶意使用

    The WMDP Benchmark: Measuring and Reducing Malicious Use With Unlearning

    [https://arxiv.org/abs/2403.03218](https://arxiv.org/abs/2403.03218)

    WMDP基准是一个公开发布的数据集，包含4157个多项选择问题，用作生物安全、网络安全和化学安全危险知识的代理测量。

    

    arXiv:2403.03218v1 公告类型：交叉领域 摘要：白宫关于人工智能的行政命令强调了大型语言模型(LLMs)赋予恶意行为者开发生物、网络和化学武器的风险。为了衡量这些恶意使用的风险，政府机构和主要人工智能实验室正在开发LLMs的危险能力评估。然而，当前的评估是私人的，阻碍了进一步研究如何减少风险。此外，它们仅专注于几条高度特定的恶意使用途径。为了填补这些空白，我们公开发布了大规模杀伤性武器代理（WMDP）基准，这是一个包含4157个多项选择问题的数据集，作为生物安全、网络安全和化学安全危险知识的代理测量。WMDP由一组学术界和技术顾问联合开发，并在公开发布前严格过滤以消除敏感信息。WMDP有两个服务

    arXiv:2403.03218v1 Announce Type: cross  Abstract: The White House Executive Order on Artificial Intelligence highlights the risks of large language models (LLMs) empowering malicious actors in developing biological, cyber, and chemical weapons. To measure these risks of malicious use, government institutions and major AI labs are developing evaluations for hazardous capabilities in LLMs. However, current evaluations are private, preventing further research into mitigating risk. Furthermore, they focus on only a few, highly specific pathways for malicious use. To fill these gaps, we publicly release the Weapons of Mass Destruction Proxy (WMDP) benchmark, a dataset of 4,157 multiple-choice questions that serve as a proxy measurement of hazardous knowledge in biosecurity, cybersecurity, and chemical security. WMDP was developed by a consortium of academics and technical consultants, and was stringently filtered to eliminate sensitive information prior to public release. WMDP serves two r
    
[^5]: 从“只要”到“即使”：是否通过反事实指导最佳半事实解释？

    Even-Ifs From If-Onlys: Are the Best Semi-Factual Explanations Found Using Counterfactuals As Guides?

    [https://arxiv.org/abs/2403.00980](https://arxiv.org/abs/2403.00980)

    这里是中文总结出的一句话要点：研究对8种半事实方法进行了全面测试，发现反事实指导并非必要，而是... (由于篇幅限制，若有省略，请见谅)

    

    最近，“只要”解释中的反事实在可解释人工智能（eXplainable AI，XAI）领域变得非常流行，因为它们描述了对黑盒AI系统的特征输入进行哪些更改会导致（通常是负面的）决策结果的变化。更近期，使用“即使”解释的半事实方法引起了更多关注。它们阐明了对AI系统的特征输入进行的更改不会改变决策结果，从而可能提出更有利的行动建议。一些半事实方法使用反事实来引导查询实例以指导半事实生成（称为反事实引导方法），而其他方法则不这样做（称为无反事实方法）。在这项工作中，我们对7个数据集上的8种半事实方法进行了全面测试，使用了5个关键指标，以确定反事实指导是否有必要找到最佳的半事实。这些测试的结果表明并不是，而是...

    arXiv:2403.00980v1 Announce Type: new  Abstract: Recently, counterfactuals using "if-only" explanations have become very popular in eXplainable AI (XAI), as they describe which changes to feature-inputs of a black-box AI system result in changes to a (usually negative) decision-outcome. Even more recently, semi-factuals using "even-if" explanations have gained more attention. They elucidate the feature-input changes that do \textit{not} change the decision-outcome of the AI system, with a potential to suggest more beneficial recourses. Some semi-factual methods use counterfactuals to the query-instance to guide semi-factual production (so-called counterfactual-guided methods), whereas others do not (so-called counterfactual-free methods). In this work, we perform comprehensive tests of 8 semi-factual methods on 7 datasets using 5 key metrics, to determine whether counterfactual guidance is necessary to find the best semi-factuals. The results of these tests suggests not, but rather tha
    
[^6]: ChemLLM: 一个化学大型语言模型

    ChemLLM: A Chemical Large Language Model

    [https://arxiv.org/abs/2402.06852](https://arxiv.org/abs/2402.06852)

    ChemLLM是第一个专门用于化学领域的大型语言模型，利用新颖的指令构建方法将结构化知识转化为对话形式，具有平滑对话交互的能力，并在化学的三个主要任务中击败了GPT-3.5。

    

    大型语言模型（LLM）在化学应用中取得了令人瞩目的进展，包括分子属性预测、分子生成、实验协议设计等。然而，该领域缺乏一个专门针对化学领域设计的基于对话的模型。这个挑战来自于事实，大多数化学数据和科学知识主要存储在结构化数据库中，直接使用这些结构化数据会影响模型维持连贯对话的能力。为了解决这个问题，我们开发了一种新颖的基于模板的指令构建方法，将结构化知识转化为简洁对话形式，适合于语言模型的训练。通过利用这种方法，我们开发了ChemLLM，第一个专门用于化学的大型语言模型，能够在化学领域的各种任务中进行平滑对话交互。ChemLLM在化学的三个主要任务，即名称转换、分子生成和实验协议设计方面，击败了GPT-3.5。

    Large language models (LLMs) have made impressive progress in chemistry applications, including molecular property prediction, molecular generation, experimental protocol design, etc. However, the community lacks a dialogue-based model specifically designed for chemistry. The challenge arises from the fact that most chemical data and scientific knowledge are primarily stored in structured databases, and the direct use of these structured data compromises the model's ability to maintain coherent dialogue. To tackle this issue, we develop a novel template-based instruction construction method that transforms structured knowledge into plain dialogue, making it suitable for language model training. By leveraging this approach, we develop ChemLLM, the first large language model dedicated to chemistry, capable of performing various tasks across chemical disciplines with smooth dialogue interaction. ChemLLM beats GPT-3.5 on all three principal tasks in chemistry, i.e., name conversion, molecu
    
[^7]: VerAs: 验证然后评估STEM实验报告

    VerAs: Verify then Assess STEM Lab Reports

    [https://arxiv.org/abs/2402.05224](https://arxiv.org/abs/2402.05224)

    VerAs是一个端到端的神经架构，用于验证和评估STEM实验报告。它通过利用多个维度的分析评估标准，以及针对学生提供详细反馈，帮助他们提高科学写作技巧。

    

    随着STEM教育对批判性思维能力的日益关注，科学写作在注重探究技能的课程中发挥着越来越重要的作用。最近发布的一份数据集是基于一套探究型物理课程的两组大学水平的实验报告，依赖于利用多个维度的分析评估标准，指定学科知识和优秀解释的一般组成部分。每个分析维度都以6分制进行评估，以提供详细反馈，帮助学生提高科学写作技巧。手动评估可能较慢，并且在大班中对所有学生进行一致性校准可能很困难。尽管在STEM学科的开放性问题的自动评估上已经有很多工作，但在实验报告等长篇写作中的工作要少得多。我们提出了一个端到端的神经架构，其中包括独立的验证器和评估模块，灵感来源于开放领域问题回答的方法。

    With an increasing focus in STEM education on critical thinking skills, science writing plays an ever more important role in curricula that stress inquiry skills. A recently published dataset of two sets of college level lab reports from an inquiry-based physics curriculum relies on analytic assessment rubrics that utilize multiple dimensions, specifying subject matter knowledge and general components of good explanations. Each analytic dimension is assessed on a 6-point scale, to provide detailed feedback to students that can help them improve their science writing skills. Manual assessment can be slow, and difficult to calibrate for consistency across all students in large classes. While much work exists on automated assessment of open-ended questions in STEM subjects, there has been far less work on long-form writing such as lab reports. We present an end-to-end neural architecture that has separate verifier and assessment modules, inspired by approaches to Open Domain Question Answ
    
[^8]: Genixer：将多模态大语言模型赋能为强大的数据生成器

    Genixer: Empowering Multimodal Large Language Models as a Powerful Data Generator

    [https://arxiv.org/abs/2312.06731](https://arxiv.org/abs/2312.06731)

    Genixer是一种为生成高质量指导数据而设计的创新数据生成管道，包括九个代表性任务，提供了四个关键步骤来改善数据生成的难度。

    

    arXiv:2312.06731v2 公告类型: 替换-交叉  摘要: 调优数据对于训练多模态大语言模型(MLLMs)至关重要。然而，高质量调优数据的创建面临重大挑战。先前依赖于GPT-4进行数据生成的方法不仅成本高昂，而且在复杂任务（即基于理解的推理任务）中的性能也不尽如人意。为解决这些问题，我们开发了一种创新的数据生成流水线Genixer，用于生成各种高质量的调优数据，包括九个代表性任务，例如常见的VQA，REC，REG和PointQ。具体而言，Genixer提供了一个统一的解决方案，包括四个关键步骤来缓解数据生成的困难：(i)指导数据收集，(ii)指导模板设计，(iii)赋能MLLM，和(iv)数据生成和过滤。随后，我们的Genixer的卓越的定性结果表明，当前的MLLM具有...

    arXiv:2312.06731v2 Announce Type: replace-cross  Abstract: Instruction tuning data is essential for training the Multimodal Large Language Models (MLLMs). However, the creation of high-quality instruction tuning data presents significant challenges. Prior methods that depended on GPT-4 for data generation were not only costly but also lacked satisfactory performance in complex tasks (i.e., grounding-based reasoning tasks). To address these issues, we developed an innovative data generation pipeline, Genixer, to generate various high-quality instruction tuning data, including nine representative tasks, e.g., Common VQA, REC, REG, and PointQ. Specifically, Genixer provides a unified solution with four key steps for alleviating the difficulty of data generation: (i) instruction data collection, (ii) instruction template design, (iii) empowering MLLM, and (iv) data generation and filtering. Subsequently, the superior qualitative results of our Genixer demonstrate that current MLLMs have a 
    
[^9]: 在线意见极化的传播解剖：超级传播者在社交网络中的关键作用

    The Anatomy Spread of Online Opinion Polarization: The Pivotal Role of Super-Spreaders in Social Networks. (arXiv:2401.01349v1 [physics.soc-ph])

    [http://arxiv.org/abs/2401.01349](http://arxiv.org/abs/2401.01349)

    该研究揭示了在社交网络中塑造意见的超级传播者的重要角色，通过对超级传播者行为的调查和分析，提供了对群体动态和意见形成的条件的理解，对改进在线交流安全和社交影响力的认识具有启示作用。

    

    该研究探讨了在网络中塑造意见的“超级传播者”的角色，区分了三种类型：A、B和C。A型在塑造意见方面具有重要影响力，B型则起到了平衡A型的作用，C型则像媒体一样，提供客观观点，并潜在地调控A型和B型的影响力。研究使用置信系数和z分数来调查超级传播者的行为，着重考察影响群体动态和意见形成的条件，包括环境因素和随时间的遗忘。研究结果为改善在线交流安全和了解社会影响力提供了深入见解。

    The study investigates the role of 'superspreaders' in shaping opinions within networks, distinguishing three types: A, B, and C. Type A has a significant influence in shaping opinions, Type B acts as a counterbalance to A, and Type C functions like media, providing an objective viewpoint and potentially regulating A and B's influence. The research uses a confidence coefficient and z-score to survey superspreaders' behaviors, with a focus on the conditions affecting group dynamics and opinion formation, including environmental factors and forgetfulness over time. The findings offer insights for improving online communication security and understanding social influence.
    
[^10]: 大型语言模型可以学习规则

    Large Language Models can Learn Rules. (arXiv:2310.07064v1 [cs.AI])

    [http://arxiv.org/abs/2310.07064](http://arxiv.org/abs/2310.07064)

    大型语言模型(LLMs)在各种推理任务中展示了令人印象深刻的性能。为了提高提示方法的准确性和一致性，我们提出了Hypotheses-to-Theories (HtT)框架，用于学习LLMs推理的规则库，从而改进了现有的提示方法。

    

    当给出一些示例和中间步骤时，大型语言模型(LLMs)在各种推理任务中展示了令人印象深刻的性能。然而，依赖LLM中的隐式知识的提示方法在隐式知识错误或与任务不一致时往往会产生错误的答案。为解决这个问题，我们提出了"假设到理论" (HtT) 框架，用于学习LLMs推理的规则库。HtT包括两个阶段，归纳阶段和演绎阶段。在归纳阶段，首先要求LLM根据一组训练示例生成和验证规则。出现并导致正确答案的规则将被收集形成一个规则库。在演绎阶段，然后要求LLM使用学习的规则库进行推理以回答测试问题。在数值推理和关系推理问题上的实验证明，HtT改进了现有的提示方法，使其性能提升。

    When prompted with a few examples and intermediate steps, large language models (LLMs) have demonstrated impressive performance in various reasoning tasks. However, prompting methods that rely on implicit knowledge in an LLM often hallucinate incorrect answers when the implicit knowledge is wrong or inconsistent with the task. To tackle this problem, we present Hypotheses-to-Theories (HtT), a framework that learns a rule library for reasoning with LLMs. HtT contains two stages, an induction stage and a deduction stage. In the induction stage, an LLM is first asked to generate and verify rules over a set of training examples. Rules that appear and lead to correct answers sufficiently often are collected to form a rule library. In the deduction stage, the LLM is then prompted to employ the learned rule library to perform reasoning to answer test questions. Experiments on both numerical reasoning and relational reasoning problems show that HtT improves existing prompting methods, with an 
    
[^11]: Lemur：在自动程序验证中集成大型语言模型

    Lemur: Integrating Large Language Models in Automated Program Verification. (arXiv:2310.04870v2 [cs.FL] UPDATED)

    [http://arxiv.org/abs/2310.04870](http://arxiv.org/abs/2310.04870)

    本论文提出了一种将LLMs和自动推理器结合起来进行自动程序验证的通用方法，并证明了其完备性。这个方法在一些合成和竞争基准上取得了实际的改进。

    

    LLMs在代码理解能力上的展示引发了一个问题：它们是否可以用于自动程序验证，这是一个通常需要高级抽象推理的任务，对于验证工具来说是具有挑战性的。我们提出了一种将LLMs的能力和自动推理器结合起来进行自动程序验证的通用方法。我们正式描述了这种方法论，将其作为推导规则的集合进行论证其完备性。我们将计算机推理形成为一个完备的自动验证过程，这在一组合成和竞争基准上带来了实际的改进。

    The demonstrated code-understanding capability of LLMs raises the question of whether they can be used for automated program verification, a task that often demands high-level abstract reasoning about program properties, which is challenging for verification tools. We propose a general methodology to combine the power of LLMs and automated reasoners for automated program verification. We formally describe this methodology as a set of derivation rules and prove its soundness. We instantiate the calculus as a sound automated verification procedure, which led to practical improvements on a set of synthetic and competition benchmarks.
    
[^12]: 游戏理论的鲁棒强化学习处理时间耦合的干扰

    Game-Theoretic Robust Reinforcement Learning Handles Temporally-Coupled Perturbations. (arXiv:2307.12062v1 [cs.LG])

    [http://arxiv.org/abs/2307.12062](http://arxiv.org/abs/2307.12062)

    这篇论文提出了一种新的鲁棒强化学习方法，通过将时间耦合的鲁棒强化学习问题视为两人零和游戏来处理问题，并通过找到近似均衡来确保代理对时间耦合干扰的鲁棒性。实验结果显示，该方法在各种连续控制任务中相比基准方法表现出显著的鲁棒性优势。

    

    鲁棒强化学习旨在训练能够在环境干扰或对抗攻击下表现良好的策略。现有方法通常假设可能干扰的空间在各个时间步骤保持不变。然而，在许多情况下，给定时间步骤上可能干扰的空间取决于过去的干扰。我们正式引入时间耦合干扰，对现有的鲁棒强化学习方法提出了新的挑战。为了应对这个挑战，我们提出了GRAD，一种新的游戏理论方法，将时间耦合鲁棒强化学习问题视为部分可观察的两人零和游戏。通过在这个游戏中找到一个近似均衡，GRAD确保了代理的对时间耦合干扰的鲁棒性。对各种连续控制任务的实证实验表明，我们提出的方法相比基准方法在标准和时间耦合干扰下具有显著的鲁棒性优势。

    Robust reinforcement learning (RL) seeks to train policies that can perform well under environment perturbations or adversarial attacks. Existing approaches typically assume that the space of possible perturbations remains the same across timesteps. However, in many settings, the space of possible perturbations at a given timestep depends on past perturbations. We formally introduce temporally-coupled perturbations, presenting a novel challenge for existing robust RL methods. To tackle this challenge, we propose GRAD, a novel game-theoretic approach that treats the temporally-coupled robust RL problem as a partially-observable two-player zero-sum game. By finding an approximate equilibrium in this game, GRAD ensures the agent's robustness against temporally-coupled perturbations. Empirical experiments on a variety of continuous control tasks demonstrate that our proposed approach exhibits significant robustness advantages compared to baselines against both standard and temporally-coupl
    
[^13]: 强化学习中的结构：调查与开放问题

    Structure in Reinforcement Learning: A Survey and Open Problems. (arXiv:2306.16021v1 [cs.LG])

    [http://arxiv.org/abs/2306.16021](http://arxiv.org/abs/2306.16021)

    这项调查研究了强化学习中结构的角色和重要性，并介绍了各个子领域在提高强化学习的性能方面所做的工作。

    

    强化学习（RL）借助深度神经网络（DNN）在函数逼近方面的表达能力，已经在许多应用中取得了相当大的成功。然而，在应对多样且不可预测的动态、嘈杂信号以及庞大的状态和动作空间等各种真实场景时，其实用性仍然有限。这个限制源于诸如数据效率低、泛化能力有限、缺少安全保证和不可解释性等问题。为了克服这些挑战并在这些关键指标上提高性能，一个有前途的途径是将问题的附加结构信息纳入强化学习的学习过程中。强化学习的各个子领域已经提出了许多方法来纳入这样的归纳偏差。我们将这些多样化的方法统一到一个框架下，揭示结构在学习问题中的作用。

    Reinforcement Learning (RL), bolstered by the expressive capabilities of Deep Neural Networks (DNNs) for function approximation, has demonstrated considerable success in numerous applications. However, its practicality in addressing a wide range of real-world scenarios, characterized by diverse and unpredictable dynamics, noisy signals, and large state and action spaces, remains limited. This limitation stems from issues such as poor data efficiency, limited generalization capabilities, a lack of safety guarantees, and the absence of interpretability, among other factors. To overcome these challenges and improve performance across these crucial metrics, one promising avenue is to incorporate additional structural information about the problem into the RL learning process. Various sub-fields of RL have proposed methods for incorporating such inductive biases. We amalgamate these diverse methodologies under a unified framework, shedding light on the role of structure in the learning prob
    
[^14]: 多天线系统中数字无线协作联邦学习的研究

    Digital Over-the-Air Federated Learning in Multi-Antenna Systems. (arXiv:2302.14648v2 [cs.IT] UPDATED)

    [http://arxiv.org/abs/2302.14648](http://arxiv.org/abs/2302.14648)

    本文研究了在多天线系统中采用数字调制和空中计算的情况下，联邦学习的性能优化问题。通过结合数字调制和AirComp，提出了一种改进的联邦平均算法，以解决无线信道衰落导致的总体失真问题。

    

    本文研究了在实际的无线多输入多输出（MIMO）通信系统中，采用数字调制和空中计算（AirComp）的情况下，联邦学习（FL）的性能优化。特别考虑了一个MIMO系统，在该系统中，边缘设备使用波束形成将其本地收集的数据训练的本地FL模型传输给参数服务器（PS），以最大化可调度传输的设备数量。PS作为中央控制器，使用接收到的本地FL模型生成一个全局FL模型并广播给所有设备。由于无线网络中的带宽有限，采用了AirComp以实现高效的无线数据聚合。然而，无线信道的衰落会产生基于AirComp的FL方案中的总体失真。为了克服这一挑战，我们提出了一种改进的联邦平均（FedAvg）算法，将数字调制与AirComp相结合以减轻失真问题。

    In this paper, the performance optimization of federated learning (FL), when deployed over a realistic wireless multiple-input multiple-output (MIMO) communication system with digital modulation and over-the-air computation (AirComp) is studied. In particular, a MIMO system is considered in which edge devices transmit their local FL models (trained using their locally collected data) to a parameter server (PS) using beamforming to maximize the number of devices scheduled for transmission. The PS, acting as a central controller, generates a global FL model using the received local FL models and broadcasts it back to all devices. Due to the limited bandwidth in a wireless network, AirComp is adopted to enable efficient wireless data aggregation. However, fading of wireless channels can produce aggregate distortions in an AirComp-based FL scheme. To tackle this challenge, we propose a modified federated averaging (FedAvg) algorithm that combines digital modulation with AirComp to mitigate
    

