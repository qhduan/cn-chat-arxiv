# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Unsolvable Problem Detection: Evaluating Trustworthiness of Vision Language Models](https://arxiv.org/abs/2403.20331) | 本文提出了一个新颖且重要的挑战，即Unsolvable Problem Detection（UPD），用于评估视觉语言模型在视觉问答任务中能否在面对不可解问题时保持答案的能力，并通过广泛实验发现大多数模型存在改进的空间。 |
| [^2] | [Provably Sample Efficient RLHF via Active Preference Optimization](https://arxiv.org/abs/2402.10500) | 通过Active Preference Optimization算法，在Bradley-Terry-Luce偏好模型下实现了RLHF的样本效率提高，优化了对提示收集偏好数据的策略。 |
| [^3] | [SecFormer: Towards Fast and Accurate Privacy-Preserving Inference for Large Language Models.](http://arxiv.org/abs/2401.00793) | SecFormer是一个优化框架，旨在实现Transformer模型的快速准确隐私保护推理。通过消除高成本的指数和线性操作，SecFormer能够有效解决在大型语言模型中应用SMPC时的性能问题。 |
| [^4] | [Large Language Model as Autonomous Decision Maker.](http://arxiv.org/abs/2308.12519) | 本文提出了一种方法JuDec，为大型语言模型(LLMs)赋予了自我判断的能力，使其能够作为自主决策者实现自主判断和决策探索。实验结果显示JuDec在不同任务上表现优异，提高了通过率并降低了成本。 |
| [^5] | [A Multiple Choices Reading Comprehension Corpus for Vietnamese Language Education.](http://arxiv.org/abs/2303.18162) | ViMMRC 2.0是一个针对越南教材中的多项选择阅读理解任务的语料库，共有699篇散文和诗歌以及5,273个问题。该数据集中的问题选项不固定为四个，且问题难度增加，需要使用多步注意力网络与变压器相结合的多阶段方法来处理。 |

# 详细

[^1]: 不可解问题检测：评估视觉语言模型的可信度

    Unsolvable Problem Detection: Evaluating Trustworthiness of Vision Language Models

    [https://arxiv.org/abs/2403.20331](https://arxiv.org/abs/2403.20331)

    本文提出了一个新颖且重要的挑战，即Unsolvable Problem Detection（UPD），用于评估视觉语言模型在视觉问答任务中能否在面对不可解问题时保持答案的能力，并通过广泛实验发现大多数模型存在改进的空间。

    

    本文介绍了一个新颖而重要的挑战，即Unsolvable Problem Detection（UPD），用于评估视觉语言模型（VLMs）在视觉问答（VQA）任务中面对不可解问题时保持答案的能力。UPD包括三个不同的设置：缺失答案检测（AAD）、不兼容答案集检测（IASD）和不兼容视觉问题检测（IVQD）。通过广泛的实验深入研究UPD问题表明，大多数VLMs，包括GPT-4V和LLaVA-Next-34B，在各种程度上都很难应对我们的基准测试，突显了改进的重要空间。为了解决UPD，我们探索了无需训练和基于训练的解决方案，提供了对其有效性和局限性的新见解。我们希望我们的见解，以及在提议的UPD设置内的未来努力，将增强对VLMs的更广泛理解和发展。

    arXiv:2403.20331v1 Announce Type: cross  Abstract: This paper introduces a novel and significant challenge for Vision Language Models (VLMs), termed Unsolvable Problem Detection (UPD). UPD examines the VLM's ability to withhold answers when faced with unsolvable problems in the context of Visual Question Answering (VQA) tasks. UPD encompasses three distinct settings: Absent Answer Detection (AAD), Incompatible Answer Set Detection (IASD), and Incompatible Visual Question Detection (IVQD). To deeply investigate the UPD problem, extensive experiments indicate that most VLMs, including GPT-4V and LLaVA-Next-34B, struggle with our benchmarks to varying extents, highlighting significant room for the improvements. To address UPD, we explore both training-free and training-based solutions, offering new insights into their effectiveness and limitations. We hope our insights, together with future efforts within the proposed UPD settings, will enhance the broader understanding and development of
    
[^2]: 通过主动偏好优化实现经验证的样本效率的RLHF

    Provably Sample Efficient RLHF via Active Preference Optimization

    [https://arxiv.org/abs/2402.10500](https://arxiv.org/abs/2402.10500)

    通过Active Preference Optimization算法，在Bradley-Terry-Luce偏好模型下实现了RLHF的样本效率提高，优化了对提示收集偏好数据的策略。

    

    强化学习从人类反馈（RLHF）在将大型语言模型（LLMs）与人类偏好相一致方面至关重要。虽然这些对齐的生成模型已经在各种任务中展示出令人印象深刻的能力，但是依赖高质量的人类偏好数据在实际RLHF实施中构成了昂贵的瓶颈。因此，需要更好和自适应的数据收集策略。为此，我们将RLHF以上下文偏好赌博机问题的形式框定，其中提示作为上下文，并表明通过随机选择提示收集偏好数据的天真方式导致一个在奖励方面具有$\Omega(1)$次优性差距的策略。然后，我们提出了$\textit{Active Preference Optimization}$（$\texttt{APO}$）算法，该算法积极选择提示以收集偏好数据。在Bradley-Terry-Luce（BTL）偏好模型下，\texttt{APO}实现了样本效率，而不会妥协于polic

    arXiv:2402.10500v1 Announce Type: cross  Abstract: Reinforcement Learning from Human Feedback (RLHF) is pivotal in aligning Large Language Models (LLMs) with human preferences. While these aligned generative models have demonstrated impressive capabilities across various tasks, the dependence on high-quality human preference data poses a costly bottleneck in practical implementation of RLHF. Hence better and adaptive strategies for data collection is needed. To this end, we frame RLHF as a contextual preference bandit problem with prompts as contexts and show that the naive way of collecting preference data by choosing prompts uniformly at random leads to a policy that suffers an $\Omega(1)$ suboptimality gap in rewards. Then we propose $\textit{Active Preference Optimization}$ ($\texttt{APO}$), an algorithm that actively selects prompts to collect preference data. Under the Bradley-Terry-Luce (BTL) preference model, \texttt{APO} achieves sample efficiency without compromising on polic
    
[^3]: SecFormer：面向大型语言模型的快速准确隐私保护推理

    SecFormer: Towards Fast and Accurate Privacy-Preserving Inference for Large Language Models. (arXiv:2401.00793v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2401.00793](http://arxiv.org/abs/2401.00793)

    SecFormer是一个优化框架，旨在实现Transformer模型的快速准确隐私保护推理。通过消除高成本的指数和线性操作，SecFormer能够有效解决在大型语言模型中应用SMPC时的性能问题。

    

    随着在云平台上部署大型语言模型以提供推理服务的使用增加，隐私问题日益加剧，尤其是涉及投资计划和银行账户等敏感数据。安全多方计算（SMPC）被视为保护推理数据和模型参数隐私的一种有前途的解决方案。然而，SMPC在大型语言模型（特别是基于Transformer架构的模型）的隐私保护推理中的应用往往会导致显著的减速或性能下降。这主要是由于Transformer架构中的众多非线性操作不适合SMPC，并且难以有效规避或优化。为了解决这个问题，我们引入了一个先进的优化框架，称为SecFormer，以实现Transformer模型的快速准确隐私保护推理。通过实施模型设计优化，我们成功消除了高成本的指数和线性操作，并取得了良好的性能。

    With the growing use of large language models hosted on cloud platforms to offer inference services, privacy concerns are escalating, especially concerning sensitive data like investment plans and bank account details. Secure Multi-Party Computing (SMPC) emerges as a promising solution to protect the privacy of inference data and model parameters. However, the application of SMPC in Privacy-Preserving Inference (PPI) for large language models, particularly those based on the Transformer architecture, often leads to considerable slowdowns or declines in performance. This is largely due to the multitude of nonlinear operations in the Transformer architecture, which are not well-suited to SMPC and difficult to circumvent or optimize effectively. To address this concern, we introduce an advanced optimization framework called SecFormer, to achieve fast and accurate PPI for Transformer models. By implementing model design optimization, we successfully eliminate the high-cost exponential and 
    
[^4]: 大型语言模型作为自主决策者

    Large Language Model as Autonomous Decision Maker. (arXiv:2308.12519v1 [cs.CL])

    [http://arxiv.org/abs/2308.12519](http://arxiv.org/abs/2308.12519)

    本文提出了一种方法JuDec，为大型语言模型(LLMs)赋予了自我判断的能力，使其能够作为自主决策者实现自主判断和决策探索。实验结果显示JuDec在不同任务上表现优异，提高了通过率并降低了成本。

    

    虽然大型语言模型(LLMs)展示了令人印象深刻的语言理解和上下文学习能力，但它们在解决现实世界任务时仍严重依赖于专家知识的指导。为了发挥LLMs作为自主决策者的潜力，本文提出了一种称为JuDec的方法，赋予LLMs自我判断的能力，使其能够实现自主判断和决策探索。具体而言，在JuDec中，设计了基于Elo的自我判断机制，通过对两个解决方案进行配对比较，为决策步骤分配Elo分数，以判断它们的价值和效用，并相应地引导决策搜索过程朝向最优解。在ToolBench数据集上的实验结果表明，JuDec相对于基准模型具有优势，在不同任务上的通过率提高了10%以上。它提供了更高质量的解决方案并降低了成本(ChatGPT API调用)。

    While large language models (LLMs) exhibit impressive language understanding and in-context learning abilities, their decision-making ability still heavily relies on the guidance of task-specific expert knowledge when solving real-world tasks. To unleash the potential of LLMs as autonomous decision makers, this paper presents an approach JuDec to endow LLMs with the self-judgment ability, enabling LLMs to achieve autonomous judgment and exploration for decision making. Specifically, in JuDec, Elo-based Self-Judgment Mechanism is designed to assign Elo scores to decision steps to judge their values and utilities via pairwise comparisons between two solutions and then guide the decision-searching process toward the optimal solution accordingly. Experimental results on the ToolBench dataset demonstrate JuDec's superiority over baselines, achieving over 10% improvement in Pass Rate on diverse tasks. It offers higher-quality solutions and reduces costs (ChatGPT API calls), highlighting its 
    
[^5]: 用于越南语教育的多项选择阅读理解语料库

    A Multiple Choices Reading Comprehension Corpus for Vietnamese Language Education. (arXiv:2303.18162v1 [cs.CL])

    [http://arxiv.org/abs/2303.18162](http://arxiv.org/abs/2303.18162)

    ViMMRC 2.0是一个针对越南教材中的多项选择阅读理解任务的语料库，共有699篇散文和诗歌以及5,273个问题。该数据集中的问题选项不固定为四个，且问题难度增加，需要使用多步注意力网络与变压器相结合的多阶段方法来处理。

    

    机器阅读理解是近年来一个有趣且具有挑战性的任务，其目的在于从文本中提取有用的信息。我们引入了ViMMRC 2.0，这是对之前ViMMRC的扩展，用于越南教材中的多项选择阅读理解任务，这些教材包含了一年级至十二年级学生的阅读文章。该数据集包含了699篇散文和诗歌，以及5,273个问题。与之前的版本不同，新数据集中的问题选项不固定为四个，同时还增加了问题的难度，这使得模型需要寻找正确的选择。电脑必须理解整个阅读文章的上下文、问题以及每个选项的内容才能提取正确答案。因此，我们提出了将多步注意力网络（MAN）与变压器相结合的多阶段方法来处理这个任务。

    Machine reading comprehension has been an interesting and challenging task in recent years, with the purpose of extracting useful information from texts. To attain the computer ability to understand the reading text and answer relevant information, we introduce ViMMRC 2.0 - an extension of the previous ViMMRC for the task of multiple-choice reading comprehension in Vietnamese Textbooks which contain the reading articles for students from Grade 1 to Grade 12. This dataset has 699 reading passages which are prose and poems, and 5,273 questions. The questions in the new dataset are not fixed with four options as in the previous version. Moreover, the difficulty of questions is increased, which challenges the models to find the correct choice. The computer must understand the whole context of the reading passage, the question, and the content of each choice to extract the right answers. Hence, we propose the multi-stage approach that combines the multi-step attention network (MAN) with the
    

