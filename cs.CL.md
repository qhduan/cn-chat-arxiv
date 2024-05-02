# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [The Role of $n$-gram Smoothing in the Age of Neural Networks](https://arxiv.org/abs/2403.17240) | 本文重新探讨了在神经语言模型时代古典$n$-gram平滑技术可能发挥的作用，并提出了将任何$n$-gram平滑技术转换为神经语言模型兼容正则化器的通用框架 |
| [^2] | [NUMTEMP: A real-world benchmark to verify claims with statistical and temporal expressions](https://arxiv.org/abs/2403.17169) | NUMTEMP是一个真实世界基准，专注于验证复杂的数字论点，量化了现有解决方案的局限性，并提供了一种解决真实世界数字论点验证挑战的方法。 |
| [^3] | [Reverse Training to Nurse the Reversal Curse](https://arxiv.org/abs/2403.13799) | 该研究提出了一种称为逆向训练的替代训练方案，通过在正向和逆向方向上训练语言模型并保留选定子串，成功解决了大型语言模型面临的逆转诅咒问题。 |
| [^4] | [Correcting misinformation on social media with a large language model](https://arxiv.org/abs/2403.11169) | 提出了一种名为MUSE的大型语言模型，通过访问最新信息并评估可信度，以解决社交媒体上误信息纠正的难题。 |
| [^5] | [Brilla AI: AI Contestant for the National Science and Maths Quiz](https://arxiv.org/abs/2403.01699) | 人工智能参赛者Brilla AI在全国科学与数学竞赛中表现优秀，为缺乏合格教师的非洲提供了学习支持。 |
| [^6] | [Measuring and Controlling Persona Drift in Language Model Dialogs](https://arxiv.org/abs/2402.10962) | 提出了一种量化基准来测量语言模型对话中的“人设”漂移，并提出了一种称为split-softmax的轻量级方法来对抗注意力衰减和“人设”漂移 |
| [^7] | [Large Language Models as Zero-shot Dialogue State Tracker through Function Calling](https://arxiv.org/abs/2402.10466) | 本研究提出了一种通过函数调用将大型语言模型用于零-shot对话状态追踪的新方法，能够在任务导向对话中取得出色的性能，适应不同领域而无需大量数据收集或模型调整。 |
| [^8] | [Doing Experiments and Revising Rules with Natural Language and Probabilistic Reasoning](https://arxiv.org/abs/2402.06025) | 本论文建立了一个计算模型来模拟人们通过实验主动推断隐藏规则的过程，并发现显式假设、概率规则和在线更新的组合可以解释人们在类似Zendo任务上的表现。 |
| [^9] | [The Power of Noise: Redefining Retrieval for RAG Systems.](http://arxiv.org/abs/2401.14887) | 本研究通过分析和评估检索增强生成（RAG）系统中的信息检索（IR）组件，填补了目前研究中忽视的领域，在有效的RAG的提示表述中，不相关文档的包含可能会对系统性能产生负面影响。 |
| [^10] | [Response: Emergent analogical reasoning in large language models.](http://arxiv.org/abs/2308.16118) | 该论文回应了关于大型语言模型中紧急类比推理的主张，并通过提供字符串类比的反例来反驳。在测试中，GPT-3无法解决最简单的类比问题。为了加强零点推理等人类推理的主张，需要发展出排除数据记忆的方法。 |
| [^11] | [GRASP: A Rehearsal Policy for Efficient Online Continual Learning.](http://arxiv.org/abs/2308.13646) | GRASP是一种新的样本选择策略，根据样本的代表性选择最适合学习的样本，从而提高了在线渐进式学习的效率。 |
| [^12] | [Retroformer: Retrospective Large Language Agents with Policy Gradient Optimization.](http://arxiv.org/abs/2308.02151) | 本文介绍了一种通过策略梯度优化的回顾性大型语言代理框架，该框架通过学习环境反馈来调整语言代理的提示，从而优化其性能。这种代理能够从多个环境和任务中学习奖励，并通过总结以前任务的根本原因来改进语言代理提示。 |

# 详细

[^1]: 在神经网络时代的$n$-gram平滑作用

    The Role of $n$-gram Smoothing in the Age of Neural Networks

    [https://arxiv.org/abs/2403.17240](https://arxiv.org/abs/2403.17240)

    本文重新探讨了在神经语言模型时代古典$n$-gram平滑技术可能发挥的作用，并提出了将任何$n$-gram平滑技术转换为神经语言模型兼容正则化器的通用框架

    

    在将近三十年的时间里，基于$n$-gram假设的语言模型一直是该任务的技术水平。它们成功的关键在于应用各种平滑技术来对抗过拟合。然而，当神经语言模型取代$n$-gram模型成为最佳表现者时，$n$-gram平滑技术变得不太相关。事实上，可以毫不夸张地说，对$n$-gram平滑技术的研究在这一时代变得停滞。本文重新探讨了在神经语言模型时代古典$n$-gram平滑技术可能发挥的作用。首先，我们在标签平滑和add-$\lambda$平滑之间建立了一个正式等价性，标签平滑是一种神经语言模型的流行正则化技术。其次，我们推导了一个通用框架，将\emph{任何} $n$-gram平滑技术转换为与神经语言模型兼容的正则化器。我们的实证结果表明

    arXiv:2403.17240v1 Announce Type: new  Abstract: For nearly three decades, language models derived from the $n$-gram assumption held the state of the art on the task. The key to their success lay in the application of various smoothing techniques that served to combat overfitting. However, when neural language models toppled $n$-gram models as the best performers, $n$-gram smoothing techniques became less relevant. Indeed, it would hardly be an understatement to suggest that the line of inquiry into $n$-gram smoothing techniques became dormant. This paper re-opens the role classical $n$-gram smoothing techniques may play in the age of neural language models. First, we draw a formal equivalence between label smoothing, a popular regularization technique for neural language models, and add-$\lambda$ smoothing. Second, we derive a generalized framework for converting \emph{any} $n$-gram smoothing technique into a regularizer compatible with neural language models. Our empirical results fi
    
[^2]: NUMTEMP：一个用于验证带有统计和时间表达式的论点的真实世界基准

    NUMTEMP: A real-world benchmark to verify claims with statistical and temporal expressions

    [https://arxiv.org/abs/2403.17169](https://arxiv.org/abs/2403.17169)

    NUMTEMP是一个真实世界基准，专注于验证复杂的数字论点，量化了现有解决方案的局限性，并提供了一种解决真实世界数字论点验证挑战的方法。

    

    自动事实检查在数字时代应对不断增长的错误信息方面引起了极大兴趣。现有系统主要专注于维基百科上的合成论点，并且在真实世界论点上也取得了显著进展。在本文中，我们发布了Numtemp，一个多样化、多领域的数据集，专门关注数字论点，包括时间、统计和多样化方面的细粒度元数据，并且具有不泄露的证据收集。这解决了验证真实世界数字论点的挑战，这些论点复杂，往往缺乏精确信息，这是现有作品主要关注合成论点未解决的问题。我们评估并量化了现有解决方案在验证数字论点任务中的局限性。我们还评估了基于论点分解的方法、基于数字理解的模型，我们的最佳基线实现了58.32的宏F1分数。这证明了Numtemp的关键价值。

    arXiv:2403.17169v1 Announce Type: cross  Abstract: Automated fact checking has gained immense interest to tackle the growing misinformation in the digital era. Existing systems primarily focus on synthetic claims on Wikipedia, and noteworthy progress has also been made on real-world claims. In this work, we release Numtemp, a diverse, multi-domain dataset focused exclusively on numerical claims, encompassing temporal, statistical and diverse aspects with fine-grained metadata and an evidence collection without leakage. This addresses the challenge of verifying real-world numerical claims, which are complex and often lack precise information, not addressed by existing works that mainly focus on synthetic claims. We evaluate and quantify the limitations of existing solutions for the task of verifying numerical claims. We also evaluate claim decomposition based methods, numerical understanding based models and our best baselines achieves a macro-F1 of 58.32. This demonstrates that Numtemp
    
[^3]: 逆向训练以消除逆转诅咒

    Reverse Training to Nurse the Reversal Curse

    [https://arxiv.org/abs/2403.13799](https://arxiv.org/abs/2403.13799)

    该研究提出了一种称为逆向训练的替代训练方案，通过在正向和逆向方向上训练语言模型并保留选定子串，成功解决了大型语言模型面临的逆转诅咒问题。

    

    大型语言模型（LLMs）存在一个令人惊讶的失败现象：当训练模型以"A具有特征B"为基础时，它们无法泛化到"B是A的特征"，这被称为逆转诅咒。即使在使用数万亿令牌进行训练时，由于齐夫定律的存在，这个问题仍然存在，这意味着即使我们在整个互联网上进行训练，该问题仍然会出现。本研究提出了一种名为逆向训练的替代训练方案，在其中所有单词被使用两次，从而使可用令牌数量加倍。该LLM在正向和逆向方向上进行训练，通过颠倒训练字符串来颠倒训练过程，同时保留（即不颠倒）选定的子串，如实体。我们展示了数据匹配的逆向训练模型在标准任务上比标准模型表现更优秀，并且计算匹配的逆向训练模型在逆转任务上表现出远远优于标准模型的性能，有助于解决逆转诅咒问题。

    arXiv:2403.13799v1 Announce Type: new  Abstract: Large language models (LLMs) have a surprising failure: when trained on "A has a feature B", they do not generalize to "B is a feature of A", which is termed the Reversal Curse. Even when training with trillions of tokens this issue still appears due to Zipf's law - hence even if we train on the entire internet. This work proposes an alternative training scheme, called reverse training, whereby all words are used twice, doubling the amount of available tokens. The LLM is trained in both forward and reverse directions by reversing the training strings while preserving (i.e., not reversing) chosen substrings, such as entities. We show that data-matched reverse-trained models provide superior performance to standard models on standard tasks, and compute-matched reverse-trained models provide far superior performance on reversal tasks, helping resolve the reversal curse issue.
    
[^4]: 使用大型语言模型纠正社交媒体上的错误信息

    Correcting misinformation on social media with a large language model

    [https://arxiv.org/abs/2403.11169](https://arxiv.org/abs/2403.11169)

    提出了一种名为MUSE的大型语言模型，通过访问最新信息并评估可信度，以解决社交媒体上误信息纠正的难题。

    

    误信息会破坏公众对科学和民主的信任，特别是在社交媒体上，不准确信息会迅速传播。专家和普通人通过手动识别和解释不准确信息已经被证明是有效的纠正误信息的方法。然而，这种方法很难扩展，这是一个担忧，因为大型语言模型（LLMs）等技术使误信息更容易生成。LLMs还具有多功能能力，可以加速纠正误信息；然而，它们由于缺乏最新信息、倾向于生成似是而非的内容和引用以及无法处理多模态信息而面临困难。为了解决这些问题，我们提出了MUSE，这是一个带有最新信息访问和可信度评估的LLM。通过检索上下文证据和反驳，MUSE可以提供准确可信的解释和参考。它还描述

    arXiv:2403.11169v1 Announce Type: cross  Abstract: Misinformation undermines public trust in science and democracy, particularly on social media where inaccuracies can spread rapidly. Experts and laypeople have shown to be effective in correcting misinformation by manually identifying and explaining inaccuracies. Nevertheless, this approach is difficult to scale, a concern as technologies like large language models (LLMs) make misinformation easier to produce. LLMs also have versatile capabilities that could accelerate misinformation correction; however, they struggle due to a lack of recent information, a tendency to produce plausible but false content and references, and limitations in addressing multimodal information. To address these issues, we propose MUSE, an LLM augmented with access to and credibility evaluation of up-to-date information. By retrieving contextual evidence and refutations, MUSE can provide accurate and trustworthy explanations and references. It also describes 
    
[^5]: Brilla AI: 全国科学与数学竞赛的人工智能参赛者

    Brilla AI: AI Contestant for the National Science and Maths Quiz

    [https://arxiv.org/abs/2403.01699](https://arxiv.org/abs/2403.01699)

    人工智能参赛者Brilla AI在全国科学与数学竞赛中表现优秀，为缺乏合格教师的非洲提供了学习支持。

    

    非洲大陆缺乏足够的合格教师，这阻碍了提供足够的学习支持。人工智能有可能增强有限数量教师的努力，从而带来更好的学习成果。本文描述并评估了NSMQ AI Grand Challenge的首要成果，该挑战提出了一个强大的现实基准，用于评估此类人工智能：“建立一个人工智能，参加加纳的全国科学与数学竞赛（NSMQ），并获胜——在比赛的所有轮次和阶段中表现优于最优秀的参赛者”。NSMQ是加纳的高中学生每年举行的现场科学与数学竞赛，3队2名学生通过回答生物学、化学、物理和数学问题在5轮比赛中竞争，逐渐晋级至最终冠军的队伍。在本研究中，我们建立了Brilla AI，一个参加NSMQ竞赛的人工智能选手。

    arXiv:2403.01699v1 Announce Type: cross  Abstract: The African continent lacks enough qualified teachers which hampers the provision of adequate learning support. An AI could potentially augment the efforts of the limited number of teachers, leading to better learning outcomes. Towards that end, this work describes and evaluates the first key output for the NSMQ AI Grand Challenge, which proposes a robust, real-world benchmark for such an AI: "Build an AI to compete live in Ghana's National Science and Maths Quiz (NSMQ) competition and win - performing better than the best contestants in all rounds and stages of the competition". The NSMQ is an annual live science and mathematics competition for senior secondary school students in Ghana in which 3 teams of 2 students compete by answering questions across biology, chemistry, physics, and math in 5 rounds over 5 progressive stages until a winning team is crowned for that year. In this work, we built Brilla AI, an AI contestant that we de
    
[^6]: 在语言模型对话中测量和控制“人设”漂移

    Measuring and Controlling Persona Drift in Language Model Dialogs

    [https://arxiv.org/abs/2402.10962](https://arxiv.org/abs/2402.10962)

    提出了一种量化基准来测量语言模型对话中的“人设”漂移，并提出了一种称为split-softmax的轻量级方法来对抗注意力衰减和“人设”漂移

    

    提示是定制语言模型聊天机器人的标准工具，使其能够承担特定的“人设”。在使用提示时的一个隐含假设是，它们将是稳定的，因此聊天机器人将在整个对话过程中继续根据规定的“人设”生成文本。我们提出了一个量化基准来测试这一假设，通过两个个性化聊天机器人之间的自我对话来评估“人设”的稳定性。我们对流行模型如LLaMA2-chat-70B进行测试，发现在八轮对话中存在显著的“人设”漂移。对这一现象的实证和理论分析表明，由于长对话中的注意力衰减，变压器注意力机制起到了一定作用。为了对抗注意力衰减和“人设”漂移，我们提出了一种称为split-softmax的轻量级方法，与两个强基线方法相比表现优异。

    arXiv:2402.10962v1 Announce Type: cross  Abstract: Prompting is a standard tool for customizing language-model chatbots, enabling them to take on a specific "persona". An implicit assumption in the use of prompts is that they will be stable, so the chatbot will continue to generate text according to the stipulated persona for the duration of a conversation. We propose a quantitative benchmark to test this assumption, evaluating persona stability via self-chats between two personalized chatbots. Testing popular models like LLaMA2-chat-70B, we reveal a significant persona drift within eight rounds of conversations. An empirical and theoretical analysis of this phenomenon suggests the transformer attention mechanism plays a role, due to attention decay over long exchanges. To combat attention decay and persona drift, we propose a lightweight method called split-softmax, which compares favorably against two strong baselines.
    
[^7]: 将大型语言模型作为零-shot对话状态追踪器通过函数调用

    Large Language Models as Zero-shot Dialogue State Tracker through Function Calling

    [https://arxiv.org/abs/2402.10466](https://arxiv.org/abs/2402.10466)

    本研究提出了一种通过函数调用将大型语言模型用于零-shot对话状态追踪的新方法，能够在任务导向对话中取得出色的性能，适应不同领域而无需大量数据收集或模型调整。

    

    大型语言模型（LLMs）在会话系统中日益普遍，这是因为它们在一般情境中具有先进的理解和生成能力。然而，在需要不仅进行响应生成还需要在特定任务和领域内进行有效对话状态追踪（DST）的任务导向对话（TOD）中，它们的有效性仍不尽人意。在这项工作中，我们提出了一种通过函数调用解决LLMs中的DST的新方法FnCTOD。这种方法改进了零-shot DST，使其能够适应各种领域，而无需进行大量数据收集或模型调整。我们的实验结果表明，我们的方法在使用开源或专有LLMs时都取得了出色的性能：通过上下文提示，使得各种7B或13B参数模型超越了之前由ChatGPT实现的最新技术成果（SOTA）的水平，并提高了ChatGPT的性能，击败了

    arXiv:2402.10466v1 Announce Type: cross  Abstract: Large language models (LLMs) are increasingly prevalent in conversational systems due to their advanced understanding and generative capabilities in general contexts. However, their effectiveness in task-oriented dialogues (TOD), which requires not only response generation but also effective dialogue state tracking (DST) within specific tasks and domains, remains less satisfying. In this work, we propose a novel approach FnCTOD for solving DST with LLMs through function calling. This method improves zero-shot DST, allowing adaptation to diverse domains without extensive data collection or model tuning. Our experimental results demonstrate that our approach achieves exceptional performance with both modestly sized open-source and also proprietary LLMs: with in-context prompting it enables various 7B or 13B parameter models to surpass the previous state-of-the-art (SOTA) achieved by ChatGPT, and improves ChatGPT's performance beating the
    
[^8]: 用自然语言和概率推理进行实验与修订规则

    Doing Experiments and Revising Rules with Natural Language and Probabilistic Reasoning

    [https://arxiv.org/abs/2402.06025](https://arxiv.org/abs/2402.06025)

    本论文建立了一个计算模型来模拟人们通过实验主动推断隐藏规则的过程，并发现显式假设、概率规则和在线更新的组合可以解释人们在类似Zendo任务上的表现。

    

    我们建立了一个计算模型，模拟人们通过实验主动推断隐藏规则的过程。该模型的基本原理是，即使规则是确定性的，学习者也会考虑更广泛的模糊概率规则，并用自然语言表示，根据近似贝叶斯原则在每次实验后在线更新自己的假设。在同一框架下，我们还根据信息论准则建立了实验设计模型。我们发现，这三个原则的组合——显式假设、概率规则和在线更新——可以解释人们在类似Zendo任务上的表现，而去掉其中任何一个组件都使得模型无法解释数据。

    We build a computational model of how humans actively infer hidden rules by doing experiments. The basic principles behind the model is that, even if the rule is deterministic, the learner considers a broader space of fuzzy probabilistic rules, which it represents in natural language, and updates its hypotheses online after each experiment according to approximately Bayesian principles. In the same framework we also model experiment design according to information-theoretic criteria. We find that the combination of these three principles -- explicit hypotheses, probabilistic rules, and online updates -- can explain human performance on a Zendo-style task, and that removing any of these components leaves the model unable to account for the data.
    
[^9]: 噪声的力量：重新定义RAG系统的检索

    The Power of Noise: Redefining Retrieval for RAG Systems. (arXiv:2401.14887v1 [cs.IR])

    [http://arxiv.org/abs/2401.14887](http://arxiv.org/abs/2401.14887)

    本研究通过分析和评估检索增强生成（RAG）系统中的信息检索（IR）组件，填补了目前研究中忽视的领域，在有效的RAG的提示表述中，不相关文档的包含可能会对系统性能产生负面影响。

    

    检索增强生成（RAG）系统相对于传统的大型语言模型（LLMs）代表了一个重大进步。RAG系统通过整合通过信息检索（IR）阶段检索的外部数据来增强其生成能力，克服了标准LLMs的限制，后者仅限于其预先训练的知识和有限的上下文窗口。这个领域的大部分研究主要集中在RAG系统内LLMs的生成方面。我们的研究填补了这一空白，通过全面而批判性地分析IR组件对RAG系统的影响。本文分析了一个检索器在有效的RAG的提示表述中应该具备的特征，重点关注应该检索哪种类型的文档。我们评估了各种因素，如文档与提示的相关性，它们的位置以及上下文中包含的数量。我们的发现揭示出，包含不相关的文档可能会…

    Retrieval-Augmented Generation (RAG) systems represent a significant advancement over traditional Large Language Models (LLMs). RAG systems enhance their generation ability by incorporating external data retrieved through an Information Retrieval (IR) phase, overcoming the limitations of standard LLMs, which are restricted to their pre-trained knowledge and limited context window. Most research in this area has predominantly concentrated on the generative aspect of LLMs within RAG systems. Our study fills this gap by thoroughly and critically analyzing the influence of IR components on RAG systems. This paper analyzes which characteristics a retriever should possess for an effective RAG's prompt formulation, focusing on the type of documents that should be retrieved. We evaluate various elements, such as the relevance of the documents to the prompt, their position, and the number included in the context. Our findings reveal, among other insights, that including irrelevant documents can
    
[^10]: 回应：大型语言模型中的紧急类比推理

    Response: Emergent analogical reasoning in large language models. (arXiv:2308.16118v1 [cs.CL])

    [http://arxiv.org/abs/2308.16118](http://arxiv.org/abs/2308.16118)

    该论文回应了关于大型语言模型中紧急类比推理的主张，并通过提供字符串类比的反例来反驳。在测试中，GPT-3无法解决最简单的类比问题。为了加强零点推理等人类推理的主张，需要发展出排除数据记忆的方法。

    

    在最近的《自然人类行为》论文中，“大型语言模型中的紧急类比推理”（Webb，Holyoak和Lu，2023），作者们认为“像GPT-3这样的大型语言模型已经获得了发现广泛类比问题的零点解的紧急能力”。在本回应中，我们提供了一些字符串类比的反例。在我们的测试中，GPT-3甚至无法解决原始论文中提出的最简单的变体问题。零点推理是一个需要非常充分证据支持的非凡主张。在我们的实验中，我们没有看到这样的证据。为了加强像零点推理这样类似人类推理的主张，重要的是该领域开发出能够排除数据记忆的方法。

    In their recent Nature Human Behaviour paper, "Emergent analogical reasoning in large language models," (Webb, Holyoak, and Lu, 2023) the authors argue that "large language models such as GPT-3 have acquired an emergent ability to find zero-shot solutions to a broad range of analogy problems." In this response, we provide counterexamples of the letter string analogies. In our tests, GPT-3 fails to solve even the easiest variants of the problems presented in the original paper. Zero-shot reasoning is an extraordinary claim that requires extraordinary evidence. We do not see that evidence in our experiments. To strengthen claims of humanlike reasoning such as zero-shot reasoning, it is important that the field develop approaches that rule out data memorization.
    
[^11]: GRASP: 一种高效的在线渐进式学习的重演策略

    GRASP: A Rehearsal Policy for Efficient Online Continual Learning. (arXiv:2308.13646v1 [cs.LG])

    [http://arxiv.org/abs/2308.13646](http://arxiv.org/abs/2308.13646)

    GRASP是一种新的样本选择策略，根据样本的代表性选择最适合学习的样本，从而提高了在线渐进式学习的效率。

    

    深度神经网络中的渐进学习涉及从不断增长的数据流中逐步累积知识。渐进学习的一个主要挑战是非平稳的数据流会导致之前学到的能力遭受灾难性遗忘。重演是一种常用且有效的缓解这个问题的方法，即将过去的观测结果存储在缓冲区中，并在学习过程中将它们与新的观测结果混合。这带来了一个问题：应该选择哪些存储样本进行重演？选择最适合学习的样本而不是随机选择样本，可能会导致学习速度显著加快。对于类增量学习，先前的研究表明简单的类均衡随机选择策略优于更复杂的方法。在这里，我们通过探索一种新的样本选择策略GRASP重新思考这个问题。GRASP首先选择最具代表性的样本，然后逐渐选择较不具代表性的样本。

    Continual learning (CL) in deep neural networks (DNNs) involves incrementally accumulating knowledge in a DNN from a growing data stream. A major challenge in CL is that non-stationary data streams cause catastrophic forgetting of previously learned abilities. Rehearsal is a popular and effective way to mitigate this problem, which is storing past observations in a buffer and mixing them with new observations during learning. This leads to a question: Which stored samples should be selected for rehearsal? Choosing samples that are best for learning, rather than simply selecting them at random, could lead to significantly faster learning. For class incremental learning, prior work has shown that a simple class balanced random selection policy outperforms more sophisticated methods. Here, we revisit this question by exploring a new sample selection policy called GRASP. GRASP selects the most prototypical (class representative) samples first and then gradually selects less prototypical (h
    
[^12]: Retroformer：使用策略梯度优化的回顾性大型语言代理

    Retroformer: Retrospective Large Language Agents with Policy Gradient Optimization. (arXiv:2308.02151v1 [cs.CL])

    [http://arxiv.org/abs/2308.02151](http://arxiv.org/abs/2308.02151)

    本文介绍了一种通过策略梯度优化的回顾性大型语言代理框架，该框架通过学习环境反馈来调整语言代理的提示，从而优化其性能。这种代理能够从多个环境和任务中学习奖励，并通过总结以前任务的根本原因来改进语言代理提示。

    

    最近几个月，出现了一个强大的新趋势，即将大型语言模型（LLMs）增强成能够自主完成目标导向多步骤任务的语言代理，而不仅仅是回答人类用户的查询。然而，大多数现有的语言代理没有使用环境特定的奖励进行优化。尽管一些代理通过口头反馈实现了迭代改进，但它们不能以与基于梯度的奖励学习相兼容的方式进行推理和规划。本文提出了一个原则性的框架，通过学习回顾模型，通过策略梯度自动调整语言代理的提示，从环境反馈中优化代理的工作。具体而言，我们提出的代理架构通过学习多个环境和任务的奖励来微调预训练语言模型，从而通过总结以前任务的根本原因来改进语言代理提示。

    Recent months have seen the emergence of a powerful new trend in which large language models (LLMs) are augmented to become autonomous language agents capable of performing objective oriented multi-step tasks on their own, rather than merely responding to queries from human users. Most existing language agents, however, are not optimized using environment-specific rewards. Although some agents enable iterative refinement through verbal feedback, they do not reason and plan in ways that are compatible with gradient-based learning from rewards. This paper introduces a principled framework for reinforcing large language agents by learning a retrospective model, which automatically tunes the language agent prompts from environment feedback through policy gradient. Specifically, our proposed agent architecture learns from rewards across multiple environments and tasks, for fine-tuning a pre-trained language model which refines the language agent prompt by summarizing the root cause of prior
    

