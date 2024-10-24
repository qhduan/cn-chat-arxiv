# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [TravelPlanner: A Benchmark for Real-World Planning with Language Agents](https://rss.arxiv.org/abs/2402.01622) | 本论文提出了一种用于自然语言代理的新的规划基准TravelPlanner，它关注于旅行规划这一常见的真实世界场景。经过全面评估，发现目前的语言代理仍无法处理如此复杂的规划任务，即使最先进的GPT-4也只能达到0.6%的成功率。 |
| [^2] | [Few-Shot Adversarial Prompt Learning on Vision-Language Models](https://arxiv.org/abs/2403.14774) | 本文提出了一个少样本对抗提示框架，在视觉-语言模型中通过有限数据调整输入序列，显著提升对抗鲁棒性，并通过端到端学习对抗性相关的文本监督。 |
| [^3] | [Interpreting Context Look-ups in Transformers: Investigating Attention-MLP Interactions](https://arxiv.org/abs/2402.15055) | 该研究探究了Transformer中注意力头和MLP之间的相互作用，并揭示了特定上下文下激活特定token预测的机制，从而阐明在LLMs中注意力如何促成依赖上下文的专门化处理。 |
| [^4] | [From Keywords to Structured Summaries: Streamlining Scholarly Knowledge Access](https://arxiv.org/abs/2402.14622) | 该论文突出了信息检索引擎在科学界的重要性，并提出了一种通过结构化记录和先进信息技术工具实现的解决方案，以革新研究人员访问和过滤文章的方式。 |
| [^5] | [Reinforcement Learning with Dynamic Multi-Reward Weighting for Multi-Style Controllable Generation](https://arxiv.org/abs/2402.14146) | 本文提出了一种使用强化学习来控制多种风格生成的方法，通过动态权重调整多重奖励，实现了在生成文本时同时控制多种风格。 |
| [^6] | [Learning to Poison Large Language Models During Instruction Tuning](https://arxiv.org/abs/2402.13459) | 通过设计新的数据注入攻击攻击LLMs，并提出一种梯度引导后门触发器学习方法，通过实验验证表明成功地破坏模型输出，仅改变1%的指导调优样本即可导致性能下降率达到约80％。 |
| [^7] | [Generative AI Security: Challenges and Countermeasures](https://arxiv.org/abs/2402.12617) | 生成式人工智能的安全挑战及对策研究。 |
| [^8] | [Jailbreaking Proprietary Large Language Models using Word Substitution Cipher](https://arxiv.org/abs/2402.10601) | 本文使用密码技术编码了越狱提示，成功地绕过了大型语言模型对有害问题的检测，实验结果显示攻击成功率高达59.42%。 |
| [^9] | [Reconfidencing LLMs from the Grouping Loss Perspective](https://arxiv.org/abs/2402.04957) | 本论文研究了大型语言模型的信心问题，发现现有的校准方法不足以解决由于分组损失导致的预测分数与实际概率偏离的问题。我们提出了一种解决方案，可以重新确定LLMs，改善它们的自信度。 |
| [^10] | [Let Me Teach You: Pedagogical Foundations of Feedback for Language Models.](http://arxiv.org/abs/2307.00279) | 这篇观点文章介绍了一个基于教育学理念的反馈框架FELT，用于对大型语言模型进行反馈，以提高模型与人类偏好的一致性。该框架不仅简化了现有的手工设计反馈，还为NLF研究开辟了新方向。 |
| [^11] | [GPT-SW3: An Autoregressive Language Model for the Nordic Languages.](http://arxiv.org/abs/2305.12987) | GPT-SW3是面向北欧语言的第一个本地化大型生成语言模型，本文介绍了其开发过程，可作为其他研究人员开发面向较小语言的大型生成模型的指南和参考。 |

# 详细

[^1]: TravelPlanner: 一种用于自然语言代理的真实世界规划基准

    TravelPlanner: A Benchmark for Real-World Planning with Language Agents

    [https://rss.arxiv.org/abs/2402.01622](https://rss.arxiv.org/abs/2402.01622)

    本论文提出了一种用于自然语言代理的新的规划基准TravelPlanner，它关注于旅行规划这一常见的真实世界场景。经过全面评估，发现目前的语言代理仍无法处理如此复杂的规划任务，即使最先进的GPT-4也只能达到0.6%的成功率。

    

    自规划起初就是人工智能的核心追求之一，但早期的人工智能代理大多集中在受限环境下，因为缺乏进行人类水平规划所需的许多认知基础。最近，由大型语言模型（LLM）驱动的语言代理展现出了工具使用和推理等有趣的能力。这些语言代理能否在超出先前人工智能代理范围的更复杂环境中进行规划？为了推进这项研究，我们提出了TravelPlanner，这是一个新的规划基准，专注于旅行规划这个常见的真实世界规划场景。它提供了一个丰富的沙盒环境，各种用于访问近400万个数据记录的工具，并包含1225个精心策划的规划意图和参考计划。综合评估显示，当前的语言代理尚不具备处理如此复杂的规划任务的能力-即使是GPT-4的成功率也只有0.6%。

    Planning has been part of the core pursuit for artificial intelligence since its conception, but earlier AI agents mostly focused on constrained settings because many of the cognitive substrates necessary for human-level planning have been lacking. Recently, language agents powered by large language models (LLMs) have shown interesting capabilities such as tool use and reasoning. Are these language agents capable of planning in more complex settings that are out of the reach of prior AI agents? To advance this investigation, we propose TravelPlanner, a new planning benchmark that focuses on travel planning, a common real-world planning scenario. It provides a rich sandbox environment, various tools for accessing nearly four million data records, and 1,225 meticulously curated planning intents and reference plans. Comprehensive evaluations show that the current language agents are not yet capable of handling such complex planning tasks-even GPT-4 only achieves a success rate of 0.6%. La
    
[^2]: 视觉-语言模型上的少样本对抗提示学习

    Few-Shot Adversarial Prompt Learning on Vision-Language Models

    [https://arxiv.org/abs/2403.14774](https://arxiv.org/abs/2403.14774)

    本文提出了一个少样本对抗提示框架，在视觉-语言模型中通过有限数据调整输入序列，显著提升对抗鲁棒性，并通过端到端学习对抗性相关的文本监督。

    

    深度神经网络对于微不可见的对抗性扰动的脆弱性已经引起了广泛关注。受到视觉-语言基础模型成功的启发，先前的努力通过将对抗性视觉特征与文本监督对齐来实现零样本对抗鲁棒性。但在实践中，由于包括重大适应成本、次优文本监督和未受控制的自然泛化能力在内的多个问题，它们仍然不尽人意。为了解决这些问题，本文提出了一个少样本对抗提示框架，通过有限的数据调整输入序列使得对抗鲁棒性得到显著提升。具体而言，我们通过提供对抗相关的文本监督，该监督是从对抗性示例中端到端学习的，来实现这一点。我们还提出了一个增强多模态特征一致性并鼓励不同

    arXiv:2403.14774v1 Announce Type: cross  Abstract: The vulnerability of deep neural networks to imperceptible adversarial perturbations has attracted widespread attention. Inspired by the success of vision-language foundation models, previous efforts achieved zero-shot adversarial robustness by aligning adversarial visual features with text supervision. However, in practice, they are still unsatisfactory due to several issues, including heavy adaptation cost, suboptimal text supervision, and uncontrolled natural generalization capacity. In this paper, to address these issues, we propose a few-shot adversarial prompt framework where adapting input sequences with limited data makes significant adversarial robustness improvement. Specifically, we achieve this by providing adversarially correlated text supervision that is end-to-end learned from adversarial examples. We also propose a novel training objective that enhances the consistency of multi-modal features while encourages differenti
    
[^3]: 在Transformer中解释上下文查找：探究注意力-MLP交互

    Interpreting Context Look-ups in Transformers: Investigating Attention-MLP Interactions

    [https://arxiv.org/abs/2402.15055](https://arxiv.org/abs/2402.15055)

    该研究探究了Transformer中注意力头和MLP之间的相互作用，并揭示了特定上下文下激活特定token预测的机制，从而阐明在LLMs中注意力如何促成依赖上下文的专门化处理。

    

    在本文中，我们研究了注意力头和Multilayer Perceptron中专门预测特定token的"next-token"神经元之间的相互作用。通过促使像GPT-4这样的LLM解释这些模型内部，我们可以阐明激活某些next-token神经元的注意力机制。我们的分析确定了识别与预测特定token相关的上下文的attention heads，通过残差连接激活相关联的神经元。我们专注于在较早的层中始终激活相同next-token神经元的attention heads。探索这些不同的激活模式揭示了为不同语言上下文专门化的头与生成某些tokens相关联。总体而言，我们的方法结合了神经解释和探测孤立的组件，以阐明注意力如何使LLMs中的依赖上下文的专门处理成为可能。

    arXiv:2402.15055v1 Announce Type: cross  Abstract: In this paper, we investigate the interplay between attention heads and specialized "next-token" neurons in the Multilayer Perceptron that predict specific tokens. By prompting an LLM like GPT-4 to explain these model internals, we can elucidate attention mechanisms that activate certain next-token neurons. Our analysis identifies attention heads that recognize contexts relevant to predicting a particular token, activating the associated neuron through the residual connection. We focus specifically on heads in earlier layers consistently activating the same next-token neuron across similar prompts. Exploring these differential activation patterns reveals that heads that specialize for distinct linguistic contexts are tied to generating certain tokens. Overall, our method combines neural explanations and probing isolated components to illuminate how attention enables context-dependent, specialized processing in LLMs.
    
[^4]: 从关键词到结构化摘要: 精简学术知识获取

    From Keywords to Structured Summaries: Streamlining Scholarly Knowledge Access

    [https://arxiv.org/abs/2402.14622](https://arxiv.org/abs/2402.14622)

    该论文突出了信息检索引擎在科学界的重要性，并提出了一种通过结构化记录和先进信息技术工具实现的解决方案，以革新研究人员访问和过滤文章的方式。

    

    这篇短文强调了信息检索引擎在科学界日益重要，指出传统基于关键词的搜索引擎由于出版物数量不断增加而效率低下。提出的解决方案涉及结构化记录，支持先进的信息技术工具，包括可视化仪表板，以彻底改变研究人员如何访问和过滤文章，取代传统的文本密集型方法。这一愿景通过一个以“传染病的繁殖数估计”研究主题为中心的概念验证得以体现，使用经过调整的大型语言模型(LLM)自动创建结构化记录以填充一个超越关键词的后端数据库。结果是一个下一代信息检索方法，可在https://orkg.org/usecases/r0-estimates 上访问。

    arXiv:2402.14622v1 Announce Type: cross  Abstract: This short paper highlights the growing importance of information retrieval (IR) engines in the scientific community, addressing the inefficiency of traditional keyword-based search engines due to the rising volume of publications. The proposed solution involves structured records, underpinning advanced information technology (IT) tools, including visualization dashboards, to revolutionize how researchers access and filter articles, replacing the traditional text-heavy approach. This vision is exemplified through a proof of concept centered on the ``reproductive number estimate of infectious diseases'' research theme, using a fine-tuned large language model (LLM) to automate the creation of structured records to populate a backend database that now goes beyond keywords. The result is a next-generation IR method accessible at https://orkg.org/usecases/r0-estimates.
    
[^5]: 使用动态多重奖励加权的强化学习用于多样式可控生成

    Reinforcement Learning with Dynamic Multi-Reward Weighting for Multi-Style Controllable Generation

    [https://arxiv.org/abs/2402.14146](https://arxiv.org/abs/2402.14146)

    本文提出了一种使用强化学习来控制多种风格生成的方法，通过动态权重调整多重奖励，实现了在生成文本时同时控制多种风格。

    

    风格是表达各种信息的文本中的一个组成部分，包括人际动态（例如正式性）和作者的情绪或态度（例如厌恶）。人类经常同时采用多种风格。一个待解决的问题是如何明确控制大型语言模型，使它们在生成文本时编织目标风格：例如，生成既消极又无毒的文本。先前的工作探讨了对单一风格的控制生成，或者对风格和其他属性的控制生成。在本文中，我们将这扩展到同时控制多种风格。具体而言，我们研究了用于受控多样式生成的强化学习（RL）方法的多种风格奖励的各种公式。这些奖励公式包括来自鉴别器的校准输出以及通过鉴别器梯度幅度进行动态加权。

    arXiv:2402.14146v1 Announce Type: new  Abstract: Style is an integral component of text that expresses a diverse set of information, including interpersonal dynamics (e.g. formality) and the author's emotions or attitudes (e.g. disgust). Humans often employ multiple styles simultaneously. An open question is how large language models can be explicitly controlled so that they weave together target styles when generating text: for example, to produce text that is both negative and non-toxic. Previous work investigates the controlled generation of a single style, or else controlled generation of a style and other attributes. In this paper, we expand this into controlling multiple styles simultaneously. Specifically, we investigate various formulations of multiple style rewards for a reinforcement learning (RL) approach to controlled multi-style generation. These reward formulations include calibrated outputs from discriminators and dynamic weighting by discriminator gradient magnitudes. W
    
[^6]: 学习在指导调优期间操纵大型语言模型

    Learning to Poison Large Language Models During Instruction Tuning

    [https://arxiv.org/abs/2402.13459](https://arxiv.org/abs/2402.13459)

    通过设计新的数据注入攻击攻击LLMs，并提出一种梯度引导后门触发器学习方法，通过实验验证表明成功地破坏模型输出，仅改变1%的指导调优样本即可导致性能下降率达到约80％。

    

    大型语言模型（LLMs）的出现标志着语言处理和推理能力方面的重大突破。虽然它们取得了显著进展，但LLMs面临着数据注入攻击的漏洞，其中对手将后门触发器插入训练数据，以操纵输出以进行恶意行为。本研究通过设计一种新的数据注入攻击，旨在利用指导调优过程，进一步识别LLMs中的额外安全风险。我们提出了一种新颖的梯度引导后门触发器学习方法，以有效识别敌对触发器，确保对传统防御手段的规避，同时保持内容的完整性。通过对各种LLMs和任务的实验验证，我们的策略表明在破坏模型输出方面取得了很高的成功率；仅对4,000个指导调优样本中的1％进行注入就导致性能降低率（PDR）约为80％。我们的工作高

    arXiv:2402.13459v1 Announce Type: cross  Abstract: The advent of Large Language Models (LLMs) has marked significant achievements in language processing and reasoning capabilities. Despite their advancements, LLMs face vulnerabilities to data poisoning attacks, where adversaries insert backdoor triggers into training data to manipulate outputs for malicious purposes. This work further identifies additional security risks in LLMs by designing a new data poisoning attack tailored to exploit the instruction tuning process. We propose a novel gradient-guided backdoor trigger learning approach to identify adversarial triggers efficiently, ensuring an evasion of detection by conventional defenses while maintaining content integrity. Through experimental validation across various LLMs and tasks, our strategy demonstrates a high success rate in compromising model outputs; poisoning only 1\% of 4,000 instruction tuning samples leads to a Performance Drop Rate (PDR) of around 80\%. Our work high
    
[^7]: 生成式人工智能安全：挑战与对策

    Generative AI Security: Challenges and Countermeasures

    [https://arxiv.org/abs/2402.12617](https://arxiv.org/abs/2402.12617)

    生成式人工智能的安全挑战及对策研究。

    

    arXiv:2402.12617v1 公告类型：跨领域 摘要：生成式人工智能在许多行业的不断扩展引发了人们的兴奋和增加的关注。本文深入探讨了生成式人工智能所带来的独特安全挑战，并概述了管理这些风险的潜在研究方向。

    arXiv:2402.12617v1 Announce Type: cross  Abstract: Generative AI's expanding footprint across numerous industries has led to both excitement and increased scrutiny. This paper delves into the unique security challenges posed by Generative AI, and outlines potential research directions for managing these risks.
    
[^8]: 使用单词替换密码来越狱专有的大型语言模型

    Jailbreaking Proprietary Large Language Models using Word Substitution Cipher

    [https://arxiv.org/abs/2402.10601](https://arxiv.org/abs/2402.10601)

    本文使用密码技术编码了越狱提示，成功地绕过了大型语言模型对有害问题的检测，实验结果显示攻击成功率高达59.42%。

    

    大型语言模型（LLMs）遵循道德和伦理准则，但仍然容易受到名为Jailbreak的创意提示的影响，这些提示可以绕过对齐过程。然而，大多数越狱提示包含自然语言（主要是英语）中的有害问题，可以被LLMs自身检测到。本文提出了使用密码技术编码的越狱提示。我们首先在最先进的LLM，GPT-4上进行了一个试点研究，解码了使用各种密码技术加密的几个安全句子，发现简单的单词替换密码可以被最有效地解码。受此结果启发，我们使用这种编码技术来编写越狱提示。我们提供了将不安全单词映射到安全单词，并使用这些映射的单词提出不安全问题的映射。实验结果显示，我们提出的越狱攻击成功率（高达59.42%）。

    arXiv:2402.10601v1 Announce Type: cross  Abstract: Large Language Models (LLMs) are aligned to moral and ethical guidelines but remain susceptible to creative prompts called Jailbreak that can bypass the alignment process. However, most jailbreaking prompts contain harmful questions in the natural language (mainly English), which can be detected by the LLM themselves. In this paper, we present jailbreaking prompts encoded using cryptographic techniques. We first present a pilot study on the state-of-the-art LLM, GPT-4, in decoding several safe sentences that have been encrypted using various cryptographic techniques and find that a straightforward word substitution cipher can be decoded most effectively. Motivated by this result, we use this encoding technique for writing jailbreaking prompts. We present a mapping of unsafe words with safe words and ask the unsafe question using these mapped words. Experimental results show an attack success rate (up to 59.42%) of our proposed jailbrea
    
[^9]: 从分组损失的角度重构大型语言模型的信心

    Reconfidencing LLMs from the Grouping Loss Perspective

    [https://arxiv.org/abs/2402.04957](https://arxiv.org/abs/2402.04957)

    本论文研究了大型语言模型的信心问题，发现现有的校准方法不足以解决由于分组损失导致的预测分数与实际概率偏离的问题。我们提出了一种解决方案，可以重新确定LLMs，改善它们的自信度。

    

    大型语言模型（LLMs），包括ChatGPT和LLaMA，在自信的口吻中容易生成虚假答案。尽管引导和校准信心分数的努力已被证明是有用的，但最近的研究发现，控制不确定性必须超越校准: 由于分组损失的影响，预测分数可能明显偏离实际的后验概率。在这项工作中，我们构建了一个新的评估数据集，从知识库中获取，以评估对Mistral和LLaMA的答案给出的信心分数。实验表明，它们倾向于过于自信。此外，我们还展示了它们在某些答案上比其他答案更过于自信，例如取决于查询中人的国籍。在不确定性量化理论中，这就是分组损失。为了解决这个问题，我们提出了一种重新确定LLMs的解决方案，不仅取消校准，还取消分组损失。经过重新确定的LLMs经过处理后，表示改进的自信度。

    Large Language Models (LLMs), including ChatGPT and LLaMA, are susceptible to generating hallucinated answers in a confident tone. While efforts to elicit and calibrate confidence scores have proven useful, recent findings show that controlling uncertainty must go beyond calibration: predicted scores may deviate significantly from the actual posterior probabilities due to the impact of grouping loss. In this work, we construct a new evaluation dataset derived from a knowledge base to assess confidence scores given to answers of Mistral and LLaMA. Experiments show that they tend to be overconfident. Further, we show that they are more overconfident on some answers than others, \emph{eg} depending on the nationality of the person in the query. In uncertainty-quantification theory, this is grouping loss. To address this, we propose a solution to reconfidence LLMs, canceling not only calibration but also grouping loss. The LLMs, after the reconfidencing process, indicate improved confidenc
    
[^10]: 让我来教你：语言模型的反馈教育基础

    Let Me Teach You: Pedagogical Foundations of Feedback for Language Models. (arXiv:2307.00279v1 [cs.CL])

    [http://arxiv.org/abs/2307.00279](http://arxiv.org/abs/2307.00279)

    这篇观点文章介绍了一个基于教育学理念的反馈框架FELT，用于对大型语言模型进行反馈，以提高模型与人类偏好的一致性。该框架不仅简化了现有的手工设计反馈，还为NLF研究开辟了新方向。

    

    自然语言反馈（NLF）是将大型语言模型（LLMs）与人类偏好对齐的一个越来越受欢迎的途径。尽管NLF可以传达丰富多样的信息，但往往是手工设计的和随意的。在不同的世界中，教育学研究长期以来建立了几种有效的反馈模型。在这篇观点文章中，我们汇编了来自教育学的思想，引入了一种名为FELT的LLMs反馈框架，概述了反馈空间的各种特征以及基于这些变量的反馈内容分类法。我们的分类法不仅提供了对反馈空间的一般映射，还提供了教育学确定的离散类别，使我们能够从经验上证明不同反馈类型对修订生成的影响。除了简化现有的NLF设计，FELT还为NLF研究带来了新的未开发的方向。我们将我们的分类法提供给社区，为映射我们的类别提供指南和示例。

    Natural Language Feedback (NLF) is an increasingly popular avenue to align Large Language Models (LLMs) to human preferences. Despite the richness and diversity of the information it can convey, NLF is often hand-designed and arbitrary. In a different world, research in pedagogy has long established several effective feedback models. In this opinion piece, we compile ideas from pedagogy to introduce FELT, a feedback framework for LLMs that outlines the various characteristics of the feedback space, and a feedback content taxonomy based on these variables. Our taxonomy offers both a general mapping of the feedback space, as well as pedagogy-established discrete categories, allowing us to empirically demonstrate the impact of different feedback types on revised generations. In addition to streamlining existing NLF designs, FELT also brings out new, unexplored directions for research in NLF. We make our taxonomy available to the community, providing guides and examples for mapping our cat
    
[^11]: GPT-SW3：一种面向北欧语言的自回归语言模型

    GPT-SW3: An Autoregressive Language Model for the Nordic Languages. (arXiv:2305.12987v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2305.12987](http://arxiv.org/abs/2305.12987)

    GPT-SW3是面向北欧语言的第一个本地化大型生成语言模型，本文介绍了其开发过程，可作为其他研究人员开发面向较小语言的大型生成模型的指南和参考。

    

    本文详细介绍了开发面向北欧语言的第一个本地化大型生成语言模型GPT-SW3的过程。我们涵盖了开发过程的所有部分，从数据收集和处理，训练配置和指令微调，到评估和发布策略的考虑。我们希望本文能够作为指南和参考，帮助其他研究人员开发面向较小语言的大型生成模型。

    This paper details the process of developing the first native large generative language model for the Nordic languages, GPT-SW3. We cover all parts of the development process, from data collection and processing, training configuration and instruction finetuning, to evaluation and considerations for release strategies. We hope that this paper can serve as a guide and reference for other researchers that undertake the development of large generative models for smaller languages.
    

