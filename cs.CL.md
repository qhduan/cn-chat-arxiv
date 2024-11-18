# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Security and Privacy Challenges of Large Language Models: A Survey](https://rss.arxiv.org/abs/2402.00888) | 大型语言模型具有卓越的能力，但也面临着安全和隐私攻击的威胁。本调查全面审查了LLM的安全和隐私挑战，涵盖了训练数据、用户和应用风险等方面，并对解决方法进行了回顾。 |
| [^2] | [Mitigating the Linguistic Gap with Phonemic Representations for Robust Multilingual Language Understanding](https://arxiv.org/abs/2402.14279) | 通过使用音素表示，本文提出了一种新颖的解决方案来减缓高资源语言和低资源语言之间的性能差距，并通过实证研究和理论分析证明了其有效性。 |
| [^3] | [REBORN: Reinforcement-Learned Boundary Segmentation with Iterative Training for Unsupervised ASR](https://arxiv.org/abs/2402.03988) | 本文提出了REBORN，在无监督语音识别中使用基于强化学习的迭代训练来实现边界分割。通过交替训练分割模型和音素预测模型，实现了学习语音和文本之间的映射，解决了无监督情况下语音信号分段结构边界的挑战。 |
| [^4] | [Evaluating and Enhancing Large Language Models for Conversational Reasoning on Knowledge Graphs](https://arxiv.org/abs/2312.11282) | 该论文评估了当前最先进的大型语言模型（GPT-4）在知识图谱上的对话推理能力，提出了一种基于KG推理的LLM基准代理（LLM-ARK），该代理利用全文环境提示来实现精确和适应性强的KG路径预测，并采用近端策略优化算法进行训练。 |

# 详细

[^1]: 大型语言模型的安全和隐私挑战：一项调查

    Security and Privacy Challenges of Large Language Models: A Survey

    [https://rss.arxiv.org/abs/2402.00888](https://rss.arxiv.org/abs/2402.00888)

    大型语言模型具有卓越的能力，但也面临着安全和隐私攻击的威胁。本调查全面审查了LLM的安全和隐私挑战，涵盖了训练数据、用户和应用风险等方面，并对解决方法进行了回顾。

    

    大型语言模型（LLM）展示了非凡的能力，并在生成和总结文本、语言翻译和问答等多个领域做出了贡献。如今，LLM正在成为计算机语言处理任务中非常流行的工具，具备分析复杂语言模式并根据上下文提供相关和适当回答的能力。然而，尽管具有显著优势，这些模型也容易受到安全和隐私攻击的威胁，如越狱攻击、数据污染攻击和个人可识别信息泄露攻击。本调查全面审查了LLM的安全和隐私挑战，包括训练数据和用户方面的问题，以及在交通、教育和医疗等各个领域中应用带来的风险。我们评估了LLM的脆弱性程度，调查了出现的安全和隐私攻击，并对潜在的解决方法进行了回顾。

    Large Language Models (LLMs) have demonstrated extraordinary capabilities and contributed to multiple fields, such as generating and summarizing text, language translation, and question-answering. Nowadays, LLM is becoming a very popular tool in computerized language processing tasks, with the capability to analyze complicated linguistic patterns and provide relevant and appropriate responses depending on the context. While offering significant advantages, these models are also vulnerable to security and privacy attacks, such as jailbreaking attacks, data poisoning attacks, and Personally Identifiable Information (PII) leakage attacks. This survey provides a thorough review of the security and privacy challenges of LLMs for both training data and users, along with the application-based risks in various domains, such as transportation, education, and healthcare. We assess the extent of LLM vulnerabilities, investigate emerging security and privacy attacks for LLMs, and review the potent
    
[^2]: 使用音素表示减缓语言差异，实现稳健的多语言理解

    Mitigating the Linguistic Gap with Phonemic Representations for Robust Multilingual Language Understanding

    [https://arxiv.org/abs/2402.14279](https://arxiv.org/abs/2402.14279)

    通过使用音素表示，本文提出了一种新颖的解决方案来减缓高资源语言和低资源语言之间的性能差距，并通过实证研究和理论分析证明了其有效性。

    

    为了改善多语言理解，通常需要在训练阶段使用多种语言，依赖复杂的训练技术，并且在高资源语言和低资源语言之间存在显著的性能差距。我们假设语言之间的性能差距受到这些语言之间的语言差异的影响，并通过使用音素表示（具体来说，将音素作为输入标记输入到语言模型中，而不是子词）提供了一种新颖的解决方案，以实现稳健的多语言建模。我们通过三个跨语言任务的定量证据展示了音素表示的有效性，这进一步得到了对跨语言性能差距的理论分析的证明。

    arXiv:2402.14279v1 Announce Type: cross  Abstract: Approaches to improving multilingual language understanding often require multiple languages during the training phase, rely on complicated training techniques, and -- importantly -- struggle with significant performance gaps between high-resource and low-resource languages. We hypothesize that the performance gaps between languages are affected by linguistic gaps between those languages and provide a novel solution for robust multilingual language modeling by employing phonemic representations (specifically, using phonemes as input tokens to LMs rather than subwords). We present quantitative evidence from three cross-lingual tasks that demonstrate the effectiveness of phonemic representation, which is further justified by a theoretical analysis of the cross-lingual performance gap.
    
[^3]: REBORN: 基于强化学习的迭代训练的无监督语音识别中的边界分割

    REBORN: Reinforcement-Learned Boundary Segmentation with Iterative Training for Unsupervised ASR

    [https://arxiv.org/abs/2402.03988](https://arxiv.org/abs/2402.03988)

    本文提出了REBORN，在无监督语音识别中使用基于强化学习的迭代训练来实现边界分割。通过交替训练分割模型和音素预测模型，实现了学习语音和文本之间的映射，解决了无监督情况下语音信号分段结构边界的挑战。

    

    无监督自动语音识别（ASR）旨在学习语音信号与其对应的文本转录之间的映射，而无需配对的语音-文本数据监督。语音信号中的单词/音素由一段长度可变且边界未知的语音信号表示，而这种分段结构使得在没有配对数据的情况下学习语音和文本之间的映射变得具有挑战性。本文提出了REBORN，基于强化学习的迭代训练的无监督语音识别中的边界分割。REBORN交替进行以下两个步骤：（1）训练一个能够预测语音信号中分段结构边界的分割模型，和（2）训练一个音素预测模型，其输入是由分割模型分割的分段结构，用于预测音素转录。由于没有用于训练分割模型的监督数据，我们使用强化学习来训练分割模型。

    Unsupervised automatic speech recognition (ASR) aims to learn the mapping between the speech signal and its corresponding textual transcription without the supervision of paired speech-text data. A word/phoneme in the speech signal is represented by a segment of speech signal with variable length and unknown boundary, and this segmental structure makes learning the mapping between speech and text challenging, especially without paired data. In this paper, we propose REBORN, Reinforcement-Learned Boundary Segmentation with Iterative Training for Unsupervised ASR. REBORN alternates between (1) training a segmentation model that predicts the boundaries of the segmental structures in speech signals and (2) training the phoneme prediction model, whose input is a segmental structure segmented by the segmentation model, to predict a phoneme transcription. Since supervised data for training the segmentation model is not available, we use reinforcement learning to train the segmentation model t
    
[^4]: 评估和增强用于知识图谱上的对话推理的大型语言模型

    Evaluating and Enhancing Large Language Models for Conversational Reasoning on Knowledge Graphs

    [https://arxiv.org/abs/2312.11282](https://arxiv.org/abs/2312.11282)

    该论文评估了当前最先进的大型语言模型（GPT-4）在知识图谱上的对话推理能力，提出了一种基于KG推理的LLM基准代理（LLM-ARK），该代理利用全文环境提示来实现精确和适应性强的KG路径预测，并采用近端策略优化算法进行训练。

    

    大型语言模型（LLM）的发展得益于预训练技术的进展。通过手动设计的提示，这些模型展示了强大的推理能力。在这项工作中，我们评估了当前最先进的LLM（GPT-4）在知识图谱（KG）上的对话推理能力。然而，由于缺乏KG环境意识和开发有效的中间推理阶段优化机制的困难，LLM的性能受到限制。我们进一步引入了LLM-ARK，一个基于KG推理的LLM基准代理，旨在提供精确和适应性强的KG路径预测。LLM-ARK利用全文环境（FTE）提示来吸收每个推理步骤中的状态信息。我们将KG上的多跳推理挑战重新框定为顺序决策任务。利用近端策略优化（PPO）在线策略梯度强化学习算法，我们的模型...

    The development of large language models (LLMs) has been catalyzed by advancements in pre-training techniques. These models have demonstrated robust reasoning capabilities through manually designed prompts. In this work, we evaluate the conversational reasoning capabilities of the current state-of-the-art LLM (GPT-4) on knowledge graphs (KGs). However, the performance of LLMs is constrained due to a lack of KG environment awareness and the difficulties in developing effective optimization mechanisms for intermediary reasoning stages. We further introduce LLM-ARK, a LLM grounded KG reasoning agent designed to deliver precise and adaptable predictions on KG paths. LLM-ARK leverages Full Textual Environment (FTE) prompt to assimilate state information within each reasoning step. We reframe the challenge of multi-hop reasoning on the KG as a sequential decision-making task. Utilizing the Proximal Policy Optimization (PPO) online policy gradient reinforcement learning algorithm, our model i
    

