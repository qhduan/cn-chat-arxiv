# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [GET: Unlocking the Multi-modal Potential of CLIP for Generalized Category Discovery](https://arxiv.org/abs/2403.09974) | 本文提出了一种文本嵌入合成器（TES），用于为无标签数据生成伪文本嵌入，以解锁CLIP用于广义类别发现任务中的多模态潜力。 |
| [^2] | [Large Language Models and Causal Inference in Collaboration: A Comprehensive Survey](https://arxiv.org/abs/2403.09606) | 大型语言模型的出现极大影响了自然语言处理领域，特别是通过其先进的推理能力。而本综述则重点评估和改进了大型语言模型在因果推断方面的应用，包括提高推理能力、解决公平和安全问题、提供解释和处理多模态。 |
| [^3] | [ContextGPT: Infusing LLMs Knowledge into Neuro-Symbolic Activity Recognition Models](https://arxiv.org/abs/2403.06586) | 将预训练的大型语言模型（LLMs）的常识知识有效地注入神经符号活动识别模型，以缓解标记数据稀缺性问题。 |
| [^4] | [Analysis and Fully Memristor-based Reservoir Computing for Temporal Data Classification](https://arxiv.org/abs/2403.01827) | 本研究引入了一种新颖的双存储器 RC 系统，通过集成不同类型的 memristor，并在处理时间数据分类任务中取得了显著成效。 |
| [^5] | [On the Challenges and Opportunities in Generative AI](https://arxiv.org/abs/2403.00025) | 现代生成人工智能范例中存在关键的未解决挑战，如何解决这些挑战将进一步增强它们的能力、多功能性和可靠性，并为研究方向提供有价值的见解。 |
| [^6] | [SPHINX-X: Scaling Data and Parameters for a Family of Multi-modal Large Language Models](https://arxiv.org/abs/2402.05935) | 本论文介绍了SPHINX-X，一种扩展的多模态大型语言模型系列。通过改进架构和训练效率，我们成功构建了一系列参数大小和多语言能力不同的MLLMs，与数据和参数规模有强相关性。 |
| [^7] | [PromptCrypt: Prompt Encryption for Secure Communication with Large Language Models](https://arxiv.org/abs/2402.05868) | PromptCrypt是一种使用表情符号对用户输入进行加密的机制，保护了大型语言模型（LLM）中用户的隐私，防止数据泄露和解密。 |
| [^8] | [LitLLM: A Toolkit for Scientific Literature Review](https://arxiv.org/abs/2402.01788) | "LitLLM: A Toolkit for Scientific Literature Review" 提出了一个基于 RAG 原则的工具包，通过使用专门的提示和指导技术，结合大型语言模型（LLM），实现了科学文献综述的自动化。这个工具包不仅可以通过转化摘要为关键词进行文献检索，还可以通过补充相关论文或关键词进行定制化的检索。 |
| [^9] | [Zero-Shot Reinforcement Learning via Function Encoders](https://arxiv.org/abs/2401.17173) | 本论文提出了一种用于实现零-shot迁移的函数编码器，通过将函数表示为学习到的非线性基函数的加权组合，代理程序通过连贯的向量表示了当前任务与先前看到任务的关联信息，从而实现了在相关任务之间的迁移，无需额外训练。 |
| [^10] | [Generating Likely Counterfactuals Using Sum-Product Networks.](http://arxiv.org/abs/2401.14086) | 由于用户需求和最近的法规要求，需要对AI系统所做出的决策进行解释。本论文提出了一种使用Sum-Product Networks模拟寻找高可能性反事实推理的系统，该系统能够提供满足多个常见要求的最佳解释。 |
| [^11] | [Bias Assessment and Mitigation in LLM-based Code Generation.](http://arxiv.org/abs/2309.14345) | 这项研究提出了一个新颖的偏差评估框架，针对代码生成任务进行设计。通过对九个最先进的基于LLM的代码生成模型进行广泛评估，发现其中31.45\%到79.93\%的代码函数具有偏见，并提出了如何缓解这种偏见的方法。 |
| [^12] | [Summaries, Highlights, and Action items: Design, implementation and evaluation of an LLM-powered meeting recap system.](http://arxiv.org/abs/2307.15793) | 这项研究设计、实现和评估了一种基于LLM的会议总结系统，通过减少个人会议负担和增加会议输出的清晰度和一致性，提高了会议体验。 |
| [^13] | [Prioritized Trajectory Replay: A Replay Memory for Data-driven Reinforcement Learning.](http://arxiv.org/abs/2306.15503) | 本研究提出了一种名为优先轨迹回放的回放记忆方法，将数据采样的视角扩展到轨迹中，从有限的数据中提取更全面的信息。这种方法通过反向采样轨迹来提高学习效率，并利用加权评论目标避免采样未见过的动作。优先轨迹回放还能根据不同的优先度指标优先采样效率更高的轨迹。 |

# 详细

[^1]: GET：解锁CLIP的多模态潜力，用于广义类别发现

    GET: Unlocking the Multi-modal Potential of CLIP for Generalized Category Discovery

    [https://arxiv.org/abs/2403.09974](https://arxiv.org/abs/2403.09974)

    本文提出了一种文本嵌入合成器（TES），用于为无标签数据生成伪文本嵌入，以解锁CLIP用于广义类别发现任务中的多模态潜力。

    

    给定包含旧类别和新类别的无标签数据集，广义类别发现（GCD）旨在准确发现新类别，并正确分类旧类别，利用从有标签样本中学习的类别概念。当前的GCD方法只使用单一的视觉信息模态，导致在视觉上相似类别的分类效果不佳。虽然某些类别在视觉上容易混淆，但它们的文本信息可能是不同的，这促使我们将文本信息引入到GCD任务中。然而，无标签数据缺乏类别名称，使得利用文本信息变得不切实际。为了解决这一具有挑战性的问题，在本文中，我们提出了一种文本嵌入合成器（TES），用于为无标签样本生成伪文本嵌入。具体而言，我们的TES利用CLIP可以生成对齐的视觉-语言特征这一特性，将视觉嵌入转换为CLIP文本模型的标记。

    arXiv:2403.09974v1 Announce Type: cross  Abstract: Given unlabelled datasets containing both old and new categories, generalized category discovery (GCD) aims to accurately discover new classes while correctly classifying old classes, leveraging the class concepts learned from labeled samples. Current GCD methods only use a single visual modality of information, resulting in poor classification of visually similar classes. Though certain classes are visually confused, their text information might be distinct, motivating us to introduce text information into the GCD task. However, the lack of class names for unlabelled data makes it impractical to utilize text information. To tackle this challenging problem, in this paper, we propose a Text Embedding Synthesizer (TES) to generate pseudo text embeddings for unlabelled samples. Specifically, our TES leverages the property that CLIP can generate aligned vision-language features, converting visual embeddings into tokens of the CLIP's text e
    
[^2]: 大型语言模型与协作中的因果推断：一项综合调查

    Large Language Models and Causal Inference in Collaboration: A Comprehensive Survey

    [https://arxiv.org/abs/2403.09606](https://arxiv.org/abs/2403.09606)

    大型语言模型的出现极大影响了自然语言处理领域，特别是通过其先进的推理能力。而本综述则重点评估和改进了大型语言模型在因果推断方面的应用，包括提高推理能力、解决公平和安全问题、提供解释和处理多模态。

    

    因果推断已经显示出潜力，通过捕捉变量之间的因果关系，提高自然语言处理（NLP）模型的预测准确性、公平性、稳健性和可解释性。生成型大型语言模型（LLMs）的出现显著影响了各种NLP领域，特别是通过其先进的推理能力。该调查重点评估和改进LLMs的因果视角，在以下领域展开：理解和改进LLMs的推理能力，解决LLMs中的公平性和安全性问题，为LLMs提供解释，并处理多模态。同时，LLMs强大的推理能力反过来可以通过帮助因果关系发现和因果效应估计来促进因果推断领域的发展。本综述探讨了因果推断框架与LLMs之间的相互作用，强调了它们的集体作用。

    arXiv:2403.09606v1 Announce Type: cross  Abstract: Causal inference has shown potential in enhancing the predictive accuracy, fairness, robustness, and explainability of Natural Language Processing (NLP) models by capturing causal relationships among variables. The emergence of generative Large Language Models (LLMs) has significantly impacted various NLP domains, particularly through their advanced reasoning capabilities. This survey focuses on evaluating and improving LLMs from a causal view in the following areas: understanding and improving the LLMs' reasoning capacity, addressing fairness and safety issues in LLMs, complementing LLMs with explanations, and handling multimodality. Meanwhile, LLMs' strong reasoning capacities can in turn contribute to the field of causal inference by aiding causal relationship discovery and causal effect estimations. This review explores the interplay between causal inference frameworks and LLMs from both perspectives, emphasizing their collective p
    
[^3]: ContextGPT: 将LLMs知识注入神经符号活动识别模型

    ContextGPT: Infusing LLMs Knowledge into Neuro-Symbolic Activity Recognition Models

    [https://arxiv.org/abs/2403.06586](https://arxiv.org/abs/2403.06586)

    将预训练的大型语言模型（LLMs）的常识知识有效地注入神经符号活动识别模型，以缓解标记数据稀缺性问题。

    

    上下文感知人类活动识别（HAR）是移动计算中一个热门的研究领域，文献中最有效的解决方案基于监督式深度学习模型。然而，这些系统的实际部署受到需要用于训练的标记数据的稀缺性的限制。神经符号人工智能（NeSy）为缓解这一问题提供了一个有趣的研究方向，即将关于人类活动及其可能发生的背景的常识知识注入HAR深度学习分类器中。现有的用于上下文感知HAR的NeSy方法依赖于逻辑模型中编码的知识（例如本体论），其设计、实施和维护以捕捉新活动和上下文需要显著的人力工程努力、技术知识和领域专业知识。最近的研究表明，预训练的大型语言模型（LLMs）有效地编码了关于人类活动的常识知识。

    arXiv:2403.06586v1 Announce Type: cross  Abstract: Context-aware Human Activity Recognition (HAR) is a hot research area in mobile computing, and the most effective solutions in the literature are based on supervised deep learning models. However, the actual deployment of these systems is limited by the scarcity of labeled data that is required for training. Neuro-Symbolic AI (NeSy) provides an interesting research direction to mitigate this issue, by infusing common-sense knowledge about human activities and the contexts in which they can be performed into HAR deep learning classifiers. Existing NeSy methods for context-aware HAR rely on knowledge encoded in logic-based models (e.g., ontologies) whose design, implementation, and maintenance to capture new activities and contexts require significant human engineering efforts, technical knowledge, and domain expertise. Recent works show that pre-trained Large Language Models (LLMs) effectively encode common-sense knowledge about human a
    
[^4]: 分析和基于全 memristor 的时间数据分类的储层计算

    Analysis and Fully Memristor-based Reservoir Computing for Temporal Data Classification

    [https://arxiv.org/abs/2403.01827](https://arxiv.org/abs/2403.01827)

    本研究引入了一种新颖的双存储器 RC 系统，通过集成不同类型的 memristor，并在处理时间数据分类任务中取得了显著成效。

    

    arXiv:2403.01827v1 公告类型: 交叉摘要: 储层计算（RC）提供了一个特别适用于处理时空信号的神经形态学框架。RC以其时间处理能力而闻名，与传统的递归神经网络相比，显著降低了训练成本。其硬件部署中的一个关键组成部分是能够生成动态储层状态的能力。我们的研究引入了一种新颖的双重存储器 RC 系统，集成了一种基于 WOx 的 memristor 的短期存储器，能够实现 16 个不同状态的编码超过 4 个比特，并在读出层中使用 TiOx-based memristor 的长期存储器组件。我们彻底研究了两种 memristor 类型，并利用 RC 系统处理时间数据集。所提出的 RC 系统的性能通过两个基准任务进行了验证: 对具有不完整输入的孤立口述数字识别和 Mackey-Glass 时间序列预测。该系统提供了令人印象深刻的 98.84% 准确率.

    arXiv:2403.01827v1 Announce Type: cross  Abstract: Reservoir computing (RC) offers a neuromorphic framework that is particularly effective for processing spatiotemporal signals. Known for its temporal processing prowess, RC significantly lowers training costs compared to conventional recurrent neural networks. A key component in its hardware deployment is the ability to generate dynamic reservoir states. Our research introduces a novel dual-memory RC system, integrating a short-term memory via a WOx-based memristor, capable of achieving 16 distinct states encoded over 4 bits, and a long-term memory component using a TiOx-based memristor within the readout layer. We thoroughly examine both memristor types and leverage the RC system to process temporal data sets. The performance of the proposed RC system is validated through two benchmark tasks: isolated spoken digit recognition with incomplete inputs and Mackey-Glass time series prediction. The system delivered an impressive 98.84% accu
    
[^5]: 关于生成人工智能中的挑战与机遇

    On the Challenges and Opportunities in Generative AI

    [https://arxiv.org/abs/2403.00025](https://arxiv.org/abs/2403.00025)

    现代生成人工智能范例中存在关键的未解决挑战，如何解决这些挑战将进一步增强它们的能力、多功能性和可靠性，并为研究方向提供有价值的见解。

    

    深度生成建模领域近年来增长迅速而稳定。随着海量训练数据的可用性以及可扩展的无监督学习范式的进步，最近的大规模生成模型展现出合成高分辨率图像和文本以及结构化数据（如视频和分子）的巨大潜力。然而，我们认为当前大规模生成人工智能模型没有充分解决若干基本问题，限制了它们在各个领域的广泛应用。在本工作中，我们旨在确定现代生成人工智能范例中的关键未解决挑战，以进一步增强它们的能力、多功能性和可靠性。通过识别这些挑战，我们旨在为研究人员提供有价值的见解，探索有益的研究方向，从而促进更加强大和可访问的生成人工智能的发展。

    arXiv:2403.00025v1 Announce Type: cross  Abstract: The field of deep generative modeling has grown rapidly and consistently over the years. With the availability of massive amounts of training data coupled with advances in scalable unsupervised learning paradigms, recent large-scale generative models show tremendous promise in synthesizing high-resolution images and text, as well as structured data such as videos and molecules. However, we argue that current large-scale generative AI models do not sufficiently address several fundamental issues that hinder their widespread adoption across domains. In this work, we aim to identify key unresolved challenges in modern generative AI paradigms that should be tackled to further enhance their capabilities, versatility, and reliability. By identifying these challenges, we aim to provide researchers with valuable insights for exploring fruitful research directions, thereby fostering the development of more robust and accessible generative AI so
    
[^6]: SPHINX-X: 扩展数据和参数用于一系列多模态大型语言模型

    SPHINX-X: Scaling Data and Parameters for a Family of Multi-modal Large Language Models

    [https://arxiv.org/abs/2402.05935](https://arxiv.org/abs/2402.05935)

    本论文介绍了SPHINX-X，一种扩展的多模态大型语言模型系列。通过改进架构和训练效率，我们成功构建了一系列参数大小和多语言能力不同的MLLMs，与数据和参数规模有强相关性。

    

    我们提出SPHINX-X，一种基于SPHINX开发的广泛多模态大型语言模型（MLLM）系列。为了改善架构和训练效率，我们通过移除冗余的视觉编码器、绕过完全填充的子图像，并将多阶段训练简化成为一阶段的全集合模式，修改了SPHINX框架。为了充分发挥MLLM的潜力，我们组装了一个综合的跨语言、跨视觉和视觉-语言任务的多领域、多模态的数据集，涵盖了公开可用的资源。我们进一步使用我们的OCR密集和Mark数据集丰富这个收集，扩展了多样性和普适性。通过对不同基础LLM进行训练，包括TinyLlama1.1B、InternLM2-7B、LLaMA2-13B和Mixtral8x7B，我们获得了一系列参数大小和多语言能力变化的MLLMs。全面的基准测试揭示了多模态性能与数据和参数规模之间的强相关性。

    We propose SPHINX-X, an extensive Multimodality Large Language Model (MLLM) series developed upon SPHINX. To improve the architecture and training efficiency, we modify the SPHINX framework by removing redundant visual encoders, bypassing fully-padded sub-images with skip tokens, and simplifying multi-stage training into a one-stage all-in-one paradigm. To fully unleash the potential of MLLMs, we assemble a comprehensive multi-domain and multimodal dataset covering publicly available resources in language, vision, and vision-language tasks. We further enrich this collection with our curated OCR intensive and Set-of-Mark datasets, extending the diversity and generality. By training over different base LLMs including TinyLlama1.1B, InternLM2-7B, LLaMA2-13B, and Mixtral8x7B, we obtain a spectrum of MLLMs that vary in parameter size and multilingual capabilities. Comprehensive benchmarking reveals a strong correlation between the multi-modal performance with the data and parameter scales. 
    
[^7]: PromptCrypt: 使用表情符号对大型语言模型进行安全通信的提示加密

    PromptCrypt: Prompt Encryption for Secure Communication with Large Language Models

    [https://arxiv.org/abs/2402.05868](https://arxiv.org/abs/2402.05868)

    PromptCrypt是一种使用表情符号对用户输入进行加密的机制，保护了大型语言模型（LLM）中用户的隐私，防止数据泄露和解密。

    

    基于云的大型语言模型（LLM）如ChatGPT在日常操作中变得越来越重要，成为各种应用程序中的重要工具。虽然这些模型在可访问性和功能性方面带来了重大好处，但它们也引入了重要的隐私问题：在云基础架构中传输和存储用户数据会产生重大的数据泄露和未经授权访问敏感信息的风险；即使数据的传输和存储被加密，LLM服务提供商仍然知道数据的真实内容，从而阻止个人或实体放心使用此类LLM服务。为了解决这些问题，本文提出了一种简单但有效的机制PromptCrypt来保护用户隐私。它使用表情符号对用户输入进行加密，然后将其发送到LLM，有效地使其对人类或LLM的检查无法理解，同时保留原始提示的意图，从而确保用户隐私。

    Cloud-based large language models (LLMs) such as ChatGPT have increasingly become integral to daily operations, serving as vital tools across various applications. While these models offer substantial benefits in terms of accessibility and functionality, they also introduce significant privacy concerns: the transmission and storage of user data in cloud infrastructures pose substantial risks of data breaches and unauthorized access to sensitive information; even if the transmission and storage of data is encrypted, the LLM service provider itself still knows the real contents of the data, preventing individuals or entities from confidently using such LLM services. To address these concerns, this paper proposes a simple yet effective mechanism PromptCrypt to protect user privacy. It uses Emoji to encrypt the user inputs before sending them to LLM, effectively rendering them indecipherable to human or LLM's examination while retaining the original intent of the prompt, thus ensuring the 
    
[^8]: LitLLM：科学文献综述工具包

    LitLLM: A Toolkit for Scientific Literature Review

    [https://arxiv.org/abs/2402.01788](https://arxiv.org/abs/2402.01788)

    "LitLLM: A Toolkit for Scientific Literature Review" 提出了一个基于 RAG 原则的工具包，通过使用专门的提示和指导技术，结合大型语言模型（LLM），实现了科学文献综述的自动化。这个工具包不仅可以通过转化摘要为关键词进行文献检索，还可以通过补充相关论文或关键词进行定制化的检索。

    

    进行科学论文的文献综述对于理解研究、其限制以及构建在现有工作基础上是必不可少的。这是一项繁琐的任务，因此自动文献综述生成器变得有吸引力。然而，许多使用大型语言模型（LLM）生成此类综述的现有工作存在显著限制。它们倾向于产生虚构的非实际信息，并忽略它们未受过训练的最新研究。为了解决这些限制，我们提出了一个基于检索增强生成（RAG）原则的工具包，在LLM的帮助下，使用专门的提示和指导技术。我们的系统首先通过将用户提供的摘要转化为关键词来进行网络搜索，以检索相关论文，其中使用了现成的LLM。作者可以通过补充相关论文或关键词来改进搜索，从而实现定制化的检索过程。其次，系统根据-

    Conducting literature reviews for scientific papers is essential for understanding research, its limitations, and building on existing work. It is a tedious task which makes an automatic literature review generator appealing. Unfortunately, many existing works that generate such reviews using Large Language Models (LLMs) have significant limitations. They tend to hallucinate-generate non-actual information-and ignore the latest research they have not been trained on. To address these limitations, we propose a toolkit that operates on Retrieval Augmented Generation (RAG) principles, specialized prompting and instructing techniques with the help of LLMs. Our system first initiates a web search to retrieve relevant papers by summarizing user-provided abstracts into keywords using an off-the-shelf LLM. Authors can enhance the search by supplementing it with relevant papers or keywords, contributing to a tailored retrieval process. Second, the system re-ranks the retrieved papers based on t
    
[^9]: 通过函数编码器实现零-shot强化学习

    Zero-Shot Reinforcement Learning via Function Encoders

    [https://arxiv.org/abs/2401.17173](https://arxiv.org/abs/2401.17173)

    本论文提出了一种用于实现零-shot迁移的函数编码器，通过将函数表示为学习到的非线性基函数的加权组合，代理程序通过连贯的向量表示了当前任务与先前看到任务的关联信息，从而实现了在相关任务之间的迁移，无需额外训练。

    

    尽管强化学习（RL）可以解决许多具有挑战性的序列决策问题，但在相关任务之间实现零-shot迁移仍然是一个挑战。难点在于寻找一个良好的表示来表达当前任务，以便代理程序理解它与先前看到的任务的关系。为了实现零-shot迁移，我们引入了函数编码器，一种表示学习算法，它将函数表示为学习到的非线性基函数的加权组合。通过使用函数编码器来表示奖励函数或转移函数，代理程序通过一个连贯的向量表示有关当前任务与先前看到的任务的关联信息。因此，代理能够在运行时在相关任务之间实现迁移，而无需进行额外的训练。通过将基本RL算法与函数编码器结合，我们在三个RL领域中展示了最先进的数据效率、渐近性能和训练稳定性。

    Although reinforcement learning (RL) can solve many challenging sequential decision making problems, achieving zero-shot transfer across related tasks remains a challenge. The difficulty lies in finding a good representation for the current task so that the agent understands how it relates to previously seen tasks. To achieve zero-shot transfer, we introduce the function encoder, a representation learning algorithm which represents a function as a weighted combination of learned, non-linear basis functions. By using a function encoder to represent the reward function or the transition function, the agent has information on how the current task relates to previously seen tasks via a coherent vector representation. Thus, the agent is able to achieve transfer between related tasks at run time with no additional training. We demonstrate state-of-the-art data efficiency, asymptotic performance, and training stability in three RL fields by augmenting basic RL algorithms with a function encod
    
[^10]: 使用Sum-Product Networks生成可能的反事实推理

    Generating Likely Counterfactuals Using Sum-Product Networks. (arXiv:2401.14086v1 [cs.AI])

    [http://arxiv.org/abs/2401.14086](http://arxiv.org/abs/2401.14086)

    由于用户需求和最近的法规要求，需要对AI系统所做出的决策进行解释。本论文提出了一种使用Sum-Product Networks模拟寻找高可能性反事实推理的系统，该系统能够提供满足多个常见要求的最佳解释。

    

    由于用户需求和最近的法规（GDPR、AI法案），需要解释AI系统所做出的决策。这些决策往往只能在事后解释，反事实推理成为常见的解释方式。什么构成了最佳的反事实解释必须考虑多个方面，其中“样本距离”是最常见的。我们认为，这一要求经常会导致不太可能且因此价值有限的解释。在这里，我们提出了一个能够提供高可能性解释的系统。我们展示了使用混合整数优化（MIO）模拟寻找满足反事实推理的许多常见要求的最有可能解释。在此过程中，我们提出了Sum-Product Network（SPN）的MIO表达，并使用SPN估计反事实的可能性，这对独立的兴趣也有用。与生成反事实解释的几种方法进行数值比较。

    Due to user demand and recent regulation (GDPR, AI Act), decisions made by AI systems need to be explained. These decisions are often explainable only post hoc, where counterfactual explanations are popular. The question of what constitutes the best counterfactual explanation must consider multiple aspects, where "distance from the sample" is the most common. We argue that this requirement frequently leads to explanations that are unlikely and, therefore, of limited value. Here, we present a system that provides high-likelihood explanations. We show that the search for the most likely explanations satisfying many common desiderata for counterfactual explanations can be modeled using mixed-integer optimization (MIO). In the process, we propose an MIO formulation of a Sum-Product Network (SPN) and use the SPN to estimate the likelihood of a counterfactual, which can be of independent interest. A numerical comparison against several methods for generating counterfactual explanations is pr
    
[^11]: 基于LLM的代码生成中的偏差评估与缓解

    Bias Assessment and Mitigation in LLM-based Code Generation. (arXiv:2309.14345v1 [cs.SE])

    [http://arxiv.org/abs/2309.14345](http://arxiv.org/abs/2309.14345)

    这项研究提出了一个新颖的偏差评估框架，针对代码生成任务进行设计。通过对九个最先进的基于LLM的代码生成模型进行广泛评估，发现其中31.45\%到79.93\%的代码函数具有偏见，并提出了如何缓解这种偏见的方法。

    

    利用最新的大型语言模型（LLM），自动代码生成模型在提高软件开发编码过程的生产力和效率方面起着至关重要的作用。随着LLM在软件编码生态系统中的普及，一个紧迫的问题已经出现：生成的代码是否包含与年龄、性别和种族相关的社会偏见？这个问题关系到依赖于这些模型生成的代码的软件应用的完整性、公平性和道德基础，然而在文献中还没有得到充分探讨。本文提出了一个专为代码生成任务设计的新颖偏差评估框架。基于该框架，我们对九个最先进的基于LLM的代码生成模型的偏差进行了广泛评估。我们的发现揭示了，首先，我们评估的代码生成模型生成的31.45\%到79.93\%的代码函数具有偏见，9.68\%到37.37\%的代码函数的功能使

    Utilizing state-of-the-art Large Language Models (LLMs), automatic code generation models play a pivotal role in enhancing the productivity and efficiency of software development coding procedures. As the adoption of LLMs becomes more widespread in software coding ecosystems, a pressing issue has emerged: does the generated code contain social biases, such as those related to age, gender, and race? This issue concerns the integrity, fairness, and ethical foundation of software applications that depend on the code generated by these models, yet is under-explored in the literature. This paper presents a novel bias assessment framework that is specifically designed for code generation tasks. Based on this framework, we conduct an extensive evaluation on the bias of nine state-of-the-art LLM-based code generation models. Our findings reveal that first, 31.45\% to 79.93\% code functions generated by our evaluated code generation models are biased, and 9.68\% to 37.37\% code functions' funct
    
[^12]: 概要、亮点和行动项目：设计、实现和评估基于LLM的会议总结系统

    Summaries, Highlights, and Action items: Design, implementation and evaluation of an LLM-powered meeting recap system. (arXiv:2307.15793v1 [cs.HC])

    [http://arxiv.org/abs/2307.15793](http://arxiv.org/abs/2307.15793)

    这项研究设计、实现和评估了一种基于LLM的会议总结系统，通过减少个人会议负担和增加会议输出的清晰度和一致性，提高了会议体验。

    

    会议在工作协调中发挥着关键的基础设施作用。近年来，由于向混合和远程工作的转变，越来越多的会议正在转移到在线计算机媒体空间。这导致了新的问题（例如在更不吸引人的会议上花费更多的时间）和新的机会（例如自动转录/字幕和总结支持）。最近的大型语言模型（LLMs）在对话总结方面取得了进展，通过减少个人的会议负担和增加会议输出的清晰度和一致性，有可能提高会议体验。尽管存在这种潜力，但由于长篇转录和无法根据用户的上下文捕捉到多样的总结需求，它们面临着技术限制。为了填补这些差距，我们设计、实现并在上下文中评估了一种会议总结系统。我们首先构思了两个明显的总结表示方式——重要亮点和结构化的分级会议纪要视图。我们开发了一个系统来实现这些表示方法。

    Meetings play a critical infrastructural role in the coordination of work. In recent years, due to shift to hybrid and remote work, more meetings are moving to online Computer Mediated Spaces. This has led to new problems (e.g. more time spent in less engaging meetings) and new opportunities (e.g. automated transcription/captioning and recap support). Recent advances in large language models (LLMs) for dialog summarization have the potential to improve the experience of meetings by reducing individuals' meeting load and increasing the clarity and alignment of meeting outputs. Despite this potential, they face technological limitation due to long transcripts and inability to capture diverse recap needs based on user's context. To address these gaps, we design, implement and evaluate in-context a meeting recap system. We first conceptualize two salient recap representations -- important highlights, and a structured, hierarchical minutes view. We develop a system to operationalize the rep
    
[^13]: 优先轨迹回放：一种用于数据驱动强化学习的回放记忆方法

    Prioritized Trajectory Replay: A Replay Memory for Data-driven Reinforcement Learning. (arXiv:2306.15503v1 [cs.LG])

    [http://arxiv.org/abs/2306.15503](http://arxiv.org/abs/2306.15503)

    本研究提出了一种名为优先轨迹回放的回放记忆方法，将数据采样的视角扩展到轨迹中，从有限的数据中提取更全面的信息。这种方法通过反向采样轨迹来提高学习效率，并利用加权评论目标避免采样未见过的动作。优先轨迹回放还能根据不同的优先度指标优先采样效率更高的轨迹。

    

    近年来，数据驱动的强化学习（RL），也称为离线RL，引起了广泛关注。然而，尽管其具有提升在线RL性能的潜力，但离线RL中的数据采样技术的作用却被忽视了。最近的研究表明，直接将采样技术应用于状态转换并不能始终提高离线RL的性能。因此，在本研究中，我们提出了一种记忆技术——优先轨迹回放（TR/PTR），它将采样的视角扩展到轨迹中，以从有限的数据中提取更全面的信息。TR通过反向采样轨迹来提高学习效率，优化后续状态信息的使用。在TR的基础上，我们构建了加权评论目标，以避免在离线训练中采样未见过的动作，并且引入了优先轨迹回放（PTR）来实现更高效的轨迹采样，根据不同的轨迹优先度指标进行优先设置。我们演示了...

    In recent years, data-driven reinforcement learning (RL), also known as offline RL, have gained significant attention. However, the role of data sampling techniques in offline RL has been overlooked despite its potential to enhance online RL performance. Recent research suggests applying sampling techniques directly to state-transitions does not consistently improve performance in offline RL. Therefore, in this study, we propose a memory technique, (Prioritized) Trajectory Replay (TR/PTR), which extends the sampling perspective to trajectories for more comprehensive information extraction from limited data. TR enhances learning efficiency by backward sampling of trajectories that optimizes the use of subsequent state information. Building on TR, we build the weighted critic target to avoid sampling unseen actions in offline training, and Prioritized Trajectory Replay (PTR) that enables more efficient trajectory sampling, prioritized by various trajectory priority metrics. We demonstrat
    

