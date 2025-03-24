# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [GET: Unlocking the Multi-modal Potential of CLIP for Generalized Category Discovery](https://arxiv.org/abs/2403.09974) | 本文提出了一种文本嵌入合成器（TES），用于为无标签数据生成伪文本嵌入，以解锁CLIP用于广义类别发现任务中的多模态潜力。 |
| [^2] | [Large Language Models and Causal Inference in Collaboration: A Comprehensive Survey](https://arxiv.org/abs/2403.09606) | 大型语言模型的出现极大影响了自然语言处理领域，特别是通过其先进的推理能力。而本综述则重点评估和改进了大型语言模型在因果推断方面的应用，包括提高推理能力、解决公平和安全问题、提供解释和处理多模态。 |
| [^3] | [ContextGPT: Infusing LLMs Knowledge into Neuro-Symbolic Activity Recognition Models](https://arxiv.org/abs/2403.06586) | 将预训练的大型语言模型（LLMs）的常识知识有效地注入神经符号活动识别模型，以缓解标记数据稀缺性问题。 |
| [^4] | [Standardizing the Measurement of Text Diversity: A Tool and a Comparative Analysis of Scores](https://arxiv.org/abs/2403.00553) | 本论文提出了一种用于衡量文本多样性的标准分数，通过实证研究发现压缩算法可以捕捉类似于$n$-gram重叠同质性得分的信息，并结合多种度量方法来报告分数，适用于不同类型的文本分析。 |
| [^5] | [SPHINX-X: Scaling Data and Parameters for a Family of Multi-modal Large Language Models](https://arxiv.org/abs/2402.05935) | 本论文介绍了SPHINX-X，一种扩展的多模态大型语言模型系列。通过改进架构和训练效率，我们成功构建了一系列参数大小和多语言能力不同的MLLMs，与数据和参数规模有强相关性。 |
| [^6] | [PromptCrypt: Prompt Encryption for Secure Communication with Large Language Models](https://arxiv.org/abs/2402.05868) | PromptCrypt是一种使用表情符号对用户输入进行加密的机制，保护了大型语言模型（LLM）中用户的隐私，防止数据泄露和解密。 |
| [^7] | [LitLLM: A Toolkit for Scientific Literature Review](https://arxiv.org/abs/2402.01788) | "LitLLM: A Toolkit for Scientific Literature Review" 提出了一个基于 RAG 原则的工具包，通过使用专门的提示和指导技术，结合大型语言模型（LLM），实现了科学文献综述的自动化。这个工具包不仅可以通过转化摘要为关键词进行文献检索，还可以通过补充相关论文或关键词进行定制化的检索。 |

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
    
[^4]: 规范文本多样性的测量：一个工具和对分数的比较分析

    Standardizing the Measurement of Text Diversity: A Tool and a Comparative Analysis of Scores

    [https://arxiv.org/abs/2403.00553](https://arxiv.org/abs/2403.00553)

    本论文提出了一种用于衡量文本多样性的标准分数，通过实证研究发现压缩算法可以捕捉类似于$n$-gram重叠同质性得分的信息，并结合多种度量方法来报告分数，适用于不同类型的文本分析。

    

    大型语言模型生成的输出之间的多样性塑造了人们对其质量和实用性的看法。我们的工作通过实证研究英语文本的多样性得分。我们发现，计算效率高的压缩算法捕捉到与$n$-gram的重叠同质性得分所衡量的信息相似。此外，结合多种度量方法——压缩比、长$n$-gram的自重复、Self-BLEU和BERTScore——足以报告，因为它们彼此之间的相互关联较低。这些分数的适用性超出了生成模型的分析；例如，我们突出了在指导调整数据集和人类生成的文本上的应用。我们发布了一个多样性程度

    arXiv:2403.00553v1 Announce Type: new  Abstract: The diversity across outputs generated by large language models shapes the perception of their quality and utility. Prompt leaks, templated answer structure, and canned responses across different interactions are readily noticed by people, but there is no standard score to measure this aspect of model behavior. In this work we empirically investigate diversity scores on English texts. We find that computationally efficient compression algorithms capture information similar to what is measured by slow to compute $n$-gram overlap homogeneity scores. Further, a combination of measures -- compression ratios, self-repetition of long $n$-grams and Self-BLEU and BERTScore -- are sufficient to report, as they have low mutual correlation with each other. The applicability of scores extends beyond analysis of generative models; for example, we highlight applications on instruction-tuning datasets and human-produced texts. We release a diversity sc
    
[^5]: SPHINX-X: 扩展数据和参数用于一系列多模态大型语言模型

    SPHINX-X: Scaling Data and Parameters for a Family of Multi-modal Large Language Models

    [https://arxiv.org/abs/2402.05935](https://arxiv.org/abs/2402.05935)

    本论文介绍了SPHINX-X，一种扩展的多模态大型语言模型系列。通过改进架构和训练效率，我们成功构建了一系列参数大小和多语言能力不同的MLLMs，与数据和参数规模有强相关性。

    

    我们提出SPHINX-X，一种基于SPHINX开发的广泛多模态大型语言模型（MLLM）系列。为了改善架构和训练效率，我们通过移除冗余的视觉编码器、绕过完全填充的子图像，并将多阶段训练简化成为一阶段的全集合模式，修改了SPHINX框架。为了充分发挥MLLM的潜力，我们组装了一个综合的跨语言、跨视觉和视觉-语言任务的多领域、多模态的数据集，涵盖了公开可用的资源。我们进一步使用我们的OCR密集和Mark数据集丰富这个收集，扩展了多样性和普适性。通过对不同基础LLM进行训练，包括TinyLlama1.1B、InternLM2-7B、LLaMA2-13B和Mixtral8x7B，我们获得了一系列参数大小和多语言能力变化的MLLMs。全面的基准测试揭示了多模态性能与数据和参数规模之间的强相关性。

    We propose SPHINX-X, an extensive Multimodality Large Language Model (MLLM) series developed upon SPHINX. To improve the architecture and training efficiency, we modify the SPHINX framework by removing redundant visual encoders, bypassing fully-padded sub-images with skip tokens, and simplifying multi-stage training into a one-stage all-in-one paradigm. To fully unleash the potential of MLLMs, we assemble a comprehensive multi-domain and multimodal dataset covering publicly available resources in language, vision, and vision-language tasks. We further enrich this collection with our curated OCR intensive and Set-of-Mark datasets, extending the diversity and generality. By training over different base LLMs including TinyLlama1.1B, InternLM2-7B, LLaMA2-13B, and Mixtral8x7B, we obtain a spectrum of MLLMs that vary in parameter size and multilingual capabilities. Comprehensive benchmarking reveals a strong correlation between the multi-modal performance with the data and parameter scales. 
    
[^6]: PromptCrypt: 使用表情符号对大型语言模型进行安全通信的提示加密

    PromptCrypt: Prompt Encryption for Secure Communication with Large Language Models

    [https://arxiv.org/abs/2402.05868](https://arxiv.org/abs/2402.05868)

    PromptCrypt是一种使用表情符号对用户输入进行加密的机制，保护了大型语言模型（LLM）中用户的隐私，防止数据泄露和解密。

    

    基于云的大型语言模型（LLM）如ChatGPT在日常操作中变得越来越重要，成为各种应用程序中的重要工具。虽然这些模型在可访问性和功能性方面带来了重大好处，但它们也引入了重要的隐私问题：在云基础架构中传输和存储用户数据会产生重大的数据泄露和未经授权访问敏感信息的风险；即使数据的传输和存储被加密，LLM服务提供商仍然知道数据的真实内容，从而阻止个人或实体放心使用此类LLM服务。为了解决这些问题，本文提出了一种简单但有效的机制PromptCrypt来保护用户隐私。它使用表情符号对用户输入进行加密，然后将其发送到LLM，有效地使其对人类或LLM的检查无法理解，同时保留原始提示的意图，从而确保用户隐私。

    Cloud-based large language models (LLMs) such as ChatGPT have increasingly become integral to daily operations, serving as vital tools across various applications. While these models offer substantial benefits in terms of accessibility and functionality, they also introduce significant privacy concerns: the transmission and storage of user data in cloud infrastructures pose substantial risks of data breaches and unauthorized access to sensitive information; even if the transmission and storage of data is encrypted, the LLM service provider itself still knows the real contents of the data, preventing individuals or entities from confidently using such LLM services. To address these concerns, this paper proposes a simple yet effective mechanism PromptCrypt to protect user privacy. It uses Emoji to encrypt the user inputs before sending them to LLM, effectively rendering them indecipherable to human or LLM's examination while retaining the original intent of the prompt, thus ensuring the 
    
[^7]: LitLLM：科学文献综述工具包

    LitLLM: A Toolkit for Scientific Literature Review

    [https://arxiv.org/abs/2402.01788](https://arxiv.org/abs/2402.01788)

    "LitLLM: A Toolkit for Scientific Literature Review" 提出了一个基于 RAG 原则的工具包，通过使用专门的提示和指导技术，结合大型语言模型（LLM），实现了科学文献综述的自动化。这个工具包不仅可以通过转化摘要为关键词进行文献检索，还可以通过补充相关论文或关键词进行定制化的检索。

    

    进行科学论文的文献综述对于理解研究、其限制以及构建在现有工作基础上是必不可少的。这是一项繁琐的任务，因此自动文献综述生成器变得有吸引力。然而，许多使用大型语言模型（LLM）生成此类综述的现有工作存在显著限制。它们倾向于产生虚构的非实际信息，并忽略它们未受过训练的最新研究。为了解决这些限制，我们提出了一个基于检索增强生成（RAG）原则的工具包，在LLM的帮助下，使用专门的提示和指导技术。我们的系统首先通过将用户提供的摘要转化为关键词来进行网络搜索，以检索相关论文，其中使用了现成的LLM。作者可以通过补充相关论文或关键词来改进搜索，从而实现定制化的检索过程。其次，系统根据-

    Conducting literature reviews for scientific papers is essential for understanding research, its limitations, and building on existing work. It is a tedious task which makes an automatic literature review generator appealing. Unfortunately, many existing works that generate such reviews using Large Language Models (LLMs) have significant limitations. They tend to hallucinate-generate non-actual information-and ignore the latest research they have not been trained on. To address these limitations, we propose a toolkit that operates on Retrieval Augmented Generation (RAG) principles, specialized prompting and instructing techniques with the help of LLMs. Our system first initiates a web search to retrieve relevant papers by summarizing user-provided abstracts into keywords using an off-the-shelf LLM. Authors can enhance the search by supplementing it with relevant papers or keywords, contributing to a tailored retrieval process. Second, the system re-ranks the retrieved papers based on t
    

