# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Institutional Platform for Secure Self-Service Large Language Model Exploration](https://rss.arxiv.org/abs/2402.00913) | 这个论文介绍了一个用户友好型平台，旨在使大型定制语言模型更易于使用，通过最新的多LoRA推理技术和定制适配器，实现了数据隔离、加密和身份验证的安全服务。 |
| [^2] | [Tur[k]ingBench: A Challenge Benchmark for Web Agents](https://arxiv.org/abs/2403.11905) | Tur[k]ingBench是一个挑战性的网络代理基准测试，用于评估最先进的多模态模型在处理包含文本指示和多模态上下文的复杂任务时的泛化能力。 |
| [^3] | [KnowAgent: Knowledge-Augmented Planning for LLM-Based Agents](https://arxiv.org/abs/2403.03101) | KnowAgent引入了显式动作知识，通过动作知识库和知识型自学习策略来增强LLM的规划能力，从而改善语言Agent的规划表现。 |
| [^4] | [TEncDM: Understanding the Properties of Diffusion Model in the Space of Language Model Encodings](https://arxiv.org/abs/2402.19097) | 通过在语言模型编码空间中训练模型，并使用基于Transformer的解码器以及自我调节，本文提出了名为TEncDM的文本编码扩散模型，在两个文本生成任务上展示了其优越性 |
| [^5] | [Data-freeWeight Compress and Denoise for Large Language Models](https://arxiv.org/abs/2402.16319) | 无需数据参与，基于大型语言模型结构提出了一种新的权重压缩方法，可有效压缩参数矩阵并保持正交性。 |
| [^6] | [BreakGPT: A Large Language Model with Multi-stage Structure for Financial Breakout Detection](https://arxiv.org/abs/2402.07536) | BreakGPT是第一个用于金融突破检测的大型语言模型，采用多阶段结构框架，提高了答案和理由的准确性。 |
| [^7] | [CIC: A framework for Culturally-aware Image Captioning](https://arxiv.org/abs/2402.05374) | CIC是一种面向文化感知图像字幕的框架，通过结合视觉问答和大型语言模型，它能够生成能描述图像中文化元素的详细字幕。 |
| [^8] | [Identifying and Analyzing Task-Encoding Tokens in Large Language Models](https://arxiv.org/abs/2401.11323) | 本文通过识别和分析任务编码标记，揭示了大型语言模型如何学习执行任务的方式。 |
| [^9] | [MacGyver: Are Large Language Models Creative Problem Solvers?](https://arxiv.org/abs/2311.09682) | 通过创建MACGYVER数据集并与人类比较，研究发现大型语言模型在创意问题解决方面独具挑战性，在知识广度和可行性方面与人类存在独特差异，同时还展示了通过新的提示技术提升大型语言模型的问题解决能力潜力。 |
| [^10] | [Augmenting Black-box LLMs with Medical Textbooks for Clinical Question Answering](https://arxiv.org/abs/2309.02233) | 该研究提出了一种名为LLMs增强医学教科书（LLM-AMT）的系统，通过插入式模块将权威医学教科书集成到LLMs的框架中，显著提高了LLMs在专业领域的能力。 |
| [^11] | [Can Large Language Models be Good Path Planners? A Benchmark and Investigation on Spatial-temporal Reasoning.](http://arxiv.org/abs/2310.03249) | 本研究提出了一种新的基准测试PPNL，评估大型语言模型的空间-时间推理能力。实验结果显示，少样本的GPT-4在空间推理方面表现良好，但仍有待改进。 |
| [^12] | [Diffusion Language Models Can Perform Many Tasks with Scaling and Instruction-Finetuning.](http://arxiv.org/abs/2308.12219) | 本文研究表明，通过扩展扩散语言模型的数据、规模和任务，可以有效使其成为强大的语言学习者。实验证明，扩展扩散语言模型在解决通用语言任务方面能够持续提高性能。 |
| [^13] | [Empower Your Model with Longer and Better Context Comprehension.](http://arxiv.org/abs/2307.13365) | 本文研究了大语言模型（LLMs）内的信息传递，并提出了一种名为注意力转移的技术，该技术能够使模型在不增加训练或对生成流畅性的影响的情况下实现更长更好的上下文理解。 |
| [^14] | [Making the Implicit Explicit: Implicit Content as a First Class Citizen in NLP.](http://arxiv.org/abs/2305.14583) | 该研究通过将自然语言处理的重点放在隐式内容上，提出了一种通过推理和分解方法降低自然语言处理复杂度的新方法，并在嵌入，计算政治学和构建发现方面实现了显著的改进和应用。 |
| [^15] | [ML-SUPERB: Multilingual Speech Universal PERformance Benchmark.](http://arxiv.org/abs/2305.10615) | 本文提出了一个覆盖143种语言、用于自我监督学习模型性能基准的多语种语音基准 ML-SUPERB，并发现自我监督学习模型可以显著提高性能且多语种模型不总是比单语言模型表现更好。 |

# 详细

[^1]: 用于安全自助大型语言模型探索的机构平台

    Institutional Platform for Secure Self-Service Large Language Model Exploration

    [https://rss.arxiv.org/abs/2402.00913](https://rss.arxiv.org/abs/2402.00913)

    这个论文介绍了一个用户友好型平台，旨在使大型定制语言模型更易于使用，通过最新的多LoRA推理技术和定制适配器，实现了数据隔离、加密和身份验证的安全服务。

    

    本文介绍了由肯塔基大学应用人工智能中心开发的用户友好型平台，旨在使大型定制语言模型（LLM）更易于使用。通过利用最近在多LoRA推理方面的进展，系统有效地适应了各类用户和项目的定制适配器。论文概述了系统的架构和关键特性，包括数据集策划、模型训练、安全推理和基于文本的特征提取。我们通过使用基于代理的方法建立了一个基于租户意识的计算网络，在安全地利用孤立资源岛的基础上形成了一个统一的系统。该平台致力于提供安全的LLM服务，强调过程和数据隔离、端到端加密以及基于角色的资源身份验证。该贡献与实现简化访问先进的AI模型和技术以支持科学发现的总体目标一致。

    This paper introduces a user-friendly platform developed by the University of Kentucky Center for Applied AI, designed to make large, customized language models (LLMs) more accessible. By capitalizing on recent advancements in multi-LoRA inference, the system efficiently accommodates custom adapters for a diverse range of users and projects. The paper outlines the system's architecture and key features, encompassing dataset curation, model training, secure inference, and text-based feature extraction.   We illustrate the establishment of a tenant-aware computational network using agent-based methods, securely utilizing islands of isolated resources as a unified system. The platform strives to deliver secure LLM services, emphasizing process and data isolation, end-to-end encryption, and role-based resource authentication. This contribution aligns with the overarching goal of enabling simplified access to cutting-edge AI models and technology in support of scientific discovery.
    
[^2]: Tur[k]ingBench：用于网络代理的挑战基准测试

    Tur[k]ingBench: A Challenge Benchmark for Web Agents

    [https://arxiv.org/abs/2403.11905](https://arxiv.org/abs/2403.11905)

    Tur[k]ingBench是一个挑战性的网络代理基准测试，用于评估最先进的多模态模型在处理包含文本指示和多模态上下文的复杂任务时的泛化能力。

    

    最近的聊天机器人展示了在原始文本形式下理解和交流的令人印象深刻的能力。然而，世界上不仅仅是原始文本。例如，人们在网页上花费大量时间，在这些网页上，文本与其他形式交织在一起，并以各种复杂互动的形式完成任务。最先进的多模型是否能够推广到这种复杂的领域呢？为了回答这个问题，我们介绍了TurkingBench，一个由包含多模态背景的文本说明制定的任务基准。与现有的使用人工合成的网页的工作不同，这里我们使用最初设计用于各种注释目的的自然HTML页面。每个任务的HTML说明也被实例化为各种值（从众包任务获得）以形成任务的新实例。这个基准包含32.2K个实例。

    arXiv:2403.11905v1 Announce Type: new  Abstract: Recent chatbots have demonstrated impressive ability to understand and communicate in raw-text form. However, there is more to the world than raw text. For example, humans spend long hours of their time on web pages, where text is intertwined with other modalities and tasks are accomplished in the form of various complex interactions. Can state-of-the-art multi-modal models generalize to such complex domains?   To address this question, we introduce TurkingBench, a benchmark of tasks formulated as web pages containing textual instructions with multi-modal context. Unlike existing work which employs artificially synthesized web pages, here we use natural HTML pages that were originally designed for crowdsourcing workers for various annotation purposes. The HTML instructions of each task are also instantiated with various values (obtained from the crowdsourcing tasks) to form new instances of the task. This benchmark contains 32.2K instanc
    
[^3]: KnowAgent: 知识增强规划用于基于LLM的Agent

    KnowAgent: Knowledge-Augmented Planning for LLM-Based Agents

    [https://arxiv.org/abs/2403.03101](https://arxiv.org/abs/2403.03101)

    KnowAgent引入了显式动作知识，通过动作知识库和知识型自学习策略来增强LLM的规划能力，从而改善语言Agent的规划表现。

    

    大型语言模型(LLMs)在复杂推理任务中表现出巨大潜力，但在处理更复杂的挑战时仍有所不足，特别是与环境互动通过生成可执行动作时。这种不足主要来自于语言Agent中缺乏内置动作知识，导致在任务求解过程中无法有效引导规划轨迹，从而导致规划幻觉。为了解决这个问题，我们引入了KnowAgent，一种旨在通过整合显式动作知识来增强LLM规划能力的新方法。具体而言，KnowAgent采用了一个动作知识库和一个知识型自学习策略来限制规划过程中的行动路径，实现更合理的轨迹合成，进而提高语言Agent的计划性能。基于HotpotQA和ALFWorld的实验结果基于不同的主干模型。

    arXiv:2403.03101v1 Announce Type: cross  Abstract: Large Language Models (LLMs) have demonstrated great potential in complex reasoning tasks, yet they fall short when tackling more sophisticated challenges, especially when interacting with environments through generating executable actions. This inadequacy primarily stems from the lack of built-in action knowledge in language agents, which fails to effectively guide the planning trajectories during task solving and results in planning hallucination. To address this issue, we introduce KnowAgent, a novel approach designed to enhance the planning capabilities of LLMs by incorporating explicit action knowledge. Specifically, KnowAgent employs an action knowledge base and a knowledgeable self-learning strategy to constrain the action path during planning, enabling more reasonable trajectory synthesis, and thereby enhancing the planning performance of language agents. Experimental results on HotpotQA and ALFWorld based on various backbone m
    
[^4]: TEncDM: 在语言模型编码空间中理解扩散模型的属性

    TEncDM: Understanding the Properties of Diffusion Model in the Space of Language Model Encodings

    [https://arxiv.org/abs/2402.19097](https://arxiv.org/abs/2402.19097)

    通过在语言模型编码空间中训练模型，并使用基于Transformer的解码器以及自我调节，本文提出了名为TEncDM的文本编码扩散模型，在两个文本生成任务上展示了其优越性

    

    受到扩散模型在各个领域取得成功的启发，许多研究论文提出了将其应用于文本数据的方法。尽管有这些努力，但没有一种方法能够达到大型语言模型的质量。本文对文本扩散模型的关键组件进行了全面分析，并介绍了一种名为Text Encoding Diffusion Model (TEncDM)的新方法。我们在语言模型编码空间中训练我们的模型，而不是通常使用的标记嵌入空间。此外，我们提出使用基于Transformer的解码器，利用上下文信息进行文本重构。我们还分析了自我调节，并发现这会增加模型输出的数量级，从而减少推理阶段的去噪步骤数量。在两个下游文本生成任务QQP和XSum上对TEncDM的评估表明其优越性。

    arXiv:2402.19097v1 Announce Type: new  Abstract: Drawing inspiration from the success of diffusion models in various domains, numerous research papers proposed methods for adapting them to text data. Despite these efforts, none of them has managed to achieve the quality of the large language models. In this paper, we conduct a comprehensive analysis of key components of the text diffusion models and introduce a novel approach named Text Encoding Diffusion Model (TEncDM). Instead of the commonly used token embedding space, we train our model in the space of the language model encodings. Additionally, we propose to use a Transformer-based decoder that utilizes contextual information for text reconstruction. We also analyse self-conditioning and find that it increases the magnitude of the model outputs, allowing the reduction of the number of denoising steps at the inference stage. Evaluation of TEncDM on two downstream text generation tasks, QQP and XSum, demonstrates its superiority ove
    
[^5]: 大型语言模型的无数据权重压缩和去噪

    Data-freeWeight Compress and Denoise for Large Language Models

    [https://arxiv.org/abs/2402.16319](https://arxiv.org/abs/2402.16319)

    无需数据参与，基于大型语言模型结构提出了一种新的权重压缩方法，可有效压缩参数矩阵并保持正交性。

    

    大型语言模型(LLMs)正在重塑人工智能研究领域的格局，特别是随着模型参数的显著扩大，跨越各个领域展现出卓越能力。然而，模型参数的可扩展性受限于GPU内存和计算速度的限制。为了解决这些限制，出现了各种权重压缩方法，如剪枝和量化。鉴于语言模型中权重矩阵的低秩特性，通过矩阵分解减少权重在压缩参数方面无疑具有显著潜力和前景。在本文中，借鉴LLMs的内在结构，我们提出了一种称为无数据联合秩-k逼近的新方法，用于压缩参数矩阵。值得注意的是，我们的方法特点在于无需额外涉及任何语料库，同时保持正交性。

    arXiv:2402.16319v1 Announce Type: new  Abstract: Large Language Models (LLMs) are reshaping the research landscape in artificial intelligence, particularly as model parameters scale up significantly, unlocking remarkable capabilities across various domains. Nevertheless, the scalability of model parameters faces constraints due to limitations in GPU memory and computational speed. To address these constraints, various weight compression methods have emerged, such as Pruning and Quantization. Given the low-rank nature of weight matrices in language models, the reduction of weights through matrix decomposition undoubtedly holds significant potential and promise. In this paper, drawing upon the intrinsic structure of LLMs, we propose a novel approach termed Data-free Joint Rank-k Approximation for compressing the parameter matrices. Significantly, our method is characterized by without necessitating additional involvement of any corpus, while simultaneously preserving orthogonality in con
    
[^6]: BreakGPT: 一种具有多阶段结构的大型语言模型用于金融突破检测

    BreakGPT: A Large Language Model with Multi-stage Structure for Financial Breakout Detection

    [https://arxiv.org/abs/2402.07536](https://arxiv.org/abs/2402.07536)

    BreakGPT是第一个用于金融突破检测的大型语言模型，采用多阶段结构框架，提高了答案和理由的准确性。

    

    交易区间突破（TRB）是金融交易技术分析中的一种关键方法，广泛应用于股票、期货和外汇等金融市场的交易者。然而，区分真假突破并提供正确的理由对投资者来说具有重要挑战。最近，大型语言模型在各种下游应用中取得了成功，但在金融突破检测领域的效果仍不理想。原因在于突破检测需要独特的数据和特定的知识。为了解决这些问题，我们引入了BreakGPT，这是第一个用于金融突破检测的大型语言模型。此外，我们还开发了一种名为多阶段结构的新颖框架，有效地减少了下游应用中的错误。实验结果表明，与GPT-3.5相比，BreakGPT的答案和理由的准确性提高了44%，有助于金融突破检测。

    Trading range breakout (TRB) is a key method in the technical analysis of financial trading, widely employed by traders in financial markets such as stocks, futures, and foreign exchange. However, distinguishing between true and false breakout and providing the correct rationale cause significant challenges to investors. Recently, large language models have achieved success in various downstream applications, but their effectiveness in the domain of financial breakout detection has been subpar. The reason is that the unique data and specific knowledge are required in breakout detection. To address these issues, we introduce BreakGPT, the first large language model for financial breakout detection. Furthermore, we have developed a novel framework for large language models, namely multi-stage structure, effectively reducing mistakes in downstream applications. Experimental results indicate that compared to GPT-3.5, BreakGPT improves the accuracy of answers and rational by 44%, with the m
    
[^7]: CIC：一种面向文化感知图像字幕的框架

    CIC: A framework for Culturally-aware Image Captioning

    [https://arxiv.org/abs/2402.05374](https://arxiv.org/abs/2402.05374)

    CIC是一种面向文化感知图像字幕的框架，通过结合视觉问答和大型语言模型，它能够生成能描述图像中文化元素的详细字幕。

    

    图像字幕通过使用视觉-语言预训练模型（VLPs）如BLIP从图像生成描述性句子，这种方法已经取得了很大的改进。然而，当前的方法缺乏对图像中所描绘的文化元素（例如亚洲文化群体的传统服装）生成详细描述性字幕的能力。在本文中，我们提出了一种新的框架，\textbf{面向文化感知图像字幕（CIC）}，该框架能够从代表不同文化的图像中生成字幕并描述文化元素。受到将视觉模态和大型语言模型（LLMs）通过适当的提示进行组合的方法的启发，我们的框架（1）根据图像中的文化类别生成问题，（2）利用生成的问题从视觉问答（VQA）中提取文化视觉元素，（3）使用带有提示的LLMs生成文化感知字幕。我们在4个不同大学的45名参与者上进行了人工评估。

    Image Captioning generates descriptive sentences from images using Vision-Language Pre-trained models (VLPs) such as BLIP, which has improved greatly. However, current methods lack the generation of detailed descriptive captions for the cultural elements depicted in the images, such as the traditional clothing worn by people from Asian cultural groups. In this paper, we propose a new framework, \textbf{Culturally-aware Image Captioning (CIC)}, that generates captions and describes cultural elements extracted from cultural visual elements in images representing cultures. Inspired by methods combining visual modality and Large Language Models (LLMs) through appropriate prompts, our framework (1) generates questions based on cultural categories from images, (2) extracts cultural visual elements from Visual Question Answering (VQA) using generated questions, and (3) generates culturally-aware captions using LLMs with the prompts. Our human evaluation conducted on 45 participants from 4 dif
    
[^8]: 辨识并分析大型语言模型中的任务编码标记

    Identifying and Analyzing Task-Encoding Tokens in Large Language Models

    [https://arxiv.org/abs/2401.11323](https://arxiv.org/abs/2401.11323)

    本文通过识别和分析任务编码标记，揭示了大型语言模型如何学习执行任务的方式。

    

    在上下文学习（ICL）已成为自然语言处理中少样本学习的有效解决方案。然而，我们对ICL的工作机制的理解有限，特别是模型如何从ICL演示中学习执行任务。本文通过识别和分析任务编码标记，调查了这个问题。我们发现，模板标记和停用词标记最容易成为任务编码标记。此外，我们实验证明，词汇意思、重复和文本格式是这些标记的主要区别特征。我们的工作揭示了大型语言模型（LLMs）学习的方式。

    arXiv:2401.11323v2 Announce Type: replace  Abstract: In-context learning (ICL) has become an effective solution for few-shot learning in natural language processing. However, our understanding of ICL's working mechanisms is limited, specifically regarding how models learn to perform tasks from ICL demonstrations. For example, unexpectedly large changes in performance can arise from small changes in the prompt, leaving prompt design a largely empirical endeavour. In this paper, we investigate this problem by identifying and analyzing task-encoding tokens on whose representations the task performance depends. Using experiments that ablate the representations of different token types, we find that template and stopword tokens are the most prone to be task-encoding. In addition, we demonstrate experimentally that lexical meaning, repetition, and text formatting are the main distinguishing characteristics of these tokens. Our work sheds light on how large language models (LLMs) learn to per
    
[^9]: MacGyver：大型语言模型是否是创意问题解决者？

    MacGyver: Are Large Language Models Creative Problem Solvers?

    [https://arxiv.org/abs/2311.09682](https://arxiv.org/abs/2311.09682)

    通过创建MACGYVER数据集并与人类比较，研究发现大型语言模型在创意问题解决方面独具挑战性，在知识广度和可行性方面与人类存在独特差异，同时还展示了通过新的提示技术提升大型语言模型的问题解决能力潜力。

    

    我们在一个全新的约束设置中探究了现代大型语言模型的创意问题解决能力。为此，我们创建了MACGYVER，这是一个自动生成的数据集，包含超过1600个特意设计的现实世界问题，旨在引发物体的创新使用，并需要超越常规思维。我们随后向大型语言模型和人类展示我们的数据集，以比较和对比它们的问题解决能力。MACGYVER对这两个群体都具有挑战性，但以独特和互补的方式呈现。例如，人类擅长熟悉的任务，但在特定领域知识上有困难，导致更高的差异。相比之下，大型语言模型暴露于各种专业知识，尝试更广泛的问题，但在提出物理上不可行的行动时失败。最后，我们对大型语言模型进行了详细的错误分析，并展示了通过新的提示技术提高它们的问题解决能力的潜力。

    arXiv:2311.09682v2 Announce Type: replace-cross  Abstract: We explore the creative problem-solving capabilities of modern LLMs in a novel constrained setting. To this end, we create MACGYVER, an automatically generated dataset consisting of over 1,600 real-world problems deliberately designed to trigger innovative usage of objects and necessitate out-of-the-box thinking. We then present our collection to both LLMs and humans to compare and contrast their problem-solving abilities. MACGYVER is challenging for both groups, but in unique and complementary ways. For instance, humans excel in tasks they are familiar with but struggle with domain-specific knowledge, leading to a higher variance. In contrast, LLMs, exposed to a variety of specialized knowledge, attempt broader problems but fail by proposing physically-infeasible actions. Finally, we provide a detailed error analysis of LLMs, and demonstrate the potential of enhancing their problem-solving ability with novel prompting techniqu
    
[^10]: 用医学教科书增强黑盒LLMs进行临床问题回答

    Augmenting Black-box LLMs with Medical Textbooks for Clinical Question Answering

    [https://arxiv.org/abs/2309.02233](https://arxiv.org/abs/2309.02233)

    该研究提出了一种名为LLMs增强医学教科书（LLM-AMT）的系统，通过插入式模块将权威医学教科书集成到LLMs的框架中，显著提高了LLMs在专业领域的能力。

    

    大规模语言模型（LLMs）如ChatGPT已经展示出根据人类指令生成响应的印象能力。然而，由于它们缺乏特定、深入的知识，它们在医学领域的应用可能具有挑战性。在这项研究中，我们提出了一种名为LLMs增强医学教科书（LLM-AMT）的系统，旨在增强LLMs在专业领域的能力。LLM-AMT通过插入式模块将权威医学教科书集成到LLMs的框架中。这些模块包括一个查询增强器、一个混合教科书检索器和一个知识自我完善。它们共同整合权威医学知识。此外，一个LLMs阅读器有助于上下文理解。我们在三个医学问答任务上的实验结果表明，LLMAMT显著提高了响应质量，准确率提高了11.6%到16.6%。值得注意的是，以GPT-4-Turbo为基础模型

    arXiv:2309.02233v2 Announce Type: replace-cross  Abstract: Large-scale language models (LLMs) like ChatGPT have demonstrated impressive abilities in generating responses based on human instructions. However, their use in the medical field can be challenging due to their lack of specific, in-depth knowledge. In this study, we present a system called LLMs Augmented with Medical Textbooks (LLM-AMT) designed to enhance the proficiency of LLMs in specialized domains. LLM-AMT integrates authoritative medical textbooks into the LLMs' framework using plug-and-play modules. These modules include a Query Augmenter, a Hybrid Textbook Retriever, and a Knowledge Self-Refiner. Together, they incorporate authoritative medical knowledge. Additionally, an LLM Reader aids in contextual understanding. Our experimental results on three medical QA tasks demonstrate that LLMAMT significantly improves response quality, with accuracy gains ranging from 11.6% to 16.6%. Notably, with GPT-4-Turbo as the base mod
    
[^11]: 大型语言模型能成为好的路径规划器吗？对空间-时间推理进行的基准测试和调查

    Can Large Language Models be Good Path Planners? A Benchmark and Investigation on Spatial-temporal Reasoning. (arXiv:2310.03249v1 [cs.CL])

    [http://arxiv.org/abs/2310.03249](http://arxiv.org/abs/2310.03249)

    本研究提出了一种新的基准测试PPNL，评估大型语言模型的空间-时间推理能力。实验结果显示，少样本的GPT-4在空间推理方面表现良好，但仍有待改进。

    

    大型语言模型（LLM）在各种任务中取得了显著的成功，但在需要长期规划和空间推理的场景中仍然面临限制。为了促进这一研究方向，本文提出了一种新的基准测试，称为自然语言路径规划（PPNL）。我们的基准测试通过制定需要LLM导航到目标位置并避开障碍物和遵守约束条件的“路径规划”任务，评估LLM的空间-时间推理能力。利用这个基准测试，我们系统地调查了包括GPT-4在内的LLM，使用不同的少样本提示方法和各种规模的BART和T5进行微调。实验结果表明，在提示LLM进行推理和交互行动时，少样本的GPT-4在空间推理方面有希望，但仍无法进行长期时间推理。相比之下，经过微调的LLM取得了较好的结果。

    Large language models (LLMs) have achieved remarkable success across a wide spectrum of tasks; however, they still face limitations in scenarios that demand long-term planning and spatial reasoning. To facilitate this line of research, in this work, we propose a new benchmark, termed $\textbf{P}$ath $\textbf{P}$lanning from $\textbf{N}$atural $\textbf{L}$anguage ($\textbf{PPNL}$). Our benchmark evaluates LLMs' spatial-temporal reasoning by formulating ''path planning'' tasks that require an LLM to navigate to target locations while avoiding obstacles and adhering to constraints. Leveraging this benchmark, we systematically investigate LLMs including GPT-4 via different few-shot prompting methodologies and BART and T5 of various sizes via fine-tuning. Our experimental results show the promise of few-shot GPT-4 in spatial reasoning, when it is prompted to reason and act interleavedly, although it still fails to make long-term temporal reasoning. In contrast, while fine-tuned LLMs achieve
    
[^12]: 扩展性和指导调优的扩散语言模型能够完成多种任务

    Diffusion Language Models Can Perform Many Tasks with Scaling and Instruction-Finetuning. (arXiv:2308.12219v1 [cs.CL])

    [http://arxiv.org/abs/2308.12219](http://arxiv.org/abs/2308.12219)

    本文研究表明，通过扩展扩散语言模型的数据、规模和任务，可以有效使其成为强大的语言学习者。实验证明，扩展扩散语言模型在解决通用语言任务方面能够持续提高性能。

    

    最近生成式人工智能的兴起得益于扩散概率模型的生成能力和大规模语言模型的可扩展性。尽管具有潜力，但扩散语言模型是否能够解决与自回归模型相媲美的通用语言任务仍然不明确。本文证明了在数据、规模和任务方面扩展扩散模型能够有效使其成为强大的语言学习者。我们通过先通过掩码语言建模预训练从大规模数据中获取知识，再通过扩散适应将预训练的掩码语言模型改进为扩散语言模型，通过任务特定的微调和指导调优来发掘其在解决通用语言任务方面的多样性。实验证明，扩展扩散语言模型能够在下游语言任务中持续提高性能。

    The recent surge of generative AI has been fueled by the generative power of diffusion probabilistic models and the scalable capabilities of large language models. Despite their potential, it remains elusive whether diffusion language models can solve general language tasks comparable to their autoregressive counterparts. This paper demonstrates that scaling diffusion models w.r.t. data, sizes, and tasks can effectively make them strong language learners. We build competent diffusion language models at scale by first acquiring knowledge from massive data via masked language modeling pretraining thanks to their intrinsic connections. We then reprogram pretrained masked language models into diffusion language models via diffusive adaptation, wherein task-specific finetuning and instruction finetuning are explored to unlock their versatility in solving general language tasks. Experiments show that scaling diffusion language models consistently improves performance across downstream langua
    
[^13]: 用更长更好的上下文理解将模型赋能

    Empower Your Model with Longer and Better Context Comprehension. (arXiv:2307.13365v1 [cs.CL])

    [http://arxiv.org/abs/2307.13365](http://arxiv.org/abs/2307.13365)

    本文研究了大语言模型（LLMs）内的信息传递，并提出了一种名为注意力转移的技术，该技术能够使模型在不增加训练或对生成流畅性的影响的情况下实现更长更好的上下文理解。

    

    最近，随着大量的大语言模型（LLMs）的出现，人工智能的实现进入了一个新的时代。无论这些模型自身的容量和结构如何，都存在对LLMs具有更长更复杂上下文的增强理解的需求，而模型通常在处理超出其理解能力范围的句子序列时会遇到上限，导致产生离题或混乱的回答。虽然最近有几项工作试图以不同的方式解决这个问题，但它们很少关注“为什么模型无法自行弥补或增强自己的能力”。在本文中，我们对LLMs内的信息传递性质进行了深入研究，并提出了一种名为注意力转移的新技术。这种技术能够使模型在最小化额外训练或对生成流利性的影响的情况下实现更长更好的上下文理解。我们的实验证明了这一点。

    Recently, with the emergence of numerous Large Language Models (LLMs), the implementation of AI has entered a new era. Irrespective of these models' own capacity and structure, there is a growing demand for LLMs to possess enhanced comprehension of longer and more complex contexts with relatively smaller sizes. Models often encounter an upper limit when processing sequences of sentences that extend beyond their comprehension capacity and result in off-topic or even chaotic responses. While several recent works attempt to address this issue in various ways, they rarely focus on "why models are unable to compensate or strengthen their capabilities on their own". In this paper, we thoroughly investigate the nature of information transfer within LLMs and propose a novel technique called Attention Transition. This technique empowers models to achieve longer and better context comprehension with minimal additional training or impact on generation fluency. Our experiments are conducted in XSu
    
[^14]: 让隐含的显性化：以NLP中的隐式内容为第一公民

    Making the Implicit Explicit: Implicit Content as a First Class Citizen in NLP. (arXiv:2305.14583v1 [cs.CL])

    [http://arxiv.org/abs/2305.14583](http://arxiv.org/abs/2305.14583)

    该研究通过将自然语言处理的重点放在隐式内容上，提出了一种通过推理和分解方法降低自然语言处理复杂度的新方法，并在嵌入，计算政治学和构建发现方面实现了显著的改进和应用。

    

    语言是多元化的，一个表述可以用等价的形式重申，而其中的隐含和显性内容支持各种逻辑和语用推理。在处理表述时，我们考虑这些不同的方面，因为我们需要理解“这里很黑”可能是一个暗示需要打开灯。然而，NLP方法通常仅仅基于表面形式操作，省略了这种细微差别。在这项工作中，我们用语言来表示语言，并引导LLM将表述分解为逻辑和可信的推理。分解的降低复杂性，使它们更容易嵌入，开启了新的应用。我们的技术变化在句子嵌入基准测试中实现了最先进的改进，在计算政治学中有实质性应用，并引出一种新的构建发现过程，我们用人工注释验证了这种过程。

    Language is multifaceted. A given utterance can be re-expressed in equivalent forms, and its implicit and explicit content support various logical and pragmatic inferences. When processing an utterance, we consider these different aspects, as mediated by our interpretive goals -- understanding that "it's dark in here" may be a veiled direction to turn on a light. Nonetheless, NLP methods typically operate over the surface form alone, eliding this nuance.  In this work, we represent language with language, and direct an LLM to decompose utterances into logical and plausible inferences. The reduced complexity of the decompositions makes them easier to embed, opening up novel applications. Variations on our technique lead to state-of-the-art improvements on sentence embedding benchmarks, a substantive application in computational political science, and to a novel construct-discovery process, which we validate with human annotations.
    
[^15]: ML-SUPERB: 多语种语音自我监督学习性能基准

    ML-SUPERB: Multilingual Speech Universal PERformance Benchmark. (arXiv:2305.10615v1 [cs.SD])

    [http://arxiv.org/abs/2305.10615](http://arxiv.org/abs/2305.10615)

    本文提出了一个覆盖143种语言、用于自我监督学习模型性能基准的多语种语音基准 ML-SUPERB，并发现自我监督学习模型可以显著提高性能且多语种模型不总是比单语言模型表现更好。

    

    语音处理Universal PERformance Benchmark (SUPERB)是一个用于各种语音处理任务的自我监督学习模型性能基准的排行榜。然而，SUPERB在评估中主要考虑英语。本文介绍了多语种SUPERB (ML-SUPERB)，覆盖了143种语言（从高资源到濒危语言），考虑了自动语音识别和语言识别。与SUPERB概念类似，ML-SUPERB利用冻结的自我监督学习特征，并通过学习浅层下游模型的简单框架，用于多语种任务。与SUPERB基准类似，我们发现语音自我监督学习模型可以显著提高性能，与FBANK特征相比。此外，我们发现多语种模型并不总是比单语言模型表现更好。我们将发布ML-SUPERB作为一个挑战，提供组织好的数据集和可重现的训练脚本，用于未来的多语种表示研究。

    Speech processing Universal PERformance Benchmark (SUPERB) is a leaderboard to benchmark the performance of Self-Supervised Learning (SSL) models on various speech processing tasks. However, SUPERB largely considers English speech in its evaluation. This paper presents multilingual SUPERB (ML-SUPERB), covering 143 languages (ranging from high-resource to endangered), and considering both automatic speech recognition and language identification. Following the concept of SUPERB, ML-SUPERB utilizes frozen SSL features and employs a simple framework for multilingual tasks by learning a shallow downstream model. Similar to the SUPERB benchmark, we find speech SSL models can significantly improve performance compared to FBANK features. Furthermore, we find that multilingual models do not always perform better than their monolingual counterparts. We will release ML-SUPERB as a challenge with organized datasets and reproducible training scripts for future multilingual representation research.
    

