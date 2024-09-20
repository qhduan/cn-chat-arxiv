# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [DPP-Based Adversarial Prompt Searching for Lanugage Models](https://arxiv.org/abs/2403.00292) | 通过Auto-regressive Selective Replacement Ascent (ASRA)算法，我们成功引导预训练语言模型生成有毒内容，具有很好的效果。 |
| [^2] | [MMSR: Symbolic Regression is a Multimodal Task](https://arxiv.org/abs/2402.18603) | 符号回归被视为一个多模态任务，研究人员将数据到表达式的映射视为翻译问题，引入大规模预训练模型。 |
| [^3] | [Rethinking the Roles of Large Language Models in Chinese Grammatical Error Correction](https://arxiv.org/abs/2402.11420) | 重新思考大型语言模型在中文语法错误纠正中的作用，利用LLMs作为解释器提供解释信息并作为评估器带来更合理的CGEC评估以增强性能 |
| [^4] | [MAC-SQL: A Multi-Agent Collaborative Framework for Text-to-SQL](https://arxiv.org/abs/2312.11242) | MAC-SQL是一种基于LLM的多智能体协作框架，针对文本到SQL任务中的巨大数据库和复杂用户问题，通过核心分解器智能体和辅助智能体的协作，利用外部工具和模型进行解析和修正，实现了高效的文本到SQL生成和查询解析。 |
| [^5] | [On Measuring Faithfulness or Self-consistency of Natural Language Explanations](https://arxiv.org/abs/2311.07466) | 本文论述了衡量自然语言解释的忠诚度或自一致性的问题。我们提出了自一致性测试来评估解释的输出级别的一致性。我们通过构建比较一致性测试库，并引入了新的自一致性度量CC-SHAP来支持我们的观点。 |
| [^6] | [ChemDFM: Dialogue Foundation Model for Chemistry.](http://arxiv.org/abs/2401.14818) | ChemDFM是首个面向化学智能的大型语言模型，它通过对化学文献和数据的训练，具备了存储、理解和推理化学知识和语言的能力，并且在化学领域的性能上优于其他开源模型。 |
| [^7] | [UniMS-RAG: A Unified Multi-source Retrieval-Augmented Generation for Personalized Dialogue Systems.](http://arxiv.org/abs/2401.13256) | 这项研究提出了一种统一多源检索增强生成系统（UniMS-RAG），通过统一知识源选择、知识检索和回复生成三个子任务，使语言模型能够根据需求自适应地检索证据和评估关联性，从而生成个性化的回复。 |
| [^8] | [Cross-Utterance Conditioned VAE for Speech Generation.](http://arxiv.org/abs/2309.04156) | 该论文提出了一种跨发言条件化变分自编码器框架，利用预训练语言模型和变分自编码器来增强语音合成的韵律，并确保自然语音生成。该框架的核心组件是跨发言CVAE，通过提取周围句子的声学、说话人和文本特征来生成上下文敏感的韵律特征，有效模拟人类韵律生成。同时，该论文还提出了两个实用算法：CUC-VAE TTS用于文本到语音合成和CUC-VAE SE用于语音编辑。 |
| [^9] | [Unleashing Infinite-Length Input Capacity for Large-scale Language Models with Self-Controlled Memory System.](http://arxiv.org/abs/2304.13343) | 该论文提出了一种自控内存系统，可以使大规模语言模型能够处理任意长度的输入，从而显著提高模型的性能表现。 |
| [^10] | [LLaMA-Adapter: Efficient Fine-tuning of Language Models with Zero-init Attention.](http://arxiv.org/abs/2303.16199) | 本文提出了一种基于适应提示和零初始化注意力机制的轻量级语言模型调整方法，可高效微调LLaMA为指令跟随模型，具有比Alpaca更短的微调时间并具有近似的响应质量。 |
| [^11] | [MM-SHAP: A Performance-agnostic Metric for Measuring Multimodal Contributions in Vision and Language Models & Tasks.](http://arxiv.org/abs/2212.08158) | 该论文提出了一种性能不可知的多模态得分方法MM-SHAP，可以可靠地量化多模态模型使用各自模态的比例，并应用于比较模型的平均多模态程度和衡量个体模型的贡献。实验结果表明单模态崩溃比以前认为的更为普遍，而MM-SHAP是分析VL模型多模态行为的有效工具。 |

# 详细

[^1]: 基于DPP的对抗性提示搜索用于语言模型

    DPP-Based Adversarial Prompt Searching for Lanugage Models

    [https://arxiv.org/abs/2403.00292](https://arxiv.org/abs/2403.00292)

    通过Auto-regressive Selective Replacement Ascent (ASRA)算法，我们成功引导预训练语言模型生成有毒内容，具有很好的效果。

    

    语言模型存在生成毫无意义和冒犯性内容的风险，这妨碍了它们的安全部署。因此，在部署之前发现并修改预训练语言模型潜在的有毒输出是至关重要的。本研究中，我们通过自动搜索提示来引导预训练语言模型生成特定目标输出的有毒内容。该问题具有挑战性，因为文本数据的离散性以及针对语言模型的单次前向传递所需的大量计算资源。为了应对这些挑战，我们提出了自回归选择替代上升（ASRA）算法，这是一种基于确定性点过程（DPP）的选择提示的离散优化算法。对六种不同预训练语言模型的实验结果表明，ASRA对引发有毒内容具有很好的效果。

    arXiv:2403.00292v1 Announce Type: new  Abstract: Language models risk generating mindless and offensive content, which hinders their safe deployment. Therefore, it is crucial to discover and modify potential toxic outputs of pre-trained language models before deployment. In this work, we elicit toxic content by automatically searching for a prompt that directs pre-trained language models towards the generation of a specific target output. The problem is challenging due to the discrete nature of textual data and the considerable computational resources required for a single forward pass of the language model. To combat these challenges, we introduce Auto-regressive Selective Replacement Ascent (ASRA), a discrete optimization algorithm that selects prompts based on both quality and similarity with determinantal point process (DPP). Experimental results on six different pre-trained language models demonstrate the efficacy of ASRA for eliciting toxic content. Furthermore, our analysis reve
    
[^2]: MMSR：符号回归是一个多模态任务

    MMSR: Symbolic Regression is a Multimodal Task

    [https://arxiv.org/abs/2402.18603](https://arxiv.org/abs/2402.18603)

    符号回归被视为一个多模态任务，研究人员将数据到表达式的映射视为翻译问题，引入大规模预训练模型。

    

    数学公式是探索自然规律几千年来人类智慧的结晶。用简洁的数学公式描述复杂的自然规律是科学家不断追求的目标，也是人工智能面临的重大挑战。这一领域被称为符号回归。在本文中，研究人员将从数据到表达式的映射视为翻译问题，并引入了相应的大规模预训练模型。

    arXiv:2402.18603v1 Announce Type: cross  Abstract: Mathematical formulas are the crystallization of human wisdom in exploring the laws of nature for thousands of years. Describing the complex laws of nature with a concise mathematical formula is a constant pursuit of scientists and a great challenge for artificial intelligence. This field is called symbolic regression. Symbolic regression was originally formulated as a combinatorial optimization problem, and GP and reinforcement learning algorithms were used to solve it. However, GP is sensitive to hyperparameters, and these two types of algorithms are inefficient. To solve this problem, researchers treat the mapping from data to expressions as a translation problem. And the corresponding large-scale pre-trained model is introduced. However, the data and expression skeletons do not have very clear word correspondences as the two languages do. Instead, they are more like two modalities (e.g., image and text). Therefore, in this paper, w
    
[^3]: 重新思考大型语言模型在中文语法错误纠正中的作用

    Rethinking the Roles of Large Language Models in Chinese Grammatical Error Correction

    [https://arxiv.org/abs/2402.11420](https://arxiv.org/abs/2402.11420)

    重新思考大型语言模型在中文语法错误纠正中的作用，利用LLMs作为解释器提供解释信息并作为评估器带来更合理的CGEC评估以增强性能

    

    最近，研究人员广泛研究大型语言模型（LLMs）在各种下游NLP任务中的作用。作为NLP领域的一项基础任务，中文语法错误纠正（CGEC）旨在纠正输入句子中的所有潜在语法错误。先前的研究表明，由于其具有挑战性的任务重点，LLMs作为CGEC校正器的性能仍然令人不满。为了推动CGEC领域更好地适应LLMs时代，我们重新思考LLMs在CGEC任务中的作用，使其能够在CGEC中得到更好的利用和探索。考虑到LLMs中存储的丰富语法知识和其强大的语义理解能力，我们利用LLMs作为解释器，为CGEC小模型提供解释信息，以增强性能。我们还将LLMs用作评估器，带来更合理的CGEC评估，从而缓解由于

    arXiv:2402.11420v1 Announce Type: new  Abstract: Recently, Large Language Models (LLMs) have been widely studied by researchers for their roles in various downstream NLP tasks. As a fundamental task in the NLP field, Chinese Grammatical Error Correction (CGEC) aims to correct all potential grammatical errors in the input sentences. Previous studies have shown that LLMs' performance as correctors on CGEC remains unsatisfactory due to its challenging task focus. To promote the CGEC field to better adapt to the era of LLMs, we rethink the roles of LLMs in the CGEC task so that they can be better utilized and explored in CGEC. Considering the rich grammatical knowledge stored in LLMs and their powerful semantic understanding capabilities, we utilize LLMs as explainers to provide explanation information for the CGEC small models during error correction to enhance performance. We also use LLMs as evaluators to bring more reasonable CGEC evaluations, thus alleviating the troubles caused by th
    
[^4]: MAC-SQL: 一种用于文本到SQL的多智能体协作框架

    MAC-SQL: A Multi-Agent Collaborative Framework for Text-to-SQL

    [https://arxiv.org/abs/2312.11242](https://arxiv.org/abs/2312.11242)

    MAC-SQL是一种基于LLM的多智能体协作框架，针对文本到SQL任务中的巨大数据库和复杂用户问题，通过核心分解器智能体和辅助智能体的协作，利用外部工具和模型进行解析和修正，实现了高效的文本到SQL生成和查询解析。

    

    最近基于LLM的文本到SQL方法通常在“巨大”的数据库和需要多步推理的复杂用户问题上遭受严重的性能下降。此外，大多数现有方法忽视了利用外部工具和模型协作的LLM的重要意义。为了解决这些挑战，我们引入了MAC-SQL，一种新颖的基于LLM的多智能体协作框架。我们的框架包括一个核心分解器智能体，用于进行带有少样本思维链的文本到SQL生成，同时还有两个辅助智能体，利用外部工具或模型获取较小的子数据库并修正错误的SQL查询。分解器智能体与辅助智能体合作，根据需要激活，并可以扩展以适应新的特性或工具，以实现有效的文本到SQL解析。在我们的框架中，我们最初利用GPT-4作为强大的LLM骨干来完成所有智能体任务，以确定...

    arXiv:2312.11242v3 Announce Type: replace  Abstract: Recent LLM-based Text-to-SQL methods usually suffer from significant performance degradation on ``huge" databases and complex user questions that require multi-step reasoning. Moreover, most existing methods neglect the crucial significance of LLMs utilizing external tools and model collaboration. To address these challenges, we introduce MAC-SQL, a novel LLM-based multi-agent collaborative framework. Our framework comprises a core decomposer agent for Text-to-SQL generation with few-shot chain-of-thought reasoning, accompanied by two auxiliary agents that utilize external tools or models to acquire smaller sub-databases and refine erroneous SQL queries. The decomposer agent collaborates with auxiliary agents, which are activated as needed and can be expanded to accommodate new features or tools for effective Text-to-SQL parsing. In our framework, We initially leverage GPT-4 as the strong backbone LLM for all agent tasks to determine
    
[^5]: 关于衡量自然语言解释的忠诚度或自一致性

    On Measuring Faithfulness or Self-consistency of Natural Language Explanations

    [https://arxiv.org/abs/2311.07466](https://arxiv.org/abs/2311.07466)

    本文论述了衡量自然语言解释的忠诚度或自一致性的问题。我们提出了自一致性测试来评估解释的输出级别的一致性。我们通过构建比较一致性测试库，并引入了新的自一致性度量CC-SHAP来支持我们的观点。

    

    大型语言模型（LLMs）可以通过事后或思维链（CoT）解释其预测。但是，LLM可能会编造听起来合理但不忠实于其基本推理的解释。最近的工作设计了旨在判断事后或CoT解释忠实度的测试。在这项工作中，我们认为这些忠实度测试不是衡量模型内部工作的忠实度，而是衡量其输出级别的自一致性。我们的贡献有三个方面：i）我们在模型可解释性的背景下澄清了忠实度测试的地位，将其描述为自一致性测试。我们通过ii）构建了一个比较一致性的测试库，首次在11个开放式LLMs和5个任务的通用套件上比较了现有测试，包括iii）我们的新的自一致性度量CC-SHAP。CC-SHAP是LLM自一致性的细粒度度量（而不是测试）。它进行比较。

    Large language models (LLMs) can explain their predictions through post-hoc or Chain-of-Thought (CoT) explanations. But an LLM could make up reasonably sounding explanations that are unfaithful to its underlying reasoning. Recent work has designed tests that aim to judge the faithfulness of post-hoc or CoT explanations. In this work we argue that these faithfulness tests do not measure faithfulness to the models' inner workings -- but rather their self-consistency at output level. Our contributions are three-fold: i) We clarify the status of faithfulness tests in view of model explainability, characterising them as self-consistency tests instead. This assessment we underline by ii) constructing a Comparative Consistency Bank for self-consistency tests that for the first time compares existing tests on a common suite of 11 open LLMs and 5 tasks -- including iii) our new self-consistency measure CC-SHAP. CC-SHAP is a fine-grained measure (not a test) of LLM self-consistency. It compares 
    
[^6]: ChemDFM: 化学领域对话基础模型

    ChemDFM: Dialogue Foundation Model for Chemistry. (arXiv:2401.14818v1 [cs.CL])

    [http://arxiv.org/abs/2401.14818](http://arxiv.org/abs/2401.14818)

    ChemDFM是首个面向化学智能的大型语言模型，它通过对化学文献和数据的训练，具备了存储、理解和推理化学知识和语言的能力，并且在化学领域的性能上优于其他开源模型。

    

    大型语言模型(LLMs)在自然语言处理的一般领域取得了巨大成功。它们的任务概括和自由对话能力可以极大地帮助设计化学智能(CGI)，以协助化学领域的实际研究。然而，在化学领域中存在专业语言和知识，如高度信息化的SMILES符号表示法，阻碍了一般领域LLMs在化学领域的性能。为此，我们开发了ChemDFM，这是首个面向CGI的LLM。ChemDFM-13B是在化学文献、教科书、说明书以及各种一般领域的数据中训练的34B令牌。因此，它可以存储、理解和推理化学知识和语言，同时具有先进的自由形式语言理解能力。广泛的定量评估表明，ChemDFM可以明显优于代表性的开源LLMs。此外，ChemDFM还可以...

    Large language models (LLMs) have established great success in the general domain of natural language processing. Their emerging task generalization and free-form dialogue capabilities can greatly help to design Chemical General Intelligence (CGI) to assist real-world research in chemistry. However, the existence of specialized language and knowledge in the field of chemistry, such as the highly informative SMILES notation, hinders the performance of general-domain LLMs in chemistry. To this end, we develop ChemDFM, the first LLM towards CGI. ChemDFM-13B is trained on 34B tokens from chemical literature, textbooks, and instructions as well as various data from the general domain. Therefore, it can store, understand, and reason over chemical knowledge and languages while still possessing advanced free-form language comprehension capabilities. Extensive quantitative evaluation shows that ChemDFM can significantly outperform the representative open-sourced LLMs. Moreover, ChemDFM can also
    
[^7]: UniMS-RAG: 用于个性化对话系统的统一多源检索增强生成模型

    UniMS-RAG: A Unified Multi-source Retrieval-Augmented Generation for Personalized Dialogue Systems. (arXiv:2401.13256v1 [cs.CL])

    [http://arxiv.org/abs/2401.13256](http://arxiv.org/abs/2401.13256)

    这项研究提出了一种统一多源检索增强生成系统（UniMS-RAG），通过统一知识源选择、知识检索和回复生成三个子任务，使语言模型能够根据需求自适应地检索证据和评估关联性，从而生成个性化的回复。

    

    大型语言模型在许多自然语言理解和生成任务中展示出了非凡的能力。然而，在对话系统中涉及到多个信息源时，个性化问题仍然是一个令人向往的属性。为了更好地计划和整合多个信息源在生成个性化回复中的使用，我们首先将其分解为三个子任务：知识源选择、知识检索和回复生成。然后，我们提出了一种新颖的统一多源检索增强生成系统（UniMS-RAG）。具体来说，我们在训练期间使用相同的序列到序列范式将这三个子任务统一起来，通过使用特殊的令牌，即行动令牌和评估令牌，能够自适应地检索证据并评估关联性。使语言模型能够生成行动令牌有助于与各种知识源进行交互，使其能够适应其上下文和生成个性化的回复。

    Large Language Models (LLMs) has shown exceptional capabilities in many natual language understanding and generation tasks. However, the personalization issue still remains a much-coveted property, especially when it comes to the multiple sources involved in the dialogue system. To better plan and incorporate the use of multiple sources in generating personalized response, we firstly decompose it into three sub-tasks: Knowledge Source Selection, Knowledge Retrieval, and Response Generation. We then propose a novel Unified Multi-Source Retrieval-Augmented Generation system (UniMS-RAG) Specifically, we unify these three sub-tasks with different formulations into the same sequence-to-sequence paradigm during the training, to adaptively retrieve evidences and evaluate the relevance on-demand using special tokens, called acting tokens and evaluation tokens. Enabling language models to generate acting tokens facilitates interaction with various knowledge sources, allowing them to adapt their
    
[^8]: 跨发言条件化VAE语音生成

    Cross-Utterance Conditioned VAE for Speech Generation. (arXiv:2309.04156v1 [cs.SD])

    [http://arxiv.org/abs/2309.04156](http://arxiv.org/abs/2309.04156)

    该论文提出了一种跨发言条件化变分自编码器框架，利用预训练语言模型和变分自编码器来增强语音合成的韵律，并确保自然语音生成。该框架的核心组件是跨发言CVAE，通过提取周围句子的声学、说话人和文本特征来生成上下文敏感的韵律特征，有效模拟人类韵律生成。同时，该论文还提出了两个实用算法：CUC-VAE TTS用于文本到语音合成和CUC-VAE SE用于语音编辑。

    

    由神经网络驱动的语音合成系统在多媒体制作中有着潜力，但常常面临产生有表现力的语音和无缝编辑的问题。为此，我们提出了跨发言条件化变分自编码器语音合成(CUC-VAE S2)框架，以增强韵律并确保自然语音生成。该框架利用预训练语言模型的强大表现能力和变分自编码器(VAEs)的再表达能力。CUC-VAE S2框架的核心组件是跨发言CVAE，它从周围的句子中提取声学、说话人和文本特征，以生成上下文敏感的韵律特征，更准确地模拟人类韵律生成。我们进一步提出了两个针对不同语音合成应用的实用算法：CUC-VAE TTS以进行文本到语音合成和CUC-VAE SE以进行语音编辑。CUC-VAE TTS是该框架的直接应用，使得能够将任意文本转成语音。

    Speech synthesis systems powered by neural networks hold promise for multimedia production, but frequently face issues with producing expressive speech and seamless editing. In response, we present the Cross-Utterance Conditioned Variational Autoencoder speech synthesis (CUC-VAE S2) framework to enhance prosody and ensure natural speech generation. This framework leverages the powerful representational capabilities of pre-trained language models and the re-expression abilities of variational autoencoders (VAEs). The core component of the CUC-VAE S2 framework is the cross-utterance CVAE, which extracts acoustic, speaker, and textual features from surrounding sentences to generate context-sensitive prosodic features, more accurately emulating human prosody generation. We further propose two practical algorithms tailored for distinct speech synthesis applications: CUC-VAE TTS for text-to-speech and CUC-VAE SE for speech editing. The CUC-VAE TTS is a direct application of the framework, de
    
[^9]: 自控内存系统释放大规模语言模型的无限输入容量

    Unleashing Infinite-Length Input Capacity for Large-scale Language Models with Self-Controlled Memory System. (arXiv:2304.13343v1 [cs.CL])

    [http://arxiv.org/abs/2304.13343](http://arxiv.org/abs/2304.13343)

    该论文提出了一种自控内存系统，可以使大规模语言模型能够处理任意长度的输入，从而显著提高模型的性能表现。

    

    大规模语言模型（LLMs）受制于无法处理过长的输入。为了解决这个问题，我们提出了自控内存（SCM）系统，以释放大规模语言模型的无限输入容量。我们的SCM系统由三个关键模块组成：语言模型代理、内存流和内存控制器。语言模型代理迭代地处理超长输入，并将所有历史信息存储在内存流中。内存控制器为代理提供长期存储器（归档存储器）和短期存储器（闪存），以生成精确连贯的响应。控制器确定应激活哪些来自归档存储器的记忆，并如何将它们合并到模型输入中。我们的SCM系统可以与任何LLMs集成，以使它们能够处理超长文本而无需修改或微调。实验结果表明，我们的SCM系统使得LLMs能够处理长度高达8192个令牌的输入，实现了在多个基准数据集上的最佳表现，证明了它在提高大规模语言模型性能方面的有效性。

    Large-scale Language Models (LLMs) are constrained by their inability to process lengthy inputs. To address this limitation, we propose the Self-Controlled Memory (SCM) system to unleash infinite-length input capacity for large-scale language models. Our SCM system is composed of three key modules: the language model agent, the memory stream, and the memory controller. The language model agent iteratively processes ultra-long inputs and stores all historical information in the memory stream. The memory controller provides the agent with both long-term memory (archived memory) and short-term memory (flash memory) to generate precise and coherent responses. The controller determines which memories from archived memory should be activated and how to incorporate them into the model input. Our SCM system can be integrated with any LLMs to enable them to process ultra-long texts without any modification or fine-tuning. Experimental results show that our SCM system enables LLMs, which are not
    
[^10]: LLaMA-Adapter: 零初始化注意力下的语言模型精细调整的高效方法

    LLaMA-Adapter: Efficient Fine-tuning of Language Models with Zero-init Attention. (arXiv:2303.16199v1 [cs.CV])

    [http://arxiv.org/abs/2303.16199](http://arxiv.org/abs/2303.16199)

    本文提出了一种基于适应提示和零初始化注意力机制的轻量级语言模型调整方法，可高效微调LLaMA为指令跟随模型，具有比Alpaca更短的微调时间并具有近似的响应质量。

    

    本文提出了LLaMA-Adapter这一轻量级适应方法，用于将LLaMA高效地微调为一个指令跟随模型。利用52K个自我指导示范，LLaMA-Adapter仅在冻结的LLaMA 7B模型上引入了1.2M个可学习参数，并且在8个A100 GPU上仅耗时不到一个小时进行微调。具体而言，我们采用一组可学习的适应提示，并在较高的变压器层中将它们预置于输入文本令牌之前。然后，提出了一种零初始化注意力机制和零门控机制，该机制可以自适应地将新的指令提示注入LLaMA，并有效地保留了其预先训练的知识。通过高效训练，LLaMA-Adapter能够产生高质量的响应，与完全微调的7B参数的Alpaca相似。此外，我们的方法还可以简单地扩展到多模态输入，例如图像，用于图像相关的LLaMA，在ScienceQA上实现了更强的推理能力。我们在https://github.com/ZrrSkywalker/LLaMA-Adapt发布了我们的代码。

    We present LLaMA-Adapter, a lightweight adaption method to efficiently fine-tune LLaMA into an instruction-following model. Using 52K self-instruct demonstrations, LLaMA-Adapter only introduces 1.2M learnable parameters upon the frozen LLaMA 7B model, and costs less than one hour for fine-tuning on 8 A100 GPUs. Specifically, we adopt a set of learnable adaption prompts, and prepend them to the input text tokens at higher transformer layers. Then, a zero-init attention mechanism with zero gating is proposed, which adaptively injects the new instructional cues into LLaMA, while effectively preserves its pre-trained knowledge. With efficient training, LLaMA-Adapter generates high-quality responses, comparable to Alpaca with fully fine-tuned 7B parameters. Furthermore, our approach can be simply extended to multi-modal input, e.g., images, for image-conditioned LLaMA, which achieves superior reasoning capacity on ScienceQA. We release our code at https://github.com/ZrrSkywalker/LLaMA-Adapt
    
[^11]: MM-SHAP：一种用于衡量视觉与语言模型和任务的多模态贡献的性能不可知度量

    MM-SHAP: A Performance-agnostic Metric for Measuring Multimodal Contributions in Vision and Language Models & Tasks. (arXiv:2212.08158v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2212.08158](http://arxiv.org/abs/2212.08158)

    该论文提出了一种性能不可知的多模态得分方法MM-SHAP，可以可靠地量化多模态模型使用各自模态的比例，并应用于比较模型的平均多模态程度和衡量个体模型的贡献。实验结果表明单模态崩溃比以前认为的更为普遍，而MM-SHAP是分析VL模型多模态行为的有效工具。

    

    已知视觉和语言模型（VL）往往利用各自模态中的不稳定指标（例如由分布偏差引入）而不是专注于每个模态中的相关信息。如果单模态模型在VL任务上达到类似多模态模型的准确度，则表明所谓的单模态崩溃已经发生。然而，基于准确度的测试无法检测例如模型预测错误但模型使用了一个模态的相关信息。因此，我们提出了MM-SHAP，一种基于Shapley值的性能不可知多模态得分，可可靠地量化多模态模型使用各自模态的比例。我们将MM-SHAP应用于两种方式：（1）比较模型的平均多模态程度，（2）衡量不同任务和数据集的个体模型对各自模态的贡献。六个VL模型的实验（LXMERT、CLIP和四个ALBEF变体）表明单模态崩溃比我们以前认为的更为普遍。我们的结果还表明，MM-SHAP是揭示和分析VL模型多模态行为的有效工具。

    Vision and language models (VL) are known to exploit unrobust indicators in individual modalities (e.g., introduced by distributional biases) instead of focusing on relevant information in each modality. That a unimodal model achieves similar accuracy on a VL task to a multimodal one, indicates that so-called unimodal collapse occurred. However, accuracy-based tests fail to detect e.g., when the model prediction is wrong, while the model used relevant information from a modality. Instead, we propose MM-SHAP, a performance-agnostic multimodality score based on Shapley values that reliably quantifies in which proportions a multimodal model uses individual modalities. We apply MM-SHAP in two ways: (1) to compare models for their average degree of multimodality, and (2) to measure for individual models the contribution of individual modalities for different tasks and datasets. Experiments with six VL models -- LXMERT, CLIP and four ALBEF variants -- on four VL tasks highlight that unimodal
    

