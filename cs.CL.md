# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Unified Framework for Model Editing](https://arxiv.org/abs/2403.14236) | 这个统一框架结合了“定位和编辑”模型编辑技术，最大化保留某些向量表示并记忆新事实信息。 |
| [^2] | [RigorLLM: Resilient Guardrails for Large Language Models against Undesired Content](https://arxiv.org/abs/2403.13031) | RigorLLM提出了一种新颖的框架，旨在高效有效地调节LLMs的有害和不安全输入和输出，包括能量数据增强、最小-最大优化安全输入后缀，以及基于数据增强的鲁棒KNN与LLMs融合模型。 |
| [^3] | [MCFEND: A Multi-source Benchmark Dataset for Chinese Fake News Detection](https://arxiv.org/abs/2403.09092) | MCFEND是第一个用于中文假新闻检测的多源基准数据集，解决了单一来源数据集应用于多源新闻数据时性能下降的问题。 |
| [^4] | [Differentially Private Synthetic Data via Foundation Model APIs 2: Text](https://arxiv.org/abs/2403.01749) | 通过基础模型API，我们提出了一种名为Aug-PE的增强PE算法，以产生差分隐私合成文本数据，为解决私有文本数据共享与隐私问题提供了一种前景和可扩展的解决方案。 |
| [^5] | [Dissecting Language Models: Machine Unlearning via Selective Pruning](https://arxiv.org/abs/2403.01267) | 介绍了一种针对大型语言模型的机器去学习方法，通过选择性修剪神经元来实现去学习，发现LLMs中的神经元在特定任务中具有不同的重要性。 |
| [^6] | [Two-stage Generative Question Answering on Temporal Knowledge Graph Using Large Language Models](https://arxiv.org/abs/2402.16568) | 该论文提出了一种新颖的生成式时间知识图问答框架GenTKGQA，利用大型语言模型(LLMs)在时间知识图问答任务中的两阶段方法，即子图检索和答案生成。 |
| [^7] | [CHATATC: Large Language Model-Driven Conversational Agents for Supporting Strategic Air Traffic Flow Management](https://arxiv.org/abs/2402.14850) | 本研究探讨了如何将大型语言模型应用于非安全关键的战略交通流量管理环境，提出了一个名为CHATATC的模型，通过训练大量历史数据集实现对话系统，并测试了其查询和响应能力。 |
| [^8] | [MM-Soc: Benchmarking Multimodal Large Language Models in Social Media Platforms](https://arxiv.org/abs/2402.14154) | 该研究介绍了MM-Soc，一个旨在评估多模态大型语言模型（MLLMs）对社交媒体内容理解的综合基准，通过对十种大小变体的四个开源MLLMs进行详尽评估，发现了显著的性能差异。 |
| [^9] | [RefuteBench: Evaluating Refuting Instruction-Following for Large Language Models](https://arxiv.org/abs/2402.13463) | 本文提出了一个名为RefuteBench的基准测试，旨在评估大型语言模型对反驳指令的遵循能力，发现LLMs倾向于固执于其内部知识而无法遵从用户反馈。 |
| [^10] | [Prompt Perturbation in Retrieval-Augmented Generation based Large Language Models](https://arxiv.org/abs/2402.07179) | 本文研究了基于检索增强生成的大型语言模型（LLM）中提示扰动的影响，并引入了一种新的优化技术GGPP。通过GGPP，我们可以将LLMs的输出引导到特定的错误答案，并应对提示中的无关上下文。 |
| [^11] | [Arrows of Time for Large Language Models](https://arxiv.org/abs/2401.17505) | 这篇论文通过研究自回归大型语言模型的时间方向性，发现了模型在建模自然语言能力上存在时间上的不对称性。从信息理论的角度来看，这种差异理论上是不应该存在的。通过稀疏性和计算复杂性的考虑，提供了一个理论框架来解释这种不对称性的出现。 |
| [^12] | [APPLS: Evaluating Evaluation Metrics for Plain Language Summarization](https://arxiv.org/abs/2305.14341) | 本文提出了一个用于评估纯语言摘要的指标测试平台APPLS，并引入了一种新的指标POMME来评估PLS中的文本简化。通过对指标的分析发现，当前的指标未能始终捕捉到简化度。 |
| [^13] | [Video Understanding with Large Language Models: A Survey.](http://arxiv.org/abs/2312.17432) | 这项调查研究提供了对大型语言模型（Vid-LLMs）在视频理解中的最新进展的详细概述。Vid-LLMs的新兴能力包括开放式时空推理和常识知识，为未来的视频理解提供了有前途的方向。 |
| [^14] | [Learning a Patent-Informed Biomedical Knowledge Graph Reveals Technological Potential of Drug Repositioning Candidates.](http://arxiv.org/abs/2309.03227) | 本研究提出了一种使用药物专利和生物医学数据库相结合的方法，识别具有技术潜力和科学证据的药物再定位候选物。通过构建科学的生物医学知识图谱和基于专利的生物医学知识图谱，我们可以综合分析多种信息源，为药物再定位研究提供新的视角。 |
| [^15] | [Retrieving Texts based on Abstract Descriptions.](http://arxiv.org/abs/2305.12517) | 本研究针对语义检索问题，提出了一种基于摘要描述的文本检索模型，通过改进当前的文本嵌入方法，在标准最近邻搜索中取得了显著性能提升。 |

# 详细

[^1]: 一个统一的模型编辑框架

    A Unified Framework for Model Editing

    [https://arxiv.org/abs/2403.14236](https://arxiv.org/abs/2403.14236)

    这个统一框架结合了“定位和编辑”模型编辑技术，最大化保留某些向量表示并记忆新事实信息。

    

    模型编辑是一个不断发展的领域，专注于更新模型中嵌入的知识。在各种方法中，ROME和MEMIT作为主要的“定位和编辑”模型编辑技术脱颖而出。而MEMIT可以批量编辑记忆，ROME则一次只能改变一个事实。本文引入了一个统一的框架，将ROME和MEMIT纳入一个单一的概念框架，优化同一目标，我们称之为“保存-记忆”目标。该目标旨在在记忆新事实信息的同时保留某些选定向量的表示。具体来说，ROME使用等式约束优化此目标，而MEMIT采用更灵活的最小二乘约束。除了批量编辑外，MEMIT还可以在多个层面编辑模型。我们将编辑的分布从多个层面分开，区别于优化目标。

    arXiv:2403.14236v1 Announce Type: cross  Abstract: Model editing is a growing area focused on updating the knowledge embedded within models. Among the various methodologies, ROME and MEMIT stand out as leading "locate-and-edit" model editing techniques. While MEMIT enables batched editing of memories, ROME is limited to changing one fact at a time. This paper introduces a unifying framework that brings ROME and MEMIT under a single conceptual umbrella, optimizing for the same goal, which we call the "preservation-memorization" objective. This objective aims to preserve the representations of certain selected vectors while memorizing the representations of new factual information. Specifically, ROME optimizes this objective using an equality constraint, whereas MEMIT employs a more flexible least-square constraint. In addition to making batched edits, MEMIT also edits the model at multiple layers. We disentangle the distribution of edits to multiple layers from the optimization objectiv
    
[^2]: RigorLLM：针对大型语言模型抵御不良内容的鲁棒防护栏

    RigorLLM: Resilient Guardrails for Large Language Models against Undesired Content

    [https://arxiv.org/abs/2403.13031](https://arxiv.org/abs/2403.13031)

    RigorLLM提出了一种新颖的框架，旨在高效有效地调节LLMs的有害和不安全输入和输出，包括能量数据增强、最小-最大优化安全输入后缀，以及基于数据增强的鲁棒KNN与LLMs融合模型。

    

    大语言模型（LLMs）的最新进展展示了其在不同领域的各种任务中的显著能力。然而，LLMs中出现的偏见以及在恶意输入下产生有害内容的潜力，尤其是对抗性攻击下，都带来了重大挑战。本文提出了面向大型语言模型的鲁棒防护栏（RigorLLM），这是一个新颖的框架，旨在高效有效地调节LLMs的有害和不安全输入和输出。通过采用多方面的方法，包括通过朗之万动力学进行基于能量的训练数据增强、通过极小极大优化针对输入优化安全后缀，以及基于我们的数据增强将鲁棒KNN与LLMs融合的基于融合的模型，RigorLLM为有害内容的调节提供了强大的解决方案。我们的实验评估

    arXiv:2403.13031v1 Announce Type: cross  Abstract: Recent advancements in Large Language Models (LLMs) have showcased remarkable capabilities across various tasks in different domains. However, the emergence of biases and the potential for generating harmful content in LLMs, particularly under malicious inputs, pose significant challenges. Current mitigation strategies, while effective, are not resilient under adversarial attacks. This paper introduces Resilient Guardrails for Large Language Models (RigorLLM), a novel framework designed to efficiently and effectively moderate harmful and unsafe inputs and outputs for LLMs. By employing a multi-faceted approach that includes energy-based training data augmentation through Langevin dynamics, optimizing a safe suffix for inputs via minimax optimization, and integrating a fusion-based model combining robust KNN with LLMs based on our data augmentation, RigorLLM offers a robust solution to harmful content moderation. Our experimental evalua
    
[^3]: MCFEND：用于中文假新闻检测的多源基准数据集

    MCFEND: A Multi-source Benchmark Dataset for Chinese Fake News Detection

    [https://arxiv.org/abs/2403.09092](https://arxiv.org/abs/2403.09092)

    MCFEND是第一个用于中文假新闻检测的多源基准数据集，解决了单一来源数据集应用于多源新闻数据时性能下降的问题。

    

    虚假新闻在各个在线来源的普遍传播对公众产生了重要影响。现有的中文假新闻检测数据集仅限于来自微博的新闻。然而，来自多个来源的虚假新闻在内容和社会背景等各个方面表现出多样性。仅在单一新闻来源上训练的方法几乎无法适用于现实场景。我们的初步实验表明，学习自一个大型中文假新闻检测数据集Weibo-21的最先进方法的F1分数，当测试数据改变为多源新闻数据时，从0.943急剧下降到0.470，未能识别超过三分之一的多源虚假新闻。为解决这一限制，我们构建了用于中文假新闻检测的第一个多源基准数据集MCFEND，由我们从各种来源收集的新闻组成。

    arXiv:2403.09092v1 Announce Type: cross  Abstract: The prevalence of fake news across various online sources has had a significant influence on the public. Existing Chinese fake news detection datasets are limited to news sourced solely from Weibo. However, fake news originating from multiple sources exhibits diversity in various aspects, including its content and social context. Methods trained on purely one single news source can hardly be applicable to real-world scenarios. Our pilot experiment demonstrates that the F1 score of the state-of-the-art method that learns from a large Chinese fake news detection dataset, Weibo-21, drops significantly from 0.943 to 0.470 when the test data is changed to multi-source news data, failing to identify more than one-third of the multi-source fake news. To address this limitation, we constructed the first multi-source benchmark dataset for Chinese fake news detection, termed MCFEND, which is composed of news we collected from diverse sources suc
    
[^4]: 通过基础模型API生成差分隐私合成数据2：文本

    Differentially Private Synthetic Data via Foundation Model APIs 2: Text

    [https://arxiv.org/abs/2403.01749](https://arxiv.org/abs/2403.01749)

    通过基础模型API，我们提出了一种名为Aug-PE的增强PE算法，以产生差分隐私合成文本数据，为解决私有文本数据共享与隐私问题提供了一种前景和可扩展的解决方案。

    

    arXiv:2403.01749v1 公告类型：新 抽象：由于学习算法的出现，文本数据变得非常有价值。现实世界中产生的许多高质量文本数据是私密的，因此由于隐私问题无法自由共享或使用。生成具有形式隐私保证（即差分隐私（DP））的私密文本数据的合成副本提供了一种有前途且可扩展的解决方案。然而，现有方法需要在私有数据上对大型语言模型（LLM）进行DP微调，以生成DP合成数据。这种方法对于专有LLM（例如GPT-3.5）并不可行，而且对于开源LLM需要相当大的计算资源。Lin等人（2024）最近引入了私有进化（PE）算法，利用扩散模型只通过API访问生成DP合成图像。在这项工作中，我们提出了增强的PE算法，名为Aug-PE，适用于文本的复杂设置。

    arXiv:2403.01749v1 Announce Type: new  Abstract: Text data has become extremely valuable due to the emergence of machine learning algorithms that learn from it. A lot of high-quality text data generated in the real world is private and therefore cannot be shared or used freely due to privacy concerns. Generating synthetic replicas of private text data with a formal privacy guarantee, i.e., differential privacy (DP), offers a promising and scalable solution. However, existing methods necessitate DP finetuning of large language models (LLMs) on private data to generate DP synthetic data. This approach is not viable for proprietary LLMs (e.g., GPT-3.5) and also demands considerable computational resources for open-source LLMs. Lin et al. (2024) recently introduced the Private Evolution (PE) algorithm to generate DP synthetic images with only API access to diffusion models. In this work, we propose an augmented PE algorithm, named Aug-PE, that applies to the complex setting of text. We use
    
[^5]: 解剖语言模型：通过选择性修剪实现机器去学习

    Dissecting Language Models: Machine Unlearning via Selective Pruning

    [https://arxiv.org/abs/2403.01267](https://arxiv.org/abs/2403.01267)

    介绍了一种针对大型语言模型的机器去学习方法，通过选择性修剪神经元来实现去学习，发现LLMs中的神经元在特定任务中具有不同的重要性。

    

    本文引入了一种专门为大型语言模型（LLMs）设计的机器去学习方法。我们提出了一种针对LLMs的选择性修剪方法，根据神经元对特定能力的相对重要性来移除神经元，而非整体网络性能。该方法是一种高效的计算和数据方法，用于识别和删除能够实现特定行为的神经元。我们的研究发现，LLMs中的前馈神经元和注意力神经元是专门化的；也就是说，对于特定任务，某些神经元比其他神经元更为关键。

    arXiv:2403.01267v1 Announce Type: cross  Abstract: Understanding and shaping the behaviour of Large Language Models (LLMs) is increasingly important as applications become more powerful and more frequently adopted. This paper introduces a machine unlearning method specifically designed for LLMs. We introduce a selective pruning method for LLMs that removes neurons based on their relative importance on a targeted capability compared to overall network performance. This approach is a compute- and data-efficient method for identifying and removing neurons that enable specific behaviours. Our findings reveal that both feed-forward and attention neurons in LLMs are specialized; that is, for specific tasks, certain neurons are more crucial than others.
    
[^6]: 使用大型语言模型在时间知识图上进行两阶段生成式问答

    Two-stage Generative Question Answering on Temporal Knowledge Graph Using Large Language Models

    [https://arxiv.org/abs/2402.16568](https://arxiv.org/abs/2402.16568)

    该论文提出了一种新颖的生成式时间知识图问答框架GenTKGQA，利用大型语言模型(LLMs)在时间知识图问答任务中的两阶段方法，即子图检索和答案生成。

    

    时间知识图问答(TKGQA)提出了一个重要的挑战任务，因为问题中隐藏着时间约束，并且从动态结构化知识中寻找答案。尽管大型语言模型(LLMs)在推理能力方面取得了相当大的进展，但它们在TKGQA任务中的应用是一个相对未开发的领域。本文首先提出了一种新颖的生成式时间知识图问答框架GenTKGQA，通过两个阶段引导LLMs回答时间性问题：子图检索和答案生成。首先，我们利用LLM的固有知识来挖掘问题中的时间约束和结构链接，无需额外训练，从而缩小了在时间和结构维度上的子图搜索空间。接下来，我们设计了虚拟知识指示器来融合子图和文本表示的图神经网络信号。

    arXiv:2402.16568v1 Announce Type: new  Abstract: Temporal knowledge graph question answering (TKGQA) poses a significant challenge task, due to the temporal constraints hidden in questions and the answers sought from dynamic structured knowledge. Although large language models (LLMs) have made considerable progress in their reasoning ability over structured data, their application to the TKGQA task is a relatively unexplored area. This paper first proposes a novel generative temporal knowledge graph question answering framework, GenTKGQA, which guides LLMs to answer temporal questions through two phases: Subgraph Retrieval and Answer Generation. First, we exploit LLM's intrinsic knowledge to mine temporal constraints and structural links in the questions without extra training, thus narrowing down the subgraph search space in both temporal and structural dimensions. Next, we design virtual knowledge indicators to fuse the graph neural network signals of the subgraph and the text repres
    
[^7]: CHATATC：用于支持战略空中交通流量管理的大型语言模型驱动的对话系统

    CHATATC: Large Language Model-Driven Conversational Agents for Supporting Strategic Air Traffic Flow Management

    [https://arxiv.org/abs/2402.14850](https://arxiv.org/abs/2402.14850)

    本研究探讨了如何将大型语言模型应用于非安全关键的战略交通流量管理环境，提出了一个名为CHATATC的模型，通过训练大量历史数据集实现对话系统，并测试了其查询和响应能力。

    

    生成人工智能（AI）和大型语言模型（LLMs）已经通过诸如ChatGPT等公开可用工具快速走红。LLMs在个人和专业领域的应用得到推动，是由于人类用户与ChatGPT等计算机应用之间自然的互动，以及强大的摘要和文本生成能力。在这项工作中，我们调查了这些生成AI工具如何在非安全关键的战略交通流量管理环境中部署。具体来说，我们基于包含超过80,000个GDP实施、修订和取消的大量历史数据集，对CHATATC进行训练。我们测试了CHATATC的查询和响应能力，记录了成功之处（例如，提供正确的GDP率、持续时间和原因）以及不足之处（例如，最高水平）

    arXiv:2402.14850v1 Announce Type: cross  Abstract: Generative artificial intelligence (AI) and large language models (LLMs) have gained rapid popularity through publicly available tools such as ChatGPT. The adoption of LLMs for personal and professional use is fueled by the natural interactions between human users and computer applications such as ChatGPT, along with powerful summarization and text generation capabilities. Given the widespread use of such generative AI tools, in this work we investigate how these tools can be deployed in a non-safety critical, strategic traffic flow management setting. Specifically, we train an LLM, CHATATC, based on a large historical data set of Ground Delay Program (GDP) issuances, spanning 2000-2023 and consisting of over 80,000 GDP implementations, revisions, and cancellations. We test the query and response capabilities of CHATATC, documenting successes (e.g., providing correct GDP rates, durations, and reason) and shortcomings (e.g,. superlative
    
[^8]: 在社交媒体平台上对多模态大型语言模型进行基准测试

    MM-Soc: Benchmarking Multimodal Large Language Models in Social Media Platforms

    [https://arxiv.org/abs/2402.14154](https://arxiv.org/abs/2402.14154)

    该研究介绍了MM-Soc，一个旨在评估多模态大型语言模型（MLLMs）对社交媒体内容理解的综合基准，通过对十种大小变体的四个开源MLLMs进行详尽评估，发现了显著的性能差异。

    

    社交媒体平台是多模态信息交流的中心，包括文本、图片和视频，这使得机器难以理解在线空间中交互所关联的信息或情绪。多模态大型语言模型（MLLMs）已经成为解决这些挑战的一个有前途的解决方案，但是它们在准确解释人类情绪和诸如虚假信息等复杂内容方面存在困难。本文介绍了MM-Soc，一个旨在评估MLLMs对多模态社交媒体内容理解的综合基准。MM-Soc整合了著名的多模态数据集，并融入了一个新颖的大规模YouTube标记数据集，旨在针对从虚假信息检测、仇恨言论检测到社交上下文生成等一系列任务。通过对四个开源MLLMs的十种不同规模变体进行详尽评估，我们发现了显著的性能差异，凸显出了对性能平衡的需求。

    arXiv:2402.14154v1 Announce Type: new  Abstract: Social media platforms are hubs for multimodal information exchange, encompassing text, images, and videos, making it challenging for machines to comprehend the information or emotions associated with interactions in online spaces. Multimodal Large Language Models (MLLMs) have emerged as a promising solution to address these challenges, yet struggle with accurately interpreting human emotions and complex contents like misinformation. This paper introduces MM-Soc, a comprehensive benchmark designed to evaluate MLLMs' understanding of multimodal social media content. MM-Soc compiles prominent multimodal datasets and incorporates a novel large-scale YouTube tagging dataset, targeting a range of tasks from misinformation detection, hate speech detection, and social context generation. Through our exhaustive evaluation on ten size-variants of four open-source MLLMs, we have identified significant performance disparities, highlighting the need
    
[^9]: RefuteBench：评估用于大型语言模型的反驳指令遵循

    RefuteBench: Evaluating Refuting Instruction-Following for Large Language Models

    [https://arxiv.org/abs/2402.13463](https://arxiv.org/abs/2402.13463)

    本文提出了一个名为RefuteBench的基准测试，旨在评估大型语言模型对反驳指令的遵循能力，发现LLMs倾向于固执于其内部知识而无法遵从用户反馈。

    

    大型语言模型（LLMs）的应用范围日益扩大。在实际使用中，用户可能根据模型的输出提供反馈，希望得到一个可以根据他们的反馈完成响应的响应模型。然而，模型能否恰当地响应用户的反驳反馈并始终执行下去尚未得到彻底分析。基于这一问题，本文提出了一个全面的基准测试，RefuteBench，涵盖了诸如问答、机器翻译和电子邮件撰写等任务。评估旨在评估模型是否能够积极接受反驳指令形式的反馈，并是否能够在对话中始终遵循用户需求。我们对众多LLMs进行了评估，并发现LLMs倾向固执，即倾向于其内部知识，经常未能遵守用户反馈。

    arXiv:2402.13463v1 Announce Type: cross  Abstract: The application scope of large language models (LLMs) is increasingly expanding. In practical use, users might provide feedback based on the model's output, hoping for a responsive model that can complete responses according to their feedback. Whether the model can appropriately respond to users' refuting feedback and consistently follow through with execution has not been thoroughly analyzed. In light of this, this paper proposes a comprehensive benchmark, RefuteBench, covering tasks such as question answering, machine translation, and email writing. The evaluation aims to assess whether models can positively accept feedback in form of refuting instructions and whether they can consistently adhere to user demands throughout the conversation. We conduct evaluations on numerous LLMs and find that LLMs are stubborn, i.e. exhibit inclination to their internal knowledge, often failing to comply with user feedback. Additionally, as the leng
    
[^10]: 在基于检索增强生成的大型语言模型中进行提示扰动

    Prompt Perturbation in Retrieval-Augmented Generation based Large Language Models

    [https://arxiv.org/abs/2402.07179](https://arxiv.org/abs/2402.07179)

    本文研究了基于检索增强生成的大型语言模型（LLM）中提示扰动的影响，并引入了一种新的优化技术GGPP。通过GGPP，我们可以将LLMs的输出引导到特定的错误答案，并应对提示中的无关上下文。

    

    大型语言模型（LLM）的鲁棒性在其在各个领域的使用迅速增长中变得越来越重要。检索增强生成（RAG）被视为提高从LLM生成文本的可信度的方法。然而，目前对RAG-based LLMs的输出如何受到稍有不同的输入影响的研究还不够充分。在本文中，我们发现即使在提示中插入一个很短的前缀也会导致生成的输出与事实正确答案相去甚远。我们系统地评估了这类前缀对RAG的影响，并引入了一种称为Gradient Guided Prompt Perturbation（GGPP）的新型优化技术。GGPP在将RAG-based LLMs的输出引导到特定错误答案方面取得了很高的成功率。它还可以应对提示中请求忽略无关上下文的指令。我们还利用LLMs在带有和不带有GGPP扰动的提示之间的神经元激活差异来提供一种改进方法。

    The robustness of large language models (LLMs) becomes increasingly important as their use rapidly grows in a wide range of domains. Retrieval-Augmented Generation (RAG) is considered as a means to improve the trustworthiness of text generation from LLMs. However, how the outputs from RAG-based LLMs are affected by slightly different inputs is not well studied. In this work, we find that the insertion of even a short prefix to the prompt leads to the generation of outputs far away from factually correct answers. We systematically evaluate the effect of such prefixes on RAG by introducing a novel optimization technique called Gradient Guided Prompt Perturbation (GGPP). GGPP achieves a high success rate in steering outputs of RAG-based LLMs to targeted wrong answers. It can also cope with instructions in the prompts requesting to ignore irrelevant context. We also exploit LLMs' neuron activation difference between prompts with and without GGPP perturbations to give a method that improves
    
[^11]: 大型语言模型中的时间箭头

    Arrows of Time for Large Language Models

    [https://arxiv.org/abs/2401.17505](https://arxiv.org/abs/2401.17505)

    这篇论文通过研究自回归大型语言模型的时间方向性，发现了模型在建模自然语言能力上存在时间上的不对称性。从信息理论的角度来看，这种差异理论上是不应该存在的。通过稀疏性和计算复杂性的考虑，提供了一个理论框架来解释这种不对称性的出现。

    

    我们通过时间方向性的视角研究了自回归大型语言模型的概率建模。我们在实证上发现这类模型在建模自然语言能力上存在时间上的不对称性：预测下一个记号和预测前一个记号时的平均对数困惑度存在差异。这种差异既微妙又在不同的模态（语言、模型大小、训练时间等）下非常一致。从信息理论的角度来看，这在理论上是令人惊讶的，不应该存在这样的差异。我们提供了一个理论框架，解释了这种不对称性如何出现在稀疏性和计算复杂性考虑中，并概述了我们的结果带来的一些展望。

    We study the probabilistic modeling performed by Autoregressive Large Language Models through the angle of time directionality. We empirically find a time asymmetry exhibited by such models in their ability to model natural language: a difference in the average log-perplexity when trying to predict the next token versus when trying to predict the previous one. This difference is at the same time subtle and very consistent across various modalities (language, model size, training time, ...). Theoretically, this is surprising: from an information-theoretic point of view, there should be no such difference. We provide a theoretical framework to explain how such an asymmetry can appear from sparsity and computational complexity considerations, and outline a number of perspectives opened by our results.
    
[^12]: APPLS: 评估纯语言摘要的评价指标

    APPLS: Evaluating Evaluation Metrics for Plain Language Summarization

    [https://arxiv.org/abs/2305.14341](https://arxiv.org/abs/2305.14341)

    本文提出了一个用于评估纯语言摘要的指标测试平台APPLS，并引入了一种新的指标POMME来评估PLS中的文本简化。通过对指标的分析发现，当前的指标未能始终捕捉到简化度。

    

    尽管对于纯语言摘要（PLS）的模型有了很大的发展，但评估仍然是一个挑战。PLS缺乏专门的评估指标，由于涉及到独特的转换（例如，添加背景解释，删除专业术语），因此对于文本生成评估指标的适用性尚不清楚。为了解决这些问题，我们的研究提出了一个细致的元评估测试平台APPLS，旨在评估PLS的指标。我们根据先前工作的启发，定义了四个标准上的一组扰动，PLS指标应该捕捉到：信息性、简化度、连贯性和忠实度。使用我们的测试平台对指标进行分析发现，当前的指标未能始终捕捉到简化度。作为回应，我们引入了一种新的指标POMME，旨在评估PLS中文本简化；该指标是根据域内和域外语言模型之间的标准化困惑度差计算得到的。我们演示了POMME的效果，并与其他指标进行了比较。

    While there has been significant development of models for Plain Language Summarization (PLS), evaluation remains a challenge. PLS lacks a dedicated assessment metric, and the suitability of text generation evaluation metrics is unclear due to the unique transformations involved (e.g., adding background explanations, removing specialized terminology). To address these concerns, our study presents a granular meta-evaluation testbed, APPLS, designed to evaluate metrics for PLS. We define a set of perturbations along four criteria inspired by previous work that a PLS metric should capture: informativeness, simplification, coherence, and faithfulness. An analysis of metrics using our testbed reveals that current metrics fail to capture simplification consistently. In response, we introduce POMME, a new metric designed to assess text simplification in PLS; the metric is calculated as the normalized perplexity difference between an in-domain and out-of-domain language model. We demonstrate P
    
[^13]: 大型语言模型在视频理解中的应用：一项调查研究

    Video Understanding with Large Language Models: A Survey. (arXiv:2312.17432v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2312.17432](http://arxiv.org/abs/2312.17432)

    这项调查研究提供了对大型语言模型（Vid-LLMs）在视频理解中的最新进展的详细概述。Vid-LLMs的新兴能力包括开放式时空推理和常识知识，为未来的视频理解提供了有前途的方向。

    

    随着在线视频平台的不断增长和视频内容的不断增多，对熟练的视频理解工具的需求显著增加。鉴于大型语言模型在语言和多模态任务中的卓越能力，本调查提供了对利用大型语言模型（Vid-LLMs）技术进行视频理解的最新进展的详细概述。Vid-LLMs的新兴能力令人惊讶，尤其是它们在开放式时空推理和常识知识方面的能力，为未来的视频理解提供了一个有前途的方向。本调查对Vid-LLMs的独特特点和能力进行了分类，分为四种主要类型：基于LLM的视频代理、Vid-LLMs的预训练、Vid-LLMs的指令调整和混合方法。此外，本调查对Vid-LLMs的任务、数据集和评估方法进行了全面的研究。另外，它还探讨了Vid-LLMs技术的局限性和未来的挑战。

    With the burgeoning growth of online video platforms and the escalating volume of video content, the demand for proficient video understanding tools has intensified markedly. Given the remarkable capabilities of Large Language Models (LLMs) in language and multimodal tasks, this survey provides a detailed overview of the recent advancements in video understanding harnessing the power of LLMs (Vid-LLMs). The emergent capabilities of Vid-LLMs are surprisingly advanced, particularly their ability for open-ended spatial-temporal reasoning combined with commonsense knowledge, suggesting a promising path for future video understanding. We examine the unique characteristics and capabilities of Vid-LLMs, categorizing the approaches into four main types: LLM-based Video Agents, Vid-LLMs Pretraining, Vid-LLMs Instruction Tuning, and Hybrid Methods. Furthermore, this survey presents a comprehensive study of the tasks, datasets, and evaluation methodologies for Vid-LLMs. Additionally, it explores 
    
[^14]: 学习基于专利的生物医学知识图谱揭示药物再定位候选物的技术潜力

    Learning a Patent-Informed Biomedical Knowledge Graph Reveals Technological Potential of Drug Repositioning Candidates. (arXiv:2309.03227v1 [cs.AI])

    [http://arxiv.org/abs/2309.03227](http://arxiv.org/abs/2309.03227)

    本研究提出了一种使用药物专利和生物医学数据库相结合的方法，识别具有技术潜力和科学证据的药物再定位候选物。通过构建科学的生物医学知识图谱和基于专利的生物医学知识图谱，我们可以综合分析多种信息源，为药物再定位研究提供新的视角。

    

    药物再定位是一种发现现有药物新治疗用途的有前途的策略，近年来在计算科学文献中使用生物医学数据库进行了广泛探索。然而，药物再定位候选物的技术潜力经常被忽视。本研究提出了一种新的方法，综合分析药物专利和生物医学数据库等多种信息源，识别具有技术潜力和科学证据的药物再定位候选物。首先，我们构建了一个科学的生物医学知识图谱（s-BKG），包括来自生物医学数据库的药物、疾病和基因之间的关系。我们的方法涉及识别在s-BKG中与目标疾病关联有限但在空间上紧密相邻的药物作为潜在的药物候选物。然后，我们通过添加药物专利信息构建了一个基于专利的生物医学知识图谱（p-BKG）。

    Drug repositioning-a promising strategy for discovering new therapeutic uses for existing drugs-has been increasingly explored in the computational science literature using biomedical databases. However, the technological potential of drug repositioning candidates has often been overlooked. This study presents a novel protocol to comprehensively analyse various sources such as pharmaceutical patents and biomedical databases, and identify drug repositioning candidates with both technological potential and scientific evidence. To this end, first, we constructed a scientific biomedical knowledge graph (s-BKG) comprising relationships between drugs, diseases, and genes derived from biomedical databases. Our protocol involves identifying drugs that exhibit limited association with the target disease but are closely located in the s-BKG, as potential drug candidates. We constructed a patent-informed biomedical knowledge graph (p-BKG) by adding pharmaceutical patent information. Finally, we d
    
[^15]: 基于摘要描述的文本检索

    Retrieving Texts based on Abstract Descriptions. (arXiv:2305.12517v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2305.12517](http://arxiv.org/abs/2305.12517)

    本研究针对语义检索问题，提出了一种基于摘要描述的文本检索模型，通过改进当前的文本嵌入方法，在标准最近邻搜索中取得了显著性能提升。

    

    虽然针对文本的信息提取，指令优化的大型语言模型表现优异，但对于在大规模文档集合中定位符合给定描述的文本（语义检索）并不适用。基于嵌入向量的相似度搜索可以通过查询执行检索，但嵌入中的相似度定义不明确且不一致，并且对于许多用例来说都是次优的。那么，什么是有效检索的好的查询表示？我们确定了根据内容的摘要描述检索句子的明确定义且一致的任务。我们展示了当前文本嵌入的不足，并提出了一种替代模型，在标准最近邻搜索中的表现显著提升。该模型使用通过提示LLM获得的正负样本对进行训练。虽然很容易从LLM中获得训练材料，但LLM无法直接执行检索任务。

    While instruction-tuned Large Language Models (LLMs) excel at extracting information from text, they are not suitable for locating texts conforming to a given description in a large document collection (semantic retrieval). Similarity search over embedding vectors does allow to perform retrieval by query, but the similarity reflected in the embedding is ill-defined and non-consistent, and is sub-optimal for many use cases. What, then, is a good query representation for effective retrieval?  We identify the well defined and consistent task of retrieving sentences based on abstract descriptions of their content. We demonstrate the inadequacy of current text embeddings and propose an alternative model that significantly improves when used in standard nearest neighbor search. The model is trained using positive and negative pairs sourced through prompting a LLM. While it is easy to source the training material from an LLM, the retrieval task cannot be performed by the LLM directly. This de
    

