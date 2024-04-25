# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Poro 34B and the Blessing of Multilinguality](https://arxiv.org/abs/2404.01856) | 多语种训练的Poro 34B模型在芬兰语等小语种上取得了显著进展，并具有比现有模型更出色的能力。 |
| [^2] | [Large Language Models in Biomedical and Health Informatics: A Bibliometric Review](https://arxiv.org/abs/2403.16303) | LLMs已成为生物医学与健康信息学中重要的工具，本文献计量学综述全面展示了LLMs在各种BHI领域中的应用，提出了其对自然语言处理应用的改进，揭示了主要发展趋势和研究网络，并讨论了伦理关切和实际挑战。 |
| [^3] | [Faster and Lighter LLMs: A Survey on Current Challenges and Way Forward](https://arxiv.org/abs/2402.01799) | 本调查文章概述了在提高LLM推理效果方面的最新方法和进展，通过实验评估不同压缩技术的有效性，并提出改进LLM推理效率的潜在未来方向。 |
| [^4] | [Can LLM-Generated Misinformation Be Detected?](https://arxiv.org/abs/2309.13788) | LLM生成的虚假信息可能比人类撰写的虚假信息更难以检测，具有更具欺骗性的风格，可能造成更多危害。 |
| [^5] | [LLMCheckup: Conversational Examination of Large Language Models via Interpretability Tools.](http://arxiv.org/abs/2401.12576) | LLMCheckup是一个可解释性工具，通过连接大型语言模型与可解释的AI工具，使用户能够与模型进行对话，生成自我解释并提供建议。 |
| [^6] | [A Survey of Graph Meets Large Language Model: Progress and Future Directions.](http://arxiv.org/abs/2311.12399) | 本综述对将大型语言模型(LLMs)与图结合的现有方法进行了全面的回顾和分析，提出了一个新的分类法，并讨论了未来研究的有希望的方向。 |
| [^7] | [DEFT: Data Efficient Fine-Tuning for Large Language Models via Unsupervised Core-Set Selection.](http://arxiv.org/abs/2310.16776) | 这项研究介绍了一种名为DEFT的数据高效微调框架，通过无监督核心集选择来最小化微调大规模语言模型所需的数据量。研究结果表明，DEFT模型在准确性上与现有模型相当，并且仅使用了70%的数据量。 |
| [^8] | [Never Train from Scratch: Fair Comparison of Long-Sequence Models Requires Data-Driven Priors.](http://arxiv.org/abs/2310.02980) | 本文研究表明使用随机初始化会导致对架构差异的严重高估，而使用标准消噪目标进行预训练可以在多种架构上实现显著的性能提升，并将Transformers与状态空间模型之间的差距缩小到很小。与之前的研究不同的是，我们发现当正确预训练时，普通的Transformers在Long Range Arena上的性能与S4相匹配，并且在PathX-256任务上改进了SSMs的最佳结果20个百分点。 |
| [^9] | [Emotionally Numb or Empathetic? Evaluating How LLMs Feel Using EmotionBench.](http://arxiv.org/abs/2308.03656) | 通过利用心理学中的情感评估理论，本研究提出利用EmotionBench评估LLMs的共情能力。通过人类评估和对五个LLMs的研究发现，尽管存在一些不一致之处，LLMs通常能在某些情境下适当地回应，但与情感对齐方面还存在不足。 |
| [^10] | [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration.](http://arxiv.org/abs/2306.00978) | AWQ是一种激活感知的权重量化方法，通过保护少量显著权重来降低量化误差，不依赖于反向传播或重构，并在语言建模和领域特定任务上优于现有方法。 |

# 详细

[^1]: Poro 34B和多语种的祝福

    Poro 34B and the Blessing of Multilinguality

    [https://arxiv.org/abs/2404.01856](https://arxiv.org/abs/2404.01856)

    多语种训练的Poro 34B模型在芬兰语等小语种上取得了显著进展，并具有比现有模型更出色的能力。

    

    最先进大型语言模型的预训练现在需要数万亿字的文本，这比绝大多数语言可获得的文本数量多几个数量级。尽管包含多种语言的文本是获取更多预训练数据的明显方法，但多语种往往被视为一种诅咒，大多数模型训练工作仍然主要集中在个别大语种上。我们相信多语种可以是一种祝福，并且应该有可能通过多语种训练显著提高小语种的模型能力。在这项研究中，我们介绍了Poro 34B，这是一个在1万亿个芬兰语、英语和编程语言标记上进行训练的拥有340亿参数的模型，并证明了多语种训练方法可以产生一个模型，不仅在芬兰语的现有模型能力上取得了显著进展，而且在表现方面表现出色。

    arXiv:2404.01856v1 Announce Type: new  Abstract: The pretraining of state-of-the-art large language models now requires trillions of words of text, which is orders of magnitude more than available for the vast majority of languages. While including text in more than one language is an obvious way to acquire more pretraining data, multilinguality is often seen as a curse, and most model training efforts continue to focus near-exclusively on individual large languages. We believe that multilinguality can be a blessing and that it should be possible to substantially improve over the capabilities of monolingual models for small languages through multilingual training. In this study, we introduce Poro 34B, a 34 billion parameter model trained for 1 trillion tokens of Finnish, English, and programming languages, and demonstrate that a multilingual training approach can produce a model that not only substantially advances over the capabilities of existing models for Finnish, but also excels i
    
[^2]: 生物医学与健康信息学中的大型语言模型：一项文献计量学综述

    Large Language Models in Biomedical and Health Informatics: A Bibliometric Review

    [https://arxiv.org/abs/2403.16303](https://arxiv.org/abs/2403.16303)

    LLMs已成为生物医学与健康信息学中重要的工具，本文献计量学综述全面展示了LLMs在各种BHI领域中的应用，提出了其对自然语言处理应用的改进，揭示了主要发展趋势和研究网络，并讨论了伦理关切和实际挑战。

    

    大型语言模型（LLMs）迅速成为生物医学与健康信息学（BHI）中的重要工具，为分析数据、治疗患者和开展研究提供了新的方式。本文献计量学综述旨在通过检查自2022年至2023年的研究文章和合作网络，全面展示LLMs在BHI中的应用情况。它进一步探讨了LLMs如何可以改进各种BHI领域中的自然语言处理（NLP）应用，如医学诊断、患者参与、电子健康记录管理和个性化医学。为此，我们的文献计量学综述确定了关键趋势，绘制了研究网络，并突出了这个快速发展领域的主要进展。最后，它讨论了在BHI中使用LLMs的伦理关切和实际挑战，如数据隐私和可靠的医疗建议。展望未来，我们考虑LLMs如何进一步改变生物医学研究。

    arXiv:2403.16303v1 Announce Type: cross  Abstract: Large Language Models (LLMs) have rapidly become important tools in Biomedical and Health Informatics (BHI), enabling new ways to analyze data, treat patients, and conduct research. This bibliometric review aims to provide a panoramic view of how LLMs have been used in BHI by examining research articles and collaboration networks from 2022 to 2023. It further explores how LLMs can improve Natural Language Processing (NLP) applications in various BHI areas like medical diagnosis, patient engagement, electronic health record management, and personalized medicine. To do this, our bibliometric review identifies key trends, maps out research networks, and highlights major developments in this fast-moving field. Lastly, it discusses the ethical concerns and practical challenges of using LLMs in BHI, such as data privacy and reliable medical recommendations. Looking ahead, we consider how LLMs could further transform biomedical research as we
    
[^3]: 更快更轻的LLMs：当前挑战和未来发展的调查

    Faster and Lighter LLMs: A Survey on Current Challenges and Way Forward

    [https://arxiv.org/abs/2402.01799](https://arxiv.org/abs/2402.01799)

    本调查文章概述了在提高LLM推理效果方面的最新方法和进展，通过实验评估不同压缩技术的有效性，并提出改进LLM推理效率的潜在未来方向。

    

    尽管LLMs表现出色，但由于推理过程中需要大量的计算和内存资源，它们的普及面临着挑战。最近在模型压缩和系统级优化方法方面的进展旨在增强LLM推理效果。本调查提供了这些方法的概述，强调了最近的发展。通过对LLaMA(/2)-7B的实验，我们评估了各种压缩技术，为在统一环境中高效部署LLM提供了实践见解。对LLaMA(/2)-7B的实证分析突出了这些方法的有效性。基于调查结果，我们确定了当前的局限性，并讨论了改善LLM推理效率的潜在未来方向。我们在https://github.com/nyunAI/Faster-LLM-Survey发布了用于复现本文结果的代码库。

    Despite the impressive performance of LLMs, their widespread adoption faces challenges due to substantial computational and memory requirements during inference. Recent advancements in model compression and system-level optimization methods aim to enhance LLM inference. This survey offers an overview of these methods, emphasizing recent developments. Through experiments on LLaMA(/2)-7B, we evaluate various compression techniques, providing practical insights for efficient LLM deployment in a unified setting. The empirical analysis on LLaMA(/2)-7B highlights the effectiveness of these methods. Drawing from survey insights, we identify current limitations and discuss potential future directions to improve LLM inference efficiency. We release the codebase to reproduce the results presented in this paper at https://github.com/nyunAI/Faster-LLM-Survey
    
[^4]: 能够检测到LLM生成的虚假信息吗?

    Can LLM-Generated Misinformation Be Detected?

    [https://arxiv.org/abs/2309.13788](https://arxiv.org/abs/2309.13788)

    LLM生成的虚假信息可能比人类撰写的虚假信息更难以检测，具有更具欺骗性的风格，可能造成更多危害。

    

    大型语言模型（LLMs）的出现产生了深远影响。然而，LLMs（如ChatGPT）可能被利用来生成虚假信息，这给在线安全和公众信任带来了严重关切。一个基本的研究问题是：LLM生成的虚假信息是否会比人类撰写的虚假信息造成更大危害?我们提出从检测难度的角度来探讨这个问题。我们首先建立了一个LLM生成的虚假信息分类法。然后，我们对利用LLMs生成虚假信息的潜在真实世界方法进行分类和验证。通过广泛的实证调查，我们发现与具有相同语义的人类撰写的虚假信息相比，LLM生成的虚假信息对人类和检测器来说更难检测，这表明它可能具有更具欺骗性的风格，潜在地造成更多危害。我们还讨论了我们发现的影响。

    arXiv:2309.13788v3 Announce Type: replace-cross  Abstract: The advent of Large Language Models (LLMs) has made a transformative impact. However, the potential that LLMs such as ChatGPT can be exploited to generate misinformation has posed a serious concern to online safety and public trust. A fundamental research question is: will LLM-generated misinformation cause more harm than human-written misinformation? We propose to tackle this question from the perspective of detection difficulty. We first build a taxonomy of LLM-generated misinformation. Then we categorize and validate the potential real-world methods for generating misinformation with LLMs. Then, through extensive empirical investigation, we discover that LLM-generated misinformation can be harder to detect for humans and detectors compared to human-written misinformation with the same semantics, which suggests it can have more deceptive styles and potentially cause more harm. We also discuss the implications of our discovery
    
[^5]: LLMCheckup：通过可解释性工具对大型语言模型进行对话式检查

    LLMCheckup: Conversational Examination of Large Language Models via Interpretability Tools. (arXiv:2401.12576v1 [cs.CL])

    [http://arxiv.org/abs/2401.12576](http://arxiv.org/abs/2401.12576)

    LLMCheckup是一个可解释性工具，通过连接大型语言模型与可解释的AI工具，使用户能够与模型进行对话，生成自我解释并提供建议。

    

    提供以对话形式进行解释的可解释性工具已经证明在增强用户理解方面具有效果，因为一次性解释有时无法提供足够的信息给用户。然而，当前基于对话的解释方案需要许多依赖项，并且不容易转移到它们未设计的任务上。通过LLMCheckup，我们提供了一个易于访问的工具，允许用户与任何最新的大型语言模型（LLM）进行对话以了解其行为。我们使LLMs能够自行生成所有解释，并通过与一系列可解释性AI（XAI）工具（例如特征归因、基于嵌入的相似性以及反事实和基于理由生成的提示策略）连接，以完成意图识别而无需微调。LLM（自我）解释以交互对话的形式呈现，支持后续问题和生成建议。

    Interpretability tools that offer explanations in the form of a dialogue have demonstrated their efficacy in enhancing users' understanding, as one-off explanations may occasionally fall short in providing sufficient information to the user. Current solutions for dialogue-based explanations, however, require many dependencies and are not easily transferable to tasks they were not designed for. With LLMCheckup, we present an easily accessible tool that allows users to chat with any state-of-the-art large language model (LLM) about its behavior. We enable LLMs to generate all explanations by themselves and take care of intent recognition without fine-tuning, by connecting them with a broad spectrum of Explainable AI (XAI) tools, e.g. feature attributions, embedding-based similarity, and prompting strategies for counterfactual and rationale generation. LLM (self-)explanations are presented as an interactive dialogue that supports follow-up questions and generates suggestions. LLMCheckup p
    
[^6]: 图遇上大型语言模型：进展与未来方向的综述

    A Survey of Graph Meets Large Language Model: Progress and Future Directions. (arXiv:2311.12399v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2311.12399](http://arxiv.org/abs/2311.12399)

    本综述对将大型语言模型(LLMs)与图结合的现有方法进行了全面的回顾和分析，提出了一个新的分类法，并讨论了未来研究的有希望的方向。

    

    图在表示和分析诸如引用网络、社交网络和生物数据等实际应用中扮演着重要角色。最近，大型语言模型(LLMs)在各个领域取得了巨大的成功，并且已经被应用于图相关任务中，超越了基于图神经网络(GNNs)的传统方法，并取得了最先进的性能。在本综述中，我们首先对将LLMs与图结合的现有方法进行全面的回顾和分析。首先，我们提出了一个新的分类法，根据LLMs在图相关任务中扮演的角色(即增强器、预测器和对齐组件)，将现有方法组织为三个类别。然后，我们系统地调查了分类法三个类别中的代表性方法。最后，我们讨论了现有研究的局限性，并突出了未来研究的有希望的方向。

    Graph plays a significant role in representing and analyzing complex relationships in real-world applications such as citation networks, social networks, and biological data. Recently, Large Language Models (LLMs), which have achieved tremendous success in various domains, have also been leveraged in graph-related tasks to surpass traditional Graph Neural Networks (GNNs) based methods and yield state-of-the-art performance. In this survey, we first present a comprehensive review and analysis of existing methods that integrate LLMs with graphs. First of all, we propose a new taxonomy, which organizes existing methods into three categories based on the role (i.e., enhancer, predictor, and alignment component) played by LLMs in graph-related tasks. Then we systematically survey the representative methods along the three categories of the taxonomy. Finally, we discuss the remaining limitations of existing studies and highlight promising avenues for future research. The relevant papers are 
    
[^7]: DEFT：通过无监督核心集选择实现大规模语言模型数据高效微调

    DEFT: Data Efficient Fine-Tuning for Large Language Models via Unsupervised Core-Set Selection. (arXiv:2310.16776v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2310.16776](http://arxiv.org/abs/2310.16776)

    这项研究介绍了一种名为DEFT的数据高效微调框架，通过无监督核心集选择来最小化微调大规模语言模型所需的数据量。研究结果表明，DEFT模型在准确性上与现有模型相当，并且仅使用了70%的数据量。

    

    最近的进展使得许多预训练语言模型（PLMs）可以使用；然而，一个仍然存在的问题是微调PLMs以用于下游任务究竟需要多少数据？在这项工作中，我们介绍了DEFT，一种数据高效的微调框架，它利用无监督的核心集选择来最小化微调PLMs所需的数据量。我们在文本编辑LM的背景下展示了DEFT框架的有效性，并与最先进的文本编辑模型CoEDIT进行了比较。我们的定量和定性结果表明，DEFT模型在准确性上与CoEDIT一样，而使用的数据量要少约70%。

    Recent advances have led to the availability of many pre-trained language models (PLMs); however, a question that remains is how much data is truly needed to fine-tune PLMs for downstream tasks? In this work, we introduce DEFT, a data-efficient fine-tuning framework that leverages unsupervised core-set selection to minimize the amount of data needed to fine-tune PLMs for downstream tasks. We demonstrate the efficacy of our DEFT framework in the context of text-editing LMs, and compare to the state-of-the art text-editing model, CoEDIT. Our quantitative and qualitative results demonstrate that DEFT models are just as accurate as CoEDIT while being finetuned on ~70% less data.
    
[^8]: 永远不要从头开始训练：公正比较长序列模型需要数据驱动的先验知识

    Never Train from Scratch: Fair Comparison of Long-Sequence Models Requires Data-Driven Priors. (arXiv:2310.02980v1 [cs.LG])

    [http://arxiv.org/abs/2310.02980](http://arxiv.org/abs/2310.02980)

    本文研究表明使用随机初始化会导致对架构差异的严重高估，而使用标准消噪目标进行预训练可以在多种架构上实现显著的性能提升，并将Transformers与状态空间模型之间的差距缩小到很小。与之前的研究不同的是，我们发现当正确预训练时，普通的Transformers在Long Range Arena上的性能与S4相匹配，并且在PathX-256任务上改进了SSMs的最佳结果20个百分点。

    

    建模序列之间的长程依赖一直是机器学习中的目标，并导致了一些架构，如状态空间模型，在处理长序列时比Transformers有显著的优势。然而，这些令人印象深刻的经验性进展主要是在随机初始化并通过预测输入序列的目标标签进行训练的基准测试（例如Long Range Arena）上展示出来的。在这项工作中，我们展示了随机初始化导致对架构之间差异的严重高估，并且使用标准消噪目标进行预训练（仅使用下游任务数据）可以在多种架构上实现显著的收益，并且可以在Transformers和状态空间模型（SSMs）之间得到很小的差距。与之前的研究形成鲜明对比的是，我们发现当正确预训练时，普通的Transformers在Long Range Arena上与S4的性能相匹配，并且我们在PathX-256任务上将SSMs的最佳报告结果提高了20个百分点。

    Modeling long-range dependencies across sequences is a longstanding goal in machine learning and has led to architectures, such as state space models, that dramatically outperform Transformers on long sequences. However, these impressive empirical gains have been by and large demonstrated on benchmarks (e.g. Long Range Arena), where models are randomly initialized and trained to predict a target label from an input sequence. In this work, we show that random initialization leads to gross overestimation of the differences between architectures and that pretraining with standard denoising objectives, using $\textit{only the downstream task data}$, leads to dramatic gains across multiple architectures and to very small gaps between Transformers and state space models (SSMs). In stark contrast to prior works, we find vanilla Transformers to match the performance of S4 on Long Range Arena when properly pretrained, and we improve the best reported results of SSMs on the PathX-256 task by 20 
    
[^9]: 感觉麻木还是有共情能力？利用EmotionBench评估LLMs的情感能力

    Emotionally Numb or Empathetic? Evaluating How LLMs Feel Using EmotionBench. (arXiv:2308.03656v3 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2308.03656](http://arxiv.org/abs/2308.03656)

    通过利用心理学中的情感评估理论，本研究提出利用EmotionBench评估LLMs的共情能力。通过人类评估和对五个LLMs的研究发现，尽管存在一些不一致之处，LLMs通常能在某些情境下适当地回应，但与情感对齐方面还存在不足。

    

    在当代话语中，评估大型语言模型（LLMs）的拟人能力变得越来越重要。利用心理学中的情感评估理论，我们提出评估LLMs的共情能力，即它们在特定情境下感受变化的能力。通过仔细而全面的调查，我们收集了一个包含超过400种情境的数据集，这些情境已被证明对我们研究的八种情感至关重要。将这些情境分为36个因素，我们进行了一项涉及全球1200多名被试的人类评估。以人类评估结果为参考，我们评估了五个LLMs，涵盖了商业和开源模型，包括模型大小的变化，以及最新的迭代版本（如GPT-4和LLaMA-2）。我们发现，尽管存在一些不一致之处，LLMs通常能在某些情境下适当地回应。然而，它们在与情感对齐方面还存在一定不足。

    Evaluating Large Language Models' (LLMs) anthropomorphic capabilities has become increasingly important in contemporary discourse. Utilizing the emotion appraisal theory from psychology, we propose to evaluate the empathy ability of LLMs, i.e., how their feelings change when presented with specific situations. After a careful and comprehensive survey, we collect a dataset containing over 400 situations that have proven effective in eliciting the eight emotions central to our study. Categorizing the situations into 36 factors, we conduct a human evaluation involving more than 1,200 subjects worldwide. With the human evaluation results as references, our evaluation includes five LLMs, covering both commercial and open-source models, including variations in model sizes, featuring the latest iterations, such as GPT-4 and LLaMA-2. We find that, despite several misalignments, LLMs can generally respond appropriately to certain situations. Nevertheless, they fall short in alignment with the e
    
[^10]: AWQ：LLM压缩与加速的激活感知权重量化方法

    AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration. (arXiv:2306.00978v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2306.00978](http://arxiv.org/abs/2306.00978)

    AWQ是一种激活感知的权重量化方法，通过保护少量显著权重来降低量化误差，不依赖于反向传播或重构，并在语言建模和领域特定任务上优于现有方法。

    

    大语言模型(LLM)在各种任务上展现出出色的性能，但巨大的模型大小提高了为服务(内存大小)带来的硬件障碍，并降低了令牌生成速度(内存带宽)。本文提出了一种名为激活感知权重量化(AWQ)的硬件友好方法，用于LLM低比特权重量化。我们的方法基于一个观察：权重并不是等重要的；仅保护1%的显著权重就能大大降低量化误差。我们提出寻找通过观察激活值而不是权重来保护显著权重的最佳按通道缩放方法。AWQ不依赖于任何反向传播或重构，因此可以很好地保持LLM在不同领域和模式下的泛化能力，而不会过度拟合校准集。AWQ在各种语言建模和领域特定基准测试上优于现有方法。由于更好的泛化能力，它实现了优秀的量化效果。

    Large language models (LLMs) have shown excellent performance on various tasks, but the astronomical model size raises the hardware barrier for serving (memory size) and slows down token generation (memory bandwidth). In this paper, we propose Activation-aware Weight Quantization (AWQ), a hardware-friendly approach for LLM low-bit weight-only quantization. Our method is based on the observation that weights are not equally important: protecting only 1% of salient weights can greatly reduce quantization error. We then propose to search for the optimal per-channel scaling that protects the salient weights by observing the activation, not weights. AWQ does not rely on any backpropagation or reconstruction, so it can well preserve LLMs' generalization ability on different domains and modalities, without overfitting to the calibration set. AWQ outperforms existing work on various language modeling and domain-specific benchmarks. Thanks to better generalization, it achieves excellent quantiz
    

