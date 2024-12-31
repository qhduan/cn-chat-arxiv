# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [AVIBench: Towards Evaluating the Robustness of Large Vision-Language Model on Adversarial Visual-Instructions](https://arxiv.org/abs/2403.09346) | AVIBench是一个框架，用于分析大型视觉-语言模型对抗各种形式的对抗性视觉指令的鲁棒性，包括图像和文本攻击以及内容偏见攻击。 |
| [^2] | [ArgMed-Agents: Explainable Clinical Decision Reasoning with Large Language Models via Argumentation Schemes](https://arxiv.org/abs/2403.06294) | 通过争论方案的自我论证迭代和构建争论过程，ArgMed-Agents实现了基于LLM的可解释临床决策推理，提高了用户对临床决策的信任。 |
| [^3] | [Yi: Open Foundation Models by 01.AI](https://arxiv.org/abs/2403.04652) | Yi模型系列基于强大的多维能力，通过基于6B和34B预训练模型的扩展，包括聊天模型、长上下文模型、深度放大模型和视觉语言模型，取得了优异的性能。 |
| [^4] | [Interpreting Grokked Transformers in Complex Modular Arithmetic](https://arxiv.org/abs/2402.16726) | 本研究通过可解释的逆向工程在复杂模块化算术中观察了Transformer内部电路学习过程，并发现减法在Transformer上造成了强烈的不对称性，乘法需要余弦偏置分量，多项式叠加了基本算术模式，但在挑战性情况下并不清晰，Grokking甚至可以在具有基本对称和交替表达式的高次公式中轻松发生。 |
| [^5] | [Beyond RMSE and MAE: Introducing EAUC to unmask hidden bias and unfairness in dyadic regression models.](http://arxiv.org/abs/2401.10690) | 本研究引入EAUC作为一种新的度量标准，用以揭示迪亚德回归模型中隐藏的偏见和不公平问题。传统的全局错误度量标准如RMSE和MAE无法捕捉到这种问题。 |
| [^6] | [In-Context Learning with Iterative Demonstration Selection.](http://arxiv.org/abs/2310.09881) | 这项研究提出了一种基于迭代示范选择的上下文学习方法，通过使用零样本链式思维推理来选择与测试样本不同但仍与之强相关的示范作为学习的上下文。 |
| [^7] | [Deep Neural Networks and Brain Alignment: Brain Encoding and Decoding (Survey).](http://arxiv.org/abs/2307.10246) | 本文综述了深度神经网络和脑对齐的研究，重点在于脑编码和解码模型的应用。这些模型对于理解大脑的信息处理机制以及设计脑机接口具有重要意义。 |

# 详细

[^1]: AVIBench：评估大型视觉-语言模型在对抗性视觉指导上的鲁棒性

    AVIBench: Towards Evaluating the Robustness of Large Vision-Language Model on Adversarial Visual-Instructions

    [https://arxiv.org/abs/2403.09346](https://arxiv.org/abs/2403.09346)

    AVIBench是一个框架，用于分析大型视觉-语言模型对抗各种形式的对抗性视觉指令的鲁棒性，包括图像和文本攻击以及内容偏见攻击。

    

    大型视觉-语言模型（LVLMs）在对用户的视觉指令作出良好响应方面取得了显著进展。然而，这些指令涵盖图像和文本，容易受到有意和无意攻击的影响。尽管LVLMs对抗此类威胁的鲁棒性至关重要，但当前该领域的研究仍然有限。为弥补这一差距，我们引入了AVIBench，一个旨在分析LVLMs在面对各种对抗性视觉指令（AVIs）时的鲁棒性的框架，包括四种基于图像的AVIs、十种基于文本的AVIs和九种内容偏见AVIs（如性别、暴力、文化和种族偏见等）。我们生成了26万个AVIs，涵盖五类多模态能力（九项任务）和内容偏见。然后，我们对包括14个开源LVLMs在内的模型进行全面评估以评估其性能。AVIBench还可作为一个便利的工具

    arXiv:2403.09346v1 Announce Type: cross  Abstract: Large Vision-Language Models (LVLMs) have shown significant progress in well responding to visual-instructions from users. However, these instructions, encompassing images and text, are susceptible to both intentional and inadvertent attacks. Despite the critical importance of LVLMs' robustness against such threats, current research in this area remains limited. To bridge this gap, we introduce AVIBench, a framework designed to analyze the robustness of LVLMs when facing various adversarial visual-instructions (AVIs), including four types of image-based AVIs, ten types of text-based AVIs, and nine types of content bias AVIs (such as gender, violence, cultural, and racial biases, among others). We generate 260K AVIs encompassing five categories of multimodal capabilities (nine tasks) and content bias. We then conduct a comprehensive evaluation involving 14 open-source LVLMs to assess their performance. AVIBench also serves as a convenie
    
[^2]: ArgMed-Agents: 使用争议方案通过大型语言模型解释性临床决策推理

    ArgMed-Agents: Explainable Clinical Decision Reasoning with Large Language Models via Argumentation Schemes

    [https://arxiv.org/abs/2403.06294](https://arxiv.org/abs/2403.06294)

    通过争论方案的自我论证迭代和构建争论过程，ArgMed-Agents实现了基于LLM的可解释临床决策推理，提高了用户对临床决策的信任。

    

    arXiv:2403.06294v1 公告类型: 新摘要: 使用大型语言模型（LLMs）进行临床推理存在两个主要障碍。首先，虽然LLMs在自然语言处理（NLP）任务中显示出巨大的潜力，但在复杂推理和规划方面的表现却不尽人意。其次，LLMs使用不可解释的方法进行临床决策，这与临床医生的认知过程本质上不同，导致用户不信任。在本文中，我们提出了一个名为ArgMed-Agents的多代理框架，旨在通过交互使基于LLM的代理能够进行可解释的临床决策推理。ArgMed-Agents通过临床决策论据（一种模拟临床决策认知过程的推理机制）执行自论证迭代，然后将争论过程构建为表示冲突关系的有向图。最终，Reasoner（一种符号求解器）识别出一个

    arXiv:2403.06294v1 Announce Type: new  Abstract: There are two main barriers to using large language models (LLMs) in clinical reasoning. Firstly, while LLMs exhibit significant promise in Natural Language Processing (NLP) tasks, their performance in complex reasoning and planning falls short of expectations. Secondly, LLMs use uninterpretable methods to make clinical decisions that are fundamentally different from the clinician's cognitive processes. This leads to user distrust. In this paper, we present a multi-agent framework called ArgMed-Agents, which aims to enable LLM-based agents to make explainable clinical decision reasoning through interaction. ArgMed-Agents performs self-argumentation iterations via Argumentation Scheme for Clinical Decision (a reasoning mechanism for modeling cognitive processes in clinical reasoning), and then constructs the argumentation process as a directed graph representing conflicting relationships. Ultimately, Reasoner(a symbolic solver) identify a
    
[^3]: Yi: 由 01.AI 推出的开放基础模型

    Yi: Open Foundation Models by 01.AI

    [https://arxiv.org/abs/2403.04652](https://arxiv.org/abs/2403.04652)

    Yi模型系列基于强大的多维能力，通过基于6B和34B预训练模型的扩展，包括聊天模型、长上下文模型、深度放大模型和视觉语言模型，取得了优异的性能。

    

    我们介绍了Yi模型系列，这是一系列具有强大多维能力的语言和多模态模型。Yi模型系列基于6B和34B的预训练语言模型，然后我们将它们扩展为聊天模型、200K长上下文模型、深度放大模型和视觉语言模型。我们的基础模型在诸如MMLU之类的各种基准测试中表现出色，而我们微调过的聊天模型在AlpacaEval和Chatbot Arena等主要评估平台上具有较高的人类偏好率。通过依赖于我们的可扩展超级计算基础设施和经典的Transformer架构，我们认为Yi模型的性能主要归因于其数据质量，这是由我们的数据工程工作所带来的。对于预训练，我们使用级联的数据去重和质量过滤流水线构建了3100亿个英文和中文语料库的标记。对于微调，我们对小规模模型进行了改进

    arXiv:2403.04652v1 Announce Type: cross  Abstract: We introduce the Yi model family, a series of language and multimodal models that demonstrate strong multi-dimensional capabilities. The Yi model family is based on 6B and 34B pretrained language models, then we extend them to chat models, 200K long context models, depth-upscaled models, and vision-language models. Our base models achieve strong performance on a wide range of benchmarks like MMLU, and our finetuned chat models deliver strong human preference rate on major evaluation platforms like AlpacaEval and Chatbot Arena. Building upon our scalable super-computing infrastructure and the classical transformer architecture, we attribute the performance of Yi models primarily to its data quality resulting from our data-engineering efforts. For pretraining, we construct 3.1 trillion tokens of English and Chinese corpora using a cascaded data deduplication and quality filtering pipeline. For finetuning, we polish a small scale (less th
    
[^4]: 在复杂模块化算术中解释理解的Transformer

    Interpreting Grokked Transformers in Complex Modular Arithmetic

    [https://arxiv.org/abs/2402.16726](https://arxiv.org/abs/2402.16726)

    本研究通过可解释的逆向工程在复杂模块化算术中观察了Transformer内部电路学习过程，并发现减法在Transformer上造成了强烈的不对称性，乘法需要余弦偏置分量，多项式叠加了基本算术模式，但在挑战性情况下并不清晰，Grokking甚至可以在具有基本对称和交替表达式的高次公式中轻松发生。

    

    Grokking一直是解开延迟泛化之谜的积极探索。在已解密模型中识别可解释的算法是理解其机制的暗示性线索。在这项工作中，除了最简单和广为研究的模块化加法外，我们通过可解释的逆向工程观察了通过Grokking在复杂模块化算术中学到的内部电路，突出显示了它们动力学上的重大差异：减法对Transformer产生强烈的不对称性；乘法在傅立叶域的所有频率上需要余弦偏置分量；多项式通常导致基本算术模式的叠加，但在挑战性情况下清晰的模式并不显现；即使在具有基本对称和交替表达式的高次公式中，Grokking也很容易发生。我们还引入了模块化算术的新颖进展度量；傅立叶频率

    arXiv:2402.16726v2 Announce Type: replace-cross  Abstract: Grokking has been actively explored to reveal the mystery of delayed generalization. Identifying interpretable algorithms inside the grokked models is a suggestive hint to understanding its mechanism. In this work, beyond the simplest and well-studied modular addition, we observe the internal circuits learned through grokking in complex modular arithmetic via interpretable reverse engineering, which highlights the significant difference in their dynamics: subtraction poses a strong asymmetry on Transformer; multiplication requires cosine-biased components at all the frequencies in a Fourier domain; polynomials often result in the superposition of the patterns from elementary arithmetic, but clear patterns do not emerge in challenging cases; grokking can easily occur even in higher-degree formulas with basic symmetric and alternating expressions. We also introduce the novel progress measure for modular arithmetic; Fourier Freque
    
[^5]: 超越RMSE和MAE：引入EAUC来揭示偏见和不公平的迪亚德回归模型中的隐藏因素

    Beyond RMSE and MAE: Introducing EAUC to unmask hidden bias and unfairness in dyadic regression models. (arXiv:2401.10690v1 [cs.LG])

    [http://arxiv.org/abs/2401.10690](http://arxiv.org/abs/2401.10690)

    本研究引入EAUC作为一种新的度量标准，用以揭示迪亚德回归模型中隐藏的偏见和不公平问题。传统的全局错误度量标准如RMSE和MAE无法捕捉到这种问题。

    

    迪亚德回归模型用于预测一对实体的实值结果，在许多领域中都是基础的（例如，在推荐系统中预测用户对产品的评分），在许多其他领域中也有许多潜力但尚未深入探索（例如，在个性化药理学中近似确定患者的适当剂量）。本研究中，我们证明个体实体观察值分布的非均匀性导致了最先进模型中的严重偏见预测，偏向于实体的观察过去值的平均值，并在另类但同样重要的情况下提供比随机预测更差的预测能力。我们表明，全局错误度量标准如均方根误差（RMSE）和平均绝对误差（MAE）不足以捕捉到这种现象，我们将其命名为另类偏见，并引入另类-曲线下面积（EAUC）作为一个新的补充度量，可以在所有研究的模型中量化它。

    Dyadic regression models, which predict real-valued outcomes for pairs of entities, are fundamental in many domains (e.g. predicting the rating of a user to a product in Recommender Systems) and promising and under exploration in many others (e.g. approximating the adequate dosage of a drug for a patient in personalized pharmacology). In this work, we demonstrate that non-uniformity in the observed value distributions of individual entities leads to severely biased predictions in state-of-the-art models, skewing predictions towards the average of observed past values for the entity and providing worse-than-random predictive power in eccentric yet equally important cases. We show that the usage of global error metrics like Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE) is insufficient to capture this phenomenon, which we name eccentricity bias, and we introduce Eccentricity-Area Under the Curve (EAUC) as a new complementary metric that can quantify it in all studied models
    
[^6]: 基于迭代示范选择的上下文学习

    In-Context Learning with Iterative Demonstration Selection. (arXiv:2310.09881v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2310.09881](http://arxiv.org/abs/2310.09881)

    这项研究提出了一种基于迭代示范选择的上下文学习方法，通过使用零样本链式思维推理来选择与测试样本不同但仍与之强相关的示范作为学习的上下文。

    

    受到规模的进展的推动，大型语言模型(LLMs)通过上下文学习(ICL)展示了强大的少样本学习能力。然而，ICL的性能已被证明对于示范的选择非常敏感。选择最合适的示范作为上下文仍然是一个持续挑战和一个未解决的问题。现有文献已经强调了选择那些与测试样本不同或语义相似性的示范的重要性，而忽视了最优示范选择维度是任务特定的事实。借鉴两个维度的优点，我们提出了迭代示范选择(IDS)。使用零样本链式思维推理(Zero-shot-CoT)，IDS迭代地选择那些与测试样本不同但仍与之强相关的示范作为ICL的示范。具体而言，IDS在示范选择之前将Zero-shot-CoT应用于测试样本。输出的推理路径是...

    Spurred by advancements in scale, large language models (LLMs) have demonstrated strong few-shot learning ability via in-context learning (ICL). However, the performance of ICL has been shown to be highly sensitive to the selection of few-shot demonstrations. Selecting the most suitable examples as context remains an ongoing challenge and an open problem. Existing literature has highlighted the importance of selecting examples that are diverse or semantically similar to the test sample while ignoring the fact that the optimal selection dimension, i.e., diversity or similarity, is task-specific. Leveraging the merits of both dimensions, we propose Iterative Demonstration Selection (IDS). Using zero-shot chain-of-thought reasoning (Zero-shot-CoT), IDS iteratively selects examples that are diverse but still strongly correlated with the test sample as ICL demonstrations. Specifically, IDS applies Zero-shot-CoT to the test sample before demonstration selection. The output reasoning path is 
    
[^7]: 深度神经网络和脑对齐：脑编码和解码（综述）

    Deep Neural Networks and Brain Alignment: Brain Encoding and Decoding (Survey). (arXiv:2307.10246v1 [q-bio.NC])

    [http://arxiv.org/abs/2307.10246](http://arxiv.org/abs/2307.10246)

    本文综述了深度神经网络和脑对齐的研究，重点在于脑编码和解码模型的应用。这些模型对于理解大脑的信息处理机制以及设计脑机接口具有重要意义。

    

    大脑如何表示不同的信息模式？我们能否设计出一个可以自动理解用户思考内容的系统？这些问题可以通过研究功能磁共振成像（fMRI）等大脑记录来回答。作为第一步，神经科学界为被动阅读/听觉/观看概念词汇、叙述、图片和电影相关的认知神经科学数据集作出了贡献。过去二十年中，还提出了使用这些数据集的编码和解码模型。这些模型作为基础研究中的额外工具，在认知科学和神经科学领域有着多种实际应用。编码模型旨在自动地生成fMRI大脑表征，给定一个刺激。它们在评估和诊断神经系统疾病以及设计大脑损伤治疗方法方面有着多种实际应用。解码模型解决了根据fMRI重构刺激的逆问题。它们对于理解大脑如何处理信息以及设计脑机接口的发展都有着重要意义。

    How does the brain represent different modes of information? Can we design a system that automatically understands what the user is thinking? Such questions can be answered by studying brain recordings like functional magnetic resonance imaging (fMRI). As a first step, the neuroscience community has contributed several large cognitive neuroscience datasets related to passive reading/listening/viewing of concept words, narratives, pictures and movies. Encoding and decoding models using these datasets have also been proposed in the past two decades. These models serve as additional tools for basic research in cognitive science and neuroscience. Encoding models aim at generating fMRI brain representations given a stimulus automatically. They have several practical applications in evaluating and diagnosing neurological conditions and thus also help design therapies for brain damage. Decoding models solve the inverse problem of reconstructing the stimuli given the fMRI. They are useful for 
    

