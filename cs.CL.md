# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Vaccine: Perturbation-aware Alignment for Large Language Model](https://rss.arxiv.org/abs/2402.01109) | 疫苗是一种针对大规模语言模型的干扰感知对齐技术，通过逐渐添加扰动产生不变的隐藏嵌入，提高对抗有害提示引起的嵌入漂移的对齐鲁棒性，同时保留对良性提示的推理能力。 |
| [^2] | [Language Agents as Optimizable Graphs](https://arxiv.org/abs/2402.16823) | 将基于LLM的代理统一描述为计算图，提出新颖的自动图优化器来改进节点和边，实现了代理之间的自动协作和改进。 |
| [^3] | [Dependency Annotation of Ottoman Turkish with Multilingual BERT](https://arxiv.org/abs/2402.14743) | 使用多语言BERT对奥斯曼土耳其语进行依存标注，加速并简化依存标注过程，将产生的树库有助于奥斯曼土耳其语文档的自动分析。 |
| [^4] | [AI-Augmented Predictions: LLM Assistants Improve Human Forecasting Accuracy](https://arxiv.org/abs/2402.07862) | 本研究发现，使用LLMs助手可以显著提高预测准确性，不仅仅是由于模型预测准确性的提升。 |
| [^5] | [Studious Bob Fight Back Against Jailbreaking via Prompt Adversarial Tuning](https://arxiv.org/abs/2402.06255) | 本文提出了一种名为Prompt Adversarial Tuning (PAT)的方法，通过训练一个防御控制机制并将其作为前缀嵌入到用户提示中，实现对大型语言模型（LLMs）的越狱行为的防御。实验证明该方法在保护LLMs免受产生有害信息的影响方面效果显著。 |
| [^6] | [Clarify: Improving Model Robustness With Natural Language Corrections](https://arxiv.org/abs/2402.03715) | 论文提出了Clarify，一种通过自然语言纠正模型错误概念的方法，该方法通过用户提供简短的文本描述来纠正模型的一致失败模式，从而提高模型的鲁棒性。 |
| [^7] | [The Curious Case of Nonverbal Abstract Reasoning with Multi-Modal Large Language Models](https://arxiv.org/abs/2401.12117) | 本研究评估了多模态大型语言模型（MLLMs）在非语言抽象推理方面的能力，并发现开源和闭源模型之间存在巨大差距和个体模块的关键缺陷。 |
| [^8] | [Generalizing Visual Question Answering from Synthetic to Human-Written Questions via a Chain of QA with a Large Language Model.](http://arxiv.org/abs/2401.06400) | 本文提出了一种新方法CoQAH，通过连接大型语言模型和训练于合成数据上的VQA模型的QA交互序列，实现了将可视问答从合成问题泛化到人工问题，并在两种类型的人工问题数据集上取得了最先进的准确率，超过了通用视觉语言模型、VQA模型和医学基础模型。 |
| [^9] | [KLoB: a Benchmark for Assessing Knowledge Locating Methods in Language Models.](http://arxiv.org/abs/2309.16535) | KLoB是一个评估语言模型中知识定位方法的基准，旨在解决现有定位方法的准确性和事实知识局部性假设的问题。 |
| [^10] | [SPICED: News Similarity Detection Dataset with Multiple Topics and Complexity Levels.](http://arxiv.org/abs/2309.13080) | 这个论文提出了一个名为SPICED的新闻相似性检测数据集，包括七个主题，并提供了四种不同的方法来生成新闻。 |
| [^11] | [Can we trust the evaluation on ChatGPT?.](http://arxiv.org/abs/2303.12767) | 本文讨论了ChatGPT评估中面临的数据污染挑战，通过倾向性检测任务阐述了这一问题，并探讨了如何在闭合且持续训练模型的时代确保模型评估的公平性。 |

# 详细

[^1]: 疫苗：针对大规模语言模型的干扰感知对齐技术

    Vaccine: Perturbation-aware Alignment for Large Language Model

    [https://rss.arxiv.org/abs/2402.01109](https://rss.arxiv.org/abs/2402.01109)

    疫苗是一种针对大规模语言模型的干扰感知对齐技术，通过逐渐添加扰动产生不变的隐藏嵌入，提高对抗有害提示引起的嵌入漂移的对齐鲁棒性，同时保留对良性提示的推理能力。

    

    作为一种新的微调即服务范 paradigm，大型语言模型 (LLM) 为用户上传的一小部分有害数据提供了新的攻击面，这些数据很容易欺骗微调过程从而产生对齐失效的模型。我们进行了实证分析，揭示了一种可能导致对齐失效的有害嵌入漂移现象。受到我们的发现启发，我们提出了疫苗 (Vaccine) ，一种针对干扰感知的对齐技术，以减轻用户微调的安全风险。疫苗的核心思想是通过在对齐阶段逐渐添加精心设计的扰动，产生不变的隐藏嵌入，从而使嵌入能够抵御来自未经消毒的用户数据的有害扰动。我们在开源主流LLM（如Llama2，Opt，Vicuna）上的实验结果表明，疫苗能够提高对抗有害提示引起的嵌入漂移的对齐鲁棒性，同时保留对良性提示的推理能力。

    The new paradigm of finetuning-as-a-service introduces a new attack surface for Large Language Models (LLMs): a few harmful data uploaded by users can easily trick the finetuning to produce an alignment-broken model. We conduct an empirical analysis and uncover a \textit{harmful embedding drift} phenomenon, showing a probable cause of the alignment-broken effect. Inspired by our findings, we propose Vaccine, a perturbation-aware alignment technique to mitigate the security risk of users finetuning. The core idea of Vaccine is to produce invariant hidden embeddings by progressively adding crafted perturbation to them in the alignment phase. This enables the embeddings to withstand harmful perturbation from un-sanitized user data in the finetuning phase. Our results on open source mainstream LLMs (e.g., Llama2, Opt, Vicuna) demonstrate that Vaccine can boost the robustness of alignment against harmful prompts induced embedding drift while reserving reasoning ability towards benign prompt
    
[^2]: 作为可优化图的语言代理

    Language Agents as Optimizable Graphs

    [https://arxiv.org/abs/2402.16823](https://arxiv.org/abs/2402.16823)

    将基于LLM的代理统一描述为计算图，提出新颖的自动图优化器来改进节点和边，实现了代理之间的自动协作和改进。

    

    多种人类设计的提升技术被提出，用于改进基于大型语言模型（LLMs）的问题求解器，产生了许多不同的代码库。我们通过将LLM代理描述为计算图来统一这些方法。节点实现处理多模态数据或查询LLMs的功能，并且边描述操作之间的信息流动。图形可以递归地组合成代表不同代理之间协作层次的更大组合图（其中边连接不同代理的操作）。我们的新颖自动图优化器（1）优化节点级LLM提示（节点优化）并（2）通过改变图连接性来改善代理协调（边缘优化）。实验证明我们的框架可用于高效开发、集成和自动改进各种LLM代理。代码可在https://github.com/metauto-ai/gptswarm找到。

    arXiv:2402.16823v1 Announce Type: cross  Abstract: Various human-designed prompt engineering techniques have been proposed to improve problem solvers based on Large Language Models (LLMs), yielding many disparate code bases. We unify these approaches by describing LLM-based agents as computational graphs. The nodes implement functions to process multimodal data or query LLMs, and the edges describe the information flow between operations. Graphs can be recursively combined into larger composite graphs representing hierarchies of inter-agent collaboration (where edges connect operations of different agents). Our novel automatic graph optimizers (1) refine node-level LLM prompts (node optimization) and (2) improve agent orchestration by changing graph connectivity (edge optimization). Experiments demonstrate that our framework can be used to efficiently develop, integrate, and automatically improve various LLM agents. The code can be found at https://github.com/metauto-ai/gptswarm.
    
[^3]: 使用多语言BERT对奥斯曼土耳其语进行依存标注

    Dependency Annotation of Ottoman Turkish with Multilingual BERT

    [https://arxiv.org/abs/2402.14743](https://arxiv.org/abs/2402.14743)

    使用多语言BERT对奥斯曼土耳其语进行依存标注，加速并简化依存标注过程，将产生的树库有助于奥斯曼土耳其语文档的自动分析。

    

    这项研究介绍了一种基于预训练大型语言模型的标注方法，用于第一个奥斯曼土耳其语依存树库。我们的实验结果表明，通过迭代地i）使用基于多语言BERT的解析模型进行伪标注数据，ii）手动修正伪标注，以及iii）用修正的标注微调解析模型，我们加快并简化了具有挑战性的依存标注过程。最终产生的树库，将成为通用依存关系（UD）项目的一部分，将便利奥斯曼土耳其语文档的自动分析，揭示了这一历史遗产中蕴含的语言丰富性。

    arXiv:2402.14743v1 Announce Type: new  Abstract: This study introduces a pretrained large language model-based annotation methodology for the first dependency treebank in Ottoman Turkish. Our experimental results show that, iteratively, i) pseudo-annotating data using a multilingual BERT-based parsing model, ii) manually correcting the pseudo-annotations, and iii) fine-tuning the parsing model with the corrected annotations, we speed up and simplify the challenging dependency annotation process. The resulting treebank, that will be a part of the Universal Dependencies (UD) project, will facilitate automated analysis of Ottoman Turkish documents, unlocking the linguistic richness embedded in this historical heritage.
    
[^4]: AI增强预测：LLM助手提高人类预测准确性

    AI-Augmented Predictions: LLM Assistants Improve Human Forecasting Accuracy

    [https://arxiv.org/abs/2402.07862](https://arxiv.org/abs/2402.07862)

    本研究发现，使用LLMs助手可以显著提高预测准确性，不仅仅是由于模型预测准确性的提升。

    

    大型语言模型(LLMs)展现出令人印象深刻的能力，在许多领域与甚至超过人类表现。本研究探讨了LLMs在预测任务中增强判断力的潜力。我们评估了两个GPT-4-Turbo助手对预测准确性的影响：一个旨在提供高质量建议（超级预测），另一个旨在过于自信和基本概率忽视。参与者（N = 991）可以在整个研究过程中咨询他们被分配的LLM助手，而对照组则使用一个较低级别的模型（DaVinci-003），不提供直接的预测支持。我们的注册分析显示，LLM增强显著提高了23%的预测准确性，无论是对于任何一种助手类型，相比于对照组。这种改进发生在超级预测助手在预测中更高的准确性的情况下，表明增强的效益不仅仅是由于模型预测准确性。

    Large language models (LLMs) show impressive capabilities, matching and sometimes exceeding human performance in many domains. This study explores the potential of LLMs to augment judgement in forecasting tasks. We evaluated the impact on forecasting accuracy of two GPT-4-Turbo assistants: one designed to provide high-quality advice ('superforecasting'), and the other designed to be overconfident and base-rate-neglecting. Participants (N = 991) had the option to consult their assigned LLM assistant throughout the study, in contrast to a control group that used a less advanced model (DaVinci-003) without direct forecasting support. Our preregistered analyses reveal that LLM augmentation significantly enhances forecasting accuracy by 23% across both types of assistants, compared to the control group. This improvement occurs despite the superforecasting assistant's higher accuracy in predictions, indicating the augmentation's benefit is not solely due to model prediction accuracy. Explora
    
[^5]: 进取的鲍勃通过提示对抗调整抵制越狱行为

    Studious Bob Fight Back Against Jailbreaking via Prompt Adversarial Tuning

    [https://arxiv.org/abs/2402.06255](https://arxiv.org/abs/2402.06255)

    本文提出了一种名为Prompt Adversarial Tuning (PAT)的方法，通过训练一个防御控制机制并将其作为前缀嵌入到用户提示中，实现对大型语言模型（LLMs）的越狱行为的防御。实验证明该方法在保护LLMs免受产生有害信息的影响方面效果显著。

    

    尽管大型语言模型（LLM）在各种应用中取得了巨大的成功，但它们也容易受到特定提示的影响，从而绕过内置的安全措施并提供危险或非法内容，这种现象被称为越狱行为。为了保护LLMs免受产生有害信息的影响，提出了各种防御策略，其中大多数集中在内容过滤或模型的对抗训练方面。在本文中，我们提出了一种名为Prompt Adversarial Tuning（PAT）的方法，通过训练一个防御控制机制并将其作为前缀嵌入到用户提示中来实现我们的防御策略。我们设计了一个类似对抗训练的训练过程，以实现我们的优化目标，交替更新攻击和防御控制机制。据我们所知，我们是第一个从提示调整的角度实施防御的人。一旦应用，我们的方法几乎不会影响LLMs的操作效率。实验表明我们的方法在抵御越狱行为方面具有良好的效果。

    Although Large Language Models (LLMs) have achieved tremendous success in various applications, they are also susceptible to certain prompts that can induce them to bypass built-in safety measures and provide dangerous or illegal content, a phenomenon known as jailbreak. To protect LLMs from producing harmful information, various defense strategies are proposed, with most focusing on content filtering or adversarial training of models. In this paper, we propose an approach named Prompt Adversarial Tuning (PAT) to train a defense control mechanism, which is then embedded as a prefix to user prompts to implement our defense strategy. We design a training process similar to adversarial training to achieve our optimized goal, alternating between updating attack and defense controls. To our knowledge, we are the first to implement defense from the perspective of prompt tuning. Once employed, our method will hardly impact the operational efficiency of LLMs. Experiments show that our method i
    
[^6]: 澄清：通过自然语言纠正提高模型的鲁棒性

    Clarify: Improving Model Robustness With Natural Language Corrections

    [https://arxiv.org/abs/2402.03715](https://arxiv.org/abs/2402.03715)

    论文提出了Clarify，一种通过自然语言纠正模型错误概念的方法，该方法通过用户提供简短的文本描述来纠正模型的一致失败模式，从而提高模型的鲁棒性。

    

    在监督学习中，模型被训练从静态数据集中提取相关性。这通常会导致模型依赖于高级错误概念。为了防止这种错误概念，我们必须提供额外的信息。现有的方法包括一些额外的实例级监督形式，例如标记虚假特征或来自平衡分布的额外标记数据。对于大规模数据集来说，这些策略可能会变得昂贵，因为它们需要以接近原始训练数据的规模进行额外注释。我们假设有针对性的关于模型错误概念的自然语言反馈是一种更有效的额外监督形式。我们引入了Clarify，一种新型界面和方法来交互式地纠正模型的错误概念。通过Clarify，用户只需要提供一个简短的文本描述来描述模型的一致性失败模式。然后，我们完全自动化地使用s

    In supervised learning, models are trained to extract correlations from a static dataset. This often leads to models that rely on high-level misconceptions. To prevent such misconceptions, we must necessarily provide additional information beyond the training data. Existing methods incorporate forms of additional instance-level supervision, such as labels for spurious features or additional labeled data from a balanced distribution. Such strategies can become prohibitively costly for large-scale datasets since they require additional annotation at a scale close to the original training data. We hypothesize that targeted natural language feedback about a model's misconceptions is a more efficient form of additional supervision. We introduce Clarify, a novel interface and method for interactively correcting model misconceptions. Through Clarify, users need only provide a short text description to describe a model's consistent failure patterns. Then, in an entirely automated way, we use s
    
[^7]: 多模态大型语言模型中非语言抽象推理的奇特案例

    The Curious Case of Nonverbal Abstract Reasoning with Multi-Modal Large Language Models

    [https://arxiv.org/abs/2401.12117](https://arxiv.org/abs/2401.12117)

    本研究评估了多模态大型语言模型（MLLMs）在非语言抽象推理方面的能力，并发现开源和闭源模型之间存在巨大差距和个体模块的关键缺陷。

    

    尽管大型语言模型（LLMs）仍在逐渐应用于新领域并在新应用中被利用，但我们正在经历新一代基础模型的涌现，即多模态大型语言模型（MLLMs）。这些模型将语言和视觉信息进行整合，为展示出更复杂的推理能力在两种模态的交集处提供了新的可能性。然而，尽管MLLMs的前景具有革命性的前景，我们对它们的推理能力的理解仍然有限。在本研究中，我们使用Raven's Progressive Matrices的变式评估了开源和闭源MLLMs的非语言抽象推理能力。我们的实验揭示了解决这类问题的困难，同时展示了开源和闭源模型之间巨大的差距。我们还揭示了个体视觉和文本模块的关键缺陷，导致模型的性能受到低谷的限制。最后，为了提高MLLMs的性能，我们进行了一系列实验。

    While large language models (LLMs) are still being adopted to new domains and utilized in novel applications, we are experiencing an influx of the new generation of foundation models, namely multi-modal large language models (MLLMs). These models integrate verbal and visual information, opening new possibilities to demonstrate more complex reasoning abilities at the intersection of the two modalities. However, despite the revolutionizing prospect of MLLMs, our understanding of their reasoning abilities is limited. In this study, we assess the nonverbal abstract reasoning abilities of open-source and closed-source MLLMs using variations of Raven's Progressive Matrices. Our experiments expose the difficulty of solving such problems while showcasing the immense gap between open-source and closed-source models. We also reveal critical shortcomings with individual visual and textual modules, subjecting the models to low-performance ceilings. Finally, to improve MLLMs' performance, we experi
    
[^8]: 通过将大型语言模型与合成数据训练的VQA模型连接起来，将可视问答从合成问题泛化到人工问题

    Generalizing Visual Question Answering from Synthetic to Human-Written Questions via a Chain of QA with a Large Language Model. (arXiv:2401.06400v1 [cs.CL])

    [http://arxiv.org/abs/2401.06400](http://arxiv.org/abs/2401.06400)

    本文提出了一种新方法CoQAH，通过连接大型语言模型和训练于合成数据上的VQA模型的QA交互序列，实现了将可视问答从合成问题泛化到人工问题，并在两种类型的人工问题数据集上取得了最先进的准确率，超过了通用视觉语言模型、VQA模型和医学基础模型。

    

    可视问答（VQA）是一个给定图像并就该图像提出一系列问题的任务。构建一个高效的VQA算法需要大量的QA数据，而且非常昂贵。根据模板生成合成的QA对是获得数据的一种实用方法。然而，训练于这些数据上的VQA模型在复杂的人工问题上表现不佳。为了解决这个问题，我们提出了一种新的方法，称为“人工问题连锁问答”（CoQAH）。CoQAH利用一个大型语言模型与一个训练于合成数据上的VQA模型之间的QA交互序列来推理和推导人工问题的逻辑答案。我们在两种类型的人工问题VQA数据集上测试了CoQAH的效果，包括3D渲染图像和胸部X线图像，并发现它在两种类型的数据上都达到了最先进的准确率。值得注意的是，CoQAH在通用视觉语言模型、VQA模型和医学基础模型方面的表现也超过了。

    Visual question answering (VQA) is a task where an image is given, and a series of questions are asked about the image. To build an efficient VQA algorithm, a large amount of QA data is required which is very expensive. Generating synthetic QA pairs based on templates is a practical way to obtain data. However, VQA models trained on those data do not perform well on complex, human-written questions. To address this issue, we propose a new method called {\it chain of QA for human-written questions} (CoQAH). CoQAH utilizes a sequence of QA interactions between a large language model and a VQA model trained on synthetic data to reason and derive logical answers for human-written questions. We tested the effectiveness of CoQAH on two types of human-written VQA datasets for 3D-rendered and chest X-ray images and found that it achieved state-of-the-art accuracy in both types of data. Notably, CoQAH outperformed general vision-language models, VQA models, and medical foundation models with no
    
[^9]: KLoB: 一种评估语言模型中知识定位方法的基准

    KLoB: a Benchmark for Assessing Knowledge Locating Methods in Language Models. (arXiv:2309.16535v1 [cs.CL])

    [http://arxiv.org/abs/2309.16535](http://arxiv.org/abs/2309.16535)

    KLoB是一个评估语言模型中知识定位方法的基准，旨在解决现有定位方法的准确性和事实知识局部性假设的问题。

    

    最近，定位然后编辑的范式已经成为改变语言模型中存储的事实知识的主要方法之一。然而，目前的定位方法是否能够准确地找到嵌入所需知识的确切参数还缺乏研究。此外，尽管许多研究人员对事实知识的局部性假设的有效性提出了质疑，但没有提供一种测试假设的方法以进行更深入的讨论和研究。因此，我们引入了KLoB，一个评估可靠的知识定位方法应满足的三个基本属性的基准。KLoB可作为评估语言模型中现有定位方法的基准，并为重新评估事实知识的局部性假设提供了一种方法。我们的代码公开可用于\url{https://github.com/juyiming/KLoB}。

    Recently, Locate-Then-Edit paradigm has emerged as one of the main approaches in changing factual knowledge stored in the Language models. However, there is a lack of research on whether present locating methods can pinpoint the exact parameters embedding the desired knowledge. Moreover, although many researchers have questioned the validity of locality hypothesis of factual knowledge, no method is provided to test the a hypothesis for more in-depth discussion and research. Therefore, we introduce KLoB, a benchmark examining three essential properties that a reliable knowledge locating method should satisfy. KLoB can serve as a benchmark for evaluating existing locating methods in language models, and can contributes a method to reassessing the validity of locality hypothesis of factual knowledge. Our is publicly available at \url{https://github.com/juyiming/KLoB}.
    
[^10]: SPICED: 具有多个主题和复杂程度的新闻相似性检测数据集

    SPICED: News Similarity Detection Dataset with Multiple Topics and Complexity Levels. (arXiv:2309.13080v1 [cs.CL])

    [http://arxiv.org/abs/2309.13080](http://arxiv.org/abs/2309.13080)

    这个论文提出了一个名为SPICED的新闻相似性检测数据集，包括七个主题，并提供了四种不同的方法来生成新闻。

    

    如今，使用智能系统来检测新闻文章中的冗余信息已经变得非常普遍，以增强用户体验，尤其是随着新闻媒体的蓬勃发展。然而，新闻的异质性可能导致这些系统中的虚假发现：简单的启发式算法，比如一对新闻是否都涉及政治问题，可以提供强大但具有误导性的下游性能。将新闻相似性数据集分割成主题可以通过强制模型学习如何在更狭窄的领域中区分显著特征来改进这些模型的训练。然而，这需要存在目前缺乏的专题特定数据集。在本文中，我们提出了一个新的相似新闻数据集SPICED，其中包括七个主题：犯罪与法律、文化与娱乐、灾难与事故、经济与商业、政治与冲突、科学与技术以及体育。此外，我们提供了四种不同的方法来生成新闻。

    Nowadays, the use of intelligent systems to detect redundant information in news articles has become especially prevalent with the proliferation of news media outlets in order to enhance user experience. However, the heterogeneous nature of news can lead to spurious findings in these systems: Simple heuristics such as whether a pair of news are both about politics can provide strong but deceptive downstream performance. Segmenting news similarity datasets into topics improves the training of these models by forcing them to learn how to distinguish salient characteristics under more narrow domains. However, this requires the existence of topic-specific datasets, which are currently lacking. In this article, we propose a new dataset of similar news, SPICED, which includes seven topics: Crime & Law, Culture & Entertainment, Disasters & Accidents, Economy & Business, Politics & Conflicts, Science & Technology, and Sports. Futhermore, we present four distinct approaches for generating news 
    
[^11]: 我们能相信ChatGPT的评估吗？

    Can we trust the evaluation on ChatGPT?. (arXiv:2303.12767v1 [cs.CL])

    [http://arxiv.org/abs/2303.12767](http://arxiv.org/abs/2303.12767)

    本文讨论了ChatGPT评估中面临的数据污染挑战，通过倾向性检测任务阐述了这一问题，并探讨了如何在闭合且持续训练模型的时代确保模型评估的公平性。

    

    ChatGPT是第一个被广泛采纳的大型语言模型，展示出在多项自然语言任务中卓越的表现。但是，由于模型的闭合性以及通过强化学习和人类反馈不断更新，评估ChatGPT在不同问题领域的表现仍然具有挑战性。本文重点讨论了在ChatGPT的评估中存在的数据污染问题，并使用倾向性检测任务作为案例进行了说明。我们还讨论了如何在闭合和持续训练模型的时代，避免数据污染和确保公平的模型评估的挑战。

    ChatGPT, the first large language model (LLM) with mass adoption, has demonstrated remarkable performance in numerous natural language tasks. Despite its evident usefulness, evaluating ChatGPT's performance in diverse problem domains remains challenging due to the closed nature of the model and its continuous updates via Reinforcement Learning from Human Feedback (RLHF). We highlight the issue of data contamination in ChatGPT evaluations, with a case study of the task of stance detection. We discuss the challenge of preventing data contamination and ensuring fair model evaluation in the age of closed and continuously trained models.
    

