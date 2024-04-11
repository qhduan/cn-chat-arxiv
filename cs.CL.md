# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [FPT: Feature Prompt Tuning for Few-shot Readability Assessment](https://arxiv.org/abs/2404.02772) | FPT提出了一种新颖的基于提示的调整框架，通过将语言特征嵌入训练软提示并设计新的损失函数，在少样本设置下改善了可读性评估任务的性能。 |
| [^2] | [KazParC: Kazakh Parallel Corpus for Machine Translation](https://arxiv.org/abs/2403.19399) | KazParC是一个跨哈萨克语、英语、俄语和土耳其语的机器翻译平行语料库，其中包含371,902个平行句子，还开发了性能优越的神经机器翻译模型Tilmash。 |
| [^3] | [KazSAnDRA: Kazakh Sentiment Analysis Dataset of Reviews and Attitudes](https://arxiv.org/abs/2403.19335) | KazSAnDRA是哈萨克情感分析领域的第一个最大的公开数据集，研究包括开发和评估四个机器学习模型，在极性分类和得分分类上取得了不错的成功结果。 |
| [^4] | [Gemma: Open Models Based on Gemini Research and Technology](https://arxiv.org/abs/2403.08295) | Gemma是基于Gemini研究和技术所构建的开放模型系列，在语言理解、推理和安全性等方面表现出色，负责任地发布这些大型语言模型对于提高前沿模型的安全性至关重要。 |
| [^5] | [ChatASU: Evoking LLM's Reflexion to Truly Understand Aspect Sentiment in Dialogues](https://arxiv.org/abs/2403.05326) | 本文提出了一个新的基于聊天的方面情绪理解（ChatASU）任务，旨在探索大型语言模型（LLMs）在对话场景中理解方面情绪的能力，并引入了一个子任务Aspect Chain Reasoning（ACR）任务来解决方面共指问题。 |
| [^6] | [A Semantic Distance Metric Learning approach for Lexical Semantic Change Detection](https://arxiv.org/abs/2403.00226) | 提出了一种用于词汇语义变化检测的语义距离度量学习方法，通过使用两个阶段的学习方法和感知编码器，实现了在多语言中优于以往方法的表现。 |
| [^7] | [$\lambda$-ECLIPSE: Multi-Concept Personalized Text-to-Image Diffusion Models by Leveraging CLIP Latent Space](https://arxiv.org/abs/2402.05195) | $\lambda$-ECLIPSE通过利用CLIP潜空间，实现了多概念个性化文本到图像的扩散模型。相比于传统方法，它减小了训练资源需求，并提供了更一致、高质量的图像生成结果。 |
| [^8] | [Zero-Shot Clinical Trial Patient Matching with LLMs](https://arxiv.org/abs/2402.05125) | 本研究基于LLMs开发了一个零样本临床试验患者匹配系统，可以高效评估患者是否符合入选标准，并通过优化提示策略和检索流程提高了数据和成本效率。 |
| [^9] | [MiniLLM: Knowledge Distillation of Large Language Models](https://arxiv.org/abs/2306.08543) | 本文提出了一种将大型语言模型的知识蒸馏到更小模型的方法，通过使用反向KLD替换标准KD方法中的前向KLD目标，有效避免了学生模型高估教师分布的低概率区域。 |
| [^10] | [Quality and Quantity of Machine Translation References for Automated Metrics.](http://arxiv.org/abs/2401.01283) | 本研究发现，机器翻译评估的较高质量参考文献对于评估指标与人类评价之间的相关性更好。每个段落平均使用7个参考文献有助于提升所有评估指标。不同质量的供应商参考文献可以混合使用来提高评估指标的准确性。这些发现可用于在特定预算下创建参考文献的共享任务的评估者。 |
| [^11] | [FedJudge: Federated Legal Large Language Model.](http://arxiv.org/abs/2309.08173) | 本文提出了第一个分布式法律大型语言模型（FedJudge）框架，可以通过在设备或客户端上进行本地微调，并将参数聚合和分布在中央服务器上来确保数据隐私。这解决了集中式训练法律LLMs引发的数据隐私问题和分布偏移导致的FL方法效果降低的挑战。 |
| [^12] | [A Two-Stage Framework with Self-Supervised Distillation For Cross-Domain Text Classification.](http://arxiv.org/abs/2304.09820) | 本文提出了一种双阶段框架，使用自监督蒸馏和来自不同但相关源领域的标记数据完成跨领域文本分类，取得在单源领域适应性和多源领域适应性上的新的最先进结果。 |
| [^13] | [Using large language models for (de-)formalization and natural argumentation exercises for beginner's students.](http://arxiv.org/abs/2304.06186) | 本研究描述了两个系统，利用大型语言模型，自动纠正初学者在逻辑语言转化和自然语言论证方面的问题。 |
| [^14] | [Improving the Diproche CNL through autoformalization via GPT-3.](http://arxiv.org/abs/2303.17513) | 本文探讨了在Diproche上使用大型语言模型进行自动形式化的可能性，并取得了令人鼓舞的初步结果。 |
| [^15] | [Using Persuasive Writing Strategies to Explain and Detect Health Misinformation.](http://arxiv.org/abs/2211.05985) | 本研究旨在通过使用说服性写作技巧的文本段落进行分类来增加自动化虚假信息检测的新层次，以产生可解释的理由。我们提出了一个包含常见说服性写作策略的注释方案和数据集，并使用 RoBERTa 文本分类模型进行实验。 |

# 详细

[^1]: FPT:特征提示调整用于少样本可读性评估

    FPT: Feature Prompt Tuning for Few-shot Readability Assessment

    [https://arxiv.org/abs/2404.02772](https://arxiv.org/abs/2404.02772)

    FPT提出了一种新颖的基于提示的调整框架，通过将语言特征嵌入训练软提示并设计新的损失函数，在少样本设置下改善了可读性评估任务的性能。

    

    基于提示的方法在大多数少样本文本分类任务中取得了有希望的结果。然而，在可读性评估任务中，传统的提示方法缺乏关键的语言知识，而已经被证明是必不可少的。此外，先前关于利用语言特征的研究显示，在少样本设置中具有非稳健性能，甚至可能损害模型性能。为了解决这些问题，我们提出了一个新颖的基于提示的调整框架，即具有丰富语言知识的特征提示调整（FPT）。具体地，我们从文本中提取语言特征，并将其嵌入可训练的软提示中。此外，我们设计了一个新的损失函数来校准类别之间的相似性排名顺序。实验结果表明，我们提出的FTP方法不仅在先前最佳的基于提示的调整方法上表现出显著的性能提升，而且超过了他们。

    arXiv:2404.02772v1 Announce Type: new  Abstract: Prompt-based methods have achieved promising results in most few-shot text classification tasks. However, for readability assessment tasks, traditional prompt methods lackcrucial linguistic knowledge, which has already been proven to be essential. Moreover, previous studies on utilizing linguistic features have shown non-robust performance in few-shot settings and may even impair model performance.To address these issues, we propose a novel prompt-based tuning framework that incorporates rich linguistic knowledge, called Feature Prompt Tuning (FPT). Specifically, we extract linguistic features from the text and embed them into trainable soft prompts. Further, we devise a new loss function to calibrate the similarity ranking order between categories. Experimental results demonstrate that our proposed method FTP not only exhibits a significant performance improvement over the prior best prompt-based tuning approaches, but also surpasses th
    
[^2]: KazParC：用于机器翻译的哈萨克平行语料库

    KazParC: Kazakh Parallel Corpus for Machine Translation

    [https://arxiv.org/abs/2403.19399](https://arxiv.org/abs/2403.19399)

    KazParC是一个跨哈萨克语、英语、俄语和土耳其语的机器翻译平行语料库，其中包含371,902个平行句子，还开发了性能优越的神经机器翻译模型Tilmash。

    

    我们介绍了KazParC，这是一个为跨哈萨克语、英语、俄语和土耳其语进行机器翻译而设计的平行语料库。作为其类别中首个也是最大的公开可用语料库，KazParC包含了371,902个平行句子的集合，涵盖不同领域，并在人类译者的协助下开发。我们的研究工作还扩展到了发展一个被昵称为Tilmash的神经机器翻译模型。引人注目的是，Tilmash的表现与工业巨头，如Google翻译和Yandex翻译，在标准评估指标（如BLEU和chrF）的基础上相媲美，并在某些情况下甚至超越。KazParC和Tilmash均可通过我们的GitHub存储库以知识共享署名4.0国际许可（CC BY 4.0）公开下载。

    arXiv:2403.19399v1 Announce Type: new  Abstract: We introduce KazParC, a parallel corpus designed for machine translation across Kazakh, English, Russian, and Turkish. The first and largest publicly available corpus of its kind, KazParC contains a collection of 371,902 parallel sentences covering different domains and developed with the assistance of human translators. Our research efforts also extend to the development of a neural machine translation model nicknamed Tilmash. Remarkably, the performance of Tilmash is on par with, and in certain instances, surpasses that of industry giants, such as Google Translate and Yandex Translate, as measured by standard evaluation metrics, such as BLEU and chrF. Both KazParC and Tilmash are openly available for download under the Creative Commons Attribution 4.0 International License (CC BY 4.0) through our GitHub repository.
    
[^3]: KazSAnDRA：哈萨克情感分析评论和态度数据集

    KazSAnDRA: Kazakh Sentiment Analysis Dataset of Reviews and Attitudes

    [https://arxiv.org/abs/2403.19335](https://arxiv.org/abs/2403.19335)

    KazSAnDRA是哈萨克情感分析领域的第一个最大的公开数据集，研究包括开发和评估四个机器学习模型，在极性分类和得分分类上取得了不错的成功结果。

    

    本文介绍了KazSAnDRA，这是一个为哈萨克情感分析开发的数据集，是第一个也是最大的公开可用数据集。KazSAnDRA包括来自各种来源的18万零64条评论的广泛收集，并包括从1到5的数字评分，提供了客户态度的定量表示。研究还通过开发和评估四个用于极性分类和得分分类的机器学习模型，致力于哈萨克情感分类的自动化。实验分析包括考虑平衡和不平衡情况下的结果评估。最成功的模型在测试集上的极性分类和得分分类的F1分别达到了0.81和0.39。数据集和优化模型是开放获取的，可在知识共享署名4.0国际许可下下载。

    arXiv:2403.19335v1 Announce Type: new  Abstract: This paper presents KazSAnDRA, a dataset developed for Kazakh sentiment analysis that is the first and largest publicly available dataset of its kind. KazSAnDRA comprises an extensive collection of 180,064 reviews obtained from various sources and includes numerical ratings ranging from 1 to 5, providing a quantitative representation of customer attitudes. The study also pursued the automation of Kazakh sentiment classification through the development and evaluation of four machine learning models trained for both polarity classification and score classification. Experimental analysis included evaluation of the results considering both balanced and imbalanced scenarios. The most successful model attained an F1-score of 0.81 for polarity classification and 0.39 for score classification on the test sets. The dataset and fine-tuned models are open access and available for download under the Creative Commons Attribution 4.0 International Lic
    
[^4]: Gemma：基于Gemini研究和技术的开放模型

    Gemma: Open Models Based on Gemini Research and Technology

    [https://arxiv.org/abs/2403.08295](https://arxiv.org/abs/2403.08295)

    Gemma是基于Gemini研究和技术所构建的开放模型系列，在语言理解、推理和安全性等方面表现出色，负责任地发布这些大型语言模型对于提高前沿模型的安全性至关重要。

    

    本文介绍了Gemma，这是一个基于Gemini模型研究和技术构建的轻量级、最先进的开放模型系列。Gemma模型在语言理解、推理和安全性等学术基准上表现出色。我们发布了两个规模的模型（20亿和70亿参数），并提供了预训练和微调的检查点。Gemma在18个基于文本的任务中，有11个任务优于类似规模的开放模型，并对模型的安全性和责任方面进行了全面评估，同时详细描述了模型开发过程。我们相信负责任地发布大型语言模型对于提高前沿模型的安全性，并实现下一波大型语言模型创新至关重要。

    arXiv:2403.08295v1 Announce Type: cross  Abstract: This work introduces Gemma, a family of lightweight, state-of-the art open models built from the research and technology used to create Gemini models. Gemma models demonstrate strong performance across academic benchmarks for language understanding, reasoning, and safety. We release two sizes of models (2 billion and 7 billion parameters), and provide both pretrained and fine-tuned checkpoints. Gemma outperforms similarly sized open models on 11 out of 18 text-based tasks, and we present comprehensive evaluations of safety and responsibility aspects of the models, alongside a detailed description of model development. We believe the responsible release of LLMs is critical for improving the safety of frontier models, and for enabling the next wave of LLM innovations.
    
[^5]: ChatASU：唤起LLM的反思，真正理解对话中的方面情绪

    ChatASU: Evoking LLM's Reflexion to Truly Understand Aspect Sentiment in Dialogues

    [https://arxiv.org/abs/2403.05326](https://arxiv.org/abs/2403.05326)

    本文提出了一个新的基于聊天的方面情绪理解（ChatASU）任务，旨在探索大型语言模型（LLMs）在对话场景中理解方面情绪的能力，并引入了一个子任务Aspect Chain Reasoning（ACR）任务来解决方面共指问题。

    

    在互动场景（例如，问答和对话）中进行方面情绪理解（ASU）近年来引起了越来越多的关注并取得了重要进展。然而，现有研究大多忽略了意见目标（即方面）的共指问题，而这种现象在互动场景特别是对话中普遍存在，限制了ASU的性能。最近，大型语言模型（LLM）展示了将各种NLP任务与聊天范式相结合的强大能力。基于此，本文提出了一项新的基于聊天的方面情绪理解（ChatASU）任务，旨在探索LLMs在对话场景中理解方面情绪的能力。特别是，这项ChatASU任务引入了一个子任务，即方面链推理（ACR）任务，以解决方面共指问题。在此基础上，我们提出了一种可信的自反思方法（TSA）与ChatGLM作为背景。

    arXiv:2403.05326v1 Announce Type: cross  Abstract: Aspect Sentiment Understanding (ASU) in interactive scenarios (e.g., Question-Answering and Dialogue) has attracted ever-more interest in recent years and achieved important progresses. However, existing studies on interactive ASU largely ignore the coreference issue for opinion targets (i.e., aspects), while this phenomenon is ubiquitous in interactive scenarios especially dialogues, limiting the ASU performance. Recently, large language models (LLMs) shows the powerful ability to integrate various NLP tasks with the chat paradigm. In this way, this paper proposes a new Chat-based Aspect Sentiment Understanding (ChatASU) task, aiming to explore LLMs' ability in understanding aspect sentiments in dialogue scenarios. Particularly, this ChatASU task introduces a sub-task, i.e., Aspect Chain Reasoning (ACR) task, to address the aspect coreference issue. On this basis, we propose a Trusted Self-reflexion Approach (TSA) with ChatGLM as back
    
[^6]: 用于词汇语义变化检测的语义距离度量学习方法

    A Semantic Distance Metric Learning approach for Lexical Semantic Change Detection

    [https://arxiv.org/abs/2403.00226](https://arxiv.org/abs/2403.00226)

    提出了一种用于词汇语义变化检测的语义距离度量学习方法，通过使用两个阶段的学习方法和感知编码器，实现了在多语言中优于以往方法的表现。

    

    检测词汇的时间语义变化是各种自然语言处理应用的重要任务，必须对时间敏感地进行预测。词汇语义变化检测（SCD）任务考虑在两个不同的文本语料库$C_1$和$C_2$之间预测给定目标词$w$是否改变了含义的问题。为此，我们提出了一种使用现有的Word-in-Context（WiC）数据集的监督两阶段SCD方法。在第一阶段，对于目标词$w$，我们学习了两个感知感知编码器，表示给定语料库中所选句子中$w$的含义。接下来，在第二阶段，我们学习了一种感知感知距离度量，比较目标词在$C_1$和$C_2$中的所有出现的语义表示。对多个SCD基准数据集的实验结果表明，我们提出的方法始终优于所有先前提出的多种语言的SCD方法。

    arXiv:2403.00226v1 Announce Type: new  Abstract: Detecting temporal semantic changes of words is an important task for various NLP applications that must make time-sensitive predictions. Lexical Semantic Change Detection (SCD) task considers the problem of predicting whether a given target word, $w$, changes its meaning between two different text corpora, $C_1$ and $C_2$. For this purpose, we propose a supervised two-staged SCD method that uses existing Word-in-Context (WiC) datasets. In the first stage, for a target word $w$, we learn two sense-aware encoder that represents the meaning of $w$ in a given sentence selected from a corpus. Next, in the second stage, we learn a sense-aware distance metric that compares the semantic representations of a target word across all of its occurrences in $C_1$ and $C_2$. Experimental results on multiple benchmark datasets for SCD show that our proposed method consistently outperforms all previously proposed SCD methods for multiple languages, esta
    
[^7]: $\lambda$-ECLIPSE: 通过利用CLIP潜空间，基于多概念个性化文本到图像扩散模型

    $\lambda$-ECLIPSE: Multi-Concept Personalized Text-to-Image Diffusion Models by Leveraging CLIP Latent Space

    [https://arxiv.org/abs/2402.05195](https://arxiv.org/abs/2402.05195)

    $\lambda$-ECLIPSE通过利用CLIP潜空间，实现了多概念个性化文本到图像的扩散模型。相比于传统方法，它减小了训练资源需求，并提供了更一致、高质量的图像生成结果。

    

    尽管个性化文本到图像(P-T2I)生成模型取得了近期的进展，但基于主题的T2I仍然具有挑战性。主要的瓶颈包括：1) 需要大量的训练资源，2) 超参数敏感性导致不一致的输出，以及3) 平衡新的视觉概念和构图对齐的复杂性。我们重新阐述了T2I扩散模型的核心理念，以解决上述限制。主要地，当代的基于主题的T2I方法依赖于潜空间扩散模型(LDMs)，通过交叉注意力层实现T2I映射。虽然LDMs提供了明显的优势，但P-T2I方法对这些扩散模型的潜空间的依赖显著增加了资源需求，导致结果不一致，并需要多次迭代才能得到一个所需的图像。最近，ECLIPSE展示了一种更具资源效率的训练UnCLIP-based T2I模型的路径，避免了需要扩散的需求。

    Despite the recent advances in personalized text-to-image (P-T2I) generative models, subject-driven T2I remains challenging. The primary bottlenecks include 1) Intensive training resource requirements, 2) Hyper-parameter sensitivity leading to inconsistent outputs, and 3) Balancing the intricacies of novel visual concept and composition alignment. We start by re-iterating the core philosophy of T2I diffusion models to address the above limitations. Predominantly, contemporary subject-driven T2I approaches hinge on Latent Diffusion Models (LDMs), which facilitate T2I mapping through cross-attention layers. While LDMs offer distinct advantages, P-T2I methods' reliance on the latent space of these diffusion models significantly escalates resource demands, leading to inconsistent results and necessitating numerous iterations for a single desired image. Recently, ECLIPSE has demonstrated a more resource-efficient pathway for training UnCLIP-based T2I models, circumventing the need for diffu
    
[^8]: 零样本临床试验患者匹配与LLMs

    Zero-Shot Clinical Trial Patient Matching with LLMs

    [https://arxiv.org/abs/2402.05125](https://arxiv.org/abs/2402.05125)

    本研究基于LLMs开发了一个零样本临床试验患者匹配系统，可以高效评估患者是否符合入选标准，并通过优化提示策略和检索流程提高了数据和成本效率。

    

    将患者与临床试验匹配是推出新药的关键难题。目前，识别符合试验入选标准的患者是高度手动的，每位患者需花费长达1小时。然而，自动筛选具有挑战性，因为它需要理解非结构化的临床文本。大型语言模型（LLMs）提供了一个有望的解决方案。在这项工作中，我们探索了它们在试验匹配中的应用。首先，我们设计了一个基于LLM的系统，可以在给定一个患者的病史作为非结构化的临床文本时，评估该患者是否符合一组包含标准（也以自由文本形式指定）。我们的零样本系统在n2c2 2018队列选择基准测试中取得了最先进的得分。其次，我们通过识别一种提示策略，改善了我们方法的数据和成本效率，该策略与现状相比可以将患者匹配时间和成本降低一个数量级，并且开发了一个两阶段的检索流程，减少了匹配消除的次数。

    Matching patients to clinical trials is a key unsolved challenge in bringing new drugs to market. Today, identifying patients who meet a trial's eligibility criteria is highly manual, taking up to 1 hour per patient. Automated screening is challenging, however, as it requires understanding unstructured clinical text. Large language models (LLMs) offer a promising solution. In this work, we explore their application to trial matching. First, we design an LLM-based system which, given a patient's medical history as unstructured clinical text, evaluates whether that patient meets a set of inclusion criteria (also specified as free text). Our zero-shot system achieves state-of-the-art scores on the n2c2 2018 cohort selection benchmark. Second, we improve the data and cost efficiency of our method by identifying a prompting strategy which matches patients an order of magnitude faster and more cheaply than the status quo, and develop a two-stage retrieval pipeline that reduces the number of 
    
[^9]: MiniLLM：大型语言模型的知识蒸馏

    MiniLLM: Knowledge Distillation of Large Language Models

    [https://arxiv.org/abs/2306.08543](https://arxiv.org/abs/2306.08543)

    本文提出了一种将大型语言模型的知识蒸馏到更小模型的方法，通过使用反向KLD替换标准KD方法中的前向KLD目标，有效避免了学生模型高估教师分布的低概率区域。

    

    知识蒸馏（KD）是一种减少大型语言模型（LLMs）高计算需求的有前途的技术。然而，先前的KD方法主要应用于白盒分类模型或训练小模型来模仿如ChatGPT之类的黑盒模型API。如何有效地将白盒LLMs的知识蒸馏到小模型中仍未得到充分探讨，随着开源LLMs的蓬勃发展，这变得更为重要。在这项工作中，我们提出一种KD方法，将LLMs蒸馏到更小的语言模型。

    arXiv:2306.08543v2 Announce Type: replace-cross  Abstract: Knowledge Distillation (KD) is a promising technique for reducing the high computational demand of large language models (LLMs). However, previous KD methods are primarily applied to white-box classification models or training small models to imitate black-box model APIs like ChatGPT. How to effectively distill the knowledge of white-box LLMs into small models is still under-explored, which becomes more important with the prosperity of open-source LLMs. In this work, we propose a KD approach that distills LLMs into smaller language models. We first replace the forward Kullback-Leibler divergence (KLD) objective in the standard KD approaches with reverse KLD, which is more suitable for KD on generative language models, to prevent the student model from overestimating the low-probability regions of the teacher distribution. Then, we derive an effective optimization approach to learn this objective. The student models are named Mi
    
[^10]: 机器翻译自动评估的参考文献质量和数量

    Quality and Quantity of Machine Translation References for Automated Metrics. (arXiv:2401.01283v1 [cs.CL])

    [http://arxiv.org/abs/2401.01283](http://arxiv.org/abs/2401.01283)

    本研究发现，机器翻译评估的较高质量参考文献对于评估指标与人类评价之间的相关性更好。每个段落平均使用7个参考文献有助于提升所有评估指标。不同质量的供应商参考文献可以混合使用来提高评估指标的准确性。这些发现可用于在特定预算下创建参考文献的共享任务的评估者。

    

    自动机器翻译评估指标通常使用人工翻译来确定系统翻译的质量。领域内的共识认为人工参考文献应具有很高的质量。然而，目前没有成本效益分析可以指导计划收集机器翻译评估参考文献的从业者。我们发现，较高质量的参考文献能够在段落级别上与人类评价的相关性更好。每个段落平均使用7个参考文献有助于所有评估指标的提升。有趣的是，来自不同质量的供应商的参考文献可以混合使用，并提高评估指标的准确性。然而，较高质量的参考文献制作成本更高，我们将其视为一个优化问题：在特定预算下，应该收集哪些参考文献以最大化评估指标的准确性。这些发现可用于在特定预算下创建参考文献的共享任务的评估者。

    Automatic machine translation metrics often use human translations to determine the quality system translations. Common wisdom in the field dictates that the human references should be of very high quality. However, there are no cost-benefit analyses that could be used to guide practitioners who plan to collect references for machine translation evaluation. We find that higher-quality references lead to better metric correlations with humans at the segment-level. Having up to 7 references per segment and taking their average helps all metrics. Interestingly, the references from vendors of different qualities can be mixed together and improve metric success. Higher quality references, however, cost more to create and we frame this as an optimization problem: given a specific budget, what references should be collected to maximize metric success. These findings can be used by evaluators of shared tasks when references need to be created under a certain budget.
    
[^11]: FedJudge: 分布式法律大型语言模型

    FedJudge: Federated Legal Large Language Model. (arXiv:2309.08173v1 [cs.CL])

    [http://arxiv.org/abs/2309.08173](http://arxiv.org/abs/2309.08173)

    本文提出了第一个分布式法律大型语言模型（FedJudge）框架，可以通过在设备或客户端上进行本地微调，并将参数聚合和分布在中央服务器上来确保数据隐私。这解决了集中式训练法律LLMs引发的数据隐私问题和分布偏移导致的FL方法效果降低的挑战。

    

    大型语言模型（LLMs）在法律智能领域得到了广泛应用，可以辅助法律专业人员和普通人。然而，这些法律LLMs的集中式训练引发了数据隐私问题，因为法律数据分散在包含敏感个人信息的各个机构之间。本文通过探索将法律LLMs与分布式学习（FL）方法相结合来解决这一挑战。通过使用FL，法律LLMs可以在设备或客户端上进行本地微调，其参数被聚合并分布在中央服务器上，确保数据隐私而无需直接共享原始数据。然而，计算和通信开销阻碍了LLMs在FL环境中的全面微调。此外，法律数据的分布偏移减少了FL方法的有效性。为此，在本文中，我们提出了第一个分布式法律大型语言模型（FedJudge）框架，可以对LLMs进行微调。

    Large Language Models (LLMs) have gained prominence in the field of Legal Intelligence, offering potential applications in assisting legal professionals and laymen. However, the centralized training of these Legal LLMs raises data privacy concerns, as legal data is distributed among various institutions containing sensitive individual information. This paper addresses this challenge by exploring the integration of Legal LLMs with Federated Learning (FL) methodologies. By employing FL, Legal LLMs can be fine-tuned locally on devices or clients, and their parameters are aggregated and distributed on a central server, ensuring data privacy without directly sharing raw data. However, computation and communication overheads hinder the full fine-tuning of LLMs under the FL setting. Moreover, the distribution shift of legal data reduces the effectiveness of FL methods. To this end, in this paper, we propose the first Federated Legal Large Language Model (FedJudge) framework, which fine-tunes 
    
[^12]: 一种自监督蒸馏的双阶段框架用于跨领域文本分类

    A Two-Stage Framework with Self-Supervised Distillation For Cross-Domain Text Classification. (arXiv:2304.09820v1 [cs.CL])

    [http://arxiv.org/abs/2304.09820](http://arxiv.org/abs/2304.09820)

    本文提出了一种双阶段框架，使用自监督蒸馏和来自不同但相关源领域的标记数据完成跨领域文本分类，取得在单源领域适应性和多源领域适应性上的新的最先进结果。

    

    跨领域文本分类旨在将模型适应于缺少标记数据的目标领域。它利用或重用不同但相关源领域的丰富标记数据和目标领域的未标记数据。为此，先前的工作要么专注于提取领域不变特征，要么忽略可能存在于目标领域中并对下游任务有用的领域感知特征的任务不可知特征。本文提出了一种双阶段框架，用于跨领域文本分类。在第一阶段，我们使用掩蔽语言建模（MLM）和来自源域的标记数据微调模型。在第二阶段，我们进一步使用自监督蒸馏（SSD）和来自目标域的未标记数据微调模型。我们基于公共的跨领域文本分类基准测试其性能，并实验结果表明，我们的方法在单源领域适应性和多源领域适应性上均取得了新的最先进结果。

    Cross-domain text classification aims to adapt models to a target domain that lacks labeled data. It leverages or reuses rich labeled data from the different but related source domain(s) and unlabeled data from the target domain. To this end, previous work focuses on either extracting domain-invariant features or task-agnostic features, ignoring domain-aware features that may be present in the target domain and could be useful for the downstream task. In this paper, we propose a two-stage framework for cross-domain text classification. In the first stage, we finetune the model with mask language modeling (MLM) and labeled data from the source domain. In the second stage, we further fine-tune the model with self-supervised distillation (SSD) and unlabeled data from the target domain. We evaluate its performance on a public cross-domain text classification benchmark and the experiment results show that our method achieves new state-of-the-art results for both single-source domain adaptat
    
[^13]: 使用大型语言模型进行初学者的（非）形式化和自然论证练习

    Using large language models for (de-)formalization and natural argumentation exercises for beginner's students. (arXiv:2304.06186v1 [cs.CL])

    [http://arxiv.org/abs/2304.06186](http://arxiv.org/abs/2304.06186)

    本研究描述了两个系统，利用大型语言模型，自动纠正初学者在逻辑语言转化和自然语言论证方面的问题。

    

    我们描述了两个系统，使用文本达芬奇-003，一个大型语言模型，自动纠正（i）自然语言与命题逻辑语言和一阶谓词逻辑语言之间转化的练习; 和（ii）在非数学场景下用自然语言编写简单论点的练习。

    We describe two systems that use text-davinci-003, a large language model, for the automatized correction of (i) exercises in translating back and forth between natural language and the languages of propositional logic and first-order predicate logic and (ii) exercises in writing simple arguments in natural language in non-mathematical scenarios.
    
[^14]: 通过GPT-3自动形式化提高Diproche CNL系统

    Improving the Diproche CNL through autoformalization via GPT-3. (arXiv:2303.17513v1 [cs.CL])

    [http://arxiv.org/abs/2303.17513](http://arxiv.org/abs/2303.17513)

    本文探讨了在Diproche上使用大型语言模型进行自动形式化的可能性，并取得了令人鼓舞的初步结果。

    

    Diproche系统是一款针对德语控制语言片段的自动化证明检查器，旨在用于教学应用，在引导学生进行证明时使用。该系统的第一个版本使用一种控制自然语言，其Prolog形式化例程已经编写好。本文中，我们探讨了在Diproche上使用大型语言模型进行自动形式化的可能性，并取得了令人鼓舞的初步结果。

    The Diproche system is an automated proof checker for texts written in a controlled fragment of German, designed for didactical applications in classes introducing students to proofs for the first time. The first version of the system used a controlled natural language for which a Prolog formalization routine was written. In this paper, we explore the possibility of prompting large language models for autoformalization in the context of Diproche, with encouraging first results.
    
[^15]: 使用说服性写作策略来解释和检测健康错误信息

    Using Persuasive Writing Strategies to Explain and Detect Health Misinformation. (arXiv:2211.05985v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2211.05985](http://arxiv.org/abs/2211.05985)

    本研究旨在通过使用说服性写作技巧的文本段落进行分类来增加自动化虚假信息检测的新层次，以产生可解释的理由。我们提出了一个包含常见说服性写作策略的注释方案和数据集，并使用 RoBERTa 文本分类模型进行实验。

    

    虚假信息的传播是当今社会的一大问题，许多学术界和工业界的研究人员正在努力解决这个问题。由于每天创造的虚假信息数量巨大，将此任务留给人工事实检查员是不切实际的。数据科学家和研究人员多年来一直致力于自动化虚假信息检测，但今天仍然是一个具有挑战性的问题。我们的研究目标是为自动化虚假信息检测添加一个新层次；使用具有说服性写作技巧的文本段落进行分类，以产生可解释的理由，说明为什么这篇文章可以标记为虚假信息。为此，我们提出了一个包含许多常见说服性写作策略的新注释方案，以及相应的人工注释数据集。我们使用 RoBERTa 文本分类模型来完成此任务，因为它在自然语言处理方面具有高性能。我们开发了几种基于语言模型的基线模型，并提供了结果分析。

    The spread of misinformation is a prominent problem in today's society, and many researchers in academia and industry are trying to combat it. Due to the vast amount of misinformation that is created every day, it is unrealistic to leave this task to human fact-checkers. Data scientists and researchers have been working on automated misinformation detection for years, and it is still a challenging problem today. The goal of our research is to add a new level to automated misinformation detection; classifying segments of text with persuasive writing techniques in order to produce interpretable reasoning for why an article can be marked as misinformation. To accomplish this, we present a novel annotation scheme containing many common persuasive writing tactics, along with a dataset with human annotations accordingly. For this task, we make use of a RoBERTa model for text classification, due to its high performance in NLP. We develop several language model-based baselines and present the 
    

