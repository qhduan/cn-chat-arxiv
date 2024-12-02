# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Cherry on Top: Parameter Heterogeneity and Quantization in Large Language Models](https://arxiv.org/abs/2404.02837) | 论文揭示了在大语言模型中存在参数异质性的现象，提出了一种名为CherryQ的量化方法，该方法能够在保留关键参数的同时将其余参数高效量化至低精度，在性能方面明显优于现有方法。 |
| [^2] | [IndicLLMSuite: A Blueprint for Creating Pre-training and Fine-Tuning Datasets for Indian Languages](https://arxiv.org/abs/2403.06350) | 为印度语言创建了一个覆盖22种语言、包含251B标记和74.8M指导-响应对的资源套件，结合高度筛选的数据、有价值的未验证数据和合成数据，建立了用于筛选预训练数据的干净开源流水线，以及用于指导微调的方法。 |
| [^3] | [Speech Translation with Speech Foundation Models and Large Language Models: What is There and What is Missing?](https://arxiv.org/abs/2402.12025) | 这项研究关注语音翻译领域的发展，通过将语音基础模型与大语言模型结合，为解决多模态任务提供了新的统一模型，但目前各种评估方法和设置多样性阻碍了确定每个架构构建块的最佳解决方案的识别。 |
| [^4] | [Asclepius: A Spectrum Evaluation Benchmark for Medical Multi-Modal Large Language Models](https://arxiv.org/abs/2402.11217) | Asclepius是一个新的医学多模态大语言模型基准，旨在为可信的Med-MLLMs评估提供单独且临床代表性的评估方案。 |
| [^5] | [Zero-shot sampling of adversarial entities in biomedical question answering](https://arxiv.org/abs/2402.10527) | 在生物医学问题回答中，我们提出了一种在嵌入空间中进行零样本采样的方案，用于发现各种对抗实体作为干扰因素，相比随机采样，在对抗问答中表现出明显优势，揭示了不同特征的两种对抗性实体制度。 |
| [^6] | [Evaluating the Data Model Robustness of Text-to-SQL Systems Based on Real User Queries](https://arxiv.org/abs/2402.08349) | 本文首次深入评估了在实践中文本到SQL系统的数据模型的鲁棒性，通过基于一个多年的国际项目集中评估，对一个在FIFA World Cup背景下连续运行了9个月的真实部署的FootballDB系统进行了评估。 |
| [^7] | [Unified Speech-Text Pretraining for Spoken Dialog Modeling](https://arxiv.org/abs/2402.05706) | 本研究提出了一个名为统一口语对话模型（USDM）的广泛语音文本模型框架，用于生成与输入语音相关的连贯口语回复。通过使用多步骤的语音文本推理方式和广义语音文本预训练方案，该方法能够有效捕捉跨模态语义，并生成自然流畅的口语回复。 |
| [^8] | [A Survey on Multimodal Large Language Models.](http://arxiv.org/abs/2306.13549) | 本文追踪和总结了多模态大语言模型（MLLM）的最新进展，包括多模态指令调整、多模态上下文学习、多模态思维链和LLM辅助视觉推理等应用，指出了现有挑战和有前途的研究方向。 |
| [^9] | [Memorization of Named Entities in Fine-tuned BERT Models.](http://arxiv.org/abs/2212.03749) | 本文研究了细调BERT模型中命名实体的记忆程度，并采用差分隐私进行实验。实验结果表明，应用差分隐私会对模型的性能产生不利影响。 |

# 详细

[^1]: 最后收官：大语言模型中的参数异质性和量化

    Cherry on Top: Parameter Heterogeneity and Quantization in Large Language Models

    [https://arxiv.org/abs/2404.02837](https://arxiv.org/abs/2404.02837)

    论文揭示了在大语言模型中存在参数异质性的现象，提出了一种名为CherryQ的量化方法，该方法能够在保留关键参数的同时将其余参数高效量化至低精度，在性能方面明显优于现有方法。

    

    这篇论文揭示了大型语言模型（LLMs）中参数异质性的现象。我们发现，少量“樱桃”参数对模型性能产生了不成比例的巨大影响，而绝大多数参数的影响较小。这种异质性在不同模型系列、规模和类型中普遍存在。在这一观察的基础上，我们提出了CherryQ，一种新颖的量化方法，统一了混合精度参数的优化。CherryQ能够识别并保留高精度下关键的樱桃参数，同时将其余参数积极量化为低精度。大量实验证明了CherryQ的有效性。CherryQ在困惑度和下游任务性能方面优于现有的量化方法。值得注意的是，我们的3位量化Vicuna-1.5与它们的16位对应物相比表现出色。

    arXiv:2404.02837v1 Announce Type: new  Abstract: This paper reveals the phenomenon of parameter heterogeneity in large language models (LLMs). We find that a small subset of ``cherry'' parameters exhibit a disproportionately large influence on model performance, while the vast majority of parameters have minimal impact. This heterogeneity is found to be prevalent across different model families, scales, and types. Motivated by this observation, we propose CherryQ, a novel quantization method that unifies the optimization of mixed-precision parameters. CherryQ identifies and preserves the critical cherry parameters in high precision while aggressively quantizing the remaining parameters to low precision. Extensive experiments demonstrate the effectiveness of CherryQ. CherryQ outperforms existing quantization approaches in terms of perplexity and downstream task performance. Notably, our 3-bit quantized Vicuna-1.5 exhibits competitive performance compared to their 16-bit counterparts. Th
    
[^2]: IndicLLMSuite: 为印度语言创建预训练和微调数据集提供了蓝图

    IndicLLMSuite: A Blueprint for Creating Pre-training and Fine-Tuning Datasets for Indian Languages

    [https://arxiv.org/abs/2403.06350](https://arxiv.org/abs/2403.06350)

    为印度语言创建了一个覆盖22种语言、包含251B标记和74.8M指导-响应对的资源套件，结合高度筛选的数据、有价值的未验证数据和合成数据，建立了用于筛选预训练数据的干净开源流水线，以及用于指导微调的方法。

    

    尽管英文LLM（Large Language Models）取得了显著进展，但由于缺乏定制资源，构建其他语言的可比模型的进展受阻。我们的工作旨在通过引入一个专门为发展印度语言LLM而设计的大量资源套件来弥合这一鸿沟，涵盖了22种语言，包含总共251B标记和7480万个指导-响应对。我们认识到数据质量和数量的重要性，我们的方法结合了经过精心筛选的手动验证数据、尚未验证但有价值的数据和合成数据。我们构建了一个干净的、开源的流水线，用于从各种来源筛选预训练数据，包括网站、PDF和视频，融入了爬取、清理、标记和去重的最佳实践。对于指导微调，我们汇集了现有的印度数据集，将英文数据集翻译/转写成印度语言，并利用了LLaMa2的技术。

    arXiv:2403.06350v1 Announce Type: new  Abstract: Despite the considerable advancements in English LLMs, the progress in building comparable models for other languages has been hindered due to the scarcity of tailored resources. Our work aims to bridge this divide by introducing an expansive suite of resources specifically designed for the development of Indic LLMs, covering 22 languages, containing a total of 251B tokens and 74.8M instruction-response pairs. Recognizing the importance of both data quality and quantity, our approach combines highly curated manually verified data, unverified yet valuable data, and synthetic data. We build a clean, open-source pipeline for curating pre-training data from diverse sources, including websites, PDFs, and videos, incorporating best practices for crawling, cleaning, flagging, and deduplication. For instruction-fine tuning, we amalgamate existing Indic datasets, translate/transliterate English datasets into Indian languages, and utilize LLaMa2 a
    
[^3]: 使用语音基础模型和大语言模型的语音翻译：存在和缺失的内容是什么？

    Speech Translation with Speech Foundation Models and Large Language Models: What is There and What is Missing?

    [https://arxiv.org/abs/2402.12025](https://arxiv.org/abs/2402.12025)

    这项研究关注语音翻译领域的发展，通过将语音基础模型与大语言模型结合，为解决多模态任务提供了新的统一模型，但目前各种评估方法和设置多样性阻碍了确定每个架构构建块的最佳解决方案的识别。

    

    自然语言处理（NLP）领域最近发生了一场变革性的转变，随着基础模型的出现，特别是彻底改变了基于文本的NLP的大型语言模型（LLMs）。这种范式已经扩展到其他形式，包括语音，在那里研究人员正在积极探索将语音基础模型（SFMs）和LLMs结合成单一的统一模型，以解决多模态任务。在这些任务中，本文着重于语音到文本翻译（ST）。通过审查该主题上发表的论文，我们提出了迄今为止提出的架构解决方案和训练策略的统一观点，强调它们之间的相似之处和差异之处。基于这一研究，我们不仅整理了所学到的经验教训，还展示了多样化的设置和评估方法如何阻碍对每个架构构建块的最佳性能解决方案的识别。

    arXiv:2402.12025v1 Announce Type: new  Abstract: The field of natural language processing (NLP) has recently witnessed a transformative shift with the emergence of foundation models, particularly Large Language Models (LLMs) that have revolutionized text-based NLP. This paradigm has extended to other modalities, including speech, where researchers are actively exploring the combination of Speech Foundation Models (SFMs) and LLMs into single, unified models capable of addressing multimodal tasks. Among such tasks, this paper focuses on speech-to-text translation (ST). By examining the published papers on the topic, we propose a unified view of the architectural solutions and training strategies presented so far, highlighting similarities and differences among them. Based on this examination, we not only organize the lessons learned but also show how diverse settings and evaluation approaches hinder the identification of the best-performing solution for each architectural building block 
    
[^4]: Asclepius：用于医学多模态大语言模型的频谱评估基准

    Asclepius: A Spectrum Evaluation Benchmark for Medical Multi-Modal Large Language Models

    [https://arxiv.org/abs/2402.11217](https://arxiv.org/abs/2402.11217)

    Asclepius是一个新的医学多模态大语言模型基准，旨在为可信的Med-MLLMs评估提供单独且临床代表性的评估方案。

    

    arXiv:2402.11217v1 公告类型：新摘要：医学多模态大语言模型（Med-MLLMs）的重大突破通过强大的信息综合和医疗决策支持改造了现代医疗保健。然而，由于现实世界诊断框架的复杂性涵盖了各种医学专业，并涉及复杂的临床决策，这些模型通常在不适合Med-MLLMs的基准上进行评估。此外，由于Med-MLLMs是在大量公开可用数据集上进行训练的，这些基准容易出现数据泄露。因此，需要一个独立且临床代表性的基准用于可信的Med-MLLMs评估。为此，我们引入了Asclepius，一个新颖的Med-MLLM基准，严格和全面评估模型在不同医学专业（心血管、胃肠等）和不同诊断能力（知觉、疾病分析等）方面的能力。

    arXiv:2402.11217v1 Announce Type: new  Abstract: The significant breakthroughs of Medical Multi-Modal Large Language Models (Med-MLLMs) renovate modern healthcare with robust information synthesis and medical decision support. However, these models are often evaluated on benchmarks that are unsuitable for the Med-MLLMs due to the intricate nature of the real-world diagnostic frameworks, which encompass diverse medical specialties and involve complex clinical decisions. Moreover, these benchmarks are susceptible to data leakage, since Med-MLLMs are trained on large assemblies of publicly available data. Thus, an isolated and clinically representative benchmark is highly desirable for credible Med-MLLMs evaluation. To this end, we introduce Asclepius, a novel Med-MLLM benchmark that rigorously and comprehensively assesses model capability in terms of: distinct medical specialties (cardiovascular, gastroenterology, etc.) and different diagnostic capacities (perception, disease analysis, e
    
[^5]: 生物医学问题回答中的零样本采样对抗实体

    Zero-shot sampling of adversarial entities in biomedical question answering

    [https://arxiv.org/abs/2402.10527](https://arxiv.org/abs/2402.10527)

    在生物医学问题回答中，我们提出了一种在嵌入空间中进行零样本采样的方案，用于发现各种对抗实体作为干扰因素，相比随机采样，在对抗问答中表现出明显优势，揭示了不同特征的两种对抗性实体制度。

    

    大型语言模型（LLM）中参数域知识的增加深度推动它们在现实世界应用中的快速部署。在高风险和知识密集型任务中，理解模型的漏洞对于量化模型预测的可信度和规范其使用至关重要。最近发现在自然语言处理任务中作为对抗示例的命名实体引发了关于它们在其他环境中可能的伪装的疑问。在这里，我们提出了一种在嵌入空间中的幂缩放距离加权采样方案，以发现多样化的对抗实体作为干扰因素。我们展示了它在生物医学主题的对抗性问题回答中优于随机采样的优势。我们的方法使得可以探索攻击表面上的不同区域，这揭示了两种在特征上明显不同的对抗性实体的制度。此外，我们展示了攻击方式如何...

    arXiv:2402.10527v1 Announce Type: new  Abstract: The increasing depth of parametric domain knowledge in large language models (LLMs) is fueling their rapid deployment in real-world applications. In high-stakes and knowledge-intensive tasks, understanding model vulnerabilities is essential for quantifying the trustworthiness of model predictions and regulating their use. The recent discovery of named entities as adversarial examples in natural language processing tasks raises questions about their potential guises in other settings. Here, we propose a powerscaled distance-weighted sampling scheme in embedding space to discover diverse adversarial entities as distractors. We demonstrate its advantage over random sampling in adversarial question answering on biomedical topics. Our approach enables the exploration of different regions on the attack surface, which reveals two regimes of adversarial entities that markedly differ in their characteristics. Moreover, we show that the attacks su
    
[^6]: 基于真实用户查询评估文本到SQL系统的数据模型鲁棒性

    Evaluating the Data Model Robustness of Text-to-SQL Systems Based on Real User Queries

    [https://arxiv.org/abs/2402.08349](https://arxiv.org/abs/2402.08349)

    本文首次深入评估了在实践中文本到SQL系统的数据模型的鲁棒性，通过基于一个多年的国际项目集中评估，对一个在FIFA World Cup背景下连续运行了9个月的真实部署的FootballDB系统进行了评估。

    

    文本到SQL系统（也称为自然语言到SQL系统）已成为弥合用户能力与基于SQL的数据访问之间差距的越来越流行的解决方案。这些系统将用户的自然语言请求转化为特定数据库的有效SQL语句。最近的基于转换器的语言模型使得文本到SQL系统受益匪浅。然而，虽然这些系统在常常是合成基准数据集上不断取得新的高分，但对于它们在真实世界、现实场景中对不同数据模型的鲁棒性的系统性探索明显缺乏。本文基于一个多年国际项目关于文本到SQL界面的集中评估，提供了对文本到SQL系统在实践中数据模型鲁棒性的首次深度评估。我们的评估基于FootballDB的真实部署，该系统在FIFA World Cup的背景下连续运行了9个月。

    Text-to-SQL systems (also known as NL-to-SQL systems) have become an increasingly popular solution for bridging the gap between user capabilities and SQL-based data access. These systems translate user requests in natural language to valid SQL statements for a specific database. Recent Text-to-SQL systems have benefited from the rapid improvement of transformer-based language models. However, while Text-to-SQL systems that incorporate such models continuously reach new high scores on -- often synthetic -- benchmark datasets, a systematic exploration of their robustness towards different data models in a real-world, realistic scenario is notably missing. This paper provides the first in-depth evaluation of the data model robustness of Text-to-SQL systems in practice based on a multi-year international project focused on Text-to-SQL interfaces. Our evaluation is based on a real-world deployment of FootballDB, a system that was deployed over a 9 month period in the context of the FIFA Wor
    
[^7]: 面向口语对话建模的统一语音文本预训练方法

    Unified Speech-Text Pretraining for Spoken Dialog Modeling

    [https://arxiv.org/abs/2402.05706](https://arxiv.org/abs/2402.05706)

    本研究提出了一个名为统一口语对话模型（USDM）的广泛语音文本模型框架，用于生成与输入语音相关的连贯口语回复。通过使用多步骤的语音文本推理方式和广义语音文本预训练方案，该方法能够有效捕捉跨模态语义，并生成自然流畅的口语回复。

    

    近期的研究表明，扩展大型语言模型（LLM）以直接理解和合成语音具有良好的结果，但用于口语对话建模的基于LLM的策略仍然难以实现，需要进一步研究。本文提出了一个广泛的语音文本LLM框架，命名为统一口语对话模型（USDM），以在不依赖于自动语音识别（ASR）或文本到语音（TTS）解决方案的情况下生成与给定输入语音相关的连贯口语回复和有机的韵律特征。我们的方法采用了一种多步骤的语音文本推理方式，利用了底层LLM所展示的推理链能力。我们还提出了一种广义的语音文本预训练方案，有助于捕捉跨模态语义。自动和人工评估结果表明，所提出的方法能够有效生成自然流畅的口语回复，并且优于之前的和级联的基线模型。详细的比较研究

    While recent work shows promising results in expanding the capabilities of large language models (LLM) to directly understand and synthesize speech, an LLM-based strategy for modeling spoken dialogs remains elusive and calls for further investigation. This work proposes an extensive speech-text LLM framework, named the Unified Spoken Dialog Model (USDM), to generate coherent spoken responses with organic prosodic features relevant to the given input speech without relying on automatic speech recognition (ASR) or text-to-speech (TTS) solutions. Our approach employs a multi-step speech-text inference scheme that leverages chain-of-reasoning capabilities exhibited by the underlying LLM. We also propose a generalized speech-text pretraining scheme that helps with capturing cross-modal semantics. Automatic and human evaluations show that the proposed approach is effective in generating natural-sounding spoken responses, outperforming both prior and cascaded baselines. Detailed comparative s
    
[^8]: 多模态大语言模型综述

    A Survey on Multimodal Large Language Models. (arXiv:2306.13549v1 [cs.CV])

    [http://arxiv.org/abs/2306.13549](http://arxiv.org/abs/2306.13549)

    本文追踪和总结了多模态大语言模型（MLLM）的最新进展，包括多模态指令调整、多模态上下文学习、多模态思维链和LLM辅助视觉推理等应用，指出了现有挑战和有前途的研究方向。

    

    多模态大语言模型（MLLM）是一种新兴的研究热点，使用强大的大语言模型作为大脑执行多模态任务。MLLM 的惊人能力，如基于图像编写故事和无OCR数学推理等，在传统方法中很少见，表明了通向人工智能的潜在路径。本文旨在追踪和总结 MLLM 的最新进展。首先，我们介绍了 MLLM 的构成，概述了相关概念。然后，讨论了关键技术和应用，包括多模态指令调整（M-IT）、多模态上下文学习（M-ICL）、多模态思维链（M-CoT）和LLM辅助视觉推理（LAVR）。最后，我们讨论了现有的挑战，并指出了有前途的研究方向。鉴于 MLLM 时代才刚刚开始，我们会不断更新这个综述，并希望能激发更多的研究。

    Multimodal Large Language Model (MLLM) recently has been a new rising research hotspot, which uses powerful Large Language Models (LLMs) as a brain to perform multimodal tasks. The surprising emergent capabilities of MLLM, such as writing stories based on images and OCR-free math reasoning, are rare in traditional methods, suggesting a potential path to artificial general intelligence. In this paper, we aim to trace and summarize the recent progress of MLLM. First of all, we present the formulation of MLLM and delineate its related concepts. Then, we discuss the key techniques and applications, including Multimodal Instruction Tuning (M-IT), Multimodal In-Context Learning (M-ICL), Multimodal Chain of Thought (M-CoT), and LLM-Aided Visual Reasoning (LAVR). Finally, we discuss existing challenges and point out promising research directions. In light of the fact that the era of MLLM has only just begun, we will keep updating this survey and hope it can inspire more research. An associated
    
[^9]: 细调BERT模型中的命名实体记忆

    Memorization of Named Entities in Fine-tuned BERT Models. (arXiv:2212.03749v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2212.03749](http://arxiv.org/abs/2212.03749)

    本文研究了细调BERT模型中命名实体的记忆程度，并采用差分隐私进行实验。实验结果表明，应用差分隐私会对模型的性能产生不利影响。

    

    隐私保护深度学习是机器学习中的新兴领域，旨在减轻深度神经网络在使用中的隐私风险。其中一个风险是从训练在个人和隐私敏感信息数据集上的语言模型中提取训练数据。在我们的研究中，我们调查了细调BERT模型中命名实体记忆的程度。我们使用单标签文本分类作为代表性的下游任务，在实验中采用三种不同的细调设置，包括一种差分隐私（DP）设置。我们利用自定义的顺序抽样策略和两种提示策略从细调BERT模型中创建了大量的文本样本。我们在这些样本中搜索命名实体，并查看它们是否也存在于细调数据集中。我们在电子邮件和博客领域使用了两个基准数据集进行实验。我们表明，DP的应用对测试性能产生了不利影响。

    Privacy preserving deep learning is an emerging field in machine learning that aims to mitigate the privacy risks in the use of deep neural networks. One such risk is training data extraction from language models that have been trained on datasets, which contain personal and privacy sensitive information. In our study, we investigate the extent of named entity memorization in fine-tuned BERT models. We use single-label text classification as representative downstream task and employ three different fine-tuning setups in our experiments, including one with Differentially Privacy (DP). We create a large number of text samples from the fine-tuned BERT models utilizing a custom sequential sampling strategy with two prompting strategies. We search in these samples for named entities and check if they are also present in the fine-tuning datasets. We experiment with two benchmark datasets in the domains of emails and blogs. We show that the application of DP has a detrimental effect on the te
    

