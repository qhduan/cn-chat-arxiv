# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [BERT-Enhanced Retrieval Tool for Homework Plagiarism Detection System](https://arxiv.org/abs/2404.01582) | 本文提出了一种基于GPT-3.5的抄袭文本数据生成方法和一种基于Faiss和BERT的高效高准确性的抄袭识别方法，填补了高水平抄袭检测研究数据集缺失的空白，实验证明该模型在多个指标上表现优异 |
| [^2] | [Enhancing Content-based Recommendation via Large Language Model](https://arxiv.org/abs/2404.00236) | 本文提出了一种名为LoID的语义知识传递方法，旨在提取多方面的语义信息以增强不同领域，并对齐用户/项目ID和内容语义特征空间。 |
| [^3] | [Ink and Individuality: Crafting a Personalised Narrative in the Age of LLMs](https://arxiv.org/abs/2404.00026) | 研究探讨了人们日益依赖的基于LLM的写作助手对创造力和个性可能造成的负面影响，旨在改进人机交互系统和提升写作助手的个性化和个性化功能。 |
| [^4] | [CODE-ACCORD: A Corpus of Building Regulatory Data for Rule Generation towards Automatic Compliance Checking](https://arxiv.org/abs/2403.02231) | 介绍了一个独特的数据集CODE-ACCORD，旨在解决自动合规检查中解释建筑法规的挑战，成为机器可读规则生成的基础。 |
| [^5] | [RAM-EHR: Retrieval Augmentation Meets Clinical Predictions on Electronic Health Records](https://arxiv.org/abs/2403.00815) | RAM-EHR通过增强检索并利用总结知识，提高了针对电子健康记录的临床预测效果。 |
| [^6] | [Agent-OM: Leveraging LLM Agents for Ontology Matching](https://arxiv.org/abs/2312.00326) | 本研究提出了Agent-OM，利用LLM代理为本体匹配系统引入了新的设计范式。 |
| [^7] | [InstructIE: A Chinese Instruction-based Information Extraction Dataset.](http://arxiv.org/abs/2305.11527) | 介绍了一份中文的基于指令的信息提取数据集InstructIE，其中包括了270,000个弱监督的数据和1,000个高质量注释实例。实验结果表明当前的模型表现有待改进，该任务仍存在挑战。 |

# 详细

[^1]: 基于BERT增强的作业抄袭检测系统

    BERT-Enhanced Retrieval Tool for Homework Plagiarism Detection System

    [https://arxiv.org/abs/2404.01582](https://arxiv.org/abs/2404.01582)

    本文提出了一种基于GPT-3.5的抄袭文本数据生成方法和一种基于Faiss和BERT的高效高准确性的抄袭识别方法，填补了高水平抄袭检测研究数据集缺失的空白，实验证明该模型在多个指标上表现优异

    

    文本抄袭检测任务是一项常见的自然语言处理任务，旨在检测给定文本是否包含从其他文本中抄袭或复制的内容。在现有研究中，由于缺乏高质量的数据集，检测高水平的抄袭仍然是一个挑战。本文提出了一种基于GPT-3.5的抄袭文本数据生成方法，产生了32,927对文本抄袭检测数据集，涵盖了各种抄袭方法，填补了这一研究领域的空白。同时，我们提出了一种基于Faiss和BERT的高效高准确性的抄袭识别方法。我们的实验证明，这种模型在准确率、精确率、召回率和F1分数等多个指标上的表现优于其他模型，分别达到了98.86％、98.90％、98.86％和0.9888。最后，我们还提供了一个用户友好的演示平台，允许用户上传文本。

    arXiv:2404.01582v1 Announce Type: cross  Abstract: Text plagiarism detection task is a common natural language processing task that aims to detect whether a given text contains plagiarism or copying from other texts. In existing research, detection of high level plagiarism is still a challenge due to the lack of high quality datasets. In this paper, we propose a plagiarized text data generation method based on GPT-3.5, which produces 32,927 pairs of text plagiarism detection datasets covering a wide range of plagiarism methods, bridging the gap in this part of research. Meanwhile, we propose a plagiarism identification method based on Faiss with BERT with high efficiency and high accuracy. Our experiments show that the performance of this model outperforms other models in several metrics, including 98.86\%, 98.90%, 98.86%, and 0.9888 for Accuracy, Precision, Recall, and F1 Score, respectively. At the end, we also provide a user-friendly demo platform that allows users to upload a text 
    
[^2]: 通过大型语言模型增强基于内容的推荐

    Enhancing Content-based Recommendation via Large Language Model

    [https://arxiv.org/abs/2404.00236](https://arxiv.org/abs/2404.00236)

    本文提出了一种名为LoID的语义知识传递方法，旨在提取多方面的语义信息以增强不同领域，并对齐用户/项目ID和内容语义特征空间。

    

    在现实世界的应用中，用户在与不同项目互动时表现出不同的行为，包括隐式的点击/点赞互动以及显式的评论/评价互动。然而，几乎所有的推荐工作都集中在如何通过隐式的点击/点赞互动来描述用户偏好，以找到人们之间的协同。对于基于内容的显式评论/评价互动，一些工作尝试利用它们来挖掘语义知识以增强推荐模型。然而，它们仍然忽视了以下两点：（1）内容语义是普适的世界知识；我们如何提取多方面的语义信息以增强不同领域？（2）用户/项目ID特征是推荐模型的基础要素；我们如何对齐ID和内容语义特征空间？在本文中，我们提出了一种“插件”语义知识传递方法LoID。

    arXiv:2404.00236v1 Announce Type: cross  Abstract: In real-world applications, users express different behaviors when they interact with different items, including implicit click/like interactions, and explicit comments/reviews interactions. Nevertheless, almost all recommender works are focused on how to describe user preferences by the implicit click/like interactions, to find the synergy of people. For the content-based explicit comments/reviews interactions, some works attempt to utilize them to mine the semantic knowledge to enhance recommender models. However, they still neglect the following two points: (1) The content semantic is a universal world knowledge; how do we extract the multi-aspect semantic information to empower different domains? (2) The user/item ID feature is a fundamental element for recommender models; how do we align the ID and content semantic feature space? In this paper, we propose a `plugin' semantic knowledge transferring method \textbf{LoID}, which inclu
    
[^3]: 墨水与个性：在LLMs时代塑造个性化叙事

    Ink and Individuality: Crafting a Personalised Narrative in the Age of LLMs

    [https://arxiv.org/abs/2404.00026](https://arxiv.org/abs/2404.00026)

    研究探讨了人们日益依赖的基于LLM的写作助手对创造力和个性可能造成的负面影响，旨在改进人机交互系统和提升写作助手的个性化和个性化功能。

    

    个性和个性化构成了使每个作家独特并影响其文字以有效吸引读者同时传达真实性的独特特征。然而，我们日益依赖基于LLM的写作助手可能会危及我们的创造力和个性。我们经常忽视这一趋势对我们的创造力和独特性的负面影响，尽管可能会造成后果。本研究通过进行简要调查探索不同的观点和概念，以及尝试理解人们的观点，结合以往在该领域的研究，来研究这些问题。解决这些问题对于改进人机交互系统和增强个性化和个性化写作助手至关重要。

    arXiv:2404.00026v1 Announce Type: cross  Abstract: Individuality and personalization comprise the distinctive characteristics that make each writer unique and influence their words in order to effectively engage readers while conveying authenticity. However, our growing reliance on LLM-based writing assistants risks compromising our creativity and individuality over time. We often overlook the negative impacts of this trend on our creativity and uniqueness, despite the possible consequences. This study investigates these concerns by performing a brief survey to explore different perspectives and concepts, as well as trying to understand people's viewpoints, in conjunction with past studies in the area. Addressing these issues is essential for improving human-computer interaction systems and enhancing writing assistants for personalization and individuality.
    
[^4]: CODE-ACCORD：用于规则生成的建筑法规数据语料库

    CODE-ACCORD: A Corpus of Building Regulatory Data for Rule Generation towards Automatic Compliance Checking

    [https://arxiv.org/abs/2403.02231](https://arxiv.org/abs/2403.02231)

    介绍了一个独特的数据集CODE-ACCORD，旨在解决自动合规检查中解释建筑法规的挑战，成为机器可读规则生成的基础。

    

    自动合规检查（ACC）在建筑、工程和施工（AEC）领域内的自动合规检查需要自动解释建筑法规，以发挥其全部潜力。然而，从文本规则中提取信息以将其转换为机器可读格式由于自然语言的复杂性以及仅能支持先进的机器学习技术的有限资源而成为一个挑战。为了解决这一挑战，我们介绍了一个独特的数据集CODE-ACCORD，这是在欧盟Horizon ACCORD项目下编制的。CODE-ACCORD包含862个来自英格兰和芬兰建筑法规的自包含句子。与我们的核心目标一致，即促进从文本中提取信息以生成机器可读规则，每个句子都注释了实体和关系。实体代表特定组件，如“窗户”和“烟雾探测器”，而re

    arXiv:2403.02231v1 Announce Type: new  Abstract: Automatic Compliance Checking (ACC) within the Architecture, Engineering, and Construction (AEC) sector necessitates automating the interpretation of building regulations to achieve its full potential. However, extracting information from textual rules to convert them to a machine-readable format has been a challenge due to the complexities associated with natural language and the limited resources that can support advanced machine-learning techniques. To address this challenge, we introduce CODE-ACCORD, a unique dataset compiled under the EU Horizon ACCORD project. CODE-ACCORD comprises 862 self-contained sentences extracted from the building regulations of England and Finland. Aligned with our core objective of facilitating information extraction from text for machine-readable rule generation, each sentence was annotated with entities and relations. Entities represent specific components such as "window" and "smoke detectors", while re
    
[^5]: RAM-EHR: 电子健康记录上的检索增强与临床预测相遇

    RAM-EHR: Retrieval Augmentation Meets Clinical Predictions on Electronic Health Records

    [https://arxiv.org/abs/2403.00815](https://arxiv.org/abs/2403.00815)

    RAM-EHR通过增强检索并利用总结知识，提高了针对电子健康记录的临床预测效果。

    

    我们提出了RAM-EHR，这是一个用于改善电子健康记录（EHR）上临床预测的检索增强（Retrieval Augmentation）流程。RAM-EHR首先收集多个知识来源，将它们转换为文本格式，并使用密集检索来获取与医学概念相关的信息。这一策略解决了与复杂概念名称相关的困难。RAM-EHR然后增广了与一致性正则化代码联合训练的本地EHR预测模型，以捕获来自患者就诊和总结知识的互补信息。在两个EHR数据集上的实验表明，RAM-EHR相对于之前的知识增强基线效果显著（AUROC增益3.4％，AUPR增益7.2％），强调了RAM-EHR的总结知识对临床预测任务的有效性。代码将发布在\url{https://github.com/ritaranx/RAM-EHR}。

    arXiv:2403.00815v1 Announce Type: cross  Abstract: We present RAM-EHR, a Retrieval AugMentation pipeline to improve clinical predictions on Electronic Health Records (EHRs). RAM-EHR first collects multiple knowledge sources, converts them into text format, and uses dense retrieval to obtain information related to medical concepts. This strategy addresses the difficulties associated with complex names for the concepts. RAM-EHR then augments the local EHR predictive model co-trained with consistency regularization to capture complementary information from patient visits and summarized knowledge. Experiments on two EHR datasets show the efficacy of RAM-EHR over previous knowledge-enhanced baselines (3.4% gain in AUROC and 7.2% gain in AUPR), emphasizing the effectiveness of the summarized knowledge from RAM-EHR for clinical prediction tasks. The code will be published at \url{https://github.com/ritaranx/RAM-EHR}.
    
[^6]: Agent-OM：利用LLM代理进行本体匹配

    Agent-OM: Leveraging LLM Agents for Ontology Matching

    [https://arxiv.org/abs/2312.00326](https://arxiv.org/abs/2312.00326)

    本研究提出了Agent-OM，利用LLM代理为本体匹配系统引入了新的设计范式。

    

    本体匹配（OM）能够实现不同本体之间的语义互操作性，通过对齐相关实体来解决其概念异构性。本研究引入了一种新颖的基于代理的LLM设计范式，命名为Agent-OM，包括两个用于检索和匹配的同体代理以及一组基于提示的简单OM工具。

    arXiv:2312.00326v2 Announce Type: replace  Abstract: Ontology matching (OM) enables semantic interoperability between different ontologies and resolves their conceptual heterogeneity by aligning related entities. OM systems currently have two prevailing design paradigms: conventional knowledge-based expert systems and newer machine learning-based predictive systems. While large language models (LLMs) and LLM agents have revolutionised data engineering and have been applied creatively in many domains, their potential for OM remains underexplored. This study introduces a novel agent-powered LLM-based design paradigm for OM systems. With consideration of several specific challenges in leveraging LLM agents for OM, we propose a generic framework, namely Agent-OM, consisting of two Siamese agents for retrieval and matching, with a set of simple prompt-based OM tools. Our framework is implemented in a proof-of-concept system. Evaluations of three Ontology Alignment Evaluation Initiative (OAE
    
[^7]: InstructIE: 一份基于指令的中文信息提取数据集

    InstructIE: A Chinese Instruction-based Information Extraction Dataset. (arXiv:2305.11527v1 [cs.CL])

    [http://arxiv.org/abs/2305.11527](http://arxiv.org/abs/2305.11527)

    介绍了一份中文的基于指令的信息提取数据集InstructIE，其中包括了270,000个弱监督的数据和1,000个高质量注释实例。实验结果表明当前的模型表现有待改进，该任务仍存在挑战。

    

    我们引入了一项新的信息提取任务，称为基于指令的信息提取 (Instruction-based IE)，它旨在要求系统遵循特定的指令或指南来提取信息。为了促进该领域的研究，我们构建了一个数据集，称为InstructIE，其中包括来自中文维基百科的 270,000 个弱监督数据和 1,000 个高质量众包注释实例。我们进一步评估了各种基线模型在InstructIE数据集上的表现。结果表明，尽管当前的模型表现很有希望，但仍有改进的空间。此外，我们进行了全面的案例研究分析，强调了基于指令的信息提取任务中固有的挑战。代码和数据集可在 https://github.com/zjunlp/DeepKE/tree/main/example/llm 找到。

    We introduce a new Information Extraction (IE) task dubbed Instruction-based IE, which aims to ask the system to follow specific instructions or guidelines to extract information. To facilitate research in this area, we construct a dataset called InstructIE, consisting of 270,000 weakly supervised data from Chinese Wikipedia and 1,000 high-quality crowdsourced annotated instances. We further evaluate the performance of various baseline models on the InstructIE dataset. The results reveal that although current models exhibit promising performance, there is still room for improvement. Furthermore, we conduct a comprehensive case study analysis, underlining the challenges inherent in the Instruction-based IE task. Code and dataset are available at https://github.com/zjunlp/DeepKE/tree/main/example/llm.
    

