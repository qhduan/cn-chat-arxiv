# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Multi-Conditional Ranking with Large Language Models](https://arxiv.org/abs/2404.00211) | 该论文提出了一种新颖的分解推理方法(MCRank)，用于解决大型语言模型在多条件排序任务中性能下降的问题。 |
| [^2] | [IllusionVQA: A Challenging Optical Illusion Dataset for Vision Language Models](https://arxiv.org/abs/2403.15952) | 提出了IllusionVQA数据集，用于测试视觉语言模型在错觉和难解场景下的表现，研究发现在理解任务和定位任务上，表现最佳的VLM为GPT4V，而人类表现更胜一筹。 |
| [^3] | [Text clustering with LLM embeddings](https://arxiv.org/abs/2403.15112) | 研究表明，LLM嵌入能够捕捉结构化语言的细微差别，BERT在性能上领先于轻量级选项，增加嵌入维度和摘要技术并不一致地提高聚类效率 |
| [^4] | [Beyond prompt brittleness: Evaluating the reliability and consistency of political worldviews in LLMs](https://arxiv.org/abs/2402.17649) | 该研究评估了大型语言模型中的政治世界观的可靠性和一致性，发现他们的可靠性随模型参数数量增加而增加，且在政策方案上有所不同。 |
| [^5] | [What Evidence Do Language Models Find Convincing?](https://arxiv.org/abs/2402.11782) | 通过构建 ConflictingQA 数据集，并进行敏感性和反事实分析，研究发现当前语言模型在预测时很大程度上依赖于网站与查询的相关性，而忽视了人类认为重要的文本风格特征。 |
| [^6] | [Eliciting Latent Knowledge from Quirky Language Models](https://arxiv.org/abs/2312.01037) | 本研究通过引入一套“古怪”的语言模型，调取了这些模型在特定上下文中的潜在知识，展示了从可信度低的模型中调取可靠知识的前景。 |
| [^7] | [Partial Diacritization: A Context-Contrastive Inference Approach.](http://arxiv.org/abs/2401.08919) | 部分音标化是选择标记部分字符来提高阅读可读性和准确性的新方法。上下文对比的部分音标化（CCPD）集成了现有的阿拉伯音标化系统，并通过衡量部分音标化的新指标来判断需要标记哪些字符。 |
| [^8] | [CodePrompt: Improving Source Code-Related Classification with Knowledge Features through Prompt Learning.](http://arxiv.org/abs/2401.05544) | CodePrompt是一种利用Prompt学习和注意机制技术改进源代码相关分类任务的新方法。它能够提取源代码和相关文本中的丰富知识以提高准确性，并且减少了计算成本。 |
| [^9] | [MVMR: Evaluating Natural Language Video Localization Bias over Multiple Reliable Videos Pool.](http://arxiv.org/abs/2309.16701) | 本文提出了一个名为MVMR的任务，旨在给定文本查询从大量视频集中定位视频帧。我们通过已有数据集进行相似性筛选来构建数据集，并引入三个MVMR数据集。我们采用了嵌入式文本相似度匹配和视频-语言对齐技术来计算相关性得分，并为MVMR任务开发了一个强大的模型，Reliable Mutual Matching Network (RMMN)。 |
| [^10] | [Casteist but Not Racist? Quantifying Disparities in Large Language Model Bias between India and the West.](http://arxiv.org/abs/2309.08573) | 本研究量化了大型语言模型在印度和西方上的陈规偏见差异，并开发了一个新的数据集来评估种姓和宗教上的刻板印象。研究发现大多数测试的模型在印度背景下对刻板印象有显著偏见，尤其是与西方背景相比。此外，研究探索了一种简单干预方法来减轻这种偏见的效果。 |

# 详细

[^1]: 大型语言模型下的多条件排序

    Multi-Conditional Ranking with Large Language Models

    [https://arxiv.org/abs/2404.00211](https://arxiv.org/abs/2404.00211)

    该论文提出了一种新颖的分解推理方法(MCRank)，用于解决大型语言模型在多条件排序任务中性能下降的问题。

    

    利用大型语言模型(LLMs)对一组项目进行排序已成为推荐和检索系统中的常见方法。在这篇论文中，我们定义并探讨了多条件排序的任务，引入了一个名为MCRank的基准，旨在评估跨不同项目类型和条件进行多条件排序。我们使用MCRank对LLMs进行分析表明，随着项目和条件数量以及复杂性的增长，性能显著下降。为了克服这一限制，我们提出了一种新颖的分解推理方法，包括提取和排序条件，然后迭代地对条件进行排序。

    arXiv:2404.00211v1 Announce Type: new  Abstract: Utilizing large language models (LLMs) to rank a set of items has become a common approach in recommendation and retrieval systems. Typically, these systems focus on ordering a substantial number of documents in a monotonic order based on a given query. However, real-world scenarios often present a different challenge: ranking a comparatively smaller set of items, but according to a variety of diverse and occasionally conflicting conditions. In this paper, we define and explore the task of multi-conditional ranking by introducing MCRank, a benchmark tailored for assessing multi-conditional ranking across various item types and conditions. Our analysis of LLMs using MCRank indicates a significant decrease in performance as the number and complexity of items and conditions grow. To overcome this limitation, we propose a novel decomposed reasoning method, consisting of EXtracting and Sorting the conditions, and then Iterativly Ranking the i
    
[^2]: IllusionVQA：一个挑战视觉语言模型的错觉数据集

    IllusionVQA: A Challenging Optical Illusion Dataset for Vision Language Models

    [https://arxiv.org/abs/2403.15952](https://arxiv.org/abs/2403.15952)

    提出了IllusionVQA数据集，用于测试视觉语言模型在错觉和难解场景下的表现，研究发现在理解任务和定位任务上，表现最佳的VLM为GPT4V，而人类表现更胜一筹。

    

    视觉语言模型（VLM）的出现使研究人员能够使用自然语言调查神经网络的视觉理解。 VLM不仅能够进行对象分类和检测，还能够进行视觉理解和常识推理。 这自然而然地引出了一个问题：当图像本身是不合理的时，VLM会如何回应？ 为此，我们提出了IllusionVQA：一个包含具有挑战性的光学错觉和难以解释的场景的多样数据集，以测试VLM在两种不同的多选VQA任务 - 理解和软定位的能力。 表现最佳的VLM GPT4V在理解任务（4-shot）上实现了62.99％的准确率，在定位任务（4-shot和Chain-of-Thought）上实现了49.7％的准确率。 人类评估表明，人类在理解和定位方面的准确率分别为91.03％和100％。 我们发现，在上下文学习（ICL）和Chain-of-Thought推理方面有很大帮助。

    arXiv:2403.15952v1 Announce Type: cross  Abstract: The advent of Vision Language Models (VLM) has allowed researchers to investigate the visual understanding of a neural network using natural language. Beyond object classification and detection, VLMs are capable of visual comprehension and common-sense reasoning. This naturally led to the question: How do VLMs respond when the image itself is inherently unreasonable? To this end, we present IllusionVQA: a diverse dataset of challenging optical illusions and hard-to-interpret scenes to test the capability of VLMs in two distinct multiple-choice VQA tasks - comprehension and soft localization. GPT4V, the best-performing VLM, achieves 62.99% accuracy (4-shot) on the comprehension task and 49.7% on the localization task (4-shot and Chain-of-Thought). Human evaluation reveals that humans achieve 91.03% and 100% accuracy in comprehension and localization. We discover that In-Context Learning (ICL) and Chain-of-Thought reasoning substantially
    
[^3]: 使用LLM嵌入进行文本聚类

    Text clustering with LLM embeddings

    [https://arxiv.org/abs/2403.15112](https://arxiv.org/abs/2403.15112)

    研究表明，LLM嵌入能够捕捉结构化语言的细微差别，BERT在性能上领先于轻量级选项，增加嵌入维度和摘要技术并不一致地提高聚类效率

    

    文本聚类是组织不断增长的数字内容的重要方法，有助于结构化和发现未分类数据中的隐藏模式。在这项研究中，我们调查了不同文本嵌入（特别是大型语言模型LLMs中使用的）和聚类算法如何影响文本数据集的聚类方式。进行了一系列实验以评估嵌入是如何影响聚类结果的，以及通过摘要进行降维和嵌入大小调整的作用。结果显示，LLM嵌入在捕获结构化语言的细微差别方面表现出色，而BERT在性能上领先于轻量级选项。此外，我们发现增加嵌入维度和摘要技术并不一致地提高聚类效率，这表明这些策略需要仔细分析才能在实际模型中使用。这些结果突出了一种

    arXiv:2403.15112v1 Announce Type: cross  Abstract: Text clustering is an important approach for organising the growing amount of digital content, helping to structure and find hidden patterns in uncategorised data. In this research, we investigated how different textual embeddings - particularly those used in large language models (LLMs) - and clustering algorithms affect how text datasets are clustered. A series of experiments were conducted to assess how embeddings influence clustering results, the role played by dimensionality reduction through summarisation, and embedding size adjustment. Results reveal that LLM embeddings excel at capturing the nuances of structured language, while BERT leads the lightweight options in performance. In addition, we find that increasing embedding dimensionality and summarisation techniques do not uniformly improve clustering efficiency, suggesting that these strategies require careful analysis to use in real-life models. These results highlight a co
    
[^4]: 超越提示脆弱性：评估LLMs中政治世界观的可靠性和一致性

    Beyond prompt brittleness: Evaluating the reliability and consistency of political worldviews in LLMs

    [https://arxiv.org/abs/2402.17649](https://arxiv.org/abs/2402.17649)

    该研究评估了大型语言模型中的政治世界观的可靠性和一致性，发现他们的可靠性随模型参数数量增加而增加，且在政策方案上有所不同。

    

    由于大型语言模型（LLMs）在广泛系统中的使用，我们需要了解它们是否嵌入了特定的世界观以及这些观点所反映的内容。最近的研究报告称，当用政治问卷进行提示时，LLMs表现出左倾自由倾向。然而，目前尚不清楚这些倾向是否可靠（对提示变化稳健）以及这种倾向是否在政策和政治倾向上保持一致。我们提出了一系列测试，评估了基于收集自七个欧盟国家的选举建议问卷并标注为政策领域的数据集上LLMs在政治声明上立场的可靠性和一致性。我们研究了参数从7B到70B的LLMs，并发现它们的可靠性随参数数量增加而增加。更大的模型显示总体上与左倾政党更强的一致性，但在政策方案中有所不同：它们表现出（左倾）积极的立场

    arXiv:2402.17649v1 Announce Type: new  Abstract: Due to the widespread use of large language models (LLMs) in ubiquitous systems, we need to understand whether they embed a specific worldview and what these views reflect. Recent studies report that, prompted with political questionnaires, LLMs show left-liberal leanings. However, it is as yet unclear whether these leanings are reliable (robust to prompt variations) and whether the leaning is consistent across policies and political leaning. We propose a series of tests which assess the reliability and consistency of LLMs' stances on political statements based on a dataset of voting-advice questionnaires collected from seven EU countries and annotated for policy domains. We study LLMs ranging in size from 7B to 70B parameters and find that their reliability increases with parameter count. Larger models show overall stronger alignment with left-leaning parties but differ among policy programs: They evince a (left-wing) positive stance to
    
[^5]: 语言模型认为哪些证据令人信服？

    What Evidence Do Language Models Find Convincing?

    [https://arxiv.org/abs/2402.11782](https://arxiv.org/abs/2402.11782)

    通过构建 ConflictingQA 数据集，并进行敏感性和反事实分析，研究发现当前语言模型在预测时很大程度上依赖于网站与查询的相关性，而忽视了人类认为重要的文本风格特征。

    

    检索增强型语言模型越来越多地被赋予主观、有争议和矛盾的查询任务，如“阿斯巴甜是否与癌症有关”。为了解决这些模糊的查询，我们必须搜索大量网站，并考虑“我认为哪些证据是令人信服的？”。在这项工作中，我们研究了语言模型是如何回答这个问题的。特别是，我们构建了一个名为 ConflictingQA 的数据集，将有争议的查询与一系列包含不同事实（如定量结果）、论证风格（如权威呼声）和答案（是或否）的真实世界证据文档配对。我们使用这个数据集进行敏感性和反事实分析，探讨哪些文本特征最影响语言模型的预测。总体而言，我们发现当前模型在很大程度上依赖网站与查询的相关性，而在很大程度上忽视了人类认为重要的风格特征，比如文本是否是

    arXiv:2402.11782v1 Announce Type: new  Abstract: Retrieval-augmented language models are being increasingly tasked with subjective, contentious, and conflicting queries such as "is aspartame linked to cancer". To resolve these ambiguous queries, one must search through a large range of websites and consider "which, if any, of this evidence do I find convincing?". In this work, we study how LLMs answer this question. In particular, we construct ConflictingQA, a dataset that pairs controversial queries with a series of real-world evidence documents that contain different facts (e.g., quantitative results), argument styles (e.g., appeals to authority), and answers (Yes or No). We use this dataset to perform sensitivity and counterfactual analyses to explore which text features most affect LLM predictions. Overall, we find that current models rely heavily on the relevance of a website to the query, while largely ignoring stylistic features that humans find important such as whether a text 
    
[^6]: 从古怪的语言模型中调取潜在知识

    Eliciting Latent Knowledge from Quirky Language Models

    [https://arxiv.org/abs/2312.01037](https://arxiv.org/abs/2312.01037)

    本研究通过引入一套“古怪”的语言模型，调取了这些模型在特定上下文中的潜在知识，展示了从可信度低的模型中调取可靠知识的前景。

    

    调取潜在知识（ELK）旨在在一个能力强大的神经网络的激活中找到模式，即使网络的明显输出是错误或误导性的，也能稳定跟踪世界的真实状态。为了进一步研究ELK，我们引入了12个数据集和一套相应的“古怪”的语言模型，这些模型在回答问题时，只有在提示中包含关键词“Bob”时才会进行系统性错误的微调。我们证明了简单的探测方法可以调取模型在这些上下文中对正确答案的潜在知识，即使问题比探测器训练的问题更困难。这是由于中间层激活中的上下文无关的知识表示的存在。我们还发现，一种机械的异常检测方法可以以94%的AUROC标识不真实行为。我们的结果显示，从能力强但不受信任的模型中调取可靠的知识，并促进未来研究ELK方法的实证研究是有希望的。

    Eliciting Latent Knowledge (ELK) aims to find patterns in a capable neural network's activations which robustly track the true state of the world, even when the network's overt output is false or misleading. To further ELK research, we introduce 12 datasets and a corresponding suite of "quirky" language models that are LoRA finetuned to make systematic errors when answering questions if and only if the keyword "Bob" is present in the prompt. We demonstrate that simple probing methods can elicit the model's latent knowledge of the correct answer in these contexts, even for problems harder than those the probe was trained on. This is enabled by context-independent knowledge representations located in middle layer activations. We also find that a mechanistic anomaly detection approach can flag untruthful behavior with 94% AUROC. Our results show promise for eliciting reliable knowledge from capable but untrusted models, and facilitates future research empirically investigating ELK methods
    
[^7]: 部分音标化：一种上下文对比推理方法

    Partial Diacritization: A Context-Contrastive Inference Approach. (arXiv:2401.08919v1 [cs.CL])

    [http://arxiv.org/abs/2401.08919](http://arxiv.org/abs/2401.08919)

    部分音标化是选择标记部分字符来提高阅读可读性和准确性的新方法。上下文对比的部分音标化（CCPD）集成了现有的阿拉伯音标化系统，并通过衡量部分音标化的新指标来判断需要标记哪些字符。

    

    音标化在提高阿拉伯文本可读性和消除歧义方面起着关键作用。目前的努力主要集中在标记每个符合条件的字符（全音标化）。相比之下，部分音标化（PD）是选择标记子集以在必要时提供帮助。研究表明，过多的音标符号会妨碍熟练读者，降低阅读速度和准确性。我们进行了一项行为实验，并显示出部分标记的文本通常比完全标记的文本更容易阅读，有时甚至比纯文本更容易。在这种情况下，我们介绍了上下文对比的部分音标化（CCPD）-一种与现有阿拉伯音标化系统无缝集成的新方法。CCPD对每个单词进行两次处理，一次有上下文，一次没有，并且只对两次推理之间存在差异的字符进行音标化。此外，我们还引入了衡量部分音标化的新指标。

    Diacritization plays a pivotal role in improving readability and disambiguating the meaning of Arabic texts. Efforts have so far focused on marking every eligible character (Full Diacritization). Comparatively overlooked, Partial Diacritzation (PD) is the selection of a subset of characters to be marked to aid comprehension where needed. Research has indicated that excessive diacritic marks can hinder skilled readers--reducing reading speed and accuracy. We conduct a behavioral experiment and show that partially marked text is often easier to read than fully marked text, and sometimes easier than plain text. In this light, we introduce Context-Contrastive Partial Diacritization (CCPD)--a novel approach to PD which integrates seamlessly with existing Arabic diacritization systems. CCPD processes each word twice, once with context and once without, and diacritizes only the characters with disparities between the two inferences. Further, we introduce novel indicators for measuring partial
    
[^8]: CodePrompt：通过Prompt学习的知识特征改进源代码相关分类

    CodePrompt: Improving Source Code-Related Classification with Knowledge Features through Prompt Learning. (arXiv:2401.05544v1 [cs.CL])

    [http://arxiv.org/abs/2401.05544](http://arxiv.org/abs/2401.05544)

    CodePrompt是一种利用Prompt学习和注意机制技术改进源代码相关分类任务的新方法。它能够提取源代码和相关文本中的丰富知识以提高准确性，并且减少了计算成本。

    

    研究人员已经探索利用预训练语言模型（如CodeBERT）改进源代码相关任务的潜力。先前的研究主要依赖CodeBERT的文本嵌入能力和"[CLS]"句子嵌入信息作为下游源代码相关任务的语义表示进行微调。然而，这些方法需要额外的神经网络层来提取有效特征，导致计算成本更高。此外，现有方法没有利用源代码和相关文本中丰富的知识，可能导致准确性降低。本文提出了一种新的方法CodePrompt，通过Prompt学习和注意机制利用预训练模型中的丰富知识来改进源代码相关分类任务。

    Researchers have explored the potential of utilizing pre-trained language models, such as CodeBERT, to improve source code-related tasks. Previous studies have mainly relied on CodeBERT's text embedding capability and the `[CLS]' sentence embedding information as semantic representations for fine-tuning downstream source code-related tasks. However, these methods require additional neural network layers to extract effective features, resulting in higher computational costs. Furthermore, existing approaches have not leveraged the rich knowledge contained in both source code and related text, which can lead to lower accuracy. This paper presents a novel approach, CodePrompt, which utilizes rich knowledge recalled from a pre-trained model by prompt learning and an attention mechanism to improve source code-related classification tasks. Our approach initially motivates the language model with prompt information to retrieve abundant knowledge associated with the input as representative feat
    
[^9]: MVMR: 在多个可靠视频集中评估自然语言视频定位偏差

    MVMR: Evaluating Natural Language Video Localization Bias over Multiple Reliable Videos Pool. (arXiv:2309.16701v1 [cs.CV])

    [http://arxiv.org/abs/2309.16701](http://arxiv.org/abs/2309.16701)

    本文提出了一个名为MVMR的任务，旨在给定文本查询从大量视频集中定位视频帧。我们通过已有数据集进行相似性筛选来构建数据集，并引入三个MVMR数据集。我们采用了嵌入式文本相似度匹配和视频-语言对齐技术来计算相关性得分，并为MVMR任务开发了一个强大的模型，Reliable Mutual Matching Network (RMMN)。

    

    随着近年来多媒体内容的激增，自然语言视频定位成为一个关键问题，它致力于检测与给定自然语言查询匹配的视频片段。然而，以往的研究都没有探索在存在多个正负视频的大量语料库中定位一个时刻。本文提出了一个名为MVMR（Massive Videos Moment Retrieval）的任务，旨在给定文本查询从大量视频集中定位视频帧。对于这个任务，我们提出了一种通过对现有视频定位数据集进行相似性筛选来构建数据集的方法，并引入了三个MVMR数据集。具体来说，我们采用基于嵌入的文本相似度匹配和视频-语言对齐技术来计算目标查询与视频之间的相关性得分，从而定义正负集。针对提出的MVMR任务，我们进一步开发了一个强大的模型，Reliable Mutual Matching Network (RMMN)。

    With the explosion of multimedia content in recent years, natural language video localization, which focuses on detecting video moment that matches a given natural language query, has become a critical problem. However, none of the previous research explores localizing a moment from a large corpus where multiple positive and negative videos exist. In this paper, we propose an MVMR (Massive Videos Moment Retrieval) task, which aims to localize video frames from a massive set of videos given a text query. For this task, we suggest methods for constructing datasets by employing similarity filtering on the existing video localization datasets and introduce three MVMR datasets. Specifically, we employ embedding-based text similarity matching and video-language grounding techniques to calculate the relevance score between a target query and videos to define positive and negative sets. For the proposed MVMR task, we further develop a strong model, Reliable Mutual Matching Network (RMMN), whic
    
[^10]: 印度也存在种姓主义但不存在种族主义吗？量化印度和西方大型语言模型偏见的差异

    Casteist but Not Racist? Quantifying Disparities in Large Language Model Bias between India and the West. (arXiv:2309.08573v1 [cs.CL])

    [http://arxiv.org/abs/2309.08573](http://arxiv.org/abs/2309.08573)

    本研究量化了大型语言模型在印度和西方上的陈规偏见差异，并开发了一个新的数据集来评估种姓和宗教上的刻板印象。研究发现大多数测试的模型在印度背景下对刻板印象有显著偏见，尤其是与西方背景相比。此外，研究探索了一种简单干预方法来减轻这种偏见的效果。

    

    大型语言模型（LLMs）现在每天被数百万用户使用，他们能够传达社会偏见，使用户遭受再现伤害。已有大量的关于LLM偏见的学术研究存在，但主要采用西方中心视角，相对较少关注全球南方地区的偏见水平和潜在伤害。在本文中，我们量化流行LLMs中的陈规偏见，采用以印度为中心的框架，并比较印度和西方背景下的偏见水平。为此，我们开发了一个新颖的数据集，称为Indian-BhED（印度偏见评估数据集），其中包含种姓和宗教上的刻板和反刻板的例子。我们发现，在印度背景下，大多数测试的LLMs对刻板印象有强烈偏见，尤其是与西方背景相比。最后，我们研究了Instruction Prompting作为一种简单的干预手段来减轻这种偏见，并发现它显著减少了刻板印象和反刻板印象。

    Large Language Models (LLMs), now used daily by millions of users, can encode societal biases, exposing their users to representational harms. A large body of scholarship on LLM bias exists but it predominantly adopts a Western-centric frame and attends comparatively less to bias levels and potential harms in the Global South. In this paper, we quantify stereotypical bias in popular LLMs according to an Indian-centric frame and compare bias levels between the Indian and Western contexts. To do this, we develop a novel dataset which we call Indian-BhED (Indian Bias Evaluation Dataset), containing stereotypical and anti-stereotypical examples for caste and religion contexts. We find that the majority of LLMs tested are strongly biased towards stereotypes in the Indian context, especially as compared to the Western context. We finally investigate Instruction Prompting as a simple intervention to mitigate such bias and find that it significantly reduces both stereotypical and anti-stereoty
    

