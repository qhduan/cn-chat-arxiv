# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Assessing the Reasoning Abilities of ChatGPT in the Context of Claim Verification](https://arxiv.org/abs/2402.10735) | 我们提出了一个逻辑推理框架，用于评估ChatGPT在声明验证中的推理能力，发现其在归纳推理方面存在困难，并提出了一种缓解方法。 |
| [^2] | [Evolution of ESG-focused DLT Research: An NLP Analysis of the Literature.](http://arxiv.org/abs/2308.12420) | 本研究通过NLP分析了ESG主导的DLT研究的演化，通过构建引用网络和命名实体识别任务，对DLT在ESG背景下的发展进行了文献综述。 |
| [^3] | [FigCaps-HF: A Figure-to-Caption Generative Framework and Benchmark with Human Feedback.](http://arxiv.org/abs/2307.10867) | FigCaps-HF是一个图像生成标题的框架，可以通过融入领域专家的反馈意见，生成符合读者偏好的高质量图像标题。将自动评估和强化学习与人类反馈相结合，可以改善生成的标题与读者偏好的一致性。 |
| [^4] | [Compression of enumerations and gain.](http://arxiv.org/abs/2304.03030) | 本文研究了枚举的可压缩性对于计算可枚举集合的相对Kolmogorov复杂度的影响，并证明了任何计算可枚举集合都可以进行强压缩和无增益弱压缩。 |

# 详细

[^1]: 在声明验证的背景下评估ChatGPT的推理能力

    Assessing the Reasoning Abilities of ChatGPT in the Context of Claim Verification

    [https://arxiv.org/abs/2402.10735](https://arxiv.org/abs/2402.10735)

    我们提出了一个逻辑推理框架，用于评估ChatGPT在声明验证中的推理能力，发现其在归纳推理方面存在困难，并提出了一种缓解方法。

    

    当前有关LLMs的推理能力的辩论正在日益激烈。我们从声明/谣言验证的角度来审视这个问题。我们提出了第一个逻辑推理框架，旨在将任何声明或传言与证据结合，拆分成验证所需的基本推理步骤。基于我们的框架，我们整理了两个注释集合，其中包括来自维基百科的合成数据集和源自Twitter上流传的谣言的真实数据集。我们使用它们来评估GPT-3.5-Turbo和GPT-4（以下简称为ChatGPT）在我们框架的背景下的推理能力，并提供了彻底的分析。我们的研究表明，ChatGPT在归纳推理方面存在困难，尽管可以通过使用手动的思维链路（Chain of Thought，CoT）来缓解这一问题，而非零编码（Zero Shot，ZS）和ZS CoT方法。我们的研究有助于不断增长的研究领域，表明Cha

    arXiv:2402.10735v1 Announce Type: new  Abstract: The reasoning capabilities of LLMs are currently hotly debated. We examine the issue from the perspective of claim/rumour verification. We propose the first logical reasoning framework designed to break down any claim or rumor paired with evidence into the atomic reasoning steps necessary for verification. Based on our framework, we curate two annotated collections of such claim/evidence pairs: a synthetic dataset from Wikipedia and a real-world set stemming from rumours circulating on Twitter. We use them to evaluate the reasoning capabilities of GPT-3.5-Turbo and GPT-4 (hereinafter referred to as ChatGPT) within the context of our framework, providing a thorough analysis. Our results show that ChatGPT struggles in abductive reasoning, although this can be somewhat mitigated by using manual Chain of Thought (CoT) as opposed to Zero Shot (ZS) and ZS CoT approaches. Our study contributes to the growing body of research suggesting that Cha
    
[^2]: ESG主导的DLT研究的演化：对文献进行NLP分析

    Evolution of ESG-focused DLT Research: An NLP Analysis of the Literature. (arXiv:2308.12420v1 [cs.IR])

    [http://arxiv.org/abs/2308.12420](http://arxiv.org/abs/2308.12420)

    本研究通过NLP分析了ESG主导的DLT研究的演化，通过构建引用网络和命名实体识别任务，对DLT在ESG背景下的发展进行了文献综述。

    

    分布式账本技术(DLT)迅速发展，需要全面了解其各个组成部分。然而，针对DLT的环境、可持续性和治理(ESG)组成部分的系统文献综述还不足。为填补这一空白，我们选择了107篇种子文献，构建了一个包含63,083个参考文献的引用网络，并将其精炼为24,539篇文献的语料库进行分析。然后，我们根据一个已建立的技术分类法从46篇论文中标记了命名实体，并通过找出DLT的ESG要素来完善这个分类法。利用基于transformer的语言模型，我们对一个预先训练的语言模型进行了细化调整，用于命名实体识别任务，使用我们标记的数据集。我们利用我们调整后的语言模型对语料库进行了精简，得到了505篇关键论文，通过命名实体和时间图分析，促进了对DLT在ESG背景下的演化的文献综述。

    Distributed Ledger Technologies (DLTs) have rapidly evolved, necessitating comprehensive insights into their diverse components. However, a systematic literature review that emphasizes the Environmental, Sustainability, and Governance (ESG) components of DLT remains lacking. To bridge this gap, we selected 107 seed papers to build a citation network of 63,083 references and refined it to a corpus of 24,539 publications for analysis. Then, we labeled the named entities in 46 papers according to twelve top-level categories derived from an established technology taxonomy and enhanced the taxonomy by pinpointing DLT's ESG elements. Leveraging transformer-based language models, we fine-tuned a pre-trained language model for a Named Entity Recognition (NER) task using our labeled dataset. We used our fine-tuned language model to distill the corpus to 505 key papers, facilitating a literature review via named entities and temporal graph analysis on DLT evolution in the context of ESG. Our con
    
[^3]: FigCaps-HF:一个基于人类反馈的图像生成标题框架和基准测试

    FigCaps-HF: A Figure-to-Caption Generative Framework and Benchmark with Human Feedback. (arXiv:2307.10867v1 [cs.CL])

    [http://arxiv.org/abs/2307.10867](http://arxiv.org/abs/2307.10867)

    FigCaps-HF是一个图像生成标题的框架，可以通过融入领域专家的反馈意见，生成符合读者偏好的高质量图像标题。将自动评估和强化学习与人类反馈相结合，可以改善生成的标题与读者偏好的一致性。

    

    标题对于理解科学可视化和文档至关重要。现有的科学图像生成标题方法依赖于从文档中提取的图像-标题配对进行训练，但其中许多配对在帮助性、解释性和视觉描述性等指标上存在不足，导致生成的标题与读者偏好不一致。为了能够生成高质量的图像标题，我们引入了FigCaps-HF，这是一个新的图像生成标题框架，可以融入领域专家的反馈意见，以生成优化了读者偏好的标题。我们的框架包含1）一种评估图像-标题配对质量的自动方法，2）一种基于人类反馈的强化学习（RLHF）方法，用于优化生成式图像生成标题模型以符合读者偏好。我们通过在不同类型的模型上改进性能，证明了我们简单的学习框架的有效性。

    Captions are crucial for understanding scientific visualizations and documents. Existing captioning methods for scientific figures rely on figure-caption pairs extracted from documents for training, many of which fall short with respect to metrics like helpfulness, explainability, and visual-descriptiveness [15] leading to generated captions being misaligned with reader preferences. To enable the generation of high-quality figure captions, we introduce FigCaps-HF a new framework for figure-caption generation that can incorporate domain expert feedback in generating captions optimized for reader preferences. Our framework comprises of 1) an automatic method for evaluating quality of figure-caption pairs, 2) a novel reinforcement learning with human feedback (RLHF) method to optimize a generative figure-to-caption model for reader preferences. We demonstrate the effectiveness of our simple learning framework by improving performance over standard fine-tuning across different types of mod
    
[^4]: 枚举压缩与增益

    Compression of enumerations and gain. (arXiv:2304.03030v1 [cs.CL])

    [http://arxiv.org/abs/2304.03030](http://arxiv.org/abs/2304.03030)

    本文研究了枚举的可压缩性对于计算可枚举集合的相对Kolmogorov复杂度的影响，并证明了任何计算可枚举集合都可以进行强压缩和无增益弱压缩。

    

    我们研究了枚举的可压缩性，以及其在计算可枚举集合的相对Kolmogorov复杂度中密度方面的作用。我们关注了强压缩和弱压缩，以及压缩枚举中嵌入的附加信息的数量：增益。我们证明了任何计算可枚举集合都可以进行强压缩和无增益弱压缩，并研究了位置游戏以理解强无增益压缩。

    We study the compressibility of enumerations, and its role in the relative Kolmogorov complexity of computably enumerable sets, with respect to density. With respect to a strong and a weak form of compression, we examine the gain: the amount of auxiliary information embedded in the compressed enumeration. Strong compression and weak gainless compression is shown for any computably enumerable set, and a positional game is studied toward understanding strong gainless compression.
    

