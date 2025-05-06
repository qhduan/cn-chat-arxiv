# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [ParaICL: Towards Robust Parallel In-Context Learning](https://arxiv.org/abs/2404.00570) | 提出了一种名为ParaICL的新方法，通过并行批处理来有效利用所有示例，在不超过可管理的输入上下文长度的情况下显著提升了不同测试样本的准确性。 |
| [^2] | [Learning with Noisy Foundation Models](https://arxiv.org/abs/2403.06869) | 本文首次全面了解和分析了预训练数据集中的噪声性质，有效减轻其对下游任务影响。 |
| [^3] | [DECIDER: A Rule-Controllable Decoding Strategy for Language Generation by Imitating Dual-System Cognitive Theory](https://arxiv.org/abs/2403.01954) | DECIDER是一种受双系统认知理论启发的规则可控解码策略，通过在预训练语言模型中引入逻辑推理器，有效地遵循给定规则以引导生成方向朝向目标。 |
| [^4] | [SMUTF: Schema Matching Using Generative Tags and Hybrid Features](https://arxiv.org/abs/2402.01685) | SMUTF是一种用于大规模表格数据模式匹配的独特方法，通过结合基于规则的特征工程、预训练语言模型和生成式大语言模型，并使用生成标签提高匹配效果。同时，作者开发并开源了HDXSM数据集来解决现有数据集不足的问题。 |
| [^5] | [MONOVAB : An Annotated Corpus for Bangla Multi-label Emotion Detection.](http://arxiv.org/abs/2309.15670) | 这个研究构建了一个基于孟加拉语的注释语料库，用于多标签情感检测。通过使用基于上下文的方法以及BERT模型，填补了这一学科领域的空白。 |

# 详细

[^1]: ParaICL：面向鲁棒性的并行上下文学习

    ParaICL: Towards Robust Parallel In-Context Learning

    [https://arxiv.org/abs/2404.00570](https://arxiv.org/abs/2404.00570)

    提出了一种名为ParaICL的新方法，通过并行批处理来有效利用所有示例，在不超过可管理的输入上下文长度的情况下显著提升了不同测试样本的准确性。

    

    大型语言模型(LLMs)已经成为自然语言处理(NLP)领域的常态，在少样本上下文学习(ICL)方面表现出色。然而，ICL的成功很大程度上取决于少样本演示示例的选择，使得选择过程变得愈发关键。现有方法已经开始优化这些示例的数量和语义相似性以提高ICL性能。然而，我们的初步实验表明，ICL的有效性受到输入上下文长度的限制。此外，不同的少样本演示示例组合可以显著提升对不同测试样本的准确性。为了解决这个问题，我们提出了一种名为parallel in-context learning (ParaICL)的新方法，可以有效利用所有示例而不会超出可管理的输入上下文长度。ParaICL采用并行批处理来分发演示

    arXiv:2404.00570v1 Announce Type: new  Abstract: Large language models (LLMs) have become the norm in natural language processing (NLP), excelling in few-shot in-context learning (ICL) with their remarkable abilities. Nonetheless, the success of ICL largely hinges on the choice of few-shot demonstration examples, making the selection process increasingly crucial. Existing methods have delved into optimizing the quantity and semantic similarity of these examples to improve ICL performances. However, our preliminary experiments indicate that the effectiveness of ICL is limited by the length of the input context. Moreover, varying combinations of few-shot demonstration examples can significantly boost accuracy across different test samples. To address this, we propose a novel method named parallel in-context learning (ParaICL) that effectively utilizes all demonstration examples without exceeding the manageable input context length. ParaICL employs parallel batching to distribute demonstr
    
[^2]: 在有噪声基础模型中学习

    Learning with Noisy Foundation Models

    [https://arxiv.org/abs/2403.06869](https://arxiv.org/abs/2403.06869)

    本文首次全面了解和分析了预训练数据集中的噪声性质，有效减轻其对下游任务影响。

    

    基础模型通常是在大规模数据集上进行预训练，然后通过调整来适应下游任务。然而，大规模预训练数据集往往无法获取或成本过高，可能包含标签噪声，这可能会对模型的泛化能力造成不利影响，并带来意想不到的风险。本文是首个全面了解和分析预训练数据集中噪声性质，并有效减轻其对下游任务影响的工作。具体而言，通过在合成有噪声的ImageNet-1K、YFCC15M和CC12M数据集上进行完全监督和图像-文本对比预训练的广泛实验，我们证明了，尽管预训练中的轻微噪声可以使同领域（ID）性能受益，即训练和测试数据共享类似分布，但它总是会破坏跨领域（OOD）性能，在那里训练和测试分布明显不同。

    arXiv:2403.06869v1 Announce Type: cross  Abstract: Foundation models are usually pre-trained on large-scale datasets and then adapted to downstream tasks through tuning. However, the large-scale pre-training datasets, often inaccessible or too expensive to handle, can contain label noise that may adversely affect the generalization of the model and pose unexpected risks. This paper stands out as the first work to comprehensively understand and analyze the nature of noise in pre-training datasets and then effectively mitigate its impacts on downstream tasks. Specifically, through extensive experiments of fully-supervised and image-text contrastive pre-training on synthetic noisy ImageNet-1K, YFCC15M, and CC12M datasets, we demonstrate that, while slight noise in pre-training can benefit in-domain (ID) performance, where the training and testing data share a similar distribution, it always deteriorates out-of-domain (OOD) performance, where training and testing distributions are signific
    
[^3]: DECIDERS：一种通过模仿双系统认知理论实现规则可控解码策略的语言生成方法

    DECIDER: A Rule-Controllable Decoding Strategy for Language Generation by Imitating Dual-System Cognitive Theory

    [https://arxiv.org/abs/2403.01954](https://arxiv.org/abs/2403.01954)

    DECIDER是一种受双系统认知理论启发的规则可控解码策略，通过在预训练语言模型中引入逻辑推理器，有效地遵循给定规则以引导生成方向朝向目标。

    

    词典约束解码方法旨在通过某些目标概念控制所生成文本的意义或风格。现有方法过于关注这些目标本身，导致缺乏关于如何实现这些目标的高层推理。然而，人类通常通过遵循某些规则来处理任务，这些规则不仅关注于目标本身，还关注于引发目标发生的语义相关概念。在这项工作中，我们提出了DECIDER，这是一种受到双系统认知理论启发的约束语言生成的规则可控解码策略。具体而言，在DECIDER中，一个预训练语言模型（PLM）配备了一个逻辑推理器，以高层规则作为输入。然后，DECIDER允许规则信号在每个解码步骤中流入PLM。广泛的实验结果表明，DECIDER能够有效地遵循给定的规则，引导生成方向朝向目标进行生成。

    arXiv:2403.01954v1 Announce Type: cross  Abstract: Lexicon-based constrained decoding approaches aim to control the meaning or style of the generated text through certain target concepts. Existing approaches over-focus the targets themselves, leading to a lack of high-level reasoning about how to achieve them. However, human usually tackles tasks by following certain rules that not only focuses on the targets but also on semantically relevant concepts that induce the occurrence of targets. In this work, we present DECIDER, a rule-controllable decoding strategy for constrained language generation inspired by dual-system cognitive theory. Specifically, in DECIDER, a pre-trained language model (PLM) is equiped with a logic reasoner that takes high-level rules as input. Then, the DECIDER allows rule signals to flow into the PLM at each decoding step. Extensive experimental results demonstrate that DECIDER can effectively follow given rules to guide generation direction toward the targets i
    
[^4]: SMUTF：使用生成标签和混合特征的模式匹配方法

    SMUTF: Schema Matching Using Generative Tags and Hybrid Features

    [https://arxiv.org/abs/2402.01685](https://arxiv.org/abs/2402.01685)

    SMUTF是一种用于大规模表格数据模式匹配的独特方法，通过结合基于规则的特征工程、预训练语言模型和生成式大语言模型，并使用生成标签提高匹配效果。同时，作者开发并开源了HDXSM数据集来解决现有数据集不足的问题。

    

    我们引入了SMUTF，一种用于大规模表格数据模式匹配的独特方法，该方法假设在开放域任务中，监督学习不会影响性能，从而实现了有效的跨域匹配。这个系统独特地结合了基于规则的特征工程、预训练语言模型和生成式大语言模型。受人道主义交换语言的启发，我们使用“生成标签”为每个数据列部署了创新的适应性，提高了模式匹配的效果。SMUTF具有广泛的灵活性，可以与任何现有的预训练嵌入、分类方法和生成模型无缝配合使用。鉴于模式匹配缺乏广泛的公开数据集，我们已经创建并开源了HDXSM数据集，该数据集来自公共人道主义数据，我们相信这是目前最全面的模式匹配数据集。

    We introduce SMUTF, a unique approach for large-scale tabular data schema matching (SM), which assumes that supervised learning does not affect performance in open-domain tasks, thereby enabling effective cross-domain matching. This system uniquely combines rule-based feature engineering, pre-trained language models, and generative large language models. In an innovative adaptation inspired by the Humanitarian Exchange Language, we deploy 'generative tags' for each data column, enhancing the effectiveness of SM. SMUTF exhibits extensive versatility, working seamlessly with any pre-existing pre-trained embeddings, classification methods, and generative models.   Recognizing the lack of extensive, publicly available datasets for SM, we have created and open-sourced the HDXSM dataset from the public humanitarian data. We believe this to be the most exhaustive SM dataset currently available. In evaluations across various public datasets and the novel HDXSM dataset, SMUTF demonstrated excep
    
[^5]: MONOVAB: 用于孟加拉语多标签情感检测的注释语料库

    MONOVAB : An Annotated Corpus for Bangla Multi-label Emotion Detection. (arXiv:2309.15670v1 [cs.LG])

    [http://arxiv.org/abs/2309.15670](http://arxiv.org/abs/2309.15670)

    这个研究构建了一个基于孟加拉语的注释语料库，用于多标签情感检测。通过使用基于上下文的方法以及BERT模型，填补了这一学科领域的空白。

    

    近年来，情感分析(SA)和情感识别(ER)在孟加拉语中越来越流行，孟加拉语是世界上第七大使用人数最多的语言。然而，孟加拉语的结构复杂，这使得准确提取情绪变得困难。在这个研究领域中，已经采用了一些不同的方法，例如提取积极和消极情感以及多类情绪。然而，在这种语言中提取多种情绪几乎是未开发的领域，它涉及基于一段文本识别出多种情感。因此，本研究展示了一种基于从Facebook上抓取的数据构建注释语料库的详细方法，以填补这个学科领域的空白，克服挑战。为了使这种注释更有成果，采用了基于上下文的方法。转换器中的双向编码器表示(BERT)。

    In recent years, Sentiment Analysis (SA) and Emotion Recognition (ER) have been increasingly popular in the Bangla language, which is the seventh most spoken language throughout the entire world. However, the language is structurally complicated, which makes this field arduous to extract emotions in an accurate manner. Several distinct approaches such as the extraction of positive and negative sentiments as well as multiclass emotions, have been implemented in this field of study. Nevertheless, the extraction of multiple sentiments is an almost untouched area in this language. Which involves identifying several feelings based on a single piece of text. Therefore, this study demonstrates a thorough method for constructing an annotated corpus based on scrapped data from Facebook to bridge the gaps in this subject area to overcome the challenges. To make this annotation more fruitful, the context-based approach has been used. Bidirectional Encoder Representations from Transformers (BERT),
    

