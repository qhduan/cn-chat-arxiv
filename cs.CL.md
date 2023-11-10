# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Ensemble of Task-Specific Language Models for Brain Encoding.](http://arxiv.org/abs/2310.15720) | 该论文提出了一种用于大脑编码的任务特定语言模型集成方法，通过将10个流行的语言模型进行集成，相较于当前基准线，平均提高了10%的性能。 |
| [^2] | [Bridging Information-Theoretic and Geometric Compression in Language Models.](http://arxiv.org/abs/2310.13620) | 本文提出了从几何和信息论的角度分析语言模型（LM）中的压缩问题，并展示了这两个视角高度相关的关系。此外，我们还发现语言数据的高压缩预测了对该数据集的快速适应。 |
| [^3] | [Cross-Lingual Consistency of Factual Knowledge in Multilingual Language Models.](http://arxiv.org/abs/2310.10378) | 本论文研究了多语言预训练语言模型中事实知识的跨语言一致性，提出了一种新的度量方法，并通过分析模型大小、语言配对等因素发现了影响一致性的因素。实验结果表明，增加模型大小可以提高准确性，但不会改善跨语言一致性。 |
| [^4] | [Statistical Mechanics of Strahler Number via Random and Natural Language Sentences.](http://arxiv.org/abs/2307.02697) | 本文通过统计力学分析自然语言句子树结构的Strahler数的上下限，发现它几乎总是3或4，并证明它是处理句子所需记忆量的下限。同时，对随机树进行的分析揭示出Strahler数的增长模式，揭示了它作为自然语言句子特征的统计基础。 |
| [^5] | [Surveying (Dis)Parities and Concerns of Compute Hungry NLP Research.](http://arxiv.org/abs/2306.16900) | 这项研究调查了自然语言处理领域计算需求量大的研究中存在的不平等和担忧，通过对NLP社区的312位参与者进行调查，发现了在资历、学术界和工业界等方面存在的（不）平等现象，并提出了相应的缓解建议。 |
| [^6] | [Quantizable Transformers: Removing Outliers by Helping Attention Heads Do Nothing.](http://arxiv.org/abs/2306.12929) | 本文提出了一种称为“Helper-Head”的方法，可以通过教授注意头忽略输入和输出的某些部分来消除离群值，从而实现对transformer的可量化。实验结果表明，该方法在低、高比特宽度设置上均优于现有方法，在两个流行的语言建模基准上实现了最先进的结果。 |
| [^7] | [Large Language Models are Fixated by Red Herrings: Exploring Creative Problem Solving and Einstellung Effect using the Only Connect Wall Dataset.](http://arxiv.org/abs/2306.11167) | 这项研究探索了大型语言模型（LLMs）对创造性问题解决的能力，并发现大型语言模型容易被误导，出现固定效应和Einstellung范式。 |
| [^8] | [Advancements in Arabic Grammatical Error Detection and Correction: An Empirical Investigation.](http://arxiv.org/abs/2305.14734) | 本文使用了两个预训练的序列到序列模型在阿拉伯语GEC方面取得了最新的结果，并提出下文第一阿拉伯语多类语法错误检测结果。文章表明，使用GED作为GEC模型的辅助输入有助于提高性能，而上下文形态预处理有助于接着GEC系统，文章在两个共享任务数据集上取得了最新的GEC结果以及一个强有力的基准。 |
| [^9] | [Let's Think Frame by Frame: Evaluating Video Chain of Thought with Video Infilling and Prediction.](http://arxiv.org/abs/2305.13903) | 该论文提出了一种新的研究方向 VideoCOT，利用视觉-语言模型的多模态生成能力，以增强视频推理，同时减少处理数百或数千帧的计算复杂度。在VIP数据集上，我们基于各种视觉-语言模型进行了基准测试，展示了使用视觉-语言模型进行VideoCOT的潜力。 |
| [^10] | [Sabi\'a: Portuguese Large Language Models.](http://arxiv.org/abs/2304.07880) | 针对葡萄牙语进行单语言预训练，可以显著提高大规模合成语言模型的质量，并能够在一系列葡萄牙语数据集上优于以英语为中心和多语言的对手，最好的模型的表现与GPT-3.5-turbo持平。 |
| [^11] | [MLRegTest: A Benchmark for the Machine Learning of Regular Languages.](http://arxiv.org/abs/2304.07687) | 本文提出了一个名为MLRegTest的新基准测试，其包含了来自1,800个正则语言的数据集。该测试根据逻辑复杂度和逻辑文字种类组织语言，并可以帮助我们了解机器学习系统在学习不同种类的长距离依赖方面的性能。 |
| [^12] | [Vicarious Offense and Noise Audit of Offensive Speech Classifiers: Unifying Human and Machine Disagreement on What is Offensive.](http://arxiv.org/abs/2301.12534) | 本文通过人工和机器审核员的共情冒犯和噪声审计研究了冒犯性言论检测中的差异性。结果表明，审核员之间存在广泛的分歧，并且人工审核员和大型语言模型分类器无法预测其他审核员的回应。这对于内容审核具有重要意义。 |
| [^13] | [Learning From How Humans Correct.](http://arxiv.org/abs/2102.00225) | 本研究提出了一种从人类矫正中学习的方法。通过标注数据中的噪声数据，收集纠错信息，并将其注入至深度学习模型中，成功将文本分类准确度提升了1.7个百分点。 |

# 详细

[^1]: 用于大脑编码的任务特定语言模型集成

    Ensemble of Task-Specific Language Models for Brain Encoding. (arXiv:2310.15720v1 [cs.CL])

    [http://arxiv.org/abs/2310.15720](http://arxiv.org/abs/2310.15720)

    该论文提出了一种用于大脑编码的任务特定语言模型集成方法，通过将10个流行的语言模型进行集成，相较于当前基准线，平均提高了10%的性能。

    

    先前的研究表明，语言模型足够丰富，可以编码我们大脑中特定兴趣区域的fMRI激活情况。先前的工作已经探索了从为流行的自然语言处理任务学习的表示向预测大脑响应的转移学习。我们的工作通过创建一个由10个流行的语言模型（2个句法和8个语义）组成的集成模型，提高了这样的编码器的性能。通过我们的集成方法，在所有兴趣区域中，我们将当前的基准线提高了平均10%。

    Language models have been shown to be rich enough to encode fMRI activations of certain Regions of Interest in our Brains. Previous works have explored transfer learning from representations learned for popular natural language processing tasks for predicting brain responses. In our work, we improve the performance of such encoders by creating an ensemble model out of 10 popular Language Models (2 syntactic and 8 semantic). We beat the current baselines by 10% on average across all ROIs through our ensembling methods.
    
[^2]: 在语言模型中连接信息论和几何压缩

    Bridging Information-Theoretic and Geometric Compression in Language Models. (arXiv:2310.13620v1 [cs.CL])

    [http://arxiv.org/abs/2310.13620](http://arxiv.org/abs/2310.13620)

    本文提出了从几何和信息论的角度分析语言模型（LM）中的压缩问题，并展示了这两个视角高度相关的关系。此外，我们还发现语言数据的高压缩预测了对该数据集的快速适应。

    

    为了真实地模拟人类语言，语言模型（LM）必须将大量的、潜在无限的信息压缩到相对较少的维度中。我们提出从几何和信息论的两个角度分析（预训练的）LM的压缩。我们证明了这两个视角高度相关，语言数据的内在几何维度可以预测它们在LM下的编码长度。然后我们展示了，进一步，语言数据集的高压缩预测了对该数据集的快速适应，确认了能够压缩语言信息是成功LM性能的重要部分。作为我们分析的实际副产品，我们首次评估了一系列在语言数据上的内在维度估计器，表明只有一部分封装了信息论压缩、几何压缩和适应度之间的关系。

    For a language model (LM) to faithfully model human language, it must compress vast, potentially infinite information into relatively few dimensions. We propose analyzing compression in (pre-trained) LMs from two points of view: geometric and information-theoretic. We demonstrate that the two views are highly correlated, such that the intrinsic geometric dimension of linguistic data predicts their coding length under the LM. We then show that, in turn, high compression of a linguistic dataset predicts rapid adaptation to that dataset, confirming that being able to compress linguistic information is an important part of successful LM performance. As a practical byproduct of our analysis, we evaluate a battery of intrinsic dimension estimators for the first time on linguistic data, showing that only some encapsulate the relationship between information-theoretic compression, geometric compression, and ease-of-adaptation.
    
[^3]: 跨语言多语言模型中事实知识的跨语言一致性

    Cross-Lingual Consistency of Factual Knowledge in Multilingual Language Models. (arXiv:2310.10378v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2310.10378](http://arxiv.org/abs/2310.10378)

    本论文研究了多语言预训练语言模型中事实知识的跨语言一致性，提出了一种新的度量方法，并通过分析模型大小、语言配对等因素发现了影响一致性的因素。实验结果表明，增加模型大小可以提高准确性，但不会改善跨语言一致性。

    

    多语言大规模预训练语言模型（PLM）显示存储了大量的事实知识，但在不同语言之间存在较大的变化。为了确保不同语言背景的用户从同一个模型中获得一致的反馈，我们研究了各种多语言PLM中事实知识的跨语言一致性（CLC）。为此，我们提出了一种基于排序的一致性（RankC）度量，用于独立于准确性评估跨语言间的知识一致性。利用这个度量方法，我们对决定CLC的因素进行了深入分析，包括模型层面和语言对层面。在其他结果中，我们发现增加模型大小可以提高大多数语言中的事实探测准确性，但不能改善跨语言一致性。最后，我们通过模型编辑在PLMs中插入新的事实关联进行了一个CLC的案例研究。对一小部分事实进行了实验。

    Multilingual large-scale Pretrained Language Models (PLMs) have been shown to store considerable amounts of factual knowledge, but large variations are observed across languages. With the ultimate goal of ensuring that users with different language backgrounds obtain consistent feedback from the same model, we study the cross-lingual consistency (CLC) of factual knowledge in various multilingual PLMs. To this end, we propose a Ranking-based Consistency (RankC) metric to evaluate knowledge consistency across languages independently from accuracy. Using this metric, we conduct an in-depth analysis of the determining factors for CLC, both at model level and at language-pair level. Among other results, we find that increasing model size leads to higher factual probing accuracy in most languages, but does not improve cross-lingual consistency. Finally, we conduct a case study on CLC when new factual associations are inserted in the PLMs via model editing. Results on a small sample of facts 
    
[^4]: Strahler数的统计力学：基于随机和自然语言句子的研究

    Statistical Mechanics of Strahler Number via Random and Natural Language Sentences. (arXiv:2307.02697v1 [cs.CL])

    [http://arxiv.org/abs/2307.02697](http://arxiv.org/abs/2307.02697)

    本文通过统计力学分析自然语言句子树结构的Strahler数的上下限，发现它几乎总是3或4，并证明它是处理句子所需记忆量的下限。同时，对随机树进行的分析揭示出Strahler数的增长模式，揭示了它作为自然语言句子特征的统计基础。

    

    Strahler数最初被提出用于描述河流分支的复杂性，并找到了各种应用。本文提出了计算自然语言句子树结构的Strahler数上下限的方法，这些结构可以在一个大型数据集中进行统计力学分析。通过对语法注释数据的经验性测量，显示自然语言句子的Strahler数几乎总是3或4，与Strahler（1957年）和Horton（1945年）报道的河流分流情况类似。从该数值的理论观点出发，我们证明它是在特定模型下处理句子所需记忆量的下限。对随机树进行的数学分析进一步假设了Strahler数的性质，揭示出它并非常数而是以对数形式增长。这一发现揭示了Strahler数作为描述自然语言句子特征的统计基础。

    The Strahler number was originally proposed to characterize the complexity of river bifurcation and has found various applications. This article proposes computation of the Strahler number's upper and lower limits for natural language sentence tree structures, which are available in a large dataset allowing for statistical mechanics analysis.  Through empirical measurements across grammatically annotated data, the Strahler number of natural language sentences is shown to be almost always 3 or 4, similar to the case of river bifurcation as reported by Strahler (1957) and Horton (1945).  From the theory behind the number, we show that it is the lower limit of the amount of memory required to process sentences under a particular model. A mathematical analysis of random trees provides a further conjecture on the nature of the Strahler number, revealing that it is not a constant but grows logarithmically. This finding uncovers the statistical basics behind the Strahler number as a character
    
[^5]: 调查计算需求量大的自然语言处理研究中的不平等和担忧

    Surveying (Dis)Parities and Concerns of Compute Hungry NLP Research. (arXiv:2306.16900v1 [cs.CL])

    [http://arxiv.org/abs/2306.16900](http://arxiv.org/abs/2306.16900)

    这项研究调查了自然语言处理领域计算需求量大的研究中存在的不平等和担忧，通过对NLP社区的312位参与者进行调查，发现了在资历、学术界和工业界等方面存在的（不）平等现象，并提出了相应的缓解建议。

    

    自然语言处理领域的许多最新进展源于开发和使用具有数十亿参数的大规模预训练语言模型（PLM）。大模型的规模使得计算成本成为训练和评估这些模型的主要限制因素之一；并且对于研究PLMs的可持续性、可重复性和包容性引发了严重的担忧。这些担忧往往基于个人经验和观察。然而，迄今为止还没有进行大规模调查来调查这些担忧。在这项工作中，我们首次尝试量化与环境影响、公平性和同行评审影响相关的这些担忧。通过对NLP社区的312位参与者进行调查，我们捕捉到不同群体内部和之间的现有（不）平等现象，包括资历、学术界和工业界，以及它们对同行评审过程的影响。对于每个主题，我们提供了分析结果，并提出了相应的缓解建议。

    Many recent improvements in NLP stem from the development and use of large pre-trained language models (PLMs) with billions of parameters. Large model sizes makes computational cost one of the main limiting factors for training and evaluating such models; and has raised severe concerns about the sustainability, reproducibility, and inclusiveness for researching PLMs. These concerns are often based on personal experiences and observations. However, there had not been any large-scale surveys that investigate them. In this work, we provide a first attempt to quantify these concerns regarding three topics, namely, environmental impact, equity, and impact on peer reviewing. By conducting a survey with 312 participants from the NLP community, we capture existing (dis)parities between different and within groups with respect to seniority, academia, and industry; and their impact on the peer reviewing process. For each topic, we provide an analysis and devise recommendations to mitigate found 
    
[^6]: 可量化Transformer：通过帮助注意力头“什么也不做”去除离群值

    Quantizable Transformers: Removing Outliers by Helping Attention Heads Do Nothing. (arXiv:2306.12929v1 [cs.LG])

    [http://arxiv.org/abs/2306.12929](http://arxiv.org/abs/2306.12929)

    本文提出了一种称为“Helper-Head”的方法，可以通过教授注意头忽略输入和输出的某些部分来消除离群值，从而实现对transformer的可量化。实验结果表明，该方法在低、高比特宽度设置上均优于现有方法，在两个流行的语言建模基准上实现了最先进的结果。

    

    过去几年里，Transformer模型已经被广泛应用于各个领域，特别是大型语言模型已经显著推进了人工智能领域的发展。由于其规模，这些网络的能力已经大大增强，但这是以极大的计算成本为代价的。量化是减少神经网络计算时间和存储器消耗的最有效方法之一。然而，许多研究表明，现代transformer模型往往学习到其激活中的强离群值，这使得它们难以量化。为保持可接受的性能，这些离群值的存在需要将激活置于更高的比特宽度或使用不同的数字格式，进行额外的微调或其他变通方法。本文展示了强离群值与特定注意头行为的相关性，这些头试图学习“无操作”或仅仅是部分残差更新。为了实现注意力头中需要的精确零位，我们引入了一个称为“Helper-Head”的方法，教授注意力头忽略输入和输出的某些部分。我们还引入了一种利用这些额外信息的量化技术，可以使用低精度量化甚至是强离群数据。在几个基准数据集上的实验证明，我们的方法在低、高比特宽度设置上均优于现有方法，在两个流行的语言建模基准上实现了最先进的结果。

    Transformer models have been widely adopted in various domains over the last years, and especially large language models have advanced the field of AI significantly. Due to their size, the capability of these networks has increased tremendously, but this has come at the cost of a significant increase in necessary compute. Quantization is one of the most effective ways to reduce the computational time and memory consumption of neural networks. Many studies have shown, however, that modern transformer models tend to learn strong outliers in their activations, making them difficult to quantize. To retain acceptable performance, the existence of these outliers requires activations to be in higher bitwidth or the use of different numeric formats, extra fine-tuning, or other workarounds. We show that strong outliers are related to very specific behavior of attention heads that try to learn a "no-op" or just a partial update of the residual. To achieve the exact zeros needed in the attention 
    
[^7]: 大型语言模型被误导：使用Only Connect Wall数据集探索创造性问题解决和Einstellung效应。

    Large Language Models are Fixated by Red Herrings: Exploring Creative Problem Solving and Einstellung Effect using the Only Connect Wall Dataset. (arXiv:2306.11167v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2306.11167](http://arxiv.org/abs/2306.11167)

    这项研究探索了大型语言模型（LLMs）对创造性问题解决的能力，并发现大型语言模型容易被误导，出现固定效应和Einstellung范式。

    

    自从人工智能诞生以来，对人类仿真智能的追求一直是人工智能研究的持久话题。最新一代的大型语言模型（LLM）的技术演进和新兴能力将这个主题从学术界带到了文化时代。尽管最近的NLP评估基准任务测试了人类仿真行为的一些方面（例如BIG-bench的“类人行为”任务），但几乎没有一个任务考察创造性问题解决能力。人类的创造性问题解决是认知神经科学中研究较为深入的主题，标准化测试主要使用将线索词之间的（异构）连接能力作为创造性的度量。在这样的任务中，暗示性的误导性刺激-被称为“诱导误解”的干扰因素-通过固定效应和Einstellung范式阻碍了人类的表现。在认知神经科学的研究中，通过事先让参与者接触到有相似拼写的错误因素来实验性地诱导这样的固定。

    The quest for human imitative AI has been an enduring topic in AI research since its inception. The technical evolution and emerging capabilities of the latest cohort of large language models (LLMs) have reinvigorated the subject beyond academia to the cultural zeitgeist. While recent NLP evaluation benchmark tasks test some aspects of human-imitative behaviour (e.g., BIG-bench's 'human-like behavior' tasks), few, if not none, examine creative problem solving abilities. Creative problem solving in humans is a well-studied topic in cognitive neuroscience with standardized tests that predominantly use the ability to associate (heterogeneous) connections among clue words as a metric for creativity. Exposure to misleading stimuli - distractors dubbed red herrings - impede human performance in such tasks via the fixation effect and Einstellung paradigm. In cognitive neuroscience studies, such fixations are experimentally induced by pre-exposing participants to orthographically similar incor
    
[^8]: 阿拉伯语语法错误检测和修正的进展：一项实证调查

    Advancements in Arabic Grammatical Error Detection and Correction: An Empirical Investigation. (arXiv:2305.14734v1 [cs.CL])

    [http://arxiv.org/abs/2305.14734](http://arxiv.org/abs/2305.14734)

    本文使用了两个预训练的序列到序列模型在阿拉伯语GEC方面取得了最新的结果，并提出下文第一阿拉伯语多类语法错误检测结果。文章表明，使用GED作为GEC模型的辅助输入有助于提高性能，而上下文形态预处理有助于接着GEC系统，文章在两个共享任务数据集上取得了最新的GEC结果以及一个强有力的基准。

    

    语法错误纠正(GEC)是一个经过深入研究的英语问题，在许多已有的模型和数据集的基础上进行了研究。然而，由于数据稀缺和语言复杂性等挑战，对形态丰富的语言进行GEC研究的限制较多。本文通过使用两个新开发的基于Transformer预训练序列到序列模型，首次在阿拉伯语GEC中获得了结果。我们解决了多类阿拉伯语语法错误检测(GED)任务，并首次在多类阿拉伯语GED上获得了结果。我们表明，在跨越不同类型的三个数据集上使用GED信息作为辅助输入的GEC模型可以提高GEC性能。此外，我们还研究了上下文形态预处理在帮助GEC系统方面的应用。我们的模型在两个阿拉伯语GEC共享任务数据集上取得了最新结果，并在新创建的数据集上建立了一个强有力的基准。

    Grammatical error correction (GEC) is a well-explored problem in English with many existing models and datasets. However, research on GEC in morphologically rich languages has been limited due to challenges such as data scarcity and language complexity. In this paper, we present the first results on Arabic GEC by using two newly developed Transformer-based pretrained sequence-to-sequence models. We address the task of multi-class Arabic grammatical error detection (GED) and present the first results on multi-class Arabic GED. We show that using GED information as auxiliary input in GEC models improves GEC performance across three datasets spanning different genres. Moreover, we also investigate the use of contextual morphological preprocessing in aiding GEC systems. Our models achieve state-of-the-art results on two Arabic GEC shared tasks datasets and establish a strong benchmark on a newly created dataset.
    
[^9]: 让我们逐帧思考：使用视频插帧和预测评估视频思维链

    Let's Think Frame by Frame: Evaluating Video Chain of Thought with Video Infilling and Prediction. (arXiv:2305.13903v1 [cs.CL])

    [http://arxiv.org/abs/2305.13903](http://arxiv.org/abs/2305.13903)

    该论文提出了一种新的研究方向 VideoCOT，利用视觉-语言模型的多模态生成能力，以增强视频推理，同时减少处理数百或数千帧的计算复杂度。在VIP数据集上，我们基于各种视觉-语言模型进行了基准测试，展示了使用视觉-语言模型进行VideoCOT的潜力。

    

    尽管在2023年构成了所有互联网流量的65％，但视频内容在生成AI研究中却被低估了。与此同时，最近的大型语言模型（LLM）已越来越多地与视觉模态融合。将视频与LLM整合是下一步自然的发展方向，那么这个鸿沟如何被填补？为了推进视频推理，我们提出了一个新的研究方向，即基于视频关键帧的VideoCOT，它利用了视觉-语言模型的多模态生成能力，以增强视频推理，同时减少处理数百或数千帧的计算复杂度。我们介绍了VIP，一种可以用来评估VideoCOT的推断时间数据集，其中包含1）各种带有关键帧的真实生活视频以及相应的非结构化和结构化场景描述，2）两个新的视频推理任务：视频插帧和场景预测。我们在VIP上对各种视觉-语言模型进行了基准测试，展示了使用视觉-语言模型进行VideoCOT的潜力。

    Despite constituting 65% of all internet traffic in 2023, video content is underrepresented in generative AI research. Meanwhile, recent large language models (LLMs) have become increasingly integrated with capabilities in the visual modality. Integrating video with LLMs is a natural next step, so how can this gap be bridged? To advance video reasoning, we propose a new research direction of VideoCOT on video keyframes, which leverages the multimodal generative abilities of vision-language models to enhance video reasoning while reducing the computational complexity of processing hundreds or thousands of frames. We introduce VIP, an inference-time dataset that can be used to evaluate VideoCOT, containing 1) a variety of real-life videos with keyframes and corresponding unstructured and structured scene descriptions, and 2) two new video reasoning tasks: video infilling and scene prediction. We benchmark various vision-language models on VIP, demonstrating the potential to use vision-la
    
[^10]: Sabiá: 葡萄牙的大型语言模型

    Sabi\'a: Portuguese Large Language Models. (arXiv:2304.07880v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2304.07880](http://arxiv.org/abs/2304.07880)

    针对葡萄牙语进行单语言预训练，可以显著提高大规模合成语言模型的质量，并能够在一系列葡萄牙语数据集上优于以英语为中心和多语言的对手，最好的模型的表现与GPT-3.5-turbo持平。

    

    随着语言模型能力的不断提高，”一刀切“的模型仍然是主流。尤其是考虑到全球使用的语言数量非常庞大，并且其中很多语言都是低资源语言，主要的做法是对多种语言进行预训练。本文对这种做法提出了质疑，证明了针对目标语言进行单语言预训练可以显著提高大规模合成语言模型的质量。我们在本文中进一步介绍了用3%或更少的原始预训练预算在葡萄牙语文本上进一步预训练GPT-J和LLaMA模型。我们在Poeta（一套由14个葡萄牙语数据集组成的套件）上进行了少样本评估，结果显示我们的模型在表现上远优于以英语为中心的和多语言的对手。我们的最佳模型Sabiá-65B的表现与GPT-3.5-turbo持平。我们在目标语言中已经设想了数据集，以及经过翻译的数据集上都进行了评估。

    As the capabilities of language models continue to advance, it is conceivable that "one-size-fits-all" model will remain as the main paradigm. For instance, given the vast number of languages worldwide, many of which are low-resource, the prevalent practice is to pretrain a single model on multiple languages. In this paper, we add to the growing body of evidence that challenges this practice, demonstrating that monolingual pretraining on the target language significantly improves models already extensively trained on diverse corpora. More specifically, we further pretrain GPT-J and LLaMA models on Portuguese texts using 3% or less of their original pretraining budget. Few-shot evaluations on Poeta, a suite of 14 Portuguese datasets, reveal that our models outperform English-centric and multilingual counterparts by a significant margin. Our best model, Sabi\'a-65B, performs on par with GPT-3.5-turbo. By evaluating on datasets originally conceived in the target language as well as transl
    
[^11]: MLRegTest：机器学习正则语言的基准测试

    MLRegTest: A Benchmark for the Machine Learning of Regular Languages. (arXiv:2304.07687v1 [cs.LG])

    [http://arxiv.org/abs/2304.07687](http://arxiv.org/abs/2304.07687)

    本文提出了一个名为MLRegTest的新基准测试，其包含了来自1,800个正则语言的数据集。该测试根据逻辑复杂度和逻辑文字种类组织语言，并可以帮助我们了解机器学习系统在学习不同种类的长距离依赖方面的性能。

    

    评估机器学习系统对已知分类器的学习能力允许细致地检查它们可以学习哪些模式，并在将它们应用于未知分类器的学习时建立信心。本文提出了一个名为MLRegTest的新的序列分类机器学习系统基准测试，其中包含来自1,800个正则语言的训练、开发和测试集。不同类型的形式语言代表着不同种类的长距离依赖，并正确地识别序列中的长距离依赖是机器学习系统成功泛化的已知挑战。MLRegTest根据它们的逻辑复杂度（单调二阶，一阶，命题或单项式表达式）和逻辑文字的种类（字符串，定级字符串，子序列或两者的组合）组织其语言。逻辑复杂度和文字的选择提供了一种系统方法来理解不同种类的长距离依赖和机器学习系统在处理它们时的性能。

    Evaluating machine learning (ML) systems on their ability to learn known classifiers allows fine-grained examination of the patterns they can learn, which builds confidence when they are applied to the learning of unknown classifiers. This article presents a new benchmark for ML systems on sequence classification called MLRegTest, which contains training, development, and test sets from 1,800 regular languages.  Different kinds of formal languages represent different kinds of long-distance dependencies, and correctly identifying long-distance dependencies in sequences is a known challenge for ML systems to generalize successfully. MLRegTest organizes its languages according to their logical complexity (monadic second order, first order, propositional, or monomial expressions) and the kind of logical literals (string, tier-string, subsequence, or combinations thereof). The logical complexity and choice of literal provides a systematic way to understand different kinds of long-distance d
    
[^12]: 人工和机器关于什么是冒犯存在较大分歧的共情冒犯和噪声审计：统一主观冒犯的人类和机器差异

    Vicarious Offense and Noise Audit of Offensive Speech Classifiers: Unifying Human and Machine Disagreement on What is Offensive. (arXiv:2301.12534v3 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2301.12534](http://arxiv.org/abs/2301.12534)

    本文通过人工和机器审核员的共情冒犯和噪声审计研究了冒犯性言论检测中的差异性。结果表明，审核员之间存在广泛的分歧，并且人工审核员和大型语言模型分类器无法预测其他审核员的回应。这对于内容审核具有重要意义。

    

    冒犯性言论检测是内容审核的一个关键组成部分。然而，什么是冒犯性的可以是高度主观的。本文研究了当涉及到现实世界社交网站政治言论时，人工和机器审核员对于什么是冒犯性的存在分歧。我们发现（1）审核员之间（包括人工和机器）存在广泛分歧；和（2）人工审核员和大型语言模型分类器无法预测其他审核员基于他们的政治倾向如何回应。对于（1），我们进行了一个前所未有规模的噪声审计，结合了机器和人工回答。对于（2），我们介绍了一个首创的共情冒犯的数据集。我们的噪声审计揭示了不同机器审核员之间的审核结果差异很大。我们与人工审核员进行的实验表明，政治倾向结合敏感问题会影响到一对一的冒犯，以及共情冒犯。数据集可通过https://github.com/Homan-Lab/voic获得。

    Offensive speech detection is a key component of content moderation. However, what is offensive can be highly subjective. This paper investigates how machine and human moderators disagree on what is offensive when it comes to real-world social web political discourse. We show that (1) there is extensive disagreement among the moderators (humans and machines); and (2) human and large-language-model classifiers are unable to predict how other human raters will respond, based on their political leanings. For (1), we conduct a noise audit at an unprecedented scale that combines both machine and human responses. For (2), we introduce a first-of-its-kind dataset of vicarious offense. Our noise audit reveals that moderation outcomes vary wildly across different machine moderators. Our experiments with human moderators suggest that political leanings combined with sensitive issues affect both first-person and vicarious offense. The dataset is available through https://github.com/Homan-Lab/voic
    
[^13]: 从人类的纠错中学习

    Learning From How Humans Correct. (arXiv:2102.00225v14 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2102.00225](http://arxiv.org/abs/2102.00225)

    本研究提出了一种从人类矫正中学习的方法。通过标注数据中的噪声数据，收集纠错信息，并将其注入至深度学习模型中，成功将文本分类准确度提升了1.7个百分点。

    

    在工业自然语言处理应用中，我们手动标注的数据中存在一定数量的噪声数据。我们提出了一种简单的方法来找到噪声数据并手动重新标注它们，同时收集纠错信息。然后，我们提出了一种将人类纠错信息融入深度学习模型的新方法。人类知道如何纠正噪声数据，因此纠错信息可以注入到深度学习模型中。我们在自己的文本分类数据集上进行了实验，该数据集是手动标注的，因为我们重新标注了我们数据集中的噪声数据，以适用于我们的工业应用。实验结果显示，我们的方法将分类准确度从91.7%提升到92.5%。91.7%的准确度是在修正后的数据集上训练的，它将基线准确度从83.3%提升到91.7%。

    In industry NLP application, our manually labeled data has a certain number of noisy data. We present a simple method to find the noisy data and re-label them manually, meanwhile we collect the correction information. Then we present novel method to incorporate the human correction information into deep learning model. Human know how to correct noisy data. So the correction information can be inject into deep learning model. We do the experiment on our own text classification dataset, which is manually labeled, because we re-label the noisy data in our dataset for our industry application. The experiment result shows that our method improve the classification accuracy from 91.7% to 92.5%. The 91.7% accuracy is trained on the corrected dataset, which improve the baseline from 83.3% to 91.7%.
    

