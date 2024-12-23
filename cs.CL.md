# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [GPT-4 Understands Discourse at Least as Well as Humans Do](https://arxiv.org/abs/2403.17196) | GPT-4在标准化语篇理解测试中表现出与人类相当的能力，尤其在推断未明确陈述信息方面显示出显著实力 |
| [^2] | [Born With a Silver Spoon? Investigating Socioeconomic Bias in Large Language Models](https://arxiv.org/abs/2403.14633) | 本文调查了大型语言模型中是否存在社会经济偏见，引入了一个新的数据集SilverSpoon，并评估了这种偏见的程度以及随着模型大小的变化。 |
| [^3] | [Small Language Model Is a Good Guide for Large Language Model in Chinese Entity Relation Extraction](https://arxiv.org/abs/2402.14373) | 本文提出了SLCoLM，一个模型协作框架，通过使用“训练-指导-预测”策略结合预训练语言模型和大语言模型，成功缓解了长尾数据问题，促进了实体关系的抽取。 |
| [^4] | [Measuring Social Biases in Masked Language Models by Proxy of Prediction Quality](https://arxiv.org/abs/2402.13954) | 本文通过提出的代理函数在迭代屏蔽实验中评估了转换器模型所编码的社会偏见，并比较了其与其他评估方法的偏见估计，发现转换器模型中存在相对较高的宗教和残疾偏见，而性别偏见则相对较低。 |
| [^5] | [Data Quality Matters: Suicide Intention Detection on Social Media Posts Using a RoBERTa-CNN Model](https://arxiv.org/abs/2402.02262) | 本文介绍了一种使用RoBERTa-CNN模型来在社交媒体帖子中检测自杀意图的新方法。RoBERTa-CNN通过在RoBERTa模型中添加卷积神经网络（CNN）层，提高了对重要模式的捕捉能力，并在实验证明在自杀和抑郁检测数据集上表现出良好的准确性。 |
| [^6] | [A survey on recent advances in named entity recognition.](http://arxiv.org/abs/2401.10825) | 这篇综述调查了最近的命名实体识别研究进展，并提供了对不同算法性能的深度比较，还探讨了数据集特征对方法行为的影响。 |
| [^7] | [Large Language Models can Learn Rules.](http://arxiv.org/abs/2310.07064) | 大型语言模型(LLMs)在各种推理任务中展示了令人印象深刻的性能。为了提高提示方法的准确性和一致性，我们提出了Hypotheses-to-Theories (HtT)框架，用于学习LLMs推理的规则库，从而改进了现有的提示方法。 |

# 详细

[^1]: GPT-4至少能够像人类一样理解语篇

    GPT-4 Understands Discourse at Least as Well as Humans Do

    [https://arxiv.org/abs/2403.17196](https://arxiv.org/abs/2403.17196)

    GPT-4在标准化语篇理解测试中表现出与人类相当的能力，尤其在推断未明确陈述信息方面显示出显著实力

    

    我们测试了一种领先的AI系统GPT-4是否像人类一样理解语篇，使用了一项标准化的语篇理解测试。参与者会被呈现简短的故事，然后回答八个是/否问题，探究他们对故事的理解。这些问题的格式旨在评估直接性（陈述 vs. 暗示）和显著性（主要观点 vs. 细节）的独立影响。鉴于人类表现水平非常高，GPT-4的表现略好于人类，但并无统计学显著差异。GPT-4和人类都表现出强大的能力，能够推断故事中未明确陈述的信息，这是对理解力的重要测试。

    arXiv:2403.17196v1 Announce Type: new  Abstract: We test whether a leading AI system GPT-4 understands discourse as well as humans do, using a standardized test of discourse comprehension. Participants are presented with brief stories and then answer eight yes/no questions probing their comprehension of the story. The questions are formatted to assess the separate impacts of directness (stated vs. implied) and salience (main idea vs. details). GPT-4 performs slightly, but not statistically significantly, better than humans given the very high level of human performance. Both GPT-4 and humans exhibit a strong ability to make inferences about information that is not explicitly stated in a story, a critical test of understanding.
    
[^2]: 出身富贵？探讨大型语言模型中的社会经济偏见

    Born With a Silver Spoon? Investigating Socioeconomic Bias in Large Language Models

    [https://arxiv.org/abs/2403.14633](https://arxiv.org/abs/2403.14633)

    本文调查了大型语言模型中是否存在社会经济偏见，引入了一个新的数据集SilverSpoon，并评估了这种偏见的程度以及随着模型大小的变化。

    

    社会经济偏见在社会中加剧了不公平现象，根据个人经济和社会背景影响获取机会和资源的机会。这一普遍问题持续地延续了系统性的不平等，阻碍了作为一个社会追求包容性进步。在本文中，我们调查了大型语言模型中是否存在社会经济偏见。为此，我们引入了一个新的数据集（SilverSpoon），包含3000个样本，展示了牵涉到弱势群体由于他们的处境而实施道德模糊行为的假设情景，并问这种行为是否在道德上成立。此外，这个数据集具有双重标记方案，并由属于社会经济两端的人进行了注释。使用SilverSpoon，我们评估了大型语言模型中表现出的社会经济偏见程度以及该程度如何随模型大小变化。

    arXiv:2403.14633v1 Announce Type: cross  Abstract: Socioeconomic bias in society exacerbates disparities, influencing access to opportunities and resources based on individuals' economic and social backgrounds. This pervasive issue perpetuates systemic inequalities, hindering the pursuit of inclusive progress as a society. In this paper, we investigate the presence of socioeconomic bias, if any, in large language models. To this end, we introduce a novel dataset (SilverSpoon), consisting of 3000 samples that illustrate hypothetical scenarios that involve underprivileged people performing ethically ambiguous actions due to their circumstances, and ask whether the action is ethically justified. Further, this dataset has a dual-labeling scheme and has been annotated by people belonging to both ends of the socioeconomic spectrum. Using SilverSpoon, we evaluate the degree of socioeconomic bias expressed in large language models and the variation of this degree as a function of model size. W
    
[^3]: 小语言模型在中文实体关系抽取中是大语言模型的良好向导

    Small Language Model Is a Good Guide for Large Language Model in Chinese Entity Relation Extraction

    [https://arxiv.org/abs/2402.14373](https://arxiv.org/abs/2402.14373)

    本文提出了SLCoLM，一个模型协作框架，通过使用“训练-指导-预测”策略结合预训练语言模型和大语言模型，成功缓解了长尾数据问题，促进了实体关系的抽取。

    

    近年来，大语言模型（LLMs）在关系抽取（RE）任务中取得了成功，尤其是在少样本学习中。关系抽取领域中一个重要问题是长尾数据，然而目前很少有关注使用LLM方法解决这个问题。因此，在本文中，我们提出了SLCoLM，一个模型协作框架，以缓解数据长尾问题。在我们的框架中，我们使用“训练-指导-预测”策略来结合预训练语言模型（PLMs）和LLMs的优势，其中一个特定于任务的PLM框架充当导师，将任务知识转移到LLM，并指导LLM执行RE任务。我们对一个富含关系类型的RE数据集进行的实验表明，本文中的方法促进了长尾关系类型的RE。

    arXiv:2402.14373v1 Announce Type: new  Abstract: Recently, large language models (LLMs) have been successful in relational extraction (RE) tasks, especially in the few-shot learning. An important problem in the field of RE is long-tailed data, while not much attention is currently paid to this problem using LLM approaches. Therefore, in this paper, we propose SLCoLM, a model collaboration framework, to mitigate the data long-tail problem. In our framework, We use the ``\textit{Training-Guide-Predict}'' strategy to combine the strengths of pre-trained language models (PLMs) and LLMs, where a task-specific PLM framework acts as a tutor, transfers task knowledge to the LLM, and guides the LLM in performing RE tasks. Our experiments on a RE dataset rich in relation types show that the approach in this paper facilitates RE of long-tail relation types.
    
[^4]: 通过预测质量间接测量掩盖语言模型中的社会偏见

    Measuring Social Biases in Masked Language Models by Proxy of Prediction Quality

    [https://arxiv.org/abs/2402.13954](https://arxiv.org/abs/2402.13954)

    本文通过提出的代理函数在迭代屏蔽实验中评估了转换器模型所编码的社会偏见，并比较了其与其他评估方法的偏见估计，发现转换器模型中存在相对较高的宗教和残疾偏见，而性别偏见则相对较低。

    

    社会和政治科学家经常旨在从文本数据表示（嵌入）中发现和衡量不同的偏见。创新的基于转换器的语言模型生成具有上下文感知的令牌嵌入，并在各种自然语言任务中取得了最先进的性能，但已被证明在下游应用中编码了不需要的偏见。本文通过提出的代理函数在迭代屏蔽实验中评估由训练有遮蔽语言建模目标的转换器所编码的社会偏见，以测量转换器模型预测质量，并评估MLM对不利群体和有利群体的偏好。我们比较使用两个基准数据集的偏见估计与其他评估方法产生的偏见，发现考虑的MLMs中存在相对较高的宗教和残疾偏见，而相对于另一个数据集，一个数据集中存在较低的性别偏见。

    arXiv:2402.13954v1 Announce Type: new  Abstract: Social and political scientists often aim to discover and measure distinct biases from text data representations (embeddings). Innovative transformer-based language models produce contextually-aware token embeddings and have achieved state-of-the-art performance for a variety of natural language tasks, but have been shown to encode unwanted biases for downstream applications. In this paper, we evaluate the social biases encoded by transformers trained with the masked language modeling objective using proposed proxy functions within an iterative masking experiment to measure the quality of transformer models' predictions, and assess the preference of MLMs towards disadvantaged and advantaged groups. We compare bias estimations with those produced by other evaluation methods using two benchmark datasets, finding relatively high religious and disability biases across considered MLMs and low gender bias in one dataset relative to the other. 
    
[^5]: 数据质量很重要：使用RoBERTa-CNN模型在社交媒体帖子中检测自杀意图

    Data Quality Matters: Suicide Intention Detection on Social Media Posts Using a RoBERTa-CNN Model

    [https://arxiv.org/abs/2402.02262](https://arxiv.org/abs/2402.02262)

    本文介绍了一种使用RoBERTa-CNN模型来在社交媒体帖子中检测自杀意图的新方法。RoBERTa-CNN通过在RoBERTa模型中添加卷积神经网络（CNN）层，提高了对重要模式的捕捉能力，并在实验证明在自杀和抑郁检测数据集上表现出良好的准确性。

    

    自杀仍然是全球健康领域的一个关注焦点，急需创新方法进行早期检测和干预。本文着重于识别SuicideWatch Reddit帖子中的自杀意图，并提出了一种使用尖端的RoBERTa-CNN模型进行自杀检测的新方法，RoBERTa-CNN是RoBERTa（鲁棒性优化BERT方法）的一种变体。RoBERTa被用于各种自然语言处理（NLP）任务，包括文本分类和情感分析。RoBERTa的有效性在于它能够捕捉文本信息并形成文本之间的语义关系。通过在原始模型中添加卷积神经网络（CNN）层，RoBERTa增强了从庞大数据集中捕捉重要模式的能力。我们在自杀和抑郁检测数据集上评估了RoBERTa-CNN，并获得了可靠的结果，例如，RoBERTa-CNN在平均准确率上获得了98％，标准差为...

    Suicide remains a global health concern for the field of health, which urgently needs innovative approaches for early detection and intervention. In this paper, we focus on identifying suicidal intentions in SuicideWatch Reddit posts and present a novel approach to suicide detection using the cutting-edge RoBERTa-CNN model, a variant of RoBERTa (Robustly optimized BERT approach). RoBERTa is used for various Natural Language Processing (NLP) tasks, including text classification and sentiment analysis. The effectiveness of the RoBERTa lies in its ability to capture textual information and form semantic relationships within texts. By adding the Convolution Neural Network (CNN) layer to the original model, the RoBERTa enhances its ability to capture important patterns from heavy datasets. To evaluate the RoBERTa-CNN, we experimented on the Suicide and Depression Detection dataset and obtained solid results. For example, RoBERTa-CNN achieves 98% mean accuracy with the standard deviation (ST
    
[^6]: 最新进展的命名实体识别综述

    A survey on recent advances in named entity recognition. (arXiv:2401.10825v1 [cs.CL])

    [http://arxiv.org/abs/2401.10825](http://arxiv.org/abs/2401.10825)

    这篇综述调查了最近的命名实体识别研究进展，并提供了对不同算法性能的深度比较，还探讨了数据集特征对方法行为的影响。

    

    命名实体识别旨在从文本中提取出命名真实世界对象的子字符串，并确定其类型（例如，是否指人物或组织）。在本综述中，我们首先概述了最近流行的方法，同时还关注了基于图和变换器的方法，包括很少在其他综述中涉及的大型语言模型（LLMs）。其次，我们重点介绍了针对稀缺注释数据集设计的方法。第三，我们评估了主要命名实体识别实现在各种具有不同特征（领域、规模和类别数）的数据集上的性能。因此，我们提供了一种从未同时考虑的算法的深度比较。我们的实验揭示了数据集特征如何影响我们比较的方法的行为。

    Named Entity Recognition seeks to extract substrings within a text that name real-world objects and to determine their type (for example, whether they refer to persons or organizations). In this survey, we first present an overview of recent popular approaches, but we also look at graph- and transformer- based methods including Large Language Models (LLMs) that have not had much coverage in other surveys. Second, we focus on methods designed for datasets with scarce annotations. Third, we evaluate the performance of the main NER implementations on a variety of datasets with differing characteristics (as regards their domain, their size, and their number of classes). We thus provide a deep comparison of algorithms that are never considered together. Our experiments shed some light on how the characteristics of datasets affect the behavior of the methods that we compare.
    
[^7]: 大型语言模型可以学习规则

    Large Language Models can Learn Rules. (arXiv:2310.07064v1 [cs.AI])

    [http://arxiv.org/abs/2310.07064](http://arxiv.org/abs/2310.07064)

    大型语言模型(LLMs)在各种推理任务中展示了令人印象深刻的性能。为了提高提示方法的准确性和一致性，我们提出了Hypotheses-to-Theories (HtT)框架，用于学习LLMs推理的规则库，从而改进了现有的提示方法。

    

    当给出一些示例和中间步骤时，大型语言模型(LLMs)在各种推理任务中展示了令人印象深刻的性能。然而，依赖LLM中的隐式知识的提示方法在隐式知识错误或与任务不一致时往往会产生错误的答案。为解决这个问题，我们提出了"假设到理论" (HtT) 框架，用于学习LLMs推理的规则库。HtT包括两个阶段，归纳阶段和演绎阶段。在归纳阶段，首先要求LLM根据一组训练示例生成和验证规则。出现并导致正确答案的规则将被收集形成一个规则库。在演绎阶段，然后要求LLM使用学习的规则库进行推理以回答测试问题。在数值推理和关系推理问题上的实验证明，HtT改进了现有的提示方法，使其性能提升。

    When prompted with a few examples and intermediate steps, large language models (LLMs) have demonstrated impressive performance in various reasoning tasks. However, prompting methods that rely on implicit knowledge in an LLM often hallucinate incorrect answers when the implicit knowledge is wrong or inconsistent with the task. To tackle this problem, we present Hypotheses-to-Theories (HtT), a framework that learns a rule library for reasoning with LLMs. HtT contains two stages, an induction stage and a deduction stage. In the induction stage, an LLM is first asked to generate and verify rules over a set of training examples. Rules that appear and lead to correct answers sufficiently often are collected to form a rule library. In the deduction stage, the LLM is then prompted to employ the learned rule library to perform reasoning to answer test questions. Experiments on both numerical reasoning and relational reasoning problems show that HtT improves existing prompting methods, with an 
    

