# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Simplicity Bias of Transformers to Learn Low Sensitivity Functions](https://arxiv.org/abs/2403.06925) | Transformers在不同数据模态上具有低敏感性，这种简单性偏差有助于解释其在视觉和语言任务中的优越性能。 |
| [^2] | [A Cognitive Evaluation Benchmark of Image Reasoning and Description for Large Vision Language Models](https://arxiv.org/abs/2402.18409) | 提出了一个新颖的评估基准，用于评估大型视觉语言模型的认知能力，发现LVLMs与人类之间存在较大的认知能力差距。 |
| [^3] | [The Developmental Landscape of In-Context Learning](https://arxiv.org/abs/2402.02364) | 在transformers模型中，我们展示了在上下文学习中的离散发展阶段，并引入了两种方法来检测这些阶段的关键里程碑。我们使用行为和结构度量验证了这些方法的有效性。 |
| [^4] | [The Calibration Gap between Model and Human Confidence in Large Language Models.](http://arxiv.org/abs/2401.13835) | 该论文研究了大型语言模型在传达置信度方面模型和人类之间存在的差距，并发现默认解释会导致用户过高估计模型置信度和准确性。 |
| [^5] | [Improving Factual Consistency of Text Summarization by Adversarially Decoupling Comprehension and Embellishment Abilities of LLMs.](http://arxiv.org/abs/2310.19347) | 本文提出了一个名为DECENT的方法，通过对抗解耦LLMs的理解和修饰能力，提高文本摘要的事实一致性。同时，采用了一种探测技术来弥补训练过程中对真与假的敏感性不足的问题。 |
| [^6] | [Exploring Large Language Models for Knowledge Graph Completion.](http://arxiv.org/abs/2308.13916) | 本文研究了利用大型语言模型（LLM）进行知识图谱补全的方法，并引入了一种创新的框架（知识图谱LLM），以提高三元组分类和关系预测的性能。 |
| [^7] | [Curating corpora with classifiers: A case study of clean energy sentiment online.](http://arxiv.org/abs/2305.03092) | 本文介绍了利用分类器来快速选择最佳的相关文档语料库进行分析的方法，探索了过滤掉不相关的推文的方法，以进行在线清洁能源情感分析。 |
| [^8] | [On the Creativity of Large Language Models.](http://arxiv.org/abs/2304.00008) | 这篇论文探讨了大型语言模型的创造性问题，分析了与之相关的机器创造性的难点和易点，并重点分析了这些技术在创意产业中的社会影响。 |

# 详细

[^1]: Transformers学习低敏感性函数的简单性偏差

    Simplicity Bias of Transformers to Learn Low Sensitivity Functions

    [https://arxiv.org/abs/2403.06925](https://arxiv.org/abs/2403.06925)

    Transformers在不同数据模态上具有低敏感性，这种简单性偏差有助于解释其在视觉和语言任务中的优越性能。

    

    Transformers在许多任务中取得了最先进的准确性和鲁棒性，但对它们具有的归纳偏差以及这些偏差如何与其他神经网络架构不同的理解仍然难以捉摸。本文中，我们将模型对输入中的随机更改的敏感性概念化为一种简单性偏差的概念，这为解释transformers在不同数据模态上的简单性和谱偏差提供了统一的度量标准。我们展示了transformers在视觉和语言任务中比其他替代架构（如LSTMs、MLPs和CNNs）具有更低的敏感性。我们还展示了低敏感性偏差与改进性能的相关性。

    arXiv:2403.06925v1 Announce Type: cross  Abstract: Transformers achieve state-of-the-art accuracy and robustness across many tasks, but an understanding of the inductive biases that they have and how those biases are different from other neural network architectures remains elusive. Various neural network architectures such as fully connected networks have been found to have a simplicity bias towards simple functions of the data; one version of this simplicity bias is a spectral bias to learn simple functions in the Fourier space. In this work, we identify the notion of sensitivity of the model to random changes in the input as a notion of simplicity bias which provides a unified metric to explain the simplicity and spectral bias of transformers across different data modalities. We show that transformers have lower sensitivity than alternative architectures, such as LSTMs, MLPs and CNNs, across both vision and language tasks. We also show that low-sensitivity bias correlates with impro
    
[^2]: 一个针对大型视觉语言模型图像推理和描述的认知评估基准

    A Cognitive Evaluation Benchmark of Image Reasoning and Description for Large Vision Language Models

    [https://arxiv.org/abs/2402.18409](https://arxiv.org/abs/2402.18409)

    提出了一个新颖的评估基准，用于评估大型视觉语言模型的认知能力，发现LVLMs与人类之间存在较大的认知能力差距。

    

    尽管大型视觉语言模型(LVLMs)近年来取得了成功，但它们很少受到全面的认知能力测试。受到人类认知测试中广泛使用的“偷饼干”任务的启发，我们提出了一个新颖的评估基准，利用具有丰富语义的图像评估LVLMs的高级认知能力。它定义了八种推理能力，并包括图像描述任务和视觉问答任务。我们对知名LVLMs进行的评估表明，在LVLMs和人类之间仍存在较大的认知能力差距。

    arXiv:2402.18409v1 Announce Type: new  Abstract: Large Vision Language Models (LVLMs), despite their recent success, are hardly comprehensively tested for their cognitive abilities. Inspired by the prevalent use of the "Cookie Theft" task in human cognition test, we propose a novel evaluation benchmark to evaluate high-level cognitive ability of LVLMs using images with rich semantics. It defines eight reasoning capabilities and consists of an image description task and a visual question answering task. Our evaluation on well-known LVLMs shows that there is still a large gap in cognitive ability between LVLMs and humans.
    
[^3]: 在上下文中学习的发展景观

    The Developmental Landscape of In-Context Learning

    [https://arxiv.org/abs/2402.02364](https://arxiv.org/abs/2402.02364)

    在transformers模型中，我们展示了在上下文学习中的离散发展阶段，并引入了两种方法来检测这些阶段的关键里程碑。我们使用行为和结构度量验证了这些方法的有效性。

    

    我们展示了在transformers中，当它们通过语言建模或线性回归任务进行训练时，上下文学习是如何以离散的发展阶段出现的。我们引入了两种方法来检测分隔这些阶段的关键里程碑，通过探测参数空间和函数空间中种群损失的几何特征。我们使用一系列行为和结构度量研究这些新方法揭示的阶段，以建立它们的有效性。

    We show that in-context learning emerges in transformers in discrete developmental stages, when they are trained on either language modeling or linear regression tasks. We introduce two methods for detecting the milestones that separate these stages, by probing the geometry of the population loss in both parameter space and function space. We study the stages revealed by these new methods using a range of behavioral and structural metrics to establish their validity.
    
[^4]: 语言模型中模型和人类置信度之间的校准差距

    The Calibration Gap between Model and Human Confidence in Large Language Models. (arXiv:2401.13835v1 [cs.LG])

    [http://arxiv.org/abs/2401.13835](http://arxiv.org/abs/2401.13835)

    该论文研究了大型语言模型在传达置信度方面模型和人类之间存在的差距，并发现默认解释会导致用户过高估计模型置信度和准确性。

    

    为了使大型语言模型（LLM）能够获得人类的信任，它们需要在某种意义上实现良好的校准，即能够准确评估和传达它们的预测正确的可能性。最近的研究关注了LLM内部置信度评估的质量，但问题仍然是LLM能够如何将这种内部模型置信度传达给人类用户。本文探讨了人类对LLM响应的外部置信度与模型内部置信度之间的差距。通过涉及多项选择题的实验，我们系统地检查了人类用户识别LLM输出可信度的能力。我们的研究重点分为两个方面：（1）评估用户对真实LLM置信度的感知和（2）调查个性化解释对该感知的影响。研究结果显示，LLM的默认解释往往会导致用户过高估计模型的置信度和准确性。通过修改解释的方式可以减小这种误差。

    For large language models (LLMs) to be trusted by humans they need to be well-calibrated in the sense that they can accurately assess and communicate how likely it is that their predictions are correct. Recent work has focused on the quality of internal LLM confidence assessments, but the question remains of how well LLMs can communicate this internal model confidence to human users. This paper explores the disparity between external human confidence in an LLM's responses and the internal confidence of the model. Through experiments involving multiple-choice questions, we systematically examine human users' ability to discern the reliability of LLM outputs. Our study focuses on two key areas: (1) assessing users' perception of true LLM confidence and (2) investigating the impact of tailored explanations on this perception. The research highlights that default explanations from LLMs often lead to user overestimation of both the model's confidence and its' accuracy. By modifying the expl
    
[^5]: 通过对LLMs的理解和修饰能力进行对抗解耦，提高文本摘要的事实一致性改进

    Improving Factual Consistency of Text Summarization by Adversarially Decoupling Comprehension and Embellishment Abilities of LLMs. (arXiv:2310.19347v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2310.19347](http://arxiv.org/abs/2310.19347)

    本文提出了一个名为DECENT的方法，通过对抗解耦LLMs的理解和修饰能力，提高文本摘要的事实一致性。同时，采用了一种探测技术来弥补训练过程中对真与假的敏感性不足的问题。

    

    尽管大型语言模型（LLMs）在文本摘要方面取得了近期的进展，但它们经常会生成与原始文章事实不一致的摘要，被称为文本生成中的“幻觉”。与之前的小型模型（如BART，T5）不同，当前的LLMs在制造愚蠢错误方面较少，但制造了更复杂的错误，例如加入因果关系、添加错误细节和过度泛化等。这些幻觉很难通过传统方法检测出来，这给提高文本摘要的事实一致性带来了很大挑战。在本文中，我们提出了一种对抗解耦方法来分离LLMs的理解和修饰能力（DECENT）。此外，我们采用一种基于探测的参数高效技术，以弥补LLMs在训练过程中对真与假的敏感性不足的问题。通过这种方式，LLMs对于修饰和理解的概念更加清晰，从而能够更准确地执行指令。

    Despite the recent progress in text summarization made by large language models (LLMs), they often generate summaries that are factually inconsistent with original articles, known as "hallucinations" in text generation. Unlike previous small models (e.g., BART, T5), current LLMs make fewer silly mistakes but more sophisticated ones, such as imposing cause and effect, adding false details, and overgeneralizing, etc. These hallucinations are challenging to detect through traditional methods, which poses great challenges for improving the factual consistency of text summarization. In this paper, we propose an adversarially DEcoupling method to disentangle the Comprehension and EmbellishmeNT abilities of LLMs (DECENT). Furthermore, we adopt a probing-based parameter-efficient technique to cover the shortage of sensitivity for true and false in the training process of LLMs. In this way, LLMs are less confused about embellishing and understanding, thus can execute the instructions more accur
    
[^6]: 探索大型语言模型用于知识图谱补全

    Exploring Large Language Models for Knowledge Graph Completion. (arXiv:2308.13916v1 [cs.CL])

    [http://arxiv.org/abs/2308.13916](http://arxiv.org/abs/2308.13916)

    本文研究了利用大型语言模型（LLM）进行知识图谱补全的方法，并引入了一种创新的框架（知识图谱LLM），以提高三元组分类和关系预测的性能。

    

    知识图谱在众多人工智能任务中发挥着重要作用，但经常面临不完整性的问题。在本研究中，我们探索了利用大型语言模型（LLM）进行知识图谱补全的方法。我们将知识图谱中的三元组视为文本序列，并引入了一种创新的框架，称为知识图谱LLM（KG-LLM），来对这些三元组进行建模。我们的技术利用三元组的实体和关系描述作为提示，并利用响应进行预测。对各种基准知识图谱的实验表明，我们的方法在三元组分类和关系预测等任务中达到了最先进的性能。我们还发现，微调相对较小的模型（例如LLaMA-7B，ChatGLM-6B）优于最新的ChatGPT和GPT-4。

    Knowledge graphs play a vital role in numerous artificial intelligence tasks, yet they frequently face the issue of incompleteness. In this study, we explore utilizing Large Language Models (LLM) for knowledge graph completion. We consider triples in knowledge graphs as text sequences and introduce an innovative framework called Knowledge Graph LLM (KG-LLM) to model these triples. Our technique employs entity and relation descriptions of a triple as prompts and utilizes the response for predictions. Experiments on various benchmark knowledge graphs demonstrate that our method attains state-of-the-art performance in tasks such as triple classification and relation prediction. We also find that fine-tuning relatively smaller models (e.g., LLaMA-7B, ChatGLM-6B) outperforms recent ChatGPT and GPT-4.
    
[^7]: 利用分类器来筛选语料库：以在线清洁能源情感分析为例

    Curating corpora with classifiers: A case study of clean energy sentiment online. (arXiv:2305.03092v1 [cs.CL])

    [http://arxiv.org/abs/2305.03092](http://arxiv.org/abs/2305.03092)

    本文介绍了利用分类器来快速选择最佳的相关文档语料库进行分析的方法，探索了过滤掉不相关的推文的方法，以进行在线清洁能源情感分析。

    

    精心策划的、大规模的社交媒体帖子语料库是补充传统调查的替代数据来源，可以提供广泛的公众意见。虽然调查在收集代表性样本和实现高准确率方面很有效，但运行成本很高，而且会滞后于公众意见数天或数周。这两个缺点可以通过实时、高容量的数据流和快速的分析管道克服。在组织这样的数据管道方面的一个核心挑战是设计一种有效的方法，快速选择最佳的相关文档语料库进行分析。仅仅通过关键词查询往往会包括不相关的文档，而这些文档很难用词袋自然语言处理方法消歧。在这里，我们使用预先训练的基于转换器的模型，通过在手动标注的推文上对其进行微调，探索了语料库策划的方法，以过滤掉不相关的推文。我们能够实现高达0.8以上的F1得分。

    Well curated, large-scale corpora of social media posts containing broad public opinion offer an alternative data source to complement traditional surveys. While surveys are effective at collecting representative samples and are capable of achieving high accuracy, they can be both expensive to run and lag public opinion by days or weeks. Both of these drawbacks could be overcome with a real-time, high volume data stream and fast analysis pipeline. A central challenge in orchestrating such a data pipeline is devising an effective method for rapidly selecting the best corpus of relevant documents for analysis. Querying with keywords alone often includes irrelevant documents that are not easily disambiguated with bag-of-words natural language processing methods. Here, we explore methods of corpus curation to filter irrelevant tweets using pre-trained transformer-based models, fine-tuned for our binary classification task on hand-labeled tweets. We are able to achieve F1 scores of up to 0.
    
[^8]: 关于大型语言模型的创造性研究

    On the Creativity of Large Language Models. (arXiv:2304.00008v1 [cs.AI])

    [http://arxiv.org/abs/2304.00008](http://arxiv.org/abs/2304.00008)

    这篇论文探讨了大型语言模型的创造性问题，分析了与之相关的机器创造性的难点和易点，并重点分析了这些技术在创意产业中的社会影响。

    

    大型语言模型(LLMs)正在颠覆人工智能的多个领域。其中最显著的应用之一是创作，例如诗歌或故事：生成的输出通常具有惊人的质量。但是，一个自然的问题是：LLMs真的可以被认为是创造性的吗？在本文中，我们首先通过创造性理论的角度分析了LLMs的发展，探讨了关键的未解决问题和挑战。然后，我们在与LLMs相关的机器创造性方面确定了一组“易”和“难”问题，并对其进行了讨论。最后，我们分析了这些技术在创意产业中的社会影响。

    Large Language Models (LLMs) are revolutionizing several areas of Artificial Intelligence. One of the most remarkable applications is creative writing, e.g., poetry or storytelling: the generated outputs are often of astonishing quality. However, a natural question arise: can LLMs really be considered creative? In this article we firstly analyze the development of LLMs under the lens of creativity theories, investigating the key open questions and challenges. Then, we identify a set of "easy" and "hard" problems in machine creativity, discussing them in relation to LLMs. Finally, we analyze the societal impact of these technologies with a particular focus on the creative industries.
    

