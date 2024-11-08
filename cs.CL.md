# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Long-form factuality in large language models](https://arxiv.org/abs/2403.18802) | 该论文提出了一种通过使用大型语言模型将长篇回应分解为单个事实，并通过发送搜索查询到Google搜索，评估事实准确性的方法，并扩展了F1分数作为长篇事实性的聚合度量。 |
| [^2] | [Gradient Cuff: Detecting Jailbreak Attacks on Large Language Models by Exploring Refusal Loss Landscapes](https://arxiv.org/abs/2403.00867) | 本文提出了一种名为Gradient Cuff的方法，通过探索拒绝损失地形图来检测对大语言模型的越狱攻击，成功设计了一种有效的两步检测策略。 |
| [^3] | [Personalized Large Language Models](https://arxiv.org/abs/2402.09269) | 本文研究了个性化大型语言模型的方法，通过比较微调和零样本推理的方法，在主观任务中发现个性化微调能提高模型的推理能力，在情感识别和仇恨言论检测方面也获得了一致的性能提升。 |
| [^4] | [Textless Low-Resource Speech-to-Speech Translation With Unit Language Models](https://arxiv.org/abs/2305.15405) | 提出了一种新的框架，用于训练只需要几十小时平行语音数据的无文本低资源语音到语音翻译系统，并通过单元到单元的序列到序列翻译任务和无监督反向翻译目标来提高模型性能 |
| [^5] | [SciEval: A Multi-Level Large Language Model Evaluation Benchmark for Scientific Research.](http://arxiv.org/abs/2308.13149) | SciEval是一个综合且多学科的评估基准，用于评估大型语言模型在科学研究中的能力。它基于布鲁姆的分类法，包括客观和主观问题，并设计了一个防止数据泄漏的“动态”子集。实验结果表明，尽管GPT-4在某些方面取得了较高的得分，但仍存在挑战。 |

# 详细

[^1]: 大型语言模型中的长篇事实性

    Long-form factuality in large language models

    [https://arxiv.org/abs/2403.18802](https://arxiv.org/abs/2403.18802)

    该论文提出了一种通过使用大型语言模型将长篇回应分解为单个事实，并通过发送搜索查询到Google搜索，评估事实准确性的方法，并扩展了F1分数作为长篇事实性的聚合度量。

    

    大型语言模型（LLMs）在回答开放性主题的事实性提示时，经常生成包含事实错误的内容。为了在开放领域中对模型的长篇事实性进行基准测试，我们首先使用GPT-4生成了一个名为LongFact的提示集，其中包含数千个囊括38个主题的问题。然后，我们提出LLM代理可以通过一种名为Search-Augmented Factuality Evaluator（SAFE）的方法作为长篇事实性的自动评估器。SAFE利用LLM将长篇回应分解为一组单独的事实，并通过发送搜索查询到Google搜索以及确定一个事实是否得到搜索结果支持的多步推理过程来评估每个事实的准确性。此外，我们还提议将F1分数扩展为长篇事实性的聚合度量。为此，我们平衡了回应中支持事实的百分比（精度）与

    arXiv:2403.18802v1 Announce Type: cross  Abstract: Large language models (LLMs) often generate content that contains factual errors when responding to fact-seeking prompts on open-ended topics. To benchmark a model's long-form factuality in open domains, we first use GPT-4 to generate LongFact, a prompt set comprising thousands of questions spanning 38 topics. We then propose that LLM agents can be used as automated evaluators for long-form factuality through a method which we call Search-Augmented Factuality Evaluator (SAFE). SAFE utilizes an LLM to break down a long-form response into a set of individual facts and to evaluate the accuracy of each fact using a multi-step reasoning process comprising sending search queries to Google Search and determining whether a fact is supported by the search results. Furthermore, we propose extending F1 score as an aggregated metric for long-form factuality. To do so, we balance the percentage of supported facts in a response (precision) with the 
    
[^2]: 梯度被罚：通过探索拒绝损失地形图来检测针对大语言模型的越狱攻击

    Gradient Cuff: Detecting Jailbreak Attacks on Large Language Models by Exploring Refusal Loss Landscapes

    [https://arxiv.org/abs/2403.00867](https://arxiv.org/abs/2403.00867)

    本文提出了一种名为Gradient Cuff的方法，通过探索拒绝损失地形图来检测对大语言模型的越狱攻击，成功设计了一种有效的两步检测策略。

    

    大型语言模型（LLMs）正成为一种突出的生成式AI工具，用户输入查询，LLM生成答案。为了减少伤害和滥用，人们通过使用先进的训练技术如来自人类反馈的强化学习（RLHF）来将这些LLMs与人类价值观保持一致。然而，最近的研究突显了LLMs对于试图颠覆嵌入的安全防护措施的对抗性越狱尝试的脆弱性。为了解决这一挑战，本文定义并调查了LLMs的拒绝损失，然后提出了一种名为Gradient Cuff的方法来检测越狱尝试。Gradient Cuff利用拒绝损失地形图中观察到的独特特性，包括功能值及其光滑性，设计了一种有效的两步检测策略。

    arXiv:2403.00867v1 Announce Type: cross  Abstract: Large Language Models (LLMs) are becoming a prominent generative AI tool, where the user enters a query and the LLM generates an answer. To reduce harm and misuse, efforts have been made to align these LLMs to human values using advanced training techniques such as Reinforcement Learning from Human Feedback (RLHF). However, recent studies have highlighted the vulnerability of LLMs to adversarial jailbreak attempts aiming at subverting the embedded safety guardrails. To address this challenge, this paper defines and investigates the Refusal Loss of LLMs and then proposes a method called Gradient Cuff to detect jailbreak attempts. Gradient Cuff exploits the unique properties observed in the refusal loss landscape, including functional values and its smoothness, to design an effective two-step detection strategy. Experimental results on two aligned LLMs (LLaMA-2-7B-Chat and Vicuna-7B-V1.5) and six types of jailbreak attacks (GCG, AutoDAN,
    
[^3]: 个性化的大型语言模型

    Personalized Large Language Models

    [https://arxiv.org/abs/2402.09269](https://arxiv.org/abs/2402.09269)

    本文研究了个性化大型语言模型的方法，通过比较微调和零样本推理的方法，在主观任务中发现个性化微调能提高模型的推理能力，在情感识别和仇恨言论检测方面也获得了一致的性能提升。

    

    近年来，大型语言模型（LLM）在自然语言处理（NLP）任务中取得了显著的进展。然而，它们的通用性在需要个性化回应的场景（如推荐系统和聊天机器人）中存在一定的局限性。本文研究了个性化LLM的方法，比较了微调和零样本推理方法在主观任务中的效果。结果表明，与非个性化模型相比，个性化微调改善了模型的推理能力。在情感识别和仇恨言论检测的数据集上进行的实验表明，个性化方法在不同的LLM架构上获得了一致的性能提升。这些发现强调了在主观文本理解任务中提升LLM能力的个性化的重要性。

    arXiv:2402.09269v1 Announce Type: cross Abstract: Large language models (LLMs) have significantly advanced Natural Language Processing (NLP) tasks in recent years. However, their universal nature poses limitations in scenarios requiring personalized responses, such as recommendation systems and chatbots. This paper investigates methods to personalize LLMs, comparing fine-tuning and zero-shot reasoning approaches on subjective tasks. Results demonstrate that personalized fine-tuning improves model reasoning compared to non-personalized models. Experiments on datasets for emotion recognition and hate speech detection show consistent performance gains with personalized methods across different LLM architectures. These findings underscore the importance of personalization for enhancing LLM capabilities in subjective text perception tasks.
    
[^4]: 具有单元语言模型的无文本低资源语音到语音翻译

    Textless Low-Resource Speech-to-Speech Translation With Unit Language Models

    [https://arxiv.org/abs/2305.15405](https://arxiv.org/abs/2305.15405)

    提出了一种新的框架，用于训练只需要几十小时平行语音数据的无文本低资源语音到语音翻译系统，并通过单元到单元的序列到序列翻译任务和无监督反向翻译目标来提高模型性能

    

    现有的语音到语音翻译模型大致分为两类：使用数百小时平行语音数据训练的无文本模型，或者将文本作为中间步骤的无监督模型。这两种方法限制了为广泛语言构建语音到语音翻译模型的可能性，因为它们排除了主要口语的语言以及缺乏大规模平行语音数据的语言对。我们提出了一个新的框架，用于训练只需要几十小时平行语音数据的无文本低资源语音到语音翻译（S2ST）系统。我们将S2ST重新构建为一个单元到单元的序列到序列翻译任务，并首先在大规模单语言语音数据上进行预训练。然后，我们使用少量平行语音数据（$20-60$小时）对其进行微调。最后，我们通过无监督反向翻译目标改善模型性能。我们为英语到德语，德语

    arXiv:2305.15405v2 Announce Type: replace  Abstract: Existing speech-to-speech translation models fall into two camps: textless models trained with hundreds of hours of parallel speech data or unsupervised models that leverage text as an intermediate step. Both approaches limit building speech-to-speech translation models for a wide range of languages, as they exclude languages that are primarily spoken and language pairs that lack large-scale parallel speech data. We present a new framework for training textless low-resource speech-to-speech translation (S2ST) systems that only need dozens of hours of parallel speech data. We reformulate S2ST as a unit-to-unit seq2seq translation task, and start by pretraining a model on large-scale monolingual speech data. Then, we finetune it with a small amount of parallel speech data ($20-60$ hours). Lastly, we improve model performance through an unsupervised backtranslation objective. We train and evaluate our models for English-to-German, Germa
    
[^5]: SciEval: 用于科学研究的多级大型语言模型评估基准

    SciEval: A Multi-Level Large Language Model Evaluation Benchmark for Scientific Research. (arXiv:2308.13149v1 [cs.CL])

    [http://arxiv.org/abs/2308.13149](http://arxiv.org/abs/2308.13149)

    SciEval是一个综合且多学科的评估基准，用于评估大型语言模型在科学研究中的能力。它基于布鲁姆的分类法，包括客观和主观问题，并设计了一个防止数据泄漏的“动态”子集。实验结果表明，尽管GPT-4在某些方面取得了较高的得分，但仍存在挑战。

    

    最近，使用大型语言模型（LLMs）进行科学研究引起了越来越多的关注。已经提出了许多基准来评估LLMs在科学研究中的能力。然而，目前的基准主要基于预先收集的客观问题。这种设计存在数据泄漏问题，并且缺乏对主观问答能力的评估。在本文中，我们提出了SciEval，这是一个综合、多学科的评估基准，以解决这些问题。基于布鲁姆的分类法，SciEval涵盖了四个维度来系统评估科学研究能力。特别地，我们设计了一个基于科学原理的“动态”子集，以防止评估出现潜在的数据泄漏。SciEval包含了客观和主观问题。这些特点使SciEval成为评估LLMs科学研究能力的更有效的基准。对最先进的LLMs进行了全面实验，结果显示，尽管GPT-4取得了较高的得分，但在某些方面仍存在挑战。

    Recently, there has been growing interest in using Large Language Models (LLMs) for scientific research. Numerous benchmarks have been proposed to evaluate the ability of LLMs for scientific research. However, current benchmarks are mostly based on pre-collected objective questions. This design suffers from data leakage problem and lacks the evaluation of subjective Q/A ability. In this paper, we propose SciEval, a comprehensive and multi-disciplinary evaluation benchmark to address these issues. Based on Bloom's taxonomy, SciEval covers four dimensions to systematically evaluate scientific research ability. In particular, we design a "dynamic" subset based on scientific principles to prevent evaluation from potential data leakage. Both objective and subjective questions are included in SciEval. These characteristics make SciEval a more effective benchmark for scientific research ability evaluation of LLMs. Comprehensive experiments on most advanced LLMs show that, although GPT-4 achie
    

