# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Wait, It's All Token Noise? Always Has Been: Interpreting LLM Behavior Using Shapley Value](https://arxiv.org/abs/2404.01332) | 使用Shapley值方法解释LLM行为，揭示了所谓的“令牌噪音”效应，揭示了LLMs的决策在很大程度上受到提示组件的影响 |
| [^2] | [LAMP: A Language Model on the Map](https://arxiv.org/abs/2403.09059) | 该研究引入了一个新颖的框架，用于在城市特定数据上微调预训练模型，使其能够为人们提供准确的推荐，同时最小化幻觉。 |
| [^3] | [Hire a Linguist!: Learning Endangered Languages with In-Context Linguistic Descriptions](https://arxiv.org/abs/2402.18025) | LINGOLLM是一种无需训练的方法，通过在提示中展示对看不见语言的语言知识，包括词典、语法书和形态分析的输入文本，从而提高了大型语言模型在处理濒危语言方面的翻译能力。 |
| [^4] | [DrAttack: Prompt Decomposition and Reconstruction Makes Powerful LLM Jailbreakers](https://arxiv.org/abs/2402.16914) | 将恶意提示分解为独立的子提示使得LLM越狱攻击更难被检测 |
| [^5] | [Plausible Extractive Rationalization through Semi-Supervised Entailment Signal](https://arxiv.org/abs/2402.08479) | 本文通过半监督方法，采用蕴涵对齐，以优化可行性，提取有理的方式提供一个可解释的替代模型 |
| [^6] | [Explainable Identification of Hate Speech towards Islam using Graph Neural Networks](https://arxiv.org/abs/2311.04916) | 使用图神经网络解释和识别伊斯兰教仇恨言论，模型在保持出色性能的同时能够解释相关性和因果关系。 |

# 详细

[^1]: 等等，这都是令牌噪音？一直就是吗：利用 Shapley 值解释 LLM 行为

    Wait, It's All Token Noise? Always Has Been: Interpreting LLM Behavior Using Shapley Value

    [https://arxiv.org/abs/2404.01332](https://arxiv.org/abs/2404.01332)

    使用Shapley值方法解释LLM行为，揭示了所谓的“令牌噪音”效应，揭示了LLMs的决策在很大程度上受到提示组件的影响

    

    大型语言模型（LLMs）的出现为模拟人类行为和认知过程开辟了新的可能性，潜在应用包括市场研究和消费者行为分析等各个领域。然而，由于LLMs的显著差异暗示了不同的基础过程在起作用，以及LLMs对提示变化的敏感性，利用LLMs作为人类主体的替代仍然存在不确定性。本文提出了一种基于合作博弈理论中Shapley值的新方法来解释LLM行为，并量化每个提示组件对模型输出的相对贡献。通过两个应用--一个离散选择实验和一个认知偏见调查，我们展示了Shapley值方法如何揭示我们所谓的“令牌噪音”效应，即LLM决策受到的影响严重偏向于

    arXiv:2404.01332v1 Announce Type: cross  Abstract: The emergence of large language models (LLMs) has opened up exciting possibilities for simulating human behavior and cognitive processes, with potential applications in various domains, including marketing research and consumer behavior analysis. However, the validity of utilizing LLMs as stand-ins for human subjects remains uncertain due to glaring divergences that suggest fundamentally different underlying processes at play and the sensitivity of LLM responses to prompt variations. This paper presents a novel approach based on Shapley values from cooperative game theory to interpret LLM behavior and quantify the relative contribution of each prompt component to the model's output. Through two applications-a discrete choice experiment and an investigation of cognitive biases-we demonstrate how the Shapley value method can uncover what we term "token noise" effects, a phenomenon where LLM decisions are disproportionately influenced by 
    
[^2]: LAMP：地图上的语言模型

    LAMP: A Language Model on the Map

    [https://arxiv.org/abs/2403.09059](https://arxiv.org/abs/2403.09059)

    该研究引入了一个新颖的框架，用于在城市特定数据上微调预训练模型，使其能够为人们提供准确的推荐，同时最小化幻觉。

    

    大型语言模型（LLMs）在我们的生活中扮演着越来越重要的角色，为我们在各种任务中提供帮助。在地理空间领域，LLMs已经展示出能够回答一般性问题的能力，比如识别一个国家的首都；然而，当涉及回答关于特定地点的细粒度问题时，比如杂货店或餐馆，这些构成了人们日常生活中重要的方面时，它们的效用受到阻碍。这主要是因为我们城市中的地点尚未被系统地输入到LLMs中，以便于理解和记忆它们。该研究引入了一个新颖的框架，用于在城市特定数据上微调预训练模型，从而使其能够提供准确的建议，同时最小化幻觉。我们分享我们的模型LAMP和用于训练它的数据。我们进行实验分析其正确检索空间对象的能力。

    arXiv:2403.09059v1 Announce Type: new  Abstract: Large Language Models (LLMs) are poised to play an increasingly important role in our lives, providing assistance across a wide array of tasks. In the geospatial domain, LLMs have demonstrated the ability to answer generic questions, such as identifying a country's capital; nonetheless, their utility is hindered when it comes to answering fine-grained questions about specific places, such as grocery stores or restaurants, which constitute essential aspects of people's everyday lives. This is mainly because the places in our cities haven't been systematically fed into LLMs, so as to understand and memorize them. This study introduces a novel framework for fine-tuning a pre-trained model on city-specific data, to enable it to provide accurate recommendations, while minimizing hallucinations. We share our model, LAMP, and the data used to train it. We conduct experiments to analyze its ability to correctly retrieving spatial objects, and co
    
[^3]: 雇佣一名语言学家！：通过上下文语言描述学习濒危语言

    Hire a Linguist!: Learning Endangered Languages with In-Context Linguistic Descriptions

    [https://arxiv.org/abs/2402.18025](https://arxiv.org/abs/2402.18025)

    LINGOLLM是一种无需训练的方法，通过在提示中展示对看不见语言的语言知识，包括词典、语法书和形态分析的输入文本，从而提高了大型语言模型在处理濒危语言方面的翻译能力。

    

    arXiv：2402.18025v1

    arXiv:2402.18025v1 Announce Type: new  Abstract: How can large language models (LLMs) process and translate endangered languages? Many languages lack a large corpus to train a decent LLM; therefore existing LLMs rarely perform well in unseen, endangered languages. On the contrary, we observe that 2000 endangered languages, though without a large corpus, have a grammar book or a dictionary. We propose LINGOLLM, a training-free approach to enable an LLM to process unseen languages that hardly occur in its pre-training. Our key insight is to demonstrate linguistic knowledge of an unseen language in an LLM's prompt, including a dictionary, a grammar book, and morphologically analyzed input text. We implement LINGOLLM on top of two models, GPT-4 and Mixtral, and evaluate their performance on 5 tasks across 8 endangered or low-resource languages. Our results show that LINGOLLM elevates translation capability from GPT-4's 0 to 10.5 BLEU for 10 language directions. Our findings demonstrate the
    
[^4]: DrAttack: 提示分解和重构使强大的LLM越狱者

    DrAttack: Prompt Decomposition and Reconstruction Makes Powerful LLM Jailbreakers

    [https://arxiv.org/abs/2402.16914](https://arxiv.org/abs/2402.16914)

    将恶意提示分解为独立的子提示使得LLM越狱攻击更难被检测

    

    本文发现将恶意提示分解为独立的子提示能够有效模糊其潜在的恶意意图，使之以片段化、不易检测的形式呈现，从而解决了这些局限性。我们引入了一个用于越狱攻击的自动提示分解和重构框架（DrAttack）。DrAttack包括三个关键组件：(a) 将原始提示进行“分解”为子提示，(b) 通过上下文学习中的语义上相似但隐含的“重构”这些子提示

    arXiv:2402.16914v1 Announce Type: cross  Abstract: The safety alignment of Large Language Models (LLMs) is vulnerable to both manual and automated jailbreak attacks, which adversarially trigger LLMs to output harmful content. However, current methods for jailbreaking LLMs, which nest entire harmful prompts, are not effective at concealing malicious intent and can be easily identified and rejected by well-aligned LLMs. This paper discovers that decomposing a malicious prompt into separated sub-prompts can effectively obscure its underlying malicious intent by presenting it in a fragmented, less detectable form, thereby addressing these limitations. We introduce an automatic prompt \textbf{D}ecomposition and \textbf{R}econstruction framework for jailbreak \textbf{Attack} (DrAttack). DrAttack includes three key components: (a) `Decomposition' of the original prompt into sub-prompts, (b) `Reconstruction' of these sub-prompts implicitly by in-context learning with semantically similar but h
    
[^5]: 可信的取样合理化通过半监督的蕴涵信号

    Plausible Extractive Rationalization through Semi-Supervised Entailment Signal

    [https://arxiv.org/abs/2402.08479](https://arxiv.org/abs/2402.08479)

    本文通过半监督方法，采用蕴涵对齐，以优化可行性，提取有理的方式提供一个可解释的替代模型

    

    复杂和不透明的黑盒子模型的增加需要采用可解释的措施，其中一种选择是提取有理的模型，它们作为更可解释的替代方案。这些模型，也称为先解释然后预测模型，使用解释模型来提取有理，然后使用提取的信息来调整预测模型。它们的主要目标是提供精确和忠实的解释，由提取的有理表示。在本文中，我们采用半监督方法来优化提取有理的可行性。我们采用一个预训练的自然语言推理（NLI）模型，并在一个小型的有监督有理集（10%）上进一步微调它。通过蕴涵对齐，NLI预测模型被利用作为解释模型的一种监督信号源。通过在问答任务中强制解释和答案之间的对齐一致，我们证明了性能得到了提升。

    The increasing use of complex and opaque black box models requires the adoption of interpretable measures, one such option is extractive rationalizing models, which serve as a more interpretable alternative. These models, also known as Explain-Then-Predict models, employ an explainer model to extract rationales and subsequently condition the predictor with the extracted information. Their primary objective is to provide precise and faithful explanations, represented by the extracted rationales. In this paper, we take a semi-supervised approach to optimize for the plausibility of extracted rationales. We adopt a pre-trained natural language inference (NLI) model and further fine-tune it on a small set of supervised rationales ($10\%$). The NLI predictor is leveraged as a source of supervisory signals to the explainer via entailment alignment. We show that, by enforcing the alignment agreement between the explanation and answer in a question-answering task, the performance can be improve
    
[^6]: 使用图神经网络解释伊斯兰教仇恨言论的研究

    Explainable Identification of Hate Speech towards Islam using Graph Neural Networks

    [https://arxiv.org/abs/2311.04916](https://arxiv.org/abs/2311.04916)

    使用图神经网络解释和识别伊斯兰教仇恨言论，模型在保持出色性能的同时能够解释相关性和因果关系。

    

    伊斯兰教仇恨言论在在线社交互动平台上是一个普遍存在的挑战。识别和消除这种仇恨是迈向和谐与和平未来的关键一步。本研究提出了一种新的范例，利用图神经网络来识别和解释针对伊斯兰教的仇恨言论。利用图神经网络发现、提取并利用不同数据点之间的关系的内在能力，我们的模型始终能够在保持出色性能的同时提供对潜在相关性和因果关系的解释。

    arXiv:2311.04916v2 Announce Type: cross  Abstract: Islamophobic language is a prevalent challenge on online social interaction platforms. Identifying and eliminating such hatred is a crucial step towards a future of harmony and peace. This study presents a novel paradigm for identifying and explaining hate speech towards Islam using graph neural networks. Utilizing the intrinsic ability of graph neural networks to find, extract, and use relationships across disparate data points, our model consistently achieves outstanding performance while offering explanations for the underlying correlations and causation.
    

