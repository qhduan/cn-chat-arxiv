# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [CLAPNQ: Cohesive Long-form Answers from Passages in Natural Questions for RAG systems](https://arxiv.org/abs/2404.02103) | 提出了ClapNQ，一个用于完整RAG管道的基准长格式问答数据集，要求RAG模型能适应包括简洁、一致和不连续段落片段的答案特性。 |
| [^2] | [Improving Retrieval Augmented Open-Domain Question-Answering with Vectorized Contexts](https://arxiv.org/abs/2404.02022) | 本文提出一种通用且便利的方法，通过利用小型编码器语言模型和交叉注意力，使原始语言模型可以覆盖更长的上下文，从而提高开放领域问答任务的性能。 |
| [^3] | [From Skepticism to Acceptance: Simulating the Attitude Dynamics Toward Fake News](https://arxiv.org/abs/2403.09498) | 本研究引入了基于大型语言模型的虚假新闻传播仿真框架，研究了虚假新闻传播的趋势和控制，每个代理人在仿真中代表具有独特个性的个体。 |
| [^4] | [OmniPred: Language Models as Universal Regressors](https://arxiv.org/abs/2402.14547) | 本文提出了OmniPred框架，用于训练语言模型作为通用的端到端回归器，实验证明，在多个任务上训练时，语言模型能够显著优于传统回归模型。 |
| [^5] | [What if LLMs Have Different World Views: Simulating Alien Civilizations with LLM-based Agents](https://arxiv.org/abs/2402.13184) | 这项研究引入了“CosmoAgent”，利用LLM模拟人类和外星文明之间的复杂互动，评估和平共存的可行性，并量化评估文明的发展轨迹，同时考虑不同文明之间的巨大多样性。 |
| [^6] | [LinkNER: Linking Local Named Entity Recognition Models to Large Language Models using Uncertainty](https://arxiv.org/abs/2402.10573) | 提出了一种结合小型微调模型和大型语言模型的LinkNER框架，通过不确定性的链接策略RDC，使微调模型能够补充黑盒LLMs |
| [^7] | [On Provable Length and Compositional Generalization](https://arxiv.org/abs/2402.04875) | 本研究针对包括深度集合、变压器、状态空间模型和简单递归神经网络等多种架构，探索了可证明的长度和组合泛化，认为对于长度和组合泛化，不同架构需要不同程度的表示识别。 |
| [^8] | [Adaptive-Solver Framework for Dynamic Strategy Selection in Large Language Model Reasoning.](http://arxiv.org/abs/2310.01446) | 这个论文提出了一个自适应求解器框架，用于在大型语言模型推理中根据问题难度调整求解策略。这解决了现有方法刚性采用统一方法的问题，提高了计算性能。 |
| [^9] | [Competence-Based Analysis of Language Models.](http://arxiv.org/abs/2303.00333) | 该论文提出了一个基于能力的语言模型分析框架CALM，通过有针对性的干预来破坏语言模型的内部表示，评估其在执行任务时对不同表示的使用。研究表明，语言模型对关系属性的利用存在一定的不一致性。 |

# 详细

[^1]: CLAPNQ：自然问答中的段落内一致长格式答案适用于RAG系统

    CLAPNQ: Cohesive Long-form Answers from Passages in Natural Questions for RAG systems

    [https://arxiv.org/abs/2404.02103](https://arxiv.org/abs/2404.02103)

    提出了ClapNQ，一个用于完整RAG管道的基准长格式问答数据集，要求RAG模型能适应包括简洁、一致和不连续段落片段的答案特性。

    

    检索增强生成（RAG）已成为大型语言模型的热门应用。成功的RAG系统最好提供由段落支持且没有错觉的准确答案。为了构建完整的RAG管道，需要开展大量工作，同时也需要能够对性能进行基准测试。我们提出了ClapNQ，一个用于完整RAG管道的基准长格式问答数据集。ClapNQ包括具有自然问题（NQ）中基于段落的金标段落的长答案，以及一个用于执行检索、生成或完整RAG管道的语料库。ClapNQ的答案简洁，比完整段落小3倍，并且一致，包含不连续的多个段落片段。RAG模型必须适应这些特性才能在ClapNQ上取得成功。我们为ClapNQ提出了基线实验和分析，突出了仍有显著挑战的领域。

    arXiv:2404.02103v1 Announce Type: new  Abstract: Retrieval Augmented Generation (RAG) has become a popular application for large language models. It is preferable that successful RAG systems provide accurate answers that are supported by being grounded in a passage without any hallucinations. While considerable work is required for building a full RAG pipeline, being able to benchmark performance is also necessary. We present ClapNQ, a benchmark Long-form Question Answering dataset for the full RAG pipeline. ClapNQ includes long answers with grounded gold passages from Natural Questions (NQ) and a corpus to perform either retrieval, generation, or the full RAG pipeline. The ClapNQ answers are concise, 3x smaller than the full passage, and cohesive, with multiple pieces of the passage that are not contiguous. RAG models must adapt to these properties to be successful at ClapNQ. We present baseline experiments and analysis for ClapNQ that highlight areas where there is still significant 
    
[^2]: 优化向量化上下文的检索增强开放领域问答

    Improving Retrieval Augmented Open-Domain Question-Answering with Vectorized Contexts

    [https://arxiv.org/abs/2404.02022](https://arxiv.org/abs/2404.02022)

    本文提出一种通用且便利的方法，通过利用小型编码器语言模型和交叉注意力，使原始语言模型可以覆盖更长的上下文，从而提高开放领域问答任务的性能。

    

    在大型语言模型时代，应用检索增强生成等技术可以更好地解决开放领域问答问题。由于模型大小和计算资源等约束，上下文长度通常受限，让模型覆盖过长的上下文并回答来自开放领域的问题变得具有挑战性。本文提出了一种在开放领域问答任务中覆盖更长上下文的通用、方便方法。它利用一个小型编码器语言模型有效编码上下文，并对原始输入应用交叉注意力。通过我们的方法，原始语言模型可以覆盖几倍长的上下文，同时保持与基线接近的计算需求。我们的实验表明，在微调后，性能在两个保存的数据集、四个保留的数据集以及两个In Context

    arXiv:2404.02022v1 Announce Type: new  Abstract: In the era of large language models, applying techniques such as Retrieval Augmented Generation can better address Open-Domain Question-Answering problems. Due to constraints including model sizes and computing resources, the length of context is often limited, and it becomes challenging to empower the model to cover overlong contexts while answering questions from open domains. This paper proposes a general and convenient method to covering longer contexts in Open-Domain Question-Answering tasks. It leverages a small encoder language model that effectively encodes contexts, and the encoding applies cross-attention with origin inputs. With our method, the origin language models can cover several times longer contexts while keeping the computing requirements close to the baseline. Our experiments demonstrate that after fine-tuning, there is improved performance across two held-in datasets, four held-out datasets, and also in two In Contex
    
[^3]: 从怀疑到接受：模拟对虚假新闻态度动态的变化

    From Skepticism to Acceptance: Simulating the Attitude Dynamics Toward Fake News

    [https://arxiv.org/abs/2403.09498](https://arxiv.org/abs/2403.09498)

    本研究引入了基于大型语言模型的虚假新闻传播仿真框架，研究了虚假新闻传播的趋势和控制，每个代理人在仿真中代表具有独特个性的个体。

    

    在数字时代，虚假新闻和谣言通过社交网络迅速传播，带来了显著的社会挑战，影响着公众舆论。传统的虚假新闻建模通常预测不同群体的普遍流行趋势或数字化代表意见转变。然而，这些方法经常过于简化现实世界的复杂性，忽视了新闻文本丰富的语义信息。大型语言模型（LLMs）的出现提供了模拟微妙意见动态的可能性。因此，在这项工作中，我们引入了基于LLM的虚假新闻传播仿真框架（FPS），详细研究虚假新闻传播的趋势和控制。具体地，仿真中的每个代理人代表具有独特个性的个人。他们配备了短期和长期记忆，以及反思机制来模仿类人思维。每天，

    arXiv:2403.09498v1 Announce Type: cross  Abstract: In the digital era, the rapid propagation of fake news and rumors via social networks brings notable societal challenges and impacts public opinion regulation. Traditional fake news modeling typically forecasts the general popularity trends of different groups or numerically represents opinions shift. However, these methods often oversimplify real-world complexities and overlook the rich semantic information of news text. The advent of large language models (LLMs) provides the possibility of modeling subtle dynamics of opinion. Consequently, in this work, we introduce a Fake news Propagation Simulation framework (FPS) based on LLM, which studies the trends and control of fake news propagation in detail. Specifically, each agent in the simulation represents an individual with a distinct personality. They are equipped with both short-term and long-term memory, as well as a reflective mechanism to mimic human-like thinking. Every day, the
    
[^4]: OmniPred：语言模型作为通用回归器

    OmniPred: Language Models as Universal Regressors

    [https://arxiv.org/abs/2402.14547](https://arxiv.org/abs/2402.14547)

    本文提出了OmniPred框架，用于训练语言模型作为通用的端到端回归器，实验证明，在多个任务上训练时，语言模型能够显著优于传统回归模型。

    

    在实验设计的广阔领域中，回归一直是一个强大的工具，可以准确预测系统或模型在给定一组参数的情况下的结果指标，但传统上只限于适用于特定任务的方法。在本文中，我们提出了OmniPred，这是一个用于训练语言模型作为通用端到端回归器的框架，使用来自多样真实世界实验的$(x,y)$评估数据。通过使用源自Google Vizier的数据，这是世界上最大的黑盒优化数据库之一，我们的大量实验表明，仅通过数学参数和值的文本表示，语言模型能够进行非常精确的数值回归，如果有机会训练多个任务，则可以显著优于传统的回归模型。

    arXiv:2402.14547v1 Announce Type: cross  Abstract: Over the broad landscape of experimental design, regression has been a powerful tool to accurately predict the outcome metrics of a system or model given a set of parameters, but has been traditionally restricted to methods which are only applicable to a specific task. In this paper, we propose OmniPred, a framework for training language models as universal end-to-end regressors over $(x,y)$ evaluation data from diverse real world experiments. Using data sourced from Google Vizier, one of the largest blackbox optimization databases in the world, our extensive experiments demonstrate that through only textual representations of mathematical parameters and values, language models are capable of very precise numerical regression, and if given the opportunity to train over multiple tasks, can significantly outperform traditional regression models.
    
[^5]: 如果LLM具有不同的世界观：使用基于LLM的代理模拟外星文明

    What if LLMs Have Different World Views: Simulating Alien Civilizations with LLM-based Agents

    [https://arxiv.org/abs/2402.13184](https://arxiv.org/abs/2402.13184)

    这项研究引入了“CosmoAgent”，利用LLM模拟人类和外星文明之间的复杂互动，评估和平共存的可行性，并量化评估文明的发展轨迹，同时考虑不同文明之间的巨大多样性。

    

    在这项研究中，我们介绍了“CosmoAgent”，这是一个创新的人工智能框架，利用大型语言模型（LLMs）来模拟人类与外星文明之间复杂的交互，特别强调史蒂芬·霍金关于不要随意向宇宙发送无线电信号的谨慎建议。该研究的目标是评估和平共存的可行性，同时考虑可能威胁善意文明的潜在风险。通过采用数学模型和状态转换矩阵，我们的方法定量评估文明的发展轨迹，为在关键增长和饱和点做出未来决策提供见解。此外，本文承认宇宙中潜在生活条件的巨大多样性可能会促进不同文明之间独特的宇宙观、道德准则和世界观。认识到地球上--

    arXiv:2402.13184v1 Announce Type: new  Abstract: In this study, we introduce "CosmoAgent," an innovative artificial intelligence framework utilizing Large Language Models (LLMs) to simulate complex interactions between human and extraterrestrial civilizations, with a special emphasis on Stephen Hawking's cautionary advice about not sending radio signals haphazardly into the universe. The goal is to assess the feasibility of peaceful coexistence while considering potential risks that could threaten well-intentioned civilizations. Employing mathematical models and state transition matrices, our approach quantitatively evaluates the development trajectories of civilizations, offering insights into future decision-making at critical points of growth and saturation. Furthermore, the paper acknowledges the vast diversity in potential living conditions across the universe, which could foster unique cosmologies, ethical codes, and worldviews among various civilizations. Recognizing the Earth-c
    
[^6]: LinkNER: 使用不确定性将本地命名实体识别模型与大语言模型进行链接

    LinkNER: Linking Local Named Entity Recognition Models to Large Language Models using Uncertainty

    [https://arxiv.org/abs/2402.10573](https://arxiv.org/abs/2402.10573)

    提出了一种结合小型微调模型和大型语言模型的LinkNER框架，通过不确定性的链接策略RDC，使微调模型能够补充黑盒LLMs

    

    命名实体识别（NER）作为自然语言理解中的基本任务，直接影响着网络内容分析、搜索引擎和信息检索系统。微调后的NER模型在标准NER基准上表现出令人满意的性能。然而，由于有限的微调数据和缺乏知识，它在未见实体识别上表现不佳。因此，NER模型在网络相关应用中的可用性和可靠性受到影响。相反，像GPT-4这样的大型语言模型（LLM）具有丰富的外部知识，但研究表明它们缺乏NER任务的专业性。此外，私有和大规模权重使LLM的调整困难。为了解决这些挑战，我们提出了一个框架，结合了小型微调模型和LLMs（LinkNER），以及一种基于不确定性的链接策略RDC，使微调模型能够补充黑盒LLMs。

    arXiv:2402.10573v1 Announce Type: new  Abstract: Named Entity Recognition (NER) serves as a fundamental task in natural language understanding, bearing direct implications for web content analysis, search engines, and information retrieval systems. Fine-tuned NER models exhibit satisfactory performance on standard NER benchmarks. However, due to limited fine-tuning data and lack of knowledge, it performs poorly on unseen entity recognition. As a result, the usability and reliability of NER models in web-related applications are compromised. Instead, Large Language Models (LLMs) like GPT-4 possess extensive external knowledge, but research indicates that they lack specialty for NER tasks. Furthermore, non-public and large-scale weights make tuning LLMs difficult. To address these challenges, we propose a framework that combines small fine-tuned models with LLMs (LinkNER) and an uncertainty-based linking strategy called RDC that enables fine-tuned models to complement black-box LLMs, ach
    
[^7]: 关于可证明的长度和组合泛化

    On Provable Length and Compositional Generalization

    [https://arxiv.org/abs/2402.04875](https://arxiv.org/abs/2402.04875)

    本研究针对包括深度集合、变压器、状态空间模型和简单递归神经网络等多种架构，探索了可证明的长度和组合泛化，认为对于长度和组合泛化，不同架构需要不同程度的表示识别。

    

    长度泛化——对训练时未见到的更长序列的泛化能力，以及组合泛化——对训练时未见到的令牌组合的泛化能力，在序列到序列模型中是重要的非分布化泛化形式。在这项工作中，我们在包括深度集合、变压器、状态空间模型和简单递归神经网络在内的一系列架构中，朝着可证明的长度和组合泛化迈出了第一步。根据架构的不同，我们证明了不同程度的表示识别的必要性，例如与真实表示具有线性或排列关系。

    Length generalization -- the ability to generalize to longer sequences than ones seen during training, and compositional generalization -- the ability to generalize to token combinations not seen during training, are crucial forms of out-of-distribution generalization in sequence-to-sequence models. In this work, we take the first steps towards provable length and compositional generalization for a range of architectures, including deep sets, transformers, state space models, and simple recurrent neural nets. Depending on the architecture, we prove different degrees of representation identification, e.g., a linear or a permutation relation with ground truth representation, is necessary for length and compositional generalization.
    
[^8]: 大型语言模型推理中的动态策略选择自适应求解器框架

    Adaptive-Solver Framework for Dynamic Strategy Selection in Large Language Model Reasoning. (arXiv:2310.01446v1 [cs.CL])

    [http://arxiv.org/abs/2310.01446](http://arxiv.org/abs/2310.01446)

    这个论文提出了一个自适应求解器框架，用于在大型语言模型推理中根据问题难度调整求解策略。这解决了现有方法刚性采用统一方法的问题，提高了计算性能。

    

    大型语言模型(LLM)在处理复杂推理任务时展示了令人印象深刻的能力。在现实世界中，问题往往涉及各种复杂性。人类本能地根据任务的复杂性调整他们的问题解决方法。然而，大多数利用LLM的方法倾向于采用一种统一的方法: 不管问题的复杂性如何，都使用一致的模型、提示方法和问题分解程度。这种刚性可能会带来不必要的计算开销或次优的性能。为了解决这个问题，我们引入了一个自适应求解器框架。它根据问题的难度策略性地调整求解策略。给定一个初始解决方案，该框架使用两个主要模块。初始评估模块评估当前解决方案的充分性。如果需要改进，接下来的自适应模块会介入。在这个模块内，有三个关键的自适应策略。

    Large Language Models (LLMs) are showcasing impressive ability in handling complex reasoning tasks. In real-world situations, problems often span a spectrum of complexities. Humans inherently adjust their problem-solving approaches based on task complexity. However, most methodologies that leverage LLMs tend to adopt a uniform approach: utilizing consistent models, prompting methods, and degrees of problem decomposition, regardless of the problem complexity. Inflexibility of them can bring unnecessary computational overhead or sub-optimal performance. To address this problem, we introduce an Adaptive-Solver framework. It strategically modulates solving strategies based on the difficulties of the problems. Given an initial solution, the framework functions with two primary modules. The initial evaluation module assesses the adequacy of the current solution. If improvements are needed, the subsequent adaptation module comes into play. Within this module, three key adaptation strategies a
    
[^9]: 基于能力的语言模型分析

    Competence-Based Analysis of Language Models. (arXiv:2303.00333v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2303.00333](http://arxiv.org/abs/2303.00333)

    该论文提出了一个基于能力的语言模型分析框架CALM，通过有针对性的干预来破坏语言模型的内部表示，评估其在执行任务时对不同表示的使用。研究表明，语言模型对关系属性的利用存在一定的不一致性。

    

    尽管大型预训练语言模型（LMs）在各种提示任务上取得了显著成功，但这些模型对输入或应用环境中的微小变化却异常脆弱。为了更好地理解这种行为并激励设计更健壮的LMs，我们提出了一个通用的实验框架CALM（基于能力的语言模型分析），其中利用有针对性的因果干预来破坏LM在各种语言属性上的内部表示，以评估它在执行给定任务时对每个表示的使用。我们将这些干预实现为基于梯度的对抗攻击，与先前的因果探查方法相比，它们能够针对任意编码的关系属性进行攻击，并进行了一个案例研究，分析了BERT-like LMs在执行相关关系提示任务时如何使用多种关系属性的表示。我们发现，虽然表示的选择对LM的性能产生了影响，但模型对某些特定关系属性的利用并不一致。

    Despite the recent success of large pretrained language models (LMs) on a variety of prompting tasks, these models can be alarmingly brittle to small changes in inputs or application contexts. To better understand such behavior and motivate the design of more robust LMs, we propose a general experimental framework, CALM (Competence-based Analysis of Language Models), where targeted causal interventions are utilized to damage an LM's internal representation of various linguistic properties in order to evaluate its use of each representation in performing a given task. We implement these interventions as gradient-based adversarial attacks, which (in contrast to prior causal probing methodologies) are able to target arbitrarily-encoded representations of relational properties, and carry out a case study of this approach to analyze how BERT-like LMs use representations of several relational properties in performing associated relation prompting tasks. We find that, while the representation
    

