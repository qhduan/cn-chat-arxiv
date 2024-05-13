# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Analyzing the Roles of Language and Vision in Learning from Limited Data](https://arxiv.org/abs/2403.19669) | 研究人工智能中的复杂视觉-语言模型，发现即使缺乏视觉输入，利用所有组件的语言模型能够恢复大部分VLM的性能，表明语言通过提供对先前知识和推理的访问来对学习新任务有贡献 |
| [^2] | [TriviaHG: A Dataset for Automatic Hint Generation from Factoid Questions](https://arxiv.org/abs/2403.18426) | 提出了一个用于自动提示生成的框架，构建了一个包含160,230个提示的大规模数据集TriviaHG，并提出了一种评估方法来衡量提示的质量属性。 |
| [^3] | [ChroniclingAmericaQA: A Large-scale Question Answering Dataset based on Historical American Newspaper Pages](https://arxiv.org/abs/2403.17859) | ChroniclingAmericaQA是一个基于历史美国报纸页面的大规模问答数据集，旨在推动QA和MRC任务的发展，并克服以往数据集的局限性。 |
| [^4] | [From explainable to interpretable deep learning for natural language processing in healthcare: how far from reality?](https://arxiv.org/abs/2403.11894) | 该研究对医疗保健NLP中的深度学习进行了全面审查，提出了可解释和可解释的人工智能（XIAI）概念，并发现注意机制是主要新兴IAI，同时面临着缺乏全局建模、最佳实践以及系统评估和基准测试的挑战。 |
| [^5] | [Can Factual Statements be Deceptive? The DeFaBel Corpus of Belief-based Deception](https://arxiv.org/abs/2403.10185) | 论文提出了DeFaBel语料库，这是一个基于信仰的欺骗的众包资源，用于研究欺骗与事实性之间的关系，并强调了论证中事实性、个人信念和欺骗意图之间的重要性。 |
| [^6] | [RNNs are not Transformers (Yet): The Key Bottleneck on In-context Retrieval](https://arxiv.org/abs/2402.18510) | 本文研究了RNNs和Transformer在处理算法问题时的表现能力差距，发现RNNs存在关键瓶颈，即无法完美地从上下文中检索信息，导致无法像Transformer那样轻松解决需要这种能力的任务。 |
| [^7] | [Nissist: An Incident Mitigation Copilot based on Troubleshooting Guides](https://arxiv.org/abs/2402.17531) | Nissist利用TSGs和事故缓解历史提供主动建议，减少人为干预，以提高企业级云服务的事故管理效率。 |
| [^8] | [Large Language Models are Clinical Reasoners: Reasoning-Aware Diagnosis Framework with Prompt-Generated Rationales](https://arxiv.org/abs/2312.07399) | 该论文提出了一种基于提示生成的理由的“推理感知”诊断框架，通过大型语言模型来进行临床推理，实现了在疾病诊断过程中的高效、时间节约和劳动节约的方法。 |
| [^9] | [From Interpolation to Extrapolation: Complete Length Generalization for Arithmetic Transformers.](http://arxiv.org/abs/2310.11984) | 本文研究了Transformer模型在学习算术算法方面的能力，并通过注意力偏置以及Attention Bias Calibration（ABC）来实现对于长长度的泛化。 |
| [^10] | [Faithfulness Measurable Masked Language Models.](http://arxiv.org/abs/2310.07819) | 本论文提出了一种可度量忠实性的掩码语言模型，通过使用一种新颖的微调方法，将屏蔽令牌作为设计使其成为分布内，以解决解释自然语言处理模型时常见的问题。 |
| [^11] | [Using Large Language Models to Generate, Validate, and Apply User Intent Taxonomies.](http://arxiv.org/abs/2309.13063) | 通过使用大型语言模型生成用户意图分类，我们提出了一种新方法来分析和验证日志数据中的用户意图，从而解决了手动或基于机器学习的标注方法在大型和不断变化的数据集上的问题。 |
| [^12] | [The Regular Expression Inference Challenge.](http://arxiv.org/abs/2308.07899) | 正则表达式推理挑战是一个以找到最小正则表达式为目标的任务，具有应用广泛、难度可调、适用于代码/语言建模和机器学习的特点。 |
| [^13] | [A Unified Review of Deep Learning for Automated Medical Coding.](http://arxiv.org/abs/2201.02797) | 本文综述了深度学习在自动医疗编码领域的发展，提出了一个统一框架，总结了最新的高级模型，并讨论了未来发展的挑战和方向。 |

# 详细

[^1]: 分析语言和视觉在从有限数据中学习中的作用

    Analyzing the Roles of Language and Vision in Learning from Limited Data

    [https://arxiv.org/abs/2403.19669](https://arxiv.org/abs/2403.19669)

    研究人工智能中的复杂视觉-语言模型，发现即使缺乏视觉输入，利用所有组件的语言模型能够恢复大部分VLM的性能，表明语言通过提供对先前知识和推理的访问来对学习新任务有贡献

    

    arXiv:2403.19669v1 公告类型：交叉摘要：语言是否有助于理解视觉世界？实际观察世界需要看到实际情况，而不是用文字描述吗？关于智能本质的这些基本问题很难回答，因为我们只有一个智能系统的例子——人类——以及有限的独立语言或视觉的案例。然而，人工智能研究人员开发出复杂的视觉-语言模型（VLMs）为我们提供了新的机会，探索语言和视觉对于学习世界的贡献。我们从这些模型的认知架构中切除组件，以确定它们对从有限数据中学习新任务的贡献。我们发现，利用所有组件的语言模型恢复了大部分VLM的性能，尽管它缺乏视觉输入，而语言似乎可以通过提供对先前知识和推理的访问来实现这一点。

    arXiv:2403.19669v1 Announce Type: cross  Abstract: Does language help make sense of the visual world? How important is it to actually see the world rather than having it described with words? These basic questions about the nature of intelligence have been difficult to answer because we only had one example of an intelligent system -- humans -- and limited access to cases that isolated language or vision. However, the development of sophisticated Vision-Language Models (VLMs) by artificial intelligence researchers offers us new opportunities to explore the contributions that language and vision make to learning about the world. We ablate components from the cognitive architecture of these models to identify their contributions to learning new tasks from limited data. We find that a language model leveraging all components recovers a majority of a VLM's performance, despite its lack of visual input, and that language seems to allow this by providing access to prior knowledge and reasoni
    
[^2]: TriviaHG：用于从事实性问题生成自动提示的数据集

    TriviaHG: A Dataset for Automatic Hint Generation from Factoid Questions

    [https://arxiv.org/abs/2403.18426](https://arxiv.org/abs/2403.18426)

    提出了一个用于自动提示生成的框架，构建了一个包含160,230个提示的大规模数据集TriviaHG，并提出了一种评估方法来衡量提示的质量属性。

    

    现在，个人倾向于与大型语言模型进行对话，寻找他们问题的答案。在这样的答案对任何人都很容易获得的时代，刺激和保持人类的认知能力，以及确保人类保持良好推理能力变得至关重要。本研究通过提出提示（而不是最终答案或在给出答案之前）作为一种可行的解决方案来满足这些需求。我们介绍了一个用于事实性问题的自动提示生成框架，利用它构建了TriviaHG，这是一个新颖的大规模数据集，包含来自TriviaQA数据集的16,645个问题对应的160,230个提示。此外，我们提出了一种自动评估方法，用于衡量提示的收敛性和熟悉度质量属性。为了评估TriviaHG数据集和所提出的评估方法，我们邀请了10名个体注释2,791个提示，并分配了6名研究人员

    arXiv:2403.18426v1 Announce Type: new  Abstract: Nowadays, individuals tend to engage in dialogues with Large Language Models, seeking answers to their questions. In times when such answers are readily accessible to anyone, the stimulation and preservation of human's cognitive abilities, as well as the assurance of maintaining good reasoning skills by humans becomes crucial. This study addresses such needs by proposing hints (instead of final answers or before giving answers) as a viable solution. We introduce a framework for the automatic hint generation for factoid questions, employing it to construct TriviaHG, a novel large-scale dataset featuring 160,230 hints corresponding to 16,645 questions from the TriviaQA dataset. Additionally, we present an automatic evaluation method that measures the Convergence and Familiarity quality attributes of hints. To evaluate the TriviaHG dataset and the proposed evaluation method, we enlisted 10 individuals to annotate 2,791 hints and tasked 6 hu
    
[^3]: ChroniclingAmericaQA:基于历史美国报纸页面的大规模问答数据集

    ChroniclingAmericaQA: A Large-scale Question Answering Dataset based on Historical American Newspaper Pages

    [https://arxiv.org/abs/2403.17859](https://arxiv.org/abs/2403.17859)

    ChroniclingAmericaQA是一个基于历史美国报纸页面的大规模问答数据集，旨在推动QA和MRC任务的发展，并克服以往数据集的局限性。

    

    arXiv:2403.17859v1公告类型：新问答（QA）和机器阅读理解（MRC）任务由于深度学习技术的快速发展以及最近的大语言模型而取得了显著进展。同时，许多基准数据集已经用于QA和MRC任务。然而，大多数现有的大规模基准数据集主要使用同步文档集合（如维基百科或网络）创建。档案文件集合，如历史报纸，包含过去的宝贵信息，但仍未被广泛用于训练大型语言模型。为了进一步推动QA和MRC任务的发展，并克服先前数据集的局限性，我们介绍了ChroniclingAmericaQA，这是一个基于历史报纸集Chronicling America创建的拥有485K问答对的大规模数据集。我们的数据集是从Chronicling Amer的子集构建的。

    arXiv:2403.17859v1 Announce Type: new  Abstract: Question answering (QA) and Machine Reading Comprehension (MRC) tasks have significantly advanced in recent years due to the rapid development of deep learning techniques and, more recently, large language models. At the same time, many benchmark datasets have become available for QA and MRC tasks. However, most existing large-scale benchmark datasets have been created predominantly using synchronous document collections like Wikipedia or the Web. Archival document collections, such as historical newspapers, contain valuable information from the past that is still not widely used to train large language models. To further contribute to advancing QA and MRC tasks and to overcome the limitation of previous datasets, we introduce ChroniclingAmericaQA, a large-scale dataset with 485K question-answer pairs created based on the historical newspaper collection Chronicling America. Our dataset is constructed from a subset of the Chronicling Amer
    
[^4]: 从可解释到可解释的深度学习在医疗自然语言处理中的应用：现实有多远？

    From explainable to interpretable deep learning for natural language processing in healthcare: how far from reality?

    [https://arxiv.org/abs/2403.11894](https://arxiv.org/abs/2403.11894)

    该研究对医疗保健NLP中的深度学习进行了全面审查，提出了可解释和可解释的人工智能（XIAI）概念，并发现注意机制是主要新兴IAI，同时面临着缺乏全局建模、最佳实践以及系统评估和基准测试的挑战。

    

    深度学习（DL）通过解决各种自然语言处理（NLP）任务，极大地增强了医疗保健研究。然而，基于DL的NLP方法日益复杂，需要透明的模型解释性，或至少是可解释性，以进行可靠的决策制定。本文对医疗健康NLP中的可解释和可解释的DL进行了彻底的范围审查。引入了术语“XIAI”（eXplainable和Interpretable Artificial Intelligence）以区分XAI和IAI。方法根据其功能（模型、输入、输出为基础）和范围（局部、全局）进一步分类。我们的分析表明，注意机制是最主要的新兴IAI。此外，IAI越来越多地用于对抗XAI。确定的主要挑战是大多数XIAI不探索“全局”建模过程，缺乏最佳实践，并且需要系统评估和基准测试。

    arXiv:2403.11894v1 Announce Type: cross  Abstract: Deep learning (DL) has substantially enhanced healthcare research by addressing various natural language processing (NLP) tasks. Yet, the increasing complexity of DL-based NLP methods necessitates transparent model interpretability, or at least explainability, for reliable decision-making. This work presents a thorough scoping review on explainable and interpretable DL in healthcare NLP. The term "XIAI" (eXplainable and Interpretable Artificial Intelligence) was introduced to distinguish XAI from IAI. Methods were further categorized based on their functionality (model-, input-, output-based) and scope (local, global). Our analysis shows that attention mechanisms were the most dominant emerging IAI. Moreover, IAI is increasingly used against XAI. The major challenges identified are that most XIAI do not explore "global" modeling processes, the lack of best practices, and the unmet need for systematic evaluation and benchmarks. Importan
    
[^5]: 可以欺骗性地陈述事实吗？基于信仰的欺骗DeFaBel语料库

    Can Factual Statements be Deceptive? The DeFaBel Corpus of Belief-based Deception

    [https://arxiv.org/abs/2403.10185](https://arxiv.org/abs/2403.10185)

    论文提出了DeFaBel语料库，这是一个基于信仰的欺骗的众包资源，用于研究欺骗与事实性之间的关系，并强调了论证中事实性、个人信念和欺骗意图之间的重要性。

    

    如果一个人坚信一个非事实性的陈述，比如“地球是平的”，并为其辩护，那么他并没有本质上的欺骗意图。由于论证源于真诚的信念，它可能不太可能展示出与欺骗或撒谎相关的语言特征。事实性、个人信念和欺骗意图之间的相互作用仍然是一个未经充分研究的领域。解开这些变量在论证中的影响对于更好地理解归因于每一个的语言特征至关重要。为了研究基于信念的欺骗与事实性之间的关系，我们提出了DeFaBel语料库，这是一个基于信仰的欺骗的众包资源。为了创建这个语料库，我们设计了一个研究，要求参与者撰写支持诸如“食用西瓜籽可能导致消化不良”的陈述，而不论其事实准确性或个人信念如何。

    arXiv:2403.10185v1 Announce Type: new  Abstract: If a person firmly believes in a non-factual statement, such as "The Earth is flat", and argues in its favor, there is no inherent intention to deceive. As the argumentation stems from genuine belief, it may be unlikely to exhibit the linguistic properties associated with deception or lying. This interplay of factuality, personal belief, and intent to deceive remains an understudied area. Disentangling the influence of these variables in argumentation is crucial to gain a better understanding of the linguistic properties attributed to each of them. To study the relation between deception and factuality, based on belief, we present the DeFaBel corpus, a crowd-sourced resource of belief-based deception. To create this corpus, we devise a study in which participants are instructed to write arguments supporting statements like "eating watermelon seeds can cause indigestion", regardless of its factual accuracy or their personal beliefs about 
    
[^6]: RNNs还不是Transformer：在上下文检索中的关键瓶颈

    RNNs are not Transformers (Yet): The Key Bottleneck on In-context Retrieval

    [https://arxiv.org/abs/2402.18510](https://arxiv.org/abs/2402.18510)

    本文研究了RNNs和Transformer在处理算法问题时的表现能力差距，发现RNNs存在关键瓶颈，即无法完美地从上下文中检索信息，导致无法像Transformer那样轻松解决需要这种能力的任务。

    

    本文探讨循环神经网络（RNNs）和Transformer在解决算法问题时的表示能力差距。我们重点关注RNNs是否能在处理长序列时，通过Chain-of-Thought (CoT)提示，与Transformer的性能相匹配。我们的理论分析显示CoT可以改进RNNs，但无法弥补与Transformer之间的差距。关键瓶颈在于RNNs无法完全从上下文中检索信息，即使经过CoT的增强：对于几个明确或隐式需要这种能力的任务，如联想召回和确定图是否为树，我们证明RNNs表达能力不足以解决这些任务，而Transformer可以轻松解决。相反，我们证明采用增强RNNs上下文检索能力的技术，包括

    arXiv:2402.18510v1 Announce Type: cross  Abstract: This paper investigates the gap in representation powers of Recurrent Neural Networks (RNNs) and Transformers in the context of solving algorithmic problems. We focus on understanding whether RNNs, known for their memory efficiency in handling long sequences, can match the performance of Transformers, particularly when enhanced with Chain-of-Thought (CoT) prompting. Our theoretical analysis reveals that CoT improves RNNs but is insufficient to close the gap with Transformers. A key bottleneck lies in the inability of RNNs to perfectly retrieve information from the context, even with CoT: for several tasks that explicitly or implicitly require this capability, such as associative recall and determining if a graph is a tree, we prove that RNNs are not expressive enough to solve the tasks while Transformers can solve them with ease. Conversely, we prove that adopting techniques to enhance the in-context retrieval capability of RNNs, inclu
    
[^7]: Nissist：基于故障排除指南的事故缓解副驾驶

    Nissist: An Incident Mitigation Copilot based on Troubleshooting Guides

    [https://arxiv.org/abs/2402.17531](https://arxiv.org/abs/2402.17531)

    Nissist利用TSGs和事故缓解历史提供主动建议，减少人为干预，以提高企业级云服务的事故管理效率。

    

    有效的事故管理对企业级云服务的顺畅运作至关重要。 为了加速事故缓解，服务团队将故障排除知识编译成供值班工程师（OCEs）访问的故障排除指南（TSGs）。 尽管自动化流水线已能够解决最常见和简单的事故，但仍存在需要OCE干预的复杂事故。 然而，TSGs通常是非结构化和不完整的，这需要OCE手动解释，导致值班疲劳和生产力下降，特别是新入职的OCE。 在这项工作中，我们提出了Nissist，它利用TSGs和事故缓解历史提供主动建议，减少人为干预。 利用大型语言模型（LLM），Nissist从非结构化TSGs和历史事故缓解讨论中提取见解，形成全面的知识库。

    arXiv:2402.17531v1 Announce Type: cross  Abstract: Effective incident management is pivotal for the smooth operation of enterprises-level cloud services. In order to expedite incident mitigation, service teams compile troubleshooting knowledge into Troubleshooting Guides (TSGs) accessible to on-call engineers (OCEs). While automated pipelines are enabled to resolve the most frequent and easy incidents, there still exist complex incidents that require OCEs' intervention. However, TSGs are often unstructured and incomplete, which requires manual interpretation by OCEs, leading to on-call fatigue and decreased productivity, especially among new-hire OCEs. In this work, we propose Nissist which leverages TSGs and incident mitigation histories to provide proactive suggestions, reducing human intervention. Leveraging Large Language Models (LLM), Nissist extracts insights from unstructured TSGs and historical incident mitigation discussions, forming a comprehensive knowledge base. Its multi-a
    
[^8]: 大型语言模型是临床推理者：基于提示生成的理由的推理感知诊断框架

    Large Language Models are Clinical Reasoners: Reasoning-Aware Diagnosis Framework with Prompt-Generated Rationales

    [https://arxiv.org/abs/2312.07399](https://arxiv.org/abs/2312.07399)

    该论文提出了一种基于提示生成的理由的“推理感知”诊断框架，通过大型语言模型来进行临床推理，实现了在疾病诊断过程中的高效、时间节约和劳动节约的方法。

    

    由于大型语言模型（LLMs）的进展，机器推理在近年来取得了巨大的进展。然而，在临床领域，大多数以自然语言处理为驱动的项目主要集中在临床分类或阅读理解上，并且由于与临床医生的理念注解成本较高，对于疾病诊断的临床推理还未得到充分的研究。在这项工作中，我们提出了一个“推理感知”的诊断框架，通过基于提示的学习以一种高效的时间和劳动方式去理性化诊断过程，并学习对提示生成的理由进行推理。具体而言，我们解决了疾病诊断的临床推理问题，其中LLM生成了诊断性的理由，提供其对呈现的患者数据的见解以及达到诊断的推理路径，即临床思维链（Clinical CoT）。我们通过广泛的实验和分析在理由生成和疾病诊断方面实证了LLMs/LMs的临床推理能力。

    Machine reasoning has made great progress in recent years owing to large language models (LLMs). In the clinical domain, however, most NLP-driven projects mainly focus on clinical classification or reading comprehension, and under-explore clinical reasoning for disease diagnosis due to the expensive rationale annotation with clinicians. In this work, we present a ``reasoning-aware'' diagnosis framework that rationalizes the diagnostic process via prompt-based learning in a time- and labor-efficient manner, and learns to reason over the prompt-generated rationales. Specifically, we address the clinical reasoning for disease diagnosis, where the LLM generates diagnostic rationales providing its insight on presented patient data and the reasoning path towards the diagnosis, namely Clinical Chain-of-Thought (Clinical CoT). We empirically demonstrate LLMs/LMs' ability of clinical reasoning via extensive experiments and analyses on both rationale generation and disease diagnosis in various s
    
[^9]: 从插值到外推：算术Transformer的完整长度泛化

    From Interpolation to Extrapolation: Complete Length Generalization for Arithmetic Transformers. (arXiv:2310.11984v1 [cs.LG])

    [http://arxiv.org/abs/2310.11984](http://arxiv.org/abs/2310.11984)

    本文研究了Transformer模型在学习算术算法方面的能力，并通过注意力偏置以及Attention Bias Calibration（ABC）来实现对于长长度的泛化。

    

    自从提出以来，Transformer模型在各种任务中展现出了优秀的性能。然而，在算法任务中，长度泛化仍存在一些未解决的问题。在本文中，我们研究了Transformer模型在学习算术算法（如加法和乘法）方面的内在能力。通过实验证明和注意力分析，我们确定了实现最佳长度泛化的几个关键因素。我们展示了Transformer模型能够通过目标指向偏置来泛化到长长度。然后，我们引入了Attention Bias Calibration（ABC），这是一个校准阶段，使模型能够自动学习适当的注意力偏置，我们将其与相对位置编码的机制联系起来。我们证明使用ABC，Transformer模型可以在某些算术任务上实现前所未有的完美长度泛化。

    Since its introduction, the transformer model has demonstrated outstanding performance across various tasks. However, there are still unresolved issues regarding length generalization, particularly in algorithmic tasks. In this paper, we investigate the inherent capabilities of transformer models in learning arithmetic algorithms, such as addition and multiplication. Through experiments and attention analysis, we identify a number of crucial factors for achieving optimal length generalization. We show that transformer models are able to generalize to long lengths with the help of targeted attention biasing. We then introduce Attention Bias Calibration (ABC), a calibration stage that enables the model to automatically learn the proper attention biases, which we link to mechanisms in relative position encoding. We demonstrate that using ABC, the transformer model can achieve unprecedented perfect length generalization on certain arithmetic tasks.
    
[^10]: 可度量忠实性的掩码语言模型

    Faithfulness Measurable Masked Language Models. (arXiv:2310.07819v1 [cs.CL])

    [http://arxiv.org/abs/2310.07819](http://arxiv.org/abs/2310.07819)

    本论文提出了一种可度量忠实性的掩码语言模型，通过使用一种新颖的微调方法，将屏蔽令牌作为设计使其成为分布内，以解决解释自然语言处理模型时常见的问题。

    

    解释自然语言处理模型的常见方法是使用重要性度量来表达哪些令牌对于预测很重要。然而，尽管这些解释具有说服力，但往往是错误的。因此，测量它们的忠实性至关重要。其中一种度量标准是如果令牌确实很重要，那么屏蔽它们应该导致模型性能变差。然而，令牌屏蔽会引入区域外问题，而现有的解决方案在计算上很昂贵并且使用代理模型。此外，其他指标的适用范围非常有限。在这项工作中，我们提出了一种固有的忠实性可度量模型来应对这些挑战。通过使用一种新颖的微调方法来实现这一目标，该方法将屏蔽令牌作为设计使其成为分布内。这与现有方法不同，现有方法完全与模型无关，但在实践中不适用。我们通过将其应用于各种任务和数据集来证明我们方法的普适性。

    A common approach to explain NLP models, is to use importance measures that express which tokens are important for a prediction. Unfortunately, such explanations are often wrong despite being persuasive. Therefore, it is essential to measure their faithfulness. One such metric is if tokens are truly important, then masking them should result in worse model performance. However, token masking introduces out-of-distribution issues and existing solutions are computationally expensive and employ proxy-models. Furthermore, other metrics are very limited in scope. In this work, we propose an inherently faithfulness measurable model that addresses these challenges. This is achieved by using a novel fine-tuning method that incorporates masking, such that masking tokens become in-distribution by design. This differs from existing approaches, which are completely model-agnostic but are inapplicable in practice. We demonstrate the generality of our approach by applying it to various tasks and val
    
[^11]: 使用大型语言模型生成、验证和应用用户意图分类方法

    Using Large Language Models to Generate, Validate, and Apply User Intent Taxonomies. (arXiv:2309.13063v1 [cs.IR])

    [http://arxiv.org/abs/2309.13063](http://arxiv.org/abs/2309.13063)

    通过使用大型语言模型生成用户意图分类，我们提出了一种新方法来分析和验证日志数据中的用户意图，从而解决了手动或基于机器学习的标注方法在大型和不断变化的数据集上的问题。

    

    日志数据可以揭示用户与网络搜索服务的交互方式、用户的需求以及满意程度等宝贵信息。然而，分析日志数据中的用户意图并不容易，尤其是对于新的网络搜索形式，如人工智能驱动的聊天。为了理解日志数据中的用户意图，我们需要一种能够用有意义的分类方式标记它们的方法，以捕捉其多样性和动态性。现有的方法依赖于手动或基于机器学习的标注，这些方法对于大型且不断变化的数据集而言，要么代价高昂要么不够灵活。我们提出了一种使用大型语言模型(LLM)的新方法，这种模型能够生成丰富且相关的概念、描述和示例来表示用户意图。然而，使用LLM生成用户意图分类并将其应用于日志分析可能存在两个主要问题：这样的分类得不到外部验证，并且可能存在不良的反馈回路。为了克服这些问题，我们提出了一种新的方法，通过人工专家和评估者来验证。

    Log data can reveal valuable information about how users interact with web search services, what they want, and how satisfied they are. However, analyzing user intents in log data is not easy, especially for new forms of web search such as AI-driven chat. To understand user intents from log data, we need a way to label them with meaningful categories that capture their diversity and dynamics. Existing methods rely on manual or ML-based labeling, which are either expensive or inflexible for large and changing datasets. We propose a novel solution using large language models (LLMs), which can generate rich and relevant concepts, descriptions, and examples for user intents. However, using LLMs to generate a user intent taxonomy and apply it to do log analysis can be problematic for two main reasons: such a taxonomy is not externally validated, and there may be an undesirable feedback loop. To overcome these issues, we propose a new methodology with human experts and assessors to verify th
    
[^12]: 正则表达式推理挑战

    The Regular Expression Inference Challenge. (arXiv:2308.07899v1 [cs.LG])

    [http://arxiv.org/abs/2308.07899](http://arxiv.org/abs/2308.07899)

    正则表达式推理挑战是一个以找到最小正则表达式为目标的任务，具有应用广泛、难度可调、适用于代码/语言建模和机器学习的特点。

    

    我们提出将正则表达式推理（REI）作为代码/语言建模以及更广泛的机器学习社区的挑战。REI是一个有监督的机器学习和程序合成任务，它提出了从示例中找到最小正则表达式的问题：给定两个有限字符串集合P和N以及一个成本函数cost(·)，任务是生成一个接受P中所有字符串并拒绝N中所有字符串的表达式r，而不存在其他表达式r'，使得cost(r')<cost(r)。REI作为一个挑战问题具有以下优势：（i）正则表达式是众所周知、广泛使用的，是代码的自然理想化；（ii）REI的渐近最坏情况复杂性已被充分理解；（iii）REI具有一小部分易于理解的参数（例如P或N的基数、示例的字符串长度或成本函数），这使得我们可以轻松调整REI的难度；（iv）对于基于深度学习的M模型而言，REI是一个未解决的问题。

    We propose \emph{regular expression inference (REI)} as a challenge for code/language modelling, and the wider machine learning community. REI is a supervised machine learning (ML) and program synthesis task, and poses the problem of finding minimal regular expressions from examples: Given two finite sets of strings $P$ and $N$ and a cost function $\text{cost}(\cdot)$, the task is to generate an expression $r$ that accepts all strings in $P$ and rejects all strings in $N$, while no other such expression $r'$ exists with $\text{cost}(r')<\text{cost}(r)$.  REI has advantages as a challenge problem: (i) regular expressions are well-known, widely used, and a natural idealisation of code; (ii) REI's asymptotic worst-case complexity is well understood; (iii) REI has a small number of easy to understand parameters (e.g.~$P$ or $N$ cardinality, string lengths of examples, or the cost function); this lets us easily finetune REI-hardness; (iv) REI is an unsolved problem for deep learning based M
    
[^13]: 深度学习在自动医疗编码中的应用综述

    A Unified Review of Deep Learning for Automated Medical Coding. (arXiv:2201.02797v3 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2201.02797](http://arxiv.org/abs/2201.02797)

    本文综述了深度学习在自动医疗编码领域的发展，提出了一个统一框架，总结了最新的高级模型，并讨论了未来发展的挑战和方向。

    

    自动医疗编码是医疗运营和服务的基本任务，通过从临床文档中预测医疗编码来管理非结构化数据。近年来，深度学习和自然语言处理的进步已广泛应用于该任务。但基于深度学习的自动医疗编码缺乏对神经网络架构设计的统一视图。本综述提出了一个统一框架，以提供对医疗编码模型组件的一般理解，并总结了在此框架下最近的高级模型。我们的统一框架将医疗编码分解为四个主要组件，即用于文本特征提取的编码器模块、构建深度编码器架构的机制、用于将隐藏表示转换成医疗代码的解码器模块以及辅助信息的使用。最后，我们介绍了基准和真实世界中的使用情况，讨论了关键的研究挑战和未来方向。

    Automated medical coding, an essential task for healthcare operation and delivery, makes unstructured data manageable by predicting medical codes from clinical documents. Recent advances in deep learning and natural language processing have been widely applied to this task. However, deep learning-based medical coding lacks a unified view of the design of neural network architectures. This review proposes a unified framework to provide a general understanding of the building blocks of medical coding models and summarizes recent advanced models under the proposed framework. Our unified framework decomposes medical coding into four main components, i.e., encoder modules for text feature extraction, mechanisms for building deep encoder architectures, decoder modules for transforming hidden representations into medical codes, and the usage of auxiliary information. Finally, we introduce the benchmarks and real-world usage and discuss key research challenges and future directions.
    

