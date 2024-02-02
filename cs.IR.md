# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Personalized Framework for Consumer and Producer Group Fairness Optimization in Recommender Systems](https://arxiv.org/abs/2402.00485) | 这篇论文提出了一个个性化推荐系统框架，通过优化算法实现了消费者和生产者两方的公平性约束。该框架具有可泛化性和灵活性，可以根据不同的群体分割、推荐模型选择和领域设置实现公平性优化。 |
| [^2] | [From PARIS to LE-PARIS: Toward Patent Response Automation with Recommender Systems and Collaborative Large Language Models](https://arxiv.org/abs/2402.00421) | 本研究介绍了专利响应智能系统PARIS和LE-PARIS，通过构建OA主题数据库、开发响应模板以及实施推荐系统和基于LLM的响应生成，旨在加快专利律师处理审查意见回应的效率。 通过多范式分析和长期数据验证，证明了OA主题的建设性和LLM对于回应自动生成的可行性。 |
| [^3] | [EASRec: Elastic Architecture Search for Efficient Long-term Sequential Recommender Systems](https://arxiv.org/abs/2402.00390) | EASRec是一个针对顺序推荐系统的弹性架构搜索方法，通过自动剪枝技术和先进模型架构结合，以及资源受限神经架构搜索技术，实现了降低计算成本和资源消耗的同时保持或增强准确性。 |
| [^4] | [An Exam-based Evaluation Approach Beyond Traditional Relevance Judgments](https://arxiv.org/abs/2402.00309) | 该论文提出了一种超越传统相关性判断的基于考试的评估方法，不依赖相关性判断，而是根据文本是否包含能回答关键问题的信息来判断相关性。通过设计EXAM可回答度指标和两种评估措施，可以评估信息检索/生成系统的主题相关信息提供能力。 |
| [^5] | [PAP-REC: Personalized Automatic Prompt for Recommendation Language Model](https://arxiv.org/abs/2402.00284) | 本研究提出了PAP-REC框架，用于生成个性化自动提示的推荐语言模型。该框架通过自动生成个性化提示标记来减轻手动设计提示所带来的效率和效果问题。 |
| [^6] | [Pareto-based Multi-Objective Recommender System with Forgetting Curve](https://arxiv.org/abs/2312.16868) | 基于Pareto的带有遗忘曲线的多目标推荐系统（PMORS）通过引入遗忘模型和Pareto优化求解器，能够处理明确的负面反馈并在多目标推荐中表现出优越性能。 |
| [^7] | [InstructRetro: Instruction Tuning post Retrieval-Augmented Pretraining](https://arxiv.org/abs/2310.07713) | InstructRetro是目前规模最大的使用检索预训练的LLM，扩展了基础模型Retro 48B，通过指令调优在各种零样例任务上取得显著改进。 |
| [^8] | [Zero-shot Generative Large Language Models for Systematic Review Screening Automation.](http://arxiv.org/abs/2401.06320) | 本研究调查了使用零样本生成式大型语言模型进行系统性综述自动筛选的有效性，结果显示指导微调和校准技术在筛选中起到重要作用，并且与零样本模型的集成相结合可以显著节省筛选时间。 |
| [^9] | [A First Look at Information Highlighting in Stack Overflow Answers.](http://arxiv.org/abs/2401.01472) | 本论文进行了首次大规模的探索性研究，研究了Stack Overflow回答中的信息高亮。通过使用神经网络架构，开发了自动推荐突出内容的方法。 |
| [^10] | [Retrieval-Augmented Generative Agent for Reaction Condition Recommendation in Chemical Synthesis.](http://arxiv.org/abs/2311.10776) | 本研究提出了一种转变性的人工智能代理，利用检索增强生成（RAG）技术自动化化学中的反应条件推荐（RCR）任务，通过模拟专家化学家的策略，使用大型语言模型（LLM）和新反应指纹，显著优于传统人工智能。此系统可以减轻化学家的工作负担，使他们能够更专注于更基础和创造性的科学问题。 |
| [^11] | [Using Large Language Models to Generate, Validate, and Apply User Intent Taxonomies.](http://arxiv.org/abs/2309.13063) | 通过使用大型语言模型生成用户意图分类，我们提出了一种新方法来分析和验证日志数据中的用户意图，从而解决了手动或基于机器学习的标注方法在大型和不断变化的数据集上的问题。 |

# 详细

[^1]: 个性化的消费者和生产者群体公平优化的推荐系统框架

    A Personalized Framework for Consumer and Producer Group Fairness Optimization in Recommender Systems

    [https://arxiv.org/abs/2402.00485](https://arxiv.org/abs/2402.00485)

    这篇论文提出了一个个性化推荐系统框架，通过优化算法实现了消费者和生产者两方的公平性约束。该框架具有可泛化性和灵活性，可以根据不同的群体分割、推荐模型选择和领域设置实现公平性优化。

    

    近年来，人们越来越意识到，当机器学习算法用于自动化决策时，可能会对个人或群体进行不公平待遇，从而产生法律、伦理或经济方面的影响。推荐系统是这些机器学习系统的重要例子，它们帮助用户做出决策。过去大部分关于推荐系统公平性的文献研究都是将用户和物品的公平性问题独立对待，忽视了推荐系统在双边市场中的作用。在本文中，我们提出了CP-FairRank，这是一种基于优化的重新排名算法，在联合目标框架中无缝集成了消费者和生产者两方的公平性约束。该框架具有可泛化性，并可以根据群体分割、推荐模型选择和领域考虑多样的公平性设置，这是其一个重要特点。例如，我们展示了该系统可以同时提高消费者和生产者的满意度，并在实验中取得了良好的结果。

    In recent years, there has been an increasing recognition that when machine learning (ML) algorithms are used to automate decisions, they may mistreat individuals or groups, with legal, ethical, or economic implications. Recommender systems are prominent examples of these machine learning (ML) systems that aid users in making decisions. The majority of past literature research on RS fairness treats user and item fairness concerns independently, ignoring the fact that recommender systems function in a two-sided marketplace. In this paper, we propose CP-FairRank, an optimization-based re-ranking algorithm that seamlessly integrates fairness constraints from both the consumer and producer side in a joint objective framework. The framework is generalizable and may take into account varied fairness settings based on group segmentation, recommendation model selection, and domain, which is one of its key characteristics. For instance, we demonstrate that the system may jointly increase consum
    
[^2]: 从PARIS到LE-PARIS：通过推荐系统和协作大型语言模型实现专利响应自动化

    From PARIS to LE-PARIS: Toward Patent Response Automation with Recommender Systems and Collaborative Large Language Models

    [https://arxiv.org/abs/2402.00421](https://arxiv.org/abs/2402.00421)

    本研究介绍了专利响应智能系统PARIS和LE-PARIS，通过构建OA主题数据库、开发响应模板以及实施推荐系统和基于LLM的响应生成，旨在加快专利律师处理审查意见回应的效率。 通过多范式分析和长期数据验证，证明了OA主题的建设性和LLM对于回应自动生成的可行性。

    

    在专利审查中，对于及时和有效地回应审查意见（OAs）对于获得专利至关重要，然而过去的自动化和人工智能研究很少涉及到这一方面。为了弥补这一空白，我们的研究介绍了专利审查意见响应智能系统（PARIS）及其先进版本LE-PARIS。这些系统旨在加快专利律师在协作处理OA回应方面的效率。系统的关键特征包括构建OA主题数据库，开发响应模板，以及实施推荐系统和基于LLM的响应生成。我们的验证涉及使用USPTO Office Action数据库和律师与我们系统的长期交互数据进行的多范式分析，为期六年。通过五个研究，我们利用主题建模和提出的Delphi过程来检验OA主题的建设性（研究1和2），还有使用推荐系统和基于LLM的响应生成来提高回应质量（研究3和4），以及经过训练的LLM对于回应自动生成的可行性（研究5）。

    In patent prosecution, timely and effective responses to Office Actions (OAs) are crucial for acquiring patents, yet past automation and AI research have scarcely addressed this aspect. To address this gap, our study introduces the Patent Office Action Response Intelligence System (PARIS) and its advanced version, the Large Language Model Enhanced PARIS (LE-PARIS). These systems are designed to expedite the efficiency of patent attorneys in collaboratively handling OA responses. The systems' key features include the construction of an OA Topics Database, development of Response Templates, and implementation of Recommender Systems and LLM-based Response Generation. Our validation involves a multi-paradigmatic analysis using the USPTO Office Action database and longitudinal data of attorney interactions with our systems over six years. Through five studies, we examine the constructiveness of OA topics (studies 1 and 2) using topic modeling and the proposed Delphi process, the efficacy of
    
[^3]: EASRec：用于高效长期顺序推荐系统的弹性架构搜索

    EASRec: Elastic Architecture Search for Efficient Long-term Sequential Recommender Systems

    [https://arxiv.org/abs/2402.00390](https://arxiv.org/abs/2402.00390)

    EASRec是一个针对顺序推荐系统的弹性架构搜索方法，通过自动剪枝技术和先进模型架构结合，以及资源受限神经架构搜索技术，实现了降低计算成本和资源消耗的同时保持或增强准确性。

    

    在数据丰富的时代，从海量信息中提取有意义的见解的能力至关重要。我们的研究解决了当前顺序推荐系统（SRSs）在计算和资源效率方面存在的问题，特别是那些采用了基于注意力模型（如SASRec）的系统。这些系统旨在为各种应用提供下一个项目的推荐，从电子商务到社交网络。然而，这些系统在推理阶段会产生相当大的计算成本和资源消耗。为了解决这些问题，我们的研究提出了一种结合自动剪枝技术和先进模型架构的新方法。我们还探索了在推荐系统领域中流行的资源受限神经架构搜索（NAS）技术的潜力，以调整模型以减少FLOPs、延迟和能量使用，同时保持或增强准确性。我们的工作的主要贡献是开发了一种

    In this age where data is abundant, the ability to distill meaningful insights from the sea of information is essential. Our research addresses the computational and resource inefficiencies that current Sequential Recommender Systems (SRSs) suffer from. especially those employing attention-based models like SASRec, These systems are designed for next-item recommendations in various applications, from e-commerce to social networks. However, such systems suffer from substantial computational costs and resource consumption during the inference stage. To tackle these issues, our research proposes a novel method that combines automatic pruning techniques with advanced model architectures. We also explore the potential of resource-constrained Neural Architecture Search (NAS), a technique prevalent in the realm of recommendation systems, to fine-tune models for reduced FLOPs, latency, and energy usage while retaining or even enhancing accuracy. The main contribution of our work is developing 
    
[^4]: 超越传统相关性判断的基于考试的评估方法

    An Exam-based Evaluation Approach Beyond Traditional Relevance Judgments

    [https://arxiv.org/abs/2402.00309](https://arxiv.org/abs/2402.00309)

    该论文提出了一种超越传统相关性判断的基于考试的评估方法，不依赖相关性判断，而是根据文本是否包含能回答关键问题的信息来判断相关性。通过设计EXAM可回答度指标和两种评估措施，可以评估信息检索/生成系统的主题相关信息提供能力。

    

    当前的信息检索评估基于相关性判断，这些判断可以手动或自动创建，并且决策通常被外包给大型语言模型（LLMs）。我们提供了一种替代范式，从不依赖任何形式的相关性判断。相反，如果一段文本包含可以回答关键问题的信息，我们将其定义为相关性。我们利用这个思想设计了EXAM可回答度指标，以评估信息检索/生成系统提供主题相关信息的能力。我们设想一个人类评委的角色是编辑和定义一个考试题库，用于测试文本中相关信息的存在。我们通过生成一个初始的考试题目集来支持这一步骤。在下一个阶段，基于LLM的问答系统将通过跟踪可以回答哪个考试题目来自动评分系统的答案。我们提出了两种评估指标：回忆导向的EXAM覆盖度指标和

    Current IR evaluation is based on relevance judgments, created either manually or automatically, with decisions outsourced to Large Language Models (LLMs). We offer an alternative paradigm, that never relies on relevance judgments in any form. Instead, a text is defined as relevant if it contains information that enables the answering of key questions. We use this idea to design the EXAM Answerability Metric to evaluate information retrieval/generation systems for their ability to provide topically relevant information.   We envision the role of a human judge to edit and define an exam question bank that will test for the presence of relevant information in text. We support this step by generating an initial set of exam questions. In the next phase, an LLM-based question answering system will automatically grade system responses by tracking which exam questions are answerable with which system responses. We propose two evaluation measures, the recall-oriented EXAM Cover metric, and the
    
[^5]: PAP-REC: 个性化自动提示的推荐语言模型

    PAP-REC: Personalized Automatic Prompt for Recommendation Language Model

    [https://arxiv.org/abs/2402.00284](https://arxiv.org/abs/2402.00284)

    本研究提出了PAP-REC框架，用于生成个性化自动提示的推荐语言模型。该框架通过自动生成个性化提示标记来减轻手动设计提示所带来的效率和效果问题。

    

    最近出现的基于提示的推荐语言模型（RLM）可以统一解决多个推荐任务。这些RLM充分利用了从丰富的预训练数据中学到的遗传知识，通过提示来解决下游推荐任务，而不需要引入额外的参数或网络训练。然而，手工设计的提示需要显著的专业知识和人力投入，稍微改写提示就可能导致性能的巨大变化。在本文中，我们提出了PAP-REC，一个用于生成个性化自动提示的推荐语言模型的框架，以缓解手动设计提示导致的低效率和低效果问题。具体而言，个性化自动提示允许不同的用户在相同任务中具有不同的提示标记，这些标记是使用梯度下降法自动生成的。个性化自动提示生成推荐语言模型的一个挑战是庞大的搜索空间。

    Recently emerged prompt-based Recommendation Language Models (RLM) can solve multiple recommendation tasks uniformly. The RLMs make full use of the inherited knowledge learned from the abundant pre-training data to solve the downstream recommendation tasks by prompts, without introducing additional parameters or network training. However, handcrafted prompts require significant expertise and human effort since slightly rewriting prompts may cause massive performance changes. In this paper, we propose PAP-REC, a framework to generate the Personalized Automatic Prompt for RECommendation language models to mitigate the inefficiency and ineffectiveness problems derived from manually designed prompts. Specifically, personalized automatic prompts allow different users to have different prompt tokens for the same task, automatically generated using a gradient-based method. One challenge for personalized automatic prompt generation for recommendation language models is the extremely large sear
    
[^6]: 基于Pareto的带有遗忘曲线的多目标推荐系统

    Pareto-based Multi-Objective Recommender System with Forgetting Curve

    [https://arxiv.org/abs/2312.16868](https://arxiv.org/abs/2312.16868)

    基于Pareto的带有遗忘曲线的多目标推荐系统（PMORS）通过引入遗忘模型和Pareto优化求解器，能够处理明确的负面反馈并在多目标推荐中表现出优越性能。

    

    带有级联架构的推荐系统在在线推荐平台中发挥着越来越重要的作用，在处理负面反馈的方法上是一个关键问题。例如，在短视频平台上，用户往往会快速地滑动删除他们不喜欢的视频，推荐系统需要接收这些明确的负面反馈并进行调整以避免这些推荐。考虑到记忆中的近期效应，我们提出了一种基于艾宾浩斯遗忘曲线的遗忘模型，用于处理负面反馈。另外，我们引入了一个Pareto优化求解器，以在近期性和模型性能之间取得更好的平衡。总结来说，我们提出了基于Pareto的带有遗忘曲线的多目标推荐系统（PMORS），可以应用于任何多目标推荐，并在面对明确的负面反馈时表现出足够的优势。我们对PMORS进行了评估，并取得了有利的结果。

    Recommender systems with cascading architecture play an increasingly significant role in online recommendation platforms, where the approach to dealing with negative feedback is a vital issue. For instance, in short video platforms, users tend to quickly slip away from candidates that they feel aversive, and recommender systems are expected to receive these explicit negative feedbacks and make adjustments to avoid these recommendations. Considering recency effect in memories, we propose a forgetting model based on Ebbinghaus Forgetting Curve to cope with negative feedback. In addition, we introduce a Pareto optimization solver to guarantee a better trade-off between recency and model performance. In conclusion, we propose Pareto-based Multi-Objective Recommender System with forgetting curve (PMORS), which can be applied to any multi-objective recommendation and show sufficiently superiority when facing explicit negative feedback. We have conducted evaluations of PMORS and achieved favo
    
[^7]: InstructRetro: 检索增强的预训练中指令调优

    InstructRetro: Instruction Tuning post Retrieval-Augmented Pretraining

    [https://arxiv.org/abs/2310.07713](https://arxiv.org/abs/2310.07713)

    InstructRetro是目前规模最大的使用检索预训练的LLM，扩展了基础模型Retro 48B，通过指令调优在各种零样例任务上取得显著改进。

    

    使用检索增强技术对自回归大型语言模型（LLM）进行预训练可以提高困惑度和事实准确性。然而，现有的预训练检索增强LLM的规模仍然有限（如Retro具有75亿个参数），这限制了指令调优和零样例泛化的效果。本文介绍了Retro 48B，这是目前规模最大的使用检索预训练的LLM。具体来说，我们使用检索技术从1.2万亿个标记中继续预训练一个43B的GPT模型，并借助Retro方法将其扩展到4800亿个参数。值得注意的是，所得到的基础模型Retro 48B在困惑度方面显著优于仅使用1.2万亿个标记进行训练的43B GPT模型，且只增加了2.58%的GPU使用时间，展示了该方法的显著扩展潜力。在对Retro进行指令调优后，InstructRetro在各种零样例任务上表现出显著的改进。

    Pretraining auto-regressive large language models (LLMs) with retrieval demonstrates better perplexity and factual accuracy by leveraging external databases. However, the size of existing pretrained retrieval-augmented LLM is still limited (e.g., Retro has 7.5B parameters), which limits the effectiveness of instruction tuning and zero-shot generalization. In this work, we introduce Retro 48B, the largest LLM pretrained with retrieval. Specifically, we continue to pretrain a 43B GPT model on additional 100 billion tokens using the Retro augmentation method by retrieving from 1.2 trillion tokens. Notably, the obtained foundation model, Retro 48B, largely outperforms the counterpart GPT 43B trained on 1.2T tokens in terms of perplexity with only 2.58% additional GPU hours, demonstrating the significant scaling potential of the method. After instruction tuning on Retro, InstructRetro demonstrates significant improvement over the instruction tuned GPT on a wide range of zero-shot tasks. Spe
    
[^8]: 零样本生成式大型语言模型用于系统性综述筛选自动化

    Zero-shot Generative Large Language Models for Systematic Review Screening Automation. (arXiv:2401.06320v1 [cs.IR])

    [http://arxiv.org/abs/2401.06320](http://arxiv.org/abs/2401.06320)

    本研究调查了使用零样本生成式大型语言模型进行系统性综述自动筛选的有效性，结果显示指导微调和校准技术在筛选中起到重要作用，并且与零样本模型的集成相结合可以显著节省筛选时间。

    

    系统性综述对于基于证据的医学非常重要，它们综合分析了特定问题的已发表研究结果。进行此类综述通常需要大量的资源和时间，特别是在筛选阶段，需要评估出版物摘要是否应包括在综述中。本研究调查了使用零样本大型语言模型（LLM）进行自动筛选的有效性。我们评估了八种不同的LLM的效果，并研究了一种使用预定义的召回阈值的校准技术，用于确定是否应将出版物包括在系统性综述中。我们的全面评估使用了五个标准测试集，结果显示指导微调在筛选中起到了重要作用，校准使LLMs在实现目标召回方面更实用，并且将这两者与零样本模型的集成相结合与现有技术相比节省了大量筛选时间。

    Systematic reviews are crucial for evidence-based medicine as they comprehensively analyse published research findings on specific questions. Conducting such reviews is often resource- and time-intensive, especially in the screening phase, where abstracts of publications are assessed for inclusion in a review. This study investigates the effectiveness of using zero-shot large language models~(LLMs) for automatic screening. We evaluate the effectiveness of eight different LLMs and investigate a calibration technique that uses a predefined recall threshold to determine whether a publication should be included in a systematic review. Our comprehensive evaluation using five standard test collections shows that instruction fine-tuning plays an important role in screening, that calibration renders LLMs practical for achieving a targeted recall, and that combining both with an ensemble of zero-shot models saves significant screening time compared to state-of-the-art approaches.
    
[^9]: Stack Overflow回答中信息高亮的初探

    A First Look at Information Highlighting in Stack Overflow Answers. (arXiv:2401.01472v1 [cs.CL])

    [http://arxiv.org/abs/2401.01472](http://arxiv.org/abs/2401.01472)

    本论文进行了首次大规模的探索性研究，研究了Stack Overflow回答中的信息高亮。通过使用神经网络架构，开发了自动推荐突出内容的方法。

    

    背景：浏览Stack Overflow（SO）的知识仍然具有挑战性。为了使帖子对用户更生动，SO允许用户使用Markdown或HTML编写和编辑帖子，以便用户可以利用各种格式化样式（例如粗体、斜体和代码）来突出重要信息。然而，关于突出信息的研究仍然有限。目标：我们在最近的研究中进行了首次大规模的探索性研究，研究了SO回答中的信息高亮。为了扩展我们之前的研究，我们利用最初设计用于命名实体识别任务的神经网络架构，开发了自动推荐带有格式化样式的突出内容的方法。方法：本文研究了Stack Overflow的31,169,429个回答。为了训练推荐模型，我们选择了CNN和BERT模型，针对每种格式化类型（即粗体、斜体、代码和标题）使用我们从SO回答收集的突出信息数据集。

    Context: Navigating the knowledge of Stack Overflow (SO) remains challenging. To make the posts vivid to users, SO allows users to write and edit posts with Markdown or HTML so that users can leverage various formatting styles (e.g., bold, italic, and code) to highlight the important information. Nonetheless, there have been limited studies on the highlighted information. Objective: We carried out the first large-scale exploratory study on the information highlighted in SO answers in our recent study. To extend our previous study, we develop approaches to automatically recommend highlighted content with formatting styles using neural network architectures initially designed for the Named Entity Recognition task. Method: In this paper, we studied 31,169,429 answers of Stack Overflow. For training recommendation models, we choose CNN and BERT models for each type of formatting (i.e., Bold, Italic, Code, and Heading) using the information highlighting dataset we collected from SO answers.
    
[^10]: 在化学合成中的反应条件推荐中，检索增强生成代理

    Retrieval-Augmented Generative Agent for Reaction Condition Recommendation in Chemical Synthesis. (arXiv:2311.10776v3 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2311.10776](http://arxiv.org/abs/2311.10776)

    本研究提出了一种转变性的人工智能代理，利用检索增强生成（RAG）技术自动化化学中的反应条件推荐（RCR）任务，通过模拟专家化学家的策略，使用大型语言模型（LLM）和新反应指纹，显著优于传统人工智能。此系统可以减轻化学家的工作负担，使他们能够更专注于更基础和创造性的科学问题。

    

    最近的人工智能研究为化学社会中的自动化化学反应铺平了一个有前途的未来。本研究提出了一种转变性的人工智能代理，利用检索增强生成（RAG）技术自动化化学中的反应条件推荐（RCR）任务。通过模拟专家化学家的搜索和分析策略，该代理使用大型语言模型（LLM）来查询分子数据库，并从在线文献中提取关键数据。此外，该人工智能代理还配备了我们为RCR任务开发的新反应指纹。由于RAG技术的使用，我们的代理使用更新的在线数据库作为知识源，显著优于仅受其训练数据固定知识限制的传统人工智能。由此产生的系统可以显著减轻化学家的工作负担，使他们能够更专注于更基础和创造性的科学问题。这一重大进展将计算技术与化学社会更紧密联系起来。

    Recent artificial intelligence (AI) research plots a promising future of automatic chemical reactions within the chemistry society. This study presents a transformative AI agent that automates the reaction condition recommendation (RCR) task in chemistry using retrieval-augmented generation (RAG) technology. By emulating expert chemists search and analysis strategies, the agent employs large language models (LLMs) to interrogate molecular databases and distill critical data from online literature. Further, the AI agent is equipped with our novel reaction fingerprint developed for the RCR task. Thanks to the RAG technology, our agent uses updated online databases as knowledge sources, significantly outperforming conventional AIs confined to the fixed knowledge within its training data. The resulting system can significantly reduce chemists workload, allowing them to focus on more fundamental and creative scientific problems. This significant advancement brings closer computational techn
    
[^11]: 使用大型语言模型生成、验证和应用用户意图分类方法

    Using Large Language Models to Generate, Validate, and Apply User Intent Taxonomies. (arXiv:2309.13063v1 [cs.IR])

    [http://arxiv.org/abs/2309.13063](http://arxiv.org/abs/2309.13063)

    通过使用大型语言模型生成用户意图分类，我们提出了一种新方法来分析和验证日志数据中的用户意图，从而解决了手动或基于机器学习的标注方法在大型和不断变化的数据集上的问题。

    

    日志数据可以揭示用户与网络搜索服务的交互方式、用户的需求以及满意程度等宝贵信息。然而，分析日志数据中的用户意图并不容易，尤其是对于新的网络搜索形式，如人工智能驱动的聊天。为了理解日志数据中的用户意图，我们需要一种能够用有意义的分类方式标记它们的方法，以捕捉其多样性和动态性。现有的方法依赖于手动或基于机器学习的标注，这些方法对于大型且不断变化的数据集而言，要么代价高昂要么不够灵活。我们提出了一种使用大型语言模型(LLM)的新方法，这种模型能够生成丰富且相关的概念、描述和示例来表示用户意图。然而，使用LLM生成用户意图分类并将其应用于日志分析可能存在两个主要问题：这样的分类得不到外部验证，并且可能存在不良的反馈回路。为了克服这些问题，我们提出了一种新的方法，通过人工专家和评估者来验证。

    Log data can reveal valuable information about how users interact with web search services, what they want, and how satisfied they are. However, analyzing user intents in log data is not easy, especially for new forms of web search such as AI-driven chat. To understand user intents from log data, we need a way to label them with meaningful categories that capture their diversity and dynamics. Existing methods rely on manual or ML-based labeling, which are either expensive or inflexible for large and changing datasets. We propose a novel solution using large language models (LLMs), which can generate rich and relevant concepts, descriptions, and examples for user intents. However, using LLMs to generate a user intent taxonomy and apply it to do log analysis can be problematic for two main reasons: such a taxonomy is not externally validated, and there may be an undesirable feedback loop. To overcome these issues, we propose a new methodology with human experts and assessors to verify th
    

