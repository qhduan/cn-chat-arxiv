# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [TrustAgent: Towards Safe and Trustworthy LLM-based Agents through Agent Constitution](https://rss.arxiv.org/abs/2402.01586) | 本文介绍了一种基于代理构成的代理框架TrustAgent，该框架通过预先规划、规划过程中和计划后检查三种策略来提高LLM代理的安全性。实验结果表明，这些方法可以有效识别和预防潜在危险。此外，还研究了安全性与使用者满意度以及模型推理能力与效率之间的关系。 |
| [^2] | [Can Language Models Recognize Convincing Arguments?](https://arxiv.org/abs/2404.00750) | 大语言模型不仅能够在识别和区分强势和弱势论点方面表现良好，还可以根据用户的信念和人口特征预测其立场，并确定论点对个人的吸引力。 |
| [^3] | [DeFT: Flash Tree-attention with IO-Awareness for Efficient Tree-search-based LLM Inference](https://arxiv.org/abs/2404.00242) | DeFT提出了一种带IO意识的树注意力算法，通过在QKV准备和注意力计算阶段实现内存高效的计算，降低内存印记，以解决当前树解码策略和推断系统不适配的问题。 |
| [^4] | [Is Factuality Decoding a Free Lunch for LLMs? Evaluation on Knowledge Editing Benchmark](https://arxiv.org/abs/2404.00216) | 大型语言模型通过事实解码方法提高了事实准确性，然而，这些方法使模型对已知事实过于自信，进一步评估显示在知识编辑基准上所有解码方法均显著降低了模型性能。 |
| [^5] | [LUQ: Long-text Uncertainty Quantification for LLMs](https://arxiv.org/abs/2403.20279) | LUQ提出了一种针对长文本设计的新型采样UQ方法，优于现有基准方法在与模型的事实得分相关方面。 |
| [^6] | [Cross-lingual Contextualized Phrase Retrieval](https://arxiv.org/abs/2403.16820) | 该研究提出了跨语言上下文化短语检索任务，并通过利用对比学习来解决多义性，从而增强了跨语言应用的性能。 |
| [^7] | [LARA: Linguistic-Adaptive Retrieval-Augmented LLMs for Multi-Turn Intent Classification](https://arxiv.org/abs/2403.16504) | LARA是一个Linguistic-Adaptive Retrieval-Augmented Language Models（语言自适应检索增强LLMs），旨在通过结合微调过的较小模型与检索增强机制来提高多语言多轮意图分类任务的准确性，从而改善对话背景的理解。 |
| [^8] | [Exploiting Semantic Reconstruction to Mitigate Hallucinations in Vision-Language Models](https://arxiv.org/abs/2403.16167) | 通过准确定位和惩罚幻觉标记，ESREAL引入了一种新颖的无监督学习框架，通过语义重建来抑制生成幻觉，解决了视觉-语言模型中幻觉问题。 |
| [^9] | [To Err Is Human, but Llamas Can Learn It Too](https://arxiv.org/abs/2403.05493) | 通过人工错误生成来提高语法错误纠正，进而在多种语言中取得优越的表现。 |
| [^10] | [Is this the real life? Is this just fantasy? The Misleading Success of Simulating Social Interactions With LLMs](https://arxiv.org/abs/2403.05020) | 研究发现，使用LLMs进行社交互动的全知模拟比非全知模拟更容易实现社交目标，尽管非全知模拟更接近实际情况。 |
| [^11] | [Aligners: Decoupling LLMs and Alignment](https://arxiv.org/abs/2403.04224) | 提出了一种通过训练对齐器模型来解耦大型语言模型（LLMs）和对齐，以减少对齐对性能的潜在负面影响。 |
| [^12] | [Cognitive Bias in High-Stakes Decision-Making with LLMs](https://arxiv.org/abs/2403.00811) | 提出了BiasBuster框架，用于揭示、评估和减轻LLMs中的认知偏见，特别是在高风险决策任务中，通过开发包含16,800个提示的数据集和测试多种偏见缓解策略，并提出一种利用LLMs自身来消除其提示中偏见的新方法。 |
| [^13] | ["Flex Tape Can't Fix That": Bias and Misinformation in Edited Language Models](https://arxiv.org/abs/2403.00180) | 该研究调查了编辑语言模型中偏见放大的问题，引入了一个新的基准数据集Seesaw-CF，首次深入研究了权重编辑方法对模型偏见的影响。 |
| [^14] | [Latent Attention for Linear Time Transformers](https://arxiv.org/abs/2402.17512) | 提出了一种基于潜在向量定义注意力的方法，将标准transformer中的注意力机制的时间复杂度从二次方降低到与时间线性相关，表现与标准注意力媲美，但允许上下文窗口扩展到远远超出标准的范围。 |
| [^15] | [Finer: Investigating and Enhancing Fine-Grained Visual Concept Recognition in Large Vision Language Models](https://arxiv.org/abs/2402.16315) | Finer工作揭示了大型视觉语言模型在细粒度视觉分类上的短板，尤其是难以生成准确的细致属性解释，尽管具有生成高水平图像解释的能力。 |
| [^16] | [GreenLLaMA: A Framework for Detoxification with Explanations](https://arxiv.org/abs/2402.15951) | GreenLLaMA是一种全面的端到端解毒框架，通过跨平台语料库训练出的模型优于当前最先进的模型。 |
| [^17] | [MultiContrievers: Analysis of Dense Retrieval Representations](https://arxiv.org/abs/2402.15925) | 该论文对稠密检索器的信息捕获进行了分析，探讨了其与语言模型的比较、信息提取的可行性以及提取性与性能、性别偏见的关系。 |
| [^18] | [CommVQA: Situating Visual Question Answering in Communicative Contexts](https://arxiv.org/abs/2402.15002) | CommVQA数据集将图像置于自然环境中，挑战了当前的VQA模型，结果表明为模型提供上下文信息能够提高性能。 |
| [^19] | [Divide-or-Conquer? Which Part Should You Distill Your LLM?](https://arxiv.org/abs/2402.15000) | 本文提出了一种将推理任务分解为问题分解阶段和问题解决阶段的策略，发现问题分解阶段相比问题解决更容易提炼为较小模型，并证实该策略胜过单阶段解决方案。 |
| [^20] | [Middleware for LLMs: Tools Are Instrumental for Language Agents in Complex Environments](https://arxiv.org/abs/2402.14672) | 这项研究探索了在复杂环境中利用工具增强大型语言模型的潜力，设计了定制化工具来辅助语言代理在庞大环境中进行探索，并展示了在知识库和数据库等复杂环境中，借助工具增强语言代理的重要潜力。 |
| [^21] | [LexC-Gen: Generating Data for Extremely Low-Resource Languages with Large Language Models and Bilingual Lexicons](https://arxiv.org/abs/2402.14086) | LexC-Gen提出了一种词典条件数据生成方法，可以以大规模生成低资源语言分类任务数据，取得了较好的效果。 |
| [^22] | [Softmax Probabilities (Mostly) Predict Large Language Model Correctness on Multiple-Choice Q&A](https://arxiv.org/abs/2402.13213) | 多项选择问答任务中，基于最大softmax概率（MSPs）的模型预测方法有助于提高大型语言模型（LLMs）的正确性，我们提出了一种根据MSP有选择地弃权的策略以提高性能。 |
| [^23] | [Are ELECTRA's Sentence Embeddings Beyond Repair? The Case of Semantic Textual Similarity](https://arxiv.org/abs/2402.13130) | 该研究探索了ELECTRA句子嵌入向量性能问题，并提出了一种新的截断模型微调方法，显著提高了语义文本相似性任务的表现 |
| [^24] | [Identifying Factual Inconsistency in Summaries: Towards Effective Utilization of Large Language Model](https://arxiv.org/abs/2402.12821) | 该研究提出了针对摘要中事实不一致性的解决方案：通过大型语言模型在正确的范式设计下无需训练即可解决任务，并提出了训练策略以精炼更小型的高准确性的语言模型。 |
| [^25] | [Standardize: Aligning Language Models with Expert-Defined Standards for Content Generation](https://arxiv.org/abs/2402.12593) | 该研究引入了一个名为Standardize的框架，通过检索式上下文学习指导大型语言模型与专家定义的标准对齐，提高了内容生成的精确性。 |
| [^26] | [AnaloBench: Benchmarking the Identification of Abstract and Long-context Analogies](https://arxiv.org/abs/2402.12370) | 通过提出ANALOBENCH基准来评估语言模型（LMs）进行类比推理的能力，发现扩展LMs规模对于处理涉及长场景或相关经验回忆的类比时带来的性能提升较小。 |
| [^27] | [KARL: Knowledge-Aware Retrieval and Representations aid Retention and Learning in Students](https://arxiv.org/abs/2402.12291) | KARL是一种基于DKT的学生模型，利用检索和BERT嵌入来实现高效准确的学生记忆预测，在AUC和校准误差方面优于现有学生模型，并提出了新颖的教学策略。 |
| [^28] | [I Learn Better If You Speak My Language: Enhancing Large Language Model Fine-Tuning with Style-Aligned Response Adjustments](https://arxiv.org/abs/2402.11192) | 将微调过程中的实际响应风格与大型语言模型固有风格相匹配能够产生更好的学习结果，开发的方法通过最小程度地调整模型响应来避免过拟合。 |
| [^29] | [Retrieval-Augmented Generation: Is Dense Passage Retrieval Retrieving?](https://arxiv.org/abs/2402.11035) | DPR微调预训练网络以增强查询和相关文本数据之间的嵌入对齐，发现训练中知识去中心化，但也揭示了模型内部知识的局限性 |
| [^30] | [Self-consistent context aware conformer transducer for speech recognition](https://arxiv.org/abs/2402.06592) | 这项研究提出了一种自洽的上下文感知转录器模型，能够在语音识别中提高不常见单词的准确性，而不影响常见单词的错误率。 |
| [^31] | [Do We Need Language-Specific Fact-Checking Models? The Case of Chinese](https://arxiv.org/abs/2401.15498) | 本文研究了语言特定事实核查模型的潜在益处，提出了一个汉语事实核查系统，并展示其优于翻译方法和多语言大型语言模型，同时对偏见更加稳健，强调了语言特定性的重要性。 |
| [^32] | [EHRAgent: Code Empowers Large Language Models for Few-shot Complex Tabular Reasoning on Electronic Health Records](https://arxiv.org/abs/2401.07128) | EHRAgent是一个由代码接口赋能的大型语言模型代理，用于自主生成和执行多表格推理代码，通过错误信息学习改进生成的代码，结合长期记忆选择并建立在过去经验中的成功案例。 |
| [^33] | [Unlocking Anticipatory Text Generation: A Constrained Approach for Large Language Models Decoding](https://arxiv.org/abs/2312.06149) | 提出了将文本生成形式化为未来受限生成问题的方法，以最小化不良行为并强制执行对指令的忠实性，并通过LLMs有效指导文本生成。 |
| [^34] | [AMRFact: Enhancing Summarization Factuality Evaluation with AMR-Driven Negative Samples Generation](https://arxiv.org/abs/2311.09521) | AMRFact是一个框架，利用AMR生成负样本，增强了摘要事实性评估，生成的连贯且事实不一致的摘要具有高错误率。 |
| [^35] | [In Search of the Long-Tail: Systematic Generation of Long-Tail Inferential Knowledge via Logical Rule Guided Search](https://arxiv.org/abs/2311.07237) | 该研究提出了一个名为LINK的框架，能够系统性地生成长尾推理知识，从而更有效地评估LLMs在推理空间中的表现。 |
| [^36] | [SLANG: New Concept Comprehension of Large Language Models.](http://arxiv.org/abs/2401.12585) | 本研究提出了一个新的基准SLANG，旨在增强大型语言模型LLMs对互联网上新概念的理解能力，同时提出了一种基于因果推断的基准方法FOCUS，能帮助LLMs更好地理解新的短语和用法模式。 |
| [^37] | [Data Augmentation for Code Translation with Comparable Corpora and Multiple References.](http://arxiv.org/abs/2311.00317) | 该论文介绍了两种数据增强方法来改善编程语言之间的代码翻译。通过构建可比较的语料库和增加多个参考翻译，实验结果表明这些方法显著提高了CodeT5在Java、Python和C++之间的翻译准确性。 |
| [^38] | [Learning Personalized Story Evaluation.](http://arxiv.org/abs/2310.03304) | 该论文提出了学习个性化故事评估的方法。为了解决大型语言模型在开放式文本生成任务的评估问题，论文创建了两个新的数据集，并开发了一个个性化故事评估模型，能够根据评审人员的示例评价进行个性化评估。 |
| [^39] | [ChatGPT an ENFJ, Bard an ISTJ: Empirical Study on Personalities of Large Language Models.](http://arxiv.org/abs/2305.19926) | 本研究通过采用特质理论框架，实验证明了ChatGPT始终表现出ENFJ型人格，无论指令或情境如何。研究揭示了LLMs的个性化，有助于促进人与机器之间更好的沟通和协作。 |
| [^40] | [SMILE: Single-turn to Multi-turn Inclusive Language Expansion via ChatGPT for Mental Health Support.](http://arxiv.org/abs/2305.00450) | 本研究提出了SMILE方法，使用ChatGPT将公共单轮对话扩展为多轮对话，生成了大规模、多样化、接近真实生活的多轮心理健康支持对话语料库，可用于训练和评估专门的对话系统。 |

# 详细

[^1]: TrustAgent: 通过代理构成实现安全可信赖的LLM代理

    TrustAgent: Towards Safe and Trustworthy LLM-based Agents through Agent Constitution

    [https://rss.arxiv.org/abs/2402.01586](https://rss.arxiv.org/abs/2402.01586)

    本文介绍了一种基于代理构成的代理框架TrustAgent，该框架通过预先规划、规划过程中和计划后检查三种策略来提高LLM代理的安全性。实验结果表明，这些方法可以有效识别和预防潜在危险。此外，还研究了安全性与使用者满意度以及模型推理能力与效率之间的关系。

    

    近年来，基于LLM的代理引起了广泛关注，但其可信度仍未得到深入探索。由于代理可以直接与物理环境交互，其可靠性和安全性至关重要。本文提出了一种基于代理构成的代理框架TrustAgent，对LLM代理的安全性维度进行了初步研究。该框架包括三种策略：预先规划策略，在生成计划之前向模型注入安全知识；规划过程中策略，在生成计划时增强安全性；计划后检查策略，通过计划后检查确保安全性。通过实验分析，我们展示了这些方法如何通过识别和预防潜在危险有效提高LLM代理的安全性。此外，我们还探讨了安全性与使用者满意度之间的复杂关系，以及模型的推理能力与其效率之间的关联。

    The emergence of LLM-based agents has garnered considerable attention, yet their trustworthiness remains an under-explored area. As agents can directly interact with the physical environment, their reliability and safety is critical. This paper presents an Agent-Constitution-based agent framework, TrustAgent, an initial investigation into improving the safety dimension of trustworthiness in LLM-based agents. This framework consists of threefold strategies: pre-planning strategy which injects safety knowledge to the model prior to plan generation, in-planning strategy which bolsters safety during plan generation, and post-planning strategy which ensures safety by post-planning inspection. Through experimental analysis, we demonstrate how these approaches can effectively elevate an LLM agent's safety by identifying and preventing potential dangers. Furthermore, we explore the intricate relationships between safety and helpfulness, and between the model's reasoning ability and its efficac
    
[^2]: 大语言模型能识别令人信服的论点吗？

    Can Language Models Recognize Convincing Arguments?

    [https://arxiv.org/abs/2404.00750](https://arxiv.org/abs/2404.00750)

    大语言模型不仅能够在识别和区分强势和弱势论点方面表现良好，还可以根据用户的信念和人口特征预测其立场，并确定论点对个人的吸引力。

    

    大型语言模型（LLMs）的显著且不断增强的能力引发了人们对它们可能被滥用用来创造个性化、令人信服的虚假信息和宣传的担忧。为了深入了解LLMs的说服能力，而又不直接与人类进行实验，我们提出研究它们在检测令人信服的论点任务上的表现。我们通过添加辩论、投票和用户特征来扩展了Durmus和Cardie（2018）的数据集，并提出了衡量LLMs能力的任务，包括（1）区分强势和弱势论点，（2）基于信念和人口特征预测立场，以及（3）根据个人特征确定对一个论点的吸引力。我们发现LLMs在这些任务中表现与人类不相上下，并且结合不同LLMs的预测可以获得显著的性能提升，甚至超过人类的表现。随文附带发布的数据和代码。

    arXiv:2404.00750v1 Announce Type: new  Abstract: The remarkable and ever-increasing capabilities of Large Language Models (LLMs) have raised concerns about their potential misuse for creating personalized, convincing misinformation and propaganda. To gain insights into LLMs' persuasive capabilities without directly engaging in experimentation with humans, we propose studying their performance on the related task of detecting convincing arguments. We extend a dataset by Durmus & Cardie (2018) with debates, votes, and user traits and propose tasks measuring LLMs' ability to (1) distinguish between strong and weak arguments, (2) predict stances based on beliefs and demographic characteristics, and (3) determine the appeal of an argument to an individual based on their traits. We show that LLMs perform on par with humans in these tasks and that combining predictions from different LLMs yields significant performance gains, even surpassing human performance. The data and code released with 
    
[^3]: DeFT：带IO意识的Flash Tree-attention用于高效的基于树搜索的LLM推断

    DeFT: Flash Tree-attention with IO-Awareness for Efficient Tree-search-based LLM Inference

    [https://arxiv.org/abs/2404.00242](https://arxiv.org/abs/2404.00242)

    DeFT提出了一种带IO意识的树注意力算法，通过在QKV准备和注意力计算阶段实现内存高效的计算，降低内存印记，以解决当前树解码策略和推断系统不适配的问题。

    

    使用树搜索进行解码可以极大地提高基于变压器的大型语言模型（LLMs）的推断质量。根据引导信号，它通过形成LLM输出从根到叶子的最佳路径来提高可控性、推理能力、对齐等。然而，由于计算冗余、内存占用和内存访问，当前的树解码策略及其推断系统互相不适配，导致推断效率低下。为解决这一问题，我们提出了DeFT，一种IO感知树注意力算法，它在两个阶段中保持内存高效的注意力计算，降低内存印记：（1）QKV准备：我们提出了一种KV引导树分裂策略，为GPU的高利用率和尽可能减少GPU全局内存和芯片上共享内存之间的KV缓存的内存读/写; （2）注意力计算...

    arXiv:2404.00242v1 Announce Type: cross  Abstract: Decoding using tree search can greatly enhance the inference quality for transformer-based Large Language Models (LLMs). Depending on the guidance signal, it searches for the best path from root to leaf in the tree by forming LLM outputs to improve controllability, reasoning ability, alignment, et cetera. However, current tree decoding strategies and their inference systems do not suit each other well due to redundancy in computation, memory footprints, and memory access, resulting in inefficient inference. To address this issue, we propose DeFT, an IO-aware tree attention algorithm that maintains memory-efficient attention calculation with low memory footprints in two stages: (1) QKV Preparation: we propose a KV-Guided Tree Split strategy to group QKV wisely for high utilization of GPUs and reduction of memory reads/writes for the KV cache between GPU global memory and on-chip shared memory as much as possible; (2) Attention Calculati
    
[^4]: 大型语言模型中的事实解码：在知识编辑基准上的评估

    Is Factuality Decoding a Free Lunch for LLMs? Evaluation on Knowledge Editing Benchmark

    [https://arxiv.org/abs/2404.00216](https://arxiv.org/abs/2404.00216)

    大型语言模型通过事实解码方法提高了事实准确性，然而，这些方法使模型对已知事实过于自信，进一步评估显示在知识编辑基准上所有解码方法均显著降低了模型性能。

    

    大型语言模型（LLMs）的快速发展使它们能够以更类似于人类的方式传达事实知识。人们已经做出了大量努力来通过修改LLMs并降低事实幻觉来提高事实准确性。然而，这些修改也存在阻碍知识更新的风险，因为它们使模型对已知事实过于自信。本文首先重新审视当前的事实解码方法，并验证了它们在提高事实准确性方面的有效性。随后，我们对几种强大的事实解码方法在知识编辑基准上进行进一步评估。所有这些解码方法与其原始解码相比均显着降低了llama2模型的性能，其中最大的降低幅度达到惊人的81.3\%。这进一步表明，当前的解码方法仍无法完全解决事实幻觉问题，因为它们忽视了先验知识的重要性。

    arXiv:2404.00216v1 Announce Type: cross  Abstract: The rapid development of large language models (LLMs) enables them to convey factual knowledge in a more human-like fashion. Extensive efforts have been made to reduce factual hallucinations by modifying LLMs with factuality decoding. However, they also pose risks of hindering knowledge updates, as they make models overly confident in known facts. In this work, we first revisite the current factuality decoding methods and verified their effectiveness in enhancing factual accuracy. Subsequently, we conduct further evaluation of several strong factuality decoding methods on the knowledge editing benchmark. All these decoding methods significantly diminish the performance of llama2 models compared to their original decoding, with the largest decrease being a staggering 81.3\%. This further indicates that the current existing decoding methods still cannot perfectly address the factual hallucinations, as they overlook the importance of pres
    
[^5]: LUQ：LLM模型的长文本不确定性量化

    LUQ: Long-text Uncertainty Quantification for LLMs

    [https://arxiv.org/abs/2403.20279](https://arxiv.org/abs/2403.20279)

    LUQ提出了一种针对长文本设计的新型采样UQ方法，优于现有基准方法在与模型的事实得分相关方面。

    

    大型语言模型（LLMs）在各种自然语言处理任务中展现出了显著的能力。尽管它们有效，但这些模型倾向于生成非事实内容。不确定性量化（UQ）对于增强我们对模型在生成内容上的信心至关重要，从而有助于减轻非事实输出。现有的UQ研究主要针对短文本生成，通常产生简短的、受词限制的响应。然而，现实世界中的应用往往需要更长的响应。我们的研究首先强调了当前UQ方法在处理长文本生成中的局限性。然后，我们介绍了一种名为\textsc{Luq}的新型基于抽样的UQ方法，专门设计用于长文本。我们的研究结果显示，\textsc{Luq}在与模型的事实得分相关方面优于现有的基准方法（Gemini Pro观察到-0.85的负相关系数）。

    arXiv:2403.20279v1 Announce Type: new  Abstract: Large Language Models (LLMs) have demonstrated remarkable capability in a variety of NLP tasks. Despite their effectiveness, these models are prone to generate nonfactual content. Uncertainty Quantification (UQ) is pivotal in enhancing our understanding of a model's confidence in its generated content, thereby aiding in the mitigation of nonfactual outputs. Existing research on UQ predominantly targets short text generation, typically yielding brief, word-limited responses. However, real-world applications frequently necessitate much longer responses. Our study first highlights the limitations of current UQ methods in handling long text generation. We then introduce \textsc{Luq}, a novel sampling-based UQ approach specifically designed for long text. Our findings reveal that \textsc{Luq} outperforms existing baseline methods in correlating with the model's factuality scores (negative coefficient of -0.85 observed for Gemini Pro). With \t
    
[^6]: 跨语言上下文化短语检索

    Cross-lingual Contextualized Phrase Retrieval

    [https://arxiv.org/abs/2403.16820](https://arxiv.org/abs/2403.16820)

    该研究提出了跨语言上下文化短语检索任务，并通过利用对比学习来解决多义性，从而增强了跨语言应用的性能。

    

    短语级密集检索通过利用短语提供的细粒度信息，在下游自然语言处理任务中展现出许多吸引人的特征。在我们的工作中，我们提出了一种新的密集检索任务形式，即跨语言上下文化短语检索，旨在通过使用上下文信息来增强解决多义性的跨语言应用。然而，缺乏特定的训练数据和模型是实现我们目标的主要挑战。因此，我们利用从平行句子中自动诱导的单词对齐信息提取跨语言短语对。随后，我们使用对比学习训练我们的跨语言上下文化短语检索器（CCPR），该对比学习鼓励具有相似上下文和语义的短语的隐藏表示紧密对齐。我们对跨语言短语检索任务和一个下游任务，即机器翻译，进行了全面的实验。

    arXiv:2403.16820v1 Announce Type: new  Abstract: Phrase-level dense retrieval has shown many appealing characteristics in downstream NLP tasks by leveraging the fine-grained information that phrases offer. In our work, we propose a new task formulation of dense retrieval, cross-lingual contextualized phrase retrieval, which aims to augment cross-lingual applications by addressing polysemy using context information. However, the lack of specific training data and models are the primary challenges to achieve our goal. As a result, we extract pairs of cross-lingual phrases using word alignment information automatically induced from parallel sentences. Subsequently, we train our Cross-lingual Contextualized Phrase Retriever (CCPR) using contrastive learning, which encourages the hidden representations of phrases with similar contexts and semantics to align closely. Comprehensive experiments on both the cross-lingual phrase retrieval task and a downstream task, i.e, machine translation, dem
    
[^7]: LARA：语言自适应检索增强LLMs用于多轮意图分类

    LARA: Linguistic-Adaptive Retrieval-Augmented LLMs for Multi-Turn Intent Classification

    [https://arxiv.org/abs/2403.16504](https://arxiv.org/abs/2403.16504)

    LARA是一个Linguistic-Adaptive Retrieval-Augmented Language Models（语言自适应检索增强LLMs），旨在通过结合微调过的较小模型与检索增强机制来提高多语言多轮意图分类任务的准确性，从而改善对话背景的理解。

    

    鉴于大型语言模型(LLMs)取得的显著成就，研究人员已经在文本分类任务中采用了上下文学习。然而，这些研究侧重于单语言、单轮分类任务。本文介绍了LARA（Linguistic-Adaptive Retrieval-Augmented Language Models），旨在增强多语言多轮分类任务的准确性，以适应聊天机器人交互中的众多意图。由于会话背景的复杂性和不断发展的性质，多轮意图分类尤为具有挑战性。LARA通过将微调过的较小模型与检索增强机制结合，嵌入LLMs的架构中来解决这些问题。这种整合使LARA能够动态利用过去的对话和相关意图，从而提高对上下文的理解。此外，我们的自适应检索技术增强了跨语言的能力。

    arXiv:2403.16504v1 Announce Type: new  Abstract: Following the significant achievements of large language models (LLMs), researchers have employed in-context learning for text classification tasks. However, these studies focused on monolingual, single-turn classification tasks. In this paper, we introduce LARA (Linguistic-Adaptive Retrieval-Augmented Language Models), designed to enhance accuracy in multi-turn classification tasks across six languages, accommodating numerous intents in chatbot interactions. Multi-turn intent classification is notably challenging due to the complexity and evolving nature of conversational contexts. LARA tackles these issues by combining a fine-tuned smaller model with a retrieval-augmented mechanism, integrated within the architecture of LLMs. This integration allows LARA to dynamically utilize past dialogues and relevant intents, thereby improving the understanding of the context. Furthermore, our adaptive retrieval techniques bolster the cross-lingual
    
[^8]: 利用语义重建减少视觉-语言模型中的幻觉

    Exploiting Semantic Reconstruction to Mitigate Hallucinations in Vision-Language Models

    [https://arxiv.org/abs/2403.16167](https://arxiv.org/abs/2403.16167)

    通过准确定位和惩罚幻觉标记，ESREAL引入了一种新颖的无监督学习框架，通过语义重建来抑制生成幻觉，解决了视觉-语言模型中幻觉问题。

    

    视觉-语言模型中的幻觉对其可靠性构成重大挑战，特别是在生成长标题时。当前方法无法准确识别和减轻这些幻觉。为了解决这个问题，我们引入了ESREAL，这是一个新颖的无监督学习框架，旨在通过准确定位和惩罚幻觉标记来抑制幻觉生成。最初，ESREAL根据生成的标题创建一个重建图像，并将其对应区域与原始图像的区域对齐。这种语义重建有助于识别生成标题中的标记级幻觉的存在和类型。随后，ESREAL通过评估对齐区域的语义相似性来计算标记级幻觉分数，基于幻觉的类型。最后，ESREAL采用一种近端策略优化算法，进行...

    arXiv:2403.16167v1 Announce Type: cross  Abstract: Hallucinations in vision-language models pose a significant challenge to their reliability, particularly in the generation of long captions. Current methods fall short of accurately identifying and mitigating these hallucinations. To address this issue, we introduce ESREAL, a novel unsupervised learning framework designed to suppress the generation of hallucinations through accurate localization and penalization of hallucinated tokens. Initially, ESREAL creates a reconstructed image based on the generated caption and aligns its corresponding regions with those of the original image. This semantic reconstruction aids in identifying both the presence and type of token-level hallucinations within the generated caption. Subsequently, ESREAL computes token-level hallucination scores by assessing the semantic similarity of aligned regions based on the type of hallucination. Finally, ESREAL employs a proximal policy optimization algorithm, wh
    
[^9]: 人类会犯错，但羊驼也能学会

    To Err Is Human, but Llamas Can Learn It Too

    [https://arxiv.org/abs/2403.05493](https://arxiv.org/abs/2403.05493)

    通过人工错误生成来提高语法错误纠正，进而在多种语言中取得优越的表现。

    

    本研究探讨了利用语言模型（LMs）通过人工错误生成（AEG）来增强语法错误纠正（GEC）。具体而言，我们对基于Llama 2的LMs进行微调以生成错误，并发现这种方法产生的合成错误类似于人类错误。接下来，我们利用这些人工错误训练GEC Llama模型，并在所有测试的语言（德语、乌克兰语和爱沙尼亚语）中取得了超过先前最先进的错误校正模型的表现，其收益在0.8至6 F0.5点之间。此外，我们证明通过微调较小的序列到序列模型和提示大型商用LMs（GPT-3.5和GPT-4）来生成错误，也会有益地影响错误生成模型的合成错误。

    arXiv:2403.05493v1 Announce Type: new  Abstract: This study explores enhancing grammatical error correction (GEC) through artificial error generation (AEG) using language models (LMs). Specifically, we fine-tune Llama 2-based LMs for error generation and find that this approach yields synthetic errors akin to human errors. Next, we train GEC Llama models with the help of these artificial errors and outperform previous state-of-the-art error correction models, with gains ranging between 0.8 and 6 F0.5 points across all tested languages (German, Ukrainian, and Estonian). Moreover, we demonstrate that generating errors by fine-tuning smaller sequence-to-sequence models and prompting large commercial LMs (GPT-3.5 and GPT-4) also results in synthetic errors beneficially affecting error generation models.
    
[^10]: 模拟社交互动成功性的误导性：以LLMs为例

    Is this the real life? Is this just fantasy? The Misleading Success of Simulating Social Interactions With LLMs

    [https://arxiv.org/abs/2403.05020](https://arxiv.org/abs/2403.05020)

    研究发现，使用LLMs进行社交互动的全知模拟比非全知模拟更容易实现社交目标，尽管非全知模拟更接近实际情况。

    

    最近大型语言模型（LLM）的进展使得社交模拟更加丰富，能够使用基于LLM的代理人研究各种社交现象。然而，大多数工作在这些模拟中采用了一种全知的透视（例如，单个LLM生成所有交谈者），这与人类具有的非全知、信息不对称的互动根本不符。为了研究这些差异，我们开发了一个评估框架，在各种设定（全知、非全知）中使用LLMs模拟社交互动。我们的实验表明，通过全知方式模拟的交谈者在实现社交目标方面比非全知代理人更成功，尽管后者更符合现实设置。此外，我们表明从全知模拟中学习可以改善交互的自然性，但在合作场景中几乎不能增强目标实现。

    arXiv:2403.05020v1 Announce Type: cross  Abstract: Recent advances in large language models (LLM) have enabled richer social simulations, allowing for the study of various social phenomena with LLM-based agents. However, most work has used an omniscient perspective on these simulations (e.g., single LLM to generate all interlocutors), which is fundamentally at odds with the non-omniscient, information asymmetric interactions that humans have. To examine these differences, we develop an evaluation framework to simulate social interactions with LLMs in various settings (omniscient, non-omniscient). Our experiments show that interlocutors simulated omnisciently are much more successful at accomplishing social goals compared to non-omniscient agents, despite the latter being the more realistic setting. Furthermore, we demonstrate that learning from omniscient simulations improves the apparent naturalness of interactions but scarcely enhances goal achievement in cooperative scenarios. Our f
    
[^11]: Aligners: 解耦LLMs和对齐

    Aligners: Decoupling LLMs and Alignment

    [https://arxiv.org/abs/2403.04224](https://arxiv.org/abs/2403.04224)

    提出了一种通过训练对齐器模型来解耦大型语言模型（LLMs）和对齐，以减少对齐对性能的潜在负面影响。

    

    大型语言模型（LLMs）需要与人类期望对齐，以确保它们在大多数应用中的安全性和实用性。对齐具有挑战性，成本高昂，并且需要为每个LLM和对齐标准重复进行。我们建议通过训练可以根据需要用于对齐给定标准的任何LLM的对齐模型来解耦LLMs和对齐，从而在一定程度上减少对性能的潜在负面影响。我们提出的对齐模型训练配方仅依赖于使用（提示的）LLM 生成的合成数据，并且可以轻松调整以适应各种对齐标准。我们通过训练一个“道德”对齐器并在实验上验证其有效性来阐明我们的方法。

    arXiv:2403.04224v1 Announce Type: cross  Abstract: Large Language Models (LLMs) need to be aligned with human expectations to ensure their safety and utility in most applications. Alignment is challenging, costly, and needs to be repeated for every LLM and alignment criterion. We propose to decouple LLMs and alignment by training aligner models that can be used to align any LLM for a given criteria on an as-needed basis, thus also reducing the potential negative impacts of alignment on performance. Our recipe for training the aligner models solely relies on synthetic data generated with a (prompted) LLM and can be easily adjusted for a variety of alignment criteria. We illustrate our method by training an "ethical" aligner and verify its efficacy empirically.
    
[^12]: LLM在高风险决策中的认知偏见

    Cognitive Bias in High-Stakes Decision-Making with LLMs

    [https://arxiv.org/abs/2403.00811](https://arxiv.org/abs/2403.00811)

    提出了BiasBuster框架，用于揭示、评估和减轻LLMs中的认知偏见，特别是在高风险决策任务中，通过开发包含16,800个提示的数据集和测试多种偏见缓解策略，并提出一种利用LLMs自身来消除其提示中偏见的新方法。

    

    大型语言模型(LLMs)在支持日益扩大的决策任务方面具有重要潜力。然而，由于它们在人类(创造的)数据上训练，LLMs可能会继承针对受保护群体的社会偏见，同时也可能受到认知偏见的影响。这种类似于人类的偏见可能会妨碍利用LLM协助做出公平和可解释的决策。我们的工作引入了BiasBuster，一个旨在揭示、评估和减轻LLMs中的认知偏见的框架，特别是在高风险决策任务中。受心理学和认知科学先前研究的启发，我们开发了一个包含16,800个提示的数据集，用于评估不同认知偏见(例如，提示诱导、顺序、固有)。我们测试了各种偏见缓解策略，同时提出了一种新方法，利用LLMs来消除它们自己的提示中的偏见。我们的分析提供了关于不同领域认知偏见存在和影响的全面图景。

    arXiv:2403.00811v1 Announce Type: new  Abstract: Large language models (LLMs) offer significant potential as tools to support an expanding range of decision-making tasks. However, given their training on human (created) data, LLMs can inherit both societal biases against protected groups, as well as be subject to cognitive bias. Such human-like bias can impede fair and explainable decisions made with LLM assistance. Our work introduces BiasBuster, a framework designed to uncover, evaluate, and mitigate cognitive bias in LLMs, particularly in high-stakes decision-making tasks. Inspired by prior research in psychology and cognitive sciences, we develop a dataset containing 16,800 prompts to evaluate different cognitive biases (e.g., prompt-induced, sequential, inherent). We test various bias mitigation strategies, amidst proposing a novel method using LLMs to debias their own prompts. Our analysis provides a comprehensive picture on the presence and effects of cognitive bias across diffe
    
[^13]: "Flex Tape不能修复这个": 编辑语言模型中的偏见和错误信息

    "Flex Tape Can't Fix That": Bias and Misinformation in Edited Language Models

    [https://arxiv.org/abs/2403.00180](https://arxiv.org/abs/2403.00180)

    该研究调查了编辑语言模型中偏见放大的问题，引入了一个新的基准数据集Seesaw-CF，首次深入研究了权重编辑方法对模型偏见的影响。

    

    模型编辑已经成为更新存储在语言模型中的知识的一种具有成本效益的策略。然而，在编辑应用后，模型编辑可能会产生意想不到的后果：与编辑无关的信息也可能被更改，并且模型的其他一般行为可能被错误地改变。在这项工作中，我们调查了模型编辑方法如何意外地加剧了模型后编辑的偏见。我们引入了一个新的基准数据集Seesaw-CF，用于衡量模型编辑的偏见相关伤害，并进行了首次深入研究不同权重编辑方法如何影响模型偏见。具体而言，我们专注于与种族、地理来源和性别等人口属性相关的偏见，以及由编辑语言模型生成的长文本中的定性缺陷。我们发现，编辑模型在变得对亚洲、非洲等属性的属性不确定度愈高时表现出不同程度的更为偏见行为。

    arXiv:2403.00180v1 Announce Type: new  Abstract: Model editing has emerged as a cost-effective strategy to update knowledge stored in language models. However, model editing can have unintended consequences after edits are applied: information unrelated to the edits can also be changed, and other general behaviors of the model can be wrongly altered. In this work, we investigate how model editing methods unexpectedly amplify model biases post-edit. We introduce a novel benchmark dataset, Seesaw-CF, for measuring bias-related harms of model editing and conduct the first in-depth investigation of how different weight-editing methods impact model bias. Specifically, we focus on biases with respect to demographic attributes such as race, geographic origin, and gender, as well as qualitative flaws in long-form texts generated by edited language models. We find that edited models exhibit, to various degrees, more biased behavior as they become less confident in attributes for Asian, African,
    
[^14]: Latent Attention for Linear Time Transformers

    Latent Attention for Linear Time Transformers

    [https://arxiv.org/abs/2402.17512](https://arxiv.org/abs/2402.17512)

    提出了一种基于潜在向量定义注意力的方法，将标准transformer中的注意力机制的时间复杂度从二次方降低到与时间线性相关，表现与标准注意力媲美，但允许上下文窗口扩展到远远超出标准的范围。

    

    标准transformer中的注意力机制的时间复杂度随着序列长度的增加呈二次方增长。我们引入一种通过定义潜在向量的注意力来将其降低到与时间线性相关的方法。该方法可以轻松作为标准注意力机制的替代品。我们的“Latte Transformer”模型可用于双向和单向任务，因果版本允许一种在推理语言生成任务中内存和时间高效的递归实现。标准transformer的下一个标记预测随着序列长度线性增长，而Latte Transformer计算下一个标记所需的时间是恒定的。我们的方法的实证表现可与标准注意力媲美，但允许将上下文窗口扩展到远远超出标准注意力实际可行的范围。

    arXiv:2402.17512v1 Announce Type: new  Abstract: The time complexity of the standard attention mechanism in a transformer scales quadratically with the length of the sequence. We introduce a method to reduce this to linear scaling with time, based on defining attention via latent vectors. The method is readily usable as a drop-in replacement for the standard attention mechanism. Our "Latte Transformer" model can be implemented for both bidirectional and unidirectional tasks, with the causal version allowing a recurrent implementation which is memory and time-efficient during inference of language generation tasks. Whilst next token prediction scales linearly with the sequence length for a standard transformer, a Latte Transformer requires constant time to compute the next token. The empirical performance of our method is comparable to standard attention, yet allows scaling to context windows much larger than practical in standard attention.
    
[^15]: Finer: 在大型视觉语言模型中研究和增强细粒度视觉概念识别

    Finer: Investigating and Enhancing Fine-Grained Visual Concept Recognition in Large Vision Language Models

    [https://arxiv.org/abs/2402.16315](https://arxiv.org/abs/2402.16315)

    Finer工作揭示了大型视觉语言模型在细粒度视觉分类上的短板，尤其是难以生成准确的细致属性解释，尽管具有生成高水平图像解释的能力。

    

    最近指导调整的大型视觉语言模型（LVLMs）的进展使模型能够轻松生成高水平的基于图像的解释。尽管这种能力主要归因于大型语言模型（LLMs）中包含的丰富世界知识，但我们的工作揭示了它们在六个不同基准设置下的细粒度视觉分类（FGVC）上的缺陷。最近的LVLMs最先进的模型，如LLaVa-1.5，InstructBLIP和GPT-4V，在分类性能方面严重下降，例如，LLaVA-1.5在斯坦福狗的EM平均下降了65.58，而且还难以根据出现在输入图像中的概念生成具有详细属性的准确解释，尽管它们有生成整体图像级描述的能力。深入分析表明，经过指导调整的LVLMs在给定文本时呈现出模态差距，显示出存在不一致性

    arXiv:2402.16315v1 Announce Type: cross  Abstract: Recent advances in instruction-tuned Large Vision-Language Models (LVLMs) have imbued the models with the ability to generate high-level, image-grounded explanations with ease. While such capability is largely attributed to the rich world knowledge contained within the Large Language Models (LLMs), our work reveals their shortcomings in fine-grained visual categorization (FGVC) across six different benchmark settings. Most recent state-of-the-art LVLMs like LLaVa-1.5, InstructBLIP and GPT-4V not only severely deteriorate in terms of classification performance, e.g., average drop of 65.58 in EM for Stanford Dogs for LLaVA-1.5, but also struggle to generate an accurate explanation with detailed attributes based on the concept that appears within an input image despite their capability to generate holistic image-level descriptions. In-depth analyses show that instruction-tuned LVLMs exhibit modality gap, showing discrepancy when given tex
    
[^16]: GreenLLaMA: 一种带有解释的解毒框架

    GreenLLaMA: A Framework for Detoxification with Explanations

    [https://arxiv.org/abs/2402.15951](https://arxiv.org/abs/2402.15951)

    GreenLLaMA是一种全面的端到端解毒框架，通过跨平台语料库训练出的模型优于当前最先进的模型。

    

    先前关于解毒的研究工作分散在某种程度上，因为它们并没有涵盖到真实场景中所需的所有解毒方面。值得注意的是，先前的研究将开发解毒模型的任务局限在仅见过的平台子集上，没有探讨模型在未知平台上的表现如何。此外，这些工作没有解决不可解毒性这一现象，即毒性文本无法在不改变含义的情况下进行解毒。我们提出了GreenLLaMA，这是第一个全面的端到端解毒框架，旨在减轻上述限制。我们首先介绍了一个跨平台伪并行语料库，应用多步数据处理和生成策略利用ChatGPT。然后，我们使用跨平台语料库训练一套解毒模型。我们展示了我们的解毒模型优于使用人工注释的最先进模型的表现。

    arXiv:2402.15951v1 Announce Type: cross  Abstract: Prior works on detoxification are scattered in the sense that they do not cover all aspects of detoxification needed in a real-world scenario. Notably, prior works restrict the task of developing detoxification models to only a seen subset of platforms, leaving the question of how the models would perform on unseen platforms unexplored. Additionally, these works do not address non-detoxifiability, a phenomenon whereby the toxic text cannot be detoxified without altering the meaning. We propose GreenLLaMA, the first comprehensive end-to-end detoxification framework, which attempts to alleviate the aforementioned limitations. We first introduce a cross-platform pseudo-parallel corpus applying multi-step data processing and generation strategies leveraging ChatGPT. We then train a suite of detoxification models with our cross-platform corpus. We show that our detoxification models outperform the SoTA model trained with human-annotated par
    
[^17]: MultiContrievers: 稠密检索表示的分析

    MultiContrievers: Analysis of Dense Retrieval Representations

    [https://arxiv.org/abs/2402.15925](https://arxiv.org/abs/2402.15925)

    该论文对稠密检索器的信息捕获进行了分析，探讨了其与语言模型的比较、信息提取的可行性以及提取性与性能、性别偏见的关系。

    

    稠密检索器将源文档压缩为（可能是有损的）向量表示，然而目前对于失去和保留的信息以及它们如何影响下游任务的分析较少。我们进行了首次对比稠密检索器捕获的信息与它们基于的语言模型（如BERT与Contriever）之间的分析。我们使用25个MultiBert检查点作为随机初始化来训练MultiContrievers，这是一组25个contriever模型。我们测试特定信息（如性别和职业）是否可以从类似维基百科的文档的contriever向量中提取。我们通过信息论探测来衡量这种可提取性。然后我们研究了可提取性与性能、性别偏见之间的关系，以及这些结果对许多随机初始化和数据洗牌的敏感性。我们发现（1）contriever模型有显著增加的可提取性

    arXiv:2402.15925v1 Announce Type: cross  Abstract: Dense retrievers compress source documents into (possibly lossy) vector representations, yet there is little analysis of what information is lost versus preserved, and how it affects downstream tasks. We conduct the first analysis of the information captured by dense retrievers compared to the language models they are based on (e.g., BERT versus Contriever). We use 25 MultiBert checkpoints as randomized initialisations to train MultiContrievers, a set of 25 contriever models. We test whether specific pieces of information -- such as gender and occupation -- can be extracted from contriever vectors of wikipedia-like documents. We measure this extractability via information theoretic probing. We then examine the relationship of extractability to performance and gender bias, as well as the sensitivity of these results to many random initialisations and data shuffles. We find that (1) contriever models have significantly increased extracta
    
[^18]: 将视觉问答置于交际背景中的CommVQA

    CommVQA: Situating Visual Question Answering in Communicative Contexts

    [https://arxiv.org/abs/2402.15002](https://arxiv.org/abs/2402.15002)

    CommVQA数据集将图像置于自然环境中，挑战了当前的VQA模型，结果表明为模型提供上下文信息能够提高性能。

    

    当前的视觉问答（VQA）模型往往在孤立的图像-问题对上进行训练和评估。然而，人们提出的问题取决于他们的信息需求和对图像内容的先前了解。为了评估将图像置于自然环境中如何塑造视觉问题，我们引入了CommVQA，这是一个包含图像、图像描述、图像可能出现的真实交际场景（例如旅行网站）以及依赖于场景的后续问题和答案的VQA数据集。我们展示了CommVQA对当前模型提出了挑战。为VQA模型提供上下文信息可广泛提高性能，突显将系统置于交际场景中的相关性。

    arXiv:2402.15002v1 Announce Type: new  Abstract: Current visual question answering (VQA) models tend to be trained and evaluated on image-question pairs in isolation. However, the questions people ask are dependent on their informational needs and prior knowledge about the image content. To evaluate how situating images within naturalistic contexts shapes visual questions, we introduce CommVQA, a VQA dataset consisting of images, image descriptions, real-world communicative scenarios where the image might appear (e.g., a travel website), and follow-up questions and answers conditioned on the scenario. We show that CommVQA poses a challenge for current models. Providing contextual information to VQA models improves performance broadly, highlighting the relevance of situating systems within a communicative scenario.
    
[^19]: 划分还是征服？你应该提炼LLM的哪一部分？

    Divide-or-Conquer? Which Part Should You Distill Your LLM?

    [https://arxiv.org/abs/2402.15000](https://arxiv.org/abs/2402.15000)

    本文提出了一种将推理任务分解为问题分解阶段和问题解决阶段的策略，发现问题分解阶段相比问题解决更容易提炼为较小模型，并证实该策略胜过单阶段解决方案。

    

    最近的研究表明，大型语言模型（LLMs）在被鼓励先解决主要任务的子任务时可以更好地解决推理任务。本文设计了一种类似的策略，将推理任务分解为问题分解阶段和问题解决阶段，并展示该策略能够胜过单阶段解决方案。此外，我们假设与解决问题相比，分解阶段更容易被提炼为较小的模型，因为后者需要大量的领域知识，而前者只需要学习一般的问题解决策略。我们提出了提炼这两种能力的方法，并评估了它们对推理结果和推理成本的影响。我们发现我们可以提炼问题分解阶段，并同时在任务、数据集和模型之间实现良好的泛化。然而，要提炼问题解决阶段就更困难了。

    arXiv:2402.15000v1 Announce Type: new  Abstract: Recent methods have demonstrated that Large Language Models (LLMs) can solve reasoning tasks better when they are encouraged to solve subtasks of the main task first. In this paper we devise a similar strategy that breaks down reasoning tasks into a problem decomposition phase and a problem solving phase and show that the strategy is able to outperform a single stage solution. Further, we hypothesize that the decomposition should be easier to distill into a smaller model compared to the problem solving because the latter requires large amounts of domain knowledge while the former only requires learning general problem solving strategies. We propose methods to distill these two capabilities and evaluate their impact on reasoning outcomes and inference cost. We find that we can distill the problem decomposition phase and at the same time achieve good generalization across tasks, datasets, and models. However, it is harder to distill the pr
    
[^20]: 语言中间件：工具在复杂环境中对语言代理至关重要

    Middleware for LLMs: Tools Are Instrumental for Language Agents in Complex Environments

    [https://arxiv.org/abs/2402.14672](https://arxiv.org/abs/2402.14672)

    这项研究探索了在复杂环境中利用工具增强大型语言模型的潜力，设计了定制化工具来辅助语言代理在庞大环境中进行探索，并展示了在知识库和数据库等复杂环境中，借助工具增强语言代理的重要潜力。

    

    大型语言模型（LLMs）的应用已经远远超出了文本处理的范围，预示着一个新时代的到来，在这个时代，LLMs被设想为能够在复杂现实环境中运行的通用语言代理。这些环境通常非常广阔，使得LLM不可能在其短期记忆中处理它们。受最近关于通过工具扩展LLMs能力的研究启发，本文探讨了工具在增强LLMs处理这种复杂性方面的潜力。为此，我们设计了定制工具，以协助在这些庞大环境中进行主动探索。这些工具可以作为一个中间件层，使LLM免受环境复杂性的影响。在两个代表性的复杂环境--知识库（KBs）和数据库中，我们展示了在复杂环境中使用工具增强语言代理的重要潜力。

    arXiv:2402.14672v1 Announce Type: cross  Abstract: The applications of large language models (LLMs) have expanded well beyond the confines of text processing, signaling a new era where LLMs are envisioned as generalist language agents capable of operating within complex real-world environments. These environments are often highly expansive, making it impossible for the LLM to process them within its short-term memory. Motivated by recent research on extending the capabilities of LLMs with tools, this paper investigates the intriguing potential of tools to augment LLMs in handling such complexity. To this end, we design customized tools to aid in the proactive exploration within these massive environments. Such tools can serve as a middleware layer shielding the LLM from environmental complexity. In two representative complex environments -- knowledge bases (KBs) and databases -- we demonstrate the significant potential of augmenting language agents with tools in complex environments. N
    
[^21]: LexC-Gen: 利用大型语言模型和双语词汇表为极低资源语言生成数据

    LexC-Gen: Generating Data for Extremely Low-Resource Languages with Large Language Models and Bilingual Lexicons

    [https://arxiv.org/abs/2402.14086](https://arxiv.org/abs/2402.14086)

    LexC-Gen提出了一种词典条件数据生成方法，可以以大规模生成低资源语言分类任务数据，取得了较好的效果。

    

    低资源语言的数据匮乏可以通过利用双语词典中从高资源语言的标记任务数据进行逐字翻译来解决，然而，双语词典通常与任务数据有限的词汇重叠，导致翻译覆盖和词典利用不佳。我们提出了一种称为LexC-Gen的词典条件数据生成方法，该方法可以大规模生成低资源语言分类任务数据。具体而言，LexC-Gen首先使用双语词典中的高资源语言单词生成与词典兼容的任务数据，然后通过单词翻译将其翻译成低资源语言。在17种极低资源语言中，LexC-Gen生成的数据在性能上与专家翻译的黄金数据竞争力相当，并且在情感分析和主题分类上平均比现有的基于词典的单词翻译方法提高了5.6和8.9个分数。

    arXiv:2402.14086v1 Announce Type: cross  Abstract: Data scarcity in low-resource languages can be addressed with word-to-word translations from labeled task data in high-resource languages using bilingual lexicons. However, bilingual lexicons often have limited lexical overlap with task data, which results in poor translation coverage and lexicon utilization. We propose lexicon-conditioned data generation (LexC-Gen), a method that generates low-resource-language classification task data at scale. Specifically, LexC-Gen first uses high-resource-language words from bilingual lexicons to generate lexicon-compatible task data, and then it translates them into low-resource languages with bilingual lexicons via word translation. Across 17 extremely low-resource languages, LexC-Gen generated data is competitive with expert-translated gold data, and yields on average 5.6 and 8.9 points improvement over existing lexicon-based word translation methods on sentiment analysis and topic classificati
    
[^22]: 软最大概率（大部分时候）在多项选择问答任务中预测大型语言模型的正确性

    Softmax Probabilities (Mostly) Predict Large Language Model Correctness on Multiple-Choice Q&A

    [https://arxiv.org/abs/2402.13213](https://arxiv.org/abs/2402.13213)

    多项选择问答任务中，基于最大softmax概率（MSPs）的模型预测方法有助于提高大型语言模型（LLMs）的正确性，我们提出了一种根据MSP有选择地弃权的策略以提高性能。

    

    尽管大型语言模型（LLMs）在许多任务上表现出色，但过度自信仍然是一个问题。我们假设在多项选择问答任务中，错误答案将与最大softmax概率（MSPs）较小相关，相比之下正确答案较大。我们在十个开源LLMs和五个数据集上全面评估了这一假设，在表现良好的原始问答任务中发现了对我们假设的强有力证据。对于表现最佳的六个LLMs，从MSP导出的AUROC在59/60个实例中都优于随机机会，p < 10^{-4}。在这六个LLMs中，平均AUROC范围在60%至69%之间。利用这些发现，我们提出了一个带有弃权选项的多项选择问答任务，并展示通过根据初始模型响应的MSP有选择地弃权可以提高性能。我们还用预softmax logits而不是softmax进行了相同的实验。

    arXiv:2402.13213v1 Announce Type: cross  Abstract: Although large language models (LLMs) perform impressively on many tasks, overconfidence remains a problem. We hypothesized that on multiple-choice Q&A tasks, wrong answers would be associated with smaller maximum softmax probabilities (MSPs) compared to correct answers. We comprehensively evaluate this hypothesis on ten open-source LLMs and five datasets, and find strong evidence for our hypothesis among models which perform well on the original Q&A task. For the six LLMs with the best Q&A performance, the AUROC derived from the MSP was better than random chance with p < 10^{-4} in 59/60 instances. Among those six LLMs, the average AUROC ranged from 60% to 69%. Leveraging these findings, we propose a multiple-choice Q&A task with an option to abstain and show that performance can be improved by selectively abstaining based on the MSP of the initial model response. We also run the same experiments with pre-softmax logits instead of sof
    
[^23]: ELECTRA的句子嵌入是否无法修复？语义文本相似性案例

    Are ELECTRA's Sentence Embeddings Beyond Repair? The Case of Semantic Textual Similarity

    [https://arxiv.org/abs/2402.13130](https://arxiv.org/abs/2402.13130)

    该研究探索了ELECTRA句子嵌入向量性能问题，并提出了一种新的截断模型微调方法，显著提高了语义文本相似性任务的表现

    

    虽然BERT生成具有高质量的句子嵌入向量，但其预训练计算成本是一个明显的缺点。相比之下，ELECTRA提供了一种经济高效的预训练目标和下游任务性能提升，但其句子嵌入向量表现不佳。社区悄然停止使用ELECTRA的句子嵌入向量进行语义文本相似性（STS）任务。我们注意到使用ELECTRA鉴别器的最后一层相对于较早的层时性能显著下降。我们探索了这种下降，并设计了一种修复ELECTRA嵌入向量的方法，提出了一种新颖的截断模型微调（TMFT）方法。在STS基准数据集上，TMFT将Spearman相关系数提高了8个多点，同时提高了参数效率。我们将我们的分析扩展到各种模型大小和语言。此外，我们发现了ELECTRA生成模型的惊人功效，它的性能与BERT持平

    arXiv:2402.13130v1 Announce Type: new  Abstract: While BERT produces high-quality sentence embeddings, its pre-training computational cost is a significant drawback. In contrast, ELECTRA delivers a cost-effective pre-training objective and downstream task performance improvements, but not as performant sentence embeddings. The community tacitly stopped utilizing ELECTRA's sentence embeddings for semantic textual similarity (STS). We notice a significant drop in performance when using the ELECTRA discriminator's last layer in comparison to earlier layers. We explore this drop and devise a way to repair ELECTRA's embeddings, proposing a novel truncated model fine-tuning (TMFT) method. TMFT improves the Spearman correlation coefficient by over 8 points while increasing parameter efficiency on the STS benchmark dataset. We extend our analysis to various model sizes and languages. Further, we discover the surprising efficacy of ELECTRA's generator model, which performs on par with BERT, usi
    
[^24]: 在摘要中识别事实不一致性：朝向大型语言模型的有效利用

    Identifying Factual Inconsistency in Summaries: Towards Effective Utilization of Large Language Model

    [https://arxiv.org/abs/2402.12821](https://arxiv.org/abs/2402.12821)

    该研究提出了针对摘要中事实不一致性的解决方案：通过大型语言模型在正确的范式设计下无需训练即可解决任务，并提出了训练策略以精炼更小型的高准确性的语言模型。

    

    事实上的不一致性对抽象性摘要生成器的商业部署构成重要障碍。本研究围绕两个重要问题展开：如何最好地利用大型语言模型来检测事实不一致性，以及如何精炼一个同时具有高效性和功效性的更小型语言模型？首先提出并评估了三种零样本范式，跨越五个不同数据集：直接推理整个摘要或每个摘要窗口；通过问题生成和回答进行实体验证。实验表明，在适当的范式设计下，语言模型本身能够在无需训练的情况下解决这一任务，平均超过强大的训练基线2.8%。为进一步促进实用性，我们提出针对精炼更小的开源语言模型的训练策略，该模型可以一次性高准确地评分整个摘要，胜过零

    arXiv:2402.12821v1 Announce Type: new  Abstract: Factual inconsistency poses a significant hurdle for the commercial deployment of abstractive summarizers. Under this Large Language Model (LLM) era, this work focuses around two important questions: what is the best way to leverage LLM for factual inconsistency detection, and how could we distill a smaller LLM with both high efficiency and efficacy? Three zero-shot paradigms are firstly proposed and evaluated across five diverse datasets: direct inference on the entire summary or each summary window; entity verification through question generation and answering. Experiments suggest that LLM itself is capable to resolve this task train-free under the proper paradigm design, surpassing strong trained baselines by 2.8% on average. To further promote practical utility, we then propose training strategies aimed at distilling smaller open-source LLM that learns to score the entire summary at once with high accuracy, which outperforms the zero
    
[^25]: 标准化: 将语言模型与专家定义的标准对齐，用于内容生成

    Standardize: Aligning Language Models with Expert-Defined Standards for Content Generation

    [https://arxiv.org/abs/2402.12593](https://arxiv.org/abs/2402.12593)

    该研究引入了一个名为Standardize的框架，通过检索式上下文学习指导大型语言模型与专家定义的标准对齐，提高了内容生成的精确性。

    

    在工程、医疗保健和教育领域，领域专家遵循严格的标准来制作质量内容，如技术手册、药物说明和儿童读物。然而，当前在可控文本生成方面的研究尚未探讨使用这些标准作为控制的参考。为此，我们引入了一种名为Standardize的检索式上下文学习框架，以指导大型语言模型与专家定义的标准对齐。以英语语言标准在教育领域作为一个使用案例，我们考虑了欧洲共同语言参考框架（CEFR）和通用核心标准（CCS）用于开放性内容生成任务。我们的研究结果表明，模型的精确性对于Llama2和GPT-4分别可以提高40%到100%，证明了从标准中提取知识工件并将其整合到生成中的可行性。

    arXiv:2402.12593v1 Announce Type: new  Abstract: Domain experts across engineering, healthcare, and education follow strict standards for producing quality content such as technical manuals, medication instructions, and children's reading materials. However, current works in controllable text generation have yet to explore using these standards as references for control. Towards this end, we introduce Standardize, a retrieval-style in-context learning-based framework to guide large language models to align with expert-defined standards. Focusing on English language standards in the education domain as a use case, we consider the Common European Framework of Reference for Languages (CEFR) and Common Core Standards (CCS) for the task of open-ended content generation. Our findings show that models can gain 40% to 100% increase in precise accuracy for Llama2 and GPT-4, respectively, demonstrating that the use of knowledge artifacts extracted from standards and integrating them in the gener
    
[^26]: AnaloBench：评估抽象和长上下文类比识别的基准

    AnaloBench: Benchmarking the Identification of Abstract and Long-context Analogies

    [https://arxiv.org/abs/2402.12370](https://arxiv.org/abs/2402.12370)

    通过提出ANALOBENCH基准来评估语言模型（LMs）进行类比推理的能力，发现扩展LMs规模对于处理涉及长场景或相关经验回忆的类比时带来的性能提升较小。

    

    人类经常进行类比思维，将个人经验与当前情况联系起来（$X$类似于$Y$是因为$Z$）。类比思维使人类能够用创造性方式解决问题，理解困难概念，更有效地表达想法。能否语言模型（LMs）也能做到这一点？为了回答这个问题，我们提出了ANALOBENCH，一个用于确定LMs类比推理能力的基准。我们的基准方法专注于人类之间共同的类比推理能力方面：（i）从大量信息中回忆相关经验，以及（ii）将类比推理应用于复杂和长度较长的场景。我们测试了大量专有模型（例如，GPT系列，Claude V2）和开源模型，如LLaMA2。与先前的结果一样，扩展LMs会带来一些性能提升。令人惊讶的是，在类比涉及长场景或回忆相关经验时，规模的提升带来的增益很小。

    arXiv:2402.12370v1 Announce Type: cross  Abstract: Humans regularly engage in analogical thinking, relating personal experiences to current situations ($X$ is analogous to $Y$ because of $Z$). Analogical thinking allows humans to solve problems in creative ways, grasp difficult concepts, and articulate ideas more effectively. Can language models (LMs) do the same? To answer this question, we propose ANALOBENCH, a benchmark to determine analogical reasoning ability in LMs. Our benchmarking approach focuses on aspects of this ability that are common among humans: (i) recalling related experiences from a large amount of information, and (ii) applying analogical reasoning to complex and lengthy scenarios. We test a broad collection of proprietary models (e.g., GPT family, Claude V2) and open source models such as LLaMA2. As in prior results, scaling up LMs results in some performance boosts. Surprisingly, scale offers minimal gains when, (i) analogies involve lengthy scenarios, or (ii) rec
    
[^27]: KARL: 知识感知检索和表示帮助学生保持和学习

    KARL: Knowledge-Aware Retrieval and Representations aid Retention and Learning in Students

    [https://arxiv.org/abs/2402.12291](https://arxiv.org/abs/2402.12291)

    KARL是一种基于DKT的学生模型，利用检索和BERT嵌入来实现高效准确的学生记忆预测，在AUC和校准误差方面优于现有学生模型，并提出了新颖的教学策略。

    

    Flashcard调度器是依赖于学生模型来预测学生掌握的单词卡，并使用教学策略根据这些预测安排词卡的工具。现有的学生模型仅使用单词卡级别的特征，比如学生的过去回答，忽略了单词卡之间的语义联系。深度知识跟踪（DKT）模型可以利用语言模型捕捉语义关系，但效率低下，缺乏内容丰富的数据集用于评估，并需要稳健的教学策略。为了解决这些问题，我们设计了KARL，这是受DKT启发的学生模型，利用检索和BERT嵌入以实现高效准确的学生记忆预测。为了测试KARL，我们收集了一个包含广泛学习历史关于琐事问题的新数据集。KARL在AUC和校准误差方面胜过现有的学生模型。最后，我们提出了一个新颖的教学策略，利用DKT模型的预测能力在线部署KARL。

    arXiv:2402.12291v1 Announce Type: new  Abstract: Flashcard schedulers are tools that rely on 1) student models to predict the flashcards a student knows; and 2) teaching policies to schedule cards based on these predictions. Existing student models, however, only use flashcard-level features, like the student's past responses, ignoring the semantic ties of flashcards. Deep Knowledge Tracing (DKT) models can capture semantic relations with language models, but are inefficient, lack content-rich datasets for evaluation, and require robust teaching policies. To address these issues, we design KARL, a DKT-inspired student model that uses retrieval and BERT embeddings for efficient and accurate student recall predictions. To test KARL, we collect a new dataset of diverse study history on trivia questions. KARL bests existing student models in AUC and calibration error. Finally, we propose a novel teaching policy that exploits the predictive power of DKT models to deploy KARL online. Based o
    
[^28]: 如果你讲我的语言，我会更好地学习：使用风格对齐响应调整增强大型语言模型微调

    I Learn Better If You Speak My Language: Enhancing Large Language Model Fine-Tuning with Style-Aligned Response Adjustments

    [https://arxiv.org/abs/2402.11192](https://arxiv.org/abs/2402.11192)

    将微调过程中的实际响应风格与大型语言模型固有风格相匹配能够产生更好的学习结果，开发的方法通过最小程度地调整模型响应来避免过拟合。

    

    使用小数据集为特定任务微调大型语言模型(LLMs)是一个普遍遇到的但复杂的挑战。在有限的示例上过多拟合可能会对模型的泛化能力和保留原始技能产生负面影响。我们的研究探讨了在微调过程中地实际响应风格的影响。我们发现将地实际响应风格与LLM固有风格匹配会产生更好的学习结果。基于这一观点，我们开发了一种方法，最小程度地修改LLM的现有响应以更正错误，使用这些调整后的响应作为训练目标。这种技术能够实现与模型固有响应风格一致的精确更正，维护模型的核心能力，从而避免过多拟合。我们的研究结果表明，这种方法不仅提高了LLM的特定任务准确性，而且关键地

    arXiv:2402.11192v1 Announce Type: cross  Abstract: Fine-tuning large language models (LLMs) with a small data set for particular tasks is a widely encountered yet complex challenge. The potential for overfitting on a limited number of examples can negatively impact the model's ability to generalize and retain its original skills. Our research explores the impact of the style of ground-truth responses during the fine-tuning process. We found that matching the ground-truth response style with the LLM's inherent style results in better learning outcomes. Building on this insight, we developed a method that minimally alters the LLM's pre-existing responses to correct errors, using these adjusted responses as training targets. This technique enables precise corrections in line with the model's native response style, safeguarding the model's core capabilities and thus avoid overfitting. Our findings show that this approach not only improves the LLM's task-specific accuracy but also crucially
    
[^29]: 密集通道检索：密集通道检索是否在检索中？

    Retrieval-Augmented Generation: Is Dense Passage Retrieval Retrieving?

    [https://arxiv.org/abs/2402.11035](https://arxiv.org/abs/2402.11035)

    DPR微调预训练网络以增强查询和相关文本数据之间的嵌入对齐，发现训练中知识去中心化，但也揭示了模型内部知识的局限性

    

    密集通道检索（DPR）是改进大型语言模型（LLM）性能的检索增强生成（RAG）范式中的第一步。 DPR微调预训练网络，以增强查询和相关文本数据之间的嵌入对齐。对DPR微调的深入理解将需要从根本上释放该方法的全部潜力。在这项工作中，我们通过使用探针、层激活分析和模型编辑的组合，机械地探索了DPR训练模型。我们的实验证明，DPR训练使网络中存储知识的方式去中心化，创建了访问相同信息的多个路径。我们还发现了这种训练风格的局限性：预训练模型的内部知识限制了检索模型可以检索的内容。这些发现为密集检索提出了一些可能的方向：（1）暴露DPR训练过程

    arXiv:2402.11035v1 Announce Type: new  Abstract: Dense passage retrieval (DPR) is the first step in the retrieval augmented generation (RAG) paradigm for improving the performance of large language models (LLM). DPR fine-tunes pre-trained networks to enhance the alignment of the embeddings between queries and relevant textual data. A deeper understanding of DPR fine-tuning will be required to fundamentally unlock the full potential of this approach. In this work, we explore DPR-trained models mechanistically by using a combination of probing, layer activation analysis, and model editing. Our experiments show that DPR training decentralizes how knowledge is stored in the network, creating multiple access pathways to the same information. We also uncover a limitation in this training style: the internal knowledge of the pre-trained model bounds what the retrieval model can retrieve. These findings suggest a few possible directions for dense retrieval: (1) expose the DPR training process 
    
[^30]: 自洽的上下文感知转录器用于语音识别

    Self-consistent context aware conformer transducer for speech recognition

    [https://arxiv.org/abs/2402.06592](https://arxiv.org/abs/2402.06592)

    这项研究提出了一种自洽的上下文感知转录器模型，能够在语音识别中提高不常见单词的准确性，而不影响常见单词的错误率。

    

    我们提出了一种基于转录器的新颖神经网络架构，为ASR系统添加了上下文信息流。我们的方法在提高识别不常见单词的准确性的同时不影响常见单词的错误率。我们探索了当我们使用新模型和/或与上下文语言模型浅度融合时，对不常见单词准确性的改善。我们发现两者的组合可以累积提高不常见单词的识别准确性。

    We propose a novel neural network architecture based on conformer transducer that adds contextual information flow to the ASR systems. Our method improves the accuracy of recognizing uncommon words while not harming the word error rate of regular words. We explore the uncommon words accuracy improvement when we use the new model and/or shallow fusion with context language model. We found that combination of both provides cumulative gain in uncommon words recognition accuracy.
    
[^31]: 我们是否需要语言特定的事实核查模型？以汉语为例

    Do We Need Language-Specific Fact-Checking Models? The Case of Chinese

    [https://arxiv.org/abs/2401.15498](https://arxiv.org/abs/2401.15498)

    本文研究了语言特定事实核查模型的潜在益处，提出了一个汉语事实核查系统，并展示其优于翻译方法和多语言大型语言模型，同时对偏见更加稳健，强调了语言特定性的重要性。

    

    本文研究了语言特定事实核查模型的潜在益处，重点关注汉语案例。我们首先展示了基于翻译方法和多语言大型语言模型（例如GPT-4）的局限性，突出了对语言特定系统的需求。我们进一步提出了一个汉语事实核查系统，通过整合上下文信息，可以更好地从文档中检索证据。为了更好地分析不同系统中的令牌级偏见，我们基于CHEF数据集构建了一个对抗数据集，其中每个实例与原始实例具有较大的词重叠，但具有相反的真实性标签。在CHEF数据集和我们的对抗数据集上的实验结果表明，我们提出的方法优于基于翻译的方法和多语言LLM，并且对偏见更加稳健，但仍有很大的改进空间，强调了语言特定性的重要性。

    arXiv:2401.15498v2 Announce Type: replace  Abstract: This paper investigates the potential benefits of language-specific fact-checking models, focusing on the case of Chinese. We first demonstrate the limitations of translation-based methods and multilingual large language models (e.g., GPT-4), highlighting the need for language-specific systems. We further propose a Chinese fact-checking system that can better retrieve evidence from a document by incorporating context information. To better analyze token-level biases in different systems, we construct an adversarial dataset based on the CHEF dataset, where each instance has large word overlap with the original one but holds the opposite veracity label. Experimental results on the CHEF dataset and our adversarial dataset show that our proposed method outperforms translation-based methods and multilingual LLMs and is more robust toward biases, while there is still large room for improvement, emphasizing the importance of language-specif
    
[^32]: EHRAgent：代码赋能大型语言模型在电子健康记录上进行少样本复杂表格推理

    EHRAgent: Code Empowers Large Language Models for Few-shot Complex Tabular Reasoning on Electronic Health Records

    [https://arxiv.org/abs/2401.07128](https://arxiv.org/abs/2401.07128)

    EHRAgent是一个由代码接口赋能的大型语言模型代理，用于自主生成和执行多表格推理代码，通过错误信息学习改进生成的代码，结合长期记忆选择并建立在过去经验中的成功案例。

    

    大型语言模型（LLMs）在规划和工具利用方面表现出色，但在医学问题解决方面尚未有太多开发。我们提出EHRAgent，这是一个由代码接口赋能的LLM代理，用于在电子健康记录（EHRs）中自主生成和执行多表格推理的代码。首先，我们将EHR问答任务制定为工具使用规划过程，将一个复杂任务高效地分解为一系列可管理的操作。通过集成交互式编码和执行反馈，EHRAgent从错误消息中学习并通过迭代改进最初生成的代码。此外，我们通过结合长期记忆来增强LLM代理，使EHRAgent能够有效地选择并建立在过去经验中最相关的成功案例上。在三个真实世界的多表格EHR数据集上进行的实验显示...

    arXiv:2401.07128v2 Announce Type: replace-cross  Abstract: Large language models (LLMs) have demonstrated exceptional capabilities in planning and tool utilization as autonomous agents, but few have been developed for medical problem-solving. We propose EHRAgent, an LLM agent empowered with a code interface, to autonomously generate and execute code for multi-tabular reasoning within electronic health records (EHRs). First, we formulate an EHR question-answering task into a tool-use planning process, efficiently decomposing a complicated task into a sequence of manageable actions. By integrating interactive coding and execution feedback, EHRAgent learns from error messages and improves the originally generated code through iterations. Furthermore, we enhance the LLM agent by incorporating long-term memory, which allows EHRAgent to effectively select and build upon the most relevant successful cases from past experiences. Experiments on three real-world multi-tabular EHR datasets show t
    
[^33]: 解锁预测性文本生成：对大型语言模型解码的受限方法

    Unlocking Anticipatory Text Generation: A Constrained Approach for Large Language Models Decoding

    [https://arxiv.org/abs/2312.06149](https://arxiv.org/abs/2312.06149)

    提出了将文本生成形式化为未来受限生成问题的方法，以最小化不良行为并强制执行对指令的忠实性，并通过LLMs有效指导文本生成。

    

    大型语言模型(LLMs)展现了强大的文本生成能力。然而，对于给定提示或指令实现最佳结果可能具有挑战性，特别是对于十亿级别的模型。此外，不良行为如毒性或幻觉可能会显现。在这项工作中，我们提出将文本生成形式化为未来受限生成问题，以最小化不良行为并强制执行对指令的忠实性。使用LLMs实现未来约束满足度的估计引导文本生成过程。我们的广泛实验表明所提出的方法在三个不同的文本生成任务中的有效性：关键词受限生成、毒性减少等。

    arXiv:2312.06149v2 Announce Type: replace-cross  Abstract: Large Language Models (LLMs) have demonstrated a powerful ability for text generation. However, achieving optimal results with a given prompt or instruction can be challenging, especially for billion-sized models. Additionally, undesired behaviors such as toxicity or hallucinations can manifest. While much larger models (e.g., ChatGPT) may demonstrate strength in mitigating these issues, there is still no guarantee of complete prevention. In this work, we propose formalizing text generation as a future-constrained generation problem to minimize undesirable behaviors and enforce faithfulness to instructions. The estimation of future constraint satisfaction, accomplished using LLMs, guides the text generation process. Our extensive experiments demonstrate the effectiveness of the proposed approach across three distinct text generation tasks: keyword-constrained generation (Lin et al., 2020), toxicity reduction (Gehman et al., 202
    
[^34]: AMRFact：利用AMR生成负样本增强摘要的事实性评估

    AMRFact: Enhancing Summarization Factuality Evaluation with AMR-Driven Negative Samples Generation

    [https://arxiv.org/abs/2311.09521](https://arxiv.org/abs/2311.09521)

    AMRFact是一个框架，利用AMR生成负样本，增强了摘要事实性评估，生成的连贯且事实不一致的摘要具有高错误率。

    

    确保事实一致性对于自然语言生成任务至关重要，特别是在提取式摘要中，保持信息的完整性至关重要。以前关于评估摘要的事实一致性的工作通常采用基于蕴涵的方法，首先生成扰动（事实不一致）摘要，然后在生成的数据上训练一个分类器，在测试时检测事实不一致。然而，先前生成扰动摘要的方法要么缺乏连贯性，要么缺乏错误类型覆盖。为了解决这些问题，我们提出了AMRFact，一个利用抽象意义表示（AMR）生成扰动摘要的框架。我们的方法将事实一致的摘要解析为AMR图，并注入可控的事实不一致，以创建负面示例，允许生成具有高错误率的连贯事实不一致的摘要。

    arXiv:2311.09521v2 Announce Type: replace  Abstract: Ensuring factual consistency is crucial for natural language generation tasks, particularly in abstractive summarization, where preserving the integrity of information is paramount. Prior works on evaluating factual consistency of summarization often take the entailment-based approaches that first generate perturbed (factual inconsistent) summaries and then train a classifier on the generated data to detect the factually inconsistencies during testing time. However, previous approaches generating perturbed summaries are either of low coherence or lack error-type coverage. To address these issues, we propose AMRFact, a framework that generates perturbed summaries using Abstract Meaning Representations (AMRs). Our approach parses factually consistent summaries into AMR graphs and injects controlled factual inconsistencies to create negative examples, allowing for coherent factually inconsistent summaries to be generated with high error
    
[^35]: 在搜索长尾中：通过逻辑规则引导搜索系统性生成长尾推理知识

    In Search of the Long-Tail: Systematic Generation of Long-Tail Inferential Knowledge via Logical Rule Guided Search

    [https://arxiv.org/abs/2311.07237](https://arxiv.org/abs/2311.07237)

    该研究提出了一个名为LINK的框架，能够系统性地生成长尾推理知识，从而更有效地评估LLMs在推理空间中的表现。

    

    最先进的LLMs在诸如自然语言推理等推理任务上胜过人类。最近评估LLMs的研究指出，在来自低概率分布——即长尾的输入数据上表现大幅下降。因此，我们专注于系统生成涉及长尾推理知识的语句，以更有效地评估LLMs在推理空间中的表现。我们首先提出了一个新颖的框架Logic-Induced-Knowledge-Search（LINK），该框架生成基于符号规则模板的事实正确且长尾知识语句；LINK有效地生成长尾分布数据，零-shot提示的LLMs无法到达，并且在事实正确性方面优于零-shot GPT4达到5%。我们进一步使用LINK生成的数据构建了一个名为Logic-Induced-Long-Tail（LINT）的数据集，可用于评估长尾分布上的下游模型；LINT包含108K个知识条目。

    arXiv:2311.07237v2 Announce Type: replace-cross  Abstract: State-of-the-art LLMs outperform humans on reasoning tasks such as Natural Language Inference. Recent works evaluating LLMs note a marked performance drop on input data from the low-probability distribution, i.e., the longtail. Therefore, we focus on systematically generating statements involving long-tail inferential knowledge for more effective evaluation of LLMs in the reasoning space. We first propose a novel framework Logic-Induced- Knowledge-Search (LINK) that generates factually correct and long-tail knowledge statements grounded on symbolic rule templates; LINK effectively generates data in the longtail distribution that zero-shot prompted LLMs are unable to reach, and outperforms zero-shot GPT4 on factual correctness by 5%. We further use the data generated by LINK to construct a dataset Logic-Induced-Long-Tail (LINT) that can be used to evaluate downstream models on the long-tail distribution; LINT contains 108K knowl
    
[^36]: SLANG: 大型语言模型对新概念的理解

    SLANG: New Concept Comprehension of Large Language Models. (arXiv:2401.12585v1 [cs.CL])

    [http://arxiv.org/abs/2401.12585](http://arxiv.org/abs/2401.12585)

    本研究提出了一个新的基准SLANG，旨在增强大型语言模型LLMs对互联网上新概念的理解能力，同时提出了一种基于因果推断的基准方法FOCUS，能帮助LLMs更好地理解新的短语和用法模式。

    

    语言的动态性，尤其在互联网上的俚语和表情包等方面的体现，给大型语言模型（LLMs）的适应性带来了严峻挑战。传统上，这些模型通常仅绑定在静态数据集上，很难跟上在线社区中快速语言进化的步伐。本研究解决了弥合这一差距的迫切需求，旨在增强LLMs对互联网上新概念的理解能力，同时避免高成本和不切实际的持续重训练。为应对这个问题，我们提出了一个新的评估LLMs在理解新兴语言趋势方面能力的基准 - SLANG，并提出了一种基于因果推断的基准方法 FOCUS，它能增强LLMs对新的短语和用法模式的理解。该方法包括对语言转变的真实世界实例进行详细研究，作为背景依据，以形成更精确和具有上下文相关性的新连接。

    The dynamic nature of language, particularly evident in the realm of slang and memes on the Internet, poses serious challenges to the adaptability of large language models (LLMs). Traditionally anchored to static datasets, these models often struggle to keep up with the rapid linguistic evolution characteristic of online communities. This research addresses the critical need to bridge this gap, aiming to enhance LLMs' comprehension of evolving new concepts on the internet, without the high cost and impracticality of continual retraining. To address this issue, we propose a new benchmark $\textbf{SLANG}$ to assess LLMs' proficiency in comprehending emerging linguistic trends and a baseline approach $\textbf{FOCUS}$, which uses causal inference to enhance LLMs to understand new phrases and usage patterns. This approach involves scrutinizing real-world instances of linguistic shifts, serving as contextual beacons, to form more precise and contextually relevant connections between newly em
    
[^37]: 用可比较的语料和多个参考文献进行代码翻译的数据增强

    Data Augmentation for Code Translation with Comparable Corpora and Multiple References. (arXiv:2311.00317v1 [cs.CL])

    [http://arxiv.org/abs/2311.00317](http://arxiv.org/abs/2311.00317)

    该论文介绍了两种数据增强方法来改善编程语言之间的代码翻译。通过构建可比较的语料库和增加多个参考翻译，实验结果表明这些方法显著提高了CodeT5在Java、Python和C++之间的翻译准确性。

    

    在编程语言之间进行代码翻译的一个主要挑战是平行训练数据通常有限。为了克服这个挑战，我们提出了两种数据增强技术，一种是构建可比较的语料库（即具有类似功能的代码对），另一种是用多个参考翻译来增强现有的平行数据。具体而言，我们构建并分析了多种类型的可比较的语料库，包括使用代码生成模型从自然语言文档中生成的程序。此外，为了减少对单个参考翻译的过拟合，我们自动生成了可用平行数据的额外翻译参考，并通过单元测试对翻译进行筛选，从而增加了目标翻译的变化。实验证明，我们的数据增强技术显著提高了CodeT5在Java、Python和C++之间的翻译准确性（平均提升了7.5%的计算准确性（CA@1））。

    One major challenge of translating code between programming languages is that parallel training data is often limited. To overcome this challenge, we present two data augmentation techniques, one that builds comparable corpora (i.e., code pairs with similar functionality), and another that augments existing parallel data with multiple reference translations. Specifically, we build and analyze multiple types of comparable corpora, including programs generated from natural language documentation using a code generation model. Furthermore, to reduce overfitting to a single reference translation, we automatically generate additional translation references for available parallel data and filter the translations by unit tests, which increases variation in target translations. Experiments show that our data augmentation techniques significantly improve CodeT5 for translation between Java, Python, and C++ by an average of 7.5% Computational Accuracy (CA@1), which verifies the correctness of tr
    
[^38]: 学习个性化故事评估

    Learning Personalized Story Evaluation. (arXiv:2310.03304v1 [cs.CL])

    [http://arxiv.org/abs/2310.03304](http://arxiv.org/abs/2310.03304)

    该论文提出了学习个性化故事评估的方法。为了解决大型语言模型在开放式文本生成任务的评估问题，论文创建了两个新的数据集，并开发了一个个性化故事评估模型，能够根据评审人员的示例评价进行个性化评估。

    

    尽管大型语言模型（LLM）在诸如问答和检索等更客观的任务上显示出令人印象深刻的结果，但评估它们在开放式文本生成方面的表现仍然是一个困难的问题，原因包括（1）数据污染；（2）多维评估标准；以及（3）来自评审人员个人偏好的主观性。为了解决这些问题，我们提出在一个无污染的开放式生成评估中建模个性化。我们使用适当的匿名化和新的个性化标签，重新利用现有数据集创建了两个新的数据集Per-MPST和Per-DOC用于个性化故事评估。我们进一步开发了一个个性化故事评估模型PERSE来推测评审人员的偏好，并提供个性化评估。具体而言，对于某个评审人员的一些示例评价，PERSE可以预测该评审人员在新的情节上的详细评审或细粒度比较（如趣味性和惊喜）。

    While large language models (LLMs) have shown impressive results for more objective tasks such as QA and retrieval, it remains nontrivial to evaluate their performance on open-ended text generation for reasons including (1) data contamination; (2) multi-dimensional evaluation criteria; and (3) subjectiveness stemming from reviewers' personal preferences. To address such issues, we propose to model personalization in an uncontaminated open-ended generation assessment. We create two new datasets Per-MPST and Per-DOC for personalized story evaluation, by re-purposing existing datasets with proper anonymization and new personalized labels. We further develop a personalized story evaluation model PERSE to infer reviewer preferences and provide a personalized evaluation. Specifically, given a few exemplary reviews from a particular reviewer, PERSE predicts either a detailed review or fine-grained comparison in several aspects (such as interestingness and surprise) for that reviewer on a new 
    
[^39]: ChatGPT是ENFJ，Bard是ISTJ：大型语言模型的个性实证研究。

    ChatGPT an ENFJ, Bard an ISTJ: Empirical Study on Personalities of Large Language Models. (arXiv:2305.19926v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2305.19926](http://arxiv.org/abs/2305.19926)

    本研究通过采用特质理论框架，实验证明了ChatGPT始终表现出ENFJ型人格，无论指令或情境如何。研究揭示了LLMs的个性化，有助于促进人与机器之间更好的沟通和协作。

    

    大型语言模型（LLMs）在人工智能领域取得了显著进展，大大重塑了人机交互。我们不仅关注LLMs的性能，还从心理学角度探索它们的特点，认识到了理解它们行为特征的重要性。本研究采用心理学的一个框架——特质理论研究LLMs所展示的行为模式。我们首先关注评估ChatGPT所展示的人格类型的一致性。此外，实验涉及七种附加语言的跨语言影响，以及六种其他LLMs的研究。此外，该研究还调查了ChatGPT是否能够展示对指令或情境线索的人格变化。研究结果表明，无论指令或情境如何，ChatGPT始终保持其ENFJ人格。通过揭示LLMs的个性化，我们预计我们的解决方案可以促进人与机器之间更好的沟通和协作。

    Large Language Models (LLMs) have made remarkable advancements in the field of artificial intelligence, significantly reshaping the human-computer interaction. We not only focus on the performance of LLMs, but also explore their features from a psychological perspective, acknowledging the importance of understanding their behavioral characteristics. Our study examines the behavioral patterns displayed by LLMs by employing trait theory, a psychological framework. We first focus on evaluating the consistency of personality types exhibited by ChatGPT. Furthermore, experiments include cross-lingual effects on seven additional languages, and the investigation of six other LLMs. Moreover, the study investigates whether ChatGPT can exhibit personality changes in response to instructions or contextual cues. The findings show that ChatGPT consistently maintains its ENFJ personality regardless of instructions or contexts. By shedding light on the personalization of LLMs, we anticipate that our s
    
[^40]: SMILE：利用ChatGPT实现单轮到多轮包容性语言扩展的心理健康支持

    SMILE: Single-turn to Multi-turn Inclusive Language Expansion via ChatGPT for Mental Health Support. (arXiv:2305.00450v1 [cs.CL])

    [http://arxiv.org/abs/2305.00450](http://arxiv.org/abs/2305.00450)

    本研究提出了SMILE方法，使用ChatGPT将公共单轮对话扩展为多轮对话，生成了大规模、多样化、接近真实生活的多轮心理健康支持对话语料库，可用于训练和评估专门的对话系统。

    

    开发专门的对话系统以提供心理健康支持已成为越来越多的研究关注点。然而，由于个人信息的敏感性以及所需的时间和成本，获取大规模的真实多轮心理健康支持对话存在困难。为了解决这些问题，我们引入了SMILE方法，一种使用ChatGPT将公共单轮对话扩展为多轮对话的包容性语言扩展技术。我们首先进行了初步的探索性研究，验证了SMILE方法的有效性。此外，我们对使用和未使用SMILE方法生成的数据集进行了全面系统的对比分析，证明SMILE方法可以产生大规模、多样化、接近真实生活的多轮心理健康支持对话语料库，包括对话主题、词汇和语义特征。最后，我们使用收集的语料库来训练和评估专门的心理健康支持对话系统。

    There has been an increasing research interest in developing specialized dialogue systems that can offer mental health support. However, gathering large-scale and real-life multi-turn conversations for mental health support poses challenges due to the sensitivity of personal information, as well as the time and cost involved. To address these issues, we introduce the SMILE approach, an inclusive language expansion technique that employs ChatGPT to extend public single-turn dialogues into multi-turn ones. Our research first presents a preliminary exploratory study that validates the effectiveness of the SMILE approach. Furthermore, we conduct a comprehensive and systematic contrastive analysis of datasets generated with and without the SMILE approach, demonstrating that the SMILE method results in a large-scale, diverse, and close-to-real-life multi-turn mental health support conversation corpus, including dialog topics, lexical and semantic features. Finally, we use the collected corpu
    

