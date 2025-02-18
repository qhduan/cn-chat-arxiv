# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [GINopic: Topic Modeling with Graph Isomorphism Network](https://arxiv.org/abs/2404.02115) | GINopic是一种主题建模框架，利用图同构网络捕捉单词之间的相关性，相比于现有主题模型，展示了更好的有效性和推进主题建模的潜力。 |
| [^2] | [TableLLM: Enabling Tabular Data Manipulation by LLMs in Real Office Usage Scenarios](https://arxiv.org/abs/2403.19318) | TableLLM是一个拥有130亿参数的强大大语言模型，专门用于熟练处理表格数据操作任务，通过远程监督方法和交叉验证策略，TableLLM相对于其他现有的通用和表格数据专注的LLMs具有明显优势。 |
| [^3] | [Enhanced Short Text Modeling: Leveraging Large Language Models for Topic Refinement](https://arxiv.org/abs/2403.17706) | 利用大型语言模型的先进能力，提出了一种名为“主题细化”的新方法，通过引入提示工程和消除离题词等方式改进短文本的主题建模质量，提高了主题的语义质量。 |
| [^4] | [Large Language Models and Causal Inference in Collaboration: A Comprehensive Survey](https://arxiv.org/abs/2403.09606) | 大型语言模型的出现极大影响了自然语言处理领域，特别是通过其先进的推理能力。而本综述则重点评估和改进了大型语言模型在因果推断方面的应用，包括提高推理能力、解决公平和安全问题、提供解释和处理多模态。 |
| [^5] | [Large Language Models are Contrastive Reasoners](https://arxiv.org/abs/2403.08211) | 对比提示方法显著提高大型语言模型进行复杂推理的能力，不仅在算术、常识和符号推理任务上表现优良，还可以与现有提示方法整合，实现更好的性能。 |
| [^6] | [You Need to Pay Better Attention](https://arxiv.org/abs/2403.01643) | 提出了三种新的注意力机制，在效率和学习能力方面优于标准的多头注意力，提高了Transformer模型的性能和更广泛的部署能力。 |
| [^7] | [Bias in Language Models: Beyond Trick Tests and Toward RUTEd Evaluation](https://arxiv.org/abs/2402.12649) | 这项研究探讨了语言模型中偏见的负面影响，研究了"技巧测试"与更现实世界中表现的RUTEd评估之间的关联性，特别关注性别-职业偏见，并进行了多项评估比较。 |
| [^8] | [Bridging Causal Discovery and Large Language Models: A Comprehensive Survey of Integrative Approaches and Future Directions](https://arxiv.org/abs/2402.11068) | 本文综合调查了将大型语言模型（如GPT4）整合到因果发现任务中的方法，揭示了它们在推断因果结构时对元数据和自然语言的创新利用，强调了LLMs在增强传统CD方法和作为专家辅助方面的潜力和挑战。 |
| [^9] | [Language Writ Large: LLMs, ChatGPT, Grounding, Meaning and Understanding](https://arxiv.org/abs/2402.02243) | ChatGPT在LLM规模上通过利用语言本身的收敛约束来做到超出预期的表现，但并不真正理解语义以及与感觉动作的直接联系。 |
| [^10] | [Assistive Large Language Model Agents for Socially-Aware Negotiation Dialogues](https://arxiv.org/abs/2402.01737) | 本文旨在开发LLM代理以减轻多代理设置下谈判中的社交规范违反。我们引入了基于价值影响的环境学习（ICL）方法，用于为基于LLM的修正代理识别高质量的ICL示例，其中价值影响函数衡量谈判结果的质量。我们展示了这种方法与策略学习的联系，并提供了丰富的实证证据来证明其在三个不同主题的谈判中的有效性。 |
| [^11] | [SWEA: Changing Factual Knowledge in Large Language Models via Subject Word Embedding Altering](https://arxiv.org/abs/2401.17809) | 提出了一种主题词嵌入修改框架（SWEA），通过在推理阶段修改主题的表示来编辑知识，保护模型的原始权重，避免不可逆的损害和额外的推理开销。 |
| [^12] | [Measuring Catastrophic Forgetting in Cross-Lingual Transfer Paradigms: Exploring Tuning Strategies.](http://arxiv.org/abs/2309.06089) | 该研究比较了不同的微调和跨语言转移策略在解决跨语言任务时的表现，评估了灾难性遗忘的程度和转移的成功程度。 |

# 详细

[^1]: GINopic：利用图同构网络进行主题建模

    GINopic: Topic Modeling with Graph Isomorphism Network

    [https://arxiv.org/abs/2404.02115](https://arxiv.org/abs/2404.02115)

    GINopic是一种主题建模框架，利用图同构网络捕捉单词之间的相关性，相比于现有主题模型，展示了更好的有效性和推进主题建模的潜力。

    

    主题建模是分析和探索大型文档集合的广泛使用方法。 最近的研究工作将预训练的上下文化语言模型，如BERT嵌入，纳入主题建模中。 然而，它们通常忽略了单词之间相互依赖传达的固有信息价值。 本研究介绍了GINopic，一种基于图同构网络的主题建模框架，以捕捉单词之间的相关性。 通过在不同基准数据集上进行内在的（定量和定性）和外部的评估，我们展示了与现有主题模型相比，GINopic的有效性，并突出了其推进主题建模的潜力。

    arXiv:2404.02115v1 Announce Type: new  Abstract: Topic modeling is a widely used approach for analyzing and exploring large document collections. Recent research efforts have incorporated pre-trained contextualized language models, such as BERT embeddings, into topic modeling. However, they often neglect the intrinsic informational value conveyed by mutual dependencies between words. In this study, we introduce GINopic, a topic modeling framework based on graph isomorphism networks to capture the correlation between words. By conducting intrinsic (quantitative as well as qualitative) and extrinsic evaluations on diverse benchmark datasets, we demonstrate the effectiveness of GINopic compared to existing topic models and highlight its potential for advancing topic modeling.
    
[^2]: TableLLM：在实际办公使用场景中实现LLMs对表格数据进行处理的能力

    TableLLM: Enabling Tabular Data Manipulation by LLMs in Real Office Usage Scenarios

    [https://arxiv.org/abs/2403.19318](https://arxiv.org/abs/2403.19318)

    TableLLM是一个拥有130亿参数的强大大语言模型，专门用于熟练处理表格数据操作任务，通过远程监督方法和交叉验证策略，TableLLM相对于其他现有的通用和表格数据专注的LLMs具有明显优势。

    

    我们介绍了TableLLM，这是一个拥有130亿参数的强大大语言模型（LLM），专门用于熟练处理表格数据操作任务，无论其嵌入在文档还是电子表格中，以满足真实办公场景需求。我们提出了一种远程监督方法进行训练，其中包括一种推理过程扩展策略，有助于训练LLMs更有效地理解推理模式，以及一种交叉验证策略，确保自动生成数据的质量。为了评估TableLLM的性能，我们构建了一个旨在解决文档和电子表格格式的基准测试，并构建了一个能够处理两种场景的组织良好的评估管线。彻底的评估凸显了TableLLM相对于各种现有通用和专注于表格数据的LLMs的优势。我们已公开发布了该模型。

    arXiv:2403.19318v1 Announce Type: new  Abstract: We introduce TableLLM, a robust large language model (LLM) with 13 billion parameters, purpose-built for proficiently handling tabular data manipulation tasks, whether they are embedded within documents or spreadsheets, catering to real-world office scenarios. We propose a distant supervision method for training, which comprises a reasoning process extension strategy, aiding in training LLMs to understand reasoning patterns more effectively as well as a cross-way validation strategy, ensuring the quality of the automatically generated data. To evaluate the performance of TableLLM, we have crafted a benchmark tailored to address both document and spreadsheet formats as well as constructed a well-organized evaluation pipeline capable of handling both scenarios. Thorough evaluations underscore the advantages of TableLLM when compared to various existing general-purpose and tabular data-focused LLMs. We have publicly released the model check
    
[^3]: 增强短文本建模：利用大型语言模型进行主题细化

    Enhanced Short Text Modeling: Leveraging Large Language Models for Topic Refinement

    [https://arxiv.org/abs/2403.17706](https://arxiv.org/abs/2403.17706)

    利用大型语言模型的先进能力，提出了一种名为“主题细化”的新方法，通过引入提示工程和消除离题词等方式改进短文本的主题建模质量，提高了主题的语义质量。

    

    有效地构建针对简短文本（如推文和新闻标题）的主题模型对捕捉社会动态的迅速变化至关重要。然而，传统主题模型往往在准确表达短文本的语义细微差异方面存在不足，这是由于它们的简洁性和缺乏上下文数据。在我们的研究中，我们利用大型语言模型（LLMs）的先进能力，引入了一种称为“主题细化”的新方法。该方法并非直接参与主题的初步建模，而是专注于改进主题在被挖掘后的阶段。通过引入提示工程，我们指导LLMs消除给定主题中的离题词，确保仅保留与语境相关的词汇或用更符合语义的词汇替换。这种方法模拟了人类般的审查和改进主题的方式，从而提升了各种主题生成的语义质量。

    arXiv:2403.17706v1 Announce Type: cross  Abstract: Crafting effective topic models for brief texts, like tweets and news headlines, is essential for capturing the swift shifts in social dynamics. Traditional topic models, however, often fall short in accurately representing the semantic intricacies of short texts due to their brevity and lack of contextual data. In our study, we harness the advanced capabilities of Large Language Models (LLMs) to introduce a novel approach termed "Topic Refinement". This approach does not directly involve itself in the initial modeling of topics but focuses on improving topics after they have been mined. By employing prompt engineering, we direct LLMs to eliminate off-topic words within a given topic, ensuring that only contextually relevant words are preserved or substituted with ones that fit better semantically. This method emulates human-like scrutiny and improvement of topics, thereby elevating the semantic quality of the topics generated by vario
    
[^4]: 大型语言模型与协作中的因果推断：一项综合调查

    Large Language Models and Causal Inference in Collaboration: A Comprehensive Survey

    [https://arxiv.org/abs/2403.09606](https://arxiv.org/abs/2403.09606)

    大型语言模型的出现极大影响了自然语言处理领域，特别是通过其先进的推理能力。而本综述则重点评估和改进了大型语言模型在因果推断方面的应用，包括提高推理能力、解决公平和安全问题、提供解释和处理多模态。

    

    因果推断已经显示出潜力，通过捕捉变量之间的因果关系，提高自然语言处理（NLP）模型的预测准确性、公平性、稳健性和可解释性。生成型大型语言模型（LLMs）的出现显著影响了各种NLP领域，特别是通过其先进的推理能力。该调查重点评估和改进LLMs的因果视角，在以下领域展开：理解和改进LLMs的推理能力，解决LLMs中的公平性和安全性问题，为LLMs提供解释，并处理多模态。同时，LLMs强大的推理能力反过来可以通过帮助因果关系发现和因果效应估计来促进因果推断领域的发展。本综述探讨了因果推断框架与LLMs之间的相互作用，强调了它们的集体作用。

    arXiv:2403.09606v1 Announce Type: cross  Abstract: Causal inference has shown potential in enhancing the predictive accuracy, fairness, robustness, and explainability of Natural Language Processing (NLP) models by capturing causal relationships among variables. The emergence of generative Large Language Models (LLMs) has significantly impacted various NLP domains, particularly through their advanced reasoning capabilities. This survey focuses on evaluating and improving LLMs from a causal view in the following areas: understanding and improving the LLMs' reasoning capacity, addressing fairness and safety issues in LLMs, complementing LLMs with explanations, and handling multimodality. Meanwhile, LLMs' strong reasoning capacities can in turn contribute to the field of causal inference by aiding causal relationship discovery and causal effect estimations. This review explores the interplay between causal inference frameworks and LLMs from both perspectives, emphasizing their collective p
    
[^5]: 大型语言模型是对比推理者

    Large Language Models are Contrastive Reasoners

    [https://arxiv.org/abs/2403.08211](https://arxiv.org/abs/2403.08211)

    对比提示方法显著提高大型语言模型进行复杂推理的能力，不仅在算术、常识和符号推理任务上表现优良，还可以与现有提示方法整合，实现更好的性能。

    

    提示方法在增强预训练大型语言模型（LLMs）的能力方面发挥着至关重要的作用。我们探讨了对比提示（CP）如何显著提高大型语言模型执行复杂推理的能力。我们通过简单地在LLMs提供答案之前添加"让我们给出一个正确答案和一个错误答案"来演示LLMs是体面的对比推理者。对两个大型语言模型的实验表明，零迁移对比提示提升了在一系列算术、常识和符号推理任务上的表现，而不需要手工制作的少量迁移示例，比如使用最先进的GPT-4模型，提高了在GSM8K上的准确率从35.9%到88.8%以及AQUA-RAT从41.3%到62.2%。我们的方法不仅在大多数算术和常识推理任务中胜过零迁移CoT和少量迁移CoT，还可以与现有的提示方法无缝整合，从而实现改进或者竞争

    arXiv:2403.08211v1 Announce Type: cross  Abstract: Prompting methods play a crucial role in enhancing the capabilities of pre-trained large language models (LLMs). We explore how contrastive prompting (CP) significantly improves the ability of large language models to perform complex reasoning. We demonstrate that LLMs are decent contrastive reasoners by simply adding "Let's give a correct and a wrong answer." before LLMs provide answers. Experiments on two large language models show that zero-shot contrastive prompting improves performance on a range of arithmetic, commonsense, and symbolic reasoning tasks without any hand-crafted few-shot examples, such as increasing the accuracy on GSM8K from 35.9% to 88.8% and AQUA-RAT from 41.3% to 62.2% with the state-of-the-art GPT-4 model. Our method not only surpasses zero-shot CoT and few-shot CoT in most arithmetic and commonsense reasoning tasks but also can seamlessly integrate with existing prompting methods, resulting in improved or comp
    
[^6]: 您需要更好地关注付费

    You Need to Pay Better Attention

    [https://arxiv.org/abs/2403.01643](https://arxiv.org/abs/2403.01643)

    提出了三种新的注意力机制，在效率和学习能力方面优于标准的多头注意力，提高了Transformer模型的性能和更广泛的部署能力。

    

    我们引入了三种新的注意力机制，这些机制在效率和学习能力方面胜过标准的多头注意力，从而提高了Transformer模型的性能和更广泛的部署能力。我们的第一个贡献是优化注意力，其性能与标准注意力相似，但参数数量少了四分之三，每个头部少了一个矩阵乘法。接下来，我们引入了高效注意力，其性能与标准注意力相当，但参数数量减少了一半，每个头部减少了两个矩阵乘法，并且比标准注意力快两倍。最后，我们介绍了超级注意力，在视觉和自然语言处理任务中明显超越了标准注意力，同时具有更少的参数和矩阵乘法。除了提供严格的数学比较，我们在MN中评估了所提出的注意力机制

    arXiv:2403.01643v1 Announce Type: cross  Abstract: We introduce three new attention mechanisms that outperform standard multi-head attention in terms of efficiency and learning capabilities, thereby improving the performance and broader deployability of Transformer models. Our first contribution is Optimised Attention, which performs similarly to standard attention, but has 3/4 as many parameters and one matrix multiplication fewer per head. Next, we introduce Efficient Attention, which performs on par with standard attention with only 1/2 as many parameters as many parameters and two matrix multiplications fewer per head and is up to twice as fast as standard attention. Lastly, we introduce Super Attention, which surpasses standard attention by a significant margin in both vision and natural language processing tasks while having fewer parameters and matrix multiplications. In addition to providing rigorous mathematical comparisons, we evaluate the presented attention mechanisms on MN
    
[^7]: 语言模型中的偏见：超越技巧测试，走向RUTEd评估

    Bias in Language Models: Beyond Trick Tests and Toward RUTEd Evaluation

    [https://arxiv.org/abs/2402.12649](https://arxiv.org/abs/2402.12649)

    这项研究探讨了语言模型中偏见的负面影响，研究了"技巧测试"与更现实世界中表现的RUTEd评估之间的关联性，特别关注性别-职业偏见，并进行了多项评估比较。

    

    Bias benchmarks are a popular method for studying the negative impacts of bias in LLMs, yet there has been little empirical investigation of whether these benchmarks are actually indicative of how real world harm may manifest in the real world. In this work, we study the correspondence between such decontextualized "trick tests" and evaluations that are more grounded in Realistic Use and Tangible {Effects (i.e. RUTEd evaluations). We explore this correlation in the context of gender-occupation bias--a popular genre of bias evaluation. We compare three de-contextualized evaluations adapted from the current literature to three analogous RUTEd evaluations applied to long-form content generation. We conduct each evaluation for seven instruction-tuned LLMs. For the RUTEd evaluations, we conduct repeated trials of three text generation tasks: children's bedtime stories, user personas, and English language learning exercises. We found no corres

    arXiv:2402.12649v1 Announce Type: new  Abstract: Bias benchmarks are a popular method for studying the negative impacts of bias in LLMs, yet there has been little empirical investigation of whether these benchmarks are actually indicative of how real world harm may manifest in the real world. In this work, we study the correspondence between such decontextualized "trick tests" and evaluations that are more grounded in Realistic Use and Tangible {Effects (i.e. RUTEd evaluations). We explore this correlation in the context of gender-occupation bias--a popular genre of bias evaluation. We compare three de-contextualized evaluations adapted from the current literature to three analogous RUTEd evaluations applied to long-form content generation. We conduct each evaluation for seven instruction-tuned LLMs. For the RUTEd evaluations, we conduct repeated trials of three text generation tasks: children's bedtime stories, user personas, and English language learning exercises. We found no corres
    
[^8]: 架起因果发现与大型语言模型之间的桥梁：整合方法和未来方向的综合调查

    Bridging Causal Discovery and Large Language Models: A Comprehensive Survey of Integrative Approaches and Future Directions

    [https://arxiv.org/abs/2402.11068](https://arxiv.org/abs/2402.11068)

    本文综合调查了将大型语言模型（如GPT4）整合到因果发现任务中的方法，揭示了它们在推断因果结构时对元数据和自然语言的创新利用，强调了LLMs在增强传统CD方法和作为专家辅助方面的潜力和挑战。

    

    因果发现（CD）和大型语言模型（LLMs）代表着两个具有重要影响力的人工智能研究领域。尽管它们起源不同，CD侧重于从数据中揭示因果关系，LLMs则侧重于处理和生成类似人类的文本，但这两个领域的融合为理解复杂系统提供了新颖的见解和方法论。本文介绍了将LLMs（如GPT4）整合到CD任务中的综合调查。我们系统地审查和比较了利用LLMs进行各种CD任务的现有方法，并突出了它们对元数据和自然语言的创新利用以推断因果结构。我们的分析揭示了LLMs在增强传统CD方法和作为不完美专家方面的优势和潜力，同时也揭示了当前实践中固有的挑战和限制。此外，我们确定了文献中的空白。

    arXiv:2402.11068v1 Announce Type: cross  Abstract: Causal discovery (CD) and Large Language Models (LLMs) represent two emerging fields of study with significant implications for artificial intelligence. Despite their distinct origins, CD focuses on uncovering cause-effect relationships from data, and LLMs on processing and generating humanlike text, the convergence of these domains offers novel insights and methodologies for understanding complex systems. This paper presents a comprehensive survey of the integration of LLMs, such as GPT4, into CD tasks. We systematically review and compare existing approaches that leverage LLMs for various CD tasks and highlight their innovative use of metadata and natural language to infer causal structures. Our analysis reveals the strengths and potential of LLMs in both enhancing traditional CD methods and as an imperfect expert, alongside the challenges and limitations inherent in current practices. Furthermore, we identify gaps in the literature 
    
[^9]: 语言扩展：LLMs，ChatGPT，接地，意义和理解

    Language Writ Large: LLMs, ChatGPT, Grounding, Meaning and Understanding

    [https://arxiv.org/abs/2402.02243](https://arxiv.org/abs/2402.02243)

    ChatGPT在LLM规模上通过利用语言本身的收敛约束来做到超出预期的表现，但并不真正理解语义以及与感觉动作的直接联系。

    

    除了OpenAI可能对我们隐瞒的少量信息外，我们都大致知道ChatGPT是如何工作的（它的大型文本数据库，统计数据，向量表示以及它巨大的参数数量，其下一个词的训练等）。但我们谁也不能说我们对ChatGPT的这些资源所能做到的事情不感到惊讶。这甚至让我们有人得出结论，ChatGPT实际上理解了。它并不理解，但我们也不能说我们理解它是如何做到这一点的。我将提出关于良性偏见的一些猜想：在LLM规模上出现的收敛约束可能有助于ChatGPT做得比我们预期的好得多。这些偏见是语言本身在LLM规模上固有的，并且与ChatGPT缺乏直接的感觉动作接地以将其词与其所指的对象以及其命题与其意义联系起来密切相关。

    Apart from what (little) OpenAI may be concealing from us, we all know (roughly) how ChatGPT works (its huge text database, its statistics, its vector representations, and their huge number of parameters, its next-word training, and so on). But none of us can say (hand on heart) that we are not surprised by what ChatGPT has proved to be able to do with these resources. This has even driven some of us to conclude that ChatGPT actually understands. It is not true that it understands. But it is also not true that we understand how it can do what it can do. I will suggest some hunches about benign biases: convergent constraints that emerge at LLM scale that may be helping ChatGPT do so much better than we would have expected. These biases are inherent in the nature of language itself, at LLM scale, and they are closely linked to what it is that ChatGPT lacks, which is direct sensorimotor grounding to connect its words to their referents and its propositions to their meanings. These converg
    
[^10]: 为社交感知的谈判对话开发辅助大型语言模型代理

    Assistive Large Language Model Agents for Socially-Aware Negotiation Dialogues

    [https://arxiv.org/abs/2402.01737](https://arxiv.org/abs/2402.01737)

    本文旨在开发LLM代理以减轻多代理设置下谈判中的社交规范违反。我们引入了基于价值影响的环境学习（ICL）方法，用于为基于LLM的修正代理识别高质量的ICL示例，其中价值影响函数衡量谈判结果的质量。我们展示了这种方法与策略学习的联系，并提供了丰富的实证证据来证明其在三个不同主题的谈判中的有效性。

    

    本文旨在开发LLM代理以减轻多代理设置下谈判中的社交规范违反。我们通过让两个大型语言模型（LLM）扮演每次对话中的两名谈判者来模拟现实世界谈判。第三个LLM充当修正代理，重新编写违反规范的话语以改善谈判结果。由于这是一个新颖的任务，不存在手动构建的数据。为解决这个限制，我们引入了基于价值影响的环境学习（ICL）方法，用于为基于LLM的修正代理识别高质量的ICL示例，其中价值影响函数衡量谈判结果的质量。我们展示了这种方法与策略学习的联系，并提供了丰富的实证证据来证明其在三个不同主题的谈判中的有效性，即产品销售、房价和薪资谈判。源代码和生成的数据集将在接受后公开。

    In this work, we aim to develop LLM agents to mitigate social norm violations in negotiations in a multi-agent setting. We simulate real-world negotiations by letting two large Language Models (LLMs) play the roles of two negotiators in each conversation. A third LLM acts as a remediation agent to rewrite utterances violating norms for improving negotiation outcomes. As it is a novel task, no manually constructed data is available. To address this limitation, we introduce a value impact based In-Context Learning (ICL) method to identify high-quality ICL examples for the LLM-based remediation agents, where the value impact function measures the quality of negotiation outcomes. We show the connection of this method to policy learning and provide rich empirical evidence to demonstrate its effectiveness in negotiations across three different topics: product sale, housing price, and salary negotiation. The source code and the generated dataset will be publicly available upon acceptance.
    
[^11]: SWEA:通过主题词嵌入修改改变大型语言模型中的事实知识

    SWEA: Changing Factual Knowledge in Large Language Models via Subject Word Embedding Altering

    [https://arxiv.org/abs/2401.17809](https://arxiv.org/abs/2401.17809)

    提出了一种主题词嵌入修改框架（SWEA），通过在推理阶段修改主题的表示来编辑知识，保护模型的原始权重，避免不可逆的损害和额外的推理开销。

    

    模型编辑近来引起了广泛关注。目前的模型编辑方法主要涉及修改模型参数或向现有模型添加附加模块。然而，前者会对LLM造成不可逆的影响，而后者会产生额外的推理开销，并且模糊的向量匹配并不总是可靠的。为了解决这些问题，我们提出了一种可扩展的主题词嵌入修改（SWEA）框架，它在推理阶段修改主题的表示，并实现编辑知识的目标。SWEA在模型外部使用精确的关键匹配，并进行可靠的主题词嵌入修改，从而保护模型的原始权重而不增加推理开销。然后，我们提出优化抑制融合方法，首先优化编辑目标的嵌入向量，然后抑制知识嵌入维度（KED）以获得最终融合的嵌入。我们因此提出了SWEAOS元方法。

    Model editing has recently gained widespread attention. Current model editing methods primarily involve modifying model parameters or adding additional modules to the existing model. However, the former causes irreversible damage to LLMs, while the latter incurs additional inference overhead and fuzzy vector matching is not always reliable. To address these issues, we propose an expandable Subject Word Embedding Altering (SWEA) framework, which modifies the representation of subjects and achieve the goal of editing knowledge during the inference stage. SWEA uses precise key matching outside the model and performs reliable subject word embedding altering, thus protecting the original weights of the model without increasing inference overhead. We then propose optimizing then suppressing fusion method, which first optimizes the embedding vector for the editing target and then suppresses the Knowledge Embedding Dimension (KED) to obtain the final fused embedding. We thus propose SWEAOS met
    
[^12]: 在跨语言转移范式中测量灾难性遗忘：探索调优策略

    Measuring Catastrophic Forgetting in Cross-Lingual Transfer Paradigms: Exploring Tuning Strategies. (arXiv:2309.06089v1 [cs.CL])

    [http://arxiv.org/abs/2309.06089](http://arxiv.org/abs/2309.06089)

    该研究比较了不同的微调和跨语言转移策略在解决跨语言任务时的表现，评估了灾难性遗忘的程度和转移的成功程度。

    

    跨语言转移是一种解决资源匮乏语言任务的有希望的技术。在这个实证研究中，我们比较了两种与零射和全射学习方法相结合的大型语言模型在跨语言设置下的微调方法。作为微调策略，我们比较了参数效率适配器方法与所有参数微调。作为跨语言转移策略，我们比较了使用每个语言依次的中间训练（IT）和在微调的验证阶段已经使用目标语言的跨语言验证（CLV）。我们评估了转移的成功程度以及源语言中由于跨语言转移而导致的灾难性遗忘的程度，即在学习不同语言中的新信息时之前获得的知识损失了多少。在两个不同的分类问题上，包括仇恨言论检测和产品评论，分别包含了多个语种数据集的结果。

    The cross-lingual transfer is a promising technique to solve tasks in less-resourced languages. In this empirical study, we compare two fine-tuning approaches combined with zero-shot and full-shot learning approaches for large language models in a cross-lingual setting. As fine-tuning strategies, we compare parameter-efficient adapter methods with fine-tuning of all parameters. As cross-lingual transfer strategies, we compare the intermediate-training (\textit{IT}) that uses each language sequentially and cross-lingual validation (\textit{CLV}) that uses a target language already in the validation phase of fine-tuning. We assess the success of transfer and the extent of catastrophic forgetting in a source language due to cross-lingual transfer, i.e., how much previously acquired knowledge is lost when we learn new information in a different language. The results on two different classification problems, hate speech detection and product reviews, each containing datasets in several lang
    

