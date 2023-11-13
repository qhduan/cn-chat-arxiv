# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [FlashDecoding++: Faster Large Language Model Inference on GPUs.](http://arxiv.org/abs/2311.01282) | FlashDecoding++是一种快速的LLM推理引擎，通过解决同步部分softmax更新、未充分利用扁平GEMM计算和静态数据流导致的性能损失等挑战，实现了大规模语言模型推理的加速。 |
| [^2] | [Chinesewebtext: Large-scale high-quality Chinese web text extracted with effective evaluation model.](http://arxiv.org/abs/2311.01149) | 本文提出了一个完整的工具链EvalWeb，用于从网络数据中提取干净的中文文本。通过手工制定的规则筛除噪音数据，并使用评估模型为每个文本分配质量分数。 |
| [^3] | [InfoEntropy Loss to Mitigate Bias of Learning Difficulties for Generative Language Models.](http://arxiv.org/abs/2310.19531) | 提出了一种信息熵损失函数，用于减少生成式语言模型对常见和易学标记的偏好，使其更关注不常见和难学的标记。 |
| [^4] | [Rosetta Stone at KSAA-RD Shared Task: A Hop From Language Modeling To Word--Definition Alignment.](http://arxiv.org/abs/2310.15823) | 本论文介绍了在KSAA-RD共享任务中Rosetta Stone的应用，将语言建模应用到词--定义对齐中。论文通过使用一组微调的阿拉伯BERT模型来预测给定定义的词嵌入，从而实现了阿拉伯词的向量表示。 |
| [^5] | [Conversational Financial Information Retrieval Model (ConFIRM).](http://arxiv.org/abs/2310.13001) | ConFIRM是一种会话式金融信息检索模型，通过合成金融领域特定问答对和评估参数微调方法，实现了超过90%的准确性，为金融对话系统提供了数据高效的解决方案。 |
| [^6] | [BRAINTEASER: Lateral Thinking Puzzles for Large Language Models.](http://arxiv.org/abs/2310.05057) | 本文介绍了一项名为BRAINTEASER的多项选择问题回答任务，旨在测试大型语言模型表现出横向思维和违反默认常识联系的能力。通过创建一个横向思维基准，丰富问题的语义和上下文重建，实验证明模型与人类之间存在显著差距。 |
| [^7] | [Wordification: A New Way of Teaching English Spelling Patterns.](http://arxiv.org/abs/2309.12981) | Wordification是一种新的教授英语拼写模式的方法，旨在解决全球范围内存在的识字问题，尤其是在青少年犯罪和教育领域。 |
| [^8] | [Retrieval-based Text Selection for Addressing Class-Imbalanced Data in Classification.](http://arxiv.org/abs/2307.14899) | 本文解决了在文本分类中选择有限数量的文本进行注释的问题，并提出了利用检索方法和语义搜索来处理类别不平衡的二元分类问题。通过利用SHAP构建高质量的查询集，帮助选择适合注释的文本，以解决长期注释任务中的挑战。 |
| [^9] | [M3Exam: A Multilingual, Multimodal, Multilevel Benchmark for Examining Large Language Models.](http://arxiv.org/abs/2306.05179) | M3Exam是一个来源于真实人类考试题目的新型基准测试，用于评估大型语言模型在多语言、多模态和多层次的情境中的普适智能。 |
| [^10] | [Flexible Grammar-Based Constrained Decoding for Language Models.](http://arxiv.org/abs/2305.13971) | 本文提出了一种使用形式语法约束丰富解码步骤的方法，有效生成符合特定语法的复杂输出结构，同时允许任何上下文无关语法集成。实验证明该方法在四个信息提取任务上实现了最先进的性能表现。 |
| [^11] | [Visual Chain of Thought: Bridging Logical Gaps with Multimodal Infillings.](http://arxiv.org/abs/2305.02317) | VCoT是一种使用思维链激励和视觉语言组合递归地弥合时序数据中逻辑差距的新颖方法，其使用视觉引导生成合成的多模态填充以添加一致且新颖的信息，并减少需要时序推理的逻辑差距。 |
| [^12] | [Prompt as Triggers for Backdoor Attack: Examining the Vulnerability in Language Models.](http://arxiv.org/abs/2305.01219) | 本研究提出一种新颖有效的“ProAttack”方法来执行干净标签的后门攻击，使用的是提示本身作为触发器。该方法不需要外部触发器，并确保毒瘤数据的标注正确，提高了后门攻击的隐蔽性，相比于现有的后门攻击方法有显著提升。 |
| [^13] | [Behavioral estimates of conceptual structure are robust across tasks in humans but not large language models.](http://arxiv.org/abs/2304.02754) | 本研究使用两种经典认知心理学技术来估算人类和GPT-3等大型语言模型的词汇语义结构，结果表明人类的概念结构稳健鲁棒，而大型语言模型的行为估算结构更多取决于具体任务。 |
| [^14] | [E2E Spoken Entity Extraction for Virtual Agents.](http://arxiv.org/abs/2302.10186) | 本文研究了利用预训练语音编码器从语音中直接提取实体的方法，无需文本转录，且在口语实体识别任务中表现优异。 |

# 详细

[^1]: FlashDecoding++: 在GPU上加速大规模语言模型推理的更快算法

    FlashDecoding++: Faster Large Language Model Inference on GPUs. (arXiv:2311.01282v1 [cs.LG])

    [http://arxiv.org/abs/2311.01282](http://arxiv.org/abs/2311.01282)

    FlashDecoding++是一种快速的LLM推理引擎，通过解决同步部分softmax更新、未充分利用扁平GEMM计算和静态数据流导致的性能损失等挑战，实现了大规模语言模型推理的加速。

    

    随着大规模语言模型在各个领域的重要性日益增加，加速语言模型推理仍然存在一些挑战未解决：(1) 同步部分softmax更新。softmax操作需要同步更新每个部分softmax结果，导致LLM中注意力计算的开销增加约20%。(2) 未充分利用扁平GEMM计算。在LLM推理中执行GEMM的矩阵形状是扁平的，导致在先前的设计中填充零后计算未充分利用，性能损失超过50%。(3) 静态数据流导致的性能损失。LLM中的内核性能取决于不同的输入数据特征、硬件配置等。单一和静态的数据流可能导致LLM推理中不同形状的GEMM的性能损失达到50.25%。我们提出了FlashDecoding++，一种快速支持主流LLM和硬件后端的LLM推理引擎。为了解决上述挑战，FlashDecoding++实现了以下目标：

    As the Large Language Model (LLM) becomes increasingly important in various domains. However, the following challenges still remain unsolved in accelerating LLM inference: (1) Synchronized partial softmax update. The softmax operation requires a synchronized update operation among each partial softmax result, leading to ~20% overheads for the attention computation in LLMs. (2) Under-utilized computation of flat GEMM. The shape of matrices performing GEMM in LLM inference is flat, leading to under-utilized computation and >50% performance loss after padding zeros in previous designs. (3) Performance loss due to static dataflow. Kernel performance in LLM depends on varied input data features, hardware configurations, etc. A single and static dataflow may lead to a 50.25% performance loss for GEMMs of different shapes in LLM inference.  We present FlashDecoding++, a fast LLM inference engine supporting mainstream LLMs and hardware back-ends. To tackle the above challenges, FlashDecoding++
    
[^2]: Chinesewebtext: 用有效的评估模型提取大规模高质量的中文网络文本

    Chinesewebtext: Large-scale high-quality Chinese web text extracted with effective evaluation model. (arXiv:2311.01149v1 [cs.CL])

    [http://arxiv.org/abs/2311.01149](http://arxiv.org/abs/2311.01149)

    本文提出了一个完整的工具链EvalWeb，用于从网络数据中提取干净的中文文本。通过手工制定的规则筛除噪音数据，并使用评估模型为每个文本分配质量分数。

    

    在大型语言模型（LLM）的发展过程中，预训练数据的规模和质量对于塑造LLM的能力起着至关重要的作用。为了加快LLM的研究进展，已经发布了一些大规模数据集，例如C4 [1]、Pile [2]、RefinedWeb [3]和WanJuan [4]等。然而，大多数已发布的语料库主要关注英文，仍然缺乏完整的工具链来从网络数据中提取出干净的文本。此外，缺乏对语料库的细粒度信息，例如每个文本的质量。为了解决这些挑战，本文提出了一个新的完整的工具链EvalWeb，用于从嘈杂的网络数据中提取中文干净的文本。首先，类似之前的工作，使用手工制定的规则来丢弃原始爬取的网络内容中的明确嘈杂的文本。然后，利用一个精心设计的评估模型来评估剩余相对干净的数据，并为每个文本分配一个特定的质量分数。最后，我们进行了大规模的实验，验证了EvalWeb工具链的有效性。

    During the development of large language models (LLMs), the scale and quality of the pre-training data play a crucial role in shaping LLMs' capabilities. To accelerate the research of LLMs, several large-scale datasets, such as C4 [1], Pile [2], RefinedWeb [3] and WanJuan [4], have been released to the public. However, most of the released corpus focus mainly on English, and there is still lack of complete tool-chain for extracting clean texts from web data. Furthermore, fine-grained information of the corpus, e.g. the quality of each text, is missing. To address these challenges, we propose in this paper a new complete tool-chain EvalWeb to extract Chinese clean texts from noisy web data. First, similar to previous work, manually crafted rules are employed to discard explicit noisy texts from the raw crawled web contents. Second, a well-designed evaluation model is leveraged to assess the remaining relatively clean data, and each text is assigned a specific quality score. Finally, we 
    
[^3]: 减少生成式语言模型学习困难的信息熵损失

    InfoEntropy Loss to Mitigate Bias of Learning Difficulties for Generative Language Models. (arXiv:2310.19531v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2310.19531](http://arxiv.org/abs/2310.19531)

    提出了一种信息熵损失函数，用于减少生成式语言模型对常见和易学标记的偏好，使其更关注不常见和难学的标记。

    

    生成式语言模型通常通过预测上一个标记（子词/词/短语）给出的下一个标记来进行预训练。最近的研究展示了大规模生成式语言模型在下游任务上的出色性能。然而，现有的生成式语言模型在训练过程中通常忽视文本语料库中的固有挑战，即频繁标记和不经常出现的标记之间的不平衡。这可能导致语言模型被常见且易学的标记所主导，从而忽视不经常出现且难以学习的标记。为了缓解这个问题，我们提出了一种信息熵损失（InfoEntropy Loss）函数。在训练过程中，它可以根据相应的预测概率分布的信息熵动态评估待学习标记的学习难度。然后，它适应地调整训练损失，试图使模型更加关注难以学习的标记。

    Generative language models are usually pretrained on large text corpus via predicting the next token (i.e., sub-word/word/phrase) given the previous ones. Recent works have demonstrated the impressive performance of large generative language models on downstream tasks. However, existing generative language models generally neglect an inherent challenge in text corpus during training, i.e., the imbalance between frequent tokens and infrequent ones. It can lead a language model to be dominated by common and easy-to-learn tokens, thereby overlooking the infrequent and difficult-to-learn ones. To alleviate that, we propose an Information Entropy Loss (InfoEntropy Loss) function. During training, it can dynamically assess the learning difficulty of a to-be-learned token, according to the information entropy of the corresponding predicted probability distribution over the vocabulary. Then it scales the training loss adaptively, trying to lead the model to focus more on the difficult-to-learn
    
[^4]: Rosetta Stone在KSAA-RD共享任务中：从语言建模到词--定义对齐的跃进。

    Rosetta Stone at KSAA-RD Shared Task: A Hop From Language Modeling To Word--Definition Alignment. (arXiv:2310.15823v1 [cs.CL])

    [http://arxiv.org/abs/2310.15823](http://arxiv.org/abs/2310.15823)

    本论文介绍了在KSAA-RD共享任务中Rosetta Stone的应用，将语言建模应用到词--定义对齐中。论文通过使用一组微调的阿拉伯BERT模型来预测给定定义的词嵌入，从而实现了阿拉伯词的向量表示。

    

    反向词典是一种工具，可根据提供的定义、含义或描述来发现一个词。这种技术在各种场景中都非常有价值，可以帮助掌握一个词的描述而不知其身份的语言学习者，并使寻求精确术语的写作者受益。这些场景通常涵盖被称为“舌尖上的词”现象。在这项工作中，我们呈现了我们在阿拉伯语反向词典共享任务中获胜的解决方案。该任务的重点是从伴随的描述中推导出阿拉伯词的向量表示。共享任务包括两个不同的子任务：第一个子任务涉及一个阿拉伯定义作为输入，而第二个子任务则使用一个英文定义。对于第一个子任务，我们的方法依赖于一组经过微调的阿拉伯BERT模型，来预测给定定义的词嵌入。最终表示是通过对每个模型输出的嵌入进行平均得到的。

    A Reverse Dictionary is a tool enabling users to discover a word based on its provided definition, meaning, or description. Such a technique proves valuable in various scenarios, aiding language learners who possess a description of a word without its identity, and benefiting writers seeking precise terminology. These scenarios often encapsulate what is referred to as the "Tip-of-the-Tongue" (TOT) phenomena. In this work, we present our winning solution for the Arabic Reverse Dictionary shared task. This task focuses on deriving a vector representation of an Arabic word from its accompanying description. The shared task encompasses two distinct subtasks: the first involves an Arabic definition as input, while the second employs an English definition. For the first subtask, our approach relies on an ensemble of finetuned Arabic BERT-based models, predicting the word embedding for a given definition. The final representation is obtained through averaging the output embeddings from each m
    
[^5]: 会话式金融信息检索模型（ConFIRM）

    Conversational Financial Information Retrieval Model (ConFIRM). (arXiv:2310.13001v1 [cs.IR])

    [http://arxiv.org/abs/2310.13001](http://arxiv.org/abs/2310.13001)

    ConFIRM是一种会话式金融信息检索模型，通过合成金融领域特定问答对和评估参数微调方法，实现了超过90%的准确性，为金融对话系统提供了数据高效的解决方案。

    

    随着大型语言模型（LLM）的指数级增长，利用它们在金融等专门领域的新兴特性具有探索的价值。然而，金融等受监管领域具有独特的约束条件，需要具备针对该领域的优化框架。我们提出了ConFIRM，一种基于LLM的会话式金融信息检索模型，用于查询意图分类和知识库标记。ConFIRM包括两个模块：1）一种合成金融领域特定问答对的方法，以及2）评估参数高效的微调方法来进行查询分类任务。我们生成了一个包含4000多个样本的数据集，并在单独的测试集上评估了准确性。ConFIRM实现了超过90%的准确性，这对于符合监管要求至关重要。ConFIRM提供了一种数据高效的解决方案，用于提取金融对话系统的精确查询意图。

    With the exponential growth in large language models (LLMs), leveraging their emergent properties for specialized domains like finance merits exploration. However, regulated fields such as finance pose unique constraints, requiring domain-optimized frameworks. We present ConFIRM, an LLM-based conversational financial information retrieval model tailored for query intent classification and knowledge base labeling.  ConFIRM comprises two modules:  1) a method to synthesize finance domain-specific question-answer pairs, and  2) evaluation of parameter efficient fine-tuning approaches for the query classification task. We generate a dataset of over 4000 samples, assessing accuracy on a separate test set.  ConFIRM achieved over 90% accuracy, essential for regulatory compliance. ConFIRM provides a data-efficient solution to extract precise query intent for financial dialog systems.
    
[^6]: BRAINTEASER：大型语言模型的横向思维难题

    BRAINTEASER: Lateral Thinking Puzzles for Large Language Models. (arXiv:2310.05057v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2310.05057](http://arxiv.org/abs/2310.05057)

    本文介绍了一项名为BRAINTEASER的多项选择问题回答任务，旨在测试大型语言模型表现出横向思维和违反默认常识联系的能力。通过创建一个横向思维基准，丰富问题的语义和上下文重建，实验证明模型与人类之间存在显著差距。

    

    语言模型的成功激励了自然语言处理社区关注需要隐含和复杂推理的任务，依赖于类人的常识机制。虽然这些垂直思维任务相对较受欢迎，但横向思维难题却鲜有关注。为了弥合这一差距，我们设计了BRAINTEASER：一个多项选择问题回答任务，旨在测试模型表现出横向思维和违反默认常识联系的能力。我们设计了一个三步骤的程序来创建第一个横向思维基准，包括数据收集、干扰项生成和对抗性样本生成，共有1,100个具有高质量注释的难题。为了评估模型的横向推理一致性，我们基于问题的语义和上下文重建来丰富BRAINTEASER。我们对最先进的指导性和常识语言模型进行的实验揭示了人类和模型之间的显著差距。

    The success of language models has inspired the NLP community to attend to tasks that require implicit and complex reasoning, relying on human-like commonsense mechanisms. While such vertical thinking tasks have been relatively popular, lateral thinking puzzles have received little attention. To bridge this gap, we devise BRAINTEASER: a multiple-choice Question Answering task designed to test the model's ability to exhibit lateral thinking and defy default commonsense associations. We design a three-step procedure for creating the first lateral thinking benchmark, consisting of data collection, distractor generation, and generation of adversarial examples, leading to 1,100 puzzles with high-quality annotations. To assess the consistency of lateral reasoning by models, we enrich BRAINTEASER based on a semantic and contextual reconstruction of its questions. Our experiments with state-of-the-art instruction- and commonsense language models reveal a significant gap between human and model
    
[^7]: Wordification:一种新的教授英语拼写模式的方法

    Wordification: A New Way of Teaching English Spelling Patterns. (arXiv:2309.12981v1 [cs.OH])

    [http://arxiv.org/abs/2309.12981](http://arxiv.org/abs/2309.12981)

    Wordification是一种新的教授英语拼写模式的方法，旨在解决全球范围内存在的识字问题，尤其是在青少年犯罪和教育领域。

    

    读写能力是生活和社会中成功的关键指标。据估计， 85% 的未成年犯罪系统的人无法足够地阅读和书写，超过一半有滥用物质问题的人在阅读或书写方面有困难，而未完成高中学业的三分之二缺乏适当的读写能力。此外，不具备与四年级匹配的阅读能力的幼儿大约有80%的可能性根本无法赶上。许多人可能认为在发达国家如美国，识字能力不再是一个问题；然而，这是一种危险的误解。全球每年因识字问题损失约1.19万亿美元；在美国，损失约为3000亿美元。更糟糕的是，现在唯一的工具是

    Literacy, or the ability to read and write, is a crucial indicator of success in life and greater society. It is estimated that 85% of people in juvenile delinquent systems cannot adequately read or write, that more than half of those with substance abuse issues have complications in reading or writing and that two-thirds of those who do not complete high school lack proper literacy skills. Furthermore, young children who do not possess reading skills matching grade level by the fourth grade are approximately 80% likely to not catch up at all. Many may believe that in a developed country such as the United States, literacy fails to be an issue; however, this is a dangerous misunderstanding. Globally an estimated 1.19 trillion dollars are lost every year due to issues in literacy; in the USA, the loss is an estimated 300 billion. To put it in more shocking terms, one in five American adults still fail to comprehend basic sentences. Making matters worse, the only tools available now to c
    
[^8]: 为解决分类中的类别不平衡问题，基于检索的文本选择方法

    Retrieval-based Text Selection for Addressing Class-Imbalanced Data in Classification. (arXiv:2307.14899v1 [cs.CL])

    [http://arxiv.org/abs/2307.14899](http://arxiv.org/abs/2307.14899)

    本文解决了在文本分类中选择有限数量的文本进行注释的问题，并提出了利用检索方法和语义搜索来处理类别不平衡的二元分类问题。通过利用SHAP构建高质量的查询集，帮助选择适合注释的文本，以解决长期注释任务中的挑战。

    

    本文解决了在文本分类中使用检索方法选择一组文本进行注释的问题，由于人力资源的限制，注释的数量有限。同时，我们还解决了二元类别的不平衡问题，即正样本数量较少的情况。在我们的情况下，注释是在较长的时间段内进行的，可以将要注释的文本批量选择，以前面的注释来指导下一组的选择。为了解决这些挑战，本文提出了利用SHAP构建Elasticsearch和语义搜索的高质量查询集，以尝试识别一组最优文本来帮助解决类别不平衡问题。该方法在描述可能的未来事件的提纲文本集上进行了测试，该文本集由参与肥胖和糖尿病管理研究的参与者构建。我们介绍一种有效的方法

    This paper addresses the problem of selecting of a set of texts for annotation in text classification using retrieval methods when there are limits on the number of annotations due to constraints on human resources. An additional challenge addressed is dealing with binary categories that have a small number of positive instances, reflecting severe class imbalance. In our situation, where annotation occurs over a long time period, the selection of texts to be annotated can be made in batches, with previous annotations guiding the choice of the next set. To address these challenges, the paper proposes leveraging SHAP to construct a quality set of queries for Elasticsearch and semantic search, to try to identify optimal sets of texts for annotation that will help with class imbalance. The approach is tested on sets of cue texts describing possible future events, constructed by participants involved in studies aimed to help with the management of obesity and diabetes. We introduce an effec
    
[^9]: M3Exam: 一种多语言、多模态、多层次的基准测试，用于评估大型语言模型

    M3Exam: A Multilingual, Multimodal, Multilevel Benchmark for Examining Large Language Models. (arXiv:2306.05179v1 [cs.CL])

    [http://arxiv.org/abs/2306.05179](http://arxiv.org/abs/2306.05179)

    M3Exam是一个来源于真实人类考试题目的新型基准测试，用于评估大型语言模型在多语言、多模态和多层次的情境中的普适智能。

    

    尽管存在着各种针对自然语言处理模型进行评估的基准测试，但我们认为考试更适合评估大型语言模型的普适智能，因为它们囊括了更广泛的能力需求，例如语言理解、领域知识和解决问题的能力。为此，我们引入了 M3Exam，这是一个基于真实和官方人类考试题目的新型基准测试，用于在多语言、多模态和多层次的情境中评估 LLM。M3Exam 具有三个独特特点:（1）多语言性，涵盖多个国家的问题，需要强大的多语言能力和文化知识；（2）多模态，考虑到许多考试问题的多模态性质，以测试模型的多模态理解能力；（3）多层次结构，特别涵盖了三个关键教育阶段的考试，全面评估模型在不同教育水平上的熟练程度。

    Despite the existence of various benchmarks for evaluating natural language processing models, we argue that human exams are a more suitable means of evaluating general intelligence for large language models (LLMs), as they inherently demand a much wider range of abilities such as language understanding, domain knowledge, and problem-solving skills. To this end, we introduce M3Exam, a novel benchmark sourced from real and official human exam questions for evaluating LLMs in a multilingual, multimodal, and multilevel context. M3Exam exhibits three unique characteristics: (1) multilingualism, encompassing questions from multiple countries that require strong multilingual proficiency and cultural knowledge; (2) multimodality, accounting for the multimodal nature of many exam questions to test the model's multimodal understanding capability; and (3) multilevel structure, featuring exams from three critical educational periods to comprehensively assess a model's proficiency at different lev
    
[^10]: 基于语法约束的语言模型灵活解码技术

    Flexible Grammar-Based Constrained Decoding for Language Models. (arXiv:2305.13971v1 [cs.CL])

    [http://arxiv.org/abs/2305.13971](http://arxiv.org/abs/2305.13971)

    本文提出了一种使用形式语法约束丰富解码步骤的方法，有效生成符合特定语法的复杂输出结构，同时允许任何上下文无关语法集成。实验证明该方法在四个信息提取任务上实现了最先进的性能表现。

    

    LLM在许多任务中展现出了惊人的少量样本表现，但在生成信息提取所需的复杂输出结构时仍存在困难。这个限制源于LLM在没有微调的情况下倾向于生成自由文本而不是遵循特定语法的精确结构。在本文中，我们提出在解码步骤中使用形式语法约束来丰富模型。在搜索过程中，只有符合语法产生规则的有效令牌能被考虑到。这样就强制只产生有效的序列。我们的框架非常通用和灵活，允许任何上下文无关语法(CFG)集成到我们的自定义约束beam搜索实现中。我们展示了许多NLP任务的输出可以被表示为形式语言，使它们适合在我们的框架中直接使用。对于输出空间取决于输入的任务，我们提出了基于输入的CFG，根据特定于输入的特征更新产生规则。实验证明了我们的方法在生成复杂输出结构方面的有效性，并在四个信息提取任务上实现了最先进的性能。

    LLMs have shown impressive few-shot performance across many tasks. However, they still struggle when it comes to generating complex output structures, such as those required for Information Extraction. This limitation stems from the fact that LLMs, without finetuning, tend to generate free text rather than precise structures that follow a specific grammar. In this work, we propose to enrich the decoding step with formal grammar constraints. During beam search, only valid token continuations compliant with the grammar production rules are considered. This enforces the generation of valid sequences exclusively. Our framework is highly general and flexible, allowing any Context-Free Grammar (CFG) to be integrated into our custom constrained beam search implementation. We demonstrate that the outputs of many NLP tasks can be represented as formal languages, making them suitable for direct use in our framework. For task where the output space is dependent on the input, we propose input-depe
    
[^11]: 视觉思维链：多模态填充技术弥合逻辑差距

    Visual Chain of Thought: Bridging Logical Gaps with Multimodal Infillings. (arXiv:2305.02317v1 [cs.CL])

    [http://arxiv.org/abs/2305.02317](http://arxiv.org/abs/2305.02317)

    VCoT是一种使用思维链激励和视觉语言组合递归地弥合时序数据中逻辑差距的新颖方法，其使用视觉引导生成合成的多模态填充以添加一致且新颖的信息，并减少需要时序推理的逻辑差距。

    

    大型自然语言模型的出现提高了模型的多步推理能力，能以人类方式分解问题。然而，该范例由于其单模态性质并且主要应用于问答任务而受到限制。我们认为将视觉增强内容纳入推理是必要的，尤其是针对复杂想象任务。因此，我们介绍了VCoT，一种新颖的方法，它利用思维链激励和视觉语言组合来递归地弥合时序数据中的逻辑差距。我们的方法使用视觉引导生成合成的多模态填充，以添加一致且新颖的信息，并减少下游任务中需要时序推理的逻辑差距，同时提供模型的多步推理的解释性。我们将VCoT应用于视觉叙事和WikiHow摘要数据集，并通过人工评估展示了其性能的提升。

    Recent advances in large language models elicit reasoning in a chain of thought that allows models to decompose problems in a human-like fashion. Though this paradigm improves multi-step reasoning ability in language models, it is limited by being unimodal and applied mainly to question-answering tasks. We claim that incorporating visual augmentation into reasoning is essential, especially for complex, imaginative tasks. Consequently, we introduce VCoT, a novel method that leverages chain of thought prompting with vision-language grounding to recursively bridge the logical gaps within sequential data. Our method uses visual guidance to generate synthetic multimodal infillings that add consistent and novel information to reduce the logical gaps for downstream tasks that can benefit from temporal reasoning, as well as provide interpretability into models' multi-step reasoning. We apply VCoT to the Visual Storytelling and WikiHow summarization datasets and demonstrate through human evalua
    
[^12]: 触发词作为后门攻击的触发器：检查语言模型的脆弱性

    Prompt as Triggers for Backdoor Attack: Examining the Vulnerability in Language Models. (arXiv:2305.01219v1 [cs.CL])

    [http://arxiv.org/abs/2305.01219](http://arxiv.org/abs/2305.01219)

    本研究提出一种新颖有效的“ProAttack”方法来执行干净标签的后门攻击，使用的是提示本身作为触发器。该方法不需要外部触发器，并确保毒瘤数据的标注正确，提高了后门攻击的隐蔽性，相比于现有的后门攻击方法有显著提升。

    

    基于提示的学习范例弥合了预训练和微调之间的差距，在几个NLP任务中取得了最先进的性能，尤其是在少样本情况下。尽管应用广泛，但基于提示的学习容易受到后门攻击。文本后门攻击旨在通过注入触发器并修改标签来在模型中引入有针对性的漏洞。然而，由于触发器的存在和毒瘤数据标注不正确等缺陷，这种攻击存在异常的自然语言表达。在本研究中，我们提出了一种新颖有效的“ProAttack”方法，基于提示来执行干净标签的后门攻击，使用的是提示本身作为触发器。我们的方法不需要外部触发器，并确保毒瘤数据的标注正确，提高了后门攻击的隐蔽性。通过在丰富的资源和少样本文本语料库上的广泛实验，我们证明了ProAttack方法在保持干净数据一致性的同时显著优于现有的后门攻击方式。

    The prompt-based learning paradigm, which bridges the gap between pre-training and fine-tuning, achieves state-of-the-art performance on several NLP tasks, particularly in few-shot settings. Despite being widely applied, prompt-based learning is vulnerable to backdoor attacks. Textual backdoor attacks are designed to introduce targeted vulnerabilities into models by poisoning a subset of training samples through trigger injection and label modification. However, they suffer from flaws such as abnormal natural language expressions resulting from the trigger and incorrect labeling of poisoned samples. In this study, we propose {\bf ProAttack}, a novel and efficient method for performing clean-label backdoor attacks based on the prompt, which uses the prompt itself as a trigger. Our method does not require external triggers and ensures correct labeling of poisoned samples, improving the stealthy nature of the backdoor attack. With extensive experiments on rich-resource and few-shot text c
    
[^13]: 人类和大型语言模型中的概念结构表现的差异性

    Behavioral estimates of conceptual structure are robust across tasks in humans but not large language models. (arXiv:2304.02754v1 [cs.AI])

    [http://arxiv.org/abs/2304.02754](http://arxiv.org/abs/2304.02754)

    本研究使用两种经典认知心理学技术来估算人类和GPT-3等大型语言模型的词汇语义结构，结果表明人类的概念结构稳健鲁棒，而大型语言模型的行为估算结构更多取决于具体任务。

    

    多年以来，神经网络语言模型一直被用作研究心理和脑部概念表征的工具。然而，在当代语言人工智能中，我们可以使用与人类参与者几乎相同的方法来探讨概念表征的潜在结构。本研究使用两种经典的认知心理学技术来估算和比较人类和一个著名的大型语言模型（GPT-3的DaVinci变体）的词汇语义结构。研究表明，人类的概念结构强大且鲁棒，不受文化、语言和估算方法的差异影响；大型语言模型中的行为估算结果相对稳定，但具体取决于任务本身。这些结果表明，虽然人类参与者的行为估算结果可靠，但在使用大型语言模型进行人类认知处理相关推断时，需要谨慎。

    Neural network models of language have long been used as a tool for developing hypotheses about conceptual representation in the mind and brain. For many years, such use involved extracting vector-space representations of words and using distances among these to predict or understand human behavior in various semantic tasks. In contemporary language AIs, however, it is possible to interrogate the latent structure of conceptual representations using methods nearly identical to those commonly used with human participants. The current work uses two common techniques borrowed from cognitive psychology to estimate and compare lexical-semantic structure in both humans and a well-known AI, the DaVinci variant of GPT-3. In humans, we show that conceptual structure is robust to differences in culture, language, and method of estimation. Structures estimated from AI behavior, while individually fairly consistent with those estimated from human behavior, depend much more upon the particular task 
    
[^14]: 虚拟代理人的端到端口语化实体提取

    E2E Spoken Entity Extraction for Virtual Agents. (arXiv:2302.10186v4 [eess.AS] UPDATED)

    [http://arxiv.org/abs/2302.10186](http://arxiv.org/abs/2302.10186)

    本文研究了利用预训练语音编码器从语音中直接提取实体的方法，无需文本转录，且在口语实体识别任务中表现优异。

    

    本文重新构想了语音处理中的一些方面，特别是关于从语音中直接提取实体，而无需中间文本表示。在人与计算机的对话中，从语音中提取实体，如姓名、邮政地址和电子邮件地址，是一项具有挑战性的任务。我们研究了微调预训练语音编码器对从语音中直接提取可读性强的实体的影响，而无需进行文本转录。我们说明这种直接方法优化了编码器，以仅转录语音中与实体相关的部分，忽略了多余的部分，如搭档语或实体拼写。在企业虚拟代理人的对话上下文中，我们展示了一步法的方法优于典型的两步法，即首先产生词汇转录，然后进行基于文本的实体提取以识别口语实体。

    This paper reimagines some aspects of speech processing using speech encoders, specifically about extracting entities directly from speech, with no intermediate textual representation. In human-computer conversations, extracting entities such as names, postal addresses and email addresses from speech is a challenging task. In this paper, we study the impact of fine-tuning pre-trained speech encoders on extracting spoken entities in human-readable form directly from speech without the need for text transcription. We illustrate that such a direct approach optimizes the encoder to transcribe only the entity relevant portions of speech, ignoring the superfluous portions such as carrier phrases and spellings of entities. In the context of dialogs from an enterprise virtual agent, we demonstrate that the 1-step approach outperforms the typical 2-step cascade of first generating lexical transcriptions followed by text-based entity extraction for identifying spoken entities.
    

