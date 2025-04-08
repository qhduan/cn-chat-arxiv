# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Authorship Verification based on the Likelihood Ratio of Grammar Models](https://arxiv.org/abs/2403.08462) | 提出了一种基于计算作者文件在候选作者语法模型与参考群体语法模型下的可能性比率的方法，用以解决作者身份验证中存在的科学解释不足和难以解释的问题 |
| [^2] | [Large Language Models are In-Context Molecule Learners](https://arxiv.org/abs/2403.04197) | 提出了上下文分子适应（ICMA）范式，允许LLMs通过上下文示例学习分子-文本对齐，解决了在分子-标题翻译任务中对LLMs的挑战。 |
| [^3] | [Emojis Decoded: Leveraging ChatGPT for Enhanced Understanding in Social Media Communications](https://arxiv.org/abs/2402.01681) | 在表情符号研究中，我们评估了ChatGPT在处理注释和下游任务中的有效性。我们的研究结果表明ChatGPT可以作为一个可行的替代人类注释者的工具，有效地解释表情符号。 |
| [^4] | [Infini-gram: Scaling Unbounded n-gram Language Models to a Trillion Tokens](https://arxiv.org/abs/2401.17377) | 这项研究展示了n-gram语言模型的价值，并介绍了一个名为infini-gram的引擎，它可以以毫秒级的延迟计算任意n的n-gram概率，使得在神经大型语言模型中对文本进行更准确的分析成为可能。 |
| [^5] | [KERMIT: Knowledge Graph Completion of Enhanced Relation Modeling with Inverse Transformation.](http://arxiv.org/abs/2309.14770) | 本研究提出了一种增强关系建模的知识图谱补全方法，通过利用外部知识库生成连贯的描述，并通过反向关系创建对称图来提供额外的标签和补充信息。实验证明这种方法在知识图谱补全方面取得了显著的改进。 |
| [^6] | [Baichuan 2: Open Large-scale Language Models.](http://arxiv.org/abs/2309.10305) | Baichuan 2是一系列开放的大规模多语言模型，拥有70亿和130亿个参数，训练自26万亿个标记。Baichuan 2在公开基准测试中表现出色，并在垂直领域如医学和法律中具有优势。 |

# 详细

[^1]: 基于语法模型似然比的作者身份验证

    Authorship Verification based on the Likelihood Ratio of Grammar Models

    [https://arxiv.org/abs/2403.08462](https://arxiv.org/abs/2403.08462)

    提出了一种基于计算作者文件在候选作者语法模型与参考群体语法模型下的可能性比率的方法，用以解决作者身份验证中存在的科学解释不足和难以解释的问题

    

    作者身份验证（AV）是分析一组文件以确定它们是否由特定作者撰写的过程。现有的最先进AV方法使用计算解决方案，对于其功能没有合理的科学解释，并且常常难以解释给分析人员。为解决这个问题，我们提出了一种方法，依赖于计算一个我们称之为 $\lambda_G$（LambdaG）的量：候选作者的上下文语法模型给出的文档的可能性与参考群体的上下文语法模型给出的相同文档的可能性之间的比率。这些语法模型是使用仅针对语法特征进行训练的 $n$-gram语言模型进行估计的。尽管不需要大量数据进行训练，LambdaG...

    arXiv:2403.08462v1 Announce Type: new  Abstract: Authorship Verification (AV) is the process of analyzing a set of documents to determine whether they were written by a specific author. This problem often arises in forensic scenarios, e.g., in cases where the documents in question constitute evidence for a crime. Existing state-of-the-art AV methods use computational solutions that are not supported by a plausible scientific explanation for their functioning and that are often difficult for analysts to interpret. To address this, we propose a method relying on calculating a quantity we call $\lambda_G$ (LambdaG): the ratio between the likelihood of a document given a model of the Grammar for the candidate author and the likelihood of the same document given a model of the Grammar for a reference population. These Grammar Models are estimated using $n$-gram language models that are trained solely on grammatical features. Despite not needing large amounts of data for training, LambdaG st
    
[^2]: 大规模语言模型是上下文分子学习器

    Large Language Models are In-Context Molecule Learners

    [https://arxiv.org/abs/2403.04197](https://arxiv.org/abs/2403.04197)

    提出了上下文分子适应（ICMA）范式，允许LLMs通过上下文示例学习分子-文本对齐，解决了在分子-标题翻译任务中对LLMs的挑战。

    

    大型语言模型（LLMs）在生物化学任务中表现出色，尤其是分子标题翻译任务，旨在弥合分子和自然语言文本之间的差距。然而，先前在适应LLMs到分子-标题翻译任务中的方法需要额外的领域特定预训练阶段，存在分子和文本空间之间的弱对齐，或对LLMs的规模有严格要求。为了解决这些挑战，我们提出了上下文分子适应（ICMA），作为一种新的范例，允许LLMs通过上下文示例学习分子-文本对齐，通过上下文分子调整。具体而言，ICMA包括以下三个阶段：跨模态检索、检索后排序和上下文分子调整。

    arXiv:2403.04197v1 Announce Type: cross  Abstract: Large Language Models (LLMs) have demonstrated exceptional performance in biochemical tasks, especially the molecule caption translation task, which aims to bridge the gap between molecules and natural language texts. However, previous methods in adapting LLMs to the molecule-caption translation task required extra domain-specific pre-training stages, suffered weak alignment between molecular and textual spaces, or imposed stringent demands on the scale of LLMs. To resolve the challenges, we propose In-Context Molecule Adaptation (ICMA), as a new paradigm allowing LLMs to learn the molecule-text alignment from context examples via In-Context Molecule Tuning. Specifically, ICMA incorporates the following three stages: Cross-modal Retrieval, Post-retrieval Re-ranking, and In-context Molecule Tuning. Initially, Cross-modal Retrieval utilizes BM25 Caption Retrieval and Molecule Graph Retrieval to retrieve informative context examples. Addi
    
[^3]: 表情符号解密：利用ChatGPT提升社交媒体沟通的理解能力

    Emojis Decoded: Leveraging ChatGPT for Enhanced Understanding in Social Media Communications

    [https://arxiv.org/abs/2402.01681](https://arxiv.org/abs/2402.01681)

    在表情符号研究中，我们评估了ChatGPT在处理注释和下游任务中的有效性。我们的研究结果表明ChatGPT可以作为一个可行的替代人类注释者的工具，有效地解释表情符号。

    

    表情符号在社交网络沟通中已经普遍存在，它们承载了超越文字或短语的语义，这引发了学术界对其属性和功能的越来越多的研究兴趣。然而，与表情符号相关的研究和应用面临两个主要挑战。首先，研究者通常依赖众包来注释表情符号，以了解其情感、使用意图和语义含义。其次，用户的主观解释往往会导致对表情符号的误解，并造成沟通障碍。大型语言模型（LLMs）在各种注释任务中取得了显著的成功，ChatGPT在多个领域展示了专业能力。在我们的研究中，我们评估了ChatGPT在处理以前注释和下游任务中的有效性。我们的目标是验证ChatGPT可以在表情符号研究中作为人类注释者的可行替代者，并验证其解释表情符号的能力。

    Emojis, which encapsulate semantics beyond mere words or phrases, have become prevalent in social network communications. This has spurred increasing scholarly interest in exploring their attributes and functionalities. However, emoji-related research and application face two primary challenges. First, researchers typically rely on crowd-sourcing to annotate emojis in order to understand their sentiments, usage intentions, and semantic meanings. Second, subjective interpretations by users can often lead to misunderstandings of emojis and cause the communication barrier. Large Language Models (LLMs) have achieved significant success in various annotation tasks, with ChatGPT demonstrating expertise across multiple domains. In our study, we assess ChatGPT's effectiveness in handling previously annotated and downstream tasks. Our objective is to validate the hypothesis that ChatGPT can serve as a viable alternative to human annotators in emoji research and that its ability to explain emoji
    
[^4]: 无限-gram：将无限n-gram语言模型扩展到万亿标记

    Infini-gram: Scaling Unbounded n-gram Language Models to a Trillion Tokens

    [https://arxiv.org/abs/2401.17377](https://arxiv.org/abs/2401.17377)

    这项研究展示了n-gram语言模型的价值，并介绍了一个名为infini-gram的引擎，它可以以毫秒级的延迟计算任意n的n-gram概率，使得在神经大型语言模型中对文本进行更准确的分析成为可能。

    

    在神经大型语言模型（LLM）时代，n-gram语言模型还具有相关性吗？我们的答案是肯定的，并且我们展示了它们在文本分析和改进神经LLM方面的价值。然而，这需要在两个方面对n-gram模型进行现代化。首先，我们将它们与神经LLM相同的数据规模训练- 1.4万亿个标记。这是迄今为止构建的最大的n-gram模型。其次，现有的n-gram模型使用的n很小，这妨碍了它们的性能；相反，我们允许n可以是任意大的，通过引入一个新的无限-gram LM与回退。我们开发了一个名为infini-gram的引擎，它可以通过后缀数组计算无限-gram（以及任意n的n-gram）概率，并且具有毫秒级的延迟，而无需预先计算n-gram计数表（这将非常昂贵）。无限-gram框架和infini-gram引擎使我们能够对人类写作和机器生成的文本进行许多新颖和有意思的分析：我们发现无限-gram LM...

    Are n-gram language models still relevant in this era of neural large language models (LLMs)? Our answer is yes, and we show their values in both text analysis and improving neural LLMs. Yet this necessitates modernizing n-gram models in two aspects. First, we train them at the same data scale as neural LLMs -- 1.4 trillion tokens. This is the largest n-gram model ever built. Second, existing n-gram models use small n which hinders their performance; we instead allow n to be arbitrarily large, by introducing a new $\infty$-gram LM with backoff. Instead of pre-computing n-gram count tables (which would be very expensive), we develop an engine named infini-gram -- powered by suffix arrays -- that can compute $\infty$-gram (as well as n-gram with arbitrary n) probabilities with millisecond-level latency. The $\infty$-gram framework and infini-gram engine enable us to conduct many novel and interesting analyses of human-written and machine-generated text: we find that the $\infty$-gram LM 
    
[^5]: KERMIT: 带有反转变换的增强关系建模的知识图谱补全

    KERMIT: Knowledge Graph Completion of Enhanced Relation Modeling with Inverse Transformation. (arXiv:2309.14770v1 [cs.CL])

    [http://arxiv.org/abs/2309.14770](http://arxiv.org/abs/2309.14770)

    本研究提出了一种增强关系建模的知识图谱补全方法，通过利用外部知识库生成连贯的描述，并通过反向关系创建对称图来提供额外的标签和补充信息。实验证明这种方法在知识图谱补全方面取得了显著的改进。

    

    知识图谱补全是一项基于知识图谱中可用信息填充缺失三元组的任务。在当前的研究中，基于文本的方法通过利用三元组的文本描述来完成任务。然而，这种建模方法可能遇到一些限制，尤其是当描述不能准确充分地表达预期含义时。为了克服这些挑战，我们提出了通过两个额外机制来增加数据的方法。首先，我们使用ChatGPT作为外部知识库，生成连贯的描述以弥补查询和答案之间的语义差距。其次，我们利用反向关系创建对称图，从而为链接预测提供额外的标签和补充信息。这种方法提供了关系实体之间额外的洞察力。通过这些努力，我们观察到了知识图谱补全方面的显著改进。

    Knowledge graph completion is a task that revolves around filling in missing triples based on the information available in a knowledge graph. Among the current studies, text-based methods complete the task by utilizing textual descriptions of triples. However, this modeling approach may encounter limitations, particularly when the description fails to accurately and adequately express the intended meaning. To overcome these challenges, we propose the augmentation of data through two additional mechanisms. Firstly, we employ ChatGPT as an external knowledge base to generate coherent descriptions to bridge the semantic gap between the queries and answers. Secondly, we leverage inverse relations to create a symmetric graph, thereby creating extra labeling and providing supplementary information for link prediction. This approach offers additional insights into the relationships between entities. Through these efforts, we have observed significant improvements in knowledge graph completion
    
[^6]: Baichuan 2: 开放的大规模语言模型

    Baichuan 2: Open Large-scale Language Models. (arXiv:2309.10305v1 [cs.CL])

    [http://arxiv.org/abs/2309.10305](http://arxiv.org/abs/2309.10305)

    Baichuan 2是一系列开放的大规模多语言模型，拥有70亿和130亿个参数，训练自26万亿个标记。Baichuan 2在公开基准测试中表现出色，并在垂直领域如医学和法律中具有优势。

    

    大型语言模型（LLMs）在仅有少量自然语言指令示例的情况下，已经在各种自然语言任务中展示出了令人瞩目的性能，减少了对广泛特征工程的需求。然而，大多数强大的LLMs是封闭源代码的，或者在除了英语以外的其他语言方面能力有限。在这篇技术报告中，我们介绍了Baichuan 2系列，这是一系列从头开始进行训练的大规模多语言模型，包含70亿和130亿个参数，使用26万亿个标记进行训练。Baichuan 2在MMLU、CMMLU、GSM8K和HumanEval等公开基准测试中与其他相同规模的开源模型相匹配或胜过。此外，Baichuan 2在医学和法律等垂直领域表现出色。我们将发布所有预训练模型检查点，以使研究界更好地理解Baichuan 2的训练动态。

    Large language models (LLMs) have demonstrated remarkable performance on a variety of natural language tasks based on just a few examples of natural language instructions, reducing the need for extensive feature engineering. However, most powerful LLMs are closed-source or limited in their capability for languages other than English. In this technical report, we present Baichuan 2, a series of large-scale multilingual language models containing 7 billion and 13 billion parameters, trained from scratch, on 2.6 trillion tokens. Baichuan 2 matches or outperforms other open-source models of similar size on public benchmarks like MMLU, CMMLU, GSM8K, and HumanEval. Furthermore, Baichuan 2 excels in vertical domains such as medicine and law. We will release all pre-training model checkpoints to benefit the research community in better understanding the training dynamics of Baichuan 2.
    

