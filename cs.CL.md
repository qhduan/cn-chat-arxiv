# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Don't Half-listen: Capturing Key-part Information in Continual Instruction Tuning](https://arxiv.org/abs/2403.10056) | 提出了一种基于关键部分信息增益的新型连续指导调整方法，通过动态重放数据和优化训练目标，使LLMs能够捕捉任务感知信息和减轻过度拟合。 |
| [^2] | [CodeMind: A Framework to Challenge Large Language Models for Code Reasoning](https://arxiv.org/abs/2402.09664) | CodeMind是一个用于挑战大型语言模型进行代码推理的框架，通过评估LLMs的代码推理能力来替代仅仅依靠测试通过来评估，对三种代码推理任务进行评估，结果显示LLMs能够公正地理解控制流结构，并且对于简单程序和复杂程序，它们通常能够推理出输入如何演变为输出。 |
| [^3] | [EntGPT: Linking Generative Large Language Models with Knowledge Bases](https://arxiv.org/abs/2402.06738) | 本文介绍了一种名为EntGPT的模型，通过Entity Disambiguation（ED）任务，连接了生成型大型语言模型与知识库。通过提示工程和指令调整，该模型在没有有监督微调的情况下，显著提高了LLMs的性能，并在实体消歧任务上取得了可比较的性能。 |
| [^4] | [Language Models are Universal Embedders.](http://arxiv.org/abs/2310.08232) | 该论文证明了多语言预训练的Transformer解码器在有限英文数据微调后能够通用地进行嵌入，实现了统一嵌入模型的目标。 |
| [^5] | [LABO: Towards Learning Optimal Label Regularization via Bi-level Optimization.](http://arxiv.org/abs/2305.04971) | 本文提出了一种基于标签正则化的通用框架，其中包括传统的LS，但也可以建模实例特定的变体。我们提出了一种双层优化的方法（LABO），用于学习标签正则化，并得到了可解释的最优标签平滑解。 |

# 详细

[^1]: 不要半心半意：捕捉连续指导调整中的关键部分信息

    Don't Half-listen: Capturing Key-part Information in Continual Instruction Tuning

    [https://arxiv.org/abs/2403.10056](https://arxiv.org/abs/2403.10056)

    提出了一种基于关键部分信息增益的新型连续指导调整方法，通过动态重放数据和优化训练目标，使LLMs能够捕捉任务感知信息和减轻过度拟合。

    

    arXiv:2403.10056v1 公告类型: 跨领域 摘要：大型语言模型（LLMs）的指导调整可以驱使它们在特定下游任务中产生符合人类目标的结果。然而，LLMs的连续指导调整（CIT）过程可能会带来灾难性遗忘（CF）问题，导致先前学到的能力退化。最近的方法尝试通过修改模型或重放数据来缓解CF问题，但这可能只记住指令的表面模式并在留存任务上感到困惑。在本文中，我们提出了一种基于关键部分信息增益（KPIG）的新型连续指导调整方法。我们的方法计算掩盖部分的信息增益，动态重放数据并优化训练目标，从而使LLMs能够捕捉与正确响应相关的任务感知信息，并减轻对指导中通用描述的过度拟合。此外，我们提出了两个指标，P分和V分，

    arXiv:2403.10056v1 Announce Type: cross  Abstract: Instruction tuning for large language models (LLMs) can drive them to produce results consistent with human goals in specific downstream tasks. However, the process of continual instruction tuning (CIT) for LLMs may bring about the catastrophic forgetting (CF) problem, where previously learned abilities are degraded. Recent methods try to alleviate the CF problem by modifying models or replaying data, which may only remember the surface-level pattern of instructions and get confused on held-out tasks. In this paper, we propose a novel continual instruction tuning method based on Key-part Information Gain (KPIG). Our method computes the information gain on masked parts to dynamically replay data and refine the training objective, which enables LLMs to capture task-aware information relevant to the correct response and alleviate overfitting to general descriptions in instructions. In addition, we propose two metrics, P-score and V-score,
    
[^2]: CodeMind:一个用于挑战大型语言模型进行代码推理的框架

    CodeMind: A Framework to Challenge Large Language Models for Code Reasoning

    [https://arxiv.org/abs/2402.09664](https://arxiv.org/abs/2402.09664)

    CodeMind是一个用于挑战大型语言模型进行代码推理的框架，通过评估LLMs的代码推理能力来替代仅仅依靠测试通过来评估，对三种代码推理任务进行评估，结果显示LLMs能够公正地理解控制流结构，并且对于简单程序和复杂程序，它们通常能够推理出输入如何演变为输出。

    

    仅靠测试通过来评估大型语言模型（LLMs）的代码合成能力可能会导致不公正的评估或促进具有数据泄漏的模型，作为一种替代方案，我们介绍了CodeMind，这是一个旨在评估LLMs的代码推理能力的框架。CodeMind目前支持三种代码推理任务：独立执行推理（IER）、依赖执行推理（DER）和规范推理（SR）。前两者评估模型以预测任意代码的执行输出，或者模型能够正确合成的代码。第三个任务评估LLMs实现指定预期行为的程度。我们使用CodeMind对两种不同编程语言中的五个基准下的九个LLMs进行了广泛的评估，结果表明LLMs能够公正地理解控制流结构，并且对于简单程序和复杂程序，它们通常能够推理出输入如何演变为输出。

    arXiv:2402.09664v1 Announce Type: cross  Abstract: Solely relying on test passing to evaluate Large Language Models (LLMs) for code synthesis may result in unfair assessment or promoting models with data leakage. As an alternative, we introduce CodeMind, a framework designed to gauge the code reasoning abilities of LLMs. CodeMind currently supports three code reasoning tasks: Independent Execution Reasoning (IER), Dependent Execution Reasoning (DER), and Specification Reasoning (SR). The first two evaluate models to predict the execution output of an arbitrary code or code the model could correctly synthesize. The third one evaluates the extent to which LLMs implement the specified expected behavior. Our extensive evaluation of nine LLMs across five benchmarks in two different programming languages using CodeMind shows that LLMs fairly understand control flow constructs and, in general, are capable of reasoning how inputs evolve to output, specifically for simple programs and the ones 
    
[^3]: EntGPT: 将生成型大型语言模型与知识库相连接

    EntGPT: Linking Generative Large Language Models with Knowledge Bases

    [https://arxiv.org/abs/2402.06738](https://arxiv.org/abs/2402.06738)

    本文介绍了一种名为EntGPT的模型，通过Entity Disambiguation（ED）任务，连接了生成型大型语言模型与知识库。通过提示工程和指令调整，该模型在没有有监督微调的情况下，显著提高了LLMs的性能，并在实体消歧任务上取得了可比较的性能。

    

    由于训练和推理过程中缺乏事实核实和知识基础，大型语言模型（LLM）生成的事实正确输出的能力相对较少被研究。在这项工作中，我们通过Entity Disambiguation（ED）任务来解决这一挑战。我们首先考虑了提示工程，并设计了一个三步硬提示方法，以在没有有监督微调（SFT）的情况下探测LLM的ED性能。总体而言，该提示方法显著提高了原始基准模型的微F_1得分，在某些情况下提高了36%甚至更高，并在10个数据集上与现有的SFT方法相比，获得了可比较的性能。我们通过使用类似的提示和响应进行指令调整（IT）进一步提高了知识基础。指令调整的模型在受监督实体消歧任务上不仅实现了更高的微F1得分性能，而且平均微F_1提高了。

    The ability of Large Language Models (LLMs) to generate factually correct output remains relatively unexplored due to the lack of fact-checking and knowledge grounding during training and inference. In this work, we aim to address this challenge through the Entity Disambiguation (ED) task. We first consider prompt engineering, and design a three-step hard-prompting method to probe LLMs' ED performance without supervised fine-tuning (SFT). Overall, the prompting method improves the micro-F_1 score of the original vanilla models by a large margin, on some cases up to 36% and higher, and obtains comparable performance across 10 datasets when compared to existing methods with SFT. We further improve the knowledge grounding ability through instruction tuning (IT) with similar prompts and responses. The instruction-tuned model not only achieves higher micro-F1 score performance as compared to several baseline methods on supervised entity disambiguation tasks with an average micro-F_1 improve
    
[^4]: 语言模型是通用的嵌入器

    Language Models are Universal Embedders. (arXiv:2310.08232v1 [cs.CL])

    [http://arxiv.org/abs/2310.08232](http://arxiv.org/abs/2310.08232)

    该论文证明了多语言预训练的Transformer解码器在有限英文数据微调后能够通用地进行嵌入，实现了统一嵌入模型的目标。

    

    在大型语言模型（LLM）革命中，嵌入是各种系统的关键组成部分。例如，它被用于为LLMs检索知识或记忆，构建内容过滤器等。由于这些情况涉及从英语到其他自然或编程语言，从检索到分类等各种情况，因此建立一个统一的嵌入模型而不是为每个场景专门建立一个是可取的。在这项工作中，我们迈出了朝这个目标迈出了初始的一步，证明了多语言（自然语言和编程语言）预训练的Transformer解码器在有限的英文数据微调后能够通用地进行嵌入。我们提供了全面的实践，并进行了彻底的评估。在英文MTEB上，我们的模型在不使用大量训练数据的情况下在不同的嵌入任务上达到了竞争性的性能。在其他基准测试中，例如多语言分类和代码搜索，我们的模型（没有任何监督）表现出与或甚至超过大量监督基线的可比性。

    In the large language model (LLM) revolution, embedding is a key component of various systems. For example, it is used to retrieve knowledge or memories for LLMs, to build content moderation filters, etc. As such cases span from English to other natural or programming languages, from retrieval to classification and beyond, it is desirable to build a unified embedding model rather than dedicated ones for each scenario. In this work, we make an initial step towards this goal, demonstrating that multiple languages (both natural and programming) pre-trained transformer decoders can embed universally when finetuned on limited English data. We provide a comprehensive practice with thorough evaluations. On English MTEB, our models achieve competitive performance on different embedding tasks by minimal training data. On other benchmarks, such as multilingual classification and code search, our models (without any supervision) perform comparably to, or even surpass heavily supervised baselines 
    
[^5]: LABO: 通过双层优化实现最佳标签正则化学习

    LABO: Towards Learning Optimal Label Regularization via Bi-level Optimization. (arXiv:2305.04971v1 [cs.LG])

    [http://arxiv.org/abs/2305.04971](http://arxiv.org/abs/2305.04971)

    本文提出了一种基于标签正则化的通用框架，其中包括传统的LS，但也可以建模实例特定的变体。我们提出了一种双层优化的方法（LABO），用于学习标签正则化，并得到了可解释的最优标签平滑解。

    

    正则化技术对于改善深度神经网络的泛化性能和训练效率至关重要。许多深度学习算法依赖于权重衰减、丢弃、批/层归一化等技术来更快地收敛和泛化。标签平滑（LS）是另一种简单、通用且高效的正则化方法，可用于各种监督分类任务。然而，传统的LS假设每个非目标类别出现的概率相等，不能根据实例对标签进行优化。本文提出了一种基于标签正则化的通用框架，包括传统的LS但也可以建模实例特定的变体。基于该框架，我们提出了一种通过设计双层优化（LABO）问题来学习标签正则化的高效方法。我们得出了内环节的确定性和可解释解，而无需存储经过训练模型的参数或输出。

    Regularization techniques are crucial to improving the generalization performance and training efficiency of deep neural networks. Many deep learning algorithms rely on weight decay, dropout, batch/layer normalization to converge faster and generalize. Label Smoothing (LS) is another simple, versatile and efficient regularization which can be applied to various supervised classification tasks. Conventional LS, however, regardless of the training instance assumes that each non-target class is equally likely. In this work, we present a general framework for training with label regularization, which includes conventional LS but can also model instance-specific variants. Based on this formulation, we propose an efficient way of learning LAbel regularization by devising a Bi-level Optimization (LABO) problem. We derive a deterministic and interpretable solution of the inner loop as the optimal label smoothing without the need to store the parameters or the output of a trained model. Finally
    

