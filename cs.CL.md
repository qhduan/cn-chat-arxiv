# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Mixed Preference Optimization: Reinforcement Learning with Data Selection and Better Reference Model](https://arxiv.org/abs/2403.19443) | 提出了一种混合偏好优化（MPO）方法，通过在简单数据集上训练Direct Preference Optimization（DPO），然后在困难数据集上执行Reinforcement Learning with Human Feedback（RLHF），从而减轻了两种方法的弱点。 |
| [^2] | [Can We Verify Step by Step for Incorrect Answer Detection?](https://arxiv.org/abs/2402.10528) | 通过推理链来预测大型语言模型输出的准确性，我们引入了一个新的基准R2PE，并提出了处理可辨识性评分（PDS）框架。 |
| [^3] | [Enhancing Textbook Question Answering Task with Large Language Models and Retrieval Augmented Generation](https://arxiv.org/abs/2402.05128) | 本论文通过引入检索增强生成（RAG）技术和利用迁移学习来处理长文本和提升推理能力，为教科书问答任务带来了显著的改进。 |

# 详细

[^1]: 混合偏好优化：强化学习中的数据选择与更好的参考模型

    Mixed Preference Optimization: Reinforcement Learning with Data Selection and Better Reference Model

    [https://arxiv.org/abs/2403.19443](https://arxiv.org/abs/2403.19443)

    提出了一种混合偏好优化（MPO）方法，通过在简单数据集上训练Direct Preference Optimization（DPO），然后在困难数据集上执行Reinforcement Learning with Human Feedback（RLHF），从而减轻了两种方法的弱点。

    

    大型语言模型（LLMs）因其处理和生成自然语言的能力而日益受到青睐。然而，由于它们是在大规模文本数据集上训练的，LLMs可能会继承有害偏见，并产生与人类价值观不一致的输出。本文研究了LLM对齐的两种主要方法：带人类反馈的强化学习（RLHF）和基于对比学习的方法如直接偏好优化（DPO）。通过分析RLHF和DPO的稳定性和鲁棒性，我们提出了MPO（混合偏好优化），这是一种缓解两种方法弱点的新方法。具体而言，我们提出了一个两阶段训练过程：首先在一个简单数据集上训练DPO，然后再在带有DPO模型作为参考模型的困难集上执行RLHF。在这里，简单和困难集是由训练良好的奖励模型构建的，将响应对分成具有较大差距的对。

    arXiv:2403.19443v1 Announce Type: new  Abstract: Large Language Models (LLMs) have become increasingly popular due to their ability to process and generate natural language. However, as they are trained on massive datasets of text, LLMs can inherit harmful biases and produce outputs that are not aligned with human values. This paper studies two main approaches to LLM alignment: Reinforcement Learning with Human Feedback (RLHF) and contrastive learning-based methods like Direct Preference Optimization (DPO). By analyzing the stability and robustness of RLHF and DPO, we propose MPO (Mixed Preference Optimization), a novel method that mitigates the weaknesses of both approaches. Specifically, we propose a two-stage training procedure: first train DPO on an easy dataset, and then perform RLHF on a difficult set with DPO model being the reference model. Here, the easy and difficult sets are constructed by a well-trained reward model that splits response pairs into those with large gaps of r
    
[^2]: 我们能否逐步验证错误答案检测？

    Can We Verify Step by Step for Incorrect Answer Detection?

    [https://arxiv.org/abs/2402.10528](https://arxiv.org/abs/2402.10528)

    通过推理链来预测大型语言模型输出的准确性，我们引入了一个新的基准R2PE，并提出了处理可辨识性评分（PDS）框架。

    

    Chain-of-Thought（CoT）提示在增强大型语言模型（LLMs）的推理能力方面取得了重大进展。先前的研究开发了各种扩展的CoT，主要集中在增强最终任务的性能上。此外，已经有研究评估了CoT中推理链的质量。这引发了一个有趣的问题：通过仔细审查它们生成的推理链，是否可以预测LLMs输出的准确性？为了回答这个研究问题，我们引入了一个基准，R2PE，专门设计用于探究不同领域涵盖五个不同推理任务中推理链与性能之间的关系。该基准旨在基于推理步骤衡量LLMs最终输出的虚假性。为了充分利用多个推理链中的信息，我们提出了打败常识分数（PDS）框架。

    arXiv:2402.10528v1 Announce Type: cross  Abstract: Chain-of-Thought (CoT) prompting has marked a significant advancement in enhancing the reasoning capabilities of large language models (LLMs). Previous studies have developed various extensions of CoT, which focus primarily on enhancing end-task performance. In addition, there has been research on assessing the quality of reasoning chains in CoT. This raises an intriguing question: Is it possible to predict the accuracy of LLM outputs by scrutinizing the reasoning chains they generate? To answer this research question, we introduce a benchmark, R2PE, designed specifically to explore the relationship between reasoning chains and performance in various reasoning tasks spanning five different domains. This benchmark aims to measure the falsehood of the final output of LLMs based on the reasoning steps. To make full use of information in multiple reasoning chains, we propose the process discernibility score (PDS) framework that beats the a
    
[^3]: 用大型语言模型和检索增强生成提升教科书问答任务

    Enhancing Textbook Question Answering Task with Large Language Models and Retrieval Augmented Generation

    [https://arxiv.org/abs/2402.05128](https://arxiv.org/abs/2402.05128)

    本论文通过引入检索增强生成（RAG）技术和利用迁移学习来处理长文本和提升推理能力，为教科书问答任务带来了显著的改进。

    

    教科书问答（TQA）是人工智能中的一项具有挑战性的任务，由于上下文和多模式数据的复杂性。尽管以前的研究在任务上取得了显著的进展，但仍存在一些限制，包括模型推理能力不足和无法捕捉长文本中的上下文信息。大型语言模型（LLMs）的引入革命了人工智能领域，然而，直接应用LLMs经常会导致不准确的答案。本文提出了一种方法来处理TQA中领域外情景，即概念分布在不同课程中，该方法结合了检索增强生成（RAG）技术和迁移学习来处理长文本并提升推理能力。通过对LLM模型Llama-2进行监督微调并加入RAG，我们的架构优于基线，在验证集上的准确度提高了4.12%，在测试集上提高了9.84%。

    Textbook question answering (TQA) is a challenging task in artificial intelligence due to the complex nature of context and multimodal data. Although previous research has significantly improved the task, there are still some limitations including the models' weak reasoning and inability to capture contextual information in the lengthy context. The introduction of large language models (LLMs) has revolutionized the field of AI, however, directly applying LLMs often leads to inaccurate answers. This paper proposes a methodology that handle the out-of-domain scenario in TQA where concepts are spread across different lessons by incorporating the retrieval augmented generation (RAG) technique and utilize transfer learning to handle the long context and enhance reasoning abilities. Through supervised fine-tuning of the LLM model Llama-2 and the incorporation of RAG, our architecture outperforms the baseline, achieving a 4.12% accuracy improvement on validation set and 9.84% on test set for 
    

