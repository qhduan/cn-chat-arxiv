# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [The Fine Line: Navigating Large Language Model Pretraining with Down-streaming Capability Analysis](https://arxiv.org/abs/2404.01204) | 本文对大型语言模型在预训练过程中不同能力的综合分析揭示了其动态差异，填补了现有标度定律中的缺失。 |
| [^2] | [Ensemble-Based Unsupervised Discontinuous Constituency Parsing by Tree Averaging](https://arxiv.org/abs/2403.00143) | 通过树平均法构建集成解析器，稳定并提升无监督不连续成分句法分析性能，实验结果表明该方法在所有指标上均优于基准线 |
| [^3] | [DropBP: Accelerating Fine-Tuning of Large Language Models by Dropping Backward Propagation](https://arxiv.org/abs/2402.17812) | DropBP提出了一种新颖的方式来加速大型语言模型的微调，通过在反向传播过程中随机丢弃层以减少计算成本同时保持准确性。 |
| [^4] | [Deep Learning Based Amharic Chatbot for FAQs in Universities](https://arxiv.org/abs/2402.01720) | 本文提出了一个基于深度学习的阿姆哈拉语常见问题解答聊天机器人模型，可以帮助大学生解答常见问题，通过使用自然语言处理和深度学习技术，采用多种机器学习模型算法进行分析和分类，取得了最好的成绩。 |

# 详细

[^1]: 大型语言模型预训练与下游能力分析的微妙平衡

    The Fine Line: Navigating Large Language Model Pretraining with Down-streaming Capability Analysis

    [https://arxiv.org/abs/2404.01204](https://arxiv.org/abs/2404.01204)

    本文对大型语言模型在预训练过程中不同能力的综合分析揭示了其动态差异，填补了现有标度定律中的缺失。

    

    揭示反映最终模型性能的早期指标是大规模预训练的一个核心原则。现有的标度定律展示了预训练损失与训练浮点数之间的幂律相关性，这对于大型语言模型当前训练状态的重要指标十分关键。然而，这一原则只关注模型在训练数据上的压缩特性，导致与下游任务能力的提升之间存在不一致性。一些后续研究试图将标度定律扩展到更复杂的指标（如超参数），但仍然缺乏对预训练过程中各种能力之间动态差异的全面分析。为解决上述限制，本文对各种预训练中间检查点下模型能力进行了全面比较。

    arXiv:2404.01204v1 Announce Type: new  Abstract: Uncovering early-stage metrics that reflect final model performance is one core principle for large-scale pretraining. The existing scaling law demonstrates the power-law correlation between pretraining loss and training flops, which serves as an important indicator of the current training state for large language models. However, this principle only focuses on the model's compression properties on the training data, resulting in an inconsistency with the ability improvements on the downstream tasks. Some follow-up works attempted to extend the scaling-law to more complex metrics (such as hyperparameters), but still lacked a comprehensive analysis of the dynamic differences among various capabilities during pretraining. To address the aforementioned limitations, this paper undertakes a comprehensive comparison of model capabilities at various pretraining intermediate checkpoints. Through this analysis, we confirm that specific downstream
    
[^2]: 基于集成的无监督不连续成分句法分析：树平均法

    Ensemble-Based Unsupervised Discontinuous Constituency Parsing by Tree Averaging

    [https://arxiv.org/abs/2403.00143](https://arxiv.org/abs/2403.00143)

    通过树平均法构建集成解析器，稳定并提升无监督不连续成分句法分析性能，实验结果表明该方法在所有指标上均优于基准线

    

    我们解决了无监督不连续成分句法分析的问题，在这个问题中我们观察到先前唯一模型的性能存在高方差。我们提出通过对现有不连续解析器的不同运行构建一个集成，并通过平均预测树来稳定和提升性能。首先，我们针对不同的二元性和连续性设置提供了全面的树平均计算复杂度分析（以P和NP完全为单位）。然后，我们开发了一种高效的精确算法来处理这一任务，在我们的实验中对所有样本运行时间均合理。在三个数据集上的结果显示我们的方法在所有指标上均优于所有基准线，我们还对我们的方法进行了深入分析。

    arXiv:2403.00143v1 Announce Type: cross  Abstract: We address unsupervised discontinuous constituency parsing, where we observe a high variance in the performance of the only previous model. We propose to build an ensemble of different runs of the existing discontinuous parser by averaging the predicted trees, to stabilize and boost performance. To begin with, we provide comprehensive computational complexity analysis (in terms of P and NP-complete) for tree averaging under different setups of binarity and continuity. We then develop an efficient exact algorithm to tackle the task, which runs in a reasonable time for all samples in our experiments. Results on three datasets show our method outperforms all baselines in all metrics; we also provide in-depth analyses of our approach.
    
[^3]: DropBP：通过丢弃反向传播加速大型语言模型的微调

    DropBP: Accelerating Fine-Tuning of Large Language Models by Dropping Backward Propagation

    [https://arxiv.org/abs/2402.17812](https://arxiv.org/abs/2402.17812)

    DropBP提出了一种新颖的方式来加速大型语言模型的微调，通过在反向传播过程中随机丢弃层以减少计算成本同时保持准确性。

    

    训练深度神经网络通常涉及正向和反向传播过程中的大量计算成本。传统的层次丢弃技术在训练过程中丢弃某些层以减少计算负担。然而，在正向传播过程中丢弃层会对训练过程产生不利影响，降低准确性。本文提出了DropBP，这是一种旨在减少计算成本同时保持准确性的新方法。DropBP在反向传播过程中随机丢弃层，不影响正向传播。此外，DropBP计算每个层的敏感性以分配适当的丢失率，从而稳定训练过程。DropBP旨在通过反向传播增强训练过程的效率，从而加速使用反向传播进行完全微调和参数高效微调。

    arXiv:2402.17812v1 Announce Type: cross  Abstract: Training deep neural networks typically involves substantial computational costs during both forward and backward propagation. The conventional layer dropping techniques drop certain layers during training for reducing the computations burden. However, dropping layers during forward propagation adversely affects the training process by degrading accuracy. In this paper, we propose Dropping Backward Propagation (DropBP), a novel approach designed to reduce computational costs while maintaining accuracy. DropBP randomly drops layers during the backward propagation, which does not deviate forward propagation. Moreover, DropBP calculates the sensitivity of each layer to assign appropriate drop rate, thereby stabilizing the training process. DropBP is designed to enhance the efficiency of the training process with backpropagation, thereby enabling the acceleration of both full fine-tuning and parameter-efficient fine-tuning using backpropag
    
[^4]: 基于深度学习的阿姆哈拉语常见问题解答聊天机器人

    Deep Learning Based Amharic Chatbot for FAQs in Universities

    [https://arxiv.org/abs/2402.01720](https://arxiv.org/abs/2402.01720)

    本文提出了一个基于深度学习的阿姆哈拉语常见问题解答聊天机器人模型，可以帮助大学生解答常见问题，通过使用自然语言处理和深度学习技术，采用多种机器学习模型算法进行分析和分类，取得了最好的成绩。

    

    大学生常常花费大量时间向管理员或教师寻求常见问题的答案。这对双方来说都很繁琐，需要找到一个解决方案。为此，本文提出了一个聊天机器人模型，利用自然语言处理和深度学习技术，在阿姆哈拉语中回答常见问题。聊天机器人是通过人工智能模拟人类对话的计算机程序，作为虚拟助手处理问题和其他任务。所提出的聊天机器人程序使用标记化、规范化、去除停用词和词干提取对阿姆哈拉语输入句子进行分析和分类。采用了三种机器学习模型算法来分类标记和检索合适的回答：支持向量机（SVM）、多项式朴素贝叶斯和通过TensorFlow、Keras和NLTK实现的深度神经网络。深度学习模型取得了最好的成绩。

    University students often spend a considerable amount of time seeking answers to common questions from administrators or teachers. This can become tedious for both parties, leading to a need for a solution. In response, this paper proposes a chatbot model that utilizes natural language processing and deep learning techniques to answer frequently asked questions (FAQs) in the Amharic language. Chatbots are computer programs that simulate human conversation through the use of artificial intelligence (AI), acting as a virtual assistant to handle questions and other tasks. The proposed chatbot program employs tokenization, normalization, stop word removal, and stemming to analyze and categorize Amharic input sentences. Three machine learning model algorithms were used to classify tokens and retrieve appropriate responses: Support Vector Machine (SVM), Multinomial Na\"ive Bayes, and deep neural networks implemented through TensorFlow, Keras, and NLTK. The deep learning model achieved the be
    

