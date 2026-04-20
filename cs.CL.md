# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Automatic Combination of Sample Selection Strategies for Few-Shot Learning](https://arxiv.org/abs/2402.03038) | 本文研究了20种样本选择策略对少样本学习性能的影响，并提出了一种自动组合样本选择策略的方法（ACSESS），在多个数据集上证明了其优越性能。 |
| [^2] | [Deep Learning Based Amharic Chatbot for FAQs in Universities](https://arxiv.org/abs/2402.01720) | 本文提出了一个基于深度学习的阿姆哈拉语常见问题解答聊天机器人模型，可以帮助大学生解答常见问题，通过使用自然语言处理和深度学习技术，采用多种机器学习模型算法进行分析和分类，取得了最好的成绩。 |

# 详细

[^1]: 自动组合样本选择策略用于少样本学习

    Automatic Combination of Sample Selection Strategies for Few-Shot Learning

    [https://arxiv.org/abs/2402.03038](https://arxiv.org/abs/2402.03038)

    本文研究了20种样本选择策略对少样本学习性能的影响，并提出了一种自动组合样本选择策略的方法（ACSESS），在多个数据集上证明了其优越性能。

    

    在少样本学习中，如元学习、少样本微调或上下文学习中，用于训练模型的有限样本数量对整体成功具有显著影响。尽管存在大量的样本选择策略，但它们对少样本学习性能的影响尚不十分明确，因为大部分只被在典型的监督设置中进行了评估。本文通过对8个图像和6个文本数据集上的5种少样本学习方法，彻底研究了20种样本选择策略对性能的影响。此外，我们提出了一种新的自动组合样本选择策略的方法（ACSESS），它充分利用了个体策略的优势和互补信息。实验结果表明，我们的方法始终优于个体选择策略，以及最近提出的上下文学习支持样本选择方法。

    In few-shot learning, such as meta-learning, few-shot fine-tuning or in-context learning, the limited number of samples used to train a model have a significant impact on the overall success. Although a large number of sample selection strategies exist, their impact on the performance of few-shot learning is not extensively known, as most of them have been so far evaluated in typical supervised settings only. In this paper, we thoroughly investigate the impact of 20 sample selection strategies on the performance of 5 few-shot learning approaches over 8 image and 6 text datasets. In addition, we propose a new method for automatic combination of sample selection strategies (ACSESS) that leverages the strengths and complementary information of the individual strategies. The experimental results show that our method consistently outperforms the individual selection strategies, as well as the recently proposed method for selecting support examples for in-context learning. We also show a str
    
[^2]: 基于深度学习的阿姆哈拉语常见问题解答聊天机器人

    Deep Learning Based Amharic Chatbot for FAQs in Universities

    [https://arxiv.org/abs/2402.01720](https://arxiv.org/abs/2402.01720)

    本文提出了一个基于深度学习的阿姆哈拉语常见问题解答聊天机器人模型，可以帮助大学生解答常见问题，通过使用自然语言处理和深度学习技术，采用多种机器学习模型算法进行分析和分类，取得了最好的成绩。

    

    大学生常常花费大量时间向管理员或教师寻求常见问题的答案。这对双方来说都很繁琐，需要找到一个解决方案。为此，本文提出了一个聊天机器人模型，利用自然语言处理和深度学习技术，在阿姆哈拉语中回答常见问题。聊天机器人是通过人工智能模拟人类对话的计算机程序，作为虚拟助手处理问题和其他任务。所提出的聊天机器人程序使用标记化、规范化、去除停用词和词干提取对阿姆哈拉语输入句子进行分析和分类。采用了三种机器学习模型算法来分类标记和检索合适的回答：支持向量机（SVM）、多项式朴素贝叶斯和通过TensorFlow、Keras和NLTK实现的深度神经网络。深度学习模型取得了最好的成绩。

    University students often spend a considerable amount of time seeking answers to common questions from administrators or teachers. This can become tedious for both parties, leading to a need for a solution. In response, this paper proposes a chatbot model that utilizes natural language processing and deep learning techniques to answer frequently asked questions (FAQs) in the Amharic language. Chatbots are computer programs that simulate human conversation through the use of artificial intelligence (AI), acting as a virtual assistant to handle questions and other tasks. The proposed chatbot program employs tokenization, normalization, stop word removal, and stemming to analyze and categorize Amharic input sentences. Three machine learning model algorithms were used to classify tokens and retrieve appropriate responses: Support Vector Machine (SVM), Multinomial Na\"ive Bayes, and deep neural networks implemented through TensorFlow, Keras, and NLTK. The deep learning model achieved the be
    

