# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [JMLR: Joint Medical LLM and Retrieval Training for Enhancing Reasoning and Professional Question Answering Capability](https://arxiv.org/abs/2402.17887) | JMLR通过联合训练信息检索系统和大型语言模型，在医学领域提高问题回答系统性能，降低计算资源需求，增强模型利用医疗知识进行推理和回答问题的能力。 |
| [^2] | [GRILLBot In Practice: Lessons and Tradeoffs Deploying Large Language Models for Adaptable Conversational Task Assistants](https://arxiv.org/abs/2402.07647) | 本论文介绍了GRILLBot在实践中的应用，该系统是用于复杂实际任务的多模态助手，并处理了开发和部署中的实际问题和挑战。作者提出了一种混合架构，利用大型语言模型和专门模型来保证任务导向的问题回答和实时任务调整的性能和低延迟，以及通过代码生成方法实现的对话状态管理。该论文对于构建适应性会话任务助手具有重要的创新和贡献。 |
| [^3] | [A Hierarchical Neural Framework for Classification and its Explanation in Large Unstructured Legal Documents.](http://arxiv.org/abs/2309.10563) | 本论文提出了一个名为MESc的分层神经框架，用于分类和解释大型非结构化法律文件。通过将文件分成多个部分并使用大型语言模型的嵌入和无监督聚类，该框架能够实现从长文档中预测判决并提取解释。 |

# 详细

[^1]: JMLR：联合医疗LLM和检索训练以增强推理和专业问题回答能力

    JMLR: Joint Medical LLM and Retrieval Training for Enhancing Reasoning and Professional Question Answering Capability

    [https://arxiv.org/abs/2402.17887](https://arxiv.org/abs/2402.17887)

    JMLR通过联合训练信息检索系统和大型语言模型，在医学领域提高问题回答系统性能，降低计算资源需求，增强模型利用医疗知识进行推理和回答问题的能力。

    

    随着医疗数据的爆炸性增长和人工智能技术的快速发展，精准医学已经成为增强医疗服务质量和效率的关键。在这种背景下，大型语言模型（LLMs）在医疗知识获取和问题回答系统中发挥越来越重要的作用。为了进一步提高这些系统在医学领域的性能，我们介绍了一种创新方法，在微调阶段同时训练信息检索（IR）系统和LLM。我们称之为联合医疗LLM和检索训练（JMLR）的方法旨在克服传统模型在处理医学问题回答任务时面临的挑战。通过采用同步训练机制，JMLR减少了对计算资源的需求，并增强了模型利用医疗知识进行推理和回答问题的能力。

    arXiv:2402.17887v1 Announce Type: new  Abstract: With the explosive growth of medical data and the rapid development of artificial intelligence technology, precision medicine has emerged as a key to enhancing the quality and efficiency of healthcare services. In this context, Large Language Models (LLMs) play an increasingly vital role in medical knowledge acquisition and question-answering systems. To further improve the performance of these systems in the medical domain, we introduce an innovative method that jointly trains an Information Retrieval (IR) system and an LLM during the fine-tuning phase. This approach, which we call Joint Medical LLM and Retrieval Training (JMLR), is designed to overcome the challenges faced by traditional models in handling medical question-answering tasks. By employing a synchronized training mechanism, JMLR reduces the demand for computational resources and enhances the model's ability to leverage medical knowledge for reasoning and answering question
    
[^2]: GRILLBot在实践中的应用：部署大型语言模型以建立适应性会话任务助手的经验与权衡

    GRILLBot In Practice: Lessons and Tradeoffs Deploying Large Language Models for Adaptable Conversational Task Assistants

    [https://arxiv.org/abs/2402.07647](https://arxiv.org/abs/2402.07647)

    本论文介绍了GRILLBot在实践中的应用，该系统是用于复杂实际任务的多模态助手，并处理了开发和部署中的实际问题和挑战。作者提出了一种混合架构，利用大型语言模型和专门模型来保证任务导向的问题回答和实时任务调整的性能和低延迟，以及通过代码生成方法实现的对话状态管理。该论文对于构建适应性会话任务助手具有重要的创新和贡献。

    

    我们致力于解决构建复杂实际任务的实际多模态助手的难题。我们描述了开发和部署GRILLBot的实践性和挑战性，该系统是Alexa Prize TaskBot挑战赛中获得第一和第二名的系统（分别在2022年和2023年）。在我们的开放助手工具包（OAT）框架的基础上，我们提出了一种混合架构，利用大型语言模型（LLMs）和为需要非常低延迟的特定子任务调优的专门模型。OAT使我们能够以结构化且可部署的方式定义何时、如何以及使用哪些LLMs。对于知识驱动的问题回答和实时任务调整，我们展示了LLM在任务背景和世界知识上的推理能力超过延迟问题。对于对话状态管理，我们实现了一种代码生成方法，并展示了专门的较小模型具有84％的有效性和100倍的低延迟。总体而言，我们提供了洞见，并讨论了权衡选择。

    We tackle the challenge of building real-world multimodal assistants for complex real-world tasks. We describe the practicalities and challenges of developing and deploying GRILLBot, a leading (first and second prize winning in 2022 and 2023) system deployed in the Alexa Prize TaskBot Challenge. Building on our Open Assistant Toolkit (OAT) framework, we propose a hybrid architecture that leverages Large Language Models (LLMs) and specialised models tuned for specific subtasks requiring very low latency. OAT allows us to define when, how and which LLMs should be used in a structured and deployable manner. For knowledge-grounded question answering and live task adaptations, we show that LLM reasoning abilities over task context and world knowledge outweigh latency concerns. For dialogue state management, we implement a code generation approach and show that specialised smaller models have 84% effectiveness with 100x lower latency. Overall, we provide insights and discuss tradeoffs for de
    
[^3]: 一个用于分类和解释大型非结构化法律文件的分层神经框架

    A Hierarchical Neural Framework for Classification and its Explanation in Large Unstructured Legal Documents. (arXiv:2309.10563v1 [cs.IR])

    [http://arxiv.org/abs/2309.10563](http://arxiv.org/abs/2309.10563)

    本论文提出了一个名为MESc的分层神经框架，用于分类和解释大型非结构化法律文件。通过将文件分成多个部分并使用大型语言模型的嵌入和无监督聚类，该框架能够实现从长文档中预测判决并提取解释。

    

    自动法律判决预测及其解释常常面临长达数万字的案例文件和非统一结构的问题。在没有结构标注的文件上预测判决并提取解释变得更具挑战性。本论文将这一问题定义为“稀缺标注法律文件”，并通过一种称为MESc（基于多阶段编码器的带聚类的监督）的深度学习分类框架来探索缺乏结构信息和长文档的特点。具体来说，我们将文档分成多个部分，从自定义微调的大型语言模型的最后四个层中提取它们的嵌入，并试图通过无监督聚类来近似它们的结构。然后，我们利用另一组Transformer编码器层学习部分之间的表示。我们探索了多十亿参数的大型语言模型在这种情况下的适应性。

    Automatic legal judgment prediction and its explanation suffer from the problem of long case documents exceeding tens of thousands of words, in general, and having a non-uniform structure. Predicting judgments from such documents and extracting their explanation becomes a challenging task, more so on documents with no structural annotation. We define this problem as "scarce annotated legal documents" and explore their lack of structural information and their long lengths with a deep learning-based classification framework which we call MESc; "Multi-stage Encoder-based Supervised with-clustering"; for judgment prediction. Specifically, we divide a document into parts to extract their embeddings from the last four layers of a custom fine-tuned Large Language Model, and try to approximate their structure through unsupervised clustering. Which we use in another set of transformer encoder layers to learn the inter-chunk representations. We explore the adaptability of LLMs with multi-billion
    

