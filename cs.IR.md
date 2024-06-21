# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Generation of Asset Administration Shell with Large Language Model Agents: Interoperability in Digital Twins with Semantic Node](https://arxiv.org/abs/2403.17209) | 通过大型语言模型代理生成AAS实例模型，实现了在数字孪生中的互操作性，降低了手动创建成本和时间。 |
| [^2] | [ERASE: Benchmarking Feature Selection Methods for Deep Recommender Systems](https://arxiv.org/abs/2403.12660) | 深度推荐系统中的特征选择方法研究面临着公平比较、选择属性分析缺乏以及过度关注峰值性能等挑战。 |
| [^3] | [LLM-Ensemble: Optimal Large Language Model Ensemble Method for E-commerce Product Attribute Value Extraction](https://arxiv.org/abs/2403.00863) | 提出了一种名为LLM-ensemble的算法，用于集成不同大型语言模型，以提高电子商务产品属性值提取的性能。 |
| [^4] | [MSynFD: Multi-hop Syntax aware Fake News Detection](https://arxiv.org/abs/2402.14834) | 提出一种新的多跳语法感知假新闻检测方法，通过引入补充的语法信息来处理假新闻中的微妙转折 |
| [^5] | [Multi-Behavior Collaborative Filtering with Partial Order Graph Convolutional Networks](https://arxiv.org/abs/2402.07659) | 这项研究提出了一种新的方法，使用部分顺序图卷积网络来解决单一图形中多个行为的协同过滤问题。该方法通过定义多个行为之间的部分顺序关系，并利用加权边合并行为图，实现了在主要任务和辅助任务上都表现良好的联合嵌入。 |
| [^6] | [Prompt Perturbation in Retrieval-Augmented Generation based Large Language Models](https://arxiv.org/abs/2402.07179) | 本文研究了基于检索增强生成的大型语言模型（LLM）中提示扰动的影响，并引入了一种新的优化技术GGPP。通过GGPP，我们可以将LLMs的输出引导到特定的错误答案，并应对提示中的无关上下文。 |
| [^7] | [Human Action Co-occurrence in Lifestyle Vlogs using Graph Link Prediction.](http://arxiv.org/abs/2309.06219) | 该论文提出了自动识别人类动作共现的任务，并创建了ACE数据集以及相应的代码。通过利用视觉和文本信息的图链接预测模型，可以有效捕捉不同数据域中的人类动作关系。 |
| [^8] | [Advancing continual lifelong learning in neural information retrieval: definition, dataset, framework, and empirical evaluation.](http://arxiv.org/abs/2308.08378) | 本文提出了一个系统的持续神经信息检索任务定义，并提供了一个模拟连续信息检索的多主题数据集。同时，还提出了一个全面的持续神经信息检索框架，能够防止灾难性遗忘并提高先前学习任务的性能。 |

# 详细

[^1]: 利用大型语言模型代理生成资产管理外壳：数字孪生和语义节点中的互操作性

    Generation of Asset Administration Shell with Large Language Model Agents: Interoperability in Digital Twins with Semantic Node

    [https://arxiv.org/abs/2403.17209](https://arxiv.org/abs/2403.17209)

    通过大型语言模型代理生成AAS实例模型，实现了在数字孪生中的互操作性，降低了手动创建成本和时间。

    

    这项研究介绍了一种新颖的方法，用于协助在工业4.0背景下为数字孪生建模创建资产管理外壳（AAS）实例，旨在增强智能制造中的互操作性，减少手动工作。我们构建了一个“语义节点”数据结构来捕捉文本数据的语义要义。然后，设计并实现了一个由大型语言模型驱动的系统，用于处理“语义节点”并从文本技术数据生成AAS实例模型。我们的评估表明，有效生成率为62-79%，表明相当比例的手动创建工作可以转换为更容易的验证工作，从而减少创建AAS实例模型的时间和成本。在我们的评估中，对不同LLM的比较分析以及检索增强生成（RAG）机制的深入消融研究提供了有关LLM有效性的见解。

    arXiv:2403.17209v1 Announce Type: new  Abstract: This research introduces a novel approach for assisting the creation of Asset Administration Shell (AAS) instances for digital twin modeling within the context of Industry 4.0, aiming to enhance interoperability in smart manufacturing and reduce manual effort. We construct a "semantic node" data structure to capture the semantic essence of textual data. Then, a system powered by large language models is designed and implemented to process "semantic node" and generate AAS instance models from textual technical data. Our evaluation demonstrates a 62-79% effective generation rate, indicating a substantial proportion of manual creation effort can be converted into easier validation effort, thereby reducing the time and cost in creating AAS instance models. In our evaluation, a comparative analysis of different LLMs and an in-depth ablation study of Retrieval-Augmented Generation (RAG) mechanisms provide insights into the effectiveness of LLM
    
[^2]: ERASE：深度推荐系统特征选择方法的基准测试

    ERASE: Benchmarking Feature Selection Methods for Deep Recommender Systems

    [https://arxiv.org/abs/2403.12660](https://arxiv.org/abs/2403.12660)

    深度推荐系统中的特征选择方法研究面临着公平比较、选择属性分析缺乏以及过度关注峰值性能等挑战。

    

    深度推荐系统(DRS)越来越依赖于大量特征字段来提供更精准的推荐。有效的特征选择方法因此变得至关重要，以进一步提高准确性并优化存储效率，以满足部署需求。研究领域，特别是在DRS的背景下，尚处于初期阶段，面临三个核心挑战：首先，研究论文之间实验设置的差异往往导致不公平比较，遮蔽了实践见解。其次，现有文献缺乏基于大规模数据集的选择属性的详细分析，并且缺乏对选择技术和DRS骨干之间进行全面比较的限制性文章的通用性研究和部署。最后，研究往往专注于比较特征选择方法可达到的峰值性能，这种方法通常在计算方面不足。

    arXiv:2403.12660v1 Announce Type: cross  Abstract: Deep Recommender Systems (DRS) are increasingly dependent on a large number of feature fields for more precise recommendations. Effective feature selection methods are consequently becoming critical for further enhancing the accuracy and optimizing storage efficiencies to align with the deployment demands. This research area, particularly in the context of DRS, is nascent and faces three core challenges. Firstly, variant experimental setups across research papers often yield unfair comparisons, obscuring practical insights. Secondly, the existing literature's lack of detailed analysis on selection attributes, based on large-scale datasets and a thorough comparison among selection techniques and DRS backbones, restricts the generalizability of findings and impedes deployment on DRS. Lastly, research often focuses on comparing the peak performance achievable by feature selection methods, an approach that is typically computationally infe
    
[^3]: LLM-Ensemble: 用于电子商务产品属性值提取的最佳大型语言模型集成方法

    LLM-Ensemble: Optimal Large Language Model Ensemble Method for E-commerce Product Attribute Value Extraction

    [https://arxiv.org/abs/2403.00863](https://arxiv.org/abs/2403.00863)

    提出了一种名为LLM-ensemble的算法，用于集成不同大型语言模型，以提高电子商务产品属性值提取的性能。

    

    arXiv:2403.00863v1 公告类型:跨领域摘要: 产品属性值提取是自然语言处理（NLP）和当代电子商务行业中至关重要的组成部分。提供精确的产品属性值在确保高质量推荐和提升客户满意度方面至关重要。最近出现的大型语言模型（LLMs）在许多属性提取任务中表现出最新技术水平，而无需进行领域特定的训练数据。然而，由于数据、架构和超参数的多样性，不同LLMs表现出不同的优势和劣势。这种变化使它们彼此互补，没有哪个LLM能完全压倒其他LLM。考虑到LLMs的多样优势和劣势，开发一种利用它们互补潜力的集成方法变得必要。在本文中，我们提出了一种名为LLM-ensemble的新算法，用于集成不同LLMs。

    arXiv:2403.00863v1 Announce Type: cross  Abstract: Product attribute value extraction is a pivotal component in Natural Language Processing (NLP) and the contemporary e-commerce industry. The provision of precise product attribute values is fundamental in ensuring high-quality recommendations and enhancing customer satisfaction. The recently emerging Large Language Models (LLMs) have demonstrated state-of-the-art performance in numerous attribute extraction tasks, without the need for domain-specific training data. Nevertheless, varying strengths and weaknesses are exhibited by different LLMs due to the diversity in data, architectures, and hyperparameters. This variation makes them complementary to each other, with no single LLM dominating all others. Considering the diverse strengths and weaknesses of LLMs, it becomes necessary to develop an ensemble method that leverages their complementary potentials. In this paper, we propose a novel algorithm called LLM-ensemble to ensemble diffe
    
[^4]: MSynFD: 多跳语法感知假新闻检测

    MSynFD: Multi-hop Syntax aware Fake News Detection

    [https://arxiv.org/abs/2402.14834](https://arxiv.org/abs/2402.14834)

    提出一种新的多跳语法感知假新闻检测方法，通过引入补充的语法信息来处理假新闻中的微妙转折

    

    社交媒体平台的广泛传播助长了假新闻的快速传播，对我们的现实社会构成威胁。现有方法使用多模态数据或上下文信息来增强对假新闻的检测，通过分析新闻内容和/或其社会背景。然而，这些方法常常忽视了基本的文本新闻内容（文章），并且过分依赖序列建模和全局注意力来提取语义信息。这些现有方法无法处理新闻文章中的复杂、微妙的转折，比如句法-语义不匹配和先验偏差，导致性能较低，并在缺失模态或社会背景时可能失败。为了弥合这些重要差距，我们提出了一种新颖的多跳语法感知假新闻检测（MSynFD）方法，该方法融合了补充的语法信息，以处理假新闻中的微妙转折。

    arXiv:2402.14834v1 Announce Type: cross  Abstract: The proliferation of social media platforms has fueled the rapid dissemination of fake news, posing threats to our real-life society. Existing methods use multimodal data or contextual information to enhance the detection of fake news by analyzing news content and/or its social context. However, these methods often overlook essential textual news content (articles) and heavily rely on sequential modeling and global attention to extract semantic information. These existing methods fail to handle the complex, subtle twists in news articles, such as syntax-semantics mismatches and prior biases, leading to lower performance and potential failure when modalities or social context are missing. To bridge these significant gaps, we propose a novel multi-hop syntax aware fake news detection (MSynFD) method, which incorporates complementary syntax information to deal with subtle twists in fake news. Specifically, we introduce a syntactical depen
    
[^5]: 具有部分顺序图卷积网络的多行为协同过滤

    Multi-Behavior Collaborative Filtering with Partial Order Graph Convolutional Networks

    [https://arxiv.org/abs/2402.07659](https://arxiv.org/abs/2402.07659)

    这项研究提出了一种新的方法，使用部分顺序图卷积网络来解决单一图形中多个行为的协同过滤问题。该方法通过定义多个行为之间的部分顺序关系，并利用加权边合并行为图，实现了在主要任务和辅助任务上都表现良好的联合嵌入。

    

    在单一图形协作过滤（CF）向量中表示多个行为的信息一直是一个长期存在的挑战。这是因为不同的行为自然形成单独的行为图，并学习单独的CF嵌入。现有模型通过指定某些行为的CF嵌入作为主要嵌入，并利用其他辅助工具来增强主要嵌入来合并这些单独的嵌入。然而，这种方法通常在主要任务上联合嵌入表现良好，但在辅助任务上表现不佳。为了解决由单独的行为图引起的问题，我们提出了部分顺序图（POG）的概念。POG定义了多个行为的部分顺序关系，并将行为组合建模为带有权重的边，以将单独的行为图合并成一个联合的POG。理论证明了POG可以推广到任何给定的多个行为集。基于POG，我们提出了定制的部分顺序模型。

    Representing the information of multiple behaviors in the single graph collaborative filtering (CF) vector has been a long-standing challenge. This is because different behaviors naturally form separate behavior graphs and learn separate CF embeddings. Existing models merge the separate embeddings by appointing the CF embeddings for some behaviors as the primary embedding and utilizing other auxiliaries to enhance the primary embedding. However, this approach often results in the joint embedding performing well on the main tasks but poorly on the auxiliary ones. To address the problem arising from the separate behavior graphs, we propose the concept of Partial Order Graphs (POG). POG defines the partial order relation of multiple behaviors and models behavior combinations as weighted edges to merge separate behavior graphs into a joint POG. Theoretical proof verifies that POG can be generalized to any given set of multiple behaviors. Based on POG, we propose the tailored Partial Order 
    
[^6]: 在基于检索增强生成的大型语言模型中进行提示扰动

    Prompt Perturbation in Retrieval-Augmented Generation based Large Language Models

    [https://arxiv.org/abs/2402.07179](https://arxiv.org/abs/2402.07179)

    本文研究了基于检索增强生成的大型语言模型（LLM）中提示扰动的影响，并引入了一种新的优化技术GGPP。通过GGPP，我们可以将LLMs的输出引导到特定的错误答案，并应对提示中的无关上下文。

    

    大型语言模型（LLM）的鲁棒性在其在各个领域的使用迅速增长中变得越来越重要。检索增强生成（RAG）被视为提高从LLM生成文本的可信度的方法。然而，目前对RAG-based LLMs的输出如何受到稍有不同的输入影响的研究还不够充分。在本文中，我们发现即使在提示中插入一个很短的前缀也会导致生成的输出与事实正确答案相去甚远。我们系统地评估了这类前缀对RAG的影响，并引入了一种称为Gradient Guided Prompt Perturbation（GGPP）的新型优化技术。GGPP在将RAG-based LLMs的输出引导到特定错误答案方面取得了很高的成功率。它还可以应对提示中请求忽略无关上下文的指令。我们还利用LLMs在带有和不带有GGPP扰动的提示之间的神经元激活差异来提供一种改进方法。

    The robustness of large language models (LLMs) becomes increasingly important as their use rapidly grows in a wide range of domains. Retrieval-Augmented Generation (RAG) is considered as a means to improve the trustworthiness of text generation from LLMs. However, how the outputs from RAG-based LLMs are affected by slightly different inputs is not well studied. In this work, we find that the insertion of even a short prefix to the prompt leads to the generation of outputs far away from factually correct answers. We systematically evaluate the effect of such prefixes on RAG by introducing a novel optimization technique called Gradient Guided Prompt Perturbation (GGPP). GGPP achieves a high success rate in steering outputs of RAG-based LLMs to targeted wrong answers. It can also cope with instructions in the prompts requesting to ignore irrelevant context. We also exploit LLMs' neuron activation difference between prompts with and without GGPP perturbations to give a method that improves
    
[^7]: 使用图链接预测在生活方式vlog中的人类动作共现

    Human Action Co-occurrence in Lifestyle Vlogs using Graph Link Prediction. (arXiv:2309.06219v1 [cs.CV])

    [http://arxiv.org/abs/2309.06219](http://arxiv.org/abs/2309.06219)

    该论文提出了自动识别人类动作共现的任务，并创建了ACE数据集以及相应的代码。通过利用视觉和文本信息的图链接预测模型，可以有效捕捉不同数据域中的人类动作关系。

    

    我们介绍了自动识别人类动作共现的任务，即确定两个人类动作是否可以在同一时间间隔内共现。我们创建并公开了ACE（Action Co-occurrencE）数据集，该数据集由约12k个共现的视觉动作对和它们对应的视频片段组成的大型图形。我们描述了利用视觉和文本信息来自动推断两个动作是否共现的图链接预测模型。我们证明了图形特别适合捕捉人类动作之间的关系，并且所学习的图形表示对于我们的任务是有效的，并且在不同的数据域中捕捉到新颖而相关的信息。本文介绍的ACE数据集和代码可在https://github.com/MichiganNLP/vlog_action_co-occurrence公开获取。

    We introduce the task of automatic human action co-occurrence identification, i.e., determine whether two human actions can co-occur in the same interval of time. We create and make publicly available the ACE (Action Co-occurrencE) dataset, consisting of a large graph of ~12k co-occurring pairs of visual actions and their corresponding video clips. We describe graph link prediction models that leverage visual and textual information to automatically infer if two actions are co-occurring. We show that graphs are particularly well suited to capture relations between human actions, and the learned graph representations are effective for our task and capture novel and relevant information across different data domains. The ACE dataset and the code introduced in this paper are publicly available at https://github.com/MichiganNLP/vlog_action_co-occurrence.
    
[^8]: 推进神经信息检索中的持续终身学习：定义、数据集、框架和实证评估

    Advancing continual lifelong learning in neural information retrieval: definition, dataset, framework, and empirical evaluation. (arXiv:2308.08378v1 [cs.IR])

    [http://arxiv.org/abs/2308.08378](http://arxiv.org/abs/2308.08378)

    本文提出了一个系统的持续神经信息检索任务定义，并提供了一个模拟连续信息检索的多主题数据集。同时，还提出了一个全面的持续神经信息检索框架，能够防止灾难性遗忘并提高先前学习任务的性能。

    

    持续学习是指机器学习模型在学习和适应新信息的同时，不影响其在先前学习任务上的性能。尽管已有多项研究探讨了信息检索任务中的持续学习方法，但仍缺乏明确的任务定义，并且目前尚不清楚在这种背景下典型的学习策略的表现如何。为了应对这一挑战，本文提出了一种系统的持续神经信息检索任务定义，并提供了一个模拟连续信息检索的多主题数据集。随后，本文提出了一个全面的持续神经信息检索框架，包括典型检索模型和持续学习策略。实证评估结果表明，所提出的框架能够成功地防止神经信息检索中的灾难性遗忘，并提高先前学习任务的性能。结果表明，基于嵌入的检索方式较传统的基于索引的检索方式具有优势，并且持续学习策略能够有效地提升检索性能。

    Continual learning refers to the capability of a machine learning model to learn and adapt to new information, without compromising its performance on previously learned tasks. Although several studies have investigated continual learning methods for information retrieval tasks, a well-defined task formulation is still lacking, and it is unclear how typical learning strategies perform in this context. To address this challenge, a systematic task formulation of continual neural information retrieval is presented, along with a multiple-topic dataset that simulates continuous information retrieval. A comprehensive continual neural information retrieval framework consisting of typical retrieval models and continual learning strategies is then proposed. Empirical evaluations illustrate that the proposed framework can successfully prevent catastrophic forgetting in neural information retrieval and enhance performance on previously learned tasks. The results indicate that embedding-based retr
    

