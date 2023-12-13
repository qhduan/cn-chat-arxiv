# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Talk2Care: Facilitating Asynchronous Patient-Provider Communication with Large-Language-Model.](http://arxiv.org/abs/2309.09357) | 本研究利用大型语言模型（LLMs）来促进患者和医生之间的异步通信，通过访谈研究了解了他们对LLMs的需求，并构建了一个名为Talk2Care的LLM驱动的通信系统。 |
| [^2] | [Agents: An Open-source Framework for Autonomous Language Agents.](http://arxiv.org/abs/2309.07870) | Agents是一个开源框架，支持构建自主语言代理的各种功能，并提供用户友好的接口和对研究人员的扩展性。 |
| [^3] | [Fly-Swat or Cannon? Cost-Effective Language Model Choice via Meta-Modeling.](http://arxiv.org/abs/2308.06077) | 本文提出了一种经济有效的语言模型选择框架（CELMOC），通过元模型预测在不同输入上表现良好的语言模型，从而在低成本下实现高整体性能。 |
| [^4] | [VisoGender: A dataset for benchmarking gender bias in image-text pronoun resolution.](http://arxiv.org/abs/2306.12424) | VisoGender是一个用于评估视觉语言模型中职业相关性别偏见的数据集。研究显示，最先进的模型缺乏正确解析复杂场景中性别的推理能力，生成字幕的模型通常比类似CLIP的模型更精确和更少偏见。 |
| [^5] | [Understanding the Effectiveness of Early Weight Averaging for Training Large Language Models.](http://arxiv.org/abs/2306.03241) | 本文研究了使用早期权重平均化方法来提高大型语言模型质量的有效性，证明该方法可以加速收敛且测试和零样本泛化效果显著，同时有效缓解了训练中的损失波动问题。 |
| [^6] | [Knowledge Refinement via Interaction Between Search Engines and Large Language Models.](http://arxiv.org/abs/2305.07402) | 本文介绍了一种新的框架InteR，通过搜索引擎和大型语言模型之间的交互促进知识精炼，从而提高检索准确性。 |
| [^7] | [USNID: A Framework for Unsupervised and Semi-supervised New Intent Discovery.](http://arxiv.org/abs/2304.07699) | 该论文提出了一个名为USNID的框架，用于无监督和半监督的新意图发现，解决了利用有限或无标记数据时难以捕捉复杂语义的问题，并设计了聚类机制来提高自我监督目标的质量，从而发现细粒度的意图簇。 |
| [^8] | [Highlighting Named Entities in Input for Auto-Formulation of Optimization Problems.](http://arxiv.org/abs/2212.13201) | 本文介绍了一种将线性规划单词问题转换为数学公式的方法。我们利用输入中的命名实体并增强输入以突出这些实体，从而实现了高准确度，赢得了NL4Opt竞赛生成赛道的第一名。 |

# 详细

[^1]: Talk2Care: 利用大型语言模型促进异步患者-医生通信

    Talk2Care: Facilitating Asynchronous Patient-Provider Communication with Large-Language-Model. (arXiv:2309.09357v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2309.09357](http://arxiv.org/abs/2309.09357)

    本研究利用大型语言模型（LLMs）来促进患者和医生之间的异步通信，通过访谈研究了解了他们对LLMs的需求，并构建了一个名为Talk2Care的LLM驱动的通信系统。

    

    尽管有大量的远程医疗应用程序来帮助家庭中的老年人和医疗提供者，但基本的消息和电话仍然是最常见的通信方法，这些方法存在有限的可用性、信息丢失和流程效率低下的问题。促进患者-医生通信的一个有希望的解决方案是利用大型语言模型(LLMs)及其强大的自然对话和摘要能力。然而，对于LLMs在通信过程中的作用还存在有限的理解。我们首先进行了两项访谈研究，分别与老年人(N=10)和医疗提供者(N=9)进行了交流，以了解他们在患者-医生异步通信中对LLMs的需求和机会。基于这些见解，我们构建了一个LLM驱动的通信系统Talk2Care，并为两个群体设计了交互组件: (1) 对于老年人，我们利用语音助手的便利性和易于获取性，构建了一个LLM驱动的语音助手

    Despite the plethora of telehealth applications to assist home-based older adults and healthcare providers, basic messaging and phone calls are still the most common communication methods, which suffer from limited availability, information loss, and process inefficiencies. One promising solution to facilitate patient-provider communication is to leverage large language models (LLMs) with their powerful natural conversation and summarization capability. However, there is a limited understanding of LLMs' role during the communication. We first conducted two interview studies with both older adults (N=10) and healthcare providers (N=9) to understand their needs and opportunities for LLMs in patient-provider asynchronous communication. Based on the insights, we built an LLM-powered communication system, Talk2Care, and designed interactive components for both groups: (1) For older adults, we leveraged the convenience and accessibility of voice assistants (VAs) and built an LLM-powered VA i
    
[^2]: 自主语言代理的开源框架：Agents

    Agents: An Open-source Framework for Autonomous Language Agents. (arXiv:2309.07870v1 [cs.CL])

    [http://arxiv.org/abs/2309.07870](http://arxiv.org/abs/2309.07870)

    Agents是一个开源框架，支持构建自主语言代理的各种功能，并提供用户友好的接口和对研究人员的扩展性。

    

    最近大型语言模型（LLMs）的高级进展使研究人员和开发人员能够构建自主语言代理，这些代理能够通过自然语言接口自动解决各种任务并与环境、人类和其他代理交互。我们将语言代理视为人工通用智能的有前途的方向，并发布Agents，一个开源库，旨在向更广泛的非专业人士开放这些进展。Agents经过精心设计，支持重要功能，包括规划、记忆、工具使用、多代理通信和细粒度的符号控制。Agents用户友好，使非专业人士能够在不需要编写太多代码的情况下构建、定制、测试、调优和部署最先进的自主语言代理。该库也对研究人员友好，其模块化设计使其易于扩展。

    Recent advances on large language models (LLMs) enable researchers and developers to build autonomous language agents that can automatically solve various tasks and interact with environments, humans, and other agents using natural language interfaces. We consider language agents as a promising direction towards artificial general intelligence and release Agents, an open-source library with the goal of opening up these advances to a wider non-specialist audience. Agents is carefully engineered to support important features including planning, memory, tool usage, multi-agent communication, and fine-grained symbolic control. Agents is user-friendly as it enables non-specialists to build, customize, test, tune, and deploy state-of-the-art autonomous language agents without much coding. The library is also research-friendly as its modularized design makes it easily extensible for researchers. Agents is available at https://github.com/aiwaves-cn/agents.
    
[^3]: 飞拍或大炮？通过元模型选择经济有效的语言模型

    Fly-Swat or Cannon? Cost-Effective Language Model Choice via Meta-Modeling. (arXiv:2308.06077v1 [cs.CL])

    [http://arxiv.org/abs/2308.06077](http://arxiv.org/abs/2308.06077)

    本文提出了一种经济有效的语言模型选择框架（CELMOC），通过元模型预测在不同输入上表现良好的语言模型，从而在低成本下实现高整体性能。

    

    生成式语言模型在数据科学领域中变得无处不在。对于各种任务，可以将输入作为自然语言提示，通过LM的输出来提取解决方案。LM的性能随着模型大小的增加而不断提高，但同时查询越来越大的模型的经济成本也在增加。然而，不是所有的输入都很难：有些输入需要更大的LM才能获得令人满意的解决方案，而对于其他输入，较小的LM就足够了。基于这个事实，我们设计了一个经济有效的语言模型选择框架（CELMOC）。给定一组输入和一组候选LM，CELMOC根据所谓的元模型聪明地将每个输入分配给一个在该输入上预测表现良好的LM，以期在低成本下实现高整体性能。用户可以灵活调整成本与性能的权衡。选项包括，最大化总体性能（或处理输入的数量）等。

    Generative language models (LMs) have become omnipresent across data science. For a wide variety of tasks, inputs can be phrased as natural language prompts for an LM, from whose output the solution can then be extracted. LM performance has consistently been increasing with model size - but so has the monetary cost of querying the ever larger models. Importantly, however, not all inputs are equally hard: some require larger LMs for obtaining a satisfactory solution, whereas for others smaller LMs suffice. Based on this fact, we design a framework for Cost-Effective Language Model Choice (CELMOC). Given a set of inputs and a set of candidate LMs, CELMOC judiciously assigns each input to an LM predicted to do well on the input according to a so-called meta-model, aiming to achieve high overall performance at low cost. The cost-performance trade-off can be flexibly tuned by the user. Options include, among others, maximizing total expected performance (or the number of processed inputs) w
    
[^4]: VisoGender：一份用于评估图像-文本代词解析中性别偏见的数据集

    VisoGender: A dataset for benchmarking gender bias in image-text pronoun resolution. (arXiv:2306.12424v1 [cs.CV])

    [http://arxiv.org/abs/2306.12424](http://arxiv.org/abs/2306.12424)

    VisoGender是一个用于评估视觉语言模型中职业相关性别偏见的数据集。研究显示，最先进的模型缺乏正确解析复杂场景中性别的推理能力，生成字幕的模型通常比类似CLIP的模型更精确和更少偏见。

    

    我们介绍了一个新的数据集VisoGender，用于评估视觉语言模型中的性别偏见。我们专注于职业相关的性别偏见，受Winograd和Winogender模式的启发，其中每个图像都与包含场景中主语和宾语代词关系的标题相关联。VisoGender在职业角色中平衡了性别代表，支持两种偏见评估方式：i）解决偏见，我们评估男性和女性解决准确性之间的差异；ii）检索偏见，我们比较在性别中立的搜索查询中检索到的男性和女性专业人员的比例。我们对几种最先进的视觉语言模型进行了基准测试，并发现它们缺乏正确解析复杂场景中性别的推理能力。虽然性别偏见的方向和幅度取决于任务和评估的模型，但生成字幕的模型通常比类似CLIP的模型更精确和更少偏见。

    We introduce VisoGender, a novel dataset for benchmarking gender bias in vision-language models. We focus on occupation-related gender biases, inspired by Winograd and Winogender schemas, where each image is associated with a caption containing a pronoun relationship of subjects and objects in the scene. VisoGender is balanced by gender representation in professional roles, supporting bias evaluation in two ways: i) resolution bias, where we evaluate the difference between gender resolution accuracies for men and women and ii) retrieval bias, where we compare ratios of male and female professionals retrieved for a gender-neutral search query. We benchmark several state-of-the-art vision-language models and find that they lack the reasoning abilities to correctly resolve gender in complex scenes. While the direction and magnitude of gender bias depends on the task and the model being evaluated, captioning models generally are more accurate and less biased than CLIP-like models. Dataset 
    
[^5]: 理解早期权重平均对训练大语言模型的有效性

    Understanding the Effectiveness of Early Weight Averaging for Training Large Language Models. (arXiv:2306.03241v1 [cs.LG])

    [http://arxiv.org/abs/2306.03241](http://arxiv.org/abs/2306.03241)

    本文研究了使用早期权重平均化方法来提高大型语言模型质量的有效性，证明该方法可以加速收敛且测试和零样本泛化效果显著，同时有效缓解了训练中的损失波动问题。

    

    训练大型语言模型代价高昂，最近的研究表明训练至收敛并不高效。在本文中，我们研究了一种简单的想法，即在训练过程中沿着轨迹进行检查点平均化，以在模型收敛之前提高其质量。这种方法在训练或推理期间不会产生额外的成本。具体而言，我们分析了具有10亿到120亿参数的Pythia LLM的训练轨迹，并证明特别是在训练的早期和中期阶段，这种想法可以加速收敛并提高测试和零样本泛化效果。损失波动是LLM训练中众所周知的问题；在我们的分析中，我们遇到了两种基础轨迹的这种情况，并且我们的平均化可以缓解这两种情况。例如，对于一个拥有69亿参数的LLM，我们的早期权重平均化配方可以节省高达4200小时的GPU时间，这对云计算成本来说是显著的节约。

    Training LLMs is expensive, and recent evidence indicates training all the way to convergence is inefficient. In this paper, we investigate the ability of a simple idea, checkpoint averaging along the trajectory of a training run to improve the quality of models before they have converged. This approach incurs no extra cost during training or inference. Specifically, we analyze the training trajectories of Pythia LLMs with 1 to 12 billion parameters and demonstrate that, particularly during the early to mid stages of training, this idea accelerates convergence and improves both test and zero-shot generalization. Loss spikes are a well recognized problem in LLM training; in our analysis we encountered two instances of this in the underlying trajectories, and both instances were mitigated by our averaging.  For a 6.9B parameter LLM, for example, our early weight averaging recipe can save upto 4200 hours of GPU time, which corresponds to significant savings in cloud compute costs.
    
[^6]: 搜索引擎与大型语言模型间的交互优化知识精炼

    Knowledge Refinement via Interaction Between Search Engines and Large Language Models. (arXiv:2305.07402v1 [cs.CL])

    [http://arxiv.org/abs/2305.07402](http://arxiv.org/abs/2305.07402)

    本文介绍了一种新的框架InteR，通过搜索引擎和大型语言模型之间的交互促进知识精炼，从而提高检索准确性。

    

    信息检索在从大量数据中定位相关资源方面具有重要作用，其应用已从传统知识库发展至现代搜索引擎（SEs）。大型语言模型（LLMs）的出现进一步通过使用自然语言与搜索系统交互革命性地改变了该领域。本文探索了LLMs和SEs的优缺点，强调它们在理解用户查询和检索最新信息方面的各自优势。为了利用两种范例的优势并避免其限制，我们提出了InteR，这是一个通过SEs和LLMs之间的交互促进知识精炼的新框架。 InteR使SEs能够使用LLM生成的摘要来调整查询，同时使LLMs能够使用SE检索到的文档来增强提示。这种迭代的精炼过程增强了SEs和LLMs的输入，从而导致更准确的检索结果。

    Information retrieval (IR) plays a crucial role in locating relevant resources from vast amounts of data, and its applications have evolved from traditional knowledge bases to modern search engines (SEs). The emergence of large language models (LLMs) has further revolutionized the field by enabling users to interact with search systems in natural language. In this paper, we explore the advantages and disadvantages of LLMs and SEs, highlighting their respective strengths in understanding user-issued queries and retrieving up-to-date information. To leverage the benefits of both paradigms while circumventing their limitations, we propose InteR, a novel framework that facilitates knowledge refinement through interaction between SEs and LLMs. InteR allows SEs to refine knowledge in query using LLM-generated summaries and enables LLMs to enhance prompts using SE-retrieved documents. This iterative refinement process augments the inputs of SEs and LLMs, leading to more accurate retrieval. Ex
    
[^7]: USNID: 无监督和半监督新意图发现的框架

    USNID: A Framework for Unsupervised and Semi-supervised New Intent Discovery. (arXiv:2304.07699v1 [cs.CL])

    [http://arxiv.org/abs/2304.07699](http://arxiv.org/abs/2304.07699)

    该论文提出了一个名为USNID的框架，用于无监督和半监督的新意图发现，解决了利用有限或无标记数据时难以捕捉复杂语义的问题，并设计了聚类机制来提高自我监督目标的质量，从而发现细粒度的意图簇。

    

    新意图发现对自然语言处理非常有价值，使我们更好地理解用户需求并提供友好的服务。然而，在有限或没有标记数据的情况下，大多数现有方法难以捕捉离散文本表示的复杂语义。为了解决这个问题，我们提出了一种名为USNID的新框架，用于无监督和半监督新意图发现，具有三个关键技术：充分利用无监督或半监督数据挖掘浅层语义相似性关系；设计聚类机制解决簇分配不一致的问题；捕获无监督或半监督数据中的高级语义，通过同时优化聚类和自我监督来发现细粒度的意图簇。

    New intent discovery is of great value to natural language processing, allowing for a better understanding of user needs and providing friendly services. However, most existing methods struggle to capture the complicated semantics of discrete text representations when limited or no prior knowledge of labeled data is available. To tackle this problem, we propose a novel framework called USNID for unsupervised and semi-supervised new intent discovery, which has three key technologies. First, it takes full use of unsupervised or semi-supervised data to mine shallow semantic similarity relations and provide well-initialized representations for clustering. Second, it designs a centroid-guided clustering mechanism to address the issue of cluster allocation inconsistency and provide high-quality self-supervised targets for representation learning. Third, it captures high-level semantics in unsupervised or semi-supervised data to discover fine-grained intent-wise clusters by optimizing both cl
    
[^8]: 输入命名实体自动解析优化问题的方法研究

    Highlighting Named Entities in Input for Auto-Formulation of Optimization Problems. (arXiv:2212.13201v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2212.13201](http://arxiv.org/abs/2212.13201)

    本文介绍了一种将线性规划单词问题转换为数学公式的方法。我们利用输入中的命名实体并增强输入以突出这些实体，从而实现了高准确度，赢得了NL4Opt竞赛生成赛道的第一名。

    

    运筹学是将现实世界问题建模为数学优化问题来解决的。虽然解决数学系统的问题是由分析软件完成的，但将问题作为一组数学操作进行表达通常是由领域专家手动完成的。最近的机器学习方法显示出将文本问题描述转换为相应的数学公式的前景。本文提出了一种将线性规划单词问题转换为数学公式的方法。我们利用输入中的命名实体并增强输入以突出这些实体。我们的方法在NL4Opt竞赛的所有提交中获得了最高的准确性，获得了生成赛道的第一名。

    Operations research deals with modeling and solving real-world problems as mathematical optimization problems. While solving mathematical systems is accomplished by analytical software, formulating a problem as a set of mathematical operations has been typically done manually by domain experts. Recent machine learning methods have shown promise in converting textual problem descriptions to corresponding mathematical formulations. This paper presents an approach that converts linear programming word problems into mathematical formulations. We leverage the named entities in the input and augment the input to highlight these entities. Our approach achieves the highest accuracy among all submissions to the NL4Opt Competition, securing first place in the generation track.
    

