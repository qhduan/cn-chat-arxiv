# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Specialized Language Models with Cheap Inference from Limited Domain Data](https://rss.arxiv.org/abs/2402.01093) | 本研究提出了一种使用有限领域数据进行廉价推理的专用语言模型。在研究中，我们通过比较不同的机器学习方法，在推理成本的限制下找到了比训练非常大的基本转换器模型更优的替代方案。具体而言，在大型预训练预算下，超网络和专家混合模型的困惑度更好，而在大型专用预算下，训练重要样本数据集上的小型模型更具吸引力。 |
| [^2] | [Narrative Feature or Structured Feature? A Study of Large Language Models to Identify Cancer Patients at Risk of Heart Failure](https://arxiv.org/abs/2403.11425) | 使用大型语言模型结合新颖的叙述特征，能够有效识别癌症患者患心力衰竭的风险，表现优于传统机器学习模型和深度学习模型。 |
| [^3] | [ERBench: An Entity-Relationship based Automatically Verifiable Hallucination Benchmark for Large Language Models](https://arxiv.org/abs/2403.05266) | ERBench是一个基于实体关系的大型语言模型幻觉基准，通过自动转换任何关系数据库并构建可自动验证的问题，以支持复杂性评估和调试 |
| [^4] | [How does Architecture Influence the Base Capabilities of Pre-trained Language Models? A Case Study Based on FFN-Wider Transformer Models](https://arxiv.org/abs/2403.02436) | 本研究探讨了建筑如何影响预训练语言模型的基本能力，发现了FFN-Wider变压器模型降低了多头注意力的贡献比，从而导致基本能力的下降。 |
| [^5] | [Implicit Bias of Next-Token Prediction](https://arxiv.org/abs/2402.18551) | 在基于梯度的优化器训练下的线性NTP模型中，确定了NTP可分离条件，并证明梯度下降能够实现其下界；同时证明了这些条件在过参数化时仍然成立。 |
| [^6] | [Microstructures and Accuracy of Graph Recall by Large Language Models](https://arxiv.org/abs/2402.11821) | 本研究首次系统研究了大型语言模型对图形召回的准确性和偏见微结构，探讨了它们与人类的异同以及对其他图形推理任务的影响。 |
| [^7] | [Studious Bob Fight Back Against Jailbreaking via Prompt Adversarial Tuning](https://arxiv.org/abs/2402.06255) | 本文提出了一种名为Prompt Adversarial Tuning (PAT)的方法，通过训练一个防御控制机制并将其作为前缀嵌入到用户提示中，实现对大型语言模型（LLMs）的越狱行为的防御。实验证明该方法在保护LLMs免受产生有害信息的影响方面效果显著。 |
| [^8] | [Can Large Language Model Agents Simulate Human Trust Behaviors?](https://arxiv.org/abs/2402.04559) | 大语言模型代理能够模拟人类的信任行为，表现出在信任游戏中的信任行为，并且与人类行为具有高度一致性，但存在一些偏见和对代理与人类的差异。 |
| [^9] | [Factuality of Large Language Models in the Year 2024](https://arxiv.org/abs/2402.02420) | 本文调查了大规模语言模型（LLM）的真实性问题，并对其现有研究进行了批判性分析，指出了改进LLM真实性的挑战和解决方案，以及自动真实性评估的障碍。未来的研究应该关注在这些方面的进一步工作。 |
| [^10] | [AutoTimes: Autoregressive Time Series Forecasters via Large Language Models](https://arxiv.org/abs/2402.02370) | AutoTimes是一种基于大型语言模型的自回归时间序列预测器，利用语言模型的转换能力来处理时间序列数据，实现了与先前模型相当的性能。 |
| [^11] | [Rendering Graphs for Graph Reasoning in Multimodal Large Language Models](https://arxiv.org/abs/2402.02130) | 本文在多模态大型语言模型中为图推理任务引入了视觉信息，并提出了一个新的基准测试数据集GITQA。实验证明，结合文本和视觉信息的结果比单一模态效果更好。 |
| [^12] | [Human-Readable Fingerprint for Large Language Models](https://arxiv.org/abs/2312.04828) | 这项研究介绍了一种大型语言模型的人类可读指纹，可以唯一识别出基本模型，并且不暴露模型参数或干扰训练。通过观察和验证，研究发现模型参数的向量方向在预训练后保持稳定，成为识别基本模型的重要条件。 |
| [^13] | [Tree of Attacks: Jailbreaking Black-Box LLMs Automatically](https://arxiv.org/abs/2312.02119) | 提出了一种名为Tree of Attacks with Pruning (TAP)的自动化方法，用于生成只需要对目标大型语言模型进行黑盒访问的越狱方法，并通过思维树推理和修剪生成准确的越狱提示。 |
| [^14] | [Benchmarking LLMs via Uncertainty Quantification.](http://arxiv.org/abs/2401.12794) | 这项研究提出了一种通过不确定性量化来对LLMs进行基准测试的方法，并引入了一种不确定性感知评估指标UAcc。研究结果显示，高准确度的LLMs可能具有较低的确定性，而大规模LLMs可能比较小规模LLMs更不确定。指令微调倾向于增加LLMs的不确定性。 |
| [^15] | [Pruning for Protection: Increasing Jailbreak Resistance in Aligned LLMs Without Fine-Tuning.](http://arxiv.org/abs/2401.10862) | 本文研究了剪枝对齐的LLMs的保护措施，发现剪枝LLM参数可以显著增强其抵抗“越狱”提示攻击的能力，并且对其他LLM行为也可能有更普遍的效果。同时，引入了一个有害任务数据集，证明剪枝有助于集中注意力在与任务相关的标记上。突出的聊天模型表现出很高的易感性。 |
| [^16] | [Why Should This Article Be Deleted? Transparent Stance Detection in Multilingual Wikipedia Editor Discussions.](http://arxiv.org/abs/2310.05779) | 该研究构建了一个多语言数据集，包含维基百科编辑讨论和决策的推理过程，在解释内容管理决策方面增加了透明度。 |

# 详细

[^1]: 使用有限领域数据进行廉价推理的专用语言模型

    Specialized Language Models with Cheap Inference from Limited Domain Data

    [https://rss.arxiv.org/abs/2402.01093](https://rss.arxiv.org/abs/2402.01093)

    本研究提出了一种使用有限领域数据进行廉价推理的专用语言模型。在研究中，我们通过比较不同的机器学习方法，在推理成本的限制下找到了比训练非常大的基本转换器模型更优的替代方案。具体而言，在大型预训练预算下，超网络和专家混合模型的困惑度更好，而在大型专用预算下，训练重要样本数据集上的小型模型更具吸引力。

    

    大型语言模型已成为一种多才多艺的工具，但在缺乏大规模推理预算和大规模领域内训练集的任务中应用起来具有挑战性。本研究对这些限制进行了形式化，并区分了四个重要的变量：预训练预算（用于在目标领域出现之前进行训练），专用预算（用于在目标领域出现之后进行训练），推理预算和领域内训练集大小。在这些设置中，我们比较了机器学习文献中的不同方法。受到推理成本的限制，我们发现比训练非常大的基本转换器模型的标准做法更好的替代方案。特别是，我们发现超网络和专家混合模型在大型预训练预算下具有更好的困惑度，而在大型专用预算下，训练重要样本数据集上的小型模型更具吸引力。

    Large language models have emerged as a versatile tool but are challenging to apply to tasks lacking large inference budgets and large in-domain training sets. This work formalizes these constraints and distinguishes four important variables: the pretraining budget (for training before the target domain is known), the specialization budget (for training after the target domain is known), the inference budget, and the in-domain training set size. Across these settings, we compare different approaches from the machine learning literature. Limited by inference cost, we find better alternatives to the standard practice of training very large vanilla transformer models. In particular, we show that hyper-networks and mixture of experts have better perplexity for large pretraining budgets, while small models trained on importance sampled datasets are attractive for large specialization budgets.
    
[^2]: 叙事特征还是结构特征？研究大型语言模型以识别患心力衰竭风险的癌症患者

    Narrative Feature or Structured Feature? A Study of Large Language Models to Identify Cancer Patients at Risk of Heart Failure

    [https://arxiv.org/abs/2403.11425](https://arxiv.org/abs/2403.11425)

    使用大型语言模型结合新颖的叙述特征，能够有效识别癌症患者患心力衰竭的风险，表现优于传统机器学习模型和深度学习模型。

    

    癌症治疗已知会引入心毒性，对预后和生存率产生负面影响。识别患心力衰竭（HF）风险的癌症患者对于改善癌症治疗结果和安全性至关重要。本研究使用来自电子健康记录（EHRs）的机器学习（ML）模型，包括传统ML、时间感知长短期记忆（T-LSTM）和使用从结构化医学代码衍生的新颖叙述特征的大型语言模型（LLMs）来识别患HF风险的癌症患者。我们从佛罗里达大学健康中心识别了一组包括12,806名肺癌、乳腺癌和结直肠癌患者的癌症队列，其中1,602人在癌症后发展为HF。LLM GatorTron-3.9B取得了最佳的F1分数，比传统支持向量机高出39%，比T-LSTM深度学习模型高出7%，比广泛使用的Transformer模型BERT高出5.6%。

    arXiv:2403.11425v1 Announce Type: cross  Abstract: Cancer treatments are known to introduce cardiotoxicity, negatively impacting outcomes and survivorship. Identifying cancer patients at risk of heart failure (HF) is critical to improving cancer treatment outcomes and safety. This study examined machine learning (ML) models to identify cancer patients at risk of HF using electronic health records (EHRs), including traditional ML, Time-Aware long short-term memory (T-LSTM), and large language models (LLMs) using novel narrative features derived from the structured medical codes. We identified a cancer cohort of 12,806 patients from the University of Florida Health, diagnosed with lung, breast, and colorectal cancers, among which 1,602 individuals developed HF after cancer. The LLM, GatorTron-3.9B, achieved the best F1 scores, outperforming the traditional support vector machines by 39%, the T-LSTM deep learning model by 7%, and a widely used transformer model, BERT, by 5.6%. The analysi
    
[^3]: ERBench：基于实体关系的可自动验证的大规模语言模型幻觉基准

    ERBench: An Entity-Relationship based Automatically Verifiable Hallucination Benchmark for Large Language Models

    [https://arxiv.org/abs/2403.05266](https://arxiv.org/abs/2403.05266)

    ERBench是一个基于实体关系的大型语言模型幻觉基准，通过自动转换任何关系数据库并构建可自动验证的问题，以支持复杂性评估和调试

    

    大型语言模型（LLMs）在各种应用中取得了前所未有的性能，然而它们的评估仍然是一个关键问题。现有的幻觉基准要么是静态的，要么缺乏可调整的复杂性进行彻底分析。我们认为利用现有的关系数据库是构建基准的一种有希望的方法，因为它们通过功能依赖关系可以准确描述知识。我们提出了ERBench，可以自动将任何关系数据库转换为基于实体关系（ER）模型的基准。我们的关键想法是使用数据库模式、记录和功能依赖来构建问题，以便可以自动验证。此外，我们使用外键约束来连接关系和构建多跳问题，这些问题可以任意复杂，用于调试LLMs的中间答案。最后，ERBench支持持续评估，多模态qu

    arXiv:2403.05266v1 Announce Type: cross  Abstract: Large language models (LLMs) have achieved unprecedented performance in various applications, yet their evaluation remains a critical issue. Existing hallucination benchmarks are either static or lack adjustable complexity for thorough analysis. We contend that utilizing existing relational databases is a promising approach for constructing benchmarks due to their accurate knowledge description via functional dependencies. We propose ERBench to automatically convert any relational database into a benchmark based on the entity-relationship (ER) model. Our key idea is to construct questions using the database schema, records, and functional dependencies such that they can be automatically verified. In addition, we use foreign key constraints to join relations and construct multihop questions, which can be arbitrarily complex and used to debug the intermediate answers of LLMs. Finally, ERBench supports continuous evaluation, multimodal qu
    
[^4]: 建筑如何影响预训练语言模型的基本能力？基于FFN-Wider变压器模型的案例研究

    How does Architecture Influence the Base Capabilities of Pre-trained Language Models? A Case Study Based on FFN-Wider Transformer Models

    [https://arxiv.org/abs/2403.02436](https://arxiv.org/abs/2403.02436)

    本研究探讨了建筑如何影响预训练语言模型的基本能力，发现了FFN-Wider变压器模型降低了多头注意力的贡献比，从而导致基本能力的下降。

    

    预训练语言模型已被证明具有强大的基本能力，不仅在分布式语言建模方面表现出色，而且在超出分布式语言建模、迁移学习和少样本学习方面也展现出强大的能力。与现有研究侧重于规模对基本能力的影响不同，我们的工作将重点放在了架构对其影响。具体地，我们关心的是：建筑如何影响预训练语言模型的基本能力？在这项工作中，我们试图解释并逆转FFN-Wider变压器的架构导致基本能力下降的情况，力求提供一些见解。通过分析，我们发现多头注意力（一种组合函数）对预训练语言建模的贡献比是影响基本能力的关键因素。FFN-Wider变压器减少了这种组合函数的贡献比，导致一种

    arXiv:2403.02436v1 Announce Type: new  Abstract: Pre-trained language models have been proven to possess strong base capabilities, which not only excel in in-distribution language modeling but also show powerful abilities in out-of-distribution language modeling, transfer learning and few-shot learning. Unlike existing work focusing on the influence of scale on base capabilities, our work examines the influence of architecture on those. Specifically, our concern is: How does architecture influence the base capabilities of pre-trained language models? In this work, we attempt to explain and reverse the decline in base capabilities caused by the architecture of FFN-Wider Transformers, seeking to provide some insights. Through analysis, we found the contribution ratio of Multi-Head Attention (a combination function) to pre-trained language modeling is a key factor affecting base capabilities. FFN-Wider Transformers reduce the contribution ratio of this combination function, leading to a d
    
[^5]: 隐性偏见的下一个标记预测

    Implicit Bias of Next-Token Prediction

    [https://arxiv.org/abs/2402.18551](https://arxiv.org/abs/2402.18551)

    在基于梯度的优化器训练下的线性NTP模型中，确定了NTP可分离条件，并证明梯度下降能够实现其下界；同时证明了这些条件在过参数化时仍然成立。

    

    下一个标记预测（NTP）是训练大型语言模型的首选范式，它涉及预测序列中的下一个标记。与传统的独热分类不同，在NTP中，多个具有不同频率的标记在给定上下文后继。本文将NTP训练框架化为跨不同上下文的交叉熵最小化，每个上下文都与有限词汇表中的稀疏经验概率向量相关联。然后，它探讨了以下问题：当NTP训练损失达到其下界（熵）时，基于梯度的优化器是否会对具有特定结构的解决方案存在偏见？具体地，对于使用梯度下降（GD）训练的线性NTP模型，我们做出以下贡献：首先，我们确定了数据上的NTP可分离条件，在这些条件下，GD能够达到其下界。我们还证明了这些条件在过参数化时仍成立。

    arXiv:2402.18551v1 Announce Type: cross  Abstract: Next-token prediction (NTP), the go-to training paradigm in training large language models, involves predicting the next token in a sequence. Departing from traditional one-hot classification, in NTP, multiple tokens with varying frequencies follow each given context. This work frames NTP training as cross-entropy minimization over distinct contexts, each associated with a sparse empirical probability vector across a finite vocabulary. It then addresses the following question: do gradient-based optimizers exhibit a bias towards solutions with specific structure as the NTP training loss reaches its lower bound (entropy)? Specifically, for linear NTP models trained using gradient descent (GD), we make the following contributions: Firstly, we determine NTP-separability conditions on the data, under which GD can attain its lower bound. We also demonstrate that these conditions hold under overparameterization. Secondly, we establish that th
    
[^6]: 大型语言模型对图形召回的微结构和准确性

    Microstructures and Accuracy of Graph Recall by Large Language Models

    [https://arxiv.org/abs/2402.11821](https://arxiv.org/abs/2402.11821)

    本研究首次系统研究了大型语言模型对图形召回的准确性和偏见微结构，探讨了它们与人类的异同以及对其他图形推理任务的影响。

    

    图形数据对许多应用至关重要，其中很多数据以文本格式描述关系。因此，准确地召回和编码先前文本中描述的图形是大型语言模型(LLMs)需要展示的基本但关键能力，以执行涉及图形结构信息的推理任务。人类在图形召回方面的表现已被认知科学家研究了几十年，发现其经常呈现与人类处理社会关系一致的某些结构性偏见模式。然而，迄今为止，我们很少了解LLMs在类似图形召回任务中的行为：它们召回的图形是否也呈现某些偏见模式，如果是，它们与人类的表现有何不同并如何影响其他图形推理任务？在这项研究中，我们进行了第一次对LLMs进行图形召回的系统研究，研究其准确性和偏见微结构（局部结构）。

    arXiv:2402.11821v1 Announce Type: cross  Abstract: Graphs data is crucial for many applications, and much of it exists in the relations described in textual format. As a result, being able to accurately recall and encode a graph described in earlier text is a basic yet pivotal ability that LLMs need to demonstrate if they are to perform reasoning tasks that involve graph-structured information. Human performance at graph recall by has been studied by cognitive scientists for decades, and has been found to often exhibit certain structural patterns of bias that align with human handling of social relationships. To date, however, we know little about how LLMs behave in analogous graph recall tasks: do their recalled graphs also exhibit certain biased patterns, and if so, how do they compare with humans and affect other graph reasoning tasks? In this work, we perform the first systematical study of graph recall by LLMs, investigating the accuracy and biased microstructures (local structura
    
[^7]: 进取的鲍勃通过提示对抗调整抵制越狱行为

    Studious Bob Fight Back Against Jailbreaking via Prompt Adversarial Tuning

    [https://arxiv.org/abs/2402.06255](https://arxiv.org/abs/2402.06255)

    本文提出了一种名为Prompt Adversarial Tuning (PAT)的方法，通过训练一个防御控制机制并将其作为前缀嵌入到用户提示中，实现对大型语言模型（LLMs）的越狱行为的防御。实验证明该方法在保护LLMs免受产生有害信息的影响方面效果显著。

    

    尽管大型语言模型（LLM）在各种应用中取得了巨大的成功，但它们也容易受到特定提示的影响，从而绕过内置的安全措施并提供危险或非法内容，这种现象被称为越狱行为。为了保护LLMs免受产生有害信息的影响，提出了各种防御策略，其中大多数集中在内容过滤或模型的对抗训练方面。在本文中，我们提出了一种名为Prompt Adversarial Tuning（PAT）的方法，通过训练一个防御控制机制并将其作为前缀嵌入到用户提示中来实现我们的防御策略。我们设计了一个类似对抗训练的训练过程，以实现我们的优化目标，交替更新攻击和防御控制机制。据我们所知，我们是第一个从提示调整的角度实施防御的人。一旦应用，我们的方法几乎不会影响LLMs的操作效率。实验表明我们的方法在抵御越狱行为方面具有良好的效果。

    Although Large Language Models (LLMs) have achieved tremendous success in various applications, they are also susceptible to certain prompts that can induce them to bypass built-in safety measures and provide dangerous or illegal content, a phenomenon known as jailbreak. To protect LLMs from producing harmful information, various defense strategies are proposed, with most focusing on content filtering or adversarial training of models. In this paper, we propose an approach named Prompt Adversarial Tuning (PAT) to train a defense control mechanism, which is then embedded as a prefix to user prompts to implement our defense strategy. We design a training process similar to adversarial training to achieve our optimized goal, alternating between updating attack and defense controls. To our knowledge, we are the first to implement defense from the perspective of prompt tuning. Once employed, our method will hardly impact the operational efficiency of LLMs. Experiments show that our method i
    
[^8]: 大语言模型代理能够模拟人类的信任行为吗？

    Can Large Language Model Agents Simulate Human Trust Behaviors?

    [https://arxiv.org/abs/2402.04559](https://arxiv.org/abs/2402.04559)

    大语言模型代理能够模拟人类的信任行为，表现出在信任游戏中的信任行为，并且与人类行为具有高度一致性，但存在一些偏见和对代理与人类的差异。

    

    大语言模型（LLM）代理已经越来越多地被采用作为模拟工具，用于模拟人类在社会科学等领域中的行为。然而，一个基本的问题仍然存在：LLM代理是否真的能够模拟人类行为？在本文中，我们专注于人类互动中最关键的行为之一，信任，旨在调查LLM代理是否能够模拟人类的信任行为。我们首先发现，在被行为经济学广泛接受的信任游戏框架下，LLM代理通常表现出信任行为，称为代理信任。然后，我们发现LLM代理在信任行为方面与人类具有较高的行为一致性，表明使用LLM代理模拟人类的信任行为是可行的。此外，我们还探索了代理信任中的偏见以及代理信任在对代理和人类之间的差异方面的内在特性。我们还探讨了包括高级推理策略在内的条件下代理信任的内在特性。

    Large Language Model (LLM) agents have been increasingly adopted as simulation tools to model humans in applications such as social science. However, one fundamental question remains: can LLM agents really simulate human behaviors? In this paper, we focus on one of the most critical behaviors in human interactions, trust, and aim to investigate whether or not LLM agents can simulate human trust behaviors. We first find that LLM agents generally exhibit trust behaviors, referred to as agent trust, under the framework of Trust Games, which are widely recognized in behavioral economics. Then, we discover that LLM agents can have high behavioral alignment with humans regarding trust behaviors, indicating the feasibility to simulate human trust behaviors with LLM agents. In addition, we probe into the biases in agent trust and the differences in agent trust towards agents and humans. We also explore the intrinsic properties of agent trust under conditions including advanced reasoning strate
    
[^9]: 2024年大规模语言模型的真实性

    Factuality of Large Language Models in the Year 2024

    [https://arxiv.org/abs/2402.02420](https://arxiv.org/abs/2402.02420)

    本文调查了大规模语言模型（LLM）的真实性问题，并对其现有研究进行了批判性分析，指出了改进LLM真实性的挑战和解决方案，以及自动真实性评估的障碍。未来的研究应该关注在这些方面的进一步工作。

    

    大规模语言模型（LLMs），尤其是在聊天方面进行指导调整后，已经成为我们日常生活的一部分，通过在一个地方直接回答各种问题，使人们从搜索、提取和整合多个信息源的过程中得到解脱。然而，很多情况下，LLM的回答是错误的，这限制了它们在现实场景中的适用性。因此，对于评估和提高LLM真实性的研究近年来引起了很多关注。在这项调查中，我们对现有的研究进行了批判性分析，旨在找出主要挑战及其原因，并指出改进LLM真实性的潜在解决方案，以及分析开放文本生成的自动真实性评估面临的障碍。我们还展望了未来研究的方向。

    Large language models (LLMs), especially when instruction-tuned for chat, have become part of our daily lives, freeing people from the process of searching, extracting, and integrating information from multiple sources by offering a straightforward answer to a variety of questions in a single place. Unfortunately, in many cases, LLM responses are factually incorrect, which limits their applicability in real-world scenarios. As a result, research on evaluating and improving the factuality of LLMs has attracted a lot of research attention recently. In this survey, we critically analyze existing work with the aim to identify the major challenges and their associated causes, pointing out to potential solutions for improving the factuality of LLMs, and analyzing the obstacles to automated factuality evaluation for open-ended text generation. We further offer an outlook on where future research should go.
    
[^10]: AutoTimes: 基于大型语言模型的自回归时间序列预测器

    AutoTimes: Autoregressive Time Series Forecasters via Large Language Models

    [https://arxiv.org/abs/2402.02370](https://arxiv.org/abs/2402.02370)

    AutoTimes是一种基于大型语言模型的自回归时间序列预测器，利用语言模型的转换能力来处理时间序列数据，实现了与先前模型相当的性能。

    

    由于大规模时间序列的有限可用性和可扩展预训练的不充分探索，时间序列的基础模型尚未完全发展。基于时间序列和自然语言的相似顺序结构，越来越多的研究证明了利用大型语言模型(LLM)进行时间序列的可行性。然而，先前的方法可能忽视了时间序列和自然语言对齐的一致性，导致对LLM潜力的利用不足。为了充分利用从语言建模中学到的通用令牌转换，我们提出了AutoTimes，将LLM重新用作自回归时间序列预测器，这与LLM的获取和利用一致，而无需更新参数。由此产生的预测器可以处理灵活的系列长度，并实现与流行模型相当的性能。此外，我们提出了基于令牌的提示方法，利用相应的时间戳来进行预测。

    Foundation models of time series have not been fully developed due to the limited availability of large-scale time series and the underexploration of scalable pre-training. Based on the similar sequential structure of time series and natural language, increasing research demonstrates the feasibility of leveraging large language models (LLM) for time series. Nevertheless, prior methods may overlook the consistency in aligning time series and natural language, resulting in insufficient utilization of the LLM potentials. To fully exploit the general-purpose token transitions learned from language modeling, we propose AutoTimes to repurpose LLMs as Autoregressive Time series forecasters, which is consistent with the acquisition and utilization of LLMs without updating the parameters. The consequent forecasters can handle flexible series lengths and achieve competitive performance as prevalent models. Further, we present token-wise prompting that utilizes corresponding timestamps to make ou
    
[^11]: 在多模态大型语言模型中为图推理渲染图形

    Rendering Graphs for Graph Reasoning in Multimodal Large Language Models

    [https://arxiv.org/abs/2402.02130](https://arxiv.org/abs/2402.02130)

    本文在多模态大型语言模型中为图推理任务引入了视觉信息，并提出了一个新的基准测试数据集GITQA。实验证明，结合文本和视觉信息的结果比单一模态效果更好。

    

    大型语言模型(LLMs)在机器人规划、知识图谱补全和常识推理等任务中越来越多地使用图结构，LLMs能够理解文本格式的图信息，但忽视了丰富的视觉模态，而视觉是人类理解结构信息和进行图推理的直观方式。将图结构表示为视觉图像(即视觉图)的潜在益处和能力仍未被探索。本文在图推理任务中首次引入视觉信息，并提出一个新的基准测试数据集GITQA，其中每个样本是一个元组(图、图像、文本描述)。我们利用最先进的多模态LLMs在GITQA基准测试数据集上进行了大量实验证明，结合文本和视觉信息的结果比单一模态效果更好。此外，在LLaVA-7B/13B模型的微调上表现出色。

    Large Language Models (LLMs) are increasingly used for various tasks with graph structures, such as robotic planning, knowledge graph completion, and common-sense reasoning. Though LLMs can comprehend graph information in a textual format, they overlook the rich visual modality, which is an intuitive way for humans to comprehend structural information and conduct graph reasoning. The potential benefits and capabilities of representing graph structures as visual images (i.e., visual graph) is still unexplored. In this paper, we take the first step in incorporating visual information into graph reasoning tasks and propose a new benchmark GITQA, where each sample is a tuple (graph, image, textual description). We conduct extensive experiments on the GITQA benchmark using state-of-the-art multimodal LLMs. Results on graph reasoning tasks show that combining textual and visual information together performs better than using one modality alone. Moreover, the LLaVA-7B/13B models finetuned on 
    
[^12]: 大型语言模型的人类可读指纹

    Human-Readable Fingerprint for Large Language Models

    [https://arxiv.org/abs/2312.04828](https://arxiv.org/abs/2312.04828)

    这项研究介绍了一种大型语言模型的人类可读指纹，可以唯一识别出基本模型，并且不暴露模型参数或干扰训练。通过观察和验证，研究发现模型参数的向量方向在预训练后保持稳定，成为识别基本模型的重要条件。

    

    由于大型语言模型（LLM）的资源密集型训练和配套的精心设计的许可证，保护LLM的版权变得至关重要。然而，由于可能的参数修改，确定LLM的原始基本模型是具有挑战性的。在本研究中，我们介绍了一种用于LLM的人类可读指纹，可以唯一地识别基本模型，而不暴露模型参数或干扰训练。我们首先观察到，在预训练期间模型收敛后，LLM参数的向量方向保持稳定，通过后续的训练步骤，包括持续预训练、监督微调和RLHF，几乎没有扰动，这使得它成为识别基本模型的足够条件。通过继续训练LLM并添加一个额外的项来推开模型参数的方向，验证了这种必要性，结果使得模型受损。然而，这个方向容易受到简单攻击的影响，如维度...

    Protecting the copyright of large language models (LLMs) has become crucial due to their resource-intensive training and accompanying carefully designed licenses. However, identifying the original base model of an LLM is challenging due to potential parameter alterations. In this study, we introduce a human-readable fingerprint for LLMs that uniquely identifies the base model without exposing model parameters or interfering with training. We first observe that the vector direction of LLM parameters remains stable after the model has converged during pretraining, showing negligible perturbations through subsequent training steps, including continued pretraining, supervised fine-tuning (SFT), and RLHF, which makes it a sufficient condition to identify the base model. The necessity is validated by continuing to train an LLM with an extra term to drive away the model parameters' direction and the model becomes damaged. However, this direction is vulnerable to simple attacks like dimension 
    
[^13]: 攻击树：自动破解黑盒大型语言模型

    Tree of Attacks: Jailbreaking Black-Box LLMs Automatically

    [https://arxiv.org/abs/2312.02119](https://arxiv.org/abs/2312.02119)

    提出了一种名为Tree of Attacks with Pruning (TAP)的自动化方法，用于生成只需要对目标大型语言模型进行黑盒访问的越狱方法，并通过思维树推理和修剪生成准确的越狱提示。

    

    大型语言模型(LLMs)展示了多功能性，但仍在生成有害、带偏见和有毒内容，这一点由人为设计的越狱行为的普遍存在得以证明。在这项工作中，我们提出了一种名为Tree of Attacks with Pruning (TAP)的自动化方法，用于生成越狱，仅需要对目标LLM进行黑盒访问。TAP利用LLM来通过思维树推理迭代地优化候选（攻击）提示，直到生成的提示之一越狱目标。关键在于，在将提示发送给目标之前，TAP对其进行评估并移除可能不会导致越狱的提示。使用思维树推理使TAP能够在大量提示的搜索空间中导航，而修剪则减少了发送给目标的总查询数量。在实证评估中，我们观察到TAP生成的提示越狱了超过80%的最先进LLMs（包括GPT4和GPT4-Turbo）。

    arXiv:2312.02119v2 Announce Type: replace-cross  Abstract: While Large Language Models (LLMs) display versatile functionality, they continue to generate harmful, biased, and toxic content, as demonstrated by the prevalence of human-designed jailbreaks. In this work, we present Tree of Attacks with Pruning (TAP), an automated method for generating jailbreaks that only requires black-box access to the target LLM. TAP utilizes an LLM to iteratively refine candidate (attack) prompts using tree-of-thought reasoning until one of the generated prompts jailbreaks the target. Crucially, before sending prompts to the target, TAP assesses them and prunes the ones unlikely to result in jailbreaks. Using tree-of-thought reasoning allows TAP to navigate a large search space of prompts and pruning reduces the total number of queries sent to the target. In empirical evaluations, we observe that TAP generates prompts that jailbreak state-of-the-art LLMs (including GPT4 and GPT4-Turbo) for more than 80%
    
[^14]: 通过不确定性量化对LLMs进行基准测试

    Benchmarking LLMs via Uncertainty Quantification. (arXiv:2401.12794v1 [cs.CL])

    [http://arxiv.org/abs/2401.12794](http://arxiv.org/abs/2401.12794)

    这项研究提出了一种通过不确定性量化来对LLMs进行基准测试的方法，并引入了一种不确定性感知评估指标UAcc。研究结果显示，高准确度的LLMs可能具有较低的确定性，而大规模LLMs可能比较小规模LLMs更不确定。指令微调倾向于增加LLMs的不确定性。

    

    随着各个机构开源大型语言模型（LLMs）的增多，彰显了对综合评估方法的迫切需求。然而，当前的评估平台，如广为人知的HuggingFace开放的LLM排行榜，忽视了一个关键方面--不确定性，而不确定性对于全面评估LLMs至关重要。为了弥补这一差距，我们引入了一种新的LLMs基准测试方法，将不确定性量化集成进去。我们的研究涵盖了五个典型的自然语言处理任务中的八个LLMs（LLM系列）。此外，我们引入了一种考虑了预测准确性和预测不确定性的不确定性感知评估指标UAcc。我们的研究结果显示：I）准确度越高的LLMs可能会表现出较低的确定性；II）规模较大的LLMs可能与较小的LLMs相比显示出更大的不确定性；III）指令微调倾向于增加LLMs的不确定性。通过考虑不确定性，我们的方法可以更全面地评估LLMs的性能。

    The proliferation of open-source Large Language Models (LLMs) from various institutions has highlighted the urgent need for comprehensive evaluation methods. However, current evaluation platforms, such as the widely recognized HuggingFace open LLM leaderboard, neglect a crucial aspect -- uncertainty, which is vital for thoroughly assessing LLMs. To bridge this gap, we introduce a new benchmarking approach for LLMs that integrates uncertainty quantification. Our examination involves eight LLMs (LLM series) spanning five representative natural language processing tasks. Additionally, we introduce an uncertainty-aware evaluation metric, UAcc, which takes into account both prediction accuracy and prediction uncertainty. Our findings reveal that: I) LLMs with higher accuracy may exhibit lower certainty; II) Larger-scale LLMs may display greater uncertainty compared to their smaller counterparts; and III) Instruction-finetuning tends to increase the uncertainty of LLMs. By taking uncertainty
    
[^15]: 基于剪枝的保护: 在不进行微调的情况下增加对齐的LLMs的越狱抵抗力

    Pruning for Protection: Increasing Jailbreak Resistance in Aligned LLMs Without Fine-Tuning. (arXiv:2401.10862v1 [cs.LG])

    [http://arxiv.org/abs/2401.10862](http://arxiv.org/abs/2401.10862)

    本文研究了剪枝对齐的LLMs的保护措施，发现剪枝LLM参数可以显著增强其抵抗“越狱”提示攻击的能力，并且对其他LLM行为也可能有更普遍的效果。同时，引入了一个有害任务数据集，证明剪枝有助于集中注意力在与任务相关的标记上。突出的聊天模型表现出很高的易感性。

    

    大型语言模型（LLMs）容易受到“越狱”提示的攻击，这种攻击可以诱使这些模型生成有害和违法内容。本文表明，剪枝LLM参数多达20％可以显著增加它们对此类攻击的抵抗力，而无需额外训练并且不损害其在标准基准测试中的性能。有趣的是，我们发现剪枝后观察到的增强安全性与模型的初始安全训练水平相关，这暗示剪枝的效果可能更普遍，也可能适用于超出安全性范畴的其他LLM行为。另外，我们还介绍了一个包含五个类别、插入到十个不同越狱提示中的225个有害任务的精选数据集，表明剪枝有助于LLMs集中注意力在越狱提示中与任务相关的标记上。最后，我们的实验揭示了突出的聊天模型（如LLaMA-2 Chat，Vicuna和Mistral Instruct）具有很高的易感性。

    Large Language Models (LLMs) are vulnerable to `Jailbreaking' prompts, a type of attack that can coax these models into generating harmful and illegal content. In this paper, we show that pruning up to 20% of LLM parameters markedly increases their resistance to such attacks without additional training and without sacrificing their performance in standard benchmarks. Intriguingly, we discovered that the enhanced safety observed post-pruning correlates to the initial safety training level of the model, hinting that the effect of pruning could be more general and may hold for other LLM behaviors beyond safety. Additionally, we introduce a curated dataset of 225 harmful tasks across five categories, inserted into ten different Jailbreaking prompts, showing that pruning aids LLMs in concentrating attention on task-relevant tokens in jailbreaking prompts. Lastly, our experiments reveal that the prominent chat models, such as LLaMA-2 Chat, Vicuna, and Mistral Instruct exhibit high susceptibi
    
[^16]: 为什么应该删除这篇文章？多语言维基百科编辑讨论中的透明立场识别。

    Why Should This Article Be Deleted? Transparent Stance Detection in Multilingual Wikipedia Editor Discussions. (arXiv:2310.05779v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2310.05779](http://arxiv.org/abs/2310.05779)

    该研究构建了一个多语言数据集，包含维基百科编辑讨论和决策的推理过程，在解释内容管理决策方面增加了透明度。

    

    在线平台上的内容管理通常是非透明的。然而，在维基百科上，这种讨论是公开进行的，编辑被鼓励使用内容管理政策来解释他们的决策。目前，只有少数评论明确提到这些政策-英文评论20%，但德文和土耳其评论只有2%。为了帮助理解内容管理过程，我们构建了一个新颖的多语言数据集，其中包含维基百科编辑讨论以及他们的推理过程。数据集包含编辑的立场（保留、删除、合并、评论），以及每个编辑决策所陈述的原因和内容管理政策。我们证明了立场和相应的原因（政策）可以被高度准确地联合预测，从而增加了决策过程的透明度。我们发布了联合预测模型和多语言内容。

    The moderation of content on online platforms is usually non-transparent. On Wikipedia, however, this discussion is carried out publicly and the editors are encouraged to use the content moderation policies as explanations for making moderation decisions. Currently, only a few comments explicitly mention those policies -- 20% of the English ones, but as few as 2% of the German and Turkish comments. To aid in this process of understanding how content is moderated, we construct a novel multilingual dataset of Wikipedia editor discussions along with their reasoning in three languages. The dataset contains the stances of the editors (keep, delete, merge, comment), along with the stated reason, and a content moderation policy, for each edit decision. We demonstrate that stance and corresponding reason (policy) can be predicted jointly with a high degree of accuracy, adding transparency to the decision-making process. We release both our joint prediction models and the multilingual content m
    

