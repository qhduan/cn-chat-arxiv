# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [The Fine Line: Navigating Large Language Model Pretraining with Down-streaming Capability Analysis](https://arxiv.org/abs/2404.01204) | 本文对大型语言模型在预训练过程中不同能力的综合分析揭示了其动态差异，填补了现有标度定律中的缺失。 |
| [^2] | [Speech Robust Bench: A Robustness Benchmark For Speech Recognition](https://arxiv.org/abs/2403.07937) | 提出了一个全面基准（SRB），用于评估自动语音识别（ASR）模型对各种破坏的鲁棒性，发现模型大小和某些建模选择有助于提高鲁棒性，并观察到在不同人口亚组上模型的鲁棒性存在明显差异。 |
| [^3] | [Towards Trustworthy Reranking: A Simple yet Effective Abstention Mechanism](https://arxiv.org/abs/2402.12997) | 提出了一种适用于现实约束的轻量级弃权机制，特别适用于再排序阶段，通过数据驱动的方法达到有效性，并提供了开源代码以促进其更广泛的应用。 |
| [^4] | [Are LLMs Ready for Real-World Materials Discovery?](https://arxiv.org/abs/2402.05200) | LLMs在材料科学中的应用受限，无法实现实际应用。我们提出了基于材料科学知识和假设测试的MatSci-LLMs框架，并描述了关键的材料科学信息提取挑战。 |
| [^5] | [How do Large Language Models Learn In-Context? Query and Key Matrices of In-Context Heads are Two Towers for Metric Learning](https://arxiv.org/abs/2402.02872) | 本论文探索了大型语言模型如何进行上下文学习的机制，提出了一个使用定位和投影方法的假设。通过查询和键矩阵来计算输入文本与每个演示之间的注意力权重，以学习它们之间的相似度度量。实验证明了我们的分析。 |
| [^6] | [Locating Factual Knowledge in Large Language Models: Exploring the Residual Stream and Analyzing Subvalues in Vocabulary Space.](http://arxiv.org/abs/2312.12141) | 通过探索剩余流和分析词汇空间中的子值，我们定位了大型语言模型中的事实知识，并找到了存储了有关“法国，首都，巴黎”的知识的位置。 |
| [^7] | [Conversational Health Agents: A Personalized LLM-Powered Agent Framework.](http://arxiv.org/abs/2310.02374) | 该论文介绍了一个基于LLM的会话式健康代理框架，旨在为代理赋予批判性思维、知识获取和问题解决能力，实现个性化的健康护理服务。该框架能够无缝集成医疗工具，实现多语言和多模态对话，并与多种用户数据分析工具连接。 |
| [^8] | [Benchmarking Cognitive Biases in Large Language Models as Evaluators.](http://arxiv.org/abs/2309.17012) | 本研究对15个不同大小的大型语言模型进行了评估，发现它们作为评估器存在认知偏差，尤其在文本质量评估中表现出较强的偏见，这对其鲁棒性提出了质疑。同时，研究还发现了人类和机器偏好之间的相关性。 |
| [^9] | [The FruitShell French synthesis system at the Blizzard 2023 Challenge.](http://arxiv.org/abs/2309.00223) | 本文介绍了一个用于Blizzard Challenge 2023的法语文本到语音合成系统，通过对数据的筛选和增强，以及添加词边界和起始/结束符号的方式，提高了语音质量并进行了标准化转录。 |

# 详细

[^1]: 大型语言模型预训练与下游能力分析的微妙平衡

    The Fine Line: Navigating Large Language Model Pretraining with Down-streaming Capability Analysis

    [https://arxiv.org/abs/2404.01204](https://arxiv.org/abs/2404.01204)

    本文对大型语言模型在预训练过程中不同能力的综合分析揭示了其动态差异，填补了现有标度定律中的缺失。

    

    揭示反映最终模型性能的早期指标是大规模预训练的一个核心原则。现有的标度定律展示了预训练损失与训练浮点数之间的幂律相关性，这对于大型语言模型当前训练状态的重要指标十分关键。然而，这一原则只关注模型在训练数据上的压缩特性，导致与下游任务能力的提升之间存在不一致性。一些后续研究试图将标度定律扩展到更复杂的指标（如超参数），但仍然缺乏对预训练过程中各种能力之间动态差异的全面分析。为解决上述限制，本文对各种预训练中间检查点下模型能力进行了全面比较。

    arXiv:2404.01204v1 Announce Type: new  Abstract: Uncovering early-stage metrics that reflect final model performance is one core principle for large-scale pretraining. The existing scaling law demonstrates the power-law correlation between pretraining loss and training flops, which serves as an important indicator of the current training state for large language models. However, this principle only focuses on the model's compression properties on the training data, resulting in an inconsistency with the ability improvements on the downstream tasks. Some follow-up works attempted to extend the scaling-law to more complex metrics (such as hyperparameters), but still lacked a comprehensive analysis of the dynamic differences among various capabilities during pretraining. To address the aforementioned limitations, this paper undertakes a comprehensive comparison of model capabilities at various pretraining intermediate checkpoints. Through this analysis, we confirm that specific downstream
    
[^2]: 语音鲁棒基准：用于语音识别的鲁棒性基准

    Speech Robust Bench: A Robustness Benchmark For Speech Recognition

    [https://arxiv.org/abs/2403.07937](https://arxiv.org/abs/2403.07937)

    提出了一个全面基准（SRB），用于评估自动语音识别（ASR）模型对各种破坏的鲁棒性，发现模型大小和某些建模选择有助于提高鲁棒性，并观察到在不同人口亚组上模型的鲁棒性存在明显差异。

    

    随着自动语音识别（ASR）模型变得越来越普遍，确保它们在物理世界和数字世界中的各种破坏下进行可靠预测变得愈发重要。我们提出了语音鲁棒基准（SRB），这是一个用于评估ASR模型对各种破坏的鲁棒性的全面基准。SRB由69个输入扰动组成，旨在模拟ASR模型可能在物理世界和数字世界中遇到的各种破坏。我们使用SRB来评估几种最先进的ASR模型的鲁棒性，并观察到模型大小和某些建模选择（如离散表示和自我训练）似乎有助于提高鲁棒性。我们将此分析扩展到衡量ASR模型在来自各种人口亚组的数据上的鲁棒性，即英语和西班牙语使用者以及男性和女性，并观察到模型的鲁棒性在不同亚组之间存在明显差异。

    arXiv:2403.07937v1 Announce Type: cross  Abstract: As Automatic Speech Recognition (ASR) models become ever more pervasive, it is important to ensure that they make reliable predictions under corruptions present in the physical and digital world. We propose Speech Robust Bench (SRB), a comprehensive benchmark for evaluating the robustness of ASR models to diverse corruptions. SRB is composed of 69 input perturbations which are intended to simulate various corruptions that ASR models may encounter in the physical and digital world. We use SRB to evaluate the robustness of several state-of-the-art ASR models and observe that model size and certain modeling choices such as discrete representations, and self-training appear to be conducive to robustness. We extend this analysis to measure the robustness of ASR models on data from various demographic subgroups, namely English and Spanish speakers, and males and females, and observed noticeable disparities in the model's robustness across su
    
[^3]: 朝着可信的再排序：一种简单但有效的弃权机制

    Towards Trustworthy Reranking: A Simple yet Effective Abstention Mechanism

    [https://arxiv.org/abs/2402.12997](https://arxiv.org/abs/2402.12997)

    提出了一种适用于现实约束的轻量级弃权机制，特别适用于再排序阶段，通过数据驱动的方法达到有效性，并提供了开源代码以促进其更广泛的应用。

    

    神经信息检索（NIR）已经显著改进了基于启发式的IR系统。然而，失败仍然频繁发生，通常所使用的模型无法检索与用户查询相关的文档。我们通过提出一种适用于现实约束的轻量级弃权机制来解决这一挑战，特别强调再排序阶段。我们介绍了一个协议，用于在黑匣子场景中评估弃权策略的效果，并提出了一种简单但有效的数据驱动机制。我们提供了实验复制和弃权实施的开源代码，促进其在不同环境中更广泛的采用和应用。

    arXiv:2402.12997v1 Announce Type: cross  Abstract: Neural Information Retrieval (NIR) has significantly improved upon heuristic-based IR systems. Yet, failures remain frequent, the models used often being unable to retrieve documents relevant to the user's query. We address this challenge by proposing a lightweight abstention mechanism tailored for real-world constraints, with particular emphasis placed on the reranking phase. We introduce a protocol for evaluating abstention strategies in a black-box scenario, demonstrating their efficacy, and propose a simple yet effective data-driven mechanism. We provide open-source code for experiment replication and abstention implementation, fostering wider adoption and application in diverse contexts.
    
[^4]: LLMs是否准备好应用于真实世界的材料发现？

    Are LLMs Ready for Real-World Materials Discovery?

    [https://arxiv.org/abs/2402.05200](https://arxiv.org/abs/2402.05200)

    LLMs在材料科学中的应用受限，无法实现实际应用。我们提出了基于材料科学知识和假设测试的MatSci-LLMs框架，并描述了关键的材料科学信息提取挑战。

    

    大型语言模型（LLMs）为材料科学中的强大语言处理工具提供了令人兴奋的可能性，加快了材料研究的进展。然而，LLMs在实际应用中仍存在不足，无法成为实用的材料科学工具。本文通过展示LLMs在材料科学中的相关失败案例，揭示了LLMs在理解和推理复杂、相互关联的材料科学知识方面的现有限制。鉴于这些缺点，我们提出了一种开发基于材料科学知识和假设生成与测试的材料科学LLMs（MatSci-LLMs）的框架。实现高性能的MatSci-LLMs的路径在很大程度上取决于建立高质量的多模态数据集，这些数据集来自科学文献，其中存在各种信息提取挑战。因此，我们描述了关键的材料科学信息提取挑战。

    Large Language Models (LLMs) create exciting possibilities for powerful language processing tools to accelerate research in materials science. While LLMs have great potential to accelerate materials understanding and discovery, they currently fall short in being practical materials science tools. In this position paper, we show relevant failure cases of LLMs in materials science that reveal current limitations of LLMs related to comprehending and reasoning over complex, interconnected materials science knowledge. Given those shortcomings, we outline a framework for developing Materials Science LLMs (MatSci-LLMs) that are grounded in materials science knowledge and hypothesis generation followed by hypothesis testing. The path to attaining performant MatSci-LLMs rests in large part on building high-quality, multi-modal datasets sourced from scientific literature where various information extraction challenges persist. As such, we describe key materials science information extraction cha
    
[^5]: 大型语言模型如何进行上下文学习？查询和键矩阵是上下文头部进行度量学习的两个关键组成部分

    How do Large Language Models Learn In-Context? Query and Key Matrices of In-Context Heads are Two Towers for Metric Learning

    [https://arxiv.org/abs/2402.02872](https://arxiv.org/abs/2402.02872)

    本论文探索了大型语言模型如何进行上下文学习的机制，提出了一个使用定位和投影方法的假设。通过查询和键矩阵来计算输入文本与每个演示之间的注意力权重，以学习它们之间的相似度度量。实验证明了我们的分析。

    

    我们探索了上下文学习的机制，并提出了使用定位和投影方法的假设。在浅层中，演示的特征被合并到相应的标签中，输入文本的特征被聚合到最后一个标记中。在深层中，上下文头部发挥了重要作用。在每个上下文头部中，值-输出矩阵提取了标签的特征。查询和键矩阵计算了输入文本与每个演示之间的注意力权重。注意力权重越大，越多的标签信息被传输到最后一个标记中，用于预测下一个单词。查询和键矩阵可以被视为学习输入文本与每个演示之间相似度度量的两个关键组成部分。基于这个假设，我们解释了为什么不平衡的标签和演示顺序会影响预测。我们在GPT2大型、Llama 7B、13B和30B上进行了实验。结果支持我们的分析。总体而言，我们的研究提供了一个关于大型语言模型如何进行上下文学习的理论解释和验证。

    We explore the mechanism of in-context learning and propose a hypothesis using locate-and-project method. In shallow layers, the features of demonstrations are merged into their corresponding labels, and the features of the input text are aggregated into the last token. In deep layers, in-context heads make great contributions. In each in-context head, the value-output matrix extracts the labels' features. Query and key matrices compute the attention weights between the input text and each demonstration. The larger the attention weight is, the more label information is transferred into the last token for predicting the next word. Query and key matrices can be regarded as two towers for learning the similarity metric between the input text and each demonstration. Based on this hypothesis, we explain why imbalanced labels and demonstration order affect predictions. We conduct experiments on GPT2 large, Llama 7B, 13B and 30B. The results can support our analysis. Overall, our study provid
    
[^6]: 在大型语言模型中定位事实知识：探索剩余流和分析词汇空间中的子值。

    Locating Factual Knowledge in Large Language Models: Exploring the Residual Stream and Analyzing Subvalues in Vocabulary Space. (arXiv:2312.12141v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2312.12141](http://arxiv.org/abs/2312.12141)

    通过探索剩余流和分析词汇空间中的子值，我们定位了大型语言模型中的事实知识，并找到了存储了有关“法国，首都，巴黎”的知识的位置。

    

    通过探索剩余流和分析词汇空间中的子值，我们找到了大型语言模型中事实知识的位置。我们发现当投影到词汇空间时，子值具有可人类解释的概念的原因。子值的softmax之前的值通过一个加法函数相加，因此词汇空间中前几个标记的概率会增加。基于此，我们发现使用对数概率增加来计算层和子值的重要性比概率增加更好，因为对数概率增加的曲线呈线性单调增形状。此外，我们计算内积来评估前馈网络（FFN）的子值被前面的层激活的程度。根据我们的方法，我们找到了事实知识“法国，首都，巴黎”存储的位置。具体来说，注意力层存储“巴黎与法国相关”。FFN层存储“巴黎是一个首都/城市”，由注意力子值激活。

    We find the location of factual knowledge in large language models by exploring the residual stream and analyzing subvalues in vocabulary space. We find the reason why subvalues have human-interpretable concepts when projecting into vocabulary space. The before-softmax values of subvalues are added by an addition function, thus the probability of top tokens in vocabulary space will increase. Based on this, we find using log probability increase to compute the significance of layers and subvalues is better than probability increase, since the curve of log probability increase has a linear monotonically increasing shape. Moreover, we calculate the inner products to evaluate how much a feed-forward network (FFN) subvalue is activated by previous layers. Base on our methods, we find where factual knowledge <France, capital, Paris> is stored. Specifically, attention layers store "Paris is related to France". FFN layers store "Paris is a capital/city", activated by attention subvalues relate
    
[^7]: 会话式健康代理：个性化的基于LLM的代理框架

    Conversational Health Agents: A Personalized LLM-Powered Agent Framework. (arXiv:2310.02374v1 [cs.CL])

    [http://arxiv.org/abs/2310.02374](http://arxiv.org/abs/2310.02374)

    该论文介绍了一个基于LLM的会话式健康代理框架，旨在为代理赋予批判性思维、知识获取和问题解决能力，实现个性化的健康护理服务。该框架能够无缝集成医疗工具，实现多语言和多模态对话，并与多种用户数据分析工具连接。

    

    会话式健康代理（CHAs）是一种互动系统，旨在通过进行共情对话和处理多模态数据来增强个人健康护理服务。当前的CHAs，特别是那些利用大型语言模型（LLMs）的系统，主要关注对话，但往往缺乏全面的代理能力。这包括从可穿戴设备、全天候数据收集源和电子健康记录中获取个人用户的健康数据的能力，以及整合最新发布的健康见解，并与已建立的多模态数据分析工具连接。我们正在开发一个框架，通过赋予CHAs批判性思维、知识获取和问题解决能力来增强它们的能力。我们的CHA平台由LLMs驱动，无缝集成了医疗工具，实现了多语言和多模态对话，并与各种用户数据分析工具进行接口。我们展示了其在处理复杂医疗任务方面的熟练性。

    Conversational Health Agents (CHAs) are interactive systems designed to enhance personal healthcare services by engaging in empathetic conversations and processing multimodal data. While current CHAs, especially those utilizing Large Language Models (LLMs), primarily focus on conversation, they often lack comprehensive agent capabilities. This includes the ability to access personal user health data from wearables, 24/7 data collection sources, and electronic health records, as well as integrating the latest published health insights and connecting with established multimodal data analysis tools. We are developing a framework to empower CHAs by equipping them with critical thinking, knowledge acquisition, and problem-solving abilities. Our CHA platform, powered by LLMs, seamlessly integrates healthcare tools, enables multilingual and multimodal conversations, and interfaces with a variety of user data analysis tools. We illustrate its proficiency in handling complex healthcare tasks, s
    
[^8]: 作为评估器的大型语言模型中认知偏差的基准测试

    Benchmarking Cognitive Biases in Large Language Models as Evaluators. (arXiv:2309.17012v1 [cs.CL])

    [http://arxiv.org/abs/2309.17012](http://arxiv.org/abs/2309.17012)

    本研究对15个不同大小的大型语言模型进行了评估，发现它们作为评估器存在认知偏差，尤其在文本质量评估中表现出较强的偏见，这对其鲁棒性提出了质疑。同时，研究还发现了人类和机器偏好之间的相关性。

    

    最近的研究表明，大型语言模型（LLMs）通过简单的提示和上下文学习作为自动评估器非常有效。本研究组装了15个大小不同的LLMs，并通过其他LLMs的偏好排名来评估它们的输出响应，例如System Star比System Square更好。然后，我们引入了用于评估LLMs输出中六种不同认知偏差的认知偏差基准测试（CoBBLEr），如自我中心偏差，即模型更喜欢将自己的输出在评估中排名较高。我们发现LLMs是有偏见的文本质量评估器，在每个评估中都表现出对我们偏见基准的强烈迹象（在所有模型上的平均比较约为40%），这对它们作为评估器的鲁棒性提出了质疑。此外，我们还研究了人类和机器偏好之间的相关性，并计算了平均的Rank-Biased O值。

    Large Language Models (LLMs) have recently been shown to be effective as automatic evaluators with simple prompting and in-context learning. In this work, we assemble 15 LLMs of four different size ranges and evaluate their output responses by preference ranking from the other LLMs as evaluators, such as System Star is better than System Square. We then evaluate the quality of ranking outputs introducing the Cognitive Bias Benchmark for LLMs as Evaluators (CoBBLEr), a benchmark to measure six different cognitive biases in LLM evaluation outputs, such as the Egocentric bias where a model prefers to rank its own outputs highly in evaluation. We find that LLMs are biased text quality evaluators, exhibiting strong indications on our bias benchmark (average of 40% of comparisons across all models) within each of their evaluations that question their robustness as evaluators. Furthermore, we examine the correlation between human and machine preferences and calculate the average Rank-Biased O
    
[^9]: FruitShell法语合成系统在Blizzard 2023挑战赛中的应用

    The FruitShell French synthesis system at the Blizzard 2023 Challenge. (arXiv:2309.00223v1 [eess.AS])

    [http://arxiv.org/abs/2309.00223](http://arxiv.org/abs/2309.00223)

    本文介绍了一个用于Blizzard Challenge 2023的法语文本到语音合成系统，通过对数据的筛选和增强，以及添加词边界和起始/结束符号的方式，提高了语音质量并进行了标准化转录。

    

    本文介绍了一个用于Blizzard Challenge 2023的法语文本到语音合成系统。该挑战包括两个任务：从女性演讲者生成高质量的语音和生成与特定个体相似的语音。关于比赛数据，我们进行了筛选过程，去除了缺失或错误的文本数据。我们对除音素以外的所有符号进行了整理，并消除了没有发音或持续时间为零的符号。此外，我们还在文本中添加了词边界和起始/结束符号，根据我们之前的经验，我们发现这样可以提高语音质量。对于Spoke任务，我们根据比赛规则进行了数据增强。我们使用了一个开源的G2P模型将法语文本转录为音素。由于G2P模型使用国际音标（IPA），我们对提供的比赛数据应用了相同的转录过程，以进行标准化。然而，由于编译器对某些技术限制的识别能力有限，所以我们为了保持竞争的公正，将数据按音标划分为不同的片段进行评估。

    This paper presents a French text-to-speech synthesis system for the Blizzard Challenge 2023. The challenge consists of two tasks: generating high-quality speech from female speakers and generating speech that closely resembles specific individuals. Regarding the competition data, we conducted a screening process to remove missing or erroneous text data. We organized all symbols except for phonemes and eliminated symbols that had no pronunciation or zero duration. Additionally, we added word boundary and start/end symbols to the text, which we have found to improve speech quality based on our previous experience. For the Spoke task, we performed data augmentation according to the competition rules. We used an open-source G2P model to transcribe the French texts into phonemes. As the G2P model uses the International Phonetic Alphabet (IPA), we applied the same transcription process to the provided competition data for standardization. However, due to compiler limitations in recognizing 
    

