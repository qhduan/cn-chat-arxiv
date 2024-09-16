# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Activation Steering for Robust Type Prediction in CodeLLMs](https://arxiv.org/abs/2404.01903) | 我们提出了一种激活导向技术，通过编辑模型内部激活来改善CodeLLMs在代码类型预测中对于语法干扰的鲁棒性，并成功应用于Python和TypeScript的类型预测，将类型误差率纠正高达90%。 |
| [^2] | [IndoCulture: Exploring Geographically-Influenced Cultural Commonsense Reasoning Across Eleven Indonesian Provinces](https://arxiv.org/abs/2404.01854) | 本研究介绍了IndoCulture项目，旨在通过当地人手动收集数据，探索印尼十一个省份间地理影响的文化常识推理。评估发现，即使是最好的语言模型也在特定省份上表现更准确，而添加地理信息有助于提高模型性能。 |
| [^3] | [Wait, It's All Token Noise? Always Has Been: Interpreting LLM Behavior Using Shapley Value](https://arxiv.org/abs/2404.01332) | 使用Shapley值方法解释LLM行为，揭示了所谓的“令牌噪音”效应，揭示了LLMs的决策在很大程度上受到提示组件的影响 |
| [^4] | [Reasoning Abilities of Large Language Models: In-Depth Analysis on the Abstraction and Reasoning Corpus](https://arxiv.org/abs/2403.11793) | 使用抽象和推理语料库（ARC）数据集评估大型语言模型的推理和上下文理解能力，结果显示虽然大型语言模型具有较弱的推理能力，但在逻辑连贯性、组合性和效率方面仍然落后，实验结果有助于提出实现人类水平推理的发展路径。 |
| [^5] | [PERL: Parameter Efficient Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2403.10704) | 使用低秩适应（LoRA）方法进行参数高效强化学习（PERL），能够在与传统RLHF设置相当的性能下，实现更快的训练和更少的内存占用。 |
| [^6] | [Truth-Aware Context Selection: Mitigating the Hallucinations of Large Language Models Being Misled by Untruthful Contexts](https://arxiv.org/abs/2403.07556) | 提出了一种名为真相感知的上下文选择（TACS）的轻量级方法，可以通过对输入上下文进行真相检测并构建相应的注意力蒙版来缓解大型语言模型被不真实上下文误导产生幻觉 |
| [^7] | [Large Language Models and Games: A Survey and Roadmap](https://arxiv.org/abs/2402.18659) | 这项研究调查了大型语言模型在游戏领域中的多种应用及其角色，指出了未开发领域和未来发展方向，同时探讨了在游戏领域中大型语言模型的潜力和限制。 |
| [^8] | [Enable Language Models to Implicitly Learn Self-Improvement From Data.](http://arxiv.org/abs/2310.00898) | 该论文探索了如何让语言模型隐式学习自我改进，并减少对人类标注的依赖。 |
| [^9] | [RRWKV: Capturing Long-range Dependencies in RWKV.](http://arxiv.org/abs/2306.05176) | 本文介绍了一种新的RRWKV架构，它在保持记忆和计算效率的同时，通过加入回顾能力有效地捕捉长距离依赖关系。 |
| [^10] | [An Experimental Study on Sentiment Classification of Moroccan dialect texts in the web.](http://arxiv.org/abs/2303.15987) | 本研究采用机器学习模型对YouTube评论中的摩洛哥方言进行情感分类，采用多种文本预处理和数据表示技术对文本进行分析，研究该方言的意见和情感表达。 |
| [^11] | [Predicting Sentence-Level Factuality of News and Bias of Media Outlets.](http://arxiv.org/abs/2301.11850) | 本论文提出了一种针对整个媒体的细粒度可靠性分析方法，在手动制作的“FactNews”数据库上，通过 fine-tuning BERT 模型预测新闻报道的句子级别事实性和媒体倾向。此方法可应用于任何其他语言。 |

# 详细

[^1]: 在CodeLLMs中实现类型预测的鲁棒激活导向技术

    Activation Steering for Robust Type Prediction in CodeLLMs

    [https://arxiv.org/abs/2404.01903](https://arxiv.org/abs/2404.01903)

    我们提出了一种激活导向技术，通过编辑模型内部激活来改善CodeLLMs在代码类型预测中对于语法干扰的鲁棒性，并成功应用于Python和TypeScript的类型预测，将类型误差率纠正高达90%。

    

    预训练在代码上的现代LLMs能够成功地完成各种编程任务。然而，它们的性能对语法特征非常敏感，例如变量和类型的名称、代码结构以及类型提示的存在。我们提出了一种推理时技术，使CodeLLMs更能抵御语法干扰因素，这些因素与语义无关。我们的方法依赖于激活导向，涉及编辑内部模型激活以将模型引导到正确的预测。我们通过从突变测试中汲取灵感构建激活向量的新方法，该方法构建最小的破坏语义的代码编辑。相比之下，我们从保留语义的代码编辑中构建激活向量。我们将我们的方法应用于逐渐类型化语言Python和TypeScript的类型预测任务。这种方法可以纠正高达90%的类型错误预测。

    arXiv:2404.01903v1 Announce Type: new  Abstract: Contemporary LLMs pretrained on code are capable of succeeding at a wide variety of programming tasks. However, their performance is very sensitive to syntactic features, such as the names of variables and types, the structure of code, and presence of type hints. We contribute an inference-time technique to make CodeLLMs more robust to syntactic distractors that are semantically irrelevant. Our methodology relies on activation steering, which involves editing internal model activations to steer the model towards the correct prediction. We contribute a novel way to construct steering vectors by taking inspiration from mutation testing, which constructs minimal semantics-breaking code edits. In contrast, we construct steering vectors from semantics-preserving code edits. We apply our approach to the task of type prediction for the gradually typed languages Python and TypeScript. This approach corrects up to 90% of type mispredictions. Fina
    
[^2]: IndoCulture: 探索印尼十一个省份间地理影响的文化常识推理

    IndoCulture: Exploring Geographically-Influenced Cultural Commonsense Reasoning Across Eleven Indonesian Provinces

    [https://arxiv.org/abs/2404.01854](https://arxiv.org/abs/2404.01854)

    本研究介绍了IndoCulture项目，旨在通过当地人手动收集数据，探索印尼十一个省份间地理影响的文化常识推理。评估发现，即使是最好的语言模型也在特定省份上表现更准确，而添加地理信息有助于提高模型性能。

    

    尽管常识推理受文化和地理因素的极大影响，先前关于语言模型的研究主要集中在英语文化上，可能导致一种以英语为中心的偏见。本文介绍了IndoCulture，旨在理解地理因素对语言模型推理能力的影响，特别强调了十一个印尼省份内所发现的多样文化。与先前依赖模板（Yin等，2022）和在线抓取（Fung等，2024）的作品不同，我们通过询问当地人手动开发预定义主题的上下文和合理选项来创建IndoCulture。对23个语言模型的评估揭示了几个见解：（1）即使是最好的开源模型也难以达到53.2％的准确性，（2）模型通常为特定省份（如巴厘岛和西爪哇）提供更准确的预测，（3）包含地理信息可显着改善模型的表现。

    arXiv:2404.01854v1 Announce Type: new  Abstract: Although commonsense reasoning is greatly shaped by cultural and geographical factors, previous studies on language models have predominantly centered on English cultures, potentially resulting in an Anglocentric bias. In this paper, we introduce IndoCulture, aimed at understanding the influence of geographical factors on language model reasoning ability, with a specific emphasis on the diverse cultures found within eleven Indonesian provinces. In contrast to prior works that relied on templates (Yin et al., 2022) and online scrapping (Fung et al., 2024), we created IndoCulture by asking local people to manually develop the context and plausible options based on predefined topics. Evaluations of 23 language models reveal several insights: (1) even the best open-source model struggles with an accuracy of 53.2%, (2) models often provide more accurate predictions for specific provinces, such as Bali and West Java, and (3) the inclusion of l
    
[^3]: 等等，这都是令牌噪音？一直就是吗：利用 Shapley 值解释 LLM 行为

    Wait, It's All Token Noise? Always Has Been: Interpreting LLM Behavior Using Shapley Value

    [https://arxiv.org/abs/2404.01332](https://arxiv.org/abs/2404.01332)

    使用Shapley值方法解释LLM行为，揭示了所谓的“令牌噪音”效应，揭示了LLMs的决策在很大程度上受到提示组件的影响

    

    大型语言模型（LLMs）的出现为模拟人类行为和认知过程开辟了新的可能性，潜在应用包括市场研究和消费者行为分析等各个领域。然而，由于LLMs的显著差异暗示了不同的基础过程在起作用，以及LLMs对提示变化的敏感性，利用LLMs作为人类主体的替代仍然存在不确定性。本文提出了一种基于合作博弈理论中Shapley值的新方法来解释LLM行为，并量化每个提示组件对模型输出的相对贡献。通过两个应用--一个离散选择实验和一个认知偏见调查，我们展示了Shapley值方法如何揭示我们所谓的“令牌噪音”效应，即LLM决策受到的影响严重偏向于

    arXiv:2404.01332v1 Announce Type: cross  Abstract: The emergence of large language models (LLMs) has opened up exciting possibilities for simulating human behavior and cognitive processes, with potential applications in various domains, including marketing research and consumer behavior analysis. However, the validity of utilizing LLMs as stand-ins for human subjects remains uncertain due to glaring divergences that suggest fundamentally different underlying processes at play and the sensitivity of LLM responses to prompt variations. This paper presents a novel approach based on Shapley values from cooperative game theory to interpret LLM behavior and quantify the relative contribution of each prompt component to the model's output. Through two applications-a discrete choice experiment and an investigation of cognitive biases-we demonstrate how the Shapley value method can uncover what we term "token noise" effects, a phenomenon where LLM decisions are disproportionately influenced by 
    
[^4]: 大型语言模型的推理能力：对抽象和推理语料库的深入分析

    Reasoning Abilities of Large Language Models: In-Depth Analysis on the Abstraction and Reasoning Corpus

    [https://arxiv.org/abs/2403.11793](https://arxiv.org/abs/2403.11793)

    使用抽象和推理语料库（ARC）数据集评估大型语言模型的推理和上下文理解能力，结果显示虽然大型语言模型具有较弱的推理能力，但在逻辑连贯性、组合性和效率方面仍然落后，实验结果有助于提出实现人类水平推理的发展路径。

    

    评估大型语言模型（LLMs）推理能力的现有方法以结果为中心，使得评估推理过程变得困难。我们引入了一种新方法，使用抽象和推理语料库（ARC）数据集以过程为中心的方式评估大型语言模型的推理和上下文理解能力。ARC要求解决问题时具有严谨的逻辑结构，这使得它成为一个能够促进模型推理能力与人类进行比较的基准。实验结果证实，虽然大型语言模型具有较弱的推理能力，但在逻辑连贯性、组合性和效率方面仍然落后。我们的实验突显了LLMs的推理能力，并提出了实现人类水平推理的发展路径。

    arXiv:2403.11793v1 Announce Type: cross  Abstract: The existing methods for evaluating the inference abilities of Large Language Models (LLMs) have been results-centric, making it difficult to assess the inference process. We introduce a new approach using the Abstract and Reasoning Corpus (ARC) dataset to evaluate the inference and contextual understanding abilities of large language models in a process-centric manner. ARC demands rigorous logical structures for problem-solving, making it a benchmark that facilitates the comparison of model inference abilities with humans. Experimental results confirm that while large language models possess weak inference abilities, they still lag in terms of logical coherence, compositionality, and productivity. Our experiments highlight the reasoning capabilities of LLMs, proposing development paths for achieving human-level reasoning.
    
[^5]: PERL: 从人类反馈中实现参数高效强化学习

    PERL: Parameter Efficient Reinforcement Learning from Human Feedback

    [https://arxiv.org/abs/2403.10704](https://arxiv.org/abs/2403.10704)

    使用低秩适应（LoRA）方法进行参数高效强化学习（PERL），能够在与传统RLHF设置相当的性能下，实现更快的训练和更少的内存占用。

    

    强化学习从人类反馈（RLHF）已被证明是一种将预训练的大型语言模型（LLMs）与人类偏好对齐的有效方法。然而，使用RLHF训练模型计算成本高昂，且整个过程复杂。在本研究中，我们研究了RLHF，其中基础模型使用胡等人提出的低秩适应（LoRA）的参数高效方法进行训练。我们探讨了“参数高效强化学习”（PERL）的设置，在其中我们使用LoRA进行奖励模型训练和强化学习。我们将PERL与传统的微调（全调）在包括2个新数据集在内的7个基准测试中的奖励建模和强化学习方面的各种配置进行了比较。我们发现，PERL的性能与传统的RLHF设置相当，同时训练速度更快，内存占用更少。这使得RLHF具有很高的性能，同时减少了计算成本。

    arXiv:2403.10704v1 Announce Type: cross  Abstract: Reinforcement Learning from Human Feedback (RLHF) has proven to be a strong method to align Pretrained Large Language Models (LLMs) with human preferences. But training models with RLHF is computationally expensive, and an overall complex process. In this work, we study RLHF where the underlying models are trained using the parameter efficient method of Low-Rank Adaptation (LoRA) introduced by Hu et al. [2021]. We investigate the setup of "Parameter Efficient Reinforcement Learning" (PERL), in which we perform reward model training and reinforcement learning using LoRA. We compare PERL to conventional fine-tuning (full-tuning) across various configurations for 7 benchmarks, including 2 novel datasets, of reward modeling and reinforcement learning. We find that PERL performs on par with the conventional RLHF setting, while training faster, and with less memory. This enables the high performance of RLHF, while reducing the computational 
    
[^6]: 真相感知的上下文选择：缓解大型语言模型被不真实上下文误导产生幻觉

    Truth-Aware Context Selection: Mitigating the Hallucinations of Large Language Models Being Misled by Untruthful Contexts

    [https://arxiv.org/abs/2403.07556](https://arxiv.org/abs/2403.07556)

    提出了一种名为真相感知的上下文选择（TACS）的轻量级方法，可以通过对输入上下文进行真相检测并构建相应的注意力蒙版来缓解大型语言模型被不真实上下文误导产生幻觉

    

    尽管大型语言模型（LLMs）展示了令人印象深刻的文本生成能力，但它们很容易被用户或知识论证工具提供的不真实上下文误导，从而产生幻觉。为了减轻LLMs被不真实信息误导并利用知识论证，我们提出了真相感知的上下文选择（TACS），这是一种轻量级方法，可以从输入中屏蔽不真实的上下文。TACS首先对输入上下文进行真相检测，利用LLM内的参数化知识。随后，根据每个位置的真实性构建相应的注意力蒙版，选择真实的上下文并丢弃不真实的上下文。此外，我们引入一个新的评估指标，扰动适应率，以进一步研究LLMs接受真实信息和抵制不真实信息的能力。

    arXiv:2403.07556v1 Announce Type: new  Abstract: Although large language models (LLMs) have demonstrated impressive text generation capabilities, they are easily misled by the untruthful context provided by users or knowledge argumentation tools, thereby producing hallucinations. To alleviate the LLMs from being misled by untruthful information and take advantage of knowledge argumentation, we propose Truth-Aware Context Selection (TACS), a lightweight method to shield untruthful context from the inputs. TACS begins by performing truth detection on the input context, leveraging the parameterized knowledge within the LLM. Subsequently, it constructs a corresponding attention mask based on the truthfulness of each position, selecting the truthful context and discarding the untruthful context. Additionally, we introduce a new evaluation metric, Disturbance Adaption Rate, to further study the LLMs' ability to accept truthful information and resist untruthful information. Experimental resul
    
[^7]: 大型语言模型与游戏：调研与路线图

    Large Language Models and Games: A Survey and Roadmap

    [https://arxiv.org/abs/2402.18659](https://arxiv.org/abs/2402.18659)

    这项研究调查了大型语言模型在游戏领域中的多种应用及其角色，指出了未开发领域和未来发展方向，同时探讨了在游戏领域中大型语言模型的潜力和限制。

    

    近年来，大型语言模型（LLMs）的研究急剧增加，并伴随着公众对该主题的参与。尽管起初是自然语言处理中的一小部分，LLMs在广泛的应用和领域中展现出显著潜力，包括游戏。本文调查了LLMs在游戏中及为游戏提供支持的各种应用的最新技术水平，并明确了LLMs在游戏中可以扮演的不同角色。重要的是，我们讨论了尚未开发的领域和LLMs在游戏中未来应用的有前途的方向，以及在游戏领域中LLMs的潜力和限制。作为LLMs和游戏交叉领域的第一份综合调查和路线图，我们希望本文能够成为这一激动人心的新领域的开创性研究和创新的基础。

    arXiv:2402.18659v1 Announce Type: cross  Abstract: Recent years have seen an explosive increase in research on large language models (LLMs), and accompanying public engagement on the topic. While starting as a niche area within natural language processing, LLMs have shown remarkable potential across a broad range of applications and domains, including games. This paper surveys the current state of the art across the various applications of LLMs in and for games, and identifies the different roles LLMs can take within a game. Importantly, we discuss underexplored areas and promising directions for future uses of LLMs in games and we reconcile the potential and limitations of LLMs within the games domain. As the first comprehensive survey and roadmap at the intersection of LLMs and games, we are hopeful that this paper will serve as the basis for groundbreaking research and innovation in this exciting new field.
    
[^8]: 让语言模型从数据中隐式学习自我改进能力

    Enable Language Models to Implicitly Learn Self-Improvement From Data. (arXiv:2310.00898v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2310.00898](http://arxiv.org/abs/2310.00898)

    该论文探索了如何让语言模型隐式学习自我改进，并减少对人类标注的依赖。

    

    大型语言模型在开放式文本生成任务中展示了卓越的能力。然而，这些任务的本质决定了模型的回答质量始终有改进的空间。为了解决这个挑战，已经提出了各种方法来增强语言模型的性能。越来越多的关注点集中在使语言模型自我改进其回答质量上，从而减少对广泛的人工标注工作来收集多样化和高质量的训练数据的依赖。最近，基于提示的方法因其有效性、高效性和便利性而受到广泛关注。然而，这些方法通常需要为语言模型提供明确和详尽的指示。对于手动推导和提供所有必要的指示来实现现实世界复杂目标的改进（例如，更有帮助性和更少有害性），这是昂贵且具有挑战性的。

    Large Language Models (LLMs) have demonstrated remarkable capabilities in open-ended text generation tasks. However, the inherent open-ended nature of these tasks implies that there is always room for improvement in the quality of model responses. To address this challenge, various approaches have been proposed to enhance the performance of LLMs. There has been a growing focus on enabling LLMs to self-improve their response quality, thereby reducing the reliance on extensive human annotation efforts for collecting diverse and high-quality training data. Recently, prompting-based methods have been widely explored among self-improvement methods owing to their effectiveness, efficiency, and convenience. However, those methods usually require explicitly and thoroughly written rubrics as inputs to LLMs. It is expensive and challenging to manually derive and provide all necessary rubrics with a real-world complex goal for improvement (e.g., being more helpful and less harmful). To this end, 
    
[^9]: RRWKV：在RWKV中捕捉长距离依赖关系

    RRWKV: Capturing Long-range Dependencies in RWKV. (arXiv:2306.05176v1 [cs.CL])

    [http://arxiv.org/abs/2306.05176](http://arxiv.org/abs/2306.05176)

    本文介绍了一种新的RRWKV架构，它在保持记忆和计算效率的同时，通过加入回顾能力有效地捕捉长距离依赖关系。

    

    由于Transformer惊人的点积注意力，它已经成为各种自然语言处理（NLP）任务中的主要架构。最近，Receptance Weighted Key Value（RWKV）架构遵循非Transformer架构，消除了点积注意力的缺点，其中存储和计算复杂度随着序列长度呈二次扩展。尽管RWKV利用了线性张量积注意机制并通过部署时间序列模式实现了并行计算，但与标准Transformer中直接交互获得的完整信息相比，它无法捕捉长距离依赖关系，因为其受限于向后查看先前信息的能力。因此，本文通过将回顾能力纳入RWKV中来设计Retrospected Receptance Weighted Key Value（RRWKV）架构，以有效地吸收信息，同时保持记忆和计算效率。

    Owing to the impressive dot-product attention, the Transformers have been the dominant architectures in various natural language processing (NLP) tasks. Recently, the Receptance Weighted Key Value (RWKV) architecture follows a non-transformer architecture to eliminate the drawbacks of dot-product attention, where memory and computational complexity exhibits quadratic scaling with sequence length. Although RWKV has exploited a linearly tensor-product attention mechanism and achieved parallelized computations by deploying the time-sequential mode, it fails to capture long-range dependencies because of its limitation on looking back at previous information, compared with full information obtained by direct interactions in the standard transformer. Therefore, the paper devises the Retrospected Receptance Weighted Key Value (RRWKV) architecture via incorporating the retrospecting ability into the RWKV to effectively absorb information, which maintains memory and computational efficiency as 
    
[^10]: 关于摩洛哥方言文本情感分类的实验研究

    An Experimental Study on Sentiment Classification of Moroccan dialect texts in the web. (arXiv:2303.15987v1 [cs.CL])

    [http://arxiv.org/abs/2303.15987](http://arxiv.org/abs/2303.15987)

    本研究采用机器学习模型对YouTube评论中的摩洛哥方言进行情感分类，采用多种文本预处理和数据表示技术对文本进行分析，研究该方言的意见和情感表达。

    

    随着社交媒体网站的迅速增长，自动获取用户反馈成为评估其在线趋势和行为的重要任务。尽管信息大量可用，阿拉伯使用者数量增加，但很少有研究处理阿拉伯方言。本文旨在准确研究在YouTube评论中表达的真实摩洛哥方言文本的观点和情感，使用一些众所周知且常用的情感分析方法进行。通过采用许多文本预处理和数据表示技术，我们旨在比较我们使用最常用的监督分类器进行分类结果：K最近邻（KNN）、支持向量机（SVM）、朴素贝叶斯（NB）和深度学习（DL）分类器，这些都是基于我们收集和手动注释的YouTube摩洛哥方言数据集。

    With the rapid growth of the use of social media websites, obtaining the users' feedback automatically became a crucial task to evaluate their tendencies and behaviors online. Despite this great availability of information, and the increasing number of Arabic users only few research has managed to treat Arabic dialects. The purpose of this paper is to study the opinion and emotion expressed in real Moroccan texts precisely in the YouTube comments using some well-known and commonly used methods for sentiment analysis. In this paper, we present our work of Moroccan dialect comments classification using Machine Learning (ML) models and based on our collected and manually annotated YouTube Moroccan dialect dataset. By employing many text preprocessing and data representation techniques we aim to compare our classification results utilizing the most commonly used supervised classifiers: k-nearest neighbors (KNN), Support Vector Machine (SVM), Naive Bayes (NB), and deep learning (DL) classif
    
[^11]: 预测新闻事实性和媒体倾向的句子级别可靠性分析

    Predicting Sentence-Level Factuality of News and Bias of Media Outlets. (arXiv:2301.11850v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2301.11850](http://arxiv.org/abs/2301.11850)

    本论文提出了一种针对整个媒体的细粒度可靠性分析方法，在手动制作的“FactNews”数据库上，通过 fine-tuning BERT 模型预测新闻报道的句子级别事实性和媒体倾向。此方法可应用于任何其他语言。

    

    预测新闻报道的事实性和媒体倾向对于自动化的新闻信誉和事实核查是很重要的。本文提出了对整个媒体进行细粒度可靠性分析的方法。我们研究了预测新闻报道的句子级别事实性和媒体倾向，这可以更精确地解释整个 source 的可靠程度。我们首先手动制作了一个大型的句子级别数据库，“FactNews”，由 6191 个专家注释的句子组成，注释依据来自 AllSides 的事实性和媒体倾向定义。最后，由于巴西存在严重的虚假新闻和政治极化问题，我们提供了用于葡萄牙语的数据集和基线模型。但是，我们的方法可以应用于任何其他语言。

    Predicting the factuality of news reporting and bias of media outlets is surely relevant for automated news credibility and fact-checking. While prior work has focused on the veracity of news, we propose a fine-grained reliability analysis of the entire media. Specifically, we study the prediction of sentence-level factuality of news reporting and bias of media outlets, which may explain more accurately the overall reliability of the entire source. We first manually produced a large sentence-level dataset, titled "FactNews", composed of 6,191 sentences expertly annotated according to factuality and media bias definitions from AllSides. As a result, baseline models for sentence-level factuality prediction were presented by fine-tuning BERT. Finally, due to the severity of fake news and political polarization in Brazil, both dataset and baseline were proposed for Portuguese. However, our approach may be applied to any other language.
    

