# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Decoding Speculative Decoding](https://rss.arxiv.org/abs/2402.01528) | 推测解码是一种用于加速大型语言模型推断的技术，但我们的实验表明，选择的草稿模型生成的令牌被目标模型接受的概率越高，吞吐量越低。我们通过大量实验，分析了各种因素对推测解码效果的影响，并提出了一个分析模型来提高效率。 |
| [^2] | [CodeBenchGen: Creating Scalable Execution-based Code Generation Benchmarks](https://arxiv.org/abs/2404.00566) | 提出了CodeBenchGen框架，通过利用大型语言模型将任意代码转化为评估示例，创造了一个包含大量代码示例的数据集Exec-CSN，展示了其可扩展性和实用性。 |
| [^3] | [SOTOPIA-$\pi$: Interactive Learning of Socially Intelligent Language Agents](https://arxiv.org/abs/2403.08715) | 提出了一种交互式学习方法SOTOPIA-$\pi$，该方法利用行为克隆和自我强化训练，改进了语言代理的社交智能，使其达到了专家模型的水平，并提高了安全性。 |
| [^4] | [Multimodal Large Language Models to Support Real-World Fact-Checking](https://arxiv.org/abs/2403.03627) | 多模态大型语言模型在支持现实世界事实核查中展现出优越性能，并能够解释恶意和误导性声明的不合理之处和潜在动机。 |
| [^5] | [Enhancing Long-Term Recommendation with Bi-level Learnable Large Language Model Planning](https://arxiv.org/abs/2403.00843) | 利用大型语言模型的规划能力来增强长期推荐，使模型在个性化推荐中更有效地理解和应用任务解决原则 |
| [^6] | [DEEM: Dynamic Experienced Expert Modeling for Stance Detection](https://arxiv.org/abs/2402.15264) | 本文提出了一种Dynamic Experienced Expert Modeling（DEEM）方法，利用生成的经验专家使LLMs能够以半参数化方式进行推理，提高了在立场检测任务中的性能。 |
| [^7] | [Fine-Tuning, Prompting, In-Context Learning and Instruction-Tuning: How Many Labelled Samples Do We Need?](https://arxiv.org/abs/2402.12819) | 专门模型通常只需少量标记样本（100-1000个）就能与通用模型持平甚至更好，取决于任务的复杂性和结果的变化。 |
| [^8] | [Enhancing Amharic-LLaMA: Integrating Task Specific and Generative Datasets](https://arxiv.org/abs/2402.08015) | 本研究通过整合任务特定和生成数据集来增强Amharic-LLaMA模型，提高了阿姆哈拉语言模型的性能。他们通过创建阿姆哈拉语指令微调数据集和微调模型，在不同的NLP任务中取得了有希望的结果。 |
| [^9] | [Forecasting Events in Soccer Matches Through Language](https://arxiv.org/abs/2402.06820) | 本文提出了一种使用语言模型预测足球比赛中下一个事件的方法，该方法受到大型语言模型方法的启发。通过深度学习和WyScout数据集，该方法在预测准确性方面明显超过了以往的方法。该方法的应用包括博彩和比赛分析，并提供了一个模拟骨架用于构建分析流水线。 |
| [^10] | [PythonSaga: Redefining the Benchmark to Evaluate Code Generating LLM](https://arxiv.org/abs/2401.03855) | PythonSaga提出了一种新的基准，针对Python代码生成进行评估,弥补了现有基准存在的编程概念偏见和简单任务普遍性的问题 |
| [^11] | [Batched Low-Rank Adaptation of Foundation Models](https://arxiv.org/abs/2312.05677) | 提出了Fast LoRA（FLoRA）框架，使得基础模型的低秩调整可以高效批处理异构请求，并在绩效上保持竞争性。 |
| [^12] | [MAIRA-1: A specialised large multimodal model for radiology report generation](https://arxiv.org/abs/2311.13668) | MAIRA-1是一种专门用于放射学报告生成的大型多模态模型，在与预训练的视觉编码器对齐和文本数据增强的基础上，利用了CXR特定的图像编码器和经过微调的大型语言模型，生成具有最先进质量的报告。 |
| [^13] | [Reasoning over Description Logic-based Contexts with Transformers](https://arxiv.org/abs/2311.08941) | 本研究构建了一个由描述逻辑知识库生成的合成自然语言问答数据集，以评估基于Transformer模型在丰富语境中的推理能力。 |
| [^14] | [A Linguistic Comparison between Human and ChatGPT-Generated Conversations.](http://arxiv.org/abs/2401.16587) | 本研究比较了人类和ChatGPT生成的对话的语言差异，发现ChatGPT在社交、分析、认知、关注焦点和积极情绪等方面表现出色，但人类对话更具变异性和真实性，尽管在情绪方面无显著差异。同时，该研究还提供了一个新颖的、由ChatGPT生成的对话组成的数据集。 |
| [^15] | [Weakly Supervised Gaussian Contrastive Grounding with Large Multimodal Models for Video Question Answering.](http://arxiv.org/abs/2401.10711) | 本论文提出了一种使用大型多模型的弱监督高斯对比基础模型来处理视频问答问题的方法。通过将问题和答案对作为事件描述，找到多个关键帧作为目标时刻，并利用这些时刻作为伪标签来强制LMMs进行推理。所提出的方法使用轻量级的基于高斯的对比基础模块（GCG）来学习时效结构。 |
| [^16] | [Adapting Large Language Models for Education: Foundational Capabilities, Potentials, and Challenges.](http://arxiv.org/abs/2401.08664) | 本文回顾了针对教育能力的大型语言模型研究，包括数学、写作、编程、推理和基于知识的问答，旨在探索其在构建下一代智能教育系统中的潜力和挑战。 |
| [^17] | [A First Look at Information Highlighting in Stack Overflow Answers.](http://arxiv.org/abs/2401.01472) | 本论文进行了首次大规模的探索性研究，研究了Stack Overflow回答中的信息高亮。通过使用神经网络架构，开发了自动推荐突出内容的方法。 |
| [^18] | [Structured Packing in LLM Training Improves Long Context Utilization.](http://arxiv.org/abs/2312.17296) | 本论文研究了长上下文大型语言模型（LLM）中上下文利用不足的问题，并通过将相关文档纳入训练示例中来改进模型的困惑度。通过引入Structured Packing for Long Context (SPLiCe)方法，使用检索方法将最互相关文档汇集到单个训练上下文中，进一步提高了模型的性能。 |
| [^19] | [Kosmos-G: Generating Images in Context with Multimodal Large Language Models.](http://arxiv.org/abs/2310.02992) | 本文介绍了Kosmos-G，一种利用多模态大型语言模型（MLLM）在上下文中生成图像的模型。该模型通过使用文本模态作为锚点，将MLLM的输出空间与CLIP对齐，并进行组合指令调整。Kosmos-G展示了零样本多实体主题驱动生成的独特能力。 |
| [^20] | [Ensemble Distillation for Unsupervised Constituency Parsing.](http://arxiv.org/abs/2310.01717) | 本论文提出了一种集成蒸馏的方法来提高无监督句法解析的性能，并且通过蒸馏将集成知识转移到一个学生模型中，解决了常见的多教师蒸馏方法中的过度平滑问题。 |
| [^21] | [Towards Understanding In-Context Learning with Contrastive Demonstrations and Saliency Maps.](http://arxiv.org/abs/2307.05052) | 本研究探索了对比演示和显著性图在上下文学习中的作用，并发现改变标签对显著性有显著影响，尤其对于更大的语言模型更为明显。在情感分析任务中，将表达情感的术语改为中性词并不像改变标签那样具有显著影响。另外，补充解释在提高上下文学习方面是有效的。 |
| [^22] | [On Bias and Fairness in NLP: How to have a fairer text classification?.](http://arxiv.org/abs/2305.12829) | 本文从上游偏见、样本偏见和过度放大偏见三方面分析了NLP模型中的偏见如何影响文本分类的公平性，并针对过度放大偏见通过微调语言模型达到公平分类效果。提出了构建公正文本分类模型的实用指南。 |
| [^23] | [Retrieving Texts based on Abstract Descriptions.](http://arxiv.org/abs/2305.12517) | 本研究针对语义检索问题，提出了一种基于摘要描述的文本检索模型，通过改进当前的文本嵌入方法，在标准最近邻搜索中取得了显著性能提升。 |

# 详细

[^1]: 解码推测解码

    Decoding Speculative Decoding

    [https://rss.arxiv.org/abs/2402.01528](https://rss.arxiv.org/abs/2402.01528)

    推测解码是一种用于加速大型语言模型推断的技术，但我们的实验表明，选择的草稿模型生成的令牌被目标模型接受的概率越高，吞吐量越低。我们通过大量实验，分析了各种因素对推测解码效果的影响，并提出了一个分析模型来提高效率。

    

    推测解码是一种常用的技术，用于加速大型语言模型（LLM）的推断，而不修改其结果。在对LLM进行推断时，推测解码使用较小的草稿模型生成推测令牌，然后使用目标LLM验证这些草稿令牌。推测解码提供的加速取决于草稿模型的选择。普遍建议选择一个草稿模型，该模型生成的令牌被LLM接受的概率很高，以实现最高吞吐量。然而，我们的实验结果与之相反，随着生成的令牌被目标模型接受的概率增加，吞吐量减少。为了理解这一现象，我们进行了大量实验，对影响推测解码的不同因素进行了表征，并研究了这些因素如何相互作用和影响加速效果。基于我们的实验结果，我们描述了一个分析模型，可以使用该模型来进行决策，提高推测解码的效率。

    Speculative Decoding is a widely used technique to speed up inference for Large Language Models (LLMs) without modifying its outcome. When performing inference on an LLM, speculative decoding uses a smaller draft model which generates speculative tokens and then uses the target LLM to verify those draft tokens. The speedup provided by speculative decoding heavily depends on the choice of the draft model. It has been widely suggested to select a draft model that provides a high probability of the generated token being accepted by the LLM to achieve the highest throughput. However, our experiments indicate the contrary with throughput diminishing as the probability of generated tokens to be accepted by the target model increases. To understand this phenomenon, we perform extensive experiments to characterize the different factors that affect speculative decoding and how those factors interact and affect the speedups. Based on our experiments we describe an analytical model which can be u
    
[^2]: CodeBenchGen: 创建可扩展的基于执行的代码生成基准

    CodeBenchGen: Creating Scalable Execution-based Code Generation Benchmarks

    [https://arxiv.org/abs/2404.00566](https://arxiv.org/abs/2404.00566)

    提出了CodeBenchGen框架，通过利用大型语言模型将任意代码转化为评估示例，创造了一个包含大量代码示例的数据集Exec-CSN，展示了其可扩展性和实用性。

    

    为了促进在不同场景下评估代码生成系统，我们提出了CodeBenchGen，这是一个框架，可以创建可扩展的基于执行的基准，仅需要轻微的人类指导。具体来说，我们利用一个大型语言模型（LLM）将任意代码片段转化为评估示例，包括用于执行评估的测试用例。我们通过创建包含来自CodeSearchNet数据集的367个GitHub存储库中的代码修改的293个库的1,931个例子的数据集Exec-CSN，展示了我们框架的实用性。为了展示Exec-CSN中示例的复杂性和可解性，我们进行了一个人类研究，结果显示81.3%的例子可以被人类解决，61%被评为“需要努力解决”。我们对开源和专有模型进行了代码生成实验，并分析了人类和模型的性能。

    arXiv:2404.00566v1 Announce Type: cross  Abstract: To facilitate evaluation of code generation systems across diverse scenarios, we present CodeBenchGen, a framework to create scalable execution-based benchmarks that only requires light guidance from humans. Specifically, we leverage a large language model (LLM) to convert an arbitrary piece of code into an evaluation example, including test cases for execution-based evaluation. We illustrate the usefulness of our framework by creating a dataset, Exec-CSN, which includes 1,931 examples involving 293 libraries revised from code in 367 GitHub repositories taken from the CodeSearchNet dataset. To demonstrate the complexity and solvability of examples in Exec-CSN, we present a human study demonstrating that 81.3% of the examples can be solved by humans and 61% are rated as ``requires effort to solve''. We conduct code generation experiments on open-source and proprietary models and analyze the performance of both humans and models. We will
    
[^3]: SOTOPIA-$\pi$: 交互式学习社交智能语言代理

    SOTOPIA-$\pi$: Interactive Learning of Socially Intelligent Language Agents

    [https://arxiv.org/abs/2403.08715](https://arxiv.org/abs/2403.08715)

    提出了一种交互式学习方法SOTOPIA-$\pi$，该方法利用行为克隆和自我强化训练，改进了语言代理的社交智能，使其达到了专家模型的水平，并提高了安全性。

    

    人类通过模仿和社交互动来学习社交技能。现有研究在构建语言代理方面很少涉及这种社交学习过程。受到这一空白的启发，我们提出了一种交互式学习方法SOTOPIA-$\pi$，改进了语言代理的社交智能。该方法利用行为克隆和自我强化训练，根据大型语言模型(LLM)评分对经过筛选的社交互动数据进行训练。我们证明了我们的训练方法使一个7B的LLM达到了专家模型(GPT-4-based agent)的社交目标完成能力，同时提高了语言代理的安全性，并在MMLU基准上保持了通用的问答能力。我们还发现，这种训练范式揭示了LLM评估社交智能的一些困难：基于LLM的评估者高估了专门针对社交互动训练的语言代理的能力。

    arXiv:2403.08715v1 Announce Type: new  Abstract: Humans learn social skills through both imitation and social interaction. This social learning process is largely understudied by existing research on building language agents. Motivated by this gap, we propose an interactive learning method, SOTOPIA-$\pi$, improving the social intelligence of language agents. This method leverages behavior cloning and self-reinforcement training on filtered social interaction data according to large language model (LLM) ratings. We show that our training method allows a 7B LLM to reach the social goal completion ability of an expert model (GPT-4-based agent), while improving the safety of language agents and maintaining general QA ability on the MMLU benchmark. We also find that this training paradigm uncovers some difficulties in LLM-based evaluation of social intelligence: LLM-based evaluators overestimate the abilities of the language agents trained specifically for social interaction.
    
[^4]: 多模态大型语言模型支持现实世界事实核查

    Multimodal Large Language Models to Support Real-World Fact-Checking

    [https://arxiv.org/abs/2403.03627](https://arxiv.org/abs/2403.03627)

    多模态大型语言模型在支持现实世界事实核查中展现出优越性能，并能够解释恶意和误导性声明的不合理之处和潜在动机。

    

    多模态大型语言模型（MLLMs）具有潜力支持人类处理大量信息。虽然MLLMs已经被用作事实核查工具，但就其在此方面的能力和局限性而言，尚未得到充分研究。我们旨在弥合这一差距。具体而言，我们提出了一个框架，系统评估当前多模态模型促进现实世界事实核查的能力。我们的方法论是无需证据的，仅利用这些模型的固有知识和推理能力。通过设计能够提取模型预测、解释和置信水平的提示，我们深入研究关于模型准确性、鲁棒性以及失败原因的研究问题。我们在实证上发现，(1) GPT-4V在识别恶意和误导性多模态声明方面表现出超凡性能，能够解释不合理的方面和潜在动机，以及(2)现有的o

    arXiv:2403.03627v1 Announce Type: cross  Abstract: Multimodal large language models (MLLMs) carry the potential to support humans in processing vast amounts of information. While MLLMs are already being used as a fact-checking tool, their abilities and limitations in this regard are understudied. Here is aim to bridge this gap. In particular, we propose a framework for systematically assessing the capacity of current multimodal models to facilitate real-world fact-checking. Our methodology is evidence-free, leveraging only these models' intrinsic knowledge and reasoning capabilities. By designing prompts that extract models' predictions, explanations, and confidence levels, we delve into research questions concerning model accuracy, robustness, and reasons for failure. We empirically find that (1) GPT-4V exhibits superior performance in identifying malicious and misleading multimodal claims, with the ability to explain the unreasonable aspects and underlying motives, and (2) existing o
    
[^5]: 利用双层可学习大型语言模型规划增强长期推荐

    Enhancing Long-Term Recommendation with Bi-level Learnable Large Language Model Planning

    [https://arxiv.org/abs/2403.00843](https://arxiv.org/abs/2403.00843)

    利用大型语言模型的规划能力来增强长期推荐，使模型在个性化推荐中更有效地理解和应用任务解决原则

    

    传统推荐系统倾向于过分迎合用户的即时兴趣而忽视他们的长期参与。 为了解决这个问题，在推荐决策过程中合并规划能力是至关重要的，以开发能够同时考虑即时兴趣和长期参与的策略。本文提出利用大型语言模型（LLMs）对稀疏数据的显著规划能力用于长期推荐。关键在于使语言模型能够在个性化推荐场景中有效理解和应用任务解决原则，因为模型的预训练可能并未自然包含这些内容。

    arXiv:2403.00843v1 Announce Type: cross  Abstract: Traditional recommendation setting tends to excessively cater to users' immediate interests and neglect their long-term engagement. To address it, it is crucial to incorporate planning capabilities into the recommendation decision-making process to develop policies that take into account both immediate interests and long-term engagement. Despite Reinforcement Learning (RL) can learn planning capacity by maximizing cumulative reward, the scarcity of recommendation data presents challenges such as instability and susceptibility to overfitting when training RL models from scratch.   In this context, we propose to leverage the remarkable planning capabilities over sparse data of Large Language Models (LLMs) for long-term recommendation. The key lies in enabling a language model to understand and apply task-solving principles effectively in personalized recommendation scenarios, as the model's pre-training may not naturally encompass these 
    
[^6]: DEEM：面向立场检测的动态体验专家建模

    DEEM: Dynamic Experienced Expert Modeling for Stance Detection

    [https://arxiv.org/abs/2402.15264](https://arxiv.org/abs/2402.15264)

    本文提出了一种Dynamic Experienced Expert Modeling（DEEM）方法，利用生成的经验专家使LLMs能够以半参数化方式进行推理，提高了在立场检测任务中的性能。

    

    最近的研究初步尝试使用大型语言模型（LLMs）来解决立场检测任务，展现了有希望的结果。然而，考虑到立场检测通常需要详细的背景知识，传统的推理方法可能会忽视领域知识，以进行专业和准确的分析。因此，LLMs的推理仍有改进空间，尤其在利用LLMs的生成能力模拟特定专家（即多智能体）来检测立场方面。与现有需要详细描述并使用固定专家的多智能体作品不同，本文提出了一种Dynamic Experienced Expert Modeling（DEEM）方法，可以利用生成的经验专家，并让LLMs以半参数化方式进行推理，使专家更具普适性和可靠性。实验结果表明，DEEM在三个场景上一直达到最佳结果。

    arXiv:2402.15264v1 Announce Type: new  Abstract: Recent work has made a preliminary attempt to use large language models (LLMs) to solve the stance detection task, showing promising results. However, considering that stance detection usually requires detailed background knowledge, the vanilla reasoning method may neglect the domain knowledge to make a professional and accurate analysis. Thus, there is still room for improvement of LLMs reasoning, especially in leveraging the generation capability of LLMs to simulate specific experts (i.e., multi-agents) to detect the stance. In this paper, different from existing multi-agent works that require detailed descriptions and use fixed experts, we propose a Dynamic Experienced Expert Modeling (DEEM) method which can leverage the generated experienced experts and let LLMs reason in a semi-parametric way, making the experts more generalizable and reliable. Experimental results demonstrate that DEEM consistently achieves the best results on thre
    
[^7]: 微调、提示、上下文学习和指导微调：我们需要多少标记样本？

    Fine-Tuning, Prompting, In-Context Learning and Instruction-Tuning: How Many Labelled Samples Do We Need?

    [https://arxiv.org/abs/2402.12819](https://arxiv.org/abs/2402.12819)

    专门模型通常只需少量标记样本（100-1000个）就能与通用模型持平甚至更好，取决于任务的复杂性和结果的变化。

    

    当解决具有有限标记数据的任务时，研究人员可以选择使用通用的大型语言模型而不进行进一步更新，或者使用少量示例来调整专门的较小模型。 当有足够的标记可用时，专门的模型在许多自然语言处理任务上表现优于通用模型。 在这项工作中，我们旨在调查专门模型需要多少标记样本才能实现这种出色的性能，同时考虑结果的变化。观察提示、上下文学习、微调和指导微调的行为，识别它们在增加不同复杂性任务的标记训练样本数量时的收支平衡点，我们发现专门模型通常只需少量样本（100-1000个）就能与通用模型持平甚至更好。 同时，所需的标记数据量强烈依赖于任务的复杂性和结果的变化。

    arXiv:2402.12819v1 Announce Type: cross  Abstract: When solving a task with limited labelled data, researchers can either use a general large language model without further update, or use the few examples to tune a specialised smaller model. When enough labels are available, the specialised models outperform the general ones on many NLP tasks. In this work, we aim to investigate how many labelled samples are required for the specialised models to achieve this superior performance, while taking the results variance into consideration. Observing the behaviour of prompting, in-context learning, fine-tuning and instruction-tuning, identifying their break-even points when increasing number of labelled training samples across three tasks of varying complexity, we find that the specialised models often need only few samples ($100-1000$) to be on par or better than the general ones. At the same time, the amount of required labelled data strongly depends on the task complexity and results varia
    
[^8]: 增强Amharic-LLaMA: 整合特定任务与生成数据集

    Enhancing Amharic-LLaMA: Integrating Task Specific and Generative Datasets

    [https://arxiv.org/abs/2402.08015](https://arxiv.org/abs/2402.08015)

    本研究通过整合任务特定和生成数据集来增强Amharic-LLaMA模型，提高了阿姆哈拉语言模型的性能。他们通过创建阿姆哈拉语指令微调数据集和微调模型，在不同的NLP任务中取得了有希望的结果。

    

    大型语言模型（LLM）因其在理解和生成人类语言方面的出色表现而在自然语言处理（NLP）研究中受到了很多关注。然而，资源匮乏的语言因缺乏资源而被落下。在这项工作中，我们致力于通过整合特定任务和生成数据集来增强LLaMA-2-Amharic模型，以提高阿姆哈拉语的语言模型性能。我们创建了一个阿姆哈拉语指令微调数据集，并对LLaMA-2-Amharic模型进行了微调。经过微调的模型在不同的NLP任务中表现出有希望的结果。我们开源了我们的数据集创建流程、指令数据集、训练模型和评估输出，以促进对这些模型的语言特定研究。

    Large language models (LLMs) have received a lot of attention in natural language processing (NLP) research because of their exceptional performance in understanding and generating human languages. However, low-resource languages are left behind due to the unavailability of resources. In this work, we focus on enhancing the LLaMA-2-Amharic model by integrating task-specific and generative datasets to improve language model performance for Amharic. We compile an Amharic instruction fine-tuning dataset and fine-tuned LLaMA-2-Amharic model. The fine-tuned model shows promising results in different NLP tasks. We open-source our dataset creation pipeline, instruction datasets, trained models, and evaluation outputs to promote language-specific studies on these models.
    
[^9]: 通过语言预测足球比赛事件

    Forecasting Events in Soccer Matches Through Language

    [https://arxiv.org/abs/2402.06820](https://arxiv.org/abs/2402.06820)

    本文提出了一种使用语言模型预测足球比赛中下一个事件的方法，该方法受到大型语言模型方法的启发。通过深度学习和WyScout数据集，该方法在预测准确性方面明显超过了以往的方法。该方法的应用包括博彩和比赛分析，并提供了一个模拟骨架用于构建分析流水线。

    

    本文介绍了一种预测足球比赛中下一个事件的方法，这是一个与大型语言模型（LLMs）面临的问题非常相似的挑战。与其他严重限制足球事件动态的方法不同，这些方法往往从很多变量中抽象出来或依赖于混合顺序模型，我们的研究提出了一种受到LLMs方法学启发的新技术。这些模型预测了组成一个事件的完整变量链，大大简化了构建足球大事件模型（LEMs）的过程。利用公开可用的WyScout数据集进行深度学习，所提出的方法在关键领域（如下一个事件类型的预测准确性）显著超越了以往LEM提案的性能。本文突显了LEM在多种应用中的实用性，包括博彩和比赛分析。此外，我们还展示了LEM提供了一个模拟骨架，可以构建许多分析流水线。

    This paper introduces an approach to predicting the next event in a soccer match, a challenge bearing remarkable similarities to the problem faced by Large Language Models (LLMs). Unlike other methods that severely limit event dynamics in soccer, often abstracting from many variables or relying on a mix of sequential models, our research proposes a novel technique inspired by the methodologies used in LLMs. These models predict a complete chain of variables that compose an event, significantly simplifying the construction of Large Event Models (LEMs) for soccer. Utilizing deep learning on the publicly available WyScout dataset, the proposed approach notably surpasses the performance of previous LEM proposals in critical areas, such as the prediction accuracy of the next event type. This paper highlights the utility of LEMs in various applications, including betting and match analytics. Moreover, we show that LEMs provide a simulation backbone on which many analytics pipelines can be bu
    
[^10]: PythonSaga：重新定义评估代码生成LLM的基准

    PythonSaga: Redefining the Benchmark to Evaluate Code Generating LLM

    [https://arxiv.org/abs/2401.03855](https://arxiv.org/abs/2401.03855)

    PythonSaga提出了一种新的基准，针对Python代码生成进行评估,弥补了现有基准存在的编程概念偏见和简单任务普遍性的问题

    

    受到使用大型语言模型(LLMs)生成代码激增的推动，出现了许多基准用于评估这些LLMs的功能。我们对HumanEval和MBPP两个流行的Python代码生成基准进行了大规模人工评估，分析了它们的多样性和难度。我们的研究揭示了对一组有限的编程概念存在严重偏见，完全忽视了大多数其他概念。此外，我们发现了大量简单任务的普遍存在，可能夸大了模型性能的估计。为了解决这些限制，我们提出了一种新颖的基准，PythonSaga，包含了185个手工制作的提示，涵盖了38个不同难度级别的编程概念。

    arXiv:2401.03855v2 Announce Type: replace-cross  Abstract: Driven by the surge in code generation using large language models (LLMs), numerous benchmarks have emerged to evaluate these LLMs capabilities. We conducted a large-scale human evaluation of HumanEval and MBPP, two popular benchmarks for Python code generation, analyzing their diversity and difficulty. Our findings unveil a critical bias towards a limited set of programming concepts, neglecting most of the other concepts entirely. Furthermore, we uncover a worrying prevalence of easy tasks, potentially inflating model performance estimations. To address these limitations, we propose a novel benchmark, PythonSaga, featuring 185 hand-crafted prompts on a balanced representation of 38 programming concepts across diverse difficulty levels.
    
[^11]: 基于批处理的基础模型低秩调整

    Batched Low-Rank Adaptation of Foundation Models

    [https://arxiv.org/abs/2312.05677](https://arxiv.org/abs/2312.05677)

    提出了Fast LoRA（FLoRA）框架，使得基础模型的低秩调整可以高效批处理异构请求，并在绩效上保持竞争性。

    

    最近，低秩适应（LoRA）因通过引入可训练的低秩矩阵微调基础模型并减少可训练参数的数量而引起关注。虽然LoRA提供了许多优点，但其在实时为各种全球用户提供服务时无法高效处理多个特定任务适配器的能力受到限制。这为需要为每个传入请求个性化、特定任务适应的场景中造成了性能瓶颈。为了减轻这一约束，我们提出了快速LoRA（FLoRA）框架，其中批处理中的每个输入示例都可以与其独特的低秩适应权重相关联，从而实现对异构请求的高效批处理。我们通过实证表明，FLoRA保留了LoRA的绩效优点，在跨越8种语言的MultiPL-E代码生成基准测试上展示出竞争结果。

    arXiv:2312.05677v2 Announce Type: replace-cross  Abstract: Low-Rank Adaptation (LoRA) has recently gained attention for fine-tuning foundation models by incorporating trainable low-rank matrices, thereby reducing the number of trainable parameters. While LoRA offers numerous advantages, its applicability for real-time serving to a diverse and global user base is constrained by its incapability to handle multiple task-specific adapters efficiently. This imposes a performance bottleneck in scenarios requiring personalized, task-specific adaptations for each incoming request. To mitigate this constraint, we introduce Fast LoRA (FLoRA), a framework in which each input example in a minibatch can be associated with its unique low-rank adaptation weights, allowing for efficient batching of heterogeneous requests. We empirically demonstrate that FLoRA retains the performance merits of LoRA, showcasing competitive results on the MultiPL-E code generation benchmark spanning over 8 languages and 
    
[^12]: MAIRA-1：一种专门用于放射学报告生成的大型多模态模型

    MAIRA-1: A specialised large multimodal model for radiology report generation

    [https://arxiv.org/abs/2311.13668](https://arxiv.org/abs/2311.13668)

    MAIRA-1是一种专门用于放射学报告生成的大型多模态模型，在与预训练的视觉编码器对齐和文本数据增强的基础上，利用了CXR特定的图像编码器和经过微调的大型语言模型，生成具有最先进质量的报告。

    

    我们提出了一种放射学特定的多模态模型，用于从胸部X光（CXR）生成放射学报告的任务。我们的工作基于一个思想，即可以通过与预训练视觉编码器对齐，使大型语言模型具备多模态能力。在自然图像上，这已被证明可以使多模态模型获得图像理解和描述能力。我们提出的模型（MAIRA-1）利用了一个CXR特定的图像编码器，结合基于Vicuna-7B的微调的大型语言模型，并进行基于文本的数据增强，以产生具有最先进质量的报告。特别地，MAIRA-1在与放射科医生对齐的RadCliQ度量和考虑的所有词汇度量上都有显著改进。对模型输出的手动审核显示出了产生报告的流畅性和准确性，同时揭示了现有评估方法所未捕捉到的失败模式。更多信息和资源可在项目网站上找到：

    We present a radiology-specific multimodal model for the task for generating radiological reports from chest X-rays (CXRs). Our work builds on the idea that large language model(s) can be equipped with multimodal capabilities through alignment with pre-trained vision encoders. On natural images, this has been shown to allow multimodal models to gain image understanding and description capabilities. Our proposed model (MAIRA-1) leverages a CXR-specific image encoder in conjunction with a fine-tuned large language model based on Vicuna-7B, and text-based data augmentation, to produce reports with state-of-the-art quality. In particular, MAIRA-1 significantly improves on the radiologist-aligned RadCliQ metric and across all lexical metrics considered. Manual review of model outputs demonstrates promising fluency and accuracy of generated reports while uncovering failure modes not captured by existing evaluation practices. More information and resources can be found on the project website:
    
[^13]: 基于Transformer的描述逻辑语境推理

    Reasoning over Description Logic-based Contexts with Transformers

    [https://arxiv.org/abs/2311.08941](https://arxiv.org/abs/2311.08941)

    本研究构建了一个由描述逻辑知识库生成的合成自然语言问答数据集，以评估基于Transformer模型在丰富语境中的推理能力。

    

    目前，衡量基于Transformer模型的推理能力的一种方式是通过评估在自然语言表达的合成语境中对逻辑问题回答或证明生成等下游任务的准确性。然而，大多数实际使用的语境非常简单；在大多数情况下，它们是由仅含有少量逻辑运算符和量词的短一阶逻辑句子生成的。本文旨在回答基于Transformer模型能够在表达丰富语境中执行推理的问题。为此，我们构建了一个由描述逻辑知识库生成的合成自然语言问答数据集。为生成知识库，我们使用了表达式语言$\mathcal{ALCQ$。生成的数据集包含384K个示例，并且在两个维度上增加：i) 推理深度，和ii) 句子长度。

    arXiv:2311.08941v2 Announce Type: replace-cross  Abstract: One way that the current state of the art measures the reasoning ability of transformer-based models is by evaluating accuracy in downstream tasks like logical question answering or proof generation over synthetic contexts expressed in natural language. However, most of the contexts used are in practice very simple; in most cases, they are generated from short first-order logic sentences with only a few logical operators and quantifiers. In this work, we seek to answer the question how well a transformer-based model will perform reasoning over expressive contexts. For this purpose, we construct a synthetic natural language question-answering dataset, generated by description logic knowledge bases. For the generation of the knowledge bases, we use the expressive language $\mathcal{ALCQ}$. The resulting dataset contains 384K examples, and increases in two dimensions: i) reasoning depth, and ii) length of sentences. We show that t
    
[^14]: 人类与ChatGPT生成对话之间的语言对比

    A Linguistic Comparison between Human and ChatGPT-Generated Conversations. (arXiv:2401.16587v1 [cs.CL])

    [http://arxiv.org/abs/2401.16587](http://arxiv.org/abs/2401.16587)

    本研究比较了人类和ChatGPT生成的对话的语言差异，发现ChatGPT在社交、分析、认知、关注焦点和积极情绪等方面表现出色，但人类对话更具变异性和真实性，尽管在情绪方面无显著差异。同时，该研究还提供了一个新颖的、由ChatGPT生成的对话组成的数据集。

    

    本研究探讨了人类和LLM生成的对话之间的语言差异，使用了由ChatGPT-3.5生成的19.5K个对话作为EmpathicDialogues数据集的补充。研究采用Linguistic Inquiry and Word Count (LIWC) 分析，比较了ChatGPT生成的对话和人类对话在118个语言类别上的差异。结果显示人类对话具有更大的变异性和真实性，但ChatGPT在社交过程、分析风格、认知、关注焦点和积极情绪色彩等方面表现出色，这进一步证明了LLMs“比真人更像真人”的最新发现。然而，在ChatGPT和人类对话之间没有找到积极或消极情绪的显著差异。对话嵌入的分类器分析表明，尽管对话中没有明确提及情绪，但对情感价值的隐性编码存在。研究还提供了一个新颖的、由两个ChatGPT生成的对话组成的数据集。

    This study explores linguistic differences between human and LLM-generated dialogues, using 19.5K dialogues generated by ChatGPT-3.5 as a companion to the EmpathicDialogues dataset. The research employs Linguistic Inquiry and Word Count (LIWC) analysis, comparing ChatGPT-generated conversations with human conversations across 118 linguistic categories. Results show greater variability and authenticity in human dialogues, but ChatGPT excels in categories such as social processes, analytical style, cognition, attentional focus, and positive emotional tone, reinforcing recent findings of LLMs being "more human than human." However, no significant difference was found in positive or negative affect between ChatGPT and human dialogues. Classifier analysis of dialogue embeddings indicates implicit coding of the valence of affect despite no explicit mention of affect in the conversations. The research also contributes a novel, companion ChatGPT-generated dataset of conversations between two i
    
[^15]: 使用大型多模型的弱监督高斯对比基础模型来处理视频问答问题

    Weakly Supervised Gaussian Contrastive Grounding with Large Multimodal Models for Video Question Answering. (arXiv:2401.10711v1 [cs.CV])

    [http://arxiv.org/abs/2401.10711](http://arxiv.org/abs/2401.10711)

    本论文提出了一种使用大型多模型的弱监督高斯对比基础模型来处理视频问答问题的方法。通过将问题和答案对作为事件描述，找到多个关键帧作为目标时刻，并利用这些时刻作为伪标签来强制LMMs进行推理。所提出的方法使用轻量级的基于高斯的对比基础模块（GCG）来学习时效结构。

    

    视频问答（VideoQA）旨在基于观察到的视频信息回答自然语言问题。尽管大型多模型（LMMs）在图像语言理解和推理方面取得了近期的成功，但它们在处理视频问答方面还不足够，仅仅是将均匀采样的帧作为视觉输入，忽略了与问题相关的视觉线索。此外，现有的视频问答数据集中没有针对问题关键时间戳的人工注释。基于此，我们提出了一种新的弱监督框架，强制LMMs使用问题关键时刻作为视觉输入推理出答案。具体来说，我们将问题和答案对合并为事件描述，以找到多个关键帧作为目标时刻，这些时刻将作为伪标签。通过将这些伪标签作为额外的弱监督，我们设计了一个轻量级的基于高斯的对比基础模块（GCG）。GCG学习多个高斯函数来描述时效结构。

    Video Question Answering (VideoQA) aims to answer natural language questions based on the information observed in videos. Despite the recent success of Large Multimodal Models (LMMs) in image-language understanding and reasoning, they deal with VideoQA insufficiently by simply taking uniformly sampled frames as visual inputs, which ignores question-relevant visual clues. Moreover, there are no human annotations for question-critical timestamps in existing VideoQA datasets. In light of this, we propose a novel weakly supervised framework to enforce the LMMs to reason out the answers with question-critical moments as visual inputs. Specifically, we fuse the question and answer pairs as event descriptions to find multiple keyframes as target moments, which will be pseudo-labels. With these pseudo-labels as additionally weak supervision, we devise a lightweight Gaussian-based Contrastive Grounding (GCG) module. GCG learns multiple Gaussian functions to characterize the temporal structure o
    
[^16]: 将大型语言模型应用于教育：基本能力、潜力和挑战

    Adapting Large Language Models for Education: Foundational Capabilities, Potentials, and Challenges. (arXiv:2401.08664v1 [cs.AI])

    [http://arxiv.org/abs/2401.08664](http://arxiv.org/abs/2401.08664)

    本文回顾了针对教育能力的大型语言模型研究，包括数学、写作、编程、推理和基于知识的问答，旨在探索其在构建下一代智能教育系统中的潜力和挑战。

    

    在线教育平台利用互联网分发教育资源，旨在提供便捷的教育，但往往在与学生的实时交流方面存在不足。由于需要解决学生在学习过程中遇到的多样化障碍的挑战，它们经常难以提供个性化的教育资源。最近出现的大型语言模型（LLMs），如ChatGPT，提供了通过理解个体请求解决这一问题的可能性。虽然LLMs在各个领域都取得了成功，但基于LLM的教育系统的构建仍然面临着广泛的教育技能要求。本文回顾了与教育能力相关的近期出现的LLM研究，包括数学、写作、编程、推理和基于知识的问答，旨在探索它们在构建下一代智能教育系统方面的潜力。

    Online education platforms, leveraging the internet to distribute education resources, seek to provide convenient education but often fall short in real-time communication with students. They often struggle to offer personalized education resources due to the challenge of addressing the diverse obstacles students encounter throughout their learning journey. Recently, the emergence of large language models (LLMs), such as ChatGPT, offers the possibility for resolving this issue by comprehending individual requests. Although LLMs have been successful in various fields, creating an LLM-based education system is still challenging for the wide range of educational skills required. This paper reviews the recently emerged LLM researches related to educational capabilities, including mathematics, writing, programming, reasoning, and knowledge-based question answering, with the aim to explore their potential in constructing the next-generation intelligent education system. Based on the current 
    
[^17]: Stack Overflow回答中信息高亮的初探

    A First Look at Information Highlighting in Stack Overflow Answers. (arXiv:2401.01472v1 [cs.CL])

    [http://arxiv.org/abs/2401.01472](http://arxiv.org/abs/2401.01472)

    本论文进行了首次大规模的探索性研究，研究了Stack Overflow回答中的信息高亮。通过使用神经网络架构，开发了自动推荐突出内容的方法。

    

    背景：浏览Stack Overflow（SO）的知识仍然具有挑战性。为了使帖子对用户更生动，SO允许用户使用Markdown或HTML编写和编辑帖子，以便用户可以利用各种格式化样式（例如粗体、斜体和代码）来突出重要信息。然而，关于突出信息的研究仍然有限。目标：我们在最近的研究中进行了首次大规模的探索性研究，研究了SO回答中的信息高亮。为了扩展我们之前的研究，我们利用最初设计用于命名实体识别任务的神经网络架构，开发了自动推荐带有格式化样式的突出内容的方法。方法：本文研究了Stack Overflow的31,169,429个回答。为了训练推荐模型，我们选择了CNN和BERT模型，针对每种格式化类型（即粗体、斜体、代码和标题）使用我们从SO回答收集的突出信息数据集。

    Context: Navigating the knowledge of Stack Overflow (SO) remains challenging. To make the posts vivid to users, SO allows users to write and edit posts with Markdown or HTML so that users can leverage various formatting styles (e.g., bold, italic, and code) to highlight the important information. Nonetheless, there have been limited studies on the highlighted information. Objective: We carried out the first large-scale exploratory study on the information highlighted in SO answers in our recent study. To extend our previous study, we develop approaches to automatically recommend highlighted content with formatting styles using neural network architectures initially designed for the Named Entity Recognition task. Method: In this paper, we studied 31,169,429 answers of Stack Overflow. For training recommendation models, we choose CNN and BERT models for each type of formatting (i.e., Bold, Italic, Code, and Heading) using the information highlighting dataset we collected from SO answers.
    
[^18]: LLM训练中的结构化填充改进了长上下文利用

    Structured Packing in LLM Training Improves Long Context Utilization. (arXiv:2312.17296v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2312.17296](http://arxiv.org/abs/2312.17296)

    本论文研究了长上下文大型语言模型（LLM）中上下文利用不足的问题，并通过将相关文档纳入训练示例中来改进模型的困惑度。通过引入Structured Packing for Long Context (SPLiCe)方法，使用检索方法将最互相关文档汇集到单个训练上下文中，进一步提高了模型的性能。

    

    长上下文大型语言模型（LCLM）的最新进展引起了广泛关注，特别是在查询科学研究论文等应用中。然而，它们的潜力往往受到上下文利用不足的限制。我们确定典型训练数据中缺乏长程语义依赖是主要障碍。为了解决这个问题，我们深入研究了频繁将相关文档纳入训练输入的好处。利用代码数据的固有目录结构作为训练示例的来源，我们证明了即使对于与编码无关的任务，囊括相关文档能够改进模型的困惑度。基于这些发现，并且更具广泛的关注，我们引入了一种名为Structured Packing for Long Context (SPLiCe)的创新方法。 SPLiCe是一种使用检索方法将最互相关文档汇集到单个训练上下文中的方法。我们的结果表明，\method{}提高了模型的性能，并可用于t

    Recent advances in long-context Large Language Models (LCLMs) have generated significant interest, especially in applications such as querying scientific research papers. However, their potential is often limited by inadequate context utilization. We identify the absence of long-range semantic dependencies in typical training data as a primary hindrance. To address this, we delve into the benefits of frequently incorporating related documents into training inputs. Using the inherent directory structure of code data as a source of training examples, we demonstrate improvements in perplexity, even for tasks unrelated to coding. Building on these findings, but with a broader focus, we introduce Structured Packing for Long Context (SPLiCe). SPLiCe is an innovative method for creating training examples by using a retrieval method to collate the most mutually relevant documents into a single training context. Our results indicate that \method{} enhances model performance and can be used to t
    
[^19]: Kosmos-G：使用多模态大型语言模型在上下文中生成图像

    Kosmos-G: Generating Images in Context with Multimodal Large Language Models. (arXiv:2310.02992v1 [cs.CV])

    [http://arxiv.org/abs/2310.02992](http://arxiv.org/abs/2310.02992)

    本文介绍了Kosmos-G，一种利用多模态大型语言模型（MLLM）在上下文中生成图像的模型。该模型通过使用文本模态作为锚点，将MLLM的输出空间与CLIP对齐，并进行组合指令调整。Kosmos-G展示了零样本多实体主题驱动生成的独特能力。

    

    最近，在文本到图像（T2I）和视觉语言到图像（VL2I）生成方面取得了显著进展。然而，从通用的视觉语言输入生成图像，特别是涉及多个图像的情况，仍然未被充分探索。本文提出了Kosmos-G，该模型利用多模态大型语言模型（MLLMs）的先进感知能力来解决上述挑战。我们的方法通过使用文本模态作为锚点，将MLLM的输出空间与CLIP对齐，并在策划数据上进行组合指令调整。Kosmos-G展示了零样本多实体主题驱动生成的独特能力。值得注意的是，分数蒸馏指令调整对图像解码器不需要进行任何修改。这允许无缝替代CLIP并轻松集成各种U-Net技术，包括细粒度控制和个性化图像解码器变体。

    Recent advancements in text-to-image (T2I) and vision-language-to-image (VL2I) generation have made significant strides. However, the generation from generalized vision-language inputs, especially involving multiple images, remains under-explored. This paper presents Kosmos-G, a model that leverages the advanced perception capabilities of Multimodal Large Language Models (MLLMs) to tackle the aforementioned challenge. Our approach aligns the output space of MLLM with CLIP using the textual modality as an anchor and performs compositional instruction tuning on curated data. Kosmos-G demonstrates a unique capability of zero-shot multi-entity subject-driven generation. Notably, the score distillation instruction tuning requires no modifications to the image decoder. This allows for a seamless substitution of CLIP and effortless integration with a myriad of U-Net techniques ranging from fine-grained controls to personalized image decoder variants. We posit Kosmos-G as an initial attempt to
    
[^20]: 无监督句法分析的集成蒸馏

    Ensemble Distillation for Unsupervised Constituency Parsing. (arXiv:2310.01717v1 [cs.CL])

    [http://arxiv.org/abs/2310.01717](http://arxiv.org/abs/2310.01717)

    本论文提出了一种集成蒸馏的方法来提高无监督句法解析的性能，并且通过蒸馏将集成知识转移到一个学生模型中，解决了常见的多教师蒸馏方法中的过度平滑问题。

    

    我们研究了无监督句法分析任务，该任务将句子的词和短语组织成一个层次结构，而不使用语言学注释的数据。我们观察到现有的无监督解析器捕捉到了解析结构的不同方面，可以利用这些来提高无监督分析的性能。为此，我们提出了“树平均”的概念，基于此我们进一步提出了一种新的无监督解析的集成方法。为了提高推理效率，我们进一步将集成知识蒸馏到一个学生模型中；这种集成-蒸馏的过程是缓解常见的多教师蒸馏方法中存在的过度平滑问题的有效方法。实验证明我们的方法超过了所有先前的方法，始终表现出其在不同集成组件和领域转移条件下的有效性和稳健性。

    We investigate the unsupervised constituency parsing task, which organizes words and phrases of a sentence into a hierarchical structure without using linguistically annotated data. We observe that existing unsupervised parsers capture differing aspects of parsing structures, which can be leveraged to enhance unsupervised parsing performance. To this end, we propose a notion of "tree averaging," based on which we further propose a novel ensemble method for unsupervised parsing. To improve inference efficiency, we further distill the ensemble knowledge into a student model; such an ensemble-then-distill process is an effective approach to mitigate the over-smoothing problem existing in common multi-teacher distilling methods. Experiments show that our method surpasses all previous approaches, consistently demonstrating its effectiveness and robustness across various runs, with different ensemble components, and under domain-shift conditions.
    
[^21]: 探索对比演示和显著性图在上下文学习中的作用

    Towards Understanding In-Context Learning with Contrastive Demonstrations and Saliency Maps. (arXiv:2307.05052v1 [cs.CL])

    [http://arxiv.org/abs/2307.05052](http://arxiv.org/abs/2307.05052)

    本研究探索了对比演示和显著性图在上下文学习中的作用，并发现改变标签对显著性有显著影响，尤其对于更大的语言模型更为明显。在情感分析任务中，将表达情感的术语改为中性词并不像改变标签那样具有显著影响。另外，补充解释在提高上下文学习方面是有效的。

    

    本文研究了在大型语言模型的上下文学习(ICL)性能中，各种演示组件的作用。具体而言，我们探讨了标签、输入分布和补充解释等因素的影响，特别是在这些因素被修改或扰动时的影响。我们基于之前的工作，这些工作对于这些元素如何影响ICL给出了不一致的结果。为了探究这些问题，我们采用了可解释的自然语言处理(XNLP)方法，并利用对比演示的显著性图进行定性和定量分析。我们的研究结果表明，改变标签对显著性有显著影响，尤其对于更大的语言模型更为明显。我们对输入分布进行了粒度级别的分析，发现在情感分析任务中，将表达情感的术语改为中性词并不像改变标签那样具有显著影响。最后，我们发现补充解释在提高ICL方面的效果是存在的。

    We investigate the role of various demonstration components in the in-context learning (ICL) performance of large language models (LLMs). Specifically, we explore the impacts of ground-truth labels, input distribution, and complementary explanations, particularly when these are altered or perturbed. We build on previous work, which offers mixed findings on how these elements influence ICL. To probe these questions, we employ explainable NLP (XNLP) methods and utilize saliency maps of contrastive demonstrations for both qualitative and quantitative analysis. Our findings reveal that flipping ground-truth labels significantly affects the saliency, though it's more noticeable in larger LLMs. Our analysis of the input distribution at a granular level reveals that changing sentiment-indicative terms in a sentiment analysis task to neutral ones does not have as substantial an impact as altering ground-truth labels. Finally, we find that the effectiveness of complementary explanations in boos
    
[^22]: 论自然语言处理中的偏见和公平：如何构建更公正的文本分类？

    On Bias and Fairness in NLP: How to have a fairer text classification?. (arXiv:2305.12829v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2305.12829](http://arxiv.org/abs/2305.12829)

    本文从上游偏见、样本偏见和过度放大偏见三方面分析了NLP模型中的偏见如何影响文本分类的公平性，并针对过度放大偏见通过微调语言模型达到公平分类效果。提出了构建公正文本分类模型的实用指南。

    

    本文全面分析了自然语言处理模型中不同来源的偏见，即上游偏见、样本偏见和过度放大偏见，以及它们对文本分类任务公平性的影响。我们还研究了使用不同去偏方法消除这些偏见对文本分类公平性的影响。研究发现过度放大偏见对文本分类公平性的影响最大。将语言模型在平衡不同类别身份群体的数据集上进行微调，可以去除过度放大偏见，进而构建更公正的文本分类模型。最后，我们基于研究发现提出了构建更公正的文本分类模型的实用指南。

    In this paper, we provide a holistic analysis of the different sources of bias, Upstream, Sample and Overampflication biases, in NLP models. We investigate how they impact the fairness of the task of text classification. We also investigate the impact of removing these biases using different debiasing techniques on the fairness of text classification. We found that overamplification bias is the most impactful bias on the fairness of text classification. And that removing overamplification bias by fine-tuning the LM models on a dataset with balanced representations of the different identity groups leads to fairer text classification models. Finally, we build on our findings and introduce practical guidelines on how to have a fairer text classification model.
    
[^23]: 基于摘要描述的文本检索

    Retrieving Texts based on Abstract Descriptions. (arXiv:2305.12517v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2305.12517](http://arxiv.org/abs/2305.12517)

    本研究针对语义检索问题，提出了一种基于摘要描述的文本检索模型，通过改进当前的文本嵌入方法，在标准最近邻搜索中取得了显著性能提升。

    

    虽然针对文本的信息提取，指令优化的大型语言模型表现优异，但对于在大规模文档集合中定位符合给定描述的文本（语义检索）并不适用。基于嵌入向量的相似度搜索可以通过查询执行检索，但嵌入中的相似度定义不明确且不一致，并且对于许多用例来说都是次优的。那么，什么是有效检索的好的查询表示？我们确定了根据内容的摘要描述检索句子的明确定义且一致的任务。我们展示了当前文本嵌入的不足，并提出了一种替代模型，在标准最近邻搜索中的表现显著提升。该模型使用通过提示LLM获得的正负样本对进行训练。虽然很容易从LLM中获得训练材料，但LLM无法直接执行检索任务。

    While instruction-tuned Large Language Models (LLMs) excel at extracting information from text, they are not suitable for locating texts conforming to a given description in a large document collection (semantic retrieval). Similarity search over embedding vectors does allow to perform retrieval by query, but the similarity reflected in the embedding is ill-defined and non-consistent, and is sub-optimal for many use cases. What, then, is a good query representation for effective retrieval?  We identify the well defined and consistent task of retrieving sentences based on abstract descriptions of their content. We demonstrate the inadequacy of current text embeddings and propose an alternative model that significantly improves when used in standard nearest neighbor search. The model is trained using positive and negative pairs sourced through prompting a LLM. While it is easy to source the training material from an LLM, the retrieval task cannot be performed by the LLM directly. This de
    

