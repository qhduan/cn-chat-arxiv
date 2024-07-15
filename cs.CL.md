# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Can large language models explore in-context?](https://arxiv.org/abs/2403.15371) | 研究发现，大型语言模型在没有实质干预的情况下很难有效进行探索，除了特定配置下的GPT-4具有满意的探索行为外，其他模型表现不稳定。 |
| [^2] | [EnvGen: Generating and Adapting Environments via LLMs for Training Embodied Agents](https://arxiv.org/abs/2403.12014) | EnvGen提出了一种新的框架，利用LLMs的推理能力自适应创建训练环境，帮助小型具身体RL代理在弱点方面学习有用技能。 |
| [^3] | [Updating the Minimum Information about CLinical Artificial Intelligence (MI-CLAIM) checklist for generative modeling research](https://arxiv.org/abs/2403.02558) | 生成模型的最新进展加速了医学中自然语言和图像处理领域的发展，并标志着生物医学模型开发和部署方式的重大范式转变。 |
| [^4] | [Reading Subtext: Evaluating Large Language Models on Short Story Summarization with Writers](https://arxiv.org/abs/2403.01061) | 评估大型语言模型在短篇小说摘要上的表现，发现它们在忠实性和解释潜台词方面存在挑战，但在进行主题分析时表现出思考深度。 |
| [^5] | [Heavy-Tailed Class Imbalance and Why Adam Outperforms Gradient Descent on Language Models](https://arxiv.org/abs/2402.19449) | 研究发现语言模型中的重尾类别不平衡问题导致了优化动态上的困难，Adam和基于符号的方法在这种情况下优于梯度下降。 |
| [^6] | [A Neural Rewriting System to Solve Algorithmic Problems](https://arxiv.org/abs/2402.17407) | 提出了一种受重写系统启发的神经架构，用于学习算法任务，通过Selector、Solver和Combiner三个专门模块实现算法任务的简化，具有较好的外推能力 |
| [^7] | [Towards Unified Task Embeddings Across Multiple Models: Bridging the Gap for Prompt-Based Large Language Models and Beyond](https://arxiv.org/abs/2402.14522) | 提出了一种框架用于统一不同模型的任务嵌入，使得任务嵌入可以跨越各种模型，并在单一向量空间内进行比较和分析。 |
| [^8] | [STENCIL: Submodular Mutual Information Based Weak Supervision for Cold-Start Active Learning](https://arxiv.org/abs/2402.13468) | STENCIL利用次模互信息选择弱标记的稀有类实例，并通过标注者强标记，提高了文本分类数据集上的准确率和稀有类F-1分数。 |
| [^9] | [BlendFilter: Advancing Retrieval-Augmented Large Language Models via Query Generation Blending and Knowledge Filtering](https://arxiv.org/abs/2402.11129) | BlendFilter通过查询生成混合和知识过滤方法提升了检索增强型大型语言模型，在多领域的问答任务中取得了显著的性能提升。 |
| [^10] | [GPT-4 Generated Narratives of Life Events using a Structured Narrative Prompt: A Validation Study](https://arxiv.org/abs/2402.05435) | 本研究通过使用结构化叙事提示，验证了GPT-4生成的叙述在传达生活事件方面的有效性。研究结果表明，大多数叙述能够足够传达提示的意图。同时，通过机器学习模型的训练和验证，可以自动识别有效和无效的叙述。 |
| [^11] | [Enhancing Transformer RNNs with Multiple Temporal Perspectives](https://arxiv.org/abs/2402.02625) | 引入了多个时间视角的概念，用于增强Transformer RNNs对顺序数据的理解能力，在参数数量最小增加的情况下取得了显著的改进。 |
| [^12] | [Generative AI in Higher Education: Seeing ChatGPT Through Universities' Policies, Resources, and Guidelines](https://arxiv.org/abs/2312.05235) | 通过分析美国前100名大学制定的学术政策和指南，揭示了大多数大学对于在高等教育中整合生成人工智能的开放但谨慎态度，主要关注点在于伦理使用、准确性和数据隐私。 |
| [^13] | [Whose wife is it anyway? Assessing bias against same-gender relationships in machine translation.](http://arxiv.org/abs/2401.04972) | 本文研究了机器翻译系统对同性关系的偏见问题，发现三个受欢迎的MT服务在准确翻译涉及同性别名词之间关系的句子时存在较大的错误率，特别是在涉及女性职业的上下文中表现更差。这项工作为评估NLP系统中固有偏见提供了一个社会关系方面的案例研究。 |
| [^14] | [Zero-Shot Continuous Prompt Transfer: Generalizing Task Semantics Across Language Models.](http://arxiv.org/abs/2310.01691) | 这项工作提出了一种零样本连续提示传递方法，通过将源提示编码到相对空间中，并搜索相应的目标提示，在不同的语言模型之间实现了任务语义的泛化，实验证实了该方法的有效性。 |
| [^15] | [Basic syntax from speech: Spontaneous concatenation in unsupervised deep neural networks.](http://arxiv.org/abs/2305.01626) | 该论文提出了一种基于语音的完全无监督的方法，可以直接从原始语音中建立基础语法模型。作者发现，在基于声音的单词记录上训练的卷积神经网络可以自发连接两个或三个单词，并且可以学会将单词嵌入到新的未见过的单词组合中，这是之前未报道的属性，这一发现对我们理解神经网络的学习方式和建立从原始声学输入中的语法及其演化的模型都有重要的意义。 |

# 详细

[^1]: 大型语言模型能够进行上下文中的探索吗？

    Can large language models explore in-context?

    [https://arxiv.org/abs/2403.15371](https://arxiv.org/abs/2403.15371)

    研究发现，大型语言模型在没有实质干预的情况下很难有效进行探索，除了特定配置下的GPT-4具有满意的探索行为外，其他模型表现不稳定。

    

    我们研究现代大型语言模型（LLMs）在进行探索方面的能力，这是强化学习和决策制定中的核心能力。我们关注现有LLMs的原生性能，没有进行训练干预。我们将LLMs部署为简单多臂老虎机环境中的代理，并完全在上下文中指定环境描述和交互历史，即在LLM提示内部进行。我们使用各种提示设计对GPT-3.5、GPT-4和Llama2进行实验，发现这些模型在没有实质干预的情况下并没有稳健地进行探索：i）在我们的所有实验中，只有一个配置导致了令人满意的探索行为：具有思维链推理和外部总结的交互历史的GPT-4，这些被呈现为充分统计的情况；ii）所有其他配置都没有产生稳健的探索行为，包括具有思维链推理的其他配置。

    arXiv:2403.15371v1 Announce Type: cross  Abstract: We investigate the extent to which contemporary Large Language Models (LLMs) can engage in exploration, a core capability in reinforcement learning and decision making. We focus on native performance of existing LLMs, without training interventions. We deploy LLMs as agents in simple multi-armed bandit environments, specifying the environment description and interaction history entirely in-context, i.e., within the LLM prompt. We experiment with GPT-3.5, GPT-4, and Llama2, using a variety of prompt designs, and find that the models do not robustly engage in exploration without substantial interventions: i) Across all of our experiments, only one configuration resulted in satisfactory exploratory behavior: GPT-4 with chain-of-thought reasoning and an externally summarized interaction history, presented as sufficient statistics; ii) All other configurations did not result in robust exploratory behavior, including those with chain-of-thou
    
[^2]: EnvGen: 通过LLMs生成和调整环境以训练具身体的代理

    EnvGen: Generating and Adapting Environments via LLMs for Training Embodied Agents

    [https://arxiv.org/abs/2403.12014](https://arxiv.org/abs/2403.12014)

    EnvGen提出了一种新的框架，利用LLMs的推理能力自适应创建训练环境，帮助小型具身体RL代理在弱点方面学习有用技能。

    

    最近有关通过互动进行具身体学习的最新方法直接采用大型语言模型（LLMs）作为代理，以确定环境中的下一步。LLM代理由于其世界知识和推理能力，比基于强化学习（RL）的以往较小的代理表现更强；但频繁调用LLMs速度慢且昂贵。我们提出EnvGen，一个处理这个问题的新框架。首先，我们提示一个LLM生成训练环境，使代理可以快速并行学习不同任务。具体而言，LLM获得任务描述和模拟器目标，然后被要求生成一组环境配置。

    arXiv:2403.12014v1 Announce Type: cross  Abstract: Recent SOTA approaches for embodied learning via interaction directly employ large language models (LLMs) as agents to determine the next steps in an environment. Due to their world knowledge and reasoning capabilities, LLM agents achieve stronger performance than previous smaller agents based on reinforcement learning (RL); however, frequently calling LLMs is slow and expensive. Instead of directly employing LLMs as agents, can we use LLMs' reasoning capabilities to adaptively create training environments to help smaller embodied RL agents learn useful skills that they are weak at? We propose EnvGen, a novel framework to address this question. First, we prompt an LLM to generate training environments that allow agents to quickly learn different tasks in parallel. Concretely, the LLM is given the task description and simulator objectives that the agents should learn and is then asked to generate a set of environment configurations (e.g
    
[^3]: 为生成建模研究更新有关临床人工智能（MI-CLAIM）检查表

    Updating the Minimum Information about CLinical Artificial Intelligence (MI-CLAIM) checklist for generative modeling research

    [https://arxiv.org/abs/2403.02558](https://arxiv.org/abs/2403.02558)

    生成模型的最新进展加速了医学中自然语言和图像处理领域的发展，并标志着生物医学模型开发和部署方式的重大范式转变。

    

    生成模型的最新进展，包括大型语言模型（LLMs）、视觉语言模型（VLMs）和扩散模型，加速了医学中自然语言和图像处理领域的发展，并标志着生物医学模型开发和部署方式的重大范式转变。尽管这些模型非常适应新任务，但在扩展和评估它们的使用过程中出现了前人框架未解决的新挑战。特别是，这些模型以少量或无需专门训练数据即可产生有用输出的能力（“零样本”或“少样本”方法），以及它们输出的开放性质，需要制定更新的使用和评估这些模型的指南。美国行政命令141103确定了有关临床人工智能工具开发的标准和最佳实践存在的差距，以及几个新兴国家临床人工智能评估网络。

    arXiv:2403.02558v1 Announce Type: new  Abstract: Recent advances in generative models, including large language models (LLMs), vision language models (VLMs), and diffusion models, have accelerated the field of natural language and image processing in medicine and marked a significant paradigm shift in how biomedical models can be developed and deployed. While these models are highly adaptable to new tasks, scaling and evaluating their usage presents new challenges not addressed in previous frameworks. In particular, the ability of these models to produce useful outputs with little to no specialized training data ("zero-" or "few-shot" approaches), as well as the open-ended nature of their outputs, necessitate the development of updated guidelines in using and evaluating these models. In response to gaps in standards and best practices for the development of clinical AI tools identified by US Executive Order 141103 and several emerging national networks for clinical AI evaluation, we be
    
[^4]: 阅读潜台词：在短篇小说摘要上评估大型语言模型与作者合作

    Reading Subtext: Evaluating Large Language Models on Short Story Summarization with Writers

    [https://arxiv.org/abs/2403.01061](https://arxiv.org/abs/2403.01061)

    评估大型语言模型在短篇小说摘要上的表现，发现它们在忠实性和解释潜台词方面存在挑战，但在进行主题分析时表现出思考深度。

    

    我们评估了最近的大型语言模型（LLMs）在摘要长篇文学作品这一具有挑战性的任务上的表现，这些作品可能长度较长，并包含微妙的潜台词或错综复杂的时间线。重要的是，我们直接与作者合作，确保这些作品尚未在网络上分享过（因此对这些模型是未知的），并获得作者本人对摘要质量的明确评价。通过基于叙事理论的定量和定性分析，我们比较了GPT-4、Claude-2.1和LLama-2-70B。我们发现这三个模型在50%以上的摘要中会出现忠实性错误，并且难以解释难以理解的潜台词。然而，在最佳状态下，这些模型可以对故事进行有深度的主题分析。此外，我们还展示了LLMs对摘要质量的判断与作家的反馈不一致。

    arXiv:2403.01061v1 Announce Type: new  Abstract: We evaluate recent Large language Models (LLMs) on the challenging task of summarizing short stories, which can be lengthy, and include nuanced subtext or scrambled timelines. Importantly, we work directly with authors to ensure that the stories have not been shared online (and therefore are unseen by the models), and to obtain informed evaluations of summary quality using judgments from the authors themselves. Through quantitative and qualitative analysis grounded in narrative theory, we compare GPT-4, Claude-2.1, and LLama-2-70B. We find that all three models make faithfulness mistakes in over 50% of summaries and struggle to interpret difficult subtext. However, at their best, the models can provide thoughtful thematic analysis of stories. We additionally demonstrate that LLM judgments of summary quality do not match the feedback from the writers.
    
[^5]: Heavy-Tailed Class Imbalance and Why Adam Outperforms Gradient Descent on Language Models

    Heavy-Tailed Class Imbalance and Why Adam Outperforms Gradient Descent on Language Models

    [https://arxiv.org/abs/2402.19449](https://arxiv.org/abs/2402.19449)

    研究发现语言模型中的重尾类别不平衡问题导致了优化动态上的困难，Adam和基于符号的方法在这种情况下优于梯度下降。

    

    本文研究了在语言建模任务中存在的重尾类别不平衡问题，以及为什么Adam在优化大型语言模型时的表现优于梯度下降方法。我们发现，由于语言建模任务中存在的重尾类别不平衡，使用梯度下降时，与不常见单词相关的损失下降速度比与常见单词相关的损失下降速度慢。由于大多数样本来自相对不常见的单词，平均损失值在梯度下降时下降速度较慢。相比之下，Adam和基于符号的方法却不受此问题影响，并改善了所有类别的预测性能。我们在不同架构和数据类型上进行了实证研究，证明了这种行为确实是由类别不平衡引起的。

    arXiv:2402.19449v1 Announce Type: cross  Abstract: Adam has been shown to outperform gradient descent in optimizing large language transformers empirically, and by a larger margin than on other tasks, but it is unclear why this happens. We show that the heavy-tailed class imbalance found in language modeling tasks leads to difficulties in the optimization dynamics. When training with gradient descent, the loss associated with infrequent words decreases slower than the loss associated with frequent ones. As most samples come from relatively infrequent words, the average loss decreases slowly with gradient descent. On the other hand, Adam and sign-based methods do not suffer from this problem and improve predictions on all classes. To establish that this behavior is indeed caused by class imbalance, we show empirically that it persist through different architectures and data types, on language transformers, vision CNNs, and linear models. We further study this phenomenon on a linear clas
    
[^6]: 用神经重写系统解决算法问题

    A Neural Rewriting System to Solve Algorithmic Problems

    [https://arxiv.org/abs/2402.17407](https://arxiv.org/abs/2402.17407)

    提出了一种受重写系统启发的神经架构，用于学习算法任务，通过Selector、Solver和Combiner三个专门模块实现算法任务的简化，具有较好的外推能力

    

    现代神经网络架构仍然难以学习需要系统应用组合规则来解决超出分布问题实例的算法程序。在这项工作中，我们提出了一种原创方法来学习受重写系统启发的算法任务，重写系统是符号人工智能中的经典框架。我们展示了重写系统可以被实现为一个由专门模块组成的神经架构：选择器识别要处理的目标子表达式，求解器通过计算相应的结果简化子表达式，组合器通过用提供的解决方案替换子表达式生成原始表达式的新版本。我们在三种涉及简化涉及列表、算术和代数表达式的符号公式的算法任务上评估我们的模型。我们测试了所提架构的外推能力

    arXiv:2402.17407v1 Announce Type: cross  Abstract: Modern neural network architectures still struggle to learn algorithmic procedures that require to systematically apply compositional rules to solve out-of-distribution problem instances. In this work, we propose an original approach to learn algorithmic tasks inspired by rewriting systems, a classic framework in symbolic artificial intelligence. We show that a rewriting system can be implemented as a neural architecture composed by specialized modules: the Selector identifies the target sub-expression to process, the Solver simplifies the sub-expression by computing the corresponding result, and the Combiner produces a new version of the original expression by replacing the sub-expression with the solution provided. We evaluate our model on three types of algorithmic tasks that require simplifying symbolic formulas involving lists, arithmetic, and algebraic expressions. We test the extrapolation capabilities of the proposed architectu
    
[^7]: 跨越多个模型的统一任务嵌入：弥合基于提示的大型语言模型及其它模型的差距

    Towards Unified Task Embeddings Across Multiple Models: Bridging the Gap for Prompt-Based Large Language Models and Beyond

    [https://arxiv.org/abs/2402.14522](https://arxiv.org/abs/2402.14522)

    提出了一种框架用于统一不同模型的任务嵌入，使得任务嵌入可以跨越各种模型，并在单一向量空间内进行比较和分析。

    

    任务嵌入是一种捕捉任务特定信息的元学习技术，已经变得流行起来，特别是在多任务学习、模型编辑和可解释性等领域。文章提出了一种名为统一任务嵌入（FUTE）的框架，该框架能够协调来自各种模型（包括较小的语言模型和具有不同提示的LLMs）的任务嵌入，使其处于单一向量空间。这种统一性使得可以比较和分析不同模型之间的相似性，扩展了现有任务嵌入方法在解决多模型应用中的范围和效用。

    arXiv:2402.14522v1 Announce Type: new  Abstract: Task embedding, a meta-learning technique that captures task-specific information, has become prevalent, especially in areas such as multi-task learning, model editing, and interpretability. However, it faces challenges with the emergence of prompt-guided Large Language Models (LLMs) operating in a gradientfree manner. Existing task embedding methods rely on fine-tuned, task-specific language models, which hinders the adaptability of task embeddings across diverse models, especially prompt-based LLMs. To unleash the power of task embedding in the era of LLMs, we propose a framework for unified task embeddings (FUTE), harmonizing task embeddings from various models, including smaller language models and LLMs with varied prompts, within a single vector space. Such uniformity enables the comparison and analysis of similarities amongst different models, extending the scope and utility of existing task embedding methods in addressing multi-mo
    
[^8]: STENCIL：基于次模互信息的冷启动主动学习弱监督

    STENCIL: Submodular Mutual Information Based Weak Supervision for Cold-Start Active Learning

    [https://arxiv.org/abs/2402.13468](https://arxiv.org/abs/2402.13468)

    STENCIL利用次模互信息选择弱标记的稀有类实例，并通过标注者强标记，提高了文本分类数据集上的准确率和稀有类F-1分数。

    

    随着在NLP应用中对预训练模型进行监督微调越来越受欢迎，需要更大量的标注数据，特别是在大型语言模型的参数计数增加时。主动学习试图挖掘和注释未标记的实例以最大限度地快速改善模型性能，是减少注释成本的常见选择；然而，大多数方法通常忽视类别不平衡，并且要么假设可以访问初始标注数据，要么要求改进稀有类之前需要多轮主动学习选择。我们提出了STENCIL，它利用一组文本示例和最近提出的次模互信息来选择一组弱标记的稀有类实例，然后由标注者对其进行强标记。我们展示了STENCIL在多个文本分类数据集上将整体准确率提高了10%-24%，将稀有类F-1分数提高了17%-40%。

    arXiv:2402.13468v1 Announce Type: cross  Abstract: As supervised fine-tuning of pre-trained models within NLP applications increases in popularity, larger corpora of annotated data are required, especially with increasing parameter counts in large language models. Active learning, which attempts to mine and annotate unlabeled instances to improve model performance maximally fast, is a common choice for reducing the annotation cost; however, most methods typically ignore class imbalance and either assume access to initial annotated data or require multiple rounds of active learning selection before improving rare classes. We present STENCIL, which utilizes a set of text exemplars and the recently proposed submodular mutual information to select a set of weakly labeled rare-class instances that are then strongly labeled by an annotator. We show that STENCIL improves overall accuracy by $10\%-24\%$ and rare-class F-1 score by $17\%-40\%$ on multiple text classification datasets over commo
    
[^9]: BlendFilter: 通过查询生成混合和知识过滤推进检索增强型大型语言模型

    BlendFilter: Advancing Retrieval-Augmented Large Language Models via Query Generation Blending and Knowledge Filtering

    [https://arxiv.org/abs/2402.11129](https://arxiv.org/abs/2402.11129)

    BlendFilter通过查询生成混合和知识过滤方法提升了检索增强型大型语言模型，在多领域的问答任务中取得了显著的性能提升。

    

    arXiv:2402.11129v1 公告类型：新摘要：检索增强型大型语言模型（LLM）在提升知识密集型场景中的性能方面具有显著优势。然而，这些方法经常面临复杂输入的挑战，并且由于嘈杂的知识检索而遇到困难，明显阻碍了模型的有效性。为解决这个问题，我们引入了BlendFilter，一种通过将查询生成混合与知识过滤相结合来提升检索增强型LLM的新方法。BlendFilter提出了通过其查询生成方法的混合过程，该方法将外部知识和内部知识增强与原始查询相结合，确保全面收集信息。此外，我们独特的知识过滤模块充分利用了LLM的固有能力，有效消除了多余的数据。我们在三个开放域问答基准上进行了大量实验，结果表明

    arXiv:2402.11129v1 Announce Type: new  Abstract: Retrieval-augmented Large Language Models (LLMs) offer substantial benefits in enhancing performance across knowledge-intensive scenarios. However, these methods often face challenges with complex inputs and encounter difficulties due to noisy knowledge retrieval, notably hindering model effectiveness. To address this issue, we introduce BlendFilter, a novel approach that elevates retrieval-augmented LLMs by integrating query generation blending with knowledge filtering. BlendFilter proposes the blending process through its query generation method, which integrates both external and internal knowledge augmentation with the original query, ensuring comprehensive information gathering. Additionally, our distinctive knowledge filtering module capitalizes on the intrinsic capabilities of the LLM, effectively eliminating extraneous data. We conduct extensive experiments on three open-domain question answering benchmarks, and the findings clea
    
[^10]: GPT-4使用结构化叙事提示生成生活事件的叙述：一项验证研究

    GPT-4 Generated Narratives of Life Events using a Structured Narrative Prompt: A Validation Study

    [https://arxiv.org/abs/2402.05435](https://arxiv.org/abs/2402.05435)

    本研究通过使用结构化叙事提示，验证了GPT-4生成的叙述在传达生活事件方面的有效性。研究结果表明，大多数叙述能够足够传达提示的意图。同时，通过机器学习模型的训练和验证，可以自动识别有效和无效的叙述。

    

    大型语言模型在生成各种叙述方面发挥重要作用，促进了对其在叙述形式中传达生活事件效果的系统探索。本研究利用零-shot结构化叙事提示，使用OpenAI的GPT-4生成了24,000个叙述。从这个数据集中，我们手动分类了2,880个叙述，并评估它们在传达出生、死亡、招聘和解雇事件方面的有效性。令人惊讶的是，87.43%的叙述足够传达结构化提示的意图。为了自动识别有效和无效的叙述，我们对分类数据集训练和验证了九个机器学习模型。利用这些模型，我们扩展了对剩余21,120个叙述的分类预测分析。所有的机器学习模型在将有效的叙述分类为有效方面表现出色，但在同时将无效的叙述分类为无效方面存在挑战。我们的研究结果不仅推进了这一领域的发展，还提供了自动识别有效叙述的有益信息。

    Large Language Models (LLMs) play a pivotal role in generating vast arrays of narratives, facilitating a systematic exploration of their effectiveness for communicating life events in narrative form. In this study, we employ a zero-shot structured narrative prompt to generate 24,000 narratives using OpenAI's GPT-4. From this dataset, we manually classify 2,880 narratives and evaluate their validity in conveying birth, death, hiring, and firing events. Remarkably, 87.43% of the narratives sufficiently convey the intention of the structured prompt. To automate the identification of valid and invalid narratives, we train and validate nine Machine Learning models on the classified datasets. Leveraging these models, we extend our analysis to predict the classifications of the remaining 21,120 narratives. All the ML models excelled at classifying valid narratives as valid, but experienced challenges at simultaneously classifying invalid narratives as invalid. Our findings not only advance th
    
[^11]: 用多个时间视角增强Transformer RNNs

    Enhancing Transformer RNNs with Multiple Temporal Perspectives

    [https://arxiv.org/abs/2402.02625](https://arxiv.org/abs/2402.02625)

    引入了多个时间视角的概念，用于增强Transformer RNNs对顺序数据的理解能力，在参数数量最小增加的情况下取得了显著的改进。

    

    我们引入了多个时间视角的概念，这是一种适用于循环神经网络（RNN）架构的新方法，用于增强其对顺序数据的理解。该方法涉及维护先前遇到的文本的多样时间视图，显著丰富了语言模型解释上下文的能力。为了展示这种方法的有效性，我们将其纳入了Receptance Weighted Key Value（RWKV）架构，解决了该架构在单个隐藏状态中保留所有历史信息的固有挑战。值得注意的是，即使参数数量增加最少（仅为最初参数数量的0.04%），也实现了此改进。此外，多个时间视角所需的额外参数经过微小的计算开销进行微调，避免了完全预训练的需要。由此产生的模型在提示推断过程中保持了线性的计算复杂度。

    We introduce the concept of multiple temporal perspectives, a novel approach applicable to Recurrent Neural Network (RNN) architectures for enhancing their understanding of sequential data. This method involves maintaining diverse temporal views of previously encountered text, significantly enriching the language models' capacity to interpret context. To show the efficacy of this approach, we incorporate it into the Receptance Weighted Key Value (RWKV) architecture, addressing its inherent challenge of retaining all historical information within a single hidden state. Notably, this improvement is achieved with a minimal increase in the number of parameters --even as little as $0.04\%$ of the original number of parameters. Further, the additional parameters necessary for the multiple temporal perspectives are fine-tuned with minimal computational overhead, avoiding the need for a full pre-training. The resulting model maintains linear computational complexity during prompt inference, en
    
[^12]: 高等教育中的生成人工智能：通过大学的政策、资源和指南了解ChatGPT

    Generative AI in Higher Education: Seeing ChatGPT Through Universities' Policies, Resources, and Guidelines

    [https://arxiv.org/abs/2312.05235](https://arxiv.org/abs/2312.05235)

    通过分析美国前100名大学制定的学术政策和指南，揭示了大多数大学对于在高等教育中整合生成人工智能的开放但谨慎态度，主要关注点在于伦理使用、准确性和数据隐私。

    

    生成人工智能（GenAI）技术的进步（如ChatGPT）为丰富教育经验提供了机会，但如果被滥用，也会引发有关学术诚信的担忧。该研究旨在通过分析美国排名前100的大学制定的学术政策和指南来探讨大学和教育者如何在其学术背景中对GenAI的发展做出响应和适应。数据来源包括这些大学制定的学术政策、声明、指南以及相关资源。结果显示，大多数大学对于整合GenAI采取了开放但谨慎的态度。主要关注点在于伦理使用、准确性和数据隐私。大多数大学积极回应并提供多种资源，如课程大纲模板/示例、研讨会、共享文章等。

    arXiv:2312.05235v2 Announce Type: replace  Abstract: The advancements in Generative Artificial Intelligence (GenAI) technologies such as ChatGPT provide opportunities to enrich educational experiences, but also raise concerns about academic integrity if misused. This study aims to explore how universities and educators respond and adapt to the development of GenAI in their academic contexts by analyzing academic policies and guidelines established by top-ranked US universities regarding the use of ChatGPT in higher education. The data sources include academic policies, statements, guidelines as well as relevant resources provided by the top 100 universities in the US. Results show that the majority of these universities adopt an open but cautious approach towards the integration of GenAI. Primary concerns lie in ethical usage, accuracy, and data privacy. Most universities actively respond and provide diverse types of resources, such as syllabus templates/samples, workshops, shared arti
    
[^13]: 机器翻译中的同性关系偏见评估：它究竟是谁的妻子？

    Whose wife is it anyway? Assessing bias against same-gender relationships in machine translation. (arXiv:2401.04972v1 [cs.CL])

    [http://arxiv.org/abs/2401.04972](http://arxiv.org/abs/2401.04972)

    本文研究了机器翻译系统对同性关系的偏见问题，发现三个受欢迎的MT服务在准确翻译涉及同性别名词之间关系的句子时存在较大的错误率，特别是在涉及女性职业的上下文中表现更差。这项工作为评估NLP系统中固有偏见提供了一个社会关系方面的案例研究。

    

    机器翻译经常受到有偏见的数据和算法的困扰，这可能导致系统输出中的不可接受的错误。虽然对性别规范的偏见进行了调查研究，但对MT系统是否对社会关系编码偏见的情况了解较少，例如“律师吻了她的妻子”这样的句子。我们通过使用从几种名词性别语言（例如西班牙语）中抽取的生成模板句子，调查MT系统针对同性关系的偏见程度。我们发现三个受欢迎的MT服务在准确翻译涉及同性别名词之间关系的句子时一直存在问题。错误率根据上下文而变化很大，例如引用女性占比较高职业的同性句子的翻译准确度较低。我们提供这项工作作为研究NLP系统中固有偏见的案例研究，涉及社会关系方面的偏见评估。

    Machine translation often suffers from biased data and algorithms that can lead to unacceptable errors in system output. While bias in gender norms has been investigated, less is known about whether MT systems encode bias about social relationships, e.g. sentences such as "the lawyer kissed her wife." We investigate the degree of bias against same-gender relationships in MT systems, using generated template sentences drawn from several noun-gender languages (e.g. Spanish). We find that three popular MT services consistently fail to accurately translate sentences concerning relationships between nouns of the same gender. The error rate varies considerably based on the context, e.g. same-gender sentences referencing high female-representation occupations are translated with lower accuracy. We provide this work as a case study in the evaluation of intrinsic bias in NLP systems, with respect to social relationships.
    
[^14]: 零样本连续提示传递：在语言模型之间泛化任务语义

    Zero-Shot Continuous Prompt Transfer: Generalizing Task Semantics Across Language Models. (arXiv:2310.01691v1 [cs.CL])

    [http://arxiv.org/abs/2310.01691](http://arxiv.org/abs/2310.01691)

    这项工作提出了一种零样本连续提示传递方法，通过将源提示编码到相对空间中，并搜索相应的目标提示，在不同的语言模型之间实现了任务语义的泛化，实验证实了该方法的有效性。

    

    在自然语言处理（NLP）中，通过调整提示已经成为一种越来越受欢迎的方法，用于将大型语言模型适应特定任务。然而，这些提示，特别是连续提示，在不同模型之间的可传递性仍然是一个挑战。在这项工作中，我们提出了一种零样本连续提示传递方法，其中源提示被编码到相对空间中，并搜索相应的目标提示以将其传递到目标模型。实验结果证实了我们方法的有效性，表明连续提示中的“任务语义”可以在各种语言模型之间泛化。此外，我们发现将来自多个源模型的“任务语义”结合可以进一步增强传递的泛化能力。

    Prompt tuning in natural language processing (NLP) has become an increasingly popular method for adapting large language models to specific tasks. However, the transferability of these prompts, especially continuous prompts, between different models remains a challenge. In this work, we propose a zero-shot continuous prompt transfer method, where source prompts are encoded into relative space and the corresponding target prompts are searched for transferring to target models. Experimental results confirm the effectiveness of our method, showing that 'task semantics' in continuous prompts can be generalized across various language models. Moreover, we find that combining 'task semantics' from multiple source models can further enhance the generalizability of transfer.
    
[^15]: 基于语音的基础语法：自发联接的自监督深度神经网络

    Basic syntax from speech: Spontaneous concatenation in unsupervised deep neural networks. (arXiv:2305.01626v1 [cs.CL])

    [http://arxiv.org/abs/2305.01626](http://arxiv.org/abs/2305.01626)

    该论文提出了一种基于语音的完全无监督的方法，可以直接从原始语音中建立基础语法模型。作者发现，在基于声音的单词记录上训练的卷积神经网络可以自发连接两个或三个单词，并且可以学会将单词嵌入到新的未见过的单词组合中，这是之前未报道的属性，这一发现对我们理解神经网络的学习方式和建立从原始声学输入中的语法及其演化的模型都有重要的意义。

    

    语法的计算模型主要基于文本。本文提出了一种完全无监督的方法，可以直接从原始语音中建立基础语法模型。我们重点研究了最普遍和基本的语法特性之一——联接。我们介绍了自发联接现象：卷积神经网络(CNN)在个别单词的声学记录上训练时，开始产生输出，这些输出将两个甚至三个单词连接在一起，而不会接触到具有多个单词的输入数据。此外，训练两个单词的网络可以学习将单词嵌入到新的未见过的单词组合中。据我们所知，这是在生成对抗网络环境下训练的原始语音CNN以前未报道的属性，它不仅对我们理解这些体系结构的学习方式有影响，还对建立从原始声学输入中的语法及其演化的模型有影响。

    Computational models of syntax are predominantly text-based. Here we propose that basic syntax can be modeled directly from raw speech in a fully unsupervised way. We focus on one of the most ubiquitous and basic properties of syntax -- concatenation. We introduce spontaneous concatenation: a phenomenon where convolutional neural networks (CNNs) trained on acoustic recordings of individual words start generating outputs with two or even three words concatenated without ever accessing data with multiple words in the input. Additionally, networks trained on two words learn to embed words into novel unobserved word combinations. To our knowledge, this is a previously unreported property of CNNs trained on raw speech in the Generative Adversarial Network setting and has implications both for our understanding of how these architectures learn as well as for modeling syntax and its evolution from raw acoustic inputs.
    

