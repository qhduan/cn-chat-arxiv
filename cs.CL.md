# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [In-Context Pretraining: Language Modeling Beyond Document Boundaries.](http://arxiv.org/abs/2310.10638) | 本论文提出了一种超越文档边界的上下文预训练方法，通过在相关文档序列上训练语言模型，鼓励模型进行跨文档的阅读和推理。该方法通过改变文档顺序并应用现有的预训练管道来实现。 |
| [^2] | [The Gift of Feedback: Improving ASR Model Quality by Learning from User Corrections through Federated Learning.](http://arxiv.org/abs/2310.00141) | 本论文通过联合学习来持续从用户纠正中学习，以解决自动语音识别模型因为语言的发展和新词汇的出现而变得过时的问题，并通过针对新词汇、长尾词汇和灾难性遗忘等技术提高模型的识别效果。 |
| [^3] | [Persona-Coded Poly-Encoder: Persona-Guided Multi-Stream Conversational Sentence Scoring.](http://arxiv.org/abs/2309.16770) | 本论文提出了一种新颖的Persona编码多流程对话句子评分方法，利用个人角色信息来提高对话生成的质量。 |
| [^4] | [QuantEase: Optimization-based Quantization for Language Models -- An Efficient and Intuitive Algorithm.](http://arxiv.org/abs/2309.01885) | QuantEase是一种基于优化的语言模型量化算法，通过逐层量化和基于坐标下降的算法，高质量地解决了复杂的非凸量化问题，并引入了对异常值敏感的变种方法。 |
| [^5] | [RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback.](http://arxiv.org/abs/2309.00267) | RLAIF是一种新的强化学习方法，利用AI反馈代替人类标注偏好，相比强化学习从人类反馈中学习（RLHF），在摘要任务上取得了类似的改进效果，并且在人类评估中得到了相同的认可。这提供了一种有潜力解决RLHF的可扩展性限制的解决方案。 |
| [^6] | [PointLLM: Empowering Large Language Models to Understand Point Clouds.](http://arxiv.org/abs/2308.16911) | PointLLM是一种使大型语言模型理解点云的方法，它利用点云编码器和强大的LLM将几何、外观和语言信息融合，并通过人类指导生成环境上恰当的响应。该方法通过收集大规模的点-文本指令对数据集进行两阶段的训练，以提高模型的感知能力和泛化能力。 |
| [^7] | [Large Language Models of Code Fail at Completing Code with Potential Bugs.](http://arxiv.org/abs/2306.03438) | 本研究探讨了存在漏洞的代码补全问题，设计了两个数据集并发现这些漏洞显著降低了Code-LLMs的生成性能。 |
| [^8] | [A Controllable QA-based Framework for Decontextualization.](http://arxiv.org/abs/2305.14772) | 本文提出了一个基于问答的去文本化框架，可以更好地展示提取的文本摘录。在问答和引证上的表现类似于端到端方法，并且支持用户信息需求及偏好的可控性。 |
| [^9] | [Pointwise Mutual Information Based Metric and Decoding Strategy for Faithful Generation in Document Grounded Dialogs.](http://arxiv.org/abs/2305.12191) | 该论文研究了文档导向对话生成中的保真度问题，并提出了一种基于点互信息的度量方法和解码策略来预测更加保真的回答。 |
| [^10] | [An Adversarial Non-Autoregressive Model for Text Generation with Incomplete Information.](http://arxiv.org/abs/2305.03977) | 提出了一种新的对抗非自回归Transformer模型用于对不完整信息的文本生成，其具有位置感知自调节和依赖接口网络，能够在与其他主流模型相比更快的解码时间内获得可比较性能，具有在潜在插值等应用中的巨大潜力。 |
| [^11] | [ReCEval: Evaluating Reasoning Chains via Correctness and Informativeness.](http://arxiv.org/abs/2304.10703) | 本文提出了一种基于推导链正确性和信息量的推理链评估框架ReCEval，用以评估多步推理能力。该框架能够客观、系统和准确地评估推理链，并在多个数据集上实现了良好的效果。 |
| [^12] | [RAFT: Reward rAnked FineTuning for Generative Foundation Model Alignment.](http://arxiv.org/abs/2304.06767) | RAFT框架引入了奖励排名微调方法，用于对齐生成型基础模型，以解决强化学习带来的低效和不稳定性问题。 |
| [^13] | [Retrieving Multimodal Information for Augmented Generation: A Survey.](http://arxiv.org/abs/2303.10868) | 这项调研综述了通过检索多模态知识来协助和增强生成模型的方法，这些方法提供了解决准确性、推理性、可解释性和鲁棒性等重要问题的有希望的解决方案。 |

# 详细

[^1]: 超越文档边界的上下文预训练：语言模型

    In-Context Pretraining: Language Modeling Beyond Document Boundaries. (arXiv:2310.10638v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2310.10638](http://arxiv.org/abs/2310.10638)

    本论文提出了一种超越文档边界的上下文预训练方法，通过在相关文档序列上训练语言模型，鼓励模型进行跨文档的阅读和推理。该方法通过改变文档顺序并应用现有的预训练管道来实现。

    

    目前，大型语言模型（LMs）通过预测给定文档前缀的标记来进行训练，从而能够直接进行长篇生成和提示式任务，这可以简化为文档完成。现有的预训练管道通过连接随机组合的短文档来训练LMs，以创建输入上下文，但前一个文档对于预测下一个文档没有提供任何信号。我们提出了一种新方法——上下文预训练，即在相关文档序列上预先训练语言模型，从而明确鼓励它们跨越文档边界进行阅读和推理。我们可以通过改变文档顺序，使每个上下文包含相关的文档，并直接应用现有的预训练管道来进行上下文预训练。然而，这个文档排序问题很具有挑战性。有数十亿个文档，我们希望在每个文档中最大化上下文相似性而不重复任何数据。

    Large language models (LMs) are currently trained to predict tokens given document prefixes, enabling them to directly perform long-form generation and prompting-style tasks which can be reduced to document completion. Existing pretraining pipelines train LMs by concatenating random sets of short documents to create input contexts but the prior documents provide no signal for predicting the next document. We instead present In-Context Pretraining, a new approach where language models are pretrained on a sequence of related documents, thereby explicitly encouraging them to read and reason across document boundaries. We can do In-Context Pretraining by simply changing the document ordering so that each context contains related documents, and directly applying existing pretraining pipelines. However, this document sorting problem is challenging. There are billions of documents and we would like the sort to maximize contextual similarity for every document without repeating any data. To do
    
[^2]: 用户反馈的馈赠：通过联合学习从用户纠正中提高ASR模型质量

    The Gift of Feedback: Improving ASR Model Quality by Learning from User Corrections through Federated Learning. (arXiv:2310.00141v1 [cs.CL])

    [http://arxiv.org/abs/2310.00141](http://arxiv.org/abs/2310.00141)

    本论文通过联合学习来持续从用户纠正中学习，以解决自动语音识别模型因为语言的发展和新词汇的出现而变得过时的问题，并通过针对新词汇、长尾词汇和灾难性遗忘等技术提高模型的识别效果。

    

    自动语音识别（ASR）模型通常在大量的转录语音数据集上进行训练。随着语言的发展和新词汇的出现，这些模型可能变得过时和陈旧。在基于服务器训练但部署在边缘设备上的模型中，错误可能是由于服务器训练数据与实际设备使用之间的不匹配导致的。在这项工作中，我们通过联合学习来不断从设备上的用户纠正中学习，从而解决这个问题。我们探索了一些技术，以针对模型以前未遇到过的新词汇，学习长尾词汇，并减轻灾难性遗忘现象。在实验评估中，我们发现所提出的技术改进了模型对新词汇的识别，同时保持了整体语言分布的质量。

    Automatic speech recognition (ASR) models are typically trained on large datasets of transcribed speech. As language evolves and new terms come into use, these models can become outdated and stale. In the context of models trained on the server but deployed on edge devices, errors may result from the mismatch between server training data and actual on-device usage. In this work, we seek to continually learn from on-device user corrections through Federated Learning (FL) to address this issue. We explore techniques to target fresh terms that the model has not previously encountered, learn long-tail words, and mitigate catastrophic forgetting. In experimental evaluations, we find that the proposed techniques improve model recognition of fresh terms, while preserving quality on the overall language distribution.
    
[^3]: Persona编码多流程对话句子评分：Persona引导的多流程对话句子评分方法

    Persona-Coded Poly-Encoder: Persona-Guided Multi-Stream Conversational Sentence Scoring. (arXiv:2309.16770v1 [cs.CL])

    [http://arxiv.org/abs/2309.16770](http://arxiv.org/abs/2309.16770)

    本论文提出了一种新颖的Persona编码多流程对话句子评分方法，利用个人角色信息来提高对话生成的质量。

    

    机器学习和深度学习的最新进展已经在许多实际应用中广泛应用于对话AI。然而，要利用可以提供对话背景或个性化调整的辅助信息以提高对话质量仍然很具挑战性。例如，关于使用个人角色信息来提高对话质量的研究仅有限，即使是最先进的对话AI技术也无法有效地利用来自多种来源的辅助数据信号，例如多模式交互数据、人口统计学数据和社会确定因素数据等。在本文中，我们提出了一种新颖的Persona编码多流程对话句子评分方法，它利用多流程编码方案中的个人角色信息来提高对话生成的质量。为了展示所提出方法的有效性，我们在两个不同的基于个人角色的对话数据集上评估了我们的方法，并与参考方法进行了比较。

    Recent advances in machine learning and deep learning have led to the widespread use of Conversational AI in many practical applications. However, it is still very challenging to leverage auxiliary information that can provide conversational context or personalized tuning to improve the quality of conversations. For example, there has only been limited research on using an individuals persona information to improve conversation quality, and even state-of-the-art conversational AI techniques are unable to effectively leverage signals from heterogeneous sources of auxiliary data, such as multi-modal interaction data, demographics, SDOH data, etc. In this paper, we present a novel Persona-Coded Poly-Encoder method that leverages persona information in a multi-stream encoding scheme to improve the quality of response generation for conversations. To show the efficacy of the proposed method, we evaluate our method on two different persona-based conversational datasets, and compared against 
    
[^4]: QuantEase: 基于优化的语言模型量化--一种高效而直观的算法

    QuantEase: Optimization-based Quantization for Language Models -- An Efficient and Intuitive Algorithm. (arXiv:2309.01885v1 [stat.ML])

    [http://arxiv.org/abs/2309.01885](http://arxiv.org/abs/2309.01885)

    QuantEase是一种基于优化的语言模型量化算法，通过逐层量化和基于坐标下降的算法，高质量地解决了复杂的非凸量化问题，并引入了对异常值敏感的变种方法。

    

    随着大型语言模型（LLM）的普及，对于能够实现其高效部署的压缩技术的兴趣日益增加。本研究侧重于LLM的后训练量化（PTQ）。借鉴最近的进展，我们的工作引入了QuantEase，一个逐层量化框架，其中各个层面经过单独的量化。该问题被视为离散结构化的非凸优化问题，促使我们开发了基于坐标下降（CD）技术的算法。这些基于CD的方法为复杂的非凸逐层量化问题提供了高质量的解决方案。值得注意的是，我们的CD方法具有简单的更新步骤，仅依赖于矩阵和向量运算，避免了矩阵求逆或分解的需要。我们还探索了一种对异常值敏感的变种方法，允许保留具有完全精度的重要权重（异常值）。我们的提议达到了最先进的状态。

    With the rising popularity of Large Language Models (LLMs), there has been an increasing interest in compression techniques that enable their efficient deployment. This study focuses on the Post-Training Quantization (PTQ) of LLMs. Drawing from recent advances, our work introduces QuantEase, a layer-wise quantization framework where individual layers undergo separate quantization. The problem is framed as a discrete-structured non-convex optimization, prompting the development of algorithms rooted in Coordinate Descent (CD) techniques. These CD-based methods provide high-quality solutions to the complex non-convex layer-wise quantization problems. Notably, our CD-based approach features straightforward updates, relying solely on matrix and vector operations, circumventing the need for matrix inversion or decomposition. We also explore an outlier-aware variant of our approach, allowing for retaining significant weights (outliers) with complete precision. Our proposal attains state-of-th
    
[^5]: RLAIF: 使用AI反馈来扩展强化学习从人类反馈中学习

    RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback. (arXiv:2309.00267v1 [cs.CL])

    [http://arxiv.org/abs/2309.00267](http://arxiv.org/abs/2309.00267)

    RLAIF是一种新的强化学习方法，利用AI反馈代替人类标注偏好，相比强化学习从人类反馈中学习（RLHF），在摘要任务上取得了类似的改进效果，并且在人类评估中得到了相同的认可。这提供了一种有潜力解决RLHF的可扩展性限制的解决方案。

    

    从人类反馈中进行强化学习（RLHF）对于将大型语言模型（LLMs）与人类偏好相一致是有效的，但是收集高质量的人类偏好标签是一个关键瓶颈。我们比较了RLHF和利用现成的LLM进行标记的RL from AI Feedback (RLAIF)技术，并发现它们都能获得类似的改善效果。在摘要任务上，人类评估者在约70%的案例中都更喜欢RLAIF和RLHF产生的文本，而不是基准的监督微调模型。此外，当被要求评估RLAIF和RLHF的摘要时，人类以相同的比率更喜欢两者。这些结果表明，RLAIF可以达到人类水平的性能，为克服RLHF的可扩展性限制提供了潜在的解决方案。

    Reinforcement learning from human feedback (RLHF) is effective at aligning large language models (LLMs) to human preferences, but gathering high quality human preference labels is a key bottleneck. We conduct a head-to-head comparison of RLHF vs. RL from AI Feedback (RLAIF) - a technique where preferences are labeled by an off-the-shelf LLM in lieu of humans, and we find that they result in similar improvements. On the task of summarization, human evaluators prefer generations from both RLAIF and RLHF over a baseline supervised fine-tuned model in ~70% of cases. Furthermore, when asked to rate RLAIF vs. RLHF summaries, humans prefer both at equal rates. These results suggest that RLAIF can yield human-level performance, offering a potential solution to the scalability limitations of RLHF.
    
[^6]: PointLLM：赋予大型语言模型理解点云的能力

    PointLLM: Empowering Large Language Models to Understand Point Clouds. (arXiv:2308.16911v1 [cs.CV])

    [http://arxiv.org/abs/2308.16911](http://arxiv.org/abs/2308.16911)

    PointLLM是一种使大型语言模型理解点云的方法，它利用点云编码器和强大的LLM将几何、外观和语言信息融合，并通过人类指导生成环境上恰当的响应。该方法通过收集大规模的点-文本指令对数据集进行两阶段的训练，以提高模型的感知能力和泛化能力。

    

    大型语言模型（LLM）的前所未有的进展对自然语言处理产生了深远影响，但在3D理解领域仍有待完全发展。本文介绍了PointLLM，这是一项填补这一空白的初步工作，使LLM能够理解点云，并提供了超越2D视觉数据的新途径。PointLLM通过人类指导处理带有颜色的物体点云，并生成环境上恰当的响应，展示了其对点云和常识的掌握。具体来说，它利用了一个点云编码器和一个强大的LLM，有效地融合了几何、外观和语言信息。我们收集了一个新颖的数据集，包括66万个简单和7万个复杂的点-文本指令对，以实现两阶段的训练策略：首先对齐潜在空间，然后对统一模型进行指令调整。为了严格评估我们模型的感知能力和其泛化能力，我们建立了评估基准数据集进行实验。

    The unprecedented advancements in Large Language Models (LLMs) have created a profound impact on natural language processing but are yet to fully embrace the realm of 3D understanding. This paper introduces PointLLM, a preliminary effort to fill this gap, thereby enabling LLMs to understand point clouds and offering a new avenue beyond 2D visual data. PointLLM processes colored object point clouds with human instructions and generates contextually appropriate responses, illustrating its grasp of point clouds and common sense. Specifically, it leverages a point cloud encoder with a powerful LLM to effectively fuse geometric, appearance, and linguistic information. We collect a novel dataset comprising 660K simple and 70K complex point-text instruction pairs to enable a two-stage training strategy: initially aligning latent spaces and subsequently instruction-tuning the unified model. To rigorously evaluate our model's perceptual abilities and its generalization capabilities, we establis
    
[^7]: 代码大语言模型在填写可能存在漏洞的代码时存在失败问题

    Large Language Models of Code Fail at Completing Code with Potential Bugs. (arXiv:2306.03438v1 [cs.LG])

    [http://arxiv.org/abs/2306.03438](http://arxiv.org/abs/2306.03438)

    本研究探讨了存在漏洞的代码补全问题，设计了两个数据集并发现这些漏洞显著降低了Code-LLMs的生成性能。

    

    最近，代码大语言模型（Code-LLMs）在代码补全方面取得了巨大进展，这是编程辅助和代码智能的基本功能。然而，大多数现有的研究忽略了在生成过程中代码上下文中可能存在的漏洞问题，在软件开发中这是不可避免的。因此，我们引入并研究了存在漏洞的代码补全问题，受实时代码建议的现实场景启发，代码上下文中包含可能的漏洞-反模式，这些反模式可以成为完成程序中的漏洞。为了系统地研究任务，我们引入了两个数据集：一个是从语义改变操作中派生的合成漏洞数据集（buggy-HumanEval），另一个是从用户提交的编程问题中派生的现实漏洞数据集（buggy-FixEval）。我们发现，可能存在漏洞的情况显著降低了高性能Code-LLMs的生成性能。例如，CodeGen-2B-mono在测试数据集上的通过率

    Large language models of code (Code-LLMs) have recently brought tremendous advances to code completion, a fundamental feature of programming assistance and code intelligence. However, most existing works ignore the possible presence of bugs in the code context for generation, which are inevitable in software development. Therefore, we introduce and study the buggy-code completion problem, inspired by the realistic scenario of real-time code suggestion where the code context contains potential bugs -- anti-patterns that can become bugs in the completed program. To systematically study the task, we introduce two datasets: one with synthetic bugs derived from semantics-altering operator changes (buggy-HumanEval) and one with realistic bugs derived from user submissions to coding problems (buggy-FixEval). We find that the presence of potential bugs significantly degrades the generation performance of the high-performing Code-LLMs. For instance, the passing rates of CodeGen-2B-mono on test 
    
[^8]: 一种可控的基于问答的去文本化框架

    A Controllable QA-based Framework for Decontextualization. (arXiv:2305.14772v1 [cs.CL])

    [http://arxiv.org/abs/2305.14772](http://arxiv.org/abs/2305.14772)

    本文提出了一个基于问答的去文本化框架，可以更好地展示提取的文本摘录。在问答和引证上的表现类似于端到端方法，并且支持用户信息需求及偏好的可控性。

    

    许多真实场景下的应用需要将提取的摘录展示给用户，这些摘录往往需要解耦原来的文本才能更好地呈现给用户。本文研究了LLMs在问答和引证上的去文本化能力，并提出了一个基于问答的去文本化框架，该框架可以更好地满足用户信息需求及偏好，并且在结果上表现出类似于端到端方法的竞争力。我们同时探讨了如何通过该框架将用户偏好融入到系统中，从而实现了可控性。

    Many real-world applications require surfacing extracted snippets to users, whether motivated by assistive tools for literature surveys or document cross-referencing, or needs to mitigate and recover from model generated inaccuracies., Yet, these passages can be difficult to consume when divorced from their original document context. In this work, we explore the limits of LLMs to perform decontextualization of document snippets in user-facing scenarios, focusing on two real-world settings - question answering and citation context previews for scientific documents. We propose a question-answering framework for decontextualization that allows for better handling of user information needs and preferences when determining the scope of rewriting. We present results showing state-of-the-art LLMs under our framework remain competitive with end-to-end approaches. We also explore incorporating user preferences into the system, finding our framework allows for controllability.
    
[^9]: 基于点互信息度量和解码策略的文档导向对话生成中的保真度研究

    Pointwise Mutual Information Based Metric and Decoding Strategy for Faithful Generation in Document Grounded Dialogs. (arXiv:2305.12191v1 [cs.CL])

    [http://arxiv.org/abs/2305.12191](http://arxiv.org/abs/2305.12191)

    该论文研究了文档导向对话生成中的保真度问题，并提出了一种基于点互信息的度量方法和解码策略来预测更加保真的回答。

    

    在基于深度学习的文档导向对话生成中，一个主要问题是生成的回答可能与底层文档不一致。现有的用于评估生成的回答是否与底层文档一致的自动化度量方法，度量生成的回答与文档内容之间的相似度。然而，这些自动化度量方法与人类判断相比差异很大。因此，为了提高保真度的测量，我们提出一种基于条件点互信息（PMI）的新度量方法，该度量方法量化生成的回答与源文档之间的PMI，受对话条件的影响。PMI量化文档对生成的回答的影响程度，PMI越高则回答越保真。我们基于此思想构建了一种新的解码技术，将PMI并入到生成流程中，以预测更加保真的回答。

    A major concern in using deep learning based generative models for document-grounded dialogs is the potential generation of responses that are not \textit{faithful} to the underlying document. Existing automated metrics used for evaluating the faithfulness of response with respect to the grounding document measure the degree of similarity between the generated response and the document's content. However, these automated metrics are far from being well aligned with human judgments. Therefore, to improve the measurement of faithfulness, we propose a new metric that utilizes (Conditional) Point-wise Mutual Information (PMI) between the generated response and the source document, conditioned on the dialogue. PMI quantifies the extent to which the document influences the generated response -- with a higher PMI indicating a more faithful response. We build upon this idea to create a new decoding technique that incorporates PMI into the response generation process to predict more faithful re
    
[^10]: 一种带有不完整信息的文本生成对抗非自回归模型

    An Adversarial Non-Autoregressive Model for Text Generation with Incomplete Information. (arXiv:2305.03977v1 [cs.CL])

    [http://arxiv.org/abs/2305.03977](http://arxiv.org/abs/2305.03977)

    提出了一种新的对抗非自回归Transformer模型用于对不完整信息的文本生成，其具有位置感知自调节和依赖接口网络，能够在与其他主流模型相比更快的解码时间内获得可比较性能，具有在潜在插值等应用中的巨大潜力。

    

    非自回归模型在完整信息情况（CIS）下已广泛研究，其中模型具有完整的输入信息来获取相应的输出。然而，它们在不完整信息情况（IIS）下的探索极为有限。我们的分析表明，IIS中不完整的输入信息将增加在最大似然估计下训练的现有非自回归模型的固有限制。在本文中，我们针对IIS提出了一种对抗非自回归Transformer （ANT）模型，具有两个新特性：1）位置感知自调节，可以提供更合理的隐藏表示；2）依赖性前馈网络，可以增强其依赖性建模能力。我们将ANT与IIS中的其他主流模型进行比较，并证明ANT可以实现可比较性能，同时也可以比其他模型更快地进行解码。此外，我们展示了ANT在潜在插值等各种应用方面的巨大潜力。

    Non-autoregressive models have been widely studied in the Complete Information Scenario (CIS), in which the models have complete input information to obtain corresponding output. However, their explorations in the Incomplete Information Scenario (IIS) are extremely limited. Our analyses reveal that the IIS's incomplete input information will augment the inherent limitations of existing non-autoregressive models trained under Maximum Likelihood Estimation. In this paper, we propose for the IIS an Adversarial Non-autoregressive Transformer (ANT) which has two novel features: 1) Position Aware Self-Modulation to provide more reasonable hidden representations, and 2) Dependency Feed Forward Network to strengthen its capacity in dependency modeling. We compare ANT with other mainstream models in the IIS and demonstrate that ANT can achieve comparable performance with much fewer decoding iterations. Furthermore, we show its great potential in various applications like latent interpolation an
    
[^11]: 通过正确性和信息量评估推理链的ReCEval

    ReCEval: Evaluating Reasoning Chains via Correctness and Informativeness. (arXiv:2304.10703v1 [cs.CL])

    [http://arxiv.org/abs/2304.10703](http://arxiv.org/abs/2304.10703)

    本文提出了一种基于推导链正确性和信息量的推理链评估框架ReCEval，用以评估多步推理能力。该框架能够客观、系统和准确地评估推理链，并在多个数据集上实现了良好的效果。

    

    多步推理能力在许多自然语言任务中都是基础，但什么构成好的推理链以及如何评估它们尚不清楚。大多数现有方法仅关注推理链是否导致正确的结论，但这种以答案为导向的观点可能会将好的推理质量与其他用于预测答案的假捷径混淆。为了弥补这一差距，我们将推理链视为推导最终答案的非正式证明，通过评估推理链的两个关键特性——（1）正确性，即每个步骤基于步骤，前置步骤和输入上下文中包含的信息进行有效推理，以及（2）信息量，即每个步骤提供新信息有助于推导生成的答案——我们提出了ReCEval（推理链评估）框架。我们使用自然语言推理模型和信息理论测量实现了ReCEval。在多个数据集上的实验表明，我们的框架在评估推理链方面比现有方法更加客观、系统和准确。

    Multi-step reasoning ability is fundamental to many natural language tasks, yet it is unclear what constitutes a good reasoning chain and how to evaluate them. Most existing methods focus solely on whether the reasoning chain leads to the correct conclusion, but this answer-oriented view may confound the quality of reasoning with other spurious shortcuts to predict the answer. To bridge this gap, we evaluate reasoning chains by viewing them as informal proofs that derive the final answer. Specifically, we propose ReCEval (Reasoning Chain Evaluation), a framework that evaluates reasoning chains through two key properties: (1) correctness, i.e., each step makes a valid inference based on the information contained within the step, preceding steps, and input context, and (2) informativeness, i.e., each step provides new information that is helpful towards deriving the generated answer. We implement ReCEval using natural language inference models and information-theoretic measures. On multi
    
[^12]: RAFT: 奖励排名微调用于生成型基础模型对齐

    RAFT: Reward rAnked FineTuning for Generative Foundation Model Alignment. (arXiv:2304.06767v1 [cs.LG])

    [http://arxiv.org/abs/2304.06767](http://arxiv.org/abs/2304.06767)

    RAFT框架引入了奖励排名微调方法，用于对齐生成型基础模型，以解决强化学习带来的低效和不稳定性问题。

    

    生成型基础模型容易受到广泛的无监督训练数据带来的隐式偏见的影响。这些偏见可能导致子优样本、扭曲的结果和不公平，可能产生重大影响。因此，将这些模型与人的伦理和偏好对齐是确保它们在真实应用中负责任和有效的部署的关键步骤。以往的研究主要采用人类反馈的强化学习（ RLHF）作为解决这个问题的手段。在 RL 算法的指导下，用人类反馈指导的奖励模型对生成模型进行微调。然而， RL 算法的低效性和不稳定性常常会对生成模型的成功对齐产生重大障碍，因此需要开发一种更为强大和简化的方法。为此，我们引入了一个新的框架，即奖励排名微调（ RAFT ），旨在对齐生成基础模型。

    Generative foundation models are susceptible to implicit biases that can arise from extensive unsupervised training data. Such biases can produce suboptimal samples, skewed outcomes, and unfairness, with potentially significant repercussions. Consequently, aligning these models with human ethics and preferences is an essential step toward ensuring their responsible and effective deployment in real-world applications. Prior research has primarily employed Reinforcement Learning from Human Feedback (RLHF) as a means of addressing this problem, wherein generative models are fine-tuned using RL algorithms guided by a human-feedback-informed reward model. However, the inefficiencies and instabilities associated with RL algorithms frequently present substantial obstacles to the successful alignment of generative models, necessitating the development of a more robust and streamlined approach. To this end, we introduce a new framework, Reward rAnked FineTuning (RAFT), designed to align generat
    
[^13]: 检索多模态信息用于增强生成：一项调研

    Retrieving Multimodal Information for Augmented Generation: A Survey. (arXiv:2303.10868v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2303.10868](http://arxiv.org/abs/2303.10868)

    这项调研综述了通过检索多模态知识来协助和增强生成模型的方法，这些方法提供了解决准确性、推理性、可解释性和鲁棒性等重要问题的有希望的解决方案。

    

    随着大型语言模型的普及，使用多模态来增强语言模型的生成能力成为一个重要趋势，这使得语言模型能更好地与世界互动。然而，对于在哪个阶段以及如何融合不同模态的问题缺乏一个统一的认识。在本调研中，我们回顾了通过检索多模态知识来协助和增强生成模型的方法，这些知识的格式包括图像、代码、表格、图形和音频。这些方法为实现准确性、推理性、可解释性和鲁棒性等重要问题提供了有希望的解决方案。通过提供深入的回顾，本调研旨在让学者们更深入地了解这些方法的应用，并鼓励他们将现有技术应用于快速发展的语言模型领域。

    As Large Language Models (LLMs) become popular, there emerged an important trend of using multimodality to augment the LLMs' generation ability, which enables LLMs to better interact with the world. However, there lacks a unified perception of at which stage and how to incorporate different modalities. In this survey, we review methods that assist and augment generative models by retrieving multimodal knowledge, whose formats range from images, codes, tables, graphs, to audio. Such methods offer a promising solution to important concerns such as factuality, reasoning, interpretability, and robustness. By providing an in-depth review, this survey is expected to provide scholars with a deeper understanding of the methods' applications and encourage them to adapt existing techniques to the fast-growing field of LLMs.
    

