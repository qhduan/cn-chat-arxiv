# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Efficient LLM Inference on CPUs.](http://arxiv.org/abs/2311.00502) | 本研究提出了一种在CPU上高效部署大型语言模型（LLMs）的方法，支持自动权重量化和优化内核，在流行的LLMs上展示了极高的推理效率。 |
| [^2] | [PromptAgent: Strategic Planning with Language Models Enables Expert-level Prompt Optimization.](http://arxiv.org/abs/2310.16427) | PromptAgent是一个通过战略规划将目标任务与大型语言模型相结合实现优化的方法，能够自主设计与专家手工打造的提示相当的优化方法。 |
| [^3] | [Unveiling the General Intelligence Factor in Language Models: A Psychometric Approach.](http://arxiv.org/abs/2310.11616) | 本研究利用心理测量理论揭示了语言模型中的普遍智能因子g的存在，并发现了该因子解释模型性能方差的85%，为模型评估和开发提供了统一的指标。 |
| [^4] | [UPAR: A Kantian-Inspired Prompting Framework for Enhancing Large Language Model Capabilities.](http://arxiv.org/abs/2310.01441) | UPAR提示框架模拟人类认知结构，在大型语言模型中提供了可理解和可检查的推理轨迹，增强了推理的可解释性和准确性，并为现有的提示技术提供了认识论基础。 |
| [^5] | [Sudden Drops in the Loss: Syntax Acquisition, Phase Transitions, and Simplicity Bias in MLMs.](http://arxiv.org/abs/2309.07311) | 本文通过对掩码语言模型中的语法习得进行案例研究，发现在训练的一个短暂窗口内，模型突然获得了语法注意结构(SAS)，并伴随着损失的陡峭下降。SAS对随后习得语言能力起到了重要的促进作用。 |
| [^6] | [Large Language Models as Optimizers.](http://arxiv.org/abs/2309.03409) | 本论文提出了一种简单有效的方法，利用大型语言模型(LLMs)作为优化器，通过自然语言描述优化任务。经过实验证明，该方法在线性回归和旅行推销员问题上表现出色，并且优化的最佳提示超过了人为设计的提示。 |
| [^7] | [Attention-Driven Multi-Modal Fusion: Enhancing Sign Language Recognition and Translation.](http://arxiv.org/abs/2309.01860) | 本文提出了一种注意力驱动的多模态融合机制，通过将光流信息与RGB图像相结合，丰富了连续手语识别和翻译流程中的特征。该方法在手语识别任务中降低了WER 0.9，在翻译任务中提高了测试集上大多数BLEU分数约0.6。 |
| [^8] | [Towards Grounded Visual Spatial Reasoning in Multi-Modal Vision Language Models.](http://arxiv.org/abs/2308.09778) | 本文旨在研究多模态视觉语言模型在理解空间关系方面的能力，提出了细粒度组合的空间关系基础，并采用自底向上的方法评估空间关系推理任务的性能。 |
| [^9] | [From Pixels to UI Actions: Learning to Follow Instructions via Graphical User Interfaces.](http://arxiv.org/abs/2306.00245) | 本文提出了一种基于像素级别的预训练方法，建立了一种模拟人类概念界面和混合动作空间的代理，实现了在GUI指令遵循任务的MiniWob++基准测试中超越人类工作者的目标。 |
| [^10] | [Post Hoc Explanations of Language Models Can Improve Language Models.](http://arxiv.org/abs/2305.11426) | 本文提出了一种新的框架AMPLIFY，利用后验解释自动化生成原因，并在多个数据集和任务上显著提高现有语言模型的性能。 |
| [^11] | [Interactive Text-to-SQL Generation via Editable Step-by-Step Explanations.](http://arxiv.org/abs/2305.07372) | 本文介绍一种交互机制，允许用户直接编辑一步步解释错误SQL以修复SQL错误，实验证明方法提高了31.6％的执行准确性。用户研究表明，该方法帮助用户以更少的时间和更高的信心解决了更多的SQL任务。 |
| [^12] | [Conversational Semantic Parsing using Dynamic Context Graphs.](http://arxiv.org/abs/2305.06164) | 本论文提出了一种新的方法，使用动态创建的子图表示话语及上下文的信息来进行会话语义解析，并利用图形神经网络编码，可表示大量看不见的节点，比静态方法更为优越。 |
| [^13] | [Can Large Language Models Transform Computational Social Science?.](http://arxiv.org/abs/2305.03514) | 本文研究了大型语言模型作为计算社会科学工具的潜力。虽然在分类任务上没有优势，但在自由形式编码任务上表现优异，今后可以作为零-shot检测工具进行使用， |
| [^14] | [ZeroNLG: Aligning and Autoencoding Domains for Zero-Shot Multimodal and Multilingual Natural Language Generation.](http://arxiv.org/abs/2303.06458) | ZeroNLG是一个零样本学习框架，可以处理多个NLG任务，包括图像到文本、视频到文本和文本到文本，跨越英语、中文、德语和法语。它不需要任何标记的下游对进行训练，通过将不同的领域投影到共享的公共潜在空间中的相应坐标，桥接不同领域之间的差异。 |

# 详细

[^1]: 在CPU上高效的LLM推理

    Efficient LLM Inference on CPUs. (arXiv:2311.00502v1 [cs.LG])

    [http://arxiv.org/abs/2311.00502](http://arxiv.org/abs/2311.00502)

    本研究提出了一种在CPU上高效部署大型语言模型（LLMs）的方法，支持自动权重量化和优化内核，在流行的LLMs上展示了极高的推理效率。

    

    大型语言模型(LLMs)已经在各种任务上展示出了令人瞩目的性能和巨大的潜力。然而，由于模型参数的庞大数量，LLMs的部署一直面临挑战，对大内存容量和高内存带宽的需求。在本文中，我们提出了一种有效的方法，可以使LLMs的部署更高效。我们支持自动的INT4权重量化流程，并设计了一个特殊的LLM运行时，具有高度优化的内核，以加速在CPU上的LLM推理。我们展示了我们的方法在流行的LLMs上的普适性，包括Llama2，Llama，GPT-NeoX，并展示了在CPU上的极高推理效率。代码公开可用于: https://github.com/intel/intel-extension-for-transformers.

    Large language models (LLMs) have demonstrated remarkable performance and tremendous potential across a wide range of tasks. However, deploying these models has been challenging due to the astronomical amount of model parameters, which requires a demand for large memory capacity and high memory bandwidth. In this paper, we propose an effective approach that can make the deployment of LLMs more efficiently. We support an automatic INT4 weight-only quantization flow and design a special LLM runtime with highly-optimized kernels to accelerate the LLM inference on CPUs. We demonstrate the general applicability of our approach on popular LLMs including Llama2, Llama, GPT-NeoX, and showcase the extreme inference efficiency on CPUs. The code is publicly available at: https://github.com/intel/intel-extension-for-transformers.
    
[^2]: PromptAgent: 通过语言模型的战略规划实现优化

    PromptAgent: Strategic Planning with Language Models Enables Expert-level Prompt Optimization. (arXiv:2310.16427v1 [cs.CL])

    [http://arxiv.org/abs/2310.16427](http://arxiv.org/abs/2310.16427)

    PromptAgent是一个通过战略规划将目标任务与大型语言模型相结合实现优化的方法，能够自主设计与专家手工打造的提示相当的优化方法。

    

    高效、任务特定的提示通常需要经过专家的精心设计，需要结合详细的指导和领域见解，基于对大型语言模型和目标任务细节的深入理解。然而，自动化生成专家级提示的方法仍然难以实现。现有的提示优化方法往往忽视领域知识的深度，并且难以高效地探索专家级提示的广阔空间。为解决这个问题，我们提出了PromptAgent，这是一种自主设计与专家手工打造的提示相当的优化方法。PromptAgent将提示优化视为一个战略规划问题，并采用一种根据Monte Carlo树搜索的规则计算算法在专家级提示空间中进行有效导航。PromptAgent通过类人类的反复试错探索，引发精确的专家级见解和深入的指令。

    Highly effective, task-specific prompts are often heavily engineered by experts to integrate detailed instructions and domain insights based on a deep understanding of both instincts of large language models (LLMs) and the intricacies of the target task. However, automating the generation of such expert-level prompts remains elusive. Existing prompt optimization methods tend to overlook the depth of domain knowledge and struggle to efficiently explore the vast space of expert-level prompts. Addressing this, we present PromptAgent, an optimization method that autonomously crafts prompts equivalent in quality to those handcrafted by experts. At its core, PromptAgent views prompt optimization as a strategic planning problem and employs a principled planning algorithm, rooted in Monte Carlo tree search, to strategically navigate the expert-level prompt space. Inspired by human-like trial-and-error exploration, PromptAgent induces precise expert-level insights and in-depth instructions by r
    
[^3]: 揭示语言模型中的普遍智能因子：一种心理测量方法

    Unveiling the General Intelligence Factor in Language Models: A Psychometric Approach. (arXiv:2310.11616v1 [cs.CL])

    [http://arxiv.org/abs/2310.11616](http://arxiv.org/abs/2310.11616)

    本研究利用心理测量理论揭示了语言模型中的普遍智能因子g的存在，并发现了该因子解释模型性能方差的85%，为模型评估和开发提供了统一的指标。

    

    本研究采用心理测量理论，揭示了语言模型中普遍智能因子g的存在，并扩展了该理论在人类和某些动物物种中的应用。通过对两个大型数据集Open LLM Leaderboard（包含1,232个模型）和General Language Understanding Evaluation（GLUE）Leaderboard（包含88个模型）进行因子分析，我们发现了一个具有一维性和高度稳定性的g因子，可以解释模型性能方差的85%。研究还发现模型大小和g之间的中度相关性为0.48。在语言模型中发现g因子为模型评估提供了统一的指标，为更强大、基于g因子的模型能力评估开辟了新的途径。这些发现为从心理测量的角度理解和未来研究人工智能提供了基础，并对模型评估和开发具有实际意义。

    This study uncovers the factor of general intelligence, or g, in language models, extending the psychometric theory traditionally applied to humans and certain animal species. Utilizing factor analysis on two extensive datasets Open LLM Leaderboard with 1,232 models and General Language Understanding Evaluation (GLUE) Leaderboard with 88 models - we find compelling evidence for a unidimensional, highly stable g factor that accounts for 85% of the variance in model performance. The study also finds a moderate correlation of .48 between model size and g. The discovery of g in language models offers a unified metric for model evaluation and opens new avenues for more robust, g-based model ability assessment. These findings lay the foundation for understanding and future research on artificial general intelligence from a psychometric perspective and have practical implications for model evaluation and development.
    
[^4]: UPAR：一种受康德启发的促进框架，用于提升大型语言模型的能力

    UPAR: A Kantian-Inspired Prompting Framework for Enhancing Large Language Model Capabilities. (arXiv:2310.01441v1 [cs.CL])

    [http://arxiv.org/abs/2310.01441](http://arxiv.org/abs/2310.01441)

    UPAR提示框架模拟人类认知结构，在大型语言模型中提供了可理解和可检查的推理轨迹，增强了推理的可解释性和准确性，并为现有的提示技术提供了认识论基础。

    

    大型语言模型(LLMs)展示了令人印象深刻的推理能力，许多研究致力于通过提示提升这种能力。尽管有这些努力，统一的认识论基础仍然明显缺失。受康德的先验哲学启发，我们提出了UPAR提示框架，旨在在LLMs中模拟人类认知结构。UPAR框架分为四个阶段：“理解”、“计划”、“行动”和“反思”，使得能够从复杂背景中提取结构化信息，事先规划解决方案，按计划执行，并进行自我反思。这个结构显著增强了LLM推理的可解释性和准确性，产生了人类可理解和可检查的推理轨迹。此外，我们的工作为现有的提示技术提供了认识论基础，可能实现这些方法的系统集成。

    Large Language Models (LLMs) have demonstrated impressive inferential capabilities, with numerous research endeavors devoted to enhancing this capacity through prompting. Despite these efforts, a unified epistemological foundation is still conspicuously absent. Drawing inspiration from Kant's a priori philosophy, we propose the UPAR prompting framework, designed to emulate the structure of human cognition within LLMs. The UPAR framework is delineated into four phases: "Understand", "Plan", "Act", and "Reflect", enabling the extraction of structured information from complex contexts, prior planning of solutions, execution according to plan, and self-reflection. This structure significantly augments the explainability and accuracy of LLM inference, producing a human-understandable and inspectable inferential trajectory. Furthermore, our work offers an epistemological foundation for existing prompting techniques, allowing for a possible systematic integration of these methods. With GPT-4,
    
[^5]: 损失突然下降：语法习得、相变和MLM中的简化偏差

    Sudden Drops in the Loss: Syntax Acquisition, Phase Transitions, and Simplicity Bias in MLMs. (arXiv:2309.07311v1 [cs.CL])

    [http://arxiv.org/abs/2309.07311](http://arxiv.org/abs/2309.07311)

    本文通过对掩码语言模型中的语法习得进行案例研究，发现在训练的一个短暂窗口内，模型突然获得了语法注意结构(SAS)，并伴随着损失的陡峭下降。SAS对随后习得语言能力起到了重要的促进作用。

    

    自然语言处理(NLP)中的大多数可解释性研究侧重于理解完全训练模型的行为和特征。然而，通过观察训练过程的轨迹，可能才能获得对模型行为的某些洞察。在本文中，我们通过对掩码语言模型(MLMs)中的语法习得进行案例研究，展示了如何通过分析训练过程中可解释性的演化来加深我们对新兴行为的理解。具体而言，我们研究了语法注意结构(SAS)，这是MLMs中自然形成的一个特性，其中特定的Transformer头倾向于关注特定的句法关系。我们发现在训练的一个短暂窗口内，模型突然获得了SAS，并发现这个窗口与损失的陡峭下降同时发生。此外，SAS促使了随后对语言能力的习得。然后，我们通过引入一个正则化项来操纵训练过程中的SAS，来研究SAS的因果作用，并进行了实验证明。

    Most interpretability research in NLP focuses on understanding the behavior and features of a fully trained model. However, certain insights into model behavior may only be accessible by observing the trajectory of the training process. In this paper, we present a case study of syntax acquisition in masked language models (MLMs). Our findings demonstrate how analyzing the evolution of interpretable artifacts throughout training deepens our understanding of emergent behavior. In particular, we study Syntactic Attention Structure (SAS), a naturally emerging property of MLMs wherein specific Transformer heads tend to focus on specific syntactic relations. We identify a brief window in training when models abruptly acquire SAS and find that this window is concurrent with a steep drop in loss. Moreover, SAS precipitates the subsequent acquisition of linguistic capabilities. We then examine the causal role of SAS by introducing a regularizer to manipulate SAS during training, and demonstrate
    
[^6]: 大型语言模型作为优化器

    Large Language Models as Optimizers. (arXiv:2309.03409v1 [cs.LG])

    [http://arxiv.org/abs/2309.03409](http://arxiv.org/abs/2309.03409)

    本论文提出了一种简单有效的方法，利用大型语言模型(LLMs)作为优化器，通过自然语言描述优化任务。经过实验证明，该方法在线性回归和旅行推销员问题上表现出色，并且优化的最佳提示超过了人为设计的提示。

    

    优化是无处不在的。虽然基于导数的算法在各种问题上是强大的工具，但是没有梯度对许多实际应用提出了挑战。在这项工作中，我们提出了一种简单有效的方法，利用大型语言模型(LLMs)作为优化器，其中优化任务以自然语言形式描述。在每一次优化步骤中，LLM从包含先前生成的解与其值的提示中生成新的解，然后对新的解进行评估并添加到提示中，用于下一次优化步骤。我们首先展示了OPRO在线性回归和旅行推销员问题上的应用，然后转向提示优化，目标是找到能最大化任务准确性的指令。通过使用各种LLM，我们证明了OPRO优化的最佳提示在GSM8K上击败了人为设计的提示高达8%，在Big-Bench Hard任务上击败了人为设计的提示高达50%。

    Optimization is ubiquitous. While derivative-based algorithms have been powerful tools for various problems, the absence of gradient imposes challenges on many real-world applications. In this work, we propose Optimization by PROmpting (OPRO), a simple and effective approach to leverage large language models (LLMs) as optimizers, where the optimization task is described in natural language. In each optimization step, the LLM generates new solutions from the prompt that contains previously generated solutions with their values, then the new solutions are evaluated and added to the prompt for the next optimization step. We first showcase OPRO on linear regression and traveling salesman problems, then move on to prompt optimization where the goal is to find instructions that maximize the task accuracy. With a variety of LLMs, we demonstrate that the best prompts optimized by OPRO outperform human-designed prompts by up to 8% on GSM8K, and by up to 50% on Big-Bench Hard tasks.
    
[^7]: 基于注意力驱动的多模态融合：增强手语识别和翻译

    Attention-Driven Multi-Modal Fusion: Enhancing Sign Language Recognition and Translation. (arXiv:2309.01860v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2309.01860](http://arxiv.org/abs/2309.01860)

    本文提出了一种注意力驱动的多模态融合机制，通过将光流信息与RGB图像相结合，丰富了连续手语识别和翻译流程中的特征。该方法在手语识别任务中降低了WER 0.9，在翻译任务中提高了测试集上大多数BLEU分数约0.6。

    

    本文中，我们设计了一种机制，用于将多模态信息与现有的连续手语识别和翻译流程相结合。在我们的过程中，我们将光流信息与RGB图像结合，以丰富具有与运动相关信息的特征。该工作通过使用跨模态编码器研究了这种模态包含的可行性。我们使用的插件非常轻量级，并且不需要以端到端的方式为新模态包括一个单独的特征提取器。我们在手语识别和翻译中应用了这些改变，改善了每个任务的结果。我们在RWTH-PHOENIX-2014数据集上评估了性能，用于手语识别，并在RWTH-PHOENIX-2014T数据集上评估了翻译任务。在识别任务上，我们的方法将WER降低了0.9，在翻译任务上，我们的方法将大部分BLEU分数在测试集上提高了约0.6。

    In this paper, we devise a mechanism for the addition of multi-modal information with an existing pipeline for continuous sign language recognition and translation. In our procedure, we have incorporated optical flow information with RGB images to enrich the features with movement-related information. This work studies the feasibility of such modality inclusion using a cross-modal encoder. The plugin we have used is very lightweight and doesn't need to include a separate feature extractor for the new modality in an end-to-end manner. We have applied the changes in both sign language recognition and translation, improving the result in each case. We have evaluated the performance on the RWTH-PHOENIX-2014 dataset for sign language recognition and the RWTH-PHOENIX-2014T dataset for translation. On the recognition task, our approach reduced the WER by 0.9, and on the translation task, our approach increased most of the BLEU scores by ~0.6 on the test set.
    
[^8]: 追求多模态视觉语言模型中的基于实际的视觉空间推理

    Towards Grounded Visual Spatial Reasoning in Multi-Modal Vision Language Models. (arXiv:2308.09778v1 [cs.CV])

    [http://arxiv.org/abs/2308.09778](http://arxiv.org/abs/2308.09778)

    本文旨在研究多模态视觉语言模型在理解空间关系方面的能力，提出了细粒度组合的空间关系基础，并采用自底向上的方法评估空间关系推理任务的性能。

    

    随着大规模视觉和语言模型（VLMs）的进展，评估它们在各种视觉推理任务（如计数、指涉表达和一般的视觉问题回答）上的表现变得越来越重要。本文的重点是研究这些模型理解空间关系的能力。先前，人们尝试使用图像-文本匹配（Liu, Emerson, and Collier 2022) 或视觉问题回答任务来处理此问题，但都表现出性能不佳并且与人类性能存在较大差距。为了更好地理解差距，我们提出了细粒度组合的空间关系基础，并提出了一种自底向上的方法来对空间从句进行排名并评估空间关系推理任务的性能。我们建议通过结合和地面化物体对应的名词短语和它们的位置的证据来计算空间从句的最终排名。我们在代表性的视觉语言模型上展示了这种方法。

    With the advances in large scale vision-and-language models (VLMs) it is of interest to assess their performance on various visual reasoning tasks such as counting, referring expressions and general visual question answering. The focus of this work is to study the ability of these models to understanding spatial relations. Previously, this has been tackled using image-text matching (Liu, Emerson, and Collier 2022) or visual question answering task, both showing poor performance and a large gap compared to human performance. To better understand the gap, we present fine-grained compositional grounding of spatial relationships and propose a bottom up approach for ranking spatial clauses and evaluating the performance of spatial relationship reasoning task. We propose to combine the evidence from grounding noun phrases corresponding to objects and their locations to compute the final rank of the spatial clause. We demonstrate the approach on representative vision-language models (Tan and 
    
[^9]: 从像素到用户界面操作：通过图形用户界面学习遵循指令

    From Pixels to UI Actions: Learning to Follow Instructions via Graphical User Interfaces. (arXiv:2306.00245v1 [cs.LG])

    [http://arxiv.org/abs/2306.00245](http://arxiv.org/abs/2306.00245)

    本文提出了一种基于像素级别的预训练方法，建立了一种模拟人类概念界面和混合动作空间的代理，实现了在GUI指令遵循任务的MiniWob++基准测试中超越人类工作者的目标。

    

    先前为了构建操作图形用户界面（GUI）的数字化代理，大多数工作都依赖基于文本的表示（从HTML或其他结构化数据源派生），这些表示并不总是容易获取。这些输入表示通常与自定义的任务特定动作空间相关联。本文旨在创建使用与人类通常使用的相同概念界面-通过基于像素的屏幕截图和对应于键盘和鼠标操作的通用动作空间与数字世界交互的代理。在近期关于像素级预训练方法的基础上，我们首次展示了这样的代理在GUI指令遵循任务的MiniWob ++基准测试中能够超越人类工作者。

    Much of the previous work towards digital agents for graphical user interfaces (GUIs) has relied on text-based representations (derived from HTML or other structured data sources), which are not always readily available. These input representations have been often coupled with custom, task-specific action spaces. This paper focuses on creating agents that interact with the digital world using the same conceptual interface that humans commonly use -via pixel-based screenshots and a generic action space corresponding to keyboard and mouse actions. Building upon recent progress in pixel-based pretraining, we show, for the first time, that it is possible for such agents to outperform human crowdworkers on the MiniWob++ benchmark of GUI-based instruction following tasks.
    
[^10]: 后验解释可以提高语言模型的性能

    Post Hoc Explanations of Language Models Can Improve Language Models. (arXiv:2305.11426v1 [cs.CL])

    [http://arxiv.org/abs/2305.11426](http://arxiv.org/abs/2305.11426)

    本文提出了一种新的框架AMPLIFY，利用后验解释自动化生成原因，并在多个数据集和任务上显著提高现有语言模型的性能。

    

    大型语言模型在执行复杂任务方面表现出了非凡的能力。最近的研究显示，在上下文学习过程中加入人类注释的原理（例如，思维链提示）可以显著提高这些模型的性能，特别是在需要推理能力的任务上。然而，这样的原理加入在可扩展性方面存在挑战，因为这需要高度的人工参与。本文提出了一种新框架，即通过利用后验解释的上下文学习来放大模型性能，来解决上述挑战。为此，我们利用后验解释方法的结果，该方法输出称为属性分数（解释）的值，用于捕获每个输入特征对模型预测的影响。更具体地说，我们构建了自动化的自然语言原理，其中包含从属性分数中获得的信息，以便用户可以更好地理解模型的决策。实验结果表明，AMPLIFY可以在多个数据集和任务上显著提高现有语言模型的性能。

    Large Language Models (LLMs) have demonstrated remarkable capabilities in performing complex tasks. Moreover, recent research has shown that incorporating human-annotated rationales (e.g., Chain-of- Thought prompting) during in-context learning can significantly enhance the performance of these models, particularly on tasks that require reasoning capabilities. However, incorporating such rationales poses challenges in terms of scalability as this requires a high degree of human involvement. In this work, we present a novel framework, Amplifying Model Performance by Leveraging In-Context Learning with Post Hoc Explanations (AMPLIFY), which addresses the aforementioned challenges by automating the process of rationale generation. To this end, we leverage post hoc explanation methods which output attribution scores (explanations) capturing the influence of each of the input features on model predictions. More specifically, we construct automated natural language rationales that embed insi
    
[^11]: 通过可编辑的逐步解释实现交互式文本转SQL

    Interactive Text-to-SQL Generation via Editable Step-by-Step Explanations. (arXiv:2305.07372v1 [cs.DB])

    [http://arxiv.org/abs/2305.07372](http://arxiv.org/abs/2305.07372)

    本文介绍一种交互机制，允许用户直接编辑一步步解释错误SQL以修复SQL错误，实验证明方法提高了31.6％的执行准确性。用户研究表明，该方法帮助用户以更少的时间和更高的信心解决了更多的SQL任务。

    

    关系数据库在大数据时代扮演着重要角色。然而，非专家很难完全释放关系数据库的分析能力，因为他们不熟悉SQL等数据库语言。许多技术已被提出自然语言自动生成SQL，但它们存在以下两个问题：（1）对于复杂查询它们仍会犯很多错误，（2）它们不提供一种灵活的方式，让非专家用户验证和改进不正确的查询。为了解决这些问题，我们引入了一种新的交互机制，允许用户直接编辑一步步解释错误SQL以修复SQL错误。在Spider基准测试上的实验证明，我们的方法至少比三种最先进方法在执行准确性方面提高了31.6％。24名参与者的用户研究进一步表明，我们的方法帮助用户以更少的时间和更高的信心解决了更多的SQL任务。

    Relational databases play an important role in this Big Data era. However, it is challenging for non-experts to fully unleash the analytical power of relational databases, since they are not familiar with database languages such as SQL. Many techniques have been proposed to automatically generate SQL from natural language, but they suffer from two issues: (1) they still make many mistakes, particularly for complex queries, and (2) they do not provide a flexible way for non-expert users to validate and refine the incorrect queries. To address these issues, we introduce a new interaction mechanism that allows users directly edit a step-by-step explanation of an incorrect SQL to fix SQL errors. Experiments on the Spider benchmark show that our approach outperforms three SOTA approaches by at least 31.6% in terms of execution accuracy. A user study with 24 participants further shows that our approach helped users solve significantly more SQL tasks with less time and higher confidence, demo
    
[^12]: 动态上下文图形实现对万物知识图谱的会话语义解析

    Conversational Semantic Parsing using Dynamic Context Graphs. (arXiv:2305.06164v1 [cs.CL])

    [http://arxiv.org/abs/2305.06164](http://arxiv.org/abs/2305.06164)

    本论文提出了一种新的方法，使用动态创建的子图表示话语及上下文的信息来进行会话语义解析，并利用图形神经网络编码，可表示大量看不见的节点，比静态方法更为优越。

    

    本文考虑了在拥有数百万个实体和数千种关系类型的通用知识图谱上进行会话语义解析的任务。我们致力于开发能够交互地将用户语言映射为可执行逻辑形式（例如SPARQL）的模型，同时考虑到对话历史的上下文。我们的关键想法是通过一个动态创建的子图来表示有关话语及其上下文的信息，即每个话语的节点数会发生变化。而且，我们利用子图的基本结构，而不是将其视为序列，使用图形神经网络进行编码，从而进一步允许我们表示大量（看不见的）节点。实验结果表明，动态建模上下文优于静态方法，可在各个方面（即简单和复杂问题）提高性能。我们的结果进一步证实，模型化上下文结构比仅考虑单个话语更好。

    In this paper we consider the task of conversational semantic parsing over general purpose knowledge graphs (KGs) with millions of entities, and thousands of relation-types. We are interested in developing models capable of interactively mapping user utterances into executable logical forms (e.g., SPARQL) in the context of the conversational history. Our key idea is to represent information about an utterance and its context via a subgraph which is created dynamically, i.e., the number of nodes varies per utterance. Moreover, rather than treating the subgraph as a sequence we exploit its underlying structure, and thus encode it using a graph neural network which further allows us to represent a large number of (unseen) nodes. Experimental results show that modeling context dynamically is superior to static approaches, delivering performance improvements across the board (i.e., for simple and complex questions). Our results further confirm that modeling the structure of context is bette
    
[^13]: 大型语言模型能否改变计算社会科学？

    Can Large Language Models Transform Computational Social Science?. (arXiv:2305.03514v1 [cs.CL])

    [http://arxiv.org/abs/2305.03514](http://arxiv.org/abs/2305.03514)

    本文研究了大型语言模型作为计算社会科学工具的潜力。虽然在分类任务上没有优势，但在自由形式编码任务上表现优异，今后可以作为零-shot检测工具进行使用，

    

    ChatGPT等大型语言模型(LLMs)能够成功地在许多语言处理任务中进行零-shot操作（无需训练数据）。如果这种能力也适用于对说服力和政治意识形态等社会现象的编码，那么LLMs就可以有效地改变计算社会科学(CSS)。本研究提供了使用LLMs作为CSS工具的路线图。为此，我们提供了一组优秀的提示实践以及一个广泛的评估流程，以测量13种语言模型在24个代表性的CSS基准测试上的零-shot性能。在分类任务上，LLMs无法超越最佳微调模型，但仍然与人类达成了公平的协议水平。在自由形式的编码任务（生成）上，LLMs生成的解释常常超过了工作者的黄金参考的质量。我们得出结论，今天的LLMs可以通过两种方式从根本上增强CSS研究流程：(1)作为零-shot检测工具进行无缝工作。

    Large Language Models (LLMs) like ChatGPT are capable of successfully performing many language processing tasks zero-shot (without the need for training data). If this capacity also applies to the coding of social phenomena like persuasiveness and political ideology, then LLMs could effectively transform Computational Social Science (CSS). This work provides a road map for using LLMs as CSS tools. Towards this end, we contribute a set of prompting best practices and an extensive evaluation pipeline to measure the zero-shot performance of 13 language models on 24 representative CSS benchmarks. On taxonomic labeling tasks (classification), LLMs fail to outperform the best fine-tuned models but still achieve fair levels of agreement with humans. On free-form coding tasks (generation), LLMs produce explanations that often exceed the quality of crowdworkers' gold references. We conclude that today's LLMs can radically augment the CSS research pipeline in two ways: (1) serving as zero-shot d
    
[^14]: ZeroNLG: 将领域对齐和自编码用于零样本多模态和多语言自然语言生成

    ZeroNLG: Aligning and Autoencoding Domains for Zero-Shot Multimodal and Multilingual Natural Language Generation. (arXiv:2303.06458v1 [cs.CL])

    [http://arxiv.org/abs/2303.06458](http://arxiv.org/abs/2303.06458)

    ZeroNLG是一个零样本学习框架，可以处理多个NLG任务，包括图像到文本、视频到文本和文本到文本，跨越英语、中文、德语和法语。它不需要任何标记的下游对进行训练，通过将不同的领域投影到共享的公共潜在空间中的相应坐标，桥接不同领域之间的差异。

    ZeroNLG is a zero-shot learning framework that can handle multiple NLG tasks, including image-to-text, video-to-text, and text-to-text, across English, Chinese, German, and French. It does not require any labeled downstream pairs for training, and bridges the differences between different domains by projecting them to corresponding coordinates in a shared common latent space.

    自然语言生成（NLG）接受以图像、视频或文本形式的输入数据，并生成相应的自然语言文本作为输出。现有的NLG方法主要采用监督方法，并且严重依赖于耦合的数据到文本对。然而，对于许多有针对性的场景和非英语语言，往往没有足够数量的标记数据。为了放松对下游任务标记数据的依赖性，我们提出了一个直观有效的零样本学习框架ZeroNLG，它可以处理多个NLG任务，包括图像到文本（图像字幕）、视频到文本（视频字幕）和文本到文本（神经机器翻译），跨越英语、中文、德语和法语在一个统一的框架内。ZeroNLG不需要任何标记的下游对进行训练。在训练期间，ZeroNLG（i）将不同的领域（跨模态和语言）投影到共享的公共潜在空间中的相应坐标；（ii）桥接差异

    Natural Language Generation (NLG) accepts input data in the form of images, videos, or text and generates corresponding natural language text as output. Existing NLG methods mainly adopt a supervised approach and rely heavily on coupled data-to-text pairs. However, for many targeted scenarios and for non-English languages, sufficient quantities of labeled data are often not available. To relax the dependency on labeled data of downstream tasks, we propose an intuitive and effective zero-shot learning framework, ZeroNLG, which can deal with multiple NLG tasks, including image-to-text (image captioning), video-to-text (video captioning), and text-to-text (neural machine translation), across English, Chinese, German, and French within a unified framework. ZeroNLG does not require any labeled downstream pairs for training. During training, ZeroNLG (i) projects different domains (across modalities and languages) to corresponding coordinates in a shared common latent space; (ii) bridges diff
    

