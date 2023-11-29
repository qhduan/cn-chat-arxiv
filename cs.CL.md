# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [People Make Better Edits: Measuring the Efficacy of LLM-Generated Counterfactually Augmented Data for Harmful Language Detection.](http://arxiv.org/abs/2311.01270) | 本论文通过自动生成的反事实增强数据（CADs）与手动生成的CADs进行比较，评估它们在提高模型鲁棒性方面的效果。结果显示，手动生成的CADs仍然是最有效的方法。 |
| [^2] | [Breaking Language Barriers in Multilingual Mathematical Reasoning: Insights and Observations.](http://arxiv.org/abs/2310.20246) | 本文首次探索并训练了强大的多语言数学推理模型，通过使用翻译构建了多语言数据集，并提出了各种训练策略来构建强大的模型。实验证实发现在多语言训练中，将目标语言的翻译与原始语言的表示结合起来以及交替训练和多语言模型的自举可以提高模型的性能。此外，模型在处理低频词和长句子方面仍面临挑战。 |
| [^3] | [CodeChain: Towards Modular Code Generation Through Chain of Self-revisions with Representative Sub-modules.](http://arxiv.org/abs/2310.08992) | CodeChain是一种通过代表性子模块的自我修订链路引导模块化代码生成的新框架，旨在解决大型语言模型在解决复杂编程任务方面的挑战。 |
| [^4] | [LoftQ: LoRA-Fine-Tuning-Aware Quantization for Large Language Models.](http://arxiv.org/abs/2310.08659) | 本论文提出了LoftQ：一种针对大型语言模型的LoRA精调感知量化框架。该框架同时对LLM进行量化，并为LoRA精调找到适当的低秩初始化，以缓解量化模型和全精度模型之间的差异，并显著提高了下游任务的泛化能力。 |
| [^5] | [A Brief History of Prompt: Leveraging Language Models.](http://arxiv.org/abs/2310.04438) | 这篇论文全面探讨了提示工程和生成在自然语言处理领域的演进历程，包括早期语言模型和信息检索系统，注意力机制的引入，强化学习技术的应用以解决偏见和暴露偏差等问题。还讨论了微调策略、控制代码和基于模板的生成的重大贡献，以及公平性、人工智能与人类合作和低资源适应的重要性。 |
| [^6] | [On the Performance of Multimodal Language Models.](http://arxiv.org/abs/2310.03211) | 本研究对不同多模态指导调优方法进行比较分析，并评估其在复杂推理、对话、图像描述等任务中的性能。通过基准测试和消融实验，为将多模态能力融入语言模型提供了关键见解。 |
| [^7] | [Towards End-to-End Embodied Decision Making via Multi-modal Large Language Model: Explorations with GPT4-Vision and Beyond.](http://arxiv.org/abs/2310.02071) | 本研究通过探索多模态大型语言模型在代理的具身决策中的应用潜力，提出了一个新的评估基准PCA-EVAL，并引入了一个多代理协作框架HOLMES，以提高决策能力。研究发现GPT4-Vision模型在端到端的具身决策中表现最佳。 |
| [^8] | [Extending CAM-based XAI methods for Remote Sensing Imagery Segmentation.](http://arxiv.org/abs/2310.01837) | 这篇论文扩展了基于CAM的可解释AI方法，使其适用于遥感图像分割。当前AI模型在高分辨率卫星图像上训练时缺乏透明度和可解释性，本文通过改进XAI分类算法，提供了解释图像分割的手段。 |
| [^9] | [Certifying LLM Safety against Adversarial Prompting.](http://arxiv.org/abs/2309.02705) | 本研究提出了首个具有可验证安全保证的框架——消除和检查，用于对抗敌对提示。通过逐个消除标记并使用安全过滤器检查生成的子序列，确保任何敌对修改的有害输入提示都能被正确标识为有害。 |
| [^10] | [Towards Populating Generalizable Engineering Design Knowledge.](http://arxiv.org/abs/2307.06985) | 这项研究提出了一种从专利文件中提取工程设计知识的方法，通过构建知识图来填充通用设计知识，并与现有方法进行了比较。 |
| [^11] | [In-Context Demonstration Selection with Cross Entropy Difference.](http://arxiv.org/abs/2305.14726) | 本论文提出了一种基于交叉熵差异的方法，用于在大型语言模型中选择最佳的上下文演示来提高零-shot任务性能。 |
| [^12] | [Pre-training Language Models for Comparative Reasoning.](http://arxiv.org/abs/2305.14457) | 本文提出一种预训练语言模型的新框架，旨在增强其在比较推理方面的能力。通过使用可扩展的基于文本实体比较数据的方法和新的预训练任务，该框架得到了显著的结果。 |

# 详细

[^1]: 人类进行更好的编辑：衡量使用LLM生成的反事实增强数据在有害语言检测中的效果

    People Make Better Edits: Measuring the Efficacy of LLM-Generated Counterfactually Augmented Data for Harmful Language Detection. (arXiv:2311.01270v1 [cs.CL])

    [http://arxiv.org/abs/2311.01270](http://arxiv.org/abs/2311.01270)

    本论文通过自动生成的反事实增强数据（CADs）与手动生成的CADs进行比较，评估它们在提高模型鲁棒性方面的效果。结果显示，手动生成的CADs仍然是最有效的方法。

    

    自然语言处理模型在许多重要的社会计算任务中被使用，如检测性别歧视、种族歧视或其他仇恨内容。因此，这些模型对虚假特征的鲁棒性至关重要。过去的工作尝试解决这些虚假特征问题，其中包括反事实增强数据（CADs）的训练数据增强方法。CADs对现有的训练数据进行最小改动并翻转标签；在其上进行训练可能减少模型对虚假特征的依赖。然而，手动生成CADs可能耗时且昂贵。因此，在这项工作中，我们评估了是否可以使用生成型自然语言处理模型自动化这个任务。我们使用Polyjuice、ChatGPT和Flan-T5自动生成CADs，并与手动生成的CADs进行比较，评估它们在提高模型鲁棒性方面的实用性。通过在多个领域外测试集上测试模型的性能以及每个数据点的有效性，我们的结果表明，尽管手动CADs仍然是最有效的方法。

    NLP models are used in a variety of critical social computing tasks, such as detecting sexist, racist, or otherwise hateful content. Therefore, it is imperative that these models are robust to spurious features. Past work has attempted to tackle such spurious features using training data augmentation, including Counterfactually Augmented Data (CADs). CADs introduce minimal changes to existing training data points and flip their labels; training on them may reduce model dependency on spurious features. However, manually generating CADs can be time-consuming and expensive. Hence in this work, we assess if this task can be automated using generative NLP models. We automatically generate CADs using Polyjuice, ChatGPT, and Flan-T5, and evaluate their usefulness in improving model robustness compared to manually-generated CADs. By testing both model performance on multiple out-of-domain test sets and individual data point efficacy, our results show that while manual CADs are still the most e
    
[^2]: 在多语言数学推理中打破语言障碍：见解与观察

    Breaking Language Barriers in Multilingual Mathematical Reasoning: Insights and Observations. (arXiv:2310.20246v1 [cs.CL])

    [http://arxiv.org/abs/2310.20246](http://arxiv.org/abs/2310.20246)

    本文首次探索并训练了强大的多语言数学推理模型，通过使用翻译构建了多语言数据集，并提出了各种训练策略来构建强大的模型。实验证实发现在多语言训练中，将目标语言的翻译与原始语言的表示结合起来以及交替训练和多语言模型的自举可以提高模型的性能。此外，模型在处理低频词和长句子方面仍面临挑战。

    

    现有研究主要集中在开发适用于单语言中的数学推理的强大语言学习模型（LLM），在多语言环境下保持效果的研究很少。为了弥补这一差距，本文首次探索和训练强大的多语言数学推理（xMR）LLM。首先，通过利用翻译，我们构建了第一个包含十种不同语言的多语言数学推理指导数据集MGSM8KInstruct，从而解决了xMR任务中训练数据稀缺的问题。根据收集的数据集，我们提出了不同的训练策略来构建强大的xMR LLMs，被命名为MathOctopus，在几次训练中表现出优于传统开源LLMs和ChatGPT的能力。值得注意的是，MathOctopus-13B在MGSM测试集上达到了47.6%的准确率，超过了ChatGPT的46.3%。除了显著的结果，我们还从大量的实验证实中发现了一些重要的观察和见解：（1）在多语言上进行训练时，最好将目标语言的翻译与原始语言的表示结合起来。 （2）交替训练和多语言模型的自举有助于提高模型的表现。 （3）模型对于低频词和长句子的处理是挑战的，需要进一步改进。

    Existing research predominantly focuses on developing powerful language learning models (LLMs) for mathematical reasoning within monolingual languages, with few explorations in preserving efficacy in a multilingual context. To bridge this gap, this paper pioneers exploring and training powerful Multilingual Math Reasoning (xMR) LLMs. Firstly, by utilizing translation, we construct the first multilingual math reasoning instruction dataset, MGSM8KInstruct, encompassing ten distinct languages, thus addressing the issue of training data scarcity in xMR tasks. Based on the collected dataset, we propose different training strategies to build powerful xMR LLMs, named MathOctopus, notably outperform conventional open-source LLMs and exhibit superiority over ChatGPT in few-shot scenarios. Notably, MathOctopus-13B reaches 47.6% accuracy which exceeds ChatGPT 46.3% on MGSM testset. Beyond remarkable results, we unearth several pivotal observations and insights from extensive experiments: (1) When
    
[^3]: CodeChain: 通过代表性子模块的自我修订链路实现模块化代码生成

    CodeChain: Towards Modular Code Generation Through Chain of Self-revisions with Representative Sub-modules. (arXiv:2310.08992v1 [cs.AI])

    [http://arxiv.org/abs/2310.08992](http://arxiv.org/abs/2310.08992)

    CodeChain是一种通过代表性子模块的自我修订链路引导模块化代码生成的新框架，旨在解决大型语言模型在解决复杂编程任务方面的挑战。

    

    大型语言模型（LLM）已经在解决简单编程任务方面非常熟练，比如在HumanEval或MBPP基准测试中的任务。然而，对于更复杂和具有竞争性的编程任务，这些模型仍然面临挑战，可能是因为它们倾向于生成作为整体代码块而不是将其分解为逻辑子任务和子模块。另一方面，有经验的程序员本能地编写具有抽象概念的模块化代码来解决复杂任务，通常会重复使用之前开发的模块。为了解决这一差距，我们提出了CodeChain，一种通过代表性子模块的自我修订链路引导模块化代码生成的新框架。具体而言，CodeChain首先通过链式思考提示指导LLM生成模块化代码。然后，它通过迭代两个步骤实施自我修订链路：1）额外...

    Large Language Models (LLMs) have already become quite proficient at solving simpler programming tasks like those in HumanEval or MBPP benchmarks. However, solving more complex and competitive programming tasks is still quite challenging for these models - possibly due to their tendency to generate solutions as monolithic code blocks instead of decomposing them into logical sub-tasks and sub-modules. On the other hand, experienced programmers instinctively write modularized code with abstraction for solving complex tasks, often reusing previously developed modules. To address this gap, we propose CodeChain, a novel framework for inference that elicits modularized code generation through a chain of self-revisions, each being guided by some representative sub-modules generated in previous iterations. Concretely, CodeChain first instructs the LLM to generate modularized codes through chain-of-thought prompting. Then it applies a chain of self-revisions by iterating the two steps: 1) extra
    
[^4]: LoftQ: LoRA-Fine-Tuning-Aware Quantization for Large Language Models

    LoftQ: LoRA-Fine-Tuning-Aware Quantization for Large Language Models. (arXiv:2310.08659v1 [cs.CL])

    [http://arxiv.org/abs/2310.08659](http://arxiv.org/abs/2310.08659)

    本论文提出了LoftQ：一种针对大型语言模型的LoRA精调感知量化框架。该框架同时对LLM进行量化，并为LoRA精调找到适当的低秩初始化，以缓解量化模型和全精度模型之间的差异，并显著提高了下游任务的泛化能力。

    

    量化是为大型语言模型提供服务的不可或缺的技术，并最近被应用于LoRA精调中。本文关注在预训练模型上同时应用量化和LoRA精调的场景。在这种情况下，常常观察到完整精调和量化加LoRA精调方法之间在下游任务表现上存在一致的差距。为了解决这个问题，我们提出了LoftQ（LoRA-Fine-Tuning-aware Quantization）——一种新的量化框架，用于同时对LLM进行量化，并找到适当的低秩初始化来进行LoRA精调。这种初始化减轻了量化模型和全精度模型之间的差异，并显著提高了下游任务的泛化能力。我们在自然语言理解、问答、摘要和自然语言生成任务上评估了我们的方法。实验证明，我们的方法非常有效，在性能上优于现有的方法。

    Quantization is an indispensable technique for serving Large Language Models (LLMs) and has recently found its way into LoRA fine-tuning. In this work we focus on the scenario where quantization and LoRA fine-tuning are applied together on a pre-trained model. In such cases it is common to observe a consistent gap in the performance on downstream tasks between full fine-tuning and quantization plus LoRA fine-tuning approach. In response, we propose LoftQ (LoRA-Fine-Tuning-aware Quantization), a novel quantization framework that simultaneously quantizes an LLM and finds a proper low-rank initialization for LoRA fine-tuning. Such an initialization alleviates the discrepancy between the quantized and full-precision model and significantly improves the generalization in downstream tasks. We evaluate our method on natural language understanding, question answering, summarization, and natural language generation tasks. Experiments show that our method is highly effective and outperforms exis
    
[^5]: 一份关于提示工程的简要历史: 利用语言模型 (arXiv:2310.04438v1 [cs.CL])

    A Brief History of Prompt: Leveraging Language Models. (arXiv:2310.04438v1 [cs.CL])

    [http://arxiv.org/abs/2310.04438](http://arxiv.org/abs/2310.04438)

    这篇论文全面探讨了提示工程和生成在自然语言处理领域的演进历程，包括早期语言模型和信息检索系统，注意力机制的引入，强化学习技术的应用以解决偏见和暴露偏差等问题。还讨论了微调策略、控制代码和基于模板的生成的重大贡献，以及公平性、人工智能与人类合作和低资源适应的重要性。

    

    本论文全面探讨了在自然语言处理（NLP）领域中提示工程和生成的演进历程。从早期的语言模型和信息检索系统开始，我们追溯了这些年来塑造提示工程的关键发展。2015年引入的注意力机制彻底改变了语言理解，推动了可控性和上下文感知的进步。随后在强化学习技术方面的突破进一步增强了提示工程，解决了暴露偏差和生成文本中的偏见等问题。我们重点考察了2018年和2019年的重大贡献，集中在微调策略、控制代码和基于模板的生成上。本论文还讨论了公平性、人工智能与人类的合作以及低资源适应的日益重要性。在2020年和2021年，上下文提示和迁移学习变得突出，而2022年和2023年见证了...

    This paper presents a comprehensive exploration of the evolution of prompt engineering and generation in the field of natural language processing (NLP). Starting from the early language models and information retrieval systems, we trace the key developments that have shaped prompt engineering over the years. The introduction of attention mechanisms in 2015 revolutionized language understanding, leading to advancements in controllability and context-awareness. Subsequent breakthroughs in reinforcement learning techniques further enhanced prompt engineering, addressing issues like exposure bias and biases in generated text. We examine the significant contributions in 2018 and 2019, focusing on fine-tuning strategies, control codes, and template-based generation. The paper also discusses the growing importance of fairness, human-AI collaboration, and low-resource adaptation. In 2020 and 2021, contextual prompting and transfer learning gained prominence, while 2022 and 2023 witnessed the e
    
[^6]: 关于多模态语言模型性能的研究

    On the Performance of Multimodal Language Models. (arXiv:2310.03211v1 [cs.CL])

    [http://arxiv.org/abs/2310.03211](http://arxiv.org/abs/2310.03211)

    本研究对不同多模态指导调优方法进行比较分析，并评估其在复杂推理、对话、图像描述等任务中的性能。通过基准测试和消融实验，为将多模态能力融入语言模型提供了关键见解。

    

    在独立预训练的视觉编码器通过模型嫁接的方式整合到大型语言模型中后，多模态语言模型展现了有望应用于各种下游任务的零样本泛化能力。这项研究对不同的多模态指导调优方法进行了比较分析，并评估了它们在复杂推理、对话、图像描述、多项选择题和二分类等任务中的性能。通过严格的基准测试和消融实验，我们揭示了在将多模态能力融入大型语言模型时指导架构选择的关键见解。然而，当前的方法存在局限性，它们没有足够地解决多样化的多模态指导数据的需求。

    Instruction-tuned large language models (LLMs) have demonstrated promising zero-shot generalization capabilities across various downstream tasks. Recent research has introduced multimodal capabilities to LLMs by integrating independently pretrained vision encoders through model grafting. These multimodal variants undergo instruction tuning, similar to LLMs, enabling effective zero-shot generalization for multimodal tasks. This study conducts a comparative analysis of different multimodal instruction tuning approaches and evaluates their performance across a range of tasks, including complex reasoning, conversation, image captioning, multiple-choice questions (MCQs), and binary classification. Through rigorous benchmarking and ablation experiments, we reveal key insights for guiding architectural choices when incorporating multimodal capabilities into LLMs. However, current approaches have limitations; they do not sufficiently address the need for a diverse multimodal instruction datase
    
[^7]: 通过多模态大型语言模型实现端到端的具身决策

    Towards End-to-End Embodied Decision Making via Multi-modal Large Language Model: Explorations with GPT4-Vision and Beyond. (arXiv:2310.02071v2 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2310.02071](http://arxiv.org/abs/2310.02071)

    本研究通过探索多模态大型语言模型在代理的具身决策中的应用潜力，提出了一个新的评估基准PCA-EVAL，并引入了一个多代理协作框架HOLMES，以提高决策能力。研究发现GPT4-Vision模型在端到端的具身决策中表现最佳。

    

    本研究探索了多模态大型语言模型（MLLMs）在改进代理的具身决策过程中的潜力。尽管由于其先进的推理能力和广泛的世界知识，大型语言模型（LLMs）被广泛使用，但像GPT4-Vision这样的MLLM提供了增强的视觉理解和推理能力。我们研究了最先进的MLLMs能否以端到端的方式处理具身决策，并且LLMs和MLLMs之间的协作是否能增强决策能力。为了回答这些问题，我们引入了一个名为PCA-EVAL的新基准，该基准从感知、认知和行动的角度评估具身决策。此外，我们提出了HOLMES，一个多代理协作框架，允许LLMs利用MLLMs和APIs获取多模态信息以进行明智的决策。我们在我们的基准上比较了端到端的具身决策和HOLMES，并发现GPT4-Vision模型的性能最优。

    In this study, we explore the potential of Multimodal Large Language Models (MLLMs) in improving embodied decision-making processes for agents. While Large Language Models (LLMs) have been widely used due to their advanced reasoning skills and vast world knowledge, MLLMs like GPT4-Vision offer enhanced visual understanding and reasoning capabilities. We investigate whether state-of-the-art MLLMs can handle embodied decision-making in an end-to-end manner and whether collaborations between LLMs and MLLMs can enhance decision-making. To address these questions, we introduce a new benchmark called PCA-EVAL, which evaluates embodied decision-making from the perspectives of Perception, Cognition, and Action. Additionally, we propose HOLMES, a multi-agent cooperation framework that allows LLMs to leverage MLLMs and APIs to gather multimodal information for informed decision-making. We compare end-to-end embodied decision-making and HOLMES on our benchmark and find that the GPT4-Vision model 
    
[^8]: 扩展基于CAM的可解释AI方法用于遥感图像分割

    Extending CAM-based XAI methods for Remote Sensing Imagery Segmentation. (arXiv:2310.01837v1 [cs.CV])

    [http://arxiv.org/abs/2310.01837](http://arxiv.org/abs/2310.01837)

    这篇论文扩展了基于CAM的可解释AI方法，使其适用于遥感图像分割。当前AI模型在高分辨率卫星图像上训练时缺乏透明度和可解释性，本文通过改进XAI分类算法，提供了解释图像分割的手段。

    

    当前的基于AI的方法无法对所使用的数据、提取的特征和预测/推理操作提供可理解的物理解释。因此，使用高分辨率卫星图像训练的深度学习模型缺乏透明度和可解释性，只能被视为一个黑盒子，这限制了它们的广泛采用。专家需要帮助理解AI模型的复杂行为和基础决策过程。可解释人工智能（XAI）领域是一个新兴领域，提供了确保AI模型稳健、实用和可信赖部署的手段。已有一些XAI技术被提出用于图像分类任务，而对于图像分割的解释则基本上没有被探索。本文通过改进最新的XAI分类算法，并使其适用于多类图像分割，以弥补这一差距，其中我们主要关注高分辨率卫星图像中的建筑物分割。

    Current AI-based methods do not provide comprehensible physical interpretations of the utilized data, extracted features, and predictions/inference operations. As a result, deep learning models trained using high-resolution satellite imagery lack transparency and explainability and can be merely seen as a black box, which limits their wide-level adoption. Experts need help understanding the complex behavior of AI models and the underlying decision-making process. The explainable artificial intelligence (XAI) field is an emerging field providing means for robust, practical, and trustworthy deployment of AI models. Several XAI techniques have been proposed for image classification tasks, whereas the interpretation of image segmentation remains largely unexplored. This paper offers to bridge this gap by adapting the recent XAI classification algorithms and making them usable for muti-class image segmentation, where we mainly focus on buildings' segmentation from high-resolution satellite 
    
[^9]: 证明LLM对抗敌对提示的安全性

    Certifying LLM Safety against Adversarial Prompting. (arXiv:2309.02705v1 [cs.CL])

    [http://arxiv.org/abs/2309.02705](http://arxiv.org/abs/2309.02705)

    本研究提出了首个具有可验证安全保证的框架——消除和检查，用于对抗敌对提示。通过逐个消除标记并使用安全过滤器检查生成的子序列，确保任何敌对修改的有害输入提示都能被正确标识为有害。

    

    为了确保语言模型的输出安全，公开使用的大型语言模型（LLM）引入了所谓的“模型对齐”防护措施。一个对齐的语言模型应该拒绝用户的请求生成有害内容。然而，这种安全措施容易受到敌对提示的攻击，敌对提示包含恶意设计的标记序列，以规避模型的安全防护并导致生成有害内容。在这项工作中，我们介绍了可验证安全保证的第一个对抗敌对提示的框架——消除和检查。我们逐个消除标记，并使用安全过滤器检查生成的子序列。如果安全过滤器检测到任何子序列或输入提示有害，我们的过程将将输入提示标记为有害。这保证了对于某个特定大小的有害输入提示的任何敌对修改也将被标记为有害。我们对抗三种攻击模式：i)敌对后缀，即附加敌对序列…

    Large language models (LLMs) released for public use incorporate guardrails to ensure their output is safe, often referred to as "model alignment." An aligned language model should decline a user's request to produce harmful content. However, such safety measures are vulnerable to adversarial prompts, which contain maliciously designed token sequences to circumvent the model's safety guards and cause it to produce harmful content. In this work, we introduce erase-and-check, the first framework to defend against adversarial prompts with verifiable safety guarantees. We erase tokens individually and inspect the resulting subsequences using a safety filter. Our procedure labels the input prompt as harmful if any subsequences or the input prompt are detected as harmful by the filter. This guarantees that any adversarial modification of a harmful prompt up to a certain size is also labeled harmful. We defend against three attack modes: i) adversarial suffix, which appends an adversarial seq
    
[^10]: 迈向填充通用工程设计知识的方法

    Towards Populating Generalizable Engineering Design Knowledge. (arXiv:2307.06985v1 [cs.CL])

    [http://arxiv.org/abs/2307.06985](http://arxiv.org/abs/2307.06985)

    这项研究提出了一种从专利文件中提取工程设计知识的方法，通过构建知识图来填充通用设计知识，并与现有方法进行了比较。

    

    为了填充通用工程设计知识，我们提出了一种从专利文件中提取head entity :: relationship :: tail entity形式事实的方法。这些事实可以在专利文件内部和跨文件之间组合形成知识图，用作表示和存储设计知识的方案。现有的工程设计文献中的方法通常利用一组预定义的关系来填充统计近似而非事实的三元组。在我们的方法中，我们训练一个标记器来识别句子中的实体和关系。在确定了一对实体后，我们训练另一个标记器来识别特定表示这对实体之间关系的关系标记。为了训练这些标记器，我们手动构建了一个包含44,227个句子和相应事实的数据集。我们还将该方法的性能与通常推荐的方法进行了比较，其中我们预.

    Aiming to populate generalizable engineering design knowledge, we propose a method to extract facts of the form head entity :: relationship :: tail entity from sentences found in patent documents. These facts could be combined within and across patent documents to form knowledge graphs that serve as schemes for representing as well as storing design knowledge. Existing methods in engineering design literature often utilise a set of predefined relationships to populate triples that are statistical approximations rather than facts. In our method, we train a tagger to identify both entities and relationships from a sentence. Given a pair of entities thus identified, we train another tagger to identify the relationship tokens that specifically denote the relationship between the pair. For training these taggers, we manually construct a dataset of 44,227 sentences and corresponding facts. We also compare the performance of the method against typically recommended approaches, wherein, we pre
    
[^11]: 基于交叉熵差异的上下文演示选择方法

    In-Context Demonstration Selection with Cross Entropy Difference. (arXiv:2305.14726v1 [cs.CL])

    [http://arxiv.org/abs/2305.14726](http://arxiv.org/abs/2305.14726)

    本论文提出了一种基于交叉熵差异的方法，用于在大型语言模型中选择最佳的上下文演示来提高零-shot任务性能。

    

    大型语言模型（LLMs）可以利用上下文演示来提高零-shot任务的性能。然而，选择最佳的上下文示例很具挑战性，因为模型性能可能因所选示例而异。我们提出了一种基于交叉熵差异（CED）的方法，用于选择上下文演示。我们的方法基于这样一个观察结果：在有限调整了这些演示的语言模型上，上下文演示的有效性与测试示例的困惑度呈负相关。我们利用参数有效的微调在训练数据上训练小型模型，以计算测试示例和每个候选上下文演示之间的交叉熵差异。该指标用于为每个测试输入独立地排名和选择上下文演示。我们在一个混合域数据集上评估了我们的方法，该数据集结合了8个基准测试，代表4个文本生成任务，结果表明CED在上下文演示选择方面提高了大型语言模型的零-shot性能。

    Large language models (LLMs) can use in-context demonstrations to improve performance on zero-shot tasks. However, selecting the best in-context examples is challenging because model performance can vary widely depending on the selected examples. We present a cross-entropy difference (CED) method for selecting in-context demonstrations. Our method is based on the observation that the effectiveness of in-context demonstrations negatively correlates with the perplexity of the test example by a language model that was finetuned on that demonstration. We utilize parameter efficient finetuning to train small models on training data that are used for computing the cross-entropy difference between a test example and every candidate in-context demonstration. This metric is used to rank and select in-context demonstrations independently for each test input. We evaluate our method on a mix-domain dataset that combines 8 benchmarks, representing 4 text generation tasks, showing that CED for in-co
    
[^12]: 为比较推理预训练语言模型

    Pre-training Language Models for Comparative Reasoning. (arXiv:2305.14457v1 [cs.CL])

    [http://arxiv.org/abs/2305.14457](http://arxiv.org/abs/2305.14457)

    本文提出一种预训练语言模型的新框架，旨在增强其在比较推理方面的能力。通过使用可扩展的基于文本实体比较数据的方法和新的预训练任务，该框架得到了显著的结果。

    

    本文提出了一种新框架，用于预训练语言模型以增强其在文本比较推理方面的能力。我们的方法涉及可扩展的用于收集基于文本实体比较数据的方法，并设计了三个新的预训练任务。在多个下游任务，包括比较问答、问句生成和摘要生成方面的评估表明，我们的预训练框架大大提高了语言模型的比较推理能力，尤其是在资源匮乏的情况下。此外，本工作还发布了第一个比较推理综合基准。

    In this paper, we propose a novel framework to pre-train language models for enhancing their abilities of comparative reasoning over texts. While recent research has developed models for NLP tasks that require comparative reasoning, they suffer from costly manual data labeling and limited generalizability to different tasks. Our approach involves a scalable method for collecting data for text-based entity comparison, which leverages both structured and unstructured data, and the design of three novel pre-training tasks. Evaluation on a range of downstream tasks including comparative question answering, question generation, and summarization shows that our pre-training framework significantly improves the comparative reasoning abilities of language models, especially under low-resource conditions. This work also releases the first integrated benchmark for comparative reasoning over texts.
    

