# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [DUMA: a Dual-Mind Conversational Agent with Fast and Slow Thinking.](http://arxiv.org/abs/2310.18075) | DUMA是一种具有快速和慢速思考能力的双重思维对话代理框架，通过利用两个生成型大型语言模型，实现了根据情况在直观响应和深思熟虑的问题解决过程之间无缝切换的能力。 |
| [^2] | [CrossCodeEval: A Diverse and Multilingual Benchmark for Cross-File Code Completion.](http://arxiv.org/abs/2310.11248) | CrossCodeEval是一个多元化和多语言的基准测试，用于跨文件代码补全，在真实的软件开发场景中，需要跨文件上下文理解才能准确完成代码。 |
| [^3] | [Targeted Image Data Augmentation Increases Basic Skills Captioning Robustness.](http://arxiv.org/abs/2309.15991) | 本论文提出了一种有针对性的图像数据增强方法（TIDA），通过使用文本到图像生成模型来填补关联结构差距以提高模型的人类感知能力，如性别识别。实验结果表明，相较于原始数据集，使用TIDA增强的数据集能够显著提高模型的鲁棒性。 |
| [^4] | [Comparative Topic Modeling for Determinants of Divergent Report Results Applied to Macular Degeneration Studies.](http://arxiv.org/abs/2309.00312) | 本研究提出了一种比较话题建模方法，用于分析马克白彦病研究中存在矛盾结果的报告。通过对比不同话题与显著结果的相关性，找到了与黄斑变性研究中显著结果报告相关的八种化合物。 |
| [^5] | [VisIT-Bench: A Benchmark for Vision-Language Instruction Following Inspired by Real-World Use.](http://arxiv.org/abs/2308.06595) | VisIT-Bench是一个用于评价真实世界中视觉语言模型指示遵循的基准，包含了各种任务并提供了详细描述，可以自动评估多模态生成的质量差距。 |
| [^6] | [InteractiveIE: Towards Assessing the Strength of Human-AI Collaboration in Improving the Performance of Information Extraction.](http://arxiv.org/abs/2305.14659) | 本文提出InteractiveIE方法，使用自动问答生成法来引出信息提取中的模板插槽，通过微小量代理人类监督，提高在生物医学和法律文档等领域中信息提取任务性能。 |
| [^7] | [Hierarchical Catalogue Generation for Literature Review: A Benchmark.](http://arxiv.org/abs/2304.03512) | 研究提出了一个名为HiCatGLR任务，致力于为文献综述生成分层目录，它可以从多篇论文中提取和组织重要信息。为了解决现有研究中缺少清晰逻辑层次结构概述的问题，提供了一个具有挑战性的解决方案并创建了新的数据集。通过此项研究，可以更加准确地评估模型性能并验证数据集的高质量和评估指标的有效性。 |
| [^8] | [Exploring the Potential of Large Language models in Traditional Korean Medicine: A Foundation Model Approach to Culturally-Adapted Healthcare.](http://arxiv.org/abs/2303.17807) | 本研究评估了大型语言模型在应用传统韩医的潜力。其中，GPT-4在应用韩国国家中医医生执照考试中取得了57.29%的准确率，潜在应用价值高。 |
| [^9] | [Language Models can Solve Computer Tasks.](http://arxiv.org/abs/2303.17491) | 本文研究表明，预训练的大型语言模型代理可以通过一个简单的提示方案使用自然语言执行计算机任务，该方法取得了很好的效果并在MiniWoB++基准测试中超越了监督学习和强化学习方法。 |

# 详细

[^1]: DUMA：具有快速和慢速思考能力的双重思维对话代理

    DUMA: a Dual-Mind Conversational Agent with Fast and Slow Thinking. (arXiv:2310.18075v1 [cs.CL])

    [http://arxiv.org/abs/2310.18075](http://arxiv.org/abs/2310.18075)

    DUMA是一种具有快速和慢速思考能力的双重思维对话代理框架，通过利用两个生成型大型语言模型，实现了根据情况在直观响应和深思熟虑的问题解决过程之间无缝切换的能力。

    

    受到人类认知的双过程理论启发，我们介绍了DUMA，一种新颖的对话代理框架，通过利用两个用于快速和慢速思考的生成型大型语言模型（LLM），体现了双重思维机制。快速思考模型作为主要接口用于外部交互和初始响应生成，根据完整响应的复杂性评估是否需要调用慢速思考模型。一旦被调用，慢速思考模型接管对话，在细致规划、推理和工具利用方面进行工作，提供经过充分分析的响应。这种双重思维配置允许根据情况在直观响应和深思熟虑的问题解决过程之间无缝切换。我们构建了一个用于处理房地产行业在线咨询的对话代理。实验证明，我们的方法在效果和效率之间取得了平衡。

    Inspired by the dual-process theory of human cognition, we introduce DUMA, a novel conversational agent framework that embodies a dual-mind mechanism through the utilization of two generative Large Language Models (LLMs) dedicated to fast and slow thinking respectively. The fast thinking model serves as the primary interface for external interactions and initial response generation, evaluating the necessity for engaging the slow thinking model based on the complexity of the complete response. When invoked, the slow thinking model takes over the conversation, engaging in meticulous planning, reasoning, and tool utilization to provide a well-analyzed response. This dual-mind configuration allows for a seamless transition between intuitive responses and deliberate problem-solving processes based on the situation. We have constructed a conversational agent to handle online inquiries in the real estate industry. The experiment proves that our method balances effectiveness and efficiency, an
    
[^2]: CrossCodeEval: 一个多元化和多语言的用于跨文件代码补全的基准测试

    CrossCodeEval: A Diverse and Multilingual Benchmark for Cross-File Code Completion. (arXiv:2310.11248v1 [cs.LG])

    [http://arxiv.org/abs/2310.11248](http://arxiv.org/abs/2310.11248)

    CrossCodeEval是一个多元化和多语言的基准测试，用于跨文件代码补全，在真实的软件开发场景中，需要跨文件上下文理解才能准确完成代码。

    

    代码补全模型在近年来取得了显著进展，然而当前流行的评估数据集，如HumanEval和MBPP，主要集中在单个文件内的代码补全任务上。这种过于简化的设置无法准确地代表现实世界中的软件开发场景，其中存储库跨越多个文件，存在大量的跨文件依赖关系，需要访问和理解跨文件上下文才能正确完成代码。为了填补这一空白，我们提出了CrossCodeEval，一个多元化和多语言的代码补全基准测试，需要深入的跨文件上下文理解才能准确完成代码。CrossCodeEval基于四种流行的编程语言（Python，Java，TypeScript和C#）中的多样化的真实世界、开源、权限许可的存储库集合构建。为了创建严格要求跨文件上下文进行准确完成的示例，我们提出了一个简单而高效的静态方法。

    Code completion models have made significant progress in recent years, yet current popular evaluation datasets, such as HumanEval and MBPP, predominantly focus on code completion tasks within a single file. This over-simplified setting falls short of representing the real-world software development scenario where repositories span multiple files with numerous cross-file dependencies, and accessing and understanding cross-file context is often required to complete the code correctly.  To fill in this gap, we propose CrossCodeEval, a diverse and multilingual code completion benchmark that necessitates an in-depth cross-file contextual understanding to complete the code accurately. CrossCodeEval is built on a diverse set of real-world, open-sourced, permissively-licensed repositories in four popular programming languages: Python, Java, TypeScript, and C#. To create examples that strictly require cross-file context for accurate completion, we propose a straightforward yet efficient static-
    
[^3]: 有针对性的图像数据增强提高了基本技能字幕鲁棒性

    Targeted Image Data Augmentation Increases Basic Skills Captioning Robustness. (arXiv:2309.15991v1 [cs.CV])

    [http://arxiv.org/abs/2309.15991](http://arxiv.org/abs/2309.15991)

    本论文提出了一种有针对性的图像数据增强方法（TIDA），通过使用文本到图像生成模型来填补关联结构差距以提高模型的人类感知能力，如性别识别。实验结果表明，相较于原始数据集，使用TIDA增强的数据集能够显著提高模型的鲁棒性。

    

    人工神经网络通常在推广到超出上下文的示例时遇到困难。这种限制的原因之一是数据集只包含有关世界潜在关联结构的部分信息。在这项工作中，我们提出了TIDA（有针对性的图像编辑数据增强）方法，它是一种专注于通过使用文本到图像生成模型来填补关联结构差距以提高模型类人能力（例如性别识别）的有针对性数据增强方法。更具体地说，TIDA识别描述图像的标题中的特定技能（例如图中特定性别的存在），改变标题（例如将“女人”改为“男人”），然后使用文本到图像模型编辑图像，以使其与新标题匹配（例如唯一地将一个女人改为一个男人同时保持上下文不变）。基于Flickr30K基准测试，我们展示了与原始数据集相比，使用TIDA增强的数据集相关的...

    Artificial neural networks typically struggle in generalizing to out-of-context examples. One reason for this limitation is caused by having datasets that incorporate only partial information regarding the potential correlational structure of the world. In this work, we propose TIDA (Targeted Image-editing Data Augmentation), a targeted data augmentation method focused on improving models' human-like abilities (e.g., gender recognition) by filling the correlational structure gap using a text-to-image generative model. More specifically, TIDA identifies specific skills in captions describing images (e.g., the presence of a specific gender in the image), changes the caption (e.g., "woman" to "man"), and then uses a text-to-image model to edit the image in order to match the novel caption (e.g., uniquely changing a woman to a man while maintaining the context identical). Based on the Flickr30K benchmark, we show that, compared with the original data set, a TIDA-enhanced dataset related to
    
[^4]: 用于马克白彦病研究的不同报告结果的比较话题建模

    Comparative Topic Modeling for Determinants of Divergent Report Results Applied to Macular Degeneration Studies. (arXiv:2309.00312v1 [cs.CL])

    [http://arxiv.org/abs/2309.00312](http://arxiv.org/abs/2309.00312)

    本研究提出了一种比较话题建模方法，用于分析马克白彦病研究中存在矛盾结果的报告。通过对比不同话题与显著结果的相关性，找到了与黄斑变性研究中显著结果报告相关的八种化合物。

    

    话题建模和文本挖掘是自然语言处理的子集，适用于进行元分析和系统审查。对于证据综述，上述NLP方法通常用于特定主题的文献搜索或从报告中提取值以自动化SR和MA的关键阶段。相反，本文提出了一种比较话题建模方法，用于分析同一广义研究问题上存在矛盾结果的报告。具体而言，目标是通过根据其比例发生和在显著结果报告中的一致性分布对其进行排名，找到与感兴趣的结果显著相关的话题。该方法在涉及补充营养化合物是否显著有益于黄斑变性(MD)的广泛范围的研究中进行了测试。确定了八种化合物与显著结果报告的特定相关性。

    Topic modeling and text mining are subsets of Natural Language Processing with relevance for conducting meta-analysis (MA) and systematic review (SR). For evidence synthesis, the above NLP methods are conventionally used for topic-specific literature searches or extracting values from reports to automate essential phases of SR and MA. Instead, this work proposes a comparative topic modeling approach to analyze reports of contradictory results on the same general research question. Specifically, the objective is to find topics exhibiting distinct associations with significant results for an outcome of interest by ranking them according to their proportional occurrence and consistency of distribution across reports of significant results. The proposed method was tested on broad-scope studies addressing whether supplemental nutritional compounds significantly benefit macular degeneration (MD). Eight compounds were identified as having a particular association with reports of significant r
    
[^5]: VisIT-Bench: 一个受真实世界使用启发的视觉语言指示评估基准

    VisIT-Bench: A Benchmark for Vision-Language Instruction Following Inspired by Real-World Use. (arXiv:2308.06595v1 [cs.CL])

    [http://arxiv.org/abs/2308.06595](http://arxiv.org/abs/2308.06595)

    VisIT-Bench是一个用于评价真实世界中视觉语言模型指示遵循的基准，包含了各种任务并提供了详细描述，可以自动评估多模态生成的质量差距。

    

    我们引入了VisIT-Bench（Visual InsTruction Benchmark），这是一个评价用于真实世界使用的视觉语言模型的指示遵循基准。我们的起点是策划了70个“指示家族”，我们认为指示调优的视觉语言模型应该能够解决这些家族。任务不仅限于VQAv2和COCO等评估，涵盖了从基本识别到游戏玩法和创造性生成的各种任务。在策划之后，我们的数据集包括592个测试查询，每个查询都带有一个人工编写的指示条件化的字幕。这些描述展现了特定指示因素，例如对于询问店面对于轮椅用户的易访问性的指示，条件化的字幕描述了斜坡/潜在障碍物。这些描述使得我们可以：1）收集每个实例的人工验证的参考输出；2）使用仅文本的语言模型对候选多模态生成进行自动评估，与人类判断相一致。我们量化了质量差距。

    We introduce VisIT-Bench (Visual InsTruction Benchmark), a benchmark for evaluation of instruction-following vision-language models for real-world use. Our starting point is curating 70 'instruction families' that we envision instruction tuned vision-language models should be able to address. Extending beyond evaluations like VQAv2 and COCO, tasks range from basic recognition to game playing and creative generation. Following curation, our dataset comprises 592 test queries, each with a human-authored instruction-conditioned caption. These descriptions surface instruction-specific factors, e.g., for an instruction asking about the accessibility of a storefront for wheelchair users, the instruction-conditioned caption describes ramps/potential obstacles. These descriptions enable 1) collecting human-verified reference outputs for each instance; and 2) automatic evaluation of candidate multimodal generations using a text-only LLM, aligning with human judgment. We quantify quality gaps be
    
[^6]: InteractiveIE：评估人工智能与人类协作的强度，提高信息提取的性能

    InteractiveIE: Towards Assessing the Strength of Human-AI Collaboration in Improving the Performance of Information Extraction. (arXiv:2305.14659v1 [cs.CL])

    [http://arxiv.org/abs/2305.14659](http://arxiv.org/abs/2305.14659)

    本文提出InteractiveIE方法，使用自动问答生成法来引出信息提取中的模板插槽，通过微小量代理人类监督，提高在生物医学和法律文档等领域中信息提取任务性能。

    

    学习基于模板的文档信息提取是一项关键且困难的任务。先前的基于模板的信息提取方法假定具有领域模板的先验知识；然而，实际的信息提取没有预定义的模式，并且是一个即兴创造的现象。为了在实际情况下快速引导模板，我们需要从文档中在零或最小监督的情况下引出模板插槽。由于问答的目的与信息提取的目标交汇，我们使用自动问答生成从文档中引出模板插槽，并研究微小量的代理人类监督（称为InteractiveIE）如何进一步提高性能。在获取训练数据昂贵的生物医学和法律文档上的大量实验揭示了使用InteractiveIE相对于仅使用人工智能的基准测试的鼓舞人心的性能提升趋势。

    Learning template based information extraction from documents is a crucial yet difficult task. Prior template-based IE approaches assume foreknowledge of the domain templates; however, real-world IE do not have pre-defined schemas and it is a figure-out-as you go phenomena. To quickly bootstrap templates in a real-world setting, we need to induce template slots from documents with zero or minimal supervision. Since the purpose of question answering intersect with the goal of information extraction, we use automatic question generation to induce template slots from the documents and investigate how a tiny amount of a proxy human-supervision on-the-fly (termed as InteractiveIE) can further boost the performance. Extensive experiments on biomedical and legal documents, where obtaining training data is expensive, reveal encouraging trends of performance improvement using InteractiveIE over AI-only baseline.
    
[^7]: 文献综述的分层目录生成：一个基准测试

    Hierarchical Catalogue Generation for Literature Review: A Benchmark. (arXiv:2304.03512v1 [cs.CL])

    [http://arxiv.org/abs/2304.03512](http://arxiv.org/abs/2304.03512)

    研究提出了一个名为HiCatGLR任务，致力于为文献综述生成分层目录，它可以从多篇论文中提取和组织重要信息。为了解决现有研究中缺少清晰逻辑层次结构概述的问题，提供了一个具有挑战性的解决方案并创建了新的数据集。通过此项研究，可以更加准确地评估模型性能并验证数据集的高质量和评估指标的有效性。

    

    多文档科学摘要可以从大量的论文中提取和组织重要信息，最近引起了广泛关注。然而，现有的研究主要集中在产生缺乏清晰和逻辑层次结构的冗长概述上。为了缓解这个问题，我们提出了一个名为“Hierarchical Catalogue Generation for Literature Review (HiCatGLR)”的原子和具有挑战性的任务，其目标是根据各种参考文献为综述论文生成分层目录。我们精心构建了一个新的英文文献综述分层目录数据集(HiCaD)，其中包含13.8k篇文献综述目录和120k篇参考论文，并通过端到端和流水线方法进行了各种实验的基准测试。为了准确评估模型性能，我们设计了从语义和结构上与参考标准相似度的评估指标。此外，我们的广泛分析验证了我们数据集的高质量和我们评估指标的有效性。

    Multi-document scientific summarization can extract and organize important information from an abundant collection of papers, arousing widespread attention recently. However, existing efforts focus on producing lengthy overviews lacking a clear and logical hierarchy. To alleviate this problem, we present an atomic and challenging task named Hierarchical Catalogue Generation for Literature Review (HiCatGLR), which aims to generate a hierarchical catalogue for a review paper given various references. We carefully construct a novel English Hierarchical Catalogues of Literature Reviews Dataset (HiCaD) with 13.8k literature review catalogues and 120k reference papers, where we benchmark diverse experiments via the end-to-end and pipeline methods. To accurately assess the model performance, we design evaluation metrics for similarity to ground truth from semantics and structure. Besides, our extensive analyses verify the high quality of our dataset and the effectiveness of our evaluation met
    
[^8]: 探索大型语言模型在传统韩医中的潜力：基于基础模型的文化适应保健方法

    Exploring the Potential of Large Language models in Traditional Korean Medicine: A Foundation Model Approach to Culturally-Adapted Healthcare. (arXiv:2303.17807v1 [cs.CL])

    [http://arxiv.org/abs/2303.17807](http://arxiv.org/abs/2303.17807)

    本研究评估了大型语言模型在应用传统韩医的潜力。其中，GPT-4在应用韩国国家中医医生执照考试中取得了57.29%的准确率，潜在应用价值高。

    

    传统韩医注重个体化诊断和治疗，数据有限且过程隐性，使AI建模困难。GPT-3.5和GPT-4等大型语言模型尽管缺乏医学专业培训，但已显示出出色的医疗知识。本研究旨在评估GPT-3.5和GPT-4在应用韩国国家中医医生执照考试中的潜力。结果显示，GPT-3.5和GPT-4分别取得了42.06%和57.29%的准确率，其中GPT-4接近及格水平。

    Introduction: Traditional Korean medicine (TKM) emphasizes individualized diagnosis and treatment, making AI modeling difficult due to limited data and implicit processes. GPT-3.5 and GPT-4, large language models, have shown impressive medical knowledge despite lacking medicine-specific training. This study aimed to assess the capabilities of GPT-3.5 and GPT-4 for TKM using the Korean National Licensing Examination for Korean Medicine Doctors. Methods: GPT-3.5 (February 2023) and GPT-4 (March 2023) models answered 340 questions from the 2022 examination across 12 subjects. Each question was independently evaluated five times in an initialized session. Results: GPT-3.5 and GPT-4 achieved 42.06% and 57.29% accuracy, respectively, with GPT-4 nearing passing performance. There were significant differences in accuracy by subjects, with 83.75% accuracy for neuropsychiatry compared to 28.75% for internal medicine (2). Both models showed high accuracy in recall-based and diagnosis-based questi
    
[^9]: 语言模型能够解决计算机任务

    Language Models can Solve Computer Tasks. (arXiv:2303.17491v1 [cs.CL])

    [http://arxiv.org/abs/2303.17491](http://arxiv.org/abs/2303.17491)

    本文研究表明，预训练的大型语言模型代理可以通过一个简单的提示方案使用自然语言执行计算机任务，该方法取得了很好的效果并在MiniWoB++基准测试中超越了监督学习和强化学习方法。

    

    能够在计算机上执行通用任务的代理可以通过自动化重复任务和协助复杂问题的解决来提高效率和生产力。理想情况下，这些代理应该能够通过自然语言命令解决新的计算机任务。然而，先前解决这个问题的方法需要大量专家示范和任务特定的奖励函数，这两者对于新任务来说都不切实际。在这项工作中，我们展示了一个预先训练的大型语言模型（LLM）代理可以使用一个简单的提示方案（RCI），通过自然语言指导执行计算机任务，并在批评和改进输出的过程中取得很好的效果。RCI方法在自动化计算机任务方面明显优于现有的LLM方法，并在MiniWoB++基准测试中超越了监督学习（SL）和强化学习（RL）方法。RCI方法使用每个任务仅有的少数示范，与最新的SL+RL方法相竞争。

    Agents capable of carrying out general tasks on a computer can improve efficiency and productivity by automating repetitive tasks and assisting in complex problem-solving. Ideally, such agents should be able to solve new computer tasks presented to them through natural language commands. However, previous approaches to this problem require large amounts of expert demonstrations and task-specific reward functions, both of which are impractical for new tasks. In this work, we show that a pre-trained large language model (LLM) agent can execute computer tasks guided by natural language using a simple prompting scheme where the agent recursively criticizes and improves its output (RCI). The RCI approach significantly outperforms existing LLM methods for automating computer tasks and surpasses supervised learning (SL) and reinforcement learning (RL) approaches on the MiniWoB++ benchmark. RCI is competitive with the state-of-the-art SL+RL method, using only a handful of demonstrations per ta
    

