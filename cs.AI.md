# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Severity Controlled Text-to-Image Generative Model Bias Manipulation](https://arxiv.org/abs/2404.02530) | 本文揭示了文本到图像生成模型对偏见操纵的敏感性，并提出了一种通过定量控制模型偏见来操纵输出严重性的技术，从而实现精确提示工程生成新颖图像的方法。 |
| [^2] | [Counterfactual contrastive learning: robust representations via causal image synthesis](https://arxiv.org/abs/2403.09605) | 本研究提出了CF-SimCLR，一种反事实对照学习方法，利用近似反事实推断创造正样本，大大提高了模型对采集偏移的稳健性，并在多个数据集上取得了较高的下游性能。 |
| [^3] | [Data Augmentation is Dead, Long Live Data Augmentation](https://arxiv.org/abs/2402.14895) | 数据增强不过是更好地微调模型，零唁态和少样本数据生成可提高性能 |
| [^4] | [COBIAS: Contextual Reliability in Bias Assessment](https://arxiv.org/abs/2402.14889) | 我们提出了COBIAS，旨在通过考虑多样情境的用户输入内容，衡量语句的情境可靠性，从而培养偏见意识。 |
| [^5] | [Building ethical guidelines for generative AI in scientific research.](http://arxiv.org/abs/2401.15284) | 本文提出了一个初步的框架，通过五个关键主题的分析和缓解策略来建立科学研究中生成AI的伦理指南。全球共识、专业培训和合理的执行对于促进AI的益处和维护研究诚信至关重要。 |
| [^6] | [Towards Goal-oriented Large Language Model Prompting: A Survey.](http://arxiv.org/abs/2401.14043) | 本文调查了大型语言模型(LLM)中目标导向提示工程的重要性。通过对35个代表性研究的回顾，我们发现引导LLM遵循人类的逻辑思维的目标导向提示公式显著提高了LLM的性能。我们还提出了一个新的分类体系，并总结了十个适用任务来展示我们框架的广泛适用性。同时，我们提出了四个未来的方向，以推动目标导向提示工程的进一步发展。 |
| [^7] | [Information That Matters: Exploring Information Needs of People Affected by Algorithmic Decisions.](http://arxiv.org/abs/2401.13324) | 本研究探讨了受算法决策影响的人的信息需求，发现解释往往不能满足他们的关注点，导致对监管框架的理解和遵守产生障碍。为了解决这个问题，研究团队提出了XAI初学者问题库，涵盖了就业预测和健康监测两个领域中受影响利益相关者的信息需求。 |
| [^8] | [AI Alignment in the Design of Interactive AI: Specification Alignment, Process Alignment, and Evaluation Support.](http://arxiv.org/abs/2311.00710) | 本文关注AI在界面设计和评估中的对齐问题，提出了规范对齐、过程对齐和评估支持等三个对齐目标，并介绍了代理过程和过程海湾的概念。 |
| [^9] | [Large language models can replicate cross-cultural differences in personality.](http://arxiv.org/abs/2310.10679) | 大型语言模型GPT-4成功复制了使用十项人格问卷测量的大五人格的跨文化差异，但其结果表明平均评级有上升偏差和较低的变异性与结构效度。 |
| [^10] | [Diverse Neural Audio Embeddings -- Bringing Features back !.](http://arxiv.org/abs/2309.08751) | 本文通过在音频分类任务中学习多样化的特征表示，包括领域特定的音高、音色和神经表示，以及端到端架构，为学习稳健、多样化的表示铺平了道路，并显著提高了性能。 |
| [^11] | [Learning by Self-Explaining.](http://arxiv.org/abs/2309.08395) | 学习通过自我解释（LSX）是一种新的学习范式，通过给予解释和批评者的反馈来改进学习者的性能。这种方法适用于图像分类等基本任务，并有潜力在人工智能研究中发挥作用。 |

# 详细

[^1]: 严重控制的文本到图像生成模型偏见操纵

    Severity Controlled Text-to-Image Generative Model Bias Manipulation

    [https://arxiv.org/abs/2404.02530](https://arxiv.org/abs/2404.02530)

    本文揭示了文本到图像生成模型对偏见操纵的敏感性，并提出了一种通过定量控制模型偏见来操纵输出严重性的技术，从而实现精确提示工程生成新颖图像的方法。

    

    文本到图像（T2I）生成模型正在广泛流行，尤其是在公共领域。然而，它们固有的偏见和潜在的恶意操纵还未被充分探讨。本文揭示了T2I模型对此类操纵的易感性，并首次提出了通过针对嵌入式语言模型动态且高效地利用模型偏见的新可能性。通过利用向量代数的数学基础，我们的技术实现了对模型偏见通过严重性的输出操纵的可扩展和方便控制。作为副产品，该控制还允许一种精确的提示工程，以生成通常不太可能通过常规文本提示生成的图像。我们还展示了我们的操纵技术在平衡生成类别频率方面的建设应用 - 如在模型去偏。我们的技术不需要训练，并且也以后门的形式构建。

    arXiv:2404.02530v1 Announce Type: cross  Abstract: Text-to-image (T2I) generative models are gaining wide popularity, especially in public domains. However, their intrinsic bias and potential malicious manipulations remain under-explored. Charting the susceptibility of T2I models to such manipulation, we first expose the new possibility of a dynamic and computationally efficient exploitation of model bias by targeting the embedded language models. By leveraging mathematical foundations of vector algebra, our technique enables a scalable and convenient control over the severity of output manipulation through model bias. As a by-product, this control also allows a form of precise prompt engineering to generate images which are generally implausible with regular text prompts. We also demonstrate a constructive application of our manipulation for balancing the frequency of generated classes - as in model debiasing. Our technique does not require training and is also framed as a backdoor at
    
[^2]: 反事实对照学习：通过因果图像合成获得稳健表示

    Counterfactual contrastive learning: robust representations via causal image synthesis

    [https://arxiv.org/abs/2403.09605](https://arxiv.org/abs/2403.09605)

    本研究提出了CF-SimCLR，一种反事实对照学习方法，利用近似反事实推断创造正样本，大大提高了模型对采集偏移的稳健性，并在多个数据集上取得了较高的下游性能。

    

    对比预训练已被广泛认为能够提高下游任务性能和模型泛化能力，特别是在有限标签设置中。然而，它对增强管道的选择敏感。正样本应保留语义信息同时破坏域特定信息。标准增强管道通过预定义的光度变换模拟域特定变化，但如果我们能够模拟真实的领域变化呢？在这项工作中，我们展示了如何利用最近在反事实图像生成方面的进展来实现这一目的。我们提出了CF-SimCLR，一种反事实对照学习方法，它利用近似反事实推断进行正样本创建。对胸部X光和乳腺X光等五个数据集的全面评估表明，CF-SimCLR显著提高了对获取偏移的稳健性，在两种数据集上的下游性能更好。

    arXiv:2403.09605v1 Announce Type: cross  Abstract: Contrastive pretraining is well-known to improve downstream task performance and model generalisation, especially in limited label settings. However, it is sensitive to the choice of augmentation pipeline. Positive pairs should preserve semantic information while destroying domain-specific information. Standard augmentation pipelines emulate domain-specific changes with pre-defined photometric transformations, but what if we could simulate realistic domain changes instead? In this work, we show how to utilise recent progress in counterfactual image generation to this effect. We propose CF-SimCLR, a counterfactual contrastive learning approach which leverages approximate counterfactual inference for positive pair creation. Comprehensive evaluation across five datasets, on chest radiography and mammography, demonstrates that CF-SimCLR substantially improves robustness to acquisition shift with higher downstream performance on both in- an
    
[^3]: 数据增强已死，数据增强万岁

    Data Augmentation is Dead, Long Live Data Augmentation

    [https://arxiv.org/abs/2402.14895](https://arxiv.org/abs/2402.14895)

    数据增强不过是更好地微调模型，零唁态和少样本数据生成可提高性能

    

    文本数据增强（DA）是一个繁荣的研究领域，不断提出新颖的技术来创建人工数据，已经在小数据环境中表现出很高的效率，至少对于文本分类任务而言。在本文中，我们质疑这些结果，表明经典的数据增强只是一种更好地进行微调的方式，并且在应用数据增强之前花更多时间进行微调会抵消其效果。这是一个重要的贡献，因为它回答了最近几年留下的几个问题，即：哪种DA技术表现最佳（只要它们生成的数据与训练集足够接近，不会损害训练），为什么DA表现出积极的结果（简化网络训练）。此外，我们还展示了通过对话代理（如ChatGPT或LLama2）零唁态和少样本数据生成可以提高性能，从而得出了结论，此法可以提高模型性能。

    arXiv:2402.14895v1 Announce Type: cross  Abstract: Textual data augmentation (DA) is a prolific field of study where novel techniques to create artificial data are regularly proposed, and that has demonstrated great efficiency on small data settings, at least for text classification tasks. In this paper, we challenge those results, showing that classical data augmentation is simply a way of performing better fine-tuning, and that spending more time fine-tuning before applying data augmentation negates its effect. This is a significant contribution as it answers several questions that were left open in recent years, namely~: which DA technique performs best (all of them as long as they generate data close enough to the training set as to not impair training) and why did DA show positive results (facilitates training of network). We furthermore show that zero and few-shot data generation via conversational agents such as ChatGPT or LLama2 can increase performances, concluding that this f
    
[^4]: COBIAS：偏见评估中的情境可靠性

    COBIAS: Contextual Reliability in Bias Assessment

    [https://arxiv.org/abs/2402.14889](https://arxiv.org/abs/2402.14889)

    我们提出了COBIAS，旨在通过考虑多样情境的用户输入内容，衡量语句的情境可靠性，从而培养偏见意识。

    

    大型语言模型（LLMs）是基于固有偏见数据训练的。以往的去偏见模型研究依赖基准数据集来衡量模型性能。然而，这些数据集由于对偏见的极其主观理解而存在多个缺陷，凸显出对情境探索的迫切需求。我们提出考虑输入用户内容的情境，考虑到输入语句可能存在的多种情况。这种方法将允许培养偏见意识的框架，而不是伤害用户参与的防护设施。我们的贡献有两个方面：(i) 我们创建了一个包含2287个陈词滥调语句以及添加情境要点的数据集；(ii) 我们开发了面向情境的偏见指标和评估分数（COBIAS）来评估语句在衡量偏见方面的情境可靠性。我们的度量是衡量偏见基准数据集情境可靠性的重要预测因子。

    arXiv:2402.14889v1 Announce Type: cross  Abstract: Large Language Models (LLMs) are trained on inherently biased data. Previous works on debiasing models rely on benchmark datasets to measure model performance. However, these datasets suffer from several pitfalls due to the extremely subjective understanding of bias, highlighting a critical need for contextual exploration. We propose understanding the context of user inputs with consideration of the diverse situations in which input statements are possible. This approach would allow for frameworks that foster bias awareness rather than guardrails that hurt user engagement. Our contribution is twofold: (i) we create a dataset of 2287 stereotyped statements augmented with points for adding context; (ii) we develop the Context-Oriented Bias Indicator and Assessment Score (COBIAS) to assess statements' contextual reliability in measuring bias. Our metric is a significant predictor of the contextual reliability of bias-benchmark datasets ($
    
[^5]: 在科学研究中建立生成AI的伦理指南

    Building ethical guidelines for generative AI in scientific research. (arXiv:2401.15284v1 [cs.CY])

    [http://arxiv.org/abs/2401.15284](http://arxiv.org/abs/2401.15284)

    本文提出了一个初步的框架，通过五个关键主题的分析和缓解策略来建立科学研究中生成AI的伦理指南。全球共识、专业培训和合理的执行对于促进AI的益处和维护研究诚信至关重要。

    

    生成人工智能工具（如大型语言模型）正在迅速改变学术研究和实际应用。然而，关于科学中生成AI的伦理指南的讨论仍然零散，强调了协商一致性标准的紧迫性。本文通过对五个关键主题的分析和缓解策略的开发，提供了一个初步的框架：了解模型在真实性和偏见方面的局限性；尊重隐私、机密和版权；在融入模型输出时避免抄袭和违反政策；确保应用带来总体利益；以及透明、可复制地使用人工智能。通过列举常见场景来展示潜在的伦理违规行为。我们认为，全球共识以及专业培训和合理的执行是促进AI的益处并维护研究诚信的关键。

    Generative artificial intelligence tools like large language models are rapidly transforming academic research and real world applications. However, discussions on ethical guidelines for generative AI in science remain fragmented, underscoring the urgent need for consensus based standards. This paper offers an initial framework by developing analyses and mitigation strategies across five key themes: understanding model limitations regarding truthfulness and bias; respecting privacy, confidentiality, and copyright; avoiding plagiarism and policy violations when incorporating model output; ensuring applications provide overall benefit; and using AI transparently and reproducibly. Common scenarios are outlined to demonstrate potential ethical violations. We argue that global consensus coupled with professional training and reasonable enforcement are critical to promoting the benefits of AI while safeguarding research integrity.
    
[^6]: 朝着目标导向的大型语言模型提示方法：一项调查

    Towards Goal-oriented Large Language Model Prompting: A Survey. (arXiv:2401.14043v1 [cs.CL])

    [http://arxiv.org/abs/2401.14043](http://arxiv.org/abs/2401.14043)

    本文调查了大型语言模型(LLM)中目标导向提示工程的重要性。通过对35个代表性研究的回顾，我们发现引导LLM遵循人类的逻辑思维的目标导向提示公式显著提高了LLM的性能。我们还提出了一个新的分类体系，并总结了十个适用任务来展示我们框架的广泛适用性。同时，我们提出了四个未来的方向，以推动目标导向提示工程的进一步发展。

    

    大型语言模型(LLM)在各种下游任务中显示出卓越的性能，而提示工程在优化LLM性能中起着关键作用。本文旨在强调设计提示的限制，同时保持人类追求LLM像人类思考的人类学假设。通过对35个代表性研究的回顾，我们展示了目标导向提示公式的重要性，该公式指导LLM遵循人类的逻辑思维，显著提高了LLM的性能。此外，我们引入了一个新的分类体系，将目标导向提示方法分为五个相互关联的阶段，并通过总结十个适用任务来展示我们框架的广泛适用性。最后，我们提出了四个未来的方向，希望进一步强调和推动目标导向提示工程。

    Large Language Models (LLMs) have shown prominent performance in various downstream tasks in which prompt engineering plays a pivotal role in optimizing LLMs' performance. This paper, not as an overview of current prompt engineering methods, aims to highlight the limitation of designing prompts while holding an anthropomorphic assumption that expects LLMs to think like humans. From our review of 35 representative studies, we demonstrate that a goal-oriented prompt formulation, which guides LLMs to follow established human logical thinking, significantly improves the performance of LLMs. Furthermore, We introduce a novel taxonomy that categorizes goal-oriented prompting methods into five interconnected stages and we demonstrate the broad applicability of our framework by summarizing ten applicable tasks. With four future directions proposed, we hope to further emphasize and promote goal-oriented prompt engineering.
    
[^7]: 有关算法决策的信息：探索受到算法决策影响的人的信息需求。

    Information That Matters: Exploring Information Needs of People Affected by Algorithmic Decisions. (arXiv:2401.13324v1 [cs.HC])

    [http://arxiv.org/abs/2401.13324](http://arxiv.org/abs/2401.13324)

    本研究探讨了受算法决策影响的人的信息需求，发现解释往往不能满足他们的关注点，导致对监管框架的理解和遵守产生障碍。为了解决这个问题，研究团队提出了XAI初学者问题库，涵盖了就业预测和健康监测两个领域中受影响利益相关者的信息需求。

    

    AI系统的解释很少涉及到受算法决策影响的人的信息需求。这种传达信息与受影响利益相关者所关心的信息之间的差距可能阻碍对监管框架（如AI法案）的理解和遵守。为了解决这个差距，我们提出了“XAI初学者问题库”：这是一个涵盖两个算法决策应用领域（就业预测和健康监测）中受影响利益相关者信息需求的目录，包括数据、系统背景、系统使用和系统规范等类别。信息需求是通过访谈研究收集的，参与者根据自己的问题获得解释。参与者还报告了他们的理解和决策信心，结果显示，尽管在接受解释后信心倾向于增加，但参与者也面临着理解上的挑战，如无法解释为什么自己的理解感觉不完整。解释还对理解产生了影响。

    Explanations of AI systems rarely address the information needs of people affected by algorithmic decision-making (ADM). This gap between conveyed information and information that matters to affected stakeholders can impede understanding and adherence to regulatory frameworks such as the AI Act. To address this gap, we present the "XAI Novice Question Bank": A catalog of affected stakeholders' information needs in two ADM use cases (employment prediction and health monitoring), covering the categories data, system context, system usage, and system specifications. Information needs were gathered in an interview study where participants received explanations in response to their inquiries. Participants further reported their understanding and decision confidence, showing that while confidence tended to increase after receiving explanations, participants also met understanding challenges, such as being unable to tell why their understanding felt incomplete. Explanations further influenced
    
[^8]: AI互动中的AI对齐：规范对齐，过程对齐和评估支持

    AI Alignment in the Design of Interactive AI: Specification Alignment, Process Alignment, and Evaluation Support. (arXiv:2311.00710v1 [cs.HC])

    [http://arxiv.org/abs/2311.00710](http://arxiv.org/abs/2311.00710)

    本文关注AI在界面设计和评估中的对齐问题，提出了规范对齐、过程对齐和评估支持等三个对齐目标，并介绍了代理过程和过程海湾的概念。

    

    AI对齐是确保AI产生期望结果而避免不良副作用的整体问题。虽然通常从安全和人类价值的角度考虑AI对齐，但也可以在设计和评估交互式AI系统的界面的背景下考虑AI对齐。本文将AI对齐的概念映射到基本的三步交互循环中，得出相应的对齐目标：1）规范对齐：确保用户能够高效可靠地将目标传达给AI；2）过程对齐：提供验证和可选择控制AI执行过程的能力；3）评估支持：确保用户能够验证和理解AI的输出。我们还介绍了代理过程的概念，它被定义为AI实际过程的简化、分离派生但可控制的表示；以及过程海湾的概念，它突显人类和AI过程之间的差异。

    AI alignment considers the overall problem of ensuring an AI produces desired outcomes, without undesirable side effects. While often considered from the perspectives of safety and human values, AI alignment can also be considered in the context of designing and evaluating interfaces for interactive AI systems. This paper maps concepts from AI alignment onto a basic, three step interaction cycle, yielding a corresponding set of alignment objectives: 1) specification alignment: ensuring the user can efficiently and reliably communicate objectives to the AI, 2) process alignment: providing the ability to verify and optionally control the AI's execution process, and 3) evaluation support: ensuring the user can verify and understand the AI's output. We also introduce the concepts of a surrogate process, defined as a simplified, separately derived, but controllable representation of the AI's actual process; and the notion of a Process Gulf, which highlights how differences between human and
    
[^9]: 大型语言模型可以复制跨文化个性差异

    Large language models can replicate cross-cultural differences in personality. (arXiv:2310.10679v1 [cs.CL])

    [http://arxiv.org/abs/2310.10679](http://arxiv.org/abs/2310.10679)

    大型语言模型GPT-4成功复制了使用十项人格问卷测量的大五人格的跨文化差异，但其结果表明平均评级有上升偏差和较低的变异性与结构效度。

    

    我们使用一项大规模实验(N=8000)来确定GPT-4是否可以复制使用十项人格问卷测量的大五人格的跨文化差异。我们选择美国和韩国作为文化对比，因为先前的研究表明这两个国家的人之间存在显著的人格差异。我们操纵了模拟的目标（美国 vs. 韩国），问卷的语言（英语 vs. 韩语）以及语言模型（GPT-4 vs. GPT-3.5）。我们的结果表明，GPT-4复制了每个因子的跨文化差异。然而，平均评级具有上升偏差，并且比人类样本的变异性更低，以及结构效度较低。总的来说，我们提供了初步的证据说明LLMs可以促进跨文化心理研究。

    We use a large-scale experiment (N=8000) to determine whether GPT-4 can replicate cross-cultural differences in the Big Five, measured using the Ten-Item Personality Inventory. We used the US and South Korea as the cultural pair, given that prior research suggests substantial personality differences between people from these two countries. We manipulated the target of the simulation (US vs. Korean), the language of the inventory (English vs. Korean), and the language model (GPT-4 vs. GPT-3.5). Our results show that GPT-4 replicated the cross-cultural differences for each factor. However, mean ratings had an upward bias and exhibited lower variation than in the human samples, as well as lower structural validity. Overall, we provide preliminary evidence that LLMs can aid cross-cultural psychological research.
    
[^10]: 多样的神经音频嵌入 - 恢复特征！

    Diverse Neural Audio Embeddings -- Bringing Features back !. (arXiv:2309.08751v1 [cs.SD])

    [http://arxiv.org/abs/2309.08751](http://arxiv.org/abs/2309.08751)

    本文通过在音频分类任务中学习多样化的特征表示，包括领域特定的音高、音色和神经表示，以及端到端架构，为学习稳健、多样化的表示铺平了道路，并显著提高了性能。

    

    随着现代人工智能架构的出现，从端到端的架构开始流行。这种转变导致了神经架构在没有领域特定偏见/知识的情况下进行训练，根据任务进行优化。本文中，我们通过多样的特征表示（在本例中是领域特定的）学习音频嵌入。对于涉及数百种声音分类的情况，我们学习分别针对音高、音色和神经表示等多样的音频属性建立稳健的嵌入，同时也通过端到端架构进行学习。我们观察到手工制作的嵌入，例如基于音高和音色的嵌入，虽然单独使用时无法击败完全端到端的表示，但将这些嵌入与端到端嵌入相结合可以显著提高性能。这项工作将为在端到端模型中引入一些领域专业知识来学习稳健、多样化的表示铺平道路，并超越仅训练端到端模型的性能。

    With the advent of modern AI architectures, a shift has happened towards end-to-end architectures. This pivot has led to neural architectures being trained without domain-specific biases/knowledge, optimized according to the task. We in this paper, learn audio embeddings via diverse feature representations, in this case, domain-specific. For the case of audio classification over hundreds of categories of sound, we learn robust separate embeddings for diverse audio properties such as pitch, timbre, and neural representation, along with also learning it via an end-to-end architecture. We observe handcrafted embeddings, e.g., pitch and timbre-based, although on their own, are not able to beat a fully end-to-end representation, yet adding these together with end-to-end embedding helps us, significantly improve performance. This work would pave the way to bring some domain expertise with end-to-end models to learn robust, diverse representations, surpassing the performance of just training 
    
[^11]: 学习通过自我解释

    Learning by Self-Explaining. (arXiv:2309.08395v1 [cs.AI])

    [http://arxiv.org/abs/2309.08395](http://arxiv.org/abs/2309.08395)

    学习通过自我解释（LSX）是一种新的学习范式，通过给予解释和批评者的反馈来改进学习者的性能。这种方法适用于图像分类等基本任务，并有潜力在人工智能研究中发挥作用。

    

    人工智能研究长期以来一直从生物学中寻找灵感，特别是人类智能。与目前主要将解释视为模型检查手段的人工智能研究相比，从心理学中发现自我解释在代理学习过程中的好处有些被忽视了。受到这个启发，我们引入了一种新的学习范式，称为学习通过自我解释 (LSX)。其中的基本思想是，一个学习模块 (学习者) 执行一个基本任务，比如图像分类，并对其决策进行解释。随后，一个内部批评者模块基于原始任务评估这些解释的质量。最后，学习者通过批评者的反馈得到改进，并根据需要重复这个循环。背后的直觉是，如果批评者能够根据相应的解释执行相同的任务，则该解释被认为是“好”的。尽管有许多实现可能性，但本文旨在提供关于实施学习通过自我解释的一般指导原则。有待进一步的研究和实践来探索这一学习范式的潜力。

    Artificial intelligence (AI) research has a long track record of drawing inspirations from findings from biology, in particular human intelligence. In contrast to current AI research that mainly treats explanations as a means for model inspection, a somewhat neglected finding from human psychology is the benefit of self-explaining in an agents' learning process. Motivated by this, we introduce a novel learning paradigm, termed Learning by Self-Explaining (LSX). The underlying idea is that a learning module (learner) performs a base task, e.g. image classification, and provides explanations to its decisions. An internal critic module next evaluates the quality of these explanations given the original task. Finally, the learner is refined with the critic's feedback and the loop is repeated as required. The intuition behind this is that an explanation is considered "good" if the critic can perform the same task given the respective explanation. Despite many implementation possibilities th
    

