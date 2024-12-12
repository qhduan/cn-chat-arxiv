# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Fusing Domain-Specific Content from Large Language Models into Knowledge Graphs for Enhanced Zero Shot Object State Classification](https://arxiv.org/abs/2403.12151) | 大型语言模型与知识图谱结合，提高零样本对象状态分类性能 |
| [^2] | [Merino: Entropy-driven Design for Generative Language Models on IoT Devices](https://arxiv.org/abs/2403.07921) | 在本文中，我们提出了一个新颖的信息熵框架，用于设计手机友好的生成式语言模型，通过最大化transformer解码器的熵来在计算预算内，成功设计了MeRino模型，在移动设置下展现出与当前最先进的自回归transformer模型竞争性能的特点 |
| [^3] | [Rainbow Teaming: Open-Ended Generation of Diverse Adversarial Prompts](https://arxiv.org/abs/2402.16822) | Rainbow Teaming提出了一种新方法，通过开放式搜索生成多样化的对抗性提示，可以帮助改善大型语言模型的稳健性，提高安全性，问答和网络安全等领域的模型漏洞。 |
| [^4] | [BiMediX: Bilingual Medical Mixture of Experts LLM](https://arxiv.org/abs/2402.13253) | 这项研究提出了BiMediX，一个旨在实现英语和阿拉伯语之间无缝医学交互的双语医学专家混合模型LLM，引入了半自动化的翻译流水线和全面的评估基准，同时推出了包含130万多样化医学交互的BiMed1.3M数据集。 |
| [^5] | [FTFT: Efficient and Robust Fine-Tuning by Transferring Training Dynamics](https://arxiv.org/abs/2310.06588) | 训练动态可在不同模型大小和预训练方法之间进行转移，通过选定的训练实例微调主模型实现比经验风险最小化更高的训练效率 |

# 详细

[^1]: 将大型语言模型中的领域特定内容融入知识图谱，以增强零样本对象状态分类

    Fusing Domain-Specific Content from Large Language Models into Knowledge Graphs for Enhanced Zero Shot Object State Classification

    [https://arxiv.org/abs/2403.12151](https://arxiv.org/abs/2403.12151)

    大型语言模型与知识图谱结合，提高零样本对象状态分类性能

    

    领域特定知识可以显著有助于解决各种视觉任务，但生成这种知识需要大量人力和时间成本。本研究探讨了大型语言模型（LLMs）在通过语义嵌入生成和提供领域特定信息方面的潜力。为实现这一目标，将LLM集成到一个流程中，该流程在视觉基础零样本对象状态分类任务的背景下利用知识图谱和预训练的语义向量。通过广泛的消融研究彻底研究了LLM的行为。我们的研究结果表明，将基于LLM的嵌入与通用的预训练嵌入结合使用可以显著提高性能。借鉴这一消融研究的见解，我们对竞争模型进行了比较分析，从而突出了最新的表现水平。

    arXiv:2403.12151v1 Announce Type: new  Abstract: Domain-specific knowledge can significantly contribute to addressing a wide variety of vision tasks. However, the generation of such knowledge entails considerable human labor and time costs. This study investigates the potential of Large Language Models (LLMs) in generating and providing domain-specific information through semantic embeddings. To achieve this, an LLM is integrated into a pipeline that utilizes Knowledge Graphs and pre-trained semantic vectors in the context of the Vision-based Zero-shot Object State Classification task. We thoroughly examine the behavior of the LLM through an extensive ablation study. Our findings reveal that the integration of LLM-based embeddings, in combination with general-purpose pre-trained embeddings, leads to substantial performance improvements. Drawing insights from this ablation study, we conduct a comparative analysis against competing models, thereby highlighting the state-of-the-art perfor
    
[^2]: Merino：基于熵驱动的IoT设备上生成式语言模型设计

    Merino: Entropy-driven Design for Generative Language Models on IoT Devices

    [https://arxiv.org/abs/2403.07921](https://arxiv.org/abs/2403.07921)

    在本文中，我们提出了一个新颖的信息熵框架，用于设计手机友好的生成式语言模型，通过最大化transformer解码器的熵来在计算预算内，成功设计了MeRino模型，在移动设置下展现出与当前最先进的自回归transformer模型竞争性能的特点

    

    大规模生成式语言模型（LLMs）作为人工智能现代时代的革命性进步，然而，直接部署LLMs在资源受限的硬件上，比如物联网（IoT）设备，由于其高计算成本而变得困难。在本文中，我们提出了一个新颖的信息熵框架，用于设计手机友好的生成式语言模型。我们的主要设计范式是在给定的计算预算内最大化transformer解码器的熵。整个设计过程涉及解决一个数学规划（MP）问题，可以在几分钟内在CPU上完成，使其几乎是零成本的。我们评估了我们设计的模型MeRino，在九个NLP下游任务上展示了它们在移动设置下对抗当前最先进的自回归transformer模型的竞争性表现。值得注意的是，MeRino在移动设置下获得了类似或更好的零性能表现

    arXiv:2403.07921v1 Announce Type: cross  Abstract: Generative Large Language Models (LLMs) stand as a revolutionary advancement in the modern era of artificial intelligence (AI). However, directly deploying LLMs in resource-constrained hardware, such as Internet-of-Things (IoT) devices, is difficult due to their high computational cost. In this paper, we propose a novel information-entropy framework for designing mobile-friendly generative language models. Our key design paradigm is to maximize the entropy of transformer decoders within the given computational budgets. The whole design procedure involves solving a mathematical programming (MP) problem, which can be done on the CPU within minutes, making it nearly zero-cost. We evaluate our designed models, termed MeRino, across nine NLP downstream tasks, showing their competitive performance against the state-of-the-art autoregressive transformer models under the mobile setting. Notably, MeRino achieves similar or better zero performan
    
[^3]: 彩虹团队：多样化对抗性提示的开放式生成

    Rainbow Teaming: Open-Ended Generation of Diverse Adversarial Prompts

    [https://arxiv.org/abs/2402.16822](https://arxiv.org/abs/2402.16822)

    Rainbow Teaming提出了一种新方法，通过开放式搜索生成多样化的对抗性提示，可以帮助改善大型语言模型的稳健性，提高安全性，问答和网络安全等领域的模型漏洞。

    

    随着大型语言模型（LLMs）在许多现实世界应用中变得越来越普遍，理解和增强它们对用户输入的稳健性至关重要。现有的用于识别敌对提示的方法往往专注于特定领域，缺乏多样性，或需要大量人工注释。为了解决这些限制，我们提出了彩虹团队，一种用于生成多样化对抗性提示的新方法。彩虹团队将对抗性提示生成视为一个质量 - 多样性问题，并使用开放式搜索来生成既有效又多样的提示。它可以揭示模型在广泛领域内的脆弱性，包括本文中的安全性、问答和网络安全。我们还证明，对由彩虹团队生成的合成数据进行微调可以提高最先进的LLMs的安全性，而不损害它们的一般能力。

    arXiv:2402.16822v1 Announce Type: new  Abstract: As large language models (LLMs) become increasingly prevalent across many real-world applications, understanding and enhancing their robustness to user inputs is of paramount importance. Existing methods for identifying adversarial prompts tend to focus on specific domains, lack diversity, or require extensive human annotations. To address these limitations, we present Rainbow Teaming, a novel approach for producing a diverse collection of adversarial prompts. Rainbow Teaming casts adversarial prompt generation as a quality-diversity problem, and uses open-ended search to generate prompts that are both effective and diverse. It can uncover a model's vulnerabilities across a broad range of domains including, in this paper, safety, question answering, and cybersecurity. We also demonstrate that fine-tuning on synthetic data generated by Rainbow Teaming improves the safety of state-of-the-art LLMs without hurting their general capabilities 
    
[^4]: BiMediX: 双语医学专家混合模型LLM

    BiMediX: Bilingual Medical Mixture of Experts LLM

    [https://arxiv.org/abs/2402.13253](https://arxiv.org/abs/2402.13253)

    这项研究提出了BiMediX，一个旨在实现英语和阿拉伯语之间无缝医学交互的双语医学专家混合模型LLM，引入了半自动化的翻译流水线和全面的评估基准，同时推出了包含130万多样化医学交互的BiMed1.3M数据集。

    

    在本文中，我们介绍了BiMediX，这是第一个旨在实现在英语和阿拉伯语之间无缝互动的双语医学专家混合模型LLM。我们的模型在英语和阿拉伯语之间促进了广泛范围的医学交互，包括多轮对话以询问关于患者症状和病史等额外细节、多项选择题回答以及开放式问题回答。我们提出了一个半自动化的英语到阿拉伯语翻译流水线，结合人工优化以确保高质量的翻译。我们还引入了一个用于阿拉伯语医学LLMs的全面评估基准。此外，我们推出了BiMed1.3M，一个涵盖130万各种医学交互的广泛的阿拉伯语-英语双语指令集，产生了超过6.32亿个医疗专业token以进行指令调整。我们的BiMed1.3M数据集包括25万个合成的医生-患者多轮对话，并保持1:2的阿拉伯语比例。

    arXiv:2402.13253v1 Announce Type: new  Abstract: In this paper, we introduce BiMediX, the first bilingual medical mixture of experts LLM designed for seamless interaction in both English and Arabic. Our model facilitates a wide range of medical interactions in English and Arabic, including multi-turn chats to inquire about additional details such as patient symptoms and medical history, multiple-choice question answering, and open-ended question answering. We propose a semi-automated English-to-Arabic translation pipeline with human refinement to ensure high-quality translations. We also introduce a comprehensive evaluation benchmark for Arabic medical LLMs. Furthermore, we introduce BiMed1.3M, an extensive Arabic-English bilingual instruction set covering 1.3 Million diverse medical interactions, resulting in over 632 million healthcare specialized tokens for instruction tuning. Our BiMed1.3M dataset includes 250k synthesized multi-turn doctor-patient chats and maintains a 1:2 Arabic-
    
[^5]: FTFT:通过转移训练动态实现高效且稳健的微调

    FTFT: Efficient and Robust Fine-Tuning by Transferring Training Dynamics

    [https://arxiv.org/abs/2310.06588](https://arxiv.org/abs/2310.06588)

    训练动态可在不同模型大小和预训练方法之间进行转移，通过选定的训练实例微调主模型实现比经验风险最小化更高的训练效率

    

    尽管微调预训练语言模型（PLMs）取得了巨大成功，但它们仍然容易受到分布外输入的影响。 数据集制图是一种简单而有效的双模型方法，可以提高微调PLMs的鲁棒性。 它涉及在原始训练集上微调模型（即参考模型），根据训练动态选择一些重要的训练实例，并仅对这些选定的示例再次进行微调（即主模型）。 然而，这种方法需要对同一模型进行两次微调，这对于大型PLMs而言在计算上是昂贵的。 在本文中，我们展示了（1）训练动态在模型大小和预训练方法之间具有高度可传递性，以及（2）使用这些选定的训练实例对主模型进行微调可以比经验风险最小化（ERM）实现更高的训练效率。 基于这些观察结果，我们提出了一种新颖的微调方法...

    arXiv:2310.06588v2 Announce Type: replace  Abstract: Despite the massive success of fine-tuning Pre-trained Language Models (PLMs), they remain susceptible to out-of-distribution input. Dataset cartography is a simple yet effective dual-model approach that improves the robustness of fine-tuned PLMs. It involves fine-tuning a model on the original training set (i.e. reference model), selecting a subset of important training instances based on the training dynamics, and fine-tuning again only on these selected examples (i.e. main model). However, this approach requires fine-tuning the same model twice, which is computationally expensive for large PLMs. In this paper, we show that (1) training dynamics are highly transferable across model sizes and pre-training methods, and that (2) fine-tuning main models using these selected training instances achieves higher training efficiency than empirical risk minimization (ERM). Building on these observations, we propose a novel fine-tuning approa
    

