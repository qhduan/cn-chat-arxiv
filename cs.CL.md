# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [LoRA-as-an-Attack! Piercing LLM Safety Under The Share-and-Play Scenario](https://arxiv.org/abs/2403.00108) | LoRA作为攻击者渗透LLM安全，研究探讨了在共享与玩耍场景下可能实现的攻击机会。 |
| [^2] | [Auditing Counterfire: Evaluating Advanced Counterargument Generation with Evidence and Style](https://arxiv.org/abs/2402.08498) | 这项研究提出了一个新的数据集，用于生成具有证据和风格的反驳，该数据集基于Reddit ChangeMyView数据集中的帖子，并可用于论证的改进和评估。评估结果显示，GPT-3.5 turbo模型在论证质量方面表现出色，并且具有很高的风格融合能力。互惠式反驳的效果最佳。 |
| [^3] | [LegalDuet: Learning Effective Representations for Legal Judgment Prediction through a Dual-View Legal Clue Reasoning.](http://arxiv.org/abs/2401.15371) | LegalDuet是一种通过双视角法律线索推理模型，使用预训练语言模型学习定制嵌入空间来进行法律判决预测。该模型通过法律案例推理和法律基础推理两个推理链进行判决。在实验中，LegalDuet在CAIL2018数据集上表现出最先进的性能，并超过了基线模型。 |
| [^4] | [Connecting Large Language Models with Evolutionary Algorithms Yields Powerful Prompt Optimizers.](http://arxiv.org/abs/2309.08532) | 本文提出了一种通过连接大型语言模型和进化算法进行提示优化的框架，名为EvoPrompt。通过利用大型语言模型的语言处理能力和进化算法的优化性能，EvoPrompt可以自动化处理需要连贯和可读性良好的提示，提高大型语言模型的性能。 |

# 详细

[^1]: 将LoRA作为攻击！在Share-and-Play场景下穿透LLM安全

    LoRA-as-an-Attack! Piercing LLM Safety Under The Share-and-Play Scenario

    [https://arxiv.org/abs/2403.00108](https://arxiv.org/abs/2403.00108)

    LoRA作为攻击者渗透LLM安全，研究探讨了在共享与玩耍场景下可能实现的攻击机会。

    

    对LLMs进行微调对于增强其特定任务的性能并确保模型行为与人类偏好保持一致至关重要。在各种微调方法中，LoRA因其效率和易用性而备受推崇，允许最终用户轻松在开源平台上发布和采用轻量的LoRA模块，以定制其模型以适应不同需求。然而，这种方便的共享与玩耍设置打开了新的攻击面，攻击者可以将LoRA作为攻击者，例如背门注入，并广泛分发对抗性LoRA给社区。这可能导致不利的后果。尽管共享LoRA模块存在巨大的潜在风险，但这一方面尚未得到充分探讨。为了填补这一空白，在本研究中，我们深入探讨了在不断增长的共享与玩耍场景中可能做出的攻击机会。具体而言，我们研究了如何将后门注入LoRA模块并深入探讨。

    arXiv:2403.00108v1 Announce Type: cross  Abstract: Fine-tuning LLMs is crucial to enhancing their task-specific performance and ensuring model behaviors are aligned with human preferences. Among various fine-tuning methods, LoRA is popular for its efficiency and ease to use, allowing end-users to easily post and adopt lightweight LoRA modules on open-source platforms to tailor their model for different customization. However, such a handy share-and-play setting opens up new attack surfaces, that the attacker can render LoRA as an attacker, such as backdoor injection, and widely distribute the adversarial LoRA to the community easily. This can result in detrimental outcomes. Despite the huge potential risks of sharing LoRA modules, this aspect however has not been fully explored. To fill the gap, in this study we thoroughly investigate the attack opportunities enabled in the growing share-and-play scenario. Specifically, we study how to inject backdoor into the LoRA module and dive deep
    
[^2]: 审计反火：评估具有证据和风格的先进反驳生成

    Auditing Counterfire: Evaluating Advanced Counterargument Generation with Evidence and Style

    [https://arxiv.org/abs/2402.08498](https://arxiv.org/abs/2402.08498)

    这项研究提出了一个新的数据集，用于生成具有证据和风格的反驳，该数据集基于Reddit ChangeMyView数据集中的帖子，并可用于论证的改进和评估。评估结果显示，GPT-3.5 turbo模型在论证质量方面表现出色，并且具有很高的风格融合能力。互惠式反驳的效果最佳。

    

    我们提出了一个新颖的数据集，用于控制性反驳的合成，旨在进一步应用于论证的改进、挖掘和评估。我们的数据集包含与Reddit ChangeMyView数据集中的帖子相结合的丰富的反驳，这些反驳融入了从高质量来源中检索到的证据，并根据用户偏好生成，调整了证据和论证风格的关键属性。由此产生的Counterfire语料库包括从GPT-3.5 turbo、Koala和PaLM 2模型以及它们的两个微调变体生成的论证（N = 32,000）。模型评估表明，在证据方面具有强大的改写能力，尽管词汇重叠有限，同时表现出高度的风格融合（对于“互惠”的得分为0.9682），显示了LLM融合多样风格的能力。在所有模型中，GPT-3.5 turbo在论证质量评估中显示出最高分数，表现出一致准确性（得分 >0.8）。在进一步的分析中，互惠式反驳证明效果最佳，能够产生更好的论证结果。

    We present a novel dataset for the controlled composition of counterarguments designed for further applications in argument refining, mining, and evaluation. Our dataset constitutes enriched counter-arguments to posts in the Reddit ChangeMyView dataset that are integrated with evidence retrieved from high-quality sources and generated based on user preferences, adjusting the critical attributes of evidence and argument style. The resultant Counterfire corpus comprises arguments generated from GPT-3.5 turbo, Koala, and PaLM 2 models and two of their finetuned variants (N = 32,000). Model evaluation indicates strong paraphrasing abilities with evidence, albeit limited word overlap, while demonstrating high style integration (0.9682 for 'reciprocity'), showing the ability of LLM to assimilate diverse styles. Of all models, GPT-3.5 turbo showed the highest scores in argument quality evaluation, showing consistent accuracy (score >0.8). In further analyses, reciprocity-style counterargument
    
[^3]: LegalDuet: 通过双视角法律线索推理学习有效的法律判决预测表示

    LegalDuet: Learning Effective Representations for Legal Judgment Prediction through a Dual-View Legal Clue Reasoning. (arXiv:2401.15371v1 [cs.CL])

    [http://arxiv.org/abs/2401.15371](http://arxiv.org/abs/2401.15371)

    LegalDuet是一种通过双视角法律线索推理模型，使用预训练语言模型学习定制嵌入空间来进行法律判决预测。该模型通过法律案例推理和法律基础推理两个推理链进行判决。在实验中，LegalDuet在CAIL2018数据集上表现出最先进的性能，并超过了基线模型。

    

    大多数现有的法律判决预测（LJP）模型侧重于发现刑事事实描述中的法律线索。然而，在现实场景中，专业法官不仅需要吸收过去判决的法律案例经验，还依赖于从专业法律知识中学到的专业法律基础推理。本文提出了一种名为LegalDuet的模型，该模型预训练语言模型以学习用于进行法律判决的定制嵌入空间。它提出了一种双视角法律线索推理机制，由两个推理链组成：1）法律案例推理，根据从类比/混淆的法律案例中学到的判决经验进行法律判决；2）法律基础推理，通过匹配刑事案件和法律决定之间的法律线索。实验证明，LegalDuet在CAIL2018数据集上实现了最先进的性能，并且超过了基线模型。

    Most existing Legal Judgment Prediction (LJP) models focus on discovering the legal triggers in the criminal fact description. However, in real-world scenarios, a professional judge not only needs to assimilate the law case experience that thrives on past sentenced legal judgments but also depends on the professional legal grounded reasoning that learned from professional legal knowledge. In this paper, we propose a LegalDuet model, which pretrains language models to learn a tailored embedding space for making legal judgments. It proposes a dual-view legal clue reasoning mechanism, which derives from two reasoning chains of judges: 1) Law Case Reasoning, which makes legal judgments according to the judgment experiences learned from analogy/confusing legal cases; 2) Legal Ground Reasoning, which lies in matching the legal clues between criminal cases and legal decisions. Our experiments show that LegalDuet achieves state-of-the-art performance on the CAIL2018 dataset and outperforms bas
    
[^4]: 通过进化算法连接大型语言模型与强大的提示优化器

    Connecting Large Language Models with Evolutionary Algorithms Yields Powerful Prompt Optimizers. (arXiv:2309.08532v1 [cs.CL])

    [http://arxiv.org/abs/2309.08532](http://arxiv.org/abs/2309.08532)

    本文提出了一种通过连接大型语言模型和进化算法进行提示优化的框架，名为EvoPrompt。通过利用大型语言模型的语言处理能力和进化算法的优化性能，EvoPrompt可以自动化处理需要连贯和可读性良好的提示，提高大型语言模型的性能。

    

    大型语言模型在各种任务中表现出色，但它们依赖于精心设计的提示，这通常需要大量的人力努力。为了自动化这个过程，本文提出了一种新颖的离散提示优化框架，称为EvoPrompt，它借鉴了进化算法的思想，因为它们表现出良好的性能和快速的收敛性。为了使进化算法能够处理需要连贯并且可读性良好的自然语言表达的离散提示，我们将大型语言模型与进化算法进行了连接。这种方法使我们可以同时利用大型语言模型的强大语言处理能力和进化算法的高效优化性能。具体而言，EvoPrompt在不使用任何梯度或参数的情况下，从一组提示中开始，并基于进化算子通过大型语言模型生成新的提示，根据开发集改进提示的种群。我们对闭源和开源的大型语言模型，包括GPT-3进行提示优化。

    Large Language Models (LLMs) excel in various tasks, but they rely on carefully crafted prompts that often demand substantial human effort. To automate this process, in this paper, we propose a novel framework for discrete prompt optimization, called EvoPrompt, which borrows the idea of evolutionary algorithms (EAs) as they exhibit good performance and fast convergence. To enable EAs to work on discrete prompts, which are natural language expressions that need to be coherent and human-readable, we connect LLMs with EAs. This approach allows us to simultaneously leverage the powerful language processing capabilities of LLMs and the efficient optimization performance of EAs. Specifically, abstaining from any gradients or parameters, EvoPrompt starts from a population of prompts and iteratively generates new prompts with LLMs based on the evolutionary operators, improving the population based on the development set. We optimize prompts for both closed- and open-source LLMs including GPT-3
    

