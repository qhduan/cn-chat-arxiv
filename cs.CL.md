# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Automatic Interactive Evaluation for Large Language Models with State Aware Patient Simulator](https://arxiv.org/abs/2403.08495) | 介绍了自动互动评估（AIE）框架和状态感知病人模拟器（SAPS），以动态、真实的平台评估LLMs，弥补传统评估方法无法满足临床任务需求的不足。 |
| [^2] | [Not all Layers of LLMs are Necessary during Inference](https://arxiv.org/abs/2403.02181) | 推理过程中，根据输入实例的不同难易程度，本文提出了一种名为AdaInfer的算法，可以自适应地使用浅层和深层，从而节省了计算资源。 |
| [^3] | [Reading Subtext: Evaluating Large Language Models on Short Story Summarization with Writers](https://arxiv.org/abs/2403.01061) | 评估大型语言模型在短篇小说摘要上的表现，发现它们在忠实性和解释潜台词方面存在挑战，但在进行主题分析时表现出思考深度。 |
| [^4] | [Probabilistically-sound beam search with masked language models](https://arxiv.org/abs/2402.15020) | 提出了在掩码语言模型上进行束搜索的概率健壮方法，表明其在多个领域中优于传统方法。 |
| [^5] | [Beyond Probabilities: Unveiling the Misalignment in Evaluating Large Language Models](https://arxiv.org/abs/2402.13887) | 本研究揭示了在使用大型语言模型进行多项选择题时，基于概率的评估方法与基于生成的预测不相吻合的固有局限性。 |
| [^6] | [Grounding Language about Belief in a Bayesian Theory-of-Mind](https://arxiv.org/abs/2402.10416) | 语义基础置于贝叶斯心灵理论中，通过模拟人们共同推断出解释代理人行为的一致性目标、信念和计划集合，再通过认识逻辑评估有关代理人信念的陈述，解释了人类信念归因的分级性和组合性，以及其与目标和计划的密切联系。 |
| [^7] | [Robust Prompt Optimization for Defending Language Models Against Jailbreaking Attacks](https://arxiv.org/abs/2401.17263) | 该论文提出了一种鲁棒的提示优化算法（RPO）用于对抗语言模型的破解攻击，通过梯度优化来确保输出的无害性，并成功降低了攻击成功率。 |
| [^8] | [Chain-of-Dictionary Prompting Elicits Translation in Large Language Models.](http://arxiv.org/abs/2305.06575) | 研究通过在大型语言模型中添加字典链提示的方法来改进低资源语言的翻译能力，实验结果表明能显著提高翻译质量。 |
| [^9] | [Machine Psychology: Investigating Emergent Capabilities and Behavior in Large Language Models Using Psychological Methods.](http://arxiv.org/abs/2303.13988) | 本文提出了一种新领域——机器心理学，利用心理学的方法考察大型语言模型的能力。该文规范了机器心理学研究的方法论标准，并对心理实验中提示设计政策进行了探讨和制定。 |

# 详细

[^1]: 具有状态感知病人模拟器的大型语言模型自动互动评估

    Automatic Interactive Evaluation for Large Language Models with State Aware Patient Simulator

    [https://arxiv.org/abs/2403.08495](https://arxiv.org/abs/2403.08495)

    介绍了自动互动评估（AIE）框架和状态感知病人模拟器（SAPS），以动态、真实的平台评估LLMs，弥补传统评估方法无法满足临床任务需求的不足。

    

    大型语言模型（LLMs）在人机互动中表现出色，但在医疗领域的应用仍未得到充分探索。本文引入了自动互动评估（AIE）框架和状态感知病人模拟器（SAPS），旨在弥补传统LLM评估与临床实践的微妙需求之间的差距。

    arXiv:2403.08495v1 Announce Type: new  Abstract: Large Language Models (LLMs) have demonstrated remarkable proficiency in human interactions, yet their application within the medical field remains insufficiently explored. Previous works mainly focus on the performance of medical knowledge with examinations, which is far from the realistic scenarios, falling short in assessing the abilities of LLMs on clinical tasks. In the quest to enhance the application of Large Language Models (LLMs) in healthcare, this paper introduces the Automated Interactive Evaluation (AIE) framework and the State-Aware Patient Simulator (SAPS), targeting the gap between traditional LLM evaluations and the nuanced demands of clinical practice. Unlike prior methods that rely on static medical knowledge assessments, AIE and SAPS provide a dynamic, realistic platform for assessing LLMs through multi-turn doctor-patient simulations. This approach offers a closer approximation to real clinical scenarios and allows f
    
[^2]: 推理过程中不是所有LLMs的层都是必要的

    Not all Layers of LLMs are Necessary during Inference

    [https://arxiv.org/abs/2403.02181](https://arxiv.org/abs/2403.02181)

    推理过程中，根据输入实例的不同难易程度，本文提出了一种名为AdaInfer的算法，可以自适应地使用浅层和深层，从而节省了计算资源。

    

    大型语言模型（LLMs）的推理阶段非常昂贵。理想的LLMs推理阶段可以利用更少的计算资源，同时仍保持其能力（例如泛化和上下文学习能力）。本文尝试回答一个问题：“在LLMs推理过程中，我们可以为简单实例使用浅层，并为难以处理的实例使用深层吗？”为了回答这个问题，我们首先通过统计分析跨任务激活的层来指出并非所有层在推理过程中都是必要的。然后，我们提出了一种简单的算法，名为AdaInfer，根据输入实例自适应地确定推理终止时刻。更重要的是，AdaInfer不改变LLMs参数，并在任务之间保持泛化能力。对知名LLMs（即Llama2系列和OPT）的实验证明，AdaInfer节省了平均14.8%的计算资源，甚至在情感方面高达50%。

    arXiv:2403.02181v1 Announce Type: cross  Abstract: The inference phase of Large Language Models (LLMs) is very expensive. An ideal inference stage of LLMs could utilize fewer computational resources while still maintaining its capabilities (e.g., generalization and in-context learning ability). In this paper, we try to answer the question, "During LLM inference, can we use shallow layers for easy instances; and deep layers for hard ones?" To answer this question, we first indicate that Not all Layers are Necessary during Inference by statistically analyzing the activated layers across tasks. Then, we propose a simple algorithm named AdaInfer to determine the inference termination moment based on the input instance adaptively. More importantly, AdaInfer does not alter LLM parameters and maintains generalizability across tasks. Experiments on well-known LLMs (i.e., Llama2 series and OPT) show that AdaInfer saves an average of 14.8% of computational resources, even up to 50% on sentiment 
    
[^3]: 阅读潜台词：在短篇小说摘要上评估大型语言模型与作者合作

    Reading Subtext: Evaluating Large Language Models on Short Story Summarization with Writers

    [https://arxiv.org/abs/2403.01061](https://arxiv.org/abs/2403.01061)

    评估大型语言模型在短篇小说摘要上的表现，发现它们在忠实性和解释潜台词方面存在挑战，但在进行主题分析时表现出思考深度。

    

    我们评估了最近的大型语言模型（LLMs）在摘要长篇文学作品这一具有挑战性的任务上的表现，这些作品可能长度较长，并包含微妙的潜台词或错综复杂的时间线。重要的是，我们直接与作者合作，确保这些作品尚未在网络上分享过（因此对这些模型是未知的），并获得作者本人对摘要质量的明确评价。通过基于叙事理论的定量和定性分析，我们比较了GPT-4、Claude-2.1和LLama-2-70B。我们发现这三个模型在50%以上的摘要中会出现忠实性错误，并且难以解释难以理解的潜台词。然而，在最佳状态下，这些模型可以对故事进行有深度的主题分析。此外，我们还展示了LLMs对摘要质量的判断与作家的反馈不一致。

    arXiv:2403.01061v1 Announce Type: new  Abstract: We evaluate recent Large language Models (LLMs) on the challenging task of summarizing short stories, which can be lengthy, and include nuanced subtext or scrambled timelines. Importantly, we work directly with authors to ensure that the stories have not been shared online (and therefore are unseen by the models), and to obtain informed evaluations of summary quality using judgments from the authors themselves. Through quantitative and qualitative analysis grounded in narrative theory, we compare GPT-4, Claude-2.1, and LLama-2-70B. We find that all three models make faithfulness mistakes in over 50% of summaries and struggle to interpret difficult subtext. However, at their best, the models can provide thoughtful thematic analysis of stories. We additionally demonstrate that LLM judgments of summary quality do not match the feedback from the writers.
    
[^4]: 具有掩码语言模型的概率健壮束搜索

    Probabilistically-sound beam search with masked language models

    [https://arxiv.org/abs/2402.15020](https://arxiv.org/abs/2402.15020)

    提出了在掩码语言模型上进行束搜索的概率健壮方法，表明其在多个领域中优于传统方法。

    

    具有掩码语言模型（MLMs）的束搜索存在挑战，部分原因是由于序列的联合概率分布不像自回归模型那样readily available。然而，估算这样的分布在许多领域中具有应用，包括蛋白工程和古代文本恢复。我们提出了一种具有概率健壮性的使用MLMs进行束搜索的方法。首先，我们阐明了在哪些条件下使用标准束搜索对MLMs执行文本填充在理论上是可靠的。当这些条件失败时，我们提供了一种具有概率健壮性的修改，而且无需额外的计算复杂性，并且证明在预期条件下它优于前述的束搜索。然后，我们提出了比较多个领域中几种使用MLMs进行填充的方法的经验结果。

    arXiv:2402.15020v1 Announce Type: cross  Abstract: Beam search with masked language models (MLMs) is challenging in part because joint probability distributions over sequences are not readily available, unlike for autoregressive models. Nevertheless, estimating such distributions has applications in many domains, including protein engineering and ancient text restoration. We present probabilistically-sound methods for beam search with MLMs. First, we clarify the conditions under which it is theoretically sound to perform text infilling with MLMs using standard beam search. When these conditions fail, we provide a probabilistically-sound modification with no additional computational complexity and demonstrate that it is superior to the aforementioned beam search in the expected conditions. We then present empirical results comparing several infilling approaches with MLMs across several domains.
    
[^5]: 超越概率：揭示评估大型语言模型中的错位问题

    Beyond Probabilities: Unveiling the Misalignment in Evaluating Large Language Models

    [https://arxiv.org/abs/2402.13887](https://arxiv.org/abs/2402.13887)

    本研究揭示了在使用大型语言模型进行多项选择题时，基于概率的评估方法与基于生成的预测不相吻合的固有局限性。

    

    大型语言模型（LLMs）在各种应用中展现出卓越的能力，从根本上改变了自然语言处理（NLP）研究的格局。然而，最近的评估框架通常依赖于LLMs的输出概率进行预测，主要是由于计算约束，偏离了真实世界的LLMs使用场景。虽然被广泛采用，基于概率的评估策略的有效性仍是一个开放的研究问题。本研究旨在审查这种基于概率的评估方法在使用LLMs进行多项选择题（MCQs）时的有效性，突显其固有局限性。我们的实证调查显示，普遍的基于概率的评估方法未能与基于生成的预测相适应。此外，当前的评估框架通常通过基于输出预测的预测任务来评估LLMs

    arXiv:2402.13887v1 Announce Type: new  Abstract: Large Language Models (LLMs) have demonstrated remarkable capabilities across various applications, fundamentally reshaping the landscape of natural language processing (NLP) research. However, recent evaluation frameworks often rely on the output probabilities of LLMs for predictions, primarily due to computational constraints, diverging from real-world LLM usage scenarios. While widely employed, the efficacy of these probability-based evaluation strategies remains an open research question. This study aims to scrutinize the validity of such probability-based evaluation methods within the context of using LLMs for Multiple Choice Questions (MCQs), highlighting their inherent limitations. Our empirical investigation reveals that the prevalent probability-based evaluation method inadequately aligns with generation-based prediction. Furthermore, current evaluation frameworks typically assess LLMs through predictive tasks based on output pr
    
[^6]: 将关于信念的语言接地于贝叶斯心灵理论

    Grounding Language about Belief in a Bayesian Theory-of-Mind

    [https://arxiv.org/abs/2402.10416](https://arxiv.org/abs/2402.10416)

    语义基础置于贝叶斯心灵理论中，通过模拟人们共同推断出解释代理人行为的一致性目标、信念和计划集合，再通过认识逻辑评估有关代理人信念的陈述，解释了人类信念归因的分级性和组合性，以及其与目标和计划的密切联系。

    

    尽管信念是无法直接观察的心理状态，人类常常使用丰富的组合语言来描述他人的想法和知识。这项研究通过将信念陈述的语义基础置于贝叶斯心灵理论中，为解释人类如何解释他人隐藏的认识内容迈出了一步：通过建模人类如何共同推断出解释一个代理人行动的一致性目标、信念和计划集合，然后通过认识逻辑对有关代理人信念的陈述进行评估，我们的框架为信念提供了一个概念角色语义，解释了人类信念归因的分级性和组合性，以及它们与目标和计划的密切联系。我们通过研究人们在观察一个代理人解决问题时是如何归因目标和信念的来评估这一框架。

    arXiv:2402.10416v1 Announce Type: new  Abstract: Despite the fact that beliefs are mental states that cannot be directly observed, humans talk about each others' beliefs on a regular basis, often using rich compositional language to describe what others think and know. What explains this capacity to interpret the hidden epistemic content of other minds? In this paper, we take a step towards an answer by grounding the semantics of belief statements in a Bayesian theory-of-mind: By modeling how humans jointly infer coherent sets of goals, beliefs, and plans that explain an agent's actions, then evaluating statements about the agent's beliefs against these inferences via epistemic logic, our framework provides a conceptual role semantics for belief, explaining the gradedness and compositionality of human belief attributions, as well as their intimate connection with goals and plans. We evaluate this framework by studying how humans attribute goals and beliefs while watching an agent solve
    
[^7]: 鲁棒的提示优化用于对抗语言模型的破解攻击

    Robust Prompt Optimization for Defending Language Models Against Jailbreaking Attacks

    [https://arxiv.org/abs/2401.17263](https://arxiv.org/abs/2401.17263)

    该论文提出了一种鲁棒的提示优化算法（RPO）用于对抗语言模型的破解攻击，通过梯度优化来确保输出的无害性，并成功降低了攻击成功率。

    

    尽管在人工智能对齐方面取得了一些进展，但语言模型（LM）仍然容易受到对抗性攻击或破解攻击的影响，其中对手修改输入提示以诱导有害行为。虽然已经提出了一些防御方法，但它们仅关注狭窄的威胁模型，并不能提供强大的防御。为了实现强大的防御，我们首次提出了用于对抗破解攻击的对抗目标，并提出了一种名为鲁棒提示优化（RPO）的算法，该算法利用基于梯度的令牌优化来确保输出的无害性。通过这种方法，我们得到了一个易于访问的后缀，显著改善了对破解攻击的强韧性，包括优化过程中出现的破解攻击以及未知的破解攻击，将攻击成功率从84%降低到8.66%，在20个破解攻击中。此外，我们还发现RPO对正常LM使用的影响较小，在适应性攻击下仍然有效，并且可以迁移到黑盒模型中，降低攻击成功率。

    Despite advances in AI alignment, language models (LM) remain vulnerable to adversarial attacks or jailbreaking, in which adversaries modify input prompts to induce harmful behavior. While some defenses have been proposed, they focus on narrow threat models and fall short of a strong defense, which we posit should be effective, universal, and practical. To achieve this, we propose the first adversarial objective for defending LMs against jailbreaking attacks and an algorithm, robust prompt optimization (RPO), that uses gradient-based token optimization to enforce harmless outputs. This results in an easily accessible suffix that significantly improves robustness to both jailbreaks seen during optimization and unknown, held-out jailbreaks, reducing the attack success rate on Starling-7B from 84% to 8.66% across 20 jailbreaks. In addition, we find that RPO has a minor effect on normal LM use, is successful under adaptive attacks, and can transfer to black-box models, reducing the success
    
[^8]: 大型语言模型中的字典链提示在翻译中的应用

    Chain-of-Dictionary Prompting Elicits Translation in Large Language Models. (arXiv:2305.06575v1 [cs.CL])

    [http://arxiv.org/abs/2305.06575](http://arxiv.org/abs/2305.06575)

    研究通过在大型语言模型中添加字典链提示的方法来改进低资源语言的翻译能力，实验结果表明能显著提高翻译质量。

    

    大型语言模型(LLMs)在多语言神经机器翻译(MNMT)中表现出惊人的性能，即使没有平行数据也能训练。然而，尽管训练数据量巨大，它们仍然难以翻译稀有词汇，特别是对于低资源语言。更糟糕的是，通常情况下，在低资源语言上，很难检索到相关示范来进行上下文学习，这限制了LLMs在翻译方面的实际应用——我们该如何缓解这个问题？为此，我们提出了一种新的方法，CoD，通过使用多语言字典链为一部分输入单词增加LLMs的先前知识，从而促进LLMs的翻译能力。广泛的实验表明，在FLORES-200全开发测试集上，通过将CoD和ChatGPT相结合，可以获得高达13倍的MNMT ChrF++分数的收益（英语到塞尔维亚语，西里尔字母书写，ChrF ++分数从3.08增加到42.63）。我们进一步展示了该方法在其他数据集上的重要作用。

    Large language models (LLMs) have shown surprisingly good performance in multilingual neural machine translation (MNMT) even when trained without parallel data. Yet, despite the fact that the amount of training data is gigantic, they still struggle with translating rare words, particularly for low-resource languages. Even worse, it is usually unrealistic to retrieve relevant demonstrations for in-context learning with low-resource languages on LLMs, which restricts the practical use of LLMs for translation -- how should we mitigate this problem? To this end, we present a novel method, CoD, which augments LLMs with prior knowledge with the chains of multilingual dictionaries for a subset of input words to elicit translation abilities for LLMs. Extensive experiments indicate that augmenting ChatGPT with CoD elicits large gains by up to 13x ChrF++ points for MNMT (3.08 to 42.63 for English to Serbian written in Cyrillic script) on FLORES-200 full devtest set. We further demonstrate the im
    
[^9]: 机器心理学：利用心理学方法探究大型语言模型的新兴能力和行为

    Machine Psychology: Investigating Emergent Capabilities and Behavior in Large Language Models Using Psychological Methods. (arXiv:2303.13988v1 [cs.CL])

    [http://arxiv.org/abs/2303.13988](http://arxiv.org/abs/2303.13988)

    本文提出了一种新领域——机器心理学，利用心理学的方法考察大型语言模型的能力。该文规范了机器心理学研究的方法论标准，并对心理实验中提示设计政策进行了探讨和制定。

    

    大型语言模型（LLM）是将人工智能系统与人类交流和日常生活紧密结合的先锋。由于快速技术进步和其极高的通用性，现今LLM已经拥有数百万用户，并正处于成为主要信息检索、内容生成、问题解决等技术的前沿。因此，对其进行全面评估和审查显得尤为重要。由于当前LLM中出现愈加复杂和新颖的行为模式，可将其视为参与人类心理实验的对象，以便更为全面地评估其能力。为此，本文引入了一个名为"机器心理学"的新兴研究领域。本文概述了各类心理学分支如何为LLM的行为测试提供有用参考。同时，本文规范了机器心理学研究的方法论标准，特别是专注于提示设计政策的制定。此外，它还描述了行为测试结果如何为未来的LLM发展提供指导。

    Large language models (LLMs) are currently at the forefront of intertwining AI systems with human communication and everyday life. Due to rapid technological advances and their extreme versatility, LLMs nowadays have millions of users and are at the cusp of being the main go-to technology for information retrieval, content generation, problem-solving, etc. Therefore, it is of great importance to thoroughly assess and scrutinize their capabilities. Due to increasingly complex and novel behavioral patterns in current LLMs, this can be done by treating them as participants in psychology experiments that were originally designed to test humans. For this purpose, the paper introduces a new field of research called "machine psychology". The paper outlines how different subfields of psychology can inform behavioral tests for LLMs. It defines methodological standards for machine psychology research, especially by focusing on policies for prompt designs. Additionally, it describes how behaviora
    

