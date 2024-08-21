# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [What's in Your "Safe" Data?: Identifying Benign Data that Breaks Safety](https://arxiv.org/abs/2404.01099) | 通过双向锚定方法，识别那些在微调后更可能降低模型安全性的良性数据子集，提高模型对有害请求的响应率。 |
| [^2] | [CAUSE: Counterfactual Assessment of User Satisfaction Estimation in Task-Oriented Dialogue Systems](https://arxiv.org/abs/2403.19056) | 本文通过利用大型语言模型生成满意感知的反事实对话来增加任务型对话系统的原始对话集合，以改善用户满意度估计的鲁棒性。 |
| [^3] | [Information-Theoretic Distillation for Reference-less Summarization](https://arxiv.org/abs/2403.13780) | 提出了一种名为InfoSumm的框架，通过信息论目标实现了无参考摘要的精炼生成器 |
| [^4] | [SoftTiger: A Clinical Foundation Model for Healthcare Workflows](https://arxiv.org/abs/2403.00868) | SoftTiger是一个专为医疗工作流设计的临床大型语言模型，通过处理临床笔记的结构化，实现了基本临床任务以及更复杂的下游临床任务的执行。 |
| [^5] | [Strong hallucinations from negation and how to fix them](https://arxiv.org/abs/2402.10543) | 论文针对语言模型在推理中造成的强幻觉问题，提出了一种处理否定的新方法，可以改善模型性能而无需使用稀疏负数据训练。 |
| [^6] | [PsySafe: A Comprehensive Framework for Psychological-based Attack, Defense, and Evaluation of Multi-agent System Safety](https://arxiv.org/abs/2401.11880) | PsySafe提出了一个综合框架，通过深入探讨智能体心理学，揭示智能体的黑暗心理状态对安全构成威胁，并提出了有效的风险缓解策略。 |
| [^7] | [F-Eval: Asssessing Fundamental Abilities with Refined Evaluation Methods.](http://arxiv.org/abs/2401.14869) | F-Eval是一个双语评估基准，用于评估大型语言模型的基本能力，包括表达、常识和逻辑。它采用多种任务形式进行评估，包括客观任务和主观任务，并提出了新的评估方法来解决无参考的主观任务评估问题。 |
| [^8] | [CodePrompt: Improving Source Code-Related Classification with Knowledge Features through Prompt Learning.](http://arxiv.org/abs/2401.05544) | CodePrompt是一种利用Prompt学习和注意机制技术改进源代码相关分类任务的新方法。它能够提取源代码和相关文本中的丰富知识以提高准确性，并且减少了计算成本。 |
| [^9] | [PromptBench: A Unified Library for Evaluation of Large Language Models.](http://arxiv.org/abs/2312.07910) | PromptBench是一个用于评估大型语言模型的统一库，包括了提示语构建、提示语工程、数据集和模型加载、对抗性提示攻击、动态评估协议和分析工具等组件，旨在促进原创研究和创建新的基准测试、部署下游应用以及设计新的评估协议。 |
| [^10] | [Fake News in Sheep's Clothing: Robust Fake News Detection Against LLM-Empowered Style Attacks.](http://arxiv.org/abs/2310.10830) | 这篇论文介绍了一种鲁棒的无风格假新闻检测器，能够对抗利用大型语言模型进行风格攻击的假新闻。通过LLM增强的新闻重构，该检测器能够适应不同的写作风格，提高了对伪装假新闻的检测能力。 |
| [^11] | [The FruitShell French synthesis system at the Blizzard 2023 Challenge.](http://arxiv.org/abs/2309.00223) | 本文介绍了一个用于Blizzard Challenge 2023的法语文本到语音合成系统，通过对数据的筛选和增强，以及添加词边界和起始/结束符号的方式，提高了语音质量并进行了标准化转录。 |
| [^12] | [Finding Already Debunked Narratives via Multistage Retrieval: Enabling Cross-Lingual, Cross-Dataset and Zero-Shot Learning.](http://arxiv.org/abs/2308.05680) | 本研究通过创建新的数据集、评估多语言预训练Transformer模型以及提出多阶段框架来解决了跨语言澄清检索问题。 |
| [^13] | [MMBench: Is Your Multi-modal Model an All-around Player?.](http://arxiv.org/abs/2307.06281) | MMBench是一个新型的多模态基准测试，旨在解决大型视觉语言模型评估的挑战，通过开发全面的评估流程和精心策划的数据集进行细粒度能力评估。 |
| [^14] | [Causal Reasoning and Large Language Models: Opening a New Frontier for Causality.](http://arxiv.org/abs/2305.00050) | 大型语言模型在因果推理任务中取得了新的最高准确率，但是其鲁棒性仍然存在难以预测的失败模式。 |

# 详细

[^1]: 你的“安全”数据中有什么？：识别破坏安全性的良性数据

    What's in Your "Safe" Data?: Identifying Benign Data that Breaks Safety

    [https://arxiv.org/abs/2404.01099](https://arxiv.org/abs/2404.01099)

    通过双向锚定方法，识别那些在微调后更可能降低模型安全性的良性数据子集，提高模型对有害请求的响应率。

    

    当前的大型语言模型（LLMs），即使经过调整以确保安全性和对齐性，也容易被越狱。一些研究表明，只是进一步使用良性数据（即没有有害内容的数据）对一个对齐模型进行微调，会导致安全性大幅下降。我们深入探讨良性微调不经意间导致越狱的数据中心方面。首先，我们通过两种视角表征微调数据：表示和梯度空间。此外，我们提出了一种双向锚定方法，该方法优先考虑靠近有害示例并远离良性示例的数据点。通过这样做，我们的方法有效地识别出更有可能在微调后降低模型安全性的良性数据子集。仅仅训练100个这些看似良性的数据点，就可以使微调模型肯定地回应超过70％的被测试的有害请求，相比之下，...

    arXiv:2404.01099v1 Announce Type: cross  Abstract: Current Large Language Models (LLMs), even those tuned for safety and alignment, are susceptible to jailbreaking. Some have found that just further fine-tuning an aligned model with benign data (i.e., data without harmful content) surprisingly leads to substantial degradation in safety. We delve into the data-centric aspects of why benign fine-tuning inadvertently contributes to jailbreaking. First, we represent fine-tuning data through two lenses: representation and gradient spaces. Furthermore, we propose a bi-directional anchoring method that prioritizes data points that are close to harmful examples and distant from benign ones. By doing so, our approach effectively identifies subsets of benign data that are more likely to degrade the model's safety after fine-tuning. Training on just 100 of these seemingly benign datapoints can lead to the fine-tuned model affirmatively responding to > 70% of tested harmful requests, compared to <
    
[^2]: CAUSE: 在面向任务型对话系统中利用反事实评估用户满意度估计

    CAUSE: Counterfactual Assessment of User Satisfaction Estimation in Task-Oriented Dialogue Systems

    [https://arxiv.org/abs/2403.19056](https://arxiv.org/abs/2403.19056)

    本文通过利用大型语言模型生成满意感知的反事实对话来增加任务型对话系统的原始对话集合，以改善用户满意度估计的鲁棒性。

    

    先前关于任务型对话系统中用户满意度估计的工作中一个重要但未被探索的方面是对其在识别用户不满意方面的鲁棒性进行评估：当前用于任务型对话系统中用户满意度估计的基准测试高度倾向于用户满意的对话。具有更平衡满意度标签集合对性能的影响是未知的。然而，通过更多的不满对话样本平衡数据需要进一步的数据收集和人工注释，这是昂贵和耗时的。本工作中，我们利用大型语言模型（LLMs）并解锁其生成满意感知反事实对话的能力，以增加测试集合的原始对话集合。我们收集人工注释以确保生成样本的可靠性。我们评估两个开源LLM作为用户满意度估计器。

    arXiv:2403.19056v1 Announce Type: new  Abstract: An important unexplored aspect in previous work on user satisfaction estimation for Task-Oriented Dialogue (TOD) systems is their evaluation in terms of robustness for the identification of user dissatisfaction: current benchmarks for user satisfaction estimation in TOD systems are highly skewed towards dialogues for which the user is satisfied. The effect of having a more balanced set of satisfaction labels on performance is unknown. However, balancing the data with more dissatisfactory dialogue samples requires further data collection and human annotation, which is costly and time-consuming. In this work, we leverage large language models (LLMs) and unlock their ability to generate satisfaction-aware counterfactual dialogues to augment the set of original dialogues of a test collection. We gather human annotations to ensure the reliability of the generated samples. We evaluate two open-source LLMs as user satisfaction estimators on our
    
[^3]: 无参考摘要的信息论精炼

    Information-Theoretic Distillation for Reference-less Summarization

    [https://arxiv.org/abs/2403.13780](https://arxiv.org/abs/2403.13780)

    提出了一种名为InfoSumm的框架，通过信息论目标实现了无参考摘要的精炼生成器

    

    当前自动摘要的主要方法是使用专有的大规模语言模型（LLMs）如ChatGPT，或者从它们作为教师模型进行模仿学习。本文提出了一种名为InfoSumm的新型框架，通过信息论目标进行精炼强大的摘要生成器，而不依赖于LLM的能力或人工编写的参考文献。

    arXiv:2403.13780v1 Announce Type: new  Abstract: The current winning recipe for automatic summarization is using proprietary large-scale language models (LLMs) such as ChatGPT as is, or imitation learning from them as teacher models. While increasingly ubiquitous dependence on such large-scale language models is convenient, there remains an important question of whether small-scale models could have achieved competitive results, if we were to seek an alternative learning method -- that allows for a more cost-efficient, controllable, yet powerful summarizer. We present InfoSumm, a novel framework to distill a powerful summarizer based on the information-theoretic objective for summarization, without relying on either the LLM's capability or human-written references. To achieve this, we first propose a novel formulation of the desiderata of summarization (saliency, faithfulness and brevity) through the lens of mutual information between the original document and the summary. Based on thi
    
[^4]: SoftTiger: 用于医疗工作流的临床基础模型

    SoftTiger: A Clinical Foundation Model for Healthcare Workflows

    [https://arxiv.org/abs/2403.00868](https://arxiv.org/abs/2403.00868)

    SoftTiger是一个专为医疗工作流设计的临床大型语言模型，通过处理临床笔记的结构化，实现了基本临床任务以及更复杂的下游临床任务的执行。

    

    我们发布并介绍了SoftTiger，一个专为医疗保健工作流设计的临床大型语言模型（CLaM）作为基础模型。临床笔记的叙述性和非结构化特性是医疗智能化的主要障碍。我们致力于按照国际互操作性标准将临床笔记结构化为临床数据，涉及国际患者摘要、临床印象和医疗接触三个关键子任务的数据收集和标注。然后，我们使用公开和验证的临床数据对最先进的LLM进行监督微调。训练过程中，目标模型首先能够支持基本的临床任务，如缩写扩展和时间信息提取，然后学习执行更复杂的下游临床任务，如印象和接触摘要。此外，我们解决了医疗模型中的一些建模挑战。

    arXiv:2403.00868v1 Announce Type: cross  Abstract: We release and introduce SoftTiger, a clinical large language model (CLaM) designed as a foundation model for healthcare workflows. The narrative and unstructured nature of clinical notes is a major obstacle for healthcare intelligentization. We address a critical problem of structuring clinical notes into clinical data, according to international interoperability standards. We collect and annotate data for three critical subtasks, namely, international patient summary, clinical impression and medical encounter. We then supervised fine-tuned a state-of-the-art LLM using public and credentialed clinical data. The training is orchestrated in a way that the target model can first support basic clinical tasks such as abbreviation expansion and temporal information extraction, and then learn to perform more complex downstream clinical tasks such as impression and encounter summary. Moreover, we address, several modeling challenges in the he
    
[^5]: 消除否定导致的强幻觉

    Strong hallucinations from negation and how to fix them

    [https://arxiv.org/abs/2402.10543](https://arxiv.org/abs/2402.10543)

    论文针对语言模型在推理中造成的强幻觉问题，提出了一种处理否定的新方法，可以改善模型性能而无需使用稀疏负数据训练。

    

    尽管语言模型（LMs）在许多任务上表现出色，但仍然在推理方面存在困难，有时会提供由于逻辑不连贯而不可能成立的响应。我们称这种响应为\textit{强幻觉}，并证明它们源于LM计算其内部表示的逻辑运算符和从这些表示中产生的输出。重点关注否定，我们提供了一种新颖的解决方案，其中否定不是作为潜在表示的另一个元素，而是作为\textit{LM潜在表示上的一个操作，约束它们可能的演变方式}。我们展示了我们的方法改善了在带否定的填空提示和自然语言推理任务中的模型性能，而无需对稀疏负数据进行训练。

    arXiv:2402.10543v1 Announce Type: cross  Abstract: Despite great performance on many tasks, language models (LMs) still struggle with reasoning, sometimes providing responses that cannot possibly be true because they stem from logical incoherence. We call such responses \textit{strong hallucinations} and prove that they follow from an LM's computation of its internal representations for logical operators and outputs from those representations. Focusing on negation, we provide a novel solution in which negation is treated not as another element of a latent representation, but as \textit{an operation over an LM's latent representations that constrains how they may evolve}. We show that our approach improves model performance in cloze prompting and natural language inference tasks with negation without requiring training on sparse negative data.
    
[^6]: PsySafe：基于心理学的多智能体系统安全攻击、防御和评估的综合框架

    PsySafe: A Comprehensive Framework for Psychological-based Attack, Defense, and Evaluation of Multi-agent System Safety

    [https://arxiv.org/abs/2401.11880](https://arxiv.org/abs/2401.11880)

    PsySafe提出了一个综合框架，通过深入探讨智能体心理学，揭示智能体的黑暗心理状态对安全构成威胁，并提出了有效的风险缓解策略。

    

    多智能体系统在加入大型语言模型（LLMs）后，展现出了集体智能的深远能力。然而，这种智能被恶意使用可能带来重大风险。迄今为止，关于多智能体系统安全问题的全面研究仍然有限。本文通过创新的视角探索了这些问题，发现智能体的黑暗心理状态构成了对安全的重大威胁。为了解决这些问题，我们提出了一个以智能体心理学为基础的综合框架（PsySafe），关注三个关键领域：首先，识别智能体中的黑暗人格特征如何导致风险行为；其次，从心理和行为角度评估多智能体系统的安全性；第三，制定有效的策略来减轻这些风险。我们的实验揭示

    arXiv:2401.11880v2 Announce Type: replace-cross  Abstract: Multi-agent systems, when enhanced with Large Language Models (LLMs), exhibit profound capabilities in collective intelligence. However, the potential misuse of this intelligence for malicious purposes presents significant risks. To date, comprehensive research on the safety issues associated with multi-agent systems remains limited. In this paper, we explore these concerns through the innovative lens of agent psychology, revealing that the dark psychological states of agents constitute a significant threat to safety. To tackle these concerns, we propose a comprehensive framework (PsySafe) grounded in agent psychology, focusing on three key areas: firstly, identifying how dark personality traits in agents can lead to risky behaviors; secondly, evaluating the safety of multi-agent systems from the psychological and behavioral perspectives, and thirdly, devising effective strategies to mitigate these risks. Our experiments reveal
    
[^7]: F-Eval:使用优化的评估方法评估基本能力

    F-Eval: Asssessing Fundamental Abilities with Refined Evaluation Methods. (arXiv:2401.14869v1 [cs.CL])

    [http://arxiv.org/abs/2401.14869](http://arxiv.org/abs/2401.14869)

    F-Eval是一个双语评估基准，用于评估大型语言模型的基本能力，包括表达、常识和逻辑。它采用多种任务形式进行评估，包括客观任务和主观任务，并提出了新的评估方法来解决无参考的主观任务评估问题。

    

    大型语言模型（LLMs）因其前所未有的性能而受到广泛关注，导致越来越多的研究评估LLMs。然而，这些评估基准仅限于评估指令遵循能力，忽视了在预训练阶段出现的基本能力。先前的主观评估方法主要依赖于由API模型评分。然而，在没有参考文献的情况下，大模型显示出有限的能力来区分细微差异。为了弥合这一差距，我们提出了F-Eval，一个双语评估基准，用于评估基本能力，包括表达、常识和逻辑。F-Eval中的任务包括多项选择客观任务、开放式客观任务、基于参考的主观任务和无参考的主观任务。对于无参考的主观任务，我们设计了新的评估方法，作为替代API模型评分的方法。我们对13个先进的LLMs进行了评估。

    Large language models (LLMs) garner significant attention for their unprecedented performance, leading to an increasing number of researches evaluating LLMs. However, these evaluation benchmarks are limited to assessing the instruction-following capabilities, overlooking the fundamental abilities that emerge during the pre-training stage. Previous subjective evaluation methods mainly reply on scoring by API models. However, in the absence of references, large models have shown limited ability to discern subtle differences. To bridge the gap, we propose F-Eval, a bilingual evaluation benchmark to evaluate the fundamental abilities, including expression, commonsense and logic. The tasks in F-Eval include multi-choice objective tasks, open-ended objective tasks, reference-based subjective tasks and reference-free subjective tasks. For reference-free subjective tasks, we devise new evaluation methods, serving as alternatives to scoring by API models. We conduct evaluations on 13 advanced L
    
[^8]: CodePrompt：通过Prompt学习的知识特征改进源代码相关分类

    CodePrompt: Improving Source Code-Related Classification with Knowledge Features through Prompt Learning. (arXiv:2401.05544v1 [cs.CL])

    [http://arxiv.org/abs/2401.05544](http://arxiv.org/abs/2401.05544)

    CodePrompt是一种利用Prompt学习和注意机制技术改进源代码相关分类任务的新方法。它能够提取源代码和相关文本中的丰富知识以提高准确性，并且减少了计算成本。

    

    研究人员已经探索利用预训练语言模型（如CodeBERT）改进源代码相关任务的潜力。先前的研究主要依赖CodeBERT的文本嵌入能力和"[CLS]"句子嵌入信息作为下游源代码相关任务的语义表示进行微调。然而，这些方法需要额外的神经网络层来提取有效特征，导致计算成本更高。此外，现有方法没有利用源代码和相关文本中丰富的知识，可能导致准确性降低。本文提出了一种新的方法CodePrompt，通过Prompt学习和注意机制利用预训练模型中的丰富知识来改进源代码相关分类任务。

    Researchers have explored the potential of utilizing pre-trained language models, such as CodeBERT, to improve source code-related tasks. Previous studies have mainly relied on CodeBERT's text embedding capability and the `[CLS]' sentence embedding information as semantic representations for fine-tuning downstream source code-related tasks. However, these methods require additional neural network layers to extract effective features, resulting in higher computational costs. Furthermore, existing approaches have not leveraged the rich knowledge contained in both source code and related text, which can lead to lower accuracy. This paper presents a novel approach, CodePrompt, which utilizes rich knowledge recalled from a pre-trained model by prompt learning and an attention mechanism to improve source code-related classification tasks. Our approach initially motivates the language model with prompt information to retrieve abundant knowledge associated with the input as representative feat
    
[^9]: PromptBench：一个用于评估大型语言模型的统一库

    PromptBench: A Unified Library for Evaluation of Large Language Models. (arXiv:2312.07910v2 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2312.07910](http://arxiv.org/abs/2312.07910)

    PromptBench是一个用于评估大型语言模型的统一库，包括了提示语构建、提示语工程、数据集和模型加载、对抗性提示攻击、动态评估协议和分析工具等组件，旨在促进原创研究和创建新的基准测试、部署下游应用以及设计新的评估协议。

    

    对大型语言模型（LLMs）的评估对于评估其性能和减轻潜在的安全风险至关重要。本文介绍了PromptBench，一个用于评估LLMs的统一库。它由几个关键组件组成，研究人员可以轻松使用和扩展：提示语构建、提示语工程、数据集和模型加载、对抗性提示攻击、动态评估协议和分析工具。PromptBench旨在成为一个开放、通用和灵活的代码库，以促进原创研究，创建新的基准测试、部署下游应用和设计新的评估协议。代码可在https://github.com/microsoft/promptbench上找到，并将持续支持。

    The evaluation of large language models (LLMs) is crucial to assess their performance and mitigate potential security risks. In this paper, we introduce PromptBench, a unified library to evaluate LLMs. It consists of several key components that are easily used and extended by researchers: prompt construction, prompt engineering, dataset and model loading, adversarial prompt attack, dynamic evaluation protocols, and analysis tools. PromptBench is designed to be an open, general, and flexible codebase for research purposes that can facilitate original study in creating new benchmarks, deploying downstream applications, and designing new evaluation protocols. The code is available at: https://github.com/microsoft/promptbench and will be continuously supported.
    
[^10]: 假新闻在绵羊的外衣中：对抗LLM增强风格攻击的鲁棒假新闻检测

    Fake News in Sheep's Clothing: Robust Fake News Detection Against LLM-Empowered Style Attacks. (arXiv:2310.10830v1 [cs.CL])

    [http://arxiv.org/abs/2310.10830](http://arxiv.org/abs/2310.10830)

    这篇论文介绍了一种鲁棒的无风格假新闻检测器，能够对抗利用大型语言模型进行风格攻击的假新闻。通过LLM增强的新闻重构，该检测器能够适应不同的写作风格，提高了对伪装假新闻的检测能力。

    

    人们常常认为在线假新闻和可靠新闻在写作风格上有明显的差异，如使用耸人听闻的语言与客观的语言。然而，我们强调风格相关特征也可以用于风格攻击。值得注意的是，强大的大型语言模型（LLM）的崛起使恶意用户能够以最低成本模仿值得信赖的新闻媒体的风格。我们的分析显示，以LLM伪装的假新闻内容导致先进的基于文本的检测器的性能显著下降（F1分数减少高达38%），给在线生态系统中的自动检测带来了重大挑战。为了解决这个问题，我们引入了SheepDog，一种对新闻写作风格鲁棒的无风格假新闻检测器。SheepDog通过LLM增强的新闻重构实现了这种适应性，通过风格导向的重构提示来定制每篇文章以适应不同的写作风格。通过采用无风格训练，SheepDog可以在不同风格的新闻中检测假新闻。

    It is commonly perceived that online fake news and reliable news exhibit stark differences in writing styles, such as the use of sensationalist versus objective language. However, we emphasize that style-related features can also be exploited for style-based attacks. Notably, the rise of powerful Large Language Models (LLMs) has enabled malicious users to mimic the style of trustworthy news outlets at minimal cost. Our analysis reveals that LLM-camouflaged fake news content leads to substantial performance degradation of state-of-the-art text-based detectors (up to 38% decrease in F1 Score), posing a significant challenge for automated detection in online ecosystems. To address this, we introduce SheepDog, a style-agnostic fake news detector robust to news writing styles. SheepDog achieves this adaptability through LLM-empowered news reframing, which customizes each article to match different writing styles using style-oriented reframing prompts. By employing style-agnostic training, S
    
[^11]: FruitShell法语合成系统在Blizzard 2023挑战赛中的应用

    The FruitShell French synthesis system at the Blizzard 2023 Challenge. (arXiv:2309.00223v1 [eess.AS])

    [http://arxiv.org/abs/2309.00223](http://arxiv.org/abs/2309.00223)

    本文介绍了一个用于Blizzard Challenge 2023的法语文本到语音合成系统，通过对数据的筛选和增强，以及添加词边界和起始/结束符号的方式，提高了语音质量并进行了标准化转录。

    

    本文介绍了一个用于Blizzard Challenge 2023的法语文本到语音合成系统。该挑战包括两个任务：从女性演讲者生成高质量的语音和生成与特定个体相似的语音。关于比赛数据，我们进行了筛选过程，去除了缺失或错误的文本数据。我们对除音素以外的所有符号进行了整理，并消除了没有发音或持续时间为零的符号。此外，我们还在文本中添加了词边界和起始/结束符号，根据我们之前的经验，我们发现这样可以提高语音质量。对于Spoke任务，我们根据比赛规则进行了数据增强。我们使用了一个开源的G2P模型将法语文本转录为音素。由于G2P模型使用国际音标（IPA），我们对提供的比赛数据应用了相同的转录过程，以进行标准化。然而，由于编译器对某些技术限制的识别能力有限，所以我们为了保持竞争的公正，将数据按音标划分为不同的片段进行评估。

    This paper presents a French text-to-speech synthesis system for the Blizzard Challenge 2023. The challenge consists of two tasks: generating high-quality speech from female speakers and generating speech that closely resembles specific individuals. Regarding the competition data, we conducted a screening process to remove missing or erroneous text data. We organized all symbols except for phonemes and eliminated symbols that had no pronunciation or zero duration. Additionally, we added word boundary and start/end symbols to the text, which we have found to improve speech quality based on our previous experience. For the Spoke task, we performed data augmentation according to the competition rules. We used an open-source G2P model to transcribe the French texts into phonemes. As the G2P model uses the International Phonetic Alphabet (IPA), we applied the same transcription process to the provided competition data for standardization. However, due to compiler limitations in recognizing 
    
[^12]: 通过多阶段检索找到已经被澄清的叙述：实现跨语言、跨数据集和零样本学习

    Finding Already Debunked Narratives via Multistage Retrieval: Enabling Cross-Lingual, Cross-Dataset and Zero-Shot Learning. (arXiv:2308.05680v1 [cs.CL])

    [http://arxiv.org/abs/2308.05680](http://arxiv.org/abs/2308.05680)

    本研究通过创建新的数据集、评估多语言预训练Transformer模型以及提出多阶段框架来解决了跨语言澄清检索问题。

    

    检索已经被澄清的叙述的任务旨在检测已经经过事实核查的故事。成功检测到已被澄清的声明不仅减少了专业事实核查人员的手动努力，还可以有助于减缓虚假信息的传播。由于缺乏可用数据，这是一个研究不足的问题，特别是在考虑跨语言任务时，即在检查的在线帖子的语言与事实核查文章的语言不同的情况下进行检索。本文通过以下方式填补了这一空白：（i）创建了一个新颖的数据集，以允许对已被澄清的叙述进行跨语言检索的研究，使用推文作为对事实核查文章数据库的查询；（ii）展示了一个全面的实验，以评估经过微调和现成的多语言预训练Transformer模型在这个任务上的性能；（iii）提出了一个新颖的多阶段框架，将这个跨语言澄清检索问题划分为不同的阶段。

    The task of retrieving already debunked narratives aims to detect stories that have already been fact-checked. The successful detection of claims that have already been debunked not only reduces the manual efforts of professional fact-checkers but can also contribute to slowing the spread of misinformation. Mainly due to the lack of readily available data, this is an understudied problem, particularly when considering the cross-lingual task, i.e. the retrieval of fact-checking articles in a language different from the language of the online post being checked. This paper fills this gap by (i) creating a novel dataset to enable research on cross-lingual retrieval of already debunked narratives, using tweets as queries to a database of fact-checking articles; (ii) presenting an extensive experiment to benchmark fine-tuned and off-the-shelf multilingual pre-trained Transformer models for this task; and (iii) proposing a novel multistage framework that divides this cross-lingual debunk ret
    
[^13]: MMBench: 您的多模态模型是全能球员吗？

    MMBench: Is Your Multi-modal Model an All-around Player?. (arXiv:2307.06281v1 [cs.CV])

    [http://arxiv.org/abs/2307.06281](http://arxiv.org/abs/2307.06281)

    MMBench是一个新型的多模态基准测试，旨在解决大型视觉语言模型评估的挑战，通过开发全面的评估流程和精心策划的数据集进行细粒度能力评估。

    

    最近，大型视觉语言模型在视觉信息的感知和推理能力方面取得了显著进展。然而，如何有效评估这些大型视觉语言模型仍然是一个主要障碍，阻碍了未来模型的发展。传统的基准测试，如VQAv2或COCO Caption提供了定量的性能测量，但在细粒度能力评估和非鲁棒评估指标方面存在不足。最近的主观基准测试，如OwlEval，通过整合人力资源，对模型的能力进行了全面评估，但不可扩展并且存在显著的偏见。针对这些挑战，我们提出了MMBench，一种新型的多模态基准测试。MMBench系统地开发了一个全面的评估流程，主要由两个元素组成。第一个元素是精心策划的数据集，在评估数量和多样性方面超越了现有的类似基准测试。

    Large vision-language models have recently achieved remarkable progress, exhibiting great perception and reasoning abilities concerning visual information. However, how to effectively evaluate these large vision-language models remains a major obstacle, hindering future model development. Traditional benchmarks like VQAv2 or COCO Caption provide quantitative performance measurements but suffer from a lack of fine-grained ability assessment and non-robust evaluation metrics. Recent subjective benchmarks, such as OwlEval, offer comprehensive evaluations of a model's abilities by incorporating human labor, but they are not scalable and display significant bias. In response to these challenges, we propose MMBench, a novel multi-modality benchmark. MMBench methodically develops a comprehensive evaluation pipeline, primarily comprised of two elements. The first element is a meticulously curated dataset that surpasses existing similar benchmarks in terms of the number and variety of evaluatio
    
[^14]: 因果推理与大型语言模型：开启因果研究的新篇章

    Causal Reasoning and Large Language Models: Opening a New Frontier for Causality. (arXiv:2305.00050v1 [cs.AI])

    [http://arxiv.org/abs/2305.00050](http://arxiv.org/abs/2305.00050)

    大型语言模型在因果推理任务中取得了新的最高准确率，但是其鲁棒性仍然存在难以预测的失败模式。

    

    大型语言模型的因果能力备受争议，并且对将其应用于医学、科学、法律和政策等具有社会影响力的领域具有重要意义。我们进一步探讨了LLMs及其因果推理的区别，以及潜在的建构和测量效度威胁。基于GPT-3.5和4的算法在多个因果基准测试上取得了新的最高准确率。与此同时，LLMs展示了难以预测的失败模式，我们提供了一些技术来解释它们的鲁棒性。

    The causal capabilities of large language models (LLMs) is a matter of significant debate, with critical implications for the use of LLMs in societally impactful domains such as medicine, science, law, and policy. We further our understanding of LLMs and their causal implications, considering the distinctions between different types of causal reasoning tasks, as well as the entangled threats of construct and measurement validity. LLM-based methods establish new state-of-the-art accuracies on multiple causal benchmarks. Algorithms based on GPT-3.5 and 4 outperform existing algorithms on a pairwise causal discovery task (97%, 13 points gain), counterfactual reasoning task (92%, 20 points gain), and actual causality (86% accuracy in determining necessary and sufficient causes in vignettes). At the same time, LLMs exhibit unpredictable failure modes and we provide some techniques to interpret their robustness.  Crucially, LLMs perform these causal tasks while relying on sources of knowledg
    

