# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [CoBOS: Constraint-Based Online Scheduler for Human-Robot Collaboration](https://arxiv.org/abs/2403.18459) | CoBOS提出了一种新颖的在线基于约束的调度方法，在人机协作中实现了机器人对不确定事件的适应性，大大减轻了用户的压力，提高了工作效率。 |
| [^2] | [Don't Half-listen: Capturing Key-part Information in Continual Instruction Tuning](https://arxiv.org/abs/2403.10056) | 提出了一种基于关键部分信息增益的新型连续指导调整方法，通过动态重放数据和优化训练目标，使LLMs能够捕捉任务感知信息和减轻过度拟合。 |
| [^3] | [Bias-Augmented Consistency Training Reduces Biased Reasoning in Chain-of-Thought](https://arxiv.org/abs/2403.05518) | 引入偏差增强的一致性训练（BCT）可以显著减少链式思维中的偏见推理问题，尤其是通过训练模型在带有和不带有偏置特征的提示下进行一致的推理。 |
| [^4] | [An In-depth Evaluation of GPT-4 in Sentence Simplification with Error-based Human Assessment](https://arxiv.org/abs/2403.04963) | 本文深入评估了GPT-4在句子简化中的表现，指出现有自动评估指标和人类评估方法对于大型语言模型的适用性仍有待进一步研究。 |
| [^5] | [Tradeoffs Between Alignment and Helpfulness in Language Models](https://arxiv.org/abs/2401.16332) | 本文研究了在语言模型中增加对齐度和减少有用性之间的权衡。我们提出了一个理论框架来提供这两个数量的边界，并通过实验证明了它们的相关性。 |
| [^6] | [Quantum Acceleration of Infinite Horizon Average-Reward Reinforcement Learning.](http://arxiv.org/abs/2310.11684) | 本研究探索了无限时域平均奖励强化学习中量子加速的潜力。我们提出了一种创新的量子框架，通过高效的量子均值估计技术，实现了指数级改进的遗憾保证。所提出的量子算法相较于经典算法，在遗憾界限上有显著改进。 |
| [^7] | [CLEVRER-Humans: Describing Physical and Causal Events the Human Way.](http://arxiv.org/abs/2310.03635) | CLEVRER-Humans是一个用于因果判断的视频推理数据集，通过人工标注来解决合成事件和合成语言描述的缺乏多样性问题，并通过迭代事件填空和神经语言生成模型提高数据收集效率。 |
| [^8] | [Multiple Different Explanations for Image Classifiers.](http://arxiv.org/abs/2309.14309) | 这篇论文介绍了一种算法和工具，可以为图像分类器的输出计算多个解释，从而提高对分类器行为的洞察力。 |
| [^9] | [WizardLM: Empowering Large Language Models to Follow Complex Instructions.](http://arxiv.org/abs/2304.12244) | 本文使用 Evol-Instruct 方法创建了大量不同复杂度的指令数据用于微调 LLaMA 模型，得到了新模型 WizardLM。人类评估结果表明 Evol-Instruct 生成的指令优于人工创建的，而 WizardLM 输出的结果也比 OpenAI ChatGPT 更受欢迎。 |

# 详细

[^1]: CoBOS: 基于约束的人机协作在线调度器

    CoBOS: Constraint-Based Online Scheduler for Human-Robot Collaboration

    [https://arxiv.org/abs/2403.18459](https://arxiv.org/abs/2403.18459)

    CoBOS提出了一种新颖的在线基于约束的调度方法，在人机协作中实现了机器人对不确定事件的适应性，大大减轻了用户的压力，提高了工作效率。

    

    涉及人类和机器人的装配过程是具有挑战性的场景，因为个人活动和共享工作空间的访问必须协调。固定的机器人程序不允许偏离固定协议。在这样的过程中工作可能会让用户感到有压力，并导致行为无效或失败。我们提出了一种新颖的在线基于约束的调度方法，位于支持行为树的反应式执行控制框架中，名为CoBOS。这使得机器人能够适应延迟活动完成和活动选择（由人类）等不确定事件。用户将体验到较少的压力，因为机器人同事会调整其行为以最好地补充人类选择的活动，以完成共同任务。除了改善的工作条件，我们的算法还导致了效率的提高，即使在高度不确定的情况下也是如此。我们使用一个概率性的si来评估我们的算法

    arXiv:2403.18459v1 Announce Type: cross  Abstract: Assembly processes involving humans and robots are challenging scenarios because the individual activities and access to shared workspace have to be coordinated. Fixed robot programs leave no room to diverge from a fixed protocol. Working on such a process can be stressful for the user and lead to ineffective behavior or failure. We propose a novel approach of online constraint-based scheduling in a reactive execution control framework facilitating behavior trees called CoBOS. This allows the robot to adapt to uncertain events such as delayed activity completions and activity selection (by the human). The user will experience less stress as the robotic coworkers adapt their behavior to best complement the human-selected activities to complete the common task. In addition to the improved working conditions, our algorithm leads to increased efficiency, even in highly uncertain scenarios. We evaluate our algorithm using a probabilistic si
    
[^2]: 不要半心半意：捕捉连续指导调整中的关键部分信息

    Don't Half-listen: Capturing Key-part Information in Continual Instruction Tuning

    [https://arxiv.org/abs/2403.10056](https://arxiv.org/abs/2403.10056)

    提出了一种基于关键部分信息增益的新型连续指导调整方法，通过动态重放数据和优化训练目标，使LLMs能够捕捉任务感知信息和减轻过度拟合。

    

    arXiv:2403.10056v1 公告类型: 跨领域 摘要：大型语言模型（LLMs）的指导调整可以驱使它们在特定下游任务中产生符合人类目标的结果。然而，LLMs的连续指导调整（CIT）过程可能会带来灾难性遗忘（CF）问题，导致先前学到的能力退化。最近的方法尝试通过修改模型或重放数据来缓解CF问题，但这可能只记住指令的表面模式并在留存任务上感到困惑。在本文中，我们提出了一种基于关键部分信息增益（KPIG）的新型连续指导调整方法。我们的方法计算掩盖部分的信息增益，动态重放数据并优化训练目标，从而使LLMs能够捕捉与正确响应相关的任务感知信息，并减轻对指导中通用描述的过度拟合。此外，我们提出了两个指标，P分和V分，

    arXiv:2403.10056v1 Announce Type: cross  Abstract: Instruction tuning for large language models (LLMs) can drive them to produce results consistent with human goals in specific downstream tasks. However, the process of continual instruction tuning (CIT) for LLMs may bring about the catastrophic forgetting (CF) problem, where previously learned abilities are degraded. Recent methods try to alleviate the CF problem by modifying models or replaying data, which may only remember the surface-level pattern of instructions and get confused on held-out tasks. In this paper, we propose a novel continual instruction tuning method based on Key-part Information Gain (KPIG). Our method computes the information gain on masked parts to dynamically replay data and refine the training objective, which enables LLMs to capture task-aware information relevant to the correct response and alleviate overfitting to general descriptions in instructions. In addition, we propose two metrics, P-score and V-score,
    
[^3]: 通过偏差增强一致性训练减少链式思维中的偏见推理

    Bias-Augmented Consistency Training Reduces Biased Reasoning in Chain-of-Thought

    [https://arxiv.org/abs/2403.05518](https://arxiv.org/abs/2403.05518)

    引入偏差增强的一致性训练（BCT）可以显著减少链式思维中的偏见推理问题，尤其是通过训练模型在带有和不带有偏置特征的提示下进行一致的推理。

    

    虽然链式思维提示（CoT）有潜力改善语言模型推理的可解释性，但它可能会系统性地歪曲影响模型行为的因素--比如，合理化答案以符合用户意见而不提及此偏见。为了减轻这一偏见推理问题，我们引入了偏差增强的一致性训练（BCT），这是一种无监督的微调方案，旨在训练模型在带有和不带有偏置特征的提示下进行一致的推理。我们构建了一个测试单元，针对七个问答任务测试了九种形式的有偏推理，发现将BCT应用于带有一种偏见的GPT-3.5-Turbo可以将有偏推理的比例在未知任务上降低86%。此外，这个模型推广到其他形式的偏见，平均将未知偏见上的有偏推理减少了37%。由于BCT将未知偏见泛化并且不需要金标签，这种方法可能会有助于

    arXiv:2403.05518v1 Announce Type: cross  Abstract: While chain-of-thought prompting (CoT) has the potential to improve the explainability of language model reasoning, it can systematically misrepresent the factors influencing models' behavior--for example, rationalizing answers in line with a user's opinion without mentioning this bias. To mitigate this biased reasoning problem, we introduce bias-augmented consistency training (BCT), an unsupervised fine-tuning scheme that trains models to give consistent reasoning across prompts with and without biasing features. We construct a suite testing nine forms of biased reasoning on seven question-answering tasks, and find that applying BCT to GPT-3.5-Turbo with one bias reduces the rate of biased reasoning by 86% on held-out tasks. Moreover, this model generalizes to other forms of bias, reducing biased reasoning on held-out biases by an average of 37%. As BCT generalizes to held-out biases and does not require gold labels, this method may h
    
[^4]: 在基于错误的人类评估中深入评估GPT-4在句子简化中的表现

    An In-depth Evaluation of GPT-4 in Sentence Simplification with Error-based Human Assessment

    [https://arxiv.org/abs/2403.04963](https://arxiv.org/abs/2403.04963)

    本文深入评估了GPT-4在句子简化中的表现，指出现有自动评估指标和人类评估方法对于大型语言模型的适用性仍有待进一步研究。

    

    句子简化是一种重写句子以便更易阅读和理解的方法，对于帮助有各种阅读难题的人来说是一种有前途的技术。随着先进大型语言模型（LLMs）的兴起，评估它们在句子简化中的表现变得迫在眉睫。最近的研究利用自动评估指标和人类评估来评估LLMs的简化能力。然而，现有评估方法对LLMs在简化评估中的适用性仍然存在疑问。首先，现有自动指标在LLMs的简化评估中的适用性仍不确定。其次，当前在句子简化中的人类评估方法通常陷入两个极端：要么过于肤浅，无法清晰理解模型的表现，要么过于详细，使注释过程复杂且容易出现不一致性，从而影响评估的可靠性。

    arXiv:2403.04963v1 Announce Type: cross  Abstract: Sentence simplification, which rewrites a sentence to be easier to read and understand, is a promising technique to help people with various reading difficulties. With the rise of advanced large language models (LLMs), evaluating their performance in sentence simplification has become imperative. Recent studies have used both automatic metrics and human evaluations to assess the simplification abilities of LLMs. However, the suitability of existing evaluation methodologies for LLMs remains in question. First, the suitability of current automatic metrics on LLMs' simplification evaluation is still uncertain. Second, current human evaluation approaches in sentence simplification often fall into two extremes: they are either too superficial, failing to offer a clear understanding of the models' performance, or overly detailed, making the annotation process complex and prone to inconsistency, which in turn affects the evaluation's reliabil
    
[^5]: 对齐和有用性之间的权衡：语言模型的研究

    Tradeoffs Between Alignment and Helpfulness in Language Models

    [https://arxiv.org/abs/2401.16332](https://arxiv.org/abs/2401.16332)

    本文研究了在语言模型中增加对齐度和减少有用性之间的权衡。我们提出了一个理论框架来提供这两个数量的边界，并通过实验证明了它们的相关性。

    

    语言模型对齐已成为人工智能安全的重要组成部分，通过增强期望行为和抑制非期望行为，实现人类与语言模型之间的安全交互。通常通过调整模型或插入预设的对齐提示来实现。最近，通过改变训练后的表示来改变模型行为的表示工程方法在对齐语言模型方面表现出了有效性。表示工程在面对对抗攻击和降低社会偏见等对齐导向任务方面取得了增益，但也导致了模型执行基本任务能力的降低。本文研究了增加对齐度和减少模型有用性之间的权衡。我们提出了一个理论框架来提供这两个数量的边界，并通过实验证明了它们的相关性。有趣的是，我们发现，尽管模型的有用性通常会减少

    Language model alignment has become an important component of AI safety, allowing safe interactions between humans and language models, by enhancing desired behaviors and inhibiting undesired ones. It is often done by tuning the model or inserting preset aligning prompts. Recently, representation engineering, a method which alters the model's behavior via changing its representations post-training, was shown to be effective in aligning LLMs (Zou et al., 2023a). Representation engineering yields gains in alignment oriented tasks such as resistance to adversarial attacks and reduction of social biases, but was also shown to cause a decrease in the ability of the model to perform basic tasks. In this paper we study the tradeoff between the increase in alignment and decrease in helpfulness of the model. We propose a theoretical framework which provides bounds for these two quantities, and demonstrate their relevance empirically. Interestingly, we find that while the helpfulness generally d
    
[^6]: 无限时域平均奖励强化学习的量子加速

    Quantum Acceleration of Infinite Horizon Average-Reward Reinforcement Learning. (arXiv:2310.11684v1 [cs.LG])

    [http://arxiv.org/abs/2310.11684](http://arxiv.org/abs/2310.11684)

    本研究探索了无限时域平均奖励强化学习中量子加速的潜力。我们提出了一种创新的量子框架，通过高效的量子均值估计技术，实现了指数级改进的遗憾保证。所提出的量子算法相较于经典算法，在遗憾界限上有显著改进。

    

    本文研究量子加速在解决无限时域Markov决策过程（MDPs）中提高平均奖励结果的潜力。我们引入了一种创新的量子框架，用于代理与未知MDP的互动，扩展了传统的交互范式。我们的方法涉及设计一种基于乐观主导的具有量子信号的表格强化学习算法，通过高效的量子均值估计技术获取代理获取的量子信号。通过深入的理论分析，我们证明了量子均值估计的优势能够在无限时域强化学习中导致遗憾保证的指数进展。具体地，所提出的量子算法实现了一个遗憾界为$\tilde{\mathcal{O}}(1)$的性能，这是相对于经典对应算法所展示的$\tilde{\mathcal{O}}(\sqrt{T})$界限的显著改进。

    This paper investigates the potential of quantum acceleration in addressing infinite horizon Markov Decision Processes (MDPs) to enhance average reward outcomes. We introduce an innovative quantum framework for the agent's engagement with an unknown MDP, extending the conventional interaction paradigm. Our approach involves the design of an optimism-driven tabular Reinforcement Learning algorithm that harnesses quantum signals acquired by the agent through efficient quantum mean estimation techniques. Through thorough theoretical analysis, we demonstrate that the quantum advantage in mean estimation leads to exponential advancements in regret guarantees for infinite horizon Reinforcement Learning. Specifically, the proposed Quantum algorithm achieves a regret bound of $\tilde{\mathcal{O}}(1)$, a significant improvement over the $\tilde{\mathcal{O}}(\sqrt{T})$ bound exhibited by classical counterparts.
    
[^7]: CLEVRER-Humans: 用人类的方式描述物理和因果事件

    CLEVRER-Humans: Describing Physical and Causal Events the Human Way. (arXiv:2310.03635v1 [cs.AI])

    [http://arxiv.org/abs/2310.03635](http://arxiv.org/abs/2310.03635)

    CLEVRER-Humans是一个用于因果判断的视频推理数据集，通过人工标注来解决合成事件和合成语言描述的缺乏多样性问题，并通过迭代事件填空和神经语言生成模型提高数据收集效率。

    

    构建能够推理物理事件及其因果关系的机器对于与物理世界进行灵活互动非常重要。然而，现有的大多数物理和因果推理基准都仅基于合成事件和合成自然语言描述的因果关系。这种设计存在两个问题：一是事件类型和自然语言描述缺乏多样性；二是基于手动定义的启发式规则的因果关系与人类判断不一致。为了解决这两个问题，我们提出了CLEVRER-Humans基准，这是一个用人工标注的视频推理数据集，用于对物理事件的因果判断。我们采用了两种技术来提高数据收集效率：首先，一种新颖的迭代事件填空任务，以 eliciting 视频中事件的新表示方式，我们称之为因果事件图 (CEGs)；其次，一种基于神经语言生成模型的数据增强技术。

    Building machines that can reason about physical events and their causal relationships is crucial for flexible interaction with the physical world. However, most existing physical and causal reasoning benchmarks are exclusively based on synthetically generated events and synthetic natural language descriptions of causal relationships. This design brings up two issues. First, there is a lack of diversity in both event types and natural language descriptions; second, causal relationships based on manually-defined heuristics are different from human judgments. To address both shortcomings, we present the CLEVRER-Humans benchmark, a video reasoning dataset for causal judgment of physical events with human labels. We employ two techniques to improve data collection efficiency: first, a novel iterative event cloze task to elicit a new representation of events in videos, which we term Causal Event Graphs (CEGs); second, a data augmentation technique based on neural language generative models.
    
[^8]: 图像分类器的多个不同解释

    Multiple Different Explanations for Image Classifiers. (arXiv:2309.14309v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2309.14309](http://arxiv.org/abs/2309.14309)

    这篇论文介绍了一种算法和工具，可以为图像分类器的输出计算多个解释，从而提高对分类器行为的洞察力。

    

    现有的图像分类器解释工具通常只会给出一种对于图像的解释。然而，对于许多图像来说，无论是人类还是图像分类器都接受多个解释来解释图像标签。因此，限制解释的数量只有一个严重限制了对分类器行为的洞察力。在本文中，我们描述了一种算法和工具REX，用于计算黑盒图像分类器对给定图像的输出的多个解释。我们的算法基于因果理论的可靠方法。我们分析了其理论复杂性，并提供了实验结果，显示REX在ImageNet-mini基准测试中找到的多个解释比之前的工作多7倍。

    Existing explanation tools for image classifiers usually give only one single explanation for an image. For many images, however, both humans and image classifiers accept more than one explanation for the image label. Thus, restricting the number of explanations to just one severely limits the insight into the behavior of the classifier. In this paper, we describe an algorithm and a tool, REX, for computing multiple explanations of the output of a black-box image classifier for a given image. Our algorithm uses a principled approach based on causal theory. We analyse its theoretical complexity and provide experimental results showing that REX finds multiple explanations on 7 times more images than the previous work on the ImageNet-mini benchmark.
    
[^9]: WizardLM: 增强大型语言模型遵循复杂指令的能力

    WizardLM: Empowering Large Language Models to Follow Complex Instructions. (arXiv:2304.12244v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2304.12244](http://arxiv.org/abs/2304.12244)

    本文使用 Evol-Instruct 方法创建了大量不同复杂度的指令数据用于微调 LLaMA 模型，得到了新模型 WizardLM。人类评估结果表明 Evol-Instruct 生成的指令优于人工创建的，而 WizardLM 输出的结果也比 OpenAI ChatGPT 更受欢迎。

    

    使用开放域指令追踪数据对大型语言模型进行训练带来了巨大的成功。然而，手动创建这样的指令数据非常耗时和劳动密集，且人类可能难以生成高复杂度指令。在本文中，我们展示了使用LLM而不是人类创建大量不同复杂度指令数据的途径。我们从一组初始指令开始，使用我们提出的Evol-Instruct逐步将其重新编写为更复杂的指令。然后，将所有生成的指令数据混合以微调LLaMA。我们称结果模型为WizardLM。针对一个复杂度平衡的测试集和Vicuna的测试集进行的人类评估表明，Evol-Instruct生成的指令优于人工创建的指令。通过分析高复杂性部分的人类评估结果，我们证明了从我们的WizardLM生成的输出比从OpenAI ChatGPT生成的输出更受欢迎。在GPT-4自动评估中，WizardLM产生了最好的结果。

    Training large language models (LLMs) with open-domain instruction following data brings colossal success. However, manually creating such instruction data is very time-consuming and labor-intensive. Moreover, humans may struggle to produce high-complexity instructions. In this paper, we show an avenue for creating large amounts of instruction data with varying levels of complexity using LLM instead of humans. Starting with an initial set of instructions, we use our proposed Evol-Instruct to rewrite them step by step into more complex instructions. Then, we mix all generated instruction data to fine-tune LLaMA. We call the resulting model WizardLM. Human evaluations on a complexity-balanced test bed and Vicuna's testset show that instructions from Evol-Instruct are superior to human-created ones. By analyzing the human evaluation results of the high complexity part, we demonstrate that outputs from our WizardLM are preferred to outputs from OpenAI ChatGPT. In GPT-4 automatic evaluation
    

