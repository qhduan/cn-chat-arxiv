# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Don't Half-listen: Capturing Key-part Information in Continual Instruction Tuning](https://arxiv.org/abs/2403.10056) | 提出了一种基于关键部分信息增益的新型连续指导调整方法，通过动态重放数据和优化训练目标，使LLMs能够捕捉任务感知信息和减轻过度拟合。 |
| [^2] | [Bias-Augmented Consistency Training Reduces Biased Reasoning in Chain-of-Thought](https://arxiv.org/abs/2403.05518) | 引入偏差增强的一致性训练（BCT）可以显著减少链式思维中的偏见推理问题，尤其是通过训练模型在带有和不带有偏置特征的提示下进行一致的推理。 |
| [^3] | [An In-depth Evaluation of GPT-4 in Sentence Simplification with Error-based Human Assessment](https://arxiv.org/abs/2403.04963) | 本文深入评估了GPT-4在句子简化中的表现，指出现有自动评估指标和人类评估方法对于大型语言模型的适用性仍有待进一步研究。 |
| [^4] | [LLMs with Industrial Lens: Deciphering the Challenges and Prospects -- A Survey](https://arxiv.org/abs/2402.14558) | 本文调查了在工业背景下利用LLMs所面临的挑战和前景，并通过对行业从业者调查以及审查大量行业论文得出有意义的结论。 |
| [^5] | [Retrieve to Explain: Evidence-driven Predictions with Language Models](https://arxiv.org/abs/2402.04068) | 检索以解释（R2E）是一种基于语言模型的检索方法，通过使用Shapley值确定证据的相对重要性，从而在黑盒模型中提供了可解释性，通过应用于药物靶点鉴定任务中，R2E模型在预测临床试验结果方面优于传统基因学方法。 |
| [^6] | [Tradeoffs Between Alignment and Helpfulness in Language Models](https://arxiv.org/abs/2401.16332) | 本文研究了在语言模型中增加对齐度和减少有用性之间的权衡。我们提出了一个理论框架来提供这两个数量的边界，并通过实验证明了它们的相关性。 |
| [^7] | [CLEVRER-Humans: Describing Physical and Causal Events the Human Way.](http://arxiv.org/abs/2310.03635) | CLEVRER-Humans是一个用于因果判断的视频推理数据集，通过人工标注来解决合成事件和合成语言描述的缺乏多样性问题，并通过迭代事件填空和神经语言生成模型提高数据收集效率。 |
| [^8] | [WizardLM: Empowering Large Language Models to Follow Complex Instructions.](http://arxiv.org/abs/2304.12244) | 本文使用 Evol-Instruct 方法创建了大量不同复杂度的指令数据用于微调 LLaMA 模型，得到了新模型 WizardLM。人类评估结果表明 Evol-Instruct 生成的指令优于人工创建的，而 WizardLM 输出的结果也比 OpenAI ChatGPT 更受欢迎。 |

# 详细

[^1]: 不要半心半意：捕捉连续指导调整中的关键部分信息

    Don't Half-listen: Capturing Key-part Information in Continual Instruction Tuning

    [https://arxiv.org/abs/2403.10056](https://arxiv.org/abs/2403.10056)

    提出了一种基于关键部分信息增益的新型连续指导调整方法，通过动态重放数据和优化训练目标，使LLMs能够捕捉任务感知信息和减轻过度拟合。

    

    arXiv:2403.10056v1 公告类型: 跨领域 摘要：大型语言模型（LLMs）的指导调整可以驱使它们在特定下游任务中产生符合人类目标的结果。然而，LLMs的连续指导调整（CIT）过程可能会带来灾难性遗忘（CF）问题，导致先前学到的能力退化。最近的方法尝试通过修改模型或重放数据来缓解CF问题，但这可能只记住指令的表面模式并在留存任务上感到困惑。在本文中，我们提出了一种基于关键部分信息增益（KPIG）的新型连续指导调整方法。我们的方法计算掩盖部分的信息增益，动态重放数据并优化训练目标，从而使LLMs能够捕捉与正确响应相关的任务感知信息，并减轻对指导中通用描述的过度拟合。此外，我们提出了两个指标，P分和V分，

    arXiv:2403.10056v1 Announce Type: cross  Abstract: Instruction tuning for large language models (LLMs) can drive them to produce results consistent with human goals in specific downstream tasks. However, the process of continual instruction tuning (CIT) for LLMs may bring about the catastrophic forgetting (CF) problem, where previously learned abilities are degraded. Recent methods try to alleviate the CF problem by modifying models or replaying data, which may only remember the surface-level pattern of instructions and get confused on held-out tasks. In this paper, we propose a novel continual instruction tuning method based on Key-part Information Gain (KPIG). Our method computes the information gain on masked parts to dynamically replay data and refine the training objective, which enables LLMs to capture task-aware information relevant to the correct response and alleviate overfitting to general descriptions in instructions. In addition, we propose two metrics, P-score and V-score,
    
[^2]: 通过偏差增强一致性训练减少链式思维中的偏见推理

    Bias-Augmented Consistency Training Reduces Biased Reasoning in Chain-of-Thought

    [https://arxiv.org/abs/2403.05518](https://arxiv.org/abs/2403.05518)

    引入偏差增强的一致性训练（BCT）可以显著减少链式思维中的偏见推理问题，尤其是通过训练模型在带有和不带有偏置特征的提示下进行一致的推理。

    

    虽然链式思维提示（CoT）有潜力改善语言模型推理的可解释性，但它可能会系统性地歪曲影响模型行为的因素--比如，合理化答案以符合用户意见而不提及此偏见。为了减轻这一偏见推理问题，我们引入了偏差增强的一致性训练（BCT），这是一种无监督的微调方案，旨在训练模型在带有和不带有偏置特征的提示下进行一致的推理。我们构建了一个测试单元，针对七个问答任务测试了九种形式的有偏推理，发现将BCT应用于带有一种偏见的GPT-3.5-Turbo可以将有偏推理的比例在未知任务上降低86%。此外，这个模型推广到其他形式的偏见，平均将未知偏见上的有偏推理减少了37%。由于BCT将未知偏见泛化并且不需要金标签，这种方法可能会有助于

    arXiv:2403.05518v1 Announce Type: cross  Abstract: While chain-of-thought prompting (CoT) has the potential to improve the explainability of language model reasoning, it can systematically misrepresent the factors influencing models' behavior--for example, rationalizing answers in line with a user's opinion without mentioning this bias. To mitigate this biased reasoning problem, we introduce bias-augmented consistency training (BCT), an unsupervised fine-tuning scheme that trains models to give consistent reasoning across prompts with and without biasing features. We construct a suite testing nine forms of biased reasoning on seven question-answering tasks, and find that applying BCT to GPT-3.5-Turbo with one bias reduces the rate of biased reasoning by 86% on held-out tasks. Moreover, this model generalizes to other forms of bias, reducing biased reasoning on held-out biases by an average of 37%. As BCT generalizes to held-out biases and does not require gold labels, this method may h
    
[^3]: 在基于错误的人类评估中深入评估GPT-4在句子简化中的表现

    An In-depth Evaluation of GPT-4 in Sentence Simplification with Error-based Human Assessment

    [https://arxiv.org/abs/2403.04963](https://arxiv.org/abs/2403.04963)

    本文深入评估了GPT-4在句子简化中的表现，指出现有自动评估指标和人类评估方法对于大型语言模型的适用性仍有待进一步研究。

    

    句子简化是一种重写句子以便更易阅读和理解的方法，对于帮助有各种阅读难题的人来说是一种有前途的技术。随着先进大型语言模型（LLMs）的兴起，评估它们在句子简化中的表现变得迫在眉睫。最近的研究利用自动评估指标和人类评估来评估LLMs的简化能力。然而，现有评估方法对LLMs在简化评估中的适用性仍然存在疑问。首先，现有自动指标在LLMs的简化评估中的适用性仍不确定。其次，当前在句子简化中的人类评估方法通常陷入两个极端：要么过于肤浅，无法清晰理解模型的表现，要么过于详细，使注释过程复杂且容易出现不一致性，从而影响评估的可靠性。

    arXiv:2403.04963v1 Announce Type: cross  Abstract: Sentence simplification, which rewrites a sentence to be easier to read and understand, is a promising technique to help people with various reading difficulties. With the rise of advanced large language models (LLMs), evaluating their performance in sentence simplification has become imperative. Recent studies have used both automatic metrics and human evaluations to assess the simplification abilities of LLMs. However, the suitability of existing evaluation methodologies for LLMs remains in question. First, the suitability of current automatic metrics on LLMs' simplification evaluation is still uncertain. Second, current human evaluation approaches in sentence simplification often fall into two extremes: they are either too superficial, failing to offer a clear understanding of the models' performance, or overly detailed, making the annotation process complex and prone to inconsistency, which in turn affects the evaluation's reliabil
    
[^4]: 具有工业视角的LLMs：揭示挑战与前景--一项调查

    LLMs with Industrial Lens: Deciphering the Challenges and Prospects -- A Survey

    [https://arxiv.org/abs/2402.14558](https://arxiv.org/abs/2402.14558)

    本文调查了在工业背景下利用LLMs所面临的挑战和前景，并通过对行业从业者调查以及审查大量行业论文得出有意义的结论。

    

    大语言模型（LLMs）已经成为推动许多工业应用的秘密武器，展示出它们在各种任务中的卓越适应性。从自然语言处理和情感分析到内容生成和个性化推荐，它们无与伦比的适应性促进了在各个行业的广泛应用。LLMs带来的这种转变强调了探索与利用中的困难和增强机会的必要性。本文的目标是揭示和评估在工业背景下利用LLMs所面临的障碍和机会。为此，我们进行了一项涉及一组行业从业者的调查，提出了四个研究问题，并根据收集到的见解审查了68篇行业论文，以解决这些问题并得出有意义的结论。

    arXiv:2402.14558v1 Announce Type: new  Abstract: Large language models (LLMs) have become the secret ingredient driving numerous industrial applications, showcasing their remarkable versatility across a diverse spectrum of tasks. From natural language processing and sentiment analysis to content generation and personalized recommendations, their unparalleled adaptability has facilitated widespread adoption across industries. This transformative shift driven by LLMs underscores the need to explore the underlying associated challenges and avenues for enhancement in their utilization. In this paper, our objective is to unravel and evaluate the obstacles and opportunities inherent in leveraging LLMs within an industrial context. To this end, we conduct a survey involving a group of industry practitioners, develop four research questions derived from the insights gathered, and examine 68 industry papers to address these questions and derive meaningful conclusions.
    
[^5]: 检索以解释：基于语言模型的证据驱动预测

    Retrieve to Explain: Evidence-driven Predictions with Language Models

    [https://arxiv.org/abs/2402.04068](https://arxiv.org/abs/2402.04068)

    检索以解释（R2E）是一种基于语言模型的检索方法，通过使用Shapley值确定证据的相对重要性，从而在黑盒模型中提供了可解释性，通过应用于药物靶点鉴定任务中，R2E模型在预测临床试验结果方面优于传统基因学方法。

    

    机器学习模型，尤其是语言模型，往往难以深入分析。黑盒模型可能掩盖了模型训练中的问题和有害偏差。对于人机协作过程来说，不透明的预测可能导致缺乏信任，限制模型的影响，即使模型的性能很好。为了解决这些问题，我们引入了检索以解释（Retrieve to Explain，简称R2E）。R2E是一种基于检索的语言模型，根据文档语料库中的证据，使用Shapley值来确定证据对最终预测的相对重要性，并根据自然语言模板将结构化数据纳入其中。R2E能够在不重新训练的情况下适应新的证据，并且能够通过模板化将结构化数据纳入到自然语言中。我们在通过分析已发表的科学文献进行药物靶点鉴定的实际案例中进行了评估，结果显示该模型在预测临床试验结果方面优于行业标准的基因学方法。

    Machine learning models, particularly language models, are notoriously difficult to introspect. Black-box models can mask both issues in model training and harmful biases. For human-in-the-loop processes, opaque predictions can drive lack of trust, limiting a model's impact even when it performs effectively. To address these issues, we introduce Retrieve to Explain (R2E). R2E is a retrieval-based language model that prioritizes amongst a pre-defined set of possible answers to a research question based on the evidence in a document corpus, using Shapley values to identify the relative importance of pieces of evidence to the final prediction. R2E can adapt to new evidence without retraining, and incorporate structured data through templating into natural language. We assess on the use case of drug target identification from published scientific literature, where we show that the model outperforms an industry-standard genetics-based approach on predicting clinical trial outcomes.
    
[^6]: 对齐和有用性之间的权衡：语言模型的研究

    Tradeoffs Between Alignment and Helpfulness in Language Models

    [https://arxiv.org/abs/2401.16332](https://arxiv.org/abs/2401.16332)

    本文研究了在语言模型中增加对齐度和减少有用性之间的权衡。我们提出了一个理论框架来提供这两个数量的边界，并通过实验证明了它们的相关性。

    

    语言模型对齐已成为人工智能安全的重要组成部分，通过增强期望行为和抑制非期望行为，实现人类与语言模型之间的安全交互。通常通过调整模型或插入预设的对齐提示来实现。最近，通过改变训练后的表示来改变模型行为的表示工程方法在对齐语言模型方面表现出了有效性。表示工程在面对对抗攻击和降低社会偏见等对齐导向任务方面取得了增益，但也导致了模型执行基本任务能力的降低。本文研究了增加对齐度和减少模型有用性之间的权衡。我们提出了一个理论框架来提供这两个数量的边界，并通过实验证明了它们的相关性。有趣的是，我们发现，尽管模型的有用性通常会减少

    Language model alignment has become an important component of AI safety, allowing safe interactions between humans and language models, by enhancing desired behaviors and inhibiting undesired ones. It is often done by tuning the model or inserting preset aligning prompts. Recently, representation engineering, a method which alters the model's behavior via changing its representations post-training, was shown to be effective in aligning LLMs (Zou et al., 2023a). Representation engineering yields gains in alignment oriented tasks such as resistance to adversarial attacks and reduction of social biases, but was also shown to cause a decrease in the ability of the model to perform basic tasks. In this paper we study the tradeoff between the increase in alignment and decrease in helpfulness of the model. We propose a theoretical framework which provides bounds for these two quantities, and demonstrate their relevance empirically. Interestingly, we find that while the helpfulness generally d
    
[^7]: CLEVRER-Humans: 用人类的方式描述物理和因果事件

    CLEVRER-Humans: Describing Physical and Causal Events the Human Way. (arXiv:2310.03635v1 [cs.AI])

    [http://arxiv.org/abs/2310.03635](http://arxiv.org/abs/2310.03635)

    CLEVRER-Humans是一个用于因果判断的视频推理数据集，通过人工标注来解决合成事件和合成语言描述的缺乏多样性问题，并通过迭代事件填空和神经语言生成模型提高数据收集效率。

    

    构建能够推理物理事件及其因果关系的机器对于与物理世界进行灵活互动非常重要。然而，现有的大多数物理和因果推理基准都仅基于合成事件和合成自然语言描述的因果关系。这种设计存在两个问题：一是事件类型和自然语言描述缺乏多样性；二是基于手动定义的启发式规则的因果关系与人类判断不一致。为了解决这两个问题，我们提出了CLEVRER-Humans基准，这是一个用人工标注的视频推理数据集，用于对物理事件的因果判断。我们采用了两种技术来提高数据收集效率：首先，一种新颖的迭代事件填空任务，以 eliciting 视频中事件的新表示方式，我们称之为因果事件图 (CEGs)；其次，一种基于神经语言生成模型的数据增强技术。

    Building machines that can reason about physical events and their causal relationships is crucial for flexible interaction with the physical world. However, most existing physical and causal reasoning benchmarks are exclusively based on synthetically generated events and synthetic natural language descriptions of causal relationships. This design brings up two issues. First, there is a lack of diversity in both event types and natural language descriptions; second, causal relationships based on manually-defined heuristics are different from human judgments. To address both shortcomings, we present the CLEVRER-Humans benchmark, a video reasoning dataset for causal judgment of physical events with human labels. We employ two techniques to improve data collection efficiency: first, a novel iterative event cloze task to elicit a new representation of events in videos, which we term Causal Event Graphs (CEGs); second, a data augmentation technique based on neural language generative models.
    
[^8]: WizardLM: 增强大型语言模型遵循复杂指令的能力

    WizardLM: Empowering Large Language Models to Follow Complex Instructions. (arXiv:2304.12244v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2304.12244](http://arxiv.org/abs/2304.12244)

    本文使用 Evol-Instruct 方法创建了大量不同复杂度的指令数据用于微调 LLaMA 模型，得到了新模型 WizardLM。人类评估结果表明 Evol-Instruct 生成的指令优于人工创建的，而 WizardLM 输出的结果也比 OpenAI ChatGPT 更受欢迎。

    

    使用开放域指令追踪数据对大型语言模型进行训练带来了巨大的成功。然而，手动创建这样的指令数据非常耗时和劳动密集，且人类可能难以生成高复杂度指令。在本文中，我们展示了使用LLM而不是人类创建大量不同复杂度指令数据的途径。我们从一组初始指令开始，使用我们提出的Evol-Instruct逐步将其重新编写为更复杂的指令。然后，将所有生成的指令数据混合以微调LLaMA。我们称结果模型为WizardLM。针对一个复杂度平衡的测试集和Vicuna的测试集进行的人类评估表明，Evol-Instruct生成的指令优于人工创建的指令。通过分析高复杂性部分的人类评估结果，我们证明了从我们的WizardLM生成的输出比从OpenAI ChatGPT生成的输出更受欢迎。在GPT-4自动评估中，WizardLM产生了最好的结果。

    Training large language models (LLMs) with open-domain instruction following data brings colossal success. However, manually creating such instruction data is very time-consuming and labor-intensive. Moreover, humans may struggle to produce high-complexity instructions. In this paper, we show an avenue for creating large amounts of instruction data with varying levels of complexity using LLM instead of humans. Starting with an initial set of instructions, we use our proposed Evol-Instruct to rewrite them step by step into more complex instructions. Then, we mix all generated instruction data to fine-tune LLaMA. We call the resulting model WizardLM. Human evaluations on a complexity-balanced test bed and Vicuna's testset show that instructions from Evol-Instruct are superior to human-created ones. By analyzing the human evaluation results of the high complexity part, we demonstrate that outputs from our WizardLM are preferred to outputs from OpenAI ChatGPT. In GPT-4 automatic evaluation
    

