# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Counting-Stars: A Simple, Efficient, and Reasonable Strategy for Evaluating Long-Context Large Language Models](https://arxiv.org/abs/2403.11802) | 提出了一种名为Counting-Stars的简单、高效、合理策略，用于评估长上下文大型语言模型的能力，并在实验中发现GPT-4 Turbo和Kimi Chat在此任务上取得显著性能。 |
| [^2] | [ProSwitch: Knowledge-Guided Language Model Fine-Tuning to Generate Professional and Non-Professional Styled Text](https://arxiv.org/abs/2403.09131) | ProSwitch通过知识引导的指令微调，在专业和非专业风格之间生成文本，并在专业性评估和质量评估方面表现出优越性。 |
| [^3] | [Detection of Non-recorded Word Senses in English and Swedish](https://arxiv.org/abs/2403.02285) | 该研究致力于在英语和瑞典语中检测未记录的词义，通过使用预训练的词-上下文嵌入器和人工注释，成功提高了检测到具有未记录词义的词语用法数量。 |
| [^4] | [Improving the Validity of Automatically Generated Feedback via Reinforcement Learning](https://arxiv.org/abs/2403.01304) | 本研究通过提出评估数学反馈的评分标准，展示了GPT-4能够有效地使用它来注释人工编写的和LLM生成的反馈，从而改进了自动生成反馈的有效性和可靠性。 |
| [^5] | [LLMs with Chain-of-Thought Are Non-Causal Reasoners](https://arxiv.org/abs/2402.16048) | 本文探讨了大型语言模型在推理过程中思维链条（CoT）的作用，发现LLMs在答案生成过程中与人类推理存在差异，相关因素包括语境学习、有监督微调以及对人类反馈的强化学习。 |
| [^6] | [ProSparse: Introducing and Enhancing Intrinsic Activation Sparsity within Large Language Models](https://arxiv.org/abs/2402.13516) | 本文介绍了一种名为"ProSparse"的有效稀疏化方法，以推动大型语言模型实现更高的激活稀疏性而不降低模型性能 |
| [^7] | [Learn To be Efficient: Build Structured Sparsity in Large Language Models](https://arxiv.org/abs/2402.06126) | 本文通过引入一种新的算法"Learn-To-be-Efficient(LTE)"，提出了在大型语言模型(LLM)中构建结构化稀疏性的方法。该方法通过训练高效意识的LLM学习激活更少的神经元，取得更好的稀疏性和性能折衷。 |
| [^8] | [Quality > Quantity: Synthetic Corpora from Foundation Models for Closed-Domain Extractive Question Answering.](http://arxiv.org/abs/2310.16995) | 这项工作研究了封闭领域的抽取式问答，引入了有针对性的预训练的概念，并使用Galactica生成了合成的"有针对性"的语料库。 |

# 详细

[^1]: Counting-Stars：一种评估长上下文大型语言模型的简单、高效、合理策略

    Counting-Stars: A Simple, Efficient, and Reasonable Strategy for Evaluating Long-Context Large Language Models

    [https://arxiv.org/abs/2403.11802](https://arxiv.org/abs/2403.11802)

    提出了一种名为Counting-Stars的简单、高效、合理策略，用于评估长上下文大型语言模型的能力，并在实验中发现GPT-4 Turbo和Kimi Chat在此任务上取得显著性能。

    

    近期的研究主要集中在开发具有强大长上下文能力的大型语言模型（LLMs），由于缺乏适当的评估策略，对领先的LLMs（例如ChatGPT和KimiChat）的长上下文处理能力和性能了解甚少。为了填补这一空白，我们提出了一个简单、高效、合理的长上下文LLMs评估策略作为一个新的基准，名为Counting-Stars。Counting-Stars旨在要求LLMs充分理解和捕捉长上下文中的长依赖关系，并能够收集跨越整个上下文的多个证据之间的相互依赖来完成任务。基于Counting-Stars，我们进行实验评估了两个领先的长上下文LLMs，即GPT-4 Turbo和Kimi Chat。实验结果表明，GPT-4 Turbo和Kimi Chat在Counting-Stars任务上取得了显著的表现。

    arXiv:2403.11802v1 Announce Type: new  Abstract: While recent research endeavors have concentrated on developing Large Language Models (LLMs) with robust long-context capabilities, due to the lack of appropriate evaluation strategies, relatively little is known about how well the long-context processing abilities and performance of leading LLMs (e.g., ChatGPT and KimiChat). To address this gap, we propose a simple, efficient, and reasonable strategy for evaluating long-context LLMs as a new benchmark, named Counting-Stars. The Counting-Stars is designed to require LLMs to fully understand and capture long dependencies in long contexts and be able to collect inter-dependency across multiple pieces of evidence spanning the entire context to finish the task. Based on the Counting-Stars, we conduct experiments to evaluate the two leading long-context LLMs, i.e., GPT-4 Turbo and Kimi Chat. The experimental results indicate that GPT-4 Turbo and Kimi Chat achieve significant performance in th
    
[^2]: ProSwitch：知识引导的语言模型微调，生成专业和非专业风格的文本

    ProSwitch: Knowledge-Guided Language Model Fine-Tuning to Generate Professional and Non-Professional Styled Text

    [https://arxiv.org/abs/2403.09131](https://arxiv.org/abs/2403.09131)

    ProSwitch通过知识引导的指令微调，在专业和非专业风格之间生成文本，并在专业性评估和质量评估方面表现出优越性。

    

    大语言模型（LLMs）在各种语言应用中表现出有效性，包括文本摘要和可控文本生成。然而，关于它们通过微调在不同风格间切换的能力的研究仍未被充分探讨。本研究聚焦于文本专业性，并引入了一种新颖的方法，名为ProSwitch，通过知识引导的指令微调，使语言模型具备生成专业和非专业回复的能力。ProSwitch分为三个阶段：数据准备，用于收集领域知识和训练语料库；指令微调，用于优化带有多种指令格式的语言模型；全面评估，用于评估生成文本的专业性区分能力和基于参考的质量。 ProSwitch相对于通用和专门语言模型的比较分析显示了我们的方法的优越性。

    arXiv:2403.09131v1 Announce Type: cross  Abstract: Large Language Models (LLMs) have demonstrated efficacy in various linguistic applications, including text summarization and controlled text generation. However, studies into their capacity of switching between styles via fine-tuning remain underexplored. This study concentrates on textual professionalism and introduces a novel methodology, named ProSwitch, which equips a language model with the ability to produce both professional and non-professional responses through knowledge-guided instruction tuning. ProSwitch unfolds across three phases: data preparation for gathering domain knowledge and training corpus; instruction tuning for optimizing language models with multiple levels of instruction formats; and comprehensive evaluation for assessing the professionalism discrimination and reference-based quality of generated text. Comparative analysis of ProSwitch against both general and specialized language models reveals that our appro
    
[^3]: 在英语和瑞典语中检测未记录的词义

    Detection of Non-recorded Word Senses in English and Swedish

    [https://arxiv.org/abs/2403.02285](https://arxiv.org/abs/2403.02285)

    该研究致力于在英语和瑞典语中检测未记录的词义，通过使用预训练的词-上下文嵌入器和人工注释，成功提高了检测到具有未记录词义的词语用法数量。

    

    这项研究探讨了在英语和瑞典语中进行未知词义检测的任务。该任务的主要目标是确定特定词语用法的含义是否在词典中有记录。为此，使用一个预训练的上下文词嵌入器来比较词义条目与现代和历史语料库中词语用法，从而在少样本情况下建模这一任务。此外，我们使用人类注释来调整和评估我们的模型。与从语料库中随机抽样相比，我们的模型能够显著增加检测到具有未记录词义的词语用法数量。

    arXiv:2403.02285v1 Announce Type: new  Abstract: This study addresses the task of Unknown Sense Detection in English and Swedish. The primary objective of this task is to determine whether the meaning of a particular word usage is documented in a dictionary or not. For this purpose, sense entries are compared with word usages from modern and historical corpora using a pre-trained Word-in-Context embedder that allows us to model this task in a few-shot scenario. Additionally, we use human annotations to adapt and evaluate our models. Compared to a random sample from a corpus, our model is able to considerably increase the detected number of word usages with non-recorded senses.
    
[^4]: 通过强化学习提高自动生成反馈的有效性

    Improving the Validity of Automatically Generated Feedback via Reinforcement Learning

    [https://arxiv.org/abs/2403.01304](https://arxiv.org/abs/2403.01304)

    本研究通过提出评估数学反馈的评分标准，展示了GPT-4能够有效地使用它来注释人工编写的和LLM生成的反馈，从而改进了自动生成反馈的有效性和可靠性。

    

    在智能辅导系统和在线学习平台中，通过大型语言模型（LLMs）自动生成反馈具有改善许多学生学习成果的潜力。本研究解决了自动生成和评估反馈的问题，同时考虑了正确性和一致性。

    arXiv:2403.01304v1 Announce Type: new  Abstract: Automatically generating feedback via large language models (LLMs) in intelligent tutoring systems and online learning platforms has the potential to improve the learning outcomes of many students. However, both feedback generation and evaluation are challenging: feedback content has to be valid especially in subjects like math, which requires models to understand the problem, the solution, and where the student's error lies. Feedback also has to be pedagogically valid to reflect effective tutoring strategies, such as explaining possible misconceptions and encouraging the student, among other desirable features. In this work, we address both problems of automatically generating and evaluating feedback while considering both correctness and alignment. First, we propose a rubric for evaluating math feedback and show that GPT-4 is able to effectively use it to annotate human-written and LLM-generated feedback. Second, we propose a framework
    
[^5]: LLMs带有思维链条是非因果推理者

    LLMs with Chain-of-Thought Are Non-Causal Reasoners

    [https://arxiv.org/abs/2402.16048](https://arxiv.org/abs/2402.16048)

    本文探讨了大型语言模型在推理过程中思维链条（CoT）的作用，发现LLMs在答案生成过程中与人类推理存在差异，相关因素包括语境学习、有监督微调以及对人类反馈的强化学习。

    

    本文探讨了大型语言模型（LLMs）推理中思维链条（CoT）的作用。尽管它有改善任务性能的潜力，但我们的分析揭示了在LLMs中正确答案跟随不正确CoTs的频率及反之。我们采用因果分析来评估CoTs/指令与LLMs答案之间的因果关系，揭示LLMs近似的结构因果模型（SCM）。通过比较暗示SCM与人类推理的SCM，我们突显了LLM和人类推理过程之间的差异。我们进一步研究了影响暗示SCM因果结构的因素，揭示了语境学习、有监督微调以及对人类反馈的强化学习显著影响因果关系。我们在https://github.com/StevenZHB/CoT_Causal_Analysis发布了代码和结果。

    arXiv:2402.16048v1 Announce Type: cross  Abstract: This paper explores the role of the Chain of Thought (CoT) in Large Language Models (LLMs) reasoning. Despite its potential to improve task performance, our analysis reveals a surprising frequency of correct answers following incorrect CoTs and vice versa. We employ causal analysis to assess the cause-effect relationship between CoTs/instructions and answers in LLMs, uncovering the Structural Causal Model (SCM) that LLMs approximate. By comparing the implied SCM with that of human reasoning, we highlight discrepancies between LLM and human reasoning processes. We further examine the factors influencing the causal structure of the implied SCM, revealing that in-context learning, supervised fine-tuning, and reinforcement learning on human feedback significantly impact the causal relations. We release the code and results at https://github.com/StevenZHB/CoT_Causal_Analysis.
    
[^6]: ProSparse: 引入和增强大型语言模型内部激活稀疏性

    ProSparse: Introducing and Enhancing Intrinsic Activation Sparsity within Large Language Models

    [https://arxiv.org/abs/2402.13516](https://arxiv.org/abs/2402.13516)

    本文介绍了一种名为"ProSparse"的有效稀疏化方法，以推动大型语言模型实现更高的激活稀疏性而不降低模型性能

    

    Activation sparsity指的是激活输出中存在许多弱贡献元素。作为使用ReLU激活函数的模型的普遍属性，已被证明是提高模型推理效率的一种有前途的范例。然而，大多数大型语言模型（LLMs）采用了没有内在激活稀疏性的激活函数（例如GELU和Swish）。一些最近的努力尝试引入ReLU或其变体作为替代激活函数，以帮助LLMs实现激活稀疏性和推理加速，但很少能同时获得高稀疏度和可比较的模型性能。本文介绍了一种名为"ProSparse"的有效稀疏化方法，以推动LLMs实现更高的激活稀疏性而不降低模型性能。具体来说，将LLMs的激活函数替换为ReLU后，ProSparse采用渐进稀疏正则化

    arXiv:2402.13516v1 Announce Type: cross  Abstract: Activation sparsity refers to the existence of considerable weakly-contributed elements among activation outputs. As a prevalent property of the models using the ReLU activation function, it has been proven a promising paradigm to boost model inference efficiency. Nevertheless, most large language models (LLMs) adopt activation functions without intrinsic activation sparsity (e.g., GELU and Swish). Some recent efforts have explored introducing ReLU or its variants as the substitutive activation function to help LLMs achieve activation sparsity and inference acceleration, but few can simultaneously obtain high sparsity and comparable model performance. This paper introduces an effective sparsification method named "ProSparse" to push LLMs for higher activation sparsity without decreasing model performance. Specifically, after substituting the activation function of LLMs with ReLU, ProSparse adopts progressive sparsity regularization wit
    
[^7]: 学习变得高效：在大型语言模型中构建结构化稀疏性

    Learn To be Efficient: Build Structured Sparsity in Large Language Models

    [https://arxiv.org/abs/2402.06126](https://arxiv.org/abs/2402.06126)

    本文通过引入一种新的算法"Learn-To-be-Efficient(LTE)"，提出了在大型语言模型(LLM)中构建结构化稀疏性的方法。该方法通过训练高效意识的LLM学习激活更少的神经元，取得更好的稀疏性和性能折衷。

    

    大型语言模型(LLM)以其十亿级参数取得了显著的成功，但它们产生了高昂的推理开销。在LLM中出现的激活稀疏性为通过仅涉及部分参数进行推理提供了一种自然的方法来减少这种成本。现有方法只关注利用这种自然形成的激活稀疏性，忽视了进一步放大这种固有稀疏性的潜力。本文中，我们假设LLM可以通过实现更结构化的激活稀疏性来学习高效。为实现这一目标，我们引入了一种新颖的算法"Learn-To-be-Efficient(LTE)", 旨在训练高效意识的LLM学习激活更少的神经元，并在稀疏性和性能之间取得更好的折衷。此外，与主要关注基于ReLU模型的SOTA MoEfication方法不同，LTE还可以应用于像GPT和LLaMA这样具有软激活函数的LLM。我们在四个模型和十一个数据集上评估了LTE。

    Large Language Models (LLMs) have achieved remarkable success with their billion-level parameters, yet they incur high inference overheads. The emergence of activation sparsity in LLMs provides a natural approach to reduce this cost by involving only parts of the parameters for inference. Existing methods only focus on utilizing this naturally formed activation sparsity, overlooking the potential for further amplifying this inherent sparsity. In this paper, we hypothesize that LLMs can learn to be efficient by achieving more structured activation sparsity.To achieve this, we introduce a novel algorithm, Learn-To-be-Efficient (LTE), designed to train efficiency-aware LLMs to learn to activate fewer neurons and achieve a better trade-off between sparsity and performance. Furthermore, unlike SOTA MoEfication methods, which mainly focus on ReLU-based models, LTE can also be applied to LLMs like GPT and LLaMA with soft activation functions. We evaluate LTE on four models and eleven datasets
    
[^8]: 质量>数量：基于基础模型的封闭领域抽取式问答的合成语料库

    Quality > Quantity: Synthetic Corpora from Foundation Models for Closed-Domain Extractive Question Answering. (arXiv:2310.16995v1 [cs.CL])

    [http://arxiv.org/abs/2310.16995](http://arxiv.org/abs/2310.16995)

    这项工作研究了封闭领域的抽取式问答，引入了有针对性的预训练的概念，并使用Galactica生成了合成的"有针对性"的语料库。

    

    领域适应是将模型在一个领域进行训练，然后应用于另一个领域的过程，在机器学习领域得到了广泛研究。虽然从头开始训练一个特定领域的基础模型(FM)是一个选择，但最近的方法集中在调整预训练的FM以满足特定领域的任务。然而，我们的实验发现，无论是哪种方法，在目标领域都无法始终达到最先进( SOTA)的结果。在这项工作中，我们研究封闭领域内的抽取式问答，并引入了有针对性的预训练的概念。这意味着确定和生成相关数据，进一步预训练我们的模型，而不是传统的利用在广泛数据上训练的特定领域的FM的理念。我们提出的框架使用Galactica生成与特定写作风格和主题(如研究论文和放射学报告)相一致的合成"有针对性"的语料库。这个过程可以看作是一种知识

    Domain adaptation, the process of training a model in one domain and applying it to another, has been extensively explored in machine learning. While training a domain-specific foundation model (FM) from scratch is an option, recent methods have focused on adapting pre-trained FMs for domain-specific tasks. However, our experiments reveal that either approach does not consistently achieve state-of-the-art (SOTA) results in the target domain. In this work, we study extractive question answering within closed domains and introduce the concept of targeted pre-training. This involves determining and generating relevant data to further pre-train our models, as opposed to the conventional philosophy of utilizing domain-specific FMs trained on a wide range of data. Our proposed framework uses Galactica to generate synthetic, ``targeted'' corpora that align with specific writing styles and topics, such as research papers and radiology reports. This process can be viewed as a form of knowledge 
    

