# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Experimenting AI Technologies for Disinformation Combat: the IDMO Project.](http://arxiv.org/abs/2310.11097) | IDMO项目旨在使用人工智能技术打击虚假信息和假新闻，其贡献包括创建新型数据集、开发自动模型、评估GPT-4等。 |
| [^2] | [Distilled Feature Fields Enable Few-Shot Language-Guided Manipulation.](http://arxiv.org/abs/2308.07931) | 本论文通过精简特征场，将精确的3D几何与2D基础模型的丰富语义相结合，实现了对未见过的物体的少样本操作的泛化能力。 |
| [^3] | [Is ChatGPT Involved in Texts? Measure the Polish Ratio to Detect ChatGPT-Generated Text.](http://arxiv.org/abs/2307.11380) | 本研究针对ChatGPT在文本生成中的作用进行了研究，提出了一种新的测量方法“波兰比率”，用于检测ChatGPT生成的文本中的涉及程度。同时，还引入了一个新的数据集HPPT，用于构建更稳健的检测器。 |
| [^4] | [Can Large Language Models Infer Causation from Correlation?.](http://arxiv.org/abs/2306.05836) | 本文提出了一个新的任务（Corr2Cause），用于测量大型语言模型的因果推断能力，并通过实验发现这些模型在这个任务上表现很差。 |
| [^5] | [WikiSQE: A Large-Scale Dataset for Sentence Quality Estimation in Wikipedia.](http://arxiv.org/abs/2305.05928) | WikiSQE是第一个用于维基百科中句子质量估计的大规模数据集，其中包含约3.4M个句子和153个质量标签。在这个数据集上进行的实验表明，具有引文、语法/语义或命题问题的句子更难以检测。 |
| [^6] | [Exploring AI-Generated Text in Student Writing: How Does AI Help?.](http://arxiv.org/abs/2304.02478) | 研究发现，在学生写作中使用人工智能生成文本有一定好处，但过度依赖此类工具也存在潜在风险。 |

# 详细

[^1]: 用于打击虚假信息的人工智能技术的实验：IDMO项目

    Experimenting AI Technologies for Disinformation Combat: the IDMO Project. (arXiv:2310.11097v1 [cs.CL])

    [http://arxiv.org/abs/2310.11097](http://arxiv.org/abs/2310.11097)

    IDMO项目旨在使用人工智能技术打击虚假信息和假新闻，其贡献包括创建新型数据集、开发自动模型、评估GPT-4等。

    

    意大利数字媒体观察项目（IDMO）是欧洲一项倡议的一部分，专注于打击虚假信息和假新闻。本报告概述了Rai-CRITS在该项目中的贡献，包括：（i）创建用于测试技术的新型数据集，（ii）开发自动模型，用于分类Pagella Politica的裁决以便于更广泛的分析，（iii）创建自动模型，对FEVER数据集上的文本蕴含具有异常精度的识别能力，（iv）使用GPT-4评估文本蕴含， （v）在国家活动中开展提高对假新闻意识的游戏。

    The Italian Digital Media Observatory (IDMO) project, part of a European initiative, focuses on countering disinformation and fake news. This report outlines contributions from Rai-CRITS to the project, including: (i) the creation of novel datasets for testing technologies (ii) development of an automatic model for categorizing Pagella Politica verdicts to facilitate broader analysis (iii) creation of an automatic model for recognizing textual entailment with exceptional accuracy on the FEVER dataset (iv) assessment using GPT-4 to identify textual entailmen (v) a game to raise awareness about fake news at national events.
    
[^2]: 精简特征场使得语言引导的少样本操作成为可能

    Distilled Feature Fields Enable Few-Shot Language-Guided Manipulation. (arXiv:2308.07931v1 [cs.CV])

    [http://arxiv.org/abs/2308.07931](http://arxiv.org/abs/2308.07931)

    本论文通过精简特征场，将精确的3D几何与2D基础模型的丰富语义相结合，实现了对未见过的物体的少样本操作的泛化能力。

    

    自监督和语言监督的图像模型包含了世界的丰富知识，对于泛化很重要。然而，许多机器人任务需要对 3D 几何的详细理解，这在 2D 图像特征中往往缺乏。本研究通过利用精简特征场，将精确的 3D 几何与 2D 基础模型的丰富语义相结合，来弥合机器人操作中的 2D 到 3D 的差距。我们提出一种针对 6 自由度抓取和放置的少样本学习方法，利用这些强大的空间和语义先验，实现对未见过的物体的自然泛化。通过从视觉语言模型 CLIP 中精简的特征，我们展示了一种通过自由文本自然语言指定新颖对象进行操作的方式，并展示了它在未见过的表达和新颖类别的物体上的泛化能力。

    Self-supervised and language-supervised image models contain rich knowledge of the world that is important for generalization. Many robotic tasks, however, require a detailed understanding of 3D geometry, which is often lacking in 2D image features. This work bridges this 2D-to-3D gap for robotic manipulation by leveraging distilled feature fields to combine accurate 3D geometry with rich semantics from 2D foundation models. We present a few-shot learning method for 6-DOF grasping and placing that harnesses these strong spatial and semantic priors to achieve in-the-wild generalization to unseen objects. Using features distilled from a vision-language model, CLIP, we present a way to designate novel objects for manipulation via free-text natural language, and demonstrate its ability to generalize to unseen expressions and novel categories of objects.
    
[^3]: 聊天GPT生成的文本中是否涉及ChatGPT？通过测量“波兰比率”来检测ChatGPT生成的文本

    Is ChatGPT Involved in Texts? Measure the Polish Ratio to Detect ChatGPT-Generated Text. (arXiv:2307.11380v1 [cs.CL])

    [http://arxiv.org/abs/2307.11380](http://arxiv.org/abs/2307.11380)

    本研究针对ChatGPT在文本生成中的作用进行了研究，提出了一种新的测量方法“波兰比率”，用于检测ChatGPT生成的文本中的涉及程度。同时，还引入了一个新的数据集HPPT，用于构建更稳健的检测器。

    

    大规模语言模型如ChatGPT在文本生成方面具有显著能力，这激发了研究人员开发检测器以减轻潜在风险，包括错误信息、网络钓鱼和学术不诚实。然而，大多数以前的研究主要针对区分纯粹由ChatGPT生成的文本和人工撰写的文本的检测器。然而，这种方法在区分通过人机协作生成的文本（例如ChatGPT润色的文本）上失效。为了填补这一空白，我们引入了一个新颖的数据集HPPT（ChatGPT润色的学术摘要），以构建更强大的检测器。该数据集与现有语料库不同，它包括人工撰写的文本和ChatGPT润色的摘要对，而不仅仅是ChatGPT生成的文本。此外，我们提出了“波兰比率”的方法，这是一种衡量ChatGPT在文本生成中参与程度的创新指标。

    The remarkable capabilities of large-scale language models, such as ChatGPT, in text generation have incited awe and spurred researchers to devise detectors to mitigate potential risks, including misinformation, phishing, and academic dishonesty. Despite this, most previous studies, including HC3, have been predominantly geared towards creating detectors that differentiate between purely ChatGPT-generated texts and human-authored texts. This approach, however, fails to work on discerning texts generated through human-machine collaboration, such as ChatGPT-polished texts. Addressing this gap, we introduce a novel dataset termed HPPT (ChatGPT-polished academic abstracts), facilitating the construction of more robust detectors. It diverges from extant corpora by comprising pairs of human-written and ChatGPT-polished abstracts instead of purely ChatGPT-generated texts. Additionally, we propose the "Polish Ratio" method, an innovative measure of ChatGPT's involvement in text generation base
    
[^4]: 大型语言模型能否从相关性中推断出因果关系?

    Can Large Language Models Infer Causation from Correlation?. (arXiv:2306.05836v1 [cs.CL])

    [http://arxiv.org/abs/2306.05836](http://arxiv.org/abs/2306.05836)

    本文提出了一个新的任务（Corr2Cause），用于测量大型语言模型的因果推断能力，并通过实验发现这些模型在这个任务上表现很差。

    

    因果推断是人类智慧的标志之一。虽然CausalNLP领域近年来引起了广泛关注，但NLP中现有的因果推断数据集主要依赖于从经验知识（例如常识知识）中发现因果关系。在本文中，我们提出了第一个基准数据集，用于测试大型语言模型（LLM）的纯因果推断能力。具体而言，我们制定了一个新的任务Corr2Cause，它采用一组相关语句并确定变量之间的因果关系。我们策划了一个大规模的数据集，其中包含超过400K个样本，我们在其中评估了17个现有的LLMs。通过我们的实验，我们确定了LLMs在因果推断技能方面的一个关键缺陷，并表明这些模型在该任务上的表现几乎接近随机。当我们尝试通过微调将LLMs重新用于这种技能时，这种缺陷在某种程度上得到了缓解，但我们发现这些模型仍然失败了。

    Causal inference is one of the hallmarks of human intelligence. While the field of CausalNLP has attracted much interest in the recent years, existing causal inference datasets in NLP primarily rely on discovering causality from empirical knowledge (e.g., commonsense knowledge). In this work, we propose the first benchmark dataset to test the pure causal inference skills of large language models (LLMs). Specifically, we formulate a novel task Corr2Cause, which takes a set of correlational statements and determines the causal relationship between the variables. We curate a large-scale dataset of more than 400K samples, on which we evaluate seventeen existing LLMs. Through our experiments, we identify a key shortcoming of LLMs in terms of their causal inference skills, and show that these models achieve almost close to random performance on the task. This shortcoming is somewhat mitigated when we try to re-purpose LLMs for this skill via finetuning, but we find that these models still fa
    
[^5]: WikiSQE：维基百科中句子质量估计的大规模数据集

    WikiSQE: A Large-Scale Dataset for Sentence Quality Estimation in Wikipedia. (arXiv:2305.05928v1 [cs.CL])

    [http://arxiv.org/abs/2305.05928](http://arxiv.org/abs/2305.05928)

    WikiSQE是第一个用于维基百科中句子质量估计的大规模数据集，其中包含约3.4M个句子和153个质量标签。在这个数据集上进行的实验表明，具有引文、语法/语义或命题问题的句子更难以检测。

    

    维基百科可以被任何人编辑，因此包含各种质量的句子。因此，维基百科包含一些质量较差的编辑，这些编辑通常会被其他编辑标记。虽然编辑的评论增强了维基百科的可信度，但很难检查所有编辑的文本。协助这个过程非常重要，但目前还没有一个大而全面的数据集来研究它。在这里，我们提出了 WikiSQE，这是第一个用于维基百科中句子质量估计的大规模数据集。每个句子都是从维基百科的整个修订历史中提取的，并且目标质量标签经过了仔细的调查和选择。WikiSQE具有约3.4 million个句子和153个质量标签。在使用竞争机器学习模型进行自动分类的实验中，发现具有引文，语法/语义或命题问题的句子更难以检测。此外，我们进行了自动作文评分实验，以评估生成摘要的有效性。

    Wikipedia can be edited by anyone and thus contains various quality sentences. Therefore, Wikipedia includes some poor-quality edits, which are often marked up by other editors. While editors' reviews enhance the credibility of Wikipedia, it is hard to check all edited text. Assisting in this process is very important, but a large and comprehensive dataset for studying it does not currently exist. Here, we propose WikiSQE, the first large-scale dataset for sentence quality estimation in Wikipedia. Each sentence is extracted from the entire revision history of Wikipedia, and the target quality labels were carefully investigated and selected. WikiSQE has about 3.4 M sentences with 153 quality labels. In the experiment with automatic classification using competitive machine learning models, sentences that had problems with citation, syntax/semantics, or propositions were found to be more difficult to detect. In addition, we conducted automated essay scoring experiments to evaluate the gen
    
[^6]: 探索学生写作中的人工智能生成文本：AI能起到什么作用？

    Exploring AI-Generated Text in Student Writing: How Does AI Help?. (arXiv:2304.02478v1 [cs.CL])

    [http://arxiv.org/abs/2304.02478](http://arxiv.org/abs/2304.02478)

    研究发现，在学生写作中使用人工智能生成文本有一定好处，但过度依赖此类工具也存在潜在风险。

    

    以英语作为外语的学生使用人工智能自然语言生成工具生成的文本可能会提高他们的写作质量。然而，目前尚不清楚这些学生的写作中使用人工智能生成文本在多大程度上会导致更高质量的写作。我们探索了23名香港中学生撰写故事（包含自己的文字和人工智能生成的文本）的尝试。人类专家对这些故事进行了内容、语言和组织方面的评分。我们分析了故事中的AI生成文本的基本组织结构和句法复杂度，并执行了多元线性回归和聚类分析。结果表明，人类词语的数量和人工智能生成词语的数量对分数有重要贡献。此外，与同龄人相比，学生的写作可以分为擅长和不擅长使用更多或更少人工智能生成文本的两组。聚类比较显示，使用人工智能生成文本在学生写作中有一定好处，但同时也强调了过度依赖这种工具的潜在风险。

    English as foreign language_EFL_students' use of text generated from artificial intelligence_AI_natural language generation_NLG_tools may improve their writing quality. However, it remains unclear to what extent AI-generated text in these students' writing might lead to higher-quality writing. We explored 23 Hong Kong secondary school students' attempts to write stories comprising their own words and AI-generated text. Human experts scored the stories for dimensions of content, language and organization. We analyzed the basic organization and structure and syntactic complexity of the stories' AI-generated text and performed multiple linear regression and cluster analyses. The results show the number of human words and the number of AI-generated words contribute significantly to scores. Besides, students can be grouped into competent and less competent writers who use more AI-generated text or less AI-generated text compared to their peers. Comparisons of clusters reveal some benefit of
    

