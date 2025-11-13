# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [NaturalTurn: A Method to Segment Transcripts into Naturalistic Conversational Turns](https://arxiv.org/abs/2403.15615) | NaturalTurn是一种专门设计用于准确捕捉自然对话交流动态的轮次分割算法，通过区分说话者的主要对话轮次和听众的次要话语，能够比现有方法更好地提取转录信息。 |
| [^2] | [Understanding Performance of Long-Document Ranking Models through Comprehensive Evaluation and Leaderboarding](https://arxiv.org/abs/2207.01262) | 在标准收集的初步实验中，我们发现长文档模型在MRR或NDCG方面性能不佳，表现低于FirstP，或平均最多超越5％。我们推测这不是因为模型无法处理长上下文，而是由于相关段落具有位置偏见，往往位于前512个文档标记之中。我们找到证据表明这种偏见至少存在于两个测试集中，这促使我们创建了一个新的收集MS MARCO FarRelevant，其中包含 |
| [^3] | [Prompt Injection Attacks and Defenses in LLM-Integrated Applications.](http://arxiv.org/abs/2310.12815) | 本文提出了一个通用框架来形式化提示注入攻击，并系统化防御这种类型的攻击。 |

# 详细

[^1]: NaturalTurn：一种将转录件分割成自然对话转折的方法

    NaturalTurn: A Method to Segment Transcripts into Naturalistic Conversational Turns

    [https://arxiv.org/abs/2403.15615](https://arxiv.org/abs/2403.15615)

    NaturalTurn是一种专门设计用于准确捕捉自然对话交流动态的轮次分割算法，通过区分说话者的主要对话轮次和听众的次要话语，能够比现有方法更好地提取转录信息。

    

    arXiv:2403.15615v1 公告类型: 新的 摘要: 对话是社会、认知和计算科学越来越感兴趣的主题。然而，随着对话数据集的规模和复杂性不断增加，研究人员缺乏可伸缩的方法将语音转录转换为会话轮次——社会互动的基本构建模块。我们介绍了“NaturalTurn”，一种旨在准确捕捉自然交流动态的轮次分割算法。NaturalTurn通过区分说话者的主要对话轮次和听众的次要话语，如背景声、简短插话和其他表现对话特征的平行言语形式，来运作。使用大型对话语料库的数据，我们展示了与现有方法派生的转录相比，NaturalTurn派生的转录表现出有利的统计和推断特性。NaturalTurn算法代表了一种改进。

    arXiv:2403.15615v1 Announce Type: new  Abstract: Conversation is the subject of increasing interest in the social, cognitive, and computational sciences. And yet, as conversational datasets continue to increase in size and complexity, researchers lack scalable methods to segment speech-to-text transcripts into conversational turns--the basic building blocks of social interaction. We introduce "NaturalTurn," a turn segmentation algorithm designed to accurately capture the dynamics of naturalistic exchange. NaturalTurn operates by distinguishing speakers' primary conversational turns from listeners' secondary utterances, such as backchannels, brief interjections, and other forms of parallel speech that characterize conversation. Using data from a large conversation corpus, we show how NaturalTurn-derived transcripts demonstrate favorable statistical and inferential characteristics compared to transcripts derived from existing methods. The NaturalTurn algorithm represents an improvement i
    
[^2]: 通过全面评估和Leaderboarding理解长文档排名模型的性能

    Understanding Performance of Long-Document Ranking Models through Comprehensive Evaluation and Leaderboarding

    [https://arxiv.org/abs/2207.01262](https://arxiv.org/abs/2207.01262)

    在标准收集的初步实验中，我们发现长文档模型在MRR或NDCG方面性能不佳，表现低于FirstP，或平均最多超越5％。我们推测这不是因为模型无法处理长上下文，而是由于相关段落具有位置偏见，往往位于前512个文档标记之中。我们找到证据表明这种偏见至少存在于两个测试集中，这促使我们创建了一个新的收集MS MARCO FarRelevant，其中包含

    

    我们评估了20多个用于长文档排名的Transformer模型（包括最近使用FlashAttention训练的LongP模型），并将它们与简单的FirstP基线进行了比较（将相同模型应用于输入截断为前512个标记）。我们使用MS MARCO文档v1作为主要训练集，并在零-shot场景下评估了模型，以及在对其他收集进行微调后评估了模型。

    arXiv:2207.01262v2 Announce Type: replace-cross  Abstract: We evaluated 20+ Transformer models for ranking of long documents (including recent LongP models trained with FlashAttention) and compared them with simple FirstP baselines (applying the same model to input truncated to the first 512 tokens). We used MS MARCO Documents v1 as a primary training set and evaluated models in the zero-shot scenario as well as after fine-tuning on other collections.   In our initial experiments with standard collections we found that long-document models underperformed FirstP or outperformed it by at most 5% on average in terms of MRR or NDCG. We then conjectured that this was not due to models inability to process long context but rather due to a positional bias of relevant passages, which tended to be among the first 512 document tokens. We found evidence that this bias was, indeed, present in at least two test sets, which motivated us to create a new collection MS MARCO FarRelevant where the relev
    
[^3]: LLM-集成应用中的提示注入攻击和防御

    Prompt Injection Attacks and Defenses in LLM-Integrated Applications. (arXiv:2310.12815v1 [cs.CR])

    [http://arxiv.org/abs/2310.12815](http://arxiv.org/abs/2310.12815)

    本文提出了一个通用框架来形式化提示注入攻击，并系统化防御这种类型的攻击。

    

    大型语言模型（LLMs）越来越多地用作各种称为LLM-集成应用的实际应用程序的后端。最近的多项研究表明，LLM-集成应用容易受到提示注入攻击的威胁，攻击者可以将恶意指令/数据注入这些应用程序的输入中，以达到攻击者的预期结果。然而，现有的研究仅限于案例研究，缺乏对提示注入攻击及其防御的系统理解。本论文旨在填补这一空白。我们提出了一个通用框架来形式化提示注入攻击，并将研究论文和博客文章中讨论的现有攻击视为我们框架的特例。我们的框架使我们能够通过组合现有攻击设计新的攻击方式。此外，我们还提出了一个系统化提示注入攻击防御的框架。利用我们的框架，我们可以预防和缓解这种类型的攻击。

    Large Language Models (LLMs) are increasingly deployed as the backend for a variety of real-world applications called LLM-Integrated Applications. Multiple recent works showed that LLM-Integrated Applications are vulnerable to prompt injection attacks, in which an attacker injects malicious instruction/data into the input of those applications such that they produce results as the attacker desires. However, existing works are limited to case studies. As a result, the literature lacks a systematic understanding of prompt injection attacks and their defenses. We aim to bridge the gap in this work. In particular, we propose a general framework to formalize prompt injection attacks. Existing attacks, which are discussed in research papers and blog posts, are special cases in our framework. Our framework enables us to design a new attack by combining existing attacks. Moreover, we also propose a framework to systematize defenses against prompt injection attacks. Using our frameworks, we con
    

