# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [TMT: Tri-Modal Translation between Speech, Image, and Text by Processing Different Modalities as Different Languages](https://arxiv.org/abs/2402.16021) | 将不同模态解释为不同语言，在语音、图像和文本之间实现了三模翻译，大大减少了计算成本。 |
| [^2] | [Structure Guided Large Language Model for SQL Generation](https://arxiv.org/abs/2402.13284) | 通过引入结构信息，提出了一个结构引导的SQL生成模型，以改善大型语言模型生成SQL的准确性和可执行性。 |
| [^3] | [CAT-LLM: Prompting Large Language Models with Text Style Definition for Chinese Article-style Transfer.](http://arxiv.org/abs/2401.05707) | CAT-LLM是一个中文文章风格转换框架，利用大语言模型（LLM）和文本风格定义（TSD）模块，可以有效地将中文文章转换为不同的风格。该框架通过从词和句子级别分析文章风格，并支持动态扩展内部风格树，使得风格转换能力更强大。 |

# 详细

[^1]: TMT: 通过将不同模态视为不同语言来实现语音、图像和文本之间的三模翻译

    TMT: Tri-Modal Translation between Speech, Image, and Text by Processing Different Modalities as Different Languages

    [https://arxiv.org/abs/2402.16021](https://arxiv.org/abs/2402.16021)

    将不同模态解释为不同语言，在语音、图像和文本之间实现了三模翻译，大大减少了计算成本。

    

    能够共同处理多模态信息正在成为一项重要任务。然而，有限的配对多模态数据和多模态学习中的大量计算要求阻碍了发展。我们提出了一种新颖的三模翻译（TMT）模型，可以在涵盖语音、图像和文本的任意模态之间进行翻译。我们引入了一个新颖的观点，即将不同模态解释为不同语言，并将多模态翻译视为一个成熟的机器翻译问题。为此，我们将语音和图像数据标记为离散标记，提供了跨模态的统一接口，并大大降低了计算成本。在提出的TMT中，多模态编码器-解码器进行核心翻译，而模态特定处理仅在标记化和去标记化阶段内进行。我们在所有六种模态上评估了提出的TMT。

    arXiv:2402.16021v1 Announce Type: cross  Abstract: The capability to jointly process multi-modal information is becoming an essential task. However, the limited number of paired multi-modal data and the large computational requirements in multi-modal learning hinder the development. We propose a novel Tri-Modal Translation (TMT) model that translates between arbitrary modalities spanning speech, image, and text. We introduce a novel viewpoint, where we interpret different modalities as different languages, and treat multi-modal translation as a well-established machine translation problem. To this end, we tokenize speech and image data into discrete tokens, which provide a unified interface across modalities and significantly decrease the computational cost. In the proposed TMT, a multi-modal encoder-decoder conducts the core translation, whereas modality-specific processing is conducted only within the tokenization and detokenization stages. We evaluate the proposed TMT on all six mod
    
[^2]: 结构引导的大型语言模型用于SQL生成

    Structure Guided Large Language Model for SQL Generation

    [https://arxiv.org/abs/2402.13284](https://arxiv.org/abs/2402.13284)

    通过引入结构信息，提出了一个结构引导的SQL生成模型，以改善大型语言模型生成SQL的准确性和可执行性。

    

    生成准确的结构化查询语言（SQL）是一个长期存在的问题，特别是在将用户的语义查询与结构化数据库匹配，然后生成结构化SQL方面。现有模型通常将查询和数据库模式输入到LLM中，并依赖LLM执行语义-结构匹配并生成结构化SQL。然而，这种解决方案忽略了用户查询和数据库中的结构信息，而这些信息可以用来增强结构化SQL的生成。这一疏忽可能导致不准确或无法执行的SQL生成。为了充分利用结构，我们提出了一个结构到SQL的框架，利用固有的结构信息来改善LLM的SQL生成。具体地，我们介绍了我们的结构引导SQL（SGU-SQL）生成模型。

    arXiv:2402.13284v1 Announce Type: cross  Abstract: Generating accurate Structured Querying Language (SQL) is a long-standing problem, especially in matching users' semantic queries with structured databases and then generating structured SQL. Existing models typically input queries and database schemas into the LLM and rely on the LLM to perform semantic-structure matching and generate structured SQL. However, such solutions overlook the structural information within user queries and databases, which can be utilized to enhance the generation of structured SQL. This oversight can lead to inaccurate or unexecutable SQL generation. To fully exploit the structure, we propose a structure-to-SQL framework, which leverages the inherent structure information to improve the SQL generation of LLMs. Specifically, we introduce our Structure Guided SQL~(SGU-SQL) generation model. SGU-SQL first links user queries and databases in a structure-enhanced manner. It then decomposes complicated linked str
    
[^3]: CAT-LLM: 使用文本风格定义为基础，为中文文章风格转换提供指导的大语言模型

    CAT-LLM: Prompting Large Language Models with Text Style Definition for Chinese Article-style Transfer. (arXiv:2401.05707v1 [cs.CL])

    [http://arxiv.org/abs/2401.05707](http://arxiv.org/abs/2401.05707)

    CAT-LLM是一个中文文章风格转换框架，利用大语言模型（LLM）和文本风格定义（TSD）模块，可以有效地将中文文章转换为不同的风格。该框架通过从词和句子级别分析文章风格，并支持动态扩展内部风格树，使得风格转换能力更强大。

    

    文本风格转换在在线娱乐和社交媒体中越来越受关注。然而，现有的研究主要集中在单个英文句子内的风格转换，而忽略了长篇中文文本的复杂性，限制了风格转换在数字媒体领域的广泛应用。为了弥补这一差距，我们提出了一个中文文章风格转换框架（CAT-LLM），利用了大语言模型（LLM）的能力。CAT-LLM包括一个定制的、可替换的文本风格定义（TSD）模块，旨在全面分析文章中的文本特征，以便有效地转换中文文章风格。TSD模块集成了一系列机器学习算法，从词和句子级别分析文章风格，从而帮助LLM全面把握目标风格，同时不损失原始文本的完整性。此外，该模块支持内部风格树的动态扩展，展示了强大的风格转换能力。

    Text style transfer is increasingly prominent in online entertainment and social media. However, existing research mainly concentrates on style transfer within individual English sentences, while ignoring the complexity of long Chinese texts, which limits the wider applicability of style transfer in digital media realm. To bridge this gap, we propose a Chinese Article-style Transfer framework (CAT-LLM), leveraging the capabilities of Large Language Models (LLMs). CAT-LLM incorporates a bespoke, pluggable Text Style Definition (TSD) module aimed at comprehensively analyzing text features in articles, prompting LLMs to efficiently transfer Chinese article-style. The TSD module integrates a series of machine learning algorithms to analyze article-style from both words and sentences levels, thereby aiding LLMs thoroughly grasp the target style without compromising the integrity of the original text. In addition, this module supports dynamic expansion of internal style trees, showcasing rob
    

