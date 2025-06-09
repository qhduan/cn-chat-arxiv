# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [TMT: Tri-Modal Translation between Speech, Image, and Text by Processing Different Modalities as Different Languages](https://arxiv.org/abs/2402.16021) | 将不同模态解释为不同语言，在语音、图像和文本之间实现了三模翻译，大大减少了计算成本。 |
| [^2] | [Structure Guided Large Language Model for SQL Generation](https://arxiv.org/abs/2402.13284) | 通过引入结构信息，提出了一个结构引导的SQL生成模型，以改善大型语言模型生成SQL的准确性和可执行性。 |
| [^3] | [Vehicle: Bridging the Embedding Gap in the Verification of Neuro-Symbolic Programs.](http://arxiv.org/abs/2401.06379) | 本文提出了一个名为Vehicle的工具，它能够在验证神经符号化程序中弥合嵌入缺口，为指定神经网络属性和解释其与嵌入空间的关系提供了方便的语言和强大的编译器。 |
| [^4] | [Diffusion Policies for Out-of-Distribution Generalization in Offline Reinforcement Learning.](http://arxiv.org/abs/2307.04726) | 该论文介绍了一种名为状态重构扩散策略 (SRDP) 的新方法，该方法在最新的扩散策略类中引入了状态重构特征学习，以解决脱机强化学习中的分布偏移和有效表示策略的问题。 |

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
    
[^3]: Vehicle: 在验证神经符号化程序中弥合嵌入缺口

    Vehicle: Bridging the Embedding Gap in the Verification of Neuro-Symbolic Programs. (arXiv:2401.06379v1 [cs.AI])

    [http://arxiv.org/abs/2401.06379](http://arxiv.org/abs/2401.06379)

    本文提出了一个名为Vehicle的工具，它能够在验证神经符号化程序中弥合嵌入缺口，为指定神经网络属性和解释其与嵌入空间的关系提供了方便的语言和强大的编译器。

    

    神经符号化程序是包含机器学习组件和传统符号化代码的程序，正在变得越来越普遍。然而，我们认为目前缺乏一种通用的方法来验证这些程序，因为它们的正确性取决于机器学习组件的行为。在本文中，我们将“嵌入缺口”——以语义有意义的“问题空间”属性与等效的“嵌入空间”属性之间缺乏技术的问题——视为关键问题之一，并描述了一个名为Vehicle的工具，以模块化的方式促进神经符号化程序的端到端验证。Vehicle提供了一种方便的语言，用于指定神经网络的“问题空间”属性并声明它们与“嵌入空间”的关系，以及一个强大的编译器，可以自动将这些属性解释为所选择的机器学习训练环境、神经网络验证工具支持的语言。

    Neuro-symbolic programs -- programs containing both machine learning components and traditional symbolic code -- are becoming increasingly widespread. However, we believe that there is still a lack of a general methodology for verifying these programs whose correctness depends on the behaviour of the machine learning components. In this paper, we identify the ``embedding gap'' -- the lack of techniques for linking semantically-meaningful ``problem-space'' properties to equivalent ``embedding-space'' properties -- as one of the key issues, and describe Vehicle, a tool designed to facilitate the end-to-end verification of neural-symbolic programs in a modular fashion. Vehicle provides a convenient language for specifying ``problem-space'' properties of neural networks and declaring their relationship to the ``embedding-space", and a powerful compiler that automates interpretation of these properties in the language of a chosen machine-learning training environment, neural network verifie
    
[^4]: 脱机强化学习中的离散策略的扩散策略

    Diffusion Policies for Out-of-Distribution Generalization in Offline Reinforcement Learning. (arXiv:2307.04726v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2307.04726](http://arxiv.org/abs/2307.04726)

    该论文介绍了一种名为状态重构扩散策略 (SRDP) 的新方法，该方法在最新的扩散策略类中引入了状态重构特征学习，以解决脱机强化学习中的分布偏移和有效表示策略的问题。

    

    脱机强化学习 (RL) 方法利用以前的经验来学习比用于数据收集的行为策略更好的策略。与行为克隆相反，行为克隆假设数据是从专家演示中收集的，而脱机 RL 可以使用非专家数据和多模态行为策略。然而，脱机 RL 算法在处理分布偏移和有效表示策略方面面临挑战，因为训练过程中缺乏在线交互。先前关于脱机 RL 的工作使用条件扩散模型来表示数据集中的多模态行为。然而，这些方法并没有针对缓解脱机分布状态泛化而制定。我们介绍了一种新的方法，名为状态重构扩散策略 (SRDP)，将状态重构特征学习纳入到最新的扩散策略类中，以解决脱机分布通用化问题。状态重构损失促进了更详细的描述。

    Offline Reinforcement Learning (RL) methods leverage previous experiences to learn better policies than the behavior policy used for data collection. In contrast to behavior cloning, which assumes the data is collected from expert demonstrations, offline RL can work with non-expert data and multimodal behavior policies. However, offline RL algorithms face challenges in handling distribution shifts and effectively representing policies due to the lack of online interaction during training. Prior work on offline RL uses conditional diffusion models to represent multimodal behavior in the dataset. Nevertheless, these methods are not tailored toward alleviating the out-of-distribution state generalization. We introduce a novel method, named State Reconstruction for Diffusion Policies (SRDP), incorporating state reconstruction feature learning in the recent class of diffusion policies to address the out-of-distribution generalization problem. State reconstruction loss promotes more descript
    

