# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Quality-Diversity through AI Feedback.](http://arxiv.org/abs/2310.13032) | 基于AI反馈的质量-多样性（QDAIF）算法利用语言模型来生成和评估创造性写作，比传统算法更广泛地覆盖高质量样本的搜索空间。 |
| [^2] | [Measuring Pointwise $\mathcal{V}$-Usable Information In-Context-ly.](http://arxiv.org/abs/2310.12300) | 这项研究适用于上下文学习范式，通过调整逐点可用信息度量指标为适用于上下文的版本，将其命名为上下文PVI，并证明了上下文PVI的可靠性和稳定性。 |
| [^3] | [Conversational Health Agents: A Personalized LLM-Powered Agent Framework.](http://arxiv.org/abs/2310.02374) | 该论文介绍了一个基于LLM的会话式健康代理框架，旨在为代理赋予批判性思维、知识获取和问题解决能力，实现个性化的健康护理服务。该框架能够无缝集成医疗工具，实现多语言和多模态对话，并与多种用户数据分析工具连接。 |
| [^4] | [Spider4SPARQL: A Complex Benchmark for Evaluating Knowledge Graph Question Answering Systems.](http://arxiv.org/abs/2309.16248) | Spider4SPARQL是一个新的SPARQL基准数据集，其中包含大量手工生成的NL问题和独特、新颖、复杂的SPARQL查询，旨在评估知识图谱问答系统。 |
| [^5] | [Defending Against Alignment-Breaking Attacks via Robustly Aligned LLM.](http://arxiv.org/abs/2309.14348) | 本文提出了一种稳健对齐的LLM（RA-LLM），用于防御可能发生的对齐破坏攻击。RA-LLM可以直接在现有的对齐LLM上构建，并通过稳健的对齐检查函数来确保其有效性。 |
| [^6] | [A Chinese Prompt Attack Dataset for LLMs with Evil Content.](http://arxiv.org/abs/2309.11830) | 这个论文介绍了一个用于LLMs的中文Prompt Attack数据集，该数据集旨在评估防御提示攻击的能力。 |
| [^7] | [A Function Interpretation Benchmark for Evaluating Interpretability Methods.](http://arxiv.org/abs/2309.03886) | 本文介绍了一个用于评估自动解释性方法的基准套件，该套件包括了类似于传统系统组件的函数。 |
| [^8] | [On the Trustworthiness Landscape of State-of-the-art Generative Models: A Comprehensive Survey.](http://arxiv.org/abs/2307.16680) | 本文综合调查了大规模生成模型的可信度问题，涵盖了隐私、安全、公平性和责任等多个维度，并提出了实际建议和未来发展方向。 |
| [^9] | [Margin Maximization in Attention Mechanism.](http://arxiv.org/abs/2306.13596) | 这篇论文证明了，在softmax-attention模型中，通过在p或等价的W上运行梯度下降，可以收敛到一个最大边缘解，这将局部最优的标记与非最优的标记分隔开。这明确地将注意力机制形式化为标记分离机制。 |
| [^10] | [ALGO: Synthesizing Algorithmic Programs with Generated Oracle Verifiers.](http://arxiv.org/abs/2305.14591) | ALGO框架使用由LLM生成的神谕指导创造和验证算法程序，以提高现有代码生成模型的算法问题解决能力。 |
| [^11] | [Narrative XL: A Large-scale Dataset For Long-Term Memory Models.](http://arxiv.org/abs/2305.13877) | 本研究提出了一个新的用于长期记忆模型的大规模自然数据集，以帮助改进现有的大型语言模型。数据集由 GPT 3.5 生成，摘要包括来自 Project Gutenberg 的 1500 本书中每个场景的总结，以及配套的阅读理解问题。 |
| [^12] | [Towards Responsible AI in the Era of ChatGPT: A Reference Architecture for Designing Foundation Model-based AI Systems.](http://arxiv.org/abs/2304.11090) | 本文提出了一个以模式为导向的负责任AI-by-design参考架构，用于设计基于基础模型的AI系统，重点关注可解释性、公平性、安全性和鲁棒性等关键设计元素。 |
| [^13] | [SEAM: An Integrated Activation-Coupled Model of Sentence Processing and Eye Movements in Reading.](http://arxiv.org/abs/2303.05221) | SEAM是一种集成了眼动控制和句子处理的模型，为实现阅读中自然语言理解的完整数学模型迈出了重要一步。 |

# 详细

[^1]: AI反馈促进的质量-多样性算法

    Quality-Diversity through AI Feedback. (arXiv:2310.13032v1 [cs.CL])

    [http://arxiv.org/abs/2310.13032](http://arxiv.org/abs/2310.13032)

    基于AI反馈的质量-多样性（QDAIF）算法利用语言模型来生成和评估创造性写作，比传统算法更广泛地覆盖高质量样本的搜索空间。

    

    在许多文本生成问题中，用户可能不仅偏好单一回复，而是希望得到多样性的高质量输出以供选择。质量-多样性（QD）搜索算法旨在通过不断改进和多样化候选人群来实现这一目标。然而，QD在创作性写作等质性领域的应用受到算法指定质量和多样性度量的困难的限制。有趣的是，最近语言模型（LMs）的发展使得通过AI反馈指导搜索成为可能，其中LMs在自然语言中被提示来评估文本的质性方面。借助这一进展，我们引入了通过AI反馈实现的质量-多样性算法（QDAIF），其中进化算法应用LMs来生成变异并评估候选文本的质量和多样性。在创作性写作领域的评估中，与非QDAIF算法相比，QDAIF更广泛地覆盖高质量样本的指定搜索空间。

    In many text-generation problems, users may prefer not only a single response, but a diverse range of high-quality outputs from which to choose. Quality-diversity (QD) search algorithms aim at such outcomes, by continually improving and diversifying a population of candidates. However, the applicability of QD to qualitative domains, like creative writing, has been limited by the difficulty of algorithmically specifying measures of quality and diversity. Interestingly, recent developments in language models (LMs) have enabled guiding search through AI feedback, wherein LMs are prompted in natural language to evaluate qualitative aspects of text. Leveraging this development, we introduce Quality-Diversity through AI Feedback (QDAIF), wherein an evolutionary algorithm applies LMs to both generate variation and evaluate the quality and diversity of candidate text. When assessed on creative writing domains, QDAIF covers more of a specified search space with high-quality samples than do non-
    
[^2]: 在上下文中测量逐点可用信息

    Measuring Pointwise $\mathcal{V}$-Usable Information In-Context-ly. (arXiv:2310.12300v1 [cs.CL])

    [http://arxiv.org/abs/2310.12300](http://arxiv.org/abs/2310.12300)

    这项研究适用于上下文学习范式，通过调整逐点可用信息度量指标为适用于上下文的版本，将其命名为上下文PVI，并证明了上下文PVI的可靠性和稳定性。

    

    在上下文学习（ICL）是一种新的学习范式，随着大型语言模型的发展而受到青睐。本文将最近提出的难度度量指标逐点可用信息（PVI）调整为适用于上下文的版本（上下文PVI）。与原始PVI相比，上下文PVI更高效，因为它只需少量示例并且不需要微调。我们进行了全面的实证分析以评估上下文PVI的可靠性。我们的研究结果表明，上下文PVI的估计值表现出类似于原始PVI的特征。具体针对上下文环境，我们展示了上下文PVI的估计值在不同示例选取和拍摄次数下保持一致。在不同的示例选取中，上下文PVI的估计值的方差是微不足道的，这表明上下文PVI是稳定的。此外，我们演示了如何利用上下文PVI来识别挑战。

    In-context learning (ICL) is a new learning paradigm that has gained popularity along with the development of large language models. In this work, we adapt a recently proposed hardness metric, pointwise $\mathcal{V}$-usable information (PVI), to an in-context version (in-context PVI). Compared to the original PVI, in-context PVI is more efficient in that it requires only a few exemplars and does not require fine-tuning. We conducted a comprehensive empirical analysis to evaluate the reliability of in-context PVI. Our findings indicate that in-context PVI estimates exhibit similar characteristics to the original PVI. Specific to the in-context setting, we show that in-context PVI estimates remain consistent across different exemplar selections and numbers of shots. The variance of in-context PVI estimates across different exemplar selections is insignificant, which suggests that in-context PVI are stable. Furthermore, we demonstrate how in-context PVI can be employed to identify challen
    
[^3]: 会话式健康代理：个性化的基于LLM的代理框架

    Conversational Health Agents: A Personalized LLM-Powered Agent Framework. (arXiv:2310.02374v1 [cs.CL])

    [http://arxiv.org/abs/2310.02374](http://arxiv.org/abs/2310.02374)

    该论文介绍了一个基于LLM的会话式健康代理框架，旨在为代理赋予批判性思维、知识获取和问题解决能力，实现个性化的健康护理服务。该框架能够无缝集成医疗工具，实现多语言和多模态对话，并与多种用户数据分析工具连接。

    

    会话式健康代理（CHAs）是一种互动系统，旨在通过进行共情对话和处理多模态数据来增强个人健康护理服务。当前的CHAs，特别是那些利用大型语言模型（LLMs）的系统，主要关注对话，但往往缺乏全面的代理能力。这包括从可穿戴设备、全天候数据收集源和电子健康记录中获取个人用户的健康数据的能力，以及整合最新发布的健康见解，并与已建立的多模态数据分析工具连接。我们正在开发一个框架，通过赋予CHAs批判性思维、知识获取和问题解决能力来增强它们的能力。我们的CHA平台由LLMs驱动，无缝集成了医疗工具，实现了多语言和多模态对话，并与各种用户数据分析工具进行接口。我们展示了其在处理复杂医疗任务方面的熟练性。

    Conversational Health Agents (CHAs) are interactive systems designed to enhance personal healthcare services by engaging in empathetic conversations and processing multimodal data. While current CHAs, especially those utilizing Large Language Models (LLMs), primarily focus on conversation, they often lack comprehensive agent capabilities. This includes the ability to access personal user health data from wearables, 24/7 data collection sources, and electronic health records, as well as integrating the latest published health insights and connecting with established multimodal data analysis tools. We are developing a framework to empower CHAs by equipping them with critical thinking, knowledge acquisition, and problem-solving abilities. Our CHA platform, powered by LLMs, seamlessly integrates healthcare tools, enables multilingual and multimodal conversations, and interfaces with a variety of user data analysis tools. We illustrate its proficiency in handling complex healthcare tasks, s
    
[^4]: Spider4SPARQL：用于评估知识图谱问答系统的复杂基准

    Spider4SPARQL: A Complex Benchmark for Evaluating Knowledge Graph Question Answering Systems. (arXiv:2309.16248v1 [cs.CL])

    [http://arxiv.org/abs/2309.16248](http://arxiv.org/abs/2309.16248)

    Spider4SPARQL是一个新的SPARQL基准数据集，其中包含大量手工生成的NL问题和独特、新颖、复杂的SPARQL查询，旨在评估知识图谱问答系统。

    

    随着大型语言模型（LLM）数量和可用性的增加，提供大型和真实的基准用于评估知识图谱问答（KBQA）系统变得越来越重要。到目前为止，大部分基准依赖基于模式的SPARQL查询生成方法。随后的自然语言（NL）问题生成通过众包或其他自动化方法进行，如基于规则的改写或NL问题模板。虽然其中一些数据集规模相当大，但它们的缺点在于基于模式的生成方法，并不总是能够很好地推广到真实世界环境中人们提出的模糊且语言多样的问题。本文介绍了Spider4SPARQL - 一个新的SPARQL基准数据集，包含9,693个先前存在的手工生成的NL问题和4,721个唯一、新颖且复杂的SPARQL查询，复杂性各不相同。

    With the recent spike in the number and availability of Large Language Models (LLMs), it has become increasingly important to provide large and realistic benchmarks for evaluating Knowledge Graph Question Answering (KBQA) systems. So far the majority of benchmarks rely on pattern-based SPARQL query generation approaches. The subsequent natural language (NL) question generation is conducted through crowdsourcing or other automated methods, such as rule-based paraphrasing or NL question templates. Although some of these datasets are of considerable size, their pitfall lies in their pattern-based generation approaches, which do not always generalize well to the vague and linguistically diverse questions asked by humans in real-world contexts.  In this paper, we introduce Spider4SPARQL - a new SPARQL benchmark dataset featuring 9,693 previously existing manually generated NL questions and 4,721 unique, novel, and complex SPARQL queries of varying complexity. In addition to the NL/SPARQL pa
    
[^5]: 通过稳健对齐的LLM抵御对齐破坏攻击

    Defending Against Alignment-Breaking Attacks via Robustly Aligned LLM. (arXiv:2309.14348v1 [cs.CL])

    [http://arxiv.org/abs/2309.14348](http://arxiv.org/abs/2309.14348)

    本文提出了一种稳健对齐的LLM（RA-LLM），用于防御可能发生的对齐破坏攻击。RA-LLM可以直接在现有的对齐LLM上构建，并通过稳健的对齐检查函数来确保其有效性。

    

    最近，大型语言模型（LLMs）取得了显著的进展，并在各个领域得到广泛应用。不幸的是，人们越来越担心LLMs可能被滥用来生成有害或恶意内容。尽管有一系列的研究专注于对齐LLMs与人类价值观，并防止它们生成不适当的内容，但这些对齐通常是脆弱的，并且可以通过对抗优化或手工构建的越狱提示来绕过。在这项工作中，我们介绍了一种稳健对齐的LLM（RA-LLM），以防范潜在的对齐破坏攻击。RA-LLM可以直接构建在现有的对齐LLM上，通过具有稳健对齐检查功能的方法，而无需对原始LLM进行任何昂贵的重新训练或微调。此外，我们还通过理论分析验证了RA-LLM在防御对齐破坏攻击方面的有效性。通过现实世界的实验，

    Recently, Large Language Models (LLMs) have made significant advancements and are now widely used across various domains. Unfortunately, there has been a rising concern that LLMs can be misused to generate harmful or malicious content. Though a line of research has focused on aligning LLMs with human values and preventing them from producing inappropriate content, such alignments are usually vulnerable and can be bypassed by alignment-breaking attacks via adversarially optimized or handcrafted jailbreaking prompts. In this work, we introduce a Robustly Aligned LLM (RA-LLM) to defend against potential alignment-breaking attacks. RA-LLM can be directly constructed upon an existing aligned LLM with a robust alignment checking function, without requiring any expensive retraining or fine-tuning process of the original LLM. Furthermore, we also provide a theoretical analysis for RA-LLM to verify its effectiveness in defending against alignment-breaking attacks. Through real-world experiments
    
[^6]: 一个用于具有恶意内容的LLMs的中文提示攻击数据集

    A Chinese Prompt Attack Dataset for LLMs with Evil Content. (arXiv:2309.11830v1 [cs.CL])

    [http://arxiv.org/abs/2309.11830](http://arxiv.org/abs/2309.11830)

    这个论文介绍了一个用于LLMs的中文Prompt Attack数据集，该数据集旨在评估防御提示攻击的能力。

    

    大型语言模型（LLMs）在文本理解和生成方面具有重要优点。然而，LLMs在应用中存在生成有害内容的风险，尤其是在使用Prompt Attack等黑盒攻击方法时。研究人员对LLMs的Prompt Attack和Defense很感兴趣，但目前没有公开可用的数据集来评估防御Prompt Attack的能力。在本文中，我们介绍了一个用于LLMs的中文Prompt Attack数据集，称为CPAD。我们的提示旨在引导LLMs生成具有多个精心设计的提示攻击方法和广泛关注的攻击内容的意外输出。与以前涉及安全估计的数据集不同，我们构建的提示考虑了三个维度：内容、攻击方法和目标，因此可以轻松评估响应。

    Large Language Models (LLMs) present significant priority in text understanding and generation. However, LLMs suffer from the risk of generating harmful contents especially while being employed to applications. There are several black-box attack methods, such as Prompt Attack, which can change the behaviour of LLMs and induce LLMs to generate unexpected answers with harmful contents. Researchers are interested in Prompt Attack and Defense with LLMs, while there is no publicly available dataset to evaluate the abilities of defending prompt attack. In this paper, we introduce a Chinese Prompt Attack Dataset for LLMs, called CPAD. Our prompts aim to induce LLMs to generate unexpected outputs with several carefully designed prompt attack approaches and widely concerned attacking contents. Different from previous datasets involving safety estimation, We construct the prompts considering three dimensions: contents, attacking methods and goals, thus the responses can be easily evaluated and a
    
[^7]: 一个用于评估解释性方法的功能解释基准

    A Function Interpretation Benchmark for Evaluating Interpretability Methods. (arXiv:2309.03886v1 [cs.CL])

    [http://arxiv.org/abs/2309.03886](http://arxiv.org/abs/2309.03886)

    本文介绍了一个用于评估自动解释性方法的基准套件，该套件包括了类似于传统系统组件的函数。

    

    使用人类可读的描述标记神经网络子模块对于许多下游任务非常有用：这些描述可以暴露失败、引导干预，甚至可以解释重要的模型行为。到目前为止，大多数基于机械原理的已训练网络描述都涉及到小模型、狭义现象，并且需要大量人力。在不断增加的模型大小和复杂性中标记出所有人可解释的子计算几乎肯定需要能够自动生成和验证描述的工具。最近，利用学习模型进行标记的技术开始受到关注，但评估其有效性的方法有限且临时。我们应该如何验证和比较开放式标记工具？本文介绍了FIND（函数解释和描述），一个用于评估自动解释方法构建模块的基准套件。FIND包含了类似于传统系统的组件的函数。

    Labeling neural network submodules with human-legible descriptions is useful for many downstream tasks: such descriptions can surface failures, guide interventions, and perhaps even explain important model behaviors. To date, most mechanistic descriptions of trained networks have involved small models, narrowly delimited phenomena, and large amounts of human labor. Labeling all human-interpretable sub-computations in models of increasing size and complexity will almost certainly require tools that can generate and validate descriptions automatically. Recently, techniques that use learned models in-the-loop for labeling have begun to gain traction, but methods for evaluating their efficacy are limited and ad-hoc. How should we validate and compare open-ended labeling tools? This paper introduces FIND (Function INterpretation and Description), a benchmark suite for evaluating the building blocks of automated interpretability methods. FIND contains functions that resemble components of tr
    
[^8]: 关于最先进生成模型的可信度景观：一项综合调查

    On the Trustworthiness Landscape of State-of-the-art Generative Models: A Comprehensive Survey. (arXiv:2307.16680v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2307.16680](http://arxiv.org/abs/2307.16680)

    本文综合调查了大规模生成模型的可信度问题，涵盖了隐私、安全、公平性和责任等多个维度，并提出了实际建议和未来发展方向。

    

    扩散模型和大规模语言模型已经成为领先的生成模型，并对人类生活的各个方面产生了革命性的影响。然而，这些模型的实际应用也暴露出固有的风险，突显了它们的双重性质，并引发了对它们可信度的担忧。尽管有大量关于这个主题的文献，但针对大规模生成模型及其可信度的综合调查仍然很少见。为了弥补这一空白，本文调查了涉及这些模型的长期和新兴威胁，涵盖了隐私、安全、公平和责任这四个基本维度。通过这种方式，我们构建了一张详尽的地图，概述了这些模型的可信度，并提供了实际建议和未来的发展方向。这些努力对于促进这些模型的可信度部署至关重要。

    Diffusion models and large language models have emerged as leading-edge generative models and have sparked a revolutionary impact on various aspects of human life. However, the practical implementation of these models has also exposed inherent risks, highlighting their dual nature and raising concerns regarding their trustworthiness. Despite the abundance of literature on this subject, a comprehensive survey specifically delving into the intersection of large-scale generative models and their trustworthiness remains largely absent. To bridge this gap, This paper investigates both the long-standing and emerging threats associated with these models across four fundamental dimensions: privacy, security, fairness, and responsibility. In this way, we construct an extensive map outlining the trustworthiness of these models, while also providing practical recommendations and identifying future directions. These efforts are crucial for promoting the trustworthy deployment of these models, ulti
    
[^9]: 注意力机制中的边缘最大化

    Margin Maximization in Attention Mechanism. (arXiv:2306.13596v1 [cs.LG])

    [http://arxiv.org/abs/2306.13596](http://arxiv.org/abs/2306.13596)

    这篇论文证明了，在softmax-attention模型中，通过在p或等价的W上运行梯度下降，可以收敛到一个最大边缘解，这将局部最优的标记与非最优的标记分隔开。这明确地将注意力机制形式化为标记分离机制。

    

    注意力机制是Transformer架构的核心组件，也是大型语言模型取得惊人成功的原因之一。然而，注意力机制背后的理论原则尚不清楚，特别是它的非凸优化动力学。本文探讨了开创性的softmax-attention模型$f(\boldsymbol{X})=\langle \boldsymbol{Xv}, \texttt{softmax}(\boldsymbol{XWp})\rangle$，其中$\boldsymbol{X}$是标记序列，$(\boldsymbol{v},\boldsymbol{W},\boldsymbol{p})$是可调参数。我们证明了在$\boldsymbol{p}$或等价的$\boldsymbol{W}$上运行梯度下降会沿着方向收敛到分隔“局部最优”标记和“非最优”标记的最大边缘解。这明确地形式化了注意力作为一种标记分离机制。值得注意的是，我们的结果适用于一般数据，并使用嵌入$\boldsymbol{Xv}$和$\texttt{softmax}(\boldsymbol{XWp})$精细地表征标记的“最优性”。

    Attention mechanism is a central component of the transformer architecture which led to the phenomenal success of large language models. However, the theoretical principles underlying the attention mechanism are poorly understood, especially its nonconvex optimization dynamics. In this work, we explore the seminal softmax-attention model $f(\boldsymbol{X})=\langle \boldsymbol{Xv}, \texttt{softmax}(\boldsymbol{XWp})\rangle$, where, $\boldsymbol{X}$ is the token sequence and $(\boldsymbol{v},\boldsymbol{W},\boldsymbol{p})$ are tunable parameters. We prove that running gradient descent on $\boldsymbol{p}$, or equivalently $\boldsymbol{W}$, converges in direction to a max-margin solution that separates $\textit{locally-optimal}$ tokens from non-optimal ones. This clearly formalizes attention as a token separation mechanism. Remarkably, our results are applicable to general data and precisely characterize $\textit{optimality}$ of tokens in terms of the value embeddings $\boldsymbol{Xv}$ and
    
[^10]: ALGO：使用生成的神谕验证程序的合成算法程序

    ALGO: Synthesizing Algorithmic Programs with Generated Oracle Verifiers. (arXiv:2305.14591v1 [cs.CL])

    [http://arxiv.org/abs/2305.14591](http://arxiv.org/abs/2305.14591)

    ALGO框架使用由LLM生成的神谕指导创造和验证算法程序，以提高现有代码生成模型的算法问题解决能力。

    

    大型语言模型(Large language models, LLMs)在实现代码的功能描述方面表现出色，但在需要确定适当算法的算法问题上亟需提升。此外，LLM生成的程序缺乏保证正确性并需要人工验证。为了解决这些挑战，我们提出了ALGO框架，该框架使用由LLM生成的神谕指导创造和验证算法程序。ALGO首先通过促使LLM枚举相关变量的所有组合来生成具有可能的正确性但可能较慢的参考神谕。然后，利用该神谕指导任意搜索策略来探索算法空间并验证合成的算法。我们的研究表明，LLM生成的神谕在88%的情况下是正确的。使用这些神谕作为验证程序，ALGO可以以模型无关的方式与任何现有的代码生成模型集成，以提高其算法问题解决能力。

    Large language models (LLMs) excel at implementing code from functionality descriptions, but struggle with algorithmic problems that require not only implementation but also identification of the suitable algorithm. Moreover, LLM-generated programs lack guaranteed correctness and require human verification. To address these challenges, we propose ALGO, a framework that synthesizes Algorithmic programs with LLM-Generated Oracles to guide the creation and verify their correctness. ALGO first generates a probably correct but possibly slow reference oracle by prompting an LLM to exhaustively enumerate all the combinations of relevant variables. This oracle is then utilized to guide an arbitrary search strategy in exploring the algorithm space and to verify the algorithms synthesized. Our study shows that the LLM-generated oracles are correct for 88% of the cases. With the oracles as verifiers, ALGO can be integrated with any existing code generation model in a model-agnostic manner to enha
    
[^11]: Narrative XL: 一个用于长期记忆模型的大规模数据集

    Narrative XL: A Large-scale Dataset For Long-Term Memory Models. (arXiv:2305.13877v1 [cs.CL])

    [http://arxiv.org/abs/2305.13877](http://arxiv.org/abs/2305.13877)

    本研究提出了一个新的用于长期记忆模型的大规模自然数据集，以帮助改进现有的大型语言模型。数据集由 GPT 3.5 生成，摘要包括来自 Project Gutenberg 的 1500 本书中每个场景的总结，以及配套的阅读理解问题。

    

    虽然大多数大型语言模型取得了巨大的成功，但它们缺乏任何长期记忆机制，这限制了它们的应用。要克服这一限制，不仅需要对典型的变压器架构或训练程序进行更改，还需要一个可以训练和评估这些新模型的数据集。我们认为现有的资源缺少一些关键属性，目前没有足够规模的自然数据集来训练（而不仅仅是评估）长期记忆语言模型。然后，我们提出了利用短期记忆语言模型的进展来创建这样一个数据集的解决方案。使用 GPT 3.5，我们总结了 Project Gutenberg 中 1500 本手工筛选的书籍中的每个场景，每本书得到大约 150 个场景级别的摘要。然后，我们创建了一些阅读理解问题，包括三种类型的多项选择场景识别问题，以及...

    Despite their tremendous successes, most large language models do not have any long-term memory mechanisms, which restricts their applications. Overcoming this limitation would not only require changes to the typical transformer architectures or training procedures, but also a dataset on which these new models could be trained and evaluated. We argue that existing resources lack a few key properties, and that at present, there are no naturalistic datasets of sufficient scale to train (and not only evaluate) long-term memory language models. We then present our solution that capitalizes on the advances in short-term memory language models to create such a dataset. Using GPT 3.5, we summarized each scene in 1500 hand-curated books from Project Gutenberg, which resulted in approximately 150 scene-level summaries per book. We then created a number of reading comprehension questions based on these summaries, including three types of multiple-choice scene recognition questions, as well as fr
    
[^12]: 在ChatGPT时代迈向负责任的人工智能：用于设计基于基础模型的AI系统的参考架构

    Towards Responsible AI in the Era of ChatGPT: A Reference Architecture for Designing Foundation Model-based AI Systems. (arXiv:2304.11090v1 [cs.CL])

    [http://arxiv.org/abs/2304.11090](http://arxiv.org/abs/2304.11090)

    本文提出了一个以模式为导向的负责任AI-by-design参考架构，用于设计基于基础模型的AI系统，重点关注可解释性、公平性、安全性和鲁棒性等关键设计元素。

    

    ChatGPT、Bard和其他大型语言模型(LLM)聊天机器人的推出在全球范围内引起了巨大关注。基础模型将成为未来大多数AI系统的基础构建块的趋势正在增长。然而，将基础模型纳入AI系统引发了对负责任AI的重大关注，这是由于其黑匣子性质和快速发展的超级智能引起的。此外，基础模型的增长能力最终可能会吞噬AI系统的其他组件，引入架构设计中的运动边界和接口演变挑战。为了应对这些挑战，本文提出了一种以模式为导向的负责任AI-by-design参考架构，用于设计基于基础模型的AI系统。特别地，本文首先呈现了基于基础模型的AI系统在架构演进方面的发展，从"基础模型作为连接器"到"基础模型作为单片机核"。然后，它提出了一个参考架构，包括五个类别的模式，重点关注关键设计元素，例如可解释性、公平性、安全性和鲁棒性。所提出的参考架构为设计负责任的基础模型的AI系统提供了系统化和透明的方法。

    The release of ChatGPT, Bard, and other large language model (LLM)-based chatbots has drawn huge attention on foundations models worldwide. There is a growing trend that foundation models will serve as the fundamental building blocks for most of the future AI systems. However, incorporating foundation models in AI systems raises significant concerns about responsible AI due to their black box nature and rapidly advancing super-intelligence. Additionally, the foundation model's growing capabilities can eventually absorb the other components of AI systems, introducing the moving boundary and interface evolution challenges in architecture design. To address these challenges, this paper proposes a pattern-oriented responsible-AI-by-design reference architecture for designing foundation model-based AI systems. Specially, the paper first presents an architecture evolution of AI systems in the era of foundation models, from "foundation-model-as-a-connector" to "foundation-model-as-a-monolithi
    
[^13]: SEAM:一种集成了句子处理与阅读中眼动的激活耦合模型

    SEAM: An Integrated Activation-Coupled Model of Sentence Processing and Eye Movements in Reading. (arXiv:2303.05221v2 [q-bio.NC] UPDATED)

    [http://arxiv.org/abs/2303.05221](http://arxiv.org/abs/2303.05221)

    SEAM是一种集成了眼动控制和句子处理的模型，为实现阅读中自然语言理解的完整数学模型迈出了重要一步。

    

    阅读中的眼动控制模型通常集中在视觉、注意、词汇和运动过程，但忽略了词汇后处理的语言处理。相比之下，句子理解过程的模型通常只关注词汇后处理的语言过程。我们提出了一种将这两种研究线索结合起来的模型，即整合眼动控制和句子处理。开发这样一个整合模型具有极大的挑战性和计算复杂性，但这样的整合是朝着完整的自然语言理解数学模型迈出的重要一步。我们将眼动控制模型SWIFT（Seelig等人，2020）与Lewis和Vasishth句子处理模型的关键组成部分（Lewis＆Vasishth，2005）结合在一起。这种整合首次变得可能，部分原因是因为。。

    Models of eye-movement control during reading, developed largely within psychology, usually focus on visual, attentional, lexical, and motor processes but neglect post-lexical language processing; by contrast, models of sentence comprehension processes, developed largely within psycholinguistics, generally focus only on post-lexical language processes. We present a model that combines these two research threads, by integrating eye-movement control and sentence processing. Developing such an integrated model is extremely challenging and computationally demanding, but such an integration is an important step toward complete mathematical models of natural language comprehension in reading. We combine the SWIFT model of eye-movement control (Seelig et al., 2020, doi:10.1016/j.jmp.2019.102313) with key components of the Lewis and Vasishth sentence processing model (Lewis & Vasishth, 2005, doi:10.1207/s15516709cog0000_25). This integration becomes possible, for the first time, due in part to
    

