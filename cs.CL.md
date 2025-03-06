# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [StableToolBench: Towards Stable Large-Scale Benchmarking on Tool Learning of Large Language Models](https://arxiv.org/abs/2403.07714) | StableToolBench提出了一种虚拟API服务器和稳定评估系统，通过缓存系统、API模拟器和稳定评估设计，解决了大语言模型利用外部工具的稳定大规模基准测试问题。 |
| [^2] | [ExpertPrompting: Instructing Large Language Models to be Distinguished Experts.](http://arxiv.org/abs/2305.14688) | 本论文提出了“专家提示”技术，用于训练大型语言模型成为杰出的专家。该方法使用上下文学习自动生成每个指令的详细和定制的专家身份描述，并要求模型根据这些提示提供答案。基于这种技术，本文提出了一个新的开源聊天助手ExpertLLaMA，该助手在评估中表现出高质量的数据和96％的ChatGPT能力。 |

# 详细

[^1]: StableToolBench：面向大规模稳定基准测试的工具学习大语言模型

    StableToolBench: Towards Stable Large-Scale Benchmarking on Tool Learning of Large Language Models

    [https://arxiv.org/abs/2403.07714](https://arxiv.org/abs/2403.07714)

    StableToolBench提出了一种虚拟API服务器和稳定评估系统，通过缓存系统、API模拟器和稳定评估设计，解决了大语言模型利用外部工具的稳定大规模基准测试问题。

    

    大语言模型（LLMs）近年来取得了显著进展，促使人们探索工具学习，将LLMs与外部工具整合以解决各种现实挑战。评估LLMs利用工具的能力需要大规模且稳定的基准测试。我们介绍了由ToolBench演变而来的StableToolBench，提出了一个虚拟API服务器和稳定评估系统。虚拟API服务器包含缓存系统和API模拟器，互补减轻API状态变化。同时，稳定的评估系统使用GPT-4作为自动评估器设计可解决的通过率和胜率，以消除评估过程中的随机性。实验结果证明

    arXiv:2403.07714v1 Announce Type: new  Abstract: Large Language Models (LLMs) have witnessed remarkable advancements in recent years, prompting the exploration of tool learning, which integrates LLMs with external tools to address diverse real-world challenges. Assessing the capability of LLMs to utilise tools necessitates large-scale and stable benchmarks. However, previous works relied on either hand-crafted online tools with limited scale, or large-scale real online APIs suffering from instability of API status. To address this problem, we introduce StableToolBench, a benchmark evolving from ToolBench, proposing a virtual API server and stable evaluation system. The virtual API server contains a caching system and API simulators which are complementary to alleviate the change in API status. Meanwhile, the stable evaluation system designs solvable pass and win rates using GPT-4 as the automatic evaluator to eliminate the randomness during evaluation. Experimental results demonstrate 
    
[^2]: 专家提示：指导大型语言模型成为杰出的专家

    ExpertPrompting: Instructing Large Language Models to be Distinguished Experts. (arXiv:2305.14688v1 [cs.CL])

    [http://arxiv.org/abs/2305.14688](http://arxiv.org/abs/2305.14688)

    本论文提出了“专家提示”技术，用于训练大型语言模型成为杰出的专家。该方法使用上下文学习自动生成每个指令的详细和定制的专家身份描述，并要求模型根据这些提示提供答案。基于这种技术，本文提出了一个新的开源聊天助手ExpertLLaMA，该助手在评估中表现出高质量的数据和96％的ChatGPT能力。

    

    如果以适当的提示方式进行处理，对齐的大型语言模型（LLM）的回答质量可以大大提高。在本文中，我们提出了专家提示，以引发LLMs作为杰出专家回答的潜力。我们首先利用上下文学习自动生成每个特定指令的详细和定制的专家身份描述，然后要求LLMs根据这种代理人背景提供答案。基于这种增强的提示策略，我们使用GPT-3.5生成了一组新的指令遵循数据，并训练了一个竞争性的开源聊天助手ExpertLLaMA。我们采用基于GPT4的评估显示：1）专家数据的质量显著高于普通答案，2）ExpertLLaMA胜过现有的开源对手，实现了ChatGPT能力的96％。所有数据和ExpertLLaMA模型将在\url{https://github.com/OFA-Sys/Exp}上公开。

    The answering quality of an aligned large language model (LLM) can be drastically improved if treated with proper crafting of prompts. In this paper, we propose ExpertPrompting to elicit the potential of LLMs to answer as distinguished experts. We first utilize In-Context Learning to automatically synthesize detailed and customized descriptions of the expert identity for each specific instruction, and then ask LLMs to provide answer conditioned on such agent background. Based on this augmented prompting strategy, we produce a new set of instruction-following data using GPT-3.5, and train a competitive open-source chat assistant called ExpertLLaMA. We employ GPT4-based evaluation to show that 1) the expert data is of significantly higher quality than vanilla answers, and 2) ExpertLLaMA outperforms existing open-source opponents and achieves 96\% of the original ChatGPT's capability. All data and the ExpertLLaMA model will be made publicly available at \url{https://github.com/OFA-Sys/Exp
    

