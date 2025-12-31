# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Application-Driven Innovation in Machine Learning](https://arxiv.org/abs/2403.17381) | 应用驱动研究在机器学习领域具有重要影响，可以与方法驱动研究有益地协同，但目前审查、招聘和教学实践往往阻碍了这种创新。 |
| [^2] | [Learnable WSN Deployment of Evidential Collaborative Sensing Model](https://arxiv.org/abs/2403.15728) | 本文提出了一种通过协同感知模型和证据理论框架下的组合规则，来提高WSNs检测能力的学习型传感器部署网络（LSDNet）。 |
| [^3] | [Never-Ending Embodied Robot Learning](https://arxiv.org/abs/2403.00336) | 提出了一种具身机器人学习代理NBCagent，通过技能特定的演化规划器和技能共享的语义渲染模块，实现从视觉观测中连续学习新的机器人操作技能知识。 |
| [^4] | [TPTU: Large Language Model-based AI Agents for Task Planning and Tool Usage.](http://arxiv.org/abs/2308.03427) | 基于大型语言模型的AI代理用于任务规划和工具使用。我们提出了一个结构框架，设计了两种代理来执行推理过程，实例化了框架，并评估了它们的任务规划和工具使用能力。 |
| [^5] | [Prompt Injection attack against LLM-integrated Applications.](http://arxiv.org/abs/2306.05499) | 本研究分析了LLM集成应用中的提示注入攻击的复杂性和影响，提出了一种新颖的黑盒提示注入攻击技术HouYi，并揭示了应用程序提示机制中以前未知和严重低估的漏洞。我们的研究呼吁进一步开发全面的防御措施，以抵御LLM集成应用中的提示注入攻击。 |
| [^6] | [Towards Global Optimality in Cooperative MARL with the Transformation And Distillation Framework.](http://arxiv.org/abs/2207.11143) | 本文研究了采用分散策略的MARL算法在梯度下降优化器下的次最优性，并提出了转化与蒸馏框架，该框架可以将多智能体MDP转化为单智能体MDP以实现分散执行。 |

# 详细

[^1]: 机器学习中的应用驱动创新

    Application-Driven Innovation in Machine Learning

    [https://arxiv.org/abs/2403.17381](https://arxiv.org/abs/2403.17381)

    应用驱动研究在机器学习领域具有重要影响，可以与方法驱动研究有益地协同，但目前审查、招聘和教学实践往往阻碍了这种创新。

    

    随着机器学习应用的不断增长，受特定现实挑战启发的创新算法变得日益重要。这样的工作不仅在应用领域具有重要影响，也在机器学习本身具有重要影响。本文描述了机器学习中应用驱动研究的范式，将其与更标准的方法驱动研究进行了对比。我们阐明了应用驱动机器学习的好处，以及这种方法如何可以与方法驱动工作有益地协同。尽管具有这些好处，我们发现机器学习中的审查、招聘和教学实践往往阻碍了应用驱动创新。我们概述了如何改进这些流程。

    arXiv:2403.17381v1 Announce Type: cross  Abstract: As applications of machine learning proliferate, innovative algorithms inspired by specific real-world challenges have become increasingly important. Such work offers the potential for significant impact not merely in domains of application but also in machine learning itself. In this paper, we describe the paradigm of application-driven research in machine learning, contrasting it with the more standard paradigm of methods-driven research. We illustrate the benefits of application-driven machine learning and how this approach can productively synergize with methods-driven work. Despite these benefits, we find that reviewing, hiring, and teaching practices in machine learning often hold back application-driven innovation. We outline how these processes may be improved.
    
[^2]: 可学习的证据协同感知模型的WSN部署

    Learnable WSN Deployment of Evidential Collaborative Sensing Model

    [https://arxiv.org/abs/2403.15728](https://arxiv.org/abs/2403.15728)

    本文提出了一种通过协同感知模型和证据理论框架下的组合规则，来提高WSNs检测能力的学习型传感器部署网络（LSDNet）。

    

    在无线传感器网络（WSNs）中，覆盖和部署是进行检测任务时最关键的两个问题。但通常来自传感器的检测信息并没有被充分利用和高效整合。本文旨在实现WSN部署的最佳覆盖质量，通过开发一种传感器的协同感知模型，利用证据理论框架下的组合规则得到的协同信息来增强WSNs的检测能力。

    arXiv:2403.15728v1 Announce Type: new  Abstract: In wireless sensor networks (WSNs), coverage and deployment are two most crucial issues when conducting detection tasks. However, the detection information collected from sensors is oftentimes not fully utilized and efficiently integrated. Such sensing model and deployment strategy, thereby, cannot reach the maximum quality of coverage, particularly when the amount of sensors within WSNs expands significantly. In this article, we aim at achieving the optimal coverage quality of WSN deployment. We develop a collaborative sensing model of sensors to enhance detection capabilities of WSNs, by leveraging the collaborative information derived from the combination rule under the framework of evidence theory. In this model, the performance evaluation of evidential fusion systems is adopted as the criterion of the sensor selection. A learnable sensor deployment network (LSDNet) considering both sensor contribution and detection capability, is pr
    
[^3]: 永不停止的具身机器人学习

    Never-Ending Embodied Robot Learning

    [https://arxiv.org/abs/2403.00336](https://arxiv.org/abs/2403.00336)

    提出了一种具身机器人学习代理NBCagent，通过技能特定的演化规划器和技能共享的语义渲染模块，实现从视觉观测中连续学习新的机器人操作技能知识。

    

    依赖于大型语言模型（LLM），具身机器人可以通过强大的泛化能力，从视觉观测中执行复杂的多模态机器人操作任务。然而，大多数视觉行为克隆代理在适应一系列具有挑战性的未见任务时，会遭受操纵性能下降以及技能知识遗忘的困扰。在本研究中，我们通过NBCagent在具身机器人中探讨了上述挑战，这是一种开创性的、以语言为条件的永不停止行为克隆代理，可以不断从特定技能和共享技能属性中学习新的机器人操作技能的观察知识。具体来说，我们建立了一个特定技能不断演化的规划器来进行知识解耦，这可以从潜在和低秩空间中不断向我们的NBCagent代理嵌入新的技能特定知识。与此同时，我们提出了一个技能共享语义渲染模块和一个技能共享表示区分

    arXiv:2403.00336v1 Announce Type: cross  Abstract: Relying on large language models (LLMs), embodied robots could perform complex multimodal robot manipulation tasks from visual observations with powerful generalization ability. However, most visual behavior-cloning agents suffer from manipulation performance degradation and skill knowledge forgetting when adapting into a series of challenging unseen tasks. We here investigate the above challenge with NBCagent in embodied robots, a pioneering language-conditioned Never-ending Behavior-Cloning agent, which can continually learn observation knowledge of novel robot manipulation skills from skill-specific and skill-shared attributes. Specifically, we establish a skill-specific evolving planner to perform knowledge decoupling, which can continually embed novel skill-specific knowledge in our NBCagent agent from latent and low-rank space. Meanwhile, we propose a skill-shared semantics rendering module and a skill-shared representation disti
    
[^4]: TPTU: 基于大型语言模型的AI代理用于任务规划和工具使用

    TPTU: Large Language Model-based AI Agents for Task Planning and Tool Usage. (arXiv:2308.03427v2 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2308.03427](http://arxiv.org/abs/2308.03427)

    基于大型语言模型的AI代理用于任务规划和工具使用。我们提出了一个结构框架，设计了两种代理来执行推理过程，实例化了框架，并评估了它们的任务规划和工具使用能力。

    

    随着自然语言处理的最新进展，大型语言模型（LLMs）已经成为各种实际应用中的强大工具。尽管它们非常强大，但是LLMs的内在生成能力可能不足以处理复杂的任务，这些任务需要结合任务规划和外部工具的使用。在本文中，我们首先提出了一个专门为LLM-based AI Agents量身定制的结构框架，并讨论了处理复杂问题所必需的关键能力。在这个框架内，我们设计了两种不同类型的代理（即一步代理和连续代理）来执行推理过程。随后，我们使用各种LLMs实例化了这个框架，并评估了它们在典型任务中的任务规划和工具使用能力。通过强调关键发现和挑战，我们的目标是为研究人员和实践者提供一个有助于在他们的AI应用中发挥LLMs能力的有用资源。我们的研究强调了

    With recent advancements in natural language processing, Large Language Models (LLMs) have emerged as powerful tools for various real-world applications. Despite their prowess, the intrinsic generative abilities of LLMs may prove insufficient for handling complex tasks which necessitate a combination of task planning and the usage of external tools. In this paper, we first propose a structured framework tailored for LLM-based AI Agents and discuss the crucial capabilities necessary for tackling intricate problems. Within this framework, we design two distinct types of agents (i.e., one-step agent and sequential agent) to execute the inference process. Subsequently, we instantiate the framework using various LLMs and evaluate their Task Planning and Tool Usage (TPTU) abilities on typical tasks. By highlighting key findings and challenges, our goal is to provide a helpful resource for researchers and practitioners to leverage the power of LLMs in their AI applications. Our study emphasiz
    
[^5]: LLM集成应用中的提示注入攻击研究

    Prompt Injection attack against LLM-integrated Applications. (arXiv:2306.05499v1 [cs.CR])

    [http://arxiv.org/abs/2306.05499](http://arxiv.org/abs/2306.05499)

    本研究分析了LLM集成应用中的提示注入攻击的复杂性和影响，提出了一种新颖的黑盒提示注入攻击技术HouYi，并揭示了应用程序提示机制中以前未知和严重低估的漏洞。我们的研究呼吁进一步开发全面的防御措施，以抵御LLM集成应用中的提示注入攻击。

    

    大语言模型(LLM)因其卓越的语言理解和生成能力而在它们周围刺激了一个充满活力的应用生态系统。然而，它们在各种服务中的广泛融合带来了重大的安全风险。本研究将解构实际LLM集成应用中的提示注入攻击的复杂性和影响。最初，我们对十个商业应用程序进行了探索性分析，突出了目前攻击策略在实践中的约束条件。受这些限制的启发，我们随后制定了HouYi，一种新颖的黑盒提示注入攻击技术，它借鉴了传统的Web注入攻击。HouYi分为三个关键元素: 一个无缝集成的预构建提示、一个注入提示诱导上下文分区以及一个恶意载荷，旨在实现攻击目标。利用HouYi，我们揭示了应用程序提示机制中以前未知和严重低估的漏洞，并演示了绕过最先进的检测机制的可行性。我们的研究呼吁进一步研究开发全面的防御措施，以抵御LLM集成应用中的提示注入攻击。

    Large Language Models (LLMs), renowned for their superior proficiency in language comprehension and generation, stimulate a vibrant ecosystem of applications around them. However, their extensive assimilation into various services introduces significant security risks. This study deconstructs the complexities and implications of prompt injection attacks on actual LLM-integrated applications. Initially, we conduct an exploratory analysis on ten commercial applications, highlighting the constraints of current attack strategies in practice. Prompted by these limitations, we subsequently formulate HouYi, a novel black-box prompt injection attack technique, which draws inspiration from traditional web injection attacks. HouYi is compartmentalized into three crucial elements: a seamlessly-incorporated pre-constructed prompt, an injection prompt inducing context partition, and a malicious payload designed to fulfill the attack objectives. Leveraging HouYi, we unveil previously unknown and sev
    
[^6]: 《采用转化与蒸馏框架实现合作MARL全局最优性》

    Towards Global Optimality in Cooperative MARL with the Transformation And Distillation Framework. (arXiv:2207.11143v3 [cs.MA] UPDATED)

    [http://arxiv.org/abs/2207.11143](http://arxiv.org/abs/2207.11143)

    本文研究了采用分散策略的MARL算法在梯度下降优化器下的次最优性，并提出了转化与蒸馏框架，该框架可以将多智能体MDP转化为单智能体MDP以实现分散执行。

    

    在合作多智能体强化学习中，分散执行是一项核心需求。目前，大多数流行的MARL算法采用分散策略来实现分散执行，并使用梯度下降作为优化器。然而，在考虑到优化方法的情况下，这些算法几乎没有任何理论分析，我们发现当梯度下降被选为优化方法时，各种流行的分散策略MARL算法在玩具任务中都是次最优的。本文在理论上分析了两种常见的采用分散策略的算法——多智能体策略梯度方法和值分解方法，证明了它们在使用梯度下降时的次最优性。此外，我们提出了转化与蒸馏（TAD）框架，它将多智能体MDP重新制定为一种具有连续结构的特殊单智能体MDP，并通过蒸馏实现分散执行。

    Decentralized execution is one core demand in cooperative multi-agent reinforcement learning (MARL). Recently, most popular MARL algorithms have adopted decentralized policies to enable decentralized execution and use gradient descent as their optimizer. However, there is hardly any theoretical analysis of these algorithms taking the optimization method into consideration, and we find that various popular MARL algorithms with decentralized policies are suboptimal in toy tasks when gradient descent is chosen as their optimization method. In this paper, we theoretically analyze two common classes of algorithms with decentralized policies -- multi-agent policy gradient methods and value-decomposition methods to prove their suboptimality when gradient descent is used. In addition, we propose the Transformation And Distillation (TAD) framework, which reformulates a multi-agent MDP as a special single-agent MDP with a sequential structure and enables decentralized execution by distilling the
    

