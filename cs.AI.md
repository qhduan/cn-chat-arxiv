# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [LoRA-as-an-Attack! Piercing LLM Safety Under The Share-and-Play Scenario](https://arxiv.org/abs/2403.00108) | LoRA作为攻击者渗透LLM安全，研究探讨了在共享与玩耍场景下可能实现的攻击机会。 |
| [^2] | [Fiddler: CPU-GPU Orchestration for Fast Inference of Mixture-of-Experts Models](https://arxiv.org/abs/2402.07033) | 本文介绍了Fiddler，一种用于Mixture-of-Experts模型的资源高效推断引擎，通过CPU-GPU编排实现最小化数据传输，相比现有方法提高了一个数量级的推断速度。 |
| [^3] | [Omitted Labels in Causality: A Study of Paradoxes](https://arxiv.org/abs/2311.06840) | 本研究探讨了所谓的“遗漏标签上下文”，即训练数据仅限于可能标签的一个子集，并利用悖论展示了在这种上下文中因果推断面临的困难。研究发现在某些情况下，必须使用非可交换的处理组和对照组进行正确的校正。此外，研究还发现了结论网络与社会选择理论之间的有趣联系。 |
| [^4] | [Connecting Large Language Models with Evolutionary Algorithms Yields Powerful Prompt Optimizers.](http://arxiv.org/abs/2309.08532) | 本文提出了一种通过连接大型语言模型和进化算法进行提示优化的框架，名为EvoPrompt。通过利用大型语言模型的语言处理能力和进化算法的优化性能，EvoPrompt可以自动化处理需要连贯和可读性良好的提示，提高大型语言模型的性能。 |

# 详细

[^1]: 将LoRA作为攻击！在Share-and-Play场景下穿透LLM安全

    LoRA-as-an-Attack! Piercing LLM Safety Under The Share-and-Play Scenario

    [https://arxiv.org/abs/2403.00108](https://arxiv.org/abs/2403.00108)

    LoRA作为攻击者渗透LLM安全，研究探讨了在共享与玩耍场景下可能实现的攻击机会。

    

    对LLMs进行微调对于增强其特定任务的性能并确保模型行为与人类偏好保持一致至关重要。在各种微调方法中，LoRA因其效率和易用性而备受推崇，允许最终用户轻松在开源平台上发布和采用轻量的LoRA模块，以定制其模型以适应不同需求。然而，这种方便的共享与玩耍设置打开了新的攻击面，攻击者可以将LoRA作为攻击者，例如背门注入，并广泛分发对抗性LoRA给社区。这可能导致不利的后果。尽管共享LoRA模块存在巨大的潜在风险，但这一方面尚未得到充分探讨。为了填补这一空白，在本研究中，我们深入探讨了在不断增长的共享与玩耍场景中可能做出的攻击机会。具体而言，我们研究了如何将后门注入LoRA模块并深入探讨。

    arXiv:2403.00108v1 Announce Type: cross  Abstract: Fine-tuning LLMs is crucial to enhancing their task-specific performance and ensuring model behaviors are aligned with human preferences. Among various fine-tuning methods, LoRA is popular for its efficiency and ease to use, allowing end-users to easily post and adopt lightweight LoRA modules on open-source platforms to tailor their model for different customization. However, such a handy share-and-play setting opens up new attack surfaces, that the attacker can render LoRA as an attacker, such as backdoor injection, and widely distribute the adversarial LoRA to the community easily. This can result in detrimental outcomes. Despite the huge potential risks of sharing LoRA modules, this aspect however has not been fully explored. To fill the gap, in this study we thoroughly investigate the attack opportunities enabled in the growing share-and-play scenario. Specifically, we study how to inject backdoor into the LoRA module and dive deep
    
[^2]: Fiddler：用于Mixture-of-Experts模型快速推断的CPU-GPU编排

    Fiddler: CPU-GPU Orchestration for Fast Inference of Mixture-of-Experts Models

    [https://arxiv.org/abs/2402.07033](https://arxiv.org/abs/2402.07033)

    本文介绍了Fiddler，一种用于Mixture-of-Experts模型的资源高效推断引擎，通过CPU-GPU编排实现最小化数据传输，相比现有方法提高了一个数量级的推断速度。

    

    基于Mixture-of-Experts（MoE）架构的大型语言模型（LLM）在各种任务上表现出了很好的性能。然而，在资源受限的环境下运行这些模型，即GPU内存资源不丰富的情况下，由于模型规模庞大，存在挑战。现有的将模型权重卸载到CPU内存的系统，由于频繁地在CPU和GPU之间移动数据而导致显著的开销。在本文中，我们提出了Fiddler，一种用于MoE模型的资源高效推断引擎，实现了CPU-GPU编排。Fiddler的核心思想是利用CPU的计算能力来最小化CPU和GPU之间的数据传输。我们的评估结果表明，Fiddler能够在单个具有24GB内存的GPU上运行未压缩的Mixtral-8x7B模型（参数超过90GB），每秒生成超过3个token，相比现有方法提高一个数量级。Fiddler的代码可以公开访问，网址为\url{https://github.com/efeslab/fiddler}

    Large Language Models (LLMs) based on Mixture-of-Experts (MoE) architecture are showing promising performance on various tasks. However, running them on resource-constrained settings, where GPU memory resources are not abundant, is challenging due to huge model sizes. Existing systems that offload model weights to CPU memory suffer from the significant overhead of frequently moving data between CPU and GPU. In this paper, we propose Fiddler, a resource-efficient inference engine with CPU-GPU orchestration for MoE models. The key idea of Fiddler is to use the computation ability of the CPU to minimize the data movement between the CPU and GPU. Our evaluation shows that Fiddler can run the uncompressed Mixtral-8x7B model, which exceeds 90GB in parameters, to generate over $3$ tokens per second on a single GPU with 24GB memory, showing an order of magnitude improvement over existing methods. The code of Fiddler is publicly available at \url{https://github.com/efeslab/fiddler}
    
[^3]: 在因果推断中的遗漏标签: 一项关于悖论的研究

    Omitted Labels in Causality: A Study of Paradoxes

    [https://arxiv.org/abs/2311.06840](https://arxiv.org/abs/2311.06840)

    本研究探讨了所谓的“遗漏标签上下文”，即训练数据仅限于可能标签的一个子集，并利用悖论展示了在这种上下文中因果推断面临的困难。研究发现在某些情况下，必须使用非可交换的处理组和对照组进行正确的校正。此外，研究还发现了结论网络与社会选择理论之间的有趣联系。

    

    我们探讨了我们所称之为“遗漏标签上下文”的概念，即训练数据仅限于可能标签的一个子集。这种设置在专业人士或特定的专注研究中非常普遍。我们利用已广泛研究的悖论（辛普森悖论和康多塞悖论）来说明在遗漏标签上下文中因果推断面临的更普遍困难。与因果推断基本原理相反，我们展示了“正确”的校正有时需要非可交换的处理组和对照组。这些陷阱引导我们研究不同上下文中得出的结论网络和其形成的结构，从而证明了这些网络与社会选择理论的有趣联系。

    We explore what we call ``omitted label contexts,'' in which training data is limited to a subset of the possible labels. This setting is common among specialized human experts or specific focused studies. We lean on well-studied paradoxes (Simpson's and Condorcet) to illustrate the more general difficulties of causal inference in omitted label contexts. Contrary to the fundamental principles on which much of causal inference is built, we show that ``correct'' adjustments sometimes require non-exchangeable treatment and control groups. These pitfalls lead us to the study networks of conclusions drawn from different contexts and the structures the form, proving an interesting connection between these networks and social choice theory.
    
[^4]: 通过进化算法连接大型语言模型与强大的提示优化器

    Connecting Large Language Models with Evolutionary Algorithms Yields Powerful Prompt Optimizers. (arXiv:2309.08532v1 [cs.CL])

    [http://arxiv.org/abs/2309.08532](http://arxiv.org/abs/2309.08532)

    本文提出了一种通过连接大型语言模型和进化算法进行提示优化的框架，名为EvoPrompt。通过利用大型语言模型的语言处理能力和进化算法的优化性能，EvoPrompt可以自动化处理需要连贯和可读性良好的提示，提高大型语言模型的性能。

    

    大型语言模型在各种任务中表现出色，但它们依赖于精心设计的提示，这通常需要大量的人力努力。为了自动化这个过程，本文提出了一种新颖的离散提示优化框架，称为EvoPrompt，它借鉴了进化算法的思想，因为它们表现出良好的性能和快速的收敛性。为了使进化算法能够处理需要连贯并且可读性良好的自然语言表达的离散提示，我们将大型语言模型与进化算法进行了连接。这种方法使我们可以同时利用大型语言模型的强大语言处理能力和进化算法的高效优化性能。具体而言，EvoPrompt在不使用任何梯度或参数的情况下，从一组提示中开始，并基于进化算子通过大型语言模型生成新的提示，根据开发集改进提示的种群。我们对闭源和开源的大型语言模型，包括GPT-3进行提示优化。

    Large Language Models (LLMs) excel in various tasks, but they rely on carefully crafted prompts that often demand substantial human effort. To automate this process, in this paper, we propose a novel framework for discrete prompt optimization, called EvoPrompt, which borrows the idea of evolutionary algorithms (EAs) as they exhibit good performance and fast convergence. To enable EAs to work on discrete prompts, which are natural language expressions that need to be coherent and human-readable, we connect LLMs with EAs. This approach allows us to simultaneously leverage the powerful language processing capabilities of LLMs and the efficient optimization performance of EAs. Specifically, abstaining from any gradients or parameters, EvoPrompt starts from a population of prompts and iteratively generates new prompts with LLMs based on the evolutionary operators, improving the population based on the development set. We optimize prompts for both closed- and open-source LLMs including GPT-3
    

