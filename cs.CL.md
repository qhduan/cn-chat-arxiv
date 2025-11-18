# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Conversational SimulMT: Efficient Simultaneous Translation with Large Language Models](https://arxiv.org/abs/2402.10552) | 通过对话式SimulMT框架，本文提高了基于LLM的SimulMT推理效率，在保持翻译质量的同时实现与专门的SimulMT模型相近的计算延迟。 |
| [^2] | [Simultaneous Machine Translation with Large Language Models.](http://arxiv.org/abs/2309.06706) | 本文研究了使用大型语言模型进行同时机器翻译的可行性，通过引入混合策略，并进行有监督微调，取得了显著的性能改进。 |
| [^3] | [FinGPT: Open-Source Financial Large Language Models.](http://arxiv.org/abs/2306.06031) | FinGPT是一个开源的金融大型语言模型，提供了可访问和透明的资源来开发金融LLMs，其重要性在于自动数据筛选管道和轻量级低秩适应技术。 |

# 详细

[^1]: Conversational SimulMT: 基于大型语言模型的高效同时翻译

    Conversational SimulMT: Efficient Simultaneous Translation with Large Language Models

    [https://arxiv.org/abs/2402.10552](https://arxiv.org/abs/2402.10552)

    通过对话式SimulMT框架，本文提高了基于LLM的SimulMT推理效率，在保持翻译质量的同时实现与专门的SimulMT模型相近的计算延迟。

    

    同声机器翻译（SimulMT）在翻译质量和延迟之间存在挑战性的权衡。最近的研究表明，大型语言模型（LLMs）在SimulMT任务中可以取得很好的表现。然而，这往往是以推理成本和延迟的增加为代价的。本文提出了一种对话式SimulMT框架，通过基于多轮对话的解码来提高基于LLM的SimulMT的推理效率。我们在两个SimulMT基准上使用Llama2-7b-chat进行实验，结果表明LLM在翻译质量上具有优势，同时实现与专门的SimulMT模型相当的计算延迟。

    arXiv:2402.10552v1 Announce Type: new  Abstract: Simultaneous machine translation (SimulMT) presents a challenging trade-off between translation quality and latency. Recent studies have shown that LLMs can achieve good performance in SimulMT tasks. However, this often comes at the expense of high inference cost and latency. In this paper, we propose a conversational SimulMT framework to enhance the inference efficiency of LLM-based SimulMT through multi-turn-dialogue-based decoding. Our experiments with Llama2-7b-chat on two SimulMT benchmarks demonstrate the superiority of LLM in translation quality while achieving comparable computational latency to specialized SimulMT models.
    
[^2]: 使用大型语言模型的同时机器翻译

    Simultaneous Machine Translation with Large Language Models. (arXiv:2309.06706v1 [cs.CL])

    [http://arxiv.org/abs/2309.06706](http://arxiv.org/abs/2309.06706)

    本文研究了使用大型语言模型进行同时机器翻译的可行性，通过引入混合策略，并进行有监督微调，取得了显著的性能改进。

    

    通过对话式交互，大型语言模型 (LLM) 已经展示出解决各种自然语言处理任务的能力。例如，研究表明，LLM可以在高资源语言的离线机器翻译任务中取得竞争性的性能。然而，将LLM应用于同时机器翻译 (SimulMT) 面临许多挑战，包括与不同解码模式产生的训练-推理不匹配问题。本文探索了利用LLM进行SimulMT的可行性。在传统方法的基础上，我们引入了一个简单而有效的混合策略，使LLM能够在不需要额外训练的情况下参与SimulMT。此外，在对全句和前缀句子进行有监督微调后，该模型展示出了显著的性能改进。我们使用MUST-C数据集上的九种语言对进行实验，结果表明LLM可以实现同时机器翻译。

    Large language models (LLM) have demonstrated their abilities to solve various natural language processing tasks through dialogue-based interactions. For instance, research indicates that LLMs can achieve competitive performance in offline machine translation tasks for high-resource languages. However, applying LLMs to simultaneous machine translation (SimulMT) poses many challenges, including issues related to the training-inference mismatch arising from different decoding patterns. In this paper, we explore the feasibility of utilizing LLMs for SimulMT. Building upon conventional approaches, we introduce a simple yet effective mixture policy that enables LLMs to engage in SimulMT without requiring additional training. Furthermore, after Supervised Fine-Tuning (SFT) on a mixture of full and prefix sentences, the model exhibits significant performance improvements. Our experiments, conducted with Llama2-7B-chat on nine language pairs from the MUST-C dataset, demonstrate that LLM can ac
    
[^3]: FinGPT：开源金融大型语言模型

    FinGPT: Open-Source Financial Large Language Models. (arXiv:2306.06031v1 [q-fin.ST])

    [http://arxiv.org/abs/2306.06031](http://arxiv.org/abs/2306.06031)

    FinGPT是一个开源的金融大型语言模型，提供了可访问和透明的资源来开发金融LLMs，其重要性在于自动数据筛选管道和轻量级低秩适应技术。

    

    大型语言模型（LLMs）展示了在各个领域革新自然语言处理任务的潜力，引起了金融领域的浓厚兴趣。获得高质量的金融数据是金融LLMs（FinLLMs）的第一个挑战。在这篇论文中，我们提出了一个针对金融领域的开源大型语言模型FinGPT。与专有模型不同，FinGPT采用数据为中心的方法，为研究人员和从业者提供可访问和透明的资源来开发他们的金融LLMs。我们强调自动数据筛选管道和轻量级低秩适应技术在建立FinGPT中的重要性。此外，我们展示了几个潜在的应用作为用户的基础，如机器顾问、算法交易和论 。

    Large language models (LLMs) have shown the potential of revolutionizing natural language processing tasks in diverse domains, sparking great interest in finance. Accessing high-quality financial data is the first challenge for financial LLMs (FinLLMs). While proprietary models like BloombergGPT have taken advantage of their unique data accumulation, such privileged access calls for an open-source alternative to democratize Internet-scale financial data.  In this paper, we present an open-source large language model, FinGPT, for the finance sector. Unlike proprietary models, FinGPT takes a data-centric approach, providing researchers and practitioners with accessible and transparent resources to develop their FinLLMs. We highlight the importance of an automatic data curation pipeline and the lightweight low-rank adaptation technique in building FinGPT. Furthermore, we showcase several potential applications as stepping stones for users, such as robo-advising, algorithmic trading, and l
    

