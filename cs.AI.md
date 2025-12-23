# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [ViGoR: Improving Visual Grounding of Large Vision Language Models with Fine-Grained Reward Modeling](https://arxiv.org/abs/2402.06118) | ViGoR通过细粒度奖励建模提高了大型视觉语言模型在视觉对接方面的性能，通过人工评估和自动化方法有效地解决了视觉对接中的误差问题。 |
| [^2] | [CodeTF: One-stop Transformer Library for State-of-the-art Code LLM.](http://arxiv.org/abs/2306.00029) | CodeTF是一个开源的Transformer库，提供了包括预训练的Code LLM模型和标准化接口等一系列功能，可以轻松地将最先进的Code LLM模型应用于各种软件工程任务中。 |

# 详细

[^1]: ViGoR：通过细粒度奖励建模改进大规模视觉语言模型的视觉对接

    ViGoR: Improving Visual Grounding of Large Vision Language Models with Fine-Grained Reward Modeling

    [https://arxiv.org/abs/2402.06118](https://arxiv.org/abs/2402.06118)

    ViGoR通过细粒度奖励建模提高了大型视觉语言模型在视觉对接方面的性能，通过人工评估和自动化方法有效地解决了视觉对接中的误差问题。

    

    通过将自然语言理解、大语言模型的生成能力和广泛知识与图像感知相结合，最近的大规模视觉语言模型（LVLMs）在现实世界中展示了前所未有的推理能力。然而，生成的文本往往在视觉输入中存在不准确的对接，导致错误，如产生幻觉的不存在场景元素、遗漏重要的场景部分，以及推测对象之间的属性和关系时出现错误。为了解决这些问题，我们引入了一个新颖的框架ViGoR（通过细粒度奖励建模进行视觉对接），它利用细粒度奖励建模来显著提升基于预训练基线的LVLMs的视觉对接能力。这种改进通过使用比完全监督更便宜的人工评估和自动化方法高效实现。我们通过多个基准测试的多个指标展示了我们方法的有效性。

    By combining natural language understanding and the generation capabilities and breadth of knowledge of large language models with image perception, recent large vision language models (LVLMs) have shown unprecedented reasoning capabilities in the real world. However, the generated text often suffers from inaccurate grounding in the visual input, resulting in errors such as hallucinating nonexistent scene elements, missing significant parts of the scene, and inferring incorrect attributes and relationships between objects. To address these issues, we introduce a novel framework, ViGoR (Visual Grounding Through Fine-Grained Reward Modeling) that utilizes fine-grained reward modeling to significantly enhance the visual grounding of LVLMs over pre-trained baselines. This improvement is efficiently achieved using much cheaper human evaluations instead of full supervisions, as well as automated methods. We show the effectiveness of our approach through numerous metrics on several benchmarks
    
[^2]: CodeTF：一站式Transformer库，实现最先进的代码LLM

    CodeTF: One-stop Transformer Library for State-of-the-art Code LLM. (arXiv:2306.00029v1 [cs.SE])

    [http://arxiv.org/abs/2306.00029](http://arxiv.org/abs/2306.00029)

    CodeTF是一个开源的Transformer库，提供了包括预训练的Code LLM模型和标准化接口等一系列功能，可以轻松地将最先进的Code LLM模型应用于各种软件工程任务中。

    

    代码智能在转型现代软件工程中扮演着重要角色。近年来，基于深度学习的模型，尤其是利用大量开源代码和编程语言特征的Transformer-based大型语言模型（LLMs），已经展示出了对这些任务的显著潜力。然而，这些模型的开发和部署通常需要对机器学习和软件工程的专业知识，从而为模型应用带来了一定的障碍。本文提出了CodeTF，一个基于Transformer的开放源代码库，用于实现最先进的Code LLM和代码智能。我们采用模块化设计和可扩展框架的原则，设计CodeTF并提供统一接口，以便快速访问和开发不同类型的模型、数据集和任务。我们的库支持预训练的Code LLM模型和流行的代码基准测试，包括标准化接口以有效地训练和服务代码LLMs，并支持双GPU训练和推理。使用CodeTF，用户可以轻松将最先进的Code LLM模型应用于各种软件工程任务中，减少训练工作量。

    Code intelligence plays a key role in transforming modern software engineering. Recently, deep learning-based models, especially Transformer-based large language models (LLMs), have demonstrated remarkable potential in tackling these tasks by leveraging massive open-source code data and programming language features. However, the development and deployment of such models often require expertise in both machine learning and software engineering, creating a barrier for the model adoption. In this paper, we present CodeTF, an open-source Transformer-based library for state-of-the-art Code LLMs and code intelligence. Following the principles of modular design and extensible framework, we design CodeTF with a unified interface to enable rapid access and development across different types of models, datasets and tasks. Our library supports a collection of pretrained Code LLM models and popular code benchmarks, including a standardized interface to train and serve code LLMs efficiently, and d
    

