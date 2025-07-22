# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Attention with Markov: A Framework for Principled Analysis of Transformers via Markov Chains](https://arxiv.org/abs/2402.04161) | 提出了一个新的框架，通过马尔可夫链的视角研究了注意力模型的顺序建模能力，理论上刻画了单层Transformer的损失景观并发现了全局最小值和坏局部最小值的存在。 |
| [^2] | [VlogQA: Task, Dataset, and Baseline Models for Vietnamese Spoken-Based Machine Reading Comprehension](https://arxiv.org/abs/2402.02655) | 本文介绍了VlogQA：越南口语机器阅读理解任务、数据集和基线模型，并提供了使用真实数据进行任务的挑战和机遇的见解。VlogQA是一个基于来自YouTube的剧本文档的问答对数据集，涵盖了食物和旅行等主题。深度学习模型在测试集取得了75.34%的最高F1分数。 |
| [^3] | [Where Do People Tell Stories Online? Story Detection Across Online Communities](https://arxiv.org/abs/2311.09675) | 介绍了一个解决在线社区中故事检测困难的挑战，提出了StorySeeker工具包，包括详细注释的Reddit数据集和模型，突出了在线叙事的文本特征，引入了叙事跨度检测作为一个新任务。 |
| [^4] | [Transformers and Ensemble methods: A solution for Hate Speech Detection in Arabic languages.](http://arxiv.org/abs/2303.09823) | 本文提出了一种使用Transformer和Ensemble方法的解决方案，用于阿语恶意言论的检测。实验结果表明，基于多数表决的集成方法具有最佳效果，其在测试集上的准确率为0.86，F1分数为0.60。 |

# 详细

[^1]: 基于马尔可夫链的注意力模型的规范分析框架：通过马尔可夫链研究Transformer的顺序建模能力

    Attention with Markov: A Framework for Principled Analysis of Transformers via Markov Chains

    [https://arxiv.org/abs/2402.04161](https://arxiv.org/abs/2402.04161)

    提出了一个新的框架，通过马尔可夫链的视角研究了注意力模型的顺序建模能力，理论上刻画了单层Transformer的损失景观并发现了全局最小值和坏局部最小值的存在。

    

    近年来，基于注意力的Transformer在包括自然语言在内的多个领域取得了巨大成功。其中一个关键因素是生成式预训练过程，模型在此过程中通过自回归的方式在大型文本语料库上进行训练。为了揭示这一现象，我们提出了一个新的框架，通过马尔可夫链的视角，允许理论和系统实验来研究Transformer的顺序建模能力。受到自然语言的马尔可夫性质的启发，我们将数据建模为一个马尔可夫源，并利用这个框架系统地研究数据分布特性、Transformer架构、学到的分布和最终模型性能之间的相互作用。特别地，我们理论上刻画了单层Transformer的损失景观，并展示了全局最小值和坏局部最小值的存在，这取决于具体的数据性质。

    In recent years, attention-based transformers have achieved tremendous success across a variety of disciplines including natural languages. A key ingredient behind their success is the generative pretraining procedure, during which these models are trained on a large text corpus in an auto-regressive manner. To shed light on this phenomenon, we propose a new framework that allows both theory and systematic experiments to study the sequential modeling capabilities of transformers through the lens of Markov chains. Inspired by the Markovianity of natural languages, we model the data as a Markovian source and utilize this framework to systematically study the interplay between the data-distributional properties, the transformer architecture, the learnt distribution, and the final model performance. In particular, we theoretically characterize the loss landscape of single-layer transformers and show the existence of global minima and bad local minima contingent upon the specific data chara
    
[^2]: VlogQA: 越南口语机器阅读理解任务、数据集和基线模型

    VlogQA: Task, Dataset, and Baseline Models for Vietnamese Spoken-Based Machine Reading Comprehension

    [https://arxiv.org/abs/2402.02655](https://arxiv.org/abs/2402.02655)

    本文介绍了VlogQA：越南口语机器阅读理解任务、数据集和基线模型，并提供了使用真实数据进行任务的挑战和机遇的见解。VlogQA是一个基于来自YouTube的剧本文档的问答对数据集，涵盖了食物和旅行等主题。深度学习模型在测试集取得了75.34%的最高F1分数。

    

    本文介绍了一个用于机器阅读理解任务的越南口语语料库的开发过程，并提供了使用真实数据进行机器阅读理解任务时遇到的挑战和机遇的见解。现有的越南机器阅读理解语料库主要关注正式的书面文档，如维基百科文章、在线报纸或教科书。与之相反，VlogQA包含了10,076个问答对，基于从YouTube获取的1,230份剧本文档，YouTube是一个包含了用户上传内容的广泛资源，涵盖了食物和旅行等主题。通过捕捉越南本土人在自然环境中的口语表达，这是越南研究中被忽视的一个角落，该语料库为未来越南语阅读理解任务的研究提供了宝贵的资源。在性能评估方面，我们的深度学习模型在测试集上取得了最高的F1分数为75.34%，表明了其优秀的性能。

    This paper presents the development process of a Vietnamese spoken language corpus for machine reading comprehension (MRC) tasks and provides insights into the challenges and opportunities associated with using real-world data for machine reading comprehension tasks. The existing MRC corpora in Vietnamese mainly focus on formal written documents such as Wikipedia articles, online newspapers, or textbooks. In contrast, the VlogQA consists of 10,076 question-answer pairs based on 1,230 transcript documents sourced from YouTube -- an extensive source of user-uploaded content, covering the topics of food and travel. By capturing the spoken language of native Vietnamese speakers in natural settings, an obscure corner overlooked in Vietnamese research, the corpus provides a valuable resource for future research in reading comprehension tasks for the Vietnamese language. Regarding performance evaluation, our deep-learning models achieved the highest F1 score of 75.34% on the test set, indicat
    
[^3]: 人们在哪里在线讲故事？跨在线社区的故事检测

    Where Do People Tell Stories Online? Story Detection Across Online Communities

    [https://arxiv.org/abs/2311.09675](https://arxiv.org/abs/2311.09675)

    介绍了一个解决在线社区中故事检测困难的挑战，提出了StorySeeker工具包，包括详细注释的Reddit数据集和模型，突出了在线叙事的文本特征，引入了叙事跨度检测作为一个新任务。

    

    在线社区中的故事检测是一项具有挑战性的任务，因为故事分散在社区中，并且与单个文本中的非叙事部分交织在一起。我们通过构建和发布StorySeeker工具包来解决这一挑战，其中包括一个包含502个Reddit帖子和评论的丰富注释数据集，一个适应社交媒体背景的详细的代码书，以及用于在文档和跨度级别预测叙事的模型。我们的数据集是从数百个流行的英语Reddit社区中抽样而来，涵盖了33个主题类别，它包含了细粒度的专家注释，包括二元故事标签，故事跨度和事件跨度。我们使用我们的数据评估了一系列检测方法，并确定了在线叙事的独特文本特征，重点关注叙事跨度检测，这是我们引入的一个新任务。我们阐明了大规模叙事的分布特征。

    arXiv:2311.09675v2 Announce Type: replace  Abstract: Story detection in online communities is a challenging task as stories are scattered across communities and interwoven with non-storytelling spans within a single text. We address this challenge by building and releasing the StorySeeker toolkit, including a richly annotated dataset of 502 Reddit posts and comments, a detailed codebook adapted to the social media context, and models to predict storytelling at the document and span level. Our dataset is sampled from hundreds of popular English-language Reddit communities ranging across 33 topic categories, and it contains fine-grained expert annotations, including binary story labels, story spans, and event spans. We evaluate a range of detection methods using our data, and we identify the distinctive textual features of online storytelling, focusing on storytelling span detection, which we introduce as a new task. We illuminate distributional characteristics of storytelling on a large
    
[^4]: Transformers和Ensemble方法：阿语恶意言论检测的一种解决方案

    Transformers and Ensemble methods: A solution for Hate Speech Detection in Arabic languages. (arXiv:2303.09823v1 [cs.CL])

    [http://arxiv.org/abs/2303.09823](http://arxiv.org/abs/2303.09823)

    本文提出了一种使用Transformer和Ensemble方法的解决方案，用于阿语恶意言论的检测。实验结果表明，基于多数表决的集成方法具有最佳效果，其在测试集上的准确率为0.86，F1分数为0.60。

    

    本文描述了我们参加CERIST NLP挑战赛2022中恶意言论检测共享任务的实验过程。我们评估了6个Transformer模型及其组合的性能，并使用了2种集成方法。在五折交叉验证的训练集上，基于多数表决的集成方法获得了最佳结果。在测试集上的评估结果为F1分数为0.60，准确性为0.86。

    This paper describes our participation in the shared task of hate speech detection, which is one of the subtasks of the CERIST NLP Challenge 2022. Our experiments evaluate the performance of six transformer models and their combination using 2 ensemble approaches. The best results on the training set, in a five-fold cross validation scenario, were obtained by using the ensemble approach based on the majority vote. The evaluation of this approach on the test set resulted in an F1-score of 0.60 and an Accuracy of 0.86.
    

