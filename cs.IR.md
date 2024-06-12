# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [SciMMIR: Benchmarking Scientific Multi-modal Information Retrieval.](http://arxiv.org/abs/2401.13478) | SciMMIR是一个专门用于科学领域的多模态信息检索基准，通过开放获取的论文集合提取与科学领域相关的图像-文本配对，从而弥补了现有基准在此领域中的差距。 |
| [^2] | [Block-Diagonal Orthogonal Relation and Matrix Entity for Knowledge Graph Embedding.](http://arxiv.org/abs/2401.05967) | 这个论文提出了一种新型知识图谱嵌入模型OrthogonalE，利用矩阵表示实体和块对角正交矩阵表示关系，增强了模型的灵活性和广泛性，并在实验中表现出比最先进模型更好的性能和较少的参数数量。 |
| [^3] | [A Comprehensive Survey on Deep Learning Techniques in Educational Data Mining.](http://arxiv.org/abs/2309.04761) | 本调研综合审查了在教育数据挖掘中深度学习技术的最新研究进展，包括对知识跟踪、学生不良行为检测、性能预测和个性化推荐等典型教育场景的应用。同时提供了公共数据集和处理工具的综合概述，并指出了未来的研究方向。 |
| [^4] | [LLaMA-E: Empowering E-commerce Authoring with Multi-Aspect Instruction Following.](http://arxiv.org/abs/2308.04913) | LLaMA-E是一种统一且定制的指导语言模型，旨在解决电子商务创作过程中遇到的各种任务，包括广告生成、查询增强的产品标题改写、产品分类、购买意图推测和常规问答。 |

# 详细

[^1]: SciMMIR:科学多模态信息检索的基准评测

    SciMMIR: Benchmarking Scientific Multi-modal Information Retrieval. (arXiv:2401.13478v1 [cs.IR])

    [http://arxiv.org/abs/2401.13478](http://arxiv.org/abs/2401.13478)

    SciMMIR是一个专门用于科学领域的多模态信息检索基准，通过开放获取的论文集合提取与科学领域相关的图像-文本配对，从而弥补了现有基准在此领域中的差距。

    

    多模态信息检索（MMIR）是一个快速发展的领域，通过先进的表示学习和跨模态对齐研究，在图像-文本配对方面取得了显著进展。然而，在科学领域内评估图像-文本配对的MMIR性能的当前基准存在明显差距，学术语言中描述的图表和表格图像通常不起重要作用。为了弥补这一差距，我们利用开放获取的论文集合构建了一个专门的科学MMIR（SciMMIR）基准，以提取与科学领域相关的数据。该基准包含了530K个精心策划的从科学文档中提取的图像-文本配对，这些图像-文本配对来自于具有详细标题的科学文档中的图表和表格。我们还使用两级子集-子类别层次注释对图像-文本配对进行了注释，以促进对基准模型的更全面评估。我们对零样本和微调进行了评估。

    Multi-modal information retrieval (MMIR) is a rapidly evolving field, where significant progress, particularly in image-text pairing, has been made through advanced representation learning and cross-modality alignment research. However, current benchmarks for evaluating MMIR performance in image-text pairing within the scientific domain show a notable gap, where chart and table images described in scholarly language usually do not play a significant role. To bridge this gap, we develop a specialised scientific MMIR (SciMMIR) benchmark by leveraging open-access paper collections to extract data relevant to the scientific domain. This benchmark comprises 530K meticulously curated image-text pairs, extracted from figures and tables with detailed captions in scientific documents. We further annotate the image-text pairs with two-level subset-subcategory hierarchy annotations to facilitate a more comprehensive evaluation of the baselines. We conducted zero-shot and fine-tuning evaluations o
    
[^2]: 知识图谱嵌入的分块对角正交关系和矩阵实体

    Block-Diagonal Orthogonal Relation and Matrix Entity for Knowledge Graph Embedding. (arXiv:2401.05967v1 [cs.CL])

    [http://arxiv.org/abs/2401.05967](http://arxiv.org/abs/2401.05967)

    这个论文提出了一种新型知识图谱嵌入模型OrthogonalE，利用矩阵表示实体和块对角正交矩阵表示关系，增强了模型的灵活性和广泛性，并在实验中表现出比最先进模型更好的性能和较少的参数数量。

    

    知识图谱嵌入的主要目标是学习实体和关系的低维表示以预测缺失的事实。旋转-based方法如RotatE和QuatE在知识图谱嵌入中表现良好，但面临两个挑战：模型的灵活性有限，需要与实体维度成比例地增加关系大小，并且难以推广到更高维度的旋转。为了解决这些问题，我们引入了OrthogonalE，一种新颖的知识图谱嵌入模型，它利用矩阵表示实体和块对角正交矩阵与Riemannian优化表示关系。这种方法增强了知识图谱嵌入模型的广泛性和灵活性。实验结果表明，我们的新型知识图谱嵌入模型OrthogonalE既具有广泛性又具有灵活性，明显优于最先进的知识图谱嵌入模型，并显著减少了关系参数的数量。

    The primary aim of Knowledge Graph embeddings (KGE) is to learn low-dimensional representations of entities and relations for predicting missing facts. While rotation-based methods like RotatE and QuatE perform well in KGE, they face two challenges: limited model flexibility requiring proportional increases in relation size with entity dimension, and difficulties in generalizing the model for higher-dimensional rotations. To address these issues, we introduce OrthogonalE, a novel KGE model employing matrices for entities and block-diagonal orthogonal matrices with Riemannian optimization for relations. This approach enhances the generality and flexibility of KGE models. The experimental results indicate that our new KGE model, OrthogonalE, is both general and flexible, significantly outperforming state-of-the-art KGE models while substantially reducing the number of relation parameters.
    
[^3]: 在教育数据挖掘中深度学习技术的综合调研

    A Comprehensive Survey on Deep Learning Techniques in Educational Data Mining. (arXiv:2309.04761v1 [cs.LG])

    [http://arxiv.org/abs/2309.04761](http://arxiv.org/abs/2309.04761)

    本调研综合审查了在教育数据挖掘中深度学习技术的最新研究进展，包括对知识跟踪、学生不良行为检测、性能预测和个性化推荐等典型教育场景的应用。同时提供了公共数据集和处理工具的综合概述，并指出了未来的研究方向。

    

    教育数据挖掘(EDM)作为研究的重要领域，利用计算技术来分析教育数据。随着教育数据的复杂性和多样性增加，深度学习技术在解决分析和建模这些数据所面临的挑战方面表现出了显著的优势。本调研旨在系统地审查深度学习在EDM领域的最新研究进展。我们首先提供了关于EDM和深度学习的简要介绍，强调了它们在现代教育环境中的重要性。接下来，我们详细回顾了在四个典型教育场景中应用的深度学习技术，包括知识跟踪、学生不良行为检测、性能预测和个性化推荐。此外，我们还提供了EDM的公共数据集和处理工具的综合概述。最后，我们指出了该研究领域的新兴趋势和未来方向。

    Educational Data Mining (EDM) has emerged as a vital field of research, which harnesses the power of computational techniques to analyze educational data. With the increasing complexity and diversity of educational data, Deep Learning techniques have shown significant advantages in addressing the challenges associated with analyzing and modeling this data. This survey aims to systematically review the state-of-the-art in EDM with Deep Learning. We begin by providing a brief introduction to EDM and Deep Learning, highlighting their relevance in the context of modern education. Next, we present a detailed review of Deep Learning techniques applied in four typical educational scenarios, including knowledge tracing, undesirable student detecting, performance prediction, and personalized recommendation. Furthermore, a comprehensive overview of public datasets and processing tools for EDM is provided. Finally, we point out emerging trends and future directions in this research area.
    
[^4]: LLaMA-E：多方面指导下的电子商务创作增强系统

    LLaMA-E: Empowering E-commerce Authoring with Multi-Aspect Instruction Following. (arXiv:2308.04913v1 [cs.CL])

    [http://arxiv.org/abs/2308.04913](http://arxiv.org/abs/2308.04913)

    LLaMA-E是一种统一且定制的指导语言模型，旨在解决电子商务创作过程中遇到的各种任务，包括广告生成、查询增强的产品标题改写、产品分类、购买意图推测和常规问答。

    

    电子商务创作涉及创建吸引人、丰富且有针对性的促销内容，以推动产品销售。大型语言模型（LLM）的出现引入了一种创新的范例，为解决这种情景中的各种创作任务提供了统一的解决方案。然而，基于通用语料库和常识知识训练的主流LLM在适应电子商务产品和客户独特的复杂和个性化特征方面存在局限性。此外，像GPT-3.5这样的LLM需要进行远程访问，引发了在传输过程中保护大量客户隐私数据的担忧。本文提出了LLaMA-E，针对多样化的电子商务创作任务的统一且定制的指导语言模型。具体而言，领域专家从广告生成、查询增强的产品标题改写、产品分类、购买意图推测和常规问答等任务中创建了种子指导集合。这些任务能够...

    E-commerce authoring involves creating attractive, abundant, and targeted promotional content to drive product sales. The emergence of large language models (LLMs) introduces an innovative paradigm, offering a unified solution to address various authoring tasks within this scenario. However, mainstream LLMs trained on general corpora with common sense knowledge reveal limitations in fitting complex and personalized features unique to e-commerce products and customers. Furthermore, LLMs like GPT-3.5 necessitate remote accessibility, raising concerns about safeguarding voluminous customer privacy data during transmission. This paper proposes the LLaMA-E, the unified and customized instruction-following language models focusing on diverse e-commerce authoring tasks. Specifically, the domain experts create the seed instruction set from the tasks of ads generation, query-enhanced product title rewriting, product classification, purchase intent speculation, and general Q&A. These tasks enabl
    

