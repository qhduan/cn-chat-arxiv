# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Survey in Characterization of Semantic Change](https://arxiv.org/abs/2402.19088) | 语义变化对计算语言学算法的结果质量可能会产生影响，因此重要性日益凸显。 |
| [^2] | [Automated Approaches to Detect Self-Admitted Technical Debt: A Systematic Literature Review](https://arxiv.org/abs/2312.15020) | 论文提出了一种特征提取技术和ML/DL算法分类法，旨在比较和基准测试其在技术债务检测中的表现。 |
| [^3] | [Automatically Estimating the Effort Required to Repay Self-Admitted Technical Debt.](http://arxiv.org/abs/2309.06020) | 本研究提出了一种新的方法，利用大规模的数据集自动估算自认技术债务的还款工作量。研究结果表明，不同类型的自认技术债务需要不同程度的还款工作量。 |

# 详细

[^1]: 对语义变化特征的调查

    Survey in Characterization of Semantic Change

    [https://arxiv.org/abs/2402.19088](https://arxiv.org/abs/2402.19088)

    语义变化对计算语言学算法的结果质量可能会产生影响，因此重要性日益凸显。

    

    活语言不断发展，以吸纳人类社会的文化变化。这种演变通过新词语（新单词）或单词的语义变化（赋予已有单词新的含义）来体现。理解单词的含义对解释来自不同文化（地方用语或俚语）、领域（例如技术术语）或时代的文本至关重要。在计算机科学中，这些单词与计算语言学算法相关，例如翻译、信息检索、问答等。语义变化可能会影响这些算法的结果质量。因此，了解和形式化表征这些变化是很重要的。研究这种影响是计算语言学界近期引起关注的问题。几种方法提出了检测语义变化的方法，具有较高的精度，但需要更多努力来对其进行表征。

    arXiv:2402.19088v1 Announce Type: cross  Abstract: Live languages continuously evolve to integrate the cultural change of human societies. This evolution manifests through neologisms (new words) or \textbf{semantic changes} of words (new meaning to existing words). Understanding the meaning of words is vital for interpreting texts coming from different cultures (regionalism or slang), domains (e.g., technical terms), or periods. In computer science, these words are relevant to computational linguistics algorithms such as translation, information retrieval, question answering, etc. Semantic changes can potentially impact the quality of the outcomes of these algorithms. Therefore, it is important to understand and characterize these changes formally. The study of this impact is a recent problem that has attracted the attention of the computational linguistics community. Several approaches propose methods to detect semantic changes with good precision, but more effort is needed to charact
    
[^2]: 自动化方法检测自我承认的技术债务：系统文献综述

    Automated Approaches to Detect Self-Admitted Technical Debt: A Systematic Literature Review

    [https://arxiv.org/abs/2312.15020](https://arxiv.org/abs/2312.15020)

    论文提出了一种特征提取技术和ML/DL算法分类法，旨在比较和基准测试其在技术债务检测中的表现。

    

    技术债务是软件开发中普遍存在的问题，通常源自开发过程中做出的权衡，在影响软件可维护性和阻碍未来开发工作方面起到作用。自我承认的技术债务（SATD）指的是开发人员明确承认代码库中存在的代码质量或设计缺陷。自动检测SATD已经成为一个重要的研究领域，旨在帮助开发人员高效地识别和解决技术债务。然而，文献中广泛采用的NLP特征提取方法和算法种类多样化常常阻碍研究人员试图提高其性能。基于此，本系统文献综述提出了一种特征提取技术和ML/DL算法分类法，其目的是比较和基准测试所考察研究中它们的性能。我们选择......

    arXiv:2312.15020v2 Announce Type: replace-cross  Abstract: Technical debt is a pervasive issue in software development, often arising from trade-offs made during development, which can impede software maintainability and hinder future development efforts. Self-admitted technical debt (SATD) refers to instances where developers explicitly acknowledge suboptimal code quality or design flaws in the codebase. Automated detection of SATD has emerged as a critical area of research, aiming to assist developers in identifying and addressing technical debt efficiently. However, the enormous variety of feature extraction approaches of NLP and algorithms employed in the literature often hinder researchers from trying to improve their performance. In light of this, this systematic literature review proposes a taxonomy of feature extraction techniques and ML/DL algorithms used in technical debt detection: its objective is to compare and benchmark their performance in the examined studies. We select
    
[^3]: 自动评估偿还自认技术债务所需的工作量

    Automatically Estimating the Effort Required to Repay Self-Admitted Technical Debt. (arXiv:2309.06020v1 [cs.SE])

    [http://arxiv.org/abs/2309.06020](http://arxiv.org/abs/2309.06020)

    本研究提出了一种新的方法，利用大规模的数据集自动估算自认技术债务的还款工作量。研究结果表明，不同类型的自认技术债务需要不同程度的还款工作量。

    

    技术债务是指在软件开发过程中为了短期利益而做出的次优决策所带来的后果。自认技术债务(SATD)是一种特定形式的技术债务，开发人员明确地在软件的源代码注释和提交消息中记录下来。由于SATD可能阻碍软件的开发和维护，因此有效地解决和优先处理它非常重要。然而，目前的方法缺乏根据SATD的文本描述自动评估其还款工作量的能力。为了解决这个限制，我们提出了一种新的方法，利用一个包括1,060个Apache代码库中共2,568,728个提交的341,740个SATD项目的全面数据集来自动估算SATD还款工作量。我们的研究结果表明，不同类型的SATD需要不同程度的还款工作量，其中代码/设计、需求和测试债务需要更多的工作量。

    Technical debt refers to the consequences of sub-optimal decisions made during software development that prioritize short-term benefits over long-term maintainability. Self-Admitted Technical Debt (SATD) is a specific form of technical debt, explicitly documented by developers within software artifacts such as source code comments and commit messages. As SATD can hinder software development and maintenance, it is crucial to address and prioritize it effectively. However, current methodologies lack the ability to automatically estimate the repayment effort of SATD based on its textual descriptions. To address this limitation, we propose a novel approach for automatically estimating SATD repayment effort, utilizing a comprehensive dataset comprising 341,740 SATD items from 2,568,728 commits across 1,060 Apache repositories. Our findings show that different types of SATD require varying levels of repayment effort, with code/design, requirement, and test debt demanding greater effort compa
    

