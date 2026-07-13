# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Contrastive Learning on Multimodal Analysis of Electronic Health Records](https://arxiv.org/abs/2403.14926) | 该论文研究了电子健康记录的多模态分析，强调了结构化和非结构化数据之间的协同作用，并尝试将多模态对比学习方法应用于提高患者医疗历史的完整性。 |
| [^2] | [Hierarchical Causal Models.](http://arxiv.org/abs/2401.05330) | 提出了一种分层因果模型来解决关于分层数据的因果问题，通过添加内部板来扩展结构因果模型和因果图模型。发现分层数据可以实现因果识别，即使使用非分层数据是不可能的。开发了用于分层数据的估计技术。 |

# 详细

[^1]: 电子健康记录的多模态分析上的对比学习

    Contrastive Learning on Multimodal Analysis of Electronic Health Records

    [https://arxiv.org/abs/2403.14926](https://arxiv.org/abs/2403.14926)

    该论文研究了电子健康记录的多模态分析，强调了结构化和非结构化数据之间的协同作用，并尝试将多模态对比学习方法应用于提高患者医疗历史的完整性。

    

    电子健康记录（EHR）系统包含大量的多模态临床数据，包括结构化数据如临床编码和非结构化数据如临床笔记。然而，许多现有的针对EHR的研究传统上要么集中于个别模态，要么以一种相当粗糙的方式合并不同的模态。这种方法通常会导致将结构化和非结构化数据视为单独实体，忽略它们之间固有的协同作用。具体来说，这两个重要的模态包含临床相关、密切相关和互补的健康信息。通过联合分析这两种数据模态可以捕捉到患者医疗历史的更完整画面。尽管多模态对比学习在视觉语言领域取得了巨大成功，但在多模态EHR领域，尤其是在理论理解方面，其潜力仍未充分挖掘。

    arXiv:2403.14926v1 Announce Type: cross  Abstract: Electronic health record (EHR) systems contain a wealth of multimodal clinical data including structured data like clinical codes and unstructured data such as clinical notes. However, many existing EHR-focused studies has traditionally either concentrated on an individual modality or merged different modalities in a rather rudimentary fashion. This approach often results in the perception of structured and unstructured data as separate entities, neglecting the inherent synergy between them. Specifically, the two important modalities contain clinically relevant, inextricably linked and complementary health information. A more complete picture of a patient's medical history is captured by the joint analysis of the two modalities of data. Despite the great success of multimodal contrastive learning on vision-language, its potential remains under-explored in the realm of multimodal EHR, particularly in terms of its theoretical understandi
    
[^2]: 分层因果模型

    Hierarchical Causal Models. (arXiv:2401.05330v1 [stat.ME])

    [http://arxiv.org/abs/2401.05330](http://arxiv.org/abs/2401.05330)

    提出了一种分层因果模型来解决关于分层数据的因果问题，通过添加内部板来扩展结构因果模型和因果图模型。发现分层数据可以实现因果识别，即使使用非分层数据是不可能的。开发了用于分层数据的估计技术。

    

    科学家们经常想要从分层数据中学习因果关系，这些数据是从嵌套在单位内部的子单元收集的。比如学校中的学生、病人的细胞或州中的城市。在这种情况下，单位级变量（例如每个学校的预算）可能会影响子单位级变量（例如每个学校每个学生的考试成绩），反之亦然。为了解决关于分层数据的因果问题，我们提出了分层因果模型，它通过添加内部板来扩展结构因果模型和因果图模型。我们开发了一种用于分层因果模型的通用图形识别技术，该技术扩展了do-计算。我们发现许多情况下，即使使用非分层数据是不可能的，分层数据也可以实现因果识别，也就是说，如果我们只有子单位级变量的单位级汇总（例如学校的平均考试成绩，而不是每个学生的成绩）。我们开发了用于分层数据的估计技术。

    Scientists often want to learn about cause and effect from hierarchical data, collected from subunits nested inside units. Consider students in schools, cells in patients, or cities in states. In such settings, unit-level variables (e.g. each school's budget) may affect subunit-level variables (e.g. the test scores of each student in each school) and vice versa. To address causal questions with hierarchical data, we propose hierarchical causal models, which extend structural causal models and causal graphical models by adding inner plates. We develop a general graphical identification technique for hierarchical causal models that extends do-calculus. We find many situations in which hierarchical data can enable causal identification even when it would be impossible with non-hierarchical data, that is, if we had only unit-level summaries of subunit-level variables (e.g. the school's average test score, rather than each student's score). We develop estimation techniques for hierarchical 
    

