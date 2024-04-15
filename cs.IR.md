# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [IISAN: Efficiently Adapting Multimodal Representation for Sequential Recommendation with Decoupled PEFT](https://arxiv.org/abs/2404.02059) | IISAN是一种简单的插拔架构，采用解耦PEFT结构，并利用内部和跨模态适应，与全微调和最先进的PEFT性能匹配，显著减少GPU内存使用量，并加速了训练时间。 |
| [^2] | [Leveraging Foundation Models for Content-Based Medical Image Retrieval in Radiology](https://arxiv.org/abs/2403.06567) | 基于内容的医学图像检索中，利用基础模型作为特征提取器，无需微调即可取得与专门模型竞争的性能，尤其在检索病理特征方面具有较大困难。 |

# 详细

[^1]: IISAN：使用解耦PEFT有效地调整多模态表示以顺序推荐

    IISAN: Efficiently Adapting Multimodal Representation for Sequential Recommendation with Decoupled PEFT

    [https://arxiv.org/abs/2404.02059](https://arxiv.org/abs/2404.02059)

    IISAN是一种简单的插拔架构，采用解耦PEFT结构，并利用内部和跨模态适应，与全微调和最先进的PEFT性能匹配，显著减少GPU内存使用量，并加速了训练时间。

    

    多模态基础模型在顺序推荐系统中具有转变性，利用强大的表示学习能力。虽然参数高效微调（PEFT）通常用于调整基础模型以进行推荐任务，但大多数研究优先考虑参数效率，通常忽略GPU内存效率和训练速度等关键因素。针对这一差距，本文引入了IISAN（多模态表示的内部和跨模态侧面适应网络），一个使用解耦PEFT结构并利用内部和跨模态适应的简单即插即用架构。IISAN与全微调（FFT）和最先进的PEFT的性能相匹配。更重要的是，它显著减少了GPU内存使用量 - 对于多模态顺序推荐任务，从47GB降低到仅3GB。此外，与FFT相比，它将每个时代的训练时间从443秒加速到22秒。

    arXiv:2404.02059v1 Announce Type: new  Abstract: Multimodal foundation models are transformative in sequential recommender systems, leveraging powerful representation learning capabilities. While Parameter-efficient Fine-tuning (PEFT) is commonly used to adapt foundation models for recommendation tasks, most research prioritizes parameter efficiency, often overlooking critical factors like GPU memory efficiency and training speed. Addressing this gap, our paper introduces IISAN (Intra- and Inter-modal Side Adapted Network for Multimodal Representation), a simple plug-and-play architecture using a Decoupled PEFT structure and exploiting both intra- and inter-modal adaptation.   IISAN matches the performance of full fine-tuning (FFT) and state-of-the-art PEFT. More importantly, it significantly reduces GPU memory usage - from 47GB to just 3GB for multimodal sequential recommendation tasks. Additionally, it accelerates training time per epoch from 443s to 22s compared to FFT. This is also
    
[^2]: 利用基础模型进行放射学中基于内容的医学图像检索

    Leveraging Foundation Models for Content-Based Medical Image Retrieval in Radiology

    [https://arxiv.org/abs/2403.06567](https://arxiv.org/abs/2403.06567)

    基于内容的医学图像检索中，利用基础模型作为特征提取器，无需微调即可取得与专门模型竞争的性能，尤其在检索病理特征方面具有较大困难。

    

    Content-based image retrieval（CBIR）有望显著改善放射学中的诊断辅助和医学研究。我们提出利用视觉基础模型作为强大且多功能的现成特征提取器，用于基于内容的医学图像检索。通过在涵盖四种模态和161种病理学的160万张2D放射图像的全面数据集上对这些模型进行基准测试，我们发现弱监督模型表现优异，P@1可达0.594。这种性能不仅与专门化模型竞争，而且无需进行微调。我们的分析进一步探讨了检索病理学与解剖结构的挑战，表明准确检索病理特征更具挑战性。

    arXiv:2403.06567v1 Announce Type: cross  Abstract: Content-based image retrieval (CBIR) has the potential to significantly improve diagnostic aid and medical research in radiology. Current CBIR systems face limitations due to their specialization to certain pathologies, limiting their utility. In response, we propose using vision foundation models as powerful and versatile off-the-shelf feature extractors for content-based medical image retrieval. By benchmarking these models on a comprehensive dataset of 1.6 million 2D radiological images spanning four modalities and 161 pathologies, we identify weakly-supervised models as superior, achieving a P@1 of up to 0.594. This performance not only competes with a specialized model but does so without the need for fine-tuning. Our analysis further explores the challenges in retrieving pathological versus anatomical structures, indicating that accurate retrieval of pathological features presents greater difficulty. Despite these challenges, our
    

