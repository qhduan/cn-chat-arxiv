# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Beyond the Hard Budget: Sparsity Regularizers for More Interpretable Top-k Sparse Autoencoders](https://arxiv.org/abs/2606.27321) | 本文针对Top-k稀疏自编码器固定预算和过拟合缺陷，提出了两种新的稀疏正则化方法，通过作用于Top-k选择前的激活值来提升模型可解释性。 |
| [^2] | [Hybrid privacy-aware semantic search: SVD-truncated document geometry and CKKS-encrypted query reranking under a restricted threat model](https://arxiv.org/abs/2606.26373) | 本文提出一种混合隐私保护语义搜索方法，通过SVD截断和秘密正交变换保护文档集合，利用CKKS同态加密保护查询，在受限威胁模型下平衡了安全性与效率。 |
| [^3] | [TTA-Nav: Test-time Adaptive Reconstruction for Point-Goal Navigation under Visual Corruptions](https://arxiv.org/abs/2403.01977) | TTA-Nav提出了一种测试时自适应方法，通过引入自顶向下解码器，从损坏图像中重建出更清晰的图像，显著增强了点目标导航性能。 |

# 详细

[^1]: 超越硬预算：面向更可解释的Top-k稀疏自编码器的稀疏正则化方法

    Beyond the Hard Budget: Sparsity Regularizers for More Interpretable Top-k Sparse Autoencoders

    [https://arxiv.org/abs/2606.27321](https://arxiv.org/abs/2606.27321)

    本文针对Top-k稀疏自编码器固定预算和过拟合缺陷，提出了两种新的稀疏正则化方法，通过作用于Top-k选择前的激活值来提升模型可解释性。

    

    arXiv:2606.27321v1 公告类型：交叉 摘要：稀疏自编码器已成为解释视觉基础模型表示的主要工具，将其多语义激活分解为更大规模的稀疏、更单语义特征集合。Top-k稀疏自编码器作为当前标准变体，通过其激活函数从架构层面强制实现稀疏性，每个输入仅保留最活跃的k个潜在变量。由于该设计旨在规避早期稀疏自编码器使用的ℓ1惩罚及其已知缺陷，因此尽管其本身存在局限性——如预算k固定不变（不随输入复杂度调整）以及倾向于对训练时设定的k值过拟合——但至今未与显式稀疏正则化方法结合。我们提出了两种与Top-k架构兼容的稀疏正则化方法，两者均在Top-k选择之前作用于激活值：对未选中单元施加ℓ1惩罚，以及一种尺度不变的ℓ...（原文截断）

    arXiv:2606.27321v1 Announce Type: cross  Abstract: Sparse autoencoders (SAEs) have become a leading tool for interpreting the representations of vision foundation models, decomposing their polysemantic activations into a larger set of sparse, more monosemantic features. The Top-$k$ SAE, a now-standard variant, enforces sparsity architecturally through its activation function, retaining only the $k$ most active latents per input. Because it was designed precisely to avoid the $\ell_1$ penalty used by earlier SAEs and its known drawbacks, it has not been combined with an explicit sparsity regularizer, despite retaining limitations of its own, such as a budget $k$ that is fixed regardless of input complexity and a tendency to overfit to the training value of $k$. We introduce two sparsity regularizers compatible with the Top-$k$ architecture, both acting on the activations before the Top-$k$ selection: an $\ell_1$ penalty on the unselected (off-support) units, and a scale-invariant $\ell_
    
[^2]: 混合隐私感知语义搜索：受限威胁模型下基于SVD截断文档几何与CKKS加密查询重排序

    Hybrid privacy-aware semantic search: SVD-truncated document geometry and CKKS-encrypted query reranking under a restricted threat model

    [https://arxiv.org/abs/2606.26373](https://arxiv.org/abs/2606.26373)

    本文提出一种混合隐私保护语义搜索方法，通过SVD截断和秘密正交变换保护文档集合，利用CKKS同态加密保护查询，在受限威胁模型下平衡了安全性与效率。

    

    arXiv:2606.26373v1 公告类型：交叉 摘要：稠密嵌入为语义搜索和检索增强生成提供了强大支持，但嵌入反转攻击可以从向量中重建源文本：当向量数据库泄露时，其背后的文档也会随之泄露。教科书式的防御措施是极端方案——对整个搜索进行同态加密是可靠的，但在百万级文档规模下速度过慢，而隐私噪声在提供保护之前就已严重降低排序质量。我们研究了一条中间路径，利用静态集合与动态查询之间的不对称性。集合通过几何方式保护：每个向量被截断到低维SVD子空间，并通过仅由所有者知道的秘密正交变换进行旋转。查询通过密码学方式保护：在CKKS同态加密下进行重排序，因此诚实但好奇的服务器永远无法看到查询或分数。CKKS参数来自一个小型离线基准测试。我们证明了重构的下界紧致性。

    arXiv:2606.26373v1 Announce Type: cross  Abstract: Dense embeddings power semantic search and retrieval-augmented generation, but embedding-inversion attacks can reconstruct source text from a vector: when a vector database leaks, the documents behind it leak too. The textbook defences are extremes - encrypting the whole search homomorphically is sound but too slow at million-document scale, while privacy noise degrades ranking long before it protects. We study a middle path exploiting the asymmetry between the static collection and the dynamic query. The collection is protected geometrically: each vector is truncated onto a lower-dimensional SVD subspace and rotated by a secret orthogonal transform known only to the owner. The query is protected cryptographically: it is reranked under CKKS homomorphic encryption, so an honest-but-curious server never sees the query or the scores. CKKS parameters come from a small offline benchmark.   We prove a tight lower bound on the reconstruction 
    
[^3]: TTA-Nav: 测试时自适应重建用于视觉损坏下的点目标导航

    TTA-Nav: Test-time Adaptive Reconstruction for Point-Goal Navigation under Visual Corruptions

    [https://arxiv.org/abs/2403.01977](https://arxiv.org/abs/2403.01977)

    TTA-Nav提出了一种测试时自适应方法，通过引入自顶向下解码器，从损坏图像中重建出更清晰的图像，显著增强了点目标导航性能。

    

    arXiv:2403.01977v1 公告类型: 跨  摘要: 在视觉损坏下的机器人导航是一个巨大的挑战。为了解决这一问题，我们提出了一种名为TTA-Nav的测试时自适应（TTA）方法，用于在视觉损坏下的点目标导航。我们的“即插即用”方法将自顶向下的解码器与预训练的导航模型相结合。首先，预训练的导航模型接收一个损坏的图像并提取特征。其次，自顶向下的解码器根据预训练模型提取的高级特征生成重建图像。然后，将损坏图像的重建图像馈送回预训练模型。最后，预训练模型再次进行前向传播以输出动作。尽管仅在清晰图像上训练，自顶向下的解码器可以从损坏图像中重建出更清晰的图像，无需基于梯度的自适应。具有我们自顶向下解码器的预训练导航模型显著提高了导航性能。

    arXiv:2403.01977v1 Announce Type: cross  Abstract: Robot navigation under visual corruption presents a formidable challenge. To address this, we propose a Test-time Adaptation (TTA) method, named as TTA-Nav, for point-goal navigation under visual corruptions. Our "plug-and-play" method incorporates a top-down decoder to a pre-trained navigation model. Firstly, the pre-trained navigation model gets a corrupted image and extracts features. Secondly, the top-down decoder produces the reconstruction given the high-level features extracted by the pre-trained model. Then, it feeds the reconstruction of a corrupted image back to the pre-trained model. Finally, the pre-trained model does forward pass again to output action. Despite being trained solely on clean images, the top-down decoder can reconstruct cleaner images from corrupted ones without the need for gradient-based adaptation. The pre-trained navigation model with our top-down decoder significantly enhances navigation performance acr
    

