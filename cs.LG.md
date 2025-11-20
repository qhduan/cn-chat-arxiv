# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Zero-shot Object-Level OOD Detection with Context-Aware Inpainting](https://arxiv.org/abs/2402.03292) | 本论文提出了一种用上下文感知修复的零样本物体级OOD检测方法RONIN。通过将检测到的对象进行修复替换，并使用预测的ID标签来条件化修复过程，使得重构的对象在OOD情况下与原始对象相差较远，从而有效区分ID和OOD样本。实验证明RONIN在多个数据集上取得了具有竞争力的结果。 |

# 详细

[^1]: 用上下文感知修复的零样本物体级OOD检测

    Zero-shot Object-Level OOD Detection with Context-Aware Inpainting

    [https://arxiv.org/abs/2402.03292](https://arxiv.org/abs/2402.03292)

    本论文提出了一种用上下文感知修复的零样本物体级OOD检测方法RONIN。通过将检测到的对象进行修复替换，并使用预测的ID标签来条件化修复过程，使得重构的对象在OOD情况下与原始对象相差较远，从而有效区分ID和OOD样本。实验证明RONIN在多个数据集上取得了具有竞争力的结果。

    

    机器学习算法越来越多地作为黑盒云服务或预训练模型提供，无法访问它们的训练数据。这就引发了零样本离群数据（OOD）检测的问题。具体而言，我们的目标是检测不属于分类器标签集但被错误地归类为入域（ID）对象的OOD对象。我们的方法RONIN使用现成的扩散模型来用修复替换掉检测到的对象。RONIN使用预测的ID标签来条件化修复过程，使输入对象接近入域域。结果是，重构的对象在ID情况下非常接近原始对象，在OOD情况下则相差较远，使得RONIN能够有效区分ID和OOD样本。通过大量实验证明，RONIN在零样本和非零样本设置下，相对于先前方法，在多个数据集上取得了具有竞争力的结果。

    Machine learning algorithms are increasingly provided as black-box cloud services or pre-trained models, without access to their training data. This motivates the problem of zero-shot out-of-distribution (OOD) detection. Concretely, we aim to detect OOD objects that do not belong to the classifier's label set but are erroneously classified as in-distribution (ID) objects. Our approach, RONIN, uses an off-the-shelf diffusion model to replace detected objects with inpainting. RONIN conditions the inpainting process with the predicted ID label, drawing the input object closer to the in-distribution domain. As a result, the reconstructed object is very close to the original in the ID cases and far in the OOD cases, allowing RONIN to effectively distinguish ID and OOD samples. Throughout extensive experiments, we demonstrate that RONIN achieves competitive results compared to previous approaches across several datasets, both in zero-shot and non-zero-shot settings.
    

