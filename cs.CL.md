# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Prompting4Debugging: Red-Teaming Text-to-Image Diffusion Models by Finding Problematic Prompts.](http://arxiv.org/abs/2309.06135) | 提出了Prompting4Debugging（P4D）作为一个调试和红队测试工具，可以自动找到扩散模型的问题提示，以测试部署的安全机制的可靠性。 |

# 详细

[^1]: Prompting4Debugging: 通过发现问题提示来对文本到图像扩散模型进行红队测试

    Prompting4Debugging: Red-Teaming Text-to-Image Diffusion Models by Finding Problematic Prompts. (arXiv:2309.06135v1 [cs.CL])

    [http://arxiv.org/abs/2309.06135](http://arxiv.org/abs/2309.06135)

    提出了Prompting4Debugging（P4D）作为一个调试和红队测试工具，可以自动找到扩散模型的问题提示，以测试部署的安全机制的可靠性。

    

    文本到图像扩散模型，例如稳定扩散（SD），最近展现出高质量内容生成的显著能力，并成为近期变革性人工智能浪潮的代表之一。然而，这种进步也带来了对该生成技术滥用的日益关注，特别是用于生成受版权保护或不适合在工作环境中查看的图像。虽然已经做出了一些努力来通过模型微调来过滤不适当的图像/提示或删除不希望的概念/风格，但这些安全机制对于多样化的问题提示的可靠性仍然不清楚。在这项工作中，我们提出了Prompting4Debugging（P4D）作为一个调试和红队测试工具，它可以自动找到扩散模型的问题提示，以测试部署的安全机制的可靠性。我们展示了我们的P4D工具在发现具有安全机制的SD模型的新漏洞方面的有效性。具体而言，我们的结果显示...

    Text-to-image diffusion models, e.g. Stable Diffusion (SD), lately have shown remarkable ability in high-quality content generation, and become one of the representatives for the recent wave of transformative AI. Nevertheless, such advance comes with an intensifying concern about the misuse of this generative technology, especially for producing copyrighted or NSFW (i.e. not safe for work) images. Although efforts have been made to filter inappropriate images/prompts or remove undesirable concepts/styles via model fine-tuning, the reliability of these safety mechanisms against diversified problematic prompts remains largely unexplored. In this work, we propose Prompting4Debugging (P4D) as a debugging and red-teaming tool that automatically finds problematic prompts for diffusion models to test the reliability of a deployed safety mechanism. We demonstrate the efficacy of our P4D tool in uncovering new vulnerabilities of SD models with safety mechanisms. Particularly, our result shows t
    

