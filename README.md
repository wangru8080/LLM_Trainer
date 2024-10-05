# LLM_Trainer
从零训练一个0.4B的大模型（灵犀大模型）。代码包括了pretrain，sft，dpo等训练方式

## Lingxi-0.4B(灵犀大模型)
<details open> 
<summary>  <b>2024-10-02</b> </summary>
第一版使用的预训练数据较少，跑通整个流程，暂无身份信息，仅支持单轮对话<br>
- 预训练数据：[firefly-pretrain-dataset](https://huggingface.co/datasets/YeungNLP/firefly-pretrain-dataset)。 大约有5B token <br>
- sft数据：[firely-train](https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M)<br>
- 执行脚本：[bash run.sh](https://github.com/wangru8080/LLM_Trainer/blob/main/run.sh)<br>
- case:

![](https://github.com/wangru8080/LLM_Trainer/blob/main/resource/case0.png)
![](https://github.com/wangru8080/LLM_Trainer/blob/main/resource/case1.png)
![](https://github.com/wangru8080/LLM_Trainer/blob/main/resource/case2.png)
![](https://github.com/wangru8080/LLM_Trainer/blob/main/resource/case3.png)

<details close> 
<summary>  <b>2024-10-05</b> </summary>
