# LLM_Trainer
从零训练一个0.4B的大模型（灵犀大模型）。代码包括了pretrain，sft，dpo等训练方式

## Lingxi-0.4B(灵犀大模型)
<details open> 
<summary>  <b>2024-10-02</b> </summary>
第一版预训练数据较少，主要是为了跑通流程<br>
- 预训练数据：[firefly-pretrain-dataset](https://huggingface.co/datasets/YeungNLP/firefly-pretrain-dataset)。差不多有5B token<br>
- sft数据：[firely-train](https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M)<br>
- 执行脚本：[bash run.sh](https://github.com/wangru8080/LLM_Trainer/blob/main/run.sh)<br>
- case:

<details close> 
<summary>  <b>2024-10-05</b> </summary>
