# LLM_Trainer
从零训练一个0.4B的大模型（灵犀大模型，寓意心有灵犀一点通）。代码包括了pretrain，sft，dpo等训练方式

## Lingxi-0.4B(灵犀大模型)
<details open> 
<summary>  <b>2024-10-02</b> </summary>
第一版使用的预训练数据较少，跑通整个流程，暂无身份信息，仅支持单轮对话。目前存在幻觉、重复等问题。<br>

- 预训练数据：使用约5B token进行预训练，训练约50个小时。 <br>
- sft数据：[firely-train](https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M) .使用Sorted batching方式进行微调<br>
- 执行脚本：[bash run.sh](https://github.com/wangru8080/LLM_Trainer/blob/main/run.sh)<br>
- loss:  
  pt-loss:
  ![](https://github.com/wangru8080/LLM_Trainer/blob/main/resource/loss-v1-pt.png)
  sft-loss:
  ![](https://github.com/wangru8080/LLM_Trainer/blob/main/resource/loss-v1-sft.png)
- case:
![](https://github.com/wangru8080/LLM_Trainer/blob/main/resource/case0.png)
![](https://github.com/wangru8080/LLM_Trainer/blob/main/resource/case1.png)
![](https://github.com/wangru8080/LLM_Trainer/blob/main/resource/case2.png)
![](https://github.com/wangru8080/LLM_Trainer/blob/main/resource/case3.png)
![](https://github.com/wangru8080/LLM_Trainer/blob/main/resource/case4.png)
</details>

<details close> 
<summary>  <b>2024-10-05</b> </summary>

使用约150B token进行预训练。进行中，由于资源的情况大约需要训练1400个小时
