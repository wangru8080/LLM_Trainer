# LLM_Trainer
从零训练一个0.4B的大模型（灵犀大模型，寓意心有灵犀一点通）。代码包括了pretrain，sft，dpo等训练方式

## Lingxi-0.4B(灵犀大模型)
<details close> 
<summary>  <b>2024-10-02</b> </summary>
第一版使用的预训练数据较少，跑通整个流程，暂无模型身份信息，仅支持单轮对话。目前存在幻觉、重复等问题。  

- huggingface模型下载：[Lingxi-0.4B-Instruct](https://huggingface.co/wangru8080/Lingxi-0.4B-Instruct)
- 词表：直接使用了qwen的词表
- 预训练：使用约5B token进行预训练，训练约50个小时。   
- sft：训练数据[firely-train](https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M) .使用Sorted batching方式进行微调  
- 执行脚本：[bash run.sh](https://github.com/wangru8080/LLM_Trainer/blob/main/run.sh)  
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

<details open> 
<summary>  <b>2024-10-05</b> </summary>

使用约150B token进行预训练。进行中，由于资源的情况大约需要训练1400个小时  

预训练迭代1个epoch后,sft训练了1.5个epoch：  
![](https://github.com/wangru8080/LLM_Trainer/blob/main/resource/case5.png)
![](https://github.com/wangru8080/LLM_Trainer/blob/main/resource/case6.png)
![](https://github.com/wangru8080/LLM_Trainer/blob/main/resource/case7.png)
![](https://github.com/wangru8080/LLM_Trainer/blob/main/resource/case8.png)
![](https://github.com/wangru8080/LLM_Trainer/blob/main/resource/case9.png)
![](https://github.com/wangru8080/LLM_Trainer/blob/main/resource/case10.png)
![](https://github.com/wangru8080/LLM_Trainer/blob/main/resource/case11.png)
初步具备多轮聊天和代码的功能
</details>
