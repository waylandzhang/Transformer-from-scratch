# 中文微调llama-3

目前最省GPU最快的方法微调llama-3就是通过usloth的方法，这个方法是在llama-3的基础上，他们预先量化到了4bit，减少微调时所需的内存。

这个方法的优点是，不需要重新训练模型，只需要下载预训练模型，然后微调即可。

这个方法的缺点是，由于量化到了4bit，所以模型的精度会有所下降，但是由于llama-3本身的精度就很高，所以这个下降是可以接受的。

#### 安装依赖
```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install torch transformers
pip install --no-deps packaging ninja einops flash-attn xformers trl peft accelerate bitsandbytes
```

这里的`flash-attn`很多机器是不支持的，那么可以直接注释掉`packaging ninja einops flash-attn`这几个库，不影响使用。


大概率需要一台GPU来运行，如果没有GPU，可以使用Colab，但是Colab的GPU可能会被限制，所以可能会出现OOM的情况。



