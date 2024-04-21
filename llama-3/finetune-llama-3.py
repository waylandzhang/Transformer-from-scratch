# More models at https://huggingface.co/unsloth
import torch
from unsloth import FastLanguageModel
from transformers import TextStreamer

max_seq_length = 2048  # 可以自定义任意窗口长度，已根据RoPE编码自动伸缩模型窗口尺寸了。
dtype = None  # 设置为None自动获取。目前 Float16 支持GPU类型：Tesla T4, V100； Bfloat16 支持GPU类型： Ampere+
load_in_4bit = True  # 使用4bit量化以减少内存使用。可以设置为False。

# 调用 unsloth 预先量化好的4bit模型
fourbit_models = [
    "unsloth/mistral-7b-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
    "unsloth/llama-2-7b-bnb-4bit",
    "unsloth/gemma-7b-bnb-4bit",
    "unsloth/gemma-7b-it-bnb-4bit",  # Instruct version of Gemma 7b
    "unsloth/gemma-2b-bnb-4bit",
    "unsloth/gemma-2b-it-bnb-4bit",  # Instruct version of Gemma 2b
    "unsloth/llama-3-8b-bnb-4bit",  # [NEW] 15 Trillion token Llama-3
]

# 调用 unsloth 预先量化好的4bit模型
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,  # 以4bit加载模型
)

# 以下是PEFT模型的默认参数，您可以根据需要进行调整。
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj", ],
    lora_alpha=16,
    lora_dropout=0,  # Supports any, but = 0 is optimized
    bias="none",  # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
    random_state=3407,
    use_rslora=False,  # We support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
)

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token  # 必须添加 EOS_TOKEN 这个特殊符号，否则生成会无限循环。。


def formatting_prompts_func(examples):
    instructions = examples["instruction_zh"]
    inputs = examples["input_zh"]
    outputs = examples["output_zh"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN  # 必须添加 EOS_TOKEN 这个特殊符号，否则生成会无限循环。
        texts.append(text)
    return {"text": texts, }


# 从数据集中加载数据
from datasets import load_dataset

# 试验可以用这个数据集：
# https://huggingface.co/datasets/silk-road/alpaca-data-gpt4-chinese
dataset = load_dataset("silk-road/alpaca-data-gpt4-chinese", split="train")
dataset = dataset.map(formatting_prompts_func, batched=True, )

# 以下是PEFT模型的默认参数，您可以根据需要进行调整。
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,  # 可以让小上下文窗口训练速度增加5倍以上
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=500,  # 微调循环次数
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=1337,
        output_dir="outputs",
    ),
)

# 训练模型
trainer_stats = trainer.train()

# 推理
FastLanguageModel.for_inference(model)  # Enable native 2x faster inference
inputs = tokenizer(
    [
        alpaca_prompt.format(
            "你中文回答问题",  # instruction
            "植物是如何呼吸的？",  # input
            "",  # output - leave this blank for generation!
        )
    ], return_tensors="pt").to("cuda")


# 使用Stream流生成文本
text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=128)

# 储存微调后的模型
model.save_pretrained("llama-3-zh_lora")  # 保存在本地文件夹 llama-3-zh_lora
