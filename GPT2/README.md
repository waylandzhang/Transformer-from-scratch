# Extra: Sample code showing fine-tuning GPT2 model

This subdirectory is used separately from root directory where I put some sample code to show how to fine-tune a pre-trained GPT2 model, as well as how to inference from it.

The best way to fine-tune a pre-trained model is to use the `transformers` library from HuggingFace.

To begin, you need to install the following packages:

```bash
pip install transformers 
```

Run `finetune-gpt2.py` will first download the GPT2 model file from HuggingFace _(probably be saved to your home directory at ~/.cache/huggingface/hub)_, then fine-tune it with a small dataset I included in this directory named `mental_health_data.txt`, and finally save the fine-tuned model to a new directory named `output`.

Once you fine-tuned, you can run `inference-gpt2.py` to generate some text from the fine-tuned model.

_**Notice: you probably need a GPU to fine-tune the model, otherwise it will take a long time to finish.**_


