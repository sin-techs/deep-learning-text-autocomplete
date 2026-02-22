from transformers import pipeline
from transformers import logging
logging.set_verbosity_error()

generator = pipeline("text-generation", model="distilgpt2")

def generate_autocomplete_gpt2(model,tokenizer,prompt, max_new_tokens=5, 
                         temperature=0.8, top_k=50, device='cuda',max_length=63):
    result = generator(prompt, max_new_tokens=max_new_tokens, max_length=None, do_sample=True, top_k=50,truncation=True,repetition_penalty=1.5)
    return result[0]["generated_text"] 
    

# print(generate_autocomplete_gpt2(None,None,"hello world",10))