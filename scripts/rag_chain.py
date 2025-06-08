import os
import torch

from huggingface_hub import login
hf_token = os.environ.get("HF_TOKEN")
if hf_token is None:
    raise ValueError("Missing HF_TOKEN environment variable. Please set it to your Hugging Face token.")
login(hf_token)

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from retriever import top_k

#model info
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4"
device = "cuda" if torch.cuda.is_available() else "cpu"

#load tokenizer and model
print('Loading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

print('Loading model...')
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
)

#generation config
generation_config = GenerationConfig(
    temperature=0.0, #0.0 for deterministic output
    top_p=1.0,       #ignored if do sample false
    dosample=False,
    max_new_tokens=512
)

#combine user query with top_k query to feed to model
def build_prompt(question, top_chunks):
    context_blocks = "\n\n".join([c['preview'] for c in top_chunks])
    prompt = (
        "You are a helpful assistant."
        "Use the following CONTEXT to answer the question."
        "If the answer cannot be found in the context, respond with 'I don't know'.\n\n"
        "CONTEXT:\n"
        f"{context_blocks}\n\n"
        "QUESTION:\n"
        f"{question}\n\n"
        "ANSWER:"
    )
    return prompt

def answer_question(question, k=5):
    #retrieve top k chunks
    top_chunks = top_k(question, k=k)
    if not top_chunks:
        return "No relevant information found."
    
    #create prompt text
    prompt_text = build_prompt(question, top_chunks)

    #tokenize prompt
    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=tokenizer.model_max_length-generation_config.max_new_tokens
    ).to(device)

    #generate response
    with torch.autocast(device_type=device):
        outputs = model.generate(
            **inputs,
            generation_config=generation_config,
            pad_token_id=tokenizer.eos_token_id
        )
    
    #decode response
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    #extract answer from output
    if "ANSWER:" in output_text:
        answer = output_text.split("ANSWER:")[-1].strip()
    else:
        answer = output_text[len(prompt_text):].strip()

    return answer

#cli
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python rag_chain.py \"Your question here\" [top_k]")
        sys.exit(1)
    question = sys.argv[1]
    answer = answer_question(question, k=int(sys.argv[2]) if len(sys.argv) > 2 else 5)
    print("\n==== Answer ===\n")
    print(answer)









