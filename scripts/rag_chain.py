import os
import torch

from huggingface_hub import login
hf_token = os.environ.get("HF_TOKEN")
if hf_token is None:
    raise ValueError("Missing HF_TOKEN environment variable. Please set it to your Hugging Face token.")
login(hf_token)

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig
from retriever import top_k

#model info
MODEL_NAME = "google/flan-t5-large"
device = "cuda" if torch.cuda.is_available() else "cpu"

#load tokenizer and model
print('Loading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

print('Loading model...')
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
)

#generation config
generation_config = GenerationConfig(
    do_sample=False,
    max_new_tokens=512
)

# Test the model with a simple prompt first
test_input = tokenizer("Hello, how are you?", return_tensors="pt").to(device)
with torch.no_grad():
    test_output = model.generate(**test_input, max_new_tokens=20, do_sample=False)
test_text = tokenizer.decode(test_output[0], skip_special_tokens=True)
print(f"Model test output: {test_text}")

#combine user query with top_k query to feed to model
def build_prompt(question, top_chunks):
    context = [c['full_text'] for c in top_chunks]
    context = "\n\n".join(context)
    prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    print(prompt)
    return prompt

def answer_question(question, k=3):
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
    return output_text.strip()

    #extract answer from output
    # if "ANSWER:" in output_text:
    #     answer = output_text.split("ANSWER:")[-1].strip()
    # else:
    #     answer = output_text[len(prompt_text):].strip()

    # return answer

#cli
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python rag_chain.py \"Your question here\" [top_k]")
        sys.exit(1)
    question = sys.argv[1]
    answer = answer_question(question, k=int(sys.argv[2]) if len(sys.argv) > 2 else 3)
    print("\n==== Answer ===\n")
    print(answer)









