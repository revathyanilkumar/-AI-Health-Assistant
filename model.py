from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_model():
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # smaller for CPU
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=None
    ).to("cpu")
    return model, tokenizer

def generate_response(model, tokenizer, user_input):
    # Strict single-answer prompt (prevents multiple turns)
    prompt = (
        f"You are a knowledgeable AI health assistant. "
        f"Answer the following question clearly and concisely. "
        f"Do not include any dialogue tags like 'User:' or 'AI:'. "
        f"Do not generate follow-up questions or conversations.\n\n"
        f"Question: {user_input}\n\n"
        f"Answer:"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=120,
        temperature=0.6,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    # Decode and clean output
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Answer:" in response:
        response = response.split("Answer:")[-1].strip()

    # Remove any accidental "User:" or "AI:"
    for tag in ["User:", "AI:", "Assistant:"]:
        response = response.replace(tag, "").strip()

    return response