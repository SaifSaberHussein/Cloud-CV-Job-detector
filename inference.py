import torch
from model_loader import make_prompt

def run_inference(resume_text: str, model, tokenizer, device: str) -> dict:
    prompt = make_prompt(resume_text)
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=5, 
            num_beams=2, 
            return_dict_in_generate=True, 
            output_scores=True
        )

    # Decode the single-word category
    raw_output = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True).strip().lower()
    
    # Calculate Confidence (Softmax of the first token generated)
    probs = torch.nn.functional.softmax(outputs.scores[0], dim=-1)
    conf = torch.max(probs).item()

    return {
        "raw_output": raw_output,
        "extracted": {"Main Job": raw_output},
        "confidence": conf
    }