import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from peft import PeftModel

def make_prompt(text: str) -> str:
    # This MUST match the exact format used during your LoRA training
    return f"""
You are a job classification system.

Read the resume below and return ONLY one word representing the job category.
Choose from: engineering, finance, healthcare, education, tech, sales-marketing, business-hr, creative-media-design, hospitality-food, aviation-transport, construction, other

Resume:
{text[:1024]}

Category:
"""

def load_model(model_path: str = "./flan-t5-lora-finetuned"):
    base_model_name = "google/flan-t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    base_model = T5ForConditionalGeneration.from_pretrained(
        base_model_name, 
        torch_dtype=torch.float32
    )
    model = PeftModel.from_pretrained(base_model, model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    return model, tokenizer, device