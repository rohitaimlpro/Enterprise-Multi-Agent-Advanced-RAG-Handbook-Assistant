from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

MODEL_NAME = "google/flan-t5-small"

_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

_model.eval()


def query_rewrite_agent(user_query: str, intent: str) -> str:
    prompt = f"""
Rewrite this employee handbook query into a short retrieval query.

Rules:
- keep it short
- include keywords
- include synonyms
- remove filler words
- do NOT answer

Intent: {intent}
Query: {user_query}

Rewritten query:
""".strip()

    inputs = _tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

    with torch.no_grad():
        output = _model.generate(
            **inputs,
            max_new_tokens=40,
            num_beams=4,
            do_sample=False
        )

    rewritten = _tokenizer.decode(output[0], skip_special_tokens=True).strip()

    # fallback safety
    if len(rewritten) < 3:
        return user_query

    return rewritten