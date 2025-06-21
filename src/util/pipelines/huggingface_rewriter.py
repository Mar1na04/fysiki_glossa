from transformers import pipeline

# T5 μπορεί να κάνει paraphrasing
paraphraser = pipeline("text2text-generation", model="Vamsi/T5_Paraphrase_Paws")


def paraphrase_text(text: str) -> str:
    sentences = text.strip().split('.')
    output = []
    for s in sentences:
        s = s.strip()
        if s:
            result = paraphraser(s, max_new_tokens=256, do_sample=True, top_k=50, top_p=0.95)[0]['generated_text']
            output.append(result)
    return ' '.join(output)
