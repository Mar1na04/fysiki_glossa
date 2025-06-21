from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')


def compare_texts(original: str, reconstructed: str) -> float:
    emb1 = model.encode(original, convert_to_tensor=True)
    emb2 = model.encode(reconstructed, convert_to_tensor=True)
    return float(util.pytorch_cos_sim(emb1, emb2)[0])
