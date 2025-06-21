from util.texts import text1, text2
from util.pipelines.custom_rewriter import rewrite_sentences, custom_rewrite
from util.pipelines.huggingface_rewriter import paraphrase_text
from util.pipelines.textattack_rewriter import rewrite_text
from util.pipelines.similarity import compare_texts

from analysis.preprocess import tokenize
from analysis.embeddings import get_average_embedding
from analysis.visualize import visualize_embeddings

# Χρήση sentence-transformers
from sentence_transformers import SentenceTransformer, util

bert_model = SentenceTransformer('all-MiniLM-L6-v2')


# Παραδοτέο 1
def run_comparison(name: str, func, original_text: str):
    print(f"\n// {name.upper()} //")
    reconstructed = func(original_text)
    print("Reconstructed Text:\n", reconstructed)
    similarity = compare_texts(original_text, reconstructed)
    print("Cosine Similarity:", round(similarity, 4))


def demo_two_sentences():
    print("/ REWRITE 2 SELECTED SENTENCES (PART A) /")
    sentences = [
        "Thank your message to show our words to the doctor.",
        "I am very appreciated the full support of the professor."
    ]
    for i, s in enumerate(sentences, 1):
        rewritten = custom_rewrite(s)
        print(f"Original {i}: {s}")
        print(f"Rewritten {i}: {rewritten}")
        print("-----")


def full_reconstruction(text: str, label: str):
    print(f"\n\n/ {label} - FULL TEXT RECONSTRUCTION (PART B & C) /")

    run_comparison("Custom Rewrite", rewrite_sentences, text)
    run_comparison("HuggingFace T5", paraphrase_text, text)
    run_comparison("TextAttack", rewrite_text, text)


# Παραδοτέο 2
def analyze_similarity(original, rewritten):
    tok1 = tokenize(original)
    tok2 = tokenize(rewritten)

    emb1 = get_average_embedding(tok1, bert_model, model_type='sentence')
    emb2 = get_average_embedding(tok2, bert_model, model_type='sentence')

    cosine = float(util.cos_sim(emb1, emb2))
    print("→ Cosine Similarity (token-level avg.):", round(cosine, 4))

    visualize_embeddings([emb1, emb2], ["Original", "Rewritten"], method='pca', title='PCA Visual')
    visualize_embeddings([emb1, emb2], ["Original", "Rewritten"], method='tsne', title='t-SNE Visual')


# Εφαρμόζεται μόνο σε ένα pipeline για κάθε κείμενο (π.χ. HuggingFace)
def run_semantic_analysis():
    print("\n/ SEMANTIC EMBEDDING ANALYSIS TEXT 1 /")
    rew1 = paraphrase_text(text1)
    analyze_similarity(text1, rew1)

    print("\n/ SEMANTIC EMBEDDING ANALYSIS TEXT 2 /")
    rew2 = paraphrase_text(text2)
    analyze_similarity(text2, rew2)


if __name__ == "__main__":
    # Ανακατασκευή 2 προτάσεων (Part A)
    demo_two_sentences()

    # Ανακατασκευή ολόκληρου κειμένου (Part B & C)
    full_reconstruction(text1, "TEXT 1")
    full_reconstruction(text2, "TEXT 2")

    # Ανάλυση ομοιότητας με embeddings (Παραδοτέο 2)
    run_semantic_analysis()
