import re


def custom_rewrite(sentence: str) -> str:
    sentence = sentence.strip()
    sentence = re.sub(r'\s+', ' ', sentence)
    sentence = sentence.replace("Thank your message", "Thank you for your message")
    sentence = sentence.replace("I am very appreciated", "I really appreciate")
    return sentence


def rewrite_sentences(text: str) -> str:
    sentences = re.split(r'\.\s*', text)
    rewritten = [custom_rewrite(s) for s in sentences if s]
    return '. '.join(rewritten) + '.'
