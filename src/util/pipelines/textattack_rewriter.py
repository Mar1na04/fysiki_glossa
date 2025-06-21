from textattack.augmentation import WordNetAugmenter

augmenter = WordNetAugmenter()


def rewrite_text(text: str) -> str:
    sentences = text.split('.')
    augmented = []
    for s in sentences:
        aug = augmenter.augment(s)
        if aug:
            augmented.append(aug[0])
    return '. '.join(augmented) + '.'
