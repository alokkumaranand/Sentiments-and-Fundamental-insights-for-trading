"""Feature layer: the frozen lexicon, the sentiment and attention families, price features.

`features/` is deterministic code only. Nothing here calls an LLM; the LLM's single
low-level contribution arrives as a frozen lexicon file that this layer applies at scale.
"""
