# Prompt templates for zero-shot CLIP classification
# Each template will be combined with species names to form text queries.

PROMPT_TEMPLATES = [
    "a photo of {}",
    "a macro shot of a {} fungus",
    "the mushroom {}",
    "{} growing in the wild",
    "a close-up photograph of {}",
]


def get_prompts_for_classes(class_names, template="a photo of {}"):
    """Generate prompt texts for all class names using a given template."""
    return [template.format(name) for name in class_names]


def get_all_prompt_variants(class_names):
    """Return dict mapping template index -> list of prompt texts."""
    return {
        f"template_{i}": get_prompts_for_classes(class_names, t)
        for i, t in enumerate(PROMPT_TEMPLATES)
    }
