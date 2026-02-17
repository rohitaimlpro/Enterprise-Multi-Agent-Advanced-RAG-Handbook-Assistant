import re


def clean_handbook_text(text: str) -> str:
    """
    Cleans handbook PDF extracted text:
    - remove repeated whitespace
    - normalize newlines
    - remove junk characters
    """
    if not text:
        return ""

    # Replace multiple spaces with single space
    text = re.sub(r"[ \t]+", " ", text)

    # Normalize newlines (avoid huge blank gaps)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove weird repeated bullets
    text = text.replace("•", "-")

    # Remove page artifacts like "Page 12" (optional)
    text = re.sub(r"\bPage\s+\d+\b", "", text, flags=re.IGNORECASE)

    # Remove extra leading/trailing whitespace
    text = text.strip()

    return text