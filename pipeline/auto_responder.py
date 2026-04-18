"""
Auto Responder Module
Generates empathetic responses to product reviews.
"""
import logging

logger = logging.getLogger(__name__)

def template_response(product_name: str, tags: list[str]) -> str:
    """Template-based auto-response when LLM is unavailable."""
    issues = ", ".join(tags[:3]) if tags else "the issue you reported"
    return (
        f"We sincerely apologise for your experience with {product_name}, "
        f"particularly regarding {issues}. Our team is looking into this and "
        f"will follow up with you shortly. Thank you for your feedback."
    )

def generate_auto_response(text: str, product_name: str, category: str, rating: int, tags: list[str], call_mistral_fn) -> tuple[str, bool]:
    prompt = (
        "You are a professional customer support representative for an e-commerce platform. "
        "A customer has left a negative review. Generate a polite, empathetic response that:\n"
        "1. Acknowledges their specific complaint\n"
        "2. Apologises for the issue\n"
        "3. Assures them the team is investigating\n"
        "4. Offers to resolve the issue\n\n"
        f"Product: {product_name}\n"
        f"Category: {category}\n"
        f"Customer rating: {rating}/5\n"
        f"Customer review: \"{text}\"\n"
        f"Issues identified: {', '.join(tags)}\n\n"
        "Generate a response in 2-3 sentences. Be professional and concise.\n\n"
        "Response:"
    )

    result = call_mistral_fn(prompt)

    if result and len(result) > 20:
        response_text = result.strip().strip('"')
        return response_text, True
    else:
        logger.info("Mistral unavailable — using template response.")
        return template_response(product_name, tags), False
