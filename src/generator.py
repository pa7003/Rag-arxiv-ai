from transformers import pipeline
from typing import List
import logging

logger = logging.getLogger(__name__)

class HuggingFaceGenerator:
    """
    LLM-based generator using HuggingFace Transformers.

    This class generates grounded answers for a user query
    using retrieved document context.
    """

    def __init__(self, model_name: str = "google/flan-t5-base", max_length: int = 256):
        """
        Initialize the HuggingFace text generation pipeline.

        Args:
            model_name: Name of the HuggingFace model to use.
            max_length: Maximum length of generated output.
        """
        self.llm = pipeline(
            "text2text-generation",
            model=model_name,
            max_length=max_length
        )

    def generate(self, query: str, contexts: List[str]) -> str:
        """
        Generate an answer for the given query using retrieved contexts.

        Args:
            query: User query string.
            contexts: List of retrieved document strings.

        Returns:
            Generated answer as a string.
        """
        if not query or not contexts:
            raise ValueError("Query and contexts must be provided")

        context_text = " ".join(contexts)[:2000]

        prompt = f"""
        Answer the question using the context.

        Context:
        {context_text}

        Question:
        {query}
        """

        try:
            return self.llm(prompt)[0]["generated_text"]
        except Exception:
            logger.error("LLM generation failed", exc_info=True)
            raise
