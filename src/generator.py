from transformers import pipeline

class HuggingFaceGenerator:
    def __init__(self):
        self.llm = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            max_length=256
        )

    def generate(self, query, contexts):
        context_text = " ".join(contexts)[:2000]
        prompt = f"""
        Answer the question using the context.

        Context:
        {context_text}

        Question:
        {query}
        """
        return self.llm(prompt)[0]["generated_text"]
