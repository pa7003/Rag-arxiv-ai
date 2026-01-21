from src.generator import HuggingFaceGenerator

def test_generator_outputs_text():
    generator = HuggingFaceGenerator()
    
    query = "What is AI?"
    contexts = ["AI stands for artificial intelligence."]

    answer = generator.generate(query, contexts)

    assert isinstance(answer, str)
    assert len(answer) > 0
