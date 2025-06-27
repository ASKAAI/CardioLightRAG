# generation/biomed_generator.py
from transformers import AutoTokenizer, AutoModelForCausalLM

class BiomedGenerator:
    def __init__(self):
        """
        Initialize the biomedical generator with a compatible model and tokenizer.
        """
        print("Loading AutoTokenizer and AutoModelForCausalLM...")
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.model = AutoModelForCausalLM.from_pretrained("gpt2")

    def set_generation_parameters(self, attention_mask=None, pad_token_id=None):
        """Set generation parameters for the model."""
        self.attention_mask = attention_mask
        self.pad_token_id = pad_token_id
        print(f"Generation parameters set: attention_mask={attention_mask}, pad_token_id={pad_token_id}")

    def generate_answer(self, context, query):
        """
        Generate an answer to the given query based on the provided context.

        Args:
            context (str): The context information for the query
            query (str): The clinical question or query text

        Returns:
            str: Generated answer text
        """
        input_text = f"{context}\n\nQuestion: {query}\nAnswer:"
        inputs = self.tokenizer.encode(input_text, return_tensors="pt")
        outputs = self.model.generate(inputs, max_length=200, num_return_sequences=1)
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer

if __name__ == "__main__":
    generator = BiomedGenerator()
    context = "Angina is chest pain caused by reduced blood flow to the heart."
    question = "What are the first-line treatments for angina?"
    print(generator.generate_answer(context, question))