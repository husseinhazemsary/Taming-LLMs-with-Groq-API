import os
import groq
from dotenv import load_dotenv

def load_api_key():
    """Loads API key from .env file"""
    load_dotenv()
    return os.getenv("GROQ_API_KEY")

class LLMClient:
    def __init__(self):
        self.api_key = load_api_key()
        self.client = groq.Client(api_key=self.api_key)
        self.model = "llama3-70b-8192"
    
    def complete(self, prompt, max_tokens=1000, temperature=0.7):
        """Basic completion function with error handling"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error: {e}")
            return None

def create_structured_prompt(text, question):
    """Creates a structured prompt for analysis."""
    return f"""
    # Analysis Report
    ## Input Text
    {text}
    ## Question
    {question}
    ## Analysis
    """

def extract_section(completion, section_start, section_end=None):
    """Extracts specific sections from completions."""
    start_idx = completion.find(section_start)
    if start_idx == -1:
        return None
    start_idx += len(section_start)
    if section_end is None:
        return completion[start_idx:].strip()
    end_idx = completion.find(section_end, start_idx)
    if end_idx == -1:
        return completion[start_idx:].strip()
    return completion[start_idx:end_idx].strip()

def classify_with_confidence(client, text, categories, confidence_threshold=0.8):
    """Classifies text into categories and checks confidence."""
    prompt = f"""
    Classify the following text into exactly one of these categories: {', '.join(categories)}.
    Response format:
    1. CATEGORY: [one of: {', '.join(categories)}]
    2. CONFIDENCE: [high|medium|low]
    3. REASONING: [explanation]
    Text to classify:
    {text}
    """
    response = client.complete(prompt)
    if response:
        category = extract_section(response, "1. CATEGORY: ", "\n")
        confidence_str = extract_section(response, "2. CONFIDENCE: ", "\n")
        reasoning = extract_section(response, "3. REASONING: ")
        confidence_score = {"high": 0.9, "medium": 0.6, "low": 0.3}.get(confidence_str, 0.0)
        if confidence_score >= confidence_threshold:
            return {"category": category, "confidence": confidence_score, "reasoning": reasoning}
    return {"category": "uncertain", "confidence": 0.0, "reasoning": "Confidence below threshold"}

def compare_prompt_strategies(client, texts, categories):
    """Compares different prompt strategies on the same classification tasks."""
    strategies = {
        "basic": lambda text: f"Classify this text: {text}\nAnswer:",
        "structured": lambda text: f"""
        Classification Task
        Categories: {', '.join(categories)}
        Text: {text}
        Classification: """,
        "few_shot": lambda text: f"""
        Here are some examples of text classification:
        Example 1:
        Text: "The product arrived damaged."
        Classification: Negative
        Example 2:
        Text: "Good product, but shipping was slow."
        Classification: Mixed
        Now classify this text:
        Text: "{text}"
        Classification: """
    }
    results = {}
    for strategy_name, prompt_func in strategies.items():
        results[strategy_name] = [classify_with_confidence(client, text, categories) for text in texts]
    return results

if __name__ == "__main__":
    # Initialize the LLM client
    client = LLMClient()

    # Sample test cases
    test_texts = [
        "I love this phone!",          # Expected: Positive
        "Terrible service.",           # Expected: Negative
        "It's okay, could be better.", # Expected: Mixed
        "This laptop is amazing!",     # Expected: Positive
        "The food was awful.",         # Expected: Negative
        "The movie was fine, not great but not bad.",  # Expected: Mixed
        "I absolutely adore this!",    # Expected: Positive
        "Horrible experience, never coming back.", # Expected: Negative
        "The product is decent but overpriced.", # Expected: Mixed
    ]

    categories = ["Positive", "Negative", "Mixed"]

    print("\n===== Testing Structured Prompt Creation =====")
    example_text = "The service was slow, but the food was good."
    example_question = "What is the sentiment of this statement?"
    structured_prompt = create_structured_prompt(example_text, example_question)
    print(structured_prompt)

    print("\n===== Testing Classification with Confidence =====")
    for text in test_texts:
        result = classify_with_confidence(client, text, categories)
        print(f"Text: \"{text}\"")
        print(f"Category: {result['category']} | Confidence: {result['confidence']}")
        print(f"Reasoning: {result['reasoning']}\n")

    print("\n===== Testing Prompt Strategy Comparison =====")
    results = compare_prompt_strategies(client, test_texts, categories)
    
    for strategy, output in results.items():
        print(f"\n### Strategy: {strategy} ###")
        for i, result in enumerate(output):
            print(f"Text: \"{test_texts[i]}\" -> Category: {result['category']} (Confidence: {result['confidence']})")
            print(f"Reasoning: {result['reasoning']}\n")
