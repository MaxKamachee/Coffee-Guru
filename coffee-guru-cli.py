import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def load_barista_model():
    model_path = "./coffee_barista_model"
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    return tokenizer, model

def generate_recipe(prompt, tokenizer, model):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
    
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=300,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def extract_recipe(response):
    recipe_start = response.find("Recipe:")
    if recipe_start != -1:
        return response[recipe_start:]
    else:
        return "I'm sorry, I couldn't generate a specific recipe. Could you please try asking in a different way?"

def main():
    tokenizer, model = load_barista_model()
    
    print("Welcome to the AI Barista! I can suggest coffee recipes based on your preferences.")
    
    while True:
        user_input = input("\nWhat kind of coffee would you like? (or type 'quit' to exit): ").strip()
        if user_input.lower() == 'quit':
            break
        
        prompt = f"Customer: {user_input}\nBarista: Certainly! I'd recommend trying this recipe:\n\nRecipe:"
        response = generate_recipe(prompt, tokenizer, model)
        recipe = extract_recipe(response)
        
        print("\nHere's a recipe suggestion for you:\n")
        print(recipe)

    print("Thank you for using the AI Barista. Enjoy your coffee!")

if __name__ == "__main__":
    main()