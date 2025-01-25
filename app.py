from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")

input_text = "What is the future of AI in health care?"
inputs = tokenizer(input_text, return_tensors='pt')
outputs = model.generate(**inputs)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))