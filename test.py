from transformers import AutoModelForCausalLM, AutoTokenizer

# Ordnerpfad, in dem die .safetensors-Dateien liegen
model_path = "./mein_lokales_modell" 

# Lädt das Modell direkt aus den Safetensors
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Text generieren
inputs = tokenizer("Hallo, wie geht es dir?", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
