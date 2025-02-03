import os
import string
from collections import Counter
from fann2 import libfann

# Extraire les fréquences des caractères
def extract_features(text):
    text = text.lower()
    count = Counter(char for char in text if char in string.ascii_lowercase)
    total = sum(count.values())
    return [count.get(char, 0) / total for char in string.ascii_lowercase]

# Charger les fichiers et générer les données
def load_data(file_list, label):
    data = []
    for filepath in file_list:
        with open(filepath, 'r', encoding='utf-8') as file:
            text = file.read()
            features = extract_features(text)
            data.append((features, label))
    return data

# Liste des fichiers et leurs labels
french_files = ["data/french1.txt", "data/french2.txt", "data/french3.txt"]
english_files = ["data/english1.txt", "data/english2.txt", "data/english3.txt"]
spanish_files = ["data/spanish1.txt", "data/spanish2.txt", "data/spanish3.txt"]

# Préparer les données d'apprentissage
french_data = load_data(french_files, [1, 0, 0])  # Français
english_data = load_data(english_files, [0, 1, 0])  # Anglais
spanish_data = load_data(spanish_files, [0, 0, 1])  # Espagnol
train_data = french_data + english_data + spanish_data
inputs = [item[0] for item in train_data]
outputs = [item[1] for item in train_data]

# Créer les données FANN
training_data = libfann.training_data()
training_data.set_train_data(inputs, outputs)
training_data.save_train("language.data")

# Créer et entraîner le réseau
ann = libfann.neural_net()
ann.create_standard_array([26, 10, 3])  # 26 entrées (lettres a-z), 10 neurones cachés, 3 sorties (français, anglais, espagnol)
ann.set_learning_rate(0.7)
ann.set_activation_function_hidden(libfann.SIGMOID_SYMMETRIC)
ann.set_activation_function_output(libfann.SIGMOID)
print("\nEntrainement en cours...")
ann.train_on_file("language.data", max_epochs=5000, epochs_between_reports=100, desired_error=0.01)
print("Entrainement termine !")

# Tester le réseau avec une entrée utilisateur
languages = ["Francais", "Anglais", "Espagnol"]

while True:
    user_input = input("\n Tapez une phrase ou un mot en francais ou en anglais ou en espagnol (ou 'exit' pour quitter) : ")
    if user_input.lower() == "exit":
        print("Au revoir !")
        break

    features = extract_features(user_input)
    output = ann.run(features)
    predicted_lang = languages[output.index(max(output))]

    print(f"\nTexte saisi : {user_input}")
    print(f"Sorties : {output}")
    print(f"Langue predite : {predicted_lang}\n")
