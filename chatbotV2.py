from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
import re
import numpy as np
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer


sentence_model = SentenceTransformer('emrecan/bert-base-turkish-cased-mean-nli-stsb-tr')


ner_model = AutoModelForTokenClassification.from_pretrained("akdeniz27/bert-base-turkish-cased-ner")
ner_tokenizer = AutoTokenizer.from_pretrained("akdeniz27/bert-base-turkish-cased-ner")
ner_pipeline = pipeline('ner', model=ner_model, tokenizer=ner_tokenizer, aggregation_strategy="first")

def calculate_name_similarity(input_name, target_names, method="cosine"):

    similarity_scores = []
    for name in target_names:
        
        embedding1 = sentence_model.encode(input_name)
        embedding2 = sentence_model.encode(name)
        score = 1 - cosine(embedding1, embedding2)
        similarity_scores.append((name, score))

    return similarity_scores



intents = {
    "greetings": {
        "merhaba": "Merhaba! Nasıl yardımcı olabilirim?",
        "hello": "Merhaba! Sizin için ne yapabilirim?",
        "hey": "Hey, nasılsın?",
        "günaydın": "Günaydın! Size nasıl yardım edebilirim?"
    }
}

teacher_phone_numbers = {
    "Ahmet": "555-1234",
    "Onur": "555-5678",
    "Barış": "555-9012"
}


def respond(user_input):
    best_match_type = None
    best_match_similarity = 0.5
    best_match_response = None

    
    for greeting_phrase, response in intents["greetings"].items():
        similarity_score = 1 - cosine(sentence_model.encode(user_input), sentence_model.encode(greeting_phrase))
        if similarity_score > best_match_similarity:
            best_match_similarity = similarity_score
            best_match_response = response
            best_match_type = "greeting"

    
    if is_phone_question(user_input):
        teacher_name = extract_teacher_name(user_input)
        if teacher_name:  
            similarity_results = calculate_name_similarity(teacher_name, teacher_phone_numbers.keys())
            filtered_results = [(name, score) for name, score in similarity_results if score >= 0.4]

            if filtered_results:
                most_similar_name = max(filtered_results, key=lambda item: item[1])[0]
                phone_number = teacher_phone_numbers[most_similar_name]
                best_match_response = f"{most_similar_name} telefon numarası: {phone_number}"
            else:
                best_match_response = f"{teacher_name} numarsına sahip değilim."
        else:
            best_match_response = "Hangi öğretmenin numarasını aradığınızı anlayamadım."

    
    if not best_match_response:
        best_match_response = "Üzgünüm, buna nasıl cevap vereceğimi henüz bilmiyorum. Başka bir şey sormayı deneyebilir misin?"

    return best_match_response

def is_phone_question(user_input):
    phone_keywords = ["phone number","numara","telefon"]  
    return any(word.lower() in user_input.lower() for word in phone_keywords)

def extract_teacher_name(user_input):
    entities = ner_pipeline(user_input)
    for entity in entities:
        if entity['entity_group'] == 'PER':
            return entity['word']
    return None  


while True:
    user_input = input("Sen: ")
    if user_input.lower() == "q":
        break
    response = respond(user_input)
    print("Bot:", response)
