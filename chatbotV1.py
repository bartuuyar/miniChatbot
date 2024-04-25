import random
import numpy as np
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('emrecan/bert-base-turkish-cased-mean-nli-stsb-tr')


intents = {
    "greetings": {
        "merhaba": "Merhaba! Nasıl yardımcı olabilirim?",
        "hello": "Merhaba! Sizin için ne yapabilirim?",
        "hey": "Hey, nasılsın?",
        "günaydın: "Günaydın! Size nasıl yardım edebilirim?"
    },
    "questions": {
        "Ms. Johnson'ın telefon numarası nedir?": "Ms. Johnson'ın numarası 555-1234",
        "Bay Smith'in telefon numarasını öğrenebilir miyim?": "Bay Smith'in numarası 555-5678",
        "Dr. Patel'ın telefonu nedir?": "Dr. Patel'ın telefonu 555-9012"
    }
}

def respond(user_input):
    best_match_type = None
    best_match_similarity = 0
    best_match_response = None

    
    for greeting_phrase, response in intents["greetings"].items():
        similarity_score = 1 - cosine(model.encode(user_input), model.encode(greeting_phrase))
        if similarity_score > best_match_similarity:
            best_match_similarity = similarity_score
            best_match_response = response
            best_match_type = "greeting"

     
    for question, answer in intents["questions"].items():
        similarity_score = 1 - cosine(model.encode(user_input), model.encode(question))
        if similarity_score > best_match_similarity:
            best_match_similarity = similarity_score
            best_match_response = answer
            best_match_type = "question"

    )
    if best_match_similarity >= 0.5:  
        print(f"Best match type: {best_match_type}, Similarity: {best_match_similarity}")
        return best_match_response
    else:
        return "Üzgünüm, buna nasıl cevap vereceğimi henüz bilmiyorum. Başka bir şey sormayı deneyebilir misin?"

    


while True:
    user_input = input("Sen: ")
    if user_input.lower() == "q":
        break
    response = respond(user_input)
    print("Bot:", response)
