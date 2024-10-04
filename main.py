import spacy
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


model = SentenceTransformer('bert-base-nli-mean-tokens')


ideal_answer = "Backpropagation adjusts the weights in a neural network by propagating the error backward from the output."
candidate_answer = "Backpropagation changes the weights based on the error it gets after predicting wrong in the neural net."


ideal_embedding = model.encode([ideal_answer])
candidate_embedding = model.encode([candidate_answer])


similarity_score = cosine_similarity(ideal_embedding, candidate_embedding)[0][0]
print(f'Similarity Score: {similarity_score}')
