from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Global cache for embeddings
EMBEDDING_CACHE = {}

STOP_WORDS = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
              'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him',
              'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its',
              'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what',
              'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am',
              'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
              'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the',
              'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
              'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
              'through', 'during', 'before', 'after', 'above', 'below', 'to',
              'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
              'again', 'further', 'then', 'once', 'here', 'there', 'when',
              'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
              'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
              'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't',
              'can', 'will', 'just', 'don', 'should', 'now']
LEMMATIZER = WordNetLemmatizer()

STEMMER = SnowballStemmer("english")

PUNCTUATION  = set('''!()-[]{};:'"\,<>./?@#$%^&*_~''')

def remove_stopwords_and_lemmatize(text, do_stemming=False, lemmatize=False):
    word_tokens = word_tokenize(text.lower())
    filtered_text = [w for w in word_tokens if not w in STOP_WORDS and not w in PUNCTUATION]

    filtered_text_stemmed = filtered_text
    if do_stemming:
        filtered_text_stemmed = [STEMMER.stem(w) for w in filtered_text_stemmed]

    filtered_text_lemmatized = filtered_text_stemmed
    if lemmatize:
        filtered_text_lemmatized = [LEMMATIZER.lemmatize(w) for w in filtered_text_lemmatized]
    return ' '.join(filtered_text_lemmatized)

def get_best_matched_action_using_sent_transformer(allowed_actions, query, model, device="cpu"):
    def encode_batch(texts):
        to_encode = []
        
        for text in texts:
            if text not in EMBEDDING_CACHE:
                to_encode.append(text)
        
        if to_encode:
            new_embeddings = model.encode(to_encode)
            for i, text in enumerate(to_encode):
                EMBEDDING_CACHE[text] = new_embeddings[i]
                
        return np.array([EMBEDDING_CACHE[text] for text in texts])

    if query in allowed_actions:
        return query, [(query, 1.0)]

    query_norm = remove_stopwords_and_lemmatize(text=query,
                                                do_stemming=True,
                                                lemmatize=True)
    query_tokens = set(query_norm.split(" "))
    allowed_actions_filtered = []
    word_sim = []
    for action in allowed_actions:
        action_norm = remove_stopwords_and_lemmatize(text=action,
                                                    do_stemming=True,
                                                    lemmatize=True)
        action_tokens = set(action_norm.split(" "))
        num_common_words = len(list((action_tokens).intersection(query_tokens)))
        word_sim.append(-1 * num_common_words)
        
    indices_actions_sorted_desc_word_sim = np.argsort(word_sim)
    if "cuda" in device:
        max_filtered_actions = 100000
    else:
        max_filtered_actions = 10000

    allowed_actions_filtered = [allowed_actions[ind] for ind in indices_actions_sorted_desc_word_sim[:max_filtered_actions]]

    action_list_embeddings = encode_batch(allowed_actions_filtered)
    
    query = query[:8000]
    if query not in EMBEDDING_CACHE:
        EMBEDDING_CACHE[query] = model.encode([query])[0]
    query_embedding = EMBEDDING_CACHE[query]
    
    sim = cosine_similarity(
        [query_embedding],
        action_list_embeddings
    )
    max_id = np.argmax(sim)

    action_sim_tuples = []
    for i in range(len(allowed_actions_filtered)):
        action_sim_tuples.append((allowed_actions_filtered[i], sim[0][i]))
    action_sim_tuples.sort(key=lambda x: x[1], reverse=True)
    top5_action_sim_tuples = [(x[0], float(x[1])) for x in action_sim_tuples]
    return allowed_actions_filtered[max_id], top5_action_sim_tuples