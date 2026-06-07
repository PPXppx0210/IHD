from utils.utils import DATA_ROOTPATH, load_pickle, save_pickle, reduce_dimension, sample_docs_foreachuser, sample_docs_foreachuser2
from utils.log import logger
import random
import os
import numpy as np
import pandas as pd
from typing import List
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pyLDAvis.sklearn
import bitermplus
import torch
from gsdmm import MovieGroupProcess
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction import text
from nltk.corpus import stopwords
from cuml.cluster import HDBSCAN
from cuml.manifold import UMAP
from gensim.models.coherencemodel import CoherenceModel
from gensim import corpora
from octis.evaluation_metrics.coherence_metrics import Coherence
from octis.evaluation_metrics.diversity_metrics import TopicDiversity
from transformers import AutoTokenizer, AutoModelForMaskedLM

def lda_model(raw_texts, num_topics, visualize):
    cv = CountVectorizer(stop_words="english")
    dtm = cv.fit_transform(raw_texts)
    model = LatentDirichletAllocation(n_components=num_topics, max_iter=50, random_state=2023)
    model.fit(dtm)
    topic_distr = model.transform(dtm)
    if visualize:
        panel = pyLDAvis.sklearn.prepare(model, dtm, cv, mds='tsne') # Create the panel for the visualization
    else:
        panel = None
    return {
        "topic-distr": topic_distr,
        "model": model,
        "cv": cv,
        "dtm": dtm,
        "pyvis-panel": panel,
    }

def btmplus_model(raw_texts, num_topics, visualize):
    cv = CountVectorizer(stop_words="english")
    dtm = cv.fit_transform(raw_texts)
    vocab = np.array(cv.get_feature_names_out()) # get_feature_names_out is only available in ver1.0
    
    # replace words with its vocab word_ids
    docs_vec = bitermplus.get_vectorized_docs(raw_texts, vocab)
    biterms = bitermplus.get_biterms(docs_vec)
    model = bitermplus.BTM(dtm, vocab, T=num_topics)
    model.fit(biterms, iterations=30)

    topic_distr = model.transform(docs_vec)
    logger.info(f"METRICS: perplexity={model.perplexity_}, coherence={model.coherence_}") # model.labels_
    return {
        "topic-distr": topic_distr,
        "model": model,
        "cv": cv,
        "dtm": dtm,
        "pyvis-panel": None,
    }

def gsdmm_model(raw_texts, num_topics, visualize):
    # NOTE: we cant first sample train-corpus, bcz vocabs extracted from train-corpus is not enough for transforming the whole train+test corpus
    cv = CountVectorizer(stop_words="english")
    dtm = cv.fit_transform(raw_texts)
    n_docs, n_terms = dtm.shape
    model = MovieGroupProcess(K=num_topics, alpha=0.01, beta=0.01, n_iters=30)
    docs = [doc.split(' ') for doc in docs] # NOTE: Important!!!, since our sentence is string"Absolutely", if we dont split it, `for word in doc` would produce 'A b s...'
    # NOTE: around 1min/10,0000docs
    labels = model.fit(docs, n_terms)

    # get topic distribution
    # NOTE: we can rewrite mgp.py in gsdmm to save intermediate variables, i.e. m_z, n_z, n_z_w, d_z
    # model = MovieGroupProcess().from_data(K=20, alpha=0.01, beta=0.01, D=n_docs, vocab_size=n_terms, 
    #     cluster_doc_count=m_z, cluster_word_count=n_z, cluster_word_distribution=n_z_w)
    topic_distrs = []
    for doc in raw_texts: # NOTE: around 1min/10,0000docs
        topic_distrs.append(model.score(doc))
    return {
        "topic-distr": topic_distrs,
        "model": model,
        "cv": cv,
        "dtm": dtm,
        "pyvis-panel": None,
    }

def bertopic_model(raw_texts, num_topics, visualize):
    """
    Arguments
        raw_texts:  lists of raw texts
        num_topics: NO USE HERE
    >>> raw_texts = load_pickle("/remote-home/share/dmb_nas/ppx/HeterGAT/tweet-embedding/bertopic/raw_texts_aggby_user_filter_lt2words_processedforbert_subg.pkl")
    >>> raw_texts = [item for sublist in sample_docs for item in sublist]
    """
    train_texts = sample_docs_foreachuser2(raw_texts, sample_frac=0.1)
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = sentence_model.encode(train_texts, show_progress_bar=True)

    # Create instances of GPU-accelerated UMAP and HDBSCAN
    umap_model = UMAP(n_components=5, n_neighbors=15, min_dist=0.0)
    hdbscan_model = HDBSCAN(min_samples=10, gen_min_span_tree=True, prediction_data=True)

    # Pass the above models to be used in BERTopic
    # stop_words = set(text.ENGLISH_STOP_WORDS)
    # stop_words |= set(stopwords.words('english'))
    # stop_words |= set(['people'])
    # vectorizer_model = CountVectorizer(ngram_range=(1, 3), stop_words=list(stop_words))
    vectorizer_model = CountVectorizer(ngram_range=(1, 3), stop_words='english')
    model = BERTopic(language="multilingual", nr_topics="auto", top_n_words=15, min_topic_size=20,
        embedding_model=sentence_model, vectorizer_model=vectorizer_model,
        umap_model=umap_model, hdbscan_model=hdbscan_model
    )

    # NOTE: fit on 1/10 dataset takes ~10min, ~8G(13G on the peak) RAM
    topic_model = model.fit(train_texts, embeddings)
    # topic_model.reduce_topics(raw_texts, nr_topics=50)

    # NOTE: transform on whole dataset takes ~2h, 1/4 takes 30min
    N_FRAC, n_docs = 4, len(raw_texts)
    topic_distr_all, topic_label_all, topic_prob_all = [], [], []
    for frac in range(N_FRAC):
        test_texts = raw_texts[int(n_docs/N_FRAC*frac):int(n_docs/N_FRAC*(frac+1))]
        topic_distr, _ = topic_model.approximate_distribution(test_texts, min_similarity=1e-5)
        topic_distr = topic_distr.astype(np.float16) # save space and s/l time
        labels, probs = topic_model.transform(test_texts)
        topic_distr_all.extend(topic_distr); topic_label_all.extend(labels); topic_prob_all.extend(probs)
    
    # NOTE: update topics and reduce outliers in class '-1'
    # new_topics = topic_model.reduce_outliers(raw_texts, topics, probabilities=topic_distr, strategy="probabilities")
    # topic_model.update_topics(raw_texts, new_topics)

    # NOTE: Visualize All Topics in One Distance-Map
    # topic_model.visualize_topics()
    # NOTE: Visualize All Document-Embeddings aggby Clusters
    # topic_model.visualize_documents(sample_docs, embeddings=embeddings, hide_document_hover=False)

    # NOTE: Get Top-N Words per Topic
    # topic_model.topic_representations_
    # NOTE: Get Topic-Word Matrix
    # topic_model.c_tf_idf

    return {
        "topic-distr": topic_distr_all,
        "model": topic_model,
        "cv": None,
        "dtm": None,
        "pyvis-panel": None,
    }

def merge_similar_topics(topic_model:BERTopic, docs:List, merge_thr:float=0.1) -> BERTopic:
    # reduce topics
    topic_model.reduce_topics(docs, nr_topics="auto")

    # merge topics
    hierarchical_topics = topic_model.hierarchical_topics(docs)
    merge_ids = [row['Topics'] for _, row in hierarchical_topics.iterrows() if row['Distance']<merge_thr]
    topic_model.merge_topics(docs, merge_ids)
    # Visualize Hierarchical Topics
    # topic_model.get_topic_tree(hierarchical_topics)
    return topic_model

def update_topic_reprs(topic_model:BERTopic, docs:List, ) -> BERTopic:
    import emoji
    demojize_texts = ['_'+demojize_text[1:-1]+'_' for demojize_text in emoji.get_emoji_unicode_dict(lang='en').keys()]

    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    # from nltk.corpus import stopwords
    # stop_words_list = stopwords.words('english') # 鑾峰彇鑻辨枃鍋滅敤璇嶅垪琛?
    stop_words = set(ENGLISH_STOP_WORDS) | set(demojize_texts)
    logger.info(f"stop_words={len(stop_words)}, ENGLISH_STOP_WORDS={len(ENGLISH_STOP_WORDS)}, demojize_texts={len(demojize_texts)}")
    vectorizer_model = CountVectorizer(ngram_range=(1,3), stop_words=list(stop_words))

    topic_model.update_topics(docs=docs, vectorizer_model=vectorizer_model)
    return topic_model

def calculate_coherence_score(docs, topics, topic_model:BERTopic):
    """
    How to get Args, i.e. docs, topics and topic_model? Use Bertopic as example...
    >>> topic_model = model.fit(docs, embeddings)
    >>> topic_distr, _ = topic_model.approximate_distribution(docs, min_similarity=1e-5)
    >>> topics, probs = topic_model.transform(docs)
    """

    # Preprocess Documents
    documents = pd.DataFrame({"Document": docs,
        "ID": range(len(docs)),
        "Topic": topics})
    documents_per_topic = documents.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})
    cleaned_docs = topic_model._preprocess_text(documents_per_topic.Document.values)

    # Extract vectorizer and tokenizer from BERTopic
    vectorizer = topic_model.vectorizer_model
    tokenizer = vectorizer.build_analyzer() # can be used for n-gram words tokenizing

    # Extract features for Topic Coherence evaluation
    # all_words = vectorizer.get_feature_names_out()
    all_words = vectorizer.get_feature_names()
    tokens = [tokenizer(doc) for doc in cleaned_docs]
    dictionary = corpora.Dictionary(tokens)
    corpus = [dictionary.doc2bow(token) for token in tokens]

    # Create topic words
    topic_words = [topic_model.get_topic(topic) for topic in range(len(topic_model.topic_labels_))]
    topic_words = [[w[0] for w in words if w[0] in all_words] for words in topic_words if words]

    coherence_model = CoherenceModel(topics=topic_words,
        texts=tokens,
        corpus=corpus,
        dictionary=dictionary,
        coherence='c_v')
    # coherence_model_npmi = CoherenceModel(topics=topic_words,
    #     texts=tokens,
    #     corpus=corpus,
    #     dictionary=dictionary,
    #     coherence='c_npmi')
    # coherence_model_npmi.get_coherence()
    npmi = Coherence(texts=tokens, topk=topic_model.top_n_words, measure='c_npmi')
    topic_diversity = TopicDiversity(topk=topic_model.top_n_words)
    results = {
        "gensim.coherence": coherence_model.get_coherence(),
        "octis.npmi": npmi.score({"topics":topic_words}),
        "octis.topic_diversity": topic_diversity.score({"topics":topic_words}),
    }
    return results

def get_tweet_feat_for_tweet_nodes(model="lda", num_topics=20, visualize=False):
    """
    valid model names are ['lda','btm','gsdmm','bertopic']
    """
    suffix = f"model{model}" # used for naming saved-results, i.e. topic-distr_{suffix}.pkl
    if model == 'lda' or model == 'btm':
        suffix += f"_numtopic{num_topics}"

    sent_emb_dir = os.path.join(DATA_ROOTPATH, "HeterGAT/lda-model/")
    raw_texts = load_pickle(os.path.join(sent_emb_dir, "processedtexts-per-text/ptexts_per_user_all.p"))
    if model == "lda":
        ret = lda_model(raw_texts=raw_texts, num_topics=num_topics, visualize=visualize)
    elif model == "btm":
        # ret = btm_model(raw_texts=raw_texts, num_topics=num_topics, visualize=visualize)
        ret = btmplus_model(raw_texts=raw_texts, num_topics=num_topics, visualize=visualize)
    elif model == 'gsdmm':
        ret = gsdmm_model(raw_texts=raw_texts, num_topics=num_topics, visualize=visualize)
    elif model == 'bertopic':
        # NOTE: there are two ways for creating topic distributions, see details in reference https://github.com/MaartenGr/BERTopic/issues/1026
        # we choose to use func approximate_distribution(), since it is fasttttt!
        # Other options for accelaerating can be referenced in https://maartengr.github.io/BERTopic/getting_started/tips_and_tricks/tips_and_tricks.html#gpu-acceleration
        ret = bertopic_model(raw_texts=raw_texts, num_topics=num_topics, visualize=visualize)
    
    # Save Final Results
    for key,val in ret.items():
        if val is None:
            continue
        save_pickle(val, f"{key}_{suffix}.pkl")

def agg_tagemb_by_user(n_user:int, cascades:dict, pretrained_model_name:str='xlm-roberta-base')->np.ndarray:
    # Prepare Pretrained-LLM
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    model = AutoModelForMaskedLM.from_pretrained(pretrained_model_name)
    if torch.cuda.is_available():
        model = model.to('cuda')

    # Per Hashtag
    tag_embs = []
    # tag2emb = {}
    user2tagids = [[] for _ in range(n_user)]
    for tag_id, tag in enumerate(sorted(cascades.keys())):
        # Calculate Emb
        with torch.no_grad():
            tokens = tokenizer(tag, return_tensors='pt', padding=True)
            if torch.cuda.is_available():
                tokens = {
                    'input_ids': tokens['input_ids'].to('cuda'),
                    # 'token_type_ids': tokens['token_type_ids'].to('cuda'),
                    'attention_mask': tokens['attention_mask'].to('cuda'),
                }
            output = model(**tokens)
            embeds = output.logits.detach().cpu().numpy()
            embeds = np.mean(embeds[0], axis=0, dtype=np.float16)
        tag_embs.append(embeds)
        # tag2emb[tag] = embeds
        del tokens, output

        # Accumulate User Appearance in Multiple Tags
        cascade = cascades[tag]
        for user_id, _ in cascade:
            user2tagids[user_id].append(tag_id)
    tag_embs = np.array(tag_embs)
    save_pickle(tag_embs, os.path.join(DATA_ROOTPATH, "Weibo-Aminer/llm/tag_embs_model_xlm-roberta-base.pkl"))
    
    # Aggregate Tag Emb foreach User
    user2emb = []
    for tag_ids in user2tagids:
        if len(tag_ids) != 0:
            emb = np.mean(tag_embs[tag_ids], axis=0)
        else:
            emb = np.zeros(shape=tag_embs.shape[1])
        user2emb.append(emb)
    user2emb = np.array(user2emb)
    save_pickle(user2emb, os.path.join(DATA_ROOTPATH, "Weibo-Aminer/llm/tag_embs_aggbyuser_model_xlm-roberta-base.pkl"))

    # TODO: Reduce LLM Dimension
    # NOTE: 1. Use Traditional dimension reduction methods, i.e. PCA; 2. Add FC-Layer in Total Model
    if pretrained_model_name == 'xlm-roberta-base':
        user2emb = reduce_dimension(user2emb, reduce_dim=128)
    save_pickle(user2emb, os.path.join(DATA_ROOTPATH, "Weibo-Aminer/llm/tag_embs_aggbyuser_model_xlm-roberta-base_pca_dim128.pkl"))
    
    return user2emb
