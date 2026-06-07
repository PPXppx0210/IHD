from utils.utils import DATA_ROOTPATH, load_pickle, save_pickle
from utils.log import logger
from utils.graph import reindex_graph
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import random
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans, AgglomerativeClustering, MiniBatchKMeans
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from sklearn.decomposition import PCA
from matplotlib import colors
from typing import Tuple

def get_centroids(X, y):
    nb_kind = len(np.unique(y))
    centers_ = []
    for idx in range(nb_kind):
        center = np.median(X[y==idx],axis=0)
        centers_.append([elem for elem in center])
    return np.array(centers_)

def get_centroids2(X, y):
    clf = NearestCentroid()
    clf.fit(X, y)
    centroids_ = clf.centroids_
    return np.array([[elem for elem in center] for center in centroids_])

def apply_clustering_algo(tweet_features, model:str='agg', desc='', thr=None, todraw=False):
    """
    NOTE: desc的一般形式为f"g{g.vcount()}", 而当model='agg'时形式为f"g{g.vcount()}_thr{thr}"
    """
    full_desc = f"{desc}"
    if model == 'agg':
        if thr is None:
            thr = 0.5 # default
        full_desc += f"_thr{thr}"
    centroids_filepath = os.path.join(DATA_ROOTPATH, f"HeterGAT/clustering/{full_desc}_{model}_centroids_.pkl")
    labels_filepath = os.path.join(DATA_ROOTPATH, f"HeterGAT/clustering/{full_desc}_{model}_labels_.pkl")
    logger.info(f"centroids={centroids_filepath}, labels={labels_filepath}")
    if os.path.exists(centroids_filepath) and os.path.exists(labels_filepath):
        return load_pickle(centroids_filepath), load_pickle(labels_filepath)

    if not isinstance(tweet_features, np.ndarray):
        tweet_features = np.array(tweet_features)
    
    if todraw:
        # Reduce Dimensions for plotting
        pca_ = PCA(n_components=3)
        tweet_features = pca_.fit_transform(tweet_features)
    
    # 1. sample a fraction of tweets for clustering and training cluster-classifier
    frac = 1e-3
    train_tweets = random.choices(tweet_features, k=int(len(tweet_features)*frac))
    train_tweets_df = pd.DataFrame(train_tweets)
    logger.info("--Finish Sampling Tweets--")

    # 3. perform clustering
    if model == 'agg':
        logger.info(f"--Perform Hierarchical Clustering Method (thr={thr})--")
        model_ = AgglomerativeClustering(n_clusters=None, distance_threshold=thr) # thr is set to <default> 
        labels_ = model_.fit_predict(train_tweets_df).astype("int")
        # 3.2 train classifier and give labels to other test data
        knnmodel_ = KNeighborsClassifier(n_neighbors=1)
        knnmodel_.fit(train_tweets_df, labels_)
        whole_labels_ = knnmodel_.predict(tweet_features)
        logger.info(f"--Finish Training KNNModel and Calculating {model.capitalize()} Labels--")
        # 3.1 get centroids
        centroids = get_centroids2(X=tweet_features, y=whole_labels_)
        logger.info(f"--Finish Calculating {model.capitalize()} Centroids (Num={model_.n_clusters_}, Shape={centroids.shape})--")
    elif model == 'mbk':
        # 2. determine the K
        elbow_ = KElbowVisualizer(MiniBatchKMeans(), k=(4,30))
        elbow_.fit(train_tweets_df)
        n_clusters = elbow_.elbow_value_
        logger.info(f"--Finish Determining K={n_clusters}--")

        model_ = MiniBatchKMeans(n_clusters=n_clusters, random_state=2023)
        # 3.2 train classifier and give labels to all data
        whole_labels_ = model_.fit_predict(tweet_features).astype("int")
        # 3.1 get centroids
        centroids = model_.cluster_centers_
        logger.info(f"--Finish Calculating {model.capitalize()} Labels and Centroids--")
    save_pickle(centroids, centroids_filepath)
    save_pickle(whole_labels_, labels_filepath)

    # 4. plot
    if todraw:
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111, projection="3d", label="bla", computed_zorder=False)
        cmap = colors.ListedColormap(["#682F2F", "#9E726F", "#D6B2B1", "#B9C0C9", "#9F8A78", "#F3AB60"])
        ax.scatter(train_tweets_df[0], train_tweets_df[1], train_tweets_df[2], c=labels_, marker=".", cmap=cmap, zorder=0)
        ax.scatter(centroids[:,0],centroids[:,1],centroids[:,2], c='red', marker="o", s=100)
        plt.show()
    
    return centroids, whole_labels_

def get_tweet_feat_for_user_nodes(model='lda', num_topics=25) -> np.ndarray:
    if model == 'lda':
        twft_filepath = os.path.join(DATA_ROOTPATH, f"HeterGAT/lda-model/twft_per_user_k{num_topics}.pkl")
    elif model == 'bertopic':
        twft_filepath = os.path.join(DATA_ROOTPATH, f"HeterGAT/tweet-embedding/bertopic/twft_per_user.pkl")
    if os.path.exists(twft_filepath):
        logger.info(f"Trying to Get Twft in path:{twft_filepath}...Success")
        return load_pickle(twft_filepath)
    
    twft = []
    if model == 'lda':
        prefix, suffix = os.path.join(DATA_ROOTPATH, "HeterGAT/lda-model"), f"_k{num_topics}_maxiter{50}"
        logger.info(f"Calculating Twft Using CountVectorizer/LDAModel/ProcessedUserTexts in path:{prefix}/.../{suffix}")
        for part in range(1,11):
            cv_ = load_pickle(f"{prefix}/cv/cv_0{part}{suffix}.p")
            lda_model_ = load_pickle(f"{prefix}/model/model_0{part}{suffix}.p")
            user_texts_l = load_pickle(f"{prefix}/processedtexts-per-user/ProcessedTexts_{part}.p")
            twft.extend(lda_model_.transform(cv_.transform(user_texts_l)))
    elif model == 'bertopic':
        prefix = os.path.join(DATA_ROOTPATH, "HeterGAT/lda-model")
        model = load_pickle(os.path.join(DATA_ROOTPATH, "HeterGAT/tweet-embedding/bertopic/topic_approx_distribution_reduce_auto_merge_lt01_subg.pkl"))
        for part in range(1,11):
            user_texts_l = load_pickle(f"{prefix}/processedtexts-per-user/ProcessedTexts_{part}.p")
            topic_distr, _ = model.approximate_distribution(user_texts_l).astype()
            topic_distr = topic_distr.astype(np.float16) # save space and s/l time
            twft.extend(topic_distr)
    twft = np.array(twft)
    save_pickle(twft, twft_filepath)
    return twft

def tweet_centralized_process(homo_g, user_tweet_mp, tweet_features, embedding_method, clustering_algo, distance_threshold)->Tuple[dict,list,np.ndarray]:
    # 1. prepare tweet feature
    lda_k = tweet_features.shape[1]

    # 2. apply clustering algo
    centroids_, tw2centroids_ = apply_clustering_algo(tweet_features, model=clustering_algo, desc=f"g{homo_g.vcount()}_emb{embedding_method}", thr=distance_threshold)

    # replace user_tweet_edges with user_tweet_centroid_edges
    user_nodes = homo_g.vs["label"]
    user_tweet_centroid_edges = []
    for user in user_nodes:
        for tweet in user_tweet_mp[user]:
            user_tweet_centroid_edges.append((user, tw2centroids_[tweet]))

    # 3. build new homo graph
    tweet_nodes = [elem for elem in range(len(centroids_))]
    uu_edges = homo_g.get_edgelist()
    logger.info(f"Graph Info: nb-users={len(user_nodes)}, nb-tweets={len(tweet_nodes)}, nb-uu-edges={len(uu_edges)}, nb-ut-edges={len(user_tweet_centroid_edges)}")
    nodes, edges = reindex_graph([user_nodes, tweet_nodes], [uu_edges, user_tweet_centroid_edges])

    # 4. get node features for both users and tweets
    twft_for_users  = get_tweet_feat_for_user_nodes(model=embedding_method, num_topics=lda_k)
    if not isinstance(twft_for_users, np.ndarray):
        twft_for_users = np.array(twft_for_users)
    feats = np.concatenate((twft_for_users[user_nodes], centroids_))
    logger.info(f"New Homo Graph Info: nb-nodes={len(nodes)}, nb-edges={len(edges[0])}:{len(edges[1])}, nb-tweets={feats.shape}")

    return nodes, edges, feats
