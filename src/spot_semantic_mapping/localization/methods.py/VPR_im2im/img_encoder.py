import numpy as np
from utils.jax_helper import *
from skimage.color import rgb2gray
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pickle as pkl
import os
import faiss
import tqdm


class ImageEncoder:
    def __init__(self, vi_transformer):
        """
        Image encoder for use of embedding and retrieval of images in the VPR_im2im framework.
        """
        self.vi_transformer = vi_transformer
        
        # Standard VLAD attributes
        self.vlad_centers = None
        
        # Domain-aware VLAD attributes
        self.domain_pca = None
        self.domain_classifier = None
        self.domain_vlad_centers = None 
            
    def embed(self, images, patches=True, agg_method="vlad",
            num_clusters=16, num_domains=6, seed=42, eps=1e-12, load=True, save=True, grayscale=False):
        """
        - agg_method: Options: "vlad", "domain_vlad", "gmp", "gap", "gem". 
        - num_domains: Number of distinct environments to cluster for "domain_vlad" (default 6 based on paper).
        """
        
        # --- CACHE CHECKING ---
        cache_path = f"VPR_im2im/cache/{agg_method}_embeddings"
        if agg_method == "vlad" and os.path.exists(cache_path + ".pkl") and load:
            print(f"Loading cached VLAD data from {cache_path}.pkl...")
            with open(cache_path + ".pkl", "rb") as f:
                vlad_data = pkl.load(f)
            self.vlad_centers = vlad_data["centers"]
            return vlad_data["embeddings"]
            
        elif agg_method == "domain_vlad" and os.path.exists(cache_path + ".pkl") and load:
            print(f"Loading cached domain VLAD data from {cache_path}.pkl...")
            with open(cache_path + ".pkl", "rb") as f:
                domain_data = pkl.load(f)
            self.domain_pca = domain_data["pca"]
            self.domain_classifier = domain_data["classifier"]
            self.domain_vlad_centers = domain_data["domain_centers"]
            return domain_data["embeddings"]
            
        elif agg_method in ["gmp", "gap", "gem"] and os.path.exists(cache_path + ".npy") and load:
            return np.load(cache_path + ".npy")
        
        # --- 1) EXTRACT FEATURES ---
        
        if grayscale:
            images = rgb2gray(images)
        if patches:
            im_features = batch_apply(self.vi_transformer.embed_images_by_patch, images)
        else:
            im_features = batch_apply(self.vi_transformer.embed_images, images)
            return l2norm_jax(np.asarray(im_features, dtype=np.float32), axis=1, eps=eps)

        im_features = np.asarray(im_features, dtype=np.float32)
        # --- 2) AGGREGATE ---
        
        if agg_method == "domain_vlad":
            # 2a. Calculate GeM representation strictly for routing
            p = 3.0
            gem_feat = (np.mean(np.maximum(im_features, 0.0) ** p, axis=(1, 2)) + eps) ** (1.0 / p)
            gem_feat = l2norm_jax(gem_feat, axis=1, eps=eps)

            # 2b. Domain Routing (Fit or Predict)
            if getattr(self, "domain_pca", None) is None:
                # Fit the unsupervised PCA & KMeans routing
                self.domain_pca = PCA(n_components=min(num_clusters, gem_feat.shape[1]), random_state=seed)
                gem_pca = self.domain_pca.fit_transform(gem_feat)
                
                self.domain_classifier = KMeans(n_clusters=num_domains, random_state=seed, n_init="auto")
                domains = self.domain_classifier.fit_predict(gem_pca)
                
                # Fit domain-specific vocabularies
                self.domain_vlad_centers = {}
                print("Fitting domain-specific VLAD centers with KMeans...")
                for d in range(num_domains):
                    mask = (domains == d)
                    if not np.any(mask): continue
                    
                    domain_patches = im_features[mask]
                    flat = domain_patches.reshape(-1, domain_patches.shape[-1])
                    km = KMeans(n_clusters=num_clusters, random_state=seed, n_init="auto")
                    km.fit(flat)
                    self.domain_vlad_centers[d] = km.cluster_centers_.astype(np.float32)
            else:
                # Predict domains using established latent space
                gem_pca = self.domain_pca.transform(gem_feat)
                domains = self.domain_classifier.predict(gem_pca)

            # 2c. Aggregate using matched vocabularies
            D_out = num_clusters * im_features.shape[-1]
            out = np.zeros((im_features.shape[0], D_out), dtype=np.float32)
            
            for d in self.domain_vlad_centers.keys():
                mask = (domains == d)
                if np.any(mask):
                    domain_patches = im_features[mask]
                    vlad_apply = lambda x: vlad_aggregate(x, self.domain_vlad_centers[d], eps=eps)
                    domain_vlad = self._batch_apply(vlad_apply, domain_patches)
                    out[mask] = domain_vlad

            if save:
                domain_data = {
                    "pca": self.domain_pca,
                    "classifier": self.domain_classifier,
                    "domain_centers": self.domain_vlad_centers,
                    "embeddings": out
                }
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path + ".pkl", "wb") as f:
                    pkl.dump(domain_data, f)
                    
        elif agg_method == "vlad":
            if getattr(self, "vlad_centers", None) is None:
                flat = im_features.reshape(-1, im_features.shape[-1])
                print("Fitting VLAD centers with KMeans...")
                km = KMeans(n_clusters=num_clusters, random_state=seed, n_init="auto")
                km.fit(flat)
                self.vlad_centers = km.cluster_centers_.astype(np.float32)
            
            vlad_apply = lambda x: vlad_aggregate(x, self.vlad_centers, eps=eps)
            out = batch_apply(vlad_apply, im_features)
            
            if save:
                vlad_embeddings = {"centers": self.vlad_centers, "embeddings": out}
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path + ".pkl", "wb") as f:
                    pkl.dump(vlad_embeddings, f)
                
        elif agg_method == "gmp":
            out = np.max(im_features, axis=(1, 2))
            if save: np.save(cache_path + ".npy", out)

        elif agg_method == "gap":
            out = np.mean(im_features, axis=(1, 2))
            if save: np.save(cache_path + ".npy", out)

        elif agg_method == "gem":
            p = 3.0
            out = (np.mean(np.maximum(im_features, 0.0) ** p, axis=(1, 2)) + eps) ** (1.0 / p)
            if save: np.save(cache_path + ".npy", out)
            
        else:
            raise ValueError(f"Unknown agg_method={agg_method}")

        return l2norm_jax(out.astype(np.float32), axis=1, eps=eps)