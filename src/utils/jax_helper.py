import numpy as np
import jax
import tqdm
import jax.numpy as jnp

def l2norm_jax(x, axis=-1, eps=1e-12):
    return x / (jnp.linalg.norm(x, axis=axis, keepdims=True) + eps)


def batch_apply(embed_func, data, batch_size=64, eps=1e-8):
    """
    Apply a function to a  in batches to manage memory usage.
    """ 
    batch_embs = []
    for i in tqdm.tqdm(range(0, len(data), batch_size)):
        end = np.min([i + batch_size, len(data)])
        batch_imgs = data[i:end]
        batch_embs.append(embed_func(batch_imgs))

    # Concatenate all batches
    batch_embs = np.concatenate(batch_embs, axis=0)
    
    # Normalize all embeddings along the feature dimension
    norms = np.linalg.norm(batch_embs, axis=-1, keepdims=True) + eps
    batch_embs = batch_embs / norms
    return batch_embs        


@jax.jit
def cdist(x, y):
  return jnp.sqrt(jnp.sum((x[:, None] - y[None, :]) ** 2, -1))


@jax.jit
def vlad_aggregate(im_features, centers, eps=1e-12):
    """
    im_features: (N,H,W,D) float32
    centers:     (K,D) float32
    returns:     (N,K*D) float32 (VLAD)
    """
    N, H, W, D = im_features.shape
    P = H * W
    
    # Cast and reshape
    X = im_features.reshape((N, P, D)).astype(jnp.float32)  # (N,P,D)
    C = centers.astype(jnp.float32)                         # (K,D)
    K = C.shape[0]

    # Assignments via nearest center (cosine or L2)
    Xn = l2norm_jax(X, axis=2, eps=eps)
    Cn = l2norm_jax(C, axis=1, eps=eps)
    sim = Xn @ Cn.T                     # (N,P,K)
    A = jnp.argmax(sim, axis=2)         # (N,P) int

    # Residual calculation
    R = X - C[A]                        # (N,P,D)
    
    # JAX equivalent of np.add.at
    vlad = jnp.zeros((N, K, D), dtype=jnp.float32)
    batch_idx = jnp.arange(N)[:, None]  # (N, 1) to broadcast with (N, P)
    vlad = vlad.at[batch_idx, A].add(R)

    # Intra-normalization per cluster
    vlad = l2norm_jax(vlad, axis=2, eps=eps)

    # Flatten + global L2
    vlad = vlad.reshape((N, K * D))
    vlad = l2norm_jax(vlad, axis=1, eps=eps)
    
    return vlad


@jax.jit
def cosine_similarity_jax(X, Y, eps=1e-12):
    """
    Computes the cosine similarity matrix between two sets of vectors.
    X: (N, D) array of query embeddings
    Y: (M, D) array of database embeddings
    Returns: (N, M) matrix of cosine similarities
    """
    X_norm = X / (jnp.linalg.norm(X, axis=-1, keepdims=True) + eps)
    Y_norm = Y / (jnp.linalg.norm(Y, axis=-1, keepdims=True) + eps)
    
    return X_norm @ Y_norm.T
