from models.models import DinoModel, SiglipModel
from configs.loader import cfg
from img_encoder import ImageEncoder
from utils.logger import *
import numpy as np
from utils.metrics import compute_metrics_from_scores
from make_dataset import load_data, load_spot_data
from utils.jax_helper import cdist, cosine_similarity_jax
from tqdm import tqdm
from collections import deque
import pandas as pd

# POT (Python Optimal Transport)
try:
    import ot
except ImportError:
    ot = None

BAD_CHARS = {'$', '#', '@', '!', '%', '^', '&', '*', '(', ')', '-', '+', '=', 
             '{', '}', '[', ']', '|', '\\', ':', ';', '"', "'", '<', '>', ',', '.', '?', '/'}

# ------------------------------
# Database Preparation (Unchanged)
# ------------------------------
def prepare_embeddings(vision_transformer, images, query_frames, methods, ts, grayscale=False, cropping=False):
    res = {k: {} for k in methods}

    # turn query frame into grayscale
    if not isinstance(query_frames, np.ndarray):
        query_frames = np.array(query_frames)
    
    # crop query images simulate narrow fov
    if cropping:
        h, w = query_frames.shape[1:3]
        crop_h, crop_w = int(h * 0.75), int(w * 0.75)
        start_h = (h - crop_h) // 2
        start_w = (w - crop_w) // 2
        query_frames = query_frames[:, start_h:start_h+crop_h, start_w:start_w+crop_w, :]
    
    # turn query images into grayscale
    if grayscale:
        query_frames = np.stack([query_frames.mean(-1)] * 3, axis=-1)

    for method, cfg in methods.items():
        encoder = ImageEncoder(vision_transformer)
        embs = encoder.embed(images, cfg["patches"], cfg["agg_method"], cfg["num_clusters"], grayscale=cfg["grayscale"], load=False)
        
        print(f"Embedding queries for method: {method}")
        X = encoder.embed(query_frames, patches=cfg["patches"], agg_method=cfg["agg_method"], num_clusters=cfg['num_clusters'], grayscale=cfg['grayscale'], save=False, load=False)
        
        res[method] = {
            "embeddings": embs,
            "encoder": encoder,
            "X": X,
            "ts": ts
        } 
        
    return res

# ------------------------------
# Main Evaluation Loop
# ------------------------------
def evaluate_localization(
    dataset, dino_encoder, window_size=1
):
    """
    ground_truth: The Adjacency List array we generated earlier.
                  Format: array([[0, list([0, 1, 2])], [1, list([1, 2, 3])], ...])
    """
    methods = {
        # "domain_patch_vlad": {"patches": True, "agg_method": "domain_vlad", "num_clusters":32, "grayscale": False},

        "patch_vlad": {"patches": True, "agg_method": "vlad", "num_clusters":32, "grayscale": False},

        "patch_gem": {"patches": True, "agg_method": "gem", "num_clusters":32, "grayscale": False},

        "patch_gmp": {"patches": True, "agg_method": "gmp", "num_clusters":32, "grayscale": False},

        "patch_gap": {"patches": True, "agg_method": "gap", "num_clusters":32, "grayscale": False}
    }


    
    print("Finished preparing embeddings for all methods. Starting evaluation...\n")
    
    results = {m: {"Hit@1": 0.0, "Hit@3": 0.0, "Hit@5": 0.0, "MRR": 0.0, "N": 0} for m in methods}

    # all_frames = dataset['query_images']
    ts = np.array(dataset['ts'])
    
    encoded_data = prepare_embeddings(dino_encoder, 
                                  images=dataset["db_images"],
                                  query_frames=dataset["query_images"],
                                  methods=methods, 
                                  ts=ts,
                                  cropping=False,
                                  grayscale=False)

                
    ground_truth = dataset["ground_truth"]
    
    for method, _ in methods.items():
        N_db = encoded_data[method]["embeddings"].shape[0]

        # combined_X = deque(maxlen=window_size) 
        
        for t in tqdm(range(ts[-1])):
            image_scores, correct_db_indices, _ = localize(encoded_data, ground_truth, method, t, n_views=10)
            
            # # ---------------------------
            
            hot_coded_correct = np.zeros(N_db)
            if len(correct_db_indices) > 0:
                hot_coded_correct[correct_db_indices] = 1.0
                
            metrics, _, _ = compute_metrics_from_scores(image_scores, hot_coded_correct, ks=(1, 3, 5))

            results[method]["Hit@1"] += metrics.get("Hit@1", metrics.get("Acc", 0.0))
            results[method]["Hit@3"] += metrics["Hit@3"]
            results[method]["Hit@5"] += metrics["Hit@5"]
            results[method]["MRR"] += metrics["MRR"]
            results[method]["N"] += 1

        print(f"finished evaluating method: {method}")
    
    print("\n" + "=" * 80)
    print(f"{'Method':<25} | {'N':>6} | {'Hit@1':>7} | {'Hit@3':>7} | {'Hit@5':>7} | {'MRR':>7}")
    print("-" * 80)
    for m, r in results.items():
        N = max(1, r["N"])
        print(f"{m:<25} | {r['N']:6d} | {100*(r['Hit@1']/N):6.2f}% | {100*(r['Hit@3']/N):6.2f}% | {100*(r['Hit@5']/N):6.2f}% | {r['MRR']/N:6.3f}")
    print("=" * 80 + "\n")
    return results


def localize(encoded_data, ground_truth, method, t, n_views=1):
    correct_db_indices = ground_truth[t][1]
    
    X = encoded_data[method]["X"][encoded_data[method]["ts"] == t][:n_views]  # (C, D)
    cosine_similarity_scores = np.array(cosine_similarity_jax(X, encoded_data[method]["embeddings"]))
    image_scores = np.sum(cosine_similarity_scores, axis=0) 

    predictions = np.argsort(image_scores)[0]

    return image_scores, correct_db_indices, predictions


def retrieve_subgraphs(dataset, ordered_indices, g, top_k=5, window=5):
    """
    This function retrieve the subgraph based on image retreiveal result
    input:
    - dataset: dict containing db images and db_trajectories
    - ordered_indices: ranking of images based on matching score
    - g (graph): {
        nodes: [{oid, class_name, text_ft, clip_ft, room, position}]
        edges: [{src_id, dst_id, src_name, dst_name, relation}]
    }
    output:
    - corresponding subgraph
    """
    # Create DataFrames
    nodes = pd.DataFrame(g['nodes'])
    edges = pd.DataFrame(g['edges'])
    n_pos = np.array(list(nodes['position']))
    # retrieved_images = dataset['db_images'][ordered_indices][:top_k] # (k, w, h, c) - unused in this block
    retrieved_images_pos = np.array(dataset['db_traj'][ordered_indices][:top_k]) # (k, 3)
    
    # Calculate distance matrix (shape: top_k x num_nodes)
    dist_matrix = cdist(retrieved_images_pos, n_pos, metric='euclidean')
    
    # Mask: True if a node is within 'window' distance to ANY of the top_k retrieved images
    node_mask = np.any(dist_matrix <= window, axis=0)
    
    # 1. Select the node chosen by the mask
    sub_nodes = nodes[node_mask]
    
    # Early exit if no nodes fall within the spatial window
    if sub_nodes.empty:
        print("State: No objects found within the specified window.")
        return {'nodes': [], 'edges': []}

    # 2. Select the edges connected to the chosen nodes (induced subgraph)
    # Both source and destination nodes must be in our masked subset
    valid_oids = set(sub_nodes['oid'])
    sub_edges = edges[edges['src_id'].isin(valid_oids) & edges['dst_id'].isin(valid_oids)]
    
    # 3. Poll the room
    room_counts = sub_nodes['room'].value_counts()
    likely_room = room_counts.idxmax() # The room with the most objects in the window
    
    # 4. Give an informative state print out
    print("\n" + "="*40)
    print("📍 SUBGRAPH STATE SUMMARY")
    print("="*40)
    print(f"Likely Current Room: **{likely_room}**\n")
    
    print("📊 Objects per room (within window):")
    for room, count in room_counts.items():
        print(f"  - {room}: {count} objects")
    
    print(f"\n🛋️  Objects detected in {likely_room}:")
    room_objects = sub_nodes[sub_nodes['room'] == likely_room]['class_name'].tolist()
    print(f"  {', '.join(room_objects)}")
    
    print("\n🔗 Object Relations (Induced Subgraph):")
    if not sub_edges.empty:
        for _, row in sub_edges.iterrows():
            print(f"  - [{row['src_name']}] {row['relation']} [{row['dst_name']}]")
    else:
        print("  - No local relations found between these objects.")
    print("="*40 + "\n")
    
    # Return the corresponding subgraph as dictionaries
    return {
        'nodes': sub_nodes.to_dict(orient='records'),
        'edges': sub_edges.to_dict(orient='records')
    }
    
# ------------------------------
# Main (Setup)
# ------------------------------
if __name__ == "__main__":
    logger = build_logger()
    
    dino = DinoModel(cfg)

    # with open("spot_dataset_w_gt_5m.pkl", 'rb') as file: 
    #     dataset = pkl.load(file)

    dataset = load_data("data/AnyLoc2023-Public-Data/Public/Datasets-All/17places")

    # We no longer pass map_data or ground_truth_labels since we aren't doing room-classification
    evaluate_localization(
        dataset=dataset,
        dino_encoder=dino
    )