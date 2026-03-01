# ============================================================
# eval_localization.py
# ============================================================
import jax
import jax.numpy as jnp
from spot.data_loading import SpotDataset
from build_sg_db import *
from Model.models import *
from utils.data_construction.plot_graph import *
from configs.loader import cfg
import json
from VPR_im2im.utils.img_encoder import ImageEncoder
import pickle as pkl
from utils.logger import *
from generate_occupancy_field import *
import numpy as np
from VPR_im2im.utils.metrics import compute_metrics_from_scores
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

# POT (Python Optimal Transport)
try:
    import ot
except ImportError:
    ot = None

BAD_CHARS = {'$', '#', '@', '!', '%', '^', '&', '*', '(', ')', '-', '+', '=', 
             '{', '}', '[', ']', '|', '\\', ':', ';', '"', "'", '<', '>', ',', '.', '?', '/'}

# ------------------------------
# Scorers
# ------------------------------
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

def get_wasserstein_costs(src_embeddings, room_node_embeddings, room_node_labels, unique_room_labels):
    """
    Computes the Wasserstein distance (OT cost) from a set of source embeddings to each room's node embeddings.
     - src_embeddings: (N, D) array of source node embeddings.
     - room_node_embeddings: (M, D) array of all room node embeddings.
     - room_node_labels: (M,) array of room labels corresponding to each node embedding.
     - unique_room_labels: List or array of unique room labels to compute costs for.
    """
    
    if ot is None:
        raise ImportError("Please install 'pot' library to use OT: pip install pot")

    R = len(unique_room_labels)
    costs = np.zeros((R,), dtype=np.float32)
    n_src = src_embeddings.shape[0]
    p = np.ones((n_src,), dtype=np.float32) / max(n_src, 1)

    for i in range(R):
        dst_embeddings = room_node_embeddings[room_node_labels == i, :]
        if dst_embeddings.shape[0] == 0:
            costs[i] = np.inf
            continue

        n_dst = dst_embeddings.shape[0]
        q = np.ones((n_dst,), dtype=np.float32) / max(n_dst, 1)

        sim_matrix = cosine_similarity(src_embeddings, dst_embeddings)
        M = 1.0 - sim_matrix
        
        # SPEED TIP: If this loop is still slow and you have hundreds of nodes per room, 
        # swap `ot.emd2` to `ot.sinkhorn2(p, q, M, reg=0.1)` for a fast approximation.
        costs[i] = float(ot.sinkhorn2(p, q, M, reg=0.05))

    return costs

def score_rooms_from_summed_features(image_scores, labels, num_rooms):
    """
    Converts (N_db,) image scores into (num_rooms,) scores.
    Each room's score is simply the score of its best-matching image.
    """
    labels = np.array(labels)
    
    # Initialize with a very low number (e.g., -1.0 for cosine similarity)
    room_scores = np.full(num_rooms, -1.0)
    
    for room_id in range(num_rooms):
        in_room_mask = (labels == room_id)
        
        if np.any(in_room_mask):
            # The room's score is the max similarity of any image inside it
            room_scores[room_id] = np.max(image_scores[in_room_mask])
            
    return room_scores

def scores_img_voting_soft(similarity_matrix, labels, num_rooms):
    """
    Thinking of outputing a scoring np array by room label
    inputs:
    - similarity_matrix (num_views, db_images)
    - labels (db_images, ) int matrix 
    
    output:
    - scores (num_rooms, )
    """
    
    # how I think this should work
    # scores[label_idx] has higher score if more camera are more confident about that room
    # I dont know how to get this
    # what I do have
    # similarity matrix M: M[r][c] means how confident at camera r that its looking at image c in the DB, which belongs to room labels[c]
    # maybe scores[i] = sum
    
    similarity_matrix = similarity_matrix **3
    room_scores = np.zeros(num_rooms)
    
    for room_id in range(num_rooms):
        # 1. Find all database images that belong to this specific room
        in_room_mask = (labels == room_id)
        
        # 2. Isolate the similarity scores for just this room's images
        # Shape: (num_views, images_in_this_room)
        room_sims = similarity_matrix[:, in_room_mask]
        
        # 3. For every camera, find its highest similarity to ANY image in this room
        # Shape: (num_views, )
        if 0 in room_sims.shape:
            continue
        cam_max_confidences = np.max(room_sims, axis=1)
        
        # 4. Sum up the confidences from all cameras to get the final room score
        room_scores[room_id] = np.sum(cam_max_confidences)
        
    return room_scores

def scores_img_voting_hard(similarity_matrix, labels, num_rooms):
    """
    Hard Voting with Similarity Tie-Breakers.
    """
    labels = np.array(labels)
    room_scores = np.zeros(num_rooms)
    
    # 1. Each camera picks its SINGLE absolute best match across the entire database
    best_image_indices = np.argmax(similarity_matrix, axis=1)  # Shape: (C, )
    
    # 2. Look up which rooms those winning images belong to
    camera_votes = labels[best_image_indices]                  # Shape: (C, )
    
    # 3. Tally the votes! (1 full point per vote)
    for room_id in camera_votes:
        room_scores[room_id] += 1.0
        
    # 4. Tie-Breaker
    # If Room A and Room B both get 2 votes, how do we rank them?
    # We add a tiny fraction of the actual cosine similarity to break the tie.
    for cam_idx, winning_img_idx in enumerate(best_image_indices):
        winning_room = labels[winning_img_idx]
        sim_score = similarity_matrix[cam_idx, winning_img_idx]
        
        # Add a scaled similarity (e.g., 0.001 * score) so it only acts as a decimal tie-breaker
        room_scores[winning_room] += (sim_score * 0.001)
        
    return room_scores

# ------------------------------
# Database Preparation (Unchanged)
# ------------------------------
def prepare_embeddings(vision_transformer, image_db, methods):
    res = {k: {} for k in methods}
    images = image_db["images"]
    i_labels = image_db["i_label"]
    
    for method, cfg in methods.items():
        encoder = ImageEncoder(vision_transformer)
        embs = encoder.embed(images, cfg["patches"], cfg["agg_method"], cfg["num_clusters"], grayscale=cfg["grayscale"])
        res[method] = {
            "embeddings": embs,
            "labels": i_labels,
            "encoder": encoder
        }    
        
    return res

# ------------------------------
# Main Evaluation Loop
# ------------------------------
def evaluate_localization(
    ds, ground_truth_labels, cam_layout, map_data, dino_encoder, iphone_image_db, start=0, end=None, step=1,
    pooling="mean", eps=1e-8
):
    if end is None: end = len(ds)
    methods = {
        # "domain_patch_vlad": {"patches": True, "agg_method": "domain_vlad", "num_clusters":32, "grayscale": False},
        "patch_vlad": {"patches": True, "agg_method": "vlad", "num_clusters":32, "grayscale": False},
        # "patch_gem": {"patches": True, "agg_method": "gem"},
        # "patch_gmp": {"patches": True, "agg_method": "gmp"},
        # "patch_gap": {"patches": True, "agg_method": "gap"}
    }
    encoders = prepare_embeddings(dino_encoder, iphone_image_db, methods)
    
    print("Finished preparing embeddings for all methods. Starting evaluation...\n")
    
    results = {m: {"Acc": 0.0, "Hit@3": 0.0, "Hit@5": 0.0, "MRR": 0.0, "N": 0} for m in methods}
    
    
    # SPEEDUP: Instead of embedding each frame one by one, we will embed all frames in a batch and store their embeddings along with timestamps.
    all_frames_list = []
    ts_list = []
    
    for t in range(start, end, step):
        snapshot = ds[t]
        for cam in cam_layout:
            frame = snapshot.cameras.get(cam)
            if frame is not None:
                # Handle whether it's an object with an 'image' attribute or a raw array
                all_frames_list.append(frame.image)
                ts_list.append(t)
        

    max_H = max(frame.shape[0] for frame in all_frames_list)
    max_W = max(frame.shape[1] for frame in all_frames_list)
    all_frames = np.array([np.array(Image.fromarray(frame).resize((max_W, max_H))) for frame in all_frames_list])
                
    ts = np.array(ts_list)                  # (Total_Frames,)

    # Embed all collected frames at once for each method (much faster than per-frame embedding)
    Xs = {}
    for method, cfg in methods.items():
        print(f"Embedding queries for method: {method}")
        encoder = encoders[method]["encoder"]
        
        # Embed all collected frames at once
        embs = encoder.embed(all_frames, patches=cfg["patches"], agg_method=cfg["agg_method"], num_clusters=cfg['num_clusters'], grayscale=cfg['grayscale'], save=False, load=False)
        Xs[method] = {"X": embs, "ts": ts}
                
    for method, cfg in methods.items():
        for t in tqdm(range(start, end, step)):
            gt_room = ground_truth_labels[t]
            gt_idx = map_data['label2i'][gt_room]
            
            X = Xs[method]["X"][Xs[method]["ts"] == t]  # (C, D)
            
            # trying out sth, collapsing all camera views features together
            # X = X.sum(0)
            
            if X.shape[0] == 0:
                print(f"No frames found for timestamp {t}. Skipping...")
                continue
            
            # from here, prediction for each t is based on majority prediction of all C frames (voting logic)
            # I need to get
            # - scores [N,] for all images
            # - hot coded correct [N,] for all images (1 if correct match, 0 otherwise)
            # - note that multiple images can match to the same room, so hot coded correct can have multiple 1s
            cosine_similarity_scores = np.array(cosine_similarity_jax(X, encoders[method]["embeddings"]))
            
            # mean_scores = cosine_similarity_scores  # (N_db,)
            # hot_coded_correct = (np.array(encoders[method]["labels"]) == gt_idx).astype(np.float32)  # (N_db,)
            # metrics, _, _ = compute_metrics_from_scores(mean_scores, hot_coded_correct, ks=(1, 3, 5))

            # mean_scores = scores_img_voting_hard(cosine_similarity_scores, encoders[method]["labels"], len(map_data['label2i']))
            summed_image_scores = np.sum(cosine_similarity_scores, axis=0)
            mean_scores = score_rooms_from_summed_features(summed_image_scores, encoders[method]["labels"], len(map_data['label2i']))
            hot_coded_correct = np.zeros((len(map_data['label2i']), ))
            hot_coded_correct[gt_idx] = 1
            metrics, _, _ = compute_metrics_from_scores(mean_scores, hot_coded_correct)

            results[method]["Acc"] += metrics["Acc"]
            results[method]["Hit@3"] += metrics["Hit@3"]
            results[method]["Hit@5"] += metrics["Hit@5"]
            results[method]["MRR"] += metrics["MRR"]
            results[method]["N"] += 1

            print(f"""Frame {t:03d} Processed | GT={gt_room} | Predicted={map_data['i2label'][np.argmax(mean_scores)]} | running acc@1: {[f'{method}: {100*(results[method]["Acc"]/(t+1)):.2f}%' for method in methods]}""")

        print("finished evaluating method:", method)
        # print("accuracy per class: ")
        
    print("\n" + "=" * 80)
    print(f"{'Method':<25} | {'N':>6} | {'Acc':>7} | {'Hit@3':>7} | {'Hit@5':>7} | {'MRR':>7}")
    print("-" * 80)
    for m, r in results.items():
        N = max(1, r["N"])
        print(f"{m:<25} | {r['N']:6d} | {100*(r['Acc']/N):6.2f}% | {100*(r['Hit@3']/N):6.2f}% | {100*(r['Hit@5']/N):6.2f}% | {r['MRR']/N:6.3f}")
    print("=" * 80 + "\n")
    return results

def get_ground_truth_label(frame_index):
    if 0 <= frame_index <= 3: return "INSITE Lab room 1"
    elif 4 <= frame_index <= 7: return "Miller Street 1"
    elif 8 <= frame_index <= 38: return "classroom1"
    elif 39 <= frame_index <= 69: return "Miller Street 1"
    elif 70 <= frame_index <= 118: return "classroom2"
    elif 119 <= frame_index < 1000: return "Miller Street 2" 
    else: return "Unknown"

# ------------------------------
# Main (Setup)
# ------------------------------
if __name__ == "__main__":
    logger = build_logger()
    ds = SpotDataset("dataset/spot/millerst/data")
    
    sg_path = "graph_dataset/graph.json"
    tracker, _ = load_full_tracker("graph_dataset/raw_extration/checkpoints", logger)
    assign_rooms_from_json(tracker, "graph_dataset/rooms.json")
    graph = json.load(open(sg_path, 'r'))

    # detection_model = YOLODetector(cfg)
    # siglip = SiglipModel(cfg)
    dino = DinoModel(cfg)

    # SPEEDUP: Do this once globally, not every frame
    # filtered_classes = [n['class_name'] for n in graph['nodes'] if BAD_CHARS.isdisjoint(n['class_name'])]
    # detection_model.model.set_classes(filtered_classes)
    # detection_model.class_names = filtered_classes

    with open("graph_dataset/coarse_embedding_field.pkl", "rb") as f: coarse_dict = pkl.load(f)
    coarse_embeddings_field = coarse_dict["coarse_field"]
    room_id_proto = coarse_dict["room_proto"]
    croom_name_to_id = coarse_dict["name_to_id"]
    croom_id_to_name = coarse_dict["id_to_name"]
    with open("VPR_im2im/image_db.pkl", 'rb') as file: iphone_image_db = pkl.load(file)

    room_labels = list(room_id_proto.keys())
    room_embeddings = np.array(list(room_id_proto.values()))
    
    room_node_embeddings, room_node_labels = [], []
    for obj in tracker.objects:
        room_node_embeddings.append(obj.clip_ft)
        room_node_labels.append(room_labels.index(obj.room))

    map_data = {
        'room_embeddings': room_embeddings,
        'room_labels': room_labels,
        'node_embeddings': np.array(room_node_embeddings).squeeze(),
        'node_labels': np.array(room_node_labels).squeeze(),
        'label2i': croom_name_to_id,
        'i2label': croom_id_to_name
    }

    ground_truth = np.array([get_ground_truth_label(i) for i in range(len(ds))])

    CAM_LAYOUT = [
        "hand_color_image",
        "frontright_fisheye_image", "frontleft_fisheye_image",
        # "left_fisheye_image", "right_fisheye_image",
        # "back_fisheye_image"
    ]
    evaluate_localization(
        ds=ds,
        ground_truth_labels=ground_truth,
        cam_layout=CAM_LAYOUT,
        map_data=map_data,
        dino_encoder=dino,
        iphone_image_db=iphone_image_db,
        start=0,
        end=len(ds),
        pooling="mean",
    )