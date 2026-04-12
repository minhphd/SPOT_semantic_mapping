import pandas as pd
import numpy as np
from utils.jax_helper import cdist, cosine_similarity_jax
from spot_semantic_mapping.localization.methods.VPR_im2im.img_encoder import ImageEncoder


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
        embs = encoder.embed(images, cfg["patches"], cfg["agg_method"], cfg["num_clusters"], grayscale=cfg["grayscale"])
        
        print(f"Embedding queries for method: {method}")
        X = encoder.embed(query_frames, patches=cfg["patches"], agg_method=cfg["agg_method"], num_clusters=cfg['num_clusters'], grayscale=cfg['grayscale'], save=False)
        
        res[method] = {
            "embeddings": embs,
            "encoder": encoder,
            "X": X,
            "ts": ts
        } 
        
    return res


def localize(X_emb, db_emb):
    im_scores = cosine_similarity_jax(X_emb, db_emb)
    sorted_ind = np.argsort(-im_scores)
    return im_scores, sorted_ind


def localize_at_t(encoded_data, ground_truth, method, t, n_views=1):
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
    dist_matrix = cdist(retrieved_images_pos, n_pos)
    
    # Mask: True if a node is within 'window' distance to ANY of the top_k retrieved images
    node_mask = np.array(np.any(dist_matrix <= window, axis=0), dtype=bool).flatten()
    
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

    return {"nodes": sub_nodes, "edges": sub_edges}


def print_subgraph(subgraph):
    """
    Parses the subgraph DataFrames and returns a formatted string summary of the state.
    """
    sub_nodes, sub_edges = subgraph['nodes'], subgraph['edges']
    
    # Safety check: if there are no nodes in the subgraph, return a default string
    if sub_nodes.empty:
        return "="*40 + "\n📍 SUBGRAPH STATE SUMMARY\n" + "="*40 + "\n  No objects detected within the current window.\n" + "="*40 + "\n"

    room_counts = sub_nodes['room'].value_counts()
    likely_room = room_counts.idxmax() # The room with the most objects in the window

    # Initialize a list to hold all the lines of our text output
    summary = []
    
    summary.append("\n" + "="*40)
    summary.append("📍 SUBGRAPH STATE SUMMARY")
    summary.append("="*40)
    summary.append(f"Likely Current Room: **{likely_room}**\n")
    
    summary.append("📊 Objects per room (within window):")
    for room, count in room_counts.items():
        summary.append(f"  - {room}: {count} objects")
    
    summary.append(f"\n🛋️  Objects detected in {likely_room}:")
    room_objects = sub_nodes[sub_nodes['room'] == likely_room]['class_name'].tolist()
    summary.append(f"  {', '.join(room_objects)}")
    
    summary.append("\n🔗 Object Relations (Induced Subgraph):")
    if not sub_edges.empty:
        for _, row in sub_edges.iterrows():
            summary.append(f"  - [{row['src_name']}] {row['relation']} [{row['dst_name']}]")
    else:
        summary.append("  - No local relations found between these objects.")
    
    summary.append("="*40 + "\n")
    
    # Join all the lines with a newline character and return the single string
    return "\n".join(summary)