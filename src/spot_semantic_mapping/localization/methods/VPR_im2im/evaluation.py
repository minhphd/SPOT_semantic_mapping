from models.models import DinoModel, SiglipModel
from configs.loader import cfg
from utils.logger import *
import numpy as np
from utils.metrics import compute_metrics_from_scores
from make_dataset import load_data, load_spot_data
from tqdm import tqdm
from spot_semantic_mapping.localization.methods.VPR_im2im.localization import localize, localize_at_t, prepare_embeddings
from collections import deque
import pickle as pkl

# POT (Python Optimal Transport)
try:
    import ot
except ImportError:
    ot = None

BAD_CHARS = {'$', '#', '@', '!', '%', '^', '&', '*', '(', ')', '-', '+', '=', 
             '{', '}', '[', ']', '|', '\\', ':', ';', '"', "'", '<', '>', ',', '.', '?', '/'}
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

        # "patch_gem": {"patches": True, "agg_method": "gem", "num_clusters":32, "grayscale": False},

        # "patch_gmp": {"patches": True, "agg_method": "gmp", "num_clusters":32, "grayscale": False},

        # "patch_gap": {"patches": True, "agg_method": "gap", "num_clusters":32, "grayscale": False}
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
            image_scores, correct_db_indices, _ = localize_at_t(encoded_data, ground_truth, method, t, n_views=10)
            
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


# ------------------------------
# Main (Setup)
# ------------------------------
if __name__ == "__main__":
    logger = build_logger()
    
    dino = DinoModel(cfg)

    with open("src/spot_semantic_mapping/localization/methods/VPR_im2im/dataset/spot_dataset_w_gt_5m.pkl", 'rb') as file: 
        dataset = pkl.load(file)

    # dataset = load_data("data/AnyLoc2023-Public-Data/Public/Datasets-All/17places")

    # We no longer pass map_data or ground_truth_labels since we aren't doing room-classification
    evaluate_localization(
        dataset=dataset,
        dino_encoder=dino
    )