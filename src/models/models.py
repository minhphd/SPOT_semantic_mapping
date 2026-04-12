import os
from groq import Groq
import time
import torch
import configparser
import cv2
from openai import OpenAI
import numpy as np
from PIL import Image
from utils.crops import concat_crops_horizontal
from io import BytesIO
import base64


from ultralytics import YOLO, SAM, FastSAM
from transformers import (
    AutoModel,
    AutoImageProcessor,
    AutoProcessor,
    Blip2Processor,
    Blip2ForConditionalGeneration,
)

try:
    config = configparser.ConfigParser()
    config.read('src/configs/api_keys.ini')
except Exception as e:
    raise e
    

# ============================================================
#                   Image Segmentation Models
# ============================================================

class YOLODetector():
    """
    YOLOv11/YOLOv8 for detection only.
    Returns:
        bboxes   (N, 4)
        class_id (N,)
        conf     (N,)
    """

    def __init__(self, cfg):
        self.cfg = cfg
        if cfg.segmentation["detector"] == "yolov11":
            ckpt = cfg.paths["yolo11_ckpt"]
        elif cfg.segmentation["detector"] == "yolov8":
             ckpt = cfg.paths["yolo8_ckpt"]
        else:
            raise ValueError(f"Invalid detector: {cfg.segmentation['detector']}")
        print(f"[YOLO Detector] Loading {ckpt}")

        self.model = YOLO(ckpt)
        self.model.to(cfg.device)

        # Set open-world class names
        self.class_names = cfg.landmarks["classes"]
        self.model.set_classes(self.class_names)

    def __call__(self, rgb):
        """rgb: np.ndarray (H,W,3)"""
        results = self.model(rgb, device=self.cfg.device, verbose=False)[0]

        if results.boxes is None or len(results.boxes) == 0:
            return [], [], []

        b = results.boxes
        bboxes = b.xyxy.cpu().numpy()
        class_ids = b.cls.cpu().numpy().astype(int)
        conf = b.conf.cpu().numpy()

        return bboxes, class_ids, conf

class FastSAMPredictor():
    """
    FastSAM segmentation backend (extremely fast, promptable).
    Good for large-object segmentation and grounding.

    Supports:
        - Free-form prompt ("big objects", "furniture", etc.)
        - Retina masks
    """

    def __init__(self, cfg):
        ckpt = cfg.paths["fastsam_ckpt"]
        self.device = cfg.device
        print(f"[FastSAM] Loading {ckpt} → {self.device}")

        self.model = FastSAM(ckpt)

    def __call__(self, rgb_full, prompt=None):
        H, W = rgb_full.shape[:2]
        imgsz = max(H, W)

        results = self.model(
            rgb_full,
            texts=prompt,
            device=self.device,
            retina_masks=True,
            imgsz=imgsz,
            verbose=False,
        )

        if results[0].masks is None:
            return []
        return results[0].masks.data.cpu().numpy().astype(bool)

class MobileSAMPredictor():
    """
    Mobile-SAM (efficient SAM).
    Useful when SAM-like quality is needed but on-device speed matters.
    """

    def __init__(self, cfg):
        ckpt = cfg.paths.get("mobile_sam_ckpt")
        self.device = cfg.device
        if ckpt is None:
            raise ValueError("Missing `mobile_sam_ckpt` in cfg.paths.")

        print(f"[MobileSAM] Loading {ckpt} → {self.device}")

        self.model = SAM(ckpt)
        self.model.to(self.device)

    def __call__(self, rgb_full, bboxes, verbose=False, prompt=None):
        if prompt:
            raise NotImplementedError("MobileSAMPredictor does not support text prompt-based segmentation.")
        results = self.model.predict(rgb_full, bboxes=bboxes, device=self.device, verbose=verbose)
        if results[0].masks is None:
            return []
        return results[0].masks.data.cpu().numpy().astype(bool)

class SAM2Predictor():
    """
    SAM2 (Segment Anything 2).
    Highest accuracy among the SAM family, slowest runtime.

    Wrapped the same way as SAM/SAM2 from ultralytics.
    """

    def __init__(self, cfg):
        ckpt = cfg.paths.get("sam2_ckpt")
        self.device = cfg.device
        if ckpt is None:
            raise ValueError("Missing `sam2_ckpt` in cfg.paths.")

        print(f"[SAM2] Loading {ckpt} → {self.device}")

        self.model = SAM(ckpt)   # ultralytics SAM loads SAM2 models too
        self.model.to(self.device)

    def __call__(self, rgb_full, bboxes, verbose=False, prompt=None):
        if prompt:
            raise NotImplementedError("MobileSAMPredictor does not support text prompt-based segmentation.")
        results = self.model.predict(rgb_full, bboxes=bboxes, device=self.device, verbose=verbose)
        if results[0].masks is None:
            return []
        return results[0].masks.data.cpu().numpy().astype(bool)

# ============================================================
#                   Image Encoder Models
# ============================================================
class DinoModel:
    """
    DINO / DINOv2 image encoder (image-only).
    Provides:
        - embed_images() → vision embeddings

    Notes:
        - No text tower (embed_texts not supported).
        - Embedding is taken from CLS token if available, else mean pooled patch tokens.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.device

        ckpt = cfg.paths["dino_ckpt"]  # e.g. "facebook/dinov2-base"
        print(f"[DINO] Loading {ckpt} → {self.device}")

        self.processor = AutoImageProcessor.from_pretrained(ckpt, use_fast=True)
        self.model = AutoModel.from_pretrained(ckpt,
            dtype=torch.float16,
            device_map="auto",
            attn_implementation="sdpa").to(self.device)
        self.model.eval()

    def embed_images(self, crops_pil):
        if len(crops_pil) == 0:
            return None

        with torch.no_grad():
            inputs = self.processor(images=crops_pil, return_tensors="pt").to(self.device)
            out = self.model(**inputs)

            # Common HF output: last_hidden_state [B, T, D]
            # For ViT-like models: token 0 is CLS; tokens 1.. are patches.
            if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
                h = out.last_hidden_state  # [B, T, D]
                if h.shape[1] >= 1:
                    # prefer CLS token if it exists
                    feats = h[:, 0, :]  # [B, D]
                else:
                    feats = h.mean(dim=1)
            # Some models expose pooler_output
            elif hasattr(out, "pooler_output") and out.pooler_output is not None:
                feats = out.pooler_output  # [B, D]
            else:
                raise RuntimeError("DINO model output does not contain last_hidden_state/pooler_output")

            feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-12)

        return feats.cpu().numpy()
    
    def embed_images_by_patch(self, crops_pil):
        if len(crops_pil) == 0:
            return None
        
        patch_size = self.model.config.patch_size if hasattr(self.model.config, "patch_size") else 16
        with torch.no_grad():
            inputs = self.processor(images=crops_pil, return_tensors="pt").to(self.device)
            _, _, img_height, img_width = inputs.pixel_values.shape
            num_patches_height, num_patches_width = img_height // patch_size, img_width // patch_size
            _ = num_patches_height * num_patches_width

            last_hidden_states = self.model(**inputs)[0]
            
            patch_features = last_hidden_states[:, 1:, :].unflatten(1, (num_patches_height, num_patches_width))
            
        patch_features = patch_features / (patch_features.norm(dim=-1, keepdim=True) + 1e-12)
        return patch_features.cpu().numpy()

    def embed_texts(self, texts):
        raise NotImplementedError("DINO is image-only; no text embeddings.")

    def classify_caption(self, caption):
        raise NotImplementedError("DINO is image-only; no caption classification.")
    

class SiglipModel():
    """
    SigLIP image encoder (open-set).
    Provides:
        - embed_images() → vision embeddings
        - classify_caption() → open-set text classification
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.device

        ckpt = cfg.paths["siglip_ckpt"]
        print(f"[SigLIP] Loading {ckpt} → {self.device}")

        self.processor = AutoProcessor.from_pretrained(ckpt)
        self.model = AutoModel.from_pretrained(ckpt).to(self.device)
        self.model.eval()

        self.class_names = cfg.landmarks["classes"]
        self.class_embs = self._build_text_embeddings()

    # ------------------------------
    # Compute image embeddings
    # ------------------------------
    def embed_images(self, crops_pil):
        if len(crops_pil) == 0:
            return None

        with torch.no_grad():
            inputs = self.processor(images=crops_pil, return_tensors="pt").to(self.device)
            feats = self.model.get_image_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)

        return feats.cpu().numpy()

    def embed_images_by_patch(self, crops_pil):
        """Return per-patch features (B, H_p, W_p, D) — mirrors DinoModel.embed_images_by_patch.

        Note: SigLIP has no CLS token — last_hidden_state is already all patch tokens,
        so we do NOT skip index 0 (unlike DINOv2 where token 0 is CLS).
        """
        if len(crops_pil) == 0:
            return None

        with torch.no_grad():
            inputs = self.processor(images=crops_pil, return_tensors="pt").to(self.device)
            vision_outputs = self.model.vision_model(**inputs)
            patch_size = self.model.config.vision_config.patch_size
            img_size   = self.model.config.vision_config.image_size
            n = img_size // patch_size
            # SigLIP: last_hidden_state is (B, n*n, D) — all tokens are patches, no CLS
            patch_feats = vision_outputs.last_hidden_state    # (B, n*n, D)
            patch_feats = patch_feats.unflatten(1, (n, n))             # (B, n, n, D)

        patch_feats = patch_feats / (patch_feats.norm(dim=-1, keepdim=True) + 1e-12)
        return patch_feats.cpu().numpy()

    def embed_texts(self, texts):
        if len(texts) == 0:
            return None

        with torch.no_grad():
            inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
            feats = self.model.get_text_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)

        return feats.cpu().numpy()

    # ------------------------------
    # Precompute text embeddings
    # ------------------------------
    def _build_text_embeddings(self):
        with torch.no_grad():
            text_inputs = self.processor(
                text=self.class_names,
                return_tensors="pt",
                padding=True,
            ).to(self.device)

            feats = self.model.get_text_features(**text_inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)

        return feats.cpu().numpy()

    # ------------------------------
    # Caption → open-set class
    # ------------------------------
    def classify_caption(self, caption):
        with torch.no_grad():
            inputs = self.processor(
                text=[caption],
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            emb = self.model.get_text_features(**inputs)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            emb_np = emb.cpu().numpy().reshape(-1)

        sims = emb_np @ self.class_embs.T
        return self.class_names[int(sims.argmax())]


# ============================================================
#                   Image Captioning Models
# ============================================================

class BlipCaptioner():
    """
    BLIP-2 caption model (Flan-T5-XL).

    Designed to:
      - fuse multi-view crops horizontally
      - produce compact noun-phrase captions ("wooden table", "black sofa")

    Very powerful for open-set object recognition.
    """

    def __init__(self, cfg):
        print("[BLIP-2] Loading blip2-flan-t5-xl")

        self.device = cfg.device
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.processor = Blip2Processor.from_pretrained(cfg.paths["blip_ckpt"])
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            cfg.paths["blip_ckpt"],
            torch_dtype=dtype,
        ).to(self.device)
        self.model.eval()

    def caption_single(self, crop_np, max_new_tokens=40):
        """
        Generate a caption for a single image crop.

        Parameters:
            crop_np: np.ndarray         (H, W, 3) RGB array
            max_new_tokens: int         maximum caption length

        Returns:
            caption string
        """
        if crop_np is None:
            return "(no caption)"

        pil_img = Image.fromarray(crop_np)

        prompt = (
            "Describe the central object in the image concisely in under 5 words. "
            "Ignore irrelevant background."
        )

        with torch.no_grad():
            inputs = self.processor(
                pil_img,
                text=prompt,
                return_tensors="pt"
            ).to(self.device)

            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens
            )

        return self.processor.decode(out[0], skip_special_tokens=True).strip()

    def caption_multi(self, crops_np, max_new_tokens=40):
        """
        Generate a caption for multi-view crops.

        Parameters:
            crops_np: List[np.ndarray]   list of (H, W, 3) RGB arrays
            max_new_tokens: int         maximum caption length

        Returns:
            caption string
        """
        fused = concat_crops_horizontal(crops_np)
        if fused is None:
            return "(no caption)"

        # Debug save
        # os.makedirs("saved_crops", exist_ok=True)
        # cv2.imwrite("saved_crops/fused_debug.jpg",
        #             cv2.cvtColor(fused, cv2.COLOR_RGB2BGR))

        pil_img = Image.fromarray(fused)

        prompt = (
            "These images show different views of the same object. "
            "Describe the object concisely in under 5 words. "
            "Ignore irrelevant background."
        )

        with torch.no_grad():
            inputs = self.processor(
                pil_img,
                text=prompt,
                return_tensors="pt"
            ).to(self.device)

            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens
            )

        return self.processor.decode(out[0], skip_special_tokens=True).strip()

# ============================================================
#                   Commercial APIs
# ============================================================

class GroqModel():
    def __init__(self, model):
        if model not in {'meta-llama/llama-4-scout-17b-16e-instruct', 'meta-llama/llama-4-maverick-17b-128e-instruct'}:
            raise ValueError(f"Unsupported vision model: {model}")
        self.model = model
        self.client = Groq(api_key = config['API_KEYS']['groq_api'])
        
    def __call__(self, input, image=None):
        content = [
            {
                "type": "text",
                "text": input
            }
        ]
        if image is not None:
            # encode image into base64 string
            buffered = BytesIO()
            image = Image.fromarray(image)
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img_str}"
                }
            })
        
        try:
            chat_completion = self.client.chat.completions.create(
                messages = [
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                model = self.model
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            print(f"Error during Groq API call: {e}")
            time.sleep(60)  # wait before retrying
            self.__call__(input, image)
        


class OpenaiEmbedding():
    def __init__(self, model):
        if model not in {'text-embedding-3-small', 'text-embedding-3-large', 'text-embedding-ada-002'}:
            raise ValueError(f"Unsupported embedding model: {model}")
        self.model = model
        self.client = OpenAI(api_key = config['API_KEYS']['openai_api'])
        
    def __call__(self, input):
        """
        Parameters
        ----------
        input : str | list[str]
            Text(s) to embed.
        Returns
        -------
        np.ndarray
            Shape:
              - (D,) if input is a single string
              - (N, D) if input is a list of strings
        """
        # Normalize input
        single = False
        if isinstance(input, str):
            input = [input]
            single = True

        response = self.client.embeddings.create(
            model=self.model,
            input=input,
        )

        embeddings = np.array(
            [item.embedding for item in response.data],
            dtype=np.float32,
        )

        return embeddings[0] if single else embeddings


class OpenaiModel():
    def __init__(self, model):
        if model not in {'gpt-5.1', 'gpt-5-mini', 'gpt-4.1-mini'}:
            raise ValueError(f"Unsupported vision model: {model}")
        self.model = model
        self.client = OpenAI(api_key = config['API_KEYS']['openai_api'])
        
    def __call__(self, input, image=None):
        content = [
            {
                "type": "input_text",
                "text": input
            }
        ]
        if image is not None:
            # encode image into base64 string
            buffered = BytesIO()
            image = Image.fromarray(image)
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            content.append({
                "type": "input_image",
                "image_url": f"data:image/png;base64,{img_str}"
            })
        
        try:
            chat_completion = self.client.responses.create(
                input = [
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                model = self.model
            )
        except Exception as e:
            print(f"Error during OpenAI API call: {e}")
        
        return chat_completion.output_text
