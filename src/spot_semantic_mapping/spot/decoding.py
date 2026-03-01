import cv2
import numpy as np

def decode_depth(resp):
    """Decode Spot depth image from either RAW uint16 or PNG."""
    rows = resp.shot.image.rows
    cols = resp.shot.image.cols
    fmt = resp.shot.image.format

    if fmt == resp.shot.image.FORMAT_RAW:
        depth = np.frombuffer(resp.shot.image.data, dtype=np.uint16)
        return depth.reshape(rows, cols)

    if fmt == resp.shot.image.FORMAT_PNG:
        arr = np.frombuffer(resp.shot.image.data, dtype=np.uint8)
        depth = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        return depth.astype(np.uint16)

    print("[WARN] Unknown depth format:", fmt)
    return None


def decode_rgb(resp):
    raw = np.frombuffer(resp.shot.image.data, dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    
    # rotate 90 degrees if camera is in {'left_fisheye_image', 'frontright_fisheye_image', 'frontleft_fisheye_image'}
    src = resp.source.name
    rotation = False
    flip = False
    if src in {"frontright_fisheye_image", "frontleft_fisheye_image"}:
        rotation = True
    if src == "right_fisheye_image":
        flip = True
    
    if rotation:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if flip:
        img = cv2.flip(img, 0)
    
    return img
