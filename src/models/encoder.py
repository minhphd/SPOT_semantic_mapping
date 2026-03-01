class CustomEncoder():
    def __init__(self, cfg):
        super().__init__()

    def embed_images_by_patch(self, images):
        return self.vision_encoder_img(images)

    def embed_captions(self, captions):
        if self.vision_encoder_text is None:
            raise ValueError("Caption encoder not initialized.")
        return self.vision_encoder_text(captions)