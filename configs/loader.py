import yaml
import json

class Config:
    def __init__(self, path="config/config.yaml"):
        with open(path, "r") as f:
            self._cfg = yaml.safe_load(f)

    def __getitem__(self, key):
        return self._cfg[key]

    def get(self, key, default=None):
        return self._cfg.get(key, default)

    def get_json(self):
        return json.dumps(self._cfg)

    def __getattr__(self, item):
        return self._cfg.get(item)
    

# Global config singleton
cfg = Config("configs/MinhNet.yml")
