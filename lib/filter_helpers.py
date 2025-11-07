import os


ASSETS_DIR = "https://blog-assets.cloud.pennylane.ai/demos/"

def parse_img_source(src: str) -> str:
    if "../_static/" in src:
        return src.replace("../_", ASSETS_DIR + os.getenv("CURRENT_DEMO") + "/main/_assets/")
    else:
        return src
