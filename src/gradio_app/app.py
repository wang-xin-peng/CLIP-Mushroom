import gradio as gr
import torch
from PIL import Image
import numpy as np
import io
import os
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models.clip_wrapper import CLIPWrapper
from src.utils.faiss_index import FaissIndex
from src.utils.gradcam import CLIPGradCAM
from src.utils.prompts import PROMPT_TEMPLATES
from src.data.split import get_metadata_splits, split_species

# Global state (loaded once on first request)
clip = None
faiss_index = None
gradcam = None
species_list = None
label_map = None
rev_label_map = None


def load_models():
    global clip, faiss_index, gradcam, species_list, label_map, rev_label_map
    if clip is not None:
        return
    print("Loading models for Gradio app...")
    clip = CLIPWrapper()
    gradcam = CLIPGradCAM(clip)

    # Load FAISS index
    index_path = "outputs/features/faiss_index.bin"
    if os.path.exists(index_path):
        faiss_index = FaissIndex()
        faiss_index.load(index_path)
        print(f"FAISS index loaded: {len(faiss_index.metadata)} images")
    else:
        print("Warning: FAISS index not found. Build it first with --build-index.")

    # Load label map
    label_map_path = "outputs/features/label_map.pkl"
    if os.path.exists(label_map_path):
        with open(label_map_path, "rb") as f:
            data = pickle.load(f)
            rev_label_map = data["rev_label_map"]
            label_map = data["label_map"]

    # Species list for zero-shot
    train_df, _, _ = get_metadata_splits()
    seen, unseen = split_species(train_df)
    species_list = sorted(seen | unseen)
    print(f"Loaded {len(species_list)} species for zero-shot classification.")


# ---------- Tab 1: Zero-shot Recognition ----------

def recognize_image(img):
    load_models()
    processor = clip.processor
    image = Image.fromarray(img.astype("uint8"), "RGB")
    inputs = processor(images=image, return_tensors="pt")

    # Zero-shot prediction with best template (index 0)
    best_template = PROMPT_TEMPLATES[0]
    probs, _ = clip.zero_shot_predict(
        inputs["pixel_values"], species_list, best_template
    )
    top5_probs, top5_indices = torch.topk(probs[0], 5)

    # Build label output
    pred_dict = {}
    for p, idx in zip(top5_probs, top5_indices):
        pred_dict[species_list[idx]] = float(p)

    # Generate heatmap
    fig = gradcam.visualize(inputs["pixel_values"], None)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    heatmap_img = Image.open(buf)

    return pred_dict, heatmap_img


# ---------- Tab 2: Image Search ----------

def search_by_image(img, k=10):
    load_models()
    if faiss_index is None:
        return [], "FAISS index not loaded. Build it first."
    processor = clip.processor
    image = Image.fromarray(img.astype("uint8"), "RGB")
    inputs = processor(images=image, return_tensors="pt")
    query_feat = clip.encode_images(inputs["pixel_values"])
    results = faiss_index.search(query_feat, k=k)

    gallery = []
    for r in results:
        label_name = rev_label_map.get(int(r["label"]), r.get("label", "unknown"))
        caption = f"{label_name} (score: {r['score']:.3f})"
        gallery.append((r["path"], caption))
    return gallery


# ---------- Tab 3: Text Search ----------

def search_by_text(query, k=10):
    load_models()
    if faiss_index is None:
        return [], "FAISS index not loaded. Build it first."
    if not query.strip():
        return [], "Please enter a text description."
    text_feat = clip.encode_text([query])
    results = faiss_index.search(text_feat, k=k)

    gallery = []
    for r in results:
        label_name = rev_label_map.get(int(r["label"]), r.get("label", "unknown"))
        caption = f"{label_name} (score: {r['score']:.3f})"
        gallery.append((r["path"], caption))
    return gallery


# ---------- Build App ----------

def create_app():
    with gr.Blocks(
        title="CLIP-Mushroom: Fungi Recognition & Retrieval",
        theme=gr.themes.Soft(),
        css="footer {visibility: hidden}",
    ) as app:
        gr.Markdown(
            "# CLIP-Mushroom: 真菌细粒度识别与多模态检索系统"
        )

        with gr.Tab("零样本识别 (Zero-shot)"):
            with gr.Row():
                with gr.Column():
                    input_img = gr.Image(label="上传真菌图片")
                    recognize_btn = gr.Button("识别", variant="primary")
                with gr.Column():
                    output_preds = gr.Label(label="预测结果 (Top-5)")
                    output_heatmap = gr.Image(label="注意力热力图")
            recognize_btn.click(
                fn=recognize_image,
                inputs=input_img,
                outputs=[output_preds, output_heatmap],
            )

        with gr.Tab("以图搜图 (Image Search)"):
            with gr.Row():
                with gr.Column():
                    search_img = gr.Image(label="上传查询图片")
                    img_k = gr.Slider(
                        5, 50, value=10, step=5, label="返回数量 K"
                    )
                    img_search_btn = gr.Button("搜索", variant="primary")
                with gr.Column():
                    img_gallery = gr.Gallery(
                        label="检索结果", columns=5, height="auto"
                    )
            img_search_btn.click(
                fn=search_by_image,
                inputs=[search_img, img_k],
                outputs=img_gallery,
            )

        with gr.Tab("文本搜图 (Text Search)"):
            with gr.Row():
                with gr.Column():
                    text_query = gr.Textbox(
                        label="输入文字描述",
                        placeholder="e.g., red mushroom with white spots",
                    )
                    text_k = gr.Slider(
                        5, 50, value=10, step=5, label="返回数量 K"
                    )
                    text_search_btn = gr.Button("搜索", variant="primary")
                with gr.Column():
                    text_gallery = gr.Gallery(
                        label="检索结果", columns=5, height="auto"
                    )
            text_search_btn.click(
                fn=search_by_text,
                inputs=[text_query, text_k],
                outputs=text_gallery,
            )

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(server_name="127.0.0.1", server_port=7860)
