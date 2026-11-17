import os
import torch
import gradio as gr
from PIL import Image
from diffusers import (
    AutoPipelineForText2Image,
    AutoPipelineForImage2Image,
    StableDiffusionControlNetPipeline,
    ControlNetModel,
)

# 国内下载慢时可以打开下面这行（在 from_pretrained 之前生效）
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

SD_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
SCRIBBLE_MODEL = "lllyasviel/control_v11p_sd15_scribble"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

_pipes = {}


def _finish(pipe):
    pipe = pipe.to(DEVICE)
    if DEVICE == "cuda":
        pipe.enable_attention_slicing()
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
    return pipe


def get_t2i():
    if "t2i" not in _pipes:
        pipe = AutoPipelineForText2Image.from_pretrained(
            SD_MODEL,
            torch_dtype=DTYPE,
            safety_checker=None,
            requires_safety_checker=False,
        )
        _pipes["t2i"] = _finish(pipe)
    return _pipes["t2i"]


def get_i2i():
    if "i2i" not in _pipes:
        pipe = AutoPipelineForImage2Image.from_pretrained(
            SD_MODEL,
            torch_dtype=DTYPE,
            safety_checker=None,
            requires_safety_checker=False,
        )
        _pipes["i2i"] = _finish(pipe)
    return _pipes["i2i"]


def get_scribble():
    if "scribble" not in _pipes:
        cn = ControlNetModel.from_pretrained(SCRIBBLE_MODEL, torch_dtype=DTYPE)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            SD_MODEL,
            controlnet=cn,
            torch_dtype=DTYPE,
            safety_checker=None,
            requires_safety_checker=False,
        )
        _pipes["scribble"] = _finish(pipe)
    return _pipes["scribble"]


def make_generator(seed):
    if seed is None or int(seed) < 0:
        return None
    return torch.Generator(device=DEVICE).manual_seed(int(seed))


def text_to_image(prompt, negative_prompt, steps, guidance, seed):
    if not prompt or not prompt.strip():
        raise gr.Error("请先填写文字描述")
    return get_t2i()(
        prompt=prompt,
        negative_prompt=negative_prompt or None,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        generator=make_generator(seed),
    ).images[0]


def image_to_image(image, prompt, negative_prompt, strength, steps, guidance, seed):
    if image is None:
        raise gr.Error("请先上传图片")
    if not prompt or not prompt.strip():
        raise gr.Error("请先填写目标风格描述")
    return get_i2i()(
        prompt=prompt,
        negative_prompt=negative_prompt or None,
        image=image.convert("RGB"),
        strength=float(strength),
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        generator=make_generator(seed),
    ).images[0]


def scribble_to_image(sketch, prompt, negative_prompt, steps, guidance, seed):
    if sketch is None:
        raise gr.Error("请先上传简笔画")
    if not prompt or not prompt.strip():
        raise gr.Error("请描述你想生成的画面")
    return get_scribble()(
        prompt=prompt,
        negative_prompt=negative_prompt or None,
        image=sketch.convert("RGB"),
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        generator=make_generator(seed),
    ).images[0]


def param_group(default_steps=25):
    with gr.Accordion("高级参数", open=False):
        neg = gr.Textbox(
            label="负面提示词",
            value="low quality, blurry, watermark, text, deformed",
        )
        steps = gr.Slider(10, 50, value=default_steps, step=1, label="采样步数")
        guidance = gr.Slider(1.0, 15.0, value=7.5, step=0.5, label="提示词引导强度")
        seed = gr.Number(value=-1, label="随机种子（-1 为随机）", precision=0)
    return neg, steps, guidance, seed


with gr.Blocks(title="AI 绘画工坊") as demo:
    gr.Markdown("# 🎨 AI 绘画工坊\n文生图 · 风格渲染 · 简笔画丰富")

    if DEVICE == "cpu":
        gr.Markdown(
            "> ⚠️ 当前运行在 CPU 上，生成一张图可能需要数分钟。切换到 GPU 后会快很多。"
        )

    with gr.Tab("文生图"):
        with gr.Row():
            with gr.Column():
                t2i_prompt = gr.Textbox(
                    label="画面描述", lines=3,
                    placeholder="例如：a watercolor painting of a fox in a snowy forest, soft light",
                )
                t2i_neg, t2i_steps, t2i_gui, t2i_seed = param_group()
                t2i_btn = gr.Button("生成", variant="primary")
            t2i_out = gr.Image(label="结果", type="pil")
        t2i_btn.click(
            text_to_image,
            [t2i_prompt, t2i_neg, t2i_steps, t2i_gui, t2i_seed],
            t2i_out,
        )

    with gr.Tab("风格渲染"):
        with gr.Row():
            with gr.Column():
                i2i_img = gr.Image(label="上传原图", type="pil")
                i2i_prompt = gr.Textbox(
                    label="目标风格描述", lines=2,
                    placeholder="例如：anime style, flat cel shading, vibrant colors",
                )
                i2i_strength = gr.Slider(
                    0.1, 0.9, value=0.6, step=0.05,
                    label="重绘强度（越高越偏离原图）",
                )
                i2i_neg, i2i_steps, i2i_gui, i2i_seed = param_group()
                i2i_btn = gr.Button("渲染", variant="primary")
            i2i_out = gr.Image(label="结果", type="pil")
        i2i_btn.click(
            image_to_image,
            [i2i_img, i2i_prompt, i2i_neg, i2i_strength, i2i_steps, i2i_gui, i2i_seed],
            i2i_out,
        )

    with gr.Tab("简笔画丰富"):
        with gr.Row():
            with gr.Column():
                sk_img = gr.Image(label="上传简笔画（建议白底黑线）", type="pil")
                sk_prompt = gr.Textbox(
                    label="想生成的画面", lines=3,
                    placeholder="例如：a cute cat sitting on a windowsill, warm sunlight, detailed fur",
                )
                sk_neg, sk_steps, sk_gui, sk_seed = param_group(default_steps=20)
                sk_btn = gr.Button("丰富画面", variant="primary")
            sk_out = gr.Image(label="结果", type="pil")
        sk_btn.click(
            scribble_to_image,
            [sk_img, sk_prompt, sk_neg, sk_steps, sk_gui, sk_seed],
            sk_out,
        )


if __name__ == "__main__":
    demo.launch()
