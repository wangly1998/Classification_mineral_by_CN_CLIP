import os
import torch
import json
from PIL import Image
import gradio as gr
from cn_clip.clip.model import CLIP, convert_weights
from cn_clip.training.main import convert_models_to_fp32
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode
from pathlib import Path
from cn_clip import clip

# 模型信息
_MODEL_INFO = {
    "ViT-B-16": {
        "struct": "ViT-B-16@RoBERTa-wwm-ext-base-chinese",
        "input_resolution": 224
    },
    "ViT-L-14": {
        "struct": "ViT-L-14@RoBERTa-wwm-ext-base-chinese",
        "input_resolution": 224
    },
    "ViT-L-14-336": {
        "struct": "ViT-L-14-336@RoBERTa-wwm-ext-base-chinese",
        "input_resolution": 336
    },
    "ViT-H-14": {
        "struct": "ViT-H-14@RoBERTa-wwm-ext-large-chinese",
        "input_resolution": 224
    },
    "RN50": {
        "struct": "RN50@RBT3-chinese",
        "input_resolution": 224
    },
}

# 选择设备
device = "cuda" if torch.cuda.is_available() else "cpu"
model = None


def _convert_to_rgb(image):
    return image.convert('RGB')


def _build_transform(resolution):
    normalize = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    return Compose([
        Resize((resolution, resolution), interpolation=InterpolationMode.BICUBIC),
        _convert_to_rgb,
        ToTensor(),
        normalize,
    ])


def load_model(model_name, device, precision="amp"):
    model_info = _MODEL_INFO[model_name]
    vision_model, text_model = model_info["struct"].split('@')
    resolution = model_info["input_resolution"]

    # 初始化模型
    vision_model_config_file = Path(
        __file__).parent.parent / f"cn_clip/clip/model_configs/{vision_model.replace('/', '-')}.json"
    print('Loading vision model config from', vision_model_config_file)
    assert os.path.exists(vision_model_config_file)

    text_model_config_file = Path(
        __file__).parent.parent / f"cn_clip/clip/model_configs/{text_model.replace('/', '-')}.json"
    print('Loading text model config from', text_model_config_file)
    assert os.path.exists(text_model_config_file)

    # 加载模型配置文件
    with open(vision_model_config_file, 'r') as fv, open(text_model_config_file, 'r') as ft:
        model_info = json.load(fv)
        if isinstance(model_info['vision_layers'], str):
            model_info['vision_layers'] = eval(model_info['vision_layers'])
        for k, v in json.load(ft).items():
            model_info[k] = v

    # 初始化模型
    model = CLIP(**model_info)
    convert_weights(model)

    # 设置模型精度
    if precision == "amp" or precision == "fp32":
        convert_models_to_fp32(model)
    model.to(device)
    if precision == "fp16":
        convert_weights(model)

    return model, resolution


def load_checkpoint(model, model_name, device):
    checkpoint_path = Path(f"resource/finetune_{model_name}.pt")
    print(f"Begin to load model checkpoint from {checkpoint_path}.")
    assert os.path.exists(checkpoint_path), f"The checkpoint file {checkpoint_path} does not exist!"

    checkpoint = torch.load(checkpoint_path, map_location=device)
    sd = checkpoint["state_dict"]

    # 如果模型权重包含module前缀，移除它
    if next(iter(sd.items()))[0].startswith('module'):
        sd = {k[len('module.'):]: v for k, v in sd.items() if "bert.pooler" not in k}

    model.load_state_dict(sd)
    print(f"Loaded checkpoint '{checkpoint_path}'")

    return model


def predict(image, descriptions, model_name, prompt_template):
    # 加载选择的模型
    global model
    model, resolution = load_model(model_name, device)
    model = load_checkpoint(model, model_name, device)

    # 处理输入图片
    transform = _build_transform(resolution)
    img = transform(image).unsqueeze(0).to(device)

    # 将描述分割成列表，如果是用逗号分隔的多个描述
    descriptions = descriptions.split(',')

    # 处理输入文本
    text = clip.tokenize(descriptions).to(device)

    with torch.no_grad():
        image_features = model.encode_image(img)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        logits_per_image, logits_per_text = model.get_similarity(img, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    # 确保输出和描述数量一致
    num_descriptions = len(descriptions)
    num_probs = probs.shape[1]

    if num_descriptions > num_probs:
        descriptions = descriptions[:num_probs]
    elif num_probs > num_descriptions:
        probs = probs[:, :num_descriptions]

    return {descriptions[i].strip(): float(probs[0, i]) for i in range(len(descriptions))}


examples = [
    ["example/1.jpg", "橄榄石,角闪石,石英,斜长石", "ViT-B-16", "这是一个包含{}矿物的薄片"],
    ["example/2.jpg", "鲕粒灰岩, 玄武岩, 花岗岩, 片麻岩", "ViT-L-14", "这是一个{} 薄片"],
    ["example/3.jpg", "单偏光镜下方柱石矽卡岩, 正交偏光镜下方柱石矽卡岩, 正交偏光镜下插入石膏试板方柱石矽卡岩", "ViT-L-14-336", "这是一个{}薄片 "]
]

# 使用Gradio构建界面
interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(label="Thin image 输入岩石薄片图片", type="pil"),
        gr.Textbox(label="Candidate Rock Labels 候选岩性标签", lines=2, placeholder="a list of labels in Chinese separated by commas/输入多个文本描述，用逗号分隔"),
        gr.Radio(choices=list(_MODEL_INFO.keys()), label="Model 模型规模", type="value"),
        gr.Textbox(label="Prompt Template Prompt模板 ({}指代候选标签)", lines=1, placeholder="一张包含{}的薄片"),
    ],
    outputs=gr.Label(num_top_classes=4),
    examples=examples,
    title="基于微调后的CN_CLIP进行岩石薄片预测(Rock thin images recognition based on fine-tuned CN_CLIP)\n",
    description="""
    
                本项目基于中文CLIP进行微调，原始项目Github: https://github.com/OFA-Sys/Chinese-CLIP         
                To play with this demo, add a picture and a list of labels in Chinese separated by commas.              
                Usage Instructions:
                1. Upload Thin Section Image: Click on the "Thin image" area to upload your image.
                2. Candidate Rock Labels: Enter multiple candidate labels in the "Candidate Rock Labels" text box, separated by commas.
                3. Select Model Size: Click on the "Model" area to choose a model for prediction.
                4. Input Prompt Template: Enter a template in the "Prompt Template Prompt" text box, using {} to represent the position for the best label.
                """
)

interface.launch()
