import gradio as gr
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO
from PIL import Image
import os
from pathlib import Path

# ==============================================================================
# 1. SETUP DE CAMINHOS
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent

# Sobe um nível para chegar na raiz do projeto: .../idp-doc-vision-br
PROJECT_ROOT = BASE_DIR.parent

# Agora monta os caminhos absolutos para a pasta weights
PATH_CLS = str(PROJECT_ROOT / 'weights' / 'classificador_doc_v1.pt')
PATH_DET = str(PROJECT_ROOT / 'weights' / 'detector_campos_v1.pt')
PATH_QUAL = str(PROJECT_ROOT / 'weights' / 'spotlight_v1.pth')

print(f"📂 Raiz do Projeto detectada: {PROJECT_ROOT}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# 2. CARREGAR MODELOS
# ==============================================================================
print("⏳ Carregando modelos...")

# --- A. Carregar YOLO (Simples) ---
try:
    model_cls = YOLO(PATH_CLS)
    print("✅ Classificador YOLO carregado.")
except:
    model_cls = None
    print(f"⚠️ Classificador não encontrado em {PATH_CLS}")

try:
    model_det = YOLO(PATH_DET)
    print("✅ Detector de Campos YOLO carregado.")
except:
    model_det = None
    print(f"⚠️ Detector não encontrado em {PATH_DET}")

# --- B. Carregar MobileNetV3 (PyTorch Puro) ---
try:
    # Recriar a arquitetura exata do treino
    model_qual = models.mobilenet_v3_small(weights=None)
    in_features = model_qual.classifier[3].in_features
    model_qual.classifier[3] = nn.Linear(in_features, 2) # 2 classes: Clean / Spotlight
    
    # Carregar os pesos
    state_dict = torch.load(PATH_QUAL, map_location=device)
    model_qual.load_state_dict(state_dict)
    model_qual.to(device)
    model_qual.eval()
    
    # Transformação necessária (mesma do treino)
    qual_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    print("✅ Fiscal de Qualidade (MobileNet) carregado.")
except Exception as e:
    model_qual = None
    print(f"⚠️ Erro ao carregar MobileNet: {e}")


# ==============================================================================
# 3. FUNÇÕES AUXILIARES (O Pulo do Gato 🐱)
# ==============================================================================
def smart_crop_opencv(img):
    """
    Tenta achar o documento na imagem usando contornos (substituto do YOLO-Crop).
    """
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blur, 75, 200)
        
        # Achar contornos
        cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: return img # Falhou, devolve original
        
        # Pegar o maior contorno por área
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:1]
        c = cnts[0]
        
        # Calcular Bounding Box
        x, y, w, h = cv2.boundingRect(c)
        
        # Só recorta se for grande o suficiente (>10% da imagem)
        if w * h > (img.shape[0] * img.shape[1]) * 0.1:
            return img[y:y+h, x:x+w]
        return img
    except:
        return img

def check_quality(img_pil):
    """Passa a imagem pelo MobileNet para ver se tem Spotlight"""
    if model_qual is None: return "N/A", 0.0
    
    img_tensor = qual_transform(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model_qual(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
        
    # Assumindo classes: 0=Clean, 1=Spotlight (ou vice-versa, ajuste se precisar)
    # Geralmente ImageFolder usa ordem alfabética. Se suas pastas eram 'clean' e 'spotlight':
    # 0 = clean, 1 = spotlight
    classe = "⚠️ COM REFLEXO" if pred.item() == 1 else "✅ LIMPA"
    return classe, conf.item()


# ==============================================================================
# 4. PIPELINE PRINCIPAL
# ==============================================================================
def processar_documento(imagem_entrada, conf_threshold):
    if imagem_entrada is None: return None, "Sem imagem.", None

    info_log = "### 🕵️‍♂️ Relatório de Processamento\n"
    
    # 1. Classificação (YOLO)
    if model_cls:
        res = model_cls(imagem_entrada, verbose=False)[0]
        classe_doc = res.names[res.probs.top1].upper()
        conf_doc = res.probs.top1conf.item()
        info_log += f"- **Documento:** {classe_doc} ({conf_doc:.1%})\n"
    
    # 2. Recorte Inteligente (OpenCV)
    img_crop = smart_crop_opencv(imagem_entrada)
    
    # 3. Fiscal de Qualidade (MobileNet)
    # Converter array numpy para PIL para o PyTorch
    img_pil = Image.fromarray(img_crop)
    status_qualidade, conf_q = check_quality(img_pil)
    info_log += f"- **Qualidade:** {status_qualidade} ({conf_q:.1%})\n"
    
    # 4. Extração de Campos (YOLO)
    img_final = img_crop.copy()
    num_campos = 0
    if model_det:
        res_det = model_det(img_final, conf=conf_threshold, verbose=False)[0]
        img_final = res_det.plot() # Desenha as caixas
        img_final = cv2.cvtColor(img_final, cv2.COLOR_BGR2RGB) # Corrige cores
        num_campos = len(res_det.boxes)
        info_log += f"- **Campos Extraídos:** {num_campos} encontrados.\n"

    return img_crop, info_log, img_final

# ==============================================================================
# 5. INTERFACE
# ==============================================================================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚀 IDP-Vision Pro: Pipeline Completo")
    
    with gr.Row():
        with gr.Column():
            img_in = gr.Image(label="Documento Original", type="numpy")
            slider = gr.Slider(0.1, 1.0, 0.5, label="Sensibilidade Detecção")
            btn = gr.Button("Analisar", variant="primary")
        
        with gr.Column():
            log_out = gr.Markdown()
            with gr.Row():
                crop_out = gr.Image(label="Recorte Automático")
                final_out = gr.Image(label="Extração de Dados")

    btn.click(processar_documento, [img_in, slider], [crop_out, log_out, final_out])

if __name__ == "__main__":
    demo.launch(share=True)