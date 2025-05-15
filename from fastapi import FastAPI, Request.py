from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from diffusers import StableDiffusionXLPipeline
import torch
import uuid

app = FastAPI()

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
).to("cuda")

@app.post("/gerar")
async def gerar(req: Request):
    dados = await req.json()
    prompt = dados.get("prompt", "um gato astronauta no espaço")
    imagem = pipe(prompt).images[0]
    nome = f"{uuid.uuid4().hex}.png"
    imagem.save(nome)
    return {"imagem_url": f"/imagem/{nome}"}

@app.get("/imagem/{nome}")
async def imagem(nome: str):
    return FileResponse(nome)
