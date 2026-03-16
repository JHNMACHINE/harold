"""
Harold — app.py  (streaming edition)
======================================
SSE streaming: i token appaiono e spariscono durante il denoising,
rendendo visibile il processo di diffusione nella chat UI.

Avvio:
  HAROLD_PASSWORD=tuapassword python app.py
"""

import os, time, math, secrets, json
from typing import Iterator

import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from jose import jwt, JWTError
from transformers import AutoTokenizer

from model import Harold, build_model

CKPT_PATH  = os.getenv("HAROLD_CKPT",     "checkpoints/harold_final.pt")
PASSWORD   = os.getenv("HAROLD_PASSWORD", "harold2024")
JWT_SECRET = os.getenv("HAROLD_SECRET",   "cambia-in-produzione")
DEVICE     = os.getenv("HAROLD_DEVICE",   "cuda" if torch.cuda.is_available() else "cpu")
JWT_ALGO   = "HS256"
JWT_EXP    = 60 * 60 * 8

print(f"Carico Harold da {CKPT_PATH} su {DEVICE}...")
_state     = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
_model_cfg = _state["model_cfg"]
_model     = build_model(_model_cfg).to(DEVICE)
_model.load_state_dict(_state["model_state"])
_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
_model.eval()
print(f"Harold pronto — {sum(p.numel() for p in _model.parameters())/1e6:.1f}M parametri")
del _state

MASK_ID = _model_cfg.mask_token_id
DIFF_T  = _model_cfg.diffusion_T


def decode_tokens(ids):
    out = []
    for tok_id in ids:
        if tok_id == MASK_ID:
            out.append("▒")
        else:
            w = _tokenizer.decode([tok_id], skip_special_tokens=True)
            out.append(w if w.strip() else "·")
    return out


@torch.no_grad()
def generate_streaming(prompt, gen_len=128, steps=32, temperature=1.0, mode="confidence"):
    try:
        prompt_ids  = _tokenizer.encode(prompt, add_special_tokens=True, truncation=True, max_length=gen_len) if prompt else []
        prompt_len  = len(prompt_ids)
        xt          = torch.full((1, gen_len), MASK_ID, dtype=torch.long, device=DEVICE)
        if prompt_len > 0:
            xt[0, :prompt_len] = torch.tensor(prompt_ids, dtype=torch.long, device=DEVICE)
        prompt_mask = torch.zeros(1, gen_len, dtype=torch.bool, device=DEVICE)
        prompt_mask[0, :prompt_len] = True
        t_schedule  = torch.linspace(DIFF_T, 1, steps).long()

        def uf(i):
            return float(1.0 - math.cos(math.pi * i / (2 * steps)) + 1.0 / steps)

        yield {"type":"step","step":0,"total_steps":steps,
               "tokens":decode_tokens(xt[0].tolist()),
               "n_masked":int((xt==MASK_ID).sum()),"prompt_len":prompt_len}

        for si, tv in enumerate(t_schedule):
            tt = torch.full((1,), tv.item(), dtype=torch.long, device=DEVICE)
            logits, _ = _model(xt, tt)
            
            rep_penalty = 3.0
            window      = 24
            logits_pen  = logits.clone()
            decoded     = xt[0][xt[0] != MASK_ID]
            decoded     = decoded[decoded != 0]
            if decoded.numel() > 0:
                recent = decoded[-window:]
                unique = recent.unique()
                logits_pen[0, :, unique] = logits_pen[0, :, unique] / rep_penalty
                # Hard block: se gli ultimi 2 token si ripetono 3+ volte, azzera il loro successore
                if decoded.numel() >= 4:
                    last2 = decoded[-2:].tolist()
                    count = sum(1 for i in range(len(decoded)-2)
                                if decoded[i].item()==last2[0] and decoded[i+1].item()==last2[1])
                    if count >= 2:
                        logits_pen[0, :, decoded[-1]] = -1e9  # blocca l'ultimo token
            logits = logits_pen

            if mode == "argmax":
                xp = logits.argmax(dim=-1)
                xt = torch.where((xt==MASK_ID)&~prompt_mask, xp, xt)

            elif mode == "sample":
                B,L,V = logits.shape
                sc = logits.view(B*L,V)/max(temperature,1e-5)
                pr = F.softmax(sc,dim=-1)
                sp,sidx = torch.sort(pr,descending=True,dim=-1)
                rm = sp.cumsum(dim=-1)-sp>0.9
                sp[rm]=0.0
                sp.div_(sp.sum(dim=-1,keepdim=True)+1e-9)
                samp = sidx.gather(-1,torch.multinomial(sp,1)).view(B,L)
                if si < steps-1:
                    nr = max(0,int((gen_len-prompt_len)*(1.0-(si+1)/steps)))
                    gp = torch.arange(prompt_len,gen_len,device=DEVICE)
                    samp[0,gp[torch.randperm(len(gp),device=DEVICE)[:nr]]]=MASK_ID
                xt = torch.where((xt==MASK_ID)&~prompt_mask, samp, xt)

            else:
                mp = (xt==MASK_ID)
                pr = F.softmax(logits/max(temperature,1e-5),dim=-1)
                cf,cd = pr.max(dim=-1)
                cf = cf.masked_fill(~mp,-1.0)
                nu = (mp.sum(-1).float()*uf(si)).ceil().long().clamp(min=1)
                xn = xt.clone()
                tk = cf[0].topk(int(nu[0].item())).indices
                xn[0,tk] = cd[0,tk]
                xt = torch.where(prompt_mask, xt, xn)

            nm = int((xt==MASK_ID).sum())
            yield {"type":"step","step":si+1,"total_steps":steps,
                   "tokens":decode_tokens(xt[0].tolist()),
                   "n_masked":nm,"prompt_len":prompt_len}
            if nm==0: break

        if (xt==MASK_ID).any():
            lg,_ = _model(xt, torch.full((1,),1,dtype=torch.long,device=DEVICE))
            xp   = lg.argmax(dim=-1)
            xt   = torch.where((xt==MASK_ID)&~prompt_mask, xp, xt)

        full = _tokenizer.decode(xt[0].tolist(), skip_special_tokens=True)
        resp = full[len(prompt):].strip() if full.lower().startswith(prompt.lower()) else full.strip()
        yield {"type":"done","response":resp,"full_text":full}

    except Exception as e:
        yield {"type":"error","detail":str(e)}


app = FastAPI(title="Harold")
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_methods=["*"],allow_headers=["*"])
security = HTTPBearer()

def create_token():
    return jwt.encode({"sub":"user","iat":int(time.time()),"exp":int(time.time())+JWT_EXP},JWT_SECRET,algorithm=JWT_ALGO)

def verify_token(c: HTTPAuthorizationCredentials = Depends(security)):
    try: jwt.decode(c.credentials,JWT_SECRET,algorithms=[JWT_ALGO])
    except JWTError: raise HTTPException(401,"Token non valido")

class LoginReq(BaseModel): password: str
class ChatReq(BaseModel):
    message:str; temperature:float=1.0; steps:int=32; gen_len:int=128; mode:str="confidence"

@app.post("/login")
def login(req:LoginReq):
    if not secrets.compare_digest(req.password,PASSWORD):
        raise HTTPException(401,"Password errata")
    return {"token":create_token()}

@app.post("/chat/stream")
def chat_stream(req:ChatReq, _=Depends(verify_token)):
    if not req.message.strip(): raise HTTPException(400,"Messaggio vuoto")
    steps   = max(8,min(req.steps,DIFF_T))
    gen_len = max(len(_tokenizer.encode(req.message))+32,min(req.gen_len,256))
    mode    = req.mode if req.mode in ("argmax","sample","confidence") else "confidence"
    def stream():
        for ev in generate_streaming(req.message,gen_len,steps,max(0.1,min(req.temperature,2.0)),mode):
            yield f"data: {json.dumps(ev)}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(stream(),media_type="text/event-stream",
                             headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

@app.get("/health")
def health():
    return {"status":"ok","device":DEVICE,"params":f"{sum(p.numel() for p in _model.parameters())/1e6:.1f}M"}

HTML = open(__file__.replace("app.py","ui.html")).read() if os.path.exists(__file__.replace("app.py","ui.html")) else "<h1>Harold</h1>"

@app.get("/",response_class=HTMLResponse)
def index(): return HTML

if __name__=="__main__":
    uvicorn.run("app:app",host="0.0.0.0",port=8000,reload=False,workers=1)