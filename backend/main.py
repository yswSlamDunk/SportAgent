import os
import sys

from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

app = FastAPI()
templates = Jinja2Templates(directory=os.path.join(project_root, "frontend/templates"))
app.mount("/static", StaticFiles(directory=os.path.join(project_root, "frontend/static")), name="static")

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})