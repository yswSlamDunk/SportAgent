import os
import sys
from dotenv import load_dotenv

from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app.api import users
from app.api import videos
from app.api import sports
from app.api import pose
from app.api import score

load_dotenv()

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

app.include_router(users.router)
app.include_router(videos.router)
app.include_router(sports.router)
app.include_router(pose.router)
app.include_router(score.router)

templates = Jinja2Templates(directory=os.path.join(project_root, "frontend/templates"))
app.mount("/static", StaticFiles(directory=os.path.join(project_root, "frontend/static")), name="static")
app.mount("/media", StaticFiles(directory=os.path.join(project_root, "media")), name="media")

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.get("/login")
async def read_login(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})    

@app.get("/signup")
async def read_signup(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request})