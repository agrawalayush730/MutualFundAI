from fastapi import FastAPI
from pydantic import BaseModel, Field 
from pymongo import MongoClient
from dotenv import load_dotenv
import os
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from typing import Optional



app=FastAPI()


templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def serve_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/feedback", response_class=HTMLResponse)
def serve_feedback_form(request: Request):
    return templates.TemplateResponse("feedback.html", {"request": request})

@app.get("/register", response_class=HTMLResponse)
async def serve_register(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

@app.get("/login", response_class=HTMLResponse)
async def serve_login(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/analyze-text", response_class=HTMLResponse)
async def serve_login(request: Request):
    return templates.TemplateResponse("analyze.html", {"request": request})

