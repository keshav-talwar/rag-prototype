from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Simple FastAPI App")



@app.get("/")
def root():
    return {"message": "API running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)


class Item(BaseModel):
    name: str
    price: float
    is_offer: Optional[bool] = None


items_db = {}


@app.post("/items", status_code=201)
def create_item(item: Item):
    item_id = len(items_db) + 1
    items_db[item_id] = item.dict()
    return {"item_id": item_id, "item": items_db[item_id]}


@app.get("/items/{item_id}")
def get_item(item_id: int):
    if item_id not in items_db:
        raise HTTPException(status_code=404, detail="Item not found")
    return {"item_id": item_id, "item": items_db[item_id]}


@app.get("/items")
def list_items():
    return items_db
