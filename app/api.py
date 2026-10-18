from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, StringConstraints

from .engine import ChatEngine
from .store import ChatStore

router = APIRouter()


class MessageIn(BaseModel):
    content: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)]


def get_store(request: Request) -> ChatStore:
    return request.app.state.store


def get_engine(request: Request) -> ChatEngine:
    return request.app.state.engine


Store = Annotated[ChatStore, Depends(get_store)]
Engine = Annotated[ChatEngine, Depends(get_engine)]


def require_chat(store: ChatStore, chat_id: int) -> dict:
    chat = store.get_chat(chat_id)
    if chat is None:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat


@router.get("/chats")
def list_chats(store: Store) -> list[dict]:
    return store.list_chats()


@router.post("/chats", status_code=201)
def create_chat(store: Store) -> dict:
    return store.create_chat()


@router.get("/chats/{chat_id}")
def get_chat(chat_id: int, store: Store) -> dict:
    chat = require_chat(store, chat_id)
    return {**chat, "messages": store.get_messages(chat_id)}


@router.delete("/chats/{chat_id}", status_code=204)
def delete_chat(chat_id: int, store: Store) -> Response:
    require_chat(store, chat_id)
    store.delete_chat(chat_id)
    return Response(status_code=204)


@router.post("/chats/{chat_id}/messages")
def send_message(chat_id: int, body: MessageIn, store: Store, engine: Engine) -> dict:
    require_chat(store, chat_id)
    # generate first so a failed reply leaves no half-saved turn behind
    history = store.get_messages(chat_id) + [{"role": "user", "content": body.content}]
    reply = engine.reply(history)
    store.add_message(chat_id, "user", body.content)
    return store.add_message(chat_id, "assistant", reply)
