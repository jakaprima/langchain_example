from pydantic import BaseModel


class Todo(BaseModel):
    id: int = None
    text: str


todos = []

from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def get_todos():
    return {
        "app": "v1"
    }


@app.post("/todos")
async def create_todo(todo: Todo):
    todo.id = len(todos) + 1  # Generate a simple ID
    todos.append(todo)
    return todo


@app.get("/todos")
async def get_todos():
    return todos


@app.get("/todos/{todo_id}")
async def get_todo(todo_id: int):
    for todo in todos:
        if todo.id == todo_id:
            return todo
    return None  # Raise an exception for better error handling in practice


@app.put("/todos/{todo_id}")
async def update_todo(todo_id: int, updated_todo: Todo):
    for i, todo in enumerate(todos):
        if todo.id == todo_id:
            todos[i] = updated_todo
            return updated_todo
    return None  # Raise an exception for better error handling in practice


@app.delete("/todos/{todo_id}")
async def delete_todo(todo_id: int):
    for i, todo in enumerate(todos):
        if todo.id == todo_id:
            del todos[i]
            return {"message": "Todo deleted successfully"}
    return None  # Raise an exception for better error handling in practice
