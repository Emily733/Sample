from fastapi import FastApi

app = FastApi()

@app.get("/")
async def health_check():
    return "Heath check succefull"