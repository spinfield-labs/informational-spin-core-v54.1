# icoer_async_api.py (v5.4.1)
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import matplotlib.pyplot as plt
import io
import base64
from icoer_extractor_v541 import extract_all
from icoer_visualizer import plot_icoer_factors

app = FastAPI()

class CoherenceRequest(BaseModel):
    text: str
    memory: list[str] = []
    model_type: str = "light"
    lang: str = "en"
    style_profile: str = "default"
    is_factually_true: bool | None = None
    visualize: bool = False

@app.get("/healthcheck")
async def healthcheck():
    return {"status": "ok", "version": "5.4.1"}

@app.post("/extract")
async def extract_coherence(data: CoherenceRequest):
    try:
        results = extract_all(
            text=data.text,
            memory_context=data.memory,
            model_type=data.model_type,
            lang=data.lang,
            style_profile=data.style_profile
        )

        response = {"scores": results}

        if data.is_factually_true is not None:
            response["factual_truth"] = data.is_factually_true
        else:
            response["warning"] = "Factual truth not provided; interpretation symbolic only."

        if data.visualize:
            fig = plt.figure()
            plot_icoer_factors(results)
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            buf.close()
            response["image"] = img_base64

        return response

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run("icoer_async_api:app", host="0.0.0.0", port=8000, reload=True)
