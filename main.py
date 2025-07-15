import io
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, HTMLResponse
from starlette.status import HTTP_303_SEE_OTHER
from services import eda, train, predict, preprocess
from fastapi.templating import Jinja2Templates
import pandas as pd

from models.enums import PlotType


templates = Jinja2Templates(directory="templates")

app = FastAPI()
app.mount("/static", StaticFiles(directory = "static"),
          name = "static"
          )

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    df = await eda.read_csv(file)
    summary = eda.generate_summary(df)
    return {"summary": summary}


@app.post("/train/")
async def train_model(file: UploadFile = File(...)):
    df = await eda.read_csv(file)
    model_path = train.train_model(df)
    return {"message": "Model trained", "model_path": model_path}


@app.post("/predict/")
async def get_prediction(file: UploadFile = File(...)):
    df = await eda.read_csv(file)
    results = predict.make_prediction(df)
    return {"predictions": results}


@app.post("/cluster_process/")
async def run_full_cluster_pipeline(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code = 400,
                            detail = "Only CSV files are supported.")
    
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

    try:
        result = preprocess.preprocess_model_clust_adj_4(df)
        return {"message": "Processing complete"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/linear_reg_process/")
async def run_full_linear_reg_pipeline(request: Request,
                                       plot_type: Optional[PlotType] = None,
                                       file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code = 400,
                            detail = 'Only CSV files are supported.')

    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
    coef, plot_url = preprocess.preprocess_linear_reg(df, plot_type)

    request.app.state.latest_result = {
        "coefficients" : coef['coefficients'],
        'plot_url' : f'/{plot_url}'
    }

    return RedirectResponse(url = '/result', status_code=HTTP_303_SEE_OTHER)

    # return {
    #     "coefficients" : coef,
    #     "plot_url" : plot_url
    #     }

    # return templates.TemplateResponse("plot_and_coef.html", {
    #     "request": request,
    #     "coefficients": coef['coefficients'],
    #     "plot_url": f"/{plot_url}"
    # })




@app.get("/result", response_class=HTMLResponse)
def show_result(request: Request):
    result = request.app.state.latest_result

    if not result:
        raise HTTPException(status_code = 400, detail = 'No results available')

    # Load coefficients if needed
    return templates.TemplateResponse("plot_and_coef.html", {
        "request": request,
        'coefficients' : result['coefficients'],
        'plot_url' : result['plot_url']
        # Possibly pass coefficients here if stored/shared
    })
