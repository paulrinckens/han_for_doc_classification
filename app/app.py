import base64
from io import BytesIO

import matplotlib.colors
import numpy as np
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from han.HAN import HAN

app = FastAPI()
app.mount("/app/static", StaticFiles(directory="app/static"), name="static")

templates = Jinja2Templates(directory="app/templates")

han = HAN()
han.load_model(model_dir="./models",
               model_filename="han-10kGNAD.h5",
               tokenizer_filename="tokenizer.pickle")

class_names = ['Etat', 'Inland', 'International', 'Kultur', 'Panorama', 'Sport',
               'Web', 'Wirtschaft', 'Wissenschaft']


@app.get("/predict")
async def predict(text: str):
    probs = han.predict([text])[0]
    return dict({class_names[i]: float(probs[i]) for i in range(len(probs))})


@app.get("/visualize")
async def visualize(request: Request, text: str):
    probs, attentions = han.predict_and_visualize_attention(text)
    prediction = class_names[np.argmax(probs)]

    fig = Figure(figsize=(8, 2), tight_layout=True)
    ax = fig.add_subplot(1, 1, 1)

    y_pos = np.arange(len(class_names))
    confidences = [probs[i] for i in range(len(class_names))]

    ax.barh(y_pos, confidences, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(class_names)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Confidence')

    output = BytesIO()
    FigureCanvas(fig).print_png(output)
    out = base64.b64encode(output.getvalue()).decode()

    # weigh word attentions by sqrt of sentence attention
    for sent in attentions:
        for word in sent[1]:
            word[0] *= np.sqrt(sent[0])

    return templates.TemplateResponse("han_template.html",
                                      {"request": request,
                                       "prediction": prediction,
                                       "attentions": attentions,
                                       "red_background_hex": red_background_hex,
                                       "blue_background_hex": blue_background_hex,
                                       "plot_url": out})


def scale_color_h_hex(c_h, scale):
    return matplotlib.colors.to_hex(
        matplotlib.colors.hsv_to_rgb((c_h, scale, 1)))


def red_background_hex(scale):
    return scale_color_h_hex(0, scale)


def blue_background_hex(scale):
    return scale_color_h_hex(0.625, scale)
