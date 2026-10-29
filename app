from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)
ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ROOT, "model", "braking_model.joblib")
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
model_columns = bundle["columns"]

ALLOWED_ROADS = {"Dry", "Wet", "Slippery"}
ALLOWED_TYRES = {"Good", "Worn"}
ALLOWED_LOADS = {"Light", "Medium", "Heavy"}
ALLOWED_SLOPES = {"Uphill", "Flat", "Downhill"}


def encode_input(speed, road, tyre, load, slope):
    row = pd.DataFrame([{
        "speed_kmh": float(speed),
        "road_condition": road,
        "tyre_condition": tyre,
        "vehicle_load": load,
        "road_slope": slope,
    }])
    encoded = pd.get_dummies(
        row,
        columns=["road_condition", "tyre_condition", "vehicle_load", "road_slope"],
    )
    return encoded.reindex(columns=model_columns, fill_value=0)


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    values = {"speed": "60", "road": "Dry", "tyre": "Good", "load": "Medium", "slope": "Flat"}

    if request.method == "POST":
        values = {
            "speed": request.form.get("speed", "60").strip(),
            "road": request.form.get("road", "Dry"),
            "tyre": request.form.get("tyre", "Good"),
            "load": request.form.get("load", "Medium"),
            "slope": request.form.get("slope", "Flat"),
        }
        try:
            speed = float(values["speed"])
            if not 20 <= speed <= 120:
                raise ValueError("Enter a speed between 20 and 120 km/h.")
            if values["road"] not in ALLOWED_ROADS:
                raise ValueError("Please select a listed road condition.")
            if values["tyre"] not in ALLOWED_TYRES:
                raise ValueError("Please select a listed tyre condition.")
            if values["load"] not in ALLOWED_LOADS:
                raise ValueError("Please select a listed vehicle load.")
            if values["slope"] not in ALLOWED_SLOPES:
                raise ValueError("Please select a listed road slope.")
            features = encode_input(speed, values["road"], values["tyre"], values["load"], values["slope"])
            result = max(0.0, round(float(model.predict(features)[0]), 2))
        except (TypeError, ValueError) as exc:
            error = str(exc) or "Please check your input values."
        except Exception:
            error = "The prediction could not be completed. Please check the inputs and try again."

    return render_template("index.html", result=result, error=error, values=values)


if __name__ == "__main__":
    # Local development only. Production hosting uses Gunicorn through the Procfile.
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), debug=False)


