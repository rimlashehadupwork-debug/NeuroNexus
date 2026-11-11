# NeuroNexus Integration Guide (Day 4)

## 1. Copy files into your project
- modules/parkinson_model.py   -> replace old modules/parkinson_model.py
- modules/alzheimers_model.py  -> new file
- modules/model_registry.py    -> new file
- models/parkinson_voice_research.joblib        -> replace old
- models/parkinson_voice_research_metadata.json -> replace old

## 2. Add admin model-registry tab in app.py
```python
from modules.model_registry import get_registry
for disease, info in get_registry().items():
    st.subheader(disease.capitalize())
    if info["available"]:
        st.json({k: v for k, v in info.items() if k != "limitations"})
        for lim in info.get("limitations", []): st.caption("⚠️ " + lim)
    else:
        st.info(info["message"])
```

## 3. Per-assessment model status card (show AFTER each assessment)
Order: assessment complete -> recording quality -> features extracted ->
model available? -> output ONLY if available -> limitations -> "consult a neurologist".
For Alzheimer's the registry returns available=False, so the UI shows the
honest "model unavailable" state automatically.

## 4. Retrain anytime (research lab page)
```python
from modules.parkinson_model import train_parkinson_model
meta = train_parkinson_model("train_data.txt", dataset="multiple_sound")
st.json(meta["metrics"])
```

## 5. Run
    pip install -r requirements.txt
    pytest -q
    streamlit run app.py
