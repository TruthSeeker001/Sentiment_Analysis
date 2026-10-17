# Sentiment Analysis Project

A Python project with two parts, shown in one Streamlit app:

1. **Text sentiment analysis**: a model trained on 25,000 IMDB reviews
   (TF-IDF + Logistic Regression, 89.2% test accuracy), compared with the
   pretrained VADER model.
2. **Face emotion detection**: OpenCV finds the face and DeepFace (pretrained)
   predicts the emotion.

## Run it

```
uv sync --link-mode=copy
uv run streamlit run app.py
```

## Live webcam demo

```
uv run python face/step2_deepface.py
```

## Files

- `step1_explore.py` to `step6_compare.py`: the text pipeline, step by step
- `app.py`: the Streamlit app
- `face/`: webcam scripts