from ultralytics.models.sam import SAM3SemanticPredictor

overrides = dict(
    conf=0.25,
    task="segment",
    mode="predict",
    model="./models/facebook/sam3/sam3.pt",
    half=True,
    save=True,
)
predictor = SAM3SemanticPredictor(overrides=overrides)
predictor.set_image("image01.jpg")
results = predictor(
    text=["the soccer players in blue jerseys in the image"])
results[0].show()