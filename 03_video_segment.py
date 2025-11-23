from sam3.model_builder import build_sam3_video_predictor

video_predictor = build_sam3_video_predictor(
    checkpoint_path="./models/facebook/sam3/sam3.pt")
video_path = "./sam3/assets/videos/bedroom.mp4" 
# Start a session
response = video_predictor.handle_request(
    request=dict(
        type="start_session",
        resource_path=video_path,
    )
)
session_id = response["session_id"]
response = video_predictor.handle_request(
    request=dict(
        type="add_prompt",
        session_id=response["session_id"],
        frame_index=0, 
        text="<YOUR_TEXT_PROMPT>",
    )
)
output = response["outputs"]
print(output)
# Close session
video_predictor.handle_request(
    request=dict(
        type="close_session",
        session_id=session_id,
    )
)
video_predictor.shutdown()