from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from utils.mediapipe_utils import MediaPipeUtils
from pathlib import Path
import cv2

app = FastAPI()
app.mount(
    "/static",
    StaticFiles(directory=Path(__file__).parent.parent.absolute() / "static"),
    name="static",
)
templates = Jinja2Templates(directory="templates")

hand_type = "Right"

Base = MediaPipeUtils()
Hands = Base.Hands_model_configuration(False, 1, 1)

def Real_time_sign_detection():
    capture = cv2.VideoCapture(0)
    while capture.isOpened():
        key = cv2.waitKey(1)
        success, image = capture.read()
        if not success:
            continue
        image = cv2.flip(image, 1)
        frame, results = Base.Hands_detection(image, Hands)
        copie_img = frame.copy()
        if results.multi_hand_landmarks:
            positions = []
            positions, key_points = Base.Detect_hand_type(hand_type, results, positions, copie_img)
            if len(positions) != 0:
                Base.Draw_Bound_Boxes(positions, frame)
        _, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/video")
async def video():
    return StreamingResponse(Real_time_sign_detection(), media_type="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)