from flask import Flask, render_template, request, Response
from ultralytics import YOLO
import cv2
import os

app = Flask(__name__)

# -----------------------------------
# Load YOLO Model
# -----------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(
    BASE_DIR,
    "runs/detect/train/weights/best.pt"
)

model = YOLO(model_path)

# -----------------------------------
# Folders
# -----------------------------------
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static/uploads")
RESULT_FOLDER = os.path.join(BASE_DIR, "static/results")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER

# -----------------------------------
# Allowed Extensions
# -----------------------------------
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

def allowed_file(filename):
    return "." in filename and \
           filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# -----------------------------------
# HOME PAGE
# -----------------------------------
@app.route("/")
def home():
    return render_template("home.html")


# -----------------------------------
# ABOUT PAGE
# -----------------------------------
@app.route("/about")
def about():
    return render_template("about.html")


# -----------------------------------
# CONTACT PAGE
# -----------------------------------
@app.route("/contact")
def contact():
    return render_template("contact.html")


# -----------------------------------
# IMAGE DETECTION
# -----------------------------------
@app.route("/detect", methods=["GET", "POST"])
def detect():

    if request.method == "POST":

        file = request.files.get("image")

        if not file or file.filename == "":
            return render_template(
                "detect.html",
                error="Please upload an image"
            )

        if not allowed_file(file.filename):
            return render_template(
                "detect.html",
                error="Only JPG/PNG allowed"
            )

        filepath = os.path.join(
            app.config["UPLOAD_FOLDER"],
            file.filename
        )

        file.save(filepath)

        # -----------------------------------
        # Fog Enhancement
        # -----------------------------------
        img = cv2.imread(filepath)

        img = cv2.detailEnhance(
            img,
            sigma_s=10,
            sigma_r=0.15
        )

        cv2.imwrite(filepath, img)

        # -----------------------------------
        # YOLO Detection
        # -----------------------------------
        results = model(filepath, conf=0.4)

        img = results[0].orig_img.copy()

        boxes = results[0].boxes

        # -----------------------------------
        # PERSON COUNT
        # -----------------------------------
        person_count = 0

        for box in boxes:

            # CLASS 0 = PERSON
            if int(box.cls[0]) == 0:

                person_count += 1

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                conf = float(box.conf[0])

                label = f"Person {conf:.2f}"

                # Rectangle
                cv2.rectangle(
                    img,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # Label
                cv2.putText(
                    img,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        # -----------------------------------
        # SHOW TOTAL PERSON COUNT
        # -----------------------------------
        cv2.putText(
            img,
            f"Total Persons: {person_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            3
        )

        # -----------------------------------
        # Save Result
        # -----------------------------------
        result_path = os.path.join(
            app.config["RESULT_FOLDER"],
            file.filename
        )

        cv2.imwrite(result_path, img)

        return render_template(
            "detect.html",
            uploaded_image="static/uploads/" + file.filename,
            result_image="static/results/" + file.filename,
            total_persons=person_count
        )

    return render_template("detect.html")


# -----------------------------------
# LIVE CAMERA DETECTION
# -----------------------------------
def generate_frames():

    cap = cv2.VideoCapture(0)

    while True:

        success, frame = cap.read()

        if not success:
            break

        # -----------------------------------
        # Fog Enhancement
        # -----------------------------------
        frame = cv2.detailEnhance(
            frame,
            sigma_s=10,
            sigma_r=0.15
        )

        # -----------------------------------
        # YOLO Detection
        # -----------------------------------
        results = model(frame, conf=0.4)

        person_count = 0

        for box in results[0].boxes:

            if int(box.cls[0]) == 0:

                person_count += 1

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                conf = float(box.conf[0])

                label = f"Person {conf:.2f}"

                # Rectangle
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # Label
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        # -----------------------------------
        # Total Person Count
        # -----------------------------------
        cv2.putText(
            frame,
            f"Total Persons: {person_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            3
        )

        # -----------------------------------
        # Convert to Stream
        # -----------------------------------
        ret, buffer = cv2.imencode('.jpg', frame)

        frame = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            frame +
            b'\r\n'
        )

    cap.release()


# -----------------------------------
# LIVE PAGE
# -----------------------------------
@app.route("/live")
def live():
    return render_template("live.html")


# -----------------------------------
# VIDEO FEED
# -----------------------------------
@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


# -----------------------------------
# RUN APP
# -----------------------------------
if __name__ == "__main__":
    app.run(debug=True)