import cv2
import numpy as np
from keras.models import model_from_json
from flask import Flask, render_template, request, Response

app = Flask(__name__)

json_file = open("emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("emotiondetector.h5")

haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)
@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/UploadImage',methods=['POST'])
def upload_image():
    return render_template('upload.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

def predict_emotion(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(image, 1.3, 5)
    try:
        for (p, q, r, s) in faces:
            face_image = gray[q:q + s, p:p + r]
            face_image = cv2.resize(face_image, (48, 48))
            img = extract_features(face_image)
            pred = model.predict(img)
            prediction_label = labels[pred.argmax()]
            cv2.putText(image, prediction_label, (p - 10, q - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))
        return image
    except cv2.error:
        return None
    
def upload_image():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
            result_image = predict_emotion(image.copy())
            if result_image is not None:
                _, img_encoded = cv2.imencode('.jpg', result_image)
                img_data = img_encoded.tobytes()
                return render_template('upload.html', img_data=img_data)
    return render_template('upload.html')

# def upload_image():
#     global im, result, percentage , i , imageName , solution
#     target = os.path.join(APP_ROOT, 'static\\')
#     print(f'Target : {target}')

#     if not os.path.isdir(target):
#         os.mkdir(target)
#     for imgg in os.listdir(target):
#         try:
#             imgPath = target + imgg
#             os.remove(imgPath)
#             print(f'Removed : {imgPath}')
#         except Exception as e:
#             print(e)
        
#     for file in request.files.getlist("file"):
#         print(f'File : {file}')
#         i += 1
#         imageName = str(i) + '.JPG'
#         filename = file.filename
#         destination = "/".join([target, imageName])
#         print(f'Destination : {destination}')
#         file.save(destination)
#         print('analysing Image')
#         try:
#             image = os.listdir('static')
#             im = destination
#             print(f'Analysing Image : {im}')
#         except Exception as e:
#             print(e)
#         result = "Failed to Analyse"
#         percentage = "0 %"
#         try:
#             detect()
#             solution = solutions(result)
#         except Exception as e:
#             print(f'Error While Loading : {e}')  
#     return render_template('complete.html', name=result, accuracy=percentage , img = imageName , soln = solution)

labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}



def generate_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            result_frame = predict_emotion(frame.copy())
            if result_frame is not None:
                ret, buffer = cv2.imencode('.jpg', result_frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



if __name__ == '__main__':
    app.run(debug=True)
