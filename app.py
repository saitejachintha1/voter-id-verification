from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from pymongo import MongoClient
from bson import ObjectId
import os
import cv2
import pytesseract
import re
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from ultralytics import YOLO

# Initialize Flask app
app = Flask(__name__)

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
model = YOLO('yolov8_trained_model.pt')

client = MongoClient(
    "mongodb+srv://pandukrishna04:Raina%40143@cluster0.4fkpx.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client['demo']
collection = db['user']

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


keywords = {
    "NAME": ["name", "full name"],
    "DOB": ["dob", "date of birth", "birth", "d.o.b", "date"],
    "GENDER": ["gender", "male", "female"]
}


def prioritize_fuzzy_match(text, field_keywords):
    exact_matches = [kw for kw in field_keywords if kw.lower() == text.lower()]
    if exact_matches:
        return sorted(exact_matches, key=len, reverse=True)

    sorted_matches = process.extract(text, field_keywords, scorer=fuzz.token_sort_ratio)
    sorted_token_matches = [match[0] for match in sorted_matches if match[1] > 75]
    if sorted_token_matches:
        return sorted_token_matches

    partial_matches = process.extract(text, field_keywords, scorer=fuzz.partial_ratio)
    sorted_partial_matches = [match[0] for match in partial_matches if match[1] > 50]
    if sorted_partial_matches:
        return sorted_partial_matches

    return []


def classify_text(text):
    for field, keys in keywords.items():
        matched_keyword = prioritize_fuzzy_match(text, keys)
        if matched_keyword:
            return field
    return "UNKNOWN"


def format_date(date_str):
    date_str = re.sub(r'\D', '', date_str)
    if len(date_str) == 8:
        return f"{date_str[:2]}/{date_str[2:4]}/{date_str[4:]}"
    elif len(date_str) == 6:
        return f"{date_str[:2]}/{date_str[2:4]}/{date_str[4:]}"
    return None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            image = cv2.imread(filepath)
            if image is None:
                return jsonify({'error': 'Error loading image'})

            resized_image = cv2.resize(image, (640, 640))
            results = model.predict(source=resized_image, conf=0.5, verbose=False)

            extracted_info = {"NAME": None, "DOB": None, "GENDER": None}
            for result in results:
                for box, label in zip(result.boxes.xyxy, result.boxes.cls):
                    x1, y1, x2, y2 = map(int, box)
                    cropped_image = resized_image[y1:y2, x1:x2]
                    gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
                    _, thresh_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY)

                    extracted_text = pytesseract.image_to_string(thresh_image, config="--psm 6").strip()

                    if extracted_text:
                        field = classify_text(result.names[int(label)])
                        if field == "DOB":
                            extracted_text = format_date(extracted_text)

                        if extracted_info.get(field) is None:
                            extracted_info[field] = extracted_text

            user_data = {
                "name": extracted_info['NAME'],
                "dob": extracted_info['DOB'],
                "gender": extracted_info['GENDER'],
                "match": None
            }
            collection.insert_one(user_data)

            return jsonify({
                'NAME': extracted_info['NAME'],
                'DOB': extracted_info['DOB'],
                'GENDER': extracted_info['GENDER']
            })

        except Exception as e:
            return jsonify({'error': f"There was an error processing the image: {str(e)}"})

    return jsonify({'error': 'Invalid file format'})


@app.route('/get_users', methods=['GET'])
def get_users():
    try:
        users = list(collection.find({}, {"_id": 1, "name": 1, "dob": 1, "gender": 1, "match": 1}))
        for user in users:
            user['_id'] = str(user['_id'])
        return jsonify(users)
    except Exception as e:
        return jsonify({"error": f"An error occurred while fetching users: {str(e)}"}), 500


@app.route('/add_users', methods=['POST'])
def add_users():
    try:
        # Get data from the request body
        user_data = request.json

        # Validate the payload is a list of dictionaries
        if not isinstance(user_data, list):
            return jsonify({"error": "Invalid data format. Expected a JSON array."}), 400

        # Insert data into the MongoDB collection
        result = collection.insert_many(user_data)

        # Respond with success message
        return jsonify({"message": "Users added successfully", "ids": [str(id) for id in result.inserted_ids]}), 201
    except Exception as e:
        return jsonify({"error": f"An error occurred while adding users: {str(e)}"}), 500


@app.route('/save_match', methods=['POST'])
def save_match():
    data = request.json
    user_id = data.get('user_id', None)
    match = data.get('match', None)

    if not user_id or match is None:
        return jsonify({"error": "Missing user_id or match status"}), 400

    try:
        user_object_id = ObjectId(user_id)
        user = collection.find_one({'_id': user_object_id})
        if not user:
            return jsonify({"error": f"User with ID {user_id} not found"}), 404

        result = collection.update_one(
            {'_id': user_object_id},
            {'$set': {'match': match}}
        )

        if result.matched_count == 0:
            return jsonify({"error": "User not updated"}), 500

        return jsonify({"status": "success", "user_id": user_id, "match": match}), 200

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True, port=3000)