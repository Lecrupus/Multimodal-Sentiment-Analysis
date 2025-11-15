# File: app.py

import os
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename

# Import your analysis functions
import analysis_logic

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS_IMG = {'png', 'jpg', 'jpeg', 'gif'}
ALLOWED_EXTENSIONS_AV = {'mp4', 'wav', 'mp3', 'mov', 'avi'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Make sure the 'uploads' directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Helper Functions ---
def allowed_file(filename, allowed_set):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_set

# --- Main Routes ---
@app.route('/', methods=['GET'])
def index():
    # This just shows the main page
    return render_template('index.html')

@app.route('/analyze_text', methods=['POST'])
def handle_text():
    text = request.form['text_input']
    if not text:
        return redirect(url_for('index'))
    
    # Call your logic
    result = analysis_logic.analyze_text(text)
    
    # Pass the result back to the same page
    return render_template('index.html', text_result=result, original_text=text)

@app.route('/analyze_image', methods=['POST'])
def handle_image():
    if 'file_input' not in request.files:
        return redirect(url_for('index'))
    
    file = request.files['file_input']
    if file.filename == '' or not allowed_file(file.filename, ALLOWED_EXTENSIONS_IMG):
        return redirect(url_for('index'))

    # Save the file to the 'uploads' folder
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Call your logic
    result = analysis_logic.analyze_image(file_path)
    
    return render_template('index.html', image_result=result, image_file=filename)

@app.route('/analyze_audio', methods=['POST'])
def handle_audio():
    if 'file_input' not in request.files:
        return redirect(url_for('index'))
    
    file = request.files['file_input']
    if file.filename == '' or not allowed_file(file.filename, ALLOWED_EXTENSIONS_AV):
        return redirect(url_for('index'))
        
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Call your logic
    result = analysis_logic.analyze_audio(file_path)
    
    return render_template('index.html', audio_result=result, audio_file=filename)

@app.route('/analyze_video', methods=['POST'])
def handle_video():
    if 'file_input' not in request.files:
        return redirect(url_for('index'))
    
    file = request.files['file_input']
    if file.filename == '' or not allowed_file(file.filename, ALLOWED_EXTENSIONS_AV):
        return redirect(url_for('index'))
        
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Call your logic (This might take a long time!)
    result = analysis_logic.analyze_video(file_path)
    
    return render_template('index.html', video_result=result, video_file=filename)

# --- Run the App ---
if __name__ == '__main__':
    app.run(debug=True)