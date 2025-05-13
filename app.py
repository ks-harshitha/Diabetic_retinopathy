from flask import Flask, render_template, request
import os
import sqlite3
import datetime

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create database and table
conn = sqlite3.connect('patients.db')
c = conn.cursor()
c.execute('''
CREATE TABLE IF NOT EXISTS uploads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    upload_time TEXT NOT NULL
)
''')
conn.commit()
conn.close()

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # Save info into database
        conn = sqlite3.connect('patients.db')
        c = conn.cursor()
        c.execute('INSERT INTO uploads (filename, upload_time) VALUES (?, ?)',
                  (file.filename, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()
        conn.close()
        
        return f"Uploaded successfully! <br><img src='/static/uploads/{file.filename}' width='300'>"
    return 'Upload failed'

if __name__ == '__main__':
    app.run(debug=True)
