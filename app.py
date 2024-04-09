from flask import Flask, render_template, request
import main

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        max_workers = request.form.get('max_workers', type=int, default=32)

        if file and file.filename.endswith('.svs'): 
            file_path = 'temp.svs' 
            file.save(file_path) 

            result = main.main(file_path, 1024, max_workers) 

            return render_template('index.html', result=result)
    return render_template('index.html', result=None)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8888)

