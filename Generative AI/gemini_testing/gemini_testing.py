from flask import Flask, render_template, request,url_for
import markdown2 
import os
import google.generativeai as genai
import PyPDF2


app = Flask(__name__)

api_key = os.getenv('GOOGLE_API')
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-pro')


# List to store user messages and processed messages as tuples
message_history = []

@app.route('/')
def index():
    return render_template('index.html', message_history=message_history)

@app.route('/upload_pdfs', methods=['POST'])
def upload_pdfs():
    if 'pdf1' not in request.files or 'pdf2' not in request.files:
        return 'Please upload both PDF files.', 400

    pdf1 = request.files['pdf1']
    pdf2 = request.files['pdf2']

    pdf1_text = extract_text_from_pdf(pdf1)
    pdf2_text = extract_text_from_pdf(pdf2)

    # You can now process the extracted text as needed
    # For demonstration, we'll append the combined text to message history
    # combined_text = f"PDF 1 Content:\n{pdf1_text}\n\nPDF 2 Content:\n{pdf2_text}"
    # message_history.append(("PDFs Uploaded", combined_text))

    # process the pdf1_text and pdf2_text 
    
    return render_template('index.html', message_history=message_history)

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page].extract_text()
    return text


@app.route('/process_user_input', methods=['POST'])
def process_user_input():
    if request.method == 'POST':
        user_message = request.form['user_message']
        
        # Process the user message (replace this with your processing logic)
        processed_message = model.generate_content(user_message)

        processed_message = markdown2.markdown(processed_message.text)
        # Store both user message and processed message as a tuple
        message_history.append((user_message, processed_message))

        return render_template('index.html', message_history=message_history)

if __name__ == '__main__':
    app.run(debug=True)
