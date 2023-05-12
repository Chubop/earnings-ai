from flask import Flask, request
from main import init
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
chat_history = []
Index = init(directory="transcripts/AAPL/2023", chat_history=chat_history)


@app.route('/receive', methods=['POST'])
def receive():
    message = request.get_json().get('message')
    print(' * Received query', message)
    response = Index.ask_question(message, verbose=True)
    chat_history.append(response)
    print(' * Response', response)
    return response[0]['answer'].strip('\n')


if __name__ == '__main__':
    app.run(debug=True)
