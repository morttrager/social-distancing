from threading import Thread
from time import sleep

from flask import Flask, render_template
from flask_socketio import SocketIO, send, emit

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)
socketio.init_app(app, cors_allowed_origins="*", async_mode='threading')


@app.route('/')
def index():
    """ Displays the index page accessible at '/'
    """
    return render_template('index1.html')


def operation():
    for i in range(10):
        print(i)
        socketio.emit("unique-id", i)
        socketio.sleep(5)


@socketio.on('connect')
def handle_connect():
    Thread(target=operation).start()


if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0')