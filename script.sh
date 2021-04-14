echo "shell /bin/bash" > /root/.screenrc
screen -dmS mlflow
screen -S mlflow -X stuff 'mlflow ui --port 5001\n'
sleep 5
screen -dmS head-detection
screen -S head-detection -X stuff 'python src/server.py\n'
tail -f /dev/null
