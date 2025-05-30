#!/bin/bash
apt-get update
apt-get install -y ffmpeg
gunicorn --bind=0.0.0.0:8000 app:app
