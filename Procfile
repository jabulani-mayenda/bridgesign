web: gunicorn app:app --worker-class gthread --workers 1 --threads ${GUNICORN_THREADS:-4} --bind 0.0.0.0:$PORT --timeout ${GUNICORN_TIMEOUT:-180}
