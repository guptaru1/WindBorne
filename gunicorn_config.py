# Gunicorn configuration file
timeout = 120
workers = 4
bind = "0.0.0.0:10000"
worker_class = "sync"
keepalive = 120 