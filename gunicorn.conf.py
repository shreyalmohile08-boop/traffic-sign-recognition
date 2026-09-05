"""
gunicorn.conf.py
Configuration file for Gunicorn on Render.
Binds dynamically to the PORT environment variable assigned by Render.
"""

import os

# Bind to 0.0.0.0 and the PORT provided by Render (default 10000)
port = os.environ.get("PORT", "10000")
bind = f"0.0.0.0:{port}"

# Single worker with 2 threads avoids exceeding Render free tier 512MB RAM
workers = 1
threads = 2
timeout = 120
keepalive = 5
preload_app = False
