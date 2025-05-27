"""
WSGI entry point for Waitress to serve the web UI
"""

from web_app import app

# This allows Waitress to import the application object
application = app

if __name__ == "__main__":
    application.run()
