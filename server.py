"""
Server module for Warden application.
Provides functionality to run both Flask and FastAPI applications using Waitress and Uvicorn.
"""
import os
import sys
import multiprocessing
from waitress import serve
import uvicorn

# Import applications
from wsgi import application as flask_app

def run_flask_app(host="127.0.0.1", port=5000, threads=4):
    """
    Run the Flask web app using Waitress WSGI server
    
    Args:
        host: Host address to bind to
        port: Port to listen on
        threads: Number of worker threads
    """
    print(f"Starting Web UI at http://{host}:{port} with Waitress ({threads} threads)")
    serve(flask_app, host=host, port=port, threads=threads)

def run_fastapi_app(host="127.0.0.1", port=8000):
    """
    Run the FastAPI application using Uvicorn
    
    Args:
        host: Host address to bind to
        port: Port to listen on
    """
    # Import here to avoid circular imports
    from fastapi_server import app as fastapi_app
    
    print(f"Starting FastAPI server at http://{host}:{port}")
    uvicorn.run(fastapi_app, host=host, port=port)

def run_combined(host="127.0.0.1", api_port=8000, web_port=5000, threads=4):
    """
    Start both FastAPI and Flask web UI servers
    
    Args:
        host: Host address for both servers
        api_port: Port for FastAPI server
        web_port: Port for Flask web UI
        threads: Number of threads for Waitress
    """
    # Start web UI in a separate process
    web_process = multiprocessing.Process(
        target=run_flask_app,
        args=(host, web_port, threads)
    )
    web_process.daemon = True
    web_process.start()
    
    print(f"Starting Web UI at http://{host}:{web_port} with Waitress ({threads} threads)")
    # Run FastAPI in the main process
    run_fastapi_app(host, api_port)

if __name__ == "__main__":
    import argparse
    
    # Create argument parser for better command-line handling
    parser = argparse.ArgumentParser(description="Warden Server - Run FastAPI and Flask applications")
    parser.add_argument("--host", default="127.0.0.1", help="Host address to bind to")
    parser.add_argument("--api-port", type=int, default=8000, help="Port for FastAPI server")
    parser.add_argument("--web-port", type=int, default=5000, help="Port for Web UI")
    parser.add_argument("--threads", type=int, default=4, help="Number of Waitress worker threads")
    parser.add_argument("--web-only", action="store_true", help="Start only the Web UI")
    parser.add_argument("--api-only", action="store_true", help="Start only the FastAPI server")
    
    args = parser.parse_args()
    
    # Process command line flags
    if args.web_only:
        # Start only the Flask web UI
        print(f"Starting Web UI only at http://{args.host}:{args.web_port}")
        run_flask_app(args.host, args.web_port, args.threads)
    elif args.api_only:
        # Start only the FastAPI server
        print(f"Starting FastAPI server only at http://{args.host}:{args.api_port}")
        run_fastapi_app(args.host, args.api_port)
    else:
        # Start both by default
        print(f"Starting both FastAPI server and Web UI")
        print(f"- FastAPI: http://{args.host}:{args.api_port}")
        print(f"- Web UI:  http://{args.host}:{args.web_port}")
        run_combined(args.host, args.api_port, args.web_port, args.threads)
