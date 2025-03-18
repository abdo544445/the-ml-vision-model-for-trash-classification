#!/usr/bin/env python
"""
Test script for the AI Trash Detection server.
This script activates the virtual environment and runs the server.py file.
"""

import os
import sys
import subprocess
import webbrowser
from pathlib import Path

def main():
    # Get the absolute path to the project root directory
    project_dir = Path(__file__).parent.absolute()
    
    # Path to the virtual environment
    venv_dir = project_dir / "venv"
    
    # Check if virtual environment exists
    if not venv_dir.exists():
        print("Error: Virtual environment not found.")
        print("Please ensure you have created a Python virtual environment in the 'venv' directory.")
        return 1
    
    # Get the path to the Python executable in the virtual environment
    if sys.platform == "win32":
        python_executable = venv_dir / "Scripts" / "python.exe"
    else:  # Linux/Mac
        python_executable = venv_dir / "bin" / "python"
    
    if not python_executable.exists():
        print(f"Error: Python executable not found at {python_executable}")
        return 1
    
    # Install required packages if needed
    print("Checking if required packages are installed...")
    packages = [
        "torch", 
        "torchvision", 
        "flask", 
        "flask-cors", 
        "pillow", 
        "ultralytics"
    ]
    
    try:
        # Use subprocess to run the pip install command
        for package in packages:
            print(f"Checking {package}...")
            cmd = [str(python_executable), "-m", "pip", "install", "-q", package]
            subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error installing packages: {e}")
        return 1
    
    # Run the server
    print("\n" + "="*50)
    print("Starting AI Trash Detection server...")
    print("="*50)
    
    # Set up environment variable for the Flask app
    env = os.environ.copy()
    env["FLASK_APP"] = "server.py"
    
    server_process = None
    try:
        # Start the server process
        server_process = subprocess.Popen(
            [str(python_executable), "server.py"],
            env=env,
            cwd=str(project_dir)
        )
        
        # Open the web browser after a delay to allow the server to start
        print("Server is starting. Opening web browser in 3 seconds...")
        import time
        time.sleep(3)
        
        # Open the website in the default browser
        webbrowser.open("http://localhost:8080")
        
        print("\nServer is now running. Press Ctrl+C to stop.")
        print("\nTesting Instructions:")
        print("1. Use the web interface to upload an image or use the webcam")
        print("2. Try both server-side and browser-side processing")
        print("3. Check the detection results for trash items")
        print("4. Test with different types of waste (plastic, paper, metal, etc.)")
        
        # Keep the script running until interrupted
        server_process.wait()
        
    except KeyboardInterrupt:
        print("\nShutting down server...")
    finally:
        if server_process:
            server_process.terminate()
    
    print("Server has been stopped.")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 