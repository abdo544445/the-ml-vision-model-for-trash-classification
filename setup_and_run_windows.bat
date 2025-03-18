@echo off
echo Setting up AI Object Detection Project...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed! Please install Python 3.8 or later from https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Check if Node.js is installed
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Node.js is not installed! Please install Node.js from https://nodejs.org/
    pause
    exit /b 1
)

REM Check if pip is installed
pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo pip is not installed! Please install pip.
    pause
    exit /b 1
)

REM Create virtual environment if it doesn't exist
if not exist venv (
    echo Creating Python virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate

REM Install Python dependencies
echo Installing Python dependencies...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics flask flask-cors pillow

REM Create necessary directories
if not exist functions\lib mkdir functions\lib
if not exist functions\src mkdir functions\src
if not exist public mkdir public

REM Create server.py if it doesn't exist
echo Creating server.py...
(
echo from flask import Flask, request, jsonify, send_from_directory
echo from flask_cors import CORS
echo import os
echo import sys
echo import logging
echo.
echo # Configure logging
echo logging.basicConfig(level=logging.DEBUG^)
echo logger = logging.getLogger(__name__^)
echo.
echo # Add the functions directory to Python path
echo current_dir = os.path.dirname(os.path.abspath(__file__^)^)
echo functions_dir = os.path.join(current_dir, 'functions', 'src'^)
echo sys.path.insert(0, functions_dir^)
echo.
echo try:
echo     from ML_Processing_Module import predict_image
echo     logger.info("Successfully imported ML_Processing_Module"^)
echo except Exception as e:
echo     logger.error(f"Error importing ML_Processing_Module: {str(e^)}"^)
echo     raise
echo.
echo app = Flask(__name__, static_folder='public', static_url_path=''^^)
echo CORS(app, resources={
echo     r"/*": {
echo         "origins": "*",
echo         "methods": ["GET", "POST", "OPTIONS"],
echo         "allow_headers": ["Content-Type"]
echo     }
echo }^)
echo.
echo @app.route('/'^^)
echo def index(^):
echo     return send_from_directory('public', 'index.html'^)
echo.
echo @app.route('/detect', methods=['POST', 'OPTIONS']^)
echo def detect_objects(^):
echo     if request.method == 'OPTIONS':
echo         return '', 204
echo     # Rest of your detection code here
echo     return jsonify({'message': 'Detection endpoint ready'}^)
echo.
echo if __name__ == '__main__':
echo     app.run(host='0.0.0.0', port=8080, debug=True^)
) > server.py

REM Create package.json if it doesn't exist
if not exist package.json (
    echo Creating package.json...
    (
    echo {
    echo   "name": "ai-object-detection",
    echo   "version": "1.0.0",
    echo   "description": "AI Object Detection with YOLOv5",
    echo   "scripts": {
    echo     "start": "http-server public -p 3000 --cors"
    echo   },
    echo   "dependencies": {
    echo     "http-server": "^14.1.1"
    echo   }
    echo }
    ) > package.json
)

REM Install Node.js dependencies
echo Installing Node.js dependencies...
npm install

REM Copy Python script to lib directory
echo Copying ML processing module...
if exist functions\src\ML_Processing_Module.py (
    copy functions\src\ML_Processing_Module.py functions\lib\
) else (
    echo Warning: ML_Processing_Module.py not found in functions/src/
)

REM Start the servers in separate windows
echo Starting servers...
start cmd /k "call venv\Scripts\activate && python server.py"
timeout /t 5
start cmd /k "npm start"

echo.
echo Setup complete! The application should now be running.
echo - Web interface: http://localhost:3000
echo - Detection server: http://localhost:8080
echo.
echo Note: Make sure both servers are running before using the application.
echo The web interface will be available at http://localhost:3000
echo The detection server will be available at http://localhost:8080
echo.
echo Press any key to exit this window. The servers will continue running in their own windows.
pause 