SmartGlow 🌟

SmartGlow is a Python-based screen-brightness controller that combines
webcam-based lighting estimation with hand-gesture recognition. It can
adjust screen brightness automatically and lets you control its
operating mode from the keyboard.

Features

Automatic brightness adjustment using brightness estimates from
the webcam image.

Hand-gesture recognition powered by MediaPipe.

Brightness smoothing to help reduce sudden changes.

Manual brightness adjustment using keyboard controls.

Automatic model download when the MediaPipe hand-landmarker
model is not already present.

Technologies and dependencies

SmartGlow uses these external Python packages:

Package                             Purpose

opencv-python                     Webcam access, video frames, and
on-screen display

mediapipe                         Hand-landmark detection for gesture
recognition

numpy                             Numerical and image-processing
operations

It also uses Python standard-library modules, including os, time,
urllib.request, and datetime. These do not need to be installed
separately.

Requirements

Windows is the intended environment for the setup instructions
below.

Python installed on your computer.

A working webcam.

An internet connection on the first run if the hand-landmarker model
needs to be downloaded.

A display supported by screen-brightness-control.

Python compatibility note: MediaPipe compatibility depends on the
Python version and package release. If installation fails with your
current Python version, try a supported Python release such as Python
3.12, then create the virtual environment with that interpreter.

Installation

1. Download or clone the repository

If you use Git:

git clone <your-repository-url>
cd SmartGlow

Or download the repository from GitHub and open its folder in a
terminal.

2. Create a virtual environment

python -m venv .venv

If you have multiple Python versions installed and want to use Python
3.12, create the environment with that interpreter:

py -3.12 -m venv .venv

3. Activate the environment

In Windows PowerShell:

.venv\Scripts\Activate.ps1

In Windows Command Prompt:

.venv\Scripts\activate.bat

4. Install the dependencies

Make sure requirements.txt is in the project folder, then run:

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

The requirements.txt file should contain:

opencv-python
mediapipe
numpy
screen-brightness-control

5. Run SmartGlow

If your main script is named SmartGlow.py, run:

python SmartGlow.py

Allow webcam access if Windows asks for permission. On the first run,
the program may download the MediaPipe hand-landmarker model. An
internet connection may be required for that download.

Keyboard controls

The program includes these keyboard controls:

Key                Action

Q                Quit the application
A                Toggle automatic brightness mode
↑ (Up Arrow)     Increase brightness
↓ (Down Arrow)   Decrease brightness

The application also displays information such as the current
brightness, target brightness, operating mode, detected gesture, and
estimated light level.

Project files

A simple repository layout can look like this:

SmartGlow/
├── SmartGlow.py
├── requirements.txt
├── README.md
└── .gitignore

The hand-landmarker model may be downloaded automatically by the program
at runtime, so it may not be present in the repository initially.

Recommended .gitignore

Create a .gitignore file in the project folder to avoid uploading your
virtual environment and Python cache files:

.venv/
.venv312/
__pycache__/
*.py[cod]

Do not commit your virtual environment folder. Other generated model
files can be excluded or included according to the project's
distribution needs; check the model's licensing and hosting terms before
redistributing it.

Troubleshooting

MediaPipe fails to install

Check which Python version is active:

python --version

If your version is not supported by the MediaPipe release being
installed, install a compatible Python version and recreate the virtual
environment using that interpreter.

The webcam does not open

Close other applications that may be using the webcam.

Check Windows camera privacy permissions.

Confirm that the correct camera is selected. The current code uses
camera index 0.

Brightness does not change

Confirm that your display is supported by
screen-brightness-control.

Try running the program on the laptop's built-in display.

Some external monitors do not support brightness control through
this package.

The hand-landmarker model does not download

Check your internet connection.

Make sure the application has permission to write to its project
directory.

Run the program again after resolving the download issue.

Notes

Webcam image brightness is an estimate of visible light in the
camera frame; it is not a dedicated ambient-light sensor reading.

Gesture recognition and brightness control may behave differently
depending on lighting, camera position, hardware, and display
support.

This project is provided for learning and experimentation. Test it
on your own hardware before relying on it.

License

No license has been specified yet. If you want others to reuse, modify,
or distribute this project, choose a license and add a LICENSE file to
the repository.
