# Subscriber widget

Small always-on-top window for Windows 11. No API keys and no account login.

## Run

1. Install Python 3.11 or newer from [python.org](https://www.python.org/downloads/windows/).
   Leave **Add python.exe to PATH** and **tcl/tk and IDLE** enabled.
2. Open this folder in a terminal.
3. Start the window:

```bat
py -3 app.py
```

Or double-click `run.bat`.

The window stays above other windows on a transparent background. Drag a ring to move it. Click a green number, type the follower goal, and press Enter: the peach arc fills toward that goal and the rest of the ring stays dark gray. Right-click to refresh or close. F5 refreshes too, and the window refreshes itself every 3 minutes.

Print the counts without opening a window:

```bat
py -3 app.py --once
```

Run the parser tests (they do not use the network):

```bat
py -3 -m unittest discover -s tests
```
