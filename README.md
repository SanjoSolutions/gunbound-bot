# Gunbound Bot

This work is devoted to God.

## Install

Python 3.8 has been used.

### Python package dependencies

```
pip install -r requirements.txt
```

### Interception

The bot requires [Interception](http://www.oblita.com/interception.html) to be installed for automatic shooting.
Interception allows to send key events to the Gunbound client in a way that
is indistinguishable from hardware key events for user-mode applications.

1. [Download Interception](https://github.com/oblitum/Interception/releases/tag/v1.0.1)
2. Extract the files from the downloaded ZIP.
3. Open a Windows command prompt as administrator
4. In the Windows command prompt:
    1. `cd "Interception\Interception\command line installer"`
    2. `install-interception.exe /install`
5. Reboot.

## Running

1. Run Gunbound in window mode.
2. Run the bot with `python main.py`

### Hotkeys

* __Insert:__ Automatically shoot to the current target position.
* __Alt:__: Hold for backshot.
