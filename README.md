> [!NOTE]
> A working bot for the [official Gunbound](https://gunbound.gnjoy.asia/) can be purchased from me [here](https://www.patreon.com/SanjoSolutions/shop/bot-for-gunbound-674682).

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
* __Alt:__ Hold for backshot.

## Known limitations

* In tag mode the mobile is determined to always be the main one.<br>
  Workaround: Set the mobile manually in the code and restart the bot.

## Open improvements

* [ ] Visualization of how the Trico shot 2 rotates along the shot line
* [ ] Boomer hook shots
* [ ] Calculation and visualization of the underground part of Nak shot 2
* [ ] Calculation and visualization of the pop point of the Turtle super shot
* [ ] Calculation and visualization of the positions of the projectiles of Big Foot shot 1
* [ ] Calculation and visualization of the positions of the projectiles of Big Foot shot 2
* [ ] Calculation and visualization of the positions of the projectiles of Big Foot super shot
