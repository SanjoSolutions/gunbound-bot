from ctypes import windll

from interception.interception import interception, MAX_DEVICES, key_stroke, interception_key_state

MAPVK_VK_TO_VSC = 0


class KeyPressing:
    def __init__(self):
        self.context = interception()
        self.keyboard = self.get_keyboard()

    def get_keyboard(self):
        for i in range(MAX_DEVICES):
            if interception.is_keyboard(i):
                return i
        return None

    def press_key(self, key):
        scan_code = windll.user32.MapVirtualKeyA(key, MAPVK_VK_TO_VSC)
        key_press = key_stroke(
            scan_code,
            interception_key_state.INTERCEPTION_KEY_DOWN.value,
            0
        )
        self.context.send(self.keyboard, key_press)

    def release_key(self, key):
        scan_code = windll.user32.MapVirtualKeyA(key, MAPVK_VK_TO_VSC)
        key_release = key_stroke(
            scan_code,
            interception_key_state.INTERCEPTION_KEY_UP.value,
            0
        )
        self.context.send(self.keyboard, key_release)
