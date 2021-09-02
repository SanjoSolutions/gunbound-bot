import sys
from enum import IntEnum
from math import cos, sin, sqrt, radians
from time import sleep, time

import numpy as np
import pyautogui
from win32con import VK_SPACE
from PyQt5.QtWidgets import QApplication
from mss import mss
from pymem import Pymem
from win32gui import GetClientRect, ClientToScreen, FindWindow, GetCursorPos, GetForegroundWindow, GetDC, GetPixel
import cv2 as cv

from key_pressing import KeyPressing
from transparent_window import TransparentWindow
from global_hotkeys import *


class CartFacingDirection(IntEnum):
    Left = 0
    Right = 1


class GunboundProcess:
    def __init__(self, process):
        self.process = process

    def read_wind_speed(self):
        a = self.process.read_uint(0x87053c)
        b = self.process.read_uint(0x870140 + a * 4)
        # b seems to be the address of some class instance
        address = b + 0x1234
        wind_speed = self.process.read_bytes(address, 1)[0]
        return wind_speed

    def read_wind_direction(self):
        """
        Wind direction is a whole number in degrees.
        Zero degree is at the right side.
        The angle increases counter-clockwise.
        """
        a = self.process.read_uint(0x87053c)
        b = self.process.read_uint(0x870140 + a * 4)
        # b seems to be the address of some class instance
        address = b + 0x1234 + 0x1
        wind_direction = self.process.read_ushort(address)
        return wind_direction

    def read_angle(self, index):
        b = self.process.read_uint(0x8F4A00 + 0x20)
        address = b + 0x19EC
        cart_angle = self.process.read_int(address)

        cart_angle2 = self.read_cart_angle(index)

        cart_facing_direction = self.read_cart_facing_direction(index)

        a = cart_angle
        if cart_facing_direction == CartFacingDirection.Left:
            a += cart_angle2
        else:
            a -= cart_angle2
        a %= 360

        return a

    def read_player_index(self):
        address = self.process.base_address + 0x4F3929
        player_index = self.process.read_bytes(address, 1)[0]
        return player_index

    def read_mobile_id(self):
        mobile_id = self.process.read_bytes(self.process.base_address + 0x497368, 1)[0]
        return mobile_id

    def read_cart_angle(self, index):
        address = self._determine_player_address(index)
        cart_angle = self.process.read_int(address + 0x8)
        return cart_angle

    def _determine_player_address(self, index):
        address = self.process.base_address + 0x482A58 + index * 0x18
        return address

    def read_cart_facing_direction(self, index):
        address = self._determine_player_address(index)
        cart_facing_direction = CartFacingDirection(
            self.process.read_bytes(address + 0xC, 1)[0]
        )
        return cart_facing_direction

    def read_cart_position(self, mobile, index):
        # # a = self.process.read_uint(0x8701CC)
        # # b = self.process.read_uint(a + 0x77138)
        # c = self.process.read_uint(b + 0x4)
        # d = self.process.read_uint(c + 0x1c)
        # e = self.process.read_uint(d + 0x10)
        # cart_position = self.process.read_uint(e + 0x56C)
        # return cart_position
        address = self._determine_player_address(index)
        x = self.process.read_ushort(address + 0x0)
        y = self.process.read_ushort(address + 0x4)
        cart_facing_direction = self.read_cart_facing_direction(index)
        cart_angle = self.read_cart_angle(index)
        if cart_facing_direction == CartFacingDirection.Left:
            cart_angle = (cart_angle + 180) % 360
        OFFSET_X = 0
        OFFSET_Y = -40
        ROTATED_OFFSET_X = 16 if cart_facing_direction == CartFacingDirection.Left else 15
        ROTATED_OFFSET_Y = 25
        x_offset_angle = cart_angle
        if mobile == Mobile.Nak:
            x_offset_angle += 180
        x_offset_angle = radians(x_offset_angle)
        angle_delta = -90 if cart_facing_direction == CartFacingDirection.Left else 90
        y_offset_angle = radians(cart_angle + angle_delta)
        return (
            int(round(
                x +
                OFFSET_X +
                ROTATED_OFFSET_X * cos(x_offset_angle) +
                ROTATED_OFFSET_Y * cos(y_offset_angle)
            )),
            int(round(
                y +
                OFFSET_Y +
                ROTATED_OFFSET_X * -sin(x_offset_angle) +
                ROTATED_OFFSET_Y * -sin(y_offset_angle)
            )),
        )

    def read_screen_center_position(self):
        return (
            self.process.read_ushort(self.process.base_address + 0x4E98D4),
            self.process.read_ushort(self.process.base_address + 0x4E98D8),
        )


SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600


def calculate_angle_2_4_power(process: GunboundProcess, window, index):
    mobile = Mobile.Turtle
    target_position = determine_target_position(process, window)
    cart_position = process.read_cart_position(mobile, index)
    distance = abs(target_position[0] - cart_position[0])
    distance_in_parts = convert_distance_to_distance_in_parts(distance)
    angle = 90 - distance_in_parts
    # angle = 90 - distance + wind_direction_factor * wind_speed
    wind_direction = process.read_wind_direction()
    wind_speed = process.read_wind_speed()
    wind_direction_factor = determine_wind_direction_factor(wind_direction, cart_position, target_position)
    angle += wind_direction_factor * wind_speed
    if distance_in_parts <= 7.5:
        power = 2.35
    elif distance_in_parts <= 22.5:
        power = 2.4
    elif distance_in_parts <= 27.5:
        power = 2.45
    else:
        power = 2.5
    if 90 - 22.5 <= wind_direction <= 90 + 22.5:
        power -= 0.1 * (wind_speed // 7)
    elif 270 - 22.5 <= wind_direction <= 270 + 22.5:
        power += 0.1 * (wind_speed // 7)

    return angle, power


def determine_target_position(process, window):
    cursor_position = GetCursorPos()
    screen_center_position = process.read_screen_center_position()
    client_area_rect = determine_client_area_rect(window)
    target_position = (
        int(
            screen_center_position[0] - 0.5 * SCREEN_WIDTH + (cursor_position[0] - client_area_rect['left']) /
            client_area_rect['width'] * SCREEN_WIDTH
        ),
        int(
            screen_center_position[1] - 0.5 * SCREEN_HEIGHT + (cursor_position[1] - client_area_rect['top']) /
            client_area_rect['height'] * SCREEN_HEIGHT
        )
    )
    return target_position


def calculate_power(angle, from_position, to_position):
    distance = calculate_distance_in_screens(from_position, to_position)
    if 0 <= angle <= 90:
        angle_cart = angle
    elif 90 < angle <= 270:
        angle_cart = 180 - angle
    elif 270 < angle <= 360:
        angle_cart = -(360 - angle)
    return (angle_cart - 90) / -distance / 6.557377049


class Mobile(IntEnum):
    Armor = 0
    Mage = 1
    Nak = 2
    Trico = 3
    BigFoot = 4
    Boomer = 5
    Raon = 6
    Lightning = 7
    JD = 8
    ASate = 9
    Ice = 10
    Turtle = 11
    Grub = 12
    Aduka = 13
    Knight = 13
    Kalsiddon = 14
    JFrog = 15
    Dragon = 15
    Random = 255


class Direction(IntEnum):
    Left = 0
    Right = 1


MIN_MAP_X = 0
MAX_MAP_X = 1800

MIN_MAP_Y = -20
MAX_MAP_Y = 1840
MAP_HEIGHT = MAX_MAP_Y - MIN_MAP_Y


g_table1 = (
    73.5,  # Armor
    71.5,  # Mage
    93.0,  # Nak
    84.0,  # Trico
    90.0,  # Bigfoot
    62.5,  # Boomer
    81.0,  # Raon
    65.0,  # Lightning
    62.5,  # JD
    76.0,  # Asate
    62.5,  # Ice
    73.5,  # Turtle
    61.0,  # Grub
    65.5,  # Aduka/Knight
    54.3,  # JFrog/Dragon
    88.5,  # Kalsiddon
    1.0,  # Slots
    1.0  # Aid
)

g_table2 = (
    0.74,  # Armor
    0.78,
    0.99,
    0.87,
    0.74,
    1.395,
    0.827,
    0.72,
    0.625,
    0.765,
    0.625,
    0.74,
    0.65,
    0.695,
    0.67,
    0.905,
    0.0,  # Slots
    0.0  # Aid
)


DEGTORAD = 0.0174532925199433


def determine_power(
    mobile,
    source_position,
    target_position,
    angle,
    direction,
    backshot,
    wind_angle,
    wind_power
):
    target_x, target_y = target_position
    target_y = MAX_MAP_Y - target_y
    minimum_distance = 9999
    power_to_shoot_with = None
    STEP_SIZE = 0.05

    for power in range(400 + 1):
        last_x = None
        last_y = None

        for position in generate_coordinates(
            mobile,
            source_position,
            angle,
            direction,
            backshot,
            wind_angle,
            wind_power,
            power,
            STEP_SIZE
        ):
            x, y = position
            if (
                last_x is not None and last_y is not None and (
                    last_y <= target_y <= y or
                    y <= target_y <= last_y
                )
            ):
                average_x = (last_x + x) / 2.0
                distance = abs(target_x - average_x)

                if distance < minimum_distance:
                    minimum_distance = distance
                    power_to_shoot_with = power
                    break

            last_x = x
            last_y = y

    return power_to_shoot_with


def generate_coordinates(
    mobile,
    source_position,
    angle,
    direction,
    backshot,
    wind_angle,
    wind_power,
    power,
    step_size
):
    # compute horizontal/wind
    x_v2 = int(cos(wind_angle * DEGTORAD) * wind_power) * g_table2[mobile]

    # compute downward/gravity
    y_v2 = int(sin(wind_angle * DEGTORAD) * wind_power) * g_table2[mobile] - g_table1[mobile]

    if mobile == Mobile.Nak and backshot and angle <= 70:
        y_v2 *= -8.0

    x_v = cos(angle * DEGTORAD)
    y_v = sin(angle * DEGTORAD)

    temp_x_v = x_v * power
    temp_y_v = y_v * power

    xxx = source_position[0]
    delta_yyy = MAX_MAP_Y - source_position[1]

    # if direction == Direction.Left:
    #     temp_x_v *= -1

    if mobile == Mobile.Nak and backshot and angle <= 70:
        temp_x_v *= 2

    if delta_yyy >= 0:
        while MIN_MAP_X < xxx < MAX_MAP_X and delta_yyy >= 0:
            # calc projectile x,y coord

            xxx += temp_x_v * step_size
            delta_yyy += temp_y_v * step_size

            yield (xxx, delta_yyy)

            # calc projectile x,y velocity (+ wind/gravity)

            temp_x_v += x_v2 * step_size
            temp_y_v += y_v2 * step_size


def draw_position(position, process, image, mobile_angle=None, cart_facing_direction=None):
    scx, scy = process.read_screen_center_position()
    visible_map_area = (
        scx - SCREEN_WIDTH // 2,
        scy - SCREEN_HEIGHT // 2,
        SCREEN_WIDTH,
        SCREEN_HEIGHT
    )
    min_x = visible_map_area[0]
    min_y = visible_map_area[1]

    x = position[0] - min_x
    y = position[1] - min_y

    RADIUS = 17

    if mobile_angle is not None:
        cv.line(
            image,
            (x, y),
            (
                int(round(x + RADIUS * cos(radians(mobile_angle)))),
                int(round(y + RADIUS * -sin(radians(mobile_angle))))
            ),
            (255, 0, 0, 255)
        )
        if cart_facing_direction is not None:
            angle_offset = -90 if cart_facing_direction == CartFacingDirection.Left else 90
            angle = mobile_angle + angle_offset
            cv.line(
                image,
                (x, y),
                (
                    int(round(x + RADIUS * cos(radians(angle)))),
                    int(round(y + RADIUS * -sin(radians(angle))))
                ),
                (255, 0, 0, 255)
            )

    cv.circle(image, (x, y), RADIUS, (0, 0, 255, 255), thickness=1)
    # cv.circle(image, (x, y), 1, (0, 0, 255, 255), thickness=1)

    if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
        image[y, x] = (0, 255, 0, 255)


BOTTOM_UI_HEIGHT = 84
MAX_SHOT_LINE_DRAW_Y = SCREEN_HEIGHT - BOTTOM_UI_HEIGHT  # to prevent drawing over the UI at the bottom


def draw_shot_line(
    process,
    image,
    mobile,
    start_position,
    direction,
    backshot,
    angle,
    power,
    wind_angle,
    wind_power
):
    scx, scy = process.read_screen_center_position()
    visible_map_area = (
        scx - SCREEN_WIDTH // 2,
        scy - SCREEN_HEIGHT // 2,
        SCREEN_WIDTH,
        SCREEN_HEIGHT
    )
    min_x = visible_map_area[0]
    min_y = visible_map_area[1]

    STEP_SIZE = 0.05

    shot_line_drawing = np.zeros((MAX_SHOT_LINE_DRAW_Y, image.shape[1], image.shape[2]))

    previous_position_on_image = None
    for position in generate_coordinates(mobile, start_position, angle, direction, backshot, wind_angle, wind_power,
                                         power, STEP_SIZE):
        x = position[0]
        y = -(position[1] - MAX_MAP_Y)
        position_on_image = (
            int(round(x - min_x)),
            int(round(y - min_y))
        )
        if previous_position_on_image is not None:
            cv.line(
                shot_line_drawing,
                previous_position_on_image,
                position_on_image,
                (0, 0, 255, 255),
                thickness=1
            )
        previous_position_on_image = position_on_image

    image[:MAX_SHOT_LINE_DRAW_Y] = shot_line_drawing


def change_direction(direction):
    if direction == Direction.Left:
        return Direction.Right
    else:
        return Direction.Left


def calculate_distance_in_screens(from_position, to_position):
    return abs(to_position[0] - from_position[0]) / SCREEN_WIDTH


def convert_distance_to_distance_in_parts(distance):
    return distance / SCREEN_WIDTH * 30


def determine_wind_direction_factor(wind_direction, cart_position, target_position):
    interval = 45
    if cart_position[0] > target_position[0]:
        wind_direction = 180 - wind_direction
        if wind_direction < 0:
            wind_direction += 360
    if wind_direction >= 360 - 0.5 * interval and wind_direction <= 360 or wind_direction < 0.5 * interval:
        factor = 0.6
    elif wind_direction < 1.5 * interval:
        factor = 0.7
    elif wind_direction < 2.5 * interval:
        factor = 0.0
    elif wind_direction < 3.5 * interval:
        factor = -0.35
    elif wind_direction < 4.5 * interval:
        factor = -0.6
    elif wind_direction < 5.5 * interval:
        factor = -0.4
    elif wind_direction < 6.5 * interval:
        factor = 0.0
    else:
        factor = 0.25
    return factor


def determine_client_area_rect(hwnd):
    left, top, right, bottom = GetClientRect(hwnd)
    left2, top2 = ClientToScreen(hwnd, (left, top))
    right2, bottom2 = ClientToScreen(hwnd, (right, bottom))
    client_area_rect = {
        'left': left2,
        'top': top2,
        'width': right2 - left2,
        'height': bottom2 - top2
    }
    return client_area_rect


screenshotter = mss()

a = {
    0: cv.imread('images/0.png'),
    1: cv.imread('images/1.png'),
    2: cv.imread('images/2.png'),
    3: cv.imread('images/3.png'),
    4: cv.imread('images/4.png'),
    5: cv.imread('images/5.png'),
    6: cv.imread('images/6.png'),
    7: cv.imread('images/7.png'),
    8: cv.imread('images/8.png'),
    9: cv.imread('images/9.png'),
}
sign_image2 = cv.imread('images/minus.png')
slice_mode_image = cv.imread('images/slice_mode.png')


def determine_angle(process: GunboundProcess, window, mobile, index, backshot):
    angle = read_angle(window)
    facing_direction = process.read_cart_facing_direction(index)
    if (
        (mobile == Mobile.Nak and facing_direction == CartFacingDirection.Right) or
        facing_direction == CartFacingDirection.Left
    ):
        angle = 90 + (90 - angle)
    elif facing_direction == CartFacingDirection.Right:
        if angle < 0:
            angle = 360 + angle
    if backshot:
        angle = 180 - angle
        if angle < 0:
            angle = 360 + angle
    return angle


def read_angle(window):
    sign_image = make_screenshot_cv(window, (
        218,
        536,
        8,
        5
    ))
    digit_1_image = make_screenshot_cv(window, (
        226,
        531,
        13,
        13
    ))
    angle = 0
    for digit, image in a.items():
        if (image == digit_1_image).all():
            angle = digit
            angle *= 10
            break
    digit_2_image = make_screenshot_cv(window, (
        239,
        531,
        13,
        13
    ))
    for digit, image in a.items():
        if (image == digit_2_image).all():
            angle += digit
            break
    if (sign_image == sign_image2).all():
        angle *= -1
    return angle


def make_screenshot_cv(window, area=None):
    image = make_screenshot_cv_rgb(window, area)
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    return image


def make_screenshot_cv_rgb(window, area=None):
    screenshot = make_screenshot(window, area)
    image = np.array(screenshot.pixels, dtype=np.uint8)
    return image


def make_screenshot(window, area=None):
    rect = determine_client_area_rect(window)
    if area:
        rect['left'] += area[0]
        rect['top'] += area[1]
        rect['width'] = area[2]
        rect['height'] = area[3]
    screenshot = screenshotter.grab(rect)
    return screenshot


class PixelGetter:
    def __init__(self, window):
        self.device_context = GetDC(window)

    def get_pixel(self, position):
        x, y = position
        color = GetPixel(self.device_context, x, y)
        return color


backshot = False
last_shot_at = None
parameters = None
shot_parameters = None


def main():
    global parameters
    global shot_parameters

    power = None
    window = FindWindow('Softnyx', None)
    process = GunboundProcess(Pymem('GunBound.gme'))
    key_pressing = KeyPressing()
    pixel_getter = PixelGetter(window)

    def on_hotkey():
        global last_shot_at
        global shot_parameters
        if power is not None:
            print('Shoot')
            last_shot_at = time()
            shot_parameters = parameters
            slice_shoot(power)

    def slice_shoot(power):
        if not is_slice_mode_active():
            activate_slice_mode()
        virtual_key_code = VK_SPACE
        key_pressing.press_key(virtual_key_code)
        while not should_release_space(power):
            pass
        key_pressing.release_key(virtual_key_code)

    def color_to_color_ref(color):
        r, g, b = color
        return (b << 16) | (g << 8) | r

    COLOR_1 = color_to_color_ref((208, 24, 32))
    COLOR_2 = color_to_color_ref((96, 0, 0))
    COLOR_3 = color_to_color_ref((192, 16, 0))

    def should_release_space(power):
        delay_offset = 0 # -7 if power < 400 else 0
        offset = max(0, power + delay_offset)
        pixel = pixel_getter.get_pixel(
            (
                power_bar_area[0] - 1 + offset,
                power_bar_area[1] + 13
            )
        )
        return (
            pixel == COLOR_1 or
            pixel == COLOR_2 or
            pixel == COLOR_3
        )

    def is_slice_mode_active():
        image = make_shoot_mode_image()
        return (image == slice_mode_image).all()

    def make_shoot_mode_image():
        return make_screenshot_cv(window, (
            205,
            576,
            31,
            12
        ))

    def activate_slice_mode():
        toggle_shoot_mode()

    def toggle_shoot_mode():
        client_area_rect = determine_client_area_rect(window)
        pyautogui.click(
            client_area_rect['left'] + 218,
            client_area_rect['top'] + 569
        )

    def on_alt_down():
        global backshot
        backshot = True

    def on_alt_up():
        global backshot
        backshot = False

    bindings = [
        [['insert'], on_hotkey, None],
        [['alt'], on_alt_down, on_alt_up],
    ]

    register_hotkeys(bindings)
    start_checking_hotkeys()

    application = QApplication(sys.argv)
    transparent_window = TransparentWindow()
    power_bar_area = (
        241,
        565,
        400,
        19
    )

    def mark_on_power_bar(image, power):
        position = calculate_power_bar_mark_position(power)
        draw_power_bar_mark(image, position)

    def calculate_power_bar_mark_position(power):
        return (
            int(power_bar_area[0] - 1 + power / 400.0 * power_bar_area[2]),
            power_bar_area[1]
        )

    def draw_power_bar_mark(image, position):
        cv.rectangle(
            image,
            (position[0] - 1, position[1] - 1),
            (position[0] + 1, position[1] + power_bar_area[3]),
            (0, 0, 255, 255),
            thickness=1
        )

    transparent_window.show()

    previous_parameters = None

    while True:
        if window != GetForegroundWindow():
            client_area_rect = determine_client_area_rect(window)
            image = create_image_with_size(client_area_rect['width'], client_area_rect['height'])
            image[0, 0] = (0, 0, 255, 255)
            transparent_window.show_image(image)
            while window != GetForegroundWindow():
                sleep(1 / 60)

        client_area_rect = determine_client_area_rect(window)
        transparent_window.setGeometry(
            client_area_rect['left'],
            client_area_rect['top'],
            client_area_rect['width'],
            client_area_rect['height']
        )

        player_index = process.read_player_index()
        mobile = Mobile(process.read_mobile_id())
        # mobile = Mobile.Grub
        if mobile == Mobile.Random:
            raise Exception(
                'It has been detected that the random mobile has been chosen. ' +
                'Please set the mobile manually in the code that has been randomly given.'
            )
        source_position = process.read_cart_position(mobile, player_index)
        source_x, source_y = source_position
        angle = determine_angle(process, window, mobile, player_index, backshot)
        wind_power = process.read_wind_speed()
        wind_angle = process.read_wind_direction()
        target_x, target_y = determine_target_position(process, window)
        target_position = (target_x, target_y)
        direction = Direction.Left if source_x > target_x else Direction.Right

        parameters = {
            'mobile': mobile,
            'source_position': source_position,
            'target_position': target_position,
            'angle': angle,
            'direction': direction,
            'backshot': backshot,
            'wind_angle': wind_angle,
            'wind_power': wind_power
        }

        if (
            (
                last_shot_at is None or
                time() - last_shot_at > 15
            ) and
            have_parameters_changed(parameters, previous_parameters)
        ):
            shot_parameters = None
            power = determine_power(
                mobile,
                source_position,
                target_position,
                angle,
                direction,
                backshot,
                wind_angle,
                wind_power
            )

        image = create_image_with_size(client_area_rect['width'], client_area_rect['height'])
        draw_parameters = shot_parameters if shot_parameters is not None else parameters
        if power is not None:
            mark_on_power_bar(image, power)
            draw_shot_line(
                process,
                image,
                mobile,
                draw_parameters['source_position'],
                draw_parameters['direction'],
                draw_parameters['backshot'],
                draw_parameters['angle'],
                power,
                draw_parameters['wind_angle'],
                draw_parameters['wind_power']
            )
        draw_position(
            draw_parameters['source_position'],
            process,
            image,
            mobile_angle=determine_mobile_angle(process, player_index),
            cart_facing_direction=process.read_cart_facing_direction(player_index)
        )
        draw_position(draw_parameters['target_position'], process, image)
        image[0, 0] = (0, 0, 255, 255)

        transparent_window.show_image(image)

        previous_parameters = parameters
        
        cv.waitKey(1)

    sys.exit(application.exec_())


def have_parameters_changed(parameters, previous_parameters):
    return parameters != previous_parameters


def create_image_with_size(width, height):
    image = np.full((height, width, 4), (0, 0, 0, 0), dtype=np.uint8)
    return image


def determine_mobile_angle(process, player_index):
    cart_angle = process.read_cart_angle(player_index)
    cart_facing_direction = process.read_cart_facing_direction(player_index)
    angle = cart_angle
    if cart_facing_direction == CartFacingDirection.Left:
        angle = (angle + 180) % 360
    return angle


if __name__ == '__main__':
    main()
