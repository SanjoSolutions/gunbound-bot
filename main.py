import sys
from enum import IntEnum
from math import cos, sin, sqrt, radians
from time import sleep

import numpy as np
from PyQt5.QtWidgets import QApplication
from mss import mss
from pymem import Pymem
from win32gui import GetClientRect, ClientToScreen, FindWindow, GetCursorPos
import cv2 as cv

from transparent_window import TransparentWindow


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

    def read_angle(self):
        b = self.process.read_uint(0x8F4A00 + 0x20)
        address = b + 0x19EC
        cart_angle = self.process.read_int(address)

        cart_angle2 = self.read_cart_angle()

        cart_facing_direction = self.read_cart_facing_direction()

        a = cart_angle
        if cart_facing_direction == CartFacingDirection.Left:
            a += cart_angle2
        else:
            a -= cart_angle2
        a %= 360

        print(cart_angle, cart_angle2, a)

        return a

    def read_cart_angle(self):
        cart_angle = self.process.read_int(self.process.base_address + 0x482A60)
        return cart_angle

    def read_cart_facing_direction(self):
        cart_facing_direction = CartFacingDirection(
            self.process.read_bytes(self.process.base_address + 0x482A64, 1)[0]
        )
        return cart_facing_direction

    def read_cart_position(self):
        # # a = self.process.read_uint(0x8701CC)
        # # b = self.process.read_uint(a + 0x77138)
        # c = self.process.read_uint(b + 0x4)
        # d = self.process.read_uint(c + 0x1c)
        # e = self.process.read_uint(d + 0x10)
        # cart_position = self.process.read_uint(e + 0x56C)
        # return cart_position
        x = self.process.read_ushort(self.process.base_address + 0x482A58)
        y = self.process.read_ushort(self.process.base_address + 0x482A5C)
        # cart_facing_direction = self.read_cart_facing_direction()
        # cart_angle = self.read_cart_angle()
        # if cart_facing_direction == CartFacingDirection.Left:
        #     cart_angle = (cart_angle + 180) % 360
        # offset_x = 0  # 15
        # offset_y = 0 # -100 # -15
        # x_offset_angle = radians(cart_angle)
        # angle_delta = -90 if cart_facing_direction == CartFacingDirection.Left else 90
        # y_offset_angle = radians(cart_angle + angle_delta)
        # print(cart_angle, cart_angle + angle_delta)
        # return (
        #     int(round(
        #         x +
        #         offset_x * cos(x_offset_angle) +
        #         offset_y * cos(y_offset_angle)
        #     )),
        #     int(round(
        #         y +
        #         # -50 +
        #         offset_x * sin(x_offset_angle) +
        #         offset_y * sin(y_offset_angle)
        #     )),
        # )
        return (x, y)

    def read_screen_center_position(self):
        return (
            self.process.read_ushort(self.process.base_address + 0x4E98D4),
            self.process.read_ushort(self.process.base_address + 0x4E98D8),
        )


SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600


def calculate_angle_2_4_power(process: GunboundProcess, window):
    target_position = determine_target_position(process, window)
    cart_position = process.read_cart_position()
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


class Mobile:
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
    AdukaKnight = 13
    JFrogDragon = 14
    Kalsiddon = 15
    Slots = 16
    Aid = 17


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


def determine_power(process: GunboundProcess, window):
    mobile = Mobile.Turtle
    start_position = process.read_cart_position()
    x, y = start_position
    angle = determine_angle(process, window)
    print('Angle: ' + str(angle))
    wind_power = process.read_wind_speed()
    wind_angle = process.read_wind_direction()
    x2, y2 = determine_target_position(process, window)
    # x2 = 728.0
    # y2 = 999.0
    direction = Direction.Left if x > x2 else Direction.Right
    backshot = False
    if backshot:
        direction = change_direction(direction)

    old_distance = 9999
    hit_power = None
    STEP_SIZE = 0.05

    delta_y2 = MAP_HEIGHT - y2

    for power in range(400 + 1):
        last_x = None
        last_y = None

        for position in generate_coordinates(
            mobile,
            start_position,
            direction,
            backshot,
            angle,
            power,
            wind_angle,
            wind_power,
            STEP_SIZE
        ):
            xxx, delta_yyy = position
            if (
                last_x is not None and last_y is not None and (
                    (last_y <= delta_y2 and delta_yyy >= delta_y2) or
                    (last_y >= delta_y2 and delta_yyy <= delta_y2)
                )
            ):
                x22 = int(round((last_x + xxx) / 2))
                # print(last_y, delta_yyy, delta_y2, last_x, xxx, x22)
                distance = abs(x2 - x22)

                if distance < old_distance:
                    old_distance = distance
                    hit_power = power
                    break

            last_x = xxx
            last_y = delta_yyy

    print('old_distance: ', old_distance)

    return hit_power


def generate_coordinates(mobile, start_position, direction, backshot, angle, power, wind_power, wind_angle, step_size):
    if backshot:
        direction = change_direction(direction)

    # compute horizontal/wind
    x_v2 = int(cos(wind_angle * DEGTORAD) * wind_power) * g_table2[mobile]

    # compute downward/gravity
    y_v2 = int(sin(wind_angle * DEGTORAD) * wind_power) * g_table2[mobile] - g_table1[mobile]

    if mobile == Mobile.Nak and backshot and angle <= 70:
        y_v2 *= -8.0
        direction = change_direction(direction)

    x_v = cos(angle * DEGTORAD)
    y_v = sin(angle * DEGTORAD)

    temp_x_v = x_v * power
    temp_y_v = y_v * power

    xxx = start_position[0]
    delta_yyy = MAX_MAP_Y - start_position[1]

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


def draw_position(position, process, image):
    scx, scy = process.read_screen_center_position()
    visible_map_area = (
        scx - SCREEN_WIDTH // 2,
        scy - SCREEN_HEIGHT // 2,
        SCREEN_WIDTH,
        SCREEN_HEIGHT
    )
    min_x = visible_map_area[0]
    min_y = visible_map_area[1]
    max_x = min_x + visible_map_area[2] - 1
    max_y = min_y + visible_map_area[3] - 1

    x = position[0] - min_x
    y = position[1] - min_y

    cv.circle(image, (x, y), 18, (0, 0, 255, 255), thickness=1)
    if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
        image[y, x] = (0, 0, 255, 255)


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
    max_x = min_x + visible_map_area[2] - 1
    max_y = min_y + visible_map_area[3] - 1

    STEP_SIZE = 0.05

    for position in generate_coordinates(
        mobile,
        start_position,
        direction,
        backshot,
        angle,
        power,
        wind_angle,
        wind_power,
        STEP_SIZE
    ):
        x = position[0]
        y = -(position[1] - MAX_MAP_Y)
        if min_x <= x <= max_x and min_y <= y <= max_y:
            image_x = int(round(x - min_x))
            image_y = int(round(y - min_y))
            image[image_y, image_x] = (0, 0, 255, 255)


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
    print(wind_direction, factor)
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


def determine_angle(process: GunboundProcess, window):
    angle = read_angle(window)
    facing_direction = process.read_cart_facing_direction()
    if facing_direction == CartFacingDirection.Left:
        angle = 90 + (90 - angle)
    elif facing_direction == CartFacingDirection.Right:
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
    cv.imwrite('sign_image.png', sign_image)
    digit_1_image = make_screenshot_cv(window, (
        226,
        531,
        13,
        13
    ))
    cv.imwrite('digit_1_image.png', digit_1_image)
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
    cv.imwrite('digit_2_image.png', digit_2_image)
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



def main():
    window = FindWindow('Softnyx', None)
    process = GunboundProcess(Pymem('GunBound.gme'))

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

    while True:
        client_area_rect = determine_client_area_rect(window)
        transparent_window.setGeometry(
            client_area_rect['left'],
            client_area_rect['top'],
            client_area_rect['width'],
            client_area_rect['height']
        )

        # angle = read_angle(window)
        # from_position = gunbound.read_cart_position()
        # to_position = determine_target_position(gunbound, window)
        # power = calculate_power(angle, from_position, to_position)
        power = determine_power(process, window)
        print('Power: ' + str(power))
        image = np.full((client_area_rect['height'], client_area_rect['width'], 4), (0, 0, 0, 0), dtype=np.uint8)
        image[0, 0] = (0, 0, 255, 255)
        start_position = process.read_cart_position()
        target_position = determine_target_position(process, window)
        draw_position(start_position, process, image)
        draw_position(target_position, process, image)
        if power is not None:
            mark_on_power_bar(image, power)

        mobile = Mobile.Turtle
        x, y = start_position
        angle = determine_angle(process, window)
        wind_power = process.read_wind_speed()
        wind_angle = process.read_wind_direction()
        x2, y2 = target_position
        direction = Direction.Left if x > x2 else Direction.Right
        backshot = False
        draw_shot_line(
            process,
            image,
            mobile,
            start_position,
            direction,
            backshot,
            angle,
            400,
            wind_angle,
            wind_power
        )

        transparent_window.show_image(image)
        cv.waitKey(1000)

    sys.exit(application.exec_())


if __name__ == '__main__':
    main()
