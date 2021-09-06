from pymem import Pymem

from main import GunboundProcess


def main():
    process = GunboundProcess(Pymem('GunBound.gme'))
    wind_direction = 0  # in degrees
    wind_speed = 2
    while True:
        process.set_wind_direction(wind_direction)
        process.set_wind_speed(wind_speed)


if __name__ == '__main__':
    main()
