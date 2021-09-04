from time import sleep

from pymem import Pymem

from main import GunboundProcess

process = GunboundProcess(Pymem('GunBound.gme'))
while True:
    current_power = process.read_current_power()
    print('Current power: ' + str(current_power))
    sleep(0.1)
