#!/usr/bin/python3

import os, subprocess, fcntl, termios, re
from time import sleep

# Check using sudo
if os.geteuid() != 0:
    exit("You need root permissions to do this")

print("Terminate all vim Start")
vim_exist = True
while (vim_exist):
    cmd = subprocess.Popen("ls -l /proc/$(pgrep -nx vim)/fd/0", stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, encoding="utf-8")

    cmd.wait(2)
    ret = cmd.poll()

    if ret == 0:
        pts_path = re.search(r"\/dev\/pts\/\d+$", cmd.stdout.read().strip()).group()
        print("Terminating " + pts_path)
        fd = os.open(pts_path, os.O_RDWR)
        cmd="\033:wqa!\n"
        for i in range(len(cmd)):
            fcntl.ioctl(fd, termios.TIOCSTI, cmd[i])
        fcntl.ioctl(fd, termios.TIOCSTI, '\n')
        os.close(fd)
        sleep(0.5)
    else:
        vim_exist = False

print("Terminate all vim End")

print("Sync Start")
subprocess.call("sync -f /home/robert", shell=True)
subprocess.call("sync -d /home/robert", shell=True)
print("Sync End")
print("Force Reboot!!!!")
subprocess.call("echo b | sudo tee /proc/sysrq-trigger", shell=True)
