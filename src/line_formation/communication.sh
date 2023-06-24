#! /bin/bash

gnome-terminal -- bash -c "cd ~/XTDrone/communication; python multirotor_communication.py iris 0; exec bash"


sleep 0.7s  

gnome-terminal -- bash -c "cd ~/XTDrone/communication; python multirotor_communication.py iris 1; exec bash"

sleep 0.7s

gnome-terminal -- bash -c "cd ~/XTDrone/communication; python multirotor_communication.py iris 2; exec bash"



