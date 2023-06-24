kill -9 $(ps -ef|grep multirotor_communication.py|gawk '$0 !~/grep/ {print $2}' |tr -s '\n' ' ')

