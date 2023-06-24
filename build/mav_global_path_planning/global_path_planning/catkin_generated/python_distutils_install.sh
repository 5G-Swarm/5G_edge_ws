#!/bin/sh

if [ -n "$DESTDIR" ] ; then
    case $DESTDIR in
        /*) # ok
            ;;
        *)
            /bin/echo "DESTDIR argument must be absolute... "
            /bin/echo "otherwise python's distutils will bork things."
            exit 1
    esac
fi

echo_and_run() { echo "+ $@" ; "$@" ; }

echo_and_run cd "/home/ubuntu/5G_edge_ws/src/mav_global_path_planning/global_path_planning"

# ensure that Python install destination exists
echo_and_run mkdir -p "$DESTDIR/home/ubuntu/5G_edge_ws/install/lib/python3/dist-packages"

# Note that PYTHONPATH is pulled from the environment to support installing
# into one location when some dependencies were installed in another
# location, #123.
echo_and_run /usr/bin/env \
    PYTHONPATH="/home/ubuntu/5G_edge_ws/install/lib/python3/dist-packages:/home/ubuntu/5G_edge_ws/build/lib/python3/dist-packages:$PYTHONPATH" \
    CATKIN_BINARY_DIR="/home/ubuntu/5G_edge_ws/build" \
    "/usr/bin/python3" \
    "/home/ubuntu/5G_edge_ws/src/mav_global_path_planning/global_path_planning/setup.py" \
     \
    build --build-base "/home/ubuntu/5G_edge_ws/build/mav_global_path_planning/global_path_planning" \
    install \
    --root="${DESTDIR-/}" \
    --install-layout=deb --prefix="/home/ubuntu/5G_edge_ws/install" --install-scripts="/home/ubuntu/5G_edge_ws/install/bin"
