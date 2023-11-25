CC = g++ -std=c++14 -O3
CFLAGS = -g -Wall
LIBS = -I/usr/local/include/opencv4 -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs
######## Alternative if cmake with argument #########
#  	-D OPENCV_GENERATE_PKGCONFIG=ON
#
#OPENCV = `pkg-config opencv --cflags --libs`
#LIBS = $(OPENCV)
#
#####################################################
SRCS = main.cpp fc_m_resnet.cpp pinball_game.hpp convolution.cpp
PROG = dqn_pinball
$(PROG):$(SRCS)
	$(CC) $(CFLAGS) -o $(PROG) $(SRCS) $(LIBS)


