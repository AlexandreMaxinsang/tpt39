EXE=vector_avg
SRCS=vector_avg.cpp
GCC=arm-linux-gnueabihf-g++  
OCLLIBSDIR=/opt/ComputeLibrary/build/
OCLINCSDIR=/opt/ComputeLibrary/include/
MGD=/opt/Mali_Graphics_Debugger_v4.4.1.0271762a_Linux_x64/target/linux/hard_float/
FLAGS=-g -Wno-deprecated-declarations -Wall -DARCH_ARM -Wextra -Wno-unused-parameter -pedantic -Wdisabled-optimization -Wformat=2 -Winit-self -Wstrict-overflow=2 -Wswitch-default -fpermissive -std=gnu++11 -Wno-vla -Woverloaded-virtual -Wctor-dtor-privacy -Wsign-promo -Weffc++ -Wno-format-nonliteral -Wno-overlength-strings -Wno-strict-overflow -Wno-implicit-fallthrough -Wlogical-op -Wnoexcept -Wstrict-null-sentinel -march=armv7-a -mthumb -mfpu=neon -mfloat-abi=hard -Werror -O3 -ftree-vectorize -fstack-protector-strong -DARM_COMPUTE_CL -I${OCLINCSDIR} -I.. -I..  
#LDFLAGS=../../build/utils/Utils.o -L../../build/ -L..  -larm_compute -larm_compute_core -lOpenCL
LDFLAGS=-L${OCLLIBSDIR} -larm_compute -larm_compute_core -lOpenCL

all: ${EXE}
${EXE}.o:${SRCS}
	$(GCC) -c ${FLAGS} ${SRCS} -o ${EXE}.o

${EXE}:${EXE}.o
	${GCC} -o ${EXE} ${EXE}.o  ${LDFLAGS}

debug:${EXE}
	LD_PRELOAD=${MGD}/libinterceptor.so ./${EXE}

clean:
	rm -rf ${EXE} ${EXE}.o	
