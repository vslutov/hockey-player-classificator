QT += core
QT -= gui

TARGET = Hockey
CONFIG += console
CONFIG -= app_bundle

INCLUDEPATH += C:/opencv_build/install/include
LIBS += -L"C:/opencv_build/install/x86/mingw/bin/" \
    -lopencv_calib3d310 \
    -lopencv_core310 \
    -lopencv_features2d310 \
    -lopencv_flann310 \
    -lopencv_highgui310 \
    -lopencv_imgcodecs310 \
    -lopencv_imgproc310 \
    -lopencv_ml310 \
    -lopencv_objdetect310 \
    -lopencv_photo310 \
    -lopencv_shape310 \
    -lopencv_stitching310 \
    -lopencv_superres310 \
    -lopencv_video310 \
    -lopencv_videoio310 \
    -lopencv_videostab310

SOURCES += main.cpp \
    videoreader.cpp
OTHER_FILES +=

TEMPLATE = app

HEADERS += \
    main.h \
    videoreader.h

QMAKE_CXXFLAGS += -O2 -Wall -Werror -Wformat-security -Wignored-qualifiers -Winit-self -Wswitch-default -Wshadow -Wpointer-arith -Wtype-limits -Wempty-body -Wlogical-op -Wmissing-field-initializers -Wctor-dtor-privacy  -Wnon-virtual-dtor -Wstrict-null-sentinel -Woverloaded-virtual -Wsign-promo -std=gnu++14
