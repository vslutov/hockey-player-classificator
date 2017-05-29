/*
 * ----------------------------------------------------------------------------
 * "THE BEER-WARE LICENSE" (Revision 42):
 * <vslutov@yandex.ru> wrote this file.   As long as you retain this notice you
 * can do whatever you want with this stuff. If we meet some day, and you think
 * this stuff is worth it, you can buy me a beer in return.      Vladimir Lutov
 * ----------------------------------------------------------------------------
 */

#ifndef VIDEOREADER_H
#define VIDEOREADER_H

#include <string>
#include <opencv2/opencv.hpp>

namespace hockey {
    class VideoReader {

    private:
        static const ssize_t FRAME_COUNT = 60 * 20;

        std::string path;
        ssize_t cam_num;
        ssize_t sec = 0;
        cv::VideoCapture capture[2];
        ssize_t frame_number = 0;

        std::string
        videoName(ssize_t field);
    public:
        VideoReader(const std::string &in_path, ssize_t in_cam_num);

        ssize_t
        next_frame(cv::Mat &frame);

        bool
        set_frame(ssize_t in_frame_number);
    };
}

#endif // VIDEOREADER_H
