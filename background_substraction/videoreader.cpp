#include "videoreader.h"

//C++
#include <sstream>
#include <stdexcept>

using namespace cv;
using namespace std;
using namespace hockey;

hockey::VideoReader::VideoReader(const std::string &in_path, ssize_t in_cam_num) :
    path(in_path),
    cam_num(in_cam_num)
{
    for (ssize_t field = 0; field < 2; ++ field) {
        this->capture[field].open(this->videoName(field));
        if(!this->capture[field].isOpened()){
            //error in opening the video input
            throw runtime_error("Unable to open video file: " + this->videoName(field));
        }
    }

    this->sec = 2;
}

ssize_t
hockey::VideoReader::next_frame(Mat &frame)
{
    bool field = this->frame_number % 2;
    ssize_t result = this->capture[field].read(frame);

    if (!result) {
        this->capture[field].open(this->videoName(field));
        if (this->capture[field].isOpened()) {
            result = this->capture[field].read(frame);
            this->sec += 1;
        }
    }

    result = result ? this->frame_number : -1;
    this->frame_number += 1;
    return result;
}

string
hockey::VideoReader::videoName(ssize_t field)
{
    stringstream converter;
    converter << this->path << "\\cam_" << this->cam_num << "_field_" << field << "_sec_" << (this->sec / 2) << ".avi";
    string video_name;
    converter >> video_name;
    return video_name;
}

bool
hockey::VideoReader::set_frame(ssize_t in_frame_number)
{
    ssize_t in_sec = in_frame_number / this->FRAME_COUNT;
    ssize_t set_frame_number = in_sec * this->FRAME_COUNT;

    if (this->sec != in_sec + 2 || this->frame_number > in_frame_number) {
        this->frame_number = set_frame_number;
        this->sec = in_sec;
        for (ssize_t field = 0; field < 2; ++ field) {
            this->capture[field].open(this->videoName(field));
        }
        this->sec = in_sec + 2;
    }

    while (this->frame_number != in_frame_number) {
        Mat empty;
        if (this->next_frame(empty) == -1) {
            return false;
        }
    }

    return true;
}
