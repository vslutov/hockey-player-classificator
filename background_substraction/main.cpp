/*
 * ----------------------------------------------------------------------------
 * "THE BEER-WARE LICENSE" (Revision 42):
 * <vslutov@yandex.ru> wrote this file.   As long as you retain this notice you
 * can do whatever you want with this stuff. If we meet some day, and you think
 * this stuff is worth it, you can buy me a beer in return.      Vladimir Lutov
 * ----------------------------------------------------------------------------
 */

#include "main.h"
#include "videoreader.h"

//C++
#include <iostream>
#include <sstream>
#include <stdexcept>

using namespace cv;
using namespace std;
using namespace hockey;

Point point1, point2; /* vertical points of the bounding box */
int drag = 0;
Rect rect; /* bounding box */
Mat roiImg; /* roiImg - the part of the image in the bounding box */
int select_flag = 0;

void hockey::help()
{
    cout
    << "Usage:"                  << endl
    << "  ./bs <video filename>" << endl
    << endl;
}

int
main(int argc, char** argv) {
    cout.sync_with_stdio(false);

    //check for the input parameter correctness
    if(argc != 2) {
        help();
        throw runtime_error("Incorret input list");
    }

    //create GUI windows
    namedWindow("Frame");
    // namedWindow("FG Mask MOG 2");

    process_video(argv[1]);

    //destroy GUI windows
    destroyAllWindows();
    return EXIT_SUCCESS;
}

Ptr<BackgroundSubtractor>
hockey::prepare_model(const string &videoPath)
{
    const ssize_t HISTORY = 10;

    // create Background Subtractor objects
    Ptr<BackgroundSubtractor> pMOG2 = createBackgroundSubtractorMOG2(HISTORY); //MOG2 approach

    // create the reader object
    VideoReader reader(videoPath, 0);

    // create morphology element
    ssize_t erosion_size = 1;
    Mat kernel = getStructuringElement(MORPH_ELLIPSE,
                                       Size(2 * erosion_size + 1, 2 * erosion_size+1),
                                       Point(erosion_size, erosion_size));

    Mat background_mask = imread(videoPath + "\\background_mask.png", CV_LOAD_IMAGE_GRAYSCALE);
    string model_file = videoPath + "\\background_model.yml.gz";

    if (true) { // need_learning
        cout << "Learning" << endl;
        for (ssize_t i = 0; i < HISTORY; ++ i) {
            Mat mask;
            reader.set_frame(i * 10);
            if (reader.next_frame(frame) == -1) {
                break;
            }
            pMOG2->apply(frame, mask);
            cout << i << " / " << HISTORY << endl;
        }
        auto storage = FileStorage(model_file, FileStorage::Mode::WRITE);
        pMOG2->write(storage);
        cout << "Finish learning" << endl;
    } else {
        cout << "Load model" << endl;
        pMOG2->read(FileStorage(model_file, FileStorage::Mode::READ).getFirstTopLevelNode());
        reader.set_frame(0);
    }

    return pMOG2;
}

void
hockey::process_video(const string &videoPath) {
    const ssize_t HISTORY = 500;

    // create Background Subtractor objects
    Ptr<BackgroundSubtractor> pMOG2 = createBackgroundSubtractorMOG2(HISTORY); //MOG2 approach

    // create the reader object
    VideoReader reader(videoPath, 0);

    // create morphology element
    ssize_t erosion_size = 1;
    Mat kernel = getStructuringElement(MORPH_ELLIPSE,
                                       Size(2 * erosion_size + 1, 2 * erosion_size+1),
                                       Point(erosion_size, erosion_size));

    Mat background_mask = imread(videoPath + "\\background_mask.png", CV_LOAD_IMAGE_GRAYSCALE);
    string model_file = videoPath + "\\background_model.yml.gz";

    if (true) { // need_learning
        cout << "Learning" << endl;
        for (ssize_t i = 0; i < HISTORY; ++ i) {
            Mat mask;
            reader.set_frame(i * 10);
            if (reader.next_frame(frame) == -1) {
                break;
            }
            pMOG2->apply(frame, mask);
            cout << i << " / " << HISTORY << endl;
        }
        auto storage = FileStorage(model_file, FileStorage::Mode::WRITE);
        pMOG2->write(storage);
        cout << "Finish learning" << endl;
    } else {
        cout << "Load model" << endl;
        pMOG2->read(FileStorage(model_file, FileStorage::Mode::READ).getFirstTopLevelNode());
        reader.set_frame(0);
    }

    //read input data. ESC or 'q' for quitting
    while (static_cast<char>(keyboard) != 'q' && static_cast<char>(keyboard) != 27){

        //read the current frame
        ssize_t frame_number = reader.next_frame(frame);
        cout << frame_number << endl;
        if(frame_number == -1) {
            throw runtime_error("Unable to read next frame.");
        }
        //cvtColor(frame, frame, CV_BGR2Lab);

        //update the background model
        pMOG2->apply(frame, fgMaskMOG2);
        threshold(fgMaskMOG2, fgMaskMOG2, 200, 255, THRESH_BINARY);
        //morphologyEx(fgMaskMOG2,fgMaskMOG2, MORPH_OPEN, kernel);

        fgMaskMOG2.copyTo(fgMaskMOG2, background_mask);
        Mat labels = fgMaskMOG2;
        //if (frame_number > 20) {
            // label(fgMaskMOG2, labels);
            // colorize(labels, labels);
            // applyMask(frame, frame, fgMaskMOG2);
        //}

        //show the current frame and the fg masks
        //imshow("Frame", labels);
        // imshow("FG Mask MOG 2", fgMaskMOG2);
        //get the input from the keyboard
        keyboard = waitKey(30);
        imshow("Frame", labels);

        if (true) { // (static_cast<char>(keyboard) == 's' || static_cast<char>(keyboard) == 's') {
            stringstream convert;
            string frame_number_str;

            convert << frame_number;
            convert >> frame_number_str;

            imwrite(videoPath + "\\mask_" + frame_number_str + ".png", labels);
            imwrite(videoPath + "\\frame_" + frame_number_str + ".png", frame);
        }
    }
}

void
hockey::applyMask(const Mat &src, Mat &dst, const Mat &mask)
{
    CV_Assert(src.depth() == CV_8U);
    CV_Assert(src.channels() == 3);
    CV_Assert(src.dims == 2);
    CV_Assert(mask.depth() == CV_8U);
    CV_Assert(mask.channels() == 1);
    CV_Assert(mask.dims == 2);

    dst = src.clone();
    Mat_<Vec3b> _dst = dst;

    for (ssize_t i = 0; i < frame.rows; ++ i) {
        for (ssize_t j = 0; j < frame.rows; ++ j) {
            if (!mask.at<uchar>(i, j)) {
                _dst(i, j)[0] = 0;
                _dst(i, j)[1] = 0;
                _dst(i, j)[2] = 0;
            }
        }
    }
}

ssize_t
hockey::dfs(uchar *p, ssize_t rows, ssize_t cols, ssize_t x, ssize_t y, uchar in, uchar out)
{
    // cout << x << " " << y << " " << (int)in << " " << (int)out << endl;

    if (x >= 0 && x < rows && y >= 0 && y < cols && p[x * cols + y] == in) {
        p[x * cols + y] = out;
        return 1 +
                dfs(p, rows, cols, x + 1, y, in, out) +
                dfs(p, rows, cols, x, y + 1, in, out) +
                dfs(p, rows, cols, x - 1, y, in, out) +
                dfs(p, rows, cols, x, y - 1, in, out);
    } else {
        return 0;
    }
}

void
hockey::label(const Mat &src, Mat &dst)
{
    CV_Assert(src.depth() == CV_8U);
    CV_Assert(src.channels() == 1);
    CV_Assert(src.dims == 2);
    CV_Assert(src.isContinuous());

    Mat binar(1, 256, CV_8U);
    uchar* p = binar.data;
    p[0] = 0;
    for (ssize_t i = 1; i < 256; ++i) {
        p[i] = 1;
    }
    LUT(src, binar, dst);

    uchar c = 1;
    p = dst.ptr<uchar>(0);

    for (ssize_t i = 0; i < dst.rows; ++ i) {
        for (ssize_t j = 0; j < dst.cols; ++ j) {
            if (dst.at<uchar>(i, j) == 1) {
                c += 1;
                if (dfs(p, src.rows, src.cols, i, j, 1, c) < 20) {
                    dfs(p, src.rows, src.cols, i, j, c, 0);
                    c -= 1;
                }
            }
        }
    }
}

void
hockey::colorize(const Mat &src, Mat &dst)
{
    CV_Assert(src.depth() == CV_8U);
    CV_Assert(src.channels() == 1);
    CV_Assert(src.dims == 2);
    CV_Assert(src.isContinuous());

    ssize_t size = src.rows * src.cols;
    const uchar *in = src.ptr<uchar>(0);

    Mat tmp;
    tmp.create(src.rows, src.cols, CV_8UC3);
    CV_Assert(tmp.isContinuous());

    uchar *out = tmp.ptr<uchar>(0);

    char t[256][3];
    for (ssize_t i = 0; i < 256; ++ i) {
        t[i][0] = i * 30;
        t[i][1] = i * 80;
        t[i][2] = i * 150;
    }

    for (ssize_t i = 0; i < size; ++ i) {
        out[3 * i] = t[in[i]][0];
        out[3 * i + 1] = t[in[i]][1];
        out[3 * i + 2] = t[in[i]][2];
    }

    dst = tmp;
}
