#include "perf_precomp.hpp"

using namespace std;
using namespace cv;
using namespace perf;
using std::tr1::make_tuple;
using std::tr1::get;

#define AKAZE_IMAGES \
    "cv/detectors_descriptors_evaluation/images_datasets/leuven/img1.png",\
    "stitching/a3.png"

typedef perf::TestBaseWithParam<std::string> akaze;

PERF_TEST_P(akaze, detect, testing::Values(AKAZE_IMAGES))
{
    string filename = getDataPath(GetParam());
    Mat frame = imread(filename, IMREAD_GRAYSCALE);

    if (frame.empty())
        FAIL() << "Unable to load source image " << filename;

    declare.in(frame);
    AKAZE alg;
    vector<KeyPoint> points;

    TEST_CYCLE() alg.detect(frame, points);

    sort(points.begin(), points.end(), comparators::KeypointGreater());
    SANITY_CHECK_KEYPOINTS(points);
}

PERF_TEST_P(akaze, extract, testing::Values(AKAZE_IMAGES))
{
    string filename = getDataPath(GetParam());
    Mat frame = imread(filename, IMREAD_GRAYSCALE);

    if (frame.empty())
        FAIL() << "Unable to load source image " << filename;

    Mat mask;
    declare.in(frame);

    AKAZE alg(DESCRIPTOR_MLDB);
    vector<KeyPoint> points;
    alg.detect(frame, points);
    sort(points.begin(), points.end(), comparators::KeypointGreater());

    Mat descriptors;

    TEST_CYCLE() alg.compute(frame, points, descriptors);

    SANITY_CHECK(descriptors);
}

PERF_TEST_P(akaze, full, testing::Values(AKAZE_IMAGES))
{
    string filename = getDataPath(GetParam());
    Mat frame = imread(filename, IMREAD_GRAYSCALE);

    if (frame.empty())
        FAIL() << "Unable to load source image " << filename;

    Mat mask;
    declare.in(frame);
    AKAZE alg(DESCRIPTOR_MLDB);

    vector<KeyPoint> points;
    Mat descriptors;

    TEST_CYCLE() alg(frame, mask, points, descriptors, false);

    perf::sort(points, descriptors);
    SANITY_CHECK_KEYPOINTS(points);
    SANITY_CHECK(descriptors);
}
