#include <stdio.h>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv/cv.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include "people_identifier/Update.h"
#include "people_identifier/Query.h"

class PeopleTrackingFilter
{
    ros::NodeHandle nh_;
    ros::NodeHandle nh_private_;
    cv::KalmanFilter KF_;
    cv::Mat_<float> measurement_;
    ros::ServiceServer updateservice_;
    ros::ServiceServer queryservice_;
    float rr_;

public:
    PeopleTrackingFilter(float refresh_rate_hz)
    {
        rr_ = refresh_rate_hz;
        ros::NodeHandle public_handle();
        nh_ = public_handle;
        ros::NodeHandle private_handle("~");
        nh_private_ = private_handle;
        cv::KalmanFilter KF(6, 3, 0);
        KF_ = KF;
        cv::Mat_<float> measurement(3,1);
        measurement.setTo(cv::Scalar(0));
        measurement_ = measurement;
        KF_.transitionMatrix = *(cv::Mat_<float>(6, 6) <<
                                 1,0,0,1,0,0,
                                 0,1,0,0,1,0,
                                 0,0,1,0,0,1,
                                 0,0,0,1,0,0,
                                 0,0,0,0,1,0,
                                 0,0,0,0,0,1);
        cv::setIdentity(KF_.measurementMatrix);
        KF_.statePre.at<float>(0) = 0;
        KF_.statePre.at<float>(1) = 0;
        KF_.statePre.at<float>(2) = 0;
        KF_.statePre.at<float>(3) = 0;
        KF_.statePre.at<float>(4) = 0;
        KF_.statePre.at<float>(5) = 0;

        cv::setIdentity(KF_.processNoiseCov, cv::Scalar::all(1e-4));
        cv::setIdentity(KF_.measurementNoiseCov, cv::Scalar::all(1e-1));
        cv::setIdentity(KF_.errorCovPost, cv::Scalar::all(100));
        updateservice_ = nh_.advertiseService("update_tracking_filter", UpdateSrv, this);
        queryservice_ = nh_.advertiseService("query_tracking_filter", QuerySrv, this);
    }

    ~PeopleTrackingFilter()
    {
    }

    bool UpdateSrv(people_identifier::Update::Request &req, people_identifier::Update::Response &res)
    {
        /** \brief Updates the tracking info */
        res.status = req.data;
        return true;
    }

    bool QuerySrv(people_identifier::Query::Request &req, people_identifier::Query::Response &res)
    {
        /** \brief Query the output of the filter and return it in the service response */
        res.status = req.data;
        return true;
    }

    void Start(void)
    {
        /** \brief Calls the filter update function at a constant rate, spins otherwise */
        ros::Rate spin_rate(rr_);
        while (ros::ok())
        {
            UpdateFilter();
            ros::spinOnce();
            spin_rate.sleep();
        }
    }

    void UpdateFilter(void)
    {
        /** \brief Do whatever you need to do to update the filter */
        ;
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "people_tracking_filter");
    PeopleTrackingFilter tracking_filter(10.0);
    tracking_filter.Start();
    return 0;
}

