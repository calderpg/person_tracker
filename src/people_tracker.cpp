#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv/cv.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <resource_retriever/retriever.h>
#include <image_geometry/pinhole_camera_model.h>
#include "person_tracker/TrackerState.h"


#ifdef HAVE_CVCONFIG_H
#include <cvconfig.h>
#endif
#ifdef HAVE_TBB
#include "tbb/task_scheduler_init.h"
#endif

#define CONF_THRESHOLD 0.4

typedef struct person
{
    CvPoint3D64f      *measured;
    CvPoint           *imgloc;
    CvRect            *bounding_box;
    cv::KalmanFilter  *KF;
    cv::MatND         *hist;
    bool              fresh;
    int               id;
}
person;

namespace enc = sensor_msgs::image_encodings;

static const char COLORWINDOW[] = "People Detector - color";
static const char DEPTHWINDOW[] = "People Detector - depth";
double x;
double y;
double z;
bool iscalibrated;
int imgscale = 3;



class PeopleDetector
{
    ros::NodeHandle nh_;
    ros::NodeHandle nh_private_;
    CvLatentSvmDetector* detector;
    image_transport::ImageTransport it_;
    image_transport::Subscriber image_sub_;
    image_transport::Subscriber depth_sub_;

    //ros::Subscriber points_sub_;
    ros::Publisher tracking_pub_;
    std::vector<cv::MatND> currHists;
    std::vector<cv::MatND> preHists;
    std::vector<person> tracked;

    image_geometry::PinholeCameraModel model_;
    IplImage depthimage;
    cv_bridge::CvImagePtr depth_ptr;
    std::string tracker_frame_;

public:

    PeopleDetector(ros::NodeHandle& nh, std::string& model_file, std::string& rgb_camera_topic, std::string& depth_camera_topic, std::string& tracker_frame) : nh_(nh), it_(nh)
    {
        tracker_frame_ = tracker_frame;
        resource_retriever::Retriever retver;
        resource_retriever::MemoryResource resource;
        try
        {
            resource = retver.get("package://people_identifier/data/" + model_file);
        }
        catch (resource_retriever::Exception& e)
        {
            printf("Resource retriever failed to find the file!\n");
        }
        FILE * tempfile = fopen("tmp_model.xml", "w");
        fwrite(resource.data.get(), resource.size, 1, tempfile);
        fclose(tempfile);
        detector = cvLoadLatentSvmDetector("tmp_model.xml");
        if (!detector)
        {
            printf("Unable to load the model file?!\n");
            fflush(stdout);
            exit(1);
        }

        iscalibrated = 0;
        tracking_pub_ = nh_.advertise<person_tracker::TrackerState>("tracker_state", 100);
        image_sub_ = it_.subscribe(rgb_camera_topic, 1, &PeopleDetector::imageCb, this);
        depth_sub_ = it_.subscribe(depth_camera_topic, 1, &PeopleDetector::depthCb, this);
        cv::namedWindow(COLORWINDOW);
        ROS_INFO("Startup complete");
    }

    ~PeopleDetector()
    {
        remove("tmp_model.xml");
        cv::destroyWindow(COLORWINDOW);
        //cv::destroyWindow(DEPTHWINDOW);
        cvReleaseLatentSvmDetector(&detector);
    }

    void depthCb(const sensor_msgs::ImageConstPtr& msg)
    {
        /** Used with depth *images*, not pointclouds!*/
        //iscalibrated =1;

        cv_bridge::CvImagePtr cv_ptr;
        try
        {
            cv_ptr = cv_bridge::toCvCopy(msg);
        }
        catch (cv_bridge::Exception& e)
        {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }
        depth_ptr = cv_bridge::CvImagePtr(cv_ptr);
        depthimage = cv_ptr->image;
        //cvShowImage(DEPTHWINDOW, &depthimage);
        cvWaitKey(3);
    }

    void imageCb(const sensor_msgs::ImageConstPtr& msg)
    {
        cv_bridge::CvImagePtr cv_ptr;
        try
        {
            cv_ptr = cv_bridge::toCvCopy(msg, enc::BGR8);
        }
        catch (cv_bridge::Exception& e)
        {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }
        //Do the fancy opencv magic
        IplImage ipled = cv_ptr->image;
        IplImage * downscaled = cvCreateImage(cvSize(640/imgscale,480/imgscale),ipled.depth, ipled.nChannels);
        cvResize(&ipled, downscaled, CV_INTER_LINEAR);
        detect_and_draw(downscaled,ipled);
    }

    IplImage* crop( IplImage* src,  CvRect roi)
    {
        // Must have dimensions of output image
        IplImage* cropped = cvCreateImage( cvSize(roi.width,roi.height), src->depth, src->nChannels );

        // Say what the source region is
        cvSetImageROI( src, roi );

        // Do the copy
        cvCopy( src, cropped );
        cvResetImageROI( src );
        cvNamedWindow( "body", 1 );
        cvShowImage( "body", cropped );


        return cropped;
    }

    cv::KalmanFilter * initKF()
    {
        cv::KalmanFilter* kf = new cv::KalmanFilter(6,6,0);

        cv::KalmanFilter KF = *kf;

        cv::Mat_<float> measurement(3,1);
        measurement.setTo(cv::Scalar(0));

        KF.transitionMatrix = *(cv::Mat_<float>(6, 6) <<
                                1,0,0,1,0,0,
                                0,1,0,0,1,0,
                                0,0,1,0,0,1,
                                0,0,0,1,0,0,
                                0,0,0,0,1,0,
                                0,0,0,0,0,1);
        cv::setIdentity(KF.measurementMatrix);
        KF.statePre.at<float>(0) = 0;
        KF.statePre.at<float>(1) = 0;
        KF.statePre.at<float>(2) = 0;
        KF.statePre.at<float>(3) = 0;
        KF.statePre.at<float>(4) = 0;
        KF.statePre.at<float>(5) = 0;

        cv::setIdentity(KF.processNoiseCov, cv::Scalar::all(.1));
        cv::setIdentity(KF.measurementNoiseCov, cv::Scalar::all(0.05));
        cv::setIdentity(KF.errorCovPost, cv::Scalar::all(20));
        return kf;
    }

    void drawHist(cv::MatND& hist,int hbins = 40,int sbins = 40,char* name = (char*)"H-S Histogram")
    {
        double maxVal=0;
        minMaxLoc(hist, 0, &maxVal, 0, 0);

        int scale = 10;
        IplImage* imgHist = cvCreateImage(cvSize(sbins*scale/2, hbins*10/2), 8,1);
        cvZero(imgHist);

        for( int h = 0; h < hbins; h++ )
        {
            for( int s = 0; s < sbins; s++ )
            {
                float binVal = hist.at<float>(h, s);
                int intensity = cvRound(3*binVal*255/maxVal);
                cvRectangle( imgHist, cv::Point(h*scale, s*scale),
                             cv::Point( (h+1)*scale - 1, (s+1)*scale - 1),
                             cv::Scalar::all(intensity),
                             CV_FILLED );
            }
        }
        cv::namedWindow(name, 1);
        cv::Mat histpic(imgHist);
        cv::imshow( name, histpic );
    }

    void computehist(IplImage* image, CvRect roi,cv::MatND* hist,int hbins = 40,int sbins = 40)
    {
        IplImage* temp = crop( image, roi);

        //cv::namedWindow( "bitch face", 1 );
        //cv::Mat test(image);
        //cv::imshow("bitch face",test);

        cv::Mat src(temp);


        cv::Mat hsv;
        cvtColor(src, hsv, CV_BGR2HSV);

        // let's quantize the hue to 30 levels
        // and the saturation to 32 levels

        int currHistsize[] = {hbins, sbins};
        // hue varies from 0 to 179, see cvtColor
        float hranges[] = { 0, 180 };
        // saturation varies from 0 (black-gray-white) to
        // 255 (pure spectrum color)
        float sranges[] = { 0, 256 };
        const float* ranges[] = { hranges, sranges };

        //cv::MatND localhist;
        // we compute the histogram from the 0-th and 1-st channels
        int channels[] = {0, 1};

        calcHist( &hsv, 1, channels, cv::Mat(), // do not use mask
                  *hist, 2, currHistsize, ranges,
                  true, // the histogram is uniform
                  false );

        //drawHist(*hist);
    }

    void detect_and_draw(IplImage * image, IplImage& fullimage)
    {
        int numThreads = 8;
        CvMemStorage* storage = cvCreateMemStorage(0);
        CvSeq* detections = 0;
        int64 start = 0, finish = 0;
#ifdef HAVE_TBB
        tbb::task_scheduler_init init(tbb::task_scheduler_init::deferred);
        if (numThreads > 0)
        {
            init.initialize(numThreads);
            printf("Number of threads %i\n", numThreads);
        }
        else
        {
            printf("Number of threads is not correct for TBB version");<<std::endl;
            return;
        }
#endif
        start = cvGetTickCount();
        detections = cvLatentSvmDetectObjects(image, this->detector, storage, 0.3f, numThreads);
        finish = cvGetTickCount();
        printf("++++++++++\ndetection time = %.3f\n", (float)(finish - start) / (float)(cvGetTickFrequency() * 1000000.0));
#ifdef HAVE_TBB
        init.terminate();
#endif
        preHists=currHists;
        currHists.clear();

        for(int i = 0; i<tracked.size(); i++)
        {
            tracked[i].fresh = false;
        }

        if(depth_ptr!=NULL)
        {
            IplImage * downscaleddepth = cvCreateImage(cvSize(640/imgscale,480/imgscale),depthimage.depth, depthimage.nChannels);
            cvResize(&depthimage,downscaleddepth, CV_INTER_LINEAR);

            std::cout << "Checking for detections" << std::endl;
            for(int i = 0; i < detections->total; i++ )
            {
                //Get the detection
                CvObjectDetection detection = *(CvObjectDetection*)cvGetSeqElem( detections, i );
                CvRect bounding_box = detection.rect;
                //Check if the detection is good enough
                if (detection.score>0)
                {
                    cvRectangle( image, cvPoint(bounding_box.x, bounding_box.y),
                                 cvPoint(bounding_box.x + bounding_box.width,
                                         bounding_box.y + bounding_box.height),
                                 CV_RGB(255,0,0), 3 );
                    cvRectangle(downscaleddepth , cvPoint(bounding_box.x, bounding_box.y),
                                 cvPoint(bounding_box.x + bounding_box.width,
                                         bounding_box.y + bounding_box.height),
                                 CV_RGB(255,0,0), 3 );
                    int locx = (bounding_box.x + (bounding_box.width / 2))*imgscale;
                    int locy = (bounding_box.y + (bounding_box.height / 2))*imgscale;
                    printf("Computed [locx,locy] [%d,%d]\n", locx, locy);

                    //grab core of person
                    bounding_box.x = (bounding_box.x + bounding_box.width/4)*imgscale;
                    bounding_box.y = (bounding_box.y + bounding_box.height/4)*imgscale;
                    bounding_box.width=bounding_box.width*imgscale/2;
                    bounding_box.height=bounding_box.height*imgscale/2;


                    int hbins = 40, sbins = 32;
                    cv::MatND* histob = new cv::MatND();
                    cv::MatND *hist = histob;

                    computehist( &fullimage, bounding_box,hist,hbins,sbins);
                    //drawHist(*hist,hbins,sbins);

                    // check if object in hist is being tracked
                    double mindis = 1;
                    int minindex =-1;
                    for(unsigned int j=0;j<tracked.size();j++)
                    {
                        if(tracked[j].fresh == false)
                        {
                            //printf("Comparing hist pointers: %d|%d\n", hist, tracked[j].hist);
                            drawHist(*hist,30,32,(char*)"hist");
                            drawHist(*(tracked[j].hist),30,32,(char*)"tracked");
                            double dis = cv::compareHist(*hist,*( tracked[j].hist ),CV_COMP_BHATTACHARYYA);
                            std::cout << "tracked distance: " << dis <<std::endl;

                            if (mindis>dis)
                            {
                                minindex = j;
                                mindis=dis;
                            }
                        }
                    }
                    if (mindis<0.4) //being tracked
                    {
                        delete tracked[minindex].hist;
                        delete tracked[minindex].imgloc;
                        tracked[minindex].hist = hist;
                        std::cout << "found in tracked: "<<minindex<<std::endl;
                        printf("probably %d is %d on [x,y] [%d,%d]\n", i,minindex, locx, locy);
                        tracked[minindex].bounding_box = &bounding_box;
                        //std::cout << "Updating existing track" << std::endl;
                        CvPoint* imgloc = new CvPoint(); // default construction of an object of type CvPoint
                        imgloc->x = locx;
                        imgloc->y = locy;
                        //CvPoint imgloc = cvPoint(locx,locy);
                        tracked[minindex].imgloc = imgloc;
                        tracked[minindex].fresh = true;
                    }
                    else
                    {
                        //std::cout << "not found in tracked"<<mindis<<std::endl;
                        // check if object in hist is noise or not
                        mindis = 1;

                        for(unsigned int j=0; j<preHists.size(); j++)
                        {
                            double dis = cv::compareHist(*hist,preHists[j],CV_COMP_BHATTACHARYYA);
                            //drawHist(preHists[j]);
                            std::cout << "filter distance: " << dis <<std::endl;
                            if (dis < mindis)
                            {
                                mindis = dis;
                            }
                        }
                        if (mindis<0.35)
                        {
                            std::cout << "Adding a new track" << std::endl;
                            person currperson;
                            CvPoint* imgloc = new CvPoint(); // default construction of an object of type CvPoint
                            imgloc->x = locx;
                            imgloc->y = locy;
                            currperson.fresh = true;
                            currperson.imgloc = imgloc;
                            currperson.bounding_box=&bounding_box;
                            currperson.hist=hist;
                            //cv::KalmanFilter KF = initKF();
                            currperson.KF =  initKF() ;
                            tracked.push_back(currperson);
                            //drawHist(*currperson.hist,hbins,sbins);
                        }
                        else
                        {
                            std::cout << "Adding a hist to storage" << std::endl;
                            currHists.push_back(*hist);
                        }
                    }

                }
            }

            //-----------------------------------------------------
            std::cout << "tracking: " << tracked.size()<<std::endl;

            float constant_x=1/525.0;
            float constant_y=1/525.0;
            float centerX = 319.5;
            float centerY = 239.5;

            if(iscalibrated)
            {
                float constant_x=1/model_.fx();
                float constant_y=1/model_.fx();
                float centerX = model_.cx();
                float centerY = model_.cy();
                ROS_INFO(" cx: %f cy: %f constant_x %f constant_y: %f",model_.cx(),model_.cy(),constant_x,constant_y);
            }
            //Make the update message
            person_tracker::TrackerState update_msg;
            update_msg.TrackerName = "opencv_latentsvm_tracker";
            update_msg.TrackerType = person_tracker::TrackerState::POSEONLY;
            update_msg.header.stamp = ros::Time::now();
            update_msg.header.frame_id = tracker_frame_;
            int valid_tracks = 0;
            printf("Making update message...\n");
            for(int i=0; i < tracked.size(); i++)
            {
                //drawHist(*tracked[i].hist);
                cv::Mat prediction = tracked[i].KF->predict();
                float x = prediction.at<float>(0);
                float y = prediction.at<float>(1);
                float z = prediction.at<float>(2);

                CvPoint3D64f measured=cvPoint3D64f(x,y,z);
                tracked[i].measured=&measured;

                bool fresh = tracked[i].fresh;
                bool safe = false;
                int rawX = tracked[i].imgloc->x;
                int rawY = tracked[i].imgloc->y;
                if (fresh)
                {
                    if (rawX < 640 && rawX >= 0 && rawY < 480 && rawY >= 0 )
                    {
                        safe = true;
                    }
                    else
                    {
                        std::cout << "fresh" <<std::endl<< fresh <<std::endl;
                        printf("Failed tracking %d on [x,y] [%d,%d]\n", i, rawX, rawY);
                        std::cout << "Prediction: " << prediction << std::endl;
                        printf("***** UNSAFE ***** UNSAFE *****\n");
                    }
                }
                if(fresh && safe)
                {
                    float rawZ = depth_ptr->image.at<float>(rawY,rawX);
                    float tx = (rawX-centerX)*rawZ *constant_x;
                    float ty = (rawY-centerY)*rawZ *constant_y;
                    float tz = rawZ;
                    if (!isnan(tx) && !isnan(ty) && !isnan(tz))
                    {
                        cv::Mat_<float> measurement(6,1);
                        measurement(0) = tx;
                        measurement(1) = ty;
                        measurement(2) = tz;
                        measurement(3) = tx-tracked[i].measured->x;
                        measurement(4) = ty-tracked[i].measured->y;
                        measurement(5) = tz-tracked[i].measured->z;
                        person tm = tracked[i];
                        cv::setIdentity(tm.KF->measurementNoiseCov, cv::Scalar::all(0.05));
                        cv::Mat estimated = tm.KF->correct(measurement);
                        tx = estimated.at<float>(0);
                        ty = estimated.at<float>(1);
                        tz = estimated.at<float>(2);
                        CvPoint3D64f measured=cvPoint3D64f(tx,ty,tz);
                        tracked[i].measured= &measured;
                        std::cout <<"measurement" << measurement << std::endl;
                        //ROS_INFO("Updating kalman filter with new data");
                    }
                    else
                    {
                        ROS_WARN("Tried to update, but got NAN instead");
                    }
                }
                else
                {
                    cv::Mat_<float> measurement(6,1);
                    measurement(0) = tracked[i].measured->x;
                    measurement(1) = tracked[i].measured->y;
                    measurement(2) = tracked[i].measured->z;
                    measurement(3) = 0;
                    measurement(4) = 0;
                    measurement(5) = 0;
                    person tm = tracked[i];
                    cv::setIdentity(tm.KF->measurementNoiseCov, cv::Scalar::all(50.));
                    cv::Mat estimated = tm.KF->correct(measurement);
                    float tx = estimated.at<float>(0);
                    float ty = estimated.at<float>(1);
                    float tz = estimated.at<float>(2);
                    CvPoint3D64f measured=cvPoint3D64f(tx,ty,tz);
                    tracked[i].measured= &measured;
                }
                //detections
                //cv::Mat prediction = tracked.at(i).KF->predict();
                std::cout << i << "'s best estimate: " << tracked[i].measured->x << std::string(", ") << tracked[i].measured->y << std::string(", ") << tracked[i].measured->z <<std::endl;
                //std::cout << i << "'s best cov: " << tracked[i].KF->error_cov_pre <<  <<std::endl;
                float conf =(4-pow(pow(tracked[i].KF->errorCovPost.at<float>(0,0),2)+
                                   pow(tracked[i].KF->errorCovPost.at<float>(0,0),2)+pow(tracked[i].KF->errorCovPost.at<float>(0,0),2),.5))/4;

                //Add the current track to the update message
                if (conf > CONF_THRESHOLD)
                {
                    ;
                }
                std::cout << i << "'s best estimate (confident): " << tracked[i].measured->x << std::string(", ") << tracked[i].measured->y << std::string(", ") << tracked[i].measured->z <<std::endl;
                // Make tracking update
                person_tracker::TrackedPerson new_track;
                //Set tracking pose
                geometry_msgs::Pose tracking_pose;
                tracking_pose.position.x = (1.0) * (double)(tracked[i].measured->x);// / 2.0;
                tracking_pose.position.y = (1.0) * (double)(tracked[i].measured->y);// / 2.0;
                tracking_pose.position.z = (1.0) * (double)(tracked[i].measured->z);// / 2.0;
                tracking_pose.orientation.x = 0.0;
                tracking_pose.orientation.y = 0.0;
                tracking_pose.orientation.z = 0.0;
                tracking_pose.orientation.w = 1.0;
                new_track.Pose = tracking_pose;
                //Set message
                new_track.Confidence = conf;
                new_track.UID = i;
                new_track.Name = std::string("target_") + boost::lexical_cast<std::string>(i);
                new_track.header = update_msg.header;
                update_msg.Tracks.push_back(new_track);
                valid_tracks++;
                std::cout << i << "'s best estimate pose (confident): " << tracking_pose.position.x << std::string(", ") << tracking_pose.position.y << std::string(", ") << tracking_pose.position.z <<std::endl;
            }
            //Send the update message
            update_msg.ActiveTracks = valid_tracks;
            update_msg.TotalTracks = tracked.size();
            tracking_pub_.publish(update_msg);
            cvShowImage(COLORWINDOW, downscaleddepth);
            //cvShowImage(COLORWINDOW, image);
            cvWaitKey(3);
            cvReleaseMemStorage( &storage );
        }
    }

};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "opencv_tracker");
    ROS_INFO("Starting opencv_tracker...");
    ros::NodeHandle nh;
    ros::NodeHandle nhp("~");
    std::string rgb_camera_topic;
    std::string depth_camera_topic;
    std::string tracker_frame;
    std::string model_file;
    nhp.param(std::string("rgb_camera_topic"), rgb_camera_topic, std::string("camera/rgb/image"));
    nhp.param(std::string("depth_camera_topic"), depth_camera_topic, std::string("camera/depth/image"));
    nhp.param(std::string("tracker_frame"), tracker_frame, std::string("camera_rgb_optical_frame"));
    nhp.param(std::string("model_file"), model_file, std::string("people.xml"));
    PeopleDetector pd = PeopleDetector(nh, model_file, rgb_camera_topic, depth_camera_topic, tracker_frame);
    ros::spin();
    return 0;
}
