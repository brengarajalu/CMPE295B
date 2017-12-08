#include <jni.h>
#include <string>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <malloc.h>
#include <android/log.h>
#include <list>
#include <vector>
#include <fstream>
#include  <stdlib.h>
#include <iostream>
#include <jni.h>
#include <string.h>
#include <stdio.h>
#include <jni.h>
#include <cstdio>
#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
#include "opencv2/core/utility.hpp"


using namespace std;
using namespace cv;
using namespace cv::ml;


struct Sensor_Reading
{
    int sensor_id;
    char value;
};
struct SensorReadings
{
    double timestamp;
    Sensor_Reading reading;
};


#define LOG_TAG "TRAIN"
#define PARA_LEN 12 // Max length for each parameter



static bool
read_num_class_data( const string& filename, int var_count,
                     Mat* _data, Mat* _responses )
{
    const int M = 1024;
    char buf[M+2];

    Mat el_ptr(1, var_count, CV_32F);
    int i;
    vector<int> responses;

    _data->release();
    _responses->release();

    FILE* f = fopen( filename.c_str(), "rt" );
    if( !f )
    {
        cout << "Could not read the database " << filename << endl;
        return false;
    }
    for(;;)
    {
        char* ptr;
        if( !fgets( buf, M, f ) || !strchr( buf, ',' ) )
            break;
        responses.push_back((int)buf[0]);
        ptr = buf+2;
        for( i = 0; i < var_count; i++ )
        {
            int n = 0;
            sscanf( ptr, "%f%n", &el_ptr.at<float>(i), &n );
            ptr += n + 1;
        }
        if( i < var_count )
            break;
        _data->push_back(el_ptr);
    }
    fclose(f);
    Mat(responses).copyTo(*_responses);

    cout << "The database " << filename << " is loaded.\n";

    return true;
}

template<typename T>
static Ptr<T> load_classifier(const string& filename_to_load)
{
    // load classifier from the specified file
    Ptr<T> model = StatModel::load<T>( filename_to_load+"/test.yml" );
    if( model.empty() )
        cout << "Could not read the classifier " << filename_to_load << endl;
    else
        cout << "The classifier " << filename_to_load << " is loaded.\n";

    return model;
}

static void test_and_save_classifier(const Ptr<StatModel>& model,
                                     const Mat& data, const Mat& responses,
                                     int ntrain_samples, int rdelta,
                                     const string& filename_to_save)
{
    int i, nsamples_all = data.rows;
    double train_hr = 0, test_hr = 0;

    // compute prediction error on train and test_1 data
    for( i = 0; i < nsamples_all; i++ )
    {
        Mat sample = data.row(i);

        float r = model->predict( sample );
        r = std::abs(r + rdelta - responses.at<int>(i)) <= FLT_EPSILON ? 1.f : 0.f;

        if( i < ntrain_samples )
            train_hr += r;
        else
            test_hr += r;
    }

    test_hr /= nsamples_all - ntrain_samples;
    train_hr = ntrain_samples > 0 ? train_hr/ntrain_samples : 1.;

    printf( "Recognition rate: train = %.1f%%, test_1 = %.1f%%\n",
            train_hr*100., test_hr*100. );

    if( !filename_to_save.empty() )
    {
        model->save( filename_to_save );
    }
}


static Ptr<TrainData>
prepare_train_data(const Mat& data, const Mat& responses, int ntrain_samples)
{
    Mat sample_idx = Mat::zeros( 1, data.rows, CV_8U );
    Mat train_samples = sample_idx.colRange(0, ntrain_samples);
    train_samples.setTo(Scalar::all(1));

    int nvars = data.cols;
    Mat var_type( nvars + 1, 1, CV_8U );
    var_type.setTo(Scalar::all(VAR_ORDERED));
    var_type.at<uchar>(nvars) = VAR_CATEGORICAL;

    return TrainData::create(data, ROW_SAMPLE, responses,
                             noArray(), sample_idx, noArray(), var_type);
}

static bool
build_rtrees_classifier( const string& data_filename,
                         const string& filename_to_save,
                         const string& filename_to_load )
{
    Mat data;
    Mat responses;
    bool ok = read_num_class_data( data_filename, 16, &data, &responses );
    if( !ok )
        return ok;

    Ptr<RTrees> model;

    int nsamples_all = data.rows;
    int ntrain_samples = (int)(nsamples_all*0.8);

    // Create or load Random Trees classifier
    if( !filename_to_load.empty() )
    {
        model = load_classifier<RTrees>(filename_to_load);
        if( model.empty() )
            return false;
        ntrain_samples = 0;
    }
    else
    {
        // create classifier by using <data> and <responses>
        cout << "Training the classifier ...\n";
//        Params( int maxDepth, int minSampleCount,
//                   double regressionAccuracy, bool useSurrogates,
//                   int maxCategories, const Mat& priors,
//                   bool calcVarImportance, int nactiveVars,
//                   TermCriteria termCrit );
        Ptr<TrainData> tdata = prepare_train_data(data, responses, ntrain_samples);
        model = RTrees::create();
        model->setMaxDepth(20);
       /* model->setMinSampleCount(10);
        model->setRegressionAccuracy(0);
        model->setUseSurrogates(false);
        model->setMaxCategories(15);
        model->setPriors(Mat());
        model->setCalculateVarImportance(true);
        model->setActiveVarCount(4); */
        //Smodel->setTermCriteria(TC(100,0.01f));
        model->train(tdata);
        cout << endl;
    }

    test_and_save_classifier(model, data, responses, ntrain_samples, 0, filename_to_save);
    cout << "Number of trees: " << model->getRoots().size() << endl;

    // Print variable importance
    Mat var_importance = model->getVarImportance();
    if( !var_importance.empty() )
    {
        double rt_imp_sum = sum( var_importance )[0];
        printf("var#\timportance (in %%):\n");
        int i, n = (int)var_importance.total();
        for( i = 0; i < n; i++ )
            printf( "%-2d\t%-4.1f\n", i, 100.f*var_importance.at<float>(i)/rt_imp_sum);
    }

    return true;
}



void ReadSensorReadings(char* fileName, char* outString)
{
    FILE *fp = fopen(fileName, "r");
    std::list<SensorReadings> readingList;
    try {
        if (fp != NULL) {
            size_t i = 0;
            char c;
            while ((c = fgetc(fp)) != EOF) {
                outString[i++] = c;
                Sensor_Reading reading = {1,c};
                SensorReadings readings = {1,reading};
                readingList.push_back(readings);

            }
            outString[i] = '\0';
            fclose(fp);

        }
    }
    catch (std::exception ex)
    {
        fclose(fp);
        throw ex;
    }

}


int PredictModel(char* dataFile, const string& modelFilePath)
{
    Ptr<RTrees> rtrees  = load_classifier<RTrees>(modelFilePath);
    if( rtrees.empty() )
        return false;
    /* rtrees->setMaxDepth(10);
     rtrees->setMinSampleCount(2);
     rtrees->setRegressionAccuracy(0);
     rtrees->setUseSurrogates(false);
     rtrees->setMaxCategories(16);
     rtrees->setPriors(Mat());
     rtrees->setCalculateVarImportance(true);
     rtrees->setActiveVarCount(0);
     rtrees->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 0));
     rtrees->load("src/main/opt/test_1.ymlml"); */
    Mat data;
    Mat responses;
    bool ok = read_num_class_data(dataFile, 16, &data, &responses );
    int i, nsamples_all = data.rows;
    for( i = 0; i < nsamples_all; i++ )
    {
        Mat sample = data.row(i);

        float r = rtrees->predict( sample );
        float value = r;
        printf("value=%d",value);

    }

}

extern "C"
JNIEXPORT jstring JNICALL

Java_edu_sjsu_vimmi_1swami_ddd_1project_MainActivity_ProcessSensorData(
        JNIEnv *pEnv, jobject obj, jobject assetManager, jstring path
)
{
    //char result[100];
    char fName[64];
    const char *modelFilePath = pEnv->GetStringUTFChars(path, 0);
    strcpy(fName,modelFilePath);
    strcat(fName,"/letterrecognition.txt");
    FILE* fp;
    try {
        fp= fopen(fName, "r");
        fflush(fp);
        fclose(fp);
    }
    catch(std::exception ex){
        throw ex;
    }
    PredictModel(fName,modelFilePath);



}



