#pragma once

#include "ofMain.h"
#include "ofVideoGrabber.h"
#include "ofxThreadedImageLoader.h"
#include "ofxGui.h"
#include "ofxOscParameterSync.h"
#include "ofxImageSequenceRecorder.h"

#include "cnpy.h"

class ofApp : public ofBaseApp{

    public:
        void setup();
        void update();
        void draw();

        void keyPressed(int key);
        void keyReleased(int key);
        void mouseMoved(int x, int y );
        void mouseDragged(int x, int y, int button);
        void mousePressed(int x, int y, int button);
        void mouseReleased(int x, int y, int button);
        void windowResized(int w, int h);
        void dragEvent(ofDragInfo dragInfo);
        void gotMessage(ofMessage msg);
        void initRandom(int& mode);

        void initWeights();
        void loadWeights();
        void randWeights(int layer, int filtsize, ofTexture &dest);
        void npyWeights(int layer, int filtsize, string prefix, ofTexture & dest);
        void randBiases(int layer, ofTexture &dest);
        void npyBiases(int layer, string prefix, ofTexture & dest);

        ofParameterGroup params;
        ofParameter<float> fb;
        ofParameter<float> tblur;
        ofParameter<float> sblur;
        ofParameter<float> warp;
        ofParameter<float> perm;
        ofParameter<int> bound;
        ofParameter<float> zoom;
        ofParameter<float> framerate;
        ofParameter<int> record;
        ofParameter<int> initmode;

        ofxOscParameterSync sync;

        ofxPanel gui;

        ofVideoGrabber camera;
        ofxImageSequenceRecorder recorder;
        ofxThreadedImageLoader loader;
        ofImage modulating_img, camera_img;
        ofPixels newstate;

        vector<ofTexture> encode_weight_tex, encode_bias_tex;
        vector<ofTexture> decode_weight_tex, decode_bias_tex;
        string npy_dir;
        int encode_filtsize, decode_filtsize;
        int layers;
        int activation_mode, pool_mode, border_mode;
        int capture;

        int weight_fmt;

        ofFbo input_fbo;
        vector<ofFbo> encoded_fbos;
        vector<ofFbo> decoded_fbos;
        vector<ofFbo> feedback_fbos;
        int frame;
        ofShader inference_shader, decoding_shader, feedback_shader;

        int movieWidth, movieHeight;
        int disp_mode, disp_layer, camera_mode;

};
