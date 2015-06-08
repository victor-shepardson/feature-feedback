#include "ofApp.h"
#include "math.h"

//--------------------------------------------------------------
void ofApp::setup(){

    params.setName("params");
	params.add(fb.set("fb",1.333));
	params.add(tblur.set("tblur",.5));
	params.add(sblur.set("sblur",.5));
	//params.add(warp.set("warp",.3));
	//params.add(perm.set("perm",-1));
    //params.add(bound.set("bound", 2));
    //params.add(zoom.set("zoom", 0));
    params.add(framerate.set("framerate", 60));
    params.add(record.set("record", 0));
    params.add(initmode.set("initmode", 0));

    initmode.addListener(this, &ofApp::initRandom);

    recorder.setFormat("jpg");
    recorder.setPrefix("capture/");
    recorder.startThread(true);

	gui.setup(params);
	sync.setup((ofParameterGroup&)gui.getParameter(),6666,"localhost",6667);

    bool fs = false;
    movieWidth=movieHeight=400;
    ofSetFullscreen(fs);

    ofEnableDataPath();
    ofSetWindowShape(movieWidth, movieHeight);//640, 480);//2*movieWidth, 2*movieHeight);
    ofEnableArbTex();

    weight_fmt = GL_RGB32F;

    inference_shader.load(ofToDataPath("shader/inference"));
    decoding_shader.load(ofToDataPath("shader/decoder"));
    feedback_shader.load(ofToDataPath("shader/feedbackh"));

    //camera.listDevices();
    camera.setDeviceID(1);
    camera.initGrabber(640, 480);

    capture = 0;

    frame = 0;
    disp_mode = 0;
    disp_layer = 0;
    activation_mode = 0; //tanh/relu
    camera_mode = 0;
    pool_mode = 0; //max/mean/stride/maxmag
    border_mode = 1; //clamp/wrap

    encode_filtsize = 3;//5;//3;
    decode_filtsize = 2;//6;//4;
    layers = 3;//2;
    npy_dir = "npy-1-meanpool/";

    input_fbo.allocate(movieWidth, movieHeight, GL_RGB);
    input_fbo.begin();
    ofBackground(0);
    input_fbo.end();
    for(int i=0; i<layers; i++){
        ofFbo f;
        f.allocate(movieWidth, movieHeight, GL_RGB);
        f.begin();
        ofBackground(0);
        f.end();
        encoded_fbos.push_back(f);
    }
    for(int i=0; i<layers; i++){
        ofFbo f;
        f.allocate(movieWidth, movieHeight, GL_RGB);
        f.begin();
        ofBackground(0);
        f.end();
        decoded_fbos.push_back(f);
    }
    for(int i=0; i<layers+1; i++){
        ofFbo f;
        f.allocate(movieWidth, movieHeight, GL_RGB);
        f.begin();
        ofBackground(0);
        f.end();
        feedback_fbos.push_back(f);
    }

    initWeights();//loadWeights();

    printf("setup complete\n");
}

void ofApp::loadWeights(){
    cout << "load weights from .npy" << endl;
    encode_weight_tex.clear();
    for(int l=0; l<layers; l++){
        ofTexture t;
        npyWeights(l, encode_filtsize, "encode", t);
        encode_weight_tex.push_back(t);
    }
    decode_weight_tex.clear();
    for(int l=0; l<layers; l++){
        ofTexture t;
        npyWeights(l, decode_filtsize, "decode", t);
        decode_weight_tex.push_back(t);
    }
    encode_bias_tex.clear();
    for(int l=0; l<layers; l++){
        ofTexture t;
        npyBiases(l, "encode", t);
        encode_bias_tex.push_back(t);
    }
    decode_bias_tex.clear();
    for(int l=0; l<layers; l++){
        ofTexture t;
        npyBiases(l, "decode", t);
        decode_bias_tex.push_back(t);
    }
}
//assuming layer names are indexed from 1
void ofApp::npyWeights(int l, int filtsize, string prefix, ofTexture &t){
    stringstream ss;
    ss << prefix << l+1 << "-filters.npy";
    string fname = ofToDataPath(npy_dir+ss.str());
    cnpy::NpyArray arr = cnpy::npy_load(fname);
    float* weights = reinterpret_cast<float*>(arr.data);
    for(int i=0; i<arr.shape.size(); i++) cout << arr.shape[i] << endl;
    int shortside = filtsize*pow(2,2*l+1);
    int longside = 3*shortside;
    t.allocate(longside, shortside, weight_fmt);
    t.loadData(weights, longside, shortside, GL_RGB);

    /*cout << "[ ";
    for(int i=0; i<shortside*longside*3; i++)
        cout<< weights[i] << " ";
    cout << "]" << endl;
*/
    delete[] weights;
}
void ofApp::npyBiases(int l, string prefix, ofTexture &t){
    stringstream ss;
    ss << prefix << l+1 << "-biases.npy";
    string fname = ofToDataPath(npy_dir+ss.str());
    cnpy::NpyArray arr = cnpy::npy_load(fname);
    float* biases = reinterpret_cast<float*>(arr.data);
    int shortside = 1;
    int longside = arr.shape[0]/3;
    t.allocate(longside, shortside, weight_fmt);
    t.loadData(biases, longside, shortside, GL_RGB);
/*
    cout << "[ ";
    for(int i=0; i<longside*3; i++)
        cout<< biases[i] << " ";
    cout << "]" << endl;
*/
    delete[] biases;
}

void ofApp::initWeights(){
    cout << "init random weights" << endl;
    encode_weight_tex.clear();
    for(int l=0; l<layers; l++){
        ofTexture t;
        randWeights(l, encode_filtsize, t);
        encode_weight_tex.push_back(t);
    }
    decode_weight_tex.clear();
    for(int l=0; l<layers; l++){
        ofTexture t;
        randWeights(l, decode_filtsize, t);
        decode_weight_tex.push_back(t);
    }
    encode_bias_tex.clear();
    for(int l=0; l<layers; l++){
        ofTexture t;
        randBiases(l, t);
        encode_bias_tex.push_back(t);
    }
    decode_bias_tex.clear();
    for(int l=0; l<layers; l++){
        ofTexture t;
        randBiases(l, t);
        decode_bias_tex.push_back(t);
    }
}

void ofApp::randWeights(int l, int filtsize, ofTexture &t){
    //whole texture should be C*4C*F*F/3 where F is filter size, C is channels at this layer
    //C = 3*4^l so we have 3*(4^l)*4*3*(4^l)*F*F/3
    //make the horizontal dimension 3x the vertical:
    int shortside = filtsize*pow(2,2*l+1);
    int longside = 3*shortside;
    t.allocate(longside, shortside, weight_fmt);
    float *weights = (float*)malloc(sizeof(float)*longside*shortside*3);
    for(int i=0; i<longside*shortside*3; i++){
        weights[i] = pow(ofRandom(-1,1),5);//.5+.5*tanh(ofRandom(-2, 2));
    }
    t.loadData(weights, longside, shortside, GL_RGB);
    free(weights);
}
void ofApp::randBiases(int l, ofTexture &t){
    //just storing biases as RGB triples in horizontal dimension for now
    int shortside = 1;
    int longside = pow(4, l+1);
    t.allocate(longside, shortside, weight_fmt);
    float *biases = (float*)malloc(sizeof(float)*longside*shortside*3);
    for(int i=0; i<longside*shortside*3; i++){
        biases[i] = pow(ofRandom(-1,1),5);//.5+.5*tanh(ofRandom(-2, 2));
    }
    t.loadData(biases, longside, shortside, GL_RGB);
    free(biases);
}

//--------------------------------------------------------------
void ofApp::update(){
    sync.update();
    ofSetFrameRate(framerate.get());
    ofSetWindowTitle(ofToString(ofGetFrameRate()));

    if(camera_mode) camera.update();
}

//--------------------------------------------------------------
void ofApp::draw(){

    ofBackground(0);

    //input
    input_fbo.begin();
    if(camera_mode) camera.draw(0,0, movieWidth, movieHeight);
    input_fbo.end();

    //feature extraction
    for(int idx = 0; idx<encoded_fbos.size(); idx++){
        //subsample in time for high layers
        if((frame>>(idx-1))%2) break;
        encoded_fbos[idx].begin();
        inference_shader.begin();
        if(camera_mode)
            inference_shader.setUniformTexture("state", input_fbo,0);
        else
            inference_shader.setUniformTexture("state", feedback_fbos[idx], 0);
        inference_shader.setUniformTexture("weights", encode_weight_tex[idx], 1);
        inference_shader.setUniformTexture("biases", encode_bias_tex[idx], 2);
        inference_shader.setUniform1i("scale", idx);
        inference_shader.setUniform2i("size", movieWidth, movieHeight);
        inference_shader.setUniform1i("filtsize", encode_filtsize);
        inference_shader.setUniform1i("squash_weights", 0);
        inference_shader.setUniform1i("activation_mode", activation_mode);
        inference_shader.setUniform1i("border_mode", border_mode);
        inference_shader.setUniform1i("pool_mode", pool_mode);
        ofRect(0, 0, movieWidth, movieHeight);
        inference_shader.end();
        encoded_fbos[idx].end();
    }

    //decoding
    for(int idx = 0; idx<decoded_fbos.size(); idx++){
        //subsample in time for high layers
        if(((frame+(1<<layers)+1)>>(idx-1))%2) break;
        decoded_fbos[idx].begin();
        decoding_shader.begin();
        if(camera_mode)
            decoding_shader.setUniformTexture("state", encoded_fbos[idx],0);
        else
            decoding_shader.setUniformTexture("state", feedback_fbos[idx+1], 0);
        decoding_shader.setUniformTexture("weights", decode_weight_tex[idx], 1);
        decoding_shader.setUniformTexture("biases", decode_bias_tex[idx], 2);
        decoding_shader.setUniform1i("scale", idx);
        decoding_shader.setUniform2i("size", movieWidth, movieHeight);
        decoding_shader.setUniform1i("filtsize", decode_filtsize);
        decoding_shader.setUniform1i("squash_weights", 0);
        decoding_shader.setUniform1i("activation_mode", activation_mode);
        decoding_shader.setUniform1i("border_mode", border_mode);
        ofRect(0, 0, movieWidth, movieHeight);
        decoding_shader.end();
        decoded_fbos[idx].end();
    }

    //feedback
    for(int idx = 0; idx<feedback_fbos.size(); idx++){
        //subsample in time for high layers
        if(((frame+1)>>(idx-1))%2) break;
        feedback_fbos[idx].begin();
        feedback_shader.begin();

        feedback_shader.setUniformTexture("state", feedback_fbos[idx], 0);

        if(idx==0) //feedback_shader.setUniformTexture("encoded", input_fbo, 1);
            feedback_shader.setUniformTexture("encoded", feedback_fbos[idx], 1);
        else feedback_shader.setUniformTexture("encoded", encoded_fbos[idx-1], 1);

        if(idx>= decoded_fbos.size()) feedback_shader.setUniformTexture("decoded", feedback_fbos[idx], 2);
        else feedback_shader.setUniformTexture("decoded", decoded_fbos[idx], 2);

        feedback_shader.setUniform1i("scale", idx);
        feedback_shader.setUniform2i("size", movieWidth, movieHeight);
        feedback_shader.setUniform1i("border_mode", border_mode);

        feedback_shader.setUniform1f("tblur", tblur);
        feedback_shader.setUniform1f("fb", fb);
        feedback_shader.setUniform1f("sblur", sblur);

        ofRect(0, 0, movieWidth, movieHeight);
        feedback_shader.end();
        feedback_fbos[idx].end();
    }

    //draw to screen
    ofFbo * to_draw;
    if(!disp_mode)
        to_draw = &feedback_fbos[disp_layer];
    else if (disp_layer<encoded_fbos.size()){
        if(disp_mode==1)
            to_draw = &encoded_fbos[disp_layer];
        if(disp_mode==2)
            to_draw = &decoded_fbos[disp_layer];
    }
    else
        to_draw = &input_fbo;
    to_draw->draw(0,0,ofGetWindowWidth(), ofGetWindowHeight());


    //set current state to new state by drawing to feedback_fbos[0]
    if(newstate.isAllocated()){
        for(int i=0; i<feedback_fbos.size(); i++){
            feedback_fbos[i].begin();
            ofImage img;
            img.allocate(movieWidth, movieHeight, OF_IMAGE_COLOR);
            img.setFromPixels(newstate);
            img.draw(0, 0);
            feedback_fbos[i].end();
        }
        newstate.clear();
    }
    if(record || capture){
        ofPixels pix;
        to_draw->getTextureReference().readToPixels(pix);
        recorder.addFrame(pix);
    //    img.saveImage(save_prefix+ofToString(frame, 5, '0')+save_suffix, OF_IMAGE_QUALITY_BEST);
        capture = 0;
    }

    frame++;

}

void ofApp::initRandom(int& mode){
    printf("init random %d\n", mode);
    newstate.allocate(movieWidth, movieHeight, OF_PIXELS_RGBA);
    for(int x=0; x<movieWidth; x++){
        for(int y=0; y<movieHeight; y++){
            ofColor c;
            if(mode==0)
                c = ofColor(ofRandom(255),ofRandom(255),ofRandom(255));
            else if (mode==1)
                c = ofColor(255*floor(ofRandom(2)), 255*floor(ofRandom(2)), 255*floor(ofRandom(2)));
            newstate.setColor(x, y, c);
        }
    }
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){
    if(key=='m'){
        string disp_mode_names[] = {"feedback", "encoded", "decoded"};
        disp_mode = ofWrap(disp_mode+1, 0, 3);
        cout << "display: " << disp_mode_names[disp_mode] << endl;
    }
    if(key=='l'){
        disp_layer = ofWrap(disp_layer+1, 0, feedback_fbos.size());
        cout << "layer: " << disp_layer << endl;
    }
    if(key=='a'){
        string activation_mode_names[] = {"tanh", "ReLU"};
        activation_mode = ofWrap(activation_mode+1, 0, 2);
        cout << "activation: " << activation_mode_names[activation_mode] << endl;
    }
    if(key=='c'){
        camera_mode = ofWrap(camera_mode+1, 0, 2);
        cout << "camera: " << camera_mode << endl;
    }
    if(key=='i'){
        int m=0;
        initRandom(m);
    }
    if(key=='w'){
        initWeights();
    }
    if(key=='n'){
        loadWeights();
    }
    if(key=='f'){
        capture=1;
    }
    if(key=='p'){
        string names[] = {"max", "mean", "stride", "max magnitude"};
        pool_mode = ofWrap(pool_mode+1, 0, 4);
        cout << "pool: " << names[pool_mode] << endl;
    }
}

//--------------------------------------------------------------
void ofApp::keyReleased(int key){

}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y ){

}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h){

}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg){

}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo){

}
