#version 400

uniform sampler2DRect state;
uniform sampler2DRect weights;
uniform sampler2DRect biases;
uniform int scale; //decode to this scale (from scale+1)
uniform ivec2 size;
uniform int filtsize;
uniform int squash_weights;
uniform int activation_mode;
uniform int border_mode;

//const vec2 invsize = 1./size;

//pixel_center_integer vec4 gl_FragCoord;

//convert between bipolar [-1, 1] and unipolar [0, 1]
vec3 u2b(vec3 u){
	return 2.*u-1.;
}
vec3 b2u(vec3 b){
	return .5*b+.5;
}

//convert (x, y, feature triple) coordinates to texture coordinates at scale s
ivec2 xyf2tex(ivec3 p, int s){
	ivec2 t = p.xy;
	if(border_mode<=0)
		t = clamp(t, ivec2(0), (size>>s)-1);
	else if(border_mode<=1)
		t = ((size>>s) + t) % (size>>s); //wrap
	t = t<<s;
	t.x += p.z%(1<<s);
	t.y += p.z>>s;
	return t;
}
ivec3 tex2xyf(ivec2 t, int s){
	ivec2 f2 = t%(1<<s);
	int f = f2.x + (f2.y<<s);
	ivec2 p = t>>s;
	return ivec3(p, f);
}

//convert (dest channel, filter y, filter x, source channel triple) to texture coordinates at scale s
ivec2 w2tex(ivec4 p, int s){
	int cmult = 1<<(2*s);
	int idx = p.w + 4*cmult * (p.z + filtsize * (p.y + filtsize * p.x));
	int longside = 3*cmult*2*filtsize; //(longside x longsize/3 texture)
	return ivec2(idx%longside, idx/longside);
}
ivec2 b2tex(int c, int s){
	return ivec2(c, 0);
}

vec3 get_bias_triple(int c_triple, int s){
	ivec2 t = b2tex(c_triple, s);
	return texelFetch(biases, t).rgb;
}

//get featurea at position p.xy and triple p.z, given scale s
vec3 get_feature_triple(ivec3 p, int s){
	ivec2 pix = xyf2tex(p, s);
	vec3 ret = texelFetch(state, pix).rgb;
	//convert to bipolar
	if(activation_mode<1) return ret-.5;
	return ret;
}

//look up weight for destination channel, filter position, source channel/3,  (cd, fy, fx, cs)
//this should be the right order to interpret a caffe blob .transpose([0, 2, 3, 1]) as 2D RGBF32 texture
vec3 get_weight_triple(ivec4 p, int s){
	ivec2 idx = w2tex(p, s);
	vec3 ret = texelFetch(weights, idx).rgb;
	//hack to work with clamped textures if float textures unavailable
	if(squash_weights>0)
		return atanh(2*ret-1);
	return ret;
}

//perform deconvolution for a single output pixel
//texture size is the same for every layer; feature triples are stored as RGB in square patches
//therefore a given pixel in an upper layer corresponds to multiple pixels in a lower layer, but a subset of features
vec3 deconv_layer(){
	//get x,y,feature triple coords of this texture coord in decoded image
	ivec3 p = tex2xyf(ivec2(gl_FragCoord.xy), scale);
	//convert feature triple to feature
	int cd0 = p.z*3;
	//calculate number of source channel triples at current scale
	int cl = 1<<(scale*2+2);
	int fc = filtsize/2;
	//loop over destination channel, filter position, source triple
	vec3 acc = vec3(0);
	for(int cdi=0; cdi<3; cdi++){
		int cd = cd0+cdi;
		vec3 acc_inner = vec3(0);
		for(int fyi=0; fyi<filtsize; fyi++){
			int fy = fyi-fc;
			for(int fxi=0; fxi<filtsize; fxi++){
				int fx = fxi-fc;
				//for each position in the filter, calculate source position
				ivec2 source = (ivec2(gl_FragCoord.xy)+ivec2(fx, fy))>>(scale+1);
				//then loop over all source features
				for(int cs = 0; cs<cl; cs++){
					//get the weights for this destination, filter pos, source triple
					vec3 w = get_weight_triple(ivec4(cd, fyi, fxi, cs), scale);
					//get the features for this source triple, filter pos, image pos
					vec3 f = get_feature_triple(ivec3(source, cs), scale+1);
					//multiply features*weights, accumulate
					acc_inner += f*w;
				}
			}
		}
		//sum over source triples, accumulate to current dest
		//the below line should work according to spec but doesn't...
		//acc[cdi] += acc_inner.x + acc_inner.y + acc_inner.z;
		//so here's a gross switch statement instead
		float sum = acc_inner.x + acc_inner.y + acc_inner.z;
		switch(cdi){
			case 0: acc[0] += sum; break;
			case 1: acc[1] += sum; break;
			case 2: acc[2] += sum;
		}
	}

	vec3 bias = get_bias_triple(p.z, scale);
	acc += bias;

	//apply activation function
	if(activation_mode<1)
		return tanh(acc);
	return max(vec3(0), acc); 
}

void main() {

	vec3 ret = deconv_layer();
	

//debug weights
	ivec2 filtidxs = ivec2(4<<scale,3<<scale);
/*	ivec2 cell = ivec2(gl_FragCoord.xy/size*filtidxs);
	ivec2 fp = ivec2(gl_FragCoord.xy*filtsize/size*filtidxs)%filtsize;
	int cd = cell.x+cell.y*filtidxs.x;
	ret = vec3(0);
	ret.x = 4*get_weight_triple(ivec4(0, fp.yx, cd/3), scale)[cd%3];
	ret.y = 4*get_weight_triple(ivec4(1, fp.yx, cd/3), scale)[cd%3];
	ret.z = 4*get_weight_triple(ivec4(2, fp.yx, cd/3), scale)[cd%3];
*/

	//debug features
	//ret = .5+tanh(-get_feature_triple(tex2xyf(ivec2(gl_FragCoord), scale), scale));

	//convert to unipolar
	if(activation_mode<1)
		ret = .5+.5*ret;

    gl_FragColor = vec4(ret,1.);

}