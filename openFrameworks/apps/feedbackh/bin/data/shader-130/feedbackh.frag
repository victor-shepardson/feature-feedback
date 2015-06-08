#version 130

uniform sampler2DRect state;
uniform sampler2DRect encoded;
uniform sampler2DRect decoded;
uniform int scale;
uniform ivec2 size;

uniform float fb;
uniform float gen;
uniform float tblur;
uniform float sblur;
uniform float warp;
uniform float perm;
uniform float frame;
uniform float zoom;
uniform int bound;

#define PI 3.1415926535897932384626433832795

vec2 invsize = 1./size;

//hsv conversion from lolengine.net: http://lolengine.net/blog/2013/07/27/rgb-to-hsv-in-glsl
vec3 rgb2hsv(vec3 c)
{
    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));

    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}
vec3 hsv2rgb(vec3 c)
{
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

vec2 pol2car(vec2 pol){
	return pol.x*vec2(cos(pol.y), sin(pol.y));
}

//convert between bipolar [-1, 1] and unipolar [0, 1]
vec3 u2b(vec3 u){
	return 2.*u-1.;
}
vec3 b2u(vec3 b){
	return .5*b+.5;
}

//convert (x, y, feature triple) coordinates to texture coordinates at scale s
ivec2 xyf2tex(ivec3 p, int s){
	ivec2 t = clamp(p.xy, ivec2(0), (size>>s)-1)<<s;
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

//sample b at position p.xy and feature triple p.z, given scale s
vec3 sample_xyf(sampler2DRect b, ivec3 p, int s){
	ivec2 t = xyf2tex(p, s);
	return texelFetch(b, t, 0).rgb;
}

void main() {
	ivec2 t = ivec2(gl_FragCoord.xy);
	ivec3 p = tex2xyf(t, scale);

	vec3 prev_state = sample_xyf(state, p, scale);
	/*vec3 conv_state =( sample_xyf(state, p+ivec3(-1, 0, 0), scale)
					+ sample_xyf(state, p+ivec3( 1, 0, 0), scale)
					+ sample_xyf(state, p+ivec3( 0,-1, 0), scale)
					+ sample_xyf(state, p+ivec3( 0, 1, 0), scale)
					+ prev_state)/5;
*/
	vec3 prev_enc = sample_xyf(encoded, p, scale);
/*	vec3 conv_enc =( sample_xyf(encoded, p+ivec3(-1, 0, 0), scale)
					+ sample_xyf(encoded, p+ivec3( 1, 0, 0), scale)
					+ sample_xyf(encoded, p+ivec3( 0,-1, 0), scale)
					+ sample_xyf(encoded, p+ivec3( 0, 1, 0), scale)
					+ prev_enc)/5;
*/
	vec3 prev_dec = sample_xyf(decoded, p, scale);
/*	vec3 conv_dec =( sample_xyf(decoded, p+ivec3(-1, 0, 0), scale)
					+ sample_xyf(decoded, p+ivec3( 1, 0, 0), scale)
					+ sample_xyf(decoded, p+ivec3( 0,-1, 0), scale)
					+ sample_xyf(decoded, p+ivec3( 0, 1, 0), scale)
					+ prev_dec)/5;
*/
	vec3 color = ( prev_dec + prev_enc );
	//vec3 color = (texelFetch(state, t).rgb + texelFetch(decoded, t).rgb + texelFetch(encoded, t).rgb );
	color = tanh(color-1);
	//color = sin(exp(color-1.5)-1);
	color = color*.5+.5;
	//color/=3;
	color = mix(color, prev_state, tblur);

	gl_FragColor = vec4(color,1);
}
	/*
	//sample last frame
	vec2 texcoordM = gl_FragCoord.xy;
	vec3 color_in = texture2DRect(state, texcoordM).rgb;
	vec3 mod_in = texture2DRect(modulation, texcoordM).rgb;

	//modulation
	vec3 mod_hsv = rgb2hsv(mod_in);
	float _perm = perm;//(mod_hsv.x - .5)*2.-1.;//float(mod_hsv.x > .5)*2.-1.;//perm;
	float _warp = warp ;//* mod_hsv.z;
	float _tblur = tblur;//.5+.5*pow(mod_hsv.z,.5)*tblur; 
	float _zoom = zoom;
	float _fb = fb;
	float _gen = gen;//mod_hsv.y*gen;
	vec3 color_gen = 2.*mod_in-1.;

	//warp
	vec3 hsv = rgb2hsv(color_in);
	vec2 disp = pol2car(vec2(hsv.y*_warp, 2*PI*hsv.x))-_zoom*invsize*(texcoordM-.5*size);

	//convolution
	vec3 color_center = sample(disp+texcoordM);
	vec3 color_left = sample(disp+texcoordM+vec2(-1.,0.));
	vec3 color_right = sample(disp+texcoordM+vec2(1.,0.));
	vec3 color_up = sample(disp+texcoordM+vec2(0.,-1.));
	vec3 color_down = sample(disp+texcoordM+vec2(0.,1.));

	vec3 color = mix(color_center, .25*(color_left+color_right+color_up+color_down), sblur);

	//to bipolar
	color = u2b(color);

	//color permutation
	color = (_perm>0.) ?  mix(color.rgb, color.gbr, _perm) : mix(color.rgb, color.brg, -_perm);

	//generators + feedback
	//vec3 color_gen = texture2DRect(image1, texcoordM).rgb;
	//color_gen = vec3(sin(2*PI*(gl_FragCoord.x+.1*frame)*invsize.x), sin(2*PI*(gl_FragCoord.y+.111*frame)*invsize.y), cos(2*PI*(gl_FragCoord.x+.11*frame)*-invsize.x));
	color = _gen * color_gen + _fb *color ;

	//bounding function
	if(bound==0){
		color = b2u(color);
	}
	else if(bound==1){
		//fract
		color = b2u(color);
		color = fract(color);
	}
	else if(bound==2){
		//sinusoid
		color = sin(.5*PI*color);
		color = b2u(color);
	}
	else if(bound==3){
		//sigmoid
		color = 1./(1.+exp(-color));
	}
	else if(bound==4){
		//clamp magnitude
		float mag = length(color.rgb);
		if(mag>1.)	color = color/mag;
		color = b2u(color);
	}
	else if(bound==5){
		//rectifying sigmoid
		color = 1./(1.+exp(-2.*color));
	}
	else if(bound==6){
		color = sin(exp(color)-1);
		color = b2u(color);
	}
	

	//temporal blur
	color = mix(color, color_in, _tblur);

    gl_FragColor = vec4(color, 1.);
*/
