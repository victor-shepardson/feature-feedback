#version 400

uniform sampler2DRect state;
uniform sampler2DRect encoded;
uniform sampler2DRect decoded;
uniform int scale;
uniform ivec2 size;
uniform int border_mode;

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
	ivec2 t = p.xy;
	int f = p.z;
	if(border_mode<=0)
		t = clamp(t, ivec2(0), (size>>s)-1); //clamp to edges
	else if(border_mode<=1)
		t = ((size>>s) + t) % (size>>s); //wrap around the torus
	t = t<<s;
	t.x += f%(1<<s);
	t.y += f>>s;
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
	return texelFetch(b, t).rgb *2. - 1.;
}

void main() {
	ivec2 t = ivec2(gl_FragCoord.xy);
	ivec3 p = tex2xyf(t, scale);

	vec3 prev_state = sample_xyf(state, p, scale);
	vec3 conv_state =( -sample_xyf(state, p+ivec3(-1, 0, 0), scale)
					- sample_xyf(state, p+ivec3( 1, 0, 0), scale)
					- sample_xyf(state, p+ivec3( 0,-1, 0), scale)
					- sample_xyf(state, p+ivec3( 0, 1, 0), scale)
					+ prev_state*4)/8;
					
	//conv_state = mix(prev_state, conv_state, sblur);
	
	vec3 prev_enc = sample_xyf(encoded, p, scale);
	vec3 conv_enc = ( sample_xyf(encoded, p+ivec3(-1, 0, 0), scale)
					+ sample_xyf(encoded, p+ivec3( 1, 0, 0), scale)
					+ sample_xyf(encoded, p+ivec3( 0,-1, 0), scale)
					+ sample_xyf(encoded, p+ivec3( 0, 1, 0), scale)
					+ prev_enc*4)/8;
	//conv_enc = mix(prev_enc, conv_enc, sblur);

	vec3 prev_dec = sample_xyf(decoded, p, scale);
	vec3 conv_dec = ( sample_xyf(decoded, p+ivec3(-1, 0, 0), scale)
					+ sample_xyf(decoded, p+ivec3( 1, 0, 0), scale)
					+ sample_xyf(decoded, p+ivec3( 0,-1, 0), scale)
					+ sample_xyf(decoded, p+ivec3( 0, 1, 0), scale)
					+ prev_dec*4)/8;
	//conv_dec = mix(prev_dec, conv_dec, sblur);

	vec3 color;
	color = fb*(conv_enc+conv_dec)*.5;
	/*switch(scale){
		case 0: color = conv_dec; break;
		case 1: color = tanh(conv_dec + conv_enc + conv_state); break;
		case 2: color = conv_enc;
	} */
	//color = sin(exp(-2*color)-1);
	color = tanh(color);
	//color = color/(color*color+.25);
	//color = sin(3.14*color);
	//color/=3;
	color = mix(color, prev_state, tblur);

	color = color*.5+.5;

	gl_FragColor = vec4(color,1);
}