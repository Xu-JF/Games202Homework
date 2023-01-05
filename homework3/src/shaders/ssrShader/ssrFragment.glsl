#ifdef GL_ES
precision highp float;
#endif

uniform vec3 uLightDir;
uniform vec3 uCameraPos;
uniform vec3 uLightRadiance;
uniform float uScreenSizeWidth;
uniform float uScreenSizeHeight;
uniform sampler2D uGDiffuse;
uniform sampler2D uGDepth;
uniform sampler2D uGNormalWorld;
uniform sampler2D uGShadow;
uniform sampler2D uGPosWorld;

varying mat4 vWorldToScreen;
varying highp vec4 vPosWorld;

#define M_PI 3.1415926535897932384626433832795
#define TWO_PI 6.283185307
#define INV_PI 0.31830988618
#define INV_TWO_PI 0.15915494309
#define MAX_THICKNESS 0.0017

float Rand1(inout float p) {
  p = fract(p * .1031);
  p *= p + 33.33;
  p *= p + p;
  return fract(p);
}

vec2 Rand2(inout float p) {
  return vec2(Rand1(p), Rand1(p));
}

float InitRand(vec2 uv) {
	vec3 p3  = fract(vec3(uv.xyx) * .1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

vec3 SampleHemisphereUniform(inout float s, out float pdf) {
  vec2 uv = Rand2(s);
  float z = uv.x;
  float phi = uv.y * TWO_PI;
  float sinTheta = sqrt(1.0 - z*z);
  vec3 dir = vec3(sinTheta * cos(phi), sinTheta * sin(phi), z);
  pdf = INV_TWO_PI;
  return dir;
}

vec3 SampleHemisphereCos(inout float s, out float pdf) {
  vec2 uv = Rand2(s);
  float z = sqrt(1.0 - uv.x);
  float phi = uv.y * TWO_PI;
  float sinTheta = sqrt(uv.x);
  vec3 dir = vec3(sinTheta * cos(phi), sinTheta * sin(phi), z);
  pdf = z * INV_PI;
  return dir;
}

void LocalBasis(vec3 n, out vec3 b1, out vec3 b2) {
  float sign_ = sign(n.z);
  if (n.z == 0.0) {
    sign_ = 1.0;
  }
  float a = -1.0 / (sign_ + n.z);
  float b = n.x * n.y * a;
  b1 = vec3(1.0 + sign_ * n.x * n.x * a, sign_ * b, -sign_ * n.x);
  b2 = vec3(b, sign_ + n.y * n.y * a, -n.y);
}

vec4 Project(vec4 a) {
  return a / a.w;
}

float GetDepth(vec3 posWorld) {
  float depth = (vWorldToScreen * vec4(posWorld, 1.0)).w;
  return depth;
}

/*
 * Transform point from world space to screen space([0, 1] x [0, 1])
 *
 */
vec2 GetScreenCoordinate(vec3 posWorld) {
  vec2 uv = Project(vWorldToScreen * vec4(posWorld, 1.0)).xy * 0.5 + 0.5;
  return uv;
}

vec3 GetScreenCoord(vec3 posWorld){
  return Project(vWorldToScreen * vec4(posWorld, 1.0)).xyz * 0.5 + 0.5;
}

float GetGBufferDepth(vec2 uv) {
  float depth = texture2D(uGDepth, uv).x;
  if (depth < 1e-2) {
    depth = 1000.0;
  }
  return depth;
}

vec3 GetGBufferNormalWorld(vec2 uv) {
  vec3 normal = texture2D(uGNormalWorld, uv).xyz;
  return normal;
}

vec3 GetGBufferPosWorld(vec2 uv) {
  vec3 posWorld = texture2D(uGPosWorld, uv).xyz;
  return posWorld;
}

float GetGBufferuShadow(vec2 uv) {
  float visibility = texture2D(uGShadow, uv).x;
  return visibility;
}

vec3 GetGBufferDiffuse(vec2 uv) {
  vec3 diffuse = texture2D(uGDiffuse, uv).xyz;
  diffuse = pow(diffuse, vec3(2.2));
  return diffuse;
}

/*
 * Evaluate diffuse bsdf value.
 *
 * wi, wo are all in world space.
 * uv is in screen space, [0, 1] x [0, 1].
 *
 */
vec3 EvalDiffuse(vec3 wi, vec3 wo, vec2 uv) {
  vec3 diff = GetGBufferDiffuse(uv);
  vec3 norm = GetGBufferNormalWorld(uv);
  float NdotL = dot(norm, wi);
  if(NdotL <= 0.0)
    return vec3(0.0);
  return NdotL * diff;
}

/*
 * Evaluate directional light with shadow map
 * uv is in screen space, [0, 1] x [0, 1].
 *
 */
vec3 EvalDirectionalLight(vec2 uv) {
  vec3 posW = GetGBufferPosWorld(uv);
  vec3 wi = normalize(uLightDir);
  vec3 wo = normalize(uCameraPos - posW);
  vec3 bsdf = EvalDiffuse(wi, wo, uv);
  return uLightRadiance * bsdf * GetGBufferuShadow(uv);
}

#define INIT_STEP 0.05
#define MAX_STEPS 200
#define EPS 1e-2
#define THRES 0.1
bool outScreen(vec3 pos){
  vec2 uv = GetScreenCoordinate(pos);
  return any(bvec4(lessThan(uv, vec2(0.0)), greaterThan(uv, vec2(1.0))));
}
bool atFront(vec3 pos){
  return GetDepth(pos) < GetGBufferDepth(GetScreenCoordinate(pos));
}
bool hasInter(vec3 pos, vec3 dir, out vec3 hitPos){
  float d1 = GetGBufferDepth(GetScreenCoordinate(pos)) - GetDepth(pos) + EPS;
  float d2 = GetDepth(pos + dir) - GetGBufferDepth(GetScreenCoordinate(pos + dir)) + EPS;
  if(d1 < THRES && d2 < THRES){
    hitPos = pos + dir * d1 / (d1 + d2);
    return true;
  }  
  return false;
}

bool RayMarch(vec3 ori, vec3 dir, out vec3 hitPos) {
  bool intersect = false, firstinter = false;
  float st = INIT_STEP/(length(dir.xy) + EPS);
  vec3 current = ori;
  vec2 screenPos = GetScreenCoordinate(ori);
  for (int i = 0;i < MAX_STEPS;i++){
    if(outScreen(current)){
      break;
    }
    else if(atFront(current + dir * st)){
      current += dir * st;
    }else{
      firstinter = true;
      if(st < EPS){
        if(hasInter(current, dir * st * 2.0, hitPos)){
          intersect = true;
        }
        break;
      }
    }
    if(firstinter)
      st *= 0.5;
  }
  return intersect;
}

#define SAMPLE_NUM 2
vec3 EvalIndirectLight(vec3 pos){
  float pdf, seed = dot(pos, vec3(100.0));
  vec3 Li = vec3(0.0), dir, hitPos;
  vec3 normal = GetGBufferNormalWorld(GetScreenCoordinate(pos)), b1, b2;
  LocalBasis(normal, b1, b2);
  mat3 TBN = mat3(b1, b2, normal);
  for(int i = 0; i < SAMPLE_NUM;i++){
    dir = normalize(TBN * SampleHemisphereCos(seed, pdf));
    vec2 uv = GetScreenCoordinate(pos);
    vec3 worldPos = GetGBufferPosWorld(uv);
    vec3 dir = normalize(reflect(worldPos - uCameraPos, GetGBufferNormalWorld(uv)));
    if(RayMarch(pos, dir, hitPos)){
      vec3 wo = normalize(uCameraPos - pos);
      vec3 L = EvalDiffuse(dir, wo, GetScreenCoordinate(pos)) / pdf;
      wo = normalize(uCameraPos - hitPos);
      vec3 wi = normalize(uLightDir);
      L *= EvalDiffuse(wi, wo, GetScreenCoordinate(hitPos)) * EvalDirectionalLight(GetScreenCoordinate(hitPos));
      Li += L;
    }
  }
  return Li / float(SAMPLE_NUM);
}

vec2 getCell(vec2 pos,vec2 startCellCount){
  return vec2(floor(pos*startCellCount));
}

vec3 intersectDepthPlane(vec3 o, vec3 d, float t){
    return o + d * t;
}

vec3 intersectCellBoundary(vec3 o,vec3  d, vec2 rayCell,vec2 cell_count, vec2 crossStep, vec2 crossOffset){

    vec2 nextPos = rayCell + crossStep ;
    nextPos = nextPos/cell_count;
    nextPos = nextPos+crossOffset;

    vec2 dis  = nextPos - o.xy;

    vec2 delta = dis/d.xy;

    float t = min(delta.x,delta.y);

    return intersectDepthPlane(o,d,t);
}

bool crossedCellBoundary(vec2 oldCellIdx,vec2 newCellIdx){
    return (oldCellIdx.x!=newCellIdx.x)||(oldCellIdx.y!=newCellIdx.y);
}

// vec2 uv = GetScreenCoordinate(pos);
vec3 GetIndirectLight(vec3 start,vec3 rayDir,float maxTraceDistance, vec3 hitPos)
{
  vec2 crossStep = vec2(rayDir.x >= 0.0 ? 1 : -1, rayDir.y >= 0.0 ? 1 : -1);
  vec2 crossOffset = crossStep / vec2(1024.0, 1024.0) / 128.0;

  crossStep = clamp(crossStep, 0.0, 1.0);

  vec3 ray = start;
  float minZ = ray.z;
  float maxZ = ray.z + rayDir.z * maxTraceDistance;
  float deltaZ = maxZ - minZ;

  vec3 o = ray;
  vec3 d = rayDir * maxTraceDistance;
  vec2 startCellCount = vec2(uScreenSizeWidth, uScreenSizeHeight);

  vec2 rayCell = getCell(ray.xy, startCellCount);
  ray = intersectCellBoundary(o, d, rayCell, startCellCount, crossStep, crossOffset*64.0);

  int level = 0;
  int iter = 0;
  bool isBackwardRay = rayDir.z < 0.0;

  float Dir = isBackwardRay ? -1.0 : 1.0;

  for (int i = 0;i < 100; i++){
    if(!(level >= 0 && ray.z * Dir <= maxZ * Dir)){
      break;
    }
    vec2 cellCount = startCellCount;
    vec2 oldCellIdx = getCell(ray.xy, cellCount);

    float cell_minZ = GetGBufferDepth(ray.xy);
    vec3 tmpRay = ((cell_minZ > ray.z) && !isBackwardRay) ? intersectDepthPlane(o,d,(cell_minZ - minZ)/deltaZ) : ray;

    vec2 newCellIdx = getCell(tmpRay.xy, cellCount);

    float thickness = ray.z - cell_minZ;
    bool crossed = (isBackwardRay && (cell_minZ > ray.z)) || (thickness > MAX_THICKNESS) || crossedCellBoundary(oldCellIdx, newCellIdx);

    ray = crossed ? intersectCellBoundary(o, d, oldCellIdx, cellCount, crossStep, crossOffset) : tmpRay;
    level = crossed ? 0:-1;
  }

  // while(level >= 0 && ray.z * Dir <= maxZ * Dir && iter < 100){
  //   vec2 cellCount = startCellCount;
  //   vec2 oldCellIdx = getCell(ray.xy, cellCount);

  //   float cell_minZ = GetGBufferDepth(ray.xy);
  //   vec3 tmpRay = ((cell_minZ > ray.z) && !isBackwardRay) ? intersectDepthPlane(o,d,(cell_minZ - minZ)/deltaZ) : ray;

  //   vec2 newCellIdx = getCell(tmpRay.xy, cellCount);

  //   float thickness = ray.z - cell_minZ;
  //   bool crossed = (isBackwardRay && (cell_minZ > ray.z)) || (thickness > MAX_THICKNESS) || crossedCellBoundary(oldCellIdx, newCellIdx);

  //   ray = crossed ? intersectCellBoundary(o, d, oldCellIdx, cellCount, crossStep, crossOffset) : tmpRay;
  //   level = crossed ? 0:-1;
  //   ++iter;
  // }

  bool intersected = level < 0;
  hitPos = ray;

  return intersected ? EvalDirectionalLight(hitPos.xy) : vec3(0.0);
}

void main() {
  float s = InitRand(gl_FragCoord.xy);

  vec3 L = vec3(0.0), Li;
  L = EvalDirectionalLight(GetScreenCoordinate(vPosWorld.xyz));

  //------------------------------
  vec2 uv = GetScreenCoordinate(vPosWorld.xyz);
  vec3 normalInWorld = GetGBufferNormalWorld(uv);
  vec3 reflectDirInWorld = normalize(reflect(vPosWorld.xyz, normalInWorld));

  vec3 endPosInWord = vPosWorld.xyz + reflectDirInWorld * 1000.0;

  vec3 startInTS = GetScreenCoord(vPosWorld.xyz);
  vec3 endInTS = GetScreenCoord(endPosInWord);
  vec3 rayDir = normalize(endInTS - startInTS);

  float maxTraceX = rayDir.x >= 0.0 ? (1.0 - startInTS.x) / rayDir.x : -startInTS.x / rayDir.x;
  float maxTraceY = rayDir.y >= 0.0 ? (1.0 - startInTS.y) / rayDir.y : -startInTS.y / rayDir.y;
  float maxTraceZ = rayDir.z >= 0.0 ? (1.0 - startInTS.z) / rayDir.z : -startInTS.z / rayDir.z;
  float maxTraceDistance = min(maxTraceX, min(maxTraceY, maxTraceZ));

  vec3 hitPos = vec3(0.0);

  Li = GetIndirectLight(startInTS, rayDir, maxTraceDistance, hitPos);

  // Li = vec3(reflectDirInWorld);
  //------------------------------
  Li = EvalIndirectLight(vPosWorld.xyz);
  L += Li;
  vec3 color = pow(clamp(L, vec3(0.0), vec3(1.0)), vec3(1.0 / 2.2));

  gl_FragColor = vec4(color, 1.0);
}