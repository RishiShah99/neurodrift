/**
 * GLSL for the NeuroDrift 3D-Gaussian-splat renderer.
 *
 * Approach: camera-facing instanced billboards. Each Gaussian is one quad
 * instance; per-instance attributes carry center / scale / rotation / color /
 * opacity. The vertex shader expands a unit quad in *view space* so every splat
 * always faces the camera, scaled anisotropically by the projection of the
 * Gaussian's two largest principal axes. The fragment shader applies a radial
 * Gaussian falloff (exp(-r^2)) so each quad reads as a soft blob rather than a
 * hard card.
 *
 * Blending tradeoff: we use *additive* blending with `depthWrite = false`.
 * Additive blend is order-independent, which lets us skip the per-frame depth
 * sort that "correct" alpha-over splatting requires (an O(n log n) CPU sort
 * every camera move). The cost is that overlapping splats brighten rather than
 * occlude — perfectly acceptable for a volumetric brain field where we *want*
 * dense regions to glow, and it keeps the renderer real-time on the existing
 * three/fiber stack with zero extra deps. A future "high-fidelity" mode could
 * swap to sorted NORMAL blending behind the same component API.
 *
 * Attribute / uniform contract (must match Viewer.tsx):
 *   geometry attribute  `position`        vec3   unit-quad corner in [-0.5,0.5]
 *   instanced attribute `iCenter`         vec3   Gaussian center (world)
 *   instanced attribute `iScale`          vec3   per-axis std-dev (world units)
 *   instanced attribute `iRotation`       vec4   quaternion (x,y,z,w)
 *   instanced attribute `iColor`          vec3   linear rgb in [0,1]
 *   instanced attribute `iOpacity`        float  [0,1]
 *   uniform             `uSplatScale`     float  global size multiplier
 *   uniform             `uMaxBillboard`   float  clamp on view-space quad size
 */

export const splatVertexShader = /* glsl */ `
precision highp float;

attribute vec3 iCenter;
attribute vec3 iScale;
attribute vec4 iRotation;
attribute vec3 iColor;
attribute float iOpacity;

uniform float uSplatScale;
uniform float uMaxBillboard;

varying vec3 vColor;
varying float vOpacity;
varying vec2 vQuad;

// Rotate a vector by a quaternion (x,y,z,w).
vec3 qrot(vec4 q, vec3 v) {
  return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}

void main() {
  vColor = iColor;
  vOpacity = iOpacity;
  // position.xy is the unit-quad corner; expand to [-1,1] for the falloff.
  vQuad = position.xy * 2.0;

  // Anisotropic world-space extent: take the two dominant principal axes of the
  // Gaussian (rotation applied to the scaled X/Y basis). This is a cheap
  // projected-cov approximation that is exact when the splat faces the camera.
  vec3 axisX = qrot(iRotation, vec3(iScale.x, 0.0, 0.0)) * uSplatScale;
  vec3 axisY = qrot(iRotation, vec3(0.0, iScale.y, 0.0)) * uSplatScale;

  // Center in view space.
  vec4 centerView = modelViewMatrix * vec4(iCenter, 1.0);

  // Project the two world axes into view space (rotation part of MV only).
  mat3 mv3 = mat3(modelViewMatrix);
  vec3 exView = mv3 * axisX;
  vec3 eyView = mv3 * axisY;

  // Billboard offset: combine the projected axes by the quad corner, then clamp
  // the on-screen radius so a single huge splat can't blanket the viewport.
  vec2 offset = position.x * exView.xy + position.y * eyView.xy;
  float r = length(offset);
  if (r > uMaxBillboard && r > 0.0) {
    offset *= uMaxBillboard / r;
  }

  vec4 viewPos = centerView;
  viewPos.xy += offset * 3.0; // 3-sigma support so the falloff reaches ~0 at the edge

  gl_Position = projectionMatrix * viewPos;
}
`;

export const splatFragmentShader = /* glsl */ `
precision highp float;

varying vec3 vColor;
varying float vOpacity;
varying vec2 vQuad;

void main() {
  // Radial Gaussian falloff. vQuad spans [-1,1] across the 3-sigma quad, so
  // map to a squared radius and apply exp(-0.5 * (3r)^2)-ish support.
  float r2 = dot(vQuad, vQuad);
  float falloff = exp(-4.5 * r2);
  if (falloff < 0.004) discard; // trim the transparent corners
  // Additive: premultiply color by weight so dense overlap glows.
  float a = falloff * vOpacity;
  gl_FragColor = vec4(vColor * a, a);
}
`;
