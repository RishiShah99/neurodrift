/**
 * Binary 3D-Gaussian-splat loader for NeuroDrift.
 *
 * ── Binary layout ("NDGS1") ──────────────────────────────────────────────
 * A compact, little-endian, interleaved-record format designed to be trivial
 * to emit from the Python `.ply`/`.ksplat` exporter on the decode side. One
 * fixed-size header followed by N interleaved float32 records.
 *
 *   HEADER (32 bytes)
 *     offset 0   uint32   magic        = 0x4E444753  ("NDGS", little-endian)
 *     offset 4   uint32   version      = 1
 *     offset 8   uint32   count        = number of Gaussians (N)
 *     offset 12  uint32   stride       = bytes per record (= 56 for v1)
 *     offset 16  uint32   flags        = reserved (0)
 *     offset 20  float32  sceneRadius  = bounding radius hint (world units, 0 = unknown)
 *     offset 24  float32  reserved0    = 0
 *     offset 28  float32  reserved1    = 0
 *
 *   RECORD (stride = 56 bytes, all float32, 14 floats):
 *     [0..2]   position   x, y, z         (world units)
 *     [3..5]   scale      sx, sy, sz      (per-axis std-dev, world units, linear — NOT log)
 *     [6..9]   rotation   qx, qy, qz, qw  (unit quaternion; loader re-normalizes)
 *     [10]     opacity    a               (linear [0,1]; loader clamps)
 *     [11..13] color      r, g, b         (linear [0,1])
 *
 * Notes for the Python exporter:
 *   - Write the header, then each Gaussian as 14 contiguous float32 in the order
 *     above. No padding between records (stride == 14*4 == 56).
 *   - Scales are LINEAR std-devs in world units. If your decoder stores log-scale
 *     (as raw 3DGS checkpoints do), apply exp() before writing.
 *   - Opacity is LINEAR. If your decoder stores logit-opacity, apply sigmoid()
 *     before writing.
 *   - Colors are LINEAR rgb in [0,1]. If you store SH DC terms, evaluate the
 *     0th-order SH (rgb = 0.5 + C0 * sh_dc, C0 = 0.28209479177387814) and clamp.
 *   - Coordinate frame is whatever the viewer's OrbitControls expect (y-up,
 *     looking down -z); center the cloud near the origin.
 *
 * This format is versioned via the header `version` field; bump it and branch in
 * `parseGaussianBuffer` if the record layout ever changes.
 */

export const NDGS_MAGIC = 0x4e444753; // "NDGS"
export const NDGS_VERSION = 1;
export const NDGS_HEADER_BYTES = 32;
export const NDGS_FLOATS_PER_RECORD = 14;
export const NDGS_STRIDE_BYTES = NDGS_FLOATS_PER_RECORD * 4; // 56

/**
 * Parsed, render-ready Gaussian cloud. Buffers are flat and interleaved-free
 * (one typed array per attribute) so they can be uploaded straight into
 * InstancedBufferAttributes without a copy.
 */
export interface GaussianCloud {
  /** Number of Gaussians. */
  count: number;
  /** xyz centers, length = count * 3. */
  positions: Float32Array;
  /** per-axis std-devs, length = count * 3. */
  scales: Float32Array;
  /** quaternions (x,y,z,w), length = count * 4. */
  rotations: Float32Array;
  /** linear opacity in [0,1], length = count. */
  opacities: Float32Array;
  /** linear rgb in [0,1], length = count * 3. */
  colors: Float32Array;
  /** Bounding radius hint in world units (0 if unknown). */
  sceneRadius: number;
}

/** An allocation-free empty cloud, used as the null / loading sentinel. */
export function emptyCloud(): GaussianCloud {
  return {
    count: 0,
    positions: new Float32Array(0),
    scales: new Float32Array(0),
    rotations: new Float32Array(0),
    opacities: new Float32Array(0),
    colors: new Float32Array(0),
    sceneRadius: 0,
  };
}

export function isEmptyCloud(cloud: GaussianCloud): boolean {
  return cloud.count === 0;
}

/**
 * Parse a packed NDGS1 binary into a {@link GaussianCloud}.
 * Throws on malformed input (bad magic, unsupported version, truncated body).
 */
export function parseGaussianBuffer(buf: ArrayBuffer): GaussianCloud {
  if (buf.byteLength < NDGS_HEADER_BYTES) {
    throw new Error(
      `gaussians: buffer too small for header (${buf.byteLength} < ${NDGS_HEADER_BYTES})`,
    );
  }
  const header = new DataView(buf);
  const magic = header.getUint32(0, true);
  if (magic !== NDGS_MAGIC) {
    throw new Error(
      `gaussians: bad magic 0x${magic.toString(16)} (expected 0x${NDGS_MAGIC.toString(16)})`,
    );
  }
  const version = header.getUint32(4, true);
  if (version !== NDGS_VERSION) {
    throw new Error(`gaussians: unsupported version ${version} (expected ${NDGS_VERSION})`);
  }
  const count = header.getUint32(8, true);
  const stride = header.getUint32(12, true);
  if (stride !== NDGS_STRIDE_BYTES) {
    throw new Error(`gaussians: unexpected stride ${stride} (expected ${NDGS_STRIDE_BYTES})`);
  }
  const sceneRadius = header.getFloat32(20, true);

  const need = NDGS_HEADER_BYTES + count * NDGS_STRIDE_BYTES;
  if (buf.byteLength < need) {
    throw new Error(
      `gaussians: truncated body (have ${buf.byteLength}, need ${need} for ${count} records)`,
    );
  }
  if (count === 0) {
    return { ...emptyCloud(), sceneRadius };
  }

  // Read records as a strided Float32 view over the body.
  const body = new Float32Array(buf, NDGS_HEADER_BYTES, count * NDGS_FLOATS_PER_RECORD);

  const positions = new Float32Array(count * 3);
  const scales = new Float32Array(count * 3);
  const rotations = new Float32Array(count * 4);
  const opacities = new Float32Array(count);
  const colors = new Float32Array(count * 3);

  for (let i = 0; i < count; i++) {
    const o = i * NDGS_FLOATS_PER_RECORD;

    positions[i * 3 + 0] = body[o + 0]!;
    positions[i * 3 + 1] = body[o + 1]!;
    positions[i * 3 + 2] = body[o + 2]!;

    // Guard against zero/negative scale producing degenerate quads.
    scales[i * 3 + 0] = Math.max(body[o + 3]!, 1e-6);
    scales[i * 3 + 1] = Math.max(body[o + 4]!, 1e-6);
    scales[i * 3 + 2] = Math.max(body[o + 5]!, 1e-6);

    // Re-normalize the quaternion defensively (exporters drift).
    let qx = body[o + 6]!;
    let qy = body[o + 7]!;
    let qz = body[o + 8]!;
    let qw = body[o + 9]!;
    const qlen = Math.hypot(qx, qy, qz, qw) || 1;
    qx /= qlen;
    qy /= qlen;
    qz /= qlen;
    qw /= qlen;
    rotations[i * 4 + 0] = qx;
    rotations[i * 4 + 1] = qy;
    rotations[i * 4 + 2] = qz;
    rotations[i * 4 + 3] = qw;

    opacities[i] = Math.min(Math.max(body[o + 10]!, 0), 1);

    colors[i * 3 + 0] = body[o + 11]!;
    colors[i * 3 + 1] = body[o + 12]!;
    colors[i * 3 + 2] = body[o + 13]!;
  }

  return { count, positions, scales, rotations, opacities, colors, sceneRadius };
}

/**
 * Fetch and parse a Gaussian cloud from `url`.
 * - A null/empty url resolves to an empty cloud (the page still renders).
 * - A 0-byte or 404 response resolves to an empty cloud rather than throwing,
 *   so a not-yet-ready inference doesn't blank the viewer.
 * - A non-empty but malformed body rejects (surfaces real exporter bugs).
 */
export async function fetchGaussians(
  url: string | null | undefined,
  signal?: AbortSignal,
): Promise<GaussianCloud> {
  if (!url) return emptyCloud();

  const res = await fetch(url, { signal });
  if (!res.ok) {
    // Treat "not there yet" as empty; only hard-fail on unexpected server errors.
    if (res.status === 404 || res.status === 204) return emptyCloud();
    throw new Error(`gaussians: fetch failed ${res.status} ${res.statusText}`);
  }

  const buf = await res.arrayBuffer();
  if (buf.byteLength === 0) return emptyCloud();
  return parseGaussianBuffer(buf);
}

/**
 * Hierarchical-culling stub hook.
 *
 * Real 3DGS viewers maintain an LOD tree and cull / decimate by screen-space
 * footprint and distance. This is a placeholder that returns the index subset
 * to draw. For now it is identity (draw all), but the signature is the seam a
 * future implementation slots into: pass camera position + a budget, get back a
 * culled, possibly-reordered index list.
 *
 * @returns null to mean "draw everything" (lets callers skip an index buffer).
 */
export function cullByDistance(
  _cloud: GaussianCloud,
  _cameraPosition: readonly [number, number, number],
  _opts?: { maxCount?: number; maxDistance?: number },
): Uint32Array | null {
  // TODO(phase4+): build a coarse octree at load time and return a distance-/
  // budget-pruned, front-to-back index list here.
  return null;
}
