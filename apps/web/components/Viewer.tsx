"use client";

import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import {
  emptyCloud,
  fetchGaussians,
  isEmptyCloud,
  type GaussianCloud,
} from "@/lib/gaussians";
import { splatFragmentShader, splatVertexShader } from "@/lib/splatShader";

interface Props {
  trajectory?: { age_now: number; age_target: number };
  /** URL to a packed NDGS1 Gaussian binary. Null/empty -> subtle placeholder. */
  gaussiansUrl?: string | null;
}

export function Viewer({ trajectory, gaussiansUrl }: Props) {
  const [cloud, setCloud] = useState<GaussianCloud>(emptyCloud);
  const [status, setStatus] = useState<"empty" | "loading" | "ready" | "error">(
    "empty",
  );

  useEffect(() => {
    if (!gaussiansUrl) {
      setCloud(emptyCloud());
      setStatus("empty");
      return;
    }
    const ctrl = new AbortController();
    setStatus("loading");
    fetchGaussians(gaussiansUrl, ctrl.signal)
      .then((c) => {
        setCloud(c);
        setStatus(isEmptyCloud(c) ? "empty" : "ready");
      })
      .catch((err: unknown) => {
        if ((err as { name?: string })?.name === "AbortError") return;
        console.error("Gaussian load failed:", err);
        setCloud(emptyCloud());
        setStatus("error");
      });
    return () => ctrl.abort();
  }, [gaussiansUrl]);

  const hasSplats = !isEmptyCloud(cloud);

  return (
    <div className="absolute inset-0">
      <Canvas
        camera={{ position: [2.5, 1.5, 2.5], fov: 45 }}
        gl={{ antialias: true, alpha: true }}
        dpr={[1, 2]}
      >
        <color attach="background" args={["#0a0a0a"]} />
        <ambientLight intensity={0.4} />
        <directionalLight position={[5, 5, 5]} intensity={1} />

        {hasSplats ? <SplatCloud cloud={cloud} /> : <BrainPlaceholder />}

        <OrbitControls enableDamping makeDefault />
      </Canvas>

      <div className="absolute bottom-4 left-4 text-xs font-mono text-foreground/50 bg-background/70 px-2 py-1 rounded">
        {overlayText(trajectory, status, cloud.count)}
      </div>
    </div>
  );
}

function overlayText(
  trajectory: Props["trajectory"],
  status: "empty" | "loading" | "ready" | "error",
  count: number,
): string {
  if (status === "loading") return "loading splats…";
  if (status === "error") return "splat load failed — showing placeholder";
  if (status === "ready") {
    const splats = `${count.toLocaleString()} gaussians`;
    return trajectory
      ? `t = ${trajectory.age_now} → ${trajectory.age_target} · ${splats}`
      : splats;
  }
  // empty / no cloud
  return trajectory
    ? `t = ${trajectory.age_now} → ${trajectory.age_target}`
    : "Three.js Gaussian-splat viewer (awaiting cloud)";
}

/**
 * Instanced billboard splat renderer.
 *
 * One quad instance per Gaussian; per-instance attributes drive the shader
 * (see lib/splatShader.ts). We build a THREE.Mesh imperatively and hand it to
 * R3F via <primitive>, which keeps this version-agnostic across fiber v8/v9 and
 * avoids JSX-intrinsic typing churn on instanced attributes.
 *
 * Blending: additive + depthWrite off -> order-independent, no per-frame sort.
 */
function SplatCloud({ cloud }: { cloud: GaussianCloud }) {
  const { camera } = useThree();

  const mesh = useMemo(() => {
    // Unit quad in the XY plane, corners in [-0.5, 0.5]; the shader expands it.
    const geometry = new THREE.InstancedBufferGeometry();
    const quad = new Float32Array([
      -0.5, -0.5, 0, 0.5, -0.5, 0, 0.5, 0.5, 0, -0.5, 0.5, 0,
    ]);
    geometry.setAttribute("position", new THREE.BufferAttribute(quad, 3));
    geometry.setIndex([0, 1, 2, 0, 2, 3]);
    geometry.instanceCount = cloud.count;

    geometry.setAttribute(
      "iCenter",
      new THREE.InstancedBufferAttribute(cloud.positions, 3),
    );
    geometry.setAttribute(
      "iScale",
      new THREE.InstancedBufferAttribute(cloud.scales, 3),
    );
    geometry.setAttribute(
      "iRotation",
      new THREE.InstancedBufferAttribute(cloud.rotations, 4),
    );
    geometry.setAttribute(
      "iOpacity",
      new THREE.InstancedBufferAttribute(cloud.opacities, 1),
    );
    geometry.setAttribute(
      "iColor",
      new THREE.InstancedBufferAttribute(cloud.colors, 3),
    );

    // Bound the geometry so frustum culling / OrbitControls auto-fit behave.
    const radius = cloud.sceneRadius > 0 ? cloud.sceneRadius : 2;
    geometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, 0, 0), radius * 1.5);

    const material = new THREE.ShaderMaterial({
      vertexShader: splatVertexShader,
      fragmentShader: splatFragmentShader,
      uniforms: {
        uSplatScale: { value: 1.0 },
        uMaxBillboard: { value: 0.6 },
      },
      transparent: true,
      depthWrite: false,
      depthTest: true,
      blending: THREE.AdditiveBlending,
    });

    const m = new THREE.Mesh(geometry, material);
    m.frustumCulled = false; // billboards extend beyond the base quad bounds
    return m;
  }, [cloud]);

  // Dispose GPU resources when the cloud changes or the component unmounts.
  useEffect(() => {
    return () => {
      mesh.geometry.dispose();
      (mesh.material as THREE.Material).dispose();
    };
  }, [mesh]);

  // Hierarchical-culling stub seam: each frame we *could* reprune by distance.
  // For now we only keep the camera position handy for a future LOD pass.
  useFrame(() => {
    // Placeholder: a future pass calls cullByDistance(cloud, cameraPos, …) and
    // swaps geometry.instanceCount / an index range here. No-op today.
    void camera.position;
  });

  return <primitive object={mesh} />;
}

function BrainPlaceholder() {
  // Subtle stand-in shown when no Gaussian cloud is loaded, so the page still
  // renders. Replaced by the splat cloud the moment a valid NDGS1 url arrives.
  const ref = useRef<THREE.Group>(null);
  useFrame((_, dt) => {
    if (ref.current) ref.current.rotation.y += dt * 0.15;
  });
  return (
    <group ref={ref}>
      <mesh>
        <sphereGeometry args={[1, 64, 64]} />
        <meshStandardMaterial
          color="#d97757"
          roughness={0.4}
          metalness={0.1}
          transparent
          opacity={0.25}
        />
      </mesh>
      <mesh>
        <sphereGeometry args={[1.02, 32, 32]} />
        <meshBasicMaterial wireframe color="#444" />
      </mesh>
    </group>
  );
}
