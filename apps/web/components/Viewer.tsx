"use client";

import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";

interface Props {
  trajectory?: { age_now: number; age_target: number };
}

export function Viewer({ trajectory }: Props) {
  return (
    <div className="absolute inset-0">
      <Canvas camera={{ position: [2.5, 1.5, 2.5], fov: 45 }}>
        <ambientLight intensity={0.4} />
        <directionalLight position={[5, 5, 5]} intensity={1} />
        <BrainPlaceholder />
        <OrbitControls enableDamping makeDefault />
      </Canvas>
      <div className="absolute bottom-4 left-4 text-xs font-mono text-foreground/50 bg-background/70 px-2 py-1 rounded">
        {trajectory
          ? `t = ${trajectory.age_now} → ${trajectory.age_target}`
          : "Three.js Gaussian-splat shader pending (Phase 4)"}
      </div>
    </div>
  );
}

function BrainPlaceholder() {
  // Phase-0 stand-in. Replaced by hierarchical 3D-Gaussian-splat shader in Phase 4.
  return (
    <group>
      <mesh>
        <sphereGeometry args={[1, 64, 64]} />
        <meshStandardMaterial color="#d97757" roughness={0.4} metalness={0.1} />
      </mesh>
      <mesh position={[0, 0, 0]}>
        <sphereGeometry args={[1.02, 32, 32]} />
        <meshBasicMaterial wireframe color="#444" />
      </mesh>
    </group>
  );
}
