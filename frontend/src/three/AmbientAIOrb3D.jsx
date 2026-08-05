import React, { useRef, useEffect } from 'react';
import * as THREE from 'three';

export default function AmbientAIOrb3D({ isThinking = false, isError = false, height = 200 }) {
  const containerRef = useRef(null);

  useEffect(() => {
    if (!containerRef.current) return;

    const container = containerRef.current;
    const width = container.clientWidth;

    // Create scene
    const scene = new THREE.Scene();

    // Create camera
    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
    camera.position.z = 5;

    // Create renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(renderer.domElement);

    // Orb Geometry: Icosahedron allows organic morphing
    const geometry = new THREE.IcosahedronGeometry(1.5, 4);
    
    // Save original vertex positions to apply noise displacement
    const positionAttribute = geometry.attributes.position;
    const originalPositions = [];
    for (let i = 0; i < positionAttribute.count; i++) {
      originalPositions.push(new THREE.Vector3().fromBufferAttribute(positionAttribute, i));
    }

    // Material with low opacity and wireframe for calm glass-like AI look
    const isDark = document.documentElement.getAttribute('data-theme') !== 'light';
    let baseColor = isDark ? 0x8b5cf6 : 0x5516be; // Intellect Violet by default
    if (isError) {
      baseColor = 0xf43f5e; // Rose red
    } else if (isThinking) {
      baseColor = 0x3b82f6; // Strategic Blue
    }

    const material = new THREE.MeshBasicMaterial({
      color: baseColor,
      wireframe: true,
      transparent: true,
      opacity: isThinking ? 0.35 : 0.15
    });

    const orb = new THREE.Mesh(geometry, material);
    scene.add(orb);

    // Inner core mesh
    const coreGeo = new THREE.SphereGeometry(0.8, 16, 16);
    const coreMat = new THREE.MeshBasicMaterial({
      color: baseColor,
      transparent: true,
      opacity: isThinking ? 0.15 : 0.05
    });
    const core = new THREE.Mesh(coreGeo, coreMat);
    scene.add(core);

    // Simple pseudo-noise function for morphing
    const simplexNoise = (x, y, z, time) => {
      // Simple trigonometric noise approximation
      return Math.sin(x * 1.5 + time) * Math.cos(y * 1.5 + time) * Math.sin(z * 1.5 + time) * 0.12;
    };

    // Resize handler
    const handleResize = () => {
      if (!container) return;
      const w = container.clientWidth;
      camera.aspect = w / height;
      camera.updateProjectionMatrix();
      renderer.setSize(w, height);
    };
    window.addEventListener('resize', handleResize);

    // Animation Loop
    let clock = new THREE.Clock();
    let requestID;

    const animate = () => {
      requestID = requestAnimationFrame(animate);

      const time = clock.getElapsedTime() * (isThinking ? 2.5 : 0.8);
      
      // Update mesh vertices to create morphing orb
      const posAttr = geometry.attributes.position;
      for (let i = 0; i < posAttr.count; i++) {
        const p = originalPositions[i].clone();
        
        // Calculate displacement
        const displacement = simplexNoise(p.x, p.y, p.z, time);
        
        // Push vertex along its normal (which is equivalent to its normalized direction from center)
        p.normalize().multiplyScalar(1.5 + displacement);
        
        posAttr.setXYZ(i, p.x, p.y, p.z);
      }
      posAttr.needsUpdate = true;

      // Slow rotations
      orb.rotation.y += isThinking ? 0.015 : 0.003;
      orb.rotation.x += isThinking ? 0.008 : 0.002;
      core.rotation.y -= 0.002;

      renderer.render(scene, camera);
    };

    animate();

    // Clean up
    return () => {
      cancelAnimationFrame(requestID);
      window.removeEventListener('resize', handleResize);
      if (container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement);
      }
      scene.clear();
      renderer.dispose();
    };
  }, [isThinking, isError, height]);

  return (
    <div 
      ref={containerRef} 
      className="ai-orb-container"
      style={{ width: '100%', height: `${height}px`, overflow: 'hidden', position: 'relative' }} 
    />
  );
}
