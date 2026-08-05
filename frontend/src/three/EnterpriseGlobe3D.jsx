import React, { useRef, useEffect } from 'react';
import * as THREE from 'three';

export default function EnterpriseGlobe3D({ height = 240 }) {
  const containerRef = useRef(null);

  useEffect(() => {
    if (!containerRef.current) return;

    const container = containerRef.current;
    const width = container.clientWidth;
    
    // Scene setup
    const scene = new THREE.Scene();

    // Camera setup
    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
    camera.position.z = 8;

    // Renderer setup
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(renderer.domElement);

    // Group to hold all globe objects for easy rotation
    const globeGroup = new THREE.Group();
    scene.add(globeGroup);

    // 1. Globe Mesh (Wireframe + Sphere)
    const sphereGeo = new THREE.SphereGeometry(2, 24, 24);
    
    // Theme colors mapping
    const isDark = document.documentElement.getAttribute('data-theme') !== 'light';
    const wireframeColor = isDark ? 0x3b82f6 : 0x005ac2; // strategic blue
    const arcColor = isDark ? 0x8b5cf6 : 0x5516be;       // intellect violet
    const pointColor = isDark ? 0x10b981 : 0x10b981;     // desaturated emerald

    const sphereMat = new THREE.MeshBasicMaterial({
      color: wireframeColor,
      wireframe: true,
      transparent: true,
      opacity: 0.1
    });
    const mainSphere = new THREE.Mesh(sphereGeo, sphereMat);
    globeGroup.add(mainSphere);

    // Inner glowing sphere
    const innerGeo = new THREE.SphereGeometry(1.95, 16, 16);
    const innerMat = new THREE.MeshBasicMaterial({
      color: wireframeColor,
      transparent: true,
      opacity: 0.03
    });
    const innerSphere = new THREE.Mesh(innerGeo, innerMat);
    globeGroup.add(innerSphere);

    // Helper: Spherical to Cartesian converter
    const convertCoords = (lat, lon, radius = 2) => {
      const phi = (90 - lat) * (Math.PI / 180);
      const theta = (lon + 180) * (Math.PI / 180);
      return new THREE.Vector3(
        -(radius * Math.sin(phi) * Math.sin(theta)),
        radius * Math.cos(phi),
        radius * Math.sin(phi) * Math.cos(theta)
      );
    };

    // 2. Hub locations
    const hubs = [
      { name: 'US-East', lat: 37.9, lon: -77.0 },
      { name: 'EU-West', lat: 53.1, lon: -8.2 },
      { name: 'Asia-Pac', lat: 35.6, lon: 139.6 }
    ];

    const hubPositions = hubs.map(hub => convertCoords(hub.lat, hub.lon));

    // Render hub points
    const pointGeo = new THREE.SphereGeometry(0.08, 8, 8);
    const pointMat = new THREE.MeshBasicMaterial({ color: pointColor });
    
    hubPositions.forEach(pos => {
      const point = new THREE.Mesh(pointGeo, pointMat);
      point.position.copy(pos);
      globeGroup.add(point);
    });

    // 3. Telemetry Arcs (Bezier curves)
    const createArc = (start, end) => {
      const mid = new THREE.Vector3().addVectors(start, end).multiplyScalar(0.5);
      const dist = start.distanceTo(end);
      
      // Pull mid point outwards to create curved arc
      mid.normalize().multiplyScalar(2 + dist * 0.2);

      const curve = new THREE.QuadraticBezierCurve3(start, mid, end);
      const points = curve.getPoints(30);
      
      const lineGeo = new THREE.BufferGeometry().setFromPoints(points);
      const lineMat = new THREE.LineBasicMaterial({
        color: arcColor,
        transparent: true,
        opacity: 0.4
      });
      
      const line = new THREE.Line(lineGeo, lineMat);
      globeGroup.add(line);
    };

    createArc(hubPositions[0], hubPositions[1]); // US to EU
    createArc(hubPositions[1], hubPositions[2]); // EU to Asia
    createArc(hubPositions[2], hubPositions[0]); // Asia to US

    // Slow rotation
    let rotationSpeed = 0.003;
    
    // Simple drag to rotate variables
    let isDragging = false;
    let previousMousePosition = { x: 0, y: 0 };

    const handleMouseDown = (e) => {
      isDragging = true;
      previousMousePosition = { x: e.clientX, y: e.clientY };
    };

    const handleMouseMove = (e) => {
      if (!isDragging) return;
      const deltaMove = {
        x: e.clientX - previousMousePosition.x,
        y: e.clientY - previousMousePosition.y
      };

      globeGroup.rotation.y += deltaMove.x * 0.005;
      globeGroup.rotation.x += deltaMove.y * 0.005;

      previousMousePosition = { x: e.clientX, y: e.clientY };
    };

    const handleMouseUp = () => {
      isDragging = false;
    };

    container.addEventListener('mousedown', handleMouseDown);
    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);

    // Window resize handler
    const handleResize = () => {
      if (!container) return;
      const w = container.clientWidth;
      camera.aspect = w / height;
      camera.updateProjectionMatrix();
      renderer.setSize(w, height);
    };
    
    window.addEventListener('resize', handleResize);

    // Animation loop
    let requestID;
    const animate = () => {
      requestID = requestAnimationFrame(animate);

      if (!isDragging) {
        globeGroup.rotation.y += rotationSpeed;
      }

      renderer.render(scene, camera);
    };
    
    animate();

    // Clean up
    return () => {
      cancelAnimationFrame(requestID);
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
      window.removeEventListener('resize', handleResize);
      if (container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement);
      }
      scene.clear();
      renderer.dispose();
    };
  }, [height]);

  return (
    <div 
      ref={containerRef} 
      className="globe-container cursor-grab active:cursor-grabbing"
      style={{ width: '100%', height: `${height}px`, overflow: 'hidden', position: 'relative' }} 
    />
  );
}
