"use client";

import { useEffect, useRef } from "react";

export function DataFlowCanvas() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Check reduced motion preference
    const prefersReducedMotion = window.matchMedia(
      "(prefers-reduced-motion: reduce)"
    ).matches;

    let animationFrameId: number;
    let width = (canvas.width = canvas.parentElement?.clientWidth || 800);
    let height = (canvas.height = canvas.parentElement?.clientHeight || 200);

    const isMobile = window.innerWidth < 768;
    const particleCount = isMobile ? 16 : 32;
    const maxDistance = isMobile ? 85 : 125;

    interface Node {
      x: number;
      y: number;
      vx: number;
      vy: number;
      radius: number;
      pulse: number;
      pulseSpeed: number;
    }

    const nodes: Node[] = Array.from({ length: particleCount }, () => ({
      x: Math.random() * width,
      y: Math.random() * height,
      vx: (Math.random() - 0.5) * 0.35,
      vy: (Math.random() - 0.5) * 0.35,
      radius: Math.random() * 1.5 + 1,
      pulse: Math.random() * Math.PI,
      pulseSpeed: 0.015 + Math.random() * 0.015,
    }));

    const resize = () => {
      if (!canvas || !canvas.parentElement) return;
      width = canvas.width = canvas.parentElement.clientWidth;
      height = canvas.height = canvas.parentElement.clientHeight;
    };

    window.addEventListener("resize", resize);

    const render = () => {
      ctx.clearRect(0, 0, width, height);

      // Draw node connections
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const dx = nodes[i].x - nodes[j].x;
          const dy = nodes[i].y - nodes[j].y;
          const dist = Math.sqrt(dx * dx + dy * dy);

          if (dist < maxDistance) {
            const alpha = (1 - dist / maxDistance) * 0.2;
            ctx.beginPath();
            ctx.moveTo(nodes[i].x, nodes[i].y);
            ctx.lineTo(nodes[j].x, nodes[j].y);
            ctx.strokeStyle = `rgba(118,255,3,${alpha})`; // brand-500 (#76FF03)
            ctx.lineWidth = 1;
            ctx.stroke();

            // Render subtle pulse traveling along vector
            if (!prefersReducedMotion && (i + j) % 3 === 0) {
              const pulsePos = (Math.sin(nodes[i].pulse) + 1) / 2;
              const px = nodes[i].x + (nodes[j].x - nodes[i].x) * pulsePos;
              const py = nodes[i].y + (nodes[j].y - nodes[i].y) * pulsePos;

              ctx.beginPath();
              ctx.arc(px, py, 1.2, 0, Math.PI * 2);
              ctx.fillStyle = `rgba(140,255,32,${alpha * 1.6})`; // brand-400 (#8CFF20)
              ctx.fill();
            }
          }
        }
      }

      // Draw nodes
      for (const node of nodes) {
        ctx.beginPath();
        ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(140,255,32,0.4)"; // brand-400
        ctx.fill();

        if (!prefersReducedMotion) {
          node.x += node.vx;
          node.y += node.vy;
          node.pulse += node.pulseSpeed;

          // Bounce off boundaries
          if (node.x < 0 || node.x > width) node.vx *= -1;
          if (node.y < 0 || node.y > height) node.vy *= -1;
        }
      }

      if (!prefersReducedMotion) {
        animationFrameId = requestAnimationFrame(render);
      }
    };

    render();

    return () => {
      window.removeEventListener("resize", resize);
      if (animationFrameId) cancelAnimationFrame(animationFrameId);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 w-full h-full pointer-events-none -z-0 opacity-50"
    />
  );
}

export default DataFlowCanvas;
