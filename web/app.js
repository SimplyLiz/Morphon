import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import init, { WasmSystem } from './pkg/morphon_core.js';

// ============================================================
// CONSTANTS
// ============================================================
const CELL_COLORS = {
  Stem:        new THREE.Color(0x8890a4),
  Sensory:     new THREE.Color(0x00d4ff),
  Associative: new THREE.Color(0xa78bfa),
  Motor:       new THREE.Color(0xff6b35),
  Modulatory:  new THREE.Color(0x34d399),
  Fused:       new THREE.Color(0xf472b6),
};

const BALL_RADIUS = 12;
const NODE_BASE_SIZE = 0.25;
const MAX_NODES = 2000;
const MAX_EDGES = 20000;

// ============================================================
// STATE
// ============================================================
let system = null;
let running = true;
let stepsPerFrame = 10;
let selectedNodeId = null;
let hoveredNodeId = null;
let prevFired = new Set();

// Connected node IDs for context dimming
let connectedToSelected = new Set();

const firingHistory = [];
const MAX_HISTORY = 120;

// Three.js objects
let renderer, scene, camera, controls, composer, bloomPass;
let nodesMesh, edgesMesh, diskMesh, fresnelBall;
let nodePositions = new Float32Array(MAX_NODES * 3);
let nodeData = [];
let nodeMap = new Map();
let edgeData = [];

// Glow particles for fired nodes
let glowMesh;

// Spike particles
const MAX_SPIKES = 3000;
let spikesMesh;
const spikes = [];

// Raycasting
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();

// ============================================================
// CUSTOM SHADERS
// ============================================================

// Fresnel shader for the Poincare ball boundary — "force field" look
const fresnelBallVertexShader = `
  varying vec3 vNormal;
  varying vec3 vViewDir;
  void main() {
    vec4 worldPos = modelMatrix * vec4(position, 1.0);
    vNormal = normalize(normalMatrix * normal);
    vViewDir = normalize(cameraPosition - worldPos.xyz);
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const fresnelBallFragmentShader = `
  uniform vec3 rimColor;
  uniform float rimPower;
  uniform float rimIntensity;
  uniform float time;
  varying vec3 vNormal;
  varying vec3 vViewDir;
  void main() {
    float rim = 1.0 - max(0.0, dot(vNormal, vViewDir));
    rim = pow(rim, rimPower) * rimIntensity;
    // Subtle pulse animation
    float pulse = 1.0 + 0.08 * sin(time * 0.5);
    float alpha = rim * pulse;
    gl_FragColor = vec4(rimColor, alpha);
  }
`;

// Vignette shader
const vignetteShader = {
  uniforms: {
    tDiffuse: { value: null },
    darkness: { value: 0.45 },
    offset: { value: 0.9 },
  },
  vertexShader: `
    varying vec2 vUv;
    void main() {
      vUv = uv;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `,
  fragmentShader: `
    uniform sampler2D tDiffuse;
    uniform float darkness;
    uniform float offset;
    varying vec2 vUv;
    void main() {
      vec4 texel = texture2D(tDiffuse, vUv);
      vec2 center = vUv - 0.5;
      float dist = length(center);
      float vig = smoothstep(offset, offset - 0.5, dist);
      texel.rgb *= mix(1.0 - darkness, 1.0, vig);
      gl_FragColor = texel;
    }
  `,
};

// ============================================================
// THREE.JS SETUP
// ============================================================
function initScene() {
  const container = document.getElementById('scene-container');

  // Renderer
  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.0;
  renderer.setClearColor(0x050510);
  container.appendChild(renderer.domElement);

  scene = new THREE.Scene();
  // Subtle fog for depth — distant elements fade slightly
  scene.fog = new THREE.FogExp2(0x050510, 0.006);

  // Camera
  camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.1, 500);
  camera.position.set(0, 8, 22);

  // Controls
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.minDistance = 5;
  controls.maxDistance = 60;
  controls.autoRotate = true;
  controls.autoRotateSpeed = 0.3;

  // === POST-PROCESSING ===
  composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(scene, camera));
  bloomPass = new UnrealBloomPass(
    new THREE.Vector2(window.innerWidth, window.innerHeight),
    0.8,   // strength
    0.4,   // radius
    0.82   // threshold — only bright/emissive objects bloom
  );
  composer.addPass(bloomPass);
  // Vignette — draws eye to center
  const vignettePass = new ShaderPass(vignetteShader);
  composer.addPass(vignettePass);
  composer.addPass(new OutputPass());

  // === 3-POINT LIGHTING ===
  scene.add(new THREE.AmbientLight(0x111122, 0.4));
  const keyLight = new THREE.DirectionalLight(0xffeedd, 0.7);
  keyLight.position.set(50, 80, 50);
  scene.add(keyLight);
  const fillLight = new THREE.DirectionalLight(0x8888ff, 0.25);
  fillLight.position.set(-50, -20, -50);
  scene.add(fillLight);
  const rimLight = new THREE.DirectionalLight(0xffffff, 0.35);
  rimLight.position.set(0, -50, -80);
  scene.add(rimLight);

  // === STARFIELD ===
  const starCount = 4000;
  const starPos = new Float32Array(starCount * 3);
  for (let i = 0; i < starCount; i++) {
    const r = 50 + Math.random() * 150;
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos(2 * Math.random() - 1);
    starPos[i * 3] = r * Math.sin(phi) * Math.cos(theta);
    starPos[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
    starPos[i * 3 + 2] = r * Math.cos(phi);
  }
  const starGeo = new THREE.BufferGeometry();
  starGeo.setAttribute('position', new THREE.BufferAttribute(starPos, 3));
  scene.add(new THREE.Points(starGeo, new THREE.PointsMaterial({
    color: 0x6688cc, size: 0.12, sizeAttenuation: true,
    transparent: true, opacity: 0.5, blending: THREE.AdditiveBlending, depthWrite: false,
  })));

  // Poincare ball boundary (transparent sphere)
  const ballGeo = new THREE.SphereGeometry(BALL_RADIUS, 64, 64);
  const ballMat = new THREE.MeshPhysicalMaterial({
    color: 0x1a2540,
    transparent: true,
    opacity: 0.04,
    roughness: 0.2,
    metalness: 0.1,
    side: THREE.BackSide,
    depthWrite: false,
  });
  diskMesh = new THREE.Mesh(ballGeo, ballMat);
  scene.add(diskMesh);

  // Wireframe rings for depth reference
  for (const r of [0.33, 0.66, 1.0]) {
    const ringGeo = new THREE.RingGeometry(BALL_RADIUS * r - 0.02, BALL_RADIUS * r + 0.02, 80);
    const ringMat = new THREE.MeshBasicMaterial({
      color: 0x508cff, transparent: true,
      opacity: r === 1.0 ? 0.08 : 0.03,
      side: THREE.DoubleSide, depthWrite: false,
    });
    const ring = new THREE.Mesh(ringGeo, ringMat);
    scene.add(ring);
    const r2 = ring.clone(); r2.rotation.x = Math.PI / 2; scene.add(r2);
    const r3 = ring.clone(); r3.rotation.y = Math.PI / 2; scene.add(r3);
  }

  // === NODE MESH — PBR with emissive for glow + 3D shading ===
  const nodeGeo = new THREE.IcosahedronGeometry(1, 3);
  const nodeMat = new THREE.MeshStandardMaterial({
    color: 0xffffff,
    emissive: 0xffffff,
    emissiveIntensity: 0.4,
    metalness: 0.25,
    roughness: 0.4,
  });
  nodesMesh = new THREE.InstancedMesh(nodeGeo, nodeMat, MAX_NODES);
  nodesMesh.count = 0;
  nodesMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  scene.add(nodesMesh);

  // === GLOW HALOS ===
  const glowGeo = new THREE.IcosahedronGeometry(1, 2);
  const glowMat = new THREE.MeshBasicMaterial({
    transparent: true, opacity: 0.15,
    blending: THREE.AdditiveBlending, depthWrite: false,
  });
  glowMesh = new THREE.InstancedMesh(glowGeo, glowMat, MAX_NODES);
  glowMesh.count = 0;
  glowMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  scene.add(glowMesh);

  // === EDGE LINES ===
  // Allocate 4 vertices per edge (2 segments for slight curve via midpoint offset)
  const edgePositions = new Float32Array(MAX_EDGES * 12); // 4 verts * 3 components
  const edgeColors = new Float32Array(MAX_EDGES * 12);
  const edgeGeo = new THREE.BufferGeometry();
  edgeGeo.setAttribute('position', new THREE.BufferAttribute(edgePositions, 3).setUsage(THREE.DynamicDrawUsage));
  edgeGeo.setAttribute('color', new THREE.BufferAttribute(edgeColors, 3).setUsage(THREE.DynamicDrawUsage));
  const edgeMat = new THREE.LineBasicMaterial({
    vertexColors: true, transparent: true, opacity: 0.3,
    blending: THREE.AdditiveBlending, depthWrite: false,
  });
  edgesMesh = new THREE.LineSegments(edgeGeo, edgeMat);
  scene.add(edgesMesh);

  // === SPIKE PARTICLES ===
  const spikeGeo = new THREE.SphereGeometry(1, 6, 6);
  const spikeMat = new THREE.MeshBasicMaterial({
    transparent: true, opacity: 0.85,
    blending: THREE.AdditiveBlending, depthWrite: false,
  });
  spikesMesh = new THREE.InstancedMesh(spikeGeo, spikeMat, MAX_SPIKES);
  spikesMesh.count = 0;
  spikesMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  scene.add(spikesMesh);

  window.addEventListener('resize', onResize);
  renderer.domElement.addEventListener('mousemove', onMouseMove);
  renderer.domElement.addEventListener('click', onMouseClick);
}

function onResize() {
  const w = window.innerWidth, h = window.innerHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
  composer.setSize(w, h);
  bloomPass.resolution.set(w, h);
}

// ============================================================
// MOUSE INTERACTION
// ============================================================
function onMouseMove(e) {
  mouse.x = (e.clientX / window.innerWidth) * 2 - 1;
  mouse.y = -(e.clientY / window.innerHeight) * 2 + 1;
}

function onMouseClick() {
  if (hoveredNodeId !== null) {
    selectedNodeId = hoveredNodeId;
    // Build set of connected node IDs for context dimming
    connectedToSelected.clear();
    connectedToSelected.add(selectedNodeId);
    for (const e of edgeData) {
      if (e.from === selectedNodeId) connectedToSelected.add(e.to);
      if (e.to === selectedNodeId) connectedToSelected.add(e.from);
    }
    updateDetailPanel();
  } else {
    // Click empty space — deselect
    selectedNodeId = null;
    connectedToSelected.clear();
    updateDetailPanel();
  }
}

function updateRaycast() {
  if (!nodesMesh || nodesMesh.count === 0) return;
  raycaster.setFromCamera(mouse, camera);

  const mat4 = new THREE.Matrix4();
  const pos = new THREE.Vector3();
  let closest = null;
  let closestDist = Infinity;

  for (let i = 0; i < nodesMesh.count; i++) {
    nodesMesh.getMatrixAt(i, mat4);
    pos.setFromMatrixPosition(mat4);
    const d = raycaster.ray.distanceToPoint(pos);
    const scale = mat4.elements[0];
    if (d < scale * 2.5) {
      const camDist = camera.position.distanceTo(pos);
      if (camDist < closestDist) { closestDist = camDist; closest = i; }
    }
  }

  hoveredNodeId = closest !== null ? nodeData[closest]?.id ?? null : null;
  renderer.domElement.style.cursor = hoveredNodeId !== null ? 'pointer' : 'default';

  const tooltip = document.getElementById('tooltip');
  if (hoveredNodeId !== null) {
    const node = nodeData.find(n => n.id === hoveredNodeId);
    if (node) {
      tooltip.style.display = 'block';
      const screenPos = pos.clone().project(camera);
      const sx = (screenPos.x * 0.5 + 0.5) * window.innerWidth;
      const sy = (-screenPos.y * 0.5 + 0.5) * window.innerHeight;
      tooltip.style.left = (sx + 16) + 'px';
      tooltip.style.top = (sy - 10) + 'px';
      tooltip.innerHTML = `
        <div class="tip-id">#${node.id}</div>
        <div class="tip-type">${node.cell_type} &middot; Gen ${node.generation}</div>
        <div class="tip-row"><span class="label">Energy</span><span class="value">${node.energy.toFixed(2)}</span></div>
        <div class="tip-row"><span class="label">Potential</span><span class="value">${node.potential.toFixed(3)}</span></div>
        <div class="tip-row"><span class="label">Fired</span><span class="value">${node.fired ? 'YES' : '-'}</span></div>
      `;
    }
  } else {
    tooltip.style.display = 'none';
  }
}

// ============================================================
// UPDATE SCENE FROM WASM DATA
// ============================================================
const dummy = new THREE.Object3D();
const tempColor = new THREE.Color();
const dimColor = new THREE.Color();

function updateScene() {
  if (!system) return;

  const topo = JSON.parse(system.topology_json());
  const nodes = topo.nodes;
  const edges = topo.edges;

  nodeData = nodes;
  edgeData = edges;
  nodeMap.clear();

  const nodeCount = Math.min(nodes.length, MAX_NODES);
  nodesMesh.count = nodeCount;
  let glowCount = 0;

  const hasSelection = selectedNodeId !== null && connectedToSelected.size > 0;

  for (let i = 0; i < nodeCount; i++) {
    const n = nodes[i];
    nodeMap.set(n.id, i);

    const px = n.x * BALL_RADIUS;
    const py = n.y * BALL_RADIUS;
    const pz = n.z * BALL_RADIUS;

    const size = NODE_BASE_SIZE + n.energy * 0.2;

    dummy.position.set(px, py, pz);
    dummy.scale.setScalar(size);
    dummy.updateMatrix();
    nodesMesh.setMatrixAt(i, dummy.matrix);

    const color = CELL_COLORS[n.cell_type] || CELL_COLORS.Stem;

    // === CONTEXT DIMMING ===
    // When a node is selected, dim non-connected nodes to 20%
    const isDimmed = hasSelection && !connectedToSelected.has(n.id);

    if (n.fired && !isDimmed) {
      tempColor.copy(color).lerp(new THREE.Color(0xffffff), 0.4);
      nodesMesh.setColorAt(i, tempColor);

      if (glowCount < MAX_NODES) {
        dummy.scale.setScalar(size * 3.0);
        dummy.updateMatrix();
        glowMesh.setMatrixAt(glowCount, dummy.matrix);
        glowMesh.setColorAt(glowCount, color);
        glowCount++;
      }
    } else if (isDimmed) {
      // Dim: desaturated and dark
      dimColor.copy(color).multiplyScalar(0.15);
      nodesMesh.setColorAt(i, dimColor);
    } else {
      nodesMesh.setColorAt(i, color);
    }

    // Selected node — bright highlight
    if (n.id === selectedNodeId) {
      tempColor.copy(color).lerp(new THREE.Color(0xffffff), 0.5);
      nodesMesh.setColorAt(i, tempColor);
      if (glowCount < MAX_NODES) {
        dummy.scale.setScalar(size * 2.5);
        dummy.updateMatrix();
        glowMesh.setMatrixAt(glowCount, dummy.matrix);
        tempColor.set(0x508cff);
        glowMesh.setColorAt(glowCount, tempColor);
        glowCount++;
      }
    }

    nodePositions[i * 3] = px;
    nodePositions[i * 3 + 1] = py;
    nodePositions[i * 3 + 2] = pz;
  }

  nodesMesh.instanceMatrix.needsUpdate = true;
  if (nodesMesh.instanceColor) nodesMesh.instanceColor.needsUpdate = true;

  glowMesh.count = glowCount;
  glowMesh.instanceMatrix.needsUpdate = true;
  if (glowMesh.instanceColor) glowMesh.instanceColor.needsUpdate = true;

  // === CURVED EDGES ===
  // Each edge becomes 2 line segments via a midpoint offset (subtle bezier approximation)
  const positions = edgesMesh.geometry.attributes.position.array;
  const colors = edgesMesh.geometry.attributes.color.array;
  let edgeIdx = 0;

  for (let i = 0; i < edges.length && edgeIdx < MAX_EDGES; i++) {
    const e = edges[i];
    const fromIdx = nodeMap.get(e.from);
    const toIdx = nodeMap.get(e.to);
    if (fromIdx === undefined || toIdx === undefined) continue;

    const w = Math.min(Math.abs(e.weight), 2.0) / 2.0;
    if (w < 0.04) continue;

    const fi = fromIdx * 3;
    const ti = toIdx * 3;
    const fx = nodePositions[fi], fy = nodePositions[fi+1], fz = nodePositions[fi+2];
    const tx = nodePositions[ti], ty = nodePositions[ti+1], tz = nodePositions[ti+2];

    // Midpoint with slight perpendicular offset for curve
    const mx = (fx + tx) * 0.5;
    const my = (fy + ty) * 0.5;
    const mz = (fz + tz) * 0.5;
    // Offset perpendicular to the edge and toward the ball center
    const dx = tx - fx, dy = ty - fy, dz = tz - fz;
    const len = Math.sqrt(dx*dx + dy*dy + dz*dz) || 1;
    // Cross product with a "up-ish" vector for perpendicular offset
    const cx = dy * 0.15 - dz * 0.05;
    const cy = dz * 0.1 - dx * 0.15;
    const cz = dx * 0.05 - dy * 0.1;
    const curveAmount = 0.15;
    const ox = mx + cx * curveAmount;
    const oy = my + cy * curveAmount;
    const oz = mz + cz * curveAmount;

    const ei = edgeIdx * 12;
    // Segment 1: from -> midpoint
    positions[ei] = fx;   positions[ei+1] = fy;   positions[ei+2] = fz;
    positions[ei+3] = ox;  positions[ei+4] = oy;  positions[ei+5] = oz;
    // Segment 2: midpoint -> to
    positions[ei+6] = ox;  positions[ei+7] = oy;  positions[ei+8] = oz;
    positions[ei+9] = tx;  positions[ei+10] = ty;  positions[ei+11] = tz;

    // Color
    const sourceColor = CELL_COLORS[nodeData[fromIdx]?.cell_type] || CELL_COLORS.Stem;
    const isHighlighted = hasSelection && (e.from === selectedNodeId || e.to === selectedNodeId);
    const edgeDimmed = hasSelection && !isHighlighted;

    let r, g, b;
    if (isHighlighted) {
      r = 0.5; g = 0.7; b = 1.0;
    } else {
      const brightness = edgeDimmed ? 0.02 : (0.05 + w * 0.18);
      r = sourceColor.r * brightness;
      g = sourceColor.g * brightness;
      b = sourceColor.b * brightness;
    }

    if (e.consolidated && !edgeDimmed) { r += 0.05; g += 0.04; b += 0.02; }

    // Apply same color to all 4 vertices
    for (let v = 0; v < 4; v++) {
      colors[ei + v*3] = r;
      colors[ei + v*3 + 1] = g;
      colors[ei + v*3 + 2] = b;
    }

    edgeIdx++;
  }

  edgesMesh.geometry.attributes.position.needsUpdate = true;
  edgesMesh.geometry.attributes.color.needsUpdate = true;
  edgesMesh.geometry.setDrawRange(0, edgeIdx * 4); // 4 verts per edge (2 segments)

  // === SPIKE SPAWNING ===
  const newFired = new Set();
  for (const n of nodes) { if (n.fired) newFired.add(n.id); }

  for (const id of newFired) {
    if (prevFired.has(id)) continue;
    if (spikes.length >= MAX_SPIKES * 0.7) break;
    const fromIdx = nodeMap.get(id);
    if (fromIdx === undefined) continue;
    const color = CELL_COLORS[nodeData[fromIdx]?.cell_type] || CELL_COLORS.Stem;

    let spawned = 0;
    for (const e of edges) {
      if (e.from !== id) continue;
      if (spawned >= 6) break;
      const toIdx = nodeMap.get(e.to);
      if (toIdx === undefined) continue;

      spikes.push({
        fromIdx, toIdx,
        progress: 0,
        speed: 0.02 + Math.random() * 0.025,
        color: color.clone(),
      });
      spawned++;
    }
  }

  prevFired = newFired;
}

// ============================================================
// UI UPDATES
// ============================================================
function updatePanels() {
  if (!system) return;
  const stats = JSON.parse(system.inspect());
  const mod = JSON.parse(system.modulation_json());

  document.getElementById('h-step').textContent = stats.step_count;
  document.getElementById('h-morphons').textContent = stats.total_morphons;
  document.getElementById('h-synapses').textContent = stats.total_synapses;

  document.getElementById('s-morphons').textContent = stats.total_morphons;
  document.getElementById('s-synapses').textContent = stats.total_synapses;
  document.getElementById('s-clusters').textContent = stats.fused_clusters;
  document.getElementById('s-gen').textContent = stats.max_generation;
  document.getElementById('s-firing').textContent = (stats.firing_rate * 100).toFixed(1) + '%';
  document.getElementById('s-energy').textContent = stats.avg_energy.toFixed(2);
  document.getElementById('s-error').textContent = stats.avg_prediction_error.toFixed(3);
  document.getElementById('s-wmem').textContent = stats.working_memory_items;

  const counts = stats.differentiation_map || {};
  for (const type of ['Stem', 'Sensory', 'Associative', 'Motor', 'Modulatory', 'Fused']) {
    const el = document.getElementById('ct-' + type);
    if (el) el.textContent = counts[type] || 0;
  }

  setModBar('mod-reward', 'mod-reward-v', mod.reward);
  setModBar('mod-novelty', 'mod-novelty-v', mod.novelty);
  setModBar('mod-arousal', 'mod-arousal-v', mod.arousal);
  setModBar('mod-homeo', 'mod-homeo-v', mod.homeostasis);

  firingHistory.push(stats.firing_rate);
  if (firingHistory.length > MAX_HISTORY) firingHistory.shift();
  drawSparkline('spark-firing', firingHistory, '#508cff');

  if (selectedNodeId !== null) updateDetailPanel();
}

function setModBar(barId, valId, value) {
  document.getElementById(barId).style.width = (value * 100) + '%';
  document.getElementById(valId).textContent = value.toFixed(2);
}

function drawSparkline(canvasId, data, color) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width, h = canvas.height;

  ctx.clearRect(0, 0, w, h);
  if (data.length < 2) return;
  const max = Math.max(...data, 0.01);

  ctx.beginPath();
  for (let i = 0; i < data.length; i++) {
    const x = (i / (data.length - 1)) * w;
    const y = h - (data[i] / max) * h * 0.85 - 2;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }
  ctx.lineTo(w, h); ctx.lineTo(0, h); ctx.closePath();
  const grad = ctx.createLinearGradient(0, 0, 0, h);
  grad.addColorStop(0, color + '30');
  grad.addColorStop(1, color + '05');
  ctx.fillStyle = grad;
  ctx.fill();

  ctx.beginPath();
  for (let i = 0; i < data.length; i++) {
    const x = (i / (data.length - 1)) * w;
    const y = h - (data[i] / max) * h * 0.85 - 2;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.stroke();
}

function updateDetailPanel() {
  const panel = document.getElementById('detail-panel');
  if (selectedNodeId === null) { panel.classList.remove('visible'); return; }
  const node = nodeData.find(n => n.id === selectedNodeId);
  if (!node) { panel.classList.remove('visible'); selectedNodeId = null; return; }

  panel.classList.add('visible');
  document.getElementById('d-id').textContent = '#' + node.id;
  document.getElementById('d-type').textContent = node.cell_type;
  document.getElementById('d-dot').style.background = CELL_COLORS[node.cell_type]?.getStyle() || '#888';
  document.getElementById('d-gen').textContent = node.generation;
  document.getElementById('d-age').textContent = node.age;
  document.getElementById('d-energy').textContent = node.energy.toFixed(2);
  document.getElementById('d-energy-bar').style.width = (node.energy * 100) + '%';
  document.getElementById('d-potential').textContent = node.potential.toFixed(3);
  document.getElementById('d-threshold').textContent = node.threshold.toFixed(3);
  document.getElementById('d-diff').textContent = node.differentiation.toFixed(2);
  document.getElementById('d-diff-bar').style.width = (node.differentiation * 100) + '%';
  document.getElementById('d-error').textContent = node.prediction_error.toFixed(3);
  document.getElementById('d-desire').textContent = node.desire.toFixed(3);
  document.getElementById('d-fired').textContent = node.fired ? 'YES' : '-';

  let inCount = 0, outCount = 0;
  for (const e of edgeData) {
    if (e.to === selectedNodeId) inCount++;
    if (e.from === selectedNodeId) outCount++;
  }
  document.getElementById('d-conns').textContent = `${inCount}\u2193 ${outCount}\u2191`;
}

// ============================================================
// EVENT LOG
// ============================================================
let lastMorphonCount = 0;
let lastSynapseCount = 0;

function detectEvents() {
  if (!system) return;
  const stats = JSON.parse(system.inspect());
  const step = stats.step_count;
  const mc = stats.total_morphons, sc = stats.total_synapses;

  if (lastMorphonCount > 0) {
    const mDiff = mc - lastMorphonCount, sDiff = sc - lastSynapseCount;
    if (mDiff > 0) addEvent(step, `+${mDiff} morphon(s) born`, 'event-birth');
    if (mDiff < 0) addEvent(step, `${mDiff} morphon(s) died`, 'event-death');
    if (sDiff > 5) addEvent(step, `+${sDiff} synapse(s) formed`, 'event-synapse');
    if (sDiff < -5) addEvent(step, `${sDiff} synapse(s) pruned`, 'event-synapse');
  }
  lastMorphonCount = mc;
  lastSynapseCount = sc;
}

function addEvent(step, text, cssClass) {
  const eventsEl = document.getElementById('events');
  const el = document.createElement('div');
  el.className = 'event-item';
  el.innerHTML = `<span class="${cssClass}">[${step}]</span> ${text}`;
  eventsEl.insertBefore(el, eventsEl.firstChild);
  while (eventsEl.children.length > 100) eventsEl.removeChild(eventsEl.lastChild);
}

// ============================================================
// INPUT PATTERNS
// ============================================================
function makeInput(pattern) {
  if (!system) return;
  const n = system.input_size();
  const input = new Array(n);
  switch (pattern) {
    case 'burst': for (let i = 0; i < n; i++) input[i] = 1.0; break;
    case 'pulse': for (let i = 0; i < n; i++) input[i] = i % 2 === 0 ? 1.0 : 0.0; break;
    case 'wave':  for (let i = 0; i < n; i++) input[i] = Math.abs(Math.sin(i * 0.5)); break;
    case 'noise': for (let i = 0; i < n; i++) input[i] = Math.random(); break;
    default:      for (let i = 0; i < n; i++) input[i] = 0.5 + Math.random() * 2.0;
  }
  system.feed_input(new Float64Array(input));
}

// ============================================================
// CONTROLS
// ============================================================
function setupControls() {
  const pauseBtn = document.getElementById('btn-pause');
  pauseBtn.addEventListener('click', () => {
    running = !running;
    pauseBtn.innerHTML = running ? '&#9646;&#9646;' : '&#9654;';
    pauseBtn.classList.toggle('active', running);
    controls.autoRotate = running;
  });

  document.getElementById('btn-step').addEventListener('click', () => {
    if (system) {
      makeInput('noise'); system.step();
      updateScene(); updatePanels(); detectEvents();
    }
  });

  document.getElementById('speed-slider').addEventListener('input', (e) => {
    stepsPerFrame = parseInt(e.target.value);
  });

  document.getElementById('btn-reset').addEventListener('click', () => {
    const program = document.getElementById('program-select').value;
    system = new WasmSystem(60, program, 3);
    selectedNodeId = null; hoveredNodeId = null;
    connectedToSelected.clear();
    firingHistory.length = 0;
    lastMorphonCount = 0; lastSynapseCount = 0;
    document.getElementById('events').innerHTML = '';
    addEvent(0, `System reset [${program}]`, 'event-diff');
    for (let i = 0; i < 20; i++) { makeInput('noise'); system.step(); }
  });

  document.getElementById('btn-reward').addEventListener('click', () => {
    if (system) { system.inject_reward(0.8); addEvent('', 'Reward injected (0.8)', 'event-birth'); }
  });
  document.getElementById('btn-novelty').addEventListener('click', () => {
    if (system) { system.inject_novelty(0.6); addEvent('', 'Novelty injected (0.6)', 'event-synapse'); }
  });
  document.getElementById('btn-arousal').addEventListener('click', () => {
    if (system) { system.inject_arousal(0.9); addEvent('', 'Arousal injected (0.9)', 'event-death'); }
  });

  document.getElementById('feed-burst').addEventListener('click', () => makeInput('burst'));
  document.getElementById('feed-pulse').addEventListener('click', () => makeInput('pulse'));
  document.getElementById('feed-wave').addEventListener('click', () => makeInput('wave'));
  document.getElementById('feed-noise').addEventListener('click', () => makeInput('noise'));

  document.getElementById('btn-clear-log').addEventListener('click', () => {
    document.getElementById('events').innerHTML = '';
  });
}

// ============================================================
// SPIKE PARTICLE ANIMATION
// ============================================================
const spikeDummy = new THREE.Object3D();

function updateSpikes() {
  let alive = 0;
  for (let i = spikes.length - 1; i >= 0; i--) {
    const s = spikes[i];
    s.progress += s.speed;
    if (s.progress >= 1.0) { spikes.splice(i, 1); continue; }

    const fi = s.fromIdx * 3, ti = s.toIdx * 3;
    const t = s.progress;
    const x = nodePositions[fi]   + (nodePositions[ti]   - nodePositions[fi])   * t;
    const y = nodePositions[fi+1] + (nodePositions[ti+1] - nodePositions[fi+1]) * t;
    const z = nodePositions[fi+2] + (nodePositions[ti+2] - nodePositions[fi+2]) * t;

    const sizeCurve = Math.sin(t * Math.PI);
    spikeDummy.position.set(x, y, z);
    spikeDummy.scale.setScalar(0.05 + sizeCurve * 0.08);
    spikeDummy.updateMatrix();

    if (alive < MAX_SPIKES) {
      spikesMesh.setMatrixAt(alive, spikeDummy.matrix);
      tempColor.copy(s.color).lerp(new THREE.Color(0xffffff), sizeCurve * 0.5);
      spikesMesh.setColorAt(alive, tempColor);
      alive++;
    }
  }

  spikesMesh.count = alive;
  if (alive > 0) {
    spikesMesh.instanceMatrix.needsUpdate = true;
    if (spikesMesh.instanceColor) spikesMesh.instanceColor.needsUpdate = true;
  }
}

// ============================================================
// ANIMATION LOOP
// ============================================================
let frameCount = 0;
const clock = new THREE.Clock();

function animate() {
  frameCount++;
  const elapsed = clock.getElapsedTime();

  if (running && system) {
    for (let i = 0; i < stepsPerFrame; i++) {
      if (frameCount % 3 === 0) makeInput('noise');
      system.step();
    }
    updateScene();
    if (frameCount % 3 === 0) { updatePanels(); detectEvents(); }
  }

  if (frameCount % 2 === 0) updateRaycast();
  updateSpikes();

  // Subtle ball rotation
  if (diskMesh) {
    diskMesh.rotation.y = elapsed * 0.02;
    diskMesh.rotation.x = elapsed * 0.008;
  }

  // Dynamic bloom
  if (bloomPass) {
    const activity = Math.min(spikes.length / 80, 1.0);
    bloomPass.strength = 0.7 + activity * 0.8;
  }

  controls.update();
  composer.render();
}

// ============================================================
// INIT
// ============================================================
async function main() {
  initScene();
  setupControls();
  await init();

  system = new WasmSystem(60, 'cortical', 3);
  for (let i = 0; i < 30; i++) { makeInput('noise'); system.step(); }

  updateScene();
  updatePanels();

  const loading = document.getElementById('loading');
  loading.classList.add('hidden');
  setTimeout(() => loading.remove(), 600);

  addEvent(0, 'System initialized [cortical, 60 seed, 3D]', 'event-diff');
  renderer.setAnimationLoop(animate);
}

main().catch(e => {
  console.error(e);
  document.getElementById('loading').querySelector('h2').textContent = 'ERROR: ' + e.message;
});
