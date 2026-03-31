import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
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

const CELL_EMISSIVE = {
  Stem:        new THREE.Color(0x444a5a),
  Sensory:     new THREE.Color(0x006a80),
  Associative: new THREE.Color(0x543e7d),
  Motor:       new THREE.Color(0x80361a),
  Modulatory:  new THREE.Color(0x1a6a4d),
  Fused:       new THREE.Color(0x7a395c),
};

const BALL_RADIUS = 12;
const NODE_BASE_SIZE = 0.15;
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

const firingHistory = [];
const MAX_HISTORY = 120;

// Three.js objects
let renderer, scene, camera, controls, composer, bloomPass;
let nodesMesh, edgesMesh, diskMesh;
let nodePositions = new Float32Array(MAX_NODES * 3);
let nodeData = [];      // current frame node data
let nodeMap = new Map(); // id -> index
let edgeData = [];       // current frame edge data

// Glow particles for fired nodes
let glowMesh;
const glowPositions = new Float32Array(MAX_NODES * 3);
const glowScales = new Float32Array(MAX_NODES);
const glowColors = new Float32Array(MAX_NODES * 3);

// Spike particles — light pulses traveling along connections
const MAX_SPIKES = 3000;
let spikesMesh;
const spikes = [];  // { fromIdx, toIdx, progress, color, speed }

// Raycasting
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();
raycaster.params.Points = { threshold: 0.5 };

// ============================================================
// THREE.JS SETUP
// ============================================================
function initScene() {
  const container = document.getElementById('scene-container');

  // Renderer
  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.2;
  container.appendChild(renderer.domElement);

  // Scene
  scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2(0x06080f, 0.012);

  // Camera
  camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.1, 200);
  camera.position.set(0, 8, 22);

  // Controls
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.minDistance = 5;
  controls.maxDistance = 60;
  controls.autoRotate = true;
  controls.autoRotateSpeed = 0.3;

  // Post-processing
  composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(scene, camera));
  bloomPass = new UnrealBloomPass(
    new THREE.Vector2(window.innerWidth, window.innerHeight),
    1.2,   // strength
    0.5,   // radius
    0.15   // threshold
  );
  composer.addPass(bloomPass);
  composer.addPass(new OutputPass());

  // Lights
  const ambient = new THREE.AmbientLight(0x1a2040, 0.6);
  scene.add(ambient);
  const point1 = new THREE.PointLight(0x508cff, 1.5, 50);
  point1.position.set(10, 10, 10);
  scene.add(point1);
  const point2 = new THREE.PointLight(0xa78bfa, 0.8, 50);
  point2.position.set(-10, -5, -10);
  scene.add(point2);

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
      color: 0x508cff,
      transparent: true,
      opacity: r === 1.0 ? 0.08 : 0.03,
      side: THREE.DoubleSide,
      depthWrite: false,
    });
    const ring = new THREE.Mesh(ringGeo, ringMat);
    scene.add(ring);
    // Add rotated copies for 3D effect
    const ring2 = ring.clone();
    ring2.rotation.x = Math.PI / 2;
    scene.add(ring2);
    const ring3 = ring.clone();
    ring3.rotation.y = Math.PI / 2;
    scene.add(ring3);
  }

  // Node instanced mesh
  const nodeGeo = new THREE.SphereGeometry(1, 12, 12);
  const nodeMat = new THREE.MeshStandardMaterial({
    roughness: 0.3,
    metalness: 0.4,
    emissive: 0x000000,
    emissiveIntensity: 0.5,
  });
  nodesMesh = new THREE.InstancedMesh(nodeGeo, nodeMat, MAX_NODES);
  nodesMesh.count = 0;
  nodesMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  scene.add(nodesMesh);

  // Glow instanced mesh (larger, emissive spheres for fired nodes)
  const glowGeo = new THREE.SphereGeometry(1, 8, 8);
  const glowMat = new THREE.MeshBasicMaterial({
    transparent: true,
    opacity: 0.25,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
  });
  glowMesh = new THREE.InstancedMesh(glowGeo, glowMat, MAX_NODES);
  glowMesh.count = 0;
  glowMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  scene.add(glowMesh);

  // Edge lines
  const edgePositions = new Float32Array(MAX_EDGES * 6);
  const edgeColors = new Float32Array(MAX_EDGES * 6);
  const edgeGeo = new THREE.BufferGeometry();
  edgeGeo.setAttribute('position', new THREE.BufferAttribute(edgePositions, 3).setUsage(THREE.DynamicDrawUsage));
  edgeGeo.setAttribute('color', new THREE.BufferAttribute(edgeColors, 3).setUsage(THREE.DynamicDrawUsage));
  const edgeMat = new THREE.LineBasicMaterial({
    vertexColors: true,
    transparent: true,
    opacity: 0.6,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
  });
  edgesMesh = new THREE.LineSegments(edgeGeo, edgeMat);
  scene.add(edgesMesh);

  // Spike particle mesh — small bright spheres that travel along edges
  const spikeGeo = new THREE.SphereGeometry(1, 6, 6);
  const spikeMat = new THREE.MeshBasicMaterial({
    transparent: true,
    opacity: 0.9,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
  });
  spikesMesh = new THREE.InstancedMesh(spikeGeo, spikeMat, MAX_SPIKES);
  spikesMesh.count = 0;
  spikesMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  scene.add(spikesMesh);

  // Resize handler
  window.addEventListener('resize', onResize);

  // Mouse handlers
  renderer.domElement.addEventListener('mousemove', onMouseMove);
  renderer.domElement.addEventListener('click', onMouseClick);
}

function onResize() {
  const w = window.innerWidth;
  const h = window.innerHeight;
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
    updateDetailPanel();
  }
}

function updateRaycast() {
  if (!nodesMesh || nodesMesh.count === 0) return;

  raycaster.setFromCamera(mouse, camera);

  // Manual sphere intersection for instanced mesh
  const dummy = new THREE.Matrix4();
  const pos = new THREE.Vector3();
  let closest = null;
  let closestDist = Infinity;

  for (let i = 0; i < nodesMesh.count; i++) {
    nodesMesh.getMatrixAt(i, dummy);
    pos.setFromMatrixPosition(dummy);
    const ray = raycaster.ray;
    const d = ray.distanceToPoint(pos);
    const scale = dummy.elements[0]; // x scale = radius
    if (d < scale * 2.5) {
      const camDist = camera.position.distanceTo(pos);
      if (camDist < closestDist) {
        closestDist = camDist;
        closest = i;
      }
    }
  }

  hoveredNodeId = closest !== null ? nodeData[closest]?.id ?? null : null;
  renderer.domElement.style.cursor = hoveredNodeId !== null ? 'pointer' : 'default';

  // Show tooltip
  const tooltip = document.getElementById('tooltip');
  if (hoveredNodeId !== null) {
    const node = nodeData.find(n => n.id === hoveredNodeId);
    if (node) {
      tooltip.style.display = 'block';
      const rect = renderer.domElement.getBoundingClientRect();
      const screenPos = pos.clone().project(camera);
      const sx = (screenPos.x * 0.5 + 0.5) * rect.width;
      const sy = (-screenPos.y * 0.5 + 0.5) * rect.height;
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

function updateScene() {
  if (!system) return;

  const topo = JSON.parse(system.topology_json());
  const nodes = topo.nodes;
  const edges = topo.edges;

  nodeData = nodes;
  nodeMap.clear();

  // === UPDATE NODES ===
  const nodeCount = Math.min(nodes.length, MAX_NODES);
  nodesMesh.count = nodeCount;
  let glowCount = 0;

  for (let i = 0; i < nodeCount; i++) {
    const n = nodes[i];
    nodeMap.set(n.id, i);

    // Position in Poincare ball -> 3D scene
    const px = n.x * BALL_RADIUS;
    const py = n.y * BALL_RADIUS;
    const pz = n.z * BALL_RADIUS;

    // Size based on energy
    const size = NODE_BASE_SIZE + n.energy * 0.2;

    dummy.position.set(px, py, pz);
    dummy.scale.setScalar(size);
    dummy.updateMatrix();
    nodesMesh.setMatrixAt(i, dummy.matrix);

    // Color by cell type
    const color = CELL_COLORS[n.cell_type] || CELL_COLORS.Stem;
    const emissive = CELL_EMISSIVE[n.cell_type] || CELL_EMISSIVE.Stem;

    if (n.fired) {
      // Bright white-ish when firing
      tempColor.copy(color).lerp(new THREE.Color(0xffffff), 0.6);
      nodesMesh.setColorAt(i, tempColor);

      // Add glow sphere
      if (glowCount < MAX_NODES) {
        dummy.scale.setScalar(size * 4.0);
        dummy.updateMatrix();
        glowMesh.setMatrixAt(glowCount, dummy.matrix);
        glowMesh.setColorAt(glowCount, color);
        glowCount++;
      }
    } else {
      nodesMesh.setColorAt(i, color);
    }

    // Highlight selected
    if (n.id === selectedNodeId) {
      tempColor.set(0xffffff);
      nodesMesh.setColorAt(i, tempColor);
      if (glowCount < MAX_NODES) {
        dummy.scale.setScalar(size * 3.0);
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

  // === UPDATE EDGES ===
  edgeData = edges;
  const positions = edgesMesh.geometry.attributes.position.array;
  const colors = edgesMesh.geometry.attributes.color.array;
  let edgeIdx = 0;

  for (let i = 0; i < edges.length && edgeIdx < MAX_EDGES; i++) {
    const e = edges[i];
    const fromIdx = nodeMap.get(e.from);
    const toIdx = nodeMap.get(e.to);
    if (fromIdx === undefined || toIdx === undefined) continue;

    const fi = fromIdx * 3;
    const ti = toIdx * 3;
    const ei = edgeIdx * 6;

    positions[ei]     = nodePositions[fi];
    positions[ei + 1] = nodePositions[fi + 1];
    positions[ei + 2] = nodePositions[fi + 2];
    positions[ei + 3] = nodePositions[ti];
    positions[ei + 4] = nodePositions[ti + 1];
    positions[ei + 5] = nodePositions[ti + 2];

    // Color edges as light filaments
    const w = Math.min(Math.abs(e.weight), 2.0) / 2.0;
    const elig = Math.min(Math.abs(e.eligibility || 0), 1.0);
    const baseBright = 0.06 + w * 0.2;

    // Eligibility makes edges "hot" — actively learning connections glow
    const heatBoost = elig * 0.4;

    // Highlight edges connected to selected/hovered node
    const isHighlighted = (e.from === selectedNodeId || e.to === selectedNodeId ||
                           e.from === hoveredNodeId || e.to === hoveredNodeId);

    let r, g, b;
    if (isHighlighted) {
      r = 0.5; g = 0.7; b = 1.0;
    } else if (e.weight >= 0) {
      // Cool blue base + warm shift from eligibility
      r = baseBright * 0.3 + heatBoost * 0.8;
      g = baseBright * 0.5 + heatBoost * 0.3;
      b = baseBright + heatBoost * 0.1;
    } else {
      // Inhibitory: reddish
      r = baseBright * 0.8 + heatBoost;
      g = baseBright * 0.15;
      b = baseBright * 0.2;
    }

    // Consolidated edges: steady warm glow (captured knowledge)
    if (e.consolidated) {
      r = r * 1.2 + 0.08;
      g = g * 1.2 + 0.06;
      b *= 1.1;
    }

    colors[ei] = r;     colors[ei + 1] = g;     colors[ei + 2] = b;
    colors[ei + 3] = r; colors[ei + 4] = g; colors[ei + 5] = b;

    edgeIdx++;
  }

  edgesMesh.geometry.attributes.position.needsUpdate = true;
  edgesMesh.geometry.attributes.color.needsUpdate = true;
  edgesMesh.geometry.setDrawRange(0, edgeIdx * 2);

  // === Spawn spike particles for newly firing morphons ===
  const newFired = new Set();
  for (const n of nodes) {
    if (n.fired) newFired.add(n.id);
  }

  // For each newly fired morphon, spawn particles along its outgoing edges
  for (const id of newFired) {
    if (prevFired.has(id)) continue;  // only on rising edge
    const fromIdx = nodeMap.get(id);
    if (fromIdx === undefined) continue;
    const color = CELL_COLORS[nodeData[fromIdx]?.cell_type] || CELL_COLORS.Stem;

    for (const e of edges) {
      if (e.from !== id) continue;
      const toIdx = nodeMap.get(e.to);
      if (toIdx === undefined) continue;
      if (spikes.length >= MAX_SPIKES) break;

      spikes.push({
        fromIdx,
        toIdx,
        progress: 0,
        speed: 0.03 + Math.random() * 0.03,
        color: color.clone(),
      });
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

  // Header
  document.getElementById('h-step').textContent = stats.step_count;
  document.getElementById('h-morphons').textContent = stats.total_morphons;
  document.getElementById('h-synapses').textContent = stats.total_synapses;

  // Stats
  document.getElementById('s-morphons').textContent = stats.total_morphons;
  document.getElementById('s-synapses').textContent = stats.total_synapses;
  document.getElementById('s-clusters').textContent = stats.fused_clusters;
  document.getElementById('s-gen').textContent = stats.max_generation;
  document.getElementById('s-firing').textContent = (stats.firing_rate * 100).toFixed(1) + '%';
  document.getElementById('s-energy').textContent = stats.avg_energy.toFixed(2);
  document.getElementById('s-error').textContent = stats.avg_prediction_error.toFixed(3);
  document.getElementById('s-wmem').textContent = stats.working_memory_items;

  // Cell type counts
  const counts = stats.differentiation_map || {};
  for (const type of ['Stem', 'Sensory', 'Associative', 'Motor', 'Modulatory', 'Fused']) {
    const el = document.getElementById('ct-' + type);
    if (el) el.textContent = counts[type] || 0;
  }

  // Neuromodulation
  setModBar('mod-reward', 'mod-reward-v', mod.reward);
  setModBar('mod-novelty', 'mod-novelty-v', mod.novelty);
  setModBar('mod-arousal', 'mod-arousal-v', mod.arousal);
  setModBar('mod-homeo', 'mod-homeo-v', mod.homeostasis);

  // Sparkline
  firingHistory.push(stats.firing_rate);
  if (firingHistory.length > MAX_HISTORY) firingHistory.shift();
  drawSparkline('spark-firing', firingHistory, '#508cff');

  // Update detail panel if selected
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
  const w = canvas.width;
  const h = canvas.height;

  ctx.clearRect(0, 0, w, h);
  if (data.length < 2) return;

  const max = Math.max(...data, 0.01);

  // Fill gradient
  ctx.beginPath();
  for (let i = 0; i < data.length; i++) {
    const x = (i / (data.length - 1)) * w;
    const y = h - (data[i] / max) * h * 0.85 - 2;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }
  const lastX = ((data.length - 1) / (data.length - 1)) * w;
  ctx.lineTo(lastX, h);
  ctx.lineTo(0, h);
  ctx.closePath();
  const grad = ctx.createLinearGradient(0, 0, 0, h);
  grad.addColorStop(0, color + '30');
  grad.addColorStop(1, color + '05');
  ctx.fillStyle = grad;
  ctx.fill();

  // Stroke
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
  if (selectedNodeId === null) {
    panel.classList.remove('visible');
    return;
  }

  const node = nodeData.find(n => n.id === selectedNodeId);
  if (!node) {
    panel.classList.remove('visible');
    selectedNodeId = null;
    return;
  }

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

  // Count connections
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
  const eventsEl = document.getElementById('events');

  const step = stats.step_count;
  const mc = stats.total_morphons;
  const sc = stats.total_synapses;

  if (lastMorphonCount > 0) {
    const mDiff = mc - lastMorphonCount;
    const sDiff = sc - lastSynapseCount;

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

  // Cap at 100 events
  while (eventsEl.children.length > 100) {
    eventsEl.removeChild(eventsEl.lastChild);
  }
}

// ============================================================
// INPUT PATTERNS
// ============================================================
function makeInput(pattern) {
  if (!system) return;
  const n = system.input_size();
  const input = new Array(n);

  switch (pattern) {
    case 'burst':
      for (let i = 0; i < n; i++) input[i] = 1.0;
      break;
    case 'pulse':
      for (let i = 0; i < n; i++) input[i] = i % 2 === 0 ? 1.0 : 0.0;
      break;
    case 'wave':
      for (let i = 0; i < n; i++) input[i] = Math.abs(Math.sin(i * 0.5));
      break;
    case 'noise':
      for (let i = 0; i < n; i++) input[i] = Math.random();
      break;
    default:
      for (let i = 0; i < n; i++) input[i] = 0.5 + Math.random() * 2.0;
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
      makeInput('noise');
      system.step();
      updateScene();
      updatePanels();
      detectEvents();
    }
  });

  document.getElementById('speed-slider').addEventListener('input', (e) => {
    stepsPerFrame = parseInt(e.target.value);
  });

  document.getElementById('btn-reset').addEventListener('click', () => {
    const program = document.getElementById('program-select').value;
    system = new WasmSystem(60, program, 3);
    selectedNodeId = null;
    hoveredNodeId = null;
    firingHistory.length = 0;
    lastMorphonCount = 0;
    lastSynapseCount = 0;
    document.getElementById('events').innerHTML = '';
    addEvent(0, `System reset [${program}]`, 'event-diff');
    // Warm up
    for (let i = 0; i < 20; i++) {
      makeInput('noise');
      system.step();
    }
  });

  // Signal injection
  document.getElementById('btn-reward').addEventListener('click', () => {
    if (system) { system.inject_reward(0.8); addEvent('', 'Reward injected (0.8)', 'event-birth'); }
  });
  document.getElementById('btn-novelty').addEventListener('click', () => {
    if (system) { system.inject_novelty(0.6); addEvent('', 'Novelty injected (0.6)', 'event-synapse'); }
  });
  document.getElementById('btn-arousal').addEventListener('click', () => {
    if (system) { system.inject_arousal(0.9); addEvent('', 'Arousal injected (0.9)', 'event-death'); }
  });

  // Feed patterns
  document.getElementById('feed-burst').addEventListener('click', () => makeInput('burst'));
  document.getElementById('feed-pulse').addEventListener('click', () => makeInput('pulse'));
  document.getElementById('feed-wave').addEventListener('click', () => makeInput('wave'));
  document.getElementById('feed-noise').addEventListener('click', () => makeInput('noise'));

  // Clear log
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

    if (s.progress >= 1.0) {
      spikes.splice(i, 1);
      continue;
    }

    // Interpolate position between source and target
    const fi = s.fromIdx * 3;
    const ti = s.toIdx * 3;
    const t = s.progress;
    const x = nodePositions[fi]     + (nodePositions[ti]     - nodePositions[fi])     * t;
    const y = nodePositions[fi + 1] + (nodePositions[ti + 1] - nodePositions[fi + 1]) * t;
    const z = nodePositions[fi + 2] + (nodePositions[ti + 2] - nodePositions[fi + 2]) * t;

    // Size: bright in the middle, fades at ends
    const sizeCurve = Math.sin(t * Math.PI);
    const size = 0.06 + sizeCurve * 0.1;

    spikeDummy.position.set(x, y, z);
    spikeDummy.scale.setScalar(size);
    spikeDummy.updateMatrix();

    if (alive < MAX_SPIKES) {
      spikesMesh.setMatrixAt(alive, spikeDummy.matrix);
      // Brighter color at the leading edge
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

function animate() {
  frameCount++;

  // Simulation
  if (running && system) {
    for (let i = 0; i < stepsPerFrame; i++) {
      // Feed random low-level stimulation to keep the system alive
      if (frameCount % 3 === 0) {
        makeInput('noise');
      }
      system.step();
    }
    updateScene();

    // Update panels at lower rate (every 3rd frame)
    if (frameCount % 3 === 0) {
      updatePanels();
      detectEvents();
    }
  }

  // Raycast at lower rate
  if (frameCount % 2 === 0) {
    updateRaycast();
  }

  // Animate spike particles every frame (even when paused, existing ones keep traveling)
  updateSpikes();

  // Subtle ball rotation for ambient feel
  if (diskMesh) {
    diskMesh.rotation.y += 0.0003;
    diskMesh.rotation.x += 0.0001;
  }

  // Adjust bloom based on spike activity (cheap proxy: spike count)
  if (bloomPass) {
    const activity = Math.min(spikes.length / 100, 1.0);
    bloomPass.strength = 0.8 + activity * 1.5;
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

  // Warm up
  for (let i = 0; i < 30; i++) {
    makeInput('noise');
    system.step();
  }

  updateScene();
  updatePanels();

  // Hide loading screen
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
