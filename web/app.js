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
let frameFired = new Set(); // accumulates all fired IDs across multi-step frames
let lastSpikeCount = 0; // for dynamic bloom calculation

// Connected node IDs for context dimming
let connectedToSelected = new Set();
// Cell type filter: null = none, string = highlight that type
let filterCellType = null;
// Per-node dim factor: 0 = full color, 1 = dimmed. Keyed by morphon ID for stable mapping.
const nodeDim = new Map();
const nodeDimTarget = new Map();
// Per-node glow: approaches 1.0 on fire, decays toward 0. Keyed by morphon ID.
const nodeGlow = new Map();

const firingHistory = [];
const morphonHistory = [];
const MAX_HISTORY = 120;

// === RASTER PLOT STATE ===
const RASTER_WINDOW = 600; // rolling window in simulation steps
const CELL_TYPE_ORDER = ['Sensory', 'Associative', 'Motor', 'Modulatory', 'Stem', 'Fused'];
const RASTER_COLORS = {
  Stem:        [0x88, 0x90, 0xa4],
  Sensory:     [0x00, 0xd4, 0xff],
  Associative: [0xa7, 0x8b, 0xfa],
  Motor:       [0xff, 0x6b, 0x35],
  Modulatory:  [0x34, 0xd3, 0x99],
  Fused:       [0xf4, 0x72, 0xb6],
};
let morphonOrder = [];       // sorted [{id, cellType}] for Y-axis
let morphonYMap = new Map(); // id → Y row
let rasterCanvas, rasterCtx;
let rasterScrollX = 0;
let rasterMorphonCount = 0;  // cached to detect topology changes
let stepAccumulator = 0;     // for sub-1x speed

// Cached stats to avoid double inspect() calls
let cachedStats = null;
// Saved state for save/load
let savedState = null;

// Three.js objects
let renderer, scene, camera, controls, composer, bloomPass;
let nodesMesh, glowMesh, edgesMesh, diskMesh, fresnelBall;
let nodePositions = new Float32Array(MAX_NODES * 3);
let nodeData = [];
let nodeMap = new Map();
let edgeData = [];

// (glow is now done via emissive brightness + bloom, no separate mesh)

// Spike particles — JS-side animated, fed from real engine firing data
const MAX_SPIKES = 3000;
const SPIKE_VISUAL_FRAMES = 30; // visual lifetime in frames (~0.5s at 60fps)
const SPIKE_COOLDOWN = 12;       // frames between spawns per morphon
let spikesMesh;
const liveSpikes = [];
const spikeCooldowns = new Map(); // morphon id → frames remaining

// Raycasting
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();
// Reusable objects to avoid per-frame allocations
const _mat4 = new THREE.Matrix4();
const _pos = new THREE.Vector3();
const _closestPos = new THREE.Vector3();
const WHITE = new THREE.Color(0xffffff);
// Track mouse drag to distinguish click from orbit drag
let mouseDownPos = { x: 0, y: 0 };
// Track last user input time to avoid noise overwrite
let lastUserInputTime = 0;

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
  renderer = new THREE.WebGLRenderer({ antialias: true, logarithmicDepthBuffer: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  const sceneH = window.innerHeight - (document.getElementById('bottom-panel')?.offsetHeight || 160);
  renderer.setSize(window.innerWidth, sceneH);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.0;
  renderer.setClearColor(0x050510);
  container.appendChild(renderer.domElement);

  scene = new THREE.Scene();
  // Subtle fog for depth — distant elements fade slightly
  scene.fog = new THREE.FogExp2(0x050510, 0.006);

  // Camera
  camera = new THREE.PerspectiveCamera(55, window.innerWidth / sceneH, 1.0, 200);
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
    0.45,  // strength
    0.4,   // radius — soft halos
    0.75   // threshold
  );
  composer.addPass(bloomPass);
  // Vignette — draws eye to center
  const vignettePass = new ShaderPass(vignetteShader);
  composer.addPass(vignettePass);
  composer.addPass(new OutputPass());

  // === 3-POINT LIGHTING ===
  scene.add(new THREE.AmbientLight(0xffffff, 0.6));
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

  // === NODE MESH (diffuse — 3D shading, no bloom) ===
  const nodeGeo = new THREE.IcosahedronGeometry(1, 3);
  const nodeMat = new THREE.MeshStandardMaterial({
    color: 0xffffff,
    emissive: 0x080810,
    metalness: 0.0,
    roughness: 0.7,
  });
  nodesMesh = new THREE.InstancedMesh(nodeGeo, nodeMat, MAX_NODES);
  nodesMesh.count = 0;
  nodesMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  nodesMesh.frustumCulled = false;
  scene.add(nodesMesh);

  // === GLOW MESH (emissive overlay — drives bloom for active nodes) ===
  const glowMat = new THREE.MeshBasicMaterial({
    color: 0xffffff,
    transparent: true,
    opacity: 0.6,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
    depthTest: false,
  });
  glowMesh = new THREE.InstancedMesh(nodeGeo, glowMat, MAX_NODES);
  glowMesh.count = 0;
  glowMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  glowMesh.frustumCulled = false;
  scene.add(glowMesh);

  // === EDGE LINES ===
  // Allocate 4 vertices per edge (2 segments for slight curve via midpoint offset)
  const edgePositions = new Float32Array(MAX_EDGES * 12); // 4 verts * 3 components
  const edgeColors = new Float32Array(MAX_EDGES * 12);
  const edgeGeo = new THREE.BufferGeometry();
  edgeGeo.setAttribute('position', new THREE.BufferAttribute(edgePositions, 3).setUsage(THREE.DynamicDrawUsage));
  edgeGeo.setAttribute('color', new THREE.BufferAttribute(edgeColors, 3).setUsage(THREE.DynamicDrawUsage));
  const edgeMat = new THREE.LineBasicMaterial({
    vertexColors: true, transparent: true, opacity: 0.25,
    blending: THREE.AdditiveBlending, depthWrite: false,
  });
  edgesMesh = new THREE.LineSegments(edgeGeo, edgeMat);
  scene.add(edgesMesh);

  // === SPIKE PARTICLES ===
  const spikeGeo = new THREE.SphereGeometry(1, 6, 6);
  const spikeMat = new THREE.MeshBasicMaterial({
    transparent: true, opacity: 0.85,
    blending: THREE.AdditiveBlending, depthWrite: false, depthTest: false,
  });
  spikesMesh = new THREE.InstancedMesh(spikeGeo, spikeMat, MAX_SPIKES);
  spikesMesh.count = 0;
  spikesMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  scene.add(spikesMesh);

  window.addEventListener('resize', onResize);
  renderer.domElement.addEventListener('mousemove', onMouseMove);
  renderer.domElement.addEventListener('mousedown', (e) => { mouseDownPos.x = e.clientX; mouseDownPos.y = e.clientY; });
  renderer.domElement.addEventListener('click', onMouseClick);
}

function onResize() {
  const bottomPanel = document.getElementById('bottom-panel');
  const rasterH = bottomPanel ? bottomPanel.offsetHeight : 160;
  const w = window.innerWidth, h = window.innerHeight - rasterH;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
  composer.setSize(w, h);
  bloomPass.resolution.set(w, h);
  resizeRasterCanvas();
}

// ============================================================
// MOUSE INTERACTION
// ============================================================
function onMouseMove(e) {
  mouse.x = (e.clientX / window.innerWidth) * 2 - 1;
  mouse.y = -(e.clientY / window.innerHeight) * 2 + 1;
}

function onMouseClick(e) {
  // Ignore clicks that were actually orbit drags
  const dx = e.clientX - mouseDownPos.x, dy = e.clientY - mouseDownPos.y;
  if (dx * dx + dy * dy > 25) return; // moved more than 5px = drag
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

  let closest = null;
  let closestDist = Infinity;

  for (let i = 0; i < nodesMesh.count; i++) {
    nodesMesh.getMatrixAt(i, _mat4);
    _pos.setFromMatrixPosition(_mat4);
    const d = raycaster.ray.distanceToPoint(_pos);
    const scale = _mat4.elements[0];
    if (d < scale * 2.5) {
      const camDist = camera.position.distanceTo(_pos);
      if (camDist < closestDist) {
        closestDist = camDist;
        closest = i;
        _closestPos.copy(_pos); // save position of the actual closest node
      }
    }
  }

  hoveredNodeId = closest !== null ? nodeData[closest]?.id ?? null : null;
  renderer.domElement.style.cursor = hoveredNodeId !== null ? 'pointer' : 'default';

  const tooltip = document.getElementById('tooltip');
  if (hoveredNodeId !== null) {
    const node = nodeData[nodeMap.get(hoveredNodeId)]; // O(1) lookup via nodeMap
    if (node) {
      tooltip.style.display = 'block';
      const screenPos = _closestPos.clone().project(camera);
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

  // Sort nodes by ID for stable frame-to-frame ordering (HashMap iteration is non-deterministic)
  nodes.sort((a, b) => a.id - b.id);
  nodeData = nodes;
  edgeData = edges;
  nodeMap.clear();

  const nodeCount = Math.min(nodes.length, MAX_NODES);
  nodesMesh.count = nodeCount;

  const hasSelection = selectedNodeId !== null && connectedToSelected.size > 0;
  // Scale nodes down as population grows to prevent z-fighting from overlapping spheres
  // At 100: 1.0×, at 500: 0.45×, at 1000: 0.32×, at 2000: 0.22×
  const popScale = Math.min(1.0, 10.0 / Math.sqrt(nodeCount));

  for (let i = 0; i < nodeCount; i++) {
    const n = nodes[i];
    nodeMap.set(n.id, i);

    let px = n.x * BALL_RADIUS;
    let py = n.y * BALL_RADIUS;
    let pz = n.z * BALL_RADIUS;
    if (!isFinite(px)) px = 0;
    if (!isFinite(py)) py = 0;
    if (!isFinite(pz)) pz = 0;

    const energy = Math.max(0, Math.min(2, isFinite(n.energy) ? n.energy : 0));
    const size = (NODE_BASE_SIZE + energy * 0.2) * popScale;

    dummy.position.set(px, py, pz);
    dummy.scale.setScalar(size);
    dummy.updateMatrix();
    nodesMesh.setMatrixAt(i, dummy.matrix);

    const color = CELL_COLORS[n.cell_type] || CELL_COLORS.Stem;

    // === GLOW: only active or near-threshold morphons ===
    const prevGlow = nodeGlow.get(n.id) || 0;
    // Pre-glow: morphons approaching threshold get a subtle hint
    const nearThreshold = (n.potential > n.threshold * 0.6 && !n.fired) ? 0.15 : 0.0;
    const glowTarget = frameFired.has(n.id) ? 1.0 : nearThreshold;
    const glow = glowTarget > prevGlow
      ? prevGlow + (glowTarget - prevGlow) * 0.35
      : prevGlow * 0.92;
    nodeGlow.set(n.id, glow);

    // === SMOOTH CONTEXT DIMMING (selection OR cell type filter) ===
    let shouldDim = false;
    if (hasSelection) {
      shouldDim = !connectedToSelected.has(n.id);
    } else if (filterCellType) {
      shouldDim = n.cell_type !== filterCellType;
    }
    const dimTarget = shouldDim ? 1.0 : 0.0;
    const prevDim = nodeDim.get(n.id) || 0;
    const dim = prevDim + (dimTarget - prevDim) * 0.1;
    nodeDim.set(n.id, dim);
    const bright = 1.0 - dim * 0.85;

    if (n.id === selectedNodeId) {
      tempColor.copy(color).multiplyScalar(2.0);
      nodesMesh.setColorAt(i, tempColor);
    } else {
      // Diffuse layer: natural cell color with subtle brightening on fire
      const intensity = bright * (0.55 + glow * 0.4);
      tempColor.copy(color).multiplyScalar(intensity);
      nodesMesh.setColorAt(i, tempColor);
    }

    nodePositions[i * 3] = px;
    nodePositions[i * 3 + 1] = py;
    nodePositions[i * 3 + 2] = pz;
  }

  // === GLOW OVERLAY — additive emissive layer for active/near-active nodes ===
  let glowCount = 0;
  for (let i = 0; i < nodeCount; i++) {
    const n = nodes[i];
    const glow = nodeGlow.get(n.id) || 0;
    if (glow < 0.02) continue; // skip fully resting nodes

    const px = nodePositions[i * 3];
    const py = nodePositions[i * 3 + 1];
    const pz = nodePositions[i * 3 + 2];
    const energy = Math.max(0, Math.min(2, isFinite(n.energy) ? n.energy : 0));
    const baseSize = (NODE_BASE_SIZE + energy * 0.2) * popScale;

    // Glow sphere: slightly larger than the node, intensity = glow level
    dummy.position.set(px, py, pz);
    dummy.scale.setScalar(baseSize * (1.0 + glow * 0.4)); // swell slightly when glowing
    dummy.updateMatrix();
    glowMesh.setMatrixAt(glowCount, dummy.matrix);

    const color = CELL_COLORS[n.cell_type] || CELL_COLORS.Stem;
    tempColor.copy(color).multiplyScalar(glow * 1.5); // drives bloom when glow > ~0.5
    glowMesh.setColorAt(glowCount, tempColor);
    glowCount++;
  }
  glowMesh.count = glowCount;
  if (glowCount > 0) {
    glowMesh.instanceMatrix.needsUpdate = true;
    if (glowMesh.instanceColor) glowMesh.instanceColor.needsUpdate = true;
  }

  nodesMesh.instanceMatrix.needsUpdate = true;
  if (nodesMesh.instanceColor) nodesMesh.instanceColor.needsUpdate = true;

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
    // Smooth edge dimming: use the avg dim of both endpoints
    const edgeDimFactor = ((nodeDim.get(e.from) || 0) + (nodeDim.get(e.to) || 0)) * 0.5;

    // Synapse heat: eligibility trace reflects recent spike traffic
    const heat = Math.min(Math.abs(e.eligibility || 0), 1.0);

    let r, g, b;
    if (isHighlighted) {
      r = 0.5; g = 0.7; b = 1.0;
    } else {
      // Base brightness from weight + heat boost from recent activity
      const brightness = (0.05 + w * 0.15 + heat * 0.25) * (1.0 - edgeDimFactor * 0.9);
      r = sourceColor.r * brightness;
      g = sourceColor.g * brightness;
      b = sourceColor.b * brightness;
      // Hot synapses shift toward white
      if (heat > 0.15) {
        const heatWhite = heat * 0.15;
        r += heatWhite; g += heatWhite; b += heatWhite;
      }
    }

    if (e.consolidated && edgeDimFactor < 0.3) { r += 0.05; g += 0.04; b += 0.02; }

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

  // === SPIKE SPAWNING from real engine firing data ===
  // One representative spike per firing morphon, with cooldown to prevent stacking.
  // Tick down all cooldowns
  for (const [id, cd] of spikeCooldowns) {
    if (cd <= 1) spikeCooldowns.delete(id);
    else spikeCooldowns.set(id, cd - 1);
  }
  for (const id of frameFired) {
    if (liveSpikes.length >= MAX_SPIKES * 0.7) break;
    if (spikeCooldowns.has(id)) continue; // still on cooldown
    const fromIdx = nodeMap.get(id);
    if (fromIdx === undefined) continue;
    const color = CELL_COLORS[nodeData[fromIdx]?.cell_type] || CELL_COLORS.Stem;

    // Pick the strongest outgoing edge as the representative
    let bestEdge = null, bestWeight = -1;
    for (const e of edges) {
      if (e.from !== id) continue;
      const aw = Math.abs(e.weight);
      if (aw > bestWeight) {
        const toIdx = nodeMap.get(e.to);
        if (toIdx !== undefined) { bestEdge = { toIdx, weight: aw }; bestWeight = aw; }
      }
    }
    if (bestEdge) {
      liveSpikes.push({
        fromIdx, toIdx: bestEdge.toIdx,
        age: 0,
        color: color.clone(),
        strength: bestEdge.weight,
      });
      spikeCooldowns.set(id, SPIKE_COOLDOWN);
    }
  }
}

// ============================================================
// UI UPDATES
// ============================================================
function updatePanels() {
  if (!system) return;
  // Cache stats — used by both updatePanels and detectEvents
  cachedStats = JSON.parse(system.inspect());
  const stats = cachedStats;
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
  document.getElementById('s-born').textContent = stats.total_born || 0;
  document.getElementById('s-died').textContent = stats.total_died || 0;
  document.getElementById('s-transdiff').textContent = stats.total_transdifferentiations || 0;

  const counts = stats.differentiation_map || {};
  for (const type of ['Stem', 'Sensory', 'Associative', 'Motor', 'Modulatory', 'Fused']) {
    const el = document.getElementById('ct-' + type);
    if (el) el.textContent = counts[type] || 0;
  }

  setModBar('mod-reward', 'mod-reward-v', mod.reward);
  setModBar('mod-novelty', 'mod-novelty-v', mod.novelty);
  setModBar('mod-arousal', 'mod-arousal-v', mod.arousal);
  setModBar('mod-homeo', 'mod-homeo-v', mod.homeostasis);

  // Sparklines
  firingHistory.push(stats.firing_rate);
  if (firingHistory.length > MAX_HISTORY) firingHistory.shift();
  drawSparkline('spark-firing', firingHistory, '#508cff');

  morphonHistory.push(stats.total_morphons);
  if (morphonHistory.length > MAX_HISTORY) morphonHistory.shift();
  drawSparkline('spark-morphons', morphonHistory, '#34d399');

  // Motor output bar chart
  updateMotorOutput();

  if (selectedNodeId !== null) updateDetailPanel();
}

function updateMotorOutput() {
  if (!system) return;
  const output = system.read_output();
  const container = document.getElementById('motor-output');
  const count = output.length;

  // Build/update bars (reuse DOM if count matches)
  if (container.children.length !== count) {
    container.innerHTML = '';
    for (let i = 0; i < count; i++) {
      const bar = document.createElement('div');
      bar.className = 'motor-bar';
      bar.innerHTML = `<span style="color:var(--text-dim);font-size:9px;width:14px">${i}</span><div class="bar-track"><div class="bar-fill" id="motor-${i}"></div></div><span class="bar-val" id="motor-v-${i}">0</span>`;
      container.appendChild(bar);
    }
  }
  for (let i = 0; i < count; i++) {
    const val = output[i];
    const pct = Math.min(Math.max(val, 0), 1) * 100;
    const fill = document.getElementById('motor-' + i);
    const label = document.getElementById('motor-v-' + i);
    if (fill) fill.style.width = pct + '%';
    if (label) label.textContent = val.toFixed(2);
  }
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
  const node = nodeData[nodeMap.get(selectedNodeId)];
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
  if (!system || !cachedStats) return;
  const stats = cachedStats;
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

// Buffer for events added while paused — flushed on resume
const pendingEvents = [];

function addEvent(step, text, cssClass) {
  if (window._logPaused && window._logPaused()) {
    pendingEvents.push({ step, text, cssClass });
    if (pendingEvents.length > 500) pendingEvents.shift(); // cap buffer
    return;
  }
  // Flush any pending events first
  while (pendingEvents.length > 0) {
    const e = pendingEvents.shift();
    _insertEvent(e.step, e.text, e.cssClass);
  }
  _insertEvent(step, text, cssClass);
}

function _insertEvent(step, text, cssClass) {
  const eventsEl = document.getElementById('events');
  const el = document.createElement('div');
  el.className = 'event-item';
  el.innerHTML = `<span class="${cssClass}">[${step}]</span> ${text}`;
  eventsEl.insertBefore(el, eventsEl.firstChild);
  while (eventsEl.children.length > 500) eventsEl.removeChild(eventsEl.lastChild);
  // Apply current filters to new event
  if (window._applyLogFilters) window._applyLogFilters();
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
      const fired = system.fired_ids();
      rasterStampStep(fired);
      if (nodeData.length !== rasterMorphonCount) rebuildMorphonOrder();
      updateScene(); updatePanels(); detectEvents(); drawRasterPlot();
    }
  });

  // Logarithmic speed mapping: 0→0.5x, 50→10x, 100→50x
  function sliderToSpeed(val) {
    return 0.5 * Math.pow(100, val / 100);
  }
  document.getElementById('speed-slider').addEventListener('input', (e) => {
    stepsPerFrame = sliderToSpeed(parseInt(e.target.value));
    document.getElementById('speed-val').textContent =
      stepsPerFrame < 1 ? stepsPerFrame.toFixed(1) + 'x' : Math.round(stepsPerFrame) + 'x';
  });

  // Also clear cell type filter on reset
  function resetSystem() {
    filterCellType = null;
    clearCellTypeActive();
    if (system) { try { system.free(); } catch(_) {} } // free WASM memory
    const program = document.getElementById('program-select').value;
    system = new WasmSystem(60, program, 3);
    selectedNodeId = null; hoveredNodeId = null;
    connectedToSelected.clear();
    nodeDim.clear(); nodeDimTarget.clear(); nodeGlow.clear();
    lastSpikeCount = 0; stepAccumulator = 0; liveSpikes.length = 0; spikeCooldowns.clear();
    rasterScrollX = 0; rasterMorphonCount = 0;
    rasterHistory.fill(null); groupRateHistory = [];
    morphonOrder = []; morphonYMap.clear(); rasterGroups = [];
    firingHistory.length = 0;
    lastMorphonCount = 0; lastSynapseCount = 0;
    document.getElementById('events').innerHTML = '';
    addEvent(0, `System reset [${program}]`, 'event-diff');
    for (let i = 0; i < 20; i++) { makeInput('noise'); system.step(); }
    updateScene(); updatePanels();
  }

  document.getElementById('btn-reset').addEventListener('click', resetSystem);
  document.getElementById('program-select').addEventListener('change', resetSystem);

  document.getElementById('btn-reward').addEventListener('click', () => {
    if (system) { system.inject_reward(0.8); addEvent('', 'Reward injected (0.8)', 'event-birth'); }
  });
  document.getElementById('btn-novelty').addEventListener('click', () => {
    if (system) { system.inject_novelty(0.6); addEvent('', 'Novelty injected (0.6)', 'event-synapse'); }
  });
  document.getElementById('btn-arousal').addEventListener('click', () => {
    if (system) { system.inject_arousal(0.9); addEvent('', 'Arousal injected (0.9)', 'event-death'); }
  });

  document.getElementById('feed-burst').addEventListener('click', () => { lastUserInputTime = performance.now(); makeInput('burst'); });
  document.getElementById('feed-pulse').addEventListener('click', () => { lastUserInputTime = performance.now(); makeInput('pulse'); });
  document.getElementById('feed-wave').addEventListener('click', () => { lastUserInputTime = performance.now(); makeInput('wave'); });
  document.getElementById('feed-noise').addEventListener('click', () => { lastUserInputTime = performance.now(); makeInput('noise'); });

  // btn-clear-log removed — now handled by unified btn-panel-clear

  // Bottom panel tabs
  document.querySelectorAll('#bottom-panel .tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('#bottom-panel .tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('#bottom-panel .tab-content').forEach(c => c.classList.remove('active'));
      btn.classList.add('active');
      document.getElementById(btn.dataset.tab)?.classList.add('active');
      resizeRasterCanvas(); // re-measure after layout change
    });
  });
  // Maximize / restore
  document.getElementById('btn-panel-maximize')?.addEventListener('click', () => {
    const panel = document.getElementById('bottom-panel');
    panel.classList.toggle('maximized');
    const isMax = panel.classList.contains('maximized');
    document.getElementById('btn-panel-maximize').textContent = isMax ? '\u25BD' : '\u25A1';
    // Update scene container bottom to match
    document.getElementById('scene-container').style.bottom = isMax ? '45vh' : '160px';
    setTimeout(() => { onResize(); }, 260); // after CSS transition
  });
  // Clear: context-dependent (raster or log)
  document.getElementById('btn-panel-clear')?.addEventListener('click', () => {
    const activeTab = document.querySelector('#bottom-panel .tab-btn.active')?.dataset.tab;
    if (activeTab === 'tab-raster') {
      rasterScrollX = 0;
      rasterHistory.fill(null);
      groupRateHistory.forEach(arr => arr.fill(0));
    } else if (activeTab === 'tab-log') {
      document.getElementById('events').innerHTML = '';
    }
  });

  // === LOG FEATURES ===
  let logPaused = false;
  const logFilters = { birth: true, death: true, synapse: true, diff: true };

  // Filter buttons
  document.querySelectorAll('.log-filter-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const f = btn.dataset.filter;
      if (f === 'all') {
        // Toggle all on/off
        const allActive = Object.values(logFilters).every(v => v);
        for (const k in logFilters) logFilters[k] = !allActive;
        document.querySelectorAll('.log-filter-btn').forEach(b => b.classList.toggle('active', !allActive));
      } else {
        logFilters[f] = !logFilters[f];
        btn.classList.toggle('active', logFilters[f]);
        // Update ALL button state
        const allOn = Object.values(logFilters).every(v => v);
        document.querySelector('.log-filter-btn[data-filter="all"]')?.classList.toggle('active', allOn);
      }
      applyLogFilters();
    });
  });

  // Search
  document.getElementById('log-search')?.addEventListener('input', () => applyLogFilters());

  function applyLogFilters() {
    const searchTerm = (document.getElementById('log-search')?.value || '').toLowerCase();
    const items = document.querySelectorAll('#events .event-item');
    let total = 0, visible = 0;
    let births = 0, deaths = 0, synapses = 0, other = 0;
    items.forEach(el => {
      total++;
      const text = el.textContent.toLowerCase();
      const matchesSearch = !searchTerm || text.includes(searchTerm);
      let matchesFilter = true;
      let type = 'other';
      if (el.querySelector('.event-birth')) { type = 'birth'; births++; }
      else if (el.querySelector('.event-death')) { type = 'death'; deaths++; }
      else if (el.querySelector('.event-synapse')) { type = 'synapse'; synapses++; }
      else { other++; }
      if (type === 'birth') matchesFilter = logFilters.birth;
      else if (type === 'death') matchesFilter = logFilters.death;
      else if (type === 'synapse') matchesFilter = logFilters.synapse;
      else matchesFilter = logFilters.diff;
      const show = matchesSearch && matchesFilter;
      el.classList.toggle('hidden', !show);
      if (show) visible++;
    });
    // Update status bar (bottom-right)
    const statusEl = document.getElementById('log-status');
    if (statusEl) {
      const parts = [
        `<span class="log-stat">Total: <span class="log-stat-val">${total}</span></span>`,
        visible !== total ? `<span class="log-stat">Showing: <span class="log-stat-val">${visible}</span></span>` : '',
        `<span class="log-stat" style="color:var(--modulatory)">B:<span class="log-stat-val">${births}</span></span>`,
        `<span class="log-stat" style="color:var(--arousal-color)">D:<span class="log-stat-val">${deaths}</span></span>`,
        `<span class="log-stat" style="color:var(--sensory)">S:<span class="log-stat-val">${synapses}</span></span>`,
        `<span class="log-stat" style="color:var(--associative)">O:<span class="log-stat-val">${other}</span></span>`,
      ].filter(Boolean);
      statusEl.innerHTML = parts.join('');
    }
  }

  // Pause log display
  document.getElementById('btn-log-pause')?.addEventListener('click', () => {
    logPaused = !logPaused;
    const btn = document.getElementById('btn-log-pause');
    btn.classList.toggle('paused', logPaused);
    btn.innerHTML = logPaused ? '&#9654;' : '&#9646;&#9646;';
    btn.title = logPaused ? 'Resume log display' : 'Pause log display';
  });

  // Menu toggle
  document.getElementById('btn-log-menu')?.addEventListener('click', (e) => {
    e.stopPropagation();
    document.getElementById('log-menu')?.classList.toggle('hidden');
  });
  document.addEventListener('click', () => {
    document.getElementById('log-menu')?.classList.add('hidden');
  });

  // Copy all to clipboard
  document.getElementById('btn-log-copy')?.addEventListener('click', () => {
    const lines = [];
    document.querySelectorAll('#events .event-item').forEach(el => lines.push(el.textContent.trim()));
    navigator.clipboard.writeText(lines.join('\n')).then(() => {
      addEvent('', `Copied ${lines.length} log entries to clipboard`, 'event-diff');
    });
    document.getElementById('log-menu')?.classList.add('hidden');
  });

  // Copy filtered to clipboard
  document.getElementById('btn-log-copy-filtered')?.addEventListener('click', () => {
    const lines = [];
    document.querySelectorAll('#events .event-item:not(.hidden)').forEach(el => lines.push(el.textContent.trim()));
    navigator.clipboard.writeText(lines.join('\n')).then(() => {
      addEvent('', `Copied ${lines.length} filtered entries to clipboard`, 'event-diff');
    });
    document.getElementById('log-menu')?.classList.add('hidden');
  });

  // Expose logPaused to addEvent
  window._logPaused = () => logPaused;
  window._applyLogFilters = applyLogFilters;

  // Save/Load
  document.getElementById('btn-save').addEventListener('click', () => {
    if (!system) return;
    try {
      savedState = system.save_json();
      addEvent('', 'State saved', 'event-diff');
    } catch (e) { addEvent('', 'Save failed: ' + e.message, 'event-death'); }
  });
  document.getElementById('btn-load').addEventListener('click', () => {
    if (!savedState) { addEvent('', 'No saved state', 'event-death'); return; }
    try {
      system = WasmSystem.loadJson(savedState);
      selectedNodeId = null; hoveredNodeId = null;
      connectedToSelected.clear(); filterCellType = null;
      nodeDim.clear(); nodeDimTarget.clear();
      clearCellTypeActive();
      addEvent('', 'State loaded', 'event-diff');
      updateScene(); updatePanels();
    } catch (e) { addEvent('', 'Load failed: ' + e.message, 'event-death'); }
  });

  // Fullscreen
  document.getElementById('btn-fullscreen').addEventListener('click', toggleFullscreen);

  // Help overlay
  const helpOverlay = document.getElementById('help-overlay');
  document.getElementById('btn-help').addEventListener('click', () => {
    helpOverlay.classList.toggle('visible');
  });
  helpOverlay.addEventListener('click', (e) => {
    if (e.target === helpOverlay) helpOverlay.classList.remove('visible');
  });

  // Cell type filtering — click to highlight, click again to clear
  document.querySelectorAll('.cell-type-row[data-type]').forEach(row => {
    row.addEventListener('click', () => {
      const type = row.dataset.type;
      if (filterCellType === type) {
        // Toggle off
        filterCellType = null;
        clearCellTypeActive();
      } else {
        filterCellType = type;
        selectedNodeId = null;
        connectedToSelected.clear();
        clearCellTypeActive();
        row.classList.add('active');
      }
    });
  });

  // Keyboard shortcuts
  window.addEventListener('keydown', (e) => {
    // Don't capture when typing in inputs
    if (['INPUT', 'SELECT', 'TEXTAREA', 'BUTTON'].includes(e.target.tagName)) return;

    switch (e.key) {
      case ' ':
        e.preventDefault();
        pauseBtn.click();
        break;
      case 'ArrowRight':
        e.preventDefault();
        document.getElementById('btn-step').click();
        break;
      case 'r':
      case 'R':
        if (!e.metaKey && !e.ctrlKey) resetSystem();
        break;
      case 'f':
      case 'F':
        if (!e.metaKey && !e.ctrlKey) toggleFullscreen();
        break;
      case '?':
        helpOverlay.classList.toggle('visible');
        break;
      case 'Escape':
        if (helpOverlay.classList.contains('visible')) {
          helpOverlay.classList.remove('visible');
        } else {
          selectedNodeId = null;
          connectedToSelected.clear();
          filterCellType = null;
          clearCellTypeActive();
          updateDetailPanel();
        }
        break;
    }
  });
}

function clearCellTypeActive() {
  document.querySelectorAll('.cell-type-row').forEach(r => r.classList.remove('active'));
}

function toggleFullscreen() {
  if (!document.fullscreenElement) {
    document.documentElement.requestFullscreen().catch(() => {});
  } else {
    document.exitFullscreen();
  }
}

// ============================================================
// SPIKE PARTICLE ANIMATION
// ============================================================
const spikeDummy = new THREE.Object3D();
const INV_BALL = 1.0 / BALL_RADIUS;

// Poincaré ball geodesic interpolation (same math as Rust exp_map/log_map).
// Operates in unit-ball coordinates (‖x‖ < 1).

// Möbius addition: x ⊕ y (curvature c=1)
function mobiusAdd(x0, x1, x2, y0, y1, y2) {
  const xdy = x0*y0 + x1*y1 + x2*y2;
  const xsq = x0*x0 + x1*x1 + x2*x2;
  const ysq = y0*y0 + y1*y1 + y2*y2;
  const d = 1.0 / Math.max(1 + 2*xdy + xsq*ysq, 1e-10);
  const a = (1 + 2*xdy + ysq) * d;
  const b = (1 - xsq) * d;
  return [a*x0 + b*y0, a*x1 + b*y1, a*x2 + b*y2];
}

// Geodesic point at parameter t between p and q in the Poincaré ball.
// γ(t) = p ⊕ (t ⊗ ((-p) ⊕ q))
function geodesicPoint(p0, p1, p2, q0, q1, q2, t) {
  // diff = (-p) ⊕ q
  const d = mobiusAdd(-p0, -p1, -p2, q0, q1, q2);
  // Möbius scalar: t ⊗ diff
  const norm = Math.sqrt(d[0]*d[0] + d[1]*d[1] + d[2]*d[2]);
  let s0, s1, s2;
  if (norm < 1e-10) {
    s0 = 0; s1 = 0; s2 = 0;
  } else {
    const atanh_n = Math.atanh(Math.min(norm, 0.999));
    const coeff = Math.tanh(t * atanh_n) / norm;
    s0 = d[0] * coeff; s1 = d[1] * coeff; s2 = d[2] * coeff;
  }
  // result = p ⊕ scaled
  return mobiusAdd(p0, p1, p2, s0, s1, s2);
}

function updateSpikes() {
  let alive = 0;

  for (let i = liveSpikes.length - 1; i >= 0; i--) {
    const s = liveSpikes[i];
    s.age++;
    if (s.age > SPIKE_VISUAL_FRAMES) { liveSpikes.splice(i, 1); continue; }

    // Linear progress — constant speed, no easing stutter
    const t = s.age / SPIKE_VISUAL_FRAMES;

    // Lerp position along the edge (simple, fast, no GC)
    const fi = s.fromIdx * 3, ti = s.toIdx * 3;
    const x = nodePositions[fi]   + (nodePositions[ti]   - nodePositions[fi])   * t;
    const y = nodePositions[fi+1] + (nodePositions[ti+1] - nodePositions[fi+1]) * t;
    const z = nodePositions[fi+2] + (nodePositions[ti+2] - nodePositions[fi+2]) * t;

    // Tiny orb
    spikeDummy.position.set(x, y, z);
    spikeDummy.scale.setScalar(0.035);
    spikeDummy.updateMatrix();

    if (alive < MAX_SPIKES) {
      spikesMesh.setMatrixAt(alive, spikeDummy.matrix);
      // Bright enough to trigger bloom (threshold 0.85), fade at end
      const fade = t < 0.8 ? 1.0 : (1.0 - t) * 5.0;
      tempColor.copy(s.color).multiplyScalar(1.8 * fade);
      spikesMesh.setColorAt(alive, tempColor);
      alive++;
    }
  }

  lastSpikeCount = alive;
  spikesMesh.count = alive;
  if (alive > 0) {
    spikesMesh.instanceMatrix.needsUpdate = true;
    if (spikesMesh.instanceColor) spikesMesh.instanceColor.needsUpdate = true;
  }
}

// ============================================================
// RASTER PLOT
// ============================================================
function initRaster() {
  rasterCanvas = document.getElementById('raster-canvas');
  rasterCtx = rasterCanvas.getContext('2d');
  resizeRasterCanvas();
}

function resizeRasterCanvas() {
  if (!rasterCanvas) return;
  const wrap = document.getElementById('raster-canvas-wrap');
  if (!wrap) return;
  const rect = wrap.getBoundingClientRect();
  rasterCanvas.width = Math.floor(rect.width);
  rasterCanvas.height = Math.floor(rect.height);
  rasterCtx.imageSmoothingEnabled = false;
}

// Group info: { type, startRow, count } — built by rebuildMorphonOrder
let rasterGroups = [];
// Spike history ring buffer: each slot = Uint32Array of fired IDs for that step
const rasterHistory = new Array(RASTER_WINDOW).fill(null);
// Per-group firing rate history: groupRateHistory[groupIdx][step % RASTER_WINDOW] = fraction
let groupRateHistory = [];

function rebuildMorphonOrder() {
  // Save old group types before rebuilding
  const oldGroupTypes = rasterGroups.map(g => g.type);
  const oldRateHistory = groupRateHistory;

  const groups = {};
  for (const type of CELL_TYPE_ORDER) groups[type] = [];
  for (const n of nodeData) {
    const type = CELL_TYPE_ORDER.includes(n.cell_type) ? n.cell_type : 'Stem';
    groups[type].push(n.id);
  }
  morphonOrder = [];
  rasterGroups = [];
  for (const type of CELL_TYPE_ORDER) {
    if (groups[type].length === 0) continue;
    groups[type].sort((a, b) => a - b);
    const start = morphonOrder.length;
    for (const id of groups[type]) morphonOrder.push({ id, cellType: type });
    rasterGroups.push({ type, startRow: start, count: groups[type].length });
  }
  morphonYMap.clear();
  morphonOrder.forEach((m, i) => morphonYMap.set(m.id, i));
  rasterMorphonCount = nodeData.length;

  // Carry over rate history by matching group type names
  const oldByType = {};
  for (let i = 0; i < oldGroupTypes.length; i++) {
    if (oldRateHistory[i]) oldByType[oldGroupTypes[i]] = oldRateHistory[i];
  }
  groupRateHistory = rasterGroups.map(g => oldByType[g.type] || new Float32Array(RASTER_WINDOW));
}

function rasterStampStep(firedIds) {
  if (morphonOrder.length === 0 || rasterGroups.length === 0) return;
  const col = rasterScrollX % RASTER_WINDOW;
  rasterHistory[col] = firedIds;

  // Compute per-group firing rate for this step
  const groupCounts = new Array(rasterGroups.length).fill(0);
  for (const id of firedIds) {
    const row = morphonYMap.get(id);
    if (row === undefined) continue;
    for (let gi = 0; gi < rasterGroups.length; gi++) {
      const g = rasterGroups[gi];
      if (row >= g.startRow && row < g.startRow + g.count) {
        groupCounts[gi]++;
        break;
      }
    }
  }
  for (let gi = 0; gi < rasterGroups.length; gi++) {
    groupRateHistory[gi][col] = groupCounts[gi] / rasterGroups[gi].count;
  }

  rasterScrollX++;
}

function drawRasterPlot() {
  if (!rasterCtx || rasterGroups.length === 0) return;
  const canvasW = rasterCanvas.width;
  const canvasH = rasterCanvas.height;
  if (canvasW <= 0 || canvasH <= 0) return;

  const labelW = 40;
  const plotW = canvasW - labelW - 8;
  const nGroups = rasterGroups.length;
  const bandGap = 3;
  const bandH = (canvasH - (nGroups - 1) * bandGap) / nGroups;

  rasterCtx.clearRect(0, 0, canvasW, canvasH);

  const stepsVisible = Math.min(rasterScrollX, RASTER_WINDOW);
  const colW = plotW / RASTER_WINDOW;

  for (let gi = 0; gi < nGroups; gi++) {
    const g = rasterGroups[gi];
    const rgb = RASTER_COLORS[g.type];
    const bandY = gi * (bandH + bandGap);
    const rates = groupRateHistory[gi];

    // Background band
    rasterCtx.fillStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},0.06)`;
    rasterCtx.fillRect(labelW, bandY, plotW, bandH);

    // Heatmap: each column colored by firing rate intensity
    for (let s = 0; s < stepsVisible; s++) {
      const bufIdx = ((rasterScrollX - stepsVisible + s) % RASTER_WINDOW + RASTER_WINDOW) % RASTER_WINDOW;
      const rate = rates[bufIdx];
      if (rate <= 0) continue;

      const x = labelW + (s / RASTER_WINDOW) * plotW;
      // Intensity: sqrt for perceptual linearity, cap at 1
      const intensity = Math.min(Math.sqrt(rate * 3), 1.0);
      const alpha = 0.15 + intensity * 0.85;
      rasterCtx.fillStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${alpha.toFixed(2)})`;
      rasterCtx.fillRect(x, bandY, Math.max(colW, 1.2), bandH);
    }

    // Firing rate line trace (overlaid)
    rasterCtx.strokeStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},0.9)`;
    rasterCtx.lineWidth = 1.5;
    rasterCtx.beginPath();
    let started = false;
    for (let s = 0; s < stepsVisible; s++) {
      const bufIdx = ((rasterScrollX - stepsVisible + s) % RASTER_WINDOW + RASTER_WINDOW) % RASTER_WINDOW;
      const rate = rates[bufIdx];
      const x = labelW + (s / RASTER_WINDOW) * plotW;
      const y = bandY + bandH - rate * bandH * 0.9; // 0% at bottom, 100% at top
      if (!started) { rasterCtx.moveTo(x, y); started = true; }
      else rasterCtx.lineTo(x, y);
    }
    rasterCtx.stroke();

    // Group label + count
    rasterCtx.font = '9px "JetBrains Mono", monospace';
    rasterCtx.textBaseline = 'middle';
    rasterCtx.textAlign = 'right';
    rasterCtx.fillStyle = `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
    rasterCtx.fillText(`${g.type.substring(0, 3).toUpperCase()}`, labelW - 4, bandY + bandH / 2);
    // Current rate % on the right
    if (stepsVisible > 0) {
      const lastBuf = ((rasterScrollX - 1) % RASTER_WINDOW + RASTER_WINDOW) % RASTER_WINDOW;
      const lastRate = rates[lastBuf];
      rasterCtx.textAlign = 'left';
      rasterCtx.fillStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},0.7)`;
      rasterCtx.fillText(`${(lastRate * 100).toFixed(0)}%`, labelW + plotW + 3, bandY + bandH / 2);
    }
  }

  // Time cursor
  if (stepsVisible > 0) {
    const cursorX = labelW + (stepsVisible / RASTER_WINDOW) * plotW;
    rasterCtx.strokeStyle = 'rgba(255,255,255,0.25)';
    rasterCtx.lineWidth = 1;
    rasterCtx.beginPath();
    rasterCtx.moveTo(cursorX, 0);
    rasterCtx.lineTo(cursorX, canvasH);
    rasterCtx.stroke();
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
    frameFired.clear();
    // Accumulate fractional steps for sub-1x speeds
    stepAccumulator += stepsPerFrame;
    const stepsThisFrame = Math.floor(stepAccumulator);
    stepAccumulator -= stepsThisFrame;

    for (let i = 0; i < stepsThisFrame; i++) {
      // Auto-inject noise unless user sent input recently (500ms grace period)
      if (frameCount % 3 === 0 && performance.now() - lastUserInputTime > 500) makeInput('noise');
      system.step();
      const fired = system.fired_ids();
      for (const id of fired) frameFired.add(id);
      // Stamp raster plot for each sub-step
      rasterStampStep(fired);
      // Flash arrival targets — spike delivery triggers a glow on the receiving morphon
      try {
        for (const id of system.delivered_target_ids()) {
          const cur = nodeGlow.get(id) || 0;
          nodeGlow.set(id, Math.min(cur + 0.5, 1.0));
        }
      } catch(_) {}
    }

    // Rebuild raster Y-axis if morphon count changed
    if (nodeData.length !== rasterMorphonCount) rebuildMorphonOrder();

    updateScene();
    drawRasterPlot();
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
    const activity = Math.min(lastSpikeCount / 80, 1.0);
    bloomPass.strength = 0.34 + activity * 0.22;
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
  initRaster();
  await init();

  system = new WasmSystem(60, 'cortical', 3);
  for (let i = 0; i < 30; i++) { makeInput('noise'); system.step(); }

  updateScene();
  rebuildMorphonOrder();
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
