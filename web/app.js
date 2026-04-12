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

// V3: Epistemic state colors
const EPISTEMIC_COLORS = {
  Hypothesis:  new THREE.Color(0x3b82f6), // blue — exploring
  Supported:   new THREE.Color(0x22c55e), // green — stable
  Outdated:    new THREE.Color(0xeab308), // amber — stale
  Contested:   new THREE.Color(0xef4444), // red — conflicting
  none:        new THREE.Color(0x4b5563), // gray — no cluster
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
// Color mode: 'celltype' (default) or 'pe' (prediction error heatmap)
let colorMode = 'celltype';
// Per-node dim factor: 0 = full color, 1 = dimmed. Keyed by morphon ID for stable mapping.
const nodeDim = new Map();
const nodeDimTarget = new Map();
// Per-node glow: approaches 1.0 on fire, decays toward 0. Keyed by morphon ID.
const nodeGlow = new Map();
// Smooth epistemic color: lerp toward target to prevent photosensitive flashing
const nodeEpistemicColor = new Map(); // id → THREE.Color (current blended color)
// Smooth PE value: lerp toward target to prevent jittery color changes
const nodePeSmooth = new Map(); // id → smoothed PE value

// Death animation: nodes that just died fade out over DEATH_ANIM_FRAMES
const DEATH_ANIM_FRAMES = 50;
const dyingNodes = []; // { px, py, pz, color: THREE.Color, size, age }
let prevNodeIds = new Set();
let _prevNodeIdsWork = new Set(); // swap partner — avoids new Set() each frame

const firingHistory = [];
const morphonHistory = [];
const synapseHistory = [];
const fieldPeHistory = [];
const justifiedHistory = [];  // V3: justified fraction sparkline
const MAX_HISTORY = 120;

// === HEATMAP STATE ===
const HEATMAP_BINS = 300;    // number of time bins visible
const HEATMAP_BIN_SIZE = 5;  // steps per bin (adjustable via UI)
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
let rasterScrollX = 0;       // total steps seen
let rasterMorphonCount = 0;  // cached to detect topology changes
let stepAccumulator = 0;     // for sub-1x speed
let heatmapBinSize = HEATMAP_BIN_SIZE;
let heatmapPaused = false;   // pause-on-hover
// Per-group accumulators for current bin
let heatmapBinAccum = [];    // [groupIdx] = count of firings in current bin
let heatmapBinSteps = 0;     // steps accumulated in current bin

// Cached stats to avoid double inspect() calls
let cachedStats = null;
let cachedTopo = null; // topology_json cache — refresh every 2 frames (topology only changes at slow clock)
// Saved state for save/load
let savedState = null;
let systemStartTime = performance.now();
let frameLoadEma = 0; // EMA of simulation step time as fraction of 16.67ms budget

// === ARENA STATE ===
let occupation = 'idle'; // 'idle' | 'arena'
const ARENA_GRID = 8;
const ARENA_CLASSES = 4;
const arenaPixels = new Float64Array(ARENA_GRID * ARENA_GRID); // current drawing
let arenaSelectedClass = 0;
const arenaTrainingSet = []; // [{pixels: Float64Array, label: number}]
const arenaConfusion = Array.from({length: ARENA_CLASSES}, () => new Array(ARENA_CLASSES).fill(0));
const arenaAccHistory = [];
let arenaTrainIdx = 0;
let arenaCycleCount = 0;
let arenaTrainSubStep = 0; // sub-step within a training cycle
let arenaLastOutput = null; // last raw output for display
let arenaDrawing = false; // mouse-down on grid
let arenaErasing = false;
const ARENA_CLASS_COLORS = ['#00d4ff', '#a78bfa', '#fbbf24', '#34d399'];
const ARENA_CLASS_NAMES = ['A', 'B', 'C', 'D'];

// Three.js objects
let renderer, scene, camera, controls, composer, bloomPass;
let nodesMesh, glowMesh, edgesMesh, diskMesh, fresnelBall;
let ambientParticles; // floating dust motes inside the ball
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
// Pooled / pre-allocated objects to avoid per-frame GC pressure
const _spikeCandidates = [];                              // reused candidate buffer per firing morphon
const _spikeCandSortFn = (a, b) => b.aw - a.aw;          // hoisted — avoids new closure per sort
const INHIBITORY_SPIKE_COLOR = new THREE.Color(0.85, 0.10, 0.45); // shared ref, never mutated

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
// Raw client coords for tooltip positioning (avoids 3D→2D projection drift)
let mouseClientX = 0, mouseClientY = 0;
// Track last user input time to avoid noise overwrite
let lastUserInputTime = 0;

// Reusable Three.js objects for learning pulses (avoid per-frame allocations)
const _pulseDummy = new THREE.Object3D();
const _pulseColor = new THREE.Color(0.98, 0.75, 0.14);
// Reusable vectors for edge hover raycast
const _edgeVecA = new THREE.Vector3();
const _edgeVecB = new THREE.Vector3();

// Cached DOM element references — populated once in initDOMCache()
const dom = {};
function initDOMCache() {
  // Header
  dom.hStep = document.getElementById('h-step');
  dom.hFired = document.getElementById('h-fired');
  dom.hUptime = document.getElementById('h-uptime');
  dom.hLoad = document.getElementById('h-load');
  // Left panel stats
  dom.sMorphons = document.getElementById('s-morphons');
  dom.sSynapses = document.getElementById('s-synapses');
  dom.sClusters = document.getElementById('s-clusters');
  dom.sGen = document.getElementById('s-gen');
  dom.sFiring = document.getElementById('s-firing');
  dom.sFired = document.getElementById('s-fired');
  dom.sEnergy = document.getElementById('s-energy');
  dom.sError = document.getElementById('s-error');
  dom.sFieldPeMax = document.getElementById('s-field-pe-max');
  dom.sFieldPeMean = document.getElementById('s-field-pe-mean');
  dom.sWmem = document.getElementById('s-wmem');
  dom.sBorn = document.getElementById('s-born');
  dom.sDied = document.getElementById('s-died');
  dom.sTransdiff = document.getElementById('s-transdiff');
  // Cell type counts
  dom.ctStem = document.getElementById('ct-Stem');
  dom.ctSensory = document.getElementById('ct-Sensory');
  dom.ctAssociative = document.getElementById('ct-Associative');
  dom.ctMotor = document.getElementById('ct-Motor');
  dom.ctModulatory = document.getElementById('ct-Modulatory');
  dom.ctFused = document.getElementById('ct-Fused');
  dom.ctMap = { Stem: null, Sensory: null, Associative: null, Motor: null, Modulatory: null, Fused: null };
  for (const t of Object.keys(dom.ctMap)) dom.ctMap[t] = document.getElementById('ct-' + t);
  // Neuromodulation bars
  dom.modReward = document.getElementById('mod-reward');
  dom.modRewardV = document.getElementById('mod-reward-v');
  dom.modNovelty = document.getElementById('mod-novelty');
  dom.modNoveltyV = document.getElementById('mod-novelty-v');
  dom.modArousal = document.getElementById('mod-arousal');
  dom.modArousalV = document.getElementById('mod-arousal-v');
  dom.modHomeo = document.getElementById('mod-homeo');
  dom.modHomeoV = document.getElementById('mod-homeo-v');
  // Endo panel
  dom.endoStage = document.getElementById('endo-stage');
  dom.endoHealthBar = document.getElementById('endo-health-bar');
  dom.endoHealthV = document.getElementById('endo-health-v');
  dom.endoChannels = document.getElementById('endo-channels');
  dom.endoInterventions = document.getElementById('endo-interventions');
  dom.endoRg = document.getElementById('endo-rg');
  dom.endoNg = document.getElementById('endo-ng');
  dom.endoAg = document.getElementById('endo-ag');
  dom.endoHg = document.getElementById('endo-hg');
  dom.endoTb = document.getElementById('endo-tb');
  dom.endoPm = document.getElementById('endo-pm');
  dom.endoCg = document.getElementById('endo-cg');
  // Governance
  dom.sJustified = document.getElementById('s-justified');
  dom.sConsolidated = document.getElementById('s-consolidated');
  dom.sSkepticism = document.getElementById('s-skepticism');
  dom.sEpistemic = document.getElementById('s-epistemic');
  // Detail panel
  dom.detailPanel = document.getElementById('detail-panel');
  dom.dId = document.getElementById('d-id');
  dom.dType = document.getElementById('d-type');
  dom.dDot = document.getElementById('d-dot');
  dom.dGen = document.getElementById('d-gen');
  dom.dAge = document.getElementById('d-age');
  dom.dEnergy = document.getElementById('d-energy');
  dom.dEnergyBar = document.getElementById('d-energy-bar');
  dom.dPotential = document.getElementById('d-potential');
  dom.dThreshold = document.getElementById('d-threshold');
  dom.dDiff = document.getElementById('d-diff');
  dom.dDiffBar = document.getElementById('d-diff-bar');
  dom.dError = document.getElementById('d-error');
  dom.dDesire = document.getElementById('d-desire');
  dom.dFired = document.getElementById('d-fired');
  dom.dConns = document.getElementById('d-conns');
  dom.dEpistemicRow = document.getElementById('d-epistemic-row');
  dom.dSkepticismRow = document.getElementById('d-skepticism-row');
  dom.dJustifiedRow = document.getElementById('d-justified-row');
  dom.dEpistemic = document.getElementById('d-epistemic');
  dom.dSkepticism = document.getElementById('d-skepticism');
  dom.dJustified = document.getElementById('d-justified');
  // Motor output
  dom.motorOutput = document.getElementById('motor-output');
  // Tooltip
  dom.tooltip = document.getElementById('tooltip');
  // Cluster list
  dom.clusterList = document.getElementById('cluster-list');
  // Learning pipeline
  dom.lpSynBar = document.getElementById('lp-syn-bar');
  dom.lpSynV = document.getElementById('lp-syn-v');
  dom.lpEligBar = document.getElementById('lp-elig-bar');
  dom.lpEligV = document.getElementById('lp-elig-v');
  dom.lpTagBar = document.getElementById('lp-tag-bar');
  dom.lpTagV = document.getElementById('lp-tag-v');
  dom.lpCapBar = document.getElementById('lp-cap-bar');
  dom.lpCapV = document.getElementById('lp-cap-v');
  dom.lpConBar = document.getElementById('lp-con-bar');
  dom.lpConV = document.getElementById('lp-con-v');
  dom.lpJusBar = document.getElementById('lp-jus-bar');
  dom.lpJusV = document.getElementById('lp-jus-v');
  dom.weightHistogram = document.getElementById('weight-histogram');
  dom.whMin = document.getElementById('wh-min');
  dom.whMax = document.getElementById('wh-max');
  dom.whStats = document.getElementById('wh-stats');
}

// Cached parsed JSON — refreshed once per update cycle
let cachedMod = null;
let cachedEndo = null;
let cachedGov = null;
let cachedLearning = null;

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
  renderer = new THREE.WebGLRenderer({ antialias: true, logarithmicDepthBuffer: true, preserveDrawingBuffer: true });
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
  // Pre-init instance colors so the shader includes USE_INSTANCING_COLOR from
  // the first render and newly-grown indices never read zero-initialized (black) data.
  nodesMesh.instanceColor = new THREE.InstancedBufferAttribute(
    new Float32Array(MAX_NODES * 3).fill(1), 3
  );
  nodesMesh.instanceColor.setUsage(THREE.DynamicDrawUsage);
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
  glowMesh.instanceColor = new THREE.InstancedBufferAttribute(
    new Float32Array(MAX_NODES * 3), 3
  );
  glowMesh.instanceColor.setUsage(THREE.DynamicDrawUsage);
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
  spikesMesh.instanceColor = new THREE.InstancedBufferAttribute(
    new Float32Array(MAX_SPIKES * 3), 3
  );
  spikesMesh.instanceColor.setUsage(THREE.DynamicDrawUsage);
  scene.add(spikesMesh);

  // === AMBIENT PARTICLES — floating dust motes inside the Poincaré ball ===
  {
    const PARTICLE_COUNT = 600;
    const pPos = new Float32Array(PARTICLE_COUNT * 3);
    const pSizes = new Float32Array(PARTICLE_COUNT);
    const pPhase = new Float32Array(PARTICLE_COUNT); // random phase for drift
    for (let i = 0; i < PARTICLE_COUNT; i++) {
      // Random position inside the ball
      const r = Math.pow(Math.random(), 0.33) * BALL_RADIUS * 0.92;
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      pPos[i * 3] = r * Math.sin(phi) * Math.cos(theta);
      pPos[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
      pPos[i * 3 + 2] = r * Math.cos(phi);
      pSizes[i] = 0.4 + Math.random() * 1.2;
      pPhase[i] = Math.random() * Math.PI * 2;
    }
    const pGeo = new THREE.BufferGeometry();
    pGeo.setAttribute('position', new THREE.BufferAttribute(pPos, 3).setUsage(THREE.DynamicDrawUsage));
    pGeo.setAttribute('size', new THREE.BufferAttribute(pSizes, 1));
    const pMat = new THREE.PointsMaterial({
      color: 0x4466aa,
      size: 0.06,
      sizeAttenuation: true,
      transparent: true,
      opacity: 0.25,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    ambientParticles = new THREE.Points(pGeo, pMat);
    ambientParticles.userData = { basePositions: pPos.slice(), phases: pPhase };
    scene.add(ambientParticles);
  }

  window.addEventListener('resize', onResize);
  renderer.domElement.addEventListener('mousemove', onMouseMove);
  renderer.domElement.addEventListener('mousedown', (e) => { mouseDownPos.x = e.clientX; mouseDownPos.y = e.clientY; });
  renderer.domElement.addEventListener('click', onMouseClick);

  // Detail float window — drag + close
  setupDetailWindow();
}

function setupDetailWindow() {
  const panel = document.getElementById('detail-panel');
  const dragBar = document.getElementById('detail-drag-bar');
  const closeBtn = document.getElementById('detail-close');

  closeBtn.addEventListener('click', () => {
    selectedNodeId = null;
    connectedToSelected.clear();
    updateDetailPanel();
  });

  let dragging = false, dragStartX = 0, dragStartY = 0, panelStartLeft = 0, panelStartTop = 0;
  dragBar.addEventListener('mousedown', (e) => {
    if (e.target === closeBtn) return;
    dragging = true;
    dragStartX = e.clientX;
    dragStartY = e.clientY;
    panelStartLeft = parseInt(panel.style.left) || panel.getBoundingClientRect().left;
    panelStartTop  = parseInt(panel.style.top)  || panel.getBoundingClientRect().top;
    e.preventDefault();
  });
  window.addEventListener('mousemove', (e) => {
    if (!dragging) return;
    panel.style.left = (panelStartLeft + e.clientX - dragStartX) + 'px';
    panel.style.top  = (panelStartTop  + e.clientY - dragStartY) + 'px';
  });
  window.addEventListener('mouseup', () => { dragging = false; });
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
  const rect = renderer.domElement.getBoundingClientRect();
  mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
  mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
  mouseClientX = e.clientX;
  mouseClientY = e.clientY;
}

function onMouseClick(e) {
  // Ignore clicks that were actually orbit drags
  const dx = e.clientX - mouseDownPos.x, dy = e.clientY - mouseDownPos.y;
  if (dx * dx + dy * dy > 25) return; // moved more than 5px = drag
  if (hoveredNodeId !== null) {
    const wasAlreadySelected = selectedNodeId === hoveredNodeId;
    selectedNodeId = hoveredNodeId;
    // Build set of connected node IDs for context dimming
    connectedToSelected.clear();
    connectedToSelected.add(selectedNodeId);
    for (const e of edgeData) {
      if (e.from === selectedNodeId) connectedToSelected.add(e.to);
      if (e.to === selectedNodeId) connectedToSelected.add(e.from);
    }
    // Position float window near click (only reposition on new selection)
    if (!wasAlreadySelected) {
      const panel = dom.detailPanel;
      const margin = 16;
      const pw = 220, ph = 300; // estimated panel size
      let left = e.clientX + margin;
      let top  = e.clientY - 60;
      if (left + pw > window.innerWidth  - margin) left = e.clientX - pw - margin;
      if (top  + ph > window.innerHeight - margin) top  = window.innerHeight - ph - margin;
      if (top < 64) top = 64;
      panel.style.left = left + 'px';
      panel.style.top  = top  + 'px';
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
    // Guarantee a minimum hit radius so tiny nodes stay clickable
    const hitRadius = Math.max(scale * 3.0, 0.35);
    if (d < hitRadius) {
      const camDist = camera.position.distanceTo(_pos);
      if (camDist < closestDist) {
        closestDist = camDist;
        closest = i;
      }
    }
  }

  hoveredNodeId = closest !== null ? nodeData[closest]?.id ?? null : null;
  renderer.domElement.style.cursor = hoveredNodeId !== null ? 'pointer' : 'default';

  if (hoveredNodeId !== null) {
    const node = nodeData[nodeMap.get(hoveredNodeId)]; // O(1) lookup via nodeMap
    if (node) {
      dom.tooltip.style.display = 'block';
      // Position at cursor — avoids 3D→2D projection drift
      dom.tooltip.style.left = (mouseClientX + 16) + 'px';
      dom.tooltip.style.top = (mouseClientY - 10) + 'px';
      const ct = node.cell_type.toLowerCase();
      const dotColor = `var(--${ct}, #888)`;
      const energyPct = Math.min(node.energy / 2, 1) * 100;
      const potPct = Math.min(Math.abs(node.potential) / Math.max(node.threshold, 0.5), 1) * 100;
      const specPct = (node.specificity || 0) * 100;
      const rate = node.firing_rate || 0;
      const ratePct = Math.min(rate / 0.3, 1) * 100; // 30% firing rate = full bar
      const rateColor = rate > 0.15 ? '#fbbf24' : rate > 0.05 ? 'var(--accent)' : 'rgba(255,255,255,0.3)';
      const fusedTag = node.fused
        ? ' <span class="tip-tag" style="background:rgba(244,114,182,0.2);color:#f472b6">FUSED</span>'
        : '';
      dom.tooltip.innerHTML = `
        <div class="tip-header">
          <span class="tip-dot" style="background:${dotColor}${node.fired ? ';box-shadow:0 0 6px ' + dotColor : ''}"></span>
          <span class="tip-id">#${node.id}${fusedTag}</span>
          <span class="tip-type">${node.cell_type}</span>
        </div>
        <hr class="tip-sep">
        <div class="tip-row">
          <span class="label">Energy</span>
          <span class="tip-bar"><span class="fill" style="width:${energyPct}%;background:${dotColor}"></span></span>
          <span class="value">${node.energy.toFixed(2)}</span>
        </div>
        <div class="tip-row">
          <span class="label">Potential</span>
          <span class="tip-bar"><span class="fill" style="width:${potPct}%;background:${node.potential >= 0 ? 'var(--accent)' : '#ff4466'}"></span></span>
          <span class="value">${node.potential.toFixed(2)}</span>
        </div>
        <div class="tip-row">
          <span class="label">Fire rate</span>
          <span class="tip-bar"><span class="fill" style="width:${ratePct}%;background:${rateColor}"></span></span>
          <span class="value">${(rate * 100).toFixed(0)}%</span>
        </div>
        <div class="tip-row">
          <span class="label">Depth</span>
          <span class="tip-bar"><span class="fill" style="width:${specPct}%;background:rgba(255,255,255,0.3)"></span></span>
          <span class="value">${(node.specificity || 0).toFixed(2)}</span>
        </div>
        <div class="tip-row" style="margin-top:1px">
          <span class="label" style="color:var(--text-dim);font-size:9px">Gen ${node.generation} &middot; Age ${node.age}</span>
        </div>
      `;
    }
  } else {
    // Check for edge hover when not hovering a node
    const edgeIdx = findClosestEdge(mouseClientX, mouseClientY);
    if (edgeIdx !== null) {
      hoveredEdgeIdx = edgeIdx;
      showEdgeTooltip(edgeIdx);
    } else {
      hoveredEdgeIdx = null;
      dom.tooltip.style.display = 'none';
    }
  }
}

// ============================================================
// UPDATE SCENE FROM WASM DATA
// ============================================================
const dummy = new THREE.Object3D();
const tempColor = new THREE.Color();
const dimColor = new THREE.Color();
const peColor = new THREE.Color();

function updateScene() {
  if (!system) return;

  // Topology only changes at the slow clock (every 100 steps). At 10 steps/frame that's
  // every ~10 frames, so re-parsing every 2 frames wastes ~80% of those calls.
  if (!cachedTopo || frameCount % 2 === 0) {
    const topo = JSON.parse(system.topology_json());
    topo.nodes.sort((a, b) => a.id - b.id);
    cachedTopo = topo;
  }
  const nodes = cachedTopo.nodes;
  const edges = cachedTopo.edges;
  nodeData = nodes;
  edgeData = edges;
  nodeMap.clear();

  const nodeCount = Math.min(nodes.length, MAX_NODES);

  // === DETECT DEATHS — IDs present last frame but missing now ===
  // Reuse two pre-allocated Sets via swap — avoids new Set() + Array from .map() each frame.
  _prevNodeIdsWork.clear();
  for (let i = 0; i < nodeCount; i++) _prevNodeIdsWork.add(nodes[i].id);
  for (const id of prevNodeIds) {
    if (!_prevNodeIdsWork.has(id)) {
      // Capture last known position + color for the fade-out
      const idx = nodeMap.get(id); // from previous frame's map
      if (idx !== undefined) {
        const ct = nodeData[idx]?.cell_type;
        const c = CELL_COLORS[ct] || CELL_COLORS.Stem;
        dyingNodes.push({
          px: nodePositions[idx * 3],
          py: nodePositions[idx * 3 + 1],
          pz: nodePositions[idx * 3 + 2],
          color: c.clone(),
          size: Math.abs(dummy.matrix?.elements?.[0]) || 0.25,
          age: 0,
        });
      }
    }
  }
  // Swap references so both Sets get reused every other frame
  const _tmpSet = prevNodeIds;
  prevNodeIds = _prevNodeIdsWork;
  _prevNodeIdsWork = _tmpSet;

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

    let color;
    if (colorMode === 'pe') {
      // Heatmap: soft blue (low PE) → warm coral (high PE), smoothed
      const rawPe = Math.max(0, Math.min(1, n.prediction_error || 0));
      const prevPe = nodePeSmooth.get(n.id) ?? rawPe;
      const pe = prevPe + (rawPe - prevPe) * 0.05; // ~40 frame smooth
      nodePeSmooth.set(n.id, pe);
      // Lerp between friendly blue and warm coral
      peColor.setRGB(0.30 + pe * 0.65, 0.45 - pe * 0.15, 0.75 - pe * 0.40);
      color = peColor;
    } else if (colorMode === 'epistemic') {
      // V3: color by epistemic state — smooth transitions to avoid flashing
      const target = EPISTEMIC_COLORS[n.epistemic_state] || EPISTEMIC_COLORS.none;
      let cur = nodeEpistemicColor.get(n.id);
      if (!cur) { cur = target.clone(); nodeEpistemicColor.set(n.id, cur); }
      cur.lerp(target, 0.06); // ~30 frame blend — no hard color snaps
      color = cur;
    } else {
      color = CELL_COLORS[n.cell_type] || CELL_COLORS.Stem;
    }

    // === GLOW: only active or near-threshold morphons ===
    const prevGlow = nodeGlow.get(n.id) || 0;
    const nearThreshold = (n.potential > n.threshold * 0.6 && !n.fired) ? 0.15 : 0.0;
    const glowTarget = frameFired.has(n.id) ? 1.0 : nearThreshold;
    // Gentler attack/decay in PE/epistemic modes to avoid aggressive flashing
    const softMode = colorMode !== 'celltype';
    const attackRate = softMode ? 0.12 : 0.35;
    const decayRate = softMode ? 0.96 : 0.92;
    const glow = glowTarget > prevGlow
      ? prevGlow + (glowTarget - prevGlow) * attackRate
      : prevGlow * decayRate;
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
      // Diffuse layer: base brightness high enough so linear→sRGB+ACES stays visible
      const intensity = bright * (0.82 + glow * 0.18);
      tempColor.copy(color).multiplyScalar(intensity);
      if (glow > 0.5) {
        if (softMode) {
          // Gentle warm brightening — no white snap, just a subtle lift
          tempColor.multiplyScalar(1.0 + glow * 0.3);
        } else {
          // Cell type mode: visible white pop on fire
          const flash = (glow - 0.5) * 2.0;
          tempColor.lerp(WHITE, flash * 0.45);
        }
      }
      nodesMesh.setColorAt(i, tempColor);
    }

    nodePositions[i * 3] = px;
    nodePositions[i * 3 + 1] = py;
    nodePositions[i * 3 + 2] = pz;
  }

  // === DYING NODES — flash bright, shrink, fade out ===
  let dyingRendered = 0;
  for (let di = dyingNodes.length - 1; di >= 0; di--) {
    const d = dyingNodes[di];
    d.age++;
    if (d.age > DEATH_ANIM_FRAMES) { dyingNodes.splice(di, 1); continue; }
    const slot = nodeCount + dyingRendered;
    if (slot >= MAX_NODES) continue;

    const t = d.age / DEATH_ANIM_FRAMES; // 0→1
    // Quick flash then fade: bright at t=0, dim by t=0.3, gone by t=1
    const brightness = t < 0.15 ? 1.5 + (1 - t / 0.15) * 1.5 : (1 - t) * 0.8;
    // Shrink: hold size briefly, then collapse
    const scale = t < 0.2 ? d.size * (1 + t * 2) : d.size * Math.max(0, 1 - (t - 0.2) * 1.25);

    dummy.position.set(d.px, d.py, d.pz);
    dummy.scale.setScalar(scale * popScale);
    dummy.updateMatrix();
    nodesMesh.setMatrixAt(slot, dummy.matrix);

    // Flash white then fade to cell color then dim
    tempColor.copy(d.color).multiplyScalar(brightness);
    if (t < 0.15) tempColor.lerp(WHITE, (1 - t / 0.15) * 0.7);
    nodesMesh.setColorAt(slot, tempColor);
    dyingRendered++;
  }
  nodesMesh.count = nodeCount + dyingRendered;

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

    let glowColor;
    if (colorMode === 'pe') {
      // Use the already-smoothed PE value
      const pe = nodePeSmooth.get(n.id) ?? 0;
      peColor.setRGB(0.30 + pe * 0.65, 0.45 - pe * 0.15, 0.75 - pe * 0.40);
      glowColor = peColor;
    } else if (colorMode === 'epistemic') {
      glowColor = nodeEpistemicColor.get(n.id) || EPISTEMIC_COLORS.none;
    } else {
      glowColor = CELL_COLORS[n.cell_type] || CELL_COLORS.Stem;
    }
    // Softer glow multiplier in heatmap modes
    const glowMul = (colorMode !== 'celltype') ? glow * 0.8 : glow * 1.5;
    tempColor.copy(glowColor).multiplyScalar(glowMul);
    glowMesh.setColorAt(glowCount, tempColor);
    glowCount++;
  }
  // Dying nodes get a strong bloom flash during early animation
  for (const d of dyingNodes) {
    const t = d.age / DEATH_ANIM_FRAMES;
    if (t > 0.4 || glowCount >= MAX_NODES) continue;
    const bloomIntensity = t < 0.15 ? 2.5 * (1 - t / 0.15) : (0.4 - t) * 2;
    if (bloomIntensity <= 0) continue;
    const scale = t < 0.2 ? d.size * (1 + t * 2) : d.size * Math.max(0, 1 - (t - 0.2) * 1.25);
    dummy.position.set(d.px, d.py, d.pz);
    dummy.scale.setScalar(scale * popScale * 1.5);
    dummy.updateMatrix();
    glowMesh.setMatrixAt(glowCount, dummy.matrix);
    tempColor.copy(d.color).multiplyScalar(bloomIntensity);
    tempColor.lerp(WHITE, 0.5);
    glowMesh.setColorAt(glowCount, tempColor);
    glowCount++;
  }

  glowMesh.count = glowCount;
  if (glowCount > 0) {
    glowMesh.instanceMatrix.needsUpdate = true;
    glowMesh.instanceColor.needsUpdate = true;
  }

  nodesMesh.instanceMatrix.needsUpdate = true;
  nodesMesh.instanceColor.needsUpdate = true;

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
    // V3: When a node is selected, color incoming edges by formation cause (provenance)
    const isIncoming = hasSelection && e.to === selectedNodeId;
    if (isIncoming && colorMode === 'celltype') {
      const pc = getProvenanceColor(e.formation_cause);
      const brightness = 0.3 + w * 0.4;
      r = pc.r * brightness; g = pc.g * brightness; b = pc.b * brightness;
    } else if (isHighlighted) {
      r = 0.5; g = 0.7; b = 1.0;
    } else {
      // Base brightness from weight + heat boost from recent activity
      const brightness = (0.05 + w * 0.15 + heat * 0.25) * (1.0 - edgeDimFactor * 0.9);
      if (e.weight < 0) {
        // Inhibitory synapses: cool magenta-red so they're visually distinct from excitatory
        r = 0.85 * brightness;
        g = 0.10 * brightness;
        b = 0.45 * brightness;
      } else {
        r = sourceColor.r * brightness;
        g = sourceColor.g * brightness;
        b = sourceColor.b * brightness;
      }
      // Hot synapses shift toward white regardless of sign
      if (heat > 0.15) {
        const heatWhite = heat * 0.15;
        r += heatWhite; g += heatWhite; b += heatWhite;
      }
    }

    if (e.consolidated && edgeDimFactor < 0.3) { r += 0.05; g += 0.04; b += 0.02; }
    // V3: justified synapses get subtle gold tint
    if (e.justified && edgeDimFactor < 0.3) { r += 0.03; g += 0.025; }

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

  // V3: Cluster hulls
  updateClusterHulls(nodes, edges);
  // V3: Detect reinforcement pulses (compare edge data between frames)
  detectReinforcementPulses();

  // === SPIKE SPAWNING from real engine firing data ===
  // Up to MAX_SPIKES_PER_MORPHON spikes per firing morphon (top edges by |weight|),
  // with cooldown to prevent stacking. Positions are snapshotted at spawn so migration
  // during flight doesn't warp the trajectory.
  const MAX_SPIKES_PER_MORPHON = 3;
  const SPIKE_WEIGHT_THRESHOLD = 0.12; // skip near-zero edges

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

    // Collect top N outgoing edges by |weight| — reuse pooled array, hoisted comparator
    _spikeCandidates.length = 0;
    for (const e of edges) {
      if (e.from !== id) continue;
      const aw = Math.abs(e.weight);
      if (aw < SPIKE_WEIGHT_THRESHOLD) continue;
      const toIdx = nodeMap.get(e.to);
      if (toIdx !== undefined) _spikeCandidates.push({ toIdx, weight: e.weight, aw });
    }
    _spikeCandidates.sort(_spikeCandSortFn);

    const fi = fromIdx * 3;
    let spawned = 0;
    for (const cand of _spikeCandidates) {
      if (spawned >= MAX_SPIKES_PER_MORPHON) break;
      if (liveSpikes.length >= MAX_SPIKES * 0.7) break;
      const ti = cand.toIdx * 3;
      liveSpikes.push({
        fromIdx, toIdx: cand.toIdx,
        // Snapshot world positions at spawn — immune to migration warping mid-flight
        fx: nodePositions[fi],   fy: nodePositions[fi+1], fz: nodePositions[fi+2],
        tx: nodePositions[ti],   ty: nodePositions[ti+1], tz: nodePositions[ti+2],
        age: 0,
        // Jitter lifetime ±8 frames so synchronized bursts don't all die at once
        lifetime: SPIKE_VISUAL_FRAMES + Math.floor(Math.random() * 17) - 8,
        // Shared color references — never mutated, so no clone needed
        color: cand.weight < 0 ? INHIBITORY_SPIKE_COLOR : color,
        strength: cand.aw,
      });
      spawned++;
    }
    if (spawned > 0) spikeCooldowns.set(id, SPIKE_COOLDOWN);
  }
}

// ============================================================
// UI UPDATES
// ============================================================
function updatePanels() {
  if (!system) return;
  // Cache all parsed JSON for this update cycle
  cachedStats = JSON.parse(system.inspect());
  cachedMod = JSON.parse(system.modulation_json());
  try {
    cachedEndo = JSON.parse(system.endo_json());
  } catch(e) {
    if (!system._endoWarned) {
      console.error('endo_json failed:', e.message, '— rebuild WASM and hard-refresh (Cmd+Shift+R)');
      system._endoWarned = true;
    }
  }
  if (system.governance_json) cachedGov = JSON.parse(system.governance_json());
  try { cachedLearning = JSON.parse(system.learning_json()); } catch(_) {}

  const stats = cachedStats;
  const mod = cachedMod;

  dom.hStep.textContent = stats.step_count;
  dom.hFired.textContent = frameFired.size;
  // Load level
  if (dom.hLoad) {
    const loadPct = Math.min(Math.round(frameLoadEma * 100), 999);
    dom.hLoad.textContent = loadPct + '%';
    dom.hLoad.style.color = loadPct > 80 ? '#ef4444' : loadPct > 50 ? '#fbbf24' : '#34d399';
  }
  // Uptime since last reset
  const uptimeSec = Math.floor((performance.now() - systemStartTime) / 1000);
  const m = Math.floor(uptimeSec / 60), s = uptimeSec % 60;
  const h = Math.floor(m / 60);
  dom.hUptime.textContent = h > 0
    ? `${h}:${String(m % 60).padStart(2, '0')}:${String(s).padStart(2, '0')}`
    : `${m}:${String(s).padStart(2, '0')}`;

  dom.sMorphons.textContent = stats.total_morphons;
  dom.sSynapses.textContent = stats.total_synapses;
  dom.sClusters.textContent = stats.fused_clusters;
  dom.sGen.textContent = stats.max_generation;
  dom.sFiring.textContent = (stats.firing_rate * 100).toFixed(1) + '%';
  dom.sFired.textContent = frameFired.size;
  dom.sEnergy.textContent = stats.avg_energy.toFixed(2);
  dom.sError.textContent = stats.avg_prediction_error.toFixed(3);
  dom.sFieldPeMax.textContent = (stats.field_pe_max || 0).toFixed(3);
  dom.sFieldPeMean.textContent = (stats.field_pe_mean || 0).toFixed(3);
  dom.sWmem.textContent = stats.working_memory_items;
  dom.sBorn.textContent = stats.total_born || 0;
  dom.sDied.textContent = stats.total_died || 0;
  dom.sTransdiff.textContent = stats.total_transdifferentiations || 0;

  const counts = stats.differentiation_map || {};
  // CellType::Fused is never assigned — fused morphons keep their original cell_type and
  // only get fused_with set. Count them from topology nodeData instead.
  counts['Fused'] = nodeData.filter(n => n.fused).length;
  for (const type of ['Stem', 'Sensory', 'Associative', 'Motor', 'Modulatory', 'Fused']) {
    const el = dom.ctMap[type];
    if (el) el.textContent = counts[type] || 0;
  }

  setModBarEl(dom.modReward, dom.modRewardV, mod.reward);
  setModBarEl(dom.modNovelty, dom.modNoveltyV, mod.novelty);
  setModBarEl(dom.modArousal, dom.modArousalV, mod.arousal);
  setModBarEl(dom.modHomeo, dom.modHomeoV, mod.homeostasis);

  // Endoquilibrium panel
  if (!cachedEndo && dom.endoStage) {
    dom.endoStage.textContent = 'N/A';
    dom.endoStage.className = 'endo-stage-badge';
  }
  if (cachedEndo) {
    const endo = cachedEndo;
    if (!endo.enabled) {
      dom.endoStage.textContent = 'OFF';
      dom.endoStage.className = 'endo-stage-badge';
      dom.endoHealthBar.style.width = '0%';
      dom.endoHealthV.textContent = '---';
      dom.endoChannels.style.opacity = '0.3';
      dom.endoInterventions.innerHTML =
        '<div class="endo-disabled-msg">Endo disabled in config</div>';
    } else {
      const stage = endo.stage.toLowerCase();
      dom.endoStage.textContent = endo.stage;
      dom.endoStage.className = 'endo-stage-badge ' + stage;

      const hp = Math.max(0, Math.min(1, endo.health_score));
      dom.endoHealthBar.style.width = (hp * 100) + '%';
      const hpHue = hp * 120;
      dom.endoHealthBar.style.background = `hsl(${hpHue}, 70%, 50%)`;
      dom.endoHealthV.textContent = hp.toFixed(2);
      dom.endoHealthV.style.color = hp < 0.5 ? '#ef4444' : hp < 0.8 ? '#fbbf24' : 'var(--text-dim)';

      dom.endoChannels.style.opacity = '1';
      const ch = endo.channels;
      dom.endoRg.textContent = ch.reward_gain.toFixed(2);
      dom.endoNg.textContent = ch.novelty_gain.toFixed(2);
      dom.endoAg.textContent = ch.arousal_gain.toFixed(2);
      dom.endoHg.textContent = ch.homeostasis_gain.toFixed(2);
      dom.endoTb.textContent = ch.threshold_bias.toFixed(3);
      dom.endoPm.textContent = ch.plasticity_mult.toFixed(2);
      dom.endoCg.textContent = ch.consolidation_gain.toFixed(2);

      // Color deviant channels
      const colorVal = (el, val, neutral) => {
        const dev = Math.abs(val - neutral);
        if (dev > 0.3) el.style.color = val > neutral ? '#fbbf24' : '#508cff';
        else el.style.color = 'var(--text-bright)';
      };
      colorVal(dom.endoRg, ch.reward_gain, 1.0);
      colorVal(dom.endoNg, ch.novelty_gain, 1.0);
      colorVal(dom.endoAg, ch.arousal_gain, 1.0);
      colorVal(dom.endoHg, ch.homeostasis_gain, 1.0);
      colorVal(dom.endoPm, ch.plasticity_mult, 1.0);
      colorVal(dom.endoCg, ch.consolidation_gain, 1.0);
      if (Math.abs(ch.threshold_bias) > 0.05) {
        dom.endoTb.style.color = ch.threshold_bias > 0 ? '#fbbf24' : '#508cff';
      } else {
        dom.endoTb.style.color = 'var(--text-bright)';
      }

      // Active interventions
      if (endo.interventions.length === 0) {
        dom.endoInterventions.innerHTML = '';
      } else {
        dom.endoInterventions.innerHTML = endo.interventions.map(iv =>
          `<div class="endo-intervention" title="${iv.vital}: ${iv.actual.toFixed(3)} → ${iv.setpoint.toFixed(3)}">${iv.rule} → ${iv.lever}</div>`
        ).join('');
      }
    }
  }

  // V3: Governance panel
  if (cachedGov) {
    const gov = cachedGov;
    dom.sJustified.textContent = (gov.justified_fraction * 100).toFixed(0) + '%';
    dom.sConsolidated.textContent = (gov.consolidated_fraction * 100).toFixed(0) + '%';
    dom.sSkepticism.textContent = gov.avg_skepticism.toFixed(2);
    const cs = gov.cluster_states;
    dom.sEpistemic.textContent =
      `H${cs.hypothesis} S${cs.supported} O${cs.outdated} C${cs.contested}`;

    justifiedHistory.push(gov.justified_fraction);
    if (justifiedHistory.length > MAX_HISTORY) justifiedHistory.shift();
    drawSparkline('spark-justified', justifiedHistory, '#eab308');
  }

  // Sparklines
  firingHistory.push(stats.firing_rate);
  if (firingHistory.length > MAX_HISTORY) firingHistory.shift();
  drawSparkline('spark-firing', firingHistory, '#508cff');

  morphonHistory.push(stats.total_morphons);
  if (morphonHistory.length > MAX_HISTORY) morphonHistory.shift();
  drawSparkline('spark-morphons', morphonHistory, '#34d399');

  synapseHistory.push(stats.total_synapses);
  if (synapseHistory.length > MAX_HISTORY) synapseHistory.shift();
  drawSparkline('spark-synapses', synapseHistory, '#a78bfa');

  fieldPeHistory.push(stats.field_pe_mean || 0);
  if (fieldPeHistory.length > MAX_HISTORY) fieldPeHistory.shift();
  drawSparkline('spark-field-pe', fieldPeHistory, '#ef4444');

  // Motor output bar chart
  updateMotorOutput();

  updateGraph(stats);
  if (selectedNodeId !== null) updateDetailPanel();

  // V3: Cluster list + timeline
  updateClusterList();
  updateLearningPanel();
  recordTimelineSnapshot();
}

function updateMotorOutput() {
  if (!system) return;
  const output = system.read_output();
  const container = dom.motorOutput;
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
function setModBarEl(barEl, valEl, value) {
  barEl.style.width = (value * 100) + '%';
  valEl.textContent = value.toFixed(2);
}

function drawSparkline(canvasId, data, color) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  // Match backing buffer to CSS size for crisp rendering
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const bw = Math.floor(rect.width * dpr);
  const bh = Math.floor(rect.height * dpr);
  if (bw <= 0 || bh <= 0) return;
  if (canvas.width !== bw || canvas.height !== bh) {
    canvas.width = bw;
    canvas.height = bh;
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const w = rect.width, h = rect.height;

  ctx.clearRect(0, 0, w, h);
  if (data.length < 2) return;

  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || max || 1; // avoid div-by-zero for flat data
  const pad = 3; // px padding top/bottom

  function yFor(v) {
    return h - pad - ((v - min) / range) * (h - pad * 2 - 10);
  }

  // Gradient fill
  ctx.beginPath();
  for (let i = 0; i < data.length; i++) {
    const x = (i / (data.length - 1)) * w;
    i === 0 ? ctx.moveTo(x, yFor(data[i])) : ctx.lineTo(x, yFor(data[i]));
  }
  ctx.lineTo(w, h); ctx.lineTo(0, h); ctx.closePath();
  const grad = ctx.createLinearGradient(0, 0, 0, h);
  grad.addColorStop(0, color + '30');
  grad.addColorStop(1, color + '05');
  ctx.fillStyle = grad;
  ctx.fill();

  // Line
  ctx.beginPath();
  for (let i = 0; i < data.length; i++) {
    const x = (i / (data.length - 1)) * w;
    i === 0 ? ctx.moveTo(x, yFor(data[i])) : ctx.lineTo(x, yFor(data[i]));
  }
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.stroke();

  // Current value label
  const last = data[data.length - 1];
  const label = last < 1 ? (last * 100).toFixed(1) + '%' : last.toFixed(0);
  ctx.font = '9px "JetBrains Mono", monospace';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'top';
  ctx.fillStyle = color;
  ctx.fillText(label, w - 2, 2);
}

function updateDetailPanel() {
  if (selectedNodeId === null) { dom.detailPanel.classList.remove('visible'); return; }
  const node = nodeData[nodeMap.get(selectedNodeId)];
  if (!node) { dom.detailPanel.classList.remove('visible'); selectedNodeId = null; return; }

  dom.detailPanel.classList.add('visible');
  dom.dId.textContent = '#' + node.id;
  dom.dType.textContent = node.cell_type;
  dom.dDot.style.background = CELL_COLORS[node.cell_type]?.getStyle() || '#888';
  dom.dGen.textContent = node.generation;
  dom.dAge.textContent = node.age;
  dom.dEnergy.textContent = node.energy.toFixed(2);
  dom.dEnergyBar.style.width = (node.energy * 100) + '%';
  dom.dPotential.textContent = node.potential.toFixed(3);
  dom.dThreshold.textContent = node.threshold.toFixed(3);
  dom.dDiff.textContent = node.differentiation.toFixed(2);
  dom.dDiffBar.style.width = (node.differentiation * 100) + '%';
  dom.dError.textContent = node.prediction_error.toFixed(3);
  dom.dDesire.textContent = node.desire.toFixed(3);
  dom.dFired.textContent = node.fired ? 'YES' : '-';

  let inCount = 0, outCount = 0, justifiedCount = 0, reinforcedCount = 0;
  for (const e of edgeData) {
    if (e.to === selectedNodeId) {
      inCount++;
      if (e.justified) justifiedCount++;
      if (e.reinforcement_count > 0) reinforcedCount++;
    }
    if (e.from === selectedNodeId) outCount++;
  }
  dom.dConns.textContent = `${inCount}\u2193 ${outCount}\u2191`;

  // V3: Epistemic state for clustered morphons
  if (node.fused && node.epistemic_state !== 'none') {
    dom.dEpistemicRow.style.display = '';
    dom.dSkepticismRow.style.display = '';
    dom.dJustifiedRow.style.display = '';
    dom.dEpistemic.textContent = node.epistemic_state;
    dom.dSkepticism.textContent = node.skepticism.toFixed(2);
    dom.dJustified.textContent = `${justifiedCount}/${inCount} in, ${reinforcedCount} reinforced`;
  } else {
    dom.dEpistemicRow.style.display = 'none';
    dom.dSkepticismRow.style.display = 'none';
    dom.dJustifiedRow.style.display = inCount > 0 ? '' : 'none';
    if (inCount > 0) {
      dom.dJustified.textContent = `${justifiedCount}/${inCount} justified`;
    }
  }
}

// ============================================================
// EVENT LOG
// ============================================================
let lastMorphonCount = 0;
let lastSynapseCount = 0;
let lastEpistemicKey = null;  // V3: track epistemic state changes for logging

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

  // V3: Log epistemic state transitions
  if (cachedGov) {
    const gov = cachedGov;
    const cs = gov.cluster_states;
    const key = `H${cs.hypothesis}S${cs.supported}O${cs.outdated}C${cs.contested}`;
    if (lastEpistemicKey && key !== lastEpistemicKey) {
      if (cs.supported > 0) addEvent(step, `Cluster(s) → Supported`, 'event-other');
      if (cs.outdated > 0) addEvent(step, `Cluster(s) → Outdated`, 'event-death');
      if (cs.contested > 0) addEvent(step, `Cluster(s) → Contested`, 'event-death');
    }
    lastEpistemicKey = key;
  }
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
// SKETCH ARENA
// ============================================================
function initArena() {
  const canvas = document.getElementById('arena-grid-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const cellW = canvas.width / ARENA_GRID;
  const cellH = canvas.height / ARENA_GRID;

  function cellFromEvent(e) {
    const rect = canvas.getBoundingClientRect();
    const x = Math.floor((e.clientX - rect.left) / rect.width * ARENA_GRID);
    const y = Math.floor((e.clientY - rect.top) / rect.height * ARENA_GRID);
    if (x >= 0 && x < ARENA_GRID && y >= 0 && y < ARENA_GRID) return y * ARENA_GRID + x;
    return -1;
  }

  function paintCell(e) {
    const idx = cellFromEvent(e);
    if (idx < 0) return;
    arenaPixels[idx] = arenaErasing ? 0 : 1;
    drawArenaGrid();
  }

  canvas.addEventListener('mousedown', (e) => {
    e.preventDefault();
    arenaErasing = e.button === 2 || e.shiftKey;
    arenaDrawing = true;
    paintCell(e);
  });
  canvas.addEventListener('mousemove', (e) => { if (arenaDrawing) paintCell(e); });
  canvas.addEventListener('mouseup', () => { arenaDrawing = false; });
  canvas.addEventListener('mouseleave', () => { arenaDrawing = false; });
  canvas.addEventListener('contextmenu', (e) => e.preventDefault());

  // Class buttons
  document.querySelectorAll('.arena-class-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.arena-class-btn').forEach(b => b.classList.remove('selected'));
      btn.classList.add('selected');
      arenaSelectedClass = parseInt(btn.dataset.class);
    });
  });

  // Teach
  document.getElementById('arena-teach').addEventListener('click', () => {
    if (!system || occupation !== 'arena') return;
    const hasPixels = arenaPixels.some(v => v > 0);
    if (!hasPixels) { setArenaStatus('Draw something first'); return; }
    arenaTrainingSet.push({ pixels: new Float64Array(arenaPixels), label: arenaSelectedClass });
    // Immediate burst training: feed + reward 20 times
    for (let i = 0; i < 20; i++) {
      system.feed_input(arenaPixels);
      for (let s = 0; s < 5; s++) system.step();
      system.reward_contrastive(arenaSelectedClass, 0.8, 0.3);
      system.teach_supervised(arenaSelectedClass, 0.05);
      for (let s = 0; s < 3; s++) system.step();
    }
    setArenaStatus(`Taught class ${ARENA_CLASS_NAMES[arenaSelectedClass]} (${arenaTrainingSet.length} samples)`);
    addEvent('', `Arena: taught class ${ARENA_CLASS_NAMES[arenaSelectedClass]}`, 'event-birth');
    updateArenaStats();
  });

  // Guess
  document.getElementById('arena-guess').addEventListener('click', () => {
    if (!system || occupation !== 'arena') return;
    const hasPixels = arenaPixels.some(v => v > 0);
    if (!hasPixels) { setArenaStatus('Draw something first'); return; }
    arenaGuess(arenaPixels);
  });

  // Clear
  document.getElementById('arena-clear').addEventListener('click', () => {
    arenaPixels.fill(0);
    drawArenaGrid();
    setArenaStatus('Draw a pattern');
  });

  drawArenaGrid();
}

function drawArenaGrid() {
  const canvas = document.getElementById('arena-grid-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const cellW = canvas.width / ARENA_GRID;
  const cellH = canvas.height / ARENA_GRID;
  const classColor = ARENA_CLASS_COLORS[arenaSelectedClass];

  ctx.fillStyle = '#0a0f1c';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  for (let y = 0; y < ARENA_GRID; y++) {
    for (let x = 0; x < ARENA_GRID; x++) {
      const idx = y * ARENA_GRID + x;
      if (arenaPixels[idx] > 0) {
        ctx.fillStyle = classColor;
        ctx.globalAlpha = 0.85;
        ctx.fillRect(x * cellW + 0.5, y * cellH + 0.5, cellW - 1, cellH - 1);
        ctx.globalAlpha = 1;
      }
    }
  }

  // Grid lines
  ctx.strokeStyle = 'rgba(80,140,255,0.08)';
  ctx.lineWidth = 0.5;
  for (let i = 1; i < ARENA_GRID; i++) {
    ctx.beginPath(); ctx.moveTo(i * cellW, 0); ctx.lineTo(i * cellW, canvas.height); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, i * cellH); ctx.lineTo(canvas.width, i * cellH); ctx.stroke();
  }
}

function arenaGuess(pixels) {
  if (!system) return;
  // Feed and propagate
  system.feed_input(pixels);
  for (let i = 0; i < 10; i++) system.step();
  const output = system.read_output();
  arenaLastOutput = output;

  // Softmax for confidence
  const max = Math.max(...output);
  const exps = output.map(v => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  const probs = exps.map(v => v / sum);

  // Find winner
  let best = 0;
  for (let i = 1; i < probs.length; i++) {
    if (probs[i] > probs[best]) best = i;
  }

  // Update prediction bars
  for (let i = 0; i < ARENA_CLASSES; i++) {
    const pct = Math.round(probs[i] * 100);
    const bar = document.getElementById(`arena-bar-${i}`);
    const val = document.getElementById(`arena-val-${i}`);
    if (bar) bar.style.width = `${pct}%`;
    if (val) val.textContent = `${pct}%`;
  }

  const resultEl = document.getElementById('arena-result');
  if (resultEl) {
    resultEl.textContent = ARENA_CLASS_NAMES[best];
    resultEl.style.color = ARENA_CLASS_COLORS[best];
  }

  setArenaStatus(`Predicted: ${ARENA_CLASS_NAMES[best]} (${Math.round(probs[best] * 100)}%)`);
  return best;
}

function arenaTrainStep() {
  // Called each frame when arena is active and training set is non-empty.
  // Cycles through training examples continuously.
  if (!system || arenaTrainingSet.length === 0) return;

  const sample = arenaTrainingSet[arenaTrainIdx % arenaTrainingSet.length];
  system.feed_input(sample.pixels);
  // Let it propagate for a few steps before reward
  for (let s = 0; s < 3; s++) system.step();
  system.reward_contrastive(sample.label, 0.6, 0.2);
  system.teach_supervised(sample.label, 0.03);

  arenaTrainSubStep++;
  if (arenaTrainSubStep >= 5) {
    arenaTrainSubStep = 0;
    arenaTrainIdx++;
    if (arenaTrainIdx % arenaTrainingSet.length === 0) {
      arenaCycleCount++;
      // Every 5 cycles, evaluate accuracy
      if (arenaCycleCount % 5 === 0) arenaEvaluate();
    }
  }
}

function arenaEvaluate() {
  if (!system || arenaTrainingSet.length === 0) return;
  let correct = 0;
  // Reset confusion
  for (let i = 0; i < ARENA_CLASSES; i++) arenaConfusion[i].fill(0);

  for (const sample of arenaTrainingSet) {
    system.feed_input(sample.pixels);
    for (let s = 0; s < 8; s++) system.step();
    const output = system.read_output();
    let best = 0;
    for (let i = 1; i < output.length; i++) {
      if (output[i] > output[best]) best = i;
    }
    if (best === sample.label) correct++;
    if (sample.label < ARENA_CLASSES && best < ARENA_CLASSES) {
      arenaConfusion[sample.label][best]++;
    }
  }

  const acc = correct / arenaTrainingSet.length;
  arenaAccHistory.push(acc);
  if (arenaAccHistory.length > 120) arenaAccHistory.shift();
  updateArenaStats();
}

function updateArenaStats() {
  const samplesEl = document.getElementById('arena-samples');
  const accEl = document.getElementById('arena-accuracy');
  const cyclesEl = document.getElementById('arena-cycles');
  if (samplesEl) samplesEl.textContent = arenaTrainingSet.length;
  if (cyclesEl) cyclesEl.textContent = arenaCycleCount;
  if (accEl) {
    if (arenaAccHistory.length > 0) {
      const last = arenaAccHistory[arenaAccHistory.length - 1];
      accEl.textContent = `${Math.round(last * 100)}%`;
      accEl.style.color = last > 0.7 ? '#34d399' : last > 0.4 ? '#fbbf24' : '#ef4444';
    } else {
      accEl.textContent = '—';
    }
  }
  drawArenaAccSparkline();
  drawArenaConfusion();
}

function drawArenaAccSparkline() {
  const canvas = document.getElementById('arena-acc-spark');
  if (!canvas || arenaAccHistory.length < 2) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  const data = arenaAccHistory;
  const step = w / (data.length - 1);

  ctx.beginPath();
  ctx.moveTo(0, h - data[0] * h);
  for (let i = 1; i < data.length; i++) {
    ctx.lineTo(i * step, h - data[i] * h);
  }
  ctx.strokeStyle = '#508cff';
  ctx.lineWidth = 1.5;
  ctx.stroke();

  // Fill
  ctx.lineTo((data.length - 1) * step, h);
  ctx.lineTo(0, h);
  ctx.closePath();
  const grad = ctx.createLinearGradient(0, 0, 0, h);
  grad.addColorStop(0, 'rgba(80,140,255,0.15)');
  grad.addColorStop(1, 'rgba(80,140,255,0.0)');
  ctx.fillStyle = grad;
  ctx.fill();
}

function drawArenaConfusion() {
  const canvas = document.getElementById('arena-confusion');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width, h = canvas.height;
  const cellW = w / ARENA_CLASSES, cellH = h / ARENA_CLASSES;
  ctx.clearRect(0, 0, w, h);

  // Find max for normalization
  let maxVal = 1;
  for (let i = 0; i < ARENA_CLASSES; i++)
    for (let j = 0; j < ARENA_CLASSES; j++)
      maxVal = Math.max(maxVal, arenaConfusion[i][j]);

  for (let row = 0; row < ARENA_CLASSES; row++) {
    for (let col = 0; col < ARENA_CLASSES; col++) {
      const v = arenaConfusion[row][col] / maxVal;
      if (row === col) {
        // Diagonal: green intensity
        ctx.fillStyle = `rgba(52,211,153,${0.1 + v * 0.7})`;
      } else {
        // Off-diagonal: red intensity
        ctx.fillStyle = `rgba(239,68,68,${v * 0.6})`;
      }
      ctx.fillRect(col * cellW, row * cellH, cellW - 1, cellH - 1);

      // Count label
      const count = arenaConfusion[row][col];
      if (count > 0) {
        ctx.fillStyle = 'rgba(255,255,255,0.7)';
        ctx.font = '9px JetBrains Mono';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(count, col * cellW + cellW / 2, row * cellH + cellH / 2);
      }
    }
  }

  // Axis labels
  ctx.fillStyle = 'rgba(255,255,255,0.3)';
  ctx.font = '8px JetBrains Mono';
  ctx.textAlign = 'center';
  for (let i = 0; i < ARENA_CLASSES; i++) {
    ctx.fillText(ARENA_CLASS_NAMES[i], i * cellW + cellW / 2, h - 1);
    ctx.textAlign = 'right';
    ctx.fillText(ARENA_CLASS_NAMES[i], cellW * 0.15, i * cellH + cellH / 2 + 3);
    ctx.textAlign = 'center';
  }
}

function setArenaStatus(msg) {
  const el = document.getElementById('arena-status');
  if (el) el.textContent = msg;
}

function switchOccupation(newOcc) {
  if (newOcc === occupation) return;
  occupation = newOcc;
  const arenaTab = document.getElementById('tab-btn-arena');

  if (occupation === 'arena') {
    // Create arena-optimized system: 8x8=64 inputs, 4 outputs
    if (system) { try { system.free(); } catch(_) {} }
    system = WasmSystem.newWithIO(100, 'cerebellar', 3, 64, 4);
    system.enable_analog_readout();
    // Warm up
    for (let i = 0; i < 30; i++) { makeInput('noise'); system.step(); }
    // Reset arena state
    arenaPixels.fill(0);
    arenaTrainingSet.length = 0;
    arenaAccHistory.length = 0;
    for (let i = 0; i < ARENA_CLASSES; i++) arenaConfusion[i].fill(0);
    arenaTrainIdx = 0; arenaCycleCount = 0; arenaTrainSubStep = 0;
    arenaLastOutput = null;
    // Show arena tab, switch to it
    if (arenaTab) arenaTab.style.display = '';
    activateTab('tab-arena');
    drawArenaGrid();
    updateArenaStats();
    setArenaStatus('Draw a pattern, pick a class, teach!');
    addEvent(0, 'Sketch Arena activated [cerebellar, 64in/4out]', 'event-diff');
  } else {
    // Back to idle
    if (system) { try { system.free(); } catch(_) {} }
    const program = document.getElementById('program-select').value;
    system = new WasmSystem(60, program, 3);
    for (let i = 0; i < 20; i++) { makeInput('noise'); system.step(); }
    if (arenaTab) arenaTab.style.display = 'none';
    activateTab('tab-log');
    addEvent(0, 'Returned to idle mode', 'event-diff');
  }

  // Reset shared visualization state
  systemStartTime = performance.now();
  selectedNodeId = null; hoveredNodeId = null;
  connectedToSelected.clear();
  nodeDim.clear(); nodeDimTarget.clear(); nodeGlow.clear();
  lastSpikeCount = 0; stepAccumulator = 0;
  dyingNodes.length = 0; prevNodeIds.clear(); nodeEpistemicColor.clear(); nodePeSmooth.clear();
  rasterScrollX = 0; rasterMorphonCount = 0;
  groupRateHistory = []; heatmapBinAccum = []; heatmapBinSteps = 0;
  morphonOrder = []; morphonYMap.clear(); rasterGroups = [];
  firingHistory.length = 0; synapseHistory.length = 0; fieldPeHistory.length = 0; justifiedHistory.length = 0;
  timelineSnapshots.length = 0; timelineLastStep = 0; timelineScrubbing = false;
  prevEdgeReinforcementCounts.clear(); learningPulses.length = 0;
  lastEpistemicKey = null;
  for (const [_, m] of clusterHullMeshes) { scene.remove(m.line); if (m.fill) scene.remove(m.fill); }
  clusterHullMeshes.clear(); clusterHullCache.clear();
  for (const k in graphData) graphData[k].length = 0;
  lastBorn = 0; lastDied = 0; lastMorphonCount = 0; lastSynapseCount = 0;
  liveSpikes.length = 0; spikeCooldowns.clear();
  updateScene(); updatePanels();
}

function activateTab(tabId) {
  document.querySelectorAll('#bottom-panel .tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('#bottom-panel .tab-content').forEach(c => c.classList.remove('active'));
  const btn = document.querySelector(`#bottom-panel .tab-btn[data-tab="${tabId}"]`);
  if (btn) btn.classList.add('active');
  const tab = document.getElementById(tabId);
  if (tab) tab.classList.add('active');
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
    // If in arena, reset re-creates the arena system
    if (occupation === 'arena') {
      switchOccupation('arena'); // re-init arena
      return;
    }
    filterCellType = null;
    clearCellTypeActive();
    if (system) { try { system.free(); } catch(_) {} } // free WASM memory
    const program = document.getElementById('program-select').value;
    system = new WasmSystem(60, program, 3);
    systemStartTime = performance.now();
    selectedNodeId = null; hoveredNodeId = null;
    connectedToSelected.clear();
    nodeDim.clear(); nodeDimTarget.clear(); nodeGlow.clear();
    lastSpikeCount = 0; stepAccumulator = 0; liveSpikes.length = 0; spikeCooldowns.clear();
    dyingNodes.length = 0; prevNodeIds.clear(); nodeEpistemicColor.clear(); nodePeSmooth.clear();
    rasterScrollX = 0; rasterMorphonCount = 0;
    groupRateHistory = []; heatmapBinAccum = []; heatmapBinSteps = 0;
    morphonOrder = []; morphonYMap.clear(); rasterGroups = [];
    firingHistory.length = 0;
    synapseHistory.length = 0;
    fieldPeHistory.length = 0;
    justifiedHistory.length = 0;
    // V3: Reset timeline and cluster hulls
    timelineSnapshots.length = 0; timelineLastStep = 0; timelineScrubbing = false;
    prevEdgeReinforcementCounts.clear(); learningPulses.length = 0;
    lastEpistemicKey = null;
    for (const [_, m] of clusterHullMeshes) { scene.remove(m.line); if (m.fill) scene.remove(m.fill); }
    clusterHullMeshes.clear(); clusterHullCache.clear();
    // Reset graph
    for (const k in graphData) graphData[k].length = 0;
    lastBorn = 0; lastDied = 0;
    lastMorphonCount = 0; lastSynapseCount = 0;
    document.getElementById('events').innerHTML = '';
    addEvent(0, `System reset [${program}]`, 'event-diff');
    for (let i = 0; i < 20; i++) { makeInput('noise'); system.step(); }
    updateScene(); updatePanels();
  }

  document.getElementById('btn-reset').addEventListener('click', resetSystem);
  document.getElementById('program-select').addEventListener('change', resetSystem);

  // Occupation selector
  document.getElementById('occupation-select').addEventListener('change', (e) => {
    switchOccupation(e.target.value);
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

  document.getElementById('btn-dream').addEventListener('click', () => {
    if (system) { system.trigger_dream(); addEvent('', 'Dream cycle triggered', 'event-other'); }
  });
  document.getElementById('btn-reset-v').addEventListener('click', () => {
    if (system) { system.reset_voltages(); addEvent('', 'Voltages reset', 'event-other'); }
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

  // Heatmap resolution buttons
  document.querySelectorAll('.heatmap-res-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.heatmap-res-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      heatmapBinSize = parseInt(btn.dataset.res);
      // Reset accumulators for clean transition
      heatmapBinAccum = new Array(rasterGroups.length).fill(0);
      heatmapBinSteps = 0;
    });
  });

  // Pause heatmap on hover
  document.getElementById('raster-canvas')?.addEventListener('mouseenter', () => { heatmapPaused = true; });
  document.getElementById('raster-canvas')?.addEventListener('mouseleave', () => { heatmapPaused = false; });

  // Graph scale buttons
  document.querySelectorAll('.graph-scale-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.graph-scale-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      setGraphScale(btn.dataset.scale);
    });
  });

  // Graph window buttons
  document.querySelectorAll('.graph-window-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.graph-window-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      graphWindowSize = parseInt(btn.dataset.win);
    });
  });

  // Maximize / restore
  document.getElementById('btn-panel-maximize')?.addEventListener('click', () => {
    const panel = document.getElementById('bottom-panel');
    panel.classList.remove('no-transition');
    panel.classList.toggle('maximized');
    const isMax = panel.classList.contains('maximized');
    document.getElementById('btn-panel-maximize').textContent = isMax ? '\u25BD' : '\u25A1';
    if (!isMax) document.documentElement.style.setProperty('--panel-h', '160px');
    setTimeout(() => { onResize(); }, 260);
  });
  // Clear: context-dependent (raster or log)
  document.getElementById('btn-panel-clear')?.addEventListener('click', () => {
    const activeTab = document.querySelector('#bottom-panel .tab-btn.active')?.dataset.tab;
    if (activeTab === 'tab-raster') {
      rasterScrollX = 0;
      groupRateHistory.forEach(arr => arr.fill(0));
      heatmapBinAccum.fill(0); heatmapBinSteps = 0;
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
    const menu = document.getElementById('log-menu');
    const btn = e.currentTarget;
    if (menu) {
      menu.classList.toggle('hidden');
      if (!menu.classList.contains('hidden')) {
        const r = btn.getBoundingClientRect();
        menu.style.top = (r.bottom + 4) + 'px';
        menu.style.right = (window.innerWidth - r.right) + 'px';
      }
    }
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

  // Export as .txt file download
  document.getElementById('btn-log-export')?.addEventListener('click', () => {
    const lines = [];
    document.querySelectorAll('#events .event-item').forEach(el => lines.push(el.textContent.trim()));
    const blob = new Blob([lines.join('\n')], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `morphon-log-${Date.now()}.txt`;
    a.click();
    URL.revokeObjectURL(url);
    addEvent('', `Exported ${lines.length} log entries`, 'event-diff');
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

  // Color mode toggle (Cell Type / PE Heatmap)
  document.querySelectorAll('.color-mode-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.color-mode-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      colorMode = btn.dataset.mode;
      // Clear cell type filter when switching to PE/epistemic mode
      if (colorMode === 'pe' || colorMode === 'epistemic') {
        filterCellType = null;
        clearCellTypeActive();
      }
    });
  });

  // Cell type filtering — click to highlight, click again to clear
  document.querySelectorAll('.cell-type-row[data-type]').forEach(row => {
    row.addEventListener('click', () => {
      if (colorMode !== 'celltype') return;
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
      case '1': case '2': case '3': {
        const modes = { '1': 'celltype', '2': 'pe', '3': 'epistemic' };
        colorMode = modes[e.key];
        document.querySelectorAll('.color-mode-btn').forEach(b => {
          b.classList.toggle('active', b.dataset.mode === colorMode);
        });
        if (colorMode !== 'celltype') { filterCellType = null; clearCellTypeActive(); }
        break;
      }
      case 'p': case 'P': {
        // Screenshot
        if (!e.metaKey && !e.ctrlKey) exportScreenshot();
        break;
      }
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
// Fully inlined — zero allocations per call (no intermediate arrays).

const _geoOut = [0, 0, 0]; // shared output buffer — safe because caller uses result before next call

// Geodesic point at parameter t between p and q in the Poincaré ball.
// γ(t) = p ⊕ (t ⊗ ((-p) ⊕ q))
// Writes into _geoOut and returns it.
function geodesicPoint(p0, p1, p2, q0, q1, q2, t) {
  // Step 1: diff = (-p) ⊕ q  (Möbius addition, inlined)
  const nx0 = -p0, nx1 = -p1, nx2 = -p2;
  const xdy1 = nx0*q0 + nx1*q1 + nx2*q2;
  const xsq1 = nx0*nx0 + nx1*nx1 + nx2*nx2;
  const ysq1 = q0*q0 + q1*q1 + q2*q2;
  const rd1 = 1.0 / Math.max(1 + 2*xdy1 + xsq1*ysq1, 1e-10);
  const da = (1 + 2*xdy1 + ysq1) * rd1;
  const db = (1 - xsq1) * rd1;
  const d0 = da*nx0 + db*q0;
  const d1 = da*nx1 + db*q1;
  const d2 = da*nx2 + db*q2;

  // Step 2: t ⊗ diff  (Möbius scalar multiplication)
  const norm = Math.sqrt(d0*d0 + d1*d1 + d2*d2);
  let s0, s1, s2;
  if (norm < 1e-10) {
    s0 = 0; s1 = 0; s2 = 0;
  } else {
    const coeff = Math.tanh(t * Math.atanh(Math.min(norm, 0.999))) / norm;
    s0 = d0 * coeff; s1 = d1 * coeff; s2 = d2 * coeff;
  }

  // Step 3: result = p ⊕ scaled  (Möbius addition, inlined — writes to _geoOut)
  const xdy2 = p0*s0 + p1*s1 + p2*s2;
  const xsq2 = p0*p0 + p1*p1 + p2*p2;
  const ysq2 = s0*s0 + s1*s1 + s2*s2;
  const rd2 = 1.0 / Math.max(1 + 2*xdy2 + xsq2*ysq2, 1e-10);
  const ra = (1 + 2*xdy2 + ysq2) * rd2;
  const rb = (1 - xsq2) * rd2;
  _geoOut[0] = ra*p0 + rb*s0;
  _geoOut[1] = ra*p1 + rb*s1;
  _geoOut[2] = ra*p2 + rb*s2;
  return _geoOut;
}

function updateSpikes() {
  let alive = 0;

  for (let i = liveSpikes.length - 1; i >= 0; i--) {
    const s = liveSpikes[i];
    s.age++;
    const sLifetime = s.lifetime || SPIKE_VISUAL_FRAMES;
    if (s.age > sLifetime) { liveSpikes.splice(i, 1); continue; }

    // Constant-speed progress along the geodesic
    const t = s.age / sLifetime;

    // Travel the Poincaré-ball geodesic between snapshotted endpoint positions.
    // Convert world→unit-ball (÷ BALL_RADIUS), interpolate, convert back.
    const gp = geodesicPoint(
      s.fx * INV_BALL, s.fy * INV_BALL, s.fz * INV_BALL,
      s.tx * INV_BALL, s.ty * INV_BALL, s.tz * INV_BALL,
      t
    );
    const x = gp[0] * BALL_RADIUS;
    const y = gp[1] * BALL_RADIUS;
    const z = gp[2] * BALL_RADIUS;

    // Tiny orb
    spikeDummy.position.set(x, y, z);
    spikeDummy.scale.setScalar(0.035);
    spikeDummy.updateMatrix();

    if (alive < MAX_SPIKES) {
      spikesMesh.setMatrixAt(alive, spikeDummy.matrix);
      // Fade in (0→1 over first 15%) and fade out (1→0 over last 20%)
      const fadeIn  = t < 0.15 ? t / 0.15 : 1.0;
      const fadeOut = t < 0.80 ? 1.0 : (1.0 - t) * 5.0;
      const fade = fadeIn * fadeOut;
      tempColor.copy(s.color).multiplyScalar(1.8 * fade);
      spikesMesh.setColorAt(alive, tempColor);
      alive++;
    }
  }

  lastSpikeCount = alive;
  spikesMesh.count = alive;
  if (alive > 0) {
    spikesMesh.instanceMatrix.needsUpdate = true;
    spikesMesh.instanceColor.needsUpdate = true;
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
// Per-group firing rate ring buffer: groupRateHistory[groupIdx][bin % HEATMAP_BINS] = rate
let groupRateHistory = [];

function rebuildMorphonOrder() {
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
  groupRateHistory = rasterGroups.map(g => oldByType[g.type] || new Float32Array(HEATMAP_BINS));
  heatmapBinAccum = new Array(rasterGroups.length).fill(0);
  heatmapBinSteps = 0;
}

function rasterStampStep(firedIds) {
  if (morphonOrder.length === 0 || rasterGroups.length === 0 || heatmapPaused) return;

  // Accumulate firings per group within the current bin
  for (const id of firedIds) {
    const row = morphonYMap.get(id);
    if (row === undefined) continue;
    for (let gi = 0; gi < rasterGroups.length; gi++) {
      const g = rasterGroups[gi];
      if (row >= g.startRow && row < g.startRow + g.count) {
        heatmapBinAccum[gi]++;
        break;
      }
    }
  }
  heatmapBinSteps++;

  // When bin is full, flush averaged rate into the ring buffer
  if (heatmapBinSteps >= heatmapBinSize) {
    const col = rasterScrollX % HEATMAP_BINS;
    for (let gi = 0; gi < rasterGroups.length; gi++) {
      groupRateHistory[gi][col] = heatmapBinAccum[gi] / (heatmapBinSize * rasterGroups[gi].count);
      heatmapBinAccum[gi] = 0;
    }
    heatmapBinSteps = 0;
    rasterScrollX++;
  }
}

function drawRasterPlot() {
  if (!rasterCtx || rasterGroups.length === 0) return;
  const canvasW = rasterCanvas.width;
  const canvasH = rasterCanvas.height;
  if (canvasW <= 0 || canvasH <= 0) return;

  const labelW = 40;
  const rightW = 36;
  const plotW = canvasW - labelW - rightW;
  const nGroups = rasterGroups.length;
  const bandGap = 2;
  const bandH = Math.max(8, (canvasH - (nGroups - 1) * bandGap) / nGroups);

  rasterCtx.clearRect(0, 0, canvasW, canvasH);

  const binsUsed = Math.min(rasterScrollX, HEATMAP_BINS);
  const colW = plotW / HEATMAP_BINS;

  for (let gi = 0; gi < nGroups; gi++) {
    const g = rasterGroups[gi];
    const rgb = RASTER_COLORS[g.type];
    const bandY = gi * (bandH + bandGap);
    const rates = groupRateHistory[gi];

    // Background band
    rasterCtx.fillStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},0.04)`;
    rasterCtx.fillRect(labelW, bandY, plotW, bandH);

    // Heatmap columns
    for (let s = 0; s < binsUsed; s++) {
      const bufIdx = ((rasterScrollX - binsUsed + s) % HEATMAP_BINS + HEATMAP_BINS) % HEATMAP_BINS;
      const rate = rates[bufIdx];
      if (rate <= 0) continue;

      const x = labelW + (s / HEATMAP_BINS) * plotW;
      const intensity = Math.min(Math.sqrt(rate * 4), 1.0);
      const alpha = 0.12 + intensity * 0.88;
      rasterCtx.fillStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${alpha.toFixed(2)})`;
      rasterCtx.fillRect(x, bandY, Math.max(colW, 1.2), bandH);
    }

    // Rate line trace
    rasterCtx.strokeStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},0.8)`;
    rasterCtx.lineWidth = 1.5;
    rasterCtx.beginPath();
    let started = false;
    for (let s = 0; s < binsUsed; s++) {
      const bufIdx = ((rasterScrollX - binsUsed + s) % HEATMAP_BINS + HEATMAP_BINS) % HEATMAP_BINS;
      const rate = rates[bufIdx];
      const x = labelW + (s / HEATMAP_BINS) * plotW;
      const y = bandY + bandH - rate * bandH * 0.85;
      if (!started) { rasterCtx.moveTo(x, y); started = true; }
      else rasterCtx.lineTo(x, y);
    }
    rasterCtx.stroke();

    // Group label
    rasterCtx.font = '9px "JetBrains Mono", monospace';
    rasterCtx.textBaseline = 'middle';
    rasterCtx.textAlign = 'right';
    rasterCtx.fillStyle = `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
    rasterCtx.fillText(g.type.substring(0, 3).toUpperCase(), labelW - 4, bandY + bandH / 2);

    // Current rate %
    if (binsUsed > 0) {
      const lastBuf = ((rasterScrollX - 1) % HEATMAP_BINS + HEATMAP_BINS) % HEATMAP_BINS;
      const lastRate = rates[lastBuf];
      rasterCtx.textAlign = 'left';
      rasterCtx.fillStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},0.7)`;
      rasterCtx.fillText(`${(lastRate * 100).toFixed(0)}%`, labelW + plotW + 4, bandY + bandH / 2);
    }
  }

  // Time cursor
  if (binsUsed > 0 && binsUsed < HEATMAP_BINS) {
    const cursorX = labelW + (binsUsed / HEATMAP_BINS) * plotW;
    rasterCtx.strokeStyle = 'rgba(255,255,255,0.2)';
    rasterCtx.lineWidth = 1;
    rasterCtx.beginPath();
    rasterCtx.moveTo(cursorX, 0);
    rasterCtx.lineTo(cursorX, canvasH);
    rasterCtx.stroke();
  }

  // Resolution label
  rasterCtx.font = '8px "JetBrains Mono", monospace';
  rasterCtx.textAlign = 'right';
  rasterCtx.textBaseline = 'bottom';
  rasterCtx.fillStyle = 'rgba(255,255,255,0.2)';
  rasterCtx.fillText(`${heatmapBinSize}x`, canvasW - 4, canvasH - 2);

  // Pause indicator
  if (heatmapPaused) {
    rasterCtx.fillStyle = 'rgba(251,191,36,0.6)';
    rasterCtx.font = '9px "JetBrains Mono", monospace';
    rasterCtx.textAlign = 'left';
    rasterCtx.textBaseline = 'top';
    rasterCtx.fillText('PAUSED', labelW + 4, 3);
  }
}

// ============================================================
// LIVE GRAPH (Chart.js)
// ============================================================
let liveChart = null;
let graphWindowSize = 500; // 0 = show all
const graphData = {
  steps: [],
  fired: [],
  firingRate: [],
  morphons: [],
  synapses: [],
  born: [],
  died: [],
  fieldPe: [],
  justifiedPct: [],  // V3
  skepticism: [],    // V3
};
let lastBorn = 0, lastDied = 0;

function initGraph() {
  const ctx = document.getElementById('graph-canvas');
  if (!ctx || typeof Chart === 'undefined') return;

  const makeDataset = (label, color, yAxisID, hidden = false) => ({
    label, borderColor: color, backgroundColor: color + '18',
    borderWidth: 1.5, pointRadius: 0, tension: 0.3, fill: false,
    yAxisID, hidden, data: [],
  });

  liveChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        makeDataset('Fired',       '#fbbf24', 'y'),
        makeDataset('Firing %',    '#508cff', 'y1'),
        makeDataset('Morphons',    '#34d399', 'y'),
        makeDataset('Synapses',    '#a78bfa', 'y', true),
        makeDataset('Born',        '#22d3ee', 'y'),
        makeDataset('Died',        '#f87171', 'y'),
        makeDataset('Field PE',    '#ef4444', 'y1', true),
        makeDataset('Justified %', '#eab308', 'y1', true),   // V3
        makeDataset('Skepticism',  '#f97316', 'y1', true),   // V3
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: {
          position: 'top',
          labels: {
            color: '#8899bb',
            font: { family: '"JetBrains Mono", monospace', size: 9 },
            boxWidth: 10, boxHeight: 2, padding: 8,
            usePointStyle: false,
          },
          onClick(e, item, legend) {
            // Default toggle behavior
            const idx = item.datasetIndex;
            const meta = legend.chart.getDatasetMeta(idx);
            meta.hidden = meta.hidden === null ? !legend.chart.data.datasets[idx].hidden : null;
            legend.chart.update('none');
          },
        },
        tooltip: {
          animation: false,
          position: 'nearest',
          backgroundColor: 'rgba(8,12,24,0.9)',
          titleFont: { family: '"JetBrains Mono", monospace', size: 9 },
          bodyFont: { family: '"JetBrains Mono", monospace', size: 9 },
          borderColor: 'rgba(80,140,255,0.15)', borderWidth: 1,
          padding: 6,
        },
      },
      scales: {
        x: {
          display: true,
          ticks: { color: '#5a6a88', font: { size: 8 }, maxTicksLimit: 6 },
          grid: { color: 'rgba(255,255,255,0.03)' },
        },
        y: {
          type: 'linear', position: 'left',
          ticks: { color: '#5a6a88', font: { size: 8 }, maxTicksLimit: 5 },
          grid: { color: 'rgba(255,255,255,0.04)' },
          beginAtZero: true,
        },
        y1: {
          type: 'linear', position: 'right',
          ticks: {
            color: '#508cff', font: { size: 8 }, maxTicksLimit: 4,
            callback: v => (v * 100).toFixed(0) + '%',
          },
          grid: { drawOnChartArea: false },
          beginAtZero: true, max: 1,
        },
      },
    },
  });
}

function updateGraph(stats) {
  if (!liveChart) return;
  const step = stats.step_count;
  const born = stats.total_born || 0;
  const died = stats.total_died || 0;

  graphData.steps.push(step);
  graphData.fired.push(frameFired.size);
  graphData.firingRate.push(stats.firing_rate);
  graphData.morphons.push(stats.total_morphons);
  graphData.synapses.push(stats.total_synapses);
  graphData.born.push(born - lastBorn);
  graphData.died.push(died - lastDied);
  graphData.fieldPe.push(stats.field_pe_mean || 0);
  // V3: push governance data
  if (cachedGov) {
    graphData.justifiedPct.push(cachedGov.justified_fraction);
    graphData.skepticism.push(cachedGov.avg_skepticism);
  } else {
    graphData.justifiedPct.push(0);
    graphData.skepticism.push(0);
  }
  lastBorn = born;
  lastDied = died;

  // Apply window
  const win = graphWindowSize > 0 ? graphWindowSize : graphData.steps.length;
  const start = Math.max(0, graphData.steps.length - win);

  liveChart.data.labels = graphData.steps.slice(start);
  liveChart.data.datasets[0].data = graphData.fired.slice(start);
  liveChart.data.datasets[1].data = graphData.firingRate.slice(start);
  liveChart.data.datasets[2].data = graphData.morphons.slice(start);
  liveChart.data.datasets[3].data = graphData.synapses.slice(start);
  liveChart.data.datasets[4].data = graphData.born.slice(start);
  liveChart.data.datasets[5].data = graphData.died.slice(start);
  liveChart.data.datasets[6].data = graphData.fieldPe.slice(start);
  liveChart.data.datasets[7].data = graphData.justifiedPct.slice(start);  // V3
  liveChart.data.datasets[8].data = graphData.skepticism.slice(start);    // V3

  liveChart.update('none');
}

function setGraphScale(type) {
  if (!liveChart) return;
  liveChart.options.scales.y.type = type;
  if (type === 'logarithmic') {
    liveChart.options.scales.y.beginAtZero = false;
    liveChart.options.scales.y.min = 1;
  } else {
    liveChart.options.scales.y.beginAtZero = true;
    delete liveChart.options.scales.y.min;
  }
  liveChart.update('none');
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

    const stepStart = performance.now();
    for (let i = 0; i < stepsThisFrame; i++) {
      // Arena mode: train on examples instead of idle noise
      if (occupation === 'arena' && arenaTrainingSet.length > 0) {
        arenaTrainStep();
      } else if (frameCount % 3 === 0 && performance.now() - lastUserInputTime > 500) {
        // Auto-inject noise unless user sent input recently (500ms grace period)
        makeInput('noise');
      }
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
    const stepMs = performance.now() - stepStart;
    frameLoadEma = frameLoadEma * 0.9 + (stepMs / 16.667) * 0.1;

    // Rebuild raster Y-axis if morphon count changed
    if (nodeData.length !== rasterMorphonCount) rebuildMorphonOrder();

    updateScene();
    drawRasterPlot();
    if (frameCount % 3 === 0) {
      updatePanels(); detectEvents();
      // Arena: update cycle count display
      if (occupation === 'arena') {
        const cyclesEl = document.getElementById('arena-cycles');
        if (cyclesEl) cyclesEl.textContent = arenaCycleCount;
      }
      // Feed performance to endo so stage detection works.
      if (occupation === 'arena' && arenaAccHistory.length > 0) {
        system.report_performance(arenaAccHistory[arenaAccHistory.length - 1] * 100);
      } else if (system.report_performance && cachedStats) {
        const perf = cachedStats.firing_rate * 100; // 0-100 scale
        system.report_performance(perf);
      }
    }
  }

  updateSpikes();
  updateLearningPulses(); // V3: gold learning pulse particles

  // Subtle ball rotation
  if (diskMesh) {
    diskMesh.rotation.y = elapsed * 0.02;
    diskMesh.rotation.x = elapsed * 0.008;
  }

  // Ambient particle drift — gentle Brownian-like motion
  if (ambientParticles) {
    const base = ambientParticles.userData.basePositions;
    const phases = ambientParticles.userData.phases;
    const pos = ambientParticles.geometry.attributes.position.array;
    const activity = Math.min(frameFired.size / 30, 1.0);
    // Particles drift more when the system is active
    const driftAmp = 0.15 + activity * 0.3;
    const speed = 0.3 + activity * 0.5;
    for (let i = 0; i < phases.length; i++) {
      const p = phases[i];
      const i3 = i * 3;
      pos[i3] = base[i3] + Math.sin(elapsed * speed + p) * driftAmp;
      pos[i3 + 1] = base[i3 + 1] + Math.cos(elapsed * speed * 0.7 + p * 1.3) * driftAmp;
      pos[i3 + 2] = base[i3 + 2] + Math.sin(elapsed * speed * 0.5 + p * 0.7) * driftAmp * 0.6;
    }
    ambientParticles.geometry.attributes.position.needsUpdate = true;
    // Brightness pulses with activity
    ambientParticles.material.opacity = 0.18 + activity * 0.15;
  }

  // Dynamic bloom — pulses with network activity
  if (bloomPass) {
    const activity = Math.min(lastSpikeCount / 80, 1.0);
    bloomPass.strength = 0.34 + activity * 0.22;
  }

  // Panel activity pulse — light up panels when system is very active
  if (frameCount % 10 === 0 && frameFired.size > 15) {
    document.querySelectorAll('#left-panel .panel, #right-panel .panel').forEach(p => {
      p.classList.add('active-pulse');
      setTimeout(() => p.classList.remove('active-pulse'), 800);
    });
  }

  // Update camera BEFORE raycast so matrices match the rendered frame
  controls.update();
  camera.updateMatrixWorld();
  updateRaycast();
  composer.render();
}

// ============================================================
// V3: SCREENSHOT EXPORT
// ============================================================
function exportScreenshot() {
  if (!renderer) return;
  // Render one frame with preserveDrawingBuffer
  composer.render();
  const dataUrl = renderer.domElement.toDataURL('image/png');
  const a = document.createElement('a');
  a.href = dataUrl;
  a.download = `morphon_${Date.now()}.png`;
  a.click();
  addEvent('', 'Screenshot exported', 'event-diff');
}

// ============================================================
// V3: SYNAPSE PROVENANCE COLORS
// ============================================================
const PROVENANCE_COLORS = {
  External:  { r: 0.89, g: 0.91, b: 0.94 }, // white-ish
  Proximity: { r: 0.13, g: 0.83, b: 0.93 }, // cyan
  Division:  { r: 0.75, g: 0.52, b: 0.99 }, // purple
  Fusion:    { r: 0.96, g: 0.45, b: 0.71 }, // pink
  Hebbian:   { r: 0.98, g: 0.75, b: 0.14 }, // yellow
  none:      { r: 0.4,  g: 0.4,  b: 0.5  }, // gray
};

function getProvenanceColor(formationCause) {
  return PROVENANCE_COLORS[formationCause] || PROVENANCE_COLORS.none;
}

// ============================================================
// V3: CLUSTER HULLS (2D convex hull rendered as translucent shapes)
// ============================================================
const clusterHullMeshes = new Map(); // clusterId → { line, fill }
const clusterHullCache = new Map();  // clusterId → memberCount (for invalidation)
const _clusterNodes = new Map();     // reused each frame — avoids new Map() in updateClusterHulls

function convexHull2D(points) {
  // Graham scan on XZ plane
  if (points.length < 3) return points.slice();
  points.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cross = (O, A, B) => (A[0]-O[0])*(B[1]-O[1]) - (A[1]-O[1])*(B[0]-O[0]);
  const lower = [];
  for (const p of points) {
    while (lower.length >= 2 && cross(lower[lower.length-2], lower[lower.length-1], p) <= 0) lower.pop();
    lower.push(p);
  }
  const upper = [];
  for (let i = points.length - 1; i >= 0; i--) {
    const p = points[i];
    while (upper.length >= 2 && cross(upper[upper.length-2], upper[upper.length-1], p) <= 0) upper.pop();
    upper.push(p);
  }
  upper.pop(); lower.pop();
  return lower.concat(upper);
}

function updateClusterHulls(nodes, edges) {
  if (!scene) return;
  // Group nodes by cluster_id — reuse Map to avoid allocation each frame
  _clusterNodes.clear();
  for (const n of nodes) {
    if (n.cluster_id != null) {
      if (!_clusterNodes.has(n.cluster_id)) _clusterNodes.set(n.cluster_id, []);
      _clusterNodes.get(n.cluster_id).push(n);
    }
  }

  // Remove stale hulls
  for (const [cid, meshes] of clusterHullMeshes) {
    if (!_clusterNodes.has(cid)) {
      scene.remove(meshes.line);
      meshes.line.geometry.dispose();
      if (meshes.fill) { scene.remove(meshes.fill); meshes.fill.geometry.dispose(); }
      clusterHullMeshes.delete(cid);
      clusterHullCache.delete(cid);
    }
  }

  for (const [cid, members] of _clusterNodes) {
    if (members.length < 3) continue;

    // Check cache — skip if unchanged
    if (clusterHullCache.get(cid) === members.length && clusterHullMeshes.has(cid)) continue;
    clusterHullCache.set(cid, members.length);

    // Remove old meshes
    if (clusterHullMeshes.has(cid)) {
      const old = clusterHullMeshes.get(cid);
      scene.remove(old.line);
      old.line.geometry.dispose();
      if (old.fill) { scene.remove(old.fill); old.fill.geometry.dispose(); }
    }

    // Get epistemic state color
    const epState = members[0].epistemic_state || 'none';
    const epColor = EPISTEMIC_COLORS[epState] || EPISTEMIC_COLORS.none;

    // Project to 3D positions (use same BALL_RADIUS scale as nodes)
    const pts2d = members.map(n => [n.x * BALL_RADIUS, n.z * BALL_RADIUS]);
    const hull = convexHull2D(pts2d);
    if (hull.length < 3) continue;

    // Compute average Y for the hull plane
    const avgY = members.reduce((s, n) => s + n.y * BALL_RADIUS, 0) / members.length;

    // Create line loop
    const linePoints = hull.map(p => new THREE.Vector3(p[0], avgY, p[1]));
    linePoints.push(linePoints[0].clone()); // close loop
    const lineGeo = new THREE.BufferGeometry().setFromPoints(linePoints);
    const lineMat = new THREE.LineBasicMaterial({
      color: epColor, transparent: true, opacity: 0.15, depthWrite: false, depthTest: false,
      blending: THREE.AdditiveBlending,
    });
    const line = new THREE.Line(lineGeo, lineMat);

    scene.add(line);
    clusterHullMeshes.set(cid, { line, fill: null });
  }
}

// ============================================================
// V3: EDGE TOOLTIP (on hover)
// ============================================================
let hoveredEdgeIdx = null;
const prevEdgeReinforcementCounts = new Map(); // edge key → count (for pulse detection)

function findClosestEdge(mouseX, mouseY) {
  if (!edgeData || edgeData.length === 0) return null;
  // Project mouse to NDC
  const ndcX = (mouseX / window.innerWidth) * 2 - 1;
  const ndcY = -(mouseY / window.innerHeight) * 2 + 1;

  let bestDist = 12; // pixel threshold (squared)
  let bestIdx = null;

  for (let i = 0; i < edgeData.length; i++) {
    const e = edgeData[i];
    const fi = nodeMap.get(e.from);
    const ti = nodeMap.get(e.to);
    if (fi === undefined || ti === undefined) continue;

    _edgeVecA.set(nodePositions[fi*3], nodePositions[fi*3+1], nodePositions[fi*3+2]).project(camera);
    _edgeVecB.set(nodePositions[ti*3], nodePositions[ti*3+1], nodePositions[ti*3+2]).project(camera);

    // Point-to-segment distance in NDC
    const dx = _edgeVecB.x - _edgeVecA.x, dy = _edgeVecB.y - _edgeVecA.y;
    const lenSq = dx*dx + dy*dy;
    if (lenSq < 0.0001) continue;
    const t = Math.max(0, Math.min(1, ((ndcX-_edgeVecA.x)*dx + (ndcY-_edgeVecA.y)*dy) / lenSq));
    const px = _edgeVecA.x + t*dx, py = _edgeVecA.y + t*dy;
    const distSq = (ndcX-px)*(ndcX-px) + (ndcY-py)*(ndcY-py);

    // Convert to approximate pixels (NDC range is 2, screen is ~1000px)
    const pixDist = Math.sqrt(distSq) * Math.min(window.innerWidth, window.innerHeight) * 0.5;
    if (pixDist < bestDist) {
      bestDist = pixDist;
      bestIdx = i;
    }
  }
  return bestIdx;
}

function showEdgeTooltip(edgeIdx) {
  const e = edgeData[edgeIdx];
  if (!e) { dom.tooltip.style.display = 'none'; return; }

  dom.tooltip.style.display = 'block';
  dom.tooltip.style.left = (mouseClientX + 16) + 'px';
  dom.tooltip.style.top = (mouseClientY - 10) + 'px';

  const wPct = Math.min(Math.abs(e.weight) / 2, 1) * 100;
  const wColor = e.weight >= 0 ? 'var(--accent)' : '#ff4466';
  const cause = e.formation_cause || 'unknown';
  const causeColor = PROVENANCE_COLORS[cause] || PROVENANCE_COLORS.none;
  const causeHex = `rgb(${Math.round(causeColor.r*255)},${Math.round(causeColor.g*255)},${Math.round(causeColor.b*255)})`;

  dom.tooltip.innerHTML = `
    <div class="tip-header">
      <span class="tip-id">${e.from} &rarr; ${e.to}</span>
      <span class="tip-type" style="font-size:9px">${e.consolidated ? 'CONSOLIDATED' : 'plastic'}</span>
    </div>
    <hr class="tip-sep">
    <div class="tip-row">
      <span class="label">Weight</span>
      <span class="tip-bar"><span class="fill" style="width:${wPct}%;background:${wColor}"></span></span>
      <span class="value">${e.weight.toFixed(3)}</span>
    </div>
    <div class="tip-row">
      <span class="label">Cause</span>
      <span class="value" style="color:${causeHex}">${cause}</span>
    </div>
    <div class="tip-row">
      <span class="label">Reinforced</span>
      <span class="value">${e.reinforcement_count || 0}x</span>
    </div>
    <div class="tip-row">
      <span class="label">Eligibility</span>
      <span class="value">${(e.eligibility || 0).toFixed(3)}</span>
    </div>
  `;
}

// ============================================================
// V3: REINFORCEMENT PULSES
// ============================================================
const learningPulses = [];
const PULSE_FRAMES = 40;

function detectReinforcementPulses() {
  if (!edgeData) return;
  for (let i = 0; i < edgeData.length; i++) {
    const e = edgeData[i];
    // Numeric key avoids template string allocation per edge per frame.
    // Safe as long as IDs stay below 65536 (MAX_NODES = 2000).
    const key = (e.from << 17) | e.to;
    const prev = prevEdgeReinforcementCounts.get(key) || 0;
    const curr = e.reinforcement_count || 0;
    if (curr > prev && prev > 0) {
      const fi = nodeMap.get(e.from);
      const ti = nodeMap.get(e.to);
      if (fi !== undefined && ti !== undefined) {
        learningPulses.push({ fromIdx: fi, toIdx: ti, age: 0 });
      }
    }
    prevEdgeReinforcementCounts.set(key, curr);
  }
}

function updateLearningPulses() {
  // Render learning pulses using the spikesMesh (shared with spike particles)
  // They get added after the regular spikes
  for (let i = learningPulses.length - 1; i >= 0; i--) {
    const p = learningPulses[i];
    p.age++;
    if (p.age > PULSE_FRAMES) { learningPulses.splice(i, 1); continue; }

    const t = p.age / PULSE_FRAMES;
    const fi = p.fromIdx * 3, ti = p.toIdx * 3;
    const x = nodePositions[fi]   + (nodePositions[ti]   - nodePositions[fi])   * t;
    const y = nodePositions[fi+1] + (nodePositions[ti+1] - nodePositions[fi+1]) * t;
    const z = nodePositions[fi+2] + (nodePositions[ti+2] - nodePositions[fi+2]) * t;

    const fade = t < 0.7 ? 1.0 : (1.0 - t) * 3.3;
    const count = spikesMesh.count;
    if (count < MAX_SPIKES) {
      _pulseDummy.position.set(x, y, z);
      _pulseDummy.scale.setScalar(0.055); // larger than spikes
      _pulseDummy.updateMatrix();
      spikesMesh.setMatrixAt(count, _pulseDummy.matrix);
      _pulseColor.setRGB(0.98, 0.75, 0.14);
      _pulseColor.multiplyScalar(2.0 * fade);
      spikesMesh.setColorAt(count, _pulseColor);
      spikesMesh.count = count + 1;
    }
  }
}

// ============================================================
// LEARNING PIPELINE PANEL
// ============================================================
function updateLearningPanel() {
  if (!cachedLearning || !dom.lpSynBar) return;
  const lp = cachedLearning;
  const total = Math.max(lp.total_synapses, 1);

  // Pipeline bars — each as fraction of total synapses
  dom.lpSynBar.style.width = '100%';
  dom.lpSynV.textContent = lp.total_synapses;

  const eligPct = (lp.eligible / total * 100);
  dom.lpEligBar.style.width = eligPct + '%';
  dom.lpEligV.textContent = lp.eligible;

  const tagPct = (lp.active_tags / total * 100);
  dom.lpTagBar.style.width = tagPct + '%';
  dom.lpTagV.textContent = lp.active_tags;

  dom.lpCapBar.style.width = Math.min(lp.total_captures / total * 100, 100) + '%';
  dom.lpCapV.textContent = lp.total_captures;

  const conPct = (lp.consolidated / total * 100);
  dom.lpConBar.style.width = conPct + '%';
  dom.lpConV.textContent = lp.consolidated;

  const jusPct = (lp.justified_fraction * 100);
  dom.lpJusBar.style.width = jusPct + '%';
  dom.lpJusV.textContent = jusPct.toFixed(0) + '%';

  // Weight histogram
  drawWeightHistogram(lp);
}

function drawWeightHistogram(lp) {
  const canvas = dom.weightHistogram;
  if (!canvas) return;

  const rect = canvas.parentElement.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const w = Math.floor(rect.width * dpr);
  const h = Math.floor((rect.height - 20) * dpr); // leave room for labels
  if (w < 10 || h < 10) return;

  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
    canvas.style.width = rect.width + 'px';
    canvas.style.height = (rect.height - 20) + 'px';
  }

  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, w, h);

  const bins = lp.weight_histogram;
  if (!bins || bins.length === 0) return;

  const maxBin = Math.max(...bins, 1);
  const numBins = bins.length;
  const barW = w / numBins;
  const midBin = Math.floor(numBins / 2);

  for (let i = 0; i < numBins; i++) {
    const barH = (bins[i] / maxBin) * (h - 4);
    const x = i * barW;
    const y = h - barH;

    // Color: blue for negative weights, warm for positive, brighter near edges
    const t = i / (numBins - 1); // 0=most negative, 1=most positive
    if (t < 0.5) {
      const intensity = (0.5 - t) * 2; // 1 at far left, 0 at center
      ctx.fillStyle = `rgba(80, 140, 255, ${0.25 + intensity * 0.6})`;
    } else {
      const intensity = (t - 0.5) * 2; // 0 at center, 1 at far right
      ctx.fillStyle = `rgba(251, 191, 36, ${0.25 + intensity * 0.6})`;
    }

    ctx.fillRect(x + 0.5, y, barW - 1, barH);
  }

  // Zero line
  const zeroX = midBin * barW + barW / 2;
  ctx.strokeStyle = 'rgba(255,255,255,0.15)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(zeroX, 0);
  ctx.lineTo(zeroX, h);
  ctx.stroke();

  // Update axis labels
  const range = lp.weight_range || 1;
  dom.whMin.textContent = (-range).toFixed(1);
  dom.whMax.textContent = '+' + range.toFixed(1);
  dom.whStats.textContent = `μ=${lp.weight_mean.toFixed(3)} σ=${lp.weight_std.toFixed(3)}`;
}

// ============================================================
// V3: CLUSTER LIST PANEL
// ============================================================
function updateClusterList() {
  if (!dom.clusterList || !cachedGov) return;

  if (cachedGov.total_clusters === 0) {
    dom.clusterList.innerHTML = '<div style="color:var(--text-dim);font-size:9px;padding:4px">No clusters</div>';
    return;
  }

  // Build cluster info from node data
  const clusters = new Map();
  for (const n of nodeData) {
    if (n.cluster_id != null) {
      if (!clusters.has(n.cluster_id)) {
        clusters.set(n.cluster_id, {
          id: n.cluster_id,
          state: n.epistemic_state,
          skepticism: n.skepticism,
          members: [],
          cx: 0, cy: 0, cz: 0,
        });
      }
      const c = clusters.get(n.cluster_id);
      c.members.push(n.id);
      c.cx += n.x; c.cy += n.y; c.cz += n.z;
    }
  }

  let html = '';
  for (const [cid, c] of clusters) {
    c.cx /= c.members.length; c.cy /= c.members.length; c.cz /= c.members.length;
    const epColor = EPISTEMIC_COLORS[c.state]?.getStyle() || '#4b5563';
    const stateLabel = (c.state || 'none').charAt(0);
    html += `<div class="cluster-row" data-cid="${cid}" style="cursor:pointer;padding:3px 4px;border-radius:3px;display:flex;align-items:center;gap:6px;font-size:10px" title="Click to zoom">
      <span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${epColor};flex-shrink:0"></span>
      <span style="color:var(--text)">#${cid}</span>
      <span style="color:${epColor};font-weight:600">${stateLabel}</span>
      <span style="color:var(--text-dim)">${c.members.length}m</span>
      <span style="color:var(--text-dim);margin-left:auto">${c.skepticism.toFixed(2)}</span>
    </div>`;
  }
  dom.clusterList.innerHTML = html;

  // Click handler — zoom to cluster
  dom.clusterList.querySelectorAll('.cluster-row').forEach(row => {
    row.addEventListener('click', () => {
      const cid = parseInt(row.dataset.cid);
      const c = clusters.get(cid);
      if (!c) return;

      // Select all members
      selectedNodeId = c.members[0];
      connectedToSelected.clear();
      for (const mid of c.members) connectedToSelected.add(mid);

      // Zoom camera to cluster centroid
      const target = new THREE.Vector3(c.cx * BALL_RADIUS, c.cy * BALL_RADIUS, c.cz * BALL_RADIUS);
      controls.target.copy(target);
      camera.position.copy(target).add(new THREE.Vector3(0, 5, 10));
      controls.update();

      updateDetailPanel();
    });
  });
}

// ============================================================
// V3: TIMELINE SCRUBBER
// ============================================================
const timelineSnapshots = [];
const TIMELINE_INTERVAL = 100;  // steps between snapshots
const TIMELINE_MAX = 50;        // max snapshots stored
let timelineLastStep = 0;
let timelineScrubbing = false;

function recordTimelineSnapshot() {
  if (!system) return;
  const step = Number(system.step_count());
  if (step - timelineLastStep >= TIMELINE_INTERVAL) {
    try {
      const json = system.save_json();
      timelineSnapshots.push({ step, json });
      if (timelineSnapshots.length > TIMELINE_MAX) timelineSnapshots.shift();
      timelineLastStep = step;

      // Update scrubber range
      const scrubber = document.getElementById('timeline-scrubber');
      if (scrubber && timelineSnapshots.length > 1) {
        scrubber.min = 0;
        scrubber.max = timelineSnapshots.length - 1;
        scrubber.value = timelineSnapshots.length - 1;
        document.getElementById('timeline-step').textContent = step;
      }
    } catch (_) {}
  }
}

function scrubTimeline(index) {
  if (index < 0 || index >= timelineSnapshots.length) return;
  const snap = timelineSnapshots[index];
  try {
    if (system) { try { system.free(); } catch(_) {} }
    system = WasmSystem.loadJson(snap.json);
    timelineScrubbing = true;
    document.getElementById('timeline-step').textContent = snap.step;
    updateScene();
    updatePanels();
  } catch(e) {
    addEvent('', 'Timeline scrub failed: ' + e.message, 'event-death');
  }
}

// ============================================================
// INIT
// ============================================================
async function main() {
  initDOMCache();
  initScene();
  setupControls();
  initArena();
  initRaster();
  initGraph();
  await init();

  system = new WasmSystem(60, 'cortical', 3);
  for (let i = 0; i < 30; i++) { makeInput('noise'); system.step(); }

  updateScene();
  rebuildMorphonOrder();
  updatePanels();

  // Dramatic reveal: start camera further out, zoom in as loading fades
  camera.position.set(0, 12, 38);
  const loading = document.getElementById('loading');
  loading.classList.add('hidden');
  setTimeout(() => loading.remove(), 800);
  // Smooth camera zoom-in over ~2 seconds
  const startPos = { y: 12, z: 38 };
  const endPos = { y: 8, z: 22 };
  const zoomStart = performance.now();
  const zoomDuration = 2000;
  function zoomIn() {
    const t = Math.min((performance.now() - zoomStart) / zoomDuration, 1);
    const ease = 1 - Math.pow(1 - t, 3); // cubic ease-out
    camera.position.y = startPos.y + (endPos.y - startPos.y) * ease;
    camera.position.z = startPos.z + (endPos.z - startPos.z) * ease;
    if (t < 1) requestAnimationFrame(zoomIn);
  }
  zoomIn();

  addEvent(0, 'System initialized [cortical, 60 seed, 3D]', 'event-diff');

  // V3: Screenshot button
  document.getElementById('btn-screenshot')?.addEventListener('click', exportScreenshot);

  // V3: Timeline scrubber
  const scrubber = document.getElementById('timeline-scrubber');
  if (scrubber) {
    scrubber.addEventListener('input', (e) => {
      scrubTimeline(parseInt(e.target.value));
    });
  }

  // V3: Lineage tab — redraw on tab switch

  renderer.setAnimationLoop(animate);
}

main().catch(e => {
  console.error(e);
  document.getElementById('loading').querySelector('h2').textContent = 'ERROR: ' + e.message;
});

// ============================================================
// OBSERVER — module-level so it works regardless of main() state
// ============================================================
let observerConnected = false;
let observerAutoTimer = null;

function obsTimestamp() {
  const d = new Date();
  return `${String(d.getHours()).padStart(2,'0')}:${String(d.getMinutes()).padStart(2,'0')}:${String(d.getSeconds()).padStart(2,'0')}`;
}

function obsAppend(text, role = 'assistant') {
  const log = document.getElementById('obs-log');
  if (!log) return null;
  const entry = document.createElement('div');
  entry.className = 'obs-entry';
  const ts = document.createElement('span');
  ts.className = 'obs-ts';
  ts.textContent = obsTimestamp();
  const textEl = document.createElement('span');
  textEl.className = 'obs-text' + (role === 'user' ? ' user' : role === 'sys' ? ' sys' : '');
  textEl.textContent = text;
  entry.appendChild(ts);
  entry.appendChild(textEl);
  log.appendChild(entry);
  log.scrollTop = log.scrollHeight;
  return textEl;
}

function obsSetStatus(cls, label) {
  const dot = document.getElementById('obs-status-dot');
  const lbl = document.getElementById('obs-status-label');
  if (dot) dot.className = 'observer-status-dot' + (cls ? ' ' + cls : '');
  if (lbl) lbl.textContent = label;
}

function obsResetModelSelect(names) {
  const sel = document.getElementById('obs-model');
  if (!sel) return;
  sel.innerHTML = '';
  if (!names || names.length === 0) {
    const opt = document.createElement('option');
    opt.value = '';
    opt.textContent = '— connect first —';
    sel.appendChild(opt);
    sel.disabled = true;
  } else {
    for (const name of names) {
      const opt = document.createElement('option');
      opt.value = name;
      opt.textContent = name;
      sel.appendChild(opt);
    }
    sel.disabled = false;
  }
}

function obsRestartAutoTimer() {
  if (observerAutoTimer) { clearInterval(observerAutoTimer); observerAutoTimer = null; }
  const secs = parseInt(document.getElementById('obs-auto-interval')?.value || '0');
  if (secs > 0 && observerConnected) {
    observerAutoTimer = setInterval(() => {
      if (observerConnected && document.querySelector('#bottom-panel .tab-btn.active')?.dataset.tab === 'tab-observer') {
        observerNarrate();
      }
    }, secs * 1000);
  }
}

function buildObserverSnapshot() {
  if (!cachedStats) return 'System not yet initialized.';
  const s = cachedStats;
  const m = cachedMod || {};
  const ct = s.differentiation_map || {};
  return [
    `Step ${s.step_count} | Morphons: ${s.total_morphons} | Synapses: ${s.total_synapses} | Clusters: ${s.fused_clusters}`,
    `Firing rate: ${(s.firing_rate * 100).toFixed(1)}% | Active: ${frameFired.size} | Avg energy: ${s.avg_energy.toFixed(2)}`,
    `Pred error: ${s.avg_prediction_error.toFixed(3)} | Field PE max: ${(s.field_pe_max || 0).toFixed(3)} | Working mem: ${s.working_memory_items}`,
    `Neuromod — reward: ${(m.reward || 0).toFixed(2)}, novelty: ${(m.novelty || 0).toFixed(2)}, arousal: ${(m.arousal || 0).toFixed(2)}, homeostasis: ${(m.homeostasis || 0).toFixed(2)}`,
    `Cells — Sensory: ${ct.Sensory || 0}, Assoc: ${ct.Associative || 0}, Motor: ${ct.Motor || 0}, Mod: ${ct.Modulatory || 0}, Stem: ${ct.Stem || 0}, Fused: ${ct.Fused || 0}`,
    `Born: ${s.total_born || 0} | Died: ${s.total_died || 0} | Gen max: ${s.max_generation}`,
  ].join('\n');
}

async function observerNarrate(userQuestion = null) {
  const url = (document.getElementById('obs-url')?.value || '').trim();
  const model = (document.getElementById('obs-model')?.value || '').trim() || 'llama3.2:1b';
  if (!url || !observerConnected || !model) return;

  const narrateBtn = document.getElementById('obs-narrate-btn');
  const askBtn = document.getElementById('obs-ask-btn');
  if (narrateBtn) narrateBtn.disabled = true;
  if (askBtn) askBtn.disabled = true;

  const snapshot = buildObserverSnapshot();
  const systemPrompt = `You are observing a live Morphogenic Intelligence neural engine — a biologically-inspired self-organizing system running in the browser. Respond in 1–2 concise sentences. Reference specific numbers. No markdown.`;
  const userMsg = userQuestion
    ? `State snapshot:\n${snapshot}\n\nQuestion: ${userQuestion}`
    : `Narrate what is happening right now:\n${snapshot}`;

  if (userQuestion) obsAppend(userQuestion, 'user');
  const textEl = obsAppend('…', 'assistant');

  try {
    const resp = await fetch(`${url}/api/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model, prompt: userMsg, system: systemPrompt, stream: false }),
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    if (textEl) textEl.textContent = (data.response || '').trim();
    document.getElementById('obs-log').scrollTop = 999999;
  } catch (err) {
    if (textEl) textEl.textContent = `Error: ${err.message}`;
    obsSetStatus('error', 'ERROR');
    observerConnected = false;
  } finally {
    if (narrateBtn) narrateBtn.disabled = false;
    if (askBtn) askBtn.disabled = false;
  }
}

async function observerConnect(silent = false) {
  const url = (document.getElementById('obs-url')?.value || '').trim();
  const connectBtn = document.getElementById('obs-connect-btn');

  // Disconnect path
  if (observerConnected) {
    observerConnected = false;
    if (observerAutoTimer) { clearInterval(observerAutoTimer); observerAutoTimer = null; }
    obsSetStatus('', 'DISCONNECTED');
    if (connectBtn) connectBtn.textContent = 'CONNECT';
    document.getElementById('obs-narrate-btn').disabled = true;
    document.getElementById('obs-question').disabled = true;
    document.getElementById('obs-ask-btn').disabled = true;
    obsResetModelSelect(null);
    return;
  }

  if (!url) return;

  obsSetStatus('connecting', 'CONNECTING');
  if (connectBtn) { connectBtn.textContent = '…'; connectBtn.disabled = true; }

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 6000);
  try {
    const resp = await fetch(`${url}/api/tags`, { signal: controller.signal });
    clearTimeout(timeoutId);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    // Sort by size ascending so the lightest model is first / pre-selected
    const sorted = (data.models || []).slice().sort((a, b) => (a.size || 0) - (b.size || 0));
    const names = sorted.map(m => m.name);

    observerConnected = true;
    obsSetStatus('connected', 'CONNECTED');
    if (connectBtn) { connectBtn.textContent = 'DISCONNECT'; connectBtn.disabled = false; }
    document.getElementById('obs-narrate-btn').disabled = false;
    document.getElementById('obs-question').disabled = false;
    document.getElementById('obs-ask-btn').disabled = false;
    obsResetModelSelect(names);
    obsAppend(
      names.length > 0
        ? `Connected · ${names.length} model${names.length !== 1 ? 's' : ''}: ${names.join(', ')}`
        : `Connected · no models pulled yet — run: ollama pull llama3.2`,
      'sys'
    );
  } catch (err) {
    clearTimeout(timeoutId);
    if (connectBtn) { connectBtn.textContent = 'CONNECT'; connectBtn.disabled = false; }
    if (silent) {
      obsSetStatus('', 'DISCONNECTED');
    } else {
      const hint = err.name === 'AbortError' ? 'Timed out' : err.message;
      obsSetStatus('error', 'FAILED');
      obsAppend(`${hint} — is Ollama running on ${url}? For CORS: OLLAMA_ORIGINS="*" ollama serve`, 'sys');
    }
  }
}

// Copy / export
window._obsCopy = () => {
  const entries = document.querySelectorAll('#obs-log .obs-entry');
  const lines = [...entries].map(e => {
    const ts = e.querySelector('.obs-ts')?.textContent || '';
    const txt = e.querySelector('.obs-text')?.textContent || '';
    return `[${ts}] ${txt}`;
  });
  navigator.clipboard.writeText(lines.join('\n')).then(
    () => obsAppend('Copied to clipboard.', 'sys'),
    () => obsAppend('Clipboard write failed.', 'sys')
  );
};

window._obsExport = () => {
  const entries = document.querySelectorAll('#obs-log .obs-entry');
  const lines = ['# Morphon Observer Log', `_Exported ${new Date().toISOString()}_`, ''];
  for (const e of entries) {
    const ts = e.querySelector('.obs-ts')?.textContent || '';
    const txt = e.querySelector('.obs-text')?.textContent || '';
    const cls = e.querySelector('.obs-text')?.className || '';
    if (cls.includes('user')) lines.push(`**[${ts}] You:** ${txt}`);
    else if (cls.includes('sys')) lines.push(`*[${ts}] ${txt}*`);
    else lines.push(`**[${ts}]** ${txt}`);
  }
  const blob = new Blob([lines.join('\n')], { type: 'text/markdown' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `morphon-observer-${Date.now()}.md`;
  a.click();
  URL.revokeObjectURL(a.href);
};

// Expose to window — onclick attributes in HTML call these directly
window._obsConnect = () => observerConnect(false);
window._obsNarrate = (q) => observerNarrate(q || null);

// Bottom panel resize drag
(function () {
  const handle = document.getElementById('panel-resize-handle');
  const panel  = document.getElementById('bottom-panel');
  if (!handle || !panel) return;
  const MIN_H = 80, MAX_H = window.innerHeight * 0.85;
  let dragging = false, startY = 0, startH = 0;

  handle.addEventListener('mousedown', e => {
    e.preventDefault();
    dragging = true;
    startY = e.clientY;
    startH = panel.getBoundingClientRect().height;
    panel.classList.add('no-transition');
    panel.classList.remove('maximized');
    document.body.style.userSelect = 'none';
  });

  document.addEventListener('mousemove', e => {
    if (!dragging) return;
    const delta = startY - e.clientY;          // drag up = bigger
    const h = Math.min(MAX_H, Math.max(MIN_H, startH + delta));
    document.documentElement.style.setProperty('--panel-h', h + 'px');
  });

  document.addEventListener('mouseup', () => {
    if (!dragging) return;
    dragging = false;
    panel.classList.remove('no-transition');
    document.body.style.userSelect = '';
  });
})();

// AUTO interval select
document.addEventListener('DOMContentLoaded', () => {
  const sel = document.getElementById('obs-auto-interval');
  if (sel) sel.addEventListener('change', obsRestartAutoTimer);
});
