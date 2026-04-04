# MORPHON Python SDK — Developer Interface Specification
## morphon-core v0.1.0 — PyPI Package
### Technical Specification v1.0 — TasteHub GmbH, April 2026

---

| | |
|---|---|
| **Package name** | `morphon-core` |
| **Rust crate** | `morphon-python` (PyO3 wrapper around `morphon-core`) |
| **Build tool** | maturin (≥1.0) |
| **Python** | ≥ 3.10 |
| **Rust** | ≥ 1.83 (PyO3 0.28 requirement) |
| **License** | Apache 2.0 (same as morphon-core) |
| **Depends on** | Phase 3 completion (MNIST >50% with local inhibition) |
| **Effort** | 6–8 weeks |

---

## 1. Design Philosophy

### 1.1 One Import, One System, One Loop

The SDK must feel like a native Python library, not a wrapper around a foreign runtime. A researcher or ML engineer should go from `pip install morphon-core` to a running experiment in under 5 minutes with no Rust knowledge whatsoever.

```python
import morphon

system = morphon.System(seed_size=200, task="rl")
system.run(env="CartPole-v1", episodes=1000)
print(system.stats)
```

That's the target. Three lines to a working RL experiment.

### 1.2 Rust Does the Work, Python Steers

The critical performance boundary: **all tick-level computation stays in Rust.** Python never touches individual morphon voltages, spike delivery, or STDP updates. Python sets up the system, feeds observations, reads rewards, and inspects diagnostics. The fast path (Pulse Kernel) runs entirely in Rust, releasing the GIL during computation so Python threads aren't blocked.

This follows the PyO3 best practice: minimize Python↔Rust boundary crossings. One `system.step(observation)` call does thousands of Rust operations and returns a summary — not thousands of individual calls.

### 1.3 Progressive Disclosure

Three levels of API complexity:

**Level 1 — Quick Start:** Pre-configured tasks with sensible defaults. CartPole, MNIST, custom Gym environments. The user provides an environment and gets results.

**Level 2 — Configuration:** Custom system parameters, developmental programs, reward shaping, diagnostic callbacks. The user understands what morphons are and wants to tune the system.

**Level 3 — Research:** Direct access to morphon states, synapse weights, genome inspection, Endoquilibrium internals, custom learning rules. The user is building on MORPHON for their own research.

---

## 2. Package Structure

```
morphon-core/
├── Cargo.toml              # Rust workspace + PyO3 dependency
├── pyproject.toml           # maturin build config
├── src/
│   └── lib.rs               # PyO3 module definition + all #[pyclass] wrappers
├── python/
│   └── morphon/
│       ├── __init__.py       # Re-exports, version, convenience functions
│       ├── system.py         # Pure-Python System wrapper (type hints, docstrings)
│       ├── envs.py           # Environment adapters (Gym, custom)
│       ├── tasks.py          # Pre-configured task profiles (CartPole, MNIST, etc.)
│       ├── diagnostics.py    # Diagnostic data classes + plotting helpers
│       ├── visualization.py  # Network visualization (matplotlib, plotly)
│       ├── callbacks.py      # Callback protocol for training hooks
│       └── py.typed          # PEP 561 marker for type checkers
├── tests/
│   ├── test_system.py
│   ├── test_cartpole.py
│   ├── test_mnist.py
│   └── test_diagnostics.py
├── notebooks/
│   ├── 01_cartpole_quickstart.ipynb
│   ├── 02_mnist_classification.ipynb
│   └── 03_custom_environment.ipynb
├── README.md
└── docs/
    └── ...                   # mkdocs site
```

### 2.1 Why Hybrid (Rust + Pure Python)?

The Rust layer (`src/lib.rs`) exposes PyO3 `#[pyclass]` objects with minimal Python-facing API. The pure Python layer (`python/morphon/`) adds type hints, docstrings, convenience methods, plotting, and Gym integration. This separation means:

- Rust compilation is fast (only core types exposed)
- Python tooling (mypy, pylint, IDEs) works perfectly on the Python layer
- Users can read the Python source to understand the API without reading Rust
- Plotting/visualization dependencies (matplotlib, plotly) stay in Python, not in the Rust build

---

## 3. Core API

### 3.1 morphon.System — The Central Object

```python
class System:
    """A MORPHON adaptive intelligence system.
    
    The System contains morphons (biological compute units), synapses
    (weighted connections), and regulatory subsystems (Endoquilibrium).
    It processes observations, learns from rewards, and produces actions.
    
    Args:
        seed_size: Initial number of morphons (default: 200)
        input_size: Dimension of observation vector
        output_size: Number of output classes/actions
        config: Optional SystemConfig for full customization
        task: Preset task profile ("rl", "classification", "temporal")
        seed: Random seed for reproducibility
    
    Example:
        >>> system = morphon.System(seed_size=200, input_size=4, output_size=2)
        >>> action = system.step(observation=[0.5, -0.1, 0.3, 0.0])
        >>> system.reward(1.0)
    """
    
    def __init__(
        self,
        seed_size: int = 200,
        input_size: int | None = None,
        output_size: int | None = None,
        config: SystemConfig | None = None,
        task: str | None = None,
        seed: int | None = None,
    ) -> None: ...
    
    # === CORE LOOP ===
    
    def step(self, observation: list[float] | np.ndarray) -> list[float]:
        """Process one observation through the network.
        
        Encodes the observation into sensory morphon activity,
        runs the network for `internal_steps` sub-ticks, and
        reads out the action/classification from motor morphons.
        
        Args:
            observation: Input vector (length must match input_size)
        
        Returns:
            Output vector (length = output_size). For RL tasks,
            argmax gives the action. For classification, argmax
            gives the predicted class.
        
        Note:
            The GIL is released during computation. Safe for threading.
        """
        ...
    
    def reward(self, value: float) -> None:
        """Deliver a reward signal to the system.
        
        Triggers neuromodulatory broadcast (reward channel),
        updates eligibility traces, and may trigger tag-and-capture
        consolidation depending on Endoquilibrium state.
        
        Args:
            value: Reward magnitude. Positive = good, negative = bad.
                   Scaled internally by Endoquilibrium's reward_gain.
        """
        ...
    
    def episode_end(self, total_reward: float) -> None:
        """Signal the end of an episode (RL tasks).
        
        Triggers episode-gated capture, updates developmental stage
        detection, and resets episodic accumulators. Call this at the
        end of each RL episode or after each classification batch.
        
        Args:
            total_reward: Cumulative reward for the episode.
        """
        ...
    
    def reset(self) -> None:
        """Reset transient state (voltages, refractory timers).
        
        Does NOT reset learned weights, structural changes, or
        Endoquilibrium state. Use for between-episode resets in RL.
        """
        ...
    
    # === BATCH OPERATIONS ===
    
    def step_batch(
        self, observations: np.ndarray
    ) -> np.ndarray:
        """Process a batch of observations (classification tasks).
        
        More efficient than calling step() in a loop because the
        entire batch is processed in Rust with the GIL released.
        
        Args:
            observations: 2D array, shape (batch_size, input_size)
        
        Returns:
            2D array, shape (batch_size, output_size)
        """
        ...
    
    # === TRAINING HELPERS ===
    
    def train_readout(
        self,
        observations: np.ndarray,
        labels: np.ndarray,
        epochs: int = 5,
        lr: float = 0.02,
        lr_decay: float = 0.5,
    ) -> dict:
        """Train the supervised readout on labeled data.
        
        Runs observations through the MI network (unsupervised features),
        then trains the linear readout using gradient descent on the
        cross-entropy loss.
        
        Args:
            observations: Training data, shape (n_samples, input_size)
            labels: Integer class labels, shape (n_samples,)
            epochs: Number of training epochs
            lr: Initial learning rate
            lr_decay: Factor to multiply lr after each epoch
        
        Returns:
            Dict with training metrics: {"accuracy": float, 
            "loss_history": list[float], "per_class_accuracy": dict}
        """
        ...
    
    def evaluate(
        self, observations: np.ndarray, labels: np.ndarray
    ) -> dict:
        """Evaluate classification accuracy on test data.
        
        Returns:
            Dict with: {"accuracy": float, "per_class_accuracy": dict,
            "confusion_matrix": np.ndarray, "predictions": np.ndarray}
        """
        ...
    
    # === DAMAGE & RECOVERY (SELF-HEALING) ===
    
    def damage(
        self, fraction: float = 0.3, cell_type: str | None = None
    ) -> dict:
        """Kill a fraction of morphons (simulates damage).
        
        Args:
            fraction: Fraction of morphons to kill (0.0–1.0)
            cell_type: Optional filter ("associative", "sensory", etc.)
        
        Returns:
            Dict: {"killed": int, "surviving": int, "types_affected": dict}
        """
        ...
    
    def recover(self, ticks: int = 5000) -> dict:
        """Run morphogenesis to regrow after damage.
        
        Args:
            ticks: Number of ticks to run recovery
        
        Returns:
            Dict: {"new_morphons": int, "new_synapses": int,
            "total_morphons": int, "recovery_ratio": float}
        """
        ...
    
    # === INSPECTION ===
    
    @property
    def stats(self) -> SystemStats:
        """Current system statistics (morphon count, synapse count,
        firing rates, Endoquilibrium stage, etc.)."""
        ...
    
    @property
    def morphon_count(self) -> int: ...
    
    @property
    def synapse_count(self) -> int: ...
    
    @property
    def developmental_stage(self) -> str:
        """Current Endoquilibrium developmental stage."""
        ...
    
    @property
    def vitals(self) -> Vitals:
        """Current Endoquilibrium vital signs."""
        ...
    
    @property
    def channels(self) -> ChannelState:
        """Current Endoquilibrium channel gains and levers."""
        ...
    
    def morphons(self, cell_type: str | None = None) -> list[MorphonView]:
        """Get read-only views of morphon states.
        
        Args:
            cell_type: Optional filter by type
        
        Returns:
            List of MorphonView objects with position, threshold,
            energy, cell_type, firing_rate, consolidation.
        """
        ...
    
    def synapses(
        self, source: int | None = None, target: int | None = None
    ) -> list[SynapseView]:
        """Get read-only views of synapses."""
        ...
    
    def adjacency_matrix(self, sparse: bool = True) -> Any:
        """Get the weight matrix as scipy.sparse or numpy dense."""
        ...
    
    # === SERIALIZATION ===
    
    def save(self, path: str) -> None:
        """Save system state to disk (MessagePack format)."""
        ...
    
    @classmethod
    def load(cls, path: str) -> "System":
        """Load system state from disk."""
        ...
    
    def to_json(self) -> str:
        """Export system config and stats as JSON."""
        ...
    
    # === GENOME (Phase 6+) ===
    
    def export_genome(self, morphon_id: int) -> Genome:
        """Export a morphon's heritable blueprint."""
        ...
    
    def import_genome(self, genome: Genome, position: tuple[float, float] | None = None) -> int:
        """Import a genome and express it as a new morphon.
        
        Returns:
            MorphonId of the newly expressed morphon.
        """
        ...
```

### 3.2 morphon.SystemConfig — Full Configuration

```python
@dataclass
class SystemConfig:
    """Complete configuration for a MORPHON system.
    
    For most users, the preset task profiles (morphon.tasks.*) provide
    sensible defaults. Use SystemConfig for full control.
    """
    # Network structure
    seed_size: int = 200
    input_size: int = 4
    output_size: int = 2
    internal_steps: int = 5
    
    # Developmental parameters
    developmental: DevelopmentalConfig = field(default_factory=DevelopmentalConfig)
    
    # Learning parameters
    learning: LearningConfig = field(default_factory=LearningConfig)
    
    # Homeostasis parameters
    homeostasis: HomeostasisConfig = field(default_factory=HomeostasisConfig)
    
    # Endoquilibrium parameters
    endoquilibrium: EndoConfig = field(default_factory=EndoConfig)
    
    # Competition mode
    competition: CompetitionConfig = field(default_factory=CompetitionConfig)
    
    # Metabolic parameters
    metabolic: MetabolicConfig = field(default_factory=MetabolicConfig)

@dataclass
class DevelopmentalConfig:
    """Controls morphogenesis: cell division, differentiation, pruning."""
    cell_type_ratios: dict[str, float] = field(
        default_factory=lambda: {"sensory": 0.3, "associative": 0.5, "motor": 0.1, "modulatory": 0.1}
    )
    division_threshold: float = 2.0
    pruning_threshold: float = 0.1
    migration_rate: float = 0.01
    max_morphons: int = 2000

@dataclass
class LearningConfig:
    """Controls STDP, tag-and-capture, readout."""
    a_plus: float = 0.01         # LTP magnitude
    a_minus: float = 0.012       # LTD magnitude (slightly > LTP for stability)
    tau_eligibility: float = 20.0
    capture_threshold: float = 0.5
    readout_lr: float = 0.02
    teach_hidden: bool = True     # Supervised readout hint (cerebellar pattern)
    feedback_alignment: bool = True  # DFA for associative layer

@dataclass 
class CompetitionConfig:
    """Controls how morphons compete for activation."""
    mode: str = "local"  # "local" (iSTDP) or "global" (k-WTA, legacy)
    # Local inhibition params
    interneuron_ratio: float = 0.1
    istdp_rate: float = 0.001
    initial_inh_weight: float = -0.3
    # Global k-WTA params (legacy)
    kwta_fraction: float = 0.05

@dataclass
class EndoConfig:
    """Endoquilibrium regulation parameters."""
    enabled: bool = True
    fast_tau: float = 50.0
    slow_tau: float = 500.0
    # Firing rate setpoints per stage
    fr_assoc_min: float = 0.08
    fr_assoc_max: float = 0.25
    # Phase A levers
    winner_adaptation_mult_range: tuple[float, float] = (0.3, 2.5)
    capture_threshold_mult_range: tuple[float, float] = (0.5, 1.5)
    # Astrocytic gating
    astrocytic_gate: bool = True
    tau_astro: float = 500.0
    gate_threshold: float = 0.3
```

### 3.3 morphon.tasks — Pre-Configured Profiles

```python
# morphon/tasks.py

def cartpole() -> SystemConfig:
    """Pre-configured for CartPole-v1 (and similar RL tasks).
    
    Returns a SystemConfig optimized for RL with:
    - 100 seed morphons (small, fast)
    - 4 inputs, 2 outputs
    - Episode-gated capture enabled
    - Endoquilibrium with reward-based stage detection
    """
    return SystemConfig(
        seed_size=100,
        input_size=4,
        output_size=2,
        learning=LearningConfig(teach_hidden=True, readout_lr=0.1),
        endoquilibrium=EndoConfig(enabled=True),
    )

def mnist(num_classes: int = 10) -> SystemConfig:
    """Pre-configured for MNIST digit classification.
    
    Returns a SystemConfig optimized for classification with:
    - 300 seed morphons
    - 784 inputs (28x28 pixels), 10 outputs
    - Local inhibitory competition
    - Higher readout learning rate
    """
    return SystemConfig(
        seed_size=300,
        input_size=784,
        output_size=num_classes,
        competition=CompetitionConfig(mode="local", interneuron_ratio=0.1),
        learning=LearningConfig(
            teach_hidden=True,
            readout_lr=0.02,
            feedback_alignment=True,
        ),
    )

def temporal(input_size: int, output_size: int) -> SystemConfig:
    """Pre-configured for temporal/event-based tasks.
    
    Longer eligibility traces, higher membrane time constants
    for temporal integration. Suitable for DVS-Gesture, SHD, etc.
    """
    return SystemConfig(
        seed_size=400,
        input_size=input_size,
        output_size=output_size,
        learning=LearningConfig(tau_eligibility=40.0),
        homeostasis=HomeostasisConfig(tau_membrane=30.0),
    )

def custom(input_size: int, output_size: int, **kwargs) -> SystemConfig:
    """Build a SystemConfig from keyword arguments.
    
    Convenience for Jupyter notebooks:
        config = morphon.tasks.custom(4, 2, seed_size=300, competition_mode="local")
    """
    ...
```

---

## 4. Environment Integration

### 4.1 Gymnasium (OpenAI Gym)

```python
# morphon/envs.py

class GymAdapter:
    """Connects a MORPHON system to a Gymnasium environment.
    
    Handles observation encoding, action decoding, reward delivery,
    and episode boundaries automatically.
    
    Example:
        >>> import gymnasium as gym
        >>> env = gym.make("CartPole-v1")
        >>> system = morphon.System(config=morphon.tasks.cartpole())
        >>> adapter = morphon.GymAdapter(system, env)
        >>> results = adapter.run(episodes=500)
        >>> print(results.avg_reward_last_100)
    """
    
    def __init__(
        self,
        system: System,
        env: gym.Env,
        reward_scale: float = 1.0,
        observation_norm: bool = True,
    ) -> None: ...
    
    def run(
        self,
        episodes: int = 1000,
        max_steps: int = 500,
        callback: Callback | None = None,
        render: bool = False,
    ) -> TrainingResult: ...
    
    def run_episode(self) -> EpisodeResult: ...

@dataclass
class TrainingResult:
    """Results from a training run."""
    episode_rewards: list[float]
    episode_lengths: list[int]
    avg_reward_last_100: float
    solved: bool
    solved_at_episode: int | None
    stage_history: list[str]  # Endoquilibrium stage per episode
    
    def plot(self) -> None:
        """Plot reward curve with stage annotations."""
        ...

@dataclass
class EpisodeResult:
    reward: float
    length: int
    stage: str
```

### 4.2 MNIST / Classification

```python
class ClassificationAdapter:
    """Connects a MORPHON system to a classification task.
    
    Example:
        >>> from morphon.datasets import load_mnist
        >>> train_x, train_y, test_x, test_y = load_mnist()
        >>> system = morphon.System(config=morphon.tasks.mnist())
        >>> adapter = morphon.ClassificationAdapter(system)
        >>> results = adapter.train(train_x, train_y, epochs=5)
        >>> test_results = adapter.evaluate(test_x, test_y)
        >>> print(f"Test accuracy: {test_results.accuracy:.1%}")
    """
    
    def __init__(self, system: System) -> None: ...
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 5,
        batch_size: int = 100,
        lr: float = 0.02,
        callback: Callback | None = None,
    ) -> TrainingResult: ...
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> EvalResult: ...
    
    def damage_and_recover(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        damage_fraction: float = 0.3,
        recovery_ticks: int = 5000,
    ) -> SelfHealingResult: ...

@dataclass
class EvalResult:
    accuracy: float
    per_class_accuracy: dict[int, float]
    confusion_matrix: np.ndarray
    predictions: np.ndarray
    
    def plot_confusion(self) -> None: ...
    def plot_per_class(self) -> None: ...

@dataclass
class SelfHealingResult:
    accuracy_before: float
    accuracy_after_damage: float
    accuracy_after_recovery: float
    killed: int
    regrown: int
    
    def plot(self) -> None: ...
```

### 4.3 Custom Environments

```python
class CustomEnvironment(Protocol):
    """Protocol for custom environments.
    
    Implement this to connect MORPHON to any task:
    
        class MyEnv:
            def reset(self) -> np.ndarray: ...
            def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]: ...
            
            @property
            def observation_size(self) -> int: ...
            @property
            def action_size(self) -> int: ...
    """
    def reset(self) -> np.ndarray: ...
    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]: ...
    @property
    def observation_size(self) -> int: ...
    @property
    def action_size(self) -> int: ...
```

---

## 5. Diagnostics & Visualization

### 5.1 Real-Time Diagnostics

```python
@dataclass
class SystemStats:
    """Snapshot of system state."""
    tick: int
    morphon_count: int
    synapse_count: int
    developmental_stage: str
    
    # Firing rates
    fr_sensory: float
    fr_associative: float
    fr_motor: float
    
    # Endoquilibrium vitals
    prediction_error: float
    weight_entropy: float
    eligibility_density: float
    energy_mean: float
    
    # Competition health
    population_sparsity: float
    lifetime_sparsity: float
    winner_diversity_entropy: float
    
    # Channel gains
    reward_gain: float
    novelty_gain: float
    arousal_gain: float
    homeostasis_gain: float
    plasticity_mult: float
    threshold_bias: float

class DiagnosticLogger:
    """Collects diagnostics over time for analysis and plotting.
    
    Example:
        >>> logger = morphon.DiagnosticLogger()
        >>> system = morphon.System(config=config)
        >>> adapter = morphon.GymAdapter(system, env)
        >>> results = adapter.run(episodes=500, callback=logger)
        >>> logger.plot_vitals()
        >>> logger.plot_stages()
        >>> logger.plot_competition_health()
    """
    
    def __call__(self, system: System, episode: int, step: int) -> None:
        """Called automatically by adapters when used as callback."""
        ...
    
    def plot_vitals(self, save_path: str | None = None) -> None:
        """Plot Endoquilibrium vital signs over time."""
        ...
    
    def plot_stages(self, save_path: str | None = None) -> None:
        """Plot developmental stage transitions over time."""
        ...
    
    def plot_competition_health(self, save_path: str | None = None) -> None:
        """Plot population sparsity, lifetime sparsity, winner entropy."""
        ...
    
    def plot_firing_rates(self, save_path: str | None = None) -> None:
        """Plot per-type firing rates over time."""
        ...
    
    def to_dataframe(self) -> "pd.DataFrame":
        """Export all diagnostics as a pandas DataFrame."""
        ...
    
    def to_json(self, path: str) -> None:
        """Export diagnostics as JSON for benchmarking."""
        ...
```

### 5.2 Network Visualization

```python
class NetworkVisualizer:
    """Visualize the morphon network topology and activity.
    
    Example:
        >>> viz = morphon.NetworkVisualizer(system)
        >>> viz.plot_poincare()       # Morphons in hyperbolic space
        >>> viz.plot_activity()       # Current firing pattern
        >>> viz.plot_weight_matrix()  # Adjacency heatmap
    """
    
    def __init__(self, system: System) -> None: ...
    
    def plot_poincare(
        self,
        color_by: str = "cell_type",  # or "energy", "firing_rate", "cluster"
        show_synapses: bool = False,
        save_path: str | None = None,
    ) -> None:
        """Plot morphons in the Poincaré ball with optional synapse overlay."""
        ...
    
    def plot_activity(
        self,
        observation: np.ndarray | None = None,
        save_path: str | None = None,
    ) -> None:
        """Plot current or stimulus-driven activity pattern."""
        ...
    
    def plot_weight_matrix(
        self,
        cell_types: list[str] | None = None,
        save_path: str | None = None,
    ) -> None:
        """Plot weight matrix as heatmap (scipy.sparse rendered)."""
        ...
    
    def plot_lineage_tree(
        self,
        max_depth: int = 10,
        save_path: str | None = None,
    ) -> None:
        """Plot morphon lineage tree (requires MorphonGenome, Phase 6)."""
        ...
    
    def animate_development(
        self,
        frames: int = 100,
        save_path: str = "morphon_dev.gif",
    ) -> None:
        """Animate network growth and differentiation over time."""
        ...
```

---

## 6. Callback Protocol

```python
class Callback(Protocol):
    """Protocol for training callbacks.
    
    Implement any subset of these methods:
    """
    def on_episode_start(self, system: System, episode: int) -> None: ...
    def on_step(self, system: System, episode: int, step: int) -> None: ...
    def on_episode_end(self, system: System, episode: int, reward: float) -> None: ...
    def on_stage_change(self, system: System, old_stage: str, new_stage: str) -> None: ...
    def on_damage(self, system: System, killed: int) -> None: ...
    def on_recovery_complete(self, system: System, regrown: int) -> None: ...

class EarlyStopping:
    """Stop training when a metric stops improving.
    
    Example:
        >>> stopper = morphon.EarlyStopping(metric="avg_reward", patience=50, threshold=195.0)
        >>> adapter.run(episodes=2000, callback=stopper)
    """
    def __init__(self, metric: str, patience: int, threshold: float | None = None): ...

class CheckpointSaver:
    """Save system state at regular intervals.
    
    Example:
        >>> saver = morphon.CheckpointSaver(every_n_episodes=100, path="checkpoints/")
        >>> adapter.run(episodes=1000, callback=saver)
    """
    def __init__(self, every_n_episodes: int, path: str): ...

class WandBLogger:
    """Log metrics to Weights & Biases.
    
    Example:
        >>> logger = morphon.WandBLogger(project="morphon-cartpole")
        >>> adapter.run(episodes=1000, callback=logger)
    """
    def __init__(self, project: str, config: dict | None = None): ...
```

---

## 7. Built-in Datasets

```python
# morphon/datasets.py

def load_mnist(
    path: str = "~/.morphon/data",
    flatten: bool = True,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load MNIST dataset. Downloads automatically on first use.
    
    Returns:
        (train_X, train_y, test_X, test_y)
        train_X: shape (60000, 784) if flatten, (60000, 28, 28) otherwise
        train_y: shape (60000,), integer labels 0-9
    """
    ...

def load_fashion_mnist(**kwargs) -> tuple: ...

def load_dvs_gesture(
    path: str = "~/.morphon/data",
    time_bins: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load DVS128 Gesture dataset (event-based, temporal).
    
    Returns:
        Spike-binned tensors suitable for temporal MORPHON processing.
    """
    ...
```

---

## 8. PyO3 Rust Layer — Implementation Strategy

### 8.1 What Gets Exposed via #[pyclass]

```rust
// src/lib.rs

use pyo3::prelude::*;

/// The core system wrapper. Owns the Rust System and provides
/// Python-callable methods.
#[pyclass]
struct PySystem {
    inner: morphon_core::System,
}

#[pymethods]
impl PySystem {
    #[new]
    fn new(config_json: &str) -> PyResult<Self> {
        let config: SystemConfig = serde_json::from_str(config_json)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner: System::new(config) })
    }
    
    /// Process observation, return output. Releases GIL during computation.
    fn step<'py>(&mut self, py: Python<'py>, obs: Vec<f64>) -> PyResult<Vec<f64>> {
        py.allow_threads(|| {
            Ok(self.inner.process_observation(&obs))
        })
    }
    
    fn reward(&mut self, value: f64) {
        self.inner.deliver_reward(value);
    }
    
    fn episode_end(&mut self, total_reward: f64) {
        self.inner.end_episode(total_reward);
    }
    
    /// Batch processing — releases GIL for the entire batch.
    fn step_batch<'py>(
        &mut self,
        py: Python<'py>,
        observations: Vec<Vec<f64>>,
    ) -> PyResult<Vec<Vec<f64>>> {
        py.allow_threads(|| {
            Ok(observations.iter()
                .map(|obs| self.inner.process_observation(obs))
                .collect())
        })
    }
    
    /// Stats as JSON string (parsed in Python layer).
    fn stats_json(&self) -> String {
        serde_json::to_string(&self.inner.diagnostics()).unwrap()
    }
    
    /// Save to bytes (MessagePack).
    fn save_bytes(&self) -> PyResult<Vec<u8>> {
        rmp_serde::to_vec(&self.inner)
            .map_err(|e| PyIOError::new_err(e.to_string()))
    }
    
    /// Load from bytes.
    #[staticmethod]
    fn load_bytes(data: Vec<u8>) -> PyResult<Self> {
        let inner: System = rmp_serde::from_slice(&data)
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }
    
    /// Get morphon data as list of dicts (for inspection).
    fn morphons_json(&self, cell_type: Option<&str>) -> String {
        let views = self.inner.morphon_views(cell_type);
        serde_json::to_string(&views).unwrap()
    }
    
    /// Get adjacency as COO sparse format: (row_indices, col_indices, weights).
    fn adjacency_coo(&self) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
        self.inner.adjacency_coo()
    }
}

/// Module definition.
#[pymodule]
fn _morphon_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySystem>()?;
    Ok(())
}
```

### 8.2 Design Principle: Thin Rust, Rich Python

The Rust layer exposes:
- `PySystem` with step/reward/episode_end/stats_json/save/load/morphons_json/adjacency_coo
- Config goes in as JSON string (parsed in Rust via serde)
- Stats come out as JSON string (parsed in Python via json.loads)
- Adjacency comes out as COO tuples (converted to scipy.sparse in Python)

This minimizes the PyO3 surface area. All convenience methods, type hints, docstrings, plotting, and environment integration live in pure Python. Adding a new Python feature never requires recompiling Rust.

### 8.3 GIL Release Strategy

Every method that calls into the MORPHON fast path releases the GIL with `py.allow_threads()`. This includes: `step()`, `step_batch()`, `reward()`, `episode_end()`, `recover()`. Inspection methods (`stats_json()`, `morphons_json()`) hold the GIL because they're fast and return Python objects.

### 8.4 NumPy Integration

For batch operations, use `numpy::PyArray` from the `numpy` PyO3 crate to avoid copying:

```rust
use numpy::{PyArray2, PyReadonlyArray2};

#[pymethods]
impl PySystem {
    fn step_batch_numpy<'py>(
        &mut self,
        py: Python<'py>,
        observations: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let obs = observations.as_array();
        let results = py.allow_threads(|| {
            // Process batch in Rust, returning Vec<Vec<f64>>
            self.inner.process_batch(obs)
        });
        Ok(PyArray2::from_vec2(py, &results)?)
    }
}
```

---

## 9. WASM Target (Browser Demos)

### 9.1 Build

```bash
# Compile morphon-core to WASM via wasm-pack
wasm-pack build --target web --out-dir pkg
```

### 9.2 JavaScript API

```javascript
import init, { WasmSystem } from './pkg/morphon_core.js';

await init();
const system = new WasmSystem('{"seed_size": 100, "input_size": 4, "output_size": 2}');
const output = system.step([0.5, -0.1, 0.3, 0.0]);
system.reward(1.0);
system.episode_end(1.0);
const stats = JSON.parse(system.stats_json());
```

### 9.3 Interactive Demo

A standalone HTML page that runs MORPHON in the browser, visualizing the Poincaré ball with morphon positions, firing patterns, and developmental stages in real-time. Uses Canvas 2D or Three.js for rendering. No server required — runs entirely client-side.

---

## 10. Testing Strategy

### 10.1 Test Matrix

| Level | What | Framework | Count |
|---|---|---|---|
| Rust unit tests | Core engine correctness | cargo test | 154 (existing) + 30 new PyO3 |
| Python unit tests | SDK API correctness | pytest | ~50 tests |
| Integration tests | End-to-end task completion | pytest + fixtures | ~15 tests |
| Notebook tests | Jupyter notebooks execute without error | nbmake | 3 notebooks |
| Type checking | mypy passes on all Python code | mypy --strict | Full coverage |

### 10.2 Key Integration Tests

```python
def test_cartpole_solves():
    """System solves CartPole within 1000 episodes."""
    system = morphon.System(config=morphon.tasks.cartpole(), seed=42)
    adapter = morphon.GymAdapter(system, gym.make("CartPole-v1"))
    results = adapter.run(episodes=1000)
    assert results.avg_reward_last_100 >= 195.0

def test_mnist_self_healing():
    """System accuracy improves after damage + recovery."""
    system = morphon.System(config=morphon.tasks.mnist(), seed=42)
    adapter = morphon.ClassificationAdapter(system)
    # Train
    adapter.train(train_X[:3000], train_y[:3000], epochs=3)
    acc_before = adapter.evaluate(test_X, test_y).accuracy
    # Damage + recover
    result = adapter.damage_and_recover(test_X, test_y, damage_fraction=0.3)
    assert result.accuracy_after_recovery >= acc_before * 0.9  # at most 10% regression

def test_save_load_roundtrip():
    """System state survives save/load cycle."""
    system = morphon.System(seed_size=50, input_size=4, output_size=2, seed=42)
    system.step([0.5, -0.1, 0.3, 0.0])
    system.reward(1.0)
    path = "/tmp/morphon_test.msgpack"
    system.save(path)
    loaded = morphon.System.load(path)
    assert loaded.morphon_count == system.morphon_count
    assert loaded.synapse_count == system.synapse_count
```

---

## 11. Documentation

### 11.1 mkdocs Site Structure

```
docs/
├── index.md                # Overview + quick start
├── getting-started/
│   ├── installation.md     # pip install, from source, WASM
│   ├── first-experiment.md # CartPole in 5 minutes
│   └── concepts.md         # Morphons, synapses, Endoquilibrium explained
├── guides/
│   ├── rl-tasks.md         # Using MORPHON for reinforcement learning
│   ├── classification.md   # MNIST and beyond
│   ├── self-healing.md     # Damage + recovery experiments
│   ├── custom-tasks.md     # Custom environments
│   ├── diagnostics.md      # Understanding vitals, stages, competition
│   └── visualization.md    # Poincaré ball, weight matrices, lineage trees
├── api/
│   ├── system.md           # System class reference
│   ├── config.md           # Configuration dataclasses
│   ├── adapters.md         # GymAdapter, ClassificationAdapter
│   ├── diagnostics.md      # DiagnosticLogger, SystemStats
│   └── callbacks.md        # Callback protocol, built-in callbacks
├── research/
│   ├── architecture.md     # Technical architecture deep-dive
│   ├── endoquilibrium.md   # Regulation system explained
│   ├── competition.md      # Local inhibition + iSTDP
│   └── references.md       # Bibliography
└── changelog.md
```

### 11.2 README.md

```markdown
# MORPHON — Adaptive Intelligence Engine

Biologically inspired software for self-organizing, continuously learning AI systems.

## Quick Start

```bash
pip install morphon-core
```

```python
import morphon
import gymnasium as gym

# Solve CartPole in ~900 episodes
system = morphon.System(config=morphon.tasks.cartpole(), seed=42)
adapter = morphon.GymAdapter(system, gym.make("CartPole-v1"))
results = adapter.run(episodes=1000)
print(f"Solved: {results.solved} at episode {results.solved_at_episode}")
print(f"Avg reward (last 100): {results.avg_reward_last_100:.1f}")
```

## What Makes MORPHON Different

- **Self-organizing:** Network topology grows and adapts at runtime
- **Self-healing:** Survives 30% neuron loss and recovers to higher accuracy
- **No retraining:** Learns continuously, no separate train/inference phases
- **Edge-first:** Runs on Raspberry Pi, no cloud required
- **Biologically grounded:** Four-channel neuromodulation, predictive regulation

## Links

- [Documentation](https://morphon.dev/docs)
- [Paper (arXiv)](https://arxiv.org/abs/XXXX.XXXXX)
- [GitHub](https://github.com/SimplyLiz/morphon-core)
```

---

## 12. CI/CD Pipeline

### 12.1 GitHub Actions

```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]
jobs:
  rust-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - run: cargo test --all-features

  python-tests:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12", "3.13"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "${{ matrix.python-version }}" }
      - run: pip install maturin[patchelf] pytest numpy
      - run: maturin develop
      - run: pytest tests/ -v

  type-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install mypy numpy-stubs
      - run: mypy python/morphon/ --strict

  notebooks:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install maturin nbmake jupyter gymnasium
      - run: maturin develop
      - run: pytest --nbmake notebooks/

  publish:
    if: startsWith(github.ref, 'refs/tags/v')
    needs: [rust-tests, python-tests]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: PyO3/maturin-action@v1
        with:
          command: publish
          args: --skip-existing
        env:
          MATURIN_PYPI_TOKEN: ${{ secrets.PYPI_TOKEN }}
```

### 12.2 Multi-Platform Wheels

Build wheels for Linux (manylinux), macOS (x86_64 + ARM), and Windows using maturin-action's matrix build. Publish to PyPI on every tagged release.

---

## 13. Implementation Timeline

| Week | What | Deliverable |
|---|---|---|
| 1 | Cargo workspace setup, PyO3 scaffolding, maturin config | `maturin develop` builds successfully |
| 2 | PySystem core: step/reward/episode_end/stats_json | CartPole runs from Python |
| 3 | SystemConfig dataclasses, task presets, JSON config bridge | `morphon.tasks.cartpole()` works |
| 4 | GymAdapter, TrainingResult, basic plotting | CartPole quickstart notebook |
| 5 | ClassificationAdapter, MNIST dataset loader, eval metrics | MNIST notebook |
| 6 | DiagnosticLogger, NetworkVisualizer, Poincaré ball plot | Diagnostics notebook |
| 7 | save/load, callbacks, WASM target, type checking | Full test suite passes |
| 8 | Documentation site, README, CI/CD, PyPI test publish | `pip install morphon-core` works |

---

## 14. Dependencies

### 14.1 Rust (Cargo.toml)

```toml
[dependencies]
pyo3 = { version = "0.28", features = ["extension-module"] }
numpy = "0.28"             # NumPy array interop
serde_json = "1"           # Config serialization
rmp-serde = "1"            # MessagePack for save/load
morphon-core = { path = "../morphon-core" }  # The engine

[build-system]
requires = ["maturin>=1.0,<2.0"]
build-backend = "maturin"
```

### 14.2 Python (pyproject.toml)

```toml
[project]
name = "morphon-core"
version = "0.1.0"
description = "Adaptive Intelligence Engine — biologically inspired self-organizing AI"
readme = "README.md"
license = { text = "Apache-2.0" }
requires-python = ">=3.10"
dependencies = ["numpy>=1.24"]

[project.optional-dependencies]
viz = ["matplotlib>=3.7", "plotly>=5.0"]
gym = ["gymnasium>=0.29"]
datasets = ["requests", "gzip"]
wandb = ["wandb>=0.16"]
all = ["morphon-core[viz,gym,datasets,wandb]"]
dev = ["morphon-core[all]", "pytest", "mypy", "nbmake", "jupyter"]

[tool.maturin]
features = ["pyo3/extension-module"]
python-source = "python"
module-name = "morphon._morphon_core"
```

---

## 15. Versioning and Release Strategy

**v0.1.0:** Core API (System, step, reward, save/load), CartPole working, basic diagnostics.
**v0.2.0:** Classification adapter, MNIST support, self-healing API.
**v0.3.0:** Visualization, WASM target, full documentation site.
**v1.0.0:** Stable API, all benchmarks validated, DeMorphon support, Genome API. Breaking changes only after v1.0.

Semantic versioning: MAJOR.MINOR.PATCH. Pre-1.0, MINOR bumps may include breaking changes. Post-1.0, breaking changes require MAJOR bump.

---

*morphon-core: three lines to adaptive intelligence.*

```python
import morphon
system = morphon.System(config=morphon.tasks.cartpole())
system.run(env="CartPole-v1", episodes=1000)
```

*TasteHub GmbH, Wien, April 2026*
