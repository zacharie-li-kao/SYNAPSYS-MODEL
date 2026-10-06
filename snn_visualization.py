"""
SNN Visualization Module
========================

Complete implementation of Spiking Neural Network visualization with:
- Poisson spike train generation
- Raster plots for spike visualization
- Weight evolution tracking
- Network activity visualization

This module extends the SpikingNeuralNetwork class from network.py with
comprehensive visualization capabilities.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches


# =============================================================================
# POISSON SPIKE TRAIN GENERATION
# =============================================================================

def generate_poisson_spike_train(rate_hz, duration_ms, seed=None):
    """
    Generate a Poisson spike train.
    
    Args:
        rate_hz: Firing rate in Hz
        duration_ms: Duration in milliseconds
        seed: Random seed for reproducibility
    
    Returns:
        list: Spike times in milliseconds
    """
    # A LOCAL generator, not the global RNG.
    #
    # `np.random.seed(seed)` reseeds the PROCESS-WIDE generator, so calling
    # this function silently changed every other stochastic result in the
    # program. In on_snn_run the network is built BEFORE the seeded call,
    # so device variability was not reproducible while spike trains were —
    # the worst of both, and impossible to diagnose from the outside.
    rng = np.random.default_rng(seed)

    n_expected = rate_hz * duration_ms / 1000.0

    # Drawing N ~ Poisson(rT) then placing the N spikes uniformly is a
    # correct Poisson process, by conditional uniformity.
    n_spikes = rng.poisson(n_expected)
    spike_times = sorted(rng.uniform(0, duration_ms, n_spikes))
    
    return list(spike_times)


def generate_poisson_spike_trains(n_neurons, rates_hz, duration_ms, seed=None):
    """
    Generate Poisson spike trains for multiple neurons.
    
    Args:
        n_neurons: Number of neurons
        rates_hz: Firing rate for each neuron (scalar or array)
        duration_ms: Duration in milliseconds
        seed: Random seed for reproducibility
    
    Returns:
        list of lists: Spike trains for each neuron
    """
    # No global reseed here either — each neuron gets its own derived seed
    # below, which is what makes the trains independent AND reproducible.

    # Handle scalar rate
    if np.isscalar(rates_hz):
        rates_hz = [rates_hz] * n_neurons
    
    spike_trains = []
    for i, rate in enumerate(rates_hz):
        # Use different seed for each neuron
        neuron_seed = seed + i if seed is not None else None
        spikes = generate_poisson_spike_train(rate, duration_ms, neuron_seed)
        spike_trains.append(spikes)
    
    return spike_trains


def image_to_spike_trains(image_array, duration_ms, max_rate_hz=100,
                          min_rate_hz=0, seed=None):
    """
    Convert 2D image to deterministic spike trains.
    Pixel intensity maps to regular (non-Poisson) firing rate.

    Args:
        image_array: 2D numpy array normalized [0,1]
        duration_ms: Simulation duration
        max_rate_hz: Maximum firing rate for intensity=1
        min_rate_hz: Minimum firing rate for intensity=0
        seed: Seed for the per-pixel phase offsets (see below). Pass a value
            for reproducible input; None draws fresh offsets.

    Returns:
        list of spike train lists (one per pixel)
    """
    image_flat = image_array.flatten()

    # Per-pixel PHASE OFFSET, uniform within one inter-spike interval.
    #
    # Every train used to start at exactly t = 0, so the whole image emitted
    # one perfectly synchronous volley and pixels of equal intensity stayed
    # phase-locked for the entire simulation. Downstream that is pathological:
    # a LIF layer sees all its input arrive in a single timestep, and STDP sees
    # every same-intensity pixel as one indistinguishable event, so no spatial
    # structure can be learned.
    #
    # The rate code is unchanged — each pixel still fires regularly at its own
    # rate — only the arbitrary common start phase is removed.
    rng = np.random.default_rng(seed)

    spike_trains = []
    for intensity in image_flat:
        # Linear mapping from intensity to rate
        rate_hz = min_rate_hz + intensity * (max_rate_hz - min_rate_hz)

        # Generate regular spike train
        if rate_hz > 0:
            interval_ms = 1000.0 / rate_hz
            phase_ms = float(rng.uniform(0.0, interval_ms))
            spikes = list(np.arange(phase_ms, duration_ms, interval_ms))
        else:
            spikes = []
        spike_trains.append(spikes)

    return spike_trains



# =============================================================================
# SPIKE RASTER PLOT
# =============================================================================

def plot_spike_raster(spike_trains, duration_ms, title="Spike Raster Plot", 
                      neuron_labels=None, colors=None, ax=None):
    """
    Create a raster plot showing spike times for multiple neurons.
    
    Args:
        spike_trains: List of spike train lists (one per neuron)
        duration_ms: Total duration in ms
        title: Plot title
        neuron_labels: Optional labels for neurons
        colors: Optional colors for each neuron
        ax: Optional matplotlib axis
    
    Returns:
        matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    n_neurons = len(spike_trains)
    
    # Default colors
    if colors is None:
        cmap = plt.cm.tab10
        colors = [cmap(i % 10) for i in range(n_neurons)]
    
    # Plot spikes
    for i, (spikes, color) in enumerate(zip(spike_trains, colors)):
        if spikes:
            ax.scatter(spikes, [i] * len(spikes), 
                      marker='|', s=100, c=[color], linewidths=2)
    
    # Labels
    if neuron_labels is None:
        neuron_labels = [f"Neuron {i}" for i in range(n_neurons)]
    
    ax.set_yticks(range(n_neurons))
    ax.set_yticklabels(neuron_labels)
    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_ylabel('Neuron', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim([0, duration_ms])
    ax.set_ylim([-0.5, n_neurons - 0.5])
    ax.grid(True, alpha=0.3, axis='x')
    
    return ax


# =============================================================================
# SNN COMPREHENSIVE VISUALIZATION
# =============================================================================

def visualize_snn_simulation(snn_results, input_spike_trains, 
                            hidden_spike_trains=None, 
                            show_weight_evolution=True,
                            figsize=(16, 10)):
    """
    Comprehensive visualization of SNN simulation results.
    
    Args:
        snn_results: Results dictionary from SpikingNeuralNetwork.simulate()
        input_spike_trains: Input spike trains (for raster plot)
        hidden_spike_trains: Hidden layer spike trains (optional, from results)
        show_weight_evolution: Whether to show weight matrix evolution
        figsize: Figure size
    
    Returns:
        matplotlib figure
    """
    # Extract data from results
    if hidden_spike_trains is None:
        hidden_spike_trains = snn_results.get('hidden_spikes', [])
    
    weight_history = snn_results.get('weight_history', [])
    time_points = snn_results.get('time_points', [])
    duration_ms = time_points[-1] if time_points else 1000
    
    # Create figure with subplots
    if show_weight_evolution and weight_history:
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # Input raster
        ax1 = fig.add_subplot(gs[0, :])
        plot_spike_raster(input_spike_trains, duration_ms, 
                         title="Input Layer Spike Trains", ax=ax1)
        
        # Hidden raster
        ax2 = fig.add_subplot(gs[1, :])
        plot_spike_raster(hidden_spike_trains, duration_ms, 
                         title="Hidden Layer Spike Trains", ax=ax2)
        
        # Initial weights
        ax3 = fig.add_subplot(gs[2, 0])
        plot_weight_matrix(weight_history[0], title="Initial Weights", ax=ax3)
        
        # Final weights
        ax4 = fig.add_subplot(gs[2, 1])
        plot_weight_matrix(weight_history[-1], title="Final Weights", ax=ax4)
        
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
        
        # Input raster
        plot_spike_raster(input_spike_trains, duration_ms, 
                         title="Input Layer Spike Trains", ax=ax1)
        
        # Hidden raster
        plot_spike_raster(hidden_spike_trains, duration_ms, 
                         title="Hidden Layer Spike Trains", ax=ax2)
    
    plt.tight_layout()
    return fig


def plot_weight_matrix(weights, title="Weight Matrix", ax=None, cmap=None):
    """
    Visualize weight matrix as a heatmap.

    Args:
        weights: 2D numpy array of weights
        title: Plot title
        ax: Optional matplotlib axis
        cmap: Colormap. Chosen automatically when None — see below.

    Returns:
        matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    # Colormap chosen from the DATA, not fixed to a diverging one.
    #
    # The default was 'RdBu_r' with vmin = 0 for strictly positive data. A
    # diverging map's midpoint is its neutral colour (white), so with the scale
    # running 0..vmax every mid-range conductance was drawn as white — read as
    # "nothing here" when it is in fact half the dynamic range. Conductances
    # are strictly positive and have no meaningful zero point, so a SEQUENTIAL
    # map is correct; a diverging one is right only when the data genuinely
    # straddles zero (e.g. a weight-CHANGE matrix).
    has_negative = bool(np.any(weights < 0)) if weights.size else False
    if cmap is None:
        cmap = 'RdBu_r' if has_negative else 'viridis'

    vmax = np.abs(weights).max() if weights.size > 0 else 1
    if has_negative:
        vmin = -vmax
    else:
        # Sequential: span the data's own range rather than anchoring at 0,
        # which would waste most of the colour scale on values that never occur.
        vmin = float(np.min(weights)) if weights.size else 0.0
        vmax = float(np.max(weights)) if weights.size else 1.0
        if vmax <= vmin:
            vmax = vmin + 1e-30

    im = ax.imshow(weights, cmap=cmap, aspect='auto',
                   interpolation='nearest', vmin=vmin, vmax=vmax)
    
    ax.set_xlabel('Hidden Neurons', fontsize=11)
    ax.set_ylabel('Input Neurons', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Weight (µS)', fontsize=10)
    
    return ax


def plot_weight_evolution(weight_history, time_points, 
                         input_idx=None, hidden_idx=None,
                         figsize=(12, 6)):
    """
    Plot evolution of specific synaptic weights over time.
    
    Args:
        weight_history: List of weight matrices over time
        time_points: Time points corresponding to weight_history
        input_idx: Input neuron indices to plot (None = all)
        hidden_idx: Hidden neuron index to plot (None = first)
        figsize: Figure size
    
    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    weight_history = np.array(weight_history)
    n_input, n_hidden = weight_history[0].shape
    
    # Default: plot all input synapses to first hidden neuron
    if hidden_idx is None:
        hidden_idx = 0
    if input_idx is None:
        input_idx = range(n_input)
    
    # Plot weight trajectories
    for i in input_idx:
        # M20: weight_history comes from get_weight_matrix(), which ALREADY
        # returns microsiemens. Multiplying by 1e6 again plotted values 1e6
        # times too large — a 50 µS synapse appeared as 5e7 µS.
        weights_over_time = weight_history[:, i, hidden_idx]
        ax.plot(time_points, weights_over_time, 
               marker='o', markersize=4, linewidth=2,
               label=f'Input {i} → Hidden {hidden_idx}')
    
    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_ylabel('Synaptic Weight (µS)', fontsize=12)
    ax.set_title(f'Weight Evolution to Hidden Neuron {hidden_idx}', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    return fig


# =============================================================================
# FIRING RATE ANALYSIS
# =============================================================================

def compute_firing_rates(spike_trains, duration_ms, bin_size_ms=100):
    """
    Compute instantaneous firing rates using sliding time window.
    
    Args:
        spike_trains: List of spike trains
        duration_ms: Total duration
        bin_size_ms: Size of time bins
    
    Returns:
        tuple: (time_bins, firing_rates) where firing_rates is n_neurons × n_bins
    """
    n_neurons = len(spike_trains)

    # M19: bin EDGES must be spaced by exactly bin_size_ms.
    #
    # This used `np.linspace(0, duration_ms, n_bins)`, whose spacing is
    # duration/(n_bins - 1) — slightly WIDER than bin_size_ms — while the
    # counting window below is bin_size_ms wide. Consecutive windows therefore
    # left a gap, and roughly 11% of spikes fell into none of them and were
    # never counted. The final bin also started at exactly duration_ms, so it
    # covered no data at all and always reported 0 Hz — a spurious drop to zero
    # at the right-hand edge of every firing-rate plot.
    #
    # arange gives edges at 0, bin, 2*bin, ... which tile the interval exactly.
    if bin_size_ms <= 0:
        raise ValueError(f"bin_size_ms must be positive, got {bin_size_ms}")

    time_bins = np.arange(0.0, duration_ms, bin_size_ms)
    if time_bins.size == 0:
        time_bins = np.array([0.0])
    n_bins = time_bins.size
    firing_rates = np.zeros((n_neurons, n_bins))

    for i, spikes in enumerate(spike_trains):
        spikes_array = np.asarray(spikes, dtype=float)
        for j, t in enumerate(time_bins):
            # Count spikes in bin. The last bin is clipped to duration_ms so a
            # duration that is not a whole number of bins does not report an
            # artificially low rate for its final, short window.
            bin_end = min(t + bin_size_ms, duration_ms)
            width_ms = bin_end - t
            if width_ms <= 0:
                continue
            bin_spikes = np.sum((spikes_array >= t) & (spikes_array < bin_end))
            firing_rates[i, j] = bin_spikes * 1000.0 / width_ms

    return time_bins, firing_rates


def plot_firing_rates(spike_trains, duration_ms, bin_size_ms=100,
                     title="Firing Rates", ax=None):
    """
    Plot instantaneous firing rates over time.
    
    Args:
        spike_trains: List of spike trains
        duration_ms: Total duration
        bin_size_ms: Size of time bins
        title: Plot title
        ax: Optional matplotlib axis
    
    Returns:
        matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    time_bins, firing_rates = compute_firing_rates(spike_trains, duration_ms, bin_size_ms)
    
    # Plot each neuron
    for i, rates in enumerate(firing_rates):
        ax.plot(time_bins, rates, linewidth=2, alpha=0.7, label=f'Neuron {i}')
    
    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_ylabel('Firing Rate (Hz)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    return ax


# =============================================================================
# STDP/SRDP VISUALIZATION FROM EXPERIMENTAL DATA
# =============================================================================

def plot_stdp_curve(stdp_data, fitted_params=None, ax=None, show_baseline=True):
    """
    Plot STDP timing window with experimental data and optional fitted curve.
    
    Args:
        stdp_data: dict with 'delta_t_ms' and 'delta_g_percent' (or 'delta_g_S')
        fitted_params: Optional dict with {A_plus, A_minus, tau_plus_ms, tau_minus_ms}
        ax: Optional matplotlib axis
        show_baseline: Show zero baseline
    
    Returns:
        matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    delta_t = np.array(stdp_data['delta_t_ms'])
    
    # Use delta_g_percent if available, else convert delta_g_S
    if 'delta_g_percent' in stdp_data:
        delta_g = np.array(stdp_data['delta_g_percent'])
        ylabel = 'ΔG (%)'
    elif 'delta_g_S' in stdp_data:
        delta_g = np.array(stdp_data['delta_g_S']) * 1e6  # Convert to µS
        ylabel = 'ΔG (µS)'
    else:
        raise ValueError("stdp_data must contain 'delta_g_percent' or 'delta_g_S'")
    
    # Plot experimental data
    ax.scatter(delta_t, delta_g, s=80, alpha=0.7, label='Experimental Data', 
               color='black', zorder=3, edgecolors='white', linewidths=1)
    
    # Plot fitted curve if provided
    if fitted_params:
        t_fit = np.linspace(delta_t.min(), delta_t.max(), 300)
        dg_fit = np.zeros_like(t_fit)
        
        A_plus = fitted_params['A_plus']
        A_minus = fitted_params['A_minus']
        tau_plus = fitted_params['tau_plus_ms']
        tau_minus = fitted_params['tau_minus_ms']
        
        # LTP side (Δt > 0)
        ltp_mask = t_fit > 0
        dg_fit[ltp_mask] = A_plus * np.exp(-t_fit[ltp_mask] / tau_plus)
        
        # LTD side (Δt < 0)
        ltd_mask = t_fit < 0
        dg_fit[ltd_mask] = -A_minus * np.exp(t_fit[ltd_mask] / tau_minus)
        
        ax.plot(t_fit, dg_fit, 'r-', linewidth=2.5, label='Fitted STDP', zorder=2)
        
        # Add fit parameters as text box
        param_text = (f'LTP: A₊={A_plus:.2f}, τ₊={tau_plus:.1f} ms\n'
                      f'LTD: A₋={A_minus:.2f}, τ₋={tau_minus:.1f} ms')
        if 'fit_quality_R2' in fitted_params:
            param_text += f'\nR² = {fitted_params["fit_quality_R2"]:.3f}'
        
        ax.text(0.98, 0.98, param_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Reference lines
    if show_baseline:
        ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.axvline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Δt (ms) [post - pre]', fontsize=13, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=13, fontweight='bold')
    ax.set_title('STDP Timing Window', fontsize=15, fontweight='bold')
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_srdp_curve(srdp_data, fitted_params=None, ax=None):
    """
    Plot SRDP frequency response with experimental data and optional fitted curve.
    
    Args:
        srdp_data: dict with 'frequencies_hz' and 'delta_g_percent' (or 'delta_g_S')
        fitted_params: Optional dict with sigmoid parameters
        ax: Optional matplotlib axis
    
    Returns:
        matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    freq = np.array(srdp_data['frequencies_hz'])
    
    # Use delta_g_percent if available, else convert delta_g_S
    if 'delta_g_percent' in srdp_data:
        delta_g = np.array(srdp_data['delta_g_percent'])
        ylabel = 'ΔG (%)'
    elif 'delta_g_S' in srdp_data:
        delta_g = np.array(srdp_data['delta_g_S']) * 1e6
        ylabel = 'ΔG (µS)'
    else:
        raise ValueError("srdp_data must contain 'delta_g_percent' or 'delta_g_S'")
    
    # Plot experimental data
    ax.scatter(freq, delta_g, s=80, alpha=0.7, label='Experimental Data',
               color='black', zorder=3, edgecolors='white', linewidths=1)
    
    # Plot fitted curve if provided.
    #
    # The keys are read WITHOUT defaults, and both the `srdp_`-prefixed names
    # that `fitting.extract_synapse_model` produces and the bare names that
    # `fit_srdp_curve` returns are accepted.
    #
    # This used to read 'amplitude'/'transition_freq_hz'/'slope' with fallback
    # defaults. A fitted_model carries `srdp_max_change`,
    # `srdp_transition_freq_hz` and `srdp_slope`, so EVERY lookup missed and
    # the function drew a fabricated f0 = 10 / k = 10 sigmoid, labelled
    # "Fitted SRDP", over the user's real data. Its sibling plot_stdp_curve
    # raises on a missing key instead; the two now agree.
    if fitted_params:
        def _require(*names):
            for name in names:
                if name in fitted_params and fitted_params[name] is not None:
                    return fitted_params[name]
            raise KeyError(
                f"SRDP parameters missing: none of {names} present. "
                "Drawing a curve from defaults would label a fabricated "
                "sigmoid as a fit."
            )

        A = _require('srdp_max_change', 'max_change')
        f0 = _require('srdp_transition_freq_hz', 'transition_freq_hz')
        # Slope is in DECADES of frequency (fitting.py fits the sigmoid in
        # log10 f, per audit M7), so the curve is evaluated in log space too.
        k_decades = _require('srdp_slope', 'slope')
        baseline = fitted_params.get('baseline', 0.0)

        f_fit = np.logspace(np.log10(freq.min()), np.log10(freq.max()), 300)
        dg_fit = baseline + A / (
            1 + np.exp(-(np.log10(f_fit) - np.log10(f0)) / k_decades)
        )

        ax.plot(f_fit, dg_fit, 'b-', linewidth=2.5, label='Fitted SRDP', zorder=2)

        param_text = (f'f₀ = {f0:.2f} Hz\n'
                      f'slope = {k_decades:.2f} decades\n'
                      f'Max ΔG = {A:.1f}')
        ax.text(0.02, 0.98, param_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    ax.set_xscale('log')
    ax.set_xlabel('Spike Frequency (Hz)', fontsize=13, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=13, fontweight='bold')
    ax.set_title('SRDP Frequency Response', fontsize=15, fontweight='bold')
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, which='both')
    
    return ax



# =============================================================================
# DEMO / TEST FUNCTION
# =============================================================================

def demo_snn_visualization():
    """
    Demonstration of SNN visualization capabilities.
    """
    print("\n" + "="*70)
    print("SNN VISUALIZATION DEMO")
    print("="*70)
    
    # Generate test spike trains
    print("\n1. Generating Poisson spike trains...")
    n_input = 5
    n_hidden = 3
    duration_ms = 1000
    
    # Different rates for input neurons
    input_rates = [10, 15, 20, 25, 30]  # Hz
    input_spikes = generate_poisson_spike_trains(n_input, input_rates, 
                                                 duration_ms, seed=42)
    
    # Simulate some hidden layer activity
    hidden_rates = [5, 8, 12]  # Hz
    hidden_spikes = generate_poisson_spike_trains(n_hidden, hidden_rates,
                                                  duration_ms, seed=123)
    
    print(f"   Generated {n_input} input spike trains")
    print(f"   Generated {n_hidden} hidden spike trains")
    
    # Create dummy weight history
    print("\n2. Creating weight evolution...")
    n_steps = 10
    time_points = np.linspace(0, duration_ms, n_steps)
    weight_history = []
    
    for t in time_points:
        # Weights evolve over time (simple linear increase for demo)
        W = np.random.randn(n_input, n_hidden) * 1e-5 * (1 + t/duration_ms)
        weight_history.append(W)
    
    # Package as results dictionary
    snn_results = {
        'hidden_spikes': hidden_spikes,
        'weight_history': weight_history,
        'time_points': time_points.tolist()
    }
    
    print("\n3. Creating visualizations...")
    
    # Full visualization
    fig1 = visualize_snn_simulation(snn_results, input_spikes, 
                                    show_weight_evolution=True)
    fig1.suptitle('SNN Simulation Results', fontsize=16, fontweight='bold', y=0.995)
    
    # Weight evolution
    fig2 = plot_weight_evolution(weight_history, time_points, 
                                 input_idx=[0, 1, 2], hidden_idx=0)
    
    # Firing rates
    fig3, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    plot_firing_rates(input_spikes, duration_ms, bin_size_ms=50,
                     title="Input Layer Firing Rates", ax=ax1)
    plot_firing_rates(hidden_spikes, duration_ms, bin_size_ms=50,
                     title="Hidden Layer Firing Rates", ax=ax2)
    plt.tight_layout()
    
    print("\n✓ Visualizations created successfully!")
    print("\n" + "="*70)
    
    return fig1, fig2, fig3


# =============================================================================
# INTEGRATION WITH NETWORK.PY
# =============================================================================

def run_snn_with_visualization(snn_network, input_rates_hz, duration_ms,
                               learning_enabled=True, seed=None):
    """
    Run SNN simulation with automatic Poisson input generation and visualization.
    
    Args:
        snn_network: SpikingNeuralNetwork instance from network.py
        input_rates_hz: Firing rates for input neurons (Hz)
        duration_ms: Simulation duration (ms)
        learning_enabled: Enable STDP learning
        seed: Random seed
    
    Returns:
        tuple: (results_dict, figure)
    """
    print("\n" + "="*70)
    print("RUNNING SNN SIMULATION WITH VISUALIZATION")
    print("="*70)
    
    # Generate Poisson input
    print(f"\n1. Generating Poisson inputs ({input_rates_hz} Hz)...")
    input_spike_trains = generate_poisson_spike_trains(
        snn_network.n_input, 
        input_rates_hz, 
        duration_ms, 
        seed=seed
    )
    
    # Count total spikes
    total_spikes = sum(len(spikes) for spikes in input_spike_trains)
    print(f"   Generated {total_spikes} total input spikes")
    
    # Run simulation
    print(f"\n2. Running simulation ({duration_ms} ms)...")
    results = snn_network.simulate(input_spike_trains, duration_ms, 
                                   learning_enabled=learning_enabled)
    
    # Count output spikes
    output_spikes = sum(len(spikes) for spikes in results['hidden_spikes'])
    print(f"   Hidden layer produced {output_spikes} spikes")
    
    # Visualize
    print("\n3. Creating visualizations...")
    fig = visualize_snn_simulation(results, input_spike_trains, 
                                   show_weight_evolution=learning_enabled)
    
    print("\n✓ Simulation complete!")
    print("="*70)
    
    return results, fig


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == "__main__":
    print("Testing SNN Visualization Module")
    
    # Run demo
    fig1, fig2, fig3 = demo_snn_visualization()
    
    plt.show()