#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Energy Symmetry Transmission (EST) — Reproducible Simulation & Optimizer
-----------------------------------------------------------------------

Single-file reference implementation for the computational experiment in:
"Energy Symmetry Transmission (EST): A Fundamental Reformulation of Electrical Energy Transfer"

Author: Mohamed Orhan Zeinel
Email : research@impirume.org
ORCID: https://orcid.org/0009-0008-1139-8102
Repo  : https://github.com/mohamedorhan/Energy-Symmetry-Transmission

What this script gives you (all-in-one):
  • Deterministic, reproducible EST toy model (1-D dissipative PDE).
  • Two control parameterizations for θ(t): constant (scalar) and Fourier series.
  • Robust SPSA optimizer (two simulations per iteration, dimension-free).
  • Baseline vs. optimized comparison (I_RMS, average boundary power P_avg, THD).
  • Safety checks: CFL stability auto-fix, bounds, optional θ smoothing (rate limiting).
  • Full logging to CSV + JSON; automatic plots (PNG) without style tweaks.
  • Zero non-standard dependencies beyond NumPy and Matplotlib (optional Pandas if available).

Usage (examples):
  python est_simulation.py --mode constant --iters 150
  python est_simulation.py --mode fourier --harmonics 2 --iters 200 --weights 1.0 -0.3 0.2
  python est_simulation.py --seed 42 --outdir runs --plot

Metrics optimized (by default):
  J = w_I*I_RMS + w_T*THD - w_P*P_avg
where I_RMS and THD are to be minimized; P_avg to be maximized.

License: MIT (c) 2025 Mohamed Orhan Zeinel
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np

# Matplotlib only used if --plot is passed
try:
    import matplotlib.pyplot as plt  # noqa: F401
    _HAVE_MPL = True
except Exception:
    _HAVE_MPL = False

# Optional Pandas for convenience CSV; falls back to csv module
try:
    import pandas as pd  # noqa: F401
    _HAVE_PANDAS = True
except Exception:
    _HAVE_PANDAS = False


# =============================================================================
# Physics & Control Model
# =============================================================================

@dataclass
class Config:
    # Domain & numerics
    L: float = 1.0                 # spatial length
    N: int = 256                   # spatial points
    dt: float = 2e-4               # time step
    Ttotal: float = 1.0            # total simulation time (s)

    # Physical parameters
    kappa: float = 1e-3
    b: float = 1.0
    gamma: float = 1.0

    # External drive
    f_amp: float = 0.25
    f_freq: float = 120.0          # Hz

    # Control smoothing (rate limiting): τ -> smaller = faster θ tracking
    theta_tau: float = 0.02        # seconds (set 0 to disable smoothing)

    # Control bounds (applied to every parameter)
    theta_min: float = -1.5
    theta_max: float = 1.5

    # Optimization (SPSA)
    iters: int = 200
    spsa_a: float = 0.05
    spsa_c: float = 0.1
    spsa_A: float = 10.0           # stability offset
    spsa_alpha: float = 0.602
    spsa_gamma: float = 0.101

    # Objective weights (w_I, w_P, w_T)
    w_I: float = 1.0               # penalize I_RMS
    w_P: float = 0.0               # reward P_avg (use negative sign inside J)
    w_T: float = 0.2               # penalize THD

    # THD parameters
    thd_harmonics: int = 8
    thd_window: str = "hann"       # "none" | "hann"
    thd_signal: str = "ut"         # "ut" (preferred) or "u"

    # Reproducibility & IO
    seed: int = 123
    outdir: str = "runs"
    plot: bool = True
    save_time_series: bool = False  # set True to store full signals

    # Control parameterization
    mode: str = "constant"         # "constant" or "fourier"
    harmonics: int = 1             # K (only for fourier): θ(t)=c0+Σ[a_k sin(kωt)+b_k cos(kωt)]


# Material/Control parameterizations (from EST paper)
def a_of_theta(theta: float) -> float:
    """Control-dependent stiffness a(θ)."""
    return -0.5 - 0.4 * math.tanh(2.0 * theta)


def sigma_of_theta(theta: float) -> float:
    """Control-dependent drive coupling σ(θ)."""
    return 0.4 + 0.3 * math.tanh(1.5 * theta)


# =============================================================================
# Utilities
# =============================================================================

def set_seed(seed: int) -> None:
    np.random.seed(seed)


def ensure_cfl(cfg: Config) -> Tuple[float, int]:
    """Check CFL-like condition and auto-fix dt if needed. Returns (dt, steps)."""
    dx = cfg.L / (cfg.N - 1)
    # For explicit Euler + diffusion term: dt <= dx^2 / (2*γ*κ)
    cfl_dt = dx * dx / (2.0 * cfg.gamma * cfg.kappa) if cfg.kappa > 0 else cfg.dt
    dt = cfg.dt
    if dt > cfl_dt:
        dt = 0.9 * cfl_dt
    steps = int(round(cfg.Ttotal / dt))
    return dt, steps


def hann_window(n: int) -> np.ndarray:
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / (n - 1))


def compute_thd(
    s: np.ndarray,
    fs: float,
    f0: float,
    n_harmonics: int = 8,
    window: str = "hann"
) -> float:
    """
    Compute THD (%) of a real signal s(t), fundamental near f0.

    Implementation:
    - Optional Hann window to reduce leakage.
    - FFT magnitude; locate the nearest bin to f0 for the fundamental.
    - Sum power of 2..n harmonics; return 100*sqrt(sum)/fundamental.

    Edge cases guarded: zero-length, zero fundamental → THD = 0.
    """
    n = len(s)
    if n < 4 or fs <= 0 or f0 <= 0:
        return 0.0

    x = np.asarray(s, dtype=float)
    if window == "hann":
        x = x * hann_window(n)

    # FFT positive frequencies
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mag = np.abs(X) / (n / 2.0)

    # fundamental index
    k1 = int(np.argmin(np.abs(freqs - f0)))
    A1 = mag[k1] if k1 < len(mag) else 0.0
    if A1 <= 1e-12:
        return 0.0

    # harmonics
    harm_power = 0.0
    for h in range(2, n_harmonics + 1):
        kh = int(np.argmin(np.abs(freqs - h * f0)))
        if kh < len(mag):
            harm_power += mag[kh] ** 2

    thd = 100.0 * math.sqrt(harm_power) / A1
    return float(thd)


def clamp(v: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.minimum(np.maximum(v, lo), hi)


def make_outdir(root: str) -> Path:
    path = Path(root).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    # create unique subdir per run
    i = 1
    sub = path / f"EST_run_{len(list(path.glob('EST_run_*')))+1:04d}"
    sub.mkdir(parents=True, exist_ok=False)
    return sub


# =============================================================================
# θ(t) Parameterizations
# =============================================================================

class ThetaSchedule:
    """Generate θ(t) from a parameter vector."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

        if cfg.mode == "constant":
            self.dim = 1
        elif cfg.mode == "fourier":
            # c0 + sum_{k=1..K} [ a_k sin(k ω t) + b_k cos(k ω t) ]
            self.dim = 1 + 2 * cfg.harmonics
        else:
            raise ValueError("mode must be 'constant' or 'fourier'")

    def theta_raw(self, t: float, p: np.ndarray) -> float:
        """Unfiltered θ_raw(t) from params p."""
        if self.cfg.mode == "constant":
            return float(p[0])
        # Fourier series with base ω = 2π f_freq
        omega = 2.0 * math.pi * self.cfg.f_freq
        val = float(p[0])  # c0
        # then pairs (a_k, b_k)
        for k in range(1, self.cfg.harmonics + 1):
            ak = p[2 * k - 1]
            bk = p[2 * k]
            val += ak * math.sin(k * omega * t) + bk * math.cos(k * omega * t)
        return float(val)

    def project(self, p: np.ndarray) -> np.ndarray:
        """Project parameters to [theta_min, theta_max]."""
        return clamp(p, self.cfg.theta_min, self.cfg.theta_max)


# =============================================================================
# Simulator
# =============================================================================

class Simulator:
    """1-D dissipative PDE with EST control; explicit Euler; Neumann BCs."""

    def __init__(self, cfg: Config, schedule: ThetaSchedule):
        self.cfg = cfg
        self.schedule = schedule

        self.dx = cfg.L / (cfg.N - 1)
        self.dt, self.steps = ensure_cfl(cfg)
        self.fs = 1.0 / self.dt

    def run(self, params: np.ndarray) -> Dict[str, np.ndarray | float]:
        """Run one simulation for given θ-params; return metrics and signals."""
        cfg = self.cfg
        p = self.schedule.project(np.array(params, dtype=float))

        u = np.zeros(cfg.N, dtype=float)
        ut_boundary: List[float] = []
        u_boundary: List[float] = []
        theta_series: List[float] = []

        theta_f = self.schedule.theta_raw(0.0, p)  # filtered θ
        alpha = 0.0
        if cfg.theta_tau and cfg.theta_tau > 1e-9:
            alpha = min(1.0, self.dt / cfg.theta_tau)  # smoothing factor

        I_vals: List[float] = []
        P_vals: List[float] = []

        for n in range(self.steps):
            t = n * self.dt

            # raw θ from schedule
            theta_r = self.schedule.theta_raw(t, p)
            # filtered θ to enforce rate limit
            theta_f = theta_f + alpha * (theta_r - theta_f)
            theta = float(theta_f)

            # drive
            f_drive = cfg.f_amp * math.sin(2.0 * math.pi * cfg.f_freq * t)

            # laplacian (second-order central)
            lap = (np.roll(u, -1) - 2.0 * u + np.roll(u, 1)) / (self.dx ** 2)
            dFdu = a_of_theta(theta) * u + cfg.b * u ** 3 - cfg.kappa * lap
            ut = -cfg.gamma * dFdu + sigma_of_theta(theta) * f_drive

            # forward Euler update + Neumann BCs
            u += self.dt * ut
            u[0] = u[1]
            u[-1] = u[-2]

            # log instantaneous metrics
            I_vals.append(float(np.mean(np.abs(ut))))
            P_out = -cfg.kappa * ((u[-1] - u[-2]) / self.dx) * ut[-1]
            P_vals.append(float(P_out))

            # signals for THD/saving
            ut_boundary.append(float(ut[-1]))
            u_boundary.append(float(u[-1]))
            theta_series.append(theta)

        # Aggregate metrics
        I_RMS = float(np.sqrt(np.mean(np.square(I_vals))))
        P_avg = float(np.mean(P_vals))
        signal = np.array(ut_boundary if cfg.thd_signal == "ut" else u_boundary, dtype=float)
        THD = compute_thd(
            signal, fs=self.fs, f0=cfg.f_freq,
            n_harmonics=cfg.thd_harmonics, window=cfg.thd_window
        )

        return {
            "I_RMS": I_RMS,
            "P_avg": P_avg,
            "THD": THD,
            "theta_params": p,
            "theta_series": np.array(theta_series),
            "u_boundary": np.array(u_boundary),
            "ut_boundary": np.array(ut_boundary),
        }


# =============================================================================
# Optimization (SPSA)
# =============================================================================

class SPSA:
    """Simultaneous Perturbation Stochastic Approximation (dimension-free)."""

    def __init__(self, cfg: Config, schedule: ThetaSchedule, sim: Simulator,
                 objective: Callable[[Dict[str, float]], float]):
        self.cfg = cfg
        self.schedule = schedule
        self.sim = sim
        self.objective = objective

    def optimize(self, p0: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, float]]]:
        cfg = self.cfg
        dim = len(p0)
        p = self.schedule.project(p0.copy())

        history: List[Dict[str, float]] = []

        for k in range(1, cfg.iters + 1):
            a_k = cfg.spsa_a / ((k + cfg.spsa_A) ** cfg.spsa_alpha)
            c_k = cfg.spsa_c / (k ** cfg.spsa_gamma)

            # Rademacher perturbation Δ_i ∈ {+1,-1}
            delta = np.where(np.random.rand(dim) < 0.5, -1.0, 1.0)

            p_plus = self.schedule.project(p + c_k * delta)
            p_minus = self.schedule.project(p - c_k * delta)

            res_plus = self.sim.run(p_plus)
            res_minus = self.sim.run(p_minus)

            J_plus = self.objective(res_plus)
            J_minus = self.objective(res_minus)

            ghat = (J_plus - J_minus) / (2.0 * c_k) * (1.0 / delta)

            # update
            p = self.schedule.project(p - a_k * ghat)

            # evaluate current point for logging
            res_curr = self.sim.run(p)
            J_curr = self.objective(res_curr)

            record = {
                "iter": k,
                "J": float(J_curr),
                "I_RMS": float(res_curr["I_RMS"]),
                "P_avg": float(res_curr["P_avg"]),
                "THD": float(res_curr["THD"]),
            }
            history.append(record)

        return p, history


# =============================================================================
# Objective
# =============================================================================

def make_objective(cfg: Config) -> Callable[[Dict[str, float]], float]:
    wI, wP, wT = cfg.w_I, cfg.w_P, cfg.w_T

    def J(res: Dict[str, float]) -> float:
        # J = wI * I_RMS + wT * THD - wP * P_avg
        return wI * float(res["I_RMS"]) + wT * float(res["THD"]) - wP * float(res["P_avg"])

    return J


# =============================================================================
# I/O Helpers
# =============================================================================

def save_history_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    if _HAVE_PANDAS:
        pd.DataFrame(rows).to_csv(path, index=False)
    else:
        if len(rows) == 0:
            return
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for r in rows:
                writer.writerow(r)


def save_json(path: Path, data: Dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def maybe_plot(cfg: Config, outdir: Path, baseline: Dict, final: Dict,
               history: List[Dict[str, float]], sim: Simulator) -> None:
    if not cfg.plot or not _HAVE_MPL:
        return

    # 1) Metrics vs iteration
    it = [r["iter"] for r in history]
    J_hist = [r["J"] for r in history]
    I_hist = [r["I_RMS"] for r in history]
    P_hist = [r["P_avg"] for r in history]
    T_hist = [r["THD"] for r in history]

    plt.figure()
    plt.plot(it, J_hist, label="Objective J")
    plt.xlabel("Iteration")
    plt.ylabel("J")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "history_J.png", dpi=160)

    plt.figure()
    plt.plot(it, I_hist, label="I_RMS")
    plt.plot(it, P_hist, label="P_avg")
    plt.plot(it, T_hist, label="THD (%)")
    plt.xlabel("Iteration")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "history_metrics.png", dpi=160)

    # 2) θ(t) final vs baseline
    plt.figure()
    plt.plot(final["theta_series"], label="theta(t) optimized")
    plt.xlabel("Time step")
    plt.ylabel("theta")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "theta_series.png", dpi=160)

    # 3) Spectrum of chosen signal at boundary (final)
    sig = np.array(final["ut_boundary"] if cfg.thd_signal == "ut" else final["u_boundary"])
    n = len(sig)
    if n >= 8:
        if cfg.thd_window == "hann":
            sig = sig * hann_window(n)
        X = np.fft.rfft(sig)
        freqs = np.fft.rfftfreq(n, d=1.0 / sim.fs)
        mag = np.abs(X) / (n / 2.0)

        plt.figure()
        plt.semilogy(freqs[1:], mag[1:])  # skip DC
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(outdir / "spectrum_boundary.png", dpi=160)

    # 4) Baseline vs final bars
    names = ["I_RMS", "P_avg", "THD"]
    bvals = [baseline["I_RMS"], baseline["P_avg"], baseline["THD"]]
    fvals = [final["I_RMS"], final["P_avg"], final["THD"]]

    plt.figure()
    x = np.arange(len(names))
    width = 0.35
    plt.bar(x - width / 2, bvals, width, label="Baseline")
    plt.bar(x + width / 2, fvals, width, label="Optimized")
    plt.xticks(x, names)
    plt.grid(True, axis="y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "baseline_vs_optimized.png", dpi=160)

    plt.close('all')


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> Config:
    p = argparse.ArgumentParser(
        description="EST reproducible simulation & optimizer (single file)."
    )
    # domain/numerics
    p.add_argument("--L", type=float, default=1.0)
    p.add_argument("--N", type=int, default=256)
    p.add_argument("--dt", type=float, default=2e-4)
    p.add_argument("--Ttotal", type=float, default=1.0)

    # physics
    p.add_argument("--kappa", type=float, default=1e-3)
    p.add_argument("--b", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=1.0)

    # drive
    p.add_argument("--f_amp", type=float, default=0.25)
    p.add_argument("--f_freq", type=float, default=120.0)

    # control smoothing
    p.add_argument("--theta_tau", type=float, default=0.02)

    # bounds
    p.add_argument("--theta_min", type=float, default=-1.5)
    p.add_argument("--theta_max", type=float, default=1.5)

    # optimization
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--spsa_a", type=float, default=0.05)
    p.add_argument("--spsa_c", type=float, default=0.1)
    p.add_argument("--spsa_A", type=float, default=10.0)
    p.add_argument("--spsa_alpha", type=float, default=0.602)
    p.add_argument("--spsa_gamma", type=float, default=0.101)

    # objective weights
    p.add_argument("--weights", nargs=3, type=float, default=[1.0, 0.0, 0.2],
                   metavar=("w_I", "w_P", "w_T"),
                   help="weights for (I_RMS, P_avg, THD). J = wI*I + wT*THD - wP*P.")

    # THD
    p.add_argument("--thd_harmonics", type=int, default=8)
    p.add_argument("--thd_window", type=str, default="hann", choices=["hann", "none"])
    p.add_argument("--thd_signal", type=str, default="ut", choices=["ut", "u"])

    # reproducibility & IO
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--outdir", type=str, default="runs")
    p.add_argument("--plot", action="store_true", default=False)
    p.add_argument("--save_time_series", action="store_true", default=False)

    # control mode
    p.add_argument("--mode", type=str, default="constant", choices=["constant", "fourier"])
    p.add_argument("--harmonics", type=int, default=1)

    args = p.parse_args()
    cfg = Config(
        L=args.L, N=args.N, dt=args.dt, Ttotal=args.Ttotal,
        kappa=args.kappa, b=args.b, gamma=args.gamma,
        f_amp=args.f_amp, f_freq=args.f_freq,
        theta_tau=args.theta_tau,
        theta_min=args.theta_min, theta_max=args.theta_max,
        iters=args.iters, spsa_a=args.spsa_a, spsa_c=args.spsa_c,
        spsa_A=args.spsa_A, spsa_alpha=args.spsa_alpha, spsa_gamma=args.spsa_gamma,
        w_I=args.weights[0], w_P=args.weights[1], w_T=args.weights[2],
        thd_harmonics=args.thd_harmonics, thd_window=args.thd_window,
        thd_signal=args.thd_signal,
        seed=args.seed, outdir=args.outdir, plot=args.plot,
        save_time_series=args.save_time_series,
        mode=args.mode, harmonics=args.harmonics,
    )
    return cfg


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)

    outdir = make_outdir(cfg.outdir)
    save_json(outdir / "config.json", asdict(cfg))

    # Schedule & simulator
    schedule = ThetaSchedule(cfg)
    sim = Simulator(cfg, schedule)
    J = make_objective(cfg)

    # Baseline (no optimization): θ=0 (or zero vector)
    p0 = np.zeros(schedule.dim, dtype=float)
    p0 = schedule.project(p0)
    baseline = sim.run(p0)
    baseline_J = J(baseline)

    # Optimize with SPSA
    opt = SPSA(cfg, schedule, sim, J)
    p_star, history = opt.optimize(p0.copy())
    final = sim.run(p_star)
    final_J = J(final)

    # Summaries
    summary = {
        "baseline": {
            "params": baseline["theta_params"].tolist(),
            "J": baseline_J,
            "I_RMS": baseline["I_RMS"],
            "P_avg": baseline["P_avg"],
            "THD": baseline["THD"],
        },
        "optimized": {
            "params": final["theta_params"].tolist(),
            "J": final_J,
            "I_RMS": final["I_RMS"],
            "P_avg": final["P_avg"],
            "THD": final["THD"],
        },
        "improvement": {
            "dJ": final_J - baseline_J,
            "I_RMS_rel": (final["I_RMS"] / (baseline["I_RMS"] + 1e-12)),
            "P_avg_rel": (final["P_avg"] / (baseline["P_avg"] + 1e-12)),
            "THD_rel": (final["THD"] / (baseline["THD"] + 1e-12)),
        },
    }
    save_json(outdir / "summary.json", summary)
    save_history_csv(outdir / "history.csv", history)

    # Optionally save time series for reproduction
    if cfg.save_time_series:
        np.save(outdir / "theta_series.npy", final["theta_series"])
        np.save(outdir / "u_boundary.npy", final["u_boundary"])
        np.save(outdir / "ut_boundary.npy", final["ut_boundary"])

    # Plots
    maybe_plot(cfg, outdir, baseline, final, history, sim)

    # Print concise report
    print("\n================ EST Report ================")
    print(f"Outdir         : {outdir}")
    print(f"Mode           : {cfg.mode} (dim={schedule.dim})")
    print(f"Baseline       : J={baseline_J:.6f} | I_RMS={baseline['I_RMS']:.6f} | "
          f"P_avg={baseline['P_avg']:.6f} | THD={baseline['THD']:.3f}%")
    print(f"Optimized      : J={final_J:.6f} | I_RMS={final['I_RMS']:.6f} | "
          f"P_avg={final['P_avg']:.6f} | THD={final['THD']:.3f}%")
    print(f"Improvement    : dJ={summary['improvement']['dJ']:.6f} | "
          f"I_RMS x{summary['improvement']['I_RMS_rel']:.3f} | "
          f"P_avg x{summary['improvement']['P_avg_rel']:.3f} | "
          f"THD x{summary['improvement']['THD_rel']:.3f}")
    print(f"Theta* params  : {np.array2string(np.array(summary['optimized']['params']), precision=6)}")
    print("===========================================\n")


if __name__ == "__main__":
    main()
