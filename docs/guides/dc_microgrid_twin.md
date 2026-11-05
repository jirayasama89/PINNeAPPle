# DC microgrid twin: a changed network, a wrong sensor, or a fast model outside its domain?

`pinneapple_systems.digital_twin.dc_microgrid` is a digital twin of a small DC microgrid that, when its readings stop
agreeing with what it expects, says *why*: a component of the network changed (a joint in a line is loosening, the PV
converter is losing efficiency, a droop converter's output resistance is growing), a sensor is wrong (a drifting
bias, more noise), or its fast model is being used outside the envelope it was trained on. It says which line,
converter, sensor or input, and how large, with a stated uncertainty.

```python
from pinneapple_systems.digital_twin.dc_microgrid import run_study

out = run_study("out", n_per_kind=20)      # summary.json, report.md, confusion.png, isolability.png, surrogates.png
print(out["trust"].decisao, out["benchmark"]["confusion"])
```

`python examples/dc_microgrid_twin/microgrid_twin.py` runs the same study (`--smoke` for a minute-long check) and
records it for the Lab; the Flow node is `twin.dc_microgrid`.

## The plant

Four buses in a ring at 380 V:

| bus | component | law |
|---|---|---|
| 1 | battery converter | droop `V1 = Vref1 - Rd1 I_S1` |
| 2 | PV boost converter in MPPT, resistive load | `V2 I_PV = eta P_pv`, `I_R2 = V2 / R_L2` |
| 3 | second converter | droop `V3 = Vref3 - Rd3 I_S3` |
| 4 | EV charger (constant power) | `V4 I_L4 = P_L4` |

Lines 1-2, 2-3, 3-4, 4-1 are two-conductor 50 mm2 copper cables (IEC 60228 class 2: 0.387 ohm/km at 20 °C). The
reference simulator (`plant.steady_state`) solves Kirchhoff's current law with these laws by Newton; every sample of
every case is checked for current mismatch and energy balance (sources = loads + line losses). Each 1 s sample is a
steady state when the converters settle much faster (`quasi_steady_valid`, 20 ms assumed). Thirteen sensors (four bus
voltages, nine currents) carry Gaussian noise.

**What is assumed:** the droop gains, the PV converter efficiency, the sensor noise (0.4 V, 0.1 A) and the operating
envelope are assumptions of this demo, not data of a measured installation. A publication needs the same study on a
hardware-in-the-loop or laboratory microgrid.

## The diagnosis

1. **Physics residuals** (`diagnose.raw_residuals`): KCL at four buses, Ohm's law on four lines, two droop laws, the
   MPPT power and the two load laws, from the readings and the known set-points. A sensor bias enters every law the
   sensor appears in; a loose joint in line 2-3 enters only the Ohm residual of line 2-3 (`signature_matrix`: every
   signature is different).
2. **Whitening**: the residual covariance is propagated from the sensor noise, `S = J diag(sigma^2) J^T = L L^T`,
   and `z = L^-1 r` is N(0, I) when healthy. With one residual per sensor and an invertible `J`, the whitened
   directions of the thirteen sensor biases are orthonormal (checked at every operating point).
3. **Generalised likelihood ratio** per hypothesis over a 30 s window, with each sample's own directions (they move
   with the operating point). Process faults are one-sided: a joint only adds resistance, a converter only loses
   efficiency. A noise increase shows as a variance excess along the sensor's direction.
4. **Kalman filter** (`FaultTracker`) on the seven fault parameters gives the size of a process fault with its
   standard deviation; the likelihood ratio gives a second estimate.
5. **Fast model and domain** (`Surrogate`, POD + GP): its innovations and the inputs against its training box.

Thresholds are calibrated on healthy runs (`Diagnoser.calibrate`); `run_benchmark` diagnoses controlled cases with a
known answer and returns the confusion matrix, the isolation of the target, size errors, false alarms and delays.

## Isolability: which faults look alike

`isolability(grid)` gives the angle between the whitened directions of every pair of hypotheses over the envelope
(90 deg: orthogonal; 0: the same fault for the data). Sensor biases fold to [0, 90] (either sign); two one-sided
process faults keep [0, 180], because a fault that looks like the *opposite* of another is told apart by the sign of
its fitted size. In this grid one pair is confusable: joints in lines 1-2 and 2-3, at the operating points where their
two currents both flow into, or both out of, bus 2. Its voltage is the noisiest quantity of both Ohm residuals and the
other laws pin it only weakly (`confusable_pairs`, `closest_pairs_unsigned` gives the folded angles too). A more
accurate V2 transducer, or a second one, is the remedy the analysis points to.

## A physics-informed fast model at equal budget

`pinn.fit_mlp` trains an MLP from the five inputs to the thirteen readings with
`pinneapple_neural.trainer.graybox.train_graybox`: labels from `N` reference solves plus the thirteen network laws on
unlabelled operating points (in residual sigmas), either in the envelope (`pinn_box`) or in a wider box
(`pinn_extended`, `EXTENDED_BOX`). The data-only network is the same call with the physics weight at zero.
`surrogate_study` compares them with POD + GP on the same labels and seeds, on held-out points inside the envelope and
in the shell around it, with the physics residual of every prediction and the rank correlation between a PINN's own
residual (computable without labels) and its true error.

For four buses the Newton solve is itself cheap (the report gives both timings); the comparison is about how few
reference solves a fast model needs and where it can be trusted, which matters for larger networks or plant models
without a closed form.

## Results

The committed run is in `examples/dc_microgrid_twin/_out/` (`report.md`, `summary.json`, figures); its trust report
lists every check with its limit, fixed in `study.LIMITS` before the run.

## Limits

- Simulated plant with assumed converter and sensor data; quasi-steady samples, no converter dynamics or switching.
- One fault at a time; simultaneous faults are not covered.
- The domain alarm uses the fast model's training box; the physics residuals hold everywhere and do not need it.
