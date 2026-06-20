# oxis (Python)

Python bindings for [OXIS](https://github.com/jpvich/oxis) — validated
quantitative finance, implemented in Rust. The bindings are a thin wrapper over
the same pure pricing cores used by the Rust crate and the `oxis` CLI: the
Python layer never duplicates pricing logic.

## Install (development)

```bash
cd python
maturin develop            # builds and installs into the active venv
```

Build release wheels with `maturin build --release`.

## Usage

```python
import oxis

# Just the price:
oxis.black_scholes(spot=100, strike=105, rate=0.05, vol=0.2, t=1.0, option_type="call")
# -> 8.021352235143176

# Full result as a dict (mirrors the CLI output):
oxis.price(spot=100, strike=100, rate=0.05, vol=0.2, t=1.0, option_type="put")
# -> {'model': 'black-scholes', 'option_type': 'put', ..., 'price': 5.5735260...}
```

Invalid inputs raise `ValueError` with the same message the core reports, e.g.
`invalid input: volatility must be >= 0`.
