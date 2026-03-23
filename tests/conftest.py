from __future__ import annotations

import os

# Force CPU for tests
os.environ['JAX_PLATFORMS'] = 'cpu'
