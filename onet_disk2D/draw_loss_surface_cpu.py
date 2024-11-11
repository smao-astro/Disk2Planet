#!/usr/bin/env python
# coding: utf-8
import os

n_cpu = 8
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={n_cpu}"
import jax

from onet_disk2D.visualization import inv_loss_surface

print(jax.devices())
inv_loss_surface.main(n_cpu=n_cpu)
