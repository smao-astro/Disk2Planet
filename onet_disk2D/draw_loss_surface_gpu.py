#!/usr/bin/env python
# coding: utf-8
import os

import jax

from onet_disk2D.visualization import inv_loss_surface

print(jax.devices())
inv_loss_surface.main(n_cpu=1)
