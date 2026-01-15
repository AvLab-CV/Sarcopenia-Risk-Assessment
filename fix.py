import numpy as np
import einops

skels = dict(np.load("output/skeletons/skels_metrabs.npz"))
skels = {k:einops.rearrange(skels[k], 'time (joint coord) -> time joint coord', joint=17, coord=3) for k in skels}
np.savez("output/skeletons/skels_metrabs_17x3.npz", **skels)
