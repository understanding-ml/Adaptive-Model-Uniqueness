# isqed/ecosystem.py

from typing import Iterable, Any, Tuple
import numpy as np

from isqed.core import ModelUnit, Intervention


class Ecosystem:
    def __init__(self, target: ModelUnit, peers: list[ModelUnit]):
        """
        Container for a target model and its peer models.

        Parameters
        ----------
        target:
            The model to be audited (the "target" in ISQED terminology).
        peers:
            A list of peer models used to construct convex combinations.
        """
        self.target = target
        self.peers = peers

    def batched_query(
        self,
        X: Iterable[Any],
        Thetas,
        intervention: Intervention,
        seeds=None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Query the target and all peers on a batch of (input, theta) pairs.

        This method applies the intervention once per sample so that the
        target and all peers see exactly the same perturbed input. This is
        important for matched in-silico experiments where we want to control
        the corruption pattern shared across models.

        Parameters
        ----------
        X:
            Iterable of raw inputs (e.g., text strings, feature vectors).
        Thetas:
            Either a single float (broadcast to all inputs) or an iterable
            of floats with the same length as X.
        intervention:
            Intervention object implementing `apply`. If the concrete
            implementation accepts a third argument `seed`, it will be
            passed from the `seeds` argument below.
        seeds:
            Optional seeding information to make the intervention deterministic.
            Can be:
              * None: no explicit seeding, intervention controls randomness;
              * a single int: broadcast to all samples;
              * an iterable of ints, same length as X.

        Returns
        -------
        y_target : np.ndarray, shape (n_samples,)
            Scalar responses of the target model.
        Y_peers : np.ndarray, shape (n_samples, n_peers)
            Scalar responses of each peer model, row-aligned with y_target.
        """
        inputs = list(X)
        n = len(inputs)

        # Normalize Thetas to a list of length n.
        if isinstance(Thetas, (int, float)):
            theta_list = [float(Thetas)] * n
        else:
            theta_list = list(Thetas)
            if len(theta_list) != n:
                raise ValueError(
                    "Length of `Thetas` must match length of `X` when it is "
                    f"an iterable. Got {len(theta_list)} vs {n}."
                )

        # Normalize seeds to a list of length n.
        if seeds is None:
            seed_list = [None] * n
        elif isinstance(seeds, (int, float)):
            seed_list = [int(seeds)] * n
        else:
            seed_list = list(seeds)
            if len(seed_list) != n:
                raise ValueError(
                    "Length of `seeds` must match length of `X` when it is "
                    f"an iterable. Got {len(seed_list)} vs {n}."
                )

        y_target = []
        Y_peers = []

        for x, theta, seed in zip(inputs, theta_list, seed_list):
            # Apply the intervention once per (x, theta, seed).
            # If the concrete intervention does not accept `seed`, we fall
            # back to calling `apply(x, theta)` only.
            try:
                perturbed_x = intervention.apply(x, theta, seed)
            except TypeError:
                perturbed_x = intervention.apply(x, theta)

            # Evaluate target model.
            raw_t = self.target._forward(perturbed_x)
            if getattr(self.target, "scalarizer", None) is not None:
                y_t = self.target.scalarizer(raw_t)
            else:
                y_t = raw_t

            # Evaluate all peer models.
            peer_vals = []
            for peer in self.peers:
                raw_p = peer._forward(perturbed_x)
                if getattr(peer, "scalarizer", None) is not None:
                    val_p = peer.scalarizer(raw_p)
                else:
                    val_p = raw_p
                peer_vals.append(val_p)

            y_target.append(float(y_t))
            Y_peers.append([float(v) for v in peer_vals])

        y_target_arr = np.asarray(y_target, dtype=float)
        Y_peers_arr = np.asarray(Y_peers, dtype=float)

        return y_target_arr, Y_peers_arr
